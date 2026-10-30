// gpu8_kernels.inc - the OpenCL C of gpu8.cpp (the Adreno 830's int8 path; bench/gpu/i8 has how it was measured).
// Built with -D T (tokens a tile). Included as a string.
R"CL(
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

// Activations in tiles of T tokens: slot s = place s % T of tile s / T. Per tile, per 16-value block k, the T places'
// 16 bytes side by side: X uint4 (tile * nb + k) * T + t, in the order a q4 word's nibbles come out (values 0 2 .. 14,
// then 1 3 .. 15); XS the block's scale (amax / 127, as the CPU's quantize_activation), XB -15 x its int8 sum (the
// q4 level offset: the codes go into the dot products as 2c, the level is 2c - 15).

inline uint pack4(int a, int b, int c, int d) {
    return (uint)(a & 255) | ((uint)(b & 255) << 8) | ((uint)(c & 255) << 16) | ((uint)(d & 255) << 24);
}

inline void put_block(const float16 v, const int at, __global uint4 *X, __global float *XS, __global int *XB) {
    const float16 a = fabs(v);
    const float8 m8 = fmax(a.lo, a.hi);
    const float4 m4 = fmax(m8.lo, m8.hi);
    const float2 m2 = fmax(m4.lo, m4.hi);
    const float amax = fmax(m2.x, m2.y);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int16 q = convert_int16(clamp(rint(v * inv), -127.0f, 127.0f));
    X[at] = (uint4)(pack4(q.s0, q.s2, q.s4, q.s6), pack4(q.s8, q.sa, q.sc, q.se), pack4(q.s1, q.s3, q.s5, q.s7),
                    pack4(q.s9, q.sb, q.sd, q.sf));
    XS[at] = amax / 127.0f;
    const int8 s8 = q.lo + q.hi;
    const int4 s4 = s8.lo + s8.hi;
    const int2 s2 = s4.lo + s4.hi;
    XB[at] = -15 * (s2.x + s2.y);
}

#ifndef GEODE_PF_ONLY   // (the prompt mixers' program has only them and the helpers above)
// rows of `cols` floats into tiles: slot s takes row rowmap[s] of `in` (-1: a zero token). NDRange (slots, cols / 16).
__kernel void quant(__global const float *in, __global const int *rowmap, int cols, __global uint4 *X,
                    __global float *XS, __global int *XB) {
    const int s = get_global_id(0), k = get_global_id(1), nb = cols >> 4;
    const int tile = s / T, at = (tile * nb + k) * T + (s - tile * T);
    const int r = rowmap[s];
    const float16 v = r < 0 ? (float16)(0.0f) : vload16(0, in + (size_t)r * cols + 16 * k);
    put_block(v, at, X, XS, XB);
}

// the routed experts' middle: slot s's gate_up output [gate ff | up ff] -> silu(gate) * up -> tiles for down.
// NDRange (slots, ff / 16).
__kernel void swiglu(__global const float *gu, int ff, __global uint4 *X, __global float *XS, __global int *XB) {
    const int s = get_global_id(0), k = get_global_id(1), nb = ff >> 4;
    const int tile = s / T, at = (tile * nb + k) * T + (s - tile * T);
    __global const float *row = gu + (size_t)s * 2 * ff + 16 * k;
    const float16 g = vload16(0, row), u = vload16(0, row + ff);
    put_block(g / (1.0f + exp(-g)) * u, at, X, XS, XB);
}

// Y = W X for a list of jobs: job j = int8 (the matrix's offset in its buffer in uint4, its token tile, an offset into
// Y in floats, global scale x 256 as bits, which of the buffers W0..W15, its rows, -, -); a q4 matrix in tile8 (kernels.hpp: tiles of 32 rows, code word j of chunk c of
// quad i at uint4 (c * 8 + j) * 8 + i, then chunk c's scales of quad i at 64 nch + c * 8 + i). Token t of the tile
// goes to row ymap[tile * T + t] of Y (-1: nowhere), rows ystride floats apart, from the job's offset on. An item: quad q x the T tokens; NDRange (jobs, the tallest's rows / 4), work groups of
// one job's quads (the activations one address for the wave). A block a loop step; mad, never fma (emulated here).
inline uint4 lv_lo(uint4 w) { return (w << 1) & 0x1E1E1E1Eu; }
inline uint4 lv_hi(uint4 w) { return (w >> 3) & 0x1E1E1E1Eu; }
inline float e4m3(uint b) { return (float)as_half((ushort)((b & 0xFFu) << 7)); }

#define DOT(a, x, acc) qcom_dot8_acc(x, a, acc)
#define ROW(s) DOT(o1.s, x.s3, DOT(o0.s, x.s2, DOT(e1.s, x.s1, DOT(e0.s, x.s0, bias))))
#define STEP(t)                                                                                       \
    {                                                                                                 \
        const uint4 x = Xb[t];                                                                        \
        const int bias = XBb[t];                                                                      \
        const int4 d = (int4)(ROW(s0), ROW(s1), ROW(s2), ROW(s3));                                     \
        acc##t = mad(convert_float4(d), ws * XSb[t], acc##t);                                        \
    }
#define STORE(t)                                                                                      \
    {                                                                                                 \
        const int yr = ymap[tile * T + t];                                                            \
        if (yr >= 0) vstore4(acc##t * gscale, 0, Y + (size_t)yr * ystride + job.s2 + 4 * q);          \
    }
#if T == 1
#define EACH(M) M(0)
#elif T == 2
#define EACH(M) M(0) M(1)
#elif T == 3
#define EACH(M) M(0) M(1) M(2)
#elif T == 4
#define EACH(M) M(0) M(1) M(2) M(3)
#elif T == 5
#define EACH(M) M(0) M(1) M(2) M(3) M(4)
#elif T == 8
#define EACH(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#endif
#define DECL(t) float4 acc##t = 0.0f;

// the product itself, from the job's matrix at W (a work item's quad q x the T tokens)
#define GEMM_BODY                                                                                     \
    const float gscale = as_float(job.s3);                                                            \
    const int nch = cols >> 6, nb = cols >> 4;                                                        \
    __global const uint4 *Wc = W + job.s0 + (q >> 3) * (72 * nch) + (q & 7); /* a tile: 32 x 36 nch bytes */ \
    __global const uint *Sc = (__global const uint *)(Wc + 64 * nch);                                 \
    __global const uint4 *Xb = X + tile * nb * T;                                                     \
    __global const int *XBb = XB + tile * nb * T;                                                     \
    __global const float *XSb = XS + tile * nb * T;                                                   \
    EACH(DECL)                                                                                        \
    for (int k = 0; k < nb; ++k) {                                                                    \
        const uint4 w0 = Wc[0], w1 = Wc[8];                                                           \
        const uint sb = Sc[(k >> 2) * 32 + (k & 3)];                                                  \
        Wc += 16;                                                                                     \
        const float4 ws = (float4)(e4m3(sb), e4m3(sb >> 8), e4m3(sb >> 16), e4m3(sb >> 24));          \
        const uint4 e0 = lv_lo(w0), o0 = lv_hi(w0), e1 = lv_lo(w1), o1 = lv_hi(w1);                   \
        EACH(STEP)                                                                                    \
        Xb += T;                                                                                      \
        XBb += T;                                                                                     \
        XSb += T;                                                                                     \
    }                                                                                                 \
    EACH(STORE)

// every job's matrix in W (its loads: one argument, the texture path); jobs from job0 on
__kernel void gemm1(__global const uint4 *W, __global const int8 *jobs, __global const uint4 *X,
                    __global const float *XS, __global const int *XB, __global const int *ymap, __global float *Y,
                    int cols, int ystride, int job0) {
    const int8 job = jobs[job0 + get_global_id(0)];
    const int q = get_global_id(1), tile = job.s1;
    if (4 * q >= job.s5) return;   // past this job's rows (a launch takes matrices of different heights)
    GEMM_BODY
}

// the jobs' matrices in up to 16 buffers (job.s4 says which)
#define WARGS(i) __global const uint4 *W##i,
__kernel void gemm(WARGS(0) WARGS(1) WARGS(2) WARGS(3) WARGS(4) WARGS(5) WARGS(6) WARGS(7) WARGS(8) WARGS(9)
                   WARGS(10) WARGS(11) WARGS(12) WARGS(13) WARGS(14) WARGS(15)
                   __global const int8 *jobs, __global const uint4 *X, __global const float *XS,
                   __global const int *XB, __global const int *ymap, __global float *Y, int cols, int ystride,
                   int job0) {
    const int8 job = jobs[job0 + get_global_id(0)];
    const int q = get_global_id(1), tile = job.s1;
    if (4 * q >= job.s5) return;
    __global const uint4 *W;
    switch (job.s4) {
#define WCASE(i) case i: W = W##i; break;
        WCASE(0) WCASE(1) WCASE(2) WCASE(3) WCASE(4) WCASE(5) WCASE(6) WCASE(7) WCASE(8) WCASE(9) WCASE(10) WCASE(11)
        WCASE(12) WCASE(13) WCASE(14)
        default: W = W15;
    }
    GEMM_BODY
}

// ---- An image's products (vision.cpp; Gpu8::vision_gemm). The matrices q8 in "q8gpu": quad q's four rows' codes
// of block k as unsigned bytes (+128) at uint4 (k * quads + q) * 4 + r, then the blocks' E4M3 scales, uint
// k * quads + q (a byte a row). The activations in tiles as above but in their natural order, XB -128 x the block's
// int8 sum (the +128 taken back out in the dot product's start).
__kernel void quant8(__global const float *in, int inoff, int xstride, int n, int cols, __global uint4 *X,
                     __global float *XS, __global int *XB) {
    const int s = get_global_id(0), k = get_global_id(1), nb = cols >> 4;
    const int tile = s / T, at = (tile * nb + k) * T + (s - tile * T);
    const float16 v = s < n ? vload16(0, in + inoff + (size_t)s * xstride + 16 * k) : (float16)(0.0f);
    const float16 a = fabs(v);
    const float8 m8 = fmax(a.lo, a.hi);
    const float4 m4 = fmax(m8.lo, m8.hi);
    const float2 m2 = fmax(m4.lo, m4.hi);
    const float amax = fmax(m2.x, m2.y);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int16 q = convert_int16(clamp(rint(v * inv), -127.0f, 127.0f));
    X[at] = (uint4)(pack4(q.s0, q.s1, q.s2, q.s3), pack4(q.s4, q.s5, q.s6, q.s7), pack4(q.s8, q.s9, q.sa, q.sb),
                    pack4(q.sc, q.sd, q.se, q.sf));
    XS[at] = amax / 127.0f;
    const int8 s8 = q.lo + q.hi;
    const int4 s4 = s8.lo + s8.hi;
    const int2 s2 = s4.lo + s4.hi;
    XB[at] = -128 * (s2.x + s2.y);
}

// Y[token][4q ..] (ystride floats apart, from yoff on) for tokens < n and quads q < qn (the first rows: the CPU may
// take the rest): an item = quad q x a tile's T tokens; NDRange (tiles, qn rounded up to 64), work groups (1, 64)
#define ROW8(w) DOT(w.s3, x.s3, DOT(w.s2, x.s2, DOT(w.s1, x.s1, DOT(w.s0, x.s0, bias))))
#define STEP8(t)                                                                                      \
    {                                                                                                 \
        const uint4 x = Xb[t];                                                                        \
        const int bias = XBb[t];                                                                      \
        const int4 d = (int4)(ROW8(w0), ROW8(w1), ROW8(w2), ROW8(w3));                                 \
        acc##t = mad(convert_float4(d), ws * XSb[t], acc##t);                                        \
    }
#define STORE8(t)                                                                                     \
    {                                                                                                 \
        const int tok = tile * T + t;                                                                 \
        if (tok < n) vstore4(acc##t * gscale, 0, Y + yoff + (size_t)tok * ystride + 4 * q);           \
    }
__kernel void gemm8(__global const uint4 *W, int woff, int quads, int qn, int cols, float gscale,
                    __global const uint4 *X, __global const float *XS, __global const int *XB, __global float *Y,
                    int yoff, int ystride, int n) {
    const int tile = get_global_id(0), q = get_global_id(1), nb = cols >> 4;
    if (q >= qn) return;
    __global const uint4 *Wq = W + woff + 4 * q;
    __global const uint *Sq = (__global const uint *)(W + woff + (size_t)nb * quads * 4) + q;
    __global const uint4 *Xb = X + tile * nb * T;
    __global const int *XBb = XB + tile * nb * T;
    __global const float *XSb = XS + tile * nb * T;
    EACH(DECL)
    for (int k = 0; k < nb; ++k) {
        const uint4 w0 = Wq[0], w1 = Wq[1], w2 = Wq[2], w3 = Wq[3];
        const uint sb = Sq[0];
        Wq += 4 * quads;
        Sq += quads;
        const float4 ws = (float4)(e4m3(sb), e4m3(sb >> 8), e4m3(sb >> 16), e4m3(sb >> 24));
        EACH(STEP8)
        Xb += T;
        XBb += T;
        XSb += T;
    }
    EACH(STORE8)
}

#if T <= 5   // the generating kernels (T > 1: a verified batch; the prompt's T = 8 program has no use for them)
// One token, the columns split (generating): a work group = 64 quads x SPLIT waves, wave s taking blocks
// s * nb / SPLIT .. of every quad (its activation loads one address, its weight loads 64 quads side by side), the
// waves' sums added in a fixed order through local memory. SPLIT times the waves of gemm1 with T = 1: a 9 MB matrix
// is 32 work groups there, too few to keep the DRAM busy. NDRange (jobs, quads rounded up to 64 x SPLIT), work groups
// (1, 64 x SPLIT).
#define SPLIT 4
#define GSTEP(t)                                                                                      \
    {                                                                                                 \
        const int xi = (tile * nb + k) * T + t;                                                       \
        const uint4 x = X[xi];                                                                        \
        const int bias = XB[xi];                                                                      \
        const int4 d = (int4)(ROW(s0), ROW(s1), ROW(s2), ROW(s3));                                     \
        acc##t = mad(convert_float4(d), ws * XS[xi], acc##t);                                        \
    }
#define GRED(t) red[(t * SPLIT + slice) * 64 + lq] = acc##t;
#define GSUM(t)                                                                                       \
    {                                                                                                 \
        float4 s = red[(t * SPLIT) * 64 + lq];                                                        \
        for (int j = 1; j < SPLIT; ++j) s += red[(t * SPLIT + j) * 64 + lq];                          \
        const int yr = ymap[tile * T + t];                                                            \
        if (yr >= 0) vstore4(s * as_float(job.s3), 0, Y + (size_t)yr * ystride + job.s2 + 4 * q);     \
    }
#define GEMVS_BODY                                                                                    \
    const int l = get_local_id(1), slice = l >> 6, lq = l & 63, q = get_group_id(1) * 64 + lq;        \
    const int active = 4 * q < job.s5;                                                                \
    const int nch = cols >> 6, nb = cols >> 4, per = nb / SPLIT, k0 = slice * per, tile = job.s1;     \
    __global const uint4 *Wq = W + job.s0 + (q >> 3) * (72 * nch) + (q & 7);                          \
    __global const uint4 *Wc = Wq + 16 * k0;                                                          \
    __global const uint *Sc = (__global const uint *)(Wq + 64 * nch);                                 \
    EACH(DECL)                                                                                        \
    if (active)                                                                                       \
        for (int k = k0; k < k0 + per; ++k) {                                                         \
            const uint4 w0 = Wc[0], w1 = Wc[8];                                                       \
            const uint sb = Sc[(k >> 2) * 32 + (k & 3)];                                              \
            Wc += 16;                                                                                 \
            const float4 ws = (float4)(e4m3(sb), e4m3(sb >> 8), e4m3(sb >> 16), e4m3(sb >> 24));      \
            const uint4 e0 = lv_lo(w0), o0 = lv_hi(w0), e1 = lv_lo(w1), o1 = lv_hi(w1);               \
            EACH(GSTEP)                                                                               \
        }                                                                                             \
    EACH(GRED)                                                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                     \
    if (slice == 0 && active) {                                                                       \
        EACH(GSUM)                                                                                    \
    }

__kernel void gemvs1(__global const uint4 *W, __global const int8 *jobs, __global const uint4 *X,
                     __global const float *XS, __global const int *XB, __global const int *ymap, __global float *Y,
                     int cols, int ystride, int job0) {
    __local float4 red[64 * SPLIT * T];
    const int8 job = jobs[job0 + get_global_id(0)];
    GEMVS_BODY
}

__kernel void gemvs(WARGS(0) WARGS(1) WARGS(2) WARGS(3) WARGS(4) WARGS(5) WARGS(6) WARGS(7) WARGS(8) WARGS(9)
                    WARGS(10) WARGS(11) WARGS(12) WARGS(13) WARGS(14) WARGS(15)
                    __global const int8 *jobs, __global const uint4 *X, __global const float *XS,
                    __global const int *XB, __global const int *ymap, __global float *Y, int cols, int ystride,
                    int job0) {
    __local float4 red[64 * SPLIT * T];
    const int8 job = jobs[job0 + get_global_id(0)];
    __global const uint4 *W;
    switch (job.s4) {
        WCASE(0) WCASE(1) WCASE(2) WCASE(3) WCASE(4) WCASE(5) WCASE(6) WCASE(7) WCASE(8) WCASE(9) WCASE(10) WCASE(11)
        WCASE(12) WCASE(13) WCASE(14)
        default: W = W15;
    }
    GEMVS_BODY
}

// ---- generating: one token (the program built with T = 1), or T tokens at once (T = 2: a draft verified) ----
// Token t of a launch is get_group_id(1) (0 in a 1-D launch); a vector's block k of token t is X[k * T + t]; each
// token's arithmetic is the one-token kernel's, in the same order (a verified token's bits are a generated one's).
// The CPU's arithmetic (model.cpp) in the same order where it matters; sums run in another order (float rounding).
// Weights and norms live in one parameter buffer P (floats, bf16 read through the same buffer as ushort), at offsets
// in its elements.

inline float silu1(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid1(float x) { return 1.0f / (1.0f + exp(-x)); }

// a work group's sum / max (its size a power of two, <= 256)
// Local memory on this Adreno: a work group's waves can read a reduction's result (wg_sum / wg_max: tmp[0] after its
// last barrier) after other waves have already written the buffer's next use - one buffer reused straight after a
// reduction gave NaN in waves 2-3 of 256-item groups (a prompt's head norms). So a reduction's buffer is never written
// again in the same kernel; the next phase or reduction takes a buffer of its own.
inline float wg_sum(float v, __local float *tmp) {
    const int l = get_local_id(0);
    tmp[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s) tmp[l] += tmp[l + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = tmp[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
inline float wg_max(float v, __local float *tmp) {
    const int l = get_local_id(0);
    tmp[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s) tmp[l] = fmax(tmp[l], tmp[l + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = tmp[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

// x (+= add) -> h = x / rms(x) * (1 + w), and h into X / XS / XB. One work group of 256; n % 4096 == 0.
__kernel void norm_q(__global float *x, __global const float *add, int has_add, __global const float *P, int w,
                     __global float *h, int n, float eps, __global uint4 *X, __global float *XS, __global int *XB) {
    __local float tmp[256];
    const int l = get_local_id(0), per = n / 256, t = get_group_id(1);
    x += t * n;
    add += t * n;
    h += t * n;
    float ss = 0.0f;
    for (int i = l * per; i < (l + 1) * per; ++i) {
        float v = x[i];
        if (has_add) {
            v += add[i];
            x[i] = v;
        }
        ss += v * v;
    }
    ss = wg_sum(ss, tmp);
    const float inv = 1.0f / sqrt(ss / (float)n + eps);
    for (int i = l * per; i < (l + 1) * per; ++i) h[i] = x[i] * inv * (1.0f + P[w + i]);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int b = l; b < n / 16; b += 256) put_block(vload16(b, h), b * T + t, X, XS, XB);
}

// The MTP layer's input, quantized for its fc: [e / rms(e) * (1 + we) | h / rms(h) * (1 + wh)] (2 n values, e = the
// next token's embedding in x, h = the main model's hidden state after the final norm). One work group of 256 a token.
__kernel void mtp_in_q(__global const float *e, __global const float *h, __global const float *P, int we, int wh, int n,
                       float eps, __global uint4 *X, __global float *XS, __global int *XB) {
    __local float tmp[256], tmp2[256];
    const int l = get_local_id(0), per = n / 256, t = get_group_id(1);
    e += t * n;
    h += t * n;
    float se = 0.0f, sh = 0.0f;
    for (int i = l * per; i < (l + 1) * per; ++i) {
        se += e[i] * e[i];
        sh += h[i] * h[i];
    }
    se = wg_sum(se, tmp);
    sh = wg_sum(sh, tmp2);
    const float ie = 1.0f / sqrt(se / (float)n + eps), ih = 1.0f / sqrt(sh / (float)n + eps);
    const int nb = n / 16;
    for (int b = l; b < nb; b += 256) {
        put_block(vload16(b, e) * ie * (1.0f + vload16(b, P + we)), b * T + t, X, XS, XB);
        put_block(vload16(b, h) * ih * (1.0f + vload16(b, P + wh)), (nb + b) * T + t, X, XS, XB);
    }
}

// out[o + t os + r] = W[r] . h[t], W bf16 rows of n (at element w of P), a work group of 64 a row, its T tokens at once
// (the row read once)
__kernel void bf16_rows(__global const ushort *P, int w, __global const float *h, int n, __global float *out, int o,
                        int os) {
    __local float tmp[64];
    const int r = get_group_id(0), l = get_local_id(0);
    __global const ushort *row = P + w + (size_t)r * n;
    float s[T];
    for (int t = 0; t < T; ++t) s[t] = 0.0f;
    for (int i = l; i < n; i += 64) {
        const float wv = as_float((uint)row[i] << 16);
        for (int t = 0; t < T; ++t) s[t] += wv * h[t * n + i];
    }
    for (int t = 0; t < T; ++t) {
        const float v = wg_sum(s[t], tmp);
        if (l == 0) out[o + t * os + r] = v;
    }
}

// Gated DeltaNet. The causal depthwise convolution in place: channel ch against the K - 1 inputs before it (state,
// oldest first), then silu; T tokens in order (rows of ps floats), the state after each but the last also into bk at
// bo + t bst (T > 1: a draft that falls takes the state back to the last token that stood).
__kernel void gdn_conv(__global float *proj, __global const float *P, int w, __global float *st, int so, int K, int ps,
                       __global float *bk, int bo, int bst) {
    const int ch = get_global_id(0);
    __global const float *wc = P + w + ch * K;
    __global float *s = st + so + ch * (K - 1);
    for (int t = 0; t < T; ++t) {
        const float in = proj[t * ps + ch];
        float acc = 0.0f;
        for (int k = 0; k < K - 1; ++k) acc += wc[k] * s[k];
        acc += wc[K - 1] * in;
        for (int k = 0; k + 1 < K - 1; ++k) s[k] = s[k + 1];
        if (K > 1) s[K - 2] = in;
        proj[t * ps + ch] = silu1(acc);
        if (T > 1 && t < T - 1)
            for (int k = 0; k < K - 1; ++k) bk[bo + t * bst + ch * (K - 1) + k] = s[k];
    }
}

// q and k L2-normalised per key head, q also / sqrt(dk): work groups of dk, 0..nk-1 the q heads, nk..2nk-1 the k ones
__kernel void gdn_qk(__global float *proj, int dk, int kd, float qscale, int ps) {
    __local float tmp[256];
    const int g = get_group_id(0), j = get_local_id(0), nk = get_num_groups(0) / 2, t = get_group_id(1);
    const int isk = g >= nk;
    __global float *p = proj + t * ps + (isk ? kd + (g - nk) * dk : g * dk);
    const float v = p[j];
    const float inv = 1.0f / sqrt(wg_sum(v * v, tmp) + 1e-6f);
    p[j] = isk ? v * inv : v * inv * qscale;
}

// each value head's decay exp(-exp(A_log) softplus(a + dt_bias)) and step sigmoid(b), in place (ab = [a nv | b nv])
__kernel void gdn_gates(__global float *ab, int nv, __global const float *P, int alog, int dtb) {
    const int vh = get_global_id(0);
    ab += get_global_id(1) * 2 * nv;
    const float x = ab[vh] + P[dtb + vh];
    const float sp = x > 20.0f ? x : log1p(exp(x));
    ab[vh] = exp(-exp(P[alog + vh]) * sp);
    ab[nv + vh] = sigmoid1(ab[nv + vh]);
}

// the gated delta rule, column j of value head vh (a work group a head: dv x 4 items, item j + dv * part takes rows
// part * dk / 4 .. of the column): S = S decay; delta = (v - S^T k) beta; S += k delta^T; o = S^T q, the four parts'
// sums added in a fixed order. The state as the CPU keeps it, in strips of 16 columns: [dv / 16][dk][16] a head.
__kernel void gdn_delta(__global float *state, int so, __global const float *proj, __global const float *ab,
                        __global float *out, int nk, int nv, int dk, int dv) {
    __local float red[1024];
    const int vh = get_group_id(0), l = get_local_id(0), j = l % dv, part = l / dv;
    const int kh = vh / (nv / nk), kd = nk * dk, i0 = part * (dk / 4), i1 = i0 + dk / 4;
    __global const float *q = proj + kh * dk, *k = proj + kd + kh * dk;
    const float v = proj[2 * kd + vh * dv + j];
    const float decay = ab[vh], beta = ab[nv + vh];
    __global float *S = state + so + (size_t)vh * dk * dv + (j / 16) * dk * 16 + (j % 16);
    float mem = 0.0f;
    for (int i = i0; i < i1; ++i) {
        const float s = S[i * 16] * decay;
        S[i * 16] = s;
        mem += s * k[i];
    }
    red[l] = mem;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float delta = (v - (((red[j] + red[dv + j]) + red[2 * dv + j]) + red[3 * dv + j])) * beta;
    barrier(CLK_LOCAL_MEM_FENCE);
    float o = 0.0f;
    for (int i = i0; i < i1; ++i) {
        const float s = S[i * 16] + k[i] * delta;
        S[i * 16] = s;
        o += s * q[i];
    }
    red[l] = o;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (part == 0) out[vh * dv + j] = ((red[j] + red[dv + j]) + red[2 * dv + j]) + red[3 * dv + j];
}

// The same for dk = 128 (Qwen3.6): an item's 32 rows of the column stay in registers between the two passes, so the
// state is read once and written once (gdn_delta reads and writes it twice: 8 MB a layer instead of 4). The same
// arithmetic in the same order: the same bits.
__kernel void gdn_delta32(__global float *state, int so, __global const float *proj, __global const float *ab,
                          __global float *out, int nk, int nv, int dk, int dv, int ps, __global float *bk, int bo,
                          int bst) {
    // a work group: 64 columns of a head x 4 parts (32 registers an item cap the group at 256); T tokens in order,
    // the state between them in registers (T > 1: the state after each token but the last also into bk at
    // bo + t bst). Two
    // reduction buffers (the S^T k sums, the outputs): two barriers a token
    __local float rm[256], ro[256];
    const int g = get_group_id(0), l = get_local_id(0), part = l / 64;
    const int vh = g / (dv / 64), j = (g % (dv / 64)) * 64 + l % 64, c = l % 64;
    const int kh = vh / (nv / nk), kd = nk * dk, i0 = part * 32;
    const size_t at = (size_t)vh * dk * dv + (j / 16) * dk * 16 + (j % 16) + i0 * 16;
    __global float *S = state + so + at;
    float sv[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) sv[i] = S[i * 16];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        __global const float *q = proj + t * ps + kh * dk + i0, *k = proj + t * ps + kd + kh * dk + i0;
        const float v = proj[t * ps + 2 * kd + vh * dv + j];
        const float decay = ab[t * 2 * nv + vh], beta = ab[t * 2 * nv + nv + vh];
        float mem = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            sv[i] = sv[i] * decay;
            mem += sv[i] * k[i];
        }
        rm[l] = mem;
        barrier(CLK_LOCAL_MEM_FENCE);
        const float delta = (v - (((rm[c] + rm[64 + c]) + rm[128 + c]) + rm[192 + c])) * beta;
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            sv[i] = sv[i] + k[i] * delta;
            o += sv[i] * q[i];
        }
        ro[l] = o;
        barrier(CLK_LOCAL_MEM_FENCE);   // (also: every rm read of this token done before the next one writes it)
        if (part == 0) out[t * nv * dv + vh * dv + j] = ((ro[c] + ro[64 + c]) + ro[128 + c]) + ro[192 + c];
        if (T > 1 && t < T - 1) {
            __global float *B = bk + bo + (size_t)t * bst + at;
#pragma unroll
            for (int i = 0; i < 32; ++i) B[i * 16] = sv[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) S[i * 16] = sv[i];
}

// gated RMSNorm per value head in place, o / rms(o) * w * silu(z), then the head's blocks into X (a work group a head)
__kernel void gdn_gnorm_q(__global float *core, __global const float *z, __global const float *P, int w, float eps,
                          __global uint4 *X, __global float *XS, __global int *XB) {
    __local float tmp[256];
    const int vh = get_group_id(0), j = get_local_id(0), dv = get_local_size(0), t = get_group_id(1);
    const int vd = get_num_groups(0) * dv;
    __global float *o = core + t * vd + vh * dv;
    const float v = o[j];
    const float inv = 1.0f / sqrt(wg_sum(v * v, tmp) / (float)dv + eps);
    o[j] = v * inv * P[w + j] * silu1(z[t * vd + vh * dv + j]);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (j < dv / 16) put_block(vload16(j, o), (vh * (dv / 16) + j) * T + t, X, XS, XB);
}

// Attention. A head's RMSNorm (1 + w) and partial RoPE (pairs i, i + hr of the first 2 hr dims): head g at
// buf + g * stride, a work group of hd; token t's heads ts floats on, at position pos + t.
__kernel void head_norm_rope(__global float *buf, int stride, __global const float *P, int w, float eps, int hr,
                             int pos, __global const float *inv_freq, int ts) {
    __local float tmp[256], rot[256];
    const int g = get_group_id(0), i = get_local_id(0), hd = get_local_size(0), t = get_group_id(1);
    __global float *x = buf + t * ts + g * stride;
    pos += t;
    const float v = x[i];
    const float inv = 1.0f / sqrt(wg_sum(v * v, tmp) / (float)hd + eps);
    const float y = v * inv * (1.0f + P[w + i]);
    rot[i] = y;
    barrier(CLK_LOCAL_MEM_FENCE);
    float r = y;
    if (i < 2 * hr) {   // (hr: half the rotated dims; "half" is a type here)
        const float f = (float)pos * inv_freq[i % hr];
        const float c = cos(f), s = sin(f);
        r = i < hr ? rot[i] * c - rot[i + hr] * s : rot[i] * c + rot[i - hr] * s;
    }
    x[i] = r;
}

// a position's K and V (from element src of k and v) into the cache (FP16, round to nearest even)
__kernel void kv_store(__global const float *k, __global const float *v, __global half *kv, int ko, int vo, int src) {
    const int i = get_global_id(0);
    vstore_half(k[src + i], ko + i, kv);
    vstore_half(v[src + i], vo + i, kv);
}

// scores of the query heads (from element qo of qg) against positions [p0, p0 + range) of a chunk whose first
// position is c0: item (p, h)
__kernel void attn_scores(__global const float *qg, __global const half *kv, int kb, int kv_row, int hd, int group,
                          int p0, int c0, __global float *scores, int np, float scale, int qo) {
    const int p = p0 + get_global_id(0), h = get_global_id(1);
    __global const float *q = qg + qo + h * 2 * hd;
    __global const half *k = kv + kb + (size_t)(p - c0) * kv_row + (h / group) * hd;
    float s = 0.0f;
    for (int i = 0; i < hd; i += 4) s += dot(vload4(0, q + i), vload_half4(0, k + i));
    scores[h * np + p] = s * scale;
}

// softmax over np scores of a head (a work group of 256 a head), in place
__kernel void attn_softmax(__global float *scores, int np) {
    __local float tmp[256], tmp2[256];
    const int h = get_group_id(0), l = get_local_id(0);
    __global float *s = scores + h * np;
    float m = -INFINITY;
    for (int p = l; p < np; p += 256) m = fmax(m, s[p]);
    m = wg_max(m, tmp);
    float sum = 0.0f;
    for (int p = l; p < np; p += 256) {
        const float e = exp(s[p] - m);
        s[p] = e;
        sum += e;
    }
    const float inv = 1.0f / wg_sum(sum, tmp2);
    for (int p = l; p < np; p += 256) s[p] *= inv;
}

// out[oo + h hd + d] (+)= the weighted values of positions [p0, p1) of a chunk (first position c0): item (d, h)
__kernel void attn_values(__global const float *scores, int np, __global const half *kv, int vb, int kv_row, int hd,
                          int group, int p0, int p1, int c0, __global float *out, int first, int oo) {
    const int d = get_global_id(0), h = get_global_id(1);
    __global const float *w = scores + h * np;
    __global const half *v = kv + vb + (h / group) * hd + d;
    out += oo;
    float acc = first ? 0.0f : out[h * hd + d];
    for (int p = p0; p < p1; ++p) acc += w[p] * vload_half(0, v + (size_t)(p - c0) * kv_row);
    out[h * hd + d] = acc;
}

// Flash decoding (head_dim 256, 8 query heads a KV head - Qwen3.6): the positions in blocks of AP. A work group takes
// one block of one KV head for the program's T tokens at once: their 8 queries each in local memory, each K row read
// for all 8 heads (item = position x token), the block's softmax max and sum per token and head (tree reductions),
// then each V row read once for all T x 8 outputs (item = dimension); attn_merge weighs the blocks by exp(max - the
// max). The same blocks for one token and two: each token's arithmetic the same (a verified pair's bits are a
// generated token's). The old kernels read the query again for every position and walked every position in one item
// (31 ms a token at 3000 positions); the loads here go eight (V) and four (K) at a time, in flight together. Token t
// sees positions < np0 + t. Tokens go two a launch (PT; local memory is 32 KB): tokens tb, tb + 1 of the T - a batch
// of more takes several launches, each token's arithmetic the one-token arithmetic.
#define AG 8
#define AB 256   // the head's dimensions = the work group
#define AP 128   // positions a block
#define PT (T < 2 ? T : 2)
__kernel void attn_part(__global const float *qg, int qs, __global const half *kv, int kb, int vb, int kv_row, int c0,
                        int b0, int np0, float scale, __global float *po, __global float *pm, __global float *pl,
                        int nb, int nh, int tb) {
    __local float qa[PT * AG * AB];   // the queries; then the reductions
    __local float pw[PT * AG * AP];   // the block's weights exp(s - max), per token and head
    const int i = get_local_id(0), b = b0 + get_group_id(0), g = get_group_id(1), nt = min(PT, T - tb);
    for (int t = 0; t < nt; ++t)
#pragma unroll
        for (int j = 0; j < AG; ++j) qa[(t * AG + j) * AB + i] = qg[(tb + t) * qs + (g * AG + j) * 2 * AB + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    // scores: item i = position i % AP of token tb + i / AP
    const int tk = i / AP, ip = i % AP, act = tk < nt;
    const int p = b * AP + ip, np = np0 + tb + tk, valid = act && p < np;
    float s[AG];
#pragma unroll
    for (int j = 0; j < AG; ++j) s[j] = 0.0f;
    if (valid) {
        __global const half *k = kv + kb + (size_t)(p - c0) * kv_row + g * AB;
        __local const float *q = qa + tk * AG * AB;
        for (int d = 0; d < AB; d += 16) {   // four loads in flight
            float4 kk[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) kk[u] = vload_half4(0, k + d + 4 * u);
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int j = 0; j < AG; ++j) s[j] += dot(vload4(0, q + j * AB + d + 4 * u), kk[u]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // the queries read: qa takes the reductions, [token][head][AP]
    __local float *red = qa + tk * AG * AP;
    if (act)
#pragma unroll
        for (int j = 0; j < AG; ++j) {
            s[j] = valid ? s[j] * scale : -INFINITY;
            red[j * AP + ip] = s[j];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = AP / 2; st > 0; st >>= 1) {
        if (act && ip < st)
#pragma unroll
            for (int j = 0; j < AG; ++j) red[j * AP + ip] = fmax(red[j * AP + ip], red[j * AP + ip + st]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m[AG];
#pragma unroll
    for (int j = 0; j < AG; ++j) m[j] = act ? red[j * AP] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (act)
#pragma unroll
        for (int j = 0; j < AG; ++j) {
            const float e = valid ? exp(s[j] - m[j]) : 0.0f;   // (a block no position of which the token sees: all 0)
            pw[(tk * AG + j) * AP + ip] = e;
            red[j * AP + ip] = e;
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = AP / 2; st > 0; st >>= 1) {
        if (act && ip < st)
#pragma unroll
            for (int j = 0; j < AG; ++j) red[j * AP + ip] += red[j * AP + ip + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // each token's partials of the block: max and sum (item 0 of the token's rows)
    if (act && ip == 0)
#pragma unroll
        for (int j = 0; j < AG; ++j) {
            const size_t at = ((size_t)(tb + tk) * nh + g * AG + j) * nb + b;
            pm[at] = m[j];
            pl[at] = red[j * AP];
        }
    // the values: item i = dimension i, every token's 8 outputs from one read of each V row
    float o[PT * AG];
#pragma unroll
    for (int j = 0; j < PT * AG; ++j) o[j] = 0.0f;
    const int pend = min(AP, np0 + tb + nt - 1 - b * AP);   // the rows any token sees (the others' weights are 0)
    __global const half *v = kv + vb + (size_t)(b * AP - c0) * kv_row + g * AB + i;
    int q = 0;
    for (; q + 8 <= pend; q += 8) {   // eight rows' loads in flight at once
        float vv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) vv[u] = vload_half(0, v + (size_t)(q + u) * kv_row);
#pragma unroll
        for (int u = 0; u < 8; ++u)
#pragma unroll
            for (int j = 0; j < PT * AG; ++j) o[j] += pw[j * AP + q + u] * vv[u];
    }
    for (; q < pend; ++q) {
        const float vv = vload_half(0, v + (size_t)q * kv_row);
#pragma unroll
        for (int j = 0; j < PT * AG; ++j) o[j] += pw[j * AP + q] * vv;
    }
#pragma unroll
    for (int j = 0; j < PT * AG; ++j) {
        const int t = j / AG;
        if (t < nt) po[(((size_t)(tb + t) * nh + g * AG + j % AG) * nb + b) * AB + i] = o[j];
    }
}

// The int8 KV cache (Model::kv_scale): a position's K and V rows of KV head h as int8, amax / 127 their scales
// (round to nearest even, as the CPU). A work group of head_dim a KV head.
__kernel void kv_store8(__global const float *k, __global const float *v, __global char *kv, int ko, int vo,
                        __global float *kvs, int ksi, int vsi, int src) {
    __local float tmp[256], tmp2[256];
    const int h = get_group_id(0), i = get_local_id(0), hd = get_local_size(0);
    const float x = k[src + h * hd + i], y = v[src + h * hd + i];
    const float ka = wg_max(fabs(x), tmp), va = wg_max(fabs(y), tmp2);
    kv[ko + h * hd + i] = (char)clamp(rint(x * (ka > 0.0f ? 127.0f / ka : 0.0f)), -127.0f, 127.0f);
    kv[vo + h * hd + i] = (char)clamp(rint(y * (va > 0.0f ? 127.0f / va : 0.0f)), -127.0f, 127.0f);
    if (i == 0) {
        kvs[ksi + h] = ka / 127.0f;
        kvs[vsi + h] = va / 127.0f;
    }
}

// attn_part over the int8 cache: a K row's dot products times its scale; the block's V scales in local memory, each
// V value times its row's scale. kb / vb: the layer's K / V bytes in the chunk, ksb / vsb: its scales (floats).
__kernel void attn_part8(__global const float *qg, int qs, __global const char *kv, int kb, int vb, int kv_row,
                         __global const float *kvs, int ksb, int vsb, int c0, int b0, int np0, float scale,
                         __global float *po, __global float *pm, __global float *pl, int nb, int nh, int tb) {
    __local float qa[PT * AG * AB];
    __local float pw[PT * AG * AP];
    __local float vsl[AP];
    const int i = get_local_id(0), b = b0 + get_group_id(0), g = get_group_id(1), nkv = nh / AG,
              nt = min(PT, T - tb);
    for (int t = 0; t < nt; ++t)
#pragma unroll
        for (int j = 0; j < AG; ++j) qa[(t * AG + j) * AB + i] = qg[(tb + t) * qs + (g * AG + j) * 2 * AB + i];
    const int last = np0 + tb + nt - 1;   // the positions any of this launch's tokens sees
    if (i < AP) vsl[i] = b * AP + i < last ? kvs[vsb + (b * AP + i - c0) * nkv + g] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int tk = i / AP, ip = i % AP, act = tk < nt;
    const int p = b * AP + ip, np = np0 + tb + tk, valid = act && p < np;
    float s[AG];
#pragma unroll
    for (int j = 0; j < AG; ++j) s[j] = 0.0f;
    if (valid) {
        __global const char *k = kv + kb + (size_t)(p - c0) * kv_row + g * AB;
        __local const float *q = qa + tk * AG * AB;
        for (int d = 0; d < AB; d += 16) {   // four loads in flight
            float4 kk[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) kk[u] = convert_float4(vload4(0, k + d + 4 * u));
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int j = 0; j < AG; ++j) s[j] += dot(vload4(0, q + j * AB + d + 4 * u), kk[u]);
        }
        const float ks = kvs[ksb + (p - c0) * nkv + g];
#pragma unroll
        for (int j = 0; j < AG; ++j) s[j] *= ks;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float *red = qa + tk * AG * AP;
    if (act)
#pragma unroll
        for (int j = 0; j < AG; ++j) {
            s[j] = valid ? s[j] * scale : -INFINITY;
            red[j * AP + ip] = s[j];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = AP / 2; st > 0; st >>= 1) {
        if (act && ip < st)
#pragma unroll
            for (int j = 0; j < AG; ++j) red[j * AP + ip] = fmax(red[j * AP + ip], red[j * AP + ip + st]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m[AG];
#pragma unroll
    for (int j = 0; j < AG; ++j) m[j] = act ? red[j * AP] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (act)
#pragma unroll
        for (int j = 0; j < AG; ++j) {
            const float e = valid ? exp(s[j] - m[j]) : 0.0f;
            pw[(tk * AG + j) * AP + ip] = e * vsl[ip];   // the row's V scale in its weight (the sum: e alone)
            red[j * AP + ip] = e;
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = AP / 2; st > 0; st >>= 1) {
        if (act && ip < st)
#pragma unroll
            for (int j = 0; j < AG; ++j) red[j * AP + ip] += red[j * AP + ip + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (act && ip == 0)
#pragma unroll
        for (int j = 0; j < AG; ++j) {
            const size_t at = ((size_t)(tb + tk) * nh + g * AG + j) * nb + b;
            pm[at] = m[j];
            pl[at] = red[j * AP];
        }
    float o[PT * AG];
#pragma unroll
    for (int j = 0; j < PT * AG; ++j) o[j] = 0.0f;
    const int pend = min(AP, last - b * AP);
    __global const char *v = kv + vb + (size_t)(b * AP - c0) * kv_row + g * AB + i;
    int q = 0;
    for (; q + 8 <= pend; q += 8) {
        float vv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) vv[u] = convert_float(v[(size_t)(q + u) * kv_row]);
#pragma unroll
        for (int u = 0; u < 8; ++u)
#pragma unroll
            for (int j = 0; j < PT * AG; ++j) o[j] += pw[j * AP + q + u] * vv[u];
    }
    for (; q < pend; ++q) {
        const float vv = convert_float(v[(size_t)q * kv_row]);
#pragma unroll
        for (int j = 0; j < PT * AG; ++j) o[j] += pw[j * AP + q] * vv;
    }
#pragma unroll
    for (int j = 0; j < PT * AG; ++j) {
        const int t = j / AG;
        if (t < nt) po[(((size_t)(tb + t) * nh + g * AG + j % AG) * nb + b) * AB + i] = o[j];
    }
}

// the blocks of head h, token t merged: out[t cs + h AB + d] (a work group of AB a head and token)
__kernel void attn_merge(__global const float *po, __global const float *pm, __global const float *pl, int nb,
                         __global float *out, int cs, int nh) {
    const int d = get_local_id(0), h = get_group_id(0), t = get_group_id(1);
    const size_t base = ((size_t)t * nh + h) * nb;
    float mx = -INFINITY;
    for (int b = 0; b < nb; ++b) mx = fmax(mx, pm[base + b]);
    float l = 0.0f, o = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float w = exp(pm[base + b] - mx);
        l += pl[base + b] * w;
        o += po[(base + b) * AB + d] * w;
    }
    out[t * cs + h * AB + d] = o / l;
}

// the attention output gated, o * sigmoid(gate), and into X: an item a 16-value block; token t's output cs floats
// on, its query/gate qs floats on
__kernel void attn_gate_q(__global float *core, __global const float *qg, int hd, __global uint4 *X,
                          __global float *XS, __global int *XB, int cs, int qs) {
    const int b = get_global_id(0), h = (b * 16) / hd, d0 = (b * 16) % hd, t = get_global_id(1);
    core += t * cs;
    const float16 g = vload16(0, qg + t * qs + h * 2 * hd + hd + d0);
    const float16 o = vload16(b, core) * (1.0f / (1.0f + exp(-g)));
    vstore16(o, b, core);
    put_block(o, b * T + t, X, XS, XB);
}

// MoE. Softmax over the router's E logits, the top K renormalised (route_one); the next layer's guess (the top G of
// its own logits); then the flag the CPU waits on. One work group of E items (E <= 256, a power of two).
inline int wg_argmax(float v, int i, __local float *tv, __local int *ti) {
    const int l = get_local_id(0);
    tv[l] = v;
    ti[l] = i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s && (tv[l + s] > tv[l] || (tv[l + s] == tv[l] && ti[l + s] < ti[l]))) {
            tv[l] = tv[l + s];
            ti[l] = ti[l + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int r = ti[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

// The host reads the picks and the guesses while the queue runs on: each word carries the route's number above the
// expert (tag << 8 | id), so the host knows each one is this route's. A flag word after them is not enough: the GPU's
// L2 reaches memory line by line in any order, fences or not (GEODE_GPU8_ROUTECHECK caught the host reading the
// previous layer's ids at the flag). The weights stay on the GPU (moe_combine reads them).
__kernel void route(__global const float *logits, __global const float *guess, int has_guess, int K, int G,
                    __global int *ids, __global float *weights, __global int *gids, int tag) {
    // the top K by rank in one pass (an item's rank: how many beat it, the lower expert winning a tie, as a repeated
    // argmax picks) instead of K argmax reductions: 2 reductions and 4 barriers, not ~50. Token t (a work group each):
    // its logits E + 1 floats on (the shared gate's last), its guess logits E on, its picks K and guesses G on.
    __local float tmp[256];
    __local float pv[256];
    __local float gv[256];
    __local float wl[16];
    const int l = get_local_id(0), E = get_local_size(0), t = get_group_id(1);
    logits += t * (E + 1);
    guess += t * E;
    ids += t * K;
    weights += t * K;
    gids += t * G;
    const float x = logits[l];
    const float m = wg_max(x, tmp);
    const float e = exp(x - m);
    const float p = e / wg_sum(e, tmp);
    pv[l] = p;
    if (has_guess) gv[l] = guess[l];
    barrier(CLK_LOCAL_MEM_FENCE);
    int rank = 0;
    for (int i = 0; i < E; ++i) {
        const float o = pv[i];
        rank += o > p || (o == p && i < l);
    }
    if (rank < K) {
        wl[rank] = p;
        ids[rank] = tag << 8 | l;
    }
    if (has_guess) {   // the guess: the G likeliest by the next layer's router logits
        const float g = gv[l];
        int gr = 0;
        for (int i = 0; i < E; ++i) {
            const float o = gv[i];
            gr += o > g || (o == g && i < l);
        }
        if (gr < G) gids[gr] = tag << 8 | l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rank < K) {
        float top = 0.0f;   // the picked probabilities in pick order, as route_one adds them
        for (int k = 0; k < K; ++k) top += wl[k];
        weights[rank] = p / top;
    }
}

// The guess two layers ahead: the top G of that layer's router logits (E a token), tagged like the route's words.
// One work group of E items a token.
__kernel void guess_top(__global const float *guess, int G, __global int *gids, int tag) {
    __local float gv[256];
    const int l = get_local_id(0), E = get_local_size(0), t = get_group_id(1);
    guess += t * E;
    gids += t * G;
    const float g = guess[l];
    gv[l] = g;
    barrier(CLK_LOCAL_MEM_FENCE);
    int gr = 0;
    for (int i = 0; i < E; ++i) {
        const float o = gv[i];
        gr += o > g || (o == g && i < l);
    }
    if (gr < G) gids[gr] = tag << 8 | l;
}

// skip: the picks left out (bit k; never computed, their rows stale); renorm: the kept weights scaled to sum 1.
// Token t (get_global_id(1)): its picks' outputs in rows u * T + t of eo, u = 6 bits k of um (word t), the shared
// expert's row sh * T + t; its weights K on, its router logits (the shared gate's at so) E + 1 on.
__kernel void moe_combine(__global float *x, __global const float *eo, __global const float *weights, uint8 skips,
                          int renorm, __global const float *sg, int so, int H, int K, ulong8 ums, int sh) {
    const int i = get_global_id(0), t = get_global_id(1);
    uint sk[8];
    ulong uk[8];
    vstore8(skips, 0, sk);
    vstore8(ums, 0, uk);
    const uint skip = sk[t];
    const ulong um = uk[t];
    x += t * H;
    weights += t * K;
    float kept = 0.0f;
    for (int k = 0; k < K; ++k)
        if (!(skip >> k & 1)) kept += weights[k];
    const float scale = renorm && skip ? 1.0f / kept : 1.0f;
    const float gs = 1.0f / (1.0f + exp(-sg[t * (so + 1) + so]));
    float m = gs * eo[(sh * T + t) * H + i];
    for (int k = 0; k < K; ++k)
        if (!(skip >> k & 1)) m += weights[k] * scale * eo[((int)(um >> (6 * k) & 63) * T + t) * H + i];
    x[i] += m;
}
#endif

#endif   // GEODE_PF_ONLY

#ifdef GEODE_PF   // the prompt's token mixers: a program of their own (Gpu8::Impl::pf_setup)
// ---- A prompt's token mixers (Gpu8::pf_gdn / pf_attn): n tokens a launch (a runtime count, not T), the tokens'
// rows `ps` (or the shape's) floats apart; what goes on into a product is put into the tiles of T tokens as quant does.
inline float pf_silu(float x) { return x / (1.0f + exp(-x)); }
inline float pf_sum(float v, __local float *tmp) {
    const int l = get_local_id(0);
    tmp[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s) tmp[l] += tmp[l + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = tmp[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
inline float pf_max(float v, __local float *tmp) {
    const int l = get_local_id(0);
    tmp[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s) tmp[l] = fmax(tmp[l], tmp[l + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = tmp[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
// slot t's block k of a product's input (rows of nb blocks) in the tiles of T
inline int pf_at(int t, int k, int nb) { return ((t / T) * nb + k) * T + t % T; }

// out[t * os + r] = W[r] . h[t] for t < n: a work group of 64 = row r x 8 tokens (dimension 1: tokens 8 g ..), the
// row read once for them; W bf16 rows of `cols`
__kernel void pf_bf16_rows(__global const ushort *P, int w, __global const float *h, int cols, __global float *out,
                           int os, int n) {
    __local float tmp[8][64];   // a buffer for each token's reduction
    const int r = get_group_id(0), l = get_local_id(0), t0 = get_group_id(1) * 8;
    __global const ushort *row = P + w + (size_t)r * cols;
    float s[8];
    for (int u = 0; u < 8; ++u) s[u] = 0.0f;
    for (int i = l; i < cols; i += 64) {
        const float wv = as_float((uint)row[i] << 16);
        for (int u = 0; u < 8; ++u)
            if (t0 + u < n) s[u] += wv * h[(size_t)(t0 + u) * cols + i];
    }
    for (int u = 0; u < 8; ++u) {
        const float v = pf_sum(s[u], tmp[u]);
        if (l == 0 && t0 + u < n) out[(size_t)(t0 + u) * os + r] = v;
    }
}

// out[t * os + r] = W[r] . h[t] for r < rows, t < n: a work group of 64 = rows 64 g .. x 8 tokens (dimension 1:
// tokens 8 g ..), an item a row; the group's rows staged 64 values at a time in local memory (read in whole 128-byte
// runs of a row: one row an item read straight from memory was 16 bytes 4 KB apart, ~60 G MAC/s), the tokens' loads
// one address for the wave. W bf16 rows of `cols` (a multiple of 64) at P + w. The router (and the shared expert's
// gate), Gated DeltaNet's a / b.
#define BF_ACC(u) float4 a##u = 0.0f;
#define BF_STEP(u)                                                                                    \
    {                                                                                                 \
        const float8 x = vload8(0, h + o##u + v);                                                     \
        a##u = mad(wlo, x.even, a##u);                                                                \
        a##u = mad(whi, x.odd, a##u);                                                                 \
    }
#define BF_OFF(u) const size_t o##u = (size_t)min(t0 + u, n - 1) * cols;
#define BF_PUT(u) \
    if (t0 + u < n) out[(size_t)(t0 + u) * os + r] = (a##u.x + a##u.y) + (a##u.z + a##u.w);
#define BF_EACH(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
__kernel void pf_bf16_gemm(__global const ushort *P, int w, __global const float *h, int cols, __global float *out,
                           int os, int rows, int n) {
    __local uint wl[64 * 36];   // 64 rows x 64 values (32 uints), rows 36 uints apart (16-byte reads, few conflicts)
    const int i = get_local_id(0), r0 = get_group_id(0) * 64, t0 = get_group_id(1) * 8, r = r0 + i;
    __global const uint *W = (__global const uint *)(P + w);
    const int rw = cols / 2;   // uints a row
    BF_EACH(BF_OFF)
    BF_EACH(BF_ACC)
    for (int k = 0; k < cols; k += 64) {
        for (int m = 0; m < 8; ++m) {   // uint4 idx of the 512: row idx / 8, its (idx % 8)th 16 bytes
            const int idx = i + 64 * m, rr = idx >> 3, c = (idx & 7) * 4;
            vstore4(vload4(0, W + (size_t)min(r0 + rr, rows - 1) * rw + k / 2 + c), 0, wl + rr * 36 + c);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int c = 0; c < 32; c += 4) {
            const uint4 wb = vload4(0, wl + i * 36 + c);
            const float4 wlo = as_float4(wb << 16), whi = as_float4(wb & 0xFFFF0000u);   // values v, v + 2 .. / v + 1 ..
            const int v = k + 2 * c;
            BF_EACH(BF_STEP)
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (r >= rows) return;
    BF_EACH(BF_PUT)
}

// The same product tiled: a work group of 64 for 32 tokens x 64 rows, an item 4 tokens x 8 rows (32 sums); 32 values a
// step staged in local memory as floats, [value][token] and [value][row], so an item's 4 tokens are one float4 and
// its 8 rows two at each value (3 loads for 32 multiply-adds; pf_bf16_gemm made 9 for 64, 8 of them the same h
// address for every item). The sums run in another order than pf_bf16_gemm's.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void pf_bf16_gemm2(__global const ushort *P, int w, __global const float *h, int cols, __global float *out, int os,
                   int rows, int n) {
    __local float hs[32 * 32];   // [value][token]
    __local float wsl[32 * 64];  // [value][row]
    const int i = get_local_id(0), r0 = get_group_id(0) * 64, t0 = get_group_id(1) * 32;
    const int it = (i % 8) * 4, ir = (i / 8) * 8;   // the item's tokens it .. it + 3, rows ir .. ir + 7 of the tile
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    __global const ushort *W = P + w;
    for (int k0 = 0; k0 < cols; k0 += 32) {
        for (int m = 0; m < 4; ++m) {   // h: 32 tokens x 8 float4, item -> (token idx % 32, quarter idx / 32)
            const int idx = i + 64 * m, t = idx % 32, kq = idx / 32;
            const float4 v = vload4(0, h + (size_t)min(t0 + t, n - 1) * cols + k0 + 4 * kq);
            hs[(4 * kq + 0) * 32 + t] = v.x;
            hs[(4 * kq + 1) * 32 + t] = v.y;
            hs[(4 * kq + 2) * 32 + t] = v.z;
            hs[(4 * kq + 3) * 32 + t] = v.w;
        }
        for (int m = 0; m < 4; ++m) {   // W: 64 rows x 4 uint4 (8 bf16 each), item -> (row idx % 64, eighth idx / 64)
            const int idx = i + 64 * m, r = idx % 64, kq = idx / 64;
            const uint4 u = vload4(0, (__global const uint *)(W + (size_t)min(r0 + r, rows - 1) * cols + k0 + 8 * kq));
            const float4 lo = as_float4(u << 16), hi = as_float4(u & 0xFFFF0000u);   // values 0 2 4 6 / 1 3 5 7
            wsl[(8 * kq + 0) * 64 + r] = lo.x;
            wsl[(8 * kq + 1) * 64 + r] = hi.x;
            wsl[(8 * kq + 2) * 64 + r] = lo.y;
            wsl[(8 * kq + 3) * 64 + r] = hi.y;
            wsl[(8 * kq + 4) * 64 + r] = lo.z;
            wsl[(8 * kq + 5) * 64 + r] = hi.z;
            wsl[(8 * kq + 6) * 64 + r] = lo.w;
            wsl[(8 * kq + 7) * 64 + r] = hi.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 32; ++k) {
            const float4 hv = vload4(0, hs + k * 32 + it);
            const float4 wa = vload4(0, wsl + k * 64 + ir), wb = vload4(0, wsl + k * 64 + ir + 4);
            a0 = mad((float4)(wa.x), hv, a0);
            a1 = mad((float4)(wa.y), hv, a1);
            a2 = mad((float4)(wa.z), hv, a2);
            a3 = mad((float4)(wa.w), hv, a3);
            a4 = mad((float4)(wb.x), hv, a4);
            a5 = mad((float4)(wb.y), hv, a5);
            a6 = mad((float4)(wb.z), hv, a6);
            a7 = mad((float4)(wb.w), hv, a7);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float4 acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int j = 0; j < 8; ++j) {
        const int r = r0 + ir + j;
        if (r >= rows) break;
        if (t0 + it + 0 < n) out[(size_t)(t0 + it + 0) * os + r] = acc[j].x;
        if (t0 + it + 1 < n) out[(size_t)(t0 + it + 1) * os + r] = acc[j].y;
        if (t0 + it + 2 < n) out[(size_t)(t0 + it + 2) * os + r] = acc[j].z;
        if (t0 + it + 3 < n) out[(size_t)(t0 + it + 3) * os + r] = acc[j].w;
    }
}

// Gated DeltaNet's causal depthwise convolution in place, then silu: channel ch, its K - 1 earlier inputs (state,
// oldest first) carried through the n tokens in order (as gdn_conv / Model::gdn, token by token)
__kernel void pf_gdn_conv(__global float *proj, __global const float *P, int w, __global float *st, int so, int K,
                          int ps, int n) {
    const int ch = get_global_id(0);
    __global const float *wc = P + w + ch * K;
    __global float *s = st + so + ch * (K - 1);
    float sv[8];
    for (int k = 0; k < K - 1; ++k) sv[k] = s[k];
    for (int t = 0; t < n; ++t) {
        const float in = proj[(size_t)t * ps + ch];
        float acc = 0.0f;
        for (int k = 0; k < K - 1; ++k) acc += wc[k] * sv[k];
        acc += wc[K - 1] * in;
        for (int k = 0; k + 1 < K - 1; ++k) sv[k] = sv[k + 1];
        if (K > 1) sv[K - 2] = in;
        proj[(size_t)t * ps + ch] = pf_silu(acc);
    }
    for (int k = 0; k < K - 1; ++k) s[k] = sv[k];
}

// q and k L2-normalised per key head (q also / sqrt(dk)): work groups of dk, groups 0..nk-1 the q heads, nk..2nk-1
// the k ones, dimension 1 the token; then (the first nv items of group 0) the token's gates: exp(g) the decay,
// sigmoid(b) the step, in place in ab = [a nv | b nv] (as gdn_qk and gdn_gates)
__kernel void pf_gdn_qk_gates(__global float *proj, int dk, int kd, float qscale, int ps, __global float *ab, int nv,
                              __global const float *P, int alog, int dtb) {
    __local float tmp[256];
    const int g = get_group_id(0), j = get_local_id(0), nk = get_num_groups(0) / 2, t = get_group_id(1);
    const int isk = g >= nk;
    __global float *p = proj + (size_t)t * ps + (isk ? kd + (g - nk) * dk : g * dk);
    const float v = p[j];
    const float inv = 1.0f / sqrt(pf_sum(v * v, tmp) + 1e-6f);
    p[j] = isk ? v * inv : v * inv * qscale;
    if (g == 0)
        for (int vh = j; vh < nv; vh += dk) {
            __global float *a = ab + (size_t)t * 2 * nv;
            const float x = a[vh] + P[dtb + vh];
            const float sp = x > 20.0f ? x : log1p(exp(x));
            a[vh] = exp(-exp(P[alog + vh]) * sp);
            a[nv + vh] = 1.0f / (1.0f + exp(-a[nv + vh]));
        }
}

// The gated delta rule through n tokens (dk = 128): as gdn_delta32 - a work group = 64 columns of a head x 4 parts of
// 32 rows, the state's column parts in registers for the whole prompt (read once, written once) - with the token
// count at run time and one barrier a token: the four parts' output sums go out as they are (out: per token 4 rows
// of nv * dv, part by part) and pf_gdn_gnorm_q adds them in gdn_delta32's order; the S^T k sums' buffer in turns of 4.
__kernel void pf_gdn_delta32(__global float *state, int so, __global const float *proj, __global const float *ab,
                             __global float *out, int nk, int nv, int dk, int dv, int ps, int n) {
    // q and k of 8 tokens at a time in local memory (each item loads one value of each token: dk = 128, k then q),
    // two buffers in turn: the loop reads them there instead of 64 global loads an item a token
    __local float rms[4][256];
    __local float qk[2][8][256];
    const int g = get_group_id(0), l = get_local_id(0), part = l / 64;
    const int vh = g / (dv / 64), j = (g % (dv / 64)) * 64 + l % 64, c = l % 64;
    const int kh = vh / (nv / nk), kd = nk * dk, i0 = part * 32;
    const size_t at = (size_t)vh * dk * dv + (j / 16) * dk * 16 + (j % 16) + i0 * 16;
    __global float *S = state + so + at;
    float sv[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) sv[i] = S[i * 16];
    for (int t0 = 0, cb = 0; t0 < n; t0 += 8, cb ^= 1) {
        const int m = min(8, n - t0);
        for (int u = 0; u < m; ++u) {
            __global const float *row = proj + (size_t)(t0 + u) * ps;
            qk[cb][u][l] = l < dk ? row[kd + kh * dk + l] : row[kh * dk + (l - dk)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int u = 0; u < m; ++u) {
            const int t = t0 + u;
            __local const float *k = qk[cb][u] + i0, *q = qk[cb][u] + dk + i0;
            const float v = proj[(size_t)t * ps + 2 * kd + vh * dv + j];
            const float decay = ab[t * 2 * nv + vh], beta = ab[t * 2 * nv + nv + vh];
            float mem = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                sv[i] = sv[i] * decay;
                mem += sv[i] * k[i];
            }
            __local float *rm = rms[t & 3];
            rm[l] = mem;
            barrier(CLK_LOCAL_MEM_FENCE);
            const float delta = (v - (((rm[c] + rm[64 + c]) + rm[128 + c]) + rm[192 + c])) * beta;
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                sv[i] = sv[i] + k[i] * delta;
                o += sv[i] * q[i];
            }
            out[((size_t)t * 4 + part) * nv * dv + vh * dv + j] = o;
        }
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) S[i * 16] = sv[i];
}

// gated RMSNorm per value head, o / rms(o) * w * silu(z) (o: the four parts of pf_gdn_delta32 added in its order,
// from core4) into core, then the head's blocks into the out product's tiles (a work group a head, dimension 1 the
// token)
__kernel void pf_gdn_gnorm_q(__global const float *core4, __global float *core, __global const float *z,
                             __global const float *P, int w, float eps, __global uint4 *X, __global float *XS,
                             __global int *XB) {
    __local float tmp[256];
    const int vh = get_group_id(0), j = get_local_id(0), dv = get_local_size(0), t = get_group_id(1);
    const int vd = get_num_groups(0) * dv;
    __global float *o = core + (size_t)t * vd + vh * dv;
    __global const float *p4 = core4 + (size_t)t * 4 * vd + vh * dv + j;
    const float v = ((p4[0] + p4[vd]) + p4[2 * vd]) + p4[3 * vd];
    const float inv = 1.0f / sqrt(pf_sum(v * v, tmp) / (float)dv + eps);
    o[j] = v * inv * P[w + j] * pf_silu(z[(size_t)t * vd + vh * dv + j]);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (j < dv / 16) put_block(vload16(j, o), pf_at(t, vh * (dv / 16) + j, vd / 16), X, XS, XB);
}

// Attention. A head's RMSNorm (1 + w) and partial RoPE, M-RoPE's way: rotated pair i (of hr) turns at the token's
// position in stream i % 3 (pos3: three a token; text: all three the same). Work group t heads + g: head g of token
// t at buf + t ts + g stride.
__kernel void pf_head_norm_rope(__global float *buf, int stride, __global const float *P, int w, float eps, int hr,
                                __global const int *pos3, __global const float *inv_freq, int ts, int heads) {
    // the reduction and the rotation's exchange in buffers of their own (one buffer for both gave NaN: some waves
    // read the reduced sum after others had written the exchange over it)
    __local float tmp[256], rot[256];
    const int g = get_group_id(0) % heads, i = get_local_id(0), hd = get_local_size(0), t = get_group_id(0) / heads;
    __global float *x = buf + (size_t)t * ts + g * stride;
    const float v = x[i];
    const float inv = 1.0f / sqrt(pf_sum(v * v, tmp) / (float)hd + eps);
    const float y = v * inv * (1.0f + P[w + i]);
    rot[i] = y;
    barrier(CLK_LOCAL_MEM_FENCE);
    float r = y;
    if (i < 2 * hr) {
        const int pi = i % hr;
        const float f = (float)pos3[t * 3 + pi % 3] * inv_freq[pi];
        const float c = cos(f), s = sin(f);
        r = i < hr ? rot[i] * c - rot[i + hr] * s : rot[i] * c + rot[i - hr] * s;
    }
    x[i] = r;
}

// n tokens' K and V rows into the int8 cache (as kv_store8: amax / 127 a position and KV head): token t's rows from
// src + t kv_row, its values at ko / vo + t kv_row, its scales at ksi / vsi + t nkv. A work group of head_dim a KV
// head, dimension 1 the token.
__kernel void pf_kv_store8(__global const float *k, __global const float *v, __global char *kv, int ko, int vo,
                           __global float *kvs, int ksi, int vsi, int src, int kv_row, int nkv) {
    __local float tmp[256], tmp2[256];
    const int h = get_group_id(0), i = get_local_id(0), hd = get_local_size(0), t = get_group_id(1);
    const float x = k[src + t * kv_row + h * hd + i], y = v[src + t * kv_row + h * hd + i];
    const float ka = pf_max(fabs(x), tmp), va = pf_max(fabs(y), tmp2);
    kv[ko + t * kv_row + h * hd + i] = (char)clamp(rint(x * (ka > 0.0f ? 127.0f / ka : 0.0f)), -127.0f, 127.0f);
    kv[vo + t * kv_row + h * hd + i] = (char)clamp(rint(y * (va > 0.0f ? 127.0f / va : 0.0f)), -127.0f, 127.0f);
    if (i == 0) {
        kvs[ksi + t * nkv + h] = ka / 127.0f;
        kvs[vsi + t * nkv + h] = va / 127.0f;
    }
}

// A prompt's causal attention over one chunk of the int8 cache, the softmax online across chunks (launches in
// order): a work group = token t x KV head g, its 8 query heads (GQA) at once, 256 items. Positions [c0, c0 + len)
// of the chunk, those <= the token's (kv0 + t) seen. pm / pl / po: each (token, head)'s running max, sum and
// unnormalised output (first: start them). Scores: item i a position, all 8 heads from one read of its K row; the
// block's max and sum of the 8 heads as float8 reductions (two a block, not sixteen); the weights in half; values:
// item i a dimension, every head from one read of each V row.
#define PG 8
__kernel void pf_attn8(__global const float *qg, int qs, __global const char *kv, int kb, int vb, int kv_row,
                       __global const float *kvs, int ksb, int vsb, int nkv, int c0, int len, int kv0, float scale,
                       __global float *pm, __global float *pl, __global float *po, int nh, int first, int n) {
    __local float q[PG * 256];
    __local float pw[PG * 256];
    __local float red[4][256];   // the reductions' buffers in turn (a reduced buffer is not written right after)
    const int t = get_group_id(0), g = get_group_id(1), i = get_local_id(0), hd = 256;
    const int last = kv0 + t;   // the token's own position: it sees up to it
    // the 8 heads' queries
    for (int j = 0; j < PG; ++j) q[j * hd + i] = qg[(size_t)t * qs + (g * PG + j) * 2 * hd + i] * scale;
    float m[PG], ls[PG], o[PG];
    for (int j = 0; j < PG; ++j) {
        const size_t at = (size_t)t * nh + g * PG + j;
        m[j] = first ? -INFINITY : pm[at];
        ls[j] = first ? 0.0f : pl[at];
        o[j] = first ? 0.0f : po[at * hd + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int end = min(c0 + len, last + 1);   // positions [c0, end) of this chunk
    for (int b0 = c0; b0 < end; b0 += 256) {
        const int p = b0 + i;
        const int valid = p < end;
        float s[PG];
        for (int j = 0; j < PG; ++j) s[j] = 0.0f;
        if (valid) {
            __global const char *kr = kv + kb + (size_t)(p - c0) * kv_row + g * hd;
            for (int d = 0; d < hd; d += 16) {
                const float16 kk = convert_float16(vload16(0, kr + d));
                for (int j = 0; j < PG; ++j) {
                    const float16 qq = vload16(0, q + j * hd + d);
                    const float16 pr = kk * qq;
                    s[j] += ((pr.s0 + pr.s1) + (pr.s2 + pr.s3)) + ((pr.s4 + pr.s5) + (pr.s6 + pr.s7)) +
                            ((pr.s8 + pr.s9) + (pr.sa + pr.sb)) + ((pr.sc + pr.sd) + (pr.se + pr.sf));
                }
            }
            const float ks = kvs[ksb + (p - c0) * nkv + g];
            for (int j = 0; j < PG; ++j) s[j] *= ks;
        }
        // the block's max per head, the running max, the weights
        for (int j = 0; j < PG; ++j) {
            const float bm = pf_max(valid ? s[j] : -INFINITY, red[(2 * j) % 4]);
            const float mn = fmax(m[j], bm);
            const float corr = exp(m[j] - mn);   // (m = -inf at first: 0)
            const float e = valid ? exp(s[j] - mn) : 0.0f;
            pw[j * 256 + i] = e;
            ls[j] = ls[j] * corr + pf_sum(e, red[(2 * j + 1) % 4]);
            o[j] *= corr;
            m[j] = mn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // the values: item i = dimension i
        const int nrow = min(256, end - b0);
        __global const char *vc = kv + vb + (size_t)(b0 - c0) * kv_row + g * hd + i;
        __global const float *vsc = kvs + vsb + (b0 - c0) * nkv + g;
        for (int r = 0; r < nrow; ++r) {
            const float vv = (float)vc[(size_t)r * kv_row] * vsc[r * nkv];
            for (int j = 0; j < PG; ++j) o[j] += pw[j * 256 + r] * vv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int j = 0; j < PG; ++j) {
        const size_t at = (size_t)t * nh + g * PG + j;
        if (i == 0) {
            pm[at] = m[j];
            pl[at] = ls[j];
        }
        po[at * hd + i] = o[j];
    }
}

// The prompt's attention in waves of 64 (pf_attn8x, pf_attn8t): the softmax's max and sum as sub-group reductions
// (pf_attn8 reduced over 256 items through local memory, 16 times a block); other sub-group sizes fall back to
// the local-memory reductions.
#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif
inline float w_max(float v, __local float *tmp, int sg) { return sg ? sub_group_reduce_max(v) : pf_max(v, tmp); }
inline float w_sum(float v, __local float *tmp, int sg) { return sg ? sub_group_reduce_add(v) : pf_sum(v, tmp); }
// A wave for token t x KV head g, 256 positions a step, an item
// four of them (i, i + 64, ..): per 8 dimensions it reads the 8 heads' queries as one float8 a dimension (q kept
// [dimension][head]) and its four K rows' 8 bytes, 256 multiply-adds; the softmax's sums as sub-group reductions; the
// weights stored [position][head] already times the position's V scale, so the values read one float8 a row.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void pf_attn8x(__global const float *qg, int qs, __global const char *kv, int kb, int vb, int kv_row,
               __global const float *kvs, int ksb, int vsb, int nkv, int c0, int len, int kv0, float scale,
               __global float *pm, __global float *pl, __global float *po, int nh, int first, int n) {
    __local float q[256 * PG];   // [dimension][head]
    __local float pw[256 * PG];  // [position][head], times the position's V scale
    __local float red[2][64];
    const int t = get_group_id(0), g = get_group_id(1), i = get_local_id(0), hd = 256;
    const int sg = get_sub_group_size() >= get_local_size(0);
    const int last = kv0 + t;
    for (int j = 0; j < PG; ++j)
        for (int d = i; d < hd; d += 64) q[d * PG + j] = qg[(size_t)t * qs + (g * PG + j) * 2 * hd + d] * scale;
    float m[PG], ls[PG];
    float4 o[PG];
    for (int j = 0; j < PG; ++j) {
        const size_t at = (size_t)t * nh + g * PG + j;
        m[j] = first ? -INFINITY : pm[at];
        ls[j] = first ? 0.0f : pl[at];
        o[j] = first ? (float4)(0.0f) : vload4(i, po + at * hd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int end = min(c0 + len, last + 1);
    for (int b0 = c0; b0 < end; b0 += 256) {
        float8 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        const int p0 = b0 + i, p1 = p0 + 64, p2 = p0 + 128, p3 = p0 + 192;
        const int ok0 = p0 < end, ok1 = p1 < end, ok2 = p2 < end, ok3 = p3 < end;
        // a position past the end reads row c0 (in the chunk): its score is thrown away below
        __global const char *k0 = kv + kb + (size_t)((ok0 ? p0 : c0) - c0) * kv_row + g * hd;
        __global const char *k1 = kv + kb + (size_t)((ok1 ? p1 : c0) - c0) * kv_row + g * hd;
        __global const char *k2 = kv + kb + (size_t)((ok2 ? p2 : c0) - c0) * kv_row + g * hd;
        __global const char *k3 = kv + kb + (size_t)((ok3 ? p3 : c0) - c0) * kv_row + g * hd;
        for (int d = 0; d < hd; d += 8) {
            const float8 a0 = convert_float8(vload8(0, k0 + d)), a1 = convert_float8(vload8(0, k1 + d));
            const float8 a2 = convert_float8(vload8(0, k2 + d)), a3 = convert_float8(vload8(0, k3 + d));
#define QK(e)                                                                                          \
    {                                                                                                  \
        const float8 qq = vload8(0, q + (d + e) * PG);                                                 \
        s0 = mad(qq, (float8)(a0.s##e), s0);                                                           \
        s1 = mad(qq, (float8)(a1.s##e), s1);                                                           \
        s2 = mad(qq, (float8)(a2.s##e), s2);                                                           \
        s3 = mad(qq, (float8)(a3.s##e), s3);                                                           \
    }
            QK(0) QK(1) QK(2) QK(3) QK(4) QK(5) QK(6) QK(7)
#undef QK
        }
        s0 *= ok0 ? kvs[ksb + (p0 - c0) * nkv + g] : 0.0f;
        s1 *= ok1 ? kvs[ksb + (p1 - c0) * nkv + g] : 0.0f;
        s2 *= ok2 ? kvs[ksb + (p2 - c0) * nkv + g] : 0.0f;
        s3 *= ok3 ? kvs[ksb + (p3 - c0) * nkv + g] : 0.0f;
        const float v0 = ok0 ? kvs[vsb + (p0 - c0) * nkv + g] : 0.0f, v1 = ok1 ? kvs[vsb + (p1 - c0) * nkv + g] : 0.0f;
        const float v2 = ok2 ? kvs[vsb + (p2 - c0) * nkv + g] : 0.0f, v3 = ok3 ? kvs[vsb + (p3 - c0) * nkv + g] : 0.0f;
        float e0[PG], e1[PG], e2[PG], e3[PG];
#define SOFT(j)                                                                                        \
    {                                                                                                  \
        float bm = -INFINITY;                                                                          \
        if (ok0) bm = s0.s##j;                                                                         \
        if (ok1) bm = fmax(bm, s1.s##j);                                                               \
        if (ok2) bm = fmax(bm, s2.s##j);                                                               \
        if (ok3) bm = fmax(bm, s3.s##j);                                                               \
        const float mn = fmax(m[j], w_max(bm, red[0], sg));                                            \
        const float corr = exp(m[j] - mn);                                                             \
        const float x0 = ok0 ? exp(s0.s##j - mn) : 0.0f, x1 = ok1 ? exp(s1.s##j - mn) : 0.0f;          \
        const float x2 = ok2 ? exp(s2.s##j - mn) : 0.0f, x3 = ok3 ? exp(s3.s##j - mn) : 0.0f;          \
        e0[j] = x0 * v0;                                                                               \
        e1[j] = x1 * v1;                                                                               \
        e2[j] = x2 * v2;                                                                               \
        e3[j] = x3 * v3;                                                                               \
        ls[j] = ls[j] * corr + w_sum((x0 + x1) + (x2 + x3), red[1], sg);                               \
        o[j] *= corr;                                                                                  \
        m[j] = mn;                                                                                     \
    }
        SOFT(0) SOFT(1) SOFT(2) SOFT(3) SOFT(4) SOFT(5) SOFT(6) SOFT(7)
#undef SOFT
        vstore8((float8)(e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7]), 0, pw + i * PG);
        vstore8((float8)(e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7]), 0, pw + (i + 64) * PG);
        vstore8((float8)(e2[0], e2[1], e2[2], e2[3], e2[4], e2[5], e2[6], e2[7]), 0, pw + (i + 128) * PG);
        vstore8((float8)(e3[0], e3[1], e3[2], e3[3], e3[4], e3[5], e3[6], e3[7]), 0, pw + (i + 192) * PG);
        barrier(CLK_LOCAL_MEM_FENCE);
        // the values: item i = dimensions 4i .. 4i + 3
        const int nrow = min(256, end - b0);
        __global const char *vc = kv + vb + (size_t)(b0 - c0) * kv_row + g * hd;
        for (int r = 0; r < nrow; ++r) {
            const float4 vv = convert_float4(vload4(i, vc + (size_t)r * kv_row));
            const float8 w = vload8(0, pw + r * PG);
            o[0] = mad((float4)(w.s0), vv, o[0]);
            o[1] = mad((float4)(w.s1), vv, o[1]);
            o[2] = mad((float4)(w.s2), vv, o[2]);
            o[3] = mad((float4)(w.s3), vv, o[3]);
            o[4] = mad((float4)(w.s4), vv, o[4]);
            o[5] = mad((float4)(w.s5), vv, o[5]);
            o[6] = mad((float4)(w.s6), vv, o[6]);
            o[7] = mad((float4)(w.s7), vv, o[7]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int j = 0; j < PG; ++j) {
        const size_t at = (size_t)t * nh + g * PG + j;
        if (i == 0) {
            pm[at] = m[j];
            pl[at] = ls[j];
        }
        vstore4(o[j], i, po + at * hd);
    }
}

// Four tokens a work group: 4 waves, wave w for token t0 + w (KV head g), each as pf_attn8x but 128 positions a step
// (an item two of them); the four read the same K and V rows in step (a barrier a step), so each row comes from
// memory about once for the four (per token the K and V reads were most of the time). The queries in half precision
// in local memory (rounding ~5e-4, far below the int8 K's), the weights (times the V scale) in float: in half they
// cost KL 3e-3 from the CPU instead of 1.5e-3 - thousands of small weights at long contexts. That fills the 32 KB of
// local memory, so the softmax's sums are sub-group reductions only: the engine takes this kernel only where a wave
// of 64 is a sub-group (pf_probe).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define AQ 4
__kernel __attribute__((reqd_work_group_size(64 * AQ, 1, 1)))
void pf_attn8t(__global const float *qg, int qs, __global const char *kv, int kb, int vb, int kv_row,
               __global const float *kvs, int ksb, int vsb, int nkv, int c0, int len, int kv0, float scale,
               __global float *pm, __global float *pl, __global float *po, int nh, int first, int n) {
    __local half qh[AQ * 256 * PG];   // [wave][dimension][head]
    __local float pw[AQ * 128 * PG];  // [wave][position][head], times the position's V scale
    const int g = get_group_id(1), l = get_local_id(0), w = l / 64, i = l % 64, hd = 256;
    const int t = (int)get_group_id(0) * AQ + w;   // this wave's token
    const int live = t < n;
    const int last = kv0 + (live ? t : n - 1);
    __local half *q = qh + w * 256 * PG;
    __local float *pwv = pw + w * 128 * PG;
    for (int j = 0; j < PG; ++j)
        for (int d = i; d < hd; d += 64)
            vstore_half_rte(live ? qg[(size_t)t * qs + (g * PG + j) * 2 * hd + d] * scale : 0.0f, d * PG + j, q);
    float m[PG], ls[PG];
    float4 o[PG];
    for (int j = 0; j < PG; ++j) {
        const size_t at = (size_t)(live ? t : 0) * nh + g * PG + j;
        m[j] = first || !live ? -INFINITY : pm[at];
        ls[j] = first || !live ? 0.0f : pl[at];
        o[j] = first || !live ? (float4)(0.0f) : vload4(i, po + at * hd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // the group's furthest token decides the steps; each wave masks its own positions
    const int tl = min((int)get_group_id(0) * AQ + AQ - 1, n - 1);
    const int gend = min(c0 + len, kv0 + tl + 1);
    const int end = min(c0 + len, last + 1);
    for (int b0 = c0; b0 < gend; b0 += 128) {
        float8 s0 = 0.0f, s1 = 0.0f;
        const int p0 = b0 + i, p1 = p0 + 64;
        const int ok0 = p0 < end, ok1 = p1 < end;
        // (positions past the group's end read row c0, in the chunk; their scores are thrown away)
        __global const char *k0 = kv + kb + (size_t)((p0 < gend ? p0 : c0) - c0) * kv_row + g * hd;
        __global const char *k1 = kv + kb + (size_t)((p1 < gend ? p1 : c0) - c0) * kv_row + g * hd;
        for (int d = 0; d < hd; d += 8) {
            const float8 a0 = convert_float8(vload8(0, k0 + d)), a1 = convert_float8(vload8(0, k1 + d));
#define QK(e)                                                                                          \
    {                                                                                                  \
        const float8 qq = vload_half8(d + e, q);                                                       \
        s0 = mad(qq, (float8)(a0.s##e), s0);                                                           \
        s1 = mad(qq, (float8)(a1.s##e), s1);                                                           \
    }
            QK(0) QK(1) QK(2) QK(3) QK(4) QK(5) QK(6) QK(7)
#undef QK
        }
        s0 *= ok0 ? kvs[ksb + (p0 - c0) * nkv + g] : 0.0f;
        s1 *= ok1 ? kvs[ksb + (p1 - c0) * nkv + g] : 0.0f;
        const float v0 = ok0 ? kvs[vsb + (p0 - c0) * nkv + g] : 0.0f, v1 = ok1 ? kvs[vsb + (p1 - c0) * nkv + g] : 0.0f;
        float8 e0, e1;
#define SOFT(j)                                                                                        \
    {                                                                                                  \
        float bm = -INFINITY;                                                                          \
        if (ok0) bm = s0.s##j;                                                                         \
        if (ok1) bm = fmax(bm, s1.s##j);                                                               \
        const float mn = fmax(m[j], sub_group_reduce_max(bm));                                         \
        const float corr = mn == -INFINITY ? 1.0f : exp(m[j] - mn);                                    \
        const float x0 = ok0 ? exp(s0.s##j - mn) : 0.0f, x1 = ok1 ? exp(s1.s##j - mn) : 0.0f;          \
        e0.s##j = x0 * v0;                                                                             \
        e1.s##j = x1 * v1;                                                                             \
        ls[j] = ls[j] * corr + sub_group_reduce_add(x0 + x1);                                          \
        o[j] *= corr;                                                                                  \
        m[j] = mn;                                                                                     \
    }
        SOFT(0) SOFT(1) SOFT(2) SOFT(3) SOFT(4) SOFT(5) SOFT(6) SOFT(7)
#undef SOFT
        vstore8(e0, i, pwv);
        vstore8(e1, i + 64, pwv);
        barrier(CLK_LOCAL_MEM_FENCE);
        // the values: item i = dimensions 4i .. 4i + 3; rows past this wave's end have weight 0
        const int nrow = min(128, gend - b0);
        __global const char *vc = kv + vb + (size_t)(b0 - c0) * kv_row + g * hd;
        for (int r = 0; r < nrow; ++r) {
            const float4 vv = convert_float4(vload4(i, vc + (size_t)r * kv_row));
            const float8 x = vload8(r, pwv);
            o[0] = mad((float4)(x.s0), vv, o[0]);
            o[1] = mad((float4)(x.s1), vv, o[1]);
            o[2] = mad((float4)(x.s2), vv, o[2]);
            o[3] = mad((float4)(x.s3), vv, o[3]);
            o[4] = mad((float4)(x.s4), vv, o[4]);
            o[5] = mad((float4)(x.s5), vv, o[5]);
            o[6] = mad((float4)(x.s6), vv, o[6]);
            o[7] = mad((float4)(x.s7), vv, o[7]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (!live) return;
    for (int j = 0; j < PG; ++j) {
        const size_t at = (size_t)t * nh + g * PG + j;
        if (i == 0) {
            pm[at] = m[j];
            pl[at] = ls[j];
        }
        vstore4(o[j], i, po + at * hd);
    }
}

// what a work group of 256 is made of here: out[0] its sub-groups' size, out[1] their number (pf_attn8t needs 64, 4)
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void pf_probe(__global int *out) {
    if (get_local_id(0) == 0) {
        out[0] = (int)get_sub_group_size();
        out[1] = (int)get_num_sub_groups();
    }
}

// the attention's output: each head's o / l times sigmoid(its gate), into the o product's tiles (an item a block of
// 16 dimensions, dimension 1 the token)
// pf_attn_out's floats, for a projection elsewhere (Gpu8::set_host_gemm): token t's nh * hd values a row
__kernel void pf_attn_out_f(__global const float *po, __global const float *pl, __global const float *qg, int qs,
                            int hd, int nh, __global float *out) {
    const int b = get_global_id(0), t = get_global_id(1), h = (b * 16) / hd, d0 = (b * 16) % hd;
    const size_t at = (size_t)t * nh + h;
    const float16 g = vload16(0, qg + (size_t)t * qs + h * 2 * hd + hd + d0);
    const float16 o = vload16(0, po + at * hd + d0) * (1.0f / pl[at]) * (1.0f / (1.0f + exp(-g)));
    vstore16(o, 0, out + (size_t)t * nh * hd + b * 16);
}

__kernel void pf_attn_out(__global const float *po, __global const float *pl, __global const float *qg, int qs, int hd,
                          int nh, __global uint4 *X, __global float *XS, __global int *XB) {
    const int b = get_global_id(0), t = get_global_id(1), h = (b * 16) / hd, d0 = (b * 16) % hd;
    const size_t at = (size_t)t * nh + h;
    const float16 g = vload16(0, qg + (size_t)t * qs + h * 2 * hd + hd + d0);
    const float16 o = vload16(0, po + at * hd + d0) * (1.0f / pl[at]) * (1.0f / (1.0f + exp(-g)));
    put_block(o, pf_at(t, b, nh * hd / 16), X, XS, XB);
}
#endif   // GEODE_PF
)CL"
