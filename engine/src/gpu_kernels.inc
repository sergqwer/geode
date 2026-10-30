// gpu_kernels.inc - the OpenCL C source of the GPU path's kernels (docs/gpu-opencl-design.md), one string the engine
// builds at run time (a program per variant: -DNT, -DLANES, -DFMT_*).
//
// Matrix-vector (generating, 1-4 tokens): y[t][r] = global * sum_k W[r][k] x[t][k], W in the pack's row-interleaved
// format (a row: its code bytes, then one E4M3 scale byte per 16 values). LANES work items share a row, each takes
// whole 16-byte chunks of its codes (neighbouring lanes, neighbouring bytes), 64 / LANES rows per work group of 64;
// the lanes' sums meet in local memory.
//
// Version 2: no arrays (the compiler put them in private memory, 0.1-0.6 GB/s), the codes decoded two at a time into
// half2 by bit operations, FP16 products of a block summed in half2, the block's sum scaled in FP32 — and mad(), never
// fma(): the Adreno compiler emulates an FP16 fma in software (90x slower). The activations
// come prepared by the host (prepare_activations in the test): each 16-value block divided by its largest magnitude
// (so FP16 sums of 16 products stay small), in the order the pairs come out of a code word, with that block's divisor
// in xs.
//   q2, q3   pair k of a block = values (k, k+8)            [x0 x8 x1 x9 ... x7 x15]
//   nvfp4    pair k of word 0 = values (k, k+4), of word 1 = (8+k, 12+k), and x times 1/2 (the nibble decodes to
//            twice its E2M1 value)
// Without cl_khr_fp16 (a PC's GPU, for the correctness test) the same code runs in FP32: h2 is float2, the FP16 bit
// patterns are widened by hand.
R"CL(
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half2 h2;
typedef half16 h16;
#define HC(v) v##h
#define LOAD_X16(p) vload16(0, p)
#define H2_BITS(u) as_half2(u)
#else
typedef float2 h2;
typedef float16 h16;
#define HC(v) v##f
#define LOAD_X16(p) vload_half16(0, p)
inline float half_bits(uint b) {
    const uint e = (b >> 10) & 31u, m = b & 1023u;
    const float v = e ? as_float(((e + 112u) << 23) | (m << 13)) : (float)m * 5.9604645e-8f;
    return (b & 0x8000u) ? -v : v;
}
#define H2_BITS(u) ((float2)(half_bits((u) & 0xFFFFu), half_bits((u) >> 16)))
#endif
#ifndef NT
#define NT 1
#endif
#ifndef LANES
#define LANES 32
#endif
#define ROWS_PER_GROUP (64 / LANES)

// an E4M3 scale byte (never negative in a pack) as a float: exponent bias 7 -> 127
inline float e4m3(uint b) {
    const uint e = (b >> 3) & 15u, m = b & 7u;
    return e ? as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
}

#if defined(FMT_Q2)
#define CODE_BYTES 4
// codes k and k+8 of a word -> their levels 1 3 -1 -3: 1.0 is 0x3C00, 3.0 is 0x4200, the high bit is the sign
inline h2 pair_q2(uint w, int k) {
    const uint p = (w >> (2 * k)) & 0x00030003u;
    return H2_BITS((0x3C003C00u + (p & 0x00010001u) * 0x0600u) | ((p & 0x00020002u) << 14));
}
#define BLOCK_DOT_Q2(w, xv, s2)                                                                                    \
    s2 = mad(pair_q2(w, 0), xv.s01, s2); s2 = mad(pair_q2(w, 1), xv.s23, s2);                                      \
    s2 = mad(pair_q2(w, 2), xv.s45, s2); s2 = mad(pair_q2(w, 3), xv.s67, s2);                                      \
    s2 = mad(pair_q2(w, 4), xv.s89, s2); s2 = mad(pair_q2(w, 5), xv.sab, s2);                                      \
    s2 = mad(pair_q2(w, 6), xv.scd, s2); s2 = mad(pair_q2(w, 7), xv.sef, s2);
#elif defined(FMT_NVFP4)
#define CODE_BYTES 8
// nibbles k and k+4 of a word -> 2 x E2M1 as normal FP16: the magnitude bits e e m to bits 11..9 plus 15 on the
// exponent give 2^e (1 + m/2), which is 2 x E2M1 for e >= 1; for e = 0 it is 1 + m/2 where 2 x E2M1 is m, so those
// halves take 2v - 2 instead (the Adreno flushes FP16 denormals: E2M1 / 2^14 as a subnormal reads 0.5 as 0). The sign
// to bit 15.
inline h2 pair_nv(uint w, int k) {
    const uint q = w >> (4 * k);
    const uint p = q & 0x00070007u;
    const h2 v = H2_BITS((p << 9) + 0x3C003C00u);
    const h2 v0 = mad(v, (h2)(HC(2.0), HC(2.0)), (h2)(HC(-2.0), HC(-2.0)));
#ifdef cl_khr_fp16
    const h2 r = select(v, v0, as_short2(p & 0x00060006u) == (short2)0);
    return as_half2(as_uint(r) ^ ((q & 0x00080008u) << 12));
#else
    const float2 r = (float2)((p & 6u) ? v.x : v0.x, (p & 0x60000u) ? v.y : v0.y);
    return (float2)((q & 8u) ? -r.x : r.x, (q & 0x80000u) ? -r.y : r.y);
#endif
}
#define WORD_DOT_NV(w, xa, s2)                                                                                     \
    s2 = mad(pair_nv(w, 0), xa.s01, s2); s2 = mad(pair_nv(w, 1), xa.s23, s2);                                      \
    s2 = mad(pair_nv(w, 2), xa.s45, s2); s2 = mad(pair_nv(w, 3), xa.s67, s2);
#elif defined(FMT_Q3)
#define CODE_BYTES 6
// low bits of values k, k+8 (as q2) -> 2c+1 through FP16's 1024 + c; the high bits (at bits k, k+8 of hb) the sign
inline h2 pair_q3(uint lw, uint hb, int k) {
    const uint p = (lw >> (2 * k)) & 0x00030003u;
    const h2 mag = mad(H2_BITS(0x64006400u | p), (h2)(HC(2.0), HC(2.0)), (h2)(HC(-2047.0), HC(-2047.0)));
    const uint t = hb >> k;
#ifdef cl_khr_fp16
    return as_half2(as_uint(mag) ^ (((t & 1u) << 15) | ((t & 0x100u) << 23)));
#else
    return (float2)((t & 1u) ? -mag.x : mag.x, (t & 0x100u) ? -mag.y : mag.y);
#endif
}
#define BLOCK_DOT_Q3(lw, hb, xv, s2)                                                                               \
    s2 = mad(pair_q3(lw, hb, 0), xv.s01, s2); s2 = mad(pair_q3(lw, hb, 1), xv.s23, s2);                            \
    s2 = mad(pair_q3(lw, hb, 2), xv.s45, s2); s2 = mad(pair_q3(lw, hb, 3), xv.s67, s2);                            \
    s2 = mad(pair_q3(lw, hb, 4), xv.s89, s2); s2 = mad(pair_q3(lw, hb, 5), xv.sab, s2);                            \
    s2 = mad(pair_q3(lw, hb, 6), xv.scd, s2); s2 = mad(pair_q3(lw, hb, 7), xv.sef, s2);
#endif

// one block of token t: its half2 sum, times the weight scale and the block's activation divisor, into acc
#define FOLD(acc, s2, ws, xs_t, blk) acc += ((float)s2.x + (float)s2.y) * ws * xs_t[blk];

__kernel void matvec(__global const uchar *w, uint w_off, int rows, int cols, int row_bytes, float global_scale,
                     __global const half *xh, __global const float *xs, __global float *y) {
    __local float part[64 * NT];
    const int lid = get_local_id(0);
    const int lane = lid % LANES;
    const int row = get_group_id(0) * ROWS_PER_GROUP + lid / LANES;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int nb = cols >> 4;
    if (row < rows) {
        __global const uchar *r = w + w_off + (size_t)row * (size_t)row_bytes;
        __global const uchar *sb = r + nb * CODE_BYTES;
#if defined(FMT_Q2)
        // a chunk: 16 code bytes = 4 blocks
        for (int ch = lane; ch < (nb >> 2); ch += LANES) {
            const uint4 cw = vload4(0, (__global const uint *)(r + ch * 16));
            const uint sw = *(__global const uint *)(sb + ch * 4);
#define Q2_BLOCK(b, word)                                                                                          \
            {                                                                                                      \
                const int blk = ch * 4 + b;                                                                        \
                const float ws = e4m3((sw >> (8 * b)) & 255u);                                                     \
                { h2 s2 = 0; const h16 xv = LOAD_X16(xh + blk * 16); BLOCK_DOT_Q2(word, xv, s2)            \
                  FOLD(acc0, s2, ws, xs, blk) }                                                                    \
                if (NT > 1) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + cols + blk * 16);                   \
                  BLOCK_DOT_Q2(word, xv, s2) FOLD(acc1, s2, ws, (xs + nb), blk) }                                  \
                if (NT > 2) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + 2 * cols + blk * 16);               \
                  BLOCK_DOT_Q2(word, xv, s2) FOLD(acc2, s2, ws, (xs + 2 * nb), blk) }                              \
                if (NT > 3) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + 3 * cols + blk * 16);               \
                  BLOCK_DOT_Q2(word, xv, s2) FOLD(acc3, s2, ws, (xs + 3 * nb), blk) }                              \
            }
            Q2_BLOCK(0, cw.s0) Q2_BLOCK(1, cw.s1) Q2_BLOCK(2, cw.s2) Q2_BLOCK(3, cw.s3)
        }
#elif defined(FMT_NVFP4)
        // a chunk: 16 code bytes = 2 blocks of 2 words
        for (int ch = lane; ch < (nb >> 1); ch += LANES) {
            const uint4 cw = vload4(0, (__global const uint *)(r + ch * 16));
            const uint sw = *(__global const ushort *)(sb + ch * 2);
#define NV_BLOCK(b, wa, wb)                                                                                        \
            {                                                                                                      \
                const int blk = ch * 2 + b;                                                                        \
                const float ws = e4m3((sw >> (8 * b)) & 255u);                                                     \
                { h2 s2 = 0; const h16 xv = LOAD_X16(xh + blk * 16); WORD_DOT_NV(wa, xv.lo, s2)            \
                  WORD_DOT_NV(wb, xv.hi, s2) FOLD(acc0, s2, ws, xs, blk) }                                         \
                if (NT > 1) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + cols + blk * 16);                    \
                  WORD_DOT_NV(wa, xv.lo, s2) WORD_DOT_NV(wb, xv.hi, s2) FOLD(acc1, s2, ws, (xs + nb), blk) }       \
                if (NT > 2) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + 2 * cols + blk * 16);               \
                  WORD_DOT_NV(wa, xv.lo, s2) WORD_DOT_NV(wb, xv.hi, s2) FOLD(acc2, s2, ws, (xs + 2 * nb), blk) }   \
                if (NT > 3) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + 3 * cols + blk * 16);               \
                  WORD_DOT_NV(wa, xv.lo, s2) WORD_DOT_NV(wb, xv.hi, s2) FOLD(acc3, s2, ws, (xs + 3 * nb), blk) }   \
            }
            NV_BLOCK(0, cw.s0, cw.s1) NV_BLOCK(1, cw.s2, cw.s3)
        }
#elif defined(FMT_Q3)
        // a chunk: one 24-byte group of 64 values = 4 blocks
        for (int g = lane; g < (nb >> 2); g += LANES) {
            __global const uchar *gp = r + g * 24;
            const uint4 lo = vload4(0, (__global const uint *)gp);
            const uint2 hi = vload2(0, (__global const uint *)(gp + 16));
            const uint sw = *(__global const uint *)(sb + g * 4);
#define Q3_BLOCK(b, lw, hb)                                                                                        \
            {                                                                                                      \
                const int blk = g * 4 + b;                                                                         \
                const float ws = e4m3((sw >> (8 * b)) & 255u);                                                     \
                { h2 s2 = 0; const h16 xv = LOAD_X16(xh + blk * 16); BLOCK_DOT_Q3(lw, hb, xv, s2)          \
                  FOLD(acc0, s2, ws, xs, blk) }                                                                    \
                if (NT > 1) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + cols + blk * 16);                    \
                  BLOCK_DOT_Q3(lw, hb, xv, s2) FOLD(acc1, s2, ws, (xs + nb), blk) }                                \
                if (NT > 2) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + 2 * cols + blk * 16);               \
                  BLOCK_DOT_Q3(lw, hb, xv, s2) FOLD(acc2, s2, ws, (xs + 2 * nb), blk) }                            \
                if (NT > 3) { h2 s2 = 0; const h16 xv = LOAD_X16(xh + 3 * cols + blk * 16);               \
                  BLOCK_DOT_Q3(lw, hb, xv, s2) FOLD(acc3, s2, ws, (xs + 3 * nb), blk) }                            \
            }
            Q3_BLOCK(0, lo.s0, hi.s0) Q3_BLOCK(1, lo.s1, (hi.s0 >> 16)) Q3_BLOCK(2, lo.s2, hi.s1)
            Q3_BLOCK(3, lo.s3, (hi.s1 >> 16))
        }
#endif
    }
    // the row's lanes meet in local memory
    part[lid * NT + 0] = acc0;
    if (NT > 1) part[lid * NT + 1] = acc1;
    if (NT > 2) part[lid * NT + 2] = acc2;
    if (NT > 3) part[lid * NT + 3] = acc3;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0 && row < rows) {
        for (int t = 0; t < NT; ++t) {
            float s = 0.0f;
            for (int i = 0; i < LANES; ++i) s += part[(lid + i) * NT + t];
            y[t * rows + row] = s * global_scale;
        }
    }
}
)CL"
