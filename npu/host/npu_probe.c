// npu/host/npu_probe.c - opens Geode's cDSP module in an unsigned PD and measures it (probe.c): FastRPC's round
// trip, then HMX GEMMs of the model's shapes.
//
//     ADSP_LIBRARY_PATH=/data/local/tmp/npu ./npu_probe [reps]
//     ./npu_probe 1 PATTERN            the raw 32x32 fp16 result of hmx_debug (PATTERN 10-19: thread probes)
//     ./npu_probe 1 M K N [MODE]       one checked GEMM
//     ./npu_probe bw [MB]              the DSP's DDR read speed (HVX threads; copy / loads / copy + l2fetch)
//     ./npu_probe gemv [threads] [variant]  one decode step of Qwen3.5-2B's products on the DSP (geode_npu_gemv,
//                                      ~1.2 GB; variant 0 qf16 sums, 1 vrmpy, 2 lookup tables)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "remote.h"
#include "rpcmem.h"
#include "geode_npu.h"
#include "probe.h"
#include "../moe_cfg.h"
#include "../attn_cfg.h"

static void to_stdout(void *ctx, const char *line) {
    (void)ctx;
    fputs(line, stdout);
}

// npu_imp.c's gemv_desc_t
typedef struct {
    uint64_t w_off;
    uint32_t k, n, x_at, y_at;
    int32_t shift;
    uint32_t pad;
} gemv_desc_t;

#define HMX4_BYTES(k) ((uint64_t)(k) * 20u)

// One decode step of Qwen3.5-2B's products on the DSP (random q4 weights in the hmx4 layout, every matrix a token
// reads, lm_head included): the first, a 2048 x 2048, checked against a float sum; then the step's time per thread
// count. The weights take ~1.17 GB of rpcmem: check MemAvailable first.
static int gemv_step(remote_handle64 h, int max_threads, int variant) {
    int shapes[200][2], count = 0;   // (n, k)
    shapes[count][0] = 2048, shapes[count++][1] = 2048;   // a z projection first: the checked one
    for (int l = 0; l < 24; ++l) {
        const int attn = l % 4 == 3;
        if (attn) {
            const int a[4][2] = {{4096, 2048}, {512, 2048}, {512, 2048}, {2048, 2048}};
            for (int i = 0; i < 4; ++i) shapes[count][0] = a[i][0], shapes[count++][1] = a[i][1];
        } else {
            const int g[3][2] = {{6144, 2048}, {2048, 2048}, {2048, 2048}};
            for (int i = 0; i < 3; ++i) shapes[count][0] = g[i][0], shapes[count++][1] = g[i][1];
        }
        const int m[3][2] = {{6144, 2048}, {6144, 2048}, {2048, 6144}};
        for (int i = 0; i < 3; ++i) shapes[count][0] = m[i][0], shapes[count++][1] = m[i][1];
    }
    shapes[count][0] = 248320, shapes[count++][1] = 2048;   // lm_head
    uint64_t w_len = 0, x_len = 0, y_len = 0, weights = 0;
    gemv_desc_t *d = (gemv_desc_t *)calloc((size_t)count, sizeof *d);
    for (int i = 0; i < count; ++i) {
        d[i].w_off = w_len;
        d[i].k = (uint32_t)shapes[i][1];
        d[i].n = (uint32_t)shapes[i][0];
        d[i].x_at = (uint32_t)x_len;
        d[i].y_at = (uint32_t)y_len;
        w_len += (uint64_t)(shapes[i][0] / 32) * HMX4_BYTES(shapes[i][1]);
        x_len += (uint64_t)shapes[i][1];
        y_len += (uint64_t)shapes[i][0];
        weights += (uint64_t)shapes[i][0] * (uint64_t)shapes[i][1];
    }
    printf("gemv: %d matrices, %.2f G weights, %.0f MB of hmx4 weights\n", count, weights / 1e9, w_len / 1048576.0);
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)w_len);
    if (!w) {
        printf("rpcmem_alloc %llu MB failed\n", (unsigned long long)(w_len >> 20));
        return 1;
    }
    uint64_t r = 0x9E3779B97F4A7C15ull;   // codes: random; scales: fp16 2^-7 times 1..2
    for (uint64_t i = 0; i + 8 <= w_len; i += 8) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        memcpy(w + i, &r, 8);
    }
    for (int i = 0; i < count; ++i)
        for (uint32_t b = 0; b < d[i].n / 32; ++b) {
            uint16_t *sc = (uint16_t *)(w + d[i].w_off + b * HMX4_BYTES(d[i].k) + (uint64_t)d[i].k * 16u);
            for (uint32_t j = 0; j < d[i].k / 16 * 32; ++j) sc[j] = (uint16_t)(0x2000u | (sc[j] & 0x3FFu));
        }
    __fp16 *x = (__fp16 *)malloc(x_len * 2);
    for (uint64_t i = 0; i < x_len; ++i) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        x[i] = (__fp16)(((double)(r >> 11) / 9007199254740992.0 * 2.0 - 1.0) * 3.0);
    }
    float *y = (float *)malloc(y_len * 4);
    const int fd = rpcmem_to_fd(w);
    int e = fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, w_len, FASTRPC_MAP_FD);
    if (e) {
        printf("fastrpc_mmap failed\n");
        rpcmem_free(w);
        return 1;
    }
    long long us = 0, ppm = 0;
    e = geode_npu_gemv(h, (const unsigned char *)d, count * (int)sizeof *d, (const unsigned char *)x, (int)(x_len * 2),
                       (unsigned char *)y, (int)(y_len * 4), fd, 4 | variant << 8, 1, &us, &ppm);
    printf("check (2048 x 2048, every 7th column): 0x%x, worst relative error %.2e\n", e, ppm / 1e6);
    for (int o = 0; o < 8; ++o) {   // the first columns against a host float sum
        double ref = 0;
        const uint8_t *blk = w + d[0].w_off;
        const uint16_t *sc = (const uint16_t *)(blk + (uint64_t)d[0].k * 16u);
        for (uint32_t i = 0; i < d[0].k; ++i) {
            const uint8_t b = blk[(i >> 1) * 32u + (uint32_t)o];
            const int code = (i & 1) ? b >> 4 : b & 15;
            union { uint16_t u; __fp16 h; } s = {sc[(i >> 4) * 32 + o]};
            ref += (double)(float)(__fp16)((float)(2 * code - 15) * (float)s.h) * (double)(float)x[i];
        }
        printf("  y[%d] = %.5f, host sum %.5f\n", o, (double)y[o], ref);
    }
    for (int t = 1; t <= max_threads && !e; ++t) {
        long long best = 0;
        for (int rep = 0; rep < 3; ++rep) {
            e = geode_npu_gemv(h, (const unsigned char *)d, count * (int)sizeof *d, (const unsigned char *)x,
                               (int)(x_len * 2), (unsigned char *)y, (int)(y_len * 4), fd, t | variant << 8, 0, &us,
                               &ppm);
            if (!e && (!best || us < best)) best = us;
        }
        if (e) printf("%d threads: 0x%x\n", t, e);
        else printf("%d threads: a step's products %.1f ms = %.1f GB/s of hmx4 weights -> %.1f tokens/s at most\n", t,
                    best / 1e3, (double)w_len / (double)best / 1e3, 1e6 / (double)best);
    }
    fastrpc_munmap(CDSP_DOMAIN_ID, fd, w, w_len);
    rpcmem_free(w);
    free(x);
    free(y);
    free(d);
    geode_npu_close(h);
    return e ? 2 : 0;
}

// npu_imp.c's gemv8_desc_t
typedef struct {
    uint64_t w_off;
    uint32_t k, n, a_at, y_at;
    float global;
    uint32_t pad;
} gemv8_desc_t;

static double e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? (1.0 + m / 8.0) * (double)(1u << e) / 128.0 : m / 8.0 / 64.0;
    return (b & 0x80) ? -v : v;
}

// One decode step's products straight from tile8 (geode_npu_gemv8): random q4 codes and E4M3 scales in tile8 (the
// GPU's layout), every matrix a token of Qwen3.5-2B (which = 2) or of the 35B-A3B's dense part (35) reads, lm_head
// included; inputs quantized per 16 as the engine's int8 path does. Checked against a double sum of the same int8
// inputs (every row of the first matrix, every 97th row of the others), then timed per thread count. The weights take
// ~1.1 GB of rpcmem: check MemAvailable first.
static int tile8_step(remote_handle64 h, int which, int max_threads, int flags) {
    int shapes[300][2], count = 0;   // (n, k)
    if (which == 1) {   // one 32 x 64 tile, to see which lanes and blocks go wrong
        shapes[count][0] = 32, shapes[count++][1] = 64;
    } else if (which == 2) {
        for (int l = 0; l < 24; ++l) {
            if (l % 4 == 3) {
                const int a[4][2] = {{4096, 2048}, {512, 2048}, {512, 2048}, {2048, 2048}};
                for (int i = 0; i < 4; ++i) shapes[count][0] = a[i][0], shapes[count++][1] = a[i][1];
            } else {
                const int g[3][2] = {{6144, 2048}, {2048, 2048}, {2048, 2048}};
                for (int i = 0; i < 3; ++i) shapes[count][0] = g[i][0], shapes[count++][1] = g[i][1];
            }
            const int m[3][2] = {{6144, 2048}, {6144, 2048}, {2048, 6144}};
            for (int i = 0; i < 3; ++i) shapes[count][0] = m[i][0], shapes[count++][1] = m[i][1];
        }
    } else {   // the 35B-A3B: 30 GDN layers (qkv, z, out), 10 attention (q with its gate, k, v, o), 40 shared experts
        for (int l = 0; l < 40; ++l) {
            if (l % 4 == 3) {
                const int a[4][2] = {{8192, 2048}, {512, 2048}, {512, 2048}, {2048, 4096}};
                for (int i = 0; i < 4; ++i) shapes[count][0] = a[i][0], shapes[count++][1] = a[i][1];
            } else {
                const int g[3][2] = {{8192, 2048}, {4096, 2048}, {2048, 4096}};
                for (int i = 0; i < 3; ++i) shapes[count][0] = g[i][0], shapes[count++][1] = g[i][1];
            }
            const int m[3][2] = {{512, 2048}, {512, 2048}, {2048, 512}};
            for (int i = 0; i < 3; ++i) shapes[count][0] = m[i][0], shapes[count++][1] = m[i][1];
        }
    }
    if (which != 1) shapes[count][0] = 248320, shapes[count++][1] = 2048;   // lm_head
    uint64_t w_len = 0, a_len = 0, y_len = 0, x_len = 0, weights = 0;
    gemv8_desc_t *d = (gemv8_desc_t *)calloc((size_t)count, sizeof *d);
    uint64_t *x_at = (uint64_t *)calloc((size_t)count, sizeof *x_at);
    for (int i = 0; i < count; ++i) {
        d[i].w_off = w_len;
        d[i].k = (uint32_t)shapes[i][1];
        d[i].n = (uint32_t)shapes[i][0];
        d[i].a_at = (uint32_t)a_len;
        d[i].y_at = (uint32_t)y_len;
        d[i].global = 1.0f / 64.0f;
        x_at[i] = x_len;
        w_len += (uint64_t)shapes[i][0] * (uint64_t)shapes[i][1] * 9u / 16u;   // 18 k bytes a 32-row tile
        a_len += (uint64_t)shapes[i][1] * 3u / 2u;
        y_len += (uint64_t)shapes[i][0];
        x_len += (uint64_t)shapes[i][1];
        weights += (uint64_t)shapes[i][0] * (uint64_t)shapes[i][1];
    }
    printf("gemv8 (%s): %d matrices, %.2f G weights, %.0f MB of tile8\n", which == 2 ? "Qwen3.5-2B" : "35B-A3B dense",
           count, weights / 1e9, w_len / 1048576.0);
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)w_len);
    if (!w) {
        printf("rpcmem_alloc %llu MB failed\n", (unsigned long long)(w_len >> 20));
        return 1;
    }
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (uint64_t i = 0; i + 8 <= w_len; i += 8) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        memcpy(w + i, &r, 8);
    }
    // the scale vectors: positive E4M3, exponents 4..9 (2^-3 .. 2^2), one byte in 64 a subnormal
    for (int i = 0; i < count; ++i) {
        const uint32_t k = d[i].k, nch = k / 64, tb = k * 18u;
        for (uint32_t t = 0; t < d[i].n / 32; ++t) {
            uint8_t *sc = w + d[i].w_off + (uint64_t)t * tb + 1024u * nch;
            for (uint32_t b = 0; b < 128u * nch; ++b) {
                r ^= r << 13, r ^= r >> 7, r ^= r << 17;
                sc[b] = (r & 63) == 0 ? (uint8_t)(1 + (r >> 8) % 7) : (uint8_t)((4 + (r >> 8) % 6) << 3 | (r >> 16) % 8);
            }
        }
    }
    float *x = (float *)malloc(x_len * 4);
    for (uint64_t i = 0; i < x_len; ++i) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        x[i] = (float)(((double)(r >> 11) / 9007199254740992.0 * 2.0 - 1.0) * 3.0);
    }
    // the inputs as the int8 path quantizes them, packed per matrix: lo / hi words, 15 x block sums, block scales
    uint8_t *act = (uint8_t *)calloc(a_len, 1);
    int8_t *q = (int8_t *)malloc(x_len);
    float *qs = (float *)malloc(x_len / 16 * 4);
    for (int i = 0; i < count; ++i) {
        const uint32_t k = d[i].k;
        const float *xi = x + x_at[i];
        int8_t *qi = q + x_at[i];
        float *si = qs + x_at[i] / 16;
        uint32_t *lo = (uint32_t *)(act + d[i].a_at), *hi = lo + k / 8;
        int32_t *ks = (int32_t *)(hi + k / 8);
        float *xs = (float *)(ks + k / 16);
        for (uint32_t g = 0; g < k / 16; ++g) {
            float mx = 0;
            for (int j = 0; j < 16; ++j) mx = fabsf(xi[16 * g + j]) > mx ? fabsf(xi[16 * g + j]) : mx;
            const float sc = mx > 0 ? mx / 127.0f : 1.0f;
            int sum = 0;
            for (int j = 0; j < 16; ++j) {
                const float v = xi[16 * g + j] / sc;
                qi[16 * g + j] = (int8_t)(v >= 0 ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
                sum += qi[16 * g + j];
            }
            ks[g] = 15 * sum;
            xs[g] = sc;
            si[g] = sc;
        }
        for (uint32_t j = 0; j < k / 8; ++j) {
            const uint8_t *v = (const uint8_t *)(qi + 8 * j);
            lo[j] = (uint32_t)v[0] | (uint32_t)v[2] << 8 | (uint32_t)v[4] << 16 | (uint32_t)v[6] << 24;
            hi[j] = (uint32_t)v[1] | (uint32_t)v[3] << 8 | (uint32_t)v[5] << 16 | (uint32_t)v[7] << 24;
        }
    }
    const int bsel = (flags >> 4) & 7;   // 1..4: only that block's inputs count (the others' scales 0)
    if (bsel)
        for (int i = 0; i < count; ++i) {
            const uint32_t k = d[i].k;
            float *xs = (float *)(act + d[i].a_at + k + k / 4);
            for (uint32_t g = 0; g < k / 16; ++g)
                if ((int)(g % 4) != bsel - 1) xs[g] = 0, qs[x_at[i] / 16 + g] = 0;
        }
    float *y = (float *)malloc((y_len + 512) * 4);   // + the debug vectors
    const int fd = rpcmem_to_fd(w);
    int e = fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, w_len, FASTRPC_MAP_FD);
    if (e) {
        printf("fastrpc_mmap failed\n");
        rpcmem_free(w);
        return 1;
    }
    long long us = 0;
    memset(y, 0, y_len * 4);
    e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)a_len, (unsigned char *)y,
                        (int)((y_len + 512) * 4), fd, 4 | flags << 8, &us);
    if (which == 1 && (flags & 0x80)) {   // the first chunk's vectors: e (quads 0-1), scales (quads 0-1), dq[0]
        const float *dv = y + 32;
        for (int l = 0; l < 32; ++l)
            printf("  lane %2d: e0 %10.4g s0 %10.4g  e1 %10.4g s1 %10.4g\n", l, dv[l], dv[128 + l], dv[32 + l],
                   dv[160 + l]);
    }
    // the check: the same int8 inputs, a double sum (the kernel's only difference: float rounding)
    double worst = 0, worst_x = 0;
    long checked = 0;
    for (int i = 0; i < count && !e; ++i) {
        const uint32_t k = d[i].k, nch = k / 64, tb = k * 18u;
        for (uint32_t o = 0; o < d[i].n; o += i == 0 ? 1 : 97) {
            const uint8_t *tile = w + d[i].w_off + (uint64_t)(o / 32) * tb;
            const uint32_t R = o % 32, qd = R / 4, rr = R % 4;
            double ref = 0, mag = 0, refx = 0;
            for (uint32_t c = 0; c < nch; ++c)
                for (uint32_t blk = 0; blk < 4; ++blk) {
                    const double ws = e4m3_value(tile[1024u * nch + (c * 8 + qd) * 16 + blk * 4 + rr]) * d[i].global;
                    const double xsc = qs[x_at[i] / 16 + c * 4 + blk];
                    for (uint32_t v = 0; v < 16; ++v) {
                        const uint32_t in = c * 64 + blk * 16 + v, j = (in % 64) / 8, b = (in % 8) / 2;
                        const uint8_t byte = tile[((c * 8 + j) * 8 + qd) * 16 + rr * 4 + b];
                        const int code = (in & 1) ? byte >> 4 : byte & 15;
                        const double wv = (2 * code - 15) * ws;
                        const double p = wv * q[x_at[i] + in] * xsc;
                        ref += p;
                        mag += fabs(p);
                        refx += wv * x[x_at[i] + in];
                    }
                }
            const double err = fabs((double)y[d[i].y_at + o] - ref) / (mag > 1e-30 ? mag : 1e-30);
            const double errx = fabs(ref - refx) / (mag > 1e-30 ? mag : 1e-30);
            if (i == 0 && o < 36 && (flags & 0x80))   // the first rows, to see which lanes go wrong
                printf("  row %2u: npu %12.5f  ref %12.5f\n", o, (double)y[d[i].y_at + o], ref);
            if (err > worst) worst = err;
            if (errx > worst_x) worst_x = errx;
            ++checked;
        }
    }
    printf("check (%ld rows): 0x%x, worst relative error against the same int8 inputs %.2e (the int8 inputs against "
           "exact ones: %.2e)\n", checked, e, worst, worst_x);
    // NPU_REPS=n: only max_threads, n steps back to back, the mean GB/s (a load beside the GPU's)
    const char *reps_env = getenv("NPU_REPS");
    if (reps_env && atoi(reps_env) > 0 && !e) {
        const int n = atoi(reps_env);
        long long sum = 0;
        for (int rep = 0; rep < n && !e; ++rep) {
            e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)a_len,
                                (unsigned char *)y, (int)((y_len + 512) * 4), fd, max_threads | flags << 8, &us);
            sum += us;
            if (rep % 50 == 49) printf("  steps %d: %.1f GB/s\n", rep + 1, (double)w_len / (double)us / 1e3);
        }
        if (!e) printf("%d threads, %d steps: mean %.2f ms = %.1f GB/s of tile8\n", max_threads, n, sum / 1e3 / n,
                       (double)w_len * n / (double)sum / 1e3);
        max_threads = 0;
    }
    for (int t = 1; t <= max_threads && !e; ++t) {
        long long best = 0;
        for (int rep = 0; rep < 3; ++rep) {
            e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)a_len,
                                (unsigned char *)y, (int)((y_len + 512) * 4), fd, t | flags << 8, &us);
            if (!e && (!best || us < best)) best = us;
        }
        if (e) printf("%d threads: 0x%x\n", t, e);
        else printf("%d threads: a step's products %.1f ms = %.1f GB/s of tile8 -> %.1f tokens/s at most\n", t,
                    best / 1e3, (double)w_len / (double)best / 1e3, 1e6 / (double)best);
    }
    fastrpc_munmap(CDSP_DOMAIN_ID, fd, w, w_len);
    rpcmem_free(w);
    free(x), free(q), free(qs), free(act), free(y), free(d), free(x_at);
    geode_npu_close(h);
    return e ? 2 : 0;
}

// ---- npu_probe t8i [layers] [Tmax] [threads] [head]: Gemma 4 E4B's decode products, q4i in tile8 -----------------
// geode_npu_gemv8 flag 4 (t8i_tile): random q4i codes (levels code - 8) and signed E4M3 scales in tile8, the matrices
// of `layers` E4B layers (every sixth a full one) and with head = 1 the lm_head, against T = 1..Tmax tokens at once
// (their activations quantized per 16 as the engine's int8 path does). Checked against a double sum of the same int8
// inputs for every T, then timed: the GB/s of tile8 read and the time per token.
static int t8i_step(remote_handle64 h, int layers, int tmax, int threads, int head) {
    int shapes[600][2], count = 0;   // (n, k)
    const int H = 2560, F = 10240, P = 256;
    for (int l = 0; l < layers && count + 9 < 600; ++l) {
        const int full = l % 6 == 5, qd = full ? 4096 : 2048, kd = full ? 1024 : 512;
        const int s[9][2] = {{qd, H}, {kd, H}, {kd, H}, {H, qd}, {F, H}, {F, H}, {H, F}, {P, H}, {H, P}};
        for (int i = 0; i < 9; ++i) shapes[count][0] = s[i][0], shapes[count++][1] = s[i][1];
    }
    if (head) shapes[count][0] = 262144, shapes[count++][1] = H;
    if (tmax < 1) tmax = 1;
    if (tmax > 4) tmax = 4;
    uint64_t w_len = 0, a_len = 0, y_len = 0, x_len = 0, weights = 0;
    gemv8_desc_t *d = (gemv8_desc_t *)calloc((size_t)count, sizeof *d);
    uint64_t *x_at = (uint64_t *)calloc((size_t)count, sizeof *x_at);
    for (int i = 0; i < count; ++i) {
        d[i].w_off = w_len;
        d[i].k = (uint32_t)shapes[i][1];
        d[i].n = (uint32_t)shapes[i][0];
        d[i].a_at = (uint32_t)a_len;
        d[i].y_at = (uint32_t)y_len;
        d[i].global = 1.0f / 64.0f;
        x_at[i] = x_len;
        w_len += (uint64_t)shapes[i][0] * (uint64_t)shapes[i][1] * 9u / 16u;
        a_len += (uint64_t)shapes[i][1] * 3u / 2u;
        y_len += (uint64_t)shapes[i][0];
        x_len += (uint64_t)shapes[i][1];
        weights += (uint64_t)shapes[i][0] * (uint64_t)shapes[i][1];
    }
    d[0].pad = (uint32_t)y_len;   // the outputs' stride a token
    printf("t8i (E4B): %d layers%s, %d matrices, %.2f G weights, %.0f MB of tile8\n", layers, head ? " + lm_head" : "",
           count, weights / 1e9, w_len / 1048576.0);
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)w_len);
    if (!w) {
        printf("rpcmem_alloc %llu MB failed\n", (unsigned long long)(w_len >> 20));
        return 1;
    }
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (uint64_t i = 0; i + 8 <= w_len; i += 8) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        memcpy(w + i, &r, 8);
    }
    // signed E4M3 scales, exponents 4..9 (2^-3 .. 2^2), one byte in 64 a subnormal
    for (int i = 0; i < count; ++i) {
        const uint32_t k = d[i].k, nch = k / 64, tb = k * 18u;
        for (uint32_t t = 0; t < d[i].n / 32; ++t) {
            uint8_t *sc = w + d[i].w_off + (uint64_t)t * tb + 1024u * nch;
            for (uint32_t b = 0; b < 128u * nch; ++b) {
                r ^= r << 13, r ^= r >> 7, r ^= r << 17;
                const uint8_t mag = (r & 63) == 0 ? (uint8_t)(1 + (r >> 8) % 7) : (uint8_t)((4 + (r >> 8) % 6) << 3 | (r >> 16) % 8);
                sc[b] = (uint8_t)(mag | ((r >> 24) & 1) << 7);
            }
        }
    }
    // T tokens' inputs, each quantized per 16: lo / hi words, 8 x block sums, block scales
    const uint64_t A = a_len;
    float *x = (float *)malloc(x_len * 4 * (size_t)tmax);
    int8_t *q = (int8_t *)malloc(x_len * (size_t)tmax);
    float *qs = (float *)malloc(x_len / 16 * 4 * (size_t)tmax);
    uint8_t *act = (uint8_t *)calloc(A * (size_t)tmax, 1);
    for (uint64_t i = 0; i < x_len * (uint64_t)tmax; ++i) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        x[i] = (float)(((double)(r >> 11) / 9007199254740992.0 * 2.0 - 1.0) * 3.0);
    }
    for (int t = 0; t < tmax; ++t)
        for (int i = 0; i < count; ++i) {
            const uint32_t k = d[i].k;
            const float *xi = x + (uint64_t)t * x_len + x_at[i];
            int8_t *qi = q + (uint64_t)t * x_len + x_at[i];
            float *si = qs + ((uint64_t)t * x_len + x_at[i]) / 16;
            uint32_t *lo = (uint32_t *)(act + (uint64_t)t * A + d[i].a_at), *hi = lo + k / 8;
            int32_t *ks = (int32_t *)(hi + k / 8);
            float *xs = (float *)(ks + k / 16);
            for (uint32_t g = 0; g < k / 16; ++g) {
                float mx = 0;
                for (int j = 0; j < 16; ++j) mx = fabsf(xi[16 * g + j]) > mx ? fabsf(xi[16 * g + j]) : mx;
                const float sc = mx > 0 ? mx / 127.0f : 1.0f;
                int sum = 0;
                for (int j = 0; j < 16; ++j) {
                    const float v = xi[16 * g + j] / sc;
                    qi[16 * g + j] = (int8_t)(v >= 0 ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
                    sum += qi[16 * g + j];
                }
                ks[g] = 8 * sum;
                xs[g] = sc;
                si[g] = sc;
            }
            for (uint32_t j = 0; j < k / 8; ++j) {
                const uint8_t *v = (const uint8_t *)(qi + 8 * j);
                lo[j] = (uint32_t)v[0] | (uint32_t)v[2] << 8 | (uint32_t)v[4] << 16 | (uint32_t)v[6] << 24;
                hi[j] = (uint32_t)v[1] | (uint32_t)v[3] << 8 | (uint32_t)v[5] << 16 | (uint32_t)v[7] << 24;
            }
        }
    float *y = (float *)malloc((y_len * (size_t)tmax + 512) * 4);
    const int fd = rpcmem_to_fd(w);
    int e = fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, w_len, FASTRPC_MAP_FD);
    if (e) {
        printf("fastrpc_mmap failed\n");
        rpcmem_free(w);
        return 1;
    }
    long long us = 0;
    // T8I_REPS=n: T = tmax only, n calls back to back at `threads`, the mean GB/s (a load beside the CPU's)
    const int reps = getenv("T8I_REPS") ? atoi(getenv("T8I_REPS")) : 0;
    if (reps > 0) {
        const int fl = 1 | 4 | (tmax - 1) << 4;
        long long sum = 0;
        for (int rep = 0; rep < reps && !e; ++rep) {
            e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)(A * (uint64_t)tmax),
                                (unsigned char *)y, (int)((y_len * (uint64_t)tmax + 512) * 4), fd, threads | fl << 8, &us);
            sum += us;
            if (rep % 100 == 99) printf("  calls %d: %.1f GB/s\n", rep + 1, (double)w_len * 100 / (double)sum / 1e3), sum = 0;
        }
        tmax = 0;
    }
    for (int T = 1; T <= tmax && !e; ++T) {
        memset(y, 0, (y_len * (size_t)tmax + 512) * 4);
        const int fl = 1 | 4 | (T - 1) << 4 | (getenv("T8I_FLAGS") ? (int)strtol(getenv("T8I_FLAGS"), NULL, 0) : 0);
        e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)(A * (uint64_t)T),
                            (unsigned char *)y, (int)((y_len * (uint64_t)T + 512) * 4), fd, threads | fl << 8, &us);
        double worst = 0, worst_x = 0;
        long checked = 0;
        for (int t = 0; t < T && !e; ++t)
            for (int i = 0; i < count; ++i) {
                const uint32_t k = d[i].k, nch = k / 64, tb = k * 18u;
                for (uint32_t o = 0; o < d[i].n; o += i == 0 ? 1 : 97) {
                    const uint8_t *tile = w + d[i].w_off + (uint64_t)(o / 32) * tb;
                    const uint32_t R = o % 32, qd = R / 4, rr = R % 4;
                    double ref = 0, mag = 0, refx = 0;
                    for (uint32_t c = 0; c < nch; ++c)
                        for (uint32_t blk = 0; blk < 4; ++blk) {
                            const double ws = e4m3_value(tile[1024u * nch + (c * 8 + qd) * 16 + blk * 4 + rr]) * d[i].global;
                            const double xsc = qs[((uint64_t)t * x_len + x_at[i]) / 16 + c * 4 + blk];
                            for (uint32_t v = 0; v < 16; ++v) {
                                const uint32_t in = c * 64 + blk * 16 + v, j = (in % 64) / 8, b = (in % 8) / 2;
                                const uint8_t byte = tile[((c * 8 + j) * 8 + qd) * 16 + rr * 4 + b];
                                const int code = (in & 1) ? byte >> 4 : byte & 15;
                                const double wv = (code - 8) * ws;
                                const double p = wv * q[(uint64_t)t * x_len + x_at[i] + in] * xsc;
                                ref += p;
                                mag += fabs(p);
                                refx += wv * x[(uint64_t)t * x_len + x_at[i] + in];
                            }
                        }
                    const double got = y[(uint64_t)t * y_len + d[i].y_at + o];
                    const double err = fabs(got - ref) / (mag > 1e-30 ? mag : 1e-30);
                    const double errx = fabs(ref - refx) / (mag > 1e-30 ? mag : 1e-30);
                    if (T == 1 && i == 0 && o < 8) printf("  row %u: npu %12.5f  ref %12.5f\n", o, got, ref);
                    if (!(err <= worst)) worst = err;
                    if (errx > worst_x) worst_x = errx;
                    ++checked;
                }
            }
        printf("T %d check (%ld rows): 0x%x, worst relative error against the same int8 inputs %.2e (int8 inputs "
               "against exact: %.2e)\n", T, checked, e, worst, worst_x);
        for (int th = threads - 1; th <= threads && !e; ++th) {
            long long best = 0;
            for (int rep = 0; rep < 4; ++rep) {
                e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)(A * (uint64_t)T),
                                    (unsigned char *)y, (int)((y_len * (uint64_t)T + 512) * 4), fd, th | fl << 8, &us);
                if (!e && (!best || us < best)) best = us;
            }
            if (e) printf("  %d threads: 0x%x\n", th, e);
            else printf("  T %d, %d threads: %.2f ms = %.1f GB/s of tile8, %.2f ms a token\n", T, th, best / 1e3,
                        (double)w_len / (double)best / 1e3, best / 1e3 / T);
        }
    }
    fastrpc_munmap(CDSP_DOMAIN_ID, fd, w, w_len);
    rpcmem_free(w);
    free(x), free(q), free(qs), free(act), free(y), free(d), free(x_at);
    geode_npu_close(h);
    return e ? 2 : 0;
}

// ---- npu_probe moe [tokens] [workers] [experts]: a prompt's routed experts of one 35B-A3B layer on HMX -------------
// (hidden 2048, ff 512, 8 picks a token) from tile8 matrices with random codes and scales: 64 distinct weight sets
// behind the experts (too many for any cache), the time a call, two tokens checked against a float reference
static double moe_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static uint32_t lcg(uint32_t *st) { return *st = *st * 1664525u + 1013904223u; }

static float e4m3_f(uint8_t u) {
    const int e = (u >> 3) & 15, m = u & 7;
    return e ? ldexpf(1.0f + (float)m / 8.0f, e - 7) : (float)m * ldexpf(1.0f, -9);
}

// value (row, input) of a tile8 matrix with k inputs (npu_imp.c's description of the layout)
static float t8_val(const uint8_t *mat, int k, int row, int in, float global) {
    const uint8_t *t = mat + (size_t)(row / 32) * 18u * (size_t)k;
    const int L = row % 32, c = in / 64, j = (in % 64) / 8, b = (in % 8) / 2, nch = k / 64;
    const uint8_t cb = t[(size_t)(c * 8 + j) * 128u + (size_t)L * 4u + (size_t)b];
    const int code = (in & 1) ? cb >> 4 : cb & 15;
    const uint8_t sb = t[(size_t)nch * 8u * 128u + (size_t)c * 128u + (size_t)(L / 4) * 16u + (size_t)((in % 64) / 16) * 4u +
                         (size_t)(L % 4)];
    return (float)(2 * code - 15) * e4m3_f(sb) * global;
}

static int moe_probe(remote_handle64 h, int n_tok, int workers, int n_ex) {
    const int H = 2048, FF = 512, K = 8, NW = 64;
    const size_t gu_bytes = (size_t)(2 * FF / 32) * 18u * H, dn_bytes = (size_t)(H / 32) * 18u * FF;
    const size_t set_bytes = gu_bytes + dn_bytes, w_len = set_bytes * NW;
    // (MOE_CACHED=1: the weights in cached rpcmem, as the engine's expert cache)
    const int cached = getenv("MOE_CACHED") && atoi(getenv("MOE_CACHED"));
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, cached ? RPCMEM_DEFAULT_FLAGS : RPCMEM_FLAG_UNCACHED,
                                         (int)w_len);
    const size_t row = (size_t)H * 4u, io_len = 2 * (size_t)n_tok * row;
    uint8_t *io = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)io_len);
    if (!w || !io) {
        printf("rpcmem_alloc failed\n");
        return 1;
    }
    uint32_t st = 12345;
    // the weights: random codes, scales 0.25 .. 1.9 (E4M3 0x28 .. 0x3F), global 1/64
    for (int ws = 0; ws < NW; ++ws) {
        uint8_t *m = w + (size_t)ws * set_bytes;
        for (int mat = 0; mat < 2; ++mat) {
            const int k = mat ? FF : H, rows = mat ? H : 2 * FF;
            uint8_t *base = mat ? m + gu_bytes : m;
            for (int t = 0; t < rows / 32; ++t) {
                uint8_t *tl = base + (size_t)t * 18u * k;
                for (size_t i = 0; i < (size_t)k * 16u; ++i) tl[i] = (uint8_t)(lcg(&st) >> 24);
                for (size_t i = 0; i < (size_t)k * 2u; ++i) tl[(size_t)k * 16u + i] = (uint8_t)(0x28 + (lcg(&st) >> 24) % 24);
            }
        }
    }
    float *x = (float *)io, *y = (float *)(io + (size_t)n_tok * row);
    for (size_t i = 0; i < (size_t)n_tok * H; ++i) x[i] = (float)(lcg(&st) >> 8) / 8388608.0f - 1.0f;
    // the picks: K distinct experts a token, weights summing to 1; grouped by expert
    int *pe = (int *)malloc((size_t)n_tok * K * sizeof(int));
    float *pw = (float *)malloc((size_t)n_tok * K * sizeof(float));
    for (int t = 0; t < n_tok; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int e, dup;
            do {
                e = (int)(lcg(&st) >> 8) % n_ex;
                dup = 0;
                for (int q = 0; q < k; ++q) dup |= pe[t * K + q] == e;
            } while (dup);
            pe[t * K + k] = e;
            sum += pw[t * K + k] = 0.1f + (float)(lcg(&st) >> 8) / 16777216.0f;
        }
        for (int k = 0; k < K; ++k) pw[t * K + k] /= sum;
    }
    const size_t desc_len = sizeof(moe_hdr_t) + 8u + (size_t)n_ex * sizeof(moe_exp_t) + (size_t)n_tok * K * sizeof(moe_pick_t);
    uint8_t *desc = (uint8_t *)calloc(1, desc_len);
    moe_hdr_t *hd = (moe_hdr_t *)desc;
    int32_t *fds = (int32_t *)(hd + 1);
    moe_exp_t *ex = (moe_exp_t *)(fds + 2);
    moe_pick_t *pk = (moe_pick_t *)(ex + n_ex);
    *hd = (moe_hdr_t){MOE_VERSION, (uint32_t)n_ex, (uint32_t)(n_tok * K), (uint32_t)H, (uint32_t)FF, (uint32_t)n_tok, 0,
                      (uint32_t)((size_t)n_tok * row), MOE_ZERO_Y | (uint32_t)(getenv("MOE_DBG") ? atoi(getenv("MOE_DBG")) : 0), (uint32_t)workers, 1, 0};
    uint32_t at = 0;
    for (int e = 0; e < n_ex; ++e) {
        ex[e].gu_off = (uint64_t)(e % NW) * set_bytes;
        ex[e].dn_off = ex[e].gu_off + gu_bytes;
        ex[e].gu_global = ex[e].dn_global = 1.0f / 64.0f;
        ex[e].first = at;
        for (int t = 0; t < n_tok; ++t)
            for (int k = 0; k < K; ++k)
                if (pe[t * K + k] == e) pk[at++] = (moe_pick_t){(uint32_t)t, pw[t * K + k]};
        ex[e].count = at - ex[e].first;
    }
    const int wfd = rpcmem_to_fd(w), iofd = rpcmem_to_fd(io);
    fds[0] = wfd;
    if (wfd < 0 || iofd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, wfd, w, 0, w_len, FASTRPC_MAP_FD) ||
        fastrpc_mmap(CDSP_DOMAIN_ID, iofd, io, 0, io_len, FASTRPC_MAP_FD)) {
        printf("fastrpc_mmap failed\n");
        return 1;
    }
    long long best = 0;
    int e = 0;
    for (int r = 0; r < 3 && !e; ++r) {
        long long us = 0;
        const double t0 = moe_now_us();
        e = geode_npu_moe(h, desc, (int)desc_len, iofd, &us);
        const double call = moe_now_us() - t0;
        if (!e) printf("moe: %d tokens, %d experts, %d workers: DSP %.2f ms (call %.2f ms)\n", n_tok, n_ex, workers,
                       us / 1e3, call / 1e3);
        if (!e && (!best || us < best)) best = us;
    }
    if (e) {
        printf("moe: 0x%x\n", e);
        return 1;
    }
    const double macs = (double)n_tok * K * 3.0 * H * FF;
    printf("best %.2f ms: %.3f TMAC/s, %.1f us an expert; a 40-layer pass %.2f s\n", best / 1e3, macs / best / 1e6,
           (double)best / n_ex, best * 40 / 1e6);
    // two tokens against floats
    double worst = 0.0, mag = 0.0;
    float *g = (float *)malloc(2 * FF * sizeof(float)), *ref = (float *)malloc(row);
    for (int ci = 0; ci < 2; ++ci) {
        const int t = ci ? n_tok - 1 : 0;
        memset(ref, 0, row);
        for (int k = 0; k < K; ++k) {
            const uint8_t *m = w + (size_t)(pe[t * K + k] % NW) * set_bytes;
            for (int r = 0; r < 2 * FF; ++r) {
                float a = 0.0f;
                for (int c = 0; c < H; ++c) a += t8_val(m, H, r, c, 1.0f / 64.0f) * (float)(__fp16)x[(size_t)t * H + c];
                g[r] = a;
            }
            for (int r = 0; r < FF; ++r) g[r] = g[r] / (1.0f + expf(-g[r])) * g[FF + r];
            for (int r = 0; r < H; ++r) {
                float a = 0.0f;
                for (int c = 0; c < FF; ++c) a += t8_val(m + gu_bytes, FF, r, c, 1.0f / 64.0f) * g[c];
                ref[r] += pw[t * K + k] * a;
            }
        }
        for (int r = 0; r < H; ++r) {
            const double d = fabs((double)ref[r] - (double)y[(size_t)t * H + r]);
            if (d > worst) worst = d;
            if (fabs(ref[r]) > mag) mag = fabs(ref[r]);
        }
        printf("token %d: y[0..3] %.4f %.4f %.4f %.4f, reference %.4f %.4f %.4f %.4f\n", t, y[(size_t)t * H],
               y[(size_t)t * H + 1], y[(size_t)t * H + 2], y[(size_t)t * H + 3], ref[0], ref[1], ref[2], ref[3]);
    }
    printf("check: worst error %.4g of the largest value %.4g (%.2e)\n", worst, mag, worst / (mag > 0 ? mag : 1));
    fastrpc_munmap(CDSP_DOMAIN_ID, wfd, w, w_len);
    fastrpc_munmap(CDSP_DOMAIN_ID, iofd, io, io_len);
    rpcmem_free(w);
    rpcmem_free(io);
    free(desc), free(pe), free(pw), free(g), free(ref);
    return worst / (mag > 0 ? mag : 1) > 2e-2;
}

// ---- a prompt's attention on HMX (npu/dsp/attn.c) against a double reference over the same int8 KV cache ---------

static double attn_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float urand(uint32_t *st) { return (float)(lcg(st) >> 8) / 8388608.0f - 1.0f; }

static int8_t *kv_at(uint8_t *const *chunks, int p, int v, int kh, int n_attn, int slot, int kv_chunk, int kv_row,
                     int hd) {
    uint8_t *base = chunks[p / kv_chunk];
    return (int8_t *)(base + ((size_t)(v * n_attn + slot) * kv_chunk + (size_t)(p % kv_chunk)) * kv_row) + kh * hd;
}
static float *kv_scale_at(uint8_t *const *chunks, int p, int v, int kh, int n_attn, int slot, int kv_chunk,
                          int kv_row, int kv_heads) {
    uint8_t *base = chunks[p / kv_chunk];
    float *scs = (float *)(base + 2u * n_attn * kv_chunk * (size_t)kv_row);
    return scs + ((size_t)(v * n_attn + slot) * kv_chunk + (size_t)(p % kv_chunk)) * kv_heads + kh;
}

// npu_probe attn n apos nh nkv hd window [flags] [reps]: random queries and an int8 KV cache in the engine's layout
// (two layers, the second one used; chunks of 256 positions, so a chunk's end falls inside), the DSP's rows against
// a double sum over the same codes and scales; the worst and the mean relative error of a row, the time.
// ATTN_QSCALE: the queries' size (the scores' spread)
static int attn_probe(remote_handle64 h, int n, int apos, int nh, int nkv, int hd, int window, int flags, int reps) {
    const int kv_chunk = 256, n_attn = 2, slot = 1, kv_heads = nkv, kv_row = nkv * hd;
    const int end = apos + n, n_chunks = (end + kv_chunk - 1) / kv_chunk;
    if (n_chunks > ATTN_MAX_CHUNKS || hd > 512 || nkv < 1 || nh % nkv) {
        printf("bad shape\n");
        return 1;
    }
    const size_t chunk_bytes = 2u * n_attn * kv_chunk * ((size_t)kv_row + (size_t)kv_heads * 4u);
    attn_desc_t d;
    memset(&d, 0, sizeof d);
    d.version = ATTN_VERSION, d.n = n, d.apos = apos, d.nh = nh, d.nkv = nkv, d.hd = hd, d.window = window;
    d.slot = slot, d.n_attn = n_attn, d.kv_chunk = kv_chunk, d.kv_row = kv_row, d.kv_heads = kv_heads;
    d.n_chunks = n_chunks, d.q_stride = nh * hd, d.o_stride = nh * hd, d.flags = flags;
    uint8_t *chunks[ATTN_MAX_CHUNKS];
    uint32_t st = 777;
    for (int c = 0; c < n_chunks; ++c) {
        chunks[c] = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)chunk_bytes);
        if (!chunks[c]) {
            printf("rpcmem_alloc failed\n");
            return 1;
        }
        memset(chunks[c], 0x55, chunk_bytes);   // (garbage where nothing is written: never to be read)
        d.kv_fd[c] = rpcmem_to_fd(chunks[c]);
        if (d.kv_fd[c] < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, d.kv_fd[c], chunks[c], 0, chunk_bytes, FASTRPC_MAP_FD)) {
            printf("fastrpc_mmap failed\n");
            return 1;
        }
    }
    // the cache: per position and KV head a float row (some dims larger, as real keys have), quantized as the
    // engine's quant_row (amax / 127)
    float row[512];
    for (int p = 0; p < end; ++p)
        for (int v = 0; v < 2; ++v)
            for (int kh = 0; kh < nkv; ++kh) {
                float amax = 0.0f;
                for (int i = 0; i < hd; ++i) {
                    row[i] = urand(&st) * (i % 17 == 0 ? 4.0f : 1.0f);
                    amax = fabsf(row[i]) > amax ? fabsf(row[i]) : amax;
                }
                const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
                int8_t *q = kv_at(chunks, p, v, kh, n_attn, slot, kv_chunk, kv_row, hd);
                for (int i = 0; i < hd; ++i) q[i] = (int8_t)lrintf(row[i] / sc);
                *kv_scale_at(chunks, p, v, kh, n_attn, slot, kv_chunk, kv_row, kv_heads) = sc;
            }
    const size_t qn = (size_t)n * d.q_stride;
    float *qb = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)(qn * 4));
    float *ob = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)(qn * 4));
    if (!qb || !ob) {
        printf("rpcmem_alloc failed\n");
        return 1;
    }
    const float qscale = getenv("ATTN_QSCALE") ? (float)atof(getenv("ATTN_QSCALE")) : 0.6f;
    for (size_t i = 0; i < qn; ++i) qb[i] = urand(&st) * qscale;
    memset(ob, 0, qn * 4);
    long long best = 0;
    int e = 0;
    for (int r = 0; r < reps && !e; ++r) {
        long long us = 0;
        const double t0 = attn_now();
        e = geode_npu_attn(h, (const unsigned char *)&d, sizeof d, (const unsigned char *)qb, (int)(qn * 4),
                           (unsigned char *)ob, (int)(qn * 4), &us);
        const double wall = attn_now() - t0;
        if (!e) printf("  call %d: DSP %.2f ms, wall %.2f ms\n", r, us / 1e3, wall * 1e3);
        if (!e && (!best || us < best)) best = us;
    }
    if (e) {
        printf("attn: 0x%x\n", e);
        return 1;
    }
    // the reference: every (token, head) in doubles
    const int group = nh / nkv;
    double worst = 0, sum_err = 0, pairs = 0;
    int worst_t = -1, worst_h = -1;
    double *w = (double *)malloc((size_t)end * sizeof(double));
    for (int t = 0; t < n; ++t)
        for (int hh = 0; hh < nh; ++hh) {
            const int kh = hh / group, hi = apos + t;
            int lo = window ? apos + t - window + 1 : 0;
            if (lo < 0) lo = 0;
            pairs += hi - lo + 1;
            const float *qr = qb + (size_t)t * d.q_stride + (size_t)hh * hd;
            double mx = -1e300;
            for (int p = lo; p <= hi; ++p) {
                const int8_t *k = kv_at(chunks, p, 0, kh, n_attn, slot, kv_chunk, kv_row, hd);
                const float ks = *kv_scale_at(chunks, p, 0, kh, n_attn, slot, kv_chunk, kv_row, kv_heads);
                double s = 0;
                for (int i = 0; i < hd; ++i) s += (double)qr[i] * k[i];
                w[p] = s * ks;
                mx = w[p] > mx ? w[p] : mx;
            }
            double den = 0;
            for (int p = lo; p <= hi; ++p) den += (w[p] = exp(w[p] - mx));
            double ref[512] = {0}, num = 0, nrm = 0;
            for (int p = lo; p <= hi; ++p) {
                const int8_t *v = kv_at(chunks, p, 1, kh, n_attn, slot, kv_chunk, kv_row, hd);
                const float vs = *kv_scale_at(chunks, p, 1, kh, n_attn, slot, kv_chunk, kv_row, kv_heads);
                for (int i = 0; i < hd; ++i) ref[i] += w[p] / den * v[i] * vs;
            }
            const float *got = ob + (size_t)t * d.o_stride + (size_t)hh * hd;
            for (int i = 0; i < hd; ++i) {
                num += (got[i] - ref[i]) * (got[i] - ref[i]);
                nrm += ref[i] * ref[i];
            }
            const double rel = sqrt(num / (nrm > 1e-30 ? nrm : 1e-30));
            if (!(rel <= worst)) worst = rel, worst_t = t, worst_h = hh;   // (a NaN is the worst)
            sum_err += rel;
        }
    free(w);
    printf("attn n %d apos %d nh %d nkv %d hd %d window %d flags 0x%x: rel err worst %.2e (token %d head %d), mean "
           "%.2e; DSP %.2f ms (%.1f G multiply-adds/s)\n", n, apos, nh, nkv, hd, window, flags, worst, worst_t, worst_h,
           sum_err / (n * nh), best / 1e3, 2.0 * pairs * hd / ((double)best * 1e3));
    for (int c = 0; c < n_chunks; ++c) {
        fastrpc_munmap(CDSP_DOMAIN_ID, d.kv_fd[c], chunks[c], chunk_bytes);
        rpcmem_free(chunks[c]);
    }
    rpcmem_free(qb);
    rpcmem_free(ob);
    return worst < 2e-2 ? 0 : 2;
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 3;
    if (argc > 1 && !strcmp(argv[1], "t8")) {   // npu_probe t8 [skel dir]: one copy (gnpu_adopt_t8) against hmx4 copies
        const char *dir = argc > 2 ? argv[2] : "/data/local/tmp/npu";
        // Gemma 4's shapes: E4B q (2560 -> 2048), ple_proj (256 -> 2560: padded to 512), down (10240 -> 2560 in 4
        // parts), E2B's down (6144 -> 1536 in 2); an MLP of each
        const int shapes[][4] = {{2048, 2560, 1024, 1}, {2560, 256, 1024, 1}, {2560, 10240, 1024, 4},
                                 {1536, 6144, 512, 2}, {256, 2560, 100, 1}};
        int bad = 0;
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
            bad += npu_t8_check(dir, shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], to_stdout, NULL) != 0;
        bad += npu_t8_mlp_check(dir, 2560, 10240, 1024, 4, to_stdout, NULL) != 0;
        bad += npu_t8_mlp_check(dir, 1536, 6144, 512, 2, to_stdout, NULL) != 0;
        printf("t8: %d bad\n", bad);
        return bad;
    }
    if (argc > 1 && !strcmp(argv[1], "f16")) {   // npu_probe f16 [skel dir]: host-made fp16 weights (the vision encoder)
        const char *dir = argc > 2 ? argv[2] : "/data/local/tmp/npu";
        // Gemma 4's vision layers at one image's 2394 patches: q k v o (768 -> 768), gate / up (768 -> 3072), down
        // (3072 -> 768); the embedder at 266 tokens (768 -> 2560); a shallow one (256 -> 512: padded)
        const int shapes[][3] = {{768, 768, 2394}, {3072, 768, 2394}, {768, 3072, 2394}, {2560, 768, 266}, {512, 256, 100}};
        int bad = 0;
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
            bad += npu_f16_check(dir, shapes[i][0], shapes[i][1], shapes[i][2], to_stdout, NULL) != 0;
        printf("f16: %d bad\n", bad);
        return bad;
    }
    if (argc > 1 && !strcmp(argv[1], "t8i")) {   // npu_probe t8i [layers] [Tmax] [threads] [head]: Gemma's q4i tile8
        remote_handle64 h = npu_open(to_stdout, NULL);
        if (!h) return 1;
        return t8i_step(h, argc > 2 ? atoi(argv[2]) : 16, argc > 3 ? atoi(argv[3]) : 4, argc > 4 ? atoi(argv[4]) : 6,
                        argc > 5 ? atoi(argv[5]) : 0);
    }
    if (argc > 1 && argv[1][0] == 't') {   // npu_probe tile8 [2|35] [threads] [flags]: decode products from tile8
        remote_handle64 h = npu_open(to_stdout, NULL);
        if (!h) return 1;
        return tile8_step(h, argc > 2 ? atoi(argv[2]) : 2, argc > 3 ? atoi(argv[3]) : 6, argc > 4 ? atoi(argv[4]) : 0);
    }
    if (argc > 1 && argv[1][0] == 'a' && strcmp(argv[1], "attn")) {   // npu_probe api: the engine's interface on pack-layout matrices
        const int shapes[][3] = {{192, 256, 100}, {2048, 2048, 512}, {2048, 256, 512}, {192, 2048, 512},
                                 {2048, 2048, 100}, {256, 256, 512}, {256, 512, 512}, {256, 1024, 512}};
        int bad = 0;
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
            bad += npu_api_check(shapes[i][0], shapes[i][1], shapes[i][2], to_stdout, NULL);
        return bad;
    }
    remote_handle64 h = npu_open(to_stdout, NULL);
    if (!h) return 1;
    if (argc > 7 && !strcmp(argv[1], "attn")) {   // npu_probe attn n apos nh nkv hd window [flags] [reps]
        const int r = attn_probe(h, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
                                 atoi(argv[7]), argc > 8 ? (int)strtol(argv[8], NULL, 0) : 0, argc > 9 ? atoi(argv[9]) : 3);
        geode_npu_close(h);
        return r;
    }
    if (argc > 1 && argv[1][0] == 'm') {   // npu_probe moe [tokens] [workers] [experts]
        const int r = moe_probe(h, argc > 2 ? atoi(argv[2]) : 4096, argc > 3 ? atoi(argv[3]) : 4,
                                argc > 4 ? atoi(argv[4]) : 256);
        geode_npu_close(h);
        return r;
    }
    if (argc > 1 && argv[1][0] == 'w') {   // npu_probe window [GB]: how much cached rpcmem the DSP maps and reads
        const int cap = argc > 2 ? atoi(argv[2]) : 4;
        const uint64_t chunk = 256ull << 20;
        uint8_t *bufs[64];
        int fds[64], n = 0;
        for (; n < 64 && (uint64_t)(n + 1) * chunk <= ((uint64_t)cap << 30); ++n) {
            bufs[n] = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)chunk);
            if (!bufs[n]) {
                printf("rpcmem_alloc failed after %d MB\n", n * 256);
                break;
            }
            memset(bufs[n] + chunk - (1u << 20), 1, 1u << 20);
            fds[n] = rpcmem_to_fd(bufs[n]);
            if (fds[n] < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fds[n], bufs[n], 0, chunk, FASTRPC_MAP_FD)) {
                printf("fastrpc_mmap failed after %d MB\n", n * 256);
                rpcmem_free(bufs[n]);
                break;
            }
            long long us = 0;
            const int e = geode_npu_bw(h, fds[n], chunk - (1u << 20), 1u << 20, 1, 0, &us);
            if (e) {
                printf("the DSP's read of the last MB failed after %d MB: 0x%x\n", (n + 1) * 256, e);
                ++n;
                break;
            }
        }
        printf("mapped and read: %d MB of cached rpcmem\n", n * 256);
        for (int i = 0; i < n; ++i) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fds[i], bufs[i], chunk);
            rpcmem_free(bufs[i]);
        }
        geode_npu_close(h);
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'b') {   // npu_probe bw [MB] [cached]: the DSP's DDR read speed over a buffer mapped once
        const uint64_t len = (uint64_t)(argc > 2 ? atoi(argv[2]) : 1024) << 20;
        const int cached = argc > 3 && atoi(argv[3]) != 0;   // rpcmem's cached flag (the hmx4 copies: uncached)
        uint8_t *buf = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, cached ? RPCMEM_DEFAULT_FLAGS : RPCMEM_FLAG_UNCACHED,
                                               (int)len);
        if (!buf) {
            printf("rpcmem_alloc %llu MB failed\n", (unsigned long long)(len >> 20));
            return 1;
        }
        memset(buf, 1, len);
        const int fd = rpcmem_to_fd(buf);
        if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf, 0, len, FASTRPC_MAP_FD)) {
            printf("fastrpc_mmap failed\n");
            rpcmem_free(buf);
            return 1;
        }
        static const char *const names[] = {"copy into VTCM", "loads only", "copy + l2fetch"};
        const int th[] = {1, 2, 3, 4, 6};
        for (int mode = 0; mode < 3; ++mode)
            for (size_t i = 0; i < sizeof th / sizeof th[0]; ++i) {
                long long best = 0;
                int e = 0;
                for (int r = 0; r < 2 && !e; ++r) {
                    long long us = 0;
                    e = geode_npu_bw(h, fd, 0, len, th[i], mode, &us);
                    if (!e && (!best || us < best)) best = us;
                }
                if (e) printf("%-15s %d threads: 0x%x\n", names[mode], th[i], e);
                else printf("%-15s %d threads: %6.1f GB/s (%llu MB in %.1f ms)\n", names[mode], th[i],
                            (double)len / (double)best / 1e3, (unsigned long long)(len >> 20), best / 1e3);
            }
        fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf, len);
        rpcmem_free(buf);
        geode_npu_close(h);
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'g') {   // npu_probe gemv [threads]: one decode step of Qwen3.5-2B's products
        return gemv_step(h, argc > 2 ? atoi(argv[2]) : 4, argc > 3 ? atoi(argv[3]) : 0);
    }
    if (argc == 3) {   // npu_probe debug PATTERN: the raw 32x32 fp16 result to stdout as text
        unsigned char buf[2048];
        int e = geode_npu_hmx_debug(h, atoi(argv[2]), buf, sizeof buf);
        if (atoi(argv[2]) >= 10 && atoi(argv[2]) < 20) {
            printf("thread probe %s: call 0x%x, create rc %d, state %d\n", argv[2], e, ((int *)buf)[0], ((int *)buf)[1]);
            geode_npu_close(h);
            return 0;
        }
        const __fp16 *v = (const __fp16 *)buf;
        printf("debug 0x%x\n", e);
        for (int i = 0; i < 1024; ++i) printf("%g%c", (double)v[i], i % 32 == 31 ? '\n' : ' ');
        geode_npu_close(h);
        return 0;
    }
    if (argc > 4) {   // one check: npu_probe 1 m k n [mode]
        long long us = 0, ppm = 0;
        const int m = atoi(argv[2]), k = atoi(argv[3]), nn = atoi(argv[4]), mode = argc > 5 ? atoi(argv[5]) : 0;
        int e = geode_npu_hmx_gemm(h, mode, m, k, nn, 1, 1, &us, &ppm);
        printf("check m %d k %d n %d mode %d: 0x%x worst rel err %.2e, %lld us (%.2f T multiply-adds/s)\n", m, k, nn,
               mode, e, ppm / 1e6, us, us > 0 ? (double)m * k * nn / (double)us * 1e-6 : 0.0);
        geode_npu_close(h);
        return 0;
    }
    const int bad = npu_bench(h, reps, to_stdout, NULL);
    geode_npu_close(h);
    return bad ? 2 : 0;
}
