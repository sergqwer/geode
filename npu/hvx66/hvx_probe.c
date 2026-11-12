// npu/hvx66/hvx_probe.c - opens the v66 module in an unsigned PD, pings it and times HVX's int8 (adb):
//     ADSP_LIBRARY_PATH=/data/local/tmp/npu ./hvx_probe
//     ADSP_LIBRARY_PATH=/data/local/tmp/npu ./hvx_probe gemm /data/local/tmp/yue2 [threads] [tokens]
//     ADSP_LIBRARY_PATH=/data/local/tmp/npu ./hvx_probe q4 [threads] [tokens] [cached] [noflush] [kchunk] [reps]
// gemm: layer 0's c8 matrices (nar_c8.bin, geode.nar_c8.json) against random activations quantized as quant_x8 does:
// the DSP's int32 sums checked bit for bit against the same arithmetic on the CPU, its products against the 8-bit
// product's exact value (fp64), and timed.
#define _GNU_SOURCE
#include <sched.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "remote.h"
#include "rpcmem.h"
#include "geode_hvx.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint64_t g_r = 0x9E3779B97F4A7C15ull;
static double frnd(void) {   // [0, 1)
    g_r ^= g_r << 13, g_r ^= g_r >> 7, g_r ^= g_r << 17;
    return (double)(g_r >> 11) / 9007199254740992.0;
}

// "name": {..."offset": N, ... "shape": [R,K]} in the manifest
static int tensor(const char *json, const char *name, long long *off, int *rows, int *k) {
    char key[128];
    snprintf(key, sizeof key, "\"%s\"", name);
    const char *p = strstr(json, key);
    if (!p) return -1;
    const char *o = strstr(p, "\"offset\":"), *s = strstr(p, "\"shape\":");
    if (!o || !s) return -1;
    *off = atoll(o + 9);
    return sscanf(s + 8, " [%d,%d]", rows, k) == 2 ? 0 : -1;
}

// vmpyo(u, h):<<1:rnd:sat on one lane (h in the word's high halfword)
static int32_t mpyo(int32_t u, int32_t h_hi) {
    const int64_t p = ((int64_t)u * (int64_t)(int16_t)(h_hi >> 16)) * 2 + 0x8000;
    const int64_t r = p >> 16;
    return r > INT32_MAX ? INT32_MAX : r < INT32_MIN ? INT32_MIN : (int32_t)r;
}

static int gemm_test(remote_handle64 h, const char *dir, int threads, int n) {
    char path[512];
    snprintf(path, sizeof path, "%s/geode.nar_c8.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return printf("no %s\n", path), 1;
    static char json[1 << 20];
    json[fread(json, 1, sizeof json - 1, f)] = 0;
    fclose(f);
    snprintf(path, sizeof path, "%s/nar_c8.bin", dir);
    FILE *wf = fopen(path, "rb");
    if (!wf) return printf("no %s\n", path), 1;
    const char *names[] = {"c8.model.layers.0.nar_qkv", "c8.model.layers.0.nar_o", "c8.model.layers.0.nar_gu",
                           "c8.model.layers.0.nar_down"};
    const int n8 = (n + 7) / 8 * 8;
    for (int m = 0; m < 4; ++m) {
        long long off = 0;
        int rows = 0, k = 0;
        if (tensor(json, names[m], &off, &rows, &k)) return printf("no %s\n", names[m]), 1;
        const int G = k / 32, nb = rows / 32;
        const size_t wlen = (size_t)nb * (size_t)k * 34;
        uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)wlen);
        if (!w) return printf("rpcmem %zu\n", wlen), 1;
        fseeko(wf, off, SEEK_SET);
        if (fread(w, 1, wlen, wf) != wlen) return printf("short read\n"), 1;
        int e = 0;
        // the activations: per group a scale over two decades (a few groups 30x louder), quantized per 32
        int8_t *x = (int8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n8 * k);
        int32_t *corr = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n8 * G * 4);
        int32_t *aq = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n8 * G * 4);
        int32_t *wq = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, rows * G * 4);
        int32_t *y = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n8 * rows * 4);
        float *as = (float *)calloc((size_t)n8 * G, 4), *asmax = (float *)calloc(n8, 4);
        float *ws = (float *)malloc((size_t)rows * G * 4), *wsmax = (float *)calloc(rows, 4);
        if (!x || !corr || !aq || !wq || !y || !as || !ws) return printf("memory\n"), 1;
        memset(x, 0, (size_t)n8 * k), memset(corr, 0, (size_t)n8 * G * 4), memset(aq, 0, (size_t)n8 * G * 4);
        for (int t = 0; t < n; ++t)
            for (int g = 0; g < G; ++g) {
                const double sc = pow(10.0, -2.0 * frnd()) * (frnd() < 0.02 ? 30.0 : 1.0);
                double v[32], amax = 0;
                for (int c = 0; c < 32; ++c) v[c] = (frnd() * 2 - 1) * sc, amax = fmax(amax, fabs(v[c]));
                const float s = (float)(amax / 127.0);
                as[t * G + g] = s;
                int sum = 0;
                for (int c = 0; c < 32; ++c) {
                    const int q = s > 0 ? (int)lrint(v[c] / s) : 0;
                    x[(size_t)t * k + g * 32 + c] = (int8_t)q;
                    sum += q;
                }
                corr[t * G + g] = 128 * sum;
                asmax[t] = fmaxf(asmax[t], s);
            }
        for (int t = 0; t < n; ++t)
            for (int g = 0; g < G; ++g)
                aq[t * G + g] = asmax[t] > 0 ? (int32_t)lrint(fmin(32767.0, as[t * G + g] / asmax[t] * 32767.0)) * 65536 : 0;
        // the weights' scales (fp16 [g][32] after each block's codes) and their ratios per row
        for (int b = 0; b < nb; ++b) {
            const __fp16 *sc = (const __fp16 *)(w + (size_t)b * k * 34 + (size_t)k * 32);
            for (int g = 0; g < G; ++g)
                for (int o = 0; o < 32; ++o) {
                    const int r = b * 32 + o;
                    ws[r * G + g] = (float)sc[g * 32 + o];
                    wsmax[r] = fmaxf(wsmax[r], fabsf(ws[r * G + g]));
                }
        }
        for (int b = 0; b < nb; ++b)
            for (int g = 0; g < G; ++g)
                for (int o = 0; o < 32; ++o) {
                    const int r = b * 32 + o;
                    wq[((size_t)b * G + g) * 32 + o] =
                        wsmax[r] > 0 ? (int32_t)lrint(fmin(32767.0, ws[r * G + g] / wsmax[r] * 32767.0)) * 65536 : 0;
                }
        // the DSP
        long long us = 0;
        const double t0 = now_ms();
        e = geode_hvx_q8_gemm(h, w, (int)wlen, rows, k, n8, (const unsigned char *)x, n8 * k, (const unsigned char *)corr,
                              n8 * G * 4, (const unsigned char *)aq, n8 * G * 4, (const unsigned char *)wq, rows * G * 4,
                              (unsigned char *)y, n8 * rows * 4, threads, &us);
        const double wall = now_ms() - t0;
        if (e) {
            printf("%s: q8_gemm 0x%x\n", names[m], e);
            return 1;
        }
        // the CPU: the same integer arithmetic (bit for bit) and the 8-bit product's exact value; a sample of rows
        long long mism = 0, checked = 0;
        double num = 0, den = 0;
        for (int r = 0; r < rows; r += (rows > 4096 ? 7 : 3)) {
            const int b = r / 32, o = r % 32;
            const uint8_t *blk = w + (size_t)b * k * 34;
            for (int t = 0; t < n; ++t) {
                int32_t acc = 0;
                double exact = 0;
                for (int g = 0; g < G; ++g) {
                    int32_t d = 0;
                    for (int c = 0; c < 32; ++c)
                        d += (int32_t)(int8_t)blk[g * 1024 + (c / 2) * 64 + o * 2 + (c & 1)] * x[(size_t)t * k + g * 32 + c];
                    acc += mpyo(mpyo(d * 4, wq[((size_t)b * G + g) * 32 + o]), aq[t * G + g]);
                    exact += (double)d * ws[r * G + g] * as[t * G + g];
                }
                ++checked;
                if (acc != y[(size_t)t * rows + r]) ++mism;
                const double got = (double)y[(size_t)t * rows + r] * wsmax[r] * asmax[t] / 4.0;
                num += (got - exact) * (got - exact), den += exact * exact;
            }
        }
        const double macs = (double)rows * k * n;
        printf("%-10s %5d x %4d, %d tokens, %d threads: DSP %.2f ms (%.1f G MAC/s), call %.2f ms; int32 mismatches "
               "%lld of %lld; rel L2 vs the exact 8-bit product %.2e\n",
               names[m] + 22, rows, k, n, threads, us / 1e3, macs / us / 1e3, wall, mism, checked, sqrt(num / den));
        rpcmem_free(w), rpcmem_free(x), rpcmem_free(corr), rpcmem_free(aq), rpcmem_free(wq), rpcmem_free(y);
        free(as), free(asmax), free(ws), free(wsmax);
    }
    fclose(wf);
    return 0;
}

// ---- q4: the 4-bit product (geode_hvx_q4_gemm) on random tile8 q4i matrices of Gemma 4 E2B's shapes, the weights at
// an offset inside one big rpcmem buffer (as the engine's), checked bit for bit against the same integer arithmetic on
// the CPU and against the CPU's int8 path (float), and timed: the DSP's time and the call's.
static float e4m3f(uint8_t b) {
    const int sign = b >> 7, ex = (b >> 3) & 15, man = b & 7;
    const float v = ex == 0 ? ldexpf((float)man / 8.0f, -6) : ldexpf(1.0f + (float)man / 8.0f, ex - 7);
    return sign ? -v : v;
}
static int16_t sat16(int64_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v; }
static int32_t sat32(int64_t v) { return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v; }

static int q4_test(remote_handle64 h, int threads, int n, int cached, int noflush, int kchunk, int reps) {
    static const int shapes[][2] = {{2048, 1536}, {1536, 2048}, {6144, 1536}, {1536, 6144}, {12288, 1536}, {1536, 12288}};
    static const char *names[] = {"q", "o", "gate", "down", "gate2", "down2"};
    const int n8 = (n + 7) / 8 * 8;
    // QUAD=1: the CPU's quad layout (the DSP turns it into tile8's order); Q4=1: Qwen's q4 (level 2c - 15) for q4i
    const int quad = getenv("QUAD") ? atoi(getenv("QUAD")) : 0, q4 = getenv("Q4") ? atoi(getenv("Q4")) : 0;
    const int kc = kchunk ? kchunk : 2048;
    // one buffer for every matrix, as the engine's (the weights of a layer and more around them)
    size_t total = 0;
    for (int m = 0; m < 6; ++m) total += (size_t)shapes[m][0] * shapes[m][1] * 18 / 32;
    const size_t pad = (size_t)(getenv("PAD_MB") ? atoi(getenv("PAD_MB")) : 64) << 20;
    uint8_t *big = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, cached ? RPCMEM_DEFAULT_FLAGS : RPCMEM_FLAG_UNCACHED,
                                           (int)(total + 2 * pad));
    if (!big) return printf("rpcmem %zu\n", total + 2 * pad), 1;
    memset(big, 0, total + 2 * pad);
    if (noflush) {
        const int fd = rpcmem_to_fd(big);
        remote_register_buf_attr(big, (int)(total + 2 * pad), fd, noflush);
        printf("registered fd %d attr %d\n", fd, noflush);
    }
    printf("%s, %s, kchunk %d\n", quad ? "quad" : "tile8", q4 ? "q4" : "q4i", kc);
    int bad = 0;
    size_t at = pad;
    for (int m = 0; m < 6; ++m) {
        const int rows = shapes[m][0], k = shapes[m][1], P = k / 32, nb = k / 16, nch = k / 64;
        const size_t tb = (size_t)k * 18, gq = (size_t)k * 9 / 4;
        uint8_t *w = big + at;
        at += (size_t)rows / 32 * tb;
        // a code byte (tile T, row l, word J of the row, byte bb) and a scale byte (block B) in either layout
#define CODEB(T, l, J, bb)                                                                                             \
    (*(quad ? &w[(size_t)(T) * tb + (size_t)((l) / 4) * gq + (((size_t)((J) % 8) * nch + (size_t)((J) / 8)) * 4 + (l) % 4) * 4 + (bb)] \
            : &w[(size_t)(T) * tb + ((size_t)(J) * 8 + (size_t)((l) / 4)) * 16 + ((l) % 4) * 4 + (bb)]))
#define SCB(T, l, B)                                                                                                   \
    (*(quad ? &w[(size_t)(T) * tb + (size_t)((l) / 4) * gq + (size_t)k * 2 + (size_t)(B) * 4 + (l) % 4]                 \
            : &w[(size_t)(T) * tb + (size_t)k * 16 + ((size_t)((B) / 4) * 8 + (size_t)((l) / 4)) * 16 + ((B) % 4) * 4 + (l) % 4]))
        for (size_t i = 0; i < (size_t)rows / 32 * tb; ++i) w[i] = (uint8_t)(frnd() * 256);
        for (int T = 0; T < rows / 32; ++T)   // E4M3 scales, exponents 3..9 (q4i signed)
            for (int l = 0; l < 32; ++l)
                for (int B = 0; B < nb; ++B) {
                    const int ex = 3 + (int)(frnd() * 7), man = (int)(frnd() * 8);
                    SCB(T, l, B) = (uint8_t)((!q4 && frnd() < 0.5 ? 0x80 : 0) | ex << 3 | man);
                }
        int8_t *xq = (int8_t *)malloc((size_t)n8 * k);
        float *as = (float *)calloc((size_t)n8 * nb, 4), *amax = (float *)calloc(n8, 4);
        int32_t *bs = (int32_t *)calloc((size_t)n8 * nb, 4);
        const int iof = getenv("UNCACHED_IO") ? RPCMEM_FLAG_UNCACHED : RPCMEM_DEFAULT_FLAGS;
        int8_t *x = (int8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, iof, n8 * k);
        int32_t *xs = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, iof, n8 * P * 8);
        int32_t *wq = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, iof, rows * P * 4);
        int32_t *y = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, iof, n8 * rows * 4);
        float *wmax = (float *)calloc(rows, 4);
        if (!xq || !as || !x || !xs || !wq || !y || !wmax) return printf("memory\n"), 1;
        memset(xq, 0, (size_t)n8 * k);
        for (int t = 0; t < n; ++t)
            for (int b = 0; b < nb; ++b) {
                const double sc = pow(10.0, -2.0 * frnd()) * (frnd() < 0.02 ? 30.0 : 1.0);
                double v[16], am = 0;
                for (int c = 0; c < 16; ++c) v[c] = (frnd() * 2 - 1) * sc, am = fmax(am, fabs(v[c]));
                const float s = (float)(am / 127.0);
                as[t * nb + b] = s;
                int sum = 0;
                for (int c = 0; c < 16; ++c) {
                    const int q = s > 0 ? (int)lrint(v[c] / s) : 0;
                    xq[(size_t)t * k + b * 16 + c] = (int8_t)q;
                    sum += q;
                }
                bs[t * nb + b] = sum;
                amax[t] = fmaxf(amax[t], s);
            }
        // the DSP's inputs (geode_hvx.idl): x permuted per 8 (0 2 4 6 1 3 5 7) and xs (-off bsum, aq Q13) pass by pass,
        // wq (Q15 of the row's largest |scale|)
        memset(x, 0, (size_t)n8 * k), memset(xs, 0, (size_t)n8 * P * 8);
        static const int perm[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        const int off = q4 ? 15 : 16;
#define XAT(t, i) ((size_t)((i) / kc * kc) * n8 + (size_t)(t) * (size_t)(k - (i) / kc * kc < kc ? k - (i) / kc * kc : kc) + (size_t)((i) % kc))
#define XSAT(t, p) (XAT(t, (p) * 32) / 16)
        for (int t = 0; t < n; ++t) {
            for (int i = 0; i < k; ++i) x[XAT(t, i)] = xq[(size_t)t * k + (i & ~7) + perm[i & 7]];
            for (int p = 0; p < P; ++p) {
                const int c0 = -off * bs[t * nb + 2 * p], c1 = -off * bs[t * nb + 2 * p + 1];
                const int a0 = amax[t] > 0 ? (int)lrint(as[t * nb + 2 * p] / amax[t] * 8192.0) : 0;
                const int a1 = amax[t] > 0 ? (int)lrint(as[t * nb + 2 * p + 1] / amax[t] * 8192.0) : 0;
                xs[XSAT(t, p)] = (int32_t)((uint32_t)(uint16_t)c0 | (uint32_t)(uint16_t)c1 << 16);
                xs[XSAT(t, p) + 1] = (int32_t)((uint32_t)(uint16_t)a0 | (uint32_t)(uint16_t)a1 << 16);
            }
        }
        for (int r = 0; r < rows; ++r) {
            float mx = 0;
            for (int b = 0; b < nb; ++b) mx = fmaxf(mx, fabsf(e4m3f(SCB(r / 32, r % 32, b))));
            wmax[r] = mx;
            for (int p = 0; p < P; ++p) {
                const int q0 = (int)lrint(fmax(-32767.0, fmin(32767.0, e4m3f(SCB(r / 32, r % 32, 2 * p)) / mx * 32767.0)));
                const int q1 = (int)lrint(fmax(-32767.0, fmin(32767.0, e4m3f(SCB(r / 32, r % 32, 2 * p + 1)) / mx * 32767.0)));
                wq[((size_t)(r / 32) * P + p) * 32 + r % 32] = (int32_t)((uint32_t)(uint16_t)q0 | (uint32_t)(uint16_t)q1 << 16);
            }
        }
        long long us = 0, best = 0;
        double wall = 0, bestwall = 1e30;
        int e = 0, am = 0;
        for (int it = 0; it < reps; ++it) {
            if (getenv("GAP_MS")) usleep(atoi(getenv("GAP_MS")) * 1000);
            const double t0 = now_ms();
            e = geode_hvx_q4_gemm(h, w, (int)((size_t)rows / 32 * tb), rows, k, n8, (const unsigned char *)x, n8 * k,
                                  (const unsigned char *)xs, n8 * P * 8, (const unsigned char *)wq, rows * P * 4,
                                  (unsigned char *)y, n8 * rows * 4, threads, kc, quad, &am, &us);
            wall = now_ms() - t0;
            if (e) return printf("%s: q4_gemm 0x%x\n", names[m], e), 1;
            if (wall < bestwall) bestwall = wall, best = us;
        }
        // the CPU: the same integer arithmetic pass by pass (each pass's sum saturated as vdmpy:sat does, the passes
        // added with saturation), and the CPU's int8 path in floats (the levels times the block scales)
        long long mism = 0, checked = 0;
        double num = 0, den = 0;
        int32_t emax = 0;
        for (int r = 0; r < rows; r += (rows > 4096 ? 13 : 5)) {
            const int T = r / 32, l = r % 32;
            for (int t = 0; t < n; ++t) {
                int32_t tot = 0;
                double ref = 0;
                for (int k0 = 0; k0 < k; k0 += kc) {
                    const int kn = k - k0 < kc ? k - k0 : kc;
                    int32_t acc = 0;
                    for (int p = k0 / 32; p < (k0 + kn) / 32; ++p) {
                        int16_t mm[2];
                        for (int blk = 0; blk < 2; ++blk) {
                            int32_t S = 0, d = 0;
                            for (int jj = 0; jj < 2; ++jj) {
                                const int J = 4 * p + 2 * blk + jj;
                                for (int bb = 0; bb < 4; ++bb) {
                                    const uint8_t byte = CODEB(T, l, J, bb);
                                    const int v0 = 8 * J + 2 * bb, lo = byte & 15, hi = byte >> 4;
                                    S += 2 * lo * xq[(size_t)t * k + v0] + 2 * hi * xq[(size_t)t * k + v0 + 1];
                                    d += (q4 ? 2 * lo - 15 : lo - 8) * xq[(size_t)t * k + v0] +
                                         (q4 ? 2 * hi - 15 : hi - 8) * xq[(size_t)t * k + v0 + 1];
                                }
                            }
                            const int b = 2 * p + blk;
                            const int16_t d16 = (int16_t)(uint16_t)(S - off * bs[t * nb + b]);
                            const uint32_t wv = (uint32_t)wq[((size_t)T * P + p) * 32 + l];
                            const int16_t wh = (int16_t)(blk ? wv >> 16 : wv & 0xFFFF);
                            mm[blk] = sat16(((int64_t)d16 * wh * 2 + 0x8000) >> 16);
                            ref += (double)d * e4m3f(SCB(T, l, b)) * as[t * nb + b];
                        }
                        const uint32_t aw = (uint32_t)xs[XSAT(t, p) + 1];
                        acc = sat32((int64_t)acc + (int64_t)mm[0] * (int16_t)(aw & 0xFFFF) + (int64_t)mm[1] * (int16_t)(aw >> 16));
                    }
                    tot = k0 ? sat32((int64_t)tot + acc) : acc;
                    emax = abs(acc) > emax ? abs(acc) : emax;
                }
                ++checked;
                const int32_t yv = y[((size_t)T * n8 + t) * 32 + l];   // (tile by tile)
                if (tot != yv) ++mism;
                const double got = (double)yv * wmax[r] * amax[t] / 8192.0 / (q4 ? 1.0 : 2.0);
                num += (got - ref) * (got - ref), den += ref * ref;
            }
        }
#undef SCB
#undef CODEB
#undef XAT
#undef XSAT
        const double macs = (double)rows * k * n;
        printf("%-6s %5d x %5d, %d tokens, %d threads: DSP %.2f ms (%.1f G MAC/s), call %.2f ms (+%.2f); int32 "
               "mismatches %lld of %lld; rel L2 vs the int8 path %.2e; amax %d (a pass's, sampled: %d)\n",
               names[m], rows, k, n, threads, best / 1e3, macs / best / 1e3, bestwall, bestwall - best / 1e3, mism,
               checked, sqrt(num / den), am, emax);
        bad |= mism != 0;
        rpcmem_free(x), rpcmem_free(xs), rpcmem_free(wq), rpcmem_free(y);
        free(xq), free(as), free(amax), free(bs), free(wmax);
    }
    rpcmem_free(big);
    return bad;
}

int main(int argc, char **argv) {
    struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
    int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof um);
    printf("unsigned PD: %s (0x%x)\n", e ? "refused" : "on", e);
    remote_handle64 h = 0;
    e = geode_hvx_open("file:///libgeode_hvx_skel_v66.so?geode_hvx_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h);
    printf("open: 0x%x\n", e);
    if (e) return 1;
    if (getenv("PROBE_CPU")) {   // (the caller's core: FastRPC's cache maintenance runs there)
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(atoi(getenv("PROBE_CPU")), &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    if (argc >= 2 && !strcmp(argv[1], "q4")) {
        // q4 [threads] [tokens] [cached weights] [remote_register_buf_attr attr, 0: none] [kchunk] [reps]
        e = q4_test(h, argc > 2 ? atoi(argv[2]) : 2, argc > 3 ? atoi(argv[3]) : 256, argc > 4 ? atoi(argv[4]) : 1,
                    argc > 5 ? atoi(argv[5]) : 0, argc > 6 ? atoi(argv[6]) : 0, argc > 7 ? atoi(argv[7]) : 3);
        geode_hvx_close(h);
        return e;
    }
    if (argc >= 3 && !strcmp(argv[1], "gemm")) {
        const int threads = argc > 3 ? atoi(argv[3]) : 2, n = argc > 4 ? atoi(argv[4]) : 106;
        for (int t = 1; t <= 2; ++t) {   // (the tile's speed in L2, the same session: a clock check)
            int units = 0, power = 0;
            long long us = 0, macs = 0;
            e = geode_hvx_bench(h, t, 2, 200, &units, &power, &us, &macs);
            printf("tile bench %d threads: 0x%x %.1f G MAC/s (power %d)\n", t, e, us > 0 ? (double)macs / us / 1e3 : 0.0, power);
        }
        e = gemm_test(h, argv[2], threads, n);
        geode_hvx_close(h);
        return e;
    }
    for (int x = 3; x < 300; x += 100) {
        int y = 0, hvx = -1, dot = 0;
        e = geode_hvx_ping(h, x, &y, &hvx, &dot);
        printf("ping %d: 0x%x y %d hvx %d dot %d (want %d)\n", x, e, y, hvx, dot, 320 * (x & 127));
    }
    for (int mode = 0; mode < 3; ++mode)
        for (int t = 1; t <= 4; t *= 2) {
            int units = 0, power = 0;
            long long us = 0, macs = 0;
            e = geode_hvx_bench(h, t, mode, mode == 2 ? 200 : mode ? 20 : 5000000, &units, &power, &us, &macs);
            printf("bench %s, %d threads: 0x%x units 0x%x power %d  %.1f ms  %.1f G MAC/s\n",
                   mode == 2 ? "product tile 128 rows x 8 tokens x 2048 (L2)" : mode ? "DDR 8 MB x 4 tokens" : "registers",
                   t, e, units, power, us / 1e3, us > 0 ? (double)macs / us / 1e3 : 0.0);
        }
    geode_hvx_close(h);
    return 0;
}
