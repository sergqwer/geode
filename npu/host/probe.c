// npu/host/probe.c - the NPU checks shared by the adb probe (npu_probe.c) and the app's debug hook (npu_app.c):
// open Geode's cDSP module in an unsigned PD, then FastRPC's and dspqueue's round trips and HMX GEMMs of the model's
// shapes. Every line goes through a callback (stdout for the probe, a buffer for the app).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <stdlib.h>

#include "remote.h"
#include "rpcmem.h"
#include "dspqueue.h"
#include "geode_npu.h"
#include "npu_api.h"
#include "probe.h"

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static void say(npu_say_fn out, void *ctx, const char *fmt, ...) {
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    out(ctx, line);
}

remote_handle64 npu_open(npu_say_fn out, void *ctx) {
    struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
    int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof um);
    say(out, ctx, "unsigned PD: %s (0x%x)\n", e == 0 ? "on" : "refused", e);
    const int arch = gnpu_arch(), skel = gnpu_skel_arch(arch);
    say(out, ctx, "Hexagon v%d (capability 0x%x): the v%d module\n", arch, gnpu_capability(ARCH_VER), skel);
    char uri[512];
    gnpu_uri(uri, sizeof uri, skel ? skel : 79);
    remote_handle64 h = 0;
    const double t0 = now_us();
    e = geode_npu_open(uri, &h);
    say(out, ctx, "open: 0x%x in %.1f ms\n", e, (now_us() - t0) / 1e3);
    return e ? 0 : h;
}

// ---- the real product (geode_npu_gemm): our buffers in, the result out, checked against a float sum ---------------
static inline uint32_t act_index(int r, int c) {   // npu_imp.c's: within a 32-row tile of K channels, fp16 elements
    return (uint32_t)(c >> 5) * 1024u + (uint32_t)(r >> 1) * 64u + (uint32_t)(c & 31) * 2u + (uint32_t)(r & 1);
}
static inline float val(uint32_t a, uint32_t b, uint32_t salt) {   // a small deterministic value in [-1, 1)
    uint32_t x = a * 2654435761u ^ (b + salt) * 2246822519u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return (float)(int32_t)(x & 0xFFFF) / 32768.0f - 1.0f;
}
// hmx4 (npu_imp.c): per 32-column block, k/2 input pairs x 32 code bytes, then k/16 groups x 32 fp16 scales
static inline float hmx4_value(const uint8_t *blk, int k, int o, int c) {
    const uint8_t b = blk[(uint32_t)(c >> 1) * 32u + (uint32_t)o];
    const int code = (c & 1) ? b >> 4 : b & 15;
    const __fp16 *sc = (const __fp16 *)(blk + (uint32_t)k * 16u);
    return (float)(__fp16)((float)(2 * code - 15) * (float)sc[(c >> 4) * 32 + o]);
}

int npu_gemm_check(remote_handle64 h, int m, int k, int n, int reps, npu_say_fn out, void *ctx) {
    const size_t q_blk = (size_t)k * 20u, n_blk = (size_t)n / 32u;
    const size_t w_len = q_blk * n_blk, a_len = (size_t)m * k * 2u, o_len = (size_t)m * n * 2u;
    // the weights: uncached (written once here, read by the DSP from DDR), mapped into the DSP once
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, w_len);
    uint8_t *a = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, a_len);
    uint8_t *o = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, o_len);
    int e = (!w || !a || !o) ? -1 : 0;
    const int w_fd = w ? rpcmem_to_fd(w) : -1;
    if (!e) e = fastrpc_mmap(CDSP_DOMAIN_ID, w_fd, w, 0, w_len, FASTRPC_MAP_FD);
    if (e) {
        say(out, ctx, "gemm %dx%dx%d: buffers 0x%x\n", m, k, n, e);
    } else {
        uint8_t *tmp = (uint8_t *)malloc(q_blk);
        for (size_t b = 0; b < n_blk; ++b) {
            for (size_t i = 0; i < (size_t)k * 16u; ++i) tmp[i] = (uint8_t)(val((uint32_t)b, (uint32_t)i, 3) * 128.0f + 128.0f);
            __fp16 *sc = (__fp16 *)(tmp + (size_t)k * 16u);
            for (int i = 0; i < k * 2; ++i) sc[i] = (__fp16)((1.5f + val((uint32_t)b, (uint32_t)i, 5)) / 256.0f);
            memcpy(w + b * q_blk, tmp, q_blk);
        }
        free(tmp);
        __fp16 *a16 = (__fp16 *)a;
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < k; ++c)
                a16[(size_t)(r >> 5) * (size_t)k * 32u + act_index(r & 31, c)] = (__fp16)val((uint32_t)r, (uint32_t)c, 1);
        long long us = 0, best = 0;
        const double t0 = now_us();
        for (int i = 0; i < reps && !e; ++i) {
            e = geode_npu_gemm(h, a, (int)a_len, w_fd, 0, o, (int)o_len, m, k, n, 0, &us);
            if (!best || us < best) best = us;
        }
        const double wall = (now_us() - t0) / reps;
        if (e) {
            say(out, ctx, "gemm %dx%dx%d: 0x%x\n", m, k, n, e);
        } else {
            // a sample of the results against a float sum of the same fp16 inputs
            const __fp16 *o16 = (const __fp16 *)o;
            double worst = 0;
            uint8_t *wcopy = (uint8_t *)malloc(q_blk);
            for (int s = 0; s < 64; ++s) {
                const int r = (int)((uint32_t)(s * 2654435761u) % (uint32_t)m), oc = (int)((uint32_t)(s * 40503u + 7u) % (uint32_t)n);
                const int b = oc / 32, oo = oc % 32;
                memcpy(wcopy, w + (size_t)b * q_blk, q_blk);   // read the uncached block once
                float ref = 0, mag = 0;
                for (int c = 0; c < k; ++c) {
                    const float p = (float)a16[(size_t)(r >> 5) * (size_t)k * 32u + act_index(r & 31, c)] * hmx4_value(wcopy, k, oo, c);
                    ref += p;
                    mag += p < 0 ? -p : p;
                }
                const float got = (float)o16[((size_t)(r >> 5) * n_blk + (size_t)b) * 1024u + (size_t)((r & 31) >> 1) * 64u +
                                             (size_t)oo * 2u + (size_t)(r & 1)];
                const double rel = (double)(got > ref ? got - ref : ref - got) / (double)(mag > 1e-6f ? mag : 1e-6f);
                if (rel > worst) worst = rel;
            }
            free(wcopy);
            const double macs = (double)m * k * n;
            say(out, ctx, "gemm %4d x %4d -> %4d: DSP %6lld us (%.2f TMAC/s), call %6.0f us (%.2f TMAC/s), worst rel err %.1e\n",
                m, k, n, best, macs / best / 1e6, wall, macs / wall / 1e6, worst);
            if (worst > 1e-3) e = -2;
        }
        fastrpc_munmap(CDSP_DOMAIN_ID, w_fd, w, w_len);
    }
    if (w) rpcmem_free(w);
    if (a) rpcmem_free(a);
    if (o) rpcmem_free(o);
    return e;
}

// the engine's interface (npu_api.c) on a matrix in the pack's q4 layout, against the CPU's dequantized product
#include "npu_api.h"
static const int8_t kQ4[16] = {1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15};
static float e4m3f(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? (float)m / 8.0f / 64.0f : (1.0f + (float)m / 8.0f) * (float)(1 << e) / 128.0f;
    return (b & 0x80) ? -v : v;
}
int npu_api_check(int rows, int cols, int n, npu_say_fn out, void *ctx) {
    char err[128];
    gnpu *g = gnpu_open("/data/local/tmp/npu", err, sizeof err);
    if (!g) {
        say(out, ctx, "api: %s\n", err);
        return -1;
    }
    const size_t rb = (size_t)cols / 2 + (size_t)cols / 16;
    uint8_t *wq = (uint8_t *)malloc(rb * (size_t)rows);
    for (int r = 0; r < rows; ++r)
        for (size_t i = 0; i < rb; ++i)
            wq[(size_t)r * rb + i] = i < (size_t)cols / 2 ? (uint8_t)(val((uint32_t)r, (uint32_t)i, 9) * 128.0f + 128.0f)
                                                          : (uint8_t)(0x30 + ((uint32_t)(r * 7 + i) % 24));   // E4M3 0.5..~4
    const float global = 0.003f;
    const int id = gnpu_adopt_q4(g, wq, rows, cols, rb, global);
    float *x = (float *)malloc(sizeof(float) * (size_t)n * cols), *y = (float *)malloc(sizeof(float) * (size_t)n * rows);
    const float **xs = (const float **)malloc(sizeof(float *) * (size_t)n);
    float **ys = (float **)malloc(sizeof(float *) * (size_t)n);
    for (int t = 0; t < n; ++t) {
        xs[t] = x + (size_t)t * cols;
        ys[t] = y + (size_t)t * rows;
        for (int c = 0; c < cols; ++c) x[(size_t)t * cols + c] = val((uint32_t)t, (uint32_t)c, 11) * 2.0f;
    }
    int e = id < 0 ? -9 : gnpu_gemm(g, id, n, xs, ys);   // warm (the buffers grow once)
    const double t0 = now_us();
    for (int i = 0; i < 5 && !e; ++i) e = gnpu_gemm(g, id, n, xs, ys);
    const double call = (now_us() - t0) / 5;
    say(out, ctx, "  api %d x %d (n %d): %.0f us a call with the conversions, %.2f TMAC/s (DSP %.0f us a call)\n", rows,
        cols, n, call, (double)rows * cols * n / call / 1e6, gnpu_dsp_seconds(g) / 6 * 1e6);
    double worst = 0;
    for (int s = 0; s < 64 && !e; ++s) {
        const int t = s * 7 % n, r = (s * 131 + 3) % rows;
        const uint8_t *row = wq + (size_t)r * rb;
        float ref = 0, mag = 0;
        for (int c = 0; c < cols; ++c) {
            const uint8_t b = row[c / 2];
            const float w = (float)kQ4[(c & 1) ? b >> 4 : b & 15] * e4m3f(row[cols / 2 + c / 16]) * global;
            ref += w * xs[t][c];
            mag += fabsf(w * xs[t][c]);
        }
        const double rel = fabs((double)ys[t][r] - ref) / (mag > 1e-9f ? mag : 1e-9f);
        if (rel > worst) worst = rel;
        if (s < 3) say(out, ctx, "  api t %d r %d: npu %.5f ref %.5f\n", t, r, ys[t][r], ref);
    }
    say(out, ctx, "api %d x %d (n %d): id %d, gemm %d, worst rel err %.2e\n", rows, cols, n, id, e, worst);
    free(wq), free(x), free(y), free(xs), free(ys);
    gnpu_close(g);
    return e || worst > 2e-3;
}

// ---- one copy (gnpu_adopt_t8): the same q4i matrices as hmx4 copies and as tile8 shared with the CPU, every result
// of both compared (they should agree to fp16's last bit or so: the same fp16 weights), and their DSP times

// a q4i matrix from the pack's rows into tile8 (engine kernels.cpp to_tile8 for Q4I: the codes as they are)
static void to_tile8_q4i(const uint8_t *src, int rows, int cols, uint8_t *dst) {
    const size_t rb = (size_t)cols / 2 + (size_t)cols / 16, nch = (size_t)cols / 64;
    for (int t0 = 0; t0 < rows; t0 += 32) {
        uint8_t *g = dst + (size_t)t0 * rb;
        for (int i = 0; i < 8; ++i)
            for (int r = 0; r < 4; ++r) {
                const uint8_t *row = src + (size_t)(t0 + 4 * i + r) * rb;
                for (size_t c = 0; c < nch; ++c) {
                    for (size_t j = 0; j < 8; ++j)
                        for (size_t k = 0; k < 4; ++k)
                            g[((c * 8 + j) * 8 + (size_t)i) * 16 + (size_t)r * 4 + k] = row[(c * 8 + j) * 4 + k];
                    for (size_t b = 0; b < 4; ++b)
                        g[16 * (size_t)cols + (c * 8 + (size_t)i) * 16 + b * 4 + (size_t)r] = row[(size_t)cols / 2 + c * 4 + b];
                }
            }
    }
}

static uint8_t *q4i_rows(int rows, int cols, uint32_t salt) {   // random codes, signed E4M3 scales ~0.5..4
    const size_t rb = (size_t)cols / 2 + (size_t)cols / 16;
    uint8_t *w = (uint8_t *)malloc(rb * (size_t)rows);
    for (int r = 0; r < rows; ++r)
        for (size_t i = 0; i < rb; ++i)
            w[(size_t)r * rb + i] = i < (size_t)cols / 2
                                        ? (uint8_t)(val((uint32_t)r, (uint32_t)i, salt) * 128.0f + 128.0f)
                                        : (uint8_t)((0x30 + ((uint32_t)(r * 7 + i * 3 + salt) % 24)) |
                                                    (((uint32_t)(r + i + salt) % 3 == 0) ? 0x80 : 0));
    return w;
}

// the worst |a - b| over the largest |b|
static double t8_diff(float *const *a, float *const *b, int n, int rows) {
    double worst = 0, big = 0;
    for (int t = 0; t < n; ++t)
        for (int r = 0; r < rows; ++r) {
            const double d = fabs((double)a[t][r] - (double)b[t][r]), m = fabs((double)b[t][r]);
            if (d > worst) worst = d;
            if (m > big) big = m;
        }
    return big > 0 ? worst / big : worst;
}

int npu_t8_check(const char *skel_dir, int rows, int cols, int n, int parts, npu_say_fn out, void *ctx) {
    char err[128];
    gnpu *g = gnpu_open(skel_dir, err, sizeof err);
    if (!g) {
        say(out, ctx, "t8: %s\n", err);
        return -1;
    }
    const size_t rb = (size_t)cols / 2 + (size_t)cols / 16, bytes = rb * (size_t)rows;
    uint8_t *wq = q4i_rows(rows, cols, 9);
    const float global = 0.0021f;
    int fd = -1;
    uint8_t *sh = (uint8_t *)gnpu_shared_alloc(g, bytes + 4096, 1, &fd);
    int bad = !sh;
    float *x = (float *)malloc(sizeof(float) * (size_t)n * cols);
    float *y4 = (float *)malloc(sizeof(float) * (size_t)n * rows), *y8 = (float *)malloc(sizeof(float) * (size_t)n * rows);
    const float **xs = (const float **)malloc(sizeof(float *) * (size_t)n);
    float **ys4 = (float **)malloc(sizeof(float *) * (size_t)n), **ys8 = (float **)malloc(sizeof(float *) * (size_t)n);
    for (int t = 0; t < n; ++t) {
        xs[t] = x + (size_t)t * cols, ys4[t] = y4 + (size_t)t * rows, ys8[t] = y8 + (size_t)t * rows;
        for (int c = 0; c < cols; ++c) x[(size_t)t * cols + c] = val((uint32_t)t, (uint32_t)c, 11) * 2.0f;
    }
    // the matrix at byte 1024 of the buffer (an offset like a piece's), adopted both ways in `parts` column parts
    const int pc = cols / parts;
    if (!bad) to_tile8_q4i(wq, rows, cols, sh + 1024);
    double worst = 0, us4 = 0, us8 = 0;
    for (int j = 0; j < parts && !bad; ++j) {
        // the part's rows in the pack's layout (its codes, then its scales: setup_npu's parts)
        const size_t prb = (size_t)pc / 2 + (size_t)pc / 16;
        uint8_t *tmp = (uint8_t *)malloc(prb * (size_t)rows);
        for (int r = 0; r < rows; ++r) {
            memcpy(tmp + (size_t)r * prb, wq + (size_t)r * rb + (size_t)(j * pc / 2), (size_t)pc / 2);
            memcpy(tmp + (size_t)r * prb + pc / 2, wq + (size_t)r * rb + (size_t)cols / 2 + (size_t)(j * pc / 16),
                   (size_t)pc / 16);
        }
        const int i4 = gnpu_adopt_q4i(g, tmp, rows, pc, prb, global);
        free(tmp);
        const int i8 = gnpu_adopt_t8(g, sh + 1024, fd, 1024, rows, cols, (int64_t)j * pc, pc, global);
        if (i4 < 0 || i8 < 0) {
            say(out, ctx, "t8 %d x %d part %d: adopt %d / %d\n", rows, cols, j, i4, i8);
            bad = 1;
            break;
        }
        const float **xp = (const float **)malloc(sizeof(float *) * (size_t)n);
        for (int t = 0; t < n; ++t) xp[t] = xs[t] + (size_t)j * pc;
        int e = gnpu_gemm(g, i4, n, xp, ys4) || gnpu_gemm(g, i8, n, xp, ys8);   // (warm)
        double d0 = gnpu_dsp_seconds(g);
        for (int r = 0; r < 3 && !e; ++r) e = gnpu_gemm(g, i4, n, xp, ys4);
        double d1 = gnpu_dsp_seconds(g);
        for (int r = 0; r < 3 && !e; ++r) e = gnpu_gemm(g, i8, n, xp, ys8);
        double d2 = gnpu_dsp_seconds(g);
        free(xp);
        if (e) {
            say(out, ctx, "t8 %d x %d part %d: gemm %d\n", rows, cols, j, e);
            bad = 1;
            break;
        }
        us4 += (d1 - d0) / 3 * 1e6, us8 += (d2 - d1) / 3 * 1e6;
        const double w = t8_diff(ys8, ys4, n, rows);
        if (w > worst) worst = w;
        if (j == 0) say(out, ctx, "  t8 t 0 r 0: hmx4 %.6f tile8 %.6f\n", ys4[0][0], ys8[0][0]);
    }
    if (!bad)
        say(out, ctx, "t8 %d x %d (n %d, %d parts): tile8 vs hmx4 worst %.2e of the largest; DSP %.0f vs %.0f us\n", rows,
            cols, n, parts, worst, us8, us4);
    free(wq), free(x), free(y4), free(y8), free(xs), free(ys4), free(ys8);
    gnpu_close(g);
    return bad || worst > 1e-3;
}

// fp16 weights the host made (gnpu_pack_f16 + gnpu_gemm_f16): a random q8 matrix (the pack's row layout) times n
// rows on HMX against a double sum of the same values; the worst difference of the largest result, the DSP's time
int npu_f16_check(const char *skel_dir, int rows, int cols, int n, npu_say_fn out, void *ctx) {
    char err[128];
    gnpu *g = gnpu_open(skel_dir, err, sizeof err);
    if (!g) {
        say(out, ctx, "f16: %s\n", err);
        return -1;
    }
    const size_t rb = (size_t)cols + (size_t)cols / 16;
    uint8_t *wq = (uint8_t *)malloc(rb * (size_t)rows);
    for (int r = 0; r < rows; ++r) {
        int8_t *c = (int8_t *)(wq + (size_t)r * rb);
        for (int i = 0; i < cols; ++i) c[i] = (int8_t)(int)(val((uint32_t)r, (uint32_t)i, 21) * 127.0f);
        for (int b = 0; b < cols / 16; ++b)   // positive and negative E4M3 scales ~0.06 .. 1
            wq[(size_t)r * rb + (size_t)cols + (size_t)b] =
                (uint8_t)(((uint32_t)(val((uint32_t)r, (uint32_t)b, 22) * 1000.0f) & 0x80) | (0x20 + ((r + 3 * b) & 31)));
    }
    const float global = 0.0137f;
    const size_t fb = gnpu_f16_bytes(rows, cols);
    int fd = -1, shift = -1;
    uint8_t *sh = (uint8_t *)gnpu_shared_alloc(g, fb + 1024, 1, &fd);
    int bad = !sh || !fb;
    const double tp = now_us();
    if (!bad) bad = gnpu_pack_f16(wq, rows, cols, rb, global, sh + 1024, &shift) != 0;
    const double pack_us = now_us() - tp;
    float *x = (float *)malloc(sizeof(float) * (size_t)n * cols), *y = (float *)malloc(sizeof(float) * (size_t)n * rows);
    const float **xs = (const float **)malloc(sizeof(float *) * (size_t)n);
    float **ys = (float **)malloc(sizeof(float *) * (size_t)n);
    for (int t = 0; t < n; ++t) {
        xs[t] = x + (size_t)t * cols, ys[t] = y + (size_t)t * rows;
        for (int c = 0; c < cols; ++c) x[(size_t)t * cols + c] = val((uint32_t)t, (uint32_t)c, 23) * 4.0f;
    }
    double us = 0, worst = 0, big = 0, wall = 0, conv = 0;
    if (!bad) {
        int e = gnpu_gemm_f16(g, fd, 1024, rows, cols, shift, n, xs, ys);   // (warm)
        const double d0 = gnpu_dsp_seconds(g), c0 = gnpu_convert_seconds(g), w0 = now_us();
        for (int r = 0; r < 3 && !e; ++r) e = gnpu_gemm_f16(g, fd, 1024, rows, cols, shift, n, xs, ys);
        wall = (now_us() - w0) / 3;
        us = (gnpu_dsp_seconds(g) - d0) / 3 * 1e6;
        conv = (gnpu_convert_seconds(g) - c0) / 3 * 1e6;
        if (e) {
            say(out, ctx, "f16 %d x %d: gemm %d\n", rows, cols, e);
            bad = 1;
        }
    }
    // the reference on a sample of the results (every 7th token, every row)
    for (int t = 0; t < n && !bad; t += 7)
        for (int r = 0; r < rows; ++r) {
            const int8_t *c = (const int8_t *)(wq + (size_t)r * rb);
            const uint8_t *sb = wq + (size_t)r * rb + (size_t)cols;
            double s = 0;
            for (int i = 0; i < cols; ++i) s += (double)c[i] * e4m3f(sb[i / 16]) * global * (double)xs[t][i];
            const double d = fabs(s - (double)ys[t][r]);
            if (fabs(s) > big) big = fabs(s);
            if (d > worst) worst = d;
        }
    // a clipped linear's clamps in the conversions (gnpu_gemm_f16c) against passes of their own around gnpu_gemm_f16:
    // the same bits
    int differ = -1;
    double wall_c = 0;
    if (!bad) {
        const float cl[4] = {-2.5f, 3.0f, (float)(-0.5 * big), (float)(0.4 * big)};
        float *xc = (float *)malloc(sizeof(float) * (size_t)n * cols), *y2 = (float *)malloc(sizeof(float) * (size_t)n * rows);
        const float **xcs = (const float **)malloc(sizeof(float *) * (size_t)n);
        float **y2s = (float **)malloc(sizeof(float *) * (size_t)n);
        for (int t = 0; t < n; ++t) {
            xcs[t] = xc + (size_t)t * cols, y2s[t] = y2 + (size_t)t * rows;
            for (int c = 0; c < cols; ++c) {
                const float v = x[(size_t)t * cols + c];
                xc[(size_t)t * cols + c] = v < cl[0] ? cl[0] : cl[1] < v ? cl[1] : v;
            }
        }
        int e = gnpu_gemm_f16(g, fd, 1024, rows, cols, shift, n, xcs, ys);
        for (size_t i = 0; i < (size_t)n * rows && !e; ++i) y[i] = y[i] < cl[2] ? cl[2] : cl[3] < y[i] ? cl[3] : y[i];
        const double w1 = now_us();
        if (!e) e = gnpu_gemm_f16c(g, fd, 1024, rows, cols, shift, n, xs, y2s, cl);
        wall_c = now_us() - w1;
        if (e) {
            say(out, ctx, "f16c %d x %d: gemm %d\n", rows, cols, e);
            bad = 1;
        } else {
            differ = 0;
            for (size_t i = 0; i < (size_t)n * rows; ++i) differ += memcmp(&y[i], &y2[i], sizeof(float)) != 0;   // (bits: a zero's sign too)
            bad = differ != 0;
        }
        free(xc), free(y2), free(xcs), free(y2s);
    }
    worst = big > 0 ? worst / big : worst;
    if (differ >= 0)
        say(out, ctx, "f16 %d x %d (n %d): worst %.2e of the largest (%.3f), shift %d; DSP %.0f us (%.2f T MAC/s), call %.0f us "
            "(conversions %.0f), pack %.1f ms; clamped in the conversions: %d differ, call %.0f us\n",
            rows, cols, n, worst, big, shift, us, us > 0 ? (double)rows * cols * n / us * 1e-6 : 0.0, wall, conv,
            pack_us * 1e-3, differ, wall_c);
    free(wq), free(x), free(y), free(xs), free(ys);
    gnpu_close(g);
    return bad || worst > 4e-3;
}

// an MLP (gnpu_mlp: gate and up in one call, down in parts) both ways
int npu_t8_mlp_check(const char *skel_dir, int H, int ff, int n, int n_down, npu_say_fn out, void *ctx) {
    char err[128];
    gnpu *g = gnpu_open(skel_dir, err, sizeof err);
    if (!g) {
        say(out, ctx, "t8 mlp: %s\n", err);
        return -1;
    }
    const size_t rbg = (size_t)H / 2 + (size_t)H / 16, rbd = (size_t)ff / 2 + (size_t)ff / 16;
    uint8_t *wg = q4i_rows(ff, H, 21), *wu = q4i_rows(ff, H, 22), *wd = q4i_rows(H, ff, 23);
    const size_t bg = rbg * (size_t)ff, bd = rbd * (size_t)H;
    int fd = -1;
    uint8_t *sh = (uint8_t *)gnpu_shared_alloc(g, 2 * bg + bd, 1, &fd);
    int bad = !sh;
    const float gl = 0.0017f;
    int ig4 = -1, iu4 = -1, ig8 = -1, iu8 = -1, d4[4], d8[4];
    if (!bad) {
        to_tile8_q4i(wg, ff, H, sh);
        to_tile8_q4i(wu, ff, H, sh + bg);
        to_tile8_q4i(wd, H, ff, sh + 2 * bg);
        ig4 = gnpu_adopt_q4i(g, wg, ff, H, rbg, gl), iu4 = gnpu_adopt_q4i(g, wu, ff, H, rbg, gl);
        ig8 = gnpu_adopt_t8(g, sh, fd, 0, ff, H, 0, H, gl), iu8 = gnpu_adopt_t8(g, sh + bg, fd, bg, ff, H, 0, H, gl);
        const int pc = ff / n_down;
        const size_t prb = (size_t)pc / 2 + (size_t)pc / 16;
        uint8_t *tmp = (uint8_t *)malloc(prb * (size_t)H);
        for (int j = 0; j < n_down; ++j) {
            for (int r = 0; r < H; ++r) {
                memcpy(tmp + (size_t)r * prb, wd + (size_t)r * rbd + (size_t)(j * pc / 2), (size_t)pc / 2);
                memcpy(tmp + (size_t)r * prb + pc / 2, wd + (size_t)r * rbd + (size_t)ff / 2 + (size_t)(j * pc / 16),
                       (size_t)pc / 16);
            }
            d4[j] = gnpu_adopt_q4i(g, tmp, H, pc, prb, gl);
            d8[j] = gnpu_adopt_t8(g, sh + 2 * bg, fd, 2 * bg, H, ff, (int64_t)j * pc, pc, gl);
            if (d4[j] < 0 || d8[j] < 0) bad = 1;
        }
        free(tmp);
        if (ig4 < 0 || iu4 < 0 || ig8 < 0 || iu8 < 0) bad = 1;
    }
    float *x = (float *)malloc(sizeof(float) * (size_t)n * H);
    float *y4 = (float *)malloc(sizeof(float) * (size_t)n * H), *y8 = (float *)malloc(sizeof(float) * (size_t)n * H);
    const float **xs = (const float **)malloc(sizeof(float *) * (size_t)n);
    float **ys4 = (float **)malloc(sizeof(float *) * (size_t)n), **ys8 = (float **)malloc(sizeof(float *) * (size_t)n);
    for (int t = 0; t < n; ++t) {
        xs[t] = x + (size_t)t * H, ys4[t] = y4 + (size_t)t * H, ys8[t] = y8 + (size_t)t * H;
        for (int c = 0; c < H; ++c) x[(size_t)t * H + c] = val((uint32_t)t, (uint32_t)c, 31) * 0.5f;
    }
    double worst = -1, us4 = 0, us8 = 0;
    if (!bad) {
        int e = gnpu_mlp(g, ig4, iu4, n_down, d4, 0, n, xs, ys4) || gnpu_mlp(g, ig8, iu8, n_down, d8, 0, n, xs, ys8);
        const double a = gnpu_dsp_seconds(g);
        for (int r = 0; r < 3 && !e; ++r) e = gnpu_mlp(g, ig4, iu4, n_down, d4, 0, n, xs, ys4);
        const double b = gnpu_dsp_seconds(g);
        for (int r = 0; r < 3 && !e; ++r) e = gnpu_mlp(g, ig8, iu8, n_down, d8, 0, n, xs, ys8);
        const double c = gnpu_dsp_seconds(g);
        if (e) {
            say(out, ctx, "t8 mlp: %d\n", e);
            bad = 1;
        } else {
            worst = t8_diff(ys8, ys4, n, H);
            us4 = (b - a) / 3 * 1e6, us8 = (c - b) / 3 * 1e6;
            say(out, ctx, "t8 mlp %d -> %d (n %d, down in %d): tile8 vs hmx4 worst %.2e of the largest (y %.4f / %.4f); "
                "DSP %.0f vs %.0f us\n", H, ff, n, n_down, worst, ys8[0][0], ys4[0][0], us8, us4);
        }
    }
    free(wg), free(wu), free(wd), free(x), free(y4), free(y8), free(xs), free(ys4), free(ys8);
    gnpu_close(g);
    return bad || worst > 1e-3;
}

int npu_bench(remote_handle64 h, int reps, npu_say_fn out, void *ctx) {
    int y = 0;
    for (int i = 0; i < 20; ++i) geode_npu_ping(h, i, &y);   // warm
    const int n = 2000;
    double t0 = now_us();
    for (int i = 0; i < n; ++i) geode_npu_ping(h, i, &y);
    say(out, ctx, "FastRPC ping, default clocks: %.1f us a round trip\n", (now_us() - t0) / n);

    int vtcm = 0, api = 0, ok = 0;
    int e = geode_npu_info(h, &vtcm, &api, &ok);   // powers the DSP up (performance mode), takes VTCM and HMX
    say(out, ctx, "info: 0x%x vtcm %d KiB, HMX lock api %d, ok %d\n", e, vtcm >> 10, api, ok);
    if (e) return e;
    t0 = now_us();
    for (int i = 0; i < n; ++i) geode_npu_ping(h, i, &y);
    say(out, ctx, "FastRPC ping, performance mode: %.1f us a round trip (last y=%d)\n", (now_us() - t0) / n, y);

    {   // dspqueue round trip: a 4-byte packet out, the answer back, blocking reads on this side
        dspqueue_t q = NULL;
        uint64_t qid = 0;
        int qe = dspqueue_create(CDSP_DOMAIN_ID, 0, 4096, 4096, NULL, NULL, NULL, &q);
        if (!qe) qe = dspqueue_export(q, &qid);
        if (!qe) qe = geode_npu_queue_start(h, qid);
        if (qe) {
            say(out, ctx, "dspqueue: setup failed 0x%x\n", qe);
        } else {
            uint32_t msg = 0, resp = 0, flags = 0, nb = 0, len = 0;
            for (int i = 0; i < 2100 && !qe; ++i) {
                if (i == 100) t0 = now_us();
                msg = (uint32_t)i;
                qe = dspqueue_write(q, 0, 0, NULL, 4, (const uint8_t *)&msg, 1000000);
                if (!qe) qe = dspqueue_read(q, &flags, 0, &nb, NULL, 4, &len, (uint8_t *)&resp, 1000000);
            }
            say(out, ctx, "dspqueue round trip: %.1f us (last answer %u, err 0x%x)\n", (now_us() - t0) / 2000, resp, qe);
            geode_npu_queue_stop(h);
        }
        if (q) dspqueue_close(q);
    }
    // the checks first, small, then the shapes: Qwen3.5-2B (hidden 2048, mlp 6144), 35B-A3B (GDN qkv 8192, out 4096,
    // expert gate_up 1024 x 2048), 512 tokens
    struct { int mode, m, k, n, check; const char *what; } t[] = {
        {0, 64, 2048, 64, 1, "check, k 2048"},
        {0, 32, 4096, 32, 1, "check, k 4096"},
        {1, 64, 2048, 64, 1, "check, streamed"},
        {2, 64, 2048, 64, 1, "check, q4 dequant"},
        {3, 64, 2048, 256, 1, "check, q4 pipelined"},
        {3, 512, 2048, 2048, 0, "512 x 2048 -> 2048 (q4 pipelined)"},
        {3, 512, 4096, 2048, 0, "512 x 4096 -> 2048 out_proj (q4 pipelined)"},
        {3, 512, 2048, 8192, 0, "512 x 2048 -> 8192 GDN qkv (q4 pipelined)"},
        {3, 256, 2048, 1024, 0, "256 x 2048 -> 1024 expert gate_up (q4 pipelined)"},
        {0, 512, 2048, 2048, 0, "512 x 2048 -> 2048 (VTCM weights)"},
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof t / sizeof t[0]; ++i) {
        long long us = 0, ppm = 0;
        e = geode_npu_hmx_gemm(h, t[i].mode, t[i].m, t[i].k, t[i].n, t[i].check ? 1 : reps, t[i].check, &us, &ppm);
        const double macs = (double)t[i].m * t[i].k * t[i].n * (t[i].check ? 1 : reps);
        if (e) {
            say(out, ctx, "%-44s error 0x%x\n", t[i].what, e);
            ++bad;
        } else if (t[i].check) {
            say(out, ctx, "%-44s worst rel err %.2e (%.0f us)\n", t[i].what, ppm / 1e6, (double)us);
            if (ppm > 1000) ++bad;   // 1e-3 of the products' sum: far above fp16's own rounding
        } else {
            say(out, ctx, "%-44s %8.0f us  %.3f TMAC/s\n", t[i].what, (double)us / reps, macs / us / 1e6);
        }
    }
    // the real product on Qwen3.5-2B's shapes (512-token prompt chunks): qkv/z/out, attention q, mlp gate|up, down
    const int shapes[][3] = {{512, 2048, 6144}, {512, 2048, 2048}, {512, 2048, 4096}, {512, 2048, 6144},
                             {512, 6144, 2048}, {128, 2048, 2048}};
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
        if (npu_gemm_check(h, shapes[i][0], shapes[i][1], shapes[i][2], reps, out, ctx)) ++bad;
    return bad;
}
