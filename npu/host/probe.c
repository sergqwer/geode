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
