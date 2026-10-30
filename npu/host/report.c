// npu/host/report.c - the device report's NPU check (gnpu_report, npu_api.h): what the phone's Hexagon cDSP is and
// whether Geode's module runs there, every step guarded, timed and checked against a CPU reference, as JSON. As with
// the GPU check (engine gpu_check.cpp), each step's name goes into step_file before it runs: a DSP fault that takes
// the process down leaves it there for the app to report. No model is needed.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "remote.h"
#include "rpcmem.h"
#include "dspqueue.h"
#include "geode_npu.h"
#include "npu_api.h"

typedef struct {
    char *b;
    int len, cap, checks;
} Json;

static void put(Json *j, const char *fmt, ...) {
    if (j->len >= j->cap - 1) return;
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(j->b + j->len, (size_t)(j->cap - j->len), fmt, ap);
    va_end(ap);
    if (n > 0) j->len = j->len + n < j->cap - 1 ? j->len + n : j->cap - 1;
}

static void put_str(Json *j, const char *s) {   // a JSON string, escaped
    put(j, "\"");
    for (; s && *s; ++s) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') put(j, "\\%c", c);
        else if (c < 0x20) put(j, "\\u%04x", c);
        else put(j, "%c", c);
    }
    put(j, "\"");
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static const char *g_step_file;
static void step(const char *name) {
    if (!g_step_file || !*g_step_file) return;
    FILE *f = fopen(g_step_file, "w");
    if (!f) return;
    fprintf(f, "npu: %s\n", name);
    fclose(f);
}

// one check: name, ok, then the numbers given as "key", value pairs (NAN: left out), the error text if any
static void check(Json *j, const char *name, int ok, const char *error, int nums, ...) {
    put(j, "%s{\"name\":", j->checks++ ? "," : "");
    put_str(j, name);
    put(j, ",\"ok\":%s", ok ? "true" : "false");
    va_list ap;
    va_start(ap, nums);
    for (int i = 0; i < nums; ++i) {
        const char *key = va_arg(ap, const char *);
        const double v = va_arg(ap, double);
        if (!isnan(v) && !isinf(v)) put(j, ",\"%s\":%.6g", key, v);
    }
    va_end(ap);
    if (error && *error) {
        put(j, ",\"error\":");
        put_str(j, error);
    }
    put(j, "}");
}

static uint64_t g_r = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    g_r ^= g_r << 13, g_r ^= g_r >> 7, g_r ^= g_r << 17;
    return g_r;
}
static float frnd(void) {   // [-1, 1)
    return (float)((double)(rnd() >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

static double e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? (1.0 + m / 8.0) * (double)(1u << e) / 128.0 : m / 8.0 / 64.0;
    return (b & 0x80) ? -v : v;
}

// the engine's own product (gnpu_adopt_q4 + gnpu_gemm) on a random q4 matrix, against a double sum
static void engine_product(Json *j, const char *skel_dir, int rows, int cols, int n) {
    char name[96], err[160] = {0};
    snprintf(name, sizeof name, "engine product %d x %d (%d tokens)", rows, cols, n);
    step(name);
    gnpu *g = gnpu_open(skel_dir, err, sizeof err);
    if (!g) {
        check(j, name, 0, err, 0);
        return;
    }
    const size_t rb = (size_t)cols / 2 + (size_t)cols / 16;
    uint8_t *wq = (uint8_t *)malloc(rb * (size_t)rows);
    float *x = (float *)malloc(sizeof(float) * (size_t)n * cols), *y = (float *)malloc(sizeof(float) * (size_t)n * rows);
    const float **xs = (const float **)malloc(sizeof(float *) * (size_t)n);
    float **ys = (float **)malloc(sizeof(float *) * (size_t)n);
    if (!wq || !x || !y || !xs || !ys) {
        check(j, name, 0, "out of memory", 0);
        goto done;
    }
    for (int r = 0; r < rows; ++r)
        for (size_t i = 0; i < rb; ++i)
            wq[(size_t)r * rb + i] = i < (size_t)cols / 2 ? (uint8_t)rnd() : (uint8_t)(0x30 + rnd() % 24);   // E4M3 ~0.5..4
    const float global = 0.003f;
    for (int t = 0; t < n; ++t) {
        xs[t] = x + (size_t)t * cols;
        ys[t] = y + (size_t)t * rows;
        for (int c = 0; c < cols; ++c) x[(size_t)t * cols + c] = frnd() * 2.0f;
    }
    const int id = gnpu_adopt_q4(g, wq, rows, cols, rb, global);
    int e = id < 0 ? -9 : gnpu_gemm(g, id, n, xs, ys);   // warm: the buffers grow once
    const double t0 = now_ms();
    for (int i = 0; i < 3 && !e; ++i) e = gnpu_gemm(g, id, n, xs, ys);
    const double call = (now_ms() - t0) / 3;
    double worst = 0;
    static const int8_t q4[16] = {1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15};
    for (int s = 0; s < 64 && !e; ++s) {
        const int t = s * 7 % n, r = (s * 131 + 3) % rows;
        const uint8_t *row = wq + (size_t)r * rb;
        double ref = 0, mag = 0;
        for (int c = 0; c < cols; ++c) {
            const uint8_t b = row[c / 2];
            const double w = q4[(c & 1) ? b >> 4 : b & 15] * e4m3_value(row[cols / 2 + c / 16]) * global;
            ref += w * xs[t][c];
            mag += fabs(w * xs[t][c]);
        }
        const double rel = fabs((double)ys[t][r] - ref) / (mag > 1e-9 ? mag : 1e-9);
        if (rel > worst) worst = rel;
    }
    char msg[64] = {0};
    if (e) snprintf(msg, sizeof msg, "error 0x%x", e);
    check(j, name, !e && worst < 2e-3, msg, 3, "ms", e ? NAN : call, "tmacs",
          e ? NAN : (double)rows * cols * n / call / 1e9, "err", e ? NAN : worst);
done:
    free(wq), free(x), free(y), free(xs), free(ys);
    gnpu_close(g);
}

// npu_imp.c's gemv8_desc_t
typedef struct {
    uint64_t w_off;
    uint32_t k, n, a_at, y_at;
    float global;
    uint32_t pad;
} gemv8_desc_t;

// a decode step's products straight from tile8 with the DMA (geode_npu_gemv8): count random k x n matrices, the first
// checked row by row against a double sum of the same int8 inputs, then timed
static void tile8_gemv(Json *j, remote_handle64 h, int count, int threads) {
    const uint32_t k = 2048, n = 2048, nch = k / 64, tb = k * 18u;
    char name[96];
    snprintf(name, sizeof name, "decode products from tile8, DMA, %d threads", threads);
    step(name);
    const uint64_t one = (uint64_t)n / 32u * tb, w_len = one * (uint64_t)count;
    const uint32_t a_one = k * 3u / 2u;
    gemv8_desc_t *d = (gemv8_desc_t *)calloc((size_t)count, sizeof *d);
    uint8_t *act = (uint8_t *)calloc((size_t)a_one * count, 1);
    float *y = (float *)calloc((size_t)n * count + 512, 4);
    int8_t *q = (int8_t *)malloc(k);
    float *qs = (float *)malloc(k / 16 * 4);
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)w_len);
    if (!d || !act || !y || !q || !qs || !w) {
        check(j, name, 0, "out of memory", 0);
        goto done;
    }
    for (uint64_t i = 0; i + 8 <= w_len; i += 8) {
        const uint64_t v = rnd();
        memcpy(w + i, &v, 8);
    }
    for (int m = 0; m < count; ++m) {
        d[m] = (gemv8_desc_t){one * (uint64_t)m, k, n, a_one * (uint32_t)m, n * (uint32_t)m, 1.0f / 64.0f, 0};
        for (uint32_t t = 0; t < n / 32; ++t) {   // the scales: positive E4M3, 2^-3 .. 2^2
            uint8_t *sc = w + d[m].w_off + (uint64_t)t * tb + 1024u * nch;
            for (uint32_t b = 0; b < 128u * nch; ++b) {
                const uint64_t r = rnd();
                sc[b] = (uint8_t)((4 + (r >> 8) % 6) << 3 | (r >> 16) % 8);
            }
        }
    }
    // one token's inputs, quantized per 16 as the engine's int8 path does, the same for every matrix
    for (uint32_t g = 0; g < k / 16; ++g) {
        float v[16], mx = 0;
        for (int i = 0; i < 16; ++i) {
            v[i] = frnd() * 3.0f;
            mx = fabsf(v[i]) > mx ? fabsf(v[i]) : mx;
        }
        const float sc = mx > 0 ? mx / 127.0f : 1.0f;
        for (int i = 0; i < 16; ++i) {
            const float t = v[i] / sc;
            q[16 * g + i] = (int8_t)(t >= 0 ? (int)(t + 0.5f) : -(int)(-t + 0.5f));
        }
        qs[g] = sc;
    }
    for (int m = 0; m < count; ++m) {
        uint32_t *lo = (uint32_t *)(act + d[m].a_at), *hi = lo + k / 8;
        int32_t *ks = (int32_t *)(hi + k / 8);
        float *xs = (float *)(ks + k / 16);
        for (uint32_t g = 0; g < k / 16; ++g) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += q[16 * g + i];
            ks[g] = 15 * sum;
            xs[g] = qs[g];
        }
        for (uint32_t i = 0; i < k / 8; ++i) {
            const uint8_t *v = (const uint8_t *)(q + 8 * i);
            lo[i] = (uint32_t)v[0] | (uint32_t)v[2] << 8 | (uint32_t)v[4] << 16 | (uint32_t)v[6] << 24;
            hi[i] = (uint32_t)v[1] | (uint32_t)v[3] << 8 | (uint32_t)v[5] << 16 | (uint32_t)v[7] << 24;
        }
    }
    const int fd = rpcmem_to_fd(w);
    int e = fd < 0 ? -1 : fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, w_len, FASTRPC_MAP_FD);
    if (e) {
        check(j, name, 0, "fastrpc_mmap failed", 0);
        goto done;
    }
    long long us = 0, best = 0;
    const int y_len = (int)((n * (uint32_t)count + 512) * 4);
    e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)a_one * count,
                        (unsigned char *)y, y_len, fd, threads | 1 << 8, &us);
    double worst = 0;
    for (uint32_t o = 0; o < n && !e; o += 3) {   // the first matrix, every third row
        const uint8_t *tile = w + (uint64_t)(o / 32) * tb;
        const uint32_t R = o % 32, qd = R / 4, rr = R % 4;
        double ref = 0, mag = 0;
        for (uint32_t c = 0; c < nch; ++c)
            for (uint32_t blk = 0; blk < 4; ++blk) {
                const double ws = e4m3_value(tile[1024u * nch + (c * 8 + qd) * 16 + blk * 4 + rr]) / 64.0;
                for (uint32_t v = 0; v < 16; ++v) {
                    const uint32_t in = c * 64 + blk * 16 + v, jw = (in % 64) / 8, b = (in % 8) / 2;
                    const uint8_t byte = tile[((c * 8 + jw) * 8 + qd) * 16 + rr * 4 + b];
                    const int code = (in & 1) ? byte >> 4 : byte & 15;
                    const double p = (2 * code - 15) * ws * q[in] * qs[c * 4 + blk];
                    ref += p;
                    mag += fabs(p);
                }
            }
        const double err = fabs((double)y[o] - ref) / (mag > 1e-30 ? mag : 1e-30);
        if (err > worst) worst = err;
    }
    for (int rep = 0; rep < 3 && !e; ++rep) {
        e = geode_npu_gemv8(h, (const unsigned char *)d, count * (int)sizeof *d, act, (int)a_one * count,
                            (unsigned char *)y, y_len, fd, threads | 1 << 8, &us);
        if (!e && (!best || us < best)) best = us;
    }
    fastrpc_munmap(CDSP_DOMAIN_ID, fd, w, w_len);
    char msg[64] = {0};
    if (e) snprintf(msg, sizeof msg, "error 0x%x", e);
    check(j, name, !e && worst < 1e-5, msg, 3, "ms", e ? NAN : best / 1e3, "gbs",
          e ? NAN : (double)w_len / (double)best / 1e3, "err", e ? NAN : worst);
done:
    if (w) rpcmem_free(w);
    free(d), free(act), free(y), free(q), free(qs);
}

// the DSP's DDR read speed: a 128 MB buffer the CPU mapped once, copied into VTCM by `threads` HVX threads
static void ddr_read(Json *j, remote_handle64 h, int threads) {
    char name[64];
    snprintf(name, sizeof name, "DDR read, %d HVX threads", threads);
    step(name);
    const uint64_t len = 128ull << 20;
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)len);
    if (!w) {
        check(j, name, 0, "out of memory", 0);
        return;
    }
    memset(w, 1, (size_t)len);
    const int fd = rpcmem_to_fd(w);
    int e = fd < 0 ? -1 : fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, len, FASTRPC_MAP_FD);
    long long us = 0, best = 0;
    for (int rep = 0; rep < 3 && !e; ++rep) {
        e = geode_npu_bw(h, fd, 0, len, threads, 0, &us);
        if (!e && (!best || us < best)) best = us;
    }
    if (fd >= 0) fastrpc_munmap(CDSP_DOMAIN_ID, fd, w, len);
    rpcmem_free(w);
    char msg[64] = {0};
    if (e) snprintf(msg, sizeof msg, "error 0x%x", e);
    check(j, name, !e, msg, 1, "gbs", e ? NAN : (double)len / (double)best / 1e3);
}

int gnpu_report(const char *skel_dir, const char *step_file, char *out, int out_len) {
    Json j = {out, 0, out_len, 0};
    if (!out || out_len < 256) return -1;
    out[0] = 0;
    g_step_file = step_file;
    gnpu_library_path(skel_dir);

    step("capabilities");
    const unsigned dom = gnpu_capability(DOMAIN_SUPPORT), upd = gnpu_capability(UNSIGNED_PD_SUPPORT);
    const unsigned arch_raw = gnpu_capability(ARCH_VER), hvx = gnpu_capability(HVX_SUPPORT_128B);
    const unsigned hmx_d = gnpu_capability(HMX_SUPPORT_DEPTH), hmx_s = gnpu_capability(HMX_SUPPORT_SPATIAL);
    const unsigned vp = gnpu_capability(VTCM_PAGE), vc = gnpu_capability(VTCM_COUNT);
    const int arch = gnpu_arch(), skel = gnpu_skel_arch(arch);
    put(&j, "{\"arch\":%d,\"arch_raw\":\"0x%x\",\"skel\":%d,\"cdsp\":%d,\"unsigned_pd_support\":%d,\"hvx128\":%d,"
            "\"hmx_depth\":%d,\"hmx_spatial\":%d,\"vtcm_page\":%u,\"vtcm_count\":%d,",
        arch, arch_raw, skel, (int)dom, (int)upd, (int)hvx, (int)hmx_d, (int)hmx_s, vp, (int)vc);
    char dev[160];
    // (the HMX attributes read 0 on the 8 Elite though its HMX works: reported as FastRPC gives them, not judged)
    char module[32];
    if (skel) snprintf(module, sizeof module, "the v%d module", skel);
    else snprintf(module, sizeof module, "no module for it");
    snprintf(dev, sizeof dev, "Hexagon v%d, HVX 128B x %d, HMX attributes %d/%d, VTCM %u x %u KiB, %s", arch,
             (int)hvx, (int)hmx_d, (int)hmx_s, vc, vp >> 10, module);
    put(&j, "\"device\":");
    put_str(&j, dev);
    put(&j, ",\"checks\":[");

    char reason[200] = {0};
    remote_handle64 h = 0;
    // DOMAIN_SUPPORT reads 0 on the 8 Elite though its cDSP runs our module (as the HMX attributes do): only a cDSP
    // that answers nothing at all (no architecture either) counts as missing; otherwise opening the session decides
    if ((dom == 0 || dom == 0xFFFFFFFFu) && arch_raw == 0xFFFFFFFFu) {
        snprintf(reason, sizeof reason, "no cDSP (FastRPC: 0x%x)", dom);
    } else if (!skel) {
        snprintf(reason, sizeof reason, "Hexagon v%d: older than v73, no HMX module for it", arch);
    } else {
        step("unsigned PD");
        struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
        int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof um);
        char msg[64] = {0};
        if (e) snprintf(msg, sizeof msg, "refused 0x%x", e);
        check(&j, "unsigned PD", !e, msg, 0);
        if (e) {
            snprintf(reason, sizeof reason, "unsigned PD refused (0x%x)", e);
        } else {
            step("open the module");
            char uri[512];
            gnpu_uri(uri, sizeof uri, skel);
            const double t0 = now_ms();
            e = geode_npu_open(uri, &h);
            msg[0] = 0;
            if (e) snprintf(msg, sizeof msg, "0x%x", e);
            check(&j, "open the module", !e, msg, 1, "ms", now_ms() - t0);
            if (e) {
                h = 0;
                snprintf(reason, sizeof reason, "the v%d module did not open (0x%x)", skel, e);
            }
        }
    }
    if (h) {
        step("VTCM and HMX");
        int vtcm = 0, api = 0, ok = 0;
        int e = geode_npu_info(h, &vtcm, &api, &ok);
        char msg[64] = {0};
        if (e || !ok) snprintf(msg, sizeof msg, "0x%x, ok %d", e, ok);
        check(&j, "VTCM and HMX", !e && ok, msg, 2, "vtcm_kib", (double)(vtcm >> 10), "lock_api", (double)api);
        if (e || !ok) {
            snprintf(reason, sizeof reason, "no VTCM / HMX (0x%x)", e);
        } else {
            step("FastRPC round trip");
            int y = 0;
            for (int i = 0; i < 20; ++i) geode_npu_ping(h, i, &y);
            double t0 = now_ms();
            for (int i = 0; i < 500 && !e; ++i) e = geode_npu_ping(h, i, &y);
            check(&j, "FastRPC round trip", !e && y == 500, NULL, 1, "us", (now_ms() - t0) * 1e3 / 500);

            step("dspqueue round trip");
            dspqueue_t qu = NULL;
            uint64_t qid = 0;
            int qe = dspqueue_create(CDSP_DOMAIN_ID, 0, 4096, 4096, NULL, NULL, NULL, &qu);
            if (!qe) qe = dspqueue_export(qu, &qid);
            if (!qe) qe = geode_npu_queue_start(h, qid);
            double qus = NAN;
            if (!qe) {
                uint32_t m = 0, resp = 0, flags = 0, nb = 0, len = 0;
                for (int i = 0; i < 550 && !qe; ++i) {
                    if (i == 50) t0 = now_ms();
                    m = (uint32_t)i;
                    qe = dspqueue_write(qu, 0, 0, NULL, 4, (const uint8_t *)&m, 1000000);
                    if (!qe) qe = dspqueue_read(qu, &flags, 0, &nb, NULL, 4, &len, (uint8_t *)&resp, 1000000);
                }
                if (!qe) qus = (now_ms() - t0) * 1e3 / 500;
                geode_npu_queue_stop(h);
            }
            if (qu) dspqueue_close(qu);
            msg[0] = 0;
            if (qe) snprintf(msg, sizeof msg, "0x%x", qe);
            check(&j, "dspqueue round trip", !qe, msg, 1, "us", qus);

            // HMX: checked against a float sum, then timed on the model's shapes (q4 weights streamed and expanded)
            step("HMX check");
            long long us = 0, ppm = 0;
            e = geode_npu_hmx_gemm(h, 3, 64, 2048, 256, 1, 1, &us, &ppm);
            msg[0] = 0;
            if (e) snprintf(msg, sizeof msg, "0x%x", e);
            check(&j, "HMX fp16 product against the CPU", !e && ppm <= 1000, msg, 1, "err", e ? NAN : ppm / 1e6);
            if (e || ppm > 1000) snprintf(reason, sizeof reason, "the HMX check failed (0x%x, %.1e)", e, ppm / 1e6);
            const int shapes[][3] = {{512, 2048, 2048}, {512, 2048, 8192}, {512, 4096, 2048}, {256, 2048, 1024}};
            for (size_t i = 0; i < sizeof shapes / sizeof shapes[0] && !e && ppm <= 1000; ++i) {
                char name[64];
                snprintf(name, sizeof name, "HMX %d x %d -> %d", shapes[i][0], shapes[i][1], shapes[i][2]);
                step(name);
                const int reps = 3;
                long long t_us = 0, p2 = 0;
                const int e2 = geode_npu_hmx_gemm(h, 3, shapes[i][0], shapes[i][1], shapes[i][2], reps, 0, &t_us, &p2);
                msg[0] = 0;
                if (e2) snprintf(msg, sizeof msg, "0x%x", e2);
                const double macs = (double)shapes[i][0] * shapes[i][1] * shapes[i][2] * reps;
                check(&j, name, !e2, msg, 2, "ms", e2 ? NAN : t_us / 1e3 / reps, "tmacs", e2 ? NAN : macs / t_us / 1e6);
            }
            ddr_read(&j, h, 4);
            if (arch >= 79) tile8_gemv(&j, h, 48, 6);
        }
    }
    const int opened = h != 0;
    if (h) geode_npu_close(h);
    if (!reason[0] && opened) engine_product(&j, skel_dir, 2048, 2048, 128);   // the engine's interface, its own session
    put(&j, "],\"path\":\"%s\",\"reason\":", reason[0] ? "off" : "npu");
    put_str(&j, reason);
    put(&j, "}");
    if (step_file && *step_file) remove(step_file);
    return 0;
}
