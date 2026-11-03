// npu/host/npu_api.c - libgeode_npu.so's interface for the engine (npu_api.h): the pack's q4 matrices turned into
// hmx4 blocks in rpcmem the DSP maps once, and products whose activations go over as fp16 row tiles and come back as
// result croutons (npu/dsp/npu_imp.c, geode_npu_gemm).
#include <math.h>
#include <stdio.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#include "remote.h"
#include "rpcmem.h"
#include "geode_npu.h"
#include "npu_api.h"

typedef struct {
    uint8_t *w;      // hmx4 blocks, rows / 32 of cols * 20 bytes
    size_t len;
    int fd;
    int64_t rows, cols, kp;   // kp: the depth the NPU multiplies (cols, padded to 512 at least)
    int shift;       // the fp16 scales are 2^shift larger; HMX's convert takes it back out
    int q4i;         // the codes are Gemma 4's q4i (level = c - 8): the gemm says so (GNPU_SHIFT_Q4I)
} Mat;

// a gemm's shift with this bit: the matrix's codes are q4i's (npu_imp.c HMX4_Q4I)
#define GNPU_SHIFT_Q4I 0x100

typedef struct {   // a gnpu_shared_alloc buffer
    void *p;
    size_t len;
    int fd;
} Shared;

struct gnpu {
    remote_handle64 h;
    Mat *mats;
    int n_mats, cap;
    Shared *shared;
    int n_shared, shared_cap, dec;
    uint8_t *a, *o[4];   // the activations' row tiles, each matrix's result croutons (rpcmem: FastRPC passes them
                         // without copies)
    uint8_t *mid;        // gnpu_mlp: act(gate) * up as the DSP leaves it, the down parts' input tiles
    size_t a_cap, o_cap[4], mid_cap, bytes;
    double dsp_s, conv_s;
    gnpu_par_fn par;     // the engine's threads for the conversions (gnpu_set_par)
    void *par_ctx;
};

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float e4m3(uint8_t b) {   // E4M3FN: bias 7, subnormal below exponent 1
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? (float)m / 8.0f * ldexpf(1.0f, -6) : (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
    return (b & 0x80) ? -v : v;
}

static inline uint32_t act_index(int r, int c) {   // npu_imp.c's: within a 32-row tile of K channels, fp16 elements
    return (uint32_t)(c >> 5) * 1024u + (uint32_t)(r >> 1) * 64u + (uint32_t)(c & 31) * 2u + (uint32_t)(r & 1);
}

unsigned gnpu_capability(int attribute) {
    struct remote_dsp_capability c = {CDSP_DOMAIN_ID, (uint32_t)attribute, 0};
    return remote_handle_control(DSPRPC_GET_DSP_INFO, &c, sizeof c) ? 0xFFFFFFFFu : c.capability;
}

int gnpu_arch(void) {
    const unsigned c = gnpu_capability(ARCH_VER);
    if (c == 0xFFFFFFFFu) return 0;
    const unsigned v = c & 0xFFu, hi = v >> 4, lo = v & 15u;   // 0x79 -> 79 (the low byte, in hex digits)
    return hi < 10 && lo < 10 ? (int)(hi * 10 + lo) : 0;
}

int gnpu_skel_arch(int arch) {
    static const int built[] = {73, 75, 79, 81};   // npu/build.py's ARCHS
    if (arch <= 0) return 79;                      // FastRPC did not say: the one we know
    if (arch < built[0]) return 0;                 // older than v73: no HMX module for it
    int best = built[0];
    for (size_t i = 0; i < sizeof built / sizeof built[0]; ++i)
        if (built[i] <= arch) best = built[i];
    return best;
}

int gnpu_uri(char *uri, int len, int skel_arch) {
    return snprintf(uri, (size_t)len,
                    "file:///libgeode_npu_skel_v%d.so?geode_npu_skel_handle_invoke&_modver=1.0&_dom=cdsp", skel_arch);
}

void gnpu_library_path(const char *skel_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp",
             skel_dir ? skel_dir : ".");
    setenv("ADSP_LIBRARY_PATH", path, 1);
}

gnpu *gnpu_open(const char *skel_dir, char *err, int err_len) {
    gnpu_library_path(skel_dir);
    const int arch = gnpu_arch(), skel = gnpu_skel_arch(arch);
    if (!skel) {
        snprintf(err, (size_t)err_len, "Hexagon v%d: no HMX (v73 or newer needed)", arch);
        return NULL;
    }
    struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
    int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof um);
    if (e) {
        snprintf(err, (size_t)err_len, "no unsigned PD (0x%x)", e);
        return NULL;
    }
    char uri[512];
    gnpu_uri(uri, sizeof uri, skel);
    remote_handle64 h = 0;
    if ((e = geode_npu_open(uri, &h))) {
        snprintf(err, (size_t)err_len, "the v%d module did not open (0x%x, Hexagon v%d)", skel, e, arch);
        return NULL;
    }
    int vtcm = 0, api = 0, ok = 0;
    if ((e = geode_npu_info(h, &vtcm, &api, &ok)) || !ok) {
        snprintf(err, (size_t)err_len, "no VTCM / HMX (0x%x)", e);
        geode_npu_close(h);
        return NULL;
    }
    // HMX checked before anything relies on it: a 64 x 2048 x 256 q4 product against a float sum of the same inputs
    // (a version whose HMX behaves otherwise than v79's must not give the engine wrong products)
    long long us = 0, ppm = 0;
    if ((e = geode_npu_hmx_gemm(h, 3, 64, 2048, 256, 1, 1, &us, &ppm)) || ppm > 1000) {
        snprintf(err, (size_t)err_len, "the HMX check failed (0x%x, worst error %.1e, Hexagon v%d)", e, ppm / 1e6, arch);
        geode_npu_close(h);
        return NULL;
    }
    gnpu *g = (gnpu *)calloc(1, sizeof *g);
    g->h = h;
    return g;
}

void gnpu_close(gnpu *g) {
    if (!g) return;
    if (g->dec) geode_npu_dec_close(g->h);
    for (int i = 0; i < g->n_shared; ++i) {
        fastrpc_munmap(CDSP_DOMAIN_ID, g->shared[i].fd, g->shared[i].p, g->shared[i].len);
        rpcmem_free(g->shared[i].p);
    }
    free(g->shared);
    for (int i = 0; i < g->n_mats; ++i) {
        fastrpc_munmap(CDSP_DOMAIN_ID, g->mats[i].fd, g->mats[i].w, g->mats[i].len);
        rpcmem_free(g->mats[i].w);
    }
    free(g->mats);
    if (g->a) rpcmem_free(g->a);
    if (g->mid) rpcmem_free(g->mid);
    for (int j = 0; j < 4; ++j)
        if (g->o[j]) rpcmem_free(g->o[j]);
    geode_npu_close(g->h);
    free(g);
}

static int adopt(gnpu *g, const uint8_t *data, int64_t rows, int64_t cols, size_t row_bytes, float global, int q4i) {
    if (!g || rows <= 0 || cols <= 0 || (rows & 31) || (cols & 31) || row_bytes != (size_t)(cols / 2 + cols / 16))
        return -1;
    // the largest scale: the 2^shift that brings it to [0.5, 1) keeps every fp16 scale away from underflow
    float big = 0.0f;
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t *sb = data + (size_t)r * row_bytes + (size_t)(cols / 2);
        for (int64_t b = 0; b < cols / 16; ++b) {
            const float v = fabsf(e4m3(sb[b]) * global);
            if (v > big) big = v;
        }
    }
    int ex = 0;
    frexpf(big > 0.0f ? big : 1.0f, &ex);
    const int shift = -ex < 0 ? 0 : -ex > 24 ? 24 : -ex;
    const float up = ldexpf(global, shift);
    // a depth under 16 croutons (k < 512) comes out wrong on v79's HMX (k = 256 measured): the inputs padded with
    // zero-scale groups, the activations with zeros
    const int64_t kp = cols < 512 ? 512 : cols;
    const size_t q_blk = (size_t)kp * 20u, n_blk = (size_t)rows / 32u, len = q_blk * n_blk;
    uint8_t *w = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, len);
    if (!w) return -1;
    uint8_t *blk = (uint8_t *)calloc(1, q_blk);
    for (size_t b = 0; b < n_blk; ++b) {
        __fp16 *sc = (__fp16 *)(blk + (size_t)kp * 16u);
        for (int o = 0; o < 32; ++o) {
            const uint8_t *row = data + (b * 32u + (size_t)o) * row_bytes;
            for (int64_t p = 0; p < cols / 2; ++p) {   // the pack's byte p: values 2p (low), 2p + 1 (high)
                if (q4i) {   // q4i's codes as they are (level = c - 8 on the DSP too)
                    blk[(size_t)p * 32u + (size_t)o] = row[p];
                    continue;
                }
                const uint8_t c0 = row[p] & 15, c1 = row[p] >> 4;
                const uint8_t a0 = c0 < 8 ? c0 + 8 : 15 - c0, a1 = c1 < 8 ? c1 + 8 : 15 - c1;   // level = 2a - 15
                blk[(size_t)p * 32u + (size_t)o] = (uint8_t)(a0 | a1 << 4);
            }
            const uint8_t *sb = row + cols / 2;
            for (int64_t gi = 0; gi < cols / 16; ++gi) sc[gi * 32 + o] = (__fp16)(e4m3(sb[gi]) * up);
        }
        memcpy(w + b * q_blk, blk, q_blk);
    }
    free(blk);
    const int fd = rpcmem_to_fd(w);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, len, FASTRPC_MAP_FD)) {
        rpcmem_free(w);
        return -1;
    }
    if (g->n_mats == g->cap) {
        g->cap = g->cap ? 2 * g->cap : 64;
        g->mats = (Mat *)realloc(g->mats, (size_t)g->cap * sizeof *g->mats);
    }
    g->mats[g->n_mats] = (Mat){w, len, fd, rows, cols, kp, shift, q4i};
    g->bytes += len;
    return g->n_mats++;
}

int gnpu_adopt_q4(gnpu *g, const uint8_t *data, int64_t rows, int64_t cols, size_t row_bytes, float global) {
    return adopt(g, data, rows, cols, row_bytes, global, 0);
}

int gnpu_adopt_q4i(gnpu *g, const uint8_t *data, int64_t rows, int64_t cols, size_t row_bytes, float global) {
    return adopt(g, data, rows, cols, row_bytes, global, 1);
}

static int grow(uint8_t **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*buf) rpcmem_free(*buf);
    *buf = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, need);
    *cap = *buf ? need : 0;
    return *buf ? 0 : -1;
}

// ---- the conversions: work items the engine's threads share (gnpu_set_par)

static void run_par(gnpu *g, int items, void (*fn)(void *, int, int), void *arg) {
    if (g->par && items > 1) g->par(g->par_ctx, items, fn, arg);
    else fn(arg, 0, items);
}

// the activations into HMX's row tiles in its own order: per tile, per 32 channels, per row pair, 64 fp16 in a row -
// (r, c) at (c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + r % 2 - so every write is sequential; an item: one
// tile's 32 channels
typedef struct {
    const float *const *xs;
    __fp16 *a16;
    int n, k, kc;
} InJob;

static void in_items(void *arg, int i0, int i1) {
    const InJob *j = (const InJob *)arg;
    static const float zeros[32] = {0};
    const int kt = j->k / 32;
    for (int it = i0; it < i1; ++it) {
        const int t = it / kt, cc = it % kt;
        __fp16 *dst = j->a16 + ((size_t)t * (size_t)kt + (size_t)cc) * 1024u;
        for (int rp = 0; rp < 16; ++rp, dst += 64) {
            const int r0 = t * 32 + 2 * rp, r1 = r0 + 1, c0 = cc * 32;
            const float *x0 = r0 < j->n && c0 < j->kc ? j->xs[r0] + c0 : zeros;
            const float *x1 = r1 < j->n && c0 < j->kc ? j->xs[r1] + c0 : zeros;
#if defined(__ARM_NEON)
            for (int q = 0; q < 32; q += 4) {
                float16x4x2_t h;   // r0, r1 interleaved channel by channel
                h.val[0] = vcvt_f16_f32(vld1q_f32(x0 + q));
                h.val[1] = vcvt_f16_f32(vld1q_f32(x1 + q));
                vst2_f16(dst + 2 * q, h);
            }
#else
            for (int q = 0; q < 32; ++q) {
                dst[2 * q] = (__fp16)x0[q];
                dst[2 * q + 1] = (__fp16)x1[q];
            }
#endif
        }
    }
}

// the result croutons read in order: each row pair's 64 fp16 split back into its two rows' 32 columns; an item: one
// tile's 32 columns
typedef struct {
    float *const *ys;
    const __fp16 *o16;
    int n, nn;
    int add;   // added into ys (gnpu_mlp's down parts after the first)
} OutJob;

static void out_items(void *arg, int i0, int i1) {
    const OutJob *j = (const OutJob *)arg;
    const int n_blk = j->nn / 32;
    for (int it = i0; it < i1; ++it) {
        const int t = it / n_blk, b = it % n_blk;
        const __fp16 *src = j->o16 + ((size_t)t * (size_t)n_blk + (size_t)b) * 1024u;
        for (int rp = 0; rp < 16; ++rp, src += 64) {
            const int r0 = t * 32 + 2 * rp, r1 = r0 + 1;
            if (r0 >= j->n) break;
            float *y0 = j->ys[r0] + b * 32, *y1 = r1 < j->n ? j->ys[r1] + b * 32 : NULL;
#if defined(__ARM_NEON)
            if (j->add)
                for (int q = 0; q < 32; q += 4) {
                    const float16x4x2_t v = vld2_f16(src + 2 * q);
                    vst1q_f32(y0 + q, vaddq_f32(vld1q_f32(y0 + q), vcvt_f32_f16(v.val[0])));
                    if (y1) vst1q_f32(y1 + q, vaddq_f32(vld1q_f32(y1 + q), vcvt_f32_f16(v.val[1])));
                }
            else
                for (int q = 0; q < 32; q += 4) {
                    const float16x4x2_t v = vld2_f16(src + 2 * q);
                    vst1q_f32(y0 + q, vcvt_f32_f16(v.val[0]));
                    if (y1) vst1q_f32(y1 + q, vcvt_f32_f16(v.val[1]));
                }
#else
            for (int q = 0; q < 32; ++q) {
                y0[q] = (j->add ? y0[q] : 0.0f) + (float)src[2 * q];
                if (y1) y1[q] = (j->add ? y1[q] : 0.0f) + (float)src[2 * q + 1];
            }
#endif
        }
    }
}

// The DSP's calls stay on the caller's thread (the module's HMX and VTCM are the opening thread's: a call from
// another thread fails); the results are converted on a thread of its own, through the engine's threads, while the DSP
// multiplies the next matrix
typedef struct {
    gnpu *g;
    const int *ids;
    float *const *const *ys;
    int count, m, n;
    int add;                 // the results after the first added into theirs (gnpu_mlp)
    int called, converted;   // matrices the DSP finished, matrices converted
    int stop;
    double conv_s;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} Conv;

static void *conv_thread(void *arg) {
    Conv *c = (Conv *)arg;
    for (int j = 0; j < c->count; ++j) {
        pthread_mutex_lock(&c->mu);
        while (c->called <= j && !c->stop) pthread_cond_wait(&c->cv, &c->mu);
        const int go = c->called > j;
        pthread_mutex_unlock(&c->mu);
        if (!go) break;
        const double t1 = now_s();
        OutJob out = {c->ys[j], (const __fp16 *)c->g->o[j], c->n, (int)c->g->mats[c->ids[j]].rows, c->add && j > 0};
        run_par(c->g, (c->m / 32) * (out.nn / 32), out_items, &out);
        pthread_mutex_lock(&c->mu);
        c->conv_s += now_s() - t1;
        c->converted = j + 1;
        pthread_cond_broadcast(&c->cv);
        pthread_mutex_unlock(&c->mu);
    }
    return NULL;
}

int gnpu_gemm_multi(gnpu *g, int count, const int *ids, int n, const float *const *xs, float *const *const *ys) {
    if (!g || count <= 0 || count > 4 || n <= 0) return -1;
    for (int j = 0; j < count; ++j)
        if (ids[j] < 0 || ids[j] >= g->n_mats || g->mats[ids[j]].kp != g->mats[ids[0]].kp ||
            g->mats[ids[j]].cols != g->mats[ids[0]].cols)
            return -1;
    const Mat *m0 = &g->mats[ids[0]];
    const int k = (int)m0->kp, kc = (int)m0->cols, m = (n + 31) & ~31;
    if (grow(&g->a, &g->a_cap, (size_t)m * k * 2u)) return -1;
    for (int j = 0; j < count; ++j)
        if (grow(&g->o[j], &g->o_cap[j], (size_t)m * (size_t)g->mats[ids[j]].rows * 2u)) return -1;
    const double t0 = now_s();
    InJob in = {xs, (__fp16 *)g->a, n, k, kc};
    run_par(g, (m / 32) * (k / 32), in_items, &in);
    double conv = now_s() - t0;
    Conv c = {g, ids, ys, count, m, n, 0, 0, 0, 0, 0.0};
    pthread_mutex_init(&c.mu, NULL);
    pthread_cond_init(&c.cv, NULL);
    pthread_t th;
    const int threaded = count > 1 && pthread_create(&th, NULL, conv_thread, &c) == 0;
    int err = 0;
    long long us_all = 0;
    for (int j = 0; j < count && !err; ++j) {
        const Mat *mt = &g->mats[ids[j]];
        long long us = 0;
        err = geode_npu_gemm(g->h, g->a, m * k * 2, mt->fd, 0, g->o[j], m * (int)mt->rows * 2, m, k, (int)mt->rows,
                             mt->shift | (mt->q4i ? GNPU_SHIFT_Q4I : 0), &us);
        us_all += us;
        if (err) break;
        if (threaded) {
            pthread_mutex_lock(&c.mu);
            c.called = j + 1;
            pthread_cond_broadcast(&c.cv);
            pthread_mutex_unlock(&c.mu);
        } else {
            const double t1 = now_s();
            OutJob out = {ys[j], (const __fp16 *)g->o[j], n, (int)mt->rows};
            run_par(g, (m / 32) * (out.nn / 32), out_items, &out);
            conv += now_s() - t1;
        }
    }
    if (threaded) {
        pthread_mutex_lock(&c.mu);
        c.stop = 1;
        pthread_cond_broadcast(&c.cv);
        pthread_mutex_unlock(&c.mu);
        pthread_join(th, NULL);
        conv += c.conv_s;
    }
    pthread_mutex_destroy(&c.mu);
    pthread_cond_destroy(&c.cv);
    g->dsp_s += (double)us_all * 1e-6;
    g->conv_s += conv;
    return err;
}

// An MLP (npu_api.h gnpu_mlp): gate and up in one DSP call that leaves act(gate) * up in fp16 as down's input tiles
// (geode_npu_gemm_glu, parts-major), then each part of down from its stretch of them - its results converted (the
// first) or added (the rest) while the DSP multiplies the next, as gnpu_gemm_multi's
int gnpu_mlp(gnpu *g, int gate, int up, int n_down, const int *downs, int act, int n, const float *const *xs,
             float *const *ys) {
    if (!g || n <= 0 || n_down < 1 || n_down > 4 || gate < 0 || gate >= g->n_mats || up < 0 || up >= g->n_mats)
        return -1;
    const Mat *mg = &g->mats[gate], *mu = &g->mats[up];
    if (mg->rows != mu->rows || mg->cols != mu->cols || mg->kp != mu->kp || mg->q4i != mu->q4i) return -1;
    const int64_t ff = mg->rows, pc = ff / n_down;
    if (pc * n_down != ff || pc % 32) return -1;
    int64_t H = -1;
    for (int j = 0; j < n_down; ++j) {
        if (downs[j] < 0 || downs[j] >= g->n_mats) return -1;
        const Mat *md = &g->mats[downs[j]];
        // (a part's depth is its inputs as they are: no padding, so the tiles of act(gate) * up are its input)
        if (md->cols != pc || md->kp != pc || md->q4i != mg->q4i || (H >= 0 && md->rows != H)) return -1;
        H = md->rows;
    }
    const int k = (int)mg->kp, kc = (int)mg->cols, m = (n + 31) & ~31;
    if (grow(&g->a, &g->a_cap, (size_t)m * k * 2u) || grow(&g->mid, &g->mid_cap, (size_t)m * (size_t)ff * 2u))
        return -1;
    for (int j = 0; j < n_down; ++j)
        if (grow(&g->o[j], &g->o_cap[j], (size_t)m * (size_t)H * 2u)) return -1;
    const double t0 = now_s();
    InJob in = {xs, (__fp16 *)g->a, n, k, kc};
    run_par(g, (m / 32) * (k / 32), in_items, &in);
    double conv = now_s() - t0;
    const int q4 = mg->q4i ? GNPU_SHIFT_Q4I : 0;
    long long us = 0, us_all = 0;
    int err = geode_npu_gemm_glu(g->h, g->a, m * k * 2, mg->fd, mu->fd, g->mid, (int)((size_t)m * (size_t)ff * 2u), m,
                                 k, (int)ff, mg->shift | q4, mu->shift | q4, (int)(pc / 32), act, &us);
    us_all += us;
    if (!err) {
        float *const *yl[4] = {ys, ys, ys, ys};
        Conv c = {g, downs, yl, n_down, m, n, 1, 0, 0, 0, 0.0};
        pthread_mutex_init(&c.mu, NULL);
        pthread_cond_init(&c.cv, NULL);
        pthread_t th;
        const int threaded = n_down > 1 && pthread_create(&th, NULL, conv_thread, &c) == 0;
        const size_t part_bytes = (size_t)m * (size_t)pc * 2u;
        for (int j = 0; j < n_down && !err; ++j) {
            const Mat *md = &g->mats[downs[j]];
            err = geode_npu_gemm(g->h, g->mid + (size_t)j * part_bytes, (int)part_bytes, md->fd, 0, g->o[j],
                                 (int)((size_t)m * (size_t)H * 2u), m, (int)pc, (int)H, md->shift | q4, &us);
            us_all += us;
            if (err) break;
            if (threaded) {
                pthread_mutex_lock(&c.mu);
                c.called = j + 1;
                pthread_cond_broadcast(&c.cv);
                pthread_mutex_unlock(&c.mu);
            } else {
                const double t1 = now_s();
                OutJob out = {ys, (const __fp16 *)g->o[j], n, (int)H, j > 0};
                run_par(g, (m / 32) * (int)(H / 32), out_items, &out);
                conv += now_s() - t1;
            }
        }
        if (threaded) {
            pthread_mutex_lock(&c.mu);
            c.stop = 1;
            pthread_cond_broadcast(&c.cv);
            pthread_mutex_unlock(&c.mu);
            pthread_join(th, NULL);
            conv += c.conv_s;
        }
        pthread_mutex_destroy(&c.mu);
        pthread_cond_destroy(&c.cv);
    }
    g->dsp_s += (double)us_all * 1e-6;
    g->conv_s += conv;
    return err;
}

int gnpu_gemm(gnpu *g, int id, int n, const float *const *xs, float *const *ys) {
    float *const *yl[1] = {ys};
    return gnpu_gemm_multi(g, 1, &id, n, xs, yl);
}

void gnpu_set_par(gnpu *g, gnpu_par_fn par, void *ctx) {
    if (!g) return;
    g->par = par;
    g->par_ctx = ctx;
}

double gnpu_convert_seconds(const gnpu *g) { return g ? g->conv_s : 0.0; }

size_t gnpu_bytes(const gnpu *g) { return g ? g->bytes : 0; }
double gnpu_dsp_seconds(const gnpu *g) { return g ? g->dsp_s : 0.0; }

// ---- the decoder

void *gnpu_shared_alloc(gnpu *g, size_t bytes, int cached, int *fd) {
    if (!g || !bytes || bytes > 0x7FFFFFFFu) return NULL;
    void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, cached ? RPCMEM_DEFAULT_FLAGS : RPCMEM_FLAG_UNCACHED, (int)bytes);
    if (!p) return NULL;
    const int f = rpcmem_to_fd(p);
    if (f < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, f, p, 0, bytes, FASTRPC_MAP_FD)) {
        rpcmem_free(p);
        return NULL;
    }
    if (g->n_shared == g->shared_cap) {
        g->shared_cap = g->shared_cap ? 2 * g->shared_cap : 16;
        g->shared = (Shared *)realloc(g->shared, (size_t)g->shared_cap * sizeof *g->shared);
    }
    g->shared[g->n_shared++] = (Shared){p, bytes, f};
    g->bytes += bytes;
    *fd = f;
    return p;
}

long long gnpu_dec_size(gnpu *g, const void *cfg, int cfg_len) {
    long long bytes = 0;
    return g && geode_npu_dec_size(g->h, (const unsigned char *)cfg, cfg_len, &bytes) == 0 ? bytes : -1;
}

int gnpu_dec_open(gnpu *g, const void *cfg, int cfg_len, int w_fd, int p_fd, int s_fd, int threads) {
    if (!g) return -1;
    const int e = geode_npu_dec_open(g->h, (const unsigned char *)cfg, cfg_len, w_fd, p_fd, s_fd, threads);
    g->dec = e == 0;
    return e;
}

int gnpu_dec_kv(gnpu *g, int chunk, int fd) { return g && g->dec ? geode_npu_dec_kv(g->h, chunk, fd) : -1; }

int gnpu_dec_step(gnpu *g, const float *x, int n, int hidden, int pos, const int *p3, int flags, double *dsp_s) {
    if (!g || !g->dec) return -1;
    long long us = 0;
    int nx = -1;
    unsigned char none[4];
    const int e = geode_npu_dec_step(g->h, (const unsigned char *)x, n * hidden * 4, pos, p3, 3 * n, flags, none, 4,
                                     &nx, &us);
    if (dsp_s) *dsp_s = (double)us * 1e-6;
    g->dsp_s += (double)us * 1e-6;
    return e;
}

int gnpu_dec_prefill(gnpu *g, const float *x, int n, int hidden, int pos, const int *p3, int flags, double *dsp_s) {
    if (!g || !g->dec) return -1;
    long long us = 0;
    const int e = geode_npu_dec_prefill(g->h, (const unsigned char *)x, n * hidden * 4, pos, p3, 3 * n, flags, &us);
    if (dsp_s) *dsp_s = (double)us * 1e-6;
    g->dsp_s += (double)us * 1e-6;
    return e;
}

int gnpu_dec_mtp_kv(gnpu *g, const float *e, const float *h, int n, int hidden, int pos, const int *p3, int flags,
                    double *dsp_s) {
    if (!g || !g->dec) return -1;
    long long us = 0;
    const int r = geode_npu_dec_mtp_kv(g->h, (const unsigned char *)e, n * hidden * 4, (const unsigned char *)h,
                                       n * hidden * 4, pos, p3, 3 * n, flags, &us);
    if (dsp_s) *dsp_s = (double)us * 1e-6;
    g->dsp_s += (double)us * 1e-6;
    return r;
}

int gnpu_moe(gnpu *g, const void *desc, int desc_len, int io_fd, double *dsp_s) {
    if (!g) return -1;
    long long us = 0;
    const int r = geode_npu_moe(g->h, (const unsigned char *)desc, desc_len, io_fd, &us);
    if (dsp_s) *dsp_s = (double)us * 1e-6;
    g->dsp_s += (double)us * 1e-6;
    return r;
}

int gnpu_mat_hmx(const gnpu *g, int id, int *fd, int *shift, long long *kp) {
    if (!g || id < 0 || id >= g->n_mats) return -1;
    *fd = g->mats[id].fd;
    *shift = g->mats[id].shift;
    *kp = g->mats[id].kp;
    return 0;
}

int gnpu_dec_draft(gnpu *g, const float *e, const float *h, int n, int hidden, int pos, const int *p3, int *draft,
                   float *prob, double *dsp_s) {
    if (!g || !g->dec) return -1;
    long long us = 0;
    const int r = geode_npu_dec_draft(g->h, (const unsigned char *)e, n * hidden * 4, (const unsigned char *)h,
                                      n * hidden * 4, pos, p3, 3 * n, draft, prob, &us);
    if (dsp_s) *dsp_s = (double)us * 1e-6;
    g->dsp_s += (double)us * 1e-6;
    return r;
}
