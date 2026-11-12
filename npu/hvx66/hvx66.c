// npu/hvx66/hvx66.c - the v66 module's DSP side (geode_hvx.idl): HVX integer vectors only.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_perf.h"
#include "HAP_mem.h"
#include "HAP_power.h"
#include "geode_hvx.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

typedef struct {
    int power;   // vote()'s answer
} hvx_t;

// the DSP's clocks and bus at their highest while the session lives, HVX powered (npu_imp.c's vote, without HMX):
// DCVS v3, else v2, else MIPS + bus bandwidth (older DSPs). Answers the one that took - 3, 2, 1 (0: none) - plus 10
// when HVX's power request failed.
static int vote(void *ctx) {
    int kind = 0;
    HAP_power_request_t r;
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_DCVS_v3;
    r.dcvs_v3.set_dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    r.dcvs_v3.set_bus_params = TRUE;
    r.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_core_params = TRUE;
    r.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_sleep_disable = TRUE;
    r.dcvs_v3.sleep_disable = TRUE;
    r.dcvs_v3.set_latency = TRUE;
    r.dcvs_v3.latency = 100;
    if (HAP_power_set(ctx, &r) == 0) kind = 3;
    if (!kind) {
        memset(&r, 0, sizeof r);
        r.type = HAP_power_set_DCVS_v2;
        r.dcvs_v2.dcvs_enable = TRUE;
        r.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        r.dcvs_v2.set_latency = TRUE;
        r.dcvs_v2.latency = 100;
        r.dcvs_v2.set_dcvs_params = TRUE;
        r.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_MAX;
        r.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_MAX;
        r.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_MAX;
        if (HAP_power_set(ctx, &r) == 0) kind = 2;
    }
    if (!kind) {
        memset(&r, 0, sizeof r);
        r.type = HAP_power_set_mips_bw;
        r.mips_bw.set_mips = TRUE;
        r.mips_bw.mipsPerThread = 2000;
        r.mips_bw.mipsTotal = 8000;
        r.mips_bw.set_bus_bw = TRUE;
        r.mips_bw.bwBytePerSec = 12000000000ull;
        r.mips_bw.busbwUsagePercentage = 100;
        r.mips_bw.set_latency = TRUE;
        r.mips_bw.latency = 100;
        if (HAP_power_set(ctx, &r) == 0) kind = 1;
    }
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = TRUE;
    return kind + (HAP_power_set(ctx, &r) ? 10 : 0);
}

int geode_hvx_open(const char *uri, remote_handle64 *h) {
    (void)uri;
    hvx_t *s = (hvx_t *)calloc(1, sizeof *s);
    if (!s) return AEE_ENOMEMORY;
    s->power = vote(s);
    *h = (remote_handle64)(uintptr_t)s;
    return AEE_SUCCESS;
}

int geode_hvx_close(remote_handle64 h) {
    free((void *)(uintptr_t)h);
    return AEE_SUCCESS;
}

int geode_hvx_ping(remote_handle64 h, int x, int *y, int *hvx, int *dot) {
    (void)h;
    *y = x + 1;
    *dot = 0;
    *hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (*hvx != QURT_EOK) return AEE_SUCCESS;
    static int32_t out[32] __attribute__((aligned(128)));
    HVX_Vector acc = Q6_V_vzero();
    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, Q6_Vb_vsplat_R(x & 0x7F), 0x04030201);
    *(HVX_Vector *)out = acc;
    int s = 0;
    for (int i = 0; i < 32; ++i) s += out[i];
    *dot = s;
    qurt_hvx_unlock();
    return AEE_SUCCESS;
}

// ---- bench: vrmpy's speed on 1..8 HVX threads
typedef struct {
    int mode, iters, err;
    const HVX_Vector *buf;
    int nvec;
    int32_t sink;
} job_t;

#define VRMPY8(W0, W1)                                                                                       \
    c0 = Q6_Vw_vrmpyacc_VwVbVb(c0, W0, a0), c1 = Q6_Vw_vrmpyacc_VwVbVb(c1, W0, a1),                         \
    c2 = Q6_Vw_vrmpyacc_VwVbVb(c2, W0, a2), c3 = Q6_Vw_vrmpyacc_VwVbVb(c3, W0, a3),                         \
    c4 = Q6_Vw_vrmpyacc_VwVbVb(c4, W1, a0), c5 = Q6_Vw_vrmpyacc_VwVbVb(c5, W1, a1),                         \
    c6 = Q6_Vw_vrmpyacc_VwVbVb(c6, W1, a2), c7 = Q6_Vw_vrmpyacc_VwVbVb(c7, W1, a3)

// ---- the 8-bit product's inner tile (geode_hvx_bench mode 2 times it): 32 output rows (the HVX lanes) x 8 tokens over
// G groups of 32 inputs. w: per group 8 vectors of hmx8 code bytes as they are (two column pairs x 32 rows each);
// a: the tokens' int8 words, token t's group g at a[t * kw + g * 8]; corr: 128 sum(a) per token and group at
// corr[t * gs + g]; wq: per group the rows' scale ratios (Q15) in the odd halfwords; aq: the tokens' ratios
// (Q15 << 16) at aq[t * gs + g]. out: 8 vectors of int32 sums.
static inline __attribute__((always_inline)) void tile32x8(const HVX_Vector *w, const int32_t *a, const int32_t *corr,
                                                          const HVX_Vector *wq, const int32_t *aq, int G, int kw,
                                                          int gs, HVX_Vector *out) {
    const HVX_Vector flip = Q6_V_vsplat_R(0x80808080);
    HVX_Vector acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = Q6_V_vzero();
    for (int g = 0; g < G; ++g) {
        HVX_Vector d0 = Q6_V_vzero(), d1 = d0, d2 = d0, d3 = d0, d4 = d0, d5 = d0, d6 = d0, d7 = d0;
        const int32_t *ag = a + g * 8;
        // the tokens' next-but-one groups into L1: the activations are scalar loads (vrmpy's Rt), and an L1 miss
        // stalls the thread - without this the product ran at 45-70% of its speed (measured)
        if (g + 2 < G)
            for (int t = 0; t < 8; ++t) Q6_dcfetch_A((void *)(ag + t * kw + 16));
        for (int j = 0; j < 8; ++j) {
            // [32 rows][4 inputs] from the two column pairs' halfwords, as offset binary (vrmpy's unsigned side)
            const HVX_Vector W = Q6_V_vxor_VV(Q6_Vh_vshuff_Vh(w[g * 8 + j]), flip);
            d0 = Q6_Vw_vrmpyacc_VwVubRb(d0, W, ag[0 * kw + j]);
            d1 = Q6_Vw_vrmpyacc_VwVubRb(d1, W, ag[1 * kw + j]);
            d2 = Q6_Vw_vrmpyacc_VwVubRb(d2, W, ag[2 * kw + j]);
            d3 = Q6_Vw_vrmpyacc_VwVubRb(d3, W, ag[3 * kw + j]);
            d4 = Q6_Vw_vrmpyacc_VwVubRb(d4, W, ag[4 * kw + j]);
            d5 = Q6_Vw_vrmpyacc_VwVubRb(d5, W, ag[5 * kw + j]);
            d6 = Q6_Vw_vrmpyacc_VwVubRb(d6, W, ag[6 * kw + j]);
            d7 = Q6_Vw_vrmpyacc_VwVubRb(d7, W, ag[7 * kw + j]);
        }
        HVX_Vector d[8] = {d0, d1, d2, d3, d4, d5, d6, d7};
        for (int t = 0; t < 8; ++t) {
            HVX_Vector s = Q6_Vw_vsub_VwVw(d[t], Q6_V_vsplat_R(corr[t * gs + g]));
            s = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_Vw_vasl_VwR(s, 2), wq[g]);
            s = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(s, Q6_V_vsplat_R(aq[t * gs + g]));
            acc[t] = Q6_Vw_vadd_VwVw(acc[t], s);
        }
    }
    for (int t = 0; t < 8; ++t) out[t] = acc[t];
}

// ---- q8_gemm: row blocks split over the HVX threads; k in chunks of 2048 inputs (the tokens' chunk, n x 2 KB, stays
// in L2 while every block of the thread goes by - down's whole 6144 does not fit); each block's chunk against every
// token in tiles of 8, its codes from L2 after the first tile. Integer sums: the chunks add exactly.
typedef long hvx_uv_t __attribute__((__vector_size__(128), __aligned__(1)));   // (an unaligned vector)
enum { kChunkG = 64 };                                                          // 2048 inputs

typedef struct {
    const uint8_t *w;   // the matrix's first 32-row block
    const int8_t *x;
    const int32_t *corr, *aq, *wq;
    int32_t *y;
    int rows, k, n8, b0, b1, err;
} gjob_t;

static void gemm_worker(void *arg) {
    gjob_t *j = (gjob_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        j->err = 1;
        qurt_thread_exit(0);
        return;
    }
    const int G = j->k / 32, kw = j->k / 4;
    const size_t bstride = (size_t)j->k * 34;   // a block's codes, then its scales
    HVX_Vector wq[kChunkG], out[8];              // (the block's ratios for the chunk, aligned)
    for (int g0 = 0; g0 < G; g0 += kChunkG) {
        const int gc = G - g0 < kChunkG ? G - g0 : kChunkG;
        for (int b = j->b0; b < j->b1; ++b) {
            const hvx_uv_t *src = (const hvx_uv_t *)(j->wq + ((size_t)b * G + g0) * 32);
            for (int g = 0; g < gc; ++g) wq[g] = (HVX_Vector)src[g];
            const HVX_Vector *wb = (const HVX_Vector *)(j->w + (size_t)b * bstride) + (size_t)g0 * 8;
            for (int t0 = 0; t0 < j->n8; t0 += 8) {
                tile32x8(wb, (const int32_t *)(j->x + (size_t)t0 * j->k) + g0 * 8, j->corr + (size_t)t0 * G + g0, wq,
                         j->aq + (size_t)t0 * G + g0, gc, kw, G, out);
                for (int t = 0; t < 8; ++t) {
                    hvx_uv_t *yp = (hvx_uv_t *)(j->y + (size_t)(t0 + t) * j->rows + b * 32);
                    *yp = g0 ? (hvx_uv_t)Q6_Vw_vadd_VwVw((HVX_Vector)*yp, out[t]) : (hvx_uv_t)out[t];
                }
            }
        }
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int geode_hvx_q8_gemm(remote_handle64 h, const unsigned char *w, int w_len, int rows, int k, int n8,
                      const unsigned char *x, int x_len, const unsigned char *corr, int corr_len,
                      const unsigned char *aq, int aq_len, const unsigned char *wq, int wq_len, unsigned char *y,
                      int y_len, int threads, int64_t *usec) {
    (void)h;
    *usec = 0;
    const int G = k / 32;
    if (rows < 32 || rows % 32 || k < 64 || k % 32 || k > 65536 || n8 < 8 || n8 % 8 || threads < 1 || threads > 4 ||
        (int64_t)x_len < (int64_t)n8 * k || (int64_t)corr_len < (int64_t)n8 * G * 4 ||
        (int64_t)aq_len < (int64_t)n8 * G * 4 || (int64_t)wq_len < (int64_t)rows * G * 4 ||
        (int64_t)y_len < (int64_t)n8 * rows * 4 || (uintptr_t)w % 128 ||
        (uint64_t)(rows / 32) * (uint64_t)k * 34u > (uint64_t)w_len)
        return AEE_EBADPARM;
    int e = AEE_SUCCESS;
    const uint64_t t0 = HAP_perf_get_time_us();
    gjob_t jobs[4];
    qurt_thread_t tid[4];
    void *stacks[4] = {0};
    enum { kStack = 24 << 10 };   // (the chunk's 64 ratio vectors live on it)
    const int nb = rows / 32;
    int made = 0;
    for (int i = 0; i < threads; ++i) {
        gjob_t *j = &jobs[i];
        j->w = (const uint8_t *)w, j->x = (const int8_t *)x, j->corr = (const int32_t *)corr;
        j->aq = (const int32_t *)aq, j->wq = (const int32_t *)wq, j->y = (int32_t *)y;
        j->rows = rows, j->k = k, j->n8 = n8, j->err = 0;
        j->b0 = nb * i / threads, j->b1 = nb * (i + 1) / threads;
        if (!(stacks[i] = malloc(kStack + 128))) {
            e = AEE_ENOMEMORY;
            break;
        }
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, (void *)(((uintptr_t)stacks[i] + 127) & ~(uintptr_t)127));
        qurt_thread_attr_set_stack_size(&attr, kStack);
        qurt_thread_attr_set_priority(&attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
        if (qurt_thread_create(&tid[i], &attr, gemm_worker, j) != QURT_EOK) {
            e = AEE_EFAILED;
            break;
        }
        ++made;
    }
    int failed = 0;
    for (int i = 0; i < made; ++i) {
        int st = 0;
        qurt_thread_join(tid[i], &st);
        failed |= jobs[i].err;
    }
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    for (int i = 0; i < threads; ++i) free(stacks[i]);
    if (!e && failed) e = AEE_EFAILED;   // (an HVX context not had)
    return e;
}

// ---- q4_gemm (geode_hvx.idl): the 4-bit product's inner tile - 32 rows (the lanes) x 8 tokens over np pairs of
// 16-value blocks. w: the tile's code vectors at the first pair's chunk; a: token 0's int8 words at the first pair (each
// token kw words on); xs: token 0's (xc, aq) words at the first pair (each token xw words on); wq: the tile's ratio
// vectors at the first pair. Per pair: the codes doubled (2c, 0..30, unsigned) from the 4 words' nibbles, each block's
// 4 vrmpy against the token's words, the two blocks' sums as the halfwords of one vector (mod 2^16: the true value
// fits), corrected by -off sum(x), times the rows' ratios (Q15), then the token's (Q13) into the 32-bit sums.
static inline __attribute__((always_inline)) void tile_q4(const HVX_Vector *w, const int32_t *a, const int32_t *xs,
                                                         const HVX_Vector *wq, int np, int kw, int xw, HVX_Vector *out) {
    const HVX_Vector m = Q6_V_vsplat_R(0x1E1E1E1E);
    HVX_Vector acc0 = Q6_V_vzero(), acc1 = acc0, acc2 = acc0, acc3 = acc0, acc4 = acc0, acc5 = acc0, acc6 = acc0,
               acc7 = acc0;
    for (int p = 0; p < np; ++p) {
        const HVX_Vector *wp = w + (p >> 1) * 8 + (p & 1) * 4;
        const HVX_Vector L0 = Q6_V_vand_VV(Q6_Vh_vasl_VhR(wp[0], 1), m), H0 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(wp[0], 3), m);
        const HVX_Vector L1 = Q6_V_vand_VV(Q6_Vh_vasl_VhR(wp[1], 1), m), H1 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(wp[1], 3), m);
        const HVX_Vector L2 = Q6_V_vand_VV(Q6_Vh_vasl_VhR(wp[2], 1), m), H2 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(wp[2], 3), m);
        const HVX_Vector L3 = Q6_V_vand_VV(Q6_Vh_vasl_VhR(wp[3], 1), m), H3 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(wp[3], 3), m);
        const HVX_Vector wv = wq[p];
        const int32_t *ag = a + p * 8, *xg = xs + p * 2;
        // the tokens' next-but-one pairs into L1 (scalar loads: an L1 miss stalls the thread - q8_gemm's lesson)
        if (p + 2 < np)
            for (int t = 0; t < 8; ++t) Q6_dcfetch_A((void *)(ag + t * kw + 16));
#define GEODE_Q4_TOKEN(T, ACC)                                                                                       \
    {                                                                                                               \
        const int32_t *at = ag + (T) * kw;                                                                          \
        HVX_Vector s0 = Q6_Vw_vrmpy_VubRb(L0, at[0]);                                                               \
        s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, H0, at[1]);                                                                 \
        s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, L1, at[2]);                                                                 \
        s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, H1, at[3]);                                                                 \
        HVX_Vector s1 = Q6_Vw_vrmpy_VubRb(L2, at[4]);                                                               \
        s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, H2, at[5]);                                                                 \
        s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, L3, at[6]);                                                                 \
        s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, H3, at[7]);                                                                 \
        HVX_Vector v = Q6_Vh_vadd_VhVh(Q6_Vh_vshuffe_VhVh(s1, s0), Q6_V_vsplat_R(xg[(T) * xw]));                    \
        ACC = Q6_Vw_vdmpyacc_VwVhRh_sat(ACC, Q6_Vh_vmpy_VhVh_s1_rnd_sat(v, wv), xg[(T) * xw + 1]);                 \
    }
        GEODE_Q4_TOKEN(0, acc0) GEODE_Q4_TOKEN(1, acc1) GEODE_Q4_TOKEN(2, acc2) GEODE_Q4_TOKEN(3, acc3)
        GEODE_Q4_TOKEN(4, acc4) GEODE_Q4_TOKEN(5, acc5) GEODE_Q4_TOKEN(6, acc6) GEODE_Q4_TOKEN(7, acc7)
#undef GEODE_Q4_TOKEN
    }
    out[0] = acc0, out[1] = acc1, out[2] = acc2, out[3] = acc3, out[4] = acc4, out[5] = acc5, out[6] = acc6, out[7] = acc7;
}

typedef struct {
    const uint8_t *w;
    const int8_t *x;
    const int32_t *xs, *wq;
    int32_t *y;
    HVX_Vector *buf;   // quad: a tile's pass of code vectors in tile8's order (kc / 8 vectors)
    int rows, k, n8, kc, quad, flags, b0, b1, err;
    int32_t amax;      // the largest |sum| a pass or the total reached (geode_hvx.idl: saturation's guard)
} q4job_t;

// quad (engine/src/kernels.hpp, the CPU's layout): one tile's pass of codes into tile8's order - per word j and 8
// chunks, the 8 quads' 128-byte runs [chunk][row][4 bytes] transposed into 8 vectors [quad][row][4 bytes] (rows in the
// lanes) by three rounds of vshuff (16, 32, 64 bytes). t: the tile's first quad (each gq bytes on, 128-byte aligned:
// k % 512 == 0); c0: the pass's first chunk; nc its chunks (a multiple of 8); dst: vector (c - c0) * 8 + j.
static void quad_to_tile8(const uint8_t *t, size_t gq, int nch, int c0, int nc, HVX_Vector *dst) {
    for (int j = 0; j < 8; ++j)
        for (int g = 0; g < nc; g += 8) {
            const size_t off = ((size_t)j * nch + (size_t)(c0 + g)) * 16;
            HVX_Vector v[8];
            for (int i = 0; i < 8; ++i) v[i] = *(const HVX_Vector *)(t + (size_t)i * gq + off);
            // round 1 (16 bytes): quads 2i, 2i+1 interleaved - chunks 0-3 (lo), 4-7 (hi)
            const HVX_VectorPair a01 = Q6_W_vshuff_VVR(v[1], v[0], -16), a23 = Q6_W_vshuff_VVR(v[3], v[2], -16);
            const HVX_VectorPair a45 = Q6_W_vshuff_VVR(v[5], v[4], -16), a67 = Q6_W_vshuff_VVR(v[7], v[6], -16);
            // round 2 (32 bytes): quads 0-3 (4-7) - chunks 0-1, 2-3, 4-5, 6-7
            const HVX_VectorPair b0 = Q6_W_vshuff_VVR(Q6_V_lo_W(a23), Q6_V_lo_W(a01), -32);
            const HVX_VectorPair b1 = Q6_W_vshuff_VVR(Q6_V_hi_W(a23), Q6_V_hi_W(a01), -32);
            const HVX_VectorPair b2 = Q6_W_vshuff_VVR(Q6_V_lo_W(a67), Q6_V_lo_W(a45), -32);
            const HVX_VectorPair b3 = Q6_W_vshuff_VVR(Q6_V_hi_W(a67), Q6_V_hi_W(a45), -32);
            // round 3 (64 bytes): all 8 quads - one chunk a vector
            const HVX_VectorPair c01 = Q6_W_vshuff_VVR(Q6_V_lo_W(b2), Q6_V_lo_W(b0), -64);
            const HVX_VectorPair c23 = Q6_W_vshuff_VVR(Q6_V_hi_W(b2), Q6_V_hi_W(b0), -64);
            const HVX_VectorPair c45 = Q6_W_vshuff_VVR(Q6_V_lo_W(b3), Q6_V_lo_W(b1), -64);
            const HVX_VectorPair c67 = Q6_W_vshuff_VVR(Q6_V_hi_W(b3), Q6_V_hi_W(b1), -64);
            HVX_Vector *d = dst + (size_t)g * 8 + j;
            d[0 * 8] = Q6_V_lo_W(c01), d[1 * 8] = Q6_V_hi_W(c01), d[2 * 8] = Q6_V_lo_W(c23), d[3 * 8] = Q6_V_hi_W(c23);
            d[4 * 8] = Q6_V_lo_W(c45), d[5 * 8] = Q6_V_hi_W(c45), d[6 * 8] = Q6_V_lo_W(c67), d[7 * 8] = Q6_V_hi_W(c67);
        }
}

static void q4_worker(void *arg) {
    q4job_t *j = (q4job_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        j->err = 1;
        qurt_thread_exit(0);
        return;
    }
    const int P = j->k / 32, nch = j->k / 64;
    const size_t tstride = (size_t)j->k * 18;   // a tile's bytes (its codes 16 k, then its scales 2 k)
    const size_t gq = (size_t)j->k * 9 / 4;      // quad: a group of 4 rows' bytes
    HVX_Vector out[8], mx = Q6_V_vzero();
    // k in passes of kc inputs (the tokens' slice stays in L2 while the thread's tiles go by); integer sums add exactly.
    // x and xs pass by pass, y tile by tile (geode_hvx.idl): what a pass touches is a few contiguous pages - a token
    // row of k bytes apart for each token, as [n][k] was, missed the DSP's TLB (n 256, k 12288: 15 against 25 G MAC/s)
    for (int k0 = 0; k0 < j->k; k0 += j->kc) {
        const int kn = j->k - k0 < j->kc ? j->k - k0 : j->kc, p0 = k0 / 32, np = kn / 32, kw = kn / 4, xw = kn / 16;
        const int8_t *xp = j->x + (size_t)k0 * j->n8;
        const int32_t *xsp = j->xs + (size_t)(k0 / 16) * j->n8;
        // the tokens in groups whose slice of the pass (tg x kn bytes) stays in L2 - 128 KB: a whole prompt of 1024
        // tokens in one group was 18 against 40 G MAC/s (each tile's codes turned again for each group: cheap)
        int tg = (128 << 10) / kn / 8 * 8;
        if (tg < 8) tg = 8;
        for (int g0 = 0; g0 < j->n8; g0 += tg) {
        const int g1 = j->n8 - g0 < tg ? j->n8 : g0 + tg;
        for (int b = j->b0; b < j->b1; ++b) {
            const HVX_Vector *wb = (const HVX_Vector *)(j->w + (size_t)b * tstride) + (size_t)(k0 / 64) * 8;
            // the next tile's codes of this pass into L2 while this one computes (the quad layout's 8 x 8 runs are
            // read back to back by the turn below: from DDR they stalled it, quad 27-34 against tile8's 34-38 G MAC/s).
            // Quad: one box - rows of k/4 bytes, quad i's word j at row 9i + j (row 9i + 8: its scales, read along)
            if (!(j->flags & 2) && b + 1 < j->b1) {   // (flags 2: none - measuring)
                const uint8_t *nt = j->w + (size_t)(b + 1) * tstride;
                const uint64_t box = j->quad ? (uint64_t)(j->k / 4) << 32 | (uint64_t)(kn / 64 * 16) << 16 | 72u
                                             : (uint64_t)(kn * 16) << 32 | (uint64_t)(kn * 16) << 16 | 1u;
                Q6_l2fetch_AP((void *)(nt + (j->quad ? (size_t)(k0 / 64) * 16 : (size_t)k0 * 16)), box);
            }
            if (j->quad) {   // (the pass's codes in tile8's order, once for all the tokens)
                quad_to_tile8(j->w + (size_t)b * tstride, gq, nch, k0 / 64, kn / 64, j->buf);
                wb = j->buf;
            }
            const HVX_Vector *wqb = (const HVX_Vector *)(j->wq + ((size_t)b * P + p0) * 32);
            HVX_Vector *yb = (HVX_Vector *)(j->y + (size_t)b * j->n8 * 32);
            for (int t0 = g0; t0 < g1; t0 += 8) {
                tile_q4(wb, (const int32_t *)(xp + (size_t)t0 * kn), xsp + (size_t)t0 * xw, wqb, np, kw, xw, out);
                for (int t = 0; t < 8; ++t) {
                    // (the largest |sum| of a pass and of the total so far: the host's guard against saturation)
                    const HVX_Vector v = k0 ? Q6_Vw_vadd_VwVw_sat(yb[t0 + t], out[t]) : out[t];
                    mx = Q6_Vw_vmax_VwVw(mx, Q6_Vw_vmax_VwVw(Q6_Vw_vabs_Vw_sat(out[t]), Q6_Vw_vabs_Vw_sat(v)));
                    yb[t0 + t] = v;
                }
            }
        }
        }
    }
    int32_t m[32] __attribute__((aligned(128)));
    *(HVX_Vector *)m = mx;
    j->amax = 0;
    for (int i = 0; i < 32; ++i) j->amax = m[i] > j->amax ? m[i] : j->amax;
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int geode_hvx_q4_gemm(remote_handle64 h, const unsigned char *w, int w_len, int rows, int k, int n8,
                      const unsigned char *x, int x_len, const unsigned char *xs, int xs_len, const unsigned char *wq,
                      int wq_len, unsigned char *y, int y_len, int threads, int kchunk, int quad, int *amax,
                      int64_t *usec) {
    (void)h;
    *usec = 0, *amax = 0;
    const int P = k / 32;
    if (!kchunk) kchunk = 2048;
    if ((quad & 1) && (k % 512 || kchunk % 512)) return AEE_EBADPARM;   // (whole runs of 8 chunks, 128-byte aligned groups)
    if (rows < 32 || rows % 32 || k < 64 || k % 64 || k > 65536 || n8 < 8 || n8 % 8 || threads < 1 || threads > 4 ||
        kchunk < 64 || kchunk % 64 || (int64_t)x_len < (int64_t)n8 * k || (int64_t)xs_len < (int64_t)n8 * P * 8 ||
        (int64_t)wq_len < (int64_t)rows * P * 4 || (int64_t)y_len < (int64_t)n8 * rows * 4 || (uintptr_t)w % 128 ||
        (uintptr_t)wq % 128 || (uintptr_t)y % 128 || (uintptr_t)x % 4 || (uintptr_t)xs % 4 ||
        (uint64_t)(rows / 32) * (uint64_t)k * 18u > (uint64_t)w_len)
        return AEE_EBADPARM;
    int e = AEE_SUCCESS;
    const uint64_t t0 = HAP_perf_get_time_us();
    q4job_t jobs[4];
    qurt_thread_t tid[4];
    void *stacks[4] = {0};
    enum { kStack = 16 << 10 };
    const int nb = rows / 32;
    int made = 0;
    for (int i = 0; i < threads; ++i) {
        q4job_t *j = &jobs[i];
        j->w = (const uint8_t *)w, j->x = (const int8_t *)x, j->xs = (const int32_t *)xs, j->wq = (const int32_t *)wq;
        j->y = (int32_t *)y, j->rows = rows, j->k = k, j->n8 = n8, j->kc = kchunk, j->quad = quad & 1, j->flags = quad, j->err = 0;
        j->b0 = nb * i / threads, j->b1 = nb * (i + 1) / threads, j->amax = 0, j->buf = NULL;
        // (quad: its pass buffer after the stack, kc / 8 vectors)
        if (!(stacks[i] = malloc(kStack + 128 + ((quad & 1) ? (size_t)kchunk * 16 : 0)))) {
            e = AEE_ENOMEMORY;
            break;
        }
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, (void *)(((uintptr_t)stacks[i] + 127) & ~(uintptr_t)127));
        qurt_thread_attr_set_stack_size(&attr, kStack);
        qurt_thread_attr_set_priority(&attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
        if (quad & 1) j->buf = (HVX_Vector *)((((uintptr_t)stacks[i] + 127) & ~(uintptr_t)127) + kStack);
        if (qurt_thread_create(&tid[i], &attr, q4_worker, j) != QURT_EOK) {
            e = AEE_EFAILED;
            break;
        }
        ++made;
    }
    int failed = 0;
    for (int i = 0; i < made; ++i) {
        int st = 0;
        qurt_thread_join(tid[i], &st);
        failed |= jobs[i].err;
        *amax = jobs[i].amax > *amax ? jobs[i].amax : *amax;
    }
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    for (int i = 0; i < threads; ++i) free(stacks[i]);
    if (!e && failed) e = AEE_EFAILED;
    return e;
}

static void worker(void *arg) {
    job_t *j = (job_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        j->err = 1;
        qurt_thread_exit(0);
        return;
    }
    if (j->mode == 2) {   // 128 rows (four tiles) x 8 tokens x 2048 inputs, the data in L2, iters times
        enum { G = 64 };
        const uint8_t *base = (const uint8_t *)j->buf;
        const HVX_Vector *w = (const HVX_Vector *)base;                    // 4 tiles x G x 8 vectors (256 KB)
        const int32_t *a = (const int32_t *)(base + (256 << 10));           // 8 x G x 8 words (16 KB)
        const int32_t *corr = (const int32_t *)(base + (272 << 10));        // 8 x G
        const HVX_Vector *wq = (const HVX_Vector *)(base + (276 << 10));    // 4 x G vectors (32 KB)
        const int32_t *aq = (const int32_t *)(base + (308 << 10));          // 8 x G
        HVX_Vector out[8], s = Q6_V_vzero();
        for (int it = 0; it < j->iters; ++it)
            for (int b = 0; b < 4; ++b) {
                tile32x8(w + b * G * 8, a, corr, wq + b * G, aq, G, G * 8, G, out);
                for (int t = 0; t < 8; ++t) s = Q6_Vw_vadd_VwVw(s, out[t]);   // (all eight: none is dead code)
            }
        int32_t o[32] __attribute__((aligned(128)));
        *(HVX_Vector *)o = s;
        j->sink = o[0];
        qurt_hvx_unlock();
        qurt_thread_exit(0);
        return;
    }
    // four tokens' activation words (the same 4 bytes in every lane, as a product's splat of one token's group)
    const HVX_Vector a0 = Q6_V_vsplat_R(0x01020304 + j->iters), a1 = Q6_V_vsplat_R(0x05060708);
    const HVX_Vector a2 = Q6_V_vsplat_R(0x090A0B0C), a3 = Q6_V_vsplat_R(0x0D0E0F10);
    HVX_Vector c0 = Q6_V_vzero(), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    if (j->mode == 0) {
        const HVX_Vector w0 = Q6_V_vsplat_R(0x7F01FE02 ^ j->iters), w1 = Q6_V_vsplat_R(0x03FD7E80);
        for (int i = 0; i < j->iters; ++i) VRMPY8(w0, w1);
    } else {
        for (int it = 0; it < j->iters; ++it)
            for (int v = 0; v + 1 < j->nvec; v += 2) {
                const HVX_Vector w0 = j->buf[v], w1 = j->buf[v + 1];
                VRMPY8(w0, w1);
            }
    }
    HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(c0, c1), Q6_Vw_vadd_VwVw(c2, c3)),
                                   Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(c4, c5), Q6_Vw_vadd_VwVw(c6, c7)));
    int32_t out[32] __attribute__((aligned(128)));
    *(HVX_Vector *)out = s;
    j->sink = out[0] ^ out[31];
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int geode_hvx_bench(remote_handle64 h, int threads, int mode, int iters, int *units, int *power, int64_t *usec,
                    int64_t *macs) {
    hvx_t *s = (hvx_t *)(uintptr_t)h;
    *units = qurt_hvx_get_units();
    *power = s ? s->power : -1;
    *usec = 0, *macs = 0;
    if (threads < 1 || threads > 8 || iters < 1 || mode < 0 || mode > 2) return AEE_EBADPARM;
    enum { kStack = 16 << 10, kBuf = 8 << 20 };
    uint8_t *raw = NULL;
    const HVX_Vector *buf = NULL;
    if (mode >= 1) {   // (mode 2: the tile's 320 KB of data)
        const size_t len = mode == 1 ? (size_t)kBuf : (size_t)(320 << 10);
        if (!(raw = (uint8_t *)malloc(len + 128))) return AEE_ENOMEMORY;
        buf = (const HVX_Vector *)(((uintptr_t)raw + 127) & ~(uintptr_t)127);
        memset((void *)buf, 3, len);
    }
    job_t jobs[8];
    qurt_thread_t tid[8];
    void *stacks[8] = {0};
    int made = 0, e = AEE_SUCCESS;
    const uint64_t t0 = HAP_perf_get_time_us();
    for (int i = 0; i < threads; ++i) {
        memset(&jobs[i], 0, sizeof jobs[i]);
        jobs[i].mode = mode, jobs[i].iters = iters, jobs[i].buf = buf, jobs[i].nvec = kBuf / 128;
        if (!(stacks[i] = malloc(kStack))) {
            e = AEE_ENOMEMORY;
            break;
        }
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, stacks[i]);
        qurt_thread_attr_set_stack_size(&attr, kStack);
        qurt_thread_attr_set_priority(&attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
        if (qurt_thread_create(&tid[i], &attr, worker, &jobs[i]) != QURT_EOK) {
            e = AEE_EFAILED;
            break;
        }
        ++made;
    }
    int failed = 0;
    for (int i = 0; i < made; ++i) {
        int st = 0;
        qurt_thread_join(tid[i], &st);
        failed |= jobs[i].err;
    }
    const uint64_t t1 = HAP_perf_get_time_us();
    for (int i = 0; i < threads; ++i) free(stacks[i]);
    free(raw);
    if (e) return e;
    *usec = (int64_t)(t1 - t0);
    if (!failed) {
        const int64_t per = mode == 0   ? (int64_t)iters * 8 * 128
                            : mode == 1 ? (int64_t)iters * (kBuf / 256) * 8 * 128
                                        : (int64_t)iters * 128 * 8 * 2048;
        *macs = per * threads;
    }
    return AEE_SUCCESS;
}
