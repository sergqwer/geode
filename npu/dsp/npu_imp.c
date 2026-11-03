// npu/dsp/npu_imp.c - Geode's code on the cDSP (an unsigned PD): HMX matrix products on fp16 croutons in VTCM.
//
// HMX layouts (Hexagon v81 HMX PRM, fp16), used for the activations, the weights and the results alike:
//   activation / result crouton: 32 rows x 32 channels, 2 KB; fp16 index (row / 2) * 64 + ch * 2 + row % 2
//   weights for 32 output columns: per 32 input channels a 2 KB block; fp16 index (in / 2) * 64 + out * 2 + in % 2
// A deep multiply takes up to 32 croutons (1024 channels) a time; the accumulator keeps summing across multiplies
// until a convert clears it.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_compute_res.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "hmx_hexagon_protos.h"
#include "geode_npu.h"
#include "qurt.h"
#include "dspqueue.h"

#define VTCM_WANT (6u << 20)

typedef struct {
    unsigned ctx;
    uint8_t *vtcm;
    unsigned vtcm_size;
    int api;              // 1 legacy, 3 lock3, 4 lock4
    int locked;
    unsigned int *busy;
    compute_res_hmx_mutex_t mtx;
    int power;
    dspqueue_t queue;
} npu_t;

static int power_up(npu_t *s) {
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
    int e = HAP_power_set(s, &r);
    if (e) return e;
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_HMX;
    r.hmx.power_up = TRUE;
    if ((e = HAP_power_set(s, &r))) return e;
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = TRUE;
    return HAP_power_set(s, &r);
}

static void release(npu_t *s) {
    if (s->locked) {
        if (s->api == 4) {
            if (s->busy) *s->busy = 0;
            HAP_compute_res_hmx_unlock4(s->ctx, s->busy);
        } else if (s->api == 3) {
            s->mtx.unlock(s->mtx.mutex);
            HAP_compute_res_hmx_unlock3(s->ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &s->mtx);
        } else {
            HAP_compute_res_hmx_unlock(s->ctx);
        }
        s->locked = 0;
    }
    if (s->ctx) {
        HAP_compute_res_release(s->ctx);
        s->ctx = 0;
    }
}

// VTCM + HMX for this session, once
static int acquire(npu_t *s) {
    if (s->locked) return 0;
    int e;
    if (!s->power && (e = power_up(s))) return e;
    s->power = 1;
    unsigned int caps = 0;
    s->api = 1;
    if (HAP_compute_res_query_capability(HAP_COMPUTE_RES_PREEMPTION_CAPABILITY, &caps) == AEE_SUCCESS) {
        if (caps & HAP_COMPUTE_RES_HMX_LOCK4) s->api = 4;
        else if (caps & HAP_COMPUTE_RES_AUTONOMOUS_PREEMPTION) s->api = 3;
    }
    compute_res_attr_t a;
    if ((e = HAP_compute_res_attr_init(&a))) return e;
    unsigned want = VTCM_WANT;
    for (;;) {
        if ((e = HAP_compute_res_attr_set_vtcm_param(&a, want, 1))) return e;
        if (s->api == 1 && (e = HAP_compute_res_attr_set_hmx_param(&a, 1))) return e;
        s->ctx = HAP_compute_res_acquire(&a, 100000);
        if (s->ctx || want <= (1u << 20)) break;
        want >>= 1;
    }
    if (!s->ctx) return AEE_ERESOURCENOTFOUND;
    s->vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&a);
    s->vtcm_size = want;
    if (!s->vtcm) return AEE_ERESOURCENOTFOUND;
    if (s->api == 4) {
        if ((e = HAP_compute_res_hmx_lock4(s->ctx, &s->busy, 100000))) return e;
        if (s->busy) *s->busy = 1;
    } else if (s->api == 3) {
        if ((e = HAP_compute_res_hmx_lock3(s->ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &s->mtx, 100000))) return e;
        s->mtx.lock(s->mtx.mutex);
    } else if ((e = HAP_compute_res_hmx_lock(s->ctx))) {
        return e;
    }
    s->locked = 1;
    Q6_mxclracc_hf();
    return 0;
}

int geode_npu_open(const char *uri, remote_handle64 *h) {
    (void)uri;
    npu_t *s = (npu_t *)calloc(1, sizeof *s);
    if (!s) return AEE_ENOMEMORY;
    *h = (remote_handle64)(uintptr_t)s;
    return 0;
}

int geode_npu_close(remote_handle64 h) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    if (s) {
        release(s);
        free(s);
    }
    return 0;
}

// the session's VTCM for dec.c's decoder (acquired once, as for the products)
int geode_npu_vtcm(remote_handle64 h, uint8_t **vtcm, unsigned *size) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    const int e = acquire(s);
    *vtcm = e ? NULL : s->vtcm;
    *size = e ? 0 : s->vtcm_size;
    return e;
}

int geode_npu_ping(remote_handle64 h, int x, int *y) {
    (void)h;
    *y = x + 1;
    return 0;
}

int geode_npu_info(remote_handle64 h, int *vtcm_bytes, int *lock_api, int *ok) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    int e = acquire(s);
    *vtcm_bytes = (int)s->vtcm_size;
    *lock_api = s->api;
    *ok = e == 0;
    return e;
}

// ---- layouts -------------------------------------------------------------------------------------------------

static inline uint32_t act_index(int r, int c) {   // within a 32-row tile of K channels, in fp16 elements
    return (uint32_t)(c >> 5) * 1024u + (uint32_t)(r >> 1) * 64u + (uint32_t)(c & 31) * 2u + (uint32_t)(r & 1);
}
static inline uint32_t wgt_index(int o, int c) {   // within a 32-column block of K input channels
    return (uint32_t)(c >> 5) * 1024u + (uint32_t)((c & 31) >> 1) * 64u + (uint32_t)(o & 31) * 2u + (uint32_t)(c & 1);
}

static inline float val(uint32_t a, uint32_t b, uint32_t salt) {   // a small deterministic value in [-1, 1)
    uint32_t x = a * 2654435761u ^ (b + salt) * 2246822519u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return (float)(int32_t)(x & 0xFFFF) / 32768.0f - 1.0f;
}

// one 32x32 result: the activation tile (k channels) times a 32-column weight block, into a 2 KB crouton
static inline void tile(const uint8_t *act, const uint8_t *wgt, uint8_t *out, int croutons) {
    for (int c = 0; c < croutons; c += 32) {
        const int n = croutons - c < 32 ? croutons - c : 32;
        Q6_activation_hf_mxmem_RR_deep((uint32_t)(uintptr_t)(act + c * 2048), ((uint32_t)(n - 1) << 11) | 0x7C);
        Q6_weight_hf_mxmem_RR((uint32_t)(uintptr_t)(wgt + c * 2048), (uint32_t)n * 2048u - 1u);
    }
    Q6_cvt_hf_acc_R(2);   // bias set 0, clear the accumulator; bit 1 keeps negative results (without it v79 clamps them to 0)
    Q6_mxmem_cvt_RR((void *)out, 0);
}

static void hvx_copy(uint8_t *dst, const uint8_t *src, uint32_t bytes) {   // both 128-byte aligned
    HVX_Vector *d = (HVX_Vector *)dst;
    const HVX_Vector *p = (const HVX_Vector *)src;
    for (uint32_t i = 0; i < bytes / 128; ++i) d[i] = p[i];
}

// ---- q4 weights for HMX ("hmx4") -----------------------------------------------------------------------------
// One 32-column block of a q4 matrix (the pack's codes: level = 2c - 15 - or, for Gemma 4's q4i, c - 8; one scale
// per 16 inputs of a column):
//   codes:  k/2 input pairs x 32 bytes; byte o of pair p = column o, input 2p in the low nibble, 2p+1 in the high
//   scales: k/16 groups x 32 fp16 (the block's E4M3 scale times the matrix's global scale), group g = inputs 16g..
// so a source byte's two nibbles land next to each other in HMX's order (index out * 2 + in % 2) as they expand.
#define HMX4_BYTES(k) ((uint32_t)(k) * 20u)
// a gemm's out_shift with this bit: the weights' codes are q4i's (level = c - 8; npu_api.c GNPU_SHIFT_Q4I)
#define HMX4_Q4I 0x100

static void dequant_block(uint8_t *dst, const uint8_t *src, int k, int q4i) {
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector k15 = Q6_Vh_vsplat_R(q4i ? 8 : 15);
    const HVX_Vector *cv = (const HVX_Vector *)src;                         // 4 input pairs a vector
    const HVX_Vector *sv = (const HVX_Vector *)(src + (uint32_t)k * 16u);  // 2 groups (64 fp16) a vector
    HVX_Vector *out = (HVX_Vector *)dst;
    for (int g2 = 0; g2 < k / 32; ++g2) {   // 32 inputs: 2 scale groups, 4 code vectors, 16 result vectors
        const HVX_VectorPair sp = Q6_W_vshuff_VVR(sv[g2], sv[g2], -2);   // [s0 s0 s1 s1 ...]: lo group 2g2, hi 2g2+1
        for (int q = 0; q < 4; ++q) {
            const HVX_Vector v = cv[g2 * 4 + q];
            const HVX_Vector lo = Q6_V_vand_VV(v, m0f);
            const HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), m0f);
            const HVX_VectorPair cb = Q6_W_vshuff_VVR(hi, lo, -1);   // the codes in HMX order, 2 pairs a vector
            const HVX_Vector sc = q < 2 ? Q6_V_lo_W(sp) : Q6_V_hi_W(sp);
            for (int half = 0; half < 2; ++half) {
                const HVX_VectorPair hw = Q6_Wuh_vunpack_Vub(half ? Q6_V_hi_W(cb) : Q6_V_lo_W(cb));
                for (int x = 0; x < 2; ++x) {
                    const HVX_Vector c = x ? Q6_V_hi_W(hw) : Q6_V_lo_W(hw);
                    const HVX_Vector lv = Q6_Vh_vsub_VhVh(q4i ? c : Q6_Vh_vadd_VhVh(c, c), k15);
                    *out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(lv), sc));
                }
            }
        }
    }
}

// the same, one value at a time (the check's reference): column o, input c
static inline float hmx4_value(const uint8_t *src, int k, int o, int c) {
    const uint8_t b = src[(uint32_t)(c >> 1) * 32u + (uint32_t)o];
    const int code = (c & 1) ? b >> 4 : b & 15;
    const __fp16 *sc = (const __fp16 *)(src + (uint32_t)k * 16u);
    return (float)(__fp16)((float)(2 * code - 15) * (float)sc[(c >> 4) * 32 + o]);
}

// ---- the q4 pipeline: worker threads expand blocks into a ring of VTCM slots, the caller multiplies them -------
#define PIPE_RING 4
#define PIPE_WORKERS 3
#define PIPE_STACK 16384

typedef struct {
    uint8_t *slots[PIPE_RING], *stage[PIPE_WORKERS];
    const uint8_t *wddr;
    uint32_t q_blk;
    int k, total, n_blk, nw;
    volatile int ready[PIPE_RING];   // the block a slot holds, once expanded
    volatile int consumed;           // the last block the multiplier is done with
    volatile int failed;             // a worker got no HVX context
    int no_hmx, workers;             // experiments: expansion alone; fewer workers
    uint8_t *oddr;                   // gemm: every result crouton copied out, [row tile][block] x 2 KB
    int tile0, n_tiles_all;          // gemm: this pass's first row tile, the whole result's row tiles
    int q4i;                         // the codes are q4i's (dequant_block)
} pipe_t;

typedef struct {
    pipe_t *p;
    int w;
} pipe_arg_t;

static void pipe_worker(void *arg) {
    pipe_arg_t *a = (pipe_arg_t *)arg;
    pipe_t *p = a->p;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        __atomic_store_n(&p->failed, 1, __ATOMIC_RELEASE);
        qurt_thread_exit(0);
    }
    for (int b = a->w; b < p->total; b += p->nw) {
        // slot b % RING last held block b - RING: free once the multiplier is past it
        while (__atomic_load_n(&p->consumed, __ATOMIC_ACQUIRE) < b - PIPE_RING) {
        }
        hvx_copy(p->stage[a->w], p->wddr + (size_t)(b % p->n_blk) * p->q_blk, p->q_blk);
        dequant_block(p->slots[b % PIPE_RING], p->stage[a->w], p->k, p->q4i);
        __atomic_store_n(&p->ready[b % PIPE_RING], b, __ATOMIC_RELEASE);
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

// The workers' stacks, made once and kept: a product allocating and freeing them each time (a prompt makes ~170 a
// call) let the heap hand its free sections back and fetch them again - which made the products up to 3x slower,
// depending on how everything else happened to lie in the heap
static void *g_pipe_stacks[PIPE_WORKERS];

static int pipe_run(pipe_t *p, const uint8_t *A, uint8_t *O, int m, int croutons) {
    pipe_arg_t args[PIPE_WORKERS];
    qurt_thread_t tid[PIPE_WORKERS];
    int prio = qurt_thread_get_priority(qurt_thread_get_id());   // the RPC thread's; a created one must be below 255
    if (prio <= 0 || prio >= 255) prio = 127;
    void *const *stacks = g_pipe_stacks;
    int started = 0;
    for (int w = 0; w < p->nw; ++w) {
        qurt_thread_attr_t at;
        qurt_thread_attr_init(&at);
        if (!g_pipe_stacks[w] && !(g_pipe_stacks[w] = memalign(128, PIPE_STACK))) break;
        qurt_thread_attr_set_name(&at, (char *)"geode_pipe");
        qurt_thread_attr_set_stack_addr(&at, stacks[w]);
        qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
        qurt_thread_attr_set_priority(&at, (unsigned short)prio);
        args[w].p = p;
        args[w].w = w;
        const int rc = qurt_thread_create(&tid[w], &at, pipe_worker, &args[w]);
        if (rc != QURT_EOK) break;
        ++started;
    }
    if (started < p->nw) {   // not every worker: let the started ones finish their share, then fail
        p->consumed = p->total;
        for (int w = 0; w < started; ++w) {
            int st;
            qurt_thread_join(tid[w], &st);
        }
        return AEE_ENOTALLOWED;
    }
    const int kk = croutons * 32;
    for (int b = 0; b < p->total; ++b) {
        while (__atomic_load_n(&p->ready[b % PIPE_RING], __ATOMIC_ACQUIRE) != b) {
            if (__atomic_load_n(&p->failed, __ATOMIC_ACQUIRE)) break;
        }
        if (p->failed) break;
        const uint8_t *wb = p->slots[b % PIPE_RING];
        if (!p->no_hmx)
            for (int t = 0; t < m / 32; ++t) {
                tile(A + (size_t)t * 32u * (size_t)kk * 2u, wb, O + (size_t)t * 2048u, croutons);
                if (p->oddr)   // the crouton out to its place in the result
                    hvx_copy(p->oddr + ((size_t)(p->tile0 + t) * (size_t)p->n_blk + (size_t)(b % p->n_blk)) * 2048u,
                             O + (size_t)t * 2048u, 2048u);
            }
        __atomic_store_n(&p->consumed, b - 1, __ATOMIC_RELEASE);   // one behind: HMX may still read this slot
    }
    __atomic_store_n(&p->consumed, p->total, __ATOMIC_RELEASE);
    for (int w = 0; w < p->nw; ++w) {
        int st;
        qurt_thread_join(tid[w], &st);
    }
    return p->failed ? AEE_ENOTALLOWED : 0;
}

int geode_npu_hmx_gemm(remote_handle64 h, int mode, int m, int k, int n, int reps, int check, int64_t *usec,
                       int64_t *err_ppm) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    *usec = 0;
    *err_ppm = -1;
    if (m <= 0 || k <= 0 || n <= 0 || (m & 31) || (k & 31) || (n & 31) || reps <= 0) return AEE_EBADPARM;
    int e = acquire(s);
    if (e) return e;
    const int croutons = k / 32;
    const uint32_t act_bytes = (uint32_t)m * (uint32_t)k * 2u;   // every row tile of the activations
    const uint32_t blk_bytes = (uint32_t)k * 64u;                // one 32-column weight block
    // VTCM: activations | bias (256 B) | outputs ring (one crouton per row tile) | weight blocks
    const uint32_t out_bytes = (uint32_t)(m / 32) * 2048u;
    uint32_t used = act_bytes + 2048u + out_bytes;
    if (used + blk_bytes * (mode >= 3 ? PIPE_RING : 2) + HMX4_BYTES(k) * (mode >= 3 ? PIPE_WORKERS : 1) > s->vtcm_size)
        return AEE_ENOMEMORY;
    int n_blk_vtcm = (int)((s->vtcm_size - used) / blk_bytes);
    if (n_blk_vtcm > n / 32) n_blk_vtcm = n / 32;
    if (mode >= 1 && n_blk_vtcm > 2) n_blk_vtcm = 2;   // streaming: a pair of blocks, one filled while one is used
    uint8_t *A = s->vtcm;
    uint8_t *bias = A + act_bytes;
    uint8_t *O = bias + 2048;
    uint8_t *W = O + out_bytes;
    __fp16 *a16 = (__fp16 *)A;
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < k; ++c) a16[(uint32_t)(r >> 5) * (uint32_t)k * 32u + act_index(r & 31, c)] =
            (__fp16)val((uint32_t)r, (uint32_t)c, 1);
    // weights in DDR, already in the block layout (what a pack would hold)
    uint8_t *wddr = NULL;
    const int n_blk = n / 32;
    const uint32_t q_blk = HMX4_BYTES(k);
    if (mode >= 2) {   // q4 in DDR, expanded into VTCM by HVX block by block
        wddr = (uint8_t *)memalign(128, (size_t)q_blk * (size_t)n_blk);
        if (!wddr) return AEE_ENOMEMORY;
        for (int b = 0; b < n_blk; ++b) {
            uint8_t *q = wddr + (size_t)b * q_blk;
            for (uint32_t i = 0; i < (uint32_t)k * 16u; ++i) q[i] = (uint8_t)(val((uint32_t)b, i, 3) * 128.0f + 128.0f);
            __fp16 *sc = (__fp16 *)(q + (uint32_t)k * 16u);
            for (int i = 0; i < k * 2; ++i) sc[i] = (__fp16)((1.5f + val((uint32_t)b, (uint32_t)i, 5)) / 256.0f);
        }
    } else if (mode == 1 || n_blk_vtcm < n_blk) {
        wddr = (uint8_t *)memalign(128, (size_t)blk_bytes * (size_t)n_blk);
        if (!wddr) return AEE_ENOMEMORY;
    }
    for (int b = 0; b < n_blk && mode < 2; ++b) {   // fp16 weights (modes 0, 1)
        __fp16 *w16 = wddr ? (__fp16 *)(wddr + (size_t)b * blk_bytes) : (__fp16 *)(W + (size_t)b * blk_bytes);
        for (int o = 0; o < 32; ++o)
            for (int c = 0; c < k; ++c) w16[wgt_index(o, c)] = (__fp16)(val((uint32_t)(b * 32 + o), (uint32_t)c, 7) / 16.0f);
    }
    if (wddr && mode == 0)   // HMX alone: as many blocks in VTCM as fit, cycled
        for (int b = 0; b < n_blk_vtcm; ++b) hvx_copy(W + (size_t)b * blk_bytes, wddr + (size_t)b * blk_bytes, blk_bytes);
    memset(bias, 0, 2048);
    {
        uint32_t *lo = (uint32_t *)bias, *hi = (uint32_t *)(bias + 128);
        for (int ch = 0; ch < 32; ++ch) {
            lo[ch] = 0x00003C00u;   // output bias 0, scale 1.0
            hi[ch] = 0;
        }
    }
    Q6_bias_mxmem2_A(bias);

    const uint64_t t0 = HAP_perf_get_time_us();
    if (mode >= 3) {   // q4 expanded by worker threads into a ring of slots while this thread runs HMX
        pipe_t p;
        memset(&p, 0, sizeof p);
        p.wddr = wddr;
        p.q_blk = q_blk;
        p.k = k;
        p.total = reps * n_blk;
        p.n_blk = n_blk;
        p.nw = mode == 5 ? 1 : PIPE_WORKERS;
        p.no_hmx = mode == 4;
        for (int i = 0; i < PIPE_RING; ++i) {
            p.slots[i] = W + (size_t)i * blk_bytes;
            p.ready[i] = -1;
        }
        for (int i = 0; i < PIPE_WORKERS; ++i) p.stage[i] = W + (size_t)PIPE_RING * blk_bytes + (size_t)i * q_blk;
        p.consumed = -1;
        e = pipe_run(&p, A, O, m, croutons);
        *usec = (int64_t)(HAP_perf_get_time_us() - t0);
        if (e) {
            free(wddr);
            return e;
        }
        reps = 0;   // done
    }
    for (int rep = 0; rep < reps; ++rep) {
        for (int b = 0; b < n_blk; ++b) {
            uint8_t *wb;
            if (mode == 2) {   // the q4 block into VTCM first (a streamed copy), then expanded there
                wb = W + (size_t)(b & 1) * blk_bytes;
                uint8_t *qv = W + 2 * (size_t)blk_bytes;
                hvx_copy(qv, wddr + (size_t)b * q_blk, q_blk);
                dequant_block(wb, qv, k, 0);
            } else if (mode == 1) {
                wb = W + (size_t)(b & 1) * blk_bytes;
                hvx_copy(wb, wddr + (size_t)b * blk_bytes, blk_bytes);
            } else {
                wb = W + (size_t)(b % n_blk_vtcm) * blk_bytes;
            }
            for (int t = 0; t < m / 32; ++t) tile(A + (size_t)t * 32u * (size_t)k * 2u, wb, O + (size_t)t * 2048u, croutons);
        }
    }
    if (mode < 3) *usec = (int64_t)(HAP_perf_get_time_us() - t0);

    if (check) {   // the last weight block's results (n_blk - 1), every row tile, against a float sum of the same fp16s
        const int b = n_blk - 1;
        const uint8_t *wsrc = mode == 2 ? W : (mode == 1 || n_blk_vtcm < n_blk) ? wddr + (size_t)b * blk_bytes
                                                                 : W + (size_t)(b % n_blk_vtcm) * blk_bytes;
        if (mode == 0 && n_blk_vtcm < n_blk) wsrc = W + (size_t)(b % n_blk_vtcm) * blk_bytes;   // what was used
        const __fp16 *w16 = (const __fp16 *)wsrc;
        // the result read in three candidate layouts: the crouton's (r/2)*64 + o*2 + r%2, row-major, column-major;
        // err_ppm = the best layout's worst error (ppm) + 1e12 x that layout's number
        double worst[3] = {0, 0, 0};
        for (int t = 0; t < m / 32; ++t) {
            const __fp16 *o16 = (const __fp16 *)(O + (size_t)t * 2048u);
            const __fp16 *at = a16 + (size_t)t * 32u * (size_t)k;
            for (int r = 0; r < 32; ++r)
                for (int o = 0; o < 32; ++o) {
                    float ref = 0, mag = 0;
                    for (int c = 0; c < k; ++c) {
                        const float wv = mode >= 2 ? hmx4_value(wddr + (size_t)b * q_blk, k, o, c)
                                                   : (float)w16[wgt_index(o, c)];
                        const float p = (float)at[act_index(r, c)] * wv;
                        ref += p;
                        mag += p < 0 ? -p : p;
                    }
                    const int idx[3] = {(r >> 1) * 64 + o * 2 + (r & 1), r * 32 + o, o * 32 + r};
                    for (int l = 0; l < 3; ++l) {
                        const float got = (float)o16[idx[l]];
                        const double rel = (double)((got - ref) < 0 ? ref - got : got - ref) /
                                           (double)(mag > 1e-6f ? mag : 1e-6f);
                        if (rel > worst[l]) worst[l] = rel;
                    }
                }
        }
        int best = 0;
        for (int l = 1; l < 3; ++l)
            if (worst[l] < worst[best]) best = l;
        *err_ppm = (int64_t)(worst[best] * 1e6) + (int64_t)best * 1000000000000LL;
    }
    free(wddr);
    return 0;
}

// ---- the real product ------------------------------------------------------------------------------------------
// O = A x W^T (geode_npu.idl): A's row tiles in passes as many as VTCM holds beside the q4 pipeline's ring; each pass
// expands every weight block once and multiplies all of its row tiles; each result crouton goes out as it is made.
// O = A x W^T on HMX with the weights at w (hmx4 blocks, mapped already): dec.c's prompt calls it on the thread that
// holds HMX (its FastRPC thread). A, O in DDR, 128-byte aligned, in HMX's tile / crouton layouts.
int geode_npu_gemm_ptr(remote_handle64 h, const uint8_t *a, const uint8_t *w, uint8_t *o, int m, int k, int n,
                       int out_shift) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    const int q4i = (out_shift & HMX4_Q4I) != 0;
    out_shift &= ~HMX4_Q4I;
    if (m <= 0 || k <= 0 || n <= 0 || (m & 31) || (k & 31) || (n & 31) || out_shift < 0 || out_shift > 24 ||
        ((uintptr_t)a & 127) || ((uintptr_t)o & 127) || ((uintptr_t)w & 127))
        return AEE_EBADPARM;
    // the output scale 2^-out_shift in fp16: normal down to 2^-14, then subnormal
    const uint32_t out_scale = out_shift <= 14 ? (uint32_t)(15 - out_shift) << 10 : 1u << (24 - out_shift);
    int e = acquire(s);
    if (e) return e;
    const int croutons = k / 32, n_blk = n / 32;
    const uint32_t blk_bytes = (uint32_t)k * 64u, q_blk = HMX4_BYTES(k), tile_bytes = (uint32_t)k * 64u;
    // VTCM: bias | ring of expanded blocks | the workers' staging | per pass: output croutons + activation tiles
    const uint32_t fixed = 2048u + blk_bytes * PIPE_RING + q_blk * PIPE_WORKERS;
    if (fixed + 2048u + tile_bytes > s->vtcm_size) return AEE_ENOMEMORY;
    int pass_tiles = (int)((s->vtcm_size - fixed) / (tile_bytes + 2048u));
    if (pass_tiles > m / 32) pass_tiles = m / 32;
    uint8_t *bias = s->vtcm;
    uint8_t *W = bias + 2048;
    uint8_t *O = W + (size_t)blk_bytes * PIPE_RING + (size_t)q_blk * PIPE_WORKERS;
    uint8_t *A = O + (size_t)pass_tiles * 2048u;
    memset(bias, 0, 2048);
    {
        uint32_t *lo = (uint32_t *)bias;
        for (int ch = 0; ch < 32; ++ch) lo[ch] = out_scale;   // output bias 0, scale 2^-out_shift
    }
    Q6_bias_mxmem2_A(bias);
    for (int t0_tile = 0; t0_tile < m / 32 && !e; t0_tile += pass_tiles) {
        const int tiles = m / 32 - t0_tile < pass_tiles ? m / 32 - t0_tile : pass_tiles;
        hvx_copy(A, a + (size_t)t0_tile * tile_bytes, (uint32_t)tiles * tile_bytes);
        pipe_t p;
        memset(&p, 0, sizeof p);
        p.wddr = w;
        p.q_blk = q_blk;
        p.k = k;
        p.total = n_blk;
        p.n_blk = n_blk;
        p.nw = PIPE_WORKERS;
        p.oddr = o;
        p.tile0 = t0_tile;
        p.n_tiles_all = m / 32;
        p.q4i = q4i;
        for (int i = 0; i < PIPE_RING; ++i) {
            p.slots[i] = W + (size_t)i * blk_bytes;
            p.ready[i] = -1;
        }
        for (int i = 0; i < PIPE_WORKERS; ++i) p.stage[i] = W + (size_t)PIPE_RING * blk_bytes + (size_t)i * q_blk;
        p.consumed = -1;
        e = pipe_run(&p, A, O, tiles * 32, croutons);
    }
    return e;
}

int geode_npu_gemm(remote_handle64 h, const unsigned char *a, int a_len, int w_fd, uint64_t w_off, unsigned char *o,
                   int o_len, int m, int k, int n, int out_shift, int64_t *usec) {
    *usec = 0;
    if (m <= 0 || k <= 0 || n <= 0 || (uint32_t)a_len < (uint32_t)m * (uint32_t)k * 2u ||
        (uint32_t)o_len < (uint32_t)m * (uint32_t)n * 2u)
        return AEE_EBADPARM;
    void *wv = NULL;
    uint64 pa = 0;
    int e = HAP_mmap_get(w_fd, &wv, &pa);
    if (e) return e;
    const uint64_t t0 = HAP_perf_get_time_us();
    e = geode_npu_gemm_ptr(h, a, (const uint8_t *)wv + w_off, o, m, k, n, out_shift);
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    HAP_mmap_put(w_fd);
    return e;
}

// ---- DDR read speed (bw): what a whole model's decode on the DSP would stream from DDR each token
#define BW_CHUNK 65536u
#define BW_THREADS 6

typedef struct {
    const uint8_t *src;
    uint8_t *dst;   // 64 KB of VTCM of its own
    uint64_t bytes;
    int mode;
    volatile int failed;
    uint32_t sum;
} bw_arg_t;

static void bw_worker(void *arg) {
    bw_arg_t *a = (bw_arg_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        a->failed = 1;
        qurt_thread_exit(0);
    }
    HVX_Vector acc = Q6_V_vzero();
    for (uint64_t off = 0; off < a->bytes; off += BW_CHUNK) {
        const uint32_t n = a->bytes - off < BW_CHUNK ? (uint32_t)(a->bytes - off) : BW_CHUNK;
        const HVX_Vector *p = (const HVX_Vector *)(a->src + off);
        if (a->mode == 2 && off + BW_CHUNK < a->bytes)   // the next 64 KB: 64 rows of 1 KB (dir 0, stride, width, height)
            Q6_l2fetch_AP((void *)(a->src + off + BW_CHUNK), ((uint64_t)1024u << 32) | (1024u << 16) | 64u);
        if (a->mode == 1) {
            for (uint32_t i = 0; i < n / 128; ++i) acc = Q6_Vw_vadd_VwVw(acc, p[i]);
        } else {
            HVX_Vector *d = (HVX_Vector *)a->dst;
            for (uint32_t i = 0; i < n / 128; ++i) d[i] = p[i];
        }
    }
    *(HVX_Vector *)a->dst = acc;   // kept, so the loads stay
    a->sum = *(uint32_t *)a->dst;
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int geode_npu_bw(remote_handle64 h, int w_fd, uint64_t w_off, uint64_t len, int threads, int mode, int64_t *usec) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    *usec = 0;
    if (threads < 1 || threads > BW_THREADS || mode < 0 || mode > 2 || len < BW_CHUNK) return AEE_EBADPARM;
    int e = acquire(s);
    if (e) return e;
    if ((uint32_t)threads * BW_CHUNK > s->vtcm_size) return AEE_ENOMEMORY;
    void *wv = NULL;
    uint64 pa = 0;
    if ((e = HAP_mmap_get(w_fd, &wv, &pa))) return e;
    const uint8_t *src = (const uint8_t *)wv + w_off;
    if ((uintptr_t)src & 127) {
        HAP_mmap_put(w_fd);
        return AEE_EBADPARM;
    }
    const uint64_t share = len / (uint64_t)threads / BW_CHUNK * BW_CHUNK;   // whole chunks, each thread its own run
    bw_arg_t args[BW_THREADS];
    qurt_thread_t tid[BW_THREADS];
    void *stacks[BW_THREADS] = {0};
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio <= 0 || prio >= 255) prio = 127;
    int started = 0;
    const uint64_t t0 = HAP_perf_get_time_us();
    for (int w = 0; w < threads; ++w) {
        args[w] = (bw_arg_t){src + (uint64_t)w * share, s->vtcm + (size_t)w * BW_CHUNK, share, mode, 0, 0};
        qurt_thread_attr_t at;
        qurt_thread_attr_init(&at);
        if (!(stacks[w] = memalign(128, PIPE_STACK))) break;
        qurt_thread_attr_set_name(&at, (char *)"geode_bw");
        qurt_thread_attr_set_stack_addr(&at, stacks[w]);
        qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
        qurt_thread_attr_set_priority(&at, (unsigned short)prio);
        if (qurt_thread_create(&tid[w], &at, bw_worker, &args[w]) != QURT_EOK) break;
        ++started;
    }
    int failed = started < threads;
    for (int w = 0; w < started; ++w) {
        int st;
        qurt_thread_join(tid[w], &st);
        failed |= args[w].failed;
    }
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    for (int w = 0; w < BW_THREADS; ++w) free(stacks[w]);
    HAP_mmap_put(w_fd);
    return failed ? AEE_ENOTALLOWED : 0;
}

// thread experiments for the pipeline: 10 a thread that returns, 11 + HVX lock, 12 + HVX writes into VTCM
static volatile int g_probe_state;
static uint8_t *g_probe_vtcm;
static void probe_thread(void *arg) {
    const int what = (int)(intptr_t)arg;
    g_probe_state = 1;
    if (what >= 11) {
        const int rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
        g_probe_state = rc == QURT_EOK ? 2 : 100 + rc;
        if (rc == QURT_EOK && what >= 12) {
            HVX_Vector *v = (HVX_Vector *)g_probe_vtcm;
            v[0] = Q6_Vb_vsplat_R(7);
            g_probe_state = 3;
        }
        if (rc == QURT_EOK && what >= 13) {   // a copy from the DDR heap into VTCM
            uint8_t *d = (uint8_t *)memalign(128, 40960);
            if (d) {
                memset(d, 1, 40960);
                hvx_copy(g_probe_vtcm + 65536, d, 40960);
                free(d);
                g_probe_state = 4;
            }
        }
        if (rc == QURT_EOK && what >= 14) {   // the expansion, VTCM to VTCM
            dequant_block(g_probe_vtcm + 131072, g_probe_vtcm + 65536, 2048, 0);
            g_probe_state = 5;
        }
        if (rc == QURT_EOK) qurt_hvx_unlock();
    }
    if (what == 15) {   // the caller spins on this
        for (volatile int i = 0; i < 2000000; ++i) {
        }
        __atomic_store_n(&g_probe_state, 77, __ATOMIC_RELEASE);
    }
    g_probe_state += 1000;
    qurt_thread_exit(0);
}

static int probe_threads(npu_t *s, int what, int *state) {
    g_probe_state = 0;
    g_probe_vtcm = s->vtcm;
    void *stack = memalign(128, PIPE_STACK);
    if (!stack) return AEE_ENOMEMORY;
    qurt_thread_attr_t at;
    qurt_thread_attr_init(&at);
    qurt_thread_attr_set_name(&at, (char *)"geode_probe");
    qurt_thread_attr_set_stack_addr(&at, stack);
    qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
    qurt_thread_attr_set_priority(&at, 127);
    qurt_thread_t tid;
    int rc = qurt_thread_create(&tid, &at, probe_thread, (void *)(intptr_t)what);
    if (rc == QURT_EOK && what == 15)
        while (__atomic_load_n(&g_probe_state, __ATOMIC_ACQUIRE) < 77) {
        }
    if (rc == QURT_EOK) {
        int st;
        qurt_thread_join(tid, &st);
    }
    *state = g_probe_state;
    free(stack);
    return rc;
}

int geode_npu_hmx_debug(remote_handle64 h, int pattern, unsigned char *out, int out_len) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    int e = acquire(s);
    if (e) return e;
    if (out_len < 2048) return AEE_EBADPARM;
    if (pattern >= 10 && pattern < 20) {   // thread experiments: the result's first ints = rc, state
        int st = -1;
        const int rc = probe_threads(s, pattern, &st);
        memset(out, 0, 2048);
        ((int *)out)[0] = rc;
        ((int *)out)[1] = st;
        return 0;
    }
    const int cvt_rs = pattern >> 8;   // the convert instruction's Rs (bits 13:12 bias set, 0 acc clear, ...)
    pattern &= 0xFF;
    uint8_t *A = s->vtcm, *Wb = A + 2048, *bias = Wb + 2048, *O = bias + 2048;
    __fp16 *a16 = (__fp16 *)A, *w16 = (__fp16 *)Wb;
    memset(A, 0, 2048);
    for (int r = 0; r < 32; ++r) a16[act_index(r, r)] = (__fp16)1.0f;
    for (int o = 0; o < 32; ++o)
        for (int c = 0; c < 32; ++c) w16[wgt_index(o, c)] = (__fp16)(float)(o * 32 + c);
    if (pattern == 2) {   // the plainest: every activation 1 -> result[r][o] = sum_c w[o][c]
        for (int i = 0; i < 1024; ++i) a16[i] = (__fp16)1.0f;
    }
    if (pattern == 3) {   // hmx_gemm's own values (val salts 1 and 7, weights / 16), its first tile
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 32; ++c) a16[act_index(r, c)] = (__fp16)val((uint32_t)r, (uint32_t)c, 1);
        for (int o = 0; o < 32; ++o)
            for (int c = 0; c < 32; ++c) w16[wgt_index(o, c)] = (__fp16)(val((uint32_t)o, (uint32_t)c, 7) / 16.0f);
    }
    memset(bias, 0, 2048);
    uint32_t *lo = (uint32_t *)bias, *hi = (uint32_t *)(bias + 128);
    for (int ch = 0; ch < 32; ++ch) { lo[ch] = 0x00003C00u; hi[ch] = 0; }
    memset(O, 0, 2048);
    Q6_bias_mxmem2_A(bias);
    Q6_activation_hf_mxmem_RR_deep((uint32_t)(uintptr_t)A, 0x7C);
    Q6_weight_hf_mxmem_RR((uint32_t)(uintptr_t)Wb, 2047u);
    Q6_cvt_hf_acc_R(cvt_rs);
    Q6_mxmem_cvt_RR((void *)O, 0);
    memcpy(out, O, 2048);
    return 0;
}

// ---- dspqueue: the low-latency path the engine's per-layer jobs would take ------------------------------------
static void queue_packet(dspqueue_t q, AEEResult err, void *ctx) {
    (void)err;
    (void)ctx;
    for (;;) {
        uint32_t flags, n_bufs, len, msg[4];
        struct dspqueue_buffer bufs[1];
        if (dspqueue_read_noblock(q, &flags, 1, &n_bufs, bufs, sizeof msg, &len, (uint8_t *)msg)) return;
        msg[0] += 1;
        dspqueue_write(q, 0, 0, NULL, len, (const uint8_t *)msg, DSPQUEUE_TIMEOUT_NONE);
    }
}

static void queue_error(dspqueue_t q, AEEResult err, void *ctx) {
    (void)q;
    (void)err;
    (void)ctx;
}

int geode_npu_queue_start(remote_handle64 h, uint64_t queue_id) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    if (s->queue) return AEE_EITEMBUSY;
    return dspqueue_import(queue_id, queue_packet, queue_error, s, &s->queue);
}

int geode_npu_queue_stop(remote_handle64 h) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    if (!s->queue) return 0;
    const int e = dspqueue_close(s->queue);
    s->queue = NULL;
    return e;
}

// ---- decode products: y = W x for one token, every matrix of a step in one call (geode_npu_gemv) ------------------
// The weights are the hmx4 blocks the prompt path multiplies on HMX (one copy serves both), read straight from DDR by
// HVX threads, each its own run of 32-column blocks of every matrix. Per block, per 16 inputs (2 code vectors of 4
// input pairs x 32 columns): sum_i code_i * x'_i in qf16 with the lanes' two halves holding different inputs, folded
// into one, minus 7.5 * sum_i x'_i (level = 2 code - 15, x' = 2 x / 2^shift), times the group's 32 scales in qf32.
// x' comes per matrix from a table of splats each thread builds in its own VTCM: for code vector j, 4 vectors (low
// nibbles of pairs 0-1, of pairs 2-3, high nibbles of pairs 0-1, of pairs 2-3), lanes 0-31 one input, 32-63 another.
#define GEMV_THREADS 6
#define GEMV_KMAX 8192
typedef struct {
    uint64_t w_off;   // the matrix's first hmx4 block in the mapped buffer
    uint32_t k, n;    // inputs, outputs (both multiples of 32; n / 32 blocks)
    uint32_t x_at;    // its input: fp16 x[x_at .. x_at + k)
    uint32_t y_at;    // its output: float y[y_at .. y_at + n)
    int32_t shift;    // x / 2^shift keeps the qf16 sums in range (the host's choice: max|x| / 2^shift <= 64)
    uint32_t pad;
} gemv_desc_t;

typedef struct {
    const uint8_t *w;
    const gemv_desc_t *d;
    int count, w_id, threads;
    const __fp16 *x;
    float *y;
    uint8_t *xv;   // this thread's VTCM: GEMV_KMAX / 8 x 4 vectors
    volatile int failed;
} gemv_arg_t;

static inline HVX_Vector splat16(uint16_t v) {   // every halfword lane = v (a word splat of both halves)
    return Q6_V_vsplat_R((int)((uint32_t)v | ((uint32_t)v << 16)));
}

static inline float pow2f(int e) {   // 2^e for |e| < 31
    return e >= 0 ? (float)(1u << e) : 1.0f / (float)(1u << -e);
}

static void gemv_worker(void *arg) {
    gemv_arg_t *a = (gemv_arg_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        a->failed = 1;
        qurt_thread_exit(0);
    }
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_VectorPred lo32 = Q6_Q_vsetq_R(64);   // the first 32 halfword lanes
    // this thread's VTCM: per code vector j 4 int16 splat-pair vectors of xq; per group the correction 15 * sum xq
    // (int16 splat) and the group's activation scale (fp16 splat)
    HVX_Vector *xv = (HVX_Vector *)a->xv;
    HVX_Vector *kv = xv + GEMV_KMAX / 8 * 4, *sxv = kv + GEMV_KMAX / 16;
    uint8_t *copy = (uint8_t *)(sxv + GEMV_KMAX / 16);   // a block copied in first: HVX loads straight from DDR stall
    static const int sel[4][2] = {{0, 2}, {4, 6}, {1, 3}, {5, 7}};   // L01, L23, H01, H23: the inputs of each half
    for (int mi = 0; mi < a->count; ++mi) {
        const gemv_desc_t *d = &a->d[mi];
        const int k = (int)d->k, n_blk = (int)d->n / 32;
        const int b0 = n_blk * a->w_id / a->threads, b1 = n_blk * (a->w_id + 1) / a->threads;
        if (b0 >= b1) continue;
        // x as int8 per 16 inputs (the engine's int8 path's activations): xq = round(x / s), s = max|x| / 127
        const __fp16 *x = a->x + d->x_at;
        for (int g = 0; g < k / 16; ++g) {
            float mx = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float v = (float)x[16 * g + i];
                mx = v > mx ? v : (-v > mx ? -v : mx);
            }
            const float sc = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / sc;
            int16_t q[16];
            int sum = 0;
            for (int i = 0; i < 16; ++i) {
                const float v = (float)x[16 * g + i] * inv;
                q[i] = (int16_t)(v >= 0.0f ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
                sum += q[i];
            }
            for (int h = 0; h < 2; ++h) {   // the group's two code vectors
                const int j = 2 * g + h;
                for (int s4 = 0; s4 < 4; ++s4)
                    xv[j * 4 + s4] = Q6_V_vmux_QVV(lo32, splat16((uint16_t)q[8 * h + sel[s4][0]]),
                                                   splat16((uint16_t)q[8 * h + sel[s4][1]]));
            }
            kv[g] = splat16((uint16_t)(int16_t)(15 * sum));
            union { __fp16 h; uint16_t u; } us = {(__fp16)sc};
            sxv[g] = splat16(us.u);
        }
        for (int b = b0; b < b1; ++b) {
            const uint8_t *blk = a->w + d->w_off + (uint64_t)b * HMX4_BYTES(k);
            hvx_copy(copy, blk, HMX4_BYTES(k));
            const HVX_Vector *cv = (const HVX_Vector *)copy;
            const HVX_Vector *sv = (const HVX_Vector *)(copy + (uint32_t)k * 16u);
            HVX_Vector acc16 = Q6_V_vzero();   // qf16
            float accf[64];
            for (int i = 0; i < 64; ++i) accf[i] = 0.0f;
            for (int g2 = 0; g2 < k / 32; ++g2) {
                HVX_Vector t[2];
                for (int gi = 0; gi < 2; ++gi) {
                    const int g = 2 * g2 + gi, j0 = 2 * g, j1 = 2 * g + 1;
                    const HVX_Vector v0 = cv[j0], v1 = cv[j1];
                    const HVX_VectorPair l0 = Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(v0, m0f));
                    const HVX_VectorPair h0 = Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v0, 4), m0f));
                    const HVX_VectorPair l1 = Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(v1, m0f));
                    const HVX_VectorPair h1 = Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v1, 4), m0f));
                    const HVX_Vector *x0 = xv + j0 * 4, *x1 = xv + j1 * 4;
                    // four independent sums of code * xq (int16: at most 15 * 127 * 4 a lane each)
                    HVX_Vector s0 = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(l0), x0[0]);
                    HVX_Vector s1 = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(l0), x0[1]);
                    HVX_Vector s2 = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(h0), x0[2]);
                    HVX_Vector s3 = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(h0), x0[3]);
                    s0 = Q6_Vh_vmpyiacc_VhVhVh(s0, Q6_V_lo_W(l1), x1[0]);
                    s1 = Q6_Vh_vmpyiacc_VhVhVh(s1, Q6_V_hi_W(l1), x1[1]);
                    s2 = Q6_Vh_vmpyiacc_VhVhVh(s2, Q6_V_lo_W(h1), x1[2]);
                    s3 = Q6_Vh_vmpyiacc_VhVhVh(s3, Q6_V_hi_W(h1), x1[3]);
                    const HVX_Vector s = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(s0, s1), Q6_Vh_vadd_VhVh(s2, s3));
                    const HVX_Vector f = Q6_Vh_vadd_VhVh(s, Q6_V_vror_VR(s, 64));   // the group's 16 inputs
                    // sum (2 code - 15) xq = 2 f - 15 sum xq: the int16 wrap of 2 f cancels (the result fits)
                    const HVX_Vector dot = Q6_Vh_vsub_VhVh(Q6_Vh_vadd_VhVh(f, f), kv[g]);
                    t[gi] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(dot), sxv[g]));
                }
                const HVX_Vector u = Q6_V_vmux_QVV(lo32, t[0], t[1]);   // lanes 0-31 group 2g2, 32-63 group 2g2+1
                // times the scales lane by lane (qf16), summed 8 group pairs at a time, then into floats
                acc16 = Q6_Vqf16_vadd_Vqf16Vqf16(acc16, Q6_Vqf16_vmpy_VhfVhf(u, sv[g2]));
                if ((g2 & 7) == 7 || g2 == k / 32 - 1) {
                    const HVX_Vector hv = Q6_Vhf_equals_Vqf16(acc16);
                    __fp16 hs[64] __attribute__((aligned(128)));
                    memcpy(hs, &hv, 128);   // (memcpy: a vector read through another type is undefined)
                    for (int i = 0; i < 64; ++i) accf[i] += (float)hs[i];
                    acc16 = Q6_V_vzero();
                }
            }
            float *y = a->y + d->y_at + (uint32_t)b * 32u;
            for (int c = 0; c < 32; ++c) y[c] = accf[c] + accf[c + 32];
        }
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

// Variant 1 (geode_npu_gemv's threads >> 8 == 1, a measuring experiment): the same sums with vrmpy, 4 int8 products
// a 32-bit lane. A code vector (4 input pairs x 32 columns) is transposed in registers (vshuffb, then vshuffh) so each
// word holds one column's 4 pairs (pair order 0, 2, 1, 3), its low and high nibbles then multiply a word of 4 int8
// inputs each (a scalar operand); per 16 inputs: 2 sum(c x) - 15 sum(x) in int32, packed to int16 (it fits), times
// the activation's and the weights' scales in qf16 as variant 0.
static void gemv_worker_vrmpy(void *arg) {
    gemv_arg_t *a = (gemv_arg_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        a->failed = 1;
        qurt_thread_exit(0);
    }
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_VectorPred lo32 = Q6_Q_vsetq_R(64);   // the first 32 halfword lanes
    HVX_Vector *kv = (HVX_Vector *)a->xv;                       // per group: 15 * sum xq (word splat)
    HVX_Vector *act2 = kv + GEMV_KMAX / 16;                    // per group pair: the activation scales (hf)
    // per code vector the low and the high nibbles' 4 inputs: scalar operands, so in cached memory (a scalar load
    // from VTCM is slow)
    uint32_t *wlo = (uint32_t *)memalign(128, GEMV_KMAX / 8 * 2 * sizeof(uint32_t));
    if (!wlo) {
        a->failed = 1;
        qurt_hvx_unlock();
        qurt_thread_exit(0);
    }
    uint32_t *whi = wlo + GEMV_KMAX / 8;
    uint8_t *copy = (uint8_t *)(act2 + GEMV_KMAX / 32);
    for (int mi = 0; mi < a->count; ++mi) {
        const gemv_desc_t *d = &a->d[mi];
        const int k = (int)d->k, n_blk = (int)d->n / 32;
        const int b0 = n_blk * a->w_id / a->threads, b1 = n_blk * (a->w_id + 1) / a->threads;
        if (b0 >= b1) continue;
        const __fp16 *x = a->x + d->x_at;
        __fp16 scs[GEMV_KMAX / 16];
        for (int g = 0; g < k / 16; ++g) {
            float mx = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float v = (float)x[16 * g + i];
                mx = v > mx ? v : (-v > mx ? -v : mx);
            }
            const float sc = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / sc;
            int8_t q[16];
            int sum = 0;
            for (int i = 0; i < 16; ++i) {
                const float v = (float)x[16 * g + i] * inv;
                q[i] = (int8_t)(v >= 0.0f ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
                sum += q[i];
            }
            for (int h = 0; h < 2; ++h) {   // code vector j = 2g + h: inputs 8 h .. 8 h + 7 of the group
                const int8_t *qq = q + 8 * h;
                const int j = 2 * g + h;
                wlo[j] = (uint32_t)(uint8_t)qq[0] | (uint32_t)(uint8_t)qq[4] << 8 | (uint32_t)(uint8_t)qq[2] << 16 |
                         (uint32_t)(uint8_t)qq[6] << 24;
                whi[j] = (uint32_t)(uint8_t)qq[1] | (uint32_t)(uint8_t)qq[5] << 8 | (uint32_t)(uint8_t)qq[3] << 16 |
                         (uint32_t)(uint8_t)qq[7] << 24;
            }
            kv[g] = Q6_V_vsplat_R(15 * sum);
            scs[g] = (__fp16)sc;
        }
        for (int g2 = 0; g2 < k / 32; ++g2) {
            union { __fp16 h; uint16_t u; } u0 = {scs[2 * g2]}, u1 = {scs[2 * g2 + 1]};
            act2[g2] = Q6_V_vmux_QVV(lo32, splat16(u0.u), splat16(u1.u));
        }
        for (int b = b0; b < b1; ++b) {
            const uint8_t *blk = a->w + d->w_off + (uint64_t)b * HMX4_BYTES(k);
            hvx_copy(copy, blk, HMX4_BYTES(k));
            const HVX_Vector *cv = (const HVX_Vector *)copy;
            const HVX_Vector *sv = (const HVX_Vector *)(copy + (uint32_t)k * 16u);
            HVX_Vector acc_e = Q6_V_vzero(), acc_o = Q6_V_vzero();   // qf32
            for (int g2 = 0; g2 < k / 32; ++g2) {
                HVX_Vector dot[2];
                for (int gi = 0; gi < 2; ++gi) {
                    HVX_Vector acc = Q6_V_vzero();
                    for (int hh = 0; hh < 2; ++hh) {
                        const int j = 4 * g2 + 2 * gi + hh;
                        const HVX_Vector t = Q6_Vh_vshuff_Vh(Q6_Vb_vshuff_Vb(cv[j]));
                        const HVX_Vector lo = Q6_V_vand_VV(t, m0f);
                        const HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(t, 4), m0f);
                        acc = Q6_Vw_vrmpyacc_VwVubRb(acc, lo, (int)wlo[j]);
                        acc = Q6_Vw_vrmpyacc_VwVubRb(acc, hi, (int)whi[j]);
                    }
                    dot[gi] = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(acc, acc), kv[2 * g2 + gi]);
                }
                const HVX_Vector packed = Q6_Vh_vpack_VwVw_sat(dot[1], dot[0]);   // lanes 0-31 group 2g2
                const HVX_Vector t = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(packed), act2[g2]));
                // times the weights' scales into qf32 (no flush to floats on the way: qf32 keeps the sum)
                const HVX_VectorPair pr = Q6_Wqf32_vmpy_VhfVhf(t, sv[g2]);
                acc_e = Q6_Vqf32_vadd_Vqf32Vqf32(acc_e, Q6_V_lo_W(pr));
                acc_o = Q6_Vqf32_vadd_Vqf32Vqf32(acc_o, Q6_V_hi_W(pr));
            }
            // the pair's even and odd halfword lanes: lane i < 16 of each is group 2g2's column 2i (+1), lane 16 + i
            // group 2g2 + 1's
            const HVX_Vector fe = Q6_Vsf_equals_Vqf32(acc_e), fo = Q6_Vsf_equals_Vqf32(acc_o);
            float se[32] __attribute__((aligned(128))), so[32] __attribute__((aligned(128)));
            memcpy(se, &fe, 128);
            memcpy(so, &fo, 128);
            float *y = a->y + d->y_at + (uint32_t)b * 32u;
            for (int i = 0; i < 16; ++i) {
                y[2 * i] = se[i] + se[16 + i];
                y[2 * i + 1] = so[i] + so[16 + i];
            }
        }
    }
    free(wlo);
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int geode_npu_gemv(remote_handle64 h, const unsigned char *descs, int descs_len, const unsigned char *x, int x_len,
                   unsigned char *y, int y_len, int w_fd, int threads, int check, int64_t *usec, int64_t *err_ppm) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    *usec = 0;
    *err_ppm = -1;
    const int variant = threads >> 8;   // 0: the qf16 sums; 1: vrmpy (measuring); 2: lookup tables (measuring)
    threads &= 0xFF;
    const int count = descs_len / (int)sizeof(gemv_desc_t);
    if (threads < 1 || threads > GEMV_THREADS || count < 1 || descs_len % (int)sizeof(gemv_desc_t)) return AEE_EBADPARM;
    const gemv_desc_t *d = (const gemv_desc_t *)descs;
    for (int i = 0; i < count; ++i)
        if (d[i].k % 32 || d[i].n % 32 || d[i].k > GEMV_KMAX || (int)(d[i].x_at + d[i].k) * 2 > x_len ||
            (int)(d[i].y_at + d[i].n) * 4 > y_len || (d[i].w_off & 127))
            return AEE_EBADPARM;
    int e = acquire(s);
    if (e) return e;
    const uint32_t xv_bytes = (GEMV_KMAX / 8 * 4 + 2 * (GEMV_KMAX / 16)) * 128 + HMX4_BYTES(GEMV_KMAX);   // + a block
    if ((uint32_t)threads * xv_bytes > s->vtcm_size) return AEE_ENOMEMORY;
    void *wv = NULL;
    uint64 pa = 0;
    if ((e = HAP_mmap_get(w_fd, &wv, &pa))) return e;
    gemv_arg_t args[GEMV_THREADS];
    qurt_thread_t tid[GEMV_THREADS];
    void *stacks[GEMV_THREADS] = {0};
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio <= 0 || prio >= 255) prio = 127;
    int started = 0;
    const uint64_t t0 = HAP_perf_get_time_us();
    for (int w = 0; w < threads; ++w) {
        args[w] = (gemv_arg_t){(const uint8_t *)wv, d, count, w, threads, (const __fp16 *)x, (float *)y,
                               s->vtcm + (size_t)w * xv_bytes, 0};
        qurt_thread_attr_t at;
        qurt_thread_attr_init(&at);
        if (!(stacks[w] = memalign(128, PIPE_STACK))) break;
        qurt_thread_attr_set_name(&at, (char *)"geode_gemv");
        qurt_thread_attr_set_stack_addr(&at, stacks[w]);
        qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
        qurt_thread_attr_set_priority(&at, (unsigned short)prio);
        void (*fn)(void *) = variant == 1 ? gemv_worker_vrmpy : gemv_worker;
        if (qurt_thread_create(&tid[w], &at, fn, &args[w]) != QURT_EOK) break;
        ++started;
    }
    int failed = started < threads;
    for (int w = 0; w < started; ++w) {
        int st;
        qurt_thread_join(tid[w], &st);
        failed |= args[w].failed;
    }
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    for (int w = 0; w < GEMV_THREADS; ++w) free(stacks[w]);
    if (!failed && check) {   // the first matrix against a float sum of the same fp16 values: every 7th column
        const gemv_desc_t *c = &d[0];
        const __fp16 *xx = (const __fp16 *)x + c->x_at;
        const float *yy = (const float *)y + c->y_at;
        double worst = 0;
        for (uint32_t o = 0; o < c->n; o += 7) {
            const uint8_t *blk = (const uint8_t *)wv + c->w_off + (uint64_t)(o / 32) * HMX4_BYTES(c->k);
            double ref = 0, mag = 0;
            for (uint32_t i = 0; i < c->k; ++i) {
                const double p = (double)hmx4_value(blk, (int)c->k, (int)(o % 32), (int)i) * (double)(float)xx[i];
                ref += p;
                mag += p < 0 ? -p : p;
            }
            const double err = ((double)yy[o] - ref) / (mag > 1e-30 ? mag : 1e-30);
            if ((err < 0 ? -err : err) > worst) worst = err < 0 ? -err : err;
        }
        *err_ppm = (int64_t)(worst * 1e6);
    }
    HAP_mmap_put(w_fd);
    return failed ? AEE_ENOTALLOWED : 0;
}

#if __HVX_ARCH__ >= 79   // IEEE single-precision HVX (vmpy sf), the 2D user DMA: v79 and newer
// ---- decode products straight from tile8 (geode_npu_gemv8): the GPU's layout, one copy for both ---------------------
// tile8 (engine kernels.hpp, the q4 layout the Adreno kernels read): a matrix in tiles of 32 rows, 18 k bytes each.
// First the codes: per 64-input chunk c and word j (inputs 64c + 8j .. + 7; byte b = input 2b in the low nibble, 2b + 1
// in the high; affine, level = 2 code - 15) a 128-byte vector whose 32-bit lane L is row L's word. Then the scales: per
// chunk a 128-byte vector of E4M3 bytes [quad i][block b][row r] (row 4i + r, a block = 16 inputs). vrmpy multiplies a
// lane's 4 bytes by 4 int8 inputs, so the low nibbles of word j meet the even inputs and the high ones the odd: no
// reordering of the weights at all. A block's int32 sum, times the input's and the weight's scales, goes to floats;
// the four blocks' vectors are interleaved into the scales' order once per chunk.
typedef struct {
    uint64_t w_off;   // the matrix's first row tile in the mapped buffer
    uint32_t k, n;    // inputs (a multiple of 64), outputs (of 32)
    uint32_t a_at;    // its activations, bytes into act: lo words [k/8], hi words [k/8], 15 x block sums int32 [k/16],
                      // block scales float [k/16]
    uint32_t y_at;    // its outputs, floats into y
    float global;     // the matrix's global scale
    uint32_t pad;
} gemv8_desc_t;

typedef struct {
    const uint8_t *w;
    const gemv8_desc_t *d;
    int count, w_id, threads;
    const uint8_t *act;
    float *y;
    uint8_t *buf;   // this thread's VTCM: one row tile
    volatile int failed;
    int debug;      // the first chunk's vectors written after the first matrix's outputs (a checking aid)
    int dma;        // the tiles fetched by the DMA engine, the next one while the current one is multiplied
} gemv8_arg_t;

#define GEMV8_KMAX 8192

// 32 lanes holding one E4M3 byte each (the scales: never negative) -> floats; a subnormal is its mantissa x 2^-9
static inline HVX_Vector e4m3_sf(HVX_Vector u) {
    const HVX_Vector normal =
        Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_V_vand_VV(u, Q6_V_vsplat_R(0x7F)), 20), Q6_V_vsplat_R(0x3C000000));
    const HVX_VectorPred sub = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(u, Q6_V_vsplat_R(0x78)), Q6_V_vzero());
    const HVX_Vector m = Q6_Vsf_equals_Vw(Q6_V_vand_VV(u, Q6_V_vsplat_R(7)));
    const HVX_Vector subv = Q6_Vsf_vmpy_VsfVsf(m, Q6_V_vsplat_R(0x3B000000));   // x 2^-9
    return Q6_V_vmux_QVV(sub, subv, normal);
}

// The user DMA's descriptor as v75+ takes it (the SDK's udma.h is older; llama.cpp's ggml-hexagon dma-queue.h, MIT,
// has the current layout): 2D with 24-bit fields (desc_type 9); a contiguous copy is one row. v75+ rejects the old
// 1D descriptor (the DSP process went down on it).
typedef struct {
    void *next;
    uint32_t dst_stride : 24;
    uint32_t desc_size : 2;   // 1: 2D
    uint32_t dst_comp : 1;
    uint32_t src_comp : 1;
    uint32_t dst_bypass : 1;
    uint32_t src_bypass : 1;
    uint32_t order : 1;
    uint32_t done : 1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type : 8;
    uint32_t reserved0 : 24;
    uint32_t row_size : 24;
    uint32_t nrows_lo : 8;
    uint32_t nrows_hi : 8;
    uint32_t src_stride : 24;
    uint32_t offset : 24;
    uint32_t reserved1 : 8;
} udma0_t;

// this thread's next row tile after (mi, t): its share of each matrix in turn; 0: none left
static int gemv8_next(const gemv8_arg_t *a, int *mi, int *t) {
    int tt = *t + 1;
    for (int m = *mi; m < a->count; ++m, tt = -1) {
        const int n_t = (int)a->d[m].n / 32, t0 = n_t * a->w_id / a->threads, t1 = n_t * (a->w_id + 1) / a->threads;
        if (tt < t0) tt = t0;
        if (tt < t1) {
            *mi = m;
            *t = tt;
            return 1;
        }
    }
    return 0;
}

// the DMA engine copies src into dst (VTCM) while the thread goes on; the previous transfer is done when this is called
static void udma_fetch(volatile udma0_t *x, void *dst, const void *src, uint32_t len) {
    x->next = NULL;
    x->dst_stride = len;
    x->desc_size = 1;
    x->dst_comp = x->src_comp = 0;
    x->dst_bypass = 1;
    x->src_bypass = 1;   // straight from DDR: the weights are read once a token, caching them only evicts others
    x->order = 0;
    x->done = 0;
    x->src = (uint32_t)(uintptr_t)src;
    x->dst = (uint32_t)(uintptr_t)dst;
    x->desc_type = 9;
    x->reserved0 = 0;
    x->row_size = len;
    x->nrows_lo = 1;
    x->nrows_hi = 0;
    x->src_stride = len;
    x->offset = 0;
    x->reserved1 = 0;
    asm volatile(" release(%0):at" : : "r"(x));   // the descriptor's stores out before the engine reads it
    asm volatile(" dmstart(%0)" : : "r"(x));
}

static void udma_wait(volatile udma0_t *x) {
    while (!x->done) {
        unsigned int st;
        asm volatile(" %0 = dmpoll" : "=r"(st) : : "memory");
    }
}

static void gemv8_worker(void *arg) {
    gemv8_arg_t *a = (gemv8_arg_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        a->failed = 1;
        qurt_thread_exit(0);
    }
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    // two tile buffers (one filled by the DMA engine while the other is multiplied); the descriptors in cached memory
    uint8_t *bufs[2] = {a->buf, a->buf + GEMV8_KMAX * 18u};
    udma0_t *desc_mem = (udma0_t *)memalign(64, 2 * sizeof(udma0_t));
    if (!desc_mem) {
        a->failed = 1;
        qurt_hvx_unlock();
        qurt_thread_exit(0);
    }
    volatile udma0_t *desc = desc_mem;
    int mi = 0, t = -1, cur = 0;
    int more = gemv8_next(a, &mi, &t);
    if (more && a->dma)
        udma_fetch(&desc[0], bufs[0], a->w + a->d[mi].w_off + (uint64_t)t * a->d[mi].k * 18u, a->d[mi].k * 18u);
    while (more) {
        const gemv8_desc_t *d = &a->d[mi];
        const int k = (int)d->k, nch = k / 64;
        const uint32_t *lo = (const uint32_t *)(a->act + d->a_at), *hi = lo + k / 8;
        const int32_t *ks = (const int32_t *)(hi + k / 8);
        const float *xs = (const float *)(ks + k / 16);
        const uint32_t tb = (uint32_t)k * 18u;
        if (a->dma) {
            udma_wait(&desc[cur]);
        } else {
            hvx_copy(bufs[cur], a->w + d->w_off + (uint64_t)t * tb, tb);
        }
        int nmi = mi, nt = t;
        more = gemv8_next(a, &nmi, &nt);
        if (more && a->dma)   // the next tile on its way while this one is multiplied
            udma_fetch(&desc[cur ^ 1], bufs[cur ^ 1], a->w + a->d[nmi].w_off + (uint64_t)nt * a->d[nmi].k * 18u,
                       a->d[nmi].k * 18u);
        {
            const HVX_Vector *cv = (const HVX_Vector *)bufs[cur];
            const HVX_Vector *sv = cv + 8 * nch;
            // IEEE single precision throughout: with qf32 accumulators one of the four came out as garbage (rows 0-7,
            // then 8-15 once the code around it moved) - a v79 codegen or hazard problem not worth chasing
            HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero(), acc2 = Q6_V_vzero(), acc3 = Q6_V_vzero();
            for (int c = 0; c < nch; ++c) {
                HVX_Vector dq[4];   // per block: the 32 rows' sums times the input's scale (sf), lane = row
                for (int b = 0; b < 4; ++b) {
                    const int j = c * 8 + 2 * b;
                    const HVX_Vector v0 = cv[j], v1 = cv[j + 1];
                    HVX_Vector s = Q6_Vw_vrmpy_VubRb(Q6_V_vand_VV(v0, m0f), (int)lo[j]);
                    s = Q6_Vw_vrmpyacc_VwVubRb(s, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v0, 4), m0f), (int)hi[j]);
                    s = Q6_Vw_vrmpyacc_VwVubRb(s, Q6_V_vand_VV(v1, m0f), (int)lo[j + 1]);
                    s = Q6_Vw_vrmpyacc_VwVubRb(s, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v1, 4), m0f), (int)hi[j + 1]);
                    // sum (2 code - 15) x = 2 s - 15 sum x (exact in int32)
                    const HVX_Vector dot = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(s, s), Q6_V_vsplat_R(ks[c * 4 + b]));
                    union { float f; int32_t i; } xsc = {xs[c * 4 + b]};
                    dq[b] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(dot), Q6_V_vsplat_R(xsc.i));
                }
                // [block][quad][row] -> [quad][block][row], the scales' order: 16-byte units interleaved, then 32-byte
                const HVX_VectorPair p = Q6_W_vshuff_VVR(dq[1], dq[0], -16), q = Q6_W_vshuff_VVR(dq[3], dq[2], -16);
                const HVX_VectorPair r0 = Q6_W_vshuff_VVR(Q6_V_lo_W(q), Q6_V_lo_W(p), -32);
                const HVX_VectorPair r1 = Q6_W_vshuff_VVR(Q6_V_hi_W(q), Q6_V_hi_W(p), -32);
                const HVX_VectorPair sh = Q6_Wuh_vunpack_Vub(sv[c]);
                const HVX_VectorPair w0 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(sh)), w1 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(sh));
                if (a->debug && mi == 0 && t == 0 && c == 0) {   // the first chunk's vectors after y (checking)
                    HVX_Vector dv[8] __attribute__((aligned(128)));
                    dv[0] = Q6_V_lo_W(r0), dv[1] = Q6_V_hi_W(r0), dv[2] = Q6_V_lo_W(r1), dv[3] = Q6_V_hi_W(r1);
                    dv[4] = e4m3_sf(Q6_V_lo_W(w0)), dv[5] = e4m3_sf(Q6_V_hi_W(w0));
                    dv[6] = e4m3_sf(Q6_V_lo_W(w1)), dv[7] = e4m3_sf(Q6_V_hi_W(w1));
                    memcpy(a->y + d->y_at + d->n, dv, sizeof dv);
                }
                acc0 = Q6_Vsf_vadd_VsfVsf(acc0, Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(r0), e4m3_sf(Q6_V_lo_W(w0))));
                acc1 = Q6_Vsf_vadd_VsfVsf(acc1, Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(r0), e4m3_sf(Q6_V_hi_W(w0))));
                acc2 = Q6_Vsf_vadd_VsfVsf(acc2, Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(r1), e4m3_sf(Q6_V_lo_W(w1))));
                acc3 = Q6_Vsf_vadd_VsfVsf(acc3, Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(r1), e4m3_sf(Q6_V_hi_W(w1))));
            }
            // vector v holds quads 2v, 2v + 1: lane (i - 2v) * 16 + b * 4 + r
            HVX_Vector fv[4] __attribute__((aligned(128)));
            fv[0] = acc0, fv[1] = acc1, fv[2] = acc2, fv[3] = acc3;
            float f[128];
            memcpy(f, fv, sizeof f);
            float *y = a->y + d->y_at + (uint32_t)t * 32u;
            for (int i = 0; i < 8; ++i)
                for (int r = 0; r < 4; ++r) {
                    const float *u = f + (i >> 1) * 32 + (i & 1) * 16 + r;
                    y[4 * i + r] = d->global * (u[0] + u[4] + u[8] + u[12]);
                }
        }
        mi = nmi;
        t = nt;
        cur ^= 1;
    }
    free(desc_mem);   // (every transfer was waited for: the last one before its tile was multiplied)
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

int geode_npu_gemv8(remote_handle64 h, const unsigned char *descs, int descs_len, const unsigned char *act, int act_len,
                    unsigned char *y, int y_len, int w_fd, int threads, int64_t *usec) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    *usec = 0;
    const int debug = (threads >> 15) & 1;   // flag 0x80: the first chunk's vectors after y (checking)
    const int dma = (threads >> 8) & 1;      // flag 1: the tiles fetched by the DMA engine
    threads &= 0xFF;
    const int count = descs_len / (int)sizeof(gemv8_desc_t);
    if (threads < 1 || threads > GEMV_THREADS || count < 1 || descs_len % (int)sizeof(gemv8_desc_t)) return AEE_EBADPARM;
    const gemv8_desc_t *d = (const gemv8_desc_t *)descs;
    for (int i = 0; i < count; ++i)
        if (d[i].k % 64 || d[i].n % 32 || d[i].k > GEMV8_KMAX || (d[i].w_off & 127) || (d[i].a_at & 3) ||
            (int64_t)d[i].a_at + (int64_t)d[i].k * 3 / 2 > act_len || (int64_t)(d[i].y_at + d[i].n) * 4 > y_len)
            return AEE_EBADPARM;
    int e = acquire(s);
    if (e) return e;
    const uint32_t buf_bytes = 2 * GEMV8_KMAX * 18u + 128u;   // two tiles + the DMA descriptors
    if ((uint32_t)threads * buf_bytes > s->vtcm_size) return AEE_ENOMEMORY;
    void *wv = NULL;
    uint64 pa = 0;
    if ((e = HAP_mmap_get(w_fd, &wv, &pa))) return e;
    gemv8_arg_t args[GEMV_THREADS];
    qurt_thread_t tid[GEMV_THREADS];
    void *stacks[GEMV_THREADS] = {0};
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio <= 0 || prio >= 255) prio = 127;
    int started = 0;
    const uint64_t t0 = HAP_perf_get_time_us();
    for (int w = 0; w < threads; ++w) {
        args[w] = (gemv8_arg_t){(const uint8_t *)wv, d, count, w, threads, act, (float *)y,
                                s->vtcm + (size_t)w * buf_bytes, 0, debug, dma};
        qurt_thread_attr_t at;
        qurt_thread_attr_init(&at);
        if (!(stacks[w] = memalign(128, PIPE_STACK))) break;
        qurt_thread_attr_set_name(&at, (char *)"geode_gemv8");
        qurt_thread_attr_set_stack_addr(&at, stacks[w]);
        qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
        qurt_thread_attr_set_priority(&at, (unsigned short)prio);
        if (qurt_thread_create(&tid[w], &at, gemv8_worker, &args[w]) != QURT_EOK) break;
        ++started;
    }
    int failed = started < threads;
    for (int w = 0; w < started; ++w) {
        int st;
        qurt_thread_join(tid[w], &st);
        failed |= args[w].failed;
    }
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    for (int w = 0; w < GEMV_THREADS; ++w) free(stacks[w]);
    HAP_mmap_put(w_fd);
    return failed ? AEE_ENOTALLOWED : 0;
}
// ---- a prompt's routed experts on HMX (geode_npu_moe, moe_cfg.h): straight from their tile8 matrices ---------------
// Each expert in turn on this (the FastRPC) thread, which holds HMX: its picks' rows of x gathered into activation
// tiles, gate|up, SwiGLU into the next tiles, down, every pick's row times its weight added into its token's y. Worker
// threads expand the weights a 32-row block at a time (a tile8 tile -> HMX's fp16 block) into a ring, one expert's
// blocks after another's, so the expansion goes on while this thread gathers and scatters.
#include "moe_cfg.h"

#define MOE_RING 12
#define MOE_ROWS 192          // an expert's picks a pass (more: several passes, its weights expanded again)
#define MOE_WORKERS_MAX 4     // v79 has 6 HVX contexts: the workers, the helper and this thread
#define MOE_WORKERS 4

static inline HVX_Vector m_sf(float f) {
    union { float f; int32_t i; } u = {f};
    return Q6_V_vsplat_R(u.i);
}
static inline HVX_Vector m_mul(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vmpy_VsfVsf(a, b); }
static inline HVX_Vector m_add(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vadd_VsfVsf(a, b); }
static inline HVX_Vector m_sub(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vsub_VsfVsf(a, b); }

// exp (2^n times a polynomial), 1/x (Newton), silu, as dec.c's but only as exact as the halves they end in: e^t to
// t^5 (|t| < ln 2: 1.5e-4), two Newton steps (2e-4)
static inline HVX_Vector m_exp(HVX_Vector x) {
    HVX_Vector y = m_mul(x, m_sf(1.44269504f));
    y = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(y, m_sf(-125.0f)), m_sf(125.0f));
    const HVX_Vector n = Q6_Vw_equals_Vsf(y);
    const HVX_Vector t = m_mul(m_sub(y, Q6_Vsf_equals_Vw(n)), m_sf(0.69314718f));
    HVX_Vector p = m_sf(1.0f / 120.0f);
    p = m_add(m_mul(p, t), m_sf(1.0f / 24.0f));
    p = m_add(m_mul(p, t), m_sf(1.0f / 6.0f));
    p = m_add(m_mul(p, t), m_sf(0.5f));
    p = m_add(m_mul(p, t), m_sf(1.0f));
    p = m_add(m_mul(p, t), m_sf(1.0f));
    return Q6_Vw_vadd_VwVw(p, Q6_Vw_vasl_VwR(n, 23));
}
static inline HVX_Vector m_recip(HVX_Vector x) {
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), x);
    for (int i = 0; i < 2; ++i) y = m_mul(y, m_sub(m_sf(2.0f), m_mul(x, y)));
    return y;
}
static inline HVX_Vector m_silu(HVX_Vector x) {
    return m_mul(x, m_recip(m_add(m_sf(1.0f), m_exp(m_sub(Q6_V_vzero(), x)))));
}

// vcvt's lane orders between sf and hf, worked out once (as dec.c's cvt_check): g_cvt_to 1 when hf = vcvt(a, b) puts
// a's values at the even lanes, 2 at the odd; g_cvt_from 1 when vcvt(hf)'s lo holds the even lanes, 2 the odd
static int g_cvt_to, g_cvt_from;
static void moe_cvt_check(void) {
    float a[32] __attribute__((aligned(128))), b[32] __attribute__((aligned(128)));
    for (int i = 0; i < 32; ++i) a[i] = (float)(i + 1), b[i] = -(float)(i + 1);
    __fp16 h[64] __attribute__((aligned(128)));
    *(HVX_Vector *)h = Q6_Vhf_vcvt_VsfVsf(*(const HVX_Vector *)a, *(const HVX_Vector *)b);
    int ea = 1, eb = 1;
    for (int i = 0; i < 32; ++i) {
        ea &= (float)h[2 * i] == a[i] && (float)h[2 * i + 1] == b[i];
        eb &= (float)h[2 * i] == b[i] && (float)h[2 * i + 1] == a[i];
    }
    g_cvt_to = ea ? 1 : eb ? 2 : -1;
    for (int i = 0; i < 32; ++i) h[2 * i] = (__fp16)a[i], h[2 * i + 1] = (__fp16)b[i];
    const HVX_VectorPair w = Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)h);
    float lo[32] __attribute__((aligned(128))), hi[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lo = Q6_V_lo_W(w);
    *(HVX_Vector *)hi = Q6_V_hi_W(w);
    int le = 1, he = 1;
    for (int i = 0; i < 32; ++i) {
        le &= lo[i] == a[i] && hi[i] == b[i];
        he &= hi[i] == a[i] && lo[i] == b[i];
    }
    g_cvt_from = le ? 1 : he ? 2 : -1;
}
// 64 halves from the even lanes' and the odd lanes' floats, and back
static inline HVX_Vector hf_of(HVX_Vector even, HVX_Vector odd) {
    return g_cvt_to == 1 ? Q6_Vhf_vcvt_VsfVsf(even, odd) : Q6_Vhf_vcvt_VsfVsf(odd, even);
}
static inline void sf_of(HVX_Vector h, HVX_Vector *even, HVX_Vector *odd) {
    const HVX_VectorPair w = Q6_Wsf_vcvt_Vhf(h);
    *even = g_cvt_from == 1 ? Q6_V_lo_W(w) : Q6_V_hi_W(w);
    *odd = g_cvt_from == 1 ? Q6_V_hi_W(w) : Q6_V_lo_W(w);
}

// One 32-row tile of a tile8 matrix (k inputs) into HMX's fp16 weight block (wgt_index): per 64-input chunk the
// scales first - the E4M3 bytes as floats, dealt from gemv8's [quad][block][row] back into a vector per block (lane =
// row; the inverse of the shuffles gemv8_worker makes), times the global scale, each twice as halves ([row][input % 2])
// - then each code word's vector: two byte deals take [row][input pair] to [input pair][row], and each nibble becomes
// level 2 code - 15 times its row's scale, as dequant_block does for hmx4. The output vectors follow one another
// (input pair after input pair: 32 c + 4 j + pair)
static void dequant_tile8(uint8_t *dst, const uint8_t *src, int k, float global) {
    const int nch = k / 64;
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F), k15 = Q6_Vh_vsplat_R(15), gv = m_sf(global);
    const HVX_Vector *cv = (const HVX_Vector *)src, *sv = cv + 8 * nch;
    HVX_Vector *out = (HVX_Vector *)dst;
    for (int c = 0; c < nch; ++c) {
        const HVX_VectorPair sh = Q6_Wuh_vunpack_Vub(sv[c]);
        const HVX_VectorPair w0 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(sh)), w1 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(sh));
        const HVX_Vector f0 = e4m3_sf(Q6_V_lo_W(w0)), f1 = e4m3_sf(Q6_V_hi_W(w0));
        const HVX_Vector f2 = e4m3_sf(Q6_V_lo_W(w1)), f3 = e4m3_sf(Q6_V_hi_W(w1));
        const HVX_VectorPair a = Q6_W_vdeal_VVR(f1, f0, -32), b = Q6_W_vdeal_VVR(f3, f2, -32);
        const HVX_VectorPair p01 = Q6_W_vdeal_VVR(Q6_V_lo_W(b), Q6_V_lo_W(a), -16);
        const HVX_VectorPair p23 = Q6_W_vdeal_VVR(Q6_V_hi_W(b), Q6_V_hi_W(a), -16);
        const HVX_Vector blk[4] = {Q6_V_lo_W(p01), Q6_V_hi_W(p01), Q6_V_lo_W(p23), Q6_V_hi_W(p23)};
        HVX_Vector sc[4];
        for (int q = 0; q < 4; ++q) {
            const HVX_Vector x = m_mul(blk[q], gv);
            sc[q] = Q6_Vhf_vcvt_VsfVsf(x, x);   // each row's scale twice, whichever lane order vcvt has
        }
        for (int j = 0; j < 8; ++j) {
            const HVX_Vector v = Q6_Vb_vdeal_Vb(Q6_Vb_vdeal_Vb(cv[c * 8 + j]));   // 4 input pairs x 32 rows
            const HVX_Vector lo = Q6_V_vand_VV(v, m0f);
            const HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), m0f);
            const HVX_VectorPair cb = Q6_W_vshuff_VVR(hi, lo, -1);   // [pair][row][input % 2]
            for (int half = 0; half < 2; ++half) {
                const HVX_VectorPair hw = Q6_Wuh_vunpack_Vub(half ? Q6_V_hi_W(cb) : Q6_V_lo_W(cb));
                for (int x = 0; x < 2; ++x) {
                    const HVX_Vector cd = x ? Q6_V_hi_W(hw) : Q6_V_lo_W(hw);
                    const HVX_Vector lv = Q6_Vh_vsub_VhVh(Q6_Vh_vadd_VhVh(cd, cd), k15);
                    *out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(lv), sc[j / 2]));
                }
            }
        }
    }
}

typedef struct {
    const moe_exp_t *e;
    uint32_t first, count;   // this pass's picks
} moe_job_t;

// a token's row (bytes) into L2 ahead of its loads
static inline void moe_l2(const void *row, uint32_t bytes) {
    const uint64_t pfc = (uint64_t)bytes << 32 | (uint64_t)bytes << 16 | 1u;
    asm volatile(" l2fetch(%0,%1) " : : "r"(row), "r"(pfc));
}

typedef struct {
    const uint8_t *const *bases;   // each buffer's address, by fds' index
    const moe_job_t *jobs;
    int n_jobs, H, ff, nb_gu, nb_dn, per_job, total, nw;
    uint32_t flags;
    uint8_t *slots[MOE_RING], *stage[MOE_WORKERS_MAX];
    uint8_t *A[2], *O2[2];   // a job's activation tiles and its down's results, two jobs' worth (j % 2)
    const float *x;
    float *y;
    const moe_pick_t *picks;
    volatile int ready[MOE_RING];
    volatile int consumed, failed;
    // the helper's progress and the multiplier's: jobs gathered, through gate|up, through down, scattered
    volatile int gathered, gu_done, dn_done, scattered;
} moe_pipe_t;

typedef struct {
    moe_pipe_t *p;
    int w;
} moe_arg_t;

static void *g_moe_stacks[MOE_WORKERS_MAX + 1];

static void moe_worker(void *arg) {
    moe_arg_t *a = (moe_arg_t *)arg;
    moe_pipe_t *p = a->p;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        __atomic_store_n(&p->failed, 1, __ATOMIC_RELEASE);
        qurt_thread_exit(0);
    }
    for (int g = a->w; g < p->total; g += p->nw) {
        // slot g % RING last held block g - RING: free once the multiplier is past it
        while (__atomic_load_n(&p->consumed, __ATOMIC_ACQUIRE) < g - MOE_RING) {
        }
        const moe_job_t *j = &p->jobs[g / p->per_job];
        const int b = g % p->per_job, gu = b < p->nb_gu, k = gu ? p->H : p->ff;
        const uint8_t *src = p->bases[j->e->fd] + (gu ? j->e->gu_off + (uint64_t)b * 18u * (uint64_t)k
                                                       : j->e->dn_off + (uint64_t)(b - p->nb_gu) * 18u * (uint64_t)k);
        hvx_copy(p->stage[a->w], src, 18u * (uint32_t)k);
        dequant_tile8(p->slots[g % MOE_RING], p->stage[a->w], k, gu ? j->e->gu_global : j->e->dn_global);
        __atomic_store_n(&p->ready[g % MOE_RING], g, __ATOMIC_RELEASE);
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

// spin until *v reaches at least `at` (0: a thread failed)
static int moe_until(moe_pipe_t *p, volatile int *v, int at) {
    while (__atomic_load_n(v, __ATOMIC_ACQUIRE) < at)
        if (__atomic_load_n(&p->failed, __ATOMIC_ACQUIRE)) return 0;
    return 1;
}

// job jb's picks' rows of x into its activation tiles, two rows a vector (rows past its count zero; the next rows
// ahead in L2)
static void moe_gather(moe_pipe_t *p, int jb) {
    const moe_job_t *j = &p->jobs[jb];
    const int m = (int)j->count, tiles = (m + 31) / 32, H = p->H;
    const moe_pick_t *pk = p->picks + j->first;
    uint8_t *A = p->A[jb % 2];
    const HVX_Vector zero = Q6_V_vzero();
    for (int r = 0; r < 4 && r < m; ++r) moe_l2(p->x + (size_t)pk[r].tok * H, (uint32_t)H * 4u);
    for (int r = 0; r < tiles * 32; r += 2) {
        for (int q = r + 4; q < r + 6 && q < m; ++q) moe_l2(p->x + (size_t)pk[q].tok * H, (uint32_t)H * 4u);
        const HVX_Vector *x0 = r < m ? (const HVX_Vector *)(p->x + (size_t)pk[r].tok * H) : NULL;
        const HVX_Vector *x1 = r + 1 < m ? (const HVX_Vector *)(p->x + (size_t)pk[r + 1].tok * H) : NULL;
        HVX_Vector *dst = (HVX_Vector *)(A + (size_t)(r / 32) * H * 64 + (size_t)(r % 32 / 2) * 128);
        for (int gi = 0; gi < H / 32; ++gi) dst[gi * 16] = hf_of(x0 ? x0[gi] : zero, x1 ? x1[gi] : zero);
    }
}

// job jb's down results, each pick's row times its weight, into its token's y: a pair of rows at a time, each row of
// y in order (the next pair's ahead in L2)
static void moe_scatter(moe_pipe_t *p, int jb) {
    const moe_job_t *j = &p->jobs[jb];
    const int m = (int)j->count, H = p->H;
    const moe_pick_t *pk = p->picks + j->first;
    const uint8_t *O2 = p->O2[jb % 2];
    for (int r = 0; r < 4 && r < m; ++r) moe_l2(p->y + (size_t)pk[r].tok * H, (uint32_t)H * 4u);
    for (int r = 0; r < m; r += 2) {
        for (int q = r + 4; q < r + 6 && q < m; ++q) moe_l2(p->y + (size_t)pk[q].tok * H, (uint32_t)H * 4u);
        const int t = r / 32, i = r % 32 / 2;
        const HVX_Vector w0 = m_sf(pk[r].weight), w1 = m_sf(r + 1 < m ? pk[r + 1].weight : 0.0f);
        HVX_Vector *y0 = (HVX_Vector *)(p->y + (size_t)pk[r].tok * H);
        HVX_Vector *y1 = r + 1 < m ? (HVX_Vector *)(p->y + (size_t)pk[r + 1].tok * H) : NULL;
        const HVX_Vector *ov = (const HVX_Vector *)(O2 + (size_t)t * p->nb_dn * 2048u) + i;
        for (int b = 0; b < p->nb_dn; ++b) {
            HVX_Vector v0, v1;
            sf_of(ov[b * 16], &v0, &v1);
            y0[b] = m_add(y0[b], m_mul(v0, w0));
            if (y1) y1[b] = m_add(y1[b], m_mul(v1, w1));
        }
    }
}

// The helper: the first two jobs' rows gathered, then for each job, once its down is through, its rows scattered and
// the job two on gathered (into the tiles that job's gate|up has finished with)
static void moe_helper(void *arg) {
    moe_pipe_t *p = (moe_pipe_t *)arg;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        __atomic_store_n(&p->failed, 1, __ATOMIC_RELEASE);
        qurt_thread_exit(0);
    }
    for (int jb = 0; jb < 2 && jb < p->n_jobs; ++jb) {
        if (!(p->flags & MOE_DEBUG_NO_GATHER)) moe_gather(p, jb);
        __atomic_store_n(&p->gathered, jb, __ATOMIC_RELEASE);
    }
    for (int jb = 0; jb < p->n_jobs; ++jb) {
        if (!moe_until(p, &p->dn_done, jb)) break;
        if (!(p->flags & MOE_DEBUG_NO_SCATTER)) moe_scatter(p, jb);
        __atomic_store_n(&p->scattered, jb, __ATOMIC_RELEASE);
        if (jb + 2 < p->n_jobs) {   // (its gate|up is through: its down is)
            if (!(p->flags & MOE_DEBUG_NO_GATHER)) moe_gather(p, jb + 2);
            __atomic_store_n(&p->gathered, jb + 2, __ATOMIC_RELEASE);
        }
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

// block g's slot once a worker has expanded it (NULL: a thread failed)
static const uint8_t *moe_wait(moe_pipe_t *p, int g) {
    while (__atomic_load_n(&p->ready[g % MOE_RING], __ATOMIC_ACQUIRE) != g)
        if (__atomic_load_n(&p->failed, __ATOMIC_ACQUIRE)) return NULL;
    return p->slots[g % MOE_RING];
}

static int moe_thread(qurt_thread_t *tid, int slot, void (*fn)(void *), void *arg, int prio) {
    qurt_thread_attr_t at;
    qurt_thread_attr_init(&at);
    if (!g_moe_stacks[slot] && !(g_moe_stacks[slot] = memalign(128, PIPE_STACK))) return 0;
    qurt_thread_attr_set_name(&at, (char *)"geode_moe");
    qurt_thread_attr_set_stack_addr(&at, g_moe_stacks[slot]);
    qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
    qurt_thread_attr_set_priority(&at, (unsigned short)prio);
    return qurt_thread_create(tid, &at, fn, arg) == QURT_EOK;
}

int geode_npu_moe(remote_handle64 h, const unsigned char *desc, int desc_len, int io_fd, int64_t *usec) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    *usec = 0;
    if (desc_len < (int)sizeof(moe_hdr_t)) return AEE_EBADPARM;
    const moe_hdr_t *hd = (const moe_hdr_t *)desc;
    const int H = (int)hd->hidden, ff = (int)hd->ff, n_fds = (int)hd->n_fds;
    const uint64_t fds_bytes = (uint64_t)((n_fds + 1) / 2 * 2) * 4u;
    if (hd->version != MOE_VERSION || H % 64 || H < 64 || H > 4096 || ff % 64 || ff < 64 || ff > 2048 ||
        n_fds < 1 || n_fds > MOE_MAX_FDS || (hd->x_at & 127) || (hd->y_at & 127) ||
        (uint64_t)desc_len != sizeof(moe_hdr_t) + fds_bytes + (uint64_t)hd->n_experts * sizeof(moe_exp_t) +
                                  (uint64_t)hd->n_picks * sizeof(moe_pick_t))
        return AEE_EBADPARM;
    const int32_t *fds = (const int32_t *)(hd + 1);
    const moe_exp_t *ex = (const moe_exp_t *)((const uint8_t *)fds + fds_bytes);
    const moe_pick_t *picks = (const moe_pick_t *)(ex + hd->n_experts);
    int n_jobs = 0;
    for (uint32_t i = 0; i < hd->n_experts; ++i) {
        if ((ex[i].gu_off & 127) || (ex[i].dn_off & 127) || ex[i].fd >= (uint32_t)n_fds ||
            (uint64_t)ex[i].first + ex[i].count > hd->n_picks)
            return AEE_EBADPARM;
        n_jobs += (int)((ex[i].count + MOE_ROWS - 1) / MOE_ROWS);
    }
    for (uint32_t i = 0; i < hd->n_picks; ++i)
        if (picks[i].tok >= hd->n_tok) return AEE_EBADPARM;
    const int nw = hd->workers >= 1 && hd->workers <= MOE_WORKERS_MAX ? (int)hd->workers : MOE_WORKERS;
    int e = acquire(s);
    if (e) return e;
    // VTCM: bias | ring | the workers' tile8 copies | two jobs' activation tiles | two jobs' down results | gate|up's
    // results | down's input
    const uint32_t slot = (uint32_t)H * 64u, stage = 18u * (uint32_t)H, a_bytes = (uint32_t)MOE_ROWS * (uint32_t)H * 2u;
    uint8_t *bias = s->vtcm, *ring = bias + 2048, *stages = ring + (size_t)MOE_RING * slot;
    uint8_t *A0 = stages + (size_t)nw * stage, *O20 = A0 + 2 * (size_t)a_bytes, *O1 = O20 + 2 * (size_t)a_bytes;
    uint8_t *A2 = O1 + (size_t)MOE_ROWS * 2 * ff * 2;
    if ((size_t)(A2 + (size_t)MOE_ROWS * ff * 2 - s->vtcm) > s->vtcm_size) return AEE_ENOMEMORY;
    // the buffers, each mapped once here (the CPU mapped them into the session)
    const uint8_t *bases[MOE_MAX_FDS];
    void *iov = NULL;
    uint64 pa = 0;
    int mapped = 0;
    for (; mapped < n_fds; ++mapped) {
        void *v = NULL;
        if ((e = HAP_mmap_get(fds[mapped], &v, &pa))) break;
        bases[mapped] = (const uint8_t *)v;
    }
    if (!e) e = HAP_mmap_get(io_fd, &iov, &pa);
    moe_job_t *jobs = e ? NULL : (moe_job_t *)malloc((size_t)(n_jobs ? n_jobs : 1) * sizeof(moe_job_t));
    if (!jobs) {
        if (iov) HAP_mmap_put(io_fd);
        for (int i = 0; i < mapped; ++i) HAP_mmap_put(fds[i]);
        return e ? e : AEE_ENOMEMORY;
    }
    n_jobs = 0;
    for (uint32_t i = 0; i < hd->n_experts; ++i)
        for (uint32_t o = 0; o < ex[i].count; o += MOE_ROWS)
            jobs[n_jobs++] = (moe_job_t){&ex[i], ex[i].first + o, ex[i].count - o < MOE_ROWS ? ex[i].count - o : MOE_ROWS};
    const uint64_t t0 = HAP_perf_get_time_us();
    if (!g_cvt_to) moe_cvt_check();
    if (g_cvt_to < 0 || g_cvt_from < 0) {   // an unknown lane order: no tiles could be built
        free(jobs);
        HAP_mmap_put(io_fd);
        for (int i = 0; i < n_fds; ++i) HAP_mmap_put(fds[i]);
        return AEE_EUNSUPPORTED;
    }
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);   // x as the CPU wrote it
    moe_pipe_t p;
    memset(&p, 0, sizeof p);
    p.x = (const float *)((uint8_t *)iov + hd->x_at);
    p.y = (float *)((uint8_t *)iov + hd->y_at);
    if (hd->flags & MOE_ZERO_Y) memset(p.y, 0, (size_t)hd->n_tok * H * 4);
    memset(bias, 0, 2048);
    for (int ch = 0; ch < 32; ++ch) ((uint32_t *)bias)[ch] = 15u << 10;   // output bias 0, scale 1
    Q6_bias_mxmem2_A(bias);
    p.bases = bases;
    p.jobs = jobs;
    p.picks = picks;
    p.flags = hd->flags;
    p.n_jobs = n_jobs, p.H = H, p.ff = ff, p.nb_gu = 2 * ff / 32, p.nb_dn = H / 32, p.per_job = p.nb_gu + p.nb_dn;
    p.total = n_jobs * p.per_job, p.nw = nw, p.consumed = -1;
    p.gathered = p.gu_done = p.dn_done = p.scattered = -1;
    p.A[0] = A0, p.A[1] = A0 + a_bytes, p.O2[0] = O20, p.O2[1] = O20 + a_bytes;
    for (int i = 0; i < MOE_RING; ++i) {
        p.slots[i] = ring + (size_t)i * slot;
        p.ready[i] = -1;
    }
    for (int i = 0; i < nw; ++i) p.stage[i] = stages + (size_t)i * stage;
    moe_arg_t args[MOE_WORKERS_MAX];
    qurt_thread_t tid[MOE_WORKERS_MAX + 1];
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio <= 0 || prio >= 255) prio = 127;
    int started = 0;
    if (n_jobs) {
        for (int w = 0; w < nw; ++w) {
            args[w].p = &p;
            args[w].w = w;
            if (!moe_thread(&tid[w], w, moe_worker, &args[w], prio)) break;
            ++started;
        }
        if (started == nw && moe_thread(&tid[nw], MOE_WORKERS_MAX, moe_helper, &p, prio)) ++started;
        if (started < nw + 1) p.failed = 1;
    }
    const int nf = ff / 32;
    int g = 0;
    for (int jb = 0; jb < n_jobs && !p.failed; ++jb) {
        const int m = (int)jobs[jb].count, tiles = (m + 31) / 32;
        uint8_t *A = p.A[jb % 2], *O2 = p.O2[jb % 2];
        // gate|up, once the helper has the rows in; each up block's column of silu(gate) * up right after it (the gate
        // blocks come first) - between the blocks, so the workers' ring keeps draining. The results' croutons are the
        // next product's activation tiles (the same layout)
        if (!moe_until(&p, &p.gathered, jb)) break;
        for (int b = 0; b < p.nb_gu; ++b, ++g) {
            const uint8_t *wb = moe_wait(&p, g);
            if (!wb) break;
            for (int t = 0; t < tiles && !(hd->flags & MOE_DEBUG_NO_HMX); ++t)
                tile(A + (size_t)t * H * 64, wb, O1 + ((size_t)t * p.nb_gu + b) * 2048u, H / 32);
            __atomic_store_n(&p.consumed, g - 1, __ATOMIC_RELEASE);   // one behind: HMX may still read this slot
            if (b < nf || (hd->flags & MOE_DEBUG_NO_SWIGLU)) continue;
            for (int t = 0; t < tiles; ++t) {
                const HVX_Vector *gv = (const HVX_Vector *)(O1 + ((size_t)t * p.nb_gu + b - nf) * 2048u);
                const HVX_Vector *uv = (const HVX_Vector *)(O1 + ((size_t)t * p.nb_gu + b) * 2048u);
                HVX_Vector *av = (HVX_Vector *)(A2 + (size_t)t * ff * 64 + (size_t)(b - nf) * 2048u);
                for (int i = 0; i < 16; ++i) {
                    HVX_Vector g0, g1, u0, u1;
                    sf_of(gv[i], &g0, &g1);
                    sf_of(uv[i], &u0, &u1);
                    av[i] = hf_of(m_mul(m_silu(g0), u0), m_mul(m_silu(g1), u1));
                }
            }
        }
        if (p.failed) break;
        __atomic_store_n(&p.gu_done, jb, __ATOMIC_RELEASE);
        // down, into the results the helper scattered two jobs ago
        if (!moe_until(&p, &p.scattered, jb - 2)) break;
        for (int b = 0; b < p.nb_dn; ++b, ++g) {
            const uint8_t *wb = moe_wait(&p, g);
            if (!wb) break;
            for (int t = 0; t < tiles && !(hd->flags & MOE_DEBUG_NO_HMX); ++t)
                tile(A2 + (size_t)t * ff * 64, wb, O2 + ((size_t)t * p.nb_dn + b) * 2048u, ff / 32);
            __atomic_store_n(&p.consumed, g - 1, __ATOMIC_RELEASE);
        }
        if (p.failed) break;
        __atomic_store_n(&p.dn_done, jb, __ATOMIC_RELEASE);
    }
    // the last job's scatter, then everyone out (a worker or the helper left waiting goes on and finishes)
    if (!p.failed && n_jobs) moe_until(&p, &p.scattered, n_jobs - 1);
    if (p.failed) __atomic_store_n(&p.dn_done, n_jobs, __ATOMIC_RELEASE);
    __atomic_store_n(&p.consumed, p.total, __ATOMIC_RELEASE);
    for (int w = 0; w < started; ++w) {
        int st;
        qurt_thread_join(tid[w], &st);
    }
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);   // y out to where the CPU reads it
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    free(jobs);
    HAP_mmap_put(io_fd);
    for (int i = 0; i < n_fds; ++i) HAP_mmap_put(fds[i]);
    return p.failed ? AEE_ENOTALLOWED : 0;
}


// ---- an MLP's way in (gemm_glu): act(A x G^T) * (A x U^T) ----------------------------------------------------
// The gate's and the up's blocks in turn through the q4 pipeline's ring (block g: the gate's g / 2, or the up's),
// each up block's results joined with its gate block's on HVX here, between the blocks - so the fp16 that comes out is
// the down projection's input as it is (the CPU neither converts the 2 x n results nor applies the activation, and
// converts no input for down). Worker threads and stacks as pipe_run's; the activation on two helper threads (v79's
// six HVX contexts: the three workers, this thread and these), block b on helper b % 2 out of buffer b % GLU_BUFS,
// while this thread goes on with the next blocks' products - on this thread it held HMX up for ~0.6 s of a 2048-token
// prompt
#define GLU_HELPERS 2
#define GLU_BUFS 4   // blocks of gate and up results in flight (a multiple of GLU_HELPERS)
typedef struct {
    uint8_t *slots[PIPE_RING], *stage[PIPE_WORKERS];
    const uint8_t *w[2];   // the gate's hmx4 blocks, the up's
    uint32_t q_blk;
    int k, total, nw, q4i;
    volatile int ready[PIPE_RING];
    volatile int consumed;
    volatile int failed;
    // the activation: block b's gate and up croutons in gs / us[b % GLU_BUFS] once posted; its results into o
    uint8_t *gs[GLU_BUFS], *us[GLU_BUFS], *o;
    int n_blk, tiles, t0, pt_all, part_blk, act;
    volatile int posted;               // the last block both of whose products are in
    volatile int done[GLU_HELPERS];    // each helper's last block done
    volatile int inline_act;           // no helpers: this thread applies it
} glu_t;

typedef struct {
    glu_t *p;
    int w;
} glu_arg_t;

static void glu_worker(void *arg) {
    glu_arg_t *a = (glu_arg_t *)arg;
    glu_t *p = a->p;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        __atomic_store_n(&p->failed, 1, __ATOMIC_RELEASE);
        qurt_thread_exit(0);
    }
    for (int g = a->w; g < p->total; g += p->nw) {
        while (__atomic_load_n(&p->consumed, __ATOMIC_ACQUIRE) < g - PIPE_RING) {
        }
        hvx_copy(p->stage[a->w], p->w[g & 1] + (size_t)(g >> 1) * p->q_blk, p->q_blk);
        dequant_block(p->slots[g % PIPE_RING], p->stage[a->w], p->k, p->q4i);
        __atomic_store_n(&p->ready[g % PIPE_RING], g, __ATOMIC_RELEASE);
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

// act(gate) * up in half precision (qf16 between the steps, so each step is one instruction: the IEEE hf intrinsics
// compile to a qf op and a conversion each, a product through a qf32 pair - twice the instructions on one chain),
// 64 lanes an op. gelu in its tanh form as the engine's gelu_fast (g sigmoid(z), z = 2 sqrt(2/pi) (g + 0.044715 g^3)),
// or silu (z = g). sigmoid(|z|) = 1 / (1 + e^-|z|), sigmoid(-|z|) = e^-|z| sigmoid(|z|): e^-|z| never overflows.
// e^-|z| = 2^-y (y = |z| log2 e, log2 e folded into z's constants, 24 at most) = 2^-f (a quartic, f in [-0.5, 1)
// whichever way the conversion rounds) times 2^-n, that in two normal powers of two (down to 2^-13, the rest) applied
// last, so a negative g's result leaves the normal range only where its value does. 1 / (1 + e) from 24/17 - 8/17 x
// (6% at most) and two Newton steps, on -(1 + e) (v79 has no hf - qf16). Against the CPU's fp32 rounded once to fp16,
// in hf arithmetic: rms 5e-4 of a value ~1 (the rounding alone 2e-4, the q4 weights' error ~100 times that)
static inline uint16_t glu_hb(float f) {
    union {
        __fp16 h;
        uint16_t b;
    } u;
    u.h = (__fp16)f;
    return u.b;
}
static inline HVX_Vector glu_h(float f) { return Q6_Vh_vsplat_R(glu_hb(f)); }
static inline int glu_r(float f) { return (int)((uint32_t)glu_hb(f) * 0x10001u); }   // both halves

typedef struct {
    HVX_Vector a, c3, c2, c1, c0, r0, two, m1, one, one_q, abs, y_max, n_max, e_one, sign;
    int b, c4, r1, log2e;
} glu_k;

static inline void glu_k_init(glu_k *k) {
    k->a = glu_h(2.3022082f);   // 2 sqrt(2 / pi) log2 e
    k->b = glu_r(0.10294324f);  // ... times 0.044715
    k->log2e = glu_r(1.4426950409f);
    k->c4 = glu_r(0.0079791746f);
    k->c3 = glu_h(-0.055424855f);
    k->c2 = glu_h(0.24068041f);
    k->c1 = glu_h(-0.69320874f);
    k->c0 = glu_h(0.99998338f);
    k->r0 = glu_h(24.0f / 17.0f);
    k->r1 = glu_r(8.0f / 17.0f);
    k->two = glu_h(2.0f);
    k->m1 = glu_h(-1.0f);
    k->one = glu_h(1.0f);
    k->one_q = Q6_Vqf16_vadd_VhfVhf(k->one, Q6_V_vzero());
    k->abs = Q6_Vh_vsplat_R(0x7FFF);
    k->y_max = Q6_Vh_vsplat_R(glu_hb(24.0f));
    k->n_max = Q6_Vh_vsplat_R(13);
    k->e_one = Q6_Vh_vsplat_R(15 << 10);
    k->sign = Q6_Vh_vsplat_R(0x8000);
}

static inline HVX_Vector glu_qf(const glu_k *k, HVX_Vector g, HVX_Vector u, int act) {
    HVX_Vector y;
    if (act) {
        y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfRhf(g, k->log2e));
    } else {
        const HVX_Vector t = Q6_Vqf16_vmpy_VhfVhf(g, g);
        const HVX_Vector w = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Rhf(t, k->b), k->a);
        y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(w, g));
    }
    y = Q6_Vh_vmin_VhVh(Q6_V_vand_VV(y, k->abs), k->y_max);   // |y| <= 24 (on the bits: both positive)
    const HVX_Vector n = Q6_Vh_equals_Vhf(y);
    const HVX_Vector f = Q6_Vqf16_vsub_VhfVhf(y, Q6_Vhf_equals_Vh(n));
    HVX_Vector q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Rhf(f, k->c4), k->c3);
    q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), k->c2);
    q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), k->c1);
    q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), k->c0);   // 2^-f
    const HVX_Vector n1 = Q6_Vh_vmin_VhVh(n, k->n_max), n2 = Q6_Vh_vsub_VhVh(n, n1);
    const HVX_Vector s1 = Q6_Vh_vsub_VhVh(k->e_one, Q6_Vh_vasl_VhR(n1, 10));   // 2^-n1, 2^-n2 (hf)
    const HVX_Vector s2 = Q6_Vh_vsub_VhVh(k->e_one, Q6_Vh_vasl_VhR(n2, 10));
    const HVX_Vector e1 = Q6_Vqf16_vmpy_Vqf16Vhf(q, s1);
    // -x = -(1 + e^-|z|), in [-2, -1]
    const HVX_Vector xn = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(e1, Q6_V_vor_VV(s2, k->sign)), k->m1);
    HVX_Vector r = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Rhf(xn, k->r1), k->r0);
    r = Q6_Vqf16_vmpy_Vqf16Vqf16(r, Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(xn, r), k->two));
    r = Q6_Vqf16_vmpy_Vqf16Vqf16(r, Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(xn, r), k->two));
    const HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), g);   // the sign bit
    HVX_Vector o = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(g, u), r);
    o = Q6_Vqf16_vmpy_Vqf16Vqf16(o, Q6_V_vmux_QVV(neg, e1, k->one_q));
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(o, Q6_V_vmux_QVV(neg, s2, k->one)));
}

// a crouton's 1024 results: act(gate) * up (the layout stays: an elementwise product), two chains in flight
static void glu_crouton(const glu_k *k, const uint8_t *gs, const uint8_t *us, uint8_t *dst, int act) {
    const HVX_Vector *gv = (const HVX_Vector *)gs, *uv = (const HVX_Vector *)us;
    HVX_Vector *ov = (HVX_Vector *)dst;
    if (act) {
#pragma clang loop unroll_count(2)
        for (int i = 0; i < 16; ++i) ov[i] = glu_qf(k, gv[i], uv[i], 1);
    } else {
#pragma clang loop unroll_count(2)
        for (int i = 0; i < 16; ++i) ov[i] = glu_qf(k, gv[i], uv[i], 0);
    }
}

// block b's results, parts-major: part b / part_blk, its row tile, its crouton b % part_blk
static void glu_block(const glu_k *k, const glu_t *p, int b) {
    const uint8_t *G = p->gs[b % GLU_BUFS], *U = p->us[b % GLU_BUFS];
    const size_t part = (size_t)(b / p->part_blk), cc = (size_t)(b % p->part_blk);
    for (int t = 0; t < p->tiles; ++t)
        glu_crouton(k, G + (size_t)t * 2048u, U + (size_t)t * 2048u,
                    p->o + ((part * (size_t)p->pt_all + (size_t)(p->t0 + t)) * (size_t)p->part_blk + cc) * 2048u,
                    p->act);
}

static void glu_helper(void *arg) {
    glu_arg_t *a = (glu_arg_t *)arg;
    glu_t *p = a->p;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        __atomic_store_n(&p->failed, 1, __ATOMIC_RELEASE);
        qurt_thread_exit(0);
    }
    glu_k kk;
    glu_k_init(&kk);
    for (int b = a->w; b < p->n_blk; b += GLU_HELPERS) {
        while (__atomic_load_n(&p->posted, __ATOMIC_ACQUIRE) < b)
            if (__atomic_load_n(&p->failed, __ATOMIC_ACQUIRE) || __atomic_load_n(&p->inline_act, __ATOMIC_ACQUIRE))
                goto out;
        glu_block(&kk, p, b);
        __atomic_store_n(&p->done[a->w], b, __ATOMIC_RELEASE);
    }
out:
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}
static void *g_glu_stacks[GLU_HELPERS];

static int glu_ptr(npu_t *s, const uint8_t *a, const uint8_t *gw, const uint8_t *uw, uint8_t *o, int m, int k, int n,
                   int g_shift, int u_shift, int q4i, int part_blk, int act) {
    int e = acquire(s);
    if (e) return e;
    const int croutons = k / 32, n_blk = n / 32, pt_all = m / 32;
    const uint32_t blk_bytes = (uint32_t)k * 64u, q_blk = HMX4_BYTES(k), tile_bytes = (uint32_t)k * 64u;
    glu_k kk;
    glu_k_init(&kk);
    // VTCM: the two matrices' biases | ring | the workers' staging | per pass: GLU_BUFS blocks' gate and up croutons,
    // the activation tiles
    const uint32_t fixed = 4096u + blk_bytes * PIPE_RING + q_blk * PIPE_WORKERS, per_tile = tile_bytes + 4096u * GLU_BUFS;
    if (fixed + per_tile > s->vtcm_size) return AEE_ENOMEMORY;
    int pass_tiles = (int)((s->vtcm_size - fixed) / per_tile);
    if (pass_tiles > pt_all) pass_tiles = pt_all;
    uint8_t *bias_g = s->vtcm, *bias_u = bias_g + 2048, *W = bias_u + 2048;
    uint8_t *GU = W + (size_t)blk_bytes * PIPE_RING + (size_t)q_blk * PIPE_WORKERS;
    uint8_t *A = GU + (size_t)pass_tiles * 4096u * GLU_BUFS;
    memset(bias_g, 0, 4096);
    for (int ch = 0; ch < 32; ++ch) {   // output bias 0, scale 2^-shift (gemm's: normal down to 2^-14, then subnormal)
        ((uint32_t *)bias_g)[ch] = g_shift <= 14 ? (uint32_t)(15 - g_shift) << 10 : 1u << (24 - g_shift);
        ((uint32_t *)bias_u)[ch] = u_shift <= 14 ? (uint32_t)(15 - u_shift) << 10 : 1u << (24 - u_shift);
    }
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio <= 0 || prio >= 255) prio = 127;
    for (int t0 = 0; t0 < pt_all && !e; t0 += pass_tiles) {
        const int tiles = pt_all - t0 < pass_tiles ? pt_all - t0 : pass_tiles;
        hvx_copy(A, a + (size_t)t0 * tile_bytes, (uint32_t)tiles * tile_bytes);
        glu_t p;
        memset(&p, 0, sizeof p);
        p.w[0] = gw;
        p.w[1] = uw;
        p.q_blk = q_blk;
        p.k = k;
        p.total = 2 * n_blk;
        p.nw = PIPE_WORKERS;
        p.q4i = q4i;
        p.consumed = -1;
        for (int i = 0; i < PIPE_RING; ++i) {
            p.slots[i] = W + (size_t)i * blk_bytes;
            p.ready[i] = -1;
        }
        for (int i = 0; i < PIPE_WORKERS; ++i) p.stage[i] = W + (size_t)PIPE_RING * blk_bytes + (size_t)i * q_blk;
        for (int i = 0; i < GLU_BUFS; ++i) {
            p.gs[i] = GU + (size_t)i * (size_t)tiles * 4096u;
            p.us[i] = p.gs[i] + (size_t)tiles * 2048u;
        }
        p.o = o;
        p.n_blk = n_blk;
        p.tiles = tiles;
        p.t0 = t0;
        p.pt_all = pt_all;
        p.part_blk = part_blk;
        p.act = act;
        p.posted = -1;
        for (int i = 0; i < GLU_HELPERS; ++i) p.done[i] = -1;
        glu_arg_t args[PIPE_WORKERS], hargs[GLU_HELPERS];
        qurt_thread_t tid[PIPE_WORKERS], htid[GLU_HELPERS];
        int started = 0, helpers = 0;
        for (int w = 0; w < p.nw; ++w) {
            qurt_thread_attr_t at;
            qurt_thread_attr_init(&at);
            if (!g_pipe_stacks[w] && !(g_pipe_stacks[w] = memalign(128, PIPE_STACK))) break;
            qurt_thread_attr_set_name(&at, (char *)"geode_glu");
            qurt_thread_attr_set_stack_addr(&at, g_pipe_stacks[w]);
            qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
            qurt_thread_attr_set_priority(&at, (unsigned short)prio);
            args[w].p = &p;
            args[w].w = w;
            if (qurt_thread_create(&tid[w], &at, glu_worker, &args[w]) != QURT_EOK) break;
            ++started;
        }
        if (started < p.nw) p.failed = 1;
        for (int h = 0; h < GLU_HELPERS && !p.failed; ++h) {
            qurt_thread_attr_t at;
            qurt_thread_attr_init(&at);
            if (!g_glu_stacks[h] && !(g_glu_stacks[h] = memalign(128, PIPE_STACK))) break;
            qurt_thread_attr_set_name(&at, (char *)"geode_glu_act");
            qurt_thread_attr_set_stack_addr(&at, g_glu_stacks[h]);
            qurt_thread_attr_set_stack_size(&at, PIPE_STACK);
            qurt_thread_attr_set_priority(&at, (unsigned short)prio);
            hargs[h].p = &p;
            hargs[h].w = h;
            if (qurt_thread_create(&htid[h], &at, glu_helper, &hargs[h]) != QURT_EOK) break;
            ++helpers;
        }
        if (helpers < GLU_HELPERS) __atomic_store_n(&p.inline_act, 1, __ATOMIC_RELEASE);
        for (int g = 0; g < p.total && !p.failed; ++g) {
            while (__atomic_load_n(&p.ready[g % PIPE_RING], __ATOMIC_ACQUIRE) != g)
                if (__atomic_load_n(&p.failed, __ATOMIC_ACQUIRE)) break;
            if (p.failed) break;
            const uint8_t *wb = p.slots[g % PIPE_RING];
            const int up = g & 1, b = g >> 1, j = b % GLU_BUFS;
            if (!up && !p.inline_act)   // block b - GLU_BUFS's activation out of buffer j before it is overwritten
                while (__atomic_load_n(&p.done[b % GLU_HELPERS], __ATOMIC_ACQUIRE) < b - GLU_BUFS)
                    if (__atomic_load_n(&p.failed, __ATOMIC_ACQUIRE)) break;
            if (p.failed) break;
            Q6_bias_mxmem2_A(up ? bias_u : bias_g);
            uint8_t *dst = up ? p.us[j] : p.gs[j];
            for (int t = 0; t < tiles; ++t) tile(A + (size_t)t * tile_bytes, wb, dst + (size_t)t * 2048u, croutons);
            __atomic_store_n(&p.consumed, g - 1, __ATOMIC_RELEASE);   // one behind: HMX may still read this slot
            if (!up) continue;
            if (p.inline_act) {
                glu_block(&kk, &p, b);
            } else {
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                __atomic_store_n(&p.posted, b, __ATOMIC_RELEASE);
            }
        }
        __atomic_store_n(&p.consumed, p.total, __ATOMIC_RELEASE);
        for (int w = 0; w < started; ++w) {
            int st;
            qurt_thread_join(tid[w], &st);
        }
        for (int h = 0; h < helpers; ++h) {   // (their last blocks: done when they return)
            int st;
            qurt_thread_join(htid[h], &st);
        }
        if (p.failed) e = AEE_ENOTALLOWED;
    }
    return e;
}

int geode_npu_gemm_glu(remote_handle64 h, const unsigned char *a, int a_len, int g_fd, int u_fd, unsigned char *o,
                       int o_len, int m, int k, int n, int g_shift, int u_shift, int part_blk, int act, int64_t *usec) {
    npu_t *s = (npu_t *)(uintptr_t)h;
    *usec = 0;
    const int q4i = (g_shift & HMX4_Q4I) != 0;
    if (q4i != ((u_shift & HMX4_Q4I) != 0)) return AEE_EBADPARM;
    g_shift &= ~HMX4_Q4I;
    u_shift &= ~HMX4_Q4I;
    if (m <= 0 || k <= 0 || n <= 0 || (m & 31) || (k & 31) || (n & 31) || part_blk <= 0 || (n / 32) % part_blk ||
        g_shift < 0 || g_shift > 24 || u_shift < 0 || u_shift > 24 || act < 0 || act > 1 ||
        (uint32_t)a_len < (uint32_t)m * (uint32_t)k * 2u || (uint32_t)o_len < (uint32_t)m * (uint32_t)n * 2u ||
        ((uintptr_t)a & 127) || ((uintptr_t)o & 127))
        return AEE_EBADPARM;
    void *gv = NULL, *uv = NULL;
    uint64 pa = 0;
    int e = HAP_mmap_get(g_fd, &gv, &pa);
    if (e) return e;
    if ((e = HAP_mmap_get(u_fd, &uv, &pa))) {
        HAP_mmap_put(g_fd);
        return e;
    }
    const uint64_t t0 = HAP_perf_get_time_us();
    e = glu_ptr(s, a, (const uint8_t *)gv, (const uint8_t *)uv, o, m, k, n, g_shift, u_shift, q4i, part_blk, act);
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    HAP_mmap_put(u_fd);
    HAP_mmap_put(g_fd);
    return e;
}

#else   // v73 / v75: the prompt's HMX products only
int geode_npu_gemm_glu(remote_handle64 h, const unsigned char *a, int a_len, int g_fd, int u_fd, unsigned char *o,
                       int o_len, int m, int k, int n, int g_shift, int u_shift, int part_blk, int act, int64_t *usec) {
    (void)h, (void)a, (void)a_len, (void)g_fd, (void)u_fd, (void)o, (void)o_len, (void)m, (void)k, (void)n;
    (void)g_shift, (void)u_shift, (void)part_blk, (void)act;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_moe(remote_handle64 h, const unsigned char *desc, int desc_len, int io_fd, int64_t *usec) {
    (void)h, (void)desc, (void)desc_len, (void)io_fd;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_gemv8(remote_handle64 h, const unsigned char *descs, int descs_len, const unsigned char *act, int act_len,
                    unsigned char *y, int y_len, int w_fd, int threads, int64_t *usec) {
    (void)h, (void)descs, (void)descs_len, (void)act, (void)act_len, (void)y, (void)y_len, (void)w_fd, (void)threads;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
#endif
