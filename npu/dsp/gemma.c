// npu/dsp/gemma.c - Gemma 4 on the DSP: a prompt's attention for one layer on HMX (npu/attn_cfg.h), its chunk of
// queries against the engine's int8 KV cache, which the DSP maps; and every layer of a prompt's chunk (npu/gdec_cfg.h,
// gdec_* at the end).
//
// Per KV head, the positions the chunk reads, a superblock at a time (as many as VTCM holds; usually one): their keys
// and values as fp16 HMX weight blocks, built once (int8 x the position's scale) - the keys transposed by a 32 x 32
// word shuffle network (a block: 32 positions as its output columns, the head's dims as its inputs), the values by
// one halfword shuffle per position pair (32 dims as the outputs, the positions as the inputs). A row tile is 32
// (token, query head of the group) rows: S = Q K^T on HMX into one crouton per 32 positions, the masked softmax on
// HVX in floats - each crouton then holds the weights in the activation layout P V reads (the layouts match: rows
// pairwise, 32 channels) - and O = P V on HMX (2^-4 on the way out: an unnormalized sum of up to a superblock of
// weights stays inside fp16), the rows written out divided by their sums, or joined with the earlier superblocks'
// as an online softmax.
//
// Threads: this one (the FastRPC thread holds HMX) multiplies; the helpers (HVX) take a row tile each in turn - its
// query tile, its softmax, its rows out - handing each step to HMX and back through counters of their own.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "geode_npu.h"
#include "attn_cfg.h"
#include "gdec_cfg.h"
#include "nar_cfg.h"
#include "wsrc.h"

#if __HVX_ARCH__ >= 79
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "hmx_hexagon_protos.h"
#include "qurt.h"

int geode_npu_vtcm(remote_handle64 h, uint8_t **vtcm, unsigned *size);   // npu_imp.c

#define A_HELPERS_MAX 5
#define A_HELPERS 4
#define A_STACK 16384
#define A_STAGES 1024
#define PV_SHIFT 4   // P V's results 2^-4 on the way out of HMX (taken back in floats)

typedef struct {
    // the call
    const attn_desc_t *d;
    const float *q;
    float *o;
    uint8_t *kv[ATTN_MAX_CHUNKS];
    int group, tpt, tiles;    // query heads a KV head, tokens a row tile, row tiles a KV head
    int hd, dgs;              // the head dim, its 32-groups
    int multi;                // more than one superblock: the rows joined (m, l), divided at the end
    // VTCM
    uint8_t *bias_s, *bias_pv;
    uint8_t *K, *V;           // the superblock's keys [block][dg][16 vectors], values [dg][block][16 vectors]
    float *m, *l;             // a row's running max and sum (row t nh + h)
    uint8_t *qs[A_HELPERS_MAX], *ss[A_HELPERS_MAX], *os[A_HELPERS_MAX], *scr[A_HELPERS_MAX + 1];
    int sb_max;               // positions a superblock at most (a multiple of 32)
    // the superblock
    int kh, sb_lo, sb_hi, npb;
    // the threads
    int nh_, nthreads;        // helpers; helpers + this one
    int arrive;
    volatile int sense;
    int ctr[A_STAGES];
    volatile int q_ready[A_HELPERS_MAX], s_done[A_HELPERS_MAX], p_ready[A_HELPERS_MAX], o_done[A_HELPERS_MAX];
    volatile int h_done[A_HELPERS_MAX];
    volatile int started, failed;
    int no_hmx, sm_ieee, sm_check, sm_v1;
    int e_up;                 // the weights 2^e_up larger than e^(s - max) (softmax_h2: E_UP; the others 0)
    int log2;                 // ATTN_LOG2: the scores in log2 units (softmax_h2 only)
    int sum_acc;              // softmax_h2's interior row sums by sf += hf x 1 (g_acc_from's order; not ATTN_SUM_CVT)
} at_t;

typedef struct {
    at_t *a;
    int th;
} at_arg_t;

static void *g_a_stacks[A_HELPERS_MAX];

// ---- lane orders, checked once ------------------------------------------------------------------------------------
// vcvt between sf and hf (npu_imp.c's moe_cvt_check); vunpack's (in order or dealt); vshuff's (which input lands on
// the even elements, which half of the pair comes first). Any order other than the expected one: unsupported.
static int g_cvt_to, g_cvt_from, g_orders_ok = -1;
// the layers' steps' qf32 pairs (q_of_hf, hf_of_q): hf -> qf32 with the even rows' lanes in lo; qf32 -> hf from
// combine(odd, even) with the even rows on the even lanes. Checked (anything else: unsupported)
static int g_q_from_ok, g_q_to_ok;
// sf += vmpy(hf, hf)'s pair (softmax_h2's row sums): 1 the even lanes' products in lo, 2 in hi, -1 neither (the sums
// then as before: ATTN_SUM_CVT's)
static int g_acc_from = -1;

static inline void q_of_hf(HVX_Vector h, HVX_Vector *even, HVX_Vector *odd);
static inline HVX_Vector hf_of_q(HVX_Vector even, HVX_Vector odd);

static void a_checks(void) {
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
    // vunpack: lo = bytes 0..63 as halfwords, hi = 64..127
    int8_t by[128] __attribute__((aligned(128)));
    for (int i = 0; i < 128; ++i) by[i] = (int8_t)(i - 64);
    int16_t hw[128] __attribute__((aligned(128)));
    const HVX_VectorPair u = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)by);
    *(HVX_Vector *)hw = Q6_V_lo_W(u);
    *(HVX_Vector *)(hw + 64) = Q6_V_hi_W(u);
    int unpack_ok = 1;
    for (int i = 0; i < 128; ++i) unpack_ok &= hw[i] == i - 64;
    // vshuff(b, a, -4): a's words at the even elements, the first 32 elements in lo
    int32_t wa[32] __attribute__((aligned(128))), wb[32] __attribute__((aligned(128)));
    int32_t sw[64] __attribute__((aligned(128)));
    for (int i = 0; i < 32; ++i) wa[i] = i, wb[i] = 100 + i;
    const HVX_VectorPair s = Q6_W_vshuff_VVR(*(const HVX_Vector *)wb, *(const HVX_Vector *)wa, -4);
    *(HVX_Vector *)sw = Q6_V_lo_W(s);
    *(HVX_Vector *)(sw + 32) = Q6_V_hi_W(s);
    int shuff_ok = 1;
    for (int i = 0; i < 32; ++i) shuff_ok &= sw[2 * i] == i && sw[2 * i + 1] == 100 + i;
    // the qf32 pairs the layers' steps use
    for (int i = 0; i < 32; ++i) h[2 * i] = (__fp16)a[i], h[2 * i + 1] = (__fp16)b[i];
    {
        HVX_Vector e, o;
        q_of_hf(*(const HVX_Vector *)h, &e, &o);
        *(HVX_Vector *)lo = Q6_Vsf_equals_Vqf32(e);
        *(HVX_Vector *)hi = Q6_Vsf_equals_Vqf32(o);
        g_q_from_ok = 1;
        for (int i = 0; i < 32; ++i) g_q_from_ok &= lo[i] == a[i] && hi[i] == b[i];
        __fp16 h2[64] __attribute__((aligned(128)));
        *(HVX_Vector *)h2 = hf_of_q(Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)a, Q6_V_vzero()),
                                    Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)b, Q6_V_vzero()));
        g_q_to_ok = 1;
        for (int i = 0; i < 32; ++i) g_q_to_ok &= (float)h2[2 * i] == a[i] && (float)h2[2 * i + 1] == b[i];
    }
    {
        for (int i = 0; i < 32; ++i) h[2 * i] = (__fp16)a[i], h[2 * i + 1] = (__fp16)b[i];
        HVX_VectorPair w2 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        w2 = Q6_Wsf_vmpyacc_WsfVhfVhf(w2, *(const HVX_Vector *)h, Q6_Vh_vsplat_R(0x3C00));
        w2 = Q6_Wsf_vmpyacc_WsfVhfVhf(w2, *(const HVX_Vector *)h, Q6_Vh_vsplat_R(0x3C00));   // (twice: the sum)
        *(HVX_Vector *)lo = Q6_V_lo_W(w2);
        *(HVX_Vector *)hi = Q6_V_hi_W(w2);
        int al = 1, ah = 1;
        for (int i = 0; i < 32; ++i) {
            al &= lo[i] == 2.0f * a[i] && hi[i] == 2.0f * b[i];
            ah &= hi[i] == 2.0f * a[i] && lo[i] == 2.0f * b[i];
        }
        g_acc_from = al ? 1 : ah ? 2 : -1;
    }
    g_orders_ok = g_cvt_to > 0 && g_cvt_from > 0 && unpack_ok && shuff_ok && g_q_from_ok && g_q_to_ok;
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

// ---- floats ---------------------------------------------------------------------------------------------------------

static inline HVX_Vector f_sp(float f) {
    union { float f; int32_t i; } u = {f};
    return Q6_V_vsplat_R(u.i);
}
static inline HVX_Vector f_mul(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vmpy_VsfVsf(a, b); }
static inline HVX_Vector f_add(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vadd_VsfVsf(a, b); }
static inline HVX_Vector f_sub(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vsub_VsfVsf(a, b); }

// e^x for x <= 0 (2^n times e^t to t^5, |t| < ln 2: 1.5e-4; the weights end in fp16 anyway)
static inline HVX_Vector f_exp(HVX_Vector x) {
    HVX_Vector y = f_mul(x, f_sp(1.44269504f));
    y = Q6_Vsf_vmax_VsfVsf(y, f_sp(-125.0f));
    const HVX_Vector n = Q6_Vw_equals_Vsf(y);
    const HVX_Vector t = f_mul(f_sub(y, Q6_Vsf_equals_Vw(n)), f_sp(0.69314718f));
    HVX_Vector p = f_sp(1.0f / 120.0f);
    p = f_add(f_mul(p, t), f_sp(1.0f / 24.0f));
    p = f_add(f_mul(p, t), f_sp(1.0f / 6.0f));
    p = f_add(f_mul(p, t), f_sp(0.5f));
    p = f_add(f_mul(p, t), f_sp(1.0f));
    p = f_add(f_mul(p, t), f_sp(1.0f));
    return Q6_Vw_vadd_VwVw(p, Q6_Vw_vasl_VwR(n, 23));
}

static inline float f_hmax(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1) v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, s));
    HVX_Vector t __attribute__((aligned(128))) = v;
    return ((const float *)&t)[0];
}
static inline float f_hsum(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1) v = f_add(v, Q6_V_vror_VR(v, s));
    HVX_Vector t __attribute__((aligned(128))) = v;
    return ((const float *)&t)[0];
}

static inline uint16_t hf_bits(float f) {
    union {
        __fp16 h;
        uint16_t b;
    } u;
    u.h = (__fp16)f;
    return u.b;
}

// ---- HMX ------------------------------------------------------------------------------------------------------------

// one 32 x 32 result: the activation tile (croutons x 32 channels) times a 32-column weight block, into a crouton
static inline void tile(const uint8_t *act, const uint8_t *wgt, uint8_t *out, int croutons) {
    for (int c = 0; c < croutons; c += 32) {
        const int n = croutons - c < 32 ? croutons - c : 32;
        Q6_activation_hf_mxmem_RR_deep((uint32_t)(uintptr_t)(act + c * 2048), ((uint32_t)(n - 1) << 11) | 0x7C);
        Q6_weight_hf_mxmem_RR((uint32_t)(uintptr_t)(wgt + c * 2048), (uint32_t)n * 2048u - 1u);
    }
    Q6_cvt_hf_acc_R(2);   // bias set 0, the accumulator cleared; bit 1 keeps negative results
    Q6_mxmem_cvt_RR((void *)out, 0);
}

// ---- the geometry ---------------------------------------------------------------------------------------------------

static inline int lo_of(const attn_desc_t *d, int t) {   // the first position token t sees
    if (t >= d->n) return 0x7FFFFFFF;
    if (!d->window) return 0;
    const int lo = d->apos + t - d->window + 1;
    return lo > 0 ? lo : 0;
}
// an encoder's keys (attn_cfg.h nk: every query sees them all)
static inline int keys_of(const attn_desc_t *d) { return d->nk ? d->nk : d->n; }
static inline int hi_of(const attn_desc_t *d, int t) {   // the last one (an encoder's tokens: every one)
    if (t >= d->n) return -1;
    if (d->flags & ATTN_ENCODER) return (d->flags & ATTN_CAUSAL) ? keys_of(d) - d->n + t : keys_of(d) - 1;
    return d->apos + t;
}
// one past the last position any token sees
static inline int end_of(const attn_desc_t *d) { return (d->flags & ATTN_ENCODER) ? keys_of(d) : d->apos + d->n; }

typedef struct {
    int t0, t1;   // the tile's tokens (t1 inclusive, < n)
    int b0, nb;   // its 32-position blocks in the superblock
    int ok;
} geo_t;

static geo_t geo(const at_t *a, int r) {
    const attn_desc_t *d = a->d;
    geo_t g;
    g.t0 = r * a->tpt;
    g.t1 = g.t0 + a->tpt - 1;
    if (g.t1 >= d->n) g.t1 = d->n - 1;
    int lo = lo_of(d, g.t0), hi = hi_of(d, g.t1) + 1;
    if (lo < a->sb_lo) lo = a->sb_lo;
    if (hi > a->sb_hi) hi = a->sb_hi;
    g.ok = lo < hi;
    g.b0 = g.ok ? (lo - a->sb_lo) / 32 : 0;
    g.nb = g.ok ? (hi - a->sb_lo + 31) / 32 - g.b0 : 0;
    return g;
}

// ---- the keys and values --------------------------------------------------------------------------------------------

static inline const int8_t *kv_row(const at_t *a, int pos, int value) {
    const attn_desc_t *d = a->d;
    const uint8_t *base = a->kv[pos / d->kv_chunk];
    return (const int8_t *)(base + ((size_t)(value * d->n_attn + d->slot) * (size_t)d->kv_chunk + (size_t)(pos % d->kv_chunk)) *
                                       (size_t)d->kv_row);
}
static inline float kv_sc(const at_t *a, int pos, int value, int kh) {
    const attn_desc_t *d = a->d;
    const uint8_t *base = a->kv[pos / d->kv_chunk];
    const float *sc = (const float *)(base + 2u * (size_t)d->n_attn * (size_t)d->kv_chunk * (size_t)d->kv_row);
    return sc[((size_t)(value * d->n_attn + d->slot) * (size_t)d->kv_chunk + (size_t)(pos % d->kv_chunk)) *
                  (size_t)d->kv_heads + (size_t)kh];
}

// 32 positions' rows of KV head kh (keys or values) as fp16, x their scales, into rows[32][hd]; unwritten ones 0
static void conv_rows(const at_t *a, int p0, int value, __fp16 *rows) {
    const attn_desc_t *d = a->d;
    const int hd = a->hd, end = d->apos + d->n;
    for (int p = 0; p < 32; ++p) {
        HVX_Vector *dst = (HVX_Vector *)(rows + (size_t)p * hd);
        const int pos = p0 + p;
        if (pos >= end) {
            for (int c = 0; c < hd / 64; ++c) dst[c] = Q6_V_vzero();
            continue;
        }
        const int8_t *src = kv_row(a, pos, value) + (size_t)a->kh * hd;
        const HVX_Vector s = Q6_Vh_vsplat_R(hf_bits(kv_sc(a, pos, value, a->kh)));
        for (int c = 0; c < hd; c += 128) {
            const HVX_Vector raw = *(const HVX_UVector *)(src + c);
            const HVX_VectorPair w = Q6_Wh_vunpack_Vb(raw);
            dst[c / 64] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(w)), s));
            dst[c / 64 + 1] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(w)), s));
        }
    }
}

// The keys of the superblock's block b: 32 positions as the output columns, the dims as the inputs (wgt layout: per
// 32 dims 16 vectors, vector j = dims 2j, 2j + 1 as a halfword pair a position). Per 64 dims the positions' 32 words
// transposed: five rounds of a perfect shuffle (vector i interleaved with i + 16 into 2i, 2i + 1: each round turns
// the (vector, word) index's ten bits one place, five make it (word, vector))
// an encoder's (ATTN_ENCODER): its blocks as the host made them, copied in - n_bytes from src to dst (128-byte aligned)
static void copy_blocks(uint8_t *dst, const uint8_t *src, size_t n_bytes) {
    HVX_Vector *d = (HVX_Vector *)dst;
    const HVX_Vector *s = (const HVX_Vector *)src;
    for (size_t i = 0; i < n_bytes / 128u; ++i) d[i] = s[i];
}
// a head's 32-position blocks in all, and those in the first two buffers (attn_cfg.h npre: the rest in kv[2], kv[3])
static inline size_t enc_blocks(const at_t *a) { return (size_t)(keys_of(a->d) + 31) / 32u; }
static inline size_t enc_pre(const at_t *a) { return a->d->npre ? (size_t)a->d->npre : enc_blocks(a); }
// block blk of the KV head's keys (value 0) or of its values' dim group dg (value 1), where the host put it
static inline const uint8_t *enc_block(const at_t *a, size_t blk, int value, int dg) {
    const size_t pre = enc_pre(a), dgs = (size_t)a->dgs;
    const int tail = blk >= pre;
    const size_t nb = tail ? enc_blocks(a) - pre : pre, j = tail ? blk - pre : blk;
    const uint8_t *base = a->kv[2 * tail + value];
    return value ? base + (((size_t)a->kh * dgs + (size_t)dg) * nb + j) * 2048u : base + ((size_t)a->kh * nb + j) * dgs * 2048u;
}

static void build_k(at_t *a, int th, int b) {
    const int hd = a->hd;
    if (a->d->flags & ATTN_ENCODER) {
        const size_t blk = (size_t)a->sb_lo / 32u + (size_t)b, bytes = (size_t)a->dgs * 2048u;
        copy_blocks(a->K + (size_t)b * bytes, enc_block(a, blk, 0, 0), bytes);
        return;
    }
    __fp16 *rows = (__fp16 *)a->scr[th];
    HVX_Vector *t0 = (HVX_Vector *)(a->scr[th] + (size_t)32 * hd * 2), *t1 = t0 + 32;
    conv_rows(a, a->sb_lo + 32 * b, 0, rows);
    uint8_t *dst = a->K + (size_t)b * (size_t)a->dgs * 2048u;
    for (int c = 0; c < hd / 64; ++c) {
        const HVX_Vector *src = (const HVX_Vector *)rows + c;   // position p's vector at src[p * hd / 64]
        const int stride = hd / 64;
        for (int i = 0; i < 16; ++i) {
            const HVX_VectorPair w = Q6_W_vshuff_VVR(src[(i + 16) * stride], src[i * stride], -4);
            t0[2 * i] = Q6_V_lo_W(w);
            t0[2 * i + 1] = Q6_V_hi_W(w);
        }
        HVX_Vector *x = t0, *y = t1;
        for (int round = 1; round < 5; ++round) {
            HVX_Vector *out = round == 4 ? (HVX_Vector *)(dst + (size_t)c * 4096u) : y;
            for (int i = 0; i < 16; ++i) {
                const HVX_VectorPair w = Q6_W_vshuff_VVR(x[i + 16], x[i], -4);
                out[2 * i] = Q6_V_lo_W(w);
                out[2 * i + 1] = Q6_V_hi_W(w);
            }
            HVX_Vector *s = x;
            x = y;
            y = s;
        }
    }
}

// The values of block b: per dim group, vector j = positions 2j, 2j + 1 interleaved by halfwords over 32 dims
static void build_v(at_t *a, int th, int b) {
    const int hd = a->hd, stride = hd / 64;
    if (a->d->flags & ATTN_ENCODER) {
        const size_t blk = (size_t)a->sb_lo / 32u + (size_t)b;
        for (int dg = 0; dg < a->dgs; ++dg)
            copy_blocks(a->V + ((size_t)dg * a->npb + b) * 2048u, enc_block(a, blk, 1, dg), 2048u);
        return;
    }
    __fp16 *rows = (__fp16 *)a->scr[th];
    conv_rows(a, a->sb_lo + 32 * b, 1, rows);
    const HVX_Vector *src = (const HVX_Vector *)rows;
    for (int j = 0; j < 16; ++j)
        for (int c = 0; c < stride; ++c) {
            const HVX_VectorPair w = Q6_W_vshuff_VVR(src[(2 * j + 1) * stride + c], src[(2 * j) * stride + c], -2);
            *(HVX_Vector *)(a->V + ((size_t)(2 * c) * a->npb + b) * 2048u + (size_t)j * 128u) = Q6_V_lo_W(w);
            *(HVX_Vector *)(a->V + ((size_t)(2 * c + 1) * a->npb + b) * 2048u + (size_t)j * 128u) = Q6_V_hi_W(w);
        }
}

// ---- a row tile -----------------------------------------------------------------------------------------------------

// row rr of a tile: its token and query head
static inline void row_of(const at_t *a, const geo_t *g, int rr, int *t, int *h) {
    *t = g->t0 + rr / a->group;
    *h = a->kh * a->group + rr % a->group;
}

// the queries of the tile's 32 rows as HMX's activation tile: per 32 dims a crouton, row pair i its vector i
static void build_q(const at_t *a, int th, const geo_t *g) {
    const attn_desc_t *d = a->d;
    uint8_t *qt = a->qs[th];
    for (int i = 0; i < 16; ++i) {
        int ta, ha, tb, hb;
        row_of(a, g, 2 * i, &ta, &ha);
        row_of(a, g, 2 * i + 1, &tb, &hb);
        const HVX_Vector *ra = ta < d->n ? (const HVX_Vector *)(a->q + (size_t)ta * d->q_stride + (size_t)ha * a->hd) : NULL;
        const HVX_Vector *rb = tb < d->n ? (const HVX_Vector *)(a->q + (size_t)tb * d->q_stride + (size_t)hb * a->hd) : NULL;
        for (int dg = 0; dg < a->dgs; ++dg)
            *(HVX_Vector *)(qt + (size_t)dg * 2048u + (size_t)i * 128u) =
                hf_of(ra ? ra[dg] : Q6_V_vzero(), rb ? rb[dg] : Q6_V_vzero());
    }
}

// The scores' croutons into the weights in place: per row pair the masked maximum, then e^(s - max) and the sums.
// mx / sm: the 32 rows' maximum (-inf: none of its positions in this superblock) and sum
static void softmax(const at_t *a, uint8_t *S, const geo_t *g, float *mx, float *sm) {
    const attn_desc_t *d = a->d;
    int32_t iota_a[32] __attribute__((aligned(128)));
    for (int i = 0; i < 32; ++i) iota_a[i] = i;
    const HVX_Vector iota = *(const HVX_Vector *)iota_a, ninf = Q6_V_vsplat_R((int)0xFF800000u), zero = Q6_V_vzero();
    // the croutons no row of the tile masks: every position inside [the largest first, the smallest last]
    const int all_lo = lo_of(d, g->t1), all_hi = hi_of(d, g->t0);
    const int base = a->sb_lo + 32 * g->b0;
    for (int i = 0; i < 16; ++i) {
        int ta, ha, tb, hb;
        row_of(a, g, 2 * i, &ta, &ha);
        row_of(a, g, 2 * i + 1, &tb, &hb);
        const int loa = lo_of(d, ta), hia = hi_of(d, ta), lob = lo_of(d, tb), hib = hi_of(d, tb);
        const HVX_Vector vla = Q6_V_vsplat_R(loa - 1), vha = Q6_V_vsplat_R(hia + 1);
        const HVX_Vector vlb = Q6_V_vsplat_R(lob - 1), vhb = Q6_V_vsplat_R(hib + 1);
        HVX_Vector ma = ninf, mb = ninf;
        for (int j = 0; j < g->nb; ++j) {
            HVX_Vector e, o;
            sf_of(*(const HVX_Vector *)(S + (size_t)j * 2048u + (size_t)i * 128u), &e, &o);
            const int p0 = base + 32 * j;
            if (p0 < all_lo || p0 + 31 > all_hi) {
                const HVX_Vector pv = Q6_Vw_vadd_VwVw(Q6_V_vsplat_R(p0), iota);
                const HVX_VectorPred qa = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(pv, vla), Q6_Q_vcmp_gt_VwVw(vha, pv));
                const HVX_VectorPred qb = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(pv, vlb), Q6_Q_vcmp_gt_VwVw(vhb, pv));
                e = Q6_V_vmux_QVV(qa, e, ninf);
                o = Q6_V_vmux_QVV(qb, o, ninf);
            }
            ma = Q6_Vsf_vmax_VsfVsf(ma, e);
            mb = Q6_Vsf_vmax_VsfVsf(mb, o);
        }
        const float fa = f_hmax(ma), fb = f_hmax(mb);
        const int oka = fa > -INFINITY, okb = fb > -INFINITY;
        const HVX_Vector sa = f_sp(oka ? fa : 0.0f), sb = f_sp(okb ? fb : 0.0f);
        HVX_Vector ua = zero, ub = zero;
        for (int j = 0; j < g->nb; ++j) {
            HVX_Vector *v = (HVX_Vector *)(S + (size_t)j * 2048u + (size_t)i * 128u);
            HVX_Vector e, o;
            sf_of(*v, &e, &o);
            e = oka ? f_exp(f_sub(e, sa)) : zero;
            o = okb ? f_exp(f_sub(o, sb)) : zero;
            const int p0 = base + 32 * j;
            if (p0 < all_lo || p0 + 31 > all_hi) {
                const HVX_Vector pv = Q6_Vw_vadd_VwVw(Q6_V_vsplat_R(p0), iota);
                const HVX_VectorPred qa = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(pv, vla), Q6_Q_vcmp_gt_VwVw(vha, pv));
                const HVX_VectorPred qb = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(pv, vlb), Q6_Q_vcmp_gt_VwVw(vhb, pv));
                e = Q6_V_vmux_QVV(qa, e, zero);
                o = Q6_V_vmux_QVV(qb, o, zero);
            }
            ua = f_add(ua, e);
            ub = f_add(ub, o);
            *v = hf_of(e, o);
        }
        mx[2 * i] = fa;
        mx[2 * i + 1] = fb;
        sm[2 * i] = oka ? f_hsum(ua) : 0.0f;
        sm[2 * i + 1] = okb ? f_hsum(ub) : 0.0f;
    }
}

// a qf32 vector's lanes summed (q_hsum's twin, here ahead of it)
static inline float q_hsum_a(HVX_Vector q) {
    for (int s = 64; s >= 4; s >>= 1) q = Q6_Vqf32_vadd_Vqf32Vqf32(q, Q6_V_vror_VR(q, s));
    HVX_Vector t __attribute__((aligned(128))) = Q6_Vsf_equals_Vqf32(q);
    return ((const float *)&t)[0];
}

// The same in hf (the scores come out of HMX in hf, the weights go back in as hf): the maximum on the crouton's 64
// lanes (even: row 2i, odd: row 2i + 1), e^(s - max) = 2^-y in qf16 as npu_imp.c's glu_qf (y = (max - s) log2 e at
// most 24; 2^-f a quartic, times 2^-n in two normal powers), the masks on halfword positions from the superblock's
// first, the sums in qf32 (each hf -> sf, then added: no qf32 product anywhere). One instruction an operation on 64
// lanes, against the IEEE sf one's qf32 op plus conversion on 32
static void softmax_h(const at_t *a, uint8_t *S, const geo_t *g, float *mx, float *sm) {
    const attn_desc_t *d = a->d;
    int16_t iota_a[64] __attribute__((aligned(128)));
    for (int i = 0; i < 64; ++i) iota_a[i] = (int16_t)(i / 2);   // lane 2c + s: position c
    const HVX_Vector iota = *(const HVX_Vector *)iota_a, ninf = Q6_Vh_vsplat_R(0xFC00), zero = Q6_V_vzero();
    const int log2e = (int)((uint32_t)hf_bits(1.4426950409f) * 0x10001u), c4 = (int)((uint32_t)hf_bits(0.0079791746f) * 0x10001u);
    const HVX_Vector c3 = Q6_Vh_vsplat_R(hf_bits(-0.055424855f)), c2 = Q6_Vh_vsplat_R(hf_bits(0.24068041f));
    const HVX_Vector c1 = Q6_Vh_vsplat_R(hf_bits(-0.69320874f)), c0 = Q6_Vh_vsplat_R(hf_bits(0.99998338f));
    const HVX_Vector habs = Q6_Vh_vsplat_R(0x7FFF), y_max = Q6_Vh_vsplat_R(hf_bits(24.0f)), n_max = Q6_Vh_vsplat_R(13);
    const HVX_Vector e_one = Q6_Vh_vsplat_R(15 << 10);
    const int all_lo = lo_of(d, g->t1), all_hi = hi_of(d, g->t0);
    const int base = a->sb_lo + 32 * g->b0;
    for (int i = 0; i < 16; ++i) {
        int ta, ha, tb, hb;
        row_of(a, g, 2 * i, &ta, &ha);
        row_of(a, g, 2 * i + 1, &tb, &hb);
        // the rows' first and last positions from base, as halfwords (clamped: a row past the chunk sees none)
        int la = lo_of(d, ta) - base, ua_ = hi_of(d, ta) - base, lb = lo_of(d, tb) - base, ub_ = hi_of(d, tb) - base;
        la = la < -30000 ? -30000 : la > 30000 ? 30000 : la;
        lb = lb < -30000 ? -30000 : lb > 30000 ? 30000 : lb;
        ua_ = ua_ < -30000 ? -30000 : ua_ > 30000 ? 30000 : ua_;
        ub_ = ub_ < -30000 ? -30000 : ub_ > 30000 ? 30000 : ub_;
        const HVX_Vector vlo = Q6_V_vsplat_R((int)(((uint32_t)(uint16_t)(la - 1)) | ((uint32_t)(uint16_t)(lb - 1) << 16)));
        const HVX_Vector vhi = Q6_V_vsplat_R((int)(((uint32_t)(uint16_t)(ua_ + 1)) | ((uint32_t)(uint16_t)(ub_ + 1) << 16)));
        HVX_Vector m = ninf;
        for (int j = 0; j < g->nb; ++j) {
            HVX_Vector v = *(const HVX_Vector *)(S + (size_t)j * 2048u + (size_t)i * 128u);
            const int p0 = base + 32 * j;
            if (p0 < all_lo || p0 + 31 > all_hi) {
                const HVX_Vector pv = Q6_Vh_vadd_VhVh(iota, Q6_Vh_vsplat_R(32 * j));
                const HVX_VectorPred q = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhVh(pv, vlo), Q6_Q_vcmp_gt_VhVh(vhi, pv));
                v = Q6_V_vmux_QVV(q, v, ninf);
            }
            m = Q6_Vhf_vfmax_VhfVhf(m, v);
        }
        for (int s = 64; s >= 4; s >>= 1) m = Q6_Vhf_vfmax_VhfVhf(m, Q6_V_vror_VR(m, s));   // (parity kept)
        HVX_Vector mt __attribute__((aligned(128))) = m;
        const uint16_t mab = ((const uint16_t *)&mt)[0], mbb = ((const uint16_t *)&mt)[1];
        const int oka = mab != 0xFC00, okb = mbb != 0xFC00;
        const HVX_Vector mv = Q6_V_vsplat_R((int)((uint32_t)(oka ? mab : 0) | ((uint32_t)(okb ? mbb : 0) << 16)));
        const HVX_Vector okm = Q6_V_vsplat_R((int)((oka ? 0xFFFFu : 0u) | (okb ? 0xFFFF0000u : 0u)));
        HVX_Vector sa0 = zero, sb0 = zero, sa1 = zero, sb1 = zero;
        for (int j = 0; j < g->nb; ++j) {
            HVX_Vector *vp = (HVX_Vector *)(S + (size_t)j * 2048u + (size_t)i * 128u);
            const HVX_Vector xh = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(mv, *vp));   // max - s >= 0
            HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfRhf(xh, log2e));
            y = Q6_Vh_vmin_VhVh(Q6_V_vand_VV(y, habs), y_max);   // (on the bits: both positive)
            const HVX_Vector n = Q6_Vh_equals_Vhf(y);
            const HVX_Vector f = Q6_Vqf16_vsub_VhfVhf(y, Q6_Vhf_equals_Vh(n));
            HVX_Vector q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Rhf(f, c4), c3);
            q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), c2);
            q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), c1);
            q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), c0);   // 2^-f
            const HVX_Vector n1 = Q6_Vh_vmin_VhVh(n, n_max), n2 = Q6_Vh_vsub_VhVh(n, n1);
            const HVX_Vector s1 = Q6_Vh_vsub_VhVh(e_one, Q6_Vh_vasl_VhR(n1, 10));   // 2^-n1, 2^-n2 (hf)
            const HVX_Vector s2 = Q6_Vh_vsub_VhVh(e_one, Q6_Vh_vasl_VhR(n2, 10));
            HVX_Vector e = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(q, s1), s2));
            e = Q6_V_vand_VV(e, okm);
            const int p0 = base + 32 * j;
            if (p0 < all_lo || p0 + 31 > all_hi) {
                const HVX_Vector pv = Q6_Vh_vadd_VhVh(iota, Q6_Vh_vsplat_R(32 * j));
                const HVX_VectorPred qm = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhVh(pv, vlo), Q6_Q_vcmp_gt_VhVh(vhi, pv));
                e = Q6_V_vmux_QVV(qm, e, zero);
            }
            *vp = e;
            HVX_Vector ea, eb;
            sf_of(e, &ea, &eb);
            // the sums in sf, each step converted back: a qf32 sum got its terms wrong - the compiler's qfloat pass
            // folded the conversions into mixed qf32 + sf adds and put the qf32 sum into the sf operand (seen in the
            // disassembly: 2.0 for 1.0, 81 for 1.1)
            if (j & 1)
                sa1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sa1, ea)), sb1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sb1, eb));
            else
                sa0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sa0, ea)), sb0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sb0, eb));
        }
        mx[2 * i] = oka ? (float)*(const __fp16 *)&mab : -INFINITY;
        mx[2 * i + 1] = okb ? (float)*(const __fp16 *)&mbb : -INFINITY;
        sm[2 * i] = oka ? f_hsum(f_add(sa0, sa1)) : 0.0f;
        sm[2 * i + 1] = okb ? f_hsum(f_add(sb0, sb1)) : 0.0f;
    }
}

// softmax_h2's weight of one crouton, 2^E_UP larger than softmax_h's: 2^E_UP e^(s - max) = 2^-f 2^(E_UP - n) for
// y = (max - s) log2 e = n + f (y at most 24; ATTN_LOG2: the scores in log2 units already, y = max - s): one power of
// two, 2^(E_UP - n) in [2^-13, 2^11], always a normal hf, and every weight >= 2^-14 a normal hf too. softmax_h's two
// powers 2^-n1 2^-n2 reach subnormal weights, and in a loop without a branch the compiler's lowering flushed those to
// zero (weights under ~6e-5 lost: the rows' error up 28% on 4K keys). 2^-f a cubic fitted with hf coefficients:
// 1.1e-4 at most, under hf's half step (2.4e-4) - softmax_h's quartic, its coefficients rounded to hf, 1.7e-4.
// y >= 0 needs no abs: a lane a mask takes (y < 0, garbage) is zeroed after; the rows' maximum covers the rest
#define E_UP 11
typedef struct {
    int log2e, c3;
    HVX_Vector c2, c1, c0, y_max, e_top;
} ex_t;
static inline __attribute__((always_inline)) HVX_Vector exp_hf(const ex_t *k, HVX_Vector mv, HVX_Vector s, int log2) {
    HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(mv, s));   // max - s
    if (!log2) y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfRhf(y, k->log2e));
    y = Q6_Vh_vmin_VhVh(y, k->y_max);   // (on the bits: both positive)
    const HVX_Vector n = Q6_Vh_equals_Vhf(y);
    const HVX_Vector f = Q6_Vqf16_vsub_VhfVhf(y, Q6_Vhf_equals_Vh(n));
    HVX_Vector q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Rhf(f, k->c3), k->c2);
    q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), k->c1);
    q = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(q, f), k->c0);   // 2^-f
    const HVX_Vector p2 = Q6_Vh_vsub_VhVh(k->e_top, Q6_Vh_vasl_VhR(n, 10));   // 2^(E_UP - n): its exponent field
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(q, p2));
}

// softmax_h restructured for the VLIW packets: the croutons no row of the tile masks - all of a prompt's but the
// edges, all of an encoder's but the last - in loops without a branch (the compiler pipelines them: +60% at 1024 x
// 16K keys), the maximum over four chains, the edges masked as before; the weights 2^E_UP larger (exp_hf), P V's
// output scale and rows_out's 2^E_UP smaller (at_t e_up)
static void softmax_h2(const at_t *a, uint8_t *S, const geo_t *g, float *mx, float *sm) {
    const attn_desc_t *d = a->d;
    int16_t iota_a[64] __attribute__((aligned(128)));
    for (int i = 0; i < 64; ++i) iota_a[i] = (int16_t)(i / 2);   // lane 2c + s: position c
    const HVX_Vector iota = *(const HVX_Vector *)iota_a, ninf = Q6_Vh_vsplat_R(0xFC00), zero = Q6_V_vzero();
    ex_t k;
    k.log2e = (int)((uint32_t)hf_bits(1.4426950409f) * 0x10001u), k.c3 = (int)((uint32_t)hf_bits(-0.0390625f) * 0x10001u);
    k.c2 = Q6_Vh_vsplat_R(hf_bits(0.23046875f)), k.c1 = Q6_Vh_vsplat_R(hf_bits(-0.69140625f));
    k.c0 = Q6_Vh_vsplat_R(hf_bits(1.0f)), k.y_max = Q6_Vh_vsplat_R(hf_bits(24.0f));
    const int lg = (d->flags & ATTN_LOG2) != 0;
    k.e_top = Q6_Vh_vsplat_R((15 + E_UP) << 10);
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);   // (hf 1.0: the sums' factor)
    const int acc_lo = g_acc_from == 1;
    const int all_lo = lo_of(d, g->t1), all_hi = hi_of(d, g->t0);
    const int base = a->sb_lo + 32 * g->b0, nb = g->nb;
    // [j0, j1): the croutons inside [the largest first, the smallest last] - every row sees all 32 positions
    int j0 = all_lo <= base ? 0 : (all_lo - base + 31) / 32;
    int j1 = all_hi - 31 < base ? 0 : (all_hi - 31 - base) / 32 + 1;
    j0 = j0 > nb ? nb : j0;
    j1 = j1 > nb ? nb : j1 < j0 ? j0 : j1;
    const size_t CS = 2048u / sizeof(HVX_Vector);   // vectors from a crouton to the next
    for (int i = 0; i < 16; ++i) {
        int ta, ha, tb, hb;
        row_of(a, g, 2 * i, &ta, &ha);
        row_of(a, g, 2 * i + 1, &tb, &hb);
        int la = lo_of(d, ta) - base, ua_ = hi_of(d, ta) - base, lb = lo_of(d, tb) - base, ub_ = hi_of(d, tb) - base;
        la = la < -30000 ? -30000 : la > 30000 ? 30000 : la;
        lb = lb < -30000 ? -30000 : lb > 30000 ? 30000 : lb;
        ua_ = ua_ < -30000 ? -30000 : ua_ > 30000 ? 30000 : ua_;
        ub_ = ub_ < -30000 ? -30000 : ub_ > 30000 ? 30000 : ub_;
        const HVX_Vector vlo = Q6_V_vsplat_R((int)(((uint32_t)(uint16_t)(la - 1)) | ((uint32_t)(uint16_t)(lb - 1) << 16)));
        const HVX_Vector vhi = Q6_V_vsplat_R((int)(((uint32_t)(uint16_t)(ua_ + 1)) | ((uint32_t)(uint16_t)(ub_ + 1) << 16)));
        HVX_Vector *const row = (HVX_Vector *)(S + (size_t)i * 128u);   // crouton j's vector at row[j * CS]
        HVX_Vector m0 = ninf, m1 = ninf, m2 = ninf, m3 = ninf;
        for (int j = 0; j < nb; ++j) {   // the edges, masked
            if (j == j0) j = j1;
            if (j >= nb) break;
            const HVX_Vector pv = Q6_Vh_vadd_VhVh(iota, Q6_Vh_vsplat_R(32 * j));
            const HVX_VectorPred q = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhVh(pv, vlo), Q6_Q_vcmp_gt_VhVh(vhi, pv));
            m0 = Q6_Vhf_vfmax_VhfVhf(m0, Q6_V_vmux_QVV(q, row[(size_t)j * CS], ninf));
        }
        int j = j0;
        for (; j + 4 <= j1; j += 4) {
            m0 = Q6_Vhf_vfmax_VhfVhf(m0, row[(size_t)j * CS]);
            m1 = Q6_Vhf_vfmax_VhfVhf(m1, row[(size_t)(j + 1) * CS]);
            m2 = Q6_Vhf_vfmax_VhfVhf(m2, row[(size_t)(j + 2) * CS]);
            m3 = Q6_Vhf_vfmax_VhfVhf(m3, row[(size_t)(j + 3) * CS]);
        }
        for (; j < j1; ++j) m1 = Q6_Vhf_vfmax_VhfVhf(m1, row[(size_t)j * CS]);
        HVX_Vector m = Q6_Vhf_vfmax_VhfVhf(Q6_Vhf_vfmax_VhfVhf(m0, m1), Q6_Vhf_vfmax_VhfVhf(m2, m3));
        for (int s = 64; s >= 4; s >>= 1) m = Q6_Vhf_vfmax_VhfVhf(m, Q6_V_vror_VR(m, s));   // (parity kept)
        HVX_Vector mt __attribute__((aligned(128))) = m;
        const uint16_t mab = ((const uint16_t *)&mt)[0], mbb = ((const uint16_t *)&mt)[1];
        const int oka = mab != 0xFC00, okb = mbb != 0xFC00;
        const HVX_Vector mv = Q6_V_vsplat_R((int)((uint32_t)(oka ? mab : 0) | ((uint32_t)(okb ? mbb : 0) << 16)));
        const HVX_Vector okm = Q6_V_vsplat_R((int)((oka ? 0xFFFFu : 0u) | (okb ? 0xFFFF0000u : 0u)));
        HVX_Vector sa0 = zero, sb0 = zero, sa1 = zero, sb1 = zero;
        for (int jj = 0; jj < nb; ++jj) {   // the edges
            if (jj == j0) jj = j1;
            if (jj >= nb) break;
            HVX_Vector e = Q6_V_vand_VV(exp_hf(&k, mv, row[(size_t)jj * CS], lg), okm);
            const HVX_Vector pv = Q6_Vh_vadd_VhVh(iota, Q6_Vh_vsplat_R(32 * jj));
            const HVX_VectorPred qm = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhVh(pv, vlo), Q6_Q_vcmp_gt_VhVh(vhi, pv));
            e = Q6_V_vmux_QVV(qm, e, zero);
            row[(size_t)jj * CS] = e;
            HVX_Vector ea, eb;
            sf_of(e, &ea, &eb);
            // (the sums in sf, each step converted back: softmax_h's note on the compiler's qfloat pass)
            sa0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sa0, ea)), sb0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sb0, eb));
        }
        // the croutons no row masks, one a step: two at a time ran slower (the registers spill), without the masks'
        // branch the compiler pipelines the chain (+60% over softmax_h at 1024 x 16K)
        // (no okm: a row without a position here is never read; lg a constant in each loop - a branch inside one
        // stops the compiler pipelining it: 62 against 45 ms)
#define SM2_INTERIOR(LG)                                                                                              \
        for (j = j0; j < j1; ++j) {                                                                                   \
            HVX_Vector *v0 = row + (size_t)j * CS;                                                                    \
            const HVX_Vector e0 = exp_hf(&k, mv, *v0, LG);                                                            \
            *v0 = e0;                                                                                                 \
            HVX_Vector a0, b0;                                                                                        \
            sf_of(e0, &a0, &b0);                                                                                      \
            sa1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sa1, a0));                                                 \
            sb1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sb1, b0));                                                 \
        }
        // the sums in IEEE sf by one instruction a vector (sf += hf x 1: the product exact, each step rounded to sf
        // as the converted sums were) in place of a conversion, two qf32 adds and two conversions back
#define SM2_INTERIOR_ACC(LG)                                                                                          \
        {                                                                                                             \
            HVX_VectorPair w = Q6_W_vcombine_VV(zero, zero);                                                          \
            for (j = j0; j < j1; ++j) {                                                                               \
                HVX_Vector *v0 = row + (size_t)j * CS;                                                                \
                const HVX_Vector e0 = exp_hf(&k, mv, *v0, LG);                                                        \
                *v0 = e0;                                                                                             \
                w = Q6_Wsf_vmpyacc_WsfVhfVhf(w, e0, one);                                                             \
            }                                                                                                         \
            sa1 = acc_lo ? Q6_V_lo_W(w) : Q6_V_hi_W(w);                                                               \
            sb1 = acc_lo ? Q6_V_hi_W(w) : Q6_V_lo_W(w);                                                               \
        }
        if (a->sum_acc) {
            if (lg) SM2_INTERIOR_ACC(1)
            else SM2_INTERIOR_ACC(0)
        } else if (lg) SM2_INTERIOR(1)
        else SM2_INTERIOR(0)
#undef SM2_INTERIOR
#undef SM2_INTERIOR_ACC
        mx[2 * i] = oka ? (float)*(const __fp16 *)&mab : -INFINITY;
        mx[2 * i + 1] = okb ? (float)*(const __fp16 *)&mbb : -INFINITY;
        sm[2 * i] = oka ? f_hsum(f_add(sa0, sa1)) : 0.0f;
        sm[2 * i + 1] = okb ? f_hsum(f_add(sb0, sb1)) : 0.0f;
    }
}

// the tile's rows out: divided by their sums (one superblock), or joined with what the earlier ones left
static void rows_out(at_t *a, int th, const geo_t *g, const float *mx, const float *sm) {
    const attn_desc_t *d = a->d;
    const uint8_t *O = a->os[th];
    for (int i = 0; i < 16; ++i) {
        float *op[2];
        HVX_Vector fo[2], fb[2];
        int ok[2], first[2];
        for (int s = 0; s < 2; ++s) {
            const int rr = 2 * i + s;
            int t, h;
            row_of(a, g, rr, &t, &h);
            ok[s] = t < d->n && mx[rr] > -INFINITY;
            op[s] = ok[s] ? a->o + (size_t)t * d->o_stride + (size_t)h * a->hd : NULL;
            first[s] = 1;
            fo[s] = fb[s] = Q6_V_vzero();
            if (!ok[s]) continue;
            const float up = (float)(1 << (PV_SHIFT + a->e_up));
            if (!a->multi) {
                fb[s] = f_sp(up / sm[rr]);
                continue;
            }
            const size_t R = (size_t)t * d->nh + (size_t)h;
            if (a->m[R] == -INFINITY) {
                a->m[R] = mx[rr];
                a->l[R] = sm[rr];
                fb[s] = f_sp(up);
                continue;
            }
            const float M = a->m[R] > mx[rr] ? a->m[R] : mx[rr];
            const float eo = a->log2 ? exp2f(a->m[R] - M) : expf(a->m[R] - M);
            const float eb = a->log2 ? exp2f(mx[rr] - M) : expf(mx[rr] - M);
            a->l[R] = a->l[R] * eo + sm[rr] * eb;
            a->m[R] = M;
            first[s] = 0;
            fo[s] = f_sp(eo);
            fb[s] = f_sp(eb * up);
        }
        if (!ok[0] && !ok[1]) continue;
        for (int dg = 0; dg < a->dgs; ++dg) {
            HVX_Vector v[2];
            sf_of(*(const HVX_Vector *)(O + (size_t)dg * 2048u + (size_t)i * 128u), &v[0], &v[1]);
            for (int s = 0; s < 2; ++s) {
                if (!ok[s]) continue;
                HVX_Vector *dst = (HVX_Vector *)(op[s] + (size_t)dg * 32);
                *dst = first[s] ? f_mul(v[s], fb[s]) : f_add(f_mul(*dst, fo[s]), f_mul(v[s], fb[s]));
            }
        }
    }
}

// ---- the threads ----------------------------------------------------------------------------------------------------

static inline int take(at_t *a, int st) { return __atomic_fetch_add(&a->ctr[st], 1, __ATOMIC_RELAXED); }

static void bar(at_t *a, int *sense) {
    const int s = *sense ^= 1;
    if (__atomic_add_fetch(&a->arrive, 1, __ATOMIC_ACQ_REL) == a->nthreads) {
        __atomic_store_n(&a->arrive, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&a->sense, s, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&a->sense, __ATOMIC_ACQUIRE) != s) asm volatile("pause(#4)");
    }
}

static inline int wait_for(at_t *a, volatile int *v, int want) {
    while (__atomic_load_n(v, __ATOMIC_ACQUIRE) != want)
        if (__atomic_load_n(&a->failed, __ATOMIC_ACQUIRE)) return 0;
    return 1;
}

// ATTN_DEBUG_SMCHECK: the IEEE softmax and the hf one in use (softmax_h2, or softmax_h with ATTN_SOFTMAX_V1) on
// copies of the tile's scores, the first difference kept (kind 1 the maximum,
// 2 the sum, 3 a weight; the row pair, the crouton, the lane) and returned in usec
static volatile int64_t g_sm_dbg;
static void sm_check(const at_t *a, int th, const geo_t *g) {
    const size_t bytes = (size_t)g->nb * 2048u;
    uint8_t *x = (uint8_t *)memalign(128, bytes), *y = (uint8_t *)memalign(128, bytes);
    if (!x || !y) {
        free(x), free(y);
        return;
    }
    memcpy(x, a->ss[th], bytes);
    memcpy(y, a->ss[th], bytes);
    float m0[32], s0[32], m1[32], s1[32];
    int64_t r = 0;
    softmax(a, x, g, m0, s0);
    float down = 1.0f;   // (softmax_h2's weights and sums 2^E_UP larger)
    if (a->sm_v1) {
        softmax_h(a, y, g, m1, s1);
    } else {
        softmax_h2(a, y, g, m1, s1);
        down = 1.0f / (float)(1 << E_UP);
        for (int k = 0; k < 32; ++k) s1[k] *= down;
    }
    for (int k = 0; k < 32 && !r; ++k) {
        const float dm = m0[k] - m1[k], ds = s0[k] - s1[k];
        if (!(m0[k] == m1[k] || (dm < 1e-2f && dm > -1e-2f))) r = 1000000000000LL + 1000000000LL * 1 + 1000000LL * k + (int64_t)(m1[k] * 100.0f) % 1000000;
        else if (!(ds < 2e-3f * s0[k] + 1e-6f && ds > -2e-3f * s0[k] - 1e-6f)) r = 1000000000000LL + 1000000000LL * 2 + 1000000LL * k + (int64_t)(s0[k] * 10.0f) % 1000 * 1000 + (int64_t)(s1[k] * 10.0f) % 1000;
    }
    const __fp16 *wx = (const __fp16 *)x, *wy = (const __fp16 *)y;
    if (r / 1000000000LL % 1000 == 2) {   // the new weights of that row summed here: its parity's lanes, and all
        const int k = (int)(r / 1000000LL % 1000), vi = k / 2;
        float se = 0, sa = 0;
        for (int j = 0; j < g->nb; ++j)
            for (int l = 0; l < 64; ++l) {
                const float w = (float)wy[(size_t)j * 1024u + (size_t)vi * 64u + (size_t)l] * down;
                sa += w;
                if ((l & 1) == (k & 1)) se += w;
            }
        r = 1000000000000LL + 4000000000LL + 1000000LL * k + (int64_t)(se * 10.0f) % 1000 * 1000 + (int64_t)(sa * 10.0f) % 1000;
    }
    for (size_t i = 0; i < bytes / 2 && !r; ++i) {
        const float u = (float)wx[i], v = (float)wy[i] * down, dd = u - v;
        if (!(dd < 3e-3f && dd > -3e-3f) || u != u || v != v)
            r = 1000000000000LL + 1000000000LL * 3 + 1000000LL * (int64_t)((i % 1024) / 64) + 1000 * (int64_t)(i / 1024) + (int64_t)(i % 64);
    }
    if (r) {
        int64_t z = 0;
        __atomic_compare_exchange_n(&g_sm_dbg, &z, r, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
    free(x), free(y);
}

// a helper's row tiles of this superblock: its query tile, then (HMX: the scores) its softmax, then (HMX: P V) its
// rows out
static void helper_tiles(at_t *a, int th) {
    float mx[32], sm[32];
    for (int r = th; r < a->tiles; r += a->nh_) {
        const geo_t g = geo(a, r);
        if (!g.ok) continue;
        build_q(a, th, &g);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);   // (the tile's stores in VTCM before HMX reads it)
        __atomic_store_n(&a->q_ready[th], r, __ATOMIC_RELEASE);
        if (!wait_for(a, &a->s_done[th], r)) break;
        if (a->sm_check && !__atomic_load_n(&g_sm_dbg, __ATOMIC_ACQUIRE)) sm_check(a, th, &g);
        if (a->sm_ieee) softmax(a, a->ss[th], &g, mx, sm);
        else if (a->sm_v1) softmax_h(a, a->ss[th], &g, mx, sm);
        else softmax_h2(a, a->ss[th], &g, mx, sm);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        __atomic_store_n(&a->p_ready[th], r, __ATOMIC_RELEASE);
        if (!wait_for(a, &a->o_done[th], r)) break;
        rows_out(a, th, &g, mx, sm);
    }
    __atomic_store_n(&a->h_done[th], 1, __ATOMIC_RELEASE);
}

// this thread: whatever a helper handed over, the scores or P V, until every helper is through
static void hmx_tiles(at_t *a) {
    for (;;) {
        int all = 1;
        for (int h = 0; h < a->nh_; ++h) {
            const int done = __atomic_load_n(&a->h_done[h], __ATOMIC_ACQUIRE);
            int r = __atomic_load_n(&a->q_ready[h], __ATOMIC_ACQUIRE);
            if (r > a->s_done[h]) {
                const geo_t g = geo(a, r);
                if (!a->no_hmx) {
                    Q6_bias_mxmem2_A(a->bias_s);
                    for (int j = 0; j < g.nb; ++j)
                        tile(a->qs[h], a->K + (size_t)(g.b0 + j) * (size_t)a->dgs * 2048u, a->ss[h] + (size_t)j * 2048u, a->dgs);
                }
                __atomic_thread_fence(__ATOMIC_SEQ_CST);   // (HMX's results in VTCM before the helper reads them)
                __atomic_store_n(&a->s_done[h], r, __ATOMIC_RELEASE);
            }
            r = __atomic_load_n(&a->p_ready[h], __ATOMIC_ACQUIRE);
            if (r > a->o_done[h]) {
                const geo_t g = geo(a, r);
                if (!a->no_hmx) {
                    Q6_bias_mxmem2_A(a->bias_pv);
                    for (int dg = 0; dg < a->dgs; ++dg)
                        tile(a->ss[h], a->V + ((size_t)dg * a->npb + g.b0) * 2048u, a->os[h] + (size_t)dg * 2048u, g.nb);
                }
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                __atomic_store_n(&a->o_done[h], r, __ATOMIC_RELEASE);
            }
            if (!done) all = 0;
        }
        if (all || __atomic_load_n(&a->failed, __ATOMIC_ACQUIRE)) return;
    }
}

// The call on every thread (th < nh_: a helper; th == nh_: this one), every one walking the same KV heads and
// superblocks, the phases apart by barriers
static void program(at_t *a, int th) {
    const attn_desc_t *d = a->d;
    int sense = 0, st = 0;
    const int r_lo = lo_of(d, 0) / 32 * 32, r_hi = (end_of(d) + 31) / 32 * 32;
    const int n_sb = (r_hi - r_lo + a->sb_max - 1) / a->sb_max;
    for (int kh = 0; kh < d->nkv; ++kh)
        for (int sbi = 0; sbi < n_sb; ++sbi) {
            if (th == a->nh_) {   // (no helper reads these before the barrier)
                a->kh = kh;
                a->sb_lo = r_lo + sbi * a->sb_max;
                a->sb_hi = a->sb_lo + a->sb_max < r_hi ? a->sb_lo + a->sb_max : r_hi;
                a->npb = (a->sb_hi - a->sb_lo) / 32;
                for (int h = 0; h < a->nh_; ++h) a->q_ready[h] = a->s_done[h] = a->p_ready[h] = a->o_done[h] = -1, a->h_done[h] = 0;
            }
            bar(a, &sense);
            for (int i = take(a, st); i < 2 * a->npb; i = take(a, st)) {
                if (i < a->npb) build_k(a, th, i);
                else build_v(a, th, i - a->npb);
            }
            ++st;
            bar(a, &sense);
            if (th < a->nh_) helper_tiles(a, th);
            else hmx_tiles(a);
            bar(a, &sense);
        }
    if (!a->multi) return;
    const int rows = d->n * d->nh;
    for (int r = take(a, st); r < rows; r = take(a, st)) {
        const int t = r / d->nh, h = r % d->nh;
        HVX_Vector *v = (HVX_Vector *)(a->o + (size_t)t * d->o_stride + (size_t)h * a->hd);
        const HVX_Vector f = f_sp(a->l[r] > 0.0f ? 1.0f / a->l[r] : 0.0f);
        for (int dg = 0; dg < a->dgs; ++dg) v[dg] = f_mul(v[dg], f);
    }
    bar(a, &sense);
}

static void helper(void *arg) {
    at_arg_t *x = (at_arg_t *)arg;
    at_t *a = x->a;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
        __atomic_store_n(&a->failed, 1, __ATOMIC_RELEASE);
        qurt_thread_exit(0);
    }
    __atomic_add_fetch(&a->started, 1, __ATOMIC_ACQ_REL);
    while (__atomic_load_n(&a->started, __ATOMIC_ACQUIRE) < a->nh_ + 1)   // until this thread says go (or not)
        if (__atomic_load_n(&a->failed, __ATOMIC_ACQUIRE)) goto out;
    program(a, x->th);
out:
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

static void *map_fd(int fd) {
    void *v = NULL;
    uint64 pa = 0;
    return HAP_mmap_get(fd, &v, &pa) ? NULL : v;
}

static int attn_check(const attn_desc_t *d) {
    const int group = d->nkv > 0 ? d->nh / d->nkv : 0;
    if ((d->flags & ATTN_LOG2) && (d->flags & (ATTN_SOFTMAX_IEEE | ATTN_SOFTMAX_V1))) return AEE_EBADPARM;   // (softmax_h2's)
    if ((d->flags & ATTN_CAUSAL) && (!(d->flags & ATTN_ENCODER) || keys_of(d) < d->n)) return AEE_EBADPARM;
    if (d->flags & ATTN_ENCODER)   // (the rows' state lives in VTCM: attn_prepare checks it fits)
        return d->version != ATTN_VERSION || d->n < 1 || d->apos != 0 || d->nkv < 1 || group < 1 ||
                       d->nh != group * d->nkv || 32 % group || d->hd < 32 || d->hd > 512 || d->hd % 32 ||
                       d->window != 0 || d->nk < 0 || keys_of(d) > 30000 || d->npre < 0 ||
                       d->npre > (keys_of(d) + 31) / 32 || d->n_chunks != (d->npre ? 4 : 2) ||
                       d->q_stride < d->nh * d->hd || d->o_stride < d->nh * d->hd || d->q_stride % 32 ||
                       d->o_stride % 32
                   ? AEE_EBADPARM
                   : 0;
    if (d->version != ATTN_VERSION || d->n < 1 || d->apos < 0 || d->nkv < 1 || group < 1 || d->nh != group * d->nkv ||
        32 % group || d->hd < 128 || d->hd > 512 || d->hd % 128 || d->window < 0 || d->slot < 0 ||
        d->slot >= d->n_attn || d->kv_chunk < 32 || d->kv_heads < d->nkv || d->kv_row < d->nkv * d->hd ||
        d->n_chunks < 1 || d->n_chunks > ATTN_MAX_CHUNKS || (d->apos + d->n + d->kv_chunk - 1) / d->kv_chunk > d->n_chunks ||
        d->q_stride < d->nh * d->hd || d->o_stride < d->nh * d->hd || d->q_stride % 32 || d->o_stride % 32 ||
        d->n * d->nh > ATTN_MAX_ROWS)
        return AEE_EBADPARM;
    return 0;
}

// The attention's state for a call (geode_npu_attn's, or a layer's in gdec_prefill): the VTCM carved for `helpers`
// HVX threads beside this one, the rows' state cleared, the KV cache's chunks as mapped. 0 or the error
static int attn_prepare(at_t *a, const attn_desc_t *d, const float *q, float *o, uint8_t *const *kv, uint8_t *vtcm,
                        unsigned vsize, int helpers) {
    memset(a, 0, sizeof *a);
    a->d = d;
    a->q = q;
    a->o = o;
    a->group = d->nh / d->nkv;
    a->tpt = 32 / a->group;
    a->tiles = (d->n + a->tpt - 1) / a->tpt;
    a->hd = d->hd;
    a->dgs = d->hd / 32;
    a->no_hmx = (d->flags & ATTN_DEBUG_NO_HMX) != 0;
    a->sm_ieee = (d->flags & ATTN_SOFTMAX_IEEE) != 0;
    a->sm_check = (d->flags & ATTN_DEBUG_SMCHECK) != 0;
    a->sm_v1 = (d->flags & ATTN_SOFTMAX_V1) != 0;
    a->e_up = a->sm_ieee || a->sm_v1 ? 0 : E_UP;
    a->log2 = (d->flags & ATTN_LOG2) != 0;
    a->sum_acc = !(d->flags & ATTN_SUM_CVT) && g_acc_from > 0;
    a->nh_ = helpers > A_HELPERS_MAX ? A_HELPERS_MAX : helpers;
    a->nthreads = a->nh_ + 1;
    // VTCM: the biases | the rows' max and sum | each thread's scratch | each helper's query, score and output tiles
    // | the superblock's keys and values
    const size_t rows_bytes = ((size_t)d->n * d->nh * 8u + 2047u) / 2048u * 2048u;
    const size_t scr = ((size_t)32 * d->hd * 2u + 64u * 128u + 2047u) / 2048u * 2048u, qo = (size_t)d->hd * 64u;
    const size_t fixed = 4096u + rows_bytes + (size_t)a->nthreads * scr + (size_t)a->nh_ * 2u * qo;
    if (fixed >= vsize) return AEE_ENOMEMORY;
    const int r_lo = lo_of(d, 0) / 32 * 32, r_hi = (end_of(d) + 31) / 32 * 32;
    int sb = (int)((vsize - fixed) / ((size_t)4u * d->hd + 64u * (size_t)a->nh_)) / 32 * 32;
    if (ATTN_SB_OF(d->flags) && sb > (int)ATTN_SB_OF(d->flags) * 32) sb = (int)ATTN_SB_OF(d->flags) * 32;
    if (sb > r_hi - r_lo) sb = r_hi - r_lo;
    if (sb < 32) return AEE_ENOMEMORY;
    // program()'s stages (a counter each): one a KV head and superblock, one more for the joins' division
    if ((int64_t)d->nkv * ((r_hi - r_lo + sb - 1) / sb) + 1 >= A_STAGES) return AEE_EBADPARM;
    a->sb_max = sb;
    a->multi = (r_hi - r_lo + sb - 1) / sb > 1;
    uint8_t *at = vtcm;
    a->bias_s = at, at += 2048;
    a->bias_pv = at, at += 2048;
    a->m = (float *)at, a->l = a->m + (size_t)d->n * d->nh, at += rows_bytes;
    for (int i = 0; i < a->nthreads; ++i) a->scr[i] = at, at += scr;
    for (int i = 0; i < a->nh_; ++i) {
        a->qs[i] = at, at += qo;
        a->os[i] = at, at += qo;
    }
    const size_t s_bytes = ((size_t)64u * sb + 2047u) / 2048u * 2048u;
    for (int i = 0; i < a->nh_; ++i) a->ss[i] = at, at += s_bytes;
    a->K = at, at += (size_t)sb * d->hd * 2u;
    a->V = at, at += (size_t)sb * d->hd * 2u;
    if ((size_t)(at - vtcm) > vsize) return AEE_ENOMEMORY;
    memset(a->bias_s, 0, 4096);
    for (int ch = 0; ch < 32; ++ch) {   // output bias 0, scale 1 / 2^-PV_SHIFT (npu_imp.c's convention)
        ((uint32_t *)a->bias_s)[ch] = 15u << 10;
        const int ps = PV_SHIFT + a->e_up;   // (past 14: a subnormal scale, as npu_imp.c's out_shift)
        ((uint32_t *)a->bias_pv)[ch] = ps <= 14 ? (uint32_t)(15 - ps) << 10 : 1u << (24 - ps);
    }
    if (a->multi)
        for (size_t r = 0; r < (size_t)d->n * d->nh; ++r) a->m[r] = -INFINITY, a->l[r] = 0.0f;
    for (int i = 0; i < d->n_chunks; ++i) a->kv[i] = kv[i];
    return 0;
}

static int helper_prio(void) {
    int prio = qurt_thread_get_priority(qurt_thread_get_id());   // the RPC thread's; a created one must be below 255
    return prio <= 0 || prio >= 255 ? 127 : prio;
}

int geode_npu_attn(remote_handle64 h, const unsigned char *desc, int desc_len, const unsigned char *q, int q_len,
                   unsigned char *o, int o_len, int64_t *usec) {
    *usec = 0;
    if (desc_len != (int)sizeof(attn_desc_t)) return AEE_EBADPARM;
    const attn_desc_t *d = (const attn_desc_t *)desc;
    if (attn_check(d) || (int64_t)q_len < (int64_t)d->n * d->q_stride * 4 ||
        (int64_t)o_len < (int64_t)d->n * d->o_stride * 4 || ((uintptr_t)q & 127) || ((uintptr_t)o & 127))
        return AEE_EBADPARM;
    if (g_orders_ok < 0) a_checks();
    if (!g_orders_ok) return AEE_EUNSUPPORTED;
    uint8_t *vtcm = NULL;
    unsigned vsize = 0;
    int e = geode_npu_vtcm(h, &vtcm, &vsize);
    if (e) return e;
    const uint64_t t0 = HAP_perf_get_time_us();
    // the KV cache's chunks
    uint8_t *kv[ATTN_MAX_CHUNKS];
    int mapped = 0;
    for (; mapped < d->n_chunks; ++mapped)
        if (!(kv[mapped] = (uint8_t *)map_fd(d->kv_fd[mapped]))) break;
    if (mapped < d->n_chunks) {
        for (int i = 0; i < mapped; ++i) HAP_mmap_put(d->kv_fd[i]);
        return AEE_EBADPARM;
    }
    static at_t a;   // (large: not on the FastRPC thread's stack)
    e = attn_prepare(&a, d, (const float *)q, (float *)o, kv, vtcm, vsize,
                     ATTN_HELPERS(d->flags) ? ATTN_HELPERS(d->flags) : A_HELPERS);
    if (e) {
        for (int i = 0; i < mapped; ++i) HAP_mmap_put(d->kv_fd[i]);
        return e;
    }
    // what the CPU wrote (the chunk's keys and values) seen afresh
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    const int prio = helper_prio();
    at_arg_t args[A_HELPERS_MAX];
    qurt_thread_t tid[A_HELPERS_MAX];
    int made = 0;
    for (int i = 0; i < a.nh_; ++i) {
        qurt_thread_attr_t ta;
        qurt_thread_attr_init(&ta);
        if (!g_a_stacks[i] && !(g_a_stacks[i] = memalign(128, A_STACK))) break;
        qurt_thread_attr_set_name(&ta, (char *)"geode_attn");
        qurt_thread_attr_set_stack_addr(&ta, g_a_stacks[i]);
        qurt_thread_attr_set_stack_size(&ta, A_STACK);
        qurt_thread_attr_set_priority(&ta, (unsigned short)prio);
        args[i].a = &a;
        args[i].th = i;
        if (qurt_thread_create(&tid[i], &ta, helper, &args[i]) != QURT_EOK) break;
        ++made;
    }
    if (made < a.nh_) __atomic_store_n(&a.failed, 1, __ATOMIC_RELEASE);
    while (!a.failed && __atomic_load_n(&a.started, __ATOMIC_ACQUIRE) < a.nh_) {
    }
    if (!a.failed) {
        __atomic_add_fetch(&a.started, 1, __ATOMIC_ACQ_REL);
        program(&a, a.nh_);
    }
    for (int i = 0; i < made; ++i) {
        int status;
        qurt_thread_join(tid[i], &status);
    }
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);
    for (int i = 0; i < mapped; ++i) HAP_mmap_put(d->kv_fd[i]);
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    if (a.sm_check) *usec = g_sm_dbg, g_sm_dbg = 0;
    return a.failed ? AEE_ENOTALLOWED : 0;
}

// =====================================================================================================================
// ---- Gemma 4's layers on the DSP (npu/gdec_cfg.h): a prompt's chunk through every layer ----------------------------
//
// The rows stay in HMX's order between the steps: a task is a row pair (rows 2p, 2p + 1 of the chunk: vector p % 16 of
// every crouton of row tile p / 16, its even lanes row 2p's 32 channels, its odd lanes row 2p + 1's), so a product's
// result croutons are read and the next product's input tiles written without any reordering; the residual rows x
// (floats) are the CPU's. The FastRPC thread multiplies (geode_npu_gemm_ptr, geode_npu_glu_ptr: their own HVX
// threads) while the pool waits without HVX; the pool's GDEC_THREADS workers take the steps between, the attention
// with this thread on HMX.

int geode_npu_gemm_ptr(remote_handle64 h, const uint8_t *a, const uint8_t *w, uint8_t *o, int m, int k, int n,
                       int out_shift);   // npu_imp.c
int geode_npu_glu_ptr(remote_handle64 h, const uint8_t *a, const uint8_t *gw, const uint8_t *uw, uint8_t *o, int m,
                      int k, int n, int g_shift, int u_shift, int part_blk, int act);   // npu_imp.c

#define GD_STACK 32768
#define GD_MATS 12   // a layer's matrices: q k v o gate up ple_gate ple_proj down[4]

enum { GP_IN, GP_QKV, GP_ATTN, GP_CORE, GP_POST, GP_DOWN, GP_PLE, GP_PLE2 };

typedef struct {
    gdec_cfg_t c;
    gdec_layout_t L;
    uint8_t *P, *IO;
    uint8_t *PQ;   // P in qf32 (the DSP's own copy)
    int p_fd, io_fd;
    uint8_t *kv[ATTN_MAX_CHUNKS];
    int kv_fd[ATTN_MAX_CHUNKS];
    const uint8_t *w[GDEC_MAX_LAYERS][GD_MATS];   // the matrices' hmx4 blocks, mapped
    const t8_t *t8[GDEC_MAX_LAYERS][GD_MATS];     // or their tile8 matrices (one copy with the CPU's: wsrc.h)
    int mfd[GDEC_MAX_LAYERS * GD_MATS], n_mfd;    // the fds mapped (each once)
    // the call
    int n, m, apos;
    int task, pf, main_joins;   // row pairs a task, the L2 prefetch (GDEC_PF_OF), the FastRPC thread in the pool
    int old_mask;               // the phases (1 << GP_*) on the steps as they were (GDEC_OLD_OF)
    const glayer_t *ly;
    int layer, phase;
    attn_desc_t ad;
    at_t at;
    uint64_t *prof;
    // the pool
    qurt_thread_t tid[GDEC_THREADS];
    void *stack[GDEC_THREADS];
    int nthreads;
    volatile int go, quit, running, failed;
    int ctr;
} gd_t;

static gd_t *g_gd;

static inline HVX_Vector *cvec(uint8_t *buf, int ncb, int p, int cb) {   // row pair p's vector of crouton cb
    return (HVX_Vector *)(buf + (((size_t)(p >> 4) * (size_t)ncb + (size_t)cb) << 11) + ((size_t)(p & 15) << 7));
}

// ---- qf32: HVX's own floats. Each IEEE sf operation above compiles to its qf32 one and a conversion back - twice
// the chain, one instruction a packet (measured: the steps between the products 0.74 s of a 2048-token E4B prompt on
// six threads, a sixth no faster than five). The steps below stay in qf32 between their loads and stores (qf32 bits
// kept in memory as they are: tmp's rows), the norms' weights in a qf32 copy of P (PQ), two or four sums in flight
static inline HVX_Vector q_sf(HVX_Vector s) { return Q6_Vqf32_vadd_VsfVsf(s, Q6_V_vzero()); }
static inline HVX_Vector sf_q(HVX_Vector q) { return Q6_Vsf_equals_Vqf32(q); }
static inline HVX_Vector q_add(HVX_Vector a, HVX_Vector b) { return Q6_Vqf32_vadd_Vqf32Vqf32(a, b); }
static inline HVX_Vector q_adds(HVX_Vector q, HVX_Vector s) { return Q6_Vqf32_vadd_Vqf32Vsf(q, s); }
static inline HVX_Vector q_sub(HVX_Vector a, HVX_Vector b) { return Q6_Vqf32_vsub_Vqf32Vqf32(a, b); }
static inline HVX_Vector q_mul(HVX_Vector a, HVX_Vector b) { return Q6_Vqf32_vmpy_Vqf32Vqf32(a, b); }
static inline HVX_Vector q_muls(HVX_Vector a, HVX_Vector b) { return Q6_Vqf32_vmpy_VsfVsf(a, b); }
// a crouton vector (two rows' 32 channels in hf) as the rows' qf32 (times 1.0: what Q6_Wsf_vcvt_Vhf compiles to,
// before its conversions); two rows' qf32 back as one. Their lane orders: a_checks (Q_FROM_LO_EVEN, Q_TO_ODD_HI)
static inline void q_of_hf(HVX_Vector h, HVX_Vector *even, HVX_Vector *odd) {
    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(h, Q6_Vh_vsplat_R(0x3C00));
    *even = Q6_V_lo_W(w);
    *odd = Q6_V_hi_W(w);
}
static inline HVX_Vector hf_of_q(HVX_Vector even, HVX_Vector odd) {
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(odd, even));
}
static inline float q_hsum(HVX_Vector q) {
    for (int s = 64; s >= 4; s >>= 1) q = q_add(q, Q6_V_vror_VR(q, s));
    HVX_Vector t __attribute__((aligned(128))) = sf_q(q);
    return ((const float *)&t)[0];
}

static inline HVX_Vector sumsq_q(const HVX_Vector *v, int nv) {   // v: sf; the lanes' partial sums (qf32)
    HVX_Vector s0 = Q6_V_vzero(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 4 <= nv; i += 4) {
        s0 = q_add(s0, q_muls(v[i], v[i]));
        s1 = q_add(s1, q_muls(v[i + 1], v[i + 1]));
        s2 = q_add(s2, q_muls(v[i + 2], v[i + 2]));
        s3 = q_add(s3, q_muls(v[i + 3], v[i + 3]));
    }
    for (; i < nv; ++i) s0 = q_add(s0, q_muls(v[i], v[i]));
    return q_add(q_add(s0, s1), q_add(s2, s3));
}
static inline float sumsq(const HVX_Vector *v, int nv) { return q_hsum(sumsq_q(v, nv)); }   // v: sf

// a qf32 vector's lanes summed, the total in every lane (sf): q_hsum's rotations, without the way out to a scalar
static inline HVX_Vector q_allsum(HVX_Vector q) {
    for (int s = 64; s >= 4; s >>= 1) q = q_add(q, Q6_V_vror_VR(q, s));
    return sf_q(q);
}

// 1 / sqrt(m) in every lane (m: sf, positive): the bit trick's start (~3.4%) and three Newton steps, each result
// back in sf at once (npu-problems 2.1) - fp32's own to an ulp or two
static inline HVX_Vector rsqrt_sf(HVX_Vector m) {
    const HVX_Vector three_half = f_sp(1.5f), hm = sf_q(q_muls(m, f_sp(0.5f)));
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5F3759DF), Q6_Vw_vasr_VwR(m, 1));
    for (int i = 0; i < 3; ++i) {
        const HVX_Vector t = sf_q(q_muls(sf_q(q_muls(y, y)), hm));
        y = sf_q(q_muls(y, sf_q(Q6_Vqf32_vsub_VsfVsf(three_half, t))));
    }
    return y;
}

// x (n floats, 32 k) normed (RMSNorm, times wq: the weights in qf32; NULL: none) in place
static inline void norm_v(float *x, int n, const HVX_Vector *wq, float eps) {
    HVX_Vector *v = (HVX_Vector *)x;
    const HVX_Vector f = f_sp(1.0f / sqrtf(sumsq(v, n / 32) / (float)n + eps));
    if (wq)
        for (int i = 0; i < n / 32; ++i) v[i] = sf_q(q_mul(q_muls(v[i], f), wq[i]));
    else
        for (int i = 0; i < n / 32; ++i) v[i] = sf_q(q_muls(v[i], f));
}

// n heads of hd floats each (one after another) RMS-normed in place, times wq (the weights in qf32): norm_v's
// arithmetic without its way through memory to a scalar square root and back for every head (~300 cycles a head of
// latency, 0.07 s of a YuE2 velocity): the sums of squares totalled across the lanes and their reciprocal square
// roots on HVX, two heads at a time (two chains interleaved)
// (every qf32 result back in sf at once: two heads' chains interleaved keep many vectors live, and a qf32 value that
// spills comes back corrupted - npu-problems 2.1; a first version kept qf32 sums and moved a velocity's error 15%)
static inline HVX_Vector sf_add(HVX_Vector a, HVX_Vector b) { return sf_q(Q6_Vqf32_vadd_VsfVsf(a, b)); }
static inline HVX_Vector sf_mul(HVX_Vector a, HVX_Vector b) { return sf_q(q_muls(a, b)); }
static inline HVX_Vector head_rms(const HVX_Vector *v, int nv, HVX_Vector inv_n, HVX_Vector ve) {
    HVX_Vector s = sf_mul(v[0], v[0]);
    for (int i = 1; i < nv; ++i) s = sf_add(s, sf_mul(v[i], v[i]));
    for (int r = 64; r >= 4; r >>= 1) s = sf_add(s, Q6_V_vror_VR(s, r));   // every lane the total
    return rsqrt_sf(sf_add(sf_mul(s, inv_n), ve));
}
static void norm_heads(float *x, int n, int hd, const HVX_Vector *wq, float eps) {
    const int nv = hd / 32;
    const HVX_Vector inv_n = f_sp(1.0f / (float)hd), ve = f_sp(eps);
    int h = 0;
    for (; h + 2 <= n; h += 2) {
        HVX_Vector *a = (HVX_Vector *)(x + (size_t)h * hd), *b = a + nv;
        const HVX_Vector fa = head_rms(a, nv, inv_n, ve), fb = head_rms(b, nv, inv_n, ve);
        for (int i = 0; i < nv; ++i) {
            a[i] = sf_q(q_mul(q_muls(a[i], fa), wq[i]));
            b[i] = sf_q(q_mul(q_muls(b[i], fb), wq[i]));
        }
    }
    for (; h < n; ++h) {
        HVX_Vector *a = (HVX_Vector *)(x + (size_t)h * hd);
        const HVX_Vector fa = head_rms(a, nv, inv_n, ve);
        for (int i = 0; i < nv; ++i) a[i] = sf_q(q_mul(q_muls(a[i], fa), wq[i]));
    }
}

// rotate-half RoPE on the first `pairs` pairs (i, i + hd / 2) of a head
static inline void rope_v(float *x, int hd, const float *cs, const float *sn, int pairs) {
    HVX_Vector *a = (HVX_Vector *)x, *b = (HVX_Vector *)(x + hd / 2);
    const HVX_Vector *c = (const HVX_Vector *)cs, *s = (const HVX_Vector *)sn;
    for (int i = 0; i < pairs / 32; ++i) {
        const HVX_Vector x1 = a[i], x2 = b[i];
        a[i] = sf_q(q_sub(q_muls(x1, c[i]), q_muls(x2, s[i])));
        b[i] = sf_q(q_add(q_muls(x2, c[i]), q_muls(x1, s[i])));
    }
}

// a head's hd floats as int8 codes (amax / 127 the scale, rounded to nearest: the engine's quant_row) into dst
static void quant_head(const float *x, int hd, int8_t *dst, float *scale) {
    const HVX_Vector *v = (const HVX_Vector *)x;
    HVX_Vector m = Q6_V_vzero();
    for (int i = 0; i < hd / 32; ++i) m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vand_VV(v[i], Q6_V_vsplat_R(0x7FFFFFFF)));
    const float amax = f_hmax(m);
    *scale = amax / 127.0f;
    const HVX_Vector inv = f_sp(amax > 0.0f ? 127.0f / amax : 0.0f);
    const HVX_Vector magic = f_sp(12582912.0f), mb = Q6_V_vsplat_R(0x4B400000);
    const HVX_Vector hi = Q6_V_vsplat_R(127), lo = Q6_V_vsplat_R(-127);
    for (int i = 0; i < hd / 128; ++i) {
        HVX_Vector w[4];
        for (int j = 0; j < 4; ++j) {
            const HVX_Vector r = Q6_Vw_vsub_VwVw(f_add(f_mul(v[4 * i + j], inv), magic), mb);
            w[j] = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(r, hi), lo);
        }
        const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(w[1], w[0]), h23 = Q6_Vh_vpack_VwVw_sat(w[3], w[2]);
        ((HVX_Vector *)dst)[i] = Q6_Vb_vpack_VhVh_sat(h23, h01);
    }
}

static inline float *xrow(gd_t *d, int r) { return (float *)(d->IO + d->c.io_x) + (size_t)r * d->c.hidden; }
static inline const HVX_Vector *pq(const gd_t *d, uint32_t at) { return (const HVX_Vector *)(d->PQ + at); }
static inline const float *pw(const gd_t *d, uint32_t at) { return (const float *)(d->P + at); }
static inline float *tmp(gd_t *d, int th, int s) {
    return (float *)(d->IO + d->L.TMP) + ((size_t)th * 2u + (size_t)s) * d->L.tmpw;
}

// a result's croutons [c0, c0 + nv) of pair p as two rows of floats
static void rows_of(uint8_t *buf, int ncb, int p, int c0, int nv, float *r0, float *r1) {
    HVX_Vector *a = (HVX_Vector *)r0, *b = (HVX_Vector *)r1;
    for (int i = 0; i < nv; ++i) {
        HVX_Vector e, o;
        q_of_hf(*cvec(buf, ncb, p, c0 + i), &e, &o);
        a[i] = sf_q(e);
        b[i] = sf_q(o);
    }
}

// the residual rows of pair p normed (wq) into the input tiles of a product of ncb croutons (past the rows: zeros);
// ok0 / ok1: rows 2p / 2p + 1 inside the chunk (constants where inlined)
static inline __attribute__((always_inline)) void tiles_k(gd_t *d, int p, const HVX_Vector *wq, int ncb, int ok0,
                                                          int ok1) {
    const int H = d->c.hidden, nv = H / 32, nc = nv < ncb ? nv : ncb;
    const HVX_Vector *x0 = (const HVX_Vector *)xrow(d, 2 * p), *x1 = (const HVX_Vector *)xrow(d, 2 * p + 1);
    const HVX_Vector g0 = f_sp(ok0 ? 1.0f / sqrtf(sumsq(x0, nv) / (float)H + d->c.eps) : 0.0f);
    const HVX_Vector g1 = f_sp(ok1 ? 1.0f / sqrtf(sumsq(x1, nv) / (float)H + d->c.eps) : 0.0f);
    const HVX_Vector z = Q6_V_vzero();
    uint8_t *A = d->IO + d->L.A;
    for (int cb = 0; cb < nc; ++cb) {
        const HVX_Vector y0 = ok0 ? q_mul(q_muls(x0[cb], g0), wq[cb]) : z;
        const HVX_Vector y1 = ok1 ? q_mul(q_muls(x1[cb], g1), wq[cb]) : z;
        *cvec(A, ncb, p, cb) = hf_of_q(y0, y1);
    }
    for (int cb = nc; cb < ncb; ++cb) *cvec(A, ncb, p, cb) = z;
}

static void gp_in(gd_t *d, int p, int th) {
    (void)th;
    const HVX_Vector *wq = pq(d, d->ly->in_norm);
    const int ncb = d->ly->q.k / 32;
    if (2 * p + 1 < d->n) tiles_k(d, p, wq, ncb, 1, 1);
    else tiles_k(d, p, wq, ncb, 2 * p < d->n, 0);
}

// the queries normed and turned into QR; the keys normed and turned, the values normed, into the KV cache as int8
static void gp_qkv(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const gdec_cfg_t *c = &d->c;
    const int hd = y->hd, qn = y->q.n, nq = qn / 32, pairs = c->pairs[y->rope];
    float *QR = (float *)(d->IO + d->L.QR);
    float *q0 = QR + (size_t)(2 * p) * qn, *q1 = q0 + qn;
    rows_of(d->IO + d->L.OQ, nq, p, 0, nq, q0, q1);
    for (int s = 0; s < 2; ++s) {
        const int r = 2 * p + s;
        if (r >= d->n) continue;
        float *q = s ? q1 : q0;
        const float *rt = (const float *)(d->IO + c->io_rope[y->rope]) + (size_t)r * 2u * pairs;
        for (int h = 0; h < c->nh; ++h) {
            norm_v(q + (size_t)h * hd, hd, pq(d, y->q_norm), c->eps);
            rope_v(q + (size_t)h * hd, hd, rt, rt + pairs, pairs);
        }
    }
    if (!y->own_kv) return;
    const int nk = y->k.n / 32;
    for (int value = 0; value < 2; ++value) {
        float *k0 = tmp(d, th, 0), *k1 = tmp(d, th, 1);
        rows_of(d->IO + (value ? d->L.OV : d->L.OK), nk, p, 0, nk, k0, k1);
        for (int s = 0; s < 2; ++s) {
            const int r = 2 * p + s;
            if (r >= d->n) continue;
            float *k = s ? k1 : k0;
            const int pos = d->apos + r;
            const float *rt = (const float *)(d->IO + c->io_rope[y->rope]) + (size_t)r * 2u * pairs;
            uint8_t *base = d->kv[pos / c->kv_chunk];
            const size_t at = (size_t)(value * c->n_attn + y->slot) * (size_t)c->kv_chunk + (size_t)(pos % c->kv_chunk);
            int8_t *row = (int8_t *)(base + at * (size_t)c->kv_row);
            float *sc = (float *)(base + 2u * (size_t)c->n_attn * (size_t)c->kv_chunk * (size_t)c->kv_row) +
                        at * (size_t)c->kv_heads;
            for (int kh = 0; kh < c->nkv; ++kh) {
                float *x = k + (size_t)kh * hd;
                norm_v(x, hd, value ? NULL : pq(d, y->k_norm), c->eps);
                if (!value) rope_v(x, hd, rt, rt + pairs, pairs);
                quant_head(x, hd, row + (size_t)kh * hd, &sc[kh]);
            }
        }
    }
}

// the attention's rows (CR) into the output projection's input tiles
static void gp_core(gd_t *d, int p, int th) {
    (void)th;
    const glayer_t *y = d->ly;
    const int qn = y->q.n, ncb = y->o.k / 32;
    const float *CR = (const float *)(d->IO + d->L.CR);
    const HVX_Vector *r0 = 2 * p < d->n ? (const HVX_Vector *)(CR + (size_t)(2 * p) * qn) : NULL;
    const HVX_Vector *r1 = 2 * p + 1 < d->n ? (const HVX_Vector *)(CR + (size_t)(2 * p + 1) * qn) : NULL;
    uint8_t *A = d->IO + d->L.A;
    for (int cb = 0; cb < ncb; ++cb) {
        const int in = cb < qn / 32;
        *cvec(A, ncb, p, cb) = hf_of(in && r0 ? r0[cb] : Q6_V_vzero(), in && r1 ? r1[cb] : Q6_V_vzero());
    }
}

// x (rows 2p, 2p + 1) += norm(m) (wm) [then times the layer's scalar], m the sum of `parts` results (their croutons of
// pair p); then (tncb) the next product's input tiles: norm(x) (wt), or x itself (raw). A pass over m (its sums of
// squares on the way, m kept in tmp as qf32), one over x (x's on the way, or the raw tiles), one for normed tiles
typedef struct {
    uint64_t offs[GDEC_MAX_DOWN];
    int ncbs[GDEC_MAX_DOWN], parts;
    const HVX_Vector *wm, *wt;   // the norms' weights: m's in sf (P), the tiles' in qf32 (PQ)
    int scaled, tncb, raw;
    float scalar;
} rs_t;

static inline __attribute__((always_inline)) void resid_k(gd_t *d, int p, int th, const rs_t *a, int ok0, int ok1) {
    // (qf32 only in the sums: a product of two qf32 that came out of an add, or out of hf x 1.0, lost precision -
    // measured: KL 0.2-7.8 against exact; each product here sf x sf, its qf32 converted back before it goes on)
    const int H = d->c.hidden, nv = H / 32;
    const float eps = d->c.eps;
    HVX_Vector *m0 = (HVX_Vector *)tmp(d, th, 0), *m1 = (HVX_Vector *)tmp(d, th, 1);   // m in sf
    const HVX_Vector z = Q6_V_vzero();
    HVX_Vector a0 = z, a1 = z, b0 = z, b1 = z;
    for (int cb = 0; cb < nv; cb += 2) {   // (nv even: gdec_open)
        HVX_Vector e0, o0, e1, o1;
        q_of_hf(*cvec(d->IO + a->offs[0], a->ncbs[0], p, cb), &e0, &o0);
        q_of_hf(*cvec(d->IO + a->offs[0], a->ncbs[0], p, cb + 1), &e1, &o1);
        HVX_Vector se0 = sf_q(e0), so0 = sf_q(o0), se1 = sf_q(e1), so1 = sf_q(o1);
        for (int j = 1; j < a->parts; ++j) {
            HVX_Vector e2, o2, e3, o3;
            q_of_hf(*cvec(d->IO + a->offs[j], a->ncbs[j], p, cb), &e2, &o2);
            q_of_hf(*cvec(d->IO + a->offs[j], a->ncbs[j], p, cb + 1), &e3, &o3);
            se0 = sf_q(Q6_Vqf32_vadd_VsfVsf(se0, sf_q(e2))), so0 = sf_q(Q6_Vqf32_vadd_VsfVsf(so0, sf_q(o2)));
            se1 = sf_q(Q6_Vqf32_vadd_VsfVsf(se1, sf_q(e3))), so1 = sf_q(Q6_Vqf32_vadd_VsfVsf(so1, sf_q(o3)));
        }
        m0[cb] = se0, m1[cb] = so0, m0[cb + 1] = se1, m1[cb + 1] = so1;
        a0 = q_add(a0, q_muls(se0, se0)), a1 = q_add(a1, q_muls(so0, so0));
        b0 = q_add(b0, q_muls(se1, se1)), b1 = q_add(b1, q_muls(so1, so1));
    }
    const HVX_Vector f0 = f_sp(ok0 ? 1.0f / sqrtf(q_hsum(q_add(a0, b0)) / (float)H + eps) : 0.0f);
    const HVX_Vector f1 = f_sp(ok1 ? 1.0f / sqrtf(q_hsum(q_add(a1, b1)) / (float)H + eps) : 0.0f);
    const HVX_Vector sc = f_sp(a->scalar);
    HVX_Vector *x0 = (HVX_Vector *)xrow(d, 2 * p), *x1 = (HVX_Vector *)xrow(d, 2 * p + 1);
    const HVX_Vector *wm = a->wm;   // sf
    const int tncb = a->tncb, normed = tncb && !a->raw, scaled = a->scaled;
    uint8_t *A = d->IO + d->L.A;
    HVX_Vector c0 = z, c1 = z, c2 = z, c3 = z;
    for (int cb = 0; cb < nv; cb += 2) {
        HVX_Vector y0 = z, y1 = z, y2 = z, y3 = z;   // sf: rows 0, 1 at cb; rows 0, 1 at cb + 1
        if (ok0) {
            const HVX_Vector t0 = sf_q(q_muls(m0[cb], wm[cb])), t2 = sf_q(q_muls(m0[cb + 1], wm[cb + 1]));
            y0 = sf_q(q_adds(q_muls(t0, f0), x0[cb])), y2 = sf_q(q_adds(q_muls(t2, f0), x0[cb + 1]));
            if (scaled) y0 = sf_q(q_muls(y0, sc)), y2 = sf_q(q_muls(y2, sc));
            x0[cb] = y0, x0[cb + 1] = y2;
        }
        if (ok1) {
            const HVX_Vector t1 = sf_q(q_muls(m1[cb], wm[cb])), t3 = sf_q(q_muls(m1[cb + 1], wm[cb + 1]));
            y1 = sf_q(q_adds(q_muls(t1, f1), x1[cb])), y3 = sf_q(q_adds(q_muls(t3, f1), x1[cb + 1]));
            if (scaled) y1 = sf_q(q_muls(y1, sc)), y3 = sf_q(q_muls(y3, sc));
            x1[cb] = y1, x1[cb + 1] = y3;
        }
        if (normed) {
            c0 = q_add(c0, q_muls(y0, y0)), c1 = q_add(c1, q_muls(y1, y1));
            c2 = q_add(c2, q_muls(y2, y2)), c3 = q_add(c3, q_muls(y3, y3));
        } else if (tncb) {   // raw: x itself
            *cvec(A, tncb, p, cb) = hf_of(y0, y1);
            *cvec(A, tncb, p, cb + 1) = hf_of(y2, y3);
        }
    }
    if (!tncb) return;
    const int nc = nv < tncb ? nv : tncb;
    if (normed) {   // as tiles_k (the weights in qf32: a product of fresh products, checked)
        const HVX_Vector g0 = f_sp(ok0 ? 1.0f / sqrtf(q_hsum(q_add(c0, c2)) / (float)H + eps) : 0.0f);
        const HVX_Vector g1 = f_sp(ok1 ? 1.0f / sqrtf(q_hsum(q_add(c1, c3)) / (float)H + eps) : 0.0f);
        const HVX_Vector *wt = a->wt;
        for (int cb = 0; cb < nc; ++cb) {
            const HVX_Vector t0 = ok0 ? q_mul(q_muls(x0[cb], g0), wt[cb]) : z;
            const HVX_Vector t1 = ok1 ? q_mul(q_muls(x1[cb], g1), wt[cb]) : z;
            *cvec(A, tncb, p, cb) = hf_of_q(t0, t1);
        }
    }
    for (int cb = nc; cb < tncb; ++cb) *cvec(A, tncb, p, cb) = z;
}

static void resid(gd_t *d, int p, int th, const rs_t *a) {
    if (2 * p + 1 < d->n) resid_k(d, p, th, a, 1, 1);
    else resid_k(d, p, th, a, 2 * p < d->n, 0);
}

static void gp_post(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    rs_t a;
    a.offs[0] = d->L.OO, a.ncbs[0] = y->o.n / 32, a.parts = 1;
    a.wm = (const HVX_Vector *)pw(d, y->post_attn), a.scaled = 0, a.scalar = 1.0f;
    a.tncb = y->gate.k / 32, a.wt = pq(d, y->pre_ff), a.raw = 0;
    resid(d, p, th, &a);
}

static void gp_down(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const int H = d->c.hidden;
    rs_t a;
    for (int j = 0; j < y->n_down; ++j)
        a.offs[j] = d->L.D + (uint64_t)j * GDEC_MAX_N * (uint64_t)H * 2u, a.ncbs[j] = H / 32;
    a.parts = y->n_down, a.wm = (const HVX_Vector *)pw(d, y->post_ff), a.wt = NULL;
    if (d->c.ple_dim)   // the PLE's gate reads x itself
        a.scaled = 0, a.scalar = 1.0f, a.tncb = y->ple_gate.k / 32, a.raw = 1;
    else
        a.scaled = 1, a.scalar = y->scalar, a.tncb = 0, a.raw = 0;
    resid(d, p, th, &a);
}

// gelu(gate) * the per-layer input, into the PLE projection's input tiles (past ple_dim: zeros). gelu as the engine's
// gelu_fast: x sigmoid(u), u = 1.5957691216 (x + 0.044715 x^3); sigmoid(u) = 1 / (1 + e^-|u|), times e^-|u| for u < 0
static void gp_ple(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const int P = d->c.ple_dim, nv = P / 32, ncb = y->ple_proj.k / 32;
    float *g0 = tmp(d, th, 0), *g1 = tmp(d, th, 1);
    rows_of(d->IO + d->L.G, y->ple_gate.n / 32, p, 0, nv, g0, g1);
    const size_t LP = (size_t)d->c.n_layers * P;
    for (int s = 0; s < 2; ++s) {
        const int r = 2 * p + s;
        HVX_Vector *g = (HVX_Vector *)(s ? g1 : g0);
        if (r >= d->n) {
            for (int i = 0; i < nv; ++i) g[i] = Q6_V_vzero();
            continue;
        }
        const HVX_Vector *pl = (const HVX_Vector *)((const float *)(d->IO + d->c.io_ple) + (size_t)r * LP + (size_t)d->layer * P);
        for (int i = 0; i < nv; ++i) {
            const HVX_Vector x = g[i];
            const HVX_Vector x3 = f_mul(f_mul(x, x), x);
            const HVX_Vector u = f_mul(f_add(x, f_mul(x3, f_sp(0.044715f))), f_sp(1.5957691216f));
            const HVX_Vector au = Q6_V_vand_VV(u, Q6_V_vsplat_R(0x7FFFFFFF));
            const HVX_Vector e = f_exp(f_sub(Q6_V_vzero(), au));   // e^-|u| in (0, 1]
            const HVX_Vector den = f_add(f_sp(1.0f), e);
            HVX_Vector rc = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), den);   // 1 / den: three Newton steps
            for (int k = 0; k < 3; ++k) rc = f_mul(rc, f_sub(f_sp(2.0f), f_mul(den, rc)));
            const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), u);   // the sign bit (u < 0)
            const HVX_Vector sg = Q6_V_vmux_QVV(neg, f_mul(e, rc), rc);
            g[i] = f_mul(f_mul(x, sg), pl[i]);
        }
    }
    uint8_t *A = d->IO + d->L.A;
    const HVX_Vector *a = (const HVX_Vector *)g0, *b = (const HVX_Vector *)g1;
    for (int cb = 0; cb < ncb; ++cb)
        *cvec(A, ncb, p, cb) = cb < nv ? hf_of(a[cb], b[cb]) : Q6_V_vzero();
}

static void gp_ple2(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    rs_t a;
    a.offs[0] = d->L.OO, a.ncbs[0] = y->ple_proj.n / 32, a.parts = 1;
    a.wm = (const HVX_Vector *)pw(d, y->ple_norm), a.scaled = 1, a.scalar = y->scalar, a.tncb = 0, a.wt = NULL, a.raw = 0;
    resid(d, p, th, &a);
}

// ---- the steps as they were (IEEE sf intrinsics), for checking the qf32 ones: GDEC_OLD's phases take these
static inline float sumsq_o(const HVX_Vector *v, int nv) {
    HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
    int i = 0;
    for (; i + 2 <= nv; i += 2) {
        s0 = f_add(s0, f_mul(v[i], v[i]));
        s1 = f_add(s1, f_mul(v[i + 1], v[i + 1]));
    }
    if (i < nv) s0 = f_add(s0, f_mul(v[i], v[i]));
    return f_hsum(f_add(s0, s1));
}

// x (n floats, 32 k) normed (RMSNorm, times w; w NULL: none) in place
static inline void norm_v_o(float *x, int n, const float *w, float eps) {
    HVX_Vector *v = (HVX_Vector *)x;
    const HVX_Vector f = f_sp(1.0f / sqrtf(sumsq_o(v, n / 32) / (float)n + eps));
    const HVX_Vector *wv = (const HVX_Vector *)w;
    if (w)
        for (int i = 0; i < n / 32; ++i) v[i] = f_mul(f_mul(v[i], f), wv[i]);
    else
        for (int i = 0; i < n / 32; ++i) v[i] = f_mul(v[i], f);
}

// rotate-half RoPE on the first `pairs` pairs (i, i + hd / 2) of a head
static inline void rope_v_o(float *x, int hd, const float *cs, const float *sn, int pairs) {
    HVX_Vector *a = (HVX_Vector *)x, *b = (HVX_Vector *)(x + hd / 2);
    const HVX_Vector *c = (const HVX_Vector *)cs, *s = (const HVX_Vector *)sn;
    for (int i = 0; i < pairs / 32; ++i) {
        const HVX_Vector x1 = a[i], x2 = b[i];
        a[i] = f_sub(f_mul(x1, c[i]), f_mul(x2, s[i]));
        b[i] = f_add(f_mul(x2, c[i]), f_mul(x1, s[i]));
    }
}

// a head's hd floats as int8 codes (amax / 127 the scale, rounded to nearest: the engine's quant_row) into dst
static void quant_head_o(const float *x, int hd, int8_t *dst, float *scale) {
    const HVX_Vector *v = (const HVX_Vector *)x;
    HVX_Vector m = Q6_V_vzero();
    for (int i = 0; i < hd / 32; ++i) m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vand_VV(v[i], Q6_V_vsplat_R(0x7FFFFFFF)));
    const float amax = f_hmax(m);
    *scale = amax / 127.0f;
    const HVX_Vector inv = f_sp(amax > 0.0f ? 127.0f / amax : 0.0f);
    const HVX_Vector magic = f_sp(12582912.0f), mb = Q6_V_vsplat_R(0x4B400000);
    const HVX_Vector hi = Q6_V_vsplat_R(127), lo = Q6_V_vsplat_R(-127);
    for (int i = 0; i < hd / 128; ++i) {
        HVX_Vector w[4];
        for (int j = 0; j < 4; ++j) {
            const HVX_Vector r = Q6_Vw_vsub_VwVw(f_add(f_mul(v[4 * i + j], inv), magic), mb);
            w[j] = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(r, hi), lo);
        }
        const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(w[1], w[0]), h23 = Q6_Vh_vpack_VwVw_sat(w[3], w[2]);
        ((HVX_Vector *)dst)[i] = Q6_Vb_vpack_VhVh_sat(h23, h01);
    }
}

// the residual_o rows of pair p normed (w) into the input tiles of a product of ncb croutons (past the rows: zeros)
static void x_tiles_o(gd_t *d, int p, const float *w, int ncb, int raw) {
    const int H = d->c.hidden, nv = H / 32;
    HVX_Vector f[2] = {Q6_V_vzero(), Q6_V_vzero()};
    const HVX_Vector *x[2] = {NULL, NULL};
    for (int s = 0; s < 2; ++s) {
        const int r = 2 * p + s;
        if (r >= d->n) continue;
        x[s] = (const HVX_Vector *)xrow(d, r);
        f[s] = raw ? f_sp(1.0f) : f_sp(1.0f / sqrtf(sumsq_o(x[s], nv) / (float)H + d->c.eps));
    }
    const HVX_Vector *wv = (const HVX_Vector *)w;
    uint8_t *A = d->IO + d->L.A;
    for (int cb = 0; cb < ncb; ++cb) {
        HVX_Vector e = Q6_V_vzero(), o = Q6_V_vzero();
        if (cb < nv) {
            if (x[0]) e = raw ? x[0][cb] : f_mul(f_mul(x[0][cb], f[0]), wv[cb]);
            if (x[1]) o = raw ? x[1][cb] : f_mul(f_mul(x[1][cb], f[1]), wv[cb]);
        }
        *cvec(A, ncb, p, cb) = hf_of(e, o);
    }
}

// a result's croutons [c0, c0 + nv) of pair p as two rows of floats (+= when add)
static void rows_of_o(uint8_t *buf, int ncb, int p, int c0, int nv, float *r0, float *r1, int add) {
    HVX_Vector *a = (HVX_Vector *)r0, *b = (HVX_Vector *)r1;
    for (int i = 0; i < nv; ++i) {
        HVX_Vector e, o;
        sf_of(*cvec(buf, ncb, p, c0 + i), &e, &o);
        a[i] = add ? f_add(a[i], e) : e;
        b[i] = add ? f_add(b[i], o) : o;
    }
}

static void gp_in_o(gd_t *d, int p, int th) {
    (void)th;
    x_tiles_o(d, p, pw(d, d->ly->in_norm), d->ly->q.k / 32, 0);
}

// the queries normed and turned into QR; the keys normed and turned, the values normed, into the KV cache as int8
static void gp_qkv_o(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const gdec_cfg_t *c = &d->c;
    const int hd = y->hd, qn = y->q.n, nq = qn / 32, pairs = c->pairs[y->rope];
    float *QR = (float *)(d->IO + d->L.QR);
    float *q0 = QR + (size_t)(2 * p) * qn, *q1 = q0 + qn;
    rows_of_o(d->IO + d->L.OQ, nq, p, 0, nq, q0, q1, 0);
    for (int s = 0; s < 2; ++s) {
        const int r = 2 * p + s;
        if (r >= d->n) continue;
        float *q = s ? q1 : q0;
        const float *rt = (const float *)(d->IO + c->io_rope[y->rope]) + (size_t)r * 2u * pairs;
        for (int h = 0; h < c->nh; ++h) {
            norm_v_o(q + (size_t)h * hd, hd, pw(d, y->q_norm), c->eps);
            rope_v_o(q + (size_t)h * hd, hd, rt, rt + pairs, pairs);
        }
    }
    if (!y->own_kv) return;
    const int nk = y->k.n / 32;
    for (int value = 0; value < 2; ++value) {
        float *k0 = tmp(d, th, 0), *k1 = tmp(d, th, 1);
        rows_of_o(d->IO + (value ? d->L.OV : d->L.OK), nk, p, 0, nk, k0, k1, 0);
        for (int s = 0; s < 2; ++s) {
            const int r = 2 * p + s;
            if (r >= d->n) continue;
            float *k = s ? k1 : k0;
            const int pos = d->apos + r;
            const float *rt = (const float *)(d->IO + c->io_rope[y->rope]) + (size_t)r * 2u * pairs;
            uint8_t *base = d->kv[pos / c->kv_chunk];
            const size_t at = (size_t)(value * c->n_attn + y->slot) * (size_t)c->kv_chunk + (size_t)(pos % c->kv_chunk);
            int8_t *row = (int8_t *)(base + at * (size_t)c->kv_row);
            float *sc = (float *)(base + 2u * (size_t)c->n_attn * (size_t)c->kv_chunk * (size_t)c->kv_row) +
                        at * (size_t)c->kv_heads;
            for (int kh = 0; kh < c->nkv; ++kh) {
                float *x = k + (size_t)kh * hd;
                norm_v_o(x, hd, value ? NULL : pw(d, y->k_norm), c->eps);
                if (!value) rope_v_o(x, hd, rt, rt + pairs, pairs);
                quant_head_o(x, hd, row + (size_t)kh * hd, &sc[kh]);
            }
        }
    }
}

// the attention's rows (CR) into the output projection's input tiles
static void gp_core_o(gd_t *d, int p, int th) {
    (void)th;
    const glayer_t *y = d->ly;
    const int qn = y->q.n, ncb = y->o.k / 32;
    const float *CR = (const float *)(d->IO + d->L.CR);
    const HVX_Vector *r0 = 2 * p < d->n ? (const HVX_Vector *)(CR + (size_t)(2 * p) * qn) : NULL;
    const HVX_Vector *r1 = 2 * p + 1 < d->n ? (const HVX_Vector *)(CR + (size_t)(2 * p + 1) * qn) : NULL;
    uint8_t *A = d->IO + d->L.A;
    for (int cb = 0; cb < ncb; ++cb) {
        const int in = cb < qn / 32;
        *cvec(A, ncb, p, cb) = hf_of(in && r0 ? r0[cb] : Q6_V_vzero(), in && r1 ? r1[cb] : Q6_V_vzero());
    }
}

// x += norm(m) (w); returns nothing: the rows m (two, of hidden floats) in tmp
static void residual_o(gd_t *d, int p, int th, uint32_t w, float scalar_after, int scalar) {
    const int H = d->c.hidden;
    for (int s = 0; s < 2; ++s) {
        const int r = 2 * p + s;
        if (r >= d->n) continue;
        float *m = tmp(d, th, s);
        norm_v_o(m, H, pw(d, w), d->c.eps);
        HVX_Vector *x = (HVX_Vector *)xrow(d, r);
        const HVX_Vector *mv = (const HVX_Vector *)m;
        const HVX_Vector sc = f_sp(scalar_after);
        for (int i = 0; i < H / 32; ++i) x[i] = scalar ? f_mul(f_add(x[i], mv[i]), sc) : f_add(x[i], mv[i]);
    }
}

static void gp_post_o(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const int nv = d->c.hidden / 32;
    rows_of_o(d->IO + d->L.OO, y->o.n / 32, p, 0, nv, tmp(d, th, 0), tmp(d, th, 1), 0);
    residual_o(d, p, th, y->post_attn, 1.0f, 0);
    x_tiles_o(d, p, pw(d, y->pre_ff), y->gate.k / 32, 0);
}

static void gp_down_o(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const int H = d->c.hidden, nv = H / 32;
    for (int j = 0; j < y->n_down; ++j)
        rows_of_o(d->IO + d->L.D + (size_t)j * GDEC_MAX_N * (size_t)H * 2u, nv, p, 0, nv, tmp(d, th, 0), tmp(d, th, 1),
                j > 0);
    if (d->c.ple_dim) {
        residual_o(d, p, th, y->post_ff, 1.0f, 0);
        x_tiles_o(d, p, NULL, y->ple_gate.k / 32, 1);   // the gate reads x itself
    } else {
        residual_o(d, p, th, y->post_ff, y->scalar, 1);
    }
}

// gelu(gate) * the per-layer input, into the PLE projection's input tiles (past ple_dim: zeros). gelu as the engine's
// gelu_fast: x sigmoid(u), u = 1.5957691216 (x + 0.044715 x^3); sigmoid(u) = 1 / (1 + e^-|u|), times e^-|u| for u < 0
static void gp_ple_o(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const int P = d->c.ple_dim, nv = P / 32, ncb = y->ple_proj.k / 32;
    float *g0 = tmp(d, th, 0), *g1 = tmp(d, th, 1);
    rows_of_o(d->IO + d->L.G, y->ple_gate.n / 32, p, 0, nv, g0, g1, 0);
    const size_t LP = (size_t)d->c.n_layers * P;
    for (int s = 0; s < 2; ++s) {
        const int r = 2 * p + s;
        HVX_Vector *g = (HVX_Vector *)(s ? g1 : g0);
        if (r >= d->n) {
            for (int i = 0; i < nv; ++i) g[i] = Q6_V_vzero();
            continue;
        }
        const HVX_Vector *pl = (const HVX_Vector *)((const float *)(d->IO + d->c.io_ple) + (size_t)r * LP + (size_t)d->layer * P);
        for (int i = 0; i < nv; ++i) {
            const HVX_Vector x = g[i];
            const HVX_Vector x3 = f_mul(f_mul(x, x), x);
            const HVX_Vector u = f_mul(f_add(x, f_mul(x3, f_sp(0.044715f))), f_sp(1.5957691216f));
            const HVX_Vector au = Q6_V_vand_VV(u, Q6_V_vsplat_R(0x7FFFFFFF));
            const HVX_Vector e = f_exp(f_sub(Q6_V_vzero(), au));   // e^-|u| in (0, 1]
            const HVX_Vector den = f_add(f_sp(1.0f), e);
            HVX_Vector rc = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), den);   // 1 / den: three Newton steps
            for (int k = 0; k < 3; ++k) rc = f_mul(rc, f_sub(f_sp(2.0f), f_mul(den, rc)));
            const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), u);   // the sign bit (u < 0)
            const HVX_Vector sg = Q6_V_vmux_QVV(neg, f_mul(e, rc), rc);
            g[i] = f_mul(f_mul(x, sg), pl[i]);
        }
    }
    uint8_t *A = d->IO + d->L.A;
    const HVX_Vector *a = (const HVX_Vector *)g0, *b = (const HVX_Vector *)g1;
    for (int cb = 0; cb < ncb; ++cb)
        *cvec(A, ncb, p, cb) = cb < nv ? hf_of(a[cb], b[cb]) : Q6_V_vzero();
}

static void gp_ple2_o(gd_t *d, int p, int th) {
    const glayer_t *y = d->ly;
    const int nv = d->c.hidden / 32;
    rows_of_o(d->IO + d->L.OO, y->ple_proj.n / 32, p, 0, nv, tmp(d, th, 0), tmp(d, th, 1), 0);
    residual_o(d, p, th, y->ple_norm, y->scalar, 1);
}
// ---- what a task reads, fetched into L2 ahead (l2fetch: rows of `width` bytes `stride` apart, `height` of them) - a
// row pair's 128 bytes of every 2 KB crouton come in one line at a time otherwise (~12 GB/s a thread, measured)
// measured (E4B, 2048 tokens; tasks of 1 / 2 / 4 / 8 pairs, the inputs fetched as a task starts / one task ahead):
// 2 pairs as they start 901 tokens/s, 4: 888, 8: 860; one ahead 800-828 (L2 crowded); none 813
#define GD_TASK 2   // row pairs a task (4 rows: each crouton's 256 bytes of them contiguous)
#define GD_PF 2

static inline void pf2d(const void *p, uint32_t stride, uint32_t width, uint32_t height) {
    if (!height || !width || stride > 0xFFFFu || width > 0xFFFFu || height > 0xFFFFu) return;
    Q6_l2fetch_AP((void *)p, ((uint64_t)stride << 32) | ((uint64_t)width << 16) | (uint64_t)height);
}

static inline void pf_cr(gd_t *d, uint64_t off, int ncb, int p0, int np, int nc) {   // pairs p0.. of croutons 0..nc
    pf2d(cvec(d->IO + off, ncb, p0, 0), 2048u, (uint32_t)np * 128u, (uint32_t)nc);
}

static void gd_pf(gd_t *d, int p0, int np) {
    const glayer_t *y = d->ly;
    const gdec_cfg_t *c = &d->c;
    const int H = c->hidden, nv = H / 32, r0 = 2 * p0;
    int nr = d->n - r0;
    if (nr > 2 * np) nr = 2 * np;
    if (nr < 0) nr = 0;
    const uint32_t xb = (uint32_t)H * 4u;
    switch (d->phase) {
    case GP_IN:
        pf2d(xrow(d, r0), xb, xb, (uint32_t)nr);
        break;
    case GP_QKV: {
        const int nq = y->q.n / 32;
        pf_cr(d, d->L.OQ, nq, p0, np, nq);
        if (y->own_kv) {
            const int nk = y->k.n / 32;
            pf_cr(d, d->L.OK, nk, p0, np, nk);
            pf_cr(d, d->L.OV, nk, p0, np, nk);
        }
        const uint32_t rb = 8u * (uint32_t)c->pairs[y->rope];
        pf2d(d->IO + c->io_rope[y->rope] + (size_t)r0 * rb, rb, rb, (uint32_t)nr);
        break;
    }
    case GP_CORE: {
        const uint32_t qb = (uint32_t)y->q.n * 4u;
        pf2d(d->IO + d->L.CR + (size_t)r0 * qb, qb, qb, (uint32_t)nr);
        break;
    }
    case GP_POST:
        pf_cr(d, d->L.OO, y->o.n / 32, p0, np, nv);
        pf2d(xrow(d, r0), xb, xb, (uint32_t)nr);
        break;
    case GP_DOWN:
        for (int j = 0; j < y->n_down; ++j) pf_cr(d, d->L.D + (uint64_t)j * GDEC_MAX_N * (uint64_t)H * 2u, nv, p0, np, nv);
        pf2d(xrow(d, r0), xb, xb, (uint32_t)nr);
        break;
    case GP_PLE: {
        const int P = c->ple_dim;
        pf_cr(d, d->L.G, y->ple_gate.n / 32, p0, np, P / 32);
        pf2d((const float *)(d->IO + c->io_ple) + (size_t)r0 * c->n_layers * P + (size_t)d->layer * P,
             (uint32_t)c->n_layers * (uint32_t)P * 4u, (uint32_t)P * 4u, (uint32_t)nr);
        break;
    }
    case GP_PLE2:
        pf_cr(d, d->L.OO, y->ple_proj.n / 32, p0, np, nv);
        pf2d(xrow(d, r0), xb, xb, (uint32_t)nr);
        break;
    default:
        break;
    }
}

static void gd_phase(gd_t *d, int th) {
    if (d->phase == GP_ATTN) {
        program(&d->at, th);
        return;
    }
    static void (*const fns[])(gd_t *, int, int) = {gp_in, gp_qkv, NULL, gp_core, gp_post, gp_down, gp_ple, gp_ple2};
    static void (*const olds[])(gd_t *, int, int) = {gp_in_o, gp_qkv_o, NULL, gp_core_o, gp_post_o, gp_down_o,
                                                      gp_ple_o, gp_ple2_o};
    void (*fn)(gd_t *, int, int) = (d->old_mask >> d->phase) & 1 ? olds[d->phase] : fns[d->phase];
    const int pairs = d->m / 2, T = d->task, tasks = (pairs + T - 1) / T, pf = d->pf;
    int t = __atomic_fetch_add(&d->ctr, 1, __ATOMIC_RELAXED);
    if (pf >= 2 && t < tasks) gd_pf(d, t * T, T);
    while (t < tasks) {
        int next = -1;
        if (pf == 3) {   // the next task taken now, its inputs on their way while this one runs
            next = __atomic_fetch_add(&d->ctr, 1, __ATOMIC_RELAXED);
            if (next < tasks) gd_pf(d, next * T, T);
        }
        const int p1 = (t + 1) * T < pairs ? (t + 1) * T : pairs;
        for (int p = t * T; p < p1; ++p) fn(d, p, th);
        __atomic_add_fetch(&d->prof[GDEC_PROF_TASKS0 + th], 1, __ATOMIC_RELAXED);
        if (pf == 3) {
            t = next;
        } else {
            t = __atomic_fetch_add(&d->ctr, 1, __ATOMIC_RELAXED);
            if (pf == 2 && t < tasks) gd_pf(d, t * T, T);
        }
    }
}

typedef struct {
    gd_t *d;
    int th;
} gd_arg_t;
static gd_arg_t g_gd_args[GDEC_THREADS];

static void gd_worker(void *arg) {
    gd_arg_t *x = (gd_arg_t *)arg;
    gd_t *d = x->d;
    int seen = 0;
    for (;;) {
        int go;
        while ((go = __atomic_load_n(&d->go, __ATOMIC_ACQUIRE)) == seen) qurt_futex_wait((void *)&d->go, seen);
        seen = go;
        if (__atomic_load_n(&d->quit, __ATOMIC_ACQUIRE)) break;
        const uint64_t tw = HAP_perf_get_time_us();
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
            d->failed = 1;
        } else {
            const uint64_t tb = HAP_perf_get_time_us();
            __atomic_add_fetch(&d->prof[GDEC_PROF_LOCKW], tb - tw, __ATOMIC_RELAXED);
            gd_phase(d, x->th);
            __atomic_add_fetch(&d->prof[GDEC_PROF_BUSY], HAP_perf_get_time_us() - tb, __ATOMIC_RELAXED);
            qurt_hvx_unlock();
        }
        __atomic_add_fetch(&d->running, -1, __ATOMIC_ACQ_REL);
    }
    qurt_thread_exit(0);
}

// the pool through one phase (the attention's: this thread on HMX meanwhile)
static void gd_run(gd_t *d, int phase, int prof) {
    const uint64_t t0 = HAP_perf_get_time_us();
    d->phase = phase;
    d->ctr = 0;
    __atomic_store_n(&d->running, d->nthreads, __ATOMIC_RELEASE);
    __atomic_add_fetch(&d->go, 1, __ATOMIC_ACQ_REL);
    qurt_futex_wake((void *)&d->go, GDEC_THREADS);
    if (phase == GP_ATTN) program(&d->at, d->nthreads);
    else if (d->main_joins) gd_phase(d, GDEC_THREADS);   // (its own TMP rows: gdec_layout's)
    while (__atomic_load_n(&d->running, __ATOMIC_ACQUIRE) > 0) asm volatile("pause(#16)");
    d->prof[prof] += HAP_perf_get_time_us() - t0;
}

static void gd_stop(gd_t *d) {
    __atomic_store_n(&d->quit, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&d->go, 1, __ATOMIC_ACQ_REL);
    qurt_futex_wake((void *)&d->go, GDEC_THREADS);
    for (int t = 0; t < d->nthreads; ++t) {
        int st;
        if (d->tid[t]) qurt_thread_join(d->tid[t], &st);
    }
    for (int t = 0; t < GDEC_THREADS; ++t) free(d->stack[t]);
    free(d->PQ);
    d->PQ = NULL;
}

static void gd_free(gd_t *d) {
    gd_stop(d);
    for (int i = 0; i < d->n_mfd; ++i) HAP_mmap_put(d->mfd[i]);
    for (int i = 0; i < ATTN_MAX_CHUNKS; ++i)
        if (d->kv[i]) HAP_mmap_put(d->kv_fd[i]);
    if (d->P) HAP_mmap_put(d->p_fd);
    if (d->IO) HAP_mmap_put(d->io_fd);
    free(d);
}

static const uint8_t *map_mat(gd_t *d, const gmat_t *m) {
    if (m->fd <= 0) return NULL;
    void *v = NULL;
    uint64 pa = 0;
    if (HAP_mmap_get(m->fd, &v, &pa)) return NULL;
    d->mfd[d->n_mfd++] = m->fd;   // (each get has its put at close)
    return (const uint8_t *)v;
}

static int gmat_ok(const gmat_t *m, int k_min, int n) {
    return m->fd > 0 && m->k >= k_min && m->k % 32 == 0 && m->k >= 512 && m->n == n && n % 32 == 0;
}

int geode_npu_gdec_open(remote_handle64 h, const unsigned char *cfg, int cfg_len, int p_fd, int io_fd) {
    (void)h;
    if (g_gd) return AEE_EALREADY;
    if (cfg_len != (int)sizeof(gdec_cfg_t)) return AEE_EBADPARM;
    if (g_orders_ok < 0) a_checks();
    if (!g_orders_ok) return AEE_EUNSUPPORTED;
    const gdec_cfg_t *c = (const gdec_cfg_t *)cfg;
    const int H = c->hidden, P = c->ple_dim;
    if (c->version != GDEC_VERSION || c->n_layers < 1 || c->n_layers > GDEC_MAX_LAYERS || H % 32 || H > 8192 ||
        H % 64 || P % 32 || P > 512 || c->nkv < 1 || c->nh % c->nkv || 32 % (c->nh / c->nkv) || c->pairs[0] % 32 ||
        c->pairs[1] % 32 || c->kv_chunk < 32)
        return AEE_EBADPARM;
    for (int l = 0; l < c->n_layers; ++l) {
        const glayer_t *y = &c->layers[l];
        const int hd = y->hd, qn = c->nh * hd, kn = c->nkv * hd;
        if (hd % 128 || hd > 512 || y->rope < 0 || y->rope > 1 || 2 * c->pairs[y->rope] > hd || y->n_down < 1 ||
            y->n_down > GDEC_MAX_DOWN || !gmat_ok(&y->q, H, qn) || !gmat_ok(&y->o, qn, H) ||
            !gmat_ok(&y->gate, H, y->gate.n) || !gmat_ok(&y->up, H, y->gate.n) || y->up.k != y->gate.k ||
            y->gate.n % (32 * y->n_down) || (y->own_kv && (!gmat_ok(&y->k, H, kn) || !gmat_ok(&y->v, H, kn) ||
                                                           y->k.k != y->q.k || y->v.k != y->q.k)) ||
            y->slot < 0 || y->slot >= c->n_attn)
            return AEE_EBADPARM;
        for (int j = 0; j < y->n_down; ++j)
            if (!gmat_ok(&y->down[j], y->gate.n / y->n_down, H) || y->down[j].k != y->gate.n / y->n_down)
                return AEE_EBADPARM;
        if (P && (!gmat_ok(&y->ple_gate, H, P) || !gmat_ok(&y->ple_proj, P, H)))
            return AEE_EBADPARM;
        if (y->gate.k != y->q.k || (P && y->ple_gate.k != y->q.k)) return AEE_EBADPARM;   // (one input: x's tiles)
    }
    gd_t *d = (gd_t *)calloc(1, sizeof *d);
    if (!d) return AEE_ENOMEMORY;
    memcpy(&d->c, c, sizeof *c);
    d->L = gdec_layout(&d->c);
    d->p_fd = p_fd, d->io_fd = io_fd;
    if (d->L.end > d->c.io_bytes || !(d->P = (uint8_t *)map_fd(p_fd))) {
        free(d);
        return AEE_EBADPARM;
    }
    if (!(d->IO = (uint8_t *)map_fd(io_fd))) {
        HAP_mmap_put(p_fd);
        free(d);
        return AEE_EBADPARM;
    }
    d->prof = (uint64_t *)(d->IO + d->c.io_bytes - 256u);
    memset(d->prof, 0, 256);
    int e = 0;
    {   // the norms' weights in qf32 (the CPU wrote them: seen afresh)
        uint32_t end = 0;
        for (int l = 0; l < c->n_layers; ++l) {
            const glayer_t *y = &c->layers[l];
            const uint32_t hb = (uint32_t)H * 4u, db = (uint32_t)y->hd * 4u;
            const uint32_t ends[7] = {y->in_norm + hb, y->post_attn + hb, y->pre_ff + hb, y->post_ff + hb,
                                      y->ple_norm + hb, y->q_norm + db, y->k_norm + db};
            for (int i = 0; i < 7; ++i)
                if (ends[i] > end) end = ends[i];
        }
        end = (end + 127u) & ~127u;   // (the engine pads P to 128 bytes)
        qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
        if (!(d->PQ = (uint8_t *)memalign(128, end ? end : 128))) {
            e = AEE_ENOMEMORY;
        } else {
            const HVX_Vector *src = (const HVX_Vector *)d->P;
            HVX_Vector *dst = (HVX_Vector *)d->PQ;
            for (uint32_t i = 0; i < end / 128u; ++i) dst[i] = q_sf(src[i]);
        }
    }
    for (int l = 0; l < c->n_layers && !e; ++l) {
        const glayer_t *y = &d->c.layers[l];
        const gmat_t *ms[GD_MATS] = {&y->q, &y->k, &y->v, &y->o, &y->gate, &y->up, &y->ple_gate, &y->ple_proj,
                                     &y->down[0], &y->down[1], &y->down[2], &y->down[3]};
        for (int i = 0; i < GD_MATS && !e; ++i) {
            const int need = i == 1 || i == 2 ? y->own_kv : i == 6 || i == 7 ? P > 0 : i >= 8 ? i - 8 < y->n_down : 1;
            if (!need) continue;
            if (ms[i]->shift & HMX4_T8) {   // a registered tile8 matrix (geode_npu_mat_t8 holds its mapping)
                const t8_t *t = geode_npu_t8_get(ms[i]->fd);
                if (!t || (int)t->n != ms[i]->n || (int)t->k > ms[i]->k) e = AEE_EBADPARM;
                else d->t8[l][i] = t, d->w[l][i] = t->w;
            } else if (!(d->w[l][i] = map_mat(d, ms[i]))) {
                e = AEE_EBADPARM;
            }
        }
    }
    // the pool, waiting (futex) between phases
    const int prio = helper_prio();
    for (int t = 0; t < GDEC_THREADS && !e; ++t) {
        if (!(d->stack[t] = memalign(128, GD_STACK))) {
            e = AEE_ENOMEMORY;
            break;
        }
        qurt_thread_attr_t at;
        qurt_thread_attr_init(&at);
        qurt_thread_attr_set_name(&at, (char *)"geode_gdec");
        qurt_thread_attr_set_stack_addr(&at, d->stack[t]);
        qurt_thread_attr_set_stack_size(&at, GD_STACK);
        qurt_thread_attr_set_priority(&at, (unsigned short)prio);
        g_gd_args[t].d = d;
        g_gd_args[t].th = t;
        if (qurt_thread_create(&d->tid[t], &at, gd_worker, &g_gd_args[t]) != QURT_EOK) {
            d->tid[t] = 0;
            e = AEE_ENOTALLOWED;
            break;
        }
        ++d->nthreads;
    }
    if (e) {
        gd_free(d);
        return e;
    }
    g_gd = d;
    return 0;
}

int geode_npu_gdec_kv(remote_handle64 h, int chunk, int fd) {
    (void)h;
    gd_t *d = g_gd;
    if (!d || chunk < 0 || chunk >= ATTN_MAX_CHUNKS) return AEE_EBADPARM;
    if (d->kv[chunk]) {
        if (d->kv_fd[chunk] == fd) return 0;
        HAP_mmap_put(d->kv_fd[chunk]);
        d->kv[chunk] = NULL;
    }
    if (!(d->kv[chunk] = (uint8_t *)map_fd(fd))) return AEE_EBADPARM;
    d->kv_fd[chunk] = fd;
    return 0;
}

static int product(gd_t *d, remote_handle64 h, int mat, const gmat_t *m, uint64_t a, uint64_t o) {
    return geode_npu_gemm_src(h, d->IO + a, d->w[d->layer][mat], d->t8[d->layer][mat], d->IO + o, d->m, m->k, m->n,
                              m->shift);
}

int geode_npu_gdec_prefill(remote_handle64 h, int n, int apos, int flags, int64_t *usec) {
    gd_t *d = g_gd;
    *usec = 0;
    if (!d) return AEE_EBADPARM;
    const gdec_cfg_t *c = &d->c;
    if (n < 1 || n > GDEC_MAX_N || apos < 0) return AEE_EBADPARM;
    for (int ch = 0; ch <= (apos + n - 1) / c->kv_chunk; ++ch)
        if (ch >= ATTN_MAX_CHUNKS || !d->kv[ch]) return AEE_EBADPARM;
    uint8_t *vtcm = NULL;
    unsigned vsize = 0;
    int e = geode_npu_vtcm(h, &vtcm, &vsize);
    if (e) return e;
    const uint64_t t0 = HAP_perf_get_time_us();
    d->n = n, d->m = (n + 31) & ~31, d->apos = apos, d->failed = 0;
    {
        const int tk = GDEC_TASK_OF(flags);
        d->task = tk && 16 % tk == 0 ? tk : GD_TASK;
        d->pf = GDEC_PF_OF(flags) ? GDEC_PF_OF(flags) : GD_PF;
        d->main_joins = !(flags & GDEC_NO_MAIN);
        d->old_mask = GDEC_OLD_OF(flags);
    }
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    d->prof[GDEC_PROF_CACHE] += HAP_perf_get_time_us() - t0;
    const int layers = GDEC_DEBUG_LAYERS(flags) ? GDEC_DEBUG_LAYERS(flags) : c->n_layers;
    const gdec_layout_t *L = &d->L;
    for (int l = 0; l < layers && l < c->n_layers && !e; ++l) {
        const glayer_t *y = &c->layers[l];
        d->ly = y, d->layer = l;
        gd_run(d, GP_IN, GDEC_PROF_IN);
        uint64_t tp = HAP_perf_get_time_us();
        if (!e) e = product(d, h, 0, &y->q, L->A, L->OQ);
        if (!e && y->own_kv) e = product(d, h, 1, &y->k, L->A, L->OK);
        if (!e && y->own_kv) e = product(d, h, 2, &y->v, L->A, L->OV);
        d->prof[GDEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - tp;
        d->prof[GDEC_PROF_P_QKV] += HAP_perf_get_time_us() - tp;
        if (e) break;
        gd_run(d, GP_QKV, GDEC_PROF_QKV);
        // the attention (the keys and values written above: in the DSP's cache, which it reads through)
        attn_desc_t *ad = &d->ad;
        memset(ad, 0, sizeof *ad);
        ad->version = ATTN_VERSION;
        ad->n = n, ad->apos = apos, ad->nh = c->nh, ad->nkv = c->nkv, ad->hd = y->hd;
        ad->window = y->full ? 0 : c->window, ad->slot = y->slot, ad->n_attn = c->n_attn, ad->kv_chunk = c->kv_chunk;
        ad->kv_row = c->kv_row, ad->kv_heads = c->kv_heads, ad->n_chunks = (apos + n + c->kv_chunk - 1) / c->kv_chunk;
        ad->q_stride = ad->o_stride = y->q.n;
        ad->flags = flags & GDEC_ATTN_IEEE ? (int32_t)ATTN_SOFTMAX_IEEE : 0;
        if ((e = attn_check(ad))) break;
        if ((e = attn_prepare(&d->at, ad, (const float *)(d->IO + L->QR), (float *)(d->IO + L->CR), d->kv, vtcm, vsize,
                              d->nthreads)))
            break;
        gd_run(d, GP_ATTN, GDEC_PROF_ATTN);
        if (d->at.failed) e = AEE_EFAILED;
        if (e) break;
        gd_run(d, GP_CORE, GDEC_PROF_CORE);
        tp = HAP_perf_get_time_us();
        e = product(d, h, 3, &y->o, L->A, L->OO);
        d->prof[GDEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - tp;
        d->prof[GDEC_PROF_P_O] += HAP_perf_get_time_us() - tp;
        if (e) break;
        gd_run(d, GP_POST, GDEC_PROF_POST);
        tp = HAP_perf_get_time_us();
        const int pc = y->gate.n / y->n_down;
        e = geode_npu_glu_src(h, d->IO + L->A, d->w[l][4], d->w[l][5], d->t8[l][4], d->t8[l][5], d->IO + L->MID, d->m,
                              y->gate.k, y->gate.n, y->gate.shift, y->up.shift, pc / 32, 0);
        d->prof[GDEC_PROF_GLU] += HAP_perf_get_time_us() - tp;
        for (int j = 0; j < y->n_down && !e; ++j)
            e = geode_npu_gemm_src(h, d->IO + L->MID + (size_t)j * d->m * (size_t)pc * 2u, d->w[l][8 + j],
                                   d->t8[l][8 + j], d->IO + L->D + (size_t)j * GDEC_MAX_N * (size_t)c->hidden * 2u, d->m,
                                   pc, c->hidden, y->down[j].shift);
        d->prof[GDEC_PROF_MLP] += HAP_perf_get_time_us() - tp;
        if (e) break;
        gd_run(d, GP_DOWN, GDEC_PROF_DOWN);
        if (!c->ple_dim) continue;
        tp = HAP_perf_get_time_us();
        e = product(d, h, 6, &y->ple_gate, L->A, L->G);
        d->prof[GDEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - tp;
        d->prof[GDEC_PROF_P_PLE] += HAP_perf_get_time_us() - tp;
        if (e) break;
        gd_run(d, GP_PLE, GDEC_PROF_PLE);
        tp = HAP_perf_get_time_us();
        e = product(d, h, 7, &y->ple_proj, L->A, L->OO);
        d->prof[GDEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - tp;
        d->prof[GDEC_PROF_P_PLE] += HAP_perf_get_time_us() - tp;
        if (e) break;
        gd_run(d, GP_PLE2, GDEC_PROF_PLE2);
    }
    if (!e && d->failed) e = AEE_EFAILED;
    const uint64_t tf = HAP_perf_get_time_us();
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);
    d->prof[GDEC_PROF_CACHE] += HAP_perf_get_time_us() - tf;
    ++d->prof[GDEC_PROF_CALLS];
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    return e;
}

int geode_npu_gdec_close(remote_handle64 h) {
    (void)h;
    if (!g_gd) return 0;
    gd_free(g_gd);
    g_gd = NULL;
    return 0;
}

// =====================================================================================================================
// ---- YuE2's NAR on the DSP (npu/nar_cfg.h): one velocity through every layer ---------------------------------------
//
// Per layer: the prefix's last A % 32 positions' blocks into the rows' bank; per slab of rows (NAR_SLAB) the input
// norm into the products' tiles, the fused q k v on HMX, the queries normed and turned into QR (every row's: the
// scale log2 e / sqrt(hd) is in q_norm), the keys normed and turned and the values into the rows' bank; then the
// attention for every row at once (ATTN_ENCODER over the prefix's banks and the rows', ATTN_LOG2); per slab the
// attention's rows (CR) into tiles, o on HMX, x += o and its norm into tiles, the MLP on HMX (gate, up and SiLU into
// the middle, down), x += down. The steps between the products as gdec's: tasks of row pairs of the slab (rows
// s0 + 2p, s0 + 2p + 1: vector p % 16 of every crouton of the slab's row tile p / 16), a pool waiting on a futex.

#define ND_STACK 16384
#define ND_TASK 2   // row pairs a task

enum { NP_IN, NP_QKV, NP_ATTN, NP_CORE, NP_POST, NP_DOWN };

typedef struct {
    nar_cfg_t c;
    nar_layout_t L;
    uint8_t *P, *IO, *PQ;
    const uint8_t *w[NAR_MAX_LAYERS];
    uint8_t *kv[NAR_MAX_LAYERS][2];   // the prefix's banks (keys, values)
    uint8_t *tk, *tv;                 // the rows'
    uint8_t *RB;                      // the A % 32 positions' blocks, per layer
    int mfd[3 * NAR_MAX_LAYERS + 4], n_mfd;   // the fds mapped (each get has its put at close)
    int nb;                           // the rows' bank's blocks a KV head
    // the call
    int s0, ns;                       // the slab's first row, its rows
    int xflags;                       // NAR_X_* (experiments)
    const nlayer_t *ly;
    int layer, phase;
    attn_desc_t ad;
    at_t at;
    uint64_t *prof;
    // the pool
    qurt_thread_t tid[NAR_THREADS];
    void *stack[NAR_THREADS];
    int nthreads;
    volatile int go, quit, running, failed;
    int ctr;
} nd_t;

static nd_t *g_nd;

static inline float *nd_x(nd_t *d, int r) { return (float *)(d->IO + d->c.io_x) + (size_t)r * (size_t)d->c.hidden; }
static inline const HVX_Vector *nd_pq(const nd_t *d, uint32_t at) { return (const HVX_Vector *)(d->PQ + at); }
static inline float *nd_tmp(nd_t *d, int th, int s) {
    const int H = d->c.hidden, qd = d->c.nh * d->c.hd, tw = H > qd ? H : qd;
    return (float *)(d->IO + d->L.TMP) + ((size_t)th * 2u + (size_t)s) * (size_t)tw;
}

// two rows (nv vectors of floats each; NULL: past the slab, zeros) into row pair p of the tiles of a product of ncb
// croutons: normed (RMSNorm, wq its weights in qf32 - as tiles_k) or as they are (wq NULL)
static void nd_tiles(nd_t *d, uint8_t *A, int ncb, int p, const float *r0, const float *r1, int nv,
                     const HVX_Vector *wq) {
    const HVX_Vector *x0 = (const HVX_Vector *)r0, *x1 = (const HVX_Vector *)r1;
    const HVX_Vector z = Q6_V_vzero();
    if (wq) {
        const float n = (float)(nv * 32);
        const HVX_Vector g0 = f_sp(x0 ? 1.0f / sqrtf(sumsq(x0, nv) / n + d->c.eps) : 0.0f);
        const HVX_Vector g1 = f_sp(x1 ? 1.0f / sqrtf(sumsq(x1, nv) / n + d->c.eps) : 0.0f);
        for (int cb = 0; cb < nv; ++cb) {
            const HVX_Vector y0 = x0 ? q_mul(q_muls(x0[cb], g0), wq[cb]) : z;
            const HVX_Vector y1 = x1 ? q_mul(q_muls(x1[cb], g1), wq[cb]) : z;
            *cvec(A, ncb, p, cb) = hf_of_q(y0, y1);
        }
    } else {
        for (int cb = 0; cb < nv; ++cb) *cvec(A, ncb, p, cb) = hf_of(x0 ? x0[cb] : z, x1 ? x1[cb] : z);
    }
    for (int cb = nv; cb < ncb; ++cb) *cvec(A, ncb, p, cb) = z;
}

// positions P0 and P0 + 1 of the rows' bank (r1 NULL: P0 alone), every KV head: the keys (value 0) in the bank's key
// layout ((p, c) at (c / 32) 1024 + ((c % 32) / 2) 64 + (p % 32) 2 + c % 2 halves of the position's block), or the
// values ((c, p) at ((p % 32) / 2) 64 + (c % 32) 2 + p % 2 of dim group c / 32's block: an even P0's pair one vector)
static const int32_t g_odd_words[32] __attribute__((aligned(128))) = {
    0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1};

static void nd_put(nd_t *d, uint8_t *tk, uint8_t *tv, size_t nb, int P0, const float *r0, const float *r1, int value) {
    const int hd = d->c.hd, dgs = hd / 32, nkv = d->c.nkv;
    uint16_t h[64] __attribute__((aligned(128)));
    const HVX_Vector z = Q6_V_vzero();
    const HVX_VectorPred odd = Q6_Q_vand_VR(*(const HVX_Vector *)g_odd_words, 0x01010101);
    // the odd halfwords' bytes (an odd P0's masked stores)
    const HVX_VectorPred odd_h = Q6_Q_vand_VR(Q6_V_vsplat_R((int)0xFFFF0000u), 0x01010101);
    for (int kh = 0; kh < nkv; ++kh) {
        const HVX_Vector *a = (const HVX_Vector *)(r0 + (size_t)kh * hd);
        const HVX_Vector *b = r1 ? (const HVX_Vector *)(r1 + (size_t)kh * hd) : NULL;
        for (int dg = 0; dg < dgs; ++dg) {
            // (P0 + 1 past the keys when r1 is NULL: masked, its zeros harmless)
            const HVX_Vector av = a[dg], bv = b ? b[dg] : z;
            if (!(P0 & 1)) {
                if (value) {   // lane 2i: P0's dim i, 2i + 1: P0 + 1's - the pair's vector of the block
                    uint16_t *blk = (uint16_t *)tv + (((size_t)kh * dgs + (size_t)dg) * nb + (size_t)(P0 / 32)) * 1024u;
                    *(HVX_Vector *)(blk + (size_t)((P0 % 32) / 2) * 64u) = hf_of(av, bv);
                } else {   // per dim pair j the 8 bytes P0's dims 2j, 2j + 1, then P0 + 1's, at line j of the block
                    const HVX_Vector x = Q6_V_vmux_QVV(odd, Q6_V_vror_VR(bv, 124), av);   // a[2m], b[2m]
                    const HVX_Vector y = Q6_V_vmux_QVV(odd, bv, Q6_V_vror_VR(av, 4));     // a[2m + 1], b[2m + 1]
                    *(HVX_Vector *)h = hf_of(x, y);
                    const uint64_t *src = (const uint64_t *)h;
                    uint64_t *dst = (uint64_t *)((uint16_t *)tk + (((size_t)kh * nb + (size_t)(P0 / 32)) * dgs + (size_t)dg) * 1024u +
                                                 (size_t)(P0 % 32) * 2u);
                    for (int j = 0; j < 16; ++j) dst[(size_t)j * 16u] = src[j];
                }
                continue;
            }
            if (!(d->xflags & NAR_X_OLDPUT)) {
                // an odd P0 (the prefix's A % 32 odd): the pair straddles two pairs of the value layout and is not
                // 8-byte aligned in the key layout - the same halves stored by two masked vector stores (values) and
                // 32-bit stores (keys) instead of 16-bit ones one at a time (~4000 a row pair, most of the phase)
                const int P1 = P0 + 1;
                if (value) {   // P0's dims to the odd lanes of its pair's vector, P0 + 1's to the even lanes of the next
                    const HVX_Vector v = hf_of(av, bv);   // lane 2i: P0's dim i, 2i + 1: P0 + 1's
                    const size_t g0 = ((size_t)kh * dgs + (size_t)dg) * nb;
                    uint16_t *o0 = (uint16_t *)tv + (g0 + (size_t)(P0 / 32)) * 1024u + (size_t)((P0 % 32) / 2) * 64u;
                    Q6_vmem_QRIV(odd_h, (HVX_Vector *)o0, Q6_V_vror_VR(v, 126));
                    if (b) {
                        uint16_t *o1 = (uint16_t *)tv + (g0 + (size_t)(P1 / 32)) * 1024u + (size_t)((P1 % 32) / 2) * 64u;
                        Q6_vmem_QRIV(Q6_Q_not_Q(odd_h), (HVX_Vector *)o1, Q6_V_vror_VR(v, 2));
                    }
                } else {   // the even path's 8 bytes a dim pair j: P0's 4 to its line, P0 + 1's 4 to its own
                    const HVX_Vector x = Q6_V_vmux_QVV(odd, Q6_V_vror_VR(bv, 124), av);
                    const HVX_Vector y = Q6_V_vmux_QVV(odd, bv, Q6_V_vror_VR(av, 4));
                    *(HVX_Vector *)h = hf_of(x, y);
                    const uint32_t *src = (const uint32_t *)h;
                    uint32_t *k0 = (uint32_t *)((uint16_t *)tk + (((size_t)kh * nb + (size_t)(P0 / 32)) * dgs + (size_t)dg) * 1024u +
                                                (size_t)(P0 % 32) * 2u);
                    uint32_t *k1 = (uint32_t *)((uint16_t *)tk + (((size_t)kh * nb + (size_t)(P1 / 32)) * dgs + (size_t)dg) * 1024u +
                                                (size_t)(P1 % 32) * 2u);
                    for (int j = 0; j < 16; ++j) {
                        k0[(size_t)j * 32u] = src[2 * j];
                        if (b) k1[(size_t)j * 32u] = src[2 * j + 1];
                    }
                }
                continue;
            }
            const HVX_Vector v = hf_of(av, bv);   // lane 2i: P0's dim i, 2i + 1: P0 + 1's
            *(HVX_Vector *)h = v;
            for (int s = 0; s < (b ? 2 : 1); ++s) {
                const int pos = P0 + s;
                if (value) {
                    uint16_t *o = (uint16_t *)tv + (((size_t)kh * dgs + (size_t)dg) * nb + (size_t)(pos / 32)) * 1024u +
                                  (size_t)((pos % 32) / 2) * 64u + (size_t)(pos & 1);
                    for (int i = 0; i < 32; ++i) o[2 * i] = h[2 * i + s];
                } else {
                    uint16_t *o = (uint16_t *)tk + (((size_t)kh * nb + (size_t)(pos / 32)) * dgs + (size_t)dg) * 1024u +
                                  (size_t)(pos % 32) * 2u;
                    for (int i = 0; i < 32; ++i) o[(i / 2) * 64 + (i & 1)] = h[2 * i + s];
                }
            }
        }
    }
}

// where row r's key and value go: a velocity's rows after the prefix's rem positions in the rows' bank; a prefix's
// row r is position r - its layer's banks (the whole blocks), the rest RB's block (nb 1)
static void nd_target(nd_t *d, int r, uint8_t **K, uint8_t **V, size_t *nb, int *P0) {
    const nar_cfg_t *c = &d->c;
    if (c->mode != NAR_MODE_PREFIX) {
        *K = d->tk, *V = d->tv, *nb = (size_t)d->nb, *P0 = c->rem + r;
    } else if (r < c->npre * 32) {
        *K = d->kv[d->layer][0], *V = d->kv[d->layer][1], *nb = (size_t)c->npre, *P0 = r;
    } else {
        *K = d->RB + d->ly->rem, *V = *K + (size_t)c->nkv * (size_t)(c->hd / 32) * 2048u, *nb = 1, *P0 = r - c->npre * 32;
    }
}

static void np_in(nd_t *d, int p, int th) {
    (void)th;
    const int H = d->c.hidden, r = d->s0 + 2 * p;
    nd_tiles(d, d->IO + d->L.A, H / 32, p, 2 * p < d->ns ? nd_x(d, r) : NULL,
             2 * p + 1 < d->ns ? nd_x(d, r + 1) : NULL, H / 32, nd_pq(d, d->ly->in_norm));
}

// the queries normed and turned into QR; the keys normed and turned, and the values, into the rows' bank
static void np_qkv(nd_t *d, int p, int th) {
    const nar_cfg_t *c = &d->c;
    const nlayer_t *y = d->ly;
    const int hd = c->hd, half = hd / 2, qd = c->nh * hd, kd = c->nkv * hd, ncb = (qd + 2 * kd) / 32;
    const int r0 = d->s0 + 2 * p, ok0 = 2 * p < d->ns, ok1 = 2 * p + 1 < d->ns;
    uint8_t *O = d->IO + d->L.QKV;
    float *q0 = (float *)(d->IO + d->L.QR) + (size_t)r0 * (size_t)qd, *q1 = q0 + qd;   // (QR holds L.m rows)
    float *k0 = nd_tmp(d, th, 0), *k1 = nd_tmp(d, th, 1);
    const float *rope = (const float *)(d->IO + c->io_rope);
    rows_of(O, ncb, p, 0, qd / 32, q0, q1);
    for (int value = 0; value < 2; ++value) {
        rows_of(O, ncb, p, (qd + value * kd) / 32, kd / 32, k0, k1);
        if (!value && !(d->xflags & NAR_X_NONORM))
            for (int s = 0; s < 2; ++s) {
                if (!(s ? ok1 : ok0)) continue;
                const float *rt = rope + (size_t)(r0 + s) * (size_t)hd;   // its cosines, then its sines
                float *q = s ? q1 : q0, *k = s ? k1 : k0;
                if (d->xflags & NAR_X_OLDNORM) {   // (each head's norm through a scalar square root, as before)
                    for (int h = 0; h < c->nh; ++h) norm_v(q + (size_t)h * hd, hd, nd_pq(d, y->q_norm), c->eps);
                    for (int kh = 0; kh < c->nkv; ++kh) norm_v(k + (size_t)kh * hd, hd, nd_pq(d, y->k_norm), c->eps);
                } else {
                    norm_heads(q, c->nh, hd, nd_pq(d, y->q_norm), c->eps);
                    norm_heads(k, c->nkv, hd, nd_pq(d, y->k_norm), c->eps);
                }
                for (int h = 0; h < c->nh; ++h) rope_v(q + (size_t)h * hd, hd, rt, rt + half, half);
                for (int kh = 0; kh < c->nkv; ++kh) rope_v(k + (size_t)kh * hd, hd, rt, rt + half, half);
            }
        if (ok0 && (value || !(d->xflags & NAR_X_NOPUTK))) {
            uint8_t *K, *V;
            size_t nb;
            int P0;
            nd_target(d, r0, &K, &V, &nb, &P0);   // (r0 even: a pair never straddles a block)
            nd_put(d, K, V, nb, P0, k0, ok1 ? k1 : NULL, value);
        }
    }
}

static void np_core(nd_t *d, int p, int th) {
    (void)th;
    const int qd = d->c.nh * d->c.hd, r = d->s0 + 2 * p;
    const float *CR = (const float *)(d->IO + d->L.CR);
    nd_tiles(d, d->IO + d->L.A, qd / 32, p, 2 * p < d->ns ? CR + (size_t)r * (size_t)qd : NULL,
             2 * p + 1 < d->ns ? CR + (size_t)(r + 1) * (size_t)qd : NULL, qd / 32, NULL);
}

// x's rows += a product's result croutons of pair p (hidden / 32 a row tile); NULL rows left out
static void nd_add(nd_t *d, int p, uint8_t *buf, float *x0, float *x1) {
    const int nv = d->c.hidden / 32;
    HVX_Vector *a = (HVX_Vector *)x0, *b = (HVX_Vector *)x1;
    for (int cb = 0; cb < nv; ++cb) {
        HVX_Vector e, o;
        q_of_hf(*cvec(buf, nv, p, cb), &e, &o);
        if (a) a[cb] = sf_q(Q6_Vqf32_vadd_VsfVsf(a[cb], sf_q(e)));
        if (b) b[cb] = sf_q(Q6_Vqf32_vadd_VsfVsf(b[cb], sf_q(o)));
    }
}

static void np_post(nd_t *d, int p, int th) {
    (void)th;
    const int H = d->c.hidden, r = d->s0 + 2 * p;
    float *x0 = 2 * p < d->ns ? nd_x(d, r) : NULL, *x1 = 2 * p + 1 < d->ns ? nd_x(d, r + 1) : NULL;
    nd_add(d, p, d->IO + d->L.OO, x0, x1);
    nd_tiles(d, d->IO + d->L.A, H / 32, p, x0, x1, H / 32, nd_pq(d, d->ly->mlp_norm));
}

static void np_down(nd_t *d, int p, int th) {
    (void)th;
    const int r = d->s0 + 2 * p;
    nd_add(d, p, d->IO + d->L.D, 2 * p < d->ns ? nd_x(d, r) : NULL, 2 * p + 1 < d->ns ? nd_x(d, r + 1) : NULL);
}

static void nd_phase(nd_t *d, int th) {
    if (d->phase == NP_ATTN) {
        program(&d->at, th);
        return;
    }
    static void (*const fns[])(nd_t *, int, int) = {np_in, np_qkv, NULL, np_core, np_post, np_down};
    void (*fn)(nd_t *, int, int) = fns[d->phase];
    const int pairs = ((d->ns + 31) & ~31) / 2, tasks = (pairs + ND_TASK - 1) / ND_TASK;
    for (int t = __atomic_fetch_add(&d->ctr, 1, __ATOMIC_RELAXED); t < tasks;
         t = __atomic_fetch_add(&d->ctr, 1, __ATOMIC_RELAXED)) {
        const int p1 = (t + 1) * ND_TASK < pairs ? (t + 1) * ND_TASK : pairs;
        for (int q = t * ND_TASK; q < p1; ++q) fn(d, q, th);
    }
}

typedef struct {
    nd_t *d;
    int th;
} nd_arg_t;
static nd_arg_t g_nd_args[NAR_THREADS];

static void nd_worker(void *arg) {
    nd_arg_t *x = (nd_arg_t *)arg;
    nd_t *d = x->d;
    int seen = 0;
    for (;;) {
        int go;
        while ((go = __atomic_load_n(&d->go, __ATOMIC_ACQUIRE)) == seen) qurt_futex_wait((void *)&d->go, seen);
        seen = go;
        if (__atomic_load_n(&d->quit, __ATOMIC_ACQUIRE)) break;
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
            d->failed = 1;
        } else {
            nd_phase(d, x->th);
            qurt_hvx_unlock();
        }
        __atomic_add_fetch(&d->running, -1, __ATOMIC_ACQ_REL);
    }
    qurt_thread_exit(0);
}

// the pool through one phase, this thread too (the attention's: this one on HMX)
static void nd_run(nd_t *d, int phase, int prof) {
    const uint64_t t0 = HAP_perf_get_time_us();
    d->phase = phase;
    d->ctr = 0;
    __atomic_store_n(&d->running, d->nthreads, __ATOMIC_RELEASE);
    __atomic_add_fetch(&d->go, 1, __ATOMIC_ACQ_REL);
    qurt_futex_wake((void *)&d->go, NAR_THREADS);
    if (phase == NP_ATTN) program(&d->at, d->nthreads);
    else nd_phase(d, NAR_THREADS);   // (its own TMP rows: nar_layout's)
    while (__atomic_load_n(&d->running, __ATOMIC_ACQUIRE) > 0) asm volatile("pause(#16)");
    d->prof[prof] += HAP_perf_get_time_us() - t0;
}

static void nd_free(nd_t *d) {
    __atomic_store_n(&d->quit, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&d->go, 1, __ATOMIC_ACQ_REL);
    qurt_futex_wake((void *)&d->go, NAR_THREADS);
    for (int t = 0; t < d->nthreads; ++t) {
        int st;
        if (d->tid[t]) qurt_thread_join(d->tid[t], &st);
    }
    for (int t = 0; t < NAR_THREADS; ++t) free(d->stack[t]);
    free(d->PQ);
    for (int i = 0; i < d->n_mfd; ++i) HAP_mmap_put(d->mfd[i]);
    free(d);
}

static uint8_t *nd_map(nd_t *d, int fd) {
    if (fd <= 0) return NULL;
    uint8_t *v = (uint8_t *)map_fd(fd);
    if (v) d->mfd[d->n_mfd++] = fd;
    return v;
}

int geode_npu_nar_open(remote_handle64 h, const unsigned char *cfg, int cfg_len, int p_fd, int io_fd) {
    (void)h;
    if (g_nd) return AEE_EALREADY;
    if (cfg_len != (int)sizeof(nar_cfg_t)) return AEE_EBADPARM;
    if (g_orders_ok < 0) a_checks();
    if (!g_orders_ok) return AEE_EUNSUPPORTED;
    const nar_cfg_t *c = (const nar_cfg_t *)cfg;
    const int H = c->hidden, hd = c->hd, qd = c->nh * c->hd, F = c->ffn, pre = c->mode == NAR_MODE_PREFIX;
    const int P = pre ? c->n : c->A;   // the prefix's positions
    if (c->version != NAR_VERSION || c->n_layers < 1 || c->n_layers > NAR_MAX_LAYERS || H % 64 || H < 512 ||
        H > 8192 || F % 32 || F < 512 || hd % 64 || hd > 512 || c->nkv < 1 || c->nh % c->nkv ||
        32 % (c->nh / c->nkv) || qd < 512 || qd > 8192 || c->n < 1 || c->A < 0 || c->rem != P % 32 ||
        c->npre != P / 32 || c->A + c->n > 30000 || c->p_norms % 128 || (c->rem && c->rb_fd <= 0) ||
        (c->mode != NAR_MODE_VELOCITY && c->mode != NAR_MODE_PREFIX) || (pre && c->A != 0) ||
        (!pre && (c->tk_fd <= 0 || c->tv_fd <= 0)))
        return AEE_EBADPARM;
    for (int l = 0; l < c->n_layers; ++l) {
        const nlayer_t *y = &c->layers[l];
        const uint32_t hb = (uint32_t)H * 4u, db = (uint32_t)hd * 4u;
        if (y->w_fd <= 0 || ((y->qkv | y->o | y->gate | y->up | y->down) & 127) || y->in_norm + hb > c->p_norms ||
            y->mlp_norm + hb > c->p_norms || y->q_norm + db > c->p_norms || y->k_norm + db > c->p_norms ||
            ((y->in_norm | y->mlp_norm | y->q_norm | y->k_norm | y->rem) & 127) || (c->npre && (y->pk_fd <= 0 || y->pv_fd <= 0)))
            return AEE_EBADPARM;
        const int sh[4] = {y->sh_qkv, y->sh_o, y->sh_gu, y->sh_down};
        for (int i = 0; i < 4; ++i)
            if (sh[i] < 0 || sh[i] > 24) return AEE_EBADPARM;
    }
    nd_t *d = (nd_t *)calloc(1, sizeof *d);
    if (!d) return AEE_ENOMEMORY;
    memcpy(&d->c, c, sizeof *c);
    d->L = nar_layout(&d->c);
    d->nb = (c->rem + c->n + 31) / 32;
    int e = 0;
    if (d->L.end > c->io_bytes || !(d->P = nd_map(d, p_fd)) || !(d->IO = nd_map(d, io_fd)) ||
        (!pre && (!(d->tk = nd_map(d, c->tk_fd)) || !(d->tv = nd_map(d, c->tv_fd)))) ||
        (c->rem && !(d->RB = nd_map(d, c->rb_fd))))
        e = AEE_EBADPARM;
    for (int l = 0; l < c->n_layers && !e; ++l) {
        const nlayer_t *y = &c->layers[l];
        if (!(d->w[l] = nd_map(d, y->w_fd))) e = AEE_EBADPARM;
        else if (c->npre && (!(d->kv[l][0] = nd_map(d, y->pk_fd)) || !(d->kv[l][1] = nd_map(d, y->pv_fd))))
            e = AEE_EBADPARM;
    }
    if (!e) {
        d->prof = (uint64_t *)(d->IO + c->io_bytes - 256u);
        memset(d->prof, 0, 256);
        // the norms' weights in qf32 (the CPU wrote them: seen afresh)
        qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
        if (!(d->PQ = (uint8_t *)memalign(128, c->p_norms ? c->p_norms : 128))) {
            e = AEE_ENOMEMORY;
        } else {
            const HVX_Vector *src = (const HVX_Vector *)d->P;
            HVX_Vector *dst = (HVX_Vector *)d->PQ;
            for (uint32_t i = 0; i < c->p_norms / 128u; ++i) dst[i] = q_sf(src[i]);
        }
    }
    const int prio = helper_prio();
    for (int t = 0; t < NAR_THREADS && !e; ++t) {
        if (!(d->stack[t] = memalign(128, ND_STACK))) {
            e = AEE_ENOMEMORY;
            break;
        }
        qurt_thread_attr_t at;
        qurt_thread_attr_init(&at);
        qurt_thread_attr_set_name(&at, (char *)"geode_nar");
        qurt_thread_attr_set_stack_addr(&at, d->stack[t]);
        qurt_thread_attr_set_stack_size(&at, ND_STACK);
        qurt_thread_attr_set_priority(&at, (unsigned short)prio);
        g_nd_args[t].d = d;
        g_nd_args[t].th = t;
        if (qurt_thread_create(&d->tid[t], &at, nd_worker, &g_nd_args[t]) != QURT_EOK) {
            d->tid[t] = 0;
            e = AEE_ENOTALLOWED;
            break;
        }
        ++d->nthreads;
    }
    if (e) {
        nd_free(d);
        return e;
    }
    g_nd = d;
    return 0;
}

extern uint64_t g_gemm_tq[5];
extern int g_gemm_x;

int geode_npu_nar_run(remote_handle64 h, int flags, int64_t *usec) {
    nd_t *d = g_nd;
    *usec = 0;
    if (!d) return AEE_EBADPARM;
    uint64_t tq0[5];
    memcpy(tq0, g_gemm_tq, sizeof tq0);
    const nar_cfg_t *c = &d->c;
    uint8_t *vtcm = NULL;
    unsigned vsize = 0;
    int e = geode_npu_vtcm(h, &vtcm, &vsize);
    if (e) return e;
    g_gemm_x = (flags & NAR_X_NOOUT) ? 1 : 0;
    const uint64_t t0 = HAP_perf_get_time_us();
    d->failed = 0;
    d->xflags = flags & (NAR_X_NOPUTK | NAR_X_NONORM | NAR_X_OLDNORM | NAR_X_OLDPUT);
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);   // x, the banks: the CPU's
    d->prof[NAR_PROF_CACHE] += HAP_perf_get_time_us() - t0;
    const int layers = NAR_DEBUG_LAYERS(flags) ? NAR_DEBUG_LAYERS(flags) : c->n_layers;
    const nar_layout_t *L = &d->L;
    const int H = c->hidden, hd = c->hd, qd = c->nh * hd, kd = c->nkv * hd, W = qd + 2 * kd, F = c->ffn, dgs = hd / 32;
    const size_t kblk = (size_t)dgs * 2048u;
    for (int l = 0; l < layers && l < c->n_layers && !e; ++l) {
        const nlayer_t *y = &c->layers[l];
        const uint8_t *w = d->w[l];
        const int wf = y->wfmt ? HMX_Q8 : HMX_F16;   // (its matrices' blocks: fp16, or int8 + scales)
        d->ly = y, d->layer = l;
        uint64_t tp = HAP_perf_get_time_us();
        const int pre = c->mode == NAR_MODE_PREFIX;
        if (c->rem && !pre) {   // the prefix's last positions: the first block of each KV head (the rows overwrite the rest)
            const uint8_t *src = d->RB + y->rem;
            for (int kh = 0; kh < c->nkv; ++kh)
                copy_blocks(d->tk + (size_t)kh * (size_t)d->nb * kblk, src + (size_t)kh * kblk, kblk);
            src += (size_t)c->nkv * kblk;
            for (int g = 0; g < c->nkv * dgs; ++g)
                copy_blocks(d->tv + (size_t)g * (size_t)d->nb * 2048u, src + (size_t)g * 2048u, 2048u);
        }
        d->prof[NAR_PROF_KV] += HAP_perf_get_time_us() - tp;
        for (int s0 = 0; s0 < c->n && !e; s0 += NAR_SLAB) {
            d->s0 = s0, d->ns = c->n - s0 < NAR_SLAB ? c->n - s0 : NAR_SLAB;
            const int ms = (d->ns + 31) & ~31;
            nd_run(d, NP_IN, NAR_PROF_IN);
            tp = HAP_perf_get_time_us();
            e = geode_npu_gemm_src(h, d->IO + L->A, w + y->qkv, NULL, d->IO + L->QKV, ms, H, W, y->sh_qkv | wf);
            d->prof[NAR_PROF_QKV] += HAP_perf_get_time_us() - tp;
            if (!e) nd_run(d, NP_QKV, NAR_PROF_KV);
        }
        if (e) break;
        // (the prefix: the model's last layer's keys and values are all it gives; a group's last before more: whole)
        if (pre && l == c->n_layers - 1 && !(flags & NAR_RUN_MORE)) break;
        // the attention: every row over the prefix's A keys and the rows' own (the rows' just written: in the DSP's
        // cache, which it reads through)
        attn_desc_t *ad = &d->ad;
        memset(ad, 0, sizeof *ad);
        ad->version = ATTN_VERSION;
        ad->n = c->n, ad->apos = 0, ad->nh = c->nh, ad->nkv = c->nkv, ad->hd = hd, ad->window = 0;
        ad->n_attn = 1, ad->kv_chunk = 32;
        ad->q_stride = ad->o_stride = qd;
        ad->flags = (int32_t)(ATTN_ENCODER | ATTN_LOG2 | (pre ? ATTN_CAUSAL : 0u) | ((uint32_t)c->attn_flags & ~7u));
        ad->nk = c->A + c->n, ad->npre = c->npre;
        uint8_t *kv[4];
        uint8_t *tk = d->tk, *tv = d->tv;   // the blocks after the whole ones (a prefix: RB's, when it has any)
        if (pre) {
            tk = c->rem ? d->RB + y->rem : d->kv[l][0];
            tv = c->rem ? tk + (size_t)c->nkv * kblk : d->kv[l][1];
        }
        if (c->npre) {
            ad->n_chunks = 4;
            kv[0] = d->kv[l][0], kv[1] = d->kv[l][1], kv[2] = tk, kv[3] = tv;
        } else {
            ad->n_chunks = 2;
            kv[0] = tk, kv[1] = tv;
        }
        if ((e = attn_check(ad))) break;
        if ((e = attn_prepare(&d->at, ad, (const float *)(d->IO + L->QR), (float *)(d->IO + L->CR), kv, vtcm, vsize,
                              d->nthreads)))
            break;
        nd_run(d, NP_ATTN, NAR_PROF_ATTN);
        if (d->at.failed) e = AEE_EFAILED;
        for (int s0 = 0; s0 < c->n && !e; s0 += NAR_SLAB) {
            d->s0 = s0, d->ns = c->n - s0 < NAR_SLAB ? c->n - s0 : NAR_SLAB;
            const int ms = (d->ns + 31) & ~31;
            nd_run(d, NP_CORE, NAR_PROF_CORE);
            tp = HAP_perf_get_time_us();
            e = geode_npu_gemm_src(h, d->IO + L->A, w + y->o, NULL, d->IO + L->OO, ms, qd, H, y->sh_o | wf);
            d->prof[NAR_PROF_O] += HAP_perf_get_time_us() - tp;
            if (e) break;
            nd_run(d, NP_POST, NAR_PROF_POST);
            tp = HAP_perf_get_time_us();
            e = geode_npu_glu_src(h, d->IO + L->A, w + y->gate, w + y->up, NULL, NULL, d->IO + L->MID, ms, H, F,
                                  y->sh_gu | wf, y->sh_gu | wf, F / 32, 1);
            d->prof[NAR_PROF_GLU] += HAP_perf_get_time_us() - tp;
            if (!e)
                e = geode_npu_gemm_src(h, d->IO + L->MID, w + y->down, NULL, d->IO + L->D, ms, F, H,
                                       y->sh_down | wf);
            d->prof[NAR_PROF_MLP] += HAP_perf_get_time_us() - tp;
            if (e) break;
            nd_run(d, NP_DOWN, NAR_PROF_DOWN);
        }
    }
    if (!e && d->failed) e = AEE_EFAILED;
    // (the times live in IO, which the CPU reads after the flush: written before it - the flush's own after it, its
    // line flushed on its own)
    ++d->prof[NAR_PROF_CALLS];
    for (int i = 0; i < 5; ++i)
        d->prof[NAR_PROF_G_ACOPY + i] += HAP_perf_qtimer_count_to_us(g_gemm_tq[i] - tq0[i]);
    g_gemm_x = 0;
    const uint64_t tf = HAP_perf_get_time_us();
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);   // x out to where the CPU reads it
    d->prof[NAR_PROF_CACHE] += HAP_perf_get_time_us() - tf;
    qurt_mem_cache_clean((qurt_addr_t)&d->prof[NAR_PROF_CACHE], sizeof d->prof[0], QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    return e;
}

int geode_npu_nar_close(remote_handle64 h) {
    (void)h;
    if (!g_nd) return 0;
    nd_free(g_nd);
    g_nd = NULL;
    return 0;
}

#else   // v73 / v75: no IEEE single-precision HVX for the softmax and the rest
int geode_npu_attn(remote_handle64 h, const unsigned char *desc, int desc_len, const unsigned char *q, int q_len,
                   unsigned char *o, int o_len, int64_t *usec) {
    (void)h, (void)desc, (void)desc_len, (void)q, (void)q_len, (void)o, (void)o_len;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_gdec_open(remote_handle64 h, const unsigned char *cfg, int cfg_len, int p_fd, int io_fd) {
    (void)h, (void)cfg, (void)cfg_len, (void)p_fd, (void)io_fd;
    return AEE_EUNSUPPORTED;
}
int geode_npu_gdec_kv(remote_handle64 h, int chunk, int fd) {
    (void)h, (void)chunk, (void)fd;
    return AEE_EUNSUPPORTED;
}
int geode_npu_gdec_prefill(remote_handle64 h, int n, int apos, int flags, int64_t *usec) {
    (void)h, (void)n, (void)apos, (void)flags;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_gdec_close(remote_handle64 h) {
    (void)h;
    return 0;
}
int geode_npu_nar_open(remote_handle64 h, const unsigned char *cfg, int cfg_len, int p_fd, int io_fd) {
    (void)h, (void)cfg, (void)cfg_len, (void)p_fd, (void)io_fd;
    return AEE_EUNSUPPORTED;
}
int geode_npu_nar_run(remote_handle64 h, int flags, int64_t *usec) {
    (void)h, (void)flags;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_nar_close(remote_handle64 h) {
    (void)h;
    return 0;
}
#endif
