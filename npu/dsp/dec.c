// npu/dsp/dec.c - a dense Qwen3.5's whole decode step on the cDSP (v79+, npu/dec_cfg.h): every product straight from
// tile8 (q4) or q8 tiles, fetched into VTCM by the DMA engine and multiplied with HVX vrmpy; the Gated DeltaNet, the
// attention and the rest in HVX floats; on a pool of threads that stays between steps. The CPU hands over a token (or
// a token and its MTP draft, to verify both at once) and reads the logits where the step leaves them. The states and
// the KV cache are the engine's own, in memory both sides map: the DSP sees what the CPU wrote (IO coherence), the CPU
// sees the DSP's writes once a step flushes them.
//
// A step is one program every worker thread runs (the caller only starts them: its stack's size is FastRPC's), its
// stages apart by a barrier; inside a stage
// the threads take tasks (a matrix's row tile, a head, 64 values) from a counter of the stage's own. Two programs: the
// model's step (one token, or a verified pair: every weight read once for both) and the MTP layer's pass (a draft).
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "geode_npu.h"
#include "dec_cfg.h"

#if __HVX_ARCH__ >= 79
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "qurt.h"

#define DEC_THREADS 6
#define DEC_STAGES 640
#define DEC_STACK 32768
#define DEC_TILE_MAX (6144u * 20u)   // the widest tile: q4b, k = 6144 (q8 tiles: 34 k bytes, k <= 3072)
#define DEC_VEC_AREA 65536u          // a thread's VTCM for vectors it builds (the GDN's k / q rows, attention's)
#define DEC_THREAD_VTCM (2u * DEC_TILE_MAX + DEC_VEC_AREA)
#define DEC_TBUF 8192                // floats of a thread's cached scratch: attention's scores, up to 32 x DEC_BLOCK
#define ATT_ROWS 32                  // a prompt's attention: (token, query head) rows a task (one KV head's)
#define DEC_MAX_CHUNKS 512

// the npu_imp.c session's VTCM (acquired for it once)
int geode_npu_vtcm(remote_handle64 h, uint8_t **vtcm, unsigned *size);   // npu_imp.c

// the user DMA's 2D descriptor (npu_imp.c's udma0_t)
typedef struct {
    void *next;
    uint32_t dst_stride : 24;
    uint32_t desc_size : 2;
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
} dma_t;

// a vector quantized for the products, the engine's int8 path's way: per 16 values a scale (amax / 127) and int8 codes
typedef struct {
    int8_t *q;          // k codes
    uint32_t *lo, *hi;  // per 8 values: codes 0, 2, 4, 6 / 1, 3, 5, 7 (tile8's nibble order)
    int32_t *bs;        // per 16: the codes' sum
    float *sc;          // per 16: the scale
} act_t;

typedef struct {
    dec_cfg_t c;
    uint8_t *W, *P, *S;
    int w_fd, p_fd, s_fd;
    uint8_t *kv[DEC_MAX_CHUNKS];
    int kv_fd[DEC_MAX_CHUNKS];
    // the scratch (in S), a row per token of the step: what the CPU reads first, then the step's own
    float *logits[DEC_T], *hid[DEC_N];
    float *xa[DEC_N], *xb[DEC_N], *mix[DEC_N], *mlp[DEC_N], *fin[DEC_N], *proj[DEC_N], *zb[DEC_N], *ab[DEC_N],
        *core[DEC_N], *qb[DEC_N], *kb[DEC_N], *vb[DEC_N], *g[DEC_N], *u[DEC_N], *cat[DEC_N];
    uint8_t *tiles_a, *tiles_o[3];   // a prompt's HMX operands: the activations' row tiles, the results' croutons
    uint8_t *hmx[3 * DEC_MAX_LAYERS + 8];   // the matrices' hmx4 copies, mapped (by their place in the cfg)
    int hmx_fds[3 * DEC_MAX_LAYERS + 8], n_hmx;
    // a prompt call's phases: which one the workers run, the layer, the residual rows
    int phase;
    const dec_layer_t *pl;
    float **rx, **rxo, **radd;
    int backup;                  // a verified pair: the first token's states into the other set
    float *part, *dlog;
    act_t act_h[DEC_T], act_c[DEC_T], act_m[DEC_T], act_w[DEC_T];   // normed input, mixer out, MLP middle, MTP input
    uint64_t *prof;              // DEC_PROF_* (thread 0 times the stages)
    uint64_t t_mark;
    // the pool
    int nthreads;
    qurt_thread_t tid[DEC_THREADS];
    void *stack[DEC_THREADS];
    volatile int go, quit, running;
    int arrive;
    volatile int sense;
    int ctr[DEC_STAGES];
    uint8_t *vtcm;
    dma_t *dma[DEC_THREADS];
    float *tbuf[DEC_THREADS];   // a thread's cached scratch (DEC_TBUF floats)
    int tree_ok;                 // tree32's lane order known (tree_check)
    int cvt_to, cvt_from;        // vcvt's lane orders between sf and hf (cvt_check; 0: not yet, -1: neither known)
    uint8_t tree_perm[32];
    // a thread's last token's rope cosines then sines (rope_cs), and its three positions + 1 once set: a buffer of its
    // own (6 KB more inside dec_t made the prompt's HMX products ~3x slower)
    float *rope;
    int *rope_key;
    volatile int failed;
    // the call's arguments and results
    int prog;                    // 0: the model's step, 1: the MTP pass
    int T, pos, p3[DEC_N][3], flags;
    float *conv_live, *gdn_live, *conv_other, *gdn_other;   // the live state set; the other: a pair's first token's
    float best[DEC_THREADS], bsum[DEC_THREADS];
    int best_at[DEC_THREADS];
    int draft;
    float dprob;
} dec_t;

static dec_t *g_dec;   // one decoder a process

// ---- small things ---------------------------------------------------------------------------------------------------

static inline float bf16f(uint16_t v) {
    union { uint32_t i; float f; } u = {(uint32_t)v << 16};
    return u.f;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x) { return x / (1.0f + expf(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static inline const float *pf(const dec_t *d, uint32_t at) { return (const float *)(d->P + at); }

static inline int take(dec_t *d, int st) { return __atomic_fetch_add(&d->ctr[st], 1, __ATOMIC_RELAXED); }

// thread 0, after a barrier: the time since the last mark to stage `what`
static inline void mark(dec_t *d, int th, int what) {
    if (th) return;
    const uint64_t now = HAP_perf_get_time_us();
    d->prof[what] += now - d->t_mark;
    d->t_mark = now;
}

static void bar(dec_t *d, int *sense) {
    const int s = *sense ^= 1;
    if (__atomic_add_fetch(&d->arrive, 1, __ATOMIC_ACQ_REL) == d->nthreads) {
        __atomic_store_n(&d->arrive, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&d->sense, s, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&d->sense, __ATOMIC_ACQUIRE) != s) asm volatile("pause(#4)");
    }
}

// ---- HVX floats ----------------------------------------------------------------------------------------------------

static inline HVX_Vector vsf(float f) {
    union { float f; int32_t i; } u = {f};
    return Q6_V_vsplat_R(u.i);
}
static inline HVX_Vector vmul(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vmpy_VsfVsf(a, b); }
static inline HVX_Vector vadd(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vadd_VsfVsf(a, b); }
static inline HVX_Vector vsub(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vsub_VsfVsf(a, b); }

// exp(x) lane by lane: 2^(x log2 e) = 2^n 2^f, |f| < 1, 2^f = e^(f ln 2) by its Taylor series to the 7th power
// (relative error < 1e-6); x clamped to where 2^n stays a normal float
static inline HVX_Vector vexp(HVX_Vector x) {
    HVX_Vector y = vmul(x, vsf(1.44269504f));
    y = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(y, vsf(-125.0f)), vsf(125.0f));
    const HVX_Vector n = Q6_Vw_equals_Vsf(y);
    const HVX_Vector t = vmul(vsub(y, Q6_Vsf_equals_Vw(n)), vsf(0.69314718f));
    HVX_Vector p = vsf(1.0f / 5040.0f);
    p = vadd(vmul(p, t), vsf(1.0f / 720.0f));
    p = vadd(vmul(p, t), vsf(1.0f / 120.0f));
    p = vadd(vmul(p, t), vsf(1.0f / 24.0f));
    p = vadd(vmul(p, t), vsf(1.0f / 6.0f));
    p = vadd(vmul(p, t), vsf(0.5f));
    p = vadd(vmul(p, t), vsf(1.0f));
    p = vadd(vmul(p, t), vsf(1.0f));
    return Q6_Vw_vadd_VwVw(p, Q6_Vw_vasl_VwR(n, 23));   // times 2^n: n into the exponent
}

// 1 / x for positive normal x: the bit trick's guess, three Newton steps (relative error ~1e-7)
static inline HVX_Vector vrecip(HVX_Vector x) {
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), x);
    for (int i = 0; i < 3; ++i) y = vmul(y, vsub(vsf(2.0f), vmul(x, y)));
    return y;
}

static inline HVX_Vector vsigmoid(HVX_Vector x) { return vrecip(vadd(vsf(1.0f), vexp(vsub(Q6_V_vzero(), x)))); }
static inline HVX_Vector vsilu(HVX_Vector x) { return vmul(x, vsigmoid(x)); }

// silu over n floats (n a multiple of 32, 128-byte aligned), in place
static void silu_n(float *x, int n) {
    HVX_Vector *v = (HVX_Vector *)x;
    for (int i = 0; i < n / 32; ++i) v[i] = vsilu(v[i]);
}

// the halves of a vector added: lanes 0..15 get lane j + lane j + 16
static inline HVX_Vector fold16(HVX_Vector v) { return vadd(v, Q6_V_vror_VR(v, 64)); }

// a vector's 32 lanes summed (read back through memory)
static inline float hsum(HVX_Vector v) {
    v = vadd(v, Q6_V_vror_VR(v, 64));
    v = vadd(v, Q6_V_vror_VR(v, 32));
    v = vadd(v, Q6_V_vror_VR(v, 16));
    v = vadd(v, Q6_V_vror_VR(v, 8));
    v = vadd(v, Q6_V_vror_VR(v, 4));
    HVX_Vector m __attribute__((aligned(128))) = v;
    return *(const float *)&m;
}

// 128 int8 values -> four float vectors, in order
static inline void i8_sf(HVX_Vector raw, HVX_Vector *f) {
    const HVX_VectorPair hw = Q6_Wh_vunpack_Vb(raw);
    const HVX_VectorPair w0 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(hw)), w1 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(hw));
    f[0] = Q6_Vsf_equals_Vw(Q6_V_lo_W(w0));
    f[1] = Q6_Vsf_equals_Vw(Q6_V_hi_W(w0));
    f[2] = Q6_Vsf_equals_Vw(Q6_V_lo_W(w1));
    f[3] = Q6_Vsf_equals_Vw(Q6_V_hi_W(w1));
}

// The sums of 32 vectors' lanes, all at once: a tree of deals (even words / odd words of a pair) and adds keeps each
// vector's partial sums apart and halves them at each level; lane l of the result is vector tree_perm[l]'s sum
// (worked out once from test vectors: tree_check)
static HVX_Vector tree32(HVX_Vector *v) {
    for (int n = 32; n > 1; n /= 2)
        for (int i = 0; i < n / 2; ++i) {
            const HVX_VectorPair w = Q6_W_vdeal_VVR(v[2 * i + 1], v[2 * i], -4);
            v[i] = vadd(Q6_V_lo_W(w), Q6_V_hi_W(w));
        }
    return v[0];
}

static int tree_check(dec_t *d) {
    HVX_Vector v[32];
    for (int i = 0; i < 32; ++i) v[i] = vsf((float)(i + 1));
    HVX_Vector r __attribute__((aligned(128))) = tree32(v);
    const float *f = (const float *)&r;
    int seen = 0;
    for (int l = 0; l < 32; ++l) {
        const int i = (int)(f[l] / 32.0f + 0.5f) - 1;
        if (i < 0 || i > 31 || f[l] != 32.0f * (float)(i + 1) || (seen >> i & 1)) return 0;
        seen |= 1 << i;
        d->tree_perm[l] = (uint8_t)i;
    }
    return 1;
}

// vcvt's lane orders, worked out once from test vectors like tree32's: cvt_to 1 when hf = vcvt(a, b) puts a's values
// at the even lanes and b's at the odd ones, 2 the other way round; cvt_from 1 when vcvt(hf)'s lo holds the even lanes,
// 2 when its hi does; -1: neither (the scalar copies)
static void cvt_check(dec_t *d) {
    float a[32] __attribute__((aligned(128))), b[32] __attribute__((aligned(128)));
    for (int i = 0; i < 32; ++i) a[i] = (float)(i + 1), b[i] = -(float)(i + 1);
    __fp16 h[64] __attribute__((aligned(128)));
    *(HVX_Vector *)h = Q6_Vhf_vcvt_VsfVsf(*(const HVX_Vector *)a, *(const HVX_Vector *)b);
    int ea = 1, eb = 1;
    for (int i = 0; i < 32; ++i) {
        ea &= (float)h[2 * i] == a[i] && (float)h[2 * i + 1] == b[i];
        eb &= (float)h[2 * i] == b[i] && (float)h[2 * i + 1] == a[i];
    }
    d->cvt_to = ea ? 1 : eb ? 2 : -1;
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
    d->cvt_from = le ? 1 : he ? 2 : -1;
}

// ---- quantizing: 64 values (four blocks of 16) at a time ------------------------------------------------------------

// values v[0..64) are the vector's [64g, 64g + 64)
static void quant64(const float *v64, const act_t *a, int g) {
    for (int b = 0; b < 4; ++b) {
        const int blk = 4 * g + b;
        const float *v = v64 + 16 * b;
        float mx = 0.0f;
        for (int j = 0; j < 16; ++j) mx = fabsf(v[j]) > mx ? fabsf(v[j]) : mx;
        const float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        int sum = 0;
        for (int j = 0; j < 16; ++j) {
            const float t = v[j] / s;
            int q = t >= 0.0f ? (int)(t + 0.5f) : -(int)(-t + 0.5f);
            q = q > 127 ? 127 : q < -127 ? -127 : q;
            a->q[16 * blk + j] = (int8_t)q;
            sum += q;
        }
        a->bs[blk] = sum;
        a->sc[blk] = s;
    }
    for (int j = 8 * g; j < 8 * g + 8; ++j) {
        const uint8_t *v = (const uint8_t *)(a->q + 8 * j);
        a->lo[j] = (uint32_t)v[0] | (uint32_t)v[2] << 8 | (uint32_t)v[4] << 16 | (uint32_t)v[6] << 24;
        a->hi[j] = (uint32_t)v[1] | (uint32_t)v[3] << 8 | (uint32_t)v[5] << 16 | (uint32_t)v[7] << 24;
    }
}

// 64 values (128-byte aligned) the same way in HVX: per 16 the absolute maximum (rotations stay within each
// 16-lane half for lanes 0 and 16), codes rounded to nearest by adding 1.5 * 2^23, packed to bytes, the even / odd
// bytes dealt apart for tile8's words
static void quant64v(const float *v64, const act_t *a, int g) {
    const HVX_Vector *x = (const HVX_Vector *)v64;
    const HVX_VectorPred lo = Q6_Q_vsetq_R(64);
    HVX_Vector qi[2];
    int32_t sums[4];
    float scs[4];
    for (int h = 0; h < 2; ++h) {
        HVX_Vector m = Q6_V_vand_VV(x[h], Q6_V_vsplat_R(0x7FFFFFFF));
        m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vror_VR(m, 4));
        m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vror_VR(m, 8));
        m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vror_VR(m, 16));
        m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vror_VR(m, 32));
        HVX_Vector mm __attribute__((aligned(128))) = m;
        const float *mf = (const float *)&mm;
        float inv[2];
        for (int b = 0; b < 2; ++b) {
            const float mx = mf[16 * b];
            scs[2 * h + b] = mx > 0.0f ? mx / 127.0f : 1.0f;
            inv[b] = 1.0f / scs[2 * h + b];
        }
        const HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(x[h], Q6_V_vmux_QVV(lo, vsf(inv[0]), vsf(inv[1])));
        // round to nearest: t + 1.5 * 2^23 holds the integer in its low mantissa bits
        HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_Vsf_vadd_VsfVsf(t, vsf(12582912.0f)), Q6_V_vsplat_R(0x4B400000));
        r = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(r, Q6_V_vsplat_R(127)), Q6_V_vsplat_R(-127));
        qi[h] = r;
        HVX_Vector s = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 4));
        s = Q6_Vw_vadd_VwVw(s, Q6_V_vror_VR(s, 8));
        s = Q6_Vw_vadd_VwVw(s, Q6_V_vror_VR(s, 16));
        s = Q6_Vw_vadd_VwVw(s, Q6_V_vror_VR(s, 32));
        HVX_Vector ss __attribute__((aligned(128))) = s;
        const int32_t *si = (const int32_t *)&ss;
        sums[2 * h] = si[0];
        sums[2 * h + 1] = si[16];
    }
    // words -> halfwords -> bytes: 64 codes in the vector's first 64 bytes, in order
    const HVX_Vector hw = Q6_Vh_vpack_VwVw_sat(qi[1], qi[0]);
    const HVX_Vector by = Q6_Vb_vpack_VhVh_sat(Q6_V_vzero(), hw);
    HVX_Vector bv __attribute__((aligned(128))) = by;
    memcpy(a->q + 64 * g, &bv, 64);
    // the even bytes to the low half, the odd ones to the high half: lo / hi words of 4 codes each
    HVX_Vector dv __attribute__((aligned(128))) = Q6_Vb_vdeal_Vb(by);
    const uint8_t *db = (const uint8_t *)&dv;
    memcpy(a->lo + 8 * g, db, 32);
    memcpy(a->hi + 8 * g, db + 64, 32);
    for (int b = 0; b < 4; ++b) {
        a->bs[4 * g + b] = sums[b];
        a->sc[4 * g + b] = scs[b];
    }
}

// 64 values for the products: the HVX quantizer (DEC_DEBUG_QSCALAR: the scalar one; DEC_DEBUG_QCHECK: both, the
// HVX one's differences from the scalar's counted)
static void quant(const float *v64, const act_t *a, int g) {
    const int f = g_dec->flags;
    if (f & DEC_DEBUG_QSCALAR) {
        quant64(v64, a, g);
        return;
    }
    quant64v(v64, a, g);
    if (!(f & DEC_DEBUG_QCHECK)) return;
    int8_t q[64];
    uint32_t lo[8], hi[8];
    int32_t bs[4];
    float sc[4];
    act_t r = {q - 64 * g, lo - 8 * g, hi - 8 * g, bs - 4 * g, sc - 4 * g};
    quant64(v64, &r, g);
    int bad = 0;
    for (int i = 0; i < 64; ++i) bad += q[i] != a->q[64 * g + i];
    for (int i = 0; i < 8; ++i) bad += (lo[i] != a->lo[8 * g + i]) + (hi[i] != a->hi[8 * g + i]);
    for (int i = 0; i < 4; ++i) bad += (bs[i] != a->bs[4 * g + i]) + (sc[i] != a->sc[4 * g + i]);
    if (bad) __atomic_add_fetch(&g_dec->prof[DEC_PROF_QCHECK], (uint64_t)bad, __ATOMIC_RELAXED);
}

// zero-centred RMSNorm's factor of x (n values)
static float rms_inv(const float *x, int n, float eps) {
    if (n % 64 == 0 && !((uintptr_t)x & 127)) {
        const HVX_Vector *xv = (const HVX_Vector *)x;
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        for (int i = 0; i < n / 32; i += 2) {
            s0 = vadd(s0, vmul(xv[i], xv[i]));
            s1 = vadd(s1, vmul(xv[i + 1], xv[i + 1]));
        }
        return 1.0f / sqrtf(hsum(vadd(s0, s1)) / (float)n + eps);
    }
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    return 1.0f / sqrtf(ss / (float)n + eps);
}

// ---- the products ---------------------------------------------------------------------------------------------------

// nrows rows of `row` bytes, src_stride apart, into dst one after another. bypass: straight from DDR (weights, read
// once a step); else through the DSP's cache, which may hold rows this step wrote (the KV cache)
static void dma_rows(volatile dma_t *x, void *dst, const void *src, uint32_t row, uint32_t nrows, uint32_t src_stride,
                     int bypass) {
    x->next = NULL;
    x->dst_stride = row;
    x->desc_size = 1;
    x->dst_comp = x->src_comp = 0;
    x->dst_bypass = 1;
    x->src_bypass = bypass ? 1 : 0;
    x->order = 0;
    x->done = 0;
    x->src = (uint32_t)(uintptr_t)src;
    x->dst = (uint32_t)(uintptr_t)dst;
    x->desc_type = 9;
    x->reserved0 = 0;
    x->row_size = row;
    x->nrows_lo = nrows & 0xFF;
    x->nrows_hi = (nrows >> 8) & 0xFF;
    x->src_stride = src_stride;
    x->offset = 0;
    x->reserved1 = 0;
    asm volatile(" release(%0):at" : : "r"(x));
    asm volatile(" dmstart(%0)" : : "r"(x));
}

static void dma_wait(volatile dma_t *x) {
    while (!x->done) {
        unsigned int st;
        asm volatile(" %0 = dmpoll" : "=r"(st) : : "memory");
    }
}

static inline HVX_Vector e4m3_sf(HVX_Vector u) {   // npu_imp.c's: 32 lanes of one E4M3 byte each -> floats
    const HVX_Vector normal =
        Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_V_vand_VV(u, Q6_V_vsplat_R(0x7F)), 20), Q6_V_vsplat_R(0x3C000000));
    const HVX_VectorPred sub = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(u, Q6_V_vsplat_R(0x78)), Q6_V_vzero());
    const HVX_Vector m = Q6_Vsf_equals_Vw(Q6_V_vand_VV(u, Q6_V_vsplat_R(7)));
    const HVX_Vector subv = Q6_Vsf_vmpy_VsfVsf(m, Q6_V_vsplat_R(0x3B000000));
    return Q6_V_vmux_QVV(sub, subv, normal);
}

// 32 rows of a q4 tile8 matrix (npu_imp.c's gemv8 arithmetic) against T tokens' activations into y[t][0..32): the
// weights' nibbles taken apart once for all of them
static inline __attribute__((always_inline)) void q4_tile_t(const uint8_t *tile, int k, const act_t *const *a, int T,
                                                            float global, float *const *y) {
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    const int nch = k / 64;
    const HVX_Vector *cv = (const HVX_Vector *)tile, *sv = cv + 8 * nch;
    HVX_Vector acc[DEC_T][4];
    for (int t = 0; t < T; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = Q6_V_vzero();
    for (int c = 0; c < nch; ++c) {
        HVX_Vector dq[DEC_T][4];
        for (int b = 0; b < 4; ++b) {
            const int j = c * 8 + 2 * b;
            const HVX_Vector v0 = cv[j], v1 = cv[j + 1];
            const HVX_Vector n0 = Q6_V_vand_VV(v0, m0f), n1 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v0, 4), m0f);
            const HVX_Vector n2 = Q6_V_vand_VV(v1, m0f), n3 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v1, 4), m0f);
            for (int t = 0; t < T; ++t) {
                HVX_Vector s = Q6_Vw_vrmpy_VubRb(n0, (int)a[t]->lo[j]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, n1, (int)a[t]->hi[j]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, n2, (int)a[t]->lo[j + 1]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, n3, (int)a[t]->hi[j + 1]);
                const HVX_Vector dot = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(s, s), Q6_V_vsplat_R(15 * a[t]->bs[c * 4 + b]));
                dq[t][b] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(dot), vsf(a[t]->sc[c * 4 + b]));
            }
        }
        const HVX_VectorPair sh = Q6_Wuh_vunpack_Vub(sv[c]);
        const HVX_VectorPair w0 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(sh)), w1 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(sh));
        const HVX_Vector s0 = e4m3_sf(Q6_V_lo_W(w0)), s1 = e4m3_sf(Q6_V_hi_W(w0));
        const HVX_Vector s2 = e4m3_sf(Q6_V_lo_W(w1)), s3 = e4m3_sf(Q6_V_hi_W(w1));
        for (int t = 0; t < T; ++t) {
            const HVX_VectorPair p = Q6_W_vshuff_VVR(dq[t][1], dq[t][0], -16), q = Q6_W_vshuff_VVR(dq[t][3], dq[t][2], -16);
            const HVX_VectorPair r0 = Q6_W_vshuff_VVR(Q6_V_lo_W(q), Q6_V_lo_W(p), -32);
            const HVX_VectorPair r1 = Q6_W_vshuff_VVR(Q6_V_hi_W(q), Q6_V_hi_W(p), -32);
            acc[t][0] = vadd(acc[t][0], vmul(Q6_V_lo_W(r0), s0));
            acc[t][1] = vadd(acc[t][1], vmul(Q6_V_hi_W(r0), s1));
            acc[t][2] = vadd(acc[t][2], vmul(Q6_V_lo_W(r1), s2));
            acc[t][3] = vadd(acc[t][3], vmul(Q6_V_hi_W(r1), s3));
        }
    }
    for (int t = 0; t < T; ++t) {
        HVX_Vector fv[4] __attribute__((aligned(128)));
        fv[0] = acc[t][0], fv[1] = acc[t][1], fv[2] = acc[t][2], fv[3] = acc[t][3];
        const float *f = (const float *)fv;
        for (int i = 0; i < 8; ++i)
            for (int r = 0; r < 4; ++r) {
                const float *u = f + (i >> 1) * 32 + (i & 1) * 16 + r;
                y[t][4 * i + r] = global * (u[0] + u[4] + u[8] + u[12]);
            }
    }
}

// 32 rows of a q4b tile (dec_cfg.h) against T tokens' activations: the scales come as bf16 in the rows' lanes, so a
// block's sums times its scales add straight into one accumulator a token - no E4M3 conversion, no reordering, a
// third fewer instructions than tile8's: the products are bound by the reads again, and a second token costs little
static inline __attribute__((always_inline)) void q4b_tile_t(const uint8_t *tile, int k, const act_t *const *a, int T,
                                                             float global, float *const *y) {
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    const int nch = k / 64;
    const HVX_Vector *cv = (const HVX_Vector *)tile, *sv = cv + 8 * nch;
    HVX_Vector acc[DEC_T];
    for (int t = 0; t < T; ++t) acc[t] = Q6_V_vzero();
    for (int c = 0; c < nch; ++c) {
        // the four blocks' scales: bf16 -> floats by a shift (lane = row)
        const HVX_VectorPair w01 = Q6_Wuw_vunpack_Vuh(sv[2 * c]), w23 = Q6_Wuw_vunpack_Vuh(sv[2 * c + 1]);
        const HVX_Vector ws[4] = {Q6_Vw_vasl_VwR(Q6_V_lo_W(w01), 16), Q6_Vw_vasl_VwR(Q6_V_hi_W(w01), 16),
                                  Q6_Vw_vasl_VwR(Q6_V_lo_W(w23), 16), Q6_Vw_vasl_VwR(Q6_V_hi_W(w23), 16)};
        for (int b = 0; b < 4; ++b) {
            const int j = c * 8 + 2 * b;
            const HVX_Vector v0 = cv[j], v1 = cv[j + 1];
            const HVX_Vector n0 = Q6_V_vand_VV(v0, m0f), n1 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v0, 4), m0f);
            const HVX_Vector n2 = Q6_V_vand_VV(v1, m0f), n3 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v1, 4), m0f);
            for (int t = 0; t < T; ++t) {
                HVX_Vector s = Q6_Vw_vrmpy_VubRb(n0, (int)a[t]->lo[j]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, n1, (int)a[t]->hi[j]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, n2, (int)a[t]->lo[j + 1]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, n3, (int)a[t]->hi[j + 1]);
                // sum (2 code - 15) x = 2 s - 15 sum x (exact in int32)
                const HVX_Vector dot = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(s, s), Q6_V_vsplat_R(15 * a[t]->bs[c * 4 + b]));
                acc[t] = vadd(acc[t], vmul(vmul(Q6_Vsf_equals_Vw(dot), vsf(a[t]->sc[c * 4 + b])), ws[b]));
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        HVX_Vector fv __attribute__((aligned(128))) = acc[t];
        const float *f = (const float *)&fv;
        for (int r = 0; r < 32; ++r) y[t][r] = global * f[r];
    }
}

// q4b_tile_t for a pair, written out: two accumulators in registers (the arrays of the general one spill - its pair
// cost 3.3 single tokens on one thread), the nibbles and the scales taken apart once for both
static void q4b_tile2(const uint8_t *tile, int k, const act_t *a0, const act_t *a1, float global, float *y0,
                      float *y1) {
    const HVX_Vector m0f = Q6_Vb_vsplat_R(0x0F);
    const int nch = k / 64;
    const HVX_Vector *cv = (const HVX_Vector *)tile, *sv = cv + 8 * nch;
    const uint32_t *lo0 = a0->lo, *hi0 = a0->hi, *lo1 = a1->lo, *hi1 = a1->hi;
    const int32_t *bs0 = a0->bs, *bs1 = a1->bs;
    const float *sc0 = a0->sc, *sc1 = a1->sc;
    HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero();
    for (int c = 0; c < nch; ++c) {
        const HVX_VectorPair w01 = Q6_Wuw_vunpack_Vuh(sv[2 * c]), w23 = Q6_Wuw_vunpack_Vuh(sv[2 * c + 1]);
        for (int b = 0; b < 4; ++b) {
            const HVX_Vector ws = Q6_Vw_vasl_VwR(b == 0 ? Q6_V_lo_W(w01) : b == 1 ? Q6_V_hi_W(w01)
                                                 : b == 2 ? Q6_V_lo_W(w23) : Q6_V_hi_W(w23), 16);
            const int j = c * 8 + 2 * b, q = c * 4 + b;
            const HVX_Vector v0 = cv[j], v1 = cv[j + 1];
            const HVX_Vector n0 = Q6_V_vand_VV(v0, m0f), n1 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v0, 4), m0f);
            const HVX_Vector n2 = Q6_V_vand_VV(v1, m0f), n3 = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v1, 4), m0f);
            HVX_Vector s0 = Q6_Vw_vrmpy_VubRb(n0, (int)lo0[j]);
            HVX_Vector s1 = Q6_Vw_vrmpy_VubRb(n0, (int)lo1[j]);
            s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, n1, (int)hi0[j]);
            s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, n1, (int)hi1[j]);
            s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, n2, (int)lo0[j + 1]);
            s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, n2, (int)lo1[j + 1]);
            s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, n3, (int)hi0[j + 1]);
            s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, n3, (int)hi1[j + 1]);
            const HVX_Vector d0 = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(s0, s0), Q6_V_vsplat_R(15 * bs0[q]));
            const HVX_Vector d1 = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(s1, s1), Q6_V_vsplat_R(15 * bs1[q]));
            acc0 = vadd(acc0, vmul(vmul(Q6_Vsf_equals_Vw(d0), vsf(sc0[q])), ws));
            acc1 = vadd(acc1, vmul(vmul(Q6_Vsf_equals_Vw(d1), vsf(sc1[q])), ws));
        }
    }
    HVX_Vector f0 __attribute__((aligned(128))) = acc0, f1 __attribute__((aligned(128))) = acc1;
    const float *g0 = (const float *)&f0, *g1 = (const float *)&f1;
    for (int r = 0; r < 32; ++r) {
        y0[r] = global * g0[r];
        y1[r] = global * g1[r];
    }
}

// 32 rows of a q8 tile against T tokens' activations: per 64 inputs 16 code vectors (lane = row, 4 inputs as
// code + 128) and the scales [block][row]
static inline __attribute__((always_inline)) void q8_tile_t(const uint8_t *tile, int k, const act_t *const *a, int T,
                                                            float global, float *const *y) {
    const int nch = k / 64;
    const HVX_Vector *v = (const HVX_Vector *)tile;
    HVX_Vector acc[DEC_T];
    for (int t = 0; t < T; ++t) acc[t] = Q6_V_vzero();
    for (int c = 0; c < nch; ++c, v += 17) {
        const HVX_VectorPair sh = Q6_Wuh_vunpack_Vub(v[16]);
        const HVX_VectorPair w0 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(sh)), w1 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(sh));
        const HVX_Vector ws[4] = {e4m3_sf(Q6_V_lo_W(w0)), e4m3_sf(Q6_V_hi_W(w0)), e4m3_sf(Q6_V_lo_W(w1)),
                                  e4m3_sf(Q6_V_hi_W(w1))};
        for (int t = 0; t < T; ++t) {
            const uint32_t *words = (const uint32_t *)a[t]->q;
            for (int b = 0; b < 4; ++b) {
                const uint32_t *x = words + c * 16 + b * 4;
                HVX_Vector s = Q6_Vw_vrmpy_VubRb(v[4 * b], (int)x[0]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, v[4 * b + 1], (int)x[1]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, v[4 * b + 2], (int)x[2]);
                s = Q6_Vw_vrmpyacc_VwVubRb(s, v[4 * b + 3], (int)x[3]);
                const HVX_Vector dot = Q6_Vw_vsub_VwVw(s, Q6_V_vsplat_R(128 * a[t]->bs[c * 4 + b]));
                const HVX_Vector dq = vmul(Q6_Vsf_equals_Vw(dot), vsf(a[t]->sc[c * 4 + b]));
                acc[t] = vadd(acc[t], vmul(dq, ws[b]));
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        HVX_Vector fv __attribute__((aligned(128))) = acc[t];
        const float *f = (const float *)&fv;
        for (int r = 0; r < 32; ++r) y[t][r] = global * f[r];
    }
}

static inline uint32_t tile_bytes(const dec_mat_t *m) {
    return m->k * (m->fmt == DEC_FMT_Q8 ? 34u : m->fmt == DEC_FMT_Q4B ? 20u : 18u);
}

typedef struct {
    const dec_mat_t *m;
    const act_t *a[DEC_T];
    float *y[DEC_T];
} job_t;

// The stage's products for T tokens: every row tile of every job, taken one at a time, each fetched by the DMA engine
// while the thread's last one is multiplied; then `extra` tasks of fn from the same counter (the GDN's a / b rows)
static void products(dec_t *d, int th, int st, const job_t *jobs, int nj, int T, int extra, const dec_layer_t *L,
                     void (*fn)(dec_t *, const dec_layer_t *, int)) {
    int start[5], total = 0;
    for (int j = 0; j < nj; ++j) {
        start[j] = total;
        total += (int)jobs[j].m->n / 32;
    }
    start[nj] = total;
    uint8_t *buf[2] = {d->vtcm + (size_t)th * DEC_THREAD_VTCM, d->vtcm + (size_t)th * DEC_THREAD_VTCM + DEC_TILE_MAX};
    volatile dma_t *dm = d->dma[th];
#define TILE_AT(i, jj, tt)                                                                    \
    do {                                                                                     \
        jj = 0;                                                                              \
        while ((i) >= start[jj + 1]) ++jj;                                                   \
        tt = (i) - start[jj];                                                                \
    } while (0)
    int cur = take(d, st), b = 0;
    if (cur < total) {
        int j, t;
        TILE_AT(cur, j, t);
        dma_rows(&dm[0], buf[0], d->W + jobs[j].m->w + (uint64_t)t * tile_bytes(jobs[j].m), tile_bytes(jobs[j].m), 1,
                 tile_bytes(jobs[j].m), 1);
    }
    while (cur < total) {
        const int nxt = take(d, st);
        dma_wait(&dm[b]);
        if (nxt < total) {   // the next tile on its way while this one is multiplied
            int j, t;
            TILE_AT(nxt, j, t);
            dma_rows(&dm[b ^ 1], buf[b ^ 1], d->W + jobs[j].m->w + (uint64_t)t * tile_bytes(jobs[j].m),
                     tile_bytes(jobs[j].m), 1, tile_bytes(jobs[j].m), 1);
        }
        int j, t;
        TILE_AT(cur, j, t);
        const dec_mat_t *m = jobs[j].m;
        float *ys[DEC_T];
        for (int i = 0; i < T; ++i) ys[i] = jobs[j].y[i] + t * 32;
        if (m->fmt == DEC_FMT_Q4B) {
            if (T == 1) q4b_tile_t(buf[b], (int)m->k, jobs[j].a, 1, m->global, ys);
            else q4b_tile2(buf[b], (int)m->k, jobs[j].a[0], jobs[j].a[1], m->global, ys[0], ys[1]);
        } else if (m->fmt == DEC_FMT_Q8) {
            if (T == 1) q8_tile_t(buf[b], (int)m->k, jobs[j].a, 1, m->global, ys);
            else q8_tile_t(buf[b], (int)m->k, jobs[j].a, 2, m->global, ys);
        } else {
            if (T == 1) q4_tile_t(buf[b], (int)m->k, jobs[j].a, 1, m->global, ys);
            else q4_tile_t(buf[b], (int)m->k, jobs[j].a, 2, m->global, ys);
        }
        cur = nxt;
        b ^= 1;
    }
#undef TILE_AT
    for (; cur < total + extra; cur = take(d, st)) fn(d, L, cur - total);
}

// ---- the layers ---------------------------------------------------------------------------------------------------

// the GDN's a / b row r of token t (task r + t * 2 nv): bf16 rows against the normed input (fin)
static void ab_row(dec_t *d, const dec_layer_t *L, int task) {
    const dec_cfg_t *c = &d->c;
    const int r = task % (2 * c->nv), t = task / (2 * c->nv);
    const uint16_t *w = (const uint16_t *)(d->P + L->ab) + (size_t)r * c->hidden;
    const float *x = d->fin[t];
    if (c->hidden % 64 == 0 && !((uintptr_t)w & 127)) {   // 64 weights a vector, as floats: bf16 << 16
        const HVX_Vector *wv = (const HVX_Vector *)w, *xv = (const HVX_Vector *)x;
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        for (int i = 0; i < c->hidden / 64; ++i) {
            const HVX_VectorPair u = Q6_Wuw_vunpack_Vuh(wv[i]);
            s0 = vadd(s0, vmul(Q6_Vw_vasl_VwR(Q6_V_lo_W(u), 16), xv[2 * i]));
            s1 = vadd(s1, vmul(Q6_Vw_vasl_VwR(Q6_V_hi_W(u), 16), xv[2 * i + 1]));
        }
        d->ab[t][r] = hsum(vadd(s0, s1));
        return;
    }
    float s = 0.0f;
    for (int i = 0; i < c->hidden; ++i) s += bf16f(w[i]) * x[i];
    d->ab[t][r] = s;
}

// a layer half's input for every token: x + add into xo (the thread's 64-value groups), normed with w (into fin too)
// and quantized; every thread works out the whole norms itself (cheaper than a barrier)
static void norm_stage(dec_t *d, int st, float *const *x, float *const *add, float *const *xo, uint32_t w_at,
                       act_t *a) {
    const dec_cfg_t *c = &d->c;
    const int H = c->hidden, T = d->T;
    float inv[DEC_N];
    for (int t = 0; t < T; ++t) {
        const HVX_Vector *xv = (const HVX_Vector *)x[t], *av = add ? (const HVX_Vector *)add[t] : NULL;
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        for (int i = 0; i < H / 32; i += 2) {
            const HVX_Vector v0 = av ? vadd(xv[i], av[i]) : xv[i], v1 = av ? vadd(xv[i + 1], av[i + 1]) : xv[i + 1];
            s0 = vadd(s0, vmul(v0, v0));
            s1 = vadd(s1, vmul(v1, v1));
        }
        inv[t] = 1.0f / sqrtf(hsum(vadd(s0, s1)) / (float)H + c->eps);
    }
    const float *w = pf(d, w_at);
    for (int task = take(d, st); task < T * (H / 64); task = take(d, st)) {
        const int t = task / (H / 64), g = task % (H / 64);
        float *h = d->fin[t] + 64 * g;
        for (int i = 0; i < 64; ++i) {
            const int at = 64 * g + i;
            const float v = add ? x[t][at] + add[t][at] : x[t][at];
            if (xo) xo[t][at] = v;
            h[i] = v * inv[t] * (1.0f + w[at]);
        }
        quant(h, &a[t], g);
    }
}

// the convolution over the last conv_k inputs, then silu, channels [64g, 64g + 64), token by token; a verified pair's
// first token's states copied into the other set as they are
static void conv_task(dec_t *d, const dec_layer_t *L, int g) {
    const dec_cfg_t *c = &d->c;
    const int K = c->conv_k, conv_dim = 2 * c->nk * c->dk + c->nv * c->dv, T = d->T;
    const size_t at = (size_t)L->index * conv_dim * (K - 1);
    float *cs = d->conv_live + at, *bk = d->conv_other + at;
    const float *cw = pf(d, L->conv);
    const int c1 = 64 * g + 64 < conv_dim ? 64 * g + 64 : conv_dim;
    if (K == 4 && c1 - 64 * g == 64) {
        // 64 channels in two vectors: the weights and the state ([channel][3] in memory) gathered by tap, the taps
        // summed in the scalar order, the state scattered back
        float wk[4][64] __attribute__((aligned(128))), sk[3][64] __attribute__((aligned(128)));
        for (int i = 0; i < 64; ++i) {
            const float *w = cw + (size_t)(64 * g + i) * 4, *st = cs + (size_t)(64 * g + i) * 3;
            wk[0][i] = w[0], wk[1][i] = w[1], wk[2][i] = w[2], wk[3][i] = w[3];
            sk[0][i] = st[0], sk[1][i] = st[1], sk[2][i] = st[2];
        }
        const HVX_Vector *W = (const HVX_Vector *)wk;   // [tap][half]
        HVX_Vector *S = (HVX_Vector *)sk;
        for (int t = 0; t < T; ++t) {
            HVX_Vector *pv = (HVX_Vector *)(d->proj[t] + 64 * g);
            for (int h = 0; h < 2; ++h) {
                const HVX_Vector in = pv[h];
                HVX_Vector acc = vmul(W[h], S[h]);
                acc = vadd(acc, vmul(W[2 + h], S[2 + h]));
                acc = vadd(acc, vmul(W[4 + h], S[4 + h]));
                acc = vadd(acc, vmul(W[6 + h], in));
                S[h] = S[2 + h];
                S[2 + h] = S[4 + h];
                S[4 + h] = in;
                pv[h] = vsilu(acc);
            }
            if (d->backup && t == 0)
                for (int i = 0; i < 64; ++i) {
                    float *b = bk + (size_t)(64 * g + i) * 3;
                    b[0] = sk[0][i], b[1] = sk[1][i], b[2] = sk[2][i];
                }
        }
        for (int i = 0; i < 64; ++i) {
            float *st = cs + (size_t)(64 * g + i) * 3;
            st[0] = sk[0][i], st[1] = sk[1][i], st[2] = sk[2][i];
        }
        return;
    }
    for (int t = 0; t < T; ++t) {
        float *proj = d->proj[t];
        for (int ch = 64 * g; ch < c1; ++ch) {
            const float *w = cw + (size_t)ch * K;
            float *s = cs + (size_t)ch * (K - 1);
            const float in = proj[ch];
            float acc = 0.0f;
            for (int k = 0; k < K - 1; ++k) acc += w[k] * s[k];
            acc += w[K - 1] * in;
            for (int k = 0; k + 1 < K - 1; ++k) s[k] = s[k + 1];
            if (K > 1) s[K - 2] = in;
            proj[ch] = acc;
        }
        if (d->backup && t == 0) memcpy(bk + (size_t)(64 * g) * (K - 1), cs + (size_t)(64 * g) * (K - 1),
                                     (size_t)(c1 - 64 * g) * (K - 1) * sizeof(float));
        if (c1 - 64 * g == 64) silu_n(proj + 64 * g, 64);
        else
            for (int ch = 64 * g; ch < c1; ++ch) proj[ch] = siluf(proj[ch]);
    }
}

// one value head, token by token: its q / k normalised, the gated delta rule on its state (a verified pair's first
// token's state copied into the other set strip by strip), the gated RMSNorm, quantized for out
static void gdn_head(dec_t *d, const dec_layer_t *L, int vh, int th) {
    const dec_cfg_t *c = &d->c;
    const int nk = c->nk, nv = c->nv, dk = c->dk, dv = c->dv, kd = nk * dk, kh = vh / (nv / nk), T = d->T;
    const size_t at = ((size_t)L->index * nv + vh) * dk * dv;
    float *state = d->gdn_live + at, *other = d->gdn_other + at;
    // per token: k's and q's rows two a vector (lanes 0-15 row 2r, 16-31 row 2r + 1), in the thread's VTCM area
    HVX_Vector *kv0 = (HVX_Vector *)(d->vtcm + (size_t)th * DEC_THREAD_VTCM + 2u * DEC_TILE_MAX);
    const HVX_VectorPred half = Q6_Q_vsetq_R(64);
    float beta[DEC_T], decay[DEC_T];
    for (int t = 0; t < T; ++t) {
        const float *mixed = d->proj[t];
        const float *qs = mixed + kh * dk, *ks = mixed + kd + kh * dk;
        float sq = 0.0f, sk = 0.0f;
        for (int j = 0; j < dk; ++j) {
            sq += qs[j] * qs[j];
            sk += ks[j] * ks[j];
        }
        const float iq = 1.0f / sqrtf(sq + 1e-6f) / sqrtf((float)dk), ik = 1.0f / sqrtf(sk + 1e-6f);
        HVX_Vector *kv = kv0 + (size_t)t * dk, *qv = kv + dk / 2;
        for (int r = 0; r < dk / 2; ++r) {
            kv[r] = Q6_V_vmux_QVV(half, vsf(ks[2 * r] * ik), vsf(ks[2 * r + 1] * ik));
            qv[r] = Q6_V_vmux_QVV(half, vsf(qs[2 * r] * iq), vsf(qs[2 * r + 1] * iq));
        }
        beta[t] = sigmoidf(d->ab[t][nv + vh]);
        decay[t] = expf(-expf(pf(d, L->A_log)[vh]) * softplusf(d->ab[t][vh] + pf(d, L->dt_bias)[vh]));
    }
    for (int j0 = 0; j0 < dv; j0 += 16) {
        HVX_Vector *S = (HVX_Vector *)(state + (size_t)j0 * dk);
        for (int t = 0; t < T; ++t) {
            const HVX_Vector *kv = kv0 + (size_t)t * dk, *qv = kv + dk / 2;
            const float *vs = d->proj[t] + 2 * kd + vh * dv;
            // S *= decay; mem = S^T k
            const HVX_Vector dec = vsf(decay[t]);
            HVX_Vector acc = Q6_V_vzero();
            for (int r = 0; r < dk / 2; ++r) {
                const HVX_Vector v = vmul(S[r], dec);
                S[r] = v;
                acc = vadd(acc, vmul(v, kv[r]));
            }
            HVX_Vector mv __attribute__((aligned(128))) = fold16(acc);
            const float *mem = (const float *)&mv;
            float dl[32] __attribute__((aligned(128)));
            for (int j = 0; j < 16; ++j) dl[j] = dl[j + 16] = (vs[j0 + j] - mem[j]) * beta[t];
            // S += k delta^T; o = S^T q
            const HVX_Vector dd = *(const HVX_Vector *)dl;
            HVX_Vector ao = Q6_V_vzero();
            for (int r = 0; r < dk / 2; ++r) {
                const HVX_Vector v = vadd(S[r], vmul(kv[r], dd));
                S[r] = v;
                ao = vadd(ao, vmul(v, qv[r]));
            }
            HVX_Vector ov __attribute__((aligned(128))) = fold16(ao);
            memcpy(d->core[t] + vh * dv + j0, &ov, 16 * sizeof(float));
            if (d->backup && t == 0) {   // the strip as after the first token, while it is in the cache
                HVX_Vector *O = (HVX_Vector *)(other + (size_t)j0 * dk);
                for (int r = 0; r < dk / 2; ++r) O[r] = S[r];
            }
        }
    }
    // gated RMSNorm: o / rms(o) * w * silu(z), each token
    const float *gw = pf(d, L->gnorm);
    for (int t = 0; t < T; ++t) {
        float *o = d->core[t] + vh * dv;
        const float *z = d->zb[t] + vh * dv;
        float ss = 0.0f;
        for (int j = 0; j < dv; ++j) ss += o[j] * o[j];
        const float inv = 1.0f / sqrtf(ss / (float)dv + c->eps);
        if (dv % 32 == 0)
            for (int j = 0; j < dv; j += 32) {
                const HVX_Vector zv = *(const HVX_Vector *)(z + j), ov = *(const HVX_Vector *)(o + j);
                HVX_Vector gwv;
                memcpy(&gwv, gw + j, 128);
                *(HVX_Vector *)(o + j) = vmul(vmul(vmul(ov, vsf(inv)), gwv), vsilu(zv));
            }
        else
            for (int j = 0; j < dv; ++j) o[j] = o[j] * inv * gw[j] * siluf(z[j]);
        for (int g = vh * dv / 64; g < (vh + 1) * dv / 64; ++g) quant(d->core[t] + 64 * g, &d->act_c[t], g);
    }
}

static inline uint16_t f16bits(float f) {
    union { __fp16 h; uint16_t u; } v;
    v.h = (__fp16)f;
    return v.u;
}
static inline float f16f(uint16_t u) {
    union { __fp16 h; uint16_t u; } v;
    v.u = u;
    return (float)v.h;
}

// a position's K or V row in its chunk: fp16, or int8 (kv_i8) with its KV heads' scales (kv_scale)
static void *kv_row(dec_t *d, int pos, int layer, int value) {
    const dec_cfg_t *c = &d->c;
    uint8_t *base = d->kv[pos / c->kv_chunk];
    if (!base) return NULL;
    const size_t kvr = (size_t)c->nkv * c->hd, at = ((size_t)value * c->n_attn + (size_t)layer) * c->kv_chunk +
                                                    (size_t)(pos % c->kv_chunk);
    return base + at * kvr * (c->kv_i8 ? 1u : 2u);
}

static float *kv_scale(dec_t *d, int pos, int layer, int value) {
    const dec_cfg_t *c = &d->c;
    uint8_t *base = d->kv[pos / c->kv_chunk];
    float *sc = (float *)(base + 2u * (size_t)c->n_attn * c->kv_chunk * (size_t)c->nkv * c->hd);
    return sc + (((size_t)value * c->n_attn + (size_t)layer) * c->kv_chunk + (size_t)(pos % c->kv_chunk)) * c->nkv;
}

// a KV head's row to int8 as the engine's quant_kv_row: amax / 127 its scale, round to nearest even
static void quant_kv(const float *x, int n, int8_t *q, float *scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = fabsf(x[i]) > amax ? fabsf(x[i]) : amax;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = nearbyintf(x[i] * inv);
        q[i] = (int8_t)(v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
    }
    *scale = amax / 127.0f;
}

// a query head (h < nh) or a KV head (h >= nh) of token t: its RMSNorm and partial RoPE; a KV head's K and V into the
// cache at the token's position
// token t's rope cosines (half) then sines, worked out once a token on each thread (a thread takes a token's heads
// one after another)
static const float *rope_cs(dec_t *d, int t, int th) {
    const int *p = d->p3[t], half = d->c.half;
    int *k = d->rope_key + 4 * th;
    float *cs = d->rope + 256 * th;
    if (!k[3] || k[0] != p[0] || k[1] != p[1] || k[2] != p[2]) {
        const float *fr = pf(d, d->c.inv_freq);
        for (int i = 0; i < half; ++i) {
            const float f = (float)p[i % 3] * fr[i];
            cs[i] = cosf(f);
            cs[half + i] = sinf(f);
        }
        k[0] = p[0], k[1] = p[1], k[2] = p[2], k[3] = 1;
    }
    return cs;
}

static void qk_task(dec_t *d, const dec_layer_t *L, int task, int th) {
    const dec_cfg_t *c = &d->c;
    const int per = c->nh + c->nkv, t = task / per, h = task % per, hd = c->hd, half = c->half, pos = d->pos + t;
    float *x = h < c->nh ? d->qb[t] + (size_t)h * 2 * hd : d->kb[t] + (size_t)(h - c->nh) * hd;
    const float inv = rms_inv(x, hd, c->eps);
    const float *w = pf(d, h < c->nh ? L->q_norm : L->k_norm), *cs = rope_cs(d, t, th), *sn = cs + half;
    if (hd % 32 == 0 && half % 32 == 0 && !((uintptr_t)x & 127)) {
        HVX_Vector *xv = (HVX_Vector *)x;
        const HVX_UVector *wv = (const HVX_UVector *)w;
        const HVX_Vector iv = vsf(inv), one = vsf(1.0f);
        for (int i = 0; i < hd / 32; ++i) xv[i] = vmul(vmul(xv[i], iv), vadd(wv[i], one));
        for (int i = 0; i < half / 32; ++i) {
            const HVX_Vector x1 = xv[i], x2 = xv[half / 32 + i];
            const HVX_Vector cv = ((const HVX_UVector *)cs)[i], sv = ((const HVX_UVector *)sn)[i];
            xv[i] = vsub(vmul(x1, cv), vmul(x2, sv));
            xv[half / 32 + i] = vadd(vmul(x2, cv), vmul(x1, sv));
        }
    } else {
        for (int i = 0; i < hd; ++i) x[i] = x[i] * inv * (1.0f + w[i]);
        for (int i = 0; i < half; ++i) {
            const float x1 = x[i], x2 = x[i + half];
            x[i] = x1 * cs[i] - x2 * sn[i];
            x[i + half] = x2 * cs[i] + x1 * sn[i];
        }
    }
    if (h >= c->nh && !(d->flags & DEC_DEBUG_NO_KV)) {
        const int kh = h - c->nh;
        void *kr = kv_row(d, pos, L->index, 0), *vr = kv_row(d, pos, L->index, 1);
        if (!kr || !vr) {
            d->failed = 1;
            return;
        }
        const float *v = d->vb[t] + (size_t)kh * hd;
        if (c->kv_i8) {
            float *ks = kv_scale(d, pos, L->index, 0) + kh, *vs = kv_scale(d, pos, L->index, 1) + kh;
            quant_kv(x, hd, (int8_t *)kr + kh * hd, ks);
            quant_kv(v, hd, (int8_t *)vr + kh * hd, vs);
        } else {
            for (int i = 0; i < hd; ++i) {
                ((uint16_t *)kr)[kh * hd + i] = f16bits(x[i]);
                ((uint16_t *)vr)[kh * hd + i] = f16bits(v[i]);
            }
        }
    }
}

// tree32 over int32 lanes (the same lane order)
static HVX_Vector tree32w(HVX_Vector *v) {
    for (int n = 32; n > 1; n /= 2)
        for (int i = 0; i < n / 2; ++i) {
            const HVX_VectorPair w = Q6_W_vdeal_VVR(v[2 * i + 1], v[2 * i], -4);
            v[i] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(w), Q6_V_hi_W(w));
        }
    return v[0];
}

// a vector's 32 lanes' maximum
static inline float hmax(HVX_Vector v) {
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, 4));
    HVX_Vector m __attribute__((aligned(128))) = v;
    return *(const float *)&m;
}

// a row's q (hd = 256 floats) as int16 with one scale, split into a signed high byte and an unsigned low one (qv[0..1]
// and qv[2..3]); its scale times 1 / sqrt(hd) into *qsc
static void q_split(const float *q, HVX_Vector *qv, float *qsc) {
    float amax = 0.0f;
    for (int i = 0; i < 256; ++i) amax = fabsf(q[i]) > amax ? fabsf(q[i]) : amax;
    const float s0 = amax > 0.0f ? amax / 32767.0f : 1.0f, inv = 1.0f / s0;
    int8_t hi[256] __attribute__((aligned(128)));
    uint8_t lo[256] __attribute__((aligned(128)));
    for (int i = 0; i < 256; ++i) {
        const float x = q[i] * inv;
        const int v = (int)(x + (x >= 0.0f ? 0.5f : -0.5f));   // |v| <= 32767
        hi[i] = (int8_t)(v >> 8);
        lo[i] = (uint8_t)(v & 255);
    }
    qv[0] = *(const HVX_Vector *)hi;
    qv[1] = *(const HVX_Vector *)(hi + 128);
    qv[2] = *(const HVX_Vector *)lo;
    qv[3] = *(const HVX_Vector *)(lo + 128);
    *qsc = s0 / 16.0f;   // 1 / sqrt(256)
}

// One block of n positions (int8 K / V rows kvr bytes apart, their scales ks / vs nkv floats apart) for R rows, two
// at a time (rows r and r + 1 see the same lim[r] positions of the block; <= 0: none) - the work of both attentions,
// all in vectors:
//   the scores: q (q_split) . k by four vrmpy into exact int32 sums, both rows; 32 positions' sums at once by
//   tree32w, stored so its lanes come out in position order; times the K rows' scales
//   per row the block's maximum bm, the lanes past lim out, exp(s - bm) and their sum bs; the weights times the V
//   rows' scales in int16 against the block's largest V scale (wb: [R][DEC_BLOCK], twice a word)
//   the values: both rows' weights against each V row by vmpyacc into int32 (vmpy's pair holds the even halfwords'
//   products in lo, the odd ones' in hi - shuffled back into order), as floats into blk ([R][8])
// Scratch: sv [R][8] vectors, part [64] vectors
static inline __attribute__((always_inline)) void attn_rows(const dec_t *d, const int8_t *krows, const int8_t *vrows, uint32_t kvr, const float *ks,
                      const float *vs, int nkv, int n, int R, const int *lim, const HVX_Vector *restrict qv,
                      const float *qsc, HVX_Vector *restrict sv, HVX_Vector *restrict part, int32_t *restrict wb,
                      float *bm, float *bs, HVX_Vector *restrict blk) {
    uint8_t perm[32];
    memcpy(perm, d->tree_perm, sizeof perm);
    float kf[DEC_BLOCK] __attribute__((aligned(128))), vf[DEC_BLOCK] __attribute__((aligned(128)));
    float vbig = 0.0f;
    for (int i = 0; i < n; ++i) {
        kf[i] = ks[(size_t)i * nkv];
        vf[i] = vs[(size_t)i * nkv];
        vbig = vf[i] > vbig ? vf[i] : vbig;
    }
    for (int i = n; i < (n + 31) / 32 * 32; ++i) kf[i] = vf[i] = 0.0f;
    const HVX_Vector *kfv = (const HVX_Vector *)kf, *vfv = (const HVX_Vector *)vf;
    const HVX_Vector to = vsf(vbig > 0.0f ? 32000.0f / vbig : 0.0f), back = vsf(vbig / 32000.0f);
    int32_t li[32] __attribute__((aligned(128)));
    for (int i = 0; i < 32; ++i) li[i] = i;
    const HVX_Vector lanes = *(const HVX_Vector *)li;
    for (int r = 0; r < R; r += 2) {
        const int nl = lim[r] < n ? lim[r] : n;
        if (nl <= 0) {
            bm[r] = bm[r + 1] = -INFINITY;
            bs[r] = bs[r + 1] = 0.0f;
            continue;
        }
        const int nc = (nl + 31) / 32;
        // (both rows' q in registers for the whole block)
        const HVX_Vector qa0 = qv[r * 4], qa1 = qv[r * 4 + 1], qa2 = qv[r * 4 + 2], qa3 = qv[r * 4 + 3];
        const HVX_Vector qb0 = qv[r * 4 + 4], qb1 = qv[r * 4 + 5], qb2 = qv[r * 4 + 6], qb3 = qv[r * 4 + 7];
        for (int ci = 0; ci < nc; ++ci) {
            const int m = nl - 32 * ci < 32 ? nl - 32 * ci : 32;
            const int8_t *kr = krows + (size_t)(32 * ci) * kvr;
            for (int j = 0; j < m; ++j, kr += kvr) {
                const HVX_Vector k0 = ((const HVX_Vector *)kr)[0], k1 = ((const HVX_Vector *)kr)[1];
                const HVX_Vector ha = Q6_Vw_vrmpyacc_VwVbVb(Q6_Vw_vrmpy_VbVb(k0, qa0), k1, qa1);
                const HVX_Vector la = Q6_Vw_vrmpyacc_VwVubVb(Q6_Vw_vrmpy_VubVb(qa2, k0), qa3, k1);
                const HVX_Vector hb = Q6_Vw_vrmpyacc_VwVbVb(Q6_Vw_vrmpy_VbVb(k0, qb0), k1, qb1);
                const HVX_Vector lb = Q6_Vw_vrmpyacc_VwVubVb(Q6_Vw_vrmpy_VubVb(qb2, k0), qb3, k1);
                const int at = perm[j];
                part[at] = Q6_Vw_vaslacc_VwVwR(la, ha, 8);
                part[32 + at] = Q6_Vw_vaslacc_VwVwR(lb, hb, 8);
            }
            for (int j = m; j < 32; ++j) part[perm[j]] = part[32 + perm[j]] = Q6_V_vzero();
            sv[r * 8 + ci] = vmul(Q6_Vsf_equals_Vw(tree32w(part)), vmul(kfv[ci], vsf(qsc[r])));
            sv[(r + 1) * 8 + ci] = vmul(Q6_Vsf_equals_Vw(tree32w(part + 32)), vmul(kfv[ci], vsf(qsc[r + 1])));
        }
        const HVX_VectorPred keep = Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(nl - 32 * (nc - 1)), lanes);
        for (int h = 0; h < 2; ++h) {
            HVX_Vector *S = sv + (r + h) * 8;
            S[nc - 1] = Q6_V_vmux_QVV(keep, S[nc - 1], vsf(-3.0e38f));
            HVX_Vector mv = S[0];
            for (int i = 1; i < nc; ++i) mv = Q6_Vsf_vmax_VsfVsf(mv, S[i]);
            const float mx = hmax(mv);
            const HVX_Vector mxv = vsf(mx);
            HVX_Vector es = Q6_V_vzero();
            int32_t *w = wb + (r + h) * DEC_BLOCK;
            for (int i = 0; i < nc; ++i) {
                HVX_Vector e = vexp(vsub(S[i], mxv));
                if (i == nc - 1) e = Q6_V_vmux_QVV(keep, e, Q6_V_vzero());
                es = vadd(es, e);
                const HVX_Vector wi = Q6_Vw_equals_Vsf(vmul(vmul(e, vfv[i]), to));
                *(HVX_Vector *)(w + 32 * i) = Q6_V_vor_VV(wi, Q6_Vw_vasl_VwR(wi, 16));
            }
            bm[r + h] = mx;
            bs[r + h] = hsum(es);
        }
        const int32_t *w0 = wb + r * DEC_BLOCK, *w1 = w0 + DEC_BLOCK;
        HVX_VectorPair a0[4], a1[4];
        for (int i = 0; i < 4; ++i) a0[i] = a1[i] = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        for (int p = 0; p < nl; ++p) {
            const HVX_Vector *v8 = (const HVX_Vector *)(vrows + (size_t)p * kvr);
            const HVX_VectorPair h0 = Q6_Wh_vunpack_Vb(v8[0]), h1 = Q6_Wh_vunpack_Vb(v8[1]);
            const int r0 = w0[p], r1 = w1[p];
            a0[0] = Q6_Ww_vmpyacc_WwVhRh(a0[0], Q6_V_lo_W(h0), r0);
            a0[1] = Q6_Ww_vmpyacc_WwVhRh(a0[1], Q6_V_hi_W(h0), r0);
            a0[2] = Q6_Ww_vmpyacc_WwVhRh(a0[2], Q6_V_lo_W(h1), r0);
            a0[3] = Q6_Ww_vmpyacc_WwVhRh(a0[3], Q6_V_hi_W(h1), r0);
            a1[0] = Q6_Ww_vmpyacc_WwVhRh(a1[0], Q6_V_lo_W(h0), r1);
            a1[1] = Q6_Ww_vmpyacc_WwVhRh(a1[1], Q6_V_hi_W(h0), r1);
            a1[2] = Q6_Ww_vmpyacc_WwVhRh(a1[2], Q6_V_lo_W(h1), r1);
            a1[3] = Q6_Ww_vmpyacc_WwVhRh(a1[3], Q6_V_hi_W(h1), r1);
        }
        for (int h = 0; h < 2; ++h) {
            HVX_Vector *bo = blk + (r + h) * 8;
            for (int i = 0; i < 4; ++i) {
                const HVX_VectorPair pr = h ? a1[i] : a0[i];
                const HVX_VectorPair o = Q6_W_vshuff_VVR(Q6_V_hi_W(pr), Q6_V_lo_W(pr), -4);
                bo[2 * i] = vmul(Q6_Vsf_equals_Vw(Q6_V_lo_W(o)), back);
                bo[2 * i + 1] = vmul(Q6_Vsf_equals_Vw(Q6_V_hi_W(o)), back);
            }
        }
    }
}

// the attention blocks of the step: task -> (token, KV head, block); a token's blocks cover its positions 0..pos+t
static int attn_blocks(const dec_t *d, int t) { return (d->pos + t + 1 + DEC_BLOCK - 1) / DEC_BLOCK; }
static int attn_tasks(const dec_t *d) {
    int n = 0;
    for (int t = 0; t < d->T; ++t) n += d->c.nkv * attn_blocks(d, t);
    return n;
}

// the scalar fallback: token t, KV head kh, block b
static void attn_block_scalar(dec_t *d, const dec_layer_t *L, int t, int kh, int b) {
    const dec_cfg_t *c = &d->c;
    const int np = d->pos + t + 1, nb = attn_blocks(d, t), G = c->nh / c->nkv, hd = c->hd;
    const int p0 = b * DEC_BLOCK, p1 = p0 + DEC_BLOCK < np ? p0 + DEC_BLOCK : np;
    const float scale = 1.0f / sqrtf((float)hd);
    float sc[DEC_BLOCK];
    for (int g = 0; g < G; ++g) {
        const int h = kh * G + g;
        const float *q = d->qb[t] + (size_t)h * 2 * hd;
        float mx = -INFINITY;
        for (int p = p0; p < p1; ++p) {
            const void *k = kv_row(d, p, L->index, 0);
            float s = 0.0f;
            if (c->kv_i8) {
                const int8_t *k8 = (const int8_t *)k + kh * hd;
                for (int i = 0; i < hd; ++i) s += q[i] * (float)k8[i];
                s *= kv_scale(d, p, L->index, 0)[kh];
            } else {
                for (int i = 0; i < hd; ++i) s += q[i] * f16f(((const uint16_t *)k)[kh * hd + i]);
            }
            s *= scale;
            sc[p - p0] = s;
            mx = s > mx ? s : mx;
        }
        float sum = 0.0f;
        for (int p = p0; p < p1; ++p) sum += (sc[p - p0] = expf(sc[p - p0] - mx));
        float *out = d->part + (((size_t)t * c->nh + (size_t)h) * (size_t)nb + (size_t)b) * (size_t)(hd + 2);
        for (int i = 0; i < hd; ++i) out[i] = 0.0f;
        for (int p = p0; p < p1; ++p) {
            const void *v = kv_row(d, p, L->index, 1);
            if (c->kv_i8) {
                const int8_t *v8 = (const int8_t *)v + kh * hd;
                const float wgt = sc[p - p0] * kv_scale(d, p, L->index, 1)[kh];
                for (int i = 0; i < hd; ++i) out[i] += wgt * (float)v8[i];
            } else {
                const float wgt = sc[p - p0];
                for (int i = 0; i < hd; ++i) out[i] += wgt * f16f(((const uint16_t *)v)[kh * hd + i]);
            }
        }
        out[hd] = mx;
        out[hd + 1] = sum;
    }
}

// Token t, KV head kh, block b of its positions, for the KV head's query heads: per head the block's maximum,
// weights' sum and weighted values into part. The block's K and V rows prefetched into L2; scores 32 positions at a
// time (tree32); the values two query heads at a time, their sums in registers
static void attn_block(dec_t *d, const dec_layer_t *L, int task, int th) {
    const dec_cfg_t *c = &d->c;
    int t = 0;
    while (task >= c->nkv * attn_blocks(d, t)) task -= c->nkv * attn_blocks(d, t++);
    const int nb = attn_blocks(d, t), kh = task / nb, b = task % nb;
    const int G = c->nh / c->nkv, hd = c->hd, nvec = hd / 32;
    if (!c->kv_i8 || hd != 256 || G * DEC_BLOCK > DEC_TBUF || G % 2 || G > 8 ||
        (size_t)G * (size_t)(nvec + 32) * 128u > DEC_VEC_AREA || (size_t)DEC_BLOCK * hd > DEC_TILE_MAX ||
        (size_t)(G * 20 + 64) * 128u > DEC_VEC_AREA || d->tree_ok != 1 || (d->flags & DEC_DEBUG_ATTN1)) {
        attn_block_scalar(d, L, t, kh, b);
        return;
    }
    if (!(d->flags & DEC_DEBUG_ATTN_F)) {
        // the int16 q / int8 K / int16 weights way (attn_rows): the block's G rows, every position theirs
        const int np = d->pos + t + 1, p0 = b * DEC_BLOCK, n = np - p0 < DEC_BLOCK ? np - p0 : DEC_BLOCK;
        const uint32_t kvr = (uint32_t)(c->nkv * hd);
        const int8_t *krows = (const int8_t *)kv_row(d, p0, L->index, 0) + kh * hd;
        const int8_t *vrows = (const int8_t *)kv_row(d, p0, L->index, 1) + kh * hd;
        const uint64_t pf = (uint64_t)kvr << 32 | (uint64_t)(uint32_t)hd << 16 | (uint64_t)(uint32_t)n;
        asm volatile(" l2fetch(%0,%1) " : : "r"(krows), "r"(pf));
        asm volatile(" l2fetch(%0,%1) " : : "r"(vrows), "r"(pf));
        HVX_Vector *qv = (HVX_Vector *)(d->vtcm + (size_t)th * DEC_THREAD_VTCM + 2u * DEC_TILE_MAX);
        HVX_Vector *sv = qv + G * 4, *part = sv + G * 8, *blk = part + 64;
        float qsc[8], bm[8], bs[8];
        int lim[8];
        for (int g = 0; g < G; ++g) {
            q_split(d->qb[t] + (size_t)(kh * G + g) * 2 * hd, qv + g * 4, &qsc[g]);
            lim[g] = n;
        }
        attn_rows(d, krows, vrows, kvr, kv_scale(d, p0, L->index, 0) + kh, kv_scale(d, p0, L->index, 1) + kh, c->nkv,
                  n, G, lim, qv, qsc, sv, part, (int32_t *)d->tbuf[th], bm, bs, blk);
        for (int g = 0; g < G; ++g) {
            float *out = d->part + (((size_t)t * c->nh + (size_t)(kh * G + g)) * (size_t)nb + (size_t)b) * (size_t)(hd + 2);
            memcpy(out, blk + g * 8, (size_t)hd * sizeof(float));
            out[hd] = bm[g];
            out[hd + 1] = bs[g];
        }
        return;
    }
    const int np = d->pos + t + 1, p0 = b * DEC_BLOCK, p1 = p0 + DEC_BLOCK < np ? p0 + DEC_BLOCK : np, n = p1 - p0;
    // The block's K and V rows into the L2 cache ahead of the loops (the DMA engine refuses the cache's memory: it is
    // the CPU's cached rpcmem): rows of hd bytes, a position's row (its KV heads side by side) apart
    const uint32_t kvr = (uint32_t)(c->nkv * hd);
    const int8_t *krows = (const int8_t *)kv_row(d, p0, L->index, 0) + kh * hd;
    const int8_t *vrows = (const int8_t *)kv_row(d, p0, L->index, 1) + kh * hd;
    const uint64_t pf = (uint64_t)kvr << 32 | (uint64_t)(uint32_t)hd << 16 | (uint64_t)(uint32_t)n;
    asm volatile(" l2fetch(%0,%1) " : : "r"(krows), "r"(pf));
    asm volatile(" l2fetch(%0,%1) " : : "r"(vrows), "r"(pf));
    HVX_Vector *qv = (HVX_Vector *)(d->vtcm + (size_t)th * DEC_THREAD_VTCM + 2u * DEC_TILE_MAX);
    HVX_Vector *part = qv + G * nvec;   // [G][32]
    const HVX_Vector scale = vsf(1.0f / sqrtf((float)hd));
    for (int g = 0; g < G; ++g) {
        const HVX_Vector *q = (const HVX_Vector *)(d->qb[t] + (size_t)(kh * G + g) * 2 * hd);
        for (int i = 0; i < nvec; ++i) qv[g * nvec + i] = vmul(q[i], scale);
    }
    const float *ksc = kv_scale(d, p0, L->index, 0) + kh, *vsc = kv_scale(d, p0, L->index, 1) + kh;
    const int nkv = c->nkv;
    float *sc = d->tbuf[th];   // [G][DEC_BLOCK]
    // the scores: 32 positions' partial sums per head, then summed all at once
    for (int p = 0; p < n; p += 32) {
        const int m = n - p < 32 ? n - p : 32;
        for (int j = 0; j < m; ++j) {
            const HVX_Vector *k8 = (const HVX_Vector *)(krows + (size_t)(p + j) * kvr);
            HVX_Vector kf[8];
            i8_sf(k8[0], kf);
            i8_sf(k8[1], kf + 4);
            for (int g = 0; g < G; ++g) {
                const HVX_Vector *q = qv + g * nvec;
                HVX_Vector a = vmul(kf[0], q[0]);
                for (int i = 1; i < nvec; ++i) a = vadd(a, vmul(kf[i], q[i]));
                part[g * 32 + j] = a;
            }
        }
        for (int g = 0; g < G; ++g) {
            for (int j = m; j < 32; ++j) part[g * 32 + j] = Q6_V_vzero();
            HVX_Vector r __attribute__((aligned(128))) = tree32(part + g * 32);
            const float *f = (const float *)&r;
            float *sg = sc + g * DEC_BLOCK + p;
            for (int l = 0; l < 32; ++l) {
                const int j = d->tree_perm[l];
                if (j < m) sg[j] = f[l] * ksc[(size_t)(p + j) * nkv];
            }
        }
    }
    // per head: the maximum, the weights exp(s - max) (32 at a time; lanes past n left out), their sum, and the
    // weights times each position's V scale
    float mx[8], sum[8];
    for (int g = 0; g < G; ++g) {
        float *sg = sc + g * DEC_BLOCK;
        float mxg = -INFINITY;
        for (int i = 0; i < n; ++i) mxg = sg[i] > mxg ? sg[i] : mxg;
        for (int i = 0; i < n; i += 32) *(HVX_Vector *)(sg + i) = vexp(vsub(*(const HVX_Vector *)(sg + i), vsf(mxg)));
        float s = 0.0f;
        for (int i = 0; i < n; ++i) {
            s += sg[i];
            sg[i] *= vsc[(size_t)i * nkv];
        }
        mx[g] = mxg;
        sum[g] = s;
    }
    // the values, two query heads at a time: their 16 sums stay in registers
    for (int g = 0; g < G; g += 2) {
        HVX_Vector a0[8], a1[8];
        for (int i = 0; i < 8; ++i) a0[i] = a1[i] = Q6_V_vzero();
        const float *w0 = sc + g * DEC_BLOCK, *w1 = w0 + DEC_BLOCK;
        for (int p = 0; p < n; ++p) {
            const HVX_Vector *v8 = (const HVX_Vector *)(vrows + (size_t)p * kvr);
            HVX_Vector vf[8];
            i8_sf(v8[0], vf);
            i8_sf(v8[1], vf + 4);
            const HVX_Vector x0 = vsf(w0[p]), x1 = vsf(w1[p]);
            for (int i = 0; i < 8; ++i) {
                a0[i] = vadd(a0[i], vmul(vf[i], x0));
                a1[i] = vadd(a1[i], vmul(vf[i], x1));
            }
        }
        for (int h = 0; h < 2; ++h) {
            float *out = d->part + (((size_t)t * c->nh + (size_t)(kh * G + g + h)) * (size_t)nb + (size_t)b) *
                                       (size_t)(hd + 2);
            HVX_Vector tmp[8] __attribute__((aligned(128)));
            for (int i = 0; i < 8; ++i) tmp[i] = h ? a1[i] : a0[i];
            memcpy(out, tmp, (size_t)hd * sizeof(float));
            out[hd] = mx[g + h];
            out[hd + 1] = sum[g + h];
        }
    }
}

// query head h of token t (task t * nh + h): its blocks joined, gated, quantized for o
static void attn_join(dec_t *d, int task) {
    const dec_cfg_t *c = &d->c;
    const int t = task / c->nh, h = task % c->nh, nb = attn_blocks(d, t), hd = c->hd;
    const float *pp = d->part + ((size_t)t * c->nh + (size_t)h) * (size_t)nb * (size_t)(hd + 2);
    float mx = -INFINITY;
    for (int b = 0; b < nb; ++b) mx = pp[(size_t)b * (hd + 2) + hd] > mx ? pp[(size_t)b * (hd + 2) + hd] : mx;
    float *o = d->core[t] + (size_t)h * hd;
    for (int i = 0; i < hd; ++i) o[i] = 0.0f;
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float *q = pp + (size_t)b * (hd + 2), f = expf(q[hd] - mx);
        sum += q[hd + 1] * f;
        for (int i = 0; i < hd; ++i) o[i] += q[i] * f;
    }
    const float inv = 1.0f / sum, *gate = d->qb[t] + (size_t)h * 2 * hd + hd;
    for (int i = 0; i < hd; ++i) o[i] = o[i] * inv * sigmoidf(gate[i]);
    for (int g = h * hd / 64; g < (h + 1) * hd / 64; ++g) quant(d->core[t] + 64 * g, &d->act_c[t], g);
}

// ---- a prompt's pieces (dec_prefill: up to DEC_N tokens, its products on HMX) --------------------------------------

// HMX's layouts: element (row r, column c) of a matrix of `cols` columns in 32 x 32 tiles of fp16, one tile after
// another (row tile major), inside one: (r % 32 / 2) * 64 + (c % 32) * 2 + r % 2
static inline size_t hmx_at(int r, int c, int cols) {
    return ((size_t)(r / 32) * (size_t)(cols / 32) + (size_t)(c / 32)) * 1024u + (size_t)(r % 32 / 2) * 64u +
           (size_t)(c % 32) * 2u + (size_t)(r % 2);
}

// rows[t][0..k) for t < n into HMX's row tiles (m = n rounded up to 32: the rows past n zero); a task: a tile's 32
// columns
static void to_tiles(dec_t *d, int st, float *const *rows, int n, int k, uint8_t *dst) {
    const int m = (n + 31) / 32 * 32, tasks = (m / 32) * (k / 32);
    __fp16 *a = (__fp16 *)dst;
    const int cv = d->cvt_to;
    for (int task = take(d, st); task < tasks; task = take(d, st)) {
        const int tr = task / (k / 32), tc = task % (k / 32);
        __fp16 *out = a + ((size_t)tr * (size_t)(k / 32) + (size_t)tc) * 1024u;
        for (int rp = 0; rp < 16; ++rp) {
            const int r0 = tr * 32 + 2 * rp, r1 = r0 + 1;
            const float *x0 = r0 < n ? rows[r0] + tc * 32 : NULL, *x1 = r1 < n ? rows[r1] + tc * 32 : NULL;
            __fp16 *o = out + rp * 64;
            if (cv > 0) {   // the pair's 32 columns, one vcvt
                const HVX_Vector v0 = x0 ? *(const HVX_Vector *)x0 : Q6_V_vzero();
                const HVX_Vector v1 = x1 ? *(const HVX_Vector *)x1 : Q6_V_vzero();
                *(HVX_Vector *)o = cv == 1 ? Q6_Vhf_vcvt_VsfVsf(v0, v1) : Q6_Vhf_vcvt_VsfVsf(v1, v0);
                continue;
            }
            for (int q = 0; q < 32; ++q) {
                o[2 * q] = x0 ? (__fp16)x0[q] : (__fp16)0.0f;
                o[2 * q + 1] = x1 ? (__fp16)x1[q] : (__fp16)0.0f;
            }
        }
    }
}

// HMX's result croutons (m rows, cols columns) into rows[t][0..cols) for t < n; a task: a tile's 32 columns
static void from_tiles(dec_t *d, int st, const uint8_t *src, int n, int cols, float *const *rows) {
    const int m = (n + 31) / 32 * 32, tasks = (m / 32) * (cols / 32);
    const __fp16 *o = (const __fp16 *)src;
    const int cv = d->cvt_from;
    for (int task = take(d, st); task < tasks; task = take(d, st)) {
        const int tr = task / (cols / 32), tc = task % (cols / 32);
        const __fp16 *in = o + ((size_t)tr * (size_t)(cols / 32) + (size_t)tc) * 1024u;
        for (int rp = 0; rp < 16; ++rp) {
            const int r0 = tr * 32 + 2 * rp, r1 = r0 + 1;
            const __fp16 *q = in + rp * 64;
            if (cv > 0) {
                const HVX_VectorPair w = Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)q);
                if (r0 < n) *(HVX_Vector *)(rows[r0] + tc * 32) = cv == 1 ? Q6_V_lo_W(w) : Q6_V_hi_W(w);
                if (r1 < n) *(HVX_Vector *)(rows[r1] + tc * 32) = cv == 1 ? Q6_V_hi_W(w) : Q6_V_lo_W(w);
                continue;
            }
            if (r0 < n)
                for (int c = 0; c < 32; ++c) rows[r0][tc * 32 + c] = (float)q[2 * c];
            if (r1 < n)
                for (int c = 0; c < 32; ++c) rows[r1][tc * 32 + c] = (float)q[2 * c + 1];
        }
    }
}

// x + add into xo and normed with w into fin, every token's row (a task: a token's 64 values)
static void norm_rows(dec_t *d, int st, float *const *x, float *const *add, float *const *xo, uint32_t w_at) {
    const dec_cfg_t *c = &d->c;
    const int H = c->hidden, T = d->T, groups = H / 64;
    const HVX_UVector *w = (const HVX_UVector *)pf(d, w_at);
    for (int task = take(d, st); task < T; task = take(d, st)) {
        const HVX_Vector *xv = (const HVX_Vector *)x[task], *av = add ? (const HVX_Vector *)add[task] : NULL;
        HVX_Vector *ov = xo ? (HVX_Vector *)xo[task] : NULL, *fv = (HVX_Vector *)d->fin[task];
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        for (int i = 0; i < groups * 2; i += 2) {
            const HVX_Vector v0 = av ? vadd(xv[i], av[i]) : xv[i], v1 = av ? vadd(xv[i + 1], av[i + 1]) : xv[i + 1];
            s0 = vadd(s0, vmul(v0, v0));
            s1 = vadd(s1, vmul(v1, v1));
            fv[i] = v0;
            fv[i + 1] = v1;
        }
        const HVX_Vector inv = vsf(1.0f / sqrtf(hsum(vadd(s0, s1)) / (float)H + c->eps)), one = vsf(1.0f);
        for (int i = 0; i < groups * 2; ++i) {
            if (ov) ov[i] = fv[i];
            fv[i] = vmul(vmul(fv[i], inv), vadd(w[i], one));
        }
    }
}

// one value head over the prompt's tokens, token by token (the state stays in the cache for all of them): q / k
// normalised, the gated delta rule, the gated RMSNorm into core
static void gdn_head_n(dec_t *d, const dec_layer_t *L, int vh, int th) {
    const dec_cfg_t *c = &d->c;
    const int nk = c->nk, nv = c->nv, dk = c->dk, dv = c->dv, kd = nk * dk, kh = vh / (nv / nk), T = d->T;
    float *state = d->gdn_live + ((size_t)L->index * nv + vh) * dk * dv;
    HVX_Vector *kv = (HVX_Vector *)(d->vtcm + (size_t)th * DEC_THREAD_VTCM + 2u * DEC_TILE_MAX), *qv = kv + dk / 2;
    const HVX_VectorPred half = Q6_Q_vsetq_R(64);
    const float *gw = pf(d, L->gnorm), A = expf(pf(d, L->A_log)[vh]), dtb = pf(d, L->dt_bias)[vh];
    for (int t = 0; t < T; ++t) {
        const float *mixed = d->proj[t];
        const float *qs = mixed + kh * dk, *ks = mixed + kd + kh * dk, *vs = mixed + 2 * kd + vh * dv;
        float sq = 0.0f, sk = 0.0f;
        if (dk % 32 == 0) {
            HVX_Vector aq = Q6_V_vzero(), ak = Q6_V_vzero();
            for (int j = 0; j < dk; j += 32) {
                const HVX_Vector q = *(const HVX_Vector *)(qs + j), k = *(const HVX_Vector *)(ks + j);
                aq = vadd(aq, vmul(q, q));
                ak = vadd(ak, vmul(k, k));
            }
            sq = hsum(aq);
            sk = hsum(ak);
        } else
            for (int j = 0; j < dk; ++j) {
                sq += qs[j] * qs[j];
                sk += ks[j] * ks[j];
            }
        const float iq = 1.0f / sqrtf(sq + 1e-6f) / sqrtf((float)dk), ik = 1.0f / sqrtf(sk + 1e-6f);
        for (int r = 0; r < dk / 2; ++r) {
            kv[r] = Q6_V_vmux_QVV(half, vsf(ks[2 * r] * ik), vsf(ks[2 * r + 1] * ik));
            qv[r] = Q6_V_vmux_QVV(half, vsf(qs[2 * r] * iq), vsf(qs[2 * r + 1] * iq));
        }
        const float beta = sigmoidf(d->ab[t][nv + vh]), decay = expf(-A * softplusf(d->ab[t][vh] + dtb));
        const HVX_Vector dec = vsf(decay);
        float *o = d->core[t] + vh * dv;
        for (int j0 = 0; j0 < dv; j0 += 16) {
            HVX_Vector *S = (HVX_Vector *)(state + (size_t)j0 * dk);
            HVX_Vector acc = Q6_V_vzero();
            for (int r = 0; r < dk / 2; ++r) {
                const HVX_Vector v = vmul(S[r], dec);
                S[r] = v;
                acc = vadd(acc, vmul(v, kv[r]));
            }
            HVX_Vector mv __attribute__((aligned(128))) = fold16(acc);
            const float *mem = (const float *)&mv;
            float dl[32] __attribute__((aligned(128)));
            for (int j = 0; j < 16; ++j) dl[j] = dl[j + 16] = (vs[j0 + j] - mem[j]) * beta;
            const HVX_Vector dd = *(const HVX_Vector *)dl;
            HVX_Vector ao = Q6_V_vzero();
            for (int r = 0; r < dk / 2; ++r) {
                const HVX_Vector v = vadd(S[r], vmul(kv[r], dd));
                S[r] = v;
                ao = vadd(ao, vmul(v, qv[r]));
            }
            HVX_Vector ov __attribute__((aligned(128))) = fold16(ao);
            memcpy(o + j0, &ov, 16 * sizeof(float));
        }
        const float *z = d->zb[t] + vh * dv;
        float ss = 0.0f;
        if (dv % 32 == 0) {
            HVX_Vector a = Q6_V_vzero();
            for (int j = 0; j < dv; j += 32) a = vadd(a, vmul(*(const HVX_Vector *)(o + j), *(const HVX_Vector *)(o + j)));
            ss = hsum(a);
        } else
            for (int j = 0; j < dv; ++j) ss += o[j] * o[j];
        const float inv = 1.0f / sqrtf(ss / (float)dv + c->eps);
        if (dv % 32 == 0)
            for (int j = 0; j < dv; j += 32) {
                const HVX_Vector zv = *(const HVX_Vector *)(z + j), ov = *(const HVX_Vector *)(o + j);
                HVX_Vector gwv;
                memcpy(&gwv, gw + j, 128);
                *(HVX_Vector *)(o + j) = vmul(vmul(vmul(ov, vsf(inv)), gwv), vsilu(zv));
            }
        else
            for (int j = 0; j < dv; ++j) o[j] = o[j] * inv * gw[j] * siluf(z[j]);
    }
}

// Token t's attention, KV head kh (task t * nkv + kh), over all its positions 0..pos+t a block at a time: the scores
// (32 positions at once, tree32), an online softmax (the running maximum rescales what was summed), the values; then
// each query head's output gated into core. The rows come through L2 (l2fetch)
static void attn_token(dec_t *d, const dec_layer_t *L, int task, int th) {
    const dec_cfg_t *c = &d->c;
    const int t = task / c->nkv, kh = task % c->nkv, G = c->nh / c->nkv, hd = c->hd, nvec = hd / 32, nkv = c->nkv;
    const int np = d->pos + t + 1;
    HVX_Vector *qv = (HVX_Vector *)(d->vtcm + (size_t)th * DEC_THREAD_VTCM + 2u * DEC_TILE_MAX);
    HVX_Vector *part = qv + G * nvec, *acc = part + G * 32;   // [G][32] score sums, [G][nvec] value sums
    const HVX_Vector scale = vsf(1.0f / sqrtf((float)hd));
    for (int g = 0; g < G; ++g) {
        const HVX_Vector *q = (const HVX_Vector *)(d->qb[t] + (size_t)(kh * G + g) * 2 * hd);
        for (int i = 0; i < nvec; ++i) {
            qv[g * nvec + i] = vmul(q[i], scale);
            acc[g * nvec + i] = Q6_V_vzero();
        }
    }
    float mx[8], sum[8];
    for (int g = 0; g < G; ++g) mx[g] = -INFINITY, sum[g] = 0.0f;
    float *sc = d->tbuf[th];   // [G][DEC_BLOCK]
    const uint32_t kvr = (uint32_t)(nkv * hd);
    for (int p0 = 0; p0 < np; p0 += DEC_BLOCK) {
        const int n = np - p0 < DEC_BLOCK ? np - p0 : DEC_BLOCK;
        const int8_t *krows = (const int8_t *)kv_row(d, p0, L->index, 0) + kh * hd;
        const int8_t *vrows = (const int8_t *)kv_row(d, p0, L->index, 1) + kh * hd;
        const uint64_t pfc = (uint64_t)kvr << 32 | (uint64_t)(uint32_t)hd << 16 | (uint64_t)(uint32_t)n;
        asm volatile(" l2fetch(%0,%1) " : : "r"(krows), "r"(pfc));
        asm volatile(" l2fetch(%0,%1) " : : "r"(vrows), "r"(pfc));
        const float *ksc = kv_scale(d, p0, L->index, 0) + kh, *vsc = kv_scale(d, p0, L->index, 1) + kh;
        for (int p = 0; p < n; p += 32) {
            const int m = n - p < 32 ? n - p : 32;
            for (int j = 0; j < m; ++j) {
                const HVX_Vector *k8 = (const HVX_Vector *)(krows + (size_t)(p + j) * kvr);
                HVX_Vector kf[8];
                i8_sf(k8[0], kf);
                i8_sf(k8[1], kf + 4);
                for (int g = 0; g < G; ++g) {
                    const HVX_Vector *q = qv + g * nvec;
                    HVX_Vector a = vmul(kf[0], q[0]);
                    for (int i = 1; i < nvec; ++i) a = vadd(a, vmul(kf[i], q[i]));
                    part[g * 32 + j] = a;
                }
            }
            for (int g = 0; g < G; ++g) {
                for (int j = m; j < 32; ++j) part[g * 32 + j] = Q6_V_vzero();
                HVX_Vector r __attribute__((aligned(128))) = tree32(part + g * 32);
                const float *f = (const float *)&r;
                float *sg = sc + g * DEC_BLOCK + p;
                for (int l = 0; l < 32; ++l) {
                    const int j = d->tree_perm[l];
                    if (j < m) sg[j] = f[l] * ksc[(size_t)(p + j) * nkv];
                }
            }
        }
        // the online softmax: the block's maximum, the sums so far rescaled to it
        for (int g = 0; g < G; ++g) {
            float *sg = sc + g * DEC_BLOCK;
            float bm = mx[g];
            for (int i = 0; i < n; ++i) bm = sg[i] > bm ? sg[i] : bm;
            const float r = mx[g] == -INFINITY ? 0.0f : expf(mx[g] - bm);
            if (r != 1.0f)
                for (int i = 0; i < nvec; ++i) acc[g * nvec + i] = vmul(acc[g * nvec + i], vsf(r));
            sum[g] *= r;
            mx[g] = bm;
            for (int i = 0; i < n; i += 32) *(HVX_Vector *)(sg + i) = vexp(vsub(*(const HVX_Vector *)(sg + i), vsf(bm)));
            for (int i = 0; i < n; ++i) {
                sum[g] += sg[i];
                sg[i] *= vsc[(size_t)i * nkv];
            }
        }
        for (int g = 0; g < G; g += 2) {
            HVX_Vector a0[8], a1[8];
            for (int i = 0; i < 8; ++i) a0[i] = acc[g * nvec + i], a1[i] = acc[(g + 1) * nvec + i];
            const float *w0 = sc + g * DEC_BLOCK, *w1 = w0 + DEC_BLOCK;
            for (int p = 0; p < n; ++p) {
                const HVX_Vector *v8 = (const HVX_Vector *)(vrows + (size_t)p * kvr);
                HVX_Vector vf[8];
                i8_sf(v8[0], vf);
                i8_sf(v8[1], vf + 4);
                const HVX_Vector x0 = vsf(w0[p]), x1 = vsf(w1[p]);
                for (int i = 0; i < 8; ++i) {
                    a0[i] = vadd(a0[i], vmul(vf[i], x0));
                    a1[i] = vadd(a1[i], vmul(vf[i], x1));
                }
            }
            for (int i = 0; i < 8; ++i) acc[g * nvec + i] = a0[i], acc[(g + 1) * nvec + i] = a1[i];
        }
    }
    for (int g = 0; g < G; ++g) {
        const int h = kh * G + g;
        float *o = d->core[t] + (size_t)h * hd;
        const float *gate = d->qb[t] + (size_t)h * 2 * hd + hd;
        HVX_Vector tmp[8] __attribute__((aligned(128)));
        for (int i = 0; i < nvec; ++i) tmp[i] = acc[g * nvec + i];
        const float *f = (const float *)tmp, inv = 1.0f / sum[g];
        for (int i = 0; i < hd; ++i) o[i] = f[i] * inv * sigmoidf(gate[i]);
    }
}

// A prompt's attention for KV head kh and a group of tokens (task kh * groups + group): its rows - (token, query
// head), ATT_ROWS of them - share each K / V block's reads (attn_rows), over every position the group's last token
// sees; the blocks merged as they come (the running maximum rescales what was summed). Then every row's output,
// gated, into core
static void attn_group(dec_t *d, const dec_layer_t *L, int task, int th) {
    const dec_cfg_t *c = &d->c;
    const int T = d->T, G = c->nh / c->nkv, hd = c->hd, tg = ATT_ROWS / G, groups = (T + tg - 1) / tg;
    const int kh = task / groups, t0 = (task % groups) * tg, nt = T - t0 < tg ? T - t0 : tg, R = nt * G;
    const int np_max = d->pos + t0 + nt;   // the positions the group's last token sees
    HVX_Vector *accf = (HVX_Vector *)(d->vtcm + (size_t)th * DEC_THREAD_VTCM);   // [R][8] the float sums
    HVX_Vector *qv = accf + ATT_ROWS * 8, *sv = qv + ATT_ROWS * 4, *blk = sv + ATT_ROWS * 8, *part = blk + ATT_ROWS * 8;
    float qsc[ATT_ROWS], mx[ATT_ROWS], sm[ATT_ROWS], bm[ATT_ROWS], bs[ATT_ROWS];
    int lim[ATT_ROWS];
    for (int r = 0; r < R; ++r) {
        q_split(d->qb[t0 + r / G] + (size_t)(kh * G + r % G) * 2 * hd, qv + r * 4, &qsc[r]);
        for (int i = 0; i < 8; ++i) accf[r * 8 + i] = Q6_V_vzero();
        mx[r] = -INFINITY;
        sm[r] = 0.0f;
    }
    const uint32_t kvr = (uint32_t)(c->nkv * hd);
    for (int p0 = 0; p0 < np_max; p0 += DEC_BLOCK) {
        const int n = np_max - p0 < DEC_BLOCK ? np_max - p0 : DEC_BLOCK;
        const int8_t *krows = (const int8_t *)kv_row(d, p0, L->index, 0) + kh * hd;
        const int8_t *vrows = (const int8_t *)kv_row(d, p0, L->index, 1) + kh * hd;
        const uint64_t pfc = (uint64_t)kvr << 32 | (uint64_t)(uint32_t)hd << 16 | (uint64_t)(uint32_t)n;
        asm volatile(" l2fetch(%0,%1) " : : "r"(krows), "r"(pfc));
        asm volatile(" l2fetch(%0,%1) " : : "r"(vrows), "r"(pfc));
        for (int r = 0; r < R; ++r) lim[r] = d->pos + t0 + r / G + 1 - p0;   // the token's positions here
        attn_rows(d, krows, vrows, kvr, kv_scale(d, p0, L->index, 0) + kh, kv_scale(d, p0, L->index, 1) + kh, c->nkv,
                  n, R, lim, qv, qsc, sv, part, (int32_t *)d->tbuf[th], bm, bs, blk);
        for (int r = 0; r < R; ++r) {
            if (bm[r] == -INFINITY) continue;
            const float m = bm[r] > mx[r] ? bm[r] : mx[r];
            const float a = mx[r] == -INFINITY ? 0.0f : expf(mx[r] - m), e = expf(bm[r] - m);
            const HVX_Vector av = vsf(a), ev = vsf(e);
            for (int i = 0; i < 8; ++i) accf[r * 8 + i] = vadd(vmul(accf[r * 8 + i], av), vmul(blk[r * 8 + i], ev));
            sm[r] = sm[r] * a + bs[r] * e;
            mx[r] = m;
        }
    }
    for (int r = 0; r < R; ++r) {
        const int t = t0 + r / G, h = kh * G + r % G;
        float *o = d->core[t] + (size_t)h * hd;
        const float *gate = d->qb[t] + (size_t)h * 2 * hd + hd;
        HVX_Vector tmp[8] __attribute__((aligned(128)));
        for (int i = 0; i < 8; ++i) tmp[i] = accf[r * 8 + i];
        const float *f = (const float *)tmp, inv = 1.0f / sm[r];
        for (int i = 0; i < hd; ++i) o[i] = f[i] * inv * sigmoidf(gate[i]);
    }
}

enum { PF_IN, PF_GDN, PF_ATT, PF_POST, PF_MLP, PF_DOWN, PF_FINAL, PF_MTP_IN, PF_MTP_X, PF_MTP_KV };

// n values normed in place with w (x / rms * (1 + w))
static void norm_seg(float *x, int n, const float *w, float eps) {
    HVX_Vector *xv = (HVX_Vector *)x, s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
    const HVX_UVector *wv = (const HVX_UVector *)w;
    for (int i = 0; i < n / 32; i += 2) {
        s0 = vadd(s0, vmul(xv[i], xv[i]));
        s1 = vadd(s1, vmul(xv[i + 1], xv[i + 1]));
    }
    const HVX_Vector inv = vsf(1.0f / sqrtf(hsum(vadd(s0, s1)) / (float)n + eps)), one = vsf(1.0f);
    for (int i = 0; i < n / 32; ++i) xv[i] = vmul(vmul(xv[i], inv), vadd(wv[i], one));
}

// the prompt call's phases, on every worker; between them the caller multiplies on HMX
static void prefill_body(dec_t *d, int th) {
    const dec_cfg_t *c = &d->c;
    const dec_layer_t *L = d->pl;
    const int T = d->T, H = c->hidden;
    int sense = d->sense, st = 0;
    if (th == 0 && d->tree_ok == 0) d->tree_ok = tree_check(d) ? 1 : -1;
    if (th == 0 && d->cvt_to == 0) cvt_check(d);
    switch (d->phase) {
    case PF_IN: {
        // x (+ the last MLP) into xo, normed; as HMX's tiles; a GDN layer's a / b rows
        norm_rows(d, st++, d->rx, d->radd, d->rxo, L->in_norm);
        bar(d, &sense);
        if (th == 0) {
            float **t = d->rx;
            d->rx = d->rxo;
            d->rxo = t;
        }
        to_tiles(d, st++, d->fin, T, H, d->tiles_a);
        if (!L->attn) {
            const int s2 = st++;
            for (int k = take(d, s2); k < T * 2 * c->nv; k = take(d, s2)) ab_row(d, L, k);
        }
        bar(d, &sense);
        mark(d, th, DEC_PROF_NORM);
        break;
    }
    case PF_GDN: {
        const int kd = c->nk * c->dk, vd = c->nv * c->dv, conv_dim = 2 * kd + vd;
        from_tiles(d, st++, d->tiles_o[0], T, conv_dim, d->proj);
        from_tiles(d, st++, d->tiles_o[1], T, vd, d->zb);
        bar(d, &sense);
        const int s1 = st++;
        for (int g = take(d, s1); g < (conv_dim + 63) / 64; g = take(d, s1)) conv_task(d, L, g);
        bar(d, &sense);
        mark(d, th, DEC_PROF_CONV);
        const int s2 = st++;
        for (int h = take(d, s2); h < c->nv; h = take(d, s2)) gdn_head_n(d, L, h, th);
        bar(d, &sense);
        mark(d, th, DEC_PROF_GDN);
        to_tiles(d, st++, d->core, T, vd, d->tiles_a);
        bar(d, &sense);
        break;
    }
    case PF_ATT: {
        from_tiles(d, st++, d->tiles_o[0], T, c->nh * 2 * c->hd, d->qb);
        from_tiles(d, st++, d->tiles_o[1], T, c->nkv * c->hd, d->kb);
        from_tiles(d, st++, d->tiles_o[2], T, c->nkv * c->hd, d->vb);
        bar(d, &sense);
        const int s1 = st++;
        for (int h = take(d, s1); h < T * (c->nh + c->nkv); h = take(d, s1)) qk_task(d, L, h, th);
        bar(d, &sense);
        mark(d, th, DEC_PROF_QK);
        const int s2 = st++;
        // (the grouped attention: a KV cache of int8, hd 256, an even number of query heads a KV head up to ATT_ROWS;
        // DEC_DEBUG_ATTN1: a token at a time in floats)
        const int G = c->nh / c->nkv;
        if (c->kv_i8 && c->hd == 256 && G % 2 == 0 && G <= ATT_ROWS && ATT_ROWS * DEC_BLOCK <= DEC_TBUF &&
            (size_t)(ATT_ROWS * 28 + 64) * 128u <= DEC_THREAD_VTCM && d->tree_ok == 1 && !(d->flags & DEC_DEBUG_ATTN1)) {
            const int groups = (T + ATT_ROWS / G - 1) / (ATT_ROWS / G);
            for (int k = take(d, s2); k < c->nkv * groups; k = take(d, s2)) attn_group(d, L, k, th);
        } else {
            for (int k = take(d, s2); k < T * c->nkv; k = take(d, s2)) attn_token(d, L, k, th);
        }
        bar(d, &sense);
        mark(d, th, DEC_PROF_ATTN);
        to_tiles(d, st++, d->core, T, c->nh * c->hd, d->tiles_a);
        bar(d, &sense);
        break;
    }
    case PF_POST:
        from_tiles(d, st++, d->tiles_o[0], T, H, d->mix);
        bar(d, &sense);
        norm_rows(d, st++, d->rx, d->mix, d->rxo, L->post_norm);
        bar(d, &sense);
        if (th == 0) {
            float **t = d->rx;
            d->rx = d->rxo;
            d->rxo = t;
        }
        to_tiles(d, st++, d->fin, T, H, d->tiles_a);
        bar(d, &sense);
        mark(d, th, DEC_PROF_NORM);
        break;
    case PF_MLP: {
        from_tiles(d, st++, d->tiles_o[0], T, c->sff, d->g);
        from_tiles(d, st++, d->tiles_o[1], T, c->sff, d->u);
        bar(d, &sense);
        const int s1 = st++, groups = c->sff / 64;
        for (int k = take(d, s1); k < T * groups; k = take(d, s1)) {
            const int t = k / groups, g = k % groups;
            HVX_Vector *gv = (HVX_Vector *)(d->g[t] + 64 * g);
            const HVX_Vector *uv = (const HVX_Vector *)(d->u[t] + 64 * g);
            gv[0] = vmul(vsilu(gv[0]), uv[0]);
            gv[1] = vmul(vsilu(gv[1]), uv[1]);
        }
        bar(d, &sense);
        to_tiles(d, st++, d->g, T, c->sff, d->tiles_a);
        bar(d, &sense);
        mark(d, th, DEC_PROF_MLP);
        break;
    }
    case PF_DOWN:
        from_tiles(d, st++, d->tiles_o[0], T, H, d->mlp);
        bar(d, &sense);
        break;
    case PF_MTP_IN: {
        // [norm_e(e) | norm_h(h)] of every token (a task: a token's half), as HMX's tiles for fc
        const int s1 = st++;
        for (int k = take(d, s1); k < 2 * T; k = take(d, s1))
            norm_seg(d->cat[k / 2] + (k % 2) * H, H, pf(d, k % 2 ? c->mtp_pre_h : c->mtp_pre_e), c->eps);
        bar(d, &sense);
        to_tiles(d, st++, d->cat, T, 2 * H, d->tiles_a);
        bar(d, &sense);
        mark(d, th, DEC_PROF_MTP);
        break;
    }
    case PF_MTP_X:
        // fc's rows, the layer's input norm, as tiles for k and v
        from_tiles(d, st++, d->tiles_o[0], T, H, d->xa);
        bar(d, &sense);
        norm_rows(d, st++, d->xa, NULL, NULL, L->in_norm);
        bar(d, &sense);
        to_tiles(d, st++, d->fin, T, H, d->tiles_a);
        bar(d, &sense);
        mark(d, th, DEC_PROF_MTP);
        break;
    case PF_MTP_KV: {
        // the keys normed and turned (qk_task's KV-head tasks), into the cache with the values
        from_tiles(d, st++, d->tiles_o[1], T, c->nkv * c->hd, d->kb);
        from_tiles(d, st++, d->tiles_o[2], T, c->nkv * c->hd, d->vb);
        bar(d, &sense);
        const int s1 = st++, per = c->nh + c->nkv;
        for (int k = take(d, s1); k < T * c->nkv; k = take(d, s1)) qk_task(d, L, k / c->nkv * per + c->nh + k % c->nkv, th);
        bar(d, &sense);
        mark(d, th, DEC_PROF_MTP);
        break;
    }
    case PF_FINAL: {
        // every token's final norm (the hidden rows the MTP layer reads), the last one's logits
        norm_rows(d, st++, d->rx, d->radd, NULL, c->final_norm);
        bar(d, &sense);
        if (th == 0)
            for (int t = 0; t < T; ++t) memcpy(d->hid[t], d->fin[t], (size_t)H * sizeof(float));
        const int s1 = st++;
        for (int g = take(d, s1); g < H / 64; g = take(d, s1)) quant(d->fin[T - 1] + 64 * g, &d->act_h[0], g);
        bar(d, &sense);
        job_t jobs[1] = {{&c->head, {&d->act_h[0], NULL}, {d->logits[0], NULL}}};
        products(d, th, st++, jobs, 1, 1, 0, NULL, NULL);
        bar(d, &sense);
        mark(d, th, DEC_PROF_PRODUCTS);
        break;
    }
    }
}

// ---- the programs -------------------------------------------------------------------------------------------------

typedef struct {
    int st, sense;
    float **x, **xo;   // the residual and the next one, a row per token
} run_t;

// one decoder layer for d->T tokens: x (plus `add`, the last MLP, as it starts) through the mixer and the MLP; on
// return x holds the residual after the mixer and d->mlp the MLP's output (the next layer adds it)
static void layer(dec_t *d, int th, const dec_layer_t *L, run_t *r, float *const *add) {
    const dec_cfg_t *c = &d->c;
    const int T = d->T;
    norm_stage(d, r->st++, r->x, add, r->xo, L->in_norm, d->act_h);
    { float **t = r->x; r->x = r->xo; r->xo = t; }
    bar(d, &r->sense);
    mark(d, th, DEC_PROF_NORM);
    if (!L->attn) {
        job_t jobs[2] = {{&L->qkv, {0}, {0}}, {&L->z, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = jobs[1].a[t] = &d->act_h[t];
            jobs[0].y[t] = d->proj[t];
            jobs[1].y[t] = d->zb[t];
        }
        products(d, th, r->st++, jobs, 2, T, T * 2 * c->nv, L, ab_row);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_PRODUCTS);
        const int conv_dim = 2 * c->nk * c->dk + c->nv * c->dv, s1 = r->st++;
        for (int g = take(d, s1); g < (conv_dim + 63) / 64; g = take(d, s1)) conv_task(d, L, g);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_CONV);
        const int s2 = r->st++;
        for (int h = take(d, s2); h < c->nv; h = take(d, s2)) gdn_head(d, L, h, th);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_GDN);
    } else {
        job_t jobs[3] = {{&L->q, {0}, {0}}, {&L->k, {0}, {0}}, {&L->v, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = jobs[1].a[t] = jobs[2].a[t] = &d->act_h[t];
            jobs[0].y[t] = d->qb[t];
            jobs[1].y[t] = d->kb[t];
            jobs[2].y[t] = d->vb[t];
        }
        products(d, th, r->st++, jobs, 3, T, 0, L, NULL);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_PRODUCTS);
        const int s1 = r->st++;
        for (int h = take(d, s1); h < T * (c->nh + c->nkv); h = take(d, s1)) qk_task(d, L, h, th);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_QK);
        const int s2 = r->st++;
        if (!(d->flags & DEC_DEBUG_NO_ATTN))
            for (int k = take(d, s2); k < attn_tasks(d); k = take(d, s2)) attn_block(d, L, k, th);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_ATTN);
        const int s3 = r->st++;
        if (!(d->flags & DEC_DEBUG_NO_ATTN))
            for (int h = take(d, s3); h < T * c->nh; h = take(d, s3)) attn_join(d, h);
        bar(d, &r->sense);
        mark(d, th, DEC_PROF_JOIN);
    }
    {
        job_t jobs[1] = {{L->attn ? &L->o : &L->out, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = &d->act_c[t];
            jobs[0].y[t] = d->mix[t];
        }
        products(d, th, r->st++, jobs, 1, T, 0, NULL, NULL);
    }
    bar(d, &r->sense);
    mark(d, th, DEC_PROF_PRODUCTS);
    norm_stage(d, r->st++, r->x, d->mix, r->xo, L->post_norm, d->act_h);
    { float **t = r->x; r->x = r->xo; r->xo = t; }
    bar(d, &r->sense);
    mark(d, th, DEC_PROF_NORM);
    {
        job_t jobs[2] = {{&L->gate, {0}, {0}}, {&L->up, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = jobs[1].a[t] = &d->act_h[t];
            jobs[0].y[t] = d->g[t];
            jobs[1].y[t] = d->u[t];
        }
        products(d, th, r->st++, jobs, 2, T, 0, NULL, NULL);
    }
    bar(d, &r->sense);
    mark(d, th, DEC_PROF_PRODUCTS);
    const int s4 = r->st++, groups = c->sff / 64;
    for (int k = take(d, s4); k < T * groups; k = take(d, s4)) {
        const int t = k / groups, g = k % groups;
        HVX_Vector *gv = (HVX_Vector *)(d->g[t] + 64 * g);
        const HVX_Vector *uv = (const HVX_Vector *)(d->u[t] + 64 * g);
        gv[0] = vmul(vsilu(gv[0]), uv[0]);
        gv[1] = vmul(vsilu(gv[1]), uv[1]);
        quant(d->g[t] + 64 * g, &d->act_m[t], g);
    }
    bar(d, &r->sense);
    mark(d, th, DEC_PROF_MLP);
    {
        job_t jobs[1] = {{&L->down, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = &d->act_m[t];
            jobs[0].y[t] = d->mlp[t];
        }
        products(d, th, r->st++, jobs, 1, T, 0, NULL, NULL);
    }
    bar(d, &r->sense);
    mark(d, th, DEC_PROF_PRODUCTS);
}

// The model's step for d->T tokens (their embeddings in xa): every layer, the final norm (the normed rows kept in hid
// for the MTP layer), the head's logits per token
static void step_body(dec_t *d, int th) {
    const dec_cfg_t *c = &d->c;
    const int T = d->T, H = c->hidden;
    run_t r = {0, d->sense, d->xa, d->xb};
    if (th == 0 && d->tree_ok == 0) d->tree_ok = tree_check(d) ? 1 : -1;
    float **add = NULL;
    const int layers = (d->flags >> 8) & 0xFF ? (d->flags >> 8) & 0xFF : c->n_layers;   // DEC_DEBUG_LAYERS
    for (int l = 0; l < layers && l < c->n_layers; ++l) {
        layer(d, th, &c->layers[l], &r, add);
        add = d->mlp;
    }
    norm_stage(d, r.st++, r.x, add, r.xo, c->final_norm, d->act_h);
    bar(d, &r.sense);
    mark(d, th, DEC_PROF_NORM);
    if (th == 0)
        for (int t = 0; t < T; ++t) memcpy(d->hid[t], d->fin[t], (size_t)H * sizeof(float));
    {
        job_t jobs[1] = {{&c->head, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = &d->act_h[t];
            jobs[0].y[t] = d->logits[t];
        }
        products(d, th, r.st++, jobs, 1, T, 0, NULL, NULL);
    }
    bar(d, &r.sense);
    mark(d, th, DEC_PROF_PRODUCTS);
}

// The MTP layer's pass for d->T positions (cat[t]'s second half: the hidden rows, the first: the next tokens'
// embeddings, both as handed over): fc over [norm(e) | norm(h)], the layer, the norm of the last, the draft head;
// the most likely draft and its probability
static void mtp_body(dec_t *d, int th) {
    const dec_cfg_t *c = &d->c;
    const int T = d->T, H = c->hidden;
    run_t r = {0, d->sense, d->xa, d->xb};
    if (th == 0 && d->tree_ok == 0) d->tree_ok = tree_check(d) ? 1 : -1;
    // [norm_e(e) | norm_h(h)], quantized for fc: every thread works out the norms, quantizes its groups
    {
        const int st = r.st++, groups = 2 * H / 64;
        float inv[DEC_T][2];
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < 2; ++s) inv[t][s] = rms_inv(d->cat[t] + s * H, H, c->eps);
        for (int k = take(d, st); k < T * groups; k = take(d, st)) {
            const int t = k / groups, g = k % groups, s = 64 * g >= H;
            const float *w = pf(d, s ? c->mtp_pre_h : c->mtp_pre_e) + (64 * g - s * H);
            float *v = d->cat[t] + 64 * g;
            float h[64] __attribute__((aligned(128)));
            for (int i = 0; i < 64; ++i) h[i] = v[i] * inv[t][s] * (1.0f + w[i]);
            quant(h, &d->act_w[t], g);
        }
        bar(d, &r.sense);
    }
    {
        job_t jobs[1] = {{&c->mtp_fc, {0}, {0}}};
        for (int t = 0; t < T; ++t) {
            jobs[0].a[t] = &d->act_w[t];
            jobs[0].y[t] = d->xa[t];
        }
        products(d, th, r.st++, jobs, 1, T, 0, NULL, NULL);
        bar(d, &r.sense);
    }
    layer(d, th, &c->mtp_layer, &r, NULL);
    // the last position's norm (x + its MLP), the draft head
    {
        const int st = r.st++, t = T - 1;
        float ss = 0.0f;
        for (int i = 0; i < H; ++i) {
            const float v = r.x[t][i] + d->mlp[t][i];
            ss += v * v;
        }
        const float in = 1.0f / sqrtf(ss / (float)H + c->eps);
        const float *w = pf(d, c->mtp_norm);
        for (int g = take(d, st); g < H / 64; g = take(d, st)) {
            float h[64] __attribute__((aligned(128)));
            for (int i = 0; i < 64; ++i) h[i] = (r.x[t][64 * g + i] + d->mlp[t][64 * g + i]) * in * (1.0f + w[64 * g + i]);
            quant(h, &d->act_h[0], g);
        }
        bar(d, &r.sense);
    }
    {
        const act_t *a1[DEC_T] = {&d->act_h[0], NULL};
        job_t jobs[1] = {{&c->draft_head, {a1[0], NULL}, {d->dlog, NULL}}};
        products(d, th, r.st++, jobs, 1, 1, 0, NULL, NULL);
        bar(d, &r.sense);
    }
    // the most likely row, and its softmax probability: each thread's rows, then thread 0 joins them
    const int V = (int)c->draft_head.n, r0 = (int)((int64_t)V * th / d->nthreads),
              r1 = (int)((int64_t)V * (th + 1) / d->nthreads);
    float best = -INFINITY;
    int at = r0;
    for (int i = r0; i < r1; ++i)
        if (d->dlog[i] > best) best = d->dlog[i], at = i;
    d->best[th] = best;
    d->best_at[th] = at;
    bar(d, &r.sense);
    float mx = -INFINITY;
    for (int i = 0; i < d->nthreads; ++i) mx = d->best[i] > mx ? d->best[i] : mx;
    float s = 0.0f;
    for (int i = r0; i < r1; ++i) s += expf(d->dlog[i] - mx);
    d->bsum[th] = s;
    bar(d, &r.sense);
    if (th == 0) {
        int b = 0;
        float tot = 0.0f;
        for (int i = 0; i < d->nthreads; ++i) {
            tot += d->bsum[i];
            if (d->best[i] > d->best[b]) b = i;
        }
        const int row = d->best_at[b];
        d->draft = c->n_draft ? ((const int32_t *)(d->P + c->draft_ids))[row] : row;
        d->dprob = 1.0f / tot;   // exp(best - mx) = 1
    }
    mark(d, th, DEC_PROF_MTP);
}

static void run_body(dec_t *d, int th) {
    if (d->prog == 2) prefill_body(d, th);
    else if (d->prog == 1) mtp_body(d, th);
    else step_body(d, th);
}

// ---- the pool --------------------------------------------------------------------------------------------------------

typedef struct {
    dec_t *d;
    int th;
} worker_arg_t;
static worker_arg_t g_args[DEC_THREADS];

static void worker(void *arg) {
    worker_arg_t *a = (worker_arg_t *)arg;
    dec_t *d = a->d;
    int seen = 0;
    for (;;) {
        int go;
        while ((go = __atomic_load_n(&d->go, __ATOMIC_ACQUIRE)) == seen) qurt_futex_wait((void *)&d->go, seen);
        seen = go;
        if (__atomic_load_n(&d->quit, __ATOMIC_ACQUIRE)) break;
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) {
            d->failed = 1;
        } else {
            run_body(d, a->th);
            qurt_hvx_unlock();
        }
        __atomic_add_fetch(&d->running, -1, __ATOMIC_ACQ_REL);
    }
    qurt_thread_exit(0);
}

static void stop_pool(dec_t *d) {
    __atomic_store_n(&d->quit, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&d->go, 1, __ATOMIC_ACQ_REL);
    qurt_futex_wake((void *)&d->go, DEC_THREADS);
    for (int t = 0; t < d->nthreads; ++t) {
        int st;
        if (d->tid[t]) qurt_thread_join(d->tid[t], &st);
    }
    for (int t = 0; t < DEC_THREADS; ++t) {
        free(d->stack[t]);
        free(d->dma[t]);
        free(d->tbuf[t]);
    }
    free(d->rope);
    free(d->rope_key);
}

// the workers through program prog once (the caller waits)
static void pool_run(dec_t *d, int prog) {
    d->prog = prog;
    d->t_mark = HAP_perf_get_time_us();   // (the stages' times from here: not the caller's own work before)
    memset(d->ctr, 0, sizeof d->ctr);
    d->arrive = 0;
    __atomic_store_n(&d->running, d->nthreads, __ATOMIC_RELEASE);
    __atomic_add_fetch(&d->go, 1, __ATOMIC_ACQ_REL);
    qurt_futex_wake((void *)&d->go, DEC_THREADS);
    while (__atomic_load_n(&d->running, __ATOMIC_ACQUIRE) > 0) asm volatile("pause(#16)");
}

// what the CPU wrote since (a prompt, the states it set) seen afresh / the states, the KV cache and the logits out to
// where the CPU reads them
static void clean_in(dec_t *d) {
    const uint64_t t0 = HAP_perf_get_time_us();
    if (!(d->flags & DEC_DEBUG_NO_CLEAN)) qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    d->t_mark = HAP_perf_get_time_us();
    d->prof[DEC_PROF_CACHE] += d->t_mark - t0;
}
static void clean_out(dec_t *d) {
    const uint64_t tf = HAP_perf_get_time_us();
    if (!(d->flags & DEC_DEBUG_NO_CLEAN)) qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);
    d->prof[DEC_PROF_CACHE] += HAP_perf_get_time_us() - tf;
}

// the call's program on every worker, the DSP's cache cleaned around it
static int run(dec_t *d, int prog) {
    clean_in(d);
    d->failed = 0;
    pool_run(d, prog);
    clean_out(d);
    return d->failed ? AEE_EFAILED : 0;
}

// ---- the interface ---------------------------------------------------------------------------------------------------

static void *map(int fd) {
    void *v = NULL;
    uint64 pa = 0;
    return HAP_mmap_get(fd, &v, &pa) ? NULL : v;
}

// The scratch, carved out of S at scratch (count == 0: only its size): the logits [T][vocab] and the final hidden
// rows [T][hidden] first (the CPU reads them there), then the step's own rows, the attention blocks' parts, the
// quantized activations; the stages' times in its last 128 bytes
static uint64_t carve(dec_t *d, const dec_cfg_t *c, uint8_t *at) {
    uint8_t *const start = at;
#define TAKE(ptr, type, count)                                                    \
    do {                                                                          \
        if (d) ptr = (type *)at;                                                  \
        at += ((size_t)(count) * sizeof(type) + 127u) / 128u * 128u;              \
    } while (0)
    const int H = c->hidden, kd = c->nk * c->dk, vd = c->nv * c->dv, conv_dim = 2 * kd + vd;
    const int cw = vd > c->nh * c->hd ? vd : c->nh * c->hd, wide = c->sff > cw ? c->sff : cw;
    for (int t = 0; t < DEC_T; ++t) TAKE(d->logits[t], float, c->vocab);
    for (int t = 0; t < DEC_N; ++t) TAKE(d->hid[t], float, H);
    for (int t = 0; t < DEC_N; ++t) {
        TAKE(d->xa[t], float, H);
        TAKE(d->xb[t], float, H);
        TAKE(d->mix[t], float, H);
        TAKE(d->mlp[t], float, H);
        TAKE(d->fin[t], float, H);
        TAKE(d->proj[t], float, conv_dim);
        TAKE(d->zb[t], float, vd);
        TAKE(d->ab[t], float, 2 * c->nv);
        TAKE(d->core[t], float, cw);
        TAKE(d->qb[t], float, c->nh * 2 * c->hd);
        TAKE(d->kb[t], float, c->nkv * c->hd);
        TAKE(d->vb[t], float, c->nkv * c->hd);
        TAKE(d->g[t], float, c->sff);
        TAKE(d->u[t], float, c->sff);
        TAKE(d->cat[t], float, 2 * H);
    }
    TAKE(d->part, float, (size_t)DEC_T * c->nh * ((size_t)c->max_chunks * c->kv_chunk / DEC_BLOCK + 1) * (c->hd + 2));
    {
        // a prompt's HMX operands: DEC_N rows of the widest input / output
        const int kmax = conv_dim > c->sff ? (conv_dim > c->nh * 2 * c->hd ? conv_dim : c->nh * 2 * c->hd)
                                           : (c->sff > c->nh * 2 * c->hd ? c->sff : c->nh * 2 * c->hd);
        TAKE(d->tiles_a, uint8_t, (size_t)DEC_N * (size_t)(kmax > H ? kmax : H) * 2u);
        for (int i = 0; i < 3; ++i) TAKE(d->tiles_o[i], uint8_t, (size_t)DEC_N * (size_t)(kmax > H ? kmax : H) * 2u);
    }
    TAKE(d->dlog, float, c->mtp ? (c->n_draft ? c->n_draft : c->vocab) : 1);
    for (int t = 0; t < DEC_T; ++t) {
        act_t *acts[4] = {&d->act_h[t], &d->act_c[t], &d->act_m[t], &d->act_w[t]};
        const int widths[4] = {H > cw ? H : cw, cw, wide, 2 * H};
        for (int i = 0; i < 4; ++i) {
            TAKE(acts[i]->q, int8_t, widths[i]);
            TAKE(acts[i]->lo, uint32_t, widths[i] / 8);
            TAKE(acts[i]->hi, uint32_t, widths[i] / 8);
            TAKE(acts[i]->bs, int32_t, widths[i] / 16);
            TAKE(acts[i]->sc, float, widths[i] / 16);
        }
    }
#undef TAKE
    return (uint64_t)(at - start) + 128u;   // + the stages' times
}

static int cfg_ok(const dec_cfg_t *c) {
    return c->version == DEC_VERSION && c->n_layers <= DEC_MAX_LAYERS && c->hidden % 64 == 0 && c->sff % 64 == 0 &&
           c->dk <= 256 && c->dv % 16 == 0 && c->hd % 64 == 0 && (c->nh * c->hd) % 64 == 0 &&
           c->max_chunks <= DEC_MAX_CHUNKS && c->conv_k >= 1 && c->conv_k <= 8;
}

int geode_npu_dec_size(remote_handle64 h, const unsigned char *cfg, int cfg_len, int64_t *bytes) {
    (void)h;
    *bytes = 0;
    if (cfg_len != (int)sizeof(dec_cfg_t) || !cfg_ok((const dec_cfg_t *)cfg)) return AEE_EBADPARM;
    *bytes = (int64_t)carve(NULL, (const dec_cfg_t *)cfg, NULL);
    return 0;
}

int geode_npu_dec_open(remote_handle64 h, const unsigned char *cfg, int cfg_len, int w_fd, int p_fd, int s_fd,
                       int threads) {
    if (g_dec) return AEE_EALREADY;
    if (cfg_len != (int)sizeof(dec_cfg_t) || !cfg_ok((const dec_cfg_t *)cfg)) return AEE_EBADPARM;
    dec_t *d = (dec_t *)calloc(1, sizeof *d);
    if (!d) return AEE_ENOMEMORY;
    memcpy(&d->c, cfg, sizeof d->c);
    const dec_cfg_t *c = &d->c;
    uint8_t *vt = NULL;
    unsigned vt_size = 0;
    int e = geode_npu_vtcm(h, &vt, &vt_size);
    d->nthreads = threads < 1 ? 6 : threads > DEC_THREADS ? DEC_THREADS : threads;
    if (e || !vt || (size_t)vt_size < (size_t)d->nthreads * DEC_THREAD_VTCM) {
        free(d);
        return e ? e : AEE_ENOMEMORY;
    }
    d->vtcm = vt;
    d->W = (uint8_t *)map(w_fd);
    d->P = (uint8_t *)map(p_fd);
    d->S = (uint8_t *)map(s_fd);
    d->w_fd = w_fd, d->p_fd = p_fd, d->s_fd = s_fd;
    if (!d->W || !d->P || !d->S) {
        free(d);
        return AEE_EBADPARM;
    }
    const uint64_t need = carve(d, c, d->S + c->scratch);
    if (need > c->scratch_bytes) {
        free(d);
        return AEE_ENOMEMORY;
    }
    d->prof = (uint64_t *)(d->S + c->scratch + c->scratch_bytes - 128);
    memset(d->prof, 0, 128);
    // every matrix's hmx4 copy (a prompt's products), each fd mapped once
    {
        const dec_mat_t *all[3 * DEC_MAX_LAYERS * 4 + 8];
        int na = 0;
        for (int l = 0; l < c->n_layers; ++l) {
            const dec_layer_t *L = &c->layers[l];
            const dec_mat_t *ms[10] = {&L->qkv, &L->z, &L->out, &L->q, &L->k, &L->v, &L->o, &L->gate, &L->up, &L->down};
            for (int i = 0; i < 10; ++i) all[na++] = ms[i];
        }
        if (c->mtp) {   // the MTP layer's K / V for a prompt (dec_mtp_kv)
            all[na++] = &c->mtp_fc;
            all[na++] = &c->mtp_layer.k;
            all[na++] = &c->mtp_layer.v;
        }
        for (int i = 0; i < na; ++i) {
            const int fd = all[i]->hmx_fd;
            if (fd <= 0) continue;
            int seen = 0;
            for (int j = 0; j < d->n_hmx; ++j) seen |= d->hmx_fds[j] == fd;
            if (seen || d->n_hmx >= (int)(sizeof d->hmx / sizeof d->hmx[0])) continue;
            uint8_t *w = (uint8_t *)map(fd);
            if (!w) continue;
            d->hmx[d->n_hmx] = w;
            d->hmx_fds[d->n_hmx++] = fd;
        }
    }
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio <= 0 || prio >= 255) prio = 127;
    for (int t = 0; t < d->nthreads; ++t)
        if (!(d->dma[t] = (dma_t *)memalign(64, 2 * sizeof(dma_t))) ||
            !(d->tbuf[t] = (float *)memalign(128, DEC_TBUF * sizeof(float))))
            e = AEE_ENOMEMORY;
    if (!(d->rope = (float *)memalign(128, DEC_THREADS * 256 * sizeof(float))) ||
        !(d->rope_key = (int *)calloc(DEC_THREADS * 4, sizeof(int))))
        e = AEE_ENOMEMORY;
    for (int t = 0; t < d->nthreads && !e; ++t) {
        qurt_thread_attr_t a;
        qurt_thread_attr_init(&a);
        if (!(d->stack[t] = memalign(128, DEC_STACK))) {
            e = AEE_ENOMEMORY;
            break;
        }
        qurt_thread_attr_set_name(&a, (char *)"geode_dec");
        qurt_thread_attr_set_stack_addr(&a, d->stack[t]);
        qurt_thread_attr_set_stack_size(&a, DEC_STACK);
        qurt_thread_attr_set_priority(&a, (unsigned short)prio);
        g_args[t] = (worker_arg_t){d, t};
        if (qurt_thread_create(&d->tid[t], &a, worker, &g_args[t]) != QURT_EOK) {
            d->tid[t] = 0;
            e = AEE_ENOTALLOWED;
        }
    }
    if (e) {
        stop_pool(d);
        free(d);
        return e;
    }
    g_dec = d;
    return 0;
}

int geode_npu_dec_kv(remote_handle64 h, int chunk, int fd) {
    (void)h;
    dec_t *d = g_dec;
    if (!d || chunk < 0 || chunk >= DEC_MAX_CHUNKS || chunk >= d->c.max_chunks) return AEE_EBADPARM;
    uint8_t *p = (uint8_t *)map(fd);
    if (!p) return AEE_EBADPARM;
    d->kv[chunk] = p;
    d->kv_fd[chunk] = fd;
    return 0;
}

// the call's tokens: n rows of x (hidden floats each), their positions from pos; their M-RoPE positions
static int take_tokens(dec_t *d, const unsigned char *x, int x_len, int pos, const int *p3, int p3_len, int flags) {
    const dec_cfg_t *c = &d->c;
    const int H = c->hidden, n = x_len / (H * 4);
    if (n < 1 || n > DEC_T || x_len != n * H * 4 || p3_len != 3 * n || pos < 0) return 0;
    for (int t = 0; t < n; ++t)
        if ((pos + t) / c->kv_chunk >= c->max_chunks || !d->kv[(pos + t) / c->kv_chunk]) return 0;
    d->T = n, d->pos = pos, d->flags = flags, d->backup = n == 2;
    for (int t = 0; t < n; ++t) d->p3[t][0] = p3[3 * t], d->p3[t][1] = p3[3 * t + 1], d->p3[t][2] = p3[3 * t + 2];
    const int set = (flags & DEC_SET1) ? 1 : 0;
    float *base = (float *)d->S;
    d->conv_live = (float *)((uint8_t *)base + c->conv_state[set]);
    d->gdn_live = (float *)((uint8_t *)base + c->gdn_state[set]);
    d->conv_other = (float *)((uint8_t *)base + c->conv_state[set ^ 1]);
    d->gdn_other = (float *)((uint8_t *)base + c->gdn_state[set ^ 1]);
    return n;
}

int geode_npu_dec_step(remote_handle64 h, const unsigned char *x, int x_len, int pos, const int *p3, int p3_len,
                       int flags, unsigned char *logits, int logits_len, int *next, int64_t *usec) {
    (void)h, (void)logits, (void)logits_len;
    dec_t *d = g_dec;
    *next = -1;
    *usec = 0;
    if (!d || !take_tokens(d, x, x_len, pos, p3, p3_len, flags)) return AEE_EBADPARM;
    const uint64_t t0 = HAP_perf_get_time_us();
    for (int t = 0; t < d->T; ++t) memcpy(d->xa[t], x + (size_t)t * d->c.hidden * 4, (size_t)d->c.hidden * 4);
    const int e = run(d, 0);
    ++d->prof[DEC_PROF_STEPS];
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    return e;
}

int geode_npu_dec_draft(remote_handle64 h, const unsigned char *e, int e_len, const unsigned char *hs, int hs_len,
                        int pos, const int *p3, int p3_len, int *draft, float *prob, int64_t *usec) {
    (void)h;
    dec_t *d = g_dec;
    *draft = -1;
    *prob = 0.0f;
    *usec = 0;
    if (!d || !d->c.mtp || e_len != hs_len || !take_tokens(d, e, e_len, pos, p3, p3_len, d->flags & ~0xFF00))
        return AEE_EBADPARM;
    const uint64_t t0 = HAP_perf_get_time_us();
    const int H = d->c.hidden;
    for (int t = 0; t < d->T; ++t) {
        memcpy(d->cat[t], e + (size_t)t * H * 4, (size_t)H * 4);
        memcpy(d->cat[t] + H, hs + (size_t)t * H * 4, (size_t)H * 4);
    }
    const int r = run(d, 1);
    *draft = d->draft;
    *prob = d->dprob;
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    return r;
}

int geode_npu_gemm_ptr(remote_handle64 h, const uint8_t *a, const uint8_t *w, uint8_t *o, int m, int k, int n,
                       int out_shift);   // npu_imp.c

// a matrix's hmx4 copy, mapped once (dec_open)
static const uint8_t *hmx_of(dec_t *d, const dec_mat_t *m) {
    for (int i = 0; i < d->n_hmx; ++i)
        if (d->hmx_fds[i] == m->hmx_fd) return d->hmx[i];
    return NULL;
}

// one HMX product of the prompt's T rows: tiles_a (k inputs) x m -> tiles_o[o]
static int hmx_product(dec_t *d, remote_handle64 h, const dec_mat_t *m, int o) {
    const uint8_t *w = hmx_of(d, m);
    if (!w) return AEE_EBADPARM;
    const int rows = (d->T + 31) / 32 * 32;
    return geode_npu_gemm_ptr(h, d->tiles_a, w, d->tiles_o[o], rows, (int)m->k, (int)m->n, m->hmx_shift);
}

// Up to DEC_N tokens of a prompt (x: their embeddings) at positions from pos through every layer: the products on HMX
// (this thread holds it: the session's FastRPC thread), the rest on the workers between them; the last token's
// logits and every token's hidden row at the scratch's start
int geode_npu_dec_prefill(remote_handle64 h, const unsigned char *x, int x_len, int pos, const int *p3, int p3_len,
                          int flags, int64_t *usec) {
    dec_t *d = g_dec;
    *usec = 0;
    if (!d) return AEE_EBADPARM;
    const dec_cfg_t *c = &d->c;
    const int H = c->hidden, n = x_len / (H * 4);
    if (n < 1 || n > DEC_N || x_len != n * H * 4 || p3_len != 3 * n || pos < 0) return AEE_EBADPARM;
    for (int t = 0; t < n; ++t)
        if ((pos + t) / c->kv_chunk >= c->max_chunks || !d->kv[(pos + t) / c->kv_chunk]) return AEE_EBADPARM;
    const uint64_t t0 = HAP_perf_get_time_us();
    d->T = n, d->pos = pos, d->flags = flags, d->backup = 0;
    for (int t = 0; t < n; ++t) d->p3[t][0] = p3[3 * t], d->p3[t][1] = p3[3 * t + 1], d->p3[t][2] = p3[3 * t + 2];
    const int set = (flags & DEC_SET1) ? 1 : 0;
    d->conv_live = (float *)(d->S + c->conv_state[set]);
    d->gdn_live = (float *)(d->S + c->gdn_state[set]);
    d->conv_other = (float *)(d->S + c->conv_state[set ^ 1]);
    d->gdn_other = (float *)(d->S + c->gdn_state[set ^ 1]);
    clean_in(d);
    for (int t = 0; t < n; ++t) memcpy(d->xa[t], x + (size_t)t * H * 4, (size_t)H * 4);
    d->rx = d->xa, d->rxo = d->xb, d->radd = NULL;
    d->failed = 0;
    int e = 0;
#define PHASE(ph)            \
    do {                     \
        d->phase = (ph);     \
        pool_run(d, 2);      \
        if (d->failed) e = AEE_EFAILED; \
    } while (0)
    for (int l = 0; l < c->n_layers && !e; ++l) {
        const dec_layer_t *L = &c->layers[l];
        d->pl = L;
        PHASE(PF_IN);
        const uint64_t th0 = HAP_perf_get_time_us();
        if (!L->attn) {
            if (!e) e = hmx_product(d, h, &L->qkv, 0);
            if (!e) e = hmx_product(d, h, &L->z, 1);
        } else {
            if (!e) e = hmx_product(d, h, &L->q, 0);
            if (!e) e = hmx_product(d, h, &L->k, 1);
            if (!e) e = hmx_product(d, h, &L->v, 2);
        }
        d->prof[DEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - th0;
        if (e) break;
        PHASE(L->attn ? PF_ATT : PF_GDN);
        const uint64_t th1 = HAP_perf_get_time_us();
        if (!e) e = hmx_product(d, h, L->attn ? &L->o : &L->out, 0);
        d->prof[DEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - th1;
        if (e) break;
        PHASE(PF_POST);
        const uint64_t th2 = HAP_perf_get_time_us();
        if (!e) e = hmx_product(d, h, &L->gate, 0);
        if (!e) e = hmx_product(d, h, &L->up, 1);
        d->prof[DEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - th2;
        if (e) break;
        PHASE(PF_MLP);
        const uint64_t th3 = HAP_perf_get_time_us();
        if (!e) e = hmx_product(d, h, &L->down, 0);
        d->prof[DEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - th3;
        if (e) break;
        PHASE(PF_DOWN);
        d->radd = d->mlp;
    }
    if (!e) PHASE(PF_FINAL);
#undef PHASE
    clean_out(d);
    ++d->prof[DEC_PROF_STEPS];
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    return e;
}

// The MTP layer's K / V for up to DEC_N positions of a prompt from pos (e: the next tokens' embeddings, hs: the
// hidden rows the layer reads): [norm_e(e) | norm_h(h)] through fc, the input norm, k and v on HMX, the keys normed and
// turned into the cache with the values. A prompt has no use for the rest of the layer (its attention's output, the
// MLP): only the drafts after it read the cache
int geode_npu_dec_mtp_kv(remote_handle64 h, const unsigned char *e, int e_len, const unsigned char *hs, int hs_len,
                         int pos, const int *p3, int p3_len, int flags, int64_t *usec) {
    dec_t *d = g_dec;
    *usec = 0;
    if (!d || !d->c.mtp) return AEE_EBADPARM;
    const dec_cfg_t *c = &d->c;
    const dec_layer_t *L = &c->mtp_layer;
    const int H = c->hidden, n = e_len / (H * 4);
    if (n < 1 || n > DEC_N || e_len != n * H * 4 || hs_len != e_len || p3_len != 3 * n || pos < 0) return AEE_EBADPARM;
    for (int t = 0; t < n; ++t)
        if ((pos + t) / c->kv_chunk >= c->max_chunks || !d->kv[(pos + t) / c->kv_chunk]) return AEE_EBADPARM;
    const uint64_t t0 = HAP_perf_get_time_us();
    d->T = n, d->pos = pos, d->flags = flags, d->backup = 0;
    for (int t = 0; t < n; ++t) d->p3[t][0] = p3[3 * t], d->p3[t][1] = p3[3 * t + 1], d->p3[t][2] = p3[3 * t + 2];
    clean_in(d);
    for (int t = 0; t < n; ++t) {
        memcpy(d->cat[t], e + (size_t)t * H * 4, (size_t)H * 4);
        memcpy(d->cat[t] + H, hs + (size_t)t * H * 4, (size_t)H * 4);
    }
    d->pl = L;
    d->failed = 0;
    int r = 0;
    d->phase = PF_MTP_IN;
    pool_run(d, 2);
    if (d->failed) r = AEE_EFAILED;
    const uint64_t th0 = HAP_perf_get_time_us();
    if (!r) r = hmx_product(d, h, &c->mtp_fc, 0);
    d->prof[DEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - th0;
    if (!r) {
        d->phase = PF_MTP_X;
        pool_run(d, 2);
        if (d->failed) r = AEE_EFAILED;
    }
    const uint64_t th1 = HAP_perf_get_time_us();
    if (!r) r = hmx_product(d, h, &L->k, 1);
    if (!r) r = hmx_product(d, h, &L->v, 2);
    d->prof[DEC_PROF_PRODUCTS] += HAP_perf_get_time_us() - th1;
    if (!r) {
        d->phase = PF_MTP_KV;
        pool_run(d, 2);
        if (d->failed) r = AEE_EFAILED;
    }
    clean_out(d);
    ++d->prof[DEC_PROF_STEPS];
    *usec = (int64_t)(HAP_perf_get_time_us() - t0);
    return r;
}

int geode_npu_dec_close(remote_handle64 h) {
    (void)h;
    dec_t *d = g_dec;
    if (!d) return 0;
    stop_pool(d);
    for (int i = 0; i < d->n_hmx; ++i) HAP_mmap_put(d->hmx_fds[i]);
    for (int i = 0; i < DEC_MAX_CHUNKS; ++i)
        if (d->kv[i]) HAP_mmap_put(d->kv_fd[i]);
    HAP_mmap_put(d->w_fd);
    HAP_mmap_put(d->p_fd);
    HAP_mmap_put(d->s_fd);
    free(d);
    g_dec = NULL;
    return 0;
}

#else   // v73 / v75: the prompt's HMX products only

int geode_npu_dec_size(remote_handle64 h, const unsigned char *cfg, int cfg_len, int64_t *bytes) {
    (void)h, (void)cfg, (void)cfg_len;
    *bytes = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_open(remote_handle64 h, const unsigned char *cfg, int cfg_len, int w_fd, int p_fd, int s_fd,
                       int threads) {
    (void)h, (void)cfg, (void)cfg_len, (void)w_fd, (void)p_fd, (void)s_fd, (void)threads;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_kv(remote_handle64 h, int chunk, int fd) {
    (void)h, (void)chunk, (void)fd;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_step(remote_handle64 h, const unsigned char *x, int x_len, int pos, const int *p3, int p3_len,
                       int flags, unsigned char *logits, int logits_len, int *next, int64_t *usec) {
    (void)h, (void)x, (void)x_len, (void)pos, (void)p3, (void)p3_len, (void)flags, (void)logits, (void)logits_len;
    *next = -1;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_draft(remote_handle64 h, const unsigned char *e, int e_len, const unsigned char *hs, int hs_len,
                        int pos, const int *p3, int p3_len, int *draft, float *prob, int64_t *usec) {
    (void)h, (void)e, (void)e_len, (void)hs, (void)hs_len, (void)pos, (void)p3, (void)p3_len;
    *draft = -1;
    *prob = 0.0f;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_prefill(remote_handle64 h, const unsigned char *x, int x_len, int pos, const int *p3, int p3_len,
                          int flags, int64_t *usec) {
    (void)h, (void)x, (void)x_len, (void)pos, (void)p3, (void)p3_len, (void)flags;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_mtp_kv(remote_handle64 h, const unsigned char *e, int e_len, const unsigned char *hs, int hs_len,
                         int pos, const int *p3, int p3_len, int flags, int64_t *usec) {
    (void)h, (void)e, (void)e_len, (void)hs, (void)hs_len, (void)pos, (void)p3, (void)p3_len, (void)flags;
    *usec = 0;
    return AEE_EUNSUPPORTED;
}
int geode_npu_dec_close(remote_handle64 h) {
    (void)h;
    return 0;
}

#endif
