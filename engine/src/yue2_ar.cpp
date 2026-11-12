// yue2_ar.cpp - see yue2_ar.hpp.
#include "yue2_ar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace geode::yue2 {
namespace {

typedef float v8 __attribute__((vector_size(32)));

inline float bf16f(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline v8 load8(const float *p) {
    v8 r;
    std::memcpy(&r, p, sizeof r);
    return r;
}
inline void store8(float *p, v8 v) { std::memcpy(p, &v, sizeof v); }
inline float hsum(v8 v) { return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7])); }

// as yue2.cpp's oracle: x * rsqrt(mean(x^2) + eps) * w, the mean in double
void rms_row(const float *x, const float *w, float *out, int n, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
    const float r = 1.0f / std::sqrt(static_cast<float>(s / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] * r) * w[i];
}
inline void rope_head(float *x, const float *cos, const float *sin, int hd) {
    const int h = hd / 2;
    for (int i = 0; i < h; ++i) {
        const float a = x[i], b = x[i + h];
        x[i] = a * cos[i] - b * sin[i];
        x[i + h] = b * cos[i] + a * sin[i];
    }
}
// a KV head's row to int8: amax / 127 its scale, rounded to nearest even (model.cpp's quant_kv_row)
inline void quant_kv(const float *x, int n, int8_t *q, float &scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < n; ++i) q[i] = static_cast<int8_t>(std::clamp(std::nearbyint(x[i] * inv), -127.0f, 127.0f));
    scale = amax / 127.0f;
}
size_t kv_elem(KvType t) { return t == KvType::I8 ? 1 : t == KvType::F16 ? 2 : 4; }

#if defined(__aarch64__) && defined(__ARM_NEON)
// One attention task on NEON: NQ queries (a tile's rows x the KV head's query heads) over np positions of one KV
// head. The portable loop below had stayed scalar (~460 cycles a position and KV head on the 8 Elite): here a key row
// is widened 16 values at a time straight into registers against every query, and the values go 8 dims at a time
// over the block's positions (the block's V rows stay in L1), the NQ x 8 sums in registers.
template <KvType KT>
inline void widen16(const uint8_t *p, float32x4_t f[4]) {
    if constexpr (KT == KvType::I8) {
        const int8x16_t b = vld1q_s8(reinterpret_cast<const int8_t *>(p));
        const int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_high_s8(b);
        f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        f[1] = vcvtq_f32_s32(vmovl_high_s16(lo));
        f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        f[3] = vcvtq_f32_s32(vmovl_high_s16(hi));
    } else if constexpr (KT == KvType::F16) {
        const float16x8_t a = vld1q_f16(reinterpret_cast<const float16_t *>(p));
        const float16x8_t b = vld1q_f16(reinterpret_cast<const float16_t *>(p) + 8);
        f[0] = vcvt_f32_f16(vget_low_f16(a));
        f[1] = vcvt_high_f32_f16(a);
        f[2] = vcvt_f32_f16(vget_low_f16(b));
        f[3] = vcvt_high_f32_f16(b);
    } else {
        const float *q = reinterpret_cast<const float *>(p);
        for (int i = 0; i < 4; ++i) f[i] = vld1q_f32(q + 4 * i);
    }
}
template <KvType KT>
inline void widen8(const uint8_t *p, float32x4_t &a, float32x4_t &b) {
    if constexpr (KT == KvType::I8) {
        const int16x8_t w = vmovl_s8(vld1_s8(reinterpret_cast<const int8_t *>(p)));
        a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
        b = vcvtq_f32_s32(vmovl_high_s16(w));
    } else if constexpr (KT == KvType::F16) {
        const float16x8_t h = vld1q_f16(reinterpret_cast<const float16_t *>(p));
        a = vcvt_f32_f16(vget_low_f16(h));
        b = vcvt_high_f32_f16(h);
    } else {
        a = vld1q_f32(reinterpret_cast<const float *>(p));
        b = vld1q_f32(reinterpret_cast<const float *>(p) + 4);
    }
}
// k / v: the block's first row of this KV head; stride: bytes between positions; ks / vs (int8): the first
// position's scales, sstride floats apart. sc: NQ x cap scores scratch. dst[i]: hd sums, then max, then the sum.
// Enough independent sums for the FMA pipes (latency ~4, 4 pipes: ~16 chains): a token's 2 queries take 2 positions
// at a time with 4 sums each for the scores, and 32 dims a pass over the values; a prompt's tiles have more queries.
template <KvType KT, int NQ>
void attend_neon(const float *const *qv, const int *lim, int np, const uint8_t *k, const uint8_t *v, size_t stride,
                 const float *ks, const float *vs, size_t sstride, float scale, int hd, float *sc, int cap,
                 float *const *dst) {
    constexpr size_t el = KT == KvType::I8 ? 1 : KT == KvType::F16 ? 2 : 4;
    constexpr int PP = NQ <= 2 ? 2 : 1;                      // positions a step
    constexpr int QA = NQ <= 4 ? 4 : 2;                      // sums a query and position
    constexpr int DV = NQ <= 2 ? 32 : NQ <= 4 ? 16 : 8;      // value dims a pass
    float mx[NQ], z[NQ];
    for (int i = 0; i < NQ; ++i) mx[i] = -std::numeric_limits<float>::infinity();
    const auto qk = [&]<int P>(int j) {
        float32x4_t acc[NQ][P][QA];
        for (int i = 0; i < NQ; ++i)
            for (int p = 0; p < P; ++p)
                for (int a = 0; a < QA; ++a) acc[i][p][a] = vdupq_n_f32(0.0f);
        for (int d = 0; d < hd; d += 16) {
            float32x4_t f[P][4];
            for (int p = 0; p < P; ++p) widen16<KT>(k + static_cast<size_t>(j + p) * stride + static_cast<size_t>(d) * el, f[p]);
            for (int i = 0; i < NQ; ++i) {
                const float *q = qv[i] + d;
                const float32x4_t q0 = vld1q_f32(q), q1 = vld1q_f32(q + 4), q2 = vld1q_f32(q + 8), q3 = vld1q_f32(q + 12);
                for (int p = 0; p < P; ++p) {
                    if constexpr (QA == 4) {
                        acc[i][p][0] = vfmaq_f32(acc[i][p][0], q0, f[p][0]);
                        acc[i][p][1] = vfmaq_f32(acc[i][p][1], q1, f[p][1]);
                        acc[i][p][2] = vfmaq_f32(acc[i][p][2], q2, f[p][2]);
                        acc[i][p][3] = vfmaq_f32(acc[i][p][3], q3, f[p][3]);
                    } else {
                        acc[i][p][0] = vfmaq_f32(acc[i][p][0], q0, f[p][0]);
                        acc[i][p][1] = vfmaq_f32(acc[i][p][1], q1, f[p][1]);
                        acc[i][p][0] = vfmaq_f32(acc[i][p][0], q2, f[p][2]);
                        acc[i][p][1] = vfmaq_f32(acc[i][p][1], q3, f[p][3]);
                    }
                }
            }
        }
        for (int p = 0; p < P; ++p) {
            const float fs = KT == KvType::I8 ? scale * ks[static_cast<size_t>(j + p) * sstride] : scale;
            for (int i = 0; i < NQ; ++i) {
                float32x4_t s4 = acc[i][p][0];
                for (int a = 1; a < QA; ++a) s4 = vaddq_f32(s4, acc[i][p][a]);
                const float s = vaddvq_f32(s4) * fs;
                sc[static_cast<size_t>(i) * cap + j + p] = s;
                if (j + p < lim[i]) mx[i] = std::max(mx[i], s);
            }
        }
    };
    int j = 0;
    for (; j + PP <= np; j += PP) qk.template operator()<PP>(j);
    for (; j < np; ++j) qk.template operator()<1>(j);
    // the weights: exp(s - max) where visible (their sum is the softmax's), times the value's scale; 0 elsewhere,
    // so the values' loop needs no mask
    for (int i = 0; i < NQ; ++i) {
        float *si = sc + static_cast<size_t>(i) * cap;
        float t = 0.0f;
        for (int jj = 0; jj < lim[i]; ++jj) {
            const float e = fast_exp(si[jj] - mx[i]);
            t += e;
            si[jj] = e;
        }
        if constexpr (KT == KvType::I8)
            for (int jj = 0; jj < lim[i]; ++jj) si[jj] *= vs[static_cast<size_t>(jj) * sstride];
        for (int jj = lim[i]; jj < np; ++jj) si[jj] = 0.0f;
        z[i] = t;
    }
    for (int c = 0; c < hd; c += DV) {
        float32x4_t acc[NQ][DV / 4];
        for (int i = 0; i < NQ; ++i)
            for (int a = 0; a < DV / 4; ++a) acc[i][a] = vdupq_n_f32(0.0f);
        const uint8_t *vr = v + static_cast<size_t>(c) * el;
        for (int jj = 0; jj < np; ++jj, vr += stride) {
            float32x4_t vv[DV / 4];
            for (int a = 0; a < DV / 8; ++a) widen8<KT>(vr + static_cast<size_t>(8 * a) * el, vv[2 * a], vv[2 * a + 1]);
            for (int i = 0; i < NQ; ++i) {
                const float w = sc[static_cast<size_t>(i) * cap + jj];
                for (int a = 0; a < DV / 4; ++a) acc[i][a] = vfmaq_n_f32(acc[i][a], vv[a], w);
            }
        }
        for (int i = 0; i < NQ; ++i)
            for (int a = 0; a < DV / 4; ++a) vst1q_f32(dst[i] + c + 4 * a, acc[i][a]);
    }
    for (int i = 0; i < NQ; ++i) {
        dst[i][hd] = mx[i];
        dst[i][hd + 1] = z[i];
    }
}
// The int8 cache's task in integers: the queries quantized to int8 per row and head (q8 / qs), the scores on SDOT
// (8 instructions a position and query for 128 dims, no widening); the weights exp(s - max) x the value's scale
// quantized to int16 against their maximum per query (15 bits: a softmax's long tail of small weights survives -
// 8-bit weights for SDOT would drop it), the values widened to int16 only and summed on SMLAL into int32 (at most
// 32767 x 127 x 256 < 2^31 a block). k / v / ks / vs / stride / sstride as attend_neon.
template <int NQ>
void attend_i8(const int8_t *const *q8, const float *qs, const int *lim, int np, const int8_t *k, const int8_t *v,
               size_t stride, const float *ks, const float *vs, size_t sstride, float scale, int hd, float *sc, int cap,
               int16_t *w16, float *const *dst) {
    constexpr int PP = NQ <= 2 ? 4 : 2;                      // positions a step (independent SDOT chains)
    constexpr int DV = NQ <= 2 ? 32 : NQ <= 4 ? 16 : 8;      // value dims a pass (the sums stay in registers)
    float mx[NQ], z[NQ];
    for (int i = 0; i < NQ; ++i) mx[i] = -std::numeric_limits<float>::infinity();
    const auto qk = [&]<int P>(int j) {
        int32x4_t acc[NQ][P];
        for (int i = 0; i < NQ; ++i)
            for (int p = 0; p < P; ++p) acc[i][p] = vdupq_n_s32(0);
        for (int d = 0; d < hd; d += 16) {
            int8x16_t qc[NQ];
            for (int i = 0; i < NQ; ++i) qc[i] = vld1q_s8(q8[i] + d);
            for (int p = 0; p < P; ++p) {
                const int8x16_t kc = vld1q_s8(k + static_cast<size_t>(j + p) * stride + d);
                for (int i = 0; i < NQ; ++i) acc[i][p] = vdotq_s32(acc[i][p], qc[i], kc);
            }
        }
        for (int p = 0; p < P; ++p) {
            const float f = scale * ks[static_cast<size_t>(j + p) * sstride];
            for (int i = 0; i < NQ; ++i) {
                const float s = static_cast<float>(vaddvq_s32(acc[i][p])) * (qs[i] * f);
                sc[static_cast<size_t>(i) * cap + j + p] = s;
                if (j + p < lim[i]) mx[i] = std::max(mx[i], s);
            }
        }
    };
    int j = 0;
    for (; j + PP <= np; j += PP) qk.template operator()<PP>(j);
    for (; j < np; ++j) qk.template operator()<1>(j);
    float wscale[NQ];
    for (int i = 0; i < NQ; ++i) {
        float *si = sc + static_cast<size_t>(i) * cap;
        int16_t *wi = w16 + static_cast<size_t>(i) * cap;
        float t = 0.0f, wmax = 0.0f;
        for (int jj = 0; jj < lim[i]; ++jj) {
            const float e = fast_exp(si[jj] - mx[i]);
            t += e;
            si[jj] = e * vs[static_cast<size_t>(jj) * sstride];
            wmax = std::max(wmax, si[jj]);
        }
        const float inv = wmax > 0.0f ? 32767.0f / wmax : 0.0f;
        for (int jj = 0; jj < lim[i]; ++jj) wi[jj] = static_cast<int16_t>(std::lrint(si[jj] * inv));
        for (int jj = lim[i]; jj < np; ++jj) wi[jj] = 0;
        z[i] = t;
        wscale[i] = wmax / 32767.0f;
    }
    // The block's value rows into a dense copy first when the dims take several passes: in the cache they are 1 KB
    // apart (every KV head's row side by side), so a pass's rows fall into a few L1 sets and each pass read them again
    // from L2 (3/5 of a decode step's attention); dense they stay in L1. The same sums.
    alignas(64) static thread_local int8_t vdense[256 * 128];
    const int8_t *vsrc = v;
    size_t vstride = stride;
    if (DV < hd && np <= 256 && hd <= 128) {
        for (int jj = 0; jj < np; ++jj) std::memcpy(vdense + static_cast<size_t>(jj) * hd, v + static_cast<size_t>(jj) * stride, static_cast<size_t>(hd));
        vsrc = vdense, vstride = static_cast<size_t>(hd);
    }
    for (int c = 0; c < hd; c += DV) {
        int32x4_t acc[NQ][DV / 4];
        for (int i = 0; i < NQ; ++i)
            for (int a = 0; a < DV / 4; ++a) acc[i][a] = vdupq_n_s32(0);
        const int8_t *vr = vsrc + c;
        for (int jj = 0; jj < np; ++jj, vr += vstride) {
            int16x8_t vv[DV / 8];
            if constexpr (DV == 8) {
                vv[0] = vmovl_s8(vld1_s8(vr));
            } else {
                for (int a = 0; a < DV / 16; ++a) {
                    const int8x16_t b = vld1q_s8(vr + 16 * a);
                    vv[2 * a] = vmovl_s8(vget_low_s8(b));
                    vv[2 * a + 1] = vmovl_high_s8(b);
                }
            }
            for (int i = 0; i < NQ; ++i) {
                const int16_t w = w16[static_cast<size_t>(i) * cap + jj];
                for (int a = 0; a < DV / 8; ++a) {
                    acc[i][2 * a] = vmlal_n_s16(acc[i][2 * a], vget_low_s16(vv[a]), w);
                    acc[i][2 * a + 1] = vmlal_high_n_s16(acc[i][2 * a + 1], vv[a], w);
                }
            }
        }
        for (int i = 0; i < NQ; ++i)
            for (int a = 0; a < DV / 4; ++a)
                vst1q_f32(dst[i] + c + 4 * a, vmulq_n_f32(vcvtq_f32_s32(acc[i][a]), wscale[i]));
    }
    for (int i = 0; i < NQ; ++i) {
        dst[i][hd] = mx[i];
        dst[i][hd + 1] = z[i];
    }
}
bool attend_i8_n(int nq, const int8_t *const *q8, const float *qs, const int *lim, int np, const int8_t *k,
                 const int8_t *v, size_t stride, const float *ks, const float *vs, size_t sstride, float scale, int hd,
                 float *sc, int cap, int16_t *w16, float *const *dst) {
    switch (nq) {
    case 1: attend_i8<1>(q8, qs, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, w16, dst); return true;
    case 2: attend_i8<2>(q8, qs, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, w16, dst); return true;
    case 4: attend_i8<4>(q8, qs, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, w16, dst); return true;
    case 6: attend_i8<6>(q8, qs, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, w16, dst); return true;
    case 8: attend_i8<8>(q8, qs, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, w16, dst); return true;
    default: return false;
    }
}
template <KvType KT>
bool attend_neon_n(int nq, const float *const *qv, const int *lim, int np, const uint8_t *k, const uint8_t *v,
                   size_t stride, const float *ks, const float *vs, size_t sstride, float scale, int hd, float *sc,
                   int cap, float *const *dst) {
    switch (nq) {
    case 1: attend_neon<KT, 1>(qv, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, dst); return true;
    case 2: attend_neon<KT, 2>(qv, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, dst); return true;
    case 4: attend_neon<KT, 4>(qv, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, dst); return true;
    case 6: attend_neon<KT, 6>(qv, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, dst); return true;
    case 8: attend_neon<KT, 8>(qv, lim, np, k, v, stride, ks, vs, sstride, scale, hd, sc, cap, dst); return true;
    default: return false;
    }
}
#endif

// a cache row (hd values) to floats
inline void row_to_f32(KvType t, const uint8_t *p, int hd, float *out) {
    if (t == KvType::I8) {
        const auto *q = reinterpret_cast<const int8_t *>(p);
        for (int i = 0; i < hd; ++i) out[i] = static_cast<float>(q[i]);
    } else if (t == KvType::F16) {
        const auto *h = reinterpret_cast<const _Float16 *>(p);
        for (int i = 0; i < hd; ++i) out[i] = static_cast<float>(h[i]);
    } else {
        std::memcpy(out, p, sizeof(float) * static_cast<size_t>(hd));
    }
}

}  // namespace

const char *kv_type_name(KvType t) { return t == KvType::I8 ? "int8" : t == KvType::F16 ? "fp16" : "fp32"; }
bool kv_type_from_name(const std::string &s, KvType &t) {
    if (s == "int8") t = KvType::I8;
    else if (s == "fp16") t = KvType::F16;
    else if (s == "fp32") t = KvType::F32;
    else return false;
    return true;
}

size_t ArBranch::bytes() const { return 2 * kv_bytes + (ks.size() + vs.size()) * sizeof(float); }

bool FastAr::open(Pack &pack, const ArOptions &opt, std::string &err) {
    pack_ = &pack;
    opt_ = opt;
    emb_cache_.clear(), emb_lo_ = emb_hi_ = 0;   // (cache_embed's rows: the pack's before)
    if (const char *w = std::getenv("GEODE_YUE2_DECODE_WIDTH")) decode_width_ = std::max(1, std::atoi(w));
    const std::string pre = opt.q4 ? "q4." : "q8.";
    if (!read_config(pack, c_, err) || !(opt.map && !opt.q4 ? pack.map(file(), err) : pack.load(file(), err))) return false;
    const int64_t H = c_.hidden, qd = static_cast<int64_t>(c_.heads) * c_.head_dim,
                  kd = static_cast<int64_t>(c_.kv_heads) * c_.head_dim, F = c_.ffn, hd = c_.head_dim;
    bool ok = true;
    const auto q8 = [&](const std::string &name, int64_t rows, int64_t cols) {   // (a layer's: Q4 with opt.q4)
        QView v;
        ok = ok && (opt.q4 ? pack.q4(pre + name, rows, cols, v, err) : pack.q8(pre + name, rows, cols, v, err));
        return v;
    };
    const auto f = [&](const std::string &name, int64_t n) {
        const float *p = ok ? pack.f32(pre + name, {n}, err) : nullptr;
        ok = ok && p;
        return p;
    };
    layers_.assign(static_cast<size_t>(c_.layers), Layer{});
    for (int l = 0; l < c_.layers; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        Layer &y = layers_[static_cast<size_t>(l)];
        y.in_norm = f(b + "input_layernorm.weight", H);
        y.q = q8(b + "self_attn.q_proj.weight", qd, H);
        y.k = q8(b + "self_attn.k_proj.weight", kd, H);
        y.v = q8(b + "self_attn.v_proj.weight", kd, H);
        y.o = q8(b + "self_attn.o_proj.weight", H, qd);
        y.q_norm = f(b + "self_attn.q_norm.weight", hd);
        y.k_norm = f(b + "self_attn.k_norm.weight", hd);
        y.mlp_norm = f(b + "post_attention_layernorm.weight", H);
        y.gate = q8(b + "mlp.gate_proj.weight", F, H);
        y.up = q8(b + "mlp.up_proj.weight", F, H);
        y.down = q8(b + "mlp.down_proj.weight", H, F);
    }
    norm_ = f("model.norm.weight", H);
    if (ok) ok = pack.q8(pre + "lm_head.weight", c_.vocab, H, lm_head_, err);   // (Q8 in either file)
    if (!ok) return false;
    embed_ = pack.find("model.embed_tokens.weight");
    if (!embed_ || embed_->type != DType::BF16 || embed_->shape != std::vector<int64_t>{c_.vocab, H}) {
        err = "the pack has no bf16 embedding [vocab, hidden] (embed.bin)";
        return false;
    }
    // attention's registers: hd <= 128 (16 x 8 floats a query), at most 8 query heads a KV head
    if (c_.heads % c_.kv_heads || c_.heads / c_.kv_heads > 8 || c_.head_dim % 32 || c_.head_dim > 128 ||
        qd % 64 || F % 64 || H % 64) {
        err = "FastAr: geometry the kernels do not take";
        return false;
    }
    return true;
}

bool FastAr::branch_init(ArBranch &b, int capacity, std::string &err) const {
    return branch_init(b, capacity, err, opt_.kv);
}

bool FastAr::branch_init(ArBranch &b, int capacity, std::string &err, KvType type) const {
    if (capacity < 1 || capacity > c_.tok.context) {
        err = "branch_init: capacity " + std::to_string(capacity) + " outside 1 .. " + std::to_string(c_.tok.context);
        return false;
    }
    const size_t kd = static_cast<size_t>(c_.kv_heads) * c_.head_dim;
    const size_t n = static_cast<size_t>(c_.layers) * static_cast<size_t>(capacity) * kd * kv_elem(type);
    b.len = 0;
    b.cap = capacity;
    b.type = type;
    b.k = aligned_bytes(n);
    b.v = aligned_bytes(n);
    b.kv_bytes = n;
    if (!b.k || !b.v) {
        err = "branch_init: cannot allocate 2 x " + std::to_string(n >> 20) + " MiB of KV cache";
        return false;
    }
    const size_t ns = type == KvType::I8 ? static_cast<size_t>(c_.layers) * capacity * c_.kv_heads : 0;
    b.ks.assign(ns, 0.0f);
    b.vs.assign(ns, 0.0f);
    return true;
}

bool FastAr::branch_copy(const ArBranch &src, ArBranch &dst, int n) const {
    if (src.type != dst.type || n < 0 || n > src.len || n > dst.cap || !src.k || !src.v || !dst.k || !dst.v) return false;
    const size_t kd = static_cast<size_t>(c_.kv_heads) * c_.head_dim, el = kv_elem(src.type);
    const size_t row = kd * el, nkv = static_cast<size_t>(c_.kv_heads), un = static_cast<size_t>(n);
    for (int l = 0; l < c_.layers; ++l) {
        const size_t s0 = static_cast<size_t>(l) * static_cast<size_t>(src.cap), d0 = static_cast<size_t>(l) * static_cast<size_t>(dst.cap);
        std::memcpy(dst.k.get() + d0 * row, src.k.get() + s0 * row, un * row);
        std::memcpy(dst.v.get() + d0 * row, src.v.get() + s0 * row, un * row);
        if (src.type == KvType::I8) {
            std::memcpy(dst.ks.data() + d0 * nkv, src.ks.data() + s0 * nkv, un * nkv * sizeof(float));
            std::memcpy(dst.vs.data() + d0 * nkv, src.vs.data() + s0 * nkv, un * nkv * sizeof(float));
        }
    }
    dst.len = n;
    return true;
}

void FastAr::product(const QView &w, const float *x, int n, float *y, ThreadPool &pool, bool quantized) const {
    const int64_t rows = w.rows, cols = w.cols;
    constexpr int64_t kRows = 32;   // a task's rows: 32 x 2 KiB of codes stay in L1/L2 against the batch's tokens
    const int64_t tasks = (rows + kRows - 1) / kRows;
    if (opt_.act_q8) {
        if (!quantized) {
            if (acts_.size() < static_cast<size_t>(n)) acts_.resize(static_cast<size_t>(n));
            const auto q = [&](int64_t a, int64_t b) {
                for (int64_t t = a; t < b; ++t) quantize_activation(x + t * cols, cols, acts_[static_cast<size_t>(t)]);
            };
            if (n >= 8) pool.parallel_for(n, q);
            else q(0, n);
        }
        std::vector<const QAct *> ap(static_cast<size_t>(n));
        std::vector<float *> yp(static_cast<size_t>(n));
        for (int t = 0; t < n; ++t) {
            ap[static_cast<size_t>(t)] = &acts_[static_cast<size_t>(t)];
            yp[static_cast<size_t>(t)] = y + static_cast<int64_t>(t) * rows;
        }
        pool.parallel_for(tasks, [&](int64_t a, int64_t b) {
            matvec_rows_q_multi(w, ap.data(), n, yp.data(), a * kRows, std::min(rows, b * kRows));
        });
    } else {
        pool.parallel_for(tasks, [&](int64_t a, int64_t b) {
            for (int t = 0; t < n; ++t)
                matvec_rows(w, x + static_cast<int64_t>(t) * cols, y + static_cast<int64_t>(t) * rows, a * kRows,
                            std::min(rows, b * kRows));
        });
    }
}

void FastAr::products(const Part *parts, int np, const float *x, int n, ThreadPool &pool, bool swiglu) const {
    // One parallel_for for several products of the same x (q, k, v; gate and up): the tasks run over every part's
    // chunks of rows in turn, so a token's small products (k, v: 1024 rows) share the threads with the big ones and
    // the pool syncs once, not three times. swiglu: parts = {gate, up}, each task both for its rows, then
    // gate = silu(gate) * up on them (the MLP's activation without a pass of its own).
    constexpr int64_t kRows = 32;
    const int64_t cols = parts[0].w->cols;
    if (opt_.act_q8) {
        if (acts_.size() < static_cast<size_t>(n)) acts_.resize(static_cast<size_t>(n));
        const auto q = [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) quantize_activation(x + t * cols, cols, acts_[static_cast<size_t>(t)]);
        };
        if (n >= 8) pool.parallel_for(n, q);
        else q(0, n);
    }
    int64_t first[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < np; ++i)
        first[i + 1] = first[i] + (swiglu && i == 1 ? 0 : (parts[i].w->rows + kRows - 1) / kRows);
    const auto run = [&](const Part &P, int64_t r0, int64_t r1, const QAct *const *ap, float **yp) {
        const int64_t rows = P.w->rows;
        if (opt_.act_q8) {
            for (int t = 0; t < n; ++t) yp[t] = P.y + static_cast<int64_t>(t) * rows;
            matvec_rows_q_multi(*P.w, ap, n, yp, r0, r1);
        } else {
            for (int t = 0; t < n; ++t)
                matvec_rows(*P.w, x + static_cast<int64_t>(t) * cols, P.y + static_cast<int64_t>(t) * rows, r0, r1);
        }
    };
    pool.parallel_for(first[np], [&](int64_t a, int64_t e) {
        thread_local std::vector<const QAct *> ap;
        thread_local std::vector<float *> yp;
        ap.resize(static_cast<size_t>(n));
        yp.resize(static_cast<size_t>(n));
        if (opt_.act_q8)
            for (int t = 0; t < n; ++t) ap[static_cast<size_t>(t)] = &acts_[static_cast<size_t>(t)];
        for (int64_t task = a; task < e; ++task) {
            int i = 0;
            while (task >= first[i + 1]) ++i;
            const Part &P = parts[i];
            const int64_t r0 = (task - first[i]) * kRows, r1 = std::min(P.w->rows, r0 + kRows);
            run(P, r0, r1, ap.data(), yp.data());
            if (swiglu) {
                const Part &U = parts[1];
                run(U, r0, r1, ap.data(), yp.data());
                for (int t = 0; t < n; ++t) {
                    float *g = P.y + static_cast<int64_t>(t) * P.w->rows;
                    const float *u = U.y + static_cast<int64_t>(t) * U.w->rows;
                    for (int64_t r = r0; r < r1; ++r) g[r] = silu_fast(g[r]) * u[r];
                }
            }
        }
    });
}

void FastAr::attention(const std::vector<Row> &rows, int layer, const float *q, float *out, ThreadPool &pool) {
    // Softmax(q K^T / sqrt(hd)) V per row over its branch's positions 0 .. its own. A task = (tile, KV head, block):
    // a tile is up to kTile consecutive rows of one branch (a prompt's), so each key and value row is turned into
    // floats once for the tile's rows and the KV head's `group` query heads (YuE2: 2); a block is kBlock positions,
    // so a long context spreads over the threads. Each task keeps its max, its weights' sum and its weighted values;
    // the blocks are joined after in their order (a row's bits do not depend on the batch or the threads).
    constexpr int kTile = 4, kBlock = 256, kMaxG = 8;
    const int nh = c_.heads, nkv = c_.kv_heads, hd = c_.head_dim, group = nh / nkv, qd = nh * hd;
    const size_t kd = static_cast<size_t>(nkv) * hd, part = static_cast<size_t>(hd) + 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    struct Tile {
        int r0, nt, nb;   // its first row, rows, blocks (of its last row's positions)
    };
    std::vector<Tile> tiles;
    int maxb = 0;
    for (int r = 0; r < static_cast<int>(rows.size());) {
        int e = r + 1;
        while (e < static_cast<int>(rows.size()) && e - r < kTile && rows[static_cast<size_t>(e)].b == rows[static_cast<size_t>(r)].b &&
               rows[static_cast<size_t>(e)].pos == rows[static_cast<size_t>(e - 1)].pos + 1)
            ++e;
        const int nb = rows[static_cast<size_t>(e - 1)].pos / kBlock + 1;
        tiles.push_back({r, e - r, nb});
        maxb = std::max(maxb, nb);
        r = e;
    }
    std::vector<int64_t> first(tiles.size() + 1, 0);   // the tiles' first task
    for (size_t i = 0; i < tiles.size(); ++i) first[i + 1] = first[i] + static_cast<int64_t>(tiles[i].nb) * nkv;
    const size_t need = rows.size() * static_cast<size_t>(nh) * maxb * part;
    if (parts_.size() < need) parts_.resize(need);
    const auto at = [&](int row, int h, int b) {
        return parts_.data() + ((static_cast<size_t>(row) * nh + h) * maxb + b) * part;
    };
#if defined(__aarch64__) && defined(__ARM_NEON)
    static const bool portable = std::getenv("GEODE_YUE2_PORTABLE_ATTN") != nullptr;   // A/B and checks
    bool int8_path = !portable && opt_.kv == KvType::I8;
    for (const Row &r : rows) int8_path = int8_path && r.b->type == KvType::I8;   // (a mixed batch: the generic path)
    if (int8_path) {   // the queries in int8 for SDOT: per row and head, amax / 127
        q8_.resize(rows.size() * static_cast<size_t>(qd));
        qs_.resize(rows.size() * static_cast<size_t>(nh));
        for (size_t r = 0; r < rows.size(); ++r)
            for (int h = 0; h < nh; ++h) {
                const size_t o = r * qd + static_cast<size_t>(h) * hd;
                quant_kv(q + o, hd, q8_.data() + o, qs_[r * nh + static_cast<size_t>(h)]);
            }
    }
#endif
    pool.parallel_for(first.back(), [&](int64_t a, int64_t e) {
        thread_local std::vector<float> sc, kf, vf;
        thread_local std::vector<int16_t> w16;
        sc.resize(static_cast<size_t>(kTile) * kMaxG * kBlock);
        w16.resize(static_cast<size_t>(kTile) * kMaxG * kBlock);
        kf.resize(static_cast<size_t>(hd));
        vf.resize(static_cast<size_t>(hd));
        for (int64_t task = a; task < e; ++task) {
            const size_t ti = static_cast<size_t>(std::upper_bound(first.begin(), first.end(), task) - first.begin() - 1);
            const Tile &T = tiles[ti];
            const int64_t local = task - first[ti];
            const int kh = static_cast<int>(local / T.nb), blk = static_cast<int>(local % T.nb);
            const Row &r0 = rows[static_cast<size_t>(T.r0)];
            const ArBranch &B = *r0.b;
            const int p0 = blk * kBlock, p1 = std::min(p0 + kBlock, r0.pos + T.nt);   // last row's positions end
            const int np = p1 - p0, nq = T.nt * group;
            const float *qv[kTile * kMaxG];
            int lim[kTile * kMaxG];   // positions visible to query i inside the block: [p0, p0 + lim)
            for (int t = 0; t < T.nt; ++t)
                for (int g = 0; g < group; ++g) {
                    const int i = t * group + g;
                    qv[i] = q + static_cast<size_t>(T.r0 + t) * qd + static_cast<size_t>(kh * group + g) * hd;
                    lim[i] = std::clamp(r0.pos + t + 1 - p0, 0, np);
                }
            const size_t lbase = static_cast<size_t>(layer) * B.cap;
            const uint8_t *kbase = B.k.get(), *vbase = B.v.get();
            const size_t el = kv_elem(B.type);
#if defined(__aarch64__) && defined(__ARM_NEON)
            if (!portable) {
                float *dst[kTile * kMaxG];
                for (int t = 0; t < T.nt; ++t)
                    for (int g = 0; g < group; ++g) dst[t * group + g] = at(T.r0 + t, kh * group + g, blk);
                const size_t first_pos = lbase + static_cast<size_t>(p0);
                const uint8_t *k0 = kbase + (first_pos * kd + static_cast<size_t>(kh) * hd) * el;
                const uint8_t *v0 = vbase + (first_pos * kd + static_cast<size_t>(kh) * hd) * el;
                const float *ks = B.type == KvType::I8 ? B.ks.data() + first_pos * nkv + kh : nullptr;
                const float *vs = B.type == KvType::I8 ? B.vs.data() + first_pos * nkv + kh : nullptr;
                const size_t stride = kd * el;
                bool done = false;
                if (int8_path) {
                    const int8_t *q8v[kTile * kMaxG];
                    float qsv[kTile * kMaxG];
                    for (int t = 0; t < T.nt; ++t)
                        for (int g = 0; g < group; ++g) {
                            const size_t r = static_cast<size_t>(T.r0 + t), h = static_cast<size_t>(kh * group + g);
                            q8v[t * group + g] = q8_.data() + r * qd + h * hd;
                            qsv[t * group + g] = qs_[r * nh + h];
                        }
                    done = attend_i8_n(nq, q8v, qsv, lim, np, reinterpret_cast<const int8_t *>(k0),
                                       reinterpret_cast<const int8_t *>(v0), stride, ks, vs, nkv, scale, hd,
                                       sc.data(), kBlock, w16.data(), dst);
                } else if (B.type == KvType::I8)
                    done = attend_neon_n<KvType::I8>(nq, qv, lim, np, k0, v0, stride, ks, vs, nkv, scale, hd,
                                                     sc.data(), kBlock, dst);
                else if (B.type == KvType::F16)
                    done = attend_neon_n<KvType::F16>(nq, qv, lim, np, k0, v0, stride, ks, vs, nkv, scale, hd,
                                                      sc.data(), kBlock, dst);
                else
                    done = attend_neon_n<KvType::F32>(nq, qv, lim, np, k0, v0, stride, ks, vs, nkv, scale, hd,
                                                      sc.data(), kBlock, dst);
                if (done) continue;
            }
#endif
            float mx[kTile * kMaxG], sum[kTile * kMaxG];
            for (int i = 0; i < nq; ++i) mx[i] = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < np; ++j) {
                const size_t pos = lbase + static_cast<size_t>(p0 + j);
                row_to_f32(B.type, kbase + (pos * kd + static_cast<size_t>(kh) * hd) * el, hd, kf.data());
                const float f = B.type == KvType::I8 ? scale * B.ks[pos * nkv + static_cast<size_t>(kh)] : scale;
                for (int i = 0; i < nq; ++i) {
                    if (j >= lim[i]) continue;
                    v8 s0 = {}, s1 = {};
                    for (int d = 0; d < hd; d += 16) {
                        s0 += load8(qv[i] + d) * load8(kf.data() + d);
                        s1 += load8(qv[i] + d + 8) * load8(kf.data() + d + 8);
                    }
                    const float s = hsum(s0 + s1) * f;
                    sc[static_cast<size_t>(i) * kBlock + j] = s;
                    mx[i] = std::max(mx[i], s);
                }
            }
            for (int i = 0; i < nq; ++i) {
                float z = 0.0f;
                float *si = &sc[static_cast<size_t>(i) * kBlock];
                for (int j = 0; j < lim[i]; ++j) z += (si[j] = fast_exp(si[j] - mx[i]));
                sum[i] = z;
            }
            v8 acc[kTile * kMaxG][16];   // hd <= 128 here (checked at open: 16 x 8)
            for (int i = 0; i < nq; ++i)
                for (int d = 0; d < hd / 8; ++d) acc[i][d] = v8{};
            for (int j = 0; j < np; ++j) {
                const size_t pos = lbase + static_cast<size_t>(p0 + j);
                row_to_f32(B.type, vbase + (pos * kd + static_cast<size_t>(kh) * hd) * el, hd, vf.data());
                const float vsc = B.type == KvType::I8 ? B.vs[pos * nkv + static_cast<size_t>(kh)] : 1.0f;
                for (int i = 0; i < nq; ++i) {
                    if (j >= lim[i]) continue;
                    const float w = sc[static_cast<size_t>(i) * kBlock + j] * vsc;
                    for (int d = 0; d < hd / 8; ++d) acc[i][d] += w * load8(vf.data() + 8 * d);
                }
            }
            for (int t = 0; t < T.nt; ++t)
                for (int g = 0; g < group; ++g) {
                    const int i = t * group + g;
                    float *dst = at(T.r0 + t, kh * group + g, blk);
                    for (int d = 0; d < hd / 8; ++d) store8(dst + 8 * d, acc[i][d]);
                    dst[hd] = mx[i];
                    dst[hd + 1] = sum[i];
                }
        }
    });
    // the blocks joined: each one's sums times exp(its max - the max), divided by the weights' total
    pool.parallel_for(static_cast<int64_t>(rows.size()) * nh, [&](int64_t a, int64_t e) {
        for (int64_t i = a; i < e; ++i) {
            const int row = static_cast<int>(i / nh), h = static_cast<int>(i % nh);
            const int nb = rows[static_cast<size_t>(row)].pos / kBlock + 1;
            float m = -std::numeric_limits<float>::infinity();
            for (int b = 0; b < nb; ++b) m = std::max(m, at(row, h, b)[hd]);
            float *o = out + static_cast<size_t>(row) * qd + static_cast<size_t>(h) * hd;
            std::fill(o, o + hd, 0.0f);
            float z = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const float *p = at(row, h, b);
                if (p[hd + 1] == 0.0f) continue;   // no visible position in that block
                const float f = std::exp(p[hd] - m);
                z += p[hd + 1] * f;
                for (int d = 0; d < hd; ++d) o[d] += p[d] * f;
            }
            const float inv = 1.0f / z;
            for (int d = 0; d < hd; ++d) o[d] *= inv;
        }
    });
}

void FastAr::batch(const std::vector<Row> &rows, const int32_t *tokens, std::vector<std::pair<int, float *>> &outs,
                   ThreadPool &pool) {
    const int R = static_cast<int>(rows.size()), H = c_.hidden, hd = c_.head_dim, nh = c_.heads, nkv = c_.kv_heads,
              qd = nh * hd, kd = nkv * hd, F = c_.ffn, half = hd / 2;
    const size_t nr = static_cast<size_t>(R);
    x_.resize(nr * H);
    h_.resize(nr * H);
    q_.resize(nr * qd);
    kn_.resize(nr * kd);
    vn_.resize(nr * kd);
    att_.resize(nr * qd);
    tmp_.resize(nr * H);
    g_.resize(nr * F);
    u_.resize(nr * F);
    cos_.resize(nr * half);
    sin_.resize(nr * half);
    // a token's step (a few rows) wants fewer threads than a prompt: the 8 Elite's DRAM is saturated by a couple
    // of cores, more only add contention (GEODE_YUE2_DECODE_WIDTH; a prompt takes the whole pool)
    pool.set_width(R <= 4 ? std::min(decode_width_, pool.size()) : pool.size());
    // GEODE_YUE2_AR_PROF=N: a step's (<= 4 rows) time by part, summed, printed every N steps (1: 256)
    static const int prof = [] { const char *e = std::getenv("GEODE_YUE2_AR_PROF"); const int n = e ? std::atoi(e) : 0; return n == 1 ? 256 : n; }();
    const bool pr = prof && R <= 4;
    double tp = pr ? now_seconds() : 0.0;
    std::vector<uint16_t> e(static_cast<size_t>(H));
    for (int r = 0; r < R; ++r) {
        const double tr = pr ? now_seconds() : 0.0;
        if (tokens[r] >= emb_lo_ && tokens[r] < emb_hi_)
            std::memcpy(e.data(), emb_cache_.data() + static_cast<size_t>(tokens[r] - emb_lo_) * H, sizeof(uint16_t) * H);
        else
            pack_->read_rows(*embed_, tokens[r], 1, e.data());   // run() checked the ids
        if (pr) prof_[7] += now_seconds() - tr;
        for (int i = 0; i < H; ++i) x_[static_cast<size_t>(r) * H + i] = bf16f(e[static_cast<size_t>(i)]);
        for (int i = 0; i < half; ++i) {
            const float angle = static_cast<float>(rows[static_cast<size_t>(r)].pos) * c_.inv_freq[static_cast<size_t>(i)];
            cos_[static_cast<size_t>(r) * half + i] = std::cos(angle);
            sin_[static_cast<size_t>(r) * half + i] = std::sin(angle);
        }
    }
    const auto rows_par = [&](const std::function<void(int)> &f) {
        if (R >= 8) pool.parallel_for(R, [&](int64_t a, int64_t b) { for (int64_t r = a; r < b; ++r) f(static_cast<int>(r)); });
        else for (int r = 0; r < R; ++r) f(r);
    };
    const auto tick = [&](int i) {
        if (!pr) return;
        const double t = now_seconds();
        prof_[i] += t - tp;
        tp = t;
    };
    tick(0);
    for (int l = 0; l < c_.layers; ++l) {
        const Layer &L = layers_[static_cast<size_t>(l)];
        rows_par([&](int r) { rms_row(&x_[static_cast<size_t>(r) * H], L.in_norm, &h_[static_cast<size_t>(r) * H], H, c_.eps); });
        {
            const Part qkv[3] = {{&L.q, q_.data()}, {&L.k, kn_.data()}, {&L.v, vn_.data()}};
            products(qkv, 3, h_.data(), R, pool);
        }
        tick(1);
        // q_norm / k_norm per head, RoPE, the row's K and V into its branch: a head an item (a token's few rows: the
        // heads over the pool - serial they were ~1 ms of a CFG step; each head's arithmetic as before)
        const auto head = [&](int r, int hh, float *t) {
            const float *c = &cos_[static_cast<size_t>(r) * half], *s = &sin_[static_cast<size_t>(r) * half];
            if (hh < nh) {
                float *v = &q_[static_cast<size_t>(r) * qd + static_cast<size_t>(hh) * hd];
                rms_row(v, L.q_norm, t, hd, c_.eps);
                std::memcpy(v, t, sizeof(float) * hd);
                rope_head(v, c, s, hd);
                return;
            }
            hh -= nh;
            const Row &row = rows[static_cast<size_t>(r)];
            ArBranch &B = *row.b;
            const size_t pos = static_cast<size_t>(l) * B.cap + static_cast<size_t>(row.pos), el = kv_elem(B.type);
            {
                float *kv = &kn_[static_cast<size_t>(r) * kd + static_cast<size_t>(hh) * hd];
                rms_row(kv, L.k_norm, t, hd, c_.eps);
                std::memcpy(kv, t, sizeof(float) * hd);
                rope_head(kv, c, s, hd);
                const float *vv = &vn_[static_cast<size_t>(r) * kd + static_cast<size_t>(hh) * hd];
                uint8_t *kd8 = B.k.get() + (pos * kd + static_cast<size_t>(hh) * hd) * el;
                uint8_t *vd8 = B.v.get() + (pos * kd + static_cast<size_t>(hh) * hd) * el;
                if (B.type == KvType::I8) {
                    quant_kv(kv, hd, reinterpret_cast<int8_t *>(kd8), B.ks[pos * nkv + static_cast<size_t>(hh)]);
                    quant_kv(vv, hd, reinterpret_cast<int8_t *>(vd8), B.vs[pos * nkv + static_cast<size_t>(hh)]);
                } else if (B.type == KvType::F16) {
                    auto *kh = reinterpret_cast<_Float16 *>(kd8), *vh = reinterpret_cast<_Float16 *>(vd8);
                    for (int i = 0; i < hd; ++i) kh[i] = static_cast<_Float16>(kv[i]), vh[i] = static_cast<_Float16>(vv[i]);
                } else {
                    std::memcpy(kd8, kv, sizeof(float) * hd);
                    std::memcpy(vd8, vv, sizeof(float) * hd);
                }
            }
        };
        if (R <= 4) {
            pool.parallel_for(static_cast<int64_t>(R) * (nh + nkv), [&](int64_t a, int64_t e) {
                float t[256];
                for (int64_t i = a; i < e; ++i) head(static_cast<int>(i / (nh + nkv)), static_cast<int>(i % (nh + nkv)), t);
            });
        } else {
            rows_par([&](int r) {
                float t[256];
                for (int hh = 0; hh < nh + nkv; ++hh) head(r, hh, t);
            });
        }
        tick(2);
        attention(rows, l, q_.data(), att_.data(), pool);
        tick(3);
        product(L.o, att_.data(), R, tmp_.data(), pool);
        tick(4);
        for (size_t i = 0; i < nr * H; ++i) x_[i] += tmp_[i];
        rows_par([&](int r) { rms_row(&x_[static_cast<size_t>(r) * H], L.mlp_norm, &h_[static_cast<size_t>(r) * H], H, c_.eps); });
        {
            const Part gu[2] = {{&L.gate, g_.data()}, {&L.up, u_.data()}};
            products(gu, 2, h_.data(), R, pool, true);
        }
        tick(5);
        product(L.down, g_.data(), R, tmp_.data(), pool);
        tick(6);
        for (size_t i = 0; i < nr * H; ++i) x_[i] += tmp_[i];
    }
    if (pr && ++prof_n_ % prof == 0) {
        const double k = 1e3 / prof_n_;
        std::fprintf(stderr, "AR step (%d rows), ms a step over %d: embed %.2f qkv %.2f norm/rope %.2f attn %.2f o %.2f "
                     "gate/up %.2f down %.2f (embedding rows %.2f)\n", R, prof_n_, prof_[0] * k, prof_[1] * k, prof_[2] * k, prof_[3] * k,
                     prof_[4] * k, prof_[5] * k, prof_[6] * k, prof_[7] * k);
    }
    for (const auto &[r, dst] : outs) rms_row(&x_[static_cast<size_t>(r) * H], norm_, dst, H, c_.eps);
}

bool FastAr::run(std::vector<Seg> &segs, ThreadPool &pool, std::string &err) {
    if (segs.empty()) {
        err = "run: no segments";
        return false;
    }
    for (size_t i = 0; i < segs.size(); ++i) {
        const Seg &s = segs[i];
        if (!s.b || s.n < 1 || !s.tokens || s.b->len + s.n > s.b->cap) {
            err = "run: segment " + std::to_string(i) + " (" + std::to_string(s.n) + " tokens) does not fit its branch";
            return false;
        }
        for (size_t j = 0; j < i; ++j)
            if (segs[j].b == s.b) {
                err = "run: two segments on one branch";
                return false;
            }
        for (int t = 0; t < s.n; ++t)
            if (s.tokens[t] < 0 || s.tokens[t] >= c_.vocab) {
                err = "run: token id " + std::to_string(s.tokens[t]);
                return false;
            }
        for (const int r : s.rows)
            if (r < 0 || r >= s.n || !s.hidden) {
                err = "run: hidden row " + std::to_string(r) + " outside the segment";
                return false;
            }
    }
    std::vector<int> done(segs.size(), 0);
    std::vector<Row> rows;
    std::vector<int32_t> toks;
    std::vector<std::pair<int, float *>> outs;
    for (;;) {
        int active = 0;
        for (size_t i = 0; i < segs.size(); ++i) active += done[i] < segs[i].n;
        if (!active) break;
        if (cancelled && cancelled()) {
            err = "cancelled";
            return false;
        }
        const int per = std::max(1, kChunk / active);
        rows.clear();
        toks.clear();
        outs.clear();
        std::vector<int> took(segs.size(), 0);
        for (size_t i = 0; i < segs.size(); ++i) {
            Seg &s = segs[i];
            const int take = std::min(per, s.n - done[i]);
            if (take <= 0) continue;
            for (int j = 0; j < take; ++j) {
                const int t = done[i] + j;
                for (size_t k = 0; k < s.rows.size(); ++k)
                    if (s.rows[k] == t) outs.emplace_back(static_cast<int>(rows.size()), s.hidden + k * c_.hidden);
                rows.push_back({s.b, s.b->len + j});
                toks.push_back(s.tokens[t]);
            }
            took[i] = take;
        }
        batch(rows, toks.data(), outs, pool);
        for (size_t i = 0; i < segs.size(); ++i) {
            segs[i].b->len += took[i];
            done[i] += took[i];
        }
    }
    return true;
}

bool FastAr::put_kv(ArBranch &b, int l, int at, int n, const float *k, const float *v, ThreadPool &pool) const {
    if (l < 0 || l >= c_.layers || at < 0 || n < 0 || at + n > b.cap || !b.k || !b.v) return false;
    const int hd = c_.head_dim, nkv = c_.kv_heads;
    const size_t kd = static_cast<size_t>(nkv) * hd, el = kv_elem(b.type);
    pool.parallel_for(n, [&](int64_t a, int64_t e) {
        for (int64_t p = a; p < e; ++p) {
            const size_t pos = static_cast<size_t>(l) * b.cap + static_cast<size_t>(at + p);
            for (int hh = 0; hh < nkv; ++hh) {
                const float *kv = k + static_cast<size_t>(p) * kd + static_cast<size_t>(hh) * hd;
                const float *vv = v + static_cast<size_t>(p) * kd + static_cast<size_t>(hh) * hd;
                uint8_t *kd8 = b.k.get() + (pos * kd + static_cast<size_t>(hh) * hd) * el;
                uint8_t *vd8 = b.v.get() + (pos * kd + static_cast<size_t>(hh) * hd) * el;
                if (b.type == KvType::I8) {
                    quant_kv(kv, hd, reinterpret_cast<int8_t *>(kd8), b.ks[pos * nkv + static_cast<size_t>(hh)]);
                    quant_kv(vv, hd, reinterpret_cast<int8_t *>(vd8), b.vs[pos * nkv + static_cast<size_t>(hh)]);
                } else if (b.type == KvType::F16) {
                    auto *kh = reinterpret_cast<_Float16 *>(kd8), *vh = reinterpret_cast<_Float16 *>(vd8);
                    for (int i = 0; i < hd; ++i) kh[i] = static_cast<_Float16>(kv[i]), vh[i] = static_cast<_Float16>(vv[i]);
                } else {
                    std::memcpy(kd8, kv, sizeof(float) * hd);
                    std::memcpy(vd8, vv, sizeof(float) * hd);
                }
            }
        }
    }, 64);
    return true;
}

bool FastAr::cache_embed(int lo, int hi) {
    if (lo >= emb_lo_ && hi <= emb_hi_ && emb_hi_ > emb_lo_) return true;
    if (!embed_ || !pack_ || lo < 0 || hi <= lo || hi > c_.vocab || hi - lo > 65536) return false;
    if (pack_->data(*embed_)) return false;   // (in memory already: nothing to gain)
    std::vector<uint16_t> rows;
    try {
        rows.resize(static_cast<size_t>(hi - lo) * static_cast<size_t>(c_.hidden));
    } catch (const std::bad_alloc &) {
        return false;
    }
    if (!pack_->read_rows(*embed_, lo, hi - lo, rows.data())) return false;
    emb_cache_ = std::move(rows);
    emb_lo_ = lo, emb_hi_ = hi;
    return true;
}

bool FastAr::logits(const float *hidden, int n, int lo, int hi, float *out, ThreadPool &pool) const {
    if (n < 1 || lo < 0 || lo >= hi || hi > c_.vocab) return false;
    QView w = lm_head_;
    w.data += static_cast<size_t>(lo) * w.row_bytes;
    w.rows = hi - lo;
    product(w, hidden, n, out, pool);
    return true;
}

}  // namespace geode::yue2
