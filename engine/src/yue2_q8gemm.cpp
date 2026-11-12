#include "yue2_q8gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define GEODE_Q8_SDOT 1
#endif

namespace geode::yue2 {

void fwht32(float *v) {
    for (int h = 1; h < 32; h <<= 1)
        for (int i = 0; i < 32; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b, v[j + h] = a - b;
            }
    for (int i = 0; i < 32; ++i) v[i] *= 0.17677669529663687f;   // 1 / sqrt(32)
}

void quant_x8(const float *const *xs, int n, int64_t cols, X8 &out, ThreadPool &pool, bool rot) {
    out.n = n, out.cols = cols;
    out.q.resize(static_cast<size_t>(n) * static_cast<size_t>(cols));
    out.s.resize(static_cast<size_t>(n) * static_cast<size_t>(cols / 32));
    pool.parallel_for(n, [&](int64_t a, int64_t b) {
        for (int64_t t = a; t < b; ++t) {
            const float *x = xs[t];
            int8_t *q = out.q.data() + t * cols;
            float *s = out.s.data() + t * (cols / 32);
            float tmp[32];
            for (int64_t g = 0; g < cols / 32; ++g) {
                const float *xg = x + g * 32;
                if (rot) {
                    std::memcpy(tmp, xg, sizeof tmp);
                    fwht32(tmp);
                    xg = tmp;
                }
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xg[i]));
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                s[g] = amax / 127.0f;
                for (int i = 0; i < 32; ++i)
                    q[g * 32 + i] = static_cast<int8_t>(std::clamp(std::nearbyint(xg[i] * inv), -127.0f, 127.0f));
            }
        }
    });
}

bool have_q8_gemm() {
#if defined(GEODE_Q8_SDOT)
    return true;
#else
    return false;
#endif
}

#if defined(GEODE_Q8_SDOT)
namespace {

constexpr int kP8 = 256;   // inputs a panel (8 KB of codes for a 32-row block: in L1 beside four rows of x)

// a 32-row block's inputs [c0, c0 + pc) (pc % 32 == 0): codes [pc / 4][32 rows][4 inputs], scales [pc / 32][32]
// (the block's fp16 scales times 2^-shift). hmx8: per 32 inputs 16 pairs x 32 rows x 2 bytes, then the scales
// [cols / 32][32] fp16 - two pairs' rows zipped by 16 bits give a row's four inputs together.
void decode_panel8(const uint8_t *blk, int64_t cols, int64_t c0, int pc, float scale, int8_t *codes, float *sc) {
    const __fp16 *bs = reinterpret_cast<const __fp16 *>(blk + cols * 32);
    const float32x4_t k = vdupq_n_f32(scale);
    for (int g = 0; g < pc / 32; ++g) {
        const int64_t gg = c0 / 32 + g;
        const uint8_t *src = blk + gg * 1024;
        for (int q = 0; q < 8; ++q) {
            const uint8_t *a = src + (2 * q) * 64, *b = a + 64;
            uint8_t *d = reinterpret_cast<uint8_t *>(codes) + static_cast<size_t>(g * 8 + q) * 128;
            for (int i = 0; i < 4; ++i) {
                const uint16x8_t va = vreinterpretq_u16_u8(vld1q_u8(a + 16 * i)), vb = vreinterpretq_u16_u8(vld1q_u8(b + 16 * i));
                vst1q_u8(d + 32 * i, vreinterpretq_u8_u16(vzip1q_u16(va, vb)));
                vst1q_u8(d + 32 * i + 16, vreinterpretq_u8_u16(vzip2q_u16(va, vb)));
            }
        }
        for (int o = 0; o < 32; o += 8) {
            const float16x8_t h = vld1q_f16(bs + gg * 32 + o);
            vst1q_f32(sc + g * 32 + o, vmulq_f32(vcvt_f32_f16(vget_low_f16(h)), k));
            vst1q_f32(sc + g * 32 + o + 4, vmulq_f32(vcvt_high_f32_f16(h), k));
        }
    }
}

// acc[t][ro * 8 .. + 8) += the panel's rows ro * 8 .. + 8 against four tokens' inputs (x[t]: the panel's first; sx[t]:
// its first group's scale): an SDOT a token's four inputs against eight rows' (two vectors), int32 per 32 inputs
inline void panel8_rows(const int8_t *codes, const float *sc, int pc, int ro, const int8_t *const x[4],
                        const float *const sx[4], float *const acc[4]) {
    float32x4_t f[4][2];
    for (int t = 0; t < 4; ++t) f[t][0] = vld1q_f32(acc[t] + ro * 8), f[t][1] = vld1q_f32(acc[t] + ro * 8 + 4);
    for (int g = 0; g < pc / 32; ++g) {
        int32x4_t a[4][2];
        for (int t = 0; t < 4; ++t) a[t][0] = vdupq_n_s32(0), a[t][1] = vdupq_n_s32(0);
        for (int qq = 0; qq < 2; ++qq) {
            int8x16_t xv[4];
            for (int t = 0; t < 4; ++t) xv[t] = vld1q_s8(x[t] + g * 32 + qq * 16);
#define GEODE_Q8_LANE(L)                                                                                     \
    {                                                                                                        \
        const int8_t *wp = codes + (static_cast<size_t>(g * 8 + qq * 4 + (L)) * 32 + ro * 8) * 4;           \
        const int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16);                                           \
        for (int t = 0; t < 4; ++t) {                                                                        \
            a[t][0] = vdotq_laneq_s32(a[t][0], w0, xv[t], L);                                                \
            a[t][1] = vdotq_laneq_s32(a[t][1], w1, xv[t], L);                                                \
        }                                                                                                    \
    }
            GEODE_Q8_LANE(0) GEODE_Q8_LANE(1) GEODE_Q8_LANE(2) GEODE_Q8_LANE(3)
#undef GEODE_Q8_LANE
        }
        const float32x4_t s0 = vld1q_f32(sc + g * 32 + ro * 8), s1 = vld1q_f32(sc + g * 32 + ro * 8 + 4);
        for (int t = 0; t < 4; ++t) {
            const float xs = sx[t][g];
            f[t][0] = vfmaq_f32(f[t][0], vcvtq_f32_s32(a[t][0]), vmulq_n_f32(s0, xs));
            f[t][1] = vfmaq_f32(f[t][1], vcvtq_f32_s32(a[t][1]), vmulq_n_f32(s1, xs));
        }
    }
    for (int t = 0; t < 4; ++t) vst1q_f32(acc[t] + ro * 8, f[t][0]), vst1q_f32(acc[t] + ro * 8 + 4, f[t][1]);
}

}  // namespace
#endif

bool cpu_gemm_hmx8(const uint8_t *w, int64_t rows, int64_t cols, int shift, const X8 &x, float *const *ys,
                   ThreadPool &pool) {
#if defined(GEODE_Q8_SDOT)
    if (rows % 32 || cols % 64 || cols < 512 || x.cols != cols) return false;
    const float scale = std::ldexp(1.0f, -shift);
    const int n = x.n;
    const int64_t blk_bytes = cols * 34;
    pool.parallel_for(rows / 32, [&](int64_t b0, int64_t b1) {
        std::vector<int8_t> codes(static_cast<size_t>(kP8) * 32u), zq(static_cast<size_t>(kP8), 0);
        std::vector<float> sc(static_cast<size_t>(kP8)), zs(static_cast<size_t>(kP8 / 32), 0.0f);
        std::vector<float> acc(static_cast<size_t>(n + 3) / 4u * 4u * 32u);
        for (int64_t blk = b0; blk < b1; ++blk) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            const uint8_t *src = w + blk * blk_bytes;
            for (int64_t c0 = 0; c0 < cols; c0 += kP8) {
                const int pc = static_cast<int>(std::min<int64_t>(kP8, cols - c0));
                decode_panel8(src, cols, c0, pc, scale, codes.data(), sc.data());
                for (int t = 0; t < n; t += 4) {
                    const int8_t *xp[4];
                    const float *sp[4];
                    float *ap[4];
                    for (int r = 0; r < 4; ++r) {   // (past the last row: zeros in, the sums thrown away)
                        const bool in = t + r < n;
                        xp[r] = in ? x.q.data() + static_cast<size_t>(t + r) * cols + c0 : zq.data();
                        sp[r] = in ? x.s.data() + static_cast<size_t>(t + r) * (cols / 32) + c0 / 32 : zs.data();
                        ap[r] = &acc[static_cast<size_t>(t + r) * 32u];
                    }
                    for (int ro = 0; ro < 4; ++ro) panel8_rows(codes.data(), sc.data(), pc, ro, xp, sp, ap);
                }
            }
            for (int t = 0; t < n; ++t)
                std::memcpy(ys[t] + blk * 32, &acc[static_cast<size_t>(t) * 32u], sizeof(float) * 32u);
        }
    }, 1);
    return true;
#else
    (void)w, (void)rows, (void)cols, (void)shift, (void)x, (void)ys, (void)pool;
    return false;
#endif
}

}  // namespace geode::yue2
