// media.cpp - media.hpp's products (moved from vision.cpp, unchanged; Products is vision.cpp's lambdas as a class).
#include "media.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace geode::media {

// ---- SMMLA (FEAT_I8MM) where the CPU has it
// (the 8 Elite; not the 720G's A76, which keeps the int8 row kernels). A q8 matrix repacked for it: its rows in pairs,
// each pair's 8-value chunks side by side (16 bytes: row 2p's 8 codes, then row 2p + 1's) and its block scales
// (E4M3 x global) as two floats. The activations likewise in pairs of tokens, int8 a 16-value block (amax / 127, as
// quantize_activation) with two floats a block. A block of 2 x 2 is then two SMMLAs, a convert and one multiply-add
// with its four scale products: 3.5x the int8 row kernels' arithmetic a cycle here.

bool have_i8mm() {
#if defined(__aarch64__) && defined(__linux__)
    static const bool yes = (getauxval(AT_HWCAP2) & (1UL << 13)) != 0 &&   // HWCAP2_I8MM
                            !std::getenv("GEODE_VISION_NO_I8MM");
    return yes;
#else
    return false;
#endif
}

// a q8 matrix (rows even) into the pairs layout
void pack_q8(const QView &m, Packed &o) {
    const int64_t R = m.rows, C = m.cols, nb = C / kBlock;
    o.rows = R;
    o.cols = C;
    o.codes.resize(static_cast<size_t>(R * C));
    o.scales.resize(static_cast<size_t>(R / 2 * nb * 2));
    for (int64_t p = 0; p < R / 2; ++p) {
        const uint8_t *r0 = m.data + static_cast<size_t>(2 * p) * m.row_bytes, *r1 = r0 + m.row_bytes;
        int8_t *c = o.codes.data() + static_cast<size_t>(p * 2 * C);
        for (int64_t k = 0; k < C / 8; ++k) {
            std::memcpy(c + 16 * k, r0 + 8 * k, 8);
            std::memcpy(c + 16 * k + 8, r1 + 8 * k, 8);
        }
        float *sc = o.scales.data() + static_cast<size_t>(p * nb * 2);
        for (int64_t b = 0; b < nb; ++b) {   // the scale bytes follow the C code bytes
            sc[2 * b] = e4m3_value(r0[C + b]) * m.global;
            sc[2 * b + 1] = e4m3_value(r1[C + b]) * m.global;
        }
    }
}

// tokens x0 and x1 (nullptr: past the last, zeros) into one pair: q [C / 8][16], s [C / 16][2]
void quant_pair(const float *x0, const float *x1, int64_t C, int8_t *q, float *s) {
    for (int64_t b = 0; b < C / kBlock; ++b)
        for (int t = 0; t < 2; ++t) {
            const float *xb = t ? x1 : x0;
            int8_t *q0 = q + 32 * b + 8 * t;   // values 0-7 of the block, then 8-15 one chunk on
            if (!xb) {
                std::memset(q0, 0, 8);
                std::memset(q0 + 16, 0, 8);
                s[2 * b + t] = 0.0f;
                continue;
            }
            xb += kBlock * b;
            float amax = 0.0f;
            for (int i = 0; i < kBlock; ++i) amax = std::max(amax, std::fabs(xb[i]));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (int i = 0; i < kBlock; ++i)
                q0[(i / 8) * 16 + i % 8] = static_cast<int8_t>(std::clamp(std::lrint(xb[i] * inv), -127L, 127L));
            s[2 * b + t] = amax / 127.0f;
        }
}

#if defined(__aarch64__)
// rows 2 p0 .. 2 p0 + 7 (four pairs) x token pairs q0 .. q0 + TP - 1: y[token][row] (ystride apart), tokens >= n
// dropped
template <int TP>
__attribute__((target("arch=armv8.2-a+dotprod+fp16+i8mm")))
void mmla_tile(const Packed &w, int64_t p0, const int8_t *aq, const float *as, int64_t q0, int n, float *y,
               size_t ystride) {
    const int64_t C = w.cols, nb = C / kBlock;
    const int8_t *wc = w.codes.data() + static_cast<size_t>(p0 * 2 * C);
    const float *ws = w.scales.data() + static_cast<size_t>(p0 * nb * 2);
    const int8_t *ac = aq + static_cast<size_t>(q0 * 2 * C);
    const float *asc = as + static_cast<size_t>(q0 * nb * 2);
    float32x4_t acc[4][TP];
    for (int p = 0; p < 4; ++p)
        for (int q = 0; q < TP; ++q) acc[p][q] = vdupq_n_f32(0.0f);
    for (int64_t b = 0; b < nb; ++b) {
        int8x16_t xa[TP][2];
        float32x4_t ad[TP];
        for (int q = 0; q < TP; ++q) {
            xa[q][0] = vld1q_s8(ac + q * 2 * C + 32 * b);
            xa[q][1] = vld1q_s8(ac + q * 2 * C + 32 * b + 16);
            const float32x2_t a2 = vld1_f32(asc + q * nb * 2 + 2 * b);
            ad[q] = vcombine_f32(a2, a2);                                   // [a0 a1 a0 a1]
        }
        for (int p = 0; p < 4; ++p) {
            const int8x16_t w0 = vld1q_s8(wc + p * 2 * C + 32 * b), w1 = vld1q_s8(wc + p * 2 * C + 32 * b + 16);
            const float32x2_t w2 = vld1_f32(ws + p * nb * 2 + 2 * b);
            const float32x4_t wd = vcombine_f32(vdup_lane_f32(w2, 0), vdup_lane_f32(w2, 1));   // [w0 w0 w1 w1]
            for (int q = 0; q < TP; ++q) {
                const int32x4_t d = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), w0, xa[q][0]), w1, xa[q][1]);
                acc[p][q] = vfmaq_f32(acc[p][q], vcvtq_f32_s32(d), vmulq_f32(wd, ad[q]));
            }
        }
    }
    // lane 2i + j: row 2 (p0 + p) + i, token 2 (q0 + q) + j
    for (int p = 0; p < 4; ++p)
        for (int q = 0; q < TP; ++q) {
            float v[4];
            vst1q_f32(v, acc[p][q]);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    const int64_t t = 2 * (q0 + q) + j;
                    if (t < n) y[static_cast<size_t>(t) * ystride + static_cast<size_t>(2 * (p0 + p) + i)] = v[2 * i + j];
                }
        }
}
#endif

// row tiles [a, b) (8 rows each) against every token: the tokens in blocks that stay in the cache while the tiles
// pass over them
void mmla_rows(const Packed &w, int64_t a, int64_t b, const int8_t *aq, const float *as, int n, float *y,
               size_t ystride) {
#if defined(__aarch64__)
    const int64_t np = (n + 1) / 2;
    constexpr int64_t kPairs = 64;
    for (int64_t q0 = 0; q0 < np; q0 += kPairs) {
        const int64_t q1 = std::min(np, q0 + kPairs);
        for (int64_t r = a; r < b; ++r) {
            int64_t q = q0;
            for (; q + 2 <= q1; q += 2) mmla_tile<2>(w, 4 * r, aq, as, q, n, y, ystride);
            if (q < q1) mmla_tile<1>(w, 4 * r, aq, as, q, n, y, ystride);
        }
    }
#else
    (void)w, (void)a, (void)b, (void)aq, (void)as, (void)n, (void)y, (void)ystride;
#endif
}

uint64_t tensor_bytes(const Tensor &t) {
    return is_quantized(t.type) ? static_cast<uint64_t>(t.rows) * row_bytes(t.type, t.cols)
                                : static_cast<uint64_t>(t.rows * t.cols) * (t.type == DType::F32 ? 4 : 2);
}


bool Products::on_npu(const QView &m, const float *xs, size_t xstride, float *yo, size_t ystride, int n, const F16 &f,
                      const float *clip) {
    xp_.resize(static_cast<size_t>(n));
    yp_.resize(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) {
        xp_[static_cast<size_t>(t)] = xs + static_cast<size_t>(t) * xstride;
        yp_[static_cast<size_t>(t)] = yo + static_cast<size_t>(t) * ystride;
    }
    const int e = clip ? npu_->gemm_f16c(npu_->g, f.fd, f.off, m.rows, m.cols, f.shift, n, xp_.data(), yp_.data(), clip)
                       : npu_->gemm_f16(npu_->g, f.fd, f.off, m.rows, m.cols, f.shift, n, xp_.data(), yp_.data());
    if (e == 0) return true;
    live_ = false;
    return false;
}

void Products::mm(ThreadPool &pool, const QView &m, const Packed *pk, const float *xs, size_t xstride, float *yo,
                  size_t ystride, int n, const F16 *f) {
    if (f && live_ && n >= 32 && on_npu(m, xs, xstride, yo, ystride, n, *f, nullptr)) return;
    if (pk) {
        const int64_t C = m.cols, nb = C / kBlock, np = (n + 1) / 2;
        aq_.resize(static_cast<size_t>(np * 2 * C));
        as_.resize(static_cast<size_t>(np * nb * 2));
        pool.parallel_for(np, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t)
                quant_pair(xs + static_cast<size_t>(2 * t) * xstride,
                           2 * t + 1 < n ? xs + static_cast<size_t>(2 * t + 1) * xstride : nullptr, C,
                           aq_.data() + static_cast<size_t>(t * 2 * C), as_.data() + static_cast<size_t>(t * nb * 2));
        }, 4, true);
        pool.parallel_for(m.rows / 8, [&](int64_t a, int64_t b) { mmla_rows(*pk, a, b, aq_.data(), as_.data(), n, yo, ystride); },
                          1, true);
        return;
    }
    if (qa_.size() < static_cast<size_t>(n)) {
        qa_.resize(static_cast<size_t>(n));
        qp_.resize(static_cast<size_t>(n));
        ys_.resize(static_cast<size_t>(n));
    }
    pool.parallel_for(n, [&](int64_t a, int64_t b) {
        for (int64_t t = a; t < b; ++t) {
            quantize_activation(xs + static_cast<size_t>(t) * xstride, m.cols, qa_[static_cast<size_t>(t)]);
            qp_[static_cast<size_t>(t)] = &qa_[static_cast<size_t>(t)];
            ys_[static_cast<size_t>(t)] = yo + static_cast<size_t>(t) * ystride;
        }
    }, 8, true);
    pool.parallel_for(m.rows, [&](int64_t a, int64_t b) { matvec_rows_q_multi(m, qp_.data(), n, ys_.data(), a, b); }, 16,
                      true);
}

void Products::clipped(ThreadPool &pool, const QView &m, const Packed *pk, const float *clip, const float *xs, float *yo,
                       int n, const F16 *f) {
    const int64_t C = m.cols, R = m.rows;
    // (on the NPU the clamps happen as the rows are converted to fp16 and back: no passes over them here)
    if (f && live_ && n >= 32 && npu_->gemm_f16c &&
        on_npu(m, xs, static_cast<size_t>(C), yo, static_cast<size_t>(R), n, *f, clip))
        return;
    if (cl_.size() < static_cast<size_t>(n * C)) cl_.resize(static_cast<size_t>(n * C));
    pool.parallel_for(static_cast<int64_t>(n) * C, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) cl_[static_cast<size_t>(i)] = std::clamp(xs[static_cast<size_t>(i)], clip[0], clip[1]);
    }, 4096, true);
    mm(pool, m, pk, cl_.data(), static_cast<size_t>(C), yo, static_cast<size_t>(R), n, f);
    pool.parallel_for(static_cast<int64_t>(n) * R, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) yo[static_cast<size_t>(i)] = std::clamp(yo[static_cast<size_t>(i)], clip[2], clip[3]);
    }, 4096, true);
}

}  // namespace geode::media
