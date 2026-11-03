// kernels.cpp - decoding and matrix-vector products for the pack's formats. Two paths:
//   exact: FP32 activations times the dequantized weights (the parity reference);
//   int8:  activations quantized per 16-value block, integer dot products per block, then the two scales; NEON
//          SDOT on the phone, the same arithmetic in plain C++ elsewhere.
#include "kernels.hpp"

#include <algorithm>
#include <array>

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define GEODE_NEON_DOTPROD 1
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif
#include <cstdlib>

namespace geode {

namespace {

// Integer levels in code order (tools/qformats.py: positive magnitudes, then their negatives); nvfp4 doubled, so
// its block sums are halved at the end.
constexpr int8_t kQ2[4] = {1, 3, -1, -3};
constexpr int8_t kQ3[8] = {1, 3, 5, 7, -1, -3, -5, -7};
constexpr int8_t kNvfp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// q4: nvfp4's nibbles with 16 evenly spaced odd levels (the GPU decodes them as 2c - 15 after a code remap)
constexpr int8_t kQ4[16] = {1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15};
// q4i: Q4_0's integer levels, code - 8 (its block scales signed)
alignas(16) constexpr int8_t kQ4I[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
// the GPU layout's affine codes: level = 2c - 3 (q2), 2c - 15 (q4)
constexpr int8_t kQ2Affine[4] = {-3, -1, 1, 3};
constexpr int8_t kQ4Affine[16] = {-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};

float level_scale(DType t) { return t == DType::NVFP4 ? 0.5f : 1.0f; }

const std::array<float, 256> &e4m3_table() {
    static const std::array<float, 256> t = [] {
        std::array<float, 256> v{};
        for (int b = 0; b < 256; ++b) {
            const int sign = b >> 7, exp = (b >> 3) & 15, man = b & 7;
            float x = exp == 0 ? std::ldexp(static_cast<float>(man) / 8.0f, -6)
                               : std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
            if (exp == 15 && man == 7) x = NAN;   // E4M3fn has no infinities, only this NaN
            v[static_cast<size_t>(b)] = sign ? -x : x;
        }
        return v;
    }();
    return t;
}

// The 16 integer levels of block b of a row (codes = the row's code bytes).
inline void block_ints(DType t, const uint8_t *codes, int b, int8_t *v) {
    switch (t) {
    case DType::Q2: {
        const uint8_t *c = codes + b * 4;
        for (int j = 0; j < 4; ++j) {
            const uint8_t x = c[j];
            v[4 * j + 0] = kQ2[x & 3];
            v[4 * j + 1] = kQ2[(x >> 2) & 3];
            v[4 * j + 2] = kQ2[(x >> 4) & 3];
            v[4 * j + 3] = kQ2[x >> 6];
        }
        break;
    }
    case DType::Q3: {
        // 24 bytes a 64-value group: low 2 bits of value 4j+s in byte j at bit 2s, its high bit (value v) in byte
        // 16 + v/8 at bit v%8
        const uint8_t *g = codes + (b / 4) * 24;
        const int bb = b % 4;
        for (int i = 0; i < 16; ++i) {
            const int j = 4 * bb + i / 4, s = i % 4, val = 16 * bb + i;
            const int lo = (g[j] >> (2 * s)) & 3;
            const int hi = (g[16 + val / 8] >> (val % 8)) & 1;
            v[i] = kQ3[lo | (hi << 2)];
        }
        break;
    }
    case DType::NVFP4:
    case DType::Q4I:
    case DType::Q4: {
        const int8_t *lv = t == DType::Q4 ? kQ4 : t == DType::Q4I ? kQ4I : kNvfp4x2;
        const uint8_t *c = codes + b * 8;
        for (int j = 0; j < 8; ++j) {
            v[2 * j] = lv[c[j] & 15];
            v[2 * j + 1] = lv[c[j] >> 4];
        }
        break;
    }
    case DType::Q8: {
        const auto *c = reinterpret_cast<const int8_t *>(codes + b * 16);
        for (int i = 0; i < 16; ++i) v[i] = c[i];
        break;
    }
    default:
        for (int i = 0; i < 16; ++i) v[i] = 0;
    }
}

#if defined(GEODE_NEON_DOTPROD)

// the lookup tables as 16-byte vectors (arm_neon.h's intrinsics are macros: no braced lists in their arguments)
alignas(16) constexpr int8_t kLutQ2[16] = {1, 3, -1, -3};
alignas(16) constexpr int8_t kLutQ3[16] = {1, 3, 5, 7, -1, -3, -5, -7};

// The 64 weights of group g as int8 levels, four vectors in the lane orders of the activation planes (decoded once
// per row and group, whatever the number of tokens), and their dot products with one token's activations: lane k of
// the result is block k's integer sum.
template <DType T>
struct Group;

template <>
struct Group<DType::Q2> {
    // (arm_neon.h's intrinsics are macros: they cannot be default member initializers)
    int8x16_t lut;
    uint8x16_t m3;
    explicit Group(const int8_t *levels = kLutQ2) {
        alignas(16) int8_t l[16] = {};
        std::memcpy(l, levels, 4);
        lut = vld1q_s8(l);
        m3 = vdupq_n_u8(3);
    }
    void load(const uint8_t *row, int64_t g, int8x16_t w[4]) const { load_v(vld1q_u8(row + 16 * g), w); }
    // the group's 16 code bytes already in a register (a quad's, transposed)
    void load_v(uint8x16_t v, int8x16_t w[4]) const {
        w[0] = vqtbl1q_s8(lut, vandq_u8(v, m3));
        w[1] = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(v, 2), m3));
        w[2] = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(v, 4), m3));
        w[3] = vqtbl1q_s8(lut, vshrq_n_u8(v, 6));
    }
    int32x4_t dot(const int8x16_t w[4], const QAct &a, int64_t g) const {
        const int8_t *xp = a.p4.data() + 64 * g;
        int32x4_t d = vdupq_n_s32(0);
        d = vdotq_s32(d, w[0], vld1q_s8(xp));
        d = vdotq_s32(d, w[1], vld1q_s8(xp + 16));
        d = vdotq_s32(d, w[2], vld1q_s8(xp + 32));
        d = vdotq_s32(d, w[3], vld1q_s8(xp + 48));
        return d;
    }
};

// q3 (geodepack.py has the layout): one token takes c = the low bits, complemented where the high bit is set, since
// the levels are 2c + 1 (1 3 5 7 -1 -3 -5 -7): a block's sum is 2 c.x + its activations' sum, kept by the
// quantizer. From four tokens on the levels themselves through a table lookup cost less than the correction per token.
// Measured on the Redmi's A76: 6.4 -> 7.1 Gweights/s one token, 15.4 -> 16.8 four (v1's bit planes needed four TBLs).
template <>
struct Group<DType::Q3> {
    int8x16_t lut;
    uint8x16_t m3, four, dup, mask[4];
    Group() {
        lut = vld1q_s8(kLutQ3);
        m3 = vdupq_n_u8(3);
        four = vdupq_n_u8(4);
        alignas(16) static constexpr uint8_t kDup[16] = {8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
        dup = vld1q_u8(kDup);
        for (int s = 0; s < 4; ++s) {
            std::array<uint8_t, 16> m{};
            for (int j = 0; j < 16; ++j) m[static_cast<size_t>(j)] = static_cast<uint8_t>(1u << (4 * (j % 2) + s));
            mask[s] = vld1q_u8(m.data());
        }
    }
    void load(const uint8_t *row, int64_t g, int8x16_t w[4]) const {
        const uint8_t *base = row + 24 * g;
        const uint8x16_t v = vld1q_u8(base);
        const uint8x16_t hb = vqtbl1q_u8(vld1q_u8(base + 8), dup);   // high-bit byte k into lanes 2k, 2k+1
        w[0] = vreinterpretq_s8_u8(veorq_u8(vandq_u8(v, m3), vtstq_u8(hb, mask[0])));
        w[1] = vreinterpretq_s8_u8(veorq_u8(vandq_u8(vshrq_n_u8(v, 2), m3), vtstq_u8(hb, mask[1])));
        w[2] = vreinterpretq_s8_u8(veorq_u8(vandq_u8(vshrq_n_u8(v, 4), m3), vtstq_u8(hb, mask[2])));
        w[3] = vreinterpretq_s8_u8(veorq_u8(vshrq_n_u8(v, 6), vtstq_u8(hb, mask[3])));
    }
    int32x4_t dot(const int8x16_t w[4], const QAct &a, int64_t g) const {
        const int8_t *xp = a.p4.data() + 64 * g;
        int32x4_t d = vdupq_n_s32(0);
        for (int s = 0; s < 4; ++s) d = vdotq_s32(d, w[s], vld1q_s8(xp + 16 * s));
        return vaddq_s32(vshlq_n_s32(d, 1), vld1q_s32(a.bsum.data() + 4 * g));   // level = 2c + 1
    }
    // many tokens: the levels themselves (one more table lookup per vector, no correction per token)
    void load_levels(const uint8_t *row, int64_t g, int8x16_t w[4]) const {
        const uint8_t *base = row + 24 * g;
        const uint8x16_t v = vld1q_u8(base);
        const uint8x16_t hb = vqtbl1q_u8(vld1q_u8(base + 8), dup);
        w[0] = vqtbl1q_s8(lut, vorrq_u8(vandq_u8(v, m3), vandq_u8(vtstq_u8(hb, mask[0]), four)));
        w[1] = vqtbl1q_s8(lut, vorrq_u8(vandq_u8(vshrq_n_u8(v, 2), m3), vandq_u8(vtstq_u8(hb, mask[1]), four)));
        w[2] = vqtbl1q_s8(lut, vorrq_u8(vandq_u8(vshrq_n_u8(v, 4), m3), vandq_u8(vtstq_u8(hb, mask[2]), four)));
        w[3] = vqtbl1q_s8(lut, vorrq_u8(vshrq_n_u8(v, 6), vandq_u8(vtstq_u8(hb, mask[3]), four)));
    }
    int32x4_t dot_levels(const int8x16_t w[4], const QAct &a, int64_t g) const {
        const int8_t *xp = a.p4.data() + 64 * g;
        int32x4_t d = vdupq_n_s32(0);
        for (int s = 0; s < 4; ++s) d = vdotq_s32(d, w[s], vld1q_s8(xp + 16 * s));
        return d;
    }
};

// nibble codes (nvfp4, q4): a TBL from the format's level table
struct NibbleGroup {
    int8x16_t lut;
    uint8x16_t m15;
    explicit NibbleGroup(const int8_t *levels) {
        lut = vld1q_s8(levels);
        m15 = vdupq_n_u8(15);
    }
    void load(const uint8_t *row, int64_t g, int8x16_t w[4]) const {
        // 2 loads of 16 bytes = 4 blocks; each load's lanes are (b, b, b+1, b+1)
        load_v(vld1q_u8(row + 32 * g), vld1q_u8(row + 32 * g + 16), w);
    }
    void load_v(uint8x16_t v0, uint8x16_t v1, int8x16_t w[4]) const {
        w[0] = vqtbl1q_s8(lut, vandq_u8(v0, m15));
        w[1] = vqtbl1q_s8(lut, vshrq_n_u8(v0, 4));
        w[2] = vqtbl1q_s8(lut, vandq_u8(v1, m15));
        w[3] = vqtbl1q_s8(lut, vshrq_n_u8(v1, 4));
    }
    int32x4_t dot(const int8x16_t w[4], const QAct &a, int64_t g) const {
        const int8_t *xp = a.p2.data() + 64 * g;
        int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
        d0 = vdotq_s32(d0, w[0], vld1q_s8(xp));
        d0 = vdotq_s32(d0, w[1], vld1q_s8(xp + 16));
        d1 = vdotq_s32(d1, w[2], vld1q_s8(xp + 32));
        d1 = vdotq_s32(d1, w[3], vld1q_s8(xp + 48));
        return vpaddq_s32(d0, d1);
    }
};

template <>
struct Group<DType::NVFP4> : NibbleGroup {
    Group() : NibbleGroup(kNvfp4x2) {}
};

template <>
struct Group<DType::Q4> : NibbleGroup {
    explicit Group(const int8_t *levels = kQ4) : NibbleGroup(levels) {}
};

template <>
struct Group<DType::Q4I> : NibbleGroup {
    Group() : NibbleGroup(kQ4I) {}
};

template <>
struct Group<DType::Q8> {
    void load(const uint8_t *row, int64_t g, int8x16_t w[4]) const {
        for (int k = 0; k < 4; ++k) w[k] = vld1q_s8(reinterpret_cast<const int8_t *>(row) + 64 * g + 16 * k);
    }
    int32x4_t dot(const int8x16_t w[4], const QAct &a, int64_t g) const {
        int32x4_t d[4];
        for (int k = 0; k < 4; ++k) d[k] = vdotq_s32(vdupq_n_s32(0), w[k], vld1q_s8(a.q.data() + 64 * g + 16 * k));
        return vpaddq_s32(vpaddq_s32(d[0], d[1]), vpaddq_s32(d[2], d[3]));
    }
};

// The four block scales at sb divided by 256: an E4M3 code shifted left by 7 is, read as FP16 bits, its value / 256
// exactly (subnormals too; the format's scales are never negative). A widen, a shift and a convert instead of four
// loads from a 256-float table, which cost a single token's product a third of its instructions. rows_q_neon puts
// the 256 back into its final multiply: every product scales by a power of two, so the bits are the table's.
inline float32x4_t block_scales_div256(const uint8_t *sb) {
    uint32_t b4;
    std::memcpy(&b4, sb, 4);
    const uint16x4_t h = vshl_n_u16(vget_low_u16(vmovl_u8(vcreate_u8(b4))), 7);
    return vcvt_f32_f16(vreinterpret_f16_u16(h));
}
// two groups' at once, straight from memory into a vector register
inline void block_scales_div256_x2(const uint8_t *sb, float32x4_t &a, float32x4_t &b) {
    const uint16x8_t h = vshlq_n_u16(vmovl_u8(vld1_u8(sb)), 7);
    a = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h)));
    b = vcvt_high_f32_f16(vreinterpretq_f16_u16(h));
}
// signed scales (q4i): the E4M3 sign bit (7) to FP16's (15), the rest shifted as above
inline uint16x8_t signed_e4m3_as_f16(uint16x8_t w) {
    return vorrq_u16(vshlq_n_u16(vandq_u16(w, vdupq_n_u16(0x7F)), 7), vshlq_n_u16(vandq_u16(w, vdupq_n_u16(0x80)), 8));
}
inline float32x4_t block_scales_div256_signed(const uint8_t *sb) {
    uint32_t b4;
    std::memcpy(&b4, sb, 4);
    const uint16x8_t h = signed_e4m3_as_f16(vmovl_u8(vcreate_u8(b4)));
    return vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h)));
}
inline void block_scales_div256_signed_x2(const uint8_t *sb, float32x4_t &a, float32x4_t &b) {
    const uint16x8_t h = signed_e4m3_as_f16(vmovl_u8(vld1_u8(sb)));
    a = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h)));
    b = vcvt_high_f32_f16(vreinterpretq_f16_u16(h));
}

// One row against NT tokens' activations: out[j] = row . x_j / 256, before the matrix's global scale.
template <DType T, int NT>
inline void row_dots(const Group<T> &G, const uint8_t *row, size_t code_bytes, int64_t cols, const QAct *const *acts,
                     float *out) {
    const uint8_t *sb = row + code_bytes;
    const int64_t groups = cols / 64;
    float32x4_t acc[NT];
    for (int j = 0; j < NT; ++j) acc[j] = vdupq_n_f32(0.0f);
    const auto step = [&](int64_t g, float32x4_t wsv) {
        int8x16_t w[4];
        constexpr bool levels = T == DType::Q3 && NT >= 4;
        if constexpr (levels) G.load_levels(row, g, w);
        else G.load(row, g, w);
        for (int j = 0; j < NT; ++j) {
            int32x4_t d;
            if constexpr (levels) d = G.dot_levels(w, *acts[j], g);
            else d = G.dot(w, *acts[j], g);
            acc[j] = vfmaq_f32(acc[j], vcvtq_f32_s32(d), vmulq_f32(wsv, vld1q_f32(acts[j]->scale.data() + 4 * g)));
        }
    };
    int64_t g = 0;
    constexpr bool sgn = T == DType::Q4I;   // signed block scales
    for (; g + 1 < groups; g += 2) {
        float32x4_t s0, s1;
        if constexpr (sgn) block_scales_div256_signed_x2(sb + 4 * g, s0, s1);
        else block_scales_div256_x2(sb + 4 * g, s0, s1);
        step(g, s0);
        step(g + 1, s1);
    }
    if (g < groups) step(g, sgn ? block_scales_div256_signed(sb + 4 * g) : block_scales_div256(sb + 4 * g));
    for (int j = 0; j < NT; ++j) out[j] = vaddvq_f32(acc[j]);
}

// row r (its bytes at row) against n tokens, up to eight per pass: the row's codes are decoded once for all of them
template <DType T>
inline void row_tokens(const Group<T> &G, const uint8_t *row, size_t code_bytes, int64_t cols, const QAct *const *acts,
                       int n, float *const *ys, int64_t r, float post) {
    for (int t0 = 0; t0 < n; t0 += 8) {
        float out[8];
        const int nt = std::min(8, n - t0);
        switch (nt) {
        case 1: row_dots<T, 1>(G, row, code_bytes, cols, acts + t0, out); break;
        case 2: row_dots<T, 2>(G, row, code_bytes, cols, acts + t0, out); break;
        case 3: row_dots<T, 3>(G, row, code_bytes, cols, acts + t0, out); break;
        case 4: row_dots<T, 4>(G, row, code_bytes, cols, acts + t0, out); break;
        case 5: row_dots<T, 5>(G, row, code_bytes, cols, acts + t0, out); break;
        case 6: row_dots<T, 6>(G, row, code_bytes, cols, acts + t0, out); break;
        case 7: row_dots<T, 7>(G, row, code_bytes, cols, acts + t0, out); break;
        default: row_dots<T, 8>(G, row, code_bytes, cols, acts + t0, out); break;
        }
        for (int j = 0; j < nt; ++j) ys[t0 + j][r] = out[j] * post;
    }
}

template <DType T>
void rows_q_neon(const QView &m, const QAct *const *acts, int n, float *const *ys, int64_t r0, int64_t r1) {
    const Group<T> G;
    const size_t code_bytes = static_cast<size_t>(m.cols / kBlock) * static_cast<size_t>(block_code_bytes(m.type));
    const float post = m.global * level_scale(m.type) * 256.0f;   // row_dots's scales are / 256
    for (int64_t r = r0; r < r1; ++r)
        row_tokens(G, m.data + static_cast<size_t>(r) * m.row_bytes, code_bytes, m.cols, acts, n, ys, r, post);
}

// The GPU layout's four rows in the four 32-bit lanes (kernels.hpp): one 16-byte load is code word j of the four
// rows, and an SDOT by element of its decoded bytes with four activation bytes (one 32-bit lane of the token's group,
// loaded once) gives the four rows' partial sums at once: no transposes. A block's integer sums, its four rows'
// scales (one word of the scale bytes) and the token's block scale then go through the same multiply-adds, in the
// same order, as row_dots' lane for that block: the result is bit for bit the pack layout's. The four activation
// bytes sit side by side in the token's planes: q2 plane s of block b = lane b of p4's 16 bytes s, q4 word j's low /
// high nibbles = lane j % 4 of p2's 16 bytes 2 (j / 4) / 2 (j / 4) + 1.
template <DType T, int NT, int B>
inline void lanes_block(const int8x16_t lut, const uint8_t *p, size_t js, const int8x16_t (&x)[NT][4],
                        const float32x4_t ws, const float *const (&as)[NT], int64_t g, float32x4_t (&acc)[NT][4]) {
    int8x16_t w[4];   // the block's levels, four bytes of a row in each lane
    if constexpr (T == DType::Q2) {
        const uint8x16_t v = vld1q_u8(p + static_cast<size_t>(B) * js), m3 = vdupq_n_u8(3);
        w[0] = vqtbl1q_s8(lut, vandq_u8(v, m3));
        w[1] = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(v, 2), m3));
        w[2] = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(v, 4), m3));
        w[3] = vqtbl1q_s8(lut, vshrq_n_u8(v, 6));
    } else {
        const uint8x16_t v0 = vld1q_u8(p + static_cast<size_t>(2 * B) * js), m15 = vdupq_n_u8(15);
        const uint8x16_t v1 = vld1q_u8(p + static_cast<size_t>(2 * B + 1) * js);
        w[0] = vqtbl1q_s8(lut, vandq_u8(v0, m15));
        w[1] = vqtbl1q_s8(lut, vshrq_n_u8(v0, 4));
        w[2] = vqtbl1q_s8(lut, vandq_u8(v1, m15));
        w[3] = vqtbl1q_s8(lut, vshrq_n_u8(v1, 4));
    }
    for (int t = 0; t < NT; ++t) {
        int32x4_t d = vdupq_n_s32(0);
        if constexpr (T == DType::Q2) {   // plane s of block B: lane B of p4's vector s
            d = vdotq_laneq_s32(d, w[0], x[t][0], B);
            d = vdotq_laneq_s32(d, w[1], x[t][1], B);
            d = vdotq_laneq_s32(d, w[2], x[t][2], B);
            d = vdotq_laneq_s32(d, w[3], x[t][3], B);
        } else {   // words 2B and 2B + 1: lanes 2 (B % 2) and 2 (B % 2) + 1 of the B / 2 pair of vectors
            constexpr int h = 2 * (B / 2), l = 2 * (B % 2);
            d = vdotq_laneq_s32(d, w[0], x[t][h], l);
            d = vdotq_laneq_s32(d, w[1], x[t][h + 1], l);
            d = vdotq_laneq_s32(d, w[2], x[t][h], l + 1);
            d = vdotq_laneq_s32(d, w[3], x[t][h + 1], l + 1);
        }
        acc[t][B] = vfmaq_f32(acc[t][B], vcvtq_f32_s32(d), vmulq_n_f32(ws, as[t][4 * g + B]));
    }
}

// Where a quad's bytes lie: its first code word, the strides from chunk to chunk and from word to word, its first
// scale bytes and their chunk stride (the quad layout and tile8 differ only in these).
struct QuadAt {
    const uint8_t *codes, *sc;
    size_t cs, js, ss;
};

template <DType T, int NT>
inline void quad_lanes(const int8x16_t lut, const QuadAt &qa, const uint8_t *next, size_t next_bytes, int64_t nch,
                       const QAct *const *acts, float32x4_t out[NT]) {
    const uint8_t *sc = qa.sc;
    const size_t js = qa.js;   // from word j to word j + 1
    // per token and per block of the chunk (as row_dots' lanes), the four rows in the lanes
    float32x4_t acc[NT][4];
    const int8_t *xb[NT];
    const float *as[NT];
    for (int t = 0; t < NT; ++t) {
        for (int b = 0; b < 4; ++b) acc[t][b] = vdupq_n_f32(0.0f);
        xb[t] = (T == DType::Q2 ? acts[t]->p4 : acts[t]->p2).data();
        as[t] = acts[t]->scale.data();
    }
    // W + 1 short streams (512 bytes at 2048 columns) train no hardware prefetcher: the next quad (tile) is fetched
    // while this one is computed, a share of its cache lines a chunk
    const size_t lines = next ? next_bytes / 64 : 0, per = (lines + static_cast<size_t>(nch) - 1) / static_cast<size_t>(nch);
    for (int64_t g = 0; g < nch; ++g) {
        const uint8_t *p = qa.codes + qa.cs * static_cast<size_t>(g);
        for (size_t l = static_cast<size_t>(g) * per; l < std::min(lines, static_cast<size_t>(g + 1) * per); ++l)
            __builtin_prefetch(next + 64 * l);
        // the scale bytes [block][row]: block b's four rows' scales / 256 as one vector (rows in lanes)
        const uint8x16_t sb = vld1q_u8(sc + qa.ss * static_cast<size_t>(g));
        uint16x8_t h01 = vshlq_n_u16(vmovl_u8(vget_low_u8(sb)), 7), h23 = vshlq_n_u16(vmovl_high_u8(sb), 7);
        if constexpr (T == DType::Q4I) {   // signed E4M3: the sign bit (7, at 14 now) to FP16's (15)
            const uint16x8_t mag = vdupq_n_u16(0x3F80), sgn = vdupq_n_u16(0x4000);
            h01 = vorrq_u16(vandq_u16(h01, mag), vshlq_n_u16(vandq_u16(h01, sgn), 1));
            h23 = vorrq_u16(vandq_u16(h23, mag), vshlq_n_u16(vandq_u16(h23, sgn), 1));
        }
        int8x16_t x[NT][4];   // the token's 64 activation bytes of the chunk, in the format's plane order
        for (int t = 0; t < NT; ++t)
            for (int k = 0; k < 4; ++k) x[t][k] = vld1q_s8(xb[t] + 64 * g + 16 * k);
        lanes_block<T, NT, 0>(lut, p, js, x, vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h01))), as, g, acc);
        lanes_block<T, NT, 1>(lut, p, js, x, vcvt_high_f32_f16(vreinterpretq_f16_u16(h01)), as, g, acc);
        lanes_block<T, NT, 2>(lut, p, js, x, vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h23))), as, g, acc);
        lanes_block<T, NT, 3>(lut, p, js, x, vcvt_high_f32_f16(vreinterpretq_f16_u16(h23)), as, g, acc);
    }
    // vaddvq_f32's order: (b0 + b1) + (b2 + b3)
    for (int t = 0; t < NT; ++t) out[t] = vaddq_f32(vaddq_f32(acc[t][0], acc[t][1]), vaddq_f32(acc[t][2], acc[t][3]));
}

// tokens a pass: 4 vectors of accumulators each (the rows in lanes, the blocks apart, as row_dots sums them)
constexpr int kQuadTokens = 4;

template <DType T>
void rows_quad_neon(const QView &m, const QAct *const *acts, int n, float *const *ys, int64_t r0, int64_t r1) {
    const int8x16_t lut = [] {
        alignas(16) int8_t l[16] = {};
        std::memcpy(l, T == DType::Q2 ? kQ2Affine : T == DType::Q4 ? kQ4Affine : T == DType::Q4I ? kQ4I : kNvfp4x2,
                    T == DType::Q2 ? 4 : 16);
        return vld1q_s8(l);
    }();
    constexpr size_t W = T == DType::Q2 ? 4 : 8;   // code words a row and chunk
    const int64_t nch = m.cols / 64;
    const size_t unch = static_cast<size_t>(nch);
    const float post = m.global * level_scale(m.type) * 256.0f;
    // a range that splits a quad computes the whole quad and keeps its own rows
    const size_t qb = 4 * m.row_bytes, tb = 32 * m.row_bytes;
    for (int64_t q = r0 & ~int64_t{3}; q < r1; q += 4) {
        QuadAt qa;
        const uint8_t *next;
        size_t next_bytes;
        if (m.tile8) {   // quad i of the tile: words eight quads apart (kernels.hpp)
            const uint8_t *tile = m.data + static_cast<size_t>(q & ~int64_t{31}) * m.row_bytes;
            const size_t i = static_cast<size_t>((q >> 2) & 7);
            qa = {tile + 16 * i, tile + 128 * W * unch + 16 * i, 128 * W, 128, 128};
            next = i == 0 && (q & ~int64_t{31}) + 32 < r1 ? tile + tb : nullptr;   // the next tile, from quad 0
            next_bytes = tb;
        } else {
            const uint8_t *quad = m.data + static_cast<size_t>(q) * m.row_bytes;
            qa = {quad, quad + 16 * W * unch, 16, 16 * unch, 16};
            next = q + 4 < r1 ? quad + qb : nullptr;
            next_bytes = qb;
        }
        for (int t0 = 0; t0 < n; t0 += kQuadTokens) {   // a few tokens a pass: each one's bits the same alone
            float32x4_t out[kQuadTokens];
            const int nt = std::min(kQuadTokens, n - t0);
            const uint8_t *pre = t0 == 0 ? next : nullptr;
            switch (nt) {
            case 1: quad_lanes<T, 1>(lut, qa, pre, next_bytes, nch, acts + t0, out); break;
            case 2: quad_lanes<T, 2>(lut, qa, pre, next_bytes, nch, acts + t0, out); break;
            case 3: quad_lanes<T, 3>(lut, qa, pre, next_bytes, nch, acts + t0, out); break;
            default: quad_lanes<T, kQuadTokens>(lut, qa, pre, next_bytes, nch, acts + t0, out); break;
            }
            for (int j = 0; j < nt; ++j) {
                float o[4];
                vst1q_f32(o, out[j]);
                for (int r = 0; r < 4; ++r)
                    if (q + r >= r0 && q + r < r1) ys[t0 + j][q + r] = o[r] * post;
            }
        }
    }
}

// ---- SMMLA (FEAT_I8MM: a 2 x 8 by 8 x 2 int8 product an instruction; the 8 Elite's Oryon cores, not the 720G's
// A76): q4 quads (the quad layout or tile8) against pairs of tokens, bit for bit rows_quad_neon<Q4>. A block's
// integer dot is the same 16 products in int32, and every (row, token) keeps rows_quad_neon's four accumulators (the
// block's place in its chunk), the same fma with the same scale product, summed the same way at the end: only the
// lanes hold (row, token) pairs instead of four rows. Half the dot-product instructions of SDOT. Off unless
// GEODE_I8MM=1 (i8mm_here).
bool i8mm_here() {
#if defined(__linux__)
    // GEODE_I8MM=1 only: measured on the RedMagic, a 1000-token prompt on the CPU was 3-4% slower with it than with
    // the SDOT kernels (56.0 / 54.6 vs 58.4 / 57.1 tok/s), though one core alone gains 11-16% at 4-16 tokens
    static const bool yes = (getauxval(AT_HWCAP2) & (1UL << 13)) != 0 &&   // HWCAP2_I8MM
                            std::getenv("GEODE_I8MM") && std::atoi(std::getenv("GEODE_I8MM")) == 1;
    return yes;
#else
    return false;
#endif
}

// Tokens a0 and a1 (a1 nullptr: zeros) as SMMLA's right operands: per chunk g and block B the 16 bytes lanes_block<Q4>
// multiplies with word 2B's levels (x[h] lane l, then x[h + 1] lane l, for each token), then the 16 with word
// 2B + 1's (lanes l + 1): 32 bytes a block; the scales as [s0 s1 s0 s1] a block
void mmla_pair(const QAct *a0, const QAct *a1, int64_t nch, int8_t *q, float *s) {
    for (int64_t g = 0; g < nch; ++g)
        for (int B = 0; B < 4; ++B) {
            const int h = 2 * (B / 2), l = 2 * (B % 2);
            int8_t *o = q + (g * 4 + B) * 32;
            for (int t = 0; t < 2; ++t) {
                const QAct *a = t ? a1 : a0;
                int8_t *first = o + 8 * t, *second = o + 16 + 8 * t;
                if (!a) {
                    std::memset(first, 0, 8);
                    std::memset(second, 0, 8);
                    continue;
                }
                const int8_t *p = a->p2.data() + 64 * g;
                std::memcpy(first, p + 16 * h + 4 * l, 4);
                std::memcpy(first + 4, p + 16 * (h + 1) + 4 * l, 4);
                std::memcpy(second, p + 16 * h + 4 * (l + 1), 4);
                std::memcpy(second + 4, p + 16 * (h + 1) + 4 * (l + 1), 4);
            }
            const float s0 = a0->scale[static_cast<size_t>(4 * g + B)];
            const float s1 = a1 ? a1->scale[static_cast<size_t>(4 * g + B)] : 0.0f;
            float *os = s + (g * 4 + B) * 4;
            os[0] = s0, os[1] = s1, os[2] = s0, os[3] = s1;
        }
}

// one quad against TP token pairs: out[P][0] = rows 0-1 x the pair's tokens (lanes r0t0 r0t1 r1t0 r1t1), [1] rows 2-3
template <int TP>
__attribute__((target("arch=armv8.2-a+dotprod+fp16+i8mm")))
inline void quad_mmla(const int8x16_t lut, const QuadAt &qa, int64_t nch, const int8_t *const *pq,
                      const float *const *ps, float32x4_t (&out)[TP][2]) {
    float32x4_t acc[TP][2][4];
    for (int P = 0; P < TP; ++P)
        for (int h = 0; h < 2; ++h)
            for (int b = 0; b < 4; ++b) acc[P][h][b] = vdupq_n_f32(0.0f);
    const uint8x16_t m15 = vdupq_n_u8(15);
    for (int64_t g = 0; g < nch; ++g) {
        const uint8_t *p = qa.codes + qa.cs * static_cast<size_t>(g);
        const uint8x16_t sb = vld1q_u8(qa.sc + qa.ss * static_cast<size_t>(g));
        const uint16x8_t h01 = vshlq_n_u16(vmovl_u8(vget_low_u8(sb)), 7), h23 = vshlq_n_u16(vmovl_high_u8(sb), 7);
        const float32x4_t ws[4] = {vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h01))),
                                   vcvt_high_f32_f16(vreinterpretq_f16_u16(h01)),
                                   vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(h23))),
                                   vcvt_high_f32_f16(vreinterpretq_f16_u16(h23))};
        for (int B = 0; B < 4; ++B) {
            const uint8x16_t v0 = vld1q_u8(p + static_cast<size_t>(2 * B) * qa.js);
            const uint8x16_t v1 = vld1q_u8(p + static_cast<size_t>(2 * B + 1) * qa.js);
            const uint32x4_t w0 = vreinterpretq_u32_s8(vqtbl1q_s8(lut, vandq_u8(v0, m15)));
            const uint32x4_t w1 = vreinterpretq_u32_s8(vqtbl1q_s8(lut, vshrq_n_u8(v0, 4)));
            const uint32x4_t w2 = vreinterpretq_u32_s8(vqtbl1q_s8(lut, vandq_u8(v1, m15)));
            const uint32x4_t w3 = vreinterpretq_u32_s8(vqtbl1q_s8(lut, vshrq_n_u8(v1, 4)));
            // a row's 8 values a 64-bit half: [w0 | w1] of rows 0 and 1, of rows 2 and 3; then [w2 | w3]
            const int8x16_t a01 = vreinterpretq_s8_u32(vzip1q_u32(w0, w1)), a23 = vreinterpretq_s8_u32(vzip2q_u32(w0, w1));
            const int8x16_t b01 = vreinterpretq_s8_u32(vzip1q_u32(w2, w3)), b23 = vreinterpretq_s8_u32(vzip2q_u32(w2, w3));
            const float32x4_t wd01 = vzip1q_f32(ws[B], ws[B]), wd23 = vzip2q_f32(ws[B], ws[B]);   // [r0 r0 r1 r1]
            for (int P = 0; P < TP; ++P) {
                const int8_t *x = pq[P] + (g * 4 + B) * 32;
                const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
                const float32x4_t sp = vld1q_f32(ps[P] + (g * 4 + B) * 4);   // [s0 s1 s0 s1]
                const int32x4_t d01 = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), a01, x0), b01, x1);
                const int32x4_t d23 = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), a23, x0), b23, x1);
                acc[P][0][B] = vfmaq_f32(acc[P][0][B], vcvtq_f32_s32(d01), vmulq_f32(wd01, sp));
                acc[P][1][B] = vfmaq_f32(acc[P][1][B], vcvtq_f32_s32(d23), vmulq_f32(wd23, sp));
            }
        }
    }
    for (int P = 0; P < TP; ++P)   // rows_quad_neon's order: (b0 + b1) + (b2 + b3)
        for (int h = 0; h < 2; ++h)
            out[P][h] = vaddq_f32(vaddq_f32(acc[P][h][0], acc[P][h][1]), vaddq_f32(acc[P][h][2], acc[P][h][3]));
}

// rows [r0, r1) against n tokens whose pairs mmla_pair packed (pq / ps: one a pair)
void rows_quad_mmla(const QView &m, const int8_t *const *pq, const float *const *ps, int n, float *const *ys, int64_t r0,
                    int64_t r1) {
    const int8x16_t lut = vld1q_s8(kQ4Affine);
    constexpr size_t W = 8;
    const int64_t nch = m.cols / 64;
    const size_t unch = static_cast<size_t>(nch);
    const float post = m.global * level_scale(m.type) * 256.0f;
    const int np = (n + 1) / 2;
    for (int64_t q = r0 & ~int64_t{3}; q < r1; q += 4) {
        QuadAt qa;
        if (m.tile8) {   // quad i of the tile: words eight quads apart (kernels.hpp)
            const uint8_t *tile = m.data + static_cast<size_t>(q & ~int64_t{31}) * m.row_bytes;
            const size_t i = static_cast<size_t>((q >> 2) & 7);
            qa = {tile + 16 * i, tile + 128 * W * unch + 16 * i, 128 * W, 128, 128};
        } else {
            const uint8_t *quad = m.data + static_cast<size_t>(q) * m.row_bytes;
            qa = {quad, quad + 16 * W * unch, 16, 16 * unch, 16};
        }
        for (int P0 = 0; P0 < np; P0 += 2) {
            const int tp = std::min(2, np - P0);
            float32x4_t out[2][2];
            if (tp == 2) {
                quad_mmla<2>(lut, qa, nch, pq + P0, ps + P0, out);
            } else {
                float32x4_t one[1][2];
                quad_mmla<1>(lut, qa, nch, pq + P0, ps + P0, one);
                out[0][0] = one[0][0], out[0][1] = one[0][1];
            }
            for (int P = 0; P < tp; ++P)
                for (int h = 0; h < 2; ++h) {
                    float o[4];
                    vst1q_f32(o, out[P][h]);
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j) {
                            const int t = 2 * (P0 + P) + j;
                            const int64_t r = q + 2 * h + i;
                            if (t < n && r >= r0 && r < r1) ys[t][r] = o[2 * i + j] * post;
                        }
                }
        }
    }
}

#endif

}  // namespace

DType dtype_from_name(const std::string &n) {
    if (n == "f32") return DType::F32;
    if (n == "bf16") return DType::BF16;
    if (n == "q2") return DType::Q2;
    if (n == "q3") return DType::Q3;
    if (n == "nvfp4") return DType::NVFP4;
    if (n == "q4") return DType::Q4;
    if (n == "q8") return DType::Q8;
    if (n == "q4i") return DType::Q4I;
    return DType::Unknown;
}

const char *dtype_name(DType t) {
    switch (t) {
    case DType::F32: return "f32";
    case DType::BF16: return "bf16";
    case DType::Q2: return "q2";
    case DType::Q3: return "q3";
    case DType::NVFP4: return "nvfp4";
    case DType::Q4: return "q4";
    case DType::Q8: return "q8";
    case DType::Q4I: return "q4i";
    default: return "?";
    }
}

bool is_quantized(DType t) {
    return t == DType::Q2 || t == DType::Q3 || t == DType::NVFP4 || t == DType::Q4 || t == DType::Q8 || t == DType::Q4I;
}

int block_code_bytes(DType t) {
    switch (t) {
    case DType::Q2: return 4;
    case DType::Q3: return 6;
    case DType::NVFP4: return 8;
    case DType::Q4: return 8;
    case DType::Q4I: return 8;
    case DType::Q8: return 16;
    default: return 0;
    }
}

size_t row_bytes(DType t, int64_t cols) {
    return static_cast<size_t>(cols / kBlock) * static_cast<size_t>(block_code_bytes(t) + 1);
}

namespace {
// a code byte in the affine order (to) and back (from): q2 levels 1 3 -1 -3 -> 2c - 3, q4 1..15 -1..-15 -> 2c - 15
struct AffineBytes {
    std::array<uint8_t, 256> to{}, from{};
};
const AffineBytes &affine_bytes(DType t) {
    static const auto make = [](DType type) {
        AffineBytes a;
        for (int b = 0; b < 256; ++b) {
            int to = 0, from = 0;
            if (type == DType::NVFP4 || type == DType::Q4I) {   // nvfp4: no GPU kernel reads it; q4i: affine already
                to = from = b;
            } else if (type == DType::Q2) {
                static constexpr int f[4] = {2, 3, 1, 0}, g[4] = {3, 2, 0, 1};
                for (int k = 0; k < 4; ++k) {
                    to |= f[(b >> (2 * k)) & 3] << (2 * k);
                    from |= g[(b >> (2 * k)) & 3] << (2 * k);
                }
            } else {
                for (int k = 0; k < 2; ++k) {
                    const int c = (b >> (4 * k)) & 15;
                    to |= (c < 8 ? c + 8 : 15 - c) << (4 * k);
                    from |= (c >= 8 ? c - 8 : 15 - c) << (4 * k);
                }
            }
            a.to[static_cast<size_t>(b)] = static_cast<uint8_t>(to);
            a.from[static_cast<size_t>(b)] = static_cast<uint8_t>(from);
        }
        return a;
    };
    static const AffineBytes q2 = make(DType::Q2), q4 = make(DType::Q4), nv = make(DType::NVFP4), q4i = make(DType::Q4I);
    return t == DType::Q2 ? q2 : t == DType::Q4 ? q4 : t == DType::Q4I ? q4i : nv;
}
}  // namespace

bool quad_capable(DType t, int64_t rows, int64_t cols) {
    // not q3: its decode in the lanes (a high-bit shuffle and test per plane) runs slower than the pack's kernel,
    // one token on par, 4 to 16 tokens -12..-38% (measured 2026-10-27)
    return (t == DType::Q2 || t == DType::Q4 || t == DType::NVFP4 || t == DType::Q4I) && rows > 0 && rows % 4 == 0 &&
           cols > 0 && cols % 64 == 0;
}

void to_quad(DType t, int64_t rows, int64_t cols, const uint8_t *src, uint8_t *dst) {
    const size_t rb = row_bytes(t, cols), W = static_cast<size_t>(block_code_bytes(t));   // words a row and chunk
    const size_t nch = static_cast<size_t>(cols / 64), code = static_cast<size_t>(cols / kBlock) * W;
    const auto &ab = affine_bytes(t);
    std::vector<uint8_t> quad(4 * rb);
    for (int64_t q = 0; q < rows; q += 4) {
        std::memcpy(quad.data(), src + static_cast<size_t>(q) * rb, quad.size());
        uint8_t *g = dst + static_cast<size_t>(q) * rb;
        for (size_t j = 0; j < W; ++j)
            for (size_t c = 0; c < nch; ++c)
                for (size_t r = 0; r < 4; ++r) {
                    const uint8_t *w = &quad[r * rb + (c * W + j) * 4];
                    uint8_t *o = g + ((j * nch + c) * 4 + r) * 4;
                    for (int k = 0; k < 4; ++k) o[k] = ab.to[w[k]];
                }
        for (size_t c = 0; c < nch; ++c)
            for (size_t b = 0; b < 4; ++b)
                for (size_t r = 0; r < 4; ++r) g[4 * code + (c * 4 + b) * 4 + r] = quad[r * rb + code + c * 4 + b];
    }
}

bool tile8_capable(DType t, int64_t rows, int64_t cols) {
    return t == DType::Q4 && rows > 0 && rows % 32 == 0 && cols > 0 && cols % 64 == 0;
}

void to_tile8(DType t, int64_t rows, int64_t cols, const uint8_t *src, uint8_t *dst) {
    const size_t rb = row_bytes(t, cols), W = static_cast<size_t>(block_code_bytes(t));
    const size_t nch = static_cast<size_t>(cols / 64), code = static_cast<size_t>(cols / kBlock) * W;
    const auto &ab = affine_bytes(t);
    std::vector<uint8_t> tile(32 * rb);
    for (int64_t t0 = 0; t0 < rows; t0 += 32) {
        std::memcpy(tile.data(), src + static_cast<size_t>(t0) * rb, tile.size());
        uint8_t *g = dst + static_cast<size_t>(t0) * rb;
        for (size_t i = 0; i < 8; ++i)
            for (size_t r = 0; r < 4; ++r) {
                const uint8_t *row = &tile[(4 * i + r) * rb];
                for (size_t c = 0; c < nch; ++c) {
                    for (size_t j = 0; j < W; ++j) {
                        uint8_t *o = g + ((c * W + j) * 8 + i) * 16 + r * 4;
                        for (size_t k = 0; k < 4; ++k) o[k] = ab.to[row[(c * W + j) * 4 + k]];
                    }
                    for (size_t b = 0; b < 4; ++b) g[128 * W * nch + (c * 8 + i) * 16 + b * 4 + r] = row[code + c * 4 + b];
                }
            }
    }
}

namespace {
// a pack row's codes as affine codes, one byte a value (cols values)
void pack_to_affine(DType t, const uint8_t *row, int64_t cols, uint8_t *c) {
    for (int64_t v = 0; v < cols; ++v) {
        int code;
        if (t == DType::Q4) {
            code = (row[v / 2] >> (4 * (v % 2))) & 15;
            c[v] = static_cast<uint8_t>(code < 8 ? code + 8 : 15 - code);
        } else if (t == DType::Q2) {
            code = (row[v / 4] >> (2 * (v % 4))) & 3;
            static constexpr uint8_t f[4] = {2, 3, 1, 0};
            c[v] = f[code];
        } else {   // q3: 24 bytes a 64-group, the low 2 bits as q2 then the high bits in value order
            const uint8_t *g = row + (v / 64) * 24;
            const int64_t u = v % 64;
            code = ((g[u / 4] >> (2 * (u % 4))) & 3) | (((g[16 + u / 8] >> (u % 8)) & 1) << 2);
            c[v] = static_cast<uint8_t>(code < 4 ? code + 4 : 7 - code);
        }
    }
}
// the inverse: affine codes -> the pack's row codes (the scales are copied by the callers)
void affine_to_pack(DType t, const uint8_t *c, int64_t cols, uint8_t *row) {
    std::memset(row, 0, static_cast<size_t>(cols / kBlock) * static_cast<size_t>(block_code_bytes(t)));
    for (int64_t v = 0; v < cols; ++v) {
        const int a = c[v];
        if (t == DType::Q4) {
            row[v / 2] |= static_cast<uint8_t>((a >= 8 ? a - 8 : 15 - a) << (4 * (v % 2)));
        } else if (t == DType::Q2) {
            static constexpr uint8_t g[4] = {3, 2, 0, 1};
            row[v / 4] |= static_cast<uint8_t>(g[a] << (2 * (v % 4)));
        } else {
            const int code = a >= 4 ? a - 4 : 7 - a;
            uint8_t *gp = row + (v / 64) * 24;
            const int64_t u = v % 64;
            gp[u / 4] |= static_cast<uint8_t>((code & 3) << (2 * (u % 4)));
            gp[16 + u / 8] |= static_cast<uint8_t>(((code >> 2) & 1) << (u % 8));
        }
    }
}
// the lanes' values: v(q, b)
constexpr int kLane[4][4] = {{0, 2, 4, 6}, {8, 10, 12, 14}, {1, 3, 5, 7}, {9, 11, 13, 15}};
// a row's W words of one 64-value chunk from its 64 affine codes
void chunk_words(DType t, const uint8_t *c, uint32_t *w) {
    const int W = tile8x_words(t);
    for (int j = 0; j < W; ++j) w[j] = 0;
    for (int k = 0; k < 4; ++k) {
        const uint8_t *v = c + 16 * k;
        for (int b = 0; b < 4; ++b) {
            uint32_t byte;
            if (t == DType::Q4) {
                w[2 * k] |= static_cast<uint32_t>(v[kLane[0][b]] | v[kLane[2][b]] << 4) << (8 * b);
                w[2 * k + 1] |= static_cast<uint32_t>(v[kLane[1][b]] | v[kLane[3][b]] << 4) << (8 * b);
                continue;
            }
            byte = 0;
            for (int q = 0; q < 4; ++q) byte |= static_cast<uint32_t>(v[kLane[q][b]] & 3) << (2 * q);
            w[k] |= byte << (8 * b);
            if (t == DType::Q3) {
                uint32_t hb = 0;
                for (int q = 0; q < 4; ++q) hb |= static_cast<uint32_t>((v[kLane[q][b]] >> 2) & 1) << q;
                w[4 + k / 2] |= hb << (8 * b + 4 * (k % 2));
            }
        }
    }
}
// where word j of chunk c of row r of a tile lies (bytes from the tile's start), and its scales
inline size_t t8x_word(int W, int64_t c, int j, int64_t r) {
    return ((static_cast<size_t>(c) * W + j) * 8 + static_cast<size_t>((r >> 2) & 7)) * 16 + static_cast<size_t>(r & 3) * 4;
}
inline size_t t8x_scale(int W, size_t nch, int64_t c, int b, int64_t r) {
    return 128 * static_cast<size_t>(W) * nch + (static_cast<size_t>(c) * 8 + static_cast<size_t>((r >> 2) & 7)) * 16 +
           static_cast<size_t>(b) * 4 + static_cast<size_t>(r & 3);
}
}  // namespace

bool tile8x_capable(DType t, int64_t rows, int64_t cols) {
    return (t == DType::Q2 || t == DType::Q3 || t == DType::Q4) && rows > 0 && rows % 32 == 0 && cols > 0 &&
           cols % 64 == 0;
}

int tile8x_words(DType t) { return t == DType::Q4 ? 8 : t == DType::Q3 ? 6 : 4; }

void to_tile8x(DType t, int64_t rows, int64_t cols, const uint8_t *src, uint8_t *dst) {
    const size_t rb = row_bytes(t, cols), nch = static_cast<size_t>(cols / 64);
    const size_t code = static_cast<size_t>(cols / kBlock) * static_cast<size_t>(block_code_bytes(t));
    const int W = tile8x_words(t);
    std::vector<uint8_t> c(static_cast<size_t>(cols));
    uint32_t w[8];
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t *row = src + static_cast<size_t>(r) * rb;
        uint8_t *g = dst + static_cast<size_t>(r & ~int64_t{31}) * rb;
        pack_to_affine(t, row, cols, c.data());
        for (size_t ch = 0; ch < nch; ++ch) {
            chunk_words(t, c.data() + 64 * ch, w);
            for (int j = 0; j < W; ++j) std::memcpy(g + t8x_word(W, static_cast<int64_t>(ch), j, r), &w[j], 4);
            for (int b = 0; b < 4; ++b) g[t8x_scale(W, nch, static_cast<int64_t>(ch), b, r)] = row[code + ch * 4 + static_cast<size_t>(b)];
        }
    }
}

void tile8x_row(DType t, int64_t cols, const uint8_t *data, int64_t r, uint8_t *out) {
    const size_t rb = row_bytes(t, cols), nch = static_cast<size_t>(cols / 64);
    const size_t code = static_cast<size_t>(cols / kBlock) * static_cast<size_t>(block_code_bytes(t));
    const int W = tile8x_words(t);
    const uint8_t *g = data + static_cast<size_t>(r & ~int64_t{31}) * rb;
    std::vector<uint8_t> c(static_cast<size_t>(cols));
    for (size_t ch = 0; ch < nch; ++ch) {
        uint32_t w[8];
        for (int j = 0; j < W; ++j) std::memcpy(&w[j], g + t8x_word(W, static_cast<int64_t>(ch), j, r), 4);
        for (int k = 0; k < 4; ++k)
            for (int b = 0; b < 4; ++b)
                for (int q = 0; q < 4; ++q) {
                    uint32_t a;
                    if (t == DType::Q4) a = (w[2 * k + (q & 1)] >> (8 * b + 4 * (q >> 1))) & 15;
                    else a = (w[k] >> (8 * b + 2 * q)) & 3;
                    if (t == DType::Q3) a |= ((w[4 + k / 2] >> (8 * b + 4 * (k % 2) + q)) & 1) << 2;
                    c[64 * ch + 16 * static_cast<size_t>(k) + static_cast<size_t>(kLane[q][b])] = static_cast<uint8_t>(a);
                }
        for (int b = 0; b < 4; ++b) out[code + ch * 4 + static_cast<size_t>(b)] = g[t8x_scale(W, nch, static_cast<int64_t>(ch), b, r)];
    }
    affine_to_pack(t, c.data(), cols, out);
}

void tile8x_lane(DType t, int64_t cols, const uint8_t *data, int64_t r, int64_t c, int blk, int q, int8_t out[4]) {
    const size_t rb = row_bytes(t, cols);
    const int W = tile8x_words(t);
    const uint8_t *g = data + static_cast<size_t>(r & ~int64_t{31}) * rb;
    auto word = [&](int j) {
        uint32_t w;
        std::memcpy(&w, g + t8x_word(W, c, j, r), 4);
        return w;
    };
    uint32_t two_c, off;   // 2c in each byte, and the level offset's bias (2^bits - 1 = 128 - off)
    if (t == DType::Q4) {
        const uint32_t w = word(2 * blk + (q & 1));
        two_c = (q >> 1) ? (w >> 3) & 0x1E1E1E1Eu : (w << 1) & 0x1E1E1E1Eu;
        off = 0x71717171u;
    } else {
        const uint32_t w = word(blk);
        two_c = q == 0 ? (w << 1) & 0x06060606u : (w >> (2 * q - 1)) & 0x06060606u;
        off = 0x7D7D7D7Du;
        if (t == DType::Q3) {
            two_c |= ((word(4 + blk / 2) >> (4 * (blk % 2) + q)) & 0x01010101u) << 3;
            off = 0x79797979u;
        }
    }
    const uint32_t lv = (two_c + off) ^ 0x80808080u;
    for (int b = 0; b < 4; ++b) out[b] = static_cast<int8_t>((lv >> (8 * b)) & 0xFF);
}

void quad_row(const QView &m, int64_t r, uint8_t *out) {
    const size_t W = static_cast<size_t>(block_code_bytes(m.type));
    const size_t nch = static_cast<size_t>(m.cols / 64), code = static_cast<size_t>(m.cols / kBlock) * W;
    const size_t i = static_cast<size_t>(r & 3);
    const auto &ab = affine_bytes(m.type);
    if (m.tile8) {   // quad (r / 4) % 8 of its tile, row r % 4 of the quad
        const uint8_t *g = m.data + static_cast<size_t>(r & ~int64_t{31}) * m.row_bytes;
        const size_t qi = static_cast<size_t>((r >> 2) & 7);
        for (size_t c = 0; c < nch; ++c) {
            for (size_t j = 0; j < W; ++j) {
                const uint8_t *w = g + ((c * W + j) * 8 + qi) * 16 + i * 4;
                for (int k = 0; k < 4; ++k) out[(c * W + j) * 4 + static_cast<size_t>(k)] = ab.from[w[k]];
            }
            for (size_t b = 0; b < 4; ++b) out[code + c * 4 + b] = g[128 * W * nch + (c * 8 + qi) * 16 + b * 4 + i];
        }
        return;
    }
    const uint8_t *g = m.data + static_cast<size_t>(r & ~int64_t{3}) * m.row_bytes;
    for (size_t c = 0; c < nch; ++c)
        for (size_t j = 0; j < W; ++j) {
            const uint8_t *w = g + ((j * nch + c) * 4 + i) * 4;
            for (int k = 0; k < 4; ++k) out[(c * W + j) * 4 + static_cast<size_t>(k)] = ab.from[w[k]];
        }
    for (size_t c = 0; c < nch; ++c)
        for (size_t b = 0; b < 4; ++b) out[code + c * 4 + b] = g[4 * code + (c * 4 + b) * 4 + i];
}

float e4m3_value(uint8_t b) { return e4m3_table()[b]; }

bool have_i8mm_kernels() {
#if defined(GEODE_NEON_DOTPROD)
    return i8mm_here();
#else
    return false;
#endif
}

bool have_dotprod_kernels() {
#if defined(GEODE_NEON_DOTPROD)
    return true;
#else
    return false;
#endif
}

namespace {
// row r in the pack's layout: in place, or a quad view's row gathered into tmp
const uint8_t *row_of(const QView &m, int64_t r, std::vector<uint8_t> &tmp) {
    if (!m.quad) return m.data + static_cast<size_t>(r) * m.row_bytes;
    tmp.resize(m.row_bytes);
    quad_row(m, r, tmp.data());
    return tmp.data();
}
}  // namespace

void dequant_row(const QView &m, int64_t r, float *out) {
    const auto &e4 = e4m3_table();
    std::vector<uint8_t> tmp;
    const uint8_t *row = row_of(m, r, tmp);
    const int nb = static_cast<int>(m.cols / kBlock);
    const uint8_t *scales = row + static_cast<size_t>(nb) * static_cast<size_t>(block_code_bytes(m.type));
    const float ls = level_scale(m.type);
    for (int b = 0; b < nb; ++b) {
        int8_t v[kBlock];
        block_ints(m.type, row, b, v);
        // the same product order as the quantizer: level x (E4M3 x global)
        const float s = e4[scales[b]] * m.global;
        for (int i = 0; i < kBlock; ++i) out[b * kBlock + i] = (static_cast<float>(v[i]) * ls) * s;
    }
}

void matvec_rows(const QView &m, const float *x, float *y, int64_t r0, int64_t r1) {
    const auto &e4 = e4m3_table();
    const int nb = static_cast<int>(m.cols / kBlock);
    const size_t code_bytes = static_cast<size_t>(nb) * static_cast<size_t>(block_code_bytes(m.type));
    const float ls = level_scale(m.type);
    std::vector<uint8_t> tmp;
    for (int64_t r = r0; r < r1; ++r) {
        const uint8_t *row = row_of(m, r, tmp);
        const uint8_t *scales = row + code_bytes;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            int8_t v[kBlock];
            block_ints(m.type, row, b, v);
            const float *xb = x + b * kBlock;
            float s = 0.0f;
            for (int i = 0; i < kBlock; ++i) s += static_cast<float>(v[i]) * xb[i];
            acc += s * e4[scales[b]];
        }
        y[r] = acc * m.global * ls;
    }
}

namespace {
// block b of x: its int8 values, scale and sum (the one rounding every way of quantizing uses)
inline void quantize_block(const float *x, int64_t b, QAct &a) {
    const float *xb = x + b * kBlock;
    float amax = 0.0f;
    for (int i = 0; i < kBlock; ++i) amax = std::max(amax, std::fabs(xb[i]));
    const float s = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    a.scale[static_cast<size_t>(b)] = s;
    int32_t sum = 0;
    for (int i = 0; i < kBlock; ++i) {
        const auto q = static_cast<int8_t>(std::clamp(std::lrint(xb[i] * inv), -127L, 127L));
        a.q[static_cast<size_t>(b * kBlock + i)] = q;
        sum += q;
    }
    a.bsum[static_cast<size_t>(b)] = sum;
}
// the orders the packed codes come out of a 16-byte load in: lane j of plane s = value c*j + s
inline void lanes4(int64_t g, QAct &a) {
    for (int s = 0; s < 4; ++s)
        for (int j = 0; j < 16; ++j) a.p4[static_cast<size_t>(g + 16 * s + j)] = a.q[static_cast<size_t>(g + 4 * j + s)];
}
inline void lanes2(int64_t g, QAct &a) {
    for (int s = 0; s < 2; ++s)
        for (int j = 0; j < 16; ++j) a.p2[static_cast<size_t>(g + 16 * s + j)] = a.q[static_cast<size_t>(g + 2 * j + s)];
}
}  // namespace

void prepare_activation(int64_t cols, QAct &a) {
    const int64_t nb = cols / kBlock;
    a.cols = cols;
    a.q.resize(static_cast<size_t>(cols));
    a.p4.resize(static_cast<size_t>(cols));
    a.p2.resize(static_cast<size_t>(cols));
    a.scale.resize(static_cast<size_t>(nb));
    a.bsum.resize(static_cast<size_t>(nb));
}

void quantize_activation(const float *x, int64_t cols, QAct &a) {
    prepare_activation(cols, a);
    const int64_t nb = cols / kBlock;
    for (int64_t b = 0; b < nb; ++b) quantize_block(x, b, a);
    for (int64_t g = 0; g + 64 <= cols; g += 64) lanes4(g, a);
    for (int64_t g = 0; g + 32 <= cols; g += 32) lanes2(g, a);
}

void quantize_activation_group(const float *x, int64_t g, QAct &a) {
    for (int64_t b = g / kBlock; b < (g + 64) / kBlock; ++b) quantize_block(x, b, a);
    lanes4(g, a);
    lanes2(g, a);
    lanes2(g + 32, a);
}

void matvec_rows_q_ref(const QView &m, const QAct &a, float *y, int64_t r0, int64_t r1) {
    const auto &e4 = e4m3_table();
    const int nb = static_cast<int>(m.cols / kBlock);
    const size_t code_bytes = static_cast<size_t>(nb) * static_cast<size_t>(block_code_bytes(m.type));
    std::vector<uint8_t> tmp;
    for (int64_t r = r0; r < r1; ++r) {
        const uint8_t *row = row_of(m, r, tmp);
        const uint8_t *scales = row + code_bytes;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            int8_t v[kBlock];
            block_ints(m.type, row, b, v);
            int32_t s = 0;
            for (int i = 0; i < kBlock; ++i) s += v[i] * a.q[static_cast<size_t>(b * kBlock + i)];
            acc += static_cast<float>(s) * (e4[scales[b]] * a.scale[static_cast<size_t>(b)]);
        }
        y[r] = acc * m.global * level_scale(m.type);
    }
}

void matvec_rows_q_multi(const QView &m, const QAct *const *acts, int n, float *const *ys, int64_t r0, int64_t r1) {
    // A long batch in tiles: a block of rows small enough to stay in L2 against a few tokens at a time (their
    // activations stay in L1). Rows outer and every token inner would stream the whole batch's activations from DRAM
    // once per row (1+ MB per row at 512 tokens). The tiles keep the groups of four tokens, so the bits are the same.
    constexpr int kTileTokens = 16;
#if defined(GEODE_NEON_DOTPROD)
    if (m.quad && m.type == DType::Q4 && n >= 2 && i8mm_here()) {
        // SMMLA: every token pair packed once, then the same row blocks and groups of tokens as below
        const int64_t nch = m.cols / 64;
        const int np = (n + 1) / 2;
        thread_local std::vector<int8_t> qbuf;
        thread_local std::vector<float> sbuf;
        thread_local std::vector<const int8_t *> pq;
        thread_local std::vector<const float *> ps;
        const size_t qper = static_cast<size_t>(nch) * 4 * 32, sper = static_cast<size_t>(nch) * 4 * 4;
        if (qbuf.size() < qper * static_cast<size_t>(np)) qbuf.resize(qper * static_cast<size_t>(np));
        if (sbuf.size() < sper * static_cast<size_t>(np)) sbuf.resize(sper * static_cast<size_t>(np));
        pq.resize(static_cast<size_t>(np));
        ps.resize(static_cast<size_t>(np));
        for (int P = 0; P < np; ++P) {
            int8_t *q = qbuf.data() + qper * static_cast<size_t>(P);
            float *sc = sbuf.data() + sper * static_cast<size_t>(P);
            mmla_pair(acts[2 * P], 2 * P + 1 < n ? acts[2 * P + 1] : nullptr, nch, q, sc);
            pq[static_cast<size_t>(P)] = q;
            ps[static_cast<size_t>(P)] = sc;
        }
        const int64_t block = n > kTileTokens ? std::max<int64_t>(16, static_cast<int64_t>((96u << 10) / std::max<size_t>(1, m.row_bytes))) &
                                                    ~int64_t{3}
                                              : r1 - r0;
        for (int64_t a = r0; a < r1; a += block)
            for (int t = 0; t < n; t += kTileTokens)
                rows_quad_mmla(m, pq.data() + t / 2, ps.data() + t / 2, std::min(kTileTokens, n - t), ys + t, a,
                               std::min(r1, a + block));
        return;
    }
#endif
    if (n > kTileTokens) {
        const int64_t block = std::max<int64_t>(16, static_cast<int64_t>((96u << 10) / std::max<size_t>(1, m.row_bytes))) &
                              ~int64_t{3};   // whole quads
        for (int64_t a = r0; a < r1; a += block)
            for (int t = 0; t < n; t += kTileTokens)
                matvec_rows_q_multi(m, acts + t, std::min(kTileTokens, n - t), ys + t, a, std::min(r1, a + block));
        return;
    }
#if defined(GEODE_NEON_DOTPROD)
    if (m.quad) {
        if (m.type == DType::Q2) return rows_quad_neon<DType::Q2>(m, acts, n, ys, r0, r1);
        if (m.type == DType::Q4) return rows_quad_neon<DType::Q4>(m, acts, n, ys, r0, r1);
        if (m.type == DType::NVFP4) return rows_quad_neon<DType::NVFP4>(m, acts, n, ys, r0, r1);
        if (m.type == DType::Q4I) return rows_quad_neon<DType::Q4I>(m, acts, n, ys, r0, r1);
    } else if (m.cols % 64 == 0) {
        switch (m.type) {
        case DType::Q2: rows_q_neon<DType::Q2>(m, acts, n, ys, r0, r1); return;
        case DType::Q3: rows_q_neon<DType::Q3>(m, acts, n, ys, r0, r1); return;
        case DType::NVFP4: rows_q_neon<DType::NVFP4>(m, acts, n, ys, r0, r1); return;
        case DType::Q4: rows_q_neon<DType::Q4>(m, acts, n, ys, r0, r1); return;
        case DType::Q4I: rows_q_neon<DType::Q4I>(m, acts, n, ys, r0, r1); return;
        case DType::Q8: rows_q_neon<DType::Q8>(m, acts, n, ys, r0, r1); return;
        default: break;
        }
    }
#endif
    for (int t = 0; t < n; ++t) matvec_rows_q_ref(m, *acts[t], ys[t], r0, r1);
}

void matvec_rows_q(const QView &m, const QAct &a, float *y, int64_t r0, int64_t r1) {
    const QAct *acts[1] = {&a};
    float *ys[1] = {y};
    matvec_rows_q_multi(m, acts, 1, ys, r0, r1);
}

void matvec_rows_f32(const float *w, int64_t cols, const float *x, float *y, int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
        const float *row = w + r * cols;
        float acc = 0.0f;
        for (int64_t c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] = acc;
    }
}

void matvec_rows_bf16(const uint16_t *w, int64_t cols, const float *x, float *y, int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *row = w + r * cols;
        float acc = 0.0f;
        for (int64_t c = 0; c < cols; ++c) acc += bf16_to_f32(row[c]) * x[c];
        y[r] = acc;
    }
}

namespace {
// NT tokens against one BF16 row, eight columns at a time (the row is widened once for all of them); the eight
// partial sums are added in a fixed order, so a token gets the same bits whichever tokens share its pass
template <int NT>
void bf16_row_dots(const uint16_t *row, int64_t cols, const float *const *xs, float *out) {
#if defined(GEODE_NEON_DOTPROD)
    // the same sums, written out: lanes 0-3 and 4-7 of the eight partial sums, fused multiply-adds, the same final
    // order (the compiler kept the portable loop in two-lane pieces)
    float32x4_t lo[NT], hi[NT];
    for (int t = 0; t < NT; ++t) lo[t] = hi[t] = vdupq_n_f32(0.0f);
    for (int64_t c = 0; c < cols; c += 8) {
        const uint16x8_t w = vld1q_u16(row + c);
        const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w), 16));
        const float32x4_t w1 = vreinterpretq_f32_u32(vshll_high_n_u16(w, 16));
        for (int t = 0; t < NT; ++t) {
            lo[t] = vfmaq_f32(lo[t], w0, vld1q_f32(xs[t] + c));
            hi[t] = vfmaq_f32(hi[t], w1, vld1q_f32(xs[t] + c + 4));
        }
    }
    for (int t = 0; t < NT; ++t) {
        const float32x4_t s4 = vaddq_f32(lo[t], hi[t]);   // a[j] + a[j + 4]
        out[t] = (vgetq_lane_f32(s4, 0) + vgetq_lane_f32(s4, 1)) + (vgetq_lane_f32(s4, 2) + vgetq_lane_f32(s4, 3));
    }
#else
    float acc[NT][8] = {};
    for (int64_t c = 0; c < cols; c += 8) {
        float wf[8];
        for (int j = 0; j < 8; ++j) wf[j] = bf16_to_f32(row[c + j]);
        for (int t = 0; t < NT; ++t)
            for (int j = 0; j < 8; ++j) acc[t][j] += wf[j] * xs[t][c + j];
    }
    for (int t = 0; t < NT; ++t) {
        const float *a = acc[t];
        out[t] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
    }
#endif
}
}  // namespace

void matvec_rows_bf16_multi(const uint16_t *w, int64_t cols, int n, const float *const *xs, float *const *ys,
                            int64_t r0, int64_t r1) {
    // tiled as matvec_rows_q_multi: rows that stay in L2 against eight tokens at a time (their FP32 rows in L1)
    constexpr int kTileTokens = 8;
    if (n > kTileTokens) {
        const int64_t block = std::max<int64_t>(8, static_cast<int64_t>((96u << 10) / (static_cast<size_t>(cols) * 2)));
        for (int64_t a = r0; a < r1; a += block)
            for (int t = 0; t < n; t += kTileTokens)
                matvec_rows_bf16_multi(w, cols, std::min(kTileTokens, n - t), xs + t, ys + t, a, std::min(r1, a + block));
        return;
    }
    if (cols % 8) {
        for (int t = 0; t < n; ++t) matvec_rows_bf16(w, cols, xs[t], ys[t], r0, r1);
        return;
    }
    for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *row = w + r * cols;
        for (int t0 = 0; t0 < n; t0 += 8) {
            float out[8];
            const int nt = std::min(8, n - t0);
            switch (nt) {
            case 1: bf16_row_dots<1>(row, cols, xs + t0, out); break;
            case 2: bf16_row_dots<2>(row, cols, xs + t0, out); break;
            case 3: bf16_row_dots<3>(row, cols, xs + t0, out); break;
            case 4: bf16_row_dots<4>(row, cols, xs + t0, out); break;
            case 5: bf16_row_dots<5>(row, cols, xs + t0, out); break;
            case 6: bf16_row_dots<6>(row, cols, xs + t0, out); break;
            case 7: bf16_row_dots<7>(row, cols, xs + t0, out); break;
            default: bf16_row_dots<8>(row, cols, xs + t0, out); break;
            }
            for (int j = 0; j < nt; ++j) ys[t0 + j][r] = out[j];
        }
    }
}

void rmsnorm(const float *x, const float *w, float *out, int n, float eps, bool zero_centered) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
    if (!w) {
        for (int i = 0; i < n; ++i) out[i] = x[i] * inv;
        return;
    }
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * (zero_centered ? 1.0f + w[i] : w[i]);
}

}  // namespace geode
