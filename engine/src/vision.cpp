// vision.cpp - Qwen3.6's vision encoder (vision.hpp; tools/vision_ref.py is the reference it is checked against).
#include "vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace geode {

namespace {

// ---- The products on SMMLA (FEAT_I8MM: a 2 x 8 by 8 x 2 int8 matrix multiply an instruction) where the CPU has it
// (the 8 Elite; not the 720G's A76, which keeps the int8 row kernels). A q8 matrix repacked for it: its rows in pairs,
// each pair's 8-value chunks side by side (16 bytes: row 2p's 8 codes, then row 2p + 1's) and its block scales
// (E4M3 x global) as two floats. The activations likewise in pairs of tokens, int8 a 16-value block (amax / 127, as
// quantize_activation) with two floats a block. A block of 2 x 2 is then two SMMLAs, a convert and one multiply-add
// with its four scale products: 3.5x the int8 row kernels' arithmetic a cycle here.
struct Packed {
    int64_t rows = 0, cols = 0;
    std::vector<int8_t> codes;   // [rows / 2][cols / 8][16]
    std::vector<float> scales;   // [rows / 2][cols / 16][2]
};

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

// torch.nn.LayerNorm in f32 (the sums in double)
void layer_norm(const float *x, const float *w, const float *b, float *y, int n, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    const double mean = s / n;
    double v = 0.0;
    for (int i = 0; i < n; ++i) v += (x[i] - mean) * (x[i] - mean);
    const auto m = static_cast<float>(mean), inv = static_cast<float>(1.0 / std::sqrt(v / n + eps));
    for (int i = 0; i < n; ++i) y[i] = (x[i] - m) * inv * w[i] + b[i];
}

// GELU's tanh form: 0.5 x (1 + tanh(u)) = x sigmoid(2u)
inline float gelu_tanh(float x) {
    return x / (1.0f + fast_exp(-1.5957691216057308f * (x + 0.044715f * x * x * x)));
}
inline float gelu_erf(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f)); }

// torch.linspace(0, end, n)'s value i (it fills the second half from the end)
float linspace(int i, int n, float end) {
    if (n <= 1) return 0.0f;
    const float step = end / static_cast<float>(n - 1);
    return i < n / 2 ? static_cast<float>(i) * step : end - static_cast<float>(n - 1 - i) * step;
}

constexpr int kQB = 4;    // queries a task
constexpr int kKB = 16;   // keys a step of the scores (kQB x kKB sums in registers)

// one head's softmax(q K^T) V for nq (<= kQB) queries: q [kQB][hd] (scaled), kt [hd][npad] (the keys' transpose),
// v [n][hd]; sc: kQB rows of npad scratch; out row r at out + r * ostride
template <int DC>
void attend_block(const float *q, int hd, const float *kt, const float *v, int n, int npad, int nq, float *sc,
                  float *out, size_t ostride) {
    for (int j0 = 0; j0 < npad; j0 += kKB) {
        float acc[kQB][kKB] = {};
        for (int d = 0; d < hd; ++d) {
            const float *kr = kt + static_cast<size_t>(d) * npad + j0;
            for (int r = 0; r < kQB; ++r) {
                const float qd = q[r * hd + d];
                for (int jj = 0; jj < kKB; ++jj) acc[r][jj] += qd * kr[jj];
            }
        }
        for (int r = 0; r < kQB; ++r)
            for (int jj = 0; jj < kKB; ++jj) sc[static_cast<size_t>(r) * npad + j0 + jj] = acc[r][jj];
    }
    for (int r = 0; r < nq; ++r) {
        float *s = sc + static_cast<size_t>(r) * npad;
        // eight running maxima and sums: the loops vectorize (one running value would be a serial chain)
        float m8[8], s8[8] = {};
        for (int i = 0; i < 8; ++i) m8[i] = s[0];
        int j = 0;
        for (; j + 8 <= n; j += 8)
            for (int i = 0; i < 8; ++i) m8[i] = std::max(m8[i], s[j + i]);
        float mx = m8[0];
        for (int i = 1; i < 8; ++i) mx = std::max(mx, m8[i]);
        for (; j < n; ++j) mx = std::max(mx, s[j]);
        for (j = 0; j < n; ++j) s[j] = fast_exp(s[j] - mx);
        for (j = 0; j + 8 <= n; j += 8)
            for (int i = 0; i < 8; ++i) s8[i] += s[j + i];
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) sum += s8[i];
        for (; j < n; ++j) sum += s[j];
        const float inv = 1.0f / sum;
        for (j = 0; j < n; ++j) s[j] *= inv;
    }
    for (int r = nq; r < kQB; ++r) std::fill(sc + static_cast<size_t>(r) * npad, sc + static_cast<size_t>(r) * npad + n, 0.0f);
    for (int dc = 0; dc < hd; dc += DC) {
        float o[kQB][DC] = {};
        for (int j = 0; j < n; ++j) {
            const float *vr = v + static_cast<size_t>(j) * hd + dc;
            for (int r = 0; r < kQB; ++r) {
                const float p = sc[static_cast<size_t>(r) * npad + j];
                for (int i = 0; i < DC; ++i) o[r][i] += p * vr[i];
            }
        }
        for (int r = 0; r < nq; ++r)
            for (int i = 0; i < DC; ++i) out[static_cast<size_t>(r) * ostride + dc + i] = o[r][i];
    }
}

}  // namespace

const Tensor *Vision::tensor(const std::string &name, std::string &err) const {
    const Tensor *t = pack_->find(name);
    if (!t) err = "geode.json: no " + name;
    return t;
}

bool Vision::open(const Pack &pack, bool direct, std::string &err) {
    ok_ = false;
    const auto &v = pack.manifest()["vision"];
    if (!v.is_object()) return false;
    pack_ = &pack;
    auto &s = s_;
    s.depth = static_cast<int>(v["depth"].i64());
    s.hidden = static_cast<int>(v["hidden"].i64());
    s.heads = static_cast<int>(v["heads"].i64());
    s.ff = static_cast<int>(v["ff"].i64());
    s.patch = static_cast<int>(v["patch"].i64());
    s.merge = static_cast<int>(v["merge"].i64());
    s.out_hidden = static_cast<int>(v["out_hidden"].i64());
    s.pos_grid = static_cast<int>(v["pos_grid"].i64());
    s.theta = static_cast<float>(v["rope_theta"].f64(10000.0));
    s.eps = static_cast<float>(v["eps"].f64(1e-6));
    for (int c = 0; c < 3; ++c) {
        s.mean[c] = static_cast<float>(v["image_mean"].at(static_cast<size_t>(c)).f64(0.5));
        s.std[c] = static_cast<float>(v["image_std"].at(static_cast<size_t>(c)).f64(0.5));
    }
    s.min_pixels = v["min_pixels"].i64(65536);
    s.max_pixels = v["max_pixels"].i64(16777216);
    s.start_token = static_cast<int>(v["vision_start"].i64(-1));
    s.end_token = static_cast<int>(v["vision_end"].i64(-1));
    s.pad_token = static_cast<int>(v["image_pad"].i64(-1));
    const int hd = s.heads > 0 ? s.hidden / s.heads : 0;
    if (s.depth <= 0 || s.heads <= 0 || s.hidden % s.heads || hd % 8 || hd > 256 || s.hidden % 64 || s.ff % 64 ||
        s.patch <= 0 || s.merge <= 0 || s.pos_grid <= 0 || s.out_hidden <= 0 || s.pad_token < 0 ||
        static_cast<int>(v["blocks"].size()) != s.depth) {
        err = "geode.json: a vision encoder this engine does not take";
        return false;
    }
    for (int i = 0; i < s.depth; ++i) {
        const auto &b = v["blocks"].at(static_cast<size_t>(i));
        blocks_.push_back({static_cast<uint64_t>(b["offset"].i64()), static_cast<uint64_t>(b["bytes"].i64())});
    }
    merger_ = {static_cast<uint64_t>(v["merger"]["offset"].i64()), static_cast<uint64_t>(v["merger"]["bytes"].i64())};

    // every tensor there, its shape, inside the range it is read with
    const int D = s.hidden, K = 3 * s.patch * s.patch, M2 = s.merge * s.merge;
    auto check = [&](const std::string &name, int64_t rows, int64_t cols, bool matrix, const Range &r) {
        const Tensor *t = tensor(name, err);
        if (!t) return false;
        if (t->rows != rows || t->cols != cols || is_quantized(t->type) != matrix || (!matrix && t->type != DType::F32) ||
            pack.path(t->file) != pack.path(pack.find("visual.patch_embed.weight")->file) || t->offset < r.offset ||
            t->offset + tensor_bytes(*t) > r.offset + r.bytes) {
            err = "geode.json: " + name + " is not what the vision encoder reads";
            return false;
        }
        return true;
    };
    if (!pack.find("visual.patch_embed.weight")) {   // vision.bin is optional (pack.cpp): not installed
        err = "vision.bin is not installed";
        return false;
    }
    {
        uint64_t lo = UINT64_MAX, hi = 0;
        for (const char *n : {"visual.patch_embed.weight", "visual.patch_embed.bias", "visual.pos_embed.weight"}) {
            const Tensor *t = tensor(n, err);
            if (!t) return false;
            lo = std::min(lo, t->offset);
            hi = std::max(hi, t->offset + tensor_bytes(*t));
        }
        head_ = {lo, hi - lo};
    }
    if (!check("visual.patch_embed.weight", D, K, true, head_) || !check("visual.patch_embed.bias", 1, D, false, head_) ||
        !check("visual.pos_embed.weight", static_cast<int64_t>(s.pos_grid) * s.pos_grid, D, false, head_))
        return false;
    for (int i = 0; i < s.depth; ++i) {
        const std::string b = "visual.blocks." + std::to_string(i) + ".";
        const Range &r = blocks_[static_cast<size_t>(i)];
        if (!check(b + "norm1.weight", 1, D, false, r) || !check(b + "norm1.bias", 1, D, false, r) ||
            !check(b + "attn.qkv.weight", 3 * D, D, true, r) || !check(b + "attn.qkv.bias", 1, 3 * D, false, r) ||
            !check(b + "attn.proj.weight", D, D, true, r) || !check(b + "attn.proj.bias", 1, D, false, r) ||
            !check(b + "norm2.weight", 1, D, false, r) || !check(b + "norm2.bias", 1, D, false, r) ||
            !check(b + "mlp.linear_fc1.weight", s.ff, D, true, r) || !check(b + "mlp.linear_fc1.bias", 1, s.ff, false, r) ||
            !check(b + "mlp.linear_fc2.weight", D, s.ff, true, r) || !check(b + "mlp.linear_fc2.bias", 1, D, false, r))
            return false;
    }
    if (!check("visual.merger.norm.weight", 1, D, false, merger_) || !check("visual.merger.norm.bias", 1, D, false, merger_) ||
        !check("visual.merger.linear_fc1.weight", M2 * D, M2 * D, true, merger_) ||
        !check("visual.merger.linear_fc1.bias", 1, M2 * D, false, merger_) ||
        !check("visual.merger.linear_fc2.weight", s.out_hidden, M2 * D, true, merger_) ||
        !check("visual.merger.linear_fc2.bias", 1, s.out_hidden, false, merger_))
        return false;
    if (!file_.open(pack.path(pack.find("visual.patch_embed.weight")->file), err, direct)) return false;
    const auto &files = pack.manifest()["files"];
    for (size_t i = 0; i < files.size(); ++i)
        if (files.at(i)["name"].str() == "vision.bin") fingerprint_ = files.at(i)["sha256"].str();
    ok_ = true;
    return true;
}

void Vision::fit(int w, int h, int64_t max_pixels, int &out_w, int &out_h) const {
    const int f = s_.patch * s_.merge;
    const double maxp = static_cast<double>(max_pixels > 0 ? std::min(max_pixels, s_.max_pixels) : s_.max_pixels);
    const double minp = std::min(static_cast<double>(s_.min_pixels), maxp);
    w = std::max(w, 1);
    h = std::max(h, 1);
    // Python's round(): half to even (the default rounding mode)
    double hb = std::max<double>(f, std::nearbyint(h / static_cast<double>(f)) * f);
    double wb = std::max<double>(f, std::nearbyint(w / static_cast<double>(f)) * f);
    if (hb * wb > maxp) {
        const double beta = std::sqrt(static_cast<double>(h) * w / maxp);
        hb = std::max<double>(f, std::floor(h / beta / f) * f);
        wb = std::max<double>(f, std::floor(w / beta / f) * f);
    } else if (hb * wb < minp) {
        const double beta = std::sqrt(minp / (static_cast<double>(h) * w));
        hb = std::ceil(h * beta / f) * f;
        wb = std::ceil(w * beta / f) * f;
    }
    out_w = static_cast<int>(wb);
    out_h = static_cast<int>(hb);
}

int Vision::tokens(int w, int h) const {
    const int f = s_.patch * s_.merge;
    return (w / f) * (h / f);
}

bool Vision::read(const Range &r, AlignedBuffer &buf, size_t &cap, uint64_t &base, std::string &err) const {
    // whole 4 KiB pages (O_DIRECT); the file's end may cut the last one short (then a buffered read)
    const uint64_t a0 = r.offset & ~uint64_t{4095};
    const uint64_t a1 = std::min<uint64_t>((r.offset + r.bytes + 4095) & ~uint64_t{4095}, file_.size());
    const auto n = static_cast<size_t>(a1 - a0);
    if (cap < n) {
        buf = aligned_bytes(n);
        cap = buf ? n : 0;
    }
    if (!buf) {
        err = "not enough memory for the vision encoder";
        return false;
    }
    if (!file_.read_at(a0, buf.get(), n)) {
        err = "cannot read vision.bin";
        return false;
    }
    base = a0;
    return true;
}

bool Vision::encode(const uint8_t *rgb, int w, int h, ThreadPool &pool, std::vector<float> &out, std::string &err,
                    Gpu8 *gpu) {
    const auto &s = s_;
    const int P = s.patch, M = s.merge, D = s.hidden, NH = s.heads, HD = D / NH, F = s.ff, G = s.pos_grid;
    if (!ok_) {
        err = "the pack has no vision encoder";
        return false;
    }
    if (!rgb || w <= 0 || h <= 0 || w % (P * M) || h % (P * M)) {
        err = "an image's sides must be multiples of " + std::to_string(P * M);
        return false;
    }
    const int gh = h / P, gw = w / P, N = gh * gw, K = 3 * P * P;
    const int npad = (N + kKB - 1) / kKB * kKB, nqb = (N + kQB - 1) / kQB;
    prof_ = {};
    prof_.patches = N;
    const double t0 = now_seconds();
    double tl = t0;
    const auto lap = [&](double &into) {
        const double t = now_seconds();
        into += t - tl;
        tl = t;
    };

    // a tensor of the buffer holding [base, ...)
    const auto at = [this](const std::string &name, const uint8_t *buf, uint64_t base) {
        const Tensor &t = *pack_->find(name);
        QView v = t.view();
        v.data = buf + (t.offset - base);
        return v;
    };
    // the matrices of block i (depth: the merger's), in the order they are used
    const auto mats = [&](int i) {
        const std::string b = i < s.depth ? "visual.blocks." + std::to_string(i) + "." : "visual.merger.";
        return i < s.depth ? std::vector<std::string>{b + "attn.qkv.weight", b + "attn.proj.weight",
                                                      b + "mlp.linear_fc1.weight", b + "mlp.linear_fc2.weight"}
                           : std::vector<std::string>{b + "linear_fc1.weight", b + "linear_fc2.weight"};
    };
    // SMMLA where the CPU has it and every matrix is q8 (rows even)
    bool mmla = have_i8mm();
    for (int i = 0; i <= s.depth && mmla; ++i)
        for (const auto &n : mats(i)) mmla = mmla && pack_->find(n)->type == DType::Q8 && pack_->find(n)->rows % 8 == 0;
    mmla = mmla && pack_->find("visual.patch_embed.weight")->type == DType::Q8 && D % 8 == 0;
    // The products on the GPU's int8 dot product when there is one (GEODE_VISION_GPU=0: not): each block's q8
    // matrices converted into its layout as they are read (two slots in turn), the activations in its shared memory,
    // every product in place there; the CPU does the rest between them.
    const char *vg = std::getenv("GEODE_VISION_GPU");
    bool on_gpu = gpu && !(vg && std::atoi(vg) == 0);
    for (int i = 0; i <= s.depth && on_gpu; ++i)
        for (const auto &n : mats(i)) on_gpu = on_gpu && pack_->find(n)->type == DType::Q8 && pack_->find(n)->rows % 4 == 0;
    on_gpu = on_gpu && pack_->find("visual.patch_embed.weight")->type == DType::Q8 && D % 4 == 0;
    struct Release {   // every piece of the GPU's memory this image borrowed, back at the end
        Gpu8 *g;
        ~Release() {
            if (g) g->vision_release();
        }
    } release{on_gpu ? gpu : nullptr};
    const auto gsize = [&](const std::string &name) {
        const Tensor *t = pack_->find(name);
        return (Gpu8::q8gpu_bytes(t->rows, t->cols) + 63) / 64 * 64;
    };
    uint8_t *gslot[2] = {nullptr, nullptr};
    if (on_gpu) {
        size_t most = 0;
        for (int i : {0, s.depth}) {   // the blocks all alike, and the merger
            size_t b = 0;
            for (const auto &n : mats(i)) b += gsize(n);
            most = std::max(most, b);
        }
        std::string ge;
        for (auto &g : gslot)
            if (on_gpu && !(g = gpu->vision_alloc(most, ge))) on_gpu = false;
    }
    // both: each product's rows shared, the GPU the first part (GEODE_VISION_GPU_SHARE, else half), the CPU's SMMLA
    // the rest at the same time (alone each does ~0.4 T MAC/s here; the CPU idling meanwhile also lets its clocks drop)
    const char *vs = std::getenv("GEODE_VISION_GPU_SHARE");
    const double share = std::clamp(vs ? std::atof(vs) : 0.5, 0.0, 1.0);
    if (on_gpu && share >= 1.0) mmla = false;
    // block i's matrix j in its slot (nullptr: the CPU's)
    const auto gmat = [&](int i, int j) -> const uint8_t * {
        if (!on_gpu) return nullptr;
        const auto names = mats(i);
        const uint8_t *o = gslot[i & 1];
        for (int q = 0; q < j; ++q) o += gsize(names[static_cast<size_t>(q)]);
        return o;
    };

    // the blocks come in one ahead of the arithmetic, two buffers in turn (repacked for SMMLA there too)
    AlignedBuffer bb[2];
    size_t bcap[2] = {0, 0};
    uint64_t bbase[2] = {0, 0};
    std::string berr[2];
    std::vector<Packed> bpk[2];
    const auto start = [&](int i) {   // block i (depth: the merger) into buffer i & 1
        const Range *r = i < s.depth ? &blocks_[static_cast<size_t>(i)] : &merger_;
        const int k = i & 1;
        return std::async(std::launch::async, [this, r, &bb, &bcap, &bbase, &berr, &bpk, &at, &gsize, k, mmla,
                                               g = on_gpu ? gslot[k] : nullptr, names = mats(i)] {
            if (!read(*r, bb[k], bcap[k], bbase[k], berr[k])) return false;
            // the GPU's copy on a thread of its own while this one packs for SMMLA (both only copy bytes)
            std::future<void> gc;
            if (g)
                gc = std::async(std::launch::async, [&] {
                    uint8_t *o = g;
                    for (const auto &n : names) {
                        Gpu8::to_q8gpu(at(n, bb[k].get(), bbase[k]), o);
                        o += gsize(n);
                    }
                });
            if (mmla) {
                bpk[k].resize(names.size());
                for (size_t j = 0; j < names.size(); ++j) pack_q8(at(names[j], bb[k].get(), bbase[k]), bpk[k][j]);
            }
            if (gc.valid()) gc.get();
            return true;
        });
    };
    std::future<bool> next = start(0);
    AlignedBuffer hbuf;
    size_t hcap = 0;
    uint64_t hbase = 0;
    if (!read(head_, hbuf, hcap, hbase, err)) {
        next.wait();
        return false;
    }
    Packed pe;
    if (mmla) pack_q8(at("visual.patch_embed.weight", hbuf.get(), hbase), pe);
    uint8_t *gpe = nullptr;
    if (on_gpu) {
        std::string ge;
        if ((gpe = gpu->vision_alloc(gsize("visual.patch_embed.weight"), ge)))
            Gpu8::to_q8gpu(at("visual.patch_embed.weight", hbuf.get(), hbase), gpe);
        else on_gpu = false;
    }
    lap(prof_.wait);
    const auto vec = [&](const std::string &name, const uint8_t *buf, uint64_t base) {
        return reinterpret_cast<const float *>(buf + (pack_->find(name)->offset - base));
    };

    // the activations the products read and write: in the GPU's memory when it runs them, else here
    std::vector<std::vector<float>> own;
    const auto buffer = [&](size_t floats) -> float * {
        if (on_gpu) {
            std::string ge;
            if (uint8_t *b = gpu->vision_alloc(floats * sizeof(float), ge)) return reinterpret_cast<float *>(b);
            on_gpu = false;   // the CPU from here on (what it got stays good for the CPU)
        }
        own.emplace_back(floats);
        return own.back().data();
    };
    float *x = buffer(static_cast<size_t>(N) * D), *hn = buffer(static_cast<size_t>(N) * std::max(D, K)),
          *y = buffer(static_cast<size_t>(N) * D), *qkv = buffer(static_cast<size_t>(N) * 3 * D),
          *ff = buffer(static_cast<size_t>(N) * F);
    std::vector<float> kt(static_cast<size_t>(NH) * HD * npad, 0.0f), vh(static_cast<size_t>(NH) * N * HD);
    std::vector<QAct> qa(static_cast<size_t>(N));
    std::vector<const QAct *> qp(static_cast<size_t>(N));
    std::vector<float *> ys(static_cast<size_t>(N));
    std::vector<int8_t> aq;
    std::vector<float> as;
    // y rows (ystride apart) = m . x rows (xstride apart) + bias, for n tokens; g: m in the GPU's layout, pk: m
    // repacked for SMMLA
    const auto mm = [&](const QView &m, const Packed *pk, const uint8_t *g, const float *bias, const float *xs,
                        size_t xstride, float *yo, size_t ystride, int n) {
        std::string ge;
        // the GPU's rows: a multiple of 64 (the CPU's rows then start on a cache line: no line written by both)
        const int64_t rg = g && on_gpu ? (pk ? static_cast<int64_t>(static_cast<double>(m.rows) * share) / 64 * 64
                                             : m.rows)
                                       : 0;
        if (rg > 0 && rg < m.rows && gpu->vision_gemm_start(g, m.rows, m.cols, rg, m.global, n, xs, xstride, yo, ystride, ge)) {
            const int64_t C = m.cols, nb = C / kBlock, np = (n + 1) / 2;
            aq.resize(static_cast<size_t>(np * 2 * C));
            as.resize(static_cast<size_t>(np * nb * 2));
            pool.parallel_for(np, [&](int64_t a, int64_t b) {
                for (int64_t q = a; q < b; ++q)
                    quant_pair(xs + static_cast<size_t>(2 * q) * xstride,
                               2 * q + 1 < n ? xs + static_cast<size_t>(2 * q + 1) * xstride : nullptr, C,
                               aq.data() + static_cast<size_t>(q * 2 * C), as.data() + static_cast<size_t>(q * nb * 2));
            }, 4, true);
            pool.parallel_for((m.rows - rg) / 8, [&](int64_t a, int64_t b) {
                mmla_rows(*pk, rg / 8 + a, rg / 8 + b, aq.data(), as.data(), n, yo, ystride);
            }, 1, true);
            if (!gpu->vision_wait(ge)) {   // the GPU's rows on the CPU after all
                on_gpu = false;
                err = ge;
                pool.parallel_for(rg / 8, [&](int64_t a, int64_t b) {
                    mmla_rows(*pk, a, b, aq.data(), as.data(), n, yo, ystride);
                }, 1, true);
            }
        } else if (rg == m.rows && gpu->vision_gemm(g, m.rows, m.cols, m.global, n, xs, xstride, yo, ystride, ge)) {
            // done there
        } else if (rg > 0 && !pk) {
            on_gpu = false;   // the CPU from here on: this product and the rest (the row kernels: no SMMLA packs)
            err = ge;
            pool.parallel_for(n, [&](int64_t a, int64_t b) {
                for (int64_t t = a; t < b; ++t) {
                    quantize_activation(xs + static_cast<size_t>(t) * xstride, m.cols, qa[static_cast<size_t>(t)]);
                    qp[static_cast<size_t>(t)] = &qa[static_cast<size_t>(t)];
                    ys[static_cast<size_t>(t)] = yo + static_cast<size_t>(t) * ystride;
                }
            }, 8, true);
            pool.parallel_for(m.rows, [&](int64_t a, int64_t b) { matvec_rows_q_multi(m, qp.data(), n, ys.data(), a, b); },
                              16, true);
        } else if (pk) {
            const int64_t C = m.cols, nb = C / kBlock, np = (n + 1) / 2;
            aq.resize(static_cast<size_t>(np * 2 * C));
            as.resize(static_cast<size_t>(np * nb * 2));
            pool.parallel_for(np, [&](int64_t a, int64_t b) {
                for (int64_t q = a; q < b; ++q)
                    quant_pair(xs + static_cast<size_t>(2 * q) * xstride,
                               2 * q + 1 < n ? xs + static_cast<size_t>(2 * q + 1) * xstride : nullptr, C,
                               aq.data() + static_cast<size_t>(q * 2 * C), as.data() + static_cast<size_t>(q * nb * 2));
            }, 4, true);
            pool.parallel_for(m.rows / 8, [&](int64_t a, int64_t b) {
                mmla_rows(*pk, a, b, aq.data(), as.data(), n, yo, ystride);
            }, 1, true);
        } else {
        pool.parallel_for(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) {
                quantize_activation(xs + static_cast<size_t>(t) * xstride, m.cols, qa[static_cast<size_t>(t)]);
                qp[static_cast<size_t>(t)] = &qa[static_cast<size_t>(t)];
                ys[static_cast<size_t>(t)] = yo + static_cast<size_t>(t) * ystride;
            }
        }, 8, true);
        pool.parallel_for(m.rows, [&](int64_t a, int64_t b) { matvec_rows_q_multi(m, qp.data(), n, ys.data(), a, b); },
                          16, true);
        }
        pool.parallel_for(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) {
                float *r = yo + static_cast<size_t>(t) * ystride;
                for (int64_t i = 0; i < m.rows; ++i) r[i] += bias[i];
            }
        }, 16, true);
    };
    // patch n (in merge-block order: block row, block column, row and column in it) -> its row and column
    const auto rc = [&](int n, int &r, int &c) {
        const int blk = n / (M * M), in = n % (M * M), bw = gw / M;
        r = blk / bw * M + in / M;
        c = blk % bw * M + in % M;
    };

    // the patches: each 3 x P x P (channel, row, column), normalized; the embedding, plus the position table
    // interpolated to this grid (bilinear, its corners on the grid's)
    pool.parallel_for(N, [&](int64_t a, int64_t b) {
        for (int64_t n = a; n < b; ++n) {
            int r, c;
            rc(static_cast<int>(n), r, c);
            float *o = hn + static_cast<size_t>(n) * K;
            for (int ch = 0; ch < 3; ++ch)
                for (int py = 0; py < P; ++py) {
                    const uint8_t *src = rgb + (static_cast<size_t>(r * P + py) * w + static_cast<size_t>(c) * P) * 3 + ch;
                    for (int px = 0; px < P; ++px)
                        *o++ = (static_cast<float>(src[px * 3]) / 255.0f - s.mean[ch]) / s.std[ch];
                }
        }
    }, 8, true);
    mm(at("visual.patch_embed.weight", hbuf.get(), hbase), mmla ? &pe : nullptr, gpe, vec("visual.patch_embed.bias", hbuf.get(), hbase), hn,
       static_cast<size_t>(K), x, static_cast<size_t>(D), N);
    {
        const float *tbl = vec("visual.pos_embed.weight", hbuf.get(), hbase);
        pool.parallel_for(N, [&](int64_t a, int64_t b) {
            for (int64_t n = a; n < b; ++n) {
                int r, c;
                rc(static_cast<int>(n), r, c);
                const float hi = linspace(r, gh, static_cast<float>(G - 1)), wi = linspace(c, gw, static_cast<float>(G - 1));
                const int h0 = static_cast<int>(hi), w0 = static_cast<int>(wi);
                const int h1 = std::min(h0 + 1, G - 1), w1 = std::min(w0 + 1, G - 1);
                const float dh = hi - static_cast<float>(h0), dw = wi - static_cast<float>(w0);
                const float *t00 = tbl + static_cast<size_t>(h0 * G + w0) * D, *t01 = tbl + static_cast<size_t>(h0 * G + w1) * D;
                const float *t10 = tbl + static_cast<size_t>(h1 * G + w0) * D, *t11 = tbl + static_cast<size_t>(h1 * G + w1) * D;
                const float a00 = (1 - dh) * (1 - dw), a01 = (1 - dh) * dw, a10 = dh * (1 - dw), a11 = dh * dw;
                float *o = x + static_cast<size_t>(n) * D;
                for (int i = 0; i < D; ++i) o[i] += t00[i] * a00 + t01[i] * a01 + t10[i] * a10 + t11[i] * a11;
            }
        }, 8, true);
    }
    hbuf.reset();
    lap(prof_.embed);

    // 2D RoPE: per patch, half the rotated pairs at its row, half at its column (pairs d, d + hd / 2)
    const int rh = HD / 2, rq = rh / 2;
    std::vector<float> cosv(static_cast<size_t>(N) * rh), sinv(static_cast<size_t>(N) * rh);
    for (int n = 0; n < N; ++n) {
        int r, c;
        rc(n, r, c);
        for (int d = 0; d < rh; ++d) {
            const float inv = 1.0f / std::pow(s.theta, static_cast<float>(2 * (d % rq)) / static_cast<float>(rh));
            const float f = static_cast<float>(d < rq ? r : c) * inv;
            cosv[static_cast<size_t>(n) * rh + d] = std::cos(f);
            sinv[static_cast<size_t>(n) * rh + d] = std::sin(f);
        }
    }
    const float qscale = 1.0f / std::sqrt(static_cast<float>(HD));

    for (int i = 0; i < s.depth; ++i) {
        if (!next.get()) {
            err = berr[i & 1];
            return false;
        }
        const uint8_t *buf = bb[i & 1].get();
        const uint64_t base = bbase[i & 1];
        next = start(i + 1);   // the next block, or the merger
        lap(prof_.wait);
        const std::string b = "visual.blocks." + std::to_string(i) + ".";

        // attention
        const float *n1w = vec(b + "norm1.weight", buf, base), *n1b = vec(b + "norm1.bias", buf, base);
        pool.parallel_for(N, [&](int64_t a, int64_t e) {
            for (int64_t n = a; n < e; ++n)
                layer_norm(x + static_cast<size_t>(n) * D, n1w, n1b, hn + static_cast<size_t>(n) * D, D, s.eps);
        }, 8, true);
        lap(prof_.other);
        const Packed *pk = mmla ? bpk[i & 1].data() : nullptr;
        mm(at(b + "attn.qkv.weight", buf, base), pk, gmat(i, 0), vec(b + "attn.qkv.bias", buf, base), hn, static_cast<size_t>(D),
           qkv, static_cast<size_t>(3 * D), N);
        lap(prof_.matmul);
        // q (scaled) and k rotated; k transposed and v gathered per head
        pool.parallel_for(N, [&](int64_t a, int64_t e) {
            for (int64_t n = a; n < e; ++n) {
                float *row = qkv + static_cast<size_t>(n) * 3 * D;
                const float *cs = cosv.data() + static_cast<size_t>(n) * rh, *sn = sinv.data() + static_cast<size_t>(n) * rh;
                for (int hh = 0; hh < NH; ++hh) {
                    float *q = row + hh * HD, *k = row + D + hh * HD;
                    const float *v = row + 2 * D + hh * HD;
                    for (int d = 0; d < rh; ++d) {
                        const float q1 = q[d], q2 = q[d + rh], k1 = k[d], k2 = k[d + rh];
                        q[d] = (q1 * cs[d] - q2 * sn[d]) * qscale;
                        q[d + rh] = (q2 * cs[d] + q1 * sn[d]) * qscale;
                        k[d] = k1 * cs[d] - k2 * sn[d];
                        k[d + rh] = k2 * cs[d] + k1 * sn[d];
                    }
                    float *kh = kt.data() + static_cast<size_t>(hh) * HD * npad;
                    for (int d = 0; d < HD; ++d) kh[static_cast<size_t>(d) * npad + n] = k[d];
                    std::copy(v, v + HD, vh.data() + (static_cast<size_t>(hh) * N + n) * HD);
                }
            }
        }, 8, true);
        pool.parallel_for(static_cast<int64_t>(NH) * nqb, [&](int64_t a, int64_t e) {
            std::vector<float> sc(static_cast<size_t>(kQB) * npad);
            float q[kQB * 256];
            for (int64_t task = a; task < e; ++task) {
                const int hh = static_cast<int>(task / nqb), q0 = static_cast<int>(task % nqb) * kQB, nq = std::min(kQB, N - q0);
                for (int r = 0; r < kQB; ++r)
                    for (int d = 0; d < HD; ++d)
                        q[r * HD + d] = r < nq ? qkv[static_cast<size_t>(q0 + r) * 3 * D + hh * HD + d] : 0.0f;
                const float *kh = kt.data() + static_cast<size_t>(hh) * HD * npad, *vv = vh.data() + static_cast<size_t>(hh) * N * HD;
                float *o = hn + static_cast<size_t>(q0) * D + hh * HD;
                if (HD % 24 == 0) attend_block<24>(q, HD, kh, vv, N, npad, nq, sc.data(), o, static_cast<size_t>(D));
                else attend_block<8>(q, HD, kh, vv, N, npad, nq, sc.data(), o, static_cast<size_t>(D));
            }
        }, 1, true);
        lap(prof_.attention);
        mm(at(b + "attn.proj.weight", buf, base), pk ? pk + 1 : nullptr, gmat(i, 1), vec(b + "attn.proj.bias", buf, base), hn, static_cast<size_t>(D),
           y, static_cast<size_t>(D), N);
        lap(prof_.matmul);

        // the MLP
        const float *n2w = vec(b + "norm2.weight", buf, base), *n2b = vec(b + "norm2.bias", buf, base);
        pool.parallel_for(N, [&](int64_t a, int64_t e) {
            for (int64_t n = a; n < e; ++n) {
                float *xr = x + static_cast<size_t>(n) * D;
                const float *yr = y + static_cast<size_t>(n) * D;
                for (int d = 0; d < D; ++d) xr[d] += yr[d];
                layer_norm(xr, n2w, n2b, hn + static_cast<size_t>(n) * D, D, s.eps);
            }
        }, 8, true);
        lap(prof_.other);
        mm(at(b + "mlp.linear_fc1.weight", buf, base), pk ? pk + 2 : nullptr, gmat(i, 2), vec(b + "mlp.linear_fc1.bias", buf, base), hn,
           static_cast<size_t>(D), ff, static_cast<size_t>(F), N);
        lap(prof_.matmul);
        pool.parallel_for(N, [&](int64_t a, int64_t e) {
            for (float *p = ff + static_cast<size_t>(a) * F, *end = ff + static_cast<size_t>(e) * F; p < end; ++p)
                *p = gelu_tanh(*p);
        }, 8, true);
        lap(prof_.other);
        mm(at(b + "mlp.linear_fc2.weight", buf, base), pk ? pk + 3 : nullptr, gmat(i, 3), vec(b + "mlp.linear_fc2.bias", buf, base), ff,
           static_cast<size_t>(F), y, static_cast<size_t>(D), N);
        lap(prof_.matmul);
        pool.parallel_for(static_cast<int64_t>(N) * D, [&](int64_t a, int64_t e) {
            for (int64_t k = a; k < e; ++k) x[static_cast<size_t>(k)] += y[static_cast<size_t>(k)];
        }, 4096, true);
        lap(prof_.other);
    }

    // the merger: each patch normed, a 2x2 block's four (consecutive) as one row, fc1, GELU, fc2
    if (!next.get()) {
        err = berr[s.depth & 1];
        return false;
    }
    const uint8_t *buf = bb[s.depth & 1].get();
    const uint64_t base = bbase[s.depth & 1];
    bb[(s.depth + 1) & 1].reset();
    lap(prof_.wait);
    const int T = N / (M * M), W4 = M * M * D;
    const float *nw = vec("visual.merger.norm.weight", buf, base), *nb = vec("visual.merger.norm.bias", buf, base);
    pool.parallel_for(N, [&](int64_t a, int64_t e) {
        for (int64_t n = a; n < e; ++n)
            layer_norm(x + static_cast<size_t>(n) * D, nw, nb, hn + static_cast<size_t>(n) * D, D, s.eps);
    }, 8, true);
    float *m1 = buffer(static_cast<size_t>(T) * W4), *mo = buffer(static_cast<size_t>(T) * s.out_hidden);
    const Packed *mpk = mmla ? bpk[s.depth & 1].data() : nullptr;
    mm(at("visual.merger.linear_fc1.weight", buf, base), mpk, gmat(s.depth, 0), vec("visual.merger.linear_fc1.bias", buf, base), hn,
       static_cast<size_t>(W4), m1, static_cast<size_t>(W4), T);
    for (size_t i = 0; i < static_cast<size_t>(T) * W4; ++i) m1[i] = gelu_erf(m1[i]);
    mm(at("visual.merger.linear_fc2.weight", buf, base), mpk ? mpk + 1 : nullptr, gmat(s.depth, 1), vec("visual.merger.linear_fc2.bias", buf, base), m1,
       static_cast<size_t>(W4), mo, static_cast<size_t>(s.out_hidden), T);
    out.assign(mo, mo + static_cast<size_t>(T) * s.out_hidden);
    prof_.where = on_gpu && mmla ? "GPU + CPU" : on_gpu ? "GPU" : mmla ? "CPU SMMLA" : "CPU";
    lap(prof_.matmul);
    prof_.total = now_seconds() - t0;
    return true;
}

namespace {
// a feature file: this header, then tokens x hidden floats
struct FeatureHeader {
    char magic[8];
    int32_t w, h, tokens, hidden;
    char fingerprint[64];
};
constexpr char kFeatureMagic[8] = {'G', 'E', 'O', 'V', 'F', 'E', 'A', '1'};

void fill_header(FeatureHeader &h, int w, int hh, int tokens, int hidden, const std::string &fp) {
    std::memset(&h, 0, sizeof h);
    std::memcpy(h.magic, kFeatureMagic, 8);
    h.w = w;
    h.h = hh;
    h.tokens = tokens;
    h.hidden = hidden;
    std::memcpy(h.fingerprint, fp.data(), std::min(fp.size(), sizeof h.fingerprint));
}
}  // namespace

bool Vision::load_features(const std::string &path, int w, int h, std::vector<float> &out) const {
    if (!ok_ || path.empty() || fingerprint_.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    FeatureHeader want, got;
    fill_header(want, w, h, tokens(w, h), s_.out_hidden, fingerprint_);
    if (!f.read(reinterpret_cast<char *>(&got), sizeof got) || std::memcmp(&want, &got, sizeof got) != 0) return false;
    std::vector<float> v(static_cast<size_t>(want.tokens) * want.hidden);
    if (!f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)))) return false;
    out = std::move(v);
    return true;
}

bool Vision::has_features(const std::string &path, int w, int h) const {
    if (!ok_ || path.empty() || fingerprint_.empty()) return false;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    FeatureHeader want, got;
    fill_header(want, w, h, tokens(w, h), s_.out_hidden, fingerprint_);
    const auto whole = static_cast<std::streamoff>(sizeof got) +
                       static_cast<std::streamoff>(want.tokens) * want.hidden * static_cast<std::streamoff>(sizeof(float));
    if (!f || f.tellg() != whole) return false;
    f.seekg(0);
    return f.read(reinterpret_cast<char *>(&got), sizeof got) && std::memcmp(&want, &got, sizeof got) == 0;
}

bool Vision::save_features(const std::string &path, int w, int h, const std::vector<float> &feats) const {
    if (!ok_ || path.empty() || fingerprint_.empty() ||
        feats.size() != static_cast<size_t>(tokens(w, h)) * static_cast<size_t>(s_.out_hidden))
        return false;
    FeatureHeader hd;
    fill_header(hd, w, h, tokens(w, h), s_.out_hidden, fingerprint_);
    // written aside, then renamed: a reader sees a whole file or none
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.write(reinterpret_cast<const char *>(&hd), sizeof hd) ||
            !f.write(reinterpret_cast<const char *>(feats.data()), static_cast<std::streamsize>(feats.size() * sizeof(float))))
            return false;
    }
    std::remove(path.c_str());
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

std::string Vision::profile() const {
    char b[320];
    std::snprintf(b, sizeof b,
                  "vision: %d patches %.2f s (reads waited %.2f, embedding %.2f, products %.2f on the %s, attention %.2f, "
                  "other %.2f)",
                  prof_.patches, prof_.total, prof_.wait, prof_.embed, prof_.matmul, prof_.where, prof_.attention,
                  prof_.other);
    return b;
}

}  // namespace geode
