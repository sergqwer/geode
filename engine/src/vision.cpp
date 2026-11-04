// vision.cpp - Qwen3.6's vision encoder (vision.hpp; tools/vision_ref.py is the reference it is checked against), and
// Gemma 4's (open_gemma, encode_gemma; tools/pack_gemma4_vision.py --ref).
#include "vision.hpp"

#include "media.hpp"
#include "../../npu/attn_cfg.h"   // the encoder's attention on HMX (ATTN_ENCODER)

#include <algorithm>
#include <array>
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

// the products' SMMLA layout and kernels, tensor_bytes: media.hpp (shared with the audio encoder)
using media::have_i8mm;
using media::mmla_rows;
using media::pack_q8;
using media::Packed;
using media::quant_pair;
using media::tensor_bytes;

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

constexpr int kKB = 16;   // K^T's rows padded to a multiple of this (npad)

// queries a task (flash order): 16 tiles of 4 sharing each block of keys. Gemma 4's 2394 patches on the 720G:
// 16 a task 8.3 s of attention, 32 6.1, 64 5.6, 128 5.2 (the previous kernel, 4 queries a task each streaming
// the whole of K and V: 21.3 s); 64 keeps a small image's tasks many enough for the little cores
constexpr int kFQ = 64;
constexpr int kFK = 64;   // keys a block: K^T's and V's rows for them (2 x 64 x hd floats) stay in L1

// the largest and the sum of x[0..n) with eight running values (the loops vectorize; one would be a serial chain)
inline float max_of(const float *x, int n) {
    float m8[8];
    for (int i = 0; i < 8; ++i) m8[i] = x[0];
    int j = 0;
    for (; j + 8 <= n; j += 8)
        for (int i = 0; i < 8; ++i) m8[i] = std::max(m8[i], x[j + i]);
    float m = m8[0];
    for (int i = 1; i < 8; ++i) m = std::max(m, m8[i]);
    for (; j < n; ++j) m = std::max(m, x[j]);
    return m;
}
inline float sum_of(const float *x, int n) {
    float s8[8] = {};
    int j = 0;
    for (; j + 8 <= n; j += 8)
        for (int i = 0; i < 8; ++i) s8[i] += x[j + i];
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += s8[i];
    for (; j < n; ++j) s += x[j];
    return s;
}

// one head's softmax(q K^T) V for nq (<= kFQ) queries, kFK keys at a time with the softmax kept running (flash
// attention's order): a block of K^T and V comes from memory once for all the task's queries, then from L1. q [kFQ][hd] (scaled; rows past nq zero), kt [hd][npad] (the
// keys' transpose, npad a multiple of 16), v [n][hd]; scratch s [kFQ][kFK], o [kFQ][hd]; out row r at out + r * ostride
template <int DC>
void attend_flash(const float *q, int hd, const float *kt, const float *v, int n, int npad, int nq, float *s, float *o,
                  float *out, size_t ostride) {
    const int rows = (nq + 3) / 4 * 4;
    float m[kFQ], l[kFQ];
    std::fill(o, o + static_cast<size_t>(rows) * hd, 0.0f);
    for (int k0 = 0; k0 < n; k0 += kFK) {
        const int kn = std::min(kFK, n - k0), kc = (kn + 15) / 16 * 16;
        for (int r0 = 0; r0 < rows; r0 += 4)
            for (int j0 = 0; j0 < kc; j0 += 16) {
                float acc[4][16] = {};
                for (int d = 0; d < hd; ++d) {
                    const float *kr = kt + static_cast<size_t>(d) * npad + k0 + j0;
                    for (int r = 0; r < 4; ++r) {
                        const float qd = q[(r0 + r) * hd + d];
                        for (int jj = 0; jj < 16; ++jj) acc[r][jj] += qd * kr[jj];
                    }
                }
                for (int r = 0; r < 4; ++r)
                    for (int jj = 0; jj < 16; ++jj) s[(r0 + r) * kFK + j0 + jj] = acc[r][jj];
            }
        for (int r = 0; r < rows; ++r) {
            float *sr = s + r * kFK;
            const float mb = max_of(sr, kn), mx = k0 > 0 ? std::max(m[r], mb) : mb;
            for (int j = 0; j < kn; ++j) sr[j] = fast_exp(sr[j] - mx);
            const float sum = sum_of(sr, kn);
            if (k0 > 0) {
                const float corr = fast_exp(m[r] - mx);
                l[r] = l[r] * corr + sum;
                if (corr != 1.0f)
                    for (int d = 0; d < hd; ++d) o[r * hd + d] *= corr;
            } else {
                l[r] = sum;
            }
            m[r] = mx;
        }
        for (int r0 = 0; r0 < rows; r0 += 4)
            for (int dc = 0; dc < hd; dc += DC) {
                float acc[4][DC];
                for (int r = 0; r < 4; ++r)
                    for (int i = 0; i < DC; ++i) acc[r][i] = o[(r0 + r) * hd + dc + i];
                for (int j = 0; j < kn; ++j) {
                    const float *vr = v + static_cast<size_t>(k0 + j) * hd + dc;
                    for (int r = 0; r < 4; ++r) {
                        const float p = s[(r0 + r) * kFK + j];
                        for (int i = 0; i < DC; ++i) acc[r][i] += p * vr[i];
                    }
                }
                for (int r = 0; r < 4; ++r)
                    for (int i = 0; i < DC; ++i) o[(r0 + r) * hd + dc + i] = acc[r][i];
            }
    }
    for (int r = 0; r < nq; ++r) {
        const float inv = 1.0f / l[r];
        for (int d = 0; d < hd; ++d) out[static_cast<size_t>(r) * ostride + d] = o[r * hd + d] * inv;
    }
}

// every head's attention over N tokens on the pool: query t of head h at q + t * qstride + h * hd (scaled), kt
// [heads][hd][npad], v [heads][N][hd]; its output at out + t * ostride + h * hd
void attend_all(ThreadPool &pool, int heads, int hd, int N, int npad, const float *q, size_t qstride, const float *kt,
                const float *v, float *out, size_t ostride) {
    const int nqb = (N + kFQ - 1) / kFQ;
    pool.parallel_for(static_cast<int64_t>(heads) * nqb, [&](int64_t a, int64_t e) {
        std::vector<float> ql(static_cast<size_t>(kFQ) * hd), o(static_cast<size_t>(kFQ) * hd), sc(static_cast<size_t>(kFQ) * kFK);
        for (int64_t task = a; task < e; ++task) {
            const int hh = static_cast<int>(task / nqb), q0 = static_cast<int>(task % nqb) * kFQ, nq = std::min(kFQ, N - q0);
            for (int r = 0; r < kFQ; ++r)
                for (int d = 0; d < hd; ++d)
                    ql[static_cast<size_t>(r) * hd + d] = r < nq ? q[static_cast<size_t>(q0 + r) * qstride + static_cast<size_t>(hh) * hd + d] : 0.0f;
            const float *kth = kt + static_cast<size_t>(hh) * hd * npad, *vh = v + static_cast<size_t>(hh) * N * hd;
            float *oh = out + static_cast<size_t>(q0) * ostride + static_cast<size_t>(hh) * hd;
            if (hd % 16 == 0) attend_flash<16>(ql.data(), hd, kth, vh, N, npad, nq, sc.data(), o.data(), oh, ostride);
            else if (hd % 24 == 0) attend_flash<24>(ql.data(), hd, kth, vh, N, npad, nq, sc.data(), o.data(), oh, ostride);
            else attend_flash<8>(ql.data(), hd, kth, vh, N, npad, nq, sc.data(), o.data(), oh, ostride);
        }
    }, 1, true);
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
    if (v["arch"].str() == "gemma4") return open_gemma(pack, direct, err);   // Gemma 4's encoder (below)
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
    if (s_.gemma) {
        // transformers' get_aspect_ratio_preserving_size: the largest size with at most budget x merge^2 patches, both
        // sides multiples of f, the aspect kept (the budget: max_pixels / f^2 tokens, else the pack's default)
        const int64_t budget = max_pixels > 0 ? std::max<int64_t>(1, max_pixels / (static_cast<int64_t>(f) * f)) : s_.soft_tokens;
        w = std::max(w, 1);
        h = std::max(h, 1);
        const double factor = std::sqrt(static_cast<double>(budget) * s_.merge * s_.merge * s_.patch * s_.patch /
                                        (static_cast<double>(h) * w));
        int th = static_cast<int>(std::floor(factor * h / f)) * f, tw = static_cast<int>(std::floor(factor * w / f)) * f;
        const int longest = static_cast<int>(budget) * f;
        if (th == 0 && tw == 0) th = tw = f;
        else if (th == 0) {
            th = f;
            tw = std::min(static_cast<int>(std::floor(static_cast<double>(w) / h)) * f, longest);
        } else if (tw == 0) {
            tw = f;
            th = std::min(static_cast<int>(std::floor(static_cast<double>(h) / w)) * f, longest);
        }
        out_w = tw;
        out_h = th;
        return;
    }
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
                    Gpu8 *gpu, const MediaNpu *npu) {
    const auto &s = s_;
    const int P = s.patch, M = s.merge, D = s.hidden, NH = s.heads, HD = D / NH, F = s.ff, G = s.pos_grid;
    if (!ok_) {
        err = "the pack has no vision encoder";
        return false;
    }
    if (s.gemma) return encode_gemma(rgb, w, h, pool, out, err, npu);
    if (!rgb || w <= 0 || h <= 0 || w % (P * M) || h % (P * M)) {
        err = "an image's sides must be multiples of " + std::to_string(P * M);
        return false;
    }
    const int gh = h / P, gw = w / P, N = gh * gw, K = 3 * P * P;
    const int npad = (N + kKB - 1) / kKB * kKB;
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
        attend_all(pool, NH, HD, N, npad, qkv, static_cast<size_t>(3) * D, kt.data(), vh.data(), hn, static_cast<size_t>(D));
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

// ---- Gemma 4's encoder (tools/gemma4_vision.py writes it out; tools/pack_gemma4_vision.py packs it): patches 16 x 16
// row-major with their (column, row), the patch embedding plus two position tables, 16 bidirectional layers (RMS norms,
// q / k normed per head and v without a weight, 2D RoPE, clipped projections, a gated GELU MLP), 3 x 3 average pooling
// times sqrt(hidden), a weightless RMS norm and the projection into the LLM's width.

bool Vision::open_gemma(const Pack &pack, bool direct, std::string &err) {
    const auto &v = pack.manifest()["vision"];
    auto &s = s_;
    s = Shape{};
    s.gemma = true;
    s.depth = static_cast<int>(v["depth"].i64());
    s.hidden = static_cast<int>(v["hidden"].i64());
    s.heads = static_cast<int>(v["heads"].i64());
    s.head_dim = static_cast<int>(v["head_dim"].i64());
    s.ff = static_cast<int>(v["ff"].i64());
    s.patch = static_cast<int>(v["patch"].i64());
    s.merge = static_cast<int>(v["pool"].i64());
    s.pos_size = static_cast<int>(v["pos_size"].i64());
    s.out_hidden = static_cast<int>(v["out_hidden"].i64());
    s.theta = static_cast<float>(v["rope_theta"].f64(100.0));
    s.eps = static_cast<float>(v["eps"].f64(1e-6));
    s.soft_tokens = static_cast<int>(v["default_soft_tokens"].i64(280));
    s.start_token = static_cast<int>(v["image_start"].i64(-1));
    s.end_token = static_cast<int>(v["image_end"].i64(-1));
    s.pad_token = static_cast<int>(v["image_pad"].i64(-1));
    s.video_pad = static_cast<int>(v["video_pad"].i64(-1));
    s.video_soft_tokens = static_cast<int>(v["video_soft_tokens"].i64(70));
    s.pos_grid = 1;
    const int D = s.hidden, HD = s.head_dim, Q = s.heads * HD, K = 3 * s.patch * s.patch;
    if (s.depth <= 0 || s.heads <= 0 || HD % 8 || HD > 256 || (HD / 2) % 2 || D % 64 || Q % 64 || s.ff % 64 ||
        K % 64 || s.patch <= 0 || s.merge <= 0 || s.pos_size <= 0 || s.out_hidden <= 0 || s.pad_token < 0 ||
        s.soft_tokens <= 0 || static_cast<int>(v["layers"].size()) != s.depth) {
        err = "geode.json: a vision encoder this engine does not take";
        return false;
    }
    if (!pack.find("vision.patch.weight")) {   // vision.bin is optional (pack.cpp): not installed
        err = "vision.bin is not installed";
        return false;
    }
    const auto range = [](const json::Value &r) {
        return Range{static_cast<uint64_t>(r["offset"].i64()), static_cast<uint64_t>(r["bytes"].i64())};
    };
    head_ = range(v["head"]);
    merger_ = range(v["embed"]);
    blocks_.clear();
    for (int i = 0; i < s.depth; ++i) blocks_.push_back(range(v["layers"].at(static_cast<size_t>(i))));
    const std::string file = pack.path(pack.find("vision.patch.weight")->file);
    auto check = [&](const std::string &name, int64_t rows, int64_t cols, bool matrix, const Range &r) {
        const Tensor *t = tensor(name, err);
        if (!t) return false;
        if (t->rows != rows || t->cols != cols || is_quantized(t->type) != matrix || (!matrix && t->type != DType::F32) ||
            (matrix && t->cols % 64) || pack.path(t->file) != file || t->offset < r.offset ||
            t->offset + tensor_bytes(*t) > r.offset + r.bytes) {
            err = "geode.json: " + name + " is not what the vision encoder reads";
            return false;
        }
        return true;
    };
    if (!check("vision.patch.weight", D, K, true, head_) || !check("vision.pos_x.weight", s.pos_size, D, true, head_) ||
        !check("vision.pos_y.weight", s.pos_size, D, true, head_) ||
        !check("vision.embed.weight", s.out_hidden, D, true, merger_))
        return false;
    for (int i = 0; i < s.depth; ++i) {
        const std::string b = "vision.layers." + std::to_string(i) + ".";
        const Range &r = blocks_[static_cast<size_t>(i)];
        for (const char *n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                              "post_feedforward_layernorm"})
            if (!check(b + n + ".weight", 1, D, false, r)) return false;
        if (!check(b + "self_attn.q_norm.weight", 1, HD, false, r) || !check(b + "self_attn.k_norm.weight", 1, HD, false, r))
            return false;
        const struct {
            const char *name;
            int64_t rows, cols;
        } mats[] = {{"self_attn.q_proj", Q, D}, {"self_attn.k_proj", Q, D}, {"self_attn.v_proj", Q, D},
                    {"self_attn.o_proj", D, Q}, {"mlp.gate_proj", s.ff, D}, {"mlp.up_proj", s.ff, D},
                    {"mlp.down_proj", D, s.ff}};
        for (const auto &m : mats)
            if (!check(b + m.name + ".weight", m.rows, m.cols, true, r) || !check(b + m.name + ".clip", 1, 4, false, r))
                return false;
    }
    if (!file_.open(file, err, direct)) return false;
    const auto &files = pack.manifest()["files"];
    for (size_t i = 0; i < files.size(); ++i)
        if (files.at(i)["name"].str() == "vision.bin") fingerprint_ = files.at(i)["sha256"].str();
    ok_ = true;
    return true;
}

bool Vision::encode_gemma(const uint8_t *rgb, int w, int h, ThreadPool &pool, std::vector<float> &out, std::string &err,
                          const MediaNpu *npu) {
    const auto &s = s_;
    const int P = s.patch, M = s.merge, D = s.hidden, NH = s.heads, HD = s.head_dim, QW = NH * HD, F = s.ff;
    if (!rgb || w <= 0 || h <= 0 || w % (P * M) || h % (P * M)) {
        err = "an image's sides must be multiples of " + std::to_string(P * M);
        return false;
    }
    const int gh = h / P, gw = w / P, N = gh * gw, K = 3 * P * P;
    const int npad = (N + kKB - 1) / kKB * kKB;
    if (gw > s.pos_size || gh > s.pos_size) {
        err = "the image is too large for the position tables";
        return false;
    }
    prof_ = {};
    prof_.patches = N;
    const double t0 = now_seconds();
    double tl = t0;
    const auto lap = [&](double &into) {
        const double t = now_seconds();
        into += t - tl;
        tl = t;
    };
    const auto at = [this](const std::string &name, const uint8_t *buf, uint64_t base) {
        const Tensor &t = *pack_->find(name);
        QView v = t.view();
        v.data = buf + (t.offset - base);
        return v;
    };
    const auto vec = [&](const std::string &name, const uint8_t *buf, uint64_t base) {
        return reinterpret_cast<const float *>(buf + (pack_->find(name)->offset - base));
    };
    static const char *const kLinears[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
    // the matrices of layer i (depth: the embedder), in the order they are used
    const auto mats = [&](int i) {
        std::vector<std::string> names;
        if (i == s.depth) names.push_back("vision.embed.weight");
        else
            for (const char *n : kLinears) names.push_back("vision.layers." + std::to_string(i) + "." + n + ".weight");
        return names;
    };
    bool mmla = have_i8mm();
    for (int i = 0; i <= s.depth && mmla; ++i)
        for (const auto &n : mats(i)) mmla = mmla && pack_->find(n)->type == DType::Q8 && pack_->find(n)->rows % 8 == 0;
    mmla = mmla && pack_->find("vision.patch.weight")->type == DType::Q8;

    // The NPU's HMX (MediaNpu; GEODE_VISION_NPU=0: not): every matrix q8 with rows and columns in 32s, turned into
    // fp16 blocks in one of two shared buffers as its layer comes in; the buffers made here, on the NPU's own thread,
    // once (a larger need: new ones; the old stay until the NPU closes). The CPU's SMMLA repack is skipped then: a
    // failure on the DSP finishes the image on the plain int8 kernels.
    using F16 = media::F16;
    constexpr int kMats = 7;
    bool on_npu = npu && npu->usable() && !(std::getenv("GEODE_VISION_NPU") && std::atoi(std::getenv("GEODE_VISION_NPU")) == 0);
    {
        const auto f16_of = [&](const std::string &n) -> size_t {
            const Tensor *t = pack_->find(n);
            if (!t || t->type != DType::Q8 || t->rows % 32 || t->cols % 32) return 0;
            return (npu->f16_bytes(t->rows, t->cols) + 127) / 128 * 128;
        };
        size_t need = 0;
        for (int i = 0; i <= s.depth && on_npu; ++i) {
            size_t sum = 0;
            for (const auto &n : mats(i)) {
                const size_t b = f16_of(n);
                on_npu = on_npu && b > 0;
                sum += b;
            }
            need = std::max(need, sum);
        }
        if (on_npu) {
            const size_t b = f16_of("vision.patch.weight");
            on_npu = b > 0;
            need = std::max(need, b);
        }
        if (on_npu && (npu_g_ != npu->g || npu_cap_ < need)) {
            npu_g_ = npu->g;
            npu_cap_ = 0;
            for (int k = 0; k < 2 && on_npu; ++k) {
                npu_buf_[k] = static_cast<uint8_t *>(npu->shared(npu->g, need, 1, &npu_fd_[k]));
                on_npu = npu_buf_[k] != nullptr;
            }
            if (on_npu) npu_cap_ = need;
        }
    }
    if (on_npu) mmla = false;
    // the products (media.hpp: on HMX while it is live - a failed call: that one and the rest on the CPU)
    media::Products prod;
    prod.use_npu(on_npu ? npu : nullptr);
    std::array<F16, kMats> f16d[2];
    bool f16ok[2] = {false, false};
    // a matrix of buffer k's into its fp16 blocks at *at (moved past them); false when it is not one HMX takes
    const auto pack_f16 = [this, npu](const QView &v, int k, uint64_t &at, F16 &f) {
        f.fd = npu_fd_[k];
        f.off = at;
        at += (npu->f16_bytes(v.rows, v.cols) + 127) / 128 * 128;
        return at <= npu_cap_ && npu->pack_f16(v.data, v.rows, v.cols, v.row_bytes, v.global, npu_buf_[k] + f.off, &f.shift) == 0;
    };

    // the layers come in one ahead of the arithmetic, two buffers in turn (repacked for SMMLA there too, or into
    // the NPU's fp16 blocks)
    AlignedBuffer bb[2];
    size_t bcap[2] = {0, 0};
    uint64_t bbase[2] = {0, 0};
    std::string berr[2];
    std::vector<Packed> bpk[2];
    const auto start = [&](int i) {
        const Range *r = i < s.depth ? &blocks_[static_cast<size_t>(i)] : &merger_;
        const int k = i & 1;
        return std::async(std::launch::async, [this, r, &bb, &bcap, &bbase, &berr, &bpk, &at, &f16d, &f16ok, &pack_f16,
                                               k, mmla, on_npu, names = mats(i)] {
            if (!read(*r, bb[k], bcap[k], bbase[k], berr[k])) return false;
            if (mmla) {
                bpk[k].resize(names.size());
                for (size_t j = 0; j < names.size(); ++j) pack_q8(at(names[j], bb[k].get(), bbase[k]), bpk[k][j]);
            }
            f16ok[k] = on_npu && names.size() <= static_cast<size_t>(kMats);
            uint64_t o = 0;
            for (size_t j = 0; j < names.size() && f16ok[k]; ++j)
                f16ok[k] = pack_f16(at(names[j], bb[k].get(), bbase[k]), k, o, f16d[k][j]);
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
    if (mmla) pack_q8(at("vision.patch.weight", hbuf.get(), hbase), pe);
    lap(prof_.wait);

    std::vector<float> x(static_cast<size_t>(N) * D), hn(static_cast<size_t>(N) * std::max({D, K, QW})),
        y(static_cast<size_t>(N) * D), q(static_cast<size_t>(N) * QW), k(static_cast<size_t>(N) * QW),
        vv(static_cast<size_t>(N) * QW), g(static_cast<size_t>(N) * F), u(static_cast<size_t>(N) * F);
    std::vector<float> kt(static_cast<size_t>(NH) * HD * npad, 0.0f), vh(static_cast<size_t>(NH) * N * HD);
    // y rows (ystride apart) = m . x rows (xstride apart), n tokens; pk: m repacked for SMMLA; f: m's fp16 blocks for
    // the NPU
    const auto mm = [&](const QView &m, const Packed *pk, const float *xs, size_t xstride, float *yo, size_t ystride,
                        int n, const F16 *f = nullptr) { prod.mm(pool, m, pk, xs, xstride, yo, ystride, n, f); };
    // a clipped linear (Gemma4ClippableLinear): its input clamped to [clip 0, clip 1], the product, its output
    // clamped to [clip 2, clip 3]; xs rows of m.cols, yo rows of m.rows
    const auto clipped = [&](const QView &m, const Packed *pk, const float *clip, const float *xs, float *yo, int n,
                             const F16 *f) { prod.clipped(pool, m, pk, clip, xs, yo, n, f); };

    // the patches (row-major; a patch's pixels row by row, R G B) as 2 x - 1, the embedding, plus the position tables'
    // rows of the patch's column and row
    pool.parallel_for(N, [&](int64_t a, int64_t b) {
        for (int64_t n = a; n < b; ++n) {
            const int r = static_cast<int>(n) / gw, c = static_cast<int>(n) % gw;
            float *o = hn.data() + static_cast<size_t>(n) * K;
            for (int py = 0; py < P; ++py) {
                const uint8_t *src = rgb + (static_cast<size_t>(r * P + py) * w + static_cast<size_t>(c) * P) * 3;
                for (int i = 0; i < 3 * P; ++i) *o++ = 2.0f * (static_cast<float>(src[i]) / 255.0f) - 1.0f;
            }
        }
    }, 8, true);
    {   // (on the NPU from buffer 1: layer 0 comes into buffer 0 meanwhile, layer 1 into this one only after it)
        F16 fp;
        uint64_t o = 0;
        const QView pm = at("vision.patch.weight", hbuf.get(), hbase);
        const bool pf = on_npu && pack_f16(pm, 1, o, fp);
        mm(pm, mmla ? &pe : nullptr, hn.data(), static_cast<size_t>(K), x.data(), static_cast<size_t>(D), N,
           pf ? &fp : nullptr);
    }
    {
        std::vector<float> px(static_cast<size_t>(gw) * D), py(static_cast<size_t>(gh) * D);
        const QView tx = at("vision.pos_x.weight", hbuf.get(), hbase), ty = at("vision.pos_y.weight", hbuf.get(), hbase);
        for (int c = 0; c < gw; ++c) dequant_row(tx, c, px.data() + static_cast<size_t>(c) * D);
        for (int r = 0; r < gh; ++r) dequant_row(ty, r, py.data() + static_cast<size_t>(r) * D);
        pool.parallel_for(N, [&](int64_t a, int64_t b) {
            for (int64_t n = a; n < b; ++n) {
                const int r = static_cast<int>(n) / gw, c = static_cast<int>(n) % gw;
                float *o = x.data() + static_cast<size_t>(n) * D;
                const float *ax = px.data() + static_cast<size_t>(c) * D, *ay = py.data() + static_cast<size_t>(r) * D;
                for (int i = 0; i < D; ++i) o[i] = (o[i] + ax[i]) + ay[i];
            }
        }, 8, true);
    }
    hbuf.reset();
    lap(prof_.embed);

    // 2D RoPE: the head's first half turned by the patch's column, the second by its row; pairs (j, j + HD / 4) of
    // each half, frequency theta^(-2 j / (HD / 2))
    const int half = HD / 2, quarter = half / 2;
    std::vector<float> cosv(static_cast<size_t>(N) * half), sinv(static_cast<size_t>(N) * half);
    for (int n = 0; n < N; ++n) {
        const int p2[2] = {n % gw, n / gw};
        for (int hk = 0; hk < 2; ++hk)
            for (int j = 0; j < quarter; ++j) {
                const float inv = 1.0f / std::pow(s.theta, static_cast<float>(2 * j) / static_cast<float>(half));
                const float f = static_cast<float>(p2[hk]) * inv;
                cosv[static_cast<size_t>(n) * half + hk * quarter + j] = std::cos(f);
                sinv[static_cast<size_t>(n) * half + hk * quarter + j] = std::sin(f);
            }
    }

    // The attention on HMX too (npu/attn_cfg.h ATTN_ENCODER; GEODE_VISION_NPU_ATTN=0: not): q_proj's output straight
    // into the NPU's query buffer, the keys and values as fp16 blocks in two shared buffers (made on this thread, kept,
    // grown with the images) as they are normed and turned, the output in the NPU's buffer; a failed call: that layer
    // and the rest on the CPU. (aarch64 only: the phone's.)
    const size_t nblk = static_cast<size_t>(N + 31) / 32, dgs = static_cast<size_t>(HD) / 32;
    const size_t qbytes = static_cast<size_t>(N) * QW * sizeof(float);
#if defined(__aarch64__)
    bool att_npu = prod.npu_live() && npu->attn && npu->attn_buf && HD % 32 == 0 &&
                   !(std::getenv("GEODE_VISION_NPU_ATTN") && std::atoi(std::getenv("GEODE_VISION_NPU_ATTN")) == 0);
#else
    bool att_npu = false;
#endif
    float *qb = nullptr, *ob = nullptr;
    if (att_npu) {
        qb = npu->attn_buf(npu->g, 0, qbytes);
        ob = npu->attn_buf(npu->g, 1, qbytes);
        const size_t kvb = static_cast<size_t>(NH) * nblk * dgs * 2048u;
        if (qb && ob && (npu_kv_g_ != npu->g || npu_kv_cap_ < kvb)) {
            npu_kv_g_ = npu->g;
            npu_kv_cap_ = 0;
            for (int j = 0; j < 2 && qb; ++j)
                if (!(npu_kv_[j] = static_cast<uint8_t *>(npu->shared(npu->g, kvb, 1, &npu_kv_fd_[j])))) qb = nullptr;
            if (qb) npu_kv_cap_ = kvb;
        }
        att_npu = qb && ob;
        if (att_npu && N % 32)   // the last block's positions past N: zero keys and values (the loop never writes them)
            for (int hh = 0; hh < NH; ++hh) {
                std::memset(npu_kv_[0] + (static_cast<size_t>(hh) * nblk + nblk - 1) * dgs * 2048u, 0, dgs * 2048u);
                for (size_t dg = 0; dg < dgs; ++dg)
                    std::memset(npu_kv_[1] + ((static_cast<size_t>(hh) * dgs + dg) * nblk + nblk - 1) * 2048u, 0, 2048u);
            }
    }
    prof_.att_npu = att_npu;

    for (int i = 0; i < s.depth; ++i) {
        if (!next.get()) {
            err = berr[i & 1];
            return false;
        }
        const uint8_t *buf = bb[i & 1].get();
        const uint64_t base = bbase[i & 1];
        next = start(i + 1);   // the next layer, or the embedder
        lap(prof_.wait);
        const std::string b = "vision.layers." + std::to_string(i) + ".";
        const Packed *pk = mmla ? bpk[i & 1].data() : nullptr;
        const F16 *fl = f16ok[i & 1] ? f16d[i & 1].data() : nullptr;
        const auto lin = [&](int j) { return at(b + kLinears[j] + std::string(".weight"), buf, base); };
        const auto clip = [&](int j) { return vec(b + kLinears[j] + std::string(".clip"), buf, base); };
        const auto rms = [&](const char *name, const float *src, float *dst) {
            const float *wn = vec(b + name + std::string(".weight"), buf, base);
            pool.parallel_for(N, [&](int64_t a, int64_t e) {
                for (int64_t n = a; n < e; ++n)
                    rmsnorm(src + static_cast<size_t>(n) * D, wn, dst + static_cast<size_t>(n) * D, D, s.eps, false);
            }, 8, true);
        };

        // attention
        rms("input_layernorm", x.data(), hn.data());
        lap(prof_.other);
        float *const qd = att_npu ? qb : q.data();   // the queries (in the NPU's buffer when it attends)
        clipped(lin(0), pk ? pk + 0 : nullptr, clip(0), hn.data(), qd, N, fl ? fl + 0 : nullptr);
        clipped(lin(1), pk ? pk + 1 : nullptr, clip(1), hn.data(), k.data(), N, fl ? fl + 1 : nullptr);
        clipped(lin(2), pk ? pk + 2 : nullptr, clip(2), hn.data(), vv.data(), N, fl ? fl + 2 : nullptr);
        lap(prof_.matmul);
        const float *qn = vec(b + "self_attn.q_norm.weight", buf, base), *kn = vec(b + "self_attn.k_norm.weight", buf, base);
        // the CPU's keys (transposed) and values from the normed and turned rows
        const auto cpu_kv = [&](int64_t n, int hh) {
            const float *kh = k.data() + static_cast<size_t>(n) * QW + hh * HD, *vr = vv.data() + static_cast<size_t>(n) * QW + hh * HD;
            float *kth = kt.data() + static_cast<size_t>(hh) * HD * npad;
            for (int d = 0; d < HD; ++d) kth[static_cast<size_t>(d) * npad + n] = kh[d];
            std::copy(vr, vr + HD, vh.data() + (static_cast<size_t>(hh) * N + n) * HD);
        };
        const bool blocks = att_npu;
        pool.parallel_for(N, [&](int64_t a, int64_t e) {
            for (int64_t n = a; n < e; ++n) {
                const float *cs = cosv.data() + static_cast<size_t>(n) * half, *sn = sinv.data() + static_cast<size_t>(n) * half;
                for (int hh = 0; hh < NH; ++hh) {
                    float *qh = qd + static_cast<size_t>(n) * QW + hh * HD, *kh = k.data() + static_cast<size_t>(n) * QW + hh * HD;
                    float *vr = vv.data() + static_cast<size_t>(n) * QW + hh * HD;
                    rmsnorm(qh, qn, qh, HD, s.eps, false);
                    rmsnorm(kh, kn, kh, HD, s.eps, false);
                    rmsnorm(vr, nullptr, vr, HD, s.eps, false);
                    for (int hk = 0; hk < 2; ++hk)
                        for (int j = 0; j < quarter; ++j) {
                            const int d0 = hk * half + j, d1 = d0 + quarter, ci = hk * quarter + j;
                            const float q1 = qh[d0], q2 = qh[d1], k1 = kh[d0], k2 = kh[d1];
                            qh[d0] = q1 * cs[ci] - q2 * sn[ci];
                            qh[d1] = q2 * cs[ci] + q1 * sn[ci];
                            kh[d0] = k1 * cs[ci] - k2 * sn[ci];
                            kh[d1] = k2 * cs[ci] + k1 * sn[ci];
                        }
#if defined(__aarch64__)
                    if (blocks) {   // into the ATTN_ENCODER blocks (attn_cfg.h): the position p of its block bn
                        const size_t p = static_cast<size_t>(n) % 32, bn = static_cast<size_t>(n) / 32;
                        __fp16 *kb = reinterpret_cast<__fp16 *>(npu_kv_[0]) + (static_cast<size_t>(hh) * nblk + bn) * dgs * 1024u;
                        __fp16 *vb = reinterpret_cast<__fp16 *>(npu_kv_[1]);
                        for (int d = 0; d < HD; d += 2) {
                            __fp16 *o = kb + static_cast<size_t>(d / 32) * 1024u + static_cast<size_t>((d % 32) / 2) * 64u + p * 2u;
                            o[0] = static_cast<__fp16>(kh[d]);
                            o[1] = static_cast<__fp16>(kh[d + 1]);
                        }
                        for (int d = 0; d < HD; ++d)
                            vb[((static_cast<size_t>(hh) * dgs + static_cast<size_t>(d / 32)) * nblk + bn) * 1024u + (p / 2) * 64u +
                               static_cast<size_t>(d % 32) * 2u + p % 2] = static_cast<__fp16>(vr[d]);
                        continue;
                    }
#endif
                    cpu_kv(n, hh);
                }
            }
        }, 8, true);
        const float *ao = hn.data();   // the attention's output rows
        bool attended = false;
        if (att_npu) {
            attn_desc_t ad;
            std::memset(&ad, 0, sizeof ad);
            ad.version = ATTN_VERSION;
            ad.n = N, ad.apos = 0, ad.nh = NH, ad.nkv = NH, ad.hd = HD, ad.window = 0;
            ad.n_attn = 1, ad.kv_chunk = 32, ad.n_chunks = 2;
            // (GEODE_VISION_ATTN_FLAGS: more attn_cfg.h flags, testing - ATTN_MAX_SB for the superblocks' joins)
            static const uint32_t more = std::getenv("GEODE_VISION_ATTN_FLAGS")
                                             ? static_cast<uint32_t>(std::strtoul(std::getenv("GEODE_VISION_ATTN_FLAGS"), nullptr, 0))
                                             : 0u;
            ad.q_stride = QW, ad.o_stride = QW, ad.flags = static_cast<int32_t>(ATTN_ENCODER | more);
            ad.kv_fd[0] = npu_kv_fd_[0], ad.kv_fd[1] = npu_kv_fd_[1];
            double ds = 0;
            if (npu->attn(npu->g, &ad, static_cast<int>(sizeof ad), qb, qbytes, ob, qbytes, &ds) == 0) {
                ao = ob;
                attended = true;
                prof_.att_dsp += ds;
            } else {   // this layer's keys and values for the CPU after all, and the rest there
                att_npu = false;
                prof_.att_npu = false;
                pool.parallel_for(N, [&](int64_t a, int64_t e) {
                    for (int64_t n = a; n < e; ++n)
                        for (int hh = 0; hh < NH; ++hh) cpu_kv(n, hh);
                }, 8, true);
            }
        }
        if (!attended)
            attend_all(pool, NH, HD, N, npad, qd, static_cast<size_t>(QW), kt.data(), vh.data(), hn.data(), static_cast<size_t>(QW));   // scale 1
        lap(prof_.attention);
        clipped(lin(3), pk ? pk + 3 : nullptr, clip(3), ao, y.data(), N, fl ? fl + 3 : nullptr);
        lap(prof_.matmul);
        rms("post_attention_layernorm", y.data(), y.data());
        pool.parallel_for(static_cast<int64_t>(N) * D, [&](int64_t a, int64_t e) {
            for (int64_t j = a; j < e; ++j) x[static_cast<size_t>(j)] += y[static_cast<size_t>(j)];
        }, 4096, true);
        rms("pre_feedforward_layernorm", x.data(), hn.data());
        lap(prof_.other);

        // the MLP: down(gelu(gate) * up)
        clipped(lin(4), pk ? pk + 4 : nullptr, clip(4), hn.data(), g.data(), N, fl ? fl + 4 : nullptr);
        clipped(lin(5), pk ? pk + 5 : nullptr, clip(5), hn.data(), u.data(), N, fl ? fl + 5 : nullptr);
        lap(prof_.matmul);
        pool.parallel_for(static_cast<int64_t>(N) * F, [&](int64_t a, int64_t e) {
            for (int64_t j = a; j < e; ++j) g[static_cast<size_t>(j)] = gelu_tanh(g[static_cast<size_t>(j)]) * u[static_cast<size_t>(j)];
        }, 4096, true);
        lap(prof_.other);
        clipped(lin(6), pk ? pk + 6 : nullptr, clip(6), g.data(), y.data(), N, fl ? fl + 6 : nullptr);
        lap(prof_.matmul);
        rms("post_feedforward_layernorm", y.data(), y.data());
        pool.parallel_for(static_cast<int64_t>(N) * D, [&](int64_t a, int64_t e) {
            for (int64_t j = a; j < e; ++j) x[static_cast<size_t>(j)] += y[static_cast<size_t>(j)];
        }, 4096, true);
        lap(prof_.other);
    }

    // each M x M patches averaged into a token (row-major over the pooled grid), times sqrt(hidden); a weightless
    // RMS norm; the embedder's projection
    if (!next.get()) {
        err = berr[s.depth & 1];
        return false;
    }
    const uint8_t *buf = bb[s.depth & 1].get();
    const uint64_t base = bbase[s.depth & 1];
    bb[(s.depth + 1) & 1].reset();
    lap(prof_.wait);
    const int pw = gw / M, T = (gh / M) * pw;
    std::vector<float> tok(static_cast<size_t>(T) * D, 0.0f);
    for (int n = 0; n < N; ++n) {
        float *o = tok.data() + static_cast<size_t>((n / gw / M) * pw + (n % gw) / M) * D;
        const float *xr = x.data() + static_cast<size_t>(n) * D;
        for (int i = 0; i < D; ++i) o[i] += xr[i];
    }
    const float scale = std::sqrt(static_cast<float>(D)), inv9 = 1.0f / static_cast<float>(M * M);
    pool.parallel_for(T, [&](int64_t a, int64_t e) {
        for (int64_t t = a; t < e; ++t) {
            float *o = tok.data() + static_cast<size_t>(t) * D;
            for (int i = 0; i < D; ++i) o[i] = o[i] * inv9 * scale;
            rmsnorm(o, nullptr, o, D, s.eps, false);
        }
    }, 4, true);
    out.assign(static_cast<size_t>(T) * s.out_hidden, 0.0f);
    mm(at("vision.embed.weight", buf, base), mmla ? bpk[s.depth & 1].data() : nullptr, tok.data(), static_cast<size_t>(D),
       out.data(), static_cast<size_t>(s.out_hidden), T, f16ok[s.depth & 1] ? f16d[s.depth & 1].data() : nullptr);
    prof_.where = on_npu ? (prod.npu_live() ? "NPU HMX" : "NPU HMX, then the CPU") : mmla ? "CPU SMMLA" : "CPU";
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
                  "vision: %d patches %.2f s (reads waited %.2f, embedding %.2f, products %.2f on the %s, attention %.2f%s, "
                  "other %.2f)",
                  prof_.patches, prof_.total, prof_.wait, prof_.embed, prof_.matmul, prof_.where, prof_.attention,
                  prof_.att_npu ? " on HMX" : "", prof_.other);
    return b;
}

}  // namespace geode
