// yue2.cpp - see yue2.hpp. The products are fp32 sums over the checkpoint's bf16 weights; vectors are clang / GCC
// vector extensions (AVX2 on the PC, NEON on the phone), no -ffast-math.
#include "yue2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

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
inline float hsum(v8 v) {
    return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
}
inline float dot(const float *a, const float *b, int n) {   // n % 8 == 0
    v8 s0 = {}, s1 = {};
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 += load8(a + i) * load8(b + i);
        s1 += load8(a + i + 8) * load8(b + i + 8);
    }
    for (; i < n; i += 8) s0 += load8(a + i) * load8(b + i);
    return hsum(s0 + s1);
}

// RMSNorm as modeling_yue2.RMSNorm: x * rsqrt(mean(x^2) + eps) * w (the mean accumulated in double)
void rms_row(const float *x, const float *w, float *out, int n, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
    const float r = 1.0f / std::sqrt(static_cast<float>(s / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] * r) * w[i];
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// RoPE on one head (rotate_half: the first half with the second)
inline void rope_head(float *x, const float *cos, const float *sin, int hd) {
    const int h = hd / 2;
    for (int i = 0; i < h; ++i) {
        const float a = x[i], b = x[i + h];
        x[i] = a * cos[i] - b * sin[i];
        x[i + h] = b * cos[i] + a * sin[i];
    }
}

// Softmax attention of n query rows [n][heads * hd] over keys / values [L][kvh * hd] (row i, causal: the keys
// [0, prior + i]); scale 1 / sqrt(hd) (SDPA's default); query head h reads KV head h / (heads / kvh) (enable_gqa).
void attention(const float *q, int n, const float *K, const float *V, int L, int prior, bool causal, int heads,
               int kvh, int hd, float *out, ThreadPool &pool) {
    const int group = heads / kvh, qd = heads * hd, kd = kvh * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // a task: one head, QB queries; each key's and value's row is read once for the block (the scores of the block
    // in s [QB][L], their softmax per row, then O = P V with the block's rows of O in o [QB][hd])
    constexpr int QB = 8, MAXD = 256;
    const int64_t qblocks = (n + QB - 1) / QB;
    const int chunks = hd / 8;
    pool.parallel_for(qblocks * heads, [&](int64_t a, int64_t b) {
        std::vector<float> s(static_cast<size_t>(QB) * L), o(static_cast<size_t>(QB) * hd);
        for (int64_t task = a; task < b; ++task) {
            const int h = static_cast<int>(task % heads), kh = h / group;
            const int i0 = static_cast<int>(task / heads) * QB, nb = std::min(n, i0 + QB) - i0;
            const float *qr[QB];
            int seen[QB], most = 0;
            for (int i = 0; i < nb; ++i) {
                qr[i] = q + static_cast<int64_t>(i0 + i) * qd + h * hd;
                seen[i] = causal ? prior + i0 + i + 1 : L;
                most = std::max(most, seen[i]);
            }
            for (int j = 0; j < most; ++j) {
                const float *kr = K + static_cast<int64_t>(j) * kd + kh * hd;
                v8 acc[QB] = {};
                for (int c = 0; c < chunks && c < MAXD / 8; ++c) {
                    const v8 kc = load8(kr + 8 * c);
                    for (int i = 0; i < nb; ++i) acc[i] += load8(qr[i] + 8 * c) * kc;
                }
                for (int i = 0; i < nb; ++i) s[static_cast<size_t>(i) * L + j] = hsum(acc[i]) * scale;
            }
            float sum[QB];
            for (int i = 0; i < nb; ++i) {
                float *si = &s[static_cast<size_t>(i) * L];
                float mx = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < seen[i]; ++j) mx = std::max(mx, si[j]);
                float z = 0.0f;
                for (int j = 0; j < seen[i]; ++j) z += (si[j] = std::exp(si[j] - mx));
                for (int j = seen[i]; j < most; ++j) si[j] = 0.0f;   // (causal: keys past this row's own)
                sum[i] = z;
            }
            std::fill(o.begin(), o.end(), 0.0f);
            for (int j = 0; j < most; ++j) {
                const float *vr = V + static_cast<int64_t>(j) * kd + kh * hd;
                for (int c = 0; c < chunks; ++c) {
                    const v8 vc = load8(vr + 8 * c);
                    for (int i = 0; i < nb; ++i) {
                        float *oi = &o[static_cast<size_t>(i) * hd + 8 * c];
                        store8(oi, load8(oi) + s[static_cast<size_t>(i) * L + j] * vc);
                    }
                }
            }
            for (int i = 0; i < nb; ++i) {
                float *orow = out + static_cast<int64_t>(i0 + i) * qd + h * hd;
                for (int d = 0; d < hd; ++d) orow[d] = o[static_cast<size_t>(i) * hd + d] / sum[i];
            }
        }
    }, 1);
}

}  // namespace

// ---- the product -----------------------------------------------------------------------------------------------

void gemm_bf16(const float *x, int64_t n, int64_t in, const uint16_t *w, int64_t out, const float *bias, float *y,
               ThreadPool &pool) {
    constexpr int64_t RB = 32, TB = 64;
    const int64_t rtasks = (out + RB - 1) / RB, ttasks = (n + TB - 1) / TB;
    pool.parallel_for(rtasks * ttasks, [&](int64_t a, int64_t b) {
        std::vector<float> wf(static_cast<size_t>(RB * in));
        int64_t have = -1;
        for (int64_t task = a; task < b; ++task) {
            const int64_t rb = task / ttasks, tb = task % ttasks;
            const int64_t r0 = rb * RB, r1 = std::min(out, r0 + RB), t0 = tb * TB, t1 = std::min(n, t0 + TB);
            if (rb != have) {   // this block's weight rows as floats, once for all its token blocks
                for (int64_t r = r0; r < r1; ++r)
                    for (int64_t c = 0; c < in; ++c) wf[static_cast<size_t>((r - r0) * in + c)] = bf16f(w[r * in + c]);
                have = rb;
            }
            int64_t t = t0;
            for (; t + 2 <= t1; t += 2) {
                const float *xa = x + t * in, *xb = xa + in;
                int64_t r = r0;
                for (; r + 4 <= r1; r += 4) {
                    const float *w0 = &wf[static_cast<size_t>((r - r0) * in)], *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
                    v8 a0 = {}, a1 = {}, a2 = {}, a3 = {}, b0 = {}, b1 = {}, b2 = {}, b3 = {};
                    for (int64_t k = 0; k < in; k += 8) {
                        const v8 p = load8(xa + k), q = load8(xb + k);
                        const v8 u0 = load8(w0 + k), u1 = load8(w1 + k), u2 = load8(w2 + k), u3 = load8(w3 + k);
                        a0 += u0 * p, a1 += u1 * p, a2 += u2 * p, a3 += u3 * p;
                        b0 += u0 * q, b1 += u1 * q, b2 += u2 * q, b3 += u3 * q;
                    }
                    float *ya = y + t * out + r, *yb = ya + out;
                    ya[0] = hsum(a0), ya[1] = hsum(a1), ya[2] = hsum(a2), ya[3] = hsum(a3);
                    yb[0] = hsum(b0), yb[1] = hsum(b1), yb[2] = hsum(b2), yb[3] = hsum(b3);
                }
                for (; r < r1; ++r) {
                    const float *wr = &wf[static_cast<size_t>((r - r0) * in)];
                    y[t * out + r] = dot(wr, xa, static_cast<int>(in));
                    y[(t + 1) * out + r] = dot(wr, xb, static_cast<int>(in));
                }
            }
            for (; t < t1; ++t)
                for (int64_t r = r0; r < r1; ++r)
                    y[t * out + r] = dot(&wf[static_cast<size_t>((r - r0) * in)], x + t * in, static_cast<int>(in));
            if (bias)
                for (int64_t tt = t0; tt < t1; ++tt)
                    for (int64_t r = r0; r < r1; ++r) y[tt * out + r] += bias[r];
        }
    }, 1);
}

// ---- the model -------------------------------------------------------------------------------------------------

bool read_config(const Pack &pack, Config &c, std::string &err) {
    const auto &j = pack.config();
    c = Config{};
    c.hidden = static_cast<int>(j["hidden_size"].i64());
    c.layers = static_cast<int>(j["num_hidden_layers"].i64());
    c.heads = static_cast<int>(j["num_attention_heads"].i64());
    c.kv_heads = static_cast<int>(j["num_key_value_heads"].i64());
    c.head_dim = static_cast<int>(j["head_dim"].i64());
    c.ffn = static_cast<int>(j["intermediate_size"].i64());
    c.vocab = static_cast<int>(j["vocab_size"].i64());
    c.max_pos = static_cast<int>(j["max_position_embeddings"].i64());
    c.latent = static_cast<int>(j["latent_dim"].i64());
    c.max_latent_frames = static_cast<int>(j["max_latent_frames"].i64());
    c.freq = static_cast<int>(j["time_frequency_size"].i64(256));
    c.eps = static_cast<float>(j["rms_norm_eps"].f64(1e-6));
    c.shift = j["timestep_shift"].f64(1.0);
    const auto &bits = j["rope_inv_freq_bits"];
    for (size_t i = 0; i < bits.size(); ++i) {
        const uint32_t u = static_cast<uint32_t>(bits.at(i).i64());
        float f;
        std::memcpy(&f, &u, 4);
        c.inv_freq.push_back(f);
    }
    const auto &t = j["tokens"];
    c.tok = {static_cast<int>(t["eod"].i64()), static_cast<int>(t["abc_start"].i64()),
             static_cast<int>(t["abc_end"].i64()), static_cast<int>(t["music_start"].i64()),
             static_cast<int>(t["music_end"].i64()), static_cast<int>(t["codec_offset"].i64()),
             static_cast<int>(t["codec_size"].i64()), static_cast<int>(t["context"].i64())};
    const auto samp = [](const json::Value &s) {
        Sampling r;
        r.temperature = static_cast<float>(s["temperature"].f64());
        r.top_p = static_cast<float>(s["top_p"].f64());
        r.top_k = static_cast<int>(s["top_k"].i64());
        r.repetition_penalty = static_cast<float>(s["repetition_penalty"].f64());
        r.penalty_window = static_cast<int>(s["penalty_window"].i64());
        r.min_tokens = static_cast<int>(s["min_tokens"].i64());
        r.max_tokens = static_cast<int>(s["max_tokens"].i64());
        return r;
    };
    c.abc = samp(j["sampling"]["abc"]);
    c.semantic = samp(j["sampling"]["semantic"]);
    c.ode_steps = static_cast<int>(j["sampling"]["ode_steps"].i64(32));
    // what the kernels assume: 8-float rows (gemm_bf16, attention), head_dim <= 256 (attention's MAXD)
    const auto &k = c.tok;
    if (c.hidden <= 0 || c.layers <= 0 || c.heads <= 0 || c.kv_heads <= 0 || c.heads % c.kv_heads ||
        c.head_dim <= 0 || c.head_dim % 16 || c.head_dim > 256 || c.hidden % 8 || c.ffn <= 0 || c.ffn % 8 ||
        c.latent <= 0 || c.latent % 8 || c.freq <= 0 || c.freq % 16 || c.max_pos <= 0 ||
        c.max_latent_frames <= 0 || !(c.eps > 0.0f) || !std::isfinite(c.shift) || c.shift <= 0.0 ||
        static_cast<int>(c.inv_freq.size()) != c.head_dim / 2) {
        err = "geode.json: inconsistent YuE2 geometry";
        return false;
    }
    if (k.eod <= 0 || k.codec_size <= 0 || k.codec_offset <= 0 || k.codec_offset + k.codec_size > c.vocab ||
        k.abc_end >= c.vocab || k.music_end >= c.vocab || k.music_start >= c.vocab || k.abc_start >= c.vocab ||
        k.context <= 0 || k.context > c.max_pos) {
        err = "geode.json: token ids outside the vocabulary";
        return false;
    }
    for (const Sampling *s : {&c.abc, &c.semantic})
        if (!(s->temperature >= 0.0f) || !std::isfinite(s->temperature) || !(s->top_p > 0.0f && s->top_p <= 1.0f) ||
            s->top_k < 1 || !(s->repetition_penalty > 0.0f) || !std::isfinite(s->repetition_penalty) ||
            s->penalty_window < 0 || s->min_tokens < 0 || s->max_tokens < 1) {
            err = "geode.json: bad sampling settings";
            return false;
        }
    if (c.ode_steps < 1) {
        err = "geode.json: ode_steps < 1";
        return false;
    }
    return true;
}

bool Model::open(Pack &pack, std::string &err) {
    pack_ = &pack;
    if (!read_config(pack, c_, err) || !pack.load("ar.bin", err) || !pack.load("nar.bin", err)) return false;
    const int64_t H = c_.hidden, qd = static_cast<int64_t>(c_.heads) * c_.head_dim,
                  kd = static_cast<int64_t>(c_.kv_heads) * c_.head_dim, F = c_.ffn, hd = c_.head_dim;
    bool ok = true;
    const auto m = [&](const std::string &name, std::vector<int64_t> shape) {
        const uint16_t *p = pack.bf16(name, shape, err);
        ok = ok && p;
        return p;
    };
    const auto f = [&](const std::string &name, std::vector<int64_t> shape) {
        const float *p = pack.f32(name, shape, err);
        ok = ok && p;
        return p;
    };
    layers_.resize(static_cast<size_t>(c_.layers));
    for (int l = 0; l < c_.layers && ok; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        for (int nar = 0; nar < 2 && ok; ++nar) {
            Path &p = nar ? layers_[static_cast<size_t>(l)].nar : layers_[static_cast<size_t>(l)].ar;
            const std::string a = b + (nar ? "nar_self_attn." : "self_attn."), mm = b + (nar ? "nar_mlp." : "mlp.");
            p.in_norm = f(b + (nar ? "nar_input_layernorm.weight" : "input_layernorm.weight"), {H});
            p.mlp_norm = f(b + (nar ? "nar_pre_mlp_layernorm.weight" : "post_attention_layernorm.weight"), {H});
            p.attn.q = m(a + "q_proj.weight", {qd, H});
            p.attn.k = m(a + "k_proj.weight", {kd, H});
            p.attn.v = m(a + "v_proj.weight", {kd, H});
            p.attn.o = m(a + "o_proj.weight", {H, qd});
            p.attn.q_norm = f(a + "q_norm.weight", {hd});
            p.attn.k_norm = f(a + "k_norm.weight", {hd});
            p.mlp.gate = m(mm + "gate_proj.weight", {F, H});
            p.mlp.up = m(mm + "up_proj.weight", {F, H});
            p.mlp.down = m(mm + "down_proj.weight", {H, F});
        }
    }
    norm_ = f("model.norm.weight", {H});
    lm_head_ = m("lm_head.weight", {c_.vocab, H});
    vae2llm_w_ = m("vae2llm.weight", {H, c_.latent});
    vae2llm_b_ = f("vae2llm.bias", {H});
    llm2vae_w_ = m("llm2vae.weight", {c_.latent, H});
    llm2vae_b_ = f("llm2vae.bias", {c_.latent});
    t0_w_ = m("time_embedder.mlp.0.weight", {H, c_.freq});
    t0_b_ = f("time_embedder.mlp.0.bias", {H});
    t2_w_ = m("time_embedder.mlp.2.weight", {H, H});
    t2_b_ = f("time_embedder.mlp.2.bias", {H});
    pe_ = m("latent_pos_embed.pe", {c_.max_latent_frames, H});
    embed_ = pack.find("model.embed_tokens.weight");
    if (ok && (!embed_ || embed_->type != DType::BF16 || embed_->shape != std::vector<int64_t>{c_.vocab, H})) {
        err = "model.embed_tokens.weight: missing or not bf16 [vocab, hidden]";
        ok = false;
    }
    return ok;
}

bool Model::cache_init(KvCache &c, int capacity, std::string &err) const {
    if (capacity < 1 || capacity > c_.tok.context) {   // the reference: prefix + budget <= 24576, no truncation
        err = "cache_init: capacity " + std::to_string(capacity) + " outside 1 .. " + std::to_string(c_.tok.context);
        return false;
    }
    const size_t n = static_cast<size_t>(c_.layers) * static_cast<size_t>(capacity) *
                     static_cast<size_t>(c_.kv_heads * c_.head_dim);
    c.len = 0;
    c.cap = capacity;
    c.k.assign(n, 0.0f);
    c.v.assign(n, 0.0f);
    return true;
}

void Model::rope_rows(int pos0, int n, std::vector<float> &cos, std::vector<float> &sin) const {
    const int h = c_.head_dim / 2;
    cos.resize(static_cast<size_t>(n) * h);
    sin.resize(static_cast<size_t>(n) * h);
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < h; ++i) {
            const float angle = static_cast<float>(pos0 + r) * c_.inv_freq[static_cast<size_t>(i)];   // fp32, as torch
            cos[static_cast<size_t>(r) * h + i] = std::cos(angle);
            sin[static_cast<size_t>(r) * h + i] = std::sin(angle);
        }
}

void Model::layer(const Path &p, float *x, int n, const float *cos, const float *sin, const float *pk,
                  const float *pv, int prior, bool causal, float *new_k, float *new_v, ThreadPool &pool) const {
    const int H = c_.hidden, hd = c_.head_dim, nh = c_.heads, nkv = c_.kv_heads, qd = nh * hd, kd = nkv * hd,
              F = c_.ffn, half = hd / 2;
    const size_t nn = static_cast<size_t>(n);
    std::vector<float> h(nn * H), q(nn * qd), att(nn * qd), tmp(nn * H);
    pool.parallel_for(n, [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; ++r) rms_row(x + r * H, p.in_norm, &h[static_cast<size_t>(r) * H], H, c_.eps);
    });
    gemm_bf16(h.data(), n, H, p.attn.q, qd, nullptr, q.data(), pool);
    gemm_bf16(h.data(), n, H, p.attn.k, kd, nullptr, new_k, pool);
    gemm_bf16(h.data(), n, H, p.attn.v, kd, nullptr, new_v, pool);
    pool.parallel_for(n, [&](int64_t a, int64_t b) {   // q_norm / k_norm per head, then RoPE
        std::vector<float> t(static_cast<size_t>(hd));
        for (int64_t r = a; r < b; ++r) {
            const float *c = cos + r * half, *s = sin + r * half;
            for (int hh = 0; hh < nh; ++hh) {
                float *v = &q[static_cast<size_t>(r) * qd + static_cast<size_t>(hh) * hd];
                rms_row(v, p.attn.q_norm, t.data(), hd, c_.eps);
                std::memcpy(v, t.data(), sizeof(float) * hd);
                rope_head(v, c, s, hd);
            }
            for (int hh = 0; hh < nkv; ++hh) {
                float *v = new_k + r * kd + static_cast<int64_t>(hh) * hd;
                rms_row(v, p.attn.k_norm, t.data(), hd, c_.eps);
                std::memcpy(v, t.data(), sizeof(float) * hd);
                rope_head(v, c, s, hd);
            }
        }
    });
    // the keys: pk / pv hold [0, prior) and new_k / new_v must follow them in the same buffers
    attention(q.data(), n, pk, pv, prior + n, prior, causal, nh, nkv, hd, att.data(), pool);
    gemm_bf16(att.data(), n, qd, p.attn.o, H, nullptr, tmp.data(), pool);
    for (size_t i = 0; i < nn * H; ++i) x[i] += tmp[i];
    pool.parallel_for(n, [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; ++r) rms_row(x + r * H, p.mlp_norm, &h[static_cast<size_t>(r) * H], H, c_.eps);
    });
    std::vector<float> g(nn * F), u(nn * F);
    gemm_bf16(h.data(), n, H, p.mlp.gate, F, nullptr, g.data(), pool);
    gemm_bf16(h.data(), n, H, p.mlp.up, F, nullptr, u.data(), pool);
    pool.parallel_for(static_cast<int64_t>(nn * F), [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) g[static_cast<size_t>(i)] = silu(g[static_cast<size_t>(i)]) * u[static_cast<size_t>(i)];
    }, 4096);
    gemm_bf16(g.data(), n, F, p.mlp.down, H, nullptr, tmp.data(), pool);
    for (size_t i = 0; i < nn * H; ++i) x[i] += tmp[i];
}

bool Model::ar_forward(const int32_t *tokens, int n, KvCache &c, const std::vector<int> &rows, float *hidden,
                       ThreadPool &pool, std::string &err) const {
    if (n <= 0 || c.len + n > c.cap) {
        err = "ar_forward: " + std::to_string(n) + " tokens past the cache's " + std::to_string(c.cap);
        return false;
    }
    for (const int r : rows)
        if (r < 0 || r >= n) {
            err = "ar_forward: row " + std::to_string(r) + " outside the " + std::to_string(n) + " tokens";
            return false;
        }
    const int H = c_.hidden, kd = c_.kv_heads * c_.head_dim;
    std::vector<float> x(static_cast<size_t>(n) * H);
    std::vector<uint16_t> row(static_cast<size_t>(H));
    for (int r = 0; r < n; ++r) {
        if (tokens[r] < 0 || tokens[r] >= c_.vocab || !pack_->read_rows(*embed_, tokens[r], 1, row.data())) {
            err = "ar_forward: token " + std::to_string(tokens[r]);
            return false;
        }
        for (int i = 0; i < H; ++i) x[static_cast<size_t>(r) * H + i] = bf16f(row[static_cast<size_t>(i)]);
    }
    std::vector<float> cos, sin;
    rope_rows(c.len, n, cos, sin);
    for (int l = 0; l < c_.layers; ++l) {
        float *K = c.k.data() + static_cast<size_t>(l) * c.cap * kd, *V = c.v.data() + static_cast<size_t>(l) * c.cap * kd;
        layer(layers_[static_cast<size_t>(l)].ar, x.data(), n, cos.data(), sin.data(), K, V, c.len, true,
              K + static_cast<size_t>(c.len) * kd, V + static_cast<size_t>(c.len) * kd, pool);
    }
    c.len += n;
    if (hidden)
        for (size_t i = 0; i < rows.size(); ++i)
            rms_row(&x[static_cast<size_t>(rows[i]) * H], norm_, hidden + i * H, H, c_.eps);
    return true;
}

bool Model::logits(const float *hidden, int n, int lo, int hi, float *out, ThreadPool &pool) const {
    if (n < 1 || lo < 0 || lo >= hi || hi > c_.vocab) return false;
    gemm_bf16(hidden, n, c_.hidden, lm_head_ + static_cast<size_t>(lo) * c_.hidden, hi - lo, nullptr, out, pool);
    return true;
}

bool Model::nar_init(Nar &nar, const KvCache &ar, int frames, std::string &err) const {
    // chunk_ranges keeps prefix + 2 frames + 3 <= context: the AR cache (prefix + frames + MUSIC_END) and the
    // NAR's own frames + 2 rows together inside the context, so positions never pass it
    if (frames < 1 || ar.len < 1 || ar.len + frames + 2 > c_.tok.context || frames + 2 > c_.max_latent_frames) {
        err = "nar_init: " + std::to_string(frames) + " frames after " + std::to_string(ar.len) +
              " AR tokens exceed the context " + std::to_string(c_.tok.context);
        return false;
    }
    nar.ar = &ar;
    nar.frames = frames;
    nar.tokens = frames + 2;
    rope_rows(ar.len, nar.tokens, nar.cos, nar.sin);
    return true;
}

void Model::velocity(const Nar &nar, const float *state, double raw_t, float *v, ThreadPool &pool,
                     Trace *trace) const {
    const int H = c_.hidden, L = c_.latent, N = nar.tokens, kd = c_.kv_heads * c_.head_dim, A = nar.ar->len;
    std::vector<float> xn(static_cast<size_t>(N) * L, 0.0f), x(static_cast<size_t>(N) * H);
    std::memcpy(&xn[static_cast<size_t>(L)], state, sizeof(float) * static_cast<size_t>(nar.frames) * L);
    gemm_bf16(xn.data(), N, L, vae2llm_w_, H, vae2llm_b_, x.data(), pool);
    // _shift_t_value in fp32, then TimestepEmbedder (sinusoid of 256, Linear, SiLU, Linear): one row for all tokens
    const float ts = 1.0f / (1.0f + std::exp(-static_cast<float>(raw_t)));
    const float shift = static_cast<float>(c_.shift), t = shift * ts / (1.0f + (shift - 1.0f) * ts);
    const int half = c_.freq / 2;
    std::vector<float> emb(static_cast<size_t>(c_.freq)), t0(static_cast<size_t>(H)), temb(static_cast<size_t>(H));
    for (int i = 0; i < half; ++i) {
        const float fr = std::exp((static_cast<float>(-std::log(10000.0)) * static_cast<float>(i)) / static_cast<float>(half));
        const float arg = t * fr;
        emb[static_cast<size_t>(i)] = std::cos(arg);
        emb[static_cast<size_t>(half + i)] = std::sin(arg);
    }
    gemm_bf16(emb.data(), 1, c_.freq, t0_w_, H, t0_b_, t0.data(), pool);
    for (auto &e : t0) e = silu(e);
    gemm_bf16(t0.data(), 1, H, t2_w_, H, t2_b_, temb.data(), pool);
    for (int r = 0; r < N; ++r) {
        const uint16_t *pe = pe_ + static_cast<size_t>(std::min(r, c_.max_latent_frames - 1)) * H;
        float *xr = &x[static_cast<size_t>(r) * H];
        for (int i = 0; i < H; ++i) xr[i] = (xr[i] + temb[static_cast<size_t>(i)]) + bf16f(pe[i]);
    }
    if (trace) {
        trace->t_shifted = t;
        trace->x_in = x;
    }
    std::vector<float> K(static_cast<size_t>(A + N) * kd), V(static_cast<size_t>(A + N) * kd);
    for (int l = 0; l < c_.layers; ++l) {
        const size_t base = static_cast<size_t>(l) * nar.ar->cap * kd;
        std::memcpy(K.data(), nar.ar->k.data() + base, sizeof(float) * static_cast<size_t>(A) * kd);
        std::memcpy(V.data(), nar.ar->v.data() + base, sizeof(float) * static_cast<size_t>(A) * kd);
        layer(layers_[static_cast<size_t>(l)].nar, x.data(), N, nar.cos.data(), nar.sin.data(), K.data(), V.data(), A,
              false, K.data() + static_cast<size_t>(A) * kd, V.data() + static_cast<size_t>(A) * kd, pool);
        if (trace && trace->x_after.count(l)) trace->x_after[l] = x;
    }
    std::vector<float> hn(static_cast<size_t>(nar.frames) * H);
    for (int r = 0; r < nar.frames; ++r)
        rms_row(&x[static_cast<size_t>(r + 1) * H], norm_, &hn[static_cast<size_t>(r) * H], H, c_.eps);
    gemm_bf16(hn.data(), nar.frames, H, llm2vae_w_, L, llm2vae_b_, v, pool);
}

bool Model::solve(const Nar &nar, const float *noise, int steps, float *out, ThreadPool &pool, std::string &err,
                  const std::function<void(int, int)> &progress) const {
    const size_t n = static_cast<size_t>(nar.frames) * c_.latent;
    if (steps < 1 || !nar.ar) {
        err = "solve: steps must be >= 1 (and nar_init first)";
        return false;
    }
    std::vector<float> state(noise, noise + n), first(n), mid(n), second(n);
    const double dt = 1.0 / steps;
    const auto raw = [](double t) {   // torch.logit in float64, clamped to +-20
        const double r = std::log(t / (1.0 - t));
        return std::isnan(r) ? r : std::min(20.0, std::max(-20.0, r));
    };
    const float half = static_cast<float>(dt / 2), full = static_cast<float>(dt);
    for (int step = 0; step < steps; ++step) {
        const double t = 1.0 - step * dt;
        velocity(nar, state.data(), raw(t), first.data(), pool);
        for (size_t i = 0; i < n; ++i) mid[i] = state[i] - first[i] * half;
        velocity(nar, mid.data(), raw(t - dt / 2), second.data(), pool);
        for (size_t i = 0; i < n; ++i) state[i] = state[i] - second[i] * full;
        if (progress) progress(step + 1, steps);
    }
    for (const float s : state)   // as nar.py: non-finite latents are an error, never audio
        if (!std::isfinite(s)) {
            err = "solve: acoustic flow matching produced non-finite latents";
            return false;
        }
    std::memcpy(out, state.data(), sizeof(float) * n);
    return true;
}

// ---- the sampler -----------------------------------------------------------------------------------------------

void distribution(const float *logits, int vocab, const Sampling &s, const Tokens &t, bool phase_abc, int step,
                  const std::vector<int> &history, std::vector<float> &probs) {
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> sc(static_cast<size_t>(vocab), ninf);
    const int end = phase_abc ? t.abc_end : t.music_end;
    const int lo = phase_abc ? 0 : t.codec_offset, hi = phase_abc ? t.eod : t.codec_offset + t.codec_size;
    for (int i = lo; i < hi; ++i) sc[static_cast<size_t>(i)] = logits[i];
    sc[static_cast<size_t>(end)] = step < s.min_tokens ? ninf : logits[end];
    if (s.repetition_penalty != 1.0f && !history.empty()) {   // window_penalty: penalty ** count of the last window
        std::map<int, int> freq;
        const size_t from = history.size() > static_cast<size_t>(s.penalty_window) ? history.size() - s.penalty_window : 0;
        for (size_t i = from; i < history.size(); ++i) ++freq[history[i]];
        for (const auto &[id, cnt] : freq) {
            if (id < 0 || id >= vocab) continue;
            const float alpha = std::pow(s.repetition_penalty, static_cast<float>(cnt));
            float &v = sc[static_cast<size_t>(id)];
            v = v < 0 ? v * alpha : v / alpha;
        }
    }
    probs.assign(static_cast<size_t>(vocab), 0.0f);
    if (s.temperature == 0.0f) {   // greedy: all weight on the argmax
        probs[static_cast<size_t>(std::max_element(sc.begin(), sc.end()) - sc.begin())] = 1.0f;
        return;
    }
    if (s.temperature != 1.0f)
        for (auto &v : sc) v = v / s.temperature;
    // top-k: everything below the k-th largest masked (ties with it kept, as masked_fill(scores < threshold))
    std::vector<float> sorted(sc);
    const int k = std::min(s.top_k, vocab);
    std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<float>());
    const float threshold = sorted[static_cast<size_t>(k - 1)];
    for (auto &v : sc)
        if (v < threshold) v = ninf;
    if (s.top_p < 1.0f) {   // top-p over the descending order; the first always kept
        std::vector<int> idx(static_cast<size_t>(vocab));
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)]; });
        const float mx = sc[static_cast<size_t>(idx[0])];
        double z = 0.0;
        for (int i : idx) z += sc[static_cast<size_t>(i)] == ninf ? 0.0 : std::exp(static_cast<double>(sc[static_cast<size_t>(i)] - mx));
        double cum = 0.0;
        for (size_t r = 0; r < idx.size(); ++r) {
            float &v = sc[static_cast<size_t>(idx[r])];
            if (v == ninf) break;
            const double p = std::exp(static_cast<double>(v - mx)) / z;
            const bool removed = r > 0 && cum > s.top_p;   // cumsum - p > top_p: the mass before this one
            cum += p;
            if (removed) v = ninf;
        }
    }
    const float mx = *std::max_element(sc.begin(), sc.end());
    double z = 0.0;
    for (size_t i = 0; i < sc.size(); ++i)
        if (sc[i] != ninf) z += std::exp(static_cast<double>(sc[i] - mx));
    for (size_t i = 0; i < sc.size(); ++i)
        probs[i] = sc[i] == ninf ? 0.0f : static_cast<float>(std::exp(static_cast<double>(sc[i] - mx)) / z);
}

}  // namespace geode::yue2
