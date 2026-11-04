// drafter.cpp - Gemma 4's MTP drafter (the -assistant checkpoint) in geode::Model: tools/pack_gemma4_mtp.py packs it
// (drafter.bin, geode.json "drafter"), tools/gemma4_mtp.py is the spec this follows step by step.
//
// A draft round starts at position P: the token there sampled, the main model's KV holding 0 .. P-1. Each step takes
// the main model's (scaled) embedding of a token and a hidden row - the main model's at P-1 first, the drafter's own
// h' after that - through pre_projection (3072 -> 256), 4 layers that attend over the main model's KV (no K / V of
// their own: a sliding layer over the last W + 1 rows below P, the full one over all of them; their queries turned at
// P), the final norm; h' = post_projection (-> 1536); the draft = the likeliest of the top 32 centroids' 128 tokens
// each (the head's rows in cluster order). Its numbers only decide which drafts the main model checks: the answer is
// the main model's whatever they are (geode.cpp, geode_generate).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "model.hpp"
#include "platform.hpp"

namespace geode {

namespace {

inline float gelu_tanh(float x) {
    constexpr float k = 0.7978845608028654f;   // sqrt(2 / pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

inline float to_f(float v) { return v; }
inline float to_f(_Float16 v) { return static_cast<float>(v); }
inline float to_f(int8_t v) { return static_cast<float>(v); }

// q . row (hd values), eight partial sums
template <typename T>
inline float dot(const float *q, const T *k, int hd) {
    float s[8] = {};
    for (int i = 0; i < hd; i += 8)
        for (int j = 0; j < 8; ++j) s[j] += q[i + j] * to_f(k[i + j]);
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

template <typename T>
inline void axpy(float w, const T *v, float *out, int hd) {
    for (int i = 0; i < hd; ++i) out[i] += w * to_f(v[i]);
}

}  // namespace

bool Model::open_drafter(std::string &err) {
    const auto &c = cfg_;
    const json::Value &d = pack_.manifest()["drafter"];
    GDraft &g = gd_;
    g.hidden = static_cast<int>(d["hidden"].i64());
    g.heads = static_cast<int>(d["heads"].i64());
    g.ff = static_cast<int>(d["ff"].i64());
    g.window = static_cast<int>(d["window"].i64());
    g.centroids = static_cast<int>(d["centroids"].i64());
    g.top_k = static_cast<int>(d["top_k"].i64());
    g.eps = static_cast<float>(d["eps"].f64(1e-6));
    const int n_layers = static_cast<int>(d["layers"].i64());
    const int kv_s = static_cast<int>(d["kv_layer_sliding"].i64(-1)), kv_f = static_cast<int>(d["kv_layer_full"].i64(-1));
    if (d["arch"].str() != "gemma4-assistant" || g.hidden <= 0 || g.hidden % 16 || g.heads <= 0 || g.ff <= 0 ||
        g.ff % 16 || g.window <= 0 || g.heads % kv_heads_ ||
        n_layers <= 0 || g.centroids <= 0 || c.vocab % g.centroids || g.top_k <= 0 || g.top_k > g.centroids ||
        static_cast<int>(d["backbone_hidden"].i64()) != c.hidden || static_cast<int>(d["vocab"].i64()) != c.vocab ||
        d["layer_types"].size() != static_cast<size_t>(n_layers)) {
        err = "geode.json: a drafter section this engine does not take";
        return false;
    }
    g.per = c.vocab / g.centroids;
    const auto kv_of = [&](int layer, bool full) -> const GLayer * {
        if (layer < 0 || layer >= c.n_layers) return nullptr;
        const GLayer &L = glayers_[static_cast<size_t>(layer)];
        return L.own_kv && L.full == full ? &L : nullptr;
    };
    const GLayer *ls = kv_of(kv_s, false), *lf = kv_of(kv_f, true);
    if (!ls || !lf || ls->hd != static_cast<int>(d["head_dim"].i64()) || lf->hd != static_cast<int>(d["global_head_dim"].i64())) {
        err = "the drafter's KV layers are not the main model's own of their type";
        return false;
    }
    // drafter.bin into memory, whole
    const Tensor *any = pack_.find("drafter.head");
    if (!any) {
        err = "no drafter.bin";
        return false;
    }
    const RandomFile &f = pack_.file(any->file);
    g.bytes = f.size();
    g.data = aligned_bytes(static_cast<size_t>(g.bytes));
    if (!g.data || !f.read_at(0, g.data.get(), static_cast<size_t>(g.bytes))) {
        err = g.data ? "cannot read drafter.bin" : "not enough memory for the drafter";
        return false;
    }
    const auto find = [&](const std::string &n) -> const Tensor * {
        const Tensor *t = pack_.find(n);
        if (!t || t->file != any->file) err = n + ": not in drafter.bin";
        return t && t->file == any->file ? t : nullptr;
    };
    const auto f32 = [&](const std::string &n, int64_t count) -> const float * {
        const Tensor *t = find(n);
        if (!t) return nullptr;
        if (t->type != DType::F32 || t->rows * t->cols != count) {
            err = n + ": expected f32 x " + std::to_string(count);
            return nullptr;
        }
        return reinterpret_cast<const float *>(g.data.get() + t->offset);
    };
    const auto mat = [&](const std::string &n, int64_t rows, int64_t cols, QView &out) -> bool {
        const Tensor *t = find(n);
        if (!t) return false;
        if (!is_quantized(t->type) || t->rows != rows || t->cols != cols) {
            err = n + ": expected a quantized [" + std::to_string(rows) + ", " + std::to_string(cols) + "]";
            return false;
        }
        out = QView{t->type, rows, cols, row_bytes(t->type, cols), g.data.get() + t->offset, t->global};
        return true;
    };
    const int D = g.hidden, H = c.hidden;
    if (!mat("drafter.pre", D, 2 * H, g.pre) || !mat("drafter.post", H, D, g.post) ||
        !mat("drafter.centroids", g.centroids, D, g.cent) || !mat("drafter.head", c.vocab, D, g.head) ||
        !(g.norm = f32("drafter.norm", D)))
        return false;
    const float *order = f32("drafter.order", c.vocab);
    if (!order) return false;
    g.order.resize(static_cast<size_t>(c.vocab));
    std::vector<uint8_t> seen(static_cast<size_t>(c.vocab), 0);
    for (int r = 0; r < c.vocab; ++r) {
        const float o = order[r];
        const bool whole = std::isfinite(o) && o >= 0.0f && o < static_cast<float>(c.vocab) && o == std::floor(o);
        const int t = whole ? static_cast<int>(o) : -1;
        if (t < 0 || seen[static_cast<size_t>(t)]) {
            err = "drafter.order: not a permutation of the vocabulary";
            return false;
        }
        seen[static_cast<size_t>(t)] = 1;
        g.order[static_cast<size_t>(r)] = t;
    }
    g.layers.assign(static_cast<size_t>(n_layers), {});
    int widest = 0;
    for (int l = 0; l < n_layers; ++l) {
        GDraftLayer &L = g.layers[static_cast<size_t>(l)];
        const std::string p = "drafter.layers." + std::to_string(l) + ".";
        const std::string type = d["layer_types"].at(static_cast<size_t>(l)).str();
        if (type != "sliding_attention" && type != "full_attention") {
            err = "geode.json: drafter layer type " + type;
            return false;
        }
        L.full = type == "full_attention";
        const GLayer &M = L.full ? *lf : *ls;   // the main layer whose KV it reads (and whose rotary it shares)
        L.slot = M.slot;
        L.hd = M.hd;
        L.pairs = M.pairs;
        L.inv_freq = M.inv_freq;
        widest = std::max(widest, L.hd);
        const int64_t qw = static_cast<int64_t>(g.heads) * L.hd;
        if (!(L.in_norm = f32(p + "in_norm", D)) || !(L.q_norm = f32(p + "q_norm", L.hd)) ||
            !(L.attn_norm = f32(p + "attn_norm", D)) || !(L.ff_norm = f32(p + "ff_norm", D)) ||
            !(L.ff_post_norm = f32(p + "ff_post_norm", D)) || !mat(p + "q", qw, D, L.q) || !mat(p + "o", D, qw, L.o) ||
            !mat(p + "gate", g.ff, D, L.gate) || !mat(p + "up", g.ff, D, L.up) || !mat(p + "down", D, g.ff, L.down))
            return false;
        const float *s = f32(p + "scalar", 1);
        if (!s) return false;
        L.scalar = s[0];
    }
    const size_t qw = static_cast<size_t>(g.heads) * widest;
    g.in.assign(2 * static_cast<size_t>(H), 0.0f);
    g.x.assign(static_cast<size_t>(D), 0.0f);
    g.h.assign(static_cast<size_t>(D), 0.0f);
    g.n.assign(static_cast<size_t>(D), 0.0f);
    g.f.assign(static_cast<size_t>(D), 0.0f);
    g.q.assign(qw, 0.0f);
    g.a.assign(qw, 0.0f);
    g.g.assign(static_cast<size_t>(g.ff), 0.0f);
    g.u.assign(static_cast<size_t>(g.ff), 0.0f);
    g.hp.assign(static_cast<size_t>(H), 0.0f);
    g.cl.assign(static_cast<size_t>(g.centroids), 0.0f);
    g.lg.assign(static_cast<size_t>(g.top_k) * g.per, 0.0f);
    if (const char *k = std::getenv("GEODE_SPEC_K")) g.k = std::max(2, std::min(Gpu8::kMaxTokens, std::atoi(k)));
    g.on = true;
    return true;
}

void Model::draft_attend(const GDraftLayer &L) {
    GDraft &g = gd_;
    const int hd = L.hd, P = g.pos;
    const int lo = L.full ? 0 : std::max(0, P - 1 - g.window);
    const int cnt = P - lo;
    std::fill(g.a.begin(), g.a.begin() + static_cast<ptrdiff_t>(g.heads) * hd, 0.0f);
    if (cnt <= 0) return;
    // a head per task (E2B: one KV head, the drafter's four query heads all read it)
    const auto head = [&](auto tag, int h) {
        using T = decltype(tag);
        thread_local std::vector<float> s;
        s.resize(static_cast<size_t>(cnt));
        const float *q = g.q.data() + static_cast<size_t>(h) * hd;
        float *out = g.a.data() + static_cast<size_t>(h) * hd;
        const int kh = h * kv_heads_ / g.heads;
        float mx = -INFINITY;
        for (int p = lo; p < P; ++p) {
            const T *k = reinterpret_cast<const T *>(kv_row_bytes(p, L.slot, false)) + static_cast<size_t>(kh) * hd;
            float v = dot(q, k, hd);
            if constexpr (std::is_same_v<T, int8_t>) v *= kv_scale(p, L.slot, false)[kh];
            s[static_cast<size_t>(p - lo)] = v;
            mx = std::max(mx, v);
        }
        float sum = 0.0f;
        for (float &v : s) sum += (v = std::exp(v - mx));
        const float inv = 1.0f / sum;
        for (int p = lo; p < P; ++p) {
            const T *v = reinterpret_cast<const T *>(kv_row_bytes(p, L.slot, true)) + static_cast<size_t>(kh) * hd;
            float w = s[static_cast<size_t>(p - lo)] * inv;
            if constexpr (std::is_same_v<T, int8_t>) w *= kv_scale(p, L.slot, true)[kh];
            axpy(w, v, out, hd);
        }
    };
    par(g.heads, [&](int64_t a, int64_t b) {
        for (int64_t h = a; h < b; ++h) {
            if (exact_) head(float{}, static_cast<int>(h));
            else if (kv_i8_) head(int8_t{}, static_cast<int>(h));
            else head(_Float16{}, static_cast<int>(h));
        }
    });
}

bool Model::gemma_draft(int tok, const float *hidden, int &draft, float *prob, std::string &err) {
    const auto &c = cfg_;
    GDraft &g = gd_;
    const int H = c.hidden, D = g.hidden;
    if (g.pos <= 0 || g.pos > pos_) {
        err = "a draft at position " + std::to_string(g.pos) + " (the model is at " + std::to_string(pos_) + ")";
        return false;
    }
    if (!pack_.embedding(tok, g.in.data())) {
        err = "cannot read the embedding of token " + std::to_string(tok);
        return false;
    }
    for (int i = 0; i < H; ++i) g.in[static_cast<size_t>(i)] *= c.embed_scale;
    std::memmove(g.in.data() + H, hidden, static_cast<size_t>(H) * sizeof(float));   // (hidden may be g.hp)
    const auto mm = [&](const QView &m, const float *x, float *y) {
        const float *xs[1] = {x};
        float *ys[1] = {y};
        matmul(m, 1, xs, ys);
    };
    float *x = g.x.data(), *h = g.h.data(), *f = g.f.data(), *n = g.n.data();
    mm(g.pre, g.in.data(), x);
    for (const GDraftLayer &L : g.layers) {
        const int hd = L.hd, half = hd / 2;
        rmsnorm(x, L.in_norm, h, D, g.eps, false);
        mm(L.q, h, g.q.data());
        // the query heads normed and turned at the round's position (the main model's rotary for this type): the
        // angles' cosines and sines once for all the heads (the same values: they were the same in every head)
        const float p = static_cast<float>(g.pos);
        thread_local std::vector<float> rc, rs;
        rc.resize(static_cast<size_t>(L.pairs));
        rs.resize(static_cast<size_t>(L.pairs));
        for (int i = 0; i < L.pairs; ++i) {
            const float a = p * L.inv_freq[static_cast<size_t>(i)];
            rc[static_cast<size_t>(i)] = std::cos(a);
            rs[static_cast<size_t>(i)] = std::sin(a);
        }
        for (int k = 0; k < g.heads; ++k) {
            float *q = g.q.data() + static_cast<size_t>(k) * hd;
            rmsnorm(q, L.q_norm, q, hd, g.eps, false);
            for (int i = 0; i < L.pairs; ++i) {
                const float cs = rc[static_cast<size_t>(i)], sn = rs[static_cast<size_t>(i)];
                const float x1 = q[i], x2 = q[i + half];
                q[i] = x1 * cs - x2 * sn;
                q[i + half] = x2 * cs + x1 * sn;
            }
        }
        draft_attend(L);
        mm(L.o, g.a.data(), f);
        rmsnorm(f, L.attn_norm, f, D, g.eps, false);
        for (int i = 0; i < D; ++i) x[i] += f[i];
        rmsnorm(x, L.ff_norm, n, D, g.eps, false);
        {   // gate and up from one quantization of their input (each matmul made its own, the same), in one round
            std::vector<float *> in1{n};
            const QAct *const *qn = quantize_h(in1, D);
            const float *xs[1] = {n};
            float *yg[1] = {g.g.data()}, *yu[1] = {g.u.data()};
            const QView *mats[2] = {&L.gate, &L.up};
            float *const *outs[2] = {yg, yu};
            matmul_many(mats, 2, 1, xs, outs, qn);
        }
        for (int i = 0; i < g.ff; ++i) g.g[static_cast<size_t>(i)] = gelu_tanh(g.g[static_cast<size_t>(i)]) * g.u[static_cast<size_t>(i)];
        mm(L.down, g.g.data(), f);
        rmsnorm(f, L.ff_post_norm, f, D, g.eps, false);
        for (int i = 0; i < D; ++i) x[i] = (x[i] + f[i]) * L.scalar;
    }
    rmsnorm(x, g.norm, n, D, g.eps, false);
    mm(g.post, n, g.hp.data());
    // the top clusters, then their tokens' logits: each cluster's rows of the head in a row
    mm(g.cent, n, g.cl.data());
    std::vector<int> &top = g.top;
    top.resize(static_cast<size_t>(g.centroids));
    for (int i = 0; i < g.centroids; ++i) top[static_cast<size_t>(i)] = i;
    std::partial_sort(top.begin(), top.begin() + g.top_k, top.end(), [&](int a, int b) {
        const float va = g.cl[static_cast<size_t>(a)], vb = g.cl[static_cast<size_t>(b)];
        return va > vb || (va == vb && a < b);
    });
    std::vector<float *> one{n};
    const QAct *const *qn = quantize_h(one, D);
    const auto cluster = [&](int k) {
        QView v = g.head;
        v.rows = g.per;
        v.data = g.head.data + static_cast<size_t>(top[static_cast<size_t>(k)]) * g.per * g.head.row_bytes;
        return v;
    };
    if (!exact_ && !check_ && !gpu_ && qn) {
        // every cluster's rows in one round of the pool (a matmul each was top_k rounds for a few hundred rows
        // apiece); each row's kernel the same, so the same logits
        const int64_t per = g.per, total = static_cast<int64_t>(g.top_k) * per;
        pool_->parallel_for(total, [&](int64_t a, int64_t b) {
            while (a < b) {
                const int k = static_cast<int>(a / per);
                const int64_t r0 = a % per, r1 = std::min(per, r0 + (b - a));
                const QView v = cluster(k);
                float *ys[1] = {g.lg.data() + static_cast<size_t>(k) * g.per};
                matvec_rows_q_multi(v, qn, 1, ys, r0, r1);
                a += r1 - r0;
            }
        }, 16);
    } else {
        for (int k = 0; k < g.top_k; ++k) {
            const QView v = cluster(k);
            const float *xs[1] = {n};
            float *ys[1] = {g.lg.data() + static_cast<size_t>(k) * g.per};
            matmul(v, 1, xs, ys, qn);
        }
    }
    // the likeliest candidate that is no media token (one would take an image's or a clip's features)
    const int pads[3] = {vision_.available() ? vision_.shape().pad_token : -1,
                         audio_.available() ? audio_.shape().pad_token : -1,
                         vision_.available() ? vision_.shape().video_pad : -1};
    const int total = g.top_k * g.per;
    int best = -1;
    float mx = -INFINITY;
    for (int i = 0; i < total; ++i) {
        const int t = g.order[static_cast<size_t>(top[static_cast<size_t>(i / g.per)]) * g.per + i % g.per];
        if (t == pads[0] || t == pads[1] || t == pads[2]) {
            g.lg[static_cast<size_t>(i)] = -INFINITY;
            continue;
        }
        if (g.lg[static_cast<size_t>(i)] > mx) {
            mx = g.lg[static_cast<size_t>(i)];
            best = i;
        }
    }
    if (best < 0) {
        err = "the drafter has no candidate";
        return false;
    }
    draft = g.order[static_cast<size_t>(top[static_cast<size_t>(best / g.per)]) * g.per + best % g.per];
    if (prob) {
        double sum = 0.0;
        for (int i = 0; i < total; ++i) sum += std::exp(static_cast<double>(g.lg[static_cast<size_t>(i)] - mx));
        *prob = static_cast<float>(1.0 / sum);
    }
    return true;
}

}  // namespace geode
