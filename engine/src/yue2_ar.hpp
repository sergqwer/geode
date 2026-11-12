// yue2_ar.hpp - YuE2-3B's AR path for generation on the phone (phase 3): the matrices in Q8 (ar_q8.bin,
// tools/pack_yue2.py --ar-q8) on the engine's int8 SDOT kernels (kernels.hpp matvec_rows_q_multi), a KV cache per
// branch in int8 (a scale per position and KV head), fp16 or fp32. Every segment of a call goes through the matrices
// together - CFG's two branches read the weights once a token; attention is per segment over its own branch.
// The arithmetic is yue2.cpp's oracle (RMSNorm, q_norm / k_norm, RoPE at the reference device's frequencies, SwiGLU)
// with the products quantized; geode-yue2gen measures what that costs against the fp32 reference's logits.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "platform.hpp"
#include "threads.hpp"
#include "yue2.hpp"
#include "yue2_pack.hpp"

namespace geode::yue2 {

enum class KvType { I8, F16, F32 };
const char *kv_type_name(KvType t);
bool kv_type_from_name(const std::string &s, KvType &t);

struct ArOptions {
    KvType kv = KvType::I8;
    // false: the activations stay fp32 against the dequantized weights (matvec_rows): the ablation that separates
    // the weights' rounding from the activations' (slow - not for the phone)
    bool act_q8 = true;
    // ar_q8.bin mapped (Pack::map) rather than copied into memory: a phone with less free memory than the file
    bool map = false;
    // ar_q4.bin instead (tools/pack_yue2.py --ar-q4): the layers' matrices in Q4 - about half the bytes a token reads,
    // so a faster generation, at Q4's rounding (lm_head stays Q8). Loaded, never mapped (its rows turned into the
    // quad layout in memory).
    bool q4 = false;
};

// One sequence's keys and values: [layer][position][kv_heads * head_dim] of the KV type, int8's scales
// [layer][position][kv_heads] (keys, values).
struct ArBranch {
    int len = 0, cap = 0;
    KvType type = KvType::I8;
    AlignedBuffer k, v;
    size_t kv_bytes = 0;   // each of k, v
    std::vector<float> ks, vs;
    size_t bytes() const;
};

class FastAr {
public:
    bool open(Pack &pack, const ArOptions &opt, std::string &err);   // reads ar_q8.bin (ar_q4.bin) into memory
    const char *file() const { return opt_.q4 ? "ar_q4.bin" : "ar_q8.bin"; }   // (what to unload when done)
    static bool has_q4(const Pack &pack) { return pack.find("q4.lm_head.weight") != nullptr; }
    const Config &config() const { return c_; }
    const ArOptions &options() const { return opt_; }

    // capacity: the branch's prefix + its generation budget, at most the context (24576)
    bool branch_init(ArBranch &b, int capacity, std::string &err) const;
    // a branch of another KV type than the options' (the NAR's prefix in fp16 beside int8 generation); a run's
    // batch takes the int8 attention only when every row's branch is int8
    bool branch_init(ArBranch &b, int capacity, std::string &err, KvType type) const;
    // positions [0, n) of src into dst (its capacity may differ; the same KV type), dst.len = n: the bytes a run of
    // the same tokens would write (takes of one song share their prompt)
    bool branch_copy(const ArBranch &src, ArBranch &dst, int n) const;

    // n tokens appended to branch b (at positions b.len .. b.len + n - 1). rows / hidden (optional): the final-normed
    // hidden states of the segment's tokens at those indices (0 .. n - 1) into hidden [rows.size()][hidden].
    struct Seg {
        ArBranch *b = nullptr;
        const int32_t *tokens = nullptr;
        int n = 0;
        std::vector<int> rows;
        float *hidden = nullptr;
    };
    // The segments' tokens through the AR path; no two segments may share a branch. Long segments go through in
    // pieces of kChunk tokens (a prompt); one token each (a CFG step) goes through as one batch.
    bool run(std::vector<Seg> &segs, ThreadPool &pool, std::string &err);
    // polled before every piece of a run (a long prompt): true stops it (run returns false, err "cancelled")
    std::function<bool()> cancelled;
    static constexpr int kChunk = 128;

    // Positions [at, at + n) of layer l from outside - a prompt's keys and values computed on the DSP
    // (NarDsp::ar_prompt): k, v [n][kv_heads * head_dim] floats after k_norm and RoPE, stored as run() stores its own
    // (int8: amax / 127 a position and KV head). The caller sets b.len once every layer has them.
    bool put_kv(ArBranch &b, int l, int at, int n, const float *k, const float *v, ThreadPool &pool) const;

    // lm_head's rows [lo, hi) for n hidden states: out [n][hi - lo] (the hidden rows quantized as every product's).
    bool logits(const float *hidden, int n, int lo, int hi, float *out, ThreadPool &pool) const;
    // the embedding's rows [lo, hi) kept in memory from now on (the codes' 32768 rows, 128 MB, in one read): each step
    // reads a row, and in a song - the page cache churned by the DSP's weights - a row from the flash cost a step
    // 1-2 ms. false: not kept (an unloaded pack, a range too wide, no memory); the rows are read as before.
    bool cache_embed(int lo, int hi);

private:
    struct Layer {
        const float *in_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *mlp_norm = nullptr;
        QView q, k, v, o, gate, up, down;
    };
    // a batch: rows from the segments' pieces, each row's branch and position
    struct Row {
        ArBranch *b;
        int pos;
    };
    void batch(const std::vector<Row> &rows, const int32_t *tokens, std::vector<std::pair<int, float *>> &outs,
               ThreadPool &pool);
    // y[t][w.rows] = W x[t] for the n rows: x quantized (act_q8) into acts_ first unless `quantized` (the same x as
    // the call before: k / v after q, up after gate), else x in fp32 against the dequantized weights
    void product(const QView &w, const float *x, int n, float *y, ThreadPool &pool, bool quantized = false) const;
    struct Part {
        const QView *w;
        float *y;
    };
    // several products of the same x in one pass of the pool (np <= 4); swiglu: {gate, up} -> gate = silu(gate) * up
    void products(const Part *parts, int np, const float *x, int n, ThreadPool &pool, bool swiglu = false) const;
    void attention(const std::vector<Row> &rows, int layer, const float *q, float *out, ThreadPool &pool);

    Pack *pack_ = nullptr;
    Config c_;
    ArOptions opt_;
    std::vector<Layer> layers_;
    const float *norm_ = nullptr;
    QView lm_head_;
    const TensorInfo *embed_ = nullptr;
    int decode_width_ = 1 << 20;   // threads for a token's step (GEODE_YUE2_DECODE_WIDTH; default: all)
    // scratch, kept between calls
    mutable std::vector<QAct> acts_;
    std::vector<float> x_, h_, q_, kn_, vn_, att_, tmp_, g_, u_, cos_, sin_, parts_;
    std::vector<int8_t> q8_;   // the queries in int8 (the int8 cache's SDOT scores), qs_ their scales
    std::vector<float> qs_;
    std::vector<uint16_t> emb_cache_;   // cache_embed's rows [emb_lo_, emb_hi_)
    int emb_lo_ = 0, emb_hi_ = 0;
    double prof_[8] = {};   // GEODE_YUE2_AR_PROF's sums
    int prof_n_ = 0;
};

}  // namespace geode::yue2
