// yue2.hpp - YuE2-3B (m-a-p/YuE2-3B) on the CPU in fp32: phase 2's reference engine. The MoT transformer's AR path
// (a causal LM with a KV cache: the ABC score, then the semantic codes) and its NAR path (flow matching's velocity
// over the acoustic latents, attending to the AR prefix's cache), the sampler and the 32-step midpoint ODE - every
// step as the Python reference (the yue2 package: modeling_yue2.py, nar.py, sampling.py) computes it, on the
// checkpoint's bf16 weights (tools/pack_yue2.py).
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "threads.hpp"
#include "yue2_pack.hpp"

namespace geode::yue2 {

struct Sampling {   // yue2/protocol.py Sampling
    float temperature = 1.0f, top_p = 1.0f;
    int top_k = 1;
    float repetition_penalty = 1.0f;
    int penalty_window = 1, min_tokens = 0, max_tokens = 1;
};

struct Tokens {     // yue2/protocol.py
    int eod = 0, abc_start = 0, abc_end = 0, music_start = 0, music_end = 0, codec_offset = 0, codec_size = 0,
        context = 0;
};

struct Config {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0, vocab = 0, max_pos = 0;
    int latent = 0, max_latent_frames = 0, freq = 256;
    float eps = 1e-6f;
    double shift = 1.0;
    std::vector<float> inv_freq;   // RoPE, head_dim / 2 (the reference device's own values, geode.json)
    Tokens tok;
    Sampling abc, semantic;
    int ode_steps = 32;
};
bool read_config(const Pack &pack, Config &c, std::string &err);

// Y[n][out] = X[n][in] W[out][in]^T (+ bias[out]): W bf16, X / Y fp32 rows of in / out floats.
void gemm_bf16(const float *x, int64_t n, int64_t in, const uint16_t *w, int64_t out, const float *bias, float *y,
               ThreadPool &pool);

// One AR branch's keys (after k_norm and RoPE) and values: [layer][position][kv_heads * head_dim].
struct KvCache {
    int len = 0, cap = 0;
    std::vector<float> k, v;
};

class Model {
public:
    bool open(Pack &pack, std::string &err);   // loads ar.bin and nar.bin
    const Config &config() const { return c_; }

    // capacity: the branch's prefix + generation budget, at most the context (24576)
    bool cache_init(KvCache &c, int capacity, std::string &err) const;
    // n tokens at positions c.len .. c.len + n - 1 through the AR path, their K / V appended to c. hidden (if not
    // null): the final-normed hidden state of the tokens at the indices `rows` (0 .. n - 1), [rows.size()][hidden].
    bool ar_forward(const int32_t *tokens, int n, KvCache &c, const std::vector<int> &rows, float *hidden,
                    ThreadPool &pool, std::string &err) const;
    // lm_head's rows [lo, hi) for n hidden states: logits [n][hi - lo]. false: a range outside the vocabulary.
    bool logits(const float *hidden, int n, int lo, int hi, float *out, ThreadPool &pool) const;

    // The NAR over one chunk: `ar` holds the chunk's AR tokens (prefix + codes + MUSIC_END), every one visible.
    struct Nar {
        const KvCache *ar = nullptr;
        int frames = 0, tokens = 0;          // tokens = frames + 2 (the START / END pads)
        std::vector<float> cos, sin;         // RoPE at positions ar->len .. ar->len + tokens - 1: [tokens][hd / 2]
    };
    // ar: ONE chunk's own sequence (the song's prefix + this chunk's codes + MUSIC_END, positions from 0 - nar.py
    // song_chunks), never the generation cache of a longer song; ar.len + frames + 2 <= the context.
    bool nar_init(Nar &nar, const KvCache &ar, int frames, std::string &err) const;
    struct Trace {                           // velocity's insides, for the parity tool (optional)
        float t_shifted = 0.0f;
        std::vector<float> x_in;             // [tokens][hidden]: vae2llm + time + positions
        std::map<int, std::vector<float>> x_after;   // the layers asked for (keys preset): the stream after them
    };
    // CachedNAR.velocity: state [frames][latent] at raw_t = logit(t) clamped -> v [frames][latent].
    void velocity(const Nar &nar, const float *state, double raw_t, float *v, ThreadPool &pool,
                  Trace *trace = nullptr) const;
    // CachedNAR.solve: `steps` midpoint steps from t = 1 to 0, from noise [frames][latent] -> out [frames][latent].
    // false: steps < 1, or non-finite latents (nar.py raises on both).
    bool solve(const Nar &nar, const float *noise, int steps, float *out, ThreadPool &pool, std::string &err,
               const std::function<void(int, int)> &progress = nullptr) const;

private:
    struct Attn {
        const uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
        const float *q_norm = nullptr, *k_norm = nullptr;
    };
    struct Mlp { const uint16_t *gate = nullptr, *up = nullptr, *down = nullptr; };
    struct Path { const float *in_norm = nullptr, *mlp_norm = nullptr; Attn attn; Mlp mlp; };
    struct Layer { Path ar, nar; };
    // one transformer layer of a path over n rows of x (in place); positions: cos / sin rows [n][hd / 2]; keys: the
    // first `prior` from (pk, pv) [prior][kv * hd], then the rows' own. causal: row i sees prior + i + 1 keys.
    // new_k / new_v (optional): the rows' own keys / values out [n][kv * hd].
    void layer(const Path &p, float *x, int n, const float *cos, const float *sin, const float *pk, const float *pv,
               int prior, bool causal, float *new_k, float *new_v, ThreadPool &pool) const;
    void rope_rows(int pos0, int n, std::vector<float> &cos, std::vector<float> &sin) const;

    Pack *pack_ = nullptr;
    Config c_;
    std::vector<Layer> layers_;
    const float *norm_ = nullptr;
    const uint16_t *lm_head_ = nullptr;
    const TensorInfo *embed_ = nullptr;
    const uint16_t *vae2llm_w_ = nullptr, *llm2vae_w_ = nullptr, *t0_w_ = nullptr, *t2_w_ = nullptr, *pe_ = nullptr;
    const float *vae2llm_b_ = nullptr, *llm2vae_b_ = nullptr, *t0_b_ = nullptr, *t2_b_ = nullptr;
};

// yue2/sampling.py distribution() for one step: fp32 logits over the whole vocabulary -> the sampling weights
// (softmax of the processed scores; zero where masked). phase_abc: ABC (text ids < EOD, end ABC_END) else semantic
// (the codes, end MUSIC_END); step: the tokens emitted so far; history: their ids.
// cot "full" / "melody" only: cot "off" (legacy_off) keeps 3 top-p candidates and its scores in bf16 - not here.
// Top-p sums the probabilities in double where torch sums fp32: a token exactly at the cutoff may differ (noise).
void distribution(const float *logits, int vocab, const Sampling &s, const Tokens &t, bool phase_abc, int step,
                  const std::vector<int> &history, std::vector<float> &probs);

}  // namespace geode::yue2
