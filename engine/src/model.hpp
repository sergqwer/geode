// model.hpp - Qwen3.6's forward pass over a geode-pack: 30 Gated DeltaNet layers and 10 gated full-attention layers,
// each followed by a 256-expert MoE with a shared expert. Tokens go through in batches (a prompt; one token when
// generating): every weight row is read once per batch, every routed expert computed once for all the batch's tokens
// that picked it. The routed experts come from flash through the expert store (experts.hpp).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "experts.hpp"
#include "pack.hpp"
#include "gpu.hpp"
#include "gpu8.hpp"
#include "threads.hpp"
#include "audio.hpp"
#include "vision.hpp"

namespace geode {

// A router's top K (Model::route_one): softmax over the E logits, the K largest, their probabilities renormalised.
// res (one byte an expert: in the expert cache) with lambda > 0: the selection on logit + lambda for the cached ones
// (ties to the lower index), the weights still the plain probabilities renormalised (upstream Strata's route prior);
// moved: how many picks the plain top K would not have made. A free function so tools/route_prior_test.cpp checks it.
void route_topk(const float *logits, int E, int K, int *ids, float *weights, const uint8_t *res, float lambda,
                int *moved);

struct ModelOptions {
    uint64_t memory_budget = 0;   // bytes for everything (dense, context, expert cache); 0 = no limit
    int compute_threads = 0;      // 0 = default_compute_threads()
    int context_length = 0;       // the longest context; 0 = 8192 (the memory budget may allow less)
    int kv_chunk = 0;             // the KV cache grows this many positions at a time; 0 = Model::kKvChunk (tests: less)
    bool exact = false;           // FP32 activations (the parity reference) instead of the int8 kernels
    bool emulate_int8 = false;    // with exact: the activations rounded as the int8 kernels round them (a test)
    bool emulate_bf16 = false;    // with exact: the activations rounded to BF16 (the baseline a GPU runs at)
    bool check_int8 = false;      // without exact: check every int8 product against its exact emulation (slow)
    int io_threads = 0;           // expert readers; 0 = default_io_threads()
    bool direct_io = false;       // O_DIRECT expert reads (the phone)
    bool prefetch = true;         // read the next layer's predicted experts ahead (one token at a time)
    int helper_threads = -1;      // little-core threads that join a prompt's products; -1 = default (none), 0 = none
    // Little-core threads that join a token's products (weighted by their speed, threads.hpp); -1 = GEODE_LITTLE_THREADS
    // or every little core, 0 = none (a token on the big cores alone, as before)
    int little_threads = -1;
    bool gpu = false;             // a prompt's big dense products shared with the GPU (gpu.hpp) where it can
    float gpu_share = 0.0f;       // the GPU's part of their rows; 0 = Model::kGpuShare
    bool quad_layout = true;      // the dense matrices in the GPU layout for the CPU too (faster; kernels.hpp)
    // The KV cache in int8, a scale per position and KV head (half FP16's memory; tools/kl.cpp measures its cost);
    // false or GEODE_KV_FP16=1: FP16. The exact path keeps FP32 either way.
    bool kv_int8 = true;
    // Generating: a routed expert ranked this low or lower (0-based, of top_k) that is neither cached nor on its way
    // is left out - its weight 0, its block never read - instead of waited for (Strata's tail skip). 0 = never;
    // -1 = GEODE_TAIL_SKIP, else never: it drops experts the model picked, and the user's rule is that a token
    // always takes all of them (leaving any out spoils the answers). Measured (tools/kl.cpp, all-q4, the phone's
    // cache, 1200 tokens) with the last quarter of the ranks (6 and 7 of 8): KL 0.008, top token 95.6% the same; on
    // the 8 Elite 14.9 -> 16.3 tokens/s on the GPU (half the flash misses go). GEODE_TAIL_SKIP=6 measures it.
    int tail_skip = -1;
    // Cache-conditioned routing (upstream Strata 0.2.7, route_prior.cu; "Mixture of Cache-Conditional Experts",
    // arXiv 2412.00099): while generating, a cached expert's logit + this for the top-k SELECTION, the weights the
    // router's own probabilities renormalised over the picked. Only when the cache cannot hold every expert, and on
    // the CPU route (the Adreno 7xx+ GPU decode routes on the GPU: unchanged). 0 = off (the default: it changes the
    // answers a little); GEODE_ROUTE_PRIOR overrides.
    float route_prior = 0.0f;
    bool tail_renorm = false;     // with tail_skip: the weights kept renormalised to sum 1 (or GEODE_TAIL_RENORM=1)
    // The routed experts a token takes (Model::set_experts): 1 .. the model's top_k; 0 = top_k
    int experts_per_token = 0;
    // A prompt's dense products on the NPU (npu/host, Hexagon v73+ with an unsigned PD) where it opens and passes its
    // HMX check; anything else (no cDSP, refused, an error) leaves the GPU / CPU path as it was. 1 = on (the app's
    // default), 0 = off. GEODE_NPU overrides either way (0 / 1).
    int npu = 0;
};

// What one layer did for one token (the parity tool compares it with tools/pack.py's reference).
struct LayerTrace {
    int layer = 0, pos = 0;
    const float *x = nullptr;       // the layer's input
    const float *mixer = nullptr;   // the token mixer's output (before the residual add)
    const float *moe = nullptr;     // the MoE's output (before the residual add)
    const float *out = nullptr;     // the layer's output
    const int *experts = nullptr;   // [experts()]
    const float *weights = nullptr; // [experts()], renormalised
    // the CPU path's parts of the MoE (nullptr on the GPU's): each pick's expert output [experts()][hidden], the
    // shared expert's output before its gate, and the gate (sigmoid): moe = gate * shared + sum weights * routed
    const float *routed = nullptr;
    const float *shared = nullptr;
    float shared_gate = 0.0f;
};

class Model {
public:
    // (the expert cache's readers stopped first: the NPU's share of its slabs goes with npu_, before store_)
    ~Model();
    // Tokens per forward_batch. A prompt goes in batches of up to kMaxBatch: each routed expert is read once per
    // batch, and from ~128 tokens on a batch needs nearly every expert of every layer, so the bigger the batch the
    // fewer times the flash is read through. Scratch for more than kSmallBatch tokens is lent by the expert cache
    // while the prompt is read; trim_batch gives it back.
    static constexpr int kMaxBatch = 1024;
    static constexpr int kSmallBatch = 64;
    // A prompt of up to kMaxPrompt tokens goes through forward_batch at once: past what the cache can lend scratch
    // for, it runs each layer in sub-batches of that size, all of them before the next layer - every routed expert
    // read once for the whole prompt instead of once per batch (the phone read ~14 GB of experts per 500 tokens).
    static constexpr int kMaxPrompt = 4096;
    // from this many tokens a batch reads the next layer's experts ahead, all of them (it needs ~98% anyway), and
    // the little cores compute with the big ones
    static constexpr int kLayerPrefetch = 128;
    // up to this many tokens (generating) the routed experts are spread over the threads, one expert per thread
    static constexpr int kSpreadTokens = 4;
    // The threads that generate (ThreadPool::set_width) and, of them, those that spread a token's routed experts:
    // all the big cores, and three quarters of them (0 = these defaults; GEODE_DECODE_THREADS / GEODE_SPREAD_THREADS
    // set them for measuring). The 8 Elite with its experts in RAM (98% hits, 10 GB budget), decode / spread threads:
    // 4/4 10.8, 8/4 13.3, 8/8 14.2, 8/6 15.4 tokens/s. With a 4 GB budget the flash ruled and 4 threads did best
    // (11.3 vs 9.1): threads waiting for reads only heat the cores.
    int decode_threads_ = 0, spread_threads_ = 0;
    // With decode_threads_ 0 a token goes on as many threads as the system gives cores (token_width): the 8 Elite
    // outside its performance mode parks its two prime cores (core_ctl) and 8 threads on the other 6 wait for each
    // other - a thread taken off its core in the middle of a chunk holds the product up (Qwen3.5-2B 22 -> 36
    // tokens/s on 6). Every kWidthWindow token passes the cores the threads ran on (ThreadPool::take_cpus) set the
    // width, after two windows that saw fewer; every kWidthProbe windows all the threads again, to see whether
    // more cores came (the performance mode turned on).
    static constexpr int kWidthWindow = 16, kWidthProbe = 128;
    int width_auto_ = 0, width_passes_ = 0, width_low_ = 0, width_low_seen_ = 0, width_windows_ = 0;
    int token_width();
    // And the little cores' threads that join them (ModelOptions::little_threads), weighted by their speed; a token's
    // routed experts then split by rows in two products (the gate and up rows of all its experts with their
    // activations, then the down rows), not one expert a thread - with 4 experts a layer (k = 4) and 8 threads the
    // experts alone would leave half of them idle and an A55 holding up three A76-thirds of a layer
    // (GEODE_ROUTED_ROWS=0: one expert a thread, as before)
    int little_threads_ = 0;
    // the little threads in a token's dense products too, not only its routed experts' (GEODE_LITTLE_DENSE=0: those
    // only). On the Redmi the dense products already move as many bytes as the RAM gives two A76 (12.6 of 12.8 GB/s),
    // yet the routed experts alone with the A55 came out no faster (4.12 vs 4.21 tokens/s): their weight, learned
    // there, takes the flash reads they wait for as slowness (0.07)
    bool little_dense_ = true;
    bool routed_rows_ = true;
    int spread_threads() const {
        const int w = pool_->width();
        return std::min(w, spread_threads_ > 0 ? spread_threads_ : w - w / 4);
    }
    // The KV cache grows by this many positions at a time; each step takes its memory from the expert cache, which
    // gives back its coldest experts (a new conversation returns them).
    static constexpr int kKvChunk = 1024;

    bool open(const std::string &dir, const ModelOptions &opt, std::string &err);

    // n (<= kMaxPrompt; all_logits and trace: <= kMaxBatch) tokens at the next positions. logits: the last token's
    // [vocab], or with all_logits every token's [n][vocab]; nullptr for none. False (with err) on a read error, or
    // when set_stop's flag went up (a half-read prompt: reset()). Afterwards trim_batch() gives the scratch back.
    bool forward_batch(const int *tokens, int n, float *logits, bool all_logits, std::string &err);
    bool forward(int token, float *logits, std::string &err) { return forward_batch(&token, 1, logits, false, err); }

    // The MTP layer (a pack with one, tools/pack_mtp.py) drafts the token after next. Position i takes the main
    // model's hidden state of token i and the embedding of token i + 1, and its logits draft token i + 2. n of them
    // at positions pos..pos+n-1 (in order: each attends to the MTP layer's own earlier positions); logits: the last
    // one's [vocab], or nullptr. hidden(t) is token t of the last forward_batch after the final norm (vLLM's
    // convention; the state before the norm, or positions one further, drafted the same in geode-mtp).
    bool has_mtp() const { return mtp_; }
    bool forward_mtp(const int *next_tokens, const float *const *hidden, int n, int pos, float *logits, std::string &err,
                     const QView *head = nullptr);   // head: the logits' rows (a draft head), lm_head by default
    // After forward_mtp with logits: its last position's output after the layer's final norm - what its logits come
    // from, and what a chained draft takes as the next hidden state (vLLM's / sglang's multi-step MTP convention).
    // Valid until the next pass.
    const float *mtp_hidden() const { return h_.data(); }
    const float *hidden(int t) const {
        return (hid_long_.empty() ? hid_ : hid_long_).data() + static_cast<size_t>(t) * cfg_.hidden;
    }
    // what the threads do (profile): the big and little threads a token runs on, the little ones' weight
    std::string threads_status() const;
    // a flag forward_batch checks between layers (geode_stop): a long prompt stops within a layer's time
    void set_stop(const std::atomic<bool> *flag) { stop_ = flag; }

    // A batch of n tokens at once on the GPU (drafted tokens verified; 2 .. spec_tokens()): toks[0] at the next
    // position, the others after it. logits [n][vocab]: each token's, bit for bit what forward() gives them one after
    // the other (the same routed experts: tail_skip decides by the cache, which differs). The position moves on by n,
    // hidden(0) .. hidden(n - 1) are theirs; verify_keep(kept) takes the last n - kept back (the states as after token
    // kept - 1: the backups the batch left; their KV rows are written over), verify_reject() = keep the first only.
    bool can_verify() const { return gd_.on || verify8_ || cpu_spec_; }
    // the most tokens a verified batch takes: GEODE_SPEC_K (2 .. Gpu8::kMaxTokens; 2, pairs, by default - more only
    // on the GPU, whose programs and state backups open() makes for that many)
    int spec_tokens() const { return gd_.on ? gd_.k : verify8_ && dec8_ ? spec_t_ : 2; }
    // the threads a product runs on (the pool's workers and the caller)
    int compute_threads() const { return pool_ ? pool_->size() : 1; }
    // testing: one token at a time on the CPU (false) or the GPU again (true, when it could)
    void set_gpu_decode(bool on) { dec8_ = on && dec8_ok_; }
    bool verify(const int *toks, float *logits, std::string &err) { return verify_n(toks, 2, logits, err); }
    bool verify_n(const int *toks, int n, float *logits, std::string &err);
    bool verify_reject(std::string &err) { return verify_keep(1, err); }
    bool verify_keep(int kept, std::string &err);
    // The MTP layer on the GPU: positions pos .. pos + n - 1 (n = 1 .. spec_tokens()), each with the main model's
    // hidden state there (hidden: n rows; nullptr: the last GPU pass's, rows 0 .. n - 1) and the token after it
    // (toks); draft: its likeliest next token after the last one (the token after toks[n - 1]'s), prob: that token's
    // softmax probability among the draft head's rows (its confidence; nullptr: not wanted). Fills the MTP layer's
    // KV there. (on the CPU too, cpu_spec_: n <= 2, the MTP layer drafts there and the pair goes through one CPU
    // batch - the dense weights read once for both tokens; GEODE_CPU_SPEC=0: not)
    bool can_draft() const { return gd_.on || (mtp8_ && verify8_) || cpu_spec_ || npu_spec_; }
    bool speculates_on_cpu() const { return gd_.on || (cpu_spec_ && !dec8_); }
    // Gemma 4's MTP drafter (drafter.cpp; a pack with drafter.bin, tools/pack_gemma4_mtp.py): draft() / draft_next()
    // through it, verify_n() / verify_keep() through the main model's own batches (no state to take back but the
    // position: no recurrence, and rejected KV rows are written over)
    bool has_drafter() const { return gd_.on; }
    const std::string &drafter_status() const { return drafter_status_; }
    // whether the passes keep their hidden rows (hidden(), last_hidden()): an MTP layer or a drafter drafts from them
    bool keeps_hidden() const { return mtp_ || gd_.on; }
    bool draft(const int *toks, int n, int pos, const float *hidden, int &draft, std::string &err, float *prob = nullptr);
    // A chained draft (the GPU): the MTP layer at position pos again, from the token drafted last (tok) and the MTP
    // layer's own normed output there - vLLM's / sglang's multi-step MTP. After draft() or draft_next().
    bool draft_next(int tok, int pos, int &draft, float *prob, std::string &err);
    // What the expert cache learned (ExpertStore::save_profile) into the pack's cache_profile.bin, read back at
    // open() - the cache starts warm (GEODE_NO_WARM=1: not read). False when it cannot be written.
    bool save_cache_profile();
    // the hidden state (after the final norm, hidden floats) of the last position the model went through; nullptr
    // when unknown (no MTP layer, or nothing yet)
    const float *last_hidden() const { return pos_ > 0 && !last_hid_.empty() ? last_hid_.data() : nullptr; }
    void reset();

    // The conversation's whole state in a file and back (state.cpp; geode.h, geode_state_save / geode_state_load):
    // the position, the GDN states, the KV cache's rows (the MTP layer's too), the last hidden state, the images'
    // rotary positions, and extra (a value the caller keeps with it: geode.cpp's pending token). Between forward
    // passes only. append: path holds an earlier state of this conversation (the same since the last reset or
    // load_state): only what grew is written, its header last; anything else is written whole (path.tmp, renamed).
    // A load replaces the conversation (as reset() and then the saved one); on failure it is empty (reset()).
    bool save_state(const std::string &path, bool append, int32_t extra, std::string &err);
    bool load_state(const std::string &path, int32_t &extra, std::string &err);

    // The reserve (geode.h, geode_reserve_memory): bytes > 0 - the expert cache gives back at least that much memory
    // now, really (the CPU path: its coldest slots' pages released; the GPU path: whole pieces of its tail freed to
    // the driver), and keeps it out across reset(), load_state() and context growth; 0 - it is over, the slots are
    // the cache's again (empty, read on demand). Between forward passes. Returns the bytes the reserve holds now; err
    // when that is less than asked (a small cache keeps what a layer pins) or the memory could not be taken back.
    uint64_t reserve_memory(uint64_t bytes, std::string &err);
    uint64_t reserved_bytes() const;

    // The routed experts each token takes (geode.h, geode_set_experts): the router's softmax over every expert, its
    // top k, their weights renormalised over those k (Qwen3.6's top_k convention, for any k); the shared expert as
    // ever; the MTP layer's MoE the same. 1 .. config().top_k. Fewer experts, fewer of them read from flash a token
    // (and computed). Between forward passes: the states and the KV cache do not depend on it, a conversation goes on
    // with another k. The tail skip acts within the k (its last quarter, unless set explicitly); the next layer's
    // guesses are k too (unless GEODE_GUESSES).
    bool set_experts(int k, std::string &err);
    int experts() const { return k_; }

    // Images (vision.hpp), when the pack has a vision encoder. add_image encodes one (rgb: h rows of w pixels, the
    // sides multiples of 32 - vision().fit) for the image_pad tokens to come: the pads forward_batch meets take the
    // images' features in order, tokens of them each, and give them M-RoPE's positions (an image's tokens a row
    // and a column each, the text after it max(rows, columns) further on). Kept until reset().
    bool has_vision() const { return vision_.available(); }
    const Vision &vision() const { return vision_; }
    const std::string &vision_status() const { return vision_status_; }
    // cache: a feature file to read instead of encoding when it holds this image (written after an encode); "" none
    bool add_image(const uint8_t *rgb, int w, int h, const std::string &cache, int &tokens, std::string &err);
    // an image of w x h pixels by its features ([tokens][the text's hidden], as the encoder gives them): the encode
    // add_image does left out (geode-parity: the reference's features)
    bool add_image_features(std::vector<float> feats, int w, int h, int &tokens, std::string &err);
    // a video's frame (Gemma 4): add_image's, its tokens <|video|> - the vision encoder's video_pad (fitted with
    // vision().fit to at most video_soft_tokens, 70); frames, images and clips share the queue
    bool add_video_frame(const uint8_t *rgb, int w, int h, const std::string &cache, int &tokens, std::string &err);
    bool add_frame_features(std::vector<float> feats, int w, int h, int &tokens, std::string &err);
    // Audio clips (audio.hpp), when the pack has an audio encoder: add_audio encodes one (pcm: n samples at 16 kHz in
    // [-1, 1]) for the <|audio|> tokens to come. Images and clips share one queue: a prompt's media tokens take them
    // in the order they were added, each its own kind of token. Kept until reset().
    bool has_audio() const { return audio_.available(); }
    const Audio &audio() const { return audio_; }
    const std::string &audio_status() const { return audio_status_; }
    bool add_audio(const float *pcm, int64_t n, int &tokens, std::string &err);
    // a clip by its features ([tokens][the text's hidden]; geode-parity: the reference's)
    bool add_audio_features(std::vector<float> feats, int &tokens, std::string &err);
    // an image's features into its cache file only (none queued): encoded unless the file holds them already
    bool cache_image(const uint8_t *rgb, int w, int h, const std::string &cache, std::string &err);
    // the image_pad tokens the images added and not yet met take (a prompt must hold exactly these)
    int pending_image_tokens() const;
    // forgets the images added and not met (a prompt that failed before them)
    void drop_pending_images();
    int prompt_batch() const;
    void trim_batch();

    int position() const { return pos_; }
    // the longest context the memory budget allows (at most the one asked for), and how much of it is allocated now
    int context_length() const { return ctx_; }
    int context_allocated() const { return std::min(ctx_, static_cast<int>(kv_ptrs_.size()) * kv_chunk_); }
    const ModelConfig &config() const { return cfg_; }
    uint64_t tail_skipped() const { return tail_skipped_; }   // experts left out (ModelOptions::tail_skip)
    uint64_t memory_bytes() const;
    uint64_t expert_cache_bytes() const { return store_.bytes(); }
    // demand picks: found in the cache, read on the spot, or already on their way thanks to a prefetch
    uint64_t expert_hits() const { return store_.hits(); }
    uint64_t expert_misses() const { return store_.misses(); }
    uint64_t expert_prefetch_hits() const { return store_.prefetch_hits(); }
    uint64_t expert_prefetches() const { return store_.prefetches(); }
    uint64_t flash_bytes() const { return store_.bytes_read(); }
    double int8_check_worst() const { return check_worst_; }
    // the GPU path: "off", or what it holds, or why it is not there
    const std::string &gpu_status() const { return gpu_status_; }
    // the NPU (GEODE_NPU=1): "off", or what it holds and its time, or why it is not there
    std::string npu_status() const;
    // where the time went, summed over every token so far
    std::string profile() const;
    void clear_profile() {
        prof_ = {};
        prof_.hits0 = store_.hits();
        prof_.prefetch_hits0 = store_.prefetch_hits();
        prof_.misses0 = store_.misses();
        prof_.prefetches0 = store_.prefetches();
        prof_.bytes0 = store_.bytes_read();
        prof_.dropped0 = store_.prefetch_dropped();
        prof_.skipped0 = tail_skipped_;
        prof_.prior_picks0 = prior_picks_;
        prof_.prior_moved0 = prior_moved_;
        prof_.class_picks0.clear();
        prof_.class_misses0.clear();
        for (size_t c = 0; c < store_.classes(); ++c) {
            prof_.class_picks0.push_back(store_.class_picks(c));
            prof_.class_misses0.push_back(store_.class_misses(c));
        }
        prof_.gpu0 = gpu_ ? gpu_->busy_seconds() : 0.0;
        prof_.gpu8_busy0 = gpu8_ ? gpu8_->busy_seconds() : 0.0;
        if (pool_) {
            prof_.weighted0 = pool_->stats().weighted;
            prof_.stolen0 = pool_->stats().stolen;
        }
    }

    std::function<void(const LayerTrace &)> trace;

private:
    struct Layer {
        bool attn = false;
        int index = 0;   // among the attention layers, or among the GDN layers
        const float *in_norm = nullptr, *post_norm = nullptr;
        // Gated DeltaNet
        QView qkv, z, out;
        const uint16_t *a = nullptr, *b = nullptr;
        const float *A_log = nullptr, *dt_bias = nullptr, *conv = nullptr, *gnorm = nullptr;
        // gated attention
        QView q, k, v, o;
        const float *q_norm = nullptr, *k_norm = nullptr;
        // MoE
        const uint16_t *router = nullptr, *shared_gate = nullptr;
        QView sh_gate, sh_up, sh_down;
        const float *expert_scales = nullptr;
    };

    bool bind(std::string &err);
    bool bind_layer(Layer &L, const std::string &prefix, bool attn, int index, std::string &err);

    // Gemma 4 (gemma.cpp; arch "gemma4", tools/gemma4.py has the forward pass): its own layers and passes, the rest
    // of the class (threads, the int8 products, the KV cache and its growth, the conversation's state) shared
    struct GLayer {
        bool full = false, own_kv = false;
        int slot = 0;                  // its K / V in the KV cache: its own, or the shared layer's it reads
        int hd = 0, ffn = 0, pairs = 0;
        float scalar = 1.0f;
        const float *in_norm = nullptr, *post_attn = nullptr, *pre_ff = nullptr, *post_ff = nullptr,
                    *ple_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        QView q, k, v, o, gate, up, down, ple_gate, ple_proj;
        std::vector<float> inv_freq;   // the rotary pairs' frequencies
    };
    std::vector<GLayer> glayers_;
    QView ple_model_;                   // per_layer_model_projection [layers * ple_dim, hidden]
    const float *ple_proj_norm_ = nullptr;
    const Tensor *ple_table_ = nullptr; // embed_tokens_per_layer (a row per token, read from flash)
    std::vector<float> ple_, pgate_;    // a chunk's per-layer inputs [n][layers * ple_dim], gate [n][ple_dim]
    std::vector<uint8_t> media_;        // which of a chunk's tokens are an image's (their PLE row: the pad token's)
    // Tokens a pass of the Gemma path takes at most (a prompt goes in chunks of this, each through every layer)
    static constexpr int kGemmaChunk = 256;
    bool bind_gemma(std::string &err);
    bool open_gemma(const ModelOptions &opt, std::string &err);
    bool forward_gemma(const int *tokens, int n, float *logits, bool all_logits, std::string &err);
    // hid_row: where the chunk's final-normed hidden rows go (keeps_hidden(): hid_ or hid_long_ from that row on)
    bool gemma_chunk(const int *tokens, int n, float *logits, bool all_logits, std::string &err, int hid_row = 0);
    // the drafter (drafter.cpp; tools/gemma4_mtp.py is its spec): 4 layers of hidden 256 with no K / V of their own,
    // each attending over the main model's KV of one layer (slot) of its type; a draft round stays at one position
    struct GDraftLayer {
        bool full = false;
        int hd = 0, pairs = 0, slot = 0;
        float scalar = 1.0f;
        const float *in_norm = nullptr, *q_norm = nullptr, *attn_norm = nullptr, *ff_norm = nullptr,
                    *ff_post_norm = nullptr;
        QView q, o, gate, up, down;
        std::vector<float> inv_freq;
    };
    struct GDraft {
        bool on = false;
        int hidden = 0, heads = 0, ff = 0, window = 0, centroids = 0, top_k = 0, per = 0;   // per: a centroid's tokens
        int k = 4;          // the most tokens a verified batch takes (GEODE_SPEC_K: 2 .. Gpu8::kMaxTokens); 4 is
                            // kSpreadTokens, still a token's threads: the best measured on the Redmi
        float eps = 1e-6f;
        AlignedBuffer data; // drafter.bin, whole
        uint64_t bytes = 0;
        QView pre, post, cent, head;
        const float *norm = nullptr;
        std::vector<int32_t> order;       // head row r: token order[r] (each centroid's tokens in a row)
        std::vector<GDraftLayer> layers;
        int pos = -1;                     // the round's position
        std::vector<float> in, x, h, a, q, f, g, u, n, hp, lg, cl;   // scratch; hp: the last step's h'
        std::vector<int> top;             // the centroids by their logits
    };
    GDraft gd_;
    std::string drafter_status_ = "none";
    bool open_drafter(std::string &err);   // after the KV cache exists (open_gemma): its slots
    // one draft step at gd_.pos: the main model's embedding of tok and a hidden row (1536) in -> the draft (the
    // likeliest of the top clusters' tokens, media pads left out) and its probability among them; gd_.hp its h'
    bool gemma_draft(int tok, const float *hidden, int &draft, float *prob, std::string &err);
    void draft_attend(const GDraftLayer &L);   // gd_.q -> gd_.a over the main KV below gd_.pos
    void gemma_attention(const GLayer &L, int n);
    template <typename T>
    void gemma_attend(const GLayer &L, int n);
    template <typename T, int G>
    void gemma_heads(const GLayer &L, int kh, int h0, int t, int b, int nb);
    // ys[t] = W . xs[t] for t < n: each weight row is read once and used for every token
    // pre: the int8 path's activations of xs, already quantized (quantize_h), or nullptr to quantize them here
    void matmul(const QView &m, int n, const float *const *xs, float *const *ys, const QAct *const *pre = nullptr);
    // the layer input's n rows quantized once for every product that reads them (nullptr on the exact path)
    const QAct *const *quantize_h(const std::vector<float *> &hs, int64_t cols);
    void matmul_bf16(const uint16_t *w, int64_t rows, int64_t cols, int n, const float *const *xs, float *const *ys);
    void gdn(const Layer &L, int n);
    void attention(const Layer &L, int n);
    template <typename T>
    void attend(const Layer &L, int n);
    // A token's attention in blocks of this many positions, from position 0 (attend): fixed, not the KV cache's chunk,
    // so a conversation moved between models whose chunks differ keeps its bits (GEODE_ATTN_BLOCK: smaller blocks,
    // for the tests - a short prompt then crosses them)
    static constexpr int kAttnBlock = 1024;
    int attn_block_ = kAttnBlock;
    // token t's heads h0..h0+G-1 (KV head kh) over block b of its positions into attn_parts_
    template <typename T, int G>
    void attend_heads(int layer, int kh, int h0, int t, int b, int nb);
    // a batch's blocks: per token, head and block the values' sum, the scores' maximum and the weights' sum (the
    // tokens from attn_t0_ on: attend takes a long prompt's a few at a time)
    std::vector<float> attn_parts_;
    int attn_t0_ = 0;
    template <typename T>
    T *kv(int pos, int layer, bool value);
    // int8 cache: the scales of a position's KV heads (after the chunk's values)
    float *kv_scale(int pos, int layer, bool value);
    bool kv_i8_ = false;
    bool reserve_context(int positions, std::string &err);
    // keep_guesses: the layer's read-ahead stays pinned (a later sub-batch of the layer may pick what it read)
    // gpu_front: a prompt whose token mixer ran on the GPU - the router (and the shared expert's gate) there too,
    // the shared expert with the routed experts
    bool moe(int l, const Layer &L, int n, std::string &err, bool keep_guesses = false, bool gpu_front = false);
    void prefetch_layer(int l);
    // A prompt's token mixer (Gated DeltaNet or attention) on the GPU (Gpu8::pf_gdn / pf_attn) when generating runs
    // there: 1 done, 0 not taken (the CPU's then), -1 failed after the GPU had begun (err; the states half moved)
    int mixer_gpu(int l, const Layer &L, int n, std::string &err);
    int pf_gpu_ = -1;                 // -1 not decided yet, 0 no (and why: gpu_status_), 1 yes
    std::vector<int32_t> pos3_;       // the tokens' rotary positions for pf_attn
    // A prompt shorter than kGuessAll (measured: 60 tokens 2.82 -> 2.30 s; from ~300 on the router this costs on the
    // CPU eats what the flash saves, and longer prompts need nearly every expert anyway): the next layer's experts its tokens will likely pick (the next layer's router
    // on this layer's input, each token's top kBatchGuesses, ~80% right a pick when generating), read ahead - not
    // all of the layer's (a 300-token prompt picks ~73% of them; the rest were ~2.7 GB of reads for nothing). Each
    // sub-batch adds its own (guessed_ marks what is pinned already).
    static constexpr int kGuessAll = 256;
    static constexpr int kBatchGuesses = 8;
    void prefetch_guess(int l, const std::vector<float *> &hs, int n);
    std::vector<float> guess_;
    std::vector<uint8_t> guessed_;
    int guessed_layer_ = -1;
    int prompt_n_ = 0;   // the tokens of the forward_batch running (a long one reads whole layers ahead)
    bool size_batch(int tokens, std::string &err);
    size_t batch_bytes(int tokens) const;
    // softmax over the router's logits, the top K, renormalised; res (a layer's cache residency, one byte an expert)
    // with lambda > 0: the selection on logit + lambda for the cached ones, the weights unchanged (route_prior)
    void route_one(const float *probs_in, int *ids, float *weights, const uint8_t *res = nullptr, float lambda = 0.0f);
    // route_prior_ applies now: on, generating on the CPU, a cache smaller than the experts; then res_l_ / res_n_ get
    // the residency of layer l (fill_residency)
    bool prior_on() const;
    const uint8_t *fill_residency(int l, std::vector<uint8_t> &buf);
    float route_prior_ = 0.0f;
    std::vector<uint8_t> res_l_, res_n_;
    uint64_t prior_picks_ = 0, prior_moved_ = 0;   // picks routed with the prior, and of them not in the plain top K
    ExpertRead expert_read(int l, int e, DType &type) const;

    // the rows of a [batch_cap_][width] scratch buffer as pointers
    std::vector<float *> rows(std::vector<float> &buf, size_t width, int n, size_t offset = 0);

    Pack pack_;
    ModelConfig cfg_;
    std::unique_ptr<ThreadPool> pool_;
    std::vector<int> big_;
    std::vector<Layer> layers_;
    // the MTP layer: a full-attention layer (its KV after the main ones'), its experts expert layer n_layers
    bool mtp_ = false;
    Layer mtp_layer_;
    QView mtp_fc_;   // [hidden, 2 hidden]: [norm(embedding) | norm(hidden)] -> the layer's input
    const float *mtp_norm_ = nullptr, *mtp_pre_e_ = nullptr, *mtp_pre_h_ = nullptr;
    int apos_ = 0;             // the position of the first token attention() is given
    std::vector<float> hid_;   // [batch_cap_][hidden]: the last forward_batch's hidden states
    QView lm_head_;
    const float *final_norm_ = nullptr;
    int ctx_ = 0, pos_ = 0;

    // the conversation's state
    // per GDN layer: [conv_dim][kernel-1], and per value head the state [k_dim][v_dim] in strips of 16 columns
    // ([v_dim/16][k_dim][16]; gdn() says why), or plain where v_dim is no multiple of 16
    std::vector<float> conv_state_, gdn_state_;
    std::vector<AlignedBuffer> kv_chunks_;        // the KV cache, kv_chunk_ positions each (FP32 exact, else FP16)
    // where the states and the KV chunks in use are: the vectors above, or shared memory when the GPU generates
    float *conv_st_ = nullptr, *gdn_st_ = nullptr;
    size_t conv_n_ = 0, gdn_n_ = 0;
    std::vector<uint8_t *> kv_ptrs_, kv_spare_;   // the chunks in use; shared-memory chunks kept after a reset
    bool new_kv_chunk(std::string &err);
    size_t kv_chunk_bytes_ = 0, kv_row_ = 0, slots_per_chunk_ = 0;
    int kv_heads_ = 0;
    int kv_chunk_ = kKvChunk, n_attn_ = 0;
    std::vector<std::unique_ptr<RandomFile>> expert_files_;
    ExpertStore store_;
    std::vector<std::vector<int>> prefetch_;      // per layer: the slots pinned by the guess made one layer earlier
    bool prefetch_on_ = true;
    int tail_skip_ = 0;           // ModelOptions::tail_skip, resolved
    bool tail_renorm_ = false;
    bool tail_wait_ = true;       // GPU: wait for the tail's picks still being read too (GEODE_TAIL_WAIT=0: no)
    uint64_t tail_skipped_ = 0;   // the experts left out so far
    // one token's picks (ids, weights in rank order): the tail's misses get weight 0 (marked in skip); how many
    int skip_tail(int l, const int *ids, float *w, uint8_t *skip);
    std::vector<float> inv_freq_;
    // the images (add_image): kv0 the KV index of its first pad (-1: not met yet), s the rotary position there,
    // gh x gw its tokens' grid, feats [gh * gw][hidden]
    struct Image {
        int kv0 = -1, s = 0, gh = 0, gw = 0;
        int pad = -1;   // the token its features take the place of: the image pad, or (a clip: 1 x tokens) the audio's
        std::vector<float> feats;
        int n() const { return gh * gw; }
    };
    Vision vision_;
    std::string vision_status_ = "none";
    Audio audio_;
    std::string audio_status_ = "none";
    std::vector<Image> images_;
    size_t image_next_ = 0;   // the first image not wholly in
    int image_fed_ = 0;       // of its pads, the ones in
    int rope_delta_ = 0;      // the text's rotary position minus its KV index (after an image; the GPU's shift too)
    void set_rope_delta(int d);
    // the threads for a batch of n tokens: a prompt on the big ones (and the helpers its options give it), a token
    // (or a pair, or the MTP layer's few positions) on the decode width and the little threads
    void threads_for(int n);
    // A conversation's identity (state.cpp): new at open() and reset(), a saved one's after load_state - an append
    // goes only to a file of the same conversation. The pack's fingerprint: geode.json hashed, computed once.
    uint64_t conv_id_ = 0;
    uint64_t pack_print_ = 0;
    // the experts a token takes (set_experts), the tail skip as asked (-1: the last quarter of k), and whether the
    // GPU's guesses were set from outside (GEODE_GUESSES); apply_experts() passes k on to the tail skip and the GPU
    int k_ = 0, tail_cfg_ = -1;
    bool guesses_env_ = false;
    void apply_experts();
    // the reserve (reserve_memory): the GPU path's freeable tail pieces (mem null: freed), the CPU path's slots taken
    struct TailPiece {
        std::vector<int> slots;
        size_t bytes = 0;
        uint8_t *mem = nullptr;
    };
    std::vector<TailPiece> tail_;
    std::vector<int> reserved_;
    size_t min_slots_ = 0;   // what a layer and its guesses pin at once (the cache never goes below)
    void new_conversation();
    uint64_t fingerprint();
    // the KV cache's bytes of a position's row in one attention layer (K or V), whatever its element type
    uint8_t *kv_row_bytes(int pos, int layer, bool value);
    size_t kv_elem() const { return kv_i8_ ? 1 : exact_ ? sizeof(float) : sizeof(_Float16); }
    // the rotary positions (temporal, row, column streams) of the token at KV index idx
    void rope_pos(int idx, int p[3]) const;
    // the feature row of the image pad at KV index idx (nullptr: none there)
    const float *image_row(int idx) const;

    // scratch, [batch_cap_][...]; more than kSmallBatch rows only while a prompt is read
    int batch_cap_ = 0;
    size_t slot_bytes_ = 0, token_floats_ = 0, max_cols_ = 0;
    uint64_t batch_retired_ = 0;   // the bytes the expert cache lent the batch's scratch
    uint64_t spare_bytes_ = 0;   // the budget beyond the expert cache's full size (all experts fit)
    uint64_t batch_loan(int tokens) const;   // expert-cache bytes a batch of this many tokens borrows (whole slots)
    // the expert cache's size classes (a pack mixing formats within layers), by format: what the profile names them
    std::vector<std::string> class_names_;
    bool budgeted_ = false;
    // A long batch's big products (kWideRows rows and more) use the helper threads too; everything else stays on
    // the compute threads (a helper that takes a GDN head or a small product's last chunk makes the others wait).
    static constexpr int64_t kWideRows = 2048;
    bool wide_ = false;
    void par(int64_t n, const std::function<void(int64_t, int64_t)> &fn, int64_t min_chunk = 1) {
        pool_->parallel_for(n, fn, min_chunk);
    }
    std::vector<float> x_, h_, mix_, moe_, xin_;
    std::vector<float> trace_gate_;              // with trace: each token's shared-expert gate (moe())
    std::vector<float> mtp_in_;                  // forward_mtp's fc input, 2 hidden a position
    const float *trace_routed_ = nullptr, *trace_shared_ = nullptr;   // with trace: where moe() left them
    std::vector<float> long_x_, hid_long_;   // a prompt longer than the batch: its residual stream, its hidden states
    const std::atomic<bool> *stop_ = nullptr;
    std::vector<float> proj_, z_, ab_, core_, qg_, kbuf_, vbuf_;
    std::vector<float> router_, sh_act_, sh_, eact_, eout_;
    std::vector<float> pf_ga_;       // moe_gpu8's CPU share: its picks' gate and up outputs
    std::vector<QAct> pf_mid_;       // and their SwiGLU, quantized
    std::vector<int> top_ids_;
    std::vector<float> top_w_, logits_;
    std::vector<QAct> qacts_;   // the activations of the product in flight, quantized, one per token
    std::vector<QAct> qh_;      // the layer input (h), quantized once for all its products
    std::vector<QAct> rows_qa_; // the routed experts split by rows: each pick's activation between gate-up and down
    std::vector<const QAct *> qh_ptr_;
    std::vector<float> xq_;
    // The GPU (gpu.hpp): the big dense matrices (GDN qkv / z / out, attention q / o) live in its memory in
    // its layout, which the CPU kernels read too; a product of kGpuMinTokens tokens and more gives the GPU the first
    // gpu_share_ of the rows and computes the rest meanwhile. One token stays on the CPU (it streams the weights at
    // DRAM speed; the GPU's launch alone costs more).
    static constexpr int64_t kGpuMinRows = 2048;
    static constexpr int kGpuMinTokens = 16;
    static constexpr int kNpuMinTokens = 64;   // fewer: the call and the conversions cost more than the CPU's product
    // the NPU's activation and result buffers (rpcmem, grown by a prompt), counted in the memory budget
    static constexpr uint64_t kNpuBufferBytes = 64ull << 20;
    static constexpr float kGpuShare = 0.3f;
    std::unique_ptr<Gpu> gpu_;
    float gpu_share_ = kGpuShare;
    // Generating on the 618 too (GEODE_GPU_DECODE): 0 no, 1 every pass of one or two tokens (singles and verified
    // pairs alike, so speculation keeps the answer of one token at a time), 2 the verified pairs only (their logits
    // then are not the single passes' bits). Its rows (GEODE_GPU_DECODE_SHARE of each held matrix, else gpu_share_)
    // are launched once the CPU's threads are on theirs, and waited for spinning (a sleeping big core is clocked down).
    int gpu_decode_ = 0;
    float gpu_dec_share_ = kGpuShare;
    bool gpu_failed_ = false;
    std::string gpu_status_ = "off";
    // The NPU (npu/host/npu_api.h): libgeode_npu.so opened at run time, a prompt's dense q4 products there (GEODE_NPU=1,
    // off by default: fp16 activations, not the CPU's int8 ones). GEODE_NPU_LIB: the library's path (adb tools).
    struct Npu {
        void *lib = nullptr, *g = nullptr;
        int (*gemm)(void *, int, int, const float *const *, float *const *) = nullptr;
        int (*gemm_multi)(void *, int, const int *, int, const float *const *, float *const *const *) = nullptr;
        void (*close)(void *) = nullptr;
        size_t (*bytes)(const void *) = nullptr;
        double (*dsp)(const void *) = nullptr;
        double (*conv)(const void *) = nullptr;
        std::unordered_map<const uint8_t *, int> ids;   // an adopted matrix's id, by its data
        std::vector<std::pair<const QView *, int>> adopted;   // which view each id is of (rekey: data may move)
        std::string status = "off";
        double wall = 0;   // the products' time here, conversions included
        uint64_t calls = 0;
        // The decoder (npu/dsp/dec.c, npu/dec_cfg.h): a dense Qwen3.5's whole decode step on the DSP, the states and
        // the KV cache in memory both sides map (setup_npu_decode, decode_npu)
        void *(*shared)(void *, size_t, int, int *) = nullptr;
        long long (*dec_size)(void *, const void *, int) = nullptr;
        int (*dec_open)(void *, const void *, int, int, int, int, int) = nullptr;
        int (*dec_kv)(void *, int, int) = nullptr;
        int (*dec_step)(void *, const float *, int, int, int, const int *, int, double *) = nullptr;
        int (*dec_draft)(void *, const float *, const float *, int, int, int, const int *, int *, float *,
                         double *) = nullptr;
        int (*dec_prefill)(void *, const float *, int, int, int, const int *, int, double *) = nullptr;
        int (*mat_hmx)(const void *, int, int *, int *, long long *) = nullptr;
        bool prefill = false;                 // a prompt goes through the DSP too (every matrix has its hmx4 copy)
        int (*dec_mtp_kv)(void *, const float *, const float *, int, int, int, const int *, int, double *) = nullptr;
        bool mtp_kv = false;                  // and the MTP layer's K / V for it (fc, k, v have their hmx4 copies)
        // A prompt's routed experts on HMX (npu/moe_cfg.h, moe_gpu8): the expert cache's first slabs are dma-bufs the
        // DSP maps and the GPU takes as pieces (Gpu8::import); the experts lying there go to the NPU while the GPU
        // computes the rest
        int (*moe)(void *, const void *, int, int, double *) = nullptr;
        struct MoeSlab {
            uint8_t *host;
            size_t bytes;
            int fd;
        };
        std::vector<MoeSlab> moe_slabs;
        uint8_t *moe_io = nullptr;   // x [kMaxBatch][hidden] floats, then y as many
        int moe_io_fd = -1;
        bool moe_on = false;
        bool products_off = false;   // a product failed while the slabs live on: no more products, the session kept
        float *conv_set[2] = {nullptr, nullptr}, *gdn_set[2] = {nullptr, nullptr};   // the two state sets in S
        int arch = 0;                         // the skel's Hexagon version (the decoder: 79 and later)
        int w_fd = -1, p_fd = -1, s_fd = -1;
        uint8_t *w = nullptr, *p = nullptr, *s = nullptr;
        std::vector<uint8_t> cfg;             // the dec_cfg_t
        std::unordered_map<const uint8_t *, int> kv_fds;   // a shared KV chunk's handle
        bool dec_planned = false;             // W and P built: the states and the KV cache go into shared memory
        bool dec = false;                     // the DSP generates
        std::string dec_status = "off";
        Npu() = default;
        Npu(const Npu &) = delete;
        Npu &operator=(const Npu &) = delete;
        ~Npu();
    } npu_;
    void setup_npu();
    // the decoder's matrices (W: tile8 copies of the q4 ones, the head in q8 tiles) and parameters (P), while the
    // matrices are still in the pack's layout; the states into shared memory (S), then the DSP's side opened
    bool build_npu_decode(std::string &why);
    bool open_npu_decode(std::string &why);
    // n tokens (1, or a token and its draft) through every layer on the DSP (the decoder on): their logits (n rows)
    // and hidden rows (hid_) read where the DSP leaves them
    bool decode_npu(const int *tokens, int n, float *logits, std::string &err);
    // a prompt's n rows (X: their embeddings) through every layer on the DSP, DEC_N at a time: the last one's logits,
    // every row's hidden state (hid_long_)
    bool prefill_npu(const float *X, int n, float *logits, std::string &err);
    // the MTP layer's K / V for a prompt's n positions from pos on the DSP (forward_mtp without logits)
    bool mtp_kv_npu(const int *next_tokens, const float *const *hidden, int n, int pos, std::string &err);
    bool npu_spec_ = false;   // the MTP layer and verified pairs on the DSP too
    // the ids keyed by each view's data again (the GPU's setup moves the dense matrices into its shared memory)
    void npu_rekey();
    // ms[j] . xs's n rows -> ys[j] on the NPU, one input for all (count <= 4); false: not taken there
    bool npu_products(const QView *const *ms, int count, int n, const float *const *xs, float *const *const *ys);
    // ms[j] . xs -> ys[j]: on the NPU together when it takes them, else one matmul each
    void matmul_many(const QView *const *ms, int count, int n, const float *const *xs, float *const *const *ys,
                     const QAct *const *pre = nullptr);
    void setup_gpu(const ModelOptions &opt);
    // The Adreno 7xx+ (gpu8.hpp): every dense q4 matrix but lm_head in shared memory in tile8, the expert cache
    // there too (when the pack's experts are tile8), and a batch of kGpuMinTokens tokens and more runs its dense
    // products and its routed experts there whole; one token stays on the CPU.
    std::unique_ptr<Gpu8> gpu8_;
    bool gpu8_experts_ = false;   // the expert cache in the GPU's memory
    bool setup_gpu8(std::string &why);
    // generating on the GPU (gpu8.hpp, dec_*): one token through every layer there, the CPU pinning the routed
    // experts once a layer
    bool dec8_ = false;
    bool verify8_ = false;                            // two tokens at once too (Gpu8::setup_verify)
    int spec_t_ = 2;                                  // the most tokens a verified batch takes (GEODE_SPEC_K)
    int last_batch_ = 0;                              // the tokens of the last verify_n (verify_keep takes some back)
    int chain_row_ = 0;                               // the MTP pass's row a chained draft goes on from (dec_h)
    bool dec8_ok_ = false;                            // setup_decode8 succeeded (set_gpu_decode)
    bool mtp8_ = false;                               // the MTP layer on the GPU too (dec_mtp_)
    Gpu8::DecodeLayer dec_mtp_;
    const float *dec_mtp_norm_ = nullptr, *dec_pre_e_ = nullptr, *dec_pre_h_ = nullptr;
    std::vector<float> mlog_;                         // the MTP pass's logits [2][vocab]
    // The draft head: lm_head's rows of the tokens in the pack's draft_vocab.bin (int32 ids; Strata's subsets) in a
    // matrix of their own on the GPU, rows padded to 32 - a draft is one of them (a token outside is never drafted,
    // only verified). Empty: the whole vocabulary.
    QView draft_head_;
    AlignedBuffer draft_mem_;   // the CPU's draft head (its rows of lm_head copied out, the pack's layout)
    std::vector<int32_t> draft_ids_;
    std::vector<float> last_hid_;                     // last_hidden()
    size_t warm_queued_ = 0;                          // blocks the cache profile had read at open()
    // after dec_front of a layer (the store's layer sl): its route, the T tokens' picks (each token's tail's misses
    // left out) pinned in one union and computed as their blocks come in, the next layer's guesses read ahead (when
    // there is one), the combine; held: the pins of the layer before (let go here), then this layer's
    bool back_gpu8(int sl, const Layer &L, const Gpu8::DecodeLayer &D, bool has_next, int T, std::vector<int> &held,
                   std::string &err, bool has_next2 = false);
    int dec_guesses2_ = 0;   // the layer after next's experts read ahead at each route (GEODE_GUESS2)
    float *conv_bk_ = nullptr, *gdn_bk_ = nullptr;    // the states after a verified pair's first token
    // Speculating on the CPU (verify, draft): the backups in the process's memory, a pair's batch keeping the states
    // after its first token there (gdn: each strip copied while it is in the cache), a rejected draft swapping them in
    bool cpu_spec_ = false, verify_keep_ = false;
    std::vector<float> conv_bk_mem_, gdn_bk_mem_;
    bool setup_cpu_spec();
    std::vector<int> spec_slots_;   // the expert cache's slots taken for them (ExpertStore::take: a reset keeps them out)
    uint64_t spec_bytes() const;   // what it holds: the backups and the CPU's draft head
    std::vector<Gpu8::DecodeLayer> dec_layers_;
    const float *dec_final_norm_ = nullptr;
    int dec_guesses_ = 8;   // the next layer's experts read ahead at each route
    bool setup_decode8(std::string &why);
    bool decode_gpu8(int token, float *logits, std::string &err);
    // the NPU's slab holding p (an index into npu_.moe_slabs), or -1
    int moe_slab_of(const uint8_t *p) const;
    // moe(): the NPU's share of the routed experts so far - its y summed per token, its picks
    std::vector<float> npu_sum_;
    std::vector<uint8_t> npu_pick_;
    bool npu_x_ = false, npu_y_ = false;   // x handed over for this moe(); npu_sum_ holds something
    bool moe_gpu8(int l, const Layer &L, int n, const std::vector<std::array<int, 3>> &picks,
                  const std::vector<std::pair<size_t, size_t>> &runs, const std::vector<int> &slots,
                  const std::vector<DType> &types, size_t first, std::string &err,
                  const Gpu8::DecodeLayer *shared = nullptr);   // the shared expert along (the GPU front)
    void quad_dense();
    // an expert's matrix in its slot (in tile8 where the pack says so)
    QView expert_view(DType type, int64_t rows, int64_t cols, const uint8_t *data, float global) const;
    int64_t gpu_rows(const QView &m, int n) const;
    bool exact_ = false, emulate_ = false, emulate_bf16_ = false, check_ = false;
    // A diagnostic of where the int8 path loses accuracy (the exact path with emulate_int8 only; model.cpp,
    // emulate_this): GEODE_EMU_KINDS (which products round their activations: qkv z out q k v o shg shu shd head mtp,
    // exp = the routed experts, expgu / expd = their gate_up / down), GEODE_EMU_LAYERS (a-b), GEODE_EMU_FMT
    // (how: i8 = the int8 kernels' rounding, i8bN = blocks of N, i8o = each block's largest value kept, f16, i16)
    std::unordered_map<const void *, std::pair<int, int>> emu_kinds_;   // a dense matrix's data -> (kind, layer)
    bool emu_alt_ = false;      // the product at hand is in GEODE_EMU_FMT2_LAYERS: rounded as GEODE_EMU_FMT2 (f16, f32...)
    bool emu_single_ = false;   // the product at hand is one token's (GEODE_EMU_FMT1)
    bool emulate_this(const QView &m);
    void emulate_round(const float *x, int64_t cols, float *dst);
    double check_worst_ = 0.0;
    struct Profile {
        int tokens = 0;
        double gdn = 0, attention = 0, moe = 0, router = 0, shared = 0, expert_wait = 0, head = 0;
        double gdn_core = 0, attn_core = 0;   // the recurrence / the scores and values, without projections
        double gdn_conv = 0, gdn_strips = 0, gdn_norm = 0;   // inside gdn_core
        // generating, experts spread over the threads: the wall time, and summed over the threads the arithmetic
        // and the waiting for reads
        double spread_wall = 0, spread_compute = 0, spread_wait = 0;
        double gpu0 = 0;   // the GPU's busy seconds then
        double gpu8 = 0, gpu8_busy0 = 0;   // the Adreno 7xx+ calls' wall time; its kernels' busy seconds then
        double npu_dec = 0, npu_dec_dsp = 0;   // tokens the DSP generated (decode_npu): the calls' wall time, the DSP's
        uint64_t npu_dec_steps = 0;
        double npu_pre = 0, npu_pre_dsp = 0;   // prompt tokens the DSP read (prefill_npu)
        uint64_t npu_pre_tokens = 0;
        double npu_moe = 0, npu_moe_dsp = 0;   // a prompt's experts on the NPU (moe_gpu8): the calls' wall, the DSP's
        uint64_t npu_moe_experts = 0, gpu_moe_experts = 0;   // how many went each way
        double pf_cpu_experts = 0;   // a prompt's experts the CPU took beside the GPU (GEODE_PF_CPU_SHARE): their time
        // generating on the GPU: enqueuing a layer's front, waiting for its route, pinning the experts, enqueuing the
        // back, the head (lm_head and the wait for the whole token)
        double d_front = 0, d_route = 0, d_pin = 0, d_back = 0, d_head = 0;
        // of the pinning: the waits for picks a guess had started reading, and for picks not cached at all
        double d_wait_flight = 0, d_wait_miss = 0;
        // back_gpu8's CPU time between a route and its ready experts enqueued (the GPU idles): deciding (skips,
        // union, what is cached), pinning, the next layer's guesses, the views, the enqueueing
        double c_decide = 0, c_pin = 0, c_guess = 0, c_views = 0, c_enq = 0;
        uint64_t d_n_flight = 0, d_n_miss = 0;
        uint64_t verified = 0, rejected = 0;   // two-token passes, and of them the second token taken back
        uint64_t drafts = 0;                   // MTP passes on the GPU
        uint64_t tail_unwaited = 0;            // the tail's picks still being read, left out (back_gpu8)
        double d_draft = 0;                    // their time
        double d_verify = 0;                   // the CPU's verified pairs' passes (verify)
        // the routed experts split by rows: the two products' wall time, the threads' waits for reads (summed)
        double rows_up = 0, rows_down = 0, rows_wait = 0;
        // the 618 generating: products shared with it, of them those whose GPU rows were not done when the CPU's
        // were, the launches (activations in, the kernel queued), the waits, the results out
        uint64_t g_dec = 0, g_late = 0;
        double g_start = 0, g_wait = 0, g_out = 0;
        uint64_t weighted0 = 0, stolen0 = 0;   // the pool's counters then
        double pass = 0, mtp_pass = 0;         // forward_batch's and forward_mtp's wall time
        uint64_t hits0 = 0, prefetch_hits0 = 0, misses0 = 0, prefetches0 = 0, bytes0 = 0, dropped0 = 0;   // the store's counters then
        uint64_t skipped0 = 0;   // tail_skipped_ then
        uint64_t prior_picks0 = 0, prior_moved0 = 0;   // the route prior's counters then
        std::vector<uint64_t> class_picks0, class_misses0;   // the store's size classes' counters then
    } prof_;
};

}  // namespace geode
