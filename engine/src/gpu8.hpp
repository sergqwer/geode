// gpu8.hpp - a prompt's q4 products on the Adreno 830's int8 dot product (PLAN.md, "The second target"; bench/gpu/i8
// has the measurements). OpenCL C kernels (gpu8_kernels.inc) on the tile8 layout (kernels.hpp), the weights in memory
// the CPU and the GPU share: CL_MEM_ALLOC_HOST_PTR mapped once, which on this SoC is cached for the CPU, takes
// O_DIRECT reads, and shows the CPU's writes to the GPU without an unmap. The dense q4 matrices move there at load time
// (adopt), the expert cache lives there (alloc'd pieces, ExpertStore's slabs), and a prompt batch's dense products and
// routed experts run on the GPU; one token at a time stays on the CPU.
//
// Activations go in as floats and are rounded on the GPU as the CPU rounds them (int8 per 16 values, amax / 127); the
// sums run in another order, so the results are the CPU int8 path's up to float rounding, not bit for bit.
#pragma once

#include <functional>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels.hpp"

namespace geode {

class Gpu8 {
public:
    Gpu8();
    ~Gpu8();
    Gpu8(const Gpu8 &) = delete;
    Gpu8 &operator=(const Gpu8 &) = delete;

    // An OpenCL GPU with cl_qcom_dot_product8, the program built; false and why otherwise.
    bool open(std::string &err);
    // Where the compiled programs are kept between runs (cl::Program::build_cached); before open(). Empty: none.
    void set_cache_dir(const std::string &dir);
    // Gemma 4's q4i matrices (levels code - 8, signed E4M3 scales) instead of q4: the programs built for them (-D Q4I),
    // holds() takes q4i only; before open(). Only the batch products (gemm) are meant for them.
    void set_q4i(bool on);
    // A prompt's projections elsewhere (the NPU: Model::npu_products): ms[j] . n rows of `in` (in_stride floats apart)
    // -> outs[j] (ms[j]->rows floats a row); false: not taken there (the GPU multiplies them). The GDN's qkv, z and out
    // and the attention's q, k, v and o go there; the rest of a layer stays here. must: taken whatever happens (the
    // prompt's pipeline has the GPU busy with the next part: the host's CPU multiplies them when the NPU cannot).
    using HostGemm = std::function<bool(const QView *const *ms, int count, int n, const float *in, int in_stride,
                                        float *const *outs, bool must)>;
    void set_host_gemm(HostGemm f);
    std::string device() const;

    // Shared memory: one piece of up to max_piece() bytes, mapped for good; nullptr on failure.
    uint8_t *alloc(size_t bytes, std::string &err);
    // A piece alloc() gave, given back to the driver whole (its pages with it), once every queue is done with it;
    // false: no piece starts there.
    bool free(uint8_t *piece);
    // A dma-buf the caller allocated (fd; host: its CPU mapping, bytes long) taken as a piece without a copy - the
    // ION type of cl_qcom_ext_host_ptr given the dma-buf's fd, IO-coherent (the GPU sees the CPU's writes as they are;
    // geode-clprobe dmabuf). It stays the caller's: free() lets the GPU go of it only. false: the driver refused it.
    bool import(int fd, uint8_t *host, size_t bytes, std::string &err);
    size_t max_piece() const;
    size_t allocated() const;   // every piece, staging included

    // A q4 matrix (rows % 32 == 0, cols % 64 == 0; the pack's layout or tile8) copied into shared memory in tile8 and
    // pointed at it (quad + tile8). Views of the same bytes move to the same copy.
    static bool fits(const QView &m);
    bool adopt(QView &m, std::string &err);
    // adopt() for every view that fits, on `threads` threads (the pieces allocated, then 32-row tiles converted, all at
    // once): the same pieces and bytes as one adopt() after another. false (why in err) if a piece could not be had;
    // the views that moved stay moved.
    bool adopt_all(const std::vector<QView *> &views, int threads, std::string &err);
    bool holds(const QView &m) const;   // a tile8 q4 matrix inside shared memory (adopted, or an expert slot)

    // Staging for batches of up to `tokens` tokens: dense inputs of up to max_cols columns and outputs of up to
    // max_rows rows; routed experts of `hidden` x `ff`, top_k picks a token out of n_experts.
    bool reserve(int tokens, int64_t max_cols, int64_t max_rows, int64_t hidden, int64_t ff, int top_k, int n_experts,
                 std::string &err);
    size_t staging_bytes(int tokens, int64_t max_cols, int64_t max_rows, int64_t hidden, int64_t ff, int top_k,
                         int n_experts) const;

    // ys[t][0, m.rows) = m . xs[t] for n tokens (n <= the reserved tokens, m.rows <= max_rows, m.cols <= max_cols).
    bool gemm(const QView &m, int n, const float *const *xs, float *const *ys, std::string &err);

    // A layer's routed experts (some of them: a wave of the cache's pins). Each expert: its gate_up and down matrices
    // (tile8 in shared memory) and the picks it serves, (token, output row). Token t's input is hs[t] (hidden
    // floats); pick (t, r)'s output goes to row r of out() (hidden floats a row): silu(gate) * up, then down.
    struct Expert {
        QView gu, dn;
        std::vector<std::pair<int, int>> picks;
    };
    struct DecodeLayer;
    // shared: that layer's shared expert too, for every token: token t's output to row n * top_k + t of out()
    bool experts(const std::vector<Expert> &ex, int n, const float *const *hs, std::string &err,
                 const DecodeLayer *shared = nullptr);
    const float *out() const;   // the routed outputs: top_k rows a token (the rows experts() was given), then the
                                // shared expert's (one a token)
    float *out_rows();          // the same, writable: rows the CPU computed copied in once experts() returned

    // ---- An image's products (vision.cpp): q8 matrices and the activations in pieces the encoder borrows for one
    // image (vision_alloc, 64-byte aligned) and gives back (vision_release); the products run in place there.
    uint8_t *vision_alloc(size_t bytes, std::string &err);
    void vision_release();
    // a q8 matrix (rows % 4 == 0) in the GPU's layout ("q8gpu", gpu8_kernels.inc): its bytes, and the conversion
    static size_t q8gpu_bytes(int64_t rows, int64_t cols);
    static void to_q8gpu(const QView &m, uint8_t *dst);
    // y[t][0, rows) (rows ystride floats apart) = W x[t] (xstride apart) for t < n, int8 x q8 as the CPU computes it
    // (the sums in another order); w (to_q8gpu's bytes), x and y inside vision pieces. Waits.
    bool vision_gemm(const uint8_t *w, int64_t rows, int64_t cols, float global, int n, const float *x, size_t xstride,
                     float *y, size_t ystride, std::string &err);
    // the same for the first `first` rows (a multiple of 4), enqueued: the CPU computes the rest meanwhile, then
    // vision_wait
    bool vision_gemm_start(const uint8_t *w, int64_t rows, int64_t cols, int64_t first, float global, int n,
                           const float *x, size_t xstride, float *y, size_t ystride, std::string &err);
    bool vision_wait(std::string &err);

    double busy_seconds() const;   // the kernels' time, summed
    int64_t launches() const;
    // a batch product the GPU takes (holds(m), within the reserved tokens, rows and columns)
    bool fits_batch(const QView &m, int n) const;

    // ---- Generating on the GPU: one token through every layer on a plain queue, the CPU in the loop once a layer
    // (it reads the route, pins the routed experts' slots, has misses read, and passes their places). Parameters,
    // states and the KV cache live in shared memory where the model puts them (dec_params, alloc).
    struct DecodeShape {
        int hidden = 0, vocab = 0, n_experts = 0, top_k = 0, moe_ff = 0;
        int nh = 0, nkv = 0, hd = 0, half = 0, n_attn = 0, kv_chunk = 0, max_ctx = 0;   // attention, the KV cache
        int nk = 0, nv = 0, dk = 0, dv = 0, conv_k = 0;                                  // Gated DeltaNet
        int guesses = 8;   // the next layer's experts read ahead at each route (<= n_experts)
        int guesses2 = 0;  // and the layer after it's (<= 16; 0: none)
        bool kv_i8 = false;   // the KV cache in int8 + a float scale a position and KV head (Model::kv_scale)
        float eps = 1e-6f;
        // verifying: the most tokens a pass takes (2: pairs; up to kMaxTokens), and the GDN states' backups - one
        // after each token of a batch but its last, bk_conv / bk_gdn floats apart (DecodeLayer::conv_bk / gdn_bk: the
        // first's)
        int max_tokens = 2;
        int bk_conv = 0, bk_gdn = 0;
    };
    static constexpr int kMaxTokens = 5;
    struct DecodeLayer {
        bool attn = false;
        int index = 0;   // among the attention layers (the KV cache), or the GDN layers
        // in shared memory: the dense q4 matrices adopted, the rest in the dec_params piece
        QView qkv, z, out, q, k, v, o, sh_gate, sh_up, sh_down;
        const float *in_norm = nullptr, *post_norm = nullptr;
        const uint16_t *ab = nullptr;       // bf16 [a rows ; b rows]
        const float *A_log = nullptr, *dt_bias = nullptr, *conv = nullptr, *gnorm = nullptr;
        const float *q_norm = nullptr, *k_norm = nullptr;
        const uint16_t *router = nullptr;   // bf16 [n_experts + 1][hidden]: the router's rows, then the shared gate
        float *conv_state = nullptr, *gdn_state = nullptr;   // this layer's (GDN), in shared memory
        // verifying a batch: the states after its first token go here (the same layout), after the second one
        // bk_conv / bk_gdn floats on, ..., for a draft that falls
        float *conv_bk = nullptr, *gdn_bk = nullptr;
    };
    bool setup_decode(const DecodeShape &s, const float *inv_freq, std::string &err);
    // The routed experts a token takes from now on (Model::set_experts), 1 .. the top_k its buffers were made for
    // (setup_decode, reserve), and the next layer's guesses (0: as they are) - the route's, the experts' slots and the
    // combine's K. Between passes.
    void set_experts(int k, int guesses);
    uint8_t *dec_params(size_t bytes, std::string &err);   // the one piece the DecodeLayer's small tensors go into
    // Batches of 2 .. the shape's max_tokens at once (drafts verified): the programs built with T = 2 .. (dk = 128
    // only). False and why otherwise.
    bool setup_verify(std::string &err);
    // The tokens of the next passes, 1 .. max_tokens (setup_verify first). Token t's rows follow token 0's in every
    // buffer; each token's arithmetic is the one-token pass's, in the same order: the same bits.
    bool set_tokens(int t);
    int tokens() const;
    float *dec_x();   // the tokens' hidden states (hidden floats each): the embeddings go here
    // the tokens' hidden states after the last norm_q (after dec_head: the final norm's, an MTP layer's input)
    float *dec_h();
    // The MTP layer's input: fc . [norm(embedding in dec_x) | norm(hidden state in dec_h)] into dec_x, enqueued (then
    // dec_front with the MTP layer, its experts, dec_head with its norm)
    bool dec_mtp_in(const QView &fc, const float *pre_e, const float *pre_h, std::string &err);
    // layer L's front, enqueued: its token mixer, the post-norm, the router (and next's, the guess), the route; the
    // tokens at positions pos, pos + 1 ..
    // (next2: the layer after next, guessed too when the shape has guesses2)
    // The rotary position of a token at KV index pos is pos + shift (after an image the two part: M-RoPE gives the
    // image's tokens fewer positions than KV rows; Model::rope_pos). Every token generated is text: one shift.
    void set_rope_shift(int shift);
    bool dec_front(const DecodeLayer &L, const DecodeLayer *next, int pos, const std::vector<uint8_t *> &kv,
                   std::string &err, const DecodeLayer *next2 = nullptr);
    // waits for the route: each token's top_k picks (ids[t * top_k ..]) and its `guesses` guesses of the next layer
    // (and its guesses2 of the one after, into gids2, when dec_front had next2)
    bool dec_route(int *ids, int *gids, std::string &err, int *gids2 = nullptr);
    // The routed experts in slots ks (gu[u] / dn[u]: one token, slot = pick; T tokens, the picks' union, slots <
    // T top_k) and the shared expert if asked, as their blocks come in, every token through each; then x += the MoE
    // once every slot needed and the shared expert went in: token t's pick k is slot (um[t] >> 6k) & 63 (one token:
    // k), skip[t] bit k = left out (never computed), renorm: the kept weights scaled to sum 1.
    bool dec_experts(const DecodeLayer &L, const QView *gu, const QView *dn, const std::vector<int> &ks, bool shared,
                     std::string &err);
    bool dec_combine(const uint32_t *skip, const uint64_t *um, bool renorm, std::string &err);
    // the final norm and lm_head into logits (vocab floats a token); waits
    bool dec_head(const float *final_norm, const QView &lm_head, float *logits, std::string &err);
    // enqueued: bytes from src to dst, both in shared memory (a rejected draft's states back)
    bool dec_copy(void *dst, const void *src, size_t bytes, std::string &err);
    bool dec_wait(std::string &err);   // everything enqueued done

    // ---- A prompt's token mixers on the GPU (Model::forward_batch, once setup_decode is done): a layer's input h (n
    // rows of hidden floats, the input norm applied) in, the mixer's output (n rows of hidden floats) out. The Gated
    // DeltaNet's states and the int8 KV cache are the ones generating uses, in shared memory as the CPU keeps them.
    // n <= the reserved tokens; dk = 128, head_dim 256 in groups of 8 query heads. Waits.
    bool pf_ready(std::string &err);   // the buffers and kernels there (the first call makes them); false: why not
    std::string pf_profile() const;    // GEODE_PF_PROF=1: the batch queue's kernels' time by name
    bool pf_gdn(const DecodeLayer &L, int n, const float *h, float *mix, std::string &err);
    // A prompt's router: n rows of hidden floats (h) -> *logits, n_experts + 1 a token (the routed experts', then the
    // shared expert's gate), in shared memory until the next call
    bool pf_router(const DecodeLayer &L, int n, const float *h, const float **logits, std::string &err);
    // pos3: each token's three rotary positions (Model::rope_pos); kv_pos: the KV index of the first token, its K and V
    // written there
    bool pf_attn(const DecodeLayer &L, int n, int kv_pos, const int32_t *pos3, const std::vector<uint8_t *> &kv,
                 const float *h, float *mix, std::string &err);
    std::string dec_profile() const;   // with GEODE_GPU8_PROF=1: the one-token kernels' time by kernel
    void dec_selftest(const QView &m, int reps);   // measuring: m's one-token product reps times, alone

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace geode
