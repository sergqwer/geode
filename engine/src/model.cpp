// model.cpp - the forward pass, written to match transformers' Qwen3_5Moe modules (modeling_qwen3_5_moe.py) term by
// term: Qwen3_5MoeRMSNorm, Qwen3_5MoeGatedDeltaNet (the recurrent rule of torch_recurrent_gated_delta_rule),
// Qwen3_5MoeAttention, Qwen3_5MoeTopKRouter / SparseMoeBlock.
//
// Batches: every per-token computation is the same code whether the token comes alone or with others (a batch only
// changes how often a weight row is fetched), so a prompt read in batches gives bit for bit the logits it gives read
// one token at a time.
#include "model.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <mutex>
#include <numeric>
#include <chrono>
#include <thread>

#if defined(__ANDROID__)
#include <dlfcn.h>   // the NPU's library (setup_npu)
#endif
#include "../../npu/dec_cfg.h"   // the decoder's model description (the DSP reads the same bytes)
#include "../../npu/moe_cfg.h"   // a prompt's experts on the NPU (moe_gpu8)

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define GEODE_NEON_ATTENTION 1
#endif

namespace geode {


namespace {

const Tensor *need(const Pack &p, const std::string &name, std::string &err) {
    const Tensor *t = p.find(name);
    if (!t) err = "the pack has no " + name;
    return t;
}


}  // namespace

bool Model::bind_layer(Layer &L, const std::string &p, bool attn, int index, std::string &err) {
    const auto &c = cfg_;
    auto f32 = [&](const std::string &n, int64_t count) -> const float * {
        const Tensor *t = need(pack_, p + n, err);
        if (!t) return nullptr;
        if (t->type != DType::F32 || t->rows * t->cols != count) {
            err = p + n + ": expected f32 x " + std::to_string(count);
            return nullptr;
        }
        return t->f32();
    };
    auto bf16 = [&](const std::string &n, int64_t rows, int64_t cols) -> const uint16_t * {
        const Tensor *t = need(pack_, p + n, err);
        if (!t) return nullptr;
        if (t->type != DType::BF16 || t->rows != rows || t->cols != cols) {
            err = p + n + ": expected bf16 [" + std::to_string(rows) + ", " + std::to_string(cols) + "]";
            return nullptr;
        }
        return t->bf16();
    };
    auto mat = [&](const std::string &n, int64_t rows, int64_t cols, QView &out) -> bool {
        const Tensor *t = need(pack_, p + n, err);
        if (!t) return false;
        if (!is_quantized(t->type) || t->rows != rows || t->cols != cols || !t->data) {
            err = p + n + ": expected a quantized [" + std::to_string(rows) + ", " + std::to_string(cols) + "]";
            return false;
        }
        out = t->view();
        return true;
    };

    const int H = c.hidden;
    L.attn = attn;
    L.index = index;
    if (!(L.in_norm = f32("input_layernorm.weight", H)) || !(L.post_norm = f32("post_attention_layernorm.weight", H)))
        return false;
    if (L.attn) {
        const int nh = c.n_heads, nkv = c.n_kv_heads, hd = c.head_dim;
        if (!mat("self_attn.q_proj.weight", nh * hd * 2, H, L.q) || !mat("self_attn.k_proj.weight", nkv * hd, H, L.k) ||
            !mat("self_attn.v_proj.weight", nkv * hd, H, L.v) || !mat("self_attn.o_proj.weight", H, nh * hd, L.o) ||
            !(L.q_norm = f32("self_attn.q_norm.weight", hd)) || !(L.k_norm = f32("self_attn.k_norm.weight", hd)))
            return false;
    } else {
        const int kd = c.gdn_k_heads * c.gdn_k_dim, vd = c.gdn_v_heads * c.gdn_v_dim, conv_dim = 2 * kd + vd;
        if (!mat("linear_attn.in_proj_qkv.weight", conv_dim, H, L.qkv) ||
            !mat("linear_attn.in_proj_z.weight", vd, H, L.z) || !mat("linear_attn.out_proj.weight", H, vd, L.out) ||
            !(L.a = bf16("linear_attn.in_proj_a.weight", c.gdn_v_heads, H)) ||
            !(L.b = bf16("linear_attn.in_proj_b.weight", c.gdn_v_heads, H)) ||
            !(L.A_log = f32("linear_attn.A_log", c.gdn_v_heads)) ||
            !(L.dt_bias = f32("linear_attn.dt_bias", c.gdn_v_heads)) ||
            !(L.conv = f32("linear_attn.conv1d.weight", static_cast<int64_t>(conv_dim) * c.conv_kernel)) ||
            !(L.gnorm = f32("linear_attn.norm.weight", c.gdn_v_dim)))
            return false;
    }
    if (c.dense) {   // the MLP as an ungated shared expert, no router, no routed experts
        L.router = nullptr, L.shared_gate = nullptr, L.expert_scales = nullptr;
        return mat("mlp.gate_proj.weight", c.shared_ff, H, L.sh_gate) &&
               mat("mlp.up_proj.weight", c.shared_ff, H, L.sh_up) &&
               mat("mlp.down_proj.weight", H, c.shared_ff, L.sh_down);
    }
    if (!(L.router = bf16("mlp.gate.weight", c.n_experts, H)) ||
        !(L.shared_gate = bf16("mlp.shared_expert_gate.weight", 1, H)) ||
        !mat("mlp.shared_expert.gate_proj.weight", c.shared_ff, H, L.sh_gate) ||
        !mat("mlp.shared_expert.up_proj.weight", c.shared_ff, H, L.sh_up) ||
        !mat("mlp.shared_expert.down_proj.weight", H, c.shared_ff, L.sh_down) ||
        !(L.expert_scales = f32("mlp.experts.scales", static_cast<int64_t>(c.n_experts) * 2)))
        return false;
    return true;
}

bool Model::bind(std::string &err) {
    const auto &c = cfg_;
    int n_attn = 0, n_gdn = 0;
    layers_.resize(static_cast<size_t>(c.n_layers));
    for (int l = 0; l < c.n_layers; ++l) {
        const bool attn = c.attention[static_cast<size_t>(l)];
        if (!bind_layer(layers_[static_cast<size_t>(l)], "layers." + std::to_string(l) + ".", attn,
                        attn ? n_attn++ : n_gdn++, err))
            return false;
    }
    if (c.mtp_layers) {
        // a full-attention layer behind fc; its KV after the main attention layers'
        if (!bind_layer(mtp_layer_, "mtp.layers.0.", true, n_attn, err)) return false;
        const Tensor *fc = need(pack_, "mtp.fc.weight", err);
        const Tensor *nm = need(pack_, "mtp.norm.weight", err);
        const Tensor *pe = need(pack_, "mtp.pre_fc_norm_embedding.weight", err);
        const Tensor *ph = need(pack_, "mtp.pre_fc_norm_hidden.weight", err);
        if (!fc || !nm || !pe || !ph) return false;
        if (!is_quantized(fc->type) || fc->rows != c.hidden || fc->cols != 2 * c.hidden || !fc->data) {
            err = "mtp.fc.weight: expected a quantized [hidden, 2 x hidden]";
            return false;
        }
        mtp_fc_ = fc->view();
        mtp_norm_ = nm->f32();
        mtp_pre_e_ = pe->f32();
        mtp_pre_h_ = ph->f32();
        mtp_ = true;
    }
    const Tensor *head = need(pack_, "lm_head.weight", err);
    const Tensor *norm = need(pack_, "norm.weight", err);
    if (!head || !norm) return false;
    if (!is_quantized(head->type) || head->cols != c.hidden || !head->data) {
        err = "lm_head.weight: expected a quantized [vocab, hidden]";
        return false;
    }
    lm_head_ = head->view();
    final_norm_ = norm->f32();
    return true;
}

namespace {
// GEODE_OPEN_TIMES: where opening a model spends its time, one line a step on stderr (a measuring switch)
struct OpenClock {
    bool on = false;
    double t = 0;
    void start() {
        on = std::getenv("GEODE_OPEN_TIMES") != nullptr;
        t = now_seconds();
    }
    void mark(const char *what) {
        if (!on) return;
        const double n = now_seconds();
        std::fprintf(stderr, "open: %-32s %7.0f ms\n", what, (n - t) * 1000.0);
        t = n;
    }
};
OpenClock open_clock;
}  // namespace

bool Model::open(const std::string &dir, const ModelOptions &opt, std::string &err) {
    open_clock.start();
    if (!pack_.open(dir, err)) return false;
    open_clock.mark("pack");
    {
        std::string ve;
        if (vision_.open(pack_, opt.direct_io, ve)) {
            vision_status_ = "ok";
            if (vision_.shape().out_hidden != pack_.config().hidden) {
                vision_.disable();
                vision_status_ = "the vision encoder's output is not the model's width";
            }
        } else if (!ve.empty()) {
            vision_.disable();
            vision_status_ = ve;   // the model works without it
        }
    }
    open_clock.mark("vision");
    cfg_ = pack_.config();
    new_conversation();
    if (!bind(err)) return false;
    open_clock.mark("bind");
    const auto &c = cfg_;
    if (c.head_dim % 8) {
        err = "head_dim " + std::to_string(c.head_dim) + " is not a multiple of 8";
        return false;
    }
    ctx_ = opt.context_length > 0 ? opt.context_length : 8192;
    exact_ = opt.exact;
    emulate_ = opt.exact && opt.emulate_int8;
    emulate_bf16_ = opt.exact && opt.emulate_bf16;
    check_ = !opt.exact && opt.check_int8;
    // the arithmetic on the big cores, the flash readers on the little ones (where the phone says which is which)
    big_ = big_cores();
    {
        // Little-core helpers for a prompt's products: none by default. On the Redmi (2 x A76 + 6 x A55) four of
        // them made a 1100-token prompt slower, 9.2 -> 8.2 tokens/s: the GDN heads they take finish 3-4x later than
        // the big cores' and the big cores wait, the small products pay more to split, and the reads share the cores.
        const std::vector<int> little = little_cores();
        const int helpers = std::max(0, opt.helper_threads);
        // A token's products are another matter: on the 720G its two A76 left the six A55 idle (one token is ~600
        // short products; a dynamic split hands an A55 a chunk the A76 then wait for), so they join with shares by
        // their speed (threads.hpp: weighted products). GEODE_LITTLE_THREADS: how many (0: none);
        // GEODE_LITTLE_WEIGHT: an A55's speed against an A76 fixed (else learned from the products, from 1/3).
        const char *lt = std::getenv("GEODE_LITTLE_THREADS");
        little_threads_ = opt.little_threads >= 0 ? opt.little_threads
                          : lt                    ? std::max(0, std::atoi(lt))
                                                  : static_cast<int>(little.size());
        pool_ = std::make_unique<ThreadPool>(opt.compute_threads > 0 ? opt.compute_threads : default_compute_threads(),
                                             big_, std::max(helpers, little_threads_), little);
        pool_->set_wide_helpers(helpers);
        if (const char *w = std::getenv("GEODE_LITTLE_WEIGHT")) pool_->set_little_weight(static_cast<float>(std::atof(w)), false);
        little_threads_ = std::min(little_threads_, pool_->helpers());
    }
    routed_rows_ = !(std::getenv("GEODE_ROUTED_ROWS") && std::atoi(std::getenv("GEODE_ROUTED_ROWS")) == 0);
    little_dense_ = !(std::getenv("GEODE_LITTLE_DENSE") && std::atoi(std::getenv("GEODE_LITTLE_DENSE")) == 0);
    open_clock.mark("threads");
    // the NPU first: it takes the pack's layout, before anything is laid out anew
    const bool npu_on = std::getenv("GEODE_NPU") ? std::atoi(std::getenv("GEODE_NPU")) != 0 : opt.npu > 0;
    if (exact_ || check_) npu_.status = "off (exact / checked arithmetic)";
    else if (!npu_on) npu_.status = "off (the setting)";
    else {
        setup_npu();
        open_clock.mark("npu");
    }
    if (opt.gpu) {
        if (exact_ || check_) gpu_status_ = "off (exact / checked arithmetic)";
        else if (c.dense) gpu_status_ = "off (a dense model: the GPU paths are built around the experts)";
        else setup_gpu(opt);
    }
    npu_rekey();
    // with both: a prompt's dense projections on the NPU, the rest of its layers (and the routed experts) on the GPU
    if (npu_.g && gpu8_)
        gpu8_->set_host_gemm([this](const QView *const *ms, int count, int n, const float *in, int in_stride,
                                    float *const *outs, bool must) {
            if (count < 1 || count > 4 || (n < kNpuMinTokens && !must)) return false;
            std::vector<const float *> xs(static_cast<size_t>(n));
            for (int t = 0; t < n; ++t) xs[static_cast<size_t>(t)] = in + static_cast<size_t>(t) * in_stride;
            std::vector<float *> rows(static_cast<size_t>(count) * n);
            float *const *ys[4];
            for (int j = 0; j < count; ++j) {
                for (int t = 0; t < n; ++t)
                    rows[static_cast<size_t>(j) * n + t] = outs[j] + static_cast<size_t>(t) * ms[j]->rows;
                ys[j] = rows.data() + static_cast<size_t>(j) * n;
            }
            if (npu_products(ms, count, n, xs.data(), ys)) return true;
            if (!must) return false;
            for (int j = 0; j < count; ++j) matmul(*ms[j], n, xs.data(), ys[j]);   // the NPU could not: here
            return true;
        });
    open_clock.mark("gpu (rest)");
    // GEODE_DECODE_THREADS / GEODE_SPREAD_THREADS: measuring switches (model.hpp)
    if (const char *e = std::getenv("GEODE_SPREAD_THREADS")) spread_threads_ = std::max(1, std::atoi(e));
    if (const char *e = std::getenv("GEODE_DECODE_THREADS")) decode_threads_ = std::max(1, std::atoi(e));
    if (const char *e = std::getenv("GEODE_ATTN_BLOCK")) attn_block_ = std::max(1, std::atoi(e));   // tests (model.hpp)
    // GEODE_NO_QUAD: the pack's layout (a comparison switch for tests)
    if (!exact_ && opt.quad_layout && !std::getenv("GEODE_NO_QUAD")) quad_dense();
    open_clock.mark("quad dense");

    int n_attn = 0, n_gdn = 0;
    for (bool a : c.attention) (a ? n_attn : n_gdn)++;
    if (mtp_) ++n_attn;   // the MTP layer's KV, after the main layers'
    const int kd = c.gdn_k_heads * c.gdn_k_dim, vd = c.gdn_v_heads * c.gdn_v_dim, conv_dim = 2 * kd + vd;
    const size_t kv_row = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    n_attn_ = n_attn;
    kv_row_ = kv_row;
    // the KV cache grows with the conversation, kv_chunk_ positions at a time: int8 with a scale a position and KV
    // head (the GPU's flash decoding reads it: head_dim 256, 8 query heads a KV head), else FP16; FP32 when exact
    kv_chunk_ = opt.kv_chunk > 0 ? opt.kv_chunk : gpu8_ && pack_.experts_tile8() ? 4096 : kKvChunk;
    kv_heads_ = c.n_kv_heads;
    kv_i8_ = !exact_ && opt.kv_int8 && !std::getenv("GEODE_KV_FP16") &&
             (!gpu8_ || (c.head_dim == 256 && c.n_heads == 8 * c.n_kv_heads));
    kv_chunk_bytes_ = kv_i8_ ? 2 * static_cast<size_t>(n_attn) * kv_chunk_ * (kv_row + static_cast<size_t>(c.n_kv_heads) * sizeof(float))
                             : 2 * static_cast<size_t>(n_attn) * kv_chunk_ * kv_row * (exact_ ? sizeof(float) : sizeof(_Float16));
    const uint64_t state_bytes =
        static_cast<uint64_t>(n_gdn) * (static_cast<uint64_t>(conv_dim) * (c.conv_kernel - 1) +
                                        static_cast<uint64_t>(c.gdn_v_heads) * c.gdn_k_dim * c.gdn_v_dim) * sizeof(float);
    // (a dense model has no experts: a token cache of two 4 KiB slots keeps the store's code paths whole)
    const size_t slot = c.dense ? 4096 : (pack_.max_expert_block_bytes() + 4095) / 4096 * 4096;
    // The experts' size classes: one per block size in the pack (a pack mixing formats within layers - tools/bitmap.py
    // - gives its small blocks small slots); one size, one class: the cache as before
    const int EL = c.n_layers + c.mtp_layers;
    std::vector<uint8_t> key_class(static_cast<size_t>(EL) * c.n_experts);
    std::vector<size_t> class_bytes, class_count;
    std::vector<DType> class_type;
    {
        std::vector<size_t> key_bytes(key_class.size());
        for (int l = 0; l < EL; ++l)
            for (int e = 0; e < c.n_experts; ++e) {
                int f;
                uint64_t off;
                DType t;
                pack_.expert(l, e, f, off, t);
                const size_t b = ExpertStore::round_slot(pack_.expert_block_bytes(t));
                key_bytes[static_cast<size_t>(l) * c.n_experts + e] = b;
                if (std::find(class_bytes.begin(), class_bytes.end(), b) == class_bytes.end()) {
                    class_bytes.push_back(b);
                    class_type.push_back(t);
                }
            }
        // ascending sizes
        std::vector<size_t> order(class_bytes.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return class_bytes[a] < class_bytes[b]; });
        std::vector<size_t> sb;
        std::vector<DType> st;
        for (size_t i : order) {
            sb.push_back(class_bytes[i]);
            st.push_back(class_type[i]);
        }
        class_bytes = sb;
        class_type = st;
        class_count.assign(class_bytes.size(), 0);
        for (size_t k = 0; k < key_class.size(); ++k) {
            const size_t ci = static_cast<size_t>(std::find(class_bytes.begin(), class_bytes.end(), key_bytes[k]) -
                                                  class_bytes.begin());
            key_class[k] = static_cast<uint8_t>(ci);
            ++class_count[ci];
        }
    }
    const bool classed = class_bytes.size() > 1;
    class_names_.clear();
    for (DType t : class_type) class_names_.push_back(dtype_name(t));
    // + batch scratch and read buffers, + the first chunk of the context
    // + the NPU's copies of the dense matrices (hmx4, beside the GPU's tile8 ones) and its activation buffers
    const uint64_t npu_bytes = npu_.g ? static_cast<uint64_t>(npu_.bytes(npu_.g)) + kNpuBufferBytes : 0;
    const uint64_t fixed = pack_.dense_bytes() + state_bytes + (32ull << 20) + kv_chunk_bytes_ +
                           (gpu8_ ? static_cast<uint64_t>(gpu8_->allocated()) : 0) + npu_bytes;
    // pinned at once at most: this layer's experts and the next layer's guesses
    const size_t min_slots = static_cast<size_t>(2 * c.top_k + 2);
    min_slots_ = min_slots;
    size_t slots = static_cast<size_t>(c.n_layers) * c.n_experts;              // no budget: room for every expert
    if (opt.memory_budget) {
        if (opt.memory_budget < fixed + min_slots * slot) {
            err = "the memory budget (" + std::to_string(opt.memory_budget >> 20) + " MB) is below what the model needs: " +
                  std::to_string((fixed + min_slots * slot) >> 20) + " MB";
            return false;
        }
    }
    if (opt.memory_budget && c.dense) {
        // no experts: the budget bounds the context only
        const uint64_t room = opt.memory_budget - fixed - min_slots * slot;
        ctx_ = std::min(ctx_, (1 + static_cast<int>(room / std::max<size_t>(1, kv_chunk_bytes_))) * kv_chunk_);
    } else if (opt.memory_budget) {
        slots = std::min<size_t>(slots, static_cast<size_t>((opt.memory_budget - fixed) / slot));
        // every further chunk of context takes its memory from the expert cache; the longest context is the one
        // that still leaves the cache its minimum
        slots_per_chunk_ = (kv_chunk_bytes_ + slot - 1) / slot;
        const size_t spare = slots > min_slots ? slots - min_slots : 0;
        const int max_chunks = 1 + static_cast<int>(spare / std::max<size_t>(1, slots_per_chunk_));
        ctx_ = std::min(ctx_, max_chunks * kv_chunk_);
    }
    slots = std::max(slots, min_slots);
    budgeted_ = opt.memory_budget != 0 && !c.dense;   // (the cache lends a prompt's scratch only with experts)
    slot_bytes_ = slot;
    if (budgeted_ && opt.memory_budget > fixed + slots * slot) spare_bytes_ = opt.memory_budget - fixed - slots * slot;
    // Size classes: the cache's bytes split by each class's expected demand in bytes - its share of the picks
    // (geode.json expert_class_shares, by format, from tools/bitmap.py; else every expert alike) times its block size;
    // a class never gets more slots than it has experts, and the largest keeps what a layer and its guesses pin
    std::vector<size_t> class_slots;
    if (classed) {
        std::vector<double> want(class_bytes.size());
        const json::Value &sh = pack_.manifest()["expert_class_shares"];
        for (size_t ci = 0; ci < class_bytes.size(); ++ci) {
            const json::Value &v = sh[class_names_[ci]];
            const double share = v.is_number() ? v.f64() : static_cast<double>(class_count[ci]);
            want[ci] = std::max(0.0, share) * static_cast<double>(class_bytes[ci]);
        }
        class_slots.assign(class_bytes.size(), 0);
        if (!budgeted_) {
            class_slots = class_count;   // no budget: room for every expert
        } else {
            const uint64_t total = opt.memory_budget - fixed;
            const size_t top = class_bytes.size() - 1;
            class_slots[top] = min_slots;
            uint64_t left = total > min_slots * class_bytes[top] ? total - min_slots * class_bytes[top] : 0;
            // the rest by demand; a class at its count gives what it cannot use back to the others
            std::vector<bool> full(class_bytes.size(), false);
            for (size_t round = 0; round < class_bytes.size() && left >= class_bytes[0]; ++round) {
                double w = 0;
                for (size_t ci = 0; ci < class_bytes.size(); ++ci)
                    if (!full[ci]) w += want[ci];
                if (w <= 0) break;
                uint64_t used = 0;
                for (size_t ci = 0; ci < class_bytes.size(); ++ci) {
                    if (full[ci]) continue;
                    const size_t more = static_cast<size_t>(static_cast<double>(left) * want[ci] / w / class_bytes[ci]);
                    const size_t room = class_count[ci] > class_slots[ci] ? class_count[ci] - class_slots[ci] : 0;
                    const size_t got = std::min(more, room);
                    if (got == room) full[ci] = true;
                    class_slots[ci] += got;
                    used += static_cast<uint64_t>(got) * class_bytes[ci];
                }
                left -= used;
                if (!used) break;
            }
        }
        uint64_t cache = 0;
        slots = 0;
        for (size_t ci = 0; ci < class_bytes.size(); ++ci) {
            cache += static_cast<uint64_t>(class_slots[ci]) * class_bytes[ci];
            slots += class_slots[ci];
        }
        spare_bytes_ = budgeted_ && opt.memory_budget > fixed + cache ? opt.memory_budget - fixed - cache : 0;
        if (budgeted_) {
            // the context's chunks come out of the cache by the bytes, a whole largest slot at a time
            const uint64_t chunk = static_cast<uint64_t>(slots_per_chunk_) * slot;
            const uint64_t room = cache > static_cast<uint64_t>(min_slots) * slot ? cache - static_cast<uint64_t>(min_slots) * slot : 0;
            const int max_chunks = 1 + static_cast<int>(room / std::max<uint64_t>(1, chunk));
            ctx_ = std::min(ctx_, max_chunks * kv_chunk_);
        }
    }
    // the expert files again, for the I/O threads: O_DIRECT on the phone (one DMA into the slot, no page cache)
    for (int f = 0; f < pack_.file_count(); ++f) {
        auto rf = std::make_unique<RandomFile>();
        if (!rf->open(pack_.path(f), err, opt.direct_io)) return false;
        expert_files_.push_back(std::move(rf));
    }
    prefetch_on_ = opt.prefetch;
    tail_cfg_ = opt.tail_skip >= 0                  ? opt.tail_skip
                : std::getenv("GEODE_TAIL_SKIP") ? std::atoi(std::getenv("GEODE_TAIL_SKIP"))
                                                 : -1;
    k_ = opt.experts_per_token > 0 ? std::min(opt.experts_per_token, cfg_.top_k) : cfg_.top_k;
    tail_skip_ = tail_cfg_ >= 0 ? tail_cfg_ : 0;   // off unless asked: every expert the router picks
    route_prior_ = std::getenv("GEODE_ROUTE_PRIOR") ? static_cast<float>(std::atof(std::getenv("GEODE_ROUTE_PRIOR")))
                                                    : opt.route_prior;
    if (!(route_prior_ > 0.0f)) route_prior_ = 0.0f;
    tail_renorm_ = opt.tail_renorm || (std::getenv("GEODE_TAIL_RENORM") && std::atoi(std::getenv("GEODE_TAIL_RENORM")) != 0);
    // waiting for the tail's picks still being read: measured no clearer faster (story 22.6 -> 23.5, thinking 21.5
    // -> 20.7 tokens/s, noisy) for 1.7x the experts left out - kept; GEODE_TAIL_WAIT=0 leaves them out
    tail_wait_ = !std::getenv("GEODE_TAIL_WAIT") || std::atoi(std::getenv("GEODE_TAIL_WAIT")) != 0;
    // with the Adreno 7xx+ path and tile8 experts the cache lives in shared memory, in pieces the GPU takes whole.
    // Its last GEODE_RESERVE_TAIL_MB (256) in pieces of GEODE_RESERVE_PIECE_MB (64) of their own: that memory belongs
    // to the driver (madvise cannot give it back), so the reserve (reserve_memory) frees whole pieces of it
    open_clock.mark("before the expert cache");
    std::vector<ExpertStore::Slab> slabs;
    tail_.clear();
    // (the GPU's int8 path takes q4 tile8 experts only: a pack of other or mixed formats keeps them on the CPU)
    if (gpu8_ && pack_.experts_tile8() && classed) gpu_status_ += " (experts of several formats: on the CPU)";
    if (gpu8_ && pack_.experts_tile8() && !classed && class_type[0] == DType::Q4) {
        const size_t per_slab = gpu8_->max_piece() / slot;
        const auto mb = [](const char *name, size_t def) {
            const char *v = std::getenv(name);
            return (v ? static_cast<size_t>(std::max(0, std::atoi(v))) : def) << 20;
        };
        const size_t tail_per = std::max<size_t>(1, std::min(per_slab, mb("GEODE_RESERVE_PIECE_MB", 64) / slot));
        size_t tail_n = per_slab ? mb("GEODE_RESERVE_TAIL_MB", 256) / (tail_per * slot) : 0;
        if (slots < tail_n * tail_per + 4 * min_slots) tail_n = 0;   // a cache that small keeps every slot
        std::string e;
        bool ok = per_slab > 0;
        // The NPU's share (moe_gpu8): the first pieces as dma-bufs the DSP maps (gnpu_shared_alloc, one at a time: it
        // keeps a list) and the GPU takes without a copy (Gpu8::import) - as many as the DSP's 3840 MB window holds
        // beside what the NPU has mapped already (its matrices' copies), the experts' x / y and a margin; 256 MB each,
        // to fill the window closely (GEODE_NPU_MOE=0: none; GEODE_NPU_MOE_MB: at most that)
        size_t npu_room = 0;
        const size_t npu_per = std::max<size_t>(1, (size_t{256} << 20) / slot);
        if (ok && npu_.g && npu_.moe && npu_.shared && npu_.bytes && npu_.arch >= 79 &&
            !(std::getenv("GEODE_NPU_MOE") && std::atoi(std::getenv("GEODE_NPU_MOE")) == 0)) {
            const size_t xy = (static_cast<size_t>(kMaxBatch) * static_cast<size_t>(c.hidden) * 4 + 127) / 128 * 128;
            int io_fd = -1;
            npu_.moe_io = static_cast<uint8_t *>(npu_.shared(npu_.g, 2 * xy, 1, &io_fd));
            npu_.moe_io_fd = io_fd;
            // (the margin: what the prompt's products map for each call - their activations and results, ~100 MB -
            // and more; with 256 MB they failed)
            const size_t mapped = npu_.bytes(npu_.g), window = size_t{3840} << 20, margin = size_t{512} << 20;
            npu_room = npu_.moe_io && window > mapped + margin ? window - mapped - margin : 0;
            if (const char *v = std::getenv("GEODE_NPU_MOE_MB"))
                npu_room = std::min(npu_room, static_cast<size_t>(std::max(0, std::atoi(v))) << 20);
        }
        // the pieces' sizes in slots: the NPU's, whole slabs, then the tail's; allocated on several threads at once (the
        // driver clears every page it hands out: 7.4 GB took 2.6 s on one thread on the 8 Elite), kept in this order
        std::vector<size_t> plan;
        size_t n_npu = 0;
        for (size_t left = ok ? slots - tail_n * tail_per : 0; left > 0;) {
            const bool npu = n_npu + 1 < MOE_MAX_FDS && npu_per * slot <= npu_room && left >= npu_per;
            plan.push_back(npu ? npu_per : std::min(left, per_slab));
            if (npu) {
                npu_room -= npu_per * slot;
                ++n_npu;
            }
            left -= plan.back();
        }
        const size_t n_whole = plan.size();
        if (ok) plan.insert(plan.end(), tail_n, tail_per);
        std::vector<uint8_t *> got(plan.size(), nullptr);
        std::vector<int> npu_fd(plan.size(), -1);
        for (size_t i = 0; i < n_npu; ++i) {
            int fd = -1;
            std::string pe;
            uint8_t *mem = static_cast<uint8_t *>(npu_.shared(npu_.g, plan[i] * slot, 1, &fd));
            if (!mem) break;   // (the rest from the GPU's driver as ever)
            static const bool dbg = std::getenv("GEODE_NPU_MOE_DEBUG") != nullptr;
            const auto touch = [&](const char *when) {
                std::fprintf(stderr, "npu slab %zu %s: touching %zu MB\n", i, when, (plan[i] * slot) >> 20);
                for (size_t o = 0; o < plan[i] * slot; o += 4096) static_cast<volatile uint8_t *>(mem)[o] = 0;
                std::fprintf(stderr, "npu slab %zu %s: ok\n", i, when);
            };
            if (dbg) touch("before the import");
            if (!gpu8_->import(fd, mem, plan[i] * slot, pe)) {   // (the dma-buf stays the NPU's until it closes)
                npu_.status += " (experts: " + pe + ")";
                break;
            }
            if (dbg) touch("after the import");
            got[i] = mem;
            npu_fd[i] = fd;
        }
        {
            const char *st = std::getenv("GEODE_SLAB_THREADS");   // measuring: how many (default 8: 2.4 s -> 0.8)
            const size_t n_threads = std::min(plan.size(), static_cast<size_t>(st ? std::max(1, std::atoi(st)) : 8));
            std::atomic<size_t> next{0};
            std::mutex err_mu;
            auto work = [&] {
                for (size_t i; (i = next.fetch_add(1)) < plan.size();) {
                    if (got[i]) continue;   // the NPU's
                    std::string pe;
                    got[i] = gpu8_->alloc(plan[i] * slot, pe);
                    if (!got[i]) {
                        std::lock_guard<std::mutex> lock(err_mu);
                        if (e.empty()) e = pe;
                    }
                }
            };
            std::vector<std::thread> threads;
            for (size_t t = 1; t < n_threads; ++t) threads.emplace_back(work);
            work();
            for (std::thread &t : threads) t.join();
        }
        for (size_t i = 0; i < plan.size(); ++i) {
            if (!got[i]) {
                ok = false;
                continue;
            }
            slabs.push_back({got[i], plan[i]});   // (freed below if another piece failed)
            if (i >= n_whole) {
                const size_t t = i - n_whole;
                TailPiece tp;
                for (size_t j = 0; j < tail_per; ++j)
                    tp.slots.push_back(static_cast<int>(slots - (tail_n - t) * tail_per + j));
                tp.bytes = tail_per * slot;
                tp.mem = got[i];
                tail_.push_back(tp);
            }
        }
        if (!ok) {
            gpu_status_ += " (expert cache: " + e + ")";
            for (const auto &b : slabs) gpu8_->free(b.mem);
            slabs.clear();
            tail_.clear();
        }
        gpu8_experts_ = !slabs.empty();
        for (size_t i = 0; i < plan.size() && gpu8_experts_; ++i)
            if (npu_fd[i] >= 0) {
                npu_.moe_slabs.push_back({got[i], plan[i] * slot, npu_fd[i]});
                if (std::getenv("GEODE_NPU_MOE_DEBUG")) std::fprintf(stderr, "npu slab %zu: %p + %zu MB (fd %d, %zu slots of %zu)\n", i, static_cast<void *>(got[i]), (plan[i] * slot) >> 20, npu_fd[i], plan[i], slot);
            }
        npu_.moe_on = !npu_.moe_slabs.empty();
    }
    open_clock.mark("expert cache: GPU slabs");
    const bool stored =
        classed ? store_.init_classes(class_slots, class_bytes, key_class, static_cast<uint32_t>(EL * c.n_experts),
                                      opt.io_threads > 0 ? opt.io_threads : default_io_threads(), little_cores())
                : store_.init(slots, slot, static_cast<uint32_t>(EL * c.n_experts),
                              opt.io_threads > 0 ? opt.io_threads : default_io_threads(), little_cores(), slabs);
    open_clock.mark("expert cache: store");
    if (!stored) {
        err = "not enough memory for the expert cache";
        return false;
    }
    store_.set_floor(min_slots);
    // a block's read split over 4 I/O threads: on UFS 4.0 a decode token waited 30 -> 25 ms for missed experts,
    // 12.0 -> 12.8 tokens/s (GEODE_READ_PARTS: another count, for measuring)
    store_.set_read_parts(std::getenv("GEODE_READ_PARTS") ? std::atoi(std::getenv("GEODE_READ_PARTS")) : 4);
    prefetch_.assign(static_cast<size_t>(c.n_layers + c.mtp_layers), {});
    conv_n_ = static_cast<size_t>(n_gdn) * conv_dim * (c.conv_kernel - 1);
    gdn_n_ = static_cast<size_t>(n_gdn) * c.gdn_v_heads * c.gdn_k_dim * c.gdn_v_dim;
    if (gpu8_ && gpu8_experts_ && !exact_) {
        // the GPU may generate: the states in shared memory (a prompt's CPU arithmetic reads them there too)
        std::string e;
        conv_st_ = reinterpret_cast<float *>(gpu8_->alloc(conv_n_ * sizeof(float), e));
        gdn_st_ = conv_st_ ? reinterpret_cast<float *>(gpu8_->alloc(gdn_n_ * sizeof(float), e)) : nullptr;
        if (!gdn_st_) {
            err = "not enough memory for the states: " + e;
            return false;
        }
        std::memset(conv_st_, 0, conv_n_ * sizeof(float));
        std::memset(gdn_st_, 0, gdn_n_ * sizeof(float));
    } else {
        if (npu_.dec_planned) {
            // the DSP may generate: the states in memory both sides map, then the decoder's scratch
            auto &cf = *reinterpret_cast<dec_cfg_t *>(npu_.cfg.data());
            cf.n_attn = n_attn_, cf.kv_chunk = kv_chunk_, cf.max_chunks = (ctx_ + kv_chunk_ - 1) / kv_chunk_;
            cf.kv_i8 = kv_i8_ ? 1 : 0;
            // two sets: a verified pair's first token's states go into the other (the CPU's backups, conv_bk_)
            const size_t set = ((conv_n_ + gdn_n_) * sizeof(float) + 127) / 128 * 128, states = 2 * set;
            const long long scratch = npu_.dec_size(npu_.g, npu_.cfg.data(), static_cast<int>(npu_.cfg.size()));
            npu_.s = scratch > 0 ? static_cast<uint8_t *>(npu_.shared(npu_.g, states + static_cast<size_t>(scratch), 1,
                                                                      &npu_.s_fd))
                                 : nullptr;
            if (npu_.s) {
                for (int i = 0; i < 2; ++i) {
                    npu_.conv_set[i] = reinterpret_cast<float *>(npu_.s + i * set);
                    npu_.gdn_set[i] = npu_.conv_set[i] + conv_n_;
                    cf.conv_state[i] = i * set;
                    cf.gdn_state[i] = i * set + conv_n_ * sizeof(float);
                }
                conv_st_ = npu_.conv_set[0], gdn_st_ = npu_.gdn_set[0];
                conv_bk_ = npu_.conv_set[1], gdn_bk_ = npu_.gdn_set[1];
                std::memset(npu_.s, 0, states);
                cf.scratch = states, cf.scratch_bytes = static_cast<uint64_t>(scratch);
            } else {
                npu_.dec_planned = false;
                npu_.dec_status = "off (no shared memory for the states)";
            }
        }
        if (!npu_.dec_planned) {
            try {
                conv_state_.assign(conv_n_, 0.0f);
                gdn_state_.assign(gdn_n_, 0.0f);
            } catch (const std::bad_alloc &) {
                err = "not enough memory for the context";
                return false;
            }
            conv_st_ = conv_state_.data();
            gdn_st_ = gdn_state_.data();
        }
    }
    open_clock.mark("states");
    if (!new_kv_chunk(err)) return false;
    open_clock.mark("kv chunk");
    if (npu_.dec_planned) {
        std::string why;
        if (!open_npu_decode(why)) npu_.dec_status = "off (" + why + ")";
        open_clock.mark("npu decoder");
    }
    // the per-token scratch (size_batch): its floats, and the widest activation an int8 product quantizes
    {
        const size_t H = static_cast<size_t>(c.hidden), cw = static_cast<size_t>(std::max(vd, c.n_heads * c.head_dim));
        token_floats_ = 7 * H + static_cast<size_t>(conv_dim) + static_cast<size_t>(vd) + 2 * static_cast<size_t>(c.gdn_v_heads) +
                        cw + static_cast<size_t>(c.n_heads) * c.head_dim * 2 + 2 * kv_row + static_cast<size_t>(c.n_experts) +
                        2 * static_cast<size_t>(c.shared_ff) + 2 * static_cast<size_t>(c.moe_ff) +
                        static_cast<size_t>(c.top_k) * H + 2 * static_cast<size_t>(c.top_k);
        max_cols_ = std::max(H, cw);
    }
    if (!size_batch(kSmallBatch, err)) return false;
    open_clock.mark("scratch");
    const int half = c.rope_dim / 2;
    inv_freq_.resize(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i)
        inv_freq_[static_cast<size_t>(i)] =
            1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / static_cast<float>(c.rope_dim));
    // the cache starts as it ended last time: the hottest blocks of the profile read in the background, after any
    // read the model asks for (GEODE_WARM_WAIT=1: open() waits for them - measuring a warm cache)
    if (!std::getenv("GEODE_NO_WARM")) {
        const int E = c.n_experts;
        const size_t queued = store_.load_profile(
            pack_.dir() + "/cache_profile.bin",
            [&](uint32_t key) {
                DType t;
                return expert_read(static_cast<int>(key) / E, static_cast<int>(key) % E, t);
            },
            store_.slots());
        if (queued && std::getenv("GEODE_WARM_WAIT"))
            while (store_.preloads_left() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        warm_queued_ = queued;
    }
    open_clock.mark("cache profile");
    if (gpu8_ && gpu8_experts_ && conv_st_ != conv_state_.data() && !std::getenv("GEODE_NO_GPU_DECODE")) {
        std::string why;
        dec8_ = dec8_ok_ = setup_decode8(why);
        gpu_status_ += dec8_ ? "; generating on the GPU" : "; generating on the CPU (" + why + ")";
        if (dec8_ && std::getenv("GEODE_GPU8_SELFTEST")) {   // measuring: lm_head and a GDN qkv alone
            gpu8_->dec_selftest(lm_head_, 10);
            for (const Layer &L : layers_)
                if (!L.attn) {
                    gpu8_->dec_selftest(L.qkv, 20);
                    break;
                }
        }
    }
    // (only with the states in the process's memory: a rejected draft swaps them with their backups)
    if (mtp_ && !dec8_ok_ && conv_st_ == conv_state_.data() &&
        !(std::getenv("GEODE_CPU_SPEC") && std::atoi(std::getenv("GEODE_CPU_SPEC")) == 0))
        cpu_spec_ = setup_cpu_spec();
    open_clock.mark("decode setup");
    apply_experts();
    open_clock.mark("apply experts");
    return true;
}

bool Model::setup_cpu_spec() {
    // the states' backups: as much again as the states (Qwen3.6: 63 MB), taken from the expert cache's budget
    try {
        conv_bk_mem_.assign(conv_n_, 0.0f);
        gdn_bk_mem_.assign(gdn_n_, 0.0f);
    } catch (const std::bad_alloc &) {
        conv_bk_mem_ = {};
        gdn_bk_mem_ = {};
        return false;
    }
    conv_bk_ = conv_bk_mem_.data();
    gdn_bk_ = gdn_bk_mem_.data();
    // the draft head: lm_head's rows of draft_vocab.bin's tokens (a draft is one of them; without the file, the whole
    // vocabulary - a full lm_head read for every draft)
    std::vector<int32_t> ids;
    if (std::FILE *f = std::fopen((pack_.dir() + "/draft_vocab.bin").c_str(), "rb")) {
        int32_t id;
        while (std::fread(&id, 4, 1, f) == 1)
            if (id >= 0 && id < cfg_.vocab) ids.push_back(id);
        std::fclose(f);
    }
    if (!ids.empty()) {
        const size_t rb = lm_head_.row_bytes;
        draft_mem_ = aligned_bytes(ids.size() * rb);
        if (draft_mem_) {
            for (size_t r = 0; r < ids.size(); ++r) {
                if (lm_head_.quad) quad_row(lm_head_, ids[r], draft_mem_.get() + r * rb);
                else std::memcpy(draft_mem_.get() + r * rb, lm_head_.data + static_cast<size_t>(ids[r]) * rb, rb);
            }
            draft_head_ = lm_head_;
            draft_head_.rows = static_cast<int64_t>(ids.size());
            draft_head_.data = draft_mem_.get();
            draft_head_.quad = draft_head_.tile8 = false;
            draft_ids_ = std::move(ids);
        }
    }
    // both from the budget: its spare first (a budget beyond every expert), then the expert cache's coldest slots,
    // taken for good (a reset gives retired slots back, taken ones it does not); a cache that cannot spare them
    // keeps them, and no speculation
    if (budgeted_ && slot_bytes_) {
        const uint64_t need = spec_bytes(), from_spare = std::min(need, spare_bytes_);
        if (need > from_spare) {
            spec_slots_ = store_.take_bytes(need - from_spare, min_slots_ + static_cast<size_t>(cfg_.top_k));
            uint64_t got = 0;
            for (int s : spec_slots_) got += store_.slot_size(s);
            if (got < need - from_spare) {
                store_.give_back(spec_slots_);
                spec_slots_.clear();
                conv_bk_mem_ = {};
                gdn_bk_mem_ = {};
                conv_bk_ = gdn_bk_ = nullptr;
                draft_mem_.reset();
                draft_head_ = QView{};
                draft_ids_.clear();
                return false;
            }
        }
        spare_bytes_ -= from_spare;
    }
    return true;
}

uint64_t Model::spec_bytes() const {
    return static_cast<uint64_t>(conv_bk_mem_.size() + gdn_bk_mem_.size()) * sizeof(float) +
           (draft_mem_ ? static_cast<uint64_t>(draft_ids_.size()) * lm_head_.row_bytes : 0);
}

bool Model::new_kv_chunk(std::string &err) {
    if (npu_.dec_planned) {   // memory the DSP maps too, kept for reuse after a reset; the DSP told where it goes
        uint8_t *m = nullptr;
        if (!kv_spare_.empty()) {
            m = kv_spare_.back();
            kv_spare_.pop_back();
        } else {
            int fd = -1;
            m = static_cast<uint8_t *>(npu_.shared(npu_.g, kv_chunk_bytes_, 1, &fd));
            if (!m) {
                err = "not enough memory to grow the context";
                return false;
            }
            npu_.kv_fds[m] = fd;
        }
        kv_ptrs_.push_back(m);
        if (npu_.dec && npu_.dec_kv(npu_.g, static_cast<int>(kv_ptrs_.size()) - 1, npu_.kv_fds[m])) {
            err = "the DSP could not take a KV chunk";
            return false;
        }
        return true;
    }
    if (gpu8_ && gpu8_experts_ && !exact_) {   // shared memory, kept for reuse after a reset
        if (!kv_spare_.empty()) {
            kv_ptrs_.push_back(kv_spare_.back());
            kv_spare_.pop_back();
            return true;
        }
        std::string e;
        uint8_t *m = gpu8_->alloc(kv_chunk_bytes_, e);
        if (!m) {
            err = "not enough memory to grow the context: " + e;
            return false;
        }
        kv_ptrs_.push_back(m);
        return true;
    }
    AlignedBuffer b = aligned_bytes(kv_chunk_bytes_);
    if (!b) {
        err = "not enough memory to grow the context";
        return false;
    }
    kv_ptrs_.push_back(b.get());
    kv_chunks_.push_back(std::move(b));
    return true;
}

bool Model::setup_decode8(std::string &why) {
    const auto &c = cfg_;
    const int H = c.hidden, E = c.n_experts, nv = c.gdn_v_heads;
    const int kd = c.gdn_k_heads * c.gdn_k_dim, vd = nv * c.gdn_v_dim, conv_dim = 2 * kd + vd;
    if (c.shared_ff != c.moe_ff) {
        why = "the shared expert's width differs from the routed ones'";
        return false;
    }
    // a pack with an MTP layer drafts tokens the GPU verifies two at once (GEODE_SPEC_K: batches of up to that many,
    // 2 .. Gpu8::kMaxTokens): a backup of the states after each token of a batch but its last, for a draft that falls
    // (GEODE_VERIFY=1: without an MTP layer too, for testing)
    spec_t_ = 2;
    if (const char *v = std::getenv("GEODE_SPEC_K")) spec_t_ = std::clamp(std::atoi(v), 2, Gpu8::kMaxTokens);
    if ((mtp_ || std::getenv("GEODE_VERIFY")) && !gdn_bk_) {
        std::string e;
        const size_t nb = static_cast<size_t>(spec_t_ - 1);
        conv_bk_ = reinterpret_cast<float *>(gpu8_->alloc(nb * conv_n_ * sizeof(float), e));
        gdn_bk_ = conv_bk_ ? reinterpret_cast<float *>(gpu8_->alloc(nb * gdn_n_ * sizeof(float), e)) : nullptr;
        if (!gdn_bk_) conv_bk_ = nullptr;
    }
    for (const Layer &L : layers_)
        for (const QView *m : {&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.sh_gate, &L.sh_up, &L.sh_down})
            if (m->data && !gpu8_->holds(*m)) {
                why = "a dense matrix outside the GPU's memory";
                return false;
            }
    if (!gpu8_->holds(lm_head_)) {
        why = "lm_head outside the GPU's memory";
        return false;
    }
    // the MTP layer on the GPU too when its matrices are there (drafting)
    bool mtp_here = mtp_ && gpu8_->holds(mtp_fc_);
    for (const QView *m : {&mtp_layer_.q, &mtp_layer_.k, &mtp_layer_.v, &mtp_layer_.o, &mtp_layer_.sh_gate,
                           &mtp_layer_.sh_up, &mtp_layer_.sh_down})
        mtp_here = mtp_here && m->data && gpu8_->holds(*m);
    std::vector<const Layer *> all;
    for (const Layer &L : layers_) all.push_back(&L);
    if (mtp_here) all.push_back(&mtp_layer_);
    // the small tensors into one piece, 64-byte aligned
    size_t total = 0;
    auto room = [&](size_t bytes) { total += (bytes + 63) / 64 * 64; };
    const size_t fH = static_cast<size_t>(H) * 4;
    for (const Layer *Lp : all) {
        const Layer &L = *Lp;
        room(fH);
        room(fH);
        room(static_cast<size_t>(E + 1) * H * 2);
        if (L.attn) {
            room(static_cast<size_t>(c.head_dim) * 4);
            room(static_cast<size_t>(c.head_dim) * 4);
        } else {
            room(static_cast<size_t>(2 * nv) * H * 2);
            room(static_cast<size_t>(nv) * 4);
            room(static_cast<size_t>(nv) * 4);
            room(static_cast<size_t>(conv_dim) * c.conv_kernel * 4);
            room(static_cast<size_t>(c.gdn_v_dim) * 4);
        }
    }
    room(fH);
    if (mtp_here) {   // the MTP layer's final norm and its two input norms
        room(fH);
        room(fH);
        room(fH);
    }
    uint8_t *base = gpu8_->dec_params(total, why);
    if (!base) return false;
    size_t at = 0;
    auto put = [&](const void *src, size_t bytes) -> uint8_t * {
        uint8_t *d = base + at;
        std::memcpy(d, src, bytes);
        at += (bytes + 63) / 64 * 64;
        return d;
    };
    dec_layers_.clear();
    for (const Layer *Lp : all) {
        const Layer &L = *Lp;
        Gpu8::DecodeLayer D;
        D.attn = L.attn;
        D.index = L.index;
        D.qkv = L.qkv, D.z = L.z, D.out = L.out, D.q = L.q, D.k = L.k, D.v = L.v, D.o = L.o;
        D.sh_gate = L.sh_gate, D.sh_up = L.sh_up, D.sh_down = L.sh_down;
        D.in_norm = reinterpret_cast<const float *>(put(L.in_norm, fH));
        D.post_norm = reinterpret_cast<const float *>(put(L.post_norm, fH));
        uint8_t *r = put(L.router, static_cast<size_t>(E) * H * 2);   // the router's rows, then the shared gate's
        std::memcpy(r + static_cast<size_t>(E) * H * 2, L.shared_gate, static_cast<size_t>(H) * 2);
        D.router = reinterpret_cast<const uint16_t *>(r);
        at += static_cast<size_t>(H) * 2;   // (the room counted E + 1 rows)
        at = (at + 63) / 64 * 64;
        if (L.attn) {
            D.q_norm = reinterpret_cast<const float *>(put(L.q_norm, static_cast<size_t>(c.head_dim) * 4));
            D.k_norm = reinterpret_cast<const float *>(put(L.k_norm, static_cast<size_t>(c.head_dim) * 4));
        } else {
            uint8_t *ab = put(L.a, static_cast<size_t>(nv) * H * 2);   // a's rows, then b's
            std::memcpy(ab + static_cast<size_t>(nv) * H * 2, L.b, static_cast<size_t>(nv) * H * 2);
            D.ab = reinterpret_cast<const uint16_t *>(ab);
            at += static_cast<size_t>(nv) * H * 2;
            at = (at + 63) / 64 * 64;
            D.A_log = reinterpret_cast<const float *>(put(L.A_log, static_cast<size_t>(nv) * 4));
            D.dt_bias = reinterpret_cast<const float *>(put(L.dt_bias, static_cast<size_t>(nv) * 4));
            D.conv = reinterpret_cast<const float *>(put(L.conv, static_cast<size_t>(conv_dim) * c.conv_kernel * 4));
            D.gnorm = reinterpret_cast<const float *>(put(L.gnorm, static_cast<size_t>(c.gdn_v_dim) * 4));
            D.conv_state = conv_st_ + static_cast<size_t>(L.index) * conv_dim * (c.conv_kernel - 1);
            D.gdn_state = gdn_st_ + static_cast<size_t>(L.index) * nv * c.gdn_k_dim * c.gdn_v_dim;
            if (gdn_bk_) {
                D.conv_bk = conv_bk_ + (D.conv_state - conv_st_);
                D.gdn_bk = gdn_bk_ + (D.gdn_state - gdn_st_);
            }
        }
        if (Lp == &mtp_layer_) dec_mtp_ = D;
        else dec_layers_.push_back(D);
    }
    dec_final_norm_ = reinterpret_cast<const float *>(put(final_norm_, fH));
    if (mtp_here) {
        dec_mtp_norm_ = reinterpret_cast<const float *>(put(mtp_norm_, fH));
        dec_pre_e_ = reinterpret_cast<const float *>(put(mtp_pre_e_, fH));
        dec_pre_h_ = reinterpret_cast<const float *>(put(mtp_pre_h_, fH));
    }
    if (at > total) {
        why = "the parameter piece overflowed";
        return false;
    }
    Gpu8::DecodeShape sh;
    sh.hidden = H, sh.vocab = c.vocab, sh.n_experts = E, sh.top_k = c.top_k, sh.moe_ff = c.moe_ff;
    sh.nh = c.n_heads, sh.nkv = c.n_kv_heads, sh.hd = c.head_dim, sh.half = c.rope_dim / 2;
    sh.n_attn = n_attn_, sh.kv_chunk = kv_chunk_, sh.max_ctx = ctx_;
    sh.nk = c.gdn_k_heads, sh.nv = nv, sh.dk = c.gdn_k_dim, sh.dv = c.gdn_v_dim, sh.conv_k = c.conv_kernel;
    sh.eps = c.eps;
    sh.kv_i8 = kv_i8_;
    sh.max_tokens = spec_t_;
    sh.bk_conv = static_cast<int>(conv_n_);
    sh.bk_gdn = static_cast<int>(gdn_n_);
    // GEODE_GUESSES: how many of the next layer's experts each route reads ahead (measuring)
    guesses_env_ = std::getenv("GEODE_GUESSES") != nullptr;
    sh.guesses = guesses_env_ ? std::clamp(std::atoi(std::getenv("GEODE_GUESSES")), 1, std::min(E, 16)) : c.top_k;
    dec_guesses_ = sh.guesses;
    // GEODE_GUESS2: also guess two layers ahead, this many (measuring: 4 or 8 read 7-29% more and took 0.6 tokens/s
    // off - the flash, not the lead, is the limit: a 1.77 MB block reads in 1.1 ms, the UFS tops out at ~2.3 GB/s)
    sh.guesses2 = std::getenv("GEODE_GUESS2") ? std::clamp(std::atoi(std::getenv("GEODE_GUESS2")), 0, std::min(E, 16)) : 0;
    dec_guesses2_ = sh.guesses2;
    open_clock.mark("decode: backups + params");
    if (!gpu8_->setup_decode(sh, inv_freq_.data(), why)) return false;
    open_clock.mark("decode: one-token program + buffers");
    if (gdn_bk_) {   // drafts verified two tokens at once
        std::string e;
        verify8_ = gpu8_->setup_verify(e);
        if (!verify8_) gpu_status_ += "; no verifying (" + e + ")";
    }
    open_clock.mark("decode: two-token program");
    mtp8_ = mtp_here;
    if (mtp8_) mlog_.resize(2 * static_cast<size_t>(c.vocab));
    return true;
}

bool Model::decode_gpu8(int token, float *logits, std::string &err) {
    const auto &c = cfg_;
    const int H = c.hidden;
    const double t0 = now_seconds();
    store_.next_token();
    if (!pack_.embedding(token, gpu8_->dec_x())) {
        err = "cannot read the embedding of token " + std::to_string(token);
        return false;
    }
    std::vector<int> held;
    auto fail = [&](const std::string &e) {
        for (int s : held) store_.release(s);
        err = "GPU generating: " + e;
        gpu_status_ = "failed: " + e;
        dec8_ = false;   // the CPU from the next token (this one is lost)
        return false;
    };
    for (int l = 0; l < c.n_layers; ++l) {
        const Gpu8::DecodeLayer *next = l + 1 < c.n_layers ? &dec_layers_[static_cast<size_t>(l + 1)] : nullptr;
        const Gpu8::DecodeLayer *next2 = dec_guesses2_ && l + 2 < c.n_layers ? &dec_layers_[static_cast<size_t>(l + 2)] : nullptr;
        std::string e;
        const double tf = now_seconds();
        if (!gpu8_->dec_front(dec_layers_[static_cast<size_t>(l)], next, pos_, kv_ptrs_, e, next2)) return fail(e);
        prof_.d_front += now_seconds() - tf;
        // the route, the picks pinned (the misses start reading) and computed as they come, the next layer's guesses
        if (!back_gpu8(l, layers_[static_cast<size_t>(l)], dec_layers_[static_cast<size_t>(l)], next != nullptr, 1, held, e,
                       next2 != nullptr))
            return fail(e);
    }
    std::string e;
    const double th = now_seconds();
    if (!gpu8_->dec_head(dec_final_norm_, lm_head_, logits, e)) return fail(e);
    prof_.d_head += now_seconds() - th;
    for (int s : held) store_.release(s);
    if (mtp_) {
        rmsnorm(gpu8_->dec_x(), final_norm_, hid_.data(), H, c.eps, true);
        last_hid_.assign(hid_.begin(), hid_.begin() + H);
    }
    pos_ += 1;
    prof_.gpu8 += now_seconds() - t0;
    return true;
}

bool Model::back_gpu8(int sl, const Layer &L, const Gpu8::DecodeLayer &D, bool has_next, int T,
                      std::vector<int> &held, std::string &err, bool has_next2) {
    const auto &c = cfg_;
    const int E = c.n_experts, K = k_, H = c.hidden, ff = c.moe_ff, G = dec_guesses_;
    const int G2 = has_next2 ? dec_guesses2_ : 0;
    const size_t UM = static_cast<size_t>(std::max(T, 2) * K);   // the picks' union, at most
    std::vector<int> ids(static_cast<size_t>(T * K)), gids(static_cast<size_t>(T * G)), uexp(UM), slots(UM), seen,
        gids2(static_cast<size_t>(T * std::max(G2, 1)));
    std::vector<DType> types(UM);
    std::vector<QView> gu(UM), dn(UM);
    std::vector<uint8_t> skip(static_cast<size_t>(K));
    const double tr = now_seconds();
    if (!gpu8_->dec_route(ids.data(), gids.data(), err, G2 ? gids2.data() : nullptr)) return false;
    const double tp = now_seconds();
    prof_.d_route += tp - tr;
    for (int s : held) store_.release(s);
    held.clear();
    // each token's picks (its tail's misses left out) into the union; token t's pick k -> slot (um[t] >> 6k) & 63
    uint32_t skipm[Gpu8::kMaxTokens] = {};
    uint64_t um[Gpu8::kMaxTokens] = {};
    int U = 0;
    for (int t = 0; t < T; ++t) {
        std::fill(skip.begin(), skip.end(), 0);
        skip_tail(sl, ids.data() + static_cast<size_t>(t) * K, nullptr, skip.data());
        for (int k = 0; k < K; ++k) {
            if (skip[static_cast<size_t>(k)]) {
                skipm[t] |= 1u << k;
                continue;
            }
            const int ex = ids[static_cast<size_t>(t * K + k)];
            int u = 0;
            while (u < U && uexp[static_cast<size_t>(u)] != ex) ++u;
            if (u == U) uexp[static_cast<size_t>(U++)] = ex;
            um[t] |= static_cast<uint64_t>(u) << (6 * k);
        }
    }
    uint32_t cached = 0;
    for (int u = 0; u < U; ++u)
        if (store_.resident(static_cast<uint32_t>(sl * E + uexp[static_cast<size_t>(u)]))) cached |= 1u << u;
    const double ca = now_seconds();
    prof_.c_decide += ca - tp;
    for (int u = 0; u < U; ++u) {
        const ExpertRead rd = expert_read(sl, uexp[static_cast<size_t>(u)], types[static_cast<size_t>(u)]);
        slots[static_cast<size_t>(u)] = store_.acquire(static_cast<uint32_t>(sl * E + uexp[static_cast<size_t>(u)]), rd, false);
        held.push_back(slots[static_cast<size_t>(u)]);
    }
    const double cb = now_seconds();
    prof_.c_pin += cb - ca;
    auto view = [&](int u, const uint8_t *blk) {
        const int ex = uexp[static_cast<size_t>(u)];
        const DType t = types[static_cast<size_t>(u)];
        const size_t gu_bytes = static_cast<size_t>(2 * ff) * row_bytes(t, H);
        gu[static_cast<size_t>(u)] = expert_view(t, 2 * ff, H, blk, L.expert_scales[2 * ex]);
        dn[static_cast<size_t>(u)] = expert_view(t, H, ff, blk + gu_bytes, L.expert_scales[2 * ex + 1]);
    };
    const double cc = cb;
    // the slots already in RAM (and the shared expert) to the GPU at once, the misses as their reads finish
    std::vector<int> ready, rest;
    for (int u = 0; u < U; ++u) (store_.ready(slots[static_cast<size_t>(u)]) ? ready : rest).push_back(u);
    // the tail not waited for either: a slot still being read that only the last ranks use is left out like a miss
    // there (its read goes on: the cache keeps it). Only with GEODE_TAIL_WAIT=0 (measuring; model.cpp, open)
    if (tail_skip_ > 0 && tail_skip_ < K && !tail_wait_ && !rest.empty()) {
        std::vector<int> keep;
        for (int u : rest) {
            bool low = true;
            for (int t = 0; t < T && low; ++t)
                for (int k = 0; k < tail_skip_ && low; ++k)
                    if (!(skipm[t] >> k & 1) && (um[t] >> (6 * k) & 63) == static_cast<uint64_t>(u)) low = false;
            if (!low) {
                keep.push_back(u);
                continue;
            }
            for (int t = 0; t < T; ++t)
                for (int k = tail_skip_; k < K; ++k)
                    if (!(skipm[t] >> k & 1) && (um[t] >> (6 * k) & 63) == static_cast<uint64_t>(u)) {
                        skipm[t] |= 1u << k;
                        ++tail_skipped_;
                        ++prof_.tail_unwaited;
                    }
        }
        rest.swap(keep);
    }
    for (int u : ready) {
        const uint8_t *blk = store_.wait(slots[static_cast<size_t>(u)]);
        if (!blk) {
            err = "cannot read expert " + std::to_string(uexp[static_cast<size_t>(u)]) + " of layer " + std::to_string(sl);
            return false;
        }
        view(u, blk);
    }
    const double t1 = now_seconds();
    prof_.d_pin += t1 - tp;
    prof_.c_views += t1 - cc;
    if (!gpu8_->dec_experts(D, gu.data(), dn.data(), ready, true, err)) return false;
    const double tw = now_seconds();
    prof_.d_back += tw - t1;
    prof_.c_enq += tw - t1;
    // the guesses made for this layer let go (what they caught is pinned now), the next layer's read ahead - after
    // the GPU has its work (the cache's victim search is not in the GPU's way)
    for (int s : prefetch_[static_cast<size_t>(sl)]) store_.release(s);
    prefetch_[static_cast<size_t>(sl)].clear();
    if (prefetch_on_ && has_next)
        for (int g : gids) {   // every token's guesses, once each
            if (std::find(seen.begin(), seen.end(), g) != seen.end()) continue;
            seen.push_back(g);
            DType t;
            const ExpertRead rd = expert_read(sl + 1, g, t);
            prefetch_[static_cast<size_t>(sl + 1)].push_back(store_.acquire(static_cast<uint32_t>((sl + 1) * E + g), rd, true));
        }
    if (prefetch_on_ && G2) {   // two layers ahead: read sooner, guessed less well
        seen.clear();
        for (int g : gids2) {
            if (std::find(seen.begin(), seen.end(), g) != seen.end()) continue;
            seen.push_back(g);
            DType t;
            const ExpertRead rd = expert_read(sl + 2, g, t);
            prefetch_[static_cast<size_t>(sl + 2)].push_back(store_.acquire(static_cast<uint32_t>((sl + 2) * E + g), rd, true));
        }
    }
    prof_.c_guess += now_seconds() - tw;
    for (int u : rest) {
        const double t = now_seconds();
        const uint8_t *blk = store_.wait(slots[static_cast<size_t>(u)]);
        if (!blk) {
            err = "cannot read expert " + std::to_string(uexp[static_cast<size_t>(u)]) + " of layer " + std::to_string(sl);
            return false;
        }
        view(u, blk);
        const bool flight = cached >> u & 1;
        (flight ? prof_.d_wait_flight : prof_.d_wait_miss) += now_seconds() - t;
        ++(flight ? prof_.d_n_flight : prof_.d_n_miss);
    }
    const double tb = now_seconds();
    prof_.expert_wait += tb - tw;
    prof_.d_pin += tb - tw;
    if (!gpu8_->dec_experts(D, gu.data(), dn.data(), rest, false, err) || !gpu8_->dec_combine(skipm, um, tail_renorm_, err))
        return false;
    prof_.d_back += now_seconds() - tb;
    return true;
}

namespace {
// a draft's confidence: the softmax probability of row best among rows logits
float softmax_at(const float *lg, int rows, int best) {
    const float m = lg[best];
    double sum = 0.0;
    for (int i = 0; i < rows; ++i) sum += std::exp(static_cast<double>(lg[i] - m));
    return static_cast<float>(1.0 / sum);
}
}  // namespace

bool Model::draft(const int *toks, int n, int pos, const float *hidden, int &draft, std::string &err, float *prob) {
    const auto &c = cfg_;
    if (npu_spec_ && npu_.dec) {
        // the MTP layer on the DSP: the next tokens' embeddings and the hidden rows go over
        if (n < 1 || n > DEC_T || pos < 0) {
            err = "bad draft";
            return false;
        }
        if (!reserve_context(pos + n, err)) return false;
        const double t0 = now_seconds();
        const int H = c.hidden;
        std::vector<float> e(static_cast<size_t>(n) * H), h(static_cast<size_t>(n) * H);
        int p3[3 * DEC_T];
        for (int t = 0; t < n; ++t) {
            const float *im = vision_.available() && toks[t] == vision_.shape().pad_token ? image_row(pos + t + 1) : nullptr;
            if (im) std::copy_n(im, H, e.data() + static_cast<size_t>(t) * H);
            else if (!pack_.embedding(toks[t], e.data() + static_cast<size_t>(t) * H)) {
                err = "cannot read the embedding of token " + std::to_string(toks[t]);
                return false;
            }
            std::memcpy(h.data() + static_cast<size_t>(t) * H, hidden ? hidden + static_cast<size_t>(t) * H : this->hidden(t),
                        static_cast<size_t>(H) * sizeof(float));
            rope_pos(pos + t, p3 + 3 * t);
        }
        double dsp = 0;
        float pr = 0.0f;
        if (npu_.dec_draft(npu_.g, e.data(), h.data(), n, H, pos, p3, &draft, &pr, &dsp)) {
            npu_spec_ = false;   // no more drafts (the main model is untouched)
            err = "the DSP's MTP pass failed";
            return false;
        }
        if (prob) *prob = pr;
        ++prof_.drafts;
        prof_.d_draft += now_seconds() - t0;
        return true;
    }
    if (cpu_spec_ && !dec8_) {
        // the MTP layer on the CPU: the hidden states given, or the last batch's (a verified pair's rows)
        if (n < 1 || n > 2 || pos < 0) {
            err = "bad draft";
            return false;
        }
        const double t0 = now_seconds();
        const int H = c.hidden;
        std::vector<const float *> hs(static_cast<size_t>(n));
        for (int t = 0; t < n; ++t) hs[static_cast<size_t>(t)] = hidden ? hidden + static_cast<size_t>(t) * H : this->hidden(t);
        // (the hidden rows live in hid_, which the MTP pass does not write)
        const bool sub = !draft_ids_.empty();
        const int rows = sub ? static_cast<int>(draft_ids_.size()) : c.vocab;
        mlog_.resize(static_cast<size_t>(rows));
        if (!forward_mtp(toks, hs.data(), n, pos, mlog_.data(), err, sub ? &draft_head_ : nullptr)) return false;
        const int best = static_cast<int>(std::max_element(mlog_.begin(), mlog_.begin() + rows) - mlog_.begin());
        draft = sub ? draft_ids_[static_cast<size_t>(best)] : best;
        if (prob) *prob = softmax_at(mlog_.data(), rows, best);
        ++prof_.drafts;
        prof_.d_draft += now_seconds() - t0;
        return true;
    }
    if (!mtp8_ || !dec8_ || (n >= 2 && !verify8_) || n < 1 || n > spec_t_ || pos < 0) {
        err = "no drafting on the GPU here";
        return false;
    }
    const int H = c.hidden, V = c.vocab;
    if (!reserve_context(pos + n, err)) return false;
    const double t0 = now_seconds();
    if (!gpu8_->set_tokens(n)) {
        err = "GPU: no program for " + std::to_string(n) + " tokens";
        return false;
    }
    struct One {
        Gpu8 *g;
        ~One() { g->set_tokens(1); }
    } one{gpu8_.get()};
    for (int t = 0; t < n; ++t)
        if (!pack_.embedding(toks[t], gpu8_->dec_x() + static_cast<size_t>(t) * H)) {
            err = "cannot read the embedding of token " + std::to_string(toks[t]);
            return false;
        }
    if (hidden) std::memcpy(gpu8_->dec_h(), hidden, static_cast<size_t>(n) * H * sizeof(float));
    std::vector<int> held;
    auto fail = [&](const std::string &e) {
        for (int s : held) store_.release(s);
        err = "GPU drafting: " + e;
        mtp8_ = false;   // no more drafts (the main model is untouched)
        return false;
    };
    std::string e;
    if (!gpu8_->dec_mtp_in(mtp_fc_, dec_pre_e_, dec_pre_h_, e) || !gpu8_->dec_front(dec_mtp_, nullptr, pos, kv_ptrs_, e) ||
        !back_gpu8(c.n_layers, mtp_layer_, dec_mtp_, false, n, held, e) ||
        !gpu8_->dec_head(dec_mtp_norm_, draft_ids_.empty() ? lm_head_ : draft_head_, mlog_.data(), e))
        return fail(e);
    for (int s : held) store_.release(s);
    const int rows = draft_ids_.empty() ? V : static_cast<int>(draft_ids_.size());
    const float *lg = mlog_.data() + static_cast<size_t>(n - 1) * rows;
    const int best = static_cast<int>(std::max_element(lg, lg + rows) - lg);
    draft = draft_ids_.empty() ? best : draft_ids_[static_cast<size_t>(best)];
    if (prob) *prob = softmax_at(lg, rows, best);
    chain_row_ = n - 1;   // dec_h's row n - 1: the MTP layer's normed output there, what a chained draft goes on from
    ++prof_.drafts;
    prof_.d_draft += now_seconds() - t0;
    return true;
}

bool Model::draft_next(int tok, int pos, int &draft, float *prob, std::string &err) {
    const auto &c = cfg_;
    if (!mtp8_ || !dec8_ || pos < 0) {
        err = "no chained drafts here (the GPU's MTP layer)";
        return false;
    }
    const int H = c.hidden, V = c.vocab;
    if (!reserve_context(pos + 1, err)) return false;
    const double t0 = now_seconds();
    // the last MTP pass's normed output (dec_head left it in dec_h) as this one's hidden state, the draft's embedding
    float *dh = gpu8_->dec_h();
    if (chain_row_ > 0) std::memmove(dh, dh + static_cast<size_t>(chain_row_) * H, static_cast<size_t>(H) * sizeof(float));
    if (!pack_.embedding(tok, gpu8_->dec_x())) {
        err = "cannot read the embedding of token " + std::to_string(tok);
        return false;
    }
    std::vector<int> held;
    auto fail = [&](const std::string &e) {
        for (int s : held) store_.release(s);
        err = "GPU drafting: " + e;
        mtp8_ = false;
        return false;
    };
    std::string e;
    if (!gpu8_->dec_mtp_in(mtp_fc_, dec_pre_e_, dec_pre_h_, e) || !gpu8_->dec_front(dec_mtp_, nullptr, pos, kv_ptrs_, e) ||
        !back_gpu8(c.n_layers, mtp_layer_, dec_mtp_, false, 1, held, e) ||
        !gpu8_->dec_head(dec_mtp_norm_, draft_ids_.empty() ? lm_head_ : draft_head_, mlog_.data(), e))
        return fail(e);
    for (int s : held) store_.release(s);
    const int rows = draft_ids_.empty() ? V : static_cast<int>(draft_ids_.size());
    const int best = static_cast<int>(std::max_element(mlog_.data(), mlog_.data() + rows) - mlog_.data());
    draft = draft_ids_.empty() ? best : draft_ids_[static_cast<size_t>(best)];
    if (prob) *prob = softmax_at(mlog_.data(), rows, best);
    chain_row_ = 0;
    ++prof_.drafts;
    prof_.d_draft += now_seconds() - t0;
    return true;
}

bool Model::save_cache_profile() { return store_.save_profile(pack_.dir() + "/cache_profile.bin"); }

bool Model::verify_n(const int *toks, int n, float *logits, std::string &err) {
    const auto &c = cfg_;
    if (npu_spec_ && npu_.dec) {
        // the pair on the DSP: every weight read once for both, the first token's states into the other set
        if (n != 2) {
            err = "the DSP verifies pairs only";
            return false;
        }
        if (!reserve_context(pos_ + 2, err)) return false;
        const double t0 = now_seconds();
        if (!decode_npu(toks, 2, logits, err)) return false;
        last_batch_ = 2;
        ++prof_.verified;
        prof_.d_verify += now_seconds() - t0;
        return true;
    }
    if (cpu_spec_ && !dec8_) {
        // both tokens in one batch: every dense weight read once for the two, each token's arithmetic the one-token
        // arithmetic (the same bits as two forward() calls), the states after the first kept for a rejected draft
        if (n != 2) {
            err = "the CPU verifies pairs only";
            return false;
        }
        const double t0 = now_seconds();
        verify_keep_ = true;
        const bool ok = forward_batch(toks, 2, logits, true, err);
        verify_keep_ = false;
        if (!ok) return false;
        last_batch_ = 2;
        ++prof_.verified;
        prof_.d_verify += now_seconds() - t0;
        return true;
    }
    if (!verify8_ || !dec8_) {
        err = "verifying needs generating on the GPU (and its batch programs)";
        return false;
    }
    if (n < 2 || n > spec_t_) {
        err = "a verified batch of " + std::to_string(n) + " tokens (2 .. " + std::to_string(spec_t_) + ")";
        return false;
    }
    const int H = c.hidden;
    prof_.tokens += static_cast<uint64_t>(n);
    pin_current_thread(big_);
    if (!reserve_context(pos_ + n, err)) return false;
    const double t0 = now_seconds();
    for (int t = 0; t < n; ++t) store_.next_token();
    if (!gpu8_->set_tokens(n)) {
        err = "GPU: no program for " + std::to_string(n) + " tokens";
        return false;
    }
    struct One {   // one token again, however this ends
        Gpu8 *g;
        ~One() { g->set_tokens(1); }
    } one{gpu8_.get()};
    for (int t = 0; t < n; ++t)
        if (!pack_.embedding(toks[t], gpu8_->dec_x() + static_cast<size_t>(t) * H)) {
            err = "cannot read the embedding of token " + std::to_string(toks[t]);
            return false;
        }
    std::vector<int> held;
    auto fail = [&](const std::string &e) {
        for (int s : held) store_.release(s);
        err = "GPU verifying: " + e;
        gpu_status_ = "failed: " + e;
        dec8_ = verify8_ = false;   // the CPU from the next token (these are lost)
        return false;
    };
    for (int l = 0; l < c.n_layers; ++l) {
        const Gpu8::DecodeLayer *next = l + 1 < c.n_layers ? &dec_layers_[static_cast<size_t>(l + 1)] : nullptr;
        const Gpu8::DecodeLayer *next2 = dec_guesses2_ && l + 2 < c.n_layers ? &dec_layers_[static_cast<size_t>(l + 2)] : nullptr;
        std::string e;
        const double tf = now_seconds();
        if (!gpu8_->dec_front(dec_layers_[static_cast<size_t>(l)], next, pos_, kv_ptrs_, e, next2)) return fail(e);
        prof_.d_front += now_seconds() - tf;
        if (!back_gpu8(l, layers_[static_cast<size_t>(l)], dec_layers_[static_cast<size_t>(l)], next != nullptr, n, held, e,
                       next2 != nullptr))
            return fail(e);
    }
    std::string e;
    const double th = now_seconds();
    if (!gpu8_->dec_head(dec_final_norm_, lm_head_, logits, e)) return fail(e);
    prof_.d_head += now_seconds() - th;
    for (int s : held) store_.release(s);
    if (mtp_) {
        if (hid_.size() < static_cast<size_t>(n) * H) hid_.resize(static_cast<size_t>(n) * H);
        for (int t = 0; t < n; ++t)
            rmsnorm(gpu8_->dec_x() + static_cast<size_t>(t) * H, final_norm_, hid_.data() + static_cast<size_t>(t) * H, H,
                    c.eps, true);
        last_hid_.assign(hid_.begin() + static_cast<ptrdiff_t>(n - 1) * H, hid_.begin() + static_cast<ptrdiff_t>(n) * H);
    }
    pos_ += n;
    last_batch_ = n;
    ++prof_.verified;
    prof_.gpu8 += now_seconds() - t0;
    return true;
}

bool Model::verify_keep(int kept, std::string &err) {
    if (kept < 1 || kept >= last_batch_) {
        err = "keeping " + std::to_string(kept) + " of a batch of " + std::to_string(last_batch_);
        return false;
    }
    if ((npu_spec_ && npu_.dec) || (cpu_spec_ && !dec8_)) {
        // the states as after the first token are the backups: swapped in (the second token's KV row is written over)
        std::swap(conv_st_, conv_bk_);
        std::swap(gdn_st_, gdn_bk_);
        pos_ -= 1;
        last_batch_ = 0;
        ++prof_.rejected;
        if (mtp_) last_hid_.assign(hidden(0), hidden(0) + cfg_.hidden);
        return true;
    }
    // the tokens after `kept` taken back: the GDN layers' states as after token kept - 1 (the batch left them in the
    // backups, one a token); waited for, so the CPU (a next prompt) reads them right
    std::string e;
    const size_t b = static_cast<size_t>(kept - 1);
    if (!gpu8_->dec_copy(conv_st_, conv_bk_ + b * conv_n_, conv_n_ * sizeof(float), e) ||
        !gpu8_->dec_copy(gdn_st_, gdn_bk_ + b * gdn_n_, gdn_n_ * sizeof(float), e) || !gpu8_->dec_wait(e)) {
        err = e;
        dec8_ = verify8_ = false;
        return false;
    }
    pos_ -= last_batch_ - kept;
    last_batch_ = 0;
    ++prof_.rejected;
    if (mtp_)
        last_hid_.assign(hid_.begin() + static_cast<ptrdiff_t>(b) * cfg_.hidden,
                         hid_.begin() + static_cast<ptrdiff_t>(b + 1) * cfg_.hidden);
    return true;
}

uint64_t Model::memory_bytes() const {
    const auto vec = [](const std::vector<float> &v) { return static_cast<uint64_t>(v.size()) * sizeof(float); };
    return pack_.dense_bytes() + store_.bytes() + static_cast<uint64_t>(conv_n_ + gdn_n_) * sizeof(float) + spec_bytes() +
           static_cast<uint64_t>(kv_ptrs_.size()) * kv_chunk_bytes_ + vec(eout_) + vec(proj_) +
           vec(qg_) + vec(router_);
}

bool Model::reserve_context(int positions, std::string &err) {
    if (positions > ctx_) {
        err = "the context is full (" + std::to_string(ctx_) + " tokens)";
        return false;
    }
    while (context_allocated() < positions) {
        // the expert cache gives the memory first, so the total stays within the budget
        if (slots_per_chunk_) store_.retire_bytes(static_cast<uint64_t>(slots_per_chunk_) * slot_bytes_);
        if (!new_kv_chunk(err)) return false;
    }
    return true;
}

size_t Model::batch_bytes(int tokens) const {
    // the float rows, the router's extra row, and each token's quantized activation (int8 codes + scales)
    const size_t B = static_cast<size_t>(tokens);
    return (B * token_floats_ + static_cast<size_t>(cfg_.n_experts)) * sizeof(float) +
           2 * B * (3 * max_cols_ + max_cols_ / kBlock * sizeof(float));
}

namespace {
// scratch: its contents do not matter, its memory does (a smaller vector must give the rest back)
template <typename T>
void fit(std::vector<T> &v, size_t n) {
    if (v.size() != n) std::vector<T>(n).swap(v);
}
}  // namespace

uint64_t Model::batch_loan(int tokens) const {
    if (!budgeted_ || tokens <= kSmallBatch) return 0;
    const uint64_t more = batch_bytes(tokens) - batch_bytes(kSmallBatch);
    return more > spare_bytes_ ? (more - spare_bytes_ + slot_bytes_ - 1) / slot_bytes_ * slot_bytes_ : 0;
}

bool Model::size_batch(int tokens, std::string &err) {
    if (tokens == batch_cap_) return true;
    // the expert cache lends what the budget has not spare (its coldest slots), so the total stays within it
    const uint64_t loan = batch_loan(tokens);
    if (loan > batch_retired_) batch_retired_ += store_.retire_bytes(loan - batch_retired_);
    const auto &c = cfg_;
    const size_t B = static_cast<size_t>(tokens), H = static_cast<size_t>(c.hidden);
    const int kd = c.gdn_k_heads * c.gdn_k_dim, vd = c.gdn_v_heads * c.gdn_v_dim;
    const size_t cw = static_cast<size_t>(std::max(vd, c.n_heads * c.head_dim));
    const size_t kv_row = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    try {
        fit(x_, B * H);
        fit(h_, B * H);
        fit(mix_, B * H);
        fit(moe_, B * H);
        fit(xin_, B * H);
        fit(hid_, B * H);
        fit(proj_, B * static_cast<size_t>(2 * kd + vd));
        fit(z_, B * vd);
        fit(ab_, B * 2 * c.gdn_v_heads);
        fit(core_, B * cw);
        fit(qg_, B * c.n_heads * c.head_dim * 2);
        fit(kbuf_, B * kv_row);
        fit(vbuf_, B * kv_row);
        fit(router_, (B + 1) * c.n_experts);   // + a row for the next layer's guess
        fit(sh_act_, B * 2 * c.shared_ff);
        fit(sh_, B * H);
        fit(eact_, B * 2 * c.moe_ff);
        fit(eout_, B * c.top_k * H);           // one output per (token, routed expert)
        fit(top_ids_, B * c.top_k);
        fit(top_w_, B * c.top_k);
        fit(qacts_, B);
        fit(qh_, B);
        fit(qh_ptr_, B);
    } catch (const std::bad_alloc &) {
        err = "not enough memory for a batch of " + std::to_string(tokens) + " tokens";
        return false;
    }
    batch_cap_ = tokens;
    return true;
}

int Model::prompt_batch() const {
    if (!budgeted_) return kMaxBatch;
    // the largest batch whose scratch the expert cache can lend now: it keeps what a layer and the next one's
    // read-ahead pin
    const size_t keep = static_cast<size_t>(2 * cfg_.n_experts + 2 * cfg_.top_k + 2);
    for (int b = kMaxBatch; b > kSmallBatch; b /= 2) {
        const uint64_t loan = batch_loan(b);
        if (loan <= batch_retired_ || store_.bytes() >= loan - batch_retired_ + static_cast<uint64_t>(keep) * slot_bytes_) return b;
    }
    return kSmallBatch;
}

void Model::trim_batch() {
    long_x_ = {};
    hid_long_ = {};
    if (batch_cap_ <= kSmallBatch) return;
    std::string e;
    size_batch(kSmallBatch, e);   // shrinking does not fail
    store_.unretire_bytes(batch_retired_);
    batch_retired_ = 0;
}

void Model::set_rope_delta(int d) {
    rope_delta_ = d;
    if (gpu8_) gpu8_->set_rope_shift(d);
}

void Model::rope_pos(int idx, int p[3]) const {
    int delta = 0;
    for (const Image &im : images_) {   // in KV order
        if (im.kv0 < 0 || idx < im.kv0) break;
        if (idx < im.kv0 + im.n()) {
            const int j = idx - im.kv0;
            p[0] = im.s;
            p[1] = im.s + j / im.gw;
            p[2] = im.s + j % im.gw;
            return;
        }
        delta = im.s + std::max(im.gh, im.gw) - (im.kv0 + im.n());
    }
    p[0] = p[1] = p[2] = idx + delta;
}

const float *Model::image_row(int idx) const {
    for (const Image &im : images_)
        // (an image of a loaded conversation has no features: its tokens are all in the KV cache already)
        if (im.kv0 >= 0 && idx >= im.kv0 && idx < im.kv0 + im.n() && !im.feats.empty())
            return im.feats.data() + static_cast<size_t>(idx - im.kv0) * cfg_.hidden;
    return nullptr;
}

bool Model::add_image(const uint8_t *rgb, int w, int h, const std::string &cache, int &tokens, std::string &err) {
    if (!vision_.available()) {
        err = "the model has no vision encoder (" + vision_status_ + ")";
        return false;
    }
    Image im;
    if (!vision_.load_features(cache, w, h, im.feats)) {
        pin_current_thread(big_);
        pool_->set_width(pool_->size());
        pool_->set_little(0);
        if (!vision_.encode(rgb, w, h, *pool_, im.feats, err, gpu8_.get())) return false;
        vision_.save_features(cache, w, h, im.feats);   // a cache: failing to write it costs the next encode only
    }
    const int f = vision_.shape().patch * vision_.shape().merge;
    im.gh = h / f;
    im.gw = w / f;
    tokens = im.n();
    images_.push_back(std::move(im));
    return true;
}

bool Model::cache_image(const uint8_t *rgb, int w, int h, const std::string &cache, std::string &err) {
    if (!vision_.available()) {
        err = "the model has no vision encoder (" + vision_status_ + ")";
        return false;
    }
    if (vision_.has_features(cache, w, h)) return true;
    pin_current_thread(big_);
    pool_->set_width(pool_->size());
    pool_->set_little(0);
    std::vector<float> feats;
    if (!vision_.encode(rgb, w, h, *pool_, feats, err, gpu8_.get())) return false;
    if (!vision_.save_features(cache, w, h, feats)) {
        err = "cannot write " + cache;
        return false;
    }
    return true;
}

int Model::pending_image_tokens() const {
    int n = 0;
    for (size_t i = image_next_; i < images_.size(); ++i) n += images_[i].n();
    return n - image_fed_;
}

void Model::drop_pending_images() {
    if (image_fed_ == 0) images_.resize(image_next_);
}

void Model::reset() {
    images_.clear();
    image_next_ = 0;
    image_fed_ = 0;
    set_rope_delta(0);
    std::fill(conv_st_, conv_st_ + conv_n_, 0.0f);
    std::fill(gdn_st_, gdn_st_ + gdn_n_, 0.0f);
    trim_batch();
    // back to one chunk of context; the expert cache gets its slots back (shared-memory chunks kept for reuse)
    if (kv_chunks_.empty())
        while (kv_ptrs_.size() > 1) {
            kv_spare_.push_back(kv_ptrs_.back());
            kv_ptrs_.pop_back();
        }
    kv_chunks_.resize(std::min<size_t>(kv_chunks_.size(), 1));
    kv_ptrs_.resize(1);
    store_.restore();
    pos_ = 0;   // the KV cache is overwritten from position 0
    new_conversation();
}

void Model::apply_experts() {
    tail_skip_ = tail_cfg_ >= 0 ? tail_cfg_ : 0;   // off unless asked: every expert the router picks
    if (!gpu8_) return;
    // the GPU's route, experts and combine take k (their buffers hold the model's top_k); its guesses follow
    if (dec8_ok_ && !guesses_env_) dec_guesses_ = k_;
    gpu8_->set_experts(k_, dec8_ok_ ? dec_guesses_ : 0);
}

bool Model::set_experts(int k, std::string &err) {
    if (k < 1 || k > cfg_.top_k) {
        err = "experts per token: 1 .. " + std::to_string(cfg_.top_k) + ", not " + std::to_string(k);
        return false;
    }
    // the guesses made with the old k let go (they would only be read ahead for nothing)
    for (auto &pins : prefetch_) {
        for (int s : pins) store_.release(s);
        pins.clear();
    }
    k_ = k;
    apply_experts();
    return true;
}

uint64_t Model::reserved_bytes() const {
    uint64_t b = 0;
    for (int s : reserved_) b += store_.slot_size(s);
    for (const TailPiece &t : tail_)
        if (!t.mem) b += t.bytes;
    return b;
}

uint64_t Model::reserve_memory(uint64_t bytes, std::string &err) {
    // the guesses' pins go first (a pinned slot is not taken; a guess is only a read ahead)
    for (auto &pins : prefetch_) {
        for (int s : pins) store_.release(s);
        pins.clear();
    }
    if (bytes == 0) {
        // the reserve is over: its slots back into the cache, empty (read again on demand); the GPU's pieces
        // allocated again first (they may not be: the slots stay out, and the next call tries again)
        for (TailPiece &t : tail_) {
            if (t.mem) continue;
            std::string e;
            uint8_t *m = gpu8_->alloc(t.bytes, e);
            if (!m) {
                err = "the reserve's memory could not be taken back: " + e;
                continue;
            }
            t.mem = m;
            std::vector<uint8_t *> at(t.slots.size());
            for (size_t i = 0; i < at.size(); ++i) at[i] = m + i * store_.slot_bytes();
            store_.give_back(t.slots, at);
        }
        store_.give_back(reserved_);
        reserved_.clear();
        return reserved_bytes();
    }
    if (gpu8_experts_) {
        // whole pieces of the tail, freed (their pages go back with them)
        for (TailPiece &t : tail_) {
            if (reserved_bytes() >= bytes) break;
            if (!t.mem) continue;
            const std::vector<int> got = store_.take(t.slots, 0, 0);
            if (got.size() != t.slots.size()) {   // a slot still pinned: not this piece
                store_.give_back(got);
                continue;
            }
            gpu8_->free(t.mem);
            t.mem = nullptr;
        }
    } else {
        // the coldest slots, their pages released (the cache keeps what a layer and its guesses pin)
        const uint64_t have = reserved_bytes();
        if (have < bytes) {
            const std::vector<int> got = store_.take_bytes(bytes - have, min_slots_ + static_cast<size_t>(cfg_.top_k));
            reserved_.insert(reserved_.end(), got.begin(), got.end());
        }
    }
    if (reserved_bytes() < bytes)
        err = "the expert cache gave " + std::to_string(reserved_bytes() >> 20) + " MB of the " +
              std::to_string(bytes >> 20) + " MB asked";
    return reserved_bytes();
}

std::vector<float *> Model::rows(std::vector<float> &buf, size_t width, int n, size_t offset) {
    std::vector<float *> r(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) r[static_cast<size_t>(t)] = buf.data() + static_cast<size_t>(t) * width + offset;
    return r;
}

const QAct *const *Model::quantize_h(const std::vector<float *> &hs, int64_t cols) {
    if (exact_) return nullptr;
    const int n = static_cast<int>(hs.size());
    par(n, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            quantize_activation(hs[static_cast<size_t>(t)], cols, qh_[static_cast<size_t>(t)]);
            qh_ptr_[static_cast<size_t>(t)] = &qh_[static_cast<size_t>(t)];
        }
    }, 8);
    return qh_ptr_.data();
}

void Model::setup_gpu(const ModelOptions &opt) {
    std::string e, e8;
    if (setup_gpu8(e8)) return;   // the Adreno 7xx+ int8 path; else the 618's hand-written one
    gpu_ = std::make_unique<Gpu>();
    if (!gpu_->open(e)) {
        gpu_status_ = e8.rfind("no Adreno 7xx", 0) == 0 ? e : e8;   // why the newer path failed, where it applies
        gpu_.reset();
        return;
    }
    gpu_share_ = opt.gpu_share > 0.0f ? std::min(opt.gpu_share, 1.0f) : kGpuShare;
    // generating on it too (model.hpp, gpu_decode_): measuring switches
    if (const char *e = std::getenv("GEODE_GPU_DECODE")) gpu_decode_ = std::clamp(std::atoi(e), 0, 2);
    gpu_dec_share_ = gpu_share_;
    if (const char *e = std::getenv("GEODE_GPU_DECODE_SHARE")) gpu_dec_share_ = std::clamp(static_cast<float>(std::atof(e)), 0.0f, 1.0f);
    gpu_->set_parallel([this](int64_t n, const std::function<void(int64_t, int64_t)> &fn) { pool_->parallel_for(n, fn, 4); });
    // each matrix moves into the GPU's memory, and its pages of dense.bin go back to the system: no byte twice
    int moved = 0;
    std::vector<Layer *> ls;
    for (Layer &L : layers_) ls.push_back(&L);
    if (mtp_) ls.push_back(&mtp_layer_);   // run over every prompt token too (its KV)
    for (Layer *lp : ls)
        for (QView *m : {&lp->qkv, &lp->z, &lp->q, &lp->out, &lp->o}) {
            if (!Gpu::fits(*m) || m->rows < kGpuMinRows) continue;
            uint8_t *old = const_cast<uint8_t *>(m->data);
            const size_t bytes = static_cast<size_t>(m->rows) * m->row_bytes;
            if (!gpu_->adopt(*m, e)) {
                gpu_status_ = e;
                gpu_failed_ = true;   // the matrices moved so far stay in the GPU layout: the CPU reads them
                return;
            }
            release_pages(old, bytes);
            ++moved;
        }
    if (!moved) {
        gpu_status_ = "off (no q2 / q4 dense matrix with 2048 columns)";
        gpu_.reset();
        return;
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "on: %d matrices, %.0f MB in GPU memory, %.0f%% of their rows", moved,
                  static_cast<double>(gpu_->adopted_bytes()) / 1048576.0, gpu_share_ * 100.0);
    gpu_status_ = buf;
}

bool Model::setup_gpu8(std::string &why) {
    auto g = std::make_unique<Gpu8>();
    g->set_cache_dir(pack_.dir());   // the compiled programs kept beside the pack (like cache_profile.bin)
    if (!g->open(why)) return false;
    open_clock.mark("gpu: OpenCL + programs");
    const auto &c = cfg_;
    // each dense q4 matrix into shared memory in tile8, its pages of dense.bin back to the system
    std::vector<QView *> views;
    auto add = [&](Layer &L) {
        for (QView *m : {&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.sh_gate, &L.sh_up, &L.sh_down})
            views.push_back(m);
    };
    for (Layer &L : layers_) add(L);
    if (mtp_) {
        add(mtp_layer_);
        views.push_back(&mtp_fc_);
    }
    int moved = 0;
    int64_t max_cols = c.hidden, max_rows = c.hidden;
    // all of them at once on the big cores (one after another: 1.07 s on the 8 Elite)
    const int adopt_threads = std::max(1, static_cast<int>(big_.size()));
    std::vector<std::pair<const uint8_t *, size_t>> olds;   // the pack's bytes of each that fits, given back after
    for (QView *m : views)
        if (Gpu8::fits(*m)) olds.emplace_back(m->data, static_cast<size_t>(m->rows) * m->row_bytes);
    const bool all_moved = g->adopt_all(views, adopt_threads, why);
    std::vector<const uint8_t *> released;
    for (const auto &[old, bytes] : olds) {
        const bool moved_out = std::none_of(views.begin(), views.end(), [&](const QView *m) { return m->data == old; });
        if (moved_out && std::find(released.begin(), released.end(), old) == released.end()) {
            release_pages(const_cast<uint8_t *>(old), bytes);
            released.push_back(old);
        }
    }
    if (!all_moved) {
        // what moved stays in tile8, which the CPU reads; the GPU takes nothing
        gpu_status_ = "off (" + why + ")";
        return true;
    }
    for (QView *m : views)
        if (m->data && m->tile8 && g->holds(*m)) {
            max_cols = std::max(max_cols, m->cols);
            max_rows = std::max(max_rows, m->rows);
        }
    moved = static_cast<int>(olds.size());
    // the draft head (model.hpp, draft_head_): lm_head's rows of draft_vocab.bin's tokens, in the pack's layout,
    // then into shared memory like the others
    if (moved && mtp_) {
        std::vector<int32_t> ids;
        if (std::FILE *f = std::fopen((pack_.dir() + "/draft_vocab.bin").c_str(), "rb")) {
            int32_t id;
            while (std::fread(&id, 4, 1, f) == 1)
                if (id >= 0 && id < c.vocab) ids.push_back(id);
            std::fclose(f);
        }
        if (!ids.empty()) {
            while (ids.size() % 32) ids.push_back(ids[0]);   // whole tiles (the copies never win: the first equal row does)
            const size_t rb = lm_head_.row_bytes;
            AlignedBuffer tmp = aligned_bytes(ids.size() * rb);
            if (tmp) {
                for (size_t r = 0; r < ids.size(); ++r) {
                    if (lm_head_.quad) quad_row(lm_head_, ids[r], tmp.get() + r * rb);
                    else std::memcpy(tmp.get() + r * rb, lm_head_.data + static_cast<size_t>(ids[r]) * rb, rb);
                }
                QView dv = lm_head_;
                dv.rows = static_cast<int64_t>(ids.size());
                dv.data = tmp.get();
                dv.quad = dv.tile8 = false;
                std::string e;
                if (Gpu8::fits(dv) && g->adopt_all({&dv}, adopt_threads, e)) {
                    draft_head_ = dv;
                    draft_ids_ = std::move(ids);
                }
            }
        }
    }
    open_clock.mark("gpu: dense into tile8 + draft head");
    // lm_head too (the GPU generates through it; a batch's logits stay on the CPU: they would need a vocab-wide stage)
    if (moved && Gpu8::fits(lm_head_)) {
        const uint8_t *old = lm_head_.data;
        if (g->adopt_all({&lm_head_}, adopt_threads, why))
            release_pages(const_cast<uint8_t *>(old), static_cast<size_t>(lm_head_.rows) * lm_head_.row_bytes);
    }
    open_clock.mark("gpu: lm_head");
    const bool reserved = moved && g->reserve(kMaxBatch, max_cols, max_rows, c.hidden, c.moe_ff, c.top_k, c.n_experts, why);
    open_clock.mark("gpu: reserve");
    if (!reserved) {
        gpu_status_ = "off (" + (moved ? why : std::string("no q4 dense matrix")) + ")";
        return true;
    }
    gpu8_ = std::move(g);
    char buf[200];
    std::snprintf(buf, sizeof buf, "on: %s, int8, %d matrices, %.0f MB of shared memory", gpu8_->device().c_str(), moved,
                  static_cast<double>(gpu8_->allocated()) / 1048576.0);
    gpu_status_ = buf;
    return true;
}

QView Model::expert_view(DType type, int64_t rows, int64_t cols, const uint8_t *data, float global) const {
    QView v{type, rows, cols, row_bytes(type, cols), data, global};
    v.quad = v.tile8 = type == DType::Q4 && pack_.experts_tile8();
    return v;
}

void Model::setup_npu() {
#if defined(__ANDROID__)
    const char *path = std::getenv("GEODE_NPU_LIB") ? std::getenv("GEODE_NPU_LIB") : "libgeode_npu.so";
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        npu_.status = std::string("off (") + (dlerror() ? dlerror() : "no libgeode_npu.so") + ")";
        return;
    }
    using Open = void *(*)(const char *, char *, int);
    using Adopt = int (*)(void *, const uint8_t *, int64_t, int64_t, size_t, float);
    const auto open = reinterpret_cast<Open>(dlsym(lib, "gnpu_open"));
    const auto adopt = reinterpret_cast<Adopt>(dlsym(lib, "gnpu_adopt_q4"));
    npu_.gemm = reinterpret_cast<decltype(npu_.gemm)>(dlsym(lib, "gnpu_gemm"));
    npu_.close = reinterpret_cast<decltype(npu_.close)>(dlsym(lib, "gnpu_close"));
    npu_.bytes = reinterpret_cast<decltype(npu_.bytes)>(dlsym(lib, "gnpu_bytes"));
    npu_.dsp = reinterpret_cast<decltype(npu_.dsp)>(dlsym(lib, "gnpu_dsp_seconds"));
    npu_.gemm_multi = reinterpret_cast<decltype(npu_.gemm_multi)>(dlsym(lib, "gnpu_gemm_multi"));
    npu_.conv = reinterpret_cast<decltype(npu_.conv)>(dlsym(lib, "gnpu_convert_seconds"));
    using SetPar = void (*)(void *, void (*)(void *, int, void (*)(void *, int, int), void *), void *);
    const auto set_par = reinterpret_cast<SetPar>(dlsym(lib, "gnpu_set_par"));
    if (!open || !adopt || !npu_.gemm || !npu_.close || !npu_.bytes || !npu_.dsp || !npu_.gemm_multi || !npu_.conv ||
        !set_par) {
        npu_.status = "off (libgeode_npu.so lacks its interface)";
        dlclose(lib);
        return;
    }
    npu_.lib = lib;
    // the skel beside the library (the app's native library folder, or the adb tools' folder)
    std::string dir = ".";
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(open), &info) && info.dli_fname) {
        dir = info.dli_fname;
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    char err[256] = {0};
    npu_.g = open(dir.c_str(), err, sizeof err);
    if (!npu_.g) {
        npu_.status = std::string("off (") + err + ")";
        return;
    }
    // the conversions (activations into fp16 tiles, results back) on the engine's threads
    set_par(npu_.g, [](void *ctx, int items, void (*fn)(void *, int, int), void *arg) {
        static_cast<Model *>(ctx)->par(items, [&](int64_t i0, int64_t i1) {
            fn(arg, static_cast<int>(i0), static_cast<int>(i1));
        });
    }, this);
    // the dense q4 matrices a prompt multiplies (each tensor once, whichever views share it)
    std::vector<const QView *> views;
    auto add = [&](const Layer &L) {
        for (const QView *m : {&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.sh_gate, &L.sh_up, &L.sh_down})
            views.push_back(m);
    };
    for (const Layer &L : layers_) add(L);
    if (mtp_) {
        add(mtp_layer_);
        views.push_back(&mtp_fc_);
    }
    int n = 0;
    for (const QView *m : views) {
        if (!m->data || m->type != DType::Q4 || m->quad || m->rows % 32 || m->cols % 32 || npu_.ids.count(m->data))
            continue;
        const int id = adopt(npu_.g, m->data, m->rows, m->cols, m->row_bytes, m->global);
        if (id < 0) break;
        npu_.ids[m->data] = id;
        npu_.adopted.emplace_back(m, id);
        ++n;
    }
    npu_.status = "on: " + std::to_string(n) + " matrices, " + std::to_string(npu_.bytes(npu_.g) >> 20) + " MB";
    // the decoder: a dense model's whole decode step and its prompts on the DSP, on v79 and later (GEODE_NPU_DECODE=0:
    // the CPU's, the NPU only for the prompt's products)
    npu_.shared = reinterpret_cast<decltype(npu_.shared)>(dlsym(lib, "gnpu_shared_alloc"));
    npu_.dec_size = reinterpret_cast<decltype(npu_.dec_size)>(dlsym(lib, "gnpu_dec_size"));
    npu_.dec_open = reinterpret_cast<decltype(npu_.dec_open)>(dlsym(lib, "gnpu_dec_open"));
    npu_.dec_kv = reinterpret_cast<decltype(npu_.dec_kv)>(dlsym(lib, "gnpu_dec_kv"));
    npu_.dec_step = reinterpret_cast<decltype(npu_.dec_step)>(dlsym(lib, "gnpu_dec_step"));
    npu_.dec_draft = reinterpret_cast<decltype(npu_.dec_draft)>(dlsym(lib, "gnpu_dec_draft"));
    npu_.dec_prefill = reinterpret_cast<decltype(npu_.dec_prefill)>(dlsym(lib, "gnpu_dec_prefill"));
    npu_.mat_hmx = reinterpret_cast<decltype(npu_.mat_hmx)>(dlsym(lib, "gnpu_mat_hmx"));
    npu_.dec_mtp_kv = reinterpret_cast<decltype(npu_.dec_mtp_kv)>(dlsym(lib, "gnpu_dec_mtp_kv"));
    npu_.moe = reinterpret_cast<decltype(npu_.moe)>(dlsym(lib, "gnpu_moe"));
    const auto arch = reinterpret_cast<int (*)()>(dlsym(lib, "gnpu_arch"));
    const auto skel_arch = reinterpret_cast<int (*)(int)>(dlsym(lib, "gnpu_skel_arch"));
    npu_.arch = arch && skel_arch ? skel_arch(arch()) : 0;
    const bool want = cfg_.dense && !(std::getenv("GEODE_NPU_DECODE") && std::atoi(std::getenv("GEODE_NPU_DECODE")) == 0);
    if (!want) {
        npu_.dec_status = cfg_.dense ? "off (GEODE_NPU_DECODE)" : "off (a dense model only)";
    } else if (!npu_.shared || !npu_.dec_size || !npu_.dec_open || !npu_.dec_kv || !npu_.dec_step || !npu_.dec_draft) {
        npu_.dec_status = "off (libgeode_npu.so has no decoder)";
    } else if (npu_.arch < 79) {
        npu_.dec_status = "off (Hexagon v" + std::to_string(npu_.arch) + ": the decoder needs v79)";
    } else {
        std::string why;
        if (!build_npu_decode(why)) npu_.dec_status = "off (" + why + ")";
    }
#else
    npu_.status = "off (no NPU here)";
#endif
}


bool Model::build_npu_decode(std::string &why) {
#if defined(__ANDROID__)
    const auto &c = cfg_;
    const int H = c.hidden, nv = c.gdn_v_heads, kd = c.gdn_k_heads * c.gdn_k_dim, vd = nv * c.gdn_v_dim;
    const int conv_dim = 2 * kd + vd, K = c.conv_kernel;
    if (c.n_layers > DEC_MAX_LAYERS || H % 64 || c.shared_ff % 64 || c.gdn_k_dim > 256 || c.gdn_v_dim % 16 ||
        c.head_dim % 64) {
        why = "a shape the decoder does not take";
        return false;
    }
    const bool head_q8 = lm_head_.type == DType::Q8;
    if (!lm_head_.data || lm_head_.quad || lm_head_.cols != H || lm_head_.rows % 32 ||
        !(head_q8 ? H <= 3072 : lm_head_.type == DType::Q4 && tile8_capable(DType::Q4, lm_head_.rows, lm_head_.cols))) {
        why = std::string("the head (") + dtype_name(lm_head_.type) + ") in no layout the decoder takes";
        return false;
    }
    npu_.cfg.assign(sizeof(dec_cfg_t), 0);
    auto &cf = *reinterpret_cast<dec_cfg_t *>(npu_.cfg.data());
    cf.version = DEC_VERSION;
    cf.hidden = H, cf.vocab = c.vocab, cf.n_layers = c.n_layers, cf.sff = c.shared_ff;
    cf.nh = c.n_heads, cf.nkv = c.n_kv_heads, cf.hd = c.head_dim, cf.half = c.rope_dim / 2;
    cf.nk = c.gdn_k_heads, cf.nv = nv, cf.dk = c.gdn_k_dim, cf.dv = c.gdn_v_dim, cf.conv_k = K;
    cf.eps = c.eps;
    // W: each q4 matrix in q4b (dec_cfg.h; GEODE_NPU_TILE8=1: tile8), a q8 one in q8 tiles, 128-byte aligned
    std::vector<std::pair<const QView *, dec_mat_t *>> mats;
    uint64_t at = 0;
    static const bool tile8 = std::getenv("GEODE_NPU_TILE8") && std::atoi(std::getenv("GEODE_NPU_TILE8")) != 0;
    const auto place = [&](const QView &v, dec_mat_t &d, bool q8) {
        d.w = at, d.k = static_cast<uint32_t>(v.cols), d.n = static_cast<uint32_t>(v.rows), d.global = v.global;
        d.fmt = q8 ? DEC_FMT_Q8 : tile8 ? DEC_FMT_Q4T8 : DEC_FMT_Q4B;
        at += (static_cast<uint64_t>(v.rows) / 32 * v.cols * (q8 ? 34u : tile8 ? 18u : 20u) + 127) / 128 * 128;
        mats.emplace_back(&v, &d);
    };
    for (int l = 0; l < c.n_layers; ++l) {
        const Layer &L = layers_[static_cast<size_t>(l)];
        dec_layer_t &D = cf.layers[l];
        D.attn = L.attn ? 1 : 0;
        D.index = L.index;
        std::vector<std::pair<const QView *, dec_mat_t *>> mine;
        if (L.attn) mine = {{&L.q, &D.q}, {&L.k, &D.k}, {&L.v, &D.v}, {&L.o, &D.o}};
        else mine = {{&L.qkv, &D.qkv}, {&L.z, &D.z}, {&L.out, &D.out}};
        mine.insert(mine.end(), {{&L.sh_gate, &D.gate}, {&L.sh_up, &D.up}, {&L.sh_down, &D.down}});
        for (auto &[v, d] : mine) {
            if (!v->data || v->type != DType::Q4 || v->quad || !tile8_capable(DType::Q4, v->rows, v->cols) || v->cols > 6144) {
                why = "a matrix not in q4 (layer " + std::to_string(l) + ")";
                return false;
            }
            place(*v, *d, false);
        }
    }
    place(lm_head_, cf.head, head_q8);
    // the MTP layer: fc, its layer's matrices, the draft head (lm_head's rows of draft_vocab.bin's tokens, padded to
    // a multiple of 32 with the last one: a repeat changes neither the argmax's token nor, much, its probability)
    std::vector<int32_t> draft_ids;
    QView draft_view = lm_head_;
    std::vector<uint8_t> draft_rows;   // the head's rows in the pack's layout (tiled with the rest below)
    const auto mtp_ok = [&](const QView &v) {
        return v.data && !v.quad && v.type == DType::Q4 && tile8_capable(DType::Q4, v.rows, v.cols) && v.cols <= 6144;
    };
    if (mtp_ && mtp_ok(mtp_fc_) && mtp_ok(mtp_layer_.q) && mtp_ok(mtp_layer_.k) && mtp_ok(mtp_layer_.v) &&
        mtp_ok(mtp_layer_.o) && mtp_ok(mtp_layer_.sh_gate) && mtp_ok(mtp_layer_.sh_up) && mtp_ok(mtp_layer_.sh_down)) {
        cf.mtp = 1;
        place(mtp_fc_, cf.mtp_fc, false);
        dec_layer_t &D = cf.mtp_layer;
        D.attn = 1;
        D.index = mtp_layer_.index;
        place(mtp_layer_.q, D.q, false);
        place(mtp_layer_.k, D.k, false);
        place(mtp_layer_.v, D.v, false);
        place(mtp_layer_.o, D.o, false);
        place(mtp_layer_.sh_gate, D.gate, false);
        place(mtp_layer_.sh_up, D.up, false);
        place(mtp_layer_.sh_down, D.down, false);
        if (std::FILE *f = std::fopen((pack_.dir() + "/draft_vocab.bin").c_str(), "rb")) {
            int32_t id;
            while (std::fread(&id, 4, 1, f) == 1)
                if (id >= 0 && id < c.vocab) draft_ids.push_back(id);
            std::fclose(f);
        }
        if (!draft_ids.empty()) {
            while (draft_ids.size() % 32) draft_ids.push_back(draft_ids.back());
            draft_rows.resize(draft_ids.size() * lm_head_.row_bytes);
            for (size_t r = 0; r < draft_ids.size(); ++r)
                std::memcpy(draft_rows.data() + r * lm_head_.row_bytes,
                            lm_head_.data + static_cast<size_t>(draft_ids[r]) * lm_head_.row_bytes, lm_head_.row_bytes);
            draft_view.rows = static_cast<int64_t>(draft_ids.size());
            draft_view.data = draft_rows.data();
            cf.n_draft = static_cast<int32_t>(draft_ids.size());
        }
        place(draft_view, cf.draft_head, head_q8);
    }
    // the hmx4 copies the prompt's HMX products read (setup_npu adopted them): a prompt on the DSP when every layer's
    // matrices have one (GEODE_NPU_PREFILL=0: the CPU's prompt)
    {
        bool all = npu_.mat_hmx && npu_.dec_prefill &&
                   !(std::getenv("GEODE_NPU_PREFILL") && std::atoi(std::getenv("GEODE_NPU_PREFILL")) == 0);
        for (auto &[v, d] : mats) {
            const auto it = npu_.ids.find(v->data);
            int hfd = 0, shift = 0;
            long long kp = 0;
            if (it != npu_.ids.end() && npu_.mat_hmx && npu_.mat_hmx(npu_.g, it->second, &hfd, &shift, &kp) == 0 &&
                kp == v->cols) {
                d->hmx_fd = hfd;
                d->hmx_shift = shift;
            }
        }
        for (int l = 0; l < c.n_layers; ++l) {
            const dec_layer_t &D = cf.layers[l];
            for (const dec_mat_t *x : {&D.qkv, &D.z, &D.out, &D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down})
                if (x->n && !x->hmx_fd) all = false;
        }
        npu_.prefill = all;
        // (GEODE_NPU_MTP_KV=0: the CPU's MTP pass over a prompt)
        npu_.mtp_kv = all && cf.mtp && npu_.dec_mtp_kv && cf.mtp_fc.hmx_fd && cf.mtp_layer.k.hmx_fd &&
                      cf.mtp_layer.v.hmx_fd &&
                      !(std::getenv("GEODE_NPU_MTP_KV") && std::atoi(std::getenv("GEODE_NPU_MTP_KV")) == 0);
    }
    int fd = -1;
    npu_.w = static_cast<uint8_t *>(npu_.shared(npu_.g, at, 0, &fd));
    if (!npu_.w) {
        why = "no shared memory for the matrices (" + std::to_string(at >> 20) + " MB)";
        return false;
    }
    npu_.w_fd = fd;
    // filled a 32-row tile at a time on the engine's threads (through a tile of the thread's own: W is uncached)
    for (auto &[v, d] : mats) {
        const QView m = *v;
        const dec_mat_t dm = *d;
        uint8_t *dst = npu_.w + dm.w;
        const size_t tb = static_cast<size_t>(m.cols) * (dm.fmt == DEC_FMT_Q8 ? 34u : dm.fmt == DEC_FMT_Q4B ? 20u : 18u);
        pool_->parallel_for(m.rows / 32, [&](int64_t t0, int64_t t1) {
            std::vector<uint8_t> tile(tb);
            for (int64_t t = t0; t < t1; ++t) {
                const uint8_t *src = m.data + static_cast<size_t>(t) * 32 * m.row_bytes;
                if (dm.fmt == DEC_FMT_Q4T8) {
                    to_tile8x(DType::Q4, 32, m.cols, src, tile.data());
                } else if (dm.fmt == DEC_FMT_Q4B) {
                    // codes: per 64 inputs word j's vector, lane = row, its bytes the pack's bytes with each nibble
                    // made affine (level 2c + 1 / -(2(c - 8) + 1) -> code c + 8 / 15 - c); then the scales as bf16
                    // (an E4M3 value is exact in it) [chunk][block][row]
                    const int64_t nch = m.cols / 64;
                    uint8_t *codes = tile.data();
                    auto *scales = reinterpret_cast<uint16_t *>(tile.data() + nch * 1024);
                    static const auto aff = [] {
                        std::array<uint8_t, 256> t{};
                        for (int b = 0; b < 256; ++b) {
                            const int lo = b & 15, hi = b >> 4;
                            t[static_cast<size_t>(b)] =
                                static_cast<uint8_t>((lo < 8 ? lo + 8 : 15 - lo) | (hi < 8 ? hi + 8 : 15 - hi) << 4);
                        }
                        return t;
                    }();
                    for (int r = 0; r < 32; ++r) {
                        const uint8_t *row = src + static_cast<size_t>(r) * m.row_bytes;
                        for (int64_t ch = 0; ch < nch; ++ch) {
                            for (int j = 0; j < 8; ++j)
                                for (int bb = 0; bb < 4; ++bb)
                                    codes[ch * 1024 + j * 128 + r * 4 + bb] = aff[row[ch * 32 + j * 4 + bb]];
                            for (int b = 0; b < 4; ++b) {
                                float f = 0.0f;
                                {
                                    const uint8_t e = row[m.cols / 2 + ch * 4 + b];
                                    const int ex = (e >> 3) & 15, mt = e & 7;
                                    f = ex == 0 ? std::ldexp(static_cast<float>(mt), -9)
                                                : std::ldexp(1.0f + static_cast<float>(mt) / 8.0f, ex - 7);
                                    if (e & 0x80) f = -f;
                                }
                                uint32_t bits;
                                std::memcpy(&bits, &f, 4);
                                scales[(ch * 4 + b) * 32 + r] = static_cast<uint16_t>(bits >> 16);
                            }
                        }
                    }
                } else {
                    // per 64 inputs: 16 code vectors (lane = row: its 4 bytes = 4 inputs as code + 128), then the
                    // scales [block][row]
                    const int64_t nch = m.cols / 64;
                    for (int64_t ch = 0; ch < nch; ++ch) {
                        uint8_t *cb = tile.data() + static_cast<size_t>(ch) * 2176u;
                        for (int r = 0; r < 32; ++r) {
                            const uint8_t *row = src + static_cast<size_t>(r) * m.row_bytes;
                            for (int w = 0; w < 16; ++w)
                                for (int i = 0; i < 4; ++i)
                                    cb[w * 128 + r * 4 + i] = static_cast<uint8_t>(row[ch * 64 + w * 4 + i] ^ 0x80);
                            for (int b = 0; b < 4; ++b) cb[2048 + b * 32 + r] = row[m.cols + ch * 4 + b];
                        }
                    }
                }
                std::memcpy(dst + static_cast<size_t>(t) * tb, tile.data(), tb);
            }
        }, 4);
    }
    // P: the small parameters
    size_t pat = 0;
    std::vector<std::pair<size_t, std::pair<const void *, size_t>>> puts;
    const auto put = [&](const void *src, size_t bytes) -> uint32_t {
        const size_t o = pat;
        puts.push_back({o, {src, bytes}});
        pat += (bytes + 127) / 128 * 128;
        return static_cast<uint32_t>(o);
    };
    const size_t fH = static_cast<size_t>(H) * 4;
    for (int l = 0; l < c.n_layers; ++l) {
        const Layer &L = layers_[static_cast<size_t>(l)];
        dec_layer_t &D = cf.layers[l];
        D.in_norm = put(L.in_norm, fH);
        D.post_norm = put(L.post_norm, fH);
        if (L.attn) {
            D.q_norm = put(L.q_norm, static_cast<size_t>(c.head_dim) * 4);
            D.k_norm = put(L.k_norm, static_cast<size_t>(c.head_dim) * 4);
        } else {
            D.ab = put(L.a, static_cast<size_t>(nv) * H * 2);
            put(L.b, static_cast<size_t>(nv) * H * 2);   // right after a's rows (nv * H * 2 is a multiple of 128)
            D.A_log = put(L.A_log, static_cast<size_t>(nv) * 4);
            D.dt_bias = put(L.dt_bias, static_cast<size_t>(nv) * 4);
            D.conv = put(L.conv, static_cast<size_t>(conv_dim) * K * 4);
            D.gnorm = put(L.gnorm, static_cast<size_t>(c.gdn_v_dim) * 4);
        }
    }
    cf.final_norm = put(final_norm_, fH);
    cf.inv_freq = put(nullptr, static_cast<size_t>(cf.half) * 4);   // (open_npu_decode: the frequencies)
    if (cf.mtp) {
        cf.mtp_pre_e = put(mtp_pre_e_, fH);
        cf.mtp_pre_h = put(mtp_pre_h_, fH);
        cf.mtp_norm = put(mtp_norm_, fH);
        dec_layer_t &D = cf.mtp_layer;
        D.in_norm = put(mtp_layer_.in_norm, fH);
        D.post_norm = put(mtp_layer_.post_norm, fH);
        D.q_norm = put(mtp_layer_.q_norm, static_cast<size_t>(c.head_dim) * 4);
        D.k_norm = put(mtp_layer_.k_norm, static_cast<size_t>(c.head_dim) * 4);
        if (!draft_ids.empty()) cf.draft_ids = put(draft_ids.data(), draft_ids.size() * 4);
    }
    if ((static_cast<size_t>(nv) * H * 2) % 128) {
        why = "a / b rows not 128-byte multiples";
        return false;
    }
    npu_.p = static_cast<uint8_t *>(npu_.shared(npu_.g, pat, 0, &fd));
    if (!npu_.p) {
        why = "no shared memory for the parameters";
        return false;
    }
    npu_.p_fd = fd;
    for (const auto &[o, sb] : puts)
        if (sb.first) std::memcpy(npu_.p + o, sb.first, sb.second);
    npu_.dec_planned = true;
    npu_.dec_status = "planned: " + std::to_string(at >> 20) + " MB of matrices";
    return true;
#else
    why = "no NPU here";
    return false;
#endif
}

bool Model::open_npu_decode(std::string &why) {
#if defined(__ANDROID__)
    const auto &c = cfg_;
    auto &cf = *reinterpret_cast<dec_cfg_t *>(npu_.cfg.data());
    float *fr = reinterpret_cast<float *>(npu_.p + cf.inv_freq);
    for (int i = 0; i < cf.half; ++i)
        fr[i] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / static_cast<float>(c.rope_dim));
    const int threads = std::getenv("GEODE_NPU_DEC_THREADS") ? std::atoi(std::getenv("GEODE_NPU_DEC_THREADS")) : 6;
    const int e = npu_.dec_open(npu_.g, npu_.cfg.data(), static_cast<int>(npu_.cfg.size()), npu_.w_fd, npu_.p_fd,
                                npu_.s_fd, threads);
    if (e) {
        char b[64];
        std::snprintf(b, sizeof b, "the DSP's side did not open (0x%x)", static_cast<unsigned>(e));
        why = b;
        return false;
    }
    npu_.dec = true;
    for (size_t i = 0; i < kv_ptrs_.size(); ++i)
        if (npu_.dec_kv(npu_.g, static_cast<int>(i), npu_.kv_fds[kv_ptrs_[i]])) {
            npu_.dec = false;
            why = "the DSP could not take the KV cache";
            return false;
        }
    // MTP drafts verified in pairs on the DSP (GEODE_NPU_SPEC=0: one token at a time): a pair's products in q4b cost
    // 1.2 a token's, the drafts make up for it (Qwen3.5-2B 30 -> 35 tok/s)
    npu_spec_ = cf.mtp != 0 && !(std::getenv("GEODE_NPU_SPEC") && std::atoi(std::getenv("GEODE_NPU_SPEC")) == 0);
    // (the app reads "on" and "prompts" from it: where the answers and the prompts run)
    npu_.dec_status = "on: " + std::to_string(threads) + " DSP threads" +
                      (npu_spec_ ? ", MTP drafts (" + std::to_string(cf.n_draft) + " rows)" : "");
    if (npu_.prefill) npu_.dec_status += npu_.mtp_kv ? ", prompts (and the MTP layer's)" : ", prompts";
    return true;
#else
    why = "no NPU here";
    return false;
#endif
}

bool Model::prefill_npu(const float *X, int n, float *logits, std::string &err) {
    const auto &c = cfg_;
    const int H = c.hidden, V = c.vocab;
    const double t0 = now_seconds();
    static const int debug = std::getenv("GEODE_NPU_DEC_FLAGS") ? std::atoi(std::getenv("GEODE_NPU_DEC_FLAGS")) : 0;
    const auto &cf = *reinterpret_cast<const dec_cfg_t *>(npu_.cfg.data());
    const size_t lrow = (static_cast<size_t>(V) * 4 + 127) / 128 * 128, hrow = (static_cast<size_t>(H) * 4 + 127) / 128 * 128;
    const uint8_t *sc = npu_.s + cf.scratch;
    hid_long_.resize(static_cast<size_t>(n) * H);
    double dsp_all = 0;
    for (int o = 0; o < n; o += DEC_N) {
        const int m = std::min(DEC_N, n - o);
        int p3[3 * DEC_N];
        for (int t = 0; t < m; ++t) rope_pos(pos_ + o + t, p3 + 3 * t);
        const int set = conv_st_ == npu_.conv_set[1] ? DEC_SET1 : 0;
        double dsp = 0;
        const int e = npu_.dec_prefill(npu_.g, X + static_cast<size_t>(o) * H, m, H, pos_ + o, p3, set | debug, &dsp);
        if (e) {
            char b[64];
            std::snprintf(b, sizeof b, "failed (0x%x)", static_cast<unsigned>(e));
            // the states are half done: the conversation is lost; the CPU reads the next ones
            npu_.prefill = false;
            npu_.dec_status = std::string("prompt ") + b;
            err = std::string("the DSP's prompt ") + b;
            return false;
        }
        dsp_all += dsp;
        for (int t = 0; t < m; ++t)
            std::memcpy(hid_long_.data() + static_cast<size_t>(o + t) * H, sc + DEC_T * lrow + t * hrow,
                        static_cast<size_t>(H) * sizeof(float));
    }
    if (logits) std::memcpy(logits, sc, static_cast<size_t>(V) * sizeof(float));
    if (mtp_) last_hid_.assign(hid_long_.end() - H, hid_long_.end());
    pos_ += n;
    prof_.npu_pre += now_seconds() - t0;
    prof_.npu_pre_dsp += dsp_all;
    prof_.npu_pre_tokens += static_cast<uint64_t>(n);
    return true;
}

bool Model::mtp_kv_npu(const int *next_tokens, const float *const *hidden, int n, int pos, std::string &err) {
    const int H = cfg_.hidden;
    static const int debug = std::getenv("GEODE_NPU_DEC_FLAGS") ? std::atoi(std::getenv("GEODE_NPU_DEC_FLAGS")) : 0;
    std::vector<float> e(static_cast<size_t>(std::min(n, DEC_N)) * H), h(e.size());
    for (int o = 0; o < n; o += DEC_N) {
        const int m = std::min(DEC_N, n - o);
        int p3[3 * DEC_N];
        for (int t = 0; t < m; ++t) {
            const int at = o + t;
            float *row = e.data() + static_cast<size_t>(t) * H;
            // the token after an image pad's position is its features (as forward_mtp's)
            const float *im = vision_.available() && next_tokens[at] == vision_.shape().pad_token
                                  ? image_row(pos + at + 1)
                                  : nullptr;
            if (im) {
                std::copy_n(im, H, row);
            } else if (!pack_.embedding(next_tokens[at], row)) {
                err = "cannot read the embedding of token " + std::to_string(next_tokens[at]);
                return false;
            }
            std::copy_n(hidden[at], H, h.data() + static_cast<size_t>(t) * H);
            rope_pos(pos + at, p3 + 3 * t);
        }
        double dsp = 0;
        const int r = npu_.dec_mtp_kv(npu_.g, e.data(), h.data(), m, H, pos + o, p3, debug, &dsp);
        if (r) {
            char b[64];
            std::snprintf(b, sizeof b, "failed (0x%x)", static_cast<unsigned>(r));
            // no drafts for this conversation (the answer is the same); the CPU's pass from the next one
            npu_.mtp_kv = false;
            err = std::string("the DSP's MTP pass ") + b;
            return false;
        }
    }
    return true;
}

bool Model::decode_npu(const int *tokens, int n, float *logits, std::string &err) {
    const auto &c = cfg_;
    const int H = c.hidden, V = c.vocab;
    const double t0 = now_seconds();
    std::vector<float> x(static_cast<size_t>(n) * H);
    int p3[3 * DEC_T];
    for (int t = 0; t < n; ++t) {
        rope_pos(pos_ + t, p3 + 3 * t);
        if (!pack_.embedding(tokens[t], x.data() + static_cast<size_t>(t) * H)) {
            err = "cannot read the embedding of token " + std::to_string(tokens[t]);
            return false;
        }
    }
    // GEODE_NPU_DEC_FLAGS: dec_cfg.h's DEC_DEBUG_* (finding faults)
    static const int debug = std::getenv("GEODE_NPU_DEC_FLAGS") ? std::atoi(std::getenv("GEODE_NPU_DEC_FLAGS")) : 0;
    const int set = conv_st_ == npu_.conv_set[1] ? DEC_SET1 : 0;
    double dsp = 0;
    const int e = npu_.dec_step(npu_.g, x.data(), n, H, pos_, p3, set | debug, &dsp);
    if (e) {
        char b[64];
        std::snprintf(b, sizeof b, "failed (0x%x)", static_cast<unsigned>(e));
        npu_.dec = npu_spec_ = false;   // the CPU from the next conversation (this one's states are half done)
        npu_.dec_status = b;
        err = std::string("the DSP's step ") + b;
        return false;
    }
    // the logits and the hidden rows where the DSP left them (dec_cfg.h: the scratch's start)
    const auto &cf = *reinterpret_cast<const dec_cfg_t *>(npu_.cfg.data());
    const size_t lrow = (static_cast<size_t>(V) * 4 + 127) / 128 * 128, hrow = (static_cast<size_t>(H) * 4 + 127) / 128 * 128;
    const uint8_t *sc = npu_.s + cf.scratch;
    if (logits)
        for (int t = 0; t < n; ++t)
            std::memcpy(logits + static_cast<size_t>(t) * V, sc + t * lrow, static_cast<size_t>(V) * sizeof(float));
    if (mtp_) {
        if (hid_.size() < static_cast<size_t>(n) * H) hid_.resize(static_cast<size_t>(n) * H);
        hid_long_ = {};
        for (int t = 0; t < n; ++t)
            std::memcpy(hid_.data() + static_cast<size_t>(t) * H, sc + DEC_T * lrow + t * hrow, static_cast<size_t>(H) * 4);
        last_hid_.assign(hid_.begin() + static_cast<ptrdiff_t>(n - 1) * H, hid_.begin() + static_cast<ptrdiff_t>(n) * H);
    }
    pos_ += n;
    prof_.npu_dec += now_seconds() - t0;
    prof_.npu_dec_dsp += dsp;
    prof_.npu_dec_steps += static_cast<uint64_t>(n);
    return true;
}

Model::~Model() { store_.stop(); }

Model::Npu::~Npu() {
#if defined(__ANDROID__)
    if (g && close) close(g);
    if (lib) dlclose(lib);
#endif
}

void Model::npu_rekey() {
    if (!npu_.g) return;
    npu_.ids.clear();
    for (const auto &[m, id] : npu_.adopted) npu_.ids[m->data] = id;
}

bool Model::npu_products(const QView *const *ms, int count, int n, const float *const *xs, float *const *const *ys) {
    if (!npu_.g || npu_.products_off || n < kNpuMinTokens || count < 1 || count > 4) return false;
    int ids[4];
    for (int j = 0; j < count; ++j) {
        const auto it = npu_.ids.find(ms[j]->data);
        if (it == npu_.ids.end()) return false;
        ids[j] = it->second;
    }
    const double t0 = now_seconds();
    const int e = npu_.gemm_multi(npu_.g, count, ids, n, xs, ys);
    npu_.wall += now_seconds() - t0;
    npu_.calls += static_cast<uint64_t>(count);
    if (!e) return true;
    npu_.status = "failed (" + std::to_string(e) + "): the CPU from now on";
    if (!npu_.moe_slabs.empty()) {   // the session holds part of the expert cache: kept, its products left alone
        npu_.products_off = true;
        return false;
    }
    npu_.close(npu_.g);
    npu_.g = nullptr;
    return false;
}

void Model::matmul_many(const QView *const *ms, int count, int n, const float *const *xs, float *const *const *ys,
                        const QAct *const *pre) {
    if (npu_products(ms, count, n, xs, ys)) return;
    for (int j = 0; j < count; ++j) matmul(*ms[j], n, xs, ys[j], pre);
}

std::string Model::npu_status() const {
    std::string s = npu_.status;
    if (npu_.calls) {
        char b[200];
        std::snprintf(b, sizeof b, "; %llu products, %.2f s (DSP %.2f s, conversions %.2f s)",
                      static_cast<unsigned long long>(npu_.calls), npu_.wall, npu_.g ? npu_.dsp(npu_.g) : 0.0,
                      npu_.g ? npu_.conv(npu_.g) : 0.0);
        s += b;
    }
    if (npu_.moe_on || prof_.npu_moe_experts) {
        size_t mb = 0;
        for (const auto &sl : npu_.moe_slabs) mb += sl.bytes >> 20;
        char b[200];
        std::snprintf(b, sizeof b, "; experts: %zu MB of the cache in its reach, %llu on it (%.2f s, DSP %.2f s), %llu on the GPU",
                      mb, static_cast<unsigned long long>(prof_.npu_moe_experts), prof_.npu_moe, prof_.npu_moe_dsp,
                      static_cast<unsigned long long>(prof_.gpu_moe_experts));
        s += b;
    }
    if (npu_.g) {
        s += "; decoder " + npu_.dec_status;
        if (prof_.npu_pre_tokens) {
            char b[160];
            std::snprintf(b, sizeof b, ", prompt %llu tokens %.1f tok/s (DSP %.2f s)",
                          static_cast<unsigned long long>(prof_.npu_pre_tokens),
                          static_cast<double>(prof_.npu_pre_tokens) / std::max(1e-9, prof_.npu_pre), prof_.npu_pre_dsp);
            s += b;
        }
        if (prof_.npu_dec_steps) {
            char b[160];
            std::snprintf(b, sizeof b, ", %llu tokens, %.2f ms a token (DSP %.2f)",
                          static_cast<unsigned long long>(prof_.npu_dec_steps),
                          1e3 * prof_.npu_dec / static_cast<double>(prof_.npu_dec_steps),
                          1e3 * prof_.npu_dec_dsp / static_cast<double>(prof_.npu_dec_steps));
            s += b;
        }
        if (npu_.dec && std::getenv("GEODE_NPU_DEC_PROF")) {   // the DSP's own time by stage (dec_cfg.h DEC_PROF_*)
            const auto &cf = *reinterpret_cast<const dec_cfg_t *>(npu_.cfg.data());
            const auto *pr = reinterpret_cast<const uint64_t *>(npu_.s + cf.scratch + cf.scratch_bytes - 128);
            const double n = pr[DEC_PROF_STEPS] ? static_cast<double>(pr[DEC_PROF_STEPS]) : 1.0;
            static const char *const names[] = {"norm", "products", "conv", "gdn", "qk", "attn", "join", "mlp",
                                                "argmax", "cache", "", "mtp"};
            s += " [ms a call:";
            for (int i = 0; i <= DEC_PROF_MTP; ++i) {
                if (i == DEC_PROF_STEPS) continue;
                char b[48];
                std::snprintf(b, sizeof b, " %s %.2f", names[i], static_cast<double>(pr[i]) / n / 1e3);
                s += b;
            }
            if (pr[DEC_PROF_QCHECK]) s += " qcheck " + std::to_string(pr[DEC_PROF_QCHECK]);
            s += "]";
        }
    }
    return s;
}

void Model::quad_dense() {
    // the rest of the dense matrices into the GPU layout too, in place: the CPU's kernels read it faster (kernels.cpp,
    // quad_lanes). Each tensor once, whichever views share it.
    std::vector<QView *> views;
    auto add = [&](Layer &L) {
        for (QView *m : {&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.sh_gate, &L.sh_up, &L.sh_down})
            views.push_back(m);
    };
    for (Layer &L : layers_) add(L);
    if (mtp_) {
        add(mtp_layer_);
        views.push_back(&mtp_fc_);
    }
    views.push_back(&lm_head_);
    // GEODE_TILE8=1: q4 in tile8 instead (the Adreno 830's int8 kernels' layout; the CPU reads it a little slower
    // one token at a time)
    const bool tile8 = std::getenv("GEODE_TILE8") && std::atoi(std::getenv("GEODE_TILE8")) != 0;
    std::vector<const uint8_t *> done;
    for (QView *m : views) {
        if (!m->data || m->quad || !quad_capable(m->type, m->rows, m->cols)) continue;
        const bool t8 = tile8 && tile8_capable(m->type, m->rows, m->cols);
        const uint8_t *d = m->data;
        if (std::find(done.begin(), done.end(), d) == done.end()) {
            // quads are independent: split across the pool
            auto *w = const_cast<uint8_t *>(d);
            const size_t rb = m->row_bytes;
            const QView v = *m;
            const int64_t g = t8 ? 32 : 4;   // rows a group: groups are independent, split across the pool
            pool_->parallel_for(v.rows / g, [&](int64_t a, int64_t b) {
                uint8_t *at = w + static_cast<size_t>(g * a) * rb;
                if (t8) to_tile8(v.type, g * (b - a), v.cols, at, at);
                else to_quad(v.type, g * (b - a), v.cols, at, at);
            }, t8 ? 8 : 64);
            done.push_back(d);
        }
        m->quad = true;
        m->tile8 = t8;
    }
}

int64_t Model::gpu_rows(const QView &m, int n) const {
    if (!gpu_ || gpu_failed_ || !gpu_->holds(m)) return 0;
    float share = gpu_share_;
    if (n < kGpuMinTokens) {
        // generating (gpu_decode_): a token or a pair, every pass or the verified pairs only
        if (n > 2 || !(gpu_decode_ == 1 || (gpu_decode_ == 2 && verify_keep_))) return 0;
        share = gpu_dec_share_;
    }
    return std::min(m.rows, static_cast<int64_t>(static_cast<float>(m.rows) * share) / 16 * 16);
}

namespace {
enum EmuKind { kEmuQkv, kEmuZ, kEmuOut, kEmuQ, kEmuK, kEmuV, kEmuO, kEmuShGate, kEmuShUp, kEmuShDown, kEmuHead,
               kEmuMtp, kEmuExpert, kEmuKinds };
const char *const kEmuNames[kEmuKinds] = {"qkv", "z", "out", "q", "k", "v", "o", "shg", "shu", "shd", "head", "mtp",
                                          "exp"};
}  // namespace

bool Model::emulate_this(const QView &m) {
    static const char *kinds_env = std::getenv("GEODE_EMU_KINDS"), *layers_env = std::getenv("GEODE_EMU_LAYERS"),
                      *alt_env = std::getenv("GEODE_EMU_FMT2_LAYERS");
    emu_alt_ = false;
    if (!kinds_env && !layers_env && !alt_env) return true;   // every product, as before
    if (emu_kinds_.empty()) {
        auto put = [&](const QView &v, int kind, int layer) {
            if (v.data) emu_kinds_[v.data] = {kind, layer};
        };
        for (const Layer &L : layers_) {
            const int l = static_cast<int>(&L - layers_.data());
            put(L.qkv, kEmuQkv, l), put(L.z, kEmuZ, l), put(L.out, kEmuOut, l);
            put(L.q, kEmuQ, l), put(L.k, kEmuK, l), put(L.v, kEmuV, l), put(L.o, kEmuO, l);
            put(L.sh_gate, kEmuShGate, l), put(L.sh_up, kEmuShUp, l), put(L.sh_down, kEmuShDown, l);
        }
        put(lm_head_, kEmuHead, -1);
        if (mtp_) {
            for (const QView *v : {&mtp_layer_.q, &mtp_layer_.k, &mtp_layer_.v, &mtp_layer_.o, &mtp_layer_.sh_gate,
                                   &mtp_layer_.sh_up, &mtp_layer_.sh_down, &mtp_fc_})
                put(*v, kEmuMtp, -1);
        }
    }
    const auto it = emu_kinds_.find(m.data);
    const int kind = it == emu_kinds_.end() ? kEmuExpert : it->second.first;   // an expert's view is its cache slot
    const int layer = it == emu_kinds_.end() ? -2 : it->second.second;         // (its layer is not known here)
    if (kinds_env) {
        bool hit = false;
        for (const char *p = kinds_env; *p;) {
            const char *e = std::strchr(p, ',');
            const size_t len = e ? static_cast<size_t>(e - p) : std::strlen(p);
            if (std::strlen(kEmuNames[kind]) == len && std::strncmp(p, kEmuNames[kind], len) == 0) hit = true;
            // an expert's two products apart: expgu (gate_up, reads the layer input), expd (down, reads silu*up)
            const char *sub = kind != kEmuExpert ? "" : m.cols == cfg_.hidden ? "expgu" : "expd";
            if (*sub && std::strlen(sub) == len && std::strncmp(p, sub, len) == 0) hit = true;
            p = e ? e + 1 : p + len;
        }
        if (!hit) return false;
    }
    if (layers_env && layer >= 0) {
        int a = 0, b = 1 << 30;
        std::sscanf(layers_env, "%d-%d", &a, &b);
        if (layer < a || layer > b) return false;
    }
    // GEODE_EMU_FMT2_LAYERS (a-b): the dense products of those layers rounded as GEODE_EMU_FMT2 says instead
    if (alt_env && layer >= 0) {
        int a = 0, b = -1;
        std::sscanf(alt_env, "%d-%d", &a, &b);
        emu_alt_ = layer >= a && layer <= b;
    }
    return true;
}

void Model::emulate_round(const float *x, int64_t cols, float *dst) {
    static const char *fmt_env = std::getenv("GEODE_EMU_FMT"), *fmt2_env = std::getenv("GEODE_EMU_FMT2");
    // GEODE_EMU_FMT1: the single-token products' rounding (generating after a prompt read as GEODE_EMU_FMT says)
    static const char *fmt1_env = std::getenv("GEODE_EMU_FMT1");
    const std::string fmt = emu_single_ && fmt1_env ? fmt1_env
                            : emu_alt_ && fmt2_env  ? fmt2_env
                            : fmt_env               ? fmt_env
                                                    : "i8";
    if (fmt == "f32") {   // not rounded at all
        std::memcpy(dst, x, static_cast<size_t>(cols) * sizeof(float));
        return;
    }
    if (fmt == "i8") {   // exactly the int8 kernels' rounding (quantize_activation)
        quantize_activation(x, cols, qacts_[0]);
        for (int64_t i = 0; i < cols; ++i)
            dst[i] = qacts_[0].q[static_cast<size_t>(i)] * qacts_[0].scale[static_cast<size_t>(i / kBlock)];
        return;
    }
    if (fmt == "f16") {
        for (int64_t i = 0; i < cols; ++i) dst[i] = static_cast<float>(static_cast<_Float16>(x[i]));
        return;
    }
    // symmetric blocks of B values: amax / L a step (L = 127, or 32767 for i16); i8o keeps each block's largest
    // value exact and scales the rest by the second largest
    const bool outlier = fmt == "i8o";
    const int B = fmt.rfind("i8b", 0) == 0 ? std::max(1, std::atoi(fmt.c_str() + 3)) : kBlock;
    const float L = fmt == "i16" ? 32767.0f : 127.0f;
    for (int64_t b0 = 0; b0 < cols; b0 += B) {
        const int64_t b1 = std::min<int64_t>(cols, b0 + B);
        int64_t top = b0;
        float amax = 0.0f, second = 0.0f;
        for (int64_t i = b0; i < b1; ++i) {
            const float v = std::fabs(x[i]);
            if (v > amax) second = amax, amax = v, top = i;
            else if (v > second) second = v;
        }
        const float m = outlier ? second : amax;
        const float inv = m > 0.0f ? L / m : 0.0f, s = m / L;
        for (int64_t i = b0; i < b1; ++i) {
            if (outlier && i == top) {
                dst[i] = x[i];
                continue;
            }
            dst[i] = static_cast<float>(std::clamp(std::lrint(x[i] * inv), -static_cast<long>(L), static_cast<long>(L))) * s;
        }
    }
}

void Model::matmul(const QView &m, int n, const float *const *xs, float *const *ys, const QAct *const *pre) {
    if (exact_) {
        std::vector<const float *> in(xs, xs + n);
        emu_single_ = n == 1;
        if ((emulate_ || emulate_bf16_) && (emulate_bf16_ || emulate_this(m))) {
            xq_.resize(static_cast<size_t>(n) * m.cols);
            for (int t = 0; t < n; ++t) {
                float *dst = xq_.data() + static_cast<size_t>(t) * m.cols;
                if (emulate_bf16_) {
                    // round to nearest even BF16: the precision GPUs feed their matrix units
                    for (int64_t i = 0; i < m.cols; ++i) {
                        uint32_t u;
                        std::memcpy(&u, xs[t] + i, 4);
                        u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
                        std::memcpy(dst + i, &u, 4);
                    }
                } else {
                    emulate_round(xs[t], m.cols, dst);   // the int8 kernels' rounding unless GEODE_EMU_FMT says
                }
                in[static_cast<size_t>(t)] = dst;
            }
        }
        par(m.rows, [&](int64_t r0, int64_t r1) {
            for (int t = 0; t < n; ++t) matvec_rows(m, in[static_cast<size_t>(t)], ys[t], r0, r1);
        }, 16);
        return;
    }
    // the NPU: a prompt's product there whole (the matrices it adopted)
    if (npu_.g && n >= kNpuMinTokens) {
        const QView *one[1] = {&m};
        float *const *out[1] = {ys};
        if (npu_products(one, 1, n, xs, out)) return;
    }
    // the Adreno 7xx+: a batch's product there whole
    if (gpu8_ && n >= kGpuMinTokens && !check_ && gpu8_->fits_batch(m, n)) {
        std::string e;
        const double t0 = now_seconds();
        if (gpu8_->gemm(m, n, xs, ys, e)) {
            prof_.gpu8 += now_seconds() - t0;
            return;
        }
        gpu_status_ = "failed: " + e;
        gpu8_.reset();   // the CPU from now on (its kernels read tile8)
        gpu8_experts_ = false;
    }
    // the GPU's rows first (it runs while the CPU quantizes and computes the rest); generating, launched once the
    // CPU's threads are on their rows (the caller's `first`: the launch would hold them all up)
    int64_t r0 = gpu_rows(m, n);
    const bool dec = r0 > 0 && n < kGpuMinTokens;
    if (r0 > 0 && !dec) {
        std::string e;
        if (!gpu_->start(m, r0, n, xs, e)) {
            gpu_failed_ = true;
            gpu_status_ = e;
            r0 = 0;
        }
    }
    std::vector<const QAct *> acts(static_cast<size_t>(n));
    if (pre) {
        for (int t = 0; t < n; ++t) acts[static_cast<size_t>(t)] = pre[t];
    } else {
        par(n, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) quantize_activation(xs[t], m.cols, qacts_[static_cast<size_t>(t)]);
        }, 8);
        for (int t = 0; t < n; ++t) acts[static_cast<size_t>(t)] = &qacts_[static_cast<size_t>(t)];
    }
    bool launched = !dec;
    std::string le;
    const std::function<void()> launch = [&] {
        const double t0 = now_seconds();
        launched = gpu_->start(m, r0, n, xs, le);
        prof_.g_start += now_seconds() - t0;
    };
    pool_->parallel_for(m.rows - r0, [&](int64_t a, int64_t b) {
        matvec_rows_q_multi(m, acts.data(), n, ys, r0 + a, r0 + b);
    }, 16, wide_ && m.rows >= kWideRows, dec ? &launch : nullptr);
    if (!launched) {   // the CPU's then
        gpu_failed_ = true;
        gpu_status_ = le;
        pool_->parallel_for(r0, [&](int64_t a, int64_t b) { matvec_rows_q_multi(m, acts.data(), n, ys, a, b); }, 16);
        r0 = 0;
    }
    if (r0 > 0) {
        std::string e;
        double waited = 0.0;
        bool late = false;
        const double t0 = now_seconds();
        const bool fin = gpu_->finish(ys, e, dec, &waited, &late);
        if (dec) {
            ++prof_.g_dec;
            prof_.g_late += late ? 1 : 0;
            prof_.g_wait += waited;
            prof_.g_out += now_seconds() - t0 - waited;
        }
        if (!fin) {   // the CPU does the GPU's rows after all
            gpu_failed_ = true;
            gpu_status_ = e;
            pool_->parallel_for(r0, [&](int64_t a, int64_t b) { matvec_rows_q_multi(m, acts.data(), n, ys, a, b); }, 16,
                                wide_);
        }
    }
    if (check_) {
        // every int8 product against the exact product of the same rounded activations (geode-parity --int8)
        std::vector<float> xq(static_cast<size_t>(m.cols)), e(static_cast<size_t>(m.rows));
        for (int t = 0; t < n; ++t) {
            const QAct &a = *acts[static_cast<size_t>(t)];
            for (int64_t i = 0; i < m.cols; ++i)
                xq[static_cast<size_t>(i)] = a.q[static_cast<size_t>(i)] * a.scale[static_cast<size_t>(i / kBlock)];
            matvec_rows(m, xq.data(), e.data(), 0, m.rows);
            double d = 0.0, r = 0.0;
            for (int64_t i = 0; i < m.rows; ++i) {
                const double x = static_cast<double>(ys[t][i]) - e[static_cast<size_t>(i)];
                d += x * x;
                r += static_cast<double>(e[static_cast<size_t>(i)]) * e[static_cast<size_t>(i)];
            }
            check_worst_ = std::max(check_worst_, std::sqrt(d / std::max(r, 1e-30)));
        }
    }
}

void Model::matmul_bf16(const uint16_t *w, int64_t rows, int64_t cols, int n, const float *const *xs, float *const *ys) {
    par(rows, [&](int64_t r0, int64_t r1) { matvec_rows_bf16_multi(w, cols, n, xs, ys, r0, r1); }, 16);
}

namespace {
// One token through a strip of W columns of one head's state (row i at S + i * stride): S = S * decay; delta =
// beta (v - S^T k); S += k delta^T; out = S^T q. The strip's sums are locals of a fixed width and nothing aliases, so
// they stay in vector registers and k[i], q[i] are loaded once per row.
template <int W>
void delta_strip(float *__restrict S, int dk, int stride, const float *__restrict q, const float *__restrict k,
                 const float *__restrict v, float decay, float beta, float *__restrict out) {
    float mem[W] = {};
    for (int i = 0; i < dk; ++i) {
        float *__restrict row = S + static_cast<size_t>(i) * stride;
        const float ki = k[i];
        for (int j = 0; j < W; ++j) {
            row[j] *= decay;
            mem[j] += row[j] * ki;
        }
    }
    for (int j = 0; j < W; ++j) mem[j] = (v[j] - mem[j]) * beta;   // delta
    float o[W] = {};
    for (int i = 0; i < dk; ++i) {
        float *__restrict row = S + static_cast<size_t>(i) * stride;
        const float ki = k[i], qi = q[i];
        for (int j = 0; j < W; ++j) {
            row[j] += ki * mem[j];
            o[j] += row[j] * qi;
        }
    }
    for (int j = 0; j < W; ++j) out[j] = o[j];
}
}  // namespace

void Model::gdn(const Layer &L, int n) {
    const auto &c = cfg_;
    const int H = c.hidden, nk = c.gdn_k_heads, nv = c.gdn_v_heads, dk = c.gdn_k_dim, dv = c.gdn_v_dim,
              K = c.conv_kernel;
    const int kd = nk * dk, vd = nv * dv, conv_dim = 2 * kd + vd;
    const size_t cw = static_cast<size_t>(std::max(vd, c.n_heads * c.head_dim));
    const auto hs = rows(h_, static_cast<size_t>(H), n);
    const auto ps = rows(proj_, static_cast<size_t>(conv_dim), n);
    const auto zs = rows(z_, static_cast<size_t>(vd), n);
    const auto as = rows(ab_, static_cast<size_t>(2 * nv), n), bs = rows(ab_, static_cast<size_t>(2 * nv), n, static_cast<size_t>(nv));
    const QAct *const *qh = quantize_h(hs, H);
    {
        const QView *ms[2] = {&L.qkv, &L.z};
        float *const *ys[2] = {ps.data(), zs.data()};
        matmul_many(ms, 2, n, hs.data(), ys, qh);
    }
    matmul_bf16(L.a, nv, H, n, hs.data(), as.data());
    matmul_bf16(L.b, nv, H, n, hs.data(), bs.data());

    float *cs = conv_st_ + static_cast<size_t>(L.index) * conv_dim * (K - 1);
    float *state = gdn_st_ + static_cast<size_t>(L.index) * nv * dk * dv;
    const int rep = nv / nk;
    const float qscale = 1.0f / std::sqrt(static_cast<float>(dk));

    // Every token's arithmetic is the one-token arithmetic (a batch gives the bits one token at a time gives); only
    // the loops are turned: channel by channel and head by head, each through all n tokens, so a head's state stays
    // in the core's cache for the whole batch and the threads meet once per stage, not once per token.

    const double tc = now_seconds();
    // the causal depthwise convolution over the last K inputs (state: the K-1 before this one, oldest first), in place
    // (each task a range of channels, token by token: rows are 32 KB apart, so walking one channel down the batch
    // would touch a new page and the same L1 set at every step; along a row the channels are contiguous)
    // a pair being verified (verify_keep_): the states after its first token kept in the backups, as each is done
    const bool keep = verify_keep_ && n == 2 && gdn_bk_;
    float *cbk = keep ? conv_bk_ + static_cast<size_t>(L.index) * conv_dim * (K - 1) : nullptr;
    float *sbk = keep ? gdn_bk_ + static_cast<size_t>(L.index) * nv * dk * dv : nullptr;
    par(conv_dim, [&](int64_t c0, int64_t c1) {
        for (int t = 0; t < n; ++t) {
            float *row = ps[static_cast<size_t>(t)];
            for (int64_t ch = c0; ch < c1; ++ch) {
                const float *w = L.conv + static_cast<size_t>(ch) * K;
                float *st = cs + static_cast<size_t>(ch) * (K - 1);
                const float in = row[ch];
                float acc = 0.0f;
                for (int k = 0; k < K - 1; ++k) acc += w[k] * st[k];
                acc += w[K - 1] * in;
                for (int k = 0; k + 1 < K - 1; ++k) st[k] = st[k + 1];
                if (K > 1) st[K - 2] = in;
                row[ch] = acc;
            }
            if (cbk && t == 0)
                std::memcpy(cbk + static_cast<size_t>(c0) * (K - 1), cs + static_cast<size_t>(c0) * (K - 1),
                            static_cast<size_t>(c1 - c0) * (K - 1) * sizeof(float));
            // silu apart, a loop that vectorizes (std::exp per channel was most of the convolution's time)
            for (int64_t ch = c0; ch < c1; ++ch) row[ch] = silu_fast(row[ch]);
        }
    }, 64);
    prof_.gdn_conv += now_seconds() - tc;
    // q and k: L2-normalised per key head (eps 1e-6), q also scaled by 1/sqrt(dk), in place
    par(static_cast<int64_t>(n) * nk, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            float *mixed = ps[static_cast<size_t>(i / nk)];
            const int kh = static_cast<int>(i % nk);
            float *q = mixed + kh * dk, *k = mixed + kd + kh * dk;
            float sq = 0.0f, sk = 0.0f;
            for (int j = 0; j < dk; ++j) {
                sq += q[j] * q[j];
                sk += k[j] * k[j];
            }
            const float iq = 1.0f / std::sqrt(sq + 1e-6f), ik = 1.0f / std::sqrt(sk + 1e-6f);
            for (int j = 0; j < dk; ++j) {
                q[j] = q[j] * iq * qscale;
                k[j] = k[j] * ik;
            }
        }
    });

    // each head's gate and step size, once per token (the strips below all use them)
    for (int t = 0; t < n; ++t) {
        float *a = as[static_cast<size_t>(t)], *b = bs[static_cast<size_t>(t)];
        for (int vh = 0; vh < nv; ++vh) {
            b[vh] = sigmoid(b[vh]);                                                   // beta
            a[vh] = std::exp(-std::exp(L.A_log[vh]) * softplus(a[vh] + L.dt_bias[vh]));   // exp(g), the decay
        }
    }

    // The gated delta rule: S = S*exp(g); S += k (beta (v - S^T k))^T; o = S^T q. Column j of a head's state S
    // [dk][dv] only ever meets column j of v, of the delta and of the output, so the state goes in strips of a few
    // columns (8 KB: they stay in L1) through all n tokens, one strip per task (delta_strip). A token gets the same
    // bits in any batch.
    const double ts = now_seconds();
    if (dv % 16) {
        // (no model has such heads yet; one strip of the whole width, the generic way)
        par(nv, [&](int64_t h0, int64_t h1) {
            std::vector<float> mem(static_cast<size_t>(dv)), o(static_cast<size_t>(dv));
            for (int64_t vh = h0; vh < h1; ++vh) {
                const int kh = static_cast<int>(vh) / rep;
                float *S = state + static_cast<size_t>(vh) * dk * dv;
                for (int t = 0; t < n; ++t) {
                    const float *mixed = ps[static_cast<size_t>(t)];
                    const float *q = mixed + kh * dk, *k = mixed + kd + kh * dk, *vv = mixed + 2 * kd + vh * dv;
                    const float decay = as[static_cast<size_t>(t)][vh], beta = bs[static_cast<size_t>(t)][vh];
                    std::fill(mem.begin(), mem.end(), 0.0f);
                    for (int i = 0; i < dk; ++i)
                        for (int j = 0; j < dv; ++j) {
                            S[i * dv + j] *= decay;
                            mem[static_cast<size_t>(j)] += S[i * dv + j] * k[i];
                        }
                    for (int j = 0; j < dv; ++j) mem[static_cast<size_t>(j)] = (vv[j] - mem[static_cast<size_t>(j)]) * beta;
                    std::fill(o.begin(), o.end(), 0.0f);
                    for (int i = 0; i < dk; ++i)
                        for (int j = 0; j < dv; ++j) {
                            S[i * dv + j] += k[i] * mem[static_cast<size_t>(j)];
                            o[static_cast<size_t>(j)] += S[i * dv + j] * q[i];
                        }
                    std::copy(o.begin(), o.end(), core_.data() + static_cast<size_t>(t) * cw + static_cast<size_t>(vh) * dv);
                    if (sbk && t == 0)
                        std::memcpy(sbk + static_cast<size_t>(vh) * dk * dv, S, static_cast<size_t>(dk) * dv * sizeof(float));
                }
            }
        });
    } else {
        constexpr int kStrip = 16;
        const int strips = dv / kStrip;
        par(static_cast<int64_t>(nv) * strips, [&](int64_t i0, int64_t i1) {
            for (int64_t task = i0; task < i1; ++task) {
                const int vh = static_cast<int>(task / strips), j0 = static_cast<int>(task % strips) * kStrip;
                const int kh = vh / rep;
                // a head's state is stored strip by strip, [dv / kStrip][dk][kStrip]: a strip is 8 contiguous KB
                // (rows dv apart would land in a few L1 sets and evict one another)
                float *S = state + static_cast<size_t>(vh) * dk * dv + static_cast<size_t>(j0) * dk;
                for (int t = 0; t < n; ++t) {
                    const float *mixed = ps[static_cast<size_t>(t)];
                    delta_strip<kStrip>(S, dk, kStrip, mixed + kh * dk, mixed + kd + kh * dk, mixed + 2 * kd + vh * dv + j0,
                                        as[static_cast<size_t>(t)][vh], bs[static_cast<size_t>(t)][vh],
                                        core_.data() + static_cast<size_t>(t) * cw + static_cast<size_t>(vh) * dv + j0);
                    if (sbk && t == 0)   // the strip, still in L1
                        std::memcpy(sbk + (S - state), S, static_cast<size_t>(dk) * kStrip * sizeof(float));
                }
            }
        });
    }
    prof_.gdn_strips += now_seconds() - ts;
    const double tn = now_seconds();
    // gated RMSNorm per token and head, in place: o / rms(o) * w * silu(z)
    par(static_cast<int64_t>(n) * nv, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            const int t = static_cast<int>(i / nv), vh = static_cast<int>(i % nv);
            float *o = core_.data() + static_cast<size_t>(t) * cw + static_cast<size_t>(vh) * dv;
            float ss = 0.0f;
            for (int j = 0; j < dv; ++j) ss += o[j] * o[j];
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dv) + c.eps);
            const float *z = zs[static_cast<size_t>(t)] + vh * dv;
            for (int j = 0; j < dv; ++j) o[j] = o[j] * inv * L.gnorm[j] * silu_fast(z[j]);
        }
    });
    prof_.gdn_norm += now_seconds() - tn;
    prof_.gdn_core += now_seconds() - tc;
    const auto cr = rows(core_, cw, n);
    const auto ms = rows(mix_, static_cast<size_t>(H), n);
    matmul(L.out, n, cr.data(), ms.data());
}

void Model::attention(const Layer &L, int n) {
    const auto &c = cfg_;
    const int H = c.hidden, nh = c.n_heads, nkv = c.n_kv_heads, hd = c.head_dim, half = c.rope_dim / 2;
    const size_t kv_row = static_cast<size_t>(nkv) * hd;
    const auto hs = rows(h_, static_cast<size_t>(H), n);
    const auto qs = rows(qg_, static_cast<size_t>(nh) * 2 * hd, n);   // per head: [query hd | gate hd]
    const auto ks = rows(kbuf_, kv_row, n);
    const auto vs = rows(vbuf_, kv_row, n);
    const QAct *const *qh = quantize_h(hs, H);
    {
        const QView *ms[3] = {&L.q, &L.k, &L.v};
        float *const *ys[3] = {qs.data(), ks.data(), vs.data()};
        matmul_many(ms, 3, n, hs.data(), ys, qh);
    }

    std::vector<float> cosv(static_cast<size_t>(half)), sinv(static_cast<size_t>(half));
    // partial RoPE on the first rope_dim dims (rotate_half pairs i and i + rope_dim/2)
    auto rope = [&](float *x) {
        for (int i = 0; i < half; ++i) {
            const float x1 = x[i], x2 = x[i + half];
            x[i] = x1 * cosv[static_cast<size_t>(i)] - x2 * sinv[static_cast<size_t>(i)];
            x[i + half] = x2 * cosv[static_cast<size_t>(i)] + x1 * sinv[static_cast<size_t>(i)];
        }
    };
    for (int t = 0; t < n; ++t) {
        // M-RoPE, interleaved: pair i turns at stream i % 3's position (text: the three are one)
        int p3[3];
        rope_pos(apos_ + t, p3);
        for (int i = 0; i < half; ++i) {
            const float f = static_cast<float>(p3[i % 3]) * inv_freq_[static_cast<size_t>(i)];
            cosv[static_cast<size_t>(i)] = std::cos(f);
            sinv[static_cast<size_t>(i)] = std::sin(f);
        }
        for (int hh = 0; hh < nh; ++hh) {
            float *q = qs[static_cast<size_t>(t)] + static_cast<size_t>(hh) * 2 * hd;
            rmsnorm(q, L.q_norm, q, hd, c.eps, true);
            rope(q);
        }
        for (int kh = 0; kh < nkv; ++kh) {
            float *k = ks[static_cast<size_t>(t)] + static_cast<size_t>(kh) * hd;
            rmsnorm(k, L.k_norm, k, hd, c.eps, true);
            rope(k);
        }
    }
    const double tc = now_seconds();
    if (exact_) attend<float>(L, n);
    else if (kv_i8_) attend<int8_t>(L, n);
    else attend<_Float16>(L, n);
    prof_.attn_core += now_seconds() - tc;
    const size_t cw = static_cast<size_t>(std::max(c.gdn_v_heads * c.gdn_v_dim, nh * hd));
    const auto cr = rows(core_, cw, n);
    const auto ms = rows(mix_, static_cast<size_t>(H), n);
    matmul(L.o, n, cr.data(), ms.data());
}

// Stores the batch's K and V, then each token attends over positions 0..its own (into core_):
// softmax(q.K / sqrt(hd)) V, times sigmoid(gate). T is the cache's element type: float for the exact path, _Float16
// otherwise (half the memory).
template <typename T>
T *Model::kv(int pos, int layer, bool value) {
    // chunk pos / kv_chunk_ holds [K, V][attention layer][kv_chunk_ positions][kv_heads * head_dim]
    T *base = reinterpret_cast<T *>(kv_ptrs_[static_cast<size_t>(pos / kv_chunk_)]);
    return base + ((static_cast<size_t>(value) * n_attn_ + static_cast<size_t>(layer)) * kv_chunk_ +
                   static_cast<size_t>(pos % kv_chunk_)) * kv_row_;
}

float *Model::kv_scale(int pos, int layer, bool value) {
    // chunk pos / kv_chunk_ (int8): [K, V][attention layer][kv_chunk_ positions][kv_row_] bytes, then the scales
    // [K, V][attention layer][kv_chunk_ positions][kv_heads] floats
    uint8_t *base = kv_ptrs_[static_cast<size_t>(pos / kv_chunk_)];
    float *sc = reinterpret_cast<float *>(base + 2 * static_cast<size_t>(n_attn_) * kv_chunk_ * kv_row_);
    return sc + ((static_cast<size_t>(value) * n_attn_ + static_cast<size_t>(layer)) * kv_chunk_ +
                 static_cast<size_t>(pos % kv_chunk_)) * static_cast<size_t>(kv_heads_);
}

namespace {
// a KV head's row to int8: amax / 127 its scale, round to nearest even (as the GPU's kv_store8)
inline void quant_kv_row(const float *x, int n, int8_t *q, float &scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < n; ++i)
        q[i] = static_cast<int8_t>(std::clamp(std::nearbyint(x[i] * inv), -127.0f, 127.0f));
    scale = amax / 127.0f;
}
}  // namespace

template <typename T>
void Model::attend(const Layer &L, int n) {
    const auto &c = cfg_;
    const int nh = c.n_heads, nkv = c.n_kv_heads, hd = c.head_dim;
    const size_t kv_row = static_cast<size_t>(nkv) * hd;
    for (int t = 0; t < n; ++t) {
        T *k = kv<T>(apos_ + t, L.index, false), *v = kv<T>(apos_ + t, L.index, true);
        if constexpr (std::is_same_v<T, int8_t>) {
            float *ks = kv_scale(apos_ + t, L.index, false), *vs = kv_scale(apos_ + t, L.index, true);
            for (int kh = 0; kh < nkv; ++kh) {
                const size_t o = static_cast<size_t>(kh) * hd;
                quant_kv_row(kbuf_.data() + static_cast<size_t>(t) * kv_row + o, hd, k + o, ks[kh]);
                quant_kv_row(vbuf_.data() + static_cast<size_t>(t) * kv_row + o, hd, v + o, vs[kh]);
            }
        } else {
            for (size_t i = 0; i < kv_row; ++i) {
                k[i] = static_cast<T>(kbuf_[static_cast<size_t>(t) * kv_row + i]);
                v[i] = static_cast<T>(vbuf_[static_cast<size_t>(t) * kv_row + i]);
            }
        }
    }

    // One task per (token, KV head, block of attn_block_ positions): the query heads sharing that KV head go together,
    // so each K and V row is read once for all of them, and a long context's positions spread over the threads
    // (Qwen3.5-2B has 2 KV heads: a token's 2 tasks left 6 of 8 threads idle, 14-19 ms a pass at 13.5k positions).
    // Each block keeps its heads' value sums, maximum and weights' sum; the blocks are joined in their order after,
    // so a token's bits depend neither on its batch, the threads nor the cache's chunks (one block: the bits of one
    // pass over the positions, as before the blocks).
    const int group = nh / nkv;
    const int nb = (apos_ + n + attn_block_ - 1) / attn_block_;   // the last token's blocks (the others have as many or fewer)
    const size_t part = static_cast<size_t>(hd) + 2;
    const size_t qw = static_cast<size_t>(nh) * 2 * hd;
    const size_t cw = static_cast<size_t>(std::max(c.gdn_v_heads * c.gdn_v_dim, nh * hd));
    // the batch's tokens a few at a time, the blocks' sums of at most ~16 MB at once (a long prompt's)
    const int per = std::max(1, static_cast<int>((size_t{16} << 20) / (sizeof(float) * nh * nb * part)));
    for (int t0 = 0; t0 < n; t0 += per) {
        const int tn = std::min(per, n - t0);
        const size_t need = static_cast<size_t>(tn) * nh * nb * part;
        if (attn_parts_.size() < need) attn_parts_.resize(need);
        attn_t0_ = t0;
        par(static_cast<int64_t>(tn) * nkv * nb, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int b = static_cast<int>(i % nb);
                const int kh = static_cast<int>(i / nb % nkv), t = t0 + static_cast<int>(i / nb / nkv);
                if (b * attn_block_ > apos_ + t) continue;   // past the token's own position
                int g = 0;
                for (; g + 8 <= group; g += 8) attend_heads<T, 8>(L.index, kh, kh * group + g, t, b, nb);
                for (; g + 4 <= group; g += 4) attend_heads<T, 4>(L.index, kh, kh * group + g, t, b, nb);
                for (; g < group; ++g) attend_heads<T, 1>(L.index, kh, kh * group + g, t, b, nb);
            }
        });
        // the blocks joined: each one's sums scaled by exp(its maximum - the maximum), softmax's division, the gate
        par(static_cast<int64_t>(tn) * nh, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int t = t0 + static_cast<int>(i / nh), h = static_cast<int>(i % nh);
                const int nbt = (apos_ + t) / attn_block_ + 1;
                const float *pp = attn_parts_.data() + (static_cast<size_t>(t - t0) * nh + h) * nb * part;
                float *o = core_.data() + static_cast<size_t>(t) * cw + static_cast<size_t>(h) * hd;
                const float *gate = qg_.data() + static_cast<size_t>(t) * qw + static_cast<size_t>(h) * 2 * hd + hd;
                float sum = pp[hd + 1];
                if (nbt == 1) {
                    std::memcpy(o, pp, sizeof(float) * static_cast<size_t>(hd));
                } else {
                    float mx = -INFINITY;
                    for (int b = 0; b < nbt; ++b) mx = std::max(mx, pp[static_cast<size_t>(b) * part + hd]);
                    std::fill(o, o + hd, 0.0f);
                    sum = 0.0f;
                    for (int b = 0; b < nbt; ++b) {
                        const float *q = pp + static_cast<size_t>(b) * part;
                        const float f = std::exp(q[hd] - mx);
                        sum += q[hd + 1] * f;
                        for (int j = 0; j < hd; ++j) o[j] += q[j] * f;
                    }
                }
                const float inv = 1.0f / sum;
                for (int j = 0; j < hd; ++j) o[j] = o[j] * inv * sigmoid(gate[j]);
            }
        });
    }
}

namespace {
#if defined(GEODE_NEON_ATTENTION)
// Eight values of a K or V row as two FP32 vectors.
inline void load8(const float *p, float32x4_t &a, float32x4_t &b) {
    a = vld1q_f32(p);
    b = vld1q_f32(p + 4);
}
inline void load8(const _Float16 *p, float32x4_t &a, float32x4_t &b) {
    const float16x8_t h = vld1q_f16(reinterpret_cast<const float16_t *>(p));
    a = vcvt_f32_f16(vget_low_f16(h));
    b = vcvt_high_f32_f16(h);
}
inline void load8(const int8_t *p, float32x4_t &a, float32x4_t &b) {
    const int16x8_t w = vmovl_s8(vld1_s8(p));
    a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
    b = vcvtq_f32_s32(vmovl_high_s16(w));
}

// The G scores q[g] . k of one position, the portable loop's arithmetic written out: its eight partial sums are
// lanes 0-3 and 4-7, fused multiply-adds, the same final order (the compiler had widened K one value at a time).
template <typename T, int G>
inline void qk_scores(const float *const *q, const T *k, int hd, float *out) {
    float32x4_t lo[G], hi[G];
    for (int g = 0; g < G; ++g) lo[g] = hi[g] = vdupq_n_f32(0.0f);
    for (int i = 0; i < hd; i += 8) {
        float32x4_t k0, k1;
        load8(k + i, k0, k1);
        for (int g = 0; g < G; ++g) {
            lo[g] = vfmaq_f32(lo[g], vld1q_f32(q[g] + i), k0);
            hi[g] = vfmaq_f32(hi[g], vld1q_f32(q[g] + i + 4), k1);
        }
    }
    for (int g = 0; g < G; ++g) {
        const float32x4_t s4 = vaddq_f32(lo[g], hi[g]);
        out[g] = (vgetq_lane_f32(s4, 0) + vgetq_lane_f32(s4, 1)) + (vgetq_lane_f32(s4, 2) + vgetq_lane_f32(s4, 3));
    }
}
#endif
}  // namespace

template <typename T, int G>
void Model::attend_heads(int layer, int kh, int h0, int t, int b, int nb) {
    // Token t's attention for query heads h0..h0+G-1, which share KV head kh, over block b of its positions: the
    // scores, their maximum, the weights exp(score - maximum) and their sum, the weighted sum of the values - attend
    // joins the blocks. Eight dims at a time, the G heads' sums in vector registers (NEON written out on the phone).
    // Nothing here depends on the batch: a token gets the same bits in any batch.
    const auto &c = cfg_;
    const int hd = c.head_dim;
    const size_t qw = static_cast<size_t>(c.n_heads) * 2 * hd;
    const int p0 = b * attn_block_, p1 = std::min(apos_ + t + 1, p0 + attn_block_), np = p1 - p0;
    // the block's positions a chunk of the cache at a time (inside one they are kv_row_ apart): f(s0, s1) for each
    const auto chunks = [&](auto &&f) {
        for (int s0 = p0; s0 < p1;) {
            const int s1 = std::min(p1, (s0 / kv_chunk_ + 1) * kv_chunk_);
            f(s0, s1);
            s0 = s1;
        }
    };
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    thread_local std::vector<float> scores;
    if (scores.size() < static_cast<size_t>(G) * np) scores.resize(static_cast<size_t>(G) * np);
    const float *q[G];
    float *sc[G], mx[G], sums[G];
    for (int g = 0; g < G; ++g) {
        q[g] = qg_.data() + static_cast<size_t>(t) * qw + static_cast<size_t>(h0 + g) * 2 * hd;
        sc[g] = scores.data() + static_cast<size_t>(g) * np;
        mx[g] = -INFINITY;
    }
    // the scores (sc indexed from p0)
    chunks([&](int s0, int s1) {
        const T *k = kv<T>(s0, layer, false) + static_cast<size_t>(kh) * hd;
        const float *ksc = nullptr;   // int8: the positions' K scales of this KV head
        if constexpr (std::is_same_v<T, int8_t>) ksc = kv_scale(s0, layer, false) + kh;
        for (int p = s0 - p0, pe = s1 - p0, r = 0; p < pe; ++p, ++r, k += kv_row_) {
            float dots[G];
#if defined(GEODE_NEON_ATTENTION)
            qk_scores<T, G>(q, k, hd, dots);
#else
            float acc[G][8] = {};
            for (int i = 0; i < hd; i += 8) {
                float kf[8];
                for (int j = 0; j < 8; ++j) kf[j] = static_cast<float>(k[i + j]);
                for (int g = 0; g < G; ++g)
                    for (int j = 0; j < 8; ++j) acc[g][j] += q[g][i + j] * kf[j];
            }
            for (int g = 0; g < G; ++g) {
                const float *a = acc[g];
                dots[g] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            }
#endif
            const float f = ksc ? scale * ksc[static_cast<size_t>(r) * kv_heads_] : scale;
            for (int g = 0; g < G; ++g) {
                const float sv = dots[g] * f;
                sc[g][p] = sv;
                mx[g] = std::max(mx[g], sv);
            }
        }
    });
    for (int g = 0; g < G; ++g) {
        float sum = 0.0f;
        for (int p = 0; p < np; ++p) {
            sc[g][p] = std::exp(sc[g][p] - mx[g]);
            sum += sc[g][p];
        }
        sums[g] = sum;
    }
    // int8: each position's V scale into its weights (the sums above are the softmax's own)
    if constexpr (std::is_same_v<T, int8_t>)
        chunks([&](int s0, int s1) {
            const float *vsc = kv_scale(s0, layer, true) + kh;
            for (int p = s0 - p0, pe = s1 - p0, r = 0; p < pe; ++p, ++r) {
                const float f = vsc[static_cast<size_t>(r) * kv_heads_];
                for (int g = 0; g < G; ++g) sc[g][p] *= f;
            }
        });
    // the values, eight dims at a time over the block's positions, into the block's part (attend joins them)
    const size_t part = static_cast<size_t>(hd) + 2;
    float *dst[G];
    for (int g = 0; g < G; ++g) {
        dst[g] = attn_parts_.data() + ((static_cast<size_t>(t - attn_t0_) * c.n_heads + (h0 + g)) * nb + b) * part;
        dst[g][hd] = mx[g];
        dst[g][hd + 1] = sums[g];
    }
    for (int i = 0; i < hd; i += 8) {
        float acc[G][8];
#if defined(GEODE_NEON_ATTENTION)
        // the same sums as the portable loop below, in vector registers
        float32x4_t lo[G], hi[G];
        for (int g = 0; g < G; ++g) lo[g] = hi[g] = vdupq_n_f32(0.0f);
        chunks([&](int s0, int s1) {
            const T *v = kv<T>(s0, layer, true) + static_cast<size_t>(kh) * hd + i;
            for (int p = s0 - p0, pe = s1 - p0; p < pe; ++p, v += kv_row_) {
                float32x4_t v0, v1;
                load8(v, v0, v1);
                for (int g = 0; g < G; ++g) {
                    lo[g] = vfmaq_n_f32(lo[g], v0, sc[g][p]);
                    hi[g] = vfmaq_n_f32(hi[g], v1, sc[g][p]);
                }
            }
        });
        for (int g = 0; g < G; ++g) {
            vst1q_f32(acc[g], lo[g]);
            vst1q_f32(acc[g] + 4, hi[g]);
        }
#else
        for (int g = 0; g < G; ++g)
            for (int j = 0; j < 8; ++j) acc[g][j] = 0.0f;
        chunks([&](int s0, int s1) {
            const T *v = kv<T>(s0, layer, true) + static_cast<size_t>(kh) * hd + i;
            for (int p = s0 - p0, pe = s1 - p0; p < pe; ++p, v += kv_row_) {
                float vf[8];
                for (int j = 0; j < 8; ++j) vf[j] = static_cast<float>(v[j]);
                for (int g = 0; g < G; ++g) {
                    const float w = sc[g][p];
                    for (int j = 0; j < 8; ++j) acc[g][j] += w * vf[j];
                }
            }
        });
#endif
        for (int g = 0; g < G; ++g) std::memcpy(dst[g] + i, acc[g], sizeof acc[g]);
    }
}

void route_topk(const float *logits_in, int E, int K, int *ids, float *weights, const uint8_t *res, float lambda,
                int *moved) {
    // softmax over all experts, the top K, renormalised (on a copy: the logits row may be reused; the scratch the
    // thread's own: a prompt's tokens are routed on every thread)
    thread_local std::vector<float> p;
    thread_local std::vector<int> order;
    p.assign(logits_in, logits_in + E);
    const float mx = *std::max_element(p.begin(), p.end());
    float sum = 0.0f;
    for (float &v : p) sum += (v = std::exp(v - mx));
    for (float &v : p) v /= sum;
    order.resize(static_cast<size_t>(E));
    std::iota(order.begin(), order.end(), 0);
    if (moved) *moved = 0;
    if (res && lambda > 0.0f) {
        // the route prior (Strata's route_prior.cu): the SELECTION on the logits with lambda added for the cached
        // experts, ties to the lower index as there; the weights below stay the router's probabilities
        thread_local std::vector<float> sel;
        thread_local std::vector<int> plain;
        sel.resize(static_cast<size_t>(E));
        for (int e = 0; e < E; ++e) sel[static_cast<size_t>(e)] = logits_in[e] + (res[e] ? lambda : 0.0f);
        std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int x, int y) {
            const float a = sel[static_cast<size_t>(x)], b = sel[static_cast<size_t>(y)];
            return a > b || (a == b && x < y);
        });
        if (moved) {   // how many picks the plain route would not have made
            plain.resize(static_cast<size_t>(E));
            std::iota(plain.begin(), plain.end(), 0);
            std::partial_sort(plain.begin(), plain.begin() + K, plain.end(),
                              [&](int x, int y) { return p[static_cast<size_t>(x)] > p[static_cast<size_t>(y)]; });
            for (int i = 0; i < K; ++i)
                if (std::find(plain.begin(), plain.begin() + K, order[static_cast<size_t>(i)]) == plain.begin() + K)
                    ++*moved;
        }
    } else {
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
                          [&](int x, int y) { return p[static_cast<size_t>(x)] > p[static_cast<size_t>(y)]; });
    }
    float top = 0.0f;
    for (int i = 0; i < K; ++i) top += p[static_cast<size_t>(order[static_cast<size_t>(i)])];
    for (int i = 0; i < K; ++i) {
        ids[i] = order[static_cast<size_t>(i)];
        if (weights) weights[i] = p[static_cast<size_t>(order[static_cast<size_t>(i)])] / top;
    }
}

void Model::route_one(const float *logits_in, int *ids, float *weights, const uint8_t *res, float lambda) {
    int moved = 0;
    route_topk(logits_in, cfg_.n_experts, k_, ids, weights, res, lambda, &moved);
    if (res && lambda > 0.0f) {
        prior_picks_ += static_cast<uint64_t>(k_);
        prior_moved_ += static_cast<uint64_t>(moved);
    }
}

bool Model::prior_on() const {
    // every expert (the MTP layer's too) fits: nothing to prefer
    return route_prior_ > 0.0f && !dec8_ &&
           store_.slots() < static_cast<size_t>(cfg_.n_layers + cfg_.mtp_layers) * static_cast<size_t>(cfg_.n_experts);
}

const uint8_t *Model::fill_residency(int l, std::vector<uint8_t> &buf) {
    const int E = cfg_.n_experts;
    buf.resize(static_cast<size_t>(E));
    store_.residency(static_cast<uint32_t>(l * E), static_cast<uint32_t>(E), buf.data());
    return buf.data();
}

ExpertRead Model::expert_read(int l, int e, DType &type) const {
    int file;
    uint64_t off;
    pack_.expert(l, e, file, off, type);
    const RandomFile *f = expert_files_[static_cast<size_t>(file)].get();
    size_t bytes = pack_.expert_block_bytes(type);
    // a whole number of 4 KiB pages lets the read go O_DIRECT (the slot has the room; blocks are 4 KiB-aligned)
    const size_t rounded = (bytes + 4095) / 4096 * 4096;
    if (f->direct() && off % 4096 == 0 && off + rounded <= f->size()) bytes = rounded;
    return ExpertRead{f, off, bytes};
}

int Model::skip_tail(int l, const int *ids, float *w, uint8_t *skip) {
    const int E = cfg_.n_experts, K = k_;
    if (tail_skip_ <= 0 || tail_skip_ >= K) return 0;
    // Strata (expert_source.cpp, tail skip): a miss among the last ranks costs a whole read for a few percent of the
    // output; left out, the other weights as they were (or renormalised, GEODE_TAIL_RENORM). Without w (the GPU
    // keeps the weights) only the marks.
    int n = 0;
    float kept = 0.0f;
    for (int k = 0; k < K; ++k) {
        if (k >= tail_skip_ && !store_.resident(static_cast<uint32_t>(l * E + ids[k]))) {
            skip[k] = 1;
            if (w) w[k] = 0.0f;
            ++n;
        }
        if (w) kept += w[k];
    }
    if (w && n > 0 && tail_renorm_ && kept > 0.0f)
        for (int k = 0; k < K; ++k) w[k] /= kept;
    tail_skipped_ += static_cast<uint64_t>(n);
    return n;
}

bool Model::moe(int l, const Layer &L, int n, std::string &err, bool keep_guesses, bool gpu_front) {
    const auto &c = cfg_;
    const int H = c.hidden, E = c.n_experts, K = k_, ff = c.moe_ff, sff = c.shared_ff;
    const auto hs = rows(h_, static_cast<size_t>(H), n);
    if (c.dense) {   // a dense model's MLP: down(silu(gate h) * up h), the block's whole output
        const auto sg = rows(sh_act_, static_cast<size_t>(2 * sff), n);
        const auto su = rows(sh_act_, static_cast<size_t>(2 * sff), n, static_cast<size_t>(sff));
        const auto out = rows(moe_, static_cast<size_t>(H), n);
        const double ts = now_seconds();
        const QAct *const *qh = quantize_h(hs, H);
        {
            const QView *ms[2] = {&L.sh_gate, &L.sh_up};
            float *const *ys[2] = {sg.data(), su.data()};
            matmul_many(ms, 2, n, hs.data(), ys, qh);
        }
        const auto act = [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t)
                for (int i = 0; i < sff; ++i)
                    sg[static_cast<size_t>(t)][i] = silu(sg[static_cast<size_t>(t)][i]) * su[static_cast<size_t>(t)][i];
        };
        if (n > kSpreadTokens) par(n, act, 8);
        else act(0, n);
        matmul(L.sh_down, n, sg.data(), out.data());
        prof_.shared += now_seconds() - ts;
        return true;
    }

    // the router, for every token: on the CPU, or (gpu_front) on the GPU with the shared expert's gate, the shared
    // expert then going there with the routed ones (moe_gpu8)
    gpu_front = gpu_front && gpu8_experts_ && gpu8_ && n > kSpreadTokens && n >= kGpuMinTokens && !exact_ &&
                !check_ && l < static_cast<int>(dec_layers_.size());
    const Gpu8::DecodeLayer *D = gpu_front ? &dec_layers_[static_cast<size_t>(l)] : nullptr;
    const float *lg = nullptr;   // the GPU's logits, E + 1 a token (the shared expert's gate last)
    const auto rs = rows(router_, static_cast<size_t>(E), n);
    const double tr = now_seconds();
    if (D) {
        if (!gpu8_->pf_router(*D, n, h_.data(), &lg, err)) return false;
    } else {
        matmul_bf16(L.router, E, H, n, hs.data(), rs.data());
    }
    // the route prior (route_prior_): generating on the CPU route, the cached experts preferred for the selection
    const uint8_t *res = !lg && n <= kSpreadTokens && (n == 1 || verify_keep_) && prior_on() ? fill_residency(l, res_l_)
                                                                                             : nullptr;
    const auto route = [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t)
            route_one(lg ? lg + static_cast<size_t>(t) * (E + 1) : rs[static_cast<size_t>(t)],
                      top_ids_.data() + static_cast<size_t>(t) * K, top_w_.data() + static_cast<size_t>(t) * K, res,
                      res ? route_prior_ : 0.0f);
    };
    if (n > kSpreadTokens) par(n, route, 16);
    else route(0, n);
    prof_.router += now_seconds() - tr;
    // generating (a token, or a pair being verified): each token's tail misses left out, as one token at a time would
    const bool one_by_one = n == 1 || verify_keep_;
    std::vector<uint8_t> skip(static_cast<size_t>(n) * K, 0);
    if (one_by_one)
        for (int t = 0; t < n; ++t)
            skip_tail(l, top_ids_.data() + static_cast<size_t>(t) * K, top_w_.data() + static_cast<size_t>(t) * K,
                      skip.data() + static_cast<size_t>(t) * K);

    // the experts picked, each with the (token, rank) pairs that picked it
    std::vector<std::array<int, 3>> picks;   // expert, token, rank
    picks.reserve(static_cast<size_t>(n) * K);
    for (int t = 0; t < n; ++t)
        for (int k = 0; k < K; ++k)
            if (!skip[static_cast<size_t>(t * K + k)]) picks.push_back({top_ids_[static_cast<size_t>(t * K + k)], t, k});
    std::sort(picks.begin(), picks.end());
    std::vector<std::pair<size_t, size_t>> runs;   // [begin, end) in picks, one per expert
    for (size_t i = 0; i < picks.size();) {
        size_t j = i;
        while (j < picks.size() && picks[j][0] == picks[i][0]) ++j;
        runs.emplace_back(i, j);
        i = j;
    }
    // as many experts pinned at once as the cache holds (next to the prefetch pins); a prompt's whole layer usually.
    // The cache shrinks as the context grows (reserve_context), so it is asked each time.
    // (with size classes: the largest class's slots, which any expert fits - ExpertStore::pin_slots)
    const size_t cache_slots = store_.pin_slots();
    const size_t wave = std::max<size_t>(1, std::min(runs.size(), cache_slots > static_cast<size_t>(2 * K + 2)
                                                                      ? cache_slots - static_cast<size_t>(K + 2)
                                                                      : static_cast<size_t>(K)));

    std::vector<int> slots;
    std::vector<DType> types;
    auto acquire_wave = [&](size_t first) {
        slots.clear();
        types.clear();
        for (size_t r = first; r < std::min(runs.size(), first + wave); ++r) {
            const int e = picks[runs[r].first][0];
            DType t;
            const ExpertRead rd = expert_read(l, e, t);
            types.push_back(t);
            slots.push_back(store_.acquire(static_cast<uint32_t>(l * E + e), rd, false));
        }
    };
    acquire_wave(0);   // the misses start reading at once

    // the guesses made for this layer one layer ago: what they caught is pinned by the demand now
    if (!keep_guesses) {
        for (int s : prefetch_[static_cast<size_t>(l)]) store_.release(s);
        prefetch_[static_cast<size_t>(l)].clear();
    }
    // a long batch: the next layer's experts, all of them, read while this layer computes (pinned until it starts;
    // the cache must hold this wave, those and one more)
    if (prefetch_on_ && n > kSpreadTokens && l + 1 < c.n_layers && cache_slots >= wave + E + K + 2) {
        static const int mode = std::getenv("GEODE_BATCH_GUESS") ? std::atoi(std::getenv("GEODE_BATCH_GUESS")) : 1;
        if (mode == 0) {   // measuring: as before (a whole layer from 128 tokens on, none below)
            if (n >= kLayerPrefetch && prefetch_[static_cast<size_t>(l + 1)].empty()) prefetch_layer(l + 1);
        } else if (prompt_n_ >= kGuessAll) {
            if (prefetch_[static_cast<size_t>(l + 1)].empty()) prefetch_layer(l + 1);
        } else {
            prefetch_guess(l + 1, hs, n);
        }
    }
    // a verified pair: each token's guess of the next layer, once each - as many as the cache has room for beside
    // the wave pinned (it leaves K + 2 slots: two tokens' guesses can be 2K, and a cache the reserve took down to its
    // floor would wait for a slot no one releases)
    if (prefetch_on_ && verify_keep_ && n > 1 && l + 1 < c.n_layers) {
        const size_t room = cache_slots > slots.size() + 2 ? cache_slots - slots.size() - 2 : 0;
        std::vector<float> g(static_cast<size_t>(E));
        std::vector<int> guess(static_cast<size_t>(K)), seen;
        // the guesses as the next layer will route (the prior's preference for the cached included)
        const uint8_t *gres = prior_on() ? fill_residency(l + 1, res_n_) : nullptr;
        for (int t = 0; t < n; ++t) {
            float *gp = g.data();
            const float *ht = hs[static_cast<size_t>(t)];
            matmul_bf16(layers_[static_cast<size_t>(l + 1)].router, E, H, 1, &ht, &gp);
            route_one(g.data(), guess.data(), nullptr, gres, gres ? route_prior_ : 0.0f);
            for (int e : guess) {
                if (std::find(seen.begin(), seen.end(), e) != seen.end() || seen.size() >= room) continue;
                seen.push_back(e);
                DType ty;
                const ExpertRead rd = expert_read(l + 1, e, ty);
                prefetch_[static_cast<size_t>(l + 1)].push_back(store_.acquire(static_cast<uint32_t>((l + 1) * E + e), rd, true));
            }
        }
    }
    // generating: the next layer's guess, its router on this layer's input (~80% right, PLAN.md P1)
    if (prefetch_on_ && n == 1 && l + 1 < c.n_layers) {
        // Measured on the Redmi (80 tokens): its top 8 is the best read-ahead. Fewer guesses wait more (6: 3.85
        // tokens/s), more read more (10: 3.69, 12: 3.22), guessing layer l + 2 as well too (3.80, 3.36), and
        // skipping unlikely guesses kills it (the experts not cached are exactly the unlikely ones: 3.47).
        float *g = router_.data() + static_cast<size_t>(batch_cap_) * E;
        const float *h0 = hs[0];
        matmul_bf16(layers_[static_cast<size_t>(l + 1)].router, E, H, 1, &h0, &g);
        std::vector<int> guess(static_cast<size_t>(K));
        const uint8_t *gres = prior_on() ? fill_residency(l + 1, res_n_) : nullptr;
        route_one(g, guess.data(), nullptr, gres, gres ? route_prior_ : 0.0f);
        for (int e : guess) {
            DType t;
            const ExpertRead rd = expert_read(l + 1, e, t);
            prefetch_[static_cast<size_t>(l + 1)].push_back(store_.acquire(static_cast<uint32_t>((l + 1) * E + e), rd, true));
        }
    }

    // the shared expert, gated by sigmoid(w . h), while the reads run (with the GPU front: on the GPU)
    const auto shs = rows(sh_, static_cast<size_t>(H), n);
    const QAct *const *qh = nullptr;
    if (!D) {
        const auto sg = rows(sh_act_, static_cast<size_t>(2 * sff), n);
        const auto su = rows(sh_act_, static_cast<size_t>(2 * sff), n, static_cast<size_t>(sff));
        const double ts = now_seconds();
        qh = quantize_h(hs, H);
        {
            const QView *ms[2] = {&L.sh_gate, &L.sh_up};
            float *const *ys[2] = {sg.data(), su.data()};
            matmul_many(ms, 2, n, hs.data(), ys, qh);
        }
        for (int t = 0; t < n; ++t)
            for (int i = 0; i < sff; ++i)
                sg[static_cast<size_t>(t)][i] = silu(sg[static_cast<size_t>(t)][i]) * su[static_cast<size_t>(t)][i];
        matmul(L.sh_down, n, sg.data(), shs.data());
        prof_.shared += now_seconds() - ts;
    }

    bool ok = true;
    const float *eout_gpu = nullptr;   // the routed outputs in the GPU's memory (moe_gpu8), else eout_
    const bool many = pool_->width() + pool_->little() > 1;
    if (n <= kSpreadTokens && !exact_ && !check_ && many && runs.size() <= wave && routed_rows_ && qh && ff % 64 == 0 &&
        H % 16 == 0) {
        // Generating: the experts' rows split over the threads (model.hpp, routed_rows_), two products a layer.
        // First every expert's gate and up rows, 64 activations a piece: the piece's 64 gate rows and 64 up rows,
        // then silu(gate) * up and those 64 values quantized (an activation's groups of 64 are independent:
        // quantize_activation's bits); then their down rows. The experts already read come first in both, so the
        // threads start on them while the rest are read. Each row's arithmetic is that of one expert computed
        // whole: the same bits.
        // the little threads join here whatever the dense products do (little_dense_), until the block ends
        struct Little {
            ThreadPool &pool;
            int back;
            ~Little() { pool.set_little(back); }
        } little_back{*pool_, pool_->little()};
        if (!dec8_) pool_->set_little(little_threads_);
        const size_t cnt = slots.size();
        const int64_t groups = ff / 64;
        const int64_t rp = pack_.experts_tile8() && H % 32 == 0 ? 32 : 16;   // down rows a piece (tile8: a tile)
        const int64_t per = H / rp;
        std::vector<char> in(cnt);
        std::vector<size_t> order;
        order.reserve(cnt);
        for (size_t i = 0; i < cnt; ++i) in[i] = store_.ready(slots[i]) ? 1 : 0;
        for (int pass = 1; pass >= 0; --pass)
            for (size_t i = 0; i < cnt; ++i)
                if (in[i] == pass) order.push_back(i);
        // each pick's [gate ff | up ff] in eact_ by its place in picks, its activation in rows_qa_
        if (rows_qa_.size() < picks.size()) rows_qa_.resize(picks.size());
        for (size_t i = 0; i < picks.size(); ++i) prepare_activation(ff, rows_qa_[i]);
        std::atomic<bool> bad{false};
        std::atomic<int64_t> wait_ns{0};
        double caller_wait = 0.0;
        const std::thread::id caller = std::this_thread::get_id();
        const auto block = [&](size_t i) -> const uint8_t * {
            if (store_.ready(slots[i])) return store_.wait(slots[i]);
            const double tw = now_seconds();
            const uint8_t *b = store_.wait(slots[i]);
            const double d = now_seconds() - tw;
            wait_ns.fetch_add(static_cast<int64_t>(d * 1e9), std::memory_order_relaxed);
            if (std::this_thread::get_id() == caller) caller_wait += d;
            return b;
        };
        const double t_up = now_seconds();
        // (waits for reads: its pieces are no measure of a thread's speed)
        pool_->parallel_for(static_cast<int64_t>(cnt) * groups, [&](int64_t a, int64_t b) {
            for (int64_t it = a; it < b; ++it) {
                const size_t i = order[static_cast<size_t>(it / groups)];
                const int64_t g0 = (it % groups) * 64;
                const uint8_t *blk = block(i);
                if (!blk) {
                    bad.store(true);
                    continue;
                }
                const auto [pb, pe] = runs[i];
                const int e = picks[pb][0];
                const QView gu = expert_view(types[i], 2 * ff, H, blk, L.expert_scales[2 * e]);
                const int m_tok = static_cast<int>(pe - pb);
                const QAct *qin[kSpreadTokens];
                float *gp[kSpreadTokens];
                for (int j = 0; j < m_tok; ++j) {
                    qin[j] = qh[picks[pb + static_cast<size_t>(j)][1]];
                    gp[j] = eact_.data() + (pb + static_cast<size_t>(j)) * 2 * static_cast<size_t>(ff);
                }
                matvec_rows_q_multi(gu, qin, m_tok, gp, g0, g0 + 64);             // gate
                matvec_rows_q_multi(gu, qin, m_tok, gp, ff + g0, ff + g0 + 64);   // up
                for (int j = 0; j < m_tok; ++j) {
                    float *g = gp[j];
                    for (int64_t v = g0; v < g0 + 64; ++v) g[v] = silu(g[v]) * g[ff + v];
                    quantize_activation_group(g, g0, rows_qa_[pb + static_cast<size_t>(j)]);
                }
            }
        }, 1, false, nullptr, false);
        const double t_down = now_seconds();
        if (!bad.load())
            pool_->parallel_for(static_cast<int64_t>(cnt) * per, [&](int64_t a, int64_t b) {
                for (int64_t it = a; it < b;) {
                    const int64_t x = it / per, end = std::min(b, (x + 1) * per);   // this expert's pieces in [a, b)
                    const size_t i = order[static_cast<size_t>(x)];
                    const uint8_t *blk = store_.wait(slots[i]);   // read: at once
                    const auto [pb, pe] = runs[i];
                    const int e = picks[pb][0];
                    const DType type = types[i];
                    const size_t gu_bytes = static_cast<size_t>(2 * ff) * row_bytes(type, H);
                    const QView dn = expert_view(type, H, ff, blk + gu_bytes, L.expert_scales[2 * e + 1]);
                    const int m_tok = static_cast<int>(pe - pb);
                    const QAct *qmid[kSpreadTokens];
                    float *yo[kSpreadTokens];
                    for (int j = 0; j < m_tok; ++j) {
                        const auto &pk = picks[pb + static_cast<size_t>(j)];
                        qmid[j] = &rows_qa_[pb + static_cast<size_t>(j)];
                        yo[j] = eout_.data() + (static_cast<size_t>(pk[1]) * K + static_cast<size_t>(pk[2])) * H;
                    }
                    matvec_rows_q_multi(dn, qmid, m_tok, yo, (it - x * per) * rp, (end - x * per) * rp);
                    it = end;
                }
            }, 1);
        for (size_t i = 0; i < cnt; ++i) store_.release(slots[i]);
        const double t_end = now_seconds();
        prof_.rows_up += t_down - t_up;
        prof_.rows_down += t_end - t_down;
        prof_.rows_wait += static_cast<double>(wait_ns.load()) * 1e-9;
        prof_.expert_wait += caller_wait;
        if (bad.load()) {
            err = "cannot read an expert of layer " + std::to_string(l);
            ok = false;
        }
    } else if (n <= kSpreadTokens && !exact_ && !check_ && pool_->size() > 1 && runs.size() <= wave) {
        // Generating: the experts spread over the threads, each computed whole by one of them. Splitting every expert
        // across the threads cost two joins per expert (640 a token), and a thread whose expert is still loading
        // now leaves the ready ones to the others. Per expert the arithmetic is the same: the same bits.
        const size_t cnt = slots.size();
        std::vector<char> taken(cnt, 0);
        std::mutex mu;
        std::string failed;
        double caller_wait = 0.0, all_wait = 0.0, all_compute = 0.0;
        const std::thread::id caller = std::this_thread::get_id();
        const double t_spread = now_seconds();
        // on three quarters of the threads (model.hpp, spread_threads)
        par(spread_threads(), [&](int64_t, int64_t) {
            thread_local std::vector<float> ga;
            thread_local std::vector<QAct> qa;
            for (;;) {
                size_t pick = cnt;
                {
                    std::lock_guard<std::mutex> lock(mu);
                    for (size_t i = 0; i < cnt && pick == cnt; ++i)
                        if (!taken[i] && store_.ready(slots[i])) pick = i;
                    for (size_t i = 0; i < cnt && pick == cnt; ++i)
                        if (!taken[i]) pick = i;
                    if (pick == cnt) return;
                    taken[pick] = 1;
                }
                const auto [b, eend] = runs[pick];
                const int e = picks[b][0];
                const double tw = now_seconds();
                const uint8_t *blk = store_.wait(slots[pick]);
                const double tc0 = now_seconds();
                if (std::this_thread::get_id() == caller) caller_wait += tc0 - tw;
                if (!blk) {
                    std::lock_guard<std::mutex> lock(mu);
                    failed = "cannot read expert " + std::to_string(e) + " of layer " + std::to_string(l);
                    store_.release(slots[pick]);
                    continue;
                }
                const DType type = types[pick];
                const size_t gu_bytes = static_cast<size_t>(2 * ff) * row_bytes(type, H);
                const QView gu = expert_view(type, 2 * ff, H, blk, L.expert_scales[2 * e]);
                const QView dn = expert_view(type, H, ff, blk + gu_bytes, L.expert_scales[2 * e + 1]);
                const int m_tok = static_cast<int>(eend - b);
                ga.resize(static_cast<size_t>(m_tok) * 2 * ff);
                if (qa.size() < static_cast<size_t>(m_tok)) qa.resize(static_cast<size_t>(m_tok));
                const QAct *qin[kSpreadTokens];
                const QAct *qmid[kSpreadTokens];
                float *gp[kSpreadTokens], *yo[kSpreadTokens];
                for (int j = 0; j < m_tok; ++j) {
                    const auto &pk = picks[b + static_cast<size_t>(j)];
                    qin[j] = qh[pk[1]];
                    gp[j] = ga.data() + static_cast<size_t>(j) * 2 * ff;
                    yo[j] = eout_.data() + (static_cast<size_t>(pk[1]) * K + static_cast<size_t>(pk[2])) * H;
                }
                matvec_rows_q_multi(gu, qin, m_tok, gp, 0, gu.rows);   // [gate ff | up ff]
                for (int j = 0; j < m_tok; ++j) {
                    float *g = gp[j];
                    for (int i = 0; i < ff; ++i) g[i] = silu(g[i]) * g[ff + i];
                    quantize_activation(g, ff, qa[static_cast<size_t>(j)]);
                    qmid[j] = &qa[static_cast<size_t>(j)];
                }
                matvec_rows_q_multi(dn, qmid, m_tok, yo, 0, dn.rows);
                store_.release(slots[pick]);
                const double tc1 = now_seconds();
                std::lock_guard<std::mutex> lock(mu);
                all_wait += tc0 - tw;
                all_compute += tc1 - tc0;
            }
        });
        prof_.expert_wait += caller_wait;
        prof_.spread_wall += now_seconds() - t_spread;
        prof_.spread_wait += all_wait;
        prof_.spread_compute += all_compute;
        if (!failed.empty()) {
            err = failed;
            ok = false;
        }
    } else if (gpu8_experts_ && gpu8_ && n >= kGpuMinTokens && !exact_ && !check_) {
        // the Adreno 7xx+: the routed experts, wave by wave, each wave in one go once its blocks are in (those in the
        // NPU's slabs on HMX meanwhile)
        npu_x_ = npu_y_ = false;
        if (npu_.moe_on) npu_pick_.assign(static_cast<size_t>(n) * K, 0);
        for (size_t first = 0; first < runs.size() && ok; first += wave) {
            if (first > 0) acquire_wave(first);
            ok = moe_gpu8(l, L, n, picks, runs, slots, types, first, err, first == 0 ? D : nullptr);
            for (int sl : slots) store_.release(sl);
        }
        if (ok) eout_gpu = gpu8_->out();
    } else
    // the routed experts, wave by wave; in a wave each as soon as its block is in, the ready ones first
    for (size_t first = 0; first < runs.size(); first += wave) {
        if (first > 0) acquire_wave(first);
        const size_t cnt = slots.size();
        std::vector<bool> done(cnt, false);
        for (size_t m = 0; m < cnt; ++m) {
            size_t pick = cnt;
            for (size_t i = 0; i < cnt && pick == cnt; ++i)
                if (!done[i] && store_.ready(slots[i])) pick = i;
            if (pick == cnt)
                for (size_t i = 0; i < cnt && pick == cnt; ++i)
                    if (!done[i]) pick = i;
            done[pick] = true;
            const auto [b, eend] = runs[first + pick];
            const int e = picks[b][0];
            const double tw = now_seconds();
            const uint8_t *blk = store_.wait(slots[pick]);
            prof_.expert_wait += now_seconds() - tw;
            if (!blk) {
                err = "cannot read expert " + std::to_string(e) + " of layer " + std::to_string(l);
                ok = false;
                store_.release(slots[pick]);
                continue;
            }
            const DType type = types[pick];
            const size_t gu_bytes = static_cast<size_t>(2 * ff) * row_bytes(type, H);
            const QView gu = expert_view(type, 2 * ff, H, blk, L.expert_scales[2 * e]);
            const QView dn = expert_view(type, H, ff, blk + gu_bytes, L.expert_scales[2 * e + 1]);
            const int m_tok = static_cast<int>(eend - b);
            std::vector<const float *> xin(static_cast<size_t>(m_tok));
            std::vector<const QAct *> qin(static_cast<size_t>(m_tok));
            std::vector<float *> ga(static_cast<size_t>(m_tok)), yo(static_cast<size_t>(m_tok));
            for (int j = 0; j < m_tok; ++j) {
                const auto &pk = picks[b + static_cast<size_t>(j)];
                xin[static_cast<size_t>(j)] = hs[static_cast<size_t>(pk[1])];
                if (qh) qin[static_cast<size_t>(j)] = qh[pk[1]];
                ga[static_cast<size_t>(j)] = eact_.data() + static_cast<size_t>(j) * 2 * ff;
                yo[static_cast<size_t>(j)] = eout_.data() + (static_cast<size_t>(pk[1]) * K + static_cast<size_t>(pk[2])) * H;
            }
            matmul(gu, m_tok, xin.data(), ga.data(), qh ? qin.data() : nullptr);   // [gate ff | up ff]
            for (int j = 0; j < m_tok; ++j) {
                float *g = ga[static_cast<size_t>(j)];
                for (int i = 0; i < ff; ++i) g[i] = silu(g[i]) * g[ff + i];
            }
            matmul(dn, m_tok, ga.data(), yo.data());
            store_.release(slots[pick]);
        }
    }

    // per token: the gated shared expert, then the routed outputs in routing order (never in arrival order: the
    // result must not depend on which read finished first); a prompt's tokens on every thread (each token's
    // arithmetic the same)
    if (!ok) return false;
    if (trace) {
        trace_gate_.assign(static_cast<size_t>(n), 0.0f);
        trace_routed_ = eout_gpu || lg ? nullptr : eout_.data();
        trace_shared_ = lg ? nullptr : sh_.data();
    }
    const auto combine = [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const float *h = hs[static_cast<size_t>(t)];
            const float *eo = eout_gpu ? eout_gpu : eout_.data();
            float gs = 0.0f;
            const float *shv;
            if (lg) {   // the GPU's gate and shared expert (its rows after the routed ones)
                gs = lg[static_cast<size_t>(t) * (E + 1) + static_cast<size_t>(E)];
                shv = eo + (static_cast<size_t>(n) * K + static_cast<size_t>(t)) * H;
            } else {
                for (int i = 0; i < H; ++i) gs += bf16_to_f32(L.shared_gate[i]) * h[i];
                shv = shs[static_cast<size_t>(t)];
            }
            gs = sigmoid(gs);
            if (trace) trace_gate_[static_cast<size_t>(t)] = gs;
            float *out = moe_.data() + static_cast<size_t>(t) * H;
            for (int i = 0; i < H; ++i) out[i] = gs * shv[i];
            for (int k = 0; k < K; ++k) {
                const float w = top_w_[static_cast<size_t>(t * K + k)];
                if (w == 0.0f) continue;   // left out (skip_tail): its row was never computed
                if (npu_y_ && npu_pick_[static_cast<size_t>(t * K + k)]) continue;   // the NPU's: in npu_sum_
                const float *y = eo + (static_cast<size_t>(t) * K + static_cast<size_t>(k)) * H;
                for (int i = 0; i < H; ++i) out[i] += w * y[i];
            }
            if (npu_y_) {   // the NPU's experts, each weighted already, summed
                const float *ny = npu_sum_.data() + static_cast<size_t>(t) * H;
                for (int i = 0; i < H; ++i) out[i] += ny[i];
            }
        }
    };
    if (n > kSpreadTokens) par(n, combine, 8);
    else combine(0, n);
    return true;
}

int Model::moe_slab_of(const uint8_t *p) const {
    for (size_t i = 0; i < npu_.moe_slabs.size(); ++i)
        if (p >= npu_.moe_slabs[i].host && p < npu_.moe_slabs[i].host + npu_.moe_slabs[i].bytes) return static_cast<int>(i);
    return -1;
}

bool Model::moe_gpu8(int l, const Layer &L, int n, const std::vector<std::array<int, 3>> &picks,
                     const std::vector<std::pair<size_t, size_t>> &runs, const std::vector<int> &slots,
                     const std::vector<DType> &types, size_t first, std::string &err,
                     const Gpu8::DecodeLayer *shared) {
    const auto &c = cfg_;
    const int H = c.hidden, K = k_, ff = c.moe_ff;
    std::vector<Gpu8::Expert> ex(slots.size());
    const double tw = now_seconds();
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto [b, end] = runs[first + i];
        const int e = picks[b][0];
        const uint8_t *blk = store_.wait(slots[i]);
        if (!blk) {
            err = "cannot read expert " + std::to_string(e) + " of layer " + std::to_string(l);
            return false;
        }
        const size_t gu_bytes = static_cast<size_t>(2 * ff) * row_bytes(types[i], H);
        ex[i].gu = expert_view(types[i], 2 * ff, H, blk, L.expert_scales[2 * e]);
        ex[i].dn = expert_view(types[i], H, ff, blk + gu_bytes, L.expert_scales[2 * e + 1]);
        for (size_t j = b; j < end; ++j) ex[i].picks.emplace_back(picks[j][1], picks[j][1] * K + picks[j][2]);
    }
    prof_.expert_wait += now_seconds() - tw;
    const auto hs = rows(h_, static_cast<size_t>(H), n);
    // The experts whose blocks lie in the NPU's slabs: on HMX (geode_npu_moe) on a thread of their own while the GPU
    // computes the rest - each pick's output times its routing weight, summed per token into the NPU's y; moe()'s
    // combine adds it (npu_sum_) and leaves those picks' GPU rows alone (npu_pick_). Below kNpuMoeMin tokens an
    // expert's weights cost the NPU more than its rows cost the GPU. GEODE_NPU_MOE_MIN: that many instead.
    static const int npu_min = std::getenv("GEODE_NPU_MOE_MIN") ? std::atoi(std::getenv("GEODE_NPU_MOE_MIN")) : 512;
    std::vector<Gpu8::Expert> ex_npu;
    std::vector<int> ex_slab;
    if (npu_.moe_on && n >= npu_min && H % 64 == 0 && ff % 64 == 0) {
        std::vector<Gpu8::Expert> rest;
        for (size_t i = 0; i < ex.size(); ++i) {
            const int s = types[i] == DType::Q4 ? moe_slab_of(ex[i].gu.data) : -1;
            if (s >= 0 && moe_slab_of(ex[i].dn.data) == s) {
                ex_npu.push_back(std::move(ex[i]));
                ex_slab.push_back(s);
            } else {
                rest.push_back(std::move(ex[i]));
            }
        }
        ex.swap(rest);
        // the NPU's cost is an expert's weights (the same for 10 picks or 300), the GPU's its picks: the NPU takes the
        // busiest share of them (GEODE_NPU_MOE_SHARE, of the experts it could take), the GPU the rest. RedMagic, a
        // 3000-token prompt of the 35B: 1.0 215 tok/s, 0.7 223, 0.5 243, 0.4 239, 0.3 239, 0.2 220 (the GPU alone 184)
        static const float share = std::getenv("GEODE_NPU_MOE_SHARE")
                                       ? std::clamp(static_cast<float>(std::atof(std::getenv("GEODE_NPU_MOE_SHARE"))), 0.0f, 1.0f)
                                       : 0.45f;
        const size_t keep = static_cast<size_t>(share * static_cast<float>(ex_npu.size()) + 0.5f);
        if (keep < ex_npu.size()) {
            std::vector<size_t> order(ex_npu.size());
            std::iota(order.begin(), order.end(), size_t{0});
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) { return ex_npu[a].picks.size() > ex_npu[b].picks.size(); });
            std::vector<Gpu8::Expert> top;
            std::vector<int> top_slab;
            for (size_t r = 0; r < order.size(); ++r) {
                if (r < keep) {
                    top.push_back(std::move(ex_npu[order[r]]));
                    top_slab.push_back(ex_slab[order[r]]);
                } else {
                    ex.push_back(std::move(ex_npu[order[r]]));
                }
            }
            ex_npu.swap(top);
            ex_slab.swap(top_slab);
        }
    }
    int npu_rc = 0;
    double npu_dsp = 0.0, npu_t0 = 0.0;
    std::vector<uint8_t> desc;
    const size_t y_at = (static_cast<size_t>(kMaxBatch) * static_cast<size_t>(H) * 4 + 127) / 128 * 128;
    if (!ex_npu.empty()) {
        size_t n_picks = 0;
        for (const auto &e : ex_npu) n_picks += e.picks.size();
        const size_t n_fds = npu_.moe_slabs.size(), fds_bytes = (n_fds + 1) / 2 * 2 * 4;
        desc.assign(sizeof(moe_hdr_t) + fds_bytes + ex_npu.size() * sizeof(moe_exp_t) + n_picks * sizeof(moe_pick_t), 0);
        auto *hd = reinterpret_cast<moe_hdr_t *>(desc.data());
        auto *fds = reinterpret_cast<int32_t *>(hd + 1);
        auto *me = reinterpret_cast<moe_exp_t *>(desc.data() + sizeof(moe_hdr_t) + fds_bytes);
        auto *mp = reinterpret_cast<moe_pick_t *>(me + ex_npu.size());
        *hd = moe_hdr_t{MOE_VERSION, static_cast<uint32_t>(ex_npu.size()), static_cast<uint32_t>(n_picks),
                        static_cast<uint32_t>(H), static_cast<uint32_t>(ff), static_cast<uint32_t>(n), 0,
                        static_cast<uint32_t>(y_at), MOE_ZERO_Y, 0, static_cast<uint32_t>(n_fds), 0};
        for (size_t i = 0; i < n_fds; ++i) fds[i] = npu_.moe_slabs[i].fd;
        uint32_t at = 0;
        for (size_t i = 0; i < ex_npu.size(); ++i) {
            const auto &sl = npu_.moe_slabs[static_cast<size_t>(ex_slab[i])];
            me[i] = moe_exp_t{static_cast<uint64_t>(ex_npu[i].gu.data - sl.host),
                              static_cast<uint64_t>(ex_npu[i].dn.data - sl.host), ex_npu[i].gu.global,
                              ex_npu[i].dn.global, at, static_cast<uint32_t>(ex_npu[i].picks.size()),
                              static_cast<uint32_t>(ex_slab[i]), 0};
            for (const auto &[tok, row] : ex_npu[i].picks)
                mp[at++] = moe_pick_t{static_cast<uint32_t>(tok), top_w_[static_cast<size_t>(row)]};
        }
        if (!npu_x_) {   // this moe()'s rows, once
            std::memcpy(npu_.moe_io, h_.data(), static_cast<size_t>(n) * H * sizeof(float));
            npu_x_ = true;
        }
    }
    // GEODE_PF_CPU_SHARE=f (off by default): the CPU takes about f of the wave's picks while the GPU does the rest -
    // the experts with the fewest tokens (they fill the GPU's tiles worst; the busiest ones instead measured worse),
    // computed by the CPU's int8 kernels as a CPU-only prompt computes them, into eout_, then copied to the GPU's
    // output rows (the CPU sat idle while the GPU ran). RedMagic, 4500 tokens: 0.25 = +3-7% (119-126 -> 128-133
    // tok/s), 0.35 slower (the CPU then takes longer than the GPU); the logits move within the int8 paths' own
    // difference (KL against the CPU-only prompt 1.34e-3, the GPU alone 1.17e-3), hence not on by default.
    static const float cpu_share = std::getenv("GEODE_PF_CPU_SHARE")
                                       ? std::clamp(static_cast<float>(std::atof(std::getenv("GEODE_PF_CPU_SHARE"))), 0.0f, 0.9f)
                                       : 0.0f;
    std::vector<size_t> on_cpu;
    if (cpu_share > 0.0f && ex_npu.empty() && have_dotprod_kernels() && ff % 64 == 0 && H % 128 == 0) {
        size_t total = 0;
        for (const auto &e : ex) total += e.picks.size();
        std::vector<size_t> order(ex.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ex[a].picks.size() < ex[b].picks.size(); });
        size_t taken = 0;
        for (size_t i : order) {
            if (static_cast<float>(taken + ex[i].picks.size()) > cpu_share * static_cast<float>(total)) break;
            taken += ex[i].picks.size();
            on_cpu.push_back(i);
        }
    }
    std::vector<Gpu8::Expert> ex_gpu;
    if (on_cpu.empty()) {
        ex_gpu.swap(ex);
    } else {
        std::vector<char> cpu(ex.size(), 0);
        for (size_t i : on_cpu) cpu[i] = 1;
        for (size_t i = 0; i < ex.size(); ++i)
            if (!cpu[i]) ex_gpu.push_back(std::move(ex[i]));
    }
    const double t0 = now_seconds();
    bool gpu_ok = true;
    std::string gpu_err;
    std::thread gpu;   // the GPU's share on a thread of its own (it mostly waits for the GPU) while the CPU computes
    if (on_cpu.empty() && ex_npu.empty()) gpu_ok = gpu8_->experts(ex_gpu, n, hs.data(), gpu_err, shared);
    else gpu = std::thread([&] { gpu_ok = gpu8_->experts(ex_gpu, n, hs.data(), gpu_err, shared); });
    if (!ex_npu.empty()) {
        // the NPU's share on this thread: the session's HMX belongs to the thread that took it (this one, for the
        // prompt's products) - from another thread the DSP ran HMX unlocked and its process went down (0x8000040d)
        npu_t0 = now_seconds();
        npu_rc = npu_.moe(npu_.g, desc.data(), static_cast<int>(desc.size()), npu_.moe_io_fd, &npu_dsp);
        if (std::getenv("GEODE_NPU_MOE_DEBUG"))
            std::fprintf(stderr, "npu moe layer %d: %zu experts, %d tokens: 0x%x, DSP %.2f ms\n", l, ex_npu.size(), n,
                         static_cast<unsigned>(npu_rc), npu_dsp * 1e3);
        prof_.npu_moe += now_seconds() - npu_t0;
        prof_.npu_moe_dsp += npu_dsp;
        gpu.join();
    }
    if (!on_cpu.empty()) {
        const QAct *const *qh = quantize_h(hs, H);
        // three passes over every thread (one expert a thread left most of them idle): every CPU expert's gate and
        // up rows in pieces, then each pick's SwiGLU and its quantization, then the down rows in pieces - a row's
        // arithmetic is the whole product's (the same bits)
        std::vector<size_t> base(on_cpu.size() + 1, 0);
        for (size_t i = 0; i < on_cpu.size(); ++i) base[i + 1] = base[i] + ex[on_cpu[i]].picks.size();
        const size_t tot = base.back();
        pf_ga_.resize(tot * 2 * static_cast<size_t>(ff));
        if (pf_mid_.size() < tot) pf_mid_.resize(tot);
        std::vector<const QAct *> qin(tot), qmid(tot);
        std::vector<float *> gp(tot), yo(tot);
        for (size_t i = 0; i < on_cpu.size(); ++i) {
            const Gpu8::Expert &e = ex[on_cpu[i]];
            for (size_t j = 0; j < e.picks.size(); ++j) {
                const size_t k = base[i] + j;
                qin[k] = qh[e.picks[j].first];
                gp[k] = pf_ga_.data() + k * 2 * static_cast<size_t>(ff);
                yo[k] = eout_.data() + static_cast<size_t>(e.picks[j].second) * H;
                qmid[k] = &pf_mid_[k];
            }
        }
        constexpr int64_t kGuRows = 64, kDnRows = 128;   // whole tiles of tile8
        const int64_t gu_pieces = 2 * ff / kGuRows, dn_pieces = H / kDnRows;
        const auto ne = static_cast<int64_t>(on_cpu.size());
        par(ne * gu_pieces, [&](int64_t a, int64_t b) {
            for (int64_t w = a; w < b; ++w) {
                const size_t i = static_cast<size_t>(w / gu_pieces);
                const int64_t r = (w % gu_pieces) * kGuRows;
                matvec_rows_q_multi(ex[on_cpu[i]].gu, qin.data() + base[i], static_cast<int>(base[i + 1] - base[i]),
                                    gp.data() + base[i], r, r + kGuRows);   // [gate ff | up ff]
            }
        });
        par(static_cast<int64_t>(tot), [&](int64_t a, int64_t b) {
            for (int64_t k = a; k < b; ++k) {
                float *g = gp[static_cast<size_t>(k)];
                for (int i = 0; i < ff; ++i) g[i] = silu(g[i]) * g[ff + i];
                quantize_activation(g, ff, pf_mid_[static_cast<size_t>(k)]);
            }
        }, 4);
        par(ne * dn_pieces, [&](int64_t a, int64_t b) {
            for (int64_t w = a; w < b; ++w) {
                const size_t i = static_cast<size_t>(w / dn_pieces);
                const int64_t r = (w % dn_pieces) * kDnRows;
                matvec_rows_q_multi(ex[on_cpu[i]].dn, qmid.data() + base[i], static_cast<int>(base[i + 1] - base[i]),
                                    yo.data() + base[i], r, r + kDnRows);
            }
        });
        prof_.pf_cpu_experts += now_seconds() - t0;
        gpu.join();
    }
    if (!gpu_ok) {
        err = "GPU experts: " + gpu_err;
        return false;
    }
    prof_.gpu_moe_experts += ex_gpu.size() + on_cpu.size();
    if (!ex_npu.empty()) {
        if (npu_rc == 0) {
            prof_.npu_moe_experts += ex_npu.size();
            const float *y = reinterpret_cast<const float *>(npu_.moe_io + y_at);
            if (!npu_y_) {
                npu_sum_.assign(y, y + static_cast<size_t>(n) * H);
                npu_y_ = true;
            } else {
                for (size_t i = 0; i < static_cast<size_t>(n) * H; ++i) npu_sum_[i] += y[i];
            }
            for (const auto &e : ex_npu)
                for (const auto &pr : e.picks) npu_pick_[static_cast<size_t>(pr.second)] = 1;
        } else {
            // the NPU failed: these experts on the GPU after all, the NPU's share off from here on
            npu_.moe_on = false;
            char b[64];
            std::snprintf(b, sizeof b, " (experts: failed 0x%x)", static_cast<unsigned>(npu_rc));
            npu_.status += b;
            if (!gpu8_->experts(ex_npu, n, hs.data(), gpu_err, nullptr)) {
                err = "GPU experts: " + gpu_err;
                return false;
            }
        }
    }
    if (!on_cpu.empty()) {   // the CPU's rows next to the GPU's (the GPU is done with the buffer)
        float *out = gpu8_->out_rows();
        for (size_t i : on_cpu)
            for (const auto &[tok, row] : ex[i].picks)
                std::memcpy(out + static_cast<size_t>(row) * H, eout_.data() + static_cast<size_t>(row) * H,
                            static_cast<size_t>(H) * sizeof(float));
    }
    prof_.gpu8 += now_seconds() - t0;
    return true;
}

void Model::prefetch_guess(int l, const std::vector<float *> &hs, int n) {
    const int E = cfg_.n_experts, H = cfg_.hidden, G = std::min({kBatchGuesses, k_, E});
    guess_.resize(static_cast<size_t>(n) * E);
    const auto gs = rows(guess_, static_cast<size_t>(E), n);
    const double tg = now_seconds();
    matmul_bf16(layers_[static_cast<size_t>(l)].router, E, H, n, hs.data(), gs.data());
    prof_.router += now_seconds() - tg;
    if (guessed_layer_ != l) {
        guessed_.assign(static_cast<size_t>(E), 0);
        guessed_layer_ = l;
    }
    std::vector<int> idx(static_cast<size_t>(E));
    std::vector<uint8_t> want(static_cast<size_t>(E), 0);
    for (int t = 0; t < n; ++t) {
        const float *g = gs[static_cast<size_t>(t)];
        for (int e = 0; e < E; ++e) idx[static_cast<size_t>(e)] = e;
        std::partial_sort(idx.begin(), idx.begin() + G, idx.end(), [g](int a, int b) { return g[a] > g[b]; });
        for (int i = 0; i < G; ++i) want[static_cast<size_t>(idx[static_cast<size_t>(i)])] = 1;
    }
    auto &pins = prefetch_[static_cast<size_t>(l)];
    for (int e = 0; e < E; ++e) {
        if (!want[static_cast<size_t>(e)] || guessed_[static_cast<size_t>(e)]) continue;
        guessed_[static_cast<size_t>(e)] = 1;
        DType t;
        const ExpertRead rd = expert_read(l, e, t);
        pins.push_back(store_.acquire(static_cast<uint32_t>(l * cfg_.n_experts + e), rd, true));
    }
}

int Model::mixer_gpu(int l, const Layer &L, int n, std::string &err) {
    if (pf_gpu_ < 0) {   // decided once: generating runs on the GPU, its states and KV cache in shared memory
        const char *env = std::getenv("GEODE_PF_GPU");
        std::string why;
        pf_gpu_ = gpu8_ && dec8_ok_ && !exact_ && !check_ && kv_i8_ && !(env && std::atoi(env) == 0) &&
                          gpu8_->pf_ready(why)
                      ? 1
                      : 0;
        if (!why.empty()) gpu_status_ += "; prompts' mixers on the CPU (" + why + ")";
    }
    // (the MTP layer's attention too, when it is on the GPU: forward_mtp)
    // (GEODE_PF_MTP=0: the MTP layer's on the CPU, as before - measuring)
    static const bool pf_mtp = !(std::getenv("GEODE_PF_MTP") && std::atoi(std::getenv("GEODE_PF_MTP")) == 0);
    const bool mtp = &L == &mtp_layer_;
    if (mtp && !pf_mtp) return 0;
    if (pf_gpu_ != 1 || n < kGpuMinTokens || trace || (mtp ? !mtp8_ : l >= static_cast<int>(dec_layers_.size())))
        return 0;
    const Gpu8::DecodeLayer &D = mtp ? dec_mtp_ : dec_layers_[static_cast<size_t>(l)];
    // GEODE_PF_CHECK=1 (diagnosis): the layer on the GPU, then its states put back and the layer on the CPU (which
    // stands); how far apart the two outputs are
    static const bool pf_check = std::getenv("GEODE_PF_CHECK") != nullptr;
    if (pf_check) {
        const auto &c = cfg_;
        const int H = c.hidden;
        const size_t conv_dim = static_cast<size_t>(2 * c.gdn_k_heads * c.gdn_k_dim + c.gdn_v_heads * c.gdn_v_dim);
        const size_t cn = conv_dim * static_cast<size_t>(c.conv_kernel - 1);
        const size_t gn = static_cast<size_t>(c.gdn_v_heads) * c.gdn_k_dim * c.gdn_v_dim;
        std::vector<float> cs, gs;
        if (!L.attn) {
            cs.assign(conv_st_ + L.index * cn, conv_st_ + (L.index + 1) * cn);
            gs.assign(gdn_st_ + L.index * gn, gdn_st_ + (L.index + 1) * gn);
        }
        std::vector<float> mg(static_cast<size_t>(n) * H);
        std::string e2;
        bool ok2;
        if (L.attn) {
            pos3_.resize(static_cast<size_t>(n) * 3);
            for (int t = 0; t < n; ++t) rope_pos(apos_ + t, pos3_.data() + static_cast<size_t>(t) * 3);
            ok2 = gpu8_->pf_attn(D, n, apos_, pos3_.data(), kv_ptrs_, h_.data(), mg.data(), e2);
        } else {
            ok2 = gpu8_->pf_gdn(D, n, h_.data(), mg.data(), e2);
        }
        std::vector<float> gs_gpu;
        if (!L.attn) {
            gs_gpu.assign(gdn_st_ + L.index * gn, gdn_st_ + (L.index + 1) * gn);
            std::copy(cs.begin(), cs.end(), conv_st_ + L.index * cn);
            std::copy(gs.begin(), gs.end(), gdn_st_ + L.index * gn);
        }
        if (L.attn) attention(L, n);
        else gdn(L, n);
        double d = 0, r = 0, ds = 0, rs = 0;
        int nan = 0;
        for (size_t i = 0; i < mg.size(); ++i) {
            if (!std::isfinite(mg[i])) ++nan;
            d += (static_cast<double>(mg[i]) - mix_[i]) * (static_cast<double>(mg[i]) - mix_[i]);
            r += static_cast<double>(mix_[i]) * mix_[i];
        }
        for (size_t i = 0; i < gs_gpu.size(); ++i) {
            const double cpu = gdn_st_[L.index * gn + i];
            ds += (gs_gpu[i] - cpu) * (gs_gpu[i] - cpu);
            rs += cpu * cpu;
        }
        std::fprintf(stderr, "pf check layer %d (%s) n %d pos %d: %s output rel %.2e (%d not finite)%s state rel %.2e\n",
                     l, L.attn ? "attn" : "gdn", n, apos_, ok2 ? "" : e2.c_str(), std::sqrt(d / std::max(r, 1e-30)), nan,
                     "", std::sqrt(ds / std::max(rs, 1e-30)));
        return 1;
    }
    bool ok;
    if (L.attn) {
        pos3_.resize(static_cast<size_t>(n) * 3);
        for (int t = 0; t < n; ++t) rope_pos(apos_ + t, pos3_.data() + static_cast<size_t>(t) * 3);
        ok = gpu8_->pf_attn(D, n, apos_, pos3_.data(), kv_ptrs_, h_.data(), mix_.data(), err);
    } else {
        ok = gpu8_->pf_gdn(D, n, h_.data(), mix_.data(), err);
    }
    if (!ok) {
        pf_gpu_ = 0;   // the next prompt's on the CPU
        err = "a prompt's layer on the GPU: " + err;
        return -1;
    }
    return 1;
}

void Model::prefetch_layer(int l) {
    const int E = cfg_.n_experts;
    auto &pins = prefetch_[static_cast<size_t>(l)];
    for (int e = 0; e < E; ++e) {
        DType t;
        const ExpertRead rd = expert_read(l, e, t);
        pins.push_back(store_.acquire(static_cast<uint32_t>(l * E + e), rd, true));
    }
}

bool Model::forward_mtp(const int *next_tokens, const float *const *hidden, int n, int pos, float *logits,
                        std::string &err, const QView *head) {
    if (!mtp_) {
        err = "the pack has no MTP layer";
        return false;
    }
    const auto &c = cfg_;
    const int H = c.hidden;
    if (n <= 0 || n > kMaxBatch || pos < 0) {
        err = "bad MTP batch";
        return false;
    }
    struct Clock {
        double &to, t0 = now_seconds();
        ~Clock() { to += now_seconds() - t0; }
    } clock{prof_.mtp_pass};
    if (!reserve_context(pos + n, err)) return false;
    // a prompt's pass (no logits): only the layer's K / V matter, on the DSP when it reads the prompts
    if (!logits && npu_.dec && npu_.mtp_kv) return mtp_kv_npu(next_tokens, hidden, n, pos, err);
    if (n > batch_cap_ && !size_batch(n, err)) return false;
    wide_ = false;
    pin_current_thread(big_);
    threads_for(n);
    // the layer's input: fc([norm(embedding of the next token) | norm(hidden state)]), rows of its own (proj_'s are
    // wide enough for Qwen3.6, not for every shape)
    if (mtp_in_.size() < static_cast<size_t>(n) * 2 * H) mtp_in_.resize(static_cast<size_t>(n) * 2 * H);
    const auto in = rows(mtp_in_, static_cast<size_t>(2 * H), n);
    for (int t = 0; t < n; ++t) {
        float *row = in[static_cast<size_t>(t)];
        // the token after an image pad's position is its features (as Strata: the drafts after an image hold)
        const float *im = vision_.available() && next_tokens[t] == vision_.shape().pad_token ? image_row(pos + t + 1)
                                                                                           : nullptr;
        if (im) {
            std::copy_n(im, H, row + H);
        } else if (!pack_.embedding(next_tokens[t], row + H)) {
            err = "cannot read the embedding of token " + std::to_string(next_tokens[t]);
            return false;
        }
        rmsnorm(row + H, mtp_pre_e_, row, H, c.eps, true);
        rmsnorm(hidden[t], mtp_pre_h_, row + H, H, c.eps, true);
    }
    const auto xs = rows(x_, static_cast<size_t>(H), n);
    matmul(mtp_fc_, n, in.data(), xs.data());

    // the decoder layer: attention over the MTP layer's own positions, then its MoE (expert layer n_layers)
    const Layer &L = mtp_layer_;
    for (int t = 0; t < n; ++t)
        rmsnorm(x_.data() + static_cast<size_t>(t) * H, L.in_norm, h_.data() + static_cast<size_t>(t) * H, H, c.eps, true);
    apos_ = pos;
    const int on_gpu = mixer_gpu(c.n_layers, L, n, err);   // a long prompt's: on the GPU
    if (on_gpu < 0) return false;
    if (on_gpu == 0) attention(L, n);
    for (size_t i = 0; i < static_cast<size_t>(n) * H; ++i) x_[i] += mix_[i];
    for (int t = 0; t < n; ++t)
        rmsnorm(x_.data() + static_cast<size_t>(t) * H, L.post_norm, h_.data() + static_cast<size_t>(t) * H, H, c.eps, true);
    {
        // many positions at once: a scan through the layer's experts, as forward_batch's (in chunks that would evict
        // one another's otherwise)
        const bool scan = n >= kLayerPrefetch;
        if (scan) store_.set_scan(c.n_layers, static_cast<uint32_t>(c.n_experts));
        const bool ok = moe(c.n_layers, L, n, err);
        if (scan) store_.set_scan(-1, 1);
        if (!ok) return false;
    }
    for (size_t i = 0; i < static_cast<size_t>(n) * H; ++i) x_[i] += moe_[i];

    if (logits) {
        float *h = h_.data();
        rmsnorm(x_.data() + static_cast<size_t>(n - 1) * H, mtp_norm_, h, H, c.eps, true);
        const float *hp = h;
        matmul(head ? *head : lm_head_, 1, &hp, &logits);
    }
    return true;
}

std::string Model::profile() const {
    const auto &p = prof_;
    char buf[4096];
    std::snprintf(buf, sizeof buf,
                  "%d tokens through: Gated DeltaNet %.0f ms (recurrence %.0f: convolution %.0f, state %.0f, gated norm "
                  "%.0f), attention %.0f ms (scores and values %.0f), MoE %.0f ms (router %.0f, shared expert %.0f, "
                  "waiting for expert reads %.0f, the rest = routed experts' arithmetic), lm_head %.0f ms",
                  p.tokens, p.gdn * 1e3, p.gdn_core * 1e3, p.gdn_conv * 1e3, p.gdn_strips * 1e3, p.gdn_norm * 1e3,
                  p.attention * 1e3, p.attn_core * 1e3, p.moe * 1e3,
                  p.router * 1e3, p.shared * 1e3, p.expert_wait * 1e3, p.head * 1e3);
    // the passes' own wall time, and what of it is none of those parts (norms, embeddings, the KV cache's growth...)
    if (p.pass > 0)
        std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf),
                      "; the passes %.0f ms (outside those parts %.0f), the MTP layer's %.0f ms", p.pass * 1e3,
                      (p.pass - p.gdn - p.attention - p.moe - p.head) * 1e3, p.mtp_pass * 1e3);
    if (gpu_) {
        std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), "; GPU busy %.0f ms",
                      (gpu_->busy_seconds() - p.gpu0) * 1e3);
        if (p.g_dec > 0)
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf),
                          " [generating: %llu products shared with it, its rows done after the CPU's in %llu; "
                          "launching %.0f ms, waiting for it %.0f ms, its results out %.0f ms]",
                          static_cast<unsigned long long>(p.g_dec), static_cast<unsigned long long>(p.g_late),
                          p.g_start * 1e3, p.g_wait * 1e3, p.g_out * 1e3);
    }
    if (gpu8_) {
        std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), "; GPU int8 %.0f ms (kernels %.0f)",
                      p.gpu8 * 1e3, (gpu8_->busy_seconds() - p.gpu8_busy0) * 1e3);
        if (p.pf_cpu_experts > 0)
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), " [the CPU's share of the experts %.0f ms]",
                          p.pf_cpu_experts * 1e3);
        if (p.d_front + p.d_route + p.d_head > 0)
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf),
                          " [generating: enqueue fronts %.0f, waiting for routes %.0f, pinning %.0f, enqueue backs %.0f, "
                          "head %.0f ms]",
                          p.d_front * 1e3, p.d_route * 1e3, p.d_pin * 1e3, p.d_back * 1e3, p.d_head * 1e3);
        if (p.c_decide > 0)
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf),
                          " [CPU between a route and its experts: deciding %.0f, pinning %.0f, guesses %.0f, views %.0f, "
                          "enqueueing %.0f ms]",
                          p.c_decide * 1e3, p.c_pin * 1e3, p.c_guess * 1e3, p.c_views * 1e3, p.c_enq * 1e3);
        if (!gpu8_->pf_profile().empty())
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), " [batch kernels: %s]",
                          gpu8_->pf_profile().c_str());
        if (p.tail_unwaited > 0)
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), " [the tail not waited for: %llu picks]",
                          static_cast<unsigned long long>(p.tail_unwaited));
        if (p.d_front + p.d_route + p.d_head > 0)
            std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), " [pinning: %llu picks being read ahead waited %.0f ms, %llu not cached %.0f ms]",
                      static_cast<unsigned long long>(p.d_n_flight), p.d_wait_flight * 1e3,
                      static_cast<unsigned long long>(p.d_n_miss), p.d_wait_miss * 1e3);
    }
    // speculating (the GPU's or the CPU's): pairs verified, drafts that stood, the MTP layer's and the pairs' time
    if (p.verified > 0)
        std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf),
                      "; speculating: %llu pairs verified, %llu drafts fell (%.0f%% stood), %llu MTP drafts %.0f ms%s",
                      static_cast<unsigned long long>(p.verified), static_cast<unsigned long long>(p.rejected),
                      100.0 * static_cast<double>(p.verified - p.rejected) / static_cast<double>(p.verified),
                      static_cast<unsigned long long>(p.drafts), p.d_draft * 1e3,
                      p.d_verify > 0 ? (", pairs " + std::to_string(static_cast<long long>(p.d_verify * 1e3)) + " ms").c_str() : "");
    // the expert picks: in the cache, caught by a read-ahead, or read on the spot; and how many reads ahead were made
    std::string out = buf;
    if (gpu8_ && dec8_) {
        const std::string dp = gpu8_->dec_profile();
        if (!dp.empty()) out += "\nGPU generating, by kernel:\n" + dp;
    }
    if (p.spread_wall > 0) {
        std::snprintf(buf, sizeof buf,
                      "; routed experts generating: %.0f ms (summed over the threads: arithmetic %.0f, waiting for "
                      "reads %.0f)",
                      p.spread_wall * 1e3, p.spread_compute * 1e3, p.spread_wait * 1e3);
        out += buf;
    }
    if (p.rows_up + p.rows_down > 0) {
        std::snprintf(buf, sizeof buf,
                      "; routed experts by rows: gate and up %.0f ms (the threads waited for reads %.0f, summed), "
                      "down %.0f ms",
                      p.rows_up * 1e3, p.rows_wait * 1e3, p.rows_down * 1e3);
        out += buf;
    }
    {
        const ThreadPool::Stats ps = pool_->stats();
        out += "; threads: " + threads_status();
        if (ps.weighted > p.weighted0) {
            std::snprintf(buf, sizeof buf, " (%llu weighted products, %llu pieces taken from another's share)",
                          static_cast<unsigned long long>(ps.weighted - p.weighted0),
                          static_cast<unsigned long long>(ps.stolen - p.stolen0));
            out += buf;
        }
    }
    std::snprintf(buf, sizeof buf,
                  "; experts: %llu cached, %llu read ahead in time or in flight, %llu read on demand (%llu reads ahead "
                  "made, %llu dropped unread, %llu left out, %.0f MB read)",
                  static_cast<unsigned long long>(store_.hits() - p.hits0),
                  static_cast<unsigned long long>(store_.prefetch_hits() - p.prefetch_hits0),
                  static_cast<unsigned long long>(store_.misses() - p.misses0),
                  static_cast<unsigned long long>(store_.prefetches() - p.prefetches0),
                  static_cast<unsigned long long>(store_.prefetch_dropped() - p.dropped0),
                  static_cast<unsigned long long>(tail_skipped_ - p.skipped0),
                  static_cast<double>(store_.bytes_read() - p.bytes0) / 1048576.0);
    out += buf;
    if (route_prior_ > 0.0f) {
        const uint64_t picks = prior_picks_ - p.prior_picks0, moved = prior_moved_ - p.prior_moved0;
        if (prior_on() || picks)
            std::snprintf(buf, sizeof buf, "; route prior %.2f (CPU route): %llu of %llu picks moved to a cached expert",
                          static_cast<double>(route_prior_), static_cast<unsigned long long>(moved),
                          static_cast<unsigned long long>(picks));
        else
            std::snprintf(buf, sizeof buf, "; route prior %.2f: idle (%s)", static_cast<double>(route_prior_),
                          dec8_ ? "the GPU routes while generating" : "the cache holds every expert");
        out += buf;
    }
    if (store_.classes() > 1) {
        out += "; expert cache by format:";
        for (size_t ci = 0; ci < store_.classes(); ++ci) {
            const uint64_t picks = store_.class_picks(ci) - (ci < p.class_picks0.size() ? p.class_picks0[ci] : 0);
            const uint64_t miss = store_.class_misses(ci) - (ci < p.class_misses0.size() ? p.class_misses0[ci] : 0);
            std::snprintf(buf, sizeof buf, " %s %zu slots of %.2f MB, %.0f%% of %llu picks cached;",
                          ci < class_names_.size() ? class_names_[ci].c_str() : "?", store_.class_slots(ci),
                          static_cast<double>(store_.class_bytes(ci)) / 1048576.0,
                          picks ? 100.0 * static_cast<double>(picks - miss) / static_cast<double>(picks) : 0.0,
                          static_cast<unsigned long long>(picks));
            out += buf;
        }
    }
    return out;
}

void Model::threads_for(int n) {
    const bool token = n <= kSpreadTokens;
    // a prompt on every big thread; a token (or a few being checked) on decode_threads_ (model.hpp), or as many as
    // the system gives cores, and the little threads - not when the Adreno 7xx generates (the CPU only pins its
    // experts then)
    int w = pool_->size();
    if (token && decode_threads_ > 0) w = decode_threads_;
    else if (token && !dec8_) w = token_width();
    else if (!token) {
        // a prompt's products are not a token's: the window starts again after it
        pool_->take_cpus();
        width_passes_ = width_low_ = width_low_seen_ = 0;
    }
    pool_->set_width(w);
    pool_->set_little(token && !dec8_ && little_dense_ ? little_threads_ : 0);
}

int Model::token_width() {
    const int all = pool_->size();
    if (width_auto_ <= 0 || width_auto_ > all) width_auto_ = all;
    if (++width_passes_ < kWidthWindow) return width_auto_;
    width_passes_ = 0;
    const uint64_t cpus = pool_->take_cpus();
    int seen = 0;
    for (uint64_t m = cpus; m; m &= m - 1) ++seen;
    if (seen > 0 && seen < width_auto_) {
        // twice in a row: one window can be a passing squeeze
        width_low_seen_ = std::max(width_low_seen_, seen);
        if (++width_low_ >= 2) {
            width_auto_ = width_low_seen_;
            width_low_ = width_low_seen_ = width_windows_ = 0;
        }
    } else {
        width_low_ = width_low_seen_ = 0;
        if (seen > width_auto_) {   // the threads met more cores than there are threads: more came
            width_auto_ = std::min(all, seen);
            width_windows_ = 0;
        } else if (width_auto_ < all && ++width_windows_ >= kWidthProbe) {
            width_auto_ = all;
            width_windows_ = 0;
        }
    }
    return width_auto_;
}

std::string Model::threads_status() const {
    const int all = pool_->size();
    const int w = decode_threads_ > 0 ? std::min(decode_threads_, all)
                : !dec8_ && width_auto_ > 0 ? std::min(width_auto_, all) : all;
    char buf[360];
    std::snprintf(buf, sizeof buf,
                  "a token on %d big%s + %d little threads (%s; a little one's speed %.2f of a big one's, %s), a prompt "
                  "on %d big + %d; routed experts %s; GPU generating %s",
                  w, decode_threads_ <= 0 && w < all ? " (the cores the system gave)" : "",
                  dec8_ ? 0 : little_threads_, little_dense_ ? "every product" : "the little ones in the routed experts only",
                  pool_->little_weight(), pool_->learning() ? "learned" : "fixed",
                  pool_->size(), pool_->wide_helpers(),
                  routed_rows_ ? "split by rows" : "one a thread",
                  gpu_decode_ == 0 || !gpu_ ? "off" : gpu_decode_ == 1 ? "on (every pass)" : "on (verified pairs)");
    std::string s = buf;
    const std::string caps = core_capacities();
    if (!caps.empty()) s += "; cores " + caps;
    s += perf_active() ? "; the system's performance hints on" : "; no performance hints";
    return s;
}

bool Model::forward_batch(const int *tokens, int n, float *logits, bool all_logits, std::string &err) {
    const auto &c = cfg_;
    prof_.tokens += n;
    struct Clock {
        double &to, t0 = now_seconds();
        ~Clock() { to += now_seconds() - t0; }
    } clock{prof_.pass};
    // the calling thread computes too (parallel_for runs a share on it): keep it on a big core
    pin_current_thread(big_);
    threads_for(n);
    if (n <= 0 || n > kMaxPrompt || (n > kMaxBatch && (all_logits || trace))) {
        err = "bad batch size " + std::to_string(n);
        return false;
    }
    if (!reserve_context(pos_ + n, err)) return false;
    const int pad = vision_.available() ? vision_.shape().pad_token : -1;
    if (n == 1 && dec8_ && !trace && tokens[0] != pad) return decode_gpu8(tokens[0], logits, err);   // one token: the GPU's
    if (n == 1 && npu_.dec && !trace && tokens[0] != pad) return decode_npu(tokens, 1, logits, err);   // or the DSP's
    // in sub-batches of equal size, as large as the cache can lend scratch for
    const int most = n <= kMaxBatch && (n <= batch_cap_ || !budgeted_) ? n : std::min(kMaxBatch, prompt_batch());
    const int parts = (n + most - 1) / most, sub = (n + parts - 1) / parts;
    if (trace && parts > 1) {
        err = "a traced batch must fit the scratch";
        return false;
    }
    if (sub > batch_cap_ && !size_batch(sub, err)) return false;
    const int H = c.hidden;
    // the residual stream: the batch's own rows, or the whole prompt's
    if (parts > 1) long_x_.resize(static_cast<size_t>(n) * H);
    float *const X = parts > 1 ? long_x_.data() : x_.data();
    wide_ = sub >= kLayerPrefetch;
    apos_ = pos_;
    prompt_n_ = n;
    guessed_layer_ = -1;
    // a long batch needs nearly every expert of the first layer too: start reading them now
    if (prefetch_on_ && !c.dense && n >= kGuessAll && prefetch_[0].empty() &&
        store_.slots() >= static_cast<size_t>(2 * c.n_experts + c.top_k + 2))
        prefetch_layer(0);
    for (int t = 0; t < n; ++t) {
        store_.next_token();
        float *xr = X + static_cast<size_t>(t) * H;
        // an image's pads take its features, in order, all of them in a row
        if (pad >= 0 && tokens[t] == pad) {
            if (image_next_ >= images_.size()) {
                err = "an image token with no image (add_image)";
                return false;
            }
            Image &im = images_[image_next_];
            if (image_fed_ == 0) {
                im.kv0 = pos_ + t;
                im.s = im.kv0 + rope_delta_;
            }
            std::copy_n(im.feats.data() + static_cast<size_t>(image_fed_) * H, H, xr);
            if (++image_fed_ == im.n()) {
                set_rope_delta(im.s + std::max(im.gh, im.gw) - (im.kv0 + im.n()));
                ++image_next_;
                image_fed_ = 0;
            }
            continue;
        }
        if (image_fed_ > 0) {
            err = "an image's tokens cut short: " + std::to_string(image_fed_) + " of " +
                  std::to_string(images_[image_next_].n());
            return false;
        }
        if (!pack_.embedding(tokens[t], xr)) {
            err = "cannot read the embedding of token " + std::to_string(tokens[t]);
            return false;
        }
    }
    // a prompt on the DSP (the decoder's prompt program): every layer there, DEC_N tokens a call
    if (npu_.dec && npu_.prefill && n > DEC_T && !all_logits && !trace) return prefill_npu(X, n, logits, err);
    // A long batch is a scan through nearly every expert: the cache evicts behind it (ExpertStore::set_scan)
    const bool scan = n >= kLayerPrefetch;
    struct ScanOff {
        ExpertStore &s;
        bool on;
        ~ScanOff() {
            if (on) s.set_scan(-1, 1);
        }
    } scan_off{store_, scan};
    perf_thread();   // the caller computes too (the performance hints' session)
    for (int l = 0; l < c.n_layers; ++l) {
        const double t_layer = now_seconds();
        struct LayerDone {
            double t0;
            ~LayerDone() { perf_work(now_seconds() - t0); }
        } layer_done{t_layer};
        const Layer &L = layers_[static_cast<size_t>(l)];
        if (stop_ && stop_->load()) {
            err = "stopped";
            return false;
        }
        if (scan) store_.set_scan(l, static_cast<uint32_t>(c.n_experts));
        if (trace) std::copy(x_.begin(), x_.begin() + static_cast<ptrdiff_t>(n) * H, xin_.begin());
        // every sub-batch through this layer, in order (the Gated DeltaNet state and the KV cache carry on)
        for (int o = 0; o < n; o += sub) {
            const int m = std::min(sub, n - o);
            float *const Xo = X + static_cast<size_t>(o) * H;
            apos_ = pos_ + o;
            // (a sub-batch's per-token steps on every thread: each token's arithmetic as one at a time)
            par(m, [&](int64_t a, int64_t b) {
                for (int64_t t = a; t < b; ++t)
                    rmsnorm(Xo + static_cast<size_t>(t) * H, L.in_norm, h_.data() + static_cast<size_t>(t) * H, H, c.eps, true);
            }, 8);
            const double tm = now_seconds();
            const int on_gpu = mixer_gpu(l, L, m, err);
            if (on_gpu < 0) return false;
            if (on_gpu == 0) {
                if (L.attn) attention(L, m);
                else gdn(L, m);
            }
            (L.attn ? prof_.attention : prof_.gdn) += now_seconds() - tm;
            par(m, [&](int64_t a, int64_t b) {
                for (int64_t t = a; t < b; ++t) {
                    float *xr = Xo + static_cast<size_t>(t) * H;
                    const float *mr = mix_.data() + static_cast<size_t>(t) * H;
                    for (int i = 0; i < H; ++i) xr[i] += mr[i];
                    rmsnorm(xr, L.post_norm, h_.data() + static_cast<size_t>(t) * H, H, c.eps, true);
                }
            }, 8);
            const double tq = now_seconds();
            // (the layer's read-ahead pinned to its last sub-batch; GEODE_PF_MOE=0: the router and the shared
            // expert on the CPU even when the mixer ran on the GPU)
            static const bool pf_moe = !(std::getenv("GEODE_PF_MOE") && std::atoi(std::getenv("GEODE_PF_MOE")) == 0);
            if (!moe(l, L, m, err, o + m < n, on_gpu == 1 && pf_moe)) return false;
            prof_.moe += now_seconds() - tq;
            par(static_cast<int64_t>(m) * H, [&](int64_t a, int64_t b) {
                for (int64_t i = a; i < b; ++i) Xo[i] += moe_[static_cast<size_t>(i)];
            }, 4096);
        }
        apos_ = pos_;
        if (trace)
            for (int t = 0; t < n; ++t) {
                const size_t o = static_cast<size_t>(t) * H;
                LayerTrace tr;
                tr.layer = l;
                tr.pos = pos_ + t;
                tr.x = xin_.data() + o;
                tr.mixer = mix_.data() + o;
                tr.moe = moe_.data() + o;
                tr.out = x_.data() + o;
                tr.experts = top_ids_.data() + static_cast<size_t>(t) * k_;
                tr.weights = top_w_.data() + static_cast<size_t>(t) * k_;
                if (trace_routed_) tr.routed = trace_routed_ + static_cast<size_t>(t) * k_ * H;
                if (trace_shared_) tr.shared = trace_shared_ + o;
                if (static_cast<size_t>(t) < trace_gate_.size()) tr.shared_gate = trace_gate_[static_cast<size_t>(t)];
                trace(tr);
            }
    }
    // the hidden states the MTP layer drafts from
    if (parts > 1) hid_long_.resize(static_cast<size_t>(n) * H);
    else hid_long_ = {};
    std::vector<float> &hid = parts > 1 ? hid_long_ : hid_;
    if (mtp_)
        for (int t = 0; t < n; ++t) {
            const float *x = X + static_cast<size_t>(t) * H;
            float *d = hid.data() + static_cast<size_t>(t) * H;
            rmsnorm(x, final_norm_, d, H, c.eps, true);
        }
    if (mtp_) last_hid_.assign(hid.begin() + static_cast<ptrdiff_t>(n - 1) * H, hid.begin() + static_cast<ptrdiff_t>(n) * H);
    if (logits) {
        const int first = all_logits ? 0 : n - 1;
        std::vector<const float *> xs;
        std::vector<float *> ys;
        for (int t = first; t < n; ++t) {
            float *h = h_.data() + static_cast<size_t>(t - first) * H;
            rmsnorm(X + static_cast<size_t>(t) * H, final_norm_, h, H, c.eps, true);
            xs.push_back(h);
            ys.push_back(logits + static_cast<size_t>(t - first) * c.vocab);
        }
        const double th = now_seconds();
        matmul(lm_head_, static_cast<int>(xs.size()), xs.data(), ys.data());
        prof_.head += now_seconds() - th;
    }
    pos_ += n;
    return true;
}

}  // namespace geode
