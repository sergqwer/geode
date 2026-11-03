// gemma.cpp - Gemma 4 E2B / E4B (arch "gemma4", tools/pack_gemma4.py) in geode::Model: the layers, the per-layer
// inputs (PLE), sliding and full attention with KV sharing, the soft-capped head. tools/gemma4.py writes the forward
// pass out step by step and is what this follows term by term; the threads, the int8 products, the KV cache and its
// growth and the conversation's state are the class's own (model.cpp, state.cpp).
//
// As in model.cpp, a token's arithmetic does not depend on its batch: a prompt read in chunks gives bit for bit the
// logits it gives read one token at a time.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "model.hpp"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define GEODE_GEMMA_NEON 1
#if defined(__ARM_FEATURE_DOTPROD)
#define GEODE_GEMMA_DOT 1   // SDOT: the int8 KV cache's scores against int8 queries
#endif
#if defined(__linux__)
#include <sys/auxv.h>   // getauxval: i8mm (i8mm_here)
#endif
#endif

namespace geode {

namespace {

const Tensor *find_t(const Pack &p, const std::string &name, std::string &err) {
    const Tensor *t = p.find(name);
    if (!t) err = "the pack has no " + name;
    return t;
}

// a prompt's chunk this long and longer attends on i8mm (gemma_attend_mm); a verified batch of the drafter's (2..5
// tokens) stays on the path a single token takes: its rows are its single passes' bit for bit
constexpr int kMmMinTokens = 32;

// gelu with tanh (transformers' gelu_pytorch_tanh)
inline float gelu_tanh(float x) {
    constexpr float k = 0.7978845608028654f;   // sqrt(2 / pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
// the same as x sigmoid(2u) on fast_exp (vision.cpp's): no call, so its loops vectorize - a prompt's MLPs take ~0.7 G
// of them for 2048 tokens (std::tanh: seconds of the prompt). Not in exact arithmetic (the reference's form there)
inline float gelu_fast(float x) {
    return x / (1.0f + fast_exp(-1.5957691216057308f * (x + 0.044715f * x * x * x)));
}
// tanh(y) = (1 - e^-2y) / (1 + e^-2y) on fast_exp (the logits' soft cap, a vocabulary's worth a token)
inline float tanh_fast(float y) {
    const float e = fast_exp(-2.0f * y);
    return (1.0f - e) / (1.0f + e);
}


// a KV head's row to int8: amax / 127 its scale (model.cpp's quant_kv_row)
inline void quant_row(const float *x, int n, int8_t *q, float &scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < n; ++i) q[i] = static_cast<int8_t>(std::clamp(std::nearbyint(x[i] * inv), -127.0f, 127.0f));
    scale = amax / 127.0f;
}

#if defined(GEODE_GEMMA_NEON)
// model.cpp's attention helpers (its anonymous namespace): eight values of a K or V row as two FP32 vectors
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
#if defined(GEODE_GEMMA_DOT) && defined(__linux__)
#define GEODE_GEMMA_MM 1
// SMMLA and USMMLA here (FEAT_I8MM: the 8 Elite's Oryon, not the 720G's A76); GEODE_NO_I8MM=1 leaves them off. On the
// Oryon's cores (measured; products a cycle on a prime core / on the others): SMMLA 128 / 64, SDOT 64 / 32, BFMMLA
// 20 / 16, FP32 FMLA 12 / 8 - of the matrix instructions only the int8 ones pay
bool i8mm_here() {
    static const bool yes = [] {
        const char *off = std::getenv("GEODE_NO_I8MM");
        return (getauxval(AT_HWCAP2) & (1UL << 13)) != 0 && !(off && std::atoi(off) != 0);   // HWCAP2_I8MM
    }();
    return yes;
}
// The integer scores of 8 heads over a block's position pairs j0..j1 (multiples of 4): qp = the token's 4 head pairs,
// [pair][hd / 8][16: the pair's two heads' 8 values]; kp = the block's keys, [position pair][hd / 8][16: the pair's
// two positions' 8 values]; out[g * ld + 2 (j - j0) + x]. An SMMLA multiplies 2 heads x 8 values by 8 values x 2
// positions (32 products); 4 head pairs x 4 position pairs a tile (16 accumulators). Exact in int32: the integers
// qk_dots_i8 gives.
__attribute__((target("arch=armv8.6-a+i8mm"))) void qk_mm8(const int8_t *qp, const int8_t *kp, int j0, int j1, int hd,
                                                            int32_t *out, int ld) {
    const int nc = hd / 8;
    for (int j = j0; j < j1; j += 4) {
        int32x4_t a[4][4];
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y) a[x][y] = vdupq_n_s32(0);
        const int8_t *k0 = kp + static_cast<size_t>(j) * nc * 16;
        for (int c = 0; c < nc; ++c) {
            int8x16_t kv[4];
            for (int y = 0; y < 4; ++y) kv[y] = vld1q_s8(k0 + (static_cast<size_t>(y) * nc + c) * 16);
            for (int x = 0; x < 4; ++x) {
                const int8x16_t qv = vld1q_s8(qp + (static_cast<size_t>(x) * nc + c) * 16);
                for (int y = 0; y < 4; ++y) a[x][y] = vmmlaq_s32(a[x][y], qv, kv[y]);
            }
        }
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y) {
                const int col = 2 * (j - j0 + y);
                vst1_s32(out + static_cast<size_t>(2 * x) * ld + col, vget_low_s32(a[x][y]));
                vst1_s32(out + static_cast<size_t>(2 * x + 1) * ld + col, vget_high_s32(a[x][y]));
            }
    }
}
// The weighted values of 8 heads over a block, in integers: w = the weights as u8, [head pair][no octets][16: the
// pair's two heads x the octet's 8 positions]; vt = the block's values transposed, [hd / 2 dim pairs][octs][16: the
// pair's two dims x an octet's 8 positions], from its octet o0; out[g * hd + d]. A USMMLA multiplies 2 heads x 8
// positions by 8 positions x 2 dims (32 products, the weights unsigned); 4 head pairs x 4 dim pairs a tile. In int32:
// 255 x 127 x a block's positions (8192 at most) < 2^31.
__attribute__((target("arch=armv8.6-a+i8mm"))) void pv_mm8(const uint8_t *w, const int8_t *vt, int o0, int no, int octs,
                                                            int hd, int32_t *out) {
    for (int dp = 0; dp < hd / 2; dp += 4) {
        int32x4_t a[4][4];
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y) a[x][y] = vdupq_n_s32(0);
        const int8_t *v0 = vt + (static_cast<size_t>(dp) * octs + o0) * 16;
        for (int o = 0; o < no; ++o) {
            int8x16_t vv[4];
            for (int y = 0; y < 4; ++y) vv[y] = vld1q_s8(v0 + (static_cast<size_t>(y) * octs + o) * 16);
            for (int x = 0; x < 4; ++x) {
                const uint8x16_t wv = vld1q_u8(w + (static_cast<size_t>(x) * no + o) * 16);
                for (int y = 0; y < 4; ++y) a[x][y] = vusmmlaq_s32(a[x][y], wv, vv[y]);
            }
        }
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y) {
                const int d = 2 * (dp + y);
                vst1_s32(out + static_cast<size_t>(2 * x) * hd + d, vget_low_s32(a[x][y]));
                vst1_s32(out + static_cast<size_t>(2 * x + 1) * hd + d, vget_high_s32(a[x][y]));
            }
    }
}
// the keys and values gemma_attend_mm rearranges at a time, bytes of each: past it, the blocks in groups
// (GEODE_MM_BUILD_KB: smaller, for the tests - a short prompt then takes several)
size_t mm_build_bytes() {
    static const size_t b = [] {
        const char *e = std::getenv("GEODE_MM_BUILD_KB");
        const long kb = e ? std::atol(e) : 0;
        return kb > 0 ? static_cast<size_t>(kb) << 10 : size_t{8} << 20;
    }();
    return b;
}
#endif

#if defined(GEODE_GEMMA_DOT)
// The integer dots of G int8 query rows (hd values) with np int8 key rows `stride` bytes apart, into out[g * ld + p]:
// four heads by four positions at a time (16 accumulators; a key row loaded once for four heads), SDOT over 16 values
// a step. Exact in int32 (127 x 127 x 512 < 2^31), so the order does not matter: a token's scores are the same in a
// batch and alone.
template <int G>
inline void qk_dots_i8(const int8_t *const *q, const int8_t *k, size_t stride, int np, int hd, int32_t *out, int ld) {
    static_assert(G == 1 || G % 4 == 0, "four heads at a time, or one");
    int p = 0;
    if constexpr (G % 4 == 0) {
        for (; p + 4 <= np; p += 4) {
            const int8_t *k0 = k + static_cast<size_t>(p) * stride, *k1 = k0 + stride, *k2 = k1 + stride,
                         *k3 = k2 + stride;
            for (int g0 = 0; g0 < G; g0 += 4) {
                int32x4_t a[4][4];
                for (int g = 0; g < 4; ++g)
                    for (int j = 0; j < 4; ++j) a[g][j] = vdupq_n_s32(0);
                for (int i = 0; i < hd; i += 16) {
                    const int8x16_t x0 = vld1q_s8(k0 + i), x1 = vld1q_s8(k1 + i), x2 = vld1q_s8(k2 + i),
                                    x3 = vld1q_s8(k3 + i);
                    for (int g = 0; g < 4; ++g) {
                        const int8x16_t qv = vld1q_s8(q[g0 + g] + i);
                        a[g][0] = vdotq_s32(a[g][0], qv, x0);
                        a[g][1] = vdotq_s32(a[g][1], qv, x1);
                        a[g][2] = vdotq_s32(a[g][2], qv, x2);
                        a[g][3] = vdotq_s32(a[g][3], qv, x3);
                    }
                }
                for (int g = 0; g < 4; ++g)
                    for (int j = 0; j < 4; ++j) out[(g0 + g) * ld + p + j] = vaddvq_s32(a[g][j]);
            }
        }
    }
    for (; p < np; ++p) {   // the rest one position at a time
        const int8_t *kr = k + static_cast<size_t>(p) * stride;
        for (int g = 0; g < G; ++g) {
            int32x4_t a = vdupq_n_s32(0);
            for (int i = 0; i < hd; i += 16) a = vdotq_s32(a, vld1q_s8(q[g] + i), vld1q_s8(kr + i));
            out[g * ld + p] = vaddvq_s32(a);
        }
    }
}
#endif

// the G scores q[g] . k of one position: the portable loop's eight partial sums as lanes 0-3 and 4-7, its order
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

bool Model::bind_gemma(std::string &err) {
    const auto &c = cfg_;
    const int H = c.hidden, nh = c.n_heads, nkv = c.n_kv_heads, P = c.ple_dim;
    auto f32 = [&](const std::string &n, int64_t count) -> const float * {
        const Tensor *t = find_t(pack_, n, err);
        if (!t) return nullptr;
        if (t->type != DType::F32 || t->rows * t->cols != count || !t->data) {
            err = n + ": expected f32 x " + std::to_string(count);
            return nullptr;
        }
        return t->f32();
    };
    auto mat = [&](const std::string &n, int64_t rows, int64_t cols, QView &out) -> bool {
        const Tensor *t = find_t(pack_, n, err);
        if (!t) return false;
        if (!is_quantized(t->type) || t->rows != rows || t->cols != cols || !t->data) {
            err = n + ": expected a quantized [" + std::to_string(rows) + ", " + std::to_string(cols) + "]";
            return false;
        }
        out = t->view();
        return true;
    };
    glayers_.assign(static_cast<size_t>(c.n_layers), {});
    int slots = 0;
    for (int l = 0; l < c.n_layers; ++l) {
        GLayer &L = glayers_[static_cast<size_t>(l)];
        const size_t i = static_cast<size_t>(l);
        const std::string p = "layers." + std::to_string(l) + ".";
        L.full = c.attention[i];
        L.own_kv = c.kv_layer[i] == l;
        L.slot = L.own_kv ? slots++ : glayers_[static_cast<size_t>(c.kv_layer[i])].slot;
        L.hd = c.head_dims[i];
        L.ffn = c.ffn[i];
        L.pairs = c.rot_pairs[i];
        L.inv_freq.resize(static_cast<size_t>(L.pairs));
        for (int j = 0; j < L.pairs; ++j)   // theta^(-2j / head_dim): "proportional" RoPE divides by the whole head
            L.inv_freq[static_cast<size_t>(j)] =
                1.0f / std::pow(c.rope_thetas[i], static_cast<float>(2 * j) / static_cast<float>(L.hd));
        const int hd = L.hd;
        if (!(L.in_norm = f32(p + "input_layernorm.weight", H)) ||
            !(L.post_attn = f32(p + "post_attention_layernorm.weight", H)) ||
            !(L.pre_ff = f32(p + "pre_feedforward_layernorm.weight", H)) ||
            !(L.post_ff = f32(p + "post_feedforward_layernorm.weight", H)) ||
            !(L.q_norm = f32(p + "self_attn.q_norm.weight", hd)) ||
            !mat(p + "self_attn.q_proj.weight", static_cast<int64_t>(nh) * hd, H, L.q) ||
            !mat(p + "self_attn.o_proj.weight", H, static_cast<int64_t>(nh) * hd, L.o) ||
            !mat(p + "mlp.gate_proj.weight", L.ffn, H, L.gate) || !mat(p + "mlp.up_proj.weight", L.ffn, H, L.up) ||
            !mat(p + "mlp.down_proj.weight", H, L.ffn, L.down))
            return false;
        if (L.own_kv && (!(L.k_norm = f32(p + "self_attn.k_norm.weight", hd)) ||
                         !mat(p + "self_attn.k_proj.weight", static_cast<int64_t>(nkv) * hd, H, L.k) ||
                         !mat(p + "self_attn.v_proj.weight", static_cast<int64_t>(nkv) * hd, H, L.v)))
            return false;
        if (P && (!(L.ple_norm = f32(p + "post_per_layer_input_norm.weight", H)) ||
                  !mat(p + "per_layer_input_gate.weight", P, H, L.ple_gate) ||
                  !mat(p + "per_layer_projection.weight", H, P, L.ple_proj)))
            return false;
        const float *s = f32(p + "layer_scalar", 1);
        if (!s) return false;
        L.scalar = s[0];
    }
    if (P) {
        if (!mat("per_layer_model_projection.weight", static_cast<int64_t>(c.n_layers) * P, H, ple_model_) ||
            !(ple_proj_norm_ = f32("per_layer_projection_norm.weight", P)))
            return false;
        ple_table_ = find_t(pack_, "embed_tokens_per_layer.weight", err);
        if (!ple_table_ || !is_quantized(ple_table_->type) || ple_table_->cols != static_cast<int64_t>(c.n_layers) * P ||
            ple_table_->rows != c.vocab) {
            err = "embed_tokens_per_layer.weight: expected a quantized [vocab, layers x ple_dim]";
            return false;
        }
    }
    const Tensor *head = find_t(pack_, "lm_head.weight", err);
    const Tensor *norm = find_t(pack_, "norm.weight", err);
    if (!head || !norm) return false;
    if (!is_quantized(head->type) || head->cols != H || head->rows != c.vocab || !head->data) {
        err = "lm_head.weight: expected a quantized [vocab, hidden]";
        return false;
    }
    lm_head_ = head->view();
    final_norm_ = norm->f32();
    return true;
}

bool Model::open_gemma(const ModelOptions &opt, std::string &err) {
    const auto &c = cfg_;
    // the NPU: a prompt's products on HMX (Model::matmul sends a batch of kNpuMinTokens and more there), the rest of
    // the prompt and the answer on the CPU. It takes the pack's row layout: before quad_dense lays them out anew
    const bool npu_on = std::getenv("GEODE_NPU") ? std::atoi(std::getenv("GEODE_NPU")) != 0 : opt.npu > 0;
    if (exact_ || check_) npu_.status = "off (exact / checked arithmetic)";
    else if (!npu_on) npu_.status = "off (the setting)";
    else setup_npu();
    // a prompt's chunk: larger with the NPU, whose call expands every weight block once whatever the tokens
    gemma_chunk_ = npu_.g ? kGemmaNpuChunk : kGemmaChunk;
    if (const char *e = std::getenv("GEODE_GEMMA_CHUNK")) gemma_chunk_ = std::clamp(std::atoi(e), 16, kMaxPrompt);
    // the GPU (an Adreno 7xx+): a prompt's products where the NPU does not take them - before quad_dense, as the NPU
    const bool gpu_on = std::getenv("GEODE_GPU") ? std::atoi(std::getenv("GEODE_GPU")) != 0 : opt.gpu;
    if (!gpu_on) {
        gpu_status_ = "off (the setting)";
    } else if (exact_ || check_) {
        gpu_status_ = "off (exact / checked arithmetic)";
    } else if (npu_.g) {
        gpu_status_ = "off (the NPU takes the prompt's products)";
    } else {
        std::string why;
        if (!setup_gpu8_gemma(why) && gpu_status_.rfind("off", 0) != 0) gpu_status_ = "off (" + why + ")";
    }
    // attention in blocks of 256 positions (model.cpp: 1024): E2B has one KV head, so a token generated would
    // otherwise attend on one thread; more blocks, more threads (the blocks start at position 0 whatever the batch)
    attn_block_ = 256;
    if (const char *e = std::getenv("GEODE_ATTN_BLOCK")) attn_block_ = std::max(1, std::atoi(e));
    if (const char *e = std::getenv("GEODE_DECODE_THREADS")) decode_threads_ = std::max(1, std::atoi(e));
    // the matrices in the quad layout (the CPU's kernels read it faster; kernels.hpp), as model.cpp's open() does;
    // the embedding's rows still come from the file (Pack::embedding), the PLE table's from flash
    if (!exact_ && opt.quad_layout && !std::getenv("GEODE_NO_QUAD")) quad_dense();
    int own = 0;
    for (const GLayer &L : glayers_) own += L.own_kv;
    // the KV cache: a row per position and own-KV layer, as wide as the widest layer's (a sliding layer's heads use
    // its first n_kv_heads x its head dim)
    n_attn_ = own;
    kv_heads_ = c.n_kv_heads;
    kv_row_ = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    kv_chunk_ = opt.kv_chunk > 0 ? opt.kv_chunk : kKvChunk;
    kv_i8_ = !exact_ && opt.kv_int8 && !std::getenv("GEODE_KV_FP16");
    kv_chunk_bytes_ = kv_i8_ ? 2 * static_cast<size_t>(own) * kv_chunk_ * (kv_row_ + static_cast<size_t>(kv_heads_) * sizeof(float))
                             : 2 * static_cast<size_t>(own) * kv_chunk_ * kv_row_ * (exact_ ? sizeof(float) : sizeof(_Float16));
    // no experts: a token cache of two 4 KiB slots keeps the store's code paths whole (as Qwen3.5-2B's); the budget
    // bounds the context only
    const size_t slot = 4096, min_slots = 2;
    min_slots_ = min_slots;
    slot_bytes_ = slot;
    budgeted_ = false;
    // the per-token scratch (size_batch's buffers, sized from the config: qg_ / core_ by the widest head dim, sh_act_
    // by the widest MLP, ple_ / pgate_ the per-layer inputs) and the widest activation an int8 product quantizes
    const size_t H = static_cast<size_t>(c.hidden), q = static_cast<size_t>(c.n_heads) * c.head_dim;
    token_floats_ = 7 * H + 3 * q + 2 * kv_row_ + 2 * static_cast<size_t>(c.shared_ff) +
                    static_cast<size_t>(c.n_layers + 1) * c.ple_dim;
    max_cols_ = std::max({H, q, static_cast<size_t>(c.shared_ff)});
    // what a prompt's chunk holds besides the weights and the KV cache: its scratch (kGemmaChunk tokens: the int8
    // activations at most a quarter more), attention's partial sums (gemma_attend: up to 16 MiB), a margin
    // the drafter (drafter.bin, read whole), when the pack has one and it is wanted
    const Tensor *dh = pack_.manifest()["drafter"].is_object() && !std::getenv("GEODE_NO_DRAFTER") ? pack_.find("drafter.head") : nullptr;
    const uint64_t draft_bytes = dh ? pack_.file(dh->file).size() : 0;
    // the fixed part with the NPU's copies of the matrices (besides the CPU's: the answer reads those) or without
    const auto fixed_of = [&](bool npu) {
        const uint64_t chunk = static_cast<uint64_t>(npu ? gemma_chunk_ : kGemmaChunk);
        const uint64_t scratch = chunk * token_floats_ * sizeof(float) * 5 / 4 + (16ull << 20);
        return pack_.dense_bytes() + (32ull << 20) + scratch + kv_chunk_bytes_ + draft_bytes +
               (npu ? static_cast<uint64_t>(npu_.bytes(npu_.g)) : 0) + (gpu8_ ? gpu_staging_ : 0);
    };
    // the NPU only where its copies leave the context asked for its room (a faster prompt is not worth a shorter
    // conversation): otherwise the CPU's prompt, its copies freed
    if (opt.memory_budget && npu_.g) {
        const uint64_t kv_all = static_cast<uint64_t>((ctx_ + kv_chunk_ - 1) / kv_chunk_ - 1) * kv_chunk_bytes_;
        if (opt.memory_budget < fixed_of(true) + min_slots * slot + kv_all) {
            npu_.status = "off (its copies of the matrices, " + std::to_string(npu_.bytes(npu_.g) >> 20) +
                          " MB, would leave too little of the memory budget for the context)";
            npu_.close(npu_.g);
            npu_.g = nullptr;
            npu_.ids.clear();
            npu_.adopted.clear();
            npu_.parts.clear();
            gemma_chunk_ = kGemmaChunk;
        }
    }
    const uint64_t fixed = fixed_of(npu_.g != nullptr);
    if (opt.memory_budget) {
        if (opt.memory_budget < fixed + min_slots * slot) {
            err = "the memory budget (" + std::to_string(opt.memory_budget >> 20) + " MB) is below what the model needs: " +
                  std::to_string((fixed + min_slots * slot) >> 20) + " MB";
            return false;
        }
        const uint64_t room = opt.memory_budget - fixed - min_slots * slot;
        ctx_ = std::min(ctx_, (1 + static_cast<int>(room / std::max<size_t>(1, kv_chunk_bytes_))) * kv_chunk_);
    }
    if (!store_.init(min_slots, slot, 0, 1, little_cores(), {})) {
        err = "not enough memory for the token cache";
        return false;
    }
    store_.set_floor(min_slots);
    prefetch_.assign(static_cast<size_t>(c.n_layers), {});
    prefetch_on_ = false;
    conv_n_ = gdn_n_ = 0;
    conv_st_ = conv_state_.data();
    gdn_st_ = gdn_state_.data();
    if (!new_kv_chunk(err)) return false;
    if (!size_batch(kSmallBatch, err)) return false;
    k_ = 0;
    // the drafter: speculative decoding when it opens (the model works without it)
    if (!pack_.manifest()["drafter"].is_object()) drafter_status_ = "none";
    else if (std::getenv("GEODE_NO_DRAFTER")) drafter_status_ = "off (GEODE_NO_DRAFTER)";
    else if (!dh) drafter_status_ = "no drafter.bin";
    else {
        std::string de;
        if (open_drafter(de)) drafter_status_ = "ok";
        else {
            gd_ = GDraft{};
            drafter_status_ = de;
        }
    }
    return true;
}

bool Model::forward_gemma(const int *tokens, int n, float *logits, bool all_logits, std::string &err) {
    // in chunks of up to gemma_chunk_ tokens, each through every layer (a dense model reads its weights once a chunk
    // either way)
    const int parts = (n + gemma_chunk_ - 1) / gemma_chunk_, sub = (n + parts - 1) / parts;
    // the drafter's hidden rows: the whole call's (hidden(t), t < n), in hid_long_ when it takes several chunks
    const bool keep = keeps_hidden();
    if (keep) {
        if (parts > 1) hid_long_.resize(static_cast<size_t>(n) * cfg_.hidden);
        else hid_long_ = {};
    }
    for (int o = 0; o < n; o += sub) {
        const int m = std::min(sub, n - o);
        const bool last = o + m == n;
        float *lg = !logits ? nullptr : all_logits ? logits + static_cast<size_t>(o) * cfg_.vocab : last ? logits : nullptr;
        if (!gemma_chunk(tokens + o, m, lg, all_logits, err, o)) return false;
    }
    if (keep) last_hid_.assign(hidden(n - 1), hidden(n - 1) + cfg_.hidden);
    return true;
}

bool Model::gemma_chunk(const int *tokens, int n, float *logits, bool all_logits, std::string &err, int hid_row) {
    const auto &c = cfg_;
    const int H = c.hidden, P = c.ple_dim, LP = c.n_layers * P;
    if (n > batch_cap_ && !size_batch(n, err)) return false;
    float *const X = x_.data();
    apos_ = pos_;
    const auto xs = rows(x_, static_cast<size_t>(H), n);
    // an image's <|image|> tokens take its soft tokens' features (add_image), a clip's <|audio|> tokens its
    // (add_audio): in the order they were added, each one's all in a row; their per-layer inputs then the pad token's
    // table row (tools/gemma4.py, Text.ple)
    const int pad = vision_.available() ? vision_.shape().pad_token : -1;
    const int apad = audio_.available() ? audio_.shape().pad_token : -1;
    const int vpad = vision_.available() ? vision_.shape().video_pad : -1;   // a video frame's <|video|>
    if (media_.size() < static_cast<size_t>(n)) media_.resize(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) {
        float *xr = X + static_cast<size_t>(t) * H;
        media_[static_cast<size_t>(t)] = (pad >= 0 && tokens[t] == pad) || (apad >= 0 && tokens[t] == apad) ||
                                         (vpad >= 0 && tokens[t] == vpad);
        if (media_[static_cast<size_t>(t)]) {
            if (image_next_ >= images_.size()) {
                err = "a media token with nothing added for it (add_image, add_audio)";
                return false;
            }
            Image &im = images_[image_next_];
            if (im.pad != tokens[t]) {
                err = "a media token of another kind than the next one added (images and clips go in as added)";
                return false;
            }
            if (image_fed_ == 0) im.kv0 = im.s = pos_ + t;
            std::copy_n(im.feats.data() + static_cast<size_t>(image_fed_) * H, H, xr);
            if (++image_fed_ == im.n()) {
                ++image_next_;
                image_fed_ = 0;
            }
            continue;
        }
        if (image_fed_ > 0) {
            err = "an image's or clip's tokens cut short: " + std::to_string(image_fed_) + " of " +
                  std::to_string(images_[image_next_].n());
            return false;
        }
        if (!pack_.embedding(tokens[t], xr)) {
            err = "cannot read the embedding of token " + std::to_string(tokens[t]);
            return false;
        }
        for (int i = 0; i < H; ++i) xr[i] *= c.embed_scale;
    }
    // the per-layer inputs: the embeddings' projection, normed per layer, plus the token's row of the table
    if (P) {
        if (ple_.size() < static_cast<size_t>(n) * LP) ple_.resize(static_cast<size_t>(n) * LP);
        if (pgate_.size() < static_cast<size_t>(n) * P) pgate_.resize(static_cast<size_t>(n) * P);
        const auto ps = rows(ple_, static_cast<size_t>(LP), n);
        matmul(ple_model_, n, xs.data(), ps.data());
        const float proj_scale = 1.0f / std::sqrt(static_cast<float>(H)), tok_scale = std::sqrt(static_cast<float>(P)),
                    mix = 1.0f / std::sqrt(2.0f);
        std::atomic<bool> read_ok{true};
        par(n, [&](int64_t t0, int64_t t1) {
            thread_local std::vector<float> row;
            row.resize(static_cast<size_t>(LP));
            for (int64_t t = t0; t < t1; ++t) {
                float *pr = ps[static_cast<size_t>(t)];
                if (!pack_.read_row(*ple_table_, media_[static_cast<size_t>(t)] ? c.pad_id : tokens[t], row.data())) {
                    read_ok.store(false, std::memory_order_relaxed);
                    continue;
                }
                for (int l = 0; l < c.n_layers; ++l) {
                    float *s = pr + static_cast<size_t>(l) * P;
                    for (int i = 0; i < P; ++i) s[i] *= proj_scale;
                    rmsnorm(s, ple_proj_norm_, s, P, c.eps, false);
                    const float *r = row.data() + static_cast<size_t>(l) * P;
                    for (int i = 0; i < P; ++i) s[i] = (s[i] + r[i] * tok_scale) * mix;
                }
            }
        });
        if (!read_ok.load()) {
            err = "cannot read the per-layer embeddings";
            return false;
        }
    }
    const auto hs = rows(h_, static_cast<size_t>(H), n);
    const auto ms = rows(mix_, static_cast<size_t>(H), n);
    const auto os = rows(moe_, static_cast<size_t>(H), n);
    for (int l = 0; l < c.n_layers; ++l) {
        const GLayer &L = glayers_[static_cast<size_t>(l)];
        if (stop_ && stop_->load()) {
            err = "stopped";
            return false;
        }
        if (trace) std::copy(x_.begin(), x_.begin() + static_cast<ptrdiff_t>(n) * H, xin_.begin());
        const double ta = now_seconds();
        par(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) rmsnorm(xs[static_cast<size_t>(t)], L.in_norm, hs[static_cast<size_t>(t)], H, c.eps, false);
        }, 8);
        gemma_attention(L, n);   // -> mix_
        par(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) {
                float *x = xs[static_cast<size_t>(t)], *m = ms[static_cast<size_t>(t)];
                rmsnorm(m, L.post_attn, m, H, c.eps, false);
                for (int i = 0; i < H; ++i) x[i] += m[i];
                rmsnorm(x, L.pre_ff, hs[static_cast<size_t>(t)], H, c.eps, false);
            }
        }, 8);
        prof_.attention += now_seconds() - ta;
        // the MLP: down(gelu(gate) * up) - on the NPU in one go where it takes it (gelu(gate) * up stays there)
        const double tm = now_seconds();
        if (exact_ || !npu_mlp(L.gate, L.up, L.down, 0, n, hs.data(), os.data())) {
            const size_t w = 2 * static_cast<size_t>(L.ffn);
            const auto gs = rows(sh_act_, w, n), us = rows(sh_act_, w, n, static_cast<size_t>(L.ffn));
            const QAct *const *qh = npu_takes(L.gate, n) || gpu_takes(L.gate, n) ? nullptr : quantize_h(hs, H);
            const QView *mats[2] = {&L.gate, &L.up};
            float *const *outs[2] = {gs.data(), us.data()};
            matmul_many(mats, 2, n, hs.data(), outs, qh);
            const int64_t ffn = L.ffn;
            par(static_cast<int64_t>(n) * ffn, [&](int64_t a, int64_t b) {
                // a stretch of a row at a time (the index divided once a stretch: the inner loop vectorizes)
                for (int64_t i = a; i < b;) {
                    const int64_t t = i / ffn, j0 = i % ffn, j1 = std::min(ffn, j0 + (b - i));
                    float *g = gs[static_cast<size_t>(t)];
                    const float *u = us[static_cast<size_t>(t)];
                    if (exact_)
                        for (int64_t j = j0; j < j1; ++j) g[j] = gelu_tanh(g[j]) * u[j];
                    else
                        for (int64_t j = j0; j < j1; ++j) g[j] = gelu_fast(g[j]) * u[j];
                    i += j1 - j0;
                }
            }, 4096);
            matmul(L.down, n, gs.data(), os.data());
        }
        par(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) {
                float *x = xs[static_cast<size_t>(t)], *m = os[static_cast<size_t>(t)];
                rmsnorm(m, L.post_ff, m, H, c.eps, false);
                for (int i = 0; i < H; ++i) x[i] += m[i];
            }
        }, 8);
        prof_.moe += now_seconds() - tm;
        // the per-layer input: proj(gelu(gate(x)) * ple_l), normed, added
        if (P) {
            const auto gs = rows(pgate_, static_cast<size_t>(P), n);
            matmul(L.ple_gate, n, xs.data(), gs.data());
            par(n, [&](int64_t a, int64_t b) {
                for (int64_t t = a; t < b; ++t) {
                    float *g = gs[static_cast<size_t>(t)];
                    const float *pl = ple_.data() + static_cast<size_t>(t) * LP + static_cast<size_t>(l) * P;
                    if (exact_)
                        for (int i = 0; i < P; ++i) g[i] = gelu_tanh(g[i]) * pl[i];
                    else
                        for (int i = 0; i < P; ++i) g[i] = gelu_fast(g[i]) * pl[i];
                }
            }, 8);
            matmul(L.ple_proj, n, gs.data(), hs.data());
            par(n, [&](int64_t a, int64_t b) {
                for (int64_t t = a; t < b; ++t) {
                    float *x = xs[static_cast<size_t>(t)], *h = hs[static_cast<size_t>(t)];
                    rmsnorm(h, L.ple_norm, h, H, c.eps, false);
                    for (int i = 0; i < H; ++i) x[i] = (x[i] + h[i]) * L.scalar;
                }
            }, 8);
        } else {
            par(static_cast<int64_t>(n) * H, [&](int64_t a, int64_t b) {
                for (int64_t i = a; i < b; ++i) X[i] *= L.scalar;
            }, 4096);
        }
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
                trace(tr);
            }
    }
    if (keeps_hidden()) {
        // every token's state after the final norm (what its logits come from): the drafter's hidden rows
        float *hid = (hid_long_.empty() ? hid_ : hid_long_).data() + static_cast<size_t>(hid_row) * H;
        par(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t)
                rmsnorm(xs[static_cast<size_t>(t)], final_norm_, hid + static_cast<size_t>(t) * H, H, c.eps, false);
        }, 8);
    }
    if (logits) {
        const int first = all_logits ? 0 : n - 1;
        std::vector<const float *> in;
        std::vector<float *> out;
        for (int t = first; t < n; ++t) {
            float *h = hs[static_cast<size_t>(t - first)];
            rmsnorm(xs[static_cast<size_t>(t)], final_norm_, h, H, c.eps, false);
            in.push_back(h);
            out.push_back(logits + static_cast<size_t>(t - first) * c.vocab);
        }
        const double th = now_seconds();
        matmul(lm_head_, static_cast<int>(in.size()), in.data(), out.data());
        if (c.softcap > 0.0f) {
            const float cap = c.softcap, inv = 1.0f / c.softcap;
            const int64_t V = c.vocab;
            par(static_cast<int64_t>(out.size()) * V, [&](int64_t a, int64_t b) {
                for (int64_t i = a; i < b;) {   // a stretch of a row at a time, as the MLP's
                    const int64_t t = i / V, j0 = i % V, j1 = std::min(V, j0 + (b - i));
                    float *v = out[static_cast<size_t>(t)];
                    if (exact_)
                        for (int64_t j = j0; j < j1; ++j) v[j] = std::tanh(v[j] * inv) * cap;
                    else
                        for (int64_t j = j0; j < j1; ++j) v[j] = tanh_fast(v[j] * inv) * cap;
                    i += j1 - j0;
                }
            }, 1 << 14);
        }
        prof_.head += now_seconds() - th;
    }
    pos_ += n;
    return true;
}

void Model::gemma_attention(const GLayer &L, int n) {
    const auto &c = cfg_;
    const int H = c.hidden, nh = c.n_heads, nkv = c.n_kv_heads, hd = L.hd, half = hd / 2;
    const size_t qw = static_cast<size_t>(nh) * hd, kw = static_cast<size_t>(nkv) * hd;
    const auto hs = rows(h_, static_cast<size_t>(H), n);
    const auto qs = rows(qg_, qw, n);
    const auto ks = rows(kbuf_, kw, n);
    const auto vs = rows(vbuf_, kw, n);
    const QAct *const *qh = npu_takes(L.q, n) || gpu_takes(L.q, n) ? nullptr : quantize_h(hs, H);
    if (L.own_kv) {
        const QView *mats[3] = {&L.q, &L.k, &L.v};
        float *const *outs[3] = {qs.data(), ks.data(), vs.data()};
        matmul_many(mats, 3, n, hs.data(), outs, qh);
    } else {
        matmul(L.q, n, hs.data(), qs.data(), qh);
    }
    // per token: the heads normed and turned (rotate-half pairs (i, i + hd/2), the first L.pairs of them)
    par(n, [&](int64_t a, int64_t b) {
        thread_local std::vector<float> cs, sn;
        cs.resize(static_cast<size_t>(L.pairs));
        sn.resize(static_cast<size_t>(L.pairs));
        const auto rope = [&](float *x) {
            for (int i = 0; i < L.pairs; ++i) {
                const float x1 = x[i], x2 = x[i + half];
                x[i] = x1 * cs[static_cast<size_t>(i)] - x2 * sn[static_cast<size_t>(i)];
                x[i + half] = x2 * cs[static_cast<size_t>(i)] + x1 * sn[static_cast<size_t>(i)];
            }
        };
        for (int64_t t = a; t < b; ++t) {
            const float p = static_cast<float>(apos_ + t);
            for (int i = 0; i < L.pairs; ++i) {
                const float f = p * L.inv_freq[static_cast<size_t>(i)];
                cs[static_cast<size_t>(i)] = std::cos(f);
                sn[static_cast<size_t>(i)] = std::sin(f);
            }
            for (int h = 0; h < nh; ++h) {
                float *q = qs[static_cast<size_t>(t)] + static_cast<size_t>(h) * hd;
                rmsnorm(q, L.q_norm, q, hd, c.eps, false);
                rope(q);
            }
            if (!L.own_kv) continue;
            for (int h = 0; h < nkv; ++h) {
                float *k = ks[static_cast<size_t>(t)] + static_cast<size_t>(h) * hd;
                float *v = vs[static_cast<size_t>(t)] + static_cast<size_t>(h) * hd;
                rmsnorm(k, L.k_norm, k, hd, c.eps, false);
                rope(k);
                rmsnorm(v, nullptr, v, hd, c.eps, false);
            }
        }
    }, 4);
    const double tc = now_seconds();
    if (exact_) gemma_attend<float>(L, n);
    else if (kv_i8_) gemma_attend<int8_t>(L, n);
    else gemma_attend<_Float16>(L, n);
    prof_.attn_core += now_seconds() - tc;
    const auto cr = rows(core_, qw, n);
    const auto ms = rows(mix_, static_cast<size_t>(H), n);
    matmul(L.o, n, cr.data(), ms.data());
}

// The batch's K and V into the cache (a layer with its own), then each token attends over its positions: every one
// up to its own on a full layer, the last `window` on a sliding one. Blocks of attn_block_ positions from position 0
// (model.cpp's attend: a token's bits depend neither on its batch nor on the threads), joined in order.
template <typename T>
void Model::gemma_attend(const GLayer &L, int n) {
    const auto &c = cfg_;
    const int nh = c.n_heads, nkv = c.n_kv_heads, hd = L.hd;
    const size_t kw = static_cast<size_t>(nkv) * hd, qw = static_cast<size_t>(nh) * hd;
    if (L.own_kv)
        for (int t = 0; t < n; ++t) {
            T *k = reinterpret_cast<T *>(kv_row_bytes(apos_ + t, L.slot, false)), *v = reinterpret_cast<T *>(kv_row_bytes(apos_ + t, L.slot, true));
            const float *kb = kbuf_.data() + static_cast<size_t>(t) * kw, *vb = vbuf_.data() + static_cast<size_t>(t) * kw;
            if constexpr (std::is_same_v<T, int8_t>) {
                float *ksc = kv_scale(apos_ + t, L.slot, false), *vsc = kv_scale(apos_ + t, L.slot, true);
                for (int h = 0; h < nkv; ++h) {
                    const size_t o = static_cast<size_t>(h) * hd;
                    quant_row(kb + o, hd, k + o, ksc[h]);
                    quant_row(vb + o, hd, v + o, vsc[h]);
                }
            } else {
                for (size_t i = 0; i < kw; ++i) {
                    k[i] = static_cast<T>(kb[i]);
                    v[i] = static_cast<T>(vb[i]);
                }
            }
        }
    const int group = nh / nkv;
    // the int8 KV cache's scores on SDOT: the queries as int8 too, a row per token and head (amax / 127 its scale,
    // as the KV rows'), once a layer
    qi8_on_ = false;
#if defined(GEODE_GEMMA_DOT)
    if constexpr (std::is_same_v<T, int8_t>)
        if (!exact_ && hd % 16 == 0 && (group == 1 || group % 4 == 0)) {
            const size_t rows_n = static_cast<size_t>(n) * nh;
            if (qi8_.size() < rows_n * hd) qi8_.resize(rows_n * hd);
            if (qi8_s_.size() < rows_n) qi8_s_.resize(rows_n);
            par(static_cast<int64_t>(rows_n), [&](int64_t a, int64_t b) {
                for (int64_t r = a; r < b; ++r)
                    quant_row(qg_.data() + static_cast<size_t>(r) * hd, hd, qi8_.data() + static_cast<size_t>(r) * hd,
                              qi8_s_[static_cast<size_t>(r)]);
            }, 64);
            qi8_on_ = true;
        }
#endif
#if defined(GEODE_GEMMA_MM)
    if constexpr (std::is_same_v<T, int8_t>)
        if (qi8_on_ && n >= kMmMinTokens && gemma_attend_mm(L, n)) return;
#endif
    const int nb = (apos_ + n + attn_block_ - 1) / attn_block_;
    const size_t part = static_cast<size_t>(hd) + 2;
    const auto lo_of = [&](int t) { return L.full ? 0 : std::max(0, apos_ + t - c.window + 1); };
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
                if (b * attn_block_ > apos_ + t || (b + 1) * attn_block_ <= lo_of(t)) continue;   // outside its span
                int g = 0;
                for (; g + 8 <= group; g += 8) gemma_heads<T, 8>(L, kh, kh * group + g, t, b, nb);
                for (; g + 4 <= group; g += 4) gemma_heads<T, 4>(L, kh, kh * group + g, t, b, nb);
                for (; g < group; ++g) gemma_heads<T, 1>(L, kh, kh * group + g, t, b, nb);
            }
        });
        par(static_cast<int64_t>(tn) * nh, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int t = t0 + static_cast<int>(i / nh), h = static_cast<int>(i % nh);
                const int b0 = lo_of(t) / attn_block_, b1 = (apos_ + t) / attn_block_;
                const float *pp = attn_parts_.data() + (static_cast<size_t>(t - t0) * nh + h) * nb * part;
                float *o = core_.data() + static_cast<size_t>(t) * qw + static_cast<size_t>(h) * hd;
                float sum;
                if (b0 == b1) {
                    const float *q = pp + static_cast<size_t>(b0) * part;
                    std::memcpy(o, q, sizeof(float) * static_cast<size_t>(hd));
                    sum = q[hd + 1];
                } else {
                    float mx = -INFINITY;
                    for (int b = b0; b <= b1; ++b) mx = std::max(mx, pp[static_cast<size_t>(b) * part + hd]);
                    std::fill(o, o + hd, 0.0f);
                    sum = 0.0f;
                    for (int b = b0; b <= b1; ++b) {
                        const float *q = pp + static_cast<size_t>(b) * part;
                        const float f = std::exp(q[hd] - mx);
                        sum += q[hd + 1] * f;
                        for (int j = 0; j < hd; ++j) o[j] += q[j] * f;
                    }
                }
                const float inv = 1.0f / sum;
                for (int j = 0; j < hd; ++j) o[j] *= inv;
            }
        });
    }
}

template <typename T, int G>
void Model::gemma_heads(const GLayer &L, int kh, int h0, int t, int b, int nb) {
    // token t's query heads h0..h0+G-1 (sharing KV head kh) over block b of its positions: the scores q.k (scale 1:
    // q and k are normed), their maximum, the weights and their sum, the weighted values - gemma_attend joins blocks
    const auto &c = cfg_;
    const int hd = L.hd, layer = L.slot;
    const size_t qw = static_cast<size_t>(c.n_heads) * hd;
    const int lo = L.full ? 0 : std::max(0, apos_ + t - c.window + 1);
    const int p0 = std::max(b * attn_block_, lo), p1 = std::min(apos_ + t + 1, (b + 1) * attn_block_), np = p1 - p0;
    const auto chunks = [&](auto &&f) {
        for (int s0 = p0; s0 < p1;) {
            const int s1 = std::min(p1, (s0 / kv_chunk_ + 1) * kv_chunk_);
            f(s0, s1);
            s0 = s1;
        }
    };
    thread_local std::vector<float> scores;
    if (scores.size() < static_cast<size_t>(G) * np) scores.resize(static_cast<size_t>(G) * np);
    const float *q[G];
    float *sc[G], mx[G], sums[G];
    for (int g = 0; g < G; ++g) {
        q[g] = qg_.data() + static_cast<size_t>(t) * qw + static_cast<size_t>(h0 + g) * hd;
        sc[g] = scores.data() + static_cast<size_t>(g) * np;
        mx[g] = -INFINITY;
    }
    bool scored = false;
#if defined(GEODE_GEMMA_DOT)
    if constexpr (std::is_same_v<T, int8_t> && (G == 1 || G % 4 == 0))
        if (qi8_on_) {
            // SDOT: the int8 queries (gemma_attend) against the int8 keys, then the two scales
            const int8_t *qi[G];
            float qs[G];
            for (int g = 0; g < G; ++g) {
                const size_t r = static_cast<size_t>(t) * c.n_heads + static_cast<size_t>(h0 + g);
                qi[g] = qi8_.data() + r * hd;
                qs[g] = qi8_s_[r];
            }
            thread_local std::vector<int32_t> dots;
            if (dots.size() < static_cast<size_t>(G) * np) dots.resize(static_cast<size_t>(G) * np);
            chunks([&](int s0, int s1) {
                const int8_t *k = reinterpret_cast<int8_t *>(kv_row_bytes(s0, layer, false)) + static_cast<size_t>(kh) * hd;
                const float *ksc = kv_scale(s0, layer, false) + kh;
                const int m = s1 - s0, p0r = s0 - p0;
                qk_dots_i8<G>(qi, k, kv_row_, m, hd, dots.data(), m);
                for (int g = 0; g < G; ++g) {
                    const int32_t *d = dots.data() + static_cast<size_t>(g) * m;
                    float *s = sc[g] + p0r;
                    float mg = mx[g];
                    for (int r = 0; r < m; ++r) {
                        const float sv = static_cast<float>(d[r]) * (qs[g] * ksc[static_cast<size_t>(r) * kv_heads_]);
                        s[r] = sv;
                        mg = std::max(mg, sv);
                    }
                    mx[g] = mg;
                }
            });
            scored = true;
        }
#endif
    if (!scored) chunks([&](int s0, int s1) {
        const T *k = reinterpret_cast<T *>(kv_row_bytes(s0, layer, false)) + static_cast<size_t>(kh) * hd;
        const float *ksc = nullptr;
        if constexpr (std::is_same_v<T, int8_t>) ksc = kv_scale(s0, layer, false) + kh;
        for (int p = s0 - p0, pe = s1 - p0, r = 0; p < pe; ++p, ++r, k += kv_row_) {
            float dots[G];
#if defined(GEODE_GEMMA_NEON)
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
            const float f = ksc ? ksc[static_cast<size_t>(r) * kv_heads_] : 1.0f;
            for (int g = 0; g < G; ++g) {
                const float sv = dots[g] * f;
                sc[g][p] = sv;
                mx[g] = std::max(mx[g], sv);
            }
        }
    });
    for (int g = 0; g < G; ++g) {
        float *w = sc[g];
        const float m = mx[g];
        if (exact_) {
            float sum = 0.0f;
            for (int p = 0; p < np; ++p) {
                w[p] = std::exp(w[p] - m);
                sum += w[p];
            }
            sums[g] = sum;
            continue;
        }
        // fast_exp (a prompt's attention takes ~0.3 G of them for 2048 tokens: std::exp's calls were a good part of
        // it), the sum in four lanes: both loops vectorize
        for (int p = 0; p < np; ++p) w[p] = fast_exp(w[p] - m);
        float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int p = 0;
        for (; p + 4 <= np; p += 4)
            for (int j = 0; j < 4; ++j) s4[j] += w[p + j];
        for (; p < np; ++p) s4[0] += w[p];
        sums[g] = (s4[0] + s4[1]) + (s4[2] + s4[3]);
    }
    if constexpr (std::is_same_v<T, int8_t>)
        chunks([&](int s0, int s1) {
            const float *vsc = kv_scale(s0, layer, true) + kh;
            for (int p = s0 - p0, pe = s1 - p0, r = 0; p < pe; ++p, ++r) {
                const float f = vsc[static_cast<size_t>(r) * kv_heads_];
                for (int g = 0; g < G; ++g) sc[g][p] *= f;
            }
        });
    const size_t part = static_cast<size_t>(hd) + 2;
    float *dst[G];
    for (int g = 0; g < G; ++g) {
        dst[g] = attn_parts_.data() + ((static_cast<size_t>(t - attn_t0_) * c.n_heads + (h0 + g)) * nb + b) * part;
        dst[g][hd] = mx[g];
        dst[g][hd + 1] = sums[g];
    }
    for (int i = 0; i < hd; i += 8) {
        float acc[G][8];
#if defined(GEODE_GEMMA_NEON)
        float32x4_t lo[G], hi[G];
        for (int g = 0; g < G; ++g) lo[g] = hi[g] = vdupq_n_f32(0.0f);
        chunks([&](int s0, int s1) {
            const T *v = reinterpret_cast<T *>(kv_row_bytes(s0, layer, true)) + static_cast<size_t>(kh) * hd + i;
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
            const T *v = reinterpret_cast<T *>(kv_row_bytes(s0, layer, true)) + static_cast<size_t>(kh) * hd + i;
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


#if defined(GEODE_GEMMA_MM)
// A prompt's chunk over the int8 KV cache on i8mm: the scores on SMMLA (qk_mm8: the integers SDOT gives), the
// weighted values on USMMLA (pv_mm8), each block's weights as u8 against its largest (255: ~0.2% of it each, rounded
// to nearest), the blocks joined as they come - an online softmax, in block order, so a token's bits depend neither
// on the threads nor on how the blocks are grouped. The keys by position pairs and the values transposed (and the
// blocks' scales in a row) are built once a layer for the octets of positions the chunk reads, mm_build_bytes() of each
// at a time; a layer sharing another's KV reuses that layer's (Model::MmOps).
bool Model::gemma_attend_mm(const GLayer &L, int n) {
    const auto &c = cfg_;
    const int nh = c.n_heads, nkv = c.n_kv_heads, hd = L.hd, group = nh / nkv, B = attn_block_;
    if (group % 8 || hd % 16 || hd > 1024 || B % 8 || B > 8192 || !i8mm_here()) return false;
    const int nc = hd / 8, octs = B / 8, end = apos_ + n;
    const size_t qw = static_cast<size_t>(nh) * hd, blk = static_cast<size_t>(B) * hd;
    const size_t rows_n = static_cast<size_t>(n) * nh;
    const auto lo_of = [&](int t) { return L.full ? 0 : std::max(0, apos_ + t - c.window + 1); };
    // the queries by head pairs, [pair][hd / 8][16]: a pair where qi8_ has its first head's row
    if (qp_.size() < rows_n * hd) qp_.resize(rows_n * hd);
    par(static_cast<int64_t>(rows_n / 2), [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; ++r) {
            const int8_t *q0 = qi8_.data() + static_cast<size_t>(2 * r) * hd, *q1 = q0 + hd;
            int8_t *o = qp_.data() + static_cast<size_t>(2 * r) * hd;
            for (int i = 0; i < hd; i += 16) {
                const int64x2_t x = vreinterpretq_s64_s8(vld1q_s8(q0 + i)), y = vreinterpretq_s64_s8(vld1q_s8(q1 + i));
                vst1q_s8(o + 2 * i, vreinterpretq_s8_s64(vzip1q_s64(x, y)));
                vst1q_s8(o + 2 * i + 16, vreinterpretq_s8_s64(vzip2q_s64(x, y)));
            }
        }
    }, 64);
    if (run_m_.size() < rows_n) {
        run_m_.resize(rows_n);
        run_s_.resize(rows_n);
    }
    std::fill(run_m_.begin(), run_m_.begin() + static_cast<std::ptrdiff_t>(rows_n), -INFINITY);
    const int lo0 = lo_of(0), b_lo = lo0 / B, b_hi = (end - 1) / B;
    const int gn = std::max(1, static_cast<int>(mm_build_bytes() / (static_cast<size_t>(nkv) * blk)));
    static const int8_t zeros[1024] = {};
    for (int g0 = b_lo; g0 <= b_hi; g0 += gn) {
        const int g1 = std::min(b_hi, g0 + gn - 1), nbk = g1 - g0 + 1;
        const bool whole = g0 == b_lo && g1 == b_hi;   // one group: the build can be kept for the layers sharing it
        MmOps *ops = nullptr;
        if (whole && !L.own_kv)
            for (MmOps &m : mm_ops_)
                if (m.slot == L.slot && m.lo == lo0 && m.end == end && m.hd == hd && m.b0 == g0 && m.nb == nbk) ops = &m;
        if (ops) {
            ops->used = ++mm_tick_;
        } else {
            // a layer with its own KV wrote its slot: its entry, or the one used longest ago
            ops = &mm_ops_[0];
            for (MmOps &m : mm_ops_) {
                if (m.slot == L.slot) {
                    ops = &m;
                    break;
                }
                if (m.used < ops->used) ops = &m;
            }
            const size_t need = static_cast<size_t>(nkv) * nbk * blk, nsc = static_cast<size_t>(nkv) * nbk * B;
            if (ops->kp.size() < need) {
                ops->kp.resize(need);
                ops->vt.resize(need);
            }
            if (ops->ks.size() < nsc) {
                ops->ks.resize(nsc);
                ops->vs.resize(nsc);
            }
            ops->slot = whole ? L.slot : -1;
            ops->lo = lo0;
            ops->end = end;
            ops->hd = hd;
            ops->b0 = g0;
            ops->nb = nbk;
            ops->used = ++mm_tick_;
            int8_t *kpb = ops->kp.data(), *vtb = ops->vt.data();
            float *ksb = ops->ks.data(), *vsb = ops->vs.data();
            // an octet of positions an item: its keys by pairs ([pair][hd / 8][16]), its values transposed ([hd / 2]
            // [octs][16]: an 8 x 16 byte transpose in three rounds of zips), its scales; positions before the window
            // or not written yet as zeros, octets no token of the chunk reads left as they are
            par(static_cast<int64_t>(nkv) * nbk * octs, [&](int64_t a, int64_t b) {
                for (int64_t i = a; i < b; ++i) {
                    const int o = static_cast<int>(i % octs), bi = static_cast<int>(i / octs % nbk),
                              kh = static_cast<int>(i / octs / nbk);
                    const int p0 = (g0 + bi) * B + 8 * o;
                    if (p0 + 8 <= lo0 || p0 >= end) continue;
                    const size_t base = (static_cast<size_t>(kh) * nbk + bi) * blk;
                    int8_t *kd = kpb + base + static_cast<size_t>(o) * 4 * nc * 16;   // its pairs 4o..4o+3
                    uint8_t *vd = reinterpret_cast<uint8_t *>(vtb + base) + static_cast<size_t>(o) * 16;
                    float *kq = ksb + (static_cast<size_t>(kh) * nbk + bi) * B + 8 * o, *vq = vsb + (kq - ksb);
                    const int8_t *kr[8], *vr[8];
                    for (int r = 0; r < 8; ++r) {
                        const int p = p0 + r;
                        const bool in = p >= lo0 && p < end;
                        kr[r] = in ? reinterpret_cast<const int8_t *>(kv_row_bytes(p, L.slot, false)) +
                                         static_cast<size_t>(kh) * hd
                                   : zeros;
                        vr[r] = in ? reinterpret_cast<const int8_t *>(kv_row_bytes(p, L.slot, true)) +
                                         static_cast<size_t>(kh) * hd
                                   : zeros;
                        kq[r] = in ? kv_scale(p, L.slot, false)[kh] : 0.0f;
                        vq[r] = in ? kv_scale(p, L.slot, true)[kh] : 0.0f;
                    }
                    for (int d = 0; d < hd; d += 16) {
                        for (int r = 0; r < 8; r += 2) {
                            const int64x2_t x = vreinterpretq_s64_s8(vld1q_s8(kr[r] + d)),
                                            y = vreinterpretq_s64_s8(vld1q_s8(kr[r + 1] + d));
                            int8_t *dst = kd + static_cast<size_t>(r / 2) * nc * 16 + static_cast<size_t>(d) * 2;
                            vst1q_s8(dst, vreinterpretq_s8_s64(vzip1q_s64(x, y)));
                            vst1q_s8(dst + 16, vreinterpretq_s8_s64(vzip2q_s64(x, y)));
                        }
                        uint8x16_t rw[8];
                        for (int r = 0; r < 8; ++r) rw[r] = vreinterpretq_u8_s8(vld1q_s8(vr[r] + d));
                        uint16x8_t a8[8];   // bytes: (row 2k, row 2k + 1) a value
                        for (int k = 0; k < 4; ++k) {
                            a8[2 * k] = vreinterpretq_u16_u8(vzip1q_u8(rw[2 * k], rw[2 * k + 1]));
                            a8[2 * k + 1] = vreinterpretq_u16_u8(vzip2q_u8(rw[2 * k], rw[2 * k + 1]));
                        }
                        uint32x4_t b8[8];   // 16 bits: four rows a value - b8[k] values 4k..4k+3 of rows 0-3, b8[4 + k] of rows 4-7
                        for (int h = 0; h < 2; ++h) {
                            const uint16x8_t *s = a8 + 4 * h;   // rows 4h..4h+3: s[0], s[2] values 0-7, s[1], s[3] 8-15
                            b8[4 * h + 0] = vreinterpretq_u32_u16(vzip1q_u16(s[0], s[2]));
                            b8[4 * h + 1] = vreinterpretq_u32_u16(vzip2q_u16(s[0], s[2]));
                            b8[4 * h + 2] = vreinterpretq_u32_u16(vzip1q_u16(s[1], s[3]));
                            b8[4 * h + 3] = vreinterpretq_u32_u16(vzip2q_u16(s[1], s[3]));
                        }
                        for (int k = 0; k < 4; ++k) {   // 32 bits: the eight rows of values 4k + 2x, 4k + 2x + 1
                            uint8_t *dst = vd + (static_cast<size_t>(d / 2 + 2 * k) * octs) * 16;
                            vst1q_u8(dst, vreinterpretq_u8_u32(vzip1q_u32(b8[k], b8[4 + k])));
                            vst1q_u8(dst + static_cast<size_t>(octs) * 16, vreinterpretq_u8_u32(vzip2q_u32(b8[k], b8[4 + k])));
                        }
                    }
                }
            }, 4);
        }
        const int8_t *kpb = ops->kp.data(), *vtb = ops->vt.data();
        const float *ksb = ops->ks.data(), *vsb = ops->vs.data();
        par(static_cast<int64_t>(n) * nkv, [&](int64_t a, int64_t b) {
            thread_local std::vector<int32_t> dots, acc;
            thread_local std::vector<float> sc;
            thread_local std::vector<uint8_t> wq, wp;
            const size_t eb = static_cast<size_t>(8) * B;
            if (dots.size() < eb) dots.resize(eb);
            if (sc.size() < eb) sc.resize(eb);
            if (wq.size() < eb) wq.resize(eb);
            if (wp.size() < eb) wp.resize(eb);
            if (acc.size() < static_cast<size_t>(8) * hd) acc.resize(static_cast<size_t>(8) * hd);
            for (int64_t i = a; i < b; ++i) {
                const int t = static_cast<int>(i / nkv), kh = static_cast<int>(i % nkv);
                const int lo = lo_of(t), hi = apos_ + t + 1;
                for (int bk = std::max(g0, lo / B), be = std::min(g1, (hi - 1) / B); bk <= be; ++bk) {
                    const int bs = bk * B, p0 = std::max(bs, lo), p1 = std::min(hi, bs + B), np = p1 - p0;
                    const size_t bb = static_cast<size_t>(kh) * nbk + static_cast<size_t>(bk - g0);
                    const float *ks = ksb + bb * B + (p0 - bs), *vs = vsb + bb * B + (p0 - bs);
                    // the scores over whole position pairs, four at a time; the weights over whole octets
                    const int j0 = ((p0 - bs) / 2) & ~3, j1 = std::min(B / 2, (((p1 - bs + 1) / 2) + 3) & ~3);
                    const int ld = 2 * (j1 - j0), off = p0 - (bs + 2 * j0);
                    const int o0 = (p0 - bs) / 8, o1 = (p1 - bs + 7) / 8, no = o1 - o0, ow = p0 - (bs + 8 * o0);
                    for (int hg = 0; hg < group; hg += 8) {
                        const int h0 = kh * group + hg;
                        const size_t r0 = static_cast<size_t>(t) * nh + static_cast<size_t>(h0);
                        qk_mm8(qp_.data() + r0 * hd, kpb + bb * blk, j0, j1, hd, dots.data(), ld);
                        float mx[8], sum[8], ws[8];
                        for (int g = 0; g < 8; ++g) {
                            const int32_t *d = dots.data() + static_cast<size_t>(g) * ld + off;
                            float *s = sc.data() + static_cast<size_t>(g) * B;
                            const float qs = qi8_s_[r0 + static_cast<size_t>(g)];
                            const float32x4_t qv = vdupq_n_f32(qs);
                            float32x4_t mv = vdupq_n_f32(-INFINITY);
                            int p = 0;
                            for (; p + 4 <= np; p += 4) {
                                const float32x4_t v = vmulq_f32(vcvtq_f32_s32(vld1q_s32(d + p)), vmulq_f32(qv, vld1q_f32(ks + p)));
                                vst1q_f32(s + p, v);
                                mv = vmaxq_f32(mv, v);
                            }
                            float m = vmaxvq_f32(mv);
                            for (; p < np; ++p) {
                                const float v = static_cast<float>(d[p]) * (qs * ks[p]);
                                s[p] = v;
                                m = std::max(m, v);
                            }
                            for (p = 0; p < np; ++p) s[p] = fast_exp(s[p] - m);
                            // their sum; the weights times the values' scales, and the largest
                            float32x4_t sv = vdupq_n_f32(0.0f), wv = vdupq_n_f32(0.0f);
                            for (p = 0; p + 4 <= np; p += 4) {
                                const float32x4_t e = vld1q_f32(s + p), w = vmulq_f32(e, vld1q_f32(vs + p));
                                sv = vaddq_f32(sv, e);
                                vst1q_f32(s + p, w);
                                wv = vmaxq_f32(wv, w);
                            }
                            float s4[4];
                            vst1q_f32(s4, sv);
                            float wm = vmaxvq_f32(wv);
                            for (; p < np; ++p) {
                                s4[0] += s[p];
                                s[p] *= vs[p];
                                wm = std::max(wm, s[p]);
                            }
                            // as u8 against the largest
                            uint8_t *q = wq.data() + static_cast<size_t>(g) * B;
                            std::memset(q, 0, static_cast<size_t>(no) * 8);
                            const float inv = wm > 0.0f ? 255.0f / wm : 0.0f;
                            const float32x4_t iv = vdupq_n_f32(inv);
                            for (p = 0; p + 8 <= np; p += 8) {
                                const uint32x4_t u0 = vcvtnq_u32_f32(vmulq_f32(vld1q_f32(s + p), iv)),
                                                 u1 = vcvtnq_u32_f32(vmulq_f32(vld1q_f32(s + p + 4), iv));
                                vst1_u8(q + ow + p, vqmovn_u16(vcombine_u16(vqmovn_u32(u0), vqmovn_u32(u1))));
                            }
                            for (; p < np; ++p) q[ow + p] = static_cast<uint8_t>(std::min(255.0f, std::nearbyint(s[p] * inv)));
                            mx[g] = m;
                            sum[g] = (s4[0] + s4[1]) + (s4[2] + s4[3]);
                            ws[g] = wm / 255.0f;
                        }
                        for (int hp = 0; hp < 4; ++hp)   // by head pairs, [pair][no][16]
                            for (int o = 0; o < no; ++o) {
                                uint8_t *d = wp.data() + (static_cast<size_t>(hp) * no + o) * 16;
                                std::memcpy(d, wq.data() + static_cast<size_t>(2 * hp) * B + 8 * o, 8);
                                std::memcpy(d + 8, wq.data() + static_cast<size_t>(2 * hp + 1) * B + 8 * o, 8);
                            }
                        pv_mm8(wp.data(), vtb + bb * blk, o0, no, octs, hd, acc.data());
                        for (int g = 0; g < 8; ++g) {   // into the token's running softmax (its first block: as it is)
                            const size_t r = r0 + static_cast<size_t>(g);
                            float *o = core_.data() + static_cast<size_t>(t) * qw + static_cast<size_t>(h0 + g) * hd;
                            const int32_t *v = acc.data() + static_cast<size_t>(g) * hd;
                            if (run_m_[r] == -INFINITY) {
                                run_m_[r] = mx[g];
                                run_s_[r] = sum[g];
                                for (int d = 0; d < hd; ++d) o[d] = static_cast<float>(v[d]) * ws[g];
                                continue;
                            }
                            const float M = std::max(run_m_[r], mx[g]);
                            const float fo = std::exp(run_m_[r] - M), fb = std::exp(mx[g] - M);
                            run_s_[r] = run_s_[r] * fo + sum[g] * fb;
                            run_m_[r] = M;
                            const float f = ws[g] * fb;
                            for (int d = 0; d < hd; ++d) o[d] = o[d] * fo + static_cast<float>(v[d]) * f;
                        }
                    }
                }
            }
        });
    }
    par(static_cast<int64_t>(rows_n), [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; ++r) {
            float *o = core_.data() + static_cast<size_t>(r) * hd;
            const float inv = 1.0f / run_s_[static_cast<size_t>(r)];
            for (int d = 0; d < hd; ++d) o[d] *= inv;
        }
    }, 64);
    return true;
}
#endif

}  // namespace geode
