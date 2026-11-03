// pack.hpp - a geode-pack opened for the engine: the manifest, dense.bin in memory, the files read on demand
// (tools/geodepack.py documents the format).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "kernels.hpp"
#include "platform.hpp"

namespace geode {

struct Tensor {
    DType type = DType::Unknown;
    std::vector<int64_t> shape;
    int64_t rows = 0, cols = 0;      // shape[0] and the product of the rest
    int file = -1;                   // index into Pack::files
    uint64_t offset = 0;
    const uint8_t *data = nullptr;   // set when the tensor lives in dense.bin (in memory)
    float global = 1.0f;

    QView view() const { return QView{type, rows, cols, row_bytes(type, cols), data, global}; }
    const float *f32() const { return reinterpret_cast<const float *>(data); }
    const uint16_t *bf16() const { return reinterpret_cast<const uint16_t *>(data); }
};

struct ModelConfig {
    int hidden = 0, n_layers = 0, vocab = 0;
    float eps = 1e-6f;
    // full attention
    int n_heads = 0, n_kv_heads = 0, head_dim = 0, rope_dim = 0;
    float rope_theta = 10000.0f;
    // Gated DeltaNet
    int gdn_k_heads = 0, gdn_v_heads = 0, gdn_k_dim = 0, gdn_v_dim = 0, conv_kernel = 0;
    // MoE. A dense model (arch "qwen35", Qwen3.5-2B): no routed experts, its MLP read as an ungated shared expert
    // of intermediate_size (shared_ff)
    int n_experts = 0, top_k = 0, moe_ff = 0, shared_ff = 0;
    bool dense = false;
    std::vector<bool> attention;     // per layer: full attention (else Gated DeltaNet)
    // the MTP layer (tools/pack_mtp.py): one full-attention layer that drafts the token after next; its experts are
    // expert layer n_layers
    int mtp_layers = 0;
    // Gemma 4 (arch "gemma4", tools/pack_gemma4.py; dense, no experts): per layer the layer whose K / V it attends
    // over (itself, or for a KV-shared layer the last non-shared one of its type), its MLP's width, its head dim, the
    // rotary pairs it turns and their theta; attention[l] is a full layer (else a sliding one). head_dim is the
    // largest head dim (the KV cache's rows), shared_ff the widest MLP. The sliding window, the per-layer inputs'
    // width (0: none), the embedding's scale, the logits' soft cap (0: none); the pad token (a media token's per-layer
    // input is its row).
    bool gemma = false;
    std::vector<int> kv_layer, ffn, head_dims, rot_pairs;
    std::vector<float> rope_thetas;
    int window = 0, ple_dim = 0, pad_id = 0;
    float embed_scale = 1.0f, softcap = 0.0f;
};

struct ExpertLayer {
    int file = -1;
    uint64_t offset = 0, stride = 0;   // uniform format: expert e at offset + e * stride
    DType type = DType::Unknown;
    std::vector<uint64_t> offsets;     // mixed formats: per expert
    std::vector<DType> types;
};

class Pack {
public:
    // Reads geode.json, checks every file, loads dense.bin into memory, opens the rest.
    bool open(const std::string &dir, std::string &err);

    const ModelConfig &config() const { return config_; }
    const json::Value &manifest() const { return manifest_; }
    const Tensor *find(const std::string &name) const;
    uint64_t dense_bytes() const { return dense_size_; }

    // Where expert e of a layer is, its format and the bytes of its block (gate_up rows, then down rows).
    void expert(int layer, int e, int &file, uint64_t &offset, DType &type) const;
    size_t expert_block_bytes(DType type) const;
    size_t max_expert_block_bytes() const;
    const RandomFile &file(int index) const { return *files_[static_cast<size_t>(index)]; }
    std::string path(int index) const;
    const std::string &dir() const { return dir_; }
    int file_count() const { return static_cast<int>(file_names_.size()); }

    // The embedding row of a token, dequantized (read from flash: the table stays there).
    bool embedding(int token, float *out) const;
    // Row r of a quantized tensor, dequantized, read from its file (a table that stays on flash: Gemma's per-layer
    // embeddings)
    bool read_row(const Tensor &t, int64_t r, float *out) const;
    // q4 experts stored in tile8 (kernels.hpp; geode.json formats.q4.layout, tools/q4_tile8.py)
    bool experts_tile8() const { return experts_tile8_; }

private:
    int file_index(const std::string &name, std::string &err);

    std::string dir_;
    json::Value manifest_;
    ModelConfig config_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<ExpertLayer> experts_;
    std::vector<std::string> file_names_;
    std::vector<std::unique_ptr<RandomFile>> files_;
    AlignedBuffer dense_;
    uint64_t dense_size_ = 0;
    const Tensor *embed_ = nullptr;
    bool experts_tile8_ = false;
};

}  // namespace geode
