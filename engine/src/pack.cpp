// pack.cpp - opening a geode-pack: manifest checks, the tensor table, dense.bin into memory, the expert layout.
#include "pack.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace geode {

namespace fs = std::filesystem;

namespace {

bool read_text(const fs::path &p, std::string &out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

int geti(const json::Value &v, const char *key, int fallback = 0) {
    return static_cast<int>(v[key].i64(fallback));
}

}  // namespace

int Pack::file_index(const std::string &name, std::string &err) {
    for (size_t i = 0; i < file_names_.size(); ++i)
        if (file_names_[i] == name) return static_cast<int>(i);
    err = "geode.json names a file it does not list: " + name;
    return -1;
}

bool Pack::open(const std::string &dir, std::string &err) {
    dir_ = dir;
    std::string text;
    if (!read_text(fs::path(dir) / "geode.json", text)) {
        err = "no geode.json in " + dir;
        return false;
    }
    std::string perr;
    if (!json::parse(text, manifest_, perr)) {
        err = "geode.json: " + perr;
        return false;
    }
    const auto &m = manifest_;
    if (m["format"].str() != "geode-pack" || m["version"].i64() != 1) {
        err = "not a geode-pack v1 (format '" + m["format"].str() + "')";
        return false;
    }
    const bool gemma = m["arch"].str() == "gemma4";   // Gemma 4 E2B / E4B (tools/pack_gemma4.py)
    if (m["arch"].str() != "qwen35moe" && m["arch"].str() != "qwen35" && !gemma) {   // qwen35: the dense models (Qwen3.5-2B)
        err = "unknown architecture '" + m["arch"].str() + "'";
        return false;
    }

    // the files: present, the right size, opened; an optional one (vision.bin) may be missing, then its tensors
    // are not there (a pack installed before it was added)
    const auto &files = m["files"];
    std::vector<std::string> missing;
    for (size_t i = 0; i < files.size(); ++i) {
        const auto &f = files.at(i);
        const std::string name = f["name"].str();
        if (f["optional"].kind == json::Value::Kind::Bool && f["optional"].boolean && !fs::exists(fs::path(dir) / name)) {
            missing.push_back(name);
            continue;
        }
        auto rf = std::make_unique<RandomFile>();
        if (!rf->open((fs::path(dir) / name).string(), err)) return false;
        if (static_cast<int64_t>(rf->size()) != f["size"].i64()) {
            err = name + ": " + std::to_string(rf->size()) + " bytes, expected " + std::to_string(f["size"].i64());
            return false;
        }
        file_names_.push_back(name);
        files_.push_back(std::move(rf));
    }

    // the geometry
    const auto &c = m["config"];
    auto &k = config_;
    k.hidden = geti(c, "hidden_size");
    k.n_layers = geti(c, "num_hidden_layers");
    k.vocab = geti(c, "vocab_size");
    k.eps = static_cast<float>(c["rms_norm_eps"].f64(1e-6));
    k.n_heads = geti(c, "num_attention_heads");
    k.n_kv_heads = geti(c, "num_key_value_heads");
    k.head_dim = geti(c, "head_dim");
    const auto &rope = c["rope_parameters"];
    k.rope_theta = static_cast<float>(rope["rope_theta"].f64(10000.0));
    k.rope_dim = static_cast<int>(k.head_dim * rope["partial_rotary_factor"].f64(1.0));
    k.gdn_k_heads = geti(c, "linear_num_key_heads");
    k.gdn_v_heads = geti(c, "linear_num_value_heads");
    k.gdn_k_dim = geti(c, "linear_key_head_dim");
    k.gdn_v_dim = geti(c, "linear_value_head_dim");
    k.conv_kernel = geti(c, "linear_conv_kernel_dim");
    k.dense = m["arch"].str() == "qwen35";
    k.n_experts = geti(c, "num_experts");
    k.top_k = geti(c, "num_experts_per_tok");
    k.moe_ff = geti(c, "moe_intermediate_size");
    k.shared_ff = geti(c, k.dense ? "intermediate_size" : "shared_expert_intermediate_size");
    const auto &types = c["layer_types"];
    if (k.n_layers <= 0 || k.n_layers > 1024 || static_cast<int>(types.size()) != k.n_layers) {
        err = "geode.json: layer_types does not match num_hidden_layers";
        return false;
    }
    for (int l = 0; l < k.n_layers; ++l) k.attention.push_back(types.at(static_cast<size_t>(l)).str() == "full_attention");
    if (gemma) {
        k.gemma = k.dense = true;
        k.n_experts = k.top_k = k.moe_ff = 0;
        const auto ints = [&](const char *key, std::vector<int> &out) {
            const auto &a = c[key];
            for (size_t i = 0; i < a.size(); ++i) out.push_back(static_cast<int>(a.at(i).i64()));
            return static_cast<int>(out.size()) == k.n_layers;
        };
        const auto &th = c["rope_thetas"];
        for (size_t i = 0; i < th.size(); ++i) k.rope_thetas.push_back(static_cast<float>(th.at(i).f64()));
        if (!ints("kv_layer", k.kv_layer) || !ints("ffn", k.ffn) || !ints("head_dims", k.head_dims) ||
            !ints("rotary_pairs", k.rot_pairs) || static_cast<int>(k.rope_thetas.size()) != k.n_layers) {
            err = "geode.json: a gemma4 pack needs kv_layer, ffn, head_dims, rotary_pairs and rope_thetas per layer";
            return false;
        }
        k.window = geti(c, "sliding_window");
        k.ple_dim = geti(c, "hidden_size_per_layer_input");
        k.embed_scale = static_cast<float>(c["embed_scale"].f64(1.0));
        k.softcap = static_cast<float>(c["final_logit_softcapping"].f64(0.0));
        k.pad_id = static_cast<int>(c["pad_token_id"].i64(0));
        k.head_dim = *std::max_element(k.head_dims.begin(), k.head_dims.end());
        k.rope_dim = k.head_dim;
        k.shared_ff = *std::max_element(k.ffn.begin(), k.ffn.end());
        for (int l = 0; l < k.n_layers; ++l) {
            const int src = k.kv_layer[static_cast<size_t>(l)];
            const size_t i = static_cast<size_t>(l), s = static_cast<size_t>(std::max(0, src));
            if (src < 0 || src > l || k.kv_layer[s] != src || k.attention[s] != k.attention[i] ||
                k.head_dims[s] != k.head_dims[i] || k.head_dims[i] % 16 || k.ffn[i] % kBlock ||
                2 * k.rot_pairs[i] > k.head_dims[i] || k.window <= 0) {
                err = "geode.json: inconsistent gemma4 layer " + std::to_string(l);
                return false;
            }
        }
    }
    const bool moe_ok = k.dense ? k.n_experts == 0 && k.shared_ff > 0 && k.shared_ff % kBlock == 0
                                : k.top_k > 0 && k.top_k <= k.n_experts;
    if (k.hidden <= 0 || k.hidden % kBlock || !moe_ok || k.gdn_v_heads % std::max(1, k.gdn_k_heads) ||
        k.n_heads % std::max(1, k.n_kv_heads) || k.rope_dim % 2) {
        err = "geode.json: inconsistent model geometry";
        return false;
    }

    // the tensor table
    const auto &tensors = m["tensors"];
    if (!tensors.is_object()) {
        err = "geode.json: no tensor table";
        return false;
    }
    int dense_file = -1;
    for (const auto &[name, t] : tensors.object) {
        if (std::find(missing.begin(), missing.end(), t["file"].str()) != missing.end()) continue;
        Tensor x;
        x.type = dtype_from_name(t["type"].str());
        if (x.type == DType::Unknown) {
            err = name + ": unknown type '" + t["type"].str() + "'";
            return false;
        }
        // (a malformed table is refused, never taken as pointers past a file: every dimension positive and below
        // 2^31, their product below 2^40, the offset and the bytes checked without overflowing)
        bool shape_ok = true;
        for (size_t i = 0; i < t["shape"].size(); ++i) {
            const int64_t d = t["shape"].at(i).i64();
            shape_ok &= d > 0 && d < (int64_t{1} << 31);
            x.shape.push_back(d);
        }
        if (x.shape.empty()) x.shape.push_back(1);
        x.rows = x.shape[0];
        x.cols = 1;
        for (size_t i = 1; i < x.shape.size() && shape_ok; ++i) {
            x.cols *= x.shape[i];
            shape_ok &= x.cols < (int64_t{1} << 40);
        }
        if (x.shape.size() == 1) {
            x.cols = x.rows;
            x.rows = 1;
        }
        const int64_t off = t["offset"].i64();
        if (!shape_ok || off < 0 || static_cast<double>(x.rows) * static_cast<double>(x.cols) > 1099511627776.0) {
            err = name + ": a shape or an offset out of range";
            return false;
        }
        x.file = file_index(t["file"].str(), err);
        if (x.file < 0) return false;
        x.offset = static_cast<uint64_t>(off);
        x.global = static_cast<float>(t["global"].f64(1.0));
        if (is_quantized(x.type) && x.cols % kBlock) {
            err = name + ": columns not a multiple of 16";
            return false;
        }
        const uint64_t bytes = is_quantized(x.type) ? static_cast<uint64_t>(x.rows) * row_bytes(x.type, x.cols)
                                                    : static_cast<uint64_t>(x.rows * x.cols) * (x.type == DType::F32 ? 4 : 2);
        const uint64_t fsize = files_[static_cast<size_t>(x.file)]->size();
        if (x.offset > fsize || bytes > fsize - x.offset) {
            err = name + ": runs past the end of " + t["file"].str();
            return false;
        }
        if (t["file"].str() == "dense.bin") dense_file = x.file;
        tensors_.emplace(name, std::move(x));
    }

    // dense.bin into memory, every tensor in it pointed at
    if (dense_file < 0) {
        err = "geode.json: no tensors in dense.bin";
        return false;
    }
    const RandomFile &df = *files_[static_cast<size_t>(dense_file)];
    dense_size_ = df.size();
    dense_ = aligned_bytes(dense_size_);
    // in 32 MB pieces on up to 8 threads (one read of the 1.15 GB took ~1.3 s on the 8 Elite)
    bool read_ok = dense_ != nullptr;
    if (read_ok) {
        constexpr size_t kPiece = size_t{32} << 20;
        const size_t pieces = (dense_size_ + kPiece - 1) / kPiece;
        std::atomic<size_t> next{0};
        std::atomic<bool> ok{true};
        auto work = [&] {
            for (size_t i; ok && (i = next.fetch_add(1)) < pieces;) {
                const size_t at = i * kPiece, n = std::min(kPiece, dense_size_ - at);
                if (!df.read_at(at, dense_.get() + at, n)) ok = false;
            }
        };
        const size_t n_threads = std::min<size_t>(pieces, std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
        std::vector<std::thread> pool;
        for (size_t t = 1; t < n_threads; ++t) pool.emplace_back(work);
        work();
        for (std::thread &t : pool) t.join();
        read_ok = ok;
    }
    if (!read_ok) {
        err = dense_ ? "cannot read dense.bin" : "not enough memory for dense.bin";
        return false;
    }
    for (auto &[name, t] : tensors_)
        if (t.file == dense_file) t.data = dense_.get() + t.offset;

    embed_ = find("embed_tokens.weight");
    if (!embed_ || !is_quantized(embed_->type) || embed_->cols != k.hidden) {
        err = "geode.json: no usable embed_tokens.weight";
        return false;
    }

    // the experts
    const auto &el = m["experts"]["layers"];
    if (static_cast<int>(el.size()) != (k.dense ? 0 : k.n_layers)) {
        err = "geode.json: the expert table does not cover every layer";
        return false;
    }
    auto expert_layer = [&](const json::Value &e, int l) -> bool {
        ExpertLayer x;
        x.file = file_index(e["file"].str(), err);
        if (x.file < 0) return false;
        x.offset = static_cast<uint64_t>(e["offset"].i64());
        if (e["stride"].is_number()) {
            x.stride = static_cast<uint64_t>(e["stride"].i64());
            x.type = dtype_from_name(e["format"].str());
        } else {
            for (size_t i = 0; i < e["offsets"].size(); ++i) {
                x.offsets.push_back(static_cast<uint64_t>(e["offsets"].at(i).i64()));
                x.types.push_back(dtype_from_name(e["formats"].at(i).str()));
            }
            if (static_cast<int>(x.offsets.size()) != k.n_experts) {
                err = "geode.json: layer " + std::to_string(l) + " lists the wrong number of experts";
                return false;
            }
        }
        experts_.push_back(std::move(x));
        const std::string scales = l < k.n_layers ? "layers." + std::to_string(l) + ".mlp.experts.scales"
                                                  : "mtp.layers.0.mlp.experts.scales";
        const Tensor *s = find(scales);
        if (!s || s->type != DType::F32 || s->rows != k.n_experts || s->cols != 2) {
            err = "geode.json: no expert scales for layer " + std::to_string(l);
            return false;
        }
        return true;
    };
    for (int l = 0; l < (k.dense ? 0 : k.n_layers); ++l)
        if (!expert_layer(el.at(static_cast<size_t>(l)), l)) return false;
    // q3's high bits moved to value order (tools/geodepack.py); an older q3 pack would decode to wrong levels
    bool any_q3 = false;
    for (const auto &[name, t] : tensors_) any_q3 |= t.type == DType::Q3;
    for (const auto &x : experts_) {
        any_q3 |= x.type == DType::Q3;
        for (DType t : x.types) any_q3 |= t == DType::Q3;
    }
    if (any_q3 && m["formats"]["q3"]["high_bits"].str() != "value order") {
        err = "geode.json: q3 with the old bit planes; convert it with tools/q3_relayout.py";
        return false;
    }
    // q4 experts in tile8 (the dense matrices are laid out at load time)
    experts_tile8_ = m["formats"]["q4"]["layout"].str() == "tile8";
    // the MTP layer, when the pack has one: its experts after the last layer's
    if (m["mtp"].is_object()) {
        if (!k.dense && !expert_layer(m["mtp"]["experts"], k.n_layers)) return false;
        config_.mtp_layers = 1;
    }
    return true;
}

std::string Pack::path(int index) const { return (fs::path(dir_) / file_names_[static_cast<size_t>(index)]).string(); }

const Tensor *Pack::find(const std::string &name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

void Pack::expert(int layer, int e, int &file, uint64_t &offset, DType &type) const {
    const auto &x = experts_[static_cast<size_t>(layer)];
    file = x.file;
    if (x.stride) {
        offset = x.offset + static_cast<uint64_t>(e) * x.stride;
        type = x.type;
    } else {
        offset = x.offsets[static_cast<size_t>(e)];
        type = x.types[static_cast<size_t>(e)];
    }
}

size_t Pack::expert_block_bytes(DType type) const {
    const auto &k = config_;
    return static_cast<size_t>(2 * k.moe_ff) * row_bytes(type, k.hidden) +
           static_cast<size_t>(k.hidden) * row_bytes(type, k.moe_ff);
}

size_t Pack::max_expert_block_bytes() const {
    size_t best = 0;
    for (const auto &x : experts_) {
        if (x.stride) best = std::max(best, expert_block_bytes(x.type));
        for (DType t : x.types) best = std::max(best, expert_block_bytes(t));
    }
    return best;
}

bool Pack::read_row(const Tensor &t, int64_t r, float *out) const {
    if (r < 0 || r >= t.rows || !is_quantized(t.type)) return false;
    const size_t rb = row_bytes(t.type, t.cols);
    thread_local std::vector<uint8_t> row;
    row.resize(rb);
    if (!files_[static_cast<size_t>(t.file)]->read_at(t.offset + static_cast<uint64_t>(r) * rb, row.data(), rb))
        return false;
    QView v{t.type, 1, t.cols, rb, row.data(), t.global};
    dequant_row(v, 0, out);
    return true;
}

bool Pack::embedding(int token, float *out) const {
    if (token < 0 || token >= embed_->rows) return false;
    const size_t rb = row_bytes(embed_->type, embed_->cols);
    std::vector<uint8_t> row(rb);
    if (!files_[static_cast<size_t>(embed_->file)]->read_at(embed_->offset + static_cast<uint64_t>(token) * rb,
                                                             row.data(), rb))
        return false;
    QView v{embed_->type, 1, embed_->cols, rb, row.data(), embed_->global};
    dequant_row(v, 0, out);
    return true;
}

}  // namespace geode
