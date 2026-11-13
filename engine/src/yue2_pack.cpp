// yue2_pack.cpp - see yue2_pack.hpp.
#include "yue2_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__ANDROID__) || defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define GEODE_PACK_MMAP 1
#endif

namespace geode::yue2 {

std::string shape_text(const std::vector<int64_t> &shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

bool Pack::open(const std::string &dir, std::string &err) {
    if (!names_.empty() || !tensors_.empty()) {   // one pack per object: no stale tensors from an earlier open
        err = "Pack::open: already open (" + dir_ + ")";
        return false;
    }
    dir_ = dir;
    std::ifstream f(dir + "/geode.json", std::ios::binary);
    if (!f) {
        err = "cannot open " + dir + "/geode.json";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    if (!json::parse(ss.str(), manifest_, err)) {
        err = "geode.json: " + err;
        return false;
    }
    if (manifest_["format"].str() != "geode-pack" || manifest_["version"].i64() != 1 ||
        manifest_["arch"].str() != "yue2") {
        err = "not a YuE2 geode-pack (arch '" + manifest_["arch"].str() + "')";
        return false;
    }
    if (!add_part(manifest_, nullptr, err)) return false;
    // the extra manifests beside it (yue2_q4.hpp's geode.ar_q4.json and geode.nar_q8.json, made on the phone): their
    // files and tensors join the pack's; one that does not check out is left out whole (a note), never the pack
    for (const char *xname : {"geode.ar_q4.json", "geode.nar_q8.json", "geode.nar_c8.json"}) {
        std::string e;
        if (std::ifstream(dir + "/" + xname).good() && !merge_extra(xname, e))
            std::fprintf(stderr, "pack: %s left out (%s)\n", xname, e.c_str());
    }
    return true;
}

bool Pack::open_selected(const std::string &dir, const std::string &selection, std::string &err, bool f16) {
    if (!open(dir, err)) return false;
    if (selection.empty()) {
        if (f16) { err = "lora: fp16 selection needs a cache entry"; return false; }
        return true;
    }
    const auto replace = [&](const char *file, const char *manifest, const char *prefix) {
        Pack replacement;
        replacement.dir_ = selection;
        std::ifstream f(selection + "/" + manifest, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() < 0 || f.tellg() > (32 << 20)) { err = "lora: invalid/missing selection manifest " + std::string(manifest); return false; }
        std::string text(static_cast<size_t>(f.tellg()), '\0'); f.seekg(0);
        json::Value m;
        if (!f.read(text.data(), text.size()) || !json::parse(text, m, err) ||
            m["format"].str() != "geode-pack-extra" || m["version"].i64() != 1 ||
            m["files"].size() != 1 || m["files"].at(0)["name"].str() != file ||
            !replacement.add_part(m, nullptr, err)) { err = "lora: invalid selection manifest: " + err; return false; }
        int original = -1;
        for (size_t i = 0; i < names_.size(); ++i) if (names_[i] == file) original = static_cast<int>(i);
        if (original < 0 || replacement.files_[0]->size() != files_[original]->size()) {
            err = "lora: missing/different-size base file " + std::string(file); return false;
        }
        size_t count = 0;
        for (const auto &[name, t] : tensors_) if (t.file == original) {
            ++count;
            const TensorInfo *v = replacement.find(name);
            if (!v || name.rfind(prefix, 0) != 0 || t.shape != v->shape || t.type != v->type ||
                t.hmx16 != v->hmx16 || t.hmx8 != v->hmx8 || t.bytes != v->bytes || t.offset != v->offset) {
                err = "lora: selection differs from base tensor " + name; return false;
            }
        }
        if (count != replacement.tensors_.size()) { err = "lora: unexpected selected tensors"; return false; }
        files_[original] = std::move(replacement.files_[0]);
        selected_paths_[file] = selection + "/" + file;
        for (auto &[name, t] : replacement.tensors_) { t.file = original; tensors_[name] = t; }
        return true;
    };
    if (!replace("ar_q8.bin", "manifest.json", "q8.")) return false;
    if (f16 && !replace("ar_f16.bin", "manifest.f16.json", "a16.")) return false;
    // Without f16, a16 stays the base's. Adapter callers must gate DSP use on LoraWeights::f16.
    for (auto it = tensors_.begin(); it != tensors_.end();) {
        if (it->first.rfind("q4.", 0) == 0) it = tensors_.erase(it);
        else ++it;
    }
    return true;
}

bool Pack::merge_extra(const std::string &name, std::string &err) {
    std::ifstream xf(dir_ + "/" + name, std::ios::binary);
    if (!xf) {
        err = "cannot open " + dir_ + "/" + name;
        return false;
    }
    std::stringstream xs;
    xs << xf.rdbuf();
    json::Value extra;
    const size_t nf = names_.size();
    std::vector<std::string> added;
    if (!json::parse(xs.str(), extra, err) || (extra["format"].str() != "geode-pack-extra" && (err = "not an extra manifest", true)) ||
        !add_part(extra, &added, err)) {
        for (const auto &t : added) tensors_.erase(t);
        names_.resize(nf), files_.resize(nf), memory_.resize(nf), mapped_.resize(nf);
        return false;
    }
    return true;
}

bool Pack::add_part(const json::Value &m, std::vector<std::string> *added, std::string &err) {
    const auto &files = m["files"];
    for (size_t i = 0; i < files.size(); ++i) {
        const auto &fe = files.at(i);
        auto rf = std::make_unique<RandomFile>();
        if (!rf->open(dir_ + "/" + fe["name"].str(), err)) return false;
        if (static_cast<int64_t>(rf->size()) != fe["size"].i64()) {
            err = fe["name"].str() + ": " + std::to_string(rf->size()) + " bytes, expected " +
                  std::to_string(fe["size"].i64());
            return false;
        }
        names_.push_back(fe["name"].str());
        files_.push_back(std::move(rf));
        memory_.emplace_back();
        mapped_.emplace_back(nullptr, 0);
    }
    const auto &tensors = m["tensors"];
    if (!tensors.is_object()) {
        err = "geode.json: no tensor table";
        return false;
    }
    for (const auto &[name, t] : tensors.object) {
        TensorInfo x;
        x.hmx16 = t["type"].str() == "hmx16";
        x.hmx8 = t["type"].str() == "hmx8";
        x.type = x.hmx16 || x.hmx8 ? DType::Unknown : dtype_from_name(t["type"].str());
        if (!x.hmx16 && !x.hmx8 && x.type != DType::F32 && x.type != DType::BF16 && x.type != DType::Q8 &&
            x.type != DType::Q4) {
            err = name + ": type '" + t["type"].str() + "' (only f32 / bf16 / q8 / q4 / hmx16 / hmx8 here)";
            return false;
        }
        // the manifest is data: every dimension positive, the element count without overflow, the offset aligned
        uint64_t n = 1;
        constexpr uint64_t kMax = 1ull << 40;   // 1 T elements: far above any tensor here, far below an overflow
        for (size_t i = 0; i < t["shape"].size(); ++i) {
            const int64_t d = t["shape"].at(i).i64();
            if (d <= 0 || static_cast<uint64_t>(d) > kMax / n) {
                err = name + ": bad shape " + shape_text(x.shape) + " ... " + std::to_string(d);
                return false;
            }
            x.shape.push_back(d);
            n *= static_cast<uint64_t>(d);
        }
        if (x.hmx16 || x.hmx8) {   // per 32 rows a block of max(cols, 512) x 32 fp16 (npu/host/npu_api.c
                                   // gnpu_f16_bytes); hmx8: cols x 32 codes + cols x 2 bytes of scales, cols >= 512
            if (x.shape.size() != 2 || x.shape[0] % 32 || x.shape[1] % (x.hmx8 ? 64 : 32) || (x.hmx8 && x.shape[1] < 512)) {
                err = name + ": hmx16 / hmx8 needs [rows, cols], multiples of 32 (hmx8: cols of 64, >= 512), not " +
                      shape_text(x.shape);
                return false;
            }
            x.bytes = x.hmx8 ? static_cast<uint64_t>(x.shape[0] / 32) * static_cast<uint64_t>(x.shape[1]) * 34u
                             : static_cast<uint64_t>(x.shape[0] / 32) * static_cast<uint64_t>(std::max<int64_t>(x.shape[1], 512)) * 64u;
            const int64_t sh = t["shift"].i64(-1);
            if (sh < 0 || sh > 24) {
                err = name + ": hmx16 / hmx8 shift " + std::to_string(sh) + " outside 0 .. 24";
                return false;
            }
            x.shift = static_cast<int>(sh);
            if (t["offset"].i64(-1) % 128) {
                err = name + ": hmx16 / hmx8 not 128-byte aligned (HMX reads its blocks so)";
                return false;
            }
        } else if (x.type == DType::Q8 || x.type == DType::Q4) {   // rows of codes + cols / 16 E4M3 scales (geodepack)
            if (x.shape.size() != 2 || x.shape[1] % 64 || (x.type == DType::Q4 && x.shape[0] % 4)) {
                err = name + ": q8 / q4 needs [rows, cols], cols a multiple of 64 (q4: rows of 4), not " + shape_text(x.shape);
                return false;
            }
            x.bytes = static_cast<uint64_t>(x.shape[0]) * row_bytes(x.type, x.shape[1]);
            x.global = static_cast<float>(t["global"].f64(0.0));
            if (!(x.global > 0.0f) || !std::isfinite(x.global)) {
                err = name + ": q8 / q4 without a positive global scale";
                return false;
            }
        } else {
            x.bytes = n * (x.type == DType::F32 ? 4u : 2u);
        }
        const int64_t off = t["offset"].i64(-1);
        if (off < 0 || off % 64) {
            err = name + ": bad offset " + std::to_string(off) + " (the packer aligns to 64)";
            return false;
        }
        x.offset = static_cast<uint64_t>(off);
        for (size_t i = 0; i < names_.size(); ++i)
            if (names_[i] == t["file"].str()) x.file = static_cast<int>(i);
        if (x.file < 0) {
            err = name + ": unknown file '" + t["file"].str() + "'";
            return false;
        }
        const uint64_t size = files_[static_cast<size_t>(x.file)]->size();
        if (x.bytes > size || x.offset > size - x.bytes) {
            err = name + ": runs past the end of " + t["file"].str();
            return false;
        }
        if (tensors_.count(name)) {
            err = name + ": twice";
            return false;
        }
        if (added) added->push_back(name);
        tensors_.emplace(name, std::move(x));
    }
    return true;
}

const TensorInfo *Pack::find(const std::string &name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool Pack::load(const std::string &file, std::string &err) {
    for (size_t i = 0; i < names_.size(); ++i) {
        if (names_[i] != file) continue;
        if (base(i)) return true;
        const uint64_t n = files_[i]->size();
        AlignedBuffer buf = aligned_bytes(static_cast<size_t>(n));
        if (!buf) {
            err = file + ": cannot allocate " + std::to_string(n) + " bytes";
            return false;
        }
        constexpr uint64_t kPiece = 64ull << 20;
        for (uint64_t off = 0; off < n; off += kPiece) {
            const size_t len = static_cast<size_t>(std::min(kPiece, n - off));
            if (!files_[i]->read_at(off, buf.get() + off, len)) {
                err = file + ": read failed at " + std::to_string(off);
                return false;
            }
        }
        memory_[i] = std::move(buf);
        for (auto &kv : tensors_)   // (fresh bytes: in the pack's layout again)
            if (kv.second.file == static_cast<int>(i)) kv.second.quad = false;
        return true;
    }
    err = "no file '" + file + "' in the pack";
    return false;
}

bool Pack::map(const std::string &file, std::string &err) {
#if GEODE_PACK_MMAP
    for (size_t i = 0; i < names_.size(); ++i) {
        if (names_[i] != file) continue;
        if (base(i)) return true;
        const uint64_t n = files_[i]->size();
        const auto selected = selected_paths_.find(file);
        const std::string path = selected == selected_paths_.end() ? dir_ + "/" + file : selected->second;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            err = path + ": cannot open for mapping";
            return false;
        }
        void *p = ::mmap(nullptr, static_cast<size_t>(n), PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);   // (the mapping keeps the file)
        if (p == MAP_FAILED) {
            err = path + ": mmap of " + std::to_string(n) + " bytes failed";
            return false;
        }
        mapped_[i] = {static_cast<const uint8_t *>(p), static_cast<size_t>(n)};
        return true;
    }
    err = "no file '" + file + "' in the pack";
    return false;
#else
    return load(file, err);
#endif
}

void Pack::unload(const std::string &file) {
    for (size_t i = 0; i < names_.size(); ++i) {
        if (names_[i] != file) continue;
        memory_[i].reset();
#if GEODE_PACK_MMAP
        if (mapped_[i].first) ::munmap(const_cast<uint8_t *>(mapped_[i].first), mapped_[i].second);
#endif
        mapped_[i] = {nullptr, 0};
    }
}

Pack::~Pack() {
    for (const auto &name : names_) unload(name);
}

bool Pack::loaded(const std::string &file) const {
    for (size_t i = 0; i < names_.size(); ++i)
        if (names_[i] == file) return base(i) != nullptr;
    return false;
}

const uint8_t *Pack::base(size_t file) const {
    if (file >= memory_.size()) return nullptr;
    if (memory_[file]) return memory_[file].get();
    return mapped_[file].first;
}

const uint8_t *Pack::data(const TensorInfo &t) const {
    const uint8_t *b = base(static_cast<size_t>(t.file));
    return b ? b + t.offset : nullptr;
}

const uint8_t *Pack::typed(const std::string &name, DType type, const std::vector<int64_t> &shape,
                           std::string &err) const {
    const TensorInfo *t = find(name);
    if (!t) {
        err = "no tensor " + name;
        return nullptr;
    }
    if (t->type != type || t->shape != shape) {
        err = name + ": " + shape_text(t->shape) + " " + dtype_name(t->type) + ", expected " + shape_text(shape) +
              " " + dtype_name(type);
        return nullptr;
    }
    const uint8_t *p = data(*t);
    if (!p) err = name + ": " + names_[static_cast<size_t>(t->file)] + " is not loaded";
    return p;
}

const float *Pack::f32(const std::string &name, const std::vector<int64_t> &shape, std::string &err) const {
    return reinterpret_cast<const float *>(typed(name, DType::F32, shape, err));
}

const uint16_t *Pack::bf16(const std::string &name, const std::vector<int64_t> &shape, std::string &err) const {
    return reinterpret_cast<const uint16_t *>(typed(name, DType::BF16, shape, err));
}

bool Pack::q8(const std::string &name, int64_t rows, int64_t cols, QView &out, std::string &err) const {
    const uint8_t *p = typed(name, DType::Q8, {rows, cols}, err);
    if (!p) return false;
    out = QView{};
    out.type = DType::Q8;
    out.rows = rows;
    out.cols = cols;
    out.row_bytes = row_bytes(DType::Q8, cols);
    out.data = p;
    out.global = find(name)->global;
    return true;
}

bool Pack::q4(const std::string &name, int64_t rows, int64_t cols, QView &out, std::string &err) const {
    const uint8_t *p = typed(name, DType::Q4, {rows, cols}, err);
    if (!p) return false;
    const TensorInfo *t = find(name);
    if (!memory_[static_cast<size_t>(t->file)]) {   // (a mapped file's pages are read-only)
        err = name + ": q4 needs " + names_[static_cast<size_t>(t->file)] + " loaded, not mapped";
        return false;
    }
    if (!t->quad) {
        to_quad(DType::Q4, rows, cols, p, const_cast<uint8_t *>(p));
        t->quad = true;
    }
    out = QView{};
    out.type = DType::Q4;
    out.rows = rows;
    out.cols = cols;
    out.row_bytes = row_bytes(DType::Q4, cols);
    out.data = p;
    out.global = t->global;
    out.quad = true;
    return true;
}

bool Pack::read_span(int file, uint64_t off, uint64_t n, void *dst) const {
    if (file < 0 || static_cast<size_t>(file) >= files_.size()) return false;
    const uint64_t size = files_[static_cast<size_t>(file)]->size();
    if (n > size || off > size - n) return false;
    if (const uint8_t *m = base(static_cast<size_t>(file))) {
        std::memcpy(dst, m + off, static_cast<size_t>(n));
        return true;
    }
    constexpr uint64_t kPiece = 64ull << 20;
    for (uint64_t at = 0; at < n; at += kPiece)
        if (!files_[static_cast<size_t>(file)]->read_at(off + at, static_cast<uint8_t *>(dst) + at,
                                                        static_cast<size_t>(std::min(kPiece, n - at))))
            return false;
    return true;
}

bool Pack::read_rows(const TensorInfo &t, int64_t row, int64_t n, void *dst) const {
    if (t.hmx16 || t.hmx8 || t.shape.size() != 2 || row < 0 || n < 0 || row + n > t.shape[0]) return false;
    const uint64_t rb = t.type == DType::Q8 ? row_bytes(DType::Q8, t.shape[1])
                                            : static_cast<uint64_t>(t.shape[1]) * (t.type == DType::F32 ? 4u : 2u);
    if (const uint8_t *p = data(t)) {
        std::memcpy(dst, p + static_cast<uint64_t>(row) * rb, static_cast<size_t>(n * static_cast<int64_t>(rb)));
        return true;
    }
    return files_[static_cast<size_t>(t.file)]->read_at(t.offset + static_cast<uint64_t>(row) * rb, dst,
                                                        static_cast<size_t>(static_cast<uint64_t>(n) * rb));
}

}  // namespace geode::yue2
