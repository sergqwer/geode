// yue2_pack.hpp - YuE2's geode-pack (tools/pack_yue2.py): the manifest, its tensors, and the files read into memory
// a stage at a time (ar.bin for the AR, nar.bin for the NAR, vae.bin for the decoder; embed.bin read by rows).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"
#include "kernels.hpp"
#include "platform.hpp"

namespace geode::yue2 {

struct TensorInfo {
    DType type = DType::Unknown;        // F32, BF16, Q8 or Q4 (2-D: tools/geodepack.py's rows); Unknown for hmx16
    std::vector<int64_t> shape;
    float global = 1.0f;                // Q8 / Q4: the matrix's fp32 global scale
    mutable bool quad = false;          // Q4: its loaded bytes turned into the quad layout (Pack::q4; reset by load)
    bool hmx16 = false;                 // HMX's fp16 weight blocks (tools/pack_yue2.py hmx16_blocks): [rows, cols]
    int shift = 0;                      // hmx16 / hmx8: the values are 2^shift larger (the DSP's product takes it out)
    bool hmx8 = false;                  // HMX's int8 blocks (npu wsrc.h HMX_Q8: codes in the fp16 blocks' element order,
                                        // an fp16 scale a column and 32 inputs; yue2_q4.hpp make_nar_q8): [rows, cols]
    int file = -1;
    uint64_t offset = 0, bytes = 0;
};

class Pack {
public:
    Pack() = default;
    ~Pack();
    Pack(const Pack &) = delete;
    Pack &operator=(const Pack &) = delete;

    // Reads geode.json (arch "yue2"), opens every file and checks its size; reads no tensor data.
    bool open(const std::string &dir, std::string &err);
    // Initialization only: substitute a completed LoRA cache's Q8 file/tensors. No a16/Q4 route in this view.
    bool open_selected(const std::string &dir, const std::string &selection, std::string &err);

    // An extra manifest beside geode.json (geode-pack-extra: yue2_q4.hpp's, made on the phone) merged now - open merges
    // the ones there already. Not while anything reads the pack. False (err why, the pack as it was) when it is not
    // there or does not check out.
    bool merge_extra(const std::string &name, std::string &err);

    const json::Value &manifest() const { return manifest_; }
    const json::Value &config() const { return manifest_["config"]; }
    const TensorInfo *find(const std::string &name) const;
    const std::map<std::string, TensorInfo> &tensors() const { return tensors_; }
    const std::string &file_name(int file) const { return names_[static_cast<size_t>(file)]; }

    // A whole file into (4096-aligned) memory, once; unload gives the memory back.
    bool load(const std::string &file, std::string &err);
    // The file mapped read-only instead (Android / Linux; elsewhere: load): its pages are the page cache's, which
    // drops and re-reads them under memory pressure - a phone whose free memory cannot hold a copy still runs (the
    // tensors' pointers stay valid until unload). The same pointers as load for every reader of data().
    bool map(const std::string &file, std::string &err);
    void unload(const std::string &file);   // (either)
    bool loaded(const std::string &file) const;

    // A tensor's bytes, its file loaded (else nullptr).
    const uint8_t *data(const TensorInfo &t) const;
    // The tensor of this name, type and shape, its file loaded; else nullptr and err says why.
    const float *f32(const std::string &name, const std::vector<int64_t> &shape, std::string &err) const;
    const uint16_t *bf16(const std::string &name, const std::vector<int64_t> &shape, std::string &err) const;
    // A Q8 matrix [rows, cols] as the kernels take it (kernels.hpp QView), its file loaded; false and err if not.
    bool q8(const std::string &name, int64_t rows, int64_t cols, QView &out, std::string &err) const;
    // A Q4 matrix [rows, cols] in the quad layout (kernels.hpp to_quad: the NEON / i8mm kernels'), turned so in its
    // file's memory the first time - the file loaded (load), not mapped. One thread at a time (the turn writes the
    // bytes in place: FastAr::open asks for its matrices in a serial loop).
    bool q4(const std::string &name, int64_t rows, int64_t cols, QView &out, std::string &err) const;
    // Rows [row, row + n) of a 2-D tensor read from its file (loaded or not) into dst.
    bool read_rows(const TensorInfo &t, int64_t row, int64_t n, void *dst) const;
    // bytes [off, off + n) of a tensor's file (the stretch a layer's tensors make) read into dst.
    bool read_span(int file, uint64_t off, uint64_t n, void *dst) const;

private:
    // m's files and tensors joined to the pack's (geode.json's, or an extra manifest's); the names of the tensors
    // added into added (optional)
    bool add_part(const json::Value &m, std::vector<std::string> *added, std::string &err);
    const uint8_t *typed(const std::string &name, DType type, const std::vector<int64_t> &shape,
                         std::string &err) const;
    std::string dir_;
    json::Value manifest_;
    std::vector<std::string> names_;
    std::vector<std::unique_ptr<RandomFile>> files_;
    std::vector<AlignedBuffer> memory_;
    std::vector<std::pair<const uint8_t *, size_t>> mapped_;   // map's (base, length)
    const uint8_t *base(size_t file) const;
    std::map<std::string, TensorInfo> tensors_;
};

// Shape as text, for messages: [2048, 64, 7]
std::string shape_text(const std::vector<int64_t> &shape);

}  // namespace geode::yue2
