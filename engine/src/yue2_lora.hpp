#pragma once
#include "threads.hpp"
#include "yue2_pack.hpp"
#include "yue2_safetensors.hpp"
#include <functional>
#include <string>
#include <vector>
namespace geode::yue2 {
struct LoraSpec {
    std::string path;
    float scale = 1;
    std::string cache;
    bool f16 = false; // caller wishes to prepare the DSP AR too, when present in the base pack
};
struct LoraWeights {
    std::string directory, sha256, key;
    float scale = 1;
    bool f16 = false;           // verified manifest.f16.json and ar_f16.bin are ready
    double t_q8 = 0, t_f16 = 0; // preparation wall times (including cache lookup)
};
using LoraProgress = std::function<void(int, int)>;
using LoraCancel = std::function<bool()>;
std::string lora_scale_text(float scale);
bool lora_sha256(const std::string &path, std::string &hex, std::string &err, const LoraCancel &cancel = {});
// The same, remembered in <cache>/hashes.json by the file's path, size and modification time: the files are the app's
// own (a changed one has a new size or time), and reading the adapter, the base and the merged copy whole (~4 GB) for
// every song costs a phone tens of seconds and the pack's cached pages. Read whole the first time and after a change.
bool lora_sha256_remembered(const std::string &path, const std::string &cache, std::string &hex, std::string &err,
                            const LoraCancel &cancel = {});
bool validate_lora(const SafeTensors &adapter, const Pack *pack, int &rank, std::string &err);
bool prepare_lora(const std::string &pack_dir, const LoraSpec &spec, ThreadPool &pool, const LoraProgress &progress,
                  const LoraCancel &cancel, LoraWeights &out, std::string &err);
// Same rank/matrix shapes as pack_yue2.q8_rows. Bytes are row codes followed by per-16 E4M3 scales.
bool lora_q8_encode(const float *x, int64_t rows, int64_t cols, std::vector<uint8_t> &bytes, float &global,
                    std::string &err, ThreadPool *pool = nullptr);
bool lora_quant_self_test(std::string &err);
} // namespace geode::yue2
