// yue2_q4.hpp - the AR's Q4 weights made on the phone from its Q8 ones (the app's "faster composing" choice):
// ar_q4.bin and its manifest geode.ar_q4.json beside the pack's geode.json, which stays as downloaded (Pack::open
// takes the extra manifest in when it is there). Every layer matrix of ar_q8.bin dequantized and quantized again as
// tools/pack_yue2.py --ar-q4 quantizes the checkpoint (q4: levels +-1 .. +-15, an E4M3 scale per 16 values searched
// for the least squared error, the global max |w| x 1.25 / (15 x 448)); lm_head and the norms copied as they are.
// From Q8 rather than bf16: Q8's own rounding is ~6% of Q4's (rms), so the result is Q4's within a fraction of a
// percent.
#pragma once

#include <functional>
#include <string>

#include "threads.hpp"

namespace geode::yue2 {

// The extra manifest's name (Pack::open merges it).
constexpr const char *kArQ4Manifest = "geode.ar_q4.json";

// Removes what a killed conversion left (ar_q4.bin.part, geode.ar_q4.json.part: up to 1.2 GB). Not while another
// thread or process converts in the same pack (the app makes one song at a time).
void clean_ar_q4_parts(const std::string &dir);

// Makes ar_q4.bin + geode.ar_q4.json from ar_q8.bin; progress(done, total) in matrices; cancelled polled between
// matrices (true: stops, the partial files removed, false returned with err "cancelled"). The data is synced to the
// storage before the manifest is renamed into place, so a manifest never outlives its file's bytes (a power loss
// included). Whether the pack then has the Q4 tensors is FastAr::has_q4 of a Pack opened afterwards.
bool make_ar_q4(const std::string &dir, ThreadPool &pool, const std::function<void(int, int)> &progress,
                const std::function<bool()> &cancelled, std::string &err);

// The NAR's 8-bit weights (the app's "8-bit sound" choice): nar_q8.bin and geode.nar_q8.json beside the pack, from
// nar_f16.bin's layers (h16.model.layers.*: the hmx16 matrices as hmx8 - pack.hpp TensorInfo::hmx8 - the norms copied;
// h8.* names, each layer's stretch page-aligned as NarDsp maps it). Per column and 32 inputs an fp16 scale amax / 127,
// the codes rounded to nearest: a velocity's error ~5x the fp16 weights' (3.5-5.3e-3 against the fp32 reference),
// still under the bf16 run's own (1.1-1.3e-2); half the bytes the DSP's products read. File safety as make_ar_q4's.
constexpr const char *kNarQ8Manifest = "geode.nar_q8.json";
// rotated: the CPU's twins instead (nar_c8.bin, geode.nar_c8.json, c8.* names): each row's 32 inputs turned by the
// normalized 32-point Hadamard (yue2_q8gemm.hpp fwht32) before the rounding - the CPU turns its activations the same
// way, so the products are unchanged while a group's outliers spread over it (RedMagic, the CPU's 8-bit products: a
// velocity's error 7.1e-3 / 9.5e-3 -> 4.9e-3 / 5.0e-3). Not for the DSP: HMX multiplies fp16 activations as they are.
constexpr const char *kNarC8Manifest = "geode.nar_c8.json";
void clean_nar_q8_parts(const std::string &dir);
bool make_nar_q8(const std::string &dir, ThreadPool &pool, const std::function<void(int, int)> &progress,
                 const std::function<bool()> &cancelled, std::string &err, bool rotated = false);
// One HMX fp16 weight block (k inputs x 32 columns in wgt_index order: npu_imp.c) as an hmx8 block (k x 34 bytes: the
// codes at the elements' own indices, then the scales [k / 32][32]). (Exposed for the tests.)
void hmx8_block(const _Float16 *src, int64_t k, uint8_t *dst);

// One 16-value block onto Q4 as qformats.best_block_scale does: its scale byte and 16 codes (0..7 the magnitudes
// 1, 3, .., 15, 8..15 their negatives), for the matrix's global s2. (Exposed for the tests.)
void q4_block(const float *x, float s2, uint8_t &scale_byte, uint8_t codes[16]);
// A non-negative float onto E4M3 (round to nearest, ties to the even code; saturating at 448).
uint8_t e4m3_byte(float x);

}  // namespace geode::yue2
