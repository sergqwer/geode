// yue2_q8gemm.hpp - the CPU's NAR products from the 8-bit weights (hmx8: yue2_q4.hpp make_nar_q8) by SDOT, for the
// phones without the NPU's HMX ("8-bit sound"): the activations in int8 per 32 values (amax / 127), each 32-row block
// of the matrix a panel at a time rearranged so a row's four inputs lie together (SDOT's lanes), int32 sums per 32
// inputs, then times the weights' scale and the activation's into fp32.
#pragma once

#include <cstdint>
#include <vector>

#include "threads.hpp"

namespace geode::yue2 {

// n activation rows of cols values (cols % 32 == 0) in int8: q [n][cols], s [n][cols / 32]
struct X8 {
    std::vector<int8_t> q;
    std::vector<float> s;
    int64_t cols = 0;
    int n = 0;
};
// rot: each 32 values turned by the normalized 32-point Hadamard first (fwht32) - for weights turned the same way, the
// products unchanged, the outliers of a group spread over it (less rounding for the rest)
void quant_x8(const float *const *xs, int n, int64_t cols, X8 &out, ThreadPool &pool, bool rot = false);
// v = H v / sqrt(32) in place (H the 32-point Walsh-Hadamard matrix: symmetric, its own inverse once normalized)
void fwht32(float *v);

// y[t][0 .. rows) = W x[t] 2^-shift for W [rows][cols] in hmx8 (rows % 32 == 0, cols % 64 == 0, cols >= 512); false
// where the build has no SDOT (have_q8_gemm)
bool cpu_gemm_hmx8(const uint8_t *w, int64_t rows, int64_t cols, int shift, const X8 &x, float *const *ys,
                   ThreadPool &pool);
bool have_q8_gemm();

// The fp16 product it stands in for (yue2_nar.cpp's CPU backend; exposed for the tests)
void cpu_gemm_hmx16_ref(const uint8_t *w, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                        float *const *ys, ThreadPool &pool);

}  // namespace geode::yue2
