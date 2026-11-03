// media.hpp - what the media encoders (vision.cpp, audio.cpp) share: their products over many tokens at once on the
// int8 kernels - SMMLA where the CPU has it (FEAT_I8MM, the 8 Elite), the int8 row kernels elsewhere - and the
// "clipped" linears of Gemma 4's encoders.
#pragma once

#include <cstdint>
#include <vector>

#include "kernels.hpp"
#include "pack.hpp"
#include "threads.hpp"

namespace geode::media {

// ---- The products on SMMLA (a 2 x 8 by 8 x 2 int8 matrix multiply an instruction). A q8 matrix repacked for it: its
// rows in pairs, each pair's 8-value chunks side by side (16 bytes: row 2p's 8 codes, then row 2p + 1's) and its
// block scales (E4M3 x global) as two floats. The activations likewise in pairs of tokens, int8 a 16-value block
// (amax / 127, as quantize_activation) with two floats a block. A block of 2 x 2 is then two SMMLAs, a convert and one
// multiply-add with its four scale products: 3.5x the int8 row kernels' arithmetic a cycle on the 8 Elite.
struct Packed {
    int64_t rows = 0, cols = 0;
    std::vector<int8_t> codes;   // [rows / 2][cols / 8][16]
    std::vector<float> scales;   // [rows / 2][cols / 16][2]
};

bool have_i8mm();   // GEODE_VISION_NO_I8MM: no
// a q8 matrix (rows even) into the pairs layout
void pack_q8(const QView &m, Packed &o);
// tokens x0 and x1 (nullptr: past the last, zeros) into one pair: q [C / 8][16], s [C / 16][2]
void quant_pair(const float *x0, const float *x1, int64_t C, int8_t *q, float *s);
// row tiles [a, b) (8 rows each) against every token (n): y[token][row] ystride apart
void mmla_rows(const Packed &w, int64_t a, int64_t b, const int8_t *aq, const float *as, int n, float *y,
               size_t ystride);
// a tensor's bytes in its file
uint64_t tensor_bytes(const Tensor &t);

// y rows (ystride apart) = m . x rows (xstride apart) for n tokens, on the pool's threads (wide: the little cores
// too); pk: m repacked for SMMLA (pack_q8), else the int8 row kernels. The scratch is kept from one product to the next.
class Products {
public:
    void mm(ThreadPool &pool, const QView &m, const Packed *pk, const float *xs, size_t xstride, float *yo, size_t ystride,
            int n);
    // Gemma4ClippableLinear: xs (rows of m.cols) clamped to [clip 0, clip 1], the product, yo (rows of m.rows)
    // clamped to [clip 2, clip 3]
    void clipped(ThreadPool &pool, const QView &m, const Packed *pk, const float *clip, const float *xs, float *yo, int n);

private:
    std::vector<QAct> qa_;
    std::vector<const QAct *> qp_;
    std::vector<float *> ys_;
    std::vector<int8_t> aq_;
    std::vector<float> as_, cl_;
};

}  // namespace geode::media
