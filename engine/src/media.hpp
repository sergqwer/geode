// media.hpp - what the media encoders (vision.cpp, audio.cpp) share: their products over many tokens at once on the
// int8 kernels - SMMLA where the CPU has it (FEAT_I8MM, the 8 Elite), the int8 row kernels elsewhere - and the
// "clipped" linears of Gemma 4's encoders.
#pragma once

#include <cstdint>
#include <vector>

#include "kernels.hpp"
#include "pack.hpp"
#include "threads.hpp"

namespace geode {

// The NPU's HMX for Gemma 4's media encoders (vision.cpp, audio.cpp; v79 and newer; npu/host/npu_api.h gnpu_pack_f16
// / gnpu_gemm_f16): their q8 layers turned into fp16 blocks a layer at a time, in two buffers the DSP shares (made on
// the thread that opened the NPU - the encode's), the products there; the rest stays on the CPU. Function pointers
// into libgeode_npu.so (Model::setup_npu), g its handle.
struct MediaNpu {
    void *g = nullptr;
    void *(*shared)(void *, size_t, int, int *) = nullptr;
    size_t (*f16_bytes)(int64_t, int64_t) = nullptr;
    int (*pack_f16)(const uint8_t *, int64_t, int64_t, size_t, float, uint8_t *, int *) = nullptr;
    int (*gemm_f16)(void *, int, uint64_t, int64_t, int64_t, int, int, const float *const *, float *const *) = nullptr;
    // the same with a clipped linear's clamps in its conversions (gnpu_gemm_f16c; an older library: none, the clamps
    // passes of their own)
    int (*gemm_f16c)(void *, int, uint64_t, int64_t, int64_t, int, int, const float *const *, float *const *,
                     const float *) = nullptr;
    // the attention there too (npu/attn_cfg.h ATTN_ENCODER: the keys and values as fp16 blocks the host makes)
    int (*attn)(void *, const void *, int, const float *, size_t, float *, size_t, double *) = nullptr;
    float *(*attn_buf)(void *, int, size_t) = nullptr;
    bool usable() const { return g && shared && f16_bytes && pack_f16 && gemm_f16; }
};

}  // namespace geode

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
// A matrix's fp16 blocks for the NPU (gnpu_pack_f16): in the shared buffer fd at off, its values x 2^shift
struct F16 {
    int fd = -1;
    uint64_t off = 0;
    int shift = 0;
};

class Products {
public:
    // f: m's fp16 blocks, multiplied on the NPU (use_npu's) when it is live - a failed call: this product and every
    // later one on the CPU
    void mm(ThreadPool &pool, const QView &m, const Packed *pk, const float *xs, size_t xstride, float *yo, size_t ystride,
            int n, const F16 *f = nullptr);
    // Gemma4ClippableLinear: xs (rows of m.cols) clamped to [clip 0, clip 1], the product, yo (rows of m.rows)
    // clamped to [clip 2, clip 3]
    void clipped(ThreadPool &pool, const QView &m, const Packed *pk, const float *clip, const float *xs, float *yo, int n,
                 const F16 *f = nullptr);
    void use_npu(const MediaNpu *npu) {
        npu_ = npu && npu->usable() ? npu : nullptr;
        live_ = npu_ != nullptr;
    }
    bool npu_live() const { return live_; }

private:
    // m . x rows on the NPU (clip: the clamps there too, gemm_f16c); false: not done (it failed: the CPU from now on)
    bool on_npu(const QView &m, const float *xs, size_t xstride, float *yo, size_t ystride, int n, const F16 &f,
                const float *clip);
    const MediaNpu *npu_ = nullptr;
    bool live_ = false;
    std::vector<const float *> xp_;
    std::vector<float *> yp_;
    std::vector<QAct> qa_;
    std::vector<const QAct *> qp_;
    std::vector<float *> ys_;
    std::vector<int8_t> aq_;
    std::vector<float> as_, cl_;
};

}  // namespace geode::media
