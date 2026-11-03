// kernels.hpp - the arithmetic of the forward pass on quantized matrices (tools/geodepack.py has the layout).
// Scalar C++ for now: it is the reference the NEON kernels will be checked against, and it runs the PC parity tests.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace geode {

// Q4I (Gemma 4's QAT Q4_0): q4's nibbles with the levels code - 8 and signed block scales (tools/geodepack.py); after
// Unknown so the older values stay what they were
enum class DType : uint8_t { F32, BF16, Q2, Q3, NVFP4, Q8, Q4, Unknown, Q4I };

DType dtype_from_name(const std::string &name);
const char *dtype_name(DType t);
bool is_quantized(DType t);

constexpr int kBlock = 16;   // values per E4M3 block scale

// Code bytes of one 16-value block; a row is cols/16 blocks of codes, then cols/16 scale bytes.
int block_code_bytes(DType t);
size_t row_bytes(DType t, int64_t cols);

// A quantized matrix in memory (or one row of it, for the embedding).
struct QView {
    DType type = DType::Unknown;
    int64_t rows = 0, cols = 0;
    size_t row_bytes = 0;
    const uint8_t *data = nullptr;
    float global = 1.0f;
    bool quad = false;   // the GPU layout (to_quad): the CPU kernels read it too
    bool tile8 = false;  // with quad: its quads interleaved in tiles of eight (to_tile8, the Adreno 830's int8 kernels)
};

// The GPU layout ("quad"; bench/gpu/asm's kernels read it): the rows in groups of four, each group's bytes permuted so
// that one 16-byte load gives the four rows' code word j of a 64-value chunk c, a word = 16 values (q2) or 8 (q4):
// word (j, c, r) at ((j * nch + c) * 4 + r) * 4, then the scale bytes, block b of chunk c of row r at
// codes + (c * 4 + b) * 4 + r (a block's four rows side by side). The codes in the affine order, level = 2c - 3 (q2) or 2c - 15 (q4): a GPU decodes a value
// in three instructions. A permutation inside each group of rows: no byte more than the pack's layout. The CPU reads
// it at least as fast as the pack's layout (faster for nibbles), so nvfp4 takes it too (its codes unchanged: no GPU
// kernel for it).
bool quad_capable(DType t, int64_t rows, int64_t cols);   // q2 / q4 / nvfp4, rows % 4 == 0, cols % 64 == 0
// rows x cols of the pack's layout at src into the GPU layout at dst (dst may be src)
void to_quad(DType t, int64_t rows, int64_t cols, const uint8_t *src, uint8_t *dst);
// The tiled quad layout ("tile8", q4; bench/gpu/i8): the quad layout's words with eight quads side by side, so a
// GPU wave's loads coalesce while the activations stay one address for it. The rows in tiles of 32, each tile in
// the bytes of its 32 rows (a permutation inside the tile): code word j of chunk c of quad i (its four rows' 32-bit
// words, affine codes) at ((c * 8 + j) * 8 + i) * 16, then the scales, chunk c of quad i ([block][row] bytes) at
// 1024 * nch + (c * 8 + i) * 16. A quad's words are the quad layout's, only further apart: the CPU's quad kernel
// reads it through strides.
bool tile8_capable(DType t, int64_t rows, int64_t cols);   // q4 / q4i, rows % 32 == 0, cols % 64 == 0
void to_tile8(DType t, int64_t rows, int64_t cols, const uint8_t *src, uint8_t *dst);   // dst may be src
// The mixed pack's expert layout ("tile8x": tile8 for q2, q3 and q4 alike, so one layer's routed experts of three
// formats read the SAME activation tiles on the GPU). Per row and 16-value block, byte b (0..3) of a block's words
// holds the values v(q, b) of the four activation lanes q: v(0, b) = 2b, v(1, b) = 8 + 2b, v(2, b) = 2b + 1,
// v(3, b) = 9 + 2b (lane q's 4 values are one 32-bit word of the activation tile, as for q4: values 0 2 4 6 | 8 10
// 12 14 | 1 3 5 7 | 9 11 13 15). Codes affine, level = 2c - (2^bits - 1). A row's words per 64-value chunk (blocks
// k = 0..3):
//   q4: 8 words, word 2k: byte b = c(v(0,b)) | c(v(2,b)) << 4; word 2k+1: byte b = c(v(1,b)) | c(v(3,b)) << 4
//       (exactly tile8's: the pack's nibbles, affine)
//   q2: 4 words, word k: byte b = c(v(0,b)) | c(v(1,b)) << 2 | c(v(2,b)) << 4 | c(v(3,b)) << 6
//   q3: 6 words, word k (k < 4): q2's word of the codes' low 2 bits; words 4 and 5: the high bits of blocks 0-1 and
//       2-3: byte b = h(v(0,b)) | h(v(1,b)) << 1 | h(v(2,b)) << 2 | h(v(3,b)) << 3 for the even block, the same << 4
//       for the odd one
// The rows in tiles of 32 as tile8: word j of chunk c of quad i (its four rows' words) at byte ((c * W + j) * 8 + i)
// * 16 + 4 r, W the words above; then the scales (the pack's E4M3 bytes), block b of chunk c of row r of quad i at
// 128 W nch + (c * 8 + i) * 16 + 4 b + r. The same bytes as the pack's layout (a permutation inside each tile).
// A GPU decodes lane q of a block into 4 signed int8 levels: q4 ((w << 1 or w >> 3) & 0x1E..) + 0x71.. ^ 0x80..;
// q2 ((w >> (2q - 1)) & 0x06..) + 0x7D.. ^ 0x80..; q3 the same from the low word | ((h >> (q + 4 odd)) & 0x01..) << 3,
// + 0x79.. ^ 0x80.. (tile8x_lane is that arithmetic, the reference the kernels are checked against).
bool tile8x_capable(DType t, int64_t rows, int64_t cols);   // q2 / q3 / q4, rows % 32 == 0, cols % 64 == 0
int tile8x_words(DType t);                                  // W: a row's 32-bit words a 64-value chunk
void to_tile8x(DType t, int64_t rows, int64_t cols, const uint8_t *src, uint8_t *dst);   // dst != src
// row r of a tile8x matrix (data: its first row's address) back in the pack's layout
void tile8x_row(DType t, int64_t cols, const uint8_t *data, int64_t r, uint8_t *out);
// lane q (0..3) of block blk (0..3) of chunk c of row r: its 4 int8 levels as a GPU decodes them from the words
void tile8x_lane(DType t, int64_t cols, const uint8_t *data, int64_t r, int64_t c, int blk, int q, int8_t out[4]);
// row r of a quad (or tile8) view in the pack's layout (row_bytes bytes)
void quad_row(const QView &m, int64_t r, uint8_t *out);

float e4m3_value(uint8_t b);

// One row to floats (exact: level x E4M3 scale x global, as the PC quantizer computed them).
void dequant_row(const QView &m, int64_t r, float *out);

// y[r] = W[r, :] . x for r in [r0, r1), exactly: FP32 activations times the dequantized weights.
void matvec_rows(const QView &m, const float *x, float *y, int64_t r0, int64_t r1);

// Activations quantized to int8 per 16-value block (scale = amax / 127), in natural order and in the lane orders the
// packed codes come out of a 16-byte load in (q2/q3: lane j of plane s = value 4j+s of a 64-group; nvfp4: 2j+s of a
// 32-group).
struct QAct {
    int64_t cols = 0;
    std::vector<int8_t> q, p4, p2;
    std::vector<float> scale;
    std::vector<int32_t> bsum;   // per 16-value block: the sum of q (the q3 kernel's odd levels need it)
};
void quantize_activation(const float *x, int64_t cols, QAct &a);
// The same a 64-value group at a time, from any thread: prepare_activation sizes a for cols values (a multiple of 64),
// quantize_activation_group fills in values [g, g + 64) of x (g a multiple of 64) - quantize_activation's bits (its
// blocks and lane orders never cross a group of 64).
void prepare_activation(int64_t cols, QAct &a);
void quantize_activation_group(const float *x, int64_t g, QAct &a);

// The int8 path: per block an integer dot product, times the weight's and the activation's scales. NEON SDOT when
// built for ARMv8.2 dotprod (and cols % 64 == 0); matvec_rows_q_ref is the same arithmetic in plain C++.
void matvec_rows_q(const QView &m, const QAct &a, float *y, int64_t r0, int64_t r1);
// The same for n tokens: each row's codes are decoded once and multiplied with up to 4 tokens' activations at a time;
// every token's result is bit for bit what matvec_rows_q gives it alone.
void matvec_rows_q_multi(const QView &m, const QAct *const *acts, int n, float *const *ys, int64_t r0, int64_t r1);
void matvec_rows_q_ref(const QView &m, const QAct &a, float *y, int64_t r0, int64_t r1);
bool have_dotprod_kernels();
// q4 quads (quad / tile8) against two or more tokens on SMMLA (FEAT_I8MM), bit for bit the SDOT kernels: with
// GEODE_I8MM=1 only (slower than SDOT in a real prompt on the 8 Elite)
bool have_i8mm_kernels();
void matvec_rows_f32(const float *w, int64_t cols, const float *x, float *y, int64_t r0, int64_t r1);
void matvec_rows_bf16(const uint16_t *w, int64_t cols, const float *x, float *y, int64_t r0, int64_t r1);
// n tokens at once (each row widened once for up to eight of them); not bit-equal to matvec_rows_bf16
void matvec_rows_bf16_multi(const uint16_t *w, int64_t cols, int n, const float *const *xs, float *const *ys,
                            int64_t r0, int64_t r1);

inline float bf16_to_f32(uint16_t v) {
    const uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x / rms(x) * (1 + w) (Qwen3.6's zero-centred norm) or * w (plain); computed in f32 as transformers does. w nullptr:
// no weight (Gemma 4's value norm).
void rmsnorm(const float *x, const float *w, float *out, int n, float eps, bool zero_centered);

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// e^x (Cephes' expf: the reduction by ln 2 in two parts, a degree-6 polynomial, the power of two added to the
// exponent's bits), within 2 ulp; below -87 it is e^-87. No branches and no call, so a loop of it vectorizes
// (std::exp does not: a prompt's Gated DeltaNet takes ~1.1 G of them for 3000 tokens, a vision encode ~0.6 G).
inline float fast_exp(float x) {
    x = std::fmin(std::fmax(x, -87.0f), 88.0f);
    const float n = std::floor(x * 1.44269504088896341f + 0.5f);
    float r = x - n * 0.693359375f;
    r = r + n * 2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    uint32_t bits;
    std::memcpy(&bits, &p, 4);
    bits += static_cast<uint32_t>(static_cast<int32_t>(n)) << 23;
    std::memcpy(&p, &bits, 4);
    return p;
}
// silu on fast_exp (the loops over a batch's channels)
inline float silu_fast(float x) { return x / (1.0f + fast_exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
// torch.nn.functional.softplus with beta 1, threshold 20
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

}  // namespace geode
