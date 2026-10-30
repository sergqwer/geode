// npu/dec_cfg.h - a dense Qwen3.5's decode step as the engine hands it to the DSP (npu/dsp/dec.c): its shape and where
// every matrix, parameter and state sits. Plain C with fixed sizes: the same bytes on the CPU and on the DSP.
//
// Three buffers the CPU allocates (rpcmem) and the DSP maps once, and the KV cache's chunks:
//   W  the matrices, in tiles of 32 rows: q4 in q4b (20 k bytes: per 64 inputs 8 code vectors - word j, lane = row,
//      its 4 bytes inputs 8j .. 8j + 7, even ones in the low nibbles, affine codes: level = 2 code - 15 - then, after
//      all the codes, per 64 inputs the scales as bf16 [block][row]: what HVX takes with neither conversion nor
//      reordering), or q4 tile8 (the GPU's layout, 18 k bytes), or q8 tiles (34 k bytes: per 64 inputs 16 code
//      vectors, lane = row, its 4 bytes = 4 inputs as code + 128, then 128 E4M3 scale bytes [block][row])
//   P  the small parameters (floats, the GDN's a / b rows bf16, the draft vocabulary's ids)
//   S  the states the CPU's prompt arithmetic reads and writes too (conv, Gated DeltaNet; two sets: a verified pair's
//      first token's states go into the other one, and a fallen draft makes it the live one), then the scratch
//   KV chunk c: [K, V][attention layer][kv_chunk positions][nkv * hd] fp16, or int8 followed by the scales
//      [K, V][attention layer][kv_chunk positions][nkv] floats (kv_i8): the engine's layout (Model::kv, kv_scale)
//
// The scratch starts with what the CPU reads after a call: the logits [DEC_T][vocab], then the final hidden rows
// [DEC_N][hidden] (normed: what the MTP layer drafts from), each row rounded up to 128 bytes. dec_size says how
// large it is.
#pragma once

#include <stdint.h>

#define DEC_MAX_LAYERS 64
#define DEC_VERSION 6
#define DEC_BLOCK 256   // attention: positions a task (the scratch keeps a part per head and block)
#define DEC_T 2         // tokens a step at most (a token and its draft)
#define DEC_N 128       // tokens a prompt call (dec_prefill) at most
#define DEC_FMT_Q4T8 0  // a matrix's tiles: q4 tile8
#define DEC_FMT_Q8 1    // q8 tiles
#define DEC_FMT_Q4B 2   // q4b

typedef struct {
    uint64_t w;      // the first row tile, bytes into W
    uint32_t k, n;   // inputs (a multiple of 64), outputs (of 32)
    float global;    // the matrix's global scale
    uint32_t fmt;    // DEC_FMT_*
    int32_t hmx_fd;      // the same matrix in hmx4 blocks for HMX (a prompt's products; libgeode_npu.so's adopted copy),
    int32_t hmx_shift;   // its fp16 scales 2^shift larger; hmx_fd 0: none
} dec_mat_t;

typedef struct {
    int32_t attn;                  // 1: full attention (index: its layer in the KV cache), 0: Gated DeltaNet (its state)
    int32_t index;
    dec_mat_t qkv, z, out;         // Gated DeltaNet
    dec_mat_t q, k, v, o;          // attention (q: per head [query hd | gate hd])
    dec_mat_t gate, up, down;      // the MLP: down(silu(gate h) * up h)
    uint32_t in_norm, post_norm;   // bytes into P: hidden floats each (zero-centred: x * (1 + w))
    uint32_t q_norm, k_norm;       // attention: hd floats each
    uint32_t ab;                   // Gated DeltaNet: a's rows, then b's, bf16 [2 nv][hidden]
    uint32_t A_log, dt_bias;       // nv floats each
    uint32_t conv;                 // [conv_dim][conv_k] floats
    uint32_t gnorm;                // dv floats (not zero-centred)
} dec_layer_t;

typedef struct {
    int32_t version, hidden, vocab, n_layers, sff;
    int32_t nh, nkv, hd, half;          // attention: heads, KV heads, head dim, rope_dim / 2
    int32_t nk, nv, dk, dv, conv_k;     // Gated DeltaNet: key heads, value heads, their dims, the convolution's taps
    int32_t n_attn, kv_chunk, max_chunks, kv_i8;
    float eps;
    uint32_t final_norm, inv_freq;      // bytes into P: hidden floats, half floats
    dec_mat_t head;                     // lm_head
    uint64_t conv_state[2], gdn_state[2];   // bytes into S, a set each: [gdn layer][conv_dim][conv_k - 1] (oldest
                                            // first), [gdn layer][nv][dv / 16][dk][16] (a head's state in strips)
    uint64_t scratch, scratch_bytes;    // bytes into S: the step's own memory (above)
    // The MTP layer (mtp 1): fc over [norm(embedding of the next token) | norm(hidden)], a full-attention layer with
    // its own KV layer, its norm, the draft head (lm_head's rows of the draft vocabulary: n_draft of them, their ids
    // at draft_ids; or the whole head, n_draft 0)
    int32_t mtp, n_draft;
    dec_mat_t mtp_fc, draft_head;
    uint32_t mtp_pre_e, mtp_pre_h, mtp_norm, draft_ids;   // bytes into P: hidden floats x 3, n_draft int32
    dec_layer_t mtp_layer;
    dec_layer_t layers[DEC_MAX_LAYERS];
} dec_cfg_t;

// The scratch's last 128 bytes: the calls' time by stage so far, in microseconds (uint64 each)
enum { DEC_PROF_NORM, DEC_PROF_PRODUCTS, DEC_PROF_CONV, DEC_PROF_GDN, DEC_PROF_QK, DEC_PROF_ATTN, DEC_PROF_JOIN,
       DEC_PROF_MLP, DEC_PROF_ARGMAX, DEC_PROF_CACHE, DEC_PROF_STEPS, DEC_PROF_MTP, DEC_PROF_QCHECK = 15,
       DEC_PROF_N = 16 };

// dec_step's flags
#define DEC_SET1 (1 << 1)              // the live states are set 1 (else 0)
// finding faults: only the first n layers (n in bits 8-15), no cache maintenance, no attention, no KV writes
#define DEC_DEBUG_LAYERS(n) ((n) << 8)
#define DEC_DEBUG_NO_CLEAN (1 << 16)
#define DEC_DEBUG_NO_ATTN (1 << 17)    // the attention's scores and values left out (its output 0)
#define DEC_DEBUG_NO_KV (1 << 18)      // no K / V into the cache
#define DEC_DEBUG_QCHECK (1 << 19)     // every 64 values quantized both ways, the differences counted (DEC_PROF_QCHECK)
#define DEC_DEBUG_QSCALAR (1 << 20)    // the scalar quantizer
#define DEC_DEBUG_ATTN1 (1 << 21)      // attention the scalar way (a prompt's: a token at a time)
#define DEC_DEBUG_ATTN_F (1 << 22)     // a step's attention in floats (q . k, the values)
