// npu/attn_cfg.h - a prompt's attention for one layer as the engine hands it to the DSP (npu/dsp/attn.c): the chunk's
// queries against the KV cache, on HMX. Plain C with fixed sizes: the same bytes on the CPU and on the DSP.
//
// The queries: n rows of floats (q_stride apart), each nh heads of hd values, normed and turned (RoPE) already.
// The KV cache: the engine's int8 chunks (Model::kv_row_bytes, kv_scale), each mapped into the DSP once (kv_fd):
//   chunk c holds positions [c kv_chunk, (c + 1) kv_chunk): [K, V][layer][kv_chunk positions][kv_row] int8 codes,
//   then the scales [K, V][layer][kv_chunk positions][kv_heads] floats; KV head h of a layer is the row's bytes
//   [h hd, (h + 1) hd). Positions up to apos + n - 1 are written (the chunk's own too) before the call.
// The output: n rows of floats (o_stride apart), each nh heads' hd values: softmax(q k) v (scale 1: Gemma 4's q and k
// are normed), a token seeing the positions up to its own (the last `window` of them on a sliding layer).
#pragma once

#include <stdint.h>

#define ATTN_VERSION 1
#define ATTN_MAX_CHUNKS 256
#define ATTN_MAX_ROWS 16384   // n x nh at most (a row: a token's query head)

typedef struct {
    int32_t version;
    int32_t n, apos;              // the chunk's tokens, the first one's position
    int32_t nh, nkv, hd;          // query heads, KV heads (nh / nkv divides 32), this layer's head dim (128 k, <= 512)
    int32_t window;               // a sliding layer's positions a token sees, itself included; 0: every one before
    int32_t slot, n_attn;         // the layer's place among the KV cache's layers, their number
    int32_t kv_chunk;             // positions a chunk
    int32_t kv_row;               // bytes of a position's int8 row
    int32_t kv_heads;             // scales a position
    int32_t n_chunks;             // the chunks holding positions 0 .. apos + n - 1
    int32_t q_stride, o_stride;   // floats from a token's row to the next one's (nh hd at least, multiples of 32)
    int32_t flags;                // ATTN_*
    int32_t kv_fd[ATTN_MAX_CHUNKS];
} attn_desc_t;

// flags
#define ATTN_HELPERS(h) ((h) & 7)        // HVX helper threads (0: the default, 4)
#define ATTN_MAX_SB(p) ((uint32_t)(p) << 8)   // finding faults / testing the joins: at most p x 32 positions a
                                              // superblock (0: as many as VTCM holds)
#define ATTN_SB_OF(f) (((uint32_t)(f) >> 8) & 0xFFFF)
#define ATTN_DEBUG_NO_HMX (1u << 24)     // the products left out (the softmax and the rest timed alone)
#define ATTN_SOFTMAX_IEEE (1u << 25)     // the softmax in IEEE sf as it was (a check of the hf one)
#define ATTN_DEBUG_SMCHECK (1u << 26)    // both softmaxes on each tile, the first difference in usec (debugging)
// An encoder's attention (Gemma 4's vision tower): apos 0, every token sees all n, and the keys and values are fp16
// HMX blocks the host made - kv_fd[0] the keys, per KV head [block of 32 positions][hd / 32] 2 KB weight blocks
// (the block's positions its outputs, 32 dims its inputs: element (p, c) at (c / 2) 64 + p 2 + c % 2), kv_fd[1] the
// values, per KV head [hd / 32][block] 2 KB weight blocks (32 dims the outputs, the positions the inputs: element
// (c, p) at (p / 2) 64 + c 2 + p % 2); positions n .. up to a multiple of 32 zero. n_chunks 2; hd a multiple of 32;
// slot, n_attn, kv_chunk, kv_row, kv_heads unused.
#define ATTN_ENCODER (1u << 27)
