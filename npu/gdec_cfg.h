// npu/gdec_cfg.h - Gemma 4's layers as the engine hands them to the DSP (npu/dsp/gemma.c gdec_*): a prompt's chunk
// through every layer there - the norms, RoPE, the KV cache's writes, attention, GELU, the per-layer inputs on HVX,
// the products on HMX over the matrices' hmx4 copies (libgeode_npu.so's adopted ones, mapped once). Plain C with
// fixed sizes: the same bytes on the CPU and on the DSP.
//
// Buffers the CPU allocates (rpcmem, mapped into the DSP once):
//   P   the norms' weights (floats), each at its offset
//   IO  what a call reads and leaves (the CPU writes the inputs before it, reads x after it):
//       x [GDEC_MAX_N][hidden] floats - the chunk's residual rows, the embeddings in, the last layer's output out
//       ple [GDEC_MAX_N][layers][ple_dim] floats - the per-layer inputs (the CPU's ple_)
//       rope[r] [GDEC_MAX_N][2][pairs[r]] floats - each token's cosines then sines, for the two kinds of layer
//       then the DSP's own scratch (the tiles, the products' results, the attention's rows), and in its last 256
//       bytes the calls' time by phase (uint64 microseconds, GDEC_PROF_*)
// The KV cache's chunks: as attn_cfg.h's, each handed over once (gdec_kv).
#pragma once

#include <stdint.h>

#define GDEC_VERSION 1
#define GDEC_MAX_LAYERS 48
#define GDEC_MAX_N 1024    // tokens a call at most
#define GDEC_MAX_DOWN 4    // the down projection's parts (adopted in column parts: Model::Npu::Parts)
#define GDEC_THREADS 5     // HVX workers (v79's six contexts: these and the FastRPC thread, which holds HMX)

typedef struct {
    int32_t fd;      // the hmx4 copy (gnpu_mat_hmx): an rpcmem buffer the DSP maps at open
    int32_t shift;   // geode_npu_gemm's out_shift, HMX4_Q4I (0x100) included
    int32_t k, n;    // the depth HMX multiplies (the inputs, padded to 512 at least), the outputs
} gmat_t;

typedef struct {
    int32_t full, own_kv, slot, hd, rope;   // rope: which table (0 the sliding layers', 1 the full ones')
    float scalar;
    uint32_t in_norm, post_attn, pre_ff, post_ff, ple_norm;   // bytes into P: hidden floats each
    uint32_t q_norm, k_norm;                                  // hd floats each
    int32_t n_down;
    gmat_t q, k, v, o, gate, up, ple_gate, ple_proj;
    gmat_t down[GDEC_MAX_DOWN];   // parts of ffn / n_down inputs each, their results added
} glayer_t;

typedef struct {
    int32_t version, hidden, n_layers, nh, nkv, ple_dim, window;
    float eps;
    int32_t n_attn, kv_chunk, kv_row, kv_heads;   // the KV cache (attn_cfg.h)
    int32_t pairs[2];                             // the rope tables' pairs (sliding, full)
    uint64_t io_x, io_ple, io_rope[2];            // bytes into IO
    uint64_t scratch, io_bytes;                   // the scratch's start, IO's size (its last 256 bytes: the times)
    glayer_t layers[GDEC_MAX_LAYERS];
} gdec_cfg_t;

// the times in IO's last 256 bytes
enum { GDEC_PROF_IN, GDEC_PROF_QKV, GDEC_PROF_ATTN, GDEC_PROF_CORE, GDEC_PROF_POST, GDEC_PROF_DOWN, GDEC_PROF_PLE,
       GDEC_PROF_PLE2, GDEC_PROF_PRODUCTS, GDEC_PROF_MLP, GDEC_PROF_CALLS, GDEC_PROF_CACHE,
       // within PRODUCTS / MLP: q k v, o, the PLE's two; gate and up (the rest of MLP: down)
       GDEC_PROF_P_QKV, GDEC_PROF_P_O, GDEC_PROF_P_PLE, GDEC_PROF_GLU,
       // the workers' waits for an HVX context, their time in the steps (summed), the tasks each thread took
       GDEC_PROF_LOCKW, GDEC_PROF_BUSY, GDEC_PROF_TASKS0 = 20, GDEC_PROF_N = 32 };

// gdec_prefill's flags
#define GDEC_DEBUG_LAYERS(n) ((n) & 0xFF)   // only the first n layers (0: all)
// experiments (0: the defaults, gemma.c GD_TASK / GD_PF): row pairs a task of the steps between the products (1, 2,
// 4, 8 or 16: within a row tile), what is fetched into L2 ahead (1: nothing, 2: a task's inputs as it starts, 3: the
// next task's too), the FastRPC thread kept out of the pool
#define GDEC_TASK(p) (((p) & 0xF) << 8)
#define GDEC_TASK_OF(f) (((f) >> 8) & 0xF)
#define GDEC_PF(m) (((m) & 3) << 12)
#define GDEC_PF_OF(f) (((f) >> 12) & 3)
#define GDEC_NO_MAIN 0x4000
#define GDEC_ATTN_IEEE 0x8000   // the attention's softmax in IEEE sf (ATTN_SOFTMAX_IEEE)
// the phases (1 << gemma.c's GP_*) on the IEEE steps the qf32 ones replaced (a check)
#define GDEC_OLD(mask) (((mask) & 0xFF) << 16)
#define GDEC_OLD_OF(f) (((f) >> 16) & 0xFF)

// The scratch (bytes into IO, each 4 KB aligned), sized for GDEC_MAX_N tokens: the products' input tiles (A), the
// MLP's middle (MID, parts-major), the products' result croutons (q, k, v; o and the PLE projection; the down
// projection's parts; the PLE gate), the attention's query and output rows (floats), a thread's two rows (floats)
typedef struct {
    uint64_t A, MID, OQ, OK, OV, OO, D, G, QR, CR, TMP, end;
    uint64_t kmax, qn, kn, ffn, tmpw;   // the widest input, q, k / v, MLP middle, thread row
} gdec_layout_t;

static inline uint64_t gdec_up4k(uint64_t x) { return (x + 4095u) & ~(uint64_t)4095u; }

static inline gdec_layout_t gdec_layout(const gdec_cfg_t *c) {
    gdec_layout_t L;
    const uint64_t N = GDEC_MAX_N, H = (uint64_t)c->hidden;
    L.kmax = 512, L.qn = 0, L.kn = 0, L.ffn = 0, L.tmpw = H;
    for (int l = 0; l < c->n_layers && l < GDEC_MAX_LAYERS; ++l) {
        const glayer_t *y = &c->layers[l];
        const int32_t ks[4] = {y->q.k, y->o.k, y->ple_gate.k, y->ple_proj.k};
        for (int i = 0; i < 4; ++i)
            if ((uint64_t)ks[i] > L.kmax) L.kmax = (uint64_t)ks[i];
        if ((uint64_t)y->q.n > L.qn) L.qn = (uint64_t)y->q.n;
        if ((uint64_t)y->k.n > L.kn) L.kn = (uint64_t)y->k.n;
        if ((uint64_t)y->gate.n > L.ffn) L.ffn = (uint64_t)y->gate.n;
    }
    if (L.kn > L.tmpw) L.tmpw = L.kn;
    uint64_t at = gdec_up4k(c->scratch);
    L.A = at, at = gdec_up4k(at + N * L.kmax * 2u);
    L.MID = at, at = gdec_up4k(at + N * L.ffn * 2u);
    L.OQ = at, at = gdec_up4k(at + N * L.qn * 2u);
    L.OK = at, at = gdec_up4k(at + N * L.kn * 2u);
    L.OV = at, at = gdec_up4k(at + N * L.kn * 2u);
    L.OO = at, at = gdec_up4k(at + N * H * 2u);
    L.D = at, at = gdec_up4k(at + (uint64_t)GDEC_MAX_DOWN * N * H * 2u);
    L.G = at, at = gdec_up4k(at + N * 512u * 2u);
    L.QR = at, at = gdec_up4k(at + N * L.qn * 4u);
    L.CR = at, at = gdec_up4k(at + N * L.qn * 4u);
    L.TMP = at, at = gdec_up4k(at + (uint64_t)(GDEC_THREADS + 1) * 2u * L.tmpw * 4u);   // (and the FastRPC thread's)
    L.end = at + 256u;   // the times
    return L;
}
