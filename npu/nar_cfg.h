// npu/nar_cfg.h - YuE2's NAR (flow matching's velocity) as the engine hands it to the DSP (npu/dsp/gemma.c nar_*):
// every layer of one velocity there - the norms, RoPE, the rows' keys and values into their bank, the attention over
// the prefix's keys and the rows' own, SiLU, the residual adds on HVX; the products on HMX over fp16 weights in HMX's
// blocks (nar_f16.bin, tools/pack_yue2.py --nar-f16: a layer's matrices in one buffer, each at its offset). The rows
// go through each layer whole (non-causal: every row's keys before any row attends), the products a slab of rows at a
// time. Plain C with fixed sizes: the same bytes on the CPU and on the DSP.
//
// Buffers the CPU allocates (gnpu_shared_alloc; mapped into the DSP while a call reads them - a long chunk's layers
// in groups whose weights and banks fit the DSP's address space, a call a group, NAR_RUN_MORE on all but the last):
//   W[l]  layer l's weights (HMX_F16 blocks: qkv fused, o, gate, up, down)
//   P     the norms' weights (floats; q_norm already times the queries' scale, log2 e / sqrt(hd): ATTN_LOG2), and per
//         layer the prefix's last A % 32 positions' keys and values as the rows' bank's first block of each KV head
//         (fp16, the bank's layout: nkv x dgs x 2 KB keys, then nkv x dgs x 2 KB values)
//   IO    x [n][hidden] floats - the rows (the CPU's input rows in, the last layer's out); rope [n][2][hd / 2] floats -
//         each row's cosines then sines; then the DSP's own scratch (nar_layout), the calls' times in its last 256 bytes
//   the banks (attn_cfg.h ATTN_ENCODER): per layer the prefix's whole blocks (keys, values), and one the layers share
//         for the rows: the prefix's last A % 32 positions, then the n rows' (rewritten every layer)
#pragma once

#include <stdint.h>

#define NAR_VERSION 5
#define NAR_MAX_LAYERS 32
#define NAR_SLAB 4096     // rows a product at most (the slab: the rows, up to this - a product's weights are read
                          // once a pass of row tiles as VTCM holds, the slab never cutting a pass short)
#define NAR_THREADS 5     // HVX workers (v79's six contexts: these and the FastRPC thread, which holds HMX)
// what a call computes: one velocity (the NAR's layers over the prefix's banks), or the prefix itself - the AR's
// layers over the chunk's A positions (n = A, causal, the queries' scale in q_norm as the NAR's), every layer's
// keys and values written into the banks the NAR then reads (the whole blocks, then the A % 32 rest in RB); the
// last layer stops after its keys and values
#define NAR_MODE_VELOCITY 0
#define NAR_MODE_PREFIX 1

typedef struct {
    int32_t w_fd;                          // W[l]
    int32_t pk_fd, pv_fd;                  // the prefix's banks (npre blocks a KV head); unused when npre is 0
    int32_t sh_qkv, sh_o, sh_gu, sh_down;  // each matrix's out_shift (2^shift: the scale its values carry)
    uint64_t qkv, o, gate, up, down;       // bytes into W[l]
    uint32_t in_norm, mlp_norm;            // bytes into P: hidden floats each
    uint32_t q_norm, k_norm;               // hd floats each
    uint32_t rem;                          // bytes into RB: the A % 32 positions' blocks (2 x nkv x dgs x 2 KB)
    uint32_t wfmt;                         // the matrices' blocks: 0 fp16 (HMX_F16), 1 int8 + scales (HMX_Q8)
} nlayer_t;

typedef struct {
    int32_t version, hidden, n_layers, nh, nkv, hd, ffn;
    float eps;
    int32_t A, n;                  // the prefix's positions, the rows (A + n keys); a prefix: 0, A
    int32_t npre, rem;             // the prefix's whole blocks, A % 32 (in the rows' bank's first block)
    int32_t tk_fd, tv_fd;          // the rows' bank (rem + n positions; a prefix: none)
    int32_t attn_flags;            // more attn_cfg.h flags (ATTN_HELPERS(...) taken from NAR_THREADS)
    int32_t mode;                  // NAR_MODE_*
    int32_t rb_fd;                 // RB: per layer the A % 32 positions' blocks (written by a prefix, read by velocities)
    uint32_t p_norms;              // bytes of P the norms take (their qf32 copy)
    uint64_t io_x, io_rope;        // bytes into IO
    uint64_t scratch, io_bytes;    // the scratch's start, IO's size (its last 256 bytes: the times)
    nlayer_t layers[NAR_MAX_LAYERS];
} nar_cfg_t;

// the times in IO's last 256 bytes (uint64 microseconds, summed over the calls)
enum { NAR_PROF_IN, NAR_PROF_QKV, NAR_PROF_KV, NAR_PROF_ATTN, NAR_PROF_CORE, NAR_PROF_O, NAR_PROF_POST, NAR_PROF_MLP,
       NAR_PROF_DOWN, NAR_PROF_CACHE, NAR_PROF_CALLS,
       // the products' insides (npu_imp.c g_gemm_tq, microseconds): the GLU's share of MLP; the activations copied in;
       // the multiplier waiting for a weight block; HMX's tiles; the result croutons out; the GLU's multiplier waiting
       NAR_PROF_GLU, NAR_PROF_G_ACOPY, NAR_PROF_G_WAIT, NAR_PROF_G_TILE, NAR_PROF_G_OUT, NAR_PROF_G_GLUWAIT,
       NAR_PROF_N = 32 };

// nar_run's flags
#define NAR_DEBUG_LAYERS(n) ((n) & 0xFF)   // only the first n layers (0: all)
// experiments (timing only: the results wrong): the keys not written into the bank; the q / k norms and RoPE left out
#define NAR_X_NOPUTK 0x100
#define NAR_X_NONORM 0x200
// (the products' result croutons not copied out: how long HMX waits for them)
#define NAR_X_NOOUT 0x800
// (the q / k norms each head's own way through a scalar square root, as before norm_heads: A/B)
#define NAR_X_OLDNORM 0x1000
// (an odd position pair into the banks 16 bits at a time, as before the masked stores: A/B)
#define NAR_X_OLDPUT 0x4000
// the model's layers in several calls (cfg's layers one group of them): more follow this one - a prefix's last
// layer of the call goes through whole (only the model's last stops after its keys and values)
#define NAR_RUN_MORE 0x400

// The scratch (bytes into IO, each 4 KB aligned): every row's queries and attention output (floats), and for a slab:
// the products' input tiles (A), the fused q k v's result croutons, o's, the MLP's middle, down's; a thread's rows
typedef struct {
    uint64_t QR, CR, A, QKV, OO, MID, D, TMP, end;
    uint64_t m;   // the rows rounded up to 32
} nar_layout_t;

static inline uint64_t nar_up4k(uint64_t x) { return (x + 4095u) & ~(uint64_t)4095u; }

static inline nar_layout_t nar_layout(const nar_cfg_t *c) {
    nar_layout_t L;
    const uint64_t H = (uint64_t)c->hidden, qd = (uint64_t)c->nh * (uint64_t)c->hd, kd = (uint64_t)c->nkv * (uint64_t)c->hd;
    L.m = ((uint64_t)c->n + 31u) & ~(uint64_t)31u;
    const uint64_t S = L.m < NAR_SLAB ? L.m : NAR_SLAB, W = qd + 2u * kd, kin = H > qd ? H : qd;
    const uint64_t tw = H > qd ? H : qd;   // a thread's row: the widest of x, a head's rows (qd >= kd)
    uint64_t at = nar_up4k(c->scratch);
    L.QR = at, at = nar_up4k(at + L.m * qd * 4u);
    L.CR = at, at = nar_up4k(at + L.m * qd * 4u);
    L.A = at, at = nar_up4k(at + S * kin * 2u);
    L.QKV = at, at = nar_up4k(at + S * W * 2u);
    L.OO = at, at = nar_up4k(at + S * H * 2u);
    L.MID = at, at = nar_up4k(at + S * (uint64_t)c->ffn * 2u);
    L.D = at, at = nar_up4k(at + S * H * 2u);
    L.TMP = at, at = nar_up4k(at + (uint64_t)(NAR_THREADS + 1) * 2u * tw * 4u);   // (and the FastRPC thread's)
    L.end = at + 256u;   // the times
    return L;
}
