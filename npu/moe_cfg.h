// A prompt's routed experts on the NPU (geode_npu_moe, npu/dsp/npu_imp.c): what the CPU hands over. The same bytes on
// both sides, so plain fixed-size fields only.
//
//   desc = moe_hdr_t, then int32 fds[n_fds] (rounded up to an even count), then moe_exp_t[n_experts], then
//          moe_pick_t[n_picks]
//   fds  = mapped buffers holding the experts' matrices in tile8 (the pack's layout, the GPU's: 32-row tiles of
//          18 x cols bytes - the codes first, then a vector of E4M3 scales per 64 inputs): the expert cache's slabs
//   io   = a mapped (cached) buffer: x [n_tok][hidden] floats (the experts' input, normed), y [n_tok][hidden] floats
//          (each pick's expert output times its weight added into its token's row)
//
// An expert: gate|up (2 ff x hidden: the gate's ff rows, then the up's), down (hidden x ff); y[tok] += weight x
// down(silu(gate x) * (up x)) for each of its picks.
#ifndef GEODE_MOE_CFG_H
#define GEODE_MOE_CFG_H

#include <stdint.h>

#define MOE_VERSION 2
#define MOE_MAX_FDS 64
#define MOE_ZERO_Y 1   // flags: y zeroed first
// measuring: parts left out (their results wrong)
#define MOE_DEBUG_NO_GATHER 2
#define MOE_DEBUG_NO_SCATTER 4
#define MOE_DEBUG_NO_HMX 8
#define MOE_DEBUG_NO_SWIGLU 16

typedef struct {
    uint32_t version;
    uint32_t n_experts, n_picks;
    uint32_t hidden, ff;   // hidden: a multiple of 64 up to 4096; ff: of 64 up to 2048
    uint32_t n_tok;
    uint32_t x_at, y_at;   // bytes into io (128-byte aligned)
    uint32_t flags;
    uint32_t workers;      // the threads expanding the weights (0: the default)
    uint32_t n_fds;
    uint32_t pad;
} moe_hdr_t;

typedef struct {
    uint64_t gu_off, dn_off;      // bytes into its buffer (128-byte aligned)
    float gu_global, dn_global;   // the matrices' global scales
    uint32_t first, count;        // its picks
    uint32_t fd;                  // its buffer: an index into fds
    uint32_t pad;
} moe_exp_t;

typedef struct {
    uint32_t tok;
    float weight;
} moe_pick_t;

#endif
