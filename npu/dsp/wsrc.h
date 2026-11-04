// npu/dsp/wsrc.h - a matrix's weights as the products' pipe workers read them (npu_imp.c), shared with gemma.c: the
// hmx4 blocks of a copy made for the NPU, or a q4i tile8 matrix the CPU's kernels read too - one copy for both
// (geode_npu_mat_t8 registers it; its id is what gemm / gemm_glu / the layers' cfg name with HMX4_T8).
#pragma once

#include <stdint.h>

#include "AEEStdDef.h"
#include "remote.h"

// a gemm's out_shift with this bit: w_off (gemm) / g_fd, u_fd (gemm_glu) / gmat_t.fd (gdec) name a registered tile8
// matrix (geode_npu_mat_t8), not an hmx4 copy's buffer (npu_api.c GNPU_SHIFT_T8)
#define HMX4_T8 0x200
// a gemm's out_shift with this bit: the weights are fp16 already in HMX's weight order (wgt_index), k x 64 bytes a
// 32-column block - copied into the slots as they are, nothing expanded (the vision encoder's q8 layers, converted by
// the host a layer at a time: npu_api.c gnpu_pack_f16 / GNPU_SHIFT_F16)
#define HMX_F16 0x400

// A q4i tile8 matrix (engine kernels.hpp to_tile8; levels code - 8, signed E4M3 scales, one a 16 inputs): row tile b
// (32 rows) at w + b tile, its codes first (per 64-input chunk 8 vectors of 128 bytes, lane = row) then its scales (per
// chunk a vector, [quad][block][row]). A product reads inputs col0 .. col0 + k of it (a deep matrix in parts); its fp16
// scales are e4m3 x up (the global scale times 2^shift, as the hmx4 copy's), HMX's convert takes 2^shift out.
typedef struct {
    const uint8_t *w;
    uint32_t tile;             // bytes a row tile: 18 kfull
    uint32_t kfull, col0, k;   // the matrix's inputs, this part's first, its count (multiples of 64)
    uint32_t n;                // outputs (a multiple of 32)
    float up;
    int fd;                    // the buffer (one HAP_mmap_get each, put back by geode_npu_mat_t8_clear)
} t8_t;

// the registered matrix `id` (1-based), NULL when there is none
const t8_t *geode_npu_t8_get(int id);

// O = A x W^T as geode_npu_gemm_ptr, the weights hmx4 blocks at w (t8 NULL) or the tile8 matrix t8 (w unused).
// out_shift as gemm's, HMX4_Q4I / HMX4_T8 included
int geode_npu_gemm_src(remote_handle64 h, const uint8_t *a, const uint8_t *w, const t8_t *t8, uint8_t *o, int m, int k,
                       int n, int out_shift);
// gemm_glu on mapped buffers, each matrix hmx4 (its t8 NULL) or tile8
int geode_npu_glu_src(remote_handle64 h, const uint8_t *a, const uint8_t *gw, const uint8_t *uw, const t8_t *gt8,
                      const t8_t *ut8, uint8_t *o, int m, int k, int n, int g_shift, int u_shift, int part_blk, int act);
