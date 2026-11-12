// yue2_nar.cpp - YuE2's NAR on the NPU (yue2_nar.hpp): one velocity = the CPU's input rows, 28 layers of HMX
// products and HMX attention with the CPU's norms / RoPE / SwiGLU between them, the CPU's output rows.
#include "yue2_nar.hpp"
#include "yue2_nar_gpu.hpp"
#include "yue2_q8gemm.hpp"
#include "yue2_q4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__ANDROID__)
#include <dlfcn.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "platform.hpp"
#include "../../npu/attn_cfg.h"
#include "../../npu/nar_cfg.h"
#include "../../npu/host/npu_api.h"

namespace geode::yue2 {
namespace {

typedef float v8 __attribute__((vector_size(32)));

inline float bf16f(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint16_t f_bf16(float f) {   // round to nearest even (finite values)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float r16(float f) { return static_cast<float>(static_cast<_Float16>(f)); }   // through fp16
inline v8 load8(const float *p) {
    v8 r;
    std::memcpy(&r, p, sizeof r);
    return r;
}
inline void store8(float *p, v8 v) { std::memcpy(p, &v, sizeof v); }
inline float hsum(v8 v) { return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7])); }
inline float dot(const float *a, const float *b, int n) {   // n % 16 == 0
    v8 s0 = {}, s1 = {};
    for (int i = 0; i < n; i += 16) {
        s0 += load8(a + i) * load8(b + i);
        s1 += load8(a + i + 8) * load8(b + i + 8);
    }
    return hsum(s0 + s1);
}

// the oracle's (yue2.cpp): RMSNorm with the mean in double, SiLU, RoPE's rotate_half
void rms_row(const float *x, const float *w, float *out, int n, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
    const float r = 1.0f / std::sqrt(static_cast<float>(s / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] * r) * w[i];
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline void rope_head(float *x, const float *cos, const float *sin, int hd) {
    const int h = hd / 2;
    for (int i = 0; i < h; ++i) {
        const float a = x[i], b = x[i + h];
        x[i] = a * cos[i] - b * sin[i];
        x[i + h] = b * cos[i] + a * sin[i];
    }
}

// ---- the banks (npu/attn_cfg.h ATTN_ENCODER): per KV head, keys [block][hd / 32] 2 KB blocks (element (p, c) at
// (c / 2) 64 + p 2 + c % 2 within its dim group's block), values [hd / 32][block] (element (c, p) at
// (p / 2) 64 + c 2 + p % 2); nb blocks a head
inline size_t k_at(size_t nb, int dgs, int kh, int p, int c) {
    return ((static_cast<size_t>(kh) * nb + static_cast<size_t>(p / 32)) * static_cast<size_t>(dgs) + static_cast<size_t>(c / 32)) * 1024u +
           static_cast<size_t>((c % 32) / 2) * 64u + static_cast<size_t>(p % 32) * 2u + static_cast<size_t>(c & 1);
}
inline size_t v_at(size_t nb, int dgs, int kh, int p, int c) {
    return ((static_cast<size_t>(kh) * static_cast<size_t>(dgs) + static_cast<size_t>(c / 32)) * nb + static_cast<size_t>(p / 32)) * 1024u +
           static_cast<size_t>((p % 32) / 2) * 64u + static_cast<size_t>(c % 32) * 2u + static_cast<size_t>(p & 1);
}
// position p's key and value rows of KV head kh (hd floats each) into a bank of nb blocks a head
inline void put_kv(_Float16 *K, _Float16 *V, size_t nb, int dgs, int kh, int p, const float *k, const float *v, int hd) {
    for (int c = 0; c < hd; c += 2) {
        _Float16 *o = K + k_at(nb, dgs, kh, p, c);
        o[0] = static_cast<_Float16>(k[c]);
        o[1] = static_cast<_Float16>(k[c + 1]);
    }
    for (int c = 0; c < hd; ++c) V[v_at(nb, dgs, kh, p, c)] = static_cast<_Float16>(v[c]);
}

// ---- the emulation of the DSP's calls (no libgeode_npu.so: the PC), and the CPU backend on its buffers -----------
// Two roles. The emulation (npu_dir ""): the DSP's arithmetic as closely as the CPU can - fp16 inputs and outputs,
// a decoded copy of every matrix - the layouts checked against the oracle off the phone. The CPU backend (npu_dir
// "cpu", a phone without HMX): the same blocks and banks, the products read straight from the fp16 blocks (nothing
// decoded kept), the layers' weights borrowed from the pack's mapping, fp32 throughout.
struct Emu {
    ThreadPool *pool = nullptr;
    bool fast = false;   // the CPU backend
    std::vector<AlignedBuffer> bufs;
    std::vector<size_t> lens;
    std::vector<std::pair<const uint8_t *, size_t>> views;   // borrowed memory (the pack's mapping): fds kViewFd + i
    std::map<std::pair<int, uint64_t>, std::vector<uint16_t>> dec;   // hmx16 matrices read back as bf16 [rows][cols]
    AlignedBuffer ab[2];
    size_t ab_len[2] = {0, 0};
    NarGpu *gpu = nullptr;   // the GPU backend's products (npu_dir "gpu"), the rest the CPU backend's
    std::string gpu_err;
    double gpu_share = 0.6;   // of a product's rows the GPU takes (gpu_gemm_f16; GEODE_YUE2_NAR_GPU_SHARE fixes it)
    bool gpu_adapt = true;
    std::vector<float> kt, vt;   // the CPU backend's attention: a layer's keys (in tiles, transposed) and values
    // the scores in int8 (attn8): the keys less their head's mean over the keys (softmax does not see a row's constant
    // q . mean), each key's amax / 127 its scale - k8 [nkv][tiles][hd / 4][kKT][4] (four channels of a key together:
    // SDOT's lanes), ks [nkv][tiles * kKT]
    bool attn8 = false, attn8_opt = false;   // (attn8: this layer's; attn8_opt: asked for)
    // an fp16 matrix (its buffer's fd, byte offset) -> its hmx8 twin: the CPU's share of its products by SDOT
    std::map<std::pair<int, uint64_t>, const uint8_t *> q8;
    bool q8rot = false;   // the twins Hadamard-turned (c8.*: yue2_q4.hpp make_nar_q8 rotated) - the activations turned too
    std::vector<int8_t> k8;
    std::vector<float> ks, kmean;
};
constexpr int kViewFd = 1 << 20;
Emu *emu_of(gnpu *g) { return reinterpret_cast<Emu *>(g); }
int emu_view(Emu *e, const uint8_t *p, size_t len) {
    e->views.emplace_back(p, len);
    return kViewFd + static_cast<int>(e->views.size()) - 1;
}

void *emu_alloc(gnpu *g, size_t bytes, int, int *fd) {
    Emu *e = emu_of(g);
    AlignedBuffer b = aligned_bytes(bytes);
    if (!b) return nullptr;
    std::memset(b.get(), 0x3C, bytes);   // (rpcmem is not cleared: neither is this - finite garbage, 1.06 as fp16)
    e->bufs.push_back(std::move(b));
    e->lens.push_back(bytes);
    *fd = static_cast<int>(e->bufs.size());
    return e->bufs.back().get();
}
uint8_t *emu_buf(Emu *e, int fd, size_t *len) {
    if (fd >= kViewFd && static_cast<size_t>(fd - kViewFd) < e->views.size()) {
        if (len) *len = e->views[static_cast<size_t>(fd - kViewFd)].second;
        return const_cast<uint8_t *>(e->views[static_cast<size_t>(fd - kViewFd)].first);   // (only ever read)
    }
    if (fd < 1 || static_cast<size_t>(fd) > e->bufs.size()) return nullptr;
    if (len) *len = e->lens[static_cast<size_t>(fd - 1)];
    return e->bufs[static_cast<size_t>(fd - 1)].get();
}

// ---- the CPU backend's product: y[t] = W x[t] for W [rows][cols] in HMX's fp16 blocks (pack_yue2.py hmx16_blocks:
// per 32 rows, per 32 columns a 2 KB block, element (o, c) at (c % 32 / 2) 64 + o 2 + c % 2), values 2^shift larger.
// A worker takes 32-row blocks; per block a panel of kPanel columns decoded to fp32, column-major (the 32 rows of a
// column together), then four rows of x at a time against it, the block's sums for every row of x kept as it goes.
constexpr int kPanel = 256;

void decode_panel(const _Float16 *blk, int64_t c0, int pc, float scale, float *panel) {
    for (int c = 0; c < pc; c += 2) {
        const int64_t cc = c0 + c;
        const _Float16 *pr = blk + (cc / 32) * 1024 + ((cc % 32) / 2) * 64;   // the pair's 32 rows, (c, c + 1) each
        float *d0 = panel + static_cast<size_t>(c) * 32u, *d1 = d0 + 32;
#if defined(__aarch64__)
        const float32x4_t sc = vdupq_n_f32(scale);
        for (int o = 0; o < 32; o += 8) {
            const float16x8x2_t v = vld2q_f16(reinterpret_cast<const float16_t *>(pr + 2 * o));
            vst1q_f32(d0 + o, vmulq_f32(vcvt_f32_f16(vget_low_f16(v.val[0])), sc));
            vst1q_f32(d0 + o + 4, vmulq_f32(vcvt_high_f32_f16(v.val[0]), sc));
            vst1q_f32(d1 + o, vmulq_f32(vcvt_f32_f16(vget_low_f16(v.val[1])), sc));
            vst1q_f32(d1 + o + 4, vmulq_f32(vcvt_high_f32_f16(v.val[1]), sc));
        }
#else
        for (int o = 0; o < 32; ++o) d0[o] = static_cast<float>(pr[2 * o]) * scale, d1[o] = static_cast<float>(pr[2 * o + 1]) * scale;
#endif
    }
}

// acc[r][0..32) += panel (pc columns) x[r][0..pc) for the four rows r (x already at the panel's first column)
void panel_rows4(const float *panel, int pc, const float *const x[4], float *const acc[4]) {
#if defined(__aarch64__)
    for (int h = 0; h < 32; h += 16) {
        float32x4_t a[4][4];
        for (int r = 0; r < 4; ++r)
            for (int q = 0; q < 4; ++q) a[r][q] = vld1q_f32(acc[r] + h + 4 * q);
        for (int c = 0; c < pc; c += 4) {
            const float32x4_t x0 = vld1q_f32(x[0] + c), x1 = vld1q_f32(x[1] + c), x2 = vld1q_f32(x[2] + c),
                              x3 = vld1q_f32(x[3] + c);
#define GEODE_LANE(L)                                                                                  \
            {                                                                                          \
                const float *wp = panel + static_cast<size_t>(c + (L)) * 32u + h;                      \
                const float32x4_t w0 = vld1q_f32(wp), w1 = vld1q_f32(wp + 4), w2 = vld1q_f32(wp + 8),  \
                                  w3 = vld1q_f32(wp + 12);                                             \
                a[0][0] = vfmaq_laneq_f32(a[0][0], w0, x0, L), a[0][1] = vfmaq_laneq_f32(a[0][1], w1, x0, L); \
                a[0][2] = vfmaq_laneq_f32(a[0][2], w2, x0, L), a[0][3] = vfmaq_laneq_f32(a[0][3], w3, x0, L); \
                a[1][0] = vfmaq_laneq_f32(a[1][0], w0, x1, L), a[1][1] = vfmaq_laneq_f32(a[1][1], w1, x1, L); \
                a[1][2] = vfmaq_laneq_f32(a[1][2], w2, x1, L), a[1][3] = vfmaq_laneq_f32(a[1][3], w3, x1, L); \
                a[2][0] = vfmaq_laneq_f32(a[2][0], w0, x2, L), a[2][1] = vfmaq_laneq_f32(a[2][1], w1, x2, L); \
                a[2][2] = vfmaq_laneq_f32(a[2][2], w2, x2, L), a[2][3] = vfmaq_laneq_f32(a[2][3], w3, x2, L); \
                a[3][0] = vfmaq_laneq_f32(a[3][0], w0, x3, L), a[3][1] = vfmaq_laneq_f32(a[3][1], w1, x3, L); \
                a[3][2] = vfmaq_laneq_f32(a[3][2], w2, x3, L), a[3][3] = vfmaq_laneq_f32(a[3][3], w3, x3, L); \
            }
            GEODE_LANE(0) GEODE_LANE(1) GEODE_LANE(2) GEODE_LANE(3)
#undef GEODE_LANE
        }
        for (int r = 0; r < 4; ++r)
            for (int q = 0; q < 4; ++q) vst1q_f32(acc[r] + h + 4 * q, a[r][q]);
    }
#else
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < pc; ++c) {
            const float xv = x[r][c];
            const float *wp = panel + static_cast<size_t>(c) * 32u;
            for (int o = 0; o < 32; ++o) acc[r][o] += wp[o] * xv;
        }
#endif
}

void cpu_gemm_hmx16(const uint8_t *w, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                    float *const *ys, ThreadPool &pool) {
    const float scale = std::ldexp(1.0f, -shift);
    const int64_t blocks = rows / 32;
    pool.parallel_for(blocks, [&](int64_t b0, int64_t b1) {
        std::vector<float> panel(static_cast<size_t>(kPanel) * 32u), acc(static_cast<size_t>(n + 3) / 4u * 4u * 32u);
        std::vector<float> zero(static_cast<size_t>(kPanel), 0.0f), sink(32);
        for (int64_t blk = b0; blk < b1; ++blk) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            const _Float16 *src = reinterpret_cast<const _Float16 *>(w) + blk * cols * 32;
            for (int64_t c0 = 0; c0 < cols; c0 += kPanel) {
                const int pc = static_cast<int>(std::min<int64_t>(kPanel, cols - c0));
                decode_panel(src, c0, pc, scale, panel.data());
                for (int t = 0; t < n; t += 4) {
                    const float *x[4];
                    float *a[4];
                    for (int r = 0; r < 4; ++r) {   // (past the last row: zeros in, the sums thrown away)
                        const bool in = t + r < n;
                        x[r] = in ? xs[t + r] + c0 : zero.data();
                        a[r] = &acc[static_cast<size_t>(t + r) * 32u];
                    }
                    panel_rows4(panel.data(), pc, x, a);
                }
            }
            for (int t = 0; t < n; ++t)
                std::memcpy(ys[t] + blk * 32, &acc[static_cast<size_t>(t) * 32u], sizeof(float) * 32u);
        }
    }, 1);
}

// HMX's fp16 product: x rounded to fp16, the weights' blocks read back, the results rounded to fp16 (the CPU
// backend: the product above, fp32)
// The CPU's rows [row0, row0 + rows) of the product of the matrix at fd / off (base: its buffer): from its hmx8 twin
// by SDOT when it has one (8-bit sound: the activations in int8 too), else the fp16 product
void cpu_rows(Emu *e, const uint8_t *base, int fd, uint64_t off, int64_t row0, int64_t rows, int64_t cols, int shift,
              int n, const float *const *xs, float *const *ys) {
    if (!e->q8.empty()) {
        const auto it = e->q8.find({fd, off});
        if (it != e->q8.end()) {
            X8 x8;
            quant_x8(xs, n, cols, x8, *e->pool, e->q8rot);
            if (cpu_gemm_hmx8(it->second + (row0 / 32) * cols * 34, rows, cols, shift, x8, ys, *e->pool)) return;
        }
    }
    cpu_gemm_hmx16(base + off + static_cast<uint64_t>(row0 * cols * 2), rows, cols, shift, n, xs, ys, *e->pool);
}

int emu_gemm_f16(gnpu *g, int fd, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                 float *const *ys) {
    Emu *e = emu_of(g);
    size_t len = 0;
    const uint8_t *base = emu_buf(e, fd, &len);
    if (!base || rows % 32 || cols % 32 || cols < 512 || off % 128 || off + static_cast<uint64_t>(rows * cols * 2) > len)
        return -1;
    if (e->fast) {
        cpu_rows(e, base, fd, off, 0, rows, cols, shift, n, xs, ys);
        return 0;
    }
    auto &w = e->dec[{fd, off}];
    if (w.empty()) {
        w.resize(static_cast<size_t>(rows * cols));
        const _Float16 *src = reinterpret_cast<const _Float16 *>(base + off);
        const float down = std::ldexp(1.0f, -shift);
        e->pool->parallel_for(rows / 32, [&](int64_t a, int64_t b) {
            for (int64_t blk = a; blk < b; ++blk)
                for (int o = 0; o < 32; ++o)
                    for (int64_t c = 0; c < cols; ++c) {
                        const size_t i = static_cast<size_t>(blk * cols * 32) + static_cast<size_t>(c / 32) * 1024u +
                                         static_cast<size_t>((c % 32) / 2) * 64u + static_cast<size_t>(o) * 2u + static_cast<size_t>(c & 1);
                        w[static_cast<size_t>((blk * 32 + o) * cols + c)] = f_bf16(static_cast<float>(src[i]) * down);
                    }
        });
    }
    std::vector<float> x(static_cast<size_t>(n * cols)), y(static_cast<size_t>(n * rows));
    for (int t = 0; t < n; ++t)
        for (int64_t c = 0; c < cols; ++c) x[static_cast<size_t>(t * cols + c)] = r16(xs[t][c]);
    gemm_bf16(x.data(), n, cols, w.data(), rows, nullptr, y.data(), *e->pool);
    for (int t = 0; t < n; ++t)
        for (int64_t r = 0; r < rows; ++r) ys[t][r] = r16(y[static_cast<size_t>(t * rows + r)]);
    return 0;
}

// the GPU backend's product (yue2_nar_gpu.hpp): the matrix where the CPU backend reads it, in the layer's view
// The rows shared: the GPU the first share of the 32-row blocks, the CPU the rest at the same time (each idles
// otherwise: alone the GPU only beats the CPU 34 to 24 G MAC/s on the SD720G). The share follows the two's measured
// rates (the GPU's from its kernels' events), so the halves end together on any phone.
int gpu_gemm_f16(gnpu *g, int fd, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                 float *const *ys) {
    Emu *e = emu_of(g);
    size_t len = 0;
    const uint8_t *base = emu_buf(e, fd, &len);
    if (!base || !e->gpu || off + static_cast<uint64_t>(rows * cols * 2) > len) return -1;
    const int64_t blocks = rows / 32;
    const int64_t gb = std::clamp<int64_t>(std::llround(static_cast<double>(blocks) * e->gpu_share), 0, blocks);
    const int64_t rg = gb * 32;
    if (rg > 0 && !e->gpu->start(base, len, off, rg, cols, shift, n, xs, e->pool, e->gpu_err)) return -2;
    const double t0 = now_seconds();
    if (rg < rows) {
        std::vector<float *> yc(static_cast<size_t>(n));
        for (int t = 0; t < n; ++t) yc[static_cast<size_t>(t)] = ys[t] + rg;
        cpu_rows(e, base, fd, off, rg, rows - rg, cols, shift, n, xs, yc.data());
    }
    const double tc = now_seconds() - t0;
    double tg = 0.0;
    if (rg > 0 && !e->gpu->finish(ys, e->pool, e->gpu_err, &tg)) return -2;
    if (e->gpu_adapt && rg > 0 && rg < rows && tg > 0.0 && tc > 0.0) {
        const double vg = static_cast<double>(rg) / tg, vc = static_cast<double>(rows - rg) / tc;
        e->gpu_share = std::clamp(0.7 * e->gpu_share + 0.3 * vg / (vg + vc), 0.05, 0.95);
    }
    return 0;
}

// gnpu_mlp_f16: gate and up as HMX's products, silu(gate) * up rounded to fp16 (the DSP's in hf arithmetic), down
int emu_mlp_f16(gnpu *g, int fd, uint64_t g_off, uint64_t u_off, uint64_t d_off, int64_t hidden, int64_t ff, int gs,
                int us, int ds, int act, int n, const float *const *xs, float *const *ys) {
    if (act != 1) return -1;   // (silu only: the NAR's)
    std::vector<float> G(static_cast<size_t>(n * ff)), U(G.size());
    std::vector<float *> gp(static_cast<size_t>(n)), up(static_cast<size_t>(n));
    std::vector<const float *> ap(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) {
        gp[static_cast<size_t>(t)] = &G[static_cast<size_t>(t * ff)];
        up[static_cast<size_t>(t)] = &U[static_cast<size_t>(t * ff)];
        ap[static_cast<size_t>(t)] = gp[static_cast<size_t>(t)];
    }
    if (emu_gemm_f16(g, fd, g_off, ff, hidden, gs, n, xs, gp.data()) ||
        emu_gemm_f16(g, fd, u_off, ff, hidden, us, n, xs, up.data()))
        return -1;
    const bool fast = emu_of(g)->fast;
    for (size_t i = 0; i < G.size(); ++i) G[i] = fast ? silu(G[i]) * U[i] : r16(silu(G[i]) * U[i]);
    return emu_gemm_f16(g, fd, d_off, hidden, ff, ds, n, ap.data(), ys);
}

float *emu_attn_buf(gnpu *g, int which, size_t bytes) {
    Emu *e = emu_of(g);
    if (which < 0 || which > 1) return nullptr;
    if (e->ab_len[which] < bytes) {
        e->ab[which] = aligned_bytes(bytes);
        e->ab_len[which] = e->ab[which] ? bytes : 0;
    }
    return reinterpret_cast<float *>(e->ab[which].get());
}

// ---- the CPU backend's attention: flash-style, fp32 ---------------------------------------------------------------
// A task: one kv head x kRows / group tokens (their `group` query heads: kRows rows). The keys in tiles of kKT,
// transposed ([hd][kKT]: four rows of queries against sixteen keys a step), the values as rows; per tile the scores,
// the running max and sum (the softmax online, in log2 units), P V into the rows' sums. A tile stays in the caches:
// the keys and values are read once a task, not once a 4-token block as the emulation's order (kept for the DSP's
// roundings) reads them - that order is bandwidth-bound, ~11 G MAC/s on the SD720G.
typedef float v4 __attribute__((vector_size(16)));
inline v4 ld4(const float *p) {
    v4 r;
    std::memcpy(&r, p, sizeof r);
    return r;
}
inline void st4(float *p, v4 v) { std::memcpy(p, &v, sizeof v); }
constexpr int kKT = 64;    // keys a tile
constexpr int kRows = 64;  // query rows a task

// 2^x for x <= 0, every lane (to ~2 ulp: 2^f on [-1/2, 1/2] by its Taylor series to f^7; below -126: ~0)
inline v4 exp2v(v4 x) {
    const v4 lo = {-126.0f, -126.0f, -126.0f, -126.0f};
    x = x < lo ? lo : x;
    const v4 magic = {12582912.0f, 12582912.0f, 12582912.0f, 12582912.0f};   // 1.5 2^23: x rounded to nearest
    const v4 big = x + magic;
    const v4 r = big - magic;
    const v4 f = x - r;
    v4 p = f * 1.525273380405984e-05f + 1.5403530393381606e-04f;
    p = p * f + 1.3333558146428443e-03f;
    p = p * f + 9.618129107628477e-03f;
    p = p * f + 5.550410866482158e-02f;
    p = p * f + 2.402265069591007e-01f;
    p = p * f + 6.931471805599453e-01f;
    p = p * f + 1.0f;
    typedef int32_t i4 __attribute__((vector_size(16)));
    i4 bi, pi, mi;
    std::memcpy(&bi, &big, 16);
    std::memcpy(&pi, &p, 16);
    std::memcpy(&mi, &magic, 16);
    pi += (bi - mi) << 23;   // times 2^r
    v4 out;
    std::memcpy(&out, &pi, 16);
    return out;
}

// S[r][0..kKT) = q_r . the tile's keys, four rows x sixteen keys at a time (kt: the tile, [hd][kKT])
inline void tile_scores(const float *qs, int rows, int hd, const float *kt, float *S) {
    for (int r0 = 0; r0 < rows; r0 += 4) {
        const int nr = std::min(4, rows - r0);
        for (int k0 = 0; k0 < kKT; k0 += 16) {
            v4 a[4][4] = {};
            for (int dd = 0; dd < hd; ++dd) {
                const float *kr = kt + static_cast<size_t>(dd) * kKT + k0;
                const v4 k0v = ld4(kr), k1v = ld4(kr + 4), k2v = ld4(kr + 8), k3v = ld4(kr + 12);
                for (int r = 0; r < 4; ++r) {
                    const float qv = r < nr ? qs[static_cast<size_t>(r0 + r) * hd + dd] : 0.0f;
                    a[r][0] += k0v * qv, a[r][1] += k1v * qv, a[r][2] += k2v * qv, a[r][3] += k3v * qv;
                }
            }
            for (int r = 0; r < nr; ++r)
                for (int c = 0; c < 4; ++c) st4(S + static_cast<size_t>(r0 + r) * kKT + k0 + c * 4, a[r][c]);
        }
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
// The same scores from int8 (attn8): q8 [rows][hd] with scales sq, the tile's keys k8 [hd / 4][kKT][4] with scales
// ks [kKT]; four rows x sixteen keys a step, an SDOT a row's four channels against four keys (no horizontal sums)
inline void tile_scores_i8(const int8_t *q8, const float *sq, int rows, int hd, const int8_t *k8, const float *ks,
                           float *S) {
    const int8x16_t z = vdupq_n_s8(0);
    for (int r0 = 0; r0 < rows; r0 += 4) {
        const int nr = std::min(4, rows - r0);
        for (int k0 = 0; k0 < kKT; k0 += 16) {
            int32x4_t a[4][4];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) a[r][c] = vdupq_n_s32(0);
            for (int c16 = 0; c16 < hd; c16 += 16) {
                int8x16_t qv[4];
                for (int r = 0; r < 4; ++r) qv[r] = r < nr ? vld1q_s8(q8 + static_cast<size_t>(r0 + r) * hd + c16) : z;
#define GEODE_K8_LANE(g)                                                                                         \
    {                                                                                                            \
        const int8_t *kb = k8 + (static_cast<size_t>(c16 / 4 + (g)) * kKT + k0) * 4;                             \
        const int8x16_t x0 = vld1q_s8(kb), x1 = vld1q_s8(kb + 16), x2 = vld1q_s8(kb + 32), x3 = vld1q_s8(kb + 48); \
        for (int r = 0; r < 4; ++r) {                                                                            \
            a[r][0] = vdotq_laneq_s32(a[r][0], x0, qv[r], g);                                                    \
            a[r][1] = vdotq_laneq_s32(a[r][1], x1, qv[r], g);                                                    \
            a[r][2] = vdotq_laneq_s32(a[r][2], x2, qv[r], g);                                                    \
            a[r][3] = vdotq_laneq_s32(a[r][3], x3, qv[r], g);                                                    \
        }                                                                                                        \
    }
                GEODE_K8_LANE(0) GEODE_K8_LANE(1) GEODE_K8_LANE(2) GEODE_K8_LANE(3)
#undef GEODE_K8_LANE
            }
            for (int r = 0; r < nr; ++r) {
                const float32x4_t s = vdupq_n_f32(sq[r0 + r]);
                float *out = S + static_cast<size_t>(r0 + r) * kKT + k0;
                for (int c = 0; c < 4; ++c)
                    vst1q_f32(out + 4 * c, vmulq_f32(vmulq_f32(vcvtq_f32_s32(a[r][c]), vld1q_f32(ks + k0 + 4 * c)), s));
            }
        }
    }
}
#endif

// O[r] += P[r] V for the tile's kKT values (vt: [kKT][hd]), four rows x sixteen columns at a time
inline void tile_pv(const float *P, int rows, int hd, const float *vt, float *O) {
    for (int r0 = 0; r0 < rows; r0 += 4) {
        const int nr = std::min(4, rows - r0);
        for (int c0 = 0; c0 < hd; c0 += 16) {
            v4 a[4][4] = {};
            for (int j = 0; j < kKT; ++j) {
                const float *vr = vt + static_cast<size_t>(j) * hd + c0;
                const v4 v0 = ld4(vr), v1 = ld4(vr + 4), v2 = ld4(vr + 8), v3 = ld4(vr + 12);
                for (int r = 0; r < 4; ++r) {
                    const float pv = r < nr ? P[static_cast<size_t>(r0 + r) * kKT + j] : 0.0f;
                    a[r][0] += v0 * pv, a[r][1] += v1 * pv, a[r][2] += v2 * pv, a[r][3] += v3 * pv;
                }
            }
            for (int r = 0; r < nr; ++r) {
                float *orow = O + static_cast<size_t>(r0 + r) * hd + c0;
                for (int c = 0; c < 4; ++c) st4(orow + c * 4, ld4(orow + c * 4) + a[r][c]);
            }
        }
    }
}

int attn_x_flags() {   // GEODE_YUE2_NAR_FLAGS (an experiment's: read per layer)
    const char *v = std::getenv("GEODE_YUE2_NAR_FLAGS");
    return v ? static_cast<int>(std::strtol(v, nullptr, 0)) : 0;
}

// the layer's keys and values out of the banks into Emu::kt / vt (once a layer: every token range reads them)
int cpu_attn_prepare(Emu *e, const attn_desc_t &d, const _Float16 *const bank[4]) {
    const int nh = d.nh, nkv = d.nkv, hd = d.hd, group = nh / nkv, dgs = hd / 32;
    const int nk = d.nk ? d.nk : d.n;
    if (hd % 16 || kRows % group) return -1;
    const size_t total = static_cast<size_t>(nk + 31) / 32u, pre = d.npre ? static_cast<size_t>(d.npre) : total;
    const int tiles = (nk + kKT - 1) / kKT;
    const size_t per = static_cast<size_t>(tiles) * kKT * hd;   // a head's keys (or values), padded to whole tiles
    e->kt.resize(per * nkv);
    e->vt.resize(per * nkv);
    float *KT = e->kt.data(), *VT = e->vt.data();
    // the banks into the tiles, a task a head's tile (the padding's keys and values zero)
    e->pool->parallel_for(static_cast<int64_t>(nkv) * tiles, [&](int64_t a, int64_t b) {
        for (int64_t task = a; task < b; ++task) {
            const int kh = static_cast<int>(task / tiles), tl = static_cast<int>(task % tiles);
            float *kt = KT + static_cast<size_t>(kh) * per + static_cast<size_t>(tl) * kKT * hd;
            float *vt = VT + static_cast<size_t>(kh) * per + static_cast<size_t>(tl) * kKT * hd;
            for (int j = 0; j < kKT; ++j) {
                const int p = tl * kKT + j;
                if (p >= nk) {
                    for (int c = 0; c < hd; ++c) kt[static_cast<size_t>(c) * kKT + j] = 0.0f, vt[static_cast<size_t>(j) * hd + c] = 0.0f;
                    continue;
                }
                const size_t blk = static_cast<size_t>(p / 32);
                const int tail = blk >= pre;
                const size_t nb = tail ? total - pre : pre;
                const int pp = static_cast<int>((tail ? blk - pre : blk) * 32u) + p % 32;
                const _Float16 *kb = bank[2 * tail], *vb = bank[2 * tail + 1];
                for (int c = 0; c < hd; ++c) {
                    kt[static_cast<size_t>(c) * kKT + j] = static_cast<float>(kb[k_at(nb, dgs, kh, pp, c)]);
                    vt[static_cast<size_t>(j) * hd + c] = static_cast<float>(vb[v_at(nb, dgs, kh, pp, c)]);
                }
            }
        }
    });
#if defined(__ARM_FEATURE_DOTPROD)
    e->attn8 = e->attn8_opt || (attn_x_flags() & 0x2000);   // (the experiments' flag: a sweep switches it)
    if (e->attn8) {
        // each head's channel means over its real keys, then every key less them in int8
        e->kmean.assign(static_cast<size_t>(nkv) * hd, 0.0f);
        e->k8.resize(per * nkv);
        e->ks.resize(static_cast<size_t>(tiles) * kKT * nkv);
        float *M = e->kmean.data();
        e->pool->parallel_for(static_cast<int64_t>(nkv) * hd, [&](int64_t a, int64_t b) {
            for (int64_t x = a; x < b; ++x) {
                const int kh = static_cast<int>(x / hd), c = static_cast<int>(x % hd);
                double s = 0.0;
                for (int tl = 0; tl < tiles; ++tl) {
                    const float *row = KT + static_cast<size_t>(kh) * per + static_cast<size_t>(tl) * kKT * hd + static_cast<size_t>(c) * kKT;
                    for (int j = 0; j < kKT; ++j) s += row[j];   // (the padding's zeros add nothing)
                }
                M[x] = static_cast<float>(s / nk);
            }
        });
        int8_t *K8 = e->k8.data();
        float *KS = e->ks.data();
        e->pool->parallel_for(static_cast<int64_t>(nkv) * tiles, [&](int64_t a, int64_t b) {
            for (int64_t task = a; task < b; ++task) {
                const int kh = static_cast<int>(task / tiles), tl = static_cast<int>(task % tiles);
                const float *kt = KT + static_cast<size_t>(kh) * per + static_cast<size_t>(tl) * kKT * hd;
                int8_t *k8 = K8 + static_cast<size_t>(kh) * per + static_cast<size_t>(tl) * kKT * hd;
                float *ks = KS + static_cast<size_t>(kh) * tiles * kKT + static_cast<size_t>(tl) * kKT;
                const float *mu = M + static_cast<size_t>(kh) * hd;
                for (int j = 0; j < kKT; ++j) {
                    const bool real = tl * kKT + j < nk;
                    float amax = 0.0f;
                    if (real)
                        for (int c = 0; c < hd; ++c) amax = std::max(amax, std::fabs(kt[static_cast<size_t>(c) * kKT + j] - mu[c]));
                    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                    ks[j] = amax / 127.0f;
                    for (int c = 0; c < hd; ++c) {
                        const float v = real ? (kt[static_cast<size_t>(c) * kKT + j] - mu[c]) * inv : 0.0f;
                        k8[(static_cast<size_t>(c / 4) * kKT + j) * 4 + c % 4] =
                            static_cast<int8_t>(std::clamp(std::nearbyint(v), -127.0f, 127.0f));
                    }
                }
            }
        });
    }
#endif
    return 0;
}

// the queries of tokens [tb, te) over every key (cpu_attn_prepare's), their outputs into o
int cpu_attn_rows(Emu *e, const attn_desc_t &d, const float *q, float *o, int tb, int te) {
    const int nh = d.nh, nkv = d.nkv, hd = d.hd, group = nh / nkv;
    const int nk = d.nk ? d.nk : d.n;
    if (hd % 16 || kRows % group) return -1;
    const int tiles = (nk + kKT - 1) / kKT;
    const size_t per = static_cast<size_t>(tiles) * kKT * hd;
    if (e->kt.size() < per * nkv || e->vt.size() < per * nkv) return -1;
    const float *KT = e->kt.data(), *VT = e->vt.data();
    const int qt = kRows / group;   // tokens a task
    const int64_t blocks = (te - tb + qt - 1) / qt;
    e->pool->parallel_for(blocks * nkv, [&](int64_t a, int64_t b) {
        std::vector<float> qs(static_cast<size_t>(kRows) * hd), O(qs.size()), S(static_cast<size_t>(kRows) * kKT);
        float m[kRows], l[kRows];
#if defined(__ARM_FEATURE_DOTPROD)
        const bool i8 = e->attn8 && e->k8.size() >= per * nkv;
        std::vector<int8_t> q8(i8 ? static_cast<size_t>(kRows) * hd : 0);
        float sq[kRows];
#endif
        for (int64_t task = a; task < b; ++task) {
            const int kh = static_cast<int>(task % nkv), t0 = tb + static_cast<int>(task / nkv) * qt;
            const int rows = (std::min(te, t0 + qt) - t0) * group;
            for (int r = 0; r < rows; ++r) {
                const int t = t0 + r / group, h = kh * group + r % group;
                std::memcpy(&qs[static_cast<size_t>(r) * hd], q + static_cast<size_t>(t) * d.q_stride + static_cast<size_t>(h) * hd,
                            sizeof(float) * static_cast<size_t>(hd));
                m[r] = -std::numeric_limits<float>::infinity(), l[r] = 0.0f;
            }
            std::fill(O.begin(), O.begin() + static_cast<std::ptrdiff_t>(rows) * hd, 0.0f);
            const float *kth = KT + static_cast<size_t>(kh) * per, *vth = VT + static_cast<size_t>(kh) * per;
#if defined(__ARM_FEATURE_DOTPROD)
            if (i8)
                for (int r = 0; r < rows; ++r) {
                    const float *x = &qs[static_cast<size_t>(r) * hd];
                    float amax = 0.0f;
                    for (int c = 0; c < hd; ++c) amax = std::max(amax, std::fabs(x[c]));
                    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                    sq[r] = amax / 127.0f;
                    for (int c = 0; c < hd; ++c)
                        q8[static_cast<size_t>(r) * hd + c] = static_cast<int8_t>(std::clamp(std::nearbyint(x[c] * inv), -127.0f, 127.0f));
                }
#endif
            for (int tl = 0; tl < tiles; ++tl) {
                const int kv = std::min(kKT, nk - tl * kKT);   // the tile's real keys
#if defined(__ARM_FEATURE_DOTPROD)
                if (i8)
                    tile_scores_i8(q8.data(), sq, rows, hd, e->k8.data() + static_cast<size_t>(kh) * per + static_cast<size_t>(tl) * kKT * hd,
                                   e->ks.data() + static_cast<size_t>(kh) * tiles * kKT + static_cast<size_t>(tl) * kKT, S.data());
                else
#endif
                tile_scores(qs.data(), rows, hd, kth + static_cast<size_t>(tl) * kKT * hd, S.data());
                for (int r = 0; r < rows; ++r) {
                    float *sr = &S[static_cast<size_t>(r) * kKT];
                    float mx = m[r];
                    for (int j = 0; j < kv; ++j) mx = std::max(mx, sr[j]);
                    const float alpha = std::exp2(m[r] - mx);   // (the first tile: 2^-inf = 0, O and l still 0)
                    v4 sum = {};
                    const v4 mv = {mx, mx, mx, mx};
                    for (int j = 0; j < kKT; j += 4) {
                        v4 pj = exp2v(ld4(sr + j) - mv);
                        for (int z = 0; z < 4; ++z)
                            if (j + z >= kv) pj[z] = 0.0f;   // (the padding: no key)
                        st4(sr + j, pj);
                        sum += pj;
                    }
                    l[r] = l[r] * alpha + ((sum[0] + sum[1]) + (sum[2] + sum[3]));
                    m[r] = mx;
                    if (alpha != 1.0f) {
                        float *orow = &O[static_cast<size_t>(r) * hd];
                        for (int c = 0; c < hd; ++c) orow[c] *= alpha;
                    }
                }
                tile_pv(S.data(), rows, hd, vth + static_cast<size_t>(tl) * kKT * hd, O.data());
            }
            for (int r = 0; r < rows; ++r) {
                const int t = t0 + r / group, h = kh * group + r % group;
                float *orow = o + static_cast<size_t>(t) * d.o_stride + static_cast<size_t>(h) * hd;
                const float inv = 1.0f / l[r];
                for (int c = 0; c < hd; ++c) orow[c] = O[static_cast<size_t>(r) * hd + c] * inv;
            }
        }
    });
    return 0;
}

int cpu_attn(Emu *e, const attn_desc_t &d, const _Float16 *const bank[4], const float *q, float *o) {
    if (const int r = cpu_attn_prepare(e, d, bank)) return r;
    return cpu_attn_rows(e, d, q, o, 0, d.n);
}

// the banks a descriptor names, as the emulation's buffers (false: one is missing)
bool attn_banks(Emu *e, const attn_desc_t &d, const _Float16 *bank[4]) {
    for (int i = 0; i < 4; ++i) bank[i] = nullptr;
    for (int i = 0; i < d.n_chunks; ++i)
        if (!(bank[i] = reinterpret_cast<const _Float16 *>(emu_buf(e, d.kv_fd[i], nullptr)))) return false;
    return true;
}

// ATTN_ENCODER with nk / npre (attn_cfg.h): every query sees the nk keys; the queries, the scores (HMX's outputs) and
// the weights in fp16, the rest in fp32 (not the DSP's P V rounding: its superblocks' fp16 sums)
int emu_attn(gnpu *g, const void *desc, int desc_len, const float *q, size_t, float *o, size_t, double *dsp_s) {
    Emu *e = emu_of(g);
    if (desc_len != static_cast<int>(sizeof(attn_desc_t))) return -1;
    const attn_desc_t &d = *static_cast<const attn_desc_t *>(desc);
    if (d.version != ATTN_VERSION || !(d.flags & ATTN_ENCODER) || d.n_chunks != (d.npre ? 4 : 2)) return -1;
    const int n = d.n, nh = d.nh, nkv = d.nkv, hd = d.hd, group = nh / nkv, dgs = hd / 32;
    const int nk = d.nk ? d.nk : d.n, log2 = (d.flags & ATTN_LOG2) != 0;
    const bool fast = e->fast;
    const auto rr = [fast](float x) { return fast ? x : r16(x); };   // (the emulation: fp16 where HMX has it)
    const size_t total = static_cast<size_t>(nk + 31) / 32u, pre = d.npre ? static_cast<size_t>(d.npre) : total;
    const _Float16 *bank[4] = {};
    for (int i = 0; i < d.n_chunks; ++i) bank[i] = reinterpret_cast<const _Float16 *>(emu_buf(e, d.kv_fd[i], nullptr));
    for (int i = 0; i < d.n_chunks; ++i)
        if (!bank[i]) return -1;
    if (fast && log2 && std::getenv("GEODE_YUE2_NAR_ATTN_OLD") == nullptr) {   // the CPU backend's (below: the old order)
        if (dsp_s) *dsp_s = 0.0;
        return cpu_attn(e, d, bank, q, o);
    }
    std::vector<float> K(static_cast<size_t>(nkv) * nk * hd), V(K.size());
    e->pool->parallel_for(static_cast<int64_t>(nkv) * nk, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) {
            const int kh = static_cast<int>(i / nk), p = static_cast<int>(i % nk);
            const size_t blk = static_cast<size_t>(p / 32);
            const int tail = blk >= pre;
            const size_t nb = tail ? total - pre : pre;
            const int pp = static_cast<int>((tail ? blk - pre : blk) * 32u) + p % 32;
            const _Float16 *kb = bank[2 * tail], *vb = bank[2 * tail + 1];
            for (int c = 0; c < hd; ++c) {
                K[static_cast<size_t>(i) * hd + c] = static_cast<float>(kb[k_at(nb, dgs, kh, pp, c)]);
                V[static_cast<size_t>(i) * hd + c] = static_cast<float>(vb[v_at(nb, dgs, kh, pp, c)]);
            }
        }
    });
    constexpr int QB = 4;
    const int64_t blocks = (n + QB - 1) / QB;
    e->pool->parallel_for(blocks * nkv, [&](int64_t a, int64_t b) {
        const int R = QB * group;
        std::vector<float> qs(static_cast<size_t>(R) * hd), s(static_cast<size_t>(R) * nk), acc(static_cast<size_t>(R) * hd);
        for (int64_t task = a; task < b; ++task) {
            const int kh = static_cast<int>(task % nkv), t0 = static_cast<int>(task / nkv) * QB;
            const int nt = std::min(n, t0 + QB) - t0, nr = nt * group;
            for (int r = 0; r < nr; ++r) {
                const int t = t0 + r / group, h = kh * group + r % group;
                for (int c = 0; c < hd; ++c) qs[static_cast<size_t>(r) * hd + c] = rr(q[static_cast<size_t>(t) * d.q_stride + static_cast<size_t>(h) * hd + c]);
            }
            const float *Kh = &K[static_cast<size_t>(kh) * nk * hd], *Vh = &V[static_cast<size_t>(kh) * nk * hd];
            for (int j = 0; j < nk; ++j)
                for (int r = 0; r < nr; ++r)
                    s[static_cast<size_t>(r) * nk + j] = rr(dot(&qs[static_cast<size_t>(r) * hd], Kh + static_cast<size_t>(j) * hd, hd));
            std::fill(acc.begin(), acc.end(), 0.0f);
            float sum[QB * 32];
            for (int r = 0; r < nr; ++r) {
                float *sr = &s[static_cast<size_t>(r) * nk], mx = -std::numeric_limits<float>::infinity(), z = 0.0f;
                for (int j = 0; j < nk; ++j) mx = std::max(mx, sr[j]);
                for (int j = 0; j < nk; ++j)   // (P in fp16, as HMX takes it; ATTN_LOG2: the scores in log2 units)
                    z += (sr[j] = rr(log2 ? std::exp2(sr[j] - mx) : std::exp(sr[j] - mx)));
                sum[r] = z;
            }
            for (int j = 0; j < nk; ++j) {
                const float *vr = Vh + static_cast<size_t>(j) * hd;
                for (int r = 0; r < nr; ++r) {
                    const float p = s[static_cast<size_t>(r) * nk + j];
                    float *ar = &acc[static_cast<size_t>(r) * hd];
                    for (int c = 0; c < hd; c += 8) store8(ar + c, load8(ar + c) + load8(vr + c) * p);
                }
            }
            for (int r = 0; r < nr; ++r) {
                const int t = t0 + r / group, h = kh * group + r % group;
                float *orow = o + static_cast<size_t>(t) * d.o_stride + static_cast<size_t>(h) * hd;
                for (int c = 0; c < hd; ++c) orow[c] = acc[static_cast<size_t>(r) * hd + c] / sum[r];
            }
        }
    });
    if (dsp_s) *dsp_s = 0.0;
    return 0;
}

void par_on_pool(void *ctx, int items, void (*fn)(void *, int, int), void *arg) {
    static_cast<ThreadPool *>(ctx)->parallel_for(items, [&](int64_t a, int64_t b) { fn(arg, static_cast<int>(a), static_cast<int>(b)); });
}

// MemAvailable in bytes (0: unknown)
int64_t mem_available() {
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    long long kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    std::fclose(f);
    return static_cast<int64_t>(kb) * 1024;
}

int env_int(const char *name, int def) {
    const char *v = std::getenv(name);
    return v && *v ? static_cast<int>(std::strtol(v, nullptr, 0)) : def;
}

// the prefixes' runs: of the experiments' flags (GEODE_YUE2_NAR_FLAGS) only those that change the arithmetic
int prefix_flags() { return env_int("GEODE_YUE2_NAR_FLAGS", 0) & NAR_X_OLDNORM; }

}  // namespace

struct NpuCalls {
    void *lib = nullptr;
    gnpu *g = nullptr;
    void (*close)(gnpu *) = nullptr;
    void *(*shared_alloc)(gnpu *, size_t, int, int *) = nullptr;
    int (*shared_free)(gnpu *, void *) = nullptr;   // (an older libgeode_npu.so, the emulation: none)
    int (*shared_map)(gnpu *, void *, int) = nullptr;   // (an older libgeode_npu.so, the emulation: none)
    int (*gemm_f16)(gnpu *, int, uint64_t, int64_t, int64_t, int, int, const float *const *, float *const *) = nullptr;
    int (*mlp_f16)(gnpu *, int, uint64_t, uint64_t, uint64_t, int64_t, int64_t, int, int, int, int, int,
                   const float *const *, float *const *) = nullptr;   // (an older libgeode_npu.so: none)
    int (*attn)(gnpu *, const void *, int, const float *, size_t, float *, size_t, double *) = nullptr;
    float *(*attn_buf)(gnpu *, int, size_t) = nullptr;
    double (*dsp_s)(const gnpu *) = nullptr, (*conv_s)(const gnpu *) = nullptr;   // the products' DSP / conversions
    // the layers on the DSP (npu/nar_cfg.h; an older libgeode_npu.so, the emulation: none)
    int (*nar_open)(gnpu *, const void *, int, int, int) = nullptr;
    int (*nar_run)(gnpu *, int, double *) = nullptr;
    void (*nar_close)(gnpu *) = nullptr;
    std::unique_ptr<Emu> emu;
    std::unique_ptr<NarGpu> gpu;   // (given back before the emulation)
    ~NpuCalls() {
        if (g && close) close(g);
#if defined(__ANDROID__)
        if (lib) dlclose(lib);
#endif
    }
};

// ---- Hexagon v66 .. v72 (no HMX; npu/hvx66): the "Fast" sound's 8-bit products on the DSP's HVX beside the GPU
// pipeline - a helper of its own takes units from the front while the attention runs (after it the CPU is ~5x faster
// a unit: the DSP takes none then). The arithmetic: the c8 twins' exact int8 sums per group of 32 (the CPU's codes,
// quantized here as quant_x8 does), the groups' scales in Q15 fixed point (geode_hvx.idl) - 1e-4 of the 8-bit
// product, whose own rounding is ~5e-3.
constexpr int kHvxRows = 128;   // a slab's tokens (the DSP's buffers' size)

struct NarDsp::Hvx {
    void *lib = nullptr;
    gnpu_hvx *h = nullptr;
    void (*close)(gnpu_hvx *) = nullptr;
    void *(*ralloc)(size_t, int) = nullptr;
    void (*rfree)(void *) = nullptr;
    int (*gemm)(gnpu_hvx *, const uint8_t *, size_t, int, int, int, const int8_t *, const int32_t *, const int32_t *,
                const int32_t *, int32_t *, int, long long *) = nullptr;
    int threads = 2;   // HVX threads (a v66: two contexts)
    int slot_cached = 1;   // the slot's memory (GEODE_YUE2_NAR_HVX_SLOT_CACHED)
    // the layer's slot (rpcmem; cached: the CPU's copy into uncached memory ran ~160 MB/s - FastRPC flushes it around
    // each call instead): o, gate/up and down's c8 blocks as the pack has them, then each one's ratios (Q15 << 16) [rows / 32][k / 32][32]; wmax: the rows' largest scales
    // times 2^-shift (the CPU multiplies by them)
    uint8_t *slot = nullptr;
    size_t slot_len = 0;
    const void *slot_of = nullptr;   // the layer it holds
    struct Mat {
        size_t w = 0, wbytes = 0, wq = 0;
        int rows = 0, k = 0;
        std::vector<float> wmax;
    } m[3];
    // a slab's in / out (cached rpcmem: FastRPC flushes them around a call)
    int8_t *x = nullptr;
    int32_t *corr = nullptr, *aq = nullptr, *y = nullptr;
    std::vector<float> s, smax;               // the activations' groups' scales, each token's largest
    std::vector<float> xr, hb, ob, gub, act;  // the slab's rows of x (written back when all came back), the glue's
    ~Hvx() {
        if (rfree) {
            rfree(slot), rfree(x), rfree(corr), rfree(aq), rfree(y);
        }
        if (h && close) close(h);
#if defined(__ANDROID__)
        if (lib) dlclose(lib);
#endif
    }
    // n tokens' rows (k wide, stride xs) as quant_x8 quantizes them (rot: fwht32-turned first) into x, with corr
    // (128 sum(q) a group), aq (each group's scale over the token's largest, Q15 << 16) and smax; rows n .. n8 zeros
    void quant(const float *in, int64_t xs, int n, int n8, int k, bool rot) {
        const int G = k / 32;
        s.resize(static_cast<size_t>(G));
        smax.assign(static_cast<size_t>(n8), 0.0f);
        float tmp[32];
        for (int t = 0; t < n; ++t) {
            const float *xr0 = in + static_cast<int64_t>(t) * xs;
            int8_t *q = x + static_cast<size_t>(t) * static_cast<size_t>(k);
            int32_t *cr = corr + static_cast<size_t>(t) * static_cast<size_t>(G);
            float mx = 0.0f;
#if defined(__aarch64__)
            for (int g = 0; g < G; ++g) {   // (NEON: fwht32's stages in its order - the same sums as quant_x8's)
                float32x4_t v[8];
                for (int i = 0; i < 8; ++i) v[i] = vld1q_f32(xr0 + g * 32 + 4 * i);
                if (rot) {
                    for (int i = 0; i < 8; ++i) {   // h = 1, 2: within a vector
                        const float32x4_t e = vuzp1q_f32(v[i], v[i]), o = vuzp2q_f32(v[i], v[i]);
                        const float32x4_t a = vzip1q_f32(vaddq_f32(e, o), vsubq_f32(e, o));   // [s0 d0 s1 d1]
                        const float32x2_t lo = vget_low_f32(a), hi = vget_high_f32(a);
                        v[i] = vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
                    }
                    for (int h = 1; h < 8; h <<= 1)   // h = 4, 8, 16: across vectors
                        for (int i = 0; i < 8; i += 2 * h)
                            for (int j = i; j < i + h; ++j) {
                                const float32x4_t a = v[j], b = v[j + h];
                                v[j] = vaddq_f32(a, b), v[j + h] = vsubq_f32(a, b);
                            }
                    for (int i = 0; i < 8; ++i) v[i] = vmulq_n_f32(v[i], 0.17677669529663687f);
                }
                float32x4_t m4 = vabsq_f32(v[0]);
                for (int i = 1; i < 8; ++i) m4 = vmaxq_f32(m4, vabsq_f32(v[i]));
                const float amax = vmaxvq_f32(m4), inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                int32x4_t sum = vdupq_n_s32(0);
                const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
                for (int i = 0; i < 8; i += 2) {
                    const int32x4_t a = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[i], inv)), lo), hi);
                    const int32x4_t b = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[i + 1], inv)), lo), hi);
                    sum = vaddq_s32(sum, vaddq_s32(a, b));
                    const int16x8_t ab = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
                    vst1_s8(q + g * 32 + 4 * i, vmovn_s16(ab));
                }
                cr[g] = 128 * vaddvq_s32(sum);
                s[static_cast<size_t>(g)] = amax / 127.0f;
                mx = std::max(mx, s[static_cast<size_t>(g)]);
            }
            (void)tmp;
#else
            for (int g = 0; g < G; ++g) {
                const float *xg = xr0 + g * 32;
                if (rot) {
                    std::memcpy(tmp, xg, sizeof tmp);
                    fwht32(tmp);
                    xg = tmp;
                }
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xg[i]));
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                int sum = 0;
                for (int i = 0; i < 32; ++i) {
                    const int v = static_cast<int>(std::clamp(std::nearbyint(xg[i] * inv), -127.0f, 127.0f));
                    q[g * 32 + i] = static_cast<int8_t>(v);
                    sum += v;
                }
                cr[g] = 128 * sum;
                s[static_cast<size_t>(g)] = amax / 127.0f;
                mx = std::max(mx, s[static_cast<size_t>(g)]);
            }
#endif
            smax[static_cast<size_t>(t)] = mx;
            int32_t *a = aq + static_cast<size_t>(t) * static_cast<size_t>(G);
            for (int g = 0; g < G; ++g) {
                const long r = mx > 0.0f ? std::lrint(std::min(32767.0f, s[static_cast<size_t>(g)] / mx * 32768.0f)) : 0;
                a[g] = static_cast<int32_t>(static_cast<uint32_t>(r) << 16);
            }
        }
        for (int t = n; t < n8; ++t) {
            std::memset(x + static_cast<size_t>(t) * static_cast<size_t>(k), 0, static_cast<size_t>(k));
            std::memset(corr + static_cast<size_t>(t) * static_cast<size_t>(G), 0, static_cast<size_t>(G) * 4u);
            std::memset(aq + static_cast<size_t>(t) * static_cast<size_t>(G), 0, static_cast<size_t>(G) * 4u);
        }
    }
};

bool NarDsp::set_hvx(const std::string &dir, std::string &why) {
    if (hvx()) return true;
    if (const char *v = std::getenv("GEODE_YUE2_NAR_HVX"); v && !std::strcmp(v, "0")) {
        why = "off (GEODE_YUE2_NAR_HVX=0)";
        return false;
    }
#if defined(__ANDROID__)
    if (!npu_ || !npu_->emu) {
        why = "only beside the CPU / GPU backends";
        return false;
    }
    auto v = std::make_unique<Hvx>();
    const std::string so = dir + "/libgeode_npu.so";
    v->lib = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!v->lib) {
        why = so + ": " + dlerror();
        return false;
    }
    const auto sym = [&](const char *name) { return dlsym(v->lib, name); };
    auto *open_fn = reinterpret_cast<gnpu_hvx *(*)(const char *, char *, int)>(sym("gnpu_hvx_open"));
    v->close = reinterpret_cast<decltype(v->close)>(sym("gnpu_hvx_close"));
    v->ralloc = reinterpret_cast<decltype(v->ralloc)>(sym("gnpu_hvx_alloc"));
    v->rfree = reinterpret_cast<decltype(v->rfree)>(sym("gnpu_hvx_free"));
    v->gemm = reinterpret_cast<decltype(v->gemm)>(sym("gnpu_hvx_q8_gemm"));
    if (!open_fn || !v->close || !v->ralloc || !v->rfree || !v->gemm) {
        why = so + " has no gnpu_hvx_* (an older one)";
        return false;
    }
    char e[256] = {0};
    v->h = open_fn(dir.c_str(), e, static_cast<int>(sizeof e));
    if (!v->h) {
        why = e;
        return false;
    }
    const int H = c_.hidden, qd = c_.heads * c_.head_dim, F = c_.ffn;
    const size_t K = static_cast<size_t>(std::max({H, qd, F})), R = static_cast<size_t>(std::max(H, 2 * F));
    v->x = static_cast<int8_t *>(v->ralloc(kHvxRows * K, 1));
    v->corr = static_cast<int32_t *>(v->ralloc(kHvxRows * (K / 32) * 4u, 1));
    v->aq = static_cast<int32_t *>(v->ralloc(kHvxRows * (K / 32) * 4u, 1));
    v->y = static_cast<int32_t *>(v->ralloc(kHvxRows * R * 4u, 1));
    if (!v->x || !v->corr || !v->aq || !v->y) {
        why = "no rpcmem for its buffers";
        return false;
    }
    v->threads = std::clamp(env_int("GEODE_YUE2_NAR_HVX_THREADS", 2), 1, 4);
    v->slot_cached = env_int("GEODE_YUE2_NAR_HVX_SLOT_CACHED", 1) != 0;
    hvx_nice_ = env_int("GEODE_YUE2_NAR_HVX_NICE", 0);
    hvx_ahead_ = env_int("GEODE_YUE2_NAR_HVX_AHEAD", 1) != 0;
    hvx_ = std::move(v);
    hvx_off_ = false;
    hvx_note_.clear();
    return true;
#else
    (void)dir;
    why = "no DSP here";
    return false;
#endif
}

bool NarDsp::hvx() const { return hvx_ && hvx_->h && !hvx_off_ && !hvx_paused_; }

// L's o, gate/up and down 8-bit twins (c8.*: the pack's mapping) into the slot, with their rows' scale ratios - on
// the DSP helper while the attention's first slice runs (~45 MB a layer)
bool NarDsp::hvx_slot(const LayerW &L, std::string &err) {
    Hvx &v = *hvx_;
    if (v.slot_of == &L) return true;
    Emu *e = npu_->emu.get();
    const int H = c_.hidden, qd = c_.heads * c_.head_dim, F = c_.ffn;
    const uint64_t off[3] = {L.o, L.gu, L.down};
    const int rows[3] = {H, 2 * F, H}, ks[3] = {qd, H, F}, sh[3] = {L.sh_o, L.sh_gu, L.sh_down};
    const uint8_t *tw[3];
    size_t need = 0;
    for (int i = 0; i < 3; ++i) {
        const auto it = e->q8.find({L.w.fd, off[i]});
        if (it == e->q8.end()) {
            err = "NAR: the v66 DSP: a matrix without its 8-bit twin";
            return false;
        }
        tw[i] = it->second;
        Hvx::Mat &M = v.m[i];
        M.rows = rows[i], M.k = ks[i];
        M.wbytes = static_cast<size_t>(rows[i] / 32) * static_cast<size_t>(ks[i]) * 34u;
        M.w = need;
        need += (M.wbytes + 127) / 128 * 128;
    }
    for (int i = 0; i < 3; ++i) {
        v.m[i].wq = need;
        need += (static_cast<size_t>(rows[i]) * static_cast<size_t>(ks[i] / 32) * 4u + 127) / 128 * 128;
    }
    v.slot_of = nullptr;
    if (v.slot_len < need) {
        if (v.slot) v.rfree(v.slot), v.slot = nullptr, v.slot_len = 0;
        v.slot = static_cast<uint8_t *>(v.ralloc(need, v.slot_cached));
        if (!v.slot) {
            err = "NAR: the v66 DSP: no " + std::to_string(need >> 20) + " MB of rpcmem for a layer";
            return false;
        }
        v.slot_len = need;
    }
    std::vector<int32_t> tmp;
    for (int i = 0; i < 3; ++i) {
        Hvx::Mat &M = v.m[i];
        std::memcpy(v.slot + M.w, tw[i], M.wbytes);
        const int G = M.k / 32, nb = M.rows / 32;
        const float down = std::ldexp(1.0f, -sh[i]);
        M.wmax.assign(static_cast<size_t>(M.rows), 0.0f);
        tmp.resize(static_cast<size_t>(G) * 32u);
        int32_t *wq = reinterpret_cast<int32_t *>(v.slot + M.wq);
        for (int b = 0; b < nb; ++b) {
            const _Float16 *sc = reinterpret_cast<const _Float16 *>(tw[i] + static_cast<size_t>(b) * static_cast<size_t>(M.k) * 34u +
                                                                     static_cast<size_t>(M.k) * 32u);
            float mx[32] = {0.0f};
            for (int g = 0; g < G; ++g)
                for (int o = 0; o < 32; ++o) mx[o] = std::max(mx[o], std::fabs(static_cast<float>(sc[g * 32 + o])));
            for (int g = 0; g < G; ++g)
                for (int o = 0; o < 32; ++o) {
                    const float r = mx[o] > 0.0f ? static_cast<float>(sc[g * 32 + o]) / mx[o] : 0.0f;
                    const long q = std::lrint(std::clamp(r * 32768.0f, -32767.0f, 32767.0f));
                    tmp[static_cast<size_t>(g) * 32u + static_cast<size_t>(o)] = static_cast<int32_t>(static_cast<uint32_t>(q) << 16);
                }
            std::memcpy(wq + static_cast<size_t>(b) * static_cast<size_t>(G) * 32u, tmp.data(), tmp.size() * 4u);   // (whole, in order)
            for (int o = 0; o < 32; ++o) M.wmax[static_cast<size_t>(b) * 32u + static_cast<size_t>(o)] = mx[o] * down;
        }
    }
    v.slot_of = &L;
    return true;
}

// The DSP's unit, kHvxRows tokens a slab: o, the residual, rmsnorm, gate/up, SiLU, down, the residual - the
// quantizing and the glue on this thread (the pool is the attention's); a slab's rows of x_ change only when all its
// products came back (upto: the rows done - the CPU takes the rest of a unit the DSP fails)
bool NarDsp::hvx_unit(const LayerW &L, int r0, int r1, Prof &P, std::string &err, int &upto) {
    Hvx &v = *hvx_;
    const int H = c_.hidden, qd = c_.heads * c_.head_dim, F = c_.ffn;
    const bool rot = npu_->emu->q8rot;
    upto = r0;
    for (int s0 = r0; s0 < r1; s0 += kHvxRows) {
        const int n = std::min(kHvxRows, r1 - s0), n8 = (n + 7) / 8 * 8;
        float *x = &x_[static_cast<size_t>(s0) * static_cast<size_t>(H)];
        const size_t nh = static_cast<size_t>(n) * static_cast<size_t>(H);
        v.xr.assign(x, x + nh);
        v.hb.resize(nh), v.ob.resize(nh);
        v.gub.resize(static_cast<size_t>(n) * 2u * static_cast<size_t>(F)), v.act.resize(static_cast<size_t>(n) * static_cast<size_t>(F));
        // matrix i (0 o, 1 gate/up, 2 down) times the slab's rows `in` (stride xs) -> out [n][rows]
        const auto prod = [&](int i, const float *in, int64_t xs, float *out) {
            const Hvx::Mat &M = v.m[i];
            const double ta = now_seconds();
            v.quant(in, xs, n, n8, M.k, rot);
            long long us = 0;
            const double tb = now_seconds();
            if (const int e = v.gemm(v.h, v.slot + M.w, M.wbytes, M.rows, M.k, n8, v.x, v.corr, v.aq,
                                     reinterpret_cast<const int32_t *>(v.slot + M.wq), v.y, v.threads, &us)) {
                err = "NAR: the v66 DSP's product failed (" + std::to_string(e) + ")";
                return false;
            }
            const double tc = now_seconds();
            for (int t = 0; t < n; ++t) {
                const float a = v.smax[static_cast<size_t>(t)] * 0.25f;
                const int32_t *yr = v.y + static_cast<size_t>(t) * static_cast<size_t>(M.rows);
                float *orow = out + static_cast<size_t>(t) * static_cast<size_t>(M.rows);
                const float *wm = M.wmax.data();
                for (int r = 0; r < M.rows; ++r) orow[r] = static_cast<float>(yr[r]) * wm[r] * a;
            }
            P.hvx_dsp += static_cast<double>(us) * 1e-6, P.hvx_call += tc - tb, P.hvx_cpu += (tb - ta) + (now_seconds() - tc);
            return true;
        };
        if (!prod(0, ob_ + static_cast<size_t>(s0) * static_cast<size_t>(qd), qd, v.ob.data())) return false;
        double t1 = now_seconds();
        for (int r = 0; r < n; ++r) {
            float *xr = v.xr.data() + static_cast<size_t>(r) * static_cast<size_t>(H);
            const float *orow = v.ob.data() + static_cast<size_t>(r) * static_cast<size_t>(H);
            for (int i = 0; i < H; ++i) xr[i] += orow[i];
            rms_row(xr, L.mlp_norm, v.hb.data() + static_cast<size_t>(r) * static_cast<size_t>(H), H, c_.eps);
        }
        P.hvx_cpu += now_seconds() - t1;
        if (!prod(1, v.hb.data(), H, v.gub.data())) return false;
        t1 = now_seconds();
        for (int r = 0; r < n; ++r) {
            const float *g = v.gub.data() + static_cast<size_t>(r) * 2u * static_cast<size_t>(F);
            float *ar = v.act.data() + static_cast<size_t>(r) * static_cast<size_t>(F);
            for (int i = 0; i < F; ++i) ar[i] = silu(g[i]) * g[F + i];
        }
        P.hvx_cpu += now_seconds() - t1;
        if (!prod(2, v.act.data(), F, v.ob.data())) return false;
        t1 = now_seconds();
        for (size_t i = 0; i < nh; ++i) v.xr[i] += v.ob[i];
        std::memcpy(x, v.xr.data(), nh * sizeof(float));
        P.hvx_cpu += now_seconds() - t1;
        upto = s0 + n;
    }
    return true;
}

NarDsp::NarDsp() = default;
NarDsp::~NarDsp() {
    join_loads();   // (their buffers are the session's: npu_ closes after)
}

std::string NarDsp::device() const { return npu_ && npu_->gpu ? npu_->gpu->name() : std::string(); }

bool NarDsp::alloc(Buf &b, size_t bytes, std::string &err) {
    if (b.p && b.len >= bytes) return true;
    free_buf(b);   // (a smaller one: every caller grows buffers with the driver closed)
    int fd = -1;
    // (lazy_: GNPU_SHARED_UNMAPPED - mapped when a group that reads it runs)
    void *p = npu_->shared_alloc(npu_->g, bytes, lazy_ ? GNPU_SHARED_CACHED | GNPU_SHARED_UNMAPPED : GNPU_SHARED_CACHED, &fd);
    if (!p) {
        err = "NAR: no " + std::to_string(bytes >> 20) + " MB of memory the DSP maps";
        return false;
    }
    b.p = static_cast<uint8_t *>(p);
    b.len = bytes;
    b.fd = fd;
    b.mapped = !lazy_;
    if (b.mapped) mapped_ += bytes;
    return true;
}

void NarDsp::free_buf(Buf &b) {
    if (b.p && b.len && npu_ && npu_->shared_free) {   // (borrowed ones, len 0, and the emulation's: not ours)
        if (b.mapped) mapped_ -= std::min(mapped_, b.len);
        npu_->shared_free(npu_->g, b.p);
        b = Buf{};
    }
}

bool NarDsp::map_buf(Buf &b, std::string &err) {
    if (!b.p || !b.len || b.mapped || !npu_->shared_map) return true;
    if (const int e = npu_->shared_map(npu_->g, b.p, 1)) {
        char h[120];
        std::snprintf(h, sizeof h, "NAR: %zu MB would not map into the DSP (0x%x; %zu MB mapped)", b.len >> 20,
                      static_cast<unsigned>(e), mapped_ >> 20);
        err = h;
        return false;
    }
    b.mapped = true;
    mapped_ += b.len;
    return true;
}

void NarDsp::unmap_buf(Buf &b) {
    if (!b.p || !b.len || !b.mapped || !npu_->shared_map) return;
    npu_->shared_map(npu_->g, b.p, 0);
    b.mapped = false;
    mapped_ -= std::min(mapped_, b.len);
}

size_t NarDsp::unit_unmapped(const std::vector<LayerW> &Ls, int l) const {
    const auto out = [](const Buf &b) { return b.p && b.len && !b.mapped ? b.len : size_t{0}; };
    size_t n = out(Ls[static_cast<size_t>(l)].w);
    if (npre_) n += out(pre_k_[static_cast<size_t>(l)]) + out(pre_v_[static_cast<size_t>(l)]);
    return n;
}

bool NarDsp::map_unit(const std::vector<LayerW> &Ls, int l, std::string &err) {
    return map_buf(const_cast<Buf &>(Ls[static_cast<size_t>(l)].w), err) &&
           (!npre_ || (map_buf(pre_k_[static_cast<size_t>(l)], err) && map_buf(pre_v_[static_cast<size_t>(l)], err)));
}

void NarDsp::unmap_unit(const std::vector<LayerW> &Ls, int l) {
    unmap_buf(const_cast<Buf &>(Ls[static_cast<size_t>(l)].w));
    if (npre_) unmap_buf(pre_k_[static_cast<size_t>(l)]), unmap_buf(pre_v_[static_cast<size_t>(l)]);
}

void cpu_gemm_hmx16_ref(const uint8_t *w, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                        float *const *ys, ThreadPool &pool) {
    cpu_gemm_hmx16(w, rows, cols, shift, n, xs, ys, pool);
}

bool NarDsp::open(Pack &pack, const std::string &npu_dir, ThreadPool &pool, std::string &err) {
    pool_ = &pool;
    pack_ = &pack;
    if (!read_config(pack, c_, err)) return false;
    npu_ = std::make_unique<NpuCalls>();
    if (npu_dir.empty() || npu_dir == "cpu" || npu_dir == "gpu") {
        npu_->emu = std::make_unique<Emu>();
        npu_->emu->pool = &pool;
        npu_->emu->fast = !npu_dir.empty();
        npu_->emu->attn8_opt = want_attn8_ || env_int("GEODE_YUE2_NAR_ATTN8", 0) != 0;
        npu_->g = reinterpret_cast<gnpu *>(npu_->emu.get());
        npu_->shared_alloc = emu_alloc;
        npu_->gemm_f16 = emu_gemm_f16;
        npu_->mlp_f16 = emu_mlp_f16;
        npu_->attn = emu_attn;
        npu_->attn_buf = emu_attn_buf;
        backend_ = npu_->emu->fast ? "cpu" : "emulation";
        cpu_ = npu_->emu->fast;
        if (npu_dir == "gpu") {   // the products on the GPU, the attention and the rest the CPU backend's
            const int64_t H = c_.hidden, qd = static_cast<int64_t>(c_.heads) * c_.head_dim;
            const int64_t kd = static_cast<int64_t>(c_.kv_heads) * c_.head_dim, F = c_.ffn;
            const size_t layer_bytes = static_cast<size_t>(2 * ((qd + 2 * kd) * H + H * qd + 2 * F * H + H * F)) + (size_t{1} << 20);
            npu_->gpu = std::make_unique<NarGpu>();
            // every layer kept on the GPU when the memory has room for it and 2 GB more (no copy a velocity: the page
            // cache's copy is not read again), else two slots the layers take turns in
            const int64_t all = static_cast<int64_t>(layer_bytes) * c_.layers;
            const int slots = env_int("GEODE_YUE2_NAR_GPU_SLOTS", mem_available() >= all + (int64_t{2} << 30) ? c_.layers : 2);
            if (!npu_->gpu->open(std::max(1, slots), layer_bytes, "", err)) return false;
            npu_->emu->gpu = npu_->gpu.get();
            if (const char *sh = std::getenv("GEODE_YUE2_NAR_GPU_SHARE")) {   // fixed (1: the GPU alone)
                npu_->emu->gpu_share = std::clamp(std::atof(sh), 0.0, 1.0);
                npu_->emu->gpu_adapt = false;
            }
            npu_->gemm_f16 = gpu_gemm_f16;
            npu_->mlp_f16 = nullptr;   // (gate / up as one product, SiLU on the CPU, down)
            backend_ = "gpu";
            pipe_ = env_int("GEODE_YUE2_NAR_PIPE", 1) != 0;
            pipe_slices_ = std::max(2, env_int("GEODE_YUE2_NAR_PIPE_SLICES", 12));
            pipe_split_ = std::max(1, env_int("GEODE_YUE2_NAR_PIPE_SPLIT", 2));
            prefetch_ = env_int("GEODE_YUE2_NAR_GPU_PREFETCH", 1) != 0;
            gpu_bits_ = env_int("GEODE_YUE2_NAR_GPU_BITS", 15);
            pipe_width_ = std::max(0, env_int("GEODE_YUE2_NAR_PIPE_WIDTH", 0));
            // the v66 DSP beside it (the Fast sound's products): its libgeode_npu.so's folder (the tools; the app
            // calls set_hvx itself)
            if (const char *d = std::getenv("GEODE_YUE2_NAR_HVX"); d && *d && std::strcmp(d, "0")) {
                std::string why;
                if (!set_hvx(d, why)) std::fprintf(stderr, "NAR: the v66 DSP is out of reach: %s\n", why.c_str());
            }
        }
    } else {
#if defined(__ANDROID__)
        const std::string so = npu_dir + "/libgeode_npu.so";
        npu_->lib = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!npu_->lib) {
            err = "NAR: " + so + ": " + dlerror();
            return false;
        }
        const auto sym = [&](const char *name) { return dlsym(npu_->lib, name); };
        auto *open_fn = reinterpret_cast<gnpu *(*)(const char *, char *, int)>(sym("gnpu_open"));
        auto *set_par = reinterpret_cast<void (*)(gnpu *, gnpu_par_fn, void *)>(sym("gnpu_set_par"));
        npu_->close = reinterpret_cast<void (*)(gnpu *)>(sym("gnpu_close"));
        npu_->shared_alloc = reinterpret_cast<void *(*)(gnpu *, size_t, int, int *)>(sym("gnpu_shared_alloc"));
        npu_->gemm_f16 = reinterpret_cast<decltype(npu_->gemm_f16)>(sym("gnpu_gemm_f16"));
        npu_->mlp_f16 = reinterpret_cast<decltype(npu_->mlp_f16)>(sym("gnpu_mlp_f16"));
        npu_->nar_open = reinterpret_cast<decltype(npu_->nar_open)>(sym("gnpu_nar_open"));
        npu_->nar_run = reinterpret_cast<decltype(npu_->nar_run)>(sym("gnpu_nar_run"));
        npu_->nar_close = reinterpret_cast<decltype(npu_->nar_close)>(sym("gnpu_nar_close"));
        npu_->shared_free = reinterpret_cast<decltype(npu_->shared_free)>(sym("gnpu_shared_free"));
        npu_->shared_map = reinterpret_cast<decltype(npu_->shared_map)>(sym("gnpu_shared_map"));
        npu_->attn = reinterpret_cast<decltype(npu_->attn)>(sym("gnpu_attn"));
        npu_->attn_buf = reinterpret_cast<decltype(npu_->attn_buf)>(sym("gnpu_attn_buffer"));
        npu_->dsp_s = reinterpret_cast<decltype(npu_->dsp_s)>(sym("gnpu_dsp_seconds"));
        npu_->conv_s = reinterpret_cast<decltype(npu_->conv_s)>(sym("gnpu_convert_seconds"));
        if (!open_fn || !set_par || !npu_->close || !npu_->shared_alloc || !npu_->gemm_f16 || !npu_->attn || !npu_->attn_buf) {
            err = "NAR: " + so + " lacks the calls the NAR needs";
            return false;
        }
        char why[256] = {0};
        npu_->g = open_fn(npu_dir.c_str(), why, static_cast<int>(sizeof why));
        if (!npu_->g) {
            err = std::string("NAR: the NPU is out of reach: ") + why;
            return false;
        }
        set_par(npu_->g, par_on_pool, &pool);
        backend_ = "dsp";
#else
        err = "NAR: no NPU here (the emulation: an empty npu dir)";
        return false;
#endif
    }
    slab_ = std::max(32, env_int("GEODE_YUE2_NAR_SLAB", 1024) / 32 * 32);
    // five HVX helpers beside HMX's thread (v79's six contexts: +19% over the default four)
    attn_flags_ = static_cast<uint32_t>(env_int("GEODE_YUE2_NAR_ATTN_FLAGS", ATTN_HELPERS(5)));
    mlp_ = npu_->mlp_f16 && env_int("GEODE_YUE2_NAR_MLP", 1) != 0;
    drv_ = npu_->nar_open && npu_->nar_run && npu_->nar_close && env_int("GEODE_YUE2_NAR_DRIVER", 1) != 0;
    // the driver's buffers mapped per group of layers (npu_probe window: 3840 MB on the 8 Elite; 128 MB left to the
    // DSP's own heap and stacks, which grow after that probe's mappings)
    lazy_ = drv_ && npu_->shared_map && npu_->shared_free && env_int("GEODE_YUE2_NAR_LAZY", 1) != 0;
    window_ = static_cast<size_t>(std::max(256, env_int("GEODE_YUE2_NAR_WINDOW_MB", 3712))) << 20;
    const int H = c_.hidden, hd = c_.head_dim, qd = c_.heads * hd, F = c_.ffn;
    if (hd % 32 || qd != 2048 || c_.heads % c_.kv_heads || 32 % (c_.heads / c_.kv_heads) || H % 32 || F % 32) {
        err = "NAR: a shape the DSP path does not take";
        return false;
    }
    // the CPU's parts
    const auto rd16 = [&](const std::string &name, const std::vector<int64_t> &shape, std::vector<uint16_t> &dst) {
        const TensorInfo *t = pack.find("h16." + name);
        if (!t || t->type != DType::BF16 || t->shape != shape) {
            err = "NAR: h16." + name + " missing or not as expected";
            return false;
        }
        dst.resize(t->bytes / 2);
        return pack.read_span(t->file, t->offset, t->bytes, dst.data()) || (err = "NAR: reading " + name, false);
    };
    const auto rd32 = [&](const std::string &name, int64_t n, std::vector<float> &dst) {
        const TensorInfo *t = pack.find("h16." + name);
        if (!t || t->type != DType::F32 || t->shape != std::vector<int64_t>{n}) {
            err = "NAR: h16." + name + " missing or not as expected";
            return false;
        }
        dst.resize(static_cast<size_t>(n));
        return pack.read_span(t->file, t->offset, t->bytes, dst.data()) || (err = "NAR: reading " + name, false);
    };
    const int Lt = c_.latent;
    return rd16("vae2llm.weight", {H, Lt}, vae2llm_w_) && rd32("vae2llm.bias", H, vae2llm_b_) &&
           rd16("llm2vae.weight", {Lt, H}, llm2vae_w_) && rd32("llm2vae.bias", Lt, llm2vae_b_) &&
           rd16("time_embedder.mlp.0.weight", {H, c_.freq}, t0_w_) && rd32("time_embedder.mlp.0.bias", H, t0_b_) &&
           rd16("time_embedder.mlp.2.weight", {H, H}, t2_w_) && rd32("time_embedder.mlp.2.bias", H, t2_b_) &&
           rd16("latent_pos_embed.pe", {c_.max_latent_frames, H}, pe_) && rd32("model.norm.weight", H, norm_);
}

// per layer: its stretch of nar_f16.bin / ar_f16.bin (the four matrices, then the norms) in one buffer the DSP maps
bool NarDsp::load_layers(bool ar, std::vector<LayerW> &out, std::string &err) {
    const int H = c_.hidden, hd = c_.head_dim, qd = c_.heads * hd, kd = c_.kv_heads * hd, F = c_.ffn;
    struct M { const char *name; int64_t rows, cols; };
    const M nar_m[4] = {{"nar_qkv", qd + 2 * kd, H}, {"nar_o", H, qd}, {"nar_gu", 2 * F, H}, {"nar_down", H, F}};
    const M ar_m[4] = {{"ar_qkv", qd + 2 * kd, H}, {"ar_o", H, qd}, {"ar_gu", 2 * F, H}, {"ar_down", H, F}};
    const char *nar_v[4] = {"nar_input_layernorm.weight", "nar_self_attn.q_norm.weight", "nar_self_attn.k_norm.weight",
                            "nar_pre_mlp_layernorm.weight"};
    const char *ar_v[4] = {"input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                           "post_attention_layernorm.weight"};
    const M *mats = ar ? ar_m : nar_m;
    const char *const *vecs = ar ? ar_v : nar_v;
    const int64_t vlen[4] = {H, hd, hd, H};
    // the 8-bit weights when asked for, the DSP's driver runs the layers (nar_run: per-product gnpu_gemm_f16 calls
    // would take their bytes for fp16) and the pack has them (made on the phone)
    const bool q8 = !ar && !cpu_ && drv_ && want_q8_ && pack_->find("h8.model.layers.0.nar_qkv");
    // the CPU backend's: the fp16 layers as they are (the GPU's), each matrix with its hmx8 twin for the CPU's share
    // (the turned c8.* when the pack has them - made for the CPU; else the DSP's h8.* as they are)
    const bool c8 = pack_->find("c8.model.layers.0.nar_qkv") != nullptr;
    const bool q8cpu = !ar && cpu_ && want_q8_ && have_q8_gemm() && (c8 || pack_->find("h8.model.layers.0.nar_qkv"));
    if (q8cpu) npu_->emu->q8rot = c8;
    out.assign(static_cast<size_t>(c_.layers), LayerW{});
    for (int l = 0; l < c_.layers; ++l) {
        const std::string b = std::string(ar ? "a16" : q8 ? "h8" : "h16") + ".model.layers." + std::to_string(l) + ".";
        const TensorInfo *mt[4], *vt[4];
        uint64_t lo = UINT64_MAX, hi = 0;
        int file = -1;
        for (int i = 0; i < 8; ++i) {
            const std::string name = b + (i < 4 ? mats[i].name : vecs[i - 4]);
            const TensorInfo *t = pack_->find(name);
            const bool ok = t && (i < 4 ? (q8 ? t->hmx8 : t->hmx16) && t->shape == std::vector<int64_t>{mats[i].rows, mats[i].cols}
                                        : t->type == DType::F32 && t->shape == std::vector<int64_t>{vlen[i - 4]});
            if (!ok) {
                err = "NAR: " + name + " missing or not as expected (tools/pack_yue2.py " + (ar ? "--ar-f16)" : "--nar-f16)");
                return false;
            }
            if (file >= 0 && t->file != file) {
                err = "NAR: layer " + std::to_string(l) + " spans two files";
                return false;
            }
            file = t->file;
            (i < 4 ? mt[i] : vt[i - 4]) = t;
            lo = std::min(lo, t->offset);
            hi = std::max(hi, t->offset + t->bytes);
        }
        if (lo % 4096) {
            err = "NAR: layer " + std::to_string(l) + "'s stretch not page-aligned";
            return false;
        }
        LayerW &L = out[static_cast<size_t>(l)];
        if (cpu_) {   // borrowed from the mapping (nothing copied: the page cache holds what the products read)
            const std::string &fname = pack_->file_name(file);
            if (!pack_->map(fname, err)) return false;
            const uint8_t *at = pack_->data(*mt[0]) - (mt[0]->offset - lo);
            L.w.p = const_cast<uint8_t *>(at), L.w.len = 0;   // (len 0: not ours to free)
            L.w.fd = emu_view(npu_->emu.get(), at, hi - lo);
            if (q8cpu) {
                const std::string b8 = std::string(c8 ? "c8" : "h8") + ".model.layers." + std::to_string(l) + ".";
                const uint8_t *tw[4];
                for (int i = 0; i < 4; ++i) {
                    const TensorInfo *t8 = pack_->find(b8 + mats[i].name);
                    if (!t8 || !t8->hmx8 || t8->shape != std::vector<int64_t>{mats[i].rows, mats[i].cols} ||
                        t8->shift != mt[i]->shift) {
                        err = "NAR: " + b8 + mats[i].name + " missing or not as its fp16 one";
                        return false;
                    }
                    if (!pack_->map(pack_->file_name(t8->file), err)) return false;
                    tw[i] = pack_->data(*t8);
                }
                auto &q8m = npu_->emu->q8;
                const int fd = L.w.fd;
                const uint64_t F8 = static_cast<uint64_t>(c_.ffn), H8 = static_cast<uint64_t>(c_.hidden);
                const uint64_t o16[4] = {mt[0]->offset - lo, mt[1]->offset - lo, mt[2]->offset - lo, mt[3]->offset - lo};
                const int keep = env_int("GEODE_YUE2_NAR_Q8_KEEP", 0);   // (experiments: 1 qkv 2 o 4 gate/up 8 down in fp16)
                for (int i = 0; i < 4; ++i)
                    if (!(keep & (1 << i))) q8m[{fd, o16[i]}] = tw[i];
                if (!(keep & 4)) q8m[{fd, o16[2] + F8 * H8 * 2u}] = tw[2] + (F8 / 32u) * H8 * 34u;   // (up: gate's F rows first)
                L.q8cpu = true;
            }
        } else {
            if (!alloc(L.w, (hi - lo + 4095) / 4096 * 4096, err)) return false;
            if (!pack_->read_span(file, lo, hi - lo, L.w.p)) {
                err = "NAR: reading layer " + std::to_string(l) + " failed";
                return false;
            }
        }
        L.qkv = mt[0]->offset - lo, L.o = mt[1]->offset - lo, L.gu = mt[2]->offset - lo, L.down = mt[3]->offset - lo;
        L.sh_qkv = mt[0]->shift, L.sh_o = mt[1]->shift, L.sh_gu = mt[2]->shift, L.sh_down = mt[3]->shift;
        L.q8 = q8;
        L.in_norm = reinterpret_cast<const float *>(L.w.p + (vt[0]->offset - lo));
        L.q_norm = reinterpret_cast<const float *>(L.w.p + (vt[1]->offset - lo));
        L.k_norm = reinterpret_cast<const float *>(L.w.p + (vt[2]->offset - lo));
        L.mlp_norm = reinterpret_cast<const float *>(L.w.p + (vt[3]->offset - lo));
    }
    return true;
}

void NarDsp::drop_layers(std::vector<LayerW> &Ls) {
    for (LayerW &L : Ls) free_buf(L.w);   // (borrowed ones, len 0, are the mapping's; none: the session keeps them)
    if (gls_ == &Ls) gls_ = nullptr;
    Ls.clear();
}

void NarDsp::set_attn8(bool on) {
    want_attn8_ = on;
    if (npu_ && npu_->emu) npu_->emu->attn8_opt = on || env_int("GEODE_YUE2_NAR_ATTN8", 0) != 0;
}

bool NarDsp::ensure_nar(std::string &err) {
    join_loads();
    if (!layers_.empty()) return true;
    if (!nar_err_.empty())   // (load_nar_async's failure: read again here, as without it)
        std::fprintf(stderr, "NAR: reading the layers early failed (%s): read now\n", nar_err_.c_str()), nar_err_.clear();
    return load_layers(false, layers_, err);
}

void NarDsp::join_loads() {
    if (nar_thread_.joinable()) nar_thread_.join();
    if (ar_thread_.joinable()) ar_thread_.join();
}

bool NarDsp::load_nar_async() {
    if (nar_thread_.joinable() || !layers_.empty() || !npu_ || backend_ != "dsp" || !drv_ || !lazy_ ||
        !npu_->shared_free || !pack_ || env_int("GEODE_YUE2_NAR_EARLY", 1) == 0)
        return false;
    const bool q8 = want_q8_ && pack_->find("h8.model.layers.0.nar_qkv");
    int64_t bytes = 0;
    for (int l = 0; l < c_.layers; ++l)
        for (const char *m : {"nar_qkv", "nar_o", "nar_gu", "nar_down"})
            if (const TensorInfo *t = pack_->find(std::string(q8 ? "h8" : "h16") + ".model.layers." + std::to_string(l) + "." + m))
                bytes += static_cast<int64_t>(t->bytes);
    if (mem_available() < bytes + (int64_t{3} << 30)) return false;
    nar_err_.clear();
    nar_thread_ = std::thread([this, bytes] {   // (nothing else calls the NPU meanwhile: every caller joins first)
        if (ar_thread_.joinable()) ar_thread_.join();   // (gnpu's tables are one thread's at a time)
        // again now that the AR's weights are in (a short prompt never waited for them): the room may be gone
        if (mem_available() < bytes + (int64_t{3} << 30)) return;
#if defined(__ANDROID__)
        // below the generation's threads (every core busy with its lockstep products): a share of a core, not one
        // taken from a product (at nice 0 the codes lost 1.6 s of the 3 s this saves; GEODE_YUE2_NAR_EARLY_NICE)
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), env_int("GEODE_YUE2_NAR_EARLY_NICE", 10));
#endif
        try {
            if (!load_layers(false, layers_, nar_err_)) drop_layers(layers_);
        } catch (const std::exception &e) {   // (ensure_nar reads them again on the caller's thread)
            nar_err_ = e.what();
            drop_layers(layers_);
        }
    });
    return true;
}

int64_t NarDsp::ar_bytes() const {
    int64_t bytes = 0;
    for (int l = 0; l < c_.layers; ++l)
        for (const char *m : {"ar_qkv", "ar_o", "ar_gu", "ar_down"})
            if (const TensorInfo *t = pack_->find("a16.model.layers." + std::to_string(l) + "." + m)) bytes += static_cast<int64_t>(t->bytes);
    return bytes;
}

bool NarDsp::load_ar_async() {
    if (ar_wanted_) return true;
    if (!npu_ || backend_ != "dsp" || !drv_ || !npu_->shared_free || !pack_ || !has_ar_f16(*pack_) ||
        env_int("GEODE_YUE2_DSP_PROMPT", 1) == 0 || nar_thread_.joinable())
        return false;
    if (mem_available() < ar_bytes() + (int64_t{3} << 30)) return false;
    ar_wanted_ = true;
    ar_err_.clear();
    ar_thread_ = std::thread([this] {   // (nothing else calls the NPU meanwhile: ensure_ar joins first)
        if (!load_layers(true, ar_, ar_err_)) drop_layers(ar_);
    });
    return true;
}

bool NarDsp::ensure_ar(std::string &err) {
    join_loads();
    if (!ar_.empty()) return true;
    if (!ar_err_.empty()) {
        err = ar_err_;
        ar_err_.clear();
        return false;
    }
    if (load_layers(true, ar_, err)) return true;
    drop_layers(ar_);
    return false;
}

bool NarDsp::embed_x(const int32_t *ids, int n, std::string &err) {
    const int H = c_.hidden;
    const TensorInfo *emb = pack_->find("model.embed_tokens.weight");
    if (!emb || emb->type != DType::BF16 || emb->shape != std::vector<int64_t>{c_.vocab, H}) {
        err = "NAR: model.embed_tokens.weight missing or not bf16 [vocab, hidden]";
        return false;
    }
    float *X = reinterpret_cast<float *>(io_.p);   // (IO's x: offset 0)
    std::vector<uint16_t> row(static_cast<size_t>(H));
    for (int r = 0; r < n; ++r) {
        if (ids[r] < 0 || ids[r] >= c_.vocab) {
            err = "NAR: token " + std::to_string(ids[r]) + " outside the vocabulary";
            return false;
        }
        if (!pack_->read_rows(*emb, ids[r], 1, row.data())) {
            err = "NAR: reading the embeddings failed";
            return false;
        }
        for (int i = 0; i < H; ++i) X[static_cast<size_t>(r) * H + i] = bf16f(row[static_cast<size_t>(i)]);
    }
    return true;
}

bool NarDsp::ar_prompt(const int32_t *ids, int n, const PrefixFn &out, std::string &err, double *seconds) {
    if (!npu_ || !drv_ || !npu_->shared_free) {
        err = "NAR: the AR's prompt on the DSP wants its driver (libgeode_npu.so's gnpu_nar_*, gnpu_shared_free)";
        return false;
    }
    if (n < 1) return true;
    if (!ensure_ar(err) || !setup_chunk(n, 1, err)) return false;   // (frames 1: the rows' bank's least; unused)
    nar_cfg_t cf;
    std::memset(&cf, 0, sizeof cf);
    cf.mode = NAR_MODE_PREFIX, cf.A = 0, cf.n = n;
    const Prof keep = prof_;
    prof_ = Prof{};
    bool ok = open_cfg(cf, ar_, 0, err) && embed_x(ids, n, err) && run_layers(0, prefix_flags(), err);
    if (seconds) *seconds = prof_.dsp_call;
    prof_ = keep;
    close_driver();
    if (ok) {
        const size_t kd = static_cast<size_t>(c_.kv_heads) * c_.head_dim;
        std::vector<float> k(static_cast<size_t>(n) * kd), v(k.size());
        for (int l = 0; l < c_.layers && ok; ++l) {
            ok = prefix_kv(l, k.data(), v.data()) && out(l, k.data(), v.data());
            if (!ok) err = "NAR: the prompt's layer " + std::to_string(l) + " not taken";
        }
    }
    A_ = frames_ = N_ = 0;   // (no chunk: begin first for velocities)
    return ok;
}

void NarDsp::close_driver() {
    if (drv_open_ && npu_->nar_close) npu_->nar_close(npu_->g);
    drv_open_ = false;
    group_ = -1;
}

bool NarDsp::has_ar_f16(const Pack &pack) {
    const TensorInfo *t = pack.find("a16.model.layers.0.ar_qkv");
    return t && t->hmx16;
}

bool NarDsp::setup_chunk(int A, int frames, std::string &err) {
    if (!npu_) {
        err = "NAR: not open";
        return false;
    }
    if (A < 1 || frames < 1 || A + frames + 2 > c_.tok.context || frames + 2 > c_.max_latent_frames) {
        err = "NAR: " + std::to_string(frames) + " frames after " + std::to_string(A) + " AR positions exceed the context";
        return false;
    }
    close_driver();   // (it maps the banks this rewrites)
    join_loads();   // (the early reads' allocations: gnpu's tables are one thread's)
    const int hd = c_.head_dim, nkv = c_.kv_heads, dgs = hd / 32;
    A_ = A, frames_ = frames, N_ = frames + 2;
    npre_ = A / 32, rem_ = A - 32 * npre_;
    const int total = (A + N_ + 31) / 32;
    ntail_ = total - npre_;
    const size_t blk = static_cast<size_t>(nkv) * static_cast<size_t>(dgs) * 2048u;   // a block of every KV head
    pre_k_.resize(static_cast<size_t>(c_.layers));
    pre_v_.resize(static_cast<size_t>(c_.layers));
    rem_k_.assign(static_cast<size_t>(c_.layers), {});
    rem_v_.assign(static_cast<size_t>(c_.layers), {});
    for (int l = 0; l < c_.layers; ++l) {
        if (!npre_) {
            free_buf(pre_k_[static_cast<size_t>(l)]), free_buf(pre_v_[static_cast<size_t>(l)]);
            continue;
        }
        if (!alloc(pre_k_[static_cast<size_t>(l)], blk * static_cast<size_t>(npre_), err) ||
            !alloc(pre_v_[static_cast<size_t>(l)], blk * static_cast<size_t>(npre_), err))
            return false;
    }
    if (!alloc(tail_k_, blk * static_cast<size_t>(ntail_), err) || !alloc(tail_v_, blk * static_cast<size_t>(ntail_), err))
        return false;
    std::memset(tail_k_.p, 0, tail_k_.len);   // positions past A + N: zeros (masked; their values must be finite)
    std::memset(tail_v_.p, 0, tail_v_.len);
    if (rem_) {   // RB: a block of every KV head a layer, keys then values; past the A % 32 positions zeros (masked)
        if (!alloc(rb_, 2u * blk * static_cast<size_t>(c_.layers), err)) return false;
        std::memset(rb_.p, 0, rb_.len);
    }
    // RoPE at the rows' positions (yue2.cpp rope_rows: the angle in fp32, as torch)
    const int half = hd / 2;
    cos_.resize(static_cast<size_t>(N_) * half);
    sin_.resize(cos_.size());
    for (int r = 0; r < N_; ++r)
        for (int i = 0; i < half; ++i) {
            const float angle = static_cast<float>(A + r) * c_.inv_freq[static_cast<size_t>(i)];
            cos_[static_cast<size_t>(r) * half + i] = std::cos(angle);
            sin_[static_cast<size_t>(r) * half + i] = std::sin(angle);
        }
    return true;
}

bool NarDsp::begin(int A, int frames, const PrefixFn &prefix, std::string &err) {
    if (!setup_chunk(A, frames, err) || !ensure_nar(err)) return false;
    const int hd = c_.head_dim, nkv = c_.kv_heads, kd = nkv * hd, dgs = hd / 32, H = c_.hidden, F = c_.ffn;
    const size_t blk = static_cast<size_t>(nkv) * static_cast<size_t>(dgs) * 2048u;
    std::vector<float> k(static_cast<size_t>(A) * kd), v(k.size());
    for (int l = 0; l < c_.layers; ++l) {
        if (!prefix(l, k.data(), v.data())) {
            err = "NAR: the prefix's layer " + std::to_string(l) + " not given";
            return false;
        }
        if (npre_) {
            Buf &K = pre_k_[static_cast<size_t>(l)], &V = pre_v_[static_cast<size_t>(l)];
            _Float16 *kb = reinterpret_cast<_Float16 *>(K.p), *vb = reinterpret_cast<_Float16 *>(V.p);
            const size_t nb = static_cast<size_t>(npre_);
            pool_->parallel_for(static_cast<int64_t>(npre_) * 32, [&](int64_t a, int64_t b) {
                for (int64_t p = a; p < b; ++p)
                    for (int kh = 0; kh < nkv; ++kh)
                        put_kv(kb, vb, nb, dgs, kh, static_cast<int>(p), &k[static_cast<size_t>(p) * kd + static_cast<size_t>(kh) * hd],
                               &v[static_cast<size_t>(p) * kd + static_cast<size_t>(kh) * hd], hd);
            }, 64);
        }
        rem_k_[static_cast<size_t>(l)].assign(k.begin() + static_cast<ptrdiff_t>(npre_) * 32 * kd, k.end());
        rem_v_[static_cast<size_t>(l)].assign(v.begin() + static_cast<ptrdiff_t>(npre_) * 32 * kd, v.end());
        if (rem_) {   // RB's image (nb 1), as the DSP's prefix writes it
            _Float16 *K = reinterpret_cast<_Float16 *>(rb_.p + 2u * blk * static_cast<size_t>(l)), *V = K + blk / 2u;
            for (int q = 0; q < rem_; ++q)
                for (int kh = 0; kh < nkv; ++kh)
                    put_kv(K, V, 1, dgs, kh, q, &rem_k_[static_cast<size_t>(l)][static_cast<size_t>(q) * kd + static_cast<size_t>(kh) * hd],
                           &rem_v_[static_cast<size_t>(l)][static_cast<size_t>(q) * kd + static_cast<size_t>(kh) * hd], hd);
        }
    }
    if (drv_) return open_driver(err);
    const size_t qbytes = static_cast<size_t>(N_) * c_.heads * hd * sizeof(float);
    qb_ = npu_->attn_buf(npu_->g, 0, qbytes);
    ob_ = npu_->attn_buf(npu_->g, 1, qbytes);
    if (!qb_ || !ob_) {
        err = "NAR: no memory for the attention's rows";
        return false;
    }
    const size_t S = static_cast<size_t>(slab_);
    x_.resize(static_cast<size_t>(N_) * H);
    h_.resize(S * H);
    qkv_.resize(S * static_cast<size_t>(c_.heads * hd + 2 * kd));
    o_.resize(S * H);
    gu_.resize(S * 2 * static_cast<size_t>(F));
    act_.resize(S * static_cast<size_t>(F));
    return true;
}

bool NarDsp::gemm(const Buf &w, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x, int64_t xs,
                  float *y, int64_t ys, std::string &err) {
    std::vector<const float *> xp(static_cast<size_t>(n));
    std::vector<float *> yp(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) xp[static_cast<size_t>(t)] = x + t * xs, yp[static_cast<size_t>(t)] = y + t * ys;
    // (the matrix at byte off of its buffer: gnpu_gemm_f16 names it by the buffer's fd)
    const int e = npu_->gemm_f16(npu_->g, w.fd, off, rows, cols, shift, n, xp.data(), yp.data());
    if (e) err = "NAR: HMX product " + std::to_string(rows) + "x" + std::to_string(cols) + " failed (" + std::to_string(e) + ")" +
                 (npu_->emu && !npu_->emu->gpu_err.empty() ? ": " + npu_->emu->gpu_err : std::string());
    return e == 0;
}

bool NarDsp::layer(int l, std::string &err) {
    const LayerW &L = layers_[static_cast<size_t>(l)];
    const int H = c_.hidden, hd = c_.head_dim, nh = c_.heads, nkv = c_.kv_heads, qd = nh * hd, kd = nkv * hd, F = c_.ffn;
    const int dgs = hd / 32, half = hd / 2, W = qd + 2 * kd, N = N_;
    // SDPA's 1 / sqrt(hd) and log2 e folded into the queries: the DSP's scores in log2 units (ATTN_LOG2)
    const float qscale = 1.4426950408889634f / std::sqrt(static_cast<float>(hd));
    _Float16 *tk = reinterpret_cast<_Float16 *>(tail_k_.p), *tv = reinterpret_cast<_Float16 *>(tail_v_.p);
    const size_t nb = static_cast<size_t>(ntail_);
    double t = now_seconds();
    // the prefix's last A % 32 positions into the rows' bank (the block it shares with them)
    for (int p = 0; p < rem_; ++p)
        for (int kh = 0; kh < nkv; ++kh)
            put_kv(tk, tv, nb, dgs, kh, p, &rem_k_[static_cast<size_t>(l)][static_cast<size_t>(p) * kd + static_cast<size_t>(kh) * hd],
                   &rem_v_[static_cast<size_t>(l)][static_cast<size_t>(p) * kd + static_cast<size_t>(kh) * hd], hd);
    // q k v for every row (each row's keys before any row attends: the NAR sees them all)
    for (int s0 = 0; s0 < N; s0 += slab_) {
        const int n = std::min(slab_, N - s0);
        const double t0 = now_seconds();
        pool_->parallel_for(n, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; ++r)
                rms_row(&x_[static_cast<size_t>(s0 + r) * H], L.in_norm, &h_[static_cast<size_t>(r) * H], H, c_.eps);
        });
        const double t1 = now_seconds();
        if (!gemm(L.w, L.qkv, W, H, L.sh_qkv, n, h_.data(), H, qkv_.data(), W, err)) return false;
        const double t2 = now_seconds();
        pool_->parallel_for(n, [&](int64_t a, int64_t b) {
            float tmp[512];
            for (int64_t r = a; r < b; ++r) {
                const int row = s0 + static_cast<int>(r);
                float *qkv = &qkv_[static_cast<size_t>(r) * W];
                const float *cs = &cos_[static_cast<size_t>(row) * half], *sn = &sin_[static_cast<size_t>(row) * half];
                float *qo = qb_ + static_cast<size_t>(row) * qd;
                for (int h = 0; h < nh; ++h) {
                    rms_row(qkv + h * hd, L.q_norm, tmp, hd, c_.eps);
                    rope_head(tmp, cs, sn, hd);
                    for (int c = 0; c < hd; ++c) qo[h * hd + c] = tmp[c] * qscale;
                }
                for (int kh = 0; kh < nkv; ++kh) {
                    rms_row(qkv + qd + kh * hd, L.k_norm, tmp, hd, c_.eps);
                    rope_head(tmp, cs, sn, hd);
                    put_kv(tk, tv, nb, dgs, kh, rem_ + row, tmp, qkv + qd + kd + kh * hd, hd);
                }
            }
        });
        prof_.cpu += (t1 - t0) + (now_seconds() - t2);
        prof_.qkv += t2 - t1;
    }
    // the attention: every row over the prefix's A keys and the N rows' own
    t = now_seconds();
    attn_desc_t d;
    std::memset(&d, 0, sizeof d);
    d.version = ATTN_VERSION;
    d.n = N, d.apos = 0, d.nh = nh, d.nkv = nkv, d.hd = hd, d.window = 0;
    d.n_attn = 1, d.kv_chunk = 32;
    d.q_stride = d.o_stride = qd;
    d.flags = static_cast<int32_t>(ATTN_ENCODER | ATTN_LOG2 | attn_flags_);
    d.nk = A_ + N, d.npre = npre_;
    if (npre_) {
        d.n_chunks = 4;
        d.kv_fd[0] = pre_k_[static_cast<size_t>(l)].fd, d.kv_fd[1] = pre_v_[static_cast<size_t>(l)].fd;
        d.kv_fd[2] = tail_k_.fd, d.kv_fd[3] = tail_v_.fd;
    } else {
        d.n_chunks = 2;
        d.kv_fd[0] = tail_k_.fd, d.kv_fd[1] = tail_v_.fd;
    }
    if (pipe_) return pipelined(L, &d, err);
    double ds = 0.0;
    const size_t qbytes = static_cast<size_t>(N) * qd * sizeof(float);
    if (const int e = npu_->attn(npu_->g, &d, static_cast<int>(sizeof d), qb_, qbytes, ob_, qbytes, &ds)) {
        err = "NAR: the DSP's attention failed (" + std::to_string(e) + ")";
        return false;
    }
    prof_.attn += now_seconds() - t;
    prof_.attn_dsp += ds;
    return post_attn(L, 0, N, Tail::Hybrid, err);
}

// The GPU backend's attention overlapped with its products: the rows' tokens in slices. The CPU's pool computes the
// slices' attention in order while a helper thread has the GPU do o, the residual and the MLP of the slices already
// attended, from the front, alone (its glue on the helper: the pool is the attention's). When the attention is done
// the CPU takes the slices from the back, its own products (its own buffers), until the two meet - so neither waits
// on the other for long. Redmi, a velocity: the GPU's o + MLP of the slices ~2.5 T MAC against the attention's 62 s.
bool NarDsp::pipelined(const LayerW &L, const void *desc, std::string &err) {
    const attn_desc_t &d = *static_cast<const attn_desc_t *>(desc);
    Emu *e = npu_->emu.get();
    const int N = N_, H = c_.hidden, F = c_.ffn;
    const double t = now_seconds();
    const _Float16 *bank[4];
    if (!attn_banks(e, d, bank) || cpu_attn_prepare(e, d, bank)) {
        err = "NAR: the attention's banks";
        return false;
    }
    // the attention in S slices; the products' units k to a slice (the last ones shared out finer)
    const int S = std::max(2, std::min(pipe_slices_, N / 64)), k = std::max(1, std::min(pipe_split_, N / (S * 32))),
              U = S * k;
    std::vector<int> b(static_cast<size_t>(U) + 1);
    for (int i = 0; i <= U; ++i) b[static_cast<size_t>(i)] = static_cast<int>(static_cast<int64_t>(N) * i / U);
    // the CPU's own scratch for its units (the helper has h_, o_, gu_, act_)
    const size_t rows2 = static_cast<size_t>(std::min(slab_, (N + U - 1) / U));
    if (h2_.size() < rows2 * H) h2_.resize(rows2 * H), o2_.resize(rows2 * H);
    if (gu2_.size() < rows2 * 2 * F) gu2_.resize(rows2 * 2 * F), act2_.resize(rows2 * F);
    std::mutex m;
    std::condition_variable cv;
    int ready = 0, front = 0, back = U - 1;   // units attended; the next the GPU takes; the last nobody took
    bool stop = false, hok = true;
    std::string herr;
    hprof_ = Prof{};
    // the next layer (the last's: the first, the next velocity's) into the GPU's other slot while the first slice's
    // attention runs - with fewer slots than layers its qkv would wait for the copy (~100 MB) otherwise
    const size_t next = (static_cast<size_t>(&L - layers_.data()) + 1) % layers_.size();
#if !defined(_WIN32)
    if (gpu_bits_ & 8) {   // the layer after next from storage meanwhile (its copy into the GPU then reads memory)
        size_t len = 0;
        const size_t li = (next + 1) % layers_.size();
        if (const uint8_t *w = emu_buf(e, layers_[li].w.fd, &len); w && len) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(w) & ~static_cast<uintptr_t>(4095);
            madvise(reinterpret_cast<void *>(a), len + (reinterpret_cast<uintptr_t>(w) - a), MADV_WILLNEED);
        }
    }
#endif
    // the helper's units wholly on the GPU, two in flight while the attention runs (the queue never empty between
    // them); from the attention's end one at a time (the CPU takes units from the back: the GPU's queue short)
    const bool async = (gpu_bits_ & 3) == 3 && (N + U - 1) / U <= slab_;
    bool attending = true;
    NarGpu::UnitShape us;
    us.o = L.o, us.gu = L.gu, us.down = L.down, us.sh_o = L.sh_o, us.sh_gu = L.sh_gu, us.sh_down = L.sh_down;
    us.H = H, us.qd = c_.heads * c_.head_dim, us.F = F, us.eps = c_.eps;
    std::thread helper([&] {
        if (prefetch_) {   // this layer first (the most recent then: its qkv may not have used the GPU), the next
            std::string perr;
            for (const size_t li : {static_cast<size_t>(&L - layers_.data()), next}) {
                size_t len = 0;
                const uint8_t *w = emu_buf(e, layers_[li].w.fd, &len);
                if (!w || !npu_->gpu->prefetch(w, len, nullptr, perr)) break;   // (the products' own copies try again)
            }
        }
        size_t wlen = 0;
        const uint8_t *wbase = async ? emu_buf(e, L.w.fd, &wlen) : nullptr;
        const auto collect = [&] {   // the oldest unit in flight
            const double t0 = now_seconds();
            double k = 0.0;
            const bool r = npu_->gpu->unit_wait(herr, &k);
            hprof_.gu += now_seconds() - t0, hprof_.gpu_k += k;
            return r;
        };
        bool failed = async && !wbase;
        if (failed) herr = "NAR: the layer's weights for the GPU";
        while (!failed) {
            int i;
            {
                std::unique_lock<std::mutex> lk(m);
                const int fl = async ? npu_->gpu->units_in_flight() : 0;
                // another unit behind the one in flight only while the attention runs and one is ready now
                if (fl >= 2 || (fl == 1 && !(attending && ready > front && front <= back && !stop))) {
                    lk.unlock();
                    if (!collect()) failed = true;
                    continue;
                }
                const double tw = now_seconds();
                cv.wait(lk, [&] { return ready > front || front > back || stop; });
                hprof_.helper_wait += now_seconds() - tw;
                if (stop || front > back || ready <= front) break;   // (all taken, or a failure)
                i = front++;
            }
            const int r0 = b[static_cast<size_t>(i)], r1 = b[static_cast<size_t>(i) + 1];
            if (async) {
                const double t0 = now_seconds();
                failed = !npu_->gpu->unit(wbase, wlen, us, L.mlp_norm, r1 - r0, ob_ + static_cast<size_t>(r0) * us.qd,
                                          &x_[static_cast<size_t>(r0) * H], herr, nullptr, false);
                hprof_.gu += now_seconds() - t0;
            } else {
                failed = !post_attn(L, r0, r1, Tail::Gpu, herr);
            }
        }
        while (!failed && async && npu_->gpu->units_in_flight() > 0)   // (every unit's rows whole before the join)
            if (!collect()) failed = true;
        if (failed) {
            std::lock_guard<std::mutex> lk(m);
            hok = false;
        }
    });
    // the v66 DSP's helper (the Fast sound on a DSP without HMX, set_hvx): units from the front too, but only while
    // the attention runs (after it the CPU does a unit ~5x faster); the layer's twins into its slot first (while the
    // first slice is attended). A unit it fails the CPU takes again after the join (its rows untouched), and the DSP
    // is left out from then on.
    Prof dprof;
    int dsp_taken = 0;
    std::vector<std::pair<int, int>> dsp_failed;   // rows the DSP took and did not finish
    std::string derr;
    std::thread dsp_helper;
    if (hvx() && L.q8cpu)
        dsp_helper = std::thread([&] {
#if !defined(_WIN32)
            // (above the attention's threads when asked: its CPU side keeps the DSP fed, the attention is the bound)
            if (hvx_nice_) setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), hvx_nice_);
#endif
            const double ts = now_seconds();
            bool failed = !hvx_slot(L, derr);
#if !defined(_WIN32)
            // the next layer's twins from storage meanwhile (the c8 file seldom stays in the page cache beside the
            // GPU's fp16 one: its copy into the slot then reads memory)
            if (!failed && hvx_ahead_) {
                const LayerW &N = layers_[next];
                for (const uint64_t o : {N.o, N.gu, N.down})
                    if (const auto it = npu_->emu->q8.find({N.w.fd, o}); it != npu_->emu->q8.end()) {
                        const int rows = o == N.gu ? 2 * c_.ffn : c_.hidden, k = o == N.o ? c_.heads * c_.head_dim
                                                                                 : o == N.gu ? c_.hidden : c_.ffn;
                        const uintptr_t a = reinterpret_cast<uintptr_t>(it->second) & ~static_cast<uintptr_t>(4095);
                        const size_t len = static_cast<size_t>(rows / 32) * static_cast<size_t>(k) * 34u +
                                           (reinterpret_cast<uintptr_t>(it->second) - a);
                        madvise(reinterpret_cast<void *>(a), len, MADV_WILLNEED);
                    }
            }
#endif
            dprof.hvx_slot += now_seconds() - ts;
            while (!failed) {
                int i;
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return ready > front || front > back || stop || !attending; });
                    if (stop || front > back || !attending || ready <= front) break;
                    i = front++;
                    ++dsp_taken;
                }
                int upto = b[static_cast<size_t>(i)];
                if (!hvx_unit(L, b[static_cast<size_t>(i)], b[static_cast<size_t>(i) + 1], dprof, derr, upto)) {
                    failed = true;
                    std::lock_guard<std::mutex> lk(m);
                    dsp_failed.emplace_back(upto, b[static_cast<size_t>(i) + 1]);
                }
            }
            if (failed) hvx_off_ = true, hvx_note_ = derr;
        });
    bool ok = true;
    const int w0 = pool_->width();   // (a core left to the helper, when asked: it feeds the GPU)
    if (pipe_width_ > 0) pool_->set_width(std::min(pipe_width_, w0));
    std::vector<int> cuts;   // the attention's slices by their last unit
    if ((gpu_bits_ & 4) && k > 1) cuts.push_back(1);
    for (int i = 1; i <= S; ++i) cuts.push_back(i * k);
    for (size_t i = 0, u0 = 0; i < cuts.size() && ok; u0 = static_cast<size_t>(cuts[i++])) {
        ok = cpu_attn_rows(e, d, qb_, ob_, b[u0], b[static_cast<size_t>(cuts[i])]) == 0;
        {
            std::lock_guard<std::mutex> lk(m);
            if (ok) ready = cuts[i];
            else stop = true;
        }
        cv.notify_all();
    }
    {
        std::lock_guard<std::mutex> lk(m);
        attending = false;
    }
    cv.notify_all();
    pool_->set_width(w0);
    prof_.attn += now_seconds() - t;
    // the slices the GPU has not taken, from the back, on the CPU
    while (ok) {
        int j;
        {
            std::lock_guard<std::mutex> lk(m);
            if (!hok || back < front) break;
            j = back--;
        }
        cv.notify_all();   // (the helper may be waiting for a slice that is now taken)
        if (!post_attn(L, b[static_cast<size_t>(j)], b[static_cast<size_t>(j) + 1], Tail::Cpu, err)) {
            std::lock_guard<std::mutex> lk(m);
            stop = true, ok = false;
            break;
        }
    }
    {
        std::lock_guard<std::mutex> lk(m);
        stop = true;
    }
    cv.notify_all();
    helper.join();
    if (dsp_helper.joinable()) dsp_helper.join();
    for (const auto &f : dsp_failed)   // (what the DSP did not finish: those rows as the attention left them)
        if (ok && !post_attn(L, f.first, f.second, Tail::Cpu, err)) ok = false;
    prof_.hvx_dsp += dprof.hvx_dsp, prof_.hvx_call += dprof.hvx_call, prof_.hvx_cpu += dprof.hvx_cpu;
    prof_.hvx_slot += dprof.hvx_slot, prof_.hvx_units += dsp_taken;
    prof_.o += hprof_.o, prof_.gu += hprof_.gu, prof_.down += hprof_.down, prof_.cpu += hprof_.cpu;
    prof_.gpu_in += hprof_.gpu_in, prof_.gpu_fin += hprof_.gpu_fin, prof_.gpu_k += hprof_.gpu_k;
    prof_.helper_wait += hprof_.helper_wait;
    prof_.overlap += now_seconds() - t;
    ++prof_.gpu_slices_n, prof_.gpu_slices += front - dsp_taken, prof_.units = U;
    if (!ok) {
        if (err.empty()) err = "NAR: the attention failed";
        return false;
    }
    if (!hok) {
        err = herr;
        return false;
    }
    return true;
}

std::pair<double, int64_t> NarDsp::gpu_uploads() const {
    if (!npu_ || !npu_->gpu) return {0.0, 0};
    const NarGpu::Times &t = npu_->gpu->times();
    return {t.upload, t.uploads};
}

// A product on the GPU alone, from the helper thread (no pool: it is the attention's)
bool NarDsp::gpu_product(const LayerW &L, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x,
                         int64_t xs, float *y, int64_t ys, std::string &err) {
    size_t len = 0;
    const uint8_t *base = emu_buf(npu_->emu.get(), L.w.fd, &len);
    if (!base) {
        err = "NAR: the layer's weights for the GPU";
        return false;
    }
    std::vector<const float *> xp(static_cast<size_t>(n));
    std::vector<float *> yp(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) xp[static_cast<size_t>(t)] = x + t * xs, yp[static_cast<size_t>(t)] = y + t * ys;
    const double t0 = now_seconds();
    if (!npu_->gpu->start(base, len, off, rows, cols, shift, n, xp.data(), nullptr, err)) return false;
    const double t1 = now_seconds();
    double k = 0.0;
    if (!npu_->gpu->finish(yp.data(), nullptr, err, &k)) return false;
    hprof_.gpu_in += t1 - t0, hprof_.gpu_fin += now_seconds() - t1, hprof_.gpu_k += k;
    return true;
}

// A product on the CPU alone, while the helper has the GPU (the pipeline's back slices)
bool NarDsp::cpu_product(const LayerW &L, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x,
                         int64_t xs, float *y, int64_t ys, std::string &err) {
    const uint8_t *base = emu_buf(npu_->emu.get(), L.w.fd, nullptr);
    if (!base) {
        err = "NAR: the layer's weights for the CPU";
        return false;
    }
    std::vector<const float *> xp(static_cast<size_t>(n));
    std::vector<float *> yp(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) xp[static_cast<size_t>(t)] = x + t * xs, yp[static_cast<size_t>(t)] = y + t * ys;
    cpu_rows(npu_->emu.get(), base, L.w.fd, off, 0, rows, cols, shift, n, xp.data(), yp.data());
    return true;
}

// o, the residual, the MLP, the residual for the rows' tokens [r0, r1) - a slab of rows at a time. Tail::Hybrid: the
// products through gemm() (the backend's own; the GPU's shared with the CPU), the pool; Tail::Gpu: the pipeline's
// helper thread - the GPU's products alone, the glue here, the times into hprof_; Tail::Cpu: the CPU's products and
// its own buffers, beside the helper.
bool NarDsp::post_attn(const LayerW &L, int r0, int r1, Tail mode, std::string &err) {
    const int H = c_.hidden, qd = c_.heads * c_.head_dim, F = c_.ffn;
    const bool helper = mode == Tail::Gpu, mine = mode == Tail::Cpu;
    Prof &P = helper ? hprof_ : prof_;
    float *h = mine ? h2_.data() : h_.data(), *o = mine ? o2_.data() : o_.data();
    float *gu = mine ? gu2_.data() : gu_.data(), *act = mine ? act2_.data() : act_.data();
    const int slab = mine ? static_cast<int>(h2_.size() / static_cast<size_t>(H)) : slab_;
    const auto par = [&](int64_t n, const std::function<void(int64_t, int64_t)> &fn, int64_t min_chunk = 1) {
        if (helper) fn(0, n);
        else pool_->parallel_for(n, fn, min_chunk);
    };
    const auto prod = [&](uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x, int64_t xs, float *y,
                          int64_t ys) {
        return helper ? gpu_product(L, off, rows, cols, shift, n, x, xs, y, ys, err)
               : mine ? cpu_product(L, off, rows, cols, shift, n, x, xs, y, ys, err)
                      : gemm(L.w, off, rows, cols, shift, n, x, xs, y, ys, err);
    };
    for (int s0 = r0; s0 < r1; s0 += slab) {
        const int n = std::min(slab, r1 - s0);
        float *x = &x_[static_cast<size_t>(s0) * H];
        double t0 = now_seconds();
        if (helper && (gpu_bits_ & 1)) {   // all of it on the GPU, the intermediates never coming back
            size_t len = 0;
            const uint8_t *base = emu_buf(npu_->emu.get(), L.w.fd, &len);
            if (!base) {
                err = "NAR: the layer's weights for the GPU";
                return false;
            }
            NarGpu::UnitShape us;
            us.o = L.o, us.gu = L.gu, us.down = L.down, us.sh_o = L.sh_o, us.sh_gu = L.sh_gu, us.sh_down = L.sh_down;
            us.H = H, us.qd = qd, us.F = F, us.eps = c_.eps;
            double k = 0.0;
            if (!npu_->gpu->unit(base, len, us, L.mlp_norm, n, ob_ + static_cast<size_t>(s0) * qd, x, err, &k)) return false;
            P.gu += now_seconds() - t0, P.gpu_k += k;
            continue;
        }
        if (!prod(L.o, H, qd, L.sh_o, n, ob_ + static_cast<size_t>(s0) * qd, qd, o, H)) return false;
        double t1 = now_seconds();
        P.o += t1 - t0;
        par(n, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; ++r) {
                float *xr = x + static_cast<size_t>(r) * H;
                const float *orow = &o[static_cast<size_t>(r) * H];
                for (int i = 0; i < H; ++i) xr[i] += orow[i];
                rms_row(xr, L.mlp_norm, &h[static_cast<size_t>(r) * H], H, c_.eps);
            }
        });
        t0 = now_seconds();
        P.cpu += t0 - t1;
        if (mlp_ && mode == Tail::Hybrid) {   // gate, up, SiLU and down in one DSP call: the middle never comes back
            std::vector<const float *> xp(static_cast<size_t>(n));
            std::vector<float *> yp(static_cast<size_t>(n));
            for (int r = 0; r < n; ++r)
                xp[static_cast<size_t>(r)] = &h_[static_cast<size_t>(r) * H], yp[static_cast<size_t>(r)] = &o_[static_cast<size_t>(r) * H];
            const uint64_t up = L.gu + static_cast<uint64_t>(F) * static_cast<uint64_t>(H) * 2u;   // (gate's F rows first)
            if (const int e = npu_->mlp_f16(npu_->g, L.w.fd, L.gu, up, L.down, H, F, L.sh_gu, L.sh_gu, L.sh_down, 1, n,
                                            xp.data(), yp.data())) {
                err = "NAR: the DSP's MLP failed (" + std::to_string(e) + ")";
                return false;
            }
            t1 = now_seconds();
            prof_.gu += t1 - t0;
            pool_->parallel_for(static_cast<int64_t>(n) * H, [&](int64_t a, int64_t b) {
                for (int64_t i = a; i < b; ++i) x[i] += o_[static_cast<size_t>(i)];
            }, 4096);
            prof_.cpu += now_seconds() - t1;
            continue;
        }
        if (!prod(L.gu, 2 * F, H, L.sh_gu, n, h, H, gu, 2 * F)) return false;
        t1 = now_seconds();
        P.gu += t1 - t0;
        par(n, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; ++r) {
                const float *g = &gu[static_cast<size_t>(r) * 2 * F];
                float *ar = &act[static_cast<size_t>(r) * F];
                for (int i = 0; i < F; ++i) ar[i] = silu(g[i]) * g[F + i];
            }
        });
        t0 = now_seconds();
        P.cpu += t0 - t1;
        if (!prod(L.down, H, F, L.sh_down, n, act, F, o, H)) return false;
        t1 = now_seconds();
        P.down += t1 - t0;
        par(static_cast<int64_t>(n) * H, [&](int64_t a, int64_t b) {
            for (int64_t i = a; i < b; ++i) x[i] += o[static_cast<size_t>(i)];
        }, 4096);
        P.cpu += now_seconds() - t1;
    }
    return true;
}

// The chunk's layers on the DSP (npu/nar_cfg.h): P - the norms (q_norm times the queries' scale); IO - x, the rope
// tables, the DSP's scratch; the banks, RB
bool NarDsp::open_cfg(nar_cfg_t &cf, const std::vector<LayerW> &Ls, int pos0, std::string &err) {
    const int H = c_.hidden, hd = c_.head_dim, nkv = c_.kv_heads, dgs = hd / 32, nl = c_.layers, n = cf.n;
    const int F = c_.ffn, half = hd / 2;
    if (nl > NAR_MAX_LAYERS || static_cast<int>(Ls.size()) != nl) {
        err = "NAR: more layers than the DSP's driver takes";
        return false;
    }
    cf.version = NAR_VERSION, cf.hidden = H, cf.n_layers = nl, cf.nh = c_.heads, cf.nkv = nkv, cf.hd = hd, cf.ffn = F;
    cf.eps = c_.eps, cf.npre = npre_, cf.rem = rem_;
    cf.rb_fd = rem_ ? rb_.fd : 0;
    cf.attn_flags = static_cast<int32_t>(attn_flags_ & ~7u);   // (the helpers: the driver's pool)
    const size_t per = (2u * static_cast<size_t>(H) + 2u * static_cast<size_t>(hd)) * 4u;   // a layer's norms
    const size_t norms = per * static_cast<size_t>(nl), img = 2u * static_cast<size_t>(nkv) * dgs * 2048u;
    if (!alloc(p_, (norms + 4095u) / 4096u * 4096u, err)) return false;
    const float qscale = 1.4426950408889634f / std::sqrt(static_cast<float>(hd));   // SDPA's, and log2 e (ATTN_LOG2)
    for (int l = 0; l < nl; ++l) {
        const LayerW &L = Ls[static_cast<size_t>(l)];
        const size_t at = per * static_cast<size_t>(l);
        float *b = reinterpret_cast<float *>(p_.p + at);
        std::memcpy(b, L.in_norm, sizeof(float) * H);
        std::memcpy(b + H, L.mlp_norm, sizeof(float) * H);
        for (int i = 0; i < hd; ++i) b[2 * H + i] = L.q_norm[i] * qscale;
        std::memcpy(b + 2 * H + hd, L.k_norm, sizeof(float) * hd);
        nlayer_t &y = cf.layers[l];
        y.w_fd = L.w.fd;
        y.pk_fd = npre_ ? pre_k_[static_cast<size_t>(l)].fd : 0, y.pv_fd = npre_ ? pre_v_[static_cast<size_t>(l)].fd : 0;
        y.sh_qkv = L.sh_qkv, y.sh_o = L.sh_o, y.sh_gu = L.sh_gu, y.sh_down = L.sh_down;
        y.qkv = L.qkv, y.o = L.o, y.gate = L.gu, y.down = L.down;
        // (gate's F rows first: F / 32 blocks of H inputs, 64 bytes an input in fp16, 34 in hmx8)
        y.up = L.gu + static_cast<uint64_t>(F / 32) * static_cast<uint64_t>(H) * (L.q8 ? 34u : 64u);
        y.in_norm = static_cast<uint32_t>(at), y.mlp_norm = static_cast<uint32_t>(at + 4u * H);
        y.q_norm = static_cast<uint32_t>(at + 8u * H), y.k_norm = static_cast<uint32_t>(at + 8u * H + 4u * hd);
        y.rem = rem_ ? static_cast<uint32_t>(img * static_cast<size_t>(l)) : 0u;
        y.wfmt = L.q8 ? 1u : 0u;
    }
    cf.p_norms = static_cast<uint32_t>(norms);
    const auto up4k = [](uint64_t x) { return (x + 4095u) & ~static_cast<uint64_t>(4095u); };
    cf.io_x = 0;
    cf.io_rope = up4k(static_cast<uint64_t>(n) * H * 4u);
    cf.scratch = up4k(cf.io_rope + static_cast<uint64_t>(n) * hd * 4u);
    cf.io_bytes = nar_layout(&cf).end;
    if (!alloc(io_, cf.io_bytes, err)) return false;
    float *rope = reinterpret_cast<float *>(io_.p + cf.io_rope);   // (yue2.cpp rope_rows: the angle in fp32)
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < half; ++i) {
            const float angle = static_cast<float>(pos0 + r) * c_.inv_freq[static_cast<size_t>(i)];
            rope[static_cast<size_t>(r) * hd + i] = std::cos(angle);
            rope[static_cast<size_t>(r) * hd + half + i] = std::sin(angle);
        }
    io_bytes_ = cf.io_bytes;
    gcf_ = cf;
    gls_ = &Ls;
    groups_.clear();
    if (!lazy_) {
        groups_.emplace_back(0, nl);
    } else {   // the layers in order, a group while its weights and banks fit beside the buffers every call maps
        const bool vel = cf.mode == NAR_MODE_VELOCITY;
        const size_t fixed = p_.len + io_.len + (rem_ ? rb_.len : 0) + (vel ? tail_k_.len + tail_v_.len : 0);
        const auto unit = [&](int l) {
            size_t n = Ls[static_cast<size_t>(l)].w.len;
            if (npre_) n += pre_k_[static_cast<size_t>(l)].len + pre_v_[static_cast<size_t>(l)].len;
            return n;
        };
        size_t used = 0;
        int first = 0;
        for (int l = 0; l < nl; ++l) {
            const size_t u = unit(l);
            if (fixed + u > window_) {
                err = "NAR: the chunk is too long for the DSP's address space (" + std::to_string((fixed + u) >> 20) +
                      " MB a layer, " + std::to_string(window_ >> 20) + " MB the window)";
                return false;
            }
            if (fixed + used + u > window_) groups_.emplace_back(first, l), first = l, used = 0;
            used += u;
        }
        groups_.emplace_back(first, nl);
    }
    return open_group(0, err);
}

bool NarDsp::open_group(int g, std::string &err) {
    close_driver();
    if (!gls_ || g < 0 || g >= static_cast<int>(groups_.size())) {
        err = "NAR: no such group of layers";
        return false;
    }
    const std::vector<LayerW> &Ls = *gls_;
    const int nl = c_.layers, first = groups_[static_cast<size_t>(g)].first, end = groups_[static_cast<size_t>(g)].second;
    if (lazy_) {
        const double t0 = now_seconds();
        const bool vel = gcf_.mode == NAR_MODE_VELOCITY;
        if (!vel) unmap_buf(tail_k_), unmap_buf(tail_v_);   // (a prefix writes the prefix's banks and RB only)
        const auto out = [](const Buf &b) { return b.p && b.len && !b.mapped ? b.len : size_t{0}; };
        size_t need = out(p_) + out(io_) + (rem_ ? out(rb_) : 0) + (vel ? out(tail_k_) + out(tail_v_) : 0);
        for (int l = first; l < end; ++l) need += unit_unmapped(Ls, l);
        // others out while the window is short: the layer whose next use is furthest first (d positions after the
        // group's end: the next velocity runs them from layer 0 again)
        for (int d = nl - 1; d >= 0 && mapped_ + need > window_; --d) {
            const int l = (end + d) % nl;
            if (l < first || l >= end) unmap_unit(Ls, l);
        }
        if (mapped_ + need > window_) {
            err = "NAR: the group's layers do not fit the DSP's address space";
            return false;
        }
        if (!map_buf(p_, err) || !map_buf(io_, err) || (rem_ && !map_buf(rb_, err)) ||
            (vel && (!map_buf(tail_k_, err) || !map_buf(tail_v_, err))))
            return false;
        for (int l = first; l < end; ++l)
            if (!map_unit(Ls, l, err)) return false;
        prof_.remap += now_seconds() - t0;
    }
    nar_cfg_t cf = gcf_;
    cf.n_layers = end - first;
    for (int i = 0; i < NAR_MAX_LAYERS; ++i) {
        if (i < cf.n_layers) cf.layers[i] = gcf_.layers[first + i];
        else std::memset(&cf.layers[i], 0, sizeof cf.layers[i]);
    }
    if (const int e = npu_->nar_open(npu_->g, &cf, static_cast<int>(sizeof cf), p_.fd, io_.fd)) {
        char h[96];
        std::snprintf(h, sizeof h, "NAR: the DSP's layers would not open (0x%x)", static_cast<unsigned>(e));
        err = h;
        return false;
    }
    drv_open_ = true;
    group_ = g;
    return true;
}

bool NarDsp::run_layers(int upto, int flags, std::string &err) {
    if (const int dbg = NAR_DEBUG_LAYERS(flags)) {   // (the flags' own count: the model's layers, not a group's)
        flags &= ~0xFF;
        if (!upto || dbg < upto) upto = dbg;
    }
    const int G = static_cast<int>(groups_.size());
    for (int g = 0; g < G; ++g) {
        const int first = groups_[static_cast<size_t>(g)].first, end = groups_[static_cast<size_t>(g)].second;
        if (upto && first >= upto) break;
        if ((!drv_open_ || group_ != g) && !open_group(g, err)) return false;
        int f = flags;
        if (upto && end > upto) f |= NAR_DEBUG_LAYERS(upto - first);   // (the trace's layers: this group's first ones)
        else if (g + 1 < G && !(upto && end == upto)) f |= NAR_RUN_MORE;
        if (!run_driver(f, err)) return false;
    }
    return true;
}

bool NarDsp::open_driver(std::string &err) {
    nar_cfg_t cf;
    std::memset(&cf, 0, sizeof cf);
    cf.mode = NAR_MODE_VELOCITY, cf.A = A_, cf.n = N_;
    cf.tk_fd = tail_k_.fd, cf.tv_fd = tail_v_.fd;
    return open_cfg(cf, layers_, A_, err);
}

bool NarDsp::begin_tokens(const int32_t *ids, int A, int frames, std::string &err) {
    if (!npu_ || !drv_ || !npu_->shared_free) {
        err = "NAR: the prefix on the DSP wants its driver (libgeode_npu.so's gnpu_nar_*, gnpu_shared_free)";
        return false;
    }
    pt_ = PrefixTimes{};
    const double t0 = now_seconds();
    if (!setup_chunk(A, frames, err)) return false;
    // the memory: the AR's weights or the NAR's - both only when they are lazily mapped (never in the window
    // together) and the memory holds both (load_nar_async's check; a later chunk: the AR's read again beside them)
    if (!layers_.empty() && lazy_ && (!ar_.empty() || mem_available() >= ar_bytes() + (int64_t{3} << 30))) {
        for (int l = 0; l < c_.layers; ++l) unmap_buf(layers_[static_cast<size_t>(l)].w);   // (the prefix maps the AR's)
    } else {
        drop_layers(layers_);
    }
    if (!ensure_ar(err)) return false;   // (load_ar_async's, when the prompts had them on the DSP: nothing to read)
    const double t1 = now_seconds();
    pt_.weights = t1 - t0;
    nar_cfg_t cf;
    std::memset(&cf, 0, sizeof cf);
    cf.mode = NAR_MODE_PREFIX, cf.A = 0, cf.n = A;   // (npre / rem: A's, as the velocities')
    bool ok = open_cfg(cf, ar_, 0, err) && embed_x(ids, A, err);
    if (ok) {
        const Prof keep = prof_;
        prof_ = Prof{};
        ok = run_layers(0, prefix_flags(), err);
        pt_.dsp = prof_.dsp_call;
        prof_ = keep;
    }
    close_driver();
    drop_layers(ar_);
    ar_wanted_ = false;   // (a later chunk reads them again)
    if (!ok) return false;
    const double t2 = now_seconds();
    if (!ensure_nar(err)) return false;
    pt_.nar_weights = now_seconds() - t2;
    if (!open_driver(err)) return false;
    pt_.all = now_seconds() - t0;
    return true;
}

bool NarDsp::prefix_kv(int l, float *k, float *v) const {
    if (!N_ || l < 0 || l >= c_.layers) return false;
    const int hd = c_.head_dim, nkv = c_.kv_heads, kd = nkv * hd, dgs = hd / 32;
    const size_t blk = static_cast<size_t>(nkv) * static_cast<size_t>(dgs) * 2048u;
    for (int p = 0; p < A_; ++p) {
        const bool whole = p < npre_ * 32;
        const _Float16 *K, *V;
        size_t nb;
        if (whole) {
            K = reinterpret_cast<const _Float16 *>(pre_k_[static_cast<size_t>(l)].p);
            V = reinterpret_cast<const _Float16 *>(pre_v_[static_cast<size_t>(l)].p);
            nb = static_cast<size_t>(npre_);
        } else {
            K = reinterpret_cast<const _Float16 *>(rb_.p + 2u * blk * static_cast<size_t>(l));
            V = K + blk / 2u;
            nb = 1;
        }
        const int pp = whole ? p : p - npre_ * 32;
        for (int kh = 0; kh < nkv; ++kh)
            for (int c = 0; c < hd; ++c) {
                k[static_cast<size_t>(p) * kd + static_cast<size_t>(kh) * hd + c] = static_cast<float>(K[k_at(nb, dgs, kh, pp, c)]);
                v[static_cast<size_t>(p) * kd + static_cast<size_t>(kh) * hd + c] = static_cast<float>(V[v_at(nb, dgs, kh, pp, c)]);
            }
    }
    return true;
}

// IO's x through the layers on the DSP (flags NAR_DEBUG_*), the DSP's times into the profile
bool NarDsp::run_driver(int flags, std::string &err) {
    const uint64_t *pr = reinterpret_cast<const uint64_t *>(io_.p + io_bytes_ - 256u);
    uint64_t was[NAR_PROF_N];
    std::memcpy(was, pr, sizeof was);
    double ds = 0.0;
    if (const int e = npu_->nar_run(npu_->g, flags, &ds)) {
        char h[96];
        std::snprintf(h, sizeof h, "NAR: the DSP's layers failed (0x%x)", static_cast<unsigned>(e));
        err = h;
        return false;
    }
    const auto d = [&](int i) { return static_cast<double>(pr[i] - was[i]) * 1e-6; };
    prof_.qkv += d(NAR_PROF_QKV), prof_.attn += d(NAR_PROF_ATTN), prof_.attn_dsp += d(NAR_PROF_ATTN);
    prof_.o += d(NAR_PROF_O), prof_.gu += d(NAR_PROF_MLP);
    prof_.gemm_dsp += d(NAR_PROF_QKV) + d(NAR_PROF_O) + d(NAR_PROF_MLP);
    prof_.hvx += d(NAR_PROF_IN) + d(NAR_PROF_KV) + d(NAR_PROF_CORE) + d(NAR_PROF_POST) + d(NAR_PROF_DOWN) + d(NAR_PROF_CACHE);
    prof_.dsp_call += ds;
    if (env_int("GEODE_YUE2_NAR_PROF", 0))
        std::fprintf(stderr, "  DSP: in %.3f qkv %.3f kv %.3f attn %.3f core %.3f o %.3f post %.3f mlp %.3f down %.3f "
                             "cache %.3f s\n", d(NAR_PROF_IN), d(NAR_PROF_QKV), d(NAR_PROF_KV), d(NAR_PROF_ATTN),
                     d(NAR_PROF_CORE), d(NAR_PROF_O), d(NAR_PROF_POST), d(NAR_PROF_MLP), d(NAR_PROF_DOWN),
                     d(NAR_PROF_CACHE));
    if (env_int("GEODE_YUE2_NAR_PROF", 0))
        std::fprintf(stderr, "  products: glu %.3f of mlp | activations in %.3f, waiting for weights %.3f, HMX tiles %.3f, "
                             "results out %.3f, the GLU waiting %.3f s\n", d(NAR_PROF_GLU), d(NAR_PROF_G_ACOPY),
                     d(NAR_PROF_G_WAIT), d(NAR_PROF_G_TILE), d(NAR_PROF_G_OUT), d(NAR_PROF_G_GLUWAIT));
    return true;
}

bool NarDsp::velocity(const float *state, double raw_t, float *v, std::string &err, Trace *trace) {
    if (!N_) {
        err = "NAR: begin() first";
        return false;
    }
    const double t0 = now_seconds();
    const double dsp0 = npu_->dsp_s ? npu_->dsp_s(npu_->g) : 0.0, conv0 = npu_->conv_s ? npu_->conv_s(npu_->g) : 0.0;
    const int H = c_.hidden, Lt = c_.latent, N = N_;
    float *X = drv_ ? reinterpret_cast<float *>(io_.p) : x_.data();   // the rows (the driver's: IO's x)
    // yue2.cpp Model::velocity's input rows: vae2llm of the padded state, the time embedding, the positions
    std::vector<float> xn(static_cast<size_t>(N) * Lt, 0.0f);
    std::memcpy(&xn[static_cast<size_t>(Lt)], state, sizeof(float) * static_cast<size_t>(frames_) * Lt);
    gemm_bf16(xn.data(), N, Lt, vae2llm_w_.data(), H, vae2llm_b_.data(), X, *pool_);
    const float ts = 1.0f / (1.0f + std::exp(-static_cast<float>(raw_t)));
    const float shift = static_cast<float>(c_.shift), tt = shift * ts / (1.0f + (shift - 1.0f) * ts);
    const int half = c_.freq / 2;
    std::vector<float> emb(static_cast<size_t>(c_.freq)), e0(static_cast<size_t>(H)), temb(static_cast<size_t>(H));
    for (int i = 0; i < half; ++i) {
        const float fr = std::exp((static_cast<float>(-std::log(10000.0)) * static_cast<float>(i)) / static_cast<float>(half));
        const float arg = tt * fr;
        emb[static_cast<size_t>(i)] = std::cos(arg);
        emb[static_cast<size_t>(half + i)] = std::sin(arg);
    }
    gemm_bf16(emb.data(), 1, c_.freq, t0_w_.data(), H, t0_b_.data(), e0.data(), *pool_);
    for (auto &e : e0) e = silu(e);
    gemm_bf16(e0.data(), 1, H, t2_w_.data(), H, t2_b_.data(), temb.data(), *pool_);
    for (int r = 0; r < N; ++r) {
        const uint16_t *pe = &pe_[static_cast<size_t>(std::min(r, c_.max_latent_frames - 1)) * H];
        float *xr = X + static_cast<size_t>(r) * H;
        for (int i = 0; i < H; ++i) xr[i] = (xr[i] + temb[static_cast<size_t>(i)]) + bf16f(pe[i]);
    }
    const size_t xn_all = static_cast<size_t>(N) * H;
    if (trace) {
        trace->t_shifted = tt;
        trace->x_in.assign(X, X + xn_all);
    }
    const double t_in = now_seconds() - t0, call0 = prof_.dsp_call;
    prof_.cpu += t_in;
    if (drv_) {
        if (trace)   // the layers asked for: the first l + 1 of them from the same rows (NAR_DEBUG_LAYERS)
            for (auto &kv : trace->x_after) {
                std::memcpy(X, trace->x_in.data(), sizeof(float) * xn_all);
                if (!run_layers(kv.first + 1, 0, err)) return false;
                kv.second.assign(X, X + xn_all);
            }
        if (trace) std::memcpy(X, trace->x_in.data(), sizeof(float) * xn_all);
        prof_.groups = static_cast<int>(groups_.size());
        if (!run_layers(0, env_int("GEODE_YUE2_NAR_FLAGS", 0), err)) return false;   // (experiments: nar_cfg.h NAR_X_*)
    } else {
        for (int l = 0; l < c_.layers; ++l) {
            if (!layer(l, err)) return false;
            if (trace && trace->x_after.count(l)) trace->x_after[l].assign(X, X + xn_all);
        }
    }
    const double t1 = now_seconds();
    std::vector<float> hn(static_cast<size_t>(frames_) * H);
    pool_->parallel_for(frames_, [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; ++r) rms_row(X + static_cast<size_t>(r + 1) * H, norm_.data(), &hn[static_cast<size_t>(r) * H], H, c_.eps);
    });
    gemm_bf16(hn.data(), frames_, H, llm2vae_w_.data(), Lt, llm2vae_b_.data(), v, *pool_);
    prof_.cpu += now_seconds() - t1;
    if (env_int("GEODE_YUE2_NAR_PROF", 0))
        std::fprintf(stderr, "  host: rows in %.3f, the layers %.3f (the DSP's own %.3f), out %.3f s\n", t_in,
                     t1 - t0 - t_in, prof_.dsp_call - call0, now_seconds() - t1);
    if (npu_->dsp_s) prof_.gemm_dsp += npu_->dsp_s(npu_->g) - dsp0;
    if (npu_->conv_s) prof_.convert += npu_->conv_s(npu_->g) - conv0;
    ++prof_.evals;
    return true;
}

bool NarDsp::solve(const float *noise, int steps, float *out, std::string &err, const std::function<void(int, int)> &progress,
                    const float *known, const uint8_t *held) {
    const size_t n = static_cast<size_t>(frames_) * c_.latent;
    if (steps < 1 || !N_) {
        err = "NAR solve: steps must be >= 1 (and begin first)";
        return false;
    }
    std::vector<float> state(noise, noise + n), first(n), mid(n), second(n);
    const double dt = 1.0 / steps;
    const auto raw = [](double t) {   // torch.logit in float64, clamped to +-20
        const double r = std::log(t / (1.0 - t));
        return std::isnan(r) ? r : std::min(20.0, std::max(-20.0, r));
    };
    const float hf = static_cast<float>(dt / 2), full = static_cast<float>(dt);
    // each evaluation checked as it comes: one that is not finite computed again once (the RedMagic gave one such song
    // in five with the 8-bit weights made in the same run, its cause not found - every rerun of the same request was
    // fine), then the solve fails naming it
    const auto eval = [&](const float *x, double rt, float *v, int e) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!velocity(x, rt, v, err)) return false;
            bool finite = true;
            for (size_t i = 0; i < n && finite; ++i) finite = std::isfinite(v[i]);
            if (finite) return true;
            std::fprintf(stderr, "NAR: evaluation %d not finite%s\n", e, attempt ? "" : " - once more");
        }
        err = "NAR solve: evaluation " + std::to_string(e) + " not finite (twice)";
        return false;
    };
    const size_t L = static_cast<size_t>(c_.latent);
    const bool holds = known && held;
    // the held frames on their line at time tt: tt * noise + (1 - tt) * known
    const auto hold = [&](std::vector<float> &x, double tt) {
        const float a = static_cast<float>(tt), b = static_cast<float>(1.0 - tt);
        for (int f = 0; f < frames_; ++f) {
            if (!held[f]) continue;
            const size_t o = static_cast<size_t>(f) * L;
            for (size_t i = o; i < o + L; ++i) x[i] = a * noise[i] + b * known[i];
        }
    };
    for (int step = 0; step < steps; ++step) {
        const double t = 1.0 - step * dt;
        if (cancelled && cancelled()) {
            err = "cancelled";
            return false;
        }
        if (on_state) on_state(2 * step, state.data());
        if (!eval(state.data(), raw(t), first.data(), 2 * step)) return false;
        for (size_t i = 0; i < n; ++i) mid[i] = state[i] - first[i] * hf;
        if (holds) hold(mid, t - dt / 2);
        if (on_state) on_state(2 * step + 1, mid.data());
        if (!eval(mid.data(), raw(t - dt / 2), second.data(), 2 * step + 1)) return false;
        for (size_t i = 0; i < n; ++i) state[i] = state[i] - second[i] * full;
        if (holds) hold(state, t - dt);
        if (progress) progress(step + 1, steps);
    }
    if (holds) hold(state, 0.0);   // (exactly the known latents: 0 * noise + 1 * known)
    for (const float s : state)
        if (!std::isfinite(s)) {
            err = "NAR solve: non-finite latents";
            return false;
        }
    std::memcpy(out, state.data(), sizeof(float) * n);
    return true;
}

}  // namespace geode::yue2
