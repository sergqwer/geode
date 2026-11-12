// yue2_nar_gpu.hpp - the NAR's products on the phone's GPU (docs/music-plan.md M6), for the phones whose NPU has no
// matrix unit: nar_f16.bin's matrices as they are (HMX's fp16 blocks) read through the texture path, the activations
// in fp32 (packed for the kernel by a transpose on the GPU), every product summed in fp32 - the CPU backend's
// results to ~1e-6 (bench/gpu/nar_gemm_f16_img.cl, measured on the Adreno 618: 33.6 G MAC/s against the CPU's 23.6).
// A layer's weights are copied into the GPU's memory the first time a product of it is asked for, into one of the
// slots (the least recently used one given up; every layer when the memory has room - yue2_nar.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "threads.hpp"

namespace geode::yue2 {

class NarGpu {
public:
    NarGpu();
    ~NarGpu();
    NarGpu(const NarGpu &) = delete;
    NarGpu &operator=(const NarGpu &) = delete;

    // The GPU, the kernel built (cache_dir: the driver's binary kept there, "" none); slots: the layers held at once;
    // layer_bytes: the largest layer's stretch (an image over it must fit the device's image buffers).
    bool open(int slots, size_t layer_bytes, const std::string &cache_dir, std::string &err);
    const std::string &name() const;

    // y[t] = W x[t] for t < n: W [rows][cols] at byte `off` of the layer stretch `layer` (len bytes, HMX's fp16
    // blocks, the values 2^shift larger); x[t] cols floats, y[t] rows floats.
    bool gemm(const uint8_t *layer, size_t len, uint64_t off, int64_t rows, int64_t cols, int shift, int n,
              const float *const *xs, float *const *ys, ThreadPool &pool, std::string &err);
    // gemm in two halves, for the CPU to compute other rows meanwhile: start() copies the activations in and queues
    // the product (it returns as the GPU starts); finish() waits for it and writes y[t][0..rows). gpu_s: the
    // kernels' own seconds (their events).
    bool start(const uint8_t *layer, size_t len, uint64_t off, int64_t rows, int64_t cols, int shift, int n,
               const float *const *xs, ThreadPool *pool, std::string &err);
    bool finish(float *const *ys, ThreadPool *pool, std::string &err, double *gpu_s = nullptr);
    // the layer copied into a slot now, ahead of its products (the least recently used slot given up); false and
    // nothing done with one slot only (it would push out the layer in use) or when the copy fails
    bool prefetch(const uint8_t *layer, size_t len, ThreadPool *pool, std::string &err);

    // The NAR layer's part after the attention for n tokens, all of it on the GPU (yue2_nar.cpp post_attn): x [n][H]
    // (in and out, rows in order) and a [n][qd] (the attention's output): x += Wo a; h = x / rms(x) * norm;
    // x += Wd (SiLU(Wg h) * Wu h) - the products as gemm()'s (fp32 sums), the glue in fp32 (exp, not native_exp).
    // The intermediates never leave the GPU: a and x copied in, x out. gpu_s: the queue's span (its events).
    struct UnitShape {
        uint64_t o = 0, gu = 0, down = 0;   // the matrices' offsets in the layer stretch (gu: gate's F rows, then up's)
        int sh_o = 0, sh_gu = 0, sh_down = 0;
        int H = 0, qd = 0, F = 0;
        float eps = 0.0f;
    };
    // wait false: everything queued (a, x and norm must stay put, x is written when unit_wait() has it) and in flight
    // on return - a second unit may be queued behind it (the queue is in order: they share the GPU's buffers)
    bool unit(const uint8_t *layer, size_t len, const UnitShape &u, const float *norm, int n, const float *a, float *x,
              std::string &err, double *gpu_s = nullptr, bool wait = true);
    // the oldest unit left in flight: waits for its rows (false: it failed; the queue drained, every unit dropped)
    bool unit_wait(std::string &err, double *gpu_s = nullptr);
    int units_in_flight() const;

    struct Times {   // seconds
        double upload = 0, pack = 0, kernel = 0, unpack = 0;
        int64_t uploads = 0;
    };
    const Times &times() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace geode::yue2
