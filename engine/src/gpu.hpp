// gpu.hpp - the dense products of a prompt on the phone's GPU (PLAN.md, P5b). Hand-written a6xx machine code
// (bench/gpu/asm, assembled into gpu_gemm.inc) put into a template program the driver compiles on the phone, the
// weights in the GPU layout (kernels.hpp, to_quad) in buffers the CPU kernels read too, so the GPU's rows and the
// CPU's rows can be split differently on every call (a prompt: both; one token: the CPU alone).
//
// Only where the code was written for: the Adreno 618 (a6xx gen 1), q2 matrices with 2048 columns, q4 with 2048 or
// 4096.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "kernels.hpp"

namespace geode {

class Gpu {
public:
    Gpu();
    ~Gpu();
    Gpu(const Gpu &) = delete;
    Gpu &operator=(const Gpu &) = delete;

    // The device, the template program and the kernels; false and why where any of it is missing.
    bool open(std::string &err);
    bool ready() const;
    static bool fits(const QView &m);   // a kind of matrix the kernels know, rows % 16 == 0, not adopted yet

    // Copies m into a GPU buffer in the GPU layout and points m at it (m.quad set): from then on the CPU kernels read
    // that buffer too, and the old bytes can be given back. False (m unchanged) on failure.
    bool adopt(QView &m, std::string &err);
    bool holds(const QView &m) const;   // adopted (not just in the GPU layout: the CPU reads others in it too)
    size_t adopted_bytes() const;

    // Rows [0, rows) of an adopted matrix against n tokens (xs[t]: m.cols floats), started on the GPU (the call
    // returns while it runs); rows a multiple of 16. finish() waits and writes ys[t][0, rows): blocking in the driver,
    // or with spin polling the kernel's status (a big core that sleeps there is clocked down: generating); waited the
    // seconds it waited for the kernel, late whether the kernel was still running when it was called.
    bool start(const QView &m, int64_t rows, int n, const float *const *xs, std::string &err);
    bool finish(float *const *ys, std::string &err, bool spin = false, double *waited = nullptr, bool *late = nullptr);

    double busy_seconds() const;   // the kernels' time, summed (the queue profiles)
    std::string timing() const;    // that and the host's side of the calls

    // the host's side of a call (activations in, results out) split over tokens on these threads (the model's pool)
    using ParallelFor = std::function<void(int64_t, const std::function<void(int64_t, int64_t)> &)>;
    void set_parallel(ParallelFor par);
    int64_t launches() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace geode
