// hvx_llm.hpp - an LLM prompt's 4-bit products on a Hexagon v66 class DSP's HVX beside the CPU (npu/hvx66,
// geode_hvx_q4_gemm): Snapdragon 7xx / 6xx DSPs without HMX (the Redmi Note 9 Pro's 720G), integer HVX only.
//
// The matrices (q4i, q4) live once, in the CPU's quad layout (kernels.hpp), in rpcmem the DSP reads
// (Model::hvx_one_copy, their pack pages given back); the DSP turns each tile's codes into tile8's order as it goes
// (tile8 itself measured 15-20% faster there, but the CPU's answer 4-10% slower on it: GEODE_HVX_TILE8=1). A product
// of kMinTokens tokens and more is split by rows: the DSP the first r rows (whole tiles of 32), the CPU the rest at the
// same time; r from the measured times of both, per matrix shape, so the two finish together.
// One product: prepare (the DSP's inputs - the tokens' int8 values permuted, the blocks' sums and scale ratios, the
// rows' ratios - on the caller's pool, the big cores), start (the call, from a helper thread pinned to a big core: it
// mostly waits in FastRPC; from an A55 a call cost 7.5-8 ms more than the DSP's time, from an A76 2.7-3.9 - the
// driver's work on the calling core), the CPU's rows meanwhile, finish (waits), results (the DSP's int32 sums into
// floats, on the pool: on the A55 10-20x slower). The per-call buffers are cached rpcmem (uncached made the CPU's
// reads of the results 10x slower).
// The arithmetic: the CPU's int8 path (per 16-value block the exact integer dot, times the block's weight and
// activation scales), the two scales as Q15 / Q13 ratios in fixed point (geode_hvx.idl) - rel L2 4-7e-4 against the
// CPU's floats (hvx_probe q4; the engine's GEODE_HVX_CHECK on a real prompt: mean 4.3e-4, worst 1.1e-3), an eighth of
// the int8 activations' own rounding. A sum that came near int32's limit (|sum| >= 2^30 in any pass: it may have
// saturated) has its rows computed by the CPU instead.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kernels.hpp"

namespace geode {

class HvxGemm {
public:
    static constexpr int kMinTokens = 64;      // fewer: the call and the conversions cost more than they save
    static constexpr int64_t kMinRows = 512;   // a product this small stays the CPU's (k, v, the PLE gate)
    static constexpr int64_t kMinCols = 512;   // (the PLE projection's 256 inputs)
    // the caller's parallel loop over [0, n) (the pool's big threads)
    using Par = std::function<void(int64_t, const std::function<void(int64_t, int64_t)> &)>;

    // libgeode_npu.so (GEODE_NPU_LIB, else from the library path or beside the executable) and its v66 module
    static std::unique_ptr<HvxGemm> open(std::string &why);
    ~HvxGemm();
    HvxGemm(const HvxGemm &) = delete;
    HvxGemm &operator=(const HvxGemm &) = delete;

    // rpcmem for the weights (cached: the CPU reads them too), kept mapped on the DSP (gnpu_hvx_keep); nullptr when
    // there is none. Freed by the destructor.
    uint8_t *weights(size_t bytes);
    // a matrix now in tile8 in weights' memory: the DSP may take its rows (its rows' largest scales noted)
    void hold(const QView &m);
    bool holds(const QView &m) const { return mats_.count(m.data) != 0; }
    size_t held_bytes() const { return weight_bytes_; }

    // the DSP's rows of a product of n tokens (a multiple of 32; 0: the CPU's whole)
    int64_t rows_for(const QView &m, int n) const;
    // The DSP's inputs for rows [0, r) of m and the n tokens' activations (acts[t] the CPU's quantization of token t);
    // same_x: the activations are the last product's (their part kept). False: no memory for them.
    bool prepare(const QView &m, int64_t r, int n, const QAct *const *acts, bool same_x, const Par &par);
    // the call on the helper thread; finish waits for it - cpu_s: the CPU's time for its rows meanwhile (the shares
    // learn from both). kDone: ys's rows to come from results(); kRedo: a sum may have saturated - the caller computes
    // rows [0, r) itself; kFailed: the DSP failed (err; the caller computes them, and the DSP is off from now on).
    enum Finish { kDone, kRedo, kFailed };
    void start();
    Finish finish(double cpu_s, std::string &err);
    // the activations prepare() keeps are no longer the next product's (matmul_many's group begins)
    void forget_x() { x_k_ = 0; }
    // its sums into ys[t][0, r)
    void results(float *const *ys, const Par &par);
    bool off() const { return off_; }
    std::string status() const;

private:
    HvxGemm() = default;
    struct Mat {
        std::vector<float> wmax;   // per row the largest |block scale|
        size_t tile = 0;           // a tile's bytes (18 k)
        bool quad = false;         // the quad layout (else tile8)
        // the tokens' scale ratios' bits (Q13; a product that came near int32's limit: 2 fewer for this matrix from
        // then on - 4x the headroom, the ratios still within 1/2048 of the largest)
        int aq_bits = 13;
    };
    // a scale byte: row l (0..31) of the tile at t, block b
    static uint8_t scale_at(const Mat &M, const uint8_t *t, int64_t k, int64_t l, int64_t b) {
        return M.quad ? t[static_cast<size_t>(l / 4) * (static_cast<size_t>(k) * 9 / 4) + static_cast<size_t>(k) * 2 +
                          static_cast<size_t>(b) * 4 + static_cast<size_t>(l % 4)]
                      : t[static_cast<size_t>(k) * 16 + static_cast<size_t>((b / 4) * 8 + l / 4) * 16 +
                          static_cast<size_t>((b % 4) * 4 + l % 4)];
    }
    struct Buf {
        void *p = nullptr;
        size_t bytes = 0;
    };
    void helper();
    void *grow(Buf &b, size_t bytes);

    void *lib_ = nullptr, *h_ = nullptr;
    void (*close_)(void *) = nullptr;
    void *(*alloc_)(size_t, int) = nullptr;
    void (*free_)(void *) = nullptr;
    int (*keep_)(void *, size_t, int) = nullptr;
    int (*gemm_)(void *, const uint8_t *, size_t, int, int, int, const int8_t *, const int32_t *, const int32_t *,
                 int32_t *, int, int, int, int *, long long *) = nullptr;
    std::vector<Buf> wbufs_;
    size_t weight_bytes_ = 0;
    std::map<const uint8_t *, Mat> mats_;
    Buf x_, xs_, wq_, y_;
    bool io_cached_ = true;   // GEODE_HVX_IO_CACHED=0: the per-call buffers uncached (measuring)
    int threads_ = 2;
    bool off_ = false;
    std::string note_;

    // the share of each shape's rows (rows, cols) the DSP takes, learned; GEODE_HVX_SHARE fixes it
    std::map<std::pair<int64_t, int64_t>, float> share_;
    float fixed_share_ = -1.0f;

    // the product in flight
    const QView *m_ = nullptr;
    const Mat *M_ = nullptr;
    int64_t r_ = 0;
    int n_ = 0, kc_ = 0;
    std::vector<float> smax_;            // the tokens' largest block scales
    int x_k_ = 0, x_n8_ = 0, x_kc_ = 0, x_aq_ = 0;   // what x_ / xs_ hold
    DType x_type_ = DType::Unknown;

    // the helper (the call)
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool has_job_ = false, done_ = false, quit_ = false;
    int err_ = 0, amax_ = 0;
    long long us_ = 0;
    double t_call_ = 0.0;

    // totals (status)
    uint64_t calls_ = 0, redone_ = 0;
    Mat *Mw_ = nullptr;   // (the product in flight's matrix, its aq_bits lowered by finish)
    double dsp_s_ = 0.0, call_s_ = 0.0, prep_s_ = 0.0, out_s_ = 0.0, macs_ = 0.0, cpu_macs_ = 0.0, cpu_s_ = 0.0;
    double t_prep0_ = 0.0;
};

}  // namespace geode
