// yue2_vae.hpp - YuE2-Vae's decoder (Oobleck with SnakeBeta, stable-audio-tools; yue2/modeling_vae.py): latents
// [frames][64] at 25 Hz -> 48 kHz stereo, fp32 on the CPU. The reference engine of phase 2: every operation as the
// Python reference runs it, the pack's folded weights (tools/pack_yue2.py, vae.bin).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "threads.hpp"
#include "yue2_pack.hpp"

namespace geode::yue2 {

class Vae {
public:
    Vae();
    ~Vae();
    Vae(const Vae &) = delete;
    Vae &operator=(const Vae &) = delete;

    // The decoder's geometry from config.vae and its tensors from vae.bin (loaded here).
    bool open(Pack &pack, std::string &err, bool map = false);   // map: vae.bin mapped (Pack::map), not copied

    // The decoder on the GPU from now on (OpenCL, fp32: the sums in the CPU's order, as mad - Adreno's fma is
    // emulated): its weights copied there. false and why where there is none;
    // a tile the GPU cannot take (its memory) goes back to the CPU, which keeps it from then on. Only where it wins
    // (an Adreno 7xx or newer: the 830 3.2x the CPU, the 618 0.67x) unless `any`; GEODE_YUE2_VAE_GPU=1 any GPU,
    // =0 none.
    bool open_gpu(std::string &err, bool any = false);
    bool on_gpu() const;
    std::string gpu_name() const;

    int latent_dim() const { return latent_dim_; }
    int ratio() const { return ratio_; }                 // 1920 samples a frame
    int halo() const { return halo_; }                   // the frames a frame's sound reaches either way (16)
    // The full decoder's output length for `frames` latent frames (yue2's natural_output_length: 1920 T - 64).
    int64_t natural_length(int64_t frames) const;

    // The whole decoder over latents [frames][latent_dim] (row-major, a frame's 64 values together), no tiling, no
    // clamp: out [channels = 2][natural_length(frames)] (channel-major). `layer` (optional) sees every layer's
    // output in the reference's module order: name "top<i>" for decoder.layers.<i>, "top<i>_sub<j>" for
    // decoder.layers.<i>.layers.<j> (the blocks' parts), with its [channels][length] data.
    using LayerHook = std::function<void(const std::string &name, int channels, int64_t length, const float *data)>;
    void decode(const float *latents, int64_t frames, std::vector<float> &out, ThreadPool &pool,
                const LayerHook &layer = nullptr) const;

    // yue2's decode_tiled with the release's cores and halo (config.vae decode_core_frames / decode_halo_frames),
    // then the pipeline's clamp to [-1, 1]: audio [natural_length(frames)][2] (interleaved, as the pipeline's
    // audio[0].T). progress(done, total) after each tile.
    // cancelled (optional) is asked before every tile: true stops it and the result is false (audio incomplete)
    bool decode_tiled(const float *latents, int64_t frames, std::vector<float> &audio, ThreadPool &pool,
                      const std::function<void(int, int)> &progress = nullptr,
                      const std::function<bool()> &cancelled = nullptr) const;

private:
    struct Gpu;
    mutable std::unique_ptr<Gpu> gpu_;   // (given up by decode when a tile fails there)
    mutable std::mutex gpu_mu_;          // one decode on the GPU at a time (its buffers and kernel arguments)
    // decode()'s work on the GPU (no hook); false: not done (out untouched), the CPU's then
    bool decode_gpu(const float *latents, int64_t frames, std::vector<float> &out) const;
    struct Conv {                 // Conv1d [out][in][k] or ConvTranspose1d [in][out][k], weight-norm folded
        const float *w = nullptr, *b = nullptr;
        int in = 0, out = 0, k = 0, stride = 1, dilation = 1, padding = 0;
        bool transposed = false;
    };
    struct Snake {                // SnakeBeta, alpha / beta as stored (log scale): exp() at open
        std::vector<float> alpha, inv_beta;   // exp(alpha), 1 / (exp(beta) + 1e-9)
    };
    struct Residual { Snake s1, s2; Conv c7, c1; };
    struct Block { Snake snake; Conv up; Residual res[3]; };
    int latent_dim_ = 64, ratio_ = 1920, out_channels_ = 2, core_ = 1024, halo_ = 16;
    Conv conv_in_, conv_out_;
    std::vector<Block> blocks_;
    Snake snake_out_;
};

}  // namespace geode::yue2
