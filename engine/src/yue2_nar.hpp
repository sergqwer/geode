// yue2_nar.hpp - YuE2-3B's NAR (flow matching's velocity, 64 evaluations a song) on the phone's NPU (phase 4): the
// products on HMX in fp16 (nar_f16.bin, tools/pack_yue2.py --nar-f16: the matrices already in HMX's blocks), the
// attention on HMX over the AR prefix's keys and the rows' own (npu/attn_cfg.h ATTN_ENCODER with nk / npre: the
// prefix's blocks written once a chunk, the rows' rewritten every layer into one buffer all layers share), the norms,
// RoPE, SwiGLU and the residual adds on the CPU. The arithmetic is yue2.cpp's oracle (Model::velocity, layer, solve)
// with the products and the attention in fp16.
// Without libgeode_npu.so (the PC) the same calls run on an emulation of the DSP's: the same blocks and banks read
// back, the products and the attention in fp32 from fp16 inputs - the layouts checked against the oracle off the phone.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../../npu/nar_cfg.h"
#include "threads.hpp"
#include "yue2.hpp"
#include "yue2_pack.hpp"

namespace geode::yue2 {

struct NpuCalls;   // libgeode_npu.so's functions, or the emulation's (yue2_nar.cpp)

class NarDsp {
public:
    NarDsp();
    ~NarDsp();
    NarDsp(const NarDsp &) = delete;
    NarDsp &operator=(const NarDsp &) = delete;

    // npu_dir: libgeode_npu.so and its skels (the phone); "cpu": the CPU backend (a phone without HMX: the products
    // straight from the fp16 blocks, the weights mapped); "gpu": the CPU backend with its products on the GPU
    // (yue2_nar_gpu.hpp); empty: the emulation (the DSP's roundings, for parity). The CPU's parts (vae2llm, llm2vae,
    // the time embedder, the positions) into memory; nar_f16.bin's layers (2.8 GB the DSP maps) at the first begin.
    // The pack stays the caller's and open while this is.
    bool open(Pack &pack, const std::string &npu_dir, ThreadPool &pool, std::string &err);
    const Config &config() const { return c_; }
    const std::string &backend() const { return backend_; }   // "dsp", "gpu", "cpu" or "emulation"
    std::string device() const;                                  // the GPU backend's GPU ("" another backend)
    bool driver() const { return drv_; }                         // the DSP runs whole velocities (npu/nar_cfg.h)
    // the NAR's 8-bit weights (yue2_q4.hpp make_nar_q8: h8.*) where the pack has them and the DSP runs the layers -
    // set before the layers load (begin / begin_tokens); weights_q8: what they loaded as
    // On the CPU and GPU backends the same files give the CPU's products an 8-bit twin of every matrix (SDOT:
    // yue2_q8gemm.hpp) - the GPU keeps the fp16 ones.
    void set_weights_q8(bool on) { want_q8_ = on; }
    bool weights_q8() const { return !layers_.empty() && (layers_[0].q8 || layers_[0].q8cpu); }
    // The CPU backend's attention scores in int8 (SDOT; the keys less their head's mean, a scale a key and a query
    // row): the "8-bit sound" of a phone without the NPU's HMX
    void set_attn8(bool on);
    // The "Fast" sound's 8-bit products also on a Hexagon v66 .. v72 DSP (no HMX; npu/hvx66 - libgeode_npu.so's
    // gnpu_hvx_* from dir, libgeode_hvx_skel_v66.so beside it): the GPU backend's pipeline gives it units while the
    // attention runs. false with why when it is out of reach (GEODE_YUE2_NAR_HVX=0: never) - the products stay the
    // CPU's and the GPU's. After open; the layers' 8-bit twins (set_weights_q8) are what it multiplies.
    bool set_hvx(const std::string &dir, std::string &why);
    bool hvx() const;
    // the parity tool's A/B: the DSP left out (on false) while it stays open; nice: its helper thread's (0: as is);
    // ahead: the next layer's twins read ahead from storage
    void set_hvx_mode(bool on, int nice, bool ahead) { hvx_paused_ = !on, hvx_nice_ = nice, hvx_ahead_ = ahead; }
    const std::string &hvx_note() const { return hvx_note_; }   // why it stopped mid-way ("" while it runs)

    // A song chunk's NAR (nar.py song_chunks): A AR positions (the prefix + this chunk's codes + MUSIC_END, from 0)
    // and `frames` latent frames. prefix(l, k, v): layer l's A keys and values [A][kv_heads * head_dim] floats (after
    // k_norm and RoPE, as the AR path caches them). A + frames + 2 within the context.
    using PrefixFn = std::function<bool(int layer, float *k, float *v)>;
    bool begin(int A, int frames, const PrefixFn &prefix, std::string &err);
    // The same chunk from its A AR tokens, the prefix's keys and values computed on the DSP: the AR's layers in fp16
    // (ar_f16.bin, tools/pack_yue2.py --ar-f16; npu/nar_cfg.h NAR_MODE_PREFIX: causal, every layer's keys and values
    // written straight into the NAR's banks) - its weights mapped only while it runs, then the NAR's (the DSP's
    // address space holds one of the two). Wants the driver (libgeode_npu.so with gnpu_nar_*, gnpu_shared_free).
    static bool has_ar_f16(const Pack &pack);
    bool begin_tokens(const int32_t *ids, int A, int frames, std::string &err);
    int tokens() const { return N_; }

    // The AR's generation prompts on the DSP as well (FastAr's prefill: ~120 tokens/s on the CPU; here ~2000):
    // load_ar_async reads ar_f16.bin into the DSP's memory on a thread of its own now (the CPU's generation beside
    // it), kept for the chunks' prefixes (begin_tokens then reads nothing). false - nothing started: no driver, no
    // ar_f16, or less memory free than its 2.6 GB and 3 GB more (GEODE_YUE2_DSP_PROMPT=0: never).
    bool load_ar_async();
    bool ar_loading() const { return ar_wanted_; }
    // The NAR's layers read into the DSP's memory on a thread of their own now, unmapped (the lazy groups), while the
    // CPU generates the codes - begin_tokens then finds them and reads nothing after the prefix. Only with the lazy
    // mappings (the AR's and the NAR's weights never in the window together), and with their 2.8 GB and 3 GB more free
    // (GEODE_YUE2_NAR_EARLY=0: never). false - nothing started (begin_tokens reads them as before).
    bool load_nar_async();
    // n tokens at positions 0 .. n - 1 through the AR's layers on the DSP (NAR_MODE_PREFIX, causal), their keys and
    // values a layer at a time through out(l, k, v): [n][kv_heads * head_dim] floats after k_norm and RoPE, as FastAr
    // caches them. Waits for load_ar_async's reading (or reads ar_f16 itself). seconds: the DSP's call.
    bool ar_prompt(const int32_t *ids, int n, const PrefixFn &out, std::string &err, double *seconds = nullptr);
    // the prefix's layer l as the banks hold it (fp16): its A keys and values [A][kv_heads * head_dim] floats
    bool prefix_kv(int l, float *k, float *v) const;
    struct PrefixTimes {   // begin_tokens': seconds
        double weights = 0, dsp = 0, nar_weights = 0, all = 0;
    };
    const PrefixTimes &prefix_times() const { return pt_; }

    struct Trace {   // velocity's insides, for the parity tool
        float t_shifted = 0.0f;
        std::vector<float> x_in;
        std::map<int, std::vector<float>> x_after;   // the layers asked for (keys preset)
    };
    // CachedNAR.velocity: state [frames][latent] at raw_t = logit(t) clamped -> v [frames][latent]
    bool velocity(const float *state, double raw_t, float *v, std::string &err, Trace *trace = nullptr);
    // CachedNAR.solve: `steps` midpoint steps from t = 1 to 0, from noise [frames][latent] -> out. An edit's
    // (YuE2-ComfyUI inpaint/core.py _solved): the frames `held` [frames] marks kept on their line from their noise to
    // `known` [frames][latent] after every half step, and `known` exactly at the end - the others drawn hearing them;
    // nothing held: solve's arithmetic as it is, to the bit
    bool solve(const float *noise, int steps, float *out, std::string &err,
               const std::function<void(int, int)> &progress = nullptr, const float *known = nullptr,
               const uint8_t *held = nullptr);
    // solve's states as each velocity reads them (evaluation i: step i / 2, its first or its midpoint one), for the
    // parity tool; none: nothing
    std::function<void(int, const float *)> on_state;
    // asked before every velocity of solve: true stops it (solve returns false, err "cancelled")
    std::function<bool()> cancelled;

    struct Prof {   // seconds, summed over the evaluations
        double cpu = 0, qkv = 0, attn = 0, o = 0, gu = 0, down = 0;   // wall time of each part (gu: the whole MLP
                                                                     // when it is one call)
        double attn_dsp = 0;                                         // the DSP's own (the attention calls)
        double gemm_dsp = 0, convert = 0;                            // the products': the DSP's, the CPU's conversions
        double hvx = 0, dsp_call = 0;   // the driver's (npu/nar_cfg.h): its steps between the products, its calls in all
        double overlap = 0;             // the GPU backend's pipeline: the attention with the helper's products beside it
        int gpu_slices = 0, gpu_slices_n = 0;   // its units the GPU took, summed; the layers it ran
        int units = 0;                          // the units a layer (the last layer's)
        int groups = 0;                         // the DSP driver's calls a velocity (its layers in groups: a long chunk)
        double remap = 0;                       // the groups' buffers into / out of the DSP's address space
        // the pipeline's helper: its products' start (rows copied in, queued), finish (the wait, rows copied out), the
        // GPU's kernels by their events, the waits for attended slices
        double gpu_in = 0, gpu_fin = 0, gpu_k = 0, helper_wait = 0;
        // the v66 DSP's helper (set_hvx): the DSP's own time, its calls, its CPU side (quantizing, converting, glue),
        // the layers' weights into its slot; its units, summed
        double hvx_dsp = 0, hvx_call = 0, hvx_cpu = 0, hvx_slot = 0;
        int hvx_units = 0;
        int evals = 0;
    };
    const Prof &prof() const { return prof_; }
    // the GPU backend's pipeline: the attention's slices, the products' units to a slice (the parity tool's sweep)
    void set_pipe(int slices, int split, int prefetch = -1, int width = -1, int unit = -1) {
        pipe_slices_ = std::max(2, slices), pipe_split_ = std::max(1, split);
        if (prefetch >= 0) prefetch_ = prefetch != 0;
        if (width >= 0) pipe_width_ = width;
        if (unit >= 0) gpu_bits_ = unit;
    }
    // the GPU backend's weight copies into its slots so far: seconds, count ({0, 0} without the GPU)
    std::pair<double, int64_t> gpu_uploads() const;
    void reset_prof() { prof_ = Prof{}; }

private:
    struct Buf {   // memory the DSP maps (gnpu_shared_alloc)
        uint8_t *p = nullptr;
        size_t len = 0;
        int fd = -1;
        bool mapped = true;   // in the DSP's address space now (lazy_: only while a group of layers that reads it runs)
    };
    struct LayerW {
        Buf w;                                          // the layer's stretch of nar_f16.bin
        uint64_t qkv = 0, o = 0, gu = 0, down = 0;     // the matrices' offsets in it
        int sh_qkv = 0, sh_o = 0, sh_gu = 0, sh_down = 0;
        const float *in_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *mlp_norm = nullptr;
        bool q8 = false;                                // its matrices hmx8 (nar_q8.bin), not hmx16
        bool q8cpu = false;                             // the CPU's products from their hmx8 twins (Emu::q8)
    };
    bool alloc(Buf &b, size_t bytes, std::string &err);   // b at least `bytes` (kept when it is; a smaller one freed)
    void free_buf(Buf &b);                                // given back (the driver closed first)
    bool map_buf(Buf &b, std::string &err);               // into the DSP's address space (lazy_)
    void unmap_buf(Buf &b);                               // out of it, its contents kept
    // layer l's buffers in Ls (its weights, its prefix banks): the bytes not mapped now; in; out
    size_t unit_unmapped(const std::vector<LayerW> &Ls, int l) const;
    bool map_unit(const std::vector<LayerW> &Ls, int l, std::string &err);
    void unmap_unit(const std::vector<LayerW> &Ls, int l);
    // layers' weights: the NAR's (h16.*, nar_f16.bin) or the AR's (a16.*, ar_f16.bin), a stretch a layer
    bool load_layers(bool ar, std::vector<LayerW> &out, std::string &err);
    void drop_layers(std::vector<LayerW> &Ls);   // their buffers given back (the driver closed first)
    bool ensure_nar(std::string &err);
    bool ensure_ar(std::string &err);   // ar_ read (load_ar_async's thread joined first)
    bool embed_x(const int32_t *ids, int n, std::string &err);   // IO's x: the tokens' embeddings (bf16 rows)
    void close_driver();
    bool setup_chunk(int A, int frames, std::string &err);   // the banks (zeroed), RB, the rows' RoPE
    // cf's layers from Ls (P: their norms, q_norm times the queries' scale; the banks; RB's images), IO for cf.n rows
    // at positions pos0 .. (x left as it is, the rope tables written); nar_open
    // The layers in groups (groups_) whose buffers fit the DSP's window beside P, IO, RB and the rows' bank; the first
    // group opened
    bool open_cfg(nar_cfg_t &cf, const std::vector<LayerW> &Ls, int pos0, std::string &err);
    bool open_group(int g, std::string &err);   // the driver over group g's layers (theirs mapped, others out if need be)
    // the layers through the driver, a call a group (upto: only the first upto layers; flags: run_driver's)
    bool run_layers(int upto, int flags, std::string &err);
    bool gemm(const Buf &w, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x, int64_t xs,
              float *y, int64_t ys, std::string &err);
    bool layer(int l, std::string &err);
    // after the attention: o, the residual, the MLP, the residual for tokens [r0, r1) (helper: the pipeline's thread)
    enum class Tail { Hybrid, Gpu, Cpu };   // post_attn's products: gemm()'s, the GPU's alone, the CPU's alone
    bool post_attn(const LayerW &L, int r0, int r1, Tail mode, std::string &err);
    bool pipelined(const LayerW &L, const void *desc, std::string &err);   // desc: the layer's attn_desc_t
    bool gpu_product(const LayerW &L, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x,
                     int64_t xs, float *y, int64_t ys, std::string &err);
    bool cpu_product(const LayerW &L, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *x,
                     int64_t xs, float *y, int64_t ys, std::string &err);
    bool open_driver(std::string &err);   // the velocity's
    struct Hvx;                           // the v66 DSP's (set_hvx)
    bool hvx_slot(const LayerW &L, std::string &err);   // L's o, gate/up, down 8-bit twins into its slot
    // its unit: o, the residual, the MLP, the residual for tokens [r0, r1) (the pipeline's DSP helper); upto: the
    // rows done (on a failure the rest, from upto, are as the attention left them)
    bool hvx_unit(const LayerW &L, int r0, int r1, Prof &P, std::string &err, int &upto);
    bool run_driver(int flags, std::string &err);

    std::unique_ptr<NpuCalls> npu_;
    std::unique_ptr<Hvx> hvx_;   // set_hvx's (given back before npu_)
    bool hvx_off_ = false;       // it failed once: the rest without it
    bool hvx_paused_ = false;    // set_hvx_mode
    int hvx_nice_ = 0;           // its helper thread's nice (GEODE_YUE2_NAR_HVX_NICE)
    bool hvx_ahead_ = true;      // the next layer's twins read ahead from storage (GEODE_YUE2_NAR_HVX_AHEAD)
    std::string hvx_note_;
    Pack *pack_ = nullptr;
    ThreadPool *pool_ = nullptr;
    Config c_;
    std::string backend_;
    std::vector<LayerW> layers_;
    std::vector<LayerW> ar_;           // the AR's layers (ar_f16.bin) while prompts and prefixes want them
    std::thread ar_thread_;            // load_ar_async's reading
    std::string ar_err_;               // its failure
    bool ar_wanted_ = false;           // load_ar_async started it
    std::thread nar_thread_;           // load_nar_async's reading (joins ar_thread_ first)
    std::string nar_err_;              // its failure
    void join_loads();                 // both reading threads joined (nar_thread_ first: it may join ar_thread_)
    int64_t ar_bytes() const;          // ar_f16's layers' bytes
    std::vector<uint16_t> vae2llm_w_, llm2vae_w_, t0_w_, t2_w_, pe_;
    std::vector<float> vae2llm_b_, llm2vae_b_, t0_b_, t2_b_, norm_;
    int slab_ = 1024;          // rows a product (GEODE_YUE2_NAR_SLAB)
    uint32_t attn_flags_ = 0;  // more attn_cfg.h flags (GEODE_YUE2_NAR_ATTN_FLAGS)
    bool mlp_ = false;         // the MLP in one DSP call (gnpu_mlp_f16; GEODE_YUE2_NAR_MLP=0: three products)
    bool drv_ = false;         // every layer on the DSP in one call (npu/nar_cfg.h; GEODE_YUE2_NAR_DRIVER=0: host-driven)
    bool cpu_ = false;         // the CPU backend (open's npu_dir "cpu"): the layers borrowed from the pack's mapping
    bool want_q8_ = false;     // set_weights_q8
    bool want_attn8_ = false;  // set_attn8
    bool pipe_ = false;        // the GPU backend: the attention overlapped with the GPU's products (GEODE_YUE2_NAR_PIPE)
    int pipe_slices_ = 12;     // its slices of the rows (GEODE_YUE2_NAR_PIPE_SLICES)
    int pipe_split_ = 2;       // the products' units to a slice (GEODE_YUE2_NAR_PIPE_SPLIT)
    bool prefetch_ = true;     // the pipeline's helper copies the next layer into the GPU (GEODE_YUE2_NAR_GPU_PREFETCH)
    // the GPU pipeline's ways (GEODE_YUE2_NAR_GPU_BITS, default all): 1 the helper's units wholly on the GPU
    // (NarGpu::unit; off: product by product), 2 two of them in flight while the attention runs, 4 the attention's
    // first slice one unit (the GPU starts sooner), 8 the layer after next read ahead from storage (MADV_WILLNEED)
    int gpu_bits_ = 15;
    int pipe_width_ = 0;       // the pool's threads for the attention beside the helper, 0: all (GEODE_YUE2_NAR_PIPE_WIDTH)
    Prof hprof_;               // the pipeline helper's times, folded into prof_ after each layer
    Buf p_, io_;               // the driver's norms and chunk blocks, its x / rope / scratch
    uint64_t io_bytes_ = 0;
    // the DSP's address space (~3.8 GB, npu_probe window) cannot hold a long chunk's weights and banks at once (a
    // 5-minute song: ~4.4 GB): with gnpu_shared_map the buffers are mapped only while a group of layers that reads them
    // runs - a velocity in several calls, the buffers of the layers needed last taken out first
    bool lazy_ = false;
    size_t window_ = 0;        // bytes the mappings may take (GEODE_YUE2_NAR_WINDOW_MB)
    size_t mapped_ = 0;        // the buffers' bytes mapped now
    std::vector<std::pair<int, int>> groups_;   // [first, end) layers a driver call
    int group_ = -1;                            // the group the driver holds open
    const std::vector<LayerW> *gls_ = nullptr;  // the layers the groups are of (the NAR's, the AR's during a prefix)
    nar_cfg_t gcf_{};                           // the call's cfg with every layer (a group's: its stretch of them)
    // the chunk
    int A_ = 0, frames_ = 0, N_ = 0, npre_ = 0, rem_ = 0, ntail_ = 0;
    std::vector<Buf> pre_k_, pre_v_;                  // per layer: the prefix's whole blocks
    Buf tail_k_, tail_v_;                             // the prefix's last A % 32 positions, then the rows'
    Buf rb_;                                          // RB: per layer those A % 32 positions' blocks (nb 1)
    bool drv_open_ = false;                           // the driver holds a chunk (nar_open)
    PrefixTimes pt_;
    std::vector<std::vector<float>> rem_k_, rem_v_;   // per layer: those A % 32 positions' rows
    std::vector<float> cos_, sin_;                    // RoPE at A .. A + N - 1: [N][hd / 2]
    std::vector<float> x_, h_, qkv_, o_, gu_, act_;
    std::vector<float> h2_, o2_, gu2_, act2_;          // the pipeline's CPU slices' own (the helper has the ones above)
    float *qb_ = nullptr, *ob_ = nullptr;             // the attention's queries and output (N x heads x hd)
    Prof prof_;
};

}  // namespace geode::yue2
