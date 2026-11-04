// gpu8.cpp - the Adreno 830's int8 path (gpu8.hpp).
#include "gpu8.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "opencl.hpp"
#include "platform.hpp"

#include <CL/cl_ext.h>

namespace geode {

namespace {

const char *const kSource =
#include "gpu8_kernels.inc"
    ;

constexpr int kT = 8;   // tokens a tile (-D T)

template <typename T>
void arg(const cl::Api *cl, cl_kernel k, cl_uint i, const T &v) {
    cl->clSetKernelArg(k, i, sizeof v, &v);
}

size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// the kernel's int8: matrix offset (uint4), tile, offset into Y, global scale bits, buffer W0..W15, rows, -, -
using Job = std::array<int32_t, 8>;
constexpr int kWBufs = 16;

int32_t float_bits(float f) {
    int32_t i;
    std::memcpy(&i, &f, 4);
    return i;
}

// the Adreno's number from "QUALCOMM Adreno(TM) 830": 0 when it is not one
int adreno_number(const std::string &name) {
    const size_t at = name.find("Adreno");
    if (at == std::string::npos) return 0;
    for (size_t i = at; i < name.size(); ++i)
        if (name[i] >= '0' && name[i] <= '9') return std::atoi(name.c_str() + i);
    return 0;
}

}  // namespace

struct Gpu8::Impl {
    cl::Device dev;
    const cl::Api *cl = nullptr;
    cl::Program prog;
    cl::Program prog_pf;   // the prompt's token mixers (pf_setup): its failure leaves the rest of the GPU path working
    std::string cache_dir;   // the compiled programs between runs (set_cache_dir)
    bool q4i = false;        // set_q4i: the programs built for q4i
    Gpu8::HostGemm host_gemm;   // a prompt's projections elsewhere (set_host_gemm)
    std::mutex pieces_mu;    // alloc() from several threads (the expert cache's slabs)
    cl_kernel k_quant = nullptr, k_swiglu = nullptr, k_gemm = nullptr, k_quant8 = nullptr, k_gemm8 = nullptr;

    struct Piece {
        uint8_t *host = nullptr;
        size_t bytes = 0;
        cl_mem mem = nullptr;
        bool imported = false;   // a dma-buf of the caller's (import): not mapped here, not counted in allocated
    };
    std::vector<Piece> pieces;
    size_t allocated = 0;
    // an image's pieces (vision_alloc: carved from blocks of their own), and its activations' int8 tiles
    std::vector<Piece> vblocks;
    size_t vused = 0;   // of the last block
    Piece vX, vXS, vXB;
    std::unordered_map<const uint8_t *, const uint8_t *> moved;   // adopted: the old bytes -> the copy

    // staging (each its own piece)
    Piece in, x, xs, xb, rowmap, ymap_d, ymap_id, ymap_dn, jobs_d, jobs_gu, jobs_dn, y, gu, x2, x2s, x2b, eout;
    int tokens = 0;
    int64_t max_cols = 0, max_rows = 0, hidden = 0, ff = 0;
    int top_k = 0, n_experts = 0;
    int top_k_max = 0, guesses_max = 0;   // what the buffers were made for (reserve, setup_decode)
    int ag = 0;   // the flash decoding's query heads a KV head (attn_part*: 8 or 4, the programs built for it), else 0
    size_t slots_e = 0;

    std::vector<cl_event> events;
    std::vector<const char *> ev_names;   // each event's kernel (GEODE_PF_PROF: a prompt's kernels timed by name)
    const char *cur = "batch";
    const bool pf_prof = std::getenv("GEODE_PF_PROF") != nullptr;
    std::unordered_map<std::string, std::pair<double, int64_t>> pf_times;
    double busy = 0.0;
    int64_t launches = 0;

    // generating: the program built with T = 1 (and T = 2 ..: verifying a batch), a plain in-order queue, the tokens'
    // buffers
    cl::Program prog1, progT[Gpu8::kMaxTokens + 1];
    cl_command_queue q1 = nullptr;
    std::unordered_map<std::string, cl_kernel> k1, kTok[Gpu8::kMaxTokens + 1];
    int T = 1;   // the tokens of the passes now (set_tokens)
    int rope_shift = 0;   // set_rope_shift
    std::unordered_map<std::string, cl_kernel> &kmap() { return T == 1 ? k1 : kTok[T]; }
    Gpu8::DecodeShape ds;
    // Gemma 4's answer (setup_gemma, gemma_step): its shape, the per-layer inputs (ple_model's output, the table's
    // rows, a layer's gate output)
    Gpu8::GemmaShape gs;
    bool gemma = false;
    Piece d_pin, d_prow, d_pg, d_flag;
    // ---- a Gemma prompt's layers (gemma_prompt): the rows of its n tokens, the int8 tiles the products read, its
    // products' jobs (a cursor, as jcur), the attention's parts past two chunks (kGpGroup tokens at a time)
    Piece gp_x, gp_h, gp_q, gp_k, gp_v, gp_core, gp_mix, gp_gu, gp_pin, gp_pg, gp_X, gp_XS, gp_XB, gp_jobs, gp_ymap,
        gp_po, gp_pm, gp_pl;
    int gp_cap = 0, gp_parts = 0, gp_jcur = 0, gp_jobs_n = 0;
    bool gp_pf = false;   // the attention flash-style (g_attn_pf*: work groups of 256 it can run)
    bool gp_gemm1 = true;   // the products by gemm1 (its matrix one argument: ~2% faster; GEODE_GP_GEMM1=0: gemm)
    cl_mem gp_Xi = nullptr;   // gp_X as an image1d_buffer (gemm1x: the activations through the texture path)
    static constexpr int kGpGroup = 64;
    std::unordered_map<std::string, cl_kernel> kb8;   // the batch program's Gemma kernels (T = 8: XAT's tiles)

    // a kernel of the batch program onto the one-token queue (go's twin: the prompt's layers run there)
    template <typename... A>
    bool bgo(const char *name, size_t g0, size_t g1, size_t l0, size_t l1, std::string &err, const A &...a) {
        auto it = kb8.find(name);
        if (it == kb8.end()) {
            err = std::string("GPU: no kernel ") + name;
            return false;
        }
        cl_uint i = 0;
        (arg(cl, it->second, i++, a), ...);
        const size_t g[2] = {g0, g1 ? g1 : 1}, l[2] = {l0, l1 ? l1 : 1};
        cl_event ev = nullptr;
        const cl_int e = cl->clEnqueueNDRangeKernel(q1, it->second, g1 ? 2 : 1, nullptr, g, l0 ? l : nullptr, 0,
                                                    nullptr, dprof ? &ev : nullptr);
        if (ev) dev_events.emplace_back(std::string("p ") + name, ev);
        if (e != CL_SUCCESS) {
            err = std::string("GPU: ") + name + ": " + cl::error_name(e);
            return false;
        }
        ++launches;
        return true;
    }
    // out's rows (ystride floats apart, from yoff) of the n tokens = m . gp_X's tiles: the batch gemm onto the
    // one-token queue, its jobs a range of gp_jobs
    bool gp_gemm(const QView &m, int n, const Piece &out, int ystride, int yoff, std::string &err) {
        const Piece *w = find(m.data);
        const int tiles = (n + kT - 1) / kT;
        if (!w || gp_jcur + tiles > gp_jobs_n) {
            err = "GPU: a prompt's product it cannot take";
            return false;
        }
        auto *jobs = reinterpret_cast<Job *>(gp_jobs.host);
        const auto off = static_cast<int32_t>((m.data - w->host) / 16);
        const int first = gp_jcur;
        for (int j = 0; j < tiles; ++j)
            jobs[gp_jcur++] = {off, j, yoff, float_bits(m.global * 256.0f), 0, static_cast<int32_t>(m.rows), 0, 0};
        // (gemm1: the matrix one argument - its loads the texture path; gemm: a buffer of 16 picked a job)
        auto g1 = kb8.find(gp_Xi ? "gemm1x" : "gemm1");
        const bool one = g1 != kb8.end() && gp_gemm1;
        cl_kernel k = one ? g1->second : k_gemm;
        const cl_uint a0 = one ? 1 : kWBufs;
        for (cl_uint i = 0; i < a0; ++i) arg(cl, k, i, w->mem);
        arg(cl, k, a0 + 0, gp_jobs.mem);
        arg(cl, k, a0 + 1, one && gp_Xi ? gp_Xi : gp_X.mem);
        arg(cl, k, a0 + 2, gp_XS.mem);
        arg(cl, k, a0 + 3, gp_XB.mem);
        arg(cl, k, a0 + 4, gp_ymap.mem);
        arg(cl, k, a0 + 5, out.mem);
        arg(cl, k, a0 + 6, static_cast<int>(m.cols));
        arg(cl, k, a0 + 7, ystride);
        arg(cl, k, a0 + 8, first);
        const size_t quads = static_cast<size_t>(m.rows / 4);
        const size_t g[2] = {static_cast<size_t>(tiles), quads}, l[2] = {1, 64};
        cl_event ev = nullptr;
        const cl_int e = cl->clEnqueueNDRangeKernel(q1, k, 2, nullptr, g, quads % 64 == 0 ? l : nullptr, 0, nullptr,
                                                    dprof ? &ev : nullptr);
        if (ev)
            dev_events.emplace_back("p gemm " + std::to_string(m.rows) + "x" + std::to_string(m.cols), ev);
        if (e != CL_SUCCESS) {
            err = std::string("GPU: a prompt's gemm: ") + cl::error_name(e);
            return false;
        }
        ++launches;
        return true;
    }
    Piece params, d_x, d_h, d_X, d_XS, d_XB, d_proj, d_z, d_ab, d_core, d_qg, d_kb, d_vb, d_mix, d_scores, d_rlog,
        d_glog, d_glog2, d_gids2, d_po, d_pm, d_pl, d_ids, d_gids, d_w, d_gu, d_x2, d_x2s, d_x2b, d_eo, d_jobs, d_ymap, d_logits, d_freq;
    int jcur = 0, seq = 0;
    bool guessed = false, guessed2 = false;   // the last route made guesses (of the next layer, of the one after)
    static constexpr int kJobs = 8192;
    // GEODE_GPU8_PROF=1: the one-token queue profiles, every launch's time summed by kernel (a measuring switch)
    bool dprof = false, wait_sleep = false;
    bool gdn_old = false;   // GEODE_GPU8_GDN_OLD=1: the two-pass gdn_delta for dk = 128 too (measuring)
    bool attn_old = false;  // GEODE_GPU8_ATTN_OLD=1: the attention without flash decoding (measuring)
    int nb_max = 0;         // flash decoding's blocks of 128 positions, the most a context has
    std::string label_prefix;   // the profile's name for the next launches (measuring)
    std::vector<std::pair<std::string, cl_event>> dev_events;
    std::unordered_map<std::string, std::pair<double, int64_t>> dtimes;
    void collect() {
        for (auto &[name, ev] : dev_events) {
            auto &t = dtimes[name];
            t.first += cl::event_seconds(cl, ev);
            ++t.second;
        }
        dev_events.clear();
    }

    ~Impl() {
        if (!cl) return;
        if (q1) {
            cl->clFinish(q1);
            if (q1 != dev.queue) cl->clReleaseCommandQueue(q1);
        }
        for (cl_event ev : events) cl->clReleaseEvent(ev);
        if (gp_Xi) cl->clReleaseMemObject(gp_Xi);   // (before its buffer)
        for (const Piece &p : pieces) {
            if (!p.imported) cl->clEnqueueUnmapMemObject(dev.queue, p.mem, p.host, 0, nullptr, nullptr);
            cl->clReleaseMemObject(p.mem);
        }
        if (dev.queue) cl->clFinish(dev.queue);
    }

    bool alloc(size_t bytes, Piece &out, std::string &err) {
        cl_int e = CL_SUCCESS;
        bytes = std::max<size_t>(round_up(bytes, 4096), 4096);
        cl_mem m = cl->clCreateBuffer(dev.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &e);
        if (!m) {
            err = std::string("GPU: a buffer of ") + std::to_string(bytes >> 20) + " MB: " + cl::error_name(e);
            return false;
        }
        void *h = cl->clEnqueueMapBuffer(dev.queue, m, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr,
                                         nullptr, &e);
        if (!h) {
            cl->clReleaseMemObject(m);
            err = std::string("GPU: mapping a buffer: ") + cl::error_name(e);
            return false;
        }
        out = {static_cast<uint8_t *>(h), bytes, m};
        std::lock_guard<std::mutex> lock(pieces_mu);   // the expert cache's slabs come from several threads at once
        pieces.push_back(out);
        allocated += bytes;
        return true;
    }

    bool import(int fd, uint8_t *host, size_t bytes, std::string &err) {
#if defined(__ANDROID__)
        cl_mem_ion_host_ptr ip{};
        ip.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
        ip.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        ip.ion_filedesc = fd;
        ip.ion_hostptr = host;
        cl_int e = CL_SUCCESS;
        cl_mem m = cl->clCreateBuffer(dev.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                      bytes, &ip, &e);
        if (!m) {
            err = std::string("GPU: a dma-buf of ") + std::to_string(bytes >> 20) + " MB: " + cl::error_name(e);
            return false;
        }
        std::lock_guard<std::mutex> lock(pieces_mu);
        pieces.push_back({host, bytes, m, true});
        return true;
#else
        (void)fd, (void)host, (void)bytes;
        err = "GPU: no dma-bufs here";
        return false;
#endif
    }

    const Piece *find(const void *p) const {
        const auto *b = static_cast<const uint8_t *>(p);
        for (const Piece &x : pieces)
            if (b >= x.host && b < x.host + x.bytes) return &x;
        return nullptr;
    }

    bool run(cl_kernel k, const size_t *off, const size_t *g, const size_t *l, std::string &err) {
        cl_event ev = nullptr;
        const cl_int e = cl->clEnqueueNDRangeKernel(dev.queue, k, 2, off, g, l, 0, nullptr, &ev);
        if (e != CL_SUCCESS) {
            err = std::string("GPU: enqueue: ") + cl::error_name(e);
            return false;
        }
        events.push_back(ev);
        ev_names.push_back(cur);
        ++launches;
        return true;
    }

    bool finish(std::string &err) {
        const cl_int e = cl->clFinish(dev.queue);
        for (size_t i = 0; i < events.size(); ++i) {
            const double t = cl::event_seconds(cl, events[i]);
            busy += t;
            if (pf_prof) {
                auto &pt = pf_times[ev_names[i]];
                pt.first += t;
                ++pt.second;
            }
        }
        events.clear();
        ev_names.clear();
        if (e != CL_SUCCESS) {
            err = std::string("GPU: finish: ") + cl::error_name(e);
            return false;
        }
        return true;
    }

    // quant: rows of `in` (cols floats) into the tiles X / XS / XB, slots of them
    bool quant(const Piece &qx, const Piece &qs, const Piece &qb, int cols, size_t slots, std::string &err) {
        arg(cl, k_quant, 0, in.mem);
        arg(cl, k_quant, 1, rowmap.mem);
        arg(cl, k_quant, 2, cols);
        arg(cl, k_quant, 3, qx.mem);
        arg(cl, k_quant, 4, qs.mem);
        arg(cl, k_quant, 5, qb.mem);
        const size_t g[2] = {slots, static_cast<size_t>(cols / 16)};
        return run(k_quant, nullptr, g, nullptr, err);
    }

    // gemm: jobs [first, first + count) of `jobs` against the matrices in `w`, out to Y rows by `ymap`
    bool gemm(const Piece &w, const Piece &jobs, size_t first, size_t count, const Piece &qx, const Piece &qs,
              const Piece &qb, const Piece &ymap, const Piece &out, int rows, int cols, std::string &err,
              int ystride = 0) {
        for (cl_uint i = 0; i < kWBufs; ++i) arg(cl, k_gemm, i, w.mem);
        arg(cl, k_gemm, kWBufs + 0, jobs.mem);
        arg(cl, k_gemm, kWBufs + 1, qx.mem);
        arg(cl, k_gemm, kWBufs + 2, qs.mem);
        arg(cl, k_gemm, kWBufs + 3, qb.mem);
        arg(cl, k_gemm, kWBufs + 4, ymap.mem);
        arg(cl, k_gemm, kWBufs + 5, out.mem);
        arg(cl, k_gemm, kWBufs + 6, cols);
        arg(cl, k_gemm, kWBufs + 7, ystride ? ystride : rows);
        arg(cl, k_gemm, kWBufs + 8, static_cast<int>(first));
        const size_t quads = static_cast<size_t>(rows / 4);
        const size_t g[2] = {count, quads}, l[2] = {1, 64};
        return run(k_gemm, nullptr, g, quads % 64 == 0 ? l : nullptr, err);
    }

    // ---- generating ----
    // a kernel of the one-token program with these arguments (cl_mem, int, float), onto q1
    template <typename... A>
    bool go(const char *name, size_t g0, size_t g1, size_t l0, size_t l1, std::string &err, const A &...a) {
        auto &km = kmap();
        auto it = km.find(name);
        if (it == km.end()) {
            err = std::string("GPU: no kernel ") + name;
            return false;
        }
        cl_uint i = 0;
        (arg(cl, it->second, i++, a), ...);
        const size_t g[2] = {g0, g1 ? g1 : 1}, l[2] = {l0, l1 ? l1 : 1};
        cl_event ev = nullptr;
        const cl_int e = cl->clEnqueueNDRangeKernel(q1, it->second, g1 ? 2 : 1, nullptr, g, l0 ? l : nullptr, 0,
                                                    nullptr, dprof ? &ev : nullptr);
        if (ev) dev_events.emplace_back(name, ev);
        if (e != CL_SUCCESS) {
            err = std::string("GPU: ") + name + ": " + cl::error_name(e);
            return false;
        }
        ++launches;
        return true;
    }

    // A one-token GEMM launch: matrices (each from a tile of X, into Y at an offset) over up to 16 buffers a launch
    struct Mat {
        const QView *m;
        int tile, z;
    };
    bool gemv_many(const std::vector<Mat> &ms, const Piece &qx, const Piece &qs, const Piece &qb, const Piece &out,
                   int cols, int ystride, std::string &err) {
        for (size_t i = 0; i < ms.size();) {
            std::vector<const Piece *> bufs;
            const int first = jcur;
            int rows = 0;
            size_t j = i;
            for (; j < ms.size(); ++j) {
                const Piece *w = find(ms[j].m->data);
                if (!w) {
                    err = "GPU: a matrix outside shared memory";
                    return false;
                }
                auto it = std::find(bufs.begin(), bufs.end(), w);
                if (it == bufs.end()) {
                    if (static_cast<int>(bufs.size()) == kWBufs) break;   // the rest in another launch
                    bufs.push_back(w);
                    it = bufs.end() - 1;
                }
                if (jcur >= kJobs) {
                    err = "GPU: out of jobs";
                    return false;
                }
                const QView &m = *ms[j].m;
                reinterpret_cast<Job *>(d_jobs.host)[jcur++] = {
                    static_cast<int32_t>((m.data - w->host) / 16), ms[j].tile, ms[j].z, float_bits(m.global * 256.0f),
                    static_cast<int32_t>(it - bufs.begin()), static_cast<int32_t>(m.rows), 0, 0};
                rows = std::max(rows, static_cast<int>(m.rows));
            }
            if (!gemv(bufs, first, jcur - first, qx, qs, qb, out, rows, cols, ystride, err)) return false;
            i = j;
        }
        return true;
    }

    // jobs [first, first + count) of the one-token GEMM (their matrices in bufs) against X / XS / XB, out into Y
    bool gemv(const std::vector<const Piece *> &bufs, int first, int count, const Piece &qx, const Piece &qs,
              const Piece &qb, const Piece &out, int rows, int cols, int ystride, std::string &err) {
        // the columns split four ways (gemvs*: 4x the waves); one buffer: the kernel without the pick among them
        const bool one = bufs.size() == 1;
        cl_kernel k = kmap()[one ? "gemvs1" : "gemvs"];
        const cl_uint nw = one ? 1 : kWBufs;
        for (cl_uint i = 0; i < nw; ++i) arg(cl, k, i, bufs[std::min<size_t>(i, bufs.size() - 1)]->mem);
        arg(cl, k, nw + 0, d_jobs.mem);
        arg(cl, k, nw + 1, qx.mem);
        arg(cl, k, nw + 2, qs.mem);
        arg(cl, k, nw + 3, qb.mem);
        arg(cl, k, nw + 4, d_ymap.mem);
        arg(cl, k, nw + 5, out.mem);
        arg(cl, k, nw + 6, cols);
        arg(cl, k, nw + 7, ystride);
        arg(cl, k, nw + 8, first);
        const size_t quads = static_cast<size_t>(rows / 4), split = 4;
        const size_t g[2] = {static_cast<size_t>(count), (quads + 63) / 64 * 64 * split}, l[2] = {1, 64 * split};
        cl_event ev = nullptr;
        const cl_int e = cl->clEnqueueNDRangeKernel(q1, k, 2, nullptr, g, l, 0, nullptr, dprof ? &ev : nullptr);
        if (ev)
            dev_events.emplace_back(label_prefix + (T > 1 ? "T" + std::to_string(T) + " " : std::string()) + (one ? "gemv1 " : "gemv ") + std::to_string(rows) + "x" + std::to_string(cols) + " x" +
                                        std::to_string(count) + " in " + std::to_string(bufs.size()),
                                    ev);
        if (e != CL_SUCCESS) {
            err = std::string("GPU: gemv: ") + cl::error_name(e);
            return false;
        }
        ++launches;
        return true;
    }

    // y = m . the vectors in X (token t's into row t of out)
    bool gemv1(const QView &m, const Piece &out, std::string &err) {
        return gemv_many({{&m, 0, 0}}, d_X, d_XS, d_XB, out, static_cast<int>(m.cols), static_cast<int>(m.rows), err);
    }

    // ---- a prompt's token mixers (pf_gdn / pf_attn): their buffers (pf_setup, at the first), their kernels on the
    // batch queue, the products from the tiles in x / xs / xb into one of them
    std::unordered_map<std::string, cl_kernel> kpf;
    Piece pf_proj, pf_z, pf_ab, pf_core, pf_core4, pf_qg, pf_kb, pf_vb, pf_mix, pf_pos, pf_jobs, pf_pm, pf_pl, pf_po,
        pf_lg, pf_sg;
    bool pf_ready = false;
    bool attn4 = false;   // pf_attn8t's sub-groups are here (pf_probe: a wave of 64 is one)
    int pf_jcur = 0;
    static constexpr int kPfJobs = 4096;
    bool pf_setup(std::string &err) {
        if (pf_ready) return true;
        if (!tokens || !ds.hidden) {
            err = "GPU: a prompt's mixers before the staging and the decoding shapes";
            return false;
        }
        const size_t n = static_cast<size_t>(tokens), H = static_cast<size_t>(ds.hidden);
        const size_t kd = static_cast<size_t>(ds.nk * ds.dk), vd = static_cast<size_t>(ds.nv * ds.dv);
        const size_t conv = 2 * kd + vd, nh = static_cast<size_t>(ds.nh), hd = static_cast<size_t>(ds.hd);
        const size_t cw = std::max(vd, nh * hd), kv_row = static_cast<size_t>(ds.nkv) * hd;
        if (!prog_pf.build_cached(dev, kSource,
                                  "-cl-std=CL3.0 -cl-mad-enable -DGEODE_PF -DGEODE_PF_ONLY -DT=" + std::to_string(kT),
                                  cache_dir, err))
            return false;
        for (const char *k : {"pf_bf16_gemm", "pf_bf16_gemm2", "pf_gdn_conv", "pf_gdn_qk_gates", "pf_gdn_delta32", "pf_gdn_gnorm_q",
                              "pf_head_norm_rope", "pf_kv_store8", "pf_attn8", "pf_attn8x", "pf_attn8t", "pf_attn_out", "pf_attn_out_f",
                              "pf_probe"}) {
            cl_kernel kk = prog_pf.kernel(k, err);
            if (!kk) return false;
            kpf[k] = kk;
        }
        if (!alloc(n * conv * 4, pf_proj, err) || !alloc(n * vd * 4, pf_z, err) ||
            !alloc(n * 2 * static_cast<size_t>(ds.nv) * 4, pf_ab, err) || !alloc(n * cw * 4, pf_core, err) ||
            !alloc(n * 4 * vd * 4, pf_core4, err) ||
            !alloc(n * nh * 2 * hd * 4, pf_qg, err) || !alloc(n * kv_row * 4, pf_kb, err) ||
            !alloc(n * kv_row * 4, pf_vb, err) || !alloc(n * H * 4, pf_mix, err) || !alloc(n * 3 * 4, pf_pos, err) ||
            !alloc(kPfJobs * sizeof(Job), pf_jobs, err) || !alloc(n * nh * 4, pf_pm, err) ||
            !alloc(n * nh * 4, pf_pl, err) || !alloc(n * nh * hd * 4, pf_po, err) ||
            !alloc(n * static_cast<size_t>(ds.n_experts + 1) * 4, pf_lg, err) || !alloc(64, pf_sg, err))
            return false;
        pf_ready = true;
        if (!pgo("pf_probe", 256, 0, 256, 0, err, pf_sg.mem) || !finish(err)) return false;
        const auto *sg = reinterpret_cast<const int32_t *>(pf_sg.host);
        attn4 = sg[0] == 64 && sg[1] == 4;
        cl_uint bits = 0;
        if (cl->clGetDeviceInfo(dev.id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof bits, &bits, nullptr) == CL_SUCCESS)
            sub_align = bits / 8;
        return true;
    }
    template <typename... A>
    bool pgo(const char *name, size_t g0, size_t g1, size_t l0, size_t l1, std::string &err, const A &...a) {
        const auto it = kpf.find(name);
        if (it == kpf.end()) {
            err = std::string("GPU: no kernel ") + name;
            return false;
        }
        cl_uint i = 0;
        (arg(cl, it->second, i++, a), ...);
        const size_t g[2] = {g0, g1 ? g1 : 1}, l[2] = {l0, l1 ? l1 : 1};
        cur = name;
        const bool ok = run(it->second, nullptr, g, l0 ? l : nullptr, err);
        cur = "batch";
        return ok;
    }
    // out (rows `os` floats apart) = the bf16 matrix at P + w (rows x cols) . in's n rows: pf_bf16_gemm2 (tiles of 32
    // tokens x 64 rows); GEODE_PF_BF16=0: pf_bf16_gemm, 8 tokens x 64 rows (measuring)
    bool bf16(cl_mem P, int w, int cols, cl_mem out, int os, int rows, int n, std::string &err, cl_mem src = nullptr) {
        static const bool old = std::getenv("GEODE_PF_BF16") && std::atoi(std::getenv("GEODE_PF_BF16")) == 0;
        if (!src) src = in.mem;
        if (old)
            return pgo("pf_bf16_gemm", round_up(static_cast<size_t>(rows), 64), (static_cast<size_t>(n) + 7) / 8, 64, 1, err,
                       P, w, src, cols, out, os, rows, n);
        return pgo("pf_bf16_gemm2", round_up(static_cast<size_t>(rows), 64), (static_cast<size_t>(n) + 31) / 32, 64, 1,
                   err, P, w, src, cols, out, os, rows, n);
    }
    // n rows of `cols` floats into in and the row maps set, no tiles (the products elsewhere: set_host_gemm)
    void pf_maps(const float *h, int n, int cols) {
        std::memcpy(in.host, h, static_cast<size_t>(n) * static_cast<size_t>(cols) * 4);
        const size_t slots = round_up(static_cast<size_t>(n), kT);
        auto *rm = reinterpret_cast<int32_t *>(rowmap.host), *ym = reinterpret_cast<int32_t *>(ymap_d.host);
        for (size_t s = 0; s < slots; ++s) rm[s] = ym[s] = s < static_cast<size_t>(n) ? static_cast<int32_t>(s) : -1;
    }
    // n rows of `cols` floats into in, then into the tiles (every product of the layer's input reads them)
    bool pf_in(const float *h, int n, int cols, std::string &err) {
        std::memcpy(in.host, h, static_cast<size_t>(n) * static_cast<size_t>(cols) * 4);
        const size_t slots = round_up(static_cast<size_t>(n), kT);
        auto *rm = reinterpret_cast<int32_t *>(rowmap.host), *ym = reinterpret_cast<int32_t *>(ymap_d.host);
        for (size_t s = 0; s < slots; ++s) rm[s] = ym[s] = s < static_cast<size_t>(n) ? static_cast<int32_t>(s) : -1;
        return quant(x, xs, xb, cols, slots, err);
    }
    // out's n rows (m.rows floats apart) = m . the tiles in x / xs / xb (enqueued; its jobs a range of pf_jobs)
    bool pf_gemm(const QView &m, int n, const Piece &out, std::string &err) {
        const Piece *w = find(m.data);
        const size_t tiles = round_up(static_cast<size_t>(n), kT) / kT;
        if (!w || m.cols > max_cols || pf_jcur + static_cast<int>(tiles) > kPfJobs) {
            err = "GPU: a prompt's product it cannot take";
            return false;
        }
        auto *jobs = reinterpret_cast<Job *>(pf_jobs.host);
        const auto off = static_cast<int32_t>((m.data - w->host) / 16);
        const int first = pf_jcur;
        for (size_t j = 0; j < tiles; ++j)
            jobs[pf_jcur++] = {off, static_cast<int32_t>(j), 0, float_bits(m.global * 256.0f), 0,
                               static_cast<int32_t>(m.rows), 0, 0};
        cur = "pf gemm";
        const bool ok = gemm(*w, pf_jobs, static_cast<size_t>(first), tiles, x, xs, xb, ymap_d, out,
                             static_cast<int>(m.rows), static_cast<int>(m.cols), err);
        cur = "batch";
        return ok;
    }

    // ---- a prompt's pipeline (pf_gdn / pf_attn with the projections on the host): the prompt in parts, the host
    // multiplying part c + 1's inputs and part c - 1's output while the GPU runs part c's mixer. The GPU's kernels see
    // a part through sub-buffers (rows of the same pieces): its states carry on from part to part as they do from one
    // call to the next (a prompt's sub-batches), so each token's arithmetic is the one of before.
    size_t sub_align = 0;        // CL_DEVICE_MEM_BASE_ADDR_ALIGN in bytes (pf_setup); 0: no sub-buffers
    std::vector<cl_mem> subs;    // released once the GPU is done with them (drop_subs)
    static constexpr int kPipeAlign = 128;   // a part starts at a multiple of this many tokens
    // part boundaries of n tokens (GEODE_NPU_PIPE parts; default 0: none), or empty: no pipeline. Off by default:
    // on the RedMagic (budget 4800 MB) 2 parts took a 1000-token prompt 125.8 -> 116.7 tok/s and a 4500-token one
    // 163.8 -> 169.2 - a part's projections on the NPU take longer than the GPU's mixer they hide behind, and the
    // smaller calls cost the NPU ~25% more; the same tokens either way
    std::vector<int> pipe_bounds(int n) const {
        static const int want = std::getenv("GEODE_NPU_PIPE") ? std::atoi(std::getenv("GEODE_NPU_PIPE")) : 0;
        if (!host_gemm || !sub_align || want < 2) return {};
        for (int parts = want; parts >= 2; --parts) {
            const int size = static_cast<int>(round_up(static_cast<size_t>((n + parts - 1) / parts), kPipeAlign));
            const int last = n - (parts - 1) * size;
            if (size < 2 * kPipeAlign || last < kPipeAlign / 2) continue;   // parts too small for the NPU
            std::vector<int> b;
            for (int c = 0; c < parts; ++c) b.push_back(c * size);
            b.push_back(n);
            return b;
        }
        return {};
    }
    // rows [first, first + count) of a piece whose rows are `row` bytes, as a buffer of their own (the piece itself
    // when they start it); false when the device cannot place a buffer there
    bool view(const Piece &p, size_t first, size_t count, size_t row, cl_mem &out, std::string &err) {
        if (first == 0) {
            out = p.mem;
            return true;
        }
        const size_t off = first * row;
        if (off % sub_align) {
            err = "GPU: a part's rows at an offset the device cannot take";
            return false;
        }
        const cl_buffer_region reg{off, count * row};
        cl_int e = CL_SUCCESS;
        out = cl->clCreateSubBuffer(p.mem, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &e);
        if (!out) {
            err = std::string("GPU: a part's sub-buffer: ") + cl::error_name(e);
            return false;
        }
        subs.push_back(out);
        return true;
    }
    void drop_subs() {
        for (cl_mem m : subs) cl->clReleaseMemObject(m);
        subs.clear();
    }
    // every part's offsets fit the device's alignment (checked before a pipeline starts: none half done)
    bool pipe_fits(const std::vector<int> &b, std::initializer_list<size_t> rows) const {
        for (size_t c = 1; c + 1 < b.size(); ++c)
            for (size_t r : rows)
                if ((static_cast<size_t>(b[c]) * r) % sub_align) return false;
        return true;
    }

    // an element offset into a piece: params (bf16 or f32), states
    int at(const Piece &p, const void *ptr, size_t elem) const {
        return static_cast<int>((static_cast<const uint8_t *>(ptr) - p.host) / static_cast<ptrdiff_t>(elem));
    }
};

Gpu8::Gpu8() : p_(std::make_unique<Impl>()) {}
Gpu8::~Gpu8() = default;

void Gpu8::set_cache_dir(const std::string &dir) { p_->cache_dir = dir; }
void Gpu8::set_q4i(bool on) { p_->q4i = on; }

void Gpu8::set_host_gemm(HostGemm f) { p_->host_gemm = std::move(f); }

bool Gpu8::open(std::string &err) {
    Impl &p = *p_;
    if (!p.dev.open(err)) return false;
    p.cl = p.dev.cl;
    // the number from the device's name, else from its version: Samsung's driver names an Adreno 740 "QUALCOMM
    // Adreno(TM)" and puts the number in "OpenCL 3.0 Adreno(TM) 740" (a Galaxy S23 Ultra, Remote Test Lab)
    const int number = std::max(adreno_number(p.dev.name), adreno_number(p.dev.version));
    if (number < 700 || !p.dev.has("cl_qcom_dot_product8")) {
        err = "no Adreno 7xx+ with cl_qcom_dot_product8 (" + p.dev.name + ", " + p.dev.version + ")";
        return false;
    }
    // (-cl-fp32-correctly-rounded-divide-sqrt: this driver refuses it)
    if (!p.prog.build_cached(p.dev, kSource,
                             "-cl-std=CL3.0 -cl-mad-enable -DT=" + std::to_string(kT) + (p.q4i ? " -DQ4I" : ""),
                             p.cache_dir, err))
        return false;
    p.k_quant = p.prog.kernel("quant", err);
    p.k_swiglu = p.prog.kernel("swiglu", err);
    p.k_gemm = p.prog.kernel("gemm", err);
    p.k_quant8 = p.prog.kernel("quant8", err);
    p.k_gemm8 = p.prog.kernel("gemm8", err);
    return p.k_quant && p.k_swiglu && p.k_gemm && p.k_quant8 && p.k_gemm8;
}

bool Gpu8::fits_batch(const QView &m, int n) const {
    const Impl &p = *p_;
    return holds(m) && n <= p.tokens && m.rows <= p.max_rows && m.cols <= p.max_cols;
}

uint8_t *Gpu8::dec_params(size_t bytes, std::string &err) {
    Impl &p = *p_;
    return p.alloc(bytes, p.params, err) ? p.params.host : nullptr;
}

float *Gpu8::dec_x() { return reinterpret_cast<float *>(p_->d_x.host); }
float *Gpu8::dec_h() { return reinterpret_cast<float *>(p_->d_h.host); }

bool Gpu8::dec_mtp_in(const QView &fc, const float *pre_e, const float *pre_h, std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    const Impl::Piece &P = p.params;
    return p.go("mtp_in_q", 256, p.T > 1 ? static_cast<size_t>(p.T) : 0, 256, 0, err, p.d_x.mem, p.d_h.mem, P.mem,
                p.at(P, pre_e, 4), p.at(P, pre_h, 4), s.hidden, s.eps, p.d_X.mem, p.d_XS.mem, p.d_XB.mem) &&
           p.gemv1(fc, p.d_x, err);
}

bool Gpu8::setup_decode(const DecodeShape &s, const float *inv_freq, std::string &err) {
    Impl &p = *p_;
    if (s.hidden % 256) {   // norm_q: 256 items
        err = "GPU: a hidden size the one-token kernels do not take";
        return false;
    }
    // route: up to 16 guesses; moe_combine: up to 8 picks (4 bits each of a map word)
    if (s.top_k > 8 || s.guesses2 > 16 || s.guesses > 16 || s.n_experts > 256 || (s.n_experts & (s.n_experts - 1)) || s.dv > 256 || s.dk > 256 || s.dk % 4 || s.hd > 256 ||
        s.dv % 16 || s.hd % 16) {
        err = "GPU: a shape the one-token kernels do not take";
        return false;
    }
    cl_int e = CL_SUCCESS;
    p.dprof = std::getenv("GEODE_GPU8_PROF") && std::atoi(std::getenv("GEODE_GPU8_PROF")) != 0;
    p.wait_sleep = std::getenv("GEODE_GPU8_WAIT") != nullptr;
    p.gdn_old = std::getenv("GEODE_GPU8_GDN_OLD") != nullptr;
    p.attn_old = std::getenv("GEODE_GPU8_ATTN_OLD") != nullptr;
    p.nb_max = (s.max_ctx + 127) / 128 + 1;
    // flash decoding for 8 query heads a KV head (Qwen3.6) or 4 (Qwen3.5-2B: the programs built with -DAG=4)
    {
        const int grp = s.nkv > 0 && s.nh % s.nkv == 0 ? s.nh / s.nkv : 0;
        p.ag = s.hd == 256 && (grp == 8 || grp == 4) ? grp : 0;
    }
    const std::string ag_opt = p.ag == 4 ? " -DAG=4" : "";
    if (std::getenv("GEODE_GPU8_Q0"))   // measuring: generate on the batch queue (it profiles)
        p.q1 = p.dev.queue;
    else
        p.q1 = p.cl->clCreateCommandQueue(p.dev.context, p.dev.id, p.dprof ? CL_QUEUE_PROFILING_ENABLE : 0, &e);
    if (!p.q1) {
        err = std::string("GPU: a plain queue: ") + cl::error_name(e);
        return false;
    }
    if (!p.prog1.build_cached(p.dev, kSource, "-cl-std=CL3.0 -cl-mad-enable -DT=1" + ag_opt, p.cache_dir, err)) return false;
    for (const char *n : {"gemm", "gemm1", "gemvs", "gemvs1", "swiglu", "norm_q", "mtp_in_q", "bf16_rows", "gdn_conv", "gdn_qk", "gdn_gates", "gdn_delta", "gdn_delta32",
                          "gdn_gnorm_q", "head_norm_rope", "kv_store", "attn_scores", "attn_softmax", "attn_values",
                          "attn_part", "attn_part8", "kv_store8", "attn_merge", "attn_gate_q", "route", "guess_top", "moe_combine",
                          "mlp_add"}) {
        cl_kernel k = p.prog1.kernel(n, err);
        if (!k) return false;
        p.k1[n] = k;
    }
    p.ds = s;
    p.top_k_max = std::max(p.top_k_max, s.top_k);
    p.guesses_max = s.guesses;
    const size_t f = 4, H = static_cast<size_t>(s.hidden), K = static_cast<size_t>(s.top_k),
                 ff = static_cast<size_t>(s.moe_ff), E = static_cast<size_t>(s.n_experts);
    const size_t vd = static_cast<size_t>(s.nv * s.dv), kd = static_cast<size_t>(s.nk * s.dk);
    // (X's widest vector: the MTP layer's input, 2 H, as wide as a mixer's output only in Qwen3.6's shape)
    const size_t cw = std::max(vd, static_cast<size_t>(s.nh * s.hd)), mc = std::max(2 * H, cw);
    const size_t kv_row = static_cast<size_t>(s.nkv * s.hd);
    // every token buffer for max_tokens (at least 2: verifying pairs); the experts' for the picks' union of that many
    // tokens (top_k slots a token) + the shared one, a row a token each
    const int mt = std::clamp(s.max_tokens, 2, kMaxTokens);
    const size_t nt = static_cast<size_t>(mt), xs = static_cast<size_t>(mt) * K + 1;
    const bool ok =
        p.alloc(nt * H * f, p.d_x, err) && p.alloc(nt * H * f, p.d_h, err) && p.alloc(nt * mc, p.d_X, err) &&
        p.alloc(nt * mc / 16 * f, p.d_XS, err) && p.alloc(nt * mc / 16 * f, p.d_XB, err) &&
        p.alloc(nt * (2 * kd + vd) * f, p.d_proj, err) && p.alloc(nt * vd * f, p.d_z, err) &&
        p.alloc(nt * 2 * static_cast<size_t>(s.nv) * f, p.d_ab, err) && p.alloc(nt * cw * f, p.d_core, err) &&
        p.alloc(nt * static_cast<size_t>(s.nh * 2 * s.hd) * f, p.d_qg, err) && p.alloc(nt * kv_row * f, p.d_kb, err) &&
        p.alloc(nt * kv_row * f, p.d_vb, err) && p.alloc(nt * H * f, p.d_mix, err) &&
        p.alloc(static_cast<size_t>(s.nh) * static_cast<size_t>(s.max_ctx) * f, p.d_scores, err) &&
        p.alloc(nt * (E + 1) * f, p.d_rlog, err) && p.alloc(nt * E * f, p.d_glog, err) &&
        p.alloc(nt * E * f, p.d_glog2, err) && p.alloc(nt * 16 * f, p.d_gids2, err) &&
        p.alloc(nt * static_cast<size_t>(s.nh) * static_cast<size_t>(p.nb_max) * 256 * f, p.d_po, err) &&
        p.alloc(nt * static_cast<size_t>(s.nh) * static_cast<size_t>(p.nb_max) * f, p.d_pm, err) &&
        p.alloc(nt * static_cast<size_t>(s.nh) * static_cast<size_t>(p.nb_max) * f, p.d_pl, err) &&
        p.alloc(nt * K * f, p.d_ids, err) && p.alloc(nt * static_cast<size_t>(s.guesses) * f, p.d_gids, err) &&
        p.alloc(nt * K * f, p.d_w, err) && p.alloc(xs * nt * 2 * ff * f, p.d_gu, err) &&
        p.alloc(xs * nt * ff, p.d_x2, err) && p.alloc(xs * nt * ff / 16 * f, p.d_x2s, err) &&
        p.alloc(xs * nt * ff / 16 * f, p.d_x2b, err) && p.alloc(xs * nt * H * f, p.d_eo, err) &&
        p.alloc(Impl::kJobs * 32, p.d_jobs, err) && p.alloc(xs * nt * f, p.d_ymap, err) &&
        p.alloc(nt * static_cast<size_t>(s.vocab) * f, p.d_logits, err) &&
        p.alloc(static_cast<size_t>(s.half) * f, p.d_freq, err);
    if (!ok) return false;
    auto *ym = reinterpret_cast<int32_t *>(p.d_ymap.host);
    for (size_t i = 0; i < xs * nt; ++i) ym[i] = static_cast<int32_t>(i);
    std::memcpy(p.d_freq.host, inv_freq, static_cast<size_t>(s.half) * f);
    std::memset(p.d_ids.host, 0, nt * K * f);   // tag 0: no route yet (the first is 1)
    std::memset(p.d_gids.host, 0, nt * static_cast<size_t>(s.guesses) * f);
    std::memset(p.d_gids2.host, 0, nt * 16 * f);
    return true;
}

bool Gpu8::setup_verify(std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    if (!p.q1) {
        err = "GPU: verifying before setup_decode";
        return false;
    }
    if (s.dk != 128 || s.dv % 64) {   // gdn_delta32 carries the state between the tokens
        err = "GPU: verifying needs dk = 128";
        return false;
    }
    const int mt = std::clamp(s.max_tokens, 2, kMaxTokens);
    for (int t = 2; t <= mt; ++t) {
        const std::string opts = "-cl-std=CL3.0 -cl-mad-enable -DT=" + std::to_string(t) + (p.ag == 4 ? " -DAG=4" : "");
        if (!p.progT[t].build_cached(p.dev, kSource, opts, p.cache_dir, err)) return false;
        for (const char *n : {"gemvs", "gemvs1", "swiglu", "norm_q", "mtp_in_q", "bf16_rows", "gdn_conv", "gdn_qk", "gdn_gates",
                              "gdn_delta32", "gdn_gnorm_q", "head_norm_rope", "kv_store", "attn_scores", "attn_softmax",
                              "attn_values", "attn_part", "attn_part8", "kv_store8", "attn_merge", "attn_gate_q", "route", "guess_top", "moe_combine",
                              "mlp_add"}) {
            cl_kernel k = p.progT[t].kernel(n, err);
            if (!k) return false;
            p.kTok[t][n] = k;
        }
    }
    return true;
}

bool Gpu8::set_tokens(int t) {
    Impl &p = *p_;
    if (t < 1 || t > kMaxTokens || (t > 1 && p.kTok[t].empty())) return false;
    p.T = t;
    return true;
}

int Gpu8::tokens() const { return p_->T; }

void Gpu8::set_rope_shift(int shift) { p_->rope_shift = shift; }

bool Gpu8::dec_front(const DecodeLayer &L, const DecodeLayer *next, int pos, const std::vector<uint8_t *> &kv,
                     std::string &err, const DecodeLayer *next2) {
    Impl &p = *p_;
    const auto &s = p.ds;
    const Impl::Piece &P = p.params;
    const int H = s.hidden, E = s.n_experts, T = p.T;
    const size_t tg = T > 1 ? static_cast<size_t>(T) : 0;   // the tokens' dimension (one token: a 1-D launch as ever)
    if (!p.go("norm_q", 256, tg, 256, 0, err, p.d_x.mem, p.d_x.mem, 0, P.mem, p.at(P, L.in_norm, 4), p.d_h.mem, H, s.eps,
              p.d_X.mem, p.d_XS.mem, p.d_XB.mem))
        return false;
    if (!L.attn) {
        const int kd = s.nk * s.dk, vd = s.nv * s.dv, conv_dim = 2 * kd + vd;
        const Impl::Piece *cs = p.find(L.conv_state), *gs = p.find(L.gdn_state);
        if (!cs || !gs) {
            err = "GPU: a state outside shared memory";
            return false;
        }
        // the states after the first token: their backups (two tokens), or nowhere (one: the argument unused)
        const Impl::Piece *cb = T > 1 ? p.find(L.conv_bk) : cs, *gb = T > 1 ? p.find(L.gdn_bk) : gs;
        if (!cb || !gb) {
            err = "GPU: no state backups for verifying";
            return false;
        }
        const int cbo = T > 1 ? p.at(*cb, L.conv_bk, 4) : 0, gbo = T > 1 ? p.at(*gb, L.gdn_bk, 4) : 0;
        const bool d32 = s.dk == 128 && s.dv % 64 == 0 && (!p.gdn_old || T > 1);
        if (!p.gemv1(L.qkv, p.d_proj, err) || !p.gemv1(L.z, p.d_z, err) ||
            !p.go("bf16_rows", static_cast<size_t>(2 * s.nv) * 64, 0, 64, 0, err, P.mem, p.at(P, L.ab, 2), p.d_h.mem, H,
                  p.d_ab.mem, 0, 2 * s.nv) ||
            !p.go("gdn_conv", static_cast<size_t>(conv_dim), 0, 0, 0, err, p.d_proj.mem, P.mem, p.at(P, L.conv, 4), cs->mem,
                  p.at(*cs, L.conv_state, 4), s.conv_k, conv_dim, cb->mem, cbo, s.bk_conv) ||
            !p.go("gdn_qk", static_cast<size_t>(2 * s.nk * s.dk), tg, static_cast<size_t>(s.dk), 0, err, p.d_proj.mem,
                  s.dk, kd, 1.0f / std::sqrt(static_cast<float>(s.dk)), conv_dim) ||
            !p.go("gdn_gates", static_cast<size_t>(s.nv), tg, 0, 0, err, p.d_ab.mem, s.nv, P.mem, p.at(P, L.A_log, 4),
                  p.at(P, L.dt_bias, 4)) ||
            !(d32 ? p.go("gdn_delta32", static_cast<size_t>(s.nv * s.dv * 4), 0, 256, 0, err, gs->mem,
                         p.at(*gs, L.gdn_state, 4), p.d_proj.mem, p.d_ab.mem, p.d_core.mem, s.nk, s.nv, s.dk, s.dv,
                         conv_dim, gb->mem, gbo, s.bk_gdn)
                  : p.go("gdn_delta", static_cast<size_t>(s.nv * s.dv * 4), 0, static_cast<size_t>(s.dv * 4), 0, err,
                         gs->mem, p.at(*gs, L.gdn_state, 4), p.d_proj.mem, p.d_ab.mem, p.d_core.mem, s.nk, s.nv, s.dk,
                         s.dv)) ||
            !p.go("gdn_gnorm_q", static_cast<size_t>(s.nv * s.dv), tg, static_cast<size_t>(s.dv), 0, err, p.d_core.mem,
                  p.d_z.mem, P.mem, p.at(P, L.gnorm, 4), s.eps, p.d_X.mem, p.d_XS.mem, p.d_XB.mem) ||
            !p.gemv1(L.out, p.d_mix, err))
            return false;
    } else {
        const int kv_row = s.nkv * s.hd, group = s.nh / s.nkv, layer = L.index, qrow = s.nh * 2 * s.hd;
        const int cw = std::max(s.nv * s.dv, s.nh * s.hd);
        if (static_cast<size_t>((pos + T - 1) / s.kv_chunk) >= kv.size()) {
            err = "GPU: the KV cache has no chunk for this position";
            return false;
        }
        if (!p.gemv1(L.q, p.d_qg, err) || !p.gemv1(L.k, p.d_kb, err) || !p.gemv1(L.v, p.d_vb, err) ||
            !p.go("head_norm_rope", static_cast<size_t>(s.nh * s.hd), tg, static_cast<size_t>(s.hd), 0, err, p.d_qg.mem,
                  2 * s.hd, P.mem, p.at(P, L.q_norm, 4), s.eps, s.half, pos + p.rope_shift, p.d_freq.mem, qrow) ||
            !p.go("head_norm_rope", static_cast<size_t>(s.nkv * s.hd), tg, static_cast<size_t>(s.hd), 0, err, p.d_kb.mem,
                  s.hd, P.mem, p.at(P, L.k_norm, 4), s.eps, s.half, pos + p.rope_shift, p.d_freq.mem, kv_row))
            return false;
        // each token's K and V into its position's chunk (two tokens may straddle chunks)
        for (int t = 0; t < T; ++t) {
            const int pt = pos + t;
            const Impl::Piece *cp = p.find(kv[static_cast<size_t>(pt / s.kv_chunk)]);
            if (!cp || cp->host != kv[static_cast<size_t>(pt / s.kv_chunk)]) {
                err = "GPU: a KV chunk outside shared memory";
                return false;
            }
            const int in_chunk = pt % s.kv_chunk;
            const int ko = ((0 * s.n_attn + layer) * s.kv_chunk + in_chunk) * kv_row;
            const int vo = ((1 * s.n_attn + layer) * s.kv_chunk + in_chunk) * kv_row;
            if (s.kv_i8) {   // (int8: the values in bytes, then the scales: Model::kv_scale)
                const int s0 = 2 * s.n_attn * s.kv_chunk * kv_row / 4;
                const int ksi = s0 + ((0 * s.n_attn + layer) * s.kv_chunk + in_chunk) * s.nkv;
                const int vsi = s0 + ((1 * s.n_attn + layer) * s.kv_chunk + in_chunk) * s.nkv;
                if (!p.go("kv_store8", static_cast<size_t>(kv_row), 0, static_cast<size_t>(s.hd), 0, err, p.d_kb.mem,
                          p.d_vb.mem, cp->mem, ko, vo, cp->mem, ksi, vsi, t * kv_row))
                    return false;
            } else if (!p.go("kv_store", static_cast<size_t>(kv_row), 0, 0, 0, err, p.d_kb.mem, p.d_vb.mem, cp->mem, ko,
                             vo, t * kv_row)) {
                return false;
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(s.hd));
        if (s.kv_i8 && !(group == p.ag && s.kv_chunk % 128 == 0)) {
            err = "GPU: an int8 KV cache for a shape flash decoding does not take";
            return false;
        }
        if (group == p.ag && s.kv_chunk % 128 == 0 && (!p.attn_old || s.kv_i8)) {
            // flash decoding: the blocks of 128 positions up to the last token's (every token in each), chunk by
            // chunk, then merged
            const int nb = (pos + T + 127) / 128;
            for (int c = 0, p0 = 0; p0 < pos + T; ++c, p0 += s.kv_chunk) {
                const Impl::Piece *ck = p.find(kv[static_cast<size_t>(c)]);
                const int n = std::min(pos + T - p0, s.kv_chunk);
                if (!ck) return false;
                const size_t groups = static_cast<size_t>((n + 127) / 128) * 256;
                for (int tb = 0; tb < T; tb += 2) {   // two tokens a launch (the kernel's local memory)
                    if (s.kv_i8) {
                        const int s0 = 2 * s.n_attn * s.kv_chunk * kv_row / 4;
                        if (!p.go("attn_part8", groups, static_cast<size_t>(s.nkv), 256, 1, err, p.d_qg.mem, qrow, ck->mem,
                                  layer * s.kv_chunk * kv_row, (s.n_attn + layer) * s.kv_chunk * kv_row, kv_row, ck->mem,
                                  s0 + layer * s.kv_chunk * s.nkv, s0 + (s.n_attn + layer) * s.kv_chunk * s.nkv, p0, p0 / 128,
                                  pos + 1, scale, p.d_po.mem, p.d_pm.mem, p.d_pl.mem, nb, s.nh, tb))
                            return false;
                    } else if (!p.go("attn_part", groups, static_cast<size_t>(s.nkv), 256, 1, err, p.d_qg.mem, qrow,
                                     ck->mem, layer * s.kv_chunk * kv_row, (s.n_attn + layer) * s.kv_chunk * kv_row, kv_row,
                                     p0, p0 / 128, pos + 1, scale, p.d_po.mem, p.d_pm.mem, p.d_pl.mem, nb, s.nh, tb)) {
                        return false;
                    }
                }
            }
            if (!p.go("attn_merge", static_cast<size_t>(s.nh) * 256, tg, 256, 0, err, p.d_po.mem, p.d_pm.mem,
                      p.d_pl.mem, nb, p.d_core.mem, cw, s.nh))
                return false;
        } else
        // each token's attention over the positions up to its own (one scores buffer, in turn)
        for (int t = 0; t < T; ++t) {
            const int np = pos + t + 1;
            for (int c = 0, p0 = 0; p0 < np; ++c, p0 += s.kv_chunk) {
                const Impl::Piece *ck = p.find(kv[static_cast<size_t>(c)]);
                const int p1 = std::min(np, p0 + s.kv_chunk), kb = layer * s.kv_chunk * kv_row;
                if (!ck || !p.go("attn_scores", static_cast<size_t>(p1 - p0), static_cast<size_t>(s.nh), 0, 0, err,
                                 p.d_qg.mem, ck->mem, kb, kv_row, s.hd, group, p0, p0, p.d_scores.mem, np, scale,
                                 t * qrow))
                    return false;
            }
            if (!p.go("attn_softmax", static_cast<size_t>(s.nh) * 256, 0, 256, 0, err, p.d_scores.mem, np)) return false;
            for (int c = 0, p0 = 0; p0 < np; ++c, p0 += s.kv_chunk) {
                const Impl::Piece *ck = p.find(kv[static_cast<size_t>(c)]);
                const int p1 = std::min(np, p0 + s.kv_chunk), vb = (s.n_attn + layer) * s.kv_chunk * kv_row;
                if (!p.go("attn_values", static_cast<size_t>(s.hd), static_cast<size_t>(s.nh), 0, 0, err, p.d_scores.mem,
                          np, ck->mem, vb, kv_row, s.hd, group, p0, p1, p0, p.d_core.mem, c == 0 ? 1 : 0, t * cw))
                    return false;
            }
        }
        if (!p.go("attn_gate_q", static_cast<size_t>(s.nh * s.hd / 16), tg, 0, 0, err, p.d_core.mem, p.d_qg.mem, s.hd,
                  p.d_X.mem, p.d_XS.mem, p.d_XB.mem, cw, qrow) ||
            !p.gemv1(L.o, p.d_mix, err))
            return false;
    }
    if (!p.go("norm_q", 256, tg, 256, 0, err, p.d_x.mem, p.d_mix.mem, 1, P.mem, p.at(P, L.post_norm, 4), p.d_h.mem, H,
              s.eps, p.d_X.mem, p.d_XS.mem, p.d_XB.mem))
        return false;
    if (E == 0) {   // a dense model: no router (its MLP: dec_experts' shared slot, dec_mlp_add)
        p.cl->clFlush(p.q1);
        return true;
    }
    if (!p.go("bf16_rows", static_cast<size_t>(E + 1) * 64, 0, 64, 0, err, P.mem, p.at(P, L.router, 2), p.d_h.mem, H,
              p.d_rlog.mem, 0, E + 1))
        return false;
    if (next && !p.go("bf16_rows", static_cast<size_t>(E) * 64, 0, 64, 0, err, P.mem, p.at(P, next->router, 2),
                      p.d_h.mem, H, p.d_glog.mem, 0, E))
        return false;
    const bool g2 = next2 && s.guesses2 > 0;
    if (g2 && !p.go("bf16_rows", static_cast<size_t>(E) * 64, 0, 64, 0, err, P.mem, p.at(P, next2->router, 2),
                    p.d_h.mem, H, p.d_glog2.mem, 0, E))
        return false;
    p.seq = p.seq % 0x7fffff + 1;   // the tag: 1 .. 2^23 - 1, never 0
    p.guessed = next != nullptr;
    p.guessed2 = g2;
    if (!p.go("route", static_cast<size_t>(E), tg, static_cast<size_t>(E), 0, err, p.d_rlog.mem, p.d_glog.mem,
              next ? 1 : 0, s.top_k, s.guesses, p.d_ids.mem, p.d_w.mem, p.d_gids.mem, p.seq))
        return false;
    if (g2 && !p.go("guess_top", static_cast<size_t>(E), tg, static_cast<size_t>(E), 0, err, p.d_glog2.mem, s.guesses2,
                    p.d_gids2.mem, p.seq))
        return false;
    p.cl->clFlush(p.q1);
    return true;
}

bool Gpu8::dec_route(int *ids, int *gids, std::string &err, int *gids2) {
    Impl &p = *p_;
    const auto *vi = reinterpret_cast<const volatile int32_t *>(p.d_ids.host);
    const auto *vg = reinterpret_cast<const volatile int32_t *>(p.d_gids.host);
    const int K = p.ds.top_k * p.T, G = p.guessed ? p.ds.guesses * p.T : 0;   // every token's, one after another
    const double t0 = now_seconds();
    if (p.wait_sleep) p.cl->clFinish(p.q1);   // GEODE_GPU8_WAIT=1: sleep in the driver instead of polling (measuring)
    // every word this route's (route kernel): the last one first, then the rest
    for (int spins = 0;; ++spins) {
        int have = 0;
        for (int k = 0; k < K; ++k) {
            const int32_t v = vi[k];
            if ((v >> 8) == p.seq) ids[have++] = v & 0xff;
            else break;
        }
        int gh = 0;
        if (have == K)
            for (int k = 0; k < G; ++k) {
                const int32_t v = vg[k];
                if ((v >> 8) == p.seq) gids[gh++] = v & 0xff;
                else break;
            }
        int g2h = 0;
        const int G2 = gids2 && p.guessed2 ? p.ds.guesses2 * p.T : 0;
        if (have == K && gh == G) {
            const auto *v2 = reinterpret_cast<const volatile int32_t *>(p.d_gids2.host);
            for (int k = 0; k < G2; ++k) {
                const int32_t v = v2[k];
                if ((v >> 8) == p.seq) gids2[g2h++] = v & 0xff;
                else break;
            }
        }
        if (have == K && gh == G && g2h == G2) break;
        if (spins % 4096 == 4095 && now_seconds() - t0 > 5.0) {
            err = "GPU: the route never came";
            return false;
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
}

bool Gpu8::dec_experts(const DecodeLayer &L, const QView *gu, const QView *dn, const std::vector<int> &ks, bool shared,
                       std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    const int H = s.hidden, K = s.top_k, ff = s.moe_ff, T = p.T;
    const int S = T == 1 ? K : T * K;   // the shared expert's slot (after the picks' union)
    if (ks.empty() && !shared) return true;
    // these slots' gate_up (and the shared expert's gate and up) in one launch, slot u's tokens into rows u T + t of
    // d_gu (the shared one's slot S, gate then up); then silu(gate) * up of every slot (the others' rows are redone
    // later, the same values); then their downs in one launch, slot u's token t into row u T + t of d_eo
    std::vector<Impl::Mat> ms;
    for (int u : ks) ms.push_back({&gu[u], 0, u * T * 2 * ff});
    if (shared) {
        ms.push_back({&L.sh_gate, 0, S * T * 2 * ff});
        ms.push_back({&L.sh_up, 0, S * T * 2 * ff + ff});
    }
    if (!p.gemv_many(ms, p.d_X, p.d_XS, p.d_XB, p.d_gu, H, 2 * ff, err)) return false;
    if (!p.go("swiglu", static_cast<size_t>((S + 1) * T), static_cast<size_t>(ff / 16), 0, 0, err, p.d_gu.mem, ff,
              p.d_x2.mem, p.d_x2s.mem, p.d_x2b.mem))
        return false;
    ms.clear();
    for (int u : ks) ms.push_back({&dn[u], u, 0});
    if (shared) ms.push_back({&L.sh_down, S, 0});
    if (!p.gemv_many(ms, p.d_x2, p.d_x2s, p.d_x2b, p.d_eo, ff, H, err)) return false;
    p.cl->clFlush(p.q1);   // started while the CPU waits for the rest
    return true;
}

bool Gpu8::dec_combine(const uint32_t *skip, const uint64_t *um, bool renorm, std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    const int T = p.T;
    cl_uint8 skips{};
    cl_ulong8 ums{};
    for (int t = 0; t < T; ++t) {
        skips.s[t] = skip[t];
        ums.s[t] = um[t];
    }
    return p.go("moe_combine", static_cast<size_t>(s.hidden), T > 1 ? static_cast<size_t>(T) : 0, 0, 0, err, p.d_x.mem,
                p.d_eo.mem, p.d_w.mem, skips, renorm ? 1 : 0, p.d_rlog.mem, s.n_experts, s.hidden, s.top_k, ums,
                T == 1 ? s.top_k : T * s.top_k);
}

bool Gpu8::dec_mlp_add(std::string &err) {
    Impl &p = *p_;
    const int T = p.T;
    return p.go("mlp_add", static_cast<size_t>(p.ds.hidden), T > 1 ? static_cast<size_t>(T) : 0, 0, 0, err, p.d_x.mem,
                p.d_eo.mem, p.ds.hidden);
}

bool Gpu8::dec_head(const float *final_norm, const QView &lm_head, float *logits, std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    const bool ok = p.go("norm_q", 256, p.T > 1 ? static_cast<size_t>(p.T) : 0, 256, 0, err, p.d_x.mem, p.d_x.mem, 0,
                         p.params.mem, p.at(p.params, final_norm, 4),
                         p.d_h.mem, s.hidden, s.eps, p.d_X.mem, p.d_XS.mem, p.d_XB.mem) &&
                    (!logits || p.gemv1(lm_head, p.d_logits, err));
    if (ok && logits && std::getenv("GEODE_GPU8_HEAD2")) {   // measuring: lm_head again, right after itself
        p.label_prefix = "again: ";
        p.gemv1(lm_head, p.d_logits, err);
        p.label_prefix.clear();
    }
    const cl_int e = p.cl->clFinish(p.q1);
    p.jcur = 0;
    if (p.dprof) p.collect();
    if (!ok) return false;
    if (e != CL_SUCCESS) {
        err = std::string("GPU: finish: ") + cl::error_name(e);
        return false;
    }
    // (lm_head_.rows a token: the vocabulary, or a draft head's subset)
    if (logits) std::memcpy(logits, p.d_logits.host, static_cast<size_t>(p.T) * static_cast<size_t>(lm_head.rows) * 4);
    return true;
}

bool Gpu8::pf_ready(std::string &err) { return p_->pf_setup(err); }

std::string Gpu8::pf_profile() const {
    std::string s;
    char b[128];
    for (const auto &[name, t] : p_->pf_times) {
        std::snprintf(b, sizeof b, "%s%s %.0f ms (%lld)", s.empty() ? "" : ", ", name.c_str(), t.first * 1e3,
                      static_cast<long long>(t.second));
        s += b;
    }
    return s;
}

bool Gpu8::pf_gdn(const DecodeLayer &L, int n, const float *h, float *mix, std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    if (!p.pf_setup(err)) return false;
    if (n <= 0 || n > p.tokens || s.dk != 128 || s.dv % 64 || s.conv_k > 9) {
        err = "GPU: a prompt's Gated DeltaNet it does not take";
        return false;
    }
    const Impl::Piece &P = p.params;
    const Impl::Piece *cs = p.find(L.conv_state), *gs = p.find(L.gdn_state);
    if (!cs || !gs) {
        err = "GPU: a state outside shared memory";
        return false;
    }
    const int H = s.hidden, kd = s.nk * s.dk, vd = s.nv * s.dv, conv_dim = 2 * kd + vd;
    const auto nt = static_cast<size_t>(n);
    p.pf_jcur = 0;
    // in parts, the host's products beside the GPU's mixer (Impl::pipe_bounds)
    if (const auto b = p.pipe_bounds(n);
        !b.empty() && p.pipe_fits(b, {static_cast<size_t>(H) * 4, static_cast<size_t>(conv_dim) * 4,
                                      static_cast<size_t>(vd) * 4, static_cast<size_t>(2 * s.nv) * 4,
                                      static_cast<size_t>(4 * vd) * 4})) {
        const int parts = static_cast<int>(b.size()) - 1;
        std::memcpy(p.in.host, h, nt * static_cast<size_t>(H) * 4);   // the a/b product reads the input here
        auto proj = [&](int c) {
            const int a = b[static_cast<size_t>(c)], m = b[static_cast<size_t>(c) + 1] - a;
            const QView *ms[2] = {&L.qkv, &L.z};
            float *outs[2] = {reinterpret_cast<float *>(p.pf_proj.host) + static_cast<size_t>(a) * conv_dim,
                              reinterpret_cast<float *>(p.pf_z.host) + static_cast<size_t>(a) * vd};
            return p.host_gemm(ms, 2, m, h + static_cast<size_t>(a) * H, H, outs, true);
        };
        auto out = [&](int c) {
            const int a = b[static_cast<size_t>(c)], m = b[static_cast<size_t>(c) + 1] - a;
            const QView *ms[1] = {&L.out};
            float *outs[1] = {mix + static_cast<size_t>(a) * H};
            return p.host_gemm(ms, 1, m, reinterpret_cast<const float *>(p.pf_core.host) + static_cast<size_t>(a) * vd,
                               vd, outs, true);
        };
        auto core = [&](int c) {   // part c's mixer on the GPU, enqueued and started
            const auto a = static_cast<size_t>(b[static_cast<size_t>(c)]);
            const int m = b[static_cast<size_t>(c) + 1] - static_cast<int>(a);
            const auto mt = static_cast<size_t>(m);
            cl_mem vin, vproj, vz, vab, vc4, vcore;
            if (!p.view(p.in, a, mt, static_cast<size_t>(H) * 4, vin, err) ||
                !p.view(p.pf_proj, a, mt, static_cast<size_t>(conv_dim) * 4, vproj, err) ||
                !p.view(p.pf_z, a, mt, static_cast<size_t>(vd) * 4, vz, err) ||
                !p.view(p.pf_ab, a, mt, static_cast<size_t>(2 * s.nv) * 4, vab, err) ||
                !p.view(p.pf_core4, a, mt, static_cast<size_t>(4 * vd) * 4, vc4, err) ||
                !p.view(p.pf_core, a, mt, static_cast<size_t>(vd) * 4, vcore, err))
                return false;
            const bool ok =
                p.bf16(P.mem, p.at(P, L.ab, 2), H, vab, 2 * s.nv, 2 * s.nv, m, err, vin) &&
                p.pgo("pf_gdn_conv", static_cast<size_t>(conv_dim), 0, 0, 0, err, vproj, P.mem, p.at(P, L.conv, 4),
                      cs->mem, p.at(*cs, L.conv_state, 4), s.conv_k, conv_dim, m) &&
                p.pgo("pf_gdn_qk_gates", static_cast<size_t>(2 * s.nk * s.dk), mt, static_cast<size_t>(s.dk), 1, err,
                      vproj, s.dk, kd, 1.0f / std::sqrt(static_cast<float>(s.dk)), conv_dim, vab, s.nv, P.mem,
                      p.at(P, L.A_log, 4), p.at(P, L.dt_bias, 4)) &&
                p.pgo("pf_gdn_delta32", static_cast<size_t>(s.nv * s.dv * 4), 0, 256, 0, err, gs->mem,
                      p.at(*gs, L.gdn_state, 4), vproj, vab, vc4, s.nk, s.nv, s.dk, s.dv, conv_dim, m) &&
                p.pgo("pf_gdn_gnorm_q", static_cast<size_t>(s.nv * s.dv), mt, static_cast<size_t>(s.dv), 1, err, vc4,
                      vcore, vz, P.mem, p.at(P, L.gnorm, 4), s.eps, p.x.mem, p.xs.mem, p.xb.mem);
            p.cl->clFlush(p.dev.queue);
            return ok;
        };
        // the host: part 0's inputs; then, while the GPU runs part c: part c + 1's inputs, and once part c's mixer
        // is done and part c + 1's started, part c's output
        bool ok = proj(0) && core(0);
        for (int c = 0; ok && c < parts; ++c) {
            if (c + 1 < parts) ok = proj(c + 1);
            ok = ok && p.finish(err);
            if (ok && c + 1 < parts) ok = core(c + 1);
            ok = ok && out(c);
        }
        if (!ok) {
            std::string e2;
            p.finish(e2);
        }
        p.drop_subs();
        return ok;
    }
    // the projections elsewhere when it takes them (the NPU): only the bf16 a/b product reads the input here
    bool host = false;
    if (p.host_gemm) {
        const QView *ms[2] = {&L.qkv, &L.z};
        float *outs[2] = {reinterpret_cast<float *>(p.pf_proj.host), reinterpret_cast<float *>(p.pf_z.host)};
        host = p.host_gemm(ms, 2, n, h, H, outs, false);
    }
    if (host) p.pf_maps(h, n, H);
    if (!(host || (p.pf_in(h, n, H, err) && p.pf_gemm(L.qkv, n, p.pf_proj, err) && p.pf_gemm(L.z, n, p.pf_z, err))) ||
        !p.bf16(P.mem, p.at(P, L.ab, 2), H, p.pf_ab.mem, 2 * s.nv, 2 * s.nv, n, err) ||
        !p.pgo("pf_gdn_conv", static_cast<size_t>(conv_dim), 0, 0, 0, err, p.pf_proj.mem, P.mem, p.at(P, L.conv, 4),
               cs->mem, p.at(*cs, L.conv_state, 4), s.conv_k, conv_dim, n) ||
        !p.pgo("pf_gdn_qk_gates", static_cast<size_t>(2 * s.nk * s.dk), nt, static_cast<size_t>(s.dk), 1, err,
               p.pf_proj.mem, s.dk, kd, 1.0f / std::sqrt(static_cast<float>(s.dk)), conv_dim, p.pf_ab.mem, s.nv, P.mem,
               p.at(P, L.A_log, 4), p.at(P, L.dt_bias, 4)) ||
        !p.pgo("pf_gdn_delta32", static_cast<size_t>(s.nv * s.dv * 4), 0, 256, 0, err, gs->mem,
               p.at(*gs, L.gdn_state, 4), p.pf_proj.mem, p.pf_ab.mem, p.pf_core4.mem, s.nk, s.nv, s.dk, s.dv, conv_dim, n) ||
        !p.pgo("pf_gdn_gnorm_q", static_cast<size_t>(s.nv * s.dv), nt, static_cast<size_t>(s.dv), 1, err,
               p.pf_core4.mem, p.pf_core.mem, p.pf_z.mem, P.mem, p.at(P, L.gnorm, 4), s.eps, p.x.mem, p.xs.mem,
               p.xb.mem))
        return false;
    if (host) {   // the out projection there too, from the gated norm's floats
        if (!p.finish(err)) return false;
        const QView *ms[1] = {&L.out};
        float *outs[1] = {mix};
        if (p.host_gemm(ms, 1, n, reinterpret_cast<const float *>(p.pf_core.host), vd, outs, false)) return true;
    }
    if (!p.pf_gemm(L.out, n, p.pf_mix, err) || !p.finish(err)) return false;
    std::memcpy(mix, p.pf_mix.host, nt * static_cast<size_t>(H) * 4);
    return true;
}

bool Gpu8::pf_router(const DecodeLayer &L, int n, const float *h, const float **logits, std::string &err) {
    Impl &p = *p_;
    if (!p.pf_setup(err)) return false;
    const int H = p.ds.hidden, rows = p.ds.n_experts + 1;
    if (n <= 0 || n > p.tokens || H % 64) {
        err = "GPU: a prompt's router it does not take";
        return false;
    }
    std::memcpy(p.in.host, h, static_cast<size_t>(n) * static_cast<size_t>(H) * 4);
    if (!p.bf16(p.params.mem, p.at(p.params, L.router, 2), H, p.pf_lg.mem, rows, rows, n, err) ||
        !p.finish(err))
        return false;
    *logits = reinterpret_cast<const float *>(p.pf_lg.host);
    return true;
}

bool Gpu8::pf_attn(const DecodeLayer &L, int n, int kv_pos, const int32_t *pos3, const std::vector<uint8_t *> &kv,
                   const float *h, float *mix, std::string &err) {
    Impl &p = *p_;
    const auto &s = p.ds;
    if (!p.pf_setup(err)) return false;
    if (n <= 0 || n > p.tokens || !s.kv_i8 || s.hd != 256 || s.nh != 8 * s.nkv) {
        err = "GPU: a prompt's attention it does not take";
        return false;
    }
    const Impl::Piece &P = p.params;
    const int H = s.hidden, hd = s.hd, kv_row = s.nkv * hd, qrow = s.nh * 2 * hd, layer = L.index;
    const auto nt = static_cast<size_t>(n);
    if (static_cast<size_t>((kv_pos + n - 1) / s.kv_chunk) >= kv.size()) {
        err = "GPU: the KV cache has no chunk for this position";
        return false;
    }
    std::memcpy(p.pf_pos.host, pos3, nt * 3 * 4);
    p.pf_jcur = 0;
    // in parts, the host's products beside the GPU's attention (Impl::pipe_bounds; as pf_gdn): each part's heads'
    // norm and RoPE, its K and V into the cache, the attention over every position up to its last, the output gate
    if (const auto b = p.pipe_bounds(n);
        !b.empty() && p.pipe_fits(b, {static_cast<size_t>(H) * 4, static_cast<size_t>(qrow) * 4,
                                      static_cast<size_t>(kv_row) * 4, 12, static_cast<size_t>(s.nh) * 4,
                                      static_cast<size_t>(s.nh * hd) * 4})) {
        const int parts = static_cast<int>(b.size()) - 1;
        const int s0 = 2 * s.n_attn * s.kv_chunk * kv_row / 4;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        auto proj = [&](int c) {
            const int a = b[static_cast<size_t>(c)], m = b[static_cast<size_t>(c) + 1] - a;
            const QView *ms[3] = {&L.q, &L.k, &L.v};
            float *outs[3] = {reinterpret_cast<float *>(p.pf_qg.host) + static_cast<size_t>(a) * qrow,
                              reinterpret_cast<float *>(p.pf_kb.host) + static_cast<size_t>(a) * kv_row,
                              reinterpret_cast<float *>(p.pf_vb.host) + static_cast<size_t>(a) * kv_row};
            return p.host_gemm(ms, 3, m, h + static_cast<size_t>(a) * H, H, outs, true);
        };
        auto out = [&](int c) {
            const int a = b[static_cast<size_t>(c)], m = b[static_cast<size_t>(c) + 1] - a;
            const QView *ms[1] = {&L.o};
            float *outs[1] = {mix + static_cast<size_t>(a) * H};
            return p.host_gemm(ms, 1, m,
                               reinterpret_cast<const float *>(p.pf_core.host) + static_cast<size_t>(a) * s.nh * hd,
                               s.nh * hd, outs, true);
        };
        auto core = [&](int c) {
            const auto a = static_cast<size_t>(b[static_cast<size_t>(c)]);
            const int m = b[static_cast<size_t>(c) + 1] - static_cast<int>(a);
            const auto mt = static_cast<size_t>(m);
            const int kp = kv_pos + static_cast<int>(a);
            cl_mem qg, kb, vb, pos, pm, pl, po, core;
            if (!p.view(p.pf_qg, a, mt, static_cast<size_t>(qrow) * 4, qg, err) ||
                !p.view(p.pf_kb, a, mt, static_cast<size_t>(kv_row) * 4, kb, err) ||
                !p.view(p.pf_vb, a, mt, static_cast<size_t>(kv_row) * 4, vb, err) ||
                !p.view(p.pf_pos, a, mt, 12, pos, err) ||
                !p.view(p.pf_pm, a, mt, static_cast<size_t>(s.nh) * 4, pm, err) ||
                !p.view(p.pf_pl, a, mt, static_cast<size_t>(s.nh) * 4, pl, err) ||
                !p.view(p.pf_po, a, mt, static_cast<size_t>(s.nh * hd) * 4, po, err) ||
                !p.view(p.pf_core, a, mt, static_cast<size_t>(s.nh * hd) * 4, core, err))
                return false;
            if (!p.pgo("pf_head_norm_rope", static_cast<size_t>(s.nh * hd) * mt, 0, static_cast<size_t>(hd), 0, err, qg,
                       2 * hd, P.mem, p.at(P, L.q_norm, 4), s.eps, s.half, pos, p.d_freq.mem, qrow, s.nh) ||
                !p.pgo("pf_head_norm_rope", static_cast<size_t>(s.nkv * hd) * mt, 0, static_cast<size_t>(hd), 0, err, kb,
                       hd, P.mem, p.at(P, L.k_norm, 4), s.eps, s.half, pos, p.d_freq.mem, kv_row, s.nkv))
                return false;
            for (int t0 = 0; t0 < m;) {
                const int pt = kp + t0, ch = pt / s.kv_chunk, in_chunk = pt % s.kv_chunk;
                const int cnt = std::min(m - t0, s.kv_chunk - in_chunk);
                const Impl::Piece *cp = p.find(kv[static_cast<size_t>(ch)]);
                if (!cp || cp->host != kv[static_cast<size_t>(ch)]) {
                    err = "GPU: a KV chunk outside shared memory";
                    return false;
                }
                const int ko = (layer * s.kv_chunk + in_chunk) * kv_row;
                const int vo = ((s.n_attn + layer) * s.kv_chunk + in_chunk) * kv_row;
                const int ksi = s0 + (layer * s.kv_chunk + in_chunk) * s.nkv;
                const int vsi = s0 + ((s.n_attn + layer) * s.kv_chunk + in_chunk) * s.nkv;
                if (!p.pgo("pf_kv_store8", static_cast<size_t>(kv_row), static_cast<size_t>(cnt), static_cast<size_t>(hd), 1,
                           err, kb, vb, cp->mem, ko, vo, cp->mem, ksi, vsi, t0 * kv_row, kv_row, s.nkv))
                    return false;
                t0 += cnt;
            }
            const int last = kp + m - 1;
            static const int which = std::getenv("GEODE_PF_ATTN") ? std::atoi(std::getenv("GEODE_PF_ATTN")) : 2;
            static const char *const names[] = {"pf_attn8", "pf_attn8x", "pf_attn8t"};
            const int wi = which >= 2 && !p.attn4 ? 1 : std::clamp(which, 0, 2);
            const size_t items = wi == 0 ? mt * 256 : wi == 2 ? (mt + 3) / 4 * 256 : mt * 64;
            for (int ch = 0, c0 = 0; c0 <= last; ++ch, c0 += s.kv_chunk) {
                const Impl::Piece *ck = p.find(kv[static_cast<size_t>(ch)]);
                if (!ck) {
                    err = "GPU: a KV chunk outside shared memory";
                    return false;
                }
                const int len = std::min(s.kv_chunk, last + 1 - c0);
                if (!p.pgo(names[wi], items, static_cast<size_t>(s.nkv), wi == 1 ? 64 : 256, 1, err, qg, qrow, ck->mem,
                           layer * s.kv_chunk * kv_row, (s.n_attn + layer) * s.kv_chunk * kv_row, kv_row, ck->mem,
                           s0 + layer * s.kv_chunk * s.nkv, s0 + (s.n_attn + layer) * s.kv_chunk * s.nkv, s.nkv, c0, len,
                           kp, scale, pm, pl, po, s.nh, ch == 0 ? 1 : 0, m))
                    return false;
            }
            const bool ok = p.pgo("pf_attn_out_f", static_cast<size_t>(s.nh * hd / 16), mt, 0, 0, err, po, pl, qg, qrow,
                                  hd, s.nh, core);
            p.cl->clFlush(p.dev.queue);
            return ok;
        };
        bool ok = proj(0) && core(0);
        for (int c = 0; ok && c < parts; ++c) {
            if (c + 1 < parts) ok = proj(c + 1);
            ok = ok && p.finish(err);
            if (ok && c + 1 < parts) ok = core(c + 1);
            ok = ok && out(c);
        }
        if (!ok) {
            std::string e2;
            p.finish(e2);
        }
        p.drop_subs();
        return ok;
    }
    bool host = false;
    if (p.host_gemm) {
        const QView *ms[3] = {&L.q, &L.k, &L.v};
        float *outs[3] = {reinterpret_cast<float *>(p.pf_qg.host), reinterpret_cast<float *>(p.pf_kb.host),
                          reinterpret_cast<float *>(p.pf_vb.host)};
        host = p.host_gemm(ms, 3, n, h, H, outs, false);
    }
    if (host) p.pf_maps(h, n, H);
    if (!(host || (p.pf_in(h, n, H, err) && p.pf_gemm(L.q, n, p.pf_qg, err) && p.pf_gemm(L.k, n, p.pf_kb, err) &&
                   p.pf_gemm(L.v, n, p.pf_vb, err))) ||
        !p.pgo("pf_head_norm_rope", static_cast<size_t>(s.nh * hd) * nt, 0, static_cast<size_t>(hd), 0, err,
               p.pf_qg.mem, 2 * hd, P.mem, p.at(P, L.q_norm, 4), s.eps, s.half, p.pf_pos.mem, p.d_freq.mem, qrow, s.nh) ||
        !p.pgo("pf_head_norm_rope", static_cast<size_t>(s.nkv * hd) * nt, 0, static_cast<size_t>(hd), 0, err,
               p.pf_kb.mem, hd, P.mem, p.at(P, L.k_norm, 4), s.eps, s.half, p.pf_pos.mem, p.d_freq.mem, kv_row, s.nkv))
        return false;
    // the tokens' K and V into the cache, a chunk's run of them a launch (Model::kv_scale's layout)
    const int s0 = 2 * s.n_attn * s.kv_chunk * kv_row / 4;
    for (int t0 = 0; t0 < n;) {
        const int pt = kv_pos + t0, c = pt / s.kv_chunk, in_chunk = pt % s.kv_chunk;
        const int cnt = std::min(n - t0, s.kv_chunk - in_chunk);
        const Impl::Piece *cp = p.find(kv[static_cast<size_t>(c)]);
        if (!cp || cp->host != kv[static_cast<size_t>(c)]) {
            err = "GPU: a KV chunk outside shared memory";
            return false;
        }
        const int ko = (layer * s.kv_chunk + in_chunk) * kv_row, vo = ((s.n_attn + layer) * s.kv_chunk + in_chunk) * kv_row;
        const int ksi = s0 + (layer * s.kv_chunk + in_chunk) * s.nkv;
        const int vsi = s0 + ((s.n_attn + layer) * s.kv_chunk + in_chunk) * s.nkv;
        if (!p.pgo("pf_kv_store8", static_cast<size_t>(kv_row), static_cast<size_t>(cnt), static_cast<size_t>(hd), 1, err,
                   p.pf_kb.mem, p.pf_vb.mem, cp->mem, ko, vo, cp->mem, ksi, vsi, t0 * kv_row, kv_row, s.nkv))
            return false;
        t0 += cnt;
    }
    // the attention chunk by chunk, the softmax carried from one to the next
    const int last = kv_pos + n - 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int c = 0, c0 = 0; c0 <= last; ++c, c0 += s.kv_chunk) {
        const Impl::Piece *ck = p.find(kv[static_cast<size_t>(c)]);
        if (!ck) {
            err = "GPU: a KV chunk outside shared memory";
            return false;
        }
        const int len = std::min(s.kv_chunk, last + 1 - c0);
        // four tokens a work group (pf_attn8t) where its sub-groups are, else a token a wave (pf_attn8x);
        // GEODE_PF_ATTN=0 / 1: the 256-item kernel of before, pf_attn8x (measuring)
        static const int which = std::getenv("GEODE_PF_ATTN") ? std::atoi(std::getenv("GEODE_PF_ATTN")) : 2;
        static const char *const names[] = {"pf_attn8", "pf_attn8x", "pf_attn8t"};
        const int wi = which >= 2 && !p.attn4 ? 1 : std::clamp(which, 0, 2);
        const size_t items = wi == 0 ? nt * 256 : wi == 2 ? (nt + 3) / 4 * 256 : nt * 64;
        if (!p.pgo(names[wi], items, static_cast<size_t>(s.nkv), wi == 1 ? 64 : 256, 1,
                   err, p.pf_qg.mem, qrow, ck->mem,
                   layer * s.kv_chunk * kv_row, (s.n_attn + layer) * s.kv_chunk * kv_row, kv_row, ck->mem,
                   s0 + layer * s.kv_chunk * s.nkv, s0 + (s.n_attn + layer) * s.kv_chunk * s.nkv, s.nkv, c0, len, kv_pos,
                   scale, p.pf_pm.mem, p.pf_pl.mem, p.pf_po.mem, s.nh, c == 0 ? 1 : 0, n))
            return false;
    }
    if (host) {   // the o projection there too, from the gated attention's floats
        if (!p.pgo("pf_attn_out_f", static_cast<size_t>(s.nh * hd / 16), nt, 0, 0, err, p.pf_po.mem, p.pf_pl.mem,
                   p.pf_qg.mem, qrow, hd, s.nh, p.pf_core.mem) ||
            !p.finish(err))
            return false;
        const QView *ms[1] = {&L.o};
        float *outs[1] = {mix};
        if (p.host_gemm(ms, 1, n, reinterpret_cast<const float *>(p.pf_core.host), s.nh * hd, outs, false)) return true;
        // not taken: the o projection here after all, from the floats quantized into the tiles
        p.pf_maps(reinterpret_cast<const float *>(p.pf_core.host), n, s.nh * hd);
        if (!p.quant(p.x, p.xs, p.xb, s.nh * hd, round_up(nt, kT), err)) return false;
    }
    if (!(host || p.pgo("pf_attn_out", static_cast<size_t>(s.nh * hd / 16), nt, 0, 0, err, p.pf_po.mem, p.pf_pl.mem,
                        p.pf_qg.mem, qrow, hd, s.nh, p.x.mem, p.xs.mem, p.xb.mem)) ||
        !p.pf_gemm(L.o, n, p.pf_mix, err) || !p.finish(err))
        return false;
    std::memcpy(mix, p.pf_mix.host, nt * static_cast<size_t>(H) * 4);
    return true;
}

bool Gpu8::dec_wait(std::string &err) {
    const cl_int e = p_->cl->clFinish(p_->q1);
    if (e != CL_SUCCESS) {
        err = std::string("GPU: finish: ") + cl::error_name(e);
        return false;
    }
    return true;
}

bool Gpu8::dec_copy(void *dst, const void *src, size_t bytes, std::string &err) {
    Impl &p = *p_;
    const Impl::Piece *d = p.find(dst), *sp = p.find(src);
    if (!d || !sp) {
        err = "GPU: a copy outside shared memory";
        return false;
    }
    const cl_int e = p.cl->clEnqueueCopyBuffer(p.q1, sp->mem, d->mem,
                                               static_cast<size_t>(static_cast<const uint8_t *>(src) - sp->host),
                                               static_cast<size_t>(static_cast<uint8_t *>(dst) - d->host), bytes, 0,
                                               nullptr, nullptr);
    if (e != CL_SUCCESS) {
        err = std::string("GPU: copy: ") + cl::error_name(e);
        return false;
    }
    return true;
}

void Gpu8::dec_selftest(const QView &m, int reps) {
    Impl &p = *p_;
    std::string err;
    p.label_prefix = "selftest: ";
    for (int i = 0; i < reps; ++i) p.gemv1(m, p.d_logits, err);
    p.cl->clFinish(p.q1);
    p.jcur = 0;
    p.label_prefix.clear();
    if (p.dprof) p.collect();
}

std::string Gpu8::dec_profile() const {
    std::vector<std::pair<double, std::string>> v;
    for (const auto &[name, t] : p_->dtimes)
        v.emplace_back(t.first, name + ": " + std::to_string(static_cast<int>(t.first * 1e3)) + " ms in " +
                                    std::to_string(t.second) + " launches");
    std::sort(v.rbegin(), v.rend());
    std::string out;
    for (const auto &[t, line] : v) out += line + "\n";
    return out;
}

std::string Gpu8::device() const { return p_->dev.name; }
size_t Gpu8::max_piece() const { return static_cast<size_t>(p_->dev.max_alloc) & ~size_t{4095}; }
uint8_t *Gpu8::vision_alloc(size_t bytes, std::string &err) {
    Impl &p = *p_;
    bytes = round_up(std::max<size_t>(bytes, 64), 64);
    // carved from blocks of at least 32 MB (fewer buffers; each its own piece for find())
    if (p.vblocks.empty() || p.vused + bytes > p.vblocks.back().bytes) {
        Impl::Piece b;
        if (!p.alloc(std::max<size_t>(bytes, size_t{32} << 20), b, err)) return nullptr;
        p.vblocks.push_back(b);
        p.vused = 0;
    }
    uint8_t *out = p.vblocks.back().host + p.vused;
    p.vused += bytes;
    return out;
}

void Gpu8::vision_release() {
    Impl &p = *p_;
    p.cl->clFinish(p.dev.queue);
    std::vector<Impl::Piece> gone = p.vblocks;
    for (const Impl::Piece *x : {&p.vX, &p.vXS, &p.vXB})
        if (x->mem) gone.push_back(*x);
    for (const Impl::Piece &g : gone) {
        p.cl->clEnqueueUnmapMemObject(p.dev.queue, g.mem, g.host, 0, nullptr, nullptr);
        p.cl->clReleaseMemObject(g.mem);
        p.allocated -= g.bytes;
        p.pieces.erase(std::remove_if(p.pieces.begin(), p.pieces.end(),
                                      [&](const Impl::Piece &x) { return x.mem == g.mem; }),
                       p.pieces.end());
    }
    p.cl->clFinish(p.dev.queue);
    p.vblocks.clear();
    p.vused = 0;
    p.vX = p.vXS = p.vXB = {};
}

size_t Gpu8::q8gpu_bytes(int64_t rows, int64_t cols) {
    return static_cast<size_t>(rows * cols + rows * (cols / 16));
}

void Gpu8::to_q8gpu(const QView &m, uint8_t *dst) {
    const int64_t quads = m.rows / 4, nb = m.cols / 16;
    uint8_t *sc = dst + static_cast<size_t>(m.rows * m.cols);
    for (int64_t q = 0; q < quads; ++q)
        for (int r = 0; r < 4; ++r) {
            const uint8_t *row = m.data + static_cast<size_t>(4 * q + r) * m.row_bytes;   // codes, then the scales
            for (int64_t k = 0; k < nb; ++k) {
                uint8_t *o = dst + static_cast<size_t>(((k * quads + q) * 4 + r) * 16);
                for (int i = 0; i < 16; ++i) o[i] = static_cast<uint8_t>(row[16 * k + i] ^ 0x80u);   // +128
                sc[static_cast<size_t>((k * quads + q) * 4 + r)] = row[m.cols + k];
            }
        }
}

bool Gpu8::vision_gemm(const uint8_t *w, int64_t rows, int64_t cols, float global, int n, const float *x,
                       size_t xstride, float *y, size_t ystride, std::string &err) {
    return vision_gemm_start(w, rows, cols, rows, global, n, x, xstride, y, ystride, err) && vision_wait(err);
}

bool Gpu8::vision_wait(std::string &err) { return p_->finish(err); }

bool Gpu8::vision_gemm_start(const uint8_t *w, int64_t rows, int64_t cols, int64_t first, float global, int n,
                             const float *x, size_t xstride, float *y, size_t ystride, std::string &err) {
    Impl &p = *p_;
    const Impl::Piece *wp = p.find(w), *xp = p.find(x), *yp = p.find(y);
    if (!wp || !xp || !yp || rows % 4 || first % 4 || first <= 0 || first > rows || cols % 16 || n <= 0 ||
        (w - wp->host) % 16) {
        err = "GPU: an image product it cannot take";
        return false;
    }
    const size_t slots = round_up(static_cast<size_t>(n), kT), nb = static_cast<size_t>(cols / 16);
    const auto grow = [&](Impl::Piece &b, size_t bytes) {
        if (b.mem && b.bytes >= bytes) return true;
        if (b.mem) {
            p.cl->clFinish(p.dev.queue);
            p.cl->clEnqueueUnmapMemObject(p.dev.queue, b.mem, b.host, 0, nullptr, nullptr);
            p.cl->clReleaseMemObject(b.mem);
            p.allocated -= b.bytes;
            const cl_mem gone = b.mem;
            p.pieces.erase(std::remove_if(p.pieces.begin(), p.pieces.end(),
                                          [&](const Impl::Piece &z) { return z.mem == gone; }),
                           p.pieces.end());
            b = {};
        }
        return p.alloc(bytes, b, err);
    };
    if (!grow(p.vX, slots * nb * 16) || !grow(p.vXS, slots * nb * 4) || !grow(p.vXB, slots * nb * 4)) return false;
    const int ci = static_cast<int>(cols), quads = static_cast<int>(rows / 4), qn = static_cast<int>(first / 4);
    arg(p.cl, p.k_quant8, 0, xp->mem);
    arg(p.cl, p.k_quant8, 1, static_cast<int>((reinterpret_cast<const uint8_t *>(x) - xp->host) / 4));
    arg(p.cl, p.k_quant8, 2, static_cast<int>(xstride));
    arg(p.cl, p.k_quant8, 3, n);
    arg(p.cl, p.k_quant8, 4, ci);
    arg(p.cl, p.k_quant8, 5, p.vX.mem);
    arg(p.cl, p.k_quant8, 6, p.vXS.mem);
    arg(p.cl, p.k_quant8, 7, p.vXB.mem);
    const size_t gq[2] = {slots, nb};
    if (!p.run(p.k_quant8, nullptr, gq, nullptr, err)) return false;
    arg(p.cl, p.k_gemm8, 0, wp->mem);
    arg(p.cl, p.k_gemm8, 1, static_cast<int>((w - wp->host) / 16));
    arg(p.cl, p.k_gemm8, 2, quads);
    arg(p.cl, p.k_gemm8, 3, qn);
    arg(p.cl, p.k_gemm8, 4, ci);
    arg(p.cl, p.k_gemm8, 5, global * 256.0f);   // the kernel's E4M3 values are / 256
    arg(p.cl, p.k_gemm8, 6, p.vX.mem);
    arg(p.cl, p.k_gemm8, 7, p.vXS.mem);
    arg(p.cl, p.k_gemm8, 8, p.vXB.mem);
    arg(p.cl, p.k_gemm8, 9, yp->mem);
    arg(p.cl, p.k_gemm8, 10, static_cast<int>((reinterpret_cast<const uint8_t *>(y) - yp->host) / 4));
    arg(p.cl, p.k_gemm8, 11, static_cast<int>(ystride));
    arg(p.cl, p.k_gemm8, 12, n);
    const size_t gg[2] = {slots / kT, round_up(static_cast<size_t>(qn), 64)}, lg[2] = {1, 64};
    if (!p.run(p.k_gemm8, nullptr, gg, lg, err)) return false;
    p.cl->clFlush(p.dev.queue);   // started now: the CPU's share runs alongside
    return true;
}

size_t Gpu8::allocated() const { return p_->allocated; }
double Gpu8::busy_seconds() const { return p_->busy; }
int64_t Gpu8::launches() const { return p_->launches; }
const float *Gpu8::out() const { return reinterpret_cast<const float *>(p_->eout.host); }
float *Gpu8::out_rows() { return reinterpret_cast<float *>(p_->eout.host); }

uint8_t *Gpu8::alloc(size_t bytes, std::string &err) {
    Impl::Piece piece;
    if (bytes > max_piece()) {
        err = "GPU: " + std::to_string(bytes >> 20) + " MB in one piece (at most " + std::to_string(max_piece() >> 20) + ")";
        return nullptr;
    }
    return p_->alloc(bytes, piece, err) ? piece.host : nullptr;
}

bool Gpu8::import(int fd, uint8_t *host, size_t bytes, std::string &err) {
    if (bytes > max_piece()) {
        err = "GPU: " + std::to_string(bytes >> 20) + " MB in one piece (at most " + std::to_string(max_piece() >> 20) + ")";
        return false;
    }
    return p_->import(fd, host, bytes, err);
}

bool Gpu8::free(uint8_t *piece) {
    Impl &p = *p_;
    const auto it = std::find_if(p.pieces.begin(), p.pieces.end(), [&](const Impl::Piece &x) { return x.host == piece; });
    if (it == p.pieces.end()) return false;
    const Impl::Piece g = *it;
    if (p.q1 && p.q1 != p.dev.queue) p.cl->clFinish(p.q1);
    p.cl->clFinish(p.dev.queue);
    if (!g.imported) p.cl->clEnqueueUnmapMemObject(p.dev.queue, g.mem, g.host, 0, nullptr, nullptr);
    p.cl->clReleaseMemObject(g.mem);
    p.cl->clFinish(p.dev.queue);
    if (!g.imported) p.allocated -= g.bytes;
    p.pieces.erase(std::remove_if(p.pieces.begin(), p.pieces.end(), [&](const Impl::Piece &x) { return x.mem == g.mem; }),
                   p.pieces.end());
    return true;
}

void Gpu8::set_experts(int k, int guesses) {
    Impl &p = *p_;
    const int most = std::max(p.top_k_max, p.ds.top_k);
    if (k < 1 || (most > 0 && k > most)) return;
    p.top_k = k;
    if (p.ds.top_k > 0) p.ds.top_k = k;
    if (guesses > 0 && guesses <= std::max(p.guesses_max, 1)) p.ds.guesses = guesses;
}

bool Gpu8::fits(const QView &m) { return m.data && !m.quad && tile8_capable(m.type, m.rows, m.cols); }

bool Gpu8::holds(const QView &m) const {
    return m.type == (p_->q4i ? DType::Q4I : DType::Q4) && m.tile8 && m.cols % 64 == 0 && p_->find(m.data) != nullptr;
}

bool Gpu8::adopt(QView &m, std::string &err) {
    Impl &p = *p_;
    if (auto it = p.moved.find(m.data); it != p.moved.end()) {
        m.data = it->second;
        m.quad = m.tile8 = true;
        return true;
    }
    if (!fits(m)) {
        err = "GPU: not a q4 matrix of 32-row tiles in the pack's layout";
        return false;
    }
    const size_t bytes = static_cast<size_t>(m.rows) * m.row_bytes;
    Impl::Piece piece;
    if (!p.alloc(bytes, piece, err)) return false;
    to_tile8(m.type, m.rows, m.cols, m.data, piece.host);
    p.moved[m.data] = piece.host;
    m.data = piece.host;
    m.quad = m.tile8 = true;
    return true;
}

bool Gpu8::adopt_all(const std::vector<QView *> &views, int threads, std::string &err) {
    Impl &p = *p_;
    struct Job {
        const QView *m;   // the first view of these bytes
        uint8_t *dst = nullptr;
        std::string e;
    };
    std::vector<Job> jobs;
    std::unordered_map<const uint8_t *, size_t> first;
    for (QView *m : views)
        if (!p.moved.count(m->data) && fits(*m) && first.emplace(m->data, jobs.size()).second) jobs.push_back({m});
    auto run = [&](size_t n, const auto &one) {   // one(i) for i < n, on up to `threads` threads
        std::atomic<size_t> next{0};
        auto work = [&] {
            for (size_t i; (i = next.fetch_add(1)) < n;) one(i);
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < std::min<int>(threads, static_cast<int>(n)); ++t) pool.emplace_back(work);
        work();
        for (std::thread &t : pool) t.join();
    };
    run(jobs.size(), [&](size_t i) {
        Impl::Piece piece;
        if (p.alloc(static_cast<size_t>(jobs[i].m->rows) * jobs[i].m->row_bytes, piece, jobs[i].e)) jobs[i].dst = piece.host;
    });
    // the conversion in units of 256 rows (8 tiles: each tile is its own bytes), the big matrices spread too
    constexpr int64_t kRows = 256;
    std::vector<std::pair<size_t, int64_t>> units;
    for (size_t i = 0; i < jobs.size(); ++i)
        if (jobs[i].dst)
            for (int64_t r = 0; r < jobs[i].m->rows; r += kRows) units.emplace_back(i, r);
    run(units.size(), [&](size_t u) {
        const Job &j = jobs[units[u].first];
        const int64_t r0 = units[u].second, n = std::min(kRows, j.m->rows - r0);
        const size_t off = static_cast<size_t>(r0) * j.m->row_bytes;
        to_tile8(j.m->type, n, j.m->cols, j.m->data + off, j.dst + off);
    });
    bool ok = true;
    for (const Job &j : jobs) {
        if (j.dst) p.moved[j.m->data] = j.dst;
        else if (ok) {
            ok = false;
            err = j.e;
        }
    }
    for (QView *m : views)
        if (auto it = p.moved.find(m->data); it != p.moved.end()) {
            m->data = it->second;
            m->quad = m->tile8 = true;
        }
    return ok;
}

size_t Gpu8::staging_bytes(int tokens, int64_t max_cols, int64_t max_rows, int64_t hidden, int64_t ff, int top_k,
                           int n_experts) const {
    const size_t T = kT, n = static_cast<size_t>(tokens);
    // the experts' slots: the routed picks, each expert's padded to whole tiles, then the shared expert's (every token)
    const size_t sd = round_up(n, T),
                 se = round_up(n * static_cast<size_t>(top_k) + static_cast<size_t>(n_experts) * (T - 1), T) + sd;
    const size_t mc = static_cast<size_t>(std::max(max_cols, hidden)), h = static_cast<size_t>(hidden),
                 f = static_cast<size_t>(ff);
    const size_t xbytes = std::max(sd * mc, se * h), slots = std::max(sd, se);
    return n * mc * 4                                       // in
           + xbytes + 2 * (xbytes / 16 * 4)                 // x, xs, xb
           + 4 * slots * 4                                  // rowmap, ymap_d, ymap_id, ymap_dn
           + (sd / T + 2 * (se / T) + sd / T) * 32          // jobs (the shared gate and up: two a tile)
           + sd * static_cast<size_t>(max_rows) * 4         // y
           + se * 2 * f * 4                                 // gu
           + se * f + 2 * (se * f / 16 * 4)                 // x2, x2s, x2b
           + n * static_cast<size_t>(top_k + 1) * h * 4;    // eout
}

bool Gpu8::reserve(int tokens, int64_t max_cols, int64_t max_rows, int64_t hidden, int64_t ff, int top_k,
                   int n_experts, std::string &err) {
    Impl &p = *p_;
    if (tokens <= p.tokens) return true;
    const size_t T = kT, n = static_cast<size_t>(tokens);
    const size_t sd = round_up(n, T),
                 se = round_up(n * static_cast<size_t>(top_k) + static_cast<size_t>(n_experts) * (T - 1), T) + sd;
    const size_t mc = static_cast<size_t>(std::max(max_cols, hidden)), h = static_cast<size_t>(hidden),
                 f = static_cast<size_t>(ff);
    const size_t xbytes = std::max(sd * mc, se * h), slots = std::max(sd, se);
    const bool ok = p.alloc(n * mc * 4, p.in, err) && p.alloc(xbytes, p.x, err) && p.alloc(xbytes / 16 * 4, p.xs, err) &&
                    p.alloc(xbytes / 16 * 4, p.xb, err) && p.alloc(slots * 4, p.rowmap, err) &&
                    p.alloc(slots * 4, p.ymap_d, err) && p.alloc(slots * 4, p.ymap_id, err) &&
                    p.alloc(slots * 4, p.ymap_dn, err) && p.alloc(sd / T * 32, p.jobs_d, err) &&
                    p.alloc((se + sd) / T * 32, p.jobs_gu, err) && p.alloc(se / T * 32, p.jobs_dn, err) &&
                    p.alloc(sd * static_cast<size_t>(max_rows) * 4, p.y, err) && p.alloc(se * 2 * f * 4, p.gu, err) &&
                    p.alloc(se * f, p.x2, err) && p.alloc(se * f / 16 * 4, p.x2s, err) &&
                    p.alloc(se * f / 16 * 4, p.x2b, err) &&
                    p.alloc(n * static_cast<size_t>(top_k + 1) * h * 4, p.eout, err);
    if (!ok) return false;
    auto *id = reinterpret_cast<int32_t *>(p.ymap_id.host);
    for (size_t s = 0; s < slots; ++s) id[s] = static_cast<int32_t>(s);
    p.tokens = tokens;
    p.max_cols = static_cast<int64_t>(mc);
    p.max_rows = max_rows;
    p.hidden = hidden;
    p.ff = ff;
    p.top_k = top_k;
    p.top_k_max = std::max(p.top_k_max, top_k);
    p.n_experts = n_experts;
    p.slots_e = se;
    return true;
}

bool Gpu8::gemm(const QView &m, int n, const float *const *xs, float *const *ys, std::string &err) {
    Impl &p = *p_;
    const Impl::Piece *w = p.find(m.data);
    if (!w || !holds(m) || n > p.tokens || m.cols > p.max_cols || m.rows > p.max_rows) {
        err = "GPU: a product it cannot take";
        return false;
    }
    const size_t T = kT, slots = round_up(static_cast<size_t>(n), T), cols = static_cast<size_t>(m.cols);
    auto *in = reinterpret_cast<float *>(p.in.host);
    for (int t = 0; t < n; ++t) std::memcpy(in + static_cast<size_t>(t) * cols, xs[t], cols * 4);
    auto *rm = reinterpret_cast<int32_t *>(p.rowmap.host), *ym = reinterpret_cast<int32_t *>(p.ymap_d.host);
    for (size_t s = 0; s < slots; ++s) rm[s] = ym[s] = s < static_cast<size_t>(n) ? static_cast<int32_t>(s) : -1;
    auto *jobs = reinterpret_cast<Job *>(p.jobs_d.host);
    const auto off = static_cast<int32_t>((m.data - w->host) / 16);
    for (size_t j = 0; j < slots / T; ++j)
        jobs[j] = {off, static_cast<int32_t>(j), 0, float_bits(m.global * 256.0f), 0, static_cast<int32_t>(m.rows), 0, 0};
    p.cur = "dense";
    if (!p.quant(p.x, p.xs, p.xb, static_cast<int>(cols), slots, err) ||
        !p.gemm(*w, p.jobs_d, 0, slots / T, p.x, p.xs, p.xb, p.ymap_d, p.y, static_cast<int>(m.rows),
                static_cast<int>(cols), err) ||
        !p.finish(err)) {
        p.cur = "batch";
        return false;
    }
    p.cur = "batch";
    const auto *y = reinterpret_cast<const float *>(p.y.host);
    for (int t = 0; t < n; ++t)
        std::memcpy(ys[t], y + static_cast<size_t>(t) * static_cast<size_t>(m.rows), static_cast<size_t>(m.rows) * 4);
    return true;
}

bool Gpu8::experts(const std::vector<Expert> &ex, int n, const float *const *hs, std::string &err,
                   const DecodeLayer *shared) {
    Impl &p = *p_;
    if (n > p.tokens) {
        err = "GPU: more tokens than reserved";
        return false;
    }
    const size_t T = kT, H = static_cast<size_t>(p.hidden);
    auto *in = reinterpret_cast<float *>(p.in.host);
    for (int t = 0; t < n; ++t) std::memcpy(in + static_cast<size_t>(t) * H, hs[t], H * 4);
    // slots: each expert's picks, padded to whole tiles; its tiles' jobs grouped by the piece its slot lies in
    auto *rm = reinterpret_cast<int32_t *>(p.rowmap.host), *yd = reinterpret_cast<int32_t *>(p.ymap_dn.host);
    struct TileJob {
        const Impl::Piece *piece;
        Job gu, dn;
    };
    std::vector<TileJob> tj;
    size_t s = 0;
    for (const Expert &e : ex) {
        const Impl::Piece *w = p.find(e.gu.data);
        if (!w || !holds(e.gu) || !holds(e.dn) || p.find(e.dn.data) != w || e.gu.rows != 2 * p.ff ||
            e.gu.cols != p.hidden || e.dn.rows != p.hidden || e.dn.cols != p.ff) {
            err = "GPU: an expert it cannot take";
            return false;
        }
        const size_t first = s, end = round_up(s + e.picks.size(), T);
        if (end > p.slots_e) {
            err = "GPU: more expert picks than reserved";
            return false;
        }
        for (const auto &[tok, row] : e.picks) {
            rm[s] = tok;
            yd[s] = row;
            ++s;
        }
        for (; s < end; ++s) rm[s] = yd[s] = -1;
        const auto og = static_cast<int32_t>((e.gu.data - w->host) / 16), od = static_cast<int32_t>((e.dn.data - w->host) / 16);
        for (size_t t = first / T; t < end / T; ++t)
            tj.push_back({w, {og, static_cast<int32_t>(t), 0, float_bits(e.gu.global * 256.0f), 0,
                              static_cast<int32_t>(e.gu.rows), 0, 0},
                          {od, static_cast<int32_t>(t), 0, float_bits(e.dn.global * 256.0f), 0,
                           static_cast<int32_t>(e.dn.rows), 0, 0}});
    }
    // the shared expert: every token, after the routed slots; its gate and up a job a tile each (into the gate and
    // the up halves of the slot's gu row), its output to row n * top_k + t; each matrix in its own piece maybe
    const Impl::Piece *pg = nullptr, *pu = nullptr, *pd = nullptr;
    const size_t s_first = s;
    if (shared) {
        const QView &g = shared->sh_gate, &u = shared->sh_up, &d = shared->sh_down;
        pg = p.find(g.data), pu = p.find(u.data), pd = p.find(d.data);
        if (!pg || !pu || !pd || !holds(g) || !holds(u) || !holds(d) || g.rows != p.ff || u.rows != p.ff ||
            g.cols != p.hidden || u.cols != p.hidden || d.rows != p.hidden || d.cols != p.ff ||
            s + round_up(static_cast<size_t>(n), T) > p.slots_e) {
            err = "GPU: a shared expert it cannot take";
            return false;
        }
        for (int t = 0; t < n; ++t, ++s) {
            rm[s] = t;
            yd[s] = n * p.top_k + t;
        }
        for (; s % T; ++s) rm[s] = yd[s] = -1;
    }
    if (s == 0) return true;
    std::stable_sort(tj.begin(), tj.end(), [](const TileJob &a, const TileJob &b) { return a.piece < b.piece; });
    auto *jg = reinterpret_cast<Job *>(p.jobs_gu.host), *jd = reinterpret_cast<Job *>(p.jobs_dn.host);
    for (size_t i = 0; i < tj.size(); ++i) {
        jg[i] = tj[i].gu;
        jd[i] = tj[i].dn;
    }
    const size_t sh_tiles = (s - s_first) / T;
    if (shared) {
        const QView &g = shared->sh_gate, &u = shared->sh_up, &d = shared->sh_down;
        const auto og = static_cast<int32_t>((g.data - pg->host) / 16), ou = static_cast<int32_t>((u.data - pu->host) / 16),
                   od = static_cast<int32_t>((d.data - pd->host) / 16);
        for (size_t t = 0; t < sh_tiles; ++t) {   // jobs: the gate's tiles, then the up's
            const auto tile = static_cast<int32_t>(s_first / T + t);
            jg[tj.size() + t] = {og, tile, 0, float_bits(g.global * 256.0f), 0, static_cast<int32_t>(g.rows), 0, 0};
            jg[tj.size() + sh_tiles + t] = {ou, tile, static_cast<int32_t>(p.ff), float_bits(u.global * 256.0f), 0,
                                            static_cast<int32_t>(u.rows), 0, 0};
            jd[tj.size() + t] = {od, tile, 0, float_bits(d.global * 256.0f), 0, static_cast<int32_t>(d.rows), 0, 0};
        }
    }
    const int h = static_cast<int>(H), f = static_cast<int>(p.ff);
    p.cur = "moe quant";
    if (!p.quant(p.x, p.xs, p.xb, h, s, err)) return false;
    p.cur = "moe gate_up";
    for (size_t i = 0; i < tj.size();) {   // gate_up, a launch per piece
        size_t j = i;
        while (j < tj.size() && tj[j].piece == tj[i].piece) ++j;
        if (!p.gemm(*tj[i].piece, p.jobs_gu, i, j - i, p.x, p.xs, p.xb, p.ymap_id, p.gu, 2 * f, h, err)) return false;
        i = j;
    }
    if (sh_tiles) {   // the gate and the up: one launch when they lie in one piece
        if (pg == pu) {
            if (!p.gemm(*pg, p.jobs_gu, tj.size(), 2 * sh_tiles, p.x, p.xs, p.xb, p.ymap_id, p.gu, f, h, err, 2 * f))
                return false;
        } else if (!p.gemm(*pg, p.jobs_gu, tj.size(), sh_tiles, p.x, p.xs, p.xb, p.ymap_id, p.gu, f, h, err, 2 * f) ||
                   !p.gemm(*pu, p.jobs_gu, tj.size() + sh_tiles, sh_tiles, p.x, p.xs, p.xb, p.ymap_id, p.gu, f, h, err,
                           2 * f)) {
            return false;
        }
    }
    p.cur = "moe swiglu";
    arg(p.cl, p.k_swiglu, 0, p.gu.mem);
    arg(p.cl, p.k_swiglu, 1, f);
    arg(p.cl, p.k_swiglu, 2, p.x2.mem);
    arg(p.cl, p.k_swiglu, 3, p.x2s.mem);
    arg(p.cl, p.k_swiglu, 4, p.x2b.mem);
    const size_t g[2] = {s, static_cast<size_t>(f / 16)};
    if (!p.run(p.k_swiglu, nullptr, g, nullptr, err)) return false;
    p.cur = "moe down";
    for (size_t i = 0; i < tj.size();) {   // down, into the picks' rows
        size_t j = i;
        while (j < tj.size() && tj[j].piece == tj[i].piece) ++j;
        if (!p.gemm(*tj[i].piece, p.jobs_dn, i, j - i, p.x2, p.x2s, p.x2b, p.ymap_dn, p.eout, h, f, err)) return false;
        i = j;
    }
    if (sh_tiles && !p.gemm(*pd, p.jobs_dn, tj.size(), sh_tiles, p.x2, p.x2s, p.x2b, p.ymap_dn, p.eout, h, f, err))
        return false;
    p.cur = "batch";
    return p.finish(err);
}

// ---- Gemma 4's answer on the GPU ----

static const char *const kGemmaKernels[] = {"gemvs",       "gemvs1",   "g_quant",   "g_norm_q", "g_post",
                                            "g_post_q",    "g_gelu_q", "g_ple_in",  "g_ple_mid", "g_ple_out",
                                            "g_head",      "g_kv_store8", "g_attn8", "g_attn_merge", "g_softcap",
                                            "g_done",      "g_attn_small", "g_kv_store8n", "g_attn_chunk", "g_attn_join",
                                            "g_attn_pf256", "g_attn_pf512"};

bool Gpu8::setup_gemma(const GemmaShape &s, std::string &err) {
    Impl &p = *p_;
    const int G = s.nkv > 0 ? s.nh / s.nkv : 0;
    if (s.hidden % 256 || s.ple <= 0 || s.ple > 256 || (s.ple & (s.ple - 1)) || s.hd_max > 512 || s.nkv <= 0 ||
        s.nh % s.nkv || G < 1 || G > 8 || s.kv_chunk % 128 || s.ffn_max % 64 || s.max_tokens < 1) {
        err = "GPU: a Gemma shape the one-token kernels do not take";
        return false;
    }
    if (!p.q4i) {
        err = "GPU: Gemma's answer needs the q4i programs";
        return false;
    }
    cl_int e = CL_SUCCESS;
    p.dprof = std::getenv("GEODE_GPU8_PROF") && std::atoi(std::getenv("GEODE_GPU8_PROF")) != 0;
    p.nb_max = (s.max_ctx + 127) / 128 + 1;
    if (!p.q1) p.q1 = p.cl->clCreateCommandQueue(p.dev.context, p.dev.id, p.dprof ? CL_QUEUE_PROFILING_ENABLE : 0, &e);
    if (!p.q1) {
        err = std::string("GPU: a plain queue: ") + cl::error_name(e);
        return false;
    }
    const int mt = std::clamp(s.max_tokens, 1, kMaxTokens);
    for (int t = 1; t <= mt; ++t) {
        cl::Program &prog = t == 1 ? p.prog1 : p.progT[t];
        auto &km = t == 1 ? p.k1 : p.kTok[t];
        if (!prog.build_cached(p.dev, kSource, "-cl-std=CL3.0 -cl-mad-enable -DQ4I -DT=" + std::to_string(t), p.cache_dir,
                               err))
            return false;
        for (const char *n : kGemmaKernels) {
            cl_kernel k = prog.kernel(n, err);
            if (!k) return false;
            km[n] = k;
        }
    }
    const size_t f = 4, nt = static_cast<size_t>(mt), H = static_cast<size_t>(s.hidden);
    const size_t qw = static_cast<size_t>(s.nh) * s.hd_max, kw = static_cast<size_t>(s.nkv) * s.hd_max;
    const size_t ff = static_cast<size_t>(s.ffn_max), lp = static_cast<size_t>(s.n_layers) * s.ple;
    const size_t mc = std::max({H, qw, ff});
    const size_t nb = static_cast<size_t>(p.nb_max), nh = static_cast<size_t>(s.nh);
    const bool ok =
        p.alloc(nt * H * f, p.d_x, err) && p.alloc(nt * H * f, p.d_h, err) && p.alloc(nt * mc, p.d_X, err) &&
        p.alloc(nt * mc / 16 * f, p.d_XS, err) && p.alloc(nt * mc / 16 * f, p.d_XB, err) &&
        p.alloc(nt * qw * f, p.d_qg, err) && p.alloc(nt * kw * f, p.d_kb, err) && p.alloc(nt * kw * f, p.d_vb, err) &&
        p.alloc(nt * qw * f, p.d_core, err) && p.alloc(nt * H * f, p.d_mix, err) &&
        p.alloc(nt * 2 * ff * f, p.d_gu, err) && p.alloc(nt * ff, p.d_x2, err) &&
        p.alloc(nt * ff / 16 * f, p.d_x2s, err) && p.alloc(nt * ff / 16 * f, p.d_x2b, err) &&
        p.alloc(nt * lp * f, p.d_pin, err) && p.alloc(nt * lp * f, p.d_prow, err) &&
        p.alloc(nt * static_cast<size_t>(s.ple) * f, p.d_pg, err) &&
        p.alloc(nt * nh * nb * 512 * f, p.d_po, err) && p.alloc(nt * nh * nb * f, p.d_pm, err) &&
        p.alloc(nt * nh * nb * f, p.d_pl, err) && p.alloc(Impl::kJobs * 32, p.d_jobs, err) &&
        p.alloc(nt * f, p.d_ymap, err) && p.alloc(nt * static_cast<size_t>(s.vocab) * f, p.d_logits, err) &&
        p.alloc(4096, p.d_flag, err);
    if (!ok) return false;
    auto *ym = reinterpret_cast<int32_t *>(p.d_ymap.host);
    for (size_t i = 0; i < nt; ++i) ym[i] = static_cast<int32_t>(i);
    std::memset(p.d_flag.host, 0, 4096);
    p.wait_sleep = std::getenv("GEODE_GPU8_WAIT") != nullptr;
    p.gs = s;
    p.gs.max_tokens = mt;
    p.gemma = true;
    p.T = 1;
    return true;
}

float *Gpu8::gemma_x() { return reinterpret_cast<float *>(p_->d_x.host); }
float *Gpu8::gemma_ple() { return reinterpret_cast<float *>(p_->d_prow.host); }

bool Gpu8::gemma_step(const std::vector<GemmaLayer> &layers, const QView &ple_model, const float *ple_proj_norm,
                      const float *final_norm, const QView &lm_head, int pos, const std::vector<uint8_t *> &kv,
                      float *logits, int first, float *hidden, std::string &err) {
    Impl &p = *p_;
    if (!p.gemma) {
        err = "GPU: Gemma's answer before setup_gemma";
        return false;
    }
    const auto &s = p.gs;
    const Impl::Piece &P = p.params;
    const int H = s.hidden, T = p.T, LP = s.n_layers * s.ple, nh = s.nh, nkv = s.nkv, G = nh / nkv;
    const size_t tg = T > 1 ? static_cast<size_t>(T) : 0;   // the tokens' dimension (one token: a 1-D launch)
    if (static_cast<size_t>((pos + T - 1) / s.kv_chunk) >= kv.size()) {
        err = "GPU: the KV cache has no chunk for this position";
        return false;
    }
    const int s0 = 2 * s.n_attn * s.kv_chunk * s.kv_row / 4;   // the scales' first float in a chunk
    const float mix = 1.0f / std::sqrt(2.0f);
    p.cur = "gemma";
    // the per-layer inputs: ple_model(x), each layer's normed and joined with the table's row
    bool ok = p.go("g_quant", static_cast<size_t>(H / 16), tg, 0, 0, err, p.d_x.mem, H, p.d_X.mem, p.d_XS.mem, p.d_XB.mem) &&
              p.gemv_many({{&ple_model, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_pin, H, LP, err) &&
              p.go("g_ple_in", static_cast<size_t>(LP), tg, static_cast<size_t>(s.ple), 0, err, p.d_pin.mem,
                   p.d_prow.mem, LP, P.mem, p.at(P, ple_proj_norm, 4), s.eps, s.ple_proj, s.ple_tok, mix);
    for (size_t li = 0; li < layers.size() && ok; ++li) {
        const GemmaLayer &L = layers[li];
        const int hd = L.hd, qw = nh * hd, kw = nkv * hd, ff = L.ffn;
        if (hd != 256 && hd != 512) {
            err = "GPU: a head size the attention does not take";
            return false;
        }
        ok = p.go("g_norm_q", 256, tg, 256, 0, err, p.d_x.mem, P.mem, p.at(P, L.in_norm, 4), p.d_h.mem, H, s.eps, p.d_X.mem,
                  p.d_XS.mem, p.d_XB.mem) &&
             p.gemv_many({{&L.q, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_qg, H, qw, err) &&
             p.go("g_head", static_cast<size_t>(qw), tg, static_cast<size_t>(hd), 0, err, p.d_qg.mem, qw, P.mem,
                  p.at(P, L.q_norm, 4), 1, s.eps, L.pairs, pos, p.at(P, L.freq, 4));
        if (ok && L.own_kv) {
            ok = p.gemv_many({{&L.k, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_kb, H, kw, err) &&
                 p.gemv_many({{&L.v, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_vb, H, kw, err) &&
                 p.go("g_head", static_cast<size_t>(kw), tg, static_cast<size_t>(hd), 0, err, p.d_kb.mem, kw, P.mem,
                      p.at(P, L.k_norm, 4), 1, s.eps, L.pairs, pos, p.at(P, L.freq, 4)) &&
                 p.go("g_head", static_cast<size_t>(kw), tg, static_cast<size_t>(hd), 0, err, p.d_vb.mem, kw, P.mem, 0, 0,
                      s.eps, 0, pos, 0);
            // each token's K and V into its position's chunk (a batch may straddle chunks)
            for (int t = 0; t < T && ok; ++t) {
                const int pt = pos + t, in_chunk = pt % s.kv_chunk;
                const Impl::Piece *cp = p.find(kv[static_cast<size_t>(pt / s.kv_chunk)]);
                if (!cp || cp->host != kv[static_cast<size_t>(pt / s.kv_chunk)]) {
                    err = "GPU: a KV chunk outside shared memory";
                    return false;
                }
                const int ko = ((0 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * s.kv_row;
                const int vo = ((1 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * s.kv_row;
                const int ksi = s0 + ((0 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * nkv;
                const int vsi = s0 + ((1 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * nkv;
                ok = p.go("g_kv_store8", static_cast<size_t>(kw), 0, static_cast<size_t>(hd), 0, err, p.d_kb.mem,
                          p.d_vb.mem, cp->mem, ko, vo, cp->mem, ksi, vsi, t * kw);
            }
        }
        if (!ok) break;
        // the positions the tokens see (a sliding layer: from the first token's window on): within two chunks and
        // 2048 positions, a work group a head and token over all of them (g_attn_small); else flash decoding - the
        // blocks of 128 positions chunk by chunk, then merged
        const int nb = (pos + T + 127) / 128;
        const int lo = L.full ? 0 : std::max(0, pos + 1 - s.window), blo = lo / 128;
        const int cf = lo / s.kv_chunk, cl = (pos + T - 1) / s.kv_chunk;
        if (cl - cf <= 1 && pos + T - lo <= 2048) {
            const Impl::Piece *k0 = p.find(kv[static_cast<size_t>(cf)]), *k1 = p.find(kv[static_cast<size_t>(cl)]);
            if (!k0 || !k1) {
                err = "GPU: a KV chunk outside shared memory";
                return false;
            }
            ok = p.go("g_attn_small", static_cast<size_t>(nh) * hd, tg, static_cast<size_t>(hd), 0, err, p.d_qg.mem, qw,
                      k0->mem, k1->mem, L.slot * s.kv_chunk * s.kv_row, (s.n_attn + L.slot) * s.kv_chunk * s.kv_row,
                      s.kv_row, s0 + L.slot * s.kv_chunk * nkv, s0 + (s.n_attn + L.slot) * s.kv_chunk * nkv, nkv, G,
                      cf * s.kv_chunk, s.kv_chunk, pos + 1, L.full ? 0 : s.window, p.d_core.mem, qw);
        }
        const bool flash = !(cl - cf <= 1 && pos + T - lo <= 2048);
        // (flash decoding: a work group a KV head's G query heads, as many tokens as local memory takes)
        const int hpg = G, pt = hd == 256 ? 2 : 1;
        for (int c = 0, p0 = 0; flash && p0 < pos + T && ok; ++c, p0 += s.kv_chunk) {
            if (p0 + s.kv_chunk <= blo * 128) continue;
            const Impl::Piece *ck = p.find(kv[static_cast<size_t>(c)]);
            if (!ck) {
                err = "GPU: a KV chunk outside shared memory";
                return false;
            }
            const int b0 = std::max(blo, p0 / 128), b1 = (std::min(pos + T, p0 + s.kv_chunk) + 127) / 128;
            if (b1 <= b0) continue;
            for (int tb = 0; tb < T && ok; tb += pt)
                ok = p.go("g_attn8", static_cast<size_t>(b1 - b0) * hd, static_cast<size_t>(nh / hpg), static_cast<size_t>(hd),
                          1, err, p.d_qg.mem, qw, ck->mem, L.slot * s.kv_chunk * s.kv_row,
                          (s.n_attn + L.slot) * s.kv_chunk * s.kv_row, s.kv_row, ck->mem, s0 + L.slot * s.kv_chunk * nkv,
                          s0 + (s.n_attn + L.slot) * s.kv_chunk * nkv, nkv, p0, b0, pos + 1, L.full ? 0 : s.window,
                          p.d_po.mem, p.d_pm.mem, p.d_pl.mem, nb, nh, tb, pt, hpg, G);
        }
        ok = ok &&
             (!flash || p.go("g_attn_merge", static_cast<size_t>(qw), tg, static_cast<size_t>(hd), 0, err, p.d_po.mem,
                             p.d_pm.mem, p.d_pl.mem, nb, blo, p.d_core.mem, qw, nh)) &&
             p.go("g_quant", static_cast<size_t>(qw / 16), tg, 0, 0, err, p.d_core.mem, qw, p.d_X.mem, p.d_XS.mem,
                  p.d_XB.mem) &&
             p.gemv_many({{&L.o, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_mix, qw, H, err) &&
             p.go("g_post", 256, tg, 256, 0, err, p.d_x.mem, p.d_mix.mem, P.mem, p.at(P, L.post_attn, 4),
                  p.at(P, L.pre_ff, 4), p.d_h.mem, H, s.eps, p.d_X.mem, p.d_XS.mem, p.d_XB.mem) &&
             // the MLP: down(gelu(gate) * up)
             p.gemv_many({{&L.gate, 0, 0}, {&L.up, 0, ff}}, p.d_X, p.d_XS, p.d_XB, p.d_gu, H, 2 * ff, err) &&
             p.go("g_gelu_q", static_cast<size_t>(ff / 16), tg, 0, 0, err, p.d_gu.mem, ff, p.d_x2.mem, p.d_x2s.mem,
                  p.d_x2b.mem) &&
             p.gemv_many({{&L.down, 0, 0}}, p.d_x2, p.d_x2s, p.d_x2b, p.d_mix, ff, H, err) &&
             p.go("g_post_q", 256, tg, 256, 0, err, p.d_x.mem, p.d_mix.mem, P.mem, p.at(P, L.post_ff, 4), H, s.eps,
                  p.d_X.mem, p.d_XS.mem, p.d_XB.mem) &&
             // the per-layer input: x = (x + norm(proj(gelu(gate(x)) * ple_l))) * scalar
             p.gemv_many({{&L.ple_gate, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_pg, H, s.ple, err) &&
             p.go("g_ple_mid", static_cast<size_t>(s.ple / 16), tg, 0, 0, err, p.d_pg.mem, s.ple, p.d_pin.mem, LP,
                  static_cast<int>(li) * s.ple, p.d_x2.mem, p.d_x2s.mem, p.d_x2b.mem) &&
             p.gemv_many({{&L.ple_proj, 0, 0}}, p.d_x2, p.d_x2s, p.d_x2b, p.d_h, s.ple, H, err) &&
             p.go("g_ple_out", 256, tg, 256, 0, err, p.d_x.mem, p.d_h.mem, P.mem, p.at(P, L.ple_norm, 4), L.scalar, H, s.eps);
        if (ok && (li % 8) == 7) p.cl->clFlush(p.q1);   // (the GPU starts while the rest is enqueued)
    }
    // the final norm, lm_head, the soft-cap
    ok = ok &&
         p.go("g_norm_q", 256, tg, 256, 0, err, p.d_x.mem, P.mem, p.at(P, final_norm, 4), p.d_h.mem, H, s.eps, p.d_X.mem,
              p.d_XS.mem, p.d_XB.mem) &&
         (!logits || (p.gemv_many({{&lm_head, 0, 0}}, p.d_X, p.d_XS, p.d_XB, p.d_logits, H, s.vocab, err) &&
                      (s.softcap <= 0.0f || p.go("g_softcap", static_cast<size_t>(T) * s.vocab, 0, 0, 0, err,
                                                   p.d_logits.mem, s.softcap))));
    // the end polled for (GEODE_GPU8_WAIT=1: asleep in the driver - a waiting CPU lets the DDR's clock fall, the GEMVs
    // with it), then the finish that makes every write the host's (the tag may arrive before the logits' lines)
    p.seq = p.seq % 0x7fffff + 1;
    if (ok && !p.wait_sleep && p.go("g_done", 1, 0, 1, 0, err, p.d_flag.mem, p.seq)) {
        p.cl->clFlush(p.q1);
        const auto *vf = reinterpret_cast<const volatile int32_t *>(p.d_flag.host);
        const double t0 = now_seconds();
        for (int spins = 0; vf[0] != p.seq; ++spins)
            if (spins % 4096 == 4095 && now_seconds() - t0 > 10.0) break;   // (the finish below says why)
    }
    const cl_int e = p.cl->clFinish(p.q1);
    p.jcur = 0;
    p.cur = "batch";
    if (p.dprof) p.collect();
    if (!ok) return false;
    if (e != CL_SUCCESS) {
        err = std::string("GPU: finish: ") + cl::error_name(e);
        return false;
    }
    if (logits && first < T)
        std::memcpy(logits, p.d_logits.host + static_cast<size_t>(first) * s.vocab * 4,
                    static_cast<size_t>(T - first) * static_cast<size_t>(s.vocab) * 4);
    if (hidden) std::memcpy(hidden, p.d_h.host, static_cast<size_t>(T) * H * 4);
    return true;
}

// ---- a Gemma prompt's layers on the GPU

bool Gpu8::setup_gemma_prompt(int cap, std::string &err) {
    Impl &p = *p_;
    if (!p.gemma) {
        err = "GPU: the generating programs first (setup_gemma)";
        return false;
    }
    const auto &s = p.gs;
    const size_t nt = round_up(static_cast<size_t>(std::max(cap, 1)), kT);
    if (p.gp_cap >= static_cast<int>(nt)) return true;
    // the batch program's kernels that write int8 tiles (the rest are the one-token program's: token-independent)
    for (const char *name : {"g_quant", "g_norm_q", "g_post", "g_post_q", "g_gelu_q", "g_ple_mid", "gemm1", "gemm1x"}) {
        if (p.kb8.count(name)) continue;
        cl_kernel k = p.prog.kernel(name, err);
        if (!k) return false;
        p.kb8[name] = k;
    }
    const size_t f = 4, H = static_cast<size_t>(s.hidden), qw = static_cast<size_t>(s.nh) * s.hd_max;
    const size_t kw = static_cast<size_t>(s.nkv) * s.hd_max, ff = static_cast<size_t>(s.ffn_max);
    const size_t P = static_cast<size_t>(s.ple), LP = static_cast<size_t>(s.n_layers) * P;
    const size_t mc = std::max({H, qw, ff, P});   // the widest input a product takes
    p.gp_parts = (s.max_ctx + s.kv_chunk - 1) / s.kv_chunk + 1;
    p.gp_jobs_n = s.n_layers * 10 * static_cast<int>(nt / kT) + 64;
    const size_t parts = static_cast<size_t>(Impl::kGpGroup) * s.nh * p.gp_parts;
    const bool ok =
        p.alloc(nt * H * f, p.gp_x, err) && p.alloc(nt * H * f, p.gp_h, err) && p.alloc(nt * qw * f, p.gp_q, err) &&
        p.alloc(nt * kw * f, p.gp_k, err) && p.alloc(nt * kw * f, p.gp_v, err) && p.alloc(nt * qw * f, p.gp_core, err) &&
        p.alloc(nt * H * f, p.gp_mix, err) && p.alloc(nt * 2 * ff * f, p.gp_gu, err) &&
        p.alloc(nt * LP * f, p.gp_pin, err) && p.alloc(nt * P * f, p.gp_pg, err) && p.alloc(nt * mc, p.gp_X, err) &&
        p.alloc(nt * mc / 16 * f, p.gp_XS, err) && p.alloc(nt * mc / 16 * f, p.gp_XB, err) &&
        p.alloc(nt * f, p.gp_ymap, err) && p.alloc(static_cast<size_t>(p.gp_jobs_n) * sizeof(Job), p.gp_jobs, err) &&
        p.alloc(parts * static_cast<size_t>(s.hd_max) * f, p.gp_po, err) && p.alloc(parts * f, p.gp_pm, err) &&
        p.alloc(parts * f, p.gp_pl, err);
    if (!ok) return false;
    // (the tiles a launch reads past the last token: zeros, never NaN - their rows go nowhere, ymap -1)
    std::memset(p.gp_X.host, 0, nt * mc);
    std::memset(p.gp_XS.host, 0, nt * mc / 16 * f);
    std::memset(p.gp_XB.host, 0, nt * mc / 16 * f);
    // the flash-style attention where its work groups of 256 fit (its registers: the compiler may allow fewer);
    // GEODE_GEMMA_PF=0: g_attn_small's
    const char *g1 = std::getenv("GEODE_GP_GEMM1");
    p.gp_gemm1 = !(g1 && std::atoi(g1) == 0);
    // the activation tiles as an image too (GEODE_GP_XI=0: not)
    const char *xi = std::getenv("GEODE_GP_XI");
    if (p.gp_Xi) p.cl->clReleaseMemObject(p.gp_Xi), p.gp_Xi = nullptr;
    if (!(xi && std::atoi(xi) == 0) && p.dev.images && p.dev.image_max_buffer >= nt * mc / 16) {
        cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        desc.image_width = nt * mc / 16;
        desc.buffer = p.gp_X.mem;
        cl_int e = CL_SUCCESS;
        p.gp_Xi = p.cl->clCreateImage(p.dev.context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
    }
    const char *pf = std::getenv("GEODE_GEMMA_PF");
    p.gp_pf = !(pf && std::atoi(pf) == 0);
    for (const char *name : {"g_attn_pf256", "g_attn_pf512"}) {
        size_t wg = 0;
        auto it = p.k1.find(name);
        if (it == p.k1.end() || p.cl->clGetKernelWorkGroupInfo(it->second, p.dev.id, CL_KERNEL_WORK_GROUP_SIZE,
                                                               sizeof(wg), &wg, nullptr) != CL_SUCCESS || wg < 256)
            p.gp_pf = false;
    }
    p.gp_cap = static_cast<int>(nt);
    return true;
}

float *Gpu8::gemma_prompt_x() { return reinterpret_cast<float *>(p_->gp_x.host); }
float *Gpu8::gemma_prompt_ple() { return reinterpret_cast<float *>(p_->gp_pin.host); }

bool Gpu8::gemma_prompt(const std::vector<GemmaLayer> &layers, int pos, int n, const std::vector<uint8_t *> &kv,
                        std::string &err) {
    Impl &p = *p_;
    if (!p.gemma || n < 1 || n > p.gp_cap) {
        err = "GPU: a prompt chunk it was not set up for";
        return false;
    }
    const auto &s = p.gs;
    if (static_cast<size_t>((pos + n - 1) / s.kv_chunk) >= kv.size()) {
        err = "GPU: the KV cache has no chunk for this position";
        return false;
    }
    const Impl::Piece &P = p.params;
    const int H = s.hidden, nh = s.nh, nkv = s.nkv, G = nh / nkv, LP = s.n_layers * s.ple;
    const int s0 = 2 * s.n_attn * s.kv_chunk * s.kv_row / 4;   // the scales' first float in a chunk
    const size_t tn = static_cast<size_t>(n);
    const int T0 = p.T;
    p.T = 1;   // (go: the one-token program's kernels - the ones used here take any number of tokens)
    auto *ym = reinterpret_cast<int32_t *>(p.gp_ymap.host);
    for (int i = 0; i < p.gp_cap; ++i) ym[i] = i < n ? i : -1;
    p.gp_jcur = 0;
    p.cur = "gemma prompt";
    bool ok = true;
    for (size_t li = 0; li < layers.size() && ok; ++li) {
        const GemmaLayer &L = layers[li];
        const int hd = L.hd, qw = nh * hd, kw = nkv * hd, ff = L.ffn;
        if (hd != 256 && hd != 512) {
            err = "GPU: a head size the attention does not take";
            ok = false;
            break;
        }
        ok = p.bgo("g_norm_q", 256, tn, 256, 1, err, p.gp_x.mem, P.mem, p.at(P, L.in_norm, 4), p.gp_h.mem, H, s.eps,
                   p.gp_X.mem, p.gp_XS.mem, p.gp_XB.mem) &&
             p.gp_gemm(L.q, n, p.gp_q, qw, 0, err) &&
             p.go("g_head", static_cast<size_t>(qw), tn, static_cast<size_t>(hd), 1, err, p.gp_q.mem, qw, P.mem,
                  p.at(P, L.q_norm, 4), 1, s.eps, L.pairs, pos, p.at(P, L.freq, 4));
        if (ok && L.own_kv) {
            ok = p.gp_gemm(L.k, n, p.gp_k, kw, 0, err) && p.gp_gemm(L.v, n, p.gp_v, kw, 0, err) &&
                 p.go("g_head", static_cast<size_t>(kw), tn, static_cast<size_t>(hd), 1, err, p.gp_k.mem, kw, P.mem,
                      p.at(P, L.k_norm, 4), 1, s.eps, L.pairs, pos, p.at(P, L.freq, 4)) &&
                 p.go("g_head", static_cast<size_t>(kw), tn, static_cast<size_t>(hd), 1, err, p.gp_v.mem, kw, P.mem, 0,
                      0, s.eps, 0, pos, 0);
            // the tokens' K and V into their positions' chunks, a run of a chunk a launch
            for (int t0 = 0; t0 < n && ok;) {
                const int pt = pos + t0, ch = pt / s.kv_chunk, in_chunk = pt % s.kv_chunk;
                const int cnt = std::min(n - t0, s.kv_chunk - in_chunk);
                const Impl::Piece *cp = p.find(kv[static_cast<size_t>(ch)]);
                if (!cp || cp->host != kv[static_cast<size_t>(ch)]) {
                    err = "GPU: a KV chunk outside shared memory";
                    ok = false;
                    break;
                }
                const int ko = ((0 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * s.kv_row;
                const int vo = ((1 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * s.kv_row;
                const int ksi = s0 + ((0 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * nkv;
                const int vsi = s0 + ((1 * s.n_attn + L.slot) * s.kv_chunk + in_chunk) * nkv;
                ok = p.go("g_kv_store8n", static_cast<size_t>(kw), static_cast<size_t>(cnt), static_cast<size_t>(hd), 1,
                          err, p.gp_k.mem, p.gp_v.mem, cp->mem, ko, vo, cp->mem, ksi, vsi, t0 * kw, kw, s.kv_row, nkv);
                t0 += cnt;
            }
        }
        if (!ok) break;
        // the attention: every token's positions within two chunks and 2048 - one launch; else a group of tokens at a
        // time, chunk by chunk, joined
        const int kb = L.slot * s.kv_chunk * s.kv_row, vb = (s.n_attn + L.slot) * s.kv_chunk * s.kv_row;
        const int ksb = s0 + L.slot * s.kv_chunk * nkv, vsb = s0 + (s.n_attn + L.slot) * s.kv_chunk * nkv;
        const int window = L.full ? 0 : s.window;
        const int lo = L.full ? 0 : std::max(0, pos + 1 - s.window);
        const int cf = lo / s.kv_chunk, cl = (pos + n - 1) / s.kv_chunk;
        // flash-style (g_attn_pf*): a work group a KV head and TT tokens - R query rows, each K and V row read once
        // for them; else a work group a head and token (g_attn_small, g_attn_chunk)
        const int R = hd == 256 ? 32 : 16;
        const bool pf = p.gp_pf && R % G == 0;
        const size_t TT = static_cast<size_t>(R / G);
        const char *pfk = hd == 256 ? "g_attn_pf256" : "g_attn_pf512";
        if (cl - cf <= 1 && (pf || pos + n - lo <= 2048)) {
            const Impl::Piece *k0 = p.find(kv[static_cast<size_t>(cf)]), *k1 = p.find(kv[static_cast<size_t>(cl)]);
            if (!k0 || !k1) {
                err = "GPU: a KV chunk outside shared memory";
                ok = false;
                break;
            }
            if (pf) {
                ok = p.go(pfk, static_cast<size_t>(256) * nkv, (tn + TT - 1) / TT, 256, 1, err, p.gp_q.mem, qw,
                          k0->mem, k1->mem, kb, vb, s.kv_row, ksb, vsb, nkv, G, cf * s.kv_chunk, s.kv_chunk, pos + 1,
                          window, p.gp_core.mem, qw, n, 0, 0, INT32_MAX, p.gp_po.mem, p.gp_pm.mem, p.gp_pl.mem, 0, 0);
            } else {
                ok = p.go("g_attn_small", static_cast<size_t>(nh) * hd, tn, static_cast<size_t>(hd), 1, err,
                          p.gp_q.mem, qw, k0->mem, k1->mem, kb, vb, s.kv_row, ksb, vsb, nkv, G, cf * s.kv_chunk,
                          s.kv_chunk, pos + 1, window, p.gp_core.mem, qw);
            }
        } else {
            for (int t0 = 0, cnt = 0; t0 < n && ok; t0 += cnt) {
                const int glo = L.full ? 0 : std::max(0, pos + t0 + 1 - s.window);
                cnt = std::min(Impl::kGpGroup, n - t0);
                if (pf) {   // (the buffers: kGpGroup tokens x gp_parts; a pair of chunks a part - fewer parts)
                    const int most = ((pos + n - 1) / s.kv_chunk - glo / s.kv_chunk + 2) / 2, tt = R / G;
                    cnt = std::min(n - t0, std::max(Impl::kGpGroup * p.gp_parts / most / tt * tt, tt));
                }
                const int gf = glo / s.kv_chunk, gl = (pos + t0 + cnt - 1) / s.kv_chunk;
                const int parts = pf ? (gl - gf + 2) / 2 : gl - gf + 1;   // (flash-style: two chunks a part)
                if (parts > p.gp_parts) {
                    err = "GPU: a prompt's attention past its parts";
                    ok = false;
                    break;
                }
                for (int c = gf; c <= gl && ok && pf; c += 2) {
                    const Impl::Piece *k0 = p.find(kv[static_cast<size_t>(c)]);
                    const Impl::Piece *k1 = c < gl ? p.find(kv[static_cast<size_t>(c + 1)]) : k0;
                    if (!k0 || !k1) {
                        err = "GPU: a KV chunk outside shared memory";
                        ok = false;
                        break;
                    }
                    ok = p.go(pfk, static_cast<size_t>(256) * nkv, (static_cast<size_t>(cnt) + TT - 1) / TT, 256, 1,
                              err, p.gp_q.mem, qw, k0->mem, k1->mem, kb, vb, s.kv_row, ksb, vsb, nkv, G,
                              c * s.kv_chunk, s.kv_chunk, pos + 1, window, p.gp_core.mem, qw, t0 + cnt, t0,
                              c * s.kv_chunk, (c + 2) * s.kv_chunk, p.gp_po.mem, p.gp_pm.mem, p.gp_pl.mem, (c - gf) / 2,
                              parts);
                }
                for (int c = gf; c <= gl && ok && !pf; ++c) {
                    const Impl::Piece *ck = p.find(kv[static_cast<size_t>(c)]);
                    if (!ck) {
                        err = "GPU: a KV chunk outside shared memory";
                        ok = false;
                        break;
                    }
                    ok = p.go("g_attn_chunk", static_cast<size_t>(nh) * hd, static_cast<size_t>(cnt),
                              static_cast<size_t>(hd), 1, err, p.gp_q.mem, qw, ck->mem, kb, vb, s.kv_row, ksb, vsb, nkv,
                              G, c * s.kv_chunk, s.kv_chunk, pos + 1, window, p.gp_po.mem, p.gp_pm.mem, p.gp_pl.mem,
                              c - gf, parts, t0);
                }
                ok = ok && p.go("g_attn_join", static_cast<size_t>(nh) * hd, static_cast<size_t>(cnt),
                                static_cast<size_t>(hd), 1, err, p.gp_po.mem, p.gp_pm.mem, p.gp_pl.mem, parts,
                                p.gp_core.mem, qw, t0);
            }
        }
        ok = ok &&
             p.bgo("g_quant", static_cast<size_t>(qw / 16), tn, 0, 0, err, p.gp_core.mem, qw, p.gp_X.mem, p.gp_XS.mem,
                   p.gp_XB.mem) &&
             p.gp_gemm(L.o, n, p.gp_mix, H, 0, err) &&
             p.bgo("g_post", 256, tn, 256, 1, err, p.gp_x.mem, p.gp_mix.mem, P.mem, p.at(P, L.post_attn, 4),
                   p.at(P, L.pre_ff, 4), p.gp_h.mem, H, s.eps, p.gp_X.mem, p.gp_XS.mem, p.gp_XB.mem) &&
             // the MLP: down(gelu(gate) * up), gate and up into one row [gate ff | up ff]
             p.gp_gemm(L.gate, n, p.gp_gu, 2 * ff, 0, err) && p.gp_gemm(L.up, n, p.gp_gu, 2 * ff, ff, err) &&
             p.bgo("g_gelu_q", static_cast<size_t>(ff / 16), tn, 0, 0, err, p.gp_gu.mem, ff, p.gp_X.mem, p.gp_XS.mem,
                   p.gp_XB.mem) &&
             p.gp_gemm(L.down, n, p.gp_mix, H, 0, err) &&
             p.bgo("g_post_q", 256, tn, 256, 1, err, p.gp_x.mem, p.gp_mix.mem, P.mem, p.at(P, L.post_ff, 4), H, s.eps,
                   p.gp_X.mem, p.gp_XS.mem, p.gp_XB.mem) &&
             // the per-layer input: x = (x + norm(proj(gelu(gate(x)) * ple_l))) * scalar
             p.gp_gemm(L.ple_gate, n, p.gp_pg, s.ple, 0, err) &&
             p.bgo("g_ple_mid", static_cast<size_t>(s.ple / 16), tn, 0, 0, err, p.gp_pg.mem, s.ple, p.gp_pin.mem, LP,
                   static_cast<int>(li) * s.ple, p.gp_X.mem, p.gp_XS.mem, p.gp_XB.mem) &&
             p.gp_gemm(L.ple_proj, n, p.gp_h, H, 0, err) &&
             p.go("g_ple_out", 256, tn, 256, 1, err, p.gp_x.mem, p.gp_h.mem, P.mem, p.at(P, L.ple_norm, 4), L.scalar, H,
                  s.eps);
        if (ok && (li % 4) == 3) p.cl->clFlush(p.q1);   // (the GPU starts while the rest is enqueued)
    }
    const cl_int e = p.cl->clFinish(p.q1);
    p.T = T0;
    p.cur = "batch";
    if (ok && e != CL_SUCCESS) {
        err = std::string("GPU: a prompt's layers: ") + cl::error_name(e);
        ok = false;
    }
    return ok;
}

}  // namespace geode
