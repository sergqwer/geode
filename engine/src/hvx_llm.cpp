// hvx_llm.cpp - HvxGemm (hvx_llm.hpp): a prompt's 4-bit products split by rows between the v66 DSP's HVX and the CPU.
#include "hvx_llm.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "threads.hpp"

#if defined(__ANDROID__)
#include <dlfcn.h>
#include <unistd.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace geode {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int env_int(const char *name, int dflt) {
    const char *e = std::getenv(name);
    return e && *e ? std::atoi(e) : dflt;
}

const std::array<float, 256> &e4m3s() {
    static const std::array<float, 256> t = [] {
        std::array<float, 256> v{};
        for (int b = 0; b < 256; ++b) v[static_cast<size_t>(b)] = e4m3_value(static_cast<uint8_t>(b));
        return v;
    }();
    return t;
}

// the inputs a pass of the DSP's keeps in L2: the tokens' slice ~128 KB (n 256, hvx_probe q4: 2048 inputs 22-25 G MAC/s,
// 1024 34-38, 512 38-42 - the quad layout 27-33 / 35-41: its turn into tile8 order takes the L2 the slice needs)
// (the quad layout: whole runs of 8 chunks, multiples of 512)
int pass_inputs(int n8, bool quad) {
    const int q = quad ? 512 : 64, e = env_int("GEODE_HVX_KC", 0);
    if (e >= q) return e / q * q;
    return std::clamp((128 << 10) / std::max(8, n8) / q * q, q, 2048);
}

// x[0..8) -> 0 2 4 6 1 3 5 7, len a multiple of 8
void permute8(const int8_t *src, int8_t *dst, int len) {
    int i = 0;
#if defined(__ARM_NEON)
    for (; i + 32 <= len; i += 32) {
        const int8x16x2_t v = vld2q_s8(src + i);   // evens, odds
        const int32x4x2_t z = vzipq_s32(vreinterpretq_s32_s8(v.val[0]), vreinterpretq_s32_s8(v.val[1]));
        vst1q_s8(dst + i, vreinterpretq_s8_s32(z.val[0]));
        vst1q_s8(dst + i + 16, vreinterpretq_s8_s32(z.val[1]));
    }
#endif
    for (; i < len; i += 8) {
        static constexpr int perm[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        for (int j = 0; j < 8; ++j) dst[i + j] = src[i + perm[j]];
    }
}

uint32_t halves(long lo, long hi) {
    return static_cast<uint32_t>(static_cast<uint16_t>(lo)) | static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16;
}

}  // namespace

std::unique_ptr<HvxGemm> HvxGemm::open(std::string &why) {
#if defined(__ANDROID__)
    if (const char *e = std::getenv("GEODE_HVX"); e && !std::strcmp(e, "0")) {
        why = "off (GEODE_HVX=0)";
        return nullptr;
    }
    std::unique_ptr<HvxGemm> g(new HvxGemm());
    const char *path = std::getenv("GEODE_NPU_LIB") ? std::getenv("GEODE_NPU_LIB") : "libgeode_npu.so";
    g->lib_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g->lib_) {   // (an adb tool: beside the executable)
        char exe[512] = {0};
        const ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
        std::string dir = len > 0 ? std::string(exe, static_cast<size_t>(len)) : std::string(".");
        dir = dir.substr(0, dir.find_last_of('/'));
        g->lib_ = dlopen((dir + "/libgeode_npu.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    if (!g->lib_) {
        const char *e = dlerror();   // (once: a second dlerror() is NULL)
        why = std::string("no libgeode_npu.so (") + (e ? e : "") + ")";
        return nullptr;
    }
    const auto sym = [&](const char *name) { return dlsym(g->lib_, name); };
    auto *open_fn = reinterpret_cast<void *(*)(const char *, char *, int)>(sym("gnpu_hvx_open"));
    g->close_ = reinterpret_cast<decltype(g->close_)>(sym("gnpu_hvx_close"));
    g->alloc_ = reinterpret_cast<decltype(g->alloc_)>(sym("gnpu_hvx_alloc"));
    g->free_ = reinterpret_cast<decltype(g->free_)>(sym("gnpu_hvx_free"));
    g->keep_ = reinterpret_cast<decltype(g->keep_)>(sym("gnpu_hvx_keep"));
    g->gemm_ = reinterpret_cast<decltype(g->gemm_)>(sym("gnpu_hvx_q4_gemm"));
    if (!open_fn || !g->close_ || !g->alloc_ || !g->free_ || !g->keep_ || !g->gemm_) {
        why = "libgeode_npu.so has no gnpu_hvx_q4_gemm (an older one)";
        return nullptr;
    }
    std::string dir = ".";
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(open_fn), &info) && info.dli_fname) {
        dir = info.dli_fname;
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    char err[256] = {0};
    g->h_ = open_fn(dir.c_str(), err, static_cast<int>(sizeof err));
    if (!g->h_) {
        why = err;
        return nullptr;
    }
    g->threads_ = std::clamp(env_int("GEODE_HVX_THREADS", 2), 1, 4);
    g->io_cached_ = env_int("GEODE_HVX_IO_CACHED", 1) != 0;
    if (const char *e = std::getenv("GEODE_HVX_SHARE")) g->fixed_share_ = std::clamp(static_cast<float>(std::atof(e)), 0.0f, 1.0f);
    g->thread_ = std::thread([p = g.get()] { p->helper(); });
    return g;
#else
    why = "no DSP here";
    return nullptr;
#endif
}

HvxGemm::~HvxGemm() {
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    for (Buf *b : {&x_, &xs_, &wq_, &y_})
        if (b->p) free_(b->p);
    for (Buf &b : wbufs_) keep_(b.p, b.bytes, 0), free_(b.p);
    if (h_ && close_) close_(h_);
    // (the library stays loaded: the NPU's setup may hold it too)
}

uint8_t *HvxGemm::weights(size_t bytes) {
    void *p = alloc_(bytes, 1);
    if (!p) return nullptr;
    // kept mapped: FastRPC otherwise maps the buffer behind every pointer into it for each call (a 256 MB one: 5-10 ms
    // a call; hvx_probe q4 with PAD_MB=256)
    if (env_int("GEODE_HVX_KEEP", 1)) keep_(p, bytes, 1);
    wbufs_.push_back({p, bytes});
    weight_bytes_ += bytes;
    return static_cast<uint8_t *>(p);
}

void *HvxGemm::grow(Buf &b, size_t bytes) {
    if (b.bytes >= bytes) return b.p;
    if (b.p) free_(b.p);
    bytes = (bytes + (1u << 20) - 1) & ~static_cast<size_t>((1u << 20) - 1);
    b.p = alloc_(bytes, io_cached_ ? 1 : 0);
    b.bytes = b.p ? bytes : 0;
    return b.p;
}

void HvxGemm::hold(const QView &m) {
    // tile8, or quad with whole runs of 8 chunks (its groups of 4 rows 128-byte aligned: cols % 512)
    if (!m.data || !m.quad || m.rows % 32 || m.cols % 64 || (!m.tile8 && m.cols % 512)) return;
    Mat M;
    const int64_t k = m.cols, nb = k / 16;
    M.tile = static_cast<size_t>(k) * 18;
    M.quad = !m.tile8;
    M.wmax.assign(static_cast<size_t>(m.rows), 0.0f);
    const auto &e4 = e4m3s();
    for (int64_t r = 0; r < m.rows; ++r) {
        const uint8_t *t = m.data + static_cast<size_t>(r / 32) * M.tile;
        float mx = 0.0f;
        for (int64_t b = 0; b < nb; ++b) mx = std::max(mx, std::fabs(e4[scale_at(M, t, k, r % 32, b)]));
        M.wmax[static_cast<size_t>(r)] = mx;
    }
    mats_[m.data] = std::move(M);
}

int64_t HvxGemm::rows_for(const QView &m, int n) const {
    if (off_ || n < kMinTokens || m.rows < kMinRows || m.cols < kMinCols || (m.type != DType::Q4I && m.type != DType::Q4) ||
        !holds(m))
        return 0;
    float s = fixed_share_;
    if (s < 0.0f) {
        const auto it = share_.find({m.rows, m.cols});
        s = it == share_.end() ? 0.5f : it->second;
    }
    return std::min(m.rows, static_cast<int64_t>(s * static_cast<float>(m.rows)) / 32 * 32);
}

bool HvxGemm::prepare(const QView &m, int64_t r, int n, const QAct *const *acts, bool same_x, const Par &par) {
    t_prep0_ = now_s();
    const auto it = mats_.find(m.data);
    if (it == mats_.end() || r <= 0) return false;
    Mat &M = it->second;
    const int k = static_cast<int>(m.cols), n8 = (n + 7) / 8 * 8, P = k / 32, nb = k / 16, kc = pass_inputs(n8, M.quad);
    m_ = &m, M_ = &M, Mw_ = &M, r_ = r, n_ = n, kc_ = kc;
    auto *x = static_cast<int8_t *>(grow(x_, static_cast<size_t>(n8) * k));
    auto *xs = static_cast<int32_t *>(grow(xs_, static_cast<size_t>(n8) * P * 8));
    auto *wq = static_cast<int32_t *>(grow(wq_, static_cast<size_t>(r) * P * 4));
    if (!x || !xs || !wq || !grow(y_, static_cast<size_t>(n8) * static_cast<size_t>(r) * 4)) return false;
    // the tokens: their int8 values pass by pass (each group of 8 permuted), per pair of blocks -off sum(x) and the
    // blocks' scale ratios (Q13 of the token's largest) - geode_hvx.idl. Kept for the next matrix of the same input.
    if (!same_x || x_k_ != k || x_n8_ != n8 || x_kc_ != kc || x_type_ != m.type || x_aq_ != M.aq_bits) {
        smax_.assign(static_cast<size_t>(n8), 0.0f);
        const int off = m.type == DType::Q4I ? 16 : 15;
        par(n8, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                const QAct *a = t < n ? acts[t] : nullptr;
                float sm = 0.0f;
                if (a)
                    for (int b = 0; b < nb; ++b) sm = std::max(sm, a->scale[static_cast<size_t>(b)]);
                smax_[static_cast<size_t>(t)] = sm;
                const float inv = sm > 0.0f ? std::ldexp(1.0f, M.aq_bits) / sm : 0.0f;
                for (int k0 = 0; k0 < k; k0 += kc) {
                    const int kn = std::min(kc, k - k0);
                    int8_t *xd = x + static_cast<size_t>(k0) * n8 + static_cast<size_t>(t) * kn;
                    int32_t *sd = xs + static_cast<size_t>(k0 / 16) * n8 + static_cast<size_t>(t) * (kn / 16);
                    if (!a) {
                        std::memset(xd, 0, static_cast<size_t>(kn));
                        std::memset(sd, 0, static_cast<size_t>(kn / 16) * 4);
                        continue;
                    }
                    permute8(a->q.data() + k0, xd, kn);
                    for (int p = 0; p < kn / 32; ++p) {
                        const size_t b = static_cast<size_t>(k0 / 16 + 2 * p);
                        sd[2 * p] = static_cast<int32_t>(halves(-off * a->bsum[b], -off * a->bsum[b + 1]));
                        sd[2 * p + 1] = static_cast<int32_t>(halves(std::lrint(a->scale[b] * inv), std::lrint(a->scale[b + 1] * inv)));
                    }
                }
            }
        });
        x_k_ = k, x_n8_ = n8, x_kc_ = kc, x_type_ = m.type, x_aq_ = M.aq_bits;
    }
    // the rows' ratios (Q15 of each row's largest |scale|), per tile and pair of blocks: rows [0, r) only
    const auto &e4 = e4m3s();
    par(r / 32, [&](int64_t T0, int64_t T1) {
        for (int64_t T = T0; T < T1; ++T) {
            const uint8_t *tl = m.data + static_cast<size_t>(T) * M.tile;
            float inv[32];
            for (int l = 0; l < 32; ++l) {
                const float w = M.wmax[static_cast<size_t>(T) * 32 + static_cast<size_t>(l)];
                inv[l] = w > 0.0f ? 32767.0f / w : 0.0f;
            }
            int32_t *o = wq + static_cast<size_t>(T) * P * 32;
            for (int p = 0; p < P; ++p) {
                int32_t *row = o + static_cast<size_t>(p) * 32;
                for (int l = 0; l < 32; ++l) {
                    const float s0 = e4[scale_at(M, tl, k, l, 2 * p)], s1 = e4[scale_at(M, tl, k, l, 2 * p + 1)];
                    row[l] = static_cast<int32_t>(halves(std::lrint(std::clamp(s0 * inv[l], -32767.0f, 32767.0f)),
                                                         std::lrint(std::clamp(s1 * inv[l], -32767.0f, 32767.0f))));
                }
            }
        }
    });
    prep_s_ += now_s() - t_prep0_;
    return true;
}

void HvxGemm::start() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        has_job_ = true, done_ = false;
    }
    cv_.notify_all();
}

HvxGemm::Finish HvxGemm::finish(double cpu_s, std::string &err) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return done_; });
    if (err_) {
        off_ = true;
        note_ = err = "the DSP's product failed (" + std::to_string(err_) + ")";
        return kFailed;
    }
    // (2^30: a pass's or the total's sum within a factor 2 of int32's limit - :sat may have clipped one)
    if (amax_ >= (1 << 30)) {
        ++redone_;
        if (Mw_->aq_bits > 7) Mw_->aq_bits -= 2;   // (this matrix's next products with 4x the headroom)
        return kRedo;
    }
    const QView &m = *m_;
    const double kn = static_cast<double>(m.cols) * n_;
    ++calls_;
    dsp_s_ += static_cast<double>(us_) * 1e-6, call_s_ += t_call_;
    macs_ += static_cast<double>(r_) * kn, cpu_macs_ += static_cast<double>(m.rows - r_) * kn, cpu_s_ += cpu_s;
    // the share both finish together with, by the rates just measured (the call against the CPU's rows)
    if (fixed_share_ < 0.0f && t_call_ > 0.0 && cpu_s > 0.0 && r_ > 0 && r_ < m.rows) {
        const double rd = static_cast<double>(r_) / t_call_, rc = static_cast<double>(m.rows - r_) / cpu_s;
        float &s = share_.try_emplace({m.rows, m.cols}, 0.5f).first->second;
        s = std::clamp(0.6f * s + 0.4f * static_cast<float>(rd / (rd + rc)), 0.05f, 0.95f);
    }
    if (env_int("GEODE_HVX_DEBUG", 0) >= 2 && calls_ <= 40)
        std::fprintf(stderr, "hvx: %lldx%lld r %lld n %d kc %d: call %.2f ms (DSP %.2f), the CPU's rows %.2f ms\n",
                     static_cast<long long>(m.rows), static_cast<long long>(m.cols), static_cast<long long>(r_), n_, kc_,
                     t_call_ * 1e3, us_ * 1e-3, cpu_s * 1e3);
    return kDone;
}

void HvxGemm::helper() {
#if defined(__ANDROID__)
    // on a big core beside a pool thread (the call mostly waits in the driver): from an A55 a call cost 7.5-8 ms more
    // than the DSP's time, from an A76 2.7-3.9 (the driver's and FastRPC's own work on the calling core; the Redmi,
    // 256-token chunks). GEODE_HVX_CPU picks the core.
    std::vector<int> cpus = big_cores();
    if (const int c = env_int("GEODE_HVX_CPU", -1); c >= 0) cpus = {c};
    else if (!cpus.empty()) cpus = {cpus.back()};
    if (!cpus.empty()) pin_current_thread(cpus);
#endif
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return has_job_ || quit_; });
        if (quit_) return;
        has_job_ = false;
        lk.unlock();
        const QView &m = *m_;
        const int k = static_cast<int>(m.cols), n8 = (n_ + 7) / 8 * 8, r = static_cast<int>(r_);
        long long us = 0;
        int amax = 0;
        const double t0 = now_s();
        const int e = gemm_(h_, m.data, static_cast<size_t>(r / 32) * M_->tile, r, k, n8, static_cast<const int8_t *>(x_.p),
                            static_cast<const int32_t *>(xs_.p), static_cast<const int32_t *>(wq_.p),
                            static_cast<int32_t *>(y_.p), threads_, kc_, M_->quad ? 1 : 0, &amax, &us);
        const double t1 = now_s();
        lk.lock();
        err_ = e, amax_ = amax, us_ = us, t_call_ = t1 - t0, done_ = true;
        cv_.notify_all();
    }
}

void HvxGemm::results(float *const *ys, const Par &par) {
    const double t0 = now_s();
    const QView &m = *m_;
    const int n = n_, n8 = (n + 7) / 8 * 8;
    const int32_t *y = static_cast<const int32_t *>(y_.p);
    // y [tiles][n8][32] -> ys[t][row]: times the row's and the token's largest scales (q4i: the level doubled)
    const float f = m.global * std::ldexp(1.0f, -M_->aq_bits - (m.type == DType::Q4I ? 1 : 0));
    par(r_ / 32, [&](int64_t T0, int64_t T1) {
        for (int64_t T = T0; T < T1; ++T) {
            const float *wm = M_->wmax.data() + static_cast<size_t>(T) * 32;
            for (int t = 0; t < n; ++t) {
                const int32_t *yr = y + (static_cast<size_t>(T) * n8 + static_cast<size_t>(t)) * 32;
                float *out = ys[t] + static_cast<size_t>(T) * 32;
                const float a = smax_[static_cast<size_t>(t)] * f;
                for (int l = 0; l < 32; ++l) out[l] = static_cast<float>(yr[l]) * wm[l] * a;
            }
        }
    });
    out_s_ += now_s() - t0;
}

std::string HvxGemm::status() const {
    char b[400];
    std::snprintf(b, sizeof b, "HVX (v66): %zu MB of matrices in its reach", weight_bytes_ >> 20);
    std::string s = b;
    if (calls_) {
        std::snprintf(b, sizeof b,
                      "; %llu products, DSP %.2f s (%.1f G MAC/s), calls %.2f s, its inputs %.2f s and results %.2f s "
                      "on the CPU; the CPU's rows %.1f G MAC/s",
                      static_cast<unsigned long long>(calls_), dsp_s_, macs_ / std::max(1e-9, dsp_s_) * 1e-9, call_s_,
                      prep_s_, out_s_, cpu_macs_ / std::max(1e-9, cpu_s_) * 1e-9);
        s += b;
        s += "; shares";
        for (const auto &kv : share_) {
            std::snprintf(b, sizeof b, " %lldx%lld %.2f", static_cast<long long>(kv.first.first),
                          static_cast<long long>(kv.first.second), kv.second);
            s += b;
        }
    }
    if (redone_) s += "; " + std::to_string(redone_) + " near int32's limit, computed by the CPU";
    if (off_) s += "; off (" + note_ + ")";
    return s;
}

}  // namespace geode
