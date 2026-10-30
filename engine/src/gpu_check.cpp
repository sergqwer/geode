// gpu_check.cpp - Geode's own GPU code on this phone's GPU (geode_bench "gpucheck", the app's Check and send): the path
// the engine takes here and why, then each GPU product it would run, on the model's own shapes with random q4
// weights (no model needed), against the CPU's and timed. A remote test lab's phone has no model and minutes to
// spare, so this says in one go whether the GPU code builds, computes right and how fast.
//
// Right: the GPU rounds the activations to int8 as the CPU's int8 path does, the sums in another order, so its error
// against the exact products (FP32 activations, dequantized weights) must stay within the CPU int8 path's own (x 1.5).
// Each step's name goes into step_file before it runs: when the driver takes the process down, the app reads there
// on which step.
#include "bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "bench_json.hpp"
#include "gpu.hpp"
#include "gpu8.hpp"
#include "kernels.hpp"

namespace geode::bench {

namespace {

// Qwen3.6-35B-A3B (the pack on Hugging Face)
constexpr int kHidden = 2048, kFf = 512, kTopK = 8, kExperts = 256, kTokens = 64;

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void step(const std::string &file, const std::string &name) {
    if (file.empty()) return;
    if (std::FILE *f = std::fopen(file.c_str(), "wb")) {
        std::fputs(name.c_str(), f);
        std::fclose(f);
    }
}

// a q4 matrix in the pack's layout: random codes, E4M3 scales 0.25 .. 2 (kernel_test.cpp's)
struct Matrix {
    std::vector<uint8_t> bytes;
    QView view;
};

Matrix random_q4(int64_t rows, int64_t cols, std::mt19937 &rng) {
    Matrix m;
    const size_t rb = row_bytes(DType::Q4, cols), code = rb - static_cast<size_t>(cols / kBlock);
    m.bytes.resize(static_cast<size_t>(rows) * rb);
    std::uniform_int_distribution<int> byte(0, 255), scale(0x30, 0x48);
    for (int64_t r = 0; r < rows; ++r) {
        uint8_t *row = m.bytes.data() + static_cast<size_t>(r) * rb;
        for (size_t i = 0; i < code; ++i) row[i] = static_cast<uint8_t>(byte(rng));
        for (size_t i = code; i < rb; ++i) row[i] = static_cast<uint8_t>(scale(rng));
    }
    m.view = QView{DType::Q4, rows, cols, rb, m.bytes.data(), 1.0f};
    return m;
}

std::vector<float> random_vec(int64_t n, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(static_cast<size_t>(n));
    for (float &x : v) x = nd(rng);
    return v;
}

// y = W x exactly, and as the CPU's int8 path computes it
void exact(const QView &m, const float *x, float *y) { matvec_rows(m, x, y, 0, m.rows); }
void int8(const QView &m, const float *x, float *y) {
    QAct a;
    quantize_activation(x, m.cols, a);
    matvec_rows_q(m, a, y, 0, m.rows);
}

// the largest difference against the exact result, relative to its largest magnitude
struct Err {
    double ref = 0;   // the exact result's largest magnitude
    double gpu = 0, cpu = 0;
    bool finite = true;
    void add(const float *ex, const float *g, const float *c, int64_t n) {
        for (int64_t i = 0; i < n; ++i) ref = std::max(ref, static_cast<double>(std::fabs(ex[i])));
        for (int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(g[i])) finite = false;
            gpu = std::max(gpu, static_cast<double>(std::fabs(g[i] - ex[i])));
            cpu = std::max(cpu, static_cast<double>(std::fabs(c[i] - ex[i])));
        }
    }
    bool ok() const { return finite && ref > 0 && gpu <= 1.5 * cpu + 1e-6 * ref; }
};

void write_check(Json &j, const std::string &name, bool ok, double ms, double bytes, const Err *e,
                 const std::string &error, double ops = 0) {
    j.begin_object();
    j.str("name", name).boolean("ok", ok);
    if (ms > 0) j.num("ms", ms);
    if (ms > 0 && bytes > 0) j.num("gbs", bytes / (ms * 1e-3) / 1e9);
    if (ms > 0 && ops > 0) j.num("gops", ops / (ms * 1e-3) / 1e9);   // a batch's: its multiply-adds x 2
    if (e) j.num("err_gpu", e->ref > 0 ? e->gpu / e->ref : -1).num("err_cpu_int8", e->ref > 0 ? e->cpu / e->ref : -1);
    if (!error.empty()) j.str("error", error);
    j.end_object();
}

// runs f until 0.25 s or 20 times (one warm-up first): the average in ms; -1 when f fails
template <class F> double timed(F &&f) {
    if (!f()) return -1;
    const double t0 = now();
    int n = 0;
    while (n < 20 && (n < 3 || now() - t0 < 0.25)) {
        if (!f()) return -1;
        ++n;
    }
    return (now() - t0) / n * 1e3;
}

// the tokens a batch's error is checked on (all of them would take the CPU seconds)
std::vector<int> sampled(int n) {
    std::vector<int> t{0};
    if (n > 2) t.push_back(n / 2);
    if (n > 1) t.push_back(n - 1);
    return t;
}

// ---- The Adreno 7xx+ path (gpu8): built, its staging reserved, the batch products and the routed experts checked,
// the one-token programs built
bool check_gpu8(Json &j, const std::string &sf, std::string &why, bool &ok_all) {
    Gpu8 g;
    step(sf, "gpu8: open (building the program)");
    double t0 = now();
    std::string err;
    const bool opened = g.open(err);
    if (!opened && err.rfind("no Adreno 7xx", 0) == 0) {
        why = err;
        return false;   // not this path: no check written
    }
    j.str("device", g.device());
    j.begin_array("checks");
    write_check(j, "build the batch program", opened, (now() - t0) * 1e3, 0, nullptr, err);
    if (!opened) {
        j.end_array();
        why = err;
        ok_all = false;
        return true;
    }
    step(sf, "gpu8: reserve");
    err.clear();
    const bool reserved = g.reserve(kTokens, 4096, 8192, kHidden, kFf, kTopK, kExperts, err);
    write_check(j, "shared memory for a batch", reserved, 0, 0, nullptr, err);
    ok_all = reserved;
    std::mt19937 rng(1234);
    // the batch products: GDN qkv 8192 x 2048 and out 2048 x 4096, one token to a full batch
    struct Shape {
        const char *name;
        int64_t rows, cols;
    };
    for (const Shape &s : {Shape{"GDN qkv 8192x2048", 8192, 2048}, Shape{"out 2048x4096", 2048, 4096}}) {
        if (!reserved) break;
        Matrix m = random_q4(s.rows, s.cols, rng);
        QView gv = m.view;   // adopt copies it to shared memory and points gv there; the CPU reads m.view
        step(sf, std::string("gpu8: adopt ") + s.name);
        err.clear();
        if (!g.adopt(gv, err)) {
            write_check(j, std::string("move ") + s.name + " to the GPU", false, 0, 0, nullptr, err);
            ok_all = false;
            continue;
        }
        for (int n : {1, 8, kTokens}) {
            const std::string name = std::string(s.name) + ", " + std::to_string(n) + (n == 1 ? " token" : " tokens");
            if (!g.fits_batch(gv, n)) {
                write_check(j, name, false, 0, 0, nullptr, "the GPU does not take this batch");
                ok_all = false;
                continue;
            }
            std::vector<std::vector<float>> xs(static_cast<size_t>(n)), ys(static_cast<size_t>(n));
            std::vector<const float *> xp;
            std::vector<float *> yp;
            for (int t = 0; t < n; ++t) {
                xs[static_cast<size_t>(t)] = random_vec(s.cols, rng);
                ys[static_cast<size_t>(t)].assign(static_cast<size_t>(s.rows), NAN);
                xp.push_back(xs[static_cast<size_t>(t)].data());
                yp.push_back(ys[static_cast<size_t>(t)].data());
            }
            step(sf, "gpu8: gemm " + name);
            err.clear();
            const double ms = timed([&] { return g.gemm(gv, n, xp.data(), yp.data(), err); });
            Err e;
            if (ms > 0) {
                std::vector<float> ex(static_cast<size_t>(s.rows)), c(static_cast<size_t>(s.rows));
                for (int t : sampled(n)) {
                    exact(m.view, xp[static_cast<size_t>(t)], ex.data());
                    int8(m.view, xp[static_cast<size_t>(t)], c.data());
                    e.add(ex.data(), yp[static_cast<size_t>(t)], c.data(), s.rows);
                }
            }
            const bool ok = ms > 0 && e.ok();
            ok_all = ok_all && ok;
            write_check(j, name, ok, ms, static_cast<double>(m.bytes.size()), ms > 0 ? &e : nullptr, err,
                        2.0 * static_cast<double>(s.rows * s.cols) * n);
        }
    }
    // the routed experts: 8 of them (gate_up 1024 x 2048, down 2048 x 512), each token through all eight; in one
    // piece of shared memory, each expert's gate_up then its down in tile8, as the expert cache's slots hold them
    if (reserved) {
        std::vector<Matrix> gu, dn;
        std::vector<Gpu8::Expert> ex(kTopK);
        step(sf, "gpu8: experts into shared memory");
        err.clear();
        for (int i = 0; i < kTopK; ++i) {
            gu.push_back(random_q4(2 * kFf, kHidden, rng));
            dn.push_back(random_q4(kHidden, kFf, rng));
        }
        const size_t gu_b = gu[0].bytes.size(), dn_b = dn[0].bytes.size();
        uint8_t *piece = g.alloc(kTopK * (gu_b + dn_b), err);
        const bool adopted = piece != nullptr;
        for (int i = 0; adopted && i < kTopK; ++i) {
            uint8_t *at = piece + static_cast<size_t>(i) * (gu_b + dn_b);
            for (auto [m, dst] : {std::pair{&gu[static_cast<size_t>(i)], at}, std::pair{&dn[static_cast<size_t>(i)], at + gu_b}}) {
                to_tile8(DType::Q4, m->view.rows, m->view.cols, m->view.data, dst);
                QView v = m->view;
                v.data = dst;
                v.quad = v.tile8 = true;
                (m == &gu[static_cast<size_t>(i)] ? ex[static_cast<size_t>(i)].gu : ex[static_cast<size_t>(i)].dn) = v;
            }
        }
        if (!adopted) {
            write_check(j, "move the experts to the GPU", false, 0, 0, nullptr, err);
            ok_all = false;
        }
        for (int n : {1, 16}) {
            if (!adopted) break;
            const std::string name = "8 experts, " + std::to_string(n) + (n == 1 ? " token" : " tokens");
            for (int i = 0; i < kTopK; ++i) {
                auto &p = ex[static_cast<size_t>(i)].picks;
                p.clear();
                for (int t = 0; t < n; ++t) p.emplace_back(t, t * kTopK + i);
            }
            std::vector<std::vector<float>> hs(static_cast<size_t>(n));
            std::vector<const float *> hp;
            for (int t = 0; t < n; ++t) {
                hs[static_cast<size_t>(t)] = random_vec(kHidden, rng);
                hp.push_back(hs[static_cast<size_t>(t)].data());
            }
            step(sf, "gpu8: experts " + name);
            err.clear();
            const double ms = timed([&] { return g.experts(ex, n, hp.data(), err); });
            Err e;
            if (ms > 0) {
                std::vector<float> ge(2 * kFf), gc(2 * kFf), se(kFf), sc(kFf), ye(kHidden), yc(kHidden);
                for (int t : sampled(n)) {
                    for (int i = 0; i < kTopK; ++i) {
                        const float *h = hp[static_cast<size_t>(t)];
                        exact(gu[static_cast<size_t>(i)].view, h, ge.data());
                        int8(gu[static_cast<size_t>(i)].view, h, gc.data());
                        for (int v = 0; v < kFf; ++v) {
                            se[static_cast<size_t>(v)] = silu(ge[static_cast<size_t>(v)]) * ge[static_cast<size_t>(kFf + v)];
                            sc[static_cast<size_t>(v)] = silu(gc[static_cast<size_t>(v)]) * gc[static_cast<size_t>(kFf + v)];
                        }
                        exact(dn[static_cast<size_t>(i)].view, se.data(), ye.data());
                        int8(dn[static_cast<size_t>(i)].view, sc.data(), yc.data());
                        e.add(ye.data(), g.out() + static_cast<size_t>(t * kTopK + i) * kHidden, yc.data(), kHidden);
                    }
                }
            }
            const bool ok = ms > 0 && e.ok();
            ok_all = ok_all && ok;
            const double bytes = static_cast<double>(kTopK) * static_cast<double>(gu_b + dn_b);
            write_check(j, name, ok, ms, bytes, ms > 0 ? &e : nullptr, err,
                        2.0 * 3.0 * kHidden * kFf * kTopK * static_cast<double>(n));
        }
    }
    // generating on the GPU: its one-token program, and the two-token one that verifies drafts (built, not run: they
    // need a whole model's layers)
    if (reserved) {
        Gpu8::DecodeShape s;
        s.hidden = kHidden, s.vocab = 248320, s.n_experts = kExperts, s.top_k = kTopK, s.moe_ff = kFf;
        s.nh = 16, s.nkv = 2, s.hd = 256, s.half = 32, s.n_attn = 10, s.kv_chunk = 4096, s.max_ctx = 4096;
        s.nk = 16, s.nv = 32, s.dk = 128, s.dv = 128, s.conv_k = 4, s.kv_i8 = true, s.guesses = kTopK;
        std::vector<float> inv_freq(static_cast<size_t>(s.half));
        for (int i = 0; i < s.half; ++i) inv_freq[static_cast<size_t>(i)] = std::pow(1e7f, -2.0f * static_cast<float>(i) / 64.0f);
        step(sf, "gpu8: build the generating program");
        err.clear();
        t0 = now();
        const bool dec = g.setup_decode(s, inv_freq.data(), err);
        write_check(j, "build the generating program", dec, (now() - t0) * 1e3, 0, nullptr, err);
        ok_all = ok_all && dec;
        if (dec) {
            step(sf, "gpu8: build the verifying program");
            err.clear();
            t0 = now();
            const bool ver = g.setup_verify(err);
            write_check(j, "build the verifying program (2 tokens)", ver, (now() - t0) * 1e3, 0, nullptr, err);
            ok_all = ok_all && ver;
        }
    }
    j.end_array();
    j.num("shared_mb", static_cast<double>(g.allocated()) / 1048576.0);
    j.num("max_piece_mb", static_cast<double>(g.max_piece()) / 1048576.0);
    return true;
}

// ---- The Adreno 618's hand-written path: opened, the prompt products checked
bool check_618(Json &j, const std::string &sf, std::string &why, bool &ok_all) {
    Gpu g;
    step(sf, "gpu618: open");
    std::string err;
    const double t0 = now();
    if (!g.open(err)) {
        why = err;
        return false;
    }
    j.begin_array("checks");
    write_check(j, "load the Adreno 618 kernels", true, (now() - t0) * 1e3, 0, nullptr, "");
    ok_all = true;
    std::mt19937 rng(4321);
    Matrix m = random_q4(8192, 2048, rng);
    QView gv = m.view;   // adopt copies it into a GPU buffer
    step(sf, "gpu618: adopt");
    if (!Gpu::fits(gv) || !g.adopt(gv, err)) {
        write_check(j, "move GDN qkv 8192x2048 to the GPU", false, 0, 0, nullptr, err.empty() ? "does not fit" : err);
        ok_all = false;
    } else {
        for (int n : {1, 4}) {
            const std::string name = "GDN qkv 8192x2048, " + std::to_string(n) + (n == 1 ? " token" : " tokens");
            std::vector<std::vector<float>> xs(static_cast<size_t>(n)), ys(static_cast<size_t>(n));
            std::vector<const float *> xp;
            std::vector<float *> yp;
            for (int t = 0; t < n; ++t) {
                xs[static_cast<size_t>(t)] = random_vec(2048, rng);
                ys[static_cast<size_t>(t)].assign(8192, NAN);
                xp.push_back(xs[static_cast<size_t>(t)].data());
                yp.push_back(ys[static_cast<size_t>(t)].data());
            }
            step(sf, "gpu618: " + name);
            err.clear();
            const double ms = timed([&] { return g.start(gv, 8192, n, xp.data(), err) && g.finish(yp.data(), err); });
            Err e;
            if (ms > 0) {
                std::vector<float> ex(8192), c(8192);
                for (int t : sampled(n)) {
                    exact(m.view, xp[static_cast<size_t>(t)], ex.data());
                    int8(m.view, xp[static_cast<size_t>(t)], c.data());
                    e.add(ex.data(), yp[static_cast<size_t>(t)], c.data(), 8192);
                }
            }
            const bool ok = ms > 0 && e.ok();
            ok_all = ok_all && ok;
            write_check(j, name, ok, ms, static_cast<double>(m.bytes.size()), ms > 0 ? &e : nullptr, err,
                        2.0 * 8192.0 * 2048.0 * n);
        }
    }
    j.end_array();
    return true;
}

}  // namespace

std::string gpu_check(const std::string &step_file) {
    Json j;
    j.begin_object();
    std::string why8, why618;
    bool ok = false;
    const double t0 = now();
    if (check_gpu8(j, step_file, why8, ok)) {
        j.str("path", "gpu8");
        if (!why8.empty()) j.str("reason", why8);
    } else if (check_618(j, step_file, why618, ok)) {
        j.str("path", "gpu618").str("reason", why8);
    } else {
        j.str("path", "cpu").str("reason", why8 + "; " + why618);
    }
    j.boolean("checks_ok", ok);
    j.num("seconds", now() - t0);
    j.end_object();
    step(step_file, "");
    return j.take();
}

}  // namespace geode::bench
