// audio.cpp - Gemma 4's audio encoder (audio.hpp). Each step as tools/gemma4_audio.py writes it out.
#include "audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>

#include "kernels.hpp"
#include "media.hpp"

namespace geode {

namespace {

using media::Packed;

// a radix-2 FFT of 512 points in double (numpy's rfft computes in double too)
struct Fft512 {
    static constexpr int N = 512;
    int rev[N];
    double cs[N / 2], sn[N / 2];
    Fft512() {
        for (int i = 0; i < N; ++i) {
            int r = 0;
            for (int b = 0; b < 9; ++b) r |= ((i >> b) & 1) << (8 - b);
            rev[i] = r;
        }
        for (int k = 0; k < N / 2; ++k) {
            cs[k] = std::cos(2.0 * 3.14159265358979323846 * k / N);
            sn[k] = -std::sin(2.0 * 3.14159265358979323846 * k / N);
        }
    }
    void run(double *re, double *im) const {
        for (int i = 0; i < N; ++i)
            if (i < rev[i]) {
                std::swap(re[i], re[rev[i]]);
                std::swap(im[i], im[rev[i]]);
            }
        for (int len = 2; len <= N; len <<= 1) {
            const int half = len / 2, step = N / len;
            for (int i = 0; i < N; i += len)
                for (int j = 0; j < half; ++j) {
                    const double wr = cs[j * step], wi = sn[j * step];
                    const double xr = re[i + j + half], xi = im[i + j + half];
                    const double vr = xr * wr - xi * wi, vi = xr * wi + xi * wr;
                    re[i + j + half] = re[i + j] - vr;
                    im[i + j + half] = im[i + j] - vi;
                    re[i + j] += vr;
                    im[i + j] += vi;
                }
        }
    }
};

// torch.nn.LayerNorm without a bias (the sums in double)
void layer_norm(float *x, const float *w, int n, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    const double mean = s / n;
    double v = 0.0;
    for (int i = 0; i < n; ++i) v += (x[i] - mean) * (x[i] - mean);
    const auto m = static_cast<float>(mean), inv = static_cast<float>(1.0 / std::sqrt(v / n + eps));
    for (int i = 0; i < n; ++i) x[i] = (x[i] - m) * inv * w[i];
}

int conv_out(int n) { return (n + 2 - 3) / 2 + 1; }   // a 3-tap, stride-2, padding-1 convolution's length

}  // namespace

bool Audio::open(const Pack &pack, bool direct, std::string &err) {
    ok_ = false;
    const auto &a = pack.manifest()["audio"];
    if (!a.is_object()) return false;
    pack_ = &pack;
    if (a["arch"].str() != "gemma4") {
        err = "geode.json: an audio encoder this engine does not take";
        return false;
    }
    auto &s = s_;
    s = Shape{};
    const auto i = [&](const char *k) { return static_cast<int>(a[k].i64()); };
    const auto f = [&](const char *k) { return static_cast<float>(a[k].f64()); };
    s.layers = i("layers");
    s.hidden = i("hidden");
    s.heads = i("heads");
    s.head_dim = i("head_dim");
    s.ff = i("ff");
    s.window = i("window");
    s.n_rel = i("n_rel");
    s.kernel = i("kernel");
    s.ch0 = static_cast<int>(a["channels"].at(0).i64());
    s.ch1 = static_cast<int>(a["channels"].at(1).i64());
    s.out_hidden = i("out_hidden");
    s.text_hidden = i("text_hidden");
    s.mel_bins = i("mel_bins");
    s.frame = i("frame");
    s.hop = i("hop");
    s.fft = i("fft");
    s.max_samples = a["max_samples"].i64();
    s.cap = f("logit_cap");
    s.residual = f("residual");
    s.eps = f("eps");
    s.k_scale = f("k_scale");
    s.mel_floor = f("mel_floor");
    s.start_token = i("audio_start");
    s.end_token = i("audio_end");
    s.pad_token = i("audio_pad");
    const int D = s.hidden;
    if (s.layers <= 0 || s.heads <= 0 || s.head_dim * s.heads != D || D % 64 || s.ff % 64 || s.window <= 0 ||
        s.window > s.n_rel || s.kernel <= 0 || s.ch0 <= 0 || s.ch1 <= 0 || s.fft != Fft512::N || s.mel_bins <= 0 ||
        s.frame <= 0 || s.frame > s.fft || s.hop <= 0 || s.max_samples <= 0 || s.pad_token < 0 ||
        static_cast<int>(a["layer_ranges"].size()) != s.layers || s.out_hidden % 64 || s.text_hidden % 64) {
        err = "geode.json: an audio encoder this engine does not take";
        return false;
    }
    const int W0 = conv_out(s.mel_bins), W1 = conv_out(W0);
    if (W1 * s.ch1 != D) {
        err = "geode.json: the audio encoder's sub-sampling does not give its width";
        return false;
    }
    if (!pack.find("audio.mel.window")) {   // audio.bin is optional (pack.cpp): not installed
        err = "audio.bin is not installed";
        return false;
    }
    const auto range = [](const json::Value &r) {
        return Range{static_cast<uint64_t>(r["offset"].i64()), static_cast<uint64_t>(r["bytes"].i64())};
    };
    head_ = range(a["head"]);
    tail_ = range(a["tail"]);
    layers_.clear();
    for (int l = 0; l < s.layers; ++l) layers_.push_back(range(a["layer_ranges"].at(static_cast<size_t>(l))));
    const std::string file = pack.path(pack.find("audio.mel.window")->file);
    auto check = [&](const std::string &name, int64_t rows, int64_t cols, bool matrix, const Range &r) {
        const Tensor *t = pack.find(name);
        if (!t || t->rows != rows || t->cols != cols || is_quantized(t->type) != matrix ||
            (!matrix && t->type != DType::F32) || (matrix && t->cols % 64) || pack.path(t->file) != file ||
            t->offset < r.offset || t->offset + media::tensor_bytes(*t) > r.offset + r.bytes) {
            err = "geode.json: " + name + " is not what the audio encoder reads";
            return false;
        }
        return true;
    };
    if (!check("audio.mel.window", 1, s.frame, false, head_) ||
        !check("audio.mel.filters", s.fft / 2 + 1, s.mel_bins, false, head_) ||
        !check("audio.sub.conv0.weight", s.ch0, 9, false, head_) || !check("audio.sub.norm0.weight", 1, s.ch0, false, head_) ||
        !check("audio.sub.conv1.weight", s.ch1, 9 * s.ch0, false, head_) ||
        !check("audio.sub.norm1.weight", 1, s.ch1, false, head_) || !check("audio.sub.proj.weight", D, D, true, head_) ||
        !check("audio.out.weight", s.out_hidden, D, true, tail_) || !check("audio.out.bias", 1, s.out_hidden, false, tail_) ||
        !check("audio.embed.weight", s.text_hidden, s.out_hidden, true, tail_))
        return false;
    for (int l = 0; l < s.layers; ++l) {
        const std::string b = "audio.layers." + std::to_string(l) + ".";
        const Range &r = layers_[static_cast<size_t>(l)];
        for (const char *n : {"feed_forward1.pre_layer_norm", "feed_forward1.post_layer_norm", "norm_pre_attn",
                              "norm_post_attn", "lconv1d.pre_layer_norm", "lconv1d.conv_norm",
                              "feed_forward2.pre_layer_norm", "feed_forward2.post_layer_norm", "norm_out"})
            if (!check(b + n + ".weight", 1, D, false, r)) return false;
        if (!check(b + "self_attn.qscale", 1, s.head_dim, false, r) || !check(b + "self_attn.relk", s.n_rel, D, false, r) ||
            !check(b + "lconv1d.dwconv", D, s.kernel, false, r))
            return false;
        const struct {
            const char *name;
            int64_t rows, cols;
        } mats[] = {{"feed_forward1.ffw_layer_1", s.ff, D}, {"feed_forward1.ffw_layer_2", D, s.ff},
                    {"self_attn.q_proj", D, D}, {"self_attn.k_proj", D, D}, {"self_attn.v_proj", D, D},
                    {"self_attn.post", D, D}, {"lconv1d.linear_start", 2 * D, D}, {"lconv1d.linear_end", D, D},
                    {"feed_forward2.ffw_layer_1", s.ff, D}, {"feed_forward2.ffw_layer_2", D, s.ff}};
        for (const auto &m : mats)
            if (!check(b + m.name + ".weight", m.rows, m.cols, true, r) || !check(b + m.name + ".clip", 1, 4, false, r))
                return false;
    }
    if (!file_.open(file, err, direct)) return false;
    ok_ = true;
    return true;
}

int Audio::tokens(int64_t n) const {
    if (!ok_ || n <= 0) return 0;
    n = std::min(n, s_.max_samples);
    const int64_t padded = (n + 127) / 128 * 128, len = s_.hop + padded;
    if (len < s_.frame + 1) return 0;
    const int frames = static_cast<int>((len - (s_.frame + 1)) / s_.hop + 1);
    // a frame is valid when the sample at its window's end is one of the clip's: f hop + frame < hop + n
    const auto valid = [&](int fr) { return static_cast<int64_t>(fr) * s_.hop + s_.frame < s_.hop + n; };
    const int h0 = conv_out(frames), h1 = conv_out(h0);
    int t = 0;
    for (int r = 0; r < h1; ++r) t += valid(4 * r);   // the masks keep every other row, twice
    return t;
}

bool Audio::read(const Range &r, AlignedBuffer &buf, size_t &cap, uint64_t &base, std::string &err) const {
    const uint64_t a0 = r.offset & ~uint64_t{4095};
    const uint64_t a1 = std::min<uint64_t>((r.offset + r.bytes + 4095) & ~uint64_t{4095}, file_.size());
    const auto n = static_cast<size_t>(a1 - a0);
    if (cap < n) {
        buf = aligned_bytes(n);
        cap = buf ? n : 0;
    }
    if (!buf) {
        err = "not enough memory for the audio encoder";
        return false;
    }
    if (!file_.read_at(a0, buf.get(), n)) {
        err = "cannot read audio.bin";
        return false;
    }
    base = a0;
    return true;
}

void Audio::features(const float *pcm, int64_t n, const float *win, const float *filters, ThreadPool &pool,
                     std::vector<float> &mel, std::vector<uint8_t> &valid) const {
    // (n as tokens() clips it) the clip zero-padded to a multiple of 128 samples with hop zeros in front; frame f is
    // samples [f hop - hop, + frame) of the clip, valid when the sample after its window is the clip's (then the
    // whole window is); invalid frames are zero
    const auto &s = s_;
    const int64_t padded = (n + 127) / 128 * 128, len = s.hop + padded;
    const int frames = static_cast<int>((len - (s.frame + 1)) / s.hop + 1), B = s.fft / 2 + 1, M = s.mel_bins;
    mel.assign(static_cast<size_t>(frames) * M, 0.0f);
    valid.assign(static_cast<size_t>(frames), 0);
    static const Fft512 fft;
    pool.parallel_for(frames, [&](int64_t a, int64_t b) {
        double re[Fft512::N], im[Fft512::N], mag[Fft512::N / 2 + 1];
        for (int64_t fr = a; fr < b; ++fr) {
            if (!(fr * s.hop + s.frame < s.hop + n)) continue;
            valid[static_cast<size_t>(fr)] = 1;
            const int64_t start = fr * s.hop - s.hop;
            for (int i = 0; i < Fft512::N; ++i) {
                const int64_t at = start + i;
                const float x = i < s.frame && at >= 0 ? pcm[at] : 0.0f;
                re[i] = i < s.frame ? static_cast<double>(x * win[i]) : 0.0;   // the product in f32, as numpy's
                im[i] = 0.0;
            }
            fft.run(re, im);
            for (int k = 0; k < B; ++k) mag[k] = std::sqrt(re[k] * re[k] + im[k] * im[k]);
            float *o = mel.data() + static_cast<size_t>(fr) * M;
            for (int m = 0; m < M; ++m) {
                double acc = 0.0;
                for (int k = 0; k < B; ++k) acc += mag[k] * static_cast<double>(filters[static_cast<size_t>(k) * M + m]);
                o[m] = static_cast<float>(std::log(acc + static_cast<double>(s.mel_floor)));
            }
        }
    }, 4, true);
}

bool Audio::encode(const float *pcm, int64_t n, ThreadPool &pool, std::vector<float> &out, std::string &err,
                   const MediaNpu *npu) {
    const auto &s = s_;
    if (!ok_) {
        err = "no audio encoder";
        return false;
    }
    n = std::min(n, s.max_samples);
    const int T_out = tokens(n);
    if (!pcm || T_out <= 0) {
        err = "an audio clip too short to take";
        return false;
    }
    const int D = s.hidden, F = s.ff, NH = s.heads, HD = s.head_dim, C0 = s.ch0, C1 = s.ch1, M = s.mel_bins;
    if (HD > 256 || s.window > 64) {
        err = "the audio encoder's heads are wider than this engine takes";
        return false;
    }
    prof_ = {};
    const double t0 = now_seconds();
    double tl = t0;
    const auto lap = [&](double &into) {
        const double t = now_seconds();
        into += t - tl;
        tl = t;
    };
    const auto at = [this](const std::string &name, const uint8_t *buf, uint64_t base) {
        const Tensor &t = *pack_->find(name);
        QView v = t.view();
        v.data = buf + (t.offset - base);
        return v;
    };
    const auto vec = [this](const std::string &name, const uint8_t *buf, uint64_t base) {
        return reinterpret_cast<const float *>(buf + (pack_->find(name)->offset - base));
    };
    static const char *const kLinears[] = {"feed_forward1.ffw_layer_1", "feed_forward1.ffw_layer_2", "self_attn.q_proj",
                                           "self_attn.k_proj", "self_attn.v_proj", "self_attn.post",
                                           "lconv1d.linear_start", "lconv1d.linear_end", "feed_forward2.ffw_layer_1",
                                           "feed_forward2.ffw_layer_2"};
    enum { kFf1a, kFf1b, kQ, kK, kV, kPost, kStart, kEnd, kFf2a, kFf2b };
    // the matrices of layer l (layers: the tail's), in the order they are used
    const auto mats = [&](int l) {
        std::vector<std::string> names;
        if (l == s.layers) {
            names = {"audio.out.weight", "audio.embed.weight"};
        } else {
            for (const char *m : kLinears) names.push_back("audio.layers." + std::to_string(l) + "." + m + ".weight");
        }
        return names;
    };
    bool mmla = media::have_i8mm();
    for (int l = 0; l <= s.layers && mmla; ++l)
        for (const auto &nm : mats(l)) mmla = mmla && pack_->find(nm)->type == DType::Q8 && pack_->find(nm)->rows % 8 == 0;
    mmla = mmla && pack_->find("audio.sub.proj.weight")->type == DType::Q8;

    // The NPU's HMX (MediaNpu; GEODE_AUDIO_NPU=0: not) as the vision encoder's: every matrix q8 with rows and columns
    // in 32s, turned into fp16 blocks in one of two shared buffers as its layer comes in - its ten matrices packed at
    // once (a layer is 46 MB of them: one thread would take longer than the layer's arithmetic); the buffers made
    // here, on the NPU's thread, once. No SMMLA repack then: a failure finishes the clip on the int8 kernels.
    constexpr int kMats = 10;
    bool on_npu = npu && npu->usable() && !(std::getenv("GEODE_AUDIO_NPU") && std::atoi(std::getenv("GEODE_AUDIO_NPU")) == 0);
    {
        const auto f16_of = [&](const std::string &nm) -> size_t {
            const Tensor *t = pack_->find(nm);
            if (!t || t->type != DType::Q8 || t->rows % 32 || t->cols % 32) return 0;
            return (npu->f16_bytes(t->rows, t->cols) + 127) / 128 * 128;
        };
        size_t need = 0;
        for (int l = 0; l <= s.layers && on_npu; ++l) {
            size_t sum = 0;
            for (const auto &nm : mats(l)) {
                const size_t b = f16_of(nm);
                on_npu = on_npu && b > 0;
                sum += b;
            }
            need = std::max(need, sum);
        }
        if (on_npu) {
            const size_t b = f16_of("audio.sub.proj.weight");
            on_npu = b > 0;
            need = std::max(need, b);
        }
        if (on_npu && (npu_g_ != npu->g || npu_cap_ < need)) {
            npu_g_ = npu->g;
            npu_cap_ = 0;
            for (int k = 0; k < 2 && on_npu; ++k) {
                npu_buf_[k] = static_cast<uint8_t *>(npu->shared(npu->g, need, 1, &npu_fd_[k]));
                on_npu = npu_buf_[k] != nullptr;
            }
            if (on_npu) npu_cap_ = need;
        }
    }
    if (on_npu) mmla = false;
    media::F16 f16d[2][kMats];
    bool f16ok[2] = {false, false};
    // matrices vs of buffer k into its fp16 blocks, each on a thread of its own; false when one is not
    const auto pack_f16 = [this, npu](const std::vector<QView> &vs, int k, media::F16 *fs) {
        uint64_t o = 0;
        std::vector<std::future<int>> jobs;
        bool ok = true;
        for (size_t j = 0; j < vs.size() && ok; ++j) {
            fs[j].fd = npu_fd_[k];
            fs[j].off = o;
            o += (npu->f16_bytes(vs[j].rows, vs[j].cols) + 127) / 128 * 128;
            ok = o <= npu_cap_;
            if (!ok) break;
            const QView v = vs[j];
            uint8_t *dst = npu_buf_[k] + fs[j].off;
            int *sh = &fs[j].shift;
            jobs.push_back(std::async(std::launch::async, [npu, v, dst, sh] {
                return npu->pack_f16(v.data, v.rows, v.cols, v.row_bytes, v.global, dst, sh);
            }));
        }
        for (auto &j : jobs) ok = j.get() == 0 && ok;
        return ok;
    };

    // the layers come in one ahead of the arithmetic, two buffers in turn (repacked for SMMLA there too, or into the
    // NPU's fp16 blocks)
    AlignedBuffer bb[2];
    size_t bcap[2] = {0, 0};
    uint64_t bbase[2] = {0, 0};
    std::string berr[2];
    std::vector<Packed> bpk[2];
    const auto start = [&](int l) {
        const Range *r = l < s.layers ? &layers_[static_cast<size_t>(l)] : &tail_;
        const int k = l & 1;
        return std::async(std::launch::async, [this, r, &bb, &bcap, &bbase, &berr, &bpk, &at, &f16d, &f16ok, &pack_f16,
                                               k, mmla, on_npu, names = mats(l)] {
            if (!read(*r, bb[k], bcap[k], bbase[k], berr[k])) return false;
            if (mmla) {
                bpk[k].resize(names.size());
                for (size_t j = 0; j < names.size(); ++j) media::pack_q8(at(names[j], bb[k].get(), bbase[k]), bpk[k][j]);
            }
            f16ok[k] = false;
            if (on_npu && names.size() <= static_cast<size_t>(kMats)) {
                std::vector<QView> vs;
                for (const auto &nm : names) vs.push_back(at(nm, bb[k].get(), bbase[k]));
                f16ok[k] = pack_f16(vs, k, f16d[k]);
            }
            return true;
        });
    };
    std::future<bool> next = start(0);
    AlignedBuffer hbuf;
    size_t hcap = 0;
    uint64_t hbase = 0;
    if (!read(head_, hbuf, hcap, hbase, err)) {
        next.wait();
        return false;
    }
    lap(prof_.wait);

    // 1. the log-mel features
    std::vector<float> mel;
    std::vector<uint8_t> valid;
    features(pcm, n, vec("audio.mel.window", hbuf.get(), hbase), vec("audio.mel.filters", hbuf.get(), hbase), pool, mel,
             valid);
    const int frames = static_cast<int>(valid.size());
    prof_.frames = frames;
    lap(prof_.features);

    // 2. the sub-sampling: [frames][mel] -> conv0 [H0][W0][C0] -> conv1 [H1][W1][C1] = [T][D]; each conv's input rows
    // of invalid frames zero (the mask keeps every other row)
    const int H0 = conv_out(frames), W0 = conv_out(M), H1 = conv_out(H0), W1 = conv_out(W0), T = H1;
    std::vector<float> c0(static_cast<size_t>(H0) * W0 * C0), x(static_cast<size_t>(T) * D);
    {
        const float *w0 = vec("audio.sub.conv0.weight", hbuf.get(), hbase), *n0 = vec("audio.sub.norm0.weight", hbuf.get(), hbase);
        pool.parallel_for(H0, [&](int64_t a, int64_t b) {
            for (int64_t i = a; i < b; ++i)
                for (int j = 0; j < W0; ++j) {
                    float px[9];
                    for (int di = 0; di < 3; ++di)
                        for (int dj = 0; dj < 3; ++dj) {
                            const int64_t r = 2 * i + di - 1;
                            const int cc = 2 * j + dj - 1;
                            px[di * 3 + dj] = r >= 0 && r < frames && cc >= 0 && cc < M ? mel[static_cast<size_t>(r) * M + cc] : 0.0f;
                        }
                    float *o = c0.data() + (static_cast<size_t>(i) * W0 + j) * C0;
                    for (int c = 0; c < C0; ++c) {
                        float acc = 0.0f;
                        for (int kk = 0; kk < 9; ++kk) acc += w0[c * 9 + kk] * px[kk];
                        o[c] = acc;
                    }
                    layer_norm(o, n0, C0, s.eps);
                    for (int c = 0; c < C0; ++c) o[c] = std::max(o[c], 0.0f);
                }
        }, 4, true);
        // the second's input: conv0's rows of invalid frames (the mask's every other row) zero
        for (int i = 0; i < H0; ++i)
            if (!valid[static_cast<size_t>(2 * i)]) std::fill_n(c0.data() + static_cast<size_t>(i) * W0 * C0, W0 * C0, 0.0f);
        // conv1's weights as [out][tap][in]: a tap's input channels are contiguous in c0
        const float *w1s = vec("audio.sub.conv1.weight", hbuf.get(), hbase), *n1 = vec("audio.sub.norm1.weight", hbuf.get(), hbase);
        std::vector<float> w1(static_cast<size_t>(C1) * 9 * C0);
        for (int c = 0; c < C1; ++c)
            for (int ci = 0; ci < C0; ++ci)
                for (int kk = 0; kk < 9; ++kk)
                    w1[(static_cast<size_t>(c) * 9 + kk) * C0 + ci] = w1s[(static_cast<size_t>(c) * C0 + ci) * 9 + kk];
        pool.parallel_for(T, [&](int64_t a, int64_t b) {
            std::vector<float> patch(static_cast<size_t>(9) * C0);
            for (int64_t i = a; i < b; ++i)
                for (int j = 0; j < W1; ++j) {
                    for (int di = 0; di < 3; ++di)
                        for (int dj = 0; dj < 3; ++dj) {
                            const int64_t r = 2 * i + di - 1;
                            const int cc = 2 * j + dj - 1;
                            float *pp = patch.data() + static_cast<size_t>(di * 3 + dj) * C0;
                            if (r >= 0 && r < H0 && cc >= 0 && cc < W0)
                                std::copy_n(c0.data() + (static_cast<size_t>(r) * W0 + cc) * C0, C0, pp);
                            else
                                std::fill_n(pp, C0, 0.0f);
                        }
                    float *o = x.data() + static_cast<size_t>(i) * D + static_cast<size_t>(j) * C1;   // [W1][C1]
                    for (int c = 0; c < C1; ++c) {
                        const float *wr = w1.data() + static_cast<size_t>(c) * 9 * C0;
                        float acc = 0.0f;
                        for (int kk = 0; kk < 9 * C0; ++kk) acc += wr[kk] * patch[static_cast<size_t>(kk)];
                        o[c] = acc;
                    }
                    layer_norm(o, n1, C1, s.eps);
                    for (int c = 0; c < C1; ++c) o[c] = std::max(o[c], 0.0f);
                }
        }, 1, true);
    }
    prof_.tokens = T_out;
    lap(prof_.sub);
    media::Products pr;
    const size_t wide = static_cast<size_t>(std::max({D, s.out_hidden, s.text_hidden}));
    std::vector<float> hn(static_cast<size_t>(T) * wide), y(static_cast<size_t>(T) * wide),
        g(static_cast<size_t>(T) * std::max(F, 2 * D)), q(static_cast<size_t>(T) * D), k(static_cast<size_t>(T) * D),
        v(static_cast<size_t>(T) * D);
    {
        Packed pp;
        const QView pw = at("audio.sub.proj.weight", hbuf.get(), hbase);
        if (mmla) media::pack_q8(pw, pp);
        // (on the NPU from buffer 1: layer 0 comes into buffer 0 meanwhile, layer 1 into this one only after it)
        pr.use_npu(on_npu ? npu : nullptr);
        media::F16 fp;
        const bool pf = on_npu && pack_f16(std::vector<QView>{pw}, 1, &fp);
        std::copy(x.begin(), x.end(), hn.begin());
        pr.mm(pool, pw, mmla ? &pp : nullptr, hn.data(), static_cast<size_t>(D), x.data(), static_cast<size_t>(D), T,
              pf ? &fp : nullptr);
    }
    hbuf.reset();
    lap(prof_.matmul);

    const auto rms = [&](const float *w, const float *in, float *o, int width) {
        pool.parallel_for(T, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t)
                rmsnorm(in + static_cast<size_t>(t) * width, w, o + static_cast<size_t>(t) * width, width, s.eps, false);
        }, 8, true);
    };
    const auto add = [&](float *dst, const float *src, float scale) {
        pool.parallel_for(static_cast<int64_t>(T) * D, [&](int64_t a, int64_t b) {
            for (int64_t i = a; i < b; ++i) dst[static_cast<size_t>(i)] += scale * src[static_cast<size_t>(i)];
        }, 4096, true);
    };

    // 3. the Conformer layers
    for (int l = 0; l < s.layers; ++l) {
        const int kb = l & 1;
        if (!next.get()) {
            err = berr[kb];
            return false;
        }
        const uint8_t *buf = bb[kb].get();
        const uint64_t base = bbase[kb];
        const std::vector<Packed> &pk = bpk[kb];
        next = start(l + 1);
        lap(prof_.wait);
        const std::string b = "audio.layers." + std::to_string(l) + ".";
        const auto lin = [&](int m) { return at(b + kLinears[m] + ".weight", buf, base); };
        const auto clip = [&](int m) { return vec(b + kLinears[m] + ".clip", buf, base); };
        const auto P = [&](int m) { return mmla ? &pk[static_cast<size_t>(m)] : nullptr; };
        const media::F16 *fl = f16ok[kb] ? f16d[kb] : nullptr;
        const auto FB = [&](int m) { return fl ? fl + m : nullptr; };
        const auto nw = [&](const char *name) { return vec(b + name + ".weight", buf, base); };
        // a half-step MLP: x += residual * post_norm(W2 silu(W1 pre_norm(x)))
        const auto ffn = [&](const char *pre, const char *post, int m1, int m2) {
            rms(nw(pre), x.data(), hn.data(), D);
            lap(prof_.other);
            pr.clipped(pool, lin(m1), P(m1), clip(m1), hn.data(), g.data(), T, FB(m1));
            lap(prof_.matmul);
            pool.parallel_for(static_cast<int64_t>(T) * F, [&](int64_t a, int64_t e) {
                for (int64_t i = a; i < e; ++i) g[static_cast<size_t>(i)] = silu_fast(g[static_cast<size_t>(i)]);
            }, 4096, true);
            lap(prof_.other);
            pr.clipped(pool, lin(m2), P(m2), clip(m2), g.data(), y.data(), T, FB(m2));
            lap(prof_.matmul);
            rms(nw(post), y.data(), y.data(), D);
            add(x.data(), y.data(), s.residual);
            lap(prof_.other);
        };
        ffn("feed_forward1.pre_layer_norm", "feed_forward1.post_layer_norm", kFf1a, kFf1b);

        // attention over the token and the window - 1 before it, with relative positions
        rms(nw("norm_pre_attn"), x.data(), hn.data(), D);
        lap(prof_.other);
        pr.clipped(pool, lin(kQ), P(kQ), clip(kQ), hn.data(), q.data(), T, FB(kQ));
        pr.clipped(pool, lin(kK), P(kK), clip(kK), hn.data(), k.data(), T, FB(kK));
        pr.clipped(pool, lin(kV), P(kV), clip(kV), hn.data(), v.data(), T, FB(kV));
        lap(prof_.matmul);
        {
            const float *qs = vec(b + "self_attn.qscale", buf, base), *rk = vec(b + "self_attn.relk", buf, base);
            pool.parallel_for(T, [&](int64_t a, int64_t e) {
                float logit[64], p[64], qh[256];
                for (int64_t t = a; t < e; ++t) {
                    const int64_t j0 = std::max<int64_t>(0, t - s.window + 1);
                    const int nk = static_cast<int>(t - j0 + 1);
                    for (int h = 0; h < NH; ++h) {
                        const float *qr = q.data() + static_cast<size_t>(t) * D + static_cast<size_t>(h) * HD;
                        for (int d = 0; d < HD; ++d) qh[d] = qr[d] * qs[d];
                        float mx = -INFINITY;
                        for (int i = 0; i < nk; ++i) {
                            const int64_t j = j0 + i;
                            const float *kr = k.data() + static_cast<size_t>(j) * D + static_cast<size_t>(h) * HD;
                            const float *rr = rk + static_cast<size_t>(t - j) * D + static_cast<size_t>(h) * HD;
                            float ac = 0.0f, bd = 0.0f;
                            for (int d = 0; d < HD; ++d) {
                                ac += qh[d] * kr[d];
                                bd += qh[d] * rr[d];
                            }
                            const float lg = std::tanh((ac * s.k_scale + bd) / s.cap) * s.cap;
                            logit[i] = lg;
                            mx = std::max(mx, lg);
                        }
                        float sum = 0.0f;
                        for (int i = 0; i < nk; ++i) {
                            p[i] = std::exp(logit[i] - mx);
                            sum += p[i];
                        }
                        float *o = hn.data() + static_cast<size_t>(t) * D + static_cast<size_t>(h) * HD;
                        std::fill_n(o, HD, 0.0f);
                        for (int i = 0; i < nk; ++i) {
                            const float w = p[i] / sum;
                            const float *vr = v.data() + static_cast<size_t>(j0 + i) * D + static_cast<size_t>(h) * HD;
                            for (int d = 0; d < HD; ++d) o[d] += w * vr[d];
                        }
                    }
                }
            }, 4, true);
        }
        lap(prof_.attention);
        pr.clipped(pool, lin(kPost), P(kPost), clip(kPost), hn.data(), y.data(), T, FB(kPost));
        lap(prof_.matmul);
        rms(nw("norm_post_attn"), y.data(), y.data(), D);
        add(x.data(), y.data(), 1.0f);

        // the light convolution: x += end(silu(conv_norm(dwconv(glu(start(pre_norm(x)))))))
        rms(nw("lconv1d.pre_layer_norm"), x.data(), hn.data(), D);
        lap(prof_.other);
        pr.clipped(pool, lin(kStart), P(kStart), clip(kStart), hn.data(), g.data(), T, FB(kStart));   // [T][2 D]
        lap(prof_.matmul);
        {
            const float *dw = vec(b + "lconv1d.dwconv", buf, base);
            const int K = s.kernel;
            pool.parallel_for(T, [&](int64_t a, int64_t e) {   // the GLU into hn: a * sigmoid(b) of the halves
                for (int64_t t = a; t < e; ++t) {
                    const float *gr = g.data() + static_cast<size_t>(t) * 2 * D;
                    float *o = hn.data() + static_cast<size_t>(t) * D;
                    for (int c = 0; c < D; ++c) o[c] = gr[c] / (1.0f + fast_exp(-gr[D + c]));
                }
            }, 8, true);
            pool.parallel_for(T, [&](int64_t a, int64_t e) {   // causal depthwise: K - 1 zeros in front
                for (int64_t t = a; t < e; ++t) {
                    float *o = y.data() + static_cast<size_t>(t) * D;
                    for (int c = 0; c < D; ++c) {
                        float acc = 0.0f;
                        for (int kk = 0; kk < K; ++kk) {
                            const int64_t src = t - (K - 1) + kk;
                            if (src >= 0) acc += dw[static_cast<size_t>(c) * K + kk] * hn[static_cast<size_t>(src) * D + c];
                        }
                        o[c] = acc;
                    }
                }
            }, 8, true);
            rms(nw("lconv1d.conv_norm"), y.data(), y.data(), D);
            pool.parallel_for(static_cast<int64_t>(T) * D, [&](int64_t a, int64_t e) {
                for (int64_t i = a; i < e; ++i) y[static_cast<size_t>(i)] = silu_fast(y[static_cast<size_t>(i)]);
            }, 4096, true);
        }
        lap(prof_.other);
        pr.clipped(pool, lin(kEnd), P(kEnd), clip(kEnd), y.data(), hn.data(), T, FB(kEnd));
        lap(prof_.matmul);
        add(x.data(), hn.data(), 1.0f);

        ffn("feed_forward2.pre_layer_norm", "feed_forward2.post_layer_norm", kFf2a, kFf2b);
        rms(nw("norm_out"), x.data(), x.data(), D);
        lap(prof_.other);
    }

    // 4. output_proj (+ bias) over the valid tokens (a prefix), then embed_audio: a weightless RMSnorm and the
    // projection into the text model's width
    if (!next.get()) {
        err = berr[s.layers & 1];
        return false;
    }
    lap(prof_.wait);
    {
        const int kb = s.layers & 1;
        const uint8_t *buf = bb[kb].get();
        const uint64_t base = bbase[kb];
        const int O = s.out_hidden;
        const media::F16 *ft = f16ok[kb] ? f16d[kb] : nullptr;
        pr.mm(pool, at("audio.out.weight", buf, base), mmla ? &bpk[kb][0] : nullptr, x.data(), static_cast<size_t>(D),
              y.data(), static_cast<size_t>(O), T_out, ft ? ft + 0 : nullptr);
        const float *bias = vec("audio.out.bias", buf, base);
        pool.parallel_for(T_out, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t) {
                float *r = y.data() + static_cast<size_t>(t) * O;
                for (int i = 0; i < O; ++i) r[i] += bias[i];
                rmsnorm(r, nullptr, hn.data() + static_cast<size_t>(t) * O, O, s.eps, false);
            }
        }, 8, true);
        out.assign(static_cast<size_t>(T_out) * s.text_hidden, 0.0f);
        pr.mm(pool, at("audio.embed.weight", buf, base), mmla ? &bpk[kb][1] : nullptr, hn.data(), static_cast<size_t>(O),
              out.data(), static_cast<size_t>(s.text_hidden), T_out, ft ? ft + 1 : nullptr);
    }
    lap(prof_.matmul);
    prof_.npu = on_npu && pr.npu_live();
    prof_.total = now_seconds() - t0;
    return true;
}

std::string Audio::profile() const {
    char b[384];
    std::snprintf(b, sizeof b,
                  "%d frames -> %d tokens %.2f s (reads waited %.2f, features %.2f, sub-sampling %.2f, products %.2f%s, "
                  "attention %.2f, other %.2f)",
                  prof_.frames, prof_.tokens, prof_.total, prof_.wait, prof_.features, prof_.sub, prof_.matmul,
                  prof_.npu ? " on HMX" : "", prof_.attention, prof_.other);
    return b;
}

}  // namespace geode
