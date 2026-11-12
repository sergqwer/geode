// yue2_master.cpp - loudness, mastering and the song's file (yue2_master.hpp)
#include "yue2_master.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <new>

namespace geode::yue2 {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct Biquad {   // direct form I in double (the loudness meter's filters)
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    double run(double x) {
        const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1, x1 = x, y2 = y1, y1 = y;
        return y;
    }
};

// BS.1770's K-weighting for any rate (the 48 kHz coefficients of the standard re-derived, as libebur128 does): the
// head's high shelf, then the RLB high-pass
void k_weighting(int rate, Biquad &shelf, Biquad &hp) {
    {
        const double f0 = 1681.974450955533, G = 3.999843853973347, Q = 0.7071752369554196;
        const double K = std::tan(kPi * f0 / rate), Vh = std::pow(10.0, G / 20.0), Vb = std::pow(Vh, 0.4996667741545416);
        const double a0 = 1.0 + K / Q + K * K;
        shelf.b0 = (Vh + Vb * K / Q + K * K) / a0;
        shelf.b1 = 2.0 * (K * K - Vh) / a0;
        shelf.b2 = (Vh - Vb * K / Q + K * K) / a0;
        shelf.a1 = 2.0 * (K * K - 1.0) / a0;
        shelf.a2 = (1.0 - K / Q + K * K) / a0;
    }
    {
        const double f0 = 38.13547087602444, Q = 0.5003270373238773;
        const double K = std::tan(kPi * f0 / rate), a0 = 1.0 + K / Q + K * K;
        hp.b0 = 1.0, hp.b1 = -2.0, hp.b2 = 1.0;
        hp.a1 = 2.0 * (K * K - 1.0) / a0;
        hp.a2 = (1.0 - K / Q + K * K) / a0;
    }
}

inline int32_t quantize(float x, int bits) {
    const double full = static_cast<double>((1 << (bits - 1)) - 1);
    const double v = std::nearbyint(static_cast<double>(std::clamp(x, -1.0f, 1.0f)) * full);
    return static_cast<int32_t>(v);
}

// ---- FLAC -----------------------------------------------------------------------------------------------------------

class Bits {   // MSB first
public:
    void put(uint64_t v, int n) {   // n <= 57
        for (int i = n - 1; i >= 0; --i) bit(static_cast<int>((v >> i) & 1u));
    }
    void bit(int b) {
        acc_ = static_cast<uint8_t>((acc_ << 1) | (b & 1));
        if (++fill_ == 8) bytes_.push_back(acc_), acc_ = 0, fill_ = 0;
    }
    void zeros(uint64_t n) {
        while (n && fill_) bit(0), --n;
        bytes_.insert(bytes_.end(), static_cast<size_t>(n / 8), 0);
        for (uint64_t i = 0; i < n % 8; ++i) bit(0);
    }
    void put_signed(int64_t v, int n) { put(static_cast<uint64_t>(v) & ((n == 64 ? 0 : (1ull << n)) - 1ull), n); }
    void rice(uint32_t u, int k) {
        zeros(u >> k);
        bit(1);
        if (k) put(u & ((1u << k) - 1u), k);
    }
    void align() {
        while (fill_) bit(0);
    }
    std::vector<uint8_t> &bytes() { return bytes_; }

private:
    std::vector<uint8_t> bytes_;
    uint8_t acc_ = 0;
    int fill_ = 0;
};

uint8_t crc8(const uint8_t *p, size_t n) {
    uint8_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int b = 0; b < 8; ++b) c = static_cast<uint8_t>((c & 0x80) ? (c << 1) ^ 0x07 : c << 1);
    }
    return c;
}
uint16_t crc16(const uint8_t *p, size_t n) {
    uint16_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        c ^= static_cast<uint16_t>(p[i] << 8);
        for (int b = 0; b < 8; ++b) c = static_cast<uint16_t>((c & 0x8000) ? (c << 1) ^ 0x8005 : c << 1);
    }
    return c;
}

// the residual of fixed predictor `order` for x[0..n) (x[-order..-1] not used: the first `order` are warm-up)
void fixed_residual(const int64_t *x, int n, int order, int64_t *e) {
    for (int i = order; i < n; ++i) {
        switch (order) {
            case 0: e[i] = x[i]; break;
            case 1: e[i] = x[i] - x[i - 1]; break;
            case 2: e[i] = x[i] - 2 * x[i - 1] + x[i - 2]; break;
            case 3: e[i] = x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3]; break;
            default: e[i] = x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4]; break;
        }
    }
}

inline uint64_t zigzag(int64_t r) { return r >= 0 ? static_cast<uint64_t>(r) << 1 : (static_cast<uint64_t>(-(r + 1)) << 1) | 1u; }

constexpr int kMaxRice = 30;   // coding method 01 (5-bit parameters); 31 = escape (not used)

// the cheapest Rice parameter for u[0..n) and its bits
std::pair<int, uint64_t> best_rice(const uint64_t *u, int n) {
    if (n == 0) return {0, 0};
    uint64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += u[i];
    int k0 = 0;
    const uint64_t mean = sum / static_cast<uint64_t>(n);
    while (k0 < kMaxRice && (1ull << (k0 + 1)) <= mean) ++k0;
    int best_k = 0;
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (int k = std::max(0, k0 - 1); k <= std::min(kMaxRice, k0 + 1); ++k) {
        uint64_t bits = static_cast<uint64_t>(n) * static_cast<uint64_t>(k + 1);
        for (int i = 0; i < n; ++i) bits += u[i] >> k;
        if (bits < best) best = bits, best_k = k;
    }
    return {best_k, best};
}

struct Plan {   // one subframe's coding
    int order = 0, porder = 0;
    uint64_t bits = std::numeric_limits<uint64_t>::max();
    std::vector<int> k;
};

// the residual's best partition order (0..8) for predictor `order`
Plan plan_residual(const std::vector<uint64_t> &u, int n, int order) {
    Plan best;
    best.order = order;
    for (int po = 0; po <= 8; ++po) {
        const int parts = 1 << po;
        if (n % parts || (n >> po) <= order) break;
        Plan p;
        p.order = order, p.porder = po;
        p.bits = 2 + 4;
        p.k.resize(static_cast<size_t>(parts));
        for (int j = 0; j < parts; ++j) {
            const int a = j == 0 ? order : j * (n >> po), b = (j + 1) * (n >> po);
            const auto r = best_rice(u.data() + a, b - a);
            p.k[static_cast<size_t>(j)] = r.first;
            p.bits += 5 + r.second;
        }
        if (p.bits < best.bits) best = p;
    }
    return best;
}

// the best subframe for x (sample_bits each) and its bits; e / u: scratch
Plan plan_channel(const int64_t *x, int n, int sample_bits, std::vector<int64_t> &e, std::vector<uint64_t> &u) {
    Plan best;
    e.resize(static_cast<size_t>(n));
    u.resize(static_cast<size_t>(n));
    for (int order = 0; order <= 4 && order < n; ++order) {
        fixed_residual(x, n, order, e.data());
        for (int i = order; i < n; ++i) u[static_cast<size_t>(i)] = zigzag(e[static_cast<size_t>(i)]);
        Plan p = plan_residual(u, n, order);
        p.bits += 8 + static_cast<uint64_t>(order) * static_cast<uint64_t>(sample_bits);
        if (p.bits < best.bits) best = p;
    }
    const uint64_t verbatim = 8 + static_cast<uint64_t>(n) * static_cast<uint64_t>(sample_bits);
    if (verbatim < best.bits) best = Plan{}, best.order = -1, best.bits = verbatim;
    return best;
}

void write_subframe(Bits &w, const int64_t *x, int n, int sample_bits, const Plan &p, std::vector<int64_t> &e) {
    if (p.order < 0) {   // verbatim
        w.put(0, 1), w.put(1, 6), w.put(0, 1);
        for (int i = 0; i < n; ++i) w.put_signed(x[i], sample_bits);
        return;
    }
    w.put(0, 1), w.put(static_cast<uint64_t>(8 | p.order), 6), w.put(0, 1);
    for (int i = 0; i < p.order; ++i) w.put_signed(x[i], sample_bits);
    e.resize(static_cast<size_t>(n));
    fixed_residual(x, n, p.order, e.data());
    w.put(1, 2);   // Rice, 5-bit parameters
    w.put(static_cast<uint64_t>(p.porder), 4);
    const int parts = 1 << p.porder;
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? p.order : j * (n >> p.porder), b = (j + 1) * (n >> p.porder);
        const int k = p.k[static_cast<size_t>(j)];
        w.put(static_cast<uint64_t>(k), 5);
        for (int i = a; i < b; ++i) w.rice(static_cast<uint32_t>(zigzag(e[static_cast<size_t>(i)])), k);
    }
}

void frame_number(Bits &w, uint64_t v) {
    if (v < 0x80) {
        w.put(v, 8);
        return;
    }
    int n = 2;   // bytes: the first carries 7 - n bits, each further one 6
    while (v >= (1ull << (6 * (n - 1) + (7 - n)))) ++n;
    w.put(((0xFFu << (8 - n)) & 0xFFu) | static_cast<unsigned>(v >> (6 * (n - 1))), 8);
    for (int i = n - 2; i >= 0; --i) w.put(0x80u | ((v >> (6 * i)) & 0x3Fu), 8);
}

}  // namespace

double integrated_lufs(const float *audio, int64_t frames, int channels, int rate) {
    if (frames <= 0 || channels <= 0 || rate <= 0) return -std::numeric_limits<double>::infinity();
    const int64_t block = static_cast<int64_t>(std::llround(0.4 * rate)), hop = static_cast<int64_t>(std::llround(0.1 * rate));
    std::vector<double> sq(static_cast<size_t>(frames), 0.0);   // the channels' K-weighted squares summed (gains 1)
    for (int ch = 0; ch < channels; ++ch) {
        Biquad shelf, hp;
        k_weighting(rate, shelf, hp);
        for (int64_t i = 0; i < frames; ++i) {
            const double y = hp.run(shelf.run(static_cast<double>(audio[i * channels + ch])));
            sq[static_cast<size_t>(i)] += y * y;
        }
    }
    std::vector<double> prefix(static_cast<size_t>(frames) + 1, 0.0);
    for (int64_t i = 0; i < frames; ++i) prefix[static_cast<size_t>(i + 1)] = prefix[static_cast<size_t>(i)] + sq[static_cast<size_t>(i)];
    std::vector<double> z;
    for (int64_t s = 0; s + block <= frames; s += hop)
        z.push_back((prefix[static_cast<size_t>(s + block)] - prefix[static_cast<size_t>(s)]) / static_cast<double>(block));
    const double abs_gate = std::pow(10.0, (-70.0 + 0.691) / 10.0);
    double sum = 0.0;
    int64_t n = 0;
    for (const double e : z)
        if (e > abs_gate) sum += e, ++n;
    if (!n) return -std::numeric_limits<double>::infinity();
    const double rel_gate = (sum / static_cast<double>(n)) * std::pow(10.0, -10.0 / 10.0);
    double kept = 0.0;
    int64_t m = 0;
    for (const double e : z)
        if (e > abs_gate && e > rel_gate) kept += e, ++m;
    if (!m) return -std::numeric_limits<double>::infinity();
    return -0.691 + 10.0 * std::log10(kept / static_cast<double>(m));
}

double sample_peak_db(const float *audio, int64_t frames, int channels) {
    float peak = 0.0f;
    for (int64_t i = 0; i < frames * channels; ++i) peak = std::max(peak, std::fabs(audio[i]));
    return peak > 0.0f ? 20.0 * std::log10(static_cast<double>(peak)) : -std::numeric_limits<double>::infinity();
}

MasterResult master(std::vector<float> &audio, int channels, int rate, const MasterOptions &o) {
    MasterResult r;
    const int64_t frames = static_cast<int64_t>(audio.size()) / channels;
    r.lufs_before = integrated_lufs(audio.data(), frames, channels, rate);
    r.peak_before_db = sample_peak_db(audio.data(), frames, channels);
    if (std::isfinite(o.fixed_gain_db)) {   // (an edit's: its song's gain, not its own audio's loudness)
        r.wanted_db = r.gain_db = o.fixed_gain_db;
    } else {
        r.wanted_db = std::isfinite(r.lufs_before) ? o.target_lufs - r.lufs_before : 0.0;
        r.gain_db = std::min(r.wanted_db, o.max_gain_db);
        r.capped = r.wanted_db > o.max_gain_db;
    }
    const float g = static_cast<float>(std::pow(10.0, r.gain_db / 20.0));
    for (float &x : audio) x *= g;
    // the limiter: the gain each frame needs (channels linked), its minimum over the look-ahead window (the frame and
    // the L before it, so every frame of the window is covered), a moving average over L (a smooth attack that still
    // never exceeds what any frame it covers needs), a one-pole release; the output delayed by L
    const int L = std::max(1, static_cast<int>(std::lround(o.attack_ms * 1e-3 * rate)));
    const double rel = 1.0 - std::exp(-1.0 / (o.release_ms * 1e-3 * rate));
    const int64_t total = frames + L;
    std::vector<float> need(static_cast<size_t>(total), 1.0f);
    for (int64_t i = 0; i < frames; ++i) {
        float peak = 0.0f;
        for (int ch = 0; ch < channels; ++ch) peak = std::max(peak, std::fabs(audio[static_cast<size_t>(i * channels + ch)]));
        need[static_cast<size_t>(i)] = peak > o.limit ? static_cast<float>(o.limit / peak) : 1.0f;
    }
    std::vector<float> out(audio.size());
    std::deque<std::pair<int64_t, float>> win;   // the window's minimum (a monotonic deque)
    std::deque<float> avg;
    double avg_sum = 0.0, gain = 1.0;
    for (int64_t n = 0; n < total; ++n) {
        // s[n] = min(need[n - L .. n]) - the input frames n - L .. n (n - L is the one leaving the delay line now)
        const float v = need[static_cast<size_t>(n)];
        while (!win.empty() && win.back().second >= v) win.pop_back();
        win.emplace_back(n, v);
        while (win.front().first < n - L) win.pop_front();
        const float s = win.front().second;
        avg.push_back(s), avg_sum += s;
        if (static_cast<int>(avg.size()) > L) avg_sum -= avg.front(), avg.pop_front();
        // (before L frames have passed, the frames before the song count as needing no cut: gain 1)
        const double a = (avg_sum + static_cast<double>(L - static_cast<int>(avg.size()))) / L;
        gain = a < gain ? a : gain + (a - gain) * rel;
        const int64_t src = n - L;
        if (src < 0) continue;
        for (int ch = 0; ch < channels; ++ch)
            out[static_cast<size_t>(src * channels + ch)] = static_cast<float>(audio[static_cast<size_t>(src * channels + ch)] * gain);
    }
    audio.swap(out);
    r.lufs_after = integrated_lufs(audio.data(), frames, channels, rate);
    r.peak_after_db = sample_peak_db(audio.data(), frames, channels);
    return r;
}

bool write_wav(const std::string &path, const float *audio, int64_t frames, int channels, int rate, int bits) {
    if (bits != 16 && bits != 24 && bits != 32) return false;
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t bytes = static_cast<uint32_t>(bits / 8), data = static_cast<uint32_t>(frames * channels * bytes);
    const auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    const auto u16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
    std::fwrite("RIFF", 1, 4, f), u32(36 + data), std::fwrite("WAVEfmt ", 1, 8, f), u32(16);
    u16(bits == 32 ? 3 : 1), u16(static_cast<uint16_t>(channels)), u32(static_cast<uint32_t>(rate));
    u32(static_cast<uint32_t>(rate) * channels * bytes), u16(static_cast<uint16_t>(channels * bytes)), u16(static_cast<uint16_t>(bits));
    std::fwrite("data", 1, 4, f), u32(data);
    std::vector<uint8_t> buf;
    constexpr int64_t kPiece = 1 << 16;
    for (int64_t a = 0; a < frames * channels; a += kPiece) {
        const int64_t n = std::min(kPiece, frames * channels - a);
        buf.resize(static_cast<size_t>(n) * bytes);
        for (int64_t i = 0; i < n; ++i) {
            uint8_t *o = &buf[static_cast<size_t>(i) * bytes];
            if (bits == 32) {
                std::memcpy(o, &audio[a + i], 4);
            } else {
                const int32_t q = quantize(audio[a + i], bits);
                for (uint32_t b = 0; b < bytes; ++b) o[b] = static_cast<uint8_t>((static_cast<uint32_t>(q) >> (8 * b)) & 0xFF);
            }
        }
        std::fwrite(buf.data(), 1, buf.size(), f);
    }
    const bool ok = std::ferror(f) == 0;
    return std::fclose(f) == 0 && ok;
}

bool write_flac(const std::string &path, const float *audio, int64_t frames, int channels, int rate, int bits) {
    if ((bits != 16 && bits != 24) || channels < 1 || channels > 2 || rate <= 0 || rate >= (1 << 20)) return false;
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    constexpr int kBlock = 4096;
    {   // "fLaC" + STREAMINFO (the last metadata block; frame sizes and MD5 unknown: zeros)
        Bits w;
        w.put('f', 8), w.put('L', 8), w.put('a', 8), w.put('C', 8);
        w.put(1, 1), w.put(0, 7), w.put(34, 24);
        w.put(kBlock, 16), w.put(kBlock, 16), w.put(0, 24), w.put(0, 24);
        w.put(static_cast<uint64_t>(rate), 20), w.put(static_cast<uint64_t>(channels - 1), 3), w.put(static_cast<uint64_t>(bits - 1), 5);
        w.put(static_cast<uint64_t>(frames), 36);
        for (int i = 0; i < 16; ++i) w.put(0, 8);
        std::fwrite(w.bytes().data(), 1, w.bytes().size(), f);
    }
    const int rate_code = rate == 48000 ? 0xA : rate == 44100 ? 0x9 : rate == 96000 ? 0xB : 0x0;
    const int size_code = bits == 16 ? 0x4 : 0x6;
    std::vector<int64_t> ch[4], e;
    std::vector<uint64_t> u;
    for (int64_t start = 0, fn = 0; start < frames; start += kBlock, ++fn) {
        const int n = static_cast<int>(std::min<int64_t>(kBlock, frames - start));
        for (int c = 0; c < 4; ++c) ch[c].resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < channels; ++c)
                ch[c][static_cast<size_t>(i)] = quantize(audio[(start + i) * channels + c], bits);
        int assign = channels - 1;   // 0: mono, 1: L R
        Plan plans[2];
        const int64_t *src[2] = {ch[0].data(), channels == 2 ? ch[1].data() : nullptr};
        int sbits[2] = {bits, bits};
        if (channels == 2) {   // the cheapest of L R, L S, R S, M S (side needs one bit more)
            for (int i = 0; i < n; ++i) {
                const int64_t l = ch[0][static_cast<size_t>(i)], r = ch[1][static_cast<size_t>(i)];
                ch[2][static_cast<size_t>(i)] = (l + r) >> 1;   // mid
                ch[3][static_cast<size_t>(i)] = l - r;          // side
            }
            Plan p[4];
            for (int c = 0; c < 4; ++c) p[c] = plan_channel(ch[c].data(), n, c == 3 ? bits + 1 : bits, e, u);
            const uint64_t cost[4] = {p[0].bits + p[1].bits, p[0].bits + p[3].bits, p[1].bits + p[3].bits, p[2].bits + p[3].bits};
            const int pick = static_cast<int>(std::min_element(cost, cost + 4) - cost);
            static const int codes[4] = {0x1, 0x8, 0x9, 0xA};
            static const int a_of[4] = {0, 0, 3, 2}, b_of[4] = {1, 3, 1, 3};   // (right/side: side first, then right)
            assign = codes[pick];
            const int a = a_of[pick], b = b_of[pick];
            src[0] = ch[a].data(), src[1] = ch[b].data();
            plans[0] = p[a], plans[1] = p[b];
            sbits[0] = a == 3 ? bits + 1 : bits, sbits[1] = b == 3 ? bits + 1 : bits;
        } else {
            plans[0] = plan_channel(ch[0].data(), n, bits, e, u);
        }
        Bits w;
        w.put(0x3FFE, 14), w.put(0, 1), w.put(0, 1);   // sync, reserved, fixed block size
        const int bs_code = n == kBlock ? 0xC : n <= 256 ? 0x6 : 0x7;
        w.put(static_cast<uint64_t>(bs_code), 4), w.put(static_cast<uint64_t>(rate_code), 4);
        w.put(static_cast<uint64_t>(assign), 4), w.put(static_cast<uint64_t>(size_code), 3), w.put(0, 1);
        frame_number(w, static_cast<uint64_t>(fn));
        if (bs_code == 0x6) w.put(static_cast<uint64_t>(n - 1), 8);
        if (bs_code == 0x7) w.put(static_cast<uint64_t>(n - 1), 16);
        w.put(crc8(w.bytes().data(), w.bytes().size()), 8);
        for (int c = 0; c < channels; ++c) write_subframe(w, src[c], n, sbits[c], plans[c], e);
        w.align();
        const uint16_t crc = crc16(w.bytes().data(), w.bytes().size());
        w.put(crc, 16);
        std::fwrite(w.bytes().data(), 1, w.bytes().size(), f);
    }
    const bool ok = std::ferror(f) == 0;
    return std::fclose(f) == 0 && ok;
}

// ---- reading a song's file back -------------------------------------------------------------------------------------

namespace {

class BitReader {   // MSB first, over a whole file in memory; reading past its end marks it bad (zeros come back)
public:
    BitReader(const uint8_t *p, size_t n) : p_(p), n_(n) {}
    bool bad() const { return bad_; }
    size_t byte() const { return (pos_ + 7) >> 3; }   // the next whole byte
    void align() { pos_ = (pos_ + 7) & ~size_t{7}; }
    uint32_t get(int k) {   // k <= 32
        if (k <= 0) return 0;
        if (pos_ + static_cast<size_t>(k) > n_ * 8) {
            bad_ = true;
            pos_ = n_ * 8;
            return 0;
        }
        const size_t b = pos_ >> 3;
        const int off = static_cast<int>(pos_ & 7);
        uint64_t w = 0;
        for (size_t i = 0; i < 8; ++i) w = (w << 8) | (b + i < n_ ? p_[b + i] : 0u);
        pos_ += static_cast<size_t>(k);
        return static_cast<uint32_t>((w << off) >> (64 - k));
    }
    int32_t sget(int k) {   // two's complement, k <= 32
        if (k <= 0) return 0;
        const uint32_t u = get(k);
        if (k == 32) return static_cast<int32_t>(u);
        const uint32_t sign = 1u << (k - 1);
        return static_cast<int32_t>((u ^ sign)) - static_cast<int32_t>(sign);
    }
    uint32_t unary() {   // zeros before a one
        uint32_t q = 0;
        for (;;) {
            if (pos_ >= n_ * 8) {
                bad_ = true;
                return 0;
            }
            const int off = static_cast<int>(pos_ & 7);
            const uint32_t v = static_cast<uint8_t>(p_[pos_ >> 3] << off);
            if (v) {
                int lz = 0;
                while (!(v & (0x80u >> lz))) ++lz;
                q += static_cast<uint32_t>(lz);
                pos_ += static_cast<size_t>(lz) + 1;
                return q;
            }
            q += static_cast<uint32_t>(8 - off);
            pos_ += static_cast<size_t>(8 - off);
            if (q > (1u << 24)) {   // (no real residual is this long: a broken stream)
                bad_ = true;
                return 0;
            }
        }
    }

private:
    const uint8_t *p_;
    size_t n_, pos_ = 0;
    bool bad_ = false;
};

bool whole_file(const std::string &path, std::vector<uint8_t> &data, std::string &err) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "audio: no file " + path;
        return false;
    }
    data.clear();
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    return true;
}

// one subframe's [n] samples (bps: its own, the side channel's one more) into s; false: not a stream we read
bool flac_subframe(BitReader &br, int n, int bps, int64_t *s) {
    if (br.get(1) != 0) return false;
    const uint32_t type = br.get(6);
    int wasted = 0;
    if (br.get(1)) wasted = static_cast<int>(br.unary()) + 1;
    if (wasted >= bps) return false;
    const int b = bps - wasted;
    // a sample's range at this depth: the residuals are at most 2^54 (a unary run is capped), the restored samples are
    // held to it as they come, so no sum below can overflow
    const int64_t lim = int64_t{1} << (b - 1);
    const auto residual = [&](int order) {
        const uint32_t method = br.get(2);
        if (method > 1) return false;
        const int pbits = method == 0 ? 4 : 5;
        const uint32_t esc = method == 0 ? 15u : 31u;
        const int porder = static_cast<int>(br.get(4));
        const int part = n >> porder;
        if ((porder > 0 && (n & ((1 << porder) - 1))) || part < order) return false;
        int i = order;
        for (int p = 0; p < (1 << porder); ++p) {
            const int count = part - (p == 0 ? order : 0);
            const uint32_t k = br.get(pbits);
            if (k == esc) {
                const int raw = static_cast<int>(br.get(5));
                for (int j = 0; j < count; ++j) s[i++] = br.sget(raw);
            } else {
                for (int j = 0; j < count; ++j) {
                    const uint32_t q = br.unary();
                    const uint64_t u = (static_cast<uint64_t>(q) << k) | br.get(static_cast<int>(k));
                    s[i++] = static_cast<int64_t>(u >> 1) ^ -static_cast<int64_t>(u & 1);
                }
            }
            if (br.bad()) return false;
        }
        return true;
    };
    if (type == 0) {
        const int64_t v = br.sget(b);
        for (int i = 0; i < n; ++i) s[i] = v;
    } else if (type == 1) {
        for (int i = 0; i < n; ++i) s[i] = br.sget(b);
    } else if (type >= 8 && type <= 12) {
        const int order = static_cast<int>(type - 8);
        if (order > n) return false;
        for (int i = 0; i < order; ++i) s[i] = br.sget(b);
        if (!residual(order)) return false;
        for (int i = order; i < n; ++i) {
            switch (order) {
                case 1: s[i] += s[i - 1]; break;
                case 2: s[i] += 2 * s[i - 1] - s[i - 2]; break;
                case 3: s[i] += 3 * s[i - 1] - 3 * s[i - 2] + s[i - 3]; break;
                case 4: s[i] += 4 * s[i - 1] - 6 * s[i - 2] + 4 * s[i - 3] - s[i - 4]; break;
                default: break;
            }
            if (s[i] < -lim || s[i] >= lim) return false;
        }
    } else if (type >= 32) {
        const int order = static_cast<int>(type & 31) + 1;
        if (order > n) return false;
        for (int i = 0; i < order; ++i) s[i] = br.sget(b);
        const int precision = static_cast<int>(br.get(4)) + 1;
        if (precision == 16) return false;
        const int shift = br.sget(5);
        if (shift < 0) return false;
        int64_t coef[32];
        for (int i = 0; i < order; ++i) coef[i] = br.sget(precision);
        if (!residual(order)) return false;
        for (int i = order; i < n; ++i) {
            int64_t sum = 0;
            for (int j = 0; j < order; ++j) sum += coef[j] * s[i - 1 - j];
            s[i] += sum >> shift;
            if (s[i] < -lim || s[i] >= lim) return false;
        }
    } else {
        return false;
    }
    if (wasted)
        for (int i = 0; i < n; ++i) s[i] = static_cast<int64_t>(static_cast<uint64_t>(s[i]) << wasted);
    return !br.bad();
}

bool read_flac(const std::vector<uint8_t> &d, size_t at, AudioFile &o, std::string &err, bool header_only) {
    BitReader br(d.data() + at, d.size() - at);
    if (br.get(32) != 0x664C6143u) return (err = "audio: not FLAC"), false;   // "fLaC"
    bool info = false, last = false;
    int64_t total = 0;
    while (!last) {
        last = br.get(1) != 0;
        const uint32_t type = br.get(7), len = br.get(24);
        if (br.bad()) return (err = "audio: FLAC metadata cut short"), false;
        if (type == 0) {
            br.get(16), br.get(16), br.get(24), br.get(24);
            o.rate = static_cast<int>(br.get(20));
            o.channels = static_cast<int>(br.get(3)) + 1;
            o.bits = static_cast<int>(br.get(5)) + 1;
            total = static_cast<int64_t>(br.get(4)) << 32;
            total |= br.get(32);
            for (int i = 0; i < 4; ++i) br.get(32);   // (the MD5)
            if (len != 34) return (err = "audio: a broken FLAC STREAMINFO"), false;
            info = true;
        } else {
            for (uint32_t i = 0; i < len; ++i) br.get(8);
        }
    }
    if (!info || br.bad()) return (err = "audio: FLAC without its STREAMINFO"), false;
    if (o.bits < 8 || o.bits > 24 || o.channels > 2 || o.rate <= 0)
        return (err = "audio: a FLAC this reader does not take (" + std::to_string(o.bits) + "-bit, " +
                      std::to_string(o.channels) + " channels)"), false;
    // (20 minutes at 48 kHz: a song of ours is a few - and a small file of constant blocks must not ask for more)
    constexpr int64_t kMost = int64_t{48000} * 60 * 20;
    if (total > kMost) return (err = "audio: a FLAC longer than this reader takes"), false;
    o.frames = total;
    if (header_only) return true;
    o.audio.clear();
    const double full = static_cast<double>((1 << (o.bits - 1)) - 1);
    std::vector<int64_t> ch[2];
    int64_t got = 0;
    while (br.byte() + 2 < d.size() - at && (total == 0 || got < total)) {
        br.align();
        if (br.get(14) != 0x3FFE) return (err = "audio: FLAC lost its frames' sync"), false;
        br.get(1), br.get(1);
        const uint32_t bs = br.get(4), sr = br.get(4), ca = br.get(4), ss = br.get(3);
        br.get(1);
        {   // the frame's number (UTF-8-like)
            const uint32_t b0 = br.get(8);
            int extra = 0;
            for (uint32_t m = 0x80; m && (b0 & m); m >>= 1) ++extra;
            if (extra == 1 || extra > 7) return (err = "audio: a broken FLAC frame number"), false;
            for (int i = 1; i < extra; ++i) br.get(8);
        }
        int n = 0;
        if (bs == 1) n = 192;
        else if (bs >= 2 && bs <= 5) n = 576 << (bs - 2);
        else if (bs == 6) n = static_cast<int>(br.get(8)) + 1;
        else if (bs == 7) n = static_cast<int>(br.get(16)) + 1;
        else if (bs >= 8) n = 256 << (bs - 8);
        else return (err = "audio: a broken FLAC block size"), false;
        if (sr == 12) br.get(8);
        else if (sr == 13 || sr == 14) br.get(16);
        else if (sr == 15) return (err = "audio: a broken FLAC sample rate"), false;
        static const int size_of[8] = {0, 8, 12, -1, 16, 20, 24, -1};
        const int bps = ss == 0 ? o.bits : size_of[ss];
        if (bps != o.bits) return (err = "audio: a FLAC whose frames change their depth"), false;
        const int nch = ca < 8 ? static_cast<int>(ca) + 1 : ca <= 10 ? 2 : 0;
        if (nch != o.channels) return (err = "audio: a FLAC whose frames change their channels"), false;
        br.get(8);   // (the header's CRC-8)
        for (int c = 0; c < nch; ++c) {
            ch[c].resize(static_cast<size_t>(n));
            const bool side = (ca == 8 && c == 1) || (ca == 9 && c == 0) || (ca == 10 && c == 1);
            if (!flac_subframe(br, n, bps + (side ? 1 : 0), ch[c].data())) return (err = "audio: a FLAC subframe this reader does not take"), false;
        }
        br.align();
        br.get(16);   // (the frame's CRC-16)
        if (br.bad()) return (err = "audio: FLAC cut short"), false;
        const int keep = total > 0 ? static_cast<int>(std::min<int64_t>(n, total - got)) : n;
        for (int i = 0; i < keep; ++i) {
            int64_t l = ch[0][static_cast<size_t>(i)], r = nch == 2 ? ch[1][static_cast<size_t>(i)] : 0;
            if (ca == 8) r = l - r;                        // left, side
            else if (ca == 9) l = l + r;                   // side, right
            else if (ca == 10) {                           // mid, side
                const int64_t mid = (l * 2) | (r & 1);
                l = (mid + r) >> 1;
                r = (mid - r) >> 1;
            }
            o.audio.push_back(static_cast<float>(static_cast<double>(l) / full));
            if (nch == 2) o.audio.push_back(static_cast<float>(static_cast<double>(r) / full));
        }
        got += keep;
        if (got > kMost) return (err = "audio: a FLAC longer than this reader takes"), false;
    }
    if (total > 0 && got < total) return (err = "audio: FLAC cut short"), false;
    o.frames = got;
    return true;
}

bool read_wav(const std::vector<uint8_t> &d, AudioFile &o, std::string &err, bool header_only) {
    const auto u32 = [&](size_t i) { return static_cast<uint32_t>(d[i]) | d[i + 1] << 8 | d[i + 2] << 16 | static_cast<uint32_t>(d[i + 3]) << 24; };
    const auto u16 = [&](size_t i) { return static_cast<uint32_t>(d[i]) | d[i + 1] << 8; };
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) != 0 || std::memcmp(d.data() + 8, "WAVE", 4) != 0)
        return (err = "audio: not WAV"), false;
    uint32_t format = 0;
    size_t data = 0, bytes = 0;
    for (size_t i = 12; i + 8 <= d.size();) {
        const uint32_t len = u32(i + 4);
        if (std::memcmp(d.data() + i, "fmt ", 4) == 0 && len >= 16 && i + 8 + 16 <= d.size()) {
            format = u16(i + 8);
            o.channels = static_cast<int>(u16(i + 10));
            o.rate = static_cast<int>(u32(i + 12));
            o.bits = static_cast<int>(u16(i + 22));
            if (format == 0xFFFE && len >= 26 && i + 8 + 26 <= d.size()) format = u16(i + 8 + 24);
        } else if (std::memcmp(d.data() + i, "data", 4) == 0) {
            data = i + 8;
            bytes = std::min<size_t>(len, d.size() - data);
            break;
        }
        const size_t step = size_t{8} + len + (len & 1u);
        if (step > d.size() - i) break;
        i += step;
    }
    const bool pcm = format == 1 && (o.bits == 16 || o.bits == 24 || o.bits == 32), flt = format == 3 && o.bits == 32;
    if (!data || !(pcm || flt) || o.channels < 1 || o.channels > 2 || o.rate <= 0)
        return (err = "audio: a WAV this reader does not take"), false;
    const size_t width = static_cast<size_t>(o.bits / 8) * o.channels;
    o.frames = static_cast<int64_t>(bytes / width);
    if (header_only) return true;
    const size_t n = static_cast<size_t>(o.frames) * o.channels, w = static_cast<size_t>(o.bits / 8);
    o.audio.resize(n);
    const double full = static_cast<double>((int64_t{1} << (o.bits - 1)) - 1);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t *p = d.data() + data + i * w;
        if (flt) {
            std::memcpy(&o.audio[i], p, 4);
            continue;
        }
        uint32_t u = 0;
        for (size_t b = 0; b < w; ++b) u |= static_cast<uint32_t>(p[b]) << (8 * b);
        const int sh = 32 - o.bits;
        const int32_t q = static_cast<int32_t>(u << sh) >> sh;
        o.audio[i] = static_cast<float>(static_cast<double>(q) / full);
    }
    return true;
}

}  // namespace

bool read_audio(const std::string &path, AudioFile &out, std::string &err, bool header_only) try {
    std::vector<uint8_t> d;
    if (!whole_file(path, d, err)) return false;
    AudioFile o;
    size_t at = 0;
    if (d.size() >= 10 && std::memcmp(d.data(), "ID3", 3) == 0)   // (an ID3v2 tag before the stream)
        at = 10 + ((d[6] & 0x7Fu) << 21 | (d[7] & 0x7Fu) << 14 | (d[8] & 0x7Fu) << 7 | (d[9] & 0x7Fu));
    bool ok;
    if (d.size() >= at + 4 && std::memcmp(d.data() + at, "fLaC", 4) == 0) ok = read_flac(d, at, o, err, header_only);
    else if (d.size() >= 4 && std::memcmp(d.data(), "RIFF", 4) == 0) ok = read_wav(d, o, err, header_only);
    else return (err = "audio: " + path + " is neither FLAC nor WAV"), false;
    if (!ok) return false;
    out = std::move(o);
    return true;
} catch (const std::bad_alloc &) {
    err = "audio: " + path + " is too large to read";
    return false;
}

}  // namespace geode::yue2
