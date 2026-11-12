// yue2_view.cpp - a song's waveform and pitch classes (yue2_view.hpp)
#include "yue2_view.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

namespace geode::yue2 {

namespace {

constexpr double kPi = 3.14159265358979323846;

// in-place radix-2 FFT of n (a power of two) complex values
void fft(std::vector<std::complex<double>> &a, const std::vector<std::complex<double>> &tw) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * tw[k * step];
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
    }
}

}  // namespace

SongView view_of(const float *audio, int64_t frames, int channels, int rate) {
    SongView v;
    v.hop = std::max(1, static_cast<int>(std::lround(rate * 0.01)));
    size_t size = 1;
    while (static_cast<double>(size) < rate * 0.04) size <<= 1;   // (2048 at 48 kHz)
    const int64_t count = frames / v.hop + 1;
    v.peaks.assign(static_cast<size_t>(count), 0);
    v.classes.assign(static_cast<size_t>(count) * 12, 0);
    std::vector<float> mono(static_cast<size_t>(frames));
    for (int64_t i = 0; i < frames; ++i) {
        float s = 0, p = 0;
        for (int c = 0; c < channels; ++c) {
            const float x = audio[i * channels + c];
            s += x, p = std::max(p, std::fabs(x));
        }
        mono[static_cast<size_t>(i)] = s / static_cast<float>(channels);
        uint8_t &pk = v.peaks[static_cast<size_t>(i / v.hop)];
        pk = std::max<uint8_t>(pk, static_cast<uint8_t>(std::lround(std::min(1.0f, p) * 255.0f)));
    }
    // the bins kept and their pitch classes (round(69 + 12 log2(f / 440)) mod 12)
    std::vector<int> bins, cls;
    for (size_t k = 1; k <= size / 2; ++k) {
        const double f = static_cast<double>(k) * rate / static_cast<double>(size);
        if (f <= 80.0 || f >= 2000.0) continue;
        bins.push_back(static_cast<int>(k));
        const long m = std::lround(69.0 + 12.0 * std::log2(f / 440.0));
        cls.push_back(static_cast<int>(((m % 12) + 12) % 12));
    }
    std::vector<double> window(size);
    for (size_t i = 0; i < size; ++i) window[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(size));
    std::vector<std::complex<double>> tw(size / 2), a(size);
    for (size_t k = 0; k < size / 2; ++k) tw[k] = std::polar(1.0, -2.0 * kPi * static_cast<double>(k) / static_cast<double>(size));
    const int64_t half = static_cast<int64_t>(size / 2);
    for (int64_t h = 0; h < count; ++h) {
        const int64_t start = h * v.hop - half;   // (centred: the signal padded with size / 2 zeros either side)
        for (size_t i = 0; i < size; ++i) {
            const int64_t at = start + static_cast<int64_t>(i);
            a[i] = at >= 0 && at < frames ? mono[static_cast<size_t>(at)] * window[i] : 0.0;
        }
        fft(a, tw);
        double pc[12] = {};
        for (size_t b = 0; b < bins.size(); ++b) pc[cls[b]] += std::sqrt(std::abs(a[static_cast<size_t>(bins[b])]));
        double norm = 0;
        for (const double x : pc) norm += x * x;
        norm = std::max(std::sqrt(norm), 1e-9);
        for (int c = 0; c < 12; ++c)
            v.classes[static_cast<size_t>(h) * 12 + c] = static_cast<uint8_t>(std::lround(std::min(1.0, pc[c] / norm) * 255.0));
    }
    return v;
}

bool write_view(const std::string &path, const SongView &v, std::string &err) {
    const std::string part = path + ".part";
    std::FILE *f = std::fopen(part.c_str(), "wb");
    if (!f) return (err = "view: could not write " + part), false;
    const uint32_t hops = static_cast<uint32_t>(v.peaks.size()), hop = static_cast<uint32_t>(v.hop);
    bool ok = v.classes.size() == v.peaks.size() * 12 && std::fwrite("GYV1", 1, 4, f) == 4 && std::fwrite(&hops, 4, 1, f) == 1 &&
              std::fwrite(&hop, 4, 1, f) == 1 && (v.peaks.empty() || std::fwrite(v.peaks.data(), 1, v.peaks.size(), f) == v.peaks.size()) &&
              (v.classes.empty() || std::fwrite(v.classes.data(), 1, v.classes.size(), f) == v.classes.size());
    if (std::fclose(f) != 0) ok = false;
#if defined(_WIN32)
    if (ok) std::remove(path.c_str());
#endif
    if (ok && std::rename(part.c_str(), path.c_str()) == 0) return true;
    std::remove(part.c_str());
    err = "view: could not write " + path;
    return false;
}

bool read_view(const std::string &path, SongView &v, std::string &err) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return (err = "view: no file " + path), false;
    char magic[4];
    uint32_t hops = 0, hop = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "GYV1", 4) == 0 && std::fread(&hops, 4, 1, f) == 1 &&
              std::fread(&hop, 4, 1, f) == 1 && hop > 0;
    if (ok) {   // (the file's size against what the header says, before anything that size is allocated)
        const long at = std::ftell(f);
        ok = at >= 0 && std::fseek(f, 0, SEEK_END) == 0;
        const long end = ok ? std::ftell(f) : -1;
        ok = ok && end >= 0 && std::fseek(f, at, SEEK_SET) == 0 && static_cast<uint64_t>(end - at) == uint64_t{hops} * 13;
    }
    SongView o;
    if (ok) {
        o.hop = static_cast<int>(hop);
        o.peaks.resize(hops), o.classes.resize(static_cast<size_t>(hops) * 12);
        ok = (!hops || std::fread(o.peaks.data(), 1, hops, f) == hops) &&
             (!hops || std::fread(o.classes.data(), 1, o.classes.size(), f) == o.classes.size()) && std::fgetc(f) == EOF;
    }
    std::fclose(f);
    if (!ok) return (err = "view: " + path + " is not a song's view"), false;
    v = std::move(o);
    return true;
}

}  // namespace geode::yue2
