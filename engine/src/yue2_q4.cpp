// yue2_q4.cpp - see yue2_q4.hpp.
#include "yue2_q4.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "kernels.hpp"
#include "yue2_pack.hpp"
#include "yue2_q8gemm.hpp"

namespace geode::yue2 {

namespace {
// E4M3's non-negative finite values, codes 0x00 .. 0x7E in increasing order
const std::array<float, 127> &e4m3_positive() {
    static const std::array<float, 127> v = [] {
        std::array<float, 127> t{};
        for (int b = 0; b < 127; ++b) t[static_cast<size_t>(b)] = e4m3_value(static_cast<uint8_t>(b));
        return t;
    }();
    return v;
}

std::string shape_json(const std::vector<int64_t> &s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? "," : "") + std::to_string(s[i]);
    return o + "]";
}

std::string float_json(float x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", static_cast<double>(x));
    return b;
}

// a written file's data on the storage (fsync; _commit on Windows)
bool sync_file(FILE *f) {
#if defined(_WIN32)
    return _commit(_fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

// a directory's entries (a rename) on the storage; best effort (Windows has no directory sync)
void sync_dir(const std::string &dir) {
#if !defined(_WIN32)
    const int fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
#else
    (void)dir;
#endif
}
}  // namespace

uint8_t e4m3_byte(float x) {
    const auto &v = e4m3_positive();
    if (!(x > 0.0f)) return 0;
    if (x >= v[126]) return 126;
    // the last code at or below x, then the nearer of it and the next (a tie: the even code, as round-to-nearest-even)
    const size_t hi = static_cast<size_t>(std::upper_bound(v.begin(), v.end(), x) - v.begin());   // v[hi] > x
    const size_t lo = hi - 1;
    const float dl = x - v[lo], dh = v[hi] - x;
    if (dl < dh) return static_cast<uint8_t>(lo);
    if (dh < dl) return static_cast<uint8_t>(hi);
    return static_cast<uint8_t>(lo % 2 == 0 ? lo : hi);
}

void q4_block(const float *x, float s2, uint8_t &scale_byte, uint8_t codes[16]) {
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[i]));
    // qformats.best_block_scale's candidates in its order: the codes mapping amax to 15, then to 13, two either side
    int cand[10], nc = 0;
    for (const float t : {15.0f, 13.0f}) {
        const int b0 = e4m3_byte(amax / (t * s2));
        for (int k = -2; k <= 2; ++k) cand[nc++] = std::clamp(b0 + k, 0, 0x7E);
    }
    float best = std::numeric_limits<float>::infinity();
    uint8_t tmp[16];
    for (int c = 0; c < nc; ++c) {
        const float scale = e4m3_value(static_cast<uint8_t>(cand[c])) * s2, safe = scale == 0.0f ? 1.0f : scale;
        float err = 0.0f;
        for (int i = 0; i < 16; ++i) {
            const float y = x[i] / safe, a = std::min(std::fabs(y), 15.0f);
            // the nearest odd magnitude 2 idx + 1; an exact tie to the even idx (qformats.round_to_grid)
            const int idx = std::clamp(static_cast<int>(std::nearbyint((a - 1.0f) * 0.5f)), 0, 7);
            const bool neg = y < 0.0f;
            const float q = static_cast<float>(neg ? -(2 * idx + 1) : 2 * idx + 1) * scale;
            err += (q - x[i]) * (q - x[i]);
            tmp[i] = static_cast<uint8_t>(idx + (neg ? 8 : 0));
        }
        if (err < best) {   // (the first of equal errors, as argmin)
            best = err;
            scale_byte = static_cast<uint8_t>(cand[c]);
            std::copy(tmp, tmp + 16, codes);
        }
    }
}

void hmx8_block(const _Float16 *src, int64_t k, uint8_t *dst) {
    int8_t *codes = reinterpret_cast<int8_t *>(dst);
    _Float16 *scales = reinterpret_cast<_Float16 *>(dst + k * 32);
    const auto at = [](int o, int64_t c) { return (c >> 5) * 1024 + ((c & 31) >> 1) * 64 + o * 2 + (c & 1); };
    for (int64_t g = 0; g < k / 32; ++g)
        for (int o = 0; o < 32; ++o) {
            float amax = 0.0f;
            for (int64_t c = g * 32; c < g * 32 + 32; ++c) amax = std::max(amax, std::fabs(static_cast<float>(src[at(o, c)])));
            const _Float16 sh = static_cast<_Float16>(amax / 127.0f);
            const float s = static_cast<float>(sh);
            scales[g * 32 + o] = sh;
            for (int64_t c = g * 32; c < g * 32 + 32; ++c) {
                const float q = s > 0.0f ? std::clamp(std::nearbyint(static_cast<float>(src[at(o, c)]) / s), -127.0f, 127.0f) : 0.0f;
                codes[at(o, c)] = static_cast<int8_t>(q);
            }
        }
}

void clean_nar_q8_parts(const std::string &dir) {
    std::remove((dir + "/nar_q8.bin.part").c_str());
    std::remove((dir + "/" + kNarQ8Manifest + ".part").c_str());
    std::remove((dir + "/nar_c8.bin.part").c_str());
    std::remove((dir + "/" + kNarC8Manifest + ".part").c_str());
}

bool make_nar_q8(const std::string &dir, ThreadPool &pool, const std::function<void(int, int)> &progress,
                 const std::function<bool()> &cancelled, std::string &err, bool rotated) {
    const std::string fname = rotated ? "nar_c8.bin" : "nar_q8.bin", tpre = rotated ? "c8." : "h8.";
    const std::string bin = dir + "/" + fname, bin_part = bin + ".part";
    const std::string man = dir + "/" + (rotated ? kNarC8Manifest : kNarQ8Manifest), man_part = man + ".part";
    std::remove(man.c_str());   // (no manifest pairs with a half-written file, whatever happens below)
    clean_nar_q8_parts(dir);
    Pack pack;
    if (!pack.open(dir, err) || !pack.map("nar_f16.bin", err)) return false;
    const std::string pre = "h16.model.layers.";
    std::vector<std::pair<std::string, const TensorInfo *>> list;
    for (const auto &[name, t] : pack.tensors())
        if (name.rfind(pre, 0) == 0 && t.file >= 0 && pack.file_name(t.file) == "nar_f16.bin") list.emplace_back(name, &t);
    std::sort(list.begin(), list.end(), [](const auto &a, const auto &b) { return a.second->offset < b.second->offset; });
    if (list.empty()) {
        err = "nar_f16.bin: no layers";
        return false;
    }
    FILE *f = std::fopen(bin_part.c_str(), "wb");
    if (!f) {
        err = bin_part + ": cannot write";
        return false;
    }
    const auto fail = [&](const std::string &why) {
        std::fclose(f);
        std::remove(bin_part.c_str());
        err = why;
        return false;
    };
    uint64_t pos = 0;
    const auto put = [&](const uint8_t *p, size_t n, uint64_t align) {
        static const uint8_t zeros[4096] = {0};
        const uint64_t pad = (align - pos % align) % align;
        if (pad && std::fwrite(zeros, 1, static_cast<size_t>(pad), f) != pad) return std::numeric_limits<uint64_t>::max();
        pos += pad;
        const uint64_t at = pos;
        if (std::fwrite(p, 1, n, f) != n) return std::numeric_limits<uint64_t>::max();
        pos += n;
        return at;
    };
    const auto layer_of = [&](const std::string &name) { return std::atoi(name.c_str() + pre.size()); };
    std::ostringstream tj;
    const int n = static_cast<int>(list.size());
    int layer = -1;
    for (int i = 0; i < n; ++i) {
        if (cancelled && cancelled()) return fail("cancelled");
        const std::string &name = list[static_cast<size_t>(i)].first;
        const TensorInfo &t = *list[static_cast<size_t>(i)].second;
        const uint8_t *src = pack.data(t);
        if (!src) return fail(name + ": not readable");
        // a layer's first tensor on a page (NarDsp maps a layer's stretch from its first byte)
        const uint64_t align = layer_of(name) != layer ? 4096u : 128u;
        layer = layer_of(name);
        uint64_t at = 0;
        std::string type = "f32";
        if (t.hmx16) {
            const int64_t rows = t.shape[0], k = t.shape[1];
            if (k < 512 || k % 64) return fail(name + ": a depth hmx8 does not take");
            const int64_t blocks = rows / 32;
            std::vector<uint8_t> q(static_cast<size_t>(blocks * k * 34));
            pool.parallel_for(blocks, [&](int64_t a, int64_t b) {
                std::vector<_Float16> turned(rotated ? static_cast<size_t>(k) * 32u : 0u);
                float v[32];
                for (int64_t blk = a; blk < b; ++blk) {
                    const _Float16 *b16 = reinterpret_cast<const _Float16 *>(src + blk * k * 64);
                    if (rotated) {   // each row's groups of 32 inputs turned (HMX's element order: hmx8_block's at)
                        std::memcpy(turned.data(), b16, turned.size() * 2);
                        for (int64_t g = 0; g < k / 32; ++g)
                            for (int o = 0; o < 32; ++o) {
                                const auto at = [&](int c) {
                                    return static_cast<size_t>(g) * 1024u + static_cast<size_t>(c >> 1) * 64u + static_cast<size_t>(o) * 2u + static_cast<size_t>(c & 1);
                                };
                                for (int c = 0; c < 32; ++c) v[c] = static_cast<float>(turned[at(c)]);
                                fwht32(v);
                                for (int c = 0; c < 32; ++c) turned[at(c)] = static_cast<_Float16>(v[c]);
                            }
                        b16 = turned.data();
                    }
                    hmx8_block(b16, k, q.data() + blk * k * 34);
                }
            }, 4);
            at = put(q.data(), q.size(), align);
            type = "hmx8";
        } else if (t.type == DType::F32) {
            at = put(src, static_cast<size_t>(t.bytes), align);
        } else {
            return fail(name + ": a type the converter does not take");
        }
        if (at == std::numeric_limits<uint64_t>::max()) return fail(bin_part + ": write failed (the storage full?)");
        tj << (i ? ",\n" : "") << "  \"" << tpre << name.substr(4) << "\": {\"file\": \"" << fname << "\", \"offset\": " << at
           << ", \"type\": \"" << type << "\", \"shape\": " << shape_json(t.shape);
        if (type == "hmx8") tj << ", \"shift\": " << t.shift;
        tj << "}";
        if (progress) progress(i + 1, n);
    }
    const bool flushed = std::fflush(f) == 0 && sync_file(f);
    const bool closed = std::fclose(f) == 0;
    if (!flushed || !closed) {
        std::remove(bin_part.c_str());
        err = bin_part + ": write failed (the storage full?)";
        return false;
    }
    if (std::rename(bin_part.c_str(), bin.c_str()) != 0) {
        std::remove(bin_part.c_str());
        err = bin + ": rename failed";
        return false;
    }
    const std::string text = "{\"format\": \"geode-pack-extra\", \"version\": 1,\n \"files\": [{\"name\": \"" + fname + "\", "
                             "\"size\": " + std::to_string(pos) + "}],\n \"tensors\": {\n" + tj.str() + "\n }}\n";
    FILE *m = std::fopen(man_part.c_str(), "wb");
    bool written = m && std::fwrite(text.data(), 1, text.size(), m) == text.size() && std::fflush(m) == 0 && sync_file(m);
    if (m && std::fclose(m) != 0) written = false;
    if (!written) {
        std::remove(man_part.c_str());
        err = man_part + ": write failed (the storage full?)";
        return false;
    }
    sync_dir(dir);
    if (std::rename(man_part.c_str(), man.c_str()) != 0) {
        std::remove(man_part.c_str());
        err = man + ": rename failed";
        return false;
    }
    sync_dir(dir);
    return true;
}

void clean_ar_q4_parts(const std::string &dir) {
    std::remove((dir + "/ar_q4.bin.part").c_str());
    std::remove((dir + "/" + kArQ4Manifest + ".part").c_str());
}

bool make_ar_q4(const std::string &dir, ThreadPool &pool, const std::function<void(int, int)> &progress,
                const std::function<bool()> &cancelled, std::string &err) {
    const std::string bin = dir + "/ar_q4.bin", bin_part = bin + ".part";
    const std::string man = dir + "/" + kArQ4Manifest, man_part = man + ".part";
    std::remove(man.c_str());   // (no manifest pairs with a half-written file, whatever happens below)
    clean_ar_q4_parts(dir);
    Pack pack;
    if (!pack.open(dir, err) || !pack.map("ar_q8.bin", err)) return false;
    std::vector<std::pair<std::string, const TensorInfo *>> list;
    for (const auto &[name, t] : pack.tensors())
        if (name.rfind("q8.", 0) == 0 && t.file >= 0 && pack.file_name(t.file) == "ar_q8.bin") list.emplace_back(name, &t);
    std::sort(list.begin(), list.end(), [](const auto &a, const auto &b) { return a.second->offset < b.second->offset; });
    if (list.empty()) {
        err = "ar_q8.bin: no tensors";
        return false;
    }
    FILE *f = std::fopen(bin_part.c_str(), "wb");
    if (!f) {
        err = bin_part + ": cannot write";
        return false;
    }
    const auto fail = [&](const std::string &why) {
        std::fclose(f);
        std::remove(bin_part.c_str());
        err = why;
        return false;
    };
    uint64_t pos = 0;
    const auto put = [&](const uint8_t *p, size_t n) {
        static const uint8_t zeros[64] = {0};
        const uint64_t pad = (64 - pos % 64) % 64;   // (Pack wants 64-byte offsets)
        if (pad && std::fwrite(zeros, 1, static_cast<size_t>(pad), f) != pad) return std::numeric_limits<uint64_t>::max();
        pos += pad;
        const uint64_t at = pos;
        if (std::fwrite(p, 1, n, f) != n) return std::numeric_limits<uint64_t>::max();
        pos += n;
        return at;
    };
    std::ostringstream tj;
    const int n = static_cast<int>(list.size());
    for (int i = 0; i < n; ++i) {
        if (cancelled && cancelled()) return fail("cancelled");
        const std::string &name = list[static_cast<size_t>(i)].first;
        const TensorInfo &t = *list[static_cast<size_t>(i)].second;
        const std::string out = "q4." + name.substr(3);
        const uint8_t *src = pack.data(t);
        if (!src) return fail(name + ": not readable");
        std::string type = "f32";
        float global = 0.0f;
        uint64_t at = 0;
        if (t.type == DType::F32 || (t.type == DType::Q8 && name == "q8.lm_head.weight")) {   // as they are
            at = put(src, static_cast<size_t>(t.bytes));
            if (t.type == DType::Q8) type = "q8", global = t.global;
        } else if (t.type == DType::Q8) {
            const int64_t rows = t.shape[0], cols = t.shape[1], nb = cols / 16;
            const size_t rb8 = row_bytes(DType::Q8, cols), rb4 = row_bytes(DType::Q4, cols);
            const auto value = [&](const uint8_t *row, int64_t c) {   // the Q8 weight in float
                return static_cast<float>(reinterpret_cast<const int8_t *>(row)[c]) * e4m3_value(row[cols + c / 16]) * t.global;
            };
            // the matrix's largest |w| for the global scale (pack.py's quantize_rtn: x 1.25 / (15 x 448))
            std::vector<float> row_max(static_cast<size_t>(rows), 0.0f);
            pool.parallel_for(rows, [&](int64_t a, int64_t b) {
                for (int64_t r = a; r < b; ++r) {
                    const uint8_t *row = src + static_cast<size_t>(r) * rb8;
                    float m = 0.0f;
                    for (int64_t c = 0; c < cols; ++c) m = std::max(m, std::fabs(value(row, c)));
                    row_max[static_cast<size_t>(r)] = m;
                }
            }, 16);
            const float wmax = *std::max_element(row_max.begin(), row_max.end());
            const float s2 = wmax * 1.25f / (15.0f * 448.0f);
            if (!(s2 > 0.0f) || !std::isfinite(s2)) return fail(name + ": no scale");
            std::vector<uint8_t> q(static_cast<size_t>(rows) * rb4);
            pool.parallel_for(rows, [&](int64_t a, int64_t b) {
                float x[16];
                uint8_t codes[16];
                for (int64_t r = a; r < b; ++r) {
                    const uint8_t *row = src + static_cast<size_t>(r) * rb8;
                    uint8_t *o = q.data() + static_cast<size_t>(r) * rb4;
                    for (int64_t blk = 0; blk < nb; ++blk) {
                        for (int k = 0; k < 16; ++k) x[k] = value(row, blk * 16 + k);
                        uint8_t sb = 0;
                        q4_block(x, s2, sb, codes);
                        for (int k = 0; k < 8; ++k)   // (geodepack.pack_codes: value 2k low nibble, 2k + 1 high)
                            o[blk * 8 + k] = static_cast<uint8_t>(codes[2 * k] | (codes[2 * k + 1] << 4));
                        o[cols / 2 + blk] = sb;
                    }
                }
            }, 16);
            at = put(q.data(), q.size());
            type = "q4", global = s2;
        } else {
            return fail(name + ": a type the converter does not take");
        }
        if (at == std::numeric_limits<uint64_t>::max()) return fail(bin_part + ": write failed (the storage full?)");
        tj << (i ? ",\n" : "") << "  \"" << out << "\": {\"file\": \"ar_q4.bin\", \"offset\": " << at << ", \"type\": \""
           << type << "\", \"shape\": " << shape_json(t.shape);
        if (type != "f32") tj << ", \"global\": " << float_json(global);
        tj << "}";
        if (progress) progress(i + 1, n);
    }
    // flushed, on the storage and closed whatever fails (a descriptor left open keeps an unlinked file's blocks)
    const bool flushed = std::fflush(f) == 0 && sync_file(f);
    const bool closed = std::fclose(f) == 0;
    if (!flushed || !closed) {
        std::remove(bin_part.c_str());
        err = bin_part + ": write failed (the storage full?)";
        return false;
    }
    if (std::rename(bin_part.c_str(), bin.c_str()) != 0) {
        std::remove(bin_part.c_str());
        err = bin + ": rename failed";
        return false;
    }
    const std::string text = "{\"format\": \"geode-pack-extra\", \"version\": 1,\n \"files\": [{\"name\": \"ar_q4.bin\", "
                             "\"size\": " + std::to_string(pos) + "}],\n \"tensors\": {\n" + tj.str() + "\n }}\n";
    FILE *m = std::fopen(man_part.c_str(), "wb");
    bool written = m && std::fwrite(text.data(), 1, text.size(), m) == text.size() && std::fflush(m) == 0 && sync_file(m);
    if (m && std::fclose(m) != 0) written = false;
    if (!written) {
        std::remove(man_part.c_str());
        err = man_part + ": write failed (the storage full?)";
        return false;
    }
    // the bin's new name on the storage before the manifest's (a crash between: a bin and no manifest - made again)
    sync_dir(dir);
    if (std::rename(man_part.c_str(), man.c_str()) != 0) {
        std::remove(man_part.c_str());
        err = man + ": rename failed";
        return false;
    }
    sync_dir(dir);
    return true;
}

}  // namespace geode::yue2
