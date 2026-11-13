#include "yue2_lora.hpp"
#include "json.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <thread>
#if defined(_WIN32)
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace geode::yue2 {
namespace {
// Portable streaming SHA-256; known-answer tests in quant-test.
struct Sha {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64]{};
    size_t used = 0;
    uint64_t length = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t *p) {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
                   p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t a = w[i - 15], b = w[i - 2];
            w[i] =
                w[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + w[i - 7] + (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], z = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t = z + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            uint32_t u = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            z = g;
            g = f;
            f = e;
            e = d + t;
            d = c;
            c = b;
            b = a;
            a = t + u;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += z;
    }
    void add(const void *data, size_t n) {
        const auto *p = static_cast<const uint8_t *>(data);
        length += n;
        while (n) {
            size_t take = std::min(n, 64 - used);
            std::memcpy(buf + used, p, take);
            used += take;
            p += take;
            n -= take;
            if (used == 64) {
                block(buf);
                used = 0;
            }
        }
    }
    std::string finish() {
        uint64_t bits = length * 8;
        uint8_t pad = 0x80;
        add(&pad, 1);
        pad = 0;
        while (used != 56)
            add(&pad, 1);
        uint8_t b[8];
        for (int i = 0; i < 8; ++i)
            b[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        add(b, 8);
        std::ostringstream o;
        o << std::hex << std::setfill('0');
        for (auto v : h)
            o << std::setw(8) << v;
        return o.str();
    }
};
std::string hash_text(const std::string &s) {
    Sha h;
    h.add(s.data(), s.size());
    return h.finish();
}
bool hash_file(const RandomFile &f, std::string &hex, std::string &err, const LoraCancel &cancel) {
    Sha h;
    std::vector<uint8_t> buf(1u << 20);
    for (uint64_t at = 0; at < f.size();) {
        if (cancel && cancel()) {
            err = "cancelled";
            return false;
        }
        size_t n = static_cast<size_t>(std::min<uint64_t>(buf.size(), f.size() - at));
        if (!f.read_at(at, buf.data(), n)) {
            err = "lora: hashing file failed";
            return false;
        }
        h.add(buf.data(), n);
        at += n;
    }
    hex = h.finish();
    return true;
}
std::string quote(const std::string &s) {
    std::string o = "\"";
    const char *digits = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += char(c);
        } else if (c < 32) {
            o += "\\u00";
            o += digits[c >> 4];
            o += digits[c & 15];
        } else
            o += char(c);
    }
    return o + '"';
}
std::string dump(const json::Value &v) {
    switch (v.kind) {
    case json::Value::Kind::Null:
        return "null";
    case json::Value::Kind::Bool:
        return v.boolean ? "true" : "false";
    case json::Value::Kind::String:
        return quote(v.string);
    case json::Value::Kind::Number: {
        std::ostringstream s;
        s.imbue(std::locale::classic());
        s << std::setprecision(17) << v.number;
        return s.str();
    }
    case json::Value::Kind::Array: {
        std::string s = "[";
        for (auto &x : v.array) {
            if (s.size() > 1)
                s += ',';
            s += dump(x);
        }
        return s + "]";
    }
    case json::Value::Kind::Object: {
        std::string s = "{";
        for (auto &[k, x] : v.object) {
            if (s.size() > 1)
                s += ',';
            s += quote(k) + ":" + dump(x);
        }
        return s + "}";
    }
    }
    return "null";
}
bool read_text(const std::string &p, std::string &s) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0 || f.tellg() > (32 << 20))
        return false;
    s.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    return s.empty() || bool(f.read(s.data(), s.size()));
}
bool sync(FILE *f) {
    if (std::fflush(f))
        return false;
#if defined(_WIN32)
    return _commit(_fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}
void sync_directory(const std::string &dir) {
#if !defined(_WIN32)
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
#else
    (void)dir;
#endif
}
struct Lock {
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~Lock() {
        if (handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
    }
#else
    int fd = -1;
    ~Lock() {
        if (fd >= 0)
            close(fd);
    }
#endif
    bool acquire(const std::string &path, const LoraCancel &cancel, std::string &err) {
#if !defined(_WIN32)
        fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            err = "lora: cannot open cache lock";
            return false;
        }
#endif
        for (;;) {
            if (cancel && cancel()) {
                err = "cancelled";
                return false;
            }
#if defined(_WIN32)
            handle = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle != INVALID_HANDLE_VALUE)
                return true;
            if (GetLastError() != ERROR_SHARING_VIOLATION) {
                err = "lora: cannot open cache lock";
                return false;
            }
#else
            if (flock(fd, LOCK_EX | LOCK_NB) == 0)
                return true;
            if (errno != EWOULDBLOCK && errno != EAGAIN) {
                err = "lora: cannot acquire cache lock";
                return false;
            }
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};
const char *projections[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
float even(float f) {
    // Values outside the Q8 grid saturate. Clamp before conversion, independent of the process rounding mode.
    if (f <= -127.f)
        return -127.f;
    if (f >= 127.f)
        return 127.f;
    const int lo = static_cast<int>(std::floor(f));
    const float d = f - static_cast<float>(lo);
    return static_cast<float>(d < 0.5f ? lo : d > 0.5f ? lo + 1 : (lo % 2 == 0 ? lo : lo + 1));
}
uint8_t fp8(float x) {
    int lo = 0, hi = 126;
    if (!(x > 0))
        return 0;
    if (x >= 448)
        return 126;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (e4m3_value(static_cast<uint8_t>(mid)) <= x)
            lo = mid;
        else
            hi = mid;
    }
    float a = x - e4m3_value(static_cast<uint8_t>(lo)), b = e4m3_value(static_cast<uint8_t>(hi)) - x;
    return static_cast<uint8_t>(a < b ? lo : b < a ? hi : (lo % 2 ? hi : lo));
}
} // namespace
std::string lora_scale_text(float s) {
    if (s == 0)
        s = 0;
    std::ostringstream o;
    o.imbue(std::locale::classic());
    o << std::setprecision(std::numeric_limits<float>::max_digits10) << s;
    return o.str();
}
bool lora_sha256(const std::string &p, std::string &hex, std::string &err, const LoraCancel &cancel) {
    RandomFile f;
    return f.open(p, err) && hash_file(f, hex, err, cancel);
}
namespace {
// "size:mtime[:inode]" of a file (empty: it cannot be read). The files are the app's own (the adapter downloaded and
// verified, the pack, the merged copy written here); the inode tells a file replaced by a rename from the old one even
// when its size and time are the same.
std::string stamp_of(const std::string &p) {
#if !defined(_WIN32)
    struct stat st{};
    if (::stat(p.c_str(), &st) != 0)
        return {};
    return std::to_string(static_cast<long long>(st.st_size)) + ":" +
           std::to_string(static_cast<long long>(st.st_mtim.tv_sec)) + "." +
           std::to_string(static_cast<long long>(st.st_mtim.tv_nsec)) + ":" +
           std::to_string(static_cast<unsigned long long>(st.st_ino));
#else
    std::error_code ec;
    const auto size = std::filesystem::file_size(p, ec);
    if (ec)
        return {};
    const auto t = std::filesystem::last_write_time(p, ec);
    if (ec)
        return {};
    return std::to_string(size) + ":" + std::to_string(static_cast<long long>(t.time_since_epoch().count()));
#endif
}
} // namespace
bool lora_sha256_remembered(const std::string &p, const std::string &cache, std::string &hex, std::string &err,
                            const LoraCancel &cancel) {
    const std::string memo = cache + "/hashes.json", stamp = stamp_of(p);
    std::string text;
    json::Value m;
    std::string ignored;
    if (stamp.empty())
        return lora_sha256(p, hex, err, cancel); // (its own error)
    if (read_text(memo, text) && json::parse(text, m, ignored) && m.is_object() && m.object.count(p)) {
        const auto &e = m[p];
        if (e["stamp"].str() == stamp && e["sha256"].str().size() == 64) {
            hex = e["sha256"].str();
            return true;
        }
    }
    if (!lora_sha256(p, hex, err, cancel))
        return false;
    if (stamp_of(p) != stamp)
        return true; // (changed while read: this digest is not remembered)
    if (!m.is_object()) {
        m = json::Value{};
        m.kind = json::Value::Kind::Object;
    }
    json::Value e;
    e.kind = json::Value::Kind::Object;
    e.object["stamp"].kind = json::Value::Kind::String, e.object["stamp"].string = stamp;
    e.object["sha256"].kind = json::Value::Kind::String, e.object["sha256"].string = hex;
    m.object[p] = e;
    std::error_code ec;
    std::filesystem::create_directories(cache, ec);
    // (a temporary of this process's own: two writers never share one; the rename is atomic, the last one wins)
    const std::string part = memo + ".part" +
                             std::to_string(static_cast<unsigned long long>(
                                 std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
                                 static_cast<size_t>(std::chrono::steady_clock::now().time_since_epoch().count()))),
                      out = dump(m);
    if (FILE *f = std::fopen(part.c_str(), "wb")) {
        const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
        if (std::fclose(f) == 0 && ok)
            std::filesystem::rename(part, memo, ec); // (a memo that cannot be written only costs the next read)
    }
    return true;
}
bool validate_lora(const SafeTensors &a, const Pack *pack, int &rank, std::string &err) {
    const auto fail = [&](const std::string &s) {
        err = "lora: " + s;
        return false;
    };
    if (a.tensors().size() != 392)
        return fail("expected exactly 392 tensors (28 layers, seven A/B pairs)");
    rank = 0;
    for (int l = 0; l < 28; ++l)
        for (auto proj : projections) {
            std::string stem = "layers." + std::to_string(l) + "." + proj;
            auto ai = a.tensors().find(stem + ".lora_A"), bi = a.tensors().find(stem + ".lora_B");
            if (ai == a.tensors().end() || bi == a.tensors().end())
                return fail("missing pair " + stem);
            const auto &as = ai->second.shape, &bs = bi->second.shape;
            if (as.size() != 2 || bs.size() != 2 || as[0] < 1 || as[0] > 256 || bs[1] != as[0])
                return fail("invalid A/B shape or rank: " + stem);
            if (rank && rank != as[0])
                return fail("inconsistent rank: " + stem);
            rank = static_cast<int>(as[0]);
            if (pack) {
                auto *w = pack->find("q8.model." + stem + ".weight");
                if (!w || w->type != DType::Q8 || w->shape != std::vector<int64_t>{bs[0], as[1]} ||
                    pack->file_name(w->file) != "ar_q8.bin")
                    return fail("base shape/type mismatch: " + stem);
            }
        }
    if (pack && pack->config()["num_hidden_layers"].i64() != 28)
        return fail("expected 28 AR layers");
    return true;
}
bool lora_q8_encode(const float *x, int64_t rows, int64_t cols, std::vector<uint8_t> &out, float &global,
                    std::string &err, ThreadPool *pool) {
    if (rows <= 0 || cols <= 0 || cols % 64 || uint64_t(rows) > (uint64_t{1} << 30) / uint64_t(cols)) {
        err = "lora Q8: invalid shape";
        return false;
    }
    float big = 0;
    for (int64_t i = 0; i < rows * cols; ++i) {
        if (!std::isfinite(x[i])) {
            err = "lora Q8: nonfinite weight";
            return false;
        }
        big = std::max(big, std::fabs(x[i]));
    }
    // Python computes the scalar in binary64 and converts once to fp32.
    global = static_cast<float>(double(big) * 1.25 / (127.0 * 448.0));
    const int64_t rb = cols + cols / 16;
    out.resize(static_cast<size_t>(rows * rb));
    if (!(global > 0)) {
        // Python produces a zero global and device-dependent NaN-to-FP8 codes here; Pack cannot represent it.
        err = "lora Q8: zero/underflowed global scale (all-zero matrices are not valid packed Q8)";
        return false;
    }
    const auto run = [&](int64_t r0, int64_t r1) {
        for (int64_t row = r0; row < r1; ++row) {
            auto *dst = out.data() + row * rb;
            for (int64_t b = 0; b < cols / 16; ++b) {
                const float *v = x + row * cols + 16 * b;
                float mx = 0;
                for (int i = 0; i < 16; ++i)
                    mx = std::max(mx, std::fabs(v[i]));
                float best = std::numeric_limits<float>::infinity();
                uint8_t sb = 0;
                int8_t codes[16]{};
                for (float target : {127.f, 126.f}) {
                    int center = fp8(mx / (target * global));
                    for (int k = -2; k <= 2; ++k) {
                        uint8_t candidate = static_cast<uint8_t>(std::clamp(center + k, 0, 126));
                        float s = e4m3_value(candidate) * global, safe = s == 0 ? 1 : s;
                        float errors[16];
                        int8_t q[16];
                        for (int i = 0; i < 16; ++i) {
                            float level = std::clamp(even(v[i] / safe), -127.f, 127.f);
                            q[i] = static_cast<int8_t>(level);
                            float d = level * s - v[i];
                            errors[i] = d * d;
                        }
                        // Four-lane contiguous reduction, matching torch's 16-wide CPU sum.
                        float e0 = ((errors[0] + errors[4]) + errors[8]) + errors[12];
                        float e1 = ((errors[1] + errors[5]) + errors[9]) + errors[13];
                        float e2 = ((errors[2] + errors[6]) + errors[10]) + errors[14];
                        float e3 = ((errors[3] + errors[7]) + errors[11]) + errors[15];
                        float error = (e0 + e1) + (e2 + e3);
                        if (error < best) {
                            best = error;
                            sb = candidate;
                            std::copy(q, q + 16, codes);
                        }
                    }
                }
                std::memcpy(dst + 16 * b, codes, 16);
                dst[cols + b] = sb;
            }
        }
    };
    if (pool)
        pool->parallel_for(rows, run);
    else
        run(0, rows);
    return true;
}
namespace {
// Explicit IEEE binary32 -> binary16 RNE, independent of host ISA and floating rounding mode.
uint16_t half_rne(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
    const uint32_t exponent = (u >> 23) & 255, mantissa = u & 0x7fffff;
    if (exponent == 255)
        return static_cast<uint16_t>(sign | (mantissa ? 0x7e00 : 0x7c00));
    const int e = static_cast<int>(exponent) - 127 + 15;
    if (e >= 31)
        return static_cast<uint16_t>(sign | 0x7c00);
    if (e < -10)
        return sign;
    if (e <= 0) {
        const uint32_t significand = mantissa | 0x800000, shift = static_cast<uint32_t>(14 - e);
        uint32_t q = significand >> shift;
        const uint32_t rem = significand & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (q & 1)))
            ++q;
        return static_cast<uint16_t>(sign | q);
    }
    uint32_t q = mantissa >> 13, rem = mantissa & 8191;
    if (rem > 4096 || (rem == 4096 && (q & 1)))
        ++q;
    return static_cast<uint16_t>(sign | ((static_cast<uint32_t>(e) << 10) + q));
}
// binary16 -> binary32, exact: a normal value's bits moved over (the exponent rebased), a subnormal one m x 2^-24 (m
// below 2^10: exact in fp32). No ldexp: this runs for every one of ar_f16.bin's 1.4 G weights.
float half_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    float x;
    if (e == 0) {
        x = static_cast<float>(m) * 0x1p-24f;
        return (h & 0x8000) ? -x : x;
    }
    if (e == 31)
        return m ? std::numeric_limits<float>::quiet_NaN()
                 : ((h & 0x8000) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());
    const uint32_t u = sign | ((e + 112u) << 23) | (m << 13);
    std::memcpy(&x, &u, 4);
    return x;
}
size_t hmx_at(int64_t r, int64_t c, int64_t cols) {
    const int64_t k = std::max<int64_t>(cols, 512);
    return static_cast<size_t>((r / 32) * k * 32 + (c / 32) * 1024 + ((c % 32) / 2) * 64 + (r % 32) * 2 + c % 2);
}
// Both precision paths use exactly this rank loop and these separate fp32 multiply/adds (-ffp-contract=off).
bool add_lora(const SafeTensors &adapter, const std::string &stem, float scale, int64_t rows, int64_t cols,
              float *matrix, ThreadPool &pool, const LoraCancel &cancel, std::string &err) {
    std::vector<float> a, b;
    if (!adapter.floats(stem + ".lora_A", a, err) || !adapter.floats(stem + ".lora_B", b, err))
        return false;
    if (scale == 0)
        return true; // validate factors, but preserve zero-scale bytes/shift without rounding
    const int rank = static_cast<int>(adapter.tensors().at(stem + ".lora_A").shape[0]);
    pool.parallel_for(rows, [&](int64_t first, int64_t stop) {
        std::vector<float> delta(static_cast<size_t>(cols));
        for (int64_t r = first; r < stop; ++r) {
            if (cancel && cancel())
                return;
            std::fill(delta.begin(), delta.end(), 0.f);
            for (int k = 0; k < rank; ++k) {
                const float v = b[static_cast<size_t>(r * rank + k)];
                const float *ar = a.data() + k * cols;
                for (int64_t c = 0; c < cols; ++c)
                    delta[c] += v * ar[c];
            }
            float *mr = matrix + r * cols;
            for (int64_t c = 0; c < cols; ++c)
                mr[c] = mr[c] + scale * delta[c];
        }
    });
    if (cancel && cancel()) {
        err = "cancelled";
        return false;
    }
    return true;
}
struct Group {
    const char *name;
    int first, stop;
};
constexpr Group groups[] = {{"ar_qkv", 0, 3}, {"ar_o", 3, 4}, {"ar_gu", 4, 6}, {"ar_down", 6, 7}};
json::Value tensor_table(const Pack &pack, const std::string &file) {
    json::Value t;
    t.kind = json::Value::Kind::Object;
    for (const auto &[name, v] : pack.manifest()["tensors"].object)
        if (v["file"].str() == file)
            t.object.emplace(name, v);
    return t;
}
bool validate_f16(const SafeTensors &adapter, const Pack &pack, std::string &err) {
    for (int l = 0; l < 28; ++l) {
        const std::string layer = "layers." + std::to_string(l) + ".";
        for (const auto &group : groups) {
            int64_t rows = 0, cols = 0;
            for (int j = group.first; j < group.stop; ++j) {
                const auto &a = adapter.tensors().at(layer + projections[j] + ".lora_A");
                const auto &b = adapter.tensors().at(layer + projections[j] + ".lora_B");
                if (cols && cols != a.shape[1]) {
                    err = "lora fp16: fused input widths differ";
                    return false;
                }
                cols = a.shape[1];
                rows += b.shape[0];
            }
            const std::string name = "a16.model." + layer + group.name;
            const auto *t = pack.find(name);
            if (!t || !t->hmx16 || t->shape != std::vector<int64_t>{rows, cols} ||
                pack.file_name(t->file) != "ar_f16.bin") {
                err = "lora fp16: base shape/type mismatch: " + name;
                return false;
            }
        }
        // The DSP loads these in the same page-aligned stretch as the four products.
        const char *norms[] = {"input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                               "post_attention_layernorm.weight"};
        const int64_t h = pack.config()["hidden_size"].i64(), hd = pack.config()["head_dim"].i64();
        for (int j = 0; j < 4; ++j) {
            const std::string name = "a16.model." + layer + norms[j];
            const auto *t = pack.find(name);
            if (!t || t->type != DType::F32 || t->shape != std::vector<int64_t>{j == 1 || j == 2 ? hd : h} ||
                pack.file_name(t->file) != "ar_f16.bin") {
                err = "lora fp16: missing/invalid norm: " + name;
                return false;
            }
        }
    }
    return true;
}
// Shared cache protocol: exact metadata checksum, stamp fast path, digest fallback, manifest published last.
bool part_hit(const std::string &dir, const std::string &manifest, const std::string &file, const std::string &key,
              const std::string &sha, float scale, const LoraCancel &cancel, bool &hit, std::string &err) {
    hit = false;
    std::string text;
    json::Value saved;
    const std::string man = dir + "/" + manifest, bin = dir + "/" + file;
    if (!read_text(man, text) || !json::parse(text, saved, err) || saved["key"].str() != key ||
        saved["format"].str() != "geode-pack-extra" || saved["version"].i64() != 1 || saved["converter"].i64() != 1 ||
        saved["adapter_sha256"].str() != sha || !saved["scale"].is_number() ||
        static_cast<float>(saved["scale"].f64()) != scale || !saved["tensors"].is_object() ||
        saved["tensor_sha256"].str() != hash_text(dump(saved["tensors"])) || saved["files"].size() != 1 ||
        saved["files"].at(0)["name"].str() != file)
        return true;
    RandomFile f;
    if (!f.open(bin, err) || double(f.size()) != saved["files"].at(0)["size"].f64())
        return true;
    const std::string stamp = stamp_of(bin);
    if (!stamp.empty() && saved["stamp"].str() == stamp) {
        hit = true;
        return true;
    }
    std::string digest;
    if (!hash_file(f, digest, err, cancel))
        return false;
    if (digest != saved["files"].at(0)["sha256"].str() || stamp.empty() || stamp_of(bin) != stamp)
        return true;
    saved.object["stamp"].kind = json::Value::Kind::String;
    saved.object["stamp"].string = stamp;
    const std::string fresh = dump(saved);
    if (FILE *mf = std::fopen((man + ".part").c_str(), "wb")) {
        const bool ok = std::fwrite(fresh.data(), 1, fresh.size(), mf) == fresh.size() && sync(mf);
        if (std::fclose(mf) == 0 && ok) {
            std::error_code ec;
            std::filesystem::rename(man + ".part", man, ec);
            sync_directory(dir);
        }
    }
    hit = true;
    return true;
}
bool write_at(FILE *f, uint64_t offset, const std::vector<uint8_t> &bytes, std::string &err) {
#if defined(_WIN32)
    const int seek = _fseeki64(f, static_cast<int64_t>(offset), SEEK_SET);
#else
    const int seek = fseeko(f, static_cast<off_t>(offset), SEEK_SET);
#endif
    if (seek || std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        err = "lora: writing projection failed";
        return false;
    }
    return true;
}
bool build_part(Pack &pack, const SafeTensors &adapter, const std::string &base, const LoraSpec &spec,
                const std::string &dir, const std::string &key, const std::string &sha, const std::string &source_stamp,
                const std::string &adapter_stamp, bool f16, int total, ThreadPool &pool, const LoraProgress &progress,
                const LoraCancel &cancel, std::string &err) {
    const std::string file = f16 ? "ar_f16.bin" : "ar_q8.bin", manifest = f16 ? "manifest.f16.json" : "manifest.json";
    const int offset = f16 ? 28 : 0;
    if (cancel && cancel()) {
        err = "cancelled";
        return false;
    }
    if (stamp_of(spec.path) != adapter_stamp || stamp_of(base + "/" + file) != source_stamp) {
        err = "lora: source changed before cache lookup";
        return false;
    }
    bool hit;
    if (!part_hit(dir, manifest, file, key, sha, spec.scale, cancel, hit, err))
        return false;
    if (hit) {
        if (progress)
            progress(offset + 28, total);
        if (cancel && cancel()) {
            err = "cancelled";
            return false;
        }
        if (stamp_of(spec.path) != adapter_stamp || stamp_of(base + "/" + file) != source_stamp) {
            err = "lora: source changed during cache lookup";
            return false;
        }
        err.clear();
        return true;
    }
    if (cancel && cancel()) {
        err = "cancelled";
        return false;
    }
    // Keep both precisions of the selected entry; only other entry directories are pruned.
    std::error_code ec;
    for (const auto &d : std::filesystem::directory_iterator(spec.cache, ec))
        if (d.is_directory() && d.path().filename() != std::filesystem::path(dir).filename()) {
            std::error_code rm;
            std::filesystem::remove_all(d.path(), rm);
        }
    ec.clear();
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        err = "lora: creating cache entry: " + ec.message();
        return false;
    }
    const std::string man = dir + "/" + manifest, bin = dir + "/" + file, part = bin + ".part";
    std::filesystem::remove(man, ec);
    FILE *f = std::fopen(part.c_str(), "wb");
    if (!f) {
        err = "lora: cannot write " + part;
        return false;
    }
    struct Output {
        FILE *f;
        std::string path;
        ~Output() {
            if (f)
                std::fclose(f);
            std::remove(path.c_str());
        }
    } output{f, part};
    RandomFile source;
    if (!source.open(base + "/" + file, err))
        return false;
    std::vector<uint8_t> copy(1u << 20);
    for (uint64_t at = 0; at < source.size();) {
        if (cancel && cancel()) {
            err = "cancelled";
            return false;
        }
        const size_t n = static_cast<size_t>(std::min<uint64_t>(copy.size(), source.size() - at));
        if (!source.read_at(at, copy.data(), n) || std::fwrite(copy.data(), 1, n, f) != n) {
            err = "lora: copying base AR failed";
            return false;
        }
        at += n;
    }
    json::Value tensors = tensor_table(pack, file);
    if (progress)
        progress(offset, total);
    for (int l = 0; l < 28; ++l) {
        const std::string layer = "layers." + std::to_string(l) + ".";
        const int products = f16 ? 4 : 7;
        for (int j = 0; j < products; ++j) {
            if (cancel && cancel()) {
                err = "cancelled";
                return false;
            }
            const std::string name =
                f16 ? "a16.model." + layer + groups[j].name : "q8.model." + layer + projections[j] + ".weight";
            const auto *t = pack.find(name);
            const int64_t rows = t->shape[0], cols = t->shape[1];
            if (uint64_t(rows) > (uint64_t{1} << 28) / uint64_t(cols)) {
                err = "lora: matrix too large";
                return false;
            }
            std::vector<uint8_t> raw(static_cast<size_t>(t->bytes));
            if (!pack.read_span(t->file, t->offset, t->bytes, raw.data())) {
                err = "lora: reading " + name;
                return false;
            }
            std::vector<float> merged(spec.scale == 0 ? 0 : static_cast<size_t>(rows * cols));
            // (x 2^-shift as a product with the power of two: exact - the smallest fp16 x 2^-24 is a normal fp32 -
            // so the same bits as ldexp, without its cost per weight)
            const float down = f16 ? std::ldexp(1.f, -t->shift) : 1.f;
            if (spec.scale != 0) {
                pool.parallel_for(rows, [&](int64_t a, int64_t b) {
                    for (int64_t r = a; r < b; ++r)
                        for (int64_t c = 0; c < cols; ++c) {
                            if (f16) {
                                const size_t at = 2 * hmx_at(r, c, cols);
                                uint16_t h = uint16_t(raw[at]) | (uint16_t(raw[at + 1]) << 8);
                                merged[r * cols + c] = half_float(h) * down;
                            } else {
                                const auto *w = raw.data() + r * (cols + cols / 16);
                                merged[r * cols + c] = static_cast<float>(static_cast<int8_t>(w[c])) *
                                                       (e4m3_value(w[cols + c / 16]) * t->global);
                            }
                        }
                });
            }
            int64_t row = 0;
            const int first = f16 ? groups[j].first : j, stop = f16 ? groups[j].stop : j + 1;
            for (int projection = first; projection < stop; ++projection) {
                const std::string stem = layer + projections[projection];
                const int64_t nr = adapter.tensors().at(stem + ".lora_B").shape[0];
                if (!add_lora(adapter, stem, spec.scale, nr, cols,
                              spec.scale == 0 ? nullptr : merged.data() + row * cols, pool, cancel, err))
                    return false;
                row += nr;
            }
            if (spec.scale != 0 && f16) {
                // every row's largest |value| (a nonfinite one: NaN), on the pool; then the matrix's
                std::vector<float> row_big(static_cast<size_t>(rows));
                pool.parallel_for(rows, [&](int64_t a, int64_t b) {
                    for (int64_t r = a; r < b; ++r) {
                        float big = 0;
                        bool finite = true;
                        for (int64_t c = 0; c < cols; ++c) {
                            const float x = merged[r * cols + c];
                            finite = finite && std::isfinite(x);
                            big = std::max(big, std::fabs(x));
                        }
                        row_big[static_cast<size_t>(r)] = finite ? big : std::numeric_limits<float>::quiet_NaN();
                    }
                });
                float big = 0;
                for (float x : row_big) {
                    if (!std::isfinite(x)) {
                        err = "lora fp16: nonfinite merged weight: " + name;
                        return false;
                    }
                    big = std::max(big, x);
                }
                int exponent;
                std::frexp(big > 0 ? big : 1.f, &exponent);
                const int shift = std::clamp(-exponent, 0, 24);
                // All logical elements are replaced; physical padding remains byte-for-byte as copied from the base.
                // Every (row, column) has its own two bytes (hmx_at): the rows written on the pool, no two threads on
                // one byte. x 2^shift as the product: exact (the largest |value| lands in [0.5, 1)).
                const float up = std::ldexp(1.f, shift);
                std::atomic<bool> overflow{false};
                pool.parallel_for(rows, [&](int64_t a, int64_t b) {
                    bool over = false;
                    for (int64_t r = a; r < b; ++r)
                        for (int64_t c = 0; c < cols; ++c) {
                            const uint16_t h = half_rne(merged[r * cols + c] * up);
                            over = over || (h & 0x7c00) == 0x7c00;
                            const size_t at = 2 * hmx_at(r, c, cols);
                            raw[at] = static_cast<uint8_t>(h);
                            raw[at + 1] = static_cast<uint8_t>(h >> 8);
                        }
                    if (over)
                        overflow = true;
                });
                if (overflow) {
                    err = "lora fp16: merged weight overflows fp16: " + name;
                    return false;
                }
                auto &sh = tensors.object[name].object["shift"];
                sh.kind = json::Value::Kind::Number;
                sh.number = shift;
            } else if (spec.scale != 0) {
                std::vector<uint8_t> encoded;
                float global;
                if (!lora_q8_encode(merged.data(), rows, cols, encoded, global, err, &pool))
                    return false;
                raw = std::move(encoded);
                auto &g = tensors.object[name].object["global"];
                g.kind = json::Value::Kind::Number;
                g.number = global;
            }
            if (!write_at(f, t->offset, raw, err))
                return false;
        }
        if (progress)
            progress(offset + l + 1, total);
    }
    if (cancel && cancel()) {
        err = "cancelled";
        return false;
    }
    if (stamp_of(spec.path) != adapter_stamp || stamp_of(base + "/" + file) != source_stamp) {
        err = "lora: source changed during merge";
        return false;
    }
    if (!sync(f)) {
        err = "lora: syncing AR failed";
        return false;
    }
    const int closed = std::fclose(f);
    output.f = nullptr;
    if (closed) {
        err = "lora: closing AR failed";
        return false;
    }
    std::string digest;
    if (!lora_sha256(part, digest, err, cancel))
        return false;
    std::filesystem::remove(bin, ec);
    ec.clear();
    std::filesystem::rename(part, bin, ec);
    if (ec) {
        err = "lora: publishing AR: " + ec.message();
        return false;
    }
    sync_directory(dir);
    const std::string manifest_text =
        "{\"format\":\"geode-pack-extra\",\"version\":1,\"converter\":1,\"key\":" + quote(key) +
        ",\"stamp\":" + quote(stamp_of(bin)) + ",\"adapter_sha256\":" + quote(sha) +
        ",\"scale\":" + lora_scale_text(spec.scale) + ",\"tensor_sha256\":" + quote(hash_text(dump(tensors))) +
        ",\"files\":[{\"name\":" + quote(file) + ",\"size\":" + std::to_string(source.size()) +
        ",\"sha256\":" + quote(digest) + "}],\"tensors\":" + dump(tensors) + "}";
    FILE *mf = std::fopen((man + ".part").c_str(), "wb");
    if (!mf) {
        err = "lora: writing manifest failed";
        return false;
    }
    bool ok = std::fwrite(manifest_text.data(), 1, manifest_text.size(), mf) == manifest_text.size() && sync(mf);
    if (std::fclose(mf))
        ok = false;
    if (!ok) {
        err = "lora: syncing manifest failed";
        return false;
    }
    std::filesystem::rename(man + ".part", man, ec);
    if (ec) {
        err = "lora: publishing manifest: " + ec.message();
        return false;
    }
    sync_directory(dir);
    err.clear();
    return true;
}
} // namespace
bool prepare_lora(const std::string &base, const LoraSpec &spec, ThreadPool &pool, const LoraProgress &progress,
                  const LoraCancel &cancel, LoraWeights &out, std::string &err) {
    const auto fail = [&](const std::string &s) {
        err = "lora: " + s;
        return false;
    };
    out = {};
    if (spec.path.empty() || spec.cache.empty() || !std::isfinite(spec.scale))
        return fail("path, finite scale and cache are required");
    if (cancel && cancel()) {
        err = "cancelled";
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(spec.cache, ec);
    if (ec)
        return fail("cannot create cache: " + ec.message());
    Lock lock;
    if (!lock.acquire(spec.cache + "/.build.lock", cancel, err))
        return false;
    const std::string adapter_stamp = stamp_of(spec.path), base_stamp = stamp_of(base + "/ar_q8.bin"),
                      candidate_f16_stamp = spec.f16 ? stamp_of(base + "/ar_f16.bin") : "";
    SafeTensors adapter;
    Pack pack;
    int rank;
    if (!adapter.open(spec.path, err) || !pack.open(base, err) || !validate_lora(adapter, &pack, rank, err))
        return false;
    const auto *first = pack.find("a16.model.layers.0.ar_qkv");
    const bool want_f16 = spec.f16 && first && first->hmx16;
    const std::string f16_stamp = want_f16 ? candidate_f16_stamp : "";
    if (want_f16 && !validate_f16(adapter, pack, err))
        return false;
    if (stamp_of(spec.path) != adapter_stamp || stamp_of(base + "/ar_q8.bin") != base_stamp ||
        (want_f16 && stamp_of(base + "/ar_f16.bin") != f16_stamp))
        return fail("source changed while opening");
    const double q8_start = now_seconds();
    std::string ah, bh;
    if (!lora_sha256_remembered(spec.path, spec.cache, ah, err, cancel) ||
        !lora_sha256_remembered(base + "/ar_q8.bin", spec.cache, bh, err, cancel))
        return false;
    const std::string key = hash_text("geode-yue2-lora-q8-v1\n" + bh + "\n" + dump(pack.manifest()) + "\n" + ah + "\n" +
                                      lora_scale_text(spec.scale));
    const std::string dir = (std::filesystem::path(spec.cache) / key).string();
    out.directory = dir;
    out.sha256 = ah;
    out.key = key;
    out.scale = spec.scale;
    const int total = want_f16 ? 56 : 28;
    if (!build_part(pack, adapter, base, spec, dir, key, ah, base_stamp, adapter_stamp, false, total, pool, progress,
                    cancel, err))
        return false;
    out.t_q8 = now_seconds() - q8_start;
    if (want_f16) {
        const double start = now_seconds();
        std::string hash;
        if (!lora_sha256_remembered(base + "/ar_f16.bin", spec.cache, hash, err, cancel))
            return false;
        const std::string fkey =
            hash_text("geode-yue2-lora-f16-v1\n" + hash + "\n" + dump(tensor_table(pack, "ar_f16.bin")) + "\n" + ah +
                      "\n" + lora_scale_text(spec.scale));
        if (!build_part(pack, adapter, base, spec, dir, fkey, ah, f16_stamp, adapter_stamp, true, total, pool, progress,
                        cancel, err))
            return false;
        out.t_f16 = now_seconds() - start;
    }
    if (stamp_of(spec.path) != adapter_stamp || stamp_of(base + "/ar_q8.bin") != base_stamp ||
        (want_f16 && stamp_of(base + "/ar_f16.bin") != f16_stamp))
        return fail("source changed while preparing weights");
    if (cancel && cancel()) {
        err = "cancelled";
        return false;
    }
    out.f16 = want_f16;
    err.clear();
    return true;
}
bool lora_quant_self_test(std::string &err) {
    if (hash_text("") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ||
        hash_text("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        err = "SHA-256 known answer";
        return false;
    }
    if (even(0.5f) != 0 || even(1.5f) != 2 || even(-0.5f) != 0 || even(-1.5f) != -2) {
        err = "half-even";
        return false;
    }
    for (int i = 0; i < 126; ++i) {
        float mid = (e4m3_value(uint8_t(i)) + e4m3_value(uint8_t(i + 1))) * 0.5f;
        if (fp8(mid) != (i % 2 ? i + 1 : i)) {
            err = "E4M3 ties";
            return false;
        }
    }
    const std::pair<float, uint16_t> halves[] = {{0.f, 0},
                                                 {-0.f, 0x8000},
                                                 {1.f, 0x3c00},
                                                 {-1.f, 0xbc00},
                                                 {0x1p-24f, 1},
                                                 {0x1p-25f, 0},
                                                 {std::nextafter(0x1p-25f, 1.f), 1},
                                                 {3 * 0x1p-25f, 2},
                                                 {0x1p-14f, 0x0400},
                                                 {0x1.ffcp-15f, 0x0400},
                                                 {1.f + 0x1p-11f, 0x3c00},
                                                 {1.f + 3 * 0x1p-11f, 0x3c02},
                                                 {65504.f, 0x7bff},
                                                 {65520.f, 0x7c00},
                                                 {std::numeric_limits<float>::max(), 0x7c00},
                                                 {-65520.f, 0xfc00},
                                                 {std::numeric_limits<float>::infinity(), 0x7c00}};
    for (const auto &[f, h] : halves)
        if (half_rne(f) != h) {
            err = "fp16 RNE known answer";
            return false;
        }
    for (uint32_t h = 0; h < 65536; ++h)
        if ((h & 0x7c00) != 0x7c00 && half_rne(half_float(static_cast<uint16_t>(h))) != h) {
            err = "fp16 finite round trip";
            return false;
        }
    float x[64]{};
    std::vector<uint8_t> out;
    float g;
    if (lora_q8_encode(x, 1, 64, out, g, err)) {
        err = "zero matrix accepted";
        return false;
    }
    for (float &v : x)
        v = std::numeric_limits<float>::infinity();
    if (lora_q8_encode(x, 1, 64, out, g, err)) {
        err = "nonfinite accepted";
        return false;
    }
    err.clear();
    return true;
}
} // namespace geode::yue2
