#include "yue2_state.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>

#include "json.hpp"

namespace geode::yue2 {

std::vector<float> noise_of(uint64_t seed, size_t n) {
    std::mt19937_64 rng(seed ^ 0x59754532ull);
    const auto u01 = [&]() { return (static_cast<double>(rng() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i += 2) {
        const double r = std::sqrt(-2.0 * std::log(u01())), a = 2.0 * 3.14159265358979323846 * u01();
        out[i] = static_cast<float>(r * std::cos(a));
        if (i + 1 < n) out[i + 1] = static_cast<float>(r * std::sin(a));
    }
    return out;
}

namespace {

// The file: "GYS1", the header's length (u32), the header (JSON: the scalars and each array's length), then the arrays
// in its order - prompt, negative, abc, codes (int32), latents, noise (float32) - little-endian, as every phone and PC
// we run on
constexpr char kMagic[4] = {'G', 'Y', 'S', '1'};

std::string quoted(const std::string &s) {
    std::string o = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (static_cast<unsigned char>(c) >= 0x20) o += c;
    }
    return o + "\"";
}

template <typename T>
bool put(std::FILE *f, const std::vector<T> &v) {
    return v.empty() || std::fwrite(v.data(), sizeof(T), v.size(), f) == v.size();
}

template <typename T>
bool get(std::FILE *f, std::vector<T> &v, int64_t n) {
    if (n < 0 || n > (int64_t{1} << 31)) return false;
    v.resize(static_cast<size_t>(n));
    return v.empty() || std::fread(v.data(), sizeof(T), v.size(), f) == v.size();
}

}  // namespace

bool write_state(const std::string &path, const SongState &s, std::string &err) {
    const size_t want = s.codes.size() * static_cast<size_t>(s.latent);
    if (s.latent <= 0 || s.latents.size() != want || s.noise.size() != want) {
        err = "state: the latents and the noise are not the codes x the latent";
        return false;
    }
    std::ostringstream h;
    h.precision(17);
    h << "{\"version\":" << SongState::kVersion << ",\"seed\":\"" << s.seed << "\",\"ode_steps\":" << s.ode_steps
      << ",\"latent\":" << s.latent << ",\"rate\":" << s.rate << ",\"ar_weights\":" << quoted(s.ar_weights)
      << ",\"nar_weights\":" << quoted(s.nar_weights) << ",\"nar_precision\":" << quoted(s.nar_precision)
      << ",\"mastered\":" << (s.mastered ? "true" : "false") << ",\"master_gain_db\":" << s.master_gain_db
      << ",\"abc_eos\":" << (s.abc_eos ? "true" : "false") << ",\"codes_eos\":" << (s.codes_eos ? "true" : "false")
      << ",\"cfg\":" << s.cfg << ",\"prompt\":" << s.prompt.size() << ",\"negative\":" << s.negative.size()
      << ",\"abc\":" << s.abc.size() << ",\"codes\":" << s.codes.size() << "}";
    std::string head = h.str();
    if (!s.lora_sha256.empty()) {
        if (s.lora_sha256.size() != 64 || s.lora_sha256.find_first_not_of("0123456789abcdef") != std::string::npos || !std::isfinite(s.lora_scale)) {
            err = "state: invalid adapter identity"; return false;
        }
        std::ostringstream l; l.precision(9);
        l << ",\"lora\":{\"sha256\":" << quoted(s.lora_sha256) << ",\"scale\":" << s.lora_scale << "}}";
        head.pop_back(); head += l.str();
    }
    const std::string part = path + ".part";
    std::FILE *f = std::fopen(part.c_str(), "wb");
    if (!f) {
        err = "state: could not write " + part;
        return false;
    }
    const uint32_t n = static_cast<uint32_t>(head.size());
    bool ok = std::fwrite(kMagic, 1, 4, f) == 4 && std::fwrite(&n, 4, 1, f) == 1 &&
              std::fwrite(head.data(), 1, head.size(), f) == head.size() && put(f, s.prompt) && put(f, s.negative) && put(f, s.abc) &&
              put(f, s.codes) && put(f, s.latents) && put(f, s.noise);
    if (std::fclose(f) != 0) ok = false;
#if defined(_WIN32)
    if (ok) std::remove(path.c_str());   // (Windows' rename does not replace)
#endif
    if (ok && std::rename(part.c_str(), path.c_str()) == 0) return true;
    std::remove(part.c_str());
    err = "state: could not write " + path;
    return false;
}

bool read_state(const std::string &path, SongState &s, std::string &err) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "state: no file " + path;
        return false;
    }
    const auto fail = [&](const std::string &why) {
        std::fclose(f);
        err = "state: " + path + ": " + why;
        return false;
    };
    char magic[4];
    uint32_t n = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, kMagic, 4) != 0) return fail("not a song's state");
    if (std::fread(&n, 4, 1, f) != 1 || n == 0 || n > (1u << 20)) return fail("a broken header");
    std::string head(n, '\0');
    if (std::fread(head.data(), 1, n, f) != n) return fail("cut short");
    json::Value j;
    std::string why;
    if (!json::parse(head, j, why)) return fail("a broken header: " + why);
    if (j["version"].i64(0) < 1 || j["version"].i64(0) > SongState::kVersion) return fail("a version this build does not read");
    SongState o;
    o.seed = std::strtoull(j["seed"].str("0").c_str(), nullptr, 10);
    o.ode_steps = static_cast<int>(j["ode_steps"].i64(0));
    o.latent = static_cast<int>(j["latent"].i64(0));
    o.rate = static_cast<int>(j["rate"].i64(48000));
    o.ar_weights = j["ar_weights"].str(), o.nar_weights = j["nar_weights"].str(), o.nar_precision = j["nar_precision"].str();
    if (j["version"].i64() >= 2 && j.object.count("lora")) {
        const auto &l = j["lora"];
        o.lora_sha256 = l["sha256"].str(); o.lora_scale = static_cast<float>(l["scale"].f64());
        if (!l.is_object() || o.lora_sha256.size() != 64 || o.lora_sha256.find_first_not_of("0123456789abcdef") != std::string::npos ||
            !l["scale"].is_number() || !std::isfinite(o.lora_scale)) return fail("invalid adapter identity");
    }
    o.mastered = j["mastered"].boolean;
    o.master_gain_db = j["master_gain_db"].f64(0);
    o.abc_eos = j["abc_eos"].boolean, o.codes_eos = j["codes_eos"].boolean;
    o.cfg = static_cast<float>(j["cfg"].f64(1.0));
    const int64_t codes = j["codes"].i64(-1), prompt = j["prompt"].i64(-1), negative = j["negative"].i64(0), abc = j["abc"].i64(-1);
    constexpr int64_t kMost = int64_t{1} << 24;
    if (o.latent <= 0 || o.latent > 4096 || codes < 0 || codes > kMost || prompt < 0 || prompt > kMost || negative < 0 ||
        negative > kMost || abc < 0 || abc > kMost)
        return fail("a broken header");
    // the arrays' bytes against the file's before anything is allocated (a broken header must not ask for gigabytes)
    const int64_t want = 4 * (prompt + negative + abc + codes + 2 * codes * o.latent);
    const long at = std::ftell(f);
    if (at < 0 || std::fseek(f, 0, SEEK_END) != 0) return fail("unreadable");
    const long end = std::ftell(f);
    if (end < 0 || std::fseek(f, at, SEEK_SET) != 0) return fail("unreadable");
    if (end - at < want) return fail("cut short");
    if (end - at > want) return fail("more than its header says");
    if (!get(f, o.prompt, prompt) || !get(f, o.negative, negative) || !get(f, o.abc, abc) || !get(f, o.codes, codes) ||
        !get(f, o.latents, codes * o.latent) || !get(f, o.noise, codes * o.latent))
        return fail("cut short");
    if (std::fgetc(f) != EOF) return fail("more than its header says");
    std::fclose(f);
    s = std::move(o);
    return true;
}

}  // namespace geode::yue2
