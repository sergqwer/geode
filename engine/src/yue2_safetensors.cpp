#include "yue2_safetensors.hpp"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
namespace geode::yue2 {
namespace {
bool integer(const json::Value &v, uint64_t &out) {
    if (!v.is_number() || !std::isfinite(v.number) || v.number < 0 || v.number > 9007199254740991.0 ||
        std::floor(v.number) != v.number)
        return false;
    out = static_cast<uint64_t>(v.number);
    return true;
}
// json::parse accepts duplicate object members. Safetensors must not: reject them before using its tensor table.
// Called only after JSON syntax validation, so quoted strings and object nesting are already well formed.
bool unique_keys(const std::string &text) {
    std::vector<std::set<std::string>> objects;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '{')
            objects.emplace_back();
        else if (text[i] == '}')
            objects.pop_back();
        else if (text[i] == '"') {
            const size_t begin = i++;
            while (i < text.size() && text[i] != '"') {
                if (text[i] == '\\')
                    ++i;
                ++i;
            }
            size_t next = i + 1;
            while (next < text.size() &&
                   (text[next] == ' ' || text[next] == '\t' || text[next] == '\r' || text[next] == '\n'))
                ++next;
            if (next < text.size() && text[next] == ':') {
                json::Value key;
                std::string error;
                if (objects.empty() || !json::parse(std::string_view(text).substr(begin, i - begin + 1), key, error) ||
                    !objects.back().insert(key.str()).second)
                    return false;
            }
        }
    }
    return true;
}
float half(uint16_t h) {
    const float s = (h & 0x8000) ? -1.f : 1.f;
    const int e = (h >> 10) & 31, m = h & 1023;
    if (!e)
        return s * std::ldexp(static_cast<float>(m), -24);
    if (e == 31)
        return m ? std::numeric_limits<float>::quiet_NaN() : s * std::numeric_limits<float>::infinity();
    return s * std::ldexp(1.f + static_cast<float>(m) / 1024.f, e - 15);
}
} // namespace
bool SafeTensors::open(const std::string &path, std::string &err) {
    const auto fail = [&](const std::string &s) {
        err = "safetensors: " + s;
        return false;
    };
    if (file_.is_open())
        return fail("already open");
    if (!file_.open(path, err))
        return false;
    uint8_t prefix[8];
    if (file_.size() < 8 || !file_.read_at(0, prefix, 8))
        return fail("truncated length");
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i)
        n |= uint64_t(prefix[i]) << (8 * i);
    if (n < 2 || n > (16u << 20) || n > file_.size() - 8)
        return fail("invalid header length (limit 16 MiB)");
    std::string header(static_cast<size_t>(n), '\0');
    if (!file_.read_at(8, header.data(), header.size()))
        return fail("truncated header");
    json::Value j;
    if (!json::parse(header, j, err) || !j.is_object())
        return fail("invalid header JSON: " + err);
    if (!unique_keys(header))
        return fail("duplicate header member");
    if (j.size() > 4097)
        return fail("too many tensors");
    std::map<std::string, SafeTensor> parsed;
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    const uint64_t start = 8 + n, available = file_.size() - start;
    for (const auto &[name, v] : j.object) {
        if (name == "__metadata__") {
            if (!v.is_object())
                return fail("invalid metadata");
            continue;
        }
        SafeTensor t;
        t.dtype = v["dtype"].str();
        const uint64_t width = t.dtype == "F32" ? 4 : (t.dtype == "BF16" || t.dtype == "F16") ? 2 : 0;
        if (!width || !v["shape"].is_array() || v["shape"].size() == 0 || v["shape"].size() > 4)
            return fail(name + ": unsupported dtype or shape");
        uint64_t count = 1;
        for (const auto &d : v["shape"].array) {
            uint64_t dim;
            if (!integer(d, dim) || !dim || dim > (1u << 24) || count > (uint64_t{1} << 28) / dim)
                return fail(name + ": shape exceeds bounds");
            count *= dim;
            t.shape.push_back(static_cast<int64_t>(dim));
        }
        uint64_t lo, hi;
        if (!v["data_offsets"].is_array() || v["data_offsets"].size() != 2 || !integer(v["data_offsets"].at(0), lo) ||
            !integer(v["data_offsets"].at(1), hi) || hi < lo || hi > available || hi - lo != count * width)
            return fail(name + ": invalid data offsets/size");
        t.offset = start + lo;
        t.bytes = hi - lo;
        spans.emplace_back(lo, hi);
        parsed.emplace(name, std::move(t));
    }
    std::sort(spans.begin(), spans.end());
    uint64_t end = 0;
    for (const auto &s : spans) {
        if (s.first != end)
            return fail("overlapping tensors or unindexed data");
        end = s.second;
    }
    if (end != available)
        return fail("trailing unindexed data");
    tensors_ = std::move(parsed);
    return true;
}
bool SafeTensors::floats(const std::string &name, std::vector<float> &out, std::string &err) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        err = "safetensors: missing " + name;
        return false;
    }
    const auto &t = it->second;
    std::vector<uint8_t> raw(static_cast<size_t>(t.bytes));
    if (!file_.read_at(t.offset, raw.data(), raw.size())) {
        err = "safetensors: short tensor " + name;
        return false;
    }
    const size_t width = t.dtype == "F32" ? 4 : 2;
    out.resize(raw.size() / width);
    for (size_t i = 0; i < out.size(); ++i) {
        uint32_t u = uint32_t(raw[i * width]) | (uint32_t(raw[i * width + 1]) << 8);
        float f;
        if (width == 4) {
            u |= uint32_t(raw[i * width + 2]) << 16 | uint32_t(raw[i * width + 3]) << 24;
            std::memcpy(&f, &u, 4);
        } else if (t.dtype == "BF16") {
            u <<= 16;
            std::memcpy(&f, &u, 4);
        } else
            f = half(static_cast<uint16_t>(u));
        if (!std::isfinite(f)) {
            err = "safetensors: nonfinite value in " + name;
            return false;
        }
        out[i] = f;
    }
    return true;
}
} // namespace geode::yue2
