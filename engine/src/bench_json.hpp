// bench_json.hpp - a minimal JSON writer for the benchmark results (objects, arrays, numbers, booleans, strings).
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace geode::bench {

class Json {
public:
    Json &begin_object(const char *key = nullptr) {
        prefix(key);
        s_ += '{';
        first_ = true;
        return *this;
    }
    Json &end_object() {
        s_ += '}';
        first_ = false;
        return *this;
    }
    Json &begin_array(const char *key = nullptr) {
        prefix(key);
        s_ += '[';
        first_ = true;
        return *this;
    }
    Json &end_array() {
        s_ += ']';
        first_ = false;
        return *this;
    }
    Json &num(const char *key, double v) {
        prefix(key);
        if (!std::isfinite(v)) {
            s_ += "null";
            return *this;
        }
        char b[40];
        std::snprintf(b, sizeof b, "%.6g", v);
        s_ += b;
        return *this;
    }
    Json &boolean(const char *key, bool v) {
        prefix(key);
        s_ += v ? "true" : "false";
        return *this;
    }
    Json &str(const char *key, const std::string &v) {
        prefix(key);
        s_ += '"';
        for (const char c : v) {
            if (c == '"' || c == '\\') {
                s_ += '\\';
                s_ += c;
            } else if (static_cast<unsigned char>(c) < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                s_ += b;
            } else {
                s_ += c;
            }
        }
        s_ += '"';
        return *this;
    }
    std::string take() { return std::move(s_); }

private:
    void prefix(const char *key) {
        if (!first_) s_ += ',';
        first_ = false;
        if (key) {
            s_ += '"';
            s_ += key;
            s_ += "\":";
        }
    }

    std::string s_;
    bool first_ = true;
};

}  // namespace geode::bench
