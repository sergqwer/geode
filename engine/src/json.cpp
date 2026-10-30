// json.cpp - see json.hpp.
#include "json.hpp"

#include <cstdlib>

namespace geode::json {

namespace {

const Value kNull;

struct Parser {
    std::string_view s;
    size_t i = 0;
    std::string error;

    bool fail(const char *what) {
        if (error.empty()) error = std::string(what) + " at byte " + std::to_string(i);
        return false;
    }
    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool literal(std::string_view word) {
        if (s.substr(i, word.size()) != word) return fail("unexpected token");
        i += word.size();
        return true;
    }
    static void put_utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    bool hex4(uint32_t &cp) {
        if (i + 4 > s.size()) return fail("short \\u escape");
        cp = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++];
            cp <<= 4;
            if (c >= '0' && c <= '9') cp |= c - '0';
            else if (c >= 'a' && c <= 'f') cp |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') cp |= c - 'A' + 10;
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool string(std::string &out) {
        ++i;  // opening quote
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return true;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) break;
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t lo;
                        if (!hex4(lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }
    bool value(Value &v, int depth) {
        if (depth > 64) return fail("nesting too deep");
        skip_ws();
        if (i >= s.size()) return fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            v.kind = Value::Kind::Object;
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                skip_ws();
                if (i >= s.size() || s[i] != '"') return fail("expected a key");
                std::string key;
                if (!string(key)) return false;
                skip_ws();
                if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                ++i;
                if (!value(v.object[key], depth + 1)) return false;
                skip_ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.kind = Value::Kind::Array;
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                v.array.emplace_back();
                if (!value(v.array.back(), depth + 1)) return false;
                skip_ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.kind = Value::Kind::String;
            return string(v.string);
        }
        if (c == 't') { v.kind = Value::Kind::Bool; v.boolean = true; return literal("true"); }
        if (c == 'f') { v.kind = Value::Kind::Bool; v.boolean = false; return literal("false"); }
        if (c == 'n') { v.kind = Value::Kind::Null; return literal("null"); }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t start = i;
            while (i < s.size() && (s[i] == '-' || s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E' ||
                                    (s[i] >= '0' && s[i] <= '9')))
                ++i;
            std::string num(s.substr(start, i - start));
            char *end = nullptr;
            v.kind = Value::Kind::Number;
            v.number = std::strtod(num.c_str(), &end);
            if (end != num.c_str() + num.size()) return fail("bad number");
            return true;
        }
        return fail("unexpected character");
    }
};

}  // namespace

const Value &Value::operator[](const std::string &key) const {
    if (kind != Kind::Object) return kNull;
    auto it = object.find(key);
    return it == object.end() ? kNull : it->second;
}

const Value &Value::at(size_t idx) const {
    return kind == Kind::Array && idx < array.size() ? array[idx] : kNull;
}

bool parse(std::string_view text, Value &out, std::string &error) {
    Parser p;
    p.s = text;
    out = Value{};
    if (!p.value(out, 0)) {
        error = p.error;
        return false;
    }
    p.skip_ws();
    if (p.i != text.size()) {
        error = "trailing characters at byte " + std::to_string(p.i);
        return false;
    }
    return true;
}

}  // namespace geode::json
