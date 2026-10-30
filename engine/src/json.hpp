// json.hpp - a small JSON reader for the model manifest (geode.json). Objects, arrays, strings, numbers, booleans,
// null; strings keep their UTF-8 bytes (\uXXXX escapes become UTF-8). Not a general-purpose library: no writer, and
// errors are reported as one message.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace geode::json {

struct Value {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    bool is_object() const { return kind == Kind::Object; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_string() const { return kind == Kind::String; }
    bool is_number() const { return kind == Kind::Number; }

    // A member of an object, or a Null value when absent (or when this is not an object).
    const Value &operator[](const std::string &key) const;
    const Value &at(size_t i) const;
    size_t size() const { return kind == Kind::Array ? array.size() : kind == Kind::Object ? object.size() : 0; }

    std::string str(const std::string &fallback = "") const { return is_string() ? string : fallback; }
    int64_t i64(int64_t fallback = 0) const { return is_number() ? static_cast<int64_t>(number) : fallback; }
    double f64(double fallback = 0.0) const { return is_number() ? number : fallback; }
};

// Parses text; on failure returns false and sets error (with the byte offset).
bool parse(std::string_view text, Value &out, std::string &error);

}  // namespace geode::json
