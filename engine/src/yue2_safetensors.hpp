// Bounded, read-only safetensors reader for AR adapters. No framework dependency.
#pragma once
#include "platform.hpp"
#include <map>
#include <string>
#include <vector>
namespace geode::yue2 {
struct SafeTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0, bytes = 0;
};
class SafeTensors {
  public:
    bool open(const std::string &path, std::string &err);
    bool floats(const std::string &name, std::vector<float> &out, std::string &err) const;
    const std::map<std::string, SafeTensor> &tensors() const { return tensors_; }
    const RandomFile &file() const { return file_; }

  private:
    RandomFile file_;
    std::map<std::string, SafeTensor> tensors_;
};
} // namespace geode::yue2
