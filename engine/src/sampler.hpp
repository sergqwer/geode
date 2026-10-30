// sampler.hpp - the next token from the logits: greedy, or temperature + top-k + top-p.
#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace geode {

struct SamplerParams {
    float temperature = 0.0f;   // 0 = greedy
    float top_p = 1.0f;         // 1 = off
    int top_k = 0;              // 0 = off
};

class Sampler {
public:
    void seed(uint64_t s) { rng_.seed(s); }
    int sample(const float *logits, int n, const SamplerParams &p);

private:
    std::mt19937_64 rng_{0};
    std::vector<std::pair<float, int>> cand_;
};

}  // namespace geode
