// sampler.cpp - top-k first (a partial sort of 248K logits is cheap next to a forward pass), then softmax with the
// temperature over the candidates, then the top-p cut.
#include "sampler.hpp"

#include <algorithm>
#include <cmath>

namespace geode {

namespace {
// with top-k off, the candidates are capped here: the probability left past the first 1024 tokens is negligible
// and sorting all 248K for every token is not
constexpr int kMaxCandidates = 1024;
}  // namespace

int Sampler::sample(const float *logits, int n, const SamplerParams &p) {
    if (p.temperature <= 0.0f) return static_cast<int>(std::max_element(logits, logits + n) - logits);
    const int k = std::min(n, p.top_k > 0 ? p.top_k : kMaxCandidates);
    cand_.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) cand_[static_cast<size_t>(i)] = {logits[i], i};
    std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    cand_.resize(static_cast<size_t>(k));
    const float mx = cand_[0].first;
    double sum = 0.0;
    for (auto &c : cand_) {
        c.first = std::exp((c.first - mx) / p.temperature);
        sum += c.first;
    }
    // top-p: the smallest prefix whose probability reaches p
    size_t keep = cand_.size();
    if (p.top_p < 1.0f) {
        double acc = 0.0;
        for (size_t i = 0; i < cand_.size(); ++i) {
            acc += cand_[i].first / sum;
            if (acc >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double kept = 0.0;
    for (size_t i = 0; i < keep; ++i) kept += cand_[i].first;
    double r = std::uniform_real_distribution<double>(0.0, kept)(rng_);
    for (size_t i = 0; i < keep; ++i) {
        r -= cand_[i].first;
        if (r <= 0.0) return cand_[i].second;
    }
    return cand_[keep - 1].second;
}

}  // namespace geode
