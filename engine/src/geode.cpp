// geode.cpp - the C API (geode.h) over geode::Model: the conversation, sampling, stop, statistics; and the memory
// test.
#include "geode/geode.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "bench.hpp"
#include "model.hpp"
#include "platform.hpp"
#include "sampler.hpp"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif

struct geode_model {
    geode::Model model;
    geode::Sampler sampler;
    std::atomic<bool> stop{false};
    geode_stats stats{};
    // the last token handed to the callback, part of the conversation but not computed yet (geode.h)
    int32_t pending = -1;
    std::vector<float> logits;
    std::vector<float> logits2;   // a verified pair's [2][vocab]
    std::string prompt_profile;   // the last call's prompt; the model's own profile then counts the answer only
};

namespace {

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
constexpr const char *kVersion = "geode 0.3.0 (NEON dotprod kernels)";
#else
constexpr const char *kVersion = "geode 0.3.0 (C++ kernels)";
#endif

int fail(char *err, size_t err_len, int code, const std::string &msg) {
    if (err && err_len) std::snprintf(err, err_len, "%s", msg.c_str());
    return code;
}

void refresh_stats(geode_model *m) {
    auto &s = m->stats;
    s.memory_used = m->model.memory_bytes();
    s.expert_cache_bytes = m->model.expert_cache_bytes();
    s.expert_hits = m->model.expert_hits() + m->model.expert_prefetch_hits();
    s.expert_misses = m->model.expert_misses();
    s.flash_bytes_read = m->model.flash_bytes();
    s.context_used = m->model.position() + (m->pending >= 0 ? 1 : 0);
    s.context_length = m->model.context_length();
    s.context_allocated = m->model.context_allocated();
    s.expert_ahead_hits = m->model.expert_prefetch_hits();
    s.expert_reads_ahead = m->model.expert_prefetches();
    s.experts_per_token = m->model.experts();
}

// The prompt into the conversation (after the token passed on and not computed yet): in chunks, each a layer at a
// time; when speculating (spec) the MTP layer's positions too, prev the hidden state before the next position (spec
// goes off when the MTP layer fails: the answer is the same without it). GEODE_OK, or a code (err) - a prompt cut
// short resets the conversation.
int read_prompt(geode_model &m, const int32_t *prompt, int32_t n_prompt, bool &spec, std::vector<float> &prev, char *err,
                size_t err_len) {
    std::vector<int32_t> feed;
    if (m.pending >= 0) feed.push_back(m.pending);
    feed.insert(feed.end(), prompt, prompt + n_prompt);
    if (feed.empty()) return fail(err, err_len, GEODE_E_ARG, "nothing to continue from");
    // the prompt's image pads are the images added (geode_add_image), every one of them
    if (m.model.has_vision()) {
        const int pad = m.model.vision().shape().pad_token;
        const auto pads = static_cast<int>(std::count(feed.begin(), feed.end(), pad));
        if (pads != m.model.pending_image_tokens()) {
            const int want = m.model.pending_image_tokens();
            m.model.drop_pending_images();
            return fail(err, err_len, GEODE_E_ARG, "the prompt holds " + std::to_string(pads) +
                        " image tokens, the images added take " + std::to_string(want));
        }
    }
    if (m.model.position() + static_cast<int>(feed.size()) >= m.model.context_length()) {
        m.model.drop_pending_images();
        return fail(err, err_len, GEODE_E_ARG, "the conversation does not fit the context");
    }

    const int H = m.model.config().hidden;
    // the MTP layer's positions follow the model's: each takes the hidden state there and the token after it; the
    // position before this prompt pairs with its first token
    if (spec && m.model.last_hidden()) prev.assign(m.model.last_hidden(), m.model.last_hidden() + H);

    // the prompt
    std::string e;
    double t0 = geode::now_seconds();
    // in chunks of up to Model::kMaxPrompt tokens: each goes through a layer at a time (in sub-batches of what the
    // cache can lend scratch for), so each routed expert is read from flash once a chunk; equal chunks
    // (GEODE_PROMPT_BATCHES=1: batch by batch as before, each through every layer - measuring)
    const size_t most = static_cast<size_t>(std::getenv("GEODE_PROMPT_BATCHES") ? m.model.prompt_batch()
                                                                                : geode::Model::kMaxPrompt);
    const size_t batch = (feed.size() + (feed.size() + most - 1) / most - 1) / ((feed.size() + most - 1) / most);
    for (size_t i = 0; i < feed.size(); i += batch) {
        if (m.stop.load()) {
            // half a prompt is no conversation: start over (the app sends it all again)
            m.model.reset();
            m.pending = -1;
            refresh_stats(&m);
            return fail(err, err_len, GEODE_E_STOPPED, "stopped while reading the prompt");
        }
        const int n = static_cast<int>(std::min(batch, feed.size() - i));
        const bool last = i + static_cast<size_t>(n) == feed.size();
        const int p0 = m.model.position();
        if (!m.model.forward_batch(feed.data() + i, n, last ? m.logits.data() : nullptr, false, e)) {
            m.model.reset();
            m.pending = -1;
            refresh_stats(&m);
            if (m.stop.load()) return fail(err, err_len, GEODE_E_STOPPED, "stopped while reading the prompt");
            return fail(err, err_len, GEODE_E_IO, e);
        }
        // GEODE_DUMP_LOGITS=FILE: the prompt's last logits as raw floats (measuring two arithmetics against each other)
        if (last && std::getenv("GEODE_DUMP_LOGITS"))
            if (FILE *f = std::fopen(std::getenv("GEODE_DUMP_LOGITS"), "wb")) {
                std::fwrite(m.logits.data(), sizeof(float), m.logits.size(), f);
                std::fclose(f);
            }
        if (spec) {
            std::vector<const float *> hs;
            std::vector<int> next;
            if (!prev.empty()) {
                hs.push_back(prev.data());
                next.push_back(feed[i]);
            }
            for (int r = 0; r + 1 < n; ++r) {
                hs.push_back(m.model.hidden(r));
                next.push_back(feed[i + static_cast<size_t>(r) + 1]);
            }
            // the MTP layer in batches it takes (a chunk may be longer)
            const int first = p0 - (prev.empty() ? 0 : 1);
            for (size_t o = 0; spec && o < hs.size(); o += static_cast<size_t>(geode::Model::kMaxBatch)) {
                const int k = static_cast<int>(std::min(hs.size() - o, static_cast<size_t>(geode::Model::kMaxBatch)));
                if (!m.model.forward_mtp(next.data() + o, hs.data() + o, k, first + static_cast<int>(o), nullptr, e))
                    spec = false;   // no drafts then (the answer is the same)
            }
            prev.assign(m.model.hidden(n - 1), m.model.hidden(n - 1) + H);
        }
    }
    m.pending = -1;
    m.model.trim_batch();   // the scratch goes back to the expert cache
    m.prompt_profile = m.model.profile();
    m.model.clear_profile();
    m.stats.prompt_tokens_per_s = static_cast<double>(feed.size()) / std::max(1e-9, geode::now_seconds() - t0);
    return GEODE_OK;
}

}  // namespace

extern "C" {

const char *geode_version(void) { return kVersion; }

int geode_open(const geode_config *config, geode_model **out, char *err, size_t err_len) {
    if (!config || !config->model_dir || !out) return fail(err, err_len, GEODE_E_ARG, "missing config or output");
    *out = nullptr;
    auto model = std::make_unique<geode_model>();
    geode::ModelOptions opt;
    opt.memory_budget = config->memory_budget;
    opt.compute_threads = config->compute_threads;
    opt.context_length = config->context_length;
    opt.helper_threads = config->helper_threads < 0 ? 0 : config->helper_threads > 0 ? config->helper_threads : -1;
    opt.io_threads = config->io_threads;
    opt.gpu = config->gpu != 0;
    opt.prefetch = config->no_prefetch == 0;
    opt.gpu_share = config->gpu_share;
    opt.experts_per_token = config->experts_per_token;
    opt.route_prior = config->route_prior;
    // 0: never (or GEODE_TAIL_SKIP, for measuring); 6 / 4: the last 2 / 4 of the 8 when they would be read
    opt.tail_skip = config->tail_skip > 0 ? config->tail_skip : -1;
    opt.npu = config->npu > 0 ? 1 : 0;
#if defined(__ANDROID__)
    opt.direct_io = true;   // expert blocks straight into the cache slots (O_DIRECT works on the phone's f2fs)
#endif
    std::string e;
    if (!model->model.open(config->model_dir, opt, e)) {
        const int code = e.find("memory") != std::string::npos ? GEODE_E_MEMORY
                       : e.find("cannot open") != std::string::npos || e.find("bytes, expected") != std::string::npos
                           ? GEODE_E_IO
                           : GEODE_E_FORMAT;
        return fail(err, err_len, code, e);
    }
    model->model.set_stop(&model->stop);   // a long prompt checks it between layers
    model->logits.resize(static_cast<size_t>(model->model.config().vocab));
    model->logits2.resize(2 * static_cast<size_t>(model->model.config().vocab));
    refresh_stats(model.get());
    *out = model.release();
    return GEODE_OK;
}

int geode_generate(geode_model *model, const int32_t *prompt, int32_t n_prompt, const geode_sampling *sampling,
                   geode_token_cb callback, void *user, char *err, size_t err_len) {
    if (!model || n_prompt < 0 || (n_prompt > 0 && !prompt) || !sampling || !callback)
        return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    auto &m = *model;
    m.stop.store(false);
    m.model.clear_profile();
    // Speculating (a pack with an MTP layer, generating on the GPU): the MTP layer drafts the token after the next
    // one, the GPU verifies the pair at once (Model::verify). The answer is the one token at a time answer: the next
    // token is sampled from the verified logits as ever, and the draft stands only when it is that token.
    // GEODE_NO_SPEC=1: one token at a time.
    bool spec = m.model.can_draft() && !std::getenv("GEODE_NO_SPEC");
    std::vector<float> prev;
    const int read = read_prompt(m, prompt, n_prompt, spec, prev, err, err_len);
    if (read != GEODE_OK) return read;
    double t1 = geode::now_seconds();
    const int V = static_cast<int>(m.logits.size());
    std::string e;
    m.sampler.seed(sampling->seed);
    const geode::SamplerParams sp{sampling->temperature, sampling->top_p, sampling->top_k};
    // presence penalty: the tokens this answer has produced so far, once each
    std::vector<int32_t> seen;
    std::vector<uint8_t> is_seen(sampling->presence_penalty != 0.0f ? m.logits.size() : 0, 0);
    const auto is_stop = [&](int32_t t) {
        for (int32_t i = 0; i < sampling->n_stop_tokens; ++i)
            if (sampling->stop_tokens[i] == t) return true;
        return false;
    };

    // the answer
    int rc = GEODE_OK;
    int produced = 0;
    // the next token from a row of logits (the presence penalty first); counted, and given to the caller
    const auto sample = [&](float *lg) {
        for (int32_t t : seen) lg[static_cast<size_t>(t)] -= sampling->presence_penalty;
        const int32_t tok = m.sampler.sample(lg, V, sp);
        if (!is_seen.empty() && tok >= 0 && static_cast<size_t>(tok) < is_seen.size() && !is_seen[static_cast<size_t>(tok)]) {
            is_seen[static_cast<size_t>(tok)] = 1;
            seen.push_back(tok);
        }
        return tok;
    };
    // after a token went out: whether to go on (rc set when stopped)
    const auto more = [&](bool wants_stop) {
        if (wants_stop || m.stop.load()) {
            rc = GEODE_E_STOPPED;
            return false;
        }
        return produced < sampling->max_tokens && m.model.position() + 1 < m.model.context_length();
    };
    // the MTP layer's next positions to fill (and draft from): the tokens after them, from the last GPU pass's hidden
    // states (rows 0..), or after the prompt from its last hidden state
    int mtp_tok[geode::Gpu8::kMaxTokens] = {}, mtp_n = 1, mtp_pos = m.model.position() - 1;
    const float *mtp_hidden = prev.empty() ? nullptr : prev.data();
    if (spec && !mtp_hidden) spec = false;
    // Batches longer than a pair (GEODE_SPEC_K > 2, generating on the GPU): the MTP layer drafts on from its own
    // output (draft_next), each further draft only while the last one's probability is GEODE_SPEC_P (0.8) or more -
    // the drafts that stand cost one pass of the model's weights between them. The answer is the same whatever the
    // length: every token is still sampled from its own verified logits, a draft stands only when it is that token.
    const int kmax = spec ? m.model.spec_tokens() : 2;
    const float pmin = std::getenv("GEODE_SPEC_P") ? static_cast<float>(std::atof(std::getenv("GEODE_SPEC_P"))) : 0.8f;
    if (m.logits2.size() < static_cast<size_t>(kmax) * V) m.logits2.resize(static_cast<size_t>(kmax) * V);
    int batch[geode::Gpu8::kMaxTokens] = {};
    // GEODE_SPEC_LOG=1: the batches' lengths and the drafts that stood, on stderr at the end (measuring)
    const bool spec_log = std::getenv("GEODE_SPEC_LOG") != nullptr;
    int steps_by_len[geode::Gpu8::kMaxTokens + 1] = {}, drafts_made = 0, drafts_stood = 0, chained = 0;
    // On the CPU a fallen draft costs its experts' reads (from flash, on a phone whose cache is small) for nothing:
    // when, after 16 pairs, fewer than GEODE_SPEC_MIN_PCT (30) percent of the drafts stood, the rest of this answer
    // goes one token at a time (the answer is the same either way)
    const bool cutoff = spec && m.model.speculates_on_cpu();
    const int min_pct = std::getenv("GEODE_SPEC_MIN_PCT") ? std::atoi(std::getenv("GEODE_SPEC_MIN_PCT")) : 30;
    int pairs = 0, stood = 0;
    int32_t tok = sample(m.logits.data());
    geode::perf_thread();
    double t_step = geode::now_seconds();
    for (;;) {
        // tok: sampled, not yet through the model (each step's time to the performance hints)
        const double t_now = geode::now_seconds();
        geode::perf_work(t_now - t_step);
        t_step = t_now;
        ++produced;
        const bool wants_stop = callback(user, tok) != 0;
        if (is_stop(tok)) break;
        m.pending = tok;
        if (!more(wants_stop)) break;
        if (spec && m.model.position() + 2 < m.model.context_length()) {
            mtp_tok[mtp_n - 1] = tok;
            int d = 0;
            float pd = 0.0f;
            if (m.model.draft(mtp_tok, mtp_n, mtp_pos, mtp_hidden, d, e, kmax > 2 ? &pd : nullptr)) {
                // the batch: tok and its drafts (chained while confident and the context has room)
                int n = 2;
                batch[0] = tok;
                batch[1] = d;
                int q = mtp_pos + mtp_n;   // the MTP position the next chained draft takes (the last draft's token)
                while (n < kmax && pd >= pmin && m.model.position() + n + 1 < m.model.context_length()) {
                    int d2 = 0;
                    if (!m.model.draft_next(batch[n - 1], q, d2, &pd, e)) break;   // (no more chaining: the batch as is)
                    ++chained;
                    if (pd < pmin) break;   // too unsure to be worth its experts
                    batch[n++] = d2;
                    ++q;
                }
                if (!m.model.verify_n(batch, n, m.logits2.data(), e)) {
                    rc = fail(err, err_len, GEODE_E_IO, e);
                    break;
                }
                m.pending = -1;
                const int P = m.model.position() - n;   // tok's position; draft i's is P + i
                ++pairs;
                ++steps_by_len[n];
                drafts_made += n - 1;
                // each token from its own verified logits: draft i + 1 stands while it is what row i samples
                int kept = 1;   // tok
                bool stop_now = false;
                int32_t next = -1;
                for (int i = 0; i < n; ++i) {
                    const int32_t y = sample(m.logits2.data() + static_cast<size_t>(i) * V);
                    if (i + 1 < n && y == batch[i + 1] && !is_stop(y)) {
                        // the draft stands: the model has it already
                        ++kept;
                        ++stood;
                        ++drafts_stood;
                        ++produced;
                        const bool ws = callback(user, y) != 0;
                        mtp_tok[i] = y;
                        if (!more(ws)) {
                            stop_now = true;
                            break;
                        }
                        continue;
                    }
                    next = y;   // the first that is no draft (or after the last draft): the next token
                    mtp_tok[i] = y;
                    break;
                }
                if (kept < n) {
                    // the drafts after the last that stood taken back: the model just after it
                    if (!m.model.verify_keep(kept, e)) {
                        rc = fail(err, err_len, GEODE_E_IO, e);
                        break;
                    }
                }
                if (stop_now) break;
                tok = next;
                mtp_n = kept;   // the MTP positions P .. P + kept - 1 (the tokens after them: the stood drafts and tok)
                mtp_pos = P;
                mtp_hidden = nullptr;   // the verified pass's rows
                if (cutoff && pairs >= 16 && stood * 100 < pairs * min_pct) spec = false;
                continue;
            }
            spec = false;   // drafting failed: one token at a time from here (the same answer)
        }
        if (!m.model.forward(tok, m.logits.data(), e)) {
            rc = fail(err, err_len, GEODE_E_IO, e);
            break;
        }
        m.pending = -1;
        tok = sample(m.logits.data());
    }
    const double t2 = geode::now_seconds();
    if (spec_log) {
        std::fprintf(stderr, "spec: %d steps (k max %d, p %.2f), drafts %d made / %d stood / %d chained passes; batches:",
                     pairs, kmax, pmin, drafts_made, drafts_stood, chained);
        for (int k = 2; k <= kmax; ++k) std::fprintf(stderr, " %d:%d", k, steps_by_len[k]);
        std::fprintf(stderr, "\n");
    }
    m.model.save_cache_profile();   // the next open starts with what this answer taught the cache (a few KB)
    m.stats.decode_tokens_per_s = produced > 1 ? static_cast<double>(produced - 1) / std::max(1e-9, t2 - t1) : 0.0;
    refresh_stats(&m);
    return rc;
}

int geode_image_size(geode_model *model, int32_t w, int32_t h, int32_t max_pixels, int32_t *out_w, int32_t *out_h,
                     int32_t *n_tokens) {
    if (!model || w <= 0 || h <= 0 || !out_w || !out_h) return GEODE_E_ARG;
    const auto &v = model->model.vision();
    if (!v.available()) return GEODE_E_NOT_IMPLEMENTED;
    int ow = 0, oh = 0;
    v.fit(w, h, max_pixels, ow, oh);
    *out_w = ow;
    *out_h = oh;
    if (n_tokens) *n_tokens = v.tokens(ow, oh);
    return GEODE_OK;
}

int geode_add_image(geode_model *model, const uint8_t *rgb, int32_t w, int32_t h, const char *cache_path,
                    int32_t *n_tokens, char *err, size_t err_len) {
    if (!model || !rgb || w <= 0 || h <= 0) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    if (!model->model.has_vision())
        return fail(err, err_len, GEODE_E_NOT_IMPLEMENTED, "no vision encoder: " + model->model.vision_status());
    std::string e;
    int n = 0;
    if (!model->model.add_image(rgb, w, h, cache_path ? cache_path : "", n, e))
        return fail(err, err_len, e.find("memory") != std::string::npos ? GEODE_E_MEMORY : GEODE_E_IO, e);
    if (n_tokens) *n_tokens = n;
    return GEODE_OK;
}

int geode_cache_image(geode_model *model, const uint8_t *rgb, int32_t w, int32_t h, const char *cache_path, char *err,
                      size_t err_len) {
    if (!model || !rgb || w <= 0 || h <= 0 || !cache_path) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    if (!model->model.has_vision())
        return fail(err, err_len, GEODE_E_NOT_IMPLEMENTED, "no vision encoder: " + model->model.vision_status());
    std::string e;
    if (!model->model.cache_image(rgb, w, h, cache_path, e))
        return fail(err, err_len, e.find("memory") != std::string::npos ? GEODE_E_MEMORY : GEODE_E_IO, e);
    return GEODE_OK;
}

void geode_stop(geode_model *model) {
    if (model) model->stop.store(true);
}

void geode_reset(geode_model *model) {
    if (!model) return;
    model->model.reset();
    model->pending = -1;
    refresh_stats(model);
}

int geode_prefill(geode_model *model, const int32_t *prompt, int32_t n_prompt, char *err, size_t err_len) {
    if (!model || n_prompt < 0 || (n_prompt > 0 && !prompt)) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    auto &m = *model;
    m.stop.store(false);
    m.model.clear_profile();
    bool spec = m.model.can_draft() && !std::getenv("GEODE_NO_SPEC");
    std::vector<float> prev;
    const int rc = read_prompt(m, prompt, n_prompt, spec, prev, err, err_len);
    refresh_stats(&m);
    return rc;
}

int geode_state_save(geode_model *model, const char *path, int32_t append, int32_t *out_position, char *err,
                     size_t err_len) {
    if (!model || !path || !*path) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    std::string e;
    if (!model->model.save_state(path, append != 0, model->pending, e))
        return fail(err, err_len, e.find("images") != std::string::npos ? GEODE_E_ARG : GEODE_E_IO, e);
    refresh_stats(model);
    if (out_position) *out_position = model->stats.context_used;
    return GEODE_OK;
}

int geode_state_load(geode_model *model, const char *path, int32_t *out_position, char *err, size_t err_len) {
    if (!model || !path || !*path) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    std::string e;
    int32_t pending = -1;
    const bool ok = model->model.load_state(path, pending, e);
    model->pending = ok ? pending : -1;
    refresh_stats(model);
    if (!ok) {
        const int code = e.find("another") != std::string::npos || e.find("damaged") != std::string::npos ||
                                 e.find("not a saved") != std::string::npos
                             ? GEODE_E_FORMAT
                         : e.find("longer than") != std::string::npos ? GEODE_E_ARG
                         : e.find("memory") != std::string::npos      ? GEODE_E_MEMORY
                                                                      : GEODE_E_IO;
        return fail(err, err_len, code, e);
    }
    if (out_position) *out_position = model->stats.context_used;
    return GEODE_OK;
}

int geode_reserve_memory(geode_model *model, uint64_t bytes, uint64_t *out_bytes, char *err, size_t err_len) {
    if (!model) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    std::string e;
    const uint64_t held = model->model.reserve_memory(bytes, e);
    refresh_stats(model);
    if (out_bytes) *out_bytes = held;
    if (bytes == 0 && held > 0) return fail(err, err_len, GEODE_E_MEMORY, e);
    if (!e.empty()) fail(err, err_len, GEODE_OK, e);   // gave less than asked: said, not failed
    return GEODE_OK;
}

int geode_set_experts(geode_model *model, int32_t k, char *err, size_t err_len) {
    if (!model) return fail(err, err_len, GEODE_E_ARG, "bad arguments");
    std::string e;
    if (!model->model.set_experts(k, e)) return fail(err, err_len, GEODE_E_ARG, e);
    refresh_stats(model);
    return GEODE_OK;
}

void geode_get_stats(geode_model *model, geode_stats *out) {
    if (!model || !out) return;
    refresh_stats(model);
    *out = model->stats;
}

void geode_close(geode_model *model) { delete model; }

void geode_profile(geode_model *model, char *buf, size_t len) {
    if (!model || !buf || !len) return;
    // how it runs, for the app to say in words: the threads, and speculative decoding (off, MTP drafts verified on
    // the GPU up to K tokens at once, or pairs on the CPU)
    const auto &m = model->model;
    char spec[16];
    if (!m.can_draft()) std::snprintf(spec, sizeof spec, "off");
    else if (m.speculates_on_cpu()) std::snprintf(spec, sizeof spec, "cpu:2");
    else std::snprintf(spec, sizeof spec, "gpu:%d", m.spec_tokens());
    std::snprintf(buf, len, "prompt: %s\nanswer: %s\nGPU: %s\nNPU: %s\nrun: threads=%d spec=%s\n%s",
                  model->prompt_profile.c_str(), m.profile().c_str(), m.gpu_status().c_str(), m.npu_status().c_str(),
                  m.compute_threads(), spec,
                  m.has_vision() ? m.vision().profile().c_str() : ("vision: " + m.vision_status()).c_str());
}

int geode_bench(const char *what, const char *arg, char *out, size_t out_len) {
    if (!what || !out || !out_len) return GEODE_E_ARG;
    const std::string w = what;
    std::string r;
    if (w == "system") r = geode::bench::system_info();
    else if (w == "memory") r = geode::bench::memory();
    else if (w == "flash") r = geode::bench::flash(arg ? arg : ".");
    else if (w == "cpu") r = geode::bench::cpu();
    else if (w == "gpu") r = geode::bench::gpu();
    else if (w == "opencl") r = geode::bench::opencl();
    else if (w == "gpucheck") r = geode::bench::gpu_check(arg ? arg : "");
    else if (w == "npucheck") r = geode::bench::npu_check(arg ? arg : "");
    else return GEODE_E_ARG;
    if (r.size() + 1 > out_len) return GEODE_E_MEMORY;
    std::memcpy(out, r.c_str(), r.size() + 1);
    return GEODE_OK;
}

#if defined(__linux__)

namespace {

long proc_value_kib(const char *path, const char *key) {
    FILE *f = std::fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    const size_t n = std::strlen(key);
    while (std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, key, n) && line[n] == ':') {
            v = std::atol(line + n + 1);
            break;
        }
    std::fclose(f);
    return v;
}

}  // namespace

int geode_memory_probe(const char *status_path, uint64_t max_bytes, uint64_t step_bytes, int32_t pause_ms) {
    if (!status_path || step_bytes == 0) return GEODE_E_ARG;
    const int fd = ::open(status_path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) return GEODE_E_IO;
    std::vector<uint64_t *> blocks;
    uint64_t held = 0, s = 0x2545F4914F6CDD1Dull;
    volatile uint64_t sink = 0;
    while (held + step_bytes <= max_bytes) {
        timespec t0{}, t1{};
        clock_gettime(CLOCK_MONOTONIC, &t0);
        auto *p = static_cast<uint64_t *>(std::malloc(step_bytes));
        if (!p) break;
        for (size_t i = 0; i < step_bytes / 8; ++i) {   // random bits: zram cannot shrink them, like real weights
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p[i] = s;
        }
        blocks.push_back(p);
        sink = sink ^ p[blocks.size() % (step_bytes / 8)];
        held += step_bytes;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const long step_ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
        char line[256];
        const int len = std::snprintf(
            line, sizeof line, "held=%llu available=%lld swap_free=%lld vm_swap=%lld step_ms=%ld\n",
            static_cast<unsigned long long>(held),
            static_cast<long long>(proc_value_kib("/proc/meminfo", "MemAvailable")) * 1024,
            static_cast<long long>(proc_value_kib("/proc/meminfo", "SwapFree")) * 1024,
            static_cast<long long>(proc_value_kib("/proc/self/status", "VmSwap")) * 1024, step_ms);
        // one line per step: after the process is killed the app reads the whole history (the page cache keeps
        // what write() put there), so it can find the last step before the memory started going to zram
        if (len > 0 && ::write(fd, line, static_cast<size_t>(len)) != len) break;
        if (pause_ms > 0) usleep(static_cast<useconds_t>(pause_ms) * 1000);
    }
    ::close(fd);
    for (auto *p : blocks) std::free(p);
    return GEODE_OK;
}

#else

int geode_memory_probe(const char *, uint64_t, uint64_t, int32_t) { return GEODE_E_NOT_IMPLEMENTED; }

#endif

}  // extern "C"
