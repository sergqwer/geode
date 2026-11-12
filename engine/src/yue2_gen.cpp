// yue2_gen.cpp - see yue2_gen.hpp.
#include "yue2_gen.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace geode::yue2 {
namespace {
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int Allowed::index(int t) const {
    if (t >= lo && t < hi) return t - lo;
    return t == extra && extra >= 0 ? hi - lo : -1;
}

Allowed allowed_rows(const Config &c, bool abc) {
    Allowed a;
    if (abc) {
        a.lo = 0;
        a.hi = c.tok.eod;
        a.extra = c.tok.abc_end;
        a.end = c.tok.abc_end;
    } else {   // MUSIC_END sits right before the codes (protocol.py): one range
        a.lo = c.tok.music_end;
        a.hi = c.tok.codec_offset + c.tok.codec_size;
        a.end = c.tok.music_end;
        if (c.tok.music_end + 1 != c.tok.codec_offset) {
            a.lo = c.tok.codec_offset;
            a.extra = c.tok.music_end;
        }
    }
    return a;
}

void distribution_rows(const float *vals, const Allowed &a, const Sampling &s, int step, const std::vector<int> &history,
                       Candidates &out) {
    // the oracle's steps (yue2.cpp distribution, sampling.py): mask, min_tokens, the window penalty, temperature,
    // top-k with its ties, top-p over the descending order with the first kept, the softmax in double
    const float ninf = -std::numeric_limits<float>::infinity();
    const int n = a.size();
    thread_local std::vector<float> sc;
    sc.assign(vals, vals + n);
    const int end = a.index(a.end);
    if (step < s.min_tokens) sc[static_cast<size_t>(end)] = ninf;
    if (s.penalty_window > 0 && !history.empty()) {
        const size_t from = history.size() > static_cast<size_t>(s.penalty_window) ? history.size() - s.penalty_window : 0;
        thread_local std::vector<int> ids;
        ids.clear();
        for (size_t i = from; i < history.size(); ++i) ids.push_back(history[i]);
        std::sort(ids.begin(), ids.end());
        for (size_t i = 0; i < ids.size();) {
            size_t j = i;
            while (j < ids.size() && ids[j] == ids[i]) ++j;
            const int k = a.index(ids[i]);
            if (k >= 0) {
                const float alpha = std::pow(s.repetition_penalty, static_cast<float>(j - i));
                float &v = sc[static_cast<size_t>(k)];
                v = v < 0 ? v * alpha : v / alpha;
            }
            i = j;
        }
    }
    out.id.clear();
    out.p.clear();
    if (s.temperature == 0.0f) {   // greedy: the argmax (the first of equal ones, as std::max_element)
        const int k = static_cast<int>(std::max_element(sc.begin(), sc.end()) - sc.begin());
        out.id.push_back(a.id(k));
        out.p.push_back(1.0f);
        return;
    }
    if (s.temperature != 1.0f)
        for (auto &v : sc) v = v / s.temperature;
    // top-k's threshold: the k-th largest over the whole vocabulary, whose other rows are -inf
    thread_local std::vector<float> sorted;
    sorted = sc;
    float threshold = ninf;
    if (s.top_k <= n) {
        std::nth_element(sorted.begin(), sorted.begin() + (s.top_k - 1), sorted.end(), std::greater<float>());
        threshold = sorted[static_cast<size_t>(s.top_k - 1)];
    }
    thread_local std::vector<int> idx;   // the survivors (>= threshold, finite), by descending score, ties by row
    idx.clear();
    for (int i = 0; i < n; ++i)
        if (sc[static_cast<size_t>(i)] >= threshold && sc[static_cast<size_t>(i)] != ninf) idx.push_back(i);
    if (idx.empty()) return;   // nothing allowed (cannot happen: the end token is masked only before min_tokens)
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int x, int y) { return sc[static_cast<size_t>(x)] > sc[static_cast<size_t>(y)]; });
    const float mx = sc[static_cast<size_t>(idx[0])];
    size_t keep = idx.size();
    if (s.top_p < 1.0f) {
        double z = 0.0;
        for (int i : idx) z += std::exp(static_cast<double>(sc[static_cast<size_t>(i)] - mx));
        double cum = 0.0;
        for (size_t r = 0; r < idx.size(); ++r) {
            const double p = std::exp(static_cast<double>(sc[static_cast<size_t>(idx[r])] - mx)) / z;
            if (r > 0 && cum > s.top_p) {
                keep = r;
                break;
            }
            cum += p;
        }
    }
    double z = 0.0;
    for (size_t r = 0; r < keep; ++r) z += std::exp(static_cast<double>(sc[static_cast<size_t>(idx[r])] - mx));
    for (size_t r = 0; r < keep; ++r) {
        out.id.push_back(a.id(idx[r]));
        out.p.push_back(static_cast<float>(std::exp(static_cast<double>(sc[static_cast<size_t>(idx[r])] - mx)) / z));
    }
}

int draw(const Candidates &c, std::mt19937_64 &rng) {
    const double u = static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
    double cum = 0.0;
    for (size_t i = 0; i < c.p.size(); ++i) {
        cum += c.p[i];
        if (u < cum) return c.id[i];
    }
    return c.id.back();   // u past the rounded total
}

namespace {
// the tokens of `want` not yet in the stream (its fed tokens must begin it)
bool missing(const Stream &s, const std::vector<int32_t> &want, std::vector<int32_t> &rest, std::string &err) {
    if (s.fed.size() > want.size() || !std::equal(s.fed.begin(), s.fed.end(), want.begin())) {
        err = "generate: the stream's cache is not a prefix of the phase's prefix";
        return false;
    }
    rest.assign(want.begin() + static_cast<std::ptrdiff_t>(s.fed.size()), want.end());
    return true;
}

// ---- prompt lookup: of the earlier places the history's last two tokens stand (the latest 16), the one whose
// tokens before them agree with the history's end the longest (ties: the latest), and what followed there - continued
// by the period when it runs into the end. The agreement's length picks the acceptance estimate (a repeated bar
// predicts its continuation better than a repeated pair).
class Lookup {
public:
    void add(const std::vector<int> &h) {   // h has grown by one token: the pair before it has a follower now
        const size_t L = h.size();
        if (L >= 3) at_[key(h[L - 3], h[L - 2])].push_back(static_cast<int>(L - 3));
    }
    void add_all(const std::vector<int> &h) {   // every pair of h with a follower (h given at once)
        for (size_t L = 3; L <= h.size(); ++L) at_[key(h[L - 3], h[L - 2])].push_back(static_cast<int>(L - 3));
    }
    // up to k drafts into d and the length of the agreement before them into m; 0 drafts: no earlier place
    int drafts(const std::vector<int> &h, int k, std::vector<int32_t> &d, int &m) const {
        d.clear();
        m = 0;
        const int L = static_cast<int>(h.size());
        if (L < 3) return 0;
        const auto it = at_.find(key(h[static_cast<size_t>(L - 2)], h[static_cast<size_t>(L - 1)]));
        if (it == at_.end()) return 0;
        const std::vector<int> &v = it->second;
        int best = -1;
        for (int i = static_cast<int>(v.size()) - 1, seen = 0; i >= 0 && seen < 16; --i, ++seen) {
            const int s = v[static_cast<size_t>(i)];   // h[s], h[s + 1] = the last two (s <= L - 3)
            int len = 2;
            while (len < 32 && s + 1 - len >= 0 && h[static_cast<size_t>(s + 1 - len)] == h[static_cast<size_t>(L - 1 - len)]) ++len;
            if (len > m) m = len, best = s;
        }
        const int period = L - 2 - best;   // >= 1
        for (int i = 0; i < k; ++i) {
            int x = best + 2 + i;
            while (x >= L) x -= period;
            d.push_back(h[static_cast<size_t>(x)]);
        }
        return k;
    }
    // the acceptance estimate's bucket of an agreement of m tokens
    static int bucket(int m) { return m <= 2 ? 0 : m == 3 ? 1 : m <= 5 ? 2 : m <= 9 ? 3 : m <= 15 ? 4 : 5; }
    static constexpr int kBuckets = 6;

private:
    static uint64_t key(int a, int b) { return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b); }
    std::unordered_map<uint64_t, std::vector<int>> at_;
};

// What a call costs on this phone, learnt as the phase goes: R rows through the path (every stream's segments
// together) t = a + b (R - 1), a weighted least-squares fit (each call 0.98 of the next; two pseudo-calls on the
// prior's line fade the same way - the 8 Elite's, geode-yue2gen bench: SMMLA 0.29 a row, SDOT 0.35). A phone bound by
// its memory (the Redmi's LPDDR4X) verifies eight drafts for little more than one token, and learns so.
class Cost {
public:
    explicit Cost(bool mmla) : s0_(mmla ? 0.29 : 0.35) {}
    double rel(int rows) const { return 1.0 + slope() * (rows - 1); }   // a call of `rows` rows against one row
    double slope() const { return fit_ ? s_ : s0_; }
    bool stale() const { return since1_ >= 64; }   // no one-row call lately (a lone stream's intercept)
    void seen(int rows, double sec) {
        const double x = rows - 1;
        if (w_ == 0.0) {
            const double u = sec / (1.0 + s0_ * x);
            add(0.0, u, 2.0);
            add(8.0, u * (1.0 + 8.0 * s0_), 2.0);
        }
        w_ *= 0.98, sx_ *= 0.98, st_ *= 0.98, sxx_ *= 0.98, sxt_ *= 0.98;
        add(x, sec, 1.0);
        since1_ = rows == 1 ? 0 : since1_ + 1;
        const double det = w_ * sxx_ - sx_ * sx_;
        if (det > 1e-9 * w_ * w_) {
            const double bb = (w_ * sxt_ - sx_ * st_) / det, aa = (st_ - bb * sx_) / w_;
            if (aa > 0.0) s_ = std::min(4.0, std::max(0.0, bb / aa)), fit_ = true;
        }
    }

private:
    void add(double x, double t, double w) { w_ += w, sx_ += w * x, st_ += w * t, sxx_ += w * x * x, sxt_ += w * x * t; }
    double s0_, s_ = 0.0, w_ = 0.0, sx_ = 0.0, st_ = 0.0, sxx_ = 0.0, sxt_ = 0.0;
    bool fit_ = false;
    int since1_ = 0;
};

constexpr int kMaxDrafts = 8;

// one stream of generate_many: its state between the calls
struct Lane {
    Stream *pos = nullptr, *neg = nullptr;
    const GenRequest *req = nullptr;
    GenResult *out = nullptr;
    bool cfg = false, lookup = false, done = false;
    std::mt19937_64 rng;
    std::vector<int> history;
    Lookup look;
    double kept[Lookup::kBuckets], refused[Lookup::kBuckets];   // the drafts' fate lately, an agreement length's
    int step = 0, token = 0;                                     // the token drawn at step, not fed yet
    int k = 0, bk = 0, len0 = 0, row = 0;                        // this call: drafts, bucket, cache length, first row
    std::vector<int32_t> d, feed, rp, rn;
    std::vector<int> rows;
};
}  // namespace

bool generate_many(FastAr &ar, const std::vector<Stream *> &pos, const std::vector<Stream *> &neg,
                   const std::vector<GenRequest> &reqs, std::vector<GenResult> &outs, ThreadPool &pool,
                   std::string &err) {
    const Config &c = ar.config();
    const size_t S = reqs.size();
    if (S == 0 || pos.size() != S || (!neg.empty() && neg.size() != S)) {
        err = "generate: the streams and the requests do not pair";
        return false;
    }
    const bool abc = reqs[0].abc;
    std::vector<Lane> L(S);
    outs.assign(S, GenResult{});
    for (size_t i = 0; i < S; ++i) {
        const GenRequest &q = reqs[i];
        Lane &l = L[i];
        l.req = &q, l.out = &outs[i], l.pos = pos[i], l.neg = neg.empty() ? nullptr : neg[i];
        l.history.assign(q.history.begin(), q.history.end());   // (an edit's: the song's tokens before it)
        l.cfg = !q.abc && q.cfg != 1.0f;
        if (!l.cfg) l.neg = nullptr;   // (a negative stream only where it is used)
        if (q.abc != abc) {
            err = "generate: the streams of one call in one phase";
            return false;
        }
        if (!l.pos || (l.cfg && !l.neg)) {
            err = l.pos ? "generate: CFG needs the negative stream" : "generate: no stream";
            return false;
        }
        if (q.s.max_tokens < 1 || q.prefix.empty() || (l.cfg && q.negative.empty())) {
            err = "generate: an empty prefix or no token budget";
            return false;
        }
        // the reference refuses a prefix + budget past the context; here the streams' capacities too
        const auto fits = [&](const Stream &st, size_t prefix) {
            return prefix + static_cast<size_t>(q.s.max_tokens) <= static_cast<size_t>(std::min(st.b.cap, c.tok.context));
        };
        if (!fits(*l.pos, q.prefix.size()) || (l.cfg && !fits(*l.neg, q.negative.size()))) {
            err = "generate: prefix + max_tokens exceed the context or the stream's capacity";
            return false;
        }
        if (!missing(*l.pos, q.prefix, l.rp, err) || (l.cfg && !missing(*l.neg, q.negative, l.rn, err))) return false;
        if (l.rp.empty() || (l.cfg && l.rn.empty())) {   // the last prefix token must be fed now: its logits start it
            err = "generate: the prefix is already in the cache (nothing to take the first logits from)";
            return false;
        }
        for (size_t j = 0; j < i; ++j)
            if (L[j].pos == l.pos || (l.neg && (L[j].neg == l.neg || L[j].pos == l.neg)) || (L[j].neg && L[j].neg == l.pos)) {
                err = "generate: two streams share a cache";
                return false;
            }
        // (int8 caches only: their attention is integer arithmetic, a query's the same in any batch; the fp16 / fp32
        // caches' attend_neon sums in fewer lanes past four queries - a valid sample still, not the one-at-a-time bits)
        l.lookup = q.abc && q.lookup && ar.options().kv == KvType::I8;
        if (l.lookup) l.look.add_all(l.history);   // (a history given - a score going on from its head - is looked up too)
        for (int b = 0; b < Lookup::kBuckets; ++b) l.kept[b] = 2.0, l.refused[b] = 1.0;   // (the prior 2 / 3)
        l.rng.seed(q.seed);   // the reference resets its generator a phase
    }
    // caches of different types in one call: the batch's attention takes the slower common path for all of them, an
    // int8 stream then not the arithmetic it has alone - so each stream on its own (the same tokens as alone)
    if (S > 1)
        for (const Lane &l : L)
            if (l.pos->b.type != L[0].pos->b.type || (l.neg && l.neg->b.type != L[0].pos->b.type)) {
                for (size_t i = 0; i < S; ++i) {
                    std::vector<GenResult> one;
                    std::vector<Stream *> ns;
                    if (L[i].neg) ns.push_back(L[i].neg);
                    const bool ok = generate_many(ar, {pos[i]}, ns, {reqs[i]}, one, pool, err);
                    if (!one.empty()) outs[i] = std::move(one[0]);
                    if (!ok) return false;
                }
                return true;
            }
    const int H = c.hidden;
    const Allowed a = allowed_rows(c, abc);
    if (!abc) ar.cache_embed(a.lo, a.hi);   // (the codes' rows; the score's are the whole text)
    const int na = a.size(), nr = a.hi - a.lo;
    const size_t rmax = S * static_cast<size_t>(kMaxDrafts + 2);
    std::vector<float> hv(rmax * static_cast<size_t>(H)), lv(rmax * static_cast<size_t>(nr)), ev(rmax),
        lc(static_cast<size_t>(na)), comb(static_cast<size_t>(na));
    Candidates cand;
    const double t0 = now();

    // ---- the prefixes: a stream whose cache and prefix are an earlier one's (takes of one song) takes its cache and
    // its first hidden state by copy - the bytes the same tokens' run would write
    std::vector<int> twin(S, -1);
    for (size_t i = 1; i < S; ++i)
        for (size_t j = 0; j < i && twin[i] < 0; ++j) {
            const Lane &x = L[i], &y = L[j];
            if (twin[j] >= 0 || x.cfg != y.cfg || x.pos->fed != y.pos->fed || x.req->prefix != y.req->prefix ||
                x.pos->b.len != static_cast<int>(x.pos->fed.size()) || y.pos->b.len != static_cast<int>(y.pos->fed.size()) ||
                x.pos->b.type != y.pos->b.type)
                continue;
            if (x.cfg && (x.neg->fed != y.neg->fed || x.req->negative != y.req->negative ||
                          x.neg->b.len != static_cast<int>(x.neg->fed.size()) ||
                          y.neg->b.len != static_cast<int>(y.neg->fed.size()) || x.neg->b.type != y.neg->b.type))
                continue;
            twin[i] = static_cast<int>(j);
        }
    {
        std::vector<FastAr::Seg> segs;
        int r = 0;
        for (size_t i = 0; i < S; ++i) {
            Lane &l = L[i];
            l.row = r;
            r += 1 + (l.cfg ? 1 : 0);
            if (twin[i] >= 0) continue;
            segs.push_back({&l.pos->b, l.rp.data(), static_cast<int>(l.rp.size()), {static_cast<int>(l.rp.size()) - 1},
                            hv.data() + static_cast<size_t>(l.row) * H});
            if (l.cfg)
                segs.push_back({&l.neg->b, l.rn.data(), static_cast<int>(l.rn.size()), {static_cast<int>(l.rn.size()) - 1},
                                hv.data() + static_cast<size_t>(l.row + 1) * H});
        }
        if (!ar.run(segs, pool, err)) return false;
        for (size_t i = 0; i < S; ++i) {
            Lane &l = L[i];
            if (twin[i] >= 0) {
                const Lane &y = L[static_cast<size_t>(twin[i])];
                if (!ar.branch_copy(y.pos->b, l.pos->b, y.pos->b.len) ||
                    (l.cfg && !ar.branch_copy(y.neg->b, l.neg->b, y.neg->b.len))) {
                    err = "generate: a stream's cache does not take its twin's";
                    return false;
                }
                std::copy_n(hv.begin() + static_cast<std::ptrdiff_t>(y.row) * H, static_cast<size_t>((l.cfg ? 2 : 1) * H),
                            hv.begin() + static_cast<std::ptrdiff_t>(l.row) * H);
            } else {
                l.out->prefill_tokens = static_cast<int>(l.rp.size() + l.rn.size());
            }
            l.pos->fed.insert(l.pos->fed.end(), l.rp.begin(), l.rp.end());
            if (l.cfg) l.neg->fed.insert(l.neg->fed.end(), l.rn.begin(), l.rn.end());
        }
    }
    const double t_prefill = now() - t0;

    // ---- the logits of the call's rows (lm_head's allowed rows, the extra one after them), as one product
    const auto logits_of = [&](int rows) {
        if (!ar.logits(hv.data(), rows, a.lo, a.hi, lv.data(), pool) ||
            (a.extra >= 0 && !ar.logits(hv.data(), rows, a.extra, a.extra + 1, ev.data(), pool))) {
            err = "generate: lm_head rows";
            return false;
        }
        return true;
    };
    const auto row_logits = [&](int row, float *out) {   // a row's, as the one-at-a-time head() lays them out
        std::copy(lv.begin() + static_cast<std::ptrdiff_t>(row) * nr, lv.begin() + static_cast<std::ptrdiff_t>(row + 1) * nr, out);
        if (a.extra >= 0) out[nr] = ev[static_cast<size_t>(row)];
    };
    // a lane's draw at step from its positive row (and its negative row: CFG); the cancel polled before every draw
    const auto draw_at = [&](Lane &l, int row, int nrow, int step, int &tok) {
        if (l.req->cancelled && l.req->cancelled()) {
            err = "cancelled";
            return false;
        }
        row_logits(row, lc.data());
        const float *lg = lc.data();
        if (l.cfg) {   // u + cfg (c - u), in fp32 (the reference rounds each operation to bf16)
            row_logits(nrow, comb.data());
            for (int i = 0; i < na; ++i)
                comb[static_cast<size_t>(i)] = comb[static_cast<size_t>(i)] + l.req->cfg * (lc[static_cast<size_t>(i)] - comb[static_cast<size_t>(i)]);
            lg = comb.data();
        }
        distribution_rows(lg, a, l.req->s, step, l.history, cand);
        if (cand.id.empty()) {
            err = "generate: no token left to sample";
            return false;
        }
        tok = draw(cand, l.rng);
        return true;
    };

    // ---- the first tokens: from the prefixes' last hidden states
    {
        int rows = 0;
        for (const Lane &l : L) rows = std::max(rows, l.row + 1 + (l.cfg ? 1 : 0));
        if (!logits_of(rows)) return false;
        for (Lane &l : L)
            if (!draw_at(l, l.row, l.row + 1, 0, l.token)) return false;
    }
    Cost cost(have_q8_mmla() && !ar.options().q4);
    for (;;) {
        // the drawn tokens: the end, the budget, else one more call
        int live = 0;
        for (Lane &l : L) {
            if (l.done) continue;
            if (l.req->on_token) l.req->on_token(l.step, l.token);
            if (l.token == a.end) {
                l.out->eos = true;
                l.done = true;
                continue;
            }
            l.history.push_back(l.token);
            l.out->tokens.push_back(l.token);
            if (l.lookup) l.look.add(l.history);
            if (l.step + 1 >= l.req->s.max_tokens) {   // (the last token not fed, as the one-at-a-time loop)
                l.done = true;
                continue;
            }
            ++live;
        }
        if (!live) break;
        // the drafts: one at a time to the stream whose next draft is likeliest kept, while the tokens a second grow
        // (a call of R rows yields a token a stream plus its kept drafts); none at all unless they win 3% together
        int base = 0;
        double e = 0.0;
        for (Lane &l : L) {
            l.k = 0;
            if (l.done) continue;
            base += 1 + (l.cfg ? 1 : 0);
            e += 1.0;
            if (!l.lookup) continue;
            // drafts only for steps whose token the loop would feed too: the cache never holds more than it would
            const int kmax = std::min(kMaxDrafts, l.req->s.max_tokens - 2 - l.step);
            int m = 0;
            const int got = kmax > 0 ? l.look.drafts(l.history, kmax, l.d, m) : 0;
            l.d.resize(static_cast<size_t>(got));
            l.bk = Lookup::bucket(m);
        }
        const bool lone = live == 1;
        int R = base;
        const double e0 = e;
        if (!(lone && cost.stale())) {   // (a lone stream's step without drafts every 64 calls: the intercept)
            for (;;) {
                Lane *best = nullptr;
                double bp = 0.0;
                for (Lane &l : L) {
                    if (l.done || !l.lookup || l.k >= static_cast<int>(l.d.size())) continue;
                    const double al = l.kept[l.bk] / (l.kept[l.bk] + l.refused[l.bk]);
                    const double pk = std::pow(al, l.k + 1);
                    if (pk > bp) bp = pk, best = &l;
                }
                if (!best || (e + bp) / cost.rel(R + 1) <= e / cost.rel(R)) break;
                ++best->k, ++R, e += bp;
            }
            if (e / cost.rel(R) <= 1.03 * e0 / cost.rel(base))
                for (Lane &l : L) l.k = 0, R = base;
        }
        // the call: every live stream's token and drafts (and its negative stream's token)
        std::vector<FastAr::Seg> segs;
        int r = 0;
        for (Lane &l : L) {
            if (l.done) continue;
            l.feed.assign(1, l.token);
            l.feed.insert(l.feed.end(), l.d.begin(), l.d.begin() + l.k);
            l.rows.resize(static_cast<size_t>(l.k + 1));
            std::iota(l.rows.begin(), l.rows.end(), 0);
            l.len0 = l.pos->b.len;
            l.row = r;
            segs.push_back({&l.pos->b, l.feed.data(), l.k + 1, l.rows, hv.data() + static_cast<size_t>(r) * H});
            r += l.k + 1;
            if (l.cfg) {
                segs.push_back({&l.neg->b, l.feed.data(), 1, {0}, hv.data() + static_cast<size_t>(r) * H});
                ++r;
            }
        }
        const double tc = now();
        if (!ar.run(segs, pool, err) || !logits_of(r)) return false;
        cost.seen(r, now() - tc);
        // each stream's draws: the drafts kept while the draw is the draft, then the token after them. A draw that
        // stops (cancelled) still leaves every stream's cache and `fed` in step: the fed token (and the drafts kept)
        bool stop = false;
        for (Lane &l : L) {
            if (l.done) continue;
            int j = 0, next = 0;
            while (!stop) {
                if (!draw_at(l, l.row + j, l.row + l.k + 1, l.step + 1, next)) {
                    stop = true;
                    break;
                }
                if (j >= l.k || next != l.d[static_cast<size_t>(j)]) break;
                ++j, ++l.step;   // the draft is the token the draw gives: kept (never the end: drafts are history)
                if (l.req->on_token) l.req->on_token(l.step, next);
                l.history.push_back(next);
                l.out->tokens.push_back(next);
                l.look.add(l.history);
            }
            l.pos->b.len = l.len0 + 1 + j;   // the drafts from the first refused one on out of the cache
            l.pos->fed.insert(l.pos->fed.end(), l.feed.begin(), l.feed.begin() + 1 + j);
            if (l.cfg) l.neg->fed.push_back(l.token);
            if (stop) continue;
            if (l.k > 0) {
                ++l.out->calls, l.out->drafted += l.k, l.out->accepted += j;
                l.out->slope = cost.slope();
                l.kept[l.bk] = 0.95 * l.kept[l.bk] + j, l.refused[l.bk] = 0.95 * l.refused[l.bk] + (j < l.k ? 1 : 0);
            }
            ++l.step;
            l.token = next;
        }
        if (stop) return false;
    }
    const double t_all = now() - t0;
    for (GenResult &o : outs) o.prefill_seconds = t_prefill, o.seconds = t_all;
    return true;
}

bool generate(FastAr &ar, Stream &pos, Stream *neg, const GenRequest &req, GenResult &out, ThreadPool &pool,
              std::string &err) {
    std::vector<GenResult> outs;
    std::vector<Stream *> ns;
    if (neg) ns.push_back(neg);
    const bool ok = generate_many(ar, {&pos}, ns, {req}, outs, pool, err);
    if (!outs.empty()) out = std::move(outs[0]);   // (a stopped one's tokens so far too, as on_token saw them)
    return ok;
}

}  // namespace geode::yue2
