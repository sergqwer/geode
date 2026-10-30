// bench.cpp - system information, DRAM, flash and CPU benchmarks (the GPU's are in vk_bench.cpp).
#include "bench.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "bench_json.hpp"
#include "json.hpp"
#include "kernels.hpp"
#include "platform.hpp"
#include "threads.hpp"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/auxv.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace geode::bench {

namespace fs = std::filesystem;

namespace {

double since(double t0) { return now_seconds() - t0; }

// ---- the CPU ----

struct Core {
    int cpu = 0;
    std::string part;
    int max_mhz = 0;
};

[[maybe_unused]] std::string part_name(unsigned implementer, unsigned part) {
    if (implementer == 0x41) {   // Arm
        static const std::map<unsigned, const char *> arm = {
            {0xd03, "Cortex-A53"}, {0xd04, "Cortex-A35"}, {0xd05, "Cortex-A55"}, {0xd07, "Cortex-A57"},
            {0xd08, "Cortex-A72"}, {0xd09, "Cortex-A73"}, {0xd0a, "Cortex-A75"}, {0xd0b, "Cortex-A76"},
            {0xd0d, "Cortex-A77"}, {0xd41, "Cortex-A78"}, {0xd44, "Cortex-X1"},  {0xd46, "Cortex-A510"},
            {0xd47, "Cortex-A710"}, {0xd48, "Cortex-X2"}, {0xd4b, "Cortex-A78C"}, {0xd4d, "Cortex-A715"},
            {0xd4e, "Cortex-X3"}, {0xd80, "Cortex-A520"}, {0xd81, "Cortex-A720"}, {0xd82, "Cortex-X4"},
            {0xd85, "Cortex-X925"}, {0xd87, "Cortex-A725"}};
        const auto it = arm.find(part);
        if (it != arm.end()) return it->second;
    }
    if (implementer == 0x51) {   // Qualcomm
        static const std::map<unsigned, const char *> qc = {
            // 0x002: the Snapdragon 8 Elite Gen 5's (a Galaxy S26 Ultra, Remote Test Lab)
            {0x001, "Oryon"}, {0x002, "Oryon v3"}, {0x800, "Kryo 2xx Gold (Cortex-A73)"}, {0x801, "Kryo 2xx Silver (Cortex-A53)"},
            {0x802, "Kryo 3xx Gold (Cortex-A75)"}, {0x803, "Kryo 3xx Silver (Cortex-A55)"},
            {0x804, "Kryo 4xx Gold (Cortex-A76)"}, {0x805, "Kryo 4xx Silver (Cortex-A55)"}};
        const auto it = qc.find(part);
        if (it != qc.end()) return it->second;
    }
    char b[48];
    std::snprintf(b, sizeof b, "0x%02x:0x%03x", implementer, part);
    return b;
}

std::vector<Core> cores() {
    std::vector<Core> out;
#if defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int cpu = -1;
    unsigned impl = 0;
    auto flush = [&](unsigned part) {
        if (cpu < 0) return;
        Core c;
        c.cpu = cpu;
        c.part = part_name(impl, part);
        std::ifstream fr("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq");
        int khz = 0;
        if (fr >> khz) c.max_mhz = khz / 1000;
        out.push_back(c);
    };
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string k = line.substr(0, colon);
        k.erase(k.find_last_not_of(" \t") + 1);
        const std::string v = line.substr(colon + 1);
        if (k == "processor") cpu = std::atoi(v.c_str());
        if (k == "CPU implementer") impl = static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 0));
        if (k == "CPU part") flush(static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 0)));
    }
#endif
    if (out.empty())
        for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency()); ++i) out.push_back({static_cast<int>(i), "", 0});
    return out;
}

struct Features {
    bool neon = false, fp16 = false, dotprod = false, i8mm = false, bf16 = false, sve = false, sve2 = false, sme = false;
};

Features features() {
    Features f;
#if defined(__aarch64__) && defined(__linux__)
    const unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
    f.neon = h1 & (1ul << 1);
    f.fp16 = h1 & (1ul << 10);      // asimdhp: FP16 arithmetic in NEON
    f.dotprod = h1 & (1ul << 20);   // asimddp: SDOT / UDOT
    f.sve = h1 & (1ul << 22);
    f.sve2 = h2 & (1ul << 1);
    f.i8mm = h2 & (1ul << 13);      // SMMLA: int8 matrix multiply
    f.bf16 = h2 & (1ul << 14);      // BFDOT / BFMMLA
    f.sme = h2 & (1ul << 23);
#endif
    return f;
}

std::string prop(const char *name) {
#if defined(__ANDROID__)
    char v[PROP_VALUE_MAX] = {};
    __system_property_get(name, v);
    return v;
#else
    (void)name;
    return "";
#endif
}

// ---- running on chosen cores ----

// Runs fn(thread index) on one thread per cpu (pinned), all started together; returns the wall time.
template <typename F>
double run_on(const std::vector<int> &cpus, int threads, F fn) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    const int n = cpus.empty() ? threads : static_cast<int>(cpus.size());
    for (int t = 0; t < n; ++t)
        ts.emplace_back([&, t] {
            if (!cpus.empty()) pin_current_thread({cpus[static_cast<size_t>(t)]});
            ready.fetch_add(1);
            while (!go.load()) cpu_relax();
            fn(t);
        });
    while (ready.load() < n) cpu_relax();
    const double t0 = now_seconds();
    go.store(true);
    for (auto &t : ts) t.join();
    return since(t0);
}

// ---- DRAM ----

uint64_t read_slice(const uint8_t *p, size_t n) {
#if defined(__aarch64__)
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 64 <= n; i += 64) {
        a0 = veorq_u64(a0, vld1q_u64(reinterpret_cast<const uint64_t *>(p + i)));
        a1 = veorq_u64(a1, vld1q_u64(reinterpret_cast<const uint64_t *>(p + i + 16)));
        a2 = veorq_u64(a2, vld1q_u64(reinterpret_cast<const uint64_t *>(p + i + 32)));
        a3 = veorq_u64(a3, vld1q_u64(reinterpret_cast<const uint64_t *>(p + i + 48)));
    }
    const uint64x2_t a = veorq_u64(veorq_u64(a0, a1), veorq_u64(a2, a3));
    return vgetq_lane_u64(a, 0) ^ vgetq_lane_u64(a, 1);
#else
    uint64_t a = 0;
    for (size_t i = 0; i + 8 <= n; i += 8) {
        uint64_t v;
        std::memcpy(&v, p + i, 8);
        a ^= v;
    }
    return a;
#endif
}

std::atomic<uint64_t> g_sink{0};

// bytes moved per second, each thread on its slice for about `seconds`
double bandwidth(uint8_t *buf, size_t bytes, const std::vector<int> &cpus, int threads, int kind, double seconds) {
    const int n = cpus.empty() ? threads : static_cast<int>(cpus.size());
    const size_t slice = bytes / static_cast<size_t>(n) / 4096 * 4096;
    std::vector<uint64_t> moved(static_cast<size_t>(n), 0);
    const double wall = run_on(cpus, threads, [&](int t) {
        uint8_t *p = buf + static_cast<size_t>(t) * slice;
        const double t0 = now_seconds();
        uint64_t acc = 0;
        int pass = 0;
        while (since(t0) < seconds) {
            if (kind == 0) {
                acc ^= read_slice(p, slice);
                moved[static_cast<size_t>(t)] += slice;
            } else if (kind == 1) {
                std::memset(p, pass & 0xff, slice);
                moved[static_cast<size_t>(t)] += slice;
            } else {
                std::memcpy(p + slice / 2, p, slice / 2);   // counted as read + write
                moved[static_cast<size_t>(t)] += slice;
            }
            ++pass;
        }
        g_sink.fetch_xor(acc);
    });
    uint64_t total = 0;
    for (uint64_t m : moved) total += m;
    return static_cast<double>(total) / wall;
}

double latency_ns(int cpu) {
    // a random cycle through 64 MiB, one cache line per hop: every load waits for the one before
    const size_t lines = (64u << 20) / 64;
    std::vector<uint32_t> order(lines);
    for (size_t i = 0; i < lines; ++i) order[i] = static_cast<uint32_t>(i);
    std::mt19937 rng(7);
    for (size_t i = lines - 1; i > 0; --i) std::swap(order[i], order[std::uniform_int_distribution<size_t>(0, i - 1)(rng)]);
    std::vector<uint32_t> next(lines * 16);
    for (size_t i = 0; i < lines; ++i) next[order[i] * 16] = order[(i + 1) % lines];
    double ns = 0.0;
    run_on(cpu >= 0 ? std::vector<int>{cpu} : std::vector<int>{}, 1, [&](int) {
        uint32_t at = order[0];
        const int steps = 4'000'000;
        const double t0 = now_seconds();
        for (int s = 0; s < steps; ++s) at = next[static_cast<size_t>(at) * 16];
        ns = since(t0) * 1e9 / steps;
        g_sink.fetch_xor(at);
    });
    return ns;
}

// ---- CPU peaks (NEON) ----

#if defined(__aarch64__)

// The peaks run hand-written instructions (inline asm): 12 independent accumulator chains, one instruction each per
// turn. The float ones multiply by zero, so the values stay put (NEON's timing does not depend on the data). A
// compiler sees through C-level versions of such loops and folds them away.
#define GEODE_PEAK_LOOP(ASM_BODY, OPS_PER_TURN)                       \
    uint64_t turns = 0;                                                \
    const double t0 = now_seconds();                                   \
    do {                                                               \
        for (int r = 0; r < 1024; ++r) {                               \
            ASM_BODY                                                   \
        }                                                              \
        turns += 1024;                                                 \
    } while (since(t0) < seconds);                                     \
    return static_cast<double>(turns) * (OPS_PER_TURN) / since(t0);

double peak_fp32(double seconds) {
    float32x4_t a[12];
    for (int k = 0; k < 12; ++k) a[k] = vdupq_n_f32(0.5f + 0.01f * k);
    const float32x4_t x = vdupq_n_f32(0.999f), y = vdupq_n_f32(0.0f);
    GEODE_PEAK_LOOP(
        asm volatile(
        "fmla %0.4s, %12.4s, %13.4s\n"
        "fmla %1.4s, %12.4s, %13.4s\n"
        "fmla %2.4s, %12.4s, %13.4s\n"
        "fmla %3.4s, %12.4s, %13.4s\n"
        "fmla %4.4s, %12.4s, %13.4s\n"
        "fmla %5.4s, %12.4s, %13.4s\n"
        "fmla %6.4s, %12.4s, %13.4s\n"
        "fmla %7.4s, %12.4s, %13.4s\n"
        "fmla %8.4s, %12.4s, %13.4s\n"
        "fmla %9.4s, %12.4s, %13.4s\n"
        "fmla %10.4s, %12.4s, %13.4s\n"
        "fmla %11.4s, %12.4s, %13.4s\n"
        : "+w"(a[0]), "+w"(a[1]), "+w"(a[2]), "+w"(a[3]), "+w"(a[4]), "+w"(a[5]), "+w"(a[6]), "+w"(a[7]), "+w"(a[8]), "+w"(a[9]), "+w"(a[10]), "+w"(a[11])
        : "w"(x), "w"(y));
        , 12.0 * 4 * 2)
}

double peak_fp16(double seconds) {
    float16x8_t a[12];
    for (int k = 0; k < 12; ++k) a[k] = vdupq_n_f16(static_cast<float16_t>(0.5f));
    const float16x8_t x = vdupq_n_f16(static_cast<float16_t>(1.0f)), y = vdupq_n_f16(static_cast<float16_t>(0.0f));
    GEODE_PEAK_LOOP(
        asm volatile(
        "fmla %0.8h, %12.8h, %13.8h\n"
        "fmla %1.8h, %12.8h, %13.8h\n"
        "fmla %2.8h, %12.8h, %13.8h\n"
        "fmla %3.8h, %12.8h, %13.8h\n"
        "fmla %4.8h, %12.8h, %13.8h\n"
        "fmla %5.8h, %12.8h, %13.8h\n"
        "fmla %6.8h, %12.8h, %13.8h\n"
        "fmla %7.8h, %12.8h, %13.8h\n"
        "fmla %8.8h, %12.8h, %13.8h\n"
        "fmla %9.8h, %12.8h, %13.8h\n"
        "fmla %10.8h, %12.8h, %13.8h\n"
        "fmla %11.8h, %12.8h, %13.8h\n"
        : "+w"(a[0]), "+w"(a[1]), "+w"(a[2]), "+w"(a[3]), "+w"(a[4]), "+w"(a[5]), "+w"(a[6]), "+w"(a[7]), "+w"(a[8]), "+w"(a[9]), "+w"(a[10]), "+w"(a[11])
        : "w"(x), "w"(y));
        , 12.0 * 8 * 2)
}

double peak_sdot(double seconds) {
    int32x4_t a[12];
    for (int k = 0; k < 12; ++k) a[k] = vdupq_n_s32(k);
    const int8x16_t x = vdupq_n_s8(3), y = vdupq_n_s8(-2);
    GEODE_PEAK_LOOP(
        asm volatile(
        "sdot %0.4s, %12.16b, %13.16b\n"
        "sdot %1.4s, %12.16b, %13.16b\n"
        "sdot %2.4s, %12.16b, %13.16b\n"
        "sdot %3.4s, %12.16b, %13.16b\n"
        "sdot %4.4s, %12.16b, %13.16b\n"
        "sdot %5.4s, %12.16b, %13.16b\n"
        "sdot %6.4s, %12.16b, %13.16b\n"
        "sdot %7.4s, %12.16b, %13.16b\n"
        "sdot %8.4s, %12.16b, %13.16b\n"
        "sdot %9.4s, %12.16b, %13.16b\n"
        "sdot %10.4s, %12.16b, %13.16b\n"
        "sdot %11.4s, %12.16b, %13.16b\n"
        : "+w"(a[0]), "+w"(a[1]), "+w"(a[2]), "+w"(a[3]), "+w"(a[4]), "+w"(a[5]), "+w"(a[6]), "+w"(a[7]), "+w"(a[8]), "+w"(a[9]), "+w"(a[10]), "+w"(a[11])
        : "w"(x), "w"(y));
        , 12.0 * 16 * 2)
}

__attribute__((target("arch=armv8.2-a+dotprod+fp16+i8mm"))) double peak_i8mm(double seconds) {
    int32x4_t a[12];
    for (int k = 0; k < 12; ++k) a[k] = vdupq_n_s32(k);
    const int8x16_t x = vdupq_n_s8(3), y = vdupq_n_s8(-2);
    GEODE_PEAK_LOOP(
        asm volatile(
        ".arch_extension i8mm\n"
        "smmla %0.4s, %12.16b, %13.16b\n"
        "smmla %1.4s, %12.16b, %13.16b\n"
        "smmla %2.4s, %12.16b, %13.16b\n"
        "smmla %3.4s, %12.16b, %13.16b\n"
        "smmla %4.4s, %12.16b, %13.16b\n"
        "smmla %5.4s, %12.16b, %13.16b\n"
        "smmla %6.4s, %12.16b, %13.16b\n"
        "smmla %7.4s, %12.16b, %13.16b\n"
        "smmla %8.4s, %12.16b, %13.16b\n"
        "smmla %9.4s, %12.16b, %13.16b\n"
        "smmla %10.4s, %12.16b, %13.16b\n"
        "smmla %11.4s, %12.16b, %13.16b\n"
        : "+w"(a[0]), "+w"(a[1]), "+w"(a[2]), "+w"(a[3]), "+w"(a[4]), "+w"(a[5]), "+w"(a[6]), "+w"(a[7]), "+w"(a[8]), "+w"(a[9]), "+w"(a[10]), "+w"(a[11])
        : "w"(x), "w"(y));
        , 12.0 * 32 * 2)   // a 2x8 by 8x2 product: 32 multiply-adds
}

__attribute__((target("arch=armv8.2-a+dotprod+fp16+bf16"))) double peak_bf16(double seconds) {
    float32x4_t a[12];
    for (int k = 0; k < 12; ++k) a[k] = vdupq_n_f32(0.5f);
    const bfloat16x8_t x = vreinterpretq_bf16_u16(vdupq_n_u16(0x3f80)), y = vreinterpretq_bf16_u16(vdupq_n_u16(0x0000));
    GEODE_PEAK_LOOP(
        asm volatile(
        ".arch_extension bf16\n"
        "bfdot %0.4s, %12.8h, %13.8h\n"
        "bfdot %1.4s, %12.8h, %13.8h\n"
        "bfdot %2.4s, %12.8h, %13.8h\n"
        "bfdot %3.4s, %12.8h, %13.8h\n"
        "bfdot %4.4s, %12.8h, %13.8h\n"
        "bfdot %5.4s, %12.8h, %13.8h\n"
        "bfdot %6.4s, %12.8h, %13.8h\n"
        "bfdot %7.4s, %12.8h, %13.8h\n"
        "bfdot %8.4s, %12.8h, %13.8h\n"
        "bfdot %9.4s, %12.8h, %13.8h\n"
        "bfdot %10.4s, %12.8h, %13.8h\n"
        "bfdot %11.4s, %12.8h, %13.8h\n"
        : "+w"(a[0]), "+w"(a[1]), "+w"(a[2]), "+w"(a[3]), "+w"(a[4]), "+w"(a[5]), "+w"(a[6]), "+w"(a[7]), "+w"(a[8]), "+w"(a[9]), "+w"(a[10]), "+w"(a[11])
        : "w"(x), "w"(y));
        , 12.0 * 8 * 2)   // 4 lanes x a 2-element dot
}

#undef GEODE_PEAK_LOOP

#endif

// one core of each cluster (the first), as (name, cpu)
[[maybe_unused]] std::vector<std::pair<std::string, int>> cluster_leads(const std::vector<Core> &cs) {
    std::vector<std::pair<std::string, int>> out;
    std::map<std::pair<std::string, int>, int> seen;
    for (const auto &c : cs) {
        const auto key = std::make_pair(c.part, c.max_mhz);
        if (seen.count(key)) continue;
        seen[key] = c.cpu;
        out.emplace_back(c.part.empty() ? "cpu" + std::to_string(c.cpu) : c.part, c.cpu);
    }
    std::sort(out.begin(), out.end(), [&](const auto &a, const auto &b) {
        auto mhz = [&](int cpu) {
            for (const auto &c : cs)
                if (c.cpu == cpu) return c.max_mhz;
            return 0;
        };
        return mhz(a.second) > mhz(b.second);
    });
    return out;
}

// ---- flash ----

struct FileSet {
    std::vector<std::unique_ptr<RandomFile>> files;
    std::vector<uint64_t> sizes;
    uint64_t total = 0;
};

}  // namespace

// ---- the public functions ----

std::string system_info() {
    const auto cs = cores();
    const Features f = features();
    Json j;
    j.begin_object();
    j.str("soc", prop("ro.soc.model"));
    j.str("soc_maker", prop("ro.soc.manufacturer"));
    j.str("platform", prop("ro.board.platform"));
    j.str("device", prop("ro.product.manufacturer") + " " + prop("ro.product.model"));
    j.str("android", prop("ro.build.version.release"));
    j.begin_array("clusters");
    std::map<std::pair<std::string, int>, std::vector<int>> groups;
    for (const auto &c : cs) groups[{c.part, c.max_mhz}].push_back(c.cpu);
    std::vector<std::pair<std::pair<std::string, int>, std::vector<int>>> ordered(groups.begin(), groups.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) { return a.first.second > b.first.second; });
    for (const auto &[key, cpus] : ordered) {
        j.begin_object();
        j.str("core", key.first);
        j.num("count", static_cast<double>(cpus.size()));
        j.num("max_mhz", key.second);
        j.begin_array("cpus");
        for (int c : cpus) j.num(nullptr, c);
        j.end_array();
        j.end_object();
    }
    j.end_array();
    j.begin_object("features");
    j.boolean("neon", f.neon);
    j.boolean("fp16", f.fp16);
    j.boolean("dotprod", f.dotprod);
    j.boolean("i8mm", f.i8mm);
    j.boolean("bf16", f.bf16);
    j.boolean("sve", f.sve);
    j.boolean("sve2", f.sve2);
    j.boolean("sme", f.sme);
    j.end_object();
#if defined(__linux__)
    j.num("ram_bytes", static_cast<double>(sysconf(_SC_PHYS_PAGES)) * static_cast<double>(sysconf(_SC_PAGESIZE)));
#endif
    j.boolean("engine_dotprod_kernels", have_dotprod_kernels());
    j.end_object();
    return j.take();
}

std::string memory() {
    const size_t bytes = 256u << 20;
    AlignedBuffer buf = aligned_bytes(bytes);
    Json j;
    j.begin_object();
    if (!buf) {
        j.str("error", "no memory for the test buffer");
        j.end_object();
        return j.take();
    }
    std::memset(buf.get(), 1, bytes);   // touch every page before timing
    const auto big = big_cores(), little = little_cores();
    std::vector<int> all;
    for (const auto &c : cores()) all.push_back(c.cpu);
    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    struct Config {
        const char *name;
        std::vector<int> cpus;
        int threads;
    };
    std::vector<Config> configs;
    if (!big.empty()) {
        configs.push_back({"one_big_core", {big[0]}, 1});
        configs.push_back({"big_cores", big, 0});
        if (!little.empty()) configs.push_back({"little_cores", little, 0});
        configs.push_back({"all_cores", all, 0});
    } else {
        configs.push_back({"one_core", {}, 1});
        configs.push_back({"all_cores", {}, hw});
    }
    j.begin_array("configs");
    for (const auto &cfg : configs) {
        j.begin_object();
        j.str("name", cfg.name);
        j.num("threads", cfg.cpus.empty() ? cfg.threads : static_cast<double>(cfg.cpus.size()));
        bandwidth(buf.get(), bytes, cfg.cpus, cfg.threads, 0, 0.3);   // warm-up: clocks up
        j.num("read_gbs", bandwidth(buf.get(), bytes, cfg.cpus, cfg.threads, 0, 0.6) / 1e9);
        j.num("write_gbs", bandwidth(buf.get(), bytes, cfg.cpus, cfg.threads, 1, 0.4) / 1e9);
        j.num("copy_gbs", bandwidth(buf.get(), bytes, cfg.cpus, cfg.threads, 2, 0.4) / 1e9);
        j.end_object();
    }
    j.end_array();
    j.num("latency_ns", latency_ns(big.empty() ? -1 : big[0]));
    j.end_object();
    return j.take();
}

std::string flash(const std::string &dir) {
    Json j;
    j.begin_object();
    FileSet set;
    std::string err;
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("experts-", 0) == 0 && e.path().extension() == ".bin") {
            auto f = std::make_unique<RandomFile>();
            if (f->open(e.path().string(), err, true)) {
                set.sizes.push_back(f->size());
                set.total += f->size();
                set.files.push_back(std::move(f));
            }
        }
    }
    // the model's expert block, read as the engine reads it
    size_t expert_block = 0;
    {
        std::ifstream mf(fs::path(dir) / "geode.json", std::ios::binary);
        std::ostringstream ss;
        ss << mf.rdbuf();
        json::Value m;
        std::string perr;
        if (!ss.str().empty() && json::parse(ss.str(), m, perr)) expert_block = static_cast<size_t>(m["experts"]["layers"].at(0)["stride"].i64());
    }
    const fs::path tmp = fs::path(dir) / "geode-bench.tmp";
    bool made = false;
    if (set.files.empty()) {
        // no model here: a 1 GiB file of random bytes (the write speed comes with it)
        const size_t total = 1ull << 30, chunk = 4u << 20;
        AlignedBuffer w = aligned_bytes(chunk);
        std::mt19937_64 rng(1);
        for (size_t i = 0; i < chunk / 8; ++i) reinterpret_cast<uint64_t *>(w.get())[i] = rng();
        const double t0 = now_seconds();
#if defined(__linux__)
        const int fd = ::open(tmp.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        bool ok = fd >= 0;
        for (size_t at = 0; ok && at < total; at += chunk) {
            reinterpret_cast<uint64_t *>(w.get())[0] = at;   // every chunk different
            ok = ::write(fd, w.get(), chunk) == static_cast<ssize_t>(chunk);
        }
        if (fd >= 0) {
            ::fsync(fd);
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // so the reads below come from flash
            ::close(fd);
        }
#else
        std::ofstream of(tmp, std::ios::binary);
        for (size_t at = 0; at < total && of; at += chunk) of.write(reinterpret_cast<const char *>(w.get()), chunk);
        of.close();
        const bool ok = static_cast<bool>(of);
#endif
        const double dt = since(t0);
        if (!ok) {
            std::error_code rm;
            fs::remove(tmp, rm);
            j.str("error", "cannot write a test file in " + dir);
            j.end_object();
            return j.take();
        }
        made = true;
        j.num("write_mbs", static_cast<double>(total) / dt / 1e6);
        auto f = std::make_unique<RandomFile>();
        if (f->open(tmp.string(), err, true)) {
            set.sizes.push_back(f->size());
            set.total += f->size();
            set.files.push_back(std::move(f));
        }
    }
    if (set.files.empty()) {
        j.str("error", err.empty() ? "nothing to read" : err);
        j.end_object();
        return j.take();
    }
    j.str("source", made ? "a temporary 1 GiB file" : "the model's expert files");
    j.num("bytes", static_cast<double>(set.total));
    j.boolean("direct", set.files[0]->direct());

    const auto little = little_cores();
    auto random_read = [&](size_t block, int threads, double seconds, double &lat_ms) {
        std::vector<uint64_t> moved(static_cast<size_t>(threads), 0), ops(static_cast<size_t>(threads), 0);
        std::vector<double> busy(static_cast<size_t>(threads), 0);
        std::vector<int> cpus;
        for (int t = 0; t < threads && !little.empty(); ++t) cpus.push_back(little[static_cast<size_t>(t) % little.size()]);
        const double wall = run_on(cpus, threads, [&](int t) {
            AlignedBuffer b = aligned_bytes(block);
            std::mt19937_64 rng(static_cast<uint64_t>(t) * 977 + 1);
            const double t0 = now_seconds();
            while (since(t0) < seconds) {
                uint64_t r = std::uniform_int_distribution<uint64_t>(0, set.total - 1)(rng);
                size_t fi = 0;
                while (r >= set.sizes[fi]) r -= set.sizes[fi++];
                const uint64_t span = set.sizes[fi] > block ? set.sizes[fi] - block : 0;
                const uint64_t off = span ? std::uniform_int_distribution<uint64_t>(0, span)(rng) / 4096 * 4096 : 0;
                if (!set.files[fi]->read_at(off, b.get(), block)) break;
                moved[static_cast<size_t>(t)] += block;
                ++ops[static_cast<size_t>(t)];
            }
            busy[static_cast<size_t>(t)] = since(t0);
        });
        uint64_t total = 0, n = 0;
        double b = 0;
        for (int t = 0; t < threads; ++t) {
            total += moved[static_cast<size_t>(t)];
            n += ops[static_cast<size_t>(t)];
            b += busy[static_cast<size_t>(t)];
        }
        lat_ms = n ? b / static_cast<double>(n) * 1e3 : 0.0;
        return static_cast<double>(total) / wall / 1e6;
    };
    std::vector<size_t> blocks = {1u << 20};
    if (expert_block && expert_block != blocks[0]) blocks.push_back(expert_block);
    blocks.push_back(256u << 10);
    j.begin_array("random");
    for (size_t block : blocks)
        for (int threads : {1, 2, 4}) {
            double lat = 0;
            const double mbs = random_read(block, threads, 1.2, lat);
            j.begin_object();
            j.num("block", static_cast<double>(block));
            j.num("threads", threads);
            j.num("mbs", mbs);
            j.num("latency_ms", lat);
            j.boolean("expert_block", block == expert_block);
            j.end_object();
        }
    j.end_array();
    {
        // sequential: 4 MiB reads from the start of the largest file
        size_t fi = 0;
        for (size_t i = 1; i < set.sizes.size(); ++i)
            if (set.sizes[i] > set.sizes[fi]) fi = i;
        const size_t chunk = 4u << 20;
        AlignedBuffer b = aligned_bytes(chunk);
        uint64_t moved = 0;
        const double t0 = now_seconds();
        for (uint64_t off = 0; off + chunk <= set.sizes[fi] && since(t0) < 1.5; off += chunk)
            if (set.files[fi]->read_at(off, b.get(), chunk)) moved += chunk;
        j.num("sequential_mbs", static_cast<double>(moved) / since(t0) / 1e6);
    }
    set.files.clear();
    if (made) {
        std::error_code rm;
        fs::remove(tmp, rm);
    }
    j.end_object();
    return j.take();
}

std::string cpu() {
    [[maybe_unused]] const auto cs = cores();
    [[maybe_unused]] const Features f = features();
    const auto big = big_cores();
    Json j;
    j.begin_object();
    j.begin_array("peaks");
#if defined(__aarch64__)
    for (const auto &[name, cpu] : cluster_leads(cs)) {
        auto one = [&, cpu = cpu](double (*fn)(double)) {
            double r = 0;
            run_on({cpu}, 1, [&](int) {
                fn(0.3);   // warm-up: the governor raises the clock only after a few hundred ms of load
                r = fn(0.3);
            });
            return r / 1e9;
        };
        j.begin_object();
        j.str("core", name);
        j.num("cpu", cpu);
        j.num("fp32_gflops", one(peak_fp32));
        if (f.fp16) j.num("fp16_gflops", one(peak_fp16));
        if (f.dotprod) j.num("int8_dot_gops", one(peak_sdot));
        if (f.i8mm) j.num("int8_mmla_gops", one(peak_i8mm));
        if (f.bf16) j.num("bf16_dot_gflops", one(peak_bf16));
        j.end_object();
    }
#endif
    j.end_array();

    // the engine's kernels on a matrix the size of an attention projection (it streams from DRAM, like the model)
    const int64_t rows = 4096, cols = 2048;
    std::mt19937 rng(42);
    std::vector<float> x(static_cast<size_t>(cols));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<QAct> acts(16);
    std::vector<const QAct *> ap(4), ap16(16);
    for (int k = 0; k < 16; ++k) {
        for (float &v : x) v = nd(rng);
        quantize_activation(x.data(), cols, acts[static_cast<size_t>(k)]);
        ap16[static_cast<size_t>(k)] = &acts[static_cast<size_t>(k)];
        if (k < 4) ap[static_cast<size_t>(k)] = &acts[static_cast<size_t>(k)];
    }
    ThreadPool pool(big.empty() ? static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))
                                : static_cast<int>(big.size()), big);
    j.num("big_cores", pool.size());
    j.begin_array("kernels");
    for (DType t : {DType::Q2, DType::Q3, DType::NVFP4, DType::Q8}) {
        const size_t rb = row_bytes(t, cols);
        const size_t code = rb - static_cast<size_t>(cols / kBlock);
        AlignedBuffer m = aligned_bytes(rows * rb);
        std::uniform_int_distribution<int> byte(0, 255), scale(0x30, 0x48);
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t *row = m.get() + r * rb;
            for (size_t i = 0; i < code; ++i) {
                row[i] = static_cast<uint8_t>(byte(rng));
                if (t == DType::Q8 && row[i] == 0x80) row[i] = 0x81;
            }
            for (size_t i = code; i < rb; ++i) row[i] = static_cast<uint8_t>(scale(rng));
        }
        const QView v{t, rows, cols, rb, m.get(), 0.01f};
        std::vector<std::vector<float>> ys(16, std::vector<float>(static_cast<size_t>(rows)));
        std::vector<float *> yp;
        for (auto &y : ys) yp.push_back(y.data());
        auto rate = [&](auto run, double weights_per_run) {
            const double w0 = now_seconds();
            while (since(w0) < 0.2) run();   // warm-up (the clock, the caches)
            int reps = 0;
            const double t0 = now_seconds();
            do {
                run();
                ++reps;
            } while (since(t0) < 0.3);
            return weights_per_run * reps / since(t0) / 1e9;
        };
        double one = 0, four = 0, sixteen = 0;
        run_on(big.empty() ? std::vector<int>{} : std::vector<int>{big[0]}, 1, [&](int) {
            one = rate([&] { matvec_rows_q(v, acts[0], yp[0], 0, rows); }, static_cast<double>(rows * cols));
            four = rate([&] { matvec_rows_q_multi(v, ap.data(), 4, yp.data(), 0, rows); },
                        static_cast<double>(rows * cols) * 4);
            // a prompt's batch: sixteen tokens a pass
            sixteen = rate([&] { matvec_rows_q_multi(v, ap16.data(), 16, yp.data(), 0, rows); },
                           static_cast<double>(rows * cols) * 16);
        });
        pin_current_thread(big);
        const double all = rate([&] {
            pool.parallel_for(rows, [&](int64_t r0, int64_t r1) { matvec_rows_q(v, acts[0], yp[0], r0, r1); }, 16);
        }, static_cast<double>(rows * cols));
        j.begin_object();
        j.str("format", dtype_name(t));
        j.num("bits_per_weight", static_cast<double>(rb) * 8.0 / static_cast<double>(cols));
        j.num("one_core_gw", one);
        j.num("big_cores_gw", all);
        j.num("four_tokens_one_core_gw", four);
        j.num("sixteen_tokens_one_core_gw", sixteen);
        j.end_object();
    }
    j.end_array();
    j.boolean("neon_kernels", have_dotprod_kernels());
    j.end_object();
    return j.take();
}

}  // namespace geode::bench
