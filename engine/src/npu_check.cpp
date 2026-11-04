// npu_check.cpp - the device report's NPU check (bench.hpp): libgeode_npu.so opened at run time, as the engine opens
// it (Model::setup_npu), and its gnpu_report (npu/host/report.c) run: what the phone's Hexagon is, whether Geode's
// module opens there, its products checked against the CPU and timed. Where the library will not load (no FastRPC, not
// a Snapdragon) that is the answer.
#include <cstdlib>
#include <string>

#include "bench.hpp"

#if defined(__ANDROID__)
#include <dlfcn.h>
#include <vector>
#endif

namespace geode::bench {

namespace {

std::string escaped(const std::string &s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

std::string off(const std::string &why) {
    return "{\"path\":\"off\",\"reason\":\"" + escaped(why) + "\",\"checks\":[]}";
}

}  // namespace

std::string npu_check(const std::string &step_file) {
#if defined(__ANDROID__)
    const char *path = std::getenv("GEODE_NPU_LIB") ? std::getenv("GEODE_NPU_LIB") : "libgeode_npu.so";
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        const char *e = dlerror();
        return off(std::string("libgeode_npu.so did not load: ") + (e ? e : "?"));
    }
    using Report = int (*)(const char *, const char *, char *, int);
    const auto report = reinterpret_cast<Report>(dlsym(lib, "gnpu_report"));
    if (!report) {
        dlclose(lib);
        return off("libgeode_npu.so has no gnpu_report");
    }
    std::string dir = ".";
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(report), &info) && info.dli_fname) {
        dir = info.dli_fname;
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    std::vector<char> out(65536, 0);
    const int e = report(dir.c_str(), step_file.c_str(), out.data(), static_cast<int>(out.size()));
    // the library stays loaded: FastRPC's sessions belong to the process, unloading under them is not safe
    return e ? off("gnpu_report failed (" + std::to_string(e) + ")") : std::string(out.data());
#else
    (void)step_file;
    return off("no NPU here (not Android)");
#endif
}

}  // namespace geode::bench

#include "bench_json.hpp"
#include "opencl.hpp"

namespace geode::bench {

// Which units can run a model here (the app's run-mode choice lists only these): the GPU OpenCL names (an Adreno: the
// engine's GPU paths), the NPU when Geode's module opens on the phone's Hexagon (its version: v73+ a prompt's
// products, v79+ the layers, so the NPU-only mode). A quick look - no model, no checks: those are npu_check's and
// gpu_check's.
std::string units() {
    Json j;
    j.begin_object();
    j.boolean("cpu", true);
    {
        cl::Device dev;
        std::string err;
        if (dev.open(err)) {
            const size_t a = dev.name.find("Adreno");
            int number = 0;
            if (a != std::string::npos) {
                size_t i = a + 6;
                while (i < dev.name.size() && (dev.name[i] < '0' || dev.name[i] > '9')) ++i;
                number = std::atoi(dev.name.c_str() + i);
            }
            j.str("gpu_name", dev.name).num("adreno", number).boolean("gpu", number >= 600);
        } else {
            j.boolean("gpu", false).str("gpu_reason", err);
        }
    }
#if defined(__ANDROID__)
    {
        const char *path = std::getenv("GEODE_NPU_LIB") ? std::getenv("GEODE_NPU_LIB") : "libgeode_npu.so";
        void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        using Open = void *(*)(const char *, char *, int);
        const auto open = lib ? reinterpret_cast<Open>(dlsym(lib, "gnpu_open")) : nullptr;
        const auto close = lib ? reinterpret_cast<void (*)(void *)>(dlsym(lib, "gnpu_close")) : nullptr;
        const auto arch = lib ? reinterpret_cast<int (*)()>(dlsym(lib, "gnpu_arch")) : nullptr;
        const auto skel_arch = lib ? reinterpret_cast<int (*)(int)>(dlsym(lib, "gnpu_skel_arch")) : nullptr;
        if (!open || !close) {
            const char *e = lib ? nullptr : dlerror();
            j.boolean("npu", false).str("npu_reason", lib ? "libgeode_npu.so lacks its interface"
                                                          : std::string("libgeode_npu.so did not load: ") + (e ? e : "?"));
        } else {
            std::string dir = ".";
            Dl_info info;
            if (dladdr(reinterpret_cast<void *>(open), &info) && info.dli_fname) {
                dir = info.dli_fname;
                const size_t slash = dir.find_last_of('/');
                dir = slash == std::string::npos ? "." : dir.substr(0, slash);
            }
            char err[256] = {0};
            void *g = open(dir.c_str(), err, sizeof err);
            const int v = g && arch && skel_arch ? skel_arch(arch()) : 0;
            if (g) close(g);
            // (the library stays loaded: FastRPC's sessions belong to the process)
            j.boolean("npu", g != nullptr && v >= 73).num("hexagon", v);
            if (!g) j.str("npu_reason", err);
            else if (v < 73) j.str("npu_reason", "Hexagon v" + std::to_string(v) + ": v73 and newer");
        }
    }
#else
    j.boolean("npu", false).str("npu_reason", "not Android");
#endif
    j.end_object();
    return j.take();
}

}  // namespace geode::bench
