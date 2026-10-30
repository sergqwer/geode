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
