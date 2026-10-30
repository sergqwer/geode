// opencl.cpp - loading libOpenCL at run time, the device, programs (see opencl.hpp).
#include "opencl.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace geode::cl {

namespace {

Api g_api;
bool g_loaded = false;
std::string g_error;
std::once_flag g_once;

void *open_library() {
#if defined(_WIN32)
    return reinterpret_cast<void *>(LoadLibraryA("OpenCL.dll"));
#else
    // the vendor driver: by name where the linker namespace allows it (an app: public.libraries.txt lists it),
    // else by path (a shell executable)
    for (const char *p : {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so",
                          "/system/lib64/libOpenCL.so", "libOpenCL.so.1"})
        if (void *h = dlopen(p, RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#endif
}

void *symbol(void *lib, const char *name) {
#if defined(_WIN32)
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void load() {
    void *lib = open_library();
    if (!lib) {
        g_error = "no OpenCL library on this device";
        return;
    }
#define GEODE_CL_LOAD(name)                                                                                         \
    g_api.name = reinterpret_cast<decltype(g_api.name)>(symbol(lib, #name));                                        \
    if (!g_api.name) {                                                                                               \
        g_error = std::string("the OpenCL library has no ") + #name;                                                \
        return;                                                                                                      \
    }
    GEODE_CL_FUNCTIONS(GEODE_CL_LOAD)
#undef GEODE_CL_LOAD
    g_loaded = true;
}

std::string device_string(const Api *cl, cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (cl->clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    cl->clGetDeviceInfo(d, what, n, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <typename T>
T device_value(const Api *cl, cl_device_id d, cl_device_info what) {
    T v{};
    cl->clGetDeviceInfo(d, what, sizeof v, &v, nullptr);
    return v;
}

}  // namespace

const Api *api() {
    std::call_once(g_once, load);
    return g_loaded ? &g_api : nullptr;
}

const std::string &load_error() {
    api();
    return g_error;
}

const char *error_name(cl_int code) {
    switch (code) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    default: return "CL error";
    }
}

Device::~Device() {
    if (cl) {
        if (queue) cl->clReleaseCommandQueue(queue);
        if (context) cl->clReleaseContext(context);
    }
}

bool Device::open(std::string &err) {
    cl = api();
    if (!cl) {
        err = load_error();
        return false;
    }
    cl_uint np = 0;
    if (cl->clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        err = "no OpenCL platform";
        return false;
    }
    std::vector<cl_platform_id> ps(np);
    cl->clGetPlatformIDs(np, ps.data(), nullptr);
    for (cl_platform_id p : ps) {
        cl_uint nd = 0;
        if (cl->clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &id, &nd) == CL_SUCCESS && nd > 0) {
            platform = p;
            break;
        }
    }
    if (!platform) {
        err = "no OpenCL GPU";
        return false;
    }
    {
        size_t n = 0;
        cl->clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &n);
        platform_name.assign(n, '\0');
        cl->clGetPlatformInfo(platform, CL_PLATFORM_NAME, n, platform_name.data(), nullptr);
        while (!platform_name.empty() && platform_name.back() == '\0') platform_name.pop_back();
    }
    name = device_string(cl, id, CL_DEVICE_NAME);
    vendor = device_string(cl, id, CL_DEVICE_VENDOR);
    version = device_string(cl, id, CL_DEVICE_VERSION);
    driver = device_string(cl, id, CL_DRIVER_VERSION);
    c_version = device_string(cl, id, CL_DEVICE_OPENCL_C_VERSION);
    extensions = device_string(cl, id, CL_DEVICE_EXTENSIONS);
    compute_units = device_value<cl_uint>(cl, id, CL_DEVICE_MAX_COMPUTE_UNITS);
    max_clock_mhz = device_value<cl_uint>(cl, id, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    global_mem = device_value<cl_ulong>(cl, id, CL_DEVICE_GLOBAL_MEM_SIZE);
    max_alloc = device_value<cl_ulong>(cl, id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    local_mem = device_value<cl_ulong>(cl, id, CL_DEVICE_LOCAL_MEM_SIZE);
    local_is_dedicated = device_value<cl_device_local_mem_type>(cl, id, CL_DEVICE_LOCAL_MEM_TYPE) == CL_LOCAL;
    max_work_group = device_value<size_t>(cl, id, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    images = device_value<cl_bool>(cl, id, CL_DEVICE_IMAGE_SUPPORT) == CL_TRUE;
    image_max_buffer = device_value<size_t>(cl, id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
    fp16 = has("cl_khr_fp16");

    cl_int e = CL_SUCCESS;
    // Qualcomm's performance and priority hints (cl_ext_qcom.h): the highest clocks and queue priority for the
    // context, where the driver has them (GEODE_CL_NO_HINT=1 leaves them out, for measuring)
    constexpr cl_context_properties kPerfHint = 0x40C2, kPerfHigh = 0x40C3, kPrioHint = 0x40C9, kPrioHigh = 0x40CA;
    std::vector<cl_context_properties> props;
    if (!std::getenv("GEODE_CL_NO_HINT")) {
        if (has("cl_qcom_perf_hint")) props.insert(props.end(), {kPerfHint, kPerfHigh});
        if (has("cl_qcom_priority_hint")) props.insert(props.end(), {kPrioHint, kPrioHigh});
    }
    if (!props.empty()) {
        props.push_back(0);
        context = cl->clCreateContext(props.data(), 1, &id, nullptr, nullptr, &e);
    }
    if (!context) context = cl->clCreateContext(nullptr, 1, &id, nullptr, nullptr, &e);
    if (!context) {
        err = std::string("clCreateContext: ") + error_name(e);
        return false;
    }
    queue = cl->clCreateCommandQueue(context, id, CL_QUEUE_PROFILING_ENABLE, &e);
    if (!queue) {
        err = std::string("clCreateCommandQueue: ") + error_name(e);
        return false;
    }
    return true;
}

bool Device::has(const std::string &extension) const {
    // whole words of the space-separated list
    size_t at = 0;
    while ((at = extensions.find(extension, at)) != std::string::npos) {
        const size_t end = at + extension.size();
        if ((at == 0 || extensions[at - 1] == ' ') && (end == extensions.size() || extensions[end] == ' ')) return true;
        at = end;
    }
    return false;
}

Program::~Program() {
    if (!cl_) return;
    for (cl_kernel k : kernels_) cl_->clReleaseKernel(k);
    if (program_) cl_->clReleaseProgram(program_);
}

bool Program::build(const Device &dev, const std::string &source, const std::string &options, std::string &err) {
    cl_ = dev.cl;
    const char *src = source.c_str();
    const size_t len = source.size();
    cl_int e = CL_SUCCESS;
    program_ = cl_->clCreateProgramWithSource(dev.context, 1, &src, &len, &e);
    if (!program_) {
        err = std::string("clCreateProgramWithSource: ") + error_name(e);
        return false;
    }
    return finish_build(dev, options, err);
}

bool Program::build_cached(const Device &dev, const std::string &source, const std::string &options,
                           const std::string &cache_dir, std::string &err) {
    if (cache_dir.empty()) return build(dev, source, options, err);
    // FNV-1a 64 over everything the compiled code depends on
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const std::string &s) {
        for (unsigned char ch : s) h = (h ^ ch) * 1099511628211ull;
        h = (h ^ 0xffu) * 1099511628211ull;   // a separator, so "ab"+"c" differs from "a"+"bc"
    };
    for (const std::string *s : {&source, &options, &dev.name, &dev.vendor, &dev.version, &dev.driver, &dev.c_version})
        mix(*s);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
    const std::string path = cache_dir + "/clprog-" + hex + ".bin";
    if (std::FILE *f = std::fopen(path.c_str(), "rb")) {
        std::vector<uint8_t> bin;
        uint8_t buf[1 << 16];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) bin.insert(bin.end(), buf, buf + n);
        std::fclose(f);
        std::string e;
        if (!bin.empty() && build_binary(dev, bin, e)) {
            std::error_code ec;   // used: its time says so (the clean-up below keeps what is still used)
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
            return true;
        }
        if (program_) {   // a stale or broken file: compile instead (and replace it)
            cl_->clReleaseProgram(program_);
            program_ = nullptr;
        }
    }
    if (!build(dev, source, options, err)) return false;
    const std::vector<uint8_t> bin = binary();
    if (!bin.empty()) {
        const std::string tmp = path + ".tmp";
        if (std::FILE *f = std::fopen(tmp.c_str(), "wb")) {
            const bool ok = std::fwrite(bin.data(), 1, bin.size(), f) == bin.size();
            if (std::fclose(f) != 0 || !ok || std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
        }
        // a new compile means new code or a new driver: the files no open has used for a week go (every program
        // in use is touched when it loads, so an older app's or driver's set is what is left behind)
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto now = fs::file_time_type::clock::now();
        for (fs::directory_iterator it(cache_dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.rfind("clprog-", 0) != 0 || it->path() == fs::path(path)) continue;
            std::error_code te;
            const auto when = fs::last_write_time(it->path(), te);
            if (!te && now - when > std::chrono::hours(24 * 7)) fs::remove(it->path(), te);
        }
    }
    return true;
}

bool Program::build_binary(const Device &dev, const std::vector<uint8_t> &binary, std::string &err) {
    cl_ = dev.cl;
    const unsigned char *p = binary.data();
    const size_t len = binary.size();
    cl_int status = CL_SUCCESS, e = CL_SUCCESS;
    program_ = cl_->clCreateProgramWithBinary(dev.context, 1, &dev.id, &len, &p, &status, &e);
    if (!program_) {
        err = std::string("clCreateProgramWithBinary: ") + error_name(e) + " / " + error_name(status);
        return false;
    }
    return finish_build(dev, "", err);
}

bool Program::finish_build(const Device &dev, const std::string &options, std::string &err) {
    const cl_int e = cl_->clBuildProgram(program_, 1, &dev.id, options.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t n = 0;
        cl_->clGetProgramBuildInfo(program_, dev.id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        cl_->clGetProgramBuildInfo(program_, dev.id, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        err = std::string("clBuildProgram: ") + error_name(e) + "\n" + log;
        return false;
    }
    return true;
}

cl_kernel Program::kernel(const char *name, std::string &err) {
    cl_int e = CL_SUCCESS;
    cl_kernel k = cl_->clCreateKernel(program_, name, &e);
    if (!k) {
        err = std::string("clCreateKernel ") + name + ": " + error_name(e);
        return nullptr;
    }
    kernels_.push_back(k);
    return k;
}

std::vector<uint8_t> Program::binary() const {
    size_t n = 0;
    if (!program_ || cl_->clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES, sizeof n, &n, nullptr) != CL_SUCCESS ||
        n == 0)
        return {};
    std::vector<uint8_t> out(n);
    unsigned char *p = out.data();
    if (cl_->clGetProgramInfo(program_, CL_PROGRAM_BINARIES, sizeof p, &p, nullptr) != CL_SUCCESS) return {};
    return out;
}

double event_seconds(const Api *cl, cl_event ev) {
    cl_ulong t0 = 0, t1 = 0;
    cl->clWaitForEvents(1, &ev);
    cl->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, nullptr);
    cl->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, nullptr);
    cl->clReleaseEvent(ev);
    return static_cast<double>(t1 - t0) * 1e-9;
}

}  // namespace geode::cl
