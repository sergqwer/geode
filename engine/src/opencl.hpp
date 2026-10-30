// opencl.hpp - the OpenCL runtime the GPU path runs on (PLAN.md, P5). libOpenCL is loaded at run time, from the
// phone's vendor driver (Qualcomm's for the Adreno), so the engine still runs, on the CPU, wherever there is none.
//
// Why OpenCL and not Vulkan on the Adreno 618: its Vulkan driver (1.1, 2021) has no 8/16-bit storage and its shader
// compiler made FP16 slower than FP32 in vk_bench; Qualcomm's OpenCL is their tuned compute stack (FP16, subgroups,
// the texture path, memory shared with the CPU without copies).
#pragma once

#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace geode::cl {

// clX(...) for every function the engine uses: a pointer filled from the loaded library
#define GEODE_CL_FUNCTIONS(X)                                                                                         \
    X(clGetPlatformIDs) X(clGetPlatformInfo) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext)                 \
    X(clReleaseContext) X(clCreateCommandQueue) X(clReleaseCommandQueue) X(clCreateProgramWithSource)                 \
    X(clCreateProgramWithBinary) X(clBuildProgram) X(clGetProgramBuildInfo) X(clGetProgramInfo) X(clReleaseProgram)  \
    X(clCreateKernel) X(clReleaseKernel) X(clSetKernelArg) X(clGetKernelWorkGroupInfo) X(clEnqueueNDRangeKernel)      \
    X(clFinish) X(clFlush) X(clCreateBuffer) X(clReleaseMemObject) X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer)    \
    X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clEnqueueCopyBuffer) X(clCreateImage) X(clGetEventProfilingInfo) X(clWaitForEvents)  \
    X(clGetEventInfo)                                                                                                \
    X(clReleaseEvent) X(clCreateSubBuffer) X(clGetExtensionFunctionAddressForPlatform)

struct Api {
#define GEODE_CL_POINTER(name) decltype(&::name) name = nullptr;
    GEODE_CL_FUNCTIONS(GEODE_CL_POINTER)
#undef GEODE_CL_POINTER
};

// The loaded library, or nullptr (then load_error() says why). Loaded once, kept for the process.
const Api *api();
const std::string &load_error();

// An OpenCL error code by name, for messages.
const char *error_name(cl_int code);

// The GPU: the first GPU device of the first platform that has one, a context and an in-order queue that profiles.
class Device {
public:
    Device() = default;
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device();

    bool open(std::string &err);
    bool has(const std::string &extension) const;

    const Api *cl = nullptr;
    cl_platform_id platform = nullptr;
    cl_device_id id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string name, vendor, version, driver, c_version, extensions, platform_name;
    cl_uint compute_units = 0, max_clock_mhz = 0;
    cl_ulong global_mem = 0, max_alloc = 0, local_mem = 0;
    size_t max_work_group = 0, image_max_buffer = 0;
    bool fp16 = false, images = false, local_is_dedicated = false;
};

// A program built for the device, its kernels by name.
class Program {
public:
    Program() = default;
    Program(const Program &) = delete;
    Program &operator=(const Program &) = delete;
    ~Program();

    bool build(const Device &dev, const std::string &source, const std::string &options, std::string &err);
    // build(), with the driver's compiled program kept in cache_dir (clprog-<hash>.bin; the hash covers the source,
    // the options, the device and its driver): a later build loads it instead of compiling again - the same machine
    // code, ~0.3-0.6 s less each on the Adreno 830. A cache that cannot be read or written only costs the compile.
    bool build_cached(const Device &dev, const std::string &source, const std::string &options,
                      const std::string &cache_dir, std::string &err);
    // From a driver binary (one the driver compiled, or one with hand-written machine code put into it).
    bool build_binary(const Device &dev, const std::vector<uint8_t> &binary, std::string &err);
    cl_kernel kernel(const char *name, std::string &err);   // owned by the program
    std::vector<uint8_t> binary() const;   // the driver's compiled program (to disassemble), empty on failure

private:
    bool finish_build(const Device &dev, const std::string &options, std::string &err);

    const Api *cl_ = nullptr;
    cl_program program_ = nullptr;
    std::vector<cl_kernel> kernels_;
};

// Seconds the kernel ran, from the event's profiling counters (the queue profiles); the event is released.
double event_seconds(const Api *cl, cl_event ev);

}  // namespace geode::cl
