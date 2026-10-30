# The Geode engine. Two products from one tree:
#   Android (the NDK, arm64-v8a): libgeode.so with the JNI bindings, loaded by the app;
#   the PC: geode_core + geode-cli, for parity tests against the Python reference.
cmake_minimum_required(VERSION 3.22)
project(geode LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(geode_core STATIC
    src/bench.cpp
    src/experts.cpp
    src/geode.cpp
    src/json.cpp
    src/kernels.cpp
    src/model.cpp
    src/pack.cpp
    src/vision.cpp
    src/platform.cpp
    src/sampler.cpp
    src/state.cpp
    src/threads.cpp
    src/vk_bench.cpp
    src/opencl.cpp
    src/cl_bench.cpp
    src/gpu_check.cpp
    src/npu_check.cpp
    src/gpu.cpp
    src/gpu8.cpp
)
find_package(Threads REQUIRED)
# libOpenCL is opened at run time (opencl.cpp): only the Khronos headers are needed to build
target_link_libraries(geode_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(geode_core PUBLIC include PRIVATE src third_party/OpenCL-Headers/include)

if(MSVC)
    target_compile_options(geode_core PRIVATE /W4)
else()
    target_compile_options(geode_core PRIVATE -Wall -Wextra -O3)
endif()

# the parity tool also runs on the phone (adb push + run): checks the ARM build against the same reference
# (the tools are not in the public copy of the repository: there the option is off)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/cli.cpp)
    option(GEODE_TOOLS "build geode-parity / geode-cli" ON)
else()
    option(GEODE_TOOLS "build geode-parity / geode-cli" OFF)
endif()

if(ANDROID)
    if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "Geode builds for arm64-v8a only (the kernels need ARMv8.2 dotprod), not ${ANDROID_ABI}")
    endif()
    target_compile_options(geode_core PRIVATE -march=armv8.2-a+dotprod+fp16)
    set_target_properties(geode_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # the GPU benchmark (libvulkan is on every Android 7+ device)
    target_link_libraries(geode_core PUBLIC vulkan)
    add_library(geode SHARED android/jni.cpp)
    target_link_libraries(geode PRIVATE geode_core log android)
    # 16 KiB pages: required for apps targeting newer Android on devices with 16 KiB page sizes
    target_link_options(geode PRIVATE "-Wl,-z,max-page-size=16384")
endif()

if(GEODE_TOOLS)
    add_executable(geode-cli tools/cli.cpp)
    target_link_libraries(geode-cli PRIVATE geode_core)
    add_executable(geode-parity tools/parity.cpp)
    target_link_libraries(geode-parity PRIVATE geode_core)
    target_compile_options(geode-parity PRIVATE -O2)
    add_executable(geode-kl tools/kl.cpp)
    target_link_libraries(geode-kl PRIVATE geode_core)
    add_executable(geode-verify tools/verify.cpp)
    target_link_libraries(geode-verify PRIVATE geode_core)
    add_executable(geode-gpu tools/gpu_test.cpp)
    target_link_libraries(geode-gpu PRIVATE geode_core)
    target_include_directories(geode-gpu PRIVATE third_party/OpenCL-Headers/include)
    add_executable(geode-clprobe tools/clprobe.cpp)
    target_link_libraries(geode-clprobe PRIVATE geode_core)
    target_include_directories(geode-clprobe PRIVATE third_party/OpenCL-Headers/include)
    add_executable(geode-mtp tools/mtp.cpp)
    target_link_libraries(geode-mtp PRIVATE geode_core)
    add_executable(geode-bench tools/bench.cpp)
    target_link_libraries(geode-bench PRIVATE geode_core)
    add_executable(geode-i8gemm tools/i8gemm.cpp)
    target_link_libraries(geode-i8gemm PRIVATE geode_core)
    target_include_directories(geode-i8gemm PRIVATE third_party/OpenCL-Headers/include)
    add_executable(geode-gpugemm tools/gpu_gemm_test.cpp)
    target_link_libraries(geode-gpugemm PRIVATE geode_core)
    target_include_directories(geode-gpugemm PRIVATE third_party/OpenCL-Headers/include)
    if(ANDROID)
        add_executable(geode-vkprobe tools/vk_probe.cpp)
        target_link_libraries(geode-vkprobe PRIVATE vulkan)
    endif()
    add_executable(geode-vision tools/vision.cpp)
    target_link_libraries(geode-vision PRIVATE geode_core)
    add_executable(geode-state tools/state_test.cpp)
    target_link_libraries(geode-state PRIVATE geode_core)
    add_executable(geode-spec tools/spec_test.cpp)
    target_link_libraries(geode-spec PRIVATE geode_core)
    add_executable(geode-threads tools/threads_test.cpp)
    target_link_libraries(geode-threads PRIVATE geode_core)
    add_executable(geode-routeprior tools/route_prior_test.cpp)
    target_link_libraries(geode-routeprior PRIVATE geode_core)
    add_executable(geode-experts tools/experts_test.cpp)
    target_link_libraries(geode-experts PRIVATE geode_core)
    add_executable(geode-reserve tools/reserve_test.cpp)
    target_link_libraries(geode-reserve PRIVATE geode_core)
    if(WIN32)
        target_link_libraries(geode-reserve PRIVATE psapi)
    endif()
    add_executable(geode-kernels tools/kernel_test.cpp)
    target_link_libraries(geode-kernels PRIVATE geode_core)
    target_compile_options(geode-kernels PRIVATE -O2)
    add_executable(geode-npu tools/npu_test.cpp)
    target_link_libraries(geode-npu PRIVATE geode_core)
endif()
