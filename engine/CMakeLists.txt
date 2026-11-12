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
    src/gemma.cpp
    src/geode.cpp
    src/json.cpp
    src/kernels.cpp
    src/model.cpp
    src/pack.cpp
    src/vision.cpp
    src/media.cpp
    src/audio.cpp
    src/drafter.cpp
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
    src/hvx_llm.cpp
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

add_library(geode_yue2_lora STATIC src/yue2_lora.cpp src/yue2_safetensors.cpp)
target_link_libraries(geode_yue2_lora PUBLIC geode_core)
target_include_directories(geode_yue2_lora PUBLIC src)
set_target_properties(geode_yue2_lora PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(geode_yue2_lora PRIVATE -O3 -Wall -Wextra -ffp-contract=off)
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
    # YuE2 songs (docs/music-plan.md): loaded only by the :music process
    add_library(geode_music SHARED android/music_jni.cpp src/yue2_song.cpp src/yue2_master.cpp src/yue2_ar.cpp
                src/yue2_gen.cpp src/yue2_nar.cpp src/yue2_nar_gpu.cpp src/yue2.cpp src/yue2_pack.cpp src/yue2_vae.cpp
                src/yue2_q4.cpp src/yue2_q8gemm.cpp src/yue2_state.cpp src/yue2_edit.cpp src/yue2_view.cpp)
    target_link_libraries(geode_music PRIVATE geode_yue2_lora geode_core log android ${CMAKE_DL_LIBS})
    target_include_directories(geode_music PRIVATE third_party/OpenCL-Headers/include)   # (yue2_nar_gpu.cpp)
    target_compile_options(geode_music PRIVATE -O3 -march=armv8.2-a+dotprod+fp16)
    target_link_options(geode_music PRIVATE "-Wl,-z,max-page-size=16384")
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
    add_executable(geode-audio tools/audio.cpp)
    target_link_libraries(geode-audio PRIVATE geode_core)
    add_executable(geode-state tools/state_test.cpp)
    target_link_libraries(geode-state PRIVATE geode_core)
    add_executable(geode-draft tools/draft_test.cpp)
    target_link_libraries(geode-draft PRIVATE geode_core)
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
    # YuE2 (music): the phase-2 reference engine, kept out of geode_core while it is written; its parity tools
    add_executable(geode-yue2lora tools/yue2_lora.cpp src/yue2_pack.cpp src/yue2_ar.cpp src/yue2.cpp)
    target_link_libraries(geode-yue2lora PRIVATE geode_yue2_lora geode_core)
    add_executable(geode-yue2 tools/yue2.cpp src/yue2.cpp src/yue2_pack.cpp)
    target_link_libraries(geode-yue2 PRIVATE geode_core)
    target_compile_options(geode-yue2 PRIVATE -O3)
    add_executable(geode-yue2vae tools/yue2_vae.cpp src/yue2_vae.cpp src/yue2_pack.cpp)
    target_link_libraries(geode-yue2vae PRIVATE geode_core ${CMAKE_DL_LIBS})
    target_compile_options(geode-yue2vae PRIVATE -O3)
    target_include_directories(geode-yue2vae PRIVATE third_party/OpenCL-Headers/include)
    # phase 3: the fast AR (Q8 on the SDOT kernels) and generation - its gates, speed and a real song's tokens
    add_executable(geode-yue2gen tools/yue2_gen.cpp src/yue2_ar.cpp src/yue2_gen.cpp src/yue2_nar.cpp src/yue2_nar_gpu.cpp src/yue2.cpp
                                 src/yue2_pack.cpp src/yue2_q4.cpp src/yue2_q8gemm.cpp)
    add_executable(geode-yue2song tools/yue2_song.cpp src/yue2_song.cpp src/yue2_master.cpp src/yue2_ar.cpp
                                  src/yue2_gen.cpp src/yue2_nar.cpp src/yue2_nar_gpu.cpp src/yue2.cpp src/yue2_pack.cpp src/yue2_vae.cpp
                                  src/yue2_q4.cpp src/yue2_q8gemm.cpp src/yue2_state.cpp src/yue2_edit.cpp src/yue2_view.cpp)
    target_link_libraries(geode-yue2song PRIVATE geode_yue2_lora geode_core ${CMAKE_DL_LIBS})
    # a song's editable state (yue2_state.hpp): the file's round trip, and a dump of a song's
    add_executable(geode-yue2state tools/yue2_state.cpp src/yue2_song.cpp src/yue2_master.cpp src/yue2_ar.cpp
                                   src/yue2_gen.cpp src/yue2_nar.cpp src/yue2_nar_gpu.cpp src/yue2.cpp src/yue2_pack.cpp src/yue2_vae.cpp
                                   src/yue2_q4.cpp src/yue2_q8gemm.cpp src/yue2_state.cpp src/yue2_edit.cpp src/yue2_view.cpp)
    target_link_libraries(geode-yue2state PRIVATE geode_yue2_lora geode_core ${CMAKE_DL_LIBS})
    target_compile_options(geode-yue2state PRIVATE -O3)
    target_include_directories(geode-yue2state PRIVATE third_party/OpenCL-Headers/include)
    target_compile_options(geode-yue2song PRIVATE -O3)
    target_link_libraries(geode-yue2gen PRIVATE geode_yue2_lora geode_core ${CMAKE_DL_LIBS})
    target_compile_options(geode-yue2gen PRIVATE -O3)
    target_include_directories(geode-yue2gen PRIVATE third_party/OpenCL-Headers/include)
    target_include_directories(geode-yue2song PRIVATE third_party/OpenCL-Headers/include)
    # the NAR's products on the GPU (docs/music-plan.md M6): a kernel from bench/gpu against the exact product
    add_executable(geode-yue2gpu tools/yue2_gpu.cpp src/yue2_pack.cpp src/yue2_nar_gpu.cpp)
    target_link_libraries(geode-yue2gpu PRIVATE geode_core ${CMAKE_DL_LIBS})
    target_include_directories(geode-yue2gpu PRIVATE third_party/OpenCL-Headers/include)
    if(ANDROID)
        foreach(t geode-yue2 geode-yue2vae geode-yue2gen geode-yue2song geode-yue2gpu geode-yue2state)
            target_compile_options(${t} PRIVATE -march=armv8.2-a+dotprod+fp16)
        endforeach()
    endif()
endif()
