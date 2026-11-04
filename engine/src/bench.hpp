// bench.hpp - the device benchmarks behind the app's Benchmark tab (geode_bench in geode.h). Each returns its results
// as one JSON object; the app keeps them and derives the speed a model can reach on this device from them.
#pragma once

#include <string>

namespace geode::bench {

// SoC, CPU clusters (core type, count, max clock), CPU features (fp16, dotprod, i8mm, bf16, SVE...), RAM size.
std::string system_info();

// DRAM: read / write / copy bandwidth on one big core, all big cores and all cores; random-access latency.
std::string memory();

// Flash: random reads (1 MiB and the model's expert block, 1-4 threads, O_DIRECT) and a sequential read. dir: a
// model directory (its expert files are read, nothing is written) or any writable directory (a temporary 1 GiB
// file is written there, its write speed measured, then deleted).
std::string flash(const std::string &dir);

// CPU arithmetic: peak FMA / dot-product rates per core type and data type, and the engine's own kernels in every
// weight format (one token, one token on all big cores, four tokens at once).
std::string cpu();

// GPU (Vulkan): device, the data types it supports, FP32 / FP16 / INT32 throughput and memory bandwidth.
std::string gpu();

// GPU (OpenCL, cl_bench.cpp): the driver, launch latency, FP32 / FP16 throughput, reading memory every way the
// engine could hold weights (GPU buffer, the CPU's memory without a copy, shared allocation, textures, local memory).
std::string opencl();

// Geode's own GPU code here (gpu_check.cpp): the path the engine takes and why, each GPU product on the model's shapes
// (random weights) against the CPU's and timed, the generating programs built. step_file: each step's name before it
// runs (the app reads it when the driver takes the process down).
std::string gpu_check(const std::string &step_file);

// The NPU (npu_check.cpp, npu/host/report.c): the Hexagon version and what it supports, whether Geode's module opens
// in an unsigned PD, FastRPC / dspqueue round trips, HMX products checked and timed on the model's shapes, the DDR
// read speed and the decode products from tile8. step_file as gpu_check's.
std::string npu_check(const std::string &step_file);

// The units that can run a model here, for the run-mode choice (npu_check.cpp): {"cpu", "gpu" (an Adreno 6xx+),
// "gpu_name", "adreno", "npu" (Geode's module opens, Hexagon v73+), "hexagon", and the reasons where one cannot}
std::string units();

}  // namespace geode::bench
