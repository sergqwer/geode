// npu/host/probe.h - the NPU checks shared by the adb probe and the app's debug hook (probe.c).
#pragma once

#include "remote.h"

typedef void (*npu_say_fn)(void *ctx, const char *line);

// The unsigned PD, then Geode's module opened on the cDSP; 0 when either is refused (the reason said).
remote_handle64 npu_open(npu_say_fn out, void *ctx);
// FastRPC and dspqueue round trips, HMX GEMM checks and the model's shapes (reps each); the number of failures.
int npu_bench(remote_handle64 h, int reps, npu_say_fn out, void *ctx);
// geode_npu_gemm on an m x k x n product of our own buffers (rpcmem; the weights mapped once), reps times: its time
// and a sample of its results against a float sum; 0 when right.
int npu_gemm_check(remote_handle64 h, int m, int k, int n, int reps, npu_say_fn out, void *ctx);
// the engine's interface (npu_api.h) on a rows x cols matrix in the pack's q4 layout times n rows; 0 when right
int npu_api_check(int rows, int cols, int n, npu_say_fn out, void *ctx);
