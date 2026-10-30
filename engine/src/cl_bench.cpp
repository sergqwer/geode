// cl_bench.cpp - the GPU through OpenCL (geode_bench "opencl"): what the driver offers, how long a kernel launch
// takes, FP32 / FP16 arithmetic, and reading memory every way the engine could hold its weights: a GPU buffer, the
// CPU's own memory without a copy (where the expert cache lives), memory the driver allocates for both, the texture
// path (image1d_buffer), and the on-chip local memory a work group shares (GMEM on the Adreno).
#include "bench.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "bench_json.hpp"
#include "opencl.hpp"
#include "platform.hpp"

namespace geode::bench {

namespace {

const char *kSource = R"CL(
#ifdef HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void empty_kernel(__global float *out) {
    if (get_global_id(0) == 0x7fffffff) out[0] = 0.0f;
}

// eight independent FMA chains of four lanes per work item
__kernel void peak_f32(__global float *out, int iters) {
    const float s = (float)get_global_id(0) * 1e-9f;
    float4 a0 = s, a1 = s + 1.0f, a2 = s + 2.0f, a3 = s + 3.0f, a4 = s + 4.0f, a5 = s + 5.0f, a6 = s + 6.0f,
           a7 = s + 7.0f;
    const float4 m = (float4)(0.9999f), k = (float4)(1e-6f);
    for (int i = 0; i < iters; ++i) {
        a0 = mad(a0, m, k); a1 = mad(a1, m, k); a2 = mad(a2, m, k); a3 = mad(a3, m, k);
        a4 = mad(a4, m, k); a5 = mad(a5, m, k); a6 = mad(a6, m, k); a7 = mad(a7, m, k);
    }
    const float4 t = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[get_global_id(0)] = t.x + t.y + t.z + t.w;
}

#ifdef HALF
// the same with eight lanes of FP16
__kernel void peak_f16(__global float *out, int iters) {
    const half s = (half)((float)(get_global_id(0) & 255) * 1e-3f);
    half8 a0 = s, a1 = s + (half)1, a2 = s + (half)2, a3 = s + (half)3, a4 = s + (half)4, a5 = s + (half)5,
          a6 = s + (half)6, a7 = s + (half)7;
    const half8 m = (half8)(0.999h), k = (half8)(0.001h);
    for (int i = 0; i < iters; ++i) {
        a0 = mad(a0, m, k); a1 = mad(a1, m, k); a2 = mad(a2, m, k); a3 = mad(a3, m, k);
        a4 = mad(a4, m, k); a5 = mad(a5, m, k); a6 = mad(a6, m, k); a7 = mad(a7, m, k);
    }
    const half8 t = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[get_global_id(0)] = (float)(t.s0 + t.s1 + t.s2 + t.s3 + t.s4 + t.s5 + t.s6 + t.s7);
}
#endif

// n float4 per work item; neighbouring work items read neighbouring float4s
__kernel void read_buf(__global const float4 *src, __global float *out, int n) {
    const int g = get_global_size(0), i = get_global_id(0);
    float4 s = 0.0f;
    for (int j = 0; j < n; ++j) s += src[i + j * g];
    out[i] = s.x + s.y + s.z + s.w;
}

// the same through the texture path
__kernel void read_img(__read_only image1d_buffer_t src, __global float *out, int n) {
    const int g = get_global_size(0), i = get_global_id(0);
    float4 s = 0.0f;
    for (int j = 0; j < n; ++j) s += read_imagef(src, i + j * g);
    out[i] = s.x + s.y + s.z + s.w;
}

// a 16 KB tile in the work group's local memory, read back again and again
__kernel void read_local(__global float *out, int iters) {
    __local float4 tile[1024];
    const int l = get_local_id(0), ls = get_local_size(0);
    for (int j = l; j < 1024; j += ls) tile[j] = (float4)((float)j);
    barrier(CLK_LOCAL_MEM_FENCE);
    // four independent sums (one would measure the add's latency, not the memory)
    float4 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int i = 0; i < iters; ++i) {
        const int b = l + i * 33;
        s0 += tile[b & 1023];
        s1 += tile[(b + 256) & 1023];
        s2 += tile[(b + 512) & 1023];
        s3 += tile[(b + 768) & 1023];
    }
    const float4 s = s0 + s1 + s2 + s3;
    out[get_global_id(0)] = s.x + s.y + s.z + s.w;
}

// every work item of the group reads the same address (a GEMM's shared tile of activations): bytes as the work items
// see them
__kernel void read_local_bcast(__global float *out, int iters) {
    __local float4 tile[1024];
    const int l = get_local_id(0), ls = get_local_size(0);
    for (int j = l; j < 1024; j += ls) tile[j] = (float4)((float)j);
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int i = 0; i < iters; ++i) {
        const int b = (i * 4) & 1023;
        s0 += tile[b];
        s1 += tile[b + 1];
        s2 += tile[b + 2];
        s3 += tile[b + 3];
    }
    const float4 s = s0 + s1 + s2 + s3;
    out[get_global_id(0)] = s.x + s.y + s.z + s.w + (float)l;
}
)CL";

struct Run {
    const cl::Api *cl;
    cl_command_queue q;

    // one launch, its kernel time in seconds (0 on failure)
    double launch(cl_kernel k, size_t global, size_t local) const {
        cl_event ev = nullptr;
        if (cl->clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, local ? &local : nullptr, 0, nullptr, &ev) !=
            CL_SUCCESS)
            return 0.0;
        return cl::event_seconds(cl, ev);
    }
    // the best of a few launches after a warm-up (the GPU clock ramps up)
    double best(cl_kernel k, size_t global, size_t local, int reps = 5) const {
        const double w0 = now_seconds();
        while (now_seconds() - w0 < 0.3)
            if (launch(k, global, local) <= 0.0) return 0.0;
        double b = 1e30;
        for (int r = 0; r < reps; ++r) b = std::min(b, launch(k, global, local));
        return b;
    }
};

template <typename T>
void arg(const cl::Api *cl, cl_kernel k, cl_uint i, const T &v) {
    cl->clSetKernelArg(k, i, sizeof v, &v);
}

}  // namespace

std::string opencl() {
    Json j;
    j.begin_object();
    cl::Device dev;
    std::string err;
    if (!dev.open(err)) {
        j.boolean("available", false).str("reason", err).end_object();
        return j.take();
    }
    const cl::Api *cl = dev.cl;
    j.boolean("available", true);
    j.str("platform", dev.platform_name).str("device", dev.name).str("vendor", dev.vendor);
    j.str("version", dev.version).str("driver", dev.driver).str("opencl_c", dev.c_version);
    j.num("compute_units", dev.compute_units).num("max_clock_mhz", dev.max_clock_mhz);
    j.num("global_mem", static_cast<double>(dev.global_mem)).num("max_alloc", static_cast<double>(dev.max_alloc));
    j.num("local_mem", static_cast<double>(dev.local_mem)).boolean("local_dedicated", dev.local_is_dedicated);
    j.num("max_work_group", static_cast<double>(dev.max_work_group));
    j.boolean("images", dev.images).num("image_max_buffer", static_cast<double>(dev.image_max_buffer));
    j.boolean("fp16", dev.fp16);
    j.str("extensions", dev.extensions);

    cl::Program prog;
    if (!prog.build(dev, kSource, std::string("-cl-mad-enable -cl-fast-relaxed-math") + (dev.fp16 ? " -DHALF" : ""),
                    err)) {
        j.str("error", err).end_object();
        return j.take();
    }
    const Run run{cl, dev.queue};
    const size_t local = std::min<size_t>(128, dev.max_work_group);
    const size_t groups = std::max<cl_uint>(1, dev.compute_units) * 16;
    const size_t items = groups * local;
    cl_int e = CL_SUCCESS;
    cl_mem out = cl->clCreateBuffer(dev.context, CL_MEM_WRITE_ONLY, (1 << 20) * sizeof(float), nullptr, &e);

    // a launch: queued and waited for one at a time, and back to back
    if (cl_kernel k = prog.kernel("empty_kernel", err)) {
        arg(cl, k, 0, out);
        size_t g = local;
        const double t0 = now_seconds();
        for (int i = 0; i < 200; ++i) {
            cl->clEnqueueNDRangeKernel(dev.queue, k, 1, nullptr, &g, &local, 0, nullptr, nullptr);
            cl->clFinish(dev.queue);
        }
        const double t1 = now_seconds();
        for (int i = 0; i < 200; ++i) cl->clEnqueueNDRangeKernel(dev.queue, k, 1, nullptr, &g, &local, 0, nullptr, nullptr);
        cl->clFinish(dev.queue);
        const double t2 = now_seconds();
        j.num("launch_wait_us", (t1 - t0) / 200 * 1e6).num("launch_queued_us", (t2 - t1) / 200 * 1e6);
    }

    // arithmetic: iterations grown until a launch takes ~40 ms
    auto peak = [&](const char *name, double flops_per_iter) -> double {
        cl_kernel k = prog.kernel(name, err);
        if (!k) return 0.0;
        arg(cl, k, 0, out);
        int iters = 256;
        double t = 0.0;
        for (;;) {
            arg(cl, k, 1, iters);
            t = run.launch(k, items, local);
            if (t <= 0.0 || t > 0.04 || iters > (1 << 22)) break;
            iters *= 2;
        }
        arg(cl, k, 1, iters);
        t = run.best(k, items, local);
        return t > 0.0 ? static_cast<double>(items) * iters * flops_per_iter / t / 1e9 : 0.0;
    };
    j.num("fp32_gflops", peak("peak_f32", 8 * 4 * 2));
    if (dev.fp16) j.num("fp16_gflops", peak("peak_f16", 8 * 8 * 2));

    // reading memory: 64 MiB every way
    const size_t bytes = 64u << 20, n4 = bytes / 16;
    const size_t read_items = 64 * 1024;
    const int per_item = static_cast<int>(n4 / read_items);
    j.begin_object("read_gbs");
    cl_kernel kb = prog.kernel("read_buf", err);
    auto read_rate = [&](cl_mem m) -> double {
        arg(cl, kb, 0, m);
        arg(cl, kb, 1, out);
        arg(cl, kb, 2, per_item);
        const double t = run.best(kb, read_items, local);
        return t > 0.0 ? static_cast<double>(bytes) / t / 1e9 : 0.0;
    };
    std::vector<float> host(n4 * 4, 1.0f);
    // a GPU buffer, filled by a copy
    cl_mem dbuf = cl->clCreateBuffer(dev.context, CL_MEM_READ_ONLY, bytes, nullptr, &e);
    if (dbuf) {
        cl->clEnqueueWriteBuffer(dev.queue, dbuf, CL_TRUE, 0, bytes, host.data(), 0, nullptr, nullptr);
        j.num("gpu_buffer", read_rate(dbuf));
    }
    // the CPU's memory as it is (page-aligned, like the expert cache): no copy if the driver can map it
    AlignedBuffer cpu_mem = aligned_bytes(bytes);
    double create_ms = 0.0;
    if (cpu_mem) {
        std::memcpy(cpu_mem.get(), host.data(), bytes);
        const double c0 = now_seconds();
        cl_mem hbuf = cl->clCreateBuffer(dev.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bytes, cpu_mem.get(), &e);
        create_ms = (now_seconds() - c0) * 1e3;
        if (hbuf) {
            j.num("cpu_memory", read_rate(hbuf));
            cl->clReleaseMemObject(hbuf);
        } else {
            j.str("cpu_memory_error", cl::error_name(e));
        }
    }
    // memory the driver allocates where both see it, filled through a mapping
    cl_mem abuf = cl->clCreateBuffer(dev.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &e);
    if (abuf) {
        void *p = cl->clEnqueueMapBuffer(dev.queue, abuf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &e);
        if (p) {
            std::memcpy(p, host.data(), bytes);
            cl->clEnqueueUnmapMemObject(dev.queue, abuf, p, 0, nullptr, nullptr);
            j.num("shared_alloc", read_rate(abuf));
        }
        cl->clReleaseMemObject(abuf);
    }
    // the texture path over the GPU buffer
    if (dev.images && dbuf && dev.image_max_buffer >= n4) {
        cl_image_format fmt{CL_RGBA, CL_FLOAT};
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        desc.image_width = n4;
        desc.buffer = dbuf;
        cl_mem img = cl->clCreateImage(dev.context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        cl_kernel ki = img ? prog.kernel("read_img", err) : nullptr;
        if (ki) {
            arg(cl, ki, 0, img);
            arg(cl, ki, 1, out);
            arg(cl, ki, 2, per_item);
            const double t = run.best(ki, read_items, local);
            if (t > 0.0) j.num("texture", static_cast<double>(bytes) / t / 1e9);
        } else {
            j.str("texture_error", cl::error_name(e));
        }
        if (img) cl->clReleaseMemObject(img);
    }
    j.end_object();
    j.num("cpu_memory_wrap_ms", create_ms);
    if (dbuf) cl->clReleaseMemObject(dbuf);

    // the work group's local memory (on-chip)
    if (cl_kernel kl = prog.kernel("read_local", err)) {
        arg(cl, kl, 0, out);
        int iters = 1024;
        double t = 0.0;
        for (;;) {
            arg(cl, kl, 1, iters);
            t = run.launch(kl, items, local);
            if (t <= 0.0 || t > 0.04 || iters > (1 << 22)) break;
            iters *= 2;
        }
        arg(cl, kl, 1, iters);
        t = run.best(kl, items, local);
        if (t > 0.0) j.num("local_gbs", static_cast<double>(items) * iters * 4 * 16 / t / 1e9);
    }
    if (cl_kernel kl = prog.kernel("read_local_bcast", err)) {
        arg(cl, kl, 0, out);
        int iters = 1024;
        double t = 0.0;
        for (;;) {
            arg(cl, kl, 1, iters);
            t = run.launch(kl, items, local);
            if (t <= 0.0 || t > 0.04 || iters > (1 << 22)) break;
            iters *= 2;
        }
        arg(cl, kl, 1, iters);
        t = run.best(kl, items, local);
        if (t > 0.0) j.num("local_broadcast_gbs", static_cast<double>(items) * iters * 4 * 16 / t / 1e9);
    }
    if (out) cl->clReleaseMemObject(out);
    j.end_object();
    return j.take();
}

}  // namespace geode::bench
