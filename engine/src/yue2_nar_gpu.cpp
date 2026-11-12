// yue2_nar_gpu.cpp - the NAR's products on the GPU (yue2_nar_gpu.hpp). The kernel is bench/gpu/nar_gemm_f16_img.cl's
// fastest exact form on the Adreno 618 (R=2 T=2 -DACC32 -DWIMG -DX32): an item 8 rows x 8 tokens, the weights and
// the activations through the texture path, no local memory, fp32 sums. The activations reach it packed by a
// transpose on the GPU: the CPU only copies its rows in (packing them on the CPU cost ~12% of a product: scattered
// stores into write-combined memory; reading the rows as they are, ~40% - eight rows an item thrash the texture cache).
#include "yue2_nar_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <vector>

#include "opencl.hpp"
#include "platform.hpp"

namespace geode::yue2 {

namespace {

// W: a layer's stretch as RGBA32UI texels (16 bytes: four rows' two columns of a column pair), the matrix from texel
// woff: per 32-row block `pairs` x 8 texels, pair p's rows 4 q .. 4 q + 3 at p 8 + q. X: per column pair, per group of
// 4 tokens 8 floats (token k's two columns at 2 k) = two RGBA32F texels; pack_x makes it from the rows.
// An item: quads q0, q0 + 1 (rows 4 q0 .. 4 q0 + 7) x groups g0, g0 + 1 (tokens 4 g0 .. 4 g0 + 7).
const char *kSource = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// rows [n][pairs] float2 -> [pairs][g4] float2 (g4: n rounded up to 4 tokens, the extra ones zero): a transpose in
// 16 x 16 tiles through local memory, the reads along a row and the writes along a column both in order
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void pack_x(__global const float2 *in, __global float2 *out, int pairs, int n, int g4) {
    __local float2 tile[16][17];
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int p0 = get_group_id(0) * 16, t0 = get_group_id(1) * 16;
    const int pr = p0 + lx, tr = t0 + ly;   // read: token tr's pair pr
    tile[ly][lx] = (pr < pairs && tr < n) ? in[(size_t)tr * pairs + pr] : (float2)0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int tw = t0 + lx, pw = p0 + ly;   // write: pair pw's token tw
    if (pw < pairs && tw < g4) out[(size_t)pw * g4 + tw] = tile[lx][ly];
}

#define XREAD(i) ((float8)(read_imagef(X, 2 * (i)), read_imagef(X, 2 * (i) + 1)))

// a row's two columns (WV) against a group's four tokens (XV): ACC's four sums (no parameter is named x, y, z or w:
// those are the components')
#define MAC(ACC, WV, XV)                                                 \
    ACC.x = mad(WV.x, XV.s0, mad(WV.y, XV.s1, ACC.x));                   \
    ACC.y = mad(WV.x, XV.s2, mad(WV.y, XV.s3, ACC.y));                   \
    ACC.z = mad(WV.x, XV.s4, mad(WV.y, XV.s5, ACC.z));                   \
    ACC.w = mad(WV.x, XV.s6, mad(WV.y, XV.s7, ACC.w));
#define QUAD(WQ, XQ, A0, A1, A2, A3)                                     \
    {                                                                    \
        const float2 r0 = convert_float2(as_half2(WQ.x)), r1 = convert_float2(as_half2(WQ.y)); \
        const float2 r2 = convert_float2(as_half2(WQ.z)), r3 = convert_float2(as_half2(WQ.w)); \
        MAC(A0, r0, XQ) MAC(A1, r1, XQ) MAC(A2, r2, XQ) MAC(A3, r3, XQ)  \
    }
#define PUT(A0, A1, A2, A3, G, ROW)                                                                          \
    if ((G) < groups) {                                                                                      \
        if ((G) * 4 + 0 < n) vstore4((float4)(A0.x, A1.x, A2.x, A3.x) * scale, 0, Y + (size_t)((G) * 4 + 0) * rows + (ROW)); \
        if ((G) * 4 + 1 < n) vstore4((float4)(A0.y, A1.y, A2.y, A3.y) * scale, 0, Y + (size_t)((G) * 4 + 1) * rows + (ROW)); \
        if ((G) * 4 + 2 < n) vstore4((float4)(A0.z, A1.z, A2.z, A3.z) * scale, 0, Y + (size_t)((G) * 4 + 2) * rows + (ROW)); \
        if ((G) * 4 + 3 < n) vstore4((float4)(A0.w, A1.w, A2.w, A3.w) * scale, 0, Y + (size_t)((G) * 4 + 3) * rows + (ROW)); \
    }

__kernel __attribute__((reqd_work_group_size(8, 16, 1)))
void nar_gemm(__read_only image1d_buffer_t W, int woff, __read_only image1d_buffer_t X, __global float *Y,
              int rows, int cols, int n, float scale) {
    const int q0 = get_global_id(0) * 2, g0 = get_global_id(1) * 2;
    const int pairs = cols / 2, groups = (n + 3) / 4;
    if (q0 * 4 >= rows || g0 >= groups) return;
    const int wb = woff + (q0 / 8) * pairs * 8 + q0 % 8;
    const int xa = g0, xb = min(g0 + 1, groups - 1);
    float4 a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;   // quad 0, group 0 / 1: rows 0..3
    float4 b00 = 0, b01 = 0, b02 = 0, b03 = 0, b10 = 0, b11 = 0, b12 = 0, b13 = 0;   // quad 1
    for (int p = 0; p < pairs; ++p) {
        const uint4 w0 = read_imageui(W, wb + p * 8), w1 = read_imageui(W, wb + p * 8 + 1);
        const float8 x0 = XREAD(p * groups + xa), x1 = XREAD(p * groups + xb);
        QUAD(w0, x0, a00, a01, a02, a03)
        QUAD(w0, x1, a10, a11, a12, a13)
        QUAD(w1, x0, b00, b01, b02, b03)
        QUAD(w1, x1, b10, b11, b12, b13)
    }
    const int row = q0 * 4;
    PUT(a00, a01, a02, a03, xa, row)
    PUT(a10, a11, a12, a13, (g0 + 1), row)
    PUT(b00, b01, b02, b03, xa, row + 4)
    PUT(b10, b11, b12, b13, (g0 + 1), row + 4)
}

// the unit's glue (NarGpu::unit). A work-group per token: x += y, then h = x / rms(x) * w (the CPU's rms_row)
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void res_norm(__global float *x, __global const float *y, __global const float *w, __global float *h, int H, float eps) {
    __local float part[128];
    const int t = get_group_id(0), l = get_local_id(0);
    __global float *xr = x + (size_t)t * H;
    __global const float *yr = y + (size_t)t * H;
    float s = 0.0f;
    for (int i = l; i < H; i += 128) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s = mad(v, v, s);
    }
    part[l] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 64; k > 0; k >>= 1) {
        if (l < k) part[l] += part[l + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = 1.0f / sqrt(part[0] / (float)H + eps);
    __global float *hr = h + (size_t)t * H;
    for (int i = l; i < H; i += 128) hr[i] = (xr[i] * r) * w[i];
}

// act[t][i] = SiLU(g) * u, gu[t] = (gate's F, up's F)
__kernel void silu_mul(__global const float *gu, __global float *act, int F) {
    const int i = get_global_id(0), t = get_global_id(1);
    if (i >= F) return;
    const float g = gu[(size_t)t * 2 * F + i], u = gu[(size_t)t * 2 * F + F + i];
    act[(size_t)t * F + i] = g / (1.0f + exp(-g)) * u;
}

// x += y, four at a time (n4: the count / 4)
__kernel void res_add(__global float4 *x, __global const float4 *y, int n4) {
    const int i = get_global_id(0);
    if (i < n4) x[i] += y[i];
}
)CL";

size_t up(size_t v, size_t m) { return (v + m - 1) / m * m; }

// fn over [0, n): on the pool, or on this thread when there is none (a helper beside a pool that is busy)
void par(ThreadPool *pool, int64_t n, const std::function<void(int64_t, int64_t)> &fn) {
    if (pool) pool->parallel_for(n, fn);
    else if (n > 0) fn(0, n);
}

}  // namespace

struct NarGpu::Impl {
    struct Slot {
        const uint8_t *key = nullptr;   // the layer stretch it holds
        size_t cap = 0;
        cl_mem buf = nullptr, img = nullptr;
        uint64_t used = 0;
    };
    cl::Device dev;
    cl::Program prog;
    cl_kernel k = nullptr, pack = nullptr, res_norm = nullptr, silu_mul = nullptr, res_add = nullptr;
    std::vector<Slot> slots;
    uint64_t clock = 0;
    cl_mem xrows = nullptr;                 // the activations' rows as the CPU copies them in
    cl_mem xbuf = nullptr, ximg = nullptr;  // packed (the GPU's own memory)
    cl_mem ybuf = nullptr;
    size_t rcap = 0, xcap = 0, ycap = 0;
    // the unit's (NarGpu::unit): a and x copied in (x out), h / act, the products' outputs, the norm's weights
    cl_mem ua = nullptr, ux = nullptr, uh = nullptr, uy = nullptr, un = nullptr;
    size_t ua_cap = 0, ux_cap = 0, uh_cap = 0, uy_cap = 0, un_cap = 0;
    std::deque<std::pair<cl_event, cl_event>> inflight;   // the units in flight: their first and last commands
    void drop_units() {   // (after a failure: nothing of theirs still running)
        if (inflight.empty()) return;
        c()->clFinish(dev.queue);
        for (auto &[a, b] : inflight) c()->clReleaseEvent(a), c()->clReleaseEvent(b);
        inflight.clear();
    }
    double span(cl_event a, cl_event b) const {   // the queue's seconds from a's start to b's end
        cl_ulong t0 = 0, t1 = 0;
        c()->clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_START, sizeof t0, &t0, nullptr);
        c()->clGetEventProfilingInfo(b, CL_PROFILING_COMMAND_END, sizeof t1, &t1, nullptr);
        return t1 > t0 ? static_cast<double>(t1 - t0) * 1e-9 : 0.0;
    }
    // a device buffer of at least `bytes` (grown, its old contents dropped)
    bool grow(cl_mem &m, size_t &cap, size_t bytes, std::string &err) {
        if (cap >= bytes) return true;
        release(m);
        cap = 0;
        if (!(m = buffer(CL_MEM_READ_WRITE, bytes, false, err))) return false;
        cap = bytes;
        return true;
    }
    // the packed activations' buffer and image for n tokens of `cols`
    bool grow_x(int64_t cols, int n, std::string &err) {
        const size_t g4 = static_cast<size_t>((n + 3) / 4 * 4);
        const size_t xbytes = static_cast<size_t>(cols / 2) * g4 * 2 * sizeof(float);
        if (xcap >= xbytes) return true;
        release(ximg), release(xbuf);
        xcap = 0;
        if (!(xbuf = buffer(CL_MEM_READ_WRITE, xbytes, false, err)) || !(ximg = image(xbuf, xbytes, true, err))) return false;
        xcap = xbytes;
        return true;
    }
    Times t;
    // the product started and not yet finished: its kernels' events (pack, product), its shape
    cl_event ev[2] = {nullptr, nullptr};
    int pending_n = 0;
    int64_t pending_rows = 0;
    double started = 0.0;
    void drop_events() {
        for (cl_event &e : ev)
            if (e) c()->clReleaseEvent(e), e = nullptr;
        pending_n = 0;
    }

    const cl::Api *c() const { return dev.cl; }
    void release(cl_mem &m) {
        if (m) c()->clReleaseMemObject(m);
        m = nullptr;
    }
    ~Impl() {
        if (dev.cl) {
            c()->clFinish(dev.queue);
            drop_events();
            drop_units();
        }
        for (Slot &s : slots) release(s.img), release(s.buf);
        release(ximg), release(xbuf), release(xrows), release(ybuf);
        release(ua), release(ux), release(uh), release(uy), release(un);
    }
    cl_mem image(cl_mem buf, size_t bytes, bool f32, std::string &err) {
        cl_image_format fmt{CL_RGBA, f32 ? static_cast<cl_channel_type>(CL_FLOAT) : static_cast<cl_channel_type>(CL_UNSIGNED_INT32)};
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        desc.image_width = bytes / 16;
        desc.buffer = buf;
        cl_int e = CL_SUCCESS;
        cl_mem m = c()->clCreateImage(dev.context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        if (!m) err = std::string("NAR GPU: an image of ") + std::to_string(bytes >> 20) + " MB: " + cl::error_name(e);
        return m;
    }
    // host: the CPU maps it (ALLOC_HOST_PTR), else the GPU's alone
    cl_mem buffer(cl_mem_flags flags, size_t bytes, bool host, std::string &err) {
        cl_int e = CL_SUCCESS;
        cl_mem m = c()->clCreateBuffer(dev.context, flags | (host ? CL_MEM_ALLOC_HOST_PTR : 0), bytes, nullptr, &e);
        if (!m) err = std::string("NAR GPU: ") + std::to_string(bytes >> 20) + " MB of GPU memory: " + cl::error_name(e);
        return m;
    }
    void *map(cl_mem m, cl_map_flags flags, size_t bytes, std::string &err) {
        cl_int e = CL_SUCCESS;
        void *p = c()->clEnqueueMapBuffer(dev.queue, m, CL_TRUE, flags, 0, bytes, 0, nullptr, nullptr, &e);
        if (!p) err = std::string("NAR GPU: mapping a buffer: ") + cl::error_name(e);
        return p;
    }
    void unmap(cl_mem m, void *p) { c()->clEnqueueUnmapMemObject(dev.queue, m, p, 0, nullptr, nullptr); }
    // len bytes in order, 1 MB a task
    static void copy(uint8_t *dst, const uint8_t *src, size_t len, ThreadPool *pool) {
        constexpr size_t kChunk = size_t{1} << 20;
        par(pool, static_cast<int64_t>((len + kChunk - 1) / kChunk), [&](int64_t a, int64_t b) {
            for (int64_t i = a; i < b; ++i) {
                const size_t o = static_cast<size_t>(i) * kChunk;
                std::memcpy(dst + o, src + o, std::min(kChunk, len - o));
            }
        });
    }

    // the slot holding `layer`, filled now if no slot does (the least recently used one given up)
    Slot *slot_of(const uint8_t *layer, size_t len, ThreadPool *pool, std::string &err) {
        Slot *best = &slots[0];
        for (Slot &s : slots) {
            if (s.key == layer && s.cap >= len) {
                s.used = ++clock;
                return &s;
            }
            if (s.used < best->used) best = &s;
        }
        const double t0 = now_seconds();
        Slot &s = *best;
        s.key = nullptr;
        if (s.cap < len) {
            release(s.img), release(s.buf);
            s.cap = 0;
            if (!(s.buf = buffer(CL_MEM_READ_ONLY, len, true, err)) || !(s.img = image(s.buf, len, false, err))) return nullptr;
            s.cap = len;
        }
        auto *dst = static_cast<uint8_t *>(map(s.buf, CL_MAP_WRITE_INVALIDATE_REGION, len, err));
        if (!dst) return nullptr;
        copy(dst, layer, len, pool);
        unmap(s.buf, dst);
        s.key = layer;
        s.used = ++clock;
        t.upload += now_seconds() - t0;
        ++t.uploads;
        return &s;
    }
};

NarGpu::NarGpu() = default;
NarGpu::~NarGpu() = default;

const std::string &NarGpu::name() const {
    static const std::string none;
    return d_ ? d_->dev.name : none;
}

const NarGpu::Times &NarGpu::times() const {
    static const Times none;
    return d_ ? d_->t : none;
}

bool NarGpu::open(int slots, size_t layer_bytes, const std::string &cache_dir, std::string &err) {
    auto d = std::make_unique<Impl>();
    if (!d->dev.open(err)) {
        err = "NAR GPU: " + err;
        return false;
    }
    if (!d->dev.images || !d->dev.fp16) {
        err = "NAR GPU: " + d->dev.name + " lacks images or fp16";
        return false;
    }
    if (d->dev.image_max_buffer && layer_bytes / 16 > d->dev.image_max_buffer) {
        err = "NAR GPU: a layer (" + std::to_string(layer_bytes >> 20) + " MB) is past the device's image buffers";
        return false;
    }
    const bool built = cache_dir.empty() ? d->prog.build(d->dev, kSource, "", err)
                                         : d->prog.build_cached(d->dev, kSource, "", cache_dir, err);
    if (!built || !(d->k = d->prog.kernel("nar_gemm", err)) || !(d->pack = d->prog.kernel("pack_x", err)) ||
        !(d->res_norm = d->prog.kernel("res_norm", err)) || !(d->silu_mul = d->prog.kernel("silu_mul", err)) ||
        !(d->res_add = d->prog.kernel("res_add", err))) {
        err = "NAR GPU: " + err;
        return false;
    }
    d->slots.resize(static_cast<size_t>(std::max(1, slots)));
    d_ = std::move(d);
    return true;
}

bool NarGpu::prefetch(const uint8_t *layer, size_t len, ThreadPool *pool, std::string &err) {
    if (!d_ || d_->slots.size() < 2) return false;
    return d_->slot_of(layer, len, pool, err) != nullptr;
}

bool NarGpu::gemm(const uint8_t *layer, size_t len, uint64_t off, int64_t rows, int64_t cols, int shift, int n,
                  const float *const *xs, float *const *ys, ThreadPool &pool, std::string &err) {
    return start(layer, len, off, rows, cols, shift, n, xs, &pool, err) && finish(ys, &pool, err);
}

bool NarGpu::unit(const uint8_t *layer, size_t len, const UnitShape &u, const float *norm, int n, const float *a,
                  float *x, std::string &err, double *gpu_s, bool wait) {
    Impl &d = *d_;
    d.drop_events();
    const int H = u.H, qd = u.qd, F = u.F;
    if (n < 1 || H % 128 || qd % 32 || F % 32 || H % 32 || u.o % 16 || u.gu % 16 || u.down % 16 ||
        u.o + static_cast<uint64_t>(H) * qd * 2 > len || u.gu + static_cast<uint64_t>(2 * F) * H * 2 > len ||
        u.down + static_cast<uint64_t>(H) * F * 2 > len) {
        err = "NAR GPU: a unit the kernels do not take";
        d.drop_units();   // (the units before it: nothing of theirs left running over the caller's rows)
        return false;
    }
    // (the buffers grow only with nothing in flight: the units queued before use them)
    const size_t nn = static_cast<size_t>(n), f4 = sizeof(float);
    const size_t xneed = static_cast<size_t>(std::max({qd, H, F}) / 2) * ((nn + 3) / 4 * 4) * 2 * f4;   // (grow_x's)
    if (!d.inflight.empty() && (d.ua_cap < nn * qd * f4 || d.ux_cap < nn * H * f4 || d.uh_cap < nn * std::max(H, F) * f4 ||
                                d.uy_cap < nn * std::max(H, 2 * F) * f4 || d.un_cap < static_cast<size_t>(H) * f4 ||
                                d.xcap < xneed))
        d.c()->clFinish(d.dev.queue);   // (they complete; unit_wait() still collects them)
    Impl::Slot *s = d.slot_of(layer, len, nullptr, err);
    if (!s || !d.grow(d.ua, d.ua_cap, nn * qd * f4, err) || !d.grow(d.ux, d.ux_cap, nn * H * f4, err) ||
        !d.grow(d.uh, d.uh_cap, nn * std::max(H, F) * f4, err) || !d.grow(d.uy, d.uy_cap, nn * std::max(H, 2 * F) * f4, err) ||
        !d.grow(d.un, d.un_cap, static_cast<size_t>(H) * f4, err) || !d.grow_x(std::max({qd, H, F}), n, err)) {
        d.drop_units();
        return false;
    }
    const cl::Api *c = d.c();
    cl_command_queue q = d.dev.queue;
    cl_event first = nullptr, last = nullptr;
    const auto fail = [&](const char *what, cl_int e) {
        err = std::string("NAR GPU: the unit's ") + what + ": " + cl::error_name(e);
        c->clFinish(q);   // (nothing of it - or of the units before it - still running over the caller's rows)
        if (first) c->clReleaseEvent(first);
        d.drop_units();
        return false;
    };
    // a and x in (the caller's rows stay put until the read below: it blocks)
    cl_int e = c->clEnqueueWriteBuffer(q, d.ua, CL_FALSE, 0, nn * qd * f4, a, 0, nullptr, &first);
    if (e == CL_SUCCESS) e = c->clEnqueueWriteBuffer(q, d.ux, CL_FALSE, 0, nn * H * f4, x, 0, nullptr, nullptr);
    if (e == CL_SUCCESS) e = c->clEnqueueWriteBuffer(q, d.un, CL_FALSE, 0, static_cast<size_t>(H) * f4, norm, 0, nullptr, nullptr);
    if (e != CL_SUCCESS) return fail("copies in", e);
    const int groups = (n + 3) / 4, g4 = groups * 4;
    // y[n][rows] = W[off] in[n][cols]: in packed by pack_x, then the product (gemm()'s kernel)
    const auto product = [&](cl_mem in, uint64_t off, int rows, int cols, int shift, cl_mem out) {
        const int ipairs = cols / 2;
        c->clSetKernelArg(d.pack, 0, sizeof(cl_mem), &in);
        c->clSetKernelArg(d.pack, 1, sizeof(cl_mem), &d.xbuf);
        c->clSetKernelArg(d.pack, 2, sizeof(int), &ipairs);
        c->clSetKernelArg(d.pack, 3, sizeof(int), &n);
        c->clSetKernelArg(d.pack, 4, sizeof(int), &g4);
        const size_t plocal[2] = {16, 16};
        const size_t pglobal[2] = {up(static_cast<size_t>(ipairs), 16), up(static_cast<size_t>(g4), 16)};
        cl_int r = c->clEnqueueNDRangeKernel(q, d.pack, 2, nullptr, pglobal, plocal, 0, nullptr, nullptr);
        if (r != CL_SUCCESS) return r;
        const int woff = static_cast<int>(off / 16);
        const float scale = std::ldexp(1.0f, -shift);
        c->clSetKernelArg(d.k, 0, sizeof(cl_mem), &s->img);
        c->clSetKernelArg(d.k, 1, sizeof(int), &woff);
        c->clSetKernelArg(d.k, 2, sizeof(cl_mem), &d.ximg);
        c->clSetKernelArg(d.k, 3, sizeof(cl_mem), &out);
        c->clSetKernelArg(d.k, 4, sizeof(int), &rows);
        c->clSetKernelArg(d.k, 5, sizeof(int), &cols);
        c->clSetKernelArg(d.k, 6, sizeof(int), &n);
        c->clSetKernelArg(d.k, 7, sizeof(float), &scale);
        const size_t local[2] = {8, 16};
        const size_t global[2] = {up(static_cast<size_t>(rows / 8), 8), up(static_cast<size_t>((groups + 1) / 2), 16)};
        return c->clEnqueueNDRangeKernel(q, d.k, 2, nullptr, global, local, 0, nullptr, nullptr);
    };
    // o, then x += o and h = the norm
    if ((e = product(d.ua, u.o, H, qd, u.sh_o, d.uy)) != CL_SUCCESS) return fail("o", e);
    {
        c->clSetKernelArg(d.res_norm, 0, sizeof(cl_mem), &d.ux);
        c->clSetKernelArg(d.res_norm, 1, sizeof(cl_mem), &d.uy);
        c->clSetKernelArg(d.res_norm, 2, sizeof(cl_mem), &d.un);
        c->clSetKernelArg(d.res_norm, 3, sizeof(cl_mem), &d.uh);
        c->clSetKernelArg(d.res_norm, 4, sizeof(int), &H);
        c->clSetKernelArg(d.res_norm, 5, sizeof(float), &u.eps);
        const size_t local = 128, global = 128 * nn;
        if ((e = c->clEnqueueNDRangeKernel(q, d.res_norm, 1, nullptr, &global, &local, 0, nullptr, nullptr)) != CL_SUCCESS)
            return fail("norm", e);
    }
    // gate / up, SiLU * up, down
    if ((e = product(d.uh, u.gu, 2 * F, H, u.sh_gu, d.uy)) != CL_SUCCESS) return fail("gate / up", e);
    {
        c->clSetKernelArg(d.silu_mul, 0, sizeof(cl_mem), &d.uy);
        c->clSetKernelArg(d.silu_mul, 1, sizeof(cl_mem), &d.uh);
        c->clSetKernelArg(d.silu_mul, 2, sizeof(int), &F);
        const size_t local[2] = {64, 1};
        const size_t global[2] = {up(static_cast<size_t>(F), 64), nn};
        if ((e = c->clEnqueueNDRangeKernel(q, d.silu_mul, 2, nullptr, global, local, 0, nullptr, nullptr)) != CL_SUCCESS)
            return fail("SiLU", e);
    }
    if ((e = product(d.uh, u.down, H, F, u.sh_down, d.uy)) != CL_SUCCESS) return fail("down", e);
    {
        const int n4 = static_cast<int>(nn * H / 4);
        c->clSetKernelArg(d.res_add, 0, sizeof(cl_mem), &d.ux);
        c->clSetKernelArg(d.res_add, 1, sizeof(cl_mem), &d.uy);
        c->clSetKernelArg(d.res_add, 2, sizeof(int), &n4);
        const size_t local = 64, global = up(static_cast<size_t>(n4), 64);
        if ((e = c->clEnqueueNDRangeKernel(q, d.res_add, 1, nullptr, &global, &local, 0, nullptr, nullptr)) != CL_SUCCESS)
            return fail("residual", e);
    }
    // x out (blocking unless left in flight: then unit_wait() has the rows whole)
    if ((e = c->clEnqueueReadBuffer(q, d.ux, wait ? CL_TRUE : CL_FALSE, 0, nn * H * f4, x, 0, nullptr, &last)) != CL_SUCCESS)
        return fail("copy out", e);
    if (!wait) {
        c->clFlush(q);
        d.inflight.emplace_back(first, last);
        if (gpu_s) *gpu_s = 0.0;
        return true;
    }
    if (gpu_s) *gpu_s = d.span(first, last);
    c->clReleaseEvent(first), c->clReleaseEvent(last);
    return true;
}

bool NarGpu::unit_wait(std::string &err, double *gpu_s) {
    Impl &d = *d_;
    if (d.inflight.empty()) {
        err = "NAR GPU: unit_wait() with no unit in flight";
        return false;
    }
    const auto [first, last] = d.inflight.front();
    d.inflight.pop_front();
    const cl::Api *c = d.c();
    cl_int st = CL_SUCCESS;
    cl_int e = c->clWaitForEvents(1, &last);
    if (e == CL_SUCCESS)
        e = c->clGetEventInfo(last, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, nullptr);
    if (e != CL_SUCCESS || st < 0) {
        err = std::string("NAR GPU: a unit failed: ") + cl::error_name(e != CL_SUCCESS ? e : st);
        c->clFinish(d.dev.queue);   // (this one too: nothing left running over the caller's rows)
        c->clReleaseEvent(first), c->clReleaseEvent(last);
        d.drop_units();
        return false;
    }
    if (gpu_s) *gpu_s = d.span(first, last);
    c->clReleaseEvent(first), c->clReleaseEvent(last);
    return true;
}

int NarGpu::units_in_flight() const { return d_ ? static_cast<int>(d_->inflight.size()) : 0; }

bool NarGpu::start(const uint8_t *layer, size_t len, uint64_t off, int64_t rows, int64_t cols, int shift, int n,
                   const float *const *xs, ThreadPool *pool, std::string &err) {
    Impl &d = *d_;
    d.drop_events();
    if (rows % 32 || cols % 32 || off % 16 || off + static_cast<uint64_t>(rows * cols * 2) > len || n < 1) {
        err = "NAR GPU: a product the kernel does not take";
        return false;
    }
    Impl::Slot *s = d.slot_of(layer, len, pool, err);
    if (!s) return false;
    const cl::Api *c = d.c();
    const auto launch = [&](cl_kernel kk, const size_t *global, const size_t *local, const char *what, cl_event *ev) {
        const cl_int e = c->clEnqueueNDRangeKernel(d.dev.queue, kk, 2, nullptr, global, local, 0, nullptr, ev);
        if (e != CL_SUCCESS) err = std::string("NAR GPU: ") + what + ": " + cl::error_name(e);
        return e == CL_SUCCESS;
    };
    // the activations' rows copied in, then packed on the GPU: [pair][group of 4 tokens] x (token, its two columns)
    double t0 = now_seconds();
    const int ipairs = static_cast<int>(cols / 2), groups = (n + 3) / 4, g4 = groups * 4;
    const size_t rbytes = static_cast<size_t>(n) * static_cast<size_t>(cols) * sizeof(float);
    const size_t xbytes = static_cast<size_t>(cols / 2) * static_cast<size_t>(g4) * 2 * sizeof(float);
    if (d.rcap < rbytes) {
        d.release(d.xrows);
        d.rcap = 0;
        if (!(d.xrows = d.buffer(CL_MEM_READ_ONLY, rbytes, true, err))) return false;
        d.rcap = rbytes;
    }
    if (d.xcap < xbytes) {
        d.release(d.ximg), d.release(d.xbuf);
        d.xcap = 0;
        if (!(d.xbuf = d.buffer(CL_MEM_READ_WRITE, xbytes, false, err)) || !(d.ximg = d.image(d.xbuf, xbytes, true, err))) return false;
        d.xcap = xbytes;
    }
    void *xm = d.map(d.xrows, CL_MAP_WRITE_INVALIDATE_REGION, rbytes, err);
    if (!xm) return false;
    const size_t rb = static_cast<size_t>(cols) * sizeof(float);
    par(pool, n, [&](int64_t a, int64_t b) {   // (in order: the mapped memory is write-combined)
        for (int64_t tk = a; tk < b; ++tk) std::memcpy(static_cast<uint8_t *>(xm) + static_cast<size_t>(tk) * rb, xs[tk], rb);
    });
    d.unmap(d.xrows, xm);
    c->clSetKernelArg(d.pack, 0, sizeof(cl_mem), &d.xrows);
    c->clSetKernelArg(d.pack, 1, sizeof(cl_mem), &d.xbuf);
    c->clSetKernelArg(d.pack, 2, sizeof(int), &ipairs);
    c->clSetKernelArg(d.pack, 3, sizeof(int), &n);
    c->clSetKernelArg(d.pack, 4, sizeof(int), &g4);
    const size_t plocal[2] = {16, 16};
    const size_t pglobal[2] = {up(static_cast<size_t>(ipairs), 16), up(static_cast<size_t>(g4), 16)};
    if (!launch(d.pack, pglobal, plocal, "the activations' packing", &d.ev[0])) return false;
    const size_t ybytes = static_cast<size_t>(n) * static_cast<size_t>(rows) * sizeof(float);
    if (d.ycap < ybytes) {
        d.release(d.ybuf);
        d.ycap = 0;
        if (!(d.ybuf = d.buffer(CL_MEM_WRITE_ONLY, ybytes, true, err))) return false;
        d.ycap = ybytes;
    }
    double t1 = now_seconds();
    d.t.pack += t1 - t0;
    const int woff = static_cast<int>(off / 16), irows = static_cast<int>(rows), icols = static_cast<int>(cols);
    const float scale = std::ldexp(1.0f, -shift);
    c->clSetKernelArg(d.k, 0, sizeof(cl_mem), &s->img);
    c->clSetKernelArg(d.k, 1, sizeof(int), &woff);
    c->clSetKernelArg(d.k, 2, sizeof(cl_mem), &d.ximg);
    c->clSetKernelArg(d.k, 3, sizeof(cl_mem), &d.ybuf);
    c->clSetKernelArg(d.k, 4, sizeof(int), &irows);
    c->clSetKernelArg(d.k, 5, sizeof(int), &icols);
    c->clSetKernelArg(d.k, 6, sizeof(int), &n);
    c->clSetKernelArg(d.k, 7, sizeof(float), &scale);
    const size_t local[2] = {8, 16};
    const size_t global[2] = {up(static_cast<size_t>(rows / 8), 8), up(static_cast<size_t>((groups + 1) / 2), 16)};
    if (!launch(d.k, global, local, "the product's launch", &d.ev[1])) return false;
    c->clFlush(d.dev.queue);   // (on its way while the caller computes)
    d.pending_n = n, d.pending_rows = rows;
    d.started = t1;
    return true;
}

bool NarGpu::finish(float *const *ys, ThreadPool *pool, std::string &err, double *gpu_s) {
    Impl &d = *d_;
    const int n = d.pending_n;
    const int64_t rows = d.pending_rows;
    if (n < 1) {
        err = "NAR GPU: finish() with no product started";
        return false;
    }
    d.pending_n = 0;
    const size_t ybytes = static_cast<size_t>(n) * static_cast<size_t>(rows) * sizeof(float);
    void *ym = d.map(d.ybuf, CL_MAP_READ, ybytes, err);   // (blocking: after both kernels)
    if (!ym) return false;
    const double t0 = now_seconds();
    d.t.kernel += t0 - d.started;
    par(pool, n, [&](int64_t a, int64_t b) {
        for (int64_t tk = a; tk < b; ++tk)
            std::memcpy(ys[tk], static_cast<const float *>(ym) + static_cast<size_t>(tk) * static_cast<size_t>(rows),
                        sizeof(float) * static_cast<size_t>(rows));
    });
    d.unmap(d.ybuf, ym);
    d.t.unpack += now_seconds() - t0;
    double s = 0.0;
    for (cl_event &ev : d.ev)
        if (ev) s += cl::event_seconds(d.c(), ev), ev = nullptr;   // (released by event_seconds)
    if (gpu_s) *gpu_s = s;
    return true;
}

}  // namespace geode::yue2
