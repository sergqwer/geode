// gpu.cpp - see gpu.hpp. The program: the driver compiles a template kernel with the gemm signature (its buffers,
// scalars and four images, 4 KB of local memory, room for the code), the hand-written machine code goes into the
// binary's code section and the header / descriptor get its registers and wave mode (bench/gpu/asm/inject.py; the
// driver checks nothing else), and the patched binary is built again.
#include "gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "opencl.hpp"
#include "platform.hpp"
#include "threads.hpp"

#include <CL/cl_ext.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__ANDROID__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace geode {

namespace {

#include "gpu_gemm.inc"

// The kernels: one 64-value chunk of K a lane, 4 rows x 4 tokens a fiber, 128-fiber groups; K = 2048 (32 lanes, 16 rows
// a group) or 4096 (64 lanes, 8 rows a group).
struct Kind {
    DType type;
    int cols;
    const uint64_t *code;
    size_t n;
};
constexpr Kind kKinds[] = {{DType::Q2, 2048, kGemm_q2, sizeof kGemm_q2 / 8},
                           {DType::Q4, 2048, kGemm_q4, sizeof kGemm_q4 / 8},
                           {DType::Q4, 4096, kGemm_q4_4096, sizeof kGemm_q4_4096 / 8}};
constexpr int kKindCount = sizeof kKinds / sizeof kKinds[0];
int kind_of(DType t, int64_t cols) {
    for (int k = 0; k < kKindCount; ++k)
        if (kKinds[k].type == t && kKinds[k].cols == cols) return k;
    return -1;
}
constexpr int kRowsPerGroup = 16;    // every kind's rows come in multiples of this
constexpr int kMaxCols = 4096;
constexpr int kGroupSize = 128;
constexpr int kTokensPerFiber = 4;
constexpr size_t kTokenBytes = static_cast<size_t>(kMaxCols) * 2;   // FP16, room for the widest

// bench/gpu/asm/template.py's kernel (1024 floats of local memory, 4 images)
std::string template_source() {
    std::string s =
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        "__kernel void gemm(__global const uchar *w, uint w_off, int rows, int cols, int row_bytes, float global_scale,\n"
        "                   __global const half *xh, int ntok, __global float *y,\n"
        "                   __read_only image1d_buffer_t wimg, __read_only image1d_buffer_t simg,\n"
        "                   __read_only image1d_buffer_t ximg, __read_only image1d_buffer_t ximg3) {\n"
        "    __local float part[1024];\n"
        "    const int lid = get_local_id(0);\n"
        "    const int gx = get_group_id(0), gy = get_group_id(1);\n"
        "    float acc = (float)lid;\n"
        "    for (int k = 0; k < ntok; ++k) {\n"
        "        const uint a = read_imageui(wimg, lid + k + gx).x;\n"
        "        const uint b = read_imageui(simg, lid + k + gy).x;\n"
        "        const half4 c = read_imageh(ximg, lid + k + rows + cols);\n"
        "        acc += (float)(a ^ b ^ (uint)row_bytes ^ w_off) * (float)c.x + (float)read_imageh(ximg3, lid + k).y;\n"
        "    }\n"
        "    part[lid] = acc;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    float s = part[(lid + 1) & 1023] + part[(lid + 512) & 1023];\n"
        "    const float m = (float)ntok * 1e-3f, k0 = global_scale;\n"
        "    float s2 = (float)cols, m2 = (float)row_bytes;\n";
    static const char *const a[4] = {"s", "m", "k0", "s2"}, *const b[4] = {"m", "k0", "s", "m2"};
    for (int n = 0; n < 1500; ++n)
        s += std::string("    s = mad(s, ") + a[n % 4] + ", " + b[n / 4 % 4] + "); s2 = s2 * s + m2;\n";
    s += "    y[get_global_id(0) + (size_t)gy * 7] = s + s2 + (float)xh[lid] + (float)w[lid];\n}\n";
    return s;
}

// The machine code into the template binary (bench/gpu/asm/inject.py).
bool inject(std::vector<uint8_t> &b, const uint64_t *code, size_t n, int full, int half, std::string &err) {
    const size_t words = b.size() / 4;
    auto dw = [&](size_t i) {
        uint32_t v;
        std::memcpy(&v, &b[4 * i], 4);
        return v;
    };
    auto put = [&](size_t at, uint32_t v) { std::memcpy(&b[at], &v, 4); };
    struct Sec {
        uint32_t off, size;
    };
    std::map<uint32_t, Sec> s;   // the section table: (type, offset, size, count, entry size) a section
    for (size_t i = 0; i + 5 < words; ++i) {
        const uint32_t t = dw(i), off = dw(i + 1), size = dw(i + 2), cnt = dw(i + 3), es = dw(i + 4);
        if (t > 0 && t <= 0x10 && off <= b.size() && es && static_cast<uint64_t>(cnt) * es == size && size &&
            !s.count(t))
            s[t] = {off, size};
    }
    if (!s.count(0xa) || !s.count(1) || !s.count(0xb)) {
        err = "GPU: the driver's program binary has an unknown layout";
        return false;
    }
    const Sec c = s[0xa], h = s[1], d = s[0xb];
    if (n * 8 > c.size || static_cast<size_t>(c.off) + c.size > b.size() || static_cast<size_t>(h.off) + 64 > b.size() ||
        static_cast<size_t>(d.off) + 0x124 > b.size()) {
        err = "GPU: the template program has no room for the code";
        return false;
    }
    std::memcpy(&b[c.off], code, n * 8);
    std::memset(&b[c.off + n * 8], 0, c.size - n * 8);   // nops
    const int regs = full + (half + 1) / 2;
    const uint32_t waves = static_cast<uint32_t>(std::min(16, 96 / std::max(regs, 1) * 2) / 2);   // wave 128
    put(h.off + 3 * 4, static_cast<uint32_t>(full));
    put(h.off + 4 * 4, static_cast<uint32_t>(half));
    put(h.off + 14 * 4, waves);
    for (uint32_t rel : {0x38u, 0xf8u}) {
        const uint32_t v = dw((d.off + rel) / 4);
        if (v != 8 && v != 16) {
            err = "GPU: the program descriptor has an unknown layout";
            return false;
        }
        put(d.off + rel, 128 / 8);
    }
    put(d.off + 0xcc, static_cast<uint32_t>(full));
    put(d.off + 0xd0, static_cast<uint32_t>(half));
    put(d.off + 0x120, waves);
    return true;
}

// the largest |x[i]|
inline float abs_max(const float *x, int n) {
#if defined(__aarch64__)
    float32x4_t m = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x + i)));
    return vmaxvq_f32(m);
#else
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    return m;
#endif
}

// FP16 of x (round to nearest even), 4 at a time
inline void halves4(const float *x, float inv, uint16_t *out) {
#if defined(__aarch64__)
    vst1_u16(out, vreinterpret_u16_f16(vcvt_f16_f32(vmulq_n_f32(vld1q_f32(x), inv))));
#else
    for (int i = 0; i < 4; ++i) {
        const float f = x[i] * inv;
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint32_t sign = (u >> 16) & 0x8000u;
        u &= 0x7FFFFFFFu;
        uint16_t h;
        if (u >= 0x47800000u) {
            h = static_cast<uint16_t>(u > 0x7F800000u ? 0x7E00u : 0x7C00u);
        } else if (u < 0x38800000u) {   // subnormal or zero: the GPU flushes them anyway
            h = 0;
        } else {
            const uint32_t r = u + 0xFFFu + ((u >> 13) & 1u);   // nearest even at bit 13
            h = static_cast<uint16_t>(((r >> 13) - ((127u - 15u) << 10)) & 0x7FFFu);
        }
        out[i] = static_cast<uint16_t>(sign | h);
    }
#endif
}

// Memory both the CPU (cached: its kernels read the weights at full speed) and the GPU (without a copy) read: ION
// buffers wrapped by cl_qcom_ion_host_ptr. CL_MEM_ALLOC_HOST_PTR memory is uncached for the CPU on the Adreno: its
// kernels ran 10x slower on it (one token 0.77 -> 7.5 ms), and CL_MEM_USE_HOST_PTR on malloc memory is a copy.
#if defined(__ANDROID__)
class Ion {
public:
    ~Ion() {
        if (dev_ >= 0) ::close(dev_);
    }
    bool open(std::string &err) {
        if (dev_ >= 0) return true;
        dev_ = ::open("/dev/ion", O_RDONLY | O_CLOEXEC);
        if (dev_ < 0) {
            err = "GPU: no /dev/ion";
            return false;
        }
        // the system heap by name (Qualcomm's id is 25 when the query is missing)
        HeapData heaps[32] = {};
        HeapQuery q{};
        q.cnt = 32;
        q.heaps = reinterpret_cast<uintptr_t>(heaps);
        mask_ = 1u << 25;
        if (::ioctl(dev_, kHeapQuery, &q) == 0)
            for (uint32_t i = 0; i < q.cnt && i < 32; ++i)
                if (std::strncmp(heaps[i].name, "system", sizeof heaps[i].name) == 0) mask_ = 1u << heaps[i].heap_id;
        return true;
    }
    // bytes of cached memory: its dma-buf fd and its mapping
    bool alloc(size_t bytes, int &fd, void *&ptr, std::string &err) const {
        AllocationData a{};
        a.len = bytes;
        a.heap_id_mask = mask_;
        a.flags = 1;   // ION_FLAG_CACHED
        if (::ioctl(dev_, kAlloc, &a) != 0) {
            err = "GPU: ION allocation of " + std::to_string(bytes >> 20) + " MB failed";
            return false;
        }
        fd = static_cast<int>(a.fd);
        ptr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            ::close(fd);
            err = "GPU: mapping an ION buffer failed";
            return false;
        }
        return true;
    }

private:
    // the kernel's ION ABI since 4.12: an allocation returns a dma-buf fd
    struct AllocationData {
        uint64_t len;
        uint32_t heap_id_mask, flags, fd, unused;
    };
    struct HeapData {
        char name[32];
        uint32_t type, heap_id, reserved0, reserved1, reserved2;
    };
    struct HeapQuery {
        uint32_t cnt, reserved0;
        uint64_t heaps;
        uint32_t reserved1, reserved2;
    };
    static constexpr unsigned long kAlloc = _IOWR('I', 0, AllocationData);
    static constexpr unsigned long kHeapQuery = _IOWR('I', 8, HeapQuery);
    int dev_ = -1;
    uint32_t mask_ = 0;
};
#endif

}  // namespace

struct Gpu::Impl {
    cl::Device dev;
    const cl::Api *cl = nullptr;
    cl::Program prog[kKindCount];
    cl_kernel kernel[kKindCount] = {};
    // a buffer the CPU writes or reads directly: ION (cached for the CPU, the driver keeps the caches right at map
    // and unmap), or with GEODE_GPU_MEM=alloc CL_MEM_ALLOC_HOST_PTR (uncached: slow for the CPU, kept to compare)
    struct Host {
        cl_mem buf = nullptr;
        int fd = -1;
        void *host = nullptr;   // the CPU's view (ION: its mapping)
        size_t host_bytes = 0;
    };
    struct Mat {
        Host h;
        DType type = DType::Unknown;
        int64_t rows = 0, cols = 0;
        size_t row_bytes = 0;
        float global = 1.0f;
    };
#if defined(__ANDROID__)
    Ion ion;
#endif
    // GEODE_GPU_MEM: wb (default: ION, cached for the CPU, cache maintenance by the driver), io (ION, IO-coherent),
    // alloc (CL_MEM_ALLOC_HOST_PTR: uncached for the CPU)
    std::string mem = "wb";
    size_t padding = 0, page = 4096;
    bool io_xy = std::getenv("GEODE_GPU_IOXY") != nullptr;   // activations and outputs IO-coherent
    std::unordered_map<const uint8_t *, Mat> mats;   // by the CPU's pointer the views hold
    size_t adopted = 0;
    // the activations: one buffer, four images of it a token apart (the kernel's one coordinate serves all four)
    Host x;
    cl_mem ximg[4] = {nullptr, nullptr, nullptr, nullptr}, xsub[3] = {nullptr, nullptr, nullptr};
    int x_tokens = 0;
    Host y;
    size_t y_bytes = 0;
    // the product in flight
    cl_event ev = nullptr;
    int n = 0, ntok = 0;
    int64_t rows = 0;
    std::vector<float> scale;
    double busy = 0.0;
    int64_t launches = 0;
    double t_prep = 0, t_launch = 0, t_wait = 0, t_out = 0;   // the calls' wall time: start's two parts, finish's
    Gpu::ParallelFor par;
    void for_tokens(int n, const std::function<void(int)> &fn) {
        if (par && n >= 8) par(n, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) fn(static_cast<int>(t));
        });
        else for (int t = 0; t < n; ++t) fn(t);
    }

    ~Impl() {
        if (!cl) return;
        if (ev) {
            cl->clWaitForEvents(1, &ev);
            cl->clReleaseEvent(ev);
        }
        free_x();
        release(y);
        for (auto &kv : mats) release(kv.second.h);
    }
    bool make(size_t bytes, cl_mem_flags flags, Host &h, std::string &err, bool coherent = false) {
        cl_int e = CL_SUCCESS;
#if defined(__ANDROID__)
        if (mem != "alloc") {
            const size_t hb = (bytes + padding + page - 1) / page * page;   // + the driver's padding, whole pages
            if (!ion.alloc(hb, h.fd, h.host, err)) return false;
            h.host_bytes = hb;
            cl_mem_ion_host_ptr ip{};
            ip.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
            ip.ext_host_ptr.host_cache_policy =
                mem == "io" || coherent ? CL_MEM_HOST_IOCOHERENT_QCOM : CL_MEM_HOST_WRITEBACK_QCOM;
            ip.ion_filedesc = h.fd;
            ip.ion_hostptr = h.host;
            h.buf = cl->clCreateBuffer(dev.context, flags | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, bytes, &ip, &e);
            if (e != CL_SUCCESS) {
                release(h);
                return fail(err, "ION buffer", e);
            }
            return true;
        }
#endif
        h.buf = cl->clCreateBuffer(dev.context, flags | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &e);
        if (e != CL_SUCCESS) {
            h.buf = nullptr;
            return fail(err, "buffer", e);
        }
        return true;
    }
    void release(Host &h) {
        if (h.buf) cl->clReleaseMemObject(h.buf);   // an ALLOC_HOST_PTR buffer's mapping goes with it
#if defined(__ANDROID__)
        if (h.fd >= 0) {
            ::munmap(h.host, h.host_bytes);
            ::close(h.fd);
        }
#endif
        h = Host{};
    }
    void free_x() {
        for (cl_mem &m : ximg)
            if (m) cl->clReleaseMemObject(m), m = nullptr;
        for (cl_mem &m : xsub)
            if (m) cl->clReleaseMemObject(m), m = nullptr;
        release(x);
        x_tokens = 0;
    }
    bool ensure_x(int tokens, std::string &err) {
        if (tokens <= x_tokens) return true;
        free_x();
        const int want = std::max(tokens, 64);
        const size_t bytes = static_cast<size_t>(want) * kTokenBytes;
        if (!make(bytes, CL_MEM_READ_ONLY, x, err, io_xy)) return false;
        auto image = [&](cl_mem buf, size_t size, cl_mem &out) {
            const cl_image_format fmt{CL_RGBA, CL_HALF_FLOAT};
            cl_image_desc desc{};
            desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            desc.image_width = size / 8;   // 4 halves a texel
            desc.buffer = buf;
            cl_int ie = CL_SUCCESS;
            out = cl->clCreateImage(dev.context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &ie);
            return ie;
        };
        cl_int e = CL_SUCCESS;
        if ((e = image(x.buf, bytes, ximg[0])) != CL_SUCCESS) return fail(err, "activations image", e);
        for (int j = 1; j <= 3; ++j) {
            const cl_buffer_region reg{static_cast<size_t>(256 * j), bytes - static_cast<size_t>(256 * j)};
            xsub[j - 1] = cl->clCreateSubBuffer(x.buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &e);
            if (e != CL_SUCCESS) return fail(err, "activations sub-buffer", e);
            if ((e = image(xsub[j - 1], reg.size, ximg[j])) != CL_SUCCESS) return fail(err, "activations image", e);
        }
        x_tokens = want;
        return true;
    }
    bool ensure_y(size_t bytes, std::string &err) {
        if (bytes <= y_bytes) return true;
        release(y);
        y_bytes = 0;
        if (!make(bytes, CL_MEM_WRITE_ONLY, y, err, io_xy)) return false;
        y_bytes = bytes;
        return true;
    }
    static bool fail(std::string &err, const char *what, cl_int e) {
        err = std::string("GPU: ") + what + ": " + cl::error_name(e);
        return false;
    }
};

Gpu::Gpu() : p_(std::make_unique<Impl>()) {}
Gpu::~Gpu() = default;

bool Gpu::ready() const { return p_->kernel[kKindCount - 1] != nullptr; }
size_t Gpu::adopted_bytes() const { return p_->adopted; }
bool Gpu::holds(const QView &m) const { return m.quad && p_->mats.count(m.data) != 0; }
double Gpu::busy_seconds() const { return p_->busy; }
void Gpu::set_parallel(ParallelFor par) { p_->par = std::move(par); }

std::string Gpu::timing() const {
    char buf[200];
    std::snprintf(buf, sizeof buf, "kernels %.0f ms; host: activations %.0f, launch %.0f, wait %.0f, outputs %.0f ms",
                  p_->busy * 1e3, p_->t_prep * 1e3, p_->t_launch * 1e3, p_->t_wait * 1e3, p_->t_out * 1e3);
    return buf;
}
int64_t Gpu::launches() const { return p_->launches; }

bool Gpu::fits(const QView &m) {
    return !m.quad && kind_of(m.type, m.cols) >= 0 && m.rows > 0 && m.rows % kRowsPerGroup == 0 && m.data;
}

bool Gpu::open(std::string &err) {
    Impl &p = *p_;
    if (ready()) return true;
    if (!p.dev.open(err)) return false;
    p.cl = p.dev.cl;
    // the code is a6xx gen 1 machine code: the Adreno 618's registers, waves and ISA
    if (p.dev.version.find("Adreno(TM) 618") == std::string::npos) {
        err = "GPU: the hand-written kernels are for the Adreno 618, not " + p.dev.version;
        return false;
    }
    if (!p.dev.images || !p.dev.fp16) {
        err = "GPU: no images or no FP16";
        return false;
    }
    if (const char *e = std::getenv("GEODE_GPU_MEM")) p.mem = e;
    if (p.mem != "alloc") {
#if defined(__ANDROID__)
        if (!p.dev.has("cl_qcom_ion_host_ptr") || !p.ion.open(err)) {
            if (err.empty()) err = "GPU: no cl_qcom_ion_host_ptr";
            return false;
        }
        cl_uint pad = 0, page = 0;
        p.cl->clGetDeviceInfo(p.dev.id, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof pad, &pad, nullptr);
        p.cl->clGetDeviceInfo(p.dev.id, CL_DEVICE_PAGE_SIZE_QCOM, sizeof page, &page, nullptr);
        p.padding = pad;
        if (page) p.page = page;
#else
        err = "GPU: shared cached memory needs ION (Android)";
        return false;
#endif
    }
    cl::Program tmpl;
    if (!tmpl.build(p.dev, template_source(), "", err)) return false;
    const std::vector<uint8_t> bin = tmpl.binary();
    if (bin.empty()) {
        err = "GPU: no program binary from the driver";
        return false;
    }
    for (int k = 0; k < kKindCount; ++k) {
        std::vector<uint8_t> b = bin;
        if (!inject(b, kKinds[k].code, kKinds[k].n, kGemmFull, kGemmHalf, err)) return false;
        if (!p.prog[k].build_binary(p.dev, b, err)) return false;
        if (!(p.kernel[k] = p.prog[k].kernel("gemm", err))) return false;
    }
    return true;
}

bool Gpu::adopt(QView &m, std::string &err) {
    Impl &p = *p_;
    if (!ready() || !fits(m)) {
        err = "GPU: the matrix does not fit the kernels";
        return false;
    }
    const size_t bytes = static_cast<size_t>(m.rows) * m.row_bytes;
    Impl::Mat mat;
    // + 4 KB of room for a kernel's read ahead past the last rows
    if (!p.make(bytes + 4096, CL_MEM_READ_ONLY, mat.h, err)) return false;
    // written through a map: at the unmap the driver cleans the CPU's caches (ION) or flushes the copy (alloc)
    cl_int e = CL_SUCCESS;
    void *w = p.cl->clEnqueueMapBuffer(p.dev.queue, mat.h.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes, 0,
                                       nullptr, nullptr, &e);
    if (e != CL_SUCCESS) {
        p.release(mat.h);
        return Impl::fail(err, "map weights", e);
    }
    to_quad(m.type, m.rows, m.cols, m.data, static_cast<uint8_t *>(w));
    p.cl->clEnqueueUnmapMemObject(p.dev.queue, mat.h.buf, w, 0, nullptr, nullptr);
    p.cl->clFinish(p.dev.queue);
    if (!mat.h.host) {   // ALLOC_HOST_PTR: mapped again for good, the CPU kernels read through that mapping
        mat.h.host = p.cl->clEnqueueMapBuffer(p.dev.queue, mat.h.buf, CL_TRUE, CL_MAP_READ, 0, bytes, 0, nullptr,
                                              nullptr, &e);
        if (e != CL_SUCCESS) {
            p.release(mat.h);
            return Impl::fail(err, "map weights", e);
        }
    }
    mat.type = m.type;
    mat.rows = m.rows;
    mat.cols = m.cols;
    mat.row_bytes = m.row_bytes;
    mat.global = m.global;
    const auto *host = static_cast<const uint8_t *>(mat.h.host);
    p.mats[host] = mat;
    p.adopted += bytes;
    m.data = host;
    m.quad = true;
    return true;
}

bool Gpu::start(const QView &m, int64_t rows, int n, const float *const *xs, std::string &err) {
    Impl &p = *p_;
    const double t0 = now_seconds();
    const auto it = p.mats.find(m.data);
    if (it == p.mats.end() || p.ev || rows <= 0 || rows % kRowsPerGroup || rows > m.rows || n <= 0) {
        err = "GPU: start on a matrix it does not hold, or while busy";
        return false;
    }
    const Impl::Mat &mat = it->second;
    const int ntok = (n + kTokensPerFiber - 1) / kTokensPerFiber * kTokensPerFiber;
    if (!p.ensure_x(ntok, err) || !p.ensure_y(static_cast<size_t>(ntok) * static_cast<size_t>(rows) * 4, err))
        return false;
    // the activations, FP16 and lane-interleaved (geode-gpu --xprep 6): token t's value (32 grp + lane) * 64 + 16 blk +
    // 4 q + e at ((((((t / 4) * cols / 2048 + grp) * 4 + blk) * 4 + q) * 4 + t % 4) * 32 + lane) * 4 + e; each token
    // scaled by a power of two to a largest value <= 8, so the kernel's FP16 partial sums cannot overflow
    const int cols = static_cast<int>(mat.cols);
    cl_int e = CL_SUCCESS;
    auto *xh = static_cast<uint16_t *>(p.cl->clEnqueueMapBuffer(p.dev.queue, p.x.buf, CL_TRUE,
                                                                CL_MAP_WRITE_INVALIDATE_REGION, 0,
                                                                static_cast<size_t>(ntok) * cols * 2, 0, nullptr,
                                                                nullptr, &e));
    if (e != CL_SUCCESS) return Impl::fail(err, "map activations", e);
    p.scale.assign(static_cast<size_t>(ntok), 1.0f);
    const int nch = cols / 64;
    p.for_tokens(ntok, [&](int t) {
        uint16_t *base = xh + static_cast<size_t>(t / 4) * static_cast<size_t>(cols) * 4 + static_cast<size_t>(t % 4) * 128;
        // chunk ch's 16 groups of 4 values: 512 halves apart from (ch / 32) * 8192 + (ch % 32) * 4
        if (t >= n) {
            for (int ch = 0; ch < nch; ++ch) {
                uint16_t *d = base + static_cast<size_t>(ch / 32) * 8192 + static_cast<size_t>(ch % 32) * 4;
                for (int g = 0; g < 16; ++g) std::memset(d + 512 * g, 0, 8);
            }
            return;
        }
        const float *xt = xs[t];
        const float amax = abs_max(xt, cols);
        float s = 1.0f;
        if (amax > 0.0f && std::isfinite(amax)) s = std::ldexp(1.0f, static_cast<int>(std::ceil(std::log2(amax / 8.0f))));
        p.scale[static_cast<size_t>(t)] = s;
        const float inv = 1.0f / s;
        for (int ch = 0; ch < nch; ++ch) {
            uint16_t *d = base + static_cast<size_t>(ch / 32) * 8192 + static_cast<size_t>(ch % 32) * 4;
            const float *src = xt + 64 * ch;
            for (int g = 0; g < 16; ++g) halves4(src + 4 * g, inv, d + 512 * g);
        }
    });
    p.cl->clEnqueueUnmapMemObject(p.dev.queue, p.x.buf, xh, 0, nullptr, nullptr);
    const double t1 = now_seconds();
    p.t_prep += t1 - t0;
    cl_kernel k = p.kernel[kind_of(mat.type, mat.cols)];
    const int rows_per_group = 128 / (cols / 64) * 4;   // 128 fibers / lanes a row quad, 4 rows each
    const cl_uint w_off = 0;
    const cl_int rows_i = static_cast<cl_int>(rows), cols_i = static_cast<cl_int>(mat.cols),
                 rb_i = static_cast<cl_int>(mat.row_bytes), ntok_i = ntok;
    const cl_float gs = mat.global;
    const cl::Api *cl = p.cl;
    cl->clSetKernelArg(k, 0, sizeof(cl_mem), &mat.h.buf);
    cl->clSetKernelArg(k, 1, sizeof w_off, &w_off);
    cl->clSetKernelArg(k, 2, sizeof rows_i, &rows_i);   // the output's row stride: this call's rows
    cl->clSetKernelArg(k, 3, sizeof cols_i, &cols_i);
    cl->clSetKernelArg(k, 4, sizeof rb_i, &rb_i);
    cl->clSetKernelArg(k, 5, sizeof gs, &gs);
    cl->clSetKernelArg(k, 6, sizeof(cl_mem), &p.x.buf);
    cl->clSetKernelArg(k, 7, sizeof ntok_i, &ntok_i);
    cl->clSetKernelArg(k, 8, sizeof(cl_mem), &p.y.buf);
    // the machine code reads token slot j of a fiber's four through t#2, t#0, t#1, t#3 = arguments 11, 9, 10, 12
    cl->clSetKernelArg(k, 11, sizeof(cl_mem), &p.ximg[0]);
    cl->clSetKernelArg(k, 9, sizeof(cl_mem), &p.ximg[1]);
    cl->clSetKernelArg(k, 10, sizeof(cl_mem), &p.ximg[2]);
    cl->clSetKernelArg(k, 12, sizeof(cl_mem), &p.ximg[3]);
    const size_t global[2] = {static_cast<size_t>(rows / rows_per_group) * kGroupSize, static_cast<size_t>(ntok / 4)};
    const size_t local[2] = {kGroupSize, 1};
    e = cl->clEnqueueNDRangeKernel(p.dev.queue, k, 2, nullptr, global, local, 0, nullptr, &p.ev);
    if (e != CL_SUCCESS) {
        p.ev = nullptr;
        return Impl::fail(err, "launch", e);
    }
    cl->clFlush(p.dev.queue);
    p.t_launch += now_seconds() - t1;
    p.n = n;
    p.ntok = ntok;
    p.rows = rows;
    ++p.launches;
    return true;
}

bool Gpu::finish(float *const *ys, std::string &err, bool spin, double *waited, bool *late) {
    Impl &p = *p_;
    if (!p.ev) {
        err = "GPU: finish without start";
        return false;
    }
    const double t0 = now_seconds();
    bool running = false;
    if (spin) {
        // the kernel's status until it is complete (CL_COMPLETE 0, an error below)
        for (;;) {
            cl_int st = CL_COMPLETE;
            if (p.cl->clGetEventInfo(p.ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, nullptr) != CL_SUCCESS ||
                st <= CL_COMPLETE)
                break;
            running = true;
            for (int i = 0; i < 64; ++i) cpu_relax();
        }
    }
    cl_int e = p.cl->clWaitForEvents(1, &p.ev);
    const double t1 = now_seconds();
    if (waited) *waited = t1 - t0;
    if (late) *late = spin ? running : t1 - t0 > 20e-6;
    p.t_wait += t1 - t0;
    p.busy += cl::event_seconds(p.cl, p.ev);   // releases it
    p.ev = nullptr;
    if (e != CL_SUCCESS) return Impl::fail(err, "kernel", e);
    const size_t rows = static_cast<size_t>(p.rows);
    auto *y = static_cast<const float *>(p.cl->clEnqueueMapBuffer(p.dev.queue, p.y.buf, CL_TRUE, CL_MAP_READ, 0,
                                                                  static_cast<size_t>(p.n) * rows * 4, 0, nullptr,
                                                                  nullptr, &e));
    if (e != CL_SUCCESS) return Impl::fail(err, "map output", e);
    p.for_tokens(p.n, [&](int t) {
        const float s = p.scale[static_cast<size_t>(t)];
        const float *src = y + static_cast<size_t>(t) * rows;
        float *dst = ys[t];
        for (size_t r = 0; r < rows; ++r) dst[r] = src[r] * s;
    });
    p.cl->clEnqueueUnmapMemObject(p.dev.queue, p.y.buf, const_cast<float *>(y), 0, nullptr, nullptr);
    p.t_out += now_seconds() - t1;
    return true;
}

}  // namespace geode
