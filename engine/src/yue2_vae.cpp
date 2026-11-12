// FP32 Oobleck decoder; geometry and crop rules follow yue2/modeling_vae.py.
#include "yue2_vae.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#include "opencl.hpp"

namespace geode::yue2 {
namespace {
// The GPU's kernels: the CPU's arithmetic below (convolution's sums in the order j = input channel x taps + tap, an
// fma each, the bias after; snake's separate multiply, square, scale and add), a work item 4 output channels x 8
// samples; a transposed convolution phase by phase (output q * stride + phase) as the CPU's two-tap convolutions.
const char *kVaeKernels = R"CL(
#ifndef TB
#define TB 8
#endif
// SnakeBeta on 4 channels at a time: y = x + inv_beta sin(alpha x)^2 (the CPU's separate operations)
__kernel void snake4(__global const float4 *x, __global float4 *y, __global const float4 *alpha,
                     __global const float4 *ib, int n) {
#pragma OPENCL FP_CONTRACT OFF
    const int t = get_global_id(0), cg = get_global_id(1);
    if (t >= n) return;
    const size_t at = (size_t)cg * (size_t)n + (size_t)t;
    const float4 v = x[at];
    const float4 a = sin(v * alpha[cg]);
    const float4 sq = a * a;
    const float4 term = ib[cg] * sq;
    y[at] = v + term;
}
static inline float4 xat(__read_only image1d_buffer_t X, int base, int t, int n) {
    const float4 v = read_imagef(X, base + clamp(t, 0, n - 1));
    return (t >= 0 && t < n) ? v : (float4)(0.0f);
}
// An item: 8 out channels (groups g, g + 1 of 4) x TB times. W's texel ((g cgi + cg) k + tap) 4 + il = the 4 out
// channels' weights of input channel 4 cg + il at that tap. mad, not fma: Adreno has no fused fp32 multiply-add (an
// fma is emulated there, ~20x slower); the sums' order the CPU's (input channels, then taps) per 4 inputs.
#define MAC2(u, xv) { \
    a0[u] = mad(wa0, (float4)(xv.x), a0[u]); a0[u] = mad(wa1, (float4)(xv.y), a0[u]); \
    a0[u] = mad(wa2, (float4)(xv.z), a0[u]); a0[u] = mad(wa3, (float4)(xv.w), a0[u]); \
    a1[u] = mad(wb0, (float4)(xv.x), a1[u]); a1[u] = mad(wb1, (float4)(xv.y), a1[u]); \
    a1[u] = mad(wb2, (float4)(xv.z), a1[u]); a1[u] = mad(wb3, (float4)(xv.w), a1[u]); }
#define WLOAD(tap) \
    const int wa = ((g * cgi + cg) * k + (tap)) * 4, wb = ((g1 * cgi + cg) * k + (tap)) * 4; \
    const float4 wa0 = read_imagef(W, wa), wa1 = read_imagef(W, wa + 1), wa2 = read_imagef(W, wa + 2), \
                 wa3 = read_imagef(W, wa + 3), wb0 = read_imagef(W, wb), wb1 = read_imagef(W, wb + 1), \
                 wb2 = read_imagef(W, wb + 2), wb3 = read_imagef(W, wb + 3);
static inline void put4(__global float4 *yr, int t, float4 v, int add) {
    if (add) v = v + yr[t];
    yr[t] = v;
}
__kernel void conv4(__read_only image1d_buffer_t X, int n, int cgi, __read_only image1d_buffer_t W,
                    __global const float4 *b, __global float4 *Y, int m, int k, int dil, int pad, int add, int cgo) {
    const int t0 = get_global_id(0) * TB, g = get_global_id(1) * 2;
    if (t0 >= m || g >= cgo) return;
    const int g1 = min(g + 1, cgo - 1);
    float4 a0[TB], a1[TB];
#pragma unroll
    for (int u = 0; u < TB; ++u) a0[u] = (float4)(0.0f), a1[u] = (float4)(0.0f);
    const bool inside = t0 - pad >= 0 && t0 + TB - 1 - pad + (k - 1) * dil < n;   // (no bounds a read)
    for (int cg = 0; cg < cgi; ++cg) {
        const int xb = cg * n;
        for (int kk = 0; kk < k; ++kk) {
            WLOAD(kk)
            const int s = t0 - pad + kk * dil;
            if (inside) {
#pragma unroll
                for (int u = 0; u < TB; ++u) { const float4 xv = read_imagef(X, xb + s + u); MAC2(u, xv) }
            } else {
#pragma unroll
                for (int u = 0; u < TB; ++u) { const float4 xv = xat(X, xb, s + u, n); MAC2(u, xv) }
            }
        }
    }
    const float4 ba = b[g], bb = b[g1];
    __global float4 *ya = Y + (size_t)g * (size_t)m, *yb = Y + (size_t)g1 * (size_t)m;
    const bool two = g + 1 < cgo;
#pragma unroll
    for (int u = 0; u < TB; ++u)
        if (t0 + u < m) {
            put4(ya, t0 + u, a0[u] + ba, add);
            if (two) put4(yb, t0 + u, a1[u] + bb, add);
        }
}
// conv4 with 16 out channels (groups g .. g + 3) x T4 times an item (the host dispatches T4 = 4): the same sums in
// the same order (the same bits), each X texel read for 16 channels, not 8 - 9% less time on the Adreno 830 (8 times
// x 4 groups spill)
#define T4 4
__kernel void conv4q(__read_only image1d_buffer_t X, int n, int cgi, __read_only image1d_buffer_t W,
                     __global const float4 *b, __global float4 *Y, int m, int k, int dil, int pad, int add, int cgo) {
    const int t0 = get_global_id(0) * T4, g = get_global_id(1) * 4;
    if (t0 >= m || g >= cgo) return;
    int gg[4];
    for (int i = 0; i < 4; ++i) gg[i] = min(g + i, cgo - 1);
    float4 a[4][T4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int u = 0; u < T4; ++u) a[i][u] = (float4)(0.0f);
    const bool inside = t0 - pad >= 0 && t0 + T4 - 1 - pad + (k - 1) * dil < n;
    for (int cg = 0; cg < cgi; ++cg) {
        const int xb = cg * n;
        for (int kk = 0; kk < k; ++kk) {
            const int s = t0 - pad + kk * dil;
            float4 xv[T4];
#pragma unroll
            for (int u = 0; u < T4; ++u) xv[u] = inside ? read_imagef(X, xb + s + u) : xat(X, xb, s + u, n);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int wi = ((gg[i] * cgi + cg) * k + kk) * 4;
                const float4 w0 = read_imagef(W, wi), w1 = read_imagef(W, wi + 1), w2 = read_imagef(W, wi + 2),
                             w3 = read_imagef(W, wi + 3);
#pragma unroll
                for (int u = 0; u < T4; ++u) {
                    a[i][u] = mad(w0, (float4)(xv[u].x), a[i][u]);
                    a[i][u] = mad(w1, (float4)(xv[u].y), a[i][u]);
                    a[i][u] = mad(w2, (float4)(xv[u].z), a[i][u]);
                    a[i][u] = mad(w3, (float4)(xv[u].w), a[i][u]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (g + i >= cgo) break;
        const float4 bb = b[g + i];
        __global float4 *yr = Y + (size_t)(g + i) * (size_t)m;
#pragma unroll
        for (int u = 0; u < T4; ++u)
            if (t0 + u < m) put4(yr, t0 + u, a[i][u] + bb, add);
    }
}
// ConvTranspose1d phase p: output q s + p from inputs q + (p + pad - h) / s, h = (p + pad) % s + t s (W as conv4's,
// the taps by h)
__kernel void convt4(__read_only image1d_buffer_t X, int n, int cgi, __read_only image1d_buffer_t W,
                     __global const float4 *b, __global float4 *Y, int m, int k, int s0, int pad, int cgo) {
    const int q0 = get_global_id(0) * TB, g = get_global_id(1) * 2, p = get_global_id(2);
    const int count = (m - 1 - p) / s0 + 1;
    if (q0 >= count || g >= cgo) return;
    const int g1 = min(g + 1, cgo - 1), taps = k / s0;
    float4 a0[TB], a1[TB];
#pragma unroll
    for (int u = 0; u < TB; ++u) a0[u] = (float4)(0.0f), a1[u] = (float4)(0.0f);
    for (int cg = 0; cg < cgi; ++cg) {
        const int xb = cg * n;
        for (int t = 0; t < taps; ++t) {
            const int h = (p + pad) % s0 + t * s0;
            WLOAD(h)
            const int s = q0 + (p + pad - h) / s0;
#pragma unroll
            for (int u = 0; u < TB; ++u) { const float4 xv = xat(X, xb, s + u, n); MAC2(u, xv) }
        }
    }
    const float4 ba = b[g], bb = b[g1];
    __global float4 *ya = Y + (size_t)g * (size_t)m, *yb = Y + (size_t)g1 * (size_t)m;
    const bool two = g + 1 < cgo;
#pragma unroll
    for (int u = 0; u < TB; ++u)
        if (q0 + u < count) {
            const size_t at = (size_t)(q0 + u) * (size_t)s0 + (size_t)p;
            ya[at] = a0[u] + ba;
            if (two) yb[at] = a1[u] + bb;
        }
}
)CL";
template<class C> int64_t length_of(const C &c, int64_t n) {
    return c.transposed ? (n-1)*c.stride-2*c.padding+c.dilation*(c.k-1)+1
                        : (n+2*c.padding-c.dilation*(c.k-1)-1)/c.stride+1;
}
#if defined(__GNUC__) || defined(__clang__)
using Vec = float __attribute__((vector_size(32)));
#else
struct Vec { float v[8]; };
Vec operator+(Vec a, Vec b) { for(int i=0;i<8;++i) a.v[i]+=b.v[i]; return a; }
Vec operator*(Vec a, Vec b) { for(int i=0;i<8;++i) a.v[i]*=b.v[i]; return a; }
#endif
Vec splat(float x) { Vec v; for(int i=0;i<8;++i) reinterpret_cast<float*>(&v)[i]=x; return v; }
Vec read8(const float *p) { Vec v; std::memcpy(&v,p,32); return v; }
void write8(float *p, Vec v) { std::memcpy(p,&v,32); }

// 4 output channels x 16 times, with K panels fitting in L1. No horizontal
// reductions: each lane is one output sample, accumulated entirely in FP32.
void product(const float *w, const float *x, float *y, int rows, int k) {
    for(int kb=0;kb<k;kb+=128) {
        int ke=std::min(k,kb+128);
        for(int o=0;o<rows;o+=4) for(int t=0;t<64;t+=16) {
            int nr=std::min(4,rows-o);
            Vec a[4], b[4];
            for(int r=0;r<nr;++r) { a[r]=read8(y+(o+r)*64+t); b[r]=read8(y+(o+r)*64+t+8); }
            for(int j=kb;j<ke;++j) {
                Vec u=read8(x+j*64+t), v=read8(x+j*64+t+8);
                for(int r=0;r<nr;++r) {
                    Vec z=splat(w[(o+r)*k+j]);
                    a[r]=a[r]+u*z; b[r]=b[r]+v*z;
                }
            }
            for(int r=0;r<nr;++r) { write8(y+(o+r)*64+t,a[r]); write8(y+(o+r)*64+t+8,b[r]); }
        }
    }
}

template<class C> std::vector<float> convolution(const C &c, const std::vector<float> &x,
                                                int64_t n, ThreadPool &pool) {
    int64_t m=length_of(c,n);
    std::vector<float> y(static_cast<size_t>(c.out)*m);
    int phases=c.transposed ? c.stride : 1;
    int taps=c.transposed ? c.k/c.stride : c.k;
    int k=c.in*taps;
    // A transposed convolution is a set of ordinary two-tap convolutions.
    // For output q*s+phase: input q+(phase+padding-kernel_index)/s.
    std::vector<float> packed;
    if(c.transposed) {
        packed.resize(static_cast<size_t>(phases)*c.out*k);
        for(int p=0;p<phases;++p) for(int o=0;o<c.out;++o)
            for(int i=0;i<c.in;++i) for(int t=0;t<taps;++t) {
                int h=(p+c.padding)%c.stride+t*c.stride;
                packed[(static_cast<size_t>(p)*c.out+o)*k+i*taps+t]=c.w[(static_cast<size_t>(i)*c.out+o)*c.k+h];
            }
    }
    int64_t phase_len=(m+phases-1)/phases, tiles=(phase_len+63)/64;
    pool.parallel_for(tiles*phases,[&](int64_t begin,int64_t end) {
        std::vector<float> panel(static_cast<size_t>(k)*64), result(static_cast<size_t>(c.out)*64);
        for(int64_t task=begin;task<end;++task) {
            int p=static_cast<int>(task/tiles);
            int64_t q=(task%tiles)*64;
            int count=static_cast<int>(std::min<int64_t>(64,(m-1-p)/phases+1-q));
            for(int i=0;i<c.in;++i) for(int t=0;t<taps;++t) {
                int h=c.transposed ? (p+c.padding)%c.stride+t*c.stride : t;
                int64_t start=c.transposed ? q+(p+c.padding-h)/c.stride : q-c.padding+t*c.dilation;
                float *dst=panel.data()+(i*taps+t)*64;
                for(int u=0;u<64;++u) dst[u]=(start+u>=0 && start+u<n) ? x[static_cast<size_t>(i)*n+start+u] : 0.0f;
            }
            // Bias is added after the reduction, like PyTorch's convolution.
            std::fill(result.begin(),result.end(),0.0f);
            const float *w=c.transposed ? packed.data()+static_cast<size_t>(p)*c.out*k : c.w;
            product(w,panel.data(),result.data(),c.out,k);
            for(int o=0;o<c.out;++o) for(int u=0;u<count;++u)
                y[static_cast<size_t>(o)*m+(q+u)*phases+p]=result[o*64+u]+(c.b ? c.b[o] : 0.0f);
        }
    });
    return y;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
template<class S> void snake(const S &s, std::vector<float> &x, int64_t n, ThreadPool &pool) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    pool.parallel_for(static_cast<int64_t>(s.alpha.size())*((n+1023)/1024),[&](int64_t b,int64_t e) {
        int64_t chunks=(n+1023)/1024;
        for(int64_t j=b;j<e;++j) {
            int64_t c=j/chunks, start=(j%chunks)*1024, end=std::min(n,start+1024);
            for(int64_t t=start;t<end;++t) {
                float &v=x[c*n+t];
                float a=std::sin(v*s.alpha[c]);
                float square=a*a;
                float term=s.inv_beta[c]*square;
                // Keep the activation's separate multiply, square, scale and add.
                v=v+term;
            }
        }
    });
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
}

struct Vae::Gpu {
    cl::Device dev;
    cl::Program prog;
    cl_kernel snake = nullptr, conv = nullptr, convt = nullptr, convq = nullptr;
    struct Mem {
        cl_mem buf = nullptr, img = nullptr;
    };
    std::map<const float *, Mem> w;   // a host tensor's GPU copy (convolutions: in conv4's layout, an image over it)
    Mem act[3], lat;
    size_t cap = 0, lat_cap = 0;      // floats each
    const cl::Api *c() const { return dev.cl; }
    void release(Mem &m) {
        if (m.img) c()->clReleaseMemObject(m.img);
        if (m.buf) c()->clReleaseMemObject(m.buf);
        m = Mem{};
    }
    ~Gpu() {
        if (!dev.cl) return;
        c()->clFinish(dev.queue);
        for (auto &kv : w) release(kv.second);
        for (Mem &m : act) release(m);
        release(lat);
    }
    // a buffer of n floats (data: copied in), with an RGBA float image over it when `image`
    bool make(Mem &m, size_t n, const float *data, bool image, std::string &err) {
        if (image && dev.image_max_buffer && n / 4 > dev.image_max_buffer) {
            err = "VAE GPU: an image of " + std::to_string(n / 4) + " texels, the device's limit " +
                  std::to_string(dev.image_max_buffer);
            return false;
        }
        cl_int e = CL_SUCCESS;
        m.buf = c()->clCreateBuffer(dev.context, (data ? CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_WRITE),
                                    n * sizeof(float), const_cast<float *>(data), &e);
        if (!m.buf) {
            err = std::string("VAE GPU: ") + std::to_string(n * 4 >> 20) + " MB: " + cl::error_name(e);
            return false;
        }
        if (!image) return true;
        cl_image_format fmt{CL_RGBA, CL_FLOAT};
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        desc.image_width = n / 4;
        desc.buffer = m.buf;
        m.img = c()->clCreateImage(dev.context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        if (!m.img) {
            err = std::string("VAE GPU: an image of ") + std::to_string(n / 4) + " texels: " + cl::error_name(e);
            release(m);
            return false;
        }
        return true;
    }
    bool upload(const float *key, std::vector<float> data, bool image, std::string &err) {
        if (w.count(key)) return true;
        Mem m;
        if (!make(m, data.size(), data.data(), image, err)) return false;
        w[key] = m;
        return true;
    }
};

Vae::Vae() = default;
Vae::~Vae() = default;

bool Vae::on_gpu() const {
    std::lock_guard<std::mutex> lock(gpu_mu_);
    return gpu_ != nullptr;
}
std::string Vae::gpu_name() const {
    std::lock_guard<std::mutex> lock(gpu_mu_);
    return gpu_ ? gpu_->dev.name : std::string();
}

namespace {
int up4(int c) { return (c + 3) / 4 * 4; }
// per 4 channels (a float4 each), zeros past c
std::vector<float> by4(const float *v, int c) {
    std::vector<float> o(static_cast<size_t>(up4(c)), 0.0f);
    if (v) std::copy(v, v + c, o.begin());
    return o;
}
}  // namespace

bool Vae::open_gpu(std::string &err, bool any) {
    std::lock_guard<std::mutex> lock(gpu_mu_);
    if (gpu_) return true;
    if (blocks_.size() != 6 || !conv_out_.w) {
        err = "VAE GPU: the VAE is not open";
        return false;
    }
    if (const char *e = std::getenv("GEODE_YUE2_VAE_GPU")) {
        if (*e == '0') {
            err = "VAE GPU: off (GEODE_YUE2_VAE_GPU=0)";
            return false;
        }
        any = any || *e == '1';
    }
    auto g = std::make_unique<Gpu>();
    if (!g->dev.open(err)) return false;
    if (!any) {   // the model's number after "Adreno", in the name or the version (as gpu8's check)
        const auto number = [](const std::string &s) {
            const size_t at = s.find("Adreno");
            const size_t d = at == std::string::npos ? std::string::npos : s.find_first_of("0123456789", at);
            return d == std::string::npos ? 0 : std::atoi(s.c_str() + d);
        };
        if (std::max(number(g->dev.name), number(g->dev.version)) < 700) {
            err = "VAE GPU: " + g->dev.name + " (" + g->dev.version + ") - the CPU decodes faster there";
            return false;
        }
    }
    const char *tb = std::getenv("GEODE_YUE2_VAE_TB");
    if (!g->dev.images ||
        !g->prog.build(g->dev, kVaeKernels, std::string("-DTB=") + (tb ? tb : "8"), err))
        return err.empty() ? (err = "VAE GPU: no images", false) : false;
    if (!(g->snake = g->prog.kernel("snake4", err)) || !(g->conv = g->prog.kernel("conv4", err)) ||
        !(g->convq = g->prog.kernel("conv4q", err)) ||
        !(g->convt = g->prog.kernel("convt4", err)))
        return false;
    // a convolution's weights in conv4's layout: texel ((og cgi + cg) k + tap) 4 + il = out channels 4 og + 0..3 of
    // input channel 4 cg + il at that tap (ConvTranspose1d's [in][out][k], Conv1d's [out][in][k])
    const auto up_conv = [&](const Conv &cv) {
        const int cgi = up4(cv.in) / 4, cgo = up4(cv.out) / 4, k = cv.k;
        std::vector<float> wt(static_cast<size_t>(cgo) * cgi * k * 16, 0.0f);
        for (int o = 0; o < cv.out; ++o)
            for (int i = 0; i < cv.in; ++i)
                for (int t = 0; t < k; ++t) {
                    const float v = cv.transposed ? cv.w[(static_cast<size_t>(i) * cv.out + o) * k + t]
                                                  : cv.w[(static_cast<size_t>(o) * cv.in + i) * k + t];
                    const size_t tex = ((static_cast<size_t>(o / 4) * cgi + i / 4) * k + t) * 4 + i % 4;
                    wt[tex * 4 + o % 4] = v;
                }
        return g->upload(cv.w, std::move(wt), true, err) && g->upload(cv.w + 1, by4(cv.b, cv.out), false, err);
    };
    const auto up_snake = [&](const Snake &sn) {
        const int ch = static_cast<int>(sn.alpha.size());
        return g->upload(sn.alpha.data(), by4(sn.alpha.data(), ch), false, err) &&
               g->upload(sn.inv_beta.data(), by4(sn.inv_beta.data(), ch), false, err);
    };
    if (!up_conv(conv_in_) || !up_conv(conv_out_) || !up_snake(snake_out_)) return false;
    for (const Block &bk : blocks_) {
        if (!up_snake(bk.snake) || !up_conv(bk.up)) return false;
        for (const Residual &r : bk.res)
            if (!up_snake(r.s1) || !up_snake(r.s2) || !up_conv(r.c7) || !up_conv(r.c1)) return false;
    }
    gpu_ = std::move(g);
    return true;
}

bool Vae::decode_gpu(const float *latents, int64_t frames, std::vector<float> &out) const {
    Gpu &g = *gpu_;
    const cl::Api *c = g.c();
    std::string err;
    // the largest activation (channels x samples) of the plan: the three buffers' size
    int64_t n = length_of(conv_in_, frames), most = static_cast<int64_t>(up4(conv_in_.out)) * n;
    for (const Block &bk : blocks_) n = length_of(bk.up, n), most = std::max(most, static_cast<int64_t>(up4(bk.up.out)) * n);
    most = std::max<int64_t>(most, static_cast<int64_t>(up4(latent_dim_)) * frames);
    if (most > (int64_t{1} << 31) - 1 || static_cast<uint64_t>(most) * 4 > g.dev.max_alloc ||
        (g.dev.image_max_buffer && static_cast<uint64_t>(most) / 4 > g.dev.image_max_buffer))
        return false;
    if (g.cap < static_cast<size_t>(most)) {
        for (Gpu::Mem &m : g.act) g.release(m);
        g.cap = 0;
        for (Gpu::Mem &m : g.act)
            if (!g.make(m, static_cast<size_t>(most), nullptr, true, err)) {
                for (Gpu::Mem &q : g.act) g.release(q);
                return false;
            }
        g.cap = static_cast<size_t>(most);
    }
    {   // the latents [frame][64] as [64 / 4][frame][4]
        const int cg = up4(latent_dim_) / 4;
        std::vector<float> x(static_cast<size_t>(cg) * frames * 4, 0.0f);
        for (int64_t t = 0; t < frames; ++t)
            for (int ch = 0; ch < latent_dim_; ++ch) {
                const float a = latents[t * latent_dim_ + ch];
                if (!std::isfinite(a)) throw std::invalid_argument("non-finite VAE latent");
                x[(static_cast<size_t>(ch / 4) * frames + t) * 4 + ch % 4] = a;
            }
        if (c->clEnqueueWriteBuffer(g.dev.queue, g.act[2].buf, CL_TRUE, 0, x.size() * sizeof(float), x.data(), 0, nullptr,
                                    nullptr) != CL_SUCCESS)
            return false;
    }
    bool ok = true;
    const auto arg = [&](cl_kernel k, cl_uint i, const auto &v) {
        ok = ok && c->clSetKernelArg(k, i, sizeof v, &v) == CL_SUCCESS;
    };
    static const int prof = [] { const char *e = std::getenv("GEODE_YUE2_VAE_PROF"); return e ? std::atoi(e) : 0; }();
    static const int tb = [] { const char *e = std::getenv("GEODE_YUE2_VAE_TB"); return e ? std::max(1, std::atoi(e)) : 8; }();
    double tk[3] = {0, 0, 0};   // snake, conv, convt
    static const std::pair<size_t, size_t> lsz = [] {
        const char *e = std::getenv("GEODE_YUE2_VAE_LOCAL");
        size_t x = 16, y = 4;   // (the Adreno 830: 16x4 and 8x8 best, 64x1 1.4x slower)
        if (e) std::sscanf(e, "%zux%zu", &x, &y);
        return std::make_pair(std::max<size_t>(1, x), std::max<size_t>(1, y));
    }();
    const auto run = [&](cl_kernel k, size_t g0, size_t g1, size_t g2) {
        const size_t ly = k == g.snake ? 1 : std::min(lsz.second, g1);
        const size_t local[3] = {lsz.first, ly, 1},
                     global[3] = {(g0 + lsz.first - 1) / lsz.first * lsz.first, (g1 + ly - 1) / ly * ly, g2};
        const auto t0 = std::chrono::steady_clock::now();
        ok = ok && c->clEnqueueNDRangeKernel(g.dev.queue, k, g2 > 1 ? 3 : 2, nullptr, global, local, 0, nullptr, nullptr) == CL_SUCCESS;
        if (prof) {
            c->clFinish(g.dev.queue);
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            tk[k == g.snake ? 0 : k == g.conv || k == g.convq ? 1 : 2] += dt;
            if (prof > 1)
                std::fprintf(stderr, "  %s %zu x %zu x %zu: %.1f ms\n", k == g.snake ? "snake" : k == g.conv ? "conv" : "convt",
                             g0, g1, g2, dt * 1e3);
        }
    };
    const auto snake = [&](const Snake &sn, const Gpu::Mem &src, const Gpu::Mem &dst, int ch, int len) {
        arg(g.snake, 0, src.buf), arg(g.snake, 1, dst.buf), arg(g.snake, 2, g.w.at(sn.alpha.data()).buf);
        arg(g.snake, 3, g.w.at(sn.inv_beta.data()).buf), arg(g.snake, 4, len);
        run(g.snake, static_cast<size_t>(len), static_cast<size_t>(up4(ch) / 4), 1);
    };
    const auto conv = [&](const Conv &cv, const Gpu::Mem &src, int len, const Gpu::Mem &dst, bool add) {
        const int m = static_cast<int>(length_of(cv, len)), cgi = up4(cv.in) / 4, a = add ? 1 : 0;
        arg(g.conv, 0, src.img), arg(g.conv, 1, len), arg(g.conv, 2, cgi), arg(g.conv, 3, g.w.at(cv.w).img);
        arg(g.conv, 4, g.w.at(cv.w + 1).buf), arg(g.conv, 5, dst.buf), arg(g.conv, 6, m), arg(g.conv, 7, cv.k);
        arg(g.conv, 8, cv.dilation), arg(g.conv, 9, cv.padding), arg(g.conv, 10, a), arg(g.conv, 11, up4(cv.out) / 4);
        // conv4q, T4 times an item (GEODE_YUE2_VAE_Q=0: conv4, 8 channels x TB times; the same bits)
        static const bool useq = [] { const char *e = std::getenv("GEODE_YUE2_VAE_Q"); return !e || *e != '0'; }();
        constexpr int q4 = 4;   // (the kernel's T4)
        if (useq) {
            const cl_kernel kq = g.convq;
            arg(kq, 0, src.img), arg(kq, 1, len), arg(kq, 2, cgi), arg(kq, 3, g.w.at(cv.w).img);
            arg(kq, 4, g.w.at(cv.w + 1).buf), arg(kq, 5, dst.buf), arg(kq, 6, m), arg(kq, 7, cv.k);
            arg(kq, 8, cv.dilation), arg(kq, 9, cv.padding), arg(kq, 10, a), arg(kq, 11, up4(cv.out) / 4);
            run(kq, static_cast<size_t>((m + q4 - 1) / q4), static_cast<size_t>((up4(cv.out) / 4 + 3) / 4), 1);
            return m;
        }
        run(g.conv, static_cast<size_t>((m + tb - 1) / tb), static_cast<size_t>((up4(cv.out) / 4 + 1) / 2), 1);
        return m;
    };
    const auto convt = [&](const Conv &cv, const Gpu::Mem &src, int len, const Gpu::Mem &dst) {
        const int m = static_cast<int>(length_of(cv, len)), per = (m + cv.stride - 1) / cv.stride, cgi = up4(cv.in) / 4;
        arg(g.convt, 0, src.img), arg(g.convt, 1, len), arg(g.convt, 2, cgi), arg(g.convt, 3, g.w.at(cv.w).img);
        arg(g.convt, 4, g.w.at(cv.w + 1).buf), arg(g.convt, 5, dst.buf), arg(g.convt, 6, m), arg(g.convt, 7, cv.k);
        arg(g.convt, 8, cv.stride), arg(g.convt, 9, cv.padding), arg(g.convt, 10, up4(cv.out) / 4);
        run(g.convt, static_cast<size_t>((per + tb - 1) / tb), static_cast<size_t>((up4(cv.out) / 4 + 1) / 2),
            static_cast<size_t>(cv.stride));
        return m;
    };
    const Gpu::Mem *x = &g.act[0], *y = &g.act[1], *z = &g.act[2];   // (z holds the latents first)
    int len = conv(conv_in_, *z, static_cast<int>(frames), *x, false);
    for (const Block &bk : blocks_) {
        snake(bk.snake, *x, *x, bk.up.in, len);
        len = convt(bk.up, *x, len, *y);
        std::swap(x, y);
        for (const Residual &r : bk.res) {   // x = conv1(snake(conv7(snake(x)))) + x
            snake(r.s1, *x, *y, r.c7.in, len);
            conv(r.c7, *y, len, *z, false);
            snake(r.s2, *z, *z, r.c1.in, len);
            conv(r.c1, *z, len, *x, true);
        }
    }
    snake(snake_out_, *x, *x, conv_out_.in, len);
    const int total = conv(conv_out_, *x, len, *y, false);
    if (prof) std::fprintf(stderr, "VAE GPU: snake %.2f conv %.2f convt %.2f s\n", tk[0], tk[1], tk[2]);
    if (!ok) return false;
    std::vector<float> y4(static_cast<size_t>(total) * 4);   // [1][total][4]: the channels 0, 1
    if (c->clEnqueueReadBuffer(g.dev.queue, y->buf, CL_TRUE, 0, y4.size() * sizeof(float), y4.data(), 0, nullptr, nullptr) !=
        CL_SUCCESS)
        return false;
    out.resize(static_cast<size_t>(out_channels_) * total);
    for (int ch = 0; ch < out_channels_; ++ch)
        for (int64_t t = 0; t < total; ++t) out[static_cast<size_t>(ch) * total + t] = y4[static_cast<size_t>(t) * 4 + ch];
    return true;
}

bool Vae::open(Pack &pack, std::string &err, bool map) {
    {
        std::lock_guard<std::mutex> lock(gpu_mu_);
        gpu_.reset();
    }
    err.clear(); blocks_.clear(); conv_in_=Conv{}; conv_out_=Conv{};
    const auto &v=pack.config()["vae"];
    auto integer=[&](const char *key,int &dst) { const auto &a=v[key]; if(!a.is_number() || a.number<1 || a.number>65536 || std::floor(a.number)!=a.number) return false; dst=static_cast<int>(a.number); return true; };
    int channels=0;
    if(!integer("latent_dim",latent_dim_) || !integer("channels",channels) ||
       !integer("out_channels",out_channels_) || !integer("downsampling_ratio",ratio_) ||
       !integer("decode_core_frames",core_) || !v["decode_halo_frames"].is_number() ||
       v["decode_halo_frames"].number<0 || v["decode_halo_frames"].number>65536 ||
       v["decode_halo_frames"].number!=std::floor(v["decode_halo_frames"].number)) {
        err="invalid VAE geometry"; return false;
    }
    halo_=static_cast<int>(v["decode_halo_frames"].number);
    // This implementation supports the audited release's architecture only.
    const int mults[]={1,2,4,8,16,32}, strides[]={2,2,4,4,5,6}, dilations[]={1,3,9};
    auto matches=[&](const char *key,const int *a,int n) { if(v[key].size()!=static_cast<size_t>(n)) return false; for(int i=0;i<n;++i) if(v[key].at(i).number!=a[i]) return false; return true; };
    if(latent_dim_!=64 || channels!=64 || out_channels_!=2 || ratio_!=1920 ||
       !matches("c_mults",mults,6) || !matches("strides",strides,6) || !matches("dilations",dilations,3) ||
       v["final_tanh"].kind!=json::Value::Kind::Bool || v["final_tanh"].boolean ||
       v["snake_eps"].f64()!=1e-9 || v["sample_rate"].i64()!=48000 || halo_<16) {
        err="unsupported VAE configuration"; return false;
    }
    if(!(map ? pack.map("vae.bin",err) : pack.load("vae.bin",err))) return false;
    auto conv=[&](Conv &c,const std::string &name,int in,int out,int k,int s,int d,bool up,bool bias) {
        c.in=in; c.out=out; c.k=k; c.stride=s; c.dilation=d; c.transposed=up; c.padding=up ? (s+1)/2 : d*(k-1)/2;
        c.w=pack.f32(name+".weight",up ? std::vector<int64_t>{in,out,k} : std::vector<int64_t>{out,in,k},err);
        if(!c.w) return false;
        c.b=bias ? pack.f32(name+".bias",{out},err) : nullptr;
        return !bias || c.b;
    };
    auto activation=[&](Snake &s,const std::string &name,int c) {
        const float *a=pack.f32(name+".alpha",{c},err); if(!a) return false;
        const float *b=pack.f32(name+".beta",{c},err); if(!b) return false;
        s.alpha.resize(c); s.inv_beta.resize(c);
        for(int i=0;i<c;++i) { s.alpha[i]=std::exp(a[i]); s.inv_beta[i]=1.0f/(std::exp(b[i])+1e-9f); }
        return true;
    };
    const std::string prefix="vae.decoder.layers.";
    if(!conv(conv_in_,prefix+"0",latent_dim_,2048,7,1,1,false,true)) return false;
    blocks_.resize(6);
    int in=2048;
    for(int i=0;i<6;++i) {
        auto &b=blocks_[i]; int out=channels*(i==5 ? 1 : mults[4-i]); int s=strides[5-i];
        std::string p=prefix+std::to_string(i+1)+".layers.";
        if(!activation(b.snake,p+"0",in) || !conv(b.up,p+"1",in,out,2*s,s,1,true,true)) return false;
        for(int j=0;j<3;++j) {
            auto &r=b.res[j]; std::string q=p+std::to_string(j+2)+".layers.";
            if(!activation(r.s1,q+"0",out) || !conv(r.c7,q+"1",out,out,7,1,dilations[j],false,true) ||
               !activation(r.s2,q+"2",out) || !conv(r.c1,q+"3",out,out,1,1,1,false,true)) return false;
        }
        in=out;
    }
    return activation(snake_out_,prefix+"7",channels) && conv(conv_out_,prefix+"8",channels,out_channels_,7,1,1,false,false);
}

int64_t Vae::natural_length(int64_t frames) const {
    if(frames<1 || frames>(std::numeric_limits<int64_t>::max()-1024)/ratio_) throw std::invalid_argument("invalid VAE frame count");
    if(blocks_.size()!=6 || !conv_out_.w) throw std::logic_error("VAE is not open");
    int64_t n=length_of(conv_in_,frames);
    for(const auto &b:blocks_) n=length_of(b.up,n);
    return length_of(conv_out_,n);
}

void Vae::decode(const float *latents,int64_t frames,std::vector<float> &out,ThreadPool &pool,const LayerHook &hook) const {
    natural_length(frames);
    if(!latents) throw std::invalid_argument("null VAE latents");
    if(!hook) {
        std::lock_guard<std::mutex> lock(gpu_mu_);
        if(gpu_) {
            if(decode_gpu(latents,frames,out)) return;
            gpu_.reset();   // (its memory, a driver's failure: the CPU from now on)
        }
    }
    std::vector<float> x(static_cast<size_t>(latent_dim_)*frames);
    for(int c=0;c<latent_dim_;++c) for(int64_t t=0;t<frames;++t) {
        float a=latents[t*latent_dim_+c]; if(!std::isfinite(a)) throw std::invalid_argument("non-finite VAE latent"); x[c*frames+t]=a;
    }
    int64_t n=frames; int channels=conv_in_.out;
    auto emit=[&](const std::string &name) { if(hook) hook(name,channels,n,x.data()); };
    x=convolution(conv_in_,x,n,pool); n=length_of(conv_in_,n); emit("top0");
    for(size_t i=0;i<blocks_.size();++i) {
        const auto &b=blocks_[i]; std::string top="top"+std::to_string(i+1);
        snake(b.snake,x,n,pool); emit(top+"_sub0");
        x=convolution(b.up,x,n,pool); n=length_of(b.up,n); channels=b.up.out; emit(top+"_sub1");
        for(int j=0;j<3;++j) {
            const auto &r=b.res[j]; std::vector<float> y=x;
            snake(r.s1,y,n,pool); y=convolution(r.c7,y,n,pool);
            snake(r.s2,y,n,pool); y=convolution(r.c1,y,n,pool);
            pool.parallel_for(static_cast<int64_t>(x.size()),[&](int64_t a,int64_t e) { for(;a<e;++a) x[a]=y[a]+x[a]; },4096);
            emit(top+"_sub"+std::to_string(j+2));
        }
        emit(top);
    }
    snake(snake_out_,x,n,pool); emit("top7");
    x=convolution(conv_out_,x,n,pool); n=length_of(conv_out_,n); channels=out_channels_; emit("top8");
    // The Python sequential ends with Identity when final_tanh is false.
    emit("top9"); out=std::move(x);
}

bool Vae::decode_tiled(const float *latents,int64_t frames,std::vector<float> &audio,ThreadPool &pool,
                     const std::function<void(int,int)> &progress, const std::function<bool()> &cancelled) const {
    int64_t total=natural_length(frames);
    if(!latents) throw std::invalid_argument("null VAE latents");
    audio.resize(static_cast<size_t>(total)*out_channels_);
    int tiles=static_cast<int>((frames+core_-1)/core_);
    for(int tile=0;tile<tiles;++tile) {
        if(cancelled && cancelled()) return false;
        int64_t start=static_cast<int64_t>(tile)*core_, end=std::min(frames,start+core_);
        int64_t left=std::max<int64_t>(0,start-halo_), right=std::min(frames,end+halo_);
        std::vector<float> wave; decode(latents+left*latent_dim_,right-left,wave,pool);
        int64_t n=natural_length(right-left), crop=(start-left)*ratio_;
        int64_t first=start*ratio_, count=std::min(end*ratio_,total)-first;
        if(crop+count>n) throw std::runtime_error("VAE tile does not cover output core");
        for(int64_t t=0;t<count;++t) for(int c=0;c<out_channels_;++c) {
            const float s=wave[c*n+crop+t];   // the pipeline rejects non-finite audio: never clamp an inf to +-1
            if(!std::isfinite(s)) throw std::runtime_error("VAE produced non-finite audio");
            audio[(first+t)*out_channels_+c]=std::clamp(s,-1.0f,1.0f);
        }
        if(progress) progress(tile+1,tiles);
    }
    return true;
}
} // namespace geode::yue2
