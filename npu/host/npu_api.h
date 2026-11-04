// npu/host/npu_api.h - libgeode_npu.so's interface for the engine, which opens it at run time (dlopen): the NPU's
// products without any Hexagon SDK type, so the engine builds and runs where neither exists.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gnpu gnpu;

// Geode's cDSP module in an unsigned PD; skel_dir: where libgeode_npu_skel.so is (ADSP_LIBRARY_PATH gets it first).
// NULL with the reason in err when the NPU is out of reach.
gnpu *gnpu_open(const char *skel_dir, char *err, int err_len);
// FastRPC's answer about the cDSP (remote.h's remote_dsp_attributes), 0xFFFFFFFF when it gives none
unsigned gnpu_capability(int attribute);
// the cDSP's Hexagon version (79 for v79), 0 when FastRPC does not say
int gnpu_arch(void);
// the skel built for arch or the newest older one (npu/build.py ARCHS), 0 when the DSP is older than any
int gnpu_skel_arch(int arch);
// the module's URI on the cDSP for that skel (libgeode_npu_skel_vNN.so)
int gnpu_uri(char *uri, int len, int skel_arch);
// ADSP_LIBRARY_PATH: skel_dir first, then the vendor's folders (before the first FastRPC call)
void gnpu_library_path(const char *skel_dir);
// The device report's NPU check (report.c): every step guarded and timed, as JSON into out; step_file gets each
// step's name before it runs (a DSP fault that takes the process down leaves it there). 0 when written.
int gnpu_report(const char *skel_dir, const char *step_file, char *out, int out_len);
void gnpu_close(gnpu *g);
// A q4 matrix in the pack's layout (rows x cols, each row: cols / 2 code bytes, then cols / 16 E4M3 scale bytes;
// level = 2c + 1 for c < 8, -(2(c - 8) + 1) above) into the NPU's hmx4 blocks, in memory mapped into the DSP once.
// rows and cols multiples of 32. Its id (>= 0), or -1.
int gnpu_adopt_q4(gnpu *g, const uint8_t *data, int64_t rows, int64_t cols, size_t row_bytes, float global);
// The same for Gemma 4's q4i (level = code - 8, signed E4M3 scales; kernels.hpp DType::Q4I), in the pack's row layout.
int gnpu_adopt_q4i(gnpu *g, const uint8_t *data, int64_t rows, int64_t cols, size_t row_bytes, float global);
// One copy instead (npu/dsp/wsrc.h): a q4i matrix the CPU keeps in tile8 (kernels.hpp to_tile8) inside a
// gnpu_shared_alloc buffer (data = its first row tile, at byte off of the buffer fd) taken as it is - nothing copied,
// nothing more allocated; inputs col0 .. col0 + cols of it (a deep matrix in parts: one id a part), kfull its inputs.
// The id, as gnpu_adopt_q4i's, or -1 (v73 / v75: unsupported; the caller makes the hmx4 copy then).
int gnpu_adopt_t8(gnpu *g, const uint8_t *data, int fd, uint64_t off, int64_t rows, int64_t kfull, int64_t col0,
                  int64_t cols, float global);
// ys[t] = W xs[t] for t < n (fp32 rows of cols / rows values): the activations go over as fp16. 0 when done.
int gnpu_gemm(gnpu *g, int id, int n, const float *const *xs, float *const *ys);
// Weights the host turns into HMX's fp16 blocks itself, a matrix at a time (the vision encoder's q8 layers, read
// for one image): the bytes a rows x cols matrix takes (0: rows, cols not multiples of 32); a q8 matrix in the pack's
// row layout (cols int8 codes, then cols / 16 E4M3 scales; value = code x scale x global) into dst (128-byte
// aligned, gnpu_f16_bytes of it) with *shift the scale its values carry (2^shift: the largest in [0.5, 1)), 0 when
// done; and ys[t] = W xs[t] for such a matrix at byte off of the gnpu_shared_alloc buffer fd.
size_t gnpu_f16_bytes(int64_t rows, int64_t cols);
int gnpu_pack_f16(const uint8_t *data, int64_t rows, int64_t cols, size_t row_bytes, float global, uint8_t *dst,
                  int *shift);
int gnpu_gemm_f16(gnpu *g, int fd, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                  float *const *ys);
// the same with clip (or NULL): xs clamped to [clip 0, clip 1] as they are converted, the results to [clip 2, clip 3]
// (Gemma4ClippableLinear, with no passes of its own over the rows)
int gnpu_gemm_f16c(gnpu *g, int fd, uint64_t off, int64_t rows, int64_t cols, int shift, int n, const float *const *xs,
                   float *const *ys, const float *clip);
// ys[j][t] = W_ids[j] xs[t] for count (<= 4) matrices of the same depth: the input converted once, each matrix's
// results converted while the DSP multiplies the next. 0 when done.
int gnpu_gemm_multi(gnpu *g, int count, const int *ids, int n, const float *const *xs, float *const *const *ys);
// An MLP's products in one go: ys[t] = down(act(gate xs[t]) * (up xs[t])), down adopted in n_down parts of its
// inputs (downs: each part's id, their results added). The input converted once; act(gate) * up stays on the DSP as
// down's input (fp16), only down's results come back. act 0: gelu (tanh form), 1: silu. 0 when done; the DSP's error
// (AEE_EUNSUPPORTED: older than v79) or -1 otherwise, ys then as they were or partly written.
int gnpu_mlp(gnpu *g, int gate, int up, int n_down, const int *downs, int act, int n, const float *const *xs,
             float *const *ys);
// The conversions' threads: par(ctx, items, fn, arg) runs fn(arg, i0, i1) over [0, items) on the caller's threads
// and returns when all are done (the engine's pool). NULL: the calling thread alone.
typedef void (*gnpu_par_fn)(void *ctx, int items, void (*fn)(void *arg, int i0, int i1), void *arg);
void gnpu_set_par(gnpu *g, gnpu_par_fn par, void *ctx);
// seconds spent converting (activations in, results out) over every gemm so far
double gnpu_convert_seconds(const gnpu *g);
// bytes of memory the adopted matrices hold; the DSP's microseconds over every gemm so far
size_t gnpu_bytes(const gnpu *g);
double gnpu_dsp_seconds(const gnpu *g);

// ---- the decoder (npu/dsp/dec.c, npu/dec_cfg.h): a dense Qwen3.5's whole decode step on the DSP
// Memory both the CPU and the DSP use, mapped into the DSP once (cached: the CPU computes on it too; the DSP sees its
// writes through IO coherence). NULL when there is none; *fd its handle for the calls below. Freed by gnpu_close.
void *gnpu_shared_alloc(gnpu *g, size_t bytes, int cached, int *fd);
// the scratch S needs after the states for cfg (a dec_cfg_t), or -1
long long gnpu_dec_size(gnpu *g, const void *cfg, int cfg_len);
// the decoder over cfg and the buffers W, P, S (gnpu_shared_alloc's fds), on `threads` DSP threads
int gnpu_dec_open(gnpu *g, const void *cfg, int cfg_len, int w_fd, int p_fd, int s_fd, int threads);
// the KV cache's chunk `chunk` is the shared buffer fd
int gnpu_dec_kv(gnpu *g, int chunk, int fd);
// n tokens (1, or a token and its draft) at positions from pos (x: their embeddings, n x hidden floats; p3: three
// M-RoPE positions each) through the model on the DSP; the logits and the hidden rows stay in S (dec_cfg.h). flags:
// DEC_SET1, DEC_DEBUG_*. *dsp_s the DSP's time. 0 when done
int gnpu_dec_step(gnpu *g, const float *x, int n, int hidden, int pos, const int *p3, int flags, double *dsp_s);
// up to DEC_N tokens of a prompt (x: n x hidden floats) at positions from pos on the DSP, the products on HMX: the
// last one's logits and every token's hidden row in S. 0 when done
int gnpu_dec_prefill(gnpu *g, const float *x, int n, int hidden, int pos, const int *p3, int flags, double *dsp_s);
// the MTP layer's K / V for up to DEC_N positions of a prompt from pos on the DSP (e: the next tokens' embeddings, h:
// the hidden rows, n x hidden floats each), the products on HMX. 0 when done
int gnpu_dec_mtp_kv(gnpu *g, const float *e, const float *h, int n, int hidden, int pos, const int *p3, int flags,
                    double *dsp_s);
// A prompt's routed experts on HMX (npu/moe_cfg.h): desc as moe_cfg.h lays it out (the experts' buffers by the fds
// gnpu_shared_alloc gave), io_fd one more such buffer holding x and y. *dsp_s the DSP's time. 0 when done
int gnpu_moe(gnpu *g, const void *desc, int desc_len, int io_fd, double *dsp_s);
// ---- a prompt's attention on HMX (npu/attn_cfg.h, npu/dsp/attn.c)
// gnpu_attn's queries (which 0) or output (1): rpcmem of at least `bytes`, kept and grown (freed by gnpu_close)
float *gnpu_attn_buffer(gnpu *g, int which, size_t bytes);
// One layer's attention for a prompt's chunk: desc an attn_desc_t (the KV cache's chunks by gnpu_shared_alloc's
// fds), q and o in gnpu_attn_buffer's buffers. 0 when done; the DSP's error otherwise (AEE_EUNSUPPORTED: older than
// v79, or lanes ordered otherwise than v79's). *dsp_s the DSP's time
int gnpu_attn(gnpu *g, const void *desc, int desc_len, const float *q, size_t q_bytes, float *o, size_t o_bytes,
              double *dsp_s);
// ---- Gemma 4's layers on the DSP (npu/gdec_cfg.h, npu/dsp/gemma.c gdec_*): a prompt's chunk through every layer
// cfg a gdec_cfg_t (the matrices by gnpu_mat_hmx's fds), P and IO gnpu_shared_alloc buffers; 0 when it opened
int gnpu_gdec_open(gnpu *g, const void *cfg, int cfg_len, int p_fd, int io_fd);
// the KV cache's chunk `chunk` is the shared buffer fd
int gnpu_gdec_kv(gnpu *g, int chunk, int fd);
// the n tokens IO holds (x, the per-layer inputs, the rope tables) at positions from apos through every layer; x left
// in IO. 0 when done; *dsp_s the DSP's time
int gnpu_gdec_prefill(gnpu *g, int n, int apos, int flags, double *dsp_s);
void gnpu_gdec_close(gnpu *g);
// an adopted matrix's hmx4 copy: its rpcmem fd and scale shift (dec_mat_t's hmx_fd / hmx_shift); 0 when known
int gnpu_mat_hmx(const gnpu *g, int id, int *fd, int *shift, long long *kp);
// the MTP layer for n positions from pos (e: the next tokens' embeddings, h: the hidden rows): the last one's draft
// and its probability
int gnpu_dec_draft(gnpu *g, const float *e, const float *h, int n, int hidden, int pos, const int *p3, int *draft,
                   float *prob, double *dsp_s);

#ifdef __cplusplus
}
#endif
