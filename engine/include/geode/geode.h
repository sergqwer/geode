/* geode.h - the Geode engine's C API (PLAN.md, "Design of the engine and the app").
 *
 * The engine works on token ids: tokenization lives in the app (Kotlin) and on the PC in Python. A model is a
 * directory in the geode-pack format (geode.json + tokenizer.bin + dense.bin + experts-NN.bin + embed.bin).
 *
 * Threading: one geode_model is driven by one thread at a time (open, generate, reset, close); geode_stop may be
 * called from any thread while geode_generate runs.
 */
#ifndef GEODE_H
#define GEODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GEODE_OK = 0,
    GEODE_E_ARG = 1,              /* a bad argument */
    GEODE_E_IO = 2,               /* a file is missing, unreadable or the wrong size */
    GEODE_E_FORMAT = 3,           /* not a geode-pack, or a version / architecture this engine does not know */
    GEODE_E_MEMORY = 4,           /* the model does not fit the memory budget */
    GEODE_E_NOT_IMPLEMENTED = 5,  /* the operation exists in the API but not yet in this build */
    GEODE_E_STOPPED = 6,          /* geode_stop interrupted the call */
};

typedef struct geode_model geode_model;

typedef struct {
    const char *model_dir;        /* the model package directory */
    uint64_t memory_budget;       /* bytes the engine may hold in all (dense weights, KV cache, expert cache) */
    int32_t compute_threads;      /* 0 = choose (the big cores) */
    int32_t io_threads;           /* 0 = choose (flash readers) */
    int32_t context_length;       /* tokens; 0 = the model's default */
    int32_t helper_threads;       /* little-core threads that join a prompt's products; 0 = default, -1 = none */
    int32_t gpu;                  /* 1 = a prompt's big dense products shared with the GPU where it can (the Adreno
                                     618, q2 / q4 dense weights); 0 = the CPU alone */
    float gpu_share;              /* the GPU's part of those products' rows; 0 = the engine's default */
    int32_t no_prefetch;          /* 1 = no reads ahead of the next layer's predicted experts (0 = read ahead) */
    int32_t experts_per_token;    /* routed experts a token takes, 1 .. the model's top_k; 0 = top_k (geode_set_experts) */
    float route_prior;            /* cache-conditioned routing (upstream Strata 0.2.7): while generating, an expert in
                                     the expert cache gets this added to its router logit for the top-k selection only
                                     (the weights stay the router's); fewer flash reads when the cache cannot hold
                                     every expert, answers change a little. 0 = off (the default); 1.5 = Strata's */
    int32_t tail_skip;            /* generating: a routed expert ranked this or lower (0-based, of top_k) that is
                                     neither cached nor being read is left out instead of read - faster where the
                                     model does not fit in memory, LESS ACCURATE answers. 0 = never: every expert
                                     the router picks (the default); 6 or 4 of the 8 (GEODE_TAIL_SKIP overrides 0) */
    int32_t npu;                  /* 1 = a prompt's dense products on the NPU (Hexagon v73+, Geode's module in an
                                     unsigned PD) where it opens and passes its check, else the path as without it;
                                     0 = never. GEODE_NPU overrides (0 / 1) */
} geode_config;

typedef struct {
    float temperature;            /* 0 = greedy */
    float top_p;                  /* 1 = off */
    int32_t top_k;                /* 0 = off */
    uint64_t seed;
    int32_t max_tokens;           /* the most tokens to generate */
    const int32_t *stop_tokens;   /* generation ends after any of these (it is still passed to the callback) */
    int32_t n_stop_tokens;
    float presence_penalty;       /* subtracted from the logit of every token this call has generated; 0 = off
                                     (Qwen3.6 recommends 1.5: it keeps long reasoning from going round in circles) */
} geode_sampling;

typedef struct {
    uint64_t memory_used;         /* bytes the engine holds now */
    uint64_t expert_cache_bytes;
    uint64_t expert_hits, expert_misses;
    uint64_t flash_bytes_read;
    double prompt_tokens_per_s, decode_tokens_per_s;
    /* context_length: the longest context the memory budget allows (at most the one asked for);
       context_allocated: the positions whose KV cache is held now; it grows 1024 at a time, each step taken from the
       expert cache, and geode_reset gives it all back */
    int32_t context_used, context_length, context_allocated;
    /* of expert_hits: the picks a read ahead had brought (or was bringing); and the reads ahead made. So
       expert_hits - expert_ahead_hits were in the cache already, expert_ahead_hits / expert_reads_ahead is how often
       the prediction was right */
    uint64_t expert_ahead_hits, expert_reads_ahead;
    int32_t experts_per_token;    /* the routed experts a token takes now (geode_set_experts) */
} geode_stats;

/* Called with each generated token; return nonzero to stop. */
typedef int (*geode_token_cb)(void *user, int32_t token);

const char *geode_version(void);

/* Opens a model. On failure returns an error code and writes a message into err (if err_len > 0). */
int geode_open(const geode_config *config, geode_model **out, char *err, size_t err_len);

/* Appends the prompt tokens to the conversation, then generates until a stop token, max_tokens, the context's end,
 * the callback asking to stop, or geode_stop.
 *
 * The conversation only grows (the Gated DeltaNet layers keep a running state that cannot be rewound): every token
 * passed to the callback becomes part of it, except a stop token. So after an answer the next prompt starts with
 * the template's end-of-turn (<|im_end|>\n<|im_start|>user\n...). The last token is computed lazily, at the start
 * of the next call, so stopping costs nothing. To edit or regenerate, geode_reset and send the whole conversation. */
int geode_generate(geode_model *model, const int32_t *prompt, int32_t n_prompt, const geode_sampling *sampling,
                   geode_token_cb callback, void *user, char *err, size_t err_len);

/* Images (a pack with a vision encoder: tools/pack_vision.py). geode_image_size gives the size an image of w x h is
 * resized to before geode_add_image (Qwen's smart_resize: each side a multiple of 32, about the aspect kept, at most
 * max_pixels pixels; 0 = the pack's limit) and the image_pad tokens it then takes. GEODE_E_NOT_IMPLEMENTED: the
 * pack has no vision encoder. */
int geode_image_size(geode_model *model, int32_t w, int32_t h, int32_t max_pixels, int32_t *out_w, int32_t *out_h,
                     int32_t *n_tokens);
/* Encodes an image now, for the conversation's next image (a few seconds: the encoder is read from flash a block at
 * a time and let go after, so it costs no memory between images): rgb is h rows of w pixels, 3 bytes each (R, G,
 * B), w x h a size geode_image_size gave. A following prompt holds it as <|vision_start|>, n_tokens times
 * <|image_pad|>, <|vision_end|>; the images go in in the order they were added. geode_reset forgets them (a
 * conversation sent again needs its images added again). cache_path (NULL: none): a file of the image's features,
 * read instead of encoding when it is this image's at this size from this vision encoder, else written after the
 * encode - the caller names one file per image and size. */
int geode_add_image(geode_model *model, const uint8_t *rgb, int32_t w, int32_t h, const char *cache_path,
                    int32_t *n_tokens, char *err, size_t err_len);

/* Video (Gemma 4): a video is its frames, each one through the vision encoder like an image but at a smaller size
 * (geode_image_size with max_pixels = 70 x 48 x 48: at most 70 tokens a frame) and with its own token. The app samples
 * the frames (about one a second, at most 32) and writes the prompt: per frame "MM:SS <|image>", n_tokens times
 * <|video|>, "<image|>", the frames joined by spaces (Gemma4Processor's layout); the soundtrack, if wanted, as a clip
 * (geode_add_audio) after them. geode_add_video_frame encodes one frame now, as geode_add_image does; frames,
 * images and clips go in in the order they were added. GEODE_E_NOT_IMPLEMENTED: the pack takes no video. */
int geode_add_video_frame(geode_model *model, const uint8_t *rgb, int32_t w, int32_t h, const char *cache_path,
                          int32_t *n_tokens, char *err, size_t err_len);

/* Audio (a Gemma 4 pack with an audio encoder: tools/pack_gemma4_audio.py). geode_audio_tokens: the <|audio|>
 * tokens a clip of n_samples at 16 kHz takes - one per 40 ms, at most 30 s of it (750) - or 0 when the pack has no
 * audio encoder. */
int32_t geode_audio_tokens(geode_model *model, int64_t n_samples);
/* Encodes a clip now, for the conversation's next audio (the encoder is read from flash a layer at a time and let
 * go after): pcm is n_samples floats, 16 kHz mono in [-1, 1] (the first 30 s of a longer clip). A following prompt
 * holds it as <|audio>, n_tokens times <|audio|>, <audio|>; images and clips go in in the order they were added.
 * geode_reset forgets them. GEODE_E_NOT_IMPLEMENTED: no audio encoder; GEODE_E_ARG: too short a clip. */
int geode_add_audio(geode_model *model, const float *pcm, int64_t n_samples, int32_t *n_tokens, char *err,
                    size_t err_len);

/* Encodes an image into its feature file (geode_add_image's cache_path) and nothing more - no prompt takes it: an
 * app does it while the question is being typed, and geode_add_image then reads the file instead of encoding. Does
 * nothing when the file holds this image already. */
int geode_cache_image(geode_model *model, const uint8_t *rgb, int32_t w, int32_t h, const char *cache_path, char *err,
                      size_t err_len);

/* Appends the prompt tokens to the conversation and generates nothing (geode_generate without its answer): a common
 * beginning (a system prompt and its tools) read once, saved (geode_state_save), and loaded to start conversations
 * from. Images are not taken here. */
int geode_prefill(geode_model *model, const int32_t *prompt, int32_t n_prompt, char *err, size_t err_len);

/* A conversation kept on flash: instead of reading it again (minutes for a long one on a phone) its state is read
 * back (a fraction of a second). Between calls only - never while geode_generate or geode_prefill runs.
 *
 * geode_state_save writes everything the next call continues from to path: the Gated DeltaNet states (~63 MB for
 * Qwen3.6, whatever the length), the KV cache's rows of every attention layer and of the MTP layer (~11 KB a token
 * in int8), the last hidden state, the images' rotary positions, the token passed on and not computed yet.
 * append: path holds an earlier state of this same conversation (saved or loaded since the last geode_reset): only
 * what changed is written (the fixed-size states, the KV rows since then), its header last - a save cut short leaves
 * a file geode_state_load refuses, never a mixed state. A path that is not that (missing, another conversation, another
 * model) is written whole, to path.tmp then renamed over path. GEODE_E_ARG: images were added that no prompt has taken
 * yet. out_position (may be NULL): the conversation's length in tokens (geode_stats.context_used).
 *
 * geode_state_load replaces the conversation with the saved one (geode_reset, then the saved state): what the next
 * call continues from is exactly what it was when saved - the same tokens follow. The model and the KV cache's format
 * must be the ones that saved it (the CPU and the GPU path load each other's). GEODE_E_FORMAT: another model, format
 * or version, or a damaged / interrupted file; GEODE_E_ARG: longer than this model's context; GEODE_E_MEMORY: no
 * memory for its KV cache; GEODE_E_IO: unreadable. On failure the conversation is empty (as after geode_reset).
 * Images of the loaded conversation need not be added again; images for the next prompt are added after the load. */
int geode_state_save(geode_model *model, const char *path, int32_t append, int32_t *out_position, char *err,
                     size_t err_len);
int geode_state_load(geode_model *model, const char *path, int32_t *out_position, char *err, size_t err_len);

/* The model gives memory back while something runs beside it (the agent's Linux: a command, an MCP server), and
 * takes it again after, without opening the model again. Between calls only, like geode_state_save.
 *
 * bytes > 0: the expert cache gives back at least that much now, to the system for real - the CPU path releases the
 * pages of its coldest slots (madvise), the GPU path frees whole pieces of its cache's tail to the driver (the tail
 * is GEODE_RESERVE_TAIL_MB, 256 by default, in pieces of GEODE_RESERVE_PIECE_MB, 64) - and keeps it out through
 * geode_reset, geode_state_load and the context's growth; generating goes on meanwhile with fewer slots (more reads
 * from flash, the same tokens). A cache that cannot give that much (it keeps what a layer and its guesses pin) gives
 * what it can: out_bytes says how much it holds (the message in err). A larger amount later takes more.
 * bytes = 0: the reserve is over; the slots are the cache's again (empty, read on demand).
 * GEODE_E_MEMORY: the GPU's pieces could not be allocated again (the cache stays smaller; the next call tries again).
 * out_bytes (may be NULL): the bytes the reserve holds now. */
int geode_reserve_memory(geode_model *model, uint64_t bytes, uint64_t *out_bytes, char *err, size_t err_len);

/* The routed experts each token takes from now on: k of the model's top_k (Qwen3.6: 8), the router's top k of its
 * softmax over every expert, their weights renormalised over those k; the shared expert as ever; the MTP layer the
 * same. Fewer experts: fewer read from flash and computed a token (a flash-bound phone's decode is mostly those reads),
 * at some cost in quality. Between calls; the conversation goes on (its states do not depend on k). The tail skip
 * acts within the k (it leaves out uncached experts of its last quarter: k 8 -> ranks 6, 7; k 4 -> rank 3; k 1 or
 * 2: none). GEODE_E_ARG: k outside 1 .. top_k. */
int geode_set_experts(geode_model *model, int32_t k, char *err, size_t err_len);

void geode_stop(geode_model *model);
void geode_reset(geode_model *model);              /* forget the conversation */
/* Current while a generation runs too, when called from its token callback (the thread that runs it). */
void geode_get_stats(geode_model *model, geode_stats *out);
void geode_close(geode_model *model);

/* Where the time went (a line of text, for tools); the counters restart at each geode_generate. */
void geode_profile(geode_model *model, char *buf, size_t len);

/* Device benchmarks (the app's Benchmark tab), each a few seconds: what = "system", "memory", "flash", "cpu", "gpu"
 * (Vulkan), "opencl", "gpucheck" (Geode's own GPU code checked on random weights) or "npucheck" (the NPU: Geode's Hexagon module opened, checked and timed); arg is the directory "flash"
 * reads (a model's: its expert files are read; any other writable directory: a temporary 1 GiB file is written there
 * and deleted), or the file "gpucheck" / "npucheck" writes each step's name into before it runs. Writes one JSON object into
 * out. Returns GEODE_OK,
 * GEODE_E_ARG for an unknown test, GEODE_E_MEMORY when out is too small. */
int geode_bench(const char *what, const char *arg, char *out, size_t out_len);

/* The memory test (PLAN.md, P6.2): grows incompressible memory by step_bytes until max_bytes or until the process is
 * killed, appending one line to status_path after every step:
 *   held=<bytes> available=<bytes> swap_free=<bytes> vm_swap=<bytes> step_ms=<ms>
 * Meant for a process of its own: Android's low-memory killer ends it, and the app reads the lines. vm_swap is how
 * much of the probe's process is in zram (not zero at the start for a Java process), step_ms how long a step took:
 * the safe level is the last step before vm_swap grows or the steps slow down (MemoryTest.lastCleanStep). */
int geode_memory_probe(const char *status_path, uint64_t max_bytes, uint64_t step_bytes, int32_t pause_ms);

#ifdef __cplusplus
}
#endif

#endif
