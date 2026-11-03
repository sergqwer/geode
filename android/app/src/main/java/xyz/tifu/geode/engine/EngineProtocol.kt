package xyz.tifu.geode.engine

/** The Messenger protocol between the UI process (EngineClient) and the :engine process (EngineService). */
object EngineProtocol {
    // UI -> engine
    const val OPEN = 1        // data: dir, budget, threads, ioThreads, helpers, context, prefetch, gpu, gpuShare
    const val GENERATE = 2    // data: prompt, temperature, topP, topK, presence, seed, maxTokens, stops; images:
                              // the prompt's media in order, encoded first - files and their sizes, media_kind each
                              // (EngineMedia: an image or a video frame as raw RGB, a clip as 16 kHz float32)
    const val STOP = 3
    const val RESET = 4
    const val CLOSE = 5
    const val BENCH = 6       // data: what, arg (closes the model first: the benchmarks want the memory and the cores)
    const val PREPARE = 7     // data: images (one raw RGB file) and its size: encoded into its feature file now, while
                              // the question is typed (the GENERATE that follows reads it)
    const val SAVE = 8        // data: path, append: the conversation's state into a file (geode_state_save)
    const val LOAD = 9        // data: path: the conversation saved there replaces the model's (geode_state_load)
    const val PREFILL = 10    // data: prompt: into the conversation, nothing generated (geode_prefill)
    const val RESERVE = 11    // data: bytes: the expert cache gives that much memory back (0: takes it again)
    const val EXPERTS = 12    // data: experts: the routed experts a token takes from now on (geode_set_experts)
    const val PROFILE = 13    // where the last generation's time went (geode_profile)

    // engine -> UI
    const val OPENED = 101    // arg1 = 1 ok / 0 failed; data: error, stats
    const val TOKEN = 102     // arg1 = token
    const val DONE = 103      // arg1 = return code; data: error, stats
    const val BENCH_DONE = 104 // data: json
    const val STATS = 105     // data: stats (while a generation runs: after its first token, then every few)
    const val ENCODING = 106  // arg1 = 1: the request's images are being encoded, 0: done
    // SAVE / LOAD / PREFILL done: arg1 = the conversation's length in tokens, or -(a geode.h code); data: error, stats
    const val SAVED = 107
    const val LOADED = 108
    const val PREFILLED = 109
    const val RESERVED = 110  // arg1 = 1 ok / 0 failed; data: bytes (the reserve holds now), error, stats
    const val EXPERTS_SET = 111 // arg1 = 1 ok / 0 failed; data: error, stats
    const val PROFILED = 112  // arg1 = 1 ok / 0 no model; data: json (the profile's text), stats

    const val K_DIR = "dir"
    const val K_BUDGET = "budget"
    const val K_THREADS = "threads"
    const val K_IO_THREADS = "io_threads"
    const val K_CONTEXT = "context"
    const val K_HELPERS = "helpers"
    const val K_PREFETCH = "prefetch"
    const val K_GPU = "gpu"
    const val K_GPU_SHARE = "gpu_share"
    const val K_ROUTE_PRIOR = "route_prior"
    const val K_TAIL_SKIP = "tail_skip"
    const val K_NPU = "npu"
    const val K_PRESENCE = "presence"
    const val K_GPU_STATUS = "gpu_status"
    const val K_NPU_STATUS = "npu_status"
    const val K_RUN = "run"
    const val K_PROMPT = "prompt"
    const val K_TEMPERATURE = "temperature"
    const val K_TOP_P = "top_p"
    const val K_TOP_K = "top_k"
    const val K_SEED = "seed"
    const val K_MAX_TOKENS = "max_tokens"
    const val K_STOPS = "stops"
    const val K_ERROR = "error"
    const val K_STATS = "stats"
    const val K_WHAT = "what"
    const val K_ARG = "arg"
    const val K_JSON = "json"
    const val K_IMAGES = "images"
    const val K_IMAGE_W = "image_w"
    const val K_IMAGE_H = "image_h"
    const val K_MEDIA_KIND = "media_kind"
    const val K_PATH = "path"
    const val K_APPEND = "append"
    const val K_BYTES = "bytes"
    const val K_EXPERTS = "experts"
}
