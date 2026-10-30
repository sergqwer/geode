package xyz.tifu.geode.engine

/** Called by the engine with each generated token; return true to stop. */
fun interface TokenListener {
    fun onToken(token: Int): Boolean
}

/** The engine's C API (engine/include/geode/geode.h) through engine/android/jni.cpp. */
object GeodeNative {
    init {
        System.loadLibrary("geode")
    }

    // geode.h's return codes
    const val OK = 0
    const val E_ARG = 1
    const val E_IO = 2
    const val E_FORMAT = 3
    const val E_MEMORY = 4
    const val E_NOT_IMPLEMENTED = 5
    const val E_STOPPED = 6

    external fun version(): String

    /** A handle (> 0), or 0 with the reason in errOut[0]. */
    external fun open(dir: String, memoryBudget: Long, computeThreads: Int, ioThreads: Int, helperThreads: Int,
                      contextLength: Int, prefetch: Boolean, gpu: Boolean, gpuShare: Float,
                      expertsPerToken: Int, routePrior: Float, tailSkip: Int, npu: Boolean,
                      errOut: Array<String?>): Long

    external fun generate(handle: Long, prompt: IntArray, temperature: Float, topP: Float, topK: Int,
                          presencePenalty: Float, seed: Long, maxTokens: Int, stopTokens: IntArray,
                          listener: TokenListener, errOut: Array<String?>): Int

    /** geode_add_image: an image (raw RGB, w x h a size ImageStore.fit gave) for the next prompt's image pads,
     *  encoded now, or its features read from [cache] when they are there (written there after an encode). Its
     *  tokens, or -(a geode.h code) with the reason in errOut[0]. */
    external fun addImage(handle: Long, rgb: ByteArray, w: Int, h: Int, cache: String?, errOut: Array<String?>): Int

    /** geode_cache_image: the image's features into [cache] only (nothing queued for a prompt). A geode.h code. */
    external fun cacheImage(handle: Long, rgb: ByteArray, w: Int, h: Int, cache: String): Int

    /** geode_prefill: the prompt into the conversation, nothing generated (a beginning to save). A geode.h code,
     *  the reason in errOut[0]. */
    external fun prefill(handle: Long, prompt: IntArray, errOut: Array<String?>): Int

    /** geode_state_save: the conversation to [path] ([append]: only what grew since [path]'s earlier state of it).
     *  Its length in tokens, or -(a geode.h code) with the reason in errOut[0]. Between generations only. */
    external fun stateSave(handle: Long, path: String, append: Boolean, errOut: Array<String?>): Int

    /** geode_state_load: the conversation saved at [path] replaces the model's. Its length in tokens, or -(a geode.h
     *  code) with the reason in errOut[0] (the conversation is then empty). */
    external fun stateLoad(handle: Long, path: String, errOut: Array<String?>): Int

    /** geode_set_experts: the routed experts each token takes from now on, 1 .. the model's top_k (the router's top
     *  k, renormalised). A geode.h code (E_ARG: out of range), the reason in errOut[0]. Between generations only. */
    external fun setExperts(handle: Long, k: Int, errOut: Array<String?>): Int

    /** geode_reserve_memory: [bytes] > 0 - the expert cache gives back at least that much (for real: pages released,
     *  GPU pieces freed) and keeps it out; 0 - the reserve is over. The bytes it holds now, or -(a geode.h code);
     *  errOut[0] says why it holds less than asked, or why it failed. Between generations only. */
    external fun reserveMemory(handle: Long, bytes: Long, errOut: Array<String?>): Long

    /** geode_profile: where the time went; a line "GPU: " has the GPU path's state. */
    external fun profile(handle: Long): String

    external fun stop(handle: Long)
    external fun reset(handle: Long)
    external fun close(handle: Long)

    /** [memory, expert cache, hits, misses, flash bytes, context used, context length, prompt t/s x1000,
     *  decode t/s x1000, context allocated, hits a read ahead brought, reads ahead made, experts per token]; the
     *  length is the limit
     *  the budget allows, the allocated part grows 1024 tokens at a time. Hits include the read-ahead ones. */
    external fun stats(handle: Long): LongArray

    external fun memoryProbe(statusPath: String, maxBytes: Long, stepBytes: Long, pauseMs: Int): Int

    /** A device benchmark's JSON (geode_bench): what = system / memory / flash / cpu / gpu; arg = flash's directory. */
    external fun bench(what: String, arg: String): String
}
