package xyz.tifu.geode.engine

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import xyz.tifu.geode.engine.EngineProtocol as P

data class Sampling(
    val temperature: Float,
    val topP: Float,
    val topK: Int,
    val maxTokens: Int,
    val presencePenalty: Float = 0f,
    val seed: Long = System.nanoTime(),
)

/** How the engine runs a model (geode_config); changing any of them means opening the model again. */
data class EngineOptions(
    val budgetBytes: Long,
    val threads: Int = 0,
    val ioThreads: Int = 0,
    /** Little-core threads that join a prompt's products: 0 = none. */
    val helpers: Int = 0,
    val contextLength: Int = 0,
    val prefetch: Boolean = true,
    val gpu: Boolean = false,
    val gpuShare: Float = 0f,
    /** Routed experts a token takes: 1 .. the model's top_k (8), fewer reading less from flash; 0 = top_k. Changed
     *  on an open model by setExperts (no reopening). */
    val expertsPerToken: Int = 0,
    /** Cache-conditioned routing (geode_config.route_prior): a cached expert's logit + this for the selection while
     *  generating; 0 = off (the default: it changes answers a little), 1.5 = upstream Strata's. */
    val routePrior: Float = 0f,
    /** geode_config.tail_skip: the ranks from which a routed expert not in memory is left out (6 / 4); 0 = never,
     *  every expert the router picks (the default) - fewer experts are faster but less accurate. */
    val tailSkip: Int = 0,
    /** geode_config.npu: a prompt's dense products on the NPU where it opens and passes its check (the default);
     *  elsewhere, or off, the GPU / CPU path as without it. */
    val npu: Boolean = true,
)

sealed interface EngineEvent {
    data class Token(val id: Int) : EngineEvent
    /** The engine's figures (GeodeNative.stats) while an answer is generated. */
    data class Stats(val stats: LongArray) : EngineEvent
    data class Done(val code: Int, val error: String?, val stats: LongArray?) : EngineEvent
    /** The request's images are being encoded (true), or are done (false). */
    data class Encoding(val on: Boolean) : EngineEvent
}

/** A conversation's state not saved, loaded or read: the engine's code (GeodeNative.E_*) and its reason. */
class EngineStateException(val code: Int, message: String) : Exception(message)

/**
 * The model in memory and how the engine runs it, as it reported when the model opened: [gpu] and [npu] are the
 * engine's status lines (geode_profile: "on: ..." or "off (why)"), [threads] the threads its products run on, [spec]
 * the speculative decoding ("off", "gpu:K" - MTP drafts verified on the GPU up to K tokens at once - or "cpu:2").
 * [memoryBytes] is what the engine holds now (its last figures), [cacheBytes] its expert cache.
 */
data class EngineRun(
    val dir: String,
    val gpu: String?,
    val npu: String?,
    val threads: Int,
    val spec: String,
    val memoryBytes: Long,
    val cacheBytes: Long,
)

/** The UI side of the engine: binds EngineService (BIND_IMPORTANT), opens a model, streams tokens. One per app. */
class EngineClient(private val context: Context) : ServiceConnection {
    private var service: Messenger? = null
    private var connected = CompletableDeferred<Unit>()
    private var pendingOpen: CompletableDeferred<String?>? = null
    private var pendingBench: CompletableDeferred<String>? = null
    // a SAVE / LOAD / PREFILL's answer: arg1 (tokens, or -code), the error, the stats; one at a time (stateMutex)
    private class StateReply(val result: Int, val error: String?, val stats: LongArray?, val bytes: Long = 0,
                             val text: String? = null)
    private var pendingState: CompletableDeferred<StateReply>? = null
    private val stateMutex = Mutex()

    /** The engine's stats after its last answer, state save / load or prefill (context used: [5]). */
    @Volatile
    var lastStats: LongArray? = null
        private set
    private val incoming = Messenger(Handler(Looper.getMainLooper()) { msg -> onMessage(msg); true })

    private val _events = MutableSharedFlow<EngineEvent>(extraBufferCapacity = 4096,
        onBufferOverflow = BufferOverflow.SUSPEND)
    val events: SharedFlow<EngineEvent> = _events.asSharedFlow()

    /** The engine's stats as the model opened (GeodeNative.stats): the context's limit among them. */
    var openStats: LongArray? = null
        private set

    /** The options the open model runs with (another set means opening it again). */
    var openOptions: EngineOptions? = null
        private set

    /** The GPU path's state as the engine reported it when the model last opened ("off", what it holds, or why
     *  it is not there); null before a model opened. */
    private val _gpuStatus = MutableStateFlow<String?>(null)
    val gpuStatus: StateFlow<String?> = _gpuStatus.asStateFlow()

    /** A generation runs in the engine (from the request until its DONE). */
    private val _running = MutableStateFlow(false)
    val running: StateFlow<Boolean> = _running.asStateFlow()

    /** The model directory currently open in the engine process, or null. */
    private val _openModel = MutableStateFlow<String?>(null)
    val openModel: StateFlow<String?> = _openModel.asStateFlow()

    /** The model in memory and how it runs (null: none in memory). */
    private val _run = MutableStateFlow<EngineRun?>(null)
    val run: StateFlow<EngineRun?> = _run.asStateFlow()
    private var openedNpu: String? = null
    private var openedRun: String? = null

    /** The model let go: nothing in memory to report. */
    private fun gone() {
        _openModel.value = null
        _run.value = null
    }

    /** The engine's newest figures: what it holds now goes into [run]. */
    private fun figures(stats: LongArray) {
        lastStats = stats
        _run.value = _run.value?.copy(memoryBytes = stats.getOrElse(0) { 0L }, cacheBytes = stats.getOrElse(1) { 0L })
    }

    fun bind() {
        if (service != null) return
        context.bindService(Intent(context, EngineService::class.java), this,
            Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT)
    }

    override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
        service = Messenger(binder)
        connected.complete(Unit)
    }

    override fun onServiceDisconnected(name: ComponentName?) {
        // the :engine process died (killed by Android, or crashed): the model is gone with it
        service = null
        connected = CompletableDeferred()
        gone()
        _running.value = false
        pendingOpen?.complete("the engine process stopped")
        pendingBench?.complete("")
        pendingState?.complete(StateReply(-GeodeNative.E_IO, "the engine process stopped", null))
        _events.tryEmit(EngineEvent.Done(-1, "the engine process stopped", null))
    }

    /** Opens a model; returns null on success or the reason it failed. */
    suspend fun open(dir: String, options: EngineOptions): String? {
        // a CPU the engine cannot run on: said so, instead of the engine's process dying on its first kernel
        if (!CpuSupport.engine) return context.getString(xyz.tifu.geode.R.string.cpu_unsupported)
        bind()
        connected.await()
        val done = CompletableDeferred<String?>()
        pendingOpen = done
        send(P.OPEN, Bundle().apply {
            putString(P.K_DIR, dir)
            putLong(P.K_BUDGET, options.budgetBytes)
            putInt(P.K_THREADS, options.threads)
            putInt(P.K_IO_THREADS, options.ioThreads)
            // geode_config: 0 = the default (none), -1 = none; the setting's 0 means none
            putInt(P.K_HELPERS, if (options.helpers > 0) options.helpers else -1)
            putInt(P.K_CONTEXT, options.contextLength)
            putBoolean(P.K_PREFETCH, options.prefetch)
            putBoolean(P.K_GPU, options.gpu)
            putFloat(P.K_GPU_SHARE, options.gpuShare)
            putInt(P.K_EXPERTS, options.expertsPerToken)
            putFloat(P.K_ROUTE_PRIOR, options.routePrior)
            putInt(P.K_TAIL_SKIP, options.tailSkip)
            putBoolean(P.K_NPU, options.npu)
        })
        val error = done.await()
        _openModel.value = if (error == null) dir else null
        openOptions = if (error == null) options else null
        _run.value = if (error == null) {
            // "threads=8 spec=gpu:2" (geode_profile's run line)
            val fields = openedRun.orEmpty().split(' ').mapNotNull { f -> f.split('=', limit = 2).takeIf { it.size == 2 } }
                .associate { it[0] to it[1] }
            EngineRun(dir, _gpuStatus.value, openedNpu, fields["threads"]?.toIntOrNull() ?: 0, fields["spec"] ?: "off",
                openStats?.getOrNull(0) ?: 0L, openStats?.getOrNull(1) ?: 0L)
        } else null
        return error
    }

    /** [media]: the prompt's images, video frames and clips in order (the files the engine reads). */
    fun generate(prompt: IntArray, sampling: Sampling, stopTokens: IntArray, media: List<EngineMedia> = emptyList()) {
        if (service != null) _running.value = true
        send(P.GENERATE, Bundle().apply {
            if (media.isNotEmpty()) {
                putStringArray(P.K_IMAGES, media.map { it.path }.toTypedArray())
                putIntArray(P.K_IMAGE_W, media.map { it.w }.toIntArray())
                putIntArray(P.K_IMAGE_H, media.map { it.h }.toIntArray())
                putIntArray(P.K_MEDIA_KIND, media.map { it.kind }.toIntArray())
            }
            putIntArray(P.K_PROMPT, prompt)
            putFloat(P.K_TEMPERATURE, sampling.temperature)
            putFloat(P.K_TOP_P, sampling.topP)
            putInt(P.K_TOP_K, sampling.topK)
            putFloat(P.K_PRESENCE, sampling.presencePenalty)
            putLong(P.K_SEED, sampling.seed)
            putInt(P.K_MAX_TOKENS, sampling.maxTokens)
            putIntArray(P.K_STOPS, stopTokens)
        })
    }

    /** Runs a device benchmark in the engine process (the model is closed first); its JSON, or "" if it failed. */
    suspend fun bench(what: String, arg: String): String {
        bind()
        connected.await()
        val done = CompletableDeferred<String>()
        pendingBench = done
        gone()
        send(P.BENCH, Bundle().apply {
            putString(P.K_WHAT, what)
            putString(P.K_ARG, arg)
        })
        return done.await()
    }

    /**
     * The open model's conversation saved to [path] (geode_state_save), after the generation sent before it (the
     * engine takes its messages in order): [append] when [path] holds an earlier state of this same conversation -
     * only what grew is written (a file that is not that is written whole). The conversation's length in tokens;
     * throws EngineStateException (E_ARG: images added and not taken yet; E_IO: not written).
     */
    suspend fun saveState(path: String, append: Boolean): Int = stateCall(P.SAVE, Bundle().apply {
        putString(P.K_PATH, path)
        putBoolean(P.K_APPEND, append)
    })

    /**
     * The conversation saved at [path] replaces the open model's (geode_state_load): the next generate continues it
     * exactly. Its length in tokens; throws EngineStateException (E_FORMAT: another model, KV format or version, or
     * damaged / interrupted; E_ARG: longer than the open context; E_MEMORY; E_IO) - the conversation is then empty.
     */
    suspend fun loadState(path: String): Int = stateCall(P.LOAD, Bundle().apply { putString(P.K_PATH, path) })

    /**
     * [prompt] into the conversation with nothing generated (geode_prefill): a beginning to save and start other
     * conversations from. The conversation's length in tokens; throws EngineStateException (E_STOPPED by [stop]: the
     * conversation is then empty).
     */
    suspend fun prefill(prompt: IntArray): Int = stateCall(P.PREFILL, Bundle().apply { putIntArray(P.K_PROMPT, prompt) })

    private suspend fun stateCall(what: Int, data: Bundle): Int = stateExchange(what, data).let { r ->
        if (r.result < 0) throw EngineStateException(-r.result, r.error ?: "the engine failed")
        r.result
    }

    /**
     * The model gives [bytes] of its memory back while the agent's Linux runs beside it (a command, an MCP server)
     * - the expert cache shrinks for real: the CPU path's pages released, the GPU path's pieces freed - and keeps it
     * out through resets, loads and the context's growth; 0: the reserve is over (the cache whole again, refilled on
     * demand). After the generation sent before it. The bytes the reserve holds now (less than asked when the cache
     * is small: lastReserveNote says so); throws EngineStateException (E_MEMORY: the GPU's memory could not be taken
     * back - the cache stays smaller, the next call tries again).
     */
    suspend fun reserveMemory(bytes: Long): Long {
        val r = stateExchange(P.RESERVE, Bundle().apply { putLong(P.K_BYTES, bytes) })
        lastReserveNote = r.error
        if (r.result != 1) throw EngineStateException(GeodeNative.E_MEMORY, r.error ?: "the engine failed")
        return r.bytes
    }

    /**
     * The routed experts each token takes from now on (geode_set_experts), on the open model: 1 .. its top_k, the
     * router's top k renormalised; the conversation goes on. After the generation sent before it. openOptions follow
     * (expertsPerToken = k), so they still say the model needs no reopening. Throws EngineStateException (E_ARG: k
     * out of range, or no model).
     */
    suspend fun setExperts(k: Int) {
        val r = stateExchange(P.EXPERTS, Bundle().apply { putInt(P.K_EXPERTS, k) })
        if (r.result != 1) throw EngineStateException(GeodeNative.E_ARG, r.error ?: "the engine failed")
        openOptions = openOptions?.copy(expertsPerToken = k)
    }

    /**
     * Where the last generation's time went (geode_profile), after the generation sent before it: the prompt's line,
     * then the answer's - Gated DeltaNet (convolution, state, gated norm), attention, MoE (router, shared expert,
     * waiting for expert reads, the routed experts' arithmetic), lm_head; speculating (pairs verified, drafts that
     * stood, the MTP drafts' and the pairs' time); the experts cached / read ahead / read on demand and the MB read
     * from flash; then the GPU path's state. Counters restart at each generation. Throws EngineStateException (E_ARG:
     * no model open).
     */
    suspend fun profile(): String {
        val r = stateExchange(P.PROFILE, Bundle())
        if (r.result != 1) throw EngineStateException(GeodeNative.E_ARG, r.error ?: "no model is open")
        return r.text.orEmpty()
    }

    /** Why the last reserveMemory holds less than it was asked, or null. */
    @Volatile
    var lastReserveNote: String? = null
        private set

    private suspend fun stateExchange(what: Int, data: Bundle): StateReply = stateMutex.withLock {
        bind()
        connected.await()
        val done = CompletableDeferred<StateReply>()
        pendingState = done
        if (service == null) done.complete(StateReply(-GeodeNative.E_IO, "the engine process stopped", null))
        else send(what, data)
        // its reply always comes (a prefill is stopped to cancel it), and must be taken here: a late one would answer
        // the next call
        val r = withContext(NonCancellable) { done.await() }
        pendingState = null
        r.stats?.let(::figures)
        r
    }

    /** An attached image (its raw RGB at the model's size) encoded ahead, while the question is typed. */
    fun prepareImage(path: String, w: Int, h: Int) = send(P.PREPARE, Bundle().apply {
        putStringArray(P.K_IMAGES, arrayOf(path))
        putIntArray(P.K_IMAGE_W, intArrayOf(w))
        putIntArray(P.K_IMAGE_H, intArrayOf(h))
    })

    /** How many times the user stopped (a chat's Stop, an unload): a request that came before the last one is not
     *  answered even when it reaches the engine after it (ApiEngine looks before it generates). */
    private val userStops = java.util.concurrent.atomic.AtomicLong()
    val stops: Long get() = userStops.get()

    /** The user's stop: the answer running and every request that came before now. */
    fun stop() {
        userStops.incrementAndGet()
        send(P.STOP, Bundle())
    }

    /** The generation running now stopped (a stop sequence met, background work giving the engine up): not a
     *  request waiting for the engine. */
    fun stopRunning() = send(P.STOP, Bundle())
    fun reset() = send(P.RESET, Bundle())

    fun close() {
        send(P.CLOSE, Bundle())
        gone()
    }

    private fun send(what: Int, data: Bundle) {
        val s = service ?: return
        try {
            s.send(Message.obtain(null, what).apply {
                this.data = data
                replyTo = incoming
            })
        } catch (e: RemoteException) {
            onServiceDisconnected(null)
        }
    }

    private fun onMessage(msg: Message) {
        when (msg.what) {
            P.OPENED -> {
                openStats = if (msg.arg1 == 1) msg.data.getLongArray(P.K_STATS) else null
                if (msg.arg1 == 1) {
                    _gpuStatus.value = msg.data.getString(P.K_GPU_STATUS)
                    openedNpu = msg.data.getString(P.K_NPU_STATUS)
                    openedRun = msg.data.getString(P.K_RUN)
                }
                pendingOpen?.complete(if (msg.arg1 == 1) null else msg.data.getString(P.K_ERROR) ?: "?")
            }
            P.TOKEN -> _events.tryEmit(EngineEvent.Token(msg.arg1))
            P.STATS -> msg.data.getLongArray(P.K_STATS)?.let { _events.tryEmit(EngineEvent.Stats(it)) }
            P.ENCODING -> _events.tryEmit(EngineEvent.Encoding(msg.arg1 == 1))
            P.DONE -> {
                _running.value = false
                msg.data.getLongArray(P.K_STATS)?.let(::figures)
                _events.tryEmit(EngineEvent.Done(msg.arg1, msg.data.getString(P.K_ERROR),
                    msg.data.getLongArray(P.K_STATS)))
            }
            P.BENCH_DONE -> pendingBench?.complete(msg.data.getString(P.K_JSON).orEmpty())
            P.SAVED, P.LOADED, P.PREFILLED -> pendingState?.complete(
                StateReply(msg.arg1, msg.data.getString(P.K_ERROR), msg.data.getLongArray(P.K_STATS)))
            P.RESERVED -> pendingState?.complete(StateReply(msg.arg1, msg.data.getString(P.K_ERROR),
                msg.data.getLongArray(P.K_STATS), msg.data.getLong(P.K_BYTES)))
            P.EXPERTS_SET -> pendingState?.complete(StateReply(msg.arg1, msg.data.getString(P.K_ERROR),
                msg.data.getLongArray(P.K_STATS)))
            P.PROFILED -> pendingState?.complete(StateReply(msg.arg1, msg.data.getString(P.K_ERROR),
                msg.data.getLongArray(P.K_STATS), text = msg.data.getString(P.K_JSON)))
        }
    }
}
