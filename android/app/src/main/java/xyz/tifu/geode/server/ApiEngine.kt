package xyz.tifu.geode.server

import android.os.SystemClock
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.cache.StateCache
import xyz.tifu.geode.data.AppSettings
import xyz.tifu.geode.engine.EngineEvent
import xyz.tifu.geode.engine.GeodeNative
import xyz.tifu.geode.engine.Sampling
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.tokenizer.ChatFormat
import xyz.tifu.geode.tokenizer.ChatTemplate
import xyz.tifu.geode.tokenizer.Part
import xyz.tifu.geode.tokenizer.Prompt
import xyz.tifu.geode.tokenizer.TemplateMessage
import xyz.tifu.geode.tokenizer.Tokenizer
import xyz.tifu.geode.tokenizer.ToolCall
import xyz.tifu.geode.tokenizer.TurnImage
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.obj
import xyz.tifu.geode.util.str
import xyz.tifu.geode.vision.ImageStore
import java.security.SecureRandom

/** A request refused: the HTTP status and why (each API words the error type its own way from the status). */
class ApiError(val status: Int, message: String) : Exception(message)

/** An API request as the engine runs it: OpenAI's and Anthropic's both come to this. */
class GenRequest(
    val messages: List<TemplateMessage>,
    /** The tools as the template renders them: OpenAI's {"type": "function", "function": {...}}. */
    val tools: List<Map<String, Any?>>,
    val thinking: Boolean,
    /** Reasoning tokens before the model is told to answer; 0 = the app's setting. */
    val thinkBudget: Int = 0,
    /** The longest answer (the reasoning in it); 0 = the room the context has. */
    val maxTokens: Int = 0,
    val temperature: Float? = null,
    val topP: Float? = null,
    val topK: Int? = null,
    val presencePenalty: Float? = null,
    val seed: Long? = null,
    /** Texts that end the answer where they appear (left out of it). */
    val stop: List<String> = emptyList(),
    /** The last message is the assistant's beginning of the answer: the model goes on from it. */
    val prefill: Boolean = false,
)

enum class Finish { STOP, LENGTH, TOOL_CALLS, STOP_SEQUENCE }

class GenResult(
    val content: String,
    val reasoning: String,
    val toolCalls: List<ToolCall>,
    val finish: Finish,
    val stopSequence: String?,
    /** The prompt's tokens in all, the part of them the engine held already, the tokens generated. */
    val promptTokens: Int,
    val cachedTokens: Int,
    val outputTokens: Int,
    val reasoningTokens: Int,
    /** The generation's speed, tokens/s (the engine's figure). */
    val decodeTps: Float = 0f,
    /** The prompt's reading speed, tokens/s, and the context the conversation fills after it (the engine's). */
    val prefillTps: Float = 0f,
    val contextUsed: Int = 0,
)

sealed interface GenEvent {
    /** Nothing new for a while (waiting for the engine, or reading a long prompt): the connection is kept up. */
    data object Waiting : GenEvent
    /** The prompt is built: its tokens, and the part the engine held already. */
    data class Started(val promptTokens: Int, val cachedTokens: Int) : GenEvent
    data class Reasoning(val text: String) : GenEvent
    data class Content(val text: String) : GenEvent
}

/**
 * The local server's way to the engine: one request at a time through the app's EngineGate, with the model and the
 * settings the app's chat uses. The engine's conversation only grows, so the conversation it holds for the API is
 * remembered (its messages, the answer as generated last): a request that repeats it and adds messages - the next
 * question, a tool's output - is read from there, the rest of the prompt untouched. Any other request (another
 * conversation, a regenerated answer, the chat having had the engine) starts it again.
 */
class ApiEngine(private val app: GeodeApp) {
    private val store = ModelStore(app)
    private val images = ImageStore(app)
    private val random = SecureRandom()

    /** The conversation the engine holds for the API: the model, its tools, its messages (the last the answer as
     *  generated), its tokens; its lineage and the cache entry it is saved as (StateCache). */
    private data class Held(val model: String, val tools: String, val messages: List<TemplateMessage>, val tokens: Int,
                            val lineage: String, val key: String? = null)

    private var held: Held? = null
    /** The shared start (its cache key, its tokens) the engine holds exactly, nothing after it: a read ahead's
     *  (background) request left it there, and the next request that begins with it goes on from it. */
    private var heldStart: Pair<String, Int>? = null

    /** The last background request (a read ahead) ended with the shared start held. */
    @Volatile
    var readAheadHeld = false
        private set
    private val scope = kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.SupervisorJob() + Dispatchers.IO)
    private var saveJob: kotlinx.coroutines.Job? = null

    private val _live = MutableSharedFlow<GenEvent>(extraBufferCapacity = 1024)

    /** Every request's answer as it is written (the agent's chat shows its model's words from here, live). */
    val live: SharedFlow<GenEvent> = _live.asSharedFlow()

    private val _results = MutableSharedFlow<GenResult>(extraBufferCapacity = 64)

    /** Every request's end: its counts and the engine's speeds (the agent's chat shows them as a chat's). */
    val results: SharedFlow<GenResult> = _results.asSharedFlow()

    /** The model the app's chat has selected (the server answers with it). */
    suspend fun model(): InstalledModel {
        val s = app.settings.flow.first()
        return withContext(Dispatchers.IO) { s.selectedModel.takeIf { it.isNotEmpty() }?.let(store::find) }
            ?: throw ApiError(503, "No model is selected in Geode: open the app and choose one on the Models screen.")
    }

    suspend fun template(model: InstalledModel): ChatTemplate =
        withContext(Dispatchers.IO) { ChatTemplate(app.tokenizer(model.dir)) }

    /** An image a request sent (its bytes) as the model will read it: stored, at the size the setting allows. */
    suspend fun image(model: InstalledModel, bytes: ByteArray): Part.Image {
        if (!model.vision) throw ApiError(400, "The selected model reads no images (its pack has no vision encoder).")
        val s = app.settings.flow.first()
        return withContext(Dispatchers.IO) {
            val name = runCatching { images.importBytes(bytes) }.getOrElse { throw ApiError(400, "An image could not be read.") }
            val (w, h) = images.size(name) ?: throw ApiError(400, "An image could not be read.")
            val px = ImageStore.DETAIL_PIXELS[s.imageDetail.coerceIn(0, ImageStore.DETAIL_PIXELS.size - 1)]
            val (fw, fh) = ImageStore.fit(w, h, px)
            Part.Image(TurnImage(name, fw, fh), name)
        }
    }

    /** The server starts: images older requests sent are let go. */
    suspend fun tidy() = withContext(Dispatchers.IO) { runCatching { images.dropServerImages() } }

    /** The prompt's tokens for [req] from the start (Anthropic's count_tokens). */
    suspend fun count(req: GenRequest): Int = template(model()).render(req.messages, req.tools, req.thinking, req.prefill).size

    /** An id for a tool call: [prefix] and 24 letters and digits. */
    fun id(prefix: String): String {
        val abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        return prefix + (1..24).map { abc[random.nextInt(abc.length)] }.joinToString("")
    }

    /**
     * Runs [req]; [emit] gets the answer as it is written (it may throw: the client went away, and the generation stops).
     * While nothing is written for [KEEPALIVE_MS], [emit] gets [GenEvent.Waiting].
     */
    suspend fun run(req: GenRequest, callPrefix: String, emit: suspend (GenEvent) -> Unit): GenResult = coroutineScope {
        val lock = Mutex()
        var last = SystemClock.elapsedRealtime()
        val send: suspend (GenEvent) -> Unit = { e ->
            _live.tryEmit(e)
            lock.withLock {
                emit(e)
                last = SystemClock.elapsedRealtime()
            }
        }
        val ticker = launch {
            while (true) {
                delay(KEEPALIVE_MS / 3)
                if (SystemClock.elapsedRealtime() - last >= KEEPALIVE_MS) send(GenEvent.Waiting)
            }
        }
        try {
            generate(req, callPrefix, send).also { _results.tryEmit(it) }
        } finally {
            ticker.cancel()
        }
    }

    private suspend fun generate(req: GenRequest, callPrefix: String, send: suspend (GenEvent) -> Unit): GenResult {
        val model = model()
        val s = app.settings.flow.first()
        val tok = withContext(Dispatchers.IO) { app.tokenizer(model.dir) }
        val tpl = ChatTemplate(tok)
        val fmt = ChatFormat(tok)
        val toolsKey = Json.write(req.tools)
        return app.gate.use(this) { kept ->
            if (!kept) held = null
            // the shared start a read ahead left in the engine: this request's, if it begins with it
            val background = currentCoroutineContext()[xyz.tifu.geode.engine.EngineGate.Background] != null
            // the shared start a read ahead left in the engine: this request's, if it begins with it (and the
            // engine was neither used nor opened again since)
            var start = heldStart.takeIf { kept }
            heldStart = null
            if (background) readAheadHeld = false
            if (ensureOpen(model, s)) start = null
            val limit = app.engine.openStats?.getOrNull(6)?.toInt()?.takeIf { it > 0 } ?: s.contextLength
            // the conversation the engine holds, when this request goes on from it; else one saved on the storage
            var from = held?.takeIf { h -> h.model == model.stateId && h.tools == toolsKey && goesOn(h.messages, req) }
                ?: fromStorage(model, toolsKey, req, s)
            var prompt: Prompt? = from?.let { tpl.continuation(req.messages, it.messages.size, req.thinking) }
            // no room left after what the engine holds: the whole conversation is shorter (no earlier reasoning)
            if (from != null && from.tokens + prompt!!.size + MIN_ROOM > limit) from = null
            var used = from?.tokens ?: 0
            var lineage = from?.lineage ?: StateCache.lineage()
            val key = from?.key
            if (from == null) {
                held = null
                val full = try {
                    tpl.render(req.messages, req.tools, req.thinking, req.prefill)
                } catch (e: IllegalArgumentException) {
                    throw ApiError(400, e.message ?: "The messages do not fit the model's chat template.")
                }
                // the start every conversation of this kind shares (a system prompt and its tools), read once
                used = sharedStart(model, full, tok, fmt, s, send, start)
                // given up meanwhile (background work another use came for): not the rest
                currentCoroutineContext().ensureActive()
                if (used == 0) app.engine.reset()
                prompt = if (used == 0) full else Prompt(full.ids.copyOfRange(used, full.size), full.images)
                // a read ahead wants the shared start alone: the engine stays at its end, for the turn that comes
                if (background && used > 0) {
                    heldStart = StateCache.prefixKey(model.stateId, "", full.ids, used) to used
                    readAheadHeld = true
                }
            }
            if (background) {
                // its Started says the shared start is held (Agent.readAhead waits for it); nothing is generated
                send(GenEvent.Started(used + prompt!!.size, used))
                throw ApiError(503, "Read ahead: the shared start only.")
            }
            val p = prompt!!
            if (used + p.size + MIN_ROOM > limit)
                throw ApiError(400, "The prompt is ${used + p.size} tokens; the context holds $limit (Geode's context " +
                    "length setting, as far as the memory allows).")
            val room = limit - used - p.size
            val maxTokens = if (req.maxTokens > 0) minOf(req.maxTokens, room) else room
            val imageFiles = if (p.images.isEmpty()) emptyList() else withContext(Dispatchers.IO) {
                p.images.map { im -> Triple(images.prepare(im.name, im.width, im.height).path, im.width, im.height) }
            }
            send(GenEvent.Started(used + p.size, used))
            val out = Output(req, fmt, tok.special("<tool_call>"), tok.StreamDecoder(), send)
            val done = stream(fmt, p, imageFiles, req, s, maxTokens, out)
            out.end()
            val ok = done.code == GeodeNative.OK || (done.code == GeodeNative.E_STOPPED && out.received > 0)
            if (!ok) {
                held = null
                throw ApiError(500, done.error ?: "The engine stopped (code ${done.code}).")
            }
            val calls = out.calls(req.tools, callPrefix, this@ApiEngine::id)
            val content = out.finalContent(calls.isNotEmpty())
            // what the engine holds now, for the next request to go on from (not after a stop sequence: the engine
            // went past the text the client gets)
            held = if (out.stopHit != null) null else {
                val prefillText = if (req.prefill) req.messages.last().text.trim() else ""
                val answer = TemplateMessage("assistant", listOf(Part.Text(prefillText + content)), out.reasoning(), calls)
                Held(model.stateId, toolsKey, (if (req.prefill) req.messages.dropLast(1) else req.messages) + answer,
                    done.stats?.getOrNull(5)?.toInt() ?: (used + p.size + out.received), lineage, key)
            }
            scheduleSave()
            GenResult(
                content = content,
                reasoning = out.reasoning(),
                toolCalls = calls,
                finish = when {
                    out.stopHit != null -> Finish.STOP_SEQUENCE
                    calls.isNotEmpty() -> Finish.TOOL_CALLS
                    done.code == GeodeNative.OK && out.lastToken !in fmt.stopTokens && out.received >= maxTokens -> Finish.LENGTH
                    else -> Finish.STOP
                },
                stopSequence = out.stopHit,
                promptTokens = used + p.size,
                cachedTokens = used,
                outputTokens = out.received,
                reasoningTokens = out.reasoningTokens,
                decodeTps = (done.stats?.getOrNull(8) ?: 0L) / 1000f,
                prefillTps = (done.stats?.getOrNull(7) ?: 0L) / 1000f,
                contextUsed = done.stats?.getOrNull(5)?.toInt() ?: (used + p.size + out.received),
            )
        }
    }

    /** [req] goes on from [messages] (the conversation a state is of): they are its first ones, it has more. */
    private fun goesOn(messages: List<TemplateMessage>, req: GenRequest): Boolean =
        !req.prefill && req.messages.size > messages.size && messages.indices.all { req.messages[it].sameAs(messages[it]) }

    /**
     * A conversation saved on the storage that [req] goes on from (the longest), loaded into the engine; null: none
     * (or it would not load: it goes).
     */
    @Suppress("UNCHECKED_CAST")
    private suspend fun fromStorage(model: InstalledModel, toolsKey: String, req: GenRequest, s: AppSettings): Held? {
        if (s.stateCacheMb <= 0 || req.prefill) return null
        val tools = sha(toolsKey)
        val best = withContext(Dispatchers.IO) {
            app.stateCache.entries("api-").mapNotNull { (k, m) ->
                if (m["model"] != model.stateId || m["tools"] != tools) return@mapNotNull null
                val msgs = (m["messages"] as? List<*>)?.map { decode(it as Map<String, Any?>) } ?: return@mapNotNull null
                if (!goesOn(msgs, req)) return@mapNotNull null
                Triple(k, m, msgs)
            }.maxByOrNull { it.third.size }
        } ?: return null
        val pos = app.stateCache.load(app.engine, best.first)
        if (pos < 0) return null
        return Held(model.stateId, toolsKey, best.third, pos, best.second["lineage"] as? String ?: StateCache.lineage(),
            best.first).also { held = it }
    }

    /**
     * The start of [full] every conversation of its kind shares - its system message with the tools, when long enough
     * to matter - in the engine: loaded when it was read before, else read now (the prompt's first part alone) and
     * saved for the next ones. Its tokens; 0: none (the engine is to read [full] from the start).
     */
    private suspend fun sharedStart(model: InstalledModel, full: Prompt, tok: Tokenizer, fmt: ChatFormat, s: AppSettings,
                                    send: suspend (GenEvent) -> Unit, held: Pair<String, Int>? = null): Int {
        if (s.stateCacheMb <= 0) return 0
        val n = systemTurn(full, tok)
        if (n < PREFIX_MIN || full.size - n < 1) return 0
        val key = StateCache.prefixKey(model.stateId, "", full.ids, n)
        // in the engine already, as a read ahead left it: neither read nor loaded
        if (held == key to n) return n
        if (app.stateCache.load(app.engine, key) == n) return n
        app.engine.reset()
        send(GenEvent.Started(full.size, 0))
        val t0 = SystemClock.elapsedRealtime()
        android.util.Log.i(TAG, "reading the shared start: $n tokens")
        if (!prefill(full.ids.copyOfRange(0, n))) {
            app.engine.reset()
            return 0
        }
        android.util.Log.i(TAG, "the shared start read in ${SystemClock.elapsedRealtime() - t0} ms")
        app.stateCache.save(app.engine, key, key, mapOf("kind" to "prefix", "model" to model.stateId, "tokens" to n))
        app.stateCache.trim(s.stateCacheMb.toLong() shl 20, key)
        return n
    }

    /** The tokens of [p]'s system turn (its <|im_end|> and newline included), when it starts with one; 0: none. */
    private fun systemTurn(p: Prompt, tok: Tokenizer): Int {
        val start = runCatching { tok.special("<|im_start|>") }.getOrNull() ?: return 0
        val end = runCatching { tok.special("<|im_end|>") }.getOrNull() ?: return 0
        if (p.ids.isEmpty() || p.ids[0] != start) return 0
        val i = p.ids.indexOf(end)
        val nl = tok.encode("\n")
        if (i < 0 || nl.size != 1 || i + 1 >= p.size || p.ids[i + 1] != nl[0]) return 0
        // a system turn holds no image (an image's tokens are read with its features: not a plain prefix)
        val pad = runCatching { tok.special("<|image_pad|>") }.getOrNull()
        if (pad != null && (0..i).any { p.ids[it] == pad }) return 0
        return i + 2
    }

    /** [ids] read into the engine, nothing generated (geode_prefill); false: the engine failed. */
    private suspend fun prefill(ids: IntArray): Boolean = try {
        app.engine.prefill(ids) == ids.size
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        false
    }

    /**
     * The conversation the engine holds, to be saved: left at the gate (it runs before anyone else takes the engine),
     * and run after a quiet while (an agent's next request usually comes first: its save then covers both).
     */
    private fun scheduleSave() {
        if (held == null) return
        app.gate.deferSave(this) { saveHeld() }
        saveJob?.cancel()
        saveJob = scope.launch {
            delay(SAVE_IDLE_MS)
            app.gate.flush(this@ApiEngine)
        }
    }

    private suspend fun saveHeld() {
        val h = held ?: return
        val s = app.settings.flow.first()
        if (s.stateCacheMb <= 0) return
        val key = h.key ?: StateCache.newApiKey()
        val meta = mapOf("kind" to "api", "model" to h.model, "tools" to sha(h.tools), "tokens" to h.tokens,
            "messages" to h.messages.map(::encode))
        if (app.stateCache.save(app.engine, key, h.lineage, meta) >= 0) {
            held = h.copy(key = key)
            app.stateCache.trim(s.stateCacheMb.toLong() shl 20, key)
        }
    }

    private fun sha(s: String): String = java.security.MessageDigest.getInstance("SHA-256").digest(s.toByteArray())
        .take(12).joinToString("") { "%02x".format(it) }

    /** A message as the cache's meta keeps it (what sameAs compares, and what the template renders). */
    private fun encode(m: TemplateMessage): Map<String, Any?> = linkedMapOf(
        "role" to m.role,
        "parts" to m.parts.map { p ->
            when (p) {
                is Part.Text -> linkedMapOf("text" to p.text)
                is Part.Image -> linkedMapOf("image" to p.image.name, "w" to p.image.width, "h" to p.image.height, "key" to p.key)
            }
        },
        "reasoning" to m.reasoning,
        "calls" to m.toolCalls.map { linkedMapOf("id" to it.id, "name" to it.name, "arguments" to it.arguments) },
    )

    @Suppress("UNCHECKED_CAST")
    private fun decode(o: Map<String, Any?>): TemplateMessage = TemplateMessage(
        o.str("role").orEmpty(),
        o.list("parts").orEmpty().map { it as Map<String, Any?> }.map { p ->
            p.str("text")?.let { Part.Text(it) } ?: Part.Image(
                TurnImage(p.str("image").orEmpty(), (p["w"] as? Number)?.toInt() ?: 0, (p["h"] as? Number)?.toInt() ?: 0),
                p.str("key").orEmpty())
        },
        o.str("reasoning"),
        o.list("calls").orEmpty().map { it as Map<String, Any?> }.map { c ->
            ToolCall(c.str("id").orEmpty(), c.str("name").orEmpty(), c.obj("arguments") ?: emptyMap())
        },
    )

    /** The engine open with the selected model and the app's settings. */
    /** True: it was opened now (whatever the engine held is gone). */
    private suspend fun ensureOpen(model: InstalledModel, s: AppSettings): Boolean {
        // a use of the user's own brings an unloaded model back; background work (the shared start's read ahead)
        // never does (EngineGate.parked)
        val background = currentCoroutineContext()[xyz.tifu.geode.engine.EngineGate.Background] != null
        if (!background) app.gate.parked.value = false
        // the same folder with other weights (a pack replaced in place) is another model: opened anew
        if (app.engine.openModel.value == model.dir.path && openedId == model.stateId &&
            app.engine.openOptions == app.engineOptions(s)) return false
        if (background && app.gate.parked.value) throw ApiError(503, "The model was unloaded: the next message opens it.")
        held = null
        openedId = null
        val err = app.engine.open(model.dir.path, app.engineOptions(s))
        if (err != null) throw ApiError(503, "The model did not open: $err")
        openedId = model.stateId
        return true
    }

    /** The weights the engine has open ([InstalledModel.stateId]): set by [ensureOpen]. */
    private var openedId: String? = null

    /** The selected model into memory now (the Models screen's Load): the user's own use, so an unloaded one returns. */
    suspend fun load() {
        val model = model()
        val s = app.settings.flow.first()
        app.gate.use(this) { kept ->
            if (!kept) held = null
            ensureOpen(model, s)
        }
    }

    /**
     * The generation itself: the prompt, then - with a reasoning budget spent - Qwen's closing of the reasoning and the
     * answer. The engine's events go through a channel so that writing to a slow client never holds them up; when this
     * is cancelled (the client went away) the engine is stopped and waited for, so the next request finds it free.
     */
    private suspend fun stream(fmt: ChatFormat, prompt: Prompt, imageFiles: List<Triple<String, Int, Int>>,
                               req: GenRequest, s: AppSettings, maxTokens: Int, out: Output): EngineEvent.Done =
        coroutineScope {
            val events = Channel<EngineEvent>(Channel.UNLIMITED)
            // subscribed before the request goes out: the events have no replay
            val listener = launch(start = CoroutineStart.UNDISPATCHED) { app.engine.events.collect { events.send(it) } }
            fun sampling(n: Int) = Sampling(
                temperature = req.temperature ?: if (req.thinking) s.thinkTemperature else s.temperature,
                topP = req.topP ?: if (req.thinking) s.thinkTopP else s.topP,
                topK = req.topK ?: if (req.thinking) s.thinkTopK else s.topK,
                maxTokens = n,
                presencePenalty = req.presencePenalty ?: s.presencePenalty,
                seed = req.seed ?: System.nanoTime(),
            )
            suspend fun pump(): EngineEvent.Done {
                while (true) {
                    when (val e = events.receive()) {
                        is EngineEvent.Token -> out.token(e.id)
                        is EngineEvent.Done -> return e
                        else -> Unit
                    }
                }
            }
            var finished = false
            try {
                val thinkBudget = if (req.thinkBudget > 0) req.thinkBudget else s.thinkBudget
                val budget = if (out.reasoning && thinkBudget > 0) minOf(thinkBudget, maxTokens) else maxTokens
                app.engine.generate(prompt.ids, sampling(budget), fmt.stopTokens, imageFiles)
                var done = pump()
                if (done.code == GeodeNative.OK && out.received >= budget && budget < maxTokens &&
                    out.lastToken !in fmt.stopTokens) {
                    if (out.reasoning) {
                        val close = fmt.closeThinking()
                        val left = maxTokens - out.received - close.size
                        if (left >= 16) {
                            out.closeReasoning()
                            app.engine.generate(close, sampling(left), fmt.stopTokens)
                            done = pump()
                        }
                    } else {
                        // the reasoning ended within its budget and the answer ran into it: the rest of the answer
                        // (an empty prompt goes on from the engine's last token)
                        app.engine.generate(IntArray(0), sampling(maxTokens - out.received), fmt.stopTokens)
                        done = pump()
                    }
                }
                finished = true
                done
            } finally {
                listener.cancel()
                if (!finished) withContext(NonCancellable) {
                    held = null
                    app.engine.stop()
                    withTimeoutOrNull(STOP_WAIT_MS) { app.engine.running.first { !it } }
                }
            }
        }

    /**
     * The tokens as they come, into the reasoning, the content and the tool calls: the content goes out as it is
     * written, but for its end that may still turn out to be the beginning of a stop sequence, and for trailing
     * whitespace (a tool call may follow it).
     */
    private class Output(
        private val req: GenRequest,
        private val fmt: ChatFormat,
        private val callStart: Int,
        private val decoder: Tokenizer.StreamDecoder,
        private val send: suspend (GenEvent) -> Unit,
    ) {
        private val reasoningText = StringBuilder()
        private val contentText = StringBuilder()
        private val toolText = StringBuilder()
        /** The content's characters sent. */
        private var sent = 0
        /** The model is still reasoning (it started so: the prompt ended with <think>). */
        var reasoning = req.thinking && !req.prefill
            private set
        private var inTool = false
        var received = 0
            private set
        var reasoningTokens = 0
            private set
        var lastToken = -1
            private set
        var stopHit: String? = null
            private set
        private val stops = req.stop.filter { it.isNotEmpty() }
        private val maxStop = stops.maxOfOrNull { it.length } ?: 0

        suspend fun token(id: Int) {
            received++
            if (stopHit != null) return
            lastToken = id
            if (reasoning) reasoningTokens++
            when {
                id == fmt.thinkEnd && reasoning -> {
                    route(decoder.flush())
                    reasoning = false
                }
                id == fmt.think || id in fmt.stopTokens -> Unit
                id == callStart && !reasoning && !inTool -> {
                    route(decoder.flush())
                    inTool = true
                    toolText.append("<tool_call>")
                }
                else -> route(decoder.push(id))
            }
        }

        /** The reasoning budget is spent: what follows the app's closing is the answer. */
        fun closeReasoning() {
            reasoning = false
        }

        /** The generation ended: the decoder's last bytes. */
        suspend fun end() = route(decoder.flush())

        private suspend fun route(piece: String) {
            if (piece.isEmpty()) return
            when {
                reasoning -> {
                    reasoningText.append(piece)
                    send(GenEvent.Reasoning(piece))
                }
                inTool -> toolText.append(piece)
                else -> content(piece)
            }
        }

        private suspend fun content(piece: String) {
            // the answer starts after "</think>\n\n" (a prefill's continuation is taken as it is)
            val text = if (contentText.isEmpty() && !req.prefill) piece.trimStart('\n') else piece
            if (text.isEmpty()) return
            contentText.append(text)
            if (maxStop > 0) {
                val from = maxOf(0, sent - maxStop)
                val hit = stops.mapNotNull { st -> contentText.indexOf(st, from).takeIf { it >= 0 }?.let { it to st } }
                    .minByOrNull { it.first }
                if (hit != null) {
                    contentText.setLength(hit.first)
                    stopHit = hit.second
                }
            }
            val safe = if (stopHit != null) contentText.length else contentText.length - holdBack()
            if (safe > sent) {
                send(GenEvent.Content(contentText.substring(sent, safe)))
                sent = safe
            }
        }

        /** The end of the content not sent yet: trailing whitespace (a tool call may follow), or what may begin a stop
         *  sequence. */
        private fun holdBack(): Int {
            var n = contentText.length - contentText.trimEnd().length
            for (st in stops) for (k in minOf(st.length - 1, contentText.length) downTo 1)
                if (contentText.endsWith(st.substring(0, k))) {
                    n = maxOf(n, k)
                    break
                }
            return n
        }

        fun reasoning(): String = reasoningText.toString().trim()

        /**
         * The content as the client gets it, trimmed, its held-back end sent now. With no call parsed out of a tool
         * call's text (cut short, or malformed) that text is the content's too.
         */
        suspend fun finalContent(hasCalls: Boolean): String {
            if (!hasCalls && toolText.isNotEmpty()) contentText.append(toolText)
            val end = contentText.trimEnd().length
            if (end > sent) {
                send(GenEvent.Content(contentText.substring(sent, end)))
                sent = end
            }
            return contentText.substring(0, end).trimStart()
        }

        /**
         * The tool calls in the model's text: <tool_call><function=NAME><parameter=P>value</parameter>...</function>
         * </tool_call>, each parameter's raw text turned into the type the tool's schema gives it (the way vLLM's
         * Qwen3-Coder parser does).
         */
        fun calls(tools: List<Map<String, Any?>>, prefix: String, id: (String) -> String): List<ToolCall> {
            if (toolText.isEmpty()) return emptyList()
            val out = ArrayList<ToolCall>()
            for (m in CALL.findAll(toolText)) {
                val name = m.groupValues[1].trim()
                val schema = tools.firstOrNull { it.obj("function")?.str("name") == name }?.obj("function")
                    ?.obj("parameters")?.obj("properties")
                val args = LinkedHashMap<String, Any?>()
                for (p in PARAM.findAll(m.groupValues[2])) {
                    var v = p.groupValues[2]
                    if (v.startsWith("\n")) v = v.substring(1)
                    if (v.endsWith("\n")) v = v.substring(0, v.length - 1)
                    val key = p.groupValues[1].trim()
                    args[key] = typed(v, schema?.obj(key))
                }
                out += ToolCall(id(prefix), name, args)
            }
            return out
        }

        private fun typed(v: String, schema: Map<String, Any?>?): Any? {
            if (v.trim().equals("null", ignoreCase = true)) return null
            val type = (schema?.get("type") as? String ?: (schema?.list("type")?.firstOrNull { it != "null" } as? String))
                ?.lowercase() ?: return v
            return when (type) {
                "string", "str", "text" -> v
                "integer", "int" -> v.trim().toLongOrNull() ?: v
                "number", "float" -> v.trim().let { t -> t.toDoubleOrNull()?.let { xyz.tifu.geode.util.JsonNumber(t) } } ?: v
                "boolean", "bool" -> when (v.trim().lowercase()) {
                    "true" -> true
                    "false" -> false
                    else -> v
                }
                else -> runCatching { Json.parse(v.trim()) }.getOrElse { v }
            }
        }

        companion object {
            private val CALL = Regex("<tool_call>\\s*<function=([^>\\n]+)>(.*?)</function>\\s*</tool_call>",
                RegexOption.DOT_MATCHES_ALL)
            private val PARAM = Regex("<parameter=([^>\\n]+)>(.*?)</parameter>", RegexOption.DOT_MATCHES_ALL)
        }
    }

    companion object {
        /** The least room an answer needs. */
        private const val MIN_ROOM = 16
        /** A shared start shorter than this is read faster than its state is loaded. */
        private const val PREFIX_MIN = 512
        private const val TAG = "GeodeApi"
        /** The quiet after an answer before the conversation is saved (an agent's next request is quicker). */
        private const val SAVE_IDLE_MS = 5_000L
        const val KEEPALIVE_MS = 10_000L
        private const val STOP_WAIT_MS = 30_000L
    }
}
