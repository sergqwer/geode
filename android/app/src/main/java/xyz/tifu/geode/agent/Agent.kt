package xyz.tifu.geode.agent

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import android.os.SystemClock
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.engine.EngineGate
import xyz.tifu.geode.server.Api
import xyz.tifu.geode.server.GenEvent
import xyz.tifu.geode.server.HttpRequest
import xyz.tifu.geode.server.HttpResponder
import xyz.tifu.geode.server.HttpServer
import xyz.tifu.geode.search.SearchException
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.obj
import xyz.tifu.geode.util.str
import java.io.File
import java.net.InetAddress
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/** What the agent does as its session reports it, and what the model does meanwhile (Geode serves it). */
sealed interface AgentEvent {
    /** The harness's session (kept with the chat: the next message, the next start of the app, go on with it). */
    data class Session(val id: String) : AgentEvent
    /** A piece of the agent's words, or of its thought, as the model writes it. */
    data class Text(val text: String) : AgentEvent
    data class Thinking(val text: String) : AgentEvent
    /** A tool called: its id, its name, its arguments (JSON). */
    data class ToolCall(val id: String, val tool: String, val input: String) : AgentEvent
    data class ToolResult(val id: String, val ok: Boolean, val output: String) : AgentEvent
    /** The model reads this many new tokens of the prompt (the slow part, on a phone). */
    data class Reading(val tokens: Int) : AgentEvent
    /** The turn waits: another one runs (the web chat's, the phone's). */
    data object Waiting : AgentEvent
    /** The turn waits for the agent's Linux to be unpacked. */
    data object Installing : AgentEvent
}

class AgentException(message: String) : Exception(message)

/** A command the agent takes in the chat ("/name args"), as it lists them (ACP's available commands). */
data class AgentCommand(val name: String, val description: String, val hint: String? = null)

/**
 * The agent: DeepSeek Harness (MIT) in the app's Linux, driven over the Agent Client Protocol (dsh --profile acp: one
 * JSON-RPC message a line on its stdin and stdout), its model Geode's own through an API on the phone's loopback only,
 * with a key made for the process. The harness runs while the open chat has the agent on: started ahead (its start is
 * the slow part under proot), kept between messages, stopped when the agent goes off, another chat opens, the app
 * goes to the background, or it idles - its RAM goes back to the model. A chat's session is resumed in a new process
 * (session/resume), so a restart of the app goes on with the same conversation.
 *
 * Each chat has its own Linux: the image is shared and stays as shipped, the chat's own folders are mounted over it
 * (files/agents/<chat>/): workspace as /workspace (the files the agent makes, the user's own put in), home as /root
 * (the harness's sessions, caches), local as /usr/local (what the agent installs with pip or npm), tmp as /tmp. A
 * chat's fork copies them; deleting the chat deletes them.
 */
class Agent(private val app: GeodeApp) {
    private val linux = Linux(app)
    private val agents = File(app.filesDir, "agents")
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val lock = Mutex()
    /** One turn at a time, whoever asks (the phone's chat, the web chat): the engine answers one request at a time. */
    private val turnLock = Mutex()
    @Volatile
    private var live: Harness? = null
    private var idle: Job? = null

    init {
        // an update the app died in finished or undone, before a harness runs in it
        linux.settle()
        scope.launch { runCatching { dropEmptyChats() } }
    }

    /**
     * The folders of chats that never had a message (each start of the app gave its new chat one, until a chat got
     * its folder with its first message): no chat of that id is saved, and nothing is in it but what a harness wrote
     * as it started - its image marker, sessions with no message. Anything else (a file, a link, a folder it cannot
     * read) keeps it; a folder changed in the last minutes too.
     */
    private fun dropEmptyChats() {
        val chats = File(app.filesDir, "chats")
        val before = System.currentTimeMillis() - 10 * 60_000L
        for (d in agents.listFiles().orEmpty()) {
            if (d.name.startsWith(".") || !d.isDirectory || File(chats, "${d.name}.json").exists()) continue
            if (d.lastModified() > before || !onlyStartFiles(d, d)) continue
            linux.wipe(d)
            android.util.Log.i(TAG, "an empty chat folder removed: ${d.name}")
        }
    }

    private fun onlyStartFiles(dir: File, root: File): Boolean {
        val kids = dir.listFiles() ?: return false
        for (f in kids) {
            val mode = runCatching { android.system.Os.lstat(f.path).st_mode }.getOrNull() ?: return false
            when {
                android.system.OsConstants.S_ISDIR(mode) -> if (!onlyStartFiles(f, root)) return false
                android.system.OsConstants.S_ISREG(mode) -> {
                    val rel = f.relativeTo(root).path.replace('\\', '/')
                    val start = rel == "home/.geode-image" ||
                        rel.startsWith("home/.geode-agent/sessions/") && rel.endsWith(".jsonl") && f.length() < 64 * 1024 &&
                        f.readLines().all { it.isBlank() || START_LINE.containsMatchIn(it) }
                    if (!start) return false
                }
                else -> return false
            }
        }
        return true
    }

    val ready: Boolean get() = linux.installed

    private val _commands = MutableStateFlow<List<AgentCommand>>(emptyList())
    /** The commands the running agent takes (none until it lists them; a harness that lists none: none). */
    val commands: StateFlow<List<AgentCommand>> = _commands.asStateFlow()

    private val _turn = MutableStateFlow<String?>(null)
    /** The chat whose turn runs now (asked from the phone or from the browser); null: none. */
    val turn: StateFlow<String?> = _turn.asStateFlow()

    /**
     * The Linux on the phone is not the image this build ships (an update of the app brought a new one): it is
     * unpacked again before the agent runs. The chats' own folders stay as they are.
     */
    val outdated: Boolean get() = ready && bundled()?.let { it != linux.version } == true

    /** The version of the image this build ships (tools/linux/publish.py), null: none in it. */
    private fun bundled(): String? =
        runCatching { app.assets.open(VERSION).bufferedReader().use { it.readText().trim() } }.getOrNull()
            ?: File(app.filesDir, "linux/$IMAGE").takeIf { it.isFile }?.let { "dev-${it.length()}" }

    private val _installing = MutableStateFlow<Float?>(null)
    /** The Linux being unpacked: the share done (null: not now). */
    val installing: StateFlow<Float?> = _installing.asStateFlow()

    private val _installError = MutableStateFlow<String?>(null)
    /** Why the last unpacking failed (null: it did not). */
    val installError: StateFlow<String?> = _installError.asStateFlow()

    private var installJob: Job? = null

    /**
     * The Linux unpacked in the background when it is not on the phone or an update of the app brought a new image
     * (the chats' folders stay); a turn waits for it.
     */
    @Synchronized
    fun installIfNeeded() {
        if (installJob?.isActive == true || (ready && !outdated)) return
        _installing.value = 0f
        _installError.value = null
        installJob = scope.launch {
            try {
                install { p -> _installing.value = p }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _installError.value = e.message ?: e.javaClass.simpleName
            } finally {
                _installing.value = null
            }
        }
    }

    /**
     * The agent's Linux onto the phone, or updated, from the image Geode ships (an install-time asset pack from
     * Google Play; a developer's adb-pushed files/linux/geode-linux.tar.xz otherwise). [progress]: the share
     * unpacked, 0..1.
     */
    suspend fun install(progress: (Float) -> Unit) = withContext(Dispatchers.IO) {
        stop()
        val version = bundled() ?: throw AgentException("the agent's Linux image is not in this build")
        val dev = File(app.filesDir, "linux/$IMAGE")
        val (input, size) = runCatching { app.assets.openFd(IMAGE).let { it.createInputStream() to it.length } }
            .getOrNull() ?: runCatching { app.assets.open(IMAGE) to -1L }.getOrNull()
            ?: (dev.takeIf { it.isFile }?.let { it.inputStream() to it.length() }
                ?: throw AgentException("the agent's Linux image is not in this build"))
        var read = 0L
        val counted = object : java.io.FilterInputStream(input) {
            override fun read(): Int = super.read().also { if (it >= 0) count(1) }
            override fun read(b: ByteArray, off: Int, len: Int): Int = super.read(b, off, len).also { if (it > 0) count(it.toLong()) }
            private fun count(n: Long) {
                read += n
                if (size > 0) progress((read.toFloat() / size).coerceIn(0f, 1f))
            }
        }
        // unpacked, the image takes about six times the archive
        counted.use { linux.install(it, version, if (size > 0) size * 6 + (64L shl 20) else 0) }
        progress(1f)
    }

    /** All of a chat's own Linux. */
    fun chatDir(chatId: String) = File(agents, chatId)

    /** The chat's folder (its /workspace). */
    fun workspace(chatId: String) = File(chatDir(chatId), "workspace")

    /**
     * The harness up for [chatId] ahead of its first message (another chat's goes); its session id. Null while a turn
     * runs or the shared start is read ahead (the harness they use stays): the first message starts it then.
     */
    suspend fun warm(chatId: String, sessionId: String?): String? {
        if (turnLock.isLocked || installJob?.isActive == true) return null
        return lock.withLock { harness(chatId, sessionId).sessionId }
    }

    /**
     * [task] to the agent of [chatId], with [images] (the chat's stored JPEGs) and [audio] (its clips) before it;
     * [onEvent] gets what it does
     * as it does it. Returns how the turn ended (end_turn, max_tokens, cancelled...). Cancelled, the turn is cancelled
     * in the harness (and the harness stopped when it does not end the turn soon). A read ahead of the shared start
     * that runs is gone on with, not thrown away: the turn begins with the same start, so it waits for the rest of it
     * (its reading shown) and goes on from where it ends; another turn (the web chat's, the phone's) is waited for
     * ([AgentEvent.Waiting]).
     */
    suspend fun prompt(chatId: String, sessionId: String?, task: String, images: List<File> = emptyList(),
                       audio: List<File> = emptyList(), videos: List<VideoInput> = emptyList(),
                       onEvent: suspend (AgentEvent) -> Unit): String {
        awaitWarm(onEvent)
        installJob?.let { if (it.isActive) onEvent(AgentEvent.Installing) }
        installJob?.join()
        if (turnLock.isLocked) onEvent(AgentEvent.Waiting)
        return turnLock.withLock {
            _turn.value = chatId
            try {
                turn(chatId, sessionId, task, images, audio, videos, onEvent)
            } finally {
                _turn.value = null
                lastTurn = SystemClock.elapsedRealtime()
                // the shared start, when it is still to be read (the user's servers or skills changed)
                warmAhead()
            }
        }
    }

    private suspend fun turn(chatId: String, sessionId: String?, task: String, images: List<File>, audio: List<File>,
                             videos: List<VideoInput>, onEvent: suspend (AgentEvent) -> Unit): String {
        idle?.cancel()
        busy = true
        val h = try {
            lock.withLock { harness(chatId, sessionId) }
        } catch (e: Exception) {
            busy = false
            throw e
        }
        onEvent(AgentEvent.Session(h.sessionId))
        try {
            return coroutineScope {
                // the prompt's reading, from the engine: the harness says nothing until the first word
                val reading = launch(start = CoroutineStart.UNDISPATCHED) {
                    app.apiEngine.live.collect { e ->
                        if (e is GenEvent.Started && e.promptTokens > e.cachedTokens)
                            onEvent(AgentEvent.Reading(e.promptTokens - e.cachedTokens))
                    }
                }
                try {
                    h.prompt(task, images, audio, videos, onEvent)
                } finally {
                    reading.cancel()
                }
            }
        } finally {
            busy = false
            // stopped after a while of nothing: its RAM is the model's again
            idle = scope.launch {
                delay(IDLE_MS)
                stop()
            }
        }
    }

    // ---- the shared start, read ahead ----

    private var warmer: Job? = null
    @Volatile
    private var reading: Job? = null
    @Volatile
    private var lastTurn = 0L
    /** The read ahead running now was given up to another use (not a failure: it is tried again). */
    @Volatile
    private var yielded = false
    private var failures = 0

    private val _warming = MutableStateFlow(false)
    /** The shared start is being read ahead (its harness is the agent's now). */
    val warming: StateFlow<Boolean> = _warming.asStateFlow()

    /**
     * The agent's shared start - its system prompt and tools, ~7k tokens - read into the state cache ahead (ApiEngine
     * keeps the system turn of a request it read: StateCache's prefix entries), so that a new chat's first answer
     * finds it instead of reading it first (minutes on a slow phone). Once per model, the user's servers and skills
     * and the app's build (the start's marker file); when the engine has been idle a while; as background work at
     * the engine's gate: anyone else's use of the engine (a chat's turn, the local server, the web chat, a benchmark,
     * a setting) stops it at once, and it starts again once the engine is idle again.
     */
    @Synchronized
    fun warmAhead() {
        if (warmer?.isActive == true) return
        warmer = scope.launch {
            while (true) {
                val path = selectedModel() ?: return@launch
                if (app.settings.flow.first().stateCacheMb <= 0 || warmed(path) || failures >= MAX_WARM_FAILURES) return@launch
                awaitIdle()
                // the model the engine is to read it with now (perhaps another was selected meanwhile)
                val now = selectedModel() ?: return@launch
                if (warmed(now)) return@launch
                yielded = false
                val job = scope.async(start = CoroutineStart.LAZY) { readAhead(now) }
                reading = job
                app.gate.yieldBackground = ::yieldNow
                _warming.value = true
                android.util.Log.i(TAG, "the shared start read ahead: starting")
                val ok = try {
                    job.await()
                } catch (e: CancellationException) {
                    ensureActive()
                    false
                } catch (e: Exception) {
                    false
                } finally {
                    app.gate.yieldBackground = null
                    reading = null
                    _warming.value = false
                }
                android.util.Log.i(TAG, "the shared start read ahead: ${if (ok) "done" else if (yielded) "given up to another use" else "failed"}")
                if (ok) return@launch
                if (!yielded) failures++
            }
        }
    }

    /** The read ahead stopped at once: cancelled, the engine stopped when it is reading for it (from any thread). */
    private fun yieldNow() {
        val r = reading ?: return
        android.util.Log.i(TAG, "the shared start read ahead: giving the engine up (it was reading: ${app.gate.backgroundIn})")
        yielded = true
        r.cancel()
        if (app.gate.backgroundIn) app.engine.stopRunning()
    }

    /**
     * A running read ahead finished for the turn that comes (its start is the turn's: what it read is not read
     * again), its reading shown to the turn; given up only when it runs far too long.
     */
    private suspend fun awaitWarm(onEvent: suspend (AgentEvent) -> Unit) {
        val r = reading ?: return
        coroutineScope {
            val shown = launch(start = CoroutineStart.UNDISPATCHED) {
                app.apiEngine.live.collect { e ->
                    if (e is GenEvent.Started && e.promptTokens > e.cachedTokens) onEvent(AgentEvent.Reading(e.promptTokens - e.cachedTokens))
                }
            }
            try {
                if (withTimeoutOrNull(HANDOVER_MS) { r.join() } == null) {
                    yieldNow()
                    withTimeoutOrNull(YIELD_MS) { r.join() }
                }
            } finally {
                shown.cancel()
            }
        }
    }

    /** Until the engine has been idle a while: nobody at its gate, no turn, the Linux in place. */
    private suspend fun awaitIdle() {
        val since = SystemClock.elapsedRealtime()
        while (true) {
            val now = SystemClock.elapsedRealtime()
            // (not while the user keeps the model unloaded: reading ahead would bring it back)
            val quiet = app.gate.holder.value == null && !turnLock.isLocked && !busy && !app.gate.parked.value &&
                now - maxOf(since, app.gate.lastFree, lastTurn) >= WARM_IDLE_MS
            if (quiet && ready && !outdated && installJob?.isActive != true) return
            delay(2000)
        }
    }

    private suspend fun selectedModel(): String? {
        val sel = app.settings.flow.first().selectedModel.takeIf { it.isNotEmpty() } ?: return null
        return withContext(Dispatchers.IO) { xyz.tifu.geode.models.ModelStore(app).find(sel)?.dir?.path }
    }

    private suspend fun warmed(modelPath: String): Boolean =
        runCatching { File(agents, WARM_MARK).readText() }.getOrNull() == warmMark(modelPath)

    /**
     * The read itself: a harness for a scratch chat asks the model once, and the request is dropped as soon as the
     * engine holds the shared start (the request's Started says it was found, not read). Holds the turn's lock: a
     * turn that comes gives it up first ([yieldWarm]). True: done.
     */
    private suspend fun readAhead(modelPath: String): Boolean = turnLock.withLock {
        val me = currentCoroutineContext()[Job]
        val h = lock.withLock { harness(WARM_CHAT, null, background = me) }
        try {
            val held = CompletableDeferred<Boolean>()
            val ok = coroutineScope {
                val watch = launch(start = CoroutineStart.UNDISPATCHED) {
                    app.apiEngine.live.collect { e ->
                        if (e is GenEvent.Started && e.cachedTokens > 0) held.complete(true)
                    }
                }
                val turn = launch {
                    runCatching { h.prompt("Reply with one word: ready.", emptyList(), emptyList(), emptyList()) {} }
                    // its request ended (at the shared start, or failed before it)
                    held.complete(app.apiEngine.readAheadHeld)
                }
                try {
                    held.await()
                } finally {
                    // its request ended at the shared start by itself (ApiEngine: background work reads no more)
                    watch.cancel()
                    if (turn.isActive) turn.cancel()
                }
            }
            if (ok) {
                agents.mkdirs()
                File(agents, WARM_MARK).writeText(warmMark(modelPath))
            }
            ok
        } finally {
            withContext(NonCancellable) {
                if (live === h) stop() else h.close()
                linux.wipe(File(agents, WARM_CHAT))
            }
        }
    }

    /**
     * What the shared start depends on: the model and whether it reads images (the agent's input), the context
     * (the agent is told it), the user's MCP servers and skills, the app's build.
     */
    private suspend fun warmMark(modelPath: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(modelPath.toByteArray())
        md.update("vision=${File(modelPath, "vision.bin").isFile};context=${app.settings.flow.first().contextLength}".toByteArray())
        runCatching { md.update(File(app.filesDir, "agent/mcp.json").readBytes()) }
        app.agentConfig.skillsDir.walkTopDown().filter { it.isFile }.sortedBy { it.path }.forEach {
            md.update("${it.path}:${it.length()}:${it.lastModified()}".toByteArray())
        }
        val build = runCatching { app.packageManager.getPackageInfo(app.packageName, 0).longVersionCode }.getOrDefault(0L)
        md.update(build.toString().toByteArray())
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /** A turn runs (the app in the background keeps the harness for it). */
    @Volatile
    var busy = false
        private set

    /** The app went to the background: an idle harness goes (a frozen one would keep its RAM). */
    fun stopIfIdle() {
        if (!busy && reading == null) stop()
    }

    /** The turn running cancelled in the harness (whoever's it is: the phone's chat, the web chat's) - an unload does
     *  not wait for it. The turn ends as "cancelled" (its prompt returns). */
    fun cancelTurn() {
        if (_turn.value != null) live?.cancelTurn()
    }

    /** The harness gone (the agent off, another chat, the app in the background): RAM back to the model. */
    fun stop() {
        val h = live ?: return
        live = null
        _commands.value = emptyList()
        h.close()
    }

    private suspend fun harness(chatId: String, sessionId: String?, background: Job? = null): Harness {
        // the same chat's, unless the user's servers or skills changed since it started, or the model (another one
        // selected: the harness is told the model and what it takes at its start)
        val model = runCatching { modelKey(app.apiEngine.model()) }.getOrNull()
        live?.let {
            if (it.chatId == chatId && it.alive && it.config == app.agentConfig.version && it.model == model &&
                background == null) return it
        }
        stop()
        _commands.value = emptyList()
        val h = Harness(app, linux, chatId, chatDir(chatId), workspace(chatId), background) { _commands.value = it }
        try {
            h.start(sessionId)
        } catch (e: Exception) {
            h.close()
            throw e
        }
        live = h
        return h
    }

    companion object {
        private const val TAG = "GeodeAgent"
        /** A session's line written at its start (its meta, its state), not a message. */
        private val START_LINE = Regex("^\\{\"t\":\"(meta|state)\"")
        private const val IDLE_MS = 5 * 60_000L
        /** The engine idle this long before the shared start is read ahead; a read ahead given up is waited for so long. */
        private const val WARM_IDLE_MS = 30_000L
        private const val YIELD_MS = 15_000L
        /** A turn waits for a running read ahead at most this long (the slowest phone reads it in ~5 min). */
        private const val HANDOVER_MS = 15 * 60_000L
        /** Read aheads that failed (not given up to another use) before none is tried until the next start. */
        private const val MAX_WARM_FAILURES = 3
        /** The scratch chat the shared start is read with (readAhead), and the file that says it was. */
        private const val WARM_CHAT = ".warm"
        private const val WARM_MARK = ".warm-start"
        /** The image's name in the asset pack (tools/linux/pack.sh), and its version's (tools/linux/publish.py). */
        const val IMAGE = "geode-linux.tar.xz"
        private const val VERSION = "geode-linux.version"
    }
}

/** One harness process, its loopback API server and its JSON-RPC exchange. */
/** A video for the agent: its frames (the chat's JPEGs) with their seconds, its sound (a clip's file), its length. */
data class VideoInput(val frames: List<Pair<File, Int>>, val audio: File?, val ms: Long)

/** What a model takes besides text, as the harness is told it (GEODE_INPUT). */
private fun input(model: xyz.tifu.geode.models.InstalledModel): String =
    (listOf("text") + listOfNotNull("image".takeIf { model.vision }, "audio".takeIf { model.audio },
        "video".takeIf { model.video })).joinToString(",")

/** A model as a harness started for it knows it: its pack and what it takes. */
private fun modelKey(model: xyz.tifu.geode.models.InstalledModel): String = model.dir.path + "|" + input(model)

private class Harness(
    private val app: GeodeApp,
    private val linux: Linux,
    val chatId: String,
    private val own: File,
    private val workspace: File,
    /** The read ahead it serves (Agent.readAhead): its requests are background work at the engine's gate, and end
     *  with it; null: a chat's. */
    private val background: Job?,
    /** The commands it lists, whenever it does. */
    private val onCommands: (List<AgentCommand>) -> Unit,
) {
    private val random = SecureRandom()
    private val key = "agent-" + ByteArray(24).also { random.nextBytes(it) }.joinToString("") { "%02x".format(it) }
    // 127.0.0.1 itself: Android's getLoopbackAddress() is ::1, where the harness does not knock
    private val api = Api(app, app.apiEngine) { key }
    private val server = HttpServer(0, ::route, InetAddress.getByAddress(byteArrayOf(127, 0, 0, 1)))
    private lateinit var process: Process
    private val ids = AtomicInteger()
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<Map<String, Any?>>>()
    @Volatile
    private var onUpdate: (suspend (Map<String, Any?>) -> Unit)? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val errors = StringBuilder()
    lateinit var sessionId: String
        private set
    val alive: Boolean get() = ::process.isInitialized && process.isAlive
    /** The user's MCP servers and skills as they were at its start (AgentConfig.version). */
    val config = app.agentConfig.version
    /** The model it was told at its start (modelKey). */
    @Volatile
    var model: String? = null
        private set

    /**
     * A chat whose Linux last ran another image (the app updated it) runs the new image's /etc/geode/migrate with the
     * old version first, if it has one - the place an image moves what a chat keeps (its /root, /usr/local,
     * /workspace) to what it needs. Told by two small files, without starting the Linux when nothing changed.
     */
    private fun migrate() {
        val version = runCatching { File(linux.root, "etc/geode/version").readText().trim() }.getOrNull() ?: return
        val marker = File(own, "home/.geode-image")
        val was = runCatching { marker.readText().trim() }.getOrDefault("")
        if (was == version) return
        if (File(linux.root, "etc/geode/migrate").isFile) runCatching {
            val p = linux.start(workspace, "/etc/geode/migrate '${was.replace("'", "")}' >/root/.geode-migrate.log 2>&1 </dev/null",
                binds = listOf(File(own, "home") to "/root", File(own, "local") to "/usr/local", File(own, "tmp") to "/tmp"))
            if (!p.waitFor(MIGRATE_S, java.util.concurrent.TimeUnit.SECONDS)) p.destroyForcibly()
        }
        marker.parentFile?.mkdirs()
        marker.writeText(version + "\n")
    }

    /** A request, for the read ahead as background work that ends with it (EngineGate.Background). */
    private suspend fun route(req: HttpRequest, r: HttpResponder) {
        val bg = background ?: return serve(req, r)
        withContext(EngineGate.Background) {
            val me = currentCoroutineContext()[Job]
            val link = bg.invokeOnCompletion { me?.cancel() }
            try {
                serve(req, r)
            } finally {
                link.dispose()
            }
        }
    }

    /** The model's API, and Geode's web search for the harness's web_search (web-search-geode.mjs), same key. */
    private suspend fun serve(req: HttpRequest, r: HttpResponder) {
        if (req.path != "/geode/search") return api.handle(req, r)
        val given = req.header("authorization")?.trim()?.removePrefix("Bearer")?.trim().orEmpty()
        if (!MessageDigest.isEqual(given.toByteArray(), key.toByteArray())) return r.json(401, mapOf("error" to "unauthorized"))
        val query = req.query["q"].orEmpty().trim()
        if (query.isEmpty()) return r.json(400, mapOf("error" to "no query"))
        try {
            val found = app.webSearch.search(query, req.query["max"]?.toIntOrNull() ?: 8)
            r.send(200, "application/json", found.toByteArray())
        } catch (e: SearchException) {
            r.json(502, mapOf("error" to e.message))
        }
    }

    suspend fun start(resume: String?) = withContext(Dispatchers.IO) {
        server.start()
        val model = app.apiEngine.model()
        val s = app.settings.flow.first()
        val env = mapOf(
            "GEODE_BASE_URL" to "http://127.0.0.1:${server.boundPort}/v1",
            "GEODE_KEY" to key,
            "GEODE_MODEL" to model.id,
            // the model as it is: its context (the harness compacts before it is full), images if its pack reads them,
            // sound if it hears
            "GEODE_CONTEXT" to s.contextLength.toString(),
            "GEODE_INPUT" to input(model),
        )
        this@Harness.model = modelKey(model)
        val native = File(app.applicationInfo.nativeLibraryDir, NATIVE)
        process = if (native.isFile) {
            // geode-agent (geode/agent: the harness in Rust) runs here, outside the Linux, which it starts only for
            // what runs in it (bash, an MCP server's command)
            migrate()
            val paths = mapOf(
                "GEODE_LIBDIR" to app.applicationInfo.nativeLibraryDir,
                "GEODE_ROOTFS" to linux.root.path,
                "GEODE_CHAT" to own.path,
                "GEODE_PROOT_TMP" to linux.tmp.apply { mkdirs() }.path,
                "GEODE_SEARCH_CACHE" to File(app.filesDir, "search").apply { mkdirs() }.path,
                "GEODE_SEARCH_LANG" to java.util.Locale.getDefault().language,
                "GEODE_SKILLS" to app.agentConfig.skillsDir.apply { mkdirs() }.path,
                "TMPDIR" to File(own, "tmp").apply { mkdirs() }.path,
                "PATH" to "/system/bin",
            )
            for (d in listOf("workspace", "home", "local", "tmp")) File(own, d).mkdirs()
            ProcessBuilder(native.path).directory(workspace.apply { mkdirs() }).redirectErrorStream(false).apply {
                environment().clear()
                environment().putAll(env + paths)
            }.start()
        } else {
            // DeepSeek Harness itself, in the Linux (Node under proot): a build without geode-agent
            linux.start(workspace, START, env + mapOf(
                "DSH_HOME" to "/root/.dsh",
                "DSH_PERMISSION_MODE" to "danger-full-access",
                // the heap bounded (the phone's RAM is mostly the model's), the young generation small (~60 MB less
                // at the same speed), the code compiled once, in the image
                "NODE_OPTIONS" to "--max-old-space-size=256 --max-semi-space-size=1",
                "NODE_COMPILE_CACHE" to "/var/cache/node-compile",
            ), listOf(File(own, "home") to "/root", File(own, "local") to "/usr/local", File(own, "tmp") to "/tmp",
                // the user's own skills, every chat's (the harness's user skills: ~/.agents/skills)
                app.agentConfig.skillsDir to "/root/.agents/skills"),
                mergeErrors = false)
        }
        scope.launch { read() }
        scope.launch {
            runCatching { process.errorStream.bufferedReader().forEachLine { synchronized(errors) { errors.appendLine(it) } } }
        }
        request("initialize", mapOf("protocolVersion" to 1, "clientCapabilities" to mapOf(
            "fs" to mapOf("readTextFile" to false, "writeTextFile" to false), "terminal" to false)))
        val args = mapOf("cwd" to "/workspace", "mcpServers" to app.agentConfig.enabledServers().map { it.acp() })
        sessionId = resume?.let { id ->
            runCatching { request("session/resume", args + ("sessionId" to id)); id }.getOrNull()
        } ?: (request("session/new", args)["sessionId"] as? String ?: throw AgentException("no session"))
    }

    suspend fun prompt(task: String, images: List<File>, audio: List<File>, videos: List<VideoInput>,
                       onEvent: suspend (AgentEvent) -> Unit): String {
        onUpdate = { u -> event(u)?.let { onEvent(it) } }
        try {
            // the images first (ACP image blocks: the chat's stored JPEGs), each with a copy in the chat's folder that the
            // model is told the path of (its tools read it; without one it guessed a path), then the clips (ACP audio
            // blocks: WAV, a copy in the folder too), then the words
            val pictures = images.mapNotNull { f ->
                runCatching {
                    val bytes = f.readBytes()
                    val copy = File(workspace, "$IMAGES/${f.name}")
                    copy.parentFile?.mkdirs()
                    if (!copy.isFile) copy.writeBytes(bytes)
                    mapOf("type" to "image", "mimeType" to "image/jpeg", "uri" to "file:///workspace/$IMAGES/${f.name}",
                        "data" to android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP))
                }.getOrNull()
            }
            val sounds = audio.mapNotNull { f ->
                runCatching {
                    val wav = xyz.tifu.geode.media.Wav.encode(xyz.tifu.geode.media.AudioStore.read(f))
                    val name = f.nameWithoutExtension + ".wav"
                    val copy = File(workspace, "$AUDIO/$name")
                    copy.parentFile?.mkdirs()
                    if (!copy.isFile) copy.writeBytes(wav)
                    mapOf("type" to "audio", "mimeType" to "audio/wav", "uri" to "file:///workspace/$AUDIO/$name",
                        "data" to android.util.Base64.encodeToString(wav, android.util.Base64.NO_WRAP))
                }.getOrNull()
            }
            // the videos (Geode's video block: their frames, each at its second, and their sound as WAV)
            val films = videos.mapNotNull { v ->
                runCatching {
                    val frames = v.frames.map { (file, second) ->
                        mapOf("data" to android.util.Base64.encodeToString(file.readBytes(), android.util.Base64.NO_WRAP),
                            "mimeType" to "image/jpeg", "second" to second)
                    }
                    val block = linkedMapOf<String, Any?>("type" to "video", "frames" to frames, "durationMs" to v.ms)
                    v.audio?.let { a ->
                        val wav = xyz.tifu.geode.media.Wav.encode(xyz.tifu.geode.media.AudioStore.read(a))
                        block["audio"] = mapOf("data" to android.util.Base64.encodeToString(wav, android.util.Base64.NO_WRAP),
                            "mimeType" to "audio/wav")
                    }
                    block
                }.getOrNull()
            }
            val blocks = pictures + films + sounds + mapOf("type" to "text", "text" to task)
            val r = request("session/prompt", mapOf("sessionId" to sessionId, "prompt" to blocks))
            return r["stopReason"] as? String ?: "end_turn"
        } catch (e: CancellationException) {
            // the turn cancelled in the harness; a harness that does not end it soon is stopped
            withContext(NonCancellable) {
                notify("session/cancel", mapOf("sessionId" to sessionId))
                if (withTimeoutOrNull(CANCEL_MS) { while (pending.isNotEmpty()) delay(100) } == null) close()
            }
            throw e
        } finally {
            onUpdate = null
        }
    }

    /** Its turn running cancelled (session/cancel): its prompt returns "cancelled". */
    fun cancelTurn() {
        if (pending.isNotEmpty()) notify("session/cancel", mapOf("sessionId" to sessionId))
    }

    private fun commands(u: Map<String, Any?>): List<AgentCommand> =
        u.list("availableCommands").orEmpty().mapNotNull { it as? Map<String, Any?> }.mapNotNull { c ->
            val name = c.str("name")?.removePrefix("/")?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
            AgentCommand(name, c.str("description").orEmpty(), c.obj("input")?.str("hint"))
        }

    /** An update as the chat shows it (null: one it does not). */
    private fun event(u: Map<String, Any?>): AgentEvent? {
        val text = u.obj("content")?.str("text")
        return when (u.str("sessionUpdate")) {
            "agent_message_chunk" -> text?.let { AgentEvent.Text(it) }
            "agent_thought_chunk" -> text?.let { AgentEvent.Thinking(it) }
            "tool_call" -> AgentEvent.ToolCall(u.str("toolCallId").orEmpty(), u.str("title").orEmpty(),
                Json.write(u["rawInput"] ?: emptyMap<String, Any?>()))
            "tool_call_update" -> {
                val status = u.str("status")
                if (status != "completed" && status != "failed") null
                else {
                    @Suppress("UNCHECKED_CAST")
                    val out = u.list("content").orEmpty().mapNotNull { (it as? Map<String, Any?>)?.obj("content")?.str("text") }
                        .joinToString("\n")
                    AgentEvent.ToolResult(u.str("toolCallId").orEmpty(), status == "completed",
                        if (out.length > MAX_OUTPUT) out.take(MAX_OUTPUT) + "\n…" else out)
                }
            }
            else -> null
        }
    }

    private suspend fun request(method: String, params: Map<String, Any?>): Map<String, Any?> {
        val id = ids.incrementAndGet()
        val done = CompletableDeferred<Map<String, Any?>>()
        pending[id] = done
        try {
            write(mapOf("jsonrpc" to "2.0", "id" to id, "method" to method, "params" to params))
            val m = done.await()
            @Suppress("UNCHECKED_CAST")
            (m["error"] as? Map<String, Any?>)?.let { throw AgentException(it.str("message") ?: method) }
            @Suppress("UNCHECKED_CAST")
            return m["result"] as? Map<String, Any?> ?: emptyMap()
        } finally {
            pending.remove(id)
        }
    }

    private fun notify(method: String, params: Map<String, Any?>) =
        runCatching { write(mapOf("jsonrpc" to "2.0", "method" to method, "params" to params)) }

    @Synchronized
    private fun write(m: Map<String, Any?>) {
        process.outputStream.write((Json.write(m) + "\n").toByteArray())
        process.outputStream.flush()
    }

    /** The harness's messages: answers to ours, updates, and its own requests (a permission: allowed - the agent
     *  may do anything in its own Linux; anything else: not offered). */
    @Suppress("UNCHECKED_CAST")
    private suspend fun read() {
        try {
            process.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    val m = runCatching { Json.parse(line) as? Map<String, Any?> }.getOrNull() ?: continue
                    val method = m.str("method")
                    val id = (m["id"] as? Number)?.toInt()
                    when {
                        method == null && id != null -> pending[id]?.complete(m)
                        method == "session/update" -> m.obj("params")?.obj("update")?.let { u ->
                            if (u.str("sessionUpdate") == "available_commands_update") onCommands(commands(u))
                            else onUpdate?.invoke(u)
                        }
                        // geode-agent starts or ends its Linux (a command, an MCP server): the model lends it memory
                        method == "geode/linux" -> {
                            val on = m.obj("params")?.get("active") == true
                            val id = m["id"]
                            scope.launch {
                                lend(on)
                                if (id != null) write(mapOf("jsonrpc" to "2.0", "id" to id, "result" to emptyMap<String, Any?>()))
                            }
                        }
                        method == "session/request_permission" && m["id"] != null -> {
                            val options = m.obj("params")?.list("options").orEmpty().mapNotNull { it as? Map<String, Any?> }
                            val allow = options.firstOrNull { it.str("kind")?.startsWith("allow") == true } ?: options.firstOrNull()
                            write(mapOf("jsonrpc" to "2.0", "id" to m["id"], "result" to mapOf("outcome" to
                                (allow?.let { mapOf("outcome" to "selected", "optionId" to it["optionId"]) }
                                    ?: mapOf("outcome" to "cancelled")))))
                        }
                        method != null && m["id"] != null -> write(mapOf("jsonrpc" to "2.0", "id" to m["id"],
                            "error" to mapOf("code" to -32601, "message" to "not offered")))
                    }
                }
            }
        } catch (_: java.io.IOException) {
            // its output closed under the read: the harness stopped (close() on another thread)
        } finally {
            // the harness went: whoever waits learns why
            val why = synchronized(errors) { errors.lines().lastOrNull { it.startsWith("dsh:") }?.removePrefix("dsh:")?.trim() }
            for (d in pending.values) d.complete(mapOf("error" to mapOf("message" to (why ?: "the agent stopped"))))
        }
    }

    /** Whether the model has lent the agent's Linux its share now. */
    @Volatile
    private var lent = false

    /**
     * The model's memory lent to the agent's Linux while it runs ([on]), or taken back: the engine gives back
     * [GeodeApp.AGENT_RESERVE] of its expert cache, and takes it again after. A failure leaves the model as it was.
     */
    private suspend fun lend(on: Boolean) {
        if (on == lent) return
        lent = on
        runCatching { app.engine.reserveMemory(if (on) GeodeApp.AGENT_RESERVE else 0L) }
    }

    fun close() {
        // what the model lent comes back with the harness gone (its Linux goes with it)
        if (lent) app.appScope.launch { lend(false) }
        runCatching { process.destroy() }
        server.stop()
        scope.launch {
            delay(3000)
            runCatching { if (process.isAlive) process.destroyForcibly() }
        }
    }

    companion object {
        /** geode-agent's file in the native library folder (tools/build_agent.py). */
        private const val NATIVE = "libgeodeagent.so"
        /** Where the user's images go in the chat's folder (a dot folder: not among the files the agent made). */
        private const val IMAGES = ".geode/images"
        private const val AUDIO = ".geode/audio"
        private const val MIGRATE_S = 600L

        /**
         * The harness's start (DeepSeek Harness in the Linux, a build without geode-agent). A chat whose Linux last ran another image (the app updated it) first runs the new
         * image's /etc/geode/migrate with the old version, if it has one: the place an image moves what a chat keeps
         * (its /root, /usr/local, /workspace) to what it needs - out of the protocol's stdout.
         */
        private const val START = "v=\$(cat /etc/geode/version 2>/dev/null); was=\$(cat /root/.geode-image 2>/dev/null); " +
            "if [ \"\$v\" != \"\$was\" ]; then " +
            "[ -x /etc/geode/migrate ] && /etc/geode/migrate \"\$was\" >/root/.geode-migrate.log 2>&1 </dev/null; " +
            "printf '%s\\n' \"\$v\" > /root/.geode-image; fi; " +
            "exec dsh --profile acp --patch /etc/geode/dsh-geode.yml --patch /etc/geode/dsh-geode-acp.yml"
        private const val MAX_OUTPUT = 8000
        private const val CANCEL_MS = 10_000L
    }
}
