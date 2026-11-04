package xyz.tifu.geode.agent.browser

import android.app.Presentation
import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.PixelFormat
import android.hardware.HardwareBuffer
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.DisplayMetrics
import android.view.Display
import android.view.InputDevice
import android.view.MotionEvent
import android.webkit.JsPromptResult
import android.webkit.JsResult
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import xyz.tifu.geode.GeodeApp
import java.io.ByteArrayOutputStream
import java.security.SecureRandom
import kotlin.math.roundToInt

/**
 * The agent's browser: one WebView (the phone's Chromium) on a display of its own - a private VirtualDisplay at the
 * phone's screen size, the WebView in a Presentation on it - drawn into an ImageReader. Whatever the chat shows of it
 * (a small card, the whole screen, nothing), the page draws at full size, WebGL included, and goes on drawing with the
 * app in the background: the model's screenshots are those frames (CDP's own leave WebGL out). The agent drives it
 * over CDP ([Cdp]: the app's own DevTools socket); the user sees the frames and touches the page through [touch].
 *
 * One page for the whole app, the chat whose agent used it last its owner ([Ui.chatId]). It pauses a minute after
 * its last use while nothing shows it, and closes after [CLOSE_MS].
 */
class AgentBrowser(private val app: GeodeApp) {
    enum class View { HIDDEN, MINI, FULL }

    data class Ui(
        /** The chat whose agent uses it; null: no browser open. */
        val chatId: String? = null,
        val view: View = View.HIDDEN,
        /** The page's address as the user knows it (the chat's files as /workspace/...). */
        val url: String = "",
        val title: String = "",
        val loading: Boolean = false,
        /** The page's display, pixels. */
        val width: Int = 0,
        val height: Int = 0,
    )

    data class Shot(val jpeg: ByteArray, val width: Int, val height: Int)

    data class Entry(val seq: Long, val level: String, val text: String)

    private val _ui = MutableStateFlow(Ui())
    val ui: StateFlow<Ui> = _ui.asStateFlow()

    /** The latest frame (a hardware bitmap) while a view watches ([watch]); null otherwise. */
    private val _frame = MutableStateFlow<Bitmap?>(null)
    val frame: StateFlow<Bitmap?> = _frame.asStateFlow()

    /** One use of the page at a time: a tool's call, the user's typing. */
    val lock = Mutex()

    private val main = Handler(Looper.getMainLooper())
    private val frameThread = HandlerThread("geode-browser").apply { start() }
    private val frameHandler = Handler(frameThread.looper)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val srgb = ColorSpace.get(ColorSpace.Named.SRGB)
    private val random = SecureRandom()

    // main thread's
    private var display: VirtualDisplay? = null
    private var presentation: Presentation? = null
    private var web: WebView? = null
    private var reader: ImageReader? = null
    private var paused = false

    @Volatile
    private var cdp: Cdp? = null

    // the frames: the newest kept open, the one the views were given last and the one before it (a view may still be
    // drawing it: a buffer goes back to the display, to be drawn over, only two showings after it was shown). The
    // renderer acknowledges nothing: a UI stalled longer than that could draw a buffer being drawn over - one torn
    // frame at worst, never a crash (the bitmap holds its own reference to the buffer)
    private val frameLock = Any()
    private var latest: Image? = null
    private var shown: Image? = null
    private var prevShown: Image? = null

    /** The chat's workspace over http for the page (a server of the browser's own: it outlives the agent's). */
    @Volatile
    private var files: xyz.tifu.geode.server.HttpServer? = null

    /** What the file server serves: one chat's workspace at one token (one object: a request sees both or neither). */
    private class Served(val root: java.io.File, val token: String, val base: String)

    @Volatile
    private var serving: Served? = null

    /** Counts the closes: a start a close overtook gives up what it opened. (main thread's) */
    @Volatile
    private var generation = 0

    @Volatile
    private var lastFrameAt = 0L

    @Volatile
    private var watchers = 0
    private var lastPublish = 0L
    private var publishQueued = false

    @Volatile
    private var lastUse = 0L
    private var userHidden = false
    private var idle: Job? = null

    /** Display pixels per CSS pixel. */
    @Volatile
    var density = 1f
        private set

    /** Display pixels per pixel of the last screenshot (act's x and y are that image's). */
    @Volatile
    var shotScale = 1f
        private set

    /** The address the chat's workspace is served at (the browser's own loopback server): shown as /workspace/. */
    val served: String? get() = serving?.base

    private val console = ArrayDeque<Entry>()
    private var seq = 0L
    private var readTo = 0L

    @Volatile
    private var loaded: CompletableDeferred<Unit>? = null

    /**
     * The page ready for [chatId]'s agent, inside [lock]: opened (or opened again after it closed), another chat's
     * page left for a blank one, shown (unless the user hid it), resumed.
     */
    suspend fun ready(chatId: String): Cdp {
        lastUse = SystemClock.elapsedRealtime()
        val c = cdp
        if (c != null && !c.closed && web != null) {
            if (_ui.value.chatId == chatId) {
                if (!userHidden && _ui.value.view == View.HIDDEN) _ui.update { it.copy(view = View.MINI) }
                resume()
                return c
            }
            // another chat's page: a blank one, with none of the other chat's pages to go back to, and the other
            // chat's files no longer (a new token, this chat's workspace) - or, the blank page not loaded, a new
            // browser altogether (below)
            val waiter = expectLoad()
            val blank = runCatching { c.call("Page.navigate", mapOf("url" to "about:blank")) }.isSuccess &&
                awaitLoad(waiter, BLANK_MS)
            if (blank && withContext(Dispatchers.Main) { web?.apply { clearHistory() } != null }) {
                serveChat(chatId)
                clearConsole()
                userHidden = false
                _ui.update { it.copy(chatId = chatId, url = "", title = "", view = View.MINI) }
                resume()
                return c
            }
        }
        val marker = "geode-browser-" + hex(8)
        val gen = withContext(Dispatchers.Main) {
            release()
            create(marker)
            generation
        }
        var server: xyz.tifu.geode.server.HttpServer? = null
        var conn: Cdp? = null
        try {
            server = xyz.tifu.geode.server.HttpServer(0, ::serveFile,
                java.net.InetAddress.getByAddress(byteArrayOf(127, 0, 0, 1)))
            withContext(Dispatchers.IO) { server.start() }
            conn = withContext(Dispatchers.IO) { Cdp.connect(target(marker), ::onEvent) }
            conn.call("Page.enable")
            conn.call("Runtime.enable")
            conn.call("Log.enable")
            // what it opened made the browser's on the main thread, unless a close came in between (the settings, the
            // page's process gone): then it is given up here
            val live = withContext(Dispatchers.Main) {
                // (the settings turn it off before they close it: a close still queued behind this is seen here)
                if (generation != gen || web == null || !app.agentConfig.browser.value) return@withContext false
                files = server
                cdp = conn
                serveChat(chatId)
                clearConsole()
                userHidden = false
                _ui.value = Ui(chatId = chatId, view = View.MINI, width = width, height = height)
                true
            }
            if (!live) throw CdpException("it was closed while it started")
            watchIdle()
            return conn
        } catch (e: Exception) {
            conn?.close()
            server?.stop()
            if (generation == gen) close()
            if (e is kotlinx.coroutines.CancellationException) throw e
            throw CdpException("the browser did not start: ${e.message}")
        }
    }

    /** [chatId]'s workspace is what the file server serves, at a new token. */
    private fun serveChat(chatId: String) {
        val server = files ?: return
        val token = hex(16)
        serving = Served(app.agent.workspace(chatId), token, "http://127.0.0.1:${server.boundPort}/w/$token/")
    }

    @Volatile
    var width = 0
        private set

    @Volatile
    var height = 0
        private set

    private fun hex(bytes: Int) = ByteArray(bytes).also(random::nextBytes).joinToString("") { "%02x".format(it) }

    /** The address a file of the chat's workspace ([path], relative to it) loads from; null: the browser is closed. */
    fun fileUrl(path: String): String? {
        val base = served ?: return null
        return base + path.trimStart('/').split('/').joinToString("/") { Uri.encode(it) }
    }

    /**
     * The chat's workspace, read-only, to the page alone: an unguessable token in every address, never another site's
     * to read (no CORS), never told to the sites the page loads from (no referrer), nothing outside the workspace.
     */
    private suspend fun serveFile(req: xyz.tifu.geode.server.HttpRequest, r: xyz.tifu.geode.server.HttpResponder) {
        r.cors = false
        // the page's icon, asked for at the server's root: nothing, quietly (a 404 would be an error in its console)
        if (req.path == "/favicon.ico") return r.send(204, "image/x-icon", ByteArray(0))
        // (one read: the workspace and its token together, whatever chat comes next)
        val s = serving
        val prefix = s?.let { "/w/${it.token}/" }
        if (s == null || prefix == null || !req.path.startsWith(prefix) || (req.method != "GET" && req.method != "HEAD"))
            return r.send(404, "text/plain", "Not found".toByteArray())
        val rel = req.path.removePrefix(prefix)
        var redirect = false
        val f = withContext(Dispatchers.IO) {
            val base = s.root.canonicalFile
            // the path as it really is (links followed: the agent's Linux can make them), inside the workspace
            fun inside(f: java.io.File) = f.toPath().startsWith(base.toPath())
            val asked = java.io.File(base, rel).canonicalFile
            when {
                !inside(asked) -> null
                asked.isDirectory && rel.isNotEmpty() && !rel.endsWith("/") -> {
                    // a folder's address ends with a slash: its page's relative files are the folder's
                    redirect = true
                    null
                }
                asked.isDirectory -> java.io.File(asked, "index.html").canonicalFile.takeIf(::inside)
                else -> asked
            }?.takeIf { it.isFile && it.length() <= FILE_MAX }
        }
        if (redirect) return r.send(301, "text/plain", ByteArray(0), mapOf("Location" to
            req.path.split('/').joinToString("/") { Uri.encode(it) } + "/" +
            req.query.entries.joinToString("&") { (k, v) -> Uri.encode(k) + "=" + Uri.encode(v) }.let { if (it.isEmpty()) "" else "?$it" }))
        f ?: return r.send(404, "text/plain", "Not found: /workspace/$rel".toByteArray())
        val bytes = withContext(Dispatchers.IO) { f.readBytes() }
        r.send(200, mime(f.name), if (req.method == "HEAD") ByteArray(0) else bytes,
            mapOf("Cache-Control" to "no-store", "Referrer-Policy" to "no-referrer", "X-Content-Type-Options" to "nosniff"))
    }

    private fun mime(name: String): String = when (name.substringAfterLast('.', "").lowercase()) {
        "html", "htm" -> "text/html; charset=utf-8"
        "js", "mjs" -> "text/javascript; charset=utf-8"
        "css" -> "text/css; charset=utf-8"
        "json", "map" -> "application/json; charset=utf-8"
        "txt", "md", "csv" -> "text/plain; charset=utf-8"
        "svg" -> "image/svg+xml"
        "png" -> "image/png"
        "jpg", "jpeg" -> "image/jpeg"
        "gif" -> "image/gif"
        "webp" -> "image/webp"
        "ico" -> "image/x-icon"
        "wasm" -> "application/wasm"
        "glb" -> "model/gltf-binary"
        "gltf" -> "model/gltf+json"
        "mp3" -> "audio/mpeg"
        "wav" -> "audio/wav"
        "ogg" -> "audio/ogg"
        "mp4" -> "video/mp4"
        "webm" -> "video/webm"
        "woff" -> "font/woff"
        "woff2" -> "font/woff2"
        "ttf" -> "font/ttf"
        "xml" -> "application/xml"
        else -> "application/octet-stream"
    }

    private fun create(marker: String) {
        WebView.setWebContentsDebuggingEnabled(true)
        val dm = app.getSystemService(DisplayManager::class.java)
        val real = DisplayMetrics()
        @Suppress("DEPRECATION")
        dm.getDisplay(Display.DEFAULT_DISPLAY).getRealMetrics(real)
        width = minOf(real.widthPixels, real.heightPixels)
        height = maxOf(real.widthPixels, real.heightPixels)
        density = real.density
        // (the screenshot it would take: act's x and y before the agent took one)
        shotScale = maxOf(1f, height.toFloat() / SHOT_MAX)
        val r = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 4,
            HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE or HardwareBuffer.USAGE_GPU_COLOR_OUTPUT)
        r.setOnImageAvailableListener({ onFrame(it) }, frameHandler)
        reader = r
        val d = dm.createVirtualDisplay("geode-browser", width, height, real.densityDpi, r.surface, 0)
        display = d
        // the application's context: the page outlives the chat's activity
        val p = Presentation(app, d.display)
        val w = WebView(p.context)
        w.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false
            // the chat's files come over http (the harness's server), not file://
            allowFileAccess = false
            allowContentAccess = false
            useWideViewPort = true
            loadWithOverviewMode = true
        }
        w.webViewClient = object : WebViewClient() {
            override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
                note("error", "The page's process ended (${if (detail.didCrash()) "it crashed" else "the phone needed the memory"}); the next call opens the browser again.")
                main.post { close() }
                return true
            }
        }
        w.webChromeClient = object : WebChromeClient() {
            // a page's dialogs answered (no one sees a dialog on this display), and told to the console
            override fun onJsAlert(view: WebView, url: String?, message: String?, result: JsResult): Boolean {
                note("dialog", "alert(\"$message\")")
                result.confirm()
                return true
            }

            override fun onJsConfirm(view: WebView, url: String?, message: String?, result: JsResult): Boolean {
                note("dialog", "confirm(\"$message\"): answered OK")
                result.confirm()
                return true
            }

            override fun onJsPrompt(view: WebView, url: String?, message: String?, defaultValue: String?,
                                    result: JsPromptResult): Boolean {
                note("dialog", "prompt(\"$message\"): answered \"${defaultValue.orEmpty()}\"")
                result.confirm(defaultValue.orEmpty())
                return true
            }

            override fun onReceivedTitle(view: WebView, title: String?) {
                if (title != null && !title.startsWith("geode-browser-")) _ui.update { it.copy(title = title) }
            }

            override fun onProgressChanged(view: WebView, progress: Int) {
                _ui.update { it.copy(loading = progress < 100) }
            }
        }
        p.setContentView(w)
        p.show()
        w.loadDataWithBaseURL(null, "<title>$marker</title>", "text/html", "utf-8", null)
        presentation = p
        web = w
        paused = false
    }

    /** The DevTools target of the page that shows [marker] as its title: its WebSocket path. */
    @Suppress("UNCHECKED_CAST")
    private suspend fun target(marker: String): String {
        val until = SystemClock.elapsedRealtime() + 8000
        while (true) {
            runCatching {
                (xyz.tifu.geode.util.Json.parse(Cdp.http("/json/list")) as? List<Map<String, Any?>>)
                    ?.firstOrNull { it["title"] == marker }?.get("webSocketDebuggerUrl") as? String
            }.getOrNull()?.let { return it.substring(it.indexOf('/', "ws://".length)) }
            if (SystemClock.elapsedRealtime() > until) throw CdpException("its page did not appear")
            delay(150)
        }
    }

    // ---- what the page says ----

    @Suppress("UNCHECKED_CAST")
    private fun onEvent(method: String, p: Map<String, Any?>) {
        when (method) {
            "Runtime.consoleAPICalled" -> {
                val args = (p["args"] as? List<Map<String, Any?>>).orEmpty().joinToString(" ") { a ->
                    when (val v = a["value"]) {
                        null -> (a["description"] ?: a["unserializableValue"] ?: a["type"]).toString()
                        is String -> v
                        else -> xyz.tifu.geode.util.Json.write(v)
                    }
                }
                val level = when (val t = p["type"] as? String) {
                    "warning" -> "warning"
                    "error", "assert" -> "error"
                    else -> t ?: "log"
                }
                note(level, args + where((p["stackTrace"] as? Map<String, Any?>)?.let {
                    (it["callFrames"] as? List<Map<String, Any?>>)?.firstOrNull()
                }))
            }
            "Runtime.exceptionThrown" -> {
                val d = p["exceptionDetails"] as? Map<String, Any?> ?: return
                val desc = ((d["exception"] as? Map<String, Any?>)?.get("description") as? String)?.lineSequence()?.firstOrNull()
                note("error", "Uncaught " + (desc ?: d["text"]) + where(d))
            }
            "Log.entryAdded" -> {
                val e = p["entry"] as? Map<String, Any?> ?: return
                val url = (e["url"] as? String)?.let { " " + shown(it) }.orEmpty()
                note(e["level"] as? String ?: "info", (e["text"] as? String).orEmpty() + url)
            }
            "Page.loadEventFired" -> loaded?.complete(Unit)
            "Page.frameNavigated" -> {
                val f = p["frame"] as? Map<String, Any?> ?: return
                if (f["parentId"] == null) (f["url"] as? String)?.let { u ->
                    if (u != "about:blank" || _ui.value.url.isNotEmpty()) _ui.update { it.copy(url = shown(u)) }
                }
            }
        }
    }

    /** " (file:line)" of a stack frame or an exception's details. */
    private fun where(f: Map<String, Any?>?): String {
        f ?: return ""
        val url = (f["url"] as? String)?.takeIf { it.isNotEmpty() } ?: return ""
        val line = (f["lineNumber"] as? Number)?.toInt()?.plus(1)
        return " (" + shown(url).substringAfterLast('/').ifEmpty { shown(url) } + (line?.let { ":$it" } ?: "") + ")"
    }

    /** An address as the user knows it: the chat's files as /workspace/... */
    fun shown(url: String): String {
        val s = served ?: return url
        return if (url.startsWith(s)) "/workspace/" + Uri.decode(url.removePrefix(s)) else url
    }

    private fun note(level: String, text: String) = synchronized(console) {
        console.addLast(Entry(++seq, level, text.take(2000)))
        while (console.size > CONSOLE_MAX) console.removeFirst()
    }

    private fun clearConsole() = synchronized(console) {
        console.clear()
        readTo = seq
    }

    /** The console's entries since the last time they were read ([all]: every one kept). */
    fun consoleSince(all: Boolean = false): List<Entry> = synchronized(console) {
        val out = console.filter { all || it.seq > readTo }
        readTo = seq
        out
    }

    /** The next load event's waiter (Page.loadEventFired), set before a navigation starts. */
    fun expectLoad(): CompletableDeferred<Unit> = CompletableDeferred<Unit>().also { loaded = it }

    // ---- the frames ----

    private fun onFrame(r: ImageReader) {
        val img = try {
            r.acquireLatestImage()
        } catch (e: IllegalStateException) {
            null
        } ?: return
        val old: Image?
        synchronized(frameLock) {
            // the one before goes back, unless the views were given it
            old = latest?.takeIf { it !== shown && it !== prevShown }
            latest = img
            lastFrameAt = SystemClock.elapsedRealtime()
        }
        old?.close()
        if (watchers > 0) publish()
    }

    /** The newest frame to the views, ~30 a second (the last one always: a page that stops changing shows it). */
    private fun publish() {
        val now = SystemClock.elapsedRealtime()
        // the whole screen at the display's pace (a game played), the small card at half
        val wait = (if (_ui.value.view == View.FULL) FRAME_FULL_MS else FRAME_MS) - (now - lastPublish)
        if (wait > 0) {
            if (!publishQueued) {
                publishQueued = true
                frameHandler.postDelayed({
                    publishQueued = false
                    publish()
                }, wait)
            }
            return
        }
        lastPublish = now
        if (watchers <= 0) return
        var done: Image? = null
        _frame.value = synchronized(frameLock) {
            val img = latest ?: return
            if (img === shown) return
            // the one shown two showings back goes back: the views have drawn the one after it since
            done = prevShown?.takeIf { it !== latest }
            prevShown = shown
            shown = img
            img.hardwareBuffer?.use { Bitmap.wrapHardwareBuffer(it, srgb) }
        }
        done?.close()
    }

    /** A view shows the frames ([on]) or no longer does. */
    fun watch(on: Boolean) {
        frameHandler.post {
            watchers = maxOf(0, watchers + if (on) 1 else -1)
            if (watchers > 0) {
                lastPublish = 0
                publish()
            } else {
                _frame.value = null
                // the frames no view shows any longer go back to the display, once the last view has surely drawn
                val done = synchronized(frameLock) {
                    listOfNotNull(shown, prevShown).filter { it !== latest }.also {
                        shown = null
                        prevShown = null
                    }
                }
                if (done.isNotEmpty()) frameHandler.postDelayed({ done.forEach { it.close() } }, RETIRE_MS)
            }
        }
        if (on) resume()
    }

    /**
     * What the page shows, as a JPEG at most [maxSide] pixels long: the first frame drawn after [after] (an action's
     * result), or once none comes, the last.
     */
    suspend fun screenshot(maxSide: Int = SHOT_MAX, after: Long = 0): Shot = withContext(Dispatchers.Default) {
        val start = SystemClock.elapsedRealtime()
        while (lastFrameAt < after && SystemClock.elapsedRealtime() - start < SETTLE_MS) delay(25)
        while (synchronized(frameLock) { latest == null } && SystemClock.elapsedRealtime() - start < 4000) delay(50)
        val soft = synchronized(frameLock) {
            val img = latest ?: throw CdpException("the browser has drawn nothing yet")
            img.hardwareBuffer?.use { hb ->
                Bitmap.wrapHardwareBuffer(hb, srgb)?.let { b -> b.copy(Bitmap.Config.ARGB_8888, false).also { b.recycle() } }
            }
        } ?: throw CdpException("the frame could not be read")
        val scale = minOf(1f, maxSide.toFloat() / maxOf(soft.width, soft.height))
        val out = if (scale < 1f) Bitmap.createScaledBitmap(soft, (soft.width * scale).roundToInt(),
            (soft.height * scale).roundToInt(), true).also { soft.recycle() } else soft
        val bytes = ByteArrayOutputStream().also { out.compress(Bitmap.CompressFormat.JPEG, 85, it) }.toByteArray()
        shotScale = width.toFloat() / out.width
        val shot = Shot(bytes, out.width, out.height)
        out.recycle()
        shot
    }

    // ---- the user ----

    /** A touch of the user's on the page, at display pixels [x], [y] (the full view's). */
    fun touch(action: Int, x: Float, y: Float, downTime: Long) {
        lastUse = SystemClock.elapsedRealtime()
        resume()
        main.post {
            val w = web ?: return@post
            val ev = MotionEvent.obtain(downTime, SystemClock.uptimeMillis(), action, x, y, 0)
            ev.source = InputDevice.SOURCE_TOUCHSCREEN
            w.dispatchTouchEvent(ev)
            ev.recycle()
        }
    }

    /** The agent's touch: a tap at display pixels [x], [y] (down, a moment, up), as a finger does it. */
    suspend fun tap(x: Float, y: Float) {
        val down = SystemClock.uptimeMillis()
        motion(MotionEvent.ACTION_DOWN, x, y, down)
        delay(TAP_MS)
        motion(MotionEvent.ACTION_UP, x, y, down)
    }

    /** The agent's swipe from ([x1], [y1]) to ([x2], [y2]), display pixels, in [ms]. */
    suspend fun swipe(x1: Float, y1: Float, x2: Float, y2: Float, ms: Long) {
        val down = SystemClock.uptimeMillis()
        motion(MotionEvent.ACTION_DOWN, x1, y1, down)
        val steps = (ms / 16).coerceIn(2, 120).toInt()
        for (i in 1..steps) {
            delay(ms / steps)
            val t = i.toFloat() / steps
            motion(MotionEvent.ACTION_MOVE, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, down)
        }
        motion(MotionEvent.ACTION_UP, x2, y2, down)
    }

    /** One touch event on the page, dispatched before this returns; false: no page. */
    private suspend fun motion(action: Int, x: Float, y: Float, downTime: Long): Boolean = withContext(Dispatchers.Main) {
        val w = web ?: return@withContext false
        val ev = MotionEvent.obtain(downTime, SystemClock.uptimeMillis(), action, x, y, 0)
        ev.source = InputDevice.SOURCE_TOUCHSCREEN
        w.dispatchTouchEvent(ev)
        ev.recycle()
        true
    }

    /** The user's text into the page's focused field ([text]), or a key ([key]: Enter, Backspace). */
    fun type(text: String? = null, key: String? = null) {
        lastUse = SystemClock.elapsedRealtime()
        scope.launch {
            lock.withLock {
                val c = cdp ?: return@withLock
                runCatching {
                    if (!text.isNullOrEmpty()) c.call("Input.insertText", mapOf("text" to text))
                    if (key != null) Keys.press(c, key)
                }
            }
        }
    }

    /** The user's view of it: [View.HIDDEN] stays hidden while the agent goes on using it. */
    fun show(view: View) {
        if (_ui.value.chatId == null) return
        userHidden = view == View.HIDDEN
        _ui.update { it.copy(view = view) }
        if (view != View.HIDDEN) resume()
    }

    fun reload() {
        lastUse = SystemClock.elapsedRealtime()
        main.post { web?.reload() }
    }

    // ---- its life ----

    private fun resume() {
        main.post {
            if (paused) web?.onResume()
            paused = false
        }
    }

    /** Paused a minute after its last use while no view shows it; closed after [CLOSE_MS]. */
    private fun watchIdle() {
        idle?.cancel()
        idle = scope.launch {
            while (isActive) {
                delay(IDLE_CHECK_MS)
                // (a tool's call holds the lock: a long eval or a load is no quiet)
                if (watchers > 0 || lock.isLocked) continue
                val quiet = SystemClock.elapsedRealtime() - lastUse
                if (quiet > CLOSE_MS) {
                    close()
                    return@launch
                }
                if (quiet > PAUSE_MS) main.post {
                    if (!paused) web?.onPause()
                    paused = true
                }
            }
        }
    }

    /** Closed: its page, its display, its memory (the next tool's call opens it again). */
    fun close() {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            main.post { close() }
            return
        }
        idle?.cancel()
        idle = null
        release()
        _ui.value = Ui()
    }

    /** (main thread) */
    private fun release() {
        // a start under way gives up (ready: its generation)
        generation++
        cdp?.close()
        cdp = null
        files?.stop()
        files = null
        serving = null
        // DevTools for this process's WebViews only while the browser is open (a debug build has it always)
        if (web != null && app.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE == 0)
            runCatching { WebView.setWebContentsDebuggingEnabled(false) }
        loaded?.cancel()
        loaded = null
        runCatching { presentation?.dismiss() }
        presentation = null
        runCatching { web?.destroy() }
        web = null
        runCatching { display?.release() }
        display = null
        val r = reader
        reader = null
        frameHandler.post {
            synchronized(frameLock) {
                listOfNotNull(latest, shown, prevShown).distinct().forEach { it.close() }
                latest = null
                shown = null
                prevShown = null
            }
            runCatching { r?.close() }
            _frame.value = null
        }
    }

    /** Waits for the page's load (after a navigation [waiter] was set for) at most [ms]; false: it did not come. */
    suspend fun awaitLoad(waiter: CompletableDeferred<Unit>, ms: Long): Boolean =
        withTimeoutOrNull(ms) { waiter.await() } != null

    /** The time a frame drawn after now would be newer than (screenshot's after). */
    fun now() = SystemClock.elapsedRealtime()

    companion object {
        private const val CONSOLE_MAX = 300
        private const val FRAME_MS = 33L
        private const val FRAME_FULL_MS = 15L
        private const val SETTLE_MS = 700L
        private const val TAP_MS = 60L
        /** A blank page for another chat waited for so long. */
        private const val BLANK_MS = 3000L
        /** Frames no view shows any longer go back to the display after this (a last draw of them may be under way). */
        private const val RETIRE_MS = 100L
        const val SHOT_MAX = 1280
        private const val IDLE_CHECK_MS = 15_000L
        private const val PAUSE_MS = 60_000L
        /** Unused and unseen this long, it closes: its renderer's memory is the model's again. */
        private const val CLOSE_MS = 5 * 60_000L
        /** The largest file of the workspace the page is served. */
        private const val FILE_MAX = 64L shl 20
    }
}
