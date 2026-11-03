package xyz.tifu.geode.server

import android.util.Base64
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.data.ChatStats
import xyz.tifu.geode.data.ChatStore
import xyz.tifu.geode.agent.AgentEvent
import xyz.tifu.geode.agent.AgentTurns
import xyz.tifu.geode.agent.finished
import xyz.tifu.geode.agent.withAgent
import xyz.tifu.geode.ui.chat.ChatMessage
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.str
import xyz.tifu.geode.vision.ImageStore
import java.io.IOException
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.ConcurrentHashMap

/**
 * The web chat: a small page (assets/web/index.html) any browser on the local network opens at the server's address,
 * with the app's own chats - read them, go on with them, start new ones, delete them. It asks for the password the
 * Local server screen shows (a random one; a new one signs everybody out); a login is a session cookie (HttpOnly,
 * SameSite=Strict: no other site's page can use it). A message goes to the chat's agent as the phone's would (the
 * same Agent, the chat's own session, Linux and folder, its tools, web search, the user's skills and MCP servers), so
 * the turns from the browser are the agent's conversation the phone goes on with, and the other way round; the
 * browser sees what the agent does as it does it (a line per tool, then the answer). Answers are saved into the chat
 * as the app saves them; the app's chat follows ([changes]). An answer goes on when the browser goes away.
 */
class WebChat(private val app: GeodeApp, private val engine: ApiEngine, private val password: () -> String?) {
    private val store = ChatStore(app)
    private val images = ImageStore(app)
    private val random = SecureRandom()
    /** Session token -> when it expires. */
    private val sessions = ConcurrentHashMap<String, Long>()
    /** When the logins that failed in the last minute did. */
    private val failures = ArrayDeque<Long>()
    @Volatile
    private var template: String? = null
    /** The chat the browser's message is answered in, and its request's job (Stop cancels it). */
    @Volatile
    private var running: String? = null
    @Volatile
    private var runningJob: Job? = null

    /** The password changed: every browser signs in again. */
    fun forget() = sessions.clear()

    suspend fun handle(req: HttpRequest, r: HttpResponder) {
        r.cors = false
        val path = req.path.trimEnd('/').ifEmpty { "/" }
        if (req.method == "GET" && (path == "/" || path == "/index.html")) return page(r)
        if (req.method == "POST" && path == "/web/login") return login(req, r)
        if (req.method == "POST" && path == "/web/qr") return qrStart(req, r)
        if (req.method == "GET" && path.startsWith("/web/qr/")) return qrWait(r, path.removePrefix("/web/qr/"))
        if (!signedIn(req)) return r.json(401, mapOf("error" to "login"))
        try {
            when {
                req.method == "GET" && path == "/web/me" -> me(r)
                req.method == "POST" && path == "/web/logout" -> logout(req, r)
                req.method == "GET" && path == "/web/chats" -> r.json(200, withContext(Dispatchers.IO) { store.list() }
                    .map { linkedMapOf("id" to it.id, "title" to it.title, "updated" to it.updated) })
                req.method == "GET" && path.startsWith("/web/chats/") -> chat(r, path.removePrefix("/web/chats/"))
                req.method == "DELETE" && path.startsWith("/web/chats/") -> delete(r, path.removePrefix("/web/chats/"))
                req.method == "GET" && path.startsWith("/web/img/") -> image(r, path.removePrefix("/web/img/"))
                req.method == "POST" && path == "/web/send" -> send(req, r)
                req.method == "POST" && path == "/web/stop" -> stop(r)
                else -> r.json(404, mapOf("error" to "No such page: ${req.method} ${req.path}"))
            }
        } catch (e: ApiError) {
            if (!r.started) r.json(e.status, mapOf("error" to e.message))
        }
    }

    // ---- sign-in ----

    /** The page in the app's language: its strings, its language tag and direction put in (the page's English stays
     *  for any string missing). */
    private fun page(r: HttpResponder) {
        val html = template ?: String(app.assets.open("web/index.html").use { it.readBytes() }, Charsets.UTF_8)
            .also { template = it }
        val (locale, strings) = pageStrings()
        val rtl = android.text.TextUtils.getLayoutDirectionFromLocale(locale) == android.view.View.LAYOUT_DIRECTION_RTL
        val localized = html.replace("/*GEODE_STRINGS*/{}", Json.write(strings).replace("</", "<\\/"))
            .replace("/*GEODE_LANG*/'en'", Json.write(locale.stripExtensions().toLanguageTag()))
            .replace("/*GEODE_DIR*/'ltr'", if (rtl) "'rtl'" else "'ltr'")
        r.send(200, "text/html; charset=utf-8", localized.toByteArray(Charsets.UTF_8), mapOf(
            "Cache-Control" to "no-cache",
            "X-Frame-Options" to "DENY",
            "Referrer-Policy" to "no-referrer",
            "Content-Security-Policy" to "default-src 'self'; img-src 'self' data: blob:; style-src 'self' " +
                "'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        ))
    }

    /**
     * The page's strings in the app's language (its Settings choice, else the phone's; Android's English where a
     * language lacks one): the app's own translations, the same words as its screens.
     */
    private fun pageStrings(): Pair<java.util.Locale, Map<String, String>> {
        val locale = androidx.appcompat.app.AppCompatDelegate.getApplicationLocales()[0]
            ?: app.resources.configuration.locales[0] ?: java.util.Locale.getDefault()
        val cfg = android.content.res.Configuration(app.resources.configuration).apply { setLocale(locale) }
        val res = app.createConfigurationContext(cfg).resources
        fun t(id: Int) = res.getString(id)
        val server = t(R.string.nav_server)
        return locale to linkedMapOf(
            "newChat" to t(R.string.chat_new),
            "untitled" to t(R.string.chat_untitled),
            "message" to t(R.string.agent_input_hint),
            "send" to t(R.string.chat_send),
            "stop" to t(R.string.chat_stop),
            "signIn" to t(R.string.qr_confirm_yes),
            "password" to t(R.string.server_web_password),
            "wrong" to t(R.string.web_wrong_password),
            "tooMany" to t(R.string.web_too_many),
            "hint" to res.getString(R.string.web_password_hint, server),
            "thinking" to t(R.string.chat_thinking),
            "summary" to t(R.string.chat_summary_title),
            "del" to t(R.string.chat_delete),
            "delAsk" to t(R.string.web_delete_ask),
            "empty" to t(R.string.chat_empty),
            "noModel" to t(R.string.chat_info_no_model),
            "signOut" to t(R.string.web_sign_out),
            "attach" to t(R.string.chat_attach_image),
            "waiting" to t(R.string.web_waiting),
            // with their %1$d / %1$.1f: the page puts the number in
            "reading" to t(R.string.agent_reading),
            "tps" to t(R.string.chat_speed),
            "qr" to res.getString(R.string.web_qr, server, t(R.string.server_web_scan)),
            "or" to t(R.string.web_or),
            "installing" to t(R.string.agent_starting),
            "files" to t(R.string.agent_files),
        )
    }

    private fun token(req: HttpRequest): String? = req.header("cookie")?.split(';')?.map { it.trim() }
        ?.firstOrNull { it.startsWith("$COOKIE=") }?.substringAfter('=')

    private fun signedIn(req: HttpRequest): Boolean {
        val t = token(req) ?: return false
        val until = sessions[t] ?: return false
        if (until < System.currentTimeMillis()) {
            sessions.remove(t)
            return false
        }
        return true
    }

    /** The password as typed, letters and digits only, lower case (the screen shows it in groups). */
    private fun normal(p: String) = p.filter { it.isLetterOrDigit() }.lowercase()

    private fun login(req: HttpRequest, r: HttpResponder) {
        val now = System.currentTimeMillis()
        synchronized(failures) {
            while (failures.isNotEmpty() && failures.first() < now - 60_000) failures.removeFirst()
            if (failures.size >= MAX_FAILURES) {
                return r.json(429, mapOf("error" to "Too many wrong passwords: wait a minute."))
            }
        }
        val given = runCatching { (Json.parse(String(req.body, Charsets.UTF_8)) as Map<*, *>)["password"] as String }
            .getOrNull().orEmpty()
        val need = password()?.let(::normal)
        if (need.isNullOrEmpty() || !MessageDigest.isEqual(normal(given).toByteArray(), need.toByteArray())) {
            synchronized(failures) { failures.addLast(now) }
            return r.json(403, mapOf("error" to "Wrong password."))
        }
        r.json(200, mapOf("ok" to true), mapOf("Set-Cookie" to session()))
    }

    /** A new session: its cookie header. */
    private fun session(): String {
        val t = ByteArray(32).also { random.nextBytes(it) }.joinToString("") { "%02x".format(it) }
        sessions[t] = System.currentTimeMillis() + SESSION_MS
        return "$COOKIE=$t; Path=/; HttpOnly; SameSite=Strict; Max-Age=${SESSION_MS / 1000}"
    }

    /** The sign-in page's QR code: a sign-in waiting for the phone's yes (QrLogins). */
    private fun qrStart(req: HttpRequest, r: HttpResponder) {
        val p = QrLogins.create(req.remote, req.header("user-agent").orEmpty())
            ?: return r.json(429, mapOf("error" to "Too many sign-ins wait already."))
        r.json(200, linkedMapOf("id" to p.id, "svg" to QrLogins.svg(QrLogins.SCHEME + p.code),
            "expires" to QrLogins.LIFETIME_MS / 1000))
    }

    /** The browser waits (up to 25 s a call) for the phone's yes; then it has its session. */
    private suspend fun qrWait(r: HttpResponder, id: String) {
        val p = QrLogins.byId(id) ?: return r.json(200, mapOf("state" to "expired"))
        if (withTimeoutOrNull(25_000) { p.approved.await() } == null) {
            return r.json(200, mapOf("state" to if (p.expired) "expired" else "waiting"))
        }
        QrLogins.finish(p)
        r.json(200, mapOf("state" to "ok"), mapOf("Set-Cookie" to session()))
    }

    private fun logout(req: HttpRequest, r: HttpResponder) {
        token(req)?.let { sessions.remove(it) }
        r.json(200, mapOf("ok" to true), mapOf("Set-Cookie" to "$COOKIE=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"))
    }

    // ---- chats ----

    private suspend fun me(r: HttpResponder) {
        val m = runCatching { engine.model() }.getOrNull()
        val s = app.settings.flow.first()
        r.json(200, linkedMapOf("model" to m?.title, "vision" to (m?.vision == true), "thinking" to s.thinking,
            "running" to running))
    }

    private fun json(m: ChatMessage) = linkedMapOf<String, Any?>("id" to m.id, "user" to m.user, "text" to m.text).apply {
        if (m.thinking.isNotEmpty()) put("thinking", m.thinking)
        m.error?.let { put("error", it) }
        if (m.tokensPerSecond > 0f) put("tps", m.tokensPerSecond.toDouble())
        if (m.summary) put("summary", true)
        if (m.images.isNotEmpty()) put("images", m.images)
        // what the agent did on the way: its words between the tools, a line per tool (its main argument, how it ended)
        if (m.steps.isNotEmpty()) put("steps", m.steps.map { st ->
            if (st.tool.isEmpty()) linkedMapOf("words" to st.title)
            else linkedMapOf("id" to st.id, "tool" to st.tool, "args" to AgentTurns.brief(st.input), "ok" to st.ok)
        })
        if (m.files.isNotEmpty()) put("files", m.files)
    }

    /** A chat's id as the store names its files (the time it began, in ms). */
    private fun validId(id: String) = id.isNotEmpty() && id.length <= 20 && id.all { it.isDigit() }

    private suspend fun chat(r: HttpResponder, id: String) {
        if (!validId(id)) throw ApiError(404, "No such chat.")
        val c = withContext(Dispatchers.IO) { store.load(id) } ?: throw ApiError(404, "No such chat.")
        r.json(200, linkedMapOf("id" to id, "title" to c.title, "messages" to c.messages.map(::json),
            "running" to (running == id)))
    }

    private suspend fun delete(r: HttpResponder, id: String) {
        if (!validId(id)) throw ApiError(404, "No such chat.")
        if (running == id || app.agent.turn.value == id) throw ApiError(409, "The chat is being answered.")
        withContext(Dispatchers.IO) { store.delete(id) }
        _changes.tryEmit(id)
        r.json(200, mapOf("ok" to true))
    }

    private suspend fun image(r: HttpResponder, name: String) {
        // a chat image's name: no paths
        if (name.isEmpty() || name.length > 80 || !name.all { it.isLetterOrDigit() || it == '-' || it == '_' || it == '.' } ||
            name.startsWith(".")) throw ApiError(404, "No such image.")
        val bytes = withContext(Dispatchers.IO) { images.file(name).takeIf { it.isFile }?.readBytes() }
            ?: throw ApiError(404, "No such image.")
        r.send(200, "image/jpeg", bytes, mapOf("Cache-Control" to "private, max-age=86400"))
    }

    /** The browser's own answer stopped whole (the agent's turn, its console and whatever it started). */
    private fun stop(r: HttpResponder) {
        if (running != null) {
            runningJob?.cancel()
            // its own turn only (not the phone's it may be waiting for); before the engine has it too
            if (app.agent.turn.value == running) app.engine.stop()
        }
        r.json(200, mapOf("ok" to true))
    }

    /**
     * A message into a chat (a new one without an id) for its agent, and what the agent does as server-sent events:
     * start (the chat, its title, the message as saved), status (it waits: for another turn, for the Linux), started
     * (the prompt read), reasoning and content as they are written, tool and tool_done (a tool's call, its end), then
     * done (the answer as saved) or error.
     */
    private suspend fun send(req: HttpRequest, r: HttpResponder) {
        val b = runCatching { Json.parse(String(req.body, Charsets.UTF_8)) as? Map<String, Any?> }.getOrNull()
            ?: throw ApiError(400, "The body is not a JSON object.")
        val text = b.str("text").orEmpty().trim()
        val sent = b.list("images").orEmpty().filterIsInstance<String>()
        if (text.isEmpty() && sent.isEmpty()) throw ApiError(400, "Nothing to send.")
        if (running != null) throw ApiError(409, "An answer is being written already.")
        val model = engine.model()
        if (sent.isNotEmpty() && !model.vision) throw ApiError(400, "The selected model reads no images.")
        val id = b.str("chat")?.takeIf(::validId) ?: System.currentTimeMillis().toString()
        // the phone's agent is answering this very chat: its answer first
        if (app.agent.turn.value == id) throw ApiError(409, "An answer is being written already.")
        val saved = withContext(Dispatchers.IO) { store.load(id) }
        val names = withContext(Dispatchers.IO) {
            sent.map { b64 ->
                val bytes = runCatching { Base64.decode(b64, Base64.DEFAULT) }.getOrNull()
                    ?: throw ApiError(400, "An image's base64 is broken.")
                runCatching { images.importChatBytes(bytes) }.getOrElse { throw ApiError(400, "An image could not be read.") }
            }
        }
        var nextId = (saved?.messages?.maxOfOrNull { it.id } ?: 0L) + 1
        val question = ChatMessage(nextId++, user = true, text = text, images = names)
        val title = saved?.title ?: text.lineSequence().first().take(60).ifBlank { app.getString(R.string.chat_image_title) }
        val created = id.toLongOrNull() ?: System.currentTimeMillis()
        val before = saved?.messages.orEmpty() + question
        var stats = saved?.stats ?: ChatStats()
        var session = saved?.agentSession
        // the question is saved before the answer is written: the app's chat shows it at once
        save(id, title, created, before, stats, session)
        running = id
        runningJob = currentCoroutineContext()[Job]
        var answer = ChatMessage(nextId, user = false, text = "", streaming = true)
        var gone = false
        fun out(name: String, v: Any?) {
            if (gone) return
            try {
                r.event(name, Json.write(v))
            } catch (e: IOException) {
                gone = true   // the browser went away: the answer goes on and is saved
            }
        }
        val started = System.currentTimeMillis()
        try {
            r.events()
            out("start", linkedMapOf("chat" to id, "title" to title, "question" to json(question)))
            // a chat the agent has no session of yet (one from before every chat was the agent's): its messages so far
            val task = (if (session == null) AgentTurns.earlier(saved?.messages.orEmpty()) else "") + text
            val files = withContext(Dispatchers.IO) { names.map { images.file(it) }.filter { it.isFile } }
            val stop = coroutineScope {
                // the counts of the agent's requests (the chat's), and the answer's speed
                val figures = launch(start = CoroutineStart.UNDISPATCHED) {
                    engine.results.collect { res ->
                        stats = stats.copy(read = stats.read + res.promptTokens - res.cachedTokens,
                            cached = stats.cached + res.cachedTokens, generated = stats.generated + res.outputTokens)
                        if (res.decodeTps > 0) answer = answer.copy(tokensPerSecond = res.decodeTps)
                    }
                }
                try {
                    app.agent.prompt(id, session, task, files) { e ->
                        when (e) {
                            is AgentEvent.Session -> session = e.id
                            AgentEvent.Waiting -> out("status", mapOf("waiting" to true))
                            AgentEvent.Installing -> out("status", mapOf("installing" to true))
                            is AgentEvent.Reading -> out("started", linkedMapOf("prompt" to e.tokens, "cached" to 0))
                            is AgentEvent.Thinking -> out("reasoning", mapOf("text" to e.text))
                            is AgentEvent.Text -> out("content", mapOf("text" to e.text))
                            is AgentEvent.ToolCall -> out("tool", linkedMapOf("id" to e.id, "tool" to e.tool,
                                "args" to AgentTurns.brief(e.input)))
                            is AgentEvent.ToolResult -> out("tool_done", linkedMapOf("id" to e.id, "ok" to e.ok))
                        }
                        answer = answer.withAgent(e)
                    }
                } finally {
                    figures.cancel()
                }
            }
            if (stop == "max_tokens") answer = answer.copy(error = app.getString(R.string.agent_max_tokens))
            answer = answer.finished(withContext(Dispatchers.IO) { AgentTurns.changedSince(app.agent.workspace(id), started) })
            save(id, title, created, before + answer, stats, session)
            out("done", json(answer))
        } catch (e: CancellationException) {
            withContext(NonCancellable) {
                save(id, title, created, before + answer.finished(emptyList()).copy(error = app.getString(R.string.chat_stopped)),
                    stats, session)
            }
            throw e
        } catch (e: Exception) {
            val why = e.message ?: e.javaClass.simpleName
            save(id, title, created, before + answer.finished(emptyList()).copy(error = why), stats, session)
            out("error", mapOf("message" to why))
        } finally {
            running = null
            runningJob = null
        }
    }

    /** The chat saved as the app saves it (every chat is the agent's: its session goes with it). */
    private suspend fun save(id: String, title: String, created: Long, messages: List<ChatMessage>, stats: ChatStats,
                             session: String?) {
        withContext(Dispatchers.IO) { runCatching { store.save(id, title, created, messages, stats, true, session) } }
        _changes.tryEmit(id)
    }

    companion object {
        private const val COOKIE = "geode_session"
        private const val SESSION_MS = 30L * 24 * 3600_000
        private const val MAX_FAILURES = 10

        private val _changes = MutableSharedFlow<String>(extraBufferCapacity = 64)

        /** A chat the web chat changed (its id): the app's chat list and the open chat follow. */
        val changes: SharedFlow<String> = _changes.asSharedFlow()

        /** A password for the web chat: 12 letters and digits that are not mistaken for each other (~59 bits),
         *  in groups of four. */
        fun newPassword(): String {
            val abc = "abcdefghjkmnpqrstuvwxyz23456789"
            val r = SecureRandom()
            return (1..3).joinToString("-") { (1..4).map { abc[r.nextInt(abc.length)] }.joinToString("") }
        }
    }
}
