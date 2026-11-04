package xyz.tifu.geode.agent.browser

import android.util.Base64
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.server.HttpRequest
import xyz.tifu.geode.server.HttpResponder
import xyz.tifu.geode.util.Json
import java.io.File

/**
 * The agent's browser ([AgentBrowser]) as an MCP server the harness speaks to over HTTP (geode-agent's mcp.rs: one
 * JSON-RPC message a POST, its answer as JSON, a notification's 202). One per harness: [chatId] the chat whose page it
 * is, [vision] whether the model takes images (no screenshot tool for one that does not).
 */
class BrowserMcp(private val app: GeodeApp, private val chatId: String, private val vision: () -> Boolean) {
    private val browser get() = app.browser

    @Suppress("UNCHECKED_CAST")
    suspend fun handle(req: HttpRequest, r: HttpResponder) {
        r.cors = false
        if (req.method != "POST") return r.json(405, mapOf("error" to "POST a JSON-RPC message"))
        val msg = runCatching { Json.parse(req.body.decodeToString()) as Map<String, Any?> }.getOrNull()
            ?: return r.json(400, rpcError(null, -32700, "not a JSON-RPC message"))
        val id = msg["id"]
        val method = msg["method"] as? String
        // a notification (initialized, cancelled): nothing to answer
        if (id == null || method == null) return r.send(202, "application/json", ByteArray(0))
        val params = msg["params"] as? Map<String, Any?> ?: emptyMap()
        // (adb logcat -s GeodeBrowser: whether the harness connected, and what it asked)
        Log.i(TAG, "$chatId: $method" + ((params["name"] as? String)?.let { " $it" } ?: ""))
        val result: Any? = when (method) {
            "initialize" -> linkedMapOf(
                "protocolVersion" to (params["protocolVersion"] as? String ?: PROTOCOL),
                "capabilities" to mapOf("tools" to emptyMap<String, Any?>()),
                "serverInfo" to mapOf("name" to "geode-browser", "version" to "1"),
                "instructions" to INSTRUCTIONS,
            )
            "ping" -> emptyMap<String, Any?>()
            "tools/list" -> mapOf("tools" to tools())
            "tools/call" -> call(params["name"] as? String ?: "", params["arguments"] as? Map<String, Any?> ?: emptyMap())
            else -> return r.json(200, rpcError(id, -32601, "no method $method"))
        }
        r.json(200, linkedMapOf("jsonrpc" to "2.0", "id" to id, "result" to result))
    }

    private fun rpcError(id: Any?, code: Int, message: String) =
        linkedMapOf("jsonrpc" to "2.0", "id" to id, "error" to mapOf("code" to code, "message" to message))

    // ---- the tools ----

    private fun obj(props: Map<String, Any?>, required: List<String> = emptyList()) = linkedMapOf(
        "type" to "object", "properties" to props, "required" to required)

    private fun str(d: String) = mapOf("type" to "string", "description" to d)
    private fun num(d: String) = mapOf("type" to "number", "description" to d)
    private fun bool(d: String) = mapOf("type" to "boolean", "description" to d)

    private fun tool(name: String, description: String, schema: Map<String, Any?>) =
        linkedMapOf("name" to name, "description" to description, "inputSchema" to schema)

    private fun tools(): List<Map<String, Any?>> {
        val v = vision()
        val at = if (v) " or x,y on the last screenshot" else ""
        return listOfNotNull(
            tool("open", "Open a page in the phone's browser (the user watches it in the chat): a web address, or a " +
                "file of the workspace (/workspace/game/index.html or game/index.html: served over http, so modules and " +
                "fetch of relative files work). Waits for it to load; returns its title and the console's lines.",
                obj(mapOf("url" to str("https://..., or a workspace path"),
                    "wait" to num("Seconds to wait for the load (default 15)")), listOf("url"))),
            if (v) tool("screenshot", "What the page shows now, as an image (WebGL and canvases included).",
                obj(emptyMap())) else null,
            tool("eval", "Run JavaScript in the page and return its value as JSON (a promise is awaited; top-level " +
                "await works). The console's lines it printed come with it.",
                obj(mapOf("expression" to str("JavaScript; its last expression is the value"),
                    "timeout" to num("Seconds (default 30, at most 100)")), listOf("expression"))),
            tool("console", "The page's console: logs, warnings, errors, uncaught exceptions, failed loads - the " +
                "lines since it was last read.",
                obj(mapOf("all" to bool("Every line kept (the last 300), not only the new ones")))),
            tool("text", "The page's text as the user reads it (or an element's), or its HTML.",
                obj(mapOf("selector" to str("A CSS selector (default: the whole page)"),
                    "html" to bool("Its HTML instead of its text")))),
            tool("act", "Do what a user does on the page. action: click (a CSS selector$at), type (text into the " +
                "selector's element, or the focused one), key (Enter, Tab, Escape, Backspace, ArrowUp, a letter, " +
                "Control+a; hold for ms), scroll (dy pixels, down is positive), swipe (x,y to x2,y2$at), back, " +
                "forward, reload. Returns the console's new lines.",
                obj(linkedMapOf(
                    "action" to mapOf("type" to "string",
                        "enum" to listOf("click", "type", "key", "scroll", "swipe", "back", "forward", "reload")),
                    "selector" to str("CSS selector of the element"),
                    "x" to num(if (v) "On the last screenshot" else "Page CSS pixels"),
                    "y" to num(if (v) "On the last screenshot" else "Page CSS pixels"),
                    "x2" to num("swipe's end"), "y2" to num("swipe's end"),
                    "text" to str("type's text"),
                    "key" to str("key's name"),
                    "ms" to num("key: how long it is held; swipe: how long it takes"),
                    "dx" to num("scroll"), "dy" to num("scroll"),
                ), listOf("action"))),
            tool("cdp", "Any Chrome DevTools Protocol command on the page (Network.*, DOM.*, Emulation.*, " +
                "Performance.*...), its result as JSON.",
                obj(mapOf("method" to str("e.g. Network.getCookies"),
                    "params" to mapOf("type" to "object", "description" to "Its parameters")), listOf("method"))),
        )
    }

    private class ToolError(message: String) : Exception(message)

    private suspend fun call(name: String, a: Map<String, Any?>): Map<String, Any?> = try {
        browser.lock.withLock {
            // turned off in the settings while this harness runs (the next one has no browser): not opened again
            if (!app.agentConfig.browser.value) throw ToolError("The user turned the browser off in the settings")
            val c = browser.ready(chatId)
            val content = when (name) {
                "open" -> open(c, a)
                "screenshot" -> if (vision()) screenshot(0) else throw ToolError("this model does not take images")
                "eval" -> eval(c, a)
                "console" -> listOf(text(lines(browser.consoleSince(a["all"] == true)).ifEmpty { "(nothing new)" }))
                "text" -> pageText(c, a)
                "act" -> act(c, a)
                "cdp" -> raw(c, a)
                else -> throw ToolError("no tool $name")
            }
            mapOf("content" to content, "isError" to false)
        }
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        if (e !is ToolError && e !is CdpException) Log.w(TAG, "the browser's $name failed", e)
        else Log.w(TAG, "the browser's $name: ${e.message}")
        mapOf("content" to listOf(text(e.message ?: e.toString())), "isError" to true)
    }

    private fun text(s: String) = mapOf("type" to "text", "text" to s)

    private fun lines(entries: List<AgentBrowser.Entry>): String =
        entries.joinToString("\n") { "[${it.level}] ${it.text}" }.let { if (it.length > OUT_MAX) "…" + it.takeLast(OUT_MAX) else it }

    /** The page as it is now: its title and address, and the console's new lines. */
    private fun state(): String {
        val ui = browser.ui.value
        val console = lines(browser.consoleSince())
        return "Title: ${ui.title.ifEmpty { "(none)" }}\nAddress: ${ui.url.ifEmpty { "(none)" }}" +
            if (console.isEmpty()) "\nConsole: (nothing new)" else "\nConsole:\n$console"
    }

    private suspend fun open(c: Cdp, a: Map<String, Any?>): List<Map<String, Any?>> {
        val given = (a["url"] as? String)?.trim().orEmpty()
        if (given.isEmpty()) throw ToolError("url is empty")
        val url = if (Regex("^(https?|about|data|blob|javascript):", RegexOption.IGNORE_CASE).containsMatchIn(given)) given
        else {
            val p = given.removePrefix("file://").substringBefore('#').substringBefore('?')
            val rel = when {
                p == "/workspace" -> ""
                p.startsWith("/workspace/") -> p.removePrefix("/workspace/")
                else -> p.removePrefix("./")
            }.trimStart('/')
            val dir = app.agent.workspace(chatId).canonicalFile
            val f = File(dir, rel).canonicalFile
            if (!f.toPath().startsWith(dir.toPath())) throw ToolError("$given is outside /workspace")
            if (!f.exists()) throw ToolError("No such file: /workspace/$rel")
            if (f.isDirectory && !File(f, "index.html").isFile) throw ToolError("/workspace/$rel is a folder with no index.html")
            val tail = given.indexOfFirst { it == '?' || it == '#' }.let { if (it < 0) "" else given.substring(it) }
            // a folder's address ends with a slash (its page's relative files are the folder's)
            val path = if (f.isDirectory && rel.isNotEmpty() && !rel.endsWith("/")) "$rel/" else rel
            (browser.fileUrl(path) ?: throw ToolError("the browser is closed")) + tail
        }
        val waiter = browser.expectLoad()
        val res = c.call("Page.navigate", mapOf("url" to url))
        (res["errorText"] as? String)?.takeIf { it.isNotEmpty() }?.let { throw ToolError("$given did not load: $it") }
        val wait = ((a["wait"] as? Number)?.toDouble() ?: 15.0).coerceIn(0.0, 90.0)
        val loaded = browser.awaitLoad(waiter, (wait * 1000).toLong())
        // a moment for its scripts' first lines
        delay(SETTLE_MS)
        return listOf(text((if (loaded) "" else "(it had not finished loading in ${wait.toInt()} s)\n") + state()))
    }

    private suspend fun screenshot(after: Long): List<Map<String, Any?>> {
        val s = browser.screenshot(after = after)
        return listOf(text("Screenshot ${s.width}x${s.height} of ${browser.ui.value.url.ifEmpty { "the page" }}"),
            mapOf("type" to "image", "mimeType" to "image/jpeg", "data" to Base64.encodeToString(s.jpeg, Base64.NO_WRAP)))
    }

    /** [expression]'s value in the page (its exception: a ToolError). */
    @Suppress("UNCHECKED_CAST")
    private suspend fun evaluate(c: Cdp, expression: String, timeoutMs: Long = 30_000): Any? {
        val r = c.call("Runtime.evaluate", linkedMapOf("expression" to expression, "returnByValue" to true,
            "awaitPromise" to true, "userGesture" to true, "replMode" to true, "timeout" to timeoutMs),
            timeoutMs + 2000)
        (r["exceptionDetails"] as? Map<String, Any?>)?.let { d ->
            val desc = (d["exception"] as? Map<String, Any?>)?.get("description") as? String
            throw ToolError(desc ?: (d["text"] as? String) ?: "the script threw")
        }
        val v = r["result"] as? Map<String, Any?> ?: return null
        return when {
            v.containsKey("value") -> v["value"]
            v["unserializableValue"] != null -> Raw(v["unserializableValue"].toString())
            v["type"] == "undefined" -> Raw("undefined")
            else -> Raw((v["description"] ?: v["type"]).toString())
        }
    }

    /** A value with no JSON of its own (undefined, NaN, a function's description), shown as it is. */
    private class Raw(val s: String)

    private fun show(v: Any?): String = when (v) {
        is Raw -> v.s
        is String -> Json.write(v)
        else -> Json.write(v)
    }.let { if (it.length > OUT_MAX) it.take(OUT_MAX) + "… (${it.length} characters)" else it }

    private suspend fun eval(c: Cdp, a: Map<String, Any?>): List<Map<String, Any?>> {
        val js = a["expression"] as? String ?: throw ToolError("expression is missing")
        val t = ((a["timeout"] as? Number)?.toDouble() ?: 30.0).coerceIn(1.0, 100.0)
        val v = try {
            evaluate(c, js, (t * 1000).toLong())
        } catch (e: ToolError) {
            // (the harness words it as an error itself)
            throw ToolError("${e.message}" + lines(browser.consoleSince()).let { if (it.isEmpty()) "" else "\nConsole:\n$it" })
        }
        val console = lines(browser.consoleSince())
        return listOf(text(show(v) + if (console.isEmpty()) "" else "\nConsole:\n$console"))
    }

    private suspend fun pageText(c: Cdp, a: Map<String, Any?>): List<Map<String, Any?>> {
        val sel = (a["selector"] as? String)?.takeIf { it.isNotBlank() }
        val html = a["html"] == true
        val js = "(() => { const e = ${if (sel == null) "document.documentElement" else "document.querySelector(${Json.write(sel)})"};" +
            " if (!e) return null; return ${if (html) "e.outerHTML" else "(e.innerText ?? e.textContent)"}; })()"
        val v = evaluate(c, js) ?: throw ToolError("No element matches $sel")
        val s = v as? String ?: show(v)
        return listOf(text(if (s.length > OUT_MAX) s.take(OUT_MAX) + "\n… (${s.length} characters; a selector narrows it)" else s))
    }

    /**
     * Display pixels of [sel]'s middle (scrolled into view first): the page's CSS pixels through its zoom
     * (visualViewport) and density, the way the WebView draws them.
     */
    @Suppress("UNCHECKED_CAST")
    private suspend fun elementAt(c: Cdp, sel: String): Pair<Float, Float> {
        val v = evaluate(c, "(() => { const e = document.querySelector(${Json.write(sel)}); if (!e) return null;" +
            " e.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});" +
            " const r = e.getBoundingClientRect(); const vv = window.visualViewport;" +
            " const s = (vv ? vv.scale : 1) * window.devicePixelRatio;" +
            " return {x: (r.left + r.width / 2 - (vv ? vv.offsetLeft : 0)) * s, y: (r.top + r.height / 2 - (vv ? vv.offsetTop : 0)) * s," +
            " w: r.width, h: r.height}; })()") as? Map<String, Any?> ?: throw ToolError("No element matches $sel")
        if ((v["w"] as? Number)?.toDouble() == 0.0 && (v["h"] as? Number)?.toDouble() == 0.0)
            throw ToolError("$sel is not shown (it has no size)")
        return ((v["x"] as Number).toFloat()) to ((v["y"] as Number).toFloat())
    }

    /** Display pixels of [x], [y]: the last screenshot's pixels (a model that sees), or the page's CSS pixels. */
    @Suppress("UNCHECKED_CAST")
    private suspend fun point(c: Cdp, x: Number, y: Number): Pair<Float, Float> {
        if (vision()) return (x.toFloat() * browser.shotScale) to (y.toFloat() * browser.shotScale)
        val s = evaluate(c, "(() => { const vv = window.visualViewport; return {s: (vv ? vv.scale : 1) * window.devicePixelRatio," +
            " l: vv ? vv.offsetLeft : 0, t: vv ? vv.offsetTop : 0}; })()") as? Map<String, Any?> ?: emptyMap()
        val k = (s["s"] as? Number)?.toFloat() ?: browser.density
        return ((x.toFloat() - ((s["l"] as? Number)?.toFloat() ?: 0f)) * k) to ((y.toFloat() - ((s["t"] as? Number)?.toFloat() ?: 0f)) * k)
    }

    private fun inside(p: Pair<Float, Float>) {
        if (p.first < 0 || p.second < 0 || p.first >= browser.width || p.second >= browser.height)
            throw ToolError("that point is off the screen")
    }

    private suspend fun act(c: Cdp, a: Map<String, Any?>): List<Map<String, Any?>> {
        val sel = (a["selector"] as? String)?.takeIf { it.isNotBlank() }
        val x = a["x"] as? Number
        val y = a["y"] as? Number
        val start = browser.now()
        val did: String = when (val action = a["action"] as? String) {
            "click", "tap" -> {
                val p = when {
                    sel != null -> elementAt(c, sel).also { delay(SCROLL_SETTLE_MS) }
                    x != null && y != null -> point(c, x, y)
                    else -> throw ToolError("click needs a selector or x and y")
                }
                inside(p)
                browser.tap(p.first, p.second)
                "Clicked ${sel ?: "at $x,$y"}"
            }
            "type" -> {
                val t = a["text"] as? String ?: throw ToolError("type needs text")
                if (sel != null) {
                    val ok = evaluate(c, "(() => { const e = document.querySelector(${Json.write(sel)}); if (!e) return false;" +
                        " e.focus(); if (typeof e.setSelectionRange === 'function' && typeof e.value === 'string')" +
                        " { try { e.setSelectionRange(e.value.length, e.value.length); } catch (_) {} } return true; })()")
                    if (ok != true) throw ToolError("No element matches $sel")
                }
                c.call("Input.insertText", mapOf("text" to t))
                "Typed ${t.length} characters" + (sel?.let { " into $it" } ?: "")
            }
            "key" -> {
                val k = a["key"] as? String ?: a["text"] as? String ?: throw ToolError("key needs a key")
                if (!Keys.known(k)) throw ToolError("unknown key \"$k\"")
                Keys.press(c, k, ((a["ms"] as? Number)?.toLong() ?: 0L).coerceIn(0, 10_000))
                "Pressed $k"
            }
            "scroll" -> {
                val dy = (a["dy"] as? Number)?.toDouble() ?: 0.0
                val dx = (a["dx"] as? Number)?.toDouble() ?: 0.0
                if (dx == 0.0 && dy == 0.0) throw ToolError("scroll needs dy (or dx)")
                val target = if (sel != null) "document.querySelector(${Json.write(sel)})" else "null"
                val moved = evaluate(c, "(() => { const e = $target; if (!e && ${sel != null}) return null;" +
                    " const t = e || document.scrollingElement || document.documentElement; const x = t.scrollLeft, y = t.scrollTop;" +
                    " (e || window).scrollBy({left: $dx, top: $dy, behavior: 'instant'});" +
                    " return [Math.round(t.scrollLeft - x), Math.round(t.scrollTop - y)]; })()") as? List<*>
                    ?: throw ToolError("No element matches $sel")
                "Scrolled by ${moved.getOrNull(1)} px down, ${moved.getOrNull(0)} px right" +
                    if (moved.all { (it as? Number)?.toInt() == 0 }) " (it did not move: nothing more to scroll)" else ""
            }
            "swipe" -> {
                val x2 = a["x2"] as? Number
                val y2 = a["y2"] as? Number
                if (x == null || y == null || x2 == null || y2 == null) throw ToolError("swipe needs x, y, x2, y2")
                val p1 = point(c, x, y)
                val p2 = point(c, x2, y2)
                inside(p1)
                browser.swipe(p1.first, p1.second, p2.first.coerceIn(0f, browser.width - 1f),
                    p2.second.coerceIn(0f, browser.height - 1f), ((a["ms"] as? Number)?.toLong() ?: 300L).coerceIn(50, 5000))
                "Swiped"
            }
            "back", "forward" -> {
                // the page's history, the browser's own first page (its marker, a data: page) left out of it
                val h = c.call("Page.getNavigationHistory")
                @Suppress("UNCHECKED_CAST")
                val entries = h["entries"] as? List<Map<String, Any?>> ?: emptyList()
                val at = (h["currentIndex"] as? Number)?.toInt() ?: (entries.size - 1)
                val to = entries.getOrNull(at + if (action == "back") -1 else 1)
                    ?.takeIf { (it["title"] as? String)?.startsWith("geode-browser-") != true &&
                        (it["url"] as? String)?.let { u -> u.startsWith("data:") || u == "about:blank" } != true }
                    ?: throw ToolError("There is no page to go $action to")
                val waiter = browser.expectLoad()
                c.call("Page.navigateToHistoryEntry", mapOf("entryId" to to["id"]))
                browser.awaitLoad(waiter, NAV_MS)
                "Went $action"
            }
            "reload" -> {
                val waiter = browser.expectLoad()
                c.call("Page.reload", mapOf("ignoreCache" to true))
                if (!browser.awaitLoad(waiter, 15_000)) "Reloaded (not loaded in 15 s)" else "Reloaded"
            }
            else -> throw ToolError("unknown action \"$action\"")
        }
        delay(SETTLE_MS)
        return listOf(text("$did.\n" + state()))
    }

    private suspend fun raw(c: Cdp, a: Map<String, Any?>): List<Map<String, Any?>> {
        val method = (a["method"] as? String)?.trim().orEmpty()
        if (method.isEmpty()) throw ToolError("method is missing")
        // what would take the page from the app (its connection, its console, its target)
        if (method.startsWith("Target.") || method.startsWith("Browser.") || method in BLOCKED)
            throw ToolError("$method is not allowed: it would cut the app's own connection to the page")
        @Suppress("UNCHECKED_CAST")
        val params = a["params"] as? Map<String, Any?> ?: emptyMap()
        val r = withTimeout(RAW_MS + 5000) { c.call(method, params, RAW_MS) }
        return listOf(text(show(r)))
    }

    companion object {
        private const val TAG = "GeodeBrowser"
        private const val PROTOCOL = "2025-06-18"
        /** The longest text a tool returns (the model's context is a phone's). */
        private const val OUT_MAX = 20_000
        private const val SETTLE_MS = 400L
        private const val SCROLL_SETTLE_MS = 150L
        private const val NAV_MS = 5000L
        private const val RAW_MS = 60_000L
        private val BLOCKED = setOf("Runtime.disable", "Page.disable", "Log.disable", "Page.close", "Page.crash",
            "Inspector.disable")

        /** Into the system prompt (mcp.rs: a server's instructions). */
        private const val INSTRUCTIONS =
            "You have a real browser: the phone's own Chromium, which the user watches in the chat. When the user asks " +
            "to open, run, show, test or look at a web page or an HTML file, open it yourself with this server's open " +
            "tool (a workspace path such as /workspace/index.html, or a web address) - never tell the user to open it " +
            "themselves, and never say you have no browser. To check a page you made, open it, read the console for " +
            "errors and look at it (a screenshot, or text). After you change its files, open it again (or act reload). " +
            "Prefer CSS selectors to coordinates."
    }
}
