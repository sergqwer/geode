package xyz.tifu.geode.server

import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import xyz.tifu.geode.models.Protocol
import xyz.tifu.geode.models.Provider
import xyz.tifu.geode.models.Providers
import xyz.tifu.geode.util.Json
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

/**
 * The agent's model requests answered by an outside service (a Provider) instead of the phone's engine. The harness
 * asks Geode's own server on the loopback as it always does (OpenAI's chat completions, streamed); from here the
 * request goes to the service with its key - the key stays in the app, never in the agent's Linux, where a command
 * could read it. An OpenAI-protocol service gets the request nearly as it is (its model, the token limit under the
 * name it takes); an Anthropic one gets it as Messages, and its events come back as chat-completion chunks.
 */
class RemoteApi(private val engine: ApiEngine) {

    /** The harness's request [req] (its body: OpenAI chat completions) answered by [p]'s service with [key]. */
    suspend fun chat(req: HttpRequest, r: HttpResponder, p: Provider, key: String): Unit = withContext(Dispatchers.IO) {
        @Suppress("UNCHECKED_CAST")
        val body = runCatching { Json.parse(req.body.decodeToString()) as Map<String, Any?> }.getOrNull()
            ?: return@withContext r.json(400, error("the request is not JSON"))
        val out = when (p.protocol) {
            Protocol.OPENAI -> openAiBody(body, p)
            Protocol.ANTHROPIC -> anthropicBody(body, p)
        }
        val c = URL(p.chatUrl()).openConnection() as HttpURLConnection
        val t0 = SystemClock.elapsedRealtime()
        val stats = Stats()
        try {
            c.requestMethod = "POST"
            c.doOutput = true
            c.connectTimeout = 20_000
            // a reasoning model may think for minutes before its first token
            c.readTimeout = 300_000
            c.setRequestProperty("Content-Type", "application/json")
            c.setRequestProperty("Accept", "text/event-stream")
            when (p.protocol) {
                Protocol.OPENAI -> {
                    c.setRequestProperty("Authorization", "Bearer $key")
                    // OpenRouter's way of naming the app a request comes from
                    c.setRequestProperty("X-Title", "Geode")
                }
                Protocol.ANTHROPIC -> {
                    c.setRequestProperty("x-api-key", key)
                    c.setRequestProperty("anthropic-version", Providers.ANTHROPIC_VERSION)
                }
            }
            val bytes = Json.write(out).toByteArray()
            c.setFixedLengthStreamingMode(bytes.size)
            c.outputStream.use { it.write(bytes) }
            val code = c.responseCode
            if (code !in 200..299) {
                val text = c.errorStream?.use { it.readBytes().decodeToString() }.orEmpty()
                // the service's own words, said as Geode's server says an error (the harness shows them)
                return@withContext r.json(if (code in 400..599) code else 502,
                    error("${p.name}: ${Providers.errorMessage(code, text)}"))
            }
            r.events()
            BufferedReader(InputStreamReader(c.inputStream, Charsets.UTF_8)).use { input ->
                when (p.protocol) {
                    Protocol.OPENAI -> relayOpenAi(input, r, stats)
                    Protocol.ANTHROPIC -> relayAnthropic(input, r, stats, p)
                }
            }
        } catch (e: IOException) {
            if (!r.started) runCatching { r.json(502, error("${p.name}: ${e.message ?: e.javaClass.simpleName}")) }
            else throw e
        } finally {
            c.disconnect()
            report(stats, t0)
        }
    }

    // ---- OpenAI's protocol: the request nearly as it is

    private fun openAiBody(body: Map<String, Any?>, p: Provider): Map<String, Any?> {
        val b = LinkedHashMap(body)
        b["model"] = p.model
        b["stream"] = true
        b["stream_options"] = mapOf("include_usage" to true)
        if (!p.openAiItself) {
            // the classic name every other service takes (and "store" is OpenAI's own)
            (b.remove("max_completion_tokens") as? Number)?.let { b["max_tokens"] = minOf(it.toInt(), p.maxTokens) }
            b.remove("store")
        }
        // an empty list of tools is refused by most
        if ((b["tools"] as? List<*>)?.isEmpty() == true) b.remove("tools")
        return b
    }

    private fun relayOpenAi(input: BufferedReader, r: HttpResponder, stats: Stats) {
        while (true) {
            val line = input.readLine() ?: break
            if (!line.startsWith("data:")) continue
            val data = line.removePrefix("data:").trim()
            if (data.isEmpty()) continue
            if (data == "[DONE]") {
                r.event(null, data)
                break
            }
            stats.chunk(data)
            r.event(null, data)
        }
    }

    // ---- Anthropic's protocol: Messages out, its events back as chunks

    @Suppress("UNCHECKED_CAST")
    private fun anthropicBody(body: Map<String, Any?>, p: Provider): Map<String, Any?> {
        val system = StringBuilder()
        val messages = ArrayList<MutableMap<String, Any?>>()
        fun add(role: String, blocks: List<Map<String, Any?>>) {
            if (blocks.isEmpty()) return
            val last = messages.lastOrNull()
            // turns alternate: a tool's results and the user's words after them are one user turn
            if (last != null && last["role"] == role) (last["content"] as MutableList<Map<String, Any?>>).addAll(blocks)
            else messages += mutableMapOf("role" to role, "content" to blocks.toMutableList())
        }
        for (m in (body["messages"] as? List<Map<String, Any?>>).orEmpty()) {
            when (m["role"]) {
                "system", "developer" -> {
                    val t = text(m["content"])
                    if (t.isNotBlank()) system.append(if (system.isEmpty()) "" else "\n\n").append(t)
                }
                "user" -> add("user", blocks(m["content"]))
                "assistant" -> {
                    val b = ArrayList<Map<String, Any?>>()
                    text(m["content"]).takeIf { it.isNotBlank() }?.let { b += mapOf("type" to "text", "text" to it) }
                    for (call in (m["tool_calls"] as? List<Map<String, Any?>>).orEmpty()) {
                        val f = call["function"] as? Map<String, Any?> ?: continue
                        val args = (f["arguments"] as? String)?.let { runCatching { Json.parse(it) }.getOrNull() }
                        b += mapOf("type" to "tool_use", "id" to (call["id"] as? String ?: "call"), "name" to f["name"],
                            "input" to (args as? Map<*, *> ?: emptyMap<String, Any?>()))
                    }
                    add("assistant", b)
                }
                "tool" -> add("user", listOf(mapOf("type" to "tool_result",
                    "tool_use_id" to (m["tool_call_id"] as? String ?: "call"),
                    "content" to text(m["content"]).ifEmpty { "(no output)" })))
            }
        }
        // the conversation starts with the user's turn
        if (messages.firstOrNull()?.get("role") != "user")
            messages.add(0, mutableMapOf("role" to "user", "content" to mutableListOf(mapOf("type" to "text", "text" to "(continued)"))))
        val out = LinkedHashMap<String, Any?>()
        out["model"] = p.model
        val max = ((body["max_completion_tokens"] ?: body["max_tokens"]) as? Number)?.toInt() ?: p.maxTokens
        out["max_tokens"] = minOf(max, p.maxTokens).coerceAtLeast(1)
        if (system.isNotEmpty()) out["system"] = system.toString()
        out["messages"] = messages
        val tools = (body["tools"] as? List<Map<String, Any?>>).orEmpty().mapNotNull { t ->
            val f = t["function"] as? Map<String, Any?> ?: return@mapNotNull null
            linkedMapOf("name" to f["name"], "description" to (f["description"] ?: ""),
                "input_schema" to (f["parameters"] ?: mapOf("type" to "object", "properties" to emptyMap<String, Any?>())))
        }
        if (tools.isNotEmpty()) out["tools"] = tools
        (body["temperature"] as? Number)?.let { out["temperature"] = it }
        (body["top_p"] as? Number)?.let { out["top_p"] = it }
        when (val stop = body["stop"]) {
            is String -> out["stop_sequences"] = listOf(stop)
            is List<*> -> if (stop.isNotEmpty()) out["stop_sequences"] = stop
        }
        out["stream"] = true
        return out
    }

    /** OpenAI's content (a string, or parts) as Anthropic's blocks: text and images. */
    @Suppress("UNCHECKED_CAST")
    private fun blocks(content: Any?): List<Map<String, Any?>> = when (content) {
        is String -> if (content.isBlank()) emptyList() else listOf(mapOf("type" to "text", "text" to content))
        is List<*> -> content.mapNotNull { part ->
            val m = part as? Map<String, Any?> ?: return@mapNotNull null
            when (m["type"]) {
                "text" -> (m["text"] as? String)?.takeIf { it.isNotBlank() }?.let { mapOf("type" to "text", "text" to it) }
                "image_url" -> {
                    val url = ((m["image_url"] as? Map<String, Any?>)?.get("url") ?: m["image_url"]) as? String
                        ?: return@mapNotNull null
                    val data = DATA_URL.matchEntire(url)
                    if (data != null) mapOf("type" to "image", "source" to mapOf("type" to "base64",
                        "media_type" to data.groupValues[1], "data" to data.groupValues[2]))
                    else mapOf("type" to "image", "source" to mapOf("type" to "url", "url" to url))
                }
                else -> null
            }
        }
        else -> emptyList()
    }

    /** The text of OpenAI's content (its text parts together). */
    private fun text(content: Any?): String = when (content) {
        is String -> content
        is List<*> -> content.mapNotNull { (it as? Map<*, *>)?.takeIf { p -> p["type"] == "text" }?.get("text") as? String }
            .joinToString("\n")
        else -> ""
    }

    @Suppress("UNCHECKED_CAST")
    private fun relayAnthropic(input: BufferedReader, r: HttpResponder, stats: Stats, p: Provider) {
        // each content block's kind and, for a tool's call, its index among the calls
        val toolOf = HashMap<Int, Int>()
        var tools = 0
        var finish: String? = null
        fun chunk(delta: Map<String, Any?>, finishReason: String? = null) {
            val choice = linkedMapOf<String, Any?>("index" to 0, "delta" to delta)
            if (finishReason != null) choice["finish_reason"] = finishReason
            r.event(null, Json.write(mapOf("object" to "chat.completion.chunk", "choices" to listOf(choice))))
        }
        while (true) {
            val line = input.readLine() ?: break
            if (!line.startsWith("data:")) continue
            val e = runCatching { Json.parse(line.removePrefix("data:").trim()) as Map<String, Any?> }.getOrNull() ?: continue
            when (e["type"]) {
                "message_start" -> {
                    val u = (e["message"] as? Map<String, Any?>)?.get("usage") as? Map<String, Any?>
                    stats.prompt = u.int("input_tokens") + u.int("cache_read_input_tokens") + u.int("cache_creation_input_tokens")
                    stats.cached = u.int("cache_read_input_tokens")
                }
                "content_block_start" -> {
                    val i = (e["index"] as? Number)?.toInt() ?: 0
                    val b = e["content_block"] as? Map<String, Any?> ?: continue
                    when (b["type"]) {
                        "tool_use" -> {
                            val n = tools++
                            toolOf[i] = n
                            chunk(mapOf("tool_calls" to listOf(mapOf("index" to n, "id" to b["id"], "type" to "function",
                                "function" to mapOf("name" to b["name"], "arguments" to "")))))
                        }
                        "text" -> (b["text"] as? String)?.takeIf { it.isNotEmpty() }?.let {
                            stats.first()
                            chunk(mapOf("content" to it))
                        }
                    }
                }
                "content_block_delta" -> {
                    val i = (e["index"] as? Number)?.toInt() ?: 0
                    val d = e["delta"] as? Map<String, Any?> ?: continue
                    when (d["type"]) {
                        "text_delta" -> {
                            stats.first()
                            chunk(mapOf("content" to d["text"]))
                        }
                        "thinking_delta" -> {
                            stats.first()
                            chunk(mapOf("reasoning_content" to d["thinking"]))
                        }
                        "input_json_delta" -> toolOf[i]?.let { n ->
                            chunk(mapOf("tool_calls" to listOf(mapOf("index" to n,
                                "function" to mapOf("arguments" to (d["partial_json"] ?: ""))))))
                        }
                    }
                }
                "message_delta" -> {
                    finish = when ((e["delta"] as? Map<String, Any?>)?.get("stop_reason")) {
                        "tool_use" -> "tool_calls"
                        "max_tokens" -> "length"
                        null -> finish
                        else -> "stop"
                    }
                    (e["usage"] as? Map<String, Any?>)?.let { stats.output = it.int("output_tokens") }
                }
                "message_stop" -> break
                "error" -> {
                    // said in the answer: the user sees why it ended
                    val msg = ((e["error"] as? Map<String, Any?>)?.get("message") as? String) ?: "error"
                    chunk(mapOf("content" to "\n\n[${p.name}: $msg]"))
                    finish = "stop"
                    break
                }
            }
        }
        chunk(emptyMap(), finish ?: "stop")
        r.event(null, Json.write(mapOf("object" to "chat.completion.chunk", "choices" to emptyList<Any>(),
            "usage" to mapOf("prompt_tokens" to stats.prompt, "completion_tokens" to stats.output,
                "total_tokens" to stats.prompt + stats.output))))
        r.event(null, "[DONE]")
    }

    // ---- what the chat shows of a request: its counts and speed

    private class Stats {
        var prompt = 0
        var cached = 0
        var output = 0
        /** When the first word came (elapsedRealtime; 0: none yet). */
        var firstAt = 0L

        fun first() {
            if (firstAt == 0L) firstAt = SystemClock.elapsedRealtime()
        }

        /** An OpenAI chunk: its first content, its usage. */
        @Suppress("UNCHECKED_CAST")
        fun chunk(data: String) {
            // (a service may write its JSON with spaces: "content": "...")
            if (firstAt == 0L && WORDS.containsMatchIn(data)) first()
            if (!data.contains("\"usage\"")) return
            val u = runCatching { (Json.parse(data) as Map<String, Any?>)["usage"] as? Map<String, Any?> }.getOrNull() ?: return
            prompt = u.int("prompt_tokens")
            output = u.int("completion_tokens")
            // OpenAI's cached part, or DeepSeek's
            cached = ((u["prompt_tokens_details"] as? Map<String, Any?>).int("cached_tokens"))
                .takeIf { it > 0 } ?: u.int("prompt_cache_hit_tokens")
        }
    }

    private fun report(s: Stats, t0: Long) {
        if (s.prompt == 0 && s.output == 0) return
        val end = SystemClock.elapsedRealtime()
        val writing = if (s.firstAt > 0) (end - s.firstAt).coerceAtLeast(1) else 0L
        val reading = if (s.firstAt > 0) (s.firstAt - t0).coerceAtLeast(1) else 0L
        engine.report(GenResult(
            content = "", reasoning = "", toolCalls = emptyList(), finish = Finish.STOP, stopSequence = null,
            promptTokens = s.prompt, cachedTokens = s.cached, outputTokens = s.output, reasoningTokens = 0,
            decodeTps = if (writing > 0 && s.output > 1) s.output * 1000f / writing else 0f,
            prefillTps = if (reading > 0) (s.prompt - s.cached).coerceAtLeast(0) * 1000f / reading else 0f,
            contextUsed = s.prompt + s.output,
        ))
    }

    private fun error(message: String) = mapOf("error" to mapOf("message" to message, "type" to "upstream_error"))

    private companion object {
        val DATA_URL = Regex("^data:([^;,]+);base64,(.*)$", RegexOption.DOT_MATCHES_ALL)
        /** A chunk with words in it: content or reasoning, not empty. */
        val WORDS = Regex("\"(content|reasoning_content|reasoning|reasoning_text)\"\\s*:\\s*\"[^\"]")

        fun Map<String, Any?>?.int(k: String): Int = (this?.get(k) as? Number)?.toInt() ?: 0
    }
}
