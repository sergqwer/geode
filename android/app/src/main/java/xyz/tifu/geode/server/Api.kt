package xyz.tifu.geode.server

import android.util.Base64
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.tokenizer.Part
import xyz.tifu.geode.tokenizer.TemplateMessage
import xyz.tifu.geode.tokenizer.ToolCall
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.bool
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.num
import xyz.tifu.geode.util.obj
import xyz.tifu.geode.util.str
import java.security.MessageDigest
import java.time.Instant

/**
 * The local server's API: OpenAI's chat completions (/v1/chat/completions, /v1/models) and Anthropic's messages
 * (/v1/messages, /v1/messages/count_tokens, /v1/models with an anthropic-version header) side by side, streamed or
 * whole, with tools, images (base64) and reasoning (OpenAI: reasoning_content; Anthropic: thinking blocks). [key]: the
 * API key requests need (Authorization: Bearer, or x-api-key), null when they need none.
 */
@Suppress("UNCHECKED_CAST")
class Api(private val app: GeodeApp, private val engine: ApiEngine, private val key: () -> String?) {

    suspend fun handle(req: HttpRequest, r: HttpResponder) {
        if (req.method == "OPTIONS") return r.send(204, "text/plain", ByteArray(0))
        // the paths with and without /v1 (clients differ in what their base URL holds)
        val path = req.path.trimEnd('/').removePrefix("/v1").ifEmpty { "/" }
        val anthropic = path.startsWith("/messages") || req.header("anthropic-version") != null
        if (req.method == "GET" && path == "/health") return r.json(200, mapOf("status" to "ok"))
        val need = key()
        if (need != null && !authorized(req, need)) {
            return error(r, anthropic, ApiError(401, "A valid API key is needed (Authorization: Bearer <key> or " +
                "x-api-key: <key>); Geode shows it on its Local server screen."))
        }
        LocalServer.status.update { it.copy(requests = it.requests + 1) }
        try {
            when {
                path == "/models" && req.method == "GET" -> models(r, anthropic, null)
                path.startsWith("/models/") && req.method == "GET" -> models(r, anthropic, path.removePrefix("/models/"))
                path == "/chat/completions" && req.method == "POST" -> chatCompletions(req, r)
                path == "/messages" && req.method == "POST" -> messages(req, r)
                path == "/messages/count_tokens" && req.method == "POST" -> countTokens(req, r)
                else -> throw ApiError(404, "No such endpoint: ${req.method} ${req.path}")
            }
        } catch (e: ApiError) {
            if (!r.started) error(r, anthropic, e)
        }
    }

    private fun authorized(req: HttpRequest, need: String): Boolean {
        val given = req.header("x-api-key") ?: req.header("authorization")?.trim()?.removePrefix("Bearer")?.trim()
            ?: return false
        return MessageDigest.isEqual(given.toByteArray(), need.toByteArray())
    }

    private fun error(r: HttpResponder, anthropic: Boolean, e: ApiError) {
        if (anthropic) r.json(e.status, errorBody(true, e)) else r.json(e.status, errorBody(false, e))
    }

    private fun errorBody(anthropic: Boolean, e: ApiError): Map<String, Any?> = if (anthropic) {
        val type = when (e.status) {
            400 -> "invalid_request_error"
            401 -> "authentication_error"
            404 -> "not_found_error"
            503 -> "overloaded_error"
            else -> "api_error"
        }
        linkedMapOf("type" to "error", "error" to linkedMapOf("type" to type, "message" to e.message))
    } else {
        val type = when (e.status) {
            400, 404 -> "invalid_request_error"
            401 -> "authentication_error"
            else -> "server_error"
        }
        linkedMapOf("error" to linkedMapOf("message" to e.message, "type" to type, "param" to null, "code" to null))
    }

    private fun body(req: HttpRequest): Map<String, Any?> =
        runCatching { Json.parse(String(req.body, Charsets.UTF_8)) as? Map<String, Any?> }.getOrNull()
            ?: throw ApiError(400, "The body is not a JSON object.")

    // ---- models ----

    private suspend fun models(r: HttpResponder, anthropic: Boolean, one: String?) {
        val m = runCatching { engine.model() }.getOrNull()
        val list = listOfNotNull(m).filter { one == null || it.id == one }
        if (one != null && list.isEmpty()) throw ApiError(404, "No model $one: the server has ${m?.id ?: "none"}.")
        val created = list.firstOrNull()?.dir?.lastModified() ?: 0L
        if (anthropic) {
            val items = list.map {
                linkedMapOf("type" to "model", "id" to it.id, "display_name" to it.title,
                    "created_at" to Instant.ofEpochMilli(created).toString(), "input_modalities" to inputs(it))
            }
            if (one != null) return r.json(200, items.first())
            r.json(200, linkedMapOf("data" to items, "has_more" to false, "first_id" to list.firstOrNull()?.id,
                "last_id" to list.lastOrNull()?.id))
        } else {
            val items = list.map {
                linkedMapOf("id" to it.id, "object" to "model", "created" to created / 1000, "owned_by" to "geode",
                    "input_modalities" to inputs(it))
            }
            if (one != null) return r.json(200, items.first())
            r.json(200, linkedMapOf("object" to "list", "data" to items))
        }
    }

    /**
     * What a model takes besides text, the field OpenRouter's model list names so (ours, beside the standard ones,
     * which clients that do not know it skip): image (image_url / Anthropic image blocks), audio (input_audio), video
     * (Geode's input_video: frames at their seconds and their sound).
     */
    private fun inputs(m: InstalledModel): List<String> =
        listOf("text") + listOfNotNull("image".takeIf { m.vision }, "audio".takeIf { m.audio }, "video".takeIf { m.video })

    // ---- images ----

    /** A data: URL's image (OpenAI's image_url); other URLs are not fetched. */
    private suspend fun dataUrl(model: InstalledModel, url: String): Part.Image {
        if (!url.startsWith("data:")) throw ApiError(400, "Images must come as data: URLs (base64); the server fetches nothing.")
        val comma = url.indexOf(',')
        if (comma < 0 || !url.substring(0, comma).endsWith(";base64")) throw ApiError(400, "An image's data: URL is not base64.")
        return engine.image(model, decode(url.substring(comma + 1)))
    }

    private fun decode(b64: String): ByteArray =
        runCatching { Base64.decode(b64, Base64.DEFAULT) }.getOrElse { throw ApiError(400, "An image's base64 is broken.") }

    private fun decodeAudio(b64: String): ByteArray =
        runCatching { Base64.decode(b64, Base64.DEFAULT) }.getOrElse { throw ApiError(400, "A clip's base64 is broken.") }

    // ---- OpenAI ----

    private suspend fun openaiParts(model: InstalledModel, content: Any?): List<Part> = when (content) {
        null -> emptyList()
        is String -> listOf(Part.Text(content))
        is List<*> -> content.mapNotNull { p ->
            p as? Map<String, Any?> ?: throw ApiError(400, "A content part is not an object.")
            when (p.str("type")) {
                "text", "input_text" -> Part.Text(p.str("text").orEmpty())
                "image_url" -> dataUrl(model, (p["image_url"] as? Map<String, Any?>)?.str("url") ?: p.str("image_url")
                    ?: throw ApiError(400, "An image_url part has no url."))
                // OpenAI's input_audio: base64 bytes and their format (wav, mp3; any Android decodes)
                "input_audio" -> {
                    val a = p.obj("input_audio") ?: throw ApiError(400, "An input_audio part has no input_audio.")
                    val data = a.str("data") ?: throw ApiError(400, "An input_audio part has no data.")
                    engine.audio(model, decodeAudio(data), a.str("format").orEmpty())
                }
                // Geode's input_video: a video's frames (base64 images, each at its second) and its sound (as
                // input_audio's), as the app's agent sends them
                "input_video" -> {
                    val v = p.obj("input_video") ?: throw ApiError(400, "An input_video part has no input_video.")
                    val frames = (v.list("frames") ?: throw ApiError(400, "An input_video part has no frames.")).map { x ->
                        val fr = x as? Map<String, Any?> ?: throw ApiError(400, "A video's frame is not an object.")
                        decode(fr.str("data") ?: throw ApiError(400, "A video's frame has no data.")) to
                            ((fr["second"] as? Number)?.toInt() ?: 0)
                    }
                    val sound = v.obj("audio")?.let { a ->
                        decodeAudio(a.str("data") ?: throw ApiError(400, "A video's sound has no data.")) to a.str("format").orEmpty()
                    }
                    engine.video(model, frames, sound)
                }
                else -> throw ApiError(400, "Content parts of type ${p.str("type")} are not supported.")
            }
        }
        else -> throw ApiError(400, "A message's content is neither a string nor a list of parts.")
    }

    private suspend fun openaiMessages(model: InstalledModel, list: List<Any?>): List<TemplateMessage> {
        val system = ArrayList<String>()
        val out = ArrayList<TemplateMessage>()
        for (x in list) {
            val m = x as? Map<String, Any?> ?: throw ApiError(400, "A message is not an object.")
            when (val role = m.str("role")) {
                // the template takes one system message, first: all of them go into it
                "system", "developer" -> system += openaiParts(model, m["content"]).filterIsInstance<Part.Text>()
                    .joinToString("") { it.text }
                "user" -> out += TemplateMessage("user", openaiParts(model, m["content"]))
                "assistant" -> out += TemplateMessage("assistant",
                    openaiParts(model, m["content"]).filterIsInstance<Part.Text>(),
                    reasoning = m.str("reasoning_content") ?: m.str("reasoning"),
                    toolCalls = m.list("tool_calls").orEmpty().map { tc ->
                        val f = (tc as? Map<String, Any?>)?.obj("function") ?: throw ApiError(400, "A tool call has no function.")
                        val args = when (val a = f["arguments"]) {
                            is String -> runCatching { Json.parse(a) as? Map<String, Any?> }.getOrNull() ?: emptyMap()
                            is Map<*, *> -> a as Map<String, Any?>
                            else -> emptyMap()
                        }
                        ToolCall((tc as Map<String, Any?>).str("id").orEmpty(), f.str("name").orEmpty(), args)
                    })
                "tool", "function" -> out += TemplateMessage("tool", openaiParts(model, m["content"]),
                    toolCallId = m.str("tool_call_id"))
                else -> throw ApiError(400, "Messages of role $role are not supported.")
            }
        }
        if (system.isNotEmpty()) out.add(0, TemplateMessage("system", listOf(Part.Text(system.joinToString("\n\n")))))
        if (out.none { it.role != "system" }) throw ApiError(400, "There are no messages to answer.")
        return out
    }

    private fun openaiTools(list: List<Any?>?): List<Map<String, Any?>> = list.orEmpty().mapNotNull { t ->
        (t as? Map<String, Any?>)?.takeIf { it.obj("function")?.str("name") != null }
    }

    /** Reasoning asked for, the ways OpenAI-style clients ask (null: the app's setting). */
    private fun openaiThinking(b: Map<String, Any?>): Boolean? {
        b.obj("chat_template_kwargs")?.bool("enable_thinking")?.let { return it }
        b.bool("enable_thinking")?.let { return it }
        b.bool("think")?.let { return it }
        val effort = b.str("reasoning_effort") ?: b.obj("reasoning")?.str("effort")
        return effort?.let { it != "none" && it != "minimal" }
    }

    private fun stops(v: Any?): List<String> = when (v) {
        is String -> listOf(v)
        is List<*> -> v.filterIsInstance<String>()
        else -> emptyList()
    }

    private suspend fun chatCompletions(req: HttpRequest, r: HttpResponder) {
        val b = body(req)
        val model = engine.model()
        val s = app.settings.flow.first()
        val messages = openaiMessages(model, b.list("messages") ?: throw ApiError(400, "messages is required."))
        val prefill = messages.last().let { it.role == "assistant" && it.toolCalls.isEmpty() }
        val g = GenRequest(
            messages = messages,
            tools = if (b["tool_choice"] == "none") emptyList() else openaiTools(b.list("tools")),
            thinking = (openaiThinking(b) ?: s.thinking) && !prefill,
            maxTokens = (b.num("max_completion_tokens") ?: b.num("max_tokens"))?.toInt() ?: 0,
            temperature = b.num("temperature")?.toFloat(),
            topP = b.num("top_p")?.toFloat(),
            topK = b.num("top_k")?.toInt()?.takeIf { it > 0 },
            presencePenalty = b.num("presence_penalty")?.toFloat(),
            seed = b.num("seed")?.toLong(),
            stop = stops(b["stop"]),
            prefill = prefill,
        )
        val id = engine.id("chatcmpl-")
        val created = System.currentTimeMillis() / 1000
        fun usage(res: GenResult) = linkedMapOf(
            "prompt_tokens" to res.promptTokens,
            "completion_tokens" to res.outputTokens,
            "total_tokens" to res.promptTokens + res.outputTokens,
            "prompt_tokens_details" to linkedMapOf("cached_tokens" to res.cachedTokens),
            "completion_tokens_details" to linkedMapOf("reasoning_tokens" to res.reasoningTokens),
        )
        fun finish(f: Finish) = when (f) {
            Finish.LENGTH -> "length"
            Finish.TOOL_CALLS -> "tool_calls"
            else -> "stop"
        }
        fun calls(res: GenResult) = res.toolCalls.mapIndexed { i, c ->
            linkedMapOf("index" to i, "id" to c.id, "type" to "function",
                "function" to linkedMapOf("name" to c.name, "arguments" to Json.write(c.arguments)))
        }

        if (b.bool("stream") != true) {
            val res = busy { engine.run(g, "call_") {} }
            val message = linkedMapOf<String, Any?>("role" to "assistant",
                "content" to res.content.takeIf { it.isNotEmpty() || res.toolCalls.isEmpty() })
            if (res.reasoning.isNotEmpty()) message["reasoning_content"] = res.reasoning
            if (res.toolCalls.isNotEmpty()) message["tool_calls"] = calls(res).map { it - "index" }
            return r.json(200, linkedMapOf("id" to id, "object" to "chat.completion", "created" to created,
                "model" to model.id, "choices" to listOf(linkedMapOf("index" to 0, "message" to message,
                    "logprobs" to null, "finish_reason" to finish(res.finish))), "usage" to usage(res)))
        }

        fun chunk(delta: Map<String, Any?>, finish: String? = null) = r.event(null, Json.write(linkedMapOf(
            "id" to id, "object" to "chat.completion.chunk", "created" to created, "model" to model.id,
            "choices" to listOf(linkedMapOf("index" to 0, "delta" to delta, "logprobs" to null, "finish_reason" to finish)))))
        // the head goes with the first event: an error before it is still a plain HTTP error
        fun open() {
            if (r.started) return
            r.events()
            chunk(linkedMapOf("role" to "assistant", "content" to ""))
        }
        val res = try {
            busy {
                engine.run(g, "call_") { e ->
                    open()
                    when (e) {
                        GenEvent.Waiting -> r.comment("waiting")
                        is GenEvent.Started -> Unit
                        is GenEvent.Reasoning -> chunk(mapOf("reasoning_content" to e.text))
                        is GenEvent.Content -> chunk(mapOf("content" to e.text))
                    }
                }
            }
        } catch (e: ApiError) {
            if (!r.started) throw e
            r.event(null, Json.write(errorBody(false, e)))
            r.event(null, "[DONE]")
            return
        }
        open()
        if (res.toolCalls.isNotEmpty()) chunk(mapOf("tool_calls" to calls(res)))
        chunk(emptyMap(), finish(res.finish))
        if (b.obj("stream_options")?.bool("include_usage") == true) {
            r.event(null, Json.write(linkedMapOf("id" to id, "object" to "chat.completion.chunk", "created" to created,
                "model" to model.id, "choices" to emptyList<Any>(), "usage" to usage(res))))
        }
        r.event(null, "[DONE]")
    }

    // ---- Anthropic ----

    private suspend fun anthropicImage(model: InstalledModel, b: Map<String, Any?>): Part.Image {
        val src = b.obj("source") ?: throw ApiError(400, "An image block has no source.")
        return when (src.str("type")) {
            "base64" -> engine.image(model, decode(src.str("data").orEmpty()))
            else -> throw ApiError(400, "Images must come as base64; the server fetches nothing.")
        }
    }

    private fun blocks(content: Any?): List<Map<String, Any?>> = when (content) {
        null -> emptyList()
        is String -> listOf(mapOf("type" to "text", "text" to content))
        is List<*> -> content.map { it as? Map<String, Any?> ?: throw ApiError(400, "A content block is not an object.") }
        else -> throw ApiError(400, "A message's content is neither a string nor a list of blocks.")
    }

    private suspend fun anthropicMessages(model: InstalledModel, b: Map<String, Any?>): List<TemplateMessage> {
        val out = ArrayList<TemplateMessage>()
        val system = when (val sys = b["system"]) {
            is String -> sys
            is List<*> -> blocks(sys).filter { it.str("type") == "text" }.joinToString("\n\n") { it.str("text").orEmpty() }
            else -> ""
        }
        if (system.isNotBlank()) out += TemplateMessage("system", listOf(Part.Text(system)))
        for (x in b.list("messages") ?: throw ApiError(400, "messages is required.")) {
            val m = x as? Map<String, Any?> ?: throw ApiError(400, "A message is not an object.")
            val bl = blocks(m["content"])
            when (m.str("role")) {
                "user" -> {
                    // tool results first (the template gathers them into a user message of <tool_response>s), then
                    // the rest as a user message of its own
                    val parts = ArrayList<Part>()
                    for (blk in bl) when (blk.str("type")) {
                        "text" -> parts += Part.Text(blk.str("text").orEmpty())
                        "image" -> parts += anthropicImage(model, blk)
                        "tool_result" -> {
                            val result = ArrayList<Part>()
                            if (blk.bool("is_error") == true) result += Part.Text("Error: ")
                            for (rb in blocks(blk["content"])) when (rb.str("type")) {
                                "text" -> result += Part.Text(rb.str("text").orEmpty())
                                "image" -> result += anthropicImage(model, rb)
                            }
                            out += TemplateMessage("tool", result, toolCallId = blk.str("tool_use_id"))
                        }
                        "document" -> blk.obj("source")?.takeIf { it.str("type") == "text" }?.str("data")
                            ?.let { parts += Part.Text(it) }
                            ?: throw ApiError(400, "Only plain-text documents are supported.")
                    }
                    if (parts.isNotEmpty()) out += TemplateMessage("user", parts)
                }
                "assistant" -> {
                    val parts = bl.filter { it.str("type") == "text" }.map { Part.Text(it.str("text").orEmpty()) }
                    val reasoning = bl.filter { it.str("type") == "thinking" }.joinToString("\n") { it.str("thinking").orEmpty() }
                    val calls = bl.filter { it.str("type") == "tool_use" }.map {
                        ToolCall(it.str("id").orEmpty(), it.str("name").orEmpty(), it.obj("input") ?: emptyMap())
                    }
                    out += TemplateMessage("assistant", parts, reasoning.takeIf { it.isNotEmpty() }, calls)
                }
                else -> throw ApiError(400, "Messages of role ${m.str("role")} are not supported.")
            }
        }
        if (out.none { it.role != "system" }) throw ApiError(400, "There are no messages to answer.")
        return out
    }

    /** Anthropic's tools as the template takes them (OpenAI's shape); the server tools (no input_schema) left out. */
    private fun anthropicTools(list: List<Any?>?): List<Map<String, Any?>> = list.orEmpty().mapNotNull { t ->
        val m = t as? Map<String, Any?> ?: return@mapNotNull null
        val schema = m.obj("input_schema") ?: return@mapNotNull null
        val f = linkedMapOf<String, Any?>("name" to m.str("name").orEmpty())
        m.str("description")?.let { f["description"] = it }
        f["parameters"] = schema
        linkedMapOf("type" to "function", "function" to f)
    }

    private suspend fun anthropicRequest(b: Map<String, Any?>): GenRequest {
        val model = engine.model()
        val messages = anthropicMessages(model, b)
        val prefill = messages.last().let { it.role == "assistant" && it.toolCalls.isEmpty() }
        val th = b.obj("thinking")
        return GenRequest(
            messages = messages,
            tools = if (b.obj("tool_choice")?.str("type") == "none") emptyList() else anthropicTools(b.list("tools")),
            thinking = th?.str("type").let { it == "enabled" || it == "adaptive" } && !prefill,
            thinkBudget = th?.num("budget_tokens")?.toInt() ?: 0,
            maxTokens = b.num("max_tokens")?.toInt() ?: 0,
            temperature = b.num("temperature")?.toFloat(),
            topP = b.num("top_p")?.toFloat(),
            topK = b.num("top_k")?.toInt()?.takeIf { it > 0 },
            stop = stops(b["stop_sequences"]),
            prefill = prefill,
        )
    }

    private suspend fun countTokens(req: HttpRequest, r: HttpResponder) {
        val g = anthropicRequest(body(req))
        r.json(200, mapOf("input_tokens" to engine.count(g)))
    }

    private suspend fun messages(req: HttpRequest, r: HttpResponder) {
        val b = body(req)
        val g = anthropicRequest(b)
        val model = engine.model()
        val id = engine.id("msg_")
        fun stopReason(f: Finish) = when (f) {
            Finish.LENGTH -> "max_tokens"
            Finish.TOOL_CALLS -> "tool_use"
            Finish.STOP_SEQUENCE -> "stop_sequence"
            Finish.STOP -> "end_turn"
        }
        fun usage(prompt: Int, cached: Int, output: Int) = linkedMapOf("input_tokens" to prompt - cached,
            "cache_creation_input_tokens" to 0, "cache_read_input_tokens" to cached, "output_tokens" to output)

        if (b.bool("stream") != true) {
            val res = busy { engine.run(g, "toolu_") {} }
            val content = ArrayList<Map<String, Any?>>()
            if (g.thinking && res.reasoning.isNotEmpty())
                content += linkedMapOf("type" to "thinking", "thinking" to res.reasoning, "signature" to SIGNATURE)
            if (res.content.isNotEmpty() || res.toolCalls.isEmpty()) content += linkedMapOf("type" to "text", "text" to res.content)
            for (c in res.toolCalls) content += linkedMapOf("type" to "tool_use", "id" to c.id, "name" to c.name, "input" to c.arguments)
            return r.json(200, linkedMapOf("id" to id, "type" to "message", "role" to "assistant", "model" to model.id,
                "content" to content, "stop_reason" to stopReason(res.finish), "stop_sequence" to res.stopSequence,
                "usage" to usage(res.promptTokens, res.cachedTokens, res.outputTokens)))
        }

        fun ev(type: String, v: Map<String, Any?>) = r.event(type, Json.write(linkedMapOf<String, Any?>("type" to type) + v))
        var index = -1
        var open: String? = null
        var startedMessage = false
        fun start(type: String, block: Map<String, Any?>) {
            index++
            ev("content_block_start", linkedMapOf("index" to index, "content_block" to block))
            open = type
        }
        fun stop() {
            if (open == "thinking") ev("content_block_delta", linkedMapOf("index" to index,
                "delta" to linkedMapOf("type" to "signature_delta", "signature" to SIGNATURE)))
            if (open != null) ev("content_block_stop", linkedMapOf("index" to index))
            open = null
        }
        fun messageStart(prompt: Int, cached: Int) {
            if (startedMessage) return
            startedMessage = true
            if (!r.started) r.events()
            ev("message_start", linkedMapOf("message" to linkedMapOf("id" to id, "type" to "message", "role" to "assistant",
                "model" to model.id, "content" to emptyList<Any>(), "stop_reason" to null, "stop_sequence" to null,
                "usage" to usage(prompt, cached, 0))))
        }
        val res = try {
            busy {
                engine.run(g, "toolu_") { e ->
                    when (e) {
                        // before message_start a comment (the head goes now), after it Anthropic's ping
                        GenEvent.Waiting -> if (startedMessage) ev("ping", emptyMap()) else {
                            if (!r.started) r.events()
                            r.comment("waiting")
                        }
                        is GenEvent.Started -> messageStart(e.promptTokens, e.cachedTokens)
                        is GenEvent.Reasoning -> if (g.thinking) {
                            if (open != "thinking") {
                                stop()
                                start("thinking", linkedMapOf("type" to "thinking", "thinking" to "", "signature" to ""))
                            }
                            ev("content_block_delta", linkedMapOf("index" to index,
                                "delta" to linkedMapOf("type" to "thinking_delta", "thinking" to e.text)))
                        }
                        is GenEvent.Content -> {
                            if (open != "text") {
                                stop()
                                start("text", linkedMapOf("type" to "text", "text" to ""))
                            }
                            ev("content_block_delta", linkedMapOf("index" to index,
                                "delta" to linkedMapOf("type" to "text_delta", "text" to e.text)))
                        }
                    }
                }
            }
        } catch (e: ApiError) {
            if (!r.started) throw e
            ev("error", linkedMapOf("error" to (errorBody(true, e)["error"])))
            return
        }
        messageStart(res.promptTokens, res.cachedTokens)
        stop()
        if (index < 0 && res.toolCalls.isEmpty()) {
            start("text", linkedMapOf("type" to "text", "text" to ""))
            stop()
        }
        for (c in res.toolCalls) {
            start("tool_use", linkedMapOf("type" to "tool_use", "id" to c.id, "name" to c.name, "input" to emptyMap<String, Any?>()))
            ev("content_block_delta", linkedMapOf("index" to index,
                "delta" to linkedMapOf("type" to "input_json_delta", "partial_json" to Json.write(c.arguments))))
            stop()
        }
        ev("message_delta", linkedMapOf("delta" to linkedMapOf("stop_reason" to stopReason(res.finish),
            "stop_sequence" to res.stopSequence), "usage" to usage(res.promptTokens, res.cachedTokens, res.outputTokens)))
        ev("message_stop", emptyMap())
    }

    /** A generation, shown on the server's screen while it runs. */
    private inline fun <T> busy(block: () -> T): T {
        LocalServer.status.update { it.copy(busy = it.busy + 1) }
        try {
            return block()
        } finally {
            LocalServer.status.update { it.copy(busy = it.busy - 1) }
        }
    }

    companion object {
        /** Thinking blocks need a signature; this server checks none. */
        private const val SIGNATURE = "geode"
    }
}
