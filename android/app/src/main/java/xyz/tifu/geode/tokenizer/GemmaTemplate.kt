package xyz.tifu.geode.tokenizer

import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.JsonNumber
import java.util.Locale

/**
 * Gemma 4's whole chat template (its chat_template.jinja, "Gemma 4 Canonical", 2026-07-09), token by token, for the
 * local server's API; GemmaTemplateTest checks it against transformers' apply_chat_template.
 *
 *     <bos><|turn>system\n[<|think|>\n]{system}<|tool>declaration:NAME{...}<tool|>...<turn|>\n
 *     <|turn>user\n{text, an image as <|image> n x <|image|> <image|>}<turn|>\n
 *     <|turn>model\n[<|channel>thought\n{reasoning}\n<channel|>]<|tool_call>call:NAME{k:v,...}<tool_call|>
 *         <|tool_response>response:NAME{value:<|"|>{output}<|"|>}<tool_response|>{content}<turn|>\n
 *     <|turn>model\n                                            (the generation prompt)
 *
 * A call's outputs stay in the model's turn, and the model goes on in it after them (no generation prompt: with
 * reasoning on, "<|channel>thought\n" opens it). Strings in a declaration or a call are between <|"|> tokens, keys in
 * order of the template's dictsort (case-insensitive). One deliberate difference: a model's content beside its calls
 * goes before them, where the model writes it - the template puts it after the calls' outputs and closes the turn
 * there without opening another, which leaves the model to answer outside any turn. The reasoning of the answers after the last user message is
 * kept, the earlier ones' dropped. The rendered text is tokenized as Hugging Face's tokenizer splits it at its added
 * tokens, but for the media tokens (an image pad without its image would leave the engine a hole): an image is put in
 * as Gemma4Processor expands its placeholder.
 */
class GemmaTemplate(private val tok: Tokenizer) : ApiTemplate {
    private val turnEnd = tok.special("<turn|>")
    private val bos = tok.special("<bos>")
    private val turnStart = tok.special("<|turn>")
    override val thinkStart: Int = tok.special("<|channel>")
    override val thinkEnd: Int = tok.special("<channel|>")
    override val callStart: Int = tok.special("<|tool_call>")
    override val modelOpensReasoning: Boolean get() = true
    override val stopTokens: IntArray =
        intArrayOf(turnEnd, tok.special("<eos>"), tok.special("<|tool_response>"))

    private fun opt(s: String) = if (tok.hasSpecial(s)) tok.special(s) else null
    private val imageStart = opt("<|image>")
    private val imagePad = opt("<|image|>")
    private val imageEnd = opt("<image|>")
    private val audioStart = opt("<|audio>")
    private val audioPad = opt("<|audio|>")
    private val audioEnd = opt("<audio|>")
    private val videoPad = opt("<|video|>")

    /** Never parsed out of text: a media token without its media. */
    private val mediaTokens = listOf("<|image>", "<|image|>", "<image|>", "<|audio>", "<|audio|>", "<audio|>",
        "<|video|>").mapNotNull(::opt).toSet()

    override fun render(messages: List<TemplateMessage>, tools: List<Any?>, thinking: Boolean, prefill: Boolean): Prompt {
        if (messages.isEmpty()) throw TemplateException("No messages provided.")
        if (!prefill) return build(messages, tools, thinking, generation = true).prompt()
        // the last message open (transformers' continue_final_message): the render cut after its content
        val b = build(messages, tools, thinking, generation = false, mark = messages.size - 1)
        return b.cut(0, b.markIds, 0, b.markImages, false)
    }

    override fun continuation(messages: List<TemplateMessage>, held: Int, thinking: Boolean): Prompt {
        require(held in 1 until messages.size)
        // the render from the held answer's end on: its calls' outputs, or its closing (the tools' declarations
        // are before it)
        val b = build(messages, emptyList(), thinking, generation = true, mark = held - 1)
        val all = b.prompt()
        // (media: the clips count too, not just the images)
        return b.cut(b.markIds, all.size, b.markImages, all.media.size, all.reasoning)
    }

    override fun closeThinking(): IntArray {
        val out = ArrayList<Int>()
        out += tok.encode("\n\nConsidering the limited time by the user, I have to give the solution based on the " +
            "thinking directly now.\n").toList()
        out += thinkEnd
        return out.toIntArray()
    }

    override fun systemTurn(p: Prompt): Int {
        if (p.size < 4 || p.ids[0] != bos || p.ids[1] != turnStart) return 0
        if (!tok.decode(p.ids.copyOfRange(2, minOf(p.size, 6))).startsWith("system\n")) return 0
        val i = p.ids.indexOf(turnEnd)
        val nl = tok.encode("\n")
        if (i < 0 || nl.size != 1 || i + 1 >= p.size || p.ids[i + 1] != nl[0]) return 0
        // a system turn holds no media (their tokens are read with their features: not a plain prefix)
        if ((0..i).any { p.ids[it] in mediaTokens }) return 0
        return i + 2
    }

    // ---- the template ----

    /**
     * The render; with [mark] the place where that message's own part ends - after its calls when it has some,
     * else after its content.
     */
    private fun build(messages: List<TemplateMessage>, tools: List<Any?>, thinking: Boolean, generation: Boolean,
                      mark: Int = -1): Builder {
        val b = Builder()
        b.t("<bos>")
        var prevType: String? = null
        var prevNonToolRole: String? = null
        val sys = messages[0].takeIf { it.role == "system" || it.role == "developer" }
        if (thinking || tools.isNotEmpty() || sys != null) {
            b.t("<|turn>system\n")
            if (thinking) {
                b.t("<|think|>\n")
                prevType = "think"
            }
            if (sys != null) {
                if (sys.parts.any { it !is Part.Text }) throw TemplateException("System message cannot contain images.")
                b.t(sys.text.trim())
            }
            if (tools.isNotEmpty()) {
                for (t in tools) {
                    b.t("<|tool>")
                    b.t(declaration(t).trim())
                    b.t("<tool|>")
                }
                prevType = "tool"
            }
            b.t("<turn|>\n")
        }
        val first = if (sys != null) 1 else 0
        val loop = messages.subList(first, messages.size)
        val lastUser = loop.indexOfLast { it.role == "user" }
        for ((i, m) in loop.withIndex()) {
            if (m.role == "tool") continue
            prevType = null
            val role = if (m.role == "assistant") "model" else m.role
            val marked = i + first == mark
            if (!(role == "model" && prevNonToolRole == "assistant")) b.t("<|turn>$role\n")
            val reasoning = m.reasoning
            if (!reasoning.isNullOrEmpty() && i > lastUser) b.t("<|channel>thought\n$reasoning\n<channel|>")
            // (the deliberate difference: a model's content with calls before them)
            val contentFirst = role == "model" && m.toolCalls.isNotEmpty()
            if (contentFirst) content(b, m, role)
            if (m.toolCalls.isNotEmpty()) {
                for (c in m.toolCalls) {
                    b.t("<|tool_call>call:${c.name}{")
                    var firstArg = true
                    for ((k, v) in dictsort(c.arguments)) {
                        if (!firstArg) b.t(",")
                        firstArg = false
                        b.t("$k:")
                        b.t(argument(v, escapeKeys = false))
                    }
                    b.t("}<tool_call|>")
                }
                prevType = "tool_call"
                if (marked) b.mark()
            }
            // the outputs of its calls: the tool messages that follow it
            var responded = false
            if (m.toolCalls.isNotEmpty()) {
                var k = i + 1
                var n = 0
                while (k < loop.size && loop[k].role == "tool") {
                    val follow = loop[k]
                    // the call it answers by its id (the template's way), else by its place (the template says
                    // "unknown")
                    val name = m.toolCalls.firstOrNull { it.id.isNotEmpty() && it.id == follow.toolCallId }?.name
                        ?: m.toolCalls.getOrNull(n)?.name ?: "unknown"
                    b.t("<|tool_response>response:$name{value:" + argument(follow.text, escapeKeys = false) +
                        "}<tool_response|>")
                    for (p in follow.parts) b.medium(p)
                    responded = true
                    prevType = "tool_response"
                    k++
                    n++
                }
            }
            val hasContent = if (contentFirst) false else content(b, m, role)
            if (marked && m.toolCalls.isEmpty()) b.mark()
            val next = loop.subList(i + 1, loop.size).firstOrNull { it.role != "tool" }?.role
            val continuesIntoNext = role == "model" && next == "assistant" && (m.toolCalls.isEmpty() || responded)
            when {
                prevType == "tool_call" && !responded -> b.t("<|tool_response>")
                continuesIntoNext -> Unit
                !(responded && !hasContent && next == null) -> b.t("<turn|>\n")
            }
            prevNonToolRole = m.role
        }
        if (generation) {
            if (prevType != "tool_response" && prevType != "tool_call") {
                b.t("<|turn>model\n")
            } else if (prevType == "tool_response" && thinking) {
                b.t("<|channel>thought\n")
                b.reasoning = true
            }
        }
        return b
    }

    /** A message's content: each text trimmed (a model's without its thinking), the images in their place; whether
     *  anything was put in. */
    private fun content(b: Builder, m: TemplateMessage, role: String): Boolean {
        var any = false
        for (p in m.parts) when (p) {
            is Part.Text -> {
                val t = if (role == "model") stripThinking(p.text) else p.text.trim()
                if (t.isNotEmpty()) any = true
                b.t(t)
            }
            else -> {
                b.medium(p)
                any = true
            }
        }
        return any
    }

    /** The template's strip_thinking: what is between <|channel> and <channel|> left out, trimmed. */
    private fun stripThinking(text: String): String {
        val sb = StringBuilder()
        for (part in text.split("<channel|>")) sb.append(if ("<|channel>" in part) part.substringBefore("<|channel>") else part)
        return sb.toString().trim()
    }

    /** The template's format_function_declaration (an OpenAI-style {"type": "function", "function": {...}}). */
    private fun declaration(tool: Any?): String {
        val f = at(tool, "function")
        val sb = StringBuilder()
        sb.append("declaration:").append(str(at(f, "name"))).append("{description:").append(Q)
            .append(str(at(f, "description"))).append(Q)
        val params = at(f, "parameters")
        if (truthy(params)) {
            sb.append(",parameters:{")
            val props = at(params, "properties")
            if (truthy(props)) {
                sb.append("properties:{")
                if (props is Map<*, *>) parameters(sb, props, false)
                sb.append("},")
            }
            val req = at(params, "required")
            if (truthy(req)) {
                sb.append("required:[")
                iterate(req).forEachIndexed { j, r ->
                    if (j > 0) sb.append(',')
                    sb.append(Q).append(str(r)).append(Q)
                }
                sb.append("],")
            }
            val type = at(params, "type")
            if (truthy(type)) sb.append("type:").append(Q).append(upper(type)).append(Q).append('}')
        }
        if (f is Map<*, *> && f.containsKey("response")) {
            val r = f["response"]
            sb.append(",response:{")
            if (truthy(at(r, "description"))) sb.append("description:").append(Q).append(str(at(r, "description")))
                .append(Q).append(',')
            if (upper(at(r, "type")) == "OBJECT") sb.append("type:").append(Q).append(upper(at(r, "type"))).append(Q)
                .append('}')
        }
        sb.append('}')
        return sb.toString()
    }

    /** The template's format_parameters: a schema's properties, [filterKeys] leaving out the standard ones. */
    private fun parameters(sb: StringBuilder, properties: Map<*, *>, filterKeys: Boolean) {
        var foundFirst = false
        for ((key, value) in dictsort(properties)) {
            if (filterKeys && key in STANDARD_KEYS) continue
            var comma = false
            fun sep() {
                if (comma) sb.append(',') else comma = true
            }
            if (foundFirst) sb.append(',')
            foundFirst = true
            sb.append(key).append(":{")
            if (truthy(at(value, "description"))) {
                sb.append("description:").append(Q).append(str(at(value, "description"))).append(Q)
                comma = true
            }
            val type = upper(at(value, "type"))
            if (type == "STRING") {
                if (truthy(at(value, "enum"))) {
                    sep()
                    sb.append("enum:").append(argument(at(value, "enum"), escapeKeys = true))
                }
            } else if (type == "ARRAY") {
                val items = at(value, "items")
                if (items is Map<*, *> && items.isNotEmpty()) {
                    sep()
                    sb.append("items:{")
                    var firstItem = true
                    for ((ik, iv) in dictsort(items)) {
                        if (iv == null) continue
                        if (!firstItem) sb.append(',')
                        firstItem = false
                        when (ik) {
                            "properties" -> {
                                sb.append("properties:{")
                                if (iv is Map<*, *>) parameters(sb, iv, false)
                                sb.append('}')
                            }
                            "required" -> {
                                sb.append("required:[")
                                iterate(iv).forEachIndexed { j, r ->
                                    if (j > 0) sb.append(',')
                                    sb.append(Q).append(str(r)).append(Q)
                                }
                                sb.append(']')
                            }
                            "type" -> sb.append("type:").append(
                                if (iv is String) argument(upper(iv), escapeKeys = true)
                                else argument(iterate(iv).map { upper(it) }, escapeKeys = true))
                            else -> sb.append(ik).append(':').append(argument(iv, escapeKeys = true))
                        }
                    }
                    sb.append('}')
                }
            }
            if (truthy(at(value, "nullable"))) {
                sep()
                sb.append("nullable:true")
            }
            if (type == "OBJECT") {
                val props = at(value, "properties")
                if (props is Map<*, *>) {
                    sep()
                    sb.append("properties:{")
                    parameters(sb, props, false)
                    sb.append('}')
                } else if (value is Map<*, *>) {
                    sep()
                    sb.append("properties:{")
                    parameters(sb, value, true)
                    sb.append('}')
                }
                val req = at(value, "required")
                if (truthy(req)) {
                    sep()
                    sb.append("required:[")
                    iterate(req).forEachIndexed { j, r ->
                        if (j > 0) sb.append(',')
                        sb.append(Q).append(str(r)).append(Q)
                    }
                    sb.append(']')
                }
            }
            sep()
            sb.append("type:").append(Q).append(type).append(Q).append('}')
        }
    }

    /** The template's format_argument. */
    private fun argument(v: Any?, escapeKeys: Boolean): String = when (v) {
        null -> "null"
        Undefined -> ""
        is String -> Q + v + Q
        is Boolean -> if (v) "true" else "false"
        is Map<*, *> -> dictsort(v).joinToString(",", "{", "}") { (k, x) ->
            (if (escapeKeys) Q + k + Q else k) + ":" + argument(x, escapeKeys)
        }
        is List<*> -> v.joinToString(",", "[", "]") { argument(it, escapeKeys) }
        is Number -> Json.dumps(v)
        else -> v.toString()
    }

    /** Jinja's dictsort: by key, ignoring case. */
    private fun dictsort(m: Map<*, *>): List<Pair<String, Any?>> =
        m.entries.map { it.key.toString() to it.value }.sortedBy { it.first.lowercase(Locale.ROOT) }

    private fun iterate(v: Any?): List<Any?> = when (v) {
        is List<*> -> v
        is String -> v.map { it.toString() }
        else -> emptyList()
    }

    /** A Jinja value's subscript: a missing key (or not a mapping) is undefined, not None. */
    private fun at(v: Any?, key: String): Any? = if (v is Map<*, *> && v.containsKey(key)) v[key] else Undefined

    private fun truthy(v: Any?): Boolean = when (v) {
        null, Undefined -> false
        is Boolean -> v
        is String -> v.isNotEmpty()
        is Collection<*> -> v.isNotEmpty()
        is Map<*, *> -> v.isNotEmpty()
        is Number -> v.toDouble() != 0.0
        else -> true
    }

    /** A value printed as Jinja prints it (Python's str). */
    private fun str(v: Any?): String = when (v) {
        Undefined -> ""
        else -> repr(v, top = true)
    }

    private fun upper(v: Any?): String = str(v).uppercase(Locale.ROOT)

    private fun repr(v: Any?, top: Boolean = false): String = when (v) {
        null -> "None"
        is Boolean -> if (v) "True" else "False"
        is String -> if (top) v else "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
        is List<*> -> v.joinToString(", ", "[", "]") { repr(it) }
        is Map<*, *> -> v.entries.joinToString(", ", "{", "}") { repr(it.key.toString()) + ": " + repr(it.value) }
        is Number -> Json.dumps(v)
        else -> v.toString()
    }

    private object Undefined

    // ---- the model's calls ----

    override fun parseCalls(text: String, tools: List<Map<String, Any?>>): List<Pair<String, Map<String, Any?>>> {
        val out = ArrayList<Pair<String, Map<String, Any?>>>()
        var i = text.indexOf(CALL_OPEN)
        while (i >= 0) {
            var p = i + CALL_OPEN.length
            val next = text.indexOf(CALL_OPEN, p).let { if (it < 0) text.length else it }
            while (p < text.length && text[p].isWhitespace()) p++
            if (text.startsWith("call:", p)) p += 5
            // the arguments' brace within this call (a broken one must not take the next call's)
            val brace = text.indexOf('{', p).takeIf { it in 0 until next }
            val name = if (brace == null) "" else text.substring(p, brace).trim()
            if (brace == null || name.isEmpty() || name.any { it == '<' || it == '>' || it == '}' || it == ',' }) {
                i = if (next < text.length) next else -1
                continue
            }
            val parser = Args(text, brace)
            val args = runCatching { parser.value() }.getOrNull()
            if (args is Map<*, *>) {
                @Suppress("UNCHECKED_CAST")
                out += name to (args as Map<String, Any?>)
            }
            i = text.indexOf(CALL_OPEN, if (args is Map<*, *>) maxOf(parser.i, brace + 1) else brace + 1)
        }
        return out
    }

    override fun callName(text: String): String? {
        val i = text.lastIndexOf(CALL_OPEN).takeIf { it >= 0 } ?: return null
        var p = i + CALL_OPEN.length
        while (p < text.length && text[p].isWhitespace()) p++
        if (text.startsWith("call:", p)) p += 5
        // the name ends at the arguments' brace: not before it is written
        val brace = text.indexOf('{', p).takeIf { it >= 0 } ?: return null
        return text.substring(p, brace).trim()
            .takeIf { n -> n.isNotEmpty() && n.none { it == '<' || it == '>' || it == '}' || it == ',' } }
    }

    /** A call's arguments as the model writes them: {key:value,...}, strings between <|"|>, keys bare or quoted. */
    private class Args(private val s: String, var i: Int) {
        fun value(): Any? {
            ws()
            if (s.startsWith(Q, i)) {
                val end = s.indexOf(Q, i + Q.length)
                if (end < 0) throw IllegalArgumentException("an open string")
                val v = s.substring(i + Q.length, end)
                i = end + Q.length
                return v
            }
            if (i >= s.length) throw IllegalArgumentException("the end")
            return when (s[i]) {
                '{' -> obj()
                '[' -> list()
                else -> bare()
            }
        }

        private fun obj(): Map<String, Any?> {
            i++
            val m = LinkedHashMap<String, Any?>()
            ws()
            if (i < s.length && s[i] == '}') {
                i++
                return m
            }
            while (true) {
                ws()
                // a trailing comma
                if (i < s.length && s[i] == '}') {
                    i++
                    return m
                }
                val key = if (s.startsWith(Q, i)) value() as String else {
                    val start = i
                    while (i < s.length && s[i] !in KEY_END) i++
                    if (i >= s.length || s[i] != ':') throw IllegalArgumentException("':' expected")
                    s.substring(start, i).trim().also { if (it.isEmpty()) throw IllegalArgumentException("no key") }
                }
                ws()
                if (i >= s.length || s[i] != ':') throw IllegalArgumentException("':' expected")
                i++
                m[key] = value()
                ws()
                if (i >= s.length) throw IllegalArgumentException("the end")
                when (s[i++]) {
                    ',' -> continue
                    '}' -> return m
                    else -> throw IllegalArgumentException("',' or '}' expected")
                }
            }
        }

        private fun list(): List<Any?> {
            i++
            val a = ArrayList<Any?>()
            ws()
            if (i < s.length && s[i] == ']') {
                i++
                return a
            }
            while (true) {
                ws()
                // a trailing comma
                if (i < s.length && s[i] == ']') {
                    i++
                    return a
                }
                a += value()
                ws()
                if (i >= s.length) throw IllegalArgumentException("the end")
                when (s[i++]) {
                    ',' -> continue
                    ']' -> return a
                    else -> throw IllegalArgumentException("',' or ']' expected")
                }
            }
        }

        /** A number, true, false, null - or a word the model left unquoted (a string). */
        private fun bare(): Any? {
            val start = i
            while (i < s.length && s[i] != ',' && s[i] != '}' && s[i] != ']') i++
            val t = s.substring(start, i).trim()
            return when {
                t == "true" -> true
                t == "false" -> false
                t == "null" -> null
                INT.matches(t) -> t.toLongOrNull() ?: JsonNumber(t)
                NUM.matches(t) -> JsonNumber(t)
                else -> t
            }
        }

        private fun ws() {
            while (i < s.length && s[i].isWhitespace()) i++
        }
    }

    // ---- tokens ----

    /** The ids as the template's string would be tokenized: the text in one go between the media, split at the
     *  added tokens. */
    private inner class Builder {
        private val ids = ArrayList<Int>()
        /** The media in the prompt's order. */
        private val images = ArrayList<TurnMedia>()
        private val text = StringBuilder()
        var reasoning = false
        var markIds = -1
            private set
        var markImages = -1
            private set

        fun t(s: String) {
            text.append(s)
        }

        /** A medium as Gemma4Processor expands its placeholder; text: nothing. */
        fun medium(p: Part) {
            when (p) {
                is Part.Text -> Unit
                is Part.Image -> image(p.image)
                is Part.Audio -> audio(p.audio)
                is Part.Video -> video(p)
            }
        }

        fun image(im: TurnImage) {
            val s = imageStart
            val p = imagePad
            val e = imageEnd
            if (s == null || p == null || e == null) throw TemplateException("This model reads no images.")
            flush()
            ids += s
            repeat(tokens(im)) { ids += p }
            ids += e
            images += im
        }

        fun audio(a: TurnAudio) {
            val s = audioStart
            val p = audioPad
            val e = audioEnd
            if (s == null || p == null || e == null) throw TemplateException("This model hears no audio.")
            val n = a.tokens
            if (n <= 0) throw TemplateException("A sound clip is too short.")
            flush()
            ids += s
            repeat(n) { ids += p }
            ids += e
            images += a
        }

        /** Its frames: "MM:SS <|image>", the frame's <|video|> tokens, "<image|>", joined by spaces; its sound after. */
        fun video(v: Part.Video) {
            val s = imageStart
            val p = videoPad
            val e = imageEnd
            if (s == null || p == null || e == null) throw TemplateException("This model watches no video.")
            if (v.frames.isEmpty()) throw TemplateException("A video has no frames.")
            v.frames.forEachIndexed { i, fr ->
                t((if (i > 0) " " else "") + "%02d:%02d ".format(Locale.ROOT, fr.second / 60, fr.second % 60))
                flush()
                ids += s
                repeat(tokens(fr)) { ids += p }
                ids += e
                images += if (fr.frame) fr else fr.copy(frame = true)
            }
            v.audio?.let { audio(it) }
        }

        fun mark() {
            flush()
            markIds = ids.size
            markImages = images.size
        }

        private fun flush() {
            if (text.isEmpty()) return
            for (id in tok.encodeWithSpecials(text.toString(), mediaTokens)) ids += id
            text.setLength(0)
        }

        fun prompt(): Prompt {
            flush()
            return Prompt(ids.toIntArray(), images.toList(), reasoning)
        }

        fun cut(from: Int, to: Int, imagesFrom: Int, imagesTo: Int, reasoning: Boolean): Prompt {
            flush()
            check(markIds >= 0) { "no mark" }
            return Prompt(ids.subList(from, to).toIntArray(), images.subList(imagesFrom, imagesTo).toList(), reasoning)
        }
    }

    companion object {
        private const val Q = "<|\"|>"
        private const val CALL_OPEN = "<|tool_call>"
        private val STANDARD_KEYS = setOf("description", "type", "properties", "required", "nullable")
        /** A number as JSON writes one (anything else the model left unquoted is a string). */
        private val INT = Regex("-?(0|[1-9]\\d*)")
        private val NUM = Regex("-?(0|[1-9]\\d*)(\\.\\d+)?([eE][-+]?\\d+)?")
        /** What ends a bare key: its colon, or what no key holds. */
        private val KEY_END = setOf(':', ',', '{', '}', '[', ']', '<')

        /** An image's soft tokens in Gemma 4 (ImageStore.fitGemma's size). */
        fun tokens(im: TurnImage): Int = xyz.tifu.geode.vision.ImageStore.gemmaTokens(im.width, im.height)
    }
}
