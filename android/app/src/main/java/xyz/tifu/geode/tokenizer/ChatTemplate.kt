package xyz.tifu.geode.tokenizer

import xyz.tifu.geode.util.Json

/** A part of a message: its text, or an image in its place among the text ([key] tells two images apart). */
sealed interface Part {
    data class Text(val text: String) : Part
    data class Image(val image: TurnImage, val key: String) : Part
}

/** An assistant's call of a function: its name and its arguments (a JSON object, its keys in their order). */
data class ToolCall(val id: String, val name: String, val arguments: Map<String, Any?>)

/**
 * A message as Qwen3.6's chat template takes it (an API request's): system, user, assistant - its reasoning and its
 * tool calls apart from the content - or tool (a function's output).
 */
data class TemplateMessage(
    val role: String,
    val parts: List<Part>,
    val reasoning: String? = null,
    val toolCalls: List<ToolCall> = emptyList(),
) {
    /** The text with the images left out. */
    val text: String get() = parts.filterIsInstance<Part.Text>().joinToString("") { it.text }

    /** The same message to the conversation the engine holds: the content as the template trims it, the images, the
     *  calls (the reasoning aside: clients seldom send it back). */
    fun sameAs(o: TemplateMessage): Boolean =
        role == o.role && key(parts) == key(o.parts) && calls() == o.calls()

    private fun calls() = toolCalls.map { it.name to Json.write(it.arguments) }

    private fun key(p: List<Part>) = p.joinToString("") {
        when (it) {
            is Part.Text -> it.text
            is Part.Image -> "\u0000${it.key}\u0000"
        }
    }.trim()
}

/** What the template refuses (its raise_exception), in its own words. */
class TemplateException(message: String) : IllegalArgumentException(message)

/**
 * Qwen3.6's whole chat template (the pack's chat_template.jinja), token by token, for the local server's API: tools
 * (their JSON in the system message, calls as XML: <tool_call><function=..><parameter=..>), tool outputs (gathered
 * into a user message of <tool_response> blocks), images in their place among the text, the reasoning of the answers
 * after the last question kept (the earlier ones dropped). TemplateTest checks it against transformers'
 * apply_chat_template. [ChatFormat] stays the app chat's: the same turns without tools.
 *
 * The text between two special tokens is encoded in one go, as Hugging Face's tokenizer splits the rendered string
 * at its added tokens; image pads are put in by the image's size (the engine puts the image's features there).
 */
class ChatTemplate(private val tok: Tokenizer) {
    private val imStart = tok.special("<|im_start|>")
    private val imEnd = tok.special("<|im_end|>")
    private val think = tok.special("<think>")
    private val thinkEnd = tok.special("</think>")
    private val callStart = tok.special("<tool_call>")
    private val callEnd = tok.special("</tool_call>")
    private val responseStart = tok.special("<tool_response>")
    private val responseEnd = tok.special("</tool_response>")
    private val visionStart = runCatching { tok.special("<|vision_start|>") }.getOrNull()
    private val visionEnd = runCatching { tok.special("<|vision_end|>") }.getOrNull()
    private val imagePad = runCatching { tok.special("<|image_pad|>") }.getOrNull()
    /** Never parsed out of text: an image token without its image. */
    private val visionTokens = listOf("<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>",
        "<|video_pad|>").mapNotNull { runCatching { tok.special(it) }.getOrNull() }.toSet()

    /**
     * The conversation from the start: the tools' system message, the messages, the assistant's generation prompt -
     * or with [prefill] the last message (the assistant's) left open, the model going on from it.
     */
    fun render(messages: List<TemplateMessage>, tools: List<Any?>, thinking: Boolean, prefill: Boolean = false): Prompt {
        if (messages.isEmpty()) throw TemplateException("No messages provided.")
        val b = Builder()
        val system = messages[0].takeIf { it.role == "system" }
        if (tools.isNotEmpty()) {
            b.sp(imStart)
            b.t("system\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
            for (t in tools) b.t("\n" + Json.dumps(t))
            b.t("\n</tools>")
            b.t(TOOLS_NOTE)
            if (system != null) {
                val c = textOnly(system, "System").trim()
                if (c.isNotEmpty()) b.t("\n\n" + c)
            }
            b.sp(imEnd)
            b.t("\n")
        } else if (system != null) {
            b.sp(imStart)
            b.t("system\n" + textOnly(system, "System").trim())
            b.sp(imEnd)
            b.t("\n")
        }
        val body = if (prefill) messages.dropLast(1) else messages
        val lastQuery = lastQuery(messages)
        for (i in body.indices) message(b, messages, i, lastQuery)
        if (prefill) {
            b.sp(imStart)
            b.t("assistant\n")
            b.sp(think)
            b.t("\n\n")
            b.sp(thinkEnd)
            b.t("\n\n" + messages.last().text.trim())
        } else {
            generationPrompt(b, thinking)
        }
        return b.prompt()
    }

    /**
     * What continues a conversation the engine holds - [messages]' first [held], the last of them the assistant's
     * answer as the engine generated it (its reasoning in, not yet closed): the answer's end, then the messages after
     * it, then the generation prompt. The answers the engine holds keep their reasoning (the template would drop it
     * from the earlier ones): that is what lets a conversation grow without being read again.
     */
    fun continuation(messages: List<TemplateMessage>, held: Int, thinking: Boolean): Prompt {
        require(held in 1 until messages.size)
        val b = Builder()
        b.sp(imEnd)
        b.t("\n")
        val lastQuery = lastQuery(messages)
        for (i in held until messages.size) message(b, messages, i, lastQuery)
        generationPrompt(b, thinking)
        return b.prompt()
    }

    /** The last user message that is a question, not tool outputs (the answers after it keep their reasoning). */
    private fun lastQuery(messages: List<TemplateMessage>): Int {
        for (i in messages.indices.reversed()) {
            val m = messages[i]
            if (m.role != "user") continue
            val c = contentString(m.parts)
            if (!(c.startsWith("<tool_response>") && c.endsWith("</tool_response>"))) return i
        }
        throw TemplateException("No user query found in messages.")
    }

    private fun message(b: Builder, messages: List<TemplateMessage>, i: Int, lastQuery: Int) {
        val m = messages[i]
        when (m.role) {
            "system" -> if (i != 0) throw TemplateException("System message must be at the beginning.")
            "user" -> {
                b.sp(imStart)
                b.t("user\n")
                content(b, m.parts)
                b.sp(imEnd)
                b.t("\n")
            }
            "assistant" -> assistant(b, m, i > lastQuery)
            "tool" -> {
                if (i > 0 && messages[i - 1].role != "tool") {
                    b.sp(imStart)
                    b.t("user")
                }
                b.t("\n")
                b.sp(responseStart)
                b.t("\n")
                content(b, m.parts)
                b.t("\n")
                b.sp(responseEnd)
                if (i == messages.size - 1 || messages[i + 1].role != "tool") {
                    b.sp(imEnd)
                    b.t("\n")
                }
            }
            else -> throw TemplateException("Unexpected message role.")
        }
    }

    private fun assistant(b: Builder, m: TemplateMessage, afterQuery: Boolean) {
        var content = m.text.trim()
        var reasoning = m.reasoning
        if (reasoning == null && content.contains("</think>")) {
            reasoning = content.substringBefore("</think>").trimEnd('\n').substringAfterLast("<think>").trimStart('\n')
            content = content.substringAfterLast("</think>").trimStart('\n')
        }
        b.sp(imStart)
        if (afterQuery) {
            b.t("assistant\n")
            b.sp(think)
            b.t("\n" + (reasoning ?: "").trim() + "\n")
            b.sp(thinkEnd)
            b.t("\n\n" + content)
        } else {
            b.t("assistant\n" + content)
        }
        m.toolCalls.forEachIndexed { j, call ->
            if (j == 0) {
                if (content.trim().isNotEmpty()) b.t("\n\n")
            } else {
                b.t("\n")
            }
            b.sp(callStart)
            b.t("\n<function=${call.name}>\n")
            for ((k, v) in call.arguments) {
                b.t("<parameter=$k>\n")
                b.t(if (v is String) v else Json.dumps(v))
                b.t("\n</parameter>\n")
            }
            b.t("</function>\n")
            b.sp(callEnd)
        }
        b.sp(imEnd)
        b.t("\n")
    }

    private fun generationPrompt(b: Builder, thinking: Boolean) {
        b.sp(imStart)
        b.t("assistant\n")
        b.sp(think)
        if (thinking) {
            b.t("\n")
        } else {
            b.t("\n\n")
            b.sp(thinkEnd)
            b.t("\n\n")
        }
    }

    /** A message's parts with the template's |trim: the whitespace off the text before the first image and after the
     *  last one. */
    private fun trimmed(parts: List<Part>): List<Part> {
        val out = parts.filter { it !is Part.Text || it.text.isNotEmpty() }.toMutableList()
        while (out.firstOrNull() is Part.Text) {
            val t = (out[0] as Part.Text).text.trimStart()
            if (t.isNotEmpty()) {
                out[0] = Part.Text(t)
                break
            }
            out.removeAt(0)
        }
        while (out.lastOrNull() is Part.Text) {
            val t = (out.last() as Part.Text).text.trimEnd()
            if (t.isNotEmpty()) {
                out[out.size - 1] = Part.Text(t)
                break
            }
            out.removeAt(out.size - 1)
        }
        return out
    }

    private fun content(b: Builder, parts: List<Part>) {
        for (p in trimmed(parts)) when (p) {
            is Part.Text -> b.t(p.text)
            is Part.Image -> {
                val vs = visionStart
                val ve = visionEnd
                val pad = imagePad
                if (vs == null || ve == null || pad == null) throw TemplateException("This model reads no images.")
                b.sp(vs)
                repeat(p.image.tokens) { b.sp(pad) }
                b.sp(ve)
                b.images += p.image
            }
        }
    }

    /** The content as the template's string (an image as its three tokens' text), trimmed. */
    private fun contentString(parts: List<Part>): String = trimmed(parts).joinToString("") {
        when (it) {
            is Part.Text -> it.text
            is Part.Image -> "<|vision_start|><|image_pad|><|vision_end|>"
        }
    }

    private fun textOnly(m: TemplateMessage, role: String): String {
        if (m.parts.any { it is Part.Image }) throw TemplateException("$role message cannot contain images.")
        return m.text
    }

    /** The ids as the template's string would be tokenized: special tokens by id, the text between them in one go. */
    private inner class Builder {
        val ids = ArrayList<Int>()
        val images = ArrayList<TurnImage>()
        private val text = StringBuilder()

        fun t(s: String) {
            text.append(s)
        }

        fun sp(id: Int) {
            flush()
            ids += id
        }

        private fun flush() {
            if (text.isEmpty()) return
            for (id in tok.encodeWithSpecials(text.toString(), visionTokens)) ids += id
            text.setLength(0)
        }

        fun prompt(): Prompt {
            flush()
            return Prompt(ids.toIntArray(), images)
        }
    }

    companion object {
        /** The template's instructions after the tools (its <tool_call> tags become the special tokens). */
        private const val TOOLS_NOTE = "\n\nIf you choose to call a function ONLY reply in the following format " +
            "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n" +
            "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n" +
            "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be " +
            "nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may " +
            "provide optional reasoning for your function call in natural language BEFORE the function call, but " +
            "NOT after\n- If there is no function call available, answer the question like normal with your current " +
            "knowledge and do not tell the user about function calls\n</IMPORTANT>"
    }
}
