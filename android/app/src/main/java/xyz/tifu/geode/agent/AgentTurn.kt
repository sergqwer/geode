package xyz.tifu.geode.agent

import xyz.tifu.geode.ui.chat.AgentStep
import xyz.tifu.geode.ui.chat.ChatMessage
import xyz.tifu.geode.util.Json
import java.io.File

/**
 * A chat's answer as the agent's events build it, the phone's chat and the web chat alike: the model's words and
 * thought as they come; a tool's call becomes a card (its result filled in when it comes), the words before it stay
 * above it and the next words start after it.
 */
fun ChatMessage.withAgent(e: AgentEvent): ChatMessage = when (e) {
    is AgentEvent.Thinking -> copy(thinking = thinking + e.text)
    is AgentEvent.Text -> copy(text = text + e.text)
    is AgentEvent.ToolCall -> {
        val words = if (text.isNotBlank()) listOf(AgentStep("", text.trim())) else emptyList()
        copy(text = "", thinking = "", steps = steps + words + AgentStep(e.tool, "", id = e.id, input = e.input))
    }
    is AgentEvent.ToolResult -> {
        val i = steps.indexOfLast { it.tool.isNotEmpty() && it.ok == null && (e.id.isEmpty() || it.id == e.id) }
        if (i < 0) this else copy(steps = steps.toMutableList().also { it[i] = it[i].copy(ok = e.ok, output = e.output) })
    }
    else -> this
}

/** The answer as it ended: its words trimmed, a tool still running failed, [files] (what it made) under it. */
fun ChatMessage.finished(files: List<String>): ChatMessage = copy(streaming = false, files = files, text = text.trim(),
    steps = steps.map { s -> if (s.ok == null && s.tool.isNotEmpty()) s.copy(ok = false) else s })

object AgentTurns {
    /** The files under an agent's answer, at most. */
    private const val MAX_FILES_SHOWN = 12
    /** The earlier messages told to the agent, at most (characters). */
    private const val EARLIER_MAX = 24_000

    /**
     * The chat's messages before the agent's session began (a chat from before every chat was the agent's, as the
     * user saw it), told to the agent ahead of the new message: the last summary and what came after it, the newest
     * kept when it is long. Empty: nothing before.
     */
    fun earlier(before: List<ChatMessage>): String {
        val done = before.filter { !it.streaming && it.error == null && (it.text.isNotBlank() || it.images.isNotEmpty() || it.audio.isNotEmpty() || it.videos.isNotEmpty()) }
        if (done.isEmpty()) return ""
        val from = done.indexOfLast { it.summary }.coerceAtLeast(0)
        val lines = done.drop(from).map { m ->
            val who = when {
                m.summary -> "Summary of the conversation before"
                m.user -> "User"
                else -> "Assistant"
            }
            val pics = if (m.images.isNotEmpty()) " [${m.images.size} image(s)]" else ""
            val sounds = if (m.audio.isNotEmpty()) " [${m.audio.size} audio clip(s)]" else ""
            val films = if (m.videos.isNotEmpty()) " [${m.videos.size} video(s)]" else ""
            "$who$pics$films$sounds: ${m.text.trim()}"
        }
        var text = lines.joinToString("\n\n")
        if (text.length > EARLIER_MAX) text = "…" + text.takeLast(EARLIER_MAX)
        return "Earlier in this conversation (for context; it went on without tools):\n\n$text\n\n---\n\n"
    }

    /** The workspace's files changed since [since] (the newest first; its dot-folders left out). */
    fun changedSince(dir: File, since: Long): List<String> {
        if (!dir.isDirectory) return emptyList()
        return dir.walkTopDown().onEnter { it == dir || !it.name.startsWith(".") && it.name != "node_modules" }
            .filter { it.isFile && !it.name.startsWith(".") && it.lastModified() >= since - 1000 }
            .sortedByDescending { it.lastModified() }
            .take(MAX_FILES_SHOWN)
            .map { it.relativeTo(dir).path }
            .toList()
    }

    /** A tool call's arguments (JSON) in a line: its main value (the command, the query, the path...), shortened. */
    fun brief(input: String, max: Int = 90): String {
        val o = runCatching { Json.parse(input) as? Map<*, *> }.getOrNull() ?: return input.take(max)
        val main = listOf("command", "cmd", "query", "q", "url", "path", "file_path", "pattern", "name")
            .firstNotNullOfOrNull { k -> o[k]?.takeIf { it is String && it.isNotBlank() } as String? }
            ?: o.values.firstOrNull { it is String && it.isNotBlank() } as String?
            ?: return ""
        val line = main.replace(Regex("\\s+"), " ").trim()
        return if (line.length > max) line.take(max - 1) + "…" else line
    }
}
