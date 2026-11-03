package xyz.tifu.geode.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import xyz.tifu.geode.ui.chat.AgentStep
import xyz.tifu.geode.ui.chat.ChatMessage
import xyz.tifu.geode.ui.chat.ChatVideo
import xyz.tifu.geode.vision.ImageStore
import java.io.File
import java.io.FileOutputStream

/** A saved chat in the list: its title (the first question), when it last changed. */
data class ChatInfo(val id: String, val title: String, val updated: Long)

/**
 * A chat's token counts over all its answers: the prompt tokens the model read, the ones it found in its context
 * already (the conversation so far, kept in the engine: the cache hits), and the tokens it generated.
 */
data class ChatStats(val read: Long = 0, val cached: Long = 0, val generated: Long = 0) {
    /** The share of the prompt tokens served from the cache (null before the first answer). */
    val hitRate: Double? get() = (read + cached).takeIf { it > 0 }?.let { cached.toDouble() / it }
}

/** A chat as saved; [agent]: its messages go to the agent, [agentSession] the agent's session to go on with. */
data class SavedChat(
    val title: String,
    val messages: List<ChatMessage>,
    val stats: ChatStats,
    val agent: Boolean = false,
    val agentSession: String? = null,
)

/**
 * The chats, one JSON file each in files/chats/<id>.json: the title, the times and every message (a summary from
 * a context compression included). Each save writes the whole chat aside, syncs it and renames it into place, so a
 * file is a whole chat or the one before (the settings file once came back as zero bytes after a restart).
 */
class ChatStore(context: Context) {
    private val dir = File(context.filesDir, "chats").apply { mkdirs() }
    private val images = ImageStore(context)
    private val clips = xyz.tifu.geode.media.AudioStore(context)
    private val agents = File(context.filesDir, "agents")

    private fun file(id: String) = File(dir, "$id.json")

    /** Every chat, the most recently changed first. */
    fun list(): List<ChatInfo> = (dir.listFiles() ?: emptyArray())
        .filter { it.isFile && it.name.endsWith(".json") }
        .mapNotNull { f ->
            runCatching {
                val o = JSONObject(f.readText())
                ChatInfo(f.name.removeSuffix(".json"), o.optString("title"), o.optLong("updated"))
            }.getOrNull()
        }
        .sortedByDescending { it.updated }

    fun load(id: String): SavedChat? = runCatching {
        val o = JSONObject(file(id).readText())
        val arr = o.getJSONArray("messages")
        val st = o.optJSONObject("stats")
        val stats = ChatStats(st?.optLong("read") ?: 0, st?.optLong("cached") ?: 0, st?.optLong("generated") ?: 0)
        val messages = (0 until arr.length()).map { i ->
            val m = arr.getJSONObject(i)
            ChatMessage(
                id = m.getLong("id"),
                user = m.getBoolean("user"),
                text = m.optString("text"),
                thinking = m.optString("thinking"),
                error = m.optString("error").takeIf { it.isNotEmpty() },
                tokensPerSecond = m.optDouble("tps", 0.0).toFloat(),
                thinkingCut = m.optBoolean("cut"),
                endedThinking = m.optBoolean("ended"),
                summary = m.optBoolean("summary"),
                images = m.optJSONArray("images")?.let { a -> (0 until a.length()).map { a.getString(it) } }
                    ?: emptyList(),
                audio = m.optJSONArray("audio")?.let { a -> (0 until a.length()).map { a.getString(it) } }
                    ?: emptyList(),
                videos = m.optJSONArray("videos")?.let { a ->
                    (0 until a.length()).map { i ->
                        val v = a.getJSONObject(i)
                        val fr = v.optJSONArray("frames")
                        val sec = v.optJSONArray("seconds")
                        ChatVideo((0 until (fr?.length() ?: 0)).map { fr!!.getString(it) },
                            (0 until (sec?.length() ?: 0)).map { sec!!.getInt(it) },
                            v.optString("audio").takeIf { it.isNotEmpty() }, v.optLong("ms"))
                    }
                } ?: emptyList(),
                steps = m.optJSONArray("steps")?.let { a ->
                    (0 until a.length()).map { i ->
                        val s = a.getJSONObject(i)
                        AgentStep(s.optString("tool"), s.optString("title"), s.optString("output"),
                            if (s.has("ok")) s.getBoolean("ok") else null, s.optString("id"), s.optString("input"))
                    }
                } ?: emptyList(),
                files = m.optJSONArray("files")?.let { a -> (0 until a.length()).map { a.getString(it) } } ?: emptyList(),
            )
        }
        SavedChat(o.optString("title"), messages, stats, o.optBoolean("agent"),
            o.optString("agentSession").takeIf { it.isNotEmpty() })
    }.getOrNull()

    fun save(id: String, title: String, created: Long, messages: List<ChatMessage>, stats: ChatStats,
             agent: Boolean = false, agentSession: String? = null) {
        val arr = JSONArray()
        for (m in messages) {
            if (m.streaming) continue   // half an answer: the next save has it whole
            arr.put(JSONObject().apply {
                put("id", m.id)
                put("user", m.user)
                put("text", m.text)
                if (m.thinking.isNotEmpty()) put("thinking", m.thinking)
                m.error?.let { put("error", it) }
                if (m.tokensPerSecond > 0f) put("tps", m.tokensPerSecond.toDouble())
                if (m.thinkingCut) put("cut", true)
                if (m.endedThinking) put("ended", true)
                if (m.summary) put("summary", true)
                if (m.images.isNotEmpty()) put("images", JSONArray(m.images))
                if (m.audio.isNotEmpty()) put("audio", JSONArray(m.audio))
                if (m.videos.isNotEmpty()) put("videos", JSONArray().apply {
                    for (v in m.videos) put(JSONObject().apply {
                        put("frames", JSONArray(v.frames))
                        put("seconds", JSONArray(v.seconds))
                        v.audio?.let { put("audio", it) }
                        put("ms", v.ms)
                    })
                })
                if (m.steps.isNotEmpty()) put("steps", JSONArray().apply {
                    for (s in m.steps) put(JSONObject().apply {
                        put("tool", s.tool)
                        put("title", s.title)
                        if (s.output.isNotEmpty()) put("output", s.output)
                        s.ok?.let { put("ok", it) }
                        if (s.id.isNotEmpty()) put("id", s.id)
                        if (s.input.isNotEmpty()) put("input", s.input)
                    })
                })
                if (m.files.isNotEmpty()) put("files", JSONArray(m.files))
            })
        }
        val o = JSONObject().apply {
            put("title", title)
            put("created", created)
            put("updated", System.currentTimeMillis())
            put("messages", arr)
            if (agent) put("agent", true)
            agentSession?.let { put("agentSession", it) }
            put("stats", JSONObject().apply {
                put("read", stats.read)
                put("cached", stats.cached)
                put("generated", stats.generated)
            })
        }
        val target = file(id)
        val tmp = File(dir, "$id.json.tmp")
        FileOutputStream(tmp).use {
            it.write(o.toString().toByteArray())
            it.fd.sync()
        }
        tmp.renameTo(target)
    }

    /**
     * The chat, the images and clips its questions had (and the local server's copies of them the agent's requests
     * made), and its agent's Linux (files/agents/<id>).
     */
    fun delete(id: String) {
        load(id)?.messages?.let { msgs ->
            val pictures = msgs.flatMap { it.images } + msgs.flatMap { m -> m.videos.flatMap { it.frames } }
            val sounds = msgs.flatMap { it.audio } + msgs.flatMap { m -> m.videos.mapNotNull { it.audio } }
            // the copies first: their names come from these files' bytes. A copy another chat shares (a fork's, the
            // same file attached there too) stays - that chat's request may be reading it right now - and goes with
            // that chat, or unused for a day; all stay when a chat cannot be read
            heldMedia(except = id)?.let { others ->
                images.deleteServerCopies(pictures - sameBytes(pictures, others, images::file))
                clips.deleteServerCopies(sounds - sameBytes(sounds, others, clips::file))
            }
            images.delete(pictures)
            clips.delete(sounds)
        }
        file(id).delete()
        File(agents, id).deleteRecursively()
    }

    /**
     * The chat images and clips no saved chat holds, an hour old or more: attached and never sent (the app closed
     * with them, a video's reading cut off). Nothing goes when a chat cannot be read - what it holds would look free.
     */
    fun dropUnheldMedia() {
        val held = heldMedia() ?: return
        images.dropUnheld(held, UNHELD_AGE_MS)
        clips.dropUnheld(held, UNHELD_AGE_MS)
    }

    /** The media files every saved chat but [except] holds (images, clips, videos' frames and sound); null when one cannot be read. */
    private fun heldMedia(except: String? = null): Set<String>? {
        val held = HashSet<String>()
        for (f in dir.listFiles() ?: return null) {
            if (!f.isFile || !f.name.endsWith(".json")) continue
            val id = f.name.removeSuffix(".json")
            if (id == except) continue
            val chat = load(id) ?: return null
            for (m in chat.messages) {
                held += m.images
                held += m.audio
                for (v in m.videos) {
                    held += v.frames
                    v.audio?.let { held += it }
                }
            }
        }
        return held
    }

    /** Those of [mine] whose bytes one of [others] has too ([file]: where a name's file is). */
    private fun sameBytes(mine: List<String>, others: Set<String>, file: (String) -> File): Set<String> {
        val bySize = others.map(file).filter { it.isFile }.groupBy { it.length() }
        return mine.filterTo(HashSet()) { n ->
            val f = file(n)
            val candidates = bySize[f.length()] ?: return@filterTo false
            val bytes = runCatching { f.readBytes() }.getOrNull() ?: return@filterTo false
            candidates.any { o -> runCatching { o.readBytes().contentEquals(bytes) }.getOrDefault(false) }
        }
    }

    /**
     * A copy of the chat to go on from apart: its messages, its images, and its agent's Linux as it is (the files,
     * what the agent installed, its session, which the copy goes on with). The new chat's id.
     */
    fun fork(id: String, suffix: String): String? {
        val c = load(id) ?: return null
        val copy = System.currentTimeMillis().toString()
        val msgs = c.messages.map { m ->
            m.copy(images = m.images.map { images.copy(it) ?: it }, audio = m.audio.map { clips.copy(it) ?: it },
                videos = m.videos.map { v ->
                    v.copy(frames = v.frames.map { images.copy(it) ?: it }, audio = v.audio?.let { clips.copy(it) ?: it })
                })
        }
        save(copy, (c.title + suffix).take(80), copy.toLong(), msgs, c.stats, c.agent, c.agentSession)
        val from = File(agents, id)
        if (from.isDirectory) copyTree(from, File(agents, copy), from.path, File(agents, copy).path)
        return copy
    }

    /**
     * A folder copied whole, its symbolic links as links (a venv's, node_modules' .bin). A link to an absolute path
     * in the old chat's folder (proot makes such links for the hard links Android does not allow) points into the
     * new chat's instead.
     */
    private fun copyTree(from: File, to: File, oldRoot: String, newRoot: String) {
        to.mkdirs()
        for (f in from.listFiles() ?: emptyArray()) {
            val t = File(to, f.name)
            val st = runCatching { android.system.Os.lstat(f.path) }.getOrNull() ?: continue
            when {
                android.system.OsConstants.S_ISLNK(st.st_mode) -> runCatching {
                    val target = android.system.Os.readlink(f.path)
                    android.system.Os.symlink(if (target.startsWith("$oldRoot/")) newRoot + target.removePrefix(oldRoot) else target, t.path)
                }
                f.isDirectory -> copyTree(f, t, oldRoot, newRoot)
                else -> {
                    f.copyTo(t, overwrite = true)
                    runCatching { android.system.Os.chmod(t.path, st.st_mode and 0x1ff) }
                }
            }
        }
    }

    private companion object {
        /** How old a chat image or clip no chat holds is before it goes ([dropUnheldMedia]): one being attached is newer. */
        const val UNHELD_AGE_MS = 3600_000L
    }
}
