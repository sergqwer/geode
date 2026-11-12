package xyz.tifu.geode.music

import xyz.tifu.geode.util.writeAtomic
import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** What became of a song's job. */
enum class SongStatus { WAITING, DONE, FAILED, CANCELLED }

/**
 * A song of the library: what the user asked for ([title], [style], [lyrics], the settings it was made with in
 * [job]) and, once made, the engine's [result] and the audio file.
 */
data class Song(
    val id: String,
    val dir: File,
    val title: String,
    val style: String,
    val lyrics: String,
    val created: Long,
    /** The job MusicService runs (MusicJob's JSON): the settings it was made with, the seed. */
    val job: JSONObject,
    /** The engine's answer (yue2::song_result_json) with "finished"; null while it is still to be made. */
    val result: JSONObject?,
) {
    val status: SongStatus get() = when {
        result == null -> SongStatus.WAITING
        result.optBoolean("ok") -> SongStatus.DONE
        result.optBoolean("cancelled") -> SongStatus.CANCELLED
        else -> SongStatus.FAILED
    }
    val audio: File? get() = result?.optString("output")?.takeIf { it.isNotEmpty() }?.let { File(it) }?.takeIf { it.isFile }
    val seconds: Double get() = result?.optDouble("seconds", 0.0) ?: 0.0
    val error: String get() = result?.optString("error").orEmpty()
    val seed: Long get() = job.optLong("seed")
    val mode: YuE2Prompt.Mode get() = YuE2Prompt.Mode.of(job.optString("mode"))
    /** The score's text ids the song was sung from (to sing the same melody again); empty: none (mode off). */
    val abc: IntArray get() = result?.optJSONArray("abc")?.let { a -> IntArray(a.length()) { a.getInt(it) } } ?: IntArray(0)
    /** A plan: its score alone, written to be heard (as MIDI) and sung if liked - no audio (MusicJob.create's
     *  scoreOnly). */
    val plan: Boolean get() = job.optBoolean("score_only")
    /** Seconds the whole song took to make. */
    val makeSeconds: Double get() = result?.optJSONObject("times")?.optDouble("all", 0.0) ?: 0.0
    /** An edit of another song (MusicJob.cut): its op, the song it was made from, its frames; null: sung. */
    val edit: JSONObject? get() = job.optJSONObject("edit")
    /** Made with its editable state beside its audio (the engine's state_out): the track editor can work on it. */
    val editable: Boolean get() = status == SongStatus.DONE && !plan && audio != null && File(dir, MusicJob.STATE_FILE).isFile
    /** The songs still to be made from this one (its edits waiting): they need its files until then. */
    fun waitingEdits(all: List<Song>): List<Song> = all.filter { it.result == null && it.edit?.optString("parent") == id }
    /** Its waveform and pitch classes (the engine's view_out), when it has them. */
    val view: File? get() = File(dir, MusicJob.VIEW_FILE).takeIf { it.isFile }
}

/**
 * The library: files/music/songs/<id>/ - song.json (the request, written by the app), result.json (the engine's answer,
 * written by MusicService in the :music process when the song is done), song.flac or song.wav. The files are the
 * truth both processes share: neither keeps a list of its own - a song read is kept only until one of its two files
 * changes (their times and sizes: [stamp]), and the screen and the web page list the library every couple of seconds.
 */
class SongStore(
    val root: File,
    /** Where a share puts a song's audio under a readable name (its link's file). */
    private val shared: File,
) {
    constructor(context: Context) : this(File(context.filesDir, "music/songs"), File(context.cacheDir, "shared"))

    init {
        root.mkdirs()
    }

    fun dir(id: String) = File(root, id)

    fun list(): List<Song> {
        val dirs = (root.listFiles() ?: emptyArray()).filter { it.isDirectory }
        val paths = dirs.mapTo(HashSet()) { it.path }
        read.keys.removeIf { File(it).parentFile == root && it !in paths }   // (a song deleted, by either process)
        return dirs.mapNotNull { read(it) }.sortedByDescending { it.created }
    }

    fun find(id: String): Song? = dir(id).takeIf { it.isDirectory }?.let { read(it) }

    /** A new song's folder and its request; the audio's name follows the job's format. */
    fun create(title: String, style: String, lyrics: String, job: JSONObject): Song {
        val id = newId()
        val d = dir(id)
        val created = System.currentTimeMillis()
        job.put("output", File(d, "song." + job.optString("format", "flac")).path)
        val o = JSONObject().put("id", id).put("title", title).put("style", style).put("lyrics", lyrics)
            .put("created", created).put("job", job)
        writeAtomic(File(d, SONG), o.toString())
        return Song(id, d, title, style, lyrics, created, job, null)
    }

    /**
     * [n] takes of one song ([job]: take i's job - its own seed), titled "title i/n" when several. The takes made
     * before a failure (the storage full, the folder not writable) are kept and returned with it.
     */
    fun createTakes(title: String, style: String, lyrics: String, n: Int, job: (Int) -> JSONObject): Pair<List<Song>, Throwable?> {
        val made = ArrayList<Song>()
        val failed = runCatching {
            for (i in 0 until n) made += create(if (n > 1) "$title ${i + 1}/$n" else title, style, lyrics, job(i))
        }.exceptionOrNull()
        return made to failed
    }

    /** The song's result: cancelled. */
    fun cancelled(id: String) = finish(id, JSONObject().put("ok", false).put("cancelled", true))

    fun rename(id: String, title: String) {
        val f = File(dir(id), SONG)
        val o = runCatching { JSONObject(f.readText()) }.getOrNull() ?: return
        writeAtomic(f, o.put("title", title).toString())
    }

    /** The engine's answer for the song (MusicService). */
    fun finish(id: String, result: JSONObject) {
        val d = dir(id)
        if (!d.isDirectory) return
        writeAtomic(File(d, RESULT), result.put("finished", System.currentTimeMillis()).toString())
    }

    /** The song gone: its folder, and the copy a share made of its audio. */
    fun delete(id: String) {
        dir(id).deleteRecursively()
        File(shared, id).deleteRecursively()
        read.remove(dir(id).path)
    }

    /** The song in [d]: as read before while its files are as they were, else read now. */
    private fun read(d: File): Song? {
        val st = stamp(d)
        read[d.path]?.takeIf { it.stamp == st }?.let { return it.song }
        return parse(d).also { read[d.path] = Read(st, it) }
    }

    /** Its two files' modification times and sizes: a rewrite (an atomic rename) changes them. */
    private fun stamp(d: File): String {
        val a = File(d, SONG)
        val b = File(d, RESULT)
        return "${a.lastModified()}:${a.length()}:${b.lastModified()}:${b.length()}"
    }

    private fun parse(d: File): Song? = runCatching {
        val o = JSONObject(File(d, SONG).readText())
        val r = File(d, RESULT).takeIf { it.isFile }?.let { runCatching { JSONObject(it.readText()) }.getOrNull() }
        Song(o.getString("id"), d, o.optString("title"), o.optString("style"), o.optString("lyrics"),
            o.optLong("created"), o.optJSONObject("job") ?: JSONObject(), r)
    }.getOrNull()

    /** A new id, its folder made here: mkdir is the reservation (two songs asked for in the same second get two). */
    private fun newId(): String {
        val t = java.text.SimpleDateFormat("yyyyMMdd-HHmmss", java.util.Locale.ROOT).format(java.util.Date())
        root.mkdirs()
        var id = t
        var n = 1
        while (!dir(id).mkdir()) {
            if (++n > 1000) throw java.io.IOException("no folder for a new song in $root")
            id = "$t-$n"
        }
        return id
    }

    /** A song as read ([song]; null: not one), at its files' [stamp]. */
    private class Read(val stamp: String, val song: Song?)

    companion object {
        const val SONG = "song.json"
        const val RESULT = "result.json"

        /** The songs read in this process, by folder (every store of it shares them: the screen's, the server's). */
        private val read = java.util.concurrent.ConcurrentHashMap<String, Read>()

        fun ints(a: IntArray): JSONArray = JSONArray().also { j -> a.forEach { j.put(it) } }
    }
}
