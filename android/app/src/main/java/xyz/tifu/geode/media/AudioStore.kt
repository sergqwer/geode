package xyz.tifu.geode.media

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/**
 * The chats' sound clips as the model hears them: 16 kHz mono float32 samples in [-1, 1], little-endian, at most
 * [MAX_SAMPLES] (files/chats/audio/<id>.f32). They stay on the device, with their chat: written by the UI process,
 * read by the :engine one (geode_add_audio).
 */
class AudioStore(context: Context) {
    private val dir = File(context.filesDir, "chats/audio").apply { mkdirs() }

    fun file(name: String) = File(dir, name)

    /** Samples (16 kHz mono) into the store, the first [MAX_SAMPLES] of them; its name. */
    fun save(pcm: FloatArray): String {
        val name = UUID.randomUUID().toString() + ".f32"
        write(pcm, name)
        return name
    }

    /**
     * A clip a local server's request sent into the store, named by its content (a client sends a conversation's
     * clips again with every request); its name. Older than a day: gone at the server's start ([dropServerClips]).
     */
    fun saveServer(pcm: FloatArray): String {
        val name = serverName(pcm)
        if (file(name).isFile) file(name).setLastModified(System.currentTimeMillis()) else write(pcm, name)
        return name
    }

    /** The name [saveServer] gives these samples: their content's (the first 30 s). */
    private fun serverName(pcm: FloatArray): String {
        val d = java.security.MessageDigest.getInstance("SHA-256")
        val b = ByteBuffer.allocate(4 * minOf(pcm.size, MAX_SAMPLES)).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until minOf(pcm.size, MAX_SAMPLES)) b.putFloat(pcm[i])
        return SERVER_PREFIX + d.digest(b.array()).take(16).joinToString("") { "%02x".format(it) } + ".f32"
    }

    /**
     * The local server's copies of these chat clips, gone with their chat. The agent sends a clip as a PCM16 WAV
     * (Agent.kt: Wav.encode) and the server stores what it reads back from it (Wav.decode): the same two steps give
     * the copy's name.
     */
    fun deleteServerCopies(names: Collection<String>) {
        for (n in names) runCatching { file(serverName(Wav.decode(Wav.encode(read(file(n)))))).delete() }
    }

    /** The clips no chat holds ([held]), older than [ageMs]; the server's have their own rule ([dropServerClips]). */
    fun dropUnheld(held: Set<String>, ageMs: Long) {
        val old = System.currentTimeMillis() - ageMs
        (dir.listFiles() ?: emptyArray())
            .filter { it.isFile && !it.name.startsWith(SERVER_PREFIX) && it.name !in held && it.lastModified() < old }
            .forEach { it.delete() }
    }

    private fun write(pcm: FloatArray, name: String) {
        val n = minOf(pcm.size, MAX_SAMPLES)
        val b = ByteBuffer.allocate(4 * n).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until n) b.putFloat(pcm[i])
        val tmp = File(dir, "$name.tmp")
        FileOutputStream(tmp).use { it.write(b.array()) }
        tmp.renameTo(file(name))
    }

    /** The clip's samples; 0 when it is gone. */
    fun samples(name: String): Int = (file(name).length() / 4).toInt()

    /** The local server's clips not used for a day. */
    fun dropServerClips() {
        val old = System.currentTimeMillis() - 24 * 3600_000L
        (dir.listFiles() ?: emptyArray()).filter { it.name.startsWith(SERVER_PREFIX) && it.lastModified() < old }
            .forEach { it.delete() }
    }

    fun delete(names: Collection<String>) {
        for (n in names) file(n).delete()
    }

    /** A copy of a clip under a new name (a forked chat's own); null when it is gone. */
    fun copy(name: String): String? {
        val src = file(name).takeIf { it.isFile } ?: return null
        val copy = UUID.randomUUID().toString() + ".f32"
        src.copyTo(file(copy))
        return copy
    }

    /** A clip's length in seconds. */
    fun seconds(name: String): Float = samples(name).toFloat() / RATE

    companion object {
        /** Gemma 4's audio encoder: 16 kHz, 20 ms windows every 10 ms, at most 30 s. */
        const val RATE = 16000
        const val MAX_SAMPLES = 30 * RATE
        private const val FRAME = 320
        private const val HOP = 160
        private const val SERVER_PREFIX = "srv_"

        /**
         * The <|audio|> tokens a clip of [n] samples takes (geode_audio_tokens, the engine's Audio::tokens): its
         * frames, two stride-2 convolutions, the frames whose window ends inside the clip - one per 40 ms, at most
         * 750; 0: too short a clip.
         */
        fun tokens(n0: Long): Int {
            if (n0 <= 0) return 0
            val n = minOf(n0, MAX_SAMPLES.toLong())
            val padded = (n + 127) / 128 * 128
            val len = HOP + padded
            if (len < FRAME + 1) return 0
            val frames = ((len - (FRAME + 1)) / HOP + 1).toInt()
            fun convOut(x: Int) = (x + 2 - 3) / 2 + 1
            val h1 = convOut(convOut(frames))
            var t = 0
            for (r in 0 until h1) if (4L * r * HOP + FRAME < HOP + n) t++
            return t
        }

        /** A clip's samples from its file (float32 little-endian). */
        fun read(f: File): FloatArray {
            val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(b.remaining() / 4) { b.float }
        }
    }
}
