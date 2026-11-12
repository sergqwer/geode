package xyz.tifu.geode.media

import android.content.Context
import java.io.File
import xyz.tifu.geode.util.writeAtomic
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/**
 * The chats' sound clips as the model hears them: 16 kHz mono float32 samples in [-1, 1], little-endian, at most
 * [MAX_SAMPLES] (files/chats/audio/<id>.f32). They stay on the device, with their chat: written by the UI process,
 * read by the :engine one (geode_add_audio).
 */
class AudioStore(context: Context) : MediaFolder(File(context.filesDir, "chats/audio"), "f32") {
    /** Samples (16 kHz mono) into the store, the first [MAX_SAMPLES] of them; its name. */
    fun save(pcm: FloatArray): String {
        val name = UUID.randomUUID().toString() + ".f32"
        writeAtomic(file(name), sync = false) { it.write(encode(pcm)) }
        return name
    }

    /**
     * A clip a local server's request sent into the store, named by its content (a client sends a conversation's
     * clips again with every request); its name. Older than a day: gone at the server's start ([dropServerClips]).
     */
    fun saveServer(pcm: FloatArray): String {
        val bytes = encode(pcm)   // (once: its name is these bytes' hash, and they are what is written)
        return serverCopy(contentName(bytes, "f32")) { name -> writeAtomic(file(name), sync = false) { it.write(bytes) } }
    }

    /** The first [MAX_SAMPLES] samples as the file holds them: float32, little-endian. */
    private fun encode(pcm: FloatArray): ByteArray {
        val n = minOf(pcm.size, MAX_SAMPLES)
        val b = ByteBuffer.allocate(4 * n).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until n) b.putFloat(pcm[i])
        return b.array()
    }

    /**
     * The local server's copies of these chat clips, gone with their chat. The agent sends a clip as a PCM16 WAV
     * (Agent.kt: Wav.encode) and the server stores what it reads back from it (Wav.decode): the same two steps give
     * the copy's name.
     */
    fun deleteServerCopies(names: Collection<String>) {
        for (n in names) runCatching { file(contentName(encode(Wav.decode(Wav.encode(read(file(n))))), "f32")).delete() }
    }


    /** The clip's samples; 0 when it is gone. */
    fun samples(name: String): Int = (file(name).length() / 4).toInt()


    /** A clip's length in seconds. */
    fun seconds(name: String): Float = samples(name).toFloat() / RATE

    companion object {
        /** Gemma 4's audio encoder: 16 kHz, 20 ms windows every 10 ms, at most 30 s. */
        const val RATE = 16000
        const val MAX_SAMPLES = 30 * RATE
        private const val FRAME = 320
        private const val HOP = 160

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
