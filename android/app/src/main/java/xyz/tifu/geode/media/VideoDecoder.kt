package xyz.tifu.geode.media

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import java.io.File
import kotlin.math.ceil

/**
 * A video as Gemma 4 watches it: its frames, about one a second - at most [MAX_FRAMES], spread over the whole video
 * when it is longer (Gemma4VideoProcessor's num_frames) - each with the second it is at, and its sound (the first
 * audio track, its first 30 s: AudioDecoder). The frames come out already turned upright (MediaMetadataRetriever
 * applies the track's rotation), at most [FRAME_SIDE] pixels a side: the model reads a frame at 70 tokens.
 */
object VideoDecoder {
    const val MAX_FRAMES = 32
    /** A frame's longer side at most, as it is kept (the model's 70 tokens are a few hundred pixels a side). */
    const val FRAME_SIDE = 768

    class NotAVideoException : IllegalArgumentException("it is not a video")

    class Frame(val bitmap: Bitmap, val second: Int)

    /** [audio]: null when the video has no sound. */
    class Video(val frames: List<Frame>, val audio: FloatArray?, val durationMs: Long)

    fun decode(context: Context, uri: Uri): Video =
        read({ setDataSource(context, uri) }) { runCatching { AudioDecoder.decode(context, uri) }.getOrNull() }

    fun decode(file: File): Video =
        read({ setDataSource(file.path) }) { runCatching { AudioDecoder.decode(file) }.getOrNull() }

    /** The seconds (ms) the frames of a video [ms] long are taken at: one a second, or [MAX_FRAMES] spread evenly. */
    fun frameTimes(ms: Long): List<Long> {
        val n = ceil(maxOf(ms, 1L) / 1000.0).toInt().coerceIn(1, MAX_FRAMES)
        return if (ms <= MAX_FRAMES * 1000L) List(n) { it * 1000L } else List(n) { it * ms / n }
    }

    private fun read(open: MediaMetadataRetriever.() -> Unit, audio: () -> FloatArray?): Video {
        val r = MediaMetadataRetriever()
        val frames = ArrayList<Frame>()
        try {
            r.open()
            if (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO) != "yes") throw NotAVideoException()
            val ms = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            for (t in frameTimes(ms)) {
                // the exact frame at that time (not the key frame before it: at one a second those repeat); scaled to
                // fit the box, its shape kept
                val bmp = r.getScaledFrameAtTime(t * 1000, MediaMetadataRetriever.OPTION_CLOSEST, FRAME_SIDE, FRAME_SIDE)
                    ?: continue
                frames += Frame(bmp, (t / 1000).toInt())
            }
            if (frames.isEmpty()) throw NotAVideoException()
            return Video(frames, audio(), ms)
        } catch (e: Exception) {
            for (f in frames) f.bitmap.recycle()
            throw e
        } finally {
            r.release()
        }
    }
}
