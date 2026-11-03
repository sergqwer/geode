package xyz.tifu.geode.media

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.File
import java.nio.ByteOrder

/**
 * Any sound Android decodes - a sound file's or a video's first audio track (MP3, AAC/M4A, Ogg/Opus/Vorbis, FLAC,
 * AMR, WAV) - as 16 kHz mono floats, its first [AudioStore.MAX_SAMPLES] at most: MediaExtractor's samples through
 * MediaCodec's decoder, the channels averaged, the rate converted (Wav.resample).
 */
object AudioDecoder {
    class NoAudioException : IllegalArgumentException("there is no sound in it")

    fun decode(context: Context, uri: Uri, from: Long = 0): FloatArray =
        withExtractor({ setDataSource(context, uri, null) }, from)

    fun decode(file: File, from: Long = 0): FloatArray = withExtractor({ setDataSource(file.path) }, from)

    /** A request's bytes (a format WAV is not) through a temporary file: MediaExtractor reads files, not arrays. */
    fun decode(bytes: ByteArray, tmpDir: File): FloatArray {
        val f = File.createTempFile("clip", null, tmpDir)
        try {
            f.writeBytes(bytes)
            return decode(f)
        } finally {
            f.delete()
        }
    }

    private fun withExtractor(open: MediaExtractor.() -> Unit, fromUs: Long): FloatArray {
        val ex = MediaExtractor()
        try {
            ex.open()
            return decode(ex, fromUs)
        } finally {
            ex.release()
        }
    }

    private fun decode(ex: MediaExtractor, fromUs: Long): FloatArray {
        val track = (0 until ex.trackCount).firstOrNull {
            ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: throw NoAudioException()
        val format = ex.getTrackFormat(track)
        ex.selectTrack(track)
        if (fromUs > 0) ex.seekTo(fromUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
        var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var encoding = AudioFormat.ENCODING_PCM_16BIT
        // the source's mono samples: the clip's length at the source's rate, a second over for the resampling
        var limit = (AudioStore.MAX_SAMPLES.toLong() * rate / AudioStore.RATE + rate).toInt()
        var out = FloatArray(minOf(limit, rate * 4))
        var n = 0
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        try {
            codec.configure(format, null, null, 0)
            codec.start()
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            // a decoder that takes and gives nothing for a while (stuck, or done without saying so): given up on
            var idle = 0
            while (n < limit && idle < IDLE_POLLS) {
                var progressed = false
                if (!inputDone) {
                    val i = codec.dequeueInputBuffer(POLL_US)
                    if (i >= 0) {
                        progressed = true
                        val size = ex.readSampleData(codec.getInputBuffer(i)!!, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(i, 0, size, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val o = codec.dequeueOutputBuffer(info, POLL_US)
                if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    progressed = true
                    val f = codec.outputFormat
                    rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
                    limit = (AudioStore.MAX_SAMPLES.toLong() * rate / AudioStore.RATE + rate).toInt()
                } else if (o >= 0) {
                    progressed = true
                    // a frame before the start a seek asked for (it lands on a sync frame before it): left out
                    if (info.size > 0 && info.presentationTimeUs >= fromUs) {
                        val buf = codec.getOutputBuffer(o)!!.order(ByteOrder.LITTLE_ENDIAN)
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        val width = when (encoding) {
                            AudioFormat.ENCODING_PCM_FLOAT, AudioFormat.ENCODING_PCM_32BIT -> 4
                            AudioFormat.ENCODING_PCM_8BIT -> 1
                            else -> 2
                        }
                        val frames = info.size / (width * channels)
                        if (n + frames > out.size) out = out.copyOf(maxOf(out.size * 2, n + frames))
                        for (f in 0 until frames) {
                            var s = 0f
                            for (c in 0 until channels) s += when (encoding) {
                                AudioFormat.ENCODING_PCM_FLOAT -> buf.float
                                AudioFormat.ENCODING_PCM_32BIT -> buf.int / 2147483648f
                                AudioFormat.ENCODING_PCM_8BIT -> ((buf.get().toInt() and 0xff) - 128) / 128f
                                else -> buf.short / 32768f
                            }
                            out[n++] = (s / channels).coerceIn(-1f, 1f)
                        }
                    }
                    codec.releaseOutputBuffer(o, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
                idle = if (progressed) 0 else idle + 1
            }
        } finally {
            runCatching { codec.stop() }
            codec.release()
        }
        if (n == 0) throw NoAudioException()
        val pcm = Wav.resample(out.copyOf(n), rate, AudioStore.RATE)
        return if (pcm.size > AudioStore.MAX_SAMPLES) pcm.copyOf(AudioStore.MAX_SAMPLES) else pcm
    }

    private const val POLL_US = 10_000L
    private const val IDLE_POLLS = 300
}
