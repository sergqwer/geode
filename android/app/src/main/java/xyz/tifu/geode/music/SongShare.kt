package xyz.tifu.geode.music

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A song for a messenger (Viber, Telegram, WhatsApp): its audio - a 24-bit FLAC of tens of megabytes, which a
 * messenger sends as a document and an iPhone does not play - as AAC in an .m4a, a few megabytes that play in the chat
 * itself. 256 kbit/s stereo: no difference a listener hears (the lossless original is still a share of its own).
 * The phone's own codecs (MediaCodec): the FLAC or WAV decoded to 16-bit PCM, encoded, muxed into MP4.
 */
object SongShare {
    const val BITRATE = 256_000
    private const val WAIT_US = 5_000L
    /** Decoded PCM chunks held for the encoder, at most (the decoder waits beyond). */
    private const val MAX_PENDING = 32

    /** [src] (FLAC or WAV) as AAC-LC in an MP4 file [out] (written whole or not at all); [cancelled] checked as it goes. */
    fun toM4a(src: File, out: File, cancelled: () -> Boolean = { false }) {
        val tmp = File(out.parentFile, ".${out.name}.tmp")
        tmp.delete()
        val ex = MediaExtractor()
        var dec: MediaCodec? = null
        var enc: MediaCodec? = null
        var mux: MediaMuxer? = null
        var muxing = false
        try {
            ex.setDataSource(src.path)
            val track = (0 until ex.trackCount).firstOrNull {
                ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: throw IOException("no audio track in ${src.name}")
            ex.selectTrack(track)
            val inFormat = ex.getTrackFormat(track)
            val rate = inFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = inFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            if (channels !in 1..2) throw IOException("$channels channels")
            dec = MediaCodec.createDecoderByType(inFormat.getString(MediaFormat.KEY_MIME)!!).apply {
                configure(inFormat, null, null, 0)
                start()
            }
            enc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC).apply {
                configure(MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, rate, channels).apply {
                    setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
                    setInteger(MediaFormat.KEY_BIT_RATE, BITRATE)
                    setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1 shl 16)
                }, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                start()
            }
            mux = MediaMuxer(tmp.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val frameBytes = 2 * channels
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            val pending = ArrayDeque<ByteArray>()
            var headAt = 0
            var frames = 0L
            var muxTrack = -1
            var readAll = false
            var decoded = false
            var fedAll = false
            var encoded = false
            val info = MediaCodec.BufferInfo()
            while (!encoded) {
                if (cancelled()) throw kotlinx.coroutines.CancellationException("share cancelled")
                // the file into the decoder
                if (!readAll && pending.size < MAX_PENDING) {
                    val i = dec.dequeueInputBuffer(WAIT_US)
                    if (i >= 0) {
                        val n = ex.readSampleData(dec.getInputBuffer(i)!!, 0)
                        if (n < 0) {
                            dec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            readAll = true
                        } else {
                            dec.queueInputBuffer(i, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                // the decoder's PCM, as 16-bit
                if (!decoded && pending.size < MAX_PENDING) {
                    val o = dec.dequeueOutputBuffer(info, WAIT_US)
                    if (o >= 0) {
                        if (info.size > 0) pending.addLast(pcm16(dec.getOutputBuffer(o)!!, info, encoding))
                        dec.releaseOutputBuffer(o, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) decoded = true
                    } else if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val f = dec.outputFormat
                        if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        if (f.containsKey(MediaFormat.KEY_CHANNEL_COUNT) && f.getInteger(MediaFormat.KEY_CHANNEL_COUNT) != channels)
                            throw IOException("the decoder changed the channels")
                    }
                }
                // the PCM into the encoder, whole frames, timed by the frames before
                if (!fedAll && (pending.isNotEmpty() || decoded)) {
                    val i = enc.dequeueInputBuffer(WAIT_US)
                    if (i >= 0) {
                        val buf = enc.getInputBuffer(i)!!
                        buf.clear()
                        var written = 0
                        while (pending.isNotEmpty()) {
                            val room = buf.remaining() - buf.remaining() % frameBytes
                            if (room <= 0) break
                            val head = pending.first()
                            val n = minOf(head.size - headAt, room)
                            buf.put(head, headAt, n)
                            headAt += n
                            written += n
                            if (headAt == head.size) {
                                pending.removeFirst()
                                headAt = 0
                            }
                        }
                        val pts = frames * 1_000_000L / rate
                        frames += written / frameBytes
                        val last = decoded && pending.isEmpty()
                        enc.queueInputBuffer(i, 0, written, pts, if (last) MediaCodec.BUFFER_FLAG_END_OF_STREAM else 0)
                        if (last) fedAll = true
                    }
                }
                // the encoder's AAC into the file
                while (true) {
                    val o = enc.dequeueOutputBuffer(info, 0)
                    if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        muxTrack = mux.addTrack(enc.outputFormat)
                        mux.start()
                        muxing = true
                        continue
                    }
                    if (o < 0) break
                    val buf = enc.getOutputBuffer(o)!!
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0) {
                        if (!muxing) throw IOException("the encoder gave no format")
                        buf.position(info.offset).limit(info.offset + info.size)
                        mux.writeSampleData(muxTrack, buf, info)
                    }
                    enc.releaseOutputBuffer(o, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        encoded = true
                        break
                    }
                }
            }
            if (!muxing) throw IOException("no audio was encoded")
            mux.stop()
            muxing = false
            mux.release()
            mux = null
            if (!tmp.renameTo(out)) throw IOException("could not put ${out.name} in place")
        } finally {
            runCatching { dec?.stop() }
            runCatching { dec?.release() }
            runCatching { enc?.stop() }
            runCatching { enc?.release() }
            if (muxing) runCatching { mux?.stop() }
            runCatching { mux?.release() }
            ex.release()
            tmp.delete()
        }
    }

    /** A decoder's output as 16-bit little-endian PCM, whatever its own sample format. */
    private fun pcm16(buf: ByteBuffer, info: MediaCodec.BufferInfo, encoding: Int): ByteArray {
        val b = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        b.position(info.offset).limit(info.offset + info.size)
        return when (encoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> {
                val n = info.size / 4
                ByteArray(n * 2).also { o ->
                    for (k in 0 until n) put16(o, k, (b.getFloat().coerceIn(-1f, 1f) * 32767f).toInt())
                }
            }
            AudioFormat.ENCODING_PCM_32BIT -> {
                val n = info.size / 4
                ByteArray(n * 2).also { o -> for (k in 0 until n) put16(o, k, b.getInt() shr 16) }
            }
            AudioFormat.ENCODING_PCM_24BIT_PACKED -> {
                val n = info.size / 3
                ByteArray(n * 2).also { o ->
                    for (k in 0 until n) {
                        b.get()   // (the lowest byte: below 16 bits)
                        val lo = b.get().toInt() and 0xff
                        val hi = b.get().toInt()
                        put16(o, k, (hi shl 8) or lo)
                    }
                }
            }
            AudioFormat.ENCODING_PCM_8BIT -> ByteArray(info.size * 2).also { o ->
                for (k in 0 until info.size) put16(o, k, ((b.get().toInt() and 0xff) - 128) shl 8)
            }
            else -> ByteArray(info.size).also { b.get(it) }
        }
    }

    private fun put16(o: ByteArray, k: Int, v: Int) {
        o[2 * k] = v.toByte()
        o[2 * k + 1] = (v shr 8).toByte()
    }
}
