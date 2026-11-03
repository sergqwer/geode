package xyz.tifu.geode.media

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.roundToInt

/**
 * WAV in and out, for the sound the model hears: a RIFF/WAVE file of PCM (8, 16, 24, 32-bit integers) or IEEE
 * floats (32, 64-bit), any rate and channels, read as 16 kHz mono floats ([decode]: the channels averaged, the rate
 * converted); and 16 kHz mono PCM16 written ([encode]: a recording sent on as a file).
 */
object Wav {
    class FormatException(message: String) : IllegalArgumentException(message)

    /** A WAV's samples as 16 kHz mono in [-1, 1] (about its first 30 s at most: AudioStore.MAX_SAMPLES). */
    fun decode(bytes: ByteArray): FloatArray {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        if (bytes.size < 12 || tag(b, 0) != "RIFF" || tag(b, 8) != "WAVE") throw FormatException("not a WAV file")
        var at = 12
        var format = -1
        var channels = 0
        var rate = 0
        var bits = 0
        var data = -1
        var dataLen = 0
        while (at + 8 <= bytes.size) {
            val id = tag(b, at)
            val len = b.getInt(at + 4)
            if (len < 0) throw FormatException("a bad chunk")
            val body = at + 8
            when (id) {
                "fmt " -> {
                    if (len < 16 || body + 16 > bytes.size) throw FormatException("a short fmt chunk")
                    format = b.getShort(body).toInt() and 0xffff
                    channels = b.getShort(body + 2).toInt() and 0xffff
                    rate = b.getInt(body + 4)
                    bits = b.getShort(body + 14).toInt() and 0xffff
                    // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes are the format
                    if (format == 0xFFFE && len >= 26 && body + 26 <= bytes.size)
                        format = b.getShort(body + 24).toInt() and 0xffff
                }
                "data" -> {
                    data = body
                    // a streamed WAV may say 0 or more than there is: the rest of the file
                    dataLen = if (len == 0 || body.toLong() + len > bytes.size) bytes.size - body else len
                }
            }
            if (data >= 0 && format >= 0) break
            at = body + len + (len and 1)
        }
        if (format < 0 || data < 0) throw FormatException("no fmt or data chunk")
        if (channels <= 0 || rate <= 0) throw FormatException("a bad format")
        val width = when {
            format == 1 && bits in setOf(8, 16, 24, 32) -> bits / 8
            format == 3 && bits in setOf(32, 64) -> bits / 8
            else -> throw FormatException("WAV format $format with $bits bits is not supported")
        }
        // what the model hears and no more (its first 30 s, a second over for the rate's conversion): a long file is
        // not read whole
        val frames = minOf(dataLen.toLong() / (width * channels),
            AudioStore.MAX_SAMPLES.toLong() * rate / AudioStore.RATE + rate).toInt()
        val mono = FloatArray(frames)
        for (f in 0 until frames) {
            var s = 0.0
            for (c in 0 until channels) {
                val o = data + (f * channels + c) * width
                s += when {
                    format == 3 && width == 4 -> b.getFloat(o).toDouble()
                    format == 3 -> b.getDouble(o)
                    width == 1 -> ((bytes[o].toInt() and 0xff) - 128) / 128.0
                    width == 2 -> b.getShort(o) / 32768.0
                    width == 3 -> ((bytes[o].toInt() and 0xff) or ((bytes[o + 1].toInt() and 0xff) shl 8) or
                        (bytes[o + 2].toInt() shl 16)) / 8388608.0
                    else -> b.getInt(o) / 2147483648.0
                }
            }
            mono[f] = (s / channels).toFloat().coerceIn(-1f, 1f)
        }
        return resample(mono, rate, AudioStore.RATE)
    }

    /**
     * [x] at [from] Hz as [to] Hz: linear interpolation, after a moving average about one output sample long when
     * the rate comes down (enough against aliasing for speech at 16 kHz).
     */
    fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || x.isEmpty()) return x
        var src = x
        if (from > to) {
            val k = maxOf(1, (from.toDouble() / to).roundToInt())
            if (k > 1) {
                val y = FloatArray(x.size)
                var acc = 0.0
                for (i in x.indices) {
                    acc += x[i]
                    if (i >= k) acc -= x[i - k]
                    y[i] = (acc / minOf(i + 1, k)).toFloat()
                }
                // the average lags (k - 1) / 2 samples: read it that far ahead
                val lag = (k - 1) / 2
                src = FloatArray(x.size) { y[minOf(x.size - 1, it + lag)] }
            }
        }
        val n = maxOf(1, (src.size.toLong() * to / from).toInt())
        val out = FloatArray(n)
        val step = from.toDouble() / to
        for (i in 0 until n) {
            val p = i * step
            val j = floor(p).toInt()
            val t = (p - j).toFloat()
            val a = src[minOf(j, src.size - 1)]
            val c = src[minOf(j + 1, src.size - 1)]
            out[i] = a + (c - a) * t
        }
        return out
    }

    /** 16 kHz mono samples as a PCM16 WAV file. */
    fun encode(pcm: FloatArray, rate: Int = AudioStore.RATE): ByteArray {
        val data = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (s in pcm) data.putShort((s.coerceIn(-1f, 1f) * 32767f).roundToInt().toShort())
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt(36 + pcm.size * 2).put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2)
            .putShort(16)
        h.put("data".toByteArray()).putInt(pcm.size * 2)
        return ByteArrayOutputStream().apply {
            write(h.array())
            write(data.array())
        }.toByteArray()
    }

    private fun tag(b: ByteBuffer, at: Int): String =
        String(ByteArray(4) { b.get(at + it) }, Charsets.US_ASCII)
}
