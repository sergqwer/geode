package xyz.tifu.geode.media

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class WavTest {
    private fun wav(format: Int, bits: Int, channels: Int, rate: Int, frames: Int, sample: (Int, Int) -> Double): ByteArray {
        val w = bits / 8
        val data = ByteBuffer.allocate(frames * channels * w).order(ByteOrder.LITTLE_ENDIAN)
        for (f in 0 until frames) for (c in 0 until channels) {
            val s = sample(f, c)
            when {
                format == 3 && w == 4 -> data.putFloat(s.toFloat())
                format == 3 -> data.putDouble(s)
                w == 1 -> data.put(((s * 127).toInt() + 128).toByte())
                w == 2 -> data.putShort((s * 32767).toInt().toShort())
                w == 3 -> (s * 8388607).toInt().let { v -> data.put(v.toByte()).put((v shr 8).toByte()).put((v shr 16).toByte()) }
                else -> data.putInt((s * 2147483647.0).toInt())
            }
        }
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt(36 + data.capacity()).put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()).putInt(16).putShort(format.toShort()).putShort(channels.toShort()).putInt(rate)
            .putInt(rate * channels * w).putShort((channels * w).toShort()).putShort(bits.toShort())
        h.put("data".toByteArray()).putInt(data.capacity())
        return h.array() + data.array()
    }

    @Test
    fun encodesAndDecodesBack() {
        val pcm = FloatArray(1600) { (0.5 * sin(2 * PI * 440 * it / 16000)).toFloat() }
        val back = Wav.decode(Wav.encode(pcm))
        assertEquals(pcm.size, back.size)
        for (i in pcm.indices) assertTrue(abs(pcm[i] - back[i]) < 1e-4f)
    }

    @Test
    fun everyFormatToSixteenKilohertzMono() {
        val tone = { f: Int, rate: Int -> 0.5 * sin(2 * PI * 300 * f / rate) }
        for ((format, bits) in listOf(1 to 8, 1 to 16, 1 to 24, 1 to 32, 3 to 32, 3 to 64)) for (rate in listOf(16000, 44100, 48000, 8000)) {
            val frames = rate / 10
            // stereo, the same in both channels: the average is the tone
            val out = Wav.decode(wav(format, bits, 2, rate, frames) { f, _ -> tone(f, rate) })
            assertEquals("$format/$bits at $rate", 1600.0, out.size.toDouble(), 1.0)
            // well inside (the edges of the moving average aside): the tone at 16 kHz
            var err = 0.0
            for (i in 100 until out.size - 100) err = maxOf(err, abs(out[i] - tone(i, 16000)))
            assertTrue("$format/$bits at $rate: $err", err < if (rate > 16000) 0.05 else 0.02)
        }
    }

    @Test(expected = Wav.FormatException::class)
    fun notAWav() {
        Wav.decode("RIFF....AVI LIST".toByteArray())
    }
}
