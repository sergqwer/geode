package xyz.tifu.geode.tokenizer

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * YuE2's text tokenizer (qwen.tiktoken, the music model's lyrics / tags / ABC) against the reference's tiktoken:
 * tools/pack_yue2.py --tokenizer-only --cases work/tok-yue2 writes tokenizer.bin and cases.json there; the test is
 * skipped when they are not there.
 */
class Yue2TokenizerTest {
    companion object {
        private lateinit var tok: Tokenizer
        private lateinit var cases: JSONObject

        @BeforeClass
        @JvmStatic
        fun load() {
            val dir = File(System.getProperty("geode.yue2tokdir") ?: "")
            val bin = File(dir, Tokenizer.FILE)
            val json = File(dir, "cases.json")
            assumeTrue("run tools/pack_yue2.py ... --tokenizer-only --cases work/tok-yue2", bin.isFile && json.isFile)
            tok = Tokenizer.load(bin)
            cases = JSONObject(json.readText())
        }

        private fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
    }

    @Test
    fun hasYue2sSpecials() {
        assertFalse(tok.sentencePiece)
        assertEquals(151643, tok.special("<|endoftext|>"))
        assertEquals(151847, tok.special("<abc>"))
        assertEquals(151848, tok.special("</abc>"))
    }

    @Test
    fun encodesLikeTiktoken() {
        val texts = cases.getJSONArray("texts")
        for (i in 0 until texts.length()) {
            val c = texts.getJSONObject(i)
            val text = c.getString("text")
            assertArrayEquals("encode(${text.take(40)})", ints(c.getJSONArray("ids")), tok.encode(text))
        }
    }

    @Test
    fun decodeRoundTrips() {
        val texts = cases.getJSONArray("texts")
        for (i in 0 until texts.length()) {
            val text = texts.getJSONObject(i).getString("text")
            val nfc = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
            assertEquals(nfc, tok.decode(tok.encode(text)))
        }
    }
}
