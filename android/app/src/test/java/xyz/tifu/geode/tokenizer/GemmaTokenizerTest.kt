package xyz.tifu.geode.tokenizer

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * Gemma 4's SentencePiece-style tokenizer against Hugging Face's: tools/tokbin.py --model google/gemma-4-E2B-it-...
 * writes tokenizer.bin and cases.json into work/tok-gemma; the test is skipped when they are not there.
 */
class GemmaTokenizerTest {
    companion object {
        private lateinit var tok: Tokenizer
        private lateinit var cases: JSONObject

        @BeforeClass
        @JvmStatic
        fun load() {
            val dir = File(System.getProperty("geode.gemmatokdir") ?: "")
            val bin = File(dir, Tokenizer.FILE)
            val json = File(dir, "cases.json")
            assumeTrue("run tools/tokbin.py --model google/gemma-4-E2B-it-qat-q4_0-unquantized " +
                "--out work/tok-gemma/tokenizer.bin --cases work/tok-gemma/cases.json", bin.isFile && json.isFile)
            tok = Tokenizer.load(bin)
            cases = JSONObject(json.readText())
        }

        private fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
    }

    @Test
    fun isSentencePieceStyle() {
        assertTrue(tok.sentencePiece)
        assertEquals(262144, tok.vocabSize)
        assertEquals(105, tok.special("<|turn>"))
        assertEquals(106, tok.special("<turn|>"))
        assertEquals(2, tok.special("<bos>"))
    }

    @Test
    fun encodesLikeHuggingFace() {
        val texts = cases.getJSONArray("texts")
        for (i in 0 until texts.length()) {
            val c = texts.getJSONObject(i)
            val text = c.getString("text")
            assertArrayEquals("encode(${text.take(40)})", ints(c.getJSONArray("ids")), tok.encode(text))
        }
    }

    @Test
    fun decodeRoundTrips() {
        // no normalizer but the space: decoding gives back the text - but for a "▁" of its own, which comes back
        // a space (Hugging Face's decoder does the same)
        val texts = cases.getJSONArray("texts")
        for (i in 0 until texts.length()) {
            val text = texts.getJSONObject(i).getString("text")
            val want = text.replace('▁', ' ')
            assertEquals(want, tok.decode(tok.encode(text)))
            val d = tok.StreamDecoder()
            val sb = StringBuilder()
            for (id in tok.encode(text)) sb.append(d.push(id))
            sb.append(d.flush())
            assertEquals(want, sb.toString())
        }
    }

    @Test
    fun renderedChatsSplitAtTheSpecialTokens() {
        // the rendered template, split at its added tokens the way transformers does it
        val chats = cases.getJSONArray("chats")
        for (i in 0 until chats.length()) {
            val c = chats.getJSONObject(i)
            assertArrayEquals("chat $i", ints(c.getJSONArray("ids")), tok.encodeWithSpecials(c.getString("text")))
        }
    }

    @Test
    fun aLongTextIsFast() {
        // the heap BPE is n log n: a 200 KB message must not take the quadratic scan's minutes
        val texts = cases.getJSONArray("texts")
        val sb = StringBuilder()
        var k = 0
        while (sb.length < 200_000) sb.append(texts.getJSONObject(k++ % texts.length()).getString("text")).append('\n')
        val text = sb.toString()
        tok.encode(text) // warm up
        val t0 = System.nanoTime()
        val ids = tok.encode(text)
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertEquals(text.replace('▁', ' '), tok.decode(ids))
        assertTrue("encoding ${text.length} chars took $ms ms", ms < 3000)
    }
}
