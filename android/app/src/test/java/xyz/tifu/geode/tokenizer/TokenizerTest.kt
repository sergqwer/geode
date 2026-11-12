package xyz.tifu.geode.tokenizer

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * The app's tokenizer against Hugging Face's: tools/tokbin.py writes tokenizer.bin and cases.json (texts and chat
 * renders with the ids transformers gives them) into work/tok; the test is skipped when they are not there.
 */
class TokenizerTest {
    companion object {
        private lateinit var tok: Tokenizer
        private lateinit var cases: JSONObject

        @BeforeClass
        @JvmStatic
        fun load() {
            val dir = File(System.getProperty("geode.tokdir") ?: "")
            val bin = File(dir, Tokenizer.FILE)
            val json = File(dir, "cases.json")
            assumeTrue("run tools/tokbin.py --out work/tok/tokenizer.bin --cases work/tok/cases.json",
                bin.isFile && json.isFile)
            tok = Tokenizer.load(bin)
            cases = JSONObject(json.readText())
        }

        private fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
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
        val texts = cases.getJSONArray("texts")
        for (i in 0 until texts.length()) {
            val text = texts.getJSONObject(i).getString("text")
            val nfc = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
            assertEquals(nfc, tok.decode(tok.encode(text)))
            // streaming, one token at a time, must give the same text
            val d = tok.StreamDecoder()
            val sb = StringBuilder()
            for (id in tok.encode(text)) sb.append(d.push(id))
            sb.append(d.flush())
            assertEquals(nfc, sb.toString())
        }
    }

    @Test
    fun chatTemplateLikeHuggingFace() {
        // the app's Qwen template (ChatTemplate, the one every request is rendered with) against apply_chat_template
        val tpl = ChatTemplate(tok)
        val chats = cases.getJSONArray("chats")
        for (i in 0 until chats.length()) {
            val c = chats.getJSONObject(i)
            val msgs = c.getJSONArray("messages")
            val turns = (0 until msgs.length()).map {
                val m = msgs.getJSONObject(it)
                TemplateMessage(m.getString("role"), listOf(Part.Text(m.getString("content"))))
            }
            assertArrayEquals("chat $i", ints(c.getJSONArray("ids")), tpl.render(turns, emptyList(), c.getBoolean("thinking")).ids)
        }
    }

    @Test
    fun anImageQuestionLikeHuggingFace() {
        // Hugging Face's ids of the template rendered for [image, text] (thinking off), the image 32 x 32: one token
        val p = ChatTemplate(tok).render(listOf(TemplateMessage("user", listOf(Part.Image(TurnImage("x.jpg", 32, 32), "x"),
            Part.Text("What is in this image? Describe it briefly.")))), emptyList(), thinking = false)
        assertArrayEquals(intArrayOf(248045, 846, 198, 248053, 248056, 248054, 3710, 369, 303, 411, 2099, 30, 58737,
            424, 25899, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271), p.ids)
        assertEquals(listOf(TurnImage("x.jpg", 32, 32)), p.images)
    }
}
