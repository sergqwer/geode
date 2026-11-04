package xyz.tifu.geode.tokenizer

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.obj
import xyz.tifu.geode.util.str
import java.io.File

/**
 * The full chat template (ChatTemplate) against transformers' apply_chat_template with Qwen3.6's chat_template.jinja:
 * tools/template_cases.py writes the cases into work/tok; skipped when they are not there.
 */
@Suppress("UNCHECKED_CAST")
class TemplateTest {
    companion object {
        private lateinit var tok: Tokenizer
        private lateinit var cases: Map<String, Any?>

        @BeforeClass
        @JvmStatic
        fun load() {
            val dir = File(System.getProperty("geode.tokdir") ?: "")
            val bin = File(dir, Tokenizer.FILE)
            val json = File(dir, "template_cases.json")
            assumeTrue("run tools/template_cases.py --out work/tok/template_cases.json", bin.isFile && json.isFile)
            tok = Tokenizer.load(bin)
            cases = Json.parse(json.readText()) as Map<String, Any?>
        }

        private fun ints(a: List<Any?>) = IntArray(a.size) { (a[it] as Number).toInt() }

        /** An image the model reads as one token (the template's single pad). */
        private val image = TurnImage("t", 32, 32)

        private fun message(m: Map<String, Any?>): TemplateMessage {
            val parts = when (val c = m["content"]) {
                is String -> listOf(Part.Text(c))
                is List<*> -> c.mapIndexed { i, p ->
                    p as Map<String, Any?>
                    if (p["type"] == "image") Part.Image(image, "image$i") else Part.Text(p.str("text")!!)
                }
                else -> emptyList()
            }
            val calls = m.list("tool_calls")?.map { tc ->
                tc as Map<String, Any?>
                val f = tc.obj("function")!!
                ToolCall(tc.str("id").orEmpty(), f.str("name")!!, f.obj("arguments")!!)
            } ?: emptyList()
            return TemplateMessage(m.str("role")!!, parts, m.str("reasoning_content"), calls)
        }

        private fun messages(c: Map<String, Any?>) = c.list("messages")!!.map { message(it as Map<String, Any?>) }
    }

    @Test
    fun rendersLikeHuggingFace() {
        val tpl = ChatTemplate(tok)
        cases.list("chats")!!.forEachIndexed { i, c ->
            c as Map<String, Any?>
            val want = ints(c.list("ids")!!)
            val got = tpl.render(messages(c), c.list("tools") ?: emptyList(), c["thinking"] as Boolean).ids
            // the text first: a readable difference
            assertEquals("chat $i", tok.decode(want), tok.decode(got))
            assertArrayEquals("chat $i", want, got)
        }
    }

    @Test
    fun theNameOfACallBeingWritten() {
        val tpl = ChatTemplate(tok)
        // the chat's status while a long call is written: the name once its tag is closed, the last call's
        assertEquals(null, tpl.callName("<tool_call>\n"))
        assertEquals(null, tpl.callName("<tool_call>\n<function=wri"))
        assertEquals("write", tpl.callName("<tool_call>\n<function=write>\n<parameter=content>\n<html><p>a</p>"))
        assertEquals("bash", tpl.callName("<tool_call>\n<function=write>\n</function>\n</tool_call>\n" +
            "<tool_call>\n<function=bash>\n<parameter=cmd>\nls"))
    }

    @Test
    fun continuesLikeHuggingFace() {
        val tpl = ChatTemplate(tok)
        cases.list("continuations")!!.forEachIndexed { i, c ->
            c as Map<String, Any?>
            // the answer the engine holds is closed first, then the rest as the whole render has it
            val want = intArrayOf(tok.special("<|im_end|>")) + tok.encode("\n") + ints(c.list("tail")!!)
            val got = tpl.continuation(messages(c), (c["held"] as Number).toInt(), c["thinking"] as Boolean).ids
            assertEquals("continuation $i", tok.decode(want), tok.decode(got))
            assertArrayEquals("continuation $i", want, got)
        }
    }

    @Test
    fun sameMessages() {
        val a = TemplateMessage("assistant", listOf(Part.Text("Hello!\n")), reasoning = "hm",
            toolCalls = listOf(ToolCall("call_1", "f", linkedMapOf("x" to 1, "y" to "z"))))
        // what a client sends back: no reasoning, another id, the content trimmed
        val b = TemplateMessage("assistant", listOf(Part.Text("Hello!")),
            toolCalls = listOf(ToolCall("call_9", "f", linkedMapOf("x" to 1, "y" to "z"))))
        assertTrue(a.sameAs(b))
        assertFalse(a.sameAs(b.copy(toolCalls = listOf(ToolCall("call_9", "f", linkedMapOf("x" to 2))))))
        assertFalse(a.sameAs(b.copy(role = "user")))
    }

    @Test
    fun jsonLikePython() {
        val v = Json.parse("""{"b": [1, 2.5, true, null], "a": "Привіт \"x\"\n\t", "c": {}}""")
        assertEquals("""{"b": [1, 2.5, true, null], "a": "Привіт \"x\"\n\t", "c": {}}""", Json.dumps(v))
        assertEquals("""{"b":[1,2.5,true,null],"a":"Привіт \"x\"\n\t","c":{}}""", Json.write(v))
        assertEquals("\u0001", Json.parse("\"\\u0001\""))
        assertEquals("\"\\u0001\"", Json.dumps("\u0001"))
    }
}
