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
 * Gemma 4's chat template (GemmaTemplate) against transformers' apply_chat_template with its chat_template.jinja:
 * tools/gemma_template_cases.py writes the cases into work/tok-gemma; skipped when they are not there.
 */
@Suppress("UNCHECKED_CAST")
class GemmaTemplateTest {
    companion object {
        private lateinit var tok: Tokenizer
        private lateinit var tpl: GemmaTemplate
        private lateinit var cases: Map<String, Any?>

        @BeforeClass
        @JvmStatic
        fun load() {
            val dir = File(System.getProperty("geode.gemmatokdir") ?: "")
            val bin = File(dir, Tokenizer.FILE)
            val json = File(dir, "template_cases.json")
            assumeTrue("run tools/gemma_template_cases.py --out work/tok-gemma/template_cases.json",
                bin.isFile && json.isFile)
            tok = Tokenizer.load(bin)
            tpl = GemmaTemplate(tok)
            cases = Json.parse(json.readText()) as Map<String, Any?>
        }

        private fun ints(a: List<Any?>) = IntArray(a.size) { (a[it] as Number).toInt() }

        /** An image of 2 soft tokens (the cases' IMAGE_TOKENS), a clip of 25 (AUDIO_TOKENS), a video of two frames
         *  of 2 (FRAMES). */
        private val image = TurnImage("t", 96, 48)
        private val clip = TurnAudio("a", 16000)
        private val frames = listOf(TurnImage("f0", 96, 48, frame = true, second = 0),
            TurnImage("f1", 96, 48, frame = true, second = 1))

        private fun message(m: Map<String, Any?>): TemplateMessage {
            val parts = when (val c = m["content"]) {
                is String -> listOf(Part.Text(c))
                is List<*> -> c.mapIndexed { i, p ->
                    p as Map<String, Any?>
                    when (p["type"]) {
                        "image" -> Part.Image(image, "image$i")
                        "audio" -> Part.Audio(clip, "audio$i")
                        "video" -> Part.Video(frames, null, "video$i")
                        else -> Part.Text(p.str("text")!!)
                    }
                }
                else -> emptyList()
            }
            val calls = m.list("tool_calls")?.map { tc ->
                tc as Map<String, Any?>
                val f = tc.obj("function")!!
                ToolCall(tc.str("id").orEmpty(), f.str("name")!!, f.obj("arguments")!!)
            } ?: emptyList()
            return TemplateMessage(m.str("role")!!, parts, m.str("reasoning_content"), calls, m.str("tool_call_id"))
        }

        private fun messages(c: Map<String, Any?>) = c.list("messages")!!.map { message(it as Map<String, Any?>) }

        /** A value with its maps' keys sorted and its numbers as text: what two ways to read it agree on. */
        private fun canon(v: Any?): Any? = when (v) {
            is Map<*, *> -> v.entries.associate { it.key.toString() to canon(it.value) }.toSortedMap()
            is List<*> -> v.map(::canon)
            is Number -> Json.write(v)
            else -> v
        }
    }

    @Test
    fun rendersLikeHuggingFace() {
        cases.list("chats")!!.forEachIndexed { i, c ->
            c as Map<String, Any?>
            val want = ints(c.list("ids")!!)
            val got = tpl.render(messages(c), c.list("tools") ?: emptyList(), c["thinking"] as Boolean).ids
            assertEquals("chat $i", tok.decode(want), tok.decode(got))
            assertArrayEquals("chat $i", want, got)
        }
    }

    @Test
    fun continuesLikeHuggingFace() {
        cases.list("continuations")!!.forEachIndexed { i, c ->
            c as Map<String, Any?>
            val want = ints(c.list("tail")!!)
            val got = tpl.continuation(messages(c), (c["held"] as Number).toInt(), c["thinking"] as Boolean)
            assertEquals("continuation $i", tok.decode(want), tok.decode(got.ids))
            assertArrayEquals("continuation $i", want, got.ids)
        }
    }

    @Test
    fun callsReadBackAsWritten() {
        // every call the template renders, read back from its text, is the call it was
        var n = 0
        cases.list("chats")!!.forEach { c ->
            c as Map<String, Any?>
            val text = tok.decode(ints(c.list("ids")!!))
            val want = messages(c).flatMap { it.toolCalls }
            if (want.isEmpty()) return@forEach
            val calls = text.split("<|tool_call>").drop(1).flatMap { tpl.parseCalls("<|tool_call>$it", emptyList()) }
            assertEquals(want.map { it.name }, calls.map { it.first })
            assertEquals(want.map { canon(it.arguments) }, calls.map { canon(it.second) })
            n += calls.size
        }
        assertTrue(n >= 5)
    }

    @Test
    fun callsAsTheModelWritesThem() {
        val text = "<|tool_call>call:search{Query:<|\"|>cats, \"big\" {x}<|\"|>,n:-3,r:2.5e-3,ok:true,none:null," +
            "word:Kyiv,list:[1,<|\"|>a<|\"|>,[]],obj:{<|\"|>k<|\"|>:{}}}<tool_call|>" +
            "<|tool_call>call:now{}<tool_call|><|tool_call>call:broken{a:<|\"|>open"
        val calls = tpl.parseCalls(text, emptyList())
        assertEquals(listOf("search", "now"), calls.map { it.first })
        val a = calls[0].second
        assertEquals("cats, \"big\" {x}", a["Query"])
        assertEquals(-3L, a["n"])
        assertEquals("2.5e-3", Json.write(a["r"]))
        assertEquals(true, a["ok"])
        assertTrue(a.containsKey("none") && a["none"] == null)
        assertEquals("Kyiv", a["word"])
        assertEquals("""[1,"a",[]]""", Json.write(a["list"]))
        assertEquals("""{"k":{}}""", Json.write(a["obj"]))
        assertTrue(calls[1].second.isEmpty())
    }

    @Test
    fun aBrokenCallDoesNotTakeTheNext() {
        val calls = tpl.parseCalls("<|tool_call>call:a<tool_call|><|tool_call>call:b{x:1}<tool_call|>" +
            "<|tool_call>call:c{y:2,l:[1,],}<tool_call|><|tool_call>call:d{n:1.,m:.5,k:007,e:1e5}<tool_call|>" +
            "<|tool_call>call:e{a:1,}<tool_call|><|tool_call>call:f{z:3}<tool_call|>", emptyList())
        assertEquals(listOf("b", "c", "d", "e", "f"), calls.map { it.first })
        assertEquals(1L, calls[0].second["x"])
        assertEquals(2L, calls[1].second["y"])
        assertEquals("[1]", Json.write(calls[1].second["l"]))
        // what JSON would not take as a number stays the model's text
        assertEquals(listOf("1.", ".5", "007"), listOf("n", "m", "k").map { calls[2].second[it] })
        assertEquals("1e5", Json.write(calls[2].second["e"]))
        assertEquals(1L, calls[3].second["a"])
        assertEquals(3L, calls[4].second["z"])
    }

    @Test
    fun systemTurnAndReasoning() {
        val chats = cases.list("chats")!!.map { it as Map<String, Any?> }
        // [system, user] with tools: the system turn is the shared start
        val c = chats[3]
        val p = tpl.render(messages(c), c.list("tools")!!, false)
        val n = tpl.systemTurn(p)
        assertTrue(n > 10)
        assertEquals(tok.special("<turn|>"), p.ids[n - 2])
        assertEquals(tok.special("<|turn>"), p.ids[n])
        // no system turn at all
        assertEquals(0, tpl.systemTurn(tpl.render(messages(chats[11]), emptyList(), false)))
        // after a call's output, with reasoning on, the prompt opens the reasoning; a question's does not
        assertTrue(tpl.render(messages(chats[5]), chats[5].list("tools")!!, true).reasoning)
        assertFalse(tpl.render(messages(chats[0]), chats[0].list("tools")!!, true).reasoning)
    }

    @Test
    fun aVideosSoundFollowsItsFrames() {
        // a video with its sound is its frames, then the clip - the media in that order for the engine
        val c = cases.list("chats")!!.map { it as Map<String, Any?> }.first { "Watch:" in (it["text"] as String) }
        val withSound = TemplateMessage("user", listOf(Part.Text("Watch: "), Part.Video(frames, clip, "v"),
            Part.Text("What happens?")))
        val p = tpl.render(listOf(withSound), emptyList(), true)
        assertArrayEquals(ints(c.list("ids")!!), p.ids)
        assertEquals(listOf("f0", "f1", "a"), p.media.map { it.name })
        assertTrue(p.images.all { it.frame })
    }

    @Test
    fun prefillLeavesTheAnswerOpen() {
        val msgs = listOf(TemplateMessage("user", listOf(Part.Text("Hi"))),
            TemplateMessage("assistant", listOf(Part.Text("  Hello, I am "))))
        val p = tpl.render(msgs, emptyList(), false, prefill = true)
        assertEquals("<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello, I am", tok.decode(p.ids))
    }

    @Test
    fun aHeldAnswersContentIsNotRepeated() {
        // the engine wrote "Let me look." and the call; the call's output follows, the content not again
        val msgs = listOf(TemplateMessage("user", listOf(Part.Text("Weather?"))),
            TemplateMessage("assistant", listOf(Part.Text("Let me look.")),
                toolCalls = listOf(ToolCall("c1", "get_weather", linkedMapOf("city" to "Kyiv")))),
            TemplateMessage("tool", listOf(Part.Text("5")), toolCallId = "c1"))
        val p = tpl.continuation(msgs, 2, false)
        assertEquals("<|tool_response>response:get_weather{value:<|\"|>5<|\"|>}<tool_response|>", tok.decode(p.ids))
    }

    @Test
    fun aConversationGoesOnAfterAClip() {
        // a voice question, its answer held, a typed one after: the continuation is the full render's end, and the
        // clip (before the held answer) is not among its media - the media counted whole, not only the images
        val msgs = listOf(TemplateMessage("user", listOf(Part.Audio(clip, "a"))),
            TemplateMessage("assistant", listOf(Part.Text("Paris"))),
            TemplateMessage("user", listOf(Part.Text("And of Germany?"))))
        val full = tpl.render(msgs, emptyList(), false)
        val p = tpl.continuation(msgs, 2, false)
        assertTrue(p.media.isEmpty())
        assertArrayEquals(full.ids.copyOfRange(full.size - p.size, full.size), p.ids)
        assertEquals("<turn|>\n<|turn>user\nAnd of Germany?<turn|>\n<|turn>model\n", tok.decode(p.ids))
    }
}
