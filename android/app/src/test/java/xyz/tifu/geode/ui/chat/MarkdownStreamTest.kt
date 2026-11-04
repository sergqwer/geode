package xyz.tifu.geode.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.random.Random

/** MarkdownParser, as an answer streams in: at every step the same blocks as a whole parse of the text so far. */
class MarkdownStreamTest {
    private val docs = listOf(
        "# Title\n\nA paragraph\nwith a second line.\n\n- one\n- two\n  continued\n\n1. first\n2. second\n\nThe end.",
        "Code:\n\n```kotlin\nfun a() {\n\n    println(1)\n}\n```\n\nAfter the code.\n\n```\nunclosed\n\nstill code",
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n| x |\n| y |\n\n> quote\n> more\n\n---\n\n***\n\ntext **bold** and *it*",
        "\n\n\nLeading blanks.\n   \n\t\nSpaces as blank lines.\r\n\r\nCRLF lines\r\nhere.\n\n",
        "- item\n\n  ```\n  indented fence\n  ```\n\n- next\n> q\n\n## h2 ##\n\n```py\nx = 1\n```",
    )

    private fun check(doc: String, steps: List<Int>) {
        val p = MarkdownParser()
        for (n in steps) assertEquals("after ${n} characters", parseMarkdown(doc.substring(0, n)), p.parse(doc.substring(0, n)))
    }

    @Test
    fun characterByCharacter() {
        for (d in docs) check(d, (0..d.length).toList())
    }

    @Test
    fun inChunks() {
        val r = Random(7)
        for (d in docs) repeat(20) {
            val steps = ArrayList<Int>()
            var n = 0
            while (n < d.length) { n = minOf(d.length, n + 1 + r.nextInt(12)); steps += n }
            check(d, steps)
        }
    }

    @Test
    fun aTextThatChangesIsParsedWhole() {
        val p = MarkdownParser()
        p.parse("first\n\nsecond paragraph\n\nthird")
        // a tool call moves the words into a step: the answer's text starts again
        assertEquals(parseMarkdown("new"), p.parse("new"))
        assertEquals(parseMarkdown("other\n\nwords\n\nhere"), p.parse("other\n\nwords\n\nhere"))
        assertEquals(parseMarkdown("other\n\nchanged"), p.parse("other\n\nchanged"))
    }

    @Test
    fun allDocsTogether() {
        val all = docs.joinToString("\n\n")
        check(all, (0..all.length).toList())
    }
}
