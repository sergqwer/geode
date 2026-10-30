package xyz.tifu.geode.ui.chat

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MarkdownTest {
    private val st = MdStyle(code = SpanStyle(color = Color.Red), link = SpanStyle(color = Color.Blue))

    @Test
    fun blocks() {
        val md = """
            # Title
            Some text
            on two lines.

            - one
            - **two**
              continued
            1. first
            2. second

            > a quote

            ---
            ```kotlin
            val x = 1
            ```
            | a | b |
            |---|:-:|
            | 1 | 2 |
        """.trimIndent()
        val b = parseMarkdown(md)
        assertEquals(MdBlock.Heading(1, "Title"), b[0])
        assertEquals(MdBlock.Paragraph("Some text\non two lines."), b[1])
        assertEquals(MdBlock.ListItem(0, "•", "one"), b[2])
        assertEquals(MdBlock.ListItem(0, "•", "**two**\ncontinued"), b[3])
        assertEquals(MdBlock.ListItem(0, "1.", "first"), b[4])
        assertEquals(MdBlock.ListItem(0, "2.", "second"), b[5])
        assertEquals(MdBlock.Quote("a quote"), b[6])
        assertEquals(MdBlock.Rule, b[7])
        assertEquals(MdBlock.Code("kotlin", "val x = 1"), b[8])
        assertEquals(MdBlock.Table(listOf("a", "b"), listOf(listOf("1", "2"))), b[9])
    }

    @Test
    fun anUnclosedFenceRunsToTheEnd() {
        // an answer still streaming
        val b = parseMarkdown("Here:\n```python\nprint(1)\nprint(2")
        assertEquals(MdBlock.Code("python", "print(1)\nprint(2"), b[1])
    }

    @Test
    fun inline() {
        val a = inlineMarkdown("a **bold** and *it* `x*y` [link](https://e.x) snake_case_name ~~s~~ \\*", st)
        assertEquals("a bold and it x*y link snake_case_name s *", a.text)
        assertTrue(a.spanStyles.any { it.item.fontWeight == FontWeight.Bold && a.text.substring(it.start, it.end) == "bold" })
        assertTrue(a.spanStyles.any { it.item.fontStyle == FontStyle.Italic && a.text.substring(it.start, it.end) == "it" })
        assertTrue(a.spanStyles.any { it.item == st.code && a.text.substring(it.start, it.end) == "x*y" })
        assertEquals(1, a.getLinkAnnotations(0, a.length).size)
    }

    @Test
    fun unclosedInlineMarksStayAsText() {
        assertEquals("**half", inlineMarkdown("**half", st).text)
        assertEquals("2 * 3 = 6", inlineMarkdown("2 * 3 = 6", st).text)
    }
}
