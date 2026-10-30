package xyz.tifu.geode.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import xyz.tifu.geode.R

/** A Markdown document as blocks; the text inside them is inline Markdown (rendered by [inlineMarkdown]). */
sealed interface MdBlock {
    data class Heading(val level: Int, val text: String) : MdBlock
    data class Paragraph(val text: String) : MdBlock
    data class Code(val lang: String, val code: String) : MdBlock
    data class ListItem(val indent: Int, val marker: String, val text: String) : MdBlock
    data class Quote(val text: String) : MdBlock
    data object Rule : MdBlock
    data class Table(val header: List<String>, val rows: List<List<String>>) : MdBlock
}

private val HEADING = Regex("^(#{1,6})\\s+(.*)$")
private val RULE = Regex("^\\s*([-*_])(\\s*\\1){2,}\\s*$")
private val BULLET = Regex("^(\\s*)([-*+])\\s+(.*)$")
private val NUMBERED = Regex("^(\\s*)(\\d{1,4})[.)]\\s+(.*)$")
private val FENCE = Regex("^\\s*(```|~~~)\\s*([\\w+#.-]*)\\s*$")
private val TABLE_SEPARATOR = Regex("^\\s*\\|?\\s*:?-+:?\\s*(\\|\\s*:?-+:?\\s*)*\\|?\\s*$")

/**
 * Block-level Markdown, line by line: fenced code (an unclosed fence runs to the end, as while an answer streams),
 * headings, rules, tables, lists, quotes, paragraphs. Single line breaks inside a paragraph are kept: chat models use
 * them as real breaks.
 */
fun parseMarkdown(src: String): List<MdBlock> {
    val lines = src.replace("\r\n", "\n").split('\n')
    val out = ArrayList<MdBlock>()
    val para = StringBuilder()
    fun flush() {
        if (para.isNotBlank()) out += MdBlock.Paragraph(para.toString().trimEnd())
        para.clear()
    }
    var i = 0
    while (i < lines.size) {
        val line = lines[i]
        val fence = FENCE.matchEntire(line)
        when {
            fence != null -> {
                flush()
                val mark = fence.groupValues[1]
                val code = StringBuilder()
                i++
                while (i < lines.size && !lines[i].trim().startsWith(mark)) {
                    if (code.isNotEmpty()) code.append('\n')
                    code.append(lines[i])
                    i++
                }
                out += MdBlock.Code(fence.groupValues[2], code.toString())
            }
            line.isBlank() -> flush()
            HEADING.matches(line) -> {
                flush()
                val m = HEADING.matchEntire(line)!!
                out += MdBlock.Heading(m.groupValues[1].length, m.groupValues[2].trimEnd('#', ' '))
            }
            RULE.matches(line) -> {
                flush()
                out += MdBlock.Rule
            }
            line.trimStart().startsWith("|") && i + 1 < lines.size && TABLE_SEPARATOR.matches(lines[i + 1]) -> {
                flush()
                val header = cells(line)
                val rows = ArrayList<List<String>>()
                i += 2
                while (i < lines.size && lines[i].trimStart().startsWith("|")) {
                    rows += cells(lines[i])
                    i++
                }
                out += MdBlock.Table(header, rows)
                continue
            }
            BULLET.matches(line) || NUMBERED.matches(line) -> {
                flush()
                val b = BULLET.matchEntire(line)
                val n = NUMBERED.matchEntire(line)
                val indent = ((b ?: n)!!.groupValues[1].replace("\t", "    ").length) / 2
                val marker = if (b != null) "•" else n!!.groupValues[2] + "."
                val text = StringBuilder((b ?: n)!!.groupValues[3])
                // continuation lines: indented, not a new item
                while (i + 1 < lines.size && lines[i + 1].isNotBlank() && lines[i + 1].startsWith("  ") &&
                    !BULLET.matches(lines[i + 1]) && !NUMBERED.matches(lines[i + 1])) {
                    text.append('\n').append(lines[i + 1].trim())
                    i++
                }
                out += MdBlock.ListItem(indent, marker, text.toString())
            }
            line.trimStart().startsWith(">") -> {
                flush()
                val q = StringBuilder()
                while (i < lines.size && lines[i].trimStart().startsWith(">")) {
                    if (q.isNotEmpty()) q.append('\n')
                    q.append(lines[i].trimStart().removePrefix(">").removePrefix(" "))
                    i++
                }
                out += MdBlock.Quote(q.toString())
                continue
            }
            else -> {
                if (para.isNotEmpty()) para.append('\n')
                para.append(line)
            }
        }
        i++
    }
    flush()
    return out
}

private fun cells(row: String): List<String> {
    var s = row.trim()
    if (s.startsWith("|")) s = s.substring(1)
    if (s.endsWith("|")) s = s.substring(0, s.length - 1)
    return s.split('|').map { it.trim() }
}

/** The colours and fonts inline Markdown is drawn with. */
data class MdStyle(val code: SpanStyle, val link: SpanStyle)

/** Inline Markdown: `code`, **bold** / __bold__, *italic* / _italic_, ~~strike~~, [links](url), \escapes. */
fun inlineMarkdown(s: String, st: MdStyle): AnnotatedString = buildAnnotatedString { appendInline(s, st) }

private fun AnnotatedString.Builder.appendInline(s: String, st: MdStyle) {
    var i = 0
    fun wordChar(c: Char?) = c != null && c.isLetterOrDigit()
    while (i < s.length) {
        val c = s[i]
        if (c == '\\' && i + 1 < s.length && s[i + 1] in "\\`*_~[]()#>|-+.!") {
            append(s[i + 1])
            i += 2
            continue
        }
        if (c == '`') {
            val end = s.indexOf('`', i + 1)
            if (end > i + 1) {
                withStyle(st.code) { append(s.substring(i + 1, end)) }
                i = end + 1
                continue
            }
        }
        if ((s.startsWith("**", i) || s.startsWith("__", i))) {
            val mark = s.substring(i, i + 2)
            val end = s.indexOf(mark, i + 2)
            if (end > i + 2) {
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { appendInline(s.substring(i + 2, end), st) }
                i = end + 2
                continue
            }
        }
        if (s.startsWith("~~", i)) {
            val end = s.indexOf("~~", i + 2)
            if (end > i + 2) {
                withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { appendInline(s.substring(i + 2, end), st) }
                i = end + 2
                continue
            }
        }
        if ((c == '*' || c == '_') && i + 1 < s.length && !s[i + 1].isWhitespace() &&
            (c == '*' || !wordChar(s.getOrNull(i - 1)))) {
            var end = s.indexOf(c, i + 1)
            // _snake_case_ is not italics: the closing underscore must not run into a word
            while (end > 0 && c == '_' && wordChar(s.getOrNull(end + 1))) end = s.indexOf(c, end + 1)
            if (end > i + 1 && !s[end - 1].isWhitespace()) {
                withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { appendInline(s.substring(i + 1, end), st) }
                i = end + 1
                continue
            }
        }
        if (c == '[') {
            val close = s.indexOf("](", i + 1)
            val paren = if (close > 0) s.indexOf(')', close + 2) else -1
            if (close > i && paren > close) {
                val url = s.substring(close + 2, paren).trim()
                withLink(LinkAnnotation.Url(url, TextLinkStyles(st.link))) { appendInline(s.substring(i + 1, close), st) }
                i = paren + 1
                continue
            }
        }
        if (s.startsWith("http://", i) || s.startsWith("https://", i)) {
            var end = i
            while (end < s.length && !s[end].isWhitespace() && s[end] !in "<>\"'") end++
            while (end > i && s[end - 1] in ".,;:!?)") end--
            val url = s.substring(i, end)
            withLink(LinkAnnotation.Url(url, TextLinkStyles(st.link))) { append(url) }
            i = end
            continue
        }
        append(c)
        i++
    }
}

@Composable
fun rememberMdStyle(): MdStyle {
    val cs = MaterialTheme.colorScheme
    return remember(cs) {
        MdStyle(
            code = SpanStyle(fontFamily = FontFamily.Monospace, background = cs.surfaceVariant, color = cs.onSurfaceVariant),
            link = SpanStyle(color = cs.primary, textDecoration = TextDecoration.Underline),
        )
    }
}

/** Markdown rendered as Compose text. Parsed again on every change: chat answers are short and stream in. */
@Composable
fun MarkdownText(markdown: String, modifier: Modifier = Modifier, color: Color = Color.Unspecified) {
    val blocks = remember(markdown) { parseMarkdown(markdown) }
    val st = rememberMdStyle()
    val body = MaterialTheme.typography.bodyLarge.copy(color = if (color != Color.Unspecified) color else MaterialTheme.colorScheme.onSurface)
    Column(modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        blocks.forEach { b ->
            when (b) {
                is MdBlock.Heading -> Text(
                    inlineMarkdown(b.text, st),
                    style = when (b.level) {
                        1 -> MaterialTheme.typography.titleLarge
                        2 -> MaterialTheme.typography.titleMedium
                        else -> MaterialTheme.typography.titleSmall
                    }.copy(fontWeight = FontWeight.Bold, color = body.color),
                )
                is MdBlock.Paragraph -> Text(inlineMarkdown(b.text, st), style = body)
                is MdBlock.Code -> CodeBlock(b)
                is MdBlock.ListItem -> Row(Modifier.padding(start = (12 * b.indent).dp)) {
                    Text(b.marker, style = body, modifier = Modifier.width(if (b.marker == "•") 16.dp else 28.dp))
                    Text(inlineMarkdown(b.text, st), style = body)
                }
                is MdBlock.Quote -> Row(Modifier.height(IntrinsicSize.Min)) {
                    Box(Modifier.width(3.dp).fillMaxHeight().background(MaterialTheme.colorScheme.outline))
                    Text(inlineMarkdown(b.text, st), style = body.copy(color = MaterialTheme.colorScheme.onSurfaceVariant),
                        modifier = Modifier.padding(start = 8.dp))
                }
                MdBlock.Rule -> HorizontalDivider()
                is MdBlock.Table -> TableBlock(b, st, body)
            }
        }
    }
}

@Composable
private fun CodeBlock(b: MdBlock.Code) {
    val clipboard = LocalClipboardManager.current
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(8.dp)) {
        Column(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().padding(start = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(b.lang, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f))
                IconButton(onClick = { clipboard.setText(AnnotatedString(b.code)) }, modifier = Modifier.size(36.dp)) {
                    Icon(Icons.Outlined.ContentCopy, stringResource(R.string.chat_copy_code), Modifier.size(16.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Text(
                b.code,
                style = TextStyle(fontFamily = FontFamily.Monospace, fontSize = MaterialTheme.typography.bodySmall.fontSize,
                    color = MaterialTheme.colorScheme.onSurfaceVariant),
                softWrap = false,
                modifier = Modifier.horizontalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, bottom = 10.dp),
            )
        }
    }
}

@Composable
private fun TableBlock(t: MdBlock.Table, st: MdStyle, body: TextStyle) {
    // every column as wide as its widest cell (up to a limit), so the rows line up
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    val n = maxOf(t.header.size, t.rows.maxOfOrNull { it.size } ?: 0)
    val widths = (0 until n).map { c ->
        val texts = listOf(t.header.getOrNull(c).orEmpty()) + t.rows.map { it.getOrNull(c).orEmpty() }
        val px = texts.maxOf { measurer.measure(inlineMarkdown(it, st), body.copy(fontWeight = FontWeight.Bold)).size.width }
        with(density) { px.toDp() }.coerceIn(48.dp, 220.dp) + 16.dp
    }
    val line = MaterialTheme.colorScheme.outlineVariant
    Column(Modifier.horizontalScroll(rememberScrollState())) {
        (listOf(t.header) + t.rows).forEachIndexed { r, row ->
            Row(Modifier.height(IntrinsicSize.Min).background(if (r == 0) MaterialTheme.colorScheme.surfaceVariant else Color.Transparent)) {
                for (c in 0 until n) {
                    Text(
                        inlineMarkdown(row.getOrNull(c).orEmpty(), st),
                        style = if (r == 0) body.copy(fontWeight = FontWeight.Bold) else body,
                        modifier = Modifier.width(widths[c]).fillMaxHeight().padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                }
            }
            HorizontalDivider(Modifier.width(widths.fold(0.dp) { a, w -> a + w }), color = line)
        }
    }
}
