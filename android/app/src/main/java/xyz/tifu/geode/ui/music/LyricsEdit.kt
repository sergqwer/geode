package xyz.tifu.geode.ui.music

import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import xyz.tifu.geode.music.LyricsWriter
import xyz.tifu.geode.music.YuE2Prompt

/** The lyrics editor's toolbar: what YuE2 reads besides the words, put in where the cursor is. */
object LyricsEdit {
    /**
     * [tag] ("Chorus") as a section header on its own line at the cursor: on the cursor's line when it is empty, else
     * after it, with a blank line before it (the convention the model was trained around). A break ([YuE2Prompt.BREAKS]:
     * played, not sung) gets a blank line after it as well - no words under it - and the cursor goes past it.
     */
    fun insertTag(v: TextFieldValue, tag: String): TextFieldValue {
        val text = v.text
        val c = v.selection.end.coerceIn(0, text.length)
        val lineStart = text.lastIndexOf('\n', (c - 1).coerceAtLeast(0)).let { if (c == 0) 0 else it + 1 }
        val lineEnd = text.indexOf('\n', c).let { if (it < 0) text.length else it }
        val blank = text.substring(lineStart, lineEnd).isBlank()
        val at = if (blank) lineStart else lineEnd
        val head = text.substring(0, at).trimEnd()
        val tail = text.substring(if (blank) lineEnd else at).trimStart('\n', '\r')
        val header = "[$tag]"
        val pause = tag in YuE2Prompt.BREAKS
        val ins = (if (head.isEmpty()) "" else "\n\n") + header + (if (pause) "\n\n" else "\n")
        val out = head + ins + tail
        return TextFieldValue(out, TextRange(head.length + ins.length))
    }

    /**
     * [line] on its own line after the cursor's (an instrumental's section plan: one bare tag a line, no blank lines -
     * the instrumental adapter's training form), the cursor after it.
     */
    fun insertLine(v: TextFieldValue, line: String): TextFieldValue {
        val text = v.text
        val c = v.selection.end.coerceIn(0, text.length)
        val lineEnd = text.indexOf('\n', c).let { if (it < 0) text.length else it }
        val head = text.substring(0, lineEnd).trimEnd()
        val tail = text.substring(lineEnd).trimStart('\n', '\r')
        val ins = (if (head.isEmpty()) "" else "\n") + line + (if (tail.isEmpty()) "" else "\n")
        return TextFieldValue(head + ins + tail, TextRange(head.length + ins.length - if (tail.isEmpty()) 0 else 1))
    }

    /**
     * The stress on the letter at the cursor: its case flipped - the selected letter, else the one before the cursor
     * (a capital inside a word splits YuE2's BPE token there: the stressed vowel as its own token, "ФраголІно"). A
     * second tap flips it back.
     */
    fun toggleStress(v: TextFieldValue): TextFieldValue {
        val text = v.text
        val s = v.selection
        val i = when {
            s.length == 1 -> s.min
            s.collapsed && s.start > 0 && text[s.start - 1].isLetter() -> s.start - 1
            s.collapsed && s.start < text.length && text[s.start].isLetter() -> s.start
            else -> return v
        }
        val ch = text[i]
        if (!ch.isLetter()) return v
        val flipped = if (ch.isUpperCase()) ch.lowercaseChar() else ch.uppercaseChar()
        return v.copy(text = text.substring(0, i) + flipped + text.substring(i + 1))
    }

    /** The style with [genre] added (a comma before it when there is a style already). */
    fun addGenre(style: String, genre: String): String {
        val s = style.trim().trimEnd(',')
        return if (s.isEmpty()) genre else "$s, $genre"
    }

    /** A preset's style, the language the current style names first kept in front of it (GUIDE.md: the language
     *  first, as there is no field for it); [language]: the settings' writer language, when the style names none. */
    fun applyPreset(current: String, preset: String, language: String): String {
        val first = current.substringBefore(',').trim()
        val named = LyricsWriter.LANGUAGES.firstOrNull { it.first.equals(first, ignoreCase = true) }?.first
            ?: language.takeIf { l -> LyricsWriter.LANGUAGES.any { it.first == l } }
        return if (named == null) preset else "$named, $preset"
    }
}
