/** Timed words and section markers from YuE2-ComfyUI's midi/karaoke.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import xyz.tifu.geode.music.YuE2Prompt
import kotlin.math.abs

object Karaoke {
    data class Section(val tick: Int, var tag: String, val lines: List<String>) { val label get() = tag.lowercase(java.util.Locale.ROOT) }
    data class Words(val sections: List<Section>, val syllables: List<Int>) {
        fun lyrics() = sections.joinToString("\n\n") { "[${it.tag}]\n" + it.lines.joinToString("\n") }
    }
    // SheetSage's larger label vocabulary maps onto the prompt's existing tag names.
    internal val tags = YuE2Prompt.TAGS.associateBy { it.lowercase(java.util.Locale.ROOT) } + mapOf(
        "post-chorus" to "Chorus", "rap" to "Verse", "solo" to "Instrumental", "fade-out" to "Outro",
        "pre-outro" to "Bridge", "loop" to "Chorus", "intro and verse" to "Verse", "pre-chorus and chorus" to "Chorus",
        "verse and pre-chorus" to "Verse", "theme" to "Verse", "development" to "Bridge", "variation" to "Verse",
        "irregular" to "Verse", "preshot" to "Intro", "silence" to "Intro")
    fun sectionLabel(text: String): String? {
        val clean = MidiParts.fold(text.lowercase(java.util.Locale.ROOT).replace('_', ' '))
        return tags.keys.sortedByDescending { it.length }.firstOrNull { Regex(Regex.escape(it) + "(?:[\\s\\d:.-]|$)").containsMatchIn(clean) && clean.startsWith(it) }
    }
    fun markerSections(song: MidiReader.Song) = song.markers.mapNotNull { m -> sectionLabel(MidiParts.decode(m.bytes))?.let { m.tick to it } }
    fun read(song: MidiReader.Song): Words? {
        fun usable(t: MidiReader.Text) = t.bytes.firstOrNull() != '@'.code.toByte()
        val lyric = song.tracks.flatMap { it.texts }.filter { it.kind == 5 && usable(it) }
        val source = if (lyric.size >= 8) lyric.sortedBy { it.tick } else
            song.tracks.map { it.texts.filter { e -> e.kind == 1 && usable(e) } }.maxByOrNull { it.size }.orEmpty().let { if (it.size >= 8) it else emptyList() }
        if (source.isEmpty()) return null
        val encoding = MidiParts.encodingOf(source.flatMap { it.bytes.toList() + listOf(32.toByte()) }.toByteArray())
        val sections = mutableListOf<Section>(); var lines = mutableListOf<String>(); var line = ""; var start: Int? = null
        fun endSection() {
            val text = MidiParts.fold(line); line = ""
            if (text.isNotEmpty()) lines += text
            if (lines.isNotEmpty()) sections += Section(start ?: 0, "Verse", lines.toList())
            lines = mutableListOf(); start = null
        }
        fun endLine() {
            val text = MidiParts.fold(line); line = ""
            if (text.isNotEmpty()) lines += text else if (lines.isNotEmpty()) endSection()
        }
        fun add(tick: Int, text: String) { if (text.isNotBlank() && start == null) start = tick; line += text }
        var previous: Int? = null
        for (event in source) {
            var piece = MidiParts.textOf(event.bytes, encoding)
            if (previous != null && event.tick.toLong() - previous >= 8L * song.division && (lines.isNotEmpty() || line.isNotBlank())) endSection()
            previous = event.tick
            if (piece.startsWith("\\")) { endSection(); piece = piece.drop(1) }
            else if (piece.startsWith("/")) { endLine(); piece = piece.drop(1) }
            val split = piece.split(Regex("\r\n|\r|\n")); add(event.tick, split[0])
            for (rest in split.drop(1)) { endLine(); add(event.tick, rest) }
        }
        endSection()
        if (sections.isEmpty()) return null
        val keys = sections.map { it.lines.joinToString(" ").lowercase(java.util.Locale.ROOT).replace(Regex("[^\\p{L}\\p{N}_]+"), "") }
        for (i in keys.indices) for (j in 0 until i) if (keys[i].isNotEmpty() && keys[i] == keys[j]) { sections[i].tag = "Chorus"; sections[j].tag = "Chorus" }
        val markers = markerSections(song)
        for (section in sections) markers.filter { abs(it.first.toLong() - section.tick) <= 4L * song.division }
            .minWithOrNull(compareBy({ abs(it.first.toLong() - section.tick) }, { it.second }))?.let { section.tag = tags.getValue(it.second) }
        return Words(sections, source.filter { MidiParts.textOf(it.bytes, encoding).trim(' ', '\\', '/', '\r', '\n').isNotEmpty() }.map { it.tick })
    }
}
