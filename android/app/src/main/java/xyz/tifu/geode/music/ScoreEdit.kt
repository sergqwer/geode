package xyz.tifu.geode.music

import xyz.tifu.geode.music.AbcScore.Fraction

/** Score deletion ported from YuE2-ComfyUI's notation.without and helpers (Apache-2.0).
 * No model, tokenizer or Android dependencies. Parsed times are quarter notes. */
object ScoreEdit {
    private operator fun Fraction.minus(other: Fraction) = this + other * -1

    /** First [bar] bars with vocals clipped at [seam], then an interlude comment for the model.
     * Port of notation.interlude_head (Apache-2.0). [seam] is absolute quarter notes, not L: ticks. */
    fun interludeHead(score: String, bar: Int, seam: Fraction): String {
        val head = GoOn.head(score, bar)
        val sheet = EditText.read(head)
        val notes = sungBefore(sheet, seam)
        val kept = rebuildVocal(sheet, notes)
        val ending = if ("\r\n" in kept) "\r\n" else "\n"
        return kept.trimEnd('\r', '\n') + ending + "% interlude" + ending
    }

    private fun sungBefore(score: AbcScore.Score, seam: Fraction) = score.voices.getValue("Vocal").notes
        .filter { it.start < seam }.map { it.copy(duration = minOf(it.start + it.duration, seam) - it.start) }

    // Unlike groups(), model output may stop inside either voice of the final group.
    private fun writtenGroups(lines: List<String>): List<Group> = buildList {
        var at = 8
        while (at < lines.size) {
            val names = mutableListOf<String>()
            while (at < lines.size && lines[at].startsWith("% ")) names += EditText.strip(lines[at++].substring(2))
            val parts = linkedMapOf<String, Part>()
            for (name in listOf("Vocal", "Ins")) {
                if (lines.getOrNull(at) != "V: $name") return@buildList
                val head = mutableListOf(lines[at++])
                while (at < lines.size && (lines[at].startsWith("M:") || lines[at].startsWith("K:"))) head += lines[at++]
                val music = lines.getOrNull(at)?.trimEnd(EditText::space) ?: return@buildList
                if (!music.endsWith('|')) return@buildList
                parts[name] = Part(head, music)
                at++
            }
            add(Group(names, parts))
        }
    }

    private fun cutAt(source: List<Group>, cuts: List<Int>): List<Pair<Int, Group>> {
        val blocks = source.toMutableList()
        val starts = mutableListOf<Int>()
        var count = 0
        for (block in blocks) { starts += count; count += barCount(block.parts.getValue("Vocal").music) }
        for (bar in cuts.distinct().sorted()) {
            if (bar <= 0 || bar >= count || bar in starts) continue
            val i = starts.indexOfLast { it < bar }
            val split = splitGroup(blocks[i], bar - starts[i])
            blocks[i] = split[0]; blocks.add(i + 1, split[1]); starts.add(i + 1, bar)
        }
        return starts.zip(blocks)
    }

    /** Insert exactly [bars] complete generated bars, dropping their vocals and restoring the parent's singing.
     * Port of notation.interluded (Apache-2.0). [seam] is absolute quarter notes; a held note splits there.
     * Unfinished trailing groups are ignored; too few usable complete bars throw IllegalArgumentException. */
    /** Refuses a break before [bar] of [score] when a bar next to it changes key halfway through (notation.py
     *  KEY_CHANGE_BREAK): checked when the break is planned, so no score is written for one that cannot go in. */
    fun breakKeys(score: String, bar: Int) = keysAround(cutAt(groups(EditText.lines(EditText.read(score).text)), listOf(bar)), bar)

    private fun keysAround(old: List<Pair<Int, Group>>, bar: Int) {
        for ((start, block) in old) for (part in block.parts.values) {
            var at = start
            for (piece in part.music.dropLast(1).split('|')) {
                require("[K:" !in piece || at !in bar - 1..bar) {
                    "Bar ${at + 1} changes key halfway through, and a break next to it would split that change. Put the break before another section."
                }
                at += restBars(piece) ?: 1
            }
        }
    }

    fun interluded(score: String, written: String, bar: Int, bars: Int, seam: Fraction): String {
        val before = EditText.read(score)
        val count = before.voices.getValue("Vocal").bars.size
        require(bar > 0 && bar < count && bars > 0) {
            "A break of $bars bars cannot go before bar ${bar + 1} of a score of $count bars."
        }
        val bodies = EditText.lines(before.text)
        val old = cutAt(groups(bodies), listOf(bar))
        keysAround(old, bar)
        val ending = EditText.lines(before.text, true).first().substring(bodies.first().length).ifEmpty { "\n" }
        try {
            val new = cutAt(writtenGroups(EditText.lines(EditText.strip(written))), listOf(bar, bar + bars))
            val playing = new.filter { (start, _) -> start in bar until bar + bars }
            val got = playing.sumOf { (_, block) -> barCount(block.parts.getValue("Vocal").music) }
            require(got == bars && playing.firstOrNull()?.first == bar) {
                "the model wrote $got whole bars of the break where $bars were asked for"
            }
            val stretches = listOf(old.filter { it.first < bar }.map { it.second }, playing.map { it.second },
                old.filter { it.first >= bar }.map { it.second })
            val state = mutableMapOf("Vocal" to (bodies[2] to bodies[7]), "Ins" to (bodies[2] to bodies[7]))
            val laidText = buildString {
                bodies.take(8).forEach { append(it).append(ending) }
                for ((which, chosen) in stretches.withIndex()) for ((place, block) in chosen.withIndex()) {
                    val names = if (which == 1) { if (place == 0) listOf("interlude") else emptyList() }
                        else if (which == 2 && place == 0 && block.names.isEmpty())
                            listOfNotNull(AbcScore.sections(before.text).lastOrNull {
                                it.start <= before.voices.getValue("Vocal").bars[bar].start }?.name?.takeIf { it.isNotEmpty() })
                        else block.names
                    names.forEach { append("% ").append(it).append(ending) }
                    for (name in listOf("Vocal", "Ins")) {
                        val part = block.parts.getValue(name)
                        val head = part.head.toMutableList()
                        if (which == 2 && place == 0) {
                            val voice = before.voices.getValue(name)
                            val b = voice.bars[bar]
                            val wanted = listOf("M:${b.meter.numerator}/${b.meter.denominator}",
                                "K:" + voice.keys.last { it.start <= b.start }.name)
                            val have = state.getValue(name).let { listOf(it.first, it.second) }
                            for ((field, current) in wanted.zip(have))
                                if (field != current && head.drop(1).none { it.startsWith(field.take(2)) }) head.add(1, field)
                        }
                        var (meter, key) = state.getValue(name)
                        for (line in head.drop(1)) when {
                            line.startsWith("M:") -> meter = line
                            line.startsWith("K:") -> key = line
                        }
                        state[name] = meter to key
                        var music = part.music
                        if (which < 2 && place == chosen.lastIndex && music.trimEnd(EditText::space).endsWith("-|"))
                            music = music.trimEnd(EditText::space).dropLast(2) + "|"
                        head.forEach { append(it).append(ending) }
                        append(music).append(ending)
                    }
                }
            }
            val laid = EditText.read(laidText)
            val laidVoice = laid.voices.getValue("Vocal")
            val shift = laidVoice.bars[bar + bars].start - laidVoice.bars[bar].start
            val vocal = sungBefore(before, seam).toMutableList()
            for (n in before.voices.getValue("Vocal").notes) {
                if (n.start >= seam) vocal += n.copy(start = n.start + shift)
                else if (n.start + n.duration > seam)
                    vocal += n.copy(start = seam + shift, duration = n.start + n.duration - seam)
            }
            val result = rebuildVocal(laid, vocal)
            val after = EditText.read(result)
            require(after.voices.getValue("Vocal").bars.size == count + bars) { "the score has the wrong number of bars" }
            for (name in listOf("Vocal", "Ins")) {
                val a = before.voices.getValue(name); val b = after.voices.getValue(name)
                for ((i, was) in a.bars.withIndex()) {
                    val now = b.bars[if (i < bar) i else i + bars]
                    require(was.duration == now.duration && was.meter == now.meter &&
                        a.keys.last { it.start <= was.start }.name == b.keys.last { it.start <= now.start }.name) {
                        "bar ${i + 1} of the $name part changed its length or its key"
                    }
                }
            }
            require(before.bpm == after.bpm && before.unit == after.unit) { "the tempo or the note length changed" }
            return score.takeWhile(EditText::space) + EditText.strip(result) + score.takeLastWhile(EditText::space)
        } catch (e: IllegalArgumentException) {
            throw IllegalArgumentException("The break could not be written into the score: ${EditText.reason(e)}.\n\n" +
                "Nothing was changed. The score is as it was.", e)
        }
    }

    /**
     * [score] with the [part]'s ("Vocal" or "Ins") notes replaced by [notes] (the piano roll's edit): one note at a
     * time, on the score's L: ticks, inside its bars. The bars, meters, keys and chords stay; a note across a bar line
     * is tied. The result is read back (IllegalArgumentException with the reason otherwise).
     */
    fun withNotes(score: String, part: String, notes: List<AbcScore.Note>): String {
        val sheet = EditText.read(score)
        require(part in sheet.voices) { "The score has no $part part." }
        val sorted = notes.sortedBy { it.start }
        require(sorted.zipWithNext().none { (a, b) -> a.start + a.duration > b.start }) { "A part sings one note at a time." }
        val out = rebuildVocal(sheet, sorted, part)
        val ending = if ("\r\n" in score) "\r\n" else "\n"
        return if (score.endsWith("\n") && !out.endsWith("\n")) out + ending else out   // (its final newline kept)
    }

    // The reference's notation.write, restricted to replacing one part's notes (Vocal unless said) on an unchanged grid.
    // Keep clean pieces verbatim; expand compressed rests only when their bars need rewriting.
    private fun rebuildVocal(sheet: AbcScore.Score, notes: List<AbcScore.Note>, part: String = "Vocal"): String {
        val voice = sheet.voices.getValue(part)
        val lines = EditText.lines(sheet.text).toMutableList()
        fun inside(list: List<AbcScore.Note>, b: AbcScore.Bar) = list.filter {
            it.start < b.start + b.duration && it.start + it.duration > b.start
        }.map { Triple(it.start, it.pitch, it.duration) }.sortedWith(compareBy({ it.first }, { it.second }, { it.third }))
        val dirtyBars = voice.bars.indices.filter {
            inside(voice.notes, voice.bars[it]) != inside(notes, voice.bars[it])
        }.toMutableSet()
        val waiting = java.util.ArrayDeque(dirtyBars)
        while (waiting.isNotEmpty()) {
            val i = waiting.removeFirst()
            val b = voice.bars[i]
            for (n in voice.notes + notes) {
                if (n.start >= b.start + b.duration || n.start + n.duration <= b.start) continue
                for (next in listOf(i - 1, i + 1)) {
                    if (next !in voice.bars.indices || next in dirtyBars) continue
                    val neighbour = voice.bars[next]
                    if (n.start < neighbour.start + neighbour.duration && n.start + n.duration > neighbour.start) {
                        dirtyBars += next; waiting.addLast(next)
                    }
                }
            }
        }
        var at = 0
        for ((line, name) in sheet.musicLines) {
            if (name != part) continue
            val pieces = lines[line].dropLast(1).split('|').map { piece ->
                val size = restBars(piece) ?: 1
                val first = at; at += size
                val dirty = (first until at).any { it in dirtyBars }
                require(!dirty || "[K:" !in piece) {
                    "bar ${first + 1} changes key halfway through and cannot have its singing rewritten"
                }
                if (!dirty) piece else (first until at).joinToString("|") { i ->
                    val b = voice.bars[i]
                    require(voice.keys.none { it.start > b.start && it.start < b.start + b.duration }) {
                        "bar ${i + 1} changes key halfway through and cannot have its singing rewritten"
                    }
                    vocalBar(sheet, b, notes, part)
                }
            }
            lines[line] = pieces.joinToString("|") + "|"
        }
        val ending = if ("\r\n" in sheet.text) "\r\n" else "\n"
        val result = lines.joinToString(ending) + if (sheet.text.endsWith("\n")) ending else ""
        val after = EditText.read(result)
        fun triples(ns: List<AbcScore.Note>) = ns.map { Triple(it.start, it.pitch, it.duration) }
            .sortedWith(compareBy({ it.first }, { it.second }, { it.third }))
        require(triples(after.voices.getValue(part).notes) == triples(notes)) { "the notes did not come out where they were put" }
        require(after.voices.getValue(part).chords == voice.chords) { "the chords did not come out where they were put" }
        for (name in listOf("Vocal", "Ins")) {
            require(after.voices.getValue(name).bars == sheet.voices.getValue(name).bars &&
                after.voices.getValue(name).keys == sheet.voices.getValue(name).keys) { "the $name bars or key changes moved" }
        }
        return result
    }

    private fun vocalBar(sheet: AbcScore.Score, bar: AbcScore.Bar, notes: List<AbcScore.Note>, part: String = "Vocal"): String {
        val end = bar.start + bar.duration
        val singing = notes.filter { it.start < end && it.start + it.duration > bar.start }
        val chords = sheet.voices.getValue(part).chords.filter { it.start >= bar.start && it.start < end }
        if (singing.isEmpty() && chords.isEmpty()) return "Z"
        val edges = (listOf(bar.start, end) + singing.flatMap {
            listOf(maxOf(bar.start, it.start), minOf(end, it.start + it.duration))
        } + chords.map { it.start }).distinct().sorted()
        val lengths = listOf(48L, 32L, 24L, 16L, 12L, 8L, 6L, 4L, 3L, 2L, 1L)
        val keys = sheet.voices.getValue(part).keys
        val signature = AbcScore.keyAccidentals(keys.last { it.start <= bar.start }.name)
        val state = signature.toMutableMap()
        val marked = mutableMapOf<Char, MutableSet<Int>>()
        return buildString {
            for ((left, right) in edges.zipWithNext()) {
                chords.filter { it.start == left }.forEach { append('"').append(it.name).append('"') }
                val n = singing.firstOrNull { it.start <= left && it.start + it.duration > left }
                val units = (right - left) * (sheet.unit.denominator / 4)
                require(units.denominator == 1L) { "a sung note falls between the score's L: ticks" }
                var remaining = units.numerator
                var first = true
                while (remaining > 0) {
                    val take = lengths.first { it <= remaining }
                    remaining -= take
                    if (n == null) append('z') else {
                        val (letter, alteration, natural) = spell(n.pitch, keys.last { it.start <= n.start }.name)
                        val octave = (natural - AbcScore.NATURAL.getValue(letter)) / 12
                        if (left == n.start && first) {
                            if (state.getValue(letter) != alteration ||
                                (alteration != signature.getValue(letter) && octave !in marked[letter].orEmpty())) {
                                if (state.getValue(letter) != alteration) marked[letter] = mutableSetOf(octave)
                                else marked.getOrPut(letter) { mutableSetOf() }.add(octave)
                                state[letter] = alteration
                                append(mapOf(-2 to "__", -1 to "_", 0 to "=", 1 to "^", 2 to "^^").getValue(alteration))
                            }
                        }
                        if (octave >= 6) append(letter.lowercaseChar()).append("'".repeat(octave - 6))
                        else append(letter).append(",".repeat(5 - octave))
                    }
                    if (take != 1L) append(take)
                    if (n != null && (remaining > 0 || n.start + n.duration > right)) append('-')
                    first = false
                }
            }
        }
    }

    // notation._spell: prefer the key signature, then fewest marks, then the key's sharp/flat leaning.
    private fun spell(pitch: Int, key: String): Triple<Char, Int, Int> {
        val signature = AbcScore.keyAccidentals(key)
        val candidates = AbcScore.NATURAL.keys.flatMap { letter -> (-2..2).mapNotNull { alteration ->
            val natural = pitch - alteration
            if (Math.floorMod(natural - AbcScore.NATURAL.getValue(letter), 12) == 0)
                Triple(letter, alteration, natural) else null
        } }
        return candidates.minWith(compareBy<Triple<Char, Int, Int>>(
            { it.second != signature.getValue(it.first) }, { kotlin.math.abs(it.second) },
            { if (key.endsWith('m')) it.second < signature.getValue(it.first)
              else if (AbcScore.KEYS.getValue(key) < 0) it.second > 0 else it.second < 0 }))
    }
    /** Original bar stretches in their new order; [to] is an original-score boundary. */
    fun order(first: Int, stop: Int, to: Int, count: Int): List<Pair<Int, Int>> {
        require(first >= 0 && first < stop && stop <= count) {
            "Bars ${first + 1} to $stop are not bars of this score, which has $count."
        }
        require(to in 0..count) { "There is no bar line ${to + 1} to move bars to in a score of $count bars." }
        require(to !in first..stop) { "Bars ${first + 1} to $stop would be moved to where they already are." }
        return (if (to < first) listOf(0 to to, first to stop, to to first, stop to count)
            else listOf(0 to first, stop to to, first to stop, to to count)).filter { (a, b) -> a < b }
    }

    private data class Part(val head: List<String>, val music: String)
    private data class Group(val names: List<String>, val parts: Map<String, Part>)
    private fun groups(lines: List<String>): List<Group> = buildList {
        var at = 8
        while (at < lines.size) {
            val names = mutableListOf<String>()
            while (lines[at].startsWith("% ")) names += EditText.strip(lines[at++].substring(2))
            val parts = listOf("Vocal", "Ins").associateWith {
                val head = mutableListOf(lines[at++])
                while (lines[at].startsWith("M:") || lines[at].startsWith("K:")) head += lines[at++]
                Part(head, lines[at++])
            }
            add(Group(names, parts))
        }
    }
    private fun barCount(music: String) = music.dropLast(1).split('|').sumOf { restBars(it) ?: 1 }
    private fun splitMusic(music: String, at: Int): Pair<String, String> {
        val head = mutableListOf<String>(); val tail = mutableListOf<String>()
        var seen = 0
        for (piece in music.dropLast(1).split('|')) {
            val count = restBars(piece) ?: 1
            when {
                seen + count <= at -> head += piece
                seen >= at -> tail += piece
                else -> { head += rests(at - seen).first(); tail += rests(seen + count - at).first() }
            }
            seen += count
        }
        return (head.joinToString("|") + "|") to (tail.joinToString("|") + "|")
    }
    private fun splitGroup(group: Group, at: Int): List<Group> {
        val split = group.parts.mapValues { (_, p) -> splitMusic(p.music, at) }
        return listOf(Group(group.names, group.parts.mapValues { (name, p) -> Part(p.head, split.getValue(name).first) }),
            Group(emptyList(), group.parts.mapValues { (name, _) -> Part(listOf("V: $name"), split.getValue(name).second) }))
    }

    /** Whole-bar reorder ported from YuE2-ComfyUI's notation.order/moved (Apache-2.0).
     * Score-only edits may split a section; TrackEdit.move restricts audio moves to whole sections. */
    fun moved(score: String, first: Int, stop: Int, to: Int): String {
        val before = EditText.read(score)
        val count = before.voices.getValue("Vocal").bars.size
        val stretches = order(first, stop, to, count)
        val named = AbcScore.sections(before.text)
        // (bars before the first section's name would join whatever section they land after)
        require(stretches.drop(1).all { (a, _) ->
            named.lastOrNull { it.start <= before.voices.getValue("Vocal").bars[a].start }?.name.orEmpty().isNotEmpty()
        }) {
            "The song's opening has no section name, so it cannot be moved or have bars put before it."
        }
        val bodies = EditText.lines(before.text)
        val blocks = groups(bodies).toMutableList()
        val starts = mutableListOf<Int>()
        var at = 0
        for (block in blocks) { starts += at; at += barCount(block.parts.getValue("Vocal").music) }
        val low = minOf(first, to); val high = maxOf(stop, to)
        for ((i, block) in blocks.withIndex()) for (part in block.parts.values) {
            var bar = starts[i]
            for (piece in part.music.dropLast(1).split('|')) {
                require("[K:" !in piece || bar !in low until high) {
                    "Bar ${bar + 1} changes key halfway through, and moving the bars around it would carry that " +
                        "change somewhere else. Move the sections on either side of it instead."
                }
                bar += restBars(piece) ?: 1
            }
        }
        val ending = EditText.lines(before.text, true).first().substring(bodies.first().length).ifEmpty { "\n" }
        try {
            for (bar in listOf(first, stop, to).distinct().sorted()) {
                if (bar <= 0 || bar >= count || bar in starts) continue
                val i = starts.indexOfLast { it < bar }
                val split = splitGroup(blocks[i], bar - starts[i])
                blocks[i] = split[0]; blocks.add(i + 1, split[1]); starts.add(i + 1, bar)
            }
            val sizes = blocks.map { barCount(it.parts.getValue("Vocal").music) }
            val state = mutableMapOf("Vocal" to (bodies[2] to bodies[7]), "Ins" to (bodies[2] to bodies[7]))
            val sections = AbcScore.sections(before.text)
            val result = buildString {
                bodies.take(8).forEach { append(it).append(ending) }
                for ((a, b) in stretches) {
                    val chosen = blocks.indices.filter { starts[it] >= a && starts[it] + sizes[it] <= b }
                    for ((place, i) in chosen.withIndex()) {
                        val block = blocks[i]
                        val section = sections.lastOrNull { it.start <= before.voices.getValue("Vocal").bars[a].start }?.name.orEmpty()
                        val names = if (place == 0 && block.names.isEmpty() && section.isNotEmpty()) listOf(section) else block.names
                        names.forEach { append("% ").append(it).append(ending) }
                        for (name in listOf("Vocal", "Ins")) {
                            val part = block.parts.getValue(name)
                            val head = part.head.toMutableList()
                            if (place == 0) {
                                val voice = before.voices.getValue(name)
                                val bar = voice.bars[a]
                                val wanted = listOf("M:${bar.meter.numerator}/${bar.meter.denominator}",
                                    "K:" + voice.keys.last { it.start <= bar.start }.name)
                                val have = state.getValue(name).let { listOf(it.first, it.second) }
                                for ((field, current) in wanted.zip(have)) {
                                    if (field != current && head.drop(1).none { it.startsWith(field.take(2)) }) head.add(1, field)
                                }
                            }
                            var (meter, key) = state.getValue(name)
                            for (line in head.drop(1)) when {
                                line.startsWith("M:") -> meter = line
                                line.startsWith("K:") -> key = line
                            }
                            state[name] = meter to key
                            var music = part.music
                            if (place == chosen.lastIndex && music.trimEnd(EditText::space).endsWith("-|"))
                                music = music.trimEnd(EditText::space).dropLast(2) + "|"
                            head.forEach { append(it).append(ending) }
                            append(music).append(ending)
                        }
                    }
                }
            }
            val after = AbcScore.parse(EditText.strip(result))
            val taken = stretches.flatMap { (a, b) -> (a until b).toList() }
            for (name in listOf("Vocal", "Ins")) {
                val old = before.voices.getValue(name); val new = after.voices.getValue(name)
                require(new.bars.size == taken.size) { "the $name part has ${new.bars.size} bars where ${taken.size} should be left" }
                for ((i, was) in taken.withIndex()) {
                    val a = old.bars[was]; val b = new.bars[i]
                    require(a.duration == b.duration && a.meter == b.meter) { "bar ${i + 1} of the $name part changed its length" }
                    require(old.keys.last { it.start <= a.start }.name == new.keys.last { it.start <= b.start }.name) {
                        "bar ${i + 1} of the $name part changed its key"
                    }
                }
            }
            require(before.bpm == after.bpm && before.unit == after.unit) { "the tempo or the note length changed" }
            return score.takeWhile(EditText::space) + EditText.strip(result) + score.takeLastWhile(EditText::space)
        } catch (e: IllegalArgumentException) {
            throw IllegalArgumentException("The bars could not be moved in the score: ${EditText.reason(e)}.\n\n" +
                "Nothing was changed. The score is as it was before the move.", e)
        }
    }

    private val fullRest = Regex("Z([2-4])?")
    private fun restBars(piece: String) = fullRest.matchEntire(EditText.strip(piece))?.groupValues?.get(1)?.ifEmpty { "1" }?.toInt()
    private fun rests(bars: Int): List<String> = List(bars / 4) { "Z4" } +
        (if (bars % 4 == 0) emptyList() else listOf(if (bars % 4 == 1) "Z" else "Z${bars % 4}"))
    private fun folded(pieces: List<String>): List<String> = buildList {
        var run = 0
        for (piece in pieces) {
            val rest = restBars(piece)
            if (rest != null) run += rest
            else { addAll(rests(run)); run = 0; add(piece) }
        }
        addAll(rests(run))
    }
    private data class Piece(val text: String, val bars: IntRange)

    /** Remove score bars [first, stop) in both voices, preserving every retained bar's meter, length and key. */
    fun without(score: String, first: Int, stop: Int): String {
        val before = EditText.read(score)
        val count = before.voices.getValue("Vocal").bars.size
        require(first >= 0 && first < stop && stop <= count) {
            "Bars ${first + 1} to $stop are not bars of this score, which has $count."
        }
        require(stop - first < count) { "That is every bar of the score, and a cut has to leave some." }
        val lines = EditText.lines(before.text, true).toMutableList()
        val counters = mutableMapOf("Vocal" to 0, "Ins" to 0)
        val pieces = before.musicLines.mapValues { (i, name) ->
            val body = lines[i].trimEnd('\r', '\n')
            body.dropLast(1).split('|').map { text ->
                val at = counters.getValue(name)
                val n = restBars(text) ?: 1
                counters[name] = at + n
                Piece(text, at until at + n)
            }
        }
        val locked = pieces.values.flatten().filter { "[K:" in it.text }.flatMap { it.bars }
            .filter { it in first until stop }.minOrNull()
        require(locked == null) {
            "Bar ${locked!! + 1} changes key halfway through, and cutting it out would move the key " +
                "of everything after it. Cut up to that bar, or from the bar after it."
        }
        val gone = mutableSetOf<Int>()
        try {
            for ((index, found) in pieces) {
                val body = lines[index].trimEnd('\r', '\n')
                val kept = mutableListOf<String>()
                var changed = false
                for (piece in found) {
                    val left = piece.bars.count { it !in first until stop }
                    var written = piece.text
                    if (left < piece.bars.count()) {
                        changed = true
                        if (left == 0) continue
                        written = rests(left).first()
                    } else if (piece.bars.first == first - 1 && piece.bars.count() == 1 &&
                        written.trimEnd(EditText::space).endsWith('-')) {
                        changed = true
                        written = written.trimEnd(EditText::space).dropLast(1)
                    }
                    kept += written
                }
                if (!changed) continue
                if (kept.isEmpty()) {
                    gone += index
                    var cursor = index - 1
                    while (cursor >= 0 && (lines[cursor].startsWith("M:") || lines[cursor].startsWith("K:"))) gone += cursor--
                    require(cursor >= 0 && lines[cursor].startsWith("V: ")) { "a line of music has no V: line above it" }
                    gone += cursor
                } else lines[index] = folded(kept).joinToString("|") + "|" + lines[index].substring(body.length)
            }
            for (index in lines.indices) {
                if (!lines[index].startsWith("% ")) continue
                val under = mutableListOf<Int>()
                for (after in index + 1 until lines.size) {
                    if (lines[after].startsWith("% ")) {
                        if (under.isNotEmpty()) break
                        continue
                    }
                    if (after in before.musicLines) under += after
                }
                if (under.isNotEmpty() && under.all { it in gone }) gone += index
            }
            val result = lines.filterIndexed { i, _ -> i !in gone }.joinToString("")
            val after = AbcScore.parse(EditText.strip(result))
            for (name in listOf("Vocal", "Ins")) {
                val old = before.voices.getValue(name)
                val new = after.voices.getValue(name)
                val part = if (name == "Vocal") "sung" else "instrumental"
                val kept = old.bars.filterIndexed { i, _ -> i !in first until stop }
                require(new.bars.size == kept.size) { "the $part part has ${new.bars.size} bars where ${kept.size} should be left" }
                kept.zip(new.bars).forEachIndexed { i, (was, now) ->
                    require(was.duration == now.duration && was.meter == now.meter) { "bar ${i + 1} of the $part part changed its length" }
                    fun key(v: AbcScore.Voice, b: AbcScore.Bar) = v.keys.last { it.start <= b.start }.name
                    require(key(old, was) == key(new, now)) { "bar ${i + 1} of the $part part changed its key" }
                }
            }
            require(before.bpm == after.bpm && before.unit == after.unit) { "the tempo or the note length changed" }
            val leading = score.takeWhile(EditText::space)
            val trailing = score.takeLastWhile(EditText::space)
            return leading + EditText.strip(result) + trailing
        } catch (e: IllegalArgumentException) {
            throw IllegalArgumentException("The bars could not be cut out of the score: ${EditText.reason(e)}.\n\n" +
                "Nothing was changed. The score is as it was before the cut.", e)
        }
    }
}
