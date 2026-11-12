package xyz.tifu.geode.music

import kotlin.math.abs

/**
 * Reads and transposes YuE's native two-voice ABC dialect, checking the sounding score after each move.
 * Ported from YuE2-ComfyUI (github.com/pytraveler/YuE2-ComfyUI, Apache-2.0)
 * transpose.py and YuE's skills/yue2-music/scripts/abc_tools.py (Apache-2.0).
 */
object AbcScore {
    const val LIMIT = 12

    class AbcError(message: String) : IllegalArgumentException(message)

    val TOKEN = Regex(
        """"(?<chord>[^"\n]*)"|\[K:(?<key>[^\]\n]+)\]|(?<acc>\^\^|__|\^|_|=)?(?<note>[A-Ga-gz])(?<oct>[,']*)(?<duration>[0-9]*)(?<tie>-?)"""
    )
    val NATURAL: Map<Char, Int> = linkedMapOf('C' to 0, 'D' to 2, 'E' to 4, 'F' to 5, 'G' to 7, 'A' to 9, 'B' to 11)
    val KEYS: Map<String, Int> = linkedMapOf<String, Int>().apply {
        listOf("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
            .forEachIndexed { i, name -> put(name, i - 7) }
        listOf("Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m")
            .forEachIndexed { i, name -> put(name, i - 7) }
    }
    private val voices = listOf("Vocal", "Ins")
    private val durations = setOf(1L, 2L, 3L, 4L, 6L, 8L, 12L, 16L, 24L, 32L, 48L)
    private val qualities = listOf("", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5",
        "sus4", "sus2", "6", "m6", "7sus4", "m(maj7)")
    private const val PITCH_NAME = "[A-G](?:bb|##|b|#)?"
    private val chord = Regex(PITCH_NAME + "(?:" + qualities.joinToString("|") { Regex.escape(it) } + ")(?:/" + PITCH_NAME + ")?")
    private val meterPattern = Regex("([1-9][0-9]*)/([1-9][0-9]*)")
    private val unitPattern = Regex("L:1/([1-9][0-9]*)")
    private val tempoPattern = Regex("Q:1/4=([1-9][0-9]*)")
    private val restPattern = Regex("Z([2-4])?")
    private const val LETTERS = "CDEFGAB"
    private val marks = mapOf(-2 to "__", -1 to "_", 0 to "=", 1 to "^", 2 to "^^")
    private val spellings = mapOf(-2 to "bb", -1 to "b", 0 to "", 1 to "#", 2 to "##")
    private val nameShifts = mapOf("" to 0, "#" to 1, "##" to 2, "b" to -1, "bb" to -2)
    private val markShifts = mapOf("=" to 0, "_" to -1, "__" to -2, "^" to 1, "^^" to 2)
    private val keyName = Regex("([A-G])(#|b)?(m?)")
    private val chordName = Regex("([A-G])(bb|##|b|#)?(.*?)(?:/([A-G])(bb|##|b|#)?)?")

    // Times stay exact, including meters whose bars end between quarter notes.
    class Fraction(numerator: Long, denominator: Long = 1) : Comparable<Fraction> {
        val numerator: Long
        val denominator: Long

        init {
            if (denominator == 0L) throw AbcError("Fraction denominator is zero")
            var a = abs(numerator)
            var b = abs(denominator)
            while (b != 0L) {
                val remainder = a % b
                a = b
                b = remainder
            }
            val divisor = if (denominator < 0) -a else a
            this.numerator = numerator / divisor
            this.denominator = denominator / divisor
        }

        operator fun plus(other: Fraction) = Fraction(
            numerator * other.denominator + other.numerator * denominator, denominator * other.denominator
        )
        operator fun times(value: Long) = Fraction(numerator * value, denominator)
        override fun compareTo(other: Fraction) = (numerator * other.denominator).compareTo(other.numerator * denominator)
        override fun equals(other: Any?) = other is Fraction && numerator == other.numerator && denominator == other.denominator
        override fun hashCode() = 31 * numerator.hashCode() + denominator.hashCode()
        override fun toString() = if (denominator == 1L) "$numerator" else "$numerator/$denominator"
        fun toDouble(): Double = numerator.toDouble() / denominator

        /** This many quarter notes in MIDI ticks of [division] a quarter, rounded as Python's round() (half to even). */
        fun ticks(division: Int): Int {
            val n = numerator * division
            val q = Math.floorDiv(n, denominator)
            val twice = 2 * Math.floorMod(n, denominator)
            return (when {
                twice > denominator -> q + 1
                twice < denominator -> q
                else -> q + Math.floorMod(q, 2L)
            }).toInt()
        }
    }

    data class Meter(val numerator: Long, val denominator: Long)
    data class Note(val start: Fraction, val pitch: Int, var duration: Fraction)
    data class Bar(val start: Fraction, val duration: Fraction, val meter: Meter)
    data class NamedEvent(val start: Fraction, val name: String)
    data class Pending(val pitch: Int, val written: Int)

    class Voice(var meter: Meter, var key: String) {
        var time = Fraction(0)
        val notes = mutableListOf<Note>()
        val bars = mutableListOf<Bar>()
        val chords = mutableListOf<NamedEvent>()
        val keys = mutableListOf<NamedEvent>()
        var pending: Pending? = null
    }

    data class Score(
        val text: String,
        val unit: Fraction,
        val bpm: Int,
        val voices: Map<String, Voice>,
        val musicLines: Map<Int, String>
    )

    data class Moved(val text: String, val before: String, val after: String)

    private fun fail(condition: Boolean, message: String) {
        if (condition) throw AbcError(message)
    }

    // Python's whitespace and splitlines include separators that String.lines does not.
    private fun space(c: Char) = c in '\t'..'\r' || c in '\u001c'..' ' || c == '\u0085' ||
        c == '\u00a0' || c == '\u1680' || c in '\u2000'..'\u200a' || c == '\u2028' ||
        c == '\u2029' || c == '\u202f' || c == '\u205f' || c == '\u3000'

    private fun strip(text: String) = text.trim { space(it) }

    private fun splitLines(text: String, keepEnds: Boolean = false): List<String> {
        val result = mutableListOf<String>()
        var start = 0
        var cursor = 0
        while (cursor < text.length) {
            val c = text[cursor]
            if (c == '\n' || c == '\r' || c == '\u000b' || c == '\u000c' ||
                c in '\u001c'..'\u001e' || c == '\u0085' || c == '\u2028' || c == '\u2029') {
                val end = cursor
                cursor += if (c == '\r' && text.getOrNull(cursor + 1) == '\n') 2 else 1
                result += text.substring(start, if (keepEnds) cursor else end)
                start = cursor
            } else cursor++
        }
        if (start < text.length) result += text.substring(start)
        return result
    }

    // Parser errors use Python's repr, including its choice of quotes.
    private fun repr(text: String): String {
        val quote = if ('\'' in text && '"' !in text) '"' else '\''
        return buildString {
            append(quote)
            for (c in text) when (c) {
                quote -> { append('\\'); append(c) }
                '\\' -> append("\\\\")
                '\t' -> append("\\t")
                '\r' -> append("\\r")
                '\n' -> append("\\n")
                else -> {
                    val type = Character.getType(c)
                    if (c != ' ' && (Character.isISOControl(c) || type == Character.SPACE_SEPARATOR.toInt() ||
                            type == Character.LINE_SEPARATOR.toInt() || type == Character.PARAGRAPH_SEPARATOR.toInt() ||
                            type == Character.FORMAT.toInt() || type == Character.UNASSIGNED.toInt())) {
                        append(if (c.code <= 255) "\\x" else "\\u")
                        append(c.code.toString(16).padStart(if (c.code <= 255) 2 else 4, '0'))
                    } else append(c)
                }
            }
            append(quote)
        }
    }

    fun keyAccidentals(key: String): Map<Char, Int> {
        val count = KEYS[key] ?: throw AbcError("Unsupported key ${repr(key)}; use a standard major or minor K: field")
        val result = NATURAL.mapValues { 0 }.toMutableMap()
        for (letter in (if (count > 0) "FCGDAEB" else "BEADGCF").take(abs(count))) {
            result[letter] = if (count > 0) 1 else -1
        }
        return result
    }

    private fun meterValue(text: String): Meter {
        val match = meterPattern.matchEntire(text)
            ?: throw AbcError("Unsupported meter ${repr(text)}; write an explicit fraction")
        val n = match.groupValues[1].toLong()
        val d = match.groupValues[2].toLong()
        fail(d > 1024 || d and (d - 1) != 0L, "Unsupported meter denominator $d")
        return Meter(n, d)
    }

    private fun written(note: Char, octave: String): Int =
        60 + NATURAL.getValue(note.uppercaseChar()) + (if (note.isLowerCase()) 12 else 0) +
            12 * (octave.count { it == '\'' } - octave.count { it == ',' })

    private fun parseBar(body: String, voice: Voice, unit: Fraction, context: String) {
        val length = Fraction(4 * voice.meter.numerator, voice.meter.denominator)
        val start = voice.time
        var offset = Fraction(0)
        val local = mutableMapOf<Char, Int>()
        if (body == "Z") {
            fail(voice.pending != null, "$context: tie enters a full-measure rest")
            offset = length
        } else {
            var cursor = 0
            while (cursor < body.length) {
                if (space(body[cursor])) {
                    cursor++
                    continue
                }
                val match = TOKEN.matchAt(body, cursor)
                    ?: throw AbcError("$context: unsupported token at ${repr(body.substring(cursor).take(24))}")
                cursor = match.range.last + 1
                fail(offset >= length, "$context: event after the measure end")
                val chordText = match.groups["chord"]?.value
                val key = match.groups["key"]?.value
                if (chordText != null) {
                    fail(chord.matchEntire(chordText) == null, "$context: unsupported chord ${repr(chordText)}")
                    voice.chords += NamedEvent(start + offset, chordText)
                    continue
                }
                if (key != null) {
                    keyAccidentals(key)
                    voice.key = key
                    voice.keys += NamedEvent(start + offset, key)
                    local.clear()
                    continue
                }
                val note = match.groups["note"]!!.value[0]
                val acc = match.groups["acc"]?.value.orEmpty()
                val octave = match.groups["oct"]!!.value
                val tie = match.groups["tie"]!!.value.isNotEmpty()
                val units = match.groups["duration"]!!.value.ifEmpty { "1" }.toLong()
                fail(units !in durations, "$context: unsupported duration $units; split it into tied supported lengths")
                val duration = unit * units * 4
                fail(offset + duration > length, "$context: note/rest exceeds meter duration")
                fail(',' in octave && '\'' in octave, "$context: mixed octave marks")
                if (note == 'z') {
                    fail(acc.isNotEmpty() || octave.isNotEmpty() || tie, "$context: a rest cannot have accidentals, octave marks or ties")
                    fail(voice.pending != null, "$context: tie enters a rest")
                } else {
                    val letter = note.uppercaseChar()
                    val natural = written(note, octave)
                    var alteration = local[letter] ?: keyAccidentals(voice.key).getValue(letter)
                    if (acc.isNotEmpty()) {
                        alteration = markShifts.getValue(acc)
                        local[letter] = alteration
                    }
                    var pitch = natural + alteration
                    val pending = voice.pending
                    if (pending != null) {
                        // A tied continuation keeps its accidental without setting the bar's marks.
                        if (acc.isEmpty() && natural == pending.written) pitch = pending.pitch
                        fail(pitch != pending.pitch, "$context: tie changes pitch from ${pending.pitch} to $pitch")
                        voice.notes.last().duration = voice.notes.last().duration + duration
                    } else {
                        fail(pitch !in 0..127, "$context: pitch $pitch is outside MIDI range")
                        voice.notes += Note(start + offset, pitch, duration)
                    }
                    voice.pending = if (tie) Pending(pitch, natural) else null
                }
                offset += duration
            }
        }
        fail(offset != length, "$context: duration $offset quarter notes != meter duration $length")
        voice.bars += Bar(start, length, voice.meter)
        voice.time += length
    }

    fun parse(text: String): Score = abcErrors { parseScore(text) }

    private fun parseScore(text: String): Score {
        val lines = splitLines(text)
        fail(lines.size < 12, "Incomplete native two-voice ABC")
        fail(lines.take(2) != listOf("X:1", "T:"), "Expected native X:1 and blank T: header")
        fail(!lines[2].startsWith("M:"), "Missing header M:")
        val meter = meterValue(lines[2].substring(2))
        val unitMatch = unitPattern.matchEntire(lines[3])
            ?: throw AbcError("Expected L:1/<power of two>, usually L:1/32")
        val denominator = unitMatch.groupValues[1].toLong()
        fail(denominator > 1024 || denominator and (denominator - 1) != 0L, "Unsupported L: denominator")
        val unit = Fraction(1, denominator)
        val tempoMatch = tempoPattern.matchEntire(lines[4])
            ?: throw AbcError("Expected integer quarter-note tempo Q:1/4=<BPM>")
        fail(lines.subList(5, 7) != listOf(
            "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
            "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\""
        ), "Preserve native Vocal and Ins voice definitions")
        fail(!lines[7].startsWith("K:"), "Missing header K:")
        val key = lines[7].substring(2)
        keyAccidentals(key)
        val parts = voices.associateWith { Voice(meter, key).apply { keys += NamedEvent(Fraction(0), key) } }
        val musicLines = linkedMapOf<Int, String>()
        var cursor = 8
        var group = 0
        while (cursor < lines.size) {
            while (cursor < lines.size && lines[cursor].startsWith("% ")) cursor++
            fail(cursor == lines.size, "Dangling section comment without music")
            group++
            val counts = mutableListOf<Int>()
            for (name in voices) {
                val context = "group $group, $name"
                fail(cursor >= lines.size || lines[cursor] != "V: $name", "$context: expected V: $name")
                cursor++
                val voice = parts.getValue(name)
                val fields = mutableSetOf<String>()
                while (cursor < lines.size && (lines[cursor].startsWith("M:") || lines[cursor].startsWith("K:"))) {
                    val field = lines[cursor].substringBefore(':')
                    val value = lines[cursor].substringAfter(':')
                    fail(field in fields, "$context: duplicate $field: field")
                    fields += field
                    if (field == "M") voice.meter = meterValue(value)
                    else {
                        keyAccidentals(value)
                        voice.key = value
                        voice.keys += NamedEvent(voice.time, value)
                    }
                    cursor++
                }
                fail(cursor >= lines.size, "$context: missing music line")
                val line = lines[cursor]
                fail(!line.endsWith("|"), "$context: music line must end with a plain barline")
                musicLines[cursor] = name
                cursor++
                val bars = mutableListOf<String>()
                for (raw in line.dropLast(1).split('|')) {
                    val bar = strip(raw)
                    fail(bar.isEmpty(), "$context: empty measure or unsupported double/repeat barline")
                    val rest = restPattern.matchEntire(bar)
                    if (rest != null) repeat(rest.groups[1]?.value?.toInt() ?: 1) { bars += "Z" }
                    else bars += bar
                }
                fail(bars.size !in 1..4, "$context: expected 1–4 measures after expanding Z rests")
                counts += bars.size
                for (bar in bars) parseBar(bar, voice, unit, "$context, bar ${voice.bars.size + 1}")
            }
            fail(counts[0] != counts[1], "group $group: voices have different measure counts")
        }
        for ((name, voice) in parts) fail(voice.pending != null, "$name: unresolved tie at end of score")
        fail(parts.getValue("Ins").chords.isNotEmpty(), "Native chord symbols belong in Vocal, not Ins")
        fail(parts.getValue("Vocal").bars != parts.getValue("Ins").bars, "Voice meter/time grids differ")
        fail(parts.getValue("Vocal").keys != parts.getValue("Ins").keys, "Voice key-change timelines differ")
        return Score(text, unit, tempoMatch.groupValues[1].toInt(), parts, musicLines)
    }

    private data class KeyParts(val letter: Char, val alteration: Int, val mode: String)
    private fun keyParts(name: String): KeyParts {
        val match = keyName.matchEntire(strip(name))
            ?: throw AbcError("'${strip(name)}' is not a plain major or minor key")
        return KeyParts(match.groupValues[1][0], nameShifts.getValue(match.groupValues[2]), match.groupValues[3])
    }

    private fun pitchClass(letter: Char, accidental: String) = Math.floorMod(NATURAL.getValue(letter) + nameShifts.getValue(accidental), 12)

    private fun noteText(letter: Char, written: Int): String {
        val octave = Math.floorDiv(written - NATURAL.getValue(letter), 12)
        return if (octave >= 6) letter.lowercaseChar() + "'".repeat(octave - 6)
        else letter + ",".repeat(5 - octave)
    }

    private data class Spelling(val letter: Char, val alteration: Int)
    private data class SpelledNote(val letter: Char, val written: Int, val alteration: Int)

    private class Interval(val semitones: Int, val steps: Int) {
        fun name(letter: Char, alteration: Int): Spelling {
            val index = LETTERS.indexOf(letter) + steps
            val target = Math.floorMod(NATURAL.getValue(letter) + alteration + semitones, 12)
            for (offset in listOf(0, -1, 1)) {
                val newLetter = LETTERS[Math.floorMod(index + offset, 7)]
                val newAlteration = Math.floorMod(target - NATURAL.getValue(newLetter) + 6, 12) - 6
                if (abs(newAlteration) <= 2) return Spelling(newLetter, newAlteration)
            }
            throw AbcError("pitch class $target has no spelling")
        }

        fun note(letter: Char, written: Int, pitch: Int): SpelledNote {
            val octave = Math.floorDiv(written - NATURAL.getValue(letter), 12)
            val degree = LETTERS.indexOf(letter) + 7 * octave + steps
            val target = pitch + semitones
            for (offset in listOf(0, -1, 1)) {
                val moved = degree + offset
                val newLetter = LETTERS[Math.floorMod(moved, 7)]
                val newWritten = 12 * Math.floorDiv(moved, 7) + NATURAL.getValue(newLetter)
                if (abs(target - newWritten) <= 2) return SpelledNote(newLetter, newWritten, target - newWritten)
            }
            throw AbcError("MIDI pitch $target has no spelling")
        }

        fun key(name: String): String {
            val parts = keyParts(name)
            val moved = this.name(parts.letter, parts.alteration)
            val result = moved.letter + spellings.getValue(moved.alteration) + parts.mode
            fail(result !in KEYS, "the key ${strip(name)} would become $result, which the score format has no name for")
            return result
        }

        fun chord(text: String): String {
            val match = chordName.matchEntire(text)
                ?: throw AbcError("'$text' is not a chord symbol the score format knows")
            val root = name(match.groupValues[1][0], nameShifts.getValue(match.groupValues[2]))
            var moved = root.letter + spellings.getValue(root.alteration) + match.groupValues[3]
            if (match.groups[4] != null) {
                val bass = name(match.groupValues[4][0], nameShifts.getValue(match.groupValues[5]))
                moved += "/" + bass.letter + spellings.getValue(bass.alteration)
            }
            return moved
        }
    }

    private fun interval(key: String, semitones: Int): Interval {
        if (Math.floorMod(semitones, 12) == 0) return Interval(semitones, 7 * Math.floorDiv(semitones, 12))
        val parts = keyParts(key)
        val tonic = Math.floorMod(NATURAL.getValue(parts.letter) + parts.alteration + semitones, 12)
        // Python minimizes the entire tuple, then keeps the first step on an equal distance.
        val candidate = KEYS.mapNotNull { (name, count) ->
            val other = keyParts(name)
            if (other.mode == parts.mode && Math.floorMod(NATURAL.getValue(other.letter) + other.alteration, 12) == tonic)
                Triple(abs(count), -count, other.letter)
            else null
        }.minWithOrNull(compareBy<Triple<Int, Int, Char>> { it.first }.thenBy { it.second }.thenBy { it.third })
            ?: throw AbcError("min() arg is an empty sequence")
        val distance = Math.floorMod(LETTERS.indexOf(candidate.third) - LETTERS.indexOf(parts.letter), 7)
        val steps = listOf(distance - 7, distance, distance + 7).minBy { abs(it * 12 - semitones * 7) }
        return Interval(semitones, steps)
    }

    private data class Waiting(val pitch: Int, val written: Int, val newWritten: Int)
    private class State(var key: String, var tie: Waiting? = null)

    private fun moveBar(bar: String, state: State, intervals: (String) -> Interval): String {
        if (restPattern.matchEntire(strip(bar)) != null) return bar
        var sourceKey = state.key
        var interval = intervals(sourceKey)
        var targetKey = interval.key(sourceKey)
        val sourceMarks = mutableMapOf<Char, Int>()
        val targetMarks = mutableMapOf<Char, Int>()
        return buildString {
            var cursor = 0
            while (cursor < bar.length) {
                if (space(bar[cursor])) {
                    append(bar[cursor++])
                    continue
                }
                val match = TOKEN.matchAt(bar, cursor)
                    ?: throw AbcError("unsupported notation at ${repr(bar.substring(cursor).take(24))}")
                cursor = match.range.last + 1
                val chord = match.groups["chord"]?.value
                val key = match.groups["key"]?.value
                if (chord != null) {
                    append('"').append(interval.chord(chord)).append('"')
                    continue
                }
                if (key != null) {
                    sourceKey = key
                    state.key = key
                    interval = intervals(key)
                    targetKey = interval.key(key)
                    sourceMarks.clear()
                    targetMarks.clear()
                    append("[K:").append(targetKey).append(']')
                    continue
                }
                val note = match.groups["note"]!!.value[0]
                if (note == 'z') {
                    append(match.value)
                    continue
                }
                val mark = match.groups["acc"]?.value.orEmpty()
                val letter = note.uppercaseChar()
                val natural = written(note, match.groups["oct"]!!.value)
                var alteration = sourceMarks[letter] ?: keyAccidentals(sourceKey).getValue(letter)
                if (mark.isNotEmpty()) {
                    alteration = markShifts.getValue(mark)
                    sourceMarks[letter] = alteration
                }
                var pitch = natural + alteration
                val waiting = state.tie
                if (waiting != null && mark.isEmpty() && natural == waiting.written) pitch = waiting.pitch
                val moved = interval.note(letter, natural, pitch)
                val signature = keyAccidentals(targetKey)
                val newMark = if (waiting != null && moved.written == waiting.newWritten) ""
                else if ((targetMarks[moved.letter] ?: signature.getValue(moved.letter)) == moved.alteration) ""
                else {
                    targetMarks[moved.letter] = moved.alteration
                    marks.getValue(moved.alteration)
                }
                val tie = match.groups["tie"]!!.value
                append(newMark).append(noteText(moved.letter, moved.written)).append(match.groups["duration"]!!.value).append(tie)
                state.tie = if (tie.isNotEmpty()) Waiting(pitch, natural, moved.written) else null
            }
        }
    }

    private fun chordMoved(before: String, after: String, semitones: Int): Boolean {
        val old = chordName.matchEntire(before) ?: return false
        val new = chordName.matchEntire(after) ?: return false
        if (old.groupValues[3] != new.groupValues[3]) return false
        if (pitchClass(new.groupValues[1][0], new.groupValues[2]) !=
            Math.floorMod(pitchClass(old.groupValues[1][0], old.groupValues[2]) + semitones, 12)) return false
        if ((old.groups[4] == null) != (new.groups[4] == null)) return false
        return old.groups[4] == null || pitchClass(new.groupValues[4][0], new.groupValues[5]) ==
            Math.floorMod(pitchClass(old.groupValues[4][0], old.groupValues[5]) + semitones, 12)
    }

    private fun keyMoved(before: String, after: String, semitones: Int): Boolean {
        val old = keyParts(before)
        val new = keyParts(after)
        return old.mode == new.mode && Math.floorMod(NATURAL.getValue(new.letter) + new.alteration, 12) ==
            Math.floorMod(NATURAL.getValue(old.letter) + old.alteration + semitones, 12)
    }

    private fun check(source: Score, result: Score, semitones: Int) {
        fail(source.bpm != result.bpm || source.unit != result.unit, "the tempo or the note length changed")
        for (name in voices) {
            val before = source.voices.getValue(name)
            val after = result.voices.getValue(name)
            fail(before.bars != after.bars, "the bars of the $name part moved")
            fail(before.notes.size != after.notes.size || before.notes.zip(after.notes).any { (old, new) ->
                old.start != new.start || old.duration != new.duration || old.pitch + semitones != new.pitch
            }, "a note of the $name part did not move by exactly $semitones")
            fail(before.chords.size != after.chords.size || before.chords.zip(after.chords).any { (old, new) ->
                old.start != new.start || !chordMoved(old.name, new.name, semitones)
            }, "a chord of the $name part did not move with its notes")
            fail(before.keys.size != after.keys.size || before.keys.zip(after.keys).any { (old, new) ->
                old.start != new.start || !keyMoved(old.name, new.name, semitones)
            }, "a key change of the $name part did not move")
        }
    }

    private inline fun <T> abcErrors(block: () -> T): T = try {
        block()
    } catch (error: AbcError) {
        throw error
    } catch (error: IllegalArgumentException) {
        throw AbcError(error.message.orEmpty())
    } catch (error: NoSuchElementException) {
        throw AbcError(error.message.orEmpty())
    } catch (error: IndexOutOfBoundsException) {
        throw AbcError(error.message.orEmpty())
    } catch (error: ArithmeticException) {
        throw AbcError(error.message.orEmpty())
    }

    fun move(text: String, semitones: Int): Moved = abcErrors {
        fail(semitones !in -LIMIT..LIMIT, "'transpose' is $semitones, and it goes from -$LIMIT to $LIMIT.")
        val sourceText = strip(text)
        fail(sourceText.isEmpty(), "no score")
        val source = parse(sourceText)
        val lines = splitLines(sourceText, keepEnds = true)
        val header = lines[7].trimEnd('\r', '\n').substring(2)
        val chosen = mutableMapOf<String, Interval>()
        fun intervals(key: String) = chosen.getOrPut(key) { interval(key, semitones) }
        val states = voices.associateWith { State(header) }
        var voice: String? = null
        val output = buildString {
            for ((index, raw) in lines.withIndex()) {
                val body = raw.trimEnd('\r', '\n')
                val ending = raw.substring(body.length)
                when {
                    index == 7 -> append("K:").append(intervals(header).key(header)).append(ending)
                    index < 8 -> append(raw)
                    body.startsWith("V: ") -> {
                        voice = strip(body.substring(3))
                        append(raw)
                    }
                    body.startsWith("K:") -> {
                        val key = body.substring(2)
                        states.getValue(voice!!).key = key
                        append("K:").append(intervals(key).key(key)).append(ending)
                    }
                    index in source.musicLines -> {
                        val state = states.getValue(source.musicLines.getValue(index))
                        append(body.dropLast(1).split('|').joinToString("|") { moveBar(it, state, ::intervals) })
                        append('|').append(ending)
                    }
                    else -> append(raw)
                }
            }
        }
        check(source, parse(output), semitones)
        Moved(output, strip(header), intervals(header).key(header))
    }

    /**
     * The sections the score's comments name (its "% Verse" lines, the last of several before a group): each one's
     * start in quarter notes and its label; music before the first comment is a section labelled "" (sheetsage's
     * sections.py). AbcError when the score cannot be read.
     */
    fun sections(text: String): List<NamedEvent> = abcErrors {
        val score = parseScore(text)
        val bars = score.voices.getValue("Vocal").bars
        val found = mutableListOf<NamedEvent>()
        val pending = mutableListOf<String>()
        var barIndex = 0
        splitLines(text).forEachIndexed { i, line ->
            if (line.startsWith("% ")) {
                pending += strip(line.substring(2))
                return@forEachIndexed
            }
            if (score.musicLines[i] != "Vocal") return@forEachIndexed
            if (barIndex < bars.size && (pending.isNotEmpty() || found.isEmpty()))
                found += NamedEvent(bars[barIndex].start, pending.lastOrNull().orEmpty())
            pending.clear()
            barIndex += line.dropLast(1).split('|').sumOf { bar ->
                restPattern.matchEntire(strip(bar))?.let { it.groups[1]?.value?.toInt() ?: 1 } ?: 1
            }
        }
        found
    }

    fun key(text: String): String? = try {
        parse(text)
        strip(splitLines(text)[7].substring(2))
    } catch (_: AbcError) {
        null
    }
}
