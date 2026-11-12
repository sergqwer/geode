package xyz.tifu.geode.ui.music

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Undo
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.KeyboardArrowDown
import androidx.compose.material.icons.outlined.KeyboardArrowUp
import androidx.compose.material.icons.outlined.KeyboardDoubleArrowDown
import androidx.compose.material.icons.outlined.KeyboardDoubleArrowUp
import androidx.compose.material.icons.outlined.ZoomIn
import androidx.compose.material.icons.outlined.ZoomOut
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import xyz.tifu.geode.R
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
import xyz.tifu.geode.music.ScoreEdit
import kotlin.math.floor
import kotlin.math.roundToInt

private val NAMES = listOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
private fun nameOf(pitch: Int) = NAMES[Math.floorMod(pitch, 12)] + (Math.floorDiv(pitch, 12) - 1)
private fun black(pitch: Int) = Math.floorMod(pitch, 12) in setOf(1, 3, 6, 8, 10)

/**
 * A score's notes as a piano roll (YuE2-ComfyUI's score editor, in a phone's size): the voice or the instrument
 * edited, the other drawn faintly behind; bars and sections along the top. A tap selects a note or, on an empty place,
 * puts one of the grid's length there (a part sings one note at a time: a taken place is refused); a drag moves the
 * selected notes by the grid and by semitones, a drag elsewhere scrolls. The buttons move the selection by a semitone
 * or an octave, make it longer or shorter, delete it, undo. Apply writes the notes back into the score
 * (ScoreEdit.withNotes: the chords, the bars and the keys kept, the result read back).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PianoRollDialog(score: String, listening: Boolean, onListen: (String) -> Unit, onApply: (String) -> Unit,
                    onDismiss: () -> Unit) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize()) {
            val sheet = remember(score) { runCatching { AbcScore.parse(score.trim()) } }
            val parsed = sheet.getOrNull()
            if (parsed == null) {
                Column(Modifier.padding(16.dp)) {
                    Text(stringResource(R.string.music_score_unreadable, sheet.exceptionOrNull()?.message.orEmpty()),
                        color = MaterialTheme.colorScheme.error)
                    TextButton(onClick = onDismiss) { Text(stringResource(R.string.music_track_close)) }
                }
                return@Surface
            }
            Roll(score, parsed, listening, onListen, onApply, onDismiss)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Roll(score: String, parsed: AbcScore.Score, listening: Boolean, onListen: (String) -> Unit,
                 onApply: (String) -> Unit, onDismiss: () -> Unit) {
    val parts = listOf("Vocal", "Ins").filter { parsed.voices.containsKey(it) }
    var part by remember { mutableStateOf(parts.first()) }
    val edits = remember { mutableStateMapOf<String, List<AbcScore.Note>>() }
    fun notesOf(v: String): List<AbcScore.Note> = edits[v] ?: parsed.voices.getValue(v).notes.map { it.copy() }
    val undo = remember { mutableStateListOf<Pair<String, List<AbcScore.Note>>>() }
    var selected by remember { mutableStateOf(setOf<Int>()) }
    var step by remember { mutableStateOf(Fraction(1, 2)) }
    var zoom by remember { mutableFloatStateOf(1f) }
    var message by remember { mutableStateOf<String?>(null) }
    var scrollX by remember { mutableFloatStateOf(-1f) }
    var scrollY by remember { mutableFloatStateOf(-1f) }
    var drag by remember { mutableStateOf(Offset.Zero) }
    var moving by remember { mutableStateOf(false) }
    val taken = stringResource(R.string.music_roll_taken)
    val outside = stringResource(R.string.music_roll_outside)

    val voice = parsed.voices.getValue(part)
    val end = voice.time
    val all = parts.flatMap { notesOf(it) }
    val low = ((all.minOfOrNull { it.pitch } ?: 60) - 5).coerceAtLeast(21)
    val high = ((all.maxOfOrNull { it.pitch } ?: 72) + 5).coerceAtMost(108).coerceAtLeast(low + 24)
    val sections = remember(score) { runCatching { AbcScore.sections(score.trim()) }.getOrDefault(emptyList()) }

    /** The notes changed: the old ones kept for an undo, the selection set to [chosen]. */
    fun commit(notes: List<AbcScore.Note>, chosen: Set<Int> = emptySet()) {
        undo += part to notesOf(part)
        if (undo.size > 50) undo.removeAt(0)
        edits[part] = notes
        selected = chosen
    }
    /** [notes] with every one apart, inside the score and in the MIDI range; else the reason. */
    fun check(notes: List<AbcScore.Note>): String? {
        if (notes.any { it.start < Fraction(0) || it.start + it.duration > end || it.pitch !in 0..127 }) return outside
        val sorted = notes.sortedBy { it.start }
        return if (sorted.zipWithNext().any { (a, b) -> a.start + a.duration > b.start }) taken else null
    }
    fun change(f: (AbcScore.Note) -> AbcScore.Note) {
        if (selected.isEmpty()) return
        val notes = notesOf(part).mapIndexed { i, n -> if (i in selected) f(n.copy()) else n }
        check(notes)?.let { message = it } ?: run { message = null; commit(notes, selected) }
    }

    Column(Modifier.fillMaxSize().padding(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.music_roll), style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            TextButton(onClick = onDismiss) { Text(stringResource(R.string.music_track_close)) }
        }
        Text(stringResource(R.string.music_roll_hint), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically) {
            for (p in parts) FilterChip(selected = p == part, onClick = { part = p; selected = emptySet() }, label = {
                Text(stringResource(if (p == "Vocal") R.string.music_roll_voice else R.string.music_roll_ins))
            })
            Text(stringResource(R.string.music_roll_grid), style = MaterialTheme.typography.bodySmall)
            for ((label, f) in listOf("1/16" to Fraction(1, 4), "1/8" to Fraction(1, 2), "1/4" to Fraction(1)))
                FilterChip(selected = step == f, onClick = { step = f }, label = { Text(label) })
            IconButton(onClick = { zoom = (zoom / 1.4f).coerceAtLeast(0.25f) }) {
                Icon(Icons.Outlined.ZoomOut, stringResource(R.string.music_roll_zoom_out))
            }
            IconButton(onClick = { zoom = (zoom * 1.4f).coerceAtMost(4f) }) {
                Icon(Icons.Outlined.ZoomIn, stringResource(R.string.music_roll_zoom_in))
            }
        }
        val measurer = rememberTextMeasurer()
        val density = LocalDensity.current
        val rowH = with(density) { 16.dp.toPx() }
        val keyW = with(density) { 40.dp.toPx() }
        val topH = with(density) { 30.dp.toPx() }
        val perQuarter = with(density) { 48.dp.toPx() } * zoom
        val green = Color(0xFF3FA34D)
        val red = Color(0xFFD9534F)
        val faint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.18f)
        val line = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f)
        val weak = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)
        val text = MaterialTheme.colorScheme.onSurface
        val surface = MaterialTheme.colorScheme.surface
        val label = TextStyle(fontSize = 10.sp, color = text)
        val onNote = TextStyle(fontSize = 9.sp, color = Color.White)
        val notes = notesOf(part)

        fun quarterAt(x: Float) = (x - keyW + scrollX) / perQuarter
        fun pitchAt(y: Float) = high - floor((y - topH + scrollY) / rowH).toInt()
        fun hit(o: Offset): Int? {
            val q = quarterAt(o.x).toDouble()
            val p = pitchAt(o.y)
            return notes.indexOfFirst { it.pitch == p && it.start.toDouble() <= q && q < (it.start + it.duration).toDouble() }
                .takeIf { it >= 0 }
        }
        fun snap(q: Float): Fraction {
            val steps = floor(q / step.toDouble()).toLong()
            return Fraction(step.numerator * steps, step.denominator)
        }

        Canvas(Modifier.weight(1f).fillMaxWidth()
            .pointerInput(part, notes, step, zoom, selected) {
                detectTapGestures { o ->
                    if (o.x < keyW || o.y < topH) return@detectTapGestures
                    val i = hit(o)
                    if (i != null) { selected = if (i in selected) selected - i else selected + i; return@detectTapGestures }
                    if (selected.isNotEmpty()) { selected = emptySet(); return@detectTapGestures }
                    val added = notes + AbcScore.Note(snap(quarterAt(o.x)), pitchAt(o.y), step)
                    check(added)?.let { message = it } ?: run {
                        message = null
                        val sorted = added.sortedBy { it.start }
                        commit(sorted, setOf(sorted.indexOfFirst { it === added.last() }))
                    }
                }
            }
            .pointerInput(part, notes, step, zoom, selected) {
                detectDragGestures(
                    onDragStart = { o ->
                        val i = hit(o)
                        moving = i != null
                        if (i != null && i !in selected) selected = setOf(i)
                        drag = Offset.Zero
                    },
                    onDragEnd = {
                        if (moving) {
                            val dq = (drag.x / perQuarter / step.toDouble()).roundToInt()
                            val dp = -(drag.y / rowH).roundToInt()
                            val shift = Fraction(step.numerator * dq, step.denominator)
                            if (dq != 0 || dp != 0) change { AbcScore.Note(it.start + shift, it.pitch + dp, it.duration) }
                        }
                        moving = false; drag = Offset.Zero
                    },
                    onDragCancel = { moving = false; drag = Offset.Zero },
                ) { change, amount ->
                    change.consume()
                    if (moving) drag += amount else {
                        val width = keyW + end.toDouble().toFloat() * perQuarter
                        scrollX = (scrollX - amount.x).coerceIn(0f, maxOf(0f, width - size.width + rowH))
                        scrollY = (scrollY - amount.y).coerceIn(0f, maxOf(0f, (high - low + 1) * rowH + topH - size.height))
                    }
                }
            }) {
            if (scrollX < 0) {   // (first drawn: the part's first note in view - a song's intro may have no voice)
                val first = notes.minOfOrNull { it.start.toDouble() } ?: 0.0
                val width = keyW + end.toDouble().toFloat() * perQuarter
                scrollX = ((first - 1.0) * perQuarter).toFloat().coerceIn(0f, maxOf(0f, width - size.width + rowH))
            }
            if (scrollY < 0) {   // (first drawn: the notes' middle in view)
                val mid = notes.map { it.pitch }.average().takeIf { !it.isNaN() } ?: 66.0
                scrollY = ((high - mid) * rowH - (size.height - topH) / 2).toFloat()
                    .coerceIn(0f, maxOf(0f, (high - low + 1) * rowH + topH - size.height))
            }
            val x0 = keyW - scrollX
            for (p in low..high) {
                val y = topH + (high - p) * rowH - scrollY
                if (y + rowH < topH || y > size.height) continue
                if (black(p)) drawRect(weak, Offset(keyW, y), Size(size.width - keyW, rowH))
            }
            // the grid, the bars and the sections
            val stepPx = step.toDouble().toFloat() * perQuarter
            if (stepPx >= 6f) {
                var g = 0f
                while (g <= end.toDouble().toFloat() * perQuarter) {
                    val x = x0 + g
                    if (x in keyW..size.width) drawLine(weak, Offset(x, topH), Offset(x, size.height))
                    g += stepPx
                }
            }
            for ((i, b) in voice.bars.withIndex()) {
                val x = x0 + b.start.toDouble().toFloat() * perQuarter
                if (x < keyW - perQuarter * 8 || x > size.width) continue
                if (x >= keyW) drawLine(line, Offset(x, topH * 0.5f), Offset(x, size.height), strokeWidth = 1.5f)
                if (x >= keyW) drawText(measurer, "${i + 1}", Offset(x + 2f, topH * 0.5f), label)
            }
            for (s in sections) {
                val x = x0 + s.start.toDouble().toFloat() * perQuarter
                if (x >= keyW && x < size.width && s.name.isNotEmpty()) drawText(measurer, s.name, Offset(x + 2f, 0f), label)
            }
            // the other part faintly, then this one's notes (the selection red, moved as it is dragged)
            val dq = if (moving) (drag.x / perQuarter / step.toDouble()).roundToInt() * step.toDouble().toFloat() else 0f
            val dp = if (moving) -(drag.y / rowH).roundToInt() else 0
            fun note(n: AbcScore.Note, color: Color, shiftQ: Float = 0f, shiftP: Int = 0, named: Boolean = false) {
                val x = x0 + (n.start.toDouble().toFloat() + shiftQ) * perQuarter
                val w = n.duration.toDouble().toFloat() * perQuarter
                val y = topH + (high - n.pitch - shiftP) * rowH - scrollY
                if (x + w < keyW || x > size.width || y + rowH < topH || y > size.height) return
                drawRoundRect(color, Offset(maxOf(x, keyW), y + 1f), Size(minOf(x + w, size.width) - maxOf(x, keyW) - 1f, rowH - 2f),
                    CornerRadius(3f, 3f))
                if (named && w > 28f && x >= keyW) drawText(measurer, nameOf(n.pitch + shiftP), Offset(x + 3f, y + 1f), onNote)
            }
            for (p in parts) if (p != part) notesOf(p).forEach { note(it, faint) }
            notes.forEachIndexed { i, n ->
                if (i in selected) note(n, red, dq, dp, true) else note(n, green, named = true)
            }
            // the keyboard down the side
            drawRect(surface, Offset(0f, topH), Size(keyW, size.height - topH))
            for (p in low..high) {
                val y = topH + (high - p) * rowH - scrollY
                if (y + rowH < topH || y > size.height) continue
                drawRect(if (black(p)) Color(0xFF303030) else Color(0xFFEDEDED), Offset(0f, y), Size(keyW - 2f, rowH - 1f))
                if (Math.floorMod(p, 12) == 0) drawText(measurer, nameOf(p), Offset(2f, y + 1f), TextStyle(fontSize = 9.sp, color = Color.Black))
            }
        }
        message?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
        // the voice moved: its middle (the median pitch) out of where the model's own scores keep it, C4 .. A#5
        // (yue2_roll.js VOICE_WINDOW 60..82, voiceMiddle; warned only after a change, as there)
        if (part == "Vocal" && edits[part] != null && notes.isNotEmpty()) {
            val sorted = notes.map { it.pitch }.sorted()
            val middle = (sorted[(sorted.size - 1) / 2] + sorted[sorted.size / 2]) / 2.0
            if (middle < 60 || middle > 82) Text(stringResource(R.string.music_roll_range), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary)
        }
        Row(Modifier.horizontalScroll(rememberScrollState()), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { change { AbcScore.Note(it.start, it.pitch - 12, it.duration) } }, enabled = selected.isNotEmpty()) {
                Icon(Icons.Outlined.KeyboardDoubleArrowDown, stringResource(R.string.music_roll_octave_down))
            }
            IconButton(onClick = { change { AbcScore.Note(it.start, it.pitch - 1, it.duration) } }, enabled = selected.isNotEmpty()) {
                Icon(Icons.Outlined.KeyboardArrowDown, stringResource(R.string.music_roll_down))
            }
            IconButton(onClick = { change { AbcScore.Note(it.start, it.pitch + 1, it.duration) } }, enabled = selected.isNotEmpty()) {
                Icon(Icons.Outlined.KeyboardArrowUp, stringResource(R.string.music_roll_up))
            }
            IconButton(onClick = { change { AbcScore.Note(it.start, it.pitch + 12, it.duration) } }, enabled = selected.isNotEmpty()) {
                Icon(Icons.Outlined.KeyboardDoubleArrowUp, stringResource(R.string.music_roll_octave_up))
            }
            TextButton(onClick = { change { n -> AbcScore.Note(n.start, n.pitch, n.duration + step) } }, enabled = selected.isNotEmpty()) {
                Text(stringResource(R.string.music_roll_longer))
            }
            TextButton(onClick = {
                change { n -> if (n.duration > step) AbcScore.Note(n.start, n.pitch, n.duration + step * -1) else n }
            }, enabled = selected.isNotEmpty()) { Text(stringResource(R.string.music_roll_shorter)) }
            IconButton(onClick = { commit(notes.filterIndexed { i, _ -> i !in selected }) }, enabled = selected.isNotEmpty()) {
                Icon(Icons.Outlined.Delete, stringResource(R.string.music_roll_delete))
            }
            IconButton(onClick = {
                undo.removeLastOrNull()?.let { (v, n) -> edits[v] = n; part = v; selected = emptySet(); message = null }
            }, enabled = undo.isNotEmpty()) { Icon(Icons.AutoMirrored.Outlined.Undo, stringResource(R.string.music_roll_undo)) }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            fun written(): String? = runCatching {
                var t = score
                for ((v, n) in edits) t = ScoreEdit.withNotes(t, v, n)
                t
            }.onFailure { message = it.message ?: it.javaClass.simpleName }.getOrNull()
            TextButton(onClick = { written()?.let(onListen) }) {
                Text(stringResource(if (listening) R.string.music_score_stop else R.string.music_score_listen))
            }
            TextButton(onClick = { written()?.let { onApply(it); onDismiss() } }, enabled = edits.isNotEmpty()) {
                Text(stringResource(R.string.music_roll_apply))
            }
        }
    }
}
