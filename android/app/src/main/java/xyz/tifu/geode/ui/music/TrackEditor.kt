package xyz.tifu.geode.ui.music

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.delay
import xyz.tifu.geode.R
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.BarGrid
import xyz.tifu.geode.music.EditText
import xyz.tifu.geode.music.GoOn
import xyz.tifu.geode.music.Song
import xyz.tifu.geode.music.TrackEdit
import xyz.tifu.geode.music.Words
import xyz.tifu.geode.music.YuE2Prompt
import kotlin.math.floor
import kotlin.math.roundToInt

/** Seconds as "1:05.3". */
private fun clock(seconds: Double): String {
    val t = seconds.coerceAtLeast(0.0)
    val m = floor(t / 60).toInt()
    return "%d:%04.1f".format(m, t - 60 * m)
}

/**
 * The track editor (YuE2-ComfyUI's Edit Track, its Cut and Retake): the song's waveform with its score's bars on it -
 * estimated from the score (BarGrid), so shown as such - a stretch chosen by its bars (a tap takes one, a drag runs
 * over several; by time when the song has no score), heard, and cut out or sung again into a new song beside it. A cut
 * at the start or the end is a trim with a fade, made at once; one inside has its seam sung again; a retake sings the
 * stretch again with the same words - the editor estimates both from what the song's own run took.
 */
@Composable
fun TrackEditorDialog(vm: MusicViewModel, song: Song, onDismiss: () -> Unit) {
    var track by remember(song.id) { mutableStateOf<MusicViewModel.Track?>(null) }
    LaunchedEffect(song.id) { track = vm.track(song) }
    val playing by vm.playing.collectAsStateWithLifecycle()
    val hearing = playing.first == MusicViewModel.RANGE + song.id
    val total = song.seconds
    val frames = (total * YuE2Prompt.FRAMES_PER_SECOND).roundToInt()
    // the selection: bars [a, b] of the grid, or seconds [a, b] without one
    var a by remember(song.id) { mutableStateOf<Double?>(null) }
    var b by remember(song.id) { mutableStateOf<Double?>(null) }
    var width by remember { mutableIntStateOf(1) }
    var playhead by remember { mutableStateOf(-1.0) }
    var goText by remember(song.id) { mutableStateOf("") }
    // the lines of the lyrics chosen for new words (l0 tapped first, l1 second) and what they become
    var l0 by remember(song.id) { mutableStateOf<Int?>(null) }
    var l1 by remember(song.id) { mutableStateOf<Int?>(null) }
    var wordsText by remember(song.id) { mutableStateOf("") }
    LaunchedEffect(hearing) {
        while (hearing) {
            playhead = vm.position().first / 1000.0
            delay(50)
        }
        playhead = -1.0
    }
    Dialog(onDismissRequest = { vm.stop(); onDismiss() }, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize()) {
            Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(stringResource(R.string.music_track_edit), style = MaterialTheme.typography.titleLarge)
                Text(song.title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                val t = track
                if (t == null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.padding(end = 12.dp))
                        Text(stringResource(R.string.music_track_reading))
                    }
                    return@Column
                }
                val grid = t.grid
                val bars = grid?.bars
                Text(stringResource(if (grid != null) R.string.music_track_hint_bars else R.string.music_track_hint_time),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                // the bar at a second (the last line at or before it), and a bar's span in seconds within the song
                fun barAt(sec: Double): Int = bars!!.indices.last { it == 0 || bars[it] <= sec }.coerceAtMost(bars.size - 2)
                fun span(lo: Double, hi: Double): Pair<Double, Double> = if (bars != null) {
                    val i = minOf(lo, hi).toInt()
                    val j = maxOf(lo, hi).toInt()
                    bars[i].coerceIn(0.0, total) to bars[j + 1].coerceIn(0.0, total)
                } else minOf(lo, hi).coerceIn(0.0, total) to maxOf(lo, hi).coerceIn(0.0, total)
                fun pick(x: Float): Double {
                    val sec = x / width * total
                    return if (bars != null) barAt(sec).toDouble() else sec
                }
                val measurer = rememberTextMeasurer()
                val wave = MaterialTheme.colorScheme.onSurfaceVariant
                val chosen = MaterialTheme.colorScheme.primary
                val lines = MaterialTheme.colorScheme.outline
                val going = MaterialTheme.colorScheme.tertiary
                val label = TextStyle(fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                val section = TextStyle(fontSize = 11.sp, color = MaterialTheme.colorScheme.primary)
                val sel = a?.let { lo -> span(lo, b ?: lo) }
                // whether the grid's bars are the score's (not an older cut song's, whose score still holds the bars it
                // lost): then a cut takes its bars out of the score and the lyrics, and the song can go on
                val tempo = grid?.let { GoOn.Clock(it.bars, it.speed) }
                val paired = remember(t) {
                    runCatching { AbcScore.parse(t.score!!.trim()).voices["Vocal"]?.bars?.size }.getOrNull()
                        ?.let { tempo != null && it + 1 == tempo.lines.size } == true
                }
                // where the song can go on from (GoOn.goingOnFrom)
                val from = remember(t) {
                    if (!paired) null else runCatching { GoOn.goingOnFrom(t.score!!, tempo!!, frames) }.getOrNull()
                }
                Box(Modifier.fillMaxWidth().height(160.dp)) {
                    Canvas(
                        Modifier.fillMaxSize()
                            .pointerInput(t) { detectTapGestures { p -> pick(p.x).let { a = it; b = it } } }
                            .pointerInput(t) {
                                detectHorizontalDragGestures(
                                    onDragStart = { p -> pick(p.x).let { a = it; b = it } },
                                    onHorizontalDrag = { change, _ -> b = pick(change.position.x.coerceIn(0f, width.toFloat())) },
                                )
                            }
                    ) {
                        width = size.width.toInt().coerceAtLeast(1)
                        val top = 18.dp.toPx()
                        val h = size.height - top - 14.dp.toPx()
                        val mid = top + h / 2
                        fun x(sec: Double) = (sec / total * size.width).toFloat()
                        sel?.let { (lo, hi) -> drawRect(chosen.copy(alpha = 0.22f), Offset(x(lo), 0f), Size(x(hi) - x(lo), size.height)) }
                        t.view?.let { v ->   // the waveform: each column's loudest hop
                            val cols = size.width.toInt()
                            for (c in 0 until cols) {
                                val h0 = (c.toDouble() / cols * v.hops).toInt()
                                val h1 = maxOf(h0 + 1, ((c + 1).toDouble() / cols * v.hops).toInt()).coerceAtMost(v.hops)
                                var pk = 0f
                                for (k in h0 until h1) pk = maxOf(pk, v.peak(k))
                                val half = pk * h / 2
                                drawLine(wave, Offset(c + 0.5f, mid - half), Offset(c + 0.5f, mid + half), 1f)
                            }
                        } ?: drawLine(wave, Offset(0f, mid), Offset(size.width, mid), 1f)
                        if (bars != null && grid != null) {   // the bars (dashed: estimated), every number that fits
                            val spacing = if (bars.size > 1) (x(bars.last()) - x(bars.first())) / (bars.size - 1) else size.width
                            var every = 1
                            while (spacing * every < 22.dp.toPx() && every < bars.size) every *= 2
                            for (i in bars.indices) {
                                val bx = x(bars[i])
                                if (bx < 0 || bx > size.width) continue
                                drawLine(lines, Offset(bx, top), Offset(bx, size.height - 12.dp.toPx()), 1f,
                                    pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(6f, 4f)))
                                if (i < bars.size - 1 && i % every == 0)
                                    drawText(measurer, "${i + 1}", Offset(bx + 2f, size.height - 12.dp.toPx()), label)
                            }
                            for ((bar, name) in grid.sections) {
                                val sx = x(bars[bar])
                                if (name.isNotEmpty() && sx >= 0 && sx < size.width) drawText(measurer, name, Offset(maxOf(0f, sx) + 2f, 0f), section)
                            }
                        }
                        if (playhead >= 0) drawLine(chosen, Offset(x(playhead), 0f), Offset(x(playhead), size.height), 2f)
                        from?.let { f ->   // (where a song going on starts: solid, the go-on colour)
                            val fx = x(f.start / 25.0)
                            drawLine(going, Offset(fx, top), Offset(fx, size.height - 12.dp.toPx()), 2f)
                        }
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(clock(0.0), style = MaterialTheme.typography.labelSmall)
                    Text(clock(total), style = MaterialTheme.typography.labelSmall)
                }
                // what is chosen (its frames: within 0.2 s of an end, that end - a trim), and what cutting it or singing
                // it again does
                val stretch = sel?.let { (lo, hi) ->
                    var first = (lo * YuE2Prompt.FRAMES_PER_SECOND).roundToInt()
                    var stop = (hi * YuE2Prompt.FRAMES_PER_SECOND).roundToInt()
                    if (first <= 5) first = 0
                    if (stop >= frames - 5) stop = frames
                    (first to stop).takeIf { stop - first >= 1 }
                }
                val cut = stretch?.takeIf { (f, s) -> frames - (s - f) >= YuE2Prompt.FRAMES_PER_SECOND }
                // a cut by bars (TrackEdit.cut): the score and the lyrics without them, the seam drawn under those; a
                // score that cannot lose them: no cut, and why (an unpaired or unscored song: the cut by time as before)
                val barsChosen = if (bars != null && a != null) minOf(a!!, b ?: a!!).toInt() to maxOf(a!!, b ?: a!!).toInt() + 1 else null
                val cutPlan = remember(t, cut, barsChosen) {
                    if (cut == null || barsChosen == null || !paired) null
                    else runCatching { TrackEdit.cut(t.score!!, song.lyrics, grid!!, barsChosen.first, barsChosen.second, cut.first, cut.second, frames) }
                }
                // (shown, heard, cut and sung again: the stretch's frames, its ends at the song's at an end)
                val shown = stretch?.let { (f, s) -> f / 25.0 to minOf(total, s / 25.0) } ?: sel
                if (sel != null && shown != null) {
                    val (lo, hi) = shown
                    val range = "${clock(lo)} – ${clock(hi)} (${"%.1f".format(hi - lo)} s)"
                    Text(if (bars != null && a != null) {
                        val i = minOf(a!!, b ?: a!!).toInt() + 1
                        val j = maxOf(a!!, b ?: a!!).toInt() + 1
                        stringResource(R.string.music_track_bars, i, j) + " · " + range
                    } else range, style = MaterialTheme.typography.bodyMedium)
                    if (cut != null) {
                        val trim = cut.first == 0 || cut.second == frames
                        Text(if (trim) stringResource(R.string.music_track_trim_note)
                            else stringResource(R.string.music_track_seam_note, duration(seamSeconds(song, frames - (cut.second - cut.first)))),
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        val untagged = stringResource(R.string.music_track_cut_untagged)
                        cutPlan?.onSuccess { plan ->
                            for (n in plan.notices) Text(when (n) {
                                is TrackEdit.Notice.Dropped -> stringResource(R.string.music_track_cut_dropped,
                                    n.tags.joinToString(", ") { if (it.isEmpty()) untagged else "[$it]" })
                                is TrackEdit.Notice.Tail -> stringResource(R.string.music_track_cut_tail, n.firstBar, n.lastBar)
                                TrackEdit.Notice.Unmatched -> stringResource(R.string.music_track_cut_unmatched)
                            }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.tertiary)
                        }?.onFailure { e ->
                            Text(e.message ?: e.javaClass.simpleName, style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error)
                        }
                    } else Text(stringResource(R.string.music_track_too_much), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error)
                    stretch?.let { (f, s) ->
                        Text(stringResource(R.string.music_track_retake_note, "%.1f s".format(retakeWidth(s - f) / 25.0),
                            duration(retakeSeconds(song, f, s, frames))),
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(enabled = sel != null, onClick = {
                        shown?.let { (lo, hi) ->
                            vm.playRange(song, (lo * 1000).toInt(), (hi * 1000).toInt())
                        }
                    }) { Text(stringResource(if (hearing) R.string.music_track_stop else R.string.music_track_play)) }
                    Button(enabled = cut != null && cutPlan?.isFailure != true, onClick = {
                        cut?.let { (first, stop) -> vm.stop(); vm.cut(song, first, stop, grid, cutPlan?.getOrNull()); onDismiss() }
                    }) { Text(stringResource(R.string.music_track_cut)) }
                }
                Button(enabled = stretch != null, onClick = {
                    stretch?.let { (first, stop) -> vm.stop(); vm.retake(song, first, stop, grid); onDismiss() }
                }) { Text(stringResource(R.string.music_track_retake)) }
                // ---- new words for the chosen stretch (YuE2-ComfyUI's change of words): lines of the lyrics rewritten,
                // the stretch sung again to them on the same tune
                if (YuE2Prompt.sungLines(song.lyrics) > 0) {   // (an instrumental has no words to change)
                    Text(stringResource(R.string.music_track_words), style = MaterialTheme.typography.titleSmall)
                    Text(stringResource(R.string.music_track_words_hint), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    val lyricLines = remember(song.id) { EditText.lines(song.lyrics) }
                    val lo = l0?.let { x -> minOf(x, l1 ?: x) }
                    val hi = l0?.let { x -> maxOf(x, l1 ?: x) }
                    Column {
                        lyricLines.forEachIndexed { i, line ->
                            val text = EditText.strip(line)
                            val tag = text.startsWith("[") && text.endsWith("]")
                            val on = lo != null && hi != null && i in lo..hi
                            Text("${i + 1}  $line", style = MaterialTheme.typography.bodySmall,
                                color = if (tag) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.fillMaxWidth()
                                    .background(if (on) MaterialTheme.colorScheme.primaryContainer else androidx.compose.ui.graphics.Color.Transparent)
                                    .clickable(enabled = !tag && text.isNotEmpty()) {
                                        // a tap starts a range, the next ends it (either way); one tap more starts again
                                        if (l0 == null || l1 != null) { l0 = i; l1 = null } else l1 = i
                                        val s = minOf(l0!!, l1 ?: l0!!)
                                        val e = maxOf(l0!!, l1 ?: l0!!)
                                        wordsText = lyricLines.subList(s, e + 1).joinToString("\n")
                                    }
                                    .padding(vertical = 3.dp))
                        }
                    }
                    if (lo != null && hi != null) {
                        OutlinedTextField(wordsText, { wordsText = it }, Modifier.fillMaxWidth(), minLines = 2,
                            label = { Text(stringResource(R.string.music_track_words_field)) })
                        val rewrite = remember(song.id, lo, hi, wordsText) { runCatching { Words.changeWords(song.lyrics, lo, hi + 1, wordsText) } }
                        rewrite.onSuccess { r ->
                            if (Words.needsFitNotice(r)) Text(stringResource(R.string.music_track_words_fits, r.newSyllables, r.oldSyllables),
                                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.tertiary)
                        }.onFailure { e ->
                            if (wordsText.isNotBlank()) Text(e.message ?: e.javaClass.simpleName, style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error)
                        }
                        val r = rewrite.getOrNull()?.takeIf { it.text != song.lyrics }
                        Button(enabled = stretch != null && r != null, onClick = {
                            stretch?.let { (first, stop) -> r?.let { vm.stop(); vm.retakeWords(song, first, stop, it, grid); onDismiss() } }
                        }) { Text(stringResource(R.string.music_track_words_button)) }
                    }
                }
                // ---- whole sections moved before another (YuE2-ComfyUI's move): the joins sung again
                if (paired && grid != null && barsChosen != null)
                    MoveSection(vm, song, t.score!!, grid, barsChosen, frames, onDismiss)
                // ---- an instrumental break put in before a section (YuE2-ComfyUI's break, experimental)
                if (paired && grid != null) BreakSection(vm, song, t.score!!, grid, frames, onDismiss)
                // ---- going on with new lines (YuE2-ComfyUI's extend): from the bar after the last sung note
                Text(stringResource(R.string.music_track_go_on), style = MaterialTheme.typography.titleSmall)
                val f = from
                if (f == null || tempo == null || t.score == null) {
                    Text(stringResource(R.string.music_track_go_on_none), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                } else {
                    OutlinedTextField(goText, { goText = it }, Modifier.fillMaxWidth(), minLines = 3,
                        label = { Text(stringResource(R.string.music_track_go_on_hint)) })
                    Text(stringResource(R.string.music_track_go_on_note, f.bar + 1, clock(f.start / 25.0),
                        duration(goOnSeconds(song, f.start, frames, goText))),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    var problem by remember(song.id) { mutableStateOf<String?>(null) }
                    problem?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
                    Button(onClick = {
                        runCatching { GoOn.plan(song.lyrics, t.score, goText, tempo, frames) }.fold({ plan ->
                            if (plan == null) problem = "?" else { vm.stop(); vm.goOn(song, plan, goText, t.score, tempo); onDismiss() }
                        }, { e -> problem = e.message })
                    }) { Text(stringResource(R.string.music_track_go_on_button)) }
                }
                TextButton(onClick = { vm.stop(); onDismiss() }) { Text(stringResource(R.string.music_track_close)) }
            }
        }
    }
}

/**
 * Moving the chosen bars [chosen] (whole sections: TrackEdit.move refuses less) before another section of [song] or to
 * its end: the places to go as chips, the plan's notices, an estimate (each join's stretch sung twice, as a retake),
 * and the button that makes the new song.
 */
@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun MoveSection(vm: MusicViewModel, song: Song, score: String, grid: BarGrid, chosen: Pair<Int, Int>, frames: Int,
                        onDismiss: () -> Unit) {
    var moveTo by remember(song.id) { mutableStateOf<Int?>(null) }
    val count = grid.bars.size - 1
    // (a song the length cap stopped: its score's later sections are not in the sound)
    val targets = (grid.sections.map { it.first }.filter { grid.bars[it] < frames / 25.0 } + count).distinct().sorted()
        .filter { it < chosen.first || it > chosen.second }
    val to = moveTo?.takeIf { it in targets }
    Text(stringResource(R.string.music_track_move), style = MaterialTheme.typography.titleSmall)
    Text(stringResource(R.string.music_track_move_hint), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (bar in targets) {
            val name = grid.sections.firstOrNull { it.first == bar }?.second.orEmpty()
            FilterChip(selected = bar == to, onClick = { moveTo = bar }, label = {
                Text(if (bar == count) stringResource(R.string.music_track_move_end)
                     else stringResource(R.string.music_track_move_to, name, bar + 1))
            })
        }
    }
    val plan = remember(score, grid, chosen, to, frames) {
        to?.let { runCatching { TrackEdit.move(score, song.lyrics, grid, chosen.first, chosen.second, it, frames) } }
    }
    plan?.onSuccess { p ->
        for (n in p.notices) Text(when (n) {
            TrackEdit.MoveNotice.Unmatched -> stringResource(R.string.music_track_cut_unmatched)
            TrackEdit.MoveNotice.Tagged -> stringResource(R.string.music_track_move_tagged)
            is TrackEdit.MoveNotice.BareHead -> stringResource(R.string.music_track_move_head, n.section)
            is TrackEdit.MoveNotice.BareTail -> stringResource(R.string.music_track_move_tail, n.section)
        }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.tertiary)
        Text(stringResource(R.string.music_track_move_note, p.sung.size,
            duration(p.sung.sumOf { (a, b) -> 2 * retakeSeconds(song, a, b, frames) })),
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }?.onFailure { e ->
        Text(e.message ?: e.javaClass.simpleName, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
    }
    Button(enabled = plan?.isSuccess == true, onClick = {
        plan?.getOrNull()?.let { vm.stop(); vm.move(song, it); onDismiss() }
    }) { Text(stringResource(R.string.music_track_move_button)) }
}

/**
 * A break of some bars put in before a section of [song] (TrackEdit.breakPlan): the sections inside the song as chips,
 * its length in bars, the plan's notices and an estimate (its score written, then sung in as a retake of about its
 * length), and the button that makes the new song.
 */
@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun BreakSection(vm: MusicViewModel, song: Song, score: String, grid: BarGrid, frames: Int, onDismiss: () -> Unit) {
    val count = grid.bars.size - 1
    // (inside the sound: a song the length cap stopped has more score than sound)
    val places = grid.sections.map { it.first }.distinct().sorted().filter { it in 1 until count && grid.bars[it] < frames / 25.0 - 1.0 }
    if (places.isEmpty()) return
    var at by remember(song.id) { mutableStateOf<Int?>(null) }
    var bars by remember(song.id) { mutableStateOf(4) }
    val bar = at?.takeIf { it in places }
    Text(stringResource(R.string.music_track_break), style = MaterialTheme.typography.titleSmall)
    Text(stringResource(R.string.music_track_break_hint), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (b in places) {
            val name = grid.sections.firstOrNull { it.first == b }?.second.orEmpty()
            FilterChip(selected = b == bar, onClick = { at = b },
                label = { Text(stringResource(R.string.music_track_move_to, name, b + 1)) })
        }
    }
    Text(stringResource(R.string.music_track_break_bars), style = MaterialTheme.typography.bodySmall)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (n in listOf(1, 2, 4, 8, 16)) FilterChip(selected = n == bars, onClick = { bars = n }, label = { Text("$n") })
    }
    val plan = remember(score, grid, bar, bars, frames) {
        bar?.let { runCatching { TrackEdit.breakPlan(score, song.lyrics, grid, it, bars, frames) } }
    }
    plan?.onSuccess { p ->
        for (n in p.notices) Text(when (n) {
            TrackEdit.BreakNotice.Unmatched -> stringResource(R.string.music_track_cut_unmatched)
        }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.tertiary)
        Text(stringResource(R.string.music_track_break_note, clock(p.start / 25.0), duration(breakSeconds(song, p, grid, frames))),
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }?.onFailure { e ->
        Text(e.message ?: e.javaClass.simpleName, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
    }
    Button(enabled = plan?.isSuccess == true, onClick = {
        plan?.getOrNull()?.let { vm.stop(); vm.breakIn(song, it, score, grid); onDismiss() }
    }) { Text(stringResource(R.string.music_track_break_button)) }
}

/** About how long [plan]'s break takes: its bars (as long as the bars before it) written as a score at the rate the
 *  song's codes were sung, about 40 tokens a bar, then sung in as a retake of that length at its frame. */
private fun breakSeconds(song: Song, plan: TrackEdit.BreakPlan, grid: BarGrid, frames: Int): Double {
    val lines = grid.bars
    val bar = (lines[plan.bar] - lines[(plan.bar - 1).coerceAtLeast(0)]).takeIf { it > 0.2 } ?: 2.0
    val length = Math.rint(bar * plan.bars * 25.0).toInt().coerceAtLeast(1)
    val times = song.result?.optJSONObject("times")
    val codes = song.result?.optInt("codes", 0) ?: 0
    val tCodes = times?.optDouble("codes", 0.0) ?: 0.0
    val perToken = if (codes > 0 && tCodes > 1.0) tCodes / codes else 0.25
    return perToken * 40 * plan.bars + retakeSeconds(song, plan.start, minOf(frames, plan.start + length), frames)
}

/** How far a retake of [length] frames may move its length either way (the engine's width: 2 s, at most a third of
 *  the stretch). */
private fun retakeWidth(length: Int): Int = minOf(50, length / 3)

/**
 * About how long a retake of frames [first, stop) of [song] takes: the new part sung at the rate the song's codes were
 * (else 4 a second), the context and the join's tokens read at the rate the song's sound read its prompt (each length
 * the join tries: a new token, then 3 s of the old codes), then the frames drawn again.
 */
private fun retakeSeconds(song: Song, first: Int, stop: Int, frames: Int): Double {
    val times = song.result?.optJSONObject("times")
    val length = stop - first
    val width = retakeWidth(length)
    val codes = song.result?.optInt("codes", 0) ?: 0
    val tCodes = times?.optDouble("codes", 0.0) ?: 0.0
    val perToken = if (codes > 0 && tCodes > 1.0) tCodes / codes else 0.25
    val prefix = times?.optDouble("prefix", 0.0) ?: 0.0
    val perRead = if (prefix > 1.0) prefix / frames.coerceAtLeast(1) else 0.05
    val counts = if (stop >= frames) 0 else 2 * width + 1
    return perToken * (length + width + counts) + perRead * (first + 75 * counts) + seamSeconds(song, frames)
}

/**
 * About how long [song] going on from [start] takes, provisional - its length is the score's the model writes: the new
 * part guessed as what it replaces and 4 s a new line, the score's tokens and the new codes at the rate the song's codes
 * were (else 4 a second), the context read at the rate its prompt was, then the frames drawn again.
 */
private fun goOnSeconds(song: Song, start: Int, frames: Int, text: String): Double {
    val times = song.result?.optJSONObject("times")
    val codes = song.result?.optInt("codes", 0) ?: 0
    val tCodes = times?.optDouble("codes", 0.0) ?: 0.0
    val perToken = if (codes > 0 && tCodes > 1.0) tCodes / codes else 0.25
    val prefix = times?.optDouble("prefix", 0.0) ?: 0.0
    val perRead = if (prefix > 1.0) prefix / frames.coerceAtLeast(1) else 0.05
    val lines = text.lines().count { it.isNotBlank() && !it.trim().startsWith("[") }
    val length = (frames - start) + 100 * lines
    return perToken * (length + 400) + perRead * start + seamSeconds(song, start + length)
}

/** About how long a seam inside [song] takes to sing again when [left] frames remain: its window (up to 40 s either
 *  side of the seam) at the rate its own sound was made, the load and the decode around it. */
private fun seamSeconds(song: Song, left: Int): Double {
    val times = song.result?.optJSONObject("times")
    val frames = (song.seconds * YuE2Prompt.FRAMES_PER_SECOND).coerceAtLeast(1.0)
    val sound = (times?.optDouble("prefix", 0.0) ?: 0.0) + (times?.optDouble("acoustic", 0.0) ?: 0.0)
    val window = minOf(left, 2 * 1000 + 24).toDouble()
    return sound * minOf(1.0, window / frames) + 20.0
}
