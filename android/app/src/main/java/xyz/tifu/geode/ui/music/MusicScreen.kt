package xyz.tifu.geode.ui.music

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material.icons.outlined.Piano
import xyz.tifu.geode.ui.settings.SwitchRow
import androidx.compose.material.icons.outlined.Stop
import xyz.tifu.geode.music.MusicClient
import kotlinx.coroutines.launch
import androidx.compose.runtime.rememberCoroutineScope
import android.content.ActivityNotFoundException
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Bookmarks
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Pause
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.Remove
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.delay
import xyz.tifu.geode.R
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.MusicAdapter
import xyz.tifu.geode.music.LyricsLength
import xyz.tifu.geode.music.LyricsWriter
import xyz.tifu.geode.music.MusicPresets
import xyz.tifu.geode.music.MusicPrefs
import xyz.tifu.geode.music.MusicProgress
import xyz.tifu.geode.music.MusicStage
import xyz.tifu.geode.music.MusicState
import xyz.tifu.geode.music.Song
import xyz.tifu.geode.music.SongStatus
import xyz.tifu.geode.music.YuE2Prompt

/**
 * The drawer's Music: songs made on the phone with YuE2. Create - the style, the mode, the lyrics with a toolbar for what
 * the model reads besides the words (the section tags, an instrumental break as the voice's pause, the stress as a
 * capital), the lyrics writer on a chat model; Library - the songs, their progress while they are made, playing,
 * sharing, a new take (on the same melody too). The song model and its settings are the Music settings'.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MusicScreen(vm: MusicViewModel, openDrawer: () -> Unit, openSettings: () -> Unit) {
    val service by vm.service.collectAsStateWithLifecycle()
    val library by vm.library.collectAsStateWithLifecycle()
    val models by vm.models.collectAsStateWithLifecycle()
    var tab by rememberSaveable { mutableIntStateOf(0) }
    // the song-ready notification's Share (MainActivity): the library, and the share sheet once the file is made
    val context = LocalContext.current
    val shareAsked by MusicClient.shareRequested.collectAsStateWithLifecycle()
    LaunchedEffect(shareAsked) {
        val id = shareAsked ?: return@LaunchedEffect
        MusicClient.shareRequested.value = null
        tab = 1
        vm.share(id)?.let { i -> try { context.startActivity(i) } catch (_: ActivityNotFoundException) {} }
    }

    Column(Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.nav_music)) },
            navigationIcon = {
                IconButton(onClick = openDrawer) {
                    Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
                }
            },
            actions = {
                IconButton(onClick = openSettings) {
                    Icon(Icons.Outlined.Tune, contentDescription = stringResource(R.string.nav_music_settings))
                }
            },
        )
        PrimaryTabRow(selectedTabIndex = tab) {
            Tab(selected = tab == 0, onClick = { tab = 0 }, text = { Text(stringResource(R.string.music_tab_create)) })
            Tab(selected = tab == 1, onClick = { tab = 1 }, text = {
                Text(stringResource(R.string.music_tab_library, library.size))
            })
        }
        RunningCard(vm, service, library)
        when (tab) {
            0 -> CreateTab(vm, noModel = models.isEmpty(), openSettings = openSettings, made = { tab = 1 })
            else -> LibraryTab(vm, service, library, toCreate = { tab = 0 })
        }
    }
}

/** The song being made now, over both tabs: its stage, its progress, Cancel. */
@Composable
private fun RunningCard(vm: MusicViewModel, service: MusicState, library: List<Song>) {
    val p = service.progress ?: return
    val song = library.firstOrNull { it.id == p.song }
    Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
        Column(Modifier.padding(12.dp)) {
            Text(song?.title ?: stringResource(R.string.music_notif_title), style = MaterialTheme.typography.titleSmall,
                maxLines = 1, overflow = TextOverflow.Ellipsis)
            ProgressRows(p)
            if (service.queued.isNotEmpty())
                Text(stringResource(R.string.music_queued_more, service.queued.size), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (song != null) TextButton(onClick = { vm.cancel(song) }, modifier = Modifier.align(Alignment.End)) {
                Text(stringResource(R.string.music_cancel))
            }
        }
    }
}

@Composable
private fun ProgressRows(p: MusicProgress) {
    val context = LocalContext.current
    Text(MusicProgress.stageText(context, p), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 2.dp))
    if (p.total > 1) LinearProgressIndicator(progress = { p.fraction }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
    else LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 6.dp))
    // the clock runs between the engine's reports (none come while the model loads)
    var now by remember { mutableStateOf(android.os.SystemClock.elapsedRealtime()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1000)
            now = android.os.SystemClock.elapsedRealtime()
        }
    }
    Text(stringResource(R.string.music_elapsed, duration(p.elapsedAt(now))), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
}

// ---- Create ----

/** The score another song lends the draft: its key moved by semitones (every note and chord with it - the style line
 *  cannot change a key, the score can), its text shown and edited. YuE2-ComfyUI's transpose and score editor, in a
 *  phone's size. */
@Composable
private fun MelodyCard(vm: MusicViewModel, draft: SongDraft) {
    val score by vm.score.collectAsStateWithLifecycle()
    var open by rememberSaveable { mutableStateOf(false) }
    var resets by remember { mutableIntStateOf(0) }
    var roll by remember { mutableStateOf(false) }
    val small = MaterialTheme.typography.bodySmall
    Card(Modifier.fillMaxWidth().padding(top = 8.dp)) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.music_same_melody_of, draft.abcFrom), Modifier.weight(1f),
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
                TextButton(onClick = { vm.dropMelody() }) { Text(stringResource(R.string.music_new_melody)) }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                val s = score
                val key = s?.key ?: "?"
                Text(if (draft.transpose != 0 && s?.moved != null) stringResource(R.string.music_key_moved, key, s.moved)
                     else stringResource(R.string.music_key, key),
                    Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                IconButton(onClick = { vm.transposeBy(-1) }, enabled = draft.transpose > -AbcScore.LIMIT) {
                    Icon(Icons.Outlined.Remove, stringResource(R.string.music_key_down))
                }
                Text(if (draft.transpose > 0) "+${draft.transpose}" else "${draft.transpose}",
                    style = MaterialTheme.typography.titleMedium)
                IconButton(onClick = { vm.transposeBy(1) }, enabled = draft.transpose < AbcScore.LIMIT) {
                    Icon(Icons.Outlined.Add, stringResource(R.string.music_key_up))
                }
            }
            score?.error?.let {
                Text(stringResource(R.string.music_score_unreadable, it), style = small, color = MaterialTheme.colorScheme.error)
            }
            // a MIDI file's tune: what its import changed or guessed, how the lyrics are laid along it
            val imported by vm.midiNotes.collectAsStateWithLifecycle()
            val context = LocalContext.current
            (imported + score?.laid.orEmpty()).forEach { n ->
                Text("\u2022 " + MidiText.notice(context, n), style = small, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 2.dp))
            }
            val playing by vm.playing.collectAsStateWithLifecycle()
            val hearing = playing.first == MusicViewModel.SCORE + "draft"
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = { open = !open }) {
                    Text(stringResource(if (open) R.string.music_score_hide else R.string.music_score_show))
                }
                TextButton(onClick = { vm.listenScore() }, enabled = score?.error == null) {
                    Text(stringResource(if (hearing) R.string.music_score_stop else R.string.music_score_listen))
                }
                TextButton(onClick = { roll = true }, enabled = score?.original != null) { Text(stringResource(R.string.music_roll)) }
            }
            // the score's notes as a piano roll: what it applies becomes the draft's score text
            if (roll) (draft.abcText ?: score?.original)?.let { text ->
                PianoRollDialog(text, playing.first == MusicViewModel.SCORE + "roll", vm::listenText,
                    onApply = { vm.editScore(it); resets++ }, onDismiss = { roll = false; vm.stop() })
            }
            if (hearing) Text(stringResource(R.string.music_score_listen_note), style = small,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (open) {
                // the field is the truth while the user types (the draft follows it); a reset puts the song's back. The
                // song's text comes with the score's view (read off the main thread): the field starts when it does.
                val original = score?.original
                var text by remember(resets, original != null) { mutableStateOf(draft.abcText ?: original.orEmpty()) }
                OutlinedTextField(
                    value = text, onValueChange = { t -> text = t; vm.editScore(t) },
                    textStyle = small.copy(fontFamily = FontFamily.Monospace),
                    minLines = 6, maxLines = 16, modifier = Modifier.fillMaxWidth(),
                )
                Text(stringResource(R.string.music_score_hint), style = small,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
                if (draft.abcText != null) TextButton(onClick = { vm.editScore(null); resets++ }) {
                    Text(stringResource(R.string.music_score_reset))
                }
            }
        }
    }
}

@Composable
private fun CreateTab(vm: MusicViewModel, noModel: Boolean, openSettings: () -> Unit, made: () -> Unit) {
    val draft by vm.draft.collectAsStateWithLifecycle()
    val prefs by vm.prefs.collectAsStateWithLifecycle()
    val tokens by vm.tokens.collectAsStateWithLifecycle()
    val replaced by vm.replaced.collectAsStateWithLifecycle()
    // a tune from a MIDI file (the system's file picker)
    val midiPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(vm::importMidi) }
    var lyrics by remember { mutableStateOf(TextFieldValue(draft.lyrics)) }
    // the draft replaced from elsewhere (its file read, the writer's song used, a song of the library edited): the field
    // takes it; while the user types the field is the truth and the draft follows it
    var title by remember { mutableStateOf(draft.title) }
    var style by remember { mutableStateOf(draft.style) }
    LaunchedEffect(replaced) {
        val d = vm.draft.value
        lyrics = TextFieldValue(d.lyrics)
        title = d.title
        style = d.style
    }
    fun setStyle(s: String) {
        style = s
        vm.editDraft { it.copy(style = s) }
    }
    var presets by remember { mutableStateOf(false) }
    var genres by remember { mutableStateOf(false) }
    var tags by remember { mutableStateOf(false) }
    var writer by remember { mutableStateOf(false) }
    var noModelAsked by remember { mutableStateOf(false) }
    var takes by rememberSaveable { mutableIntStateOf(1) }
    var scoresFirst by rememberSaveable { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
        if (noModel) Card(Modifier.fillMaxWidth().padding(top = 8.dp)) {
            Column(Modifier.padding(12.dp)) {
                Text(stringResource(R.string.music_no_model), style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = openSettings) { Text(stringResource(R.string.music_get_model)) }
            }
        }
        OutlinedTextField(
            value = title, onValueChange = { t -> title = t; vm.editDraft { it.copy(title = t) } },
            label = { Text(stringResource(R.string.music_title)) }, singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        )
        OutlinedTextField(
            value = style, onValueChange = { t -> setStyle(t) },
            label = { Text(stringResource(R.string.music_style)) },
            supportingText = { Text(stringResource(if (draft.instrumental) R.string.music_instrumental_style_hint else R.string.music_style_hint)) },
            minLines = 2, maxLines = 5, modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        )
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            AssistChip(onClick = { presets = true }, label = { Text(stringResource(R.string.music_presets)) },
                leadingIcon = { Icon(Icons.Outlined.Bookmarks, null, Modifier.size(18.dp)) })
            Box {
                AssistChip(onClick = { genres = true }, label = { Text(stringResource(R.string.music_add_genre)) })
                DropdownMenu(expanded = genres, onDismissRequest = { genres = false }, modifier = Modifier.heightIn(max = 360.dp)) {
                    MusicPresets.GENRES.forEach { g ->
                        DropdownMenuItem(text = { Text(g) }, onClick = {
                            genres = false
                            setStyle(LyricsEdit.addGenre(style, g))
                        })
                    }
                }
            }
        }

        // an instrumental with the adapter (MusicAdapter): the model plans the sections and plays them, no voice
        val context = LocalContext.current
        val adapterDownload by MusicAdapter.download.collectAsStateWithLifecycle()
        val adapter = remember(draft.instrumental, adapterDownload) { MusicAdapter.installed(context) }
        SwitchRow(stringResource(R.string.music_instrumental_lora),
            stringResource(if (adapter) R.string.music_instrumental_lora_desc else R.string.music_instrumental_lora_get),
            draft.instrumental, { vm.setInstrumental(it) }, modifier = Modifier.padding(top = 8.dp),
            titleStyle = MaterialTheme.typography.bodyLarge, enabled = adapter || draft.instrumental)
        if (!adapter) TextButton(onClick = openSettings) { Text(stringResource(R.string.music_instrumental_lora_settings)) }

        // the mode: a score with chords, a melody-only score, no score
        Text(stringResource(R.string.music_mode), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 12.dp))
        xyz.tifu.geode.ui.settings.Choice(
            listOf(stringResource(R.string.music_mode_full), stringResource(R.string.music_mode_melody),
                stringResource(R.string.music_mode_off)),
            draft.mode.ordinal, !draft.instrumental,
        ) { i -> vm.setMode(YuE2Prompt.Mode.entries[i]) }
        Text(stringResource(when (draft.mode) {
            YuE2Prompt.Mode.FULL -> R.string.music_mode_full_desc
            YuE2Prompt.Mode.MELODY -> R.string.music_mode_melody_desc
            YuE2Prompt.Mode.OFF -> R.string.music_mode_off_desc
        }), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        if (draft.abc.isNotEmpty()) MelodyCard(vm, draft)
        else AssistChip(onClick = { midiPicker.launch(MidiText.TYPES) }, label = { Text(stringResource(R.string.music_midi_import)) },
            leadingIcon = { Icon(Icons.Outlined.Piano, null, Modifier.size(18.dp)) })

        // the lyrics and their toolbar (an instrumental's: its section plan, the adapter's own bare tags)
        Text(stringResource(if (draft.instrumental) R.string.music_instrumental_plan else R.string.music_lyrics),
            style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 12.dp))
        if (draft.instrumental) {
            Row(Modifier.horizontalScroll(rememberScrollState()).padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MusicAdapter.TAGS.forEach { t ->
                    AssistChip(onClick = {
                        lyrics = LyricsEdit.insertLine(lyrics, "[$t]")
                        vm.editDraft { it.copy(lyrics = lyrics.text) }
                    }, label = { Text("[$t]") })
                }
            }
        } else Row(Modifier.horizontalScroll(rememberScrollState()).padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Box {
                AssistChip(onClick = { tags = true }, label = { Text(stringResource(R.string.music_tool_section)) })
                DropdownMenu(expanded = tags, onDismissRequest = { tags = false }) {
                    YuE2Prompt.TAGS.filter { it !in YuE2Prompt.BREAKS }.forEach { t ->
                        DropdownMenuItem(text = { Text("[$t]") }, onClick = {
                            tags = false
                            lyrics = LyricsEdit.insertTag(lyrics, t)
                            vm.editDraft { it.copy(lyrics = lyrics.text) }
                        })
                    }
                }
            }
            AssistChip(onClick = {
                lyrics = LyricsEdit.insertTag(lyrics, "Interlude")
                vm.editDraft { it.copy(lyrics = lyrics.text) }
            }, label = { Text(stringResource(R.string.music_tool_pause)) })
            AssistChip(onClick = {
                lyrics = LyricsEdit.toggleStress(lyrics)
                vm.editDraft { it.copy(lyrics = lyrics.text) }
            }, label = { Text(stringResource(R.string.music_tool_stress)) })
            AssistChip(onClick = { writer = true }, label = { Text(stringResource(R.string.music_tool_write)) },
                leadingIcon = { Icon(Icons.Outlined.AutoAwesome, null, Modifier.size(18.dp)) })
        }
        OutlinedTextField(
            value = lyrics,
            onValueChange = {
                val changed = it.text != lyrics.text
                lyrics = it
                if (changed) vm.editDraft { d -> d.copy(lyrics = it.text) }
            },
            placeholder = { Text(if (draft.instrumental) MusicAdapter.BARE else stringResource(R.string.music_lyrics_placeholder)) },
            textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences),
            minLines = 8, modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
        )
        Text(stringResource(if (draft.instrumental) R.string.music_instrumental_plan_hint else R.string.music_tools_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))

        // what the song will be: its lines, its ceiling, the prompt's tokens, the time it may take here
        val sung = YuE2Prompt.sungLines(draft.lyrics)
        val ceiling = if (prefs.maxSeconds > 0) prefs.maxSeconds else YuE2Prompt.autoSeconds(draft.lyrics)
        val info = buildList {
            add(if (sung == 0) stringResource(R.string.music_instrumental) else stringResource(R.string.music_sung_lines, sung))
            add(stringResource(R.string.music_ceiling, duration(ceiling.toDouble())))
            tokens?.let { add(stringResource(R.string.music_prompt_tokens, it)) }
            vm.estimate(ceiling, takes)?.let { add(stringResource(R.string.music_estimate, duration(it))) }
        }
        Text(info.joinToString(" · "), style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 8.dp))
        // takes: the same words sung several times, each with its own seed, queued one after another
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.music_takes, takes), style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(R.string.music_takes_desc), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            IconButton(onClick = { takes-- }, enabled = takes > 1) {
                Icon(Icons.Outlined.Remove, stringResource(R.string.music_takes_fewer))
            }
            IconButton(onClick = { takes++ }, enabled = takes < MusicPrefs.MAX_TAKES) {
                Icon(Icons.Outlined.Add, stringResource(R.string.music_takes_more))
            }
        }
        // scores first: only the tunes written (a fraction of the time), heard as MIDI, the one liked sung - when the
        // model writes the score (not a borrowed one, not the mode without one)
        val planning = scoresFirst && draft.abc.isEmpty() && draft.mode != YuE2Prompt.Mode.OFF
        if (draft.abc.isEmpty() && draft.mode != YuE2Prompt.Mode.OFF)
            SwitchRow(stringResource(R.string.music_scores_first), stringResource(R.string.music_scores_first_desc),
                scoresFirst, { scoresFirst = it }, modifier = Modifier.padding(top = 4.dp),
                titleStyle = MaterialTheme.typography.bodyMedium)
        Button(
            onClick = {
                if (vm.make(takes, planning)) made() else noModelAsked = true
            },
            // (an instrumental with an empty plan is one too: the model chooses its sections)
            enabled = if (draft.instrumental) adapter else style.isNotBlank() || lyrics.text.isNotBlank(),
            modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
        ) { Text(stringResource(if (planning) R.string.music_make_scores else R.string.music_make)) }
        Spacer(Modifier.height(24.dp))
    }

    if (presets) PresetsDialog(onDismiss = { presets = false }) { p ->
        presets = false
        setStyle(LyricsEdit.applyPreset(style, p.style, prefs.writerLanguage))
    }
    if (writer) WriterSheet(vm, onDismiss = { writer = false })
    if (noModelAsked) AlertDialog(
        onDismissRequest = { noModelAsked = false },
        text = { Text(stringResource(R.string.music_no_model)) },
        confirmButton = { TextButton(onClick = { noModelAsked = false; openSettings() }) { Text(stringResource(R.string.music_get_model)) } },
        dismissButton = { TextButton(onClick = { noModelAsked = false }) { Text(stringResource(android.R.string.cancel)) } },
    )
}

@Composable
private fun PresetsDialog(onDismiss: () -> Unit, onPick: (xyz.tifu.geode.music.StylePreset) -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.music_presets)) },
        text = {
            LazyColumn(Modifier.heightIn(max = 460.dp)) {
                items(MusicPresets.EXAMPLES) { p ->
                    Column(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
                        TextButton(onClick = { onPick(p) }, modifier = Modifier.fillMaxWidth()) {
                            Column(Modifier.fillMaxWidth()) {
                                Text(p.name, style = MaterialTheme.typography.titleSmall)
                                Text(p.style, style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 3, overflow = TextOverflow.Ellipsis)
                            }
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(stringResource(android.R.string.cancel)) } },
    )
}

/** The lyrics writer: an idea in, a style line and tagged lyrics out, written by a chat model of the phone. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun WriterSheet(vm: MusicViewModel, onDismiss: () -> Unit) {
    val w by vm.writer.collectAsStateWithLifecycle()
    val prefs by vm.prefs.collectAsStateWithLifecycle()
    val chatModels by vm.chatModels.collectAsStateWithLifecycle()
    val chatSelected by vm.chatSelected.collectAsStateWithLifecycle()
    val state = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = state) {
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Text(stringResource(R.string.music_writer_title), style = MaterialTheme.typography.titleMedium)
            val model = chatModels.firstOrNull { it.id == prefs.writerModel.ifEmpty { chatSelected } }
            Text(model?.let { stringResource(R.string.music_writer_model, it.title) } ?: stringResource(R.string.music_writer_no_model),
                style = MaterialTheme.typography.bodySmall,
                color = if (model == null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedTextField(
                value = w.idea, onValueChange = { t -> vm.editWriter { it.copy(idea = t) } },
                label = { Text(stringResource(R.string.music_writer_idea)) },
                placeholder = { Text(stringResource(R.string.music_writer_idea_hint)) },
                minLines = 2, modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            )
            OutlinedTextField(
                value = w.extra, onValueChange = { t -> vm.editWriter { it.copy(extra = t) } },
                label = { Text(stringResource(R.string.music_writer_extra)) },
                minLines = 1, modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            )
            // the length and the language right here (the rest is in the Music settings)
            Text(stringResource(R.string.music_writer_length), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 8.dp))
            xyz.tifu.geode.ui.settings.Choice(lengthLabels(), prefs.writerLength.ordinal) { i ->
                vm.setPrefs { it.copy(writerLength = LyricsLength.entries[i]) }
            }
            LanguagePicker(prefs.writerLanguage) { l -> vm.setPrefs { it.copy(writerLanguage = l) } }
            Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                if (w.running) {
                    OutlinedButton(onClick = vm::stopWriting) { Text(stringResource(R.string.music_writer_stop)) }
                    CircularProgressIndicator(Modifier.padding(start = 12.dp).size(18.dp), strokeWidth = 2.dp)
                } else Button(onClick = vm::write, enabled = w.idea.isNotBlank() && model != null) {
                    Text(stringResource(if (w.lyrics.isEmpty()) R.string.music_writer_write else R.string.music_writer_again))
                }
            }
            w.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            if (w.running && w.text.isNotEmpty()) Text(w.text, style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace, modifier = Modifier.padding(top = 8.dp))
            if (!w.running && (w.style.isNotEmpty() || w.lyrics.isNotEmpty())) {
                HorizontalDivider(Modifier.padding(vertical = 8.dp))
                Text(w.style, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
                Text(w.lyrics, style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(top = 6.dp))
                w.notes.forEach { n ->
                    Text(stringResource(n.text), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp))
                }
                Button(onClick = { vm.useWritten(); onDismiss() }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                    Text(stringResource(R.string.music_writer_use))
                }
            }
            Spacer(Modifier.height(32.dp))
        }
    }
}

@Composable
internal fun lengthLabels(): List<String> = listOf(
    stringResource(R.string.music_length_short), stringResource(R.string.music_length_normal),
    stringResource(R.string.music_length_long), stringResource(R.string.music_length_very_long),
)

/** The language the song is sung in: the writer's choice, or one named (its name in the phone's language). */
@Composable
internal fun LanguagePicker(value: String, onPick: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    val shown = LyricsWriter.LANGUAGES.firstOrNull { it.first == value }?.let { languageName(it.second) }
        ?: stringResource(R.string.music_language_auto)
    Row(Modifier.padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(stringResource(R.string.music_language), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
        Box {
            TextButton(onClick = { open = true }) { Text(shown) }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                DropdownMenuItem(text = { Text(stringResource(R.string.music_language_auto)) }, onClick = { open = false; onPick("") })
                LyricsWriter.LANGUAGES.forEach { (name, tag) ->
                    DropdownMenuItem(text = { Text(languageName(tag)) }, onClick = { open = false; onPick(name) })
                }
            }
        }
    }
}

private fun languageName(tag: String): String {
    val l = java.util.Locale.forLanguageTag(tag)
    return l.getDisplayLanguage(java.util.Locale.getDefault()).replaceFirstChar { it.titlecase(java.util.Locale.getDefault()) }
}

// ---- Library ----

@Composable
private fun LibraryTab(vm: MusicViewModel, service: MusicState, library: List<Song>, toCreate: () -> Unit) {
    if (library.isEmpty()) {
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Text(stringResource(R.string.music_library_empty), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        return
    }
    var query by rememberSaveable { mutableStateOf("") }
    val shown = remember(library, query) { LibrarySearch.filter(library, query) }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (library.size >= LibrarySearch.FROM || query.isNotEmpty()) item(key = "search") {
            xyz.tifu.geode.ui.settings.SearchField(query, { query = it }, stringResource(R.string.music_search),
                stringResource(R.string.models_cancel))
        }
        items(shown, key = { it.id }) { song -> SongCard(vm, song, service, toCreate) }
        if (shown.isEmpty()) item(key = "none") {
            Text(stringResource(R.string.music_search_none), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(8.dp))
        }
    }
}

@Composable
private fun SongCard(vm: MusicViewModel, song: Song, service: MusicState, toCreate: () -> Unit) {
    val context = LocalContext.current
    val playing by vm.playing.collectAsStateWithLifecycle()
    val sharing by vm.sharing.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    // the share sheet (Viber, Telegram, WhatsApp...) once the file is ready: an .m4a for a messenger, or the original
    fun share(lossless: Boolean) = scope.launch {
        vm.share(song, lossless)?.let { i -> try { context.startActivity(i) } catch (_: ActivityNotFoundException) {} }
    }
    var menu by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf(false) }
    var details by remember { mutableStateOf(false) }
    var deleting by remember { mutableStateOf(false) }
    var trackEditing by remember { mutableStateOf(false) }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(start = 12.dp, top = 8.dp, bottom = 8.dp, end = 4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(song.title, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(statusLine(song, service), style = MaterialTheme.typography.bodySmall,
                        color = if (song.status == SongStatus.FAILED) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis)
                }
                if (song.status == SongStatus.DONE && song.plan) {
                    val hearing = playing.first == MusicViewModel.SCORE + song.id
                    IconButton(onClick = { vm.listenScore(song) }) {
                        Icon(if (hearing) Icons.Outlined.Stop else Icons.Outlined.PlayArrow,
                            contentDescription = stringResource(if (hearing) R.string.music_score_stop else R.string.music_score_listen))
                    }
                    TextButton(onClick = { vm.sing(song) }) { Text(stringResource(R.string.music_sing_score)) }
                }
                if (song.status == SongStatus.DONE && !song.plan) {
                    val on = playing.first == song.id && playing.second
                    IconButton(onClick = { vm.play(song) }) {
                        Icon(if (on) Icons.Outlined.Pause else Icons.Outlined.PlayArrow,
                            contentDescription = stringResource(if (on) R.string.music_pause else R.string.music_play))
                    }
                    if (sharing == song.id) Box(Modifier.size(48.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                    } else IconButton(onClick = { share(lossless = false) }, enabled = sharing == null) {
                        Icon(Icons.Outlined.Share, contentDescription = stringResource(R.string.music_share))
                    }
                }
                Box {
                    IconButton(onClick = { menu = true }) {
                        Icon(Icons.Outlined.MoreVert, contentDescription = stringResource(R.string.music_more))
                    }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        val busy = service.busy(song.id)
                        if (song.status != SongStatus.WAITING || !busy) {
                            DropdownMenuItem(text = { Text(stringResource(R.string.music_again)) },
                                onClick = { menu = false; vm.again(song, melody = false) })
                            if (song.abc.isNotEmpty()) DropdownMenuItem(text = { Text(stringResource(R.string.music_again_melody)) },
                                onClick = { menu = false; vm.again(song, melody = true) })
                        }
                        if (song.status == SongStatus.WAITING && !busy && service.connected)
                            DropdownMenuItem(text = { Text(stringResource(R.string.music_resume)) },
                                onClick = { menu = false; vm.resume(song) })
                        if (song.status == SongStatus.WAITING)
                            DropdownMenuItem(text = { Text(stringResource(R.string.music_cancel)) },
                                onClick = { menu = false; vm.cancel(song) })
                        // the song itself: bars cut out (the track editor; songs made with their state)
                        if (song.editable) DropdownMenuItem(text = { Text(stringResource(R.string.music_track_edit)) },
                            onClick = { menu = false; trackEditing = true })
                        DropdownMenuItem(text = { Text(stringResource(R.string.music_edit)) },
                            onClick = { menu = false; vm.edit(song); toCreate() })
                        // its score kept: another key, the notes edited (the Create tab's melody card)
                        if (song.abc.isNotEmpty()) DropdownMenuItem(text = { Text(stringResource(R.string.music_edit_melody)) },
                            onClick = { menu = false; vm.edit(song, melody = true); toCreate() })
                        if (song.abc.isNotEmpty()) {
                            val hearing = playing.first == MusicViewModel.SCORE + song.id
                            DropdownMenuItem(text = { Text(stringResource(if (hearing) R.string.music_score_stop else R.string.music_score_listen)) },
                                onClick = { menu = false; vm.listenScore(song) })
                            DropdownMenuItem(text = { Text(stringResource(R.string.music_score_midi)) }, onClick = {
                                menu = false
                                scope.launch { vm.shareScore(song)?.let { i -> try { context.startActivity(i) } catch (_: ActivityNotFoundException) {} } }
                            })
                        }
                        // the file itself (FLAC or WAV, lossless): for a computer, a player, an editor
                        if (song.status == SongStatus.DONE && !song.plan && sharing == null)
                            DropdownMenuItem(text = { Text(stringResource(R.string.music_share_original)) },
                                onClick = { menu = false; share(lossless = true) })
                        DropdownMenuItem(text = { Text(stringResource(R.string.music_rename)) },
                            onClick = { menu = false; renaming = true })
                        DropdownMenuItem(text = { Text(stringResource(R.string.music_details)) },
                            onClick = { menu = false; details = true })
                        DropdownMenuItem(text = { Text(stringResource(R.string.music_delete)) },
                            onClick = { menu = false; deleting = true })
                    }
                }
            }
            if (song.status == SongStatus.WAITING) service.progress?.takeIf { it.song == song.id }?.let { p ->
                Column(Modifier.padding(end = 8.dp)) { ProgressRows(p) }
            }
            if (playing.first == song.id) PlayerBar(vm)
        }
    }
    if (renaming) RenameDialog(song.title, onDismiss = { renaming = false }) { t -> renaming = false; vm.rename(song, t) }
    if (trackEditing) TrackEditorDialog(vm, song) { trackEditing = false }
    if (details) DetailsDialog(song) { details = false }
    if (deleting) AlertDialog(
        onDismissRequest = { deleting = false },
        text = { Text(stringResource(R.string.music_delete_ask, song.title)) },
        confirmButton = { TextButton(onClick = { deleting = false; vm.delete(song) }) { Text(stringResource(R.string.music_delete)) } },
        dismissButton = { TextButton(onClick = { deleting = false }) { Text(stringResource(android.R.string.cancel)) } },
    )
}

@Composable
private fun statusLine(song: Song, service: MusicState): String = when (song.status) {
    SongStatus.DONE -> if (song.plan) stringResource(R.string.music_plan_line, song.seed, duration(song.makeSeconds))
        else stringResource(R.string.music_done_line, duration(song.seconds), song.seed, duration(song.makeSeconds))
    SongStatus.FAILED -> stringResource(R.string.music_failed, song.error)
    SongStatus.CANCELLED -> stringResource(R.string.music_cancelled)
    SongStatus.WAITING -> when {
        song.id == service.running -> stringResource(R.string.music_making)
        song.id in service.queued -> stringResource(R.string.music_waiting)
        service.connected -> stringResource(R.string.music_interrupted)
        else -> stringResource(R.string.music_waiting)
    }
}

/** The song playing: where it is, a bar to move in it. */
@Composable
private fun PlayerBar(vm: MusicViewModel) {
    var pos by remember { mutableIntStateOf(0) }
    var len by remember { mutableIntStateOf(0) }
    var dragging by remember { mutableStateOf(false) }
    var drag by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(Unit) {
        while (true) {
            val (p, l) = vm.position()
            if (!dragging) pos = p
            len = l
            delay(250)
        }
    }
    Row(Modifier.padding(end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(duration(pos / 1000.0), style = MaterialTheme.typography.labelSmall)
        Slider(
            value = if (dragging) drag else pos.toFloat(),
            onValueChange = { dragging = true; drag = it },
            onValueChangeFinished = { vm.seek(drag.toInt()); pos = drag.toInt(); dragging = false },
            valueRange = 0f..maxOf(1, len).toFloat(),
            modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
        )
        Text(duration(len / 1000.0), style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun RenameDialog(title: String, onDismiss: () -> Unit, onDone: (String) -> Unit) {
    var t by remember { mutableStateOf(title) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.music_rename)) },
        text = { OutlinedTextField(t, { t = it }, singleLine = true) },
        confirmButton = { TextButton(onClick = { onDone(t) }) { Text(stringResource(android.R.string.ok)) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(android.R.string.cancel)) } },
    )
}

/** What the song was made from and how: the style, the words, the seed and settings, the stages' times. */
@Composable
private fun DetailsDialog(song: Song, onDismiss: () -> Unit) {
    val j = song.job
    val r = song.result
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(song.title) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text(song.style, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                Text(song.lyrics.ifEmpty { stringResource(R.string.music_instrumental) }, style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace, modifier = Modifier.padding(top = 6.dp))
                HorizontalDivider(Modifier.padding(vertical = 8.dp))
                Text(stringResource(R.string.music_details_settings, j.optString("mode"), j.optLong("seed"),
                    j.optDouble("cfg", 1.0), j.optInt("ode_steps"), j.optInt("seconds")) +
                    (if (j.optString("ar") == "q4") " · Q4" else "") +   // (the 4-bit score and codes)
                    // (the result's when there is one, else the request's; nar_weights "q8": a song from before the levels)
                    (if ((r ?: j).let { it.optString("nar_precision") == "fast" || it.optString("nar_weights") == "q8" }) " · 8-bit" else ""),
                    style = MaterialTheme.typography.bodySmall)
                if (r != null) {
                    val t = r.optJSONObject("times")
                    if (t != null) Text(stringResource(R.string.music_details_times,
                        duration(t.optDouble("score")), duration(t.optDouble("codes")), duration(t.optDouble("prefix")),
                        duration(t.optDouble("acoustic")), duration(t.optDouble("decode")), r.optString("nar")),
                        style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
                    r.optJSONObject("master")?.let { m ->
                        Text(stringResource(R.string.music_details_master, m.optDouble("lufs_before"), m.optDouble("lufs_after"),
                            m.optDouble("peak_after_db")), style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
                    }
                    if (!r.optBoolean("codes_eos", true)) Text(stringResource(R.string.music_details_cut),
                        style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(stringResource(android.R.string.ok)) } },
    )
}

/** 75 -> "1:15", 3725 -> "1:02:05". */
internal fun duration(seconds: Double): String {
    val s = seconds.coerceAtLeast(0.0).toLong()
    return if (s >= 3600) "%d:%02d:%02d".format(s / 3600, s / 60 % 60, s % 60) else "%d:%02d".format(s / 60, s % 60)
}
