package xyz.tifu.geode.ui.music

import xyz.tifu.geode.music.MusicAdapter

import xyz.tifu.geode.ui.settings.RadioOption
import xyz.tifu.geode.ui.settings.RadioRow
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlin.math.roundToInt
import xyz.tifu.geode.R
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.TransferPhase
import xyz.tifu.geode.music.SoundPrecision
import xyz.tifu.geode.music.LyricsLength
import xyz.tifu.geode.music.MusicModels
import xyz.tifu.geode.music.MusicPrefs
import xyz.tifu.geode.music.NarUnit
import xyz.tifu.geode.music.StageSampling
import xyz.tifu.geode.music.YuE2Prompt
import xyz.tifu.geode.ui.settings.Choice
import xyz.tifu.geode.ui.settings.Choices
import xyz.tifu.geode.ui.settings.FloatSlider
import xyz.tifu.geode.ui.settings.Section
import xyz.tifu.geode.ui.settings.SwitchRow
import xyz.tifu.geode.util.formatBytes

/**
 * The music's own settings (the Music screen's ⚙, the drawer's Music settings): the song model - downloaded here from
 * Hugging Face, kept apart from the chat's models -, how it makes a song (the defaults are the settings the model was
 * made for: 32 steps, the score with chords, guidance per song), what computes it, the mastering, and the lyrics
 * writer: which chat model writes, how long, with stress marks and breaks (the language: the writer's own window).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MusicSettingsScreen(vm: MusicViewModel, back: () -> Unit) {
    val p by vm.prefs.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) {
        vm.refresh()
        vm.loadCatalog()
    }
    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.nav_music_settings)) }, navigationIcon = {
            IconButton(onClick = back) {
                Icon(Icons.AutoMirrored.Outlined.ArrowBack, contentDescription = stringResource(R.string.music_back))
            }
        })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            ModelRows(vm, p)
            AdapterRows(vm)
            GenerationRows(vm, p)
            PerformanceRows(vm, p)
            MasteringRows(vm, p)
            WriterRows(vm, p)
            Spacer(Modifier.height(32.dp))
        }
    }
}

// ---- the song model ----

@Composable
private fun ModelRows(vm: MusicViewModel, p: MusicPrefs) {
    val models by vm.models.collectAsStateWithLifecycle()
    val partials by vm.partials.collectAsStateWithLifecycle()
    val catalog by vm.catalog.collectAsStateWithLifecycle()
    val transfer by vm.transfer.collectAsStateWithLifecycle()
    var deleting by remember { mutableStateOf<InstalledModel?>(null) }
    Section(stringResource(R.string.music_section_model))
    Text(stringResource(R.string.music_model_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    val selected = models.firstOrNull { it.id == p.modelId } ?: models.firstOrNull()
    models.forEach { m ->
        RadioRow(m.id == selected?.id, { vm.setPrefs { it.copy(modelId = m.id) } }, Modifier.padding(top = 2.dp)) {
            Column(Modifier.weight(1f)) {
                Text(m.title, style = MaterialTheme.typography.bodyLarge)
                Text(formatBytes(m.manifest.bytes) + if (m.manifest.files.any { it.name == MusicModels.AR_F16 })
                    " · " + stringResource(R.string.music_model_npu) else "",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            TextButton(onClick = { deleting = m }) { Text(stringResource(R.string.music_delete)) }
        }
    }
    // the download running (the song model's), its progress and Cancel
    transfer?.takeIf { it.music }?.let { t ->
        Card(Modifier.fillMaxWidth().padding(top = 8.dp)) {
            Column(Modifier.padding(12.dp)) {
                Text(t.title, style = MaterialTheme.typography.titleSmall)
                when (t.phase) {
                    TransferPhase.FAILED -> Text(t.error.orEmpty(), color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall)
                    TransferPhase.DONE -> Text(stringResource(R.string.music_model_installed), style = MaterialTheme.typography.bodySmall)
                    TransferPhase.CANCELLED -> Text(stringResource(R.string.music_cancelled), style = MaterialTheme.typography.bodySmall)
                    else -> {
                        LinearProgressIndicator(progress = { t.fraction }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
                        Text(stringResource(R.string.music_download_progress, formatBytes(t.done), formatBytes(t.total),
                            formatBytes(t.bytesPerSecond)), style = MaterialTheme.typography.bodySmall)
                        TextButton(onClick = vm::cancelDownload) { Text(stringResource(R.string.music_cancel)) }
                    }
                }
            }
        }
    }
    partials.forEach { part ->
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.music_download_partial, part.id, formatBytes(part.bytes)), Modifier.weight(1f),
                style = MaterialTheme.typography.bodySmall)
            if (transfer?.active != true) TextButton(onClick = { vm.resumeDownload(part) }) { Text(stringResource(R.string.music_resume)) }
            TextButton(onClick = { vm.deletePartial(part) }) { Text(stringResource(R.string.music_delete)) }
        }
    }
    // the packs on Hugging Face
    Text(stringResource(R.string.music_model_download), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 12.dp))
    when {
        catalog.loading -> CircularProgressIndicator(Modifier.padding(8.dp).size(20.dp), strokeWidth = 2.dp)
        catalog.error != null -> Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.music_catalog_error, catalog.error.orEmpty()), Modifier.weight(1f),
                color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = vm::loadCatalog) { Text(stringResource(R.string.music_retry)) }
        }
        else -> {
            val packs = catalog.packs.orEmpty()
            val best = MusicModels.recommended(packs)
            // (looked up again when a model comes or goes, or a download starts or ends - not on every frame of one)
            val free = remember(models, partials, transfer?.active) { vm.freeBytes() }
            packs.forEach { pk ->
                val installed = models.any { it.id == xyz.tifu.geode.models.ModelStore.idFor(pk.pack.repo, pk.pack.path) }
                Card(Modifier.fillMaxWidth().padding(top = 6.dp)) {
                    Column(Modifier.padding(12.dp)) {
                        Text(pk.manifest.name.ifEmpty { pk.pack.label } + if (pk == best) " · " + stringResource(R.string.music_recommended) else "",
                            style = MaterialTheme.typography.titleSmall)
                        if (pk.manifest.description.isNotEmpty()) Text(pk.manifest.description, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(stringResource(R.string.music_pack_size, formatBytes(pk.pack.bytes), formatBytes(free)),
                            style = MaterialTheme.typography.bodySmall)
                        val room = free > pk.pack.bytes + (512L shl 20)
                        Button(onClick = { vm.download(pk) }, enabled = !installed && room && transfer?.active != true,
                            modifier = Modifier.padding(top = 4.dp)) {
                            Text(stringResource(if (installed) R.string.music_model_installed else R.string.music_download))
                        }
                        if (!room && !installed) Text(stringResource(R.string.music_no_room), color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            Text(stringResource(R.string.music_license), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
        }
    }
    deleting?.let { m ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            text = { Text(stringResource(R.string.music_model_delete_ask, m.title, formatBytes(m.manifest.bytes))) },
            confirmButton = { TextButton(onClick = { vm.deleteModel(m); deleting = null }) { Text(stringResource(R.string.music_delete)) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text(stringResource(android.R.string.cancel)) } },
        )
    }
}

// ---- the instrumental adapter (MusicAdapter) ----

@Composable
private fun AdapterRows(vm: MusicViewModel) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val d by MusicAdapter.download.collectAsStateWithLifecycle()
    val service by vm.service.collectAsStateWithLifecycle()
    val library by vm.library.collectAsStateWithLifecycle()
    var changed by remember { mutableStateOf(0) }
    val installed = remember(d.running, changed) { MusicAdapter.installed(context) }
    // (looked at again when a song starts or ends: the first song with the adapter prepares the weights)
    val merged = remember(d.running, changed, service.running, service.queued) { MusicAdapter.mergedBytes(context) }
    // while a song is made or waits, the adapter and its weights stay: a song preparing or reading them would fail
    val busy = service.running.isNotEmpty() || service.queued.isNotEmpty() ||
        library.any { it.status == xyz.tifu.geode.music.SongStatus.WAITING }
    Section(stringResource(R.string.music_section_adapter))
    Text(stringResource(R.string.music_adapter_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    Card(Modifier.fillMaxWidth().padding(top = 6.dp)) {
        Column(Modifier.padding(12.dp)) {
            Text(stringResource(R.string.music_adapter_name), style = MaterialTheme.typography.titleSmall)
            Text(stringResource(R.string.music_adapter_size, formatBytes(MusicAdapter.SIZE), formatBytes(MusicAdapter.MERGED_BYTES)),
                style = MaterialTheme.typography.bodySmall)
            when {
                d.running -> {
                    LinearProgressIndicator(progress = { d.done.toFloat() / MusicAdapter.SIZE }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
                    TextButton(onClick = { MusicAdapter.cancel() }) { Text(stringResource(R.string.music_cancel)) }
                }
                installed -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.music_model_installed), Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { MusicAdapter.delete(context); changed++ }, enabled = !busy) {
                        Text(stringResource(R.string.music_delete))
                    }
                }
                else -> Button(onClick = { MusicAdapter.start(context) }, modifier = Modifier.padding(top = 4.dp)) {
                    Text(stringResource(R.string.music_download))
                }
            }
            d.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            if (merged > 0) Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.music_adapter_merged, formatBytes(merged)), Modifier.weight(1f),
                    style = MaterialTheme.typography.bodySmall)
                TextButton(onClick = { MusicAdapter.clearMerged(context); changed++ }, enabled = !busy) {
                    Text(stringResource(R.string.music_adapter_clear))
                }
            }
            Text(stringResource(R.string.music_adapter_license), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
        }
    }
}

// ---- how a song is made ----

@Composable
private fun GenerationRows(vm: MusicViewModel, p: MusicPrefs) {
    Section(stringResource(R.string.music_section_generation))
    // guidance: 0 = the mode's own
    FloatSlider(p.cfg, 0f..2f, 0.05f, { v ->
        if (v < 1f) stringResource(R.string.music_cfg_auto, YuE2Prompt.defaultCfg(p.mode))
        else stringResource(R.string.music_cfg, v)
    }) { v -> vm.setPrefs { it.copy(cfg = if (v < 1f) 0f else v) } }
    Text(stringResource(R.string.music_cfg_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)

    Choices(MusicPrefs.STEPS, p.odeSteps, { stringResource(R.string.music_steps, it) }) { v -> vm.setPrefs { it.copy(odeSteps = v) } }
    Text(stringResource(R.string.music_steps_desc), style = MaterialTheme.typography.bodySmall,
        color = if (p.odeSteps < 32) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)

    // the score and the codes from 4-bit weights: quicker, less precise (off: the best)
    SwitchRow(stringResource(R.string.music_ar_q4), stringResource(R.string.music_ar_q4_desc), p.arQ4,
        { v -> vm.setArQ4(v) }, warn = p.arQ4)
    // the sound stage's precision: balanced by default (losses a fraction of the model's own bf16 noise), fast an
    // experiment (8-bit weights: noticeably less precise)
    Text(stringResource(R.string.music_precision), style = MaterialTheme.typography.bodyLarge,
        modifier = Modifier.padding(top = 8.dp))
    Choice(listOf(stringResource(R.string.music_precision_best), stringResource(R.string.music_precision_balanced),
        stringResource(R.string.music_precision_fast)), p.soundPrecision.ordinal, true) { i ->
        vm.setSoundPrecision(SoundPrecision.entries[i])
    }
    Text(stringResource(when (p.soundPrecision) {
        SoundPrecision.BEST -> R.string.music_precision_best_desc
        SoundPrecision.BALANCED -> R.string.music_precision_balanced_desc
        SoundPrecision.FAST -> R.string.music_precision_fast_desc
    }), style = MaterialTheme.typography.bodySmall,
        color = if (p.soundPrecision == SoundPrecision.FAST) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)

    Choices(MusicPrefs.LENGTHS, p.maxSeconds, { if (it == 0) stringResource(R.string.music_length_auto) else stringResource(R.string.music_length_max, duration(it.toDouble())) }) { v ->
        vm.setPrefs { it.copy(maxSeconds = v) }
    }
    Text(stringResource(R.string.music_length_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)

    SwitchRow(stringResource(R.string.music_seed_random), stringResource(R.string.music_seed_random_desc), p.randomSeed,
        { v -> vm.setPrefs { it.copy(randomSeed = v) } })
    if (!p.randomSeed) NumberField(stringResource(R.string.music_seed), p.seed) { v -> vm.setPrefs { it.copy(seed = v) } }

    SwitchRow(stringResource(R.string.music_sampling_default), stringResource(R.string.music_sampling_default_desc),
        p.defaultSampling, { v -> vm.setPrefs { it.copy(defaultSampling = v) } })
    if (!p.defaultSampling) {
        Text(stringResource(R.string.music_sampling_score), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 8.dp))
        SamplingRows(p.abc, StageSampling.ABC) { s -> vm.setPrefs { it.copy(abc = s) } }
        Text(stringResource(R.string.music_sampling_sound), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 8.dp))
        SamplingRows(p.semantic, StageSampling.SEMANTIC) { s -> vm.setPrefs { it.copy(semantic = s) } }
    }
}

@Composable
private fun SamplingRows(s: StageSampling, d: StageSampling, set: (StageSampling) -> Unit) {
    FloatSlider(s.temperature, 0.1f..1.5f, 0.05f, { stringResource(R.string.music_temperature, it) }) { v -> set(s.copy(temperature = v)) }
    FloatSlider(s.topP, 0.5f..1f, 0.01f, { stringResource(R.string.music_top_p, it) }) { v -> set(s.copy(topP = v)) }
    Choices(listOf(10, 20, 30, 40, 50, 100, 200, 500), s.topK, { stringResource(R.string.music_top_k, it) }) { v -> set(s.copy(topK = v)) }
    FloatSlider(s.repetitionPenalty, 1f..1.5f, 0.005f, { stringResource(R.string.music_repetition, it) }) { v -> set(s.copy(repetitionPenalty = v)) }
    if (s != d) TextButton(onClick = { set(d) }) { Text(stringResource(R.string.music_sampling_reset)) }
}

// ---- what computes it ----

@Composable
private fun PerformanceRows(vm: MusicViewModel, p: MusicPrefs) {
    Section(stringResource(R.string.music_section_performance))
    Text(stringResource(R.string.music_nar_unit), style = MaterialTheme.typography.bodyLarge)
    Choice(listOf(stringResource(R.string.music_unit_auto), stringResource(R.string.music_unit_npu), stringResource(R.string.music_unit_gpu),
        stringResource(R.string.music_unit_cpu)), p.narUnit.ordinal, true) { i -> vm.setPrefs { it.copy(narUnit = NarUnit.entries[i]) } }
    Text(stringResource(R.string.music_nar_unit_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    Choices(listOf(0, 1, 2, 3, 4, 6, 8), p.threads, { if (it == 0) stringResource(R.string.music_threads_auto) else stringResource(R.string.music_threads, it) }) { v ->
        vm.setPrefs { it.copy(threads = v) }
    }
    SwitchRow(stringResource(R.string.music_unload_chat), stringResource(R.string.music_unload_chat_desc), p.unloadChat,
        { v -> vm.setPrefs { it.copy(unloadChat = v) } })
}

// ---- the mastering ----

@Composable
private fun MasteringRows(vm: MusicViewModel, p: MusicPrefs) {
    Section(stringResource(R.string.music_section_mastering))
    SwitchRow(stringResource(R.string.music_master), stringResource(R.string.music_master_desc), p.master,
        { v -> vm.setPrefs { it.copy(master = v) } })
    if (p.master) {
        FloatSlider(p.targetLufs, -20f..-8f, 0.5f, { stringResource(R.string.music_target_lufs, it) }) { v -> vm.setPrefs { it.copy(targetLufs = v) } }
        FloatSlider(p.maxGainDb, 0f..12f, 0.5f, { stringResource(R.string.music_max_gain, it) }) { v -> vm.setPrefs { it.copy(maxGainDb = v) } }
    }
    Text(stringResource(R.string.music_format), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 8.dp))
    Choice(listOf("FLAC 24", "FLAC 16", "WAV 24", "WAV 16"), MusicPrefs.FORMATS.indexOf(p.format to p.bits).coerceAtLeast(0)) { i ->
        vm.setPrefs { it.copy(format = MusicPrefs.FORMATS[i].first, bits = MusicPrefs.FORMATS[i].second) }
    }
}

// ---- the lyrics writer ----

@Composable
private fun WriterRows(vm: MusicViewModel, p: MusicPrefs) {
    val chatModels by vm.chatModels.collectAsStateWithLifecycle()
    val chatSelected by vm.chatSelected.collectAsStateWithLifecycle()
    Section(stringResource(R.string.music_section_writer))
    Text(stringResource(R.string.music_writer_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    Text(stringResource(R.string.music_writer_model_title), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 8.dp))
    val chat = chatModels.firstOrNull { it.id == chatSelected }
    RadioOption(stringResource(R.string.music_writer_chat_model, chat?.title ?: "-"), p.writerModel.isEmpty(),
        MaterialTheme.typography.bodyMedium) {
        vm.setPrefs { it.copy(writerModel = "") }
    }
    chatModels.forEach { m ->
        RadioOption(m.title, p.writerModel == m.id, MaterialTheme.typography.bodyMedium) { vm.setPrefs { it.copy(writerModel = m.id) } }
    }
    if (chatModels.isEmpty()) Text(stringResource(R.string.music_writer_no_model), color = MaterialTheme.colorScheme.error,
        style = MaterialTheme.typography.bodySmall)
    // (the language is the song's, not a setting: the writer's own window asks it, beside the idea)
    Text(stringResource(R.string.music_writer_length), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 8.dp))
    Choice(lengthLabels(), p.writerLength.ordinal) { i -> vm.setPrefs { it.copy(writerLength = LyricsLength.entries[i]) } }
    SwitchRow(stringResource(R.string.music_writer_stress), stringResource(R.string.music_writer_stress_desc), p.writerStress,
        { v -> vm.setPrefs { it.copy(writerStress = v) } })
    SwitchRow(stringResource(R.string.music_writer_breaks), stringResource(R.string.music_writer_breaks_desc), p.writerBreaks,
        { v -> vm.setPrefs { it.copy(writerBreaks = v) } })
    SwitchRow(stringResource(R.string.music_writer_thinking), stringResource(R.string.music_writer_thinking_desc), p.writerThinking,
        { v -> vm.setPrefs { it.copy(writerThinking = v) } })
    FloatSlider(p.writerTemperature, 0.1f..1.5f, 0.05f, { stringResource(R.string.music_temperature, it) }) { v ->
        vm.setPrefs { it.copy(writerTemperature = v) }
    }
    FloatSlider(p.writerTopP, 0.5f..1f, 0.01f, { stringResource(R.string.music_top_p, it) }) { v -> vm.setPrefs { it.copy(writerTopP = v) } }
    Choices(listOf(10, 20, 40, 64, 100), p.writerTopK, { stringResource(R.string.music_top_k, it) }) { v -> vm.setPrefs { it.copy(writerTopK = v) } }
    Choices(listOf(400, 600, 900, 1200, 2000), p.writerMaxTokens, { stringResource(R.string.music_writer_max_tokens, it) }) { v ->
        vm.setPrefs { it.copy(writerMaxTokens = v) }
    }
}

@Composable
private fun NumberField(label: String, value: Long, onDone: (Long) -> Unit) {
    var t by remember(value) { mutableStateOf(value.toString()) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(t, { s -> t = s.filter { it.isDigit() }.take(18) }, label = { Text(label) }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.weight(1f))
        OutlinedButton(onClick = { t.toLongOrNull()?.let(onDone) }, enabled = t.toLongOrNull() != null && t.toLongOrNull() != value,
            modifier = Modifier.padding(start = 8.dp)) { Text(stringResource(android.R.string.ok)) }
    }
}

