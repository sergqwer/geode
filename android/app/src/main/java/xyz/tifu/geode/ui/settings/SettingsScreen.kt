package xyz.tifu.geode.ui.settings

import androidx.compose.material3.IconButton
import androidx.compose.material3.Icon
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.Icons
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlin.math.roundToInt
import kotlinx.coroutines.launch
import xyz.tifu.geode.R

/**
 * The app's own settings (the drawer's "App settings"): appearance, the screen, answers read aloud, the language,
 * the lock, about. How the models work is ModelSettingsScreen's.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: SettingsViewModel, openDrawer: () -> Unit) {
    val s by vm.settings.collectAsStateWithLifecycle()

    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.settings_title)) }, navigationIcon = {
            IconButton(onClick = openDrawer) {
                Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
            }
        })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {

            // ---- appearance ----
            Section(stringResource(R.string.settings_section_appearance))
            Text(stringResource(R.string.settings_theme), style = MaterialTheme.typography.bodyLarge)
            Choice(
                listOf(R.string.settings_theme_system, R.string.settings_theme_light, R.string.settings_theme_dark)
                    .map { stringResource(it) },
                s.theme, vm::setTheme,
            )
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                SwitchRow(stringResource(R.string.settings_dynamic_color),
                    stringResource(R.string.settings_dynamic_color_desc), s.dynamicColor, vm::setDynamicColor)
            }
            FloatSlider(s.chatTextScale, CHAT_TEXT_MIN..CHAT_TEXT_MAX, 0.05f,
                { stringResource(R.string.settings_chat_text, (it * 100).roundToInt()) }, vm::setChatTextScale)
            Text(stringResource(R.string.settings_chat_text_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)

            // ---- the screen: kept on, and black after a while without a touch ----
            Section(stringResource(R.string.settings_section_screen))
            ScreenRows(s, vm)

            // ---- answers read aloud ----
            Section(stringResource(R.string.settings_section_voice))
            VoiceRows(s, vm)

            Section(stringResource(R.string.settings_section_language))
            LanguageRow(vm)

            // ---- the app's lock ----
            Section(stringResource(R.string.settings_section_privacy))
            LockSection(vm.lock)

            // ---- about ----
            Section(stringResource(R.string.settings_section_about))
            Text(stringResource(R.string.settings_about_version, vm.appVersion), style = MaterialTheme.typography.bodyMedium)
            Text(stringResource(R.string.settings_about_engine, vm.engineVersion), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(stringResource(R.string.settings_about_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
            // the one way the chats leave the phone, when the user takes it
            Text(stringResource(R.string.settings_about_services), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** "no limit" for 0, else "N tokens". */
@Composable
internal fun tokens(n: Int): String =
    if (n == 0) stringResource(R.string.settings_no_limit) else stringResource(R.string.settings_tokens, n)

/** A slider over a list of values (the label shows the one under the finger; written when it lifts). */
@Composable
internal fun Choices(values: List<Int>, value: Int, label: @Composable (Int) -> String, onCommit: (Int) -> Unit) {
    val index = values.indexOf(value).coerceAtLeast(0)
    FloatSlider(index.toFloat(), 0f..(values.size - 1).toFloat(), 1f, { label(values[it.roundToInt()]) }) {
        onCommit(values[it.roundToInt()])
    }
}

@Composable
internal fun SwitchRow(title: String, desc: String, checked: Boolean, onChange: (Boolean) -> Unit,
                       enabled: Boolean = true) {
    Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(desc, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onChange, enabled = enabled)
    }
}

internal const val MIN_MB = 1024
/** The chat's text size, as a share of the system's (the chat's pinch keeps to the same range). */
const val CHAT_TEXT_MIN = 0.7f
const val CHAT_TEXT_MAX = 2f
/** The storage the saved conversations may take (MB; 0: none kept). */
internal val CACHE_MB = listOf(0, 1024, 2048, 4096, 8192, 16384)
internal const val STEP_MB = 64

@Composable
internal fun Section(title: String) {
    HorizontalDivider(Modifier.padding(top = 16.dp))
    Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 12.dp, bottom = 6.dp))
}

/** A slider that writes its value once the finger lifts (DataStore writes are not for every frame). */
@Composable
internal fun FloatSlider(
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    step: Float,
    label: @Composable (Float) -> String,
    onCommit: (Float) -> Unit,
) {
    var v by remember(value) { mutableFloatStateOf(value) }
    fun snap(x: Float) = ((x - range.start) / step).roundToInt() * step + range.start
    Text(label(snap(v)), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 4.dp))
    Slider(
        value = v.coerceIn(range),
        onValueChange = { v = it },
        // the state now, not this composition's: a tap changes the value and finishes before it recomposes
        onValueChangeFinished = { onCommit(snap(v)) },
        valueRange = range,
    )
}

/**
 * Whether Geode keeps the screen on (never, while it answers, all the time it is open), and then the black screen against
 * burn-in: its minutes, or never - which is warned about. The black screen is no lock, and the text says so.
 */
@Composable
private fun ScreenRows(s: xyz.tifu.geode.data.AppSettings, vm: SettingsViewModel) {
    Text(stringResource(R.string.screen_keep), style = MaterialTheme.typography.bodyLarge)
    listOf(
        xyz.tifu.geode.data.AppSettings.KEEP_OFF to R.string.screen_keep_off,
        xyz.tifu.geode.data.AppSettings.KEEP_ANSWERING to R.string.screen_keep_answering,
        xyz.tifu.geode.data.AppSettings.KEEP_ALWAYS to R.string.screen_keep_always,
    ).forEach { (mode, label) ->
        LanguageOption(stringResource(label), s.keepScreen == mode) { vm.setKeepScreen(mode) }
    }
    if (s.keepScreen == xyz.tifu.geode.data.AppSettings.KEEP_OFF) return
    Choices(BLACK_MIN, s.blackScreenMin, { m ->
        if (m == 0) stringResource(R.string.screen_black_never) else stringResource(R.string.screen_black_after, m)
    }, vm::setBlackScreenMin)
    Text(stringResource(R.string.screen_black_desc), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    if (s.blackScreenMin == 0) {
        Text(stringResource(R.string.screen_black_off_warning), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 4.dp))
    }
}

/**
 * Answers read aloud as they are written; what reads them (the phone's text-to-speech, or Geode's own voice: downloaded
 * here, deleted here, its licence shown); the voice (each language's default or one of the engine's for its language;
 * of Geode's voice one of its ten), how fast and how high, an example to hear them by, and the system's text-to-speech
 * settings (the engine, its downloaded voices). A phone without a text-to-speech engine is told so.
 */
@Composable
private fun VoiceRows(s: xyz.tifu.geode.data.AppSettings, vm: SettingsViewModel) {
    val available by vm.speakerAvailable.collectAsState()
    val voices by vm.voices.collectAsState()
    val installed by vm.neuralInstalled.collectAsState()
    val download by vm.neuralDownload.collectAsState()
    val context = androidx.compose.ui.platform.LocalContext.current
    var picking by remember { mutableStateOf(false) }
    var licence by remember { mutableStateOf<String?>(null) }
    val scope = androidx.compose.runtime.rememberCoroutineScope()
    SwitchRow(stringResource(R.string.voice_read_aloud), stringResource(R.string.voice_read_aloud_desc), s.readAloud,
        vm::setReadAloud)
    // ---- what reads: the phone's engine or Geode's own voice ----
    Text(stringResource(R.string.voice_engine), style = MaterialTheme.typography.bodyLarge,
        modifier = Modifier.padding(top = 8.dp))
    LanguageOption(stringResource(R.string.voice_engine_system), s.speechEngine == 0) { vm.setSpeechEngine(0) }
    LanguageOption(stringResource(R.string.voice_engine_geode), s.speechEngine == 1) { vm.setSpeechEngine(1) }
    val geode = s.speechEngine == 1
    if (geode) {
        Text(stringResource(R.string.voice_engine_geode_desc), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        when {
            installed -> Row(horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
                TextButton(onClick = vm::deleteNeural) { Text(stringResource(R.string.voice_delete)) }
                TextButton(onClick = { scope.launch { licence = vm.neuralLicence() } }) {
                    Text(stringResource(R.string.voice_license))
                }
            }
            download.running -> {
                val f = if (download.total > 0) download.done.toFloat() / download.total else 0f
                androidx.compose.material3.LinearProgressIndicator(progress = { f },
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.voice_downloading, (f * 100).roundToInt()),
                        style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    TextButton(onClick = vm::cancelNeural) { Text(stringResource(R.string.chat_cancel)) }
                }
            }
            else -> {
                OutlinedButton(onClick = vm::downloadNeural, modifier = Modifier.padding(top = 4.dp)) {
                    Text(stringResource(R.string.voice_download,
                        xyz.tifu.geode.util.formatBytes(xyz.tifu.geode.media.NeuralVoice.TOTAL)))
                }
                download.error?.let {
                    Text(stringResource(R.string.voice_download_failed, it), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error)
                }
            }
        }
        Text(stringResource(R.string.voice_license_note), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    val picked = voices.firstOrNull { it.name == s.speechVoice }
    Column(Modifier.fillMaxWidth().clickable {
        if (!geode) vm.prepareVoices()
        picking = true
    }.padding(vertical = 8.dp)) {
        Text(stringResource(R.string.voice_voice), style = MaterialTheme.typography.bodyLarge)
        Text(if (geode) neuralLabel(s.neuralVoice)
            else if (s.speechVoice.isEmpty()) stringResource(R.string.voice_auto)
            else picked?.let { voiceLabel(it) } ?: s.speechVoice,
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
    }
    FloatSlider(s.speechRate, 0.5f..2f, 0.1f, { stringResource(R.string.voice_rate, (it * 100).roundToInt()) },
        vm::setSpeechRate)
    FloatSlider(s.speechPitch, 0.5f..2f, 0.1f, { stringResource(R.string.voice_pitch, (it * 100).roundToInt()) },
        vm::setSpeechPitch)
    val sample = stringResource(R.string.voice_sample)
    Row(horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = { vm.sampleVoice(sample) }) { Text(stringResource(R.string.voice_listen)) }
        TextButton(onClick = {
            // the system's text-to-speech page: the engine, its voices to download (not on every phone)
            runCatching {
                context.startActivity(android.content.Intent("com.android.settings.TTS_SETTINGS")
                    .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK))
            }
        }) { Text(stringResource(R.string.voice_engine_settings)) }
    }
    if (available == false) {
        Text(stringResource(R.string.voice_none), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error)
    }
    licence?.let { text ->
        AlertDialog(
            onDismissRequest = { licence = null },
            title = { Text(stringResource(R.string.voice_license)) },
            text = {
                Text(text, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.verticalScroll(rememberScrollState()))
            },
            confirmButton = { TextButton(onClick = { licence = null }) { Text(stringResource(android.R.string.ok)) } },
        )
    }
    if (!picking) return
    if (geode) {
        // Geode's voice: its ten styles, each reading every language it speaks
        AlertDialog(
            onDismissRequest = { picking = false },
            title = { Text(stringResource(R.string.voice_voice)) },
            text = {
                LazyColumn {
                    items((0 until xyz.tifu.geode.media.NeuralVoice.VOICES).toList()) { v ->
                        LanguageOption(neuralLabel(v), s.neuralVoice == v) {
                            vm.setNeuralVoice(v)
                            picking = false
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { picking = false }) { Text(stringResource(R.string.chat_cancel)) } },
        )
        return
    }
    AlertDialog(
        onDismissRequest = { picking = false },
        title = { Text(stringResource(R.string.voice_voice)) },
        text = {
            Column {
                Text(stringResource(R.string.voice_voice_desc), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (available == false) {
                    Text(stringResource(R.string.voice_none), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp))
                } else if (voices.isEmpty()) {
                    CircularProgressIndicatorSmall()
                }
                LazyColumn(Modifier.padding(top = 8.dp)) {
                    item {
                        LanguageOption(stringResource(R.string.voice_auto), s.speechVoice.isEmpty()) {
                            vm.setSpeechVoice("")
                            picking = false
                        }
                    }
                    items(voices) { v ->
                        LanguageOption(voiceLabel(v), s.speechVoice == v.name) {
                            vm.setSpeechVoice(v.name)
                            picking = false
                        }
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = { TextButton(onClick = { picking = false }) { Text(stringResource(R.string.chat_cancel)) } },
    )
}

/** One of Geode's voice's styles: 0-4 the female voices, 5-9 the male ones. */
@Composable
private fun neuralLabel(v: Int): String =
    if (v < 5) stringResource(R.string.voice_female, v + 1) else stringResource(R.string.voice_male, v - 4)

/** A voice as the list shows it: its language, the engine's name for it, and whether it needs the internet. */
@Composable
private fun voiceLabel(v: xyz.tifu.geode.media.Speaker.VoiceInfo): String {
    val lang = v.locale.getDisplayName(java.util.Locale.getDefault()).replaceFirstChar { it.titlecase() }
    // the engine's names (en-us-x-iol-local) say more than the language alone; their language part is repeated
    val name = v.name.removePrefix(v.locale.toLanguageTag().lowercase() + "-").removePrefix("x-")
    return "$lang · $name" + if (v.network) " (" + stringResource(R.string.voice_network) + ")" else ""
}

@Composable
private fun CircularProgressIndicatorSmall() {
    androidx.compose.material3.CircularProgressIndicator(Modifier.padding(top = 8.dp).size(18.dp), strokeWidth = 2.dp)
}

/** The black screen's minutes without a touch; 0 = never (last on the slider). */
private val BLACK_MIN = listOf(1, 2, 5, 10, 15, 30, 0)

/**
 * The app's language: the phone's by default (English where Geode has no translation), or one picked from the list
 * whatever the phone's is. The row shows the language the texts are in now.
 */
@Composable
private fun LanguageRow(vm: SettingsViewModel) {
    var picked by remember { mutableStateOf(Languages.canonical(vm.language())) }
    var open by remember { mutableStateOf(false) }
    val system = stringResource(R.string.settings_language_system)
    // the system's choice when nothing is picked: the first of the phone's languages Geode has, else English
    val shown = LocalConfiguration.current.locales.toLanguageTags().split(',')
        .firstNotNullOfOrNull { Languages.name(it) } ?: "English"
    Column(Modifier.fillMaxWidth().clickable { open = true }.padding(vertical = 8.dp)) {
        Text(stringResource(R.string.settings_language_app), style = MaterialTheme.typography.bodyLarge)
        Text(if (picked.isEmpty()) "$system · $shown" else Languages.name(picked) ?: picked,
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
    }
    if (!open) return
    val choose = { tag: String ->
        open = false
        picked = tag
        vm.setLanguage(tag)
    }
    AlertDialog(
        onDismissRequest = { open = false },
        title = { Text(stringResource(R.string.settings_language_app)) },
        text = {
            LazyColumn {
                item { LanguageOption(system, picked.isEmpty()) { choose("") } }
                items(Languages.ALL) { (tag, name) -> LanguageOption(name, picked == tag) { choose(tag) } }
            }
        },
        confirmButton = {},
        dismissButton = { TextButton(onClick = { open = false }) { Text(stringResource(R.string.chat_cancel)) } },
    )
}

@Composable
private fun LanguageOption(name: String, selected: Boolean, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().clickable(onClick = onClick).padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically) {
        RadioButton(selected = selected, onClick = onClick)
        Text(name, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(start = 8.dp))
    }
}

/** Experts read for each word: the model's own 8, or the last 2 / 4 left out when not in memory */
internal val WORD_EXPERTS = listOf(8, 6, 4)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun Choice(labels: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        labels.forEachIndexed { i, l ->
            SegmentedButton(
                selected = i == selected,
                onClick = { onSelect(i) },
                shape = SegmentedButtonDefaults.itemShape(i, labels.size),
            ) { Text(l, maxLines = 1) }
        }
    }
}
