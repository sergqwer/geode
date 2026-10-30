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
import androidx.compose.material3.CircularProgressIndicator
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
import xyz.tifu.geode.R
import xyz.tifu.geode.util.formatBytes
import xyz.tifu.geode.util.formatMb

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: SettingsViewModel, openDrawer: () -> Unit) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val test by vm.test.collectAsStateWithLifecycle()
    val shape by vm.shape.collectAsStateWithLifecycle()
    val gpuStatus by vm.gpuStatus.collectAsStateWithLifecycle()

    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.settings_title)) }, navigationIcon = {
            IconButton(onClick = openDrawer) {
                Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
            }
        })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {

            // ---- memory ----
            Section(stringResource(R.string.settings_section_memory))
            val maxMb = maxOf(s.limitMb, s.recommendedMb, (vm.totalMemBytes * 85 / 100 shr 20).toInt())
            var budget by remember(s.effectiveBudgetMb) { mutableFloatStateOf(s.effectiveBudgetMb.toFloat()) }
            val budgetMb = (budget / STEP_MB).roundToInt() * STEP_MB
            Text(stringResource(R.string.settings_memory_budget), style = MaterialTheme.typography.bodyLarge)
            Text(
                if (s.recommendedMb > 0) stringResource(R.string.settings_memory_budget_value, formatMb(budgetMb),
                    formatMb(s.recommendedMb))
                else stringResource(R.string.settings_memory_budget_value_untested, formatMb(budgetMb)),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Slider(
                value = budget.coerceIn(MIN_MB.toFloat(), maxMb.toFloat()),
                onValueChange = { budget = it },
                // the state now, not this composition's: a tap changes and finishes before it recomposes
                onValueChangeFinished = { vm.setMemoryBudget((budget / STEP_MB).roundToInt() * STEP_MB) },
                valueRange = MIN_MB.toFloat()..maxMb.toFloat(),
            )
            if (s.recommendedMb > 0 && budgetMb > s.recommendedMb) {
                Text(stringResource(R.string.settings_memory_warning), color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall)
            }
            val split = shape?.let { vm.split(budgetMb, s.contextLength, it) }
            if (split != null) {
                Text(stringResource(R.string.settings_memory_split, formatBytes(split.weights),
                    formatBytes(split.cacheStart)), style = MaterialTheme.typography.bodySmall,
                    color = if (split.cacheStart > 0) MaterialTheme.colorScheme.onSurfaceVariant
                    else MaterialTheme.colorScheme.error)
            }
            Spacer(Modifier.height(8.dp))
            Text(stringResource(R.string.settings_memory_test_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(Modifier.padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = vm::runMemoryTest, enabled = !test.running) {
                    Text(stringResource(R.string.settings_memory_test))
                }
                if (test.running) {
                    CircularProgressIndicator(Modifier.padding(start = 12.dp).size(18.dp), strokeWidth = 2.dp)
                    Text(stringResource(R.string.settings_memory_test_running, formatBytes(test.heldBytes)),
                        Modifier.padding(start = 8.dp), style = MaterialTheme.typography.bodySmall)
                }
            }
            if (!test.running && s.limitMb > 0) {
                Text(stringResource(R.string.settings_memory_test_result, formatMb(s.recommendedMb), formatMb(s.limitMb)),
                    style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
                if (s.memoryBudgetMb > 0 && s.memoryBudgetMb != s.recommendedMb) {
                    TextButton(onClick = { vm.setMemoryBudget(0) }) {
                        Text(stringResource(R.string.settings_memory_use_recommended))
                    }
                }
            }
            test.error?.let {
                Text(stringResource(R.string.settings_memory_test_failed, it), color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall)
            }

            // ---- generation ----
            Section(stringResource(R.string.settings_section_generation))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.settings_thinking), style = MaterialTheme.typography.bodyLarge)
                    Text(stringResource(R.string.settings_thinking_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.thinking, onCheckedChange = vm::setThinking)
            }
            // the sampling of the mode the switch is in: Qwen3.6 recommends different values for each
            Text(stringResource(if (s.thinking) R.string.settings_sampling_think else R.string.settings_sampling_plain),
                style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(top = 8.dp))
            if (s.thinking) {
                FloatSlider(s.thinkTemperature, 0f..1.5f, 0.05f, { stringResource(R.string.settings_temperature, it) },
                    vm::setThinkTemperature)
                FloatSlider(s.thinkTopP, 0.05f..1f, 0.05f, { stringResource(R.string.settings_top_p, it) },
                    vm::setThinkTopP)
                FloatSlider(s.thinkTopK.toFloat(), 0f..100f, 1f,
                    { stringResource(R.string.settings_top_k, it.roundToInt()) }) { vm.setThinkTopK(it.roundToInt()) }
            } else {
                FloatSlider(s.temperature, 0f..1.5f, 0.05f, { stringResource(R.string.settings_temperature, it) },
                    vm::setTemperature)
                FloatSlider(s.topP, 0.05f..1f, 0.05f, { stringResource(R.string.settings_top_p, it) }, vm::setTopP)
                FloatSlider(s.topK.toFloat(), 0f..100f, 1f, { stringResource(R.string.settings_top_k, it.roundToInt()) }) {
                    vm.setTopK(it.roundToInt())
                }
            }
            FloatSlider(s.presencePenalty, 0f..2f, 0.1f, { stringResource(R.string.settings_presence, it) },
                vm::setPresencePenalty)
            Text(stringResource(R.string.settings_presence_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            TextButton(onClick = vm::resetSampling) { Text(stringResource(R.string.settings_sampling_reset)) }
            if (s.thinking) {
                Choices(SettingsViewModel.THINK_BUDGETS, s.thinkBudget,
                    { stringResource(R.string.settings_think_budget, tokens(it)) }, vm::setThinkBudget)
                Text(stringResource(R.string.settings_think_budget_desc), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Choices(SettingsViewModel.ANSWER_LIMITS, s.maxAnswer,
                { stringResource(R.string.settings_max_answer, tokens(it)) }, vm::setMaxAnswer)
            val ctxIndex = SettingsViewModel.CONTEXT_CHOICES.indexOf(s.contextLength).coerceAtLeast(0)
            FloatSlider(ctxIndex.toFloat(), 0f..(SettingsViewModel.CONTEXT_CHOICES.size - 1).toFloat(), 1f,
                { stringResource(R.string.settings_context, SettingsViewModel.CONTEXT_CHOICES[it.roundToInt()]) }) {
                vm.setContextLength(SettingsViewModel.CONTEXT_CHOICES[it.roundToInt()])
            }
            Text(stringResource(R.string.settings_context_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (split != null) {
                if (split.maxContext < s.contextLength) {
                    Text(stringResource(R.string.settings_context_capped, split.maxContext),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                }
                // what the context takes at its longest (int8 KV), and what the expert cache keeps then
                Text(stringResource(R.string.settings_context_memory, formatBytes(split.kvAtMax), split.maxContext,
                    formatBytes(split.kvPerThousand), formatBytes(split.cacheAtMax)),
                    style = MaterialTheme.typography.bodySmall,
                    color = if (split.cacheAtMax > 0) MaterialTheme.colorScheme.onSurfaceVariant
                    else MaterialTheme.colorScheme.error)
            }
            Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.settings_auto_compress), style = MaterialTheme.typography.bodyLarge)
                    Text(stringResource(R.string.settings_auto_compress_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.autoCompress, onCheckedChange = vm::setAutoCompress)
            }

            // ---- the conversations kept on the storage ----
            val cacheBytes by vm.cacheBytes.collectAsStateWithLifecycle()
            androidx.compose.runtime.LaunchedEffect(Unit) { vm.refreshCache() }
            Text(stringResource(R.string.settings_state_cache), style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 12.dp))
            Choices(CACHE_MB, s.stateCacheMb, { mb ->
                if (mb == 0) stringResource(R.string.settings_state_cache_off)
                else xyz.tifu.geode.util.formatBytes(mb.toLong() shl 20)
            }, vm::setStateCacheMb)
            Text(stringResource(R.string.settings_state_cache_desc, xyz.tifu.geode.util.formatBytes(cacheBytes)),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (cacheBytes > 0) {
                androidx.compose.material3.TextButton(onClick = vm::clearCache) {
                    Text(stringResource(R.string.settings_state_cache_clear))
                }
            }

            // ---- images ----
            Text(stringResource(R.string.settings_image_detail), style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 12.dp))
            Choice(listOf(stringResource(R.string.settings_image_detail_fast),
                stringResource(R.string.settings_image_detail_fine),
                stringResource(R.string.settings_image_detail_most)), s.imageDetail, vm::setImageDetail)
            Text(stringResource(R.string.settings_image_detail_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)

            // ---- performance ----
            Section(stringResource(R.string.settings_section_performance))
            FloatSlider(s.computeThreads.toFloat(), 0f..8f, 1f, {
                val n = it.roundToInt()
                stringResource(R.string.settings_threads,
                    if (n == 0) stringResource(R.string.settings_threads_auto) else n.toString())
            }) { vm.setComputeThreads(it.roundToInt()) }
            FloatSlider(s.ioThreads.toFloat(), 0f..6f, 1f, {
                val n = it.roundToInt()
                stringResource(R.string.settings_io_threads,
                    if (n == 0) stringResource(R.string.settings_threads_auto) else n.toString())
            }) { vm.setIoThreads(it.roundToInt()) }
            FloatSlider(s.helperThreads.toFloat(), 0f..6f, 1f, {
                val n = it.roundToInt()
                stringResource(R.string.settings_helpers,
                    if (n == 0) stringResource(R.string.settings_helpers_none) else n.toString())
            }) { vm.setHelperThreads(it.roundToInt()) }
            Text(stringResource(R.string.settings_helpers_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            SwitchRow(stringResource(R.string.settings_prefetch), stringResource(R.string.settings_prefetch_desc),
                s.prefetch, vm::setPrefetch)
            SwitchRow(stringResource(R.string.settings_gpu), stringResource(R.string.settings_gpu_desc), s.gpu,
                vm::setGpu)
            if (s.gpu) {
                FloatSlider(s.gpuShare, 0.1f..0.6f, 0.05f,
                    { stringResource(R.string.settings_gpu_share, (it * 100).roundToInt()) }, vm::setGpuShare)
            }
            gpuStatus?.let {
                Text(stringResource(R.string.settings_gpu_status, it), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            SwitchRow(stringResource(R.string.settings_npu), stringResource(R.string.settings_npu_desc), s.npu, vm::setNpu)
            SwitchRow(stringResource(R.string.settings_route_prior), stringResource(R.string.settings_route_prior_desc),
                s.routePrior, vm::setRoutePrior)
            // experts for each word: 8 the model's own; fewer (left out when not in memory) said to cost accuracy
            Text(stringResource(R.string.settings_word_experts), style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 8.dp))
            Choice(WORD_EXPERTS.map { it.toString() }, WORD_EXPERTS.indexOf(s.wordExperts).coerceAtLeast(0)) {
                vm.setWordExperts(WORD_EXPERTS[it])
            }
            Text(stringResource(R.string.settings_word_experts_desc), style = MaterialTheme.typography.bodySmall,
                color = if (s.wordExperts < 8) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
            Text(stringResource(R.string.settings_engine_reopen), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))

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

            Section(stringResource(R.string.settings_section_language))
            LanguageRow(vm)

            // ---- about ----
            Section(stringResource(R.string.settings_section_about))
            Text(stringResource(R.string.settings_about_version, vm.appVersion), style = MaterialTheme.typography.bodyMedium)
            Text(stringResource(R.string.settings_about_engine, vm.engineVersion), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(stringResource(R.string.settings_about_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** "no limit" for 0, else "N tokens". */
@Composable
private fun tokens(n: Int): String =
    if (n == 0) stringResource(R.string.settings_no_limit) else stringResource(R.string.settings_tokens, n)

/** A slider over a list of values (the label shows the one under the finger; written when it lifts). */
@Composable
private fun Choices(values: List<Int>, value: Int, label: @Composable (Int) -> String, onCommit: (Int) -> Unit) {
    val index = values.indexOf(value).coerceAtLeast(0)
    FloatSlider(index.toFloat(), 0f..(values.size - 1).toFloat(), 1f, { label(values[it.roundToInt()]) }) {
        onCommit(values[it.roundToInt()])
    }
}

@Composable
private fun SwitchRow(title: String, desc: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(desc, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

private const val MIN_MB = 1024
/** The storage the saved conversations may take (MB; 0: none kept). */
private val CACHE_MB = listOf(0, 1024, 2048, 4096, 8192, 16384)
private const val STEP_MB = 64

@Composable
private fun Section(title: String) {
    HorizontalDivider(Modifier.padding(top = 16.dp))
    Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 12.dp, bottom = 6.dp))
}

/** A slider that writes its value once the finger lifts (DataStore writes are not for every frame). */
@Composable
private fun FloatSlider(
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
private val WORD_EXPERTS = listOf(8, 6, 4)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Choice(labels: List<String>, selected: Int, onSelect: (Int) -> Unit) {
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
