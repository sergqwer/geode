package xyz.tifu.geode.ui.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlin.math.roundToInt
import xyz.tifu.geode.R
import xyz.tifu.geode.data.ModelFamily
import xyz.tifu.geode.data.RunModes
import xyz.tifu.geode.models.OFF_AUDIO
import xyz.tifu.geode.models.OFF_DRAFT
import xyz.tifu.geode.models.OFF_IMAGES
import xyz.tifu.geode.models.OFF_VIDEO
import xyz.tifu.geode.ui.runModeLabel
import xyz.tifu.geode.util.formatBytes
import xyz.tifu.geode.util.formatMb

/**
 * How the models work (the drawer's "Model settings"): the memory they may take, how they answer, the conversations
 * kept on the storage, how finely they see images, and the engine's performance options. A row that does nothing for
 * the selected model is left out: the experts' options for a model without experts, image detail for one that sees
 * no images. The app's own settings (appearance, screen, voice, language, privacy) are SettingsScreen's.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSettingsScreen(vm: SettingsViewModel, openDrawer: () -> Unit) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val test by vm.test.collectAsStateWithLifecycle()
    val shape by vm.shape.collectAsStateWithLifecycle()
    val model by vm.model.collectAsStateWithLifecycle()
    val gpuStatus by vm.gpuStatus.collectAsStateWithLifecycle()
    val units by vm.units.collectAsStateWithLifecycle()
    // a model of experts (Qwen3.6-35B-A3B): its expert cache, readers, read ahead, routing (ModelShape: null for a
    // dense model, which has no expert table)
    val experts = (shape?.experts ?: 0) > 0

    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.nav_model_settings)) }, navigationIcon = {
            IconButton(onClick = openDrawer) {
                Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
            }
        })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Text(model?.let { stringResource(R.string.model_settings_selected, it.title) }
                ?: stringResource(R.string.model_settings_none),
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp))

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
            // the selected model's family's sampling (Qwen3.6's when none is selected), of the mode the switch is in:
            // Qwen3.6 and Qwen3.5-2B recommend different values for each (and from each other), Gemma 4 the same
            val family = ModelFamily.of(model)
            val sp = s.samplingOf(family)
            Text(stringResource(if (s.thinking) R.string.settings_sampling_think else R.string.settings_sampling_plain),
                style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(top = 8.dp))
            if (s.thinking) {
                FloatSlider(sp.thinkTemperature, 0f..1.5f, 0.05f, { stringResource(R.string.settings_temperature, it) },
                    vm::setThinkTemperature)
                FloatSlider(sp.thinkTopP, 0.05f..1f, 0.05f, { stringResource(R.string.settings_top_p, it) },
                    vm::setThinkTopP)
                FloatSlider(sp.thinkTopK.toFloat(), 0f..100f, 1f,
                    { stringResource(R.string.settings_top_k, it.roundToInt()) }) { vm.setThinkTopK(it.roundToInt()) }
            } else {
                FloatSlider(sp.temperature, 0f..1.5f, 0.05f, { stringResource(R.string.settings_temperature, it) },
                    vm::setTemperature)
                FloatSlider(sp.topP, 0.05f..1f, 0.05f, { stringResource(R.string.settings_top_p, it) }, vm::setTopP)
                FloatSlider(sp.topK.toFloat(), 0f..100f, 1f, { stringResource(R.string.settings_top_k, it.roundToInt()) }) {
                    vm.setTopK(it.roundToInt())
                }
            }
            FloatSlider(sp.presence(s.thinking), 0f..2f, 0.1f, { stringResource(R.string.settings_presence, it) },
                if (s.thinking) vm::setThinkPresencePenalty else vm::setPresencePenalty)
            Text(stringResource(R.string.settings_presence_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            // what this model's card recommends in this mode (the button below puts them back)
            val rec = family.defaults
            Text(stringResource(R.string.settings_sampling_recommended, model?.title ?: "Qwen3.6",
                rec.temperature(s.thinking), rec.topP(s.thinking), rec.topK(s.thinking), rec.presence(s.thinking)),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp))
            TextButton(onClick = vm::resetSampling) { Text(stringResource(R.string.settings_sampling_reset)) }
            if (s.thinking) {
                Choices(SettingsViewModel.THINK_BUDGETS, s.thinkBudget,
                    { stringResource(R.string.settings_think_budget, tokens(it)) }, vm::setThinkBudget)
                Text(stringResource(R.string.settings_think_budget_desc), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Choices(SettingsViewModel.ANSWER_LIMITS, s.maxAnswer,
                { stringResource(R.string.settings_max_answer, tokens(it)) }, vm::setMaxAnswer)
            // the choices up to the longest context the model was trained for (Gemma 4: 128K, Qwen: 256K)
            val maxCtx = model?.manifest?.maxContext ?: 0
            val ctxChoices = SettingsViewModel.CONTEXT_CHOICES.filter { maxCtx <= 0 || it <= maxCtx }
                .ifEmpty { SettingsViewModel.CONTEXT_CHOICES.take(1) }
            val ctxShown = s.contextFor(model)
            val ctxIndex = ctxChoices.indexOf(ctxShown).takeIf { it >= 0 }
                ?: ctxChoices.indexOfLast { it <= ctxShown }.coerceAtLeast(0)
            FloatSlider(ctxIndex.toFloat(), 0f..(ctxChoices.size - 1).coerceAtLeast(1).toFloat(), 1f,
                { stringResource(R.string.settings_context, ctxChoices[it.roundToInt().coerceIn(0, ctxChoices.size - 1)]) }) {
                vm.setContextLength(ctxChoices[it.roundToInt().coerceIn(0, ctxChoices.size - 1)])
            }
            if (maxCtx > 0) {
                Text(stringResource(R.string.settings_context_model_max, maxCtx), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            // a model of experts lends the context its cache's memory; a dense one has no cache
            Text(stringResource(if (experts) R.string.settings_context_desc else R.string.settings_context_desc_dense),
                style = MaterialTheme.typography.bodySmall,
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

            // ---- the conversations kept on the storage ----
            val cacheBytes by vm.cacheBytes.collectAsStateWithLifecycle()
            LaunchedEffect(Unit) { vm.refreshCache() }
            Text(stringResource(R.string.settings_state_cache), style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 12.dp))
            Choices(CACHE_MB, s.stateCacheMb, { mb ->
                if (mb == 0) stringResource(R.string.settings_state_cache_off) else formatBytes(mb.toLong() shl 20)
            }, vm::setStateCacheMb)
            Text(stringResource(R.string.settings_state_cache_desc, formatBytes(cacheBytes)),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (cacheBytes > 0) {
                TextButton(onClick = vm::clearCache) { Text(stringResource(R.string.settings_state_cache_clear)) }
            }

            // ---- what the model takes: each input its pack has (and its drafting), on or off ----
            val m = model
            if (m != null && (m.hasVision || m.hasVideo || m.hasAudio || m.hasDraft)) {
                Section(stringResource(R.string.settings_section_features))
                if (m.hasVision) {
                    SwitchRow(stringResource(R.string.settings_feature_images),
                        stringResource(R.string.settings_feature_images_desc), m.vision, { vm.setFeature(OFF_IMAGES, it) })
                }
                if (m.hasVideo) {
                    SwitchRow(stringResource(R.string.settings_feature_video),
                        stringResource(R.string.settings_feature_video_desc), m.video, { vm.setFeature(OFF_VIDEO, it) })
                }
                if (m.hasAudio) {
                    SwitchRow(stringResource(R.string.settings_feature_audio),
                        stringResource(R.string.settings_feature_audio_desc), m.audio, { vm.setFeature(OFF_AUDIO, it) })
                }
                if (m.hasDraft) {
                    SwitchRow(stringResource(R.string.settings_feature_draft),
                        stringResource(if (m.manifest.gemma) R.string.settings_feature_draft_desc_drafter
                            else R.string.settings_feature_draft_desc_mtp), m.draft, { vm.setFeature(OFF_DRAFT, it) })
                }
            }

            // ---- images: only for a model that sees them (or before one is picked) ----
            if (model == null || model?.vision == true) {
                Text(stringResource(R.string.settings_image_detail), style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(top = 12.dp))
                Choice(listOf(stringResource(R.string.settings_image_detail_fast),
                    stringResource(R.string.settings_image_detail_fine),
                    stringResource(R.string.settings_image_detail_most)), s.imageDetail, vm::setImageDetail)
                Text(stringResource(R.string.settings_image_detail_desc), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            // ---- performance ----
            Section(stringResource(R.string.settings_section_performance))
            FloatSlider(s.computeThreads.toFloat(), 0f..8f, 1f, {
                val n = it.roundToInt()
                stringResource(R.string.settings_threads,
                    if (n == 0) stringResource(R.string.settings_threads_auto) else n.toString())
            }) { vm.setComputeThreads(it.roundToInt()) }
            if (experts) {
                FloatSlider(s.ioThreads.toFloat(), 0f..6f, 1f, {
                    val n = it.roundToInt()
                    stringResource(R.string.settings_io_threads,
                        if (n == 0) stringResource(R.string.settings_threads_auto) else n.toString())
                }) { vm.setIoThreads(it.roundToInt()) }
            }
            FloatSlider(s.helperThreads.toFloat(), 0f..6f, 1f, {
                val n = it.roundToInt()
                stringResource(R.string.settings_helpers,
                    if (n == 0) stringResource(R.string.settings_helpers_none) else n.toString())
            }) { vm.setHelperThreads(it.roundToInt()) }
            Text(stringResource(R.string.settings_helpers_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (experts) {
                SwitchRow(stringResource(R.string.settings_prefetch), stringResource(R.string.settings_prefetch_desc),
                    s.prefetch, vm::setPrefetch)
            }
            // which units run the model (geode_config.mode): auto, one of them, or two - the prompt to the one faster
            // at prompts, the answer to the other
            Text(stringResource(R.string.settings_runs_on), style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 8.dp))
            // (only the units this phone has; a choice it lost shows as the mode the engine falls back to)
            val chosen = RunModes.nearest(s.runMode, units)
            RunModes.available(units).forEach { m ->
                Row(Modifier.fillMaxWidth().selectable(selected = chosen == m, role = Role.RadioButton,
                    onClick = { vm.setRunMode(m) }).padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = chosen == m, onClick = null)
                    Text(runModeLabel(m), style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(start = 12.dp))
                }
            }
            Text(stringResource(R.string.settings_runs_on_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            // the share splits a prompt's rows on the Adreno 618's path only (its status "... of their rows",
            // model.cpp); the Adreno 7xx+ int8 path takes whole products (a dense model gets that path only): there
            // the slider would change nothing
            if (RunModes.usesGpu(s.runMode) && gpuStatus?.endsWith("of their rows") == true) {
                FloatSlider(s.gpuShare, 0.1f..0.6f, 0.05f,
                    { stringResource(R.string.settings_gpu_share, (it * 100).roundToInt()) }, vm::setGpuShare)
            }
            gpuStatus?.let {
                Text(stringResource(R.string.settings_gpu_status, it), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (experts) {
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
            }
            Text(stringResource(R.string.settings_engine_reopen), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
            Spacer(Modifier.height(24.dp))
        }
    }
}
