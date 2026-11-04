package xyz.tifu.geode.ui

import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import xyz.tifu.geode.R
import xyz.tifu.geode.data.RunModes
import xyz.tifu.geode.engine.EngineRun
import xyz.tifu.geode.util.formatBytes

/**
 * How the open model runs, in plain words, from what the engine reported when it opened (EngineRun): which of the
 * NPU, the GPU and the CPU compute it ("NPU only", "NPU + GPU (Adreno 830)", "CPU only"...), why the NPU is off if it
 * is, the speculative decoding, the expert cache. E.g. "NPU + GPU (Adreno 830) · MTP, up to 2 tokens at once · expert
 * cache 7.4 GB".
 */
@Composable
fun runModeText(r: EngineRun): String {
    val gpu = r.gpu.orEmpty()
    val npu = r.npu.orEmpty()
    val gpuOn = gpu.startsWith("on")
    val npuOn = npu.startsWith("on")
    // the answers on the GPU unless it hands them to the CPU ("...; generating on the CPU (why)"); on the NPU when its
    // decoder runs the whole step ("...; decoder on: ..."), and the prompts there too when it says "prompts"
    val gpuAnswers = gpuOn && "generating on the CPU" !in gpu
    val decoder = npu.substringAfter("; decoder ", "").substringBefore(';')
    val npuAnswers = npuOn && decoder.startsWith("on")
    val npuPrompts = npuAnswers && "prompts" in decoder
    // the CPU computes the answers when neither does, and the prompts' other parts when the NPU only multiplies
    val cpu = !(gpuAnswers || npuAnswers) || (npuAnswers && !npuPrompts)

    val units = mutableListOf<String>()
    if (npuOn) units += "NPU"
    if (gpuOn) {
        // "on: QUALCOMM Adreno(TM) 830, int8, 255 matrices, ..." -> the chip's name
        val device = gpu.removePrefix("on:").trim().substringBefore(',').substringBefore(';').trim()
            .replace("QUALCOMM ", "").replace("(TM)", "").replace(Regex("\\s+"), " ").trim()
            .takeUnless { it.isEmpty() || it.first().isDigit() } ?: ""
        units += stringResource(R.string.models_mode_gpu, device.ifEmpty { "Adreno" })
    }
    if (cpu) units += "CPU"
    val parts = mutableListOf(
        if (units.size == 1) stringResource(R.string.models_mode_only, units[0]) else units.joinToString(" + "))
    // (off by the mode the user chose: nothing to explain)
    if (!npuOn && npu.isNotEmpty() && "the mode:" !in npu)
        parts += stringResource(R.string.models_mode_npu_off,
            npu.removePrefix("off").removePrefix("failed").trim().removePrefix("(").substringBeforeLast(')').trim()
                .substringBefore(';').ifEmpty { npu })
    when {
        r.spec.startsWith("gpu:") -> parts += stringResource(R.string.models_mode_mtp_gpu,
            r.spec.removePrefix("gpu:").toIntOrNull() ?: 2)
        r.spec.startsWith("cpu") -> parts += stringResource(R.string.models_mode_mtp_cpu)
    }
    // (a dense model has no experts: its few bytes of cache are not worth a word)
    if (r.cacheBytes >= 1L shl 20) parts += stringResource(R.string.models_mode_cache, formatBytes(r.cacheBytes))
    return parts.joinToString(" · ")
}

/** A run mode's name in the settings' choice (RunModes): "CPU only", "NPU + GPU"... - of two, the prompts' unit first */
@Composable
fun runModeLabel(mode: Int): String = when (mode) {
    RunModes.GPU -> stringResource(R.string.models_mode_only, "GPU")
    RunModes.NPU -> stringResource(R.string.models_mode_only, "NPU")
    RunModes.NPU_CPU -> "NPU + CPU"
    RunModes.NPU_GPU -> "NPU + GPU"
    RunModes.CPU_GPU -> "GPU + CPU"
    else -> stringResource(R.string.models_mode_only, "CPU")
}
