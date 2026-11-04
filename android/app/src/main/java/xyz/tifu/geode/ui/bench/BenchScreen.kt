package xyz.tifu.geode.ui.bench

import androidx.compose.material3.IconButton
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.semantics.Role
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.material3.Checkbox
import androidx.compose.foundation.selection.toggleable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import xyz.tifu.geode.R
import xyz.tifu.geode.bench.DeviceReport
import xyz.tifu.geode.bench.Estimate
import xyz.tifu.geode.util.formatBytes

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchScreen(vm: BenchViewModel, openDrawer: () -> Unit) {
    val st by vm.state.collectAsStateWithLifecycle()
    // the screen kept on while the tests run (a remote test lab's session, a phone left on the table)
    val view = LocalView.current
    val busy = st.running != null
    DisposableEffect(busy) {
        view.keepScreenOn = busy
        onDispose { view.keepScreenOn = false }
    }
    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.bench_title)) }, navigationIcon = {
            IconButton(onClick = openDrawer) {
                Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
            }
        })
        if (st.running != null) LinearProgressIndicator(Modifier.fillMaxWidth())
        LazyColumn(
            Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                Column {
                    // one button: the tests, and with the box ticked (the default) the device report sent as they go
                    var send by rememberSaveable { mutableStateOf(true) }
                    Button(onClick = { if (send) vm.checkAndSend() else vm.runAll() }, enabled = st.running == null,
                        modifier = Modifier.fillMaxWidth()) {
                        Text(stringResource(R.string.bench_run_all))
                    }
                    Row(verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth().toggleable(value = send, role = Role.Checkbox,
                            onValueChange = { send = it })) {
                        Checkbox(checked = send, onCheckedChange = null)
                        Text(stringResource(R.string.bench_send_label), style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(start = 8.dp))
                    }
                    // what goes, said where the user decides (the privacy page has the rest)
                    if (send) Dim(stringResource(R.string.bench_report_desc))
                    Text(
                        st.running?.let { stringResource(R.string.bench_running, testName(it)) }
                            ?: stringResource(R.string.bench_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                    st.error?.let {
                        Text(stringResource(R.string.bench_failed, testName(it)), color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            if (st.report != null) item { ReportCard(st) }
            item { EstimateCard(st) }
            item { Section(R.string.bench_device, "system", st, vm) { SystemBody(it) } }
            item { Section(R.string.bench_memory, "memory", st, vm) { MemoryBody(it) } }
            item { Section(R.string.bench_flash, "flash", st, vm) { FlashBody(it) } }
            item { Section(R.string.bench_cpu, "cpu", st, vm) { CpuBody(it) } }
            item { Section(R.string.bench_gpu, "gpu", st, vm) { GpuBody(it) } }
            item { Section(R.string.bench_opencl, "opencl", st, vm) { OpenclBody(it) } }
            item { Section(R.string.bench_gpucheck, "gpucheck", st, vm) { GpuCheckBody(it) } }
            item { Section(R.string.bench_npucheck, "npucheck", st, vm) { NpuCheckBody(it) } }
        }
    }
}

@Composable
private fun testName(t: String): String = when (t) {
    "system" -> stringResource(R.string.bench_device)
    "memory" -> stringResource(R.string.bench_memory)
    "flash" -> stringResource(R.string.bench_flash)
    "cpu" -> stringResource(R.string.bench_cpu)
    "opencl" -> stringResource(R.string.bench_opencl)
    "gpucheck" -> stringResource(R.string.bench_gpucheck)
    "npucheck" -> stringResource(R.string.bench_npucheck)
    else -> stringResource(R.string.bench_gpu)
}

/** The device report of the last run: its number big enough to read off a remote screen, what went, what did not */
@Composable
private fun ReportCard(st: BenchUiState) {
    Card {
        Column(Modifier.padding(16.dp)) {
            Text(stringResource(R.string.bench_report_title), style = MaterialTheme.typography.titleMedium)
            val r = st.report ?: return@Column
            Spacer(Modifier.height(8.dp))
            Text(stringResource(R.string.bench_report_id, DeviceReport.display(r.id)),
                style = MaterialTheme.typography.headlineSmall, fontFamily = FontFamily.Monospace)
            val status = when {
                r.local -> stringResource(R.string.bench_report_saved, r.file)
                r.unsent > 0 -> stringResource(R.string.bench_report_unsent, r.unsent, r.sent + r.unsent)
                r.finished -> stringResource(R.string.bench_report_done, r.sent)
                else -> stringResource(R.string.bench_report_sending, r.sent, r.total)
            }
            Text(status, style = MaterialTheme.typography.bodyMedium,
                color = if (r.unsent > 0) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary)
            // where it is kept, when it did not go: sent at the next start, or read with adb
            if (!r.local && r.unsent > 0 && r.file.isNotEmpty()) Dim(r.file)
            if (r.json.isNotEmpty()) {
                val clip = LocalClipboardManager.current
                TextButton(onClick = { clip.setText(AnnotatedString(r.json)) }) {
                    Text(stringResource(R.string.bench_report_copy))
                }
            }
        }
    }
}

@Composable
private fun Section(title: Int, test: String, st: BenchUiState, vm: BenchViewModel, body: @Composable (JSONObject) -> Unit) {
    Card {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(title), style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                TextButton(onClick = { vm.run(listOf(test)) }, enabled = st.running == null) {
                    Text(stringResource(R.string.bench_run))
                }
            }
            val obj = st.results.optJSONObject(test)
            if (obj == null) Dim(stringResource(R.string.bench_not_run)) else body(obj)
        }
    }
}

@Composable
private fun Dim(text: String) {
    Text(text, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun Line(text: String, mono: Boolean = false, bold: Boolean = false) {
    Text(
        text,
        style = MaterialTheme.typography.bodyMedium,
        fontFamily = if (mono) FontFamily.Monospace else null,
        fontWeight = if (bold) FontWeight.SemiBold else null,
    )
}

/** A number as the screen shows it: a dash for one the test did not get (the engine writes those as null) */
private fun f(v: Double, digits: Int = 1) =
    if (v.isFinite()) String.format(Locale.getDefault(), "%.${digits}f", v) else "\u2014"

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Flags(flags: List<Pair<String, Boolean>>) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        flags.forEach { (name, on) ->
            SuggestionChip(
                onClick = {},
                label = { Text(name) },
                icon = {
                    Icon(if (on) Icons.Outlined.Check else Icons.Outlined.Close, null,
                        tint = if (on) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
                },
            )
        }
    }
}

@Composable
private fun SystemBody(o: JSONObject) {
    Line(listOf(o.optString("device"), o.optString("soc"), "Android " + o.optString("android"))
        .filter { it.isNotBlank() }.joinToString(" · "))
    val cl = o.optJSONArray("clusters") ?: JSONArray()
    for (i in 0 until cl.length()) {
        val c = cl.optJSONObject(i) ?: continue
        Line(stringResource(R.string.bench_cores, c.optInt("count"), c.optString("core"), c.optInt("max_mhz") / 1000.0))
    }
    if (o.has("ram_bytes")) Line(stringResource(R.string.bench_ram_total, formatBytes(o.optDouble("ram_bytes").toLong())))
    Spacer(Modifier.height(6.dp))
    Dim(stringResource(R.string.bench_cpu_features))
    val fe = o.optJSONObject("features") ?: JSONObject()
    Flags(listOf("FP16" to "fp16", "INT8 dot" to "dotprod", "INT8 matmul" to "i8mm", "BF16" to "bf16", "SVE" to "sve",
        "SVE2" to "sve2", "SME" to "sme").map { (n, k) -> n to fe.optBoolean(k) })
}

@Composable
private fun MemoryBody(o: JSONObject) {
    Dim(stringResource(R.string.bench_mem_cols))
    val cs = o.optJSONArray("configs") ?: JSONArray()
    for (i in 0 until cs.length()) {
        val c = cs.optJSONObject(i) ?: continue
        val name = when (c.optString("name")) {
            "one_big_core" -> stringResource(R.string.bench_cfg_one_big_core)
            "big_cores" -> stringResource(R.string.bench_cfg_big_cores)
            "little_cores" -> stringResource(R.string.bench_cfg_little_cores)
            "one_core" -> stringResource(R.string.bench_cfg_one_core)
            else -> stringResource(R.string.bench_cfg_all_cores)
        }
        Row(Modifier.fillMaxWidth()) {
            Text(name, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            Line("${f(c.optDouble("read_gbs"))} · ${f(c.optDouble("write_gbs"))} · ${f(c.optDouble("copy_gbs"))}", mono = true)
        }
    }
    if (o.has("latency_ns")) Line(stringResource(R.string.bench_latency, o.optDouble("latency_ns")))
}

@Composable
private fun FlashBody(o: JSONObject) {
    if (o.has("error")) {
        Text(o.optString("error"), color = MaterialTheme.colorScheme.error)
        return
    }
    Dim(
        (if (o.optString("source").contains("model")) stringResource(R.string.bench_flash_source_model)
        else stringResource(R.string.bench_flash_source_temp)) + " · " +
            stringResource(R.string.bench_flash_direct, if (o.optBoolean("direct")) "✓" else "✗")
    )
    Spacer(Modifier.height(4.dp))
    Dim(stringResource(R.string.bench_flash_random))
    val rs = o.optJSONArray("random") ?: JSONArray()
    val byBlock = (0 until rs.length()).mapNotNull { rs.optJSONObject(it) }.groupBy { it.optDouble("block").toLong() }
    byBlock.forEach { (block, rows) ->
        val expert = rows.first().optBoolean("expert_block")
        Row(Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.bench_flash_block, formatBytes(block)) +
                (if (expert) stringResource(R.string.bench_flash_expert) else ""),
                Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            Line(rows.joinToString("  ") { "${it.optInt("threads")}×${f(it.optDouble("mbs"), 0)}" }, mono = true)
        }
        val res = LocalContext.current.resources
        Dim(rows.joinToString("  ") {
            res.getString(R.string.bench_threads, it.optInt("threads"), f(it.optDouble("latency_ms")))
        })
    }
    if (o.has("sequential_mbs")) Line(stringResource(R.string.bench_flash_seq, o.optDouble("sequential_mbs")))
    if (o.has("write_mbs")) Line(stringResource(R.string.bench_flash_write, o.optDouble("write_mbs")))
}

@Composable
private fun CpuBody(o: JSONObject) {
    val ps = o.optJSONArray("peaks") ?: JSONArray()
    if (ps.length() > 0) {
        // a table: a row per data type, a column per core type
        Dim(stringResource(R.string.bench_cpu_peaks))
        val peaks = (0 until ps.length()).mapNotNull { ps.optJSONObject(it) }
        Row(Modifier.fillMaxWidth()) {
            Spacer(Modifier.weight(1f))
            peaks.forEach { p ->
                val core = p.optString("core")
                Text(Regex("Cortex-(\\w+)").find(core)?.groupValues?.get(1) ?: core, Modifier.weight(1f),
                    style = MaterialTheme.typography.labelLarge, textAlign = TextAlign.End)
            }
        }
        listOf("FP32" to "fp32_gflops", "FP16" to "fp16_gflops", "INT8 dot" to "int8_dot_gops",
            "INT8 matmul" to "int8_mmla_gops", "BF16" to "bf16_dot_gflops").forEach { (name, key) ->
            Row(Modifier.fillMaxWidth()) {
                Text(name, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                peaks.forEach { p ->
                    Text(if (p.has(key)) f(p.optDouble(key), 0) else "—", Modifier.weight(1f),
                        fontFamily = FontFamily.Monospace, textAlign = TextAlign.End)
                }
            }
        }
        Spacer(Modifier.height(6.dp))
    }
    Dim(stringResource(R.string.bench_cpu_kernels))
    Dim(stringResource(R.string.bench_kernel_cols, o.optInt("big_cores", 1)))
    val ks = o.optJSONArray("kernels") ?: JSONArray()
    for (i in 0 until ks.length()) {
        val k = ks.optJSONObject(i) ?: continue
        Row(Modifier.fillMaxWidth()) {
            Text("${k.optString("format")} · ${f(k.optDouble("bits_per_weight"))} bit", Modifier.weight(1f),
                style = MaterialTheme.typography.bodyMedium)
            Line("${f(k.optDouble("one_core_gw"))} · ${f(k.optDouble("big_cores_gw"))} · " +
                f(k.optDouble("four_tokens_one_core_gw")), mono = true)
        }
    }
}

@Composable
private fun GpuBody(o: JSONObject) {
    if (!o.optBoolean("available")) {
        Dim(stringResource(R.string.bench_gpu_none, o.optString("reason")))
        return
    }
    Line("${o.optString("name")} · Vulkan ${o.optString("vulkan")}")
    Spacer(Modifier.height(4.dp))
    Dim(stringResource(R.string.bench_gpu_types))
    val t = o.optJSONObject("types") ?: JSONObject()
    Flags(listOf("FP32" to "fp32", "FP16" to "fp16_arithmetic", "INT16" to "int16_arithmetic", "INT8" to "int8_arithmetic",
        "FP16 buffers" to "fp16_storage", "INT8 buffers" to "int8_storage", "INT8 dot" to "int8_dot_product",
        "FP64" to "fp64").map { (n, k) -> n to t.optBoolean(k) })
    Line(stringResource(R.string.bench_gpu_numbers, o.optDouble("fp32_gflops", 0.0),
        if (o.has("fp16_gflops")) f(o.optDouble("fp16_gflops"), 0) else "—", o.optDouble("int32_gops", 0.0),
        o.optDouble("read_gbs", 0.0)))
    Dim(stringResource(R.string.bench_gpu_note))
}

@Composable
private fun EstimateCard(st: BenchUiState) {
    Card {
        Column(Modifier.padding(16.dp)) {
            Text(stringResource(R.string.bench_estimate), style = MaterialTheme.typography.titleMedium)
            val e = st.estimate
            val m = st.model
            when {
                m == null -> Dim(stringResource(R.string.bench_estimate_none_model))
                e == null -> Dim(stringResource(R.string.bench_estimate_none_tests))
                else -> {
                    Line(m.name, bold = true)
                    if (st.modelBundled) Dim(stringResource(R.string.bench_estimate_bundled))
                    Dim(stringResource(R.string.bench_est_cache, e.cacheExperts.toString(), e.cacheShare * 100,
                        e.hitRate * 100, stringResource(if (e.hitMeasured) R.string.bench_est_measured else R.string.bench_est_estimated)))
                    Dim(stringResource(R.string.bench_est_bytes, e.denseMbPerToken, e.expertMbPerToken, e.flashMbPerToken))
                    Spacer(Modifier.height(6.dp))
                    Line(stringResource(R.string.bench_est_decode, e.computeMs, e.ramMs, e.flashMs))
                    Text(stringResource(R.string.bench_est_decode_speed, e.decodeOverlapped, e.decodeSerial),
                        style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                    Dim(stringResource(R.string.bench_est_limit, limit(e)))
                    Spacer(Modifier.height(6.dp))
                    Line(stringResource(R.string.bench_est_prefill, e.prefillComputeMs, e.prefillFlashMs, e.prefill))
                    if (st.whatIf.isNotEmpty()) {
                        Spacer(Modifier.height(8.dp))
                        Dim(stringResource(R.string.bench_whatif))
                        st.whatIf.forEach { w ->
                            Row(Modifier.fillMaxWidth()) {
                                Text("${w.format} · ${f(w.bits)} bit", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                                Line("${f(w.estimate.decodeSerial)}–${f(w.estimate.decodeOverlapped)}  " +
                                    stringResource(R.string.bench_whatif_prompt, w.estimate.prefill), mono = true)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun limit(e: Estimate): String {
    val worst = maxOf(e.computeMs, e.ramMs, e.flashMs)
    return stringResource(when (worst) {
        e.flashMs -> R.string.bench_limit_flash
        e.ramMs -> R.string.bench_limit_ram
        else -> R.string.bench_limit_compute
    })
}

@Composable
private fun OpenclBody(o: JSONObject) {
    if (!o.optBoolean("available")) {
        Dim(stringResource(R.string.bench_opencl_none, o.optString("reason")))
        return
    }
    Line(o.optString("device"), bold = true)
    Dim(stringResource(R.string.bench_opencl_driver, o.optString("driver"), o.optString("opencl_c")))
    Line(stringResource(R.string.bench_opencl_limits, o.optInt("compute_units"), o.optInt("max_clock_mhz"),
        formatBytes(o.optDouble("global_mem").toLong()), formatBytes(o.optDouble("max_alloc").toLong()),
        formatBytes(o.optDouble("local_mem").toLong())))
    val ext = o.optString("extensions").split(' ').toSet()
    Spacer(Modifier.height(4.dp))
    Flags(listOf("FP16" to o.optBoolean("fp16"), "images" to o.optBoolean("images"),
        "INT8 dot" to ("cl_qcom_dot_product8" in ext), "ION / DMA" to ("cl_qcom_ion_host_ptr" in ext),
        "subgroups" to ("cl_khr_subgroups" in ext || "cl_qcom_reqd_sub_group_size" in ext),
        "recordable queues" to ("cl_qcom_recordable_queues" in ext)))
    if (o.has("error")) {
        Text(o.optString("error").take(600), color = MaterialTheme.colorScheme.error, fontFamily = FontFamily.Monospace,
            style = MaterialTheme.typography.bodySmall)
        return
    }
    val rd = o.optJSONObject("read_gbs") ?: JSONObject()
    Line(stringResource(R.string.bench_opencl_numbers, o.optDouble("fp32_gflops", 0.0),
        if (o.has("fp16_gflops")) f(o.optDouble("fp16_gflops"), 0) else "\u2014",
        rd.optDouble("gpu_buffer", 0.0), rd.optDouble("cpu_memory", 0.0), o.optDouble("launch_wait_us", 0.0)))
}

/** Geode's NPU code on this phone's Hexagon: whether the engine can use it (and why not), each step checked and timed */
@Composable
private fun NpuCheckBody(o: JSONObject) {
    val on = o.optString("path") == "npu"
    Line(stringResource(if (on) R.string.bench_npucheck_path_npu else R.string.bench_npucheck_path_off), bold = true)
    o.optString("reason").takeIf { it.isNotEmpty() }?.let { Dim(it.take(300)) }
    o.optString("device").takeIf { it.isNotEmpty() }?.let { Dim(it) }
    CheckList(o)
}

/** Each check of a GPU / NPU check: its name, whether it passed, its numbers, its error */
@Composable
private fun CheckList(o: JSONObject) {
    val cs = o.optJSONArray("checks") ?: JSONArray()
    for (i in 0 until cs.length()) {
        val c = cs.optJSONObject(i) ?: continue
        val ok = c.optBoolean("ok")
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Icon(if (ok) Icons.Outlined.Check else Icons.Outlined.Close, null,
                tint = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
            Text(c.optString("name"), Modifier.weight(1f).padding(start = 6.dp), style = MaterialTheme.typography.bodyMedium)
            val nums = listOfNotNull(
                c.optDouble("tmacs").takeIf { !it.isNaN() }?.let { "${f(it, 2)} TMAC/s" },
                c.optDouble("gbs").takeIf { !it.isNaN() }?.let { "${f(it)} GB/s" },
                c.optDouble("us").takeIf { !it.isNaN() }?.let { "${f(it)} \u00b5s" },
                c.optDouble("ms").takeIf { !it.isNaN() }?.let { "${f(it, 2)} ms" })
            if (nums.isNotEmpty()) Line(nums.joinToString(" \u00b7 "), mono = true)
        }
        if (c.has("err_gpu")) Dim(stringResource(R.string.bench_gpucheck_err, f(c.optDouble("err_gpu") * 100, 2),
            f(c.optDouble("err_cpu_int8") * 100, 2)))
        c.optString("error").takeIf { it.isNotEmpty() }?.let {
            Text(it.take(600), color = MaterialTheme.colorScheme.error, fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodySmall)
        }
    }
}

/** Geode's own GPU code on this GPU: the path the engine takes, each kernel checked against the CPU and timed */
@Composable
private fun GpuCheckBody(o: JSONObject) {
    val path = o.optString("path")
    Line(stringResource(when (path) {
        "gpu8" -> R.string.bench_gpucheck_path_gpu8
        "gpu618" -> R.string.bench_gpucheck_path_618
        else -> R.string.bench_gpucheck_path_cpu
    }), bold = true)
    o.optString("reason").takeIf { it.isNotEmpty() && path != "gpu8" }?.let { Dim(it.take(300)) }
    o.optString("device").takeIf { it.isNotEmpty() }?.let { Dim(it) }
    val cs = o.optJSONArray("checks") ?: JSONArray()
    for (i in 0 until cs.length()) {
        val c = cs.optJSONObject(i) ?: continue
        val ok = c.optBoolean("ok")
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Icon(if (ok) Icons.Outlined.Check else Icons.Outlined.Close, null,
                tint = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
            Text(c.optString("name"), Modifier.weight(1f).padding(start = 6.dp), style = MaterialTheme.typography.bodyMedium)
            val nums = listOfNotNull(
                c.optDouble("gbs").takeIf { !it.isNaN() }?.let { "${f(it)} GB/s" },
                c.optDouble("ms").takeIf { !it.isNaN() }?.let { "${f(it, 2)} ms" })
            if (nums.isNotEmpty()) Line(nums.joinToString(" · "), mono = true)
        }
        if (c.has("err_gpu")) Dim(stringResource(R.string.bench_gpucheck_err, f(c.optDouble("err_gpu") * 100, 2),
            f(c.optDouble("err_cpu_int8") * 100, 2)))
        c.optString("error").takeIf { it.isNotEmpty() }?.let {
            Text(it.take(600), color = MaterialTheme.colorScheme.error, fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodySmall)
        }
    }
}
