package xyz.tifu.geode.ui.bench

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.bench.BenchStore
import xyz.tifu.geode.bench.DeviceReport
import xyz.tifu.geode.bench.DeviceSpeed
import xyz.tifu.geode.bench.Estimate
import xyz.tifu.geode.bench.ModelShape
import xyz.tifu.geode.bench.SpeedModel
import xyz.tifu.geode.engine.CpuSupport
import xyz.tifu.geode.models.ModelStore
import java.io.File

data class WhatIf(val format: String, val bits: Double, val estimate: Estimate)

/**
 * A device report on its way: its number, the parts sent and how many, the ones that did not go, the file it is kept
 * in (every part over the last); local: started from adb, kept in the file only
 */
data class ReportUi(val id: String, val total: Int, val sent: Int = 0, val unsent: Int = 0, val finished: Boolean = false,
                    val json: String = "", val file: String = "", val local: Boolean = false)

data class BenchUiState(
    val results: JSONObject = JSONObject(),
    /** The test running now, or null. */
    val running: String? = null,
    val error: String? = null,
    val model: ModelShape? = null,
    /** The estimate is for the Q4 model from Hugging Face, none being picked */
    val modelBundled: Boolean = false,
    val estimate: Estimate? = null,
    val whatIf: List<WhatIf> = emptyList(),
    val budgetMb: Int = 0,
    val context: Int = 0,
    val report: ReportUi? = null,
)

class BenchViewModel(private val app: GeodeApp) : ViewModel() {
    private val store = ModelStore(app)
    private val _state = MutableStateFlow(BenchUiState())
    val state: StateFlow<BenchUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            combine(app.bench.results, app.settings.flow) { r, s -> r to s }.collect { (r, s) ->
                val (shape, picked, est) = estimate(r, s)
                val dev = DeviceSpeed.from(r)
                val budget = s.effectiveBudgetMb.toLong() shl 20
                val whatIf = if (shape != null && dev != null) listOf("q2", "q3", "nvfp4", "q8").map { f ->
                    WhatIf(f, ModelShape.bitsPerWeight(f),
                        SpeedModel.estimate(SpeedModel.withFormat(shape, f), dev, budget, 0.0))
                } else emptyList()
                _state.update {
                    it.copy(results = r, model = shape, modelBundled = !picked, estimate = est,
                        whatIf = whatIf, budgetMb = s.effectiveBudgetMb, context = s.contextLength)
                }
            }
        }
    }

    /** The model the estimate is for (the picked one, else the Q4 pack's shape), whether it is picked, the estimate */
    private suspend fun estimate(r: JSONObject, s: xyz.tifu.geode.data.AppSettings): Triple<ModelShape?, Boolean, Estimate?> {
        val picked = withContext(Dispatchers.IO) {
            s.selectedModel.takeIf { it.isNotEmpty() }?.let { store.find(it) }?.let { ModelShape.read(it.dir) }
        }
        val shape = picked ?: withContext(Dispatchers.IO) { ModelShape.bundled(app) }
        val dev = DeviceSpeed.from(r)
        val hit = if (picked != null) s.lastHitRate.toDouble() else 0.0
        val est = if (shape != null && dev != null)
            SpeedModel.estimate(shape, dev, s.effectiveBudgetMb.toLong() shl 20, hit) else null
        return Triple(shape, picked != null, est)
    }

    fun runAll() = run(BenchStore.TESTS)

    fun run(tests: List<String>) {
        if (_state.value.running != null) return
        viewModelScope.launch {
            _state.update { it.copy(error = null) }
            for (t in tests) runOne(t)?.let { _state.update { s -> s.copy(error = t) } }
            _state.update { it.copy(running = null) }
        }
    }

    /**
     * Check and send: every test, the most telling first (a remote test lab's session can end any minute), and the
     * report sent at the start and after each test under one number, and kept in a file each time (a part that did
     * not go is sent at the next start). send = false (the check adb started, DeviceReport.ACTION_CHECK): the file
     * only.
     */
    fun checkAndSend(send: Boolean = true) {
        if (_state.value.running != null) return
        val id = DeviceReport.newId()
        val progress = DeviceReport.Progress(CHECK_ORDER)
        val started = System.currentTimeMillis()
        _state.update { it.copy(error = null, report = ReportUi(id, total = CHECK_ORDER.size + 1, local = !send)) }
        viewModelScope.launch {
            suspend fun push() {
                val settings = app.settings.flow.first()
                // the estimate from the results as they are now (the screen's state may not have caught up yet)
                val results = app.bench.results.value
                val (shape, picked, est) = estimate(results, settings)
                val report = DeviceReport.build(app, id, started, progress, results,
                    est?.let { estimateJson(shape, !picked, it) }, JSONObject().apply {
                        put("model", settings.selectedModel)
                        put("budget_mb", settings.effectiveBudgetMb)
                        put("context", settings.contextLength)
                        put("threads", settings.computeThreads)
                        put("io_threads", settings.ioThreads)
                        put("gpu", settings.gpu)
                        put("npu", settings.npu)
                    })
                val (ok, file) = withContext(Dispatchers.IO) {
                    val ok = send && DeviceReport.send(app, report)
                    ok to runCatching { DeviceReport.save(app, report, pending = send && !ok).path }.getOrDefault("")
                }
                android.util.Log.i("GeodeCheck", "report $id: done ${progress.done}, failed ${progress.failed.keys}, " +
                    "finished ${progress.finished}, " + (if (send) "sent $ok, " else "") + "file $file")
                _state.update {
                    val r = it.report ?: return@update it
                    it.copy(report = r.copy(sent = r.sent + if (ok) 1 else 0, unsent = r.unsent + if (ok || !send) 0 else 1,
                        finished = progress.finished, json = report.toString(1), file = file))
                }
            }
            push()
            for (t in CHECK_ORDER) {
                progress.running = t
                val t0 = System.currentTimeMillis()
                val err = runOne(t)
                progress.ms[t] = System.currentTimeMillis() - t0
                if (app.bench.results.value.optLong("at_$t") >= t0) progress.stored += t
                if (err == null) progress.done += t else {
                    progress.failed[t] = err
                    _state.update { it.copy(error = t) }
                }
                progress.running = null
                progress.finished = t == CHECK_ORDER.last()
                push()
            }
            _state.update { it.copy(running = null) }
        }
    }

    /** One test: its result stored; null, or why it failed (the engine stopping during the GPU check: at which step) */
    private suspend fun runOne(t: String): String? {
        _state.update { it.copy(running = t) }
        // the tests that run the engine's own kernels: on a CPU without them they would stop the engine (CpuSupport)
        if (t in KERNEL_TESTS && !CpuSupport.engine) return app.getString(xyz.tifu.geode.R.string.cpu_unsupported)
        val s = app.settings.flow.first()
        val arg = when (t) {
            // flash: the selected model's expert files (read only); otherwise a temporary file in the app's storage
            "flash" -> s.selectedModel.takeIf { it.isNotEmpty() }?.let { store.find(it) }?.dir?.path ?: app.filesDir.path
            // the GPU check writes each step there before it, read back if the engine dies on one
            "gpucheck", "npucheck" -> stepFile(t).also { withContext(Dispatchers.IO) { it.delete() } }.path
            else -> ""
        }
        // the benchmarks close the model: after an answer the chat or the local server is writing
        val json = app.gate.use(this@BenchViewModel) { app.engine.bench(t, arg) }
        val obj = runCatching { JSONObject(json) }.getOrNull()
        if (obj == null) {
            if (json.isNotEmpty()) return "no result: " + json.take(200)
            val step = if (t == "gpucheck" || t == "npucheck") withContext(Dispatchers.IO) {
                runCatching { stepFile(t).readText().trim() }.getOrNull()
            } else null
            return "the engine stopped" + (step?.let { " during: $it" } ?: "")
        }
        app.bench.put(t, obj)
        return obj.optString("error").takeIf { it.isNotEmpty() && t != "gpucheck" && t != "npucheck" }
    }

    /** The GPU / NPU check writes each step's name here before it: what was running when a driver or the DSP took the
     *  engine down */
    private fun stepFile(t: String) = File(app.cacheDir, "$t.step")

    private fun estimateJson(model: ModelShape?, bundled: Boolean, e: Estimate) = JSONObject().apply {
        put("model", model?.name ?: "")
        put("bundled_model", bundled)
        put("decode_tps_overlapped", e.decodeOverlapped)
        put("decode_tps_serial", e.decodeSerial)
        put("prefill_tps", e.prefill)
        put("compute_ms", e.computeMs)
        put("ram_ms", e.ramMs)
        put("flash_ms", e.flashMs)
        put("hit_rate", e.hitRate)
        put("cache_experts", e.cacheExperts)
    }

    companion object {
        /** Check and send's order: what the phone is, whether Geode's GPU code runs on it, then the rest, flash last */
        val CHECK_ORDER = listOf("system", "gpucheck", "npucheck", "opencl", "cpu", "memory", "gpu", "flash")
        /** The tests built with the engine's ARMv8.2 kernels (the rest only ask the system, the GPU or the DSP) */
        private val KERNEL_TESTS = setOf("cpu", "memory", "flash")
    }
}
