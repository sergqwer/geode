package xyz.tifu.geode.bench

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject
import java.io.File

/** The last result of every benchmark (files/bench.json), kept so the speed estimate works without rerunning them. */
class BenchStore(context: Context) {
    private val file = File(context.filesDir, "bench.json")
    private val _results = MutableStateFlow(load())
    val results: StateFlow<JSONObject> = _results.asStateFlow()

    private fun load(): JSONObject =
        runCatching { if (file.isFile) JSONObject(file.readText()) else JSONObject() }.getOrDefault(JSONObject())

    fun put(what: String, result: JSONObject) {
        val next = JSONObject(_results.value.toString())
        next.put(what, result)
        next.put("at_$what", System.currentTimeMillis())
        runCatching { file.writeText(next.toString()) }
        _results.value = next
    }

    companion object {
        val TESTS = listOf("system", "memory", "flash", "cpu", "gpu", "opencl", "gpucheck", "npucheck")
    }
}
