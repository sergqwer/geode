package xyz.tifu.geode.data

import org.json.JSONObject

/**
 * geode_config.mode (geode.h GEODE_MODE_*): which of the phone's units run the model. Of two, the first reads the
 * prompts and the second writes the answer. The settings list only the modes of the units this phone has (the
 * engine's "units" probe); a mode the phone or the model still cannot run falls back to the nearest it can, and the
 * run-mode label says what really ran.
 */
object RunModes {
    const val AUTO = 0   // (the engine's own pick: the CLI's and the API's default, not offered in the app)
    const val CPU = 1
    const val GPU = 2
    const val NPU_CPU = 3
    const val NPU_GPU = 4
    const val CPU_GPU = 5   // the prompt on the GPU, the answer on the CPU: "GPU + CPU" in the app
    const val NPU = 6

    /** The default: the fastest where the phone has both (the RedMagic's Snapdragon 8 Elite: E2B 43 tokens/s, against
     *  36 with the answer on the CPU). */
    const val DEFAULT = NPU_GPU

    /** The choices in the order the settings list them. */
    val ALL = listOf(CPU, GPU, NPU, NPU_GPU, NPU_CPU, CPU_GPU)

    fun usesNpu(mode: Int) = mode == NPU || mode == NPU_GPU || mode == NPU_CPU
    fun usesGpu(mode: Int) = mode == GPU || mode == NPU_GPU || mode == CPU_GPU

    /** The units the engine found on the phone (geode_bench "units"). */
    data class Units(val gpu: Boolean, val npu: Boolean)

    /** The probe's JSON read; null when there is none yet (every mode offered meanwhile). */
    fun parseUnits(json: String): Units? = if (json.isEmpty()) null else runCatching {
        val o = JSONObject(json)
        Units(gpu = o.optBoolean("gpu", true), npu = o.optBoolean("npu", true))
    }.getOrNull()

    /** The modes the phone's units can run. */
    fun available(units: Units?): List<Int> =
        ALL.filter { units == null || ((!usesNpu(it) || units.npu) && (!usesGpu(it) || units.gpu)) }

    /** The nearest mode the phone can run: a missing unit's part to the one left (as the engine falls back). */
    fun nearest(mode: Int, units: Units?): Int {
        if (units == null) return mode
        val npu = usesNpu(mode) && units.npu
        val gpu = usesGpu(mode) && units.gpu
        return when {
            mode == NPU -> if (npu) NPU else CPU
            npu && gpu -> NPU_GPU
            npu -> NPU_CPU
            gpu -> if (mode == CPU_GPU) CPU_GPU else GPU
            else -> CPU
        }
    }

    /** What the GPU and NPU switches the choice replaced meant: both on (the default) NPU + GPU, else the units left
     *  on. */
    fun fromSwitches(gpu: Boolean, npu: Boolean) = when {
        gpu && npu -> NPU_GPU
        npu -> NPU_CPU
        gpu -> GPU
        else -> CPU
    }
}
