package xyz.tifu.geode.bench

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import kotlinx.coroutines.flow.MutableStateFlow
import org.json.JSONArray
import org.json.JSONObject
import xyz.tifu.geode.data.SiteApi
import java.io.File
import java.security.SecureRandom

/**
 * The device report (Benchmark -> Check and send), for the developer: what the phone is - maker, model, SoC, firmware
 * build (its GPU driver comes with it), memory, storage, power state - and what every benchmark and the GPU checks
 * found, under a random number shown on the screen. Sent to geode.tifu.xyz (signed, SiteApi) only when the owner taps
 * the button: once at the start, then after each test under the same number, so that a session cut short (a remote
 * test lab's, a crash) still leaves everything found so far. No account, no device id, no chats or files.
 */
object DeviceReport {
    const val PATH = "/api/device"
    /**
     * The check started from adb (a remote test lab's phone, often offline):
     *     am start -n xyz.tifu.geode/.MainActivity -a xyz.tifu.geode.action.DEVICE_CHECK
     * Every test runs and the report is kept in a file only (dir()), never sent: any app could start this, and
     * only the owner's tap sends.
     */
    const val ACTION_CHECK = "xyz.tifu.geode.action.DEVICE_CHECK"
    val requested = MutableStateFlow(false)
    /** The server takes 64k; the results are kept under it */
    private const val MAX_BYTES = 60_000
    private const val ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    private val random = SecureRandom()

    fun newId(): String = String(CharArray(8) { ALPHABET[random.nextInt(ALPHABET.length)] })

    /** "K7Q2-M9XD": as the screen shows it (the server and the /delete page take either) */
    fun display(id: String) = id.chunked(4).joinToString("-")

    /** How a check went: which tests ran, failed (and why), took how long */
    class Progress(val order: List<String>) {
        val done = mutableListOf<String>()
        /** the tests whose result this check stored, done or failed */
        val stored = mutableListOf<String>()
        val failed = LinkedHashMap<String, String>()
        val ms = LinkedHashMap<String, Long>()
        var running: String? = null
        var finished = false

        fun json(): JSONObject = JSONObject().apply {
            put("order", JSONArray(order))
            put("done", JSONArray(done))
            put("failed", JSONObject(failed as Map<*, *>))
            put("ms", JSONObject(ms as Map<*, *>))
            running?.let { put("running", it) }
            put("finished", finished)
        }
    }

    fun build(context: Context, id: String, started: Long, progress: Progress, results: JSONObject,
              estimate: JSONObject?, settings: JSONObject): JSONObject {
        val r = JSONObject()
        r.put("id", id)
        r.put("schema", 1)
        r.put("app", SiteApi.version(context))
        r.put("started", started)
        r.put("sent", System.currentTimeMillis())
        r.put("device", device())
        r.put("languages", languages(context))
        r.put("state", state(context))
        r.put("settings", settings)
        r.put("progress", progress.json())
        estimate?.let { r.put("estimate", it) }
        // the results this check stored, a failed test's too (an older one, from before it, stays out)
        val res = JSONObject()
        for (t in progress.stored) results.optJSONObject(t)?.let { res.put(t, it) }
        r.put("results", res)
        // over the limit: the largest results dropped, each marked with its size
        while (r.toString().length > MAX_BYTES) {
            val big = res.keys().asSequence().maxByOrNull { res.get(it).toString().length } ?: break
            val size = res.get(big).toString().length
            if (size < 200) break
            res.put(big, JSONObject().put("dropped_bytes", size))
        }
        return r
    }

    fun send(context: Context, report: JSONObject): Boolean =
        SiteApi.post(context, PATH, "application/json", report.toString().toByteArray())

    /**
     * Where every report is kept, each part over the last: the app's external files, which adb reads
     * (adb shell cat /sdcard/Android/data/xyz.tifu.geode/files/reports/<id>.json) and other apps do not.
     */
    fun dir(context: Context) = File(context.getExternalFilesDir(null) ?: context.filesDir, "reports").apply { mkdirs() }

    /** The report written whole (a new file renamed over the old); pending: one the owner sent that did not go */
    fun save(context: Context, report: JSONObject, pending: Boolean): File {
        val d = dir(context)
        val id = report.getString("id")
        val f = File(d, "$id.json")
        val tmp = File(d, "$id.json.tmp")
        tmp.writeText(report.toString(1))
        if (!tmp.renameTo(f)) {
            f.delete()
            tmp.renameTo(f)
        }
        File(d, "$id.unsent").let { if (pending) it.writeText("") else it.delete() }
        return f
    }

    /** The reports the owner sent while offline, sent now where possible (at the app's start) */
    fun flush(context: Context) {
        val d = dir(context)
        for (mark in d.listFiles { f -> f.name.endsWith(".unsent") }.orEmpty()) {
            val f = File(d, mark.name.removeSuffix(".unsent") + ".json")
            val r = runCatching { JSONObject(f.readText()) }.getOrNull()
            if (r == null) mark.delete() else if (send(context, r)) mark.delete()
        }
    }

    private fun device() = JSONObject().apply {
        put("manufacturer", Build.MANUFACTURER)
        put("brand", Build.BRAND)
        put("model", Build.MODEL)
        put("device", Build.DEVICE)
        put("product", Build.PRODUCT)
        put("hardware", Build.HARDWARE)
        put("board", Build.BOARD)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            put("soc_manufacturer", Build.SOC_MANUFACTURER)
            put("soc_model", Build.SOC_MODEL)
        }
        put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
        put("android", Build.VERSION.RELEASE)
        put("sdk", Build.VERSION.SDK_INT)
        put("security_patch", Build.VERSION.SECURITY_PATCH)
        // the firmware build, the same on every phone with this firmware: the GPU driver's version comes with it
        put("build", Build.FINGERPRINT)
    }

    /** The phone's languages, the one picked for the app ("" = the phone's) and the one its texts are shown in */
    private fun languages(context: Context) = JSONObject().apply {
        put("device", android.content.res.Resources.getSystem().configuration.locales.toLanguageTags())
        put("app_picked", androidx.appcompat.app.AppCompatDelegate.getApplicationLocales().toLanguageTags())
        put("app_shown", context.resources.configuration.locales.toLanguageTags())
    }

    /** What moves the numbers: free memory, storage, power saving, heat, battery */
    private fun state(context: Context) = JSONObject().apply {
        val am = context.getSystemService(ActivityManager::class.java)
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        put("ram_total", mi.totalMem)
        put("ram_available", mi.availMem)
        put("ram_low_threshold", mi.threshold)
        put("ram_low", mi.lowMemory)
        put("heap_large_mb", am.largeMemoryClass)
        runCatching { StatFs(context.filesDir.path) }.getOrNull()?.let {
            put("storage_free", it.availableBytes)
            put("storage_total", it.totalBytes)
        }
        val pm = context.getSystemService(PowerManager::class.java)
        put("power_save", pm.isPowerSaveMode)
        put("thermal", pm.currentThermalStatus)
        put("sustained_performance", pm.isSustainedPerformanceModeSupported)
        val bm = context.getSystemService(BatteryManager::class.java)
        put("battery", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY))
        put("charging", bm.isCharging)
        put("cpus", Runtime.getRuntime().availableProcessors())
    }
}
