package xyz.tifu.geode.probe

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.os.Process
import java.io.File
import xyz.tifu.geode.engine.GeodeNative

/**
 * The memory test's victim (PLAN.md P6.2). Runs in the :probe process, bound like the engine (BIND_IMPORTANT), so
 * Android treats it the way it will treat the engine; it grows memory until the low-memory killer ends it. The app
 * then reads the per-step history the probe left in [statusFile].
 */
class ProbeService : Service() {
    override fun onBind(intent: Intent): IBinder {
        val maxBytes = intent.getLongExtra(EXTRA_MAX_BYTES, 0L)
        Thread({
            val rc = GeodeNative.memoryProbe(statusFile(this).path, maxBytes, STEP_BYTES, PAUSE_MS)
            // it reached maxBytes alive (a phone with more memory than the test asks for)
            statusFile(this).appendText("end=$rc\n")
            Process.killProcess(Process.myPid())
        }, "geode-probe").start()
        return Binder()
    }

    companion object {
        const val EXTRA_MAX_BYTES = "max_bytes"
        const val STEP_BYTES = 64L shl 20
        const val PAUSE_MS = 50

        fun statusFile(service: android.content.Context) = File(service.filesDir, "memprobe.txt")
    }
}
