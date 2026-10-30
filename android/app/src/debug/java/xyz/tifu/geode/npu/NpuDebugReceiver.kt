package xyz.tifu.geode.npu

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import java.io.File
import kotlin.concurrent.thread

/**
 * Debug builds only: does an app process (SELinux domain untrusted_app) reach the NPU? Runs npu/host/probe.c in the
 * engine's process (:engine): Geode's cDSP module opened in an unsigned PD through the vendor's public
 * libcdsprpc.so, then FastRPC/dspqueue round trips and HMX GEMM checks and timings.
 *
 *     adb shell am broadcast -n xyz.tifu.geode/.npu.NpuDebugReceiver [--ei reps 5]
 *
 * The result goes to files/npu/probe.txt (adb shell run-as xyz.tifu.geode cat files/npu/probe.txt).
 */
class NpuDebugReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val app = context.applicationContext
        val out = File(app.filesDir, "npu/probe.txt").apply { parentFile?.mkdirs() }
        out.writeText("running\n")
        val reps = intent.getIntExtra("reps", 5)
        val pending = goAsync()
        thread(name = "npu-probe") {
            val t0 = System.currentTimeMillis()
            val text = runCatching {
                System.loadLibrary("geode_npu")
                probe(app.applicationInfo.nativeLibraryDir, reps)
            }.getOrElse { "failed: $it" }
            out.writeText(text + "done in ${System.currentTimeMillis() - t0} ms\n")
            pending.finish()
        }
    }

    companion object {
        @JvmStatic
        external fun probe(libDir: String, reps: Int): String
    }
}
