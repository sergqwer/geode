package xyz.tifu.geode.probe

import android.app.ActivityManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/** One line of the probe's history (geode.h, geode_memory_probe). */
data class ProbeStep(val held: Long, val available: Long, val swapFree: Long, val vmSwap: Long, val stepMs: Long)

data class MemoryTestResult(
    /** The last level where none of the probe's memory had gone to zram, less a margin: what the app recommends. */
    val safeBytes: Long,
    /** The last level the probe held before it was killed (or the test's ceiling if it never was). */
    val limitBytes: Long,
    val killed: Boolean,
    val steps: List<ProbeStep>,
)

/**
 * Runs the memory test: starts ProbeService in its own process, follows its history while it grows, and reads the
 * result once the low-memory killer has ended it. The UI process is small and has the same priority, so Android
 * kills the probe (the biggest process at that priority), not the app.
 */
class MemoryTest(private val context: Context) {

    /** Runs the test; [progress] gets the memory held so far, about every 200 ms. */
    suspend fun run(progress: (Long) -> Unit): MemoryTestResult {
        val file = ProbeService.statusFile(context)
        withContext(Dispatchers.IO) { file.delete() }
        val total = context.getSystemService(ActivityManager::class.java).let { am ->
            ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem
        }
        val gone = CompletableDeferred<Unit>()
        val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                // binderDied comes a little sooner than onServiceDisconnected
                service?.linkToDeath({ gone.complete(Unit) }, 0)
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                gone.complete(Unit)
            }

            override fun onBindingDied(name: ComponentName?) {
                gone.complete(Unit)
            }
        }
        val intent = Intent(context, ProbeService::class.java).putExtra(ProbeService.EXTRA_MAX_BYTES, total)
        check(context.bindService(intent, conn, Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT)) {
            "cannot start the memory test"
        }
        try {
            while (!gone.isCompleted) {
                delay(200)
                parse(file).lastOrNull()?.let { progress(it.held) }
            }
        } finally {
            context.unbindService(conn)
        }
        // the history is in the page cache; give the kernel a moment after the kill
        delay(300)
        val steps = parse(file)
        val text = withContext(Dispatchers.IO) { if (file.exists()) file.readText() else "" }
        val killed = !text.contains("end=")
        val limit = steps.lastOrNull()?.held ?: 0L
        val clean = lastCleanStep(steps)?.held ?: 0L
        return MemoryTestResult(
            safeBytes = (clean - SAFETY_MARGIN).coerceAtLeast(0L),
            limitBytes = limit,
            killed = killed,
            steps = steps,
        )
    }

    private suspend fun parse(file: java.io.File): List<ProbeStep> = withContext(Dispatchers.IO) {
        if (file.exists()) parseSteps(file.readLines()) else emptyList()
    }

    companion object {
        /** Room for the engine's own code, threads and the buffers around the budget. */
        const val SAFETY_MARGIN = 256L shl 20
        /** VmSwap growth that means the kernel started pushing the probe itself into zram: this much, or a 32nd of
         *  what the probe holds. A 16 GB phone with a big swap moves a little early and stays there (the RedMagic 10
         *  Pro: 0.12 GB of 8.8 GB, then nothing more up to 10.1 GB; 4.7 GB at 10.4 GB). */
        const val SWAP_GROWTH = 32L shl 20
        /** Two steps in a row this many times slower than the early ones mean the kernel is reclaiming hard (one
         *  alone happens: the RedMagic 10 Pro had a 2.7x step at 9.5 GB and fast ones up to 10.5 GB). */
        const val SLOWDOWN = 2.5
        /** Below this MemAvailable the next allocations come from killing apps (the keyboard, the launcher). */
        const val AVAILABLE_FLOOR = 128L shl 20

        /**
         * The last step before memory got tight. Three signs, whichever comes first: the probe's own VmSwap grows
         * (measured from the first step: a Java process starts with some of its pages already in zram), a step
         * takes much longer than the early ones (direct reclaim), or MemAvailable runs out. On the Redmi Note 9 Pro
         * that is ~3.6-3.9 GB depending on what else runs, while the low-memory killer ends the probe at ~4.1-4.3 GB.
         */
        fun parseSteps(lines: List<String>): List<ProbeStep> = lines.mapNotNull { line ->
            val kv = line.trim().split(' ').mapNotNull { p ->
                val i = p.indexOf('=')
                if (i > 0) p.substring(0, i) to (p.substring(i + 1).toLongOrNull() ?: return@mapNotNull null) else null
            }.toMap()
            val held = kv["held"] ?: return@mapNotNull null
            ProbeStep(held, kv["available"] ?: -1, kv["swap_free"] ?: -1, kv["vm_swap"] ?: -1, kv["step_ms"] ?: -1)
        }

        fun lastCleanStep(steps: List<ProbeStep>): ProbeStep? {
            if (steps.isEmpty()) return null
            val swap0 = steps.first().vmSwap
            val early = steps.take(8).map { it.stepMs }.sorted()
            val typical = early[early.size / 2].coerceAtLeast(1)
            fun slow(i: Int) = steps[i].stepMs > typical * SLOWDOWN
            val bad = steps.indices.indexOfFirst { i ->
                val it = steps[i]
                it.vmSwap - swap0 > maxOf(SWAP_GROWTH, it.held / 32) ||
                    (slow(i) && (i + 1 >= steps.size || slow(i + 1))) ||
                    it.available in 0 until AVAILABLE_FLOOR
            }
            return when (bad) {
                -1 -> steps.last()
                0 -> null
                else -> steps[bad - 1]
            }
        }
    }
}
