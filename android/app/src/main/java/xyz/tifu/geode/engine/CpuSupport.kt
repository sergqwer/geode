package xyz.tifu.geode.engine

import java.io.File

/**
 * Whether this phone's CPU runs the engine. libgeode is built for ARMv8.2 with the dot product (SDOT / UDOT, the
 * kernel's "asimddp") and half precision ("asimdhp"): on a CPU without them - Cortex-A53 / A73 era phones, or an x86
 * emulator translating ARM code, as Play's pre-launch robot runs - its first kernel kills the engine's process with an
 * illegal instruction. Android lists every core's features in /proc/cpuinfo; when it cannot be read, the engine is tried.
 */
object CpuSupport {
    val engine: Boolean by lazy {
        runCatching { File("/proc/cpuinfo").readLines() }.getOrNull()?.let(::runsEngine) ?: true
    }

    /** From /proc/cpuinfo's lines: every core's "Features" has both (no ARM features listed, as on x86: no) */
    fun runsEngine(cpuinfo: List<String>): Boolean {
        val cores = cpuinfo.filter { it.startsWith("Features") }
            .map { it.substringAfter(':').trim().split(Regex("\\s+")).toSet() }
        return cores.isNotEmpty() && cores.all { "asimddp" in it && "asimdhp" in it }
    }
}
