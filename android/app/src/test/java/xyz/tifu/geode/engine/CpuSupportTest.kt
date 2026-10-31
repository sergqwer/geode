package xyz.tifu.geode.engine

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** CpuSupport against real /proc/cpuinfo lines (the phones' and the emulator's of Play's pre-launch reports) */
class CpuSupportTest {
    @Test
    fun cortexA76AndA55() {
        // Snapdragon 720G (Redmi Note 9 Pro): every core has both
        val features = "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp"
        assertTrue(CpuSupport.runsEngine(listOf("processor\t: 0", features, "", "processor\t: 7", features)))
    }

    @Test
    fun cortexA53() {
        // ARMv8.0: no half precision, no dot product
        val features = "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid"
        assertFalse(CpuSupport.runsEngine(listOf("processor\t: 0", features)))
    }

    @Test
    fun oneCoreWithout() {
        assertFalse(CpuSupport.runsEngine(listOf(
            "Features\t: fp asimd fphp asimdhp asimddp",
            "Features\t: fp asimd fphp asimdhp")))
    }

    @Test
    fun x86Emulator() {
        // Play's pre-launch device: x86_64 translating ARM code, its cpuinfo lists x86 flags only
        assertFalse(CpuSupport.runsEngine(listOf("processor\t: 0", "vendor_id\t: GenuineIntel",
            "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov sse sse2 ssse3 sse4_1 sse4_2 avx")))
    }
}
