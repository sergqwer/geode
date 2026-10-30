package xyz.tifu.geode.data

import org.junit.Assert.assertEquals
import org.junit.Test

/** The request signature as the server checks it (site/testers/src/sig.rs, test `shared_vector`: the same value) */
class SiteApiTest {
    @Test
    fun sharedVector() {
        val key = ByteArray(32) { 0x11 }
        val sig = SiteApi.signature(key, "POST", "/api/device", "0.9.4+5", "1761700000", "0123456789abcdef",
            "{\"id\":\"ABCD2345\"}".toByteArray())
        assertEquals(SHARED, sig)
    }

    companion object {
        // Python's hmac gives the same: a third implementation
        const val SHARED = "0d0802e4c03024f63a3f2ea61d5b94d2ef8e20660b0cc1e9b343d8b4d71978d6"
    }
}
