package xyz.tifu.geode.models

import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

/** fetchVerified against a local server: a resume, a server that ignores the range, a file with the wrong hash. */
class HfFetchTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val body = Random(7).nextBytes(300_000)
    private val sha = MessageDigest.getInstance("SHA-256").digest(body).toHex()
    private lateinit var server: ServerSocket
    private val requests = AtomicInteger()
    private val ranges = mutableListOf<String>()
    @Volatile private var honourRange = true
    @Volatile private var corrupt = false

    /** A server of one file, one request a connection (the JDK's HttpServer is not on an Android test's classpath). */
    @Before
    fun start() {
        server = ServerSocket(0, 8, InetAddress.getLoopbackAddress())
        Thread {
            while (!server.isClosed) {
                val c = runCatching { server.accept() }.getOrNull() ?: break
                c.use {
                    val head = c.getInputStream().bufferedReader().let { r ->
                        generateSequence { r.readLine()?.takeIf { it.isNotEmpty() } }.toList()
                    }
                    requests.incrementAndGet()
                    val range = head.firstOrNull { it.startsWith("Range:", ignoreCase = true) }?.substringAfter(':')?.trim()
                    synchronized(ranges) { ranges += range.orEmpty() }
                    val bytes = if (corrupt) body.copyOf().also { it[123] = (it[123] + 1).toByte() } else body
                    val from = range?.takeIf { honourRange }?.removePrefix("bytes=")?.removeSuffix("-")?.toInt() ?: 0
                    val part = bytes.copyOfRange(from, bytes.size)
                    val out = c.getOutputStream()
                    out.write(("HTTP/1.1 ${if (from > 0) "206 Partial Content" else "200 OK"}\r\n" +
                        "Content-Length: ${part.size}\r\nConnection: close\r\n\r\n").toByteArray())
                    out.write(part)
                    out.flush()
                }
            }
        }.apply { isDaemon = true }.start()
    }

    @After
    fun stop() = server.close()

    private fun url() = "http://127.0.0.1:${server.localPort}/f"

    private fun fetch(dest: File, counted: LongArray, verifying: MutableList<Boolean> = mutableListOf()) = runBlocking {
        HfApi.fetchVerified(url(), "", dest, body.size.toLong(), sha, badHash = { "bad" },
            progress = { counted[0] += it }, verifying = { verifying += it })
    }

    @Test
    fun aPartFromBeforeGoesOnWhereItEnded() {
        val dest = File(tmp.root, "x.bin").apply { writeBytes(body.copyOf(100_000)) }
        val counted = LongArray(1)
        val verifying = mutableListOf<Boolean>()
        fetch(dest, counted, verifying)
        assertArrayEquals(body, dest.readBytes())
        assertEquals(listOf("bytes=100000-"), ranges)
        assertEquals(body.size.toLong(), counted[0])   // the bytes on disk counted as they were hashed, then the rest
        assertTrue(verifying.first())
        assertFalse(verifying.last())
    }

    @Test
    fun aServerThatSendsTheWholeFileStartsItOver() {
        honourRange = false
        val dest = File(tmp.root, "x.bin").apply { writeBytes(body.copyOf(100_000)) }
        val counted = LongArray(1)
        fetch(dest, counted)
        assertArrayEquals(body, dest.readBytes())
        assertEquals(body.size.toLong(), counted[0])   // the part's bytes taken back
    }

    @Test
    fun theWrongHashTwiceIsAnErrorWithoutMoreTries() {
        corrupt = true
        val dest = File(tmp.root, "x.bin")
        val counted = LongArray(1)
        try {
            fetch(dest, counted)
            fail("a file with the wrong hash was kept")
        } catch (e: BadHashException) {
            assertEquals("bad", e.message)
        }
        assertEquals(2, requests.get())
        assertFalse(dest.exists())
        assertEquals(0L, counted[0])
    }

    @Test
    fun aGitFileIsCheckedByItsBlobHash() {
        val oid = MessageDigest.getInstance("SHA-1").apply { update("blob ${body.size}\u0000".toByteArray()); update(body) }
            .digest().toHex()
        val dest = File(tmp.root, "g.bin")
        runBlocking {
            HfApi.fetchVerified(url(), "", dest, body.size.toLong(), "", oid, badHash = { "bad" }, progress = {})
        }
        assertArrayEquals(body, dest.readBytes())
    }
}
