package xyz.tifu.geode.agent.browser

import android.net.LocalSocket
import android.net.LocalSocketAddress
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeoutOrNull
import xyz.tifu.geode.util.Json
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.security.SecureRandom
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

class CdpException(message: String) : Exception(message)

/**
 * The Chrome DevTools Protocol to one of the app's own WebViews: its DevTools socket (@webview_devtools_remote_<pid>,
 * open once WebView.setWebContentsDebuggingEnabled is on; Chromium lets in the app's own uid), one WebSocket to the
 * page's target. A reader thread answers calls by their id and hands the events to [onEvent].
 */
class Cdp private constructor(
    private val socket: LocalSocket,
    private val onEvent: (method: String, params: Map<String, Any?>) -> Unit,
) : Closeable {
    private val input = BufferedInputStream(socket.inputStream)
    private val output = socket.outputStream
    private val ids = AtomicInteger()
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<Map<String, Any?>>>()
    private val random = SecureRandom()

    @Volatile
    var closed = false
        private set

    init {
        Thread({ read() }, "geode-cdp").apply { isDaemon = true }.start()
    }

    /** [method] with [params]; its result, or a CdpException (its error, or no answer in [timeoutMs]). */
    @Suppress("UNCHECKED_CAST")
    suspend fun call(method: String, params: Map<String, Any?> = emptyMap(), timeoutMs: Long = 30_000): Map<String, Any?> {
        if (closed) throw CdpException("the browser's connection is closed")
        val id = ids.incrementAndGet()
        val done = CompletableDeferred<Map<String, Any?>>()
        pending[id] = done
        try {
            send(Json.write(linkedMapOf("id" to id, "method" to method, "params" to params)))
            val m = withTimeoutOrNull(timeoutMs) { done.await() }
                ?: throw CdpException("$method: no answer in ${timeoutMs / 1000} s")
            (m["error"] as? Map<String, Any?>)?.let { throw CdpException("$method: ${it["message"] ?: it}") }
            return m["result"] as? Map<String, Any?> ?: emptyMap()
        } finally {
            pending.remove(id)
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { socket.close() }
        for (d in pending.values) d.complete(mapOf("error" to mapOf("message" to "the browser's connection closed")))
    }

    @Suppress("UNCHECKED_CAST")
    private fun read() {
        try {
            while (!closed) {
                val m = Json.parse(frame()) as? Map<String, Any?> ?: continue
                val id = (m["id"] as? Number)?.toInt()
                if (id != null) pending[id]?.complete(m)
                else (m["method"] as? String)?.let { runCatching { onEvent(it, m["params"] as? Map<String, Any?> ?: emptyMap()) } }
            }
        } catch (_: IOException) {
        } catch (_: RuntimeException) {
        } finally {
            close()
        }
    }

    private fun send(text: String) {
        val p = text.toByteArray(Charsets.UTF_8)
        val f = ByteArrayOutputStream(p.size + 14)
        f.write(0x81)   // a whole text frame
        when {
            p.size < 126 -> f.write(0x80 or p.size)
            p.size < 65536 -> {
                f.write(0x80 or 126)
                f.write(p.size shr 8)
                f.write(p.size)
            }
            else -> {
                f.write(0x80 or 127)
                for (i in 7 downTo 0) f.write((p.size.toLong() shr (8 * i)).toInt())
            }
        }
        // a client's frames are masked
        val mask = ByteArray(4).also(random::nextBytes)
        f.write(mask)
        for (i in p.indices) f.write(p[i].toInt() xor mask[i and 3].toInt())
        synchronized(output) {
            output.write(f.toByteArray())
            output.flush()
        }
    }

    private fun u8(): Int {
        val c = input.read()
        if (c < 0) throw IOException("closed")
        return c
    }

    /** The next message's text (its fragments joined; pings and pongs skipped). */
    private fun frame(): String {
        val msg = ByteArrayOutputStream()
        while (true) {
            val b0 = u8()
            val b1 = u8()
            var len = (b1 and 0x7f).toLong()
            if (len == 126L) len = ((u8() shl 8) or u8()).toLong()
            else if (len == 127L) {
                len = 0
                repeat(8) { len = (len shl 8) or u8().toLong() }
            }
            if (len > MAX_MESSAGE) throw IOException("a message of $len bytes")
            val mask = if (b1 and 0x80 != 0) ByteArray(4) { u8().toByte() } else null
            val p = ByteArray(len.toInt())
            var got = 0
            while (got < p.size) {
                val r = input.read(p, got, p.size - got)
                if (r < 0) throw IOException("closed")
                got += r
            }
            if (mask != null) for (i in p.indices) p[i] = (p[i].toInt() xor mask[i and 3].toInt()).toByte()
            when (b0 and 0x0f) {
                8 -> throw IOException("closed by the browser")
                9, 10 -> continue
            }
            msg.write(p)
            if (b0 and 0x80 != 0) return msg.toString("UTF-8")
        }
    }

    companion object {
        private const val MAX_MESSAGE = 64L shl 20

        /** This process's WebView DevTools socket. */
        fun socketName() = "webview_devtools_remote_" + android.os.Process.myPid()

        private fun open(): LocalSocket = LocalSocket().apply {
            connect(LocalSocketAddress(socketName(), LocalSocketAddress.Namespace.ABSTRACT))
        }

        private fun head(input: InputStream): String {
            val b = ByteArrayOutputStream()
            var ends = 0
            while (ends < 4) {
                val c = input.read()
                if (c < 0) break
                b.write(c)
                ends = if (c == '\r'.code || c == '\n'.code) ends + 1 else 0
            }
            return b.toString("UTF-8")
        }

        /** One GET of the DevTools HTTP endpoints (/json/list): its body. */
        fun http(path: String): String = open().use { s ->
            s.soTimeout = 5000
            s.outputStream.write("GET $path HTTP/1.1\r\nHost: localhost\r\n\r\n".toByteArray())
            s.outputStream.flush()
            val input = BufferedInputStream(s.inputStream)
            val h = head(input)
            val len = h.lineSequence().firstOrNull { it.startsWith("content-length:", ignoreCase = true) }
                ?.substringAfter(':')?.trim()?.toIntOrNull() ?: return h
            val body = ByteArray(len)
            var got = 0
            while (got < len) {
                val r = input.read(body, got, len - got)
                if (r < 0) break
                got += r
            }
            String(body, 0, got, Charsets.UTF_8)
        }

        /** A connection to the target at [path] (its webSocketDebuggerUrl's path). */
        fun connect(path: String, onEvent: (String, Map<String, Any?>) -> Unit): Cdp {
            val s = open()
            s.outputStream.write(("GET $path HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
                "Sec-WebSocket-Key: ${android.util.Base64.encodeToString(ByteArray(16).also { SecureRandom().nextBytes(it) },
                    android.util.Base64.NO_WRAP)}\r\nSec-WebSocket-Version: 13\r\n\r\n").toByteArray())
            s.outputStream.flush()
            // the head byte by byte, from the socket itself: the reader's buffer starts at the first frame
            val raw = s.inputStream
            val h = head(raw)
            if (!h.startsWith("HTTP/1.1 101")) {
                runCatching { s.close() }
                throw CdpException("the browser refused the connection: ${h.lineSequence().firstOrNull()}")
            }
            return Cdp(s, onEvent)
        }
    }
}
