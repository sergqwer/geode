package xyz.tifu.geode.server

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import xyz.tifu.geode.util.Json
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.Inet4Address
import java.net.Inet6Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/** A request: the method, the path (decoded) and the query, the headers (names in lower case), the body, the
 *  client's address. */
class HttpRequest(
    val method: String,
    val path: String,
    val query: Map<String, String>,
    val headers: Map<String, String>,
    val body: ByteArray,
    val remote: String = "",
) {
    fun header(name: String): String? = headers[name.lowercase()]
}

/** The answer to a request, whole or as server-sent events; the connection closes after it. */
class HttpResponder(private val out: OutputStream) {
    /** The status line went out (an error now can only go inside the body). */
    var started = false
        private set

    /** Other sites' pages may read the answer (the API's); the web chat's pages are its own. */
    var cors = true

    fun send(status: Int, contentType: String, body: ByteArray, headers: Map<String, String> = emptyMap()) {
        head(status, contentType, headers + ("Content-Length" to body.size.toString()))
        out.write(body)
        out.flush()
    }

    fun json(status: Int, v: Any?, headers: Map<String, String> = emptyMap()) =
        send(status, "application/json", Json.write(v).toByteArray(), headers)

    fun text(status: Int, s: String, type: String = "text/plain; charset=utf-8") = send(status, type, s.toByteArray())

    /** Server-sent events: the head now, then [event] and [comment] as they come, each sent at once. */
    fun events() = head(200, "text/event-stream; charset=utf-8", mapOf("Cache-Control" to "no-cache",
        "X-Accel-Buffering" to "no"))

    fun event(name: String?, data: String) {
        val sb = StringBuilder()
        if (name != null) sb.append("event: ").append(name).append('\n')
        sb.append("data: ").append(data).append("\n\n")
        out.write(sb.toString().toByteArray())
        out.flush()
    }

    /** A line clients skip: keeps the connection up while the prompt is read (minutes, on a phone). */
    fun comment(text: String) {
        out.write(": $text\n\n".toByteArray())
        out.flush()
    }

    private fun head(status: Int, contentType: String, headers: Map<String, String>) {
        check(!started)
        started = true
        val sb = StringBuilder("HTTP/1.1 ").append(status).append(' ').append(reason(status)).append("\r\n")
        sb.append("Content-Type: ").append(contentType).append("\r\n")
        for ((k, v) in (if (cors) CORS else emptyMap()) + headers) sb.append(k).append(": ").append(v).append("\r\n")
        sb.append("Connection: close\r\n\r\n")
        out.write(sb.toString().toByteArray(Charsets.UTF_8))
        out.flush()
    }

    private fun reason(status: Int) = when (status) {
        200 -> "OK"
        204 -> "No Content"
        400 -> "Bad Request"
        401 -> "Unauthorized"
        404 -> "Not Found"
        405 -> "Method Not Allowed"
        409 -> "Conflict"
        429 -> "Too Many Requests"
        413 -> "Payload Too Large"
        500 -> "Internal Server Error"
        503 -> "Service Unavailable"
        else -> "Status"
    }

    companion object {
        /** A web page on another device may call the API too (it needs the key all the same). */
        private val CORS = mapOf(
            "Access-Control-Allow-Origin" to "*",
            "Access-Control-Allow-Headers" to "*",
            "Access-Control-Allow-Methods" to "GET, POST, OPTIONS",
        )
    }
}

/**
 * A small HTTP/1.1 server for the local network: one request a connection (Connection: close), its body by
 * Content-Length or chunked, the answer whole or as server-sent events. It answers the local network only
 * ([LocalNetwork.accepts]): a connection from anywhere else - a public IPv6 address, the mobile network's - is closed
 * unread.
 */
class HttpServer(
    private val port: Int,
    private val handler: suspend (HttpRequest, HttpResponder) -> Unit,
    /** The address it listens on: every interface (null), or the loopback alone (the agent's server). */
    private val bind: InetAddress? = null,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var socket: ServerSocket? = null
    private val open = AtomicInteger()

    /** The port it listens on (the one asked for, or the one the system gave for 0). */
    val boundPort: Int get() = socket?.localPort ?: port

    /** Binds the port (an IOException when it cannot: another app has it) and starts accepting. */
    fun start() {
        val s = ServerSocket()
        s.reuseAddress = true
        s.bind(if (bind != null) InetSocketAddress(bind, port) else InetSocketAddress(port), BACKLOG)
        socket = s
        thread(name = "geode-http", isDaemon = true) { accept(s) }
    }

    fun stop() {
        runCatching { socket?.close() }
        scope.cancel()
    }

    private fun accept(s: ServerSocket) {
        while (!s.isClosed) {
            val c = try {
                s.accept()
            } catch (e: IOException) {
                break
            }
            if (!LocalNetwork.accepts(c) || open.get() >= MAX_CONNECTIONS) {
                runCatching { c.close() }
                continue
            }
            open.incrementAndGet()
            scope.launch {
                try {
                    serve(c)
                } finally {
                    open.decrementAndGet()
                    runCatching { c.close() }
                }
            }
        }
    }

    private suspend fun serve(c: Socket) {
        c.soTimeout = READ_TIMEOUT_MS
        c.tcpNoDelay = true
        val input = BufferedInputStream(c.getInputStream())
        val out = BufferedOutputStream(c.getOutputStream(), 16 * 1024)
        val r = HttpResponder(out)
        val req = try {
            read(input, out, c.inetAddress?.hostAddress.orEmpty())
        } catch (e: BadRequest) {
            runCatching { r.json(e.status, mapOf("error" to mapOf("message" to e.message, "type" to "invalid_request_error"))) }
            return
        } catch (e: IOException) {
            return
        }
        // reading is done; an answer may take minutes (no timeout while it is written)
        c.soTimeout = 0
        try {
            handler(req, r)
        } catch (e: CancellationException) {
            throw e
        } catch (e: IOException) {
            // the client went away
        } catch (e: Exception) {
            if (!r.started) runCatching {
                r.json(500, mapOf("error" to mapOf("message" to (e.message ?: e.javaClass.simpleName), "type" to "server_error")))
            }
        }
        runCatching { out.flush() }
    }

    private class BadRequest(val status: Int, message: String) : IOException(message)

    private fun read(input: InputStream, out: OutputStream, remote: String): HttpRequest {
        val head = readHead(input)
        val lines = head.split("\r\n")
        val first = lines[0].split(' ')
        if (first.size < 3) throw BadRequest(400, "bad request line")
        val headers = HashMap<String, String>()
        for (l in lines.drop(1)) {
            val i = l.indexOf(':')
            if (i > 0) headers[l.substring(0, i).trim().lowercase()] = l.substring(i + 1).trim()
        }
        // curl waits a second for this before a body over 1 KB
        if (headers["expect"].equals("100-continue", ignoreCase = true)) {
            out.write("HTTP/1.1 100 Continue\r\n\r\n".toByteArray())
            out.flush()
        }
        val body = if (headers["transfer-encoding"]?.contains("chunked", ignoreCase = true) == true) {
            chunked(input)
        } else {
            val n = headers["content-length"]?.toLongOrNull() ?: 0L
            if (n < 0 || n > MAX_BODY) throw BadRequest(413, "the body is larger than ${MAX_BODY shr 20} MB")
            readExactly(input, n.toInt())
        }
        val target = first[1]
        val query = target.substringAfter('?', "").split('&').filter { it.isNotEmpty() }.associate {
            decode(it.substringBefore('=')) to decode(it.substringAfter('=', ""))
        }
        return HttpRequest(first[0].uppercase(), decode(target.substringBefore('?')), query, headers, body, remote)
    }

    private fun decode(s: String) = runCatching { URLDecoder.decode(s, "UTF-8") }.getOrDefault(s)

    /** The request line and the headers, up to the empty line. */
    private fun readHead(input: InputStream): String {
        val b = ByteArrayOutputStream()
        var tail = 0
        while (true) {
            val c = input.read()
            if (c < 0) throw IOException("closed")
            b.write(c)
            tail = (tail shl 8) or c
            if (tail == 0x0d0a0d0a) break
            if (b.size() > MAX_HEAD) throw BadRequest(400, "the headers are too long")
        }
        return b.toString(Charsets.ISO_8859_1.name()).trimEnd()
    }

    private fun readExactly(input: InputStream, n: Int): ByteArray {
        val a = ByteArray(n)
        var at = 0
        while (at < n) {
            val k = input.read(a, at, n - at)
            if (k < 0) throw IOException("closed")
            at += k
        }
        return a
    }

    private fun chunked(input: InputStream): ByteArray {
        val b = ByteArrayOutputStream()
        while (true) {
            val line = readLine(input)
            val n = line.substringBefore(';').trim().toIntOrNull(16) ?: throw BadRequest(400, "a bad chunk")
            if (n == 0) {
                while (readLine(input).isNotEmpty()) Unit  // trailers
                return b.toByteArray()
            }
            if (b.size() + n > MAX_BODY) throw BadRequest(413, "the body is larger than ${MAX_BODY shr 20} MB")
            b.write(readExactly(input, n))
            readLine(input)
        }
    }

    private fun readLine(input: InputStream): String {
        val b = ByteArrayOutputStream()
        while (true) {
            val c = input.read()
            if (c < 0) throw IOException("closed")
            if (c == '\n'.code) return b.toString(Charsets.ISO_8859_1.name()).trimEnd('\r')
            b.write(c)
            if (b.size() > MAX_HEAD) throw BadRequest(400, "a line is too long")
        }
    }

    companion object {
        private const val BACKLOG = 32
        private const val MAX_CONNECTIONS = 32
        private const val READ_TIMEOUT_MS = 30_000
        private const val MAX_HEAD = 32 * 1024
        /** Images come as base64 inside the JSON. */
        private const val MAX_BODY = 64L shl 20
    }
}

/** The phone on the local network: its addresses there, and which connections the server takes. */
object LocalNetwork {
    /** Mobile-data, VPN and virtual interfaces: not the local network. */
    private val notLocal = listOf("rmnet", "r_rmnet", "v4-rmnet", "ccmni", "seth", "pdp", "ppp", "tun", "dummy", "ifb",
        "ip_vti", "ip6_vti", "sit", "p2p-dev")

    private fun lanInterface(n: NetworkInterface?): Boolean =
        n != null && !n.isLoopback && notLocal.none { n.name.startsWith(it) }

    /** Private (or link-local) addresses: 10/8, 172.16/12, 192.168/16, 169.254/16, fc00::/7, fe80::/10. */
    private fun private(a: InetAddress): Boolean {
        if (a.isSiteLocalAddress || a.isLinkLocalAddress) return true
        if (a is Inet6Address) {
            val b = a.address
            if ((b[0].toInt() and 0xfe) == 0xfc) return true
            // an IPv4 address mapped into IPv6 (::ffff:a.b.c.d)
            if ((0 until 10).all { b[it] == 0.toByte() } && b[10] == 0xff.toByte() && b[11] == 0xff.toByte())
                return private(InetAddress.getByAddress(b.copyOfRange(12, 16)))
        }
        return false
    }

    /**
     * A connection the server answers: from this phone itself, or from a private address that reached it through a
     * local network's interface (Wi-Fi, its hotspot, USB or Bluetooth tethering, Ethernet) - not through the mobile
     * network, whose operator's private addresses are other subscribers'.
     */
    fun accepts(c: Socket): Boolean {
        val remote = c.inetAddress ?: return false
        if (remote.isLoopbackAddress) return true
        if (!private(remote)) return false
        return lanInterface(runCatching { NetworkInterface.getByInetAddress(c.localAddress) }.getOrNull())
    }

    /** The phone's IPv4 addresses on its local networks (Wi-Fi first). */
    fun addresses(): List<String> = runCatching {
        NetworkInterface.getNetworkInterfaces().toList()
            .filter { it.isUp && lanInterface(it) }
            .sortedBy { if (it.name.startsWith("wlan")) 0 else 1 }
            .flatMap { it.inetAddresses.toList() }
            .filterIsInstance<Inet4Address>()
            .filter { it.isSiteLocalAddress }
            .mapNotNull { it.hostAddress }
            .distinct()
    }.getOrDefault(emptyList())
}
