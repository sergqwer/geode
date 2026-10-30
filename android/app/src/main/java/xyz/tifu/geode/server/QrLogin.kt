package xyz.tifu.geode.server

import com.google.zxing.BarcodeFormat
import com.google.zxing.EncodeHintType
import com.google.zxing.qrcode.QRCodeWriter
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import java.security.SecureRandom
import java.util.Base64
import java.util.concurrent.ConcurrentHashMap

/**
 * A browser waiting to be signed in by a QR code: [id] only the browser knows (it collects the session with it),
 * [code] is in the QR code (with it the app's user says yes - the app, never a request). Its address and browser are
 * what the app shows before the yes.
 */
class QrPending(val id: String, val code: String, val address: String, val agent: String, val created: Long) {
    val approved = CompletableDeferred<Unit>()
    val expired: Boolean get() = System.currentTimeMillis() - created > QrLogins.LIFETIME_MS
}

/**
 * The web chat's sign-in by QR code, like a messenger's web version: the sign-in page shows a code; the phone with
 * Geode scans it (its own scanner, or the camera app opening the geode:// link) and, the user agreeing, the browser
 * is signed in - no password typed. In the app's process only: approving takes no request.
 */
object QrLogins {
    const val LIFETIME_MS = 3 * 60_000L
    private const val MAX_PENDING = 32
    const val SCHEME = "geode://login?c="

    private val random = SecureRandom()
    private val byId = ConcurrentHashMap<String, QrPending>()
    private val byCode = ConcurrentHashMap<String, QrPending>()

    /** A code the app was given (scanned, or opened as a link) and asks the user about; null: none. */
    val incoming = MutableStateFlow<String?>(null)

    private fun token(bytes: Int): String =
        Base64.getUrlEncoder().withoutPadding().encodeToString(ByteArray(bytes).also { random.nextBytes(it) })

    private fun prune() {
        for (p in byId.values) if (p.expired) {
            byId.remove(p.id)
            byCode.remove(p.code)
        }
    }

    /** A new sign-in for the browser at [address]; null when too many wait already. */
    fun create(address: String, agent: String): QrPending? {
        prune()
        if (byId.size >= MAX_PENDING) return null
        val p = QrPending(token(18), token(18), address, agent, System.currentTimeMillis())
        byId[p.id] = p
        byCode[p.code] = p
        return p
    }

    fun byId(id: String): QrPending? = byId[id]?.takeIf { !it.expired }

    /** The sign-in a scanned text names (a geode://login link or the bare code), while it is valid. */
    fun byScan(text: String): QrPending? {
        val code = text.trim().removePrefix(SCHEME).substringBefore('&')
        return byCode[code]?.takeIf { !it.expired && !it.approved.isCompleted }
    }

    /** The user said yes: the browser's next look collects its session. */
    fun approve(p: QrPending) {
        byCode.remove(p.code)
        p.approved.complete(Unit)
    }

    /** The browser has its session: the sign-in is done with. */
    fun finish(p: QrPending) {
        byId.remove(p.id)
        byCode.remove(p.code)
    }

    /** A QR code of [text] as SVG: black modules on white, a quiet zone of four, at [px] pixels a side. */
    fun svg(text: String, px: Int = 220): String {
        val m = QRCodeWriter().encode(text, BarcodeFormat.QR_CODE, 0, 0,
            mapOf(EncodeHintType.ERROR_CORRECTION to ErrorCorrectionLevel.M, EncodeHintType.MARGIN to 0))
        val n = m.width + 8
        val d = StringBuilder()
        for (y in 0 until m.height) for (x in 0 until m.width) if (m.get(x, y)) d.append("M${x + 4} ${y + 4}h1v1h-1z")
        return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $n $n\" width=\"$px\" height=\"$px\" " +
            "shape-rendering=\"crispEdges\"><rect width=\"$n\" height=\"$n\" fill=\"#fff\"/><path fill=\"#000\" d=\"$d\"/></svg>"
    }

    /** "Edge · Windows" from a User-Agent (what the app shows before the yes). */
    fun describe(agent: String): String {
        val browser = when {
            "Edg/" in agent -> "Edge"
            "OPR/" in agent -> "Opera"
            "Firefox/" in agent -> "Firefox"
            "SamsungBrowser/" in agent -> "Samsung Internet"
            "Chrome/" in agent -> "Chrome"
            "Safari/" in agent -> "Safari"
            else -> null
        }
        val system = when {
            "Windows" in agent -> "Windows"
            "Android" in agent -> "Android"
            "iPhone" in agent || "iPad" in agent -> "iOS"
            "Mac OS" in agent -> "macOS"
            "Linux" in agent -> "Linux"
            else -> null
        }
        return listOfNotNull(browser, system).joinToString(" · ")
    }
}
