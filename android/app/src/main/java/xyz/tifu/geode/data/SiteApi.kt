package xyz.tifu.geode.data

import android.content.Context
import android.content.pm.ApplicationInfo
import xyz.tifu.geode.BuildConfig
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * The app's requests to our server, geode.tifu.xyz (answer reports, device reports), and only those: Hugging Face and
 * the web search's engines get none of this. Each carries the app's version and a signature (site/testers/src/sig.rs):
 *
 *     X-Geode-App: 0.9.4+5   X-Geode-Time: unix seconds   X-Geode-Nonce: 32 hex   X-Geode-Sig: HMAC-SHA256, hex
 *
 * signed over "GEODE1\n" method "\n" path "\n" app "\n" time "\n" nonce "\n" hex(SHA-256(body)) with the key built in
 * (build.gradle.kts: BuildConfig.SITE_KEY under SITE_MASK). The server takes a signature once, within ten minutes of
 * its time. A key inside an app can be dug out by someone determined: this keeps forged versions, edited bodies and
 * replays out, not a reverse engineer.
 */
object SiteApi {
    const val BASE = "https://geode.tifu.xyz"
    private val random = SecureRandom()

    /** The version as the server sees it: name + code, "-debug" from a debuggable build */
    fun version(context: Context): String {
        val info = runCatching { context.packageManager.getPackageInfo(context.packageName, 0) }.getOrNull()
        val debug = context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
        val name = (info?.versionName ?: "0").filter { it.isLetterOrDigit() || it in ".+-_" }
        return "$name+${info?.longVersionCode ?: 0}" + if (debug) "-debug" else ""
    }

    /** True when the server answered 200 with "ok":true */
    fun post(context: Context, path: String, contentType: String, body: ByteArray): Boolean = runCatching {
        val c = URL(BASE + path).openConnection() as HttpURLConnection
        try {
            c.requestMethod = "POST"
            c.connectTimeout = 10_000
            c.readTimeout = 15_000
            c.doOutput = true
            c.setRequestProperty("Content-Type", contentType)
            sign(version(context), "POST", path, body).forEach { (k, v) -> c.setRequestProperty(k, v) }
            c.outputStream.use { it.write(body) }
            c.responseCode == 200 && c.inputStream.bufferedReader().readText().contains("\"ok\":true")
        } finally {
            c.disconnect()
        }
    }.getOrDefault(false)

    /** The headers of a request signed now (the version alone when this build has no key) */
    fun sign(app: String, method: String, path: String, body: ByteArray): List<Pair<String, String>> {
        val headers = mutableListOf("X-Geode-App" to app)
        val k = BuildConfig.SITE_KEY
        val m = BuildConfig.SITE_MASK
        val key = ByteArray(k.size) { (k[it].toInt() xor m[it].toInt()).toByte() }
        if (key.isEmpty()) return headers
        val time = (System.currentTimeMillis() / 1000).toString()
        val nonce = hex(ByteArray(16).also { random.nextBytes(it) })
        val sig = signature(key, method, path, app, time, nonce, body)
        key.fill(0)
        headers += listOf("X-Geode-Time" to time, "X-Geode-Nonce" to nonce, "X-Geode-Sig" to sig)
        return headers
    }

    /** The signature itself (sig.rs computes the same; SiteApiTest and its tests share a vector) */
    fun signature(key: ByteArray, method: String, path: String, app: String, time: String, nonce: String,
                  body: ByteArray): String {
        val text = "GEODE1\n$method\n$path\n$app\n$time\n$nonce\n" + hex(MessageDigest.getInstance("SHA-256").digest(body))
        val mac = Mac.getInstance("HmacSHA256").apply { init(SecretKeySpec(key, "HmacSHA256")) }
        return hex(mac.doFinal(text.toByteArray()))
    }

    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
}
