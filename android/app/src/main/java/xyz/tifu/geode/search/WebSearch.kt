package xyz.tifu.geode.search

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import java.util.concurrent.TimeUnit

class SearchException(message: String) : Exception(message)

/**
 * The web search: geode-search (SearXNG's metasearch rewritten in Rust, AGPL-3.0, a program of its own: geode/search),
 * shipped in the APK's native library folder and run for each search - it asks the search engines itself, from the
 * phone, as Chrome on Android does (its TLS, HTTP/2 and headers): no search API, no key, nothing kept running between
 * searches. Its engines' state (tokens, an engine resting after a captcha) is kept in files/search.
 */
class WebSearch(context: Context) {
    private val exe = File(context.applicationInfo.nativeLibraryDir, "libgeodesearch.so")
    private val cache = File(context.filesDir, "search")

    val available: Boolean get() = exe.isFile

    /** [query]'s results as geode-search prints them (one JSON object: answers, results, unresponsive engines). */
    suspend fun search(query: String, max: Int = 8, lang: String? = null): String = withContext(Dispatchers.IO) {
        if (!available) throw SearchException("web search is not in this build")
        cache.mkdirs()
        val p = ProcessBuilder(exe.path, "--max", max.coerceIn(1, 20).toString(), "--cache", cache.path, "--", query)
            .apply {
                environment()["GEODE_SEARCH_LANG"] = lang ?: Locale.getDefault().language
            }
            .start()
        p.outputStream.close()
        val out = ByteArrayOutputStream()
        val err = StringBuilder()
        val reader = Thread { runCatching { p.inputStream.use { copyCapped(it, out) } } }.apply { start() }
        val errors = Thread { runCatching { p.errorStream.bufferedReader().use { r -> r.lineSequence().take(64).forEach { err.appendLine(it) } } } }
            .apply { start() }
        if (!p.waitFor(TIMEOUT_S, TimeUnit.SECONDS)) {
            p.destroyForcibly()
            throw SearchException("the search took too long")
        }
        reader.join(2000)
        errors.join(2000)
        if (p.exitValue() != 0) throw SearchException(err.lines().lastOrNull { it.isNotBlank() } ?: "the search failed")
        out.toString(Charsets.UTF_8)
    }

    private fun copyCapped(input: java.io.InputStream, out: ByteArrayOutputStream) {
        val buf = ByteArray(16 shl 10)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            if (out.size() < MAX_OUTPUT) out.write(buf, 0, n)
        }
    }

    companion object {
        private const val TIMEOUT_S = 20L
        private const val MAX_OUTPUT = 2 shl 20
    }
}
