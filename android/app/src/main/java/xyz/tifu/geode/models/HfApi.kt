package xyz.tifu.geode.models

import org.json.JSONArray
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/** One file of a Hugging Face repository. sha256 is the LFS object id (empty for files kept in git). */
data class HfFile(val path: String, val size: Long, val sha256: String, val gitOid: String)

/** A directory of the repository that holds a geode.json: one downloadable model. */
data class HfPack(val repo: String, val revision: String, val path: String, val files: List<HfFile>) {
    val bytes: Long get() = files.sumOf { it.size }
    val label: String get() = path.ifEmpty { repo.substringAfter('/') }
}

class HfException(message: String, val httpCode: Int = 0) : IOException(message)

/** The few Hugging Face Hub endpoints a download needs: the file tree and resolve/ for the bytes. */
object HfApi {
    private const val HOST = "https://huggingface.co"
    private const val USER_AGENT = "Geode/0.1 (Android)"

    /** Lists the repository and finds every model pack in it (a directory with geode.json, the root included). */
    fun findPacks(repo: String, revision: String, token: String): List<HfPack> {
        val files = tree(repo, revision, token)
        val packDirs = files.filter { it.path == PackManifest.FILE || it.path.endsWith("/" + PackManifest.FILE) }
            .map { it.path.removeSuffix(PackManifest.FILE).removeSuffix("/") }
        return packDirs.map { dir ->
            val prefix = if (dir.isEmpty()) "" else "$dir/"
            // only the pack's own files (not nested packs, not README.md and friends at the root)
            val own = files.filter { f ->
                f.path.startsWith(prefix) && !f.path.removePrefix(prefix).contains('/') &&
                    !f.path.endsWith(".md") && !f.path.endsWith(".gitattributes")
            }
            HfPack(repo, revision, dir, own.map { it.copy(path = it.path.removePrefix(prefix)) })
        }.sortedBy { it.path }
    }

    /** Every file of the repository, following the API's pages. */
    fun tree(repo: String, revision: String, token: String): List<HfFile> {
        val out = ArrayList<HfFile>()
        var url: String? = "$HOST/api/models/$repo/tree/${enc(revision)}?recursive=true&expand=false"
        while (url != null) {
            val conn = open(url, token, null)
            val code = conn.responseCode
            if (code != 200) {
                conn.disconnect()
                throw HfException(when (code) {
                    401, 403 -> "access denied: the repository is private or gated (set a token)"
                    404 -> "no such repository or revision: $repo@$revision"
                    else -> "Hugging Face answered HTTP $code"
                }, code)
            }
            val body = conn.inputStream.use { it.readBytes().decodeToString() }
            url = nextLink(conn.getHeaderField("Link"))
            conn.disconnect()
            val arr = JSONArray(body)
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                if (o.optString("type") != "file") continue
                val lfs = o.optJSONObject("lfs")
                out += HfFile(
                    path = o.getString("path"),
                    size = lfs?.optLong("size") ?: o.optLong("size"),
                    sha256 = lfs?.optString("oid").orEmpty(),
                    gitOid = o.optString("oid"),
                )
            }
        }
        return out
    }

    /** A small text file of the repository (a geode.json), at most 4 MB. */
    fun text(repo: String, revision: String, path: String, token: String): String {
        val (conn, _) = download(resolveUrl(repo, revision, path), token, 0)
        try {
            val bytes = conn.inputStream.use { it.readNBytes(4 shl 20) }
            return bytes.decodeToString()
        } finally {
            conn.disconnect()
        }
    }

    fun resolveUrl(repo: String, revision: String, path: String): String =
        "$HOST/$repo/resolve/${enc(revision)}/" + path.split('/').joinToString("/") { enc(it) }

    /**
     * Opens a download from [offset]. Follows the redirects itself: Hugging Face sends the bytes from a CDN, the
     * token must go only to huggingface.co, and the Range header must survive every hop.
     * Returns the stream and whether the server honoured the range (false: it starts from byte 0).
     */
    fun download(url: String, token: String, offset: Long): Pair<HttpURLConnection, Boolean> {
        var current = url
        repeat(10) {
            val conn = open(current, token.takeIf { URL(current).host.endsWith("huggingface.co") } ?: "",
                if (offset > 0) "bytes=$offset-" else null)
            conn.instanceFollowRedirects = false
            when (val code = conn.responseCode) {
                200 -> return conn to false
                206 -> return conn to true
                301, 302, 303, 307, 308 -> {
                    val location = conn.getHeaderField("Location") ?: throw HfException("redirect without Location")
                    conn.disconnect()
                    current = URL(URL(current), location).toString()
                }
                416 -> {
                    conn.disconnect()
                    throw HfException("the server rejected the resume point", code)
                }
                else -> {
                    conn.disconnect()
                    throw HfException(when (code) {
                        401, 403 -> "access denied (set a token for private or gated repositories)"
                        404 -> "file not found"
                        else -> "HTTP $code"
                    }, code)
                }
            }
        }
        throw HfException("too many redirects")
    }

    private fun open(url: String, token: String, range: String?): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            setRequestProperty("User-Agent", USER_AGENT)
            if (token.isNotBlank()) setRequestProperty("Authorization", "Bearer ${token.trim()}")
            if (range != null) {
                setRequestProperty("Range", range)
                // a compressed body would make the byte offsets meaningless
                setRequestProperty("Accept-Encoding", "identity")
            }
        }

    /** RFC 8288 Link: <url>; rel="next" */
    private fun nextLink(header: String?): String? = header?.split(',')?.firstNotNullOfOrNull { part ->
        val m = Regex("<([^>]+)>\\s*;\\s*rel=\"?next\"?").find(part.trim())
        m?.groupValues?.get(1)
    }

    private fun enc(s: String) = URLEncoder.encode(s, "UTF-8").replace("+", "%20")
}
