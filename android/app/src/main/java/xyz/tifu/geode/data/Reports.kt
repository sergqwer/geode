package xyz.tifu.geode.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URLEncoder
import java.util.Locale

/**
 * Reporting an answer (Google Play's AI-generated content policy: offensive output can be flagged to the developer
 * without leaving the app). A report goes to geode.tifu.xyz: the reason, the user's note and - when they leave the
 * box ticked - the question and the answer, with the model's and the app's names; nothing that identifies the user.
 * Signed with the app's version (SiteApi).
 * One that cannot go now (offline) waits in a file, one form-encoded line each, and goes at the next start.
 */
class Reports(private val context: Context) {
    data class Report(val reason: String, val note: String, val question: String, val answer: String, val model: String)

    /** True: sent; false: kept and sent later */
    suspend fun send(r: Report): Boolean = withContext(Dispatchers.IO) {
        val body = form(r)
        post(body) || run {
            synchronized(this@Reports) { pending().appendText(body + "\n") }
            false
        }
    }

    /** The reports kept before, sent now where possible */
    suspend fun flush() = withContext(Dispatchers.IO) {
        synchronized(this@Reports) {
            val f = pending()
            if (!f.exists()) return@withContext
            val left = f.readLines().filter { it.isNotBlank() && !post(it) }
            if (left.isEmpty()) f.delete() else f.writeText(left.joinToString("\n", postfix = "\n"))
        }
    }

    private fun pending() = File(context.filesDir, "reports_pending.txt")

    private fun form(r: Report): String {
        val version = runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }
            .getOrNull() ?: "?"
        fun enc(s: String) = URLEncoder.encode(s, "UTF-8")
        return listOf(
            "reason" to r.reason,
            "note" to r.note.take(MAX_NOTE),
            "question" to r.question.take(MAX_QUESTION),
            "answer" to r.answer.take(MAX_ANSWER),
            "model" to r.model.take(80),
            "app" to version,
            "lang" to Locale.getDefault().language,
        ).joinToString("&") { (k, v) -> k + "=" + enc(v) }
    }

    // signed when sent (SiteApi), so a report kept offline gets a fresh time then
    private fun post(body: String): Boolean =
        SiteApi.post(context, PATH, "application/x-www-form-urlencoded", body.toByteArray())

    companion object {
        const val PATH = "/api/report"
        val REASONS = listOf("offensive", "harmful", "sexual", "false", "other")
        const val MAX_NOTE = 1000
        const val MAX_QUESTION = 4000
        const val MAX_ANSWER = 8000
    }
}
