package xyz.tifu.geode.models

import org.json.JSONObject
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Where a model comes from; saved in a partial transfer so it can resume. */
sealed interface TransferSource {
    data class Hf(val repo: String, val revision: String, val packPath: String) : TransferSource
    /** A folder picked with the system picker; move = delete each original once its copy is checked. */
    data class Folder(val uri: String, val move: Boolean) : TransferSource

    fun write(f: File) {
        val o = when (this) {
            is Hf -> JSONObject().put("kind", "hf").put("repo", repo).put("revision", revision).put("pack", packPath)
            is Folder -> JSONObject().put("kind", "folder").put("uri", uri).put("move", move)
        }
        f.writeText(o.toString())
    }

    companion object {
        fun read(f: File): TransferSource? = runCatching {
            val o = JSONObject(f.readText())
            when (o.getString("kind")) {
                "hf" -> Hf(o.getString("repo"), o.getString("revision"), o.getString("pack"))
                "folder" -> Folder(o.getString("uri"), o.getBoolean("move"))
                else -> null
            }
        }.getOrNull()
    }
}

enum class TransferPhase { PREPARE, DOWNLOAD, COPY, VERIFY, DONE, FAILED, CANCELLED }

data class TransferState(
    val id: String,
    val title: String,
    val phase: TransferPhase,
    val done: Long = 0,
    val total: Long = 0,
    val bytesPerSecond: Long = 0,
    val error: String? = null,
) {
    val active: Boolean get() = phase != TransferPhase.DONE && phase != TransferPhase.FAILED &&
        phase != TransferPhase.CANCELLED
    val fraction: Float get() = if (total > 0) (done.toDouble() / total).toFloat().coerceIn(0f, 1f) else 0f
}

/** The one transfer running (TransferService and the UI share the main process). */
object Transfers {
    private val _state = MutableStateFlow<TransferState?>(null)
    val state: StateFlow<TransferState?> = _state.asStateFlow()

    internal fun set(s: TransferState?) {
        _state.value = s
    }

    /** Hides a finished transfer's card. */
    fun dismiss() {
        if (_state.value?.active == false) _state.value = null
    }
}
