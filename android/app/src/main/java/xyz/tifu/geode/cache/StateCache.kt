package xyz.tifu.geode.cache

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import xyz.tifu.geode.engine.EngineClient
import xyz.tifu.geode.util.Json
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

/**
 * The engine's conversations kept on the phone's storage (files/kvcache): what the model read and wrote - its KV cache
 * and its recurrent states, exactly - so a conversation goes on where it stopped instead of being read again (a long
 * one takes minutes on a phone; reading it back takes a fraction of a second). Three kinds:
 *
 *  - prefix-<hash>: the start every conversation of a kind shares (a system prompt and its tools: the agent's is ~3k
 *    tokens), read once and loaded under every new one;
 *  - api-<id>: a conversation of the local API (the agent's, a LAN client's), its messages kept with it to recognise
 *    the next request that goes on from it;
 *  - chat-<id>: a chat of the app's own.
 *
 * Each is a state file (the engine's, geode_state_save) and a meta file (JSON: what the state is of, its lineage - the
 * conversation it is a state of, so a later state can be written as an append - its position, when it was last used).
 * Saved only between generations, through the engine's gate.
 */
class StateCache(context: Context) {
    val dir = File(context.filesDir, "kvcache").apply { mkdirs() }

    /** The state and meta files of [key]. */
    fun stateFile(key: String) = File(dir, "$key.state")
    private fun metaFile(key: String) = File(dir, "$key.json")

    @Suppress("UNCHECKED_CAST")
    fun meta(key: String): Map<String, Any?>? =
        runCatching { Json.parse(metaFile(key).readText()) as? Map<String, Any?> }.getOrNull()
            ?.takeIf { stateFile(key).isFile }

    /** Every entry of a kind ("api-", "chat-", "prefix-") with its meta. */
    fun entries(prefix: String): List<Pair<String, Map<String, Any?>>> =
        (dir.listFiles() ?: emptyArray())
            .filter { it.name.startsWith(prefix) && it.name.endsWith(".json") }
            .mapNotNull { f -> f.name.removeSuffix(".json").let { k -> meta(k)?.let { k to it } } }

    /**
     * The engine's conversation (it holds [lineage]'s, at its current position) saved as [key] with [meta]: only what
     * is new since the last save when the file holds an earlier state of the same lineage. The position, or -1.
     */
    suspend fun save(engine: EngineClient, key: String, lineage: String, meta: Map<String, Any?>): Int {
        val old = meta(key)
        val append = old != null && old["lineage"] == lineage
        val pos = try {
            engine.saveState(stateFile(key).path, append)
        } catch (e: xyz.tifu.geode.engine.EngineStateException) {
            // nothing written (no model open - its process died and came back since the answer -, or images not
            // taken yet): what the entry holds is still whole, and stays
            if (e.code == xyz.tifu.geode.engine.GeodeNative.E_ARG) return -1
            -1
        } catch (e: Exception) {
            -1
        }
        // a write that failed may have left the file half appended: the entry goes
        if (pos < 0) {
            delete(key)
            return -1
        }
        writeMeta(key, meta + mapOf("lineage" to lineage, "position" to pos, "used" to System.currentTimeMillis()))
        return pos
    }

    /** [key]'s state into the engine (its conversation replaced); the position, or -1 (the entry goes: unusable). */
    suspend fun load(engine: EngineClient, key: String): Int {
        val m = meta(key) ?: return -1
        val pos = runCatching { engine.loadState(stateFile(key).path) }.getOrDefault(-1)
        if (pos < 0 || (m["position"] as? Number)?.toInt() != pos) {
            delete(key)
            return -1
        }
        writeMeta(key, m + ("used" to System.currentTimeMillis()))
        return pos
    }

    fun delete(key: String) {
        stateFile(key).delete()
        metaFile(key).delete()
    }

    /** Every saved conversation gone (the setting's "clear"), or those of a kind. */
    fun clear(prefix: String = "") {
        for (f in dir.listFiles() ?: emptyArray()) if (f.name.startsWith(prefix)) f.delete()
    }

    /** The bytes the saved conversations take. */
    fun bytes(): Long = (dir.listFiles() ?: emptyArray()).sumOf { it.length() }

    /**
     * Within [maxBytes]: the least recently used go first, the shared starts (prefixes) last - each saves the most
     * reading. [keep]: an entry not to delete (the one just saved).
     */
    suspend fun trim(maxBytes: Long, keep: String? = null) = withContext(Dispatchers.IO) {
        val all = (dir.listFiles() ?: emptyArray()).filter { it.name.endsWith(".state") }
            .map { it.name.removeSuffix(".state") }
            .filter { it != keep }
            .map { k -> Triple(k, stateFile(k).length() + metaFile(k).length(), (meta(k)?.get("used") as? Number)?.toLong() ?: 0L) }
            .sortedWith(compareBy({ it.first.startsWith("prefix-") }, { it.third }))
        var total = bytes()
        for ((k, size, _) in all) {
            if (total <= maxBytes) break
            delete(k)
            total -= size
        }
    }

    private fun writeMeta(key: String, meta: Map<String, Any?>) {
        val tmp = File(dir, "$key.json.tmp")
        FileOutputStream(tmp).use {
            it.write(Json.write(meta).toByteArray())
            it.fd.sync()
        }
        tmp.renameTo(metaFile(key))
    }

    companion object {
        /** A lineage: one conversation as the engine reads and writes it, from its start (a reset or a load). */
        fun lineage(): String = UUID.randomUUID().toString()

        /** An id for a new API conversation's entry. */
        fun newApiKey(): String = "api-" + UUID.randomUUID().toString().replace("-", "").take(16)

        /** A shared start's key: the model, how the engine is opened (its KV format), and the tokens. */
        fun prefixKey(model: String, options: String, ids: IntArray, n: Int): String {
            val d = MessageDigest.getInstance("SHA-256")
            d.update(model.toByteArray())
            d.update(0)
            d.update(options.toByteArray())
            d.update(0)
            val b = java.nio.ByteBuffer.allocate(n * 4)
            for (i in 0 until n) b.putInt(ids[i])
            d.update(b.array())
            return "prefix-" + d.digest().take(10).joinToString("") { "%02x".format(it) }
        }
    }
}
