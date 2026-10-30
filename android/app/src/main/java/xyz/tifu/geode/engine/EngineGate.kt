package xyz.tifu.geode.engine

import android.os.SystemClock
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.CoroutineContext

/**
 * The engine answers one request at a time and holds one conversation; the app's chat, the local server and whatever
 * closes the model (a setting, a benchmark, a delete) take turns through this gate. Coming through it, each learns
 * whether the engine still holds its conversation: nobody else came through since.
 *
 * A conversation's save to the storage (StateCache) waits here too: its owner leaves it after an answer, and it runs
 * before anyone else takes the engine (the conversation would be gone), on [flush], or not at all when the owner
 * comes back first (its next answer's save covers it).
 *
 * Background work (the agent's shared start read ahead) comes through with [Background] in its context and gives the
 * engine up to anyone else at once: every other use calls [yieldBackground] as it comes, before it waits.
 */
class EngineGate {
    @PublishedApi
    internal val mutex = Mutex()

    @PublishedApi
    internal var last: Any? = null

    @PublishedApi
    internal val current = MutableStateFlow<Any?>(null)

    @PublishedApi
    internal var pending: Pair<Any, suspend () -> Unit>? = null

    /** Who has the engine now (null: nobody). */
    val holder: StateFlow<Any?> = current.asStateFlow()

    /** The background work running (it stops and lets the engine go when called); null: none. */
    @Volatile
    var yieldBackground: (() -> Unit)? = null

    /** A [Background] use has the engine now. */
    @Volatile
    var backgroundIn = false

    /** When the engine was last let go (SystemClock.elapsedRealtime; 0: not yet). */
    @Volatile
    var lastFree = 0L

    /** [block] with the engine to [who] alone (waiting for its turn); kept: the engine still holds [who]'s conversation. */
    suspend inline fun <T> use(who: Any, block: (kept: Boolean) -> T): T {
        val background = currentCoroutineContext()[Background] != null
        // background work gives the engine up to anyone else's
        if (!background) yieldBackground?.invoke()
        return mutex.withLock {
            // someone else's conversation is about to go: its waiting save first
            pending?.let { (owner, save) -> if (owner !== who) runPending(owner, save) }
            val kept = last === who
            last = who
            current.value = who
            backgroundIn = background
            try {
                block(kept)
            } finally {
                current.value = null
                backgroundIn = false
                lastFree = SystemClock.elapsedRealtime()
            }
        }
    }

    /** From inside [owner]'s [use]: the save its conversation waits for (replacing an earlier one). */
    fun deferSave(owner: Any, save: suspend () -> Unit) {
        pending = owner to save
    }

    /** [owner]'s waiting save run now, if the engine still holds its conversation (an idle while, the app leaving). */
    suspend fun flush(owner: Any? = null) = mutex.withLock {
        pending?.let { (o, save) -> if (owner == null || o === owner) runPending(o, save) }
    }

    @PublishedApi
    internal suspend fun runPending(owner: Any, save: suspend () -> Unit) {
        pending = null
        if (last === owner) runCatching { save() }
    }

    /** Closes the model once nobody is answering (a setting it opens with changed, or its files are to go). */
    suspend fun close(engine: EngineClient) = use(this) { engine.close() }

    /**
     * The user let the model go (the Models screen, the chat's details): nothing in the background opens it again (the
     * agent's shared start read ahead waits) until a use of the user's own does - a message, a request to the local
     * server, the Models screen's Load (ApiEngine.ensureOpen clears it).
     */
    val parked = MutableStateFlow(false)

    /** The model out of memory once nobody is answering (an answer running ends first), and kept out ([parked]). */
    suspend fun unload(engine: EngineClient) = use(this) {
        engine.close()
        parked.value = true
    }

    /** In a coroutine's context: its uses of the engine are background work (they yield to everyone else's). */
    object Background : CoroutineContext.Element, CoroutineContext.Key<Background> {
        override val key: CoroutineContext.Key<*> get() = this
    }
}
