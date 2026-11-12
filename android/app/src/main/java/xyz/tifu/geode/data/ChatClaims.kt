package xyz.tifu.geode.data

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Which side writes a chat's next answer - the phone's chat or the web chat. Claimed when a question is accepted
 * (before its turn waits for another one to end), given back when its answer is saved: the two never answer one chat
 * at once. Each saves the whole chat it knows, so the later save would drop the other's question and answer; the
 * agent's own turn ([xyz.tifu.geode.agent.Agent.turn]) starts only once the turn before it ends - too late to tell.
 */
object ChatClaims {
    enum class Side { PHONE, WEB }

    /** A side's hold on a chat ([release] gives back only this one: a newer claim of the same side replaces it). */
    class Claim internal constructor(val chatId: String, val side: Side)

    private val _held = MutableStateFlow<Map<String, Claim>>(emptyMap())

    /** The chats claimed now, by id. */
    val held: StateFlow<Map<String, Claim>> = _held.asStateFlow()

    /**
     * [chatId] for [side]; null while the other side holds it. With [takeover] the same side's claim replaces its own
     * older one (the phone's: a stopped answer whose turn is still ending - the next question may go, the agent runs
     * them in order); without it, null while anyone holds it (the web's: two requests never answer one chat).
     */
    fun claim(chatId: String, side: Side, takeover: Boolean = true): Claim? = synchronized(this) {
        val cur = _held.value[chatId]
        if (cur != null && (cur.side != side || !takeover)) return null
        val c = Claim(chatId, side)
        _held.value = _held.value + (chatId to c)
        c
    }

    /** [c] given back (nothing when a newer claim replaced it). */
    fun release(c: Claim?) {
        if (c == null) return
        synchronized(this) {
            if (_held.value[c.chatId] === c) _held.value = _held.value - c.chatId
        }
    }

    /** Who holds [chatId] now (null: nobody). */
    fun holder(chatId: String): Side? = _held.value[chatId]?.side
}
