package xyz.tifu.geode.data

import xyz.tifu.geode.ui.chat.ChatMessage

/**
 * A chat's history across the agent's compactions (ChatMessage.compacted: the agent condensed its context before that
 * answer). The phone keeps the messages of the last [KEEP] spans between compactions, to scroll back through; what
 * came before them goes - the agent went on from its summary of them long since, and a chat's file, read and written
 * whole at every answer, stops growing without end.
 */
object Compactions {
    /**
     * The spans kept before the newest compaction: 10 messages, compaction, 10 messages, compaction, 10 messages,
     * compaction - only now do the first 10 go.
     */
    const val KEEP = 2

    /** Where each compaction's turn starts: the question before a compacted answer (the answer, without one). */
    fun boundaries(messages: List<ChatMessage>): List<Int> =
        messages.indices.filter { !messages[it].user && messages[it].compacted }
            .map { i -> if (i > 0 && messages[i - 1].user) i - 1 else i }

    /**
     * The messages a chat keeps: everything from the turn of the compaction [keep] before the newest one, the newest's
     * own turn included; all of them while there are no more than [keep] compactions.
     */
    fun prune(messages: List<ChatMessage>, keep: Int = KEEP): List<ChatMessage> {
        val b = boundaries(messages)
        if (b.size <= keep) return messages
        val cut = b[b.size - 1 - keep]
        return if (cut <= 0) messages else messages.subList(cut, messages.size).toList()
    }

    /** Whether older messages went: the chat starts at a compaction's turn (no chat compacts in its first one). */
    fun pruned(messages: List<ChatMessage>): Boolean = boundaries(messages).firstOrNull() == 0
}
