package xyz.tifu.geode.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import xyz.tifu.geode.ui.chat.ChatMessage

/** What a chat keeps across the agent's compactions: the spans of the last [Compactions.KEEP] before the newest. */
class CompactionsTest {
    private var id = 0L

    /** [n] turns (a question and its answer); the first answer compacted when [compacted]. */
    private fun span(n: Int, compacted: Boolean): List<ChatMessage> = (0 until n).flatMap { t ->
        listOf(ChatMessage(++id, user = true, text = "q$id"),
            ChatMessage(++id, user = false, text = "a$id", compacted = compacted && t == 0))
    }

    @Test
    fun keepsEverythingUpToTwoCompactions() {
        val m = span(5, false) + span(5, true) + span(5, true)
        assertEquals(m, Compactions.prune(m))
        assertFalse(Compactions.pruned(m))
    }

    @Test
    fun theThirdCompactionDropsTheFirstSpan() {
        // 10 messages, compaction, 10 messages, compaction, 10 messages, compaction: only now do the first 10 go
        val first = span(5, false)
        val second = span(5, true)
        val third = span(5, true)
        val fourth = span(1, true)
        val m = first + second + third + fourth
        val kept = Compactions.prune(m)
        assertEquals(second + third + fourth, kept)
        assertTrue(Compactions.pruned(kept))
        // the next compaction: the second span goes too
        val fifth = span(2, true)
        assertEquals(third + fourth + fifth, Compactions.prune(kept + fifth))
    }

    @Test
    fun aCompactionBeforeItsQuestionCutsAtTheQuestion() {
        val m = span(3, false) + span(1, true) + span(1, true) + span(1, true)
        val kept = Compactions.prune(m)
        // the kept history starts with the question whose answer was compacted, not the answer alone
        assertTrue(kept.first().user)
        assertEquals(6, kept.size)
    }

    @Test
    fun aChatWithoutCompactionsStaysWhole() {
        val m = span(50, false)
        assertEquals(m, Compactions.prune(m))
    }
}
