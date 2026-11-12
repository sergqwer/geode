package xyz.tifu.geode.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ChatClaimsTest {
    @Test
    fun oneChatIsAnsweredByOneSide() {
        val id = "claims-test-a"
        val phone = ChatClaims.claim(id, ChatClaims.Side.PHONE)
        assertNotNull(phone)
        assertEquals(ChatClaims.Side.PHONE, ChatClaims.holder(id))
        // the web cannot take a chat the phone answers, nor the phone one the web answers
        assertNull(ChatClaims.claim(id, ChatClaims.Side.WEB, takeover = false))
        assertNull(ChatClaims.claim(id, ChatClaims.Side.WEB))
        ChatClaims.release(phone)
        assertNull(ChatClaims.holder(id))
        val web = ChatClaims.claim(id, ChatClaims.Side.WEB, takeover = false)
        assertNotNull(web)
        assertNull(ChatClaims.claim(id, ChatClaims.Side.PHONE))
        // two web requests never answer one chat
        assertNull(ChatClaims.claim(id, ChatClaims.Side.WEB, takeover = false))
        ChatClaims.release(web)
        assertNull(ChatClaims.holder(id))
    }

    @Test
    fun theSameSideTakesOverAndAnOldReleaseKeepsTheNewClaim() {
        val id = "claims-test-b"
        val first = ChatClaims.claim(id, ChatClaims.Side.PHONE)
        // a stopped answer whose turn still ends: the next question takes the chat over
        val second = ChatClaims.claim(id, ChatClaims.Side.PHONE)
        assertNotNull(second)
        ChatClaims.release(first)   // (the old turn ends: the new claim stays)
        assertEquals(ChatClaims.Side.PHONE, ChatClaims.holder(id))
        ChatClaims.release(second)
        assertNull(ChatClaims.holder(id))
        ChatClaims.release(null)   // (nothing held: nothing happens)
    }
}
