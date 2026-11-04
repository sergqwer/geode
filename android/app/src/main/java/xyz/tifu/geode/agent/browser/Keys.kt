package xyz.tifu.geode.agent.browser

import kotlinx.coroutines.delay

/**
 * A key pressed on the page the way a keyboard does (CDP Input.dispatchKeyEvent: its down, its up), by the name a
 * page's code knows it by: Enter, Tab, Escape, Backspace, ArrowUp..., F1..F12, a character ("a", "7", " "), with
 * modifiers before it ("Control+a", "Shift+Tab").
 */
internal object Keys {
    private class K(val key: String, val code: String, val vk: Int, val text: String? = null)

    private val named: Map<String, K> = buildMap {
        fun k(names: List<String>, key: K) = names.forEach { put(it, key) }
        k(listOf("enter", "return"), K("Enter", "Enter", 13, "\r"))
        k(listOf("tab"), K("Tab", "Tab", 9))
        k(listOf("escape", "esc"), K("Escape", "Escape", 27))
        k(listOf("backspace"), K("Backspace", "Backspace", 8))
        k(listOf("delete", "del"), K("Delete", "Delete", 46))
        k(listOf("space", "spacebar", " "), K(" ", "Space", 32, " "))
        k(listOf("arrowup", "up"), K("ArrowUp", "ArrowUp", 38))
        k(listOf("arrowdown", "down"), K("ArrowDown", "ArrowDown", 40))
        k(listOf("arrowleft", "left"), K("ArrowLeft", "ArrowLeft", 37))
        k(listOf("arrowright", "right"), K("ArrowRight", "ArrowRight", 39))
        k(listOf("home"), K("Home", "Home", 36))
        k(listOf("end"), K("End", "End", 35))
        k(listOf("pageup"), K("PageUp", "PageUp", 33))
        k(listOf("pagedown"), K("PageDown", "PageDown", 34))
        k(listOf("insert"), K("Insert", "Insert", 45))
        k(listOf("shift"), K("Shift", "ShiftLeft", 16))
        k(listOf("control", "ctrl"), K("Control", "ControlLeft", 17))
        k(listOf("alt"), K("Alt", "AltLeft", 18))
        k(listOf("meta"), K("Meta", "MetaLeft", 91))
        for (i in 1..12) put("f$i", K("F$i", "F$i", 111 + i))
    }

    private val modifierBits = mapOf("alt" to 1, "control" to 2, "ctrl" to 2, "meta" to 4, "cmd" to 4, "shift" to 8)

    /** The key a page sees for a character: its code (KeyA, Digit7...) and virtual key. */
    private fun char(c: Char, shift: Boolean): K {
        val ch = if (shift) c.uppercaseChar() else c
        return when {
            c.isLetter() && c.code < 128 -> K(ch.toString(), "Key" + c.uppercaseChar(), c.uppercaseChar().code, ch.toString())
            c.isDigit() -> K(c.toString(), "Digit$c", c.code, c.toString())
            else -> K(ch.toString(), "", 0, ch.toString())
        }
    }

    /** True: [combo] names a key this knows. */
    fun known(combo: String): Boolean = runCatching { parse(combo) }.isSuccess

    private fun parse(combo: String): Pair<K, Int> {
        // "+" itself, and "Shift++"
        val parts = if (combo.length > 1 && combo.endsWith("++")) combo.dropLast(2).split('+') + "+"
        else if (combo == "+") listOf("+") else combo.split('+')
        var mods = 0
        for (m in parts.dropLast(1)) mods = mods or (modifierBits[m.trim().lowercase()]
            ?: throw IllegalArgumentException("unknown modifier \"$m\" (Control, Shift, Alt, Meta)"))
        val last = parts.last()
        val name = if (last.length == 1) last else last.trim()
        val k = named[name.lowercase()] ?: if (name.length == 1) char(name[0], mods and 8 != 0)
            else throw IllegalArgumentException("unknown key \"$name\"")
        return k to mods
    }

    /** [combo] pressed (held [holdMs] before it is let go: a game's key). */
    suspend fun press(c: Cdp, combo: String, holdMs: Long = 0) {
        val (k, mods) = parse(combo)
        // a key with a command modifier types nothing (Control+a selects, it does not write "a")
        val text = k.text.takeIf { mods and (1 or 2 or 4) == 0 }
        val base = linkedMapOf<String, Any?>("modifiers" to mods, "key" to k.key, "code" to k.code,
            "windowsVirtualKeyCode" to k.vk, "nativeVirtualKeyCode" to k.vk)
        c.call("Input.dispatchKeyEvent", LinkedHashMap(base).apply {
            put("type", if (text != null) "keyDown" else "rawKeyDown")
            if (text != null) {
                put("text", text)
                put("unmodifiedText", text)
            }
        }, 10_000)
        if (holdMs > 0) delay(holdMs)
        c.call("Input.dispatchKeyEvent", LinkedHashMap(base).apply { put("type", "keyUp") }, 10_000)
    }
}
