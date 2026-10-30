package xyz.tifu.geode.util

/**
 * A JSON number as it was written: Python's json.dumps writes the usual ones (integers, plain decimals) back the same,
 * which the chat template's tojson needs (a tool's schema rendered into the prompt token for token).
 */
class JsonNumber(val text: String) : Number() {
    override fun toDouble(): Double = text.toDouble()
    override fun toFloat(): Float = text.toFloat()
    override fun toLong(): Long = text.toLongOrNull() ?: toDouble().toLong()
    override fun toInt(): Int = toLong().toInt()
    override fun toShort(): Short = toLong().toShort()
    override fun toByte(): Byte = toLong().toByte()
    override fun equals(other: Any?): Boolean = other is JsonNumber && other.text == text
    override fun hashCode(): Int = text.hashCode()
    override fun toString(): String = text
}

/**
 * JSON for the local server's API: objects are LinkedHashMaps (their keys in the order written: org.json's JVM
 * implementation loses it), arrays Lists, numbers [JsonNumber], strings, booleans, null. [write] is compact (the
 * API's answers), [dumps] is Python's json.dumps(v, ensure_ascii=False) - the chat template's tojson.
 */
object Json {
    class ParseException(message: String) : IllegalArgumentException(message)

    fun parse(s: String): Any? {
        val p = Parser(s)
        p.ws()
        val v = p.value()
        p.ws()
        if (p.i != s.length) throw ParseException("unexpected text at ${p.i}")
        return v
    }

    fun write(v: Any?): String = StringBuilder().also { out(it, v, ",", ":") }.toString()

    fun dumps(v: Any?): String = StringBuilder().also { out(it, v, ", ", ": ") }.toString()

    private fun out(sb: StringBuilder, v: Any?, comma: String, colon: String) {
        when (v) {
            null -> sb.append("null")
            is String -> str(sb, v)
            is Boolean -> sb.append(if (v) "true" else "false")
            is JsonNumber -> sb.append(v.text)
            is Int, is Long, is Short, is Byte -> sb.append(v.toString())
            is Double -> sb.append(if (v.isFinite()) num(v) else "null")
            is Float -> sb.append(if (v.isFinite()) num(v.toDouble()) else "null")
            is Map<*, *> -> {
                sb.append('{')
                var first = true
                for ((k, x) in v) {
                    if (!first) sb.append(comma)
                    first = false
                    str(sb, k.toString())
                    sb.append(colon)
                    out(sb, x, comma, colon)
                }
                sb.append('}')
            }
            is List<*> -> {
                sb.append('[')
                v.forEachIndexed { i, x ->
                    if (i > 0) sb.append(comma)
                    out(sb, x, comma, colon)
                }
                sb.append(']')
            }
            else -> str(sb, v.toString())
        }
    }

    /** A double as Python writes a float: a whole one keeps its ".0". */
    private fun num(d: Double): String =
        if (d == Math.rint(d) && kotlin.math.abs(d) < 1e16) String.format(java.util.Locale.ROOT, "%.1f", d) else d.toString()

    /** Python's escaping with ensure_ascii=False: the quote, the backslash and the control characters only. */
    private fun str(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                '\b' -> sb.append("\\b")
                '\u000c' -> sb.append("\\f")
                else -> if (c < ' ') sb.append(String.format(java.util.Locale.ROOT, "\\u%04x", c.code)) else sb.append(c)
            }
        }
        sb.append('"')
    }

    private class Parser(val s: String) {
        var i = 0

        fun ws() {
            while (i < s.length && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++
        }

        fun value(): Any? {
            if (i >= s.length) throw ParseException("unexpected end")
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> string()
                't' -> word("true", true)
                'f' -> word("false", false)
                'n' -> word("null", null)
                else -> if (c == '-' || c in '0'..'9') number() else throw ParseException("unexpected '$c' at $i")
            }
        }

        private fun word(w: String, v: Any?): Any? {
            if (!s.startsWith(w, i)) throw ParseException("unexpected text at $i")
            i += w.length
            return v
        }

        private fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++
            ws()
            if (i < s.length && s[i] == '}') {
                i++
                return m
            }
            while (true) {
                ws()
                if (i >= s.length || s[i] != '"') throw ParseException("a key expected at $i")
                val k = string()
                ws()
                if (i >= s.length || s[i] != ':') throw ParseException("':' expected at $i")
                i++
                ws()
                m[k] = value()
                ws()
                if (i >= s.length) throw ParseException("unexpected end")
                when (s[i++]) {
                    ',' -> continue
                    '}' -> return m
                    else -> throw ParseException("',' or '}' expected at ${i - 1}")
                }
            }
        }

        private fun arr(): List<Any?> {
            val a = ArrayList<Any?>()
            i++
            ws()
            if (i < s.length && s[i] == ']') {
                i++
                return a
            }
            while (true) {
                ws()
                a += value()
                ws()
                if (i >= s.length) throw ParseException("unexpected end")
                when (s[i++]) {
                    ',' -> continue
                    ']' -> return a
                    else -> throw ParseException("',' or ']' expected at ${i - 1}")
                }
            }
        }

        private fun number(): JsonNumber {
            val start = i
            if (s[i] == '-') i++
            while (i < s.length && (s[i].isDigit() || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' ||
                    s[i] == '-')) i++
            val t = s.substring(start, i)
            if (t.toDoubleOrNull() == null) throw ParseException("a bad number at $start")
            return JsonNumber(t)
        }

        private fun string(): String {
            i++
            // the common case, no escapes: one substring (an image's base64 is megabytes)
            val plain = s.indexOf('"', i)
            if (plain < 0) throw ParseException("unterminated string")
            if (s.indexOf('\\', i).let { it < 0 || it > plain }) {
                return s.substring(i, plain).also { i = plain + 1 }
            }
            val sb = StringBuilder()
            while (true) {
                if (i >= s.length) throw ParseException("unterminated string")
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        if (i >= s.length) throw ParseException("unterminated string")
                        when (val e = s[i++]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000c')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                if (i + 4 > s.length) throw ParseException("a bad escape at $i")
                                sb.append(s.substring(i, i + 4).toIntOrNull(16)?.toChar()
                                    ?: throw ParseException("a bad escape at $i"))
                                i += 4
                            }
                            else -> throw ParseException("a bad escape '\\$e' at $i")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }
    }
}

/** A JSON object's field as an object, a list, a string, a number, a boolean (null when absent or of another type). */
@Suppress("UNCHECKED_CAST")
fun Map<String, Any?>.obj(k: String): Map<String, Any?>? = this[k] as? Map<String, Any?>
fun Map<String, Any?>.list(k: String): List<Any?>? = this[k] as? List<Any?>
fun Map<String, Any?>.str(k: String): String? = this[k] as? String
fun Map<String, Any?>.num(k: String): Number? = this[k] as? Number
fun Map<String, Any?>.bool(k: String): Boolean? = this[k] as? Boolean
