package xyz.tifu.geode.music

import org.json.JSONObject
import org.junit.Assert.fail
import org.junit.Test

class AbcScoreTest {
    @Test
    fun matchesPythonTranspositions() {
        val resource = javaClass.classLoader!!.getResource("abc_transpose.json")
            ?: error("Missing abc_transpose.json test resource")
        val golden = JSONObject(resource.readText())
        val scores = golden.getJSONObject("scores")
        val cases = golden.getJSONArray("cases")
        val failures = mutableListOf<String>()
        var passed = 0

        fun difference(label: String, expected: String, actual: String) {
            if (expected == actual) return
            val at = expected.zip(actual).indexOfFirst { (a, b) -> a != b }
                .let { if (it < 0) minOf(expected.length, actual.length) else it }
            fun excerpt(text: String) = text.substring(maxOf(0, at - 30), minOf(text.length, at + 70))
                .replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
            failures += "$label: first difference at $at (lengths ${expected.length}/${actual.length})" +
                "\n  expected: ${excerpt(expected)}\n  actual:   ${excerpt(actual)}"
        }

        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val name = case.getString("score")
            val semitones = case.getInt("semitones")
            val label = "case ${index + 1}: $name, $semitones"
            val failureCount = failures.size
            try {
                val moved = AbcScore.move(scores.getString(name), semitones)
                if (!case.getBoolean("ok")) failures += "$label: expected AbcError, got a moved score"
                else {
                    difference("$label text", case.getString("text"), moved.text)
                    difference("$label before", case.getString("before"), moved.before)
                    difference("$label after", case.getString("after"), moved.after)
                }
            } catch (error: AbcScore.AbcError) {
                if (case.getBoolean("ok")) failures += "$label: unexpected AbcError: ${error.message}"
            } catch (error: Exception) {
                failures += "$label: unexpected ${error.javaClass.simpleName}: ${error.message}"
            }
            if (failures.size == failureCount) passed++
        }

        try {
            difference("awkward up 2", golden.getString("awkward_up_2").trim(), AbcScore.move(scores.getString("awkward"), 2).text)
        } catch (error: Exception) {
            failures += "awkward up 2: ${error.javaClass.simpleName}: ${error.message}"
        }
        val key = AbcScore.key(scores.getString("awkward"))
        if (key != "D") failures += "awkward header key: expected D, got $key"
        if (failures.isNotEmpty()) fail(
            "$passed/${cases.length()} golden cases passed; ${failures.size} failures including extra checks:\n" +
                failures.take(8).joinToString("\n\n")
        )
    }
}
