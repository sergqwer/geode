/** Word pieces required by underlay.py, from YuE2-ComfyUI's syllables.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import java.util.Locale

object LyricSyllables {
    private const val latin = "aeiouyæøœ"
    private val spelled = """may-be be-yond ne-on vid-e-o qui-et qui-et-ly dia-mond po-em po-et po-et-ry i-de-a cre-ate cre-a-ted
        cre-a-tion crea-ture re-al-i-ty so-ci-e-ty va-ri-e-ty an-xi-e-ty sci-ence rhy-thm clothes a-ny-one
        some-one eve-ry-one real real-ly a-re-a o-a-sis cha-os tri-al gi-ant cli-ent fu-el cru-el jew-el du-et
        ru-in flu-id ge-nius au-di-o stu-di-o ro-de-o ste-re-o mu-se-um the-a-ter the-a-tre hal-le-lu-jah
        na-ive he-ro he-roes e-cho e-choes prayer fire hour our light-ning ar-gue eve-ry eve-ry-thing
        eve-ry-bo-dy eve-ry-where eve-ry-day eve-ning eve-nings diffe-rent diffe-rence diffe-rent-ly seve-ral
        favo-rite favou-rite came-ra choco-late inte-rest inte-rest-ing busi-ness vege-ta-ble comfor-ta-ble
        tempe-ra-ture lone-li-ness despe-rate despe-rate-ly""".trim().split(Regex("\\s+")).associateBy { it.replace("-","") }
    private val suffixes = "fully ly ful ments ment less ness times time things thing where body lands land work days day side self like some how what".split(' ')
    private val onsets = "th sh ch ph wh bl br cl cr dr fl fr gl gr pl pr tr".split(' ').toSet()
    private val hushed = listOf("s","x","z","ce","se","ge","ze","ch","sh")
    private data class Run(val start: Int,val stop: Int)
    private fun runs(text: String,vowels: String): MutableList<Run> {
        val found=mutableListOf<Run>(); var opened: Int?=null
        for ((i,c) in text.withIndex()) if (c in vowels) { if (opened==null) opened=i } else if (opened!=null) { found+=Run(opened,i); opened=null }
        if (opened!=null) found+=Run(opened,text.length)
        return found
    }
    private fun slice(text: String,cuts: List<Int>): List<String> = (listOf(0)+cuts.filter { it>0 && it<text.length }.distinct().sorted()+text.length).zipWithNext().map { (a,b) -> text.substring(a,b) }
    fun words(line: String)=Phrasing.word.findAll(line).map { it.value }.toList()
    fun pieces(text: String): List<String> {
        if (text.isEmpty()) return emptyList()
        var lower=text.lowercase(Locale.ROOT).replace('’','\'')
        if (lower.length!=text.length) lower=text.map { it.lowercase(Locale.ROOT).first() }.joinToString("").replace('’','\'')
        if (lower.any { it in '\u0400'..'\u04ff' }) {
            val nuclei=lower.indices.filter { lower[it] in Phrasing.cyrillicVowels }
            if (nuclei.isEmpty()) return if (lower.count { it !in "ьъ'" }<=2) emptyList() else listOf(text)
            return slice(text,nuclei.zipWithNext().map { (a,b) ->
                val low=a+1; val sounded=lower.substring(low,b).count { it !in "ьъ'" }
                if (sounded<2) low else { var cut=low+1; while (cut<b && lower[cut] in "ьъ'") cut++; cut }
            })
        }
        if (text.any { Phrasing.wide(it) }) {
            val found=mutableListOf<String>(); var held=""
            for (c in text) if (Phrasing.wide(c)) { found+=held+c; held="" } else if (found.isNotEmpty()) found[found.lastIndex]=found.last()+c else held+=c
            return found.ifEmpty { listOf(text) }
        }
        var stem=lower.length; val sung=mutableListOf<String>()
        repeat(2) {
            val tail=listOf("n't","'ll","'ve","'re","'s","'d","'m").firstOrNull { lower.take(stem).endsWith(it) && stem>it.length }
            if (tail!=null) { stem-=tail.length; sung.add(0,tail) }
        }
        var plain=lower.take(stem).map { Phrasing.plain(it.toString()).firstOrNull() ?: it }.joinToString("")
        if (plain.length!=stem) plain=lower.take(stem)
        val greek=plain.any { it in '\u0370'..'\u03ff' }
        val groups=runs(plain,if (greek) "αεηιουω" else latin)
        val cuts=if (greek) groups.zipWithNext().map { (a,b) -> if (b.start-a.stop<2) a.stop else b.start-1 }.toMutableList()
        else latinCuts(plain,lower.take(stem))
        var at=stem
        for (tail in sung) {
            val before=if (at==stem) plain.lastOrNull() else null
            var closed=before!=null && before !in "aeiouyw"
            if (tail=="n't") closed=(closed || plain.endsWith('e')) && !plain.endsWith("re")
            else if (tail !in listOf("'ll","'ve","'re","'d")) closed=false
            if (closed && groups.isNotEmpty()) cuts+=at
            at+=tail.length
        }
        return slice(text,cuts)
    }
    private fun latinCuts(plain: String,lower: String): MutableList<Int> {
        val size=plain.length
        val explicit=spelled[plain] ?: if (plain.endsWith('s') && spelled.containsKey(plain.dropLast(1)) && hushed.none { plain.dropLast(1).endsWith(it) }) spelled.getValue(plain.dropLast(1))+"s" else null
        if (explicit!=null) { var at=0; return explicit.split('-').dropLast(1).map { at+=it.length; at }.toMutableList() }
        val groups=runs(plain,latin)
        fun part(index: Int) {
            val i=groups.indexOfFirst { it.start<index && index<it.stop }
            if (i>=0) { val old=groups.removeAt(i); groups.addAll(i,listOf(Run(old.start,index),Run(index,old.stop))) }
        }
        for (i in 1 until size) if (lower[i] in "ïëüÿ" && plain[i-1] in latin) part(i)
        if (plain.endsWith("ing") && size>3 && plain[size-4] in "aeiouy") part(size-3)
        Phrasing.soundedI.findAll(plain).forEach { part(it.range.last+1) }
        val silent=mutableSetOf<Int>()
        for (suffix in suffixes) {
            val stem=plain.take(maxOf(0,size-suffix.length))
            if (plain.endsWith(suffix) && stem.length>=3 && stem.endsWith('e') && stem[stem.length-2] !in "aeiouy" && stem.dropLast(2).any { it in "aeiouy" } && !stem.endsWith("le") && Run(stem.length-1,stem.length) in groups) {
                silent+=stem.length-1; break
            }
        }
        var sounded=groups.filter { it.start !in silent }; var syllabic: Run?=null
        if (sounded.size>1 && size>2) {
            val last=sounded.last()
            if (plain.endsWith('e') && lower.last() !in "éèêë") {
                if (plain.endsWith("le") && plain[size-3] !in "aeiouy") syllabic=last
                else if (last==Run(size-1,size) || plain.endsWith("gue") || plain.endsWith("que")) silent+=(last.start until last.stop)
            } else if (plain.endsWith("es") && size>3 && plain[size-3] !in "aeiouysxzcgh") {
                if (plain.endsWith("les") && size>4 && plain[size-4] !in "aeiouy") syllabic=last else silent+=size-2
            } else if (plain.endsWith("ed") && size>3 && plain[size-3] !in "aeiouytd") silent+=size-2
        }
        sounded=groups.filter { it.start !in silent }
        return sounded.zipWithNext().map { (left,right) ->
            val low=left.stop; val high=right.start; val quiet=(low until high).filter { it in silent }
            when {
                high==low -> high
                quiet.isNotEmpty() -> quiet.last()+1
                high-low==1 -> low
                plain.substring(high-2,high)=="gh" || plain.substring(high-2,high)=="ng" && plain.startsWith("ing",high) -> high
                plain.substring(high-2,high) in onsets || right==syllabic && plain[high-1]=='l' || high-low>2 && plain[high-2]=='s' && plain[high-1] in "ptckw" -> high-2
                else -> high-1
            }
        }.toMutableList()
    }
}
