/** Syllables drawn under named score sections, ported from YuE2-ComfyUI's underlay.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
import kotlin.math.abs
import kotlin.math.ln

object LyricUnderlay {
    enum class Kind { FIT, SPARE, CROWD }
    data class Part(val text: String,val word: Int,val closes: Boolean)
    data class Row(val line: Int,val text: String,val parts: List<Part>)
    data class Block(val label: String,val rows: MutableList<Row>)
    data class Line(val line: Int,val text: String,val first: Int,val stop: Int,val syllables: Int,val notes: Int,val kind: Kind)
    data class Drawn(val note: Int,val line: Int,val text: String)
    data class Laid(val bare: Boolean,val paired: Boolean,val lines: List<Line>,val notes: List<Drawn>,val spare: List<Int>,val sung: Int) {
        val counts get() = Kind.values().associateWith { kind -> lines.count { it.kind==kind } }
    }
    fun partsOf(line: String): List<Part> {
        val found=mutableListOf<Part>(); var waiting=""
        for ((index,word) in LyricSyllables.words(line).withIndex()) {
            val cut=LyricSyllables.pieces(word)
            if (cut.isEmpty()) { waiting=(waiting+" "+word).trim(); continue }
            for ((place,piece) in cut.withIndex()) {
                val text=if (place==0 && waiting.isNotEmpty()) "$waiting $piece".also { waiting="" } else piece
                found+=Part(text,index,place==cut.lastIndex)
            }
        }
        if (waiting.isNotEmpty()) {
            if (found.isNotEmpty()) found[found.lastIndex]=found.last().copy(text=found.last().text+" "+waiting)
            else found+=Part(waiting,0,true)
        }
        return found
    }
    fun blocksOf(lyrics: String): List<Block> {
        val found=mutableListOf<Block>(); var current: Block?=null
        for ((number,raw) in lyrics.lines().withIndex()) {
            val line=raw.trim(); if (line.isEmpty()) continue
            val tag=Regex("\\[\\s*([^\\]]*?)\\s*\\]").matchEntire(line)
            if (tag!=null) { current=Block(Phrasing.labelOf(tag.groupValues[1]),mutableListOf()); found+=current; continue }
            val parts=partsOf(line); if (parts.isEmpty()) continue
            if (current==null) { current=Block("verse",mutableListOf()); found+=current }
            current.rows+=Row(number,line,parts)
        }
        return found.filter { it.rows.isNotEmpty() }
    }
    fun byName(labels: List<String>,sections: List<Pair<String,Int>>): List<Int>? {
        val found=mutableListOf<Int>(); var at=0
        for (label in labels) {
            while (at<sections.size && sections[at].first!=label) at++
            if (at==sections.size) break
            found+=at++
        }
        if (found.size==labels.size) return found
        return sections.indices.filter { sections[it].second>0 }.takeIf { it.size==labels.size }
    }
    fun bySize(labels: List<String>,counts: List<Int>,sections: List<Pair<String,Int>>): List<Int>? {
        val blocks=labels.size; val size=sections.size
        val best=Array(blocks+1) { DoubleArray(size+1) { Double.POSITIVE_INFINITY } }
        data class Back(val block: Int,val at: Int,val took: Boolean)
        val back=Array(blocks+1) { arrayOfNulls<Back>(size+1) }; best[0][0]=0.0
        for (block in 0..blocks) for (at in 0 until size) {
            val here=best[block][at]; if (!here.isFinite()) continue
            val (name,notes)=sections[at]
            if (here+0.04*notes<best[block][at+1]) { best[block][at+1]=here+0.04*notes; back[block][at+1]=Back(block,at,false) }
            if (block<blocks && notes>0) {
                val value=here+abs(ln((notes+1.0)/(counts[block]+1.0)))+(if (name==labels[block]) 0.0 else 0.5)+0.001*at
                if (value<best[block+1][at+1]) { best[block+1][at+1]=value; back[block+1][at+1]=Back(block,at,true) }
            }
        }
        if (!best[blocks][size].isFinite()) return null
        val found=mutableListOf<Int>(); var block=blocks; var at=size
        while (block!=0 || at!=0) { val b=back[block][at]!!; block=b.block; at=b.at; if (b.took) found+=at }
        return found.asReversed()
    }
    fun cut(notes: List<AbcScore.Note>,counts: List<Int>,first: Int,stop: Int): List<Pair<Int,Int>> {
        val size=stop-first; val lines=counts.size
        if (size<=0) return List(lines) { first to first }
        val rest=DoubleArray(size+1)
        for (i in 1 until size) rest[i]=if (notes[first+i].start-(notes[first+i-1].start+notes[first+i-1].duration)>=Fraction(1,2)) 0.0 else 0.6
        val best=Array(lines+1) { DoubleArray(size+1) { Double.POSITIVE_INFINITY } }; val back=Array(lines+1) { IntArray(size+1) }
        best[0][0]=0.0
        for (line in 0 until lines) for (done in 0..size) {
            val here=best[line][done]; if (!here.isFinite()) continue
            val left=size-done
            if (left==0) { if (here+3<best[line+1][done]) { best[line+1][done]=here+3; back[line+1][done]=done }; continue }
            for (take in 1..minOf(40,left)) {
                val end=done+take; if (line==lines-1 && end!=size) continue
                val value=here+abs(ln(take.toDouble())-ln(maxOf(1,counts[line]).toDouble()))+rest[end]
                if (value<best[line+1][end]) { best[line+1][end]=value; back[line+1][end]=done }
            }
        }
        var end=if (best[lines][size].isFinite()) size else (0..size).minByOrNull { best[lines][it] }!!
        val spans=mutableListOf<Pair<Int,Int>>()
        for (line in lines downTo 1) { val begin=back[line][end]; spans+=first+begin to first+end; end=begin }
        return spans.asReversed()
    }
    private data class Section(val start: Fraction,val end: Fraction,val label: String,val b0: Int,val b1: Int)
    private data class Phrase(val start: Fraction,var end: Fraction,val onsets: MutableList<Fraction>)
    /** The section's sung opening follows grid.seam_at, including pickups in the preceding bar. */
    private fun opening(bar: Int,starts: List<Fraction>,sections: List<Section>,runs: List<Phrase>): Fraction {
        if (bar<=0 || bar>=starts.lastIndex) return starts[bar.coerceIn(0,starts.lastIndex)]
        val down=starts[bar]; val index=runs.indexOfFirst { it.end>down }
        if (index<0) return down
        val run=runs[index]
        if (run.start>=starts[bar-1]) return minOf(down,run.start)
        if (down !in run.onsets && index+1<runs.size && run.end<=starts[bar+1]) return minOf(down,runs[index+1].start)
        val section=sections.firstOrNull { bar>=it.b0 && bar<it.b1 }
        val low=section?.start ?: starts[0]; val high=section?.end ?: starts.last()
        val leads=runs.filter { it.start>=low && it.start<high }.mapNotNull {
            val line=starts.firstOrNull { s -> s>=it.start }
            when { line==it.start -> ZERO; line!=null && it.end>line -> line-it.start; else -> null }
        }
        if (leads.isEmpty()) return down
        val counted=leads.groupingBy { it }.eachCount(); val most=counted.values.max()
        val lead=counted.filterValues { it==most }.keys.min()
        return if (down-lead in run.onsets) down-lead else down
    }
    fun lay(abc: String,lyrics: String): Laid {
        val parsed=AbcScore.parse(abc); val vocal=parsed.voices.getValue("Vocal")
        val notes=vocal.notes.sortedWith(compareBy({ it.start },{ it.pitch })); val blocks=blocksOf(lyrics)
        val named=AbcScore.sections(abc)
        if (named.none { it.name.isNotBlank() }) return Laid(true,true,emptyList(),emptyList(),emptyList(),notes.size)
        if (blocks.isEmpty()) return Laid(false,true,emptyList(),emptyList(),emptyList(),notes.size)
        val starts=vocal.bars.map { it.start }+vocal.time
        val sections=named.mapIndexed { i,s ->
            val end=named.getOrNull(i+1)?.start ?: vocal.time
            Section(s.start,end,MidiParts.fold(s.name.lowercase(java.util.Locale.ROOT)),starts.indexOf(s.start),starts.indexOf(end))
        }
        val sizes=sections.map { s -> s.label to notes.count { it.start>=s.start && it.start<s.end } }
        val pairs=byName(blocks.map { it.label },sizes) ?: bySize(blocks.map { it.label },blocks.map { b -> b.rows.sumOf { it.parts.size } },sizes)
        val runs=mutableListOf<Phrase>()
        for (n in notes) {
            if (runs.isNotEmpty() && n.start<=runs.last().end) { runs.last().end=maxOf(runs.last().end,n.start+n.duration); runs.last().onsets+=n.start }
            else runs+=Phrase(n.start,n.start+n.duration,mutableListOf(n.start))
        }
        val rows=mutableListOf<Pair<Row,Pair<Int,Int>>>()
        if (pairs==null) {
            val all=blocks.flatMap { it.rows }; rows+=all.zip(cut(notes,all.map { it.parts.size },0,notes.size))
        } else for ((block,at) in blocks.zip(pairs)) {
            val s=sections[at]; val low=opening(s.b0,starts,sections,runs); val high=opening(s.b1,starts,sections,runs)
            val inside=notes.indices.filter { notes[it].start>=low && notes[it].start<high }
            rows+=block.rows.zip(cut(notes,block.rows.map { it.parts.size },inside.firstOrNull() ?: 0,inside.lastOrNull()?.plus(1) ?: 0))
        }
        val lines=mutableListOf<Line>(); val drawn=mutableListOf<Drawn>(); val spare=mutableListOf<Int>()
        for ((index,pair) in rows.sortedBy { it.first.line }.withIndex()) {
            val (row,span)=pair; val (low,high)=span; val held=high-low
            val kind=when { held==row.parts.size -> Kind.FIT; held>row.parts.size -> Kind.SPARE; else -> Kind.CROWD }
            lines+=Line(row.line,row.text,low,high,row.parts.size,held,kind)
            for (place in 0 until minOf(held,row.parts.size)) {
                val parts=if (place==held-1) row.parts.drop(place) else listOf(row.parts[place])
                val text=buildString { parts.forEachIndexed { i,p -> append(p.text); if (i<parts.lastIndex && p.closes) append(' ') }; if (!parts.last().closes) append('-') }
                drawn+=Drawn(low+place,index,text)
            }
            spare+=(low+row.parts.size until high)
        }
        return Laid(false,pairs!=null,lines,drawn.sortedWith(compareBy({ it.note },{ it.line },{ it.text })),spare,notes.size)
    }
}
