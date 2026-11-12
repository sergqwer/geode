/** The entry point for YuE2-ComfyUI's phrasing.py, with underlay.py's note drawing in LyricUnderlay (Apache-2.0). */
package xyz.tifu.geode.music.midi

object Underlay {
    /** maxSeconds rounds phrasing.py's decimal ceiling upward to a whole second; notices retain the decimal value. */
    data class Laid(val abc: String, val maxSeconds: Int, val notices: List<Notice>)
    /** Null when sections are already named, the score cannot be parsed, or there are no sung notes or words. */
    fun lay(abc: String, lyrics: String): Laid? = Phrasing.lay(abc,lyrics)
}
