/** Coded messages from YuE2-ComfyUI's midi modules and phrasing.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

/** Arguments are unlocalized values, in the order documented on each code. */
data class Notice(val code: Code, val args: List<String> = emptyList())

enum class Code {
    /**
     * The file's tempo moves between {low:.0f} and {high:.0f} BPM. A score keeps one tempo, so it is written at
     * the average, {bpm:.0f} BPM, and the song will not speed up or slow down where the file does.
     * Arguments: low, high, bpm.
     */
    TEMPO_AVERAGED,
    /**
     * The notes of the file do not sit on a grid of sixteenths or thirty-seconds -- it was probably played in
     * by hand -- so they were rounded to the nearest sixteenth.
     */
    PLAYED_IN,
    /**
     * The {line} line (track {number}) sat {where} for YuE2's scores, so it was moved {way} {count}.
     * Arguments: line (voice/instrument), number, signed semitones (positive means low/up).
     */
    OCTAVE_MOVED,
    /**
     * Track {number} strikes {count} notes together with a higher one. A line of the score holds one note at a
     * time, so the top note of each chord was kept. Arguments: number, count.
     */
    TOP_NOTES_KEPT,
    /**
     * The chord symbols were guessed from what the file's parts play together: a harmony to follow, not a
     * transcription of it.
     */
    CHORDS_GUESSED,
    /** The chord symbols were read from track {number}, '{name}', chord by chord. */
    CHORDS_READ,
    /**
     * Track {number} sounds notes that make no chord the score can name at {bars}, so the nearest chord was
     * written there, or the one before it kept. Arguments: track number, comma-separated first eight bars, count omitted.
     */
    UNNAMED_CHORDS,
    /**
     * The score came without sections, so the lyrics were laid along its tune before it was sung: {where}.
     * {unsung}The song ends one empty bar after the last line, {seconds:.0f} seconds in; with 'max_seconds' at 0
     * the singing is stopped at {ceiling:.0f} seconds should it run on.
     * Arguments: alternating tag and comma-separated source bars (an intro uses the code "intro"), then seconds, decimal ceiling.
     * UNSUNG_BARS carries the optional unsung clause separately. A localizer may show the first six sections, as Python does.
     */
    LYRICS_LAID,
    /** Not sung: {bar numbers} of the tune. */
    UNSUNG_BARS,
    /**
     * The tune has about {ratio} notes for each syllable of these lyrics -- its phrases hold about {notes} notes,
     * the lines about {syllables} syllables -- so YuE2 is likely to sing sounds or words of its own on the notes
     * left over. Arguments: ratio, notes, syllables.
     */
    CROWDED_NOTES,
    /**
     * These lyrics have more syllables than the tune has notes for them -- its phrases hold about {notes} notes,
     * the lines about {syllables} syllables -- so YuE2 is likely to crowd or drop words. Arguments: notes, syllables.
     */
    CROWDED_WORDS
}

enum class Why {
    /** The track was chosen by number. */
    CHOSEN,
    /** The karaoke syllables fall on this part's note onsets. */
    KARAOKE,
    /** The part is named for this line. */
    NAME,
    /** The highest melodic line, marked down for notes struck in chords. */
    HIGHEST,
    /** The busiest remaining melodic line. */
    BUSIEST,
    /** Channel 10 is drums and has no tune to sing. */
    DRUMS,
    /** A part named Chords holds the harmony rather than either melody. */
    CHORDS,
    /** This part was not selected by the automatic choice. */
    NOT_SELECTED
}

/** Why a file cannot be read or sung; the message contains a stable code, never UI prose. */
class MidiError(val code: ErrorCode, val args: List<String> = emptyList()) : IllegalArgumentException(code.name)

enum class ErrorCode {
    /** This is not a MIDI file: it does not begin with 'MThd'. */
    NOT_MIDI,
    /** The MIDI header is cut short. */
    SHORT_HEADER,
    /** The MIDI header is {size} bytes long, shorter than the format allows. */
    HEADER_LENGTH,
    /** The file is timed in SMPTE frames rather than beats, so it has no bars to write a score in. */
    SMPTE,
    /** The file gives a quarter note no ticks. */
    ZERO_DIVISION,
    /** The MIDI file holds no track. */
    NO_TRACK,
    /** This RIFF file says it holds MIDI, but it has no 'data' chunk. */
    RIFF_NO_DATA,
    /** The file ends in the middle of {context}. */
    TRUNCATED,
    /** A length in {context} runs past four bytes, which no MIDI file writes. */
    LONG_NUMBER,
    /** Track {number} holds a {status:02X} event, which does not belong in a MIDI file. */
    SYSTEM_EVENT,
    /** Track {number} has a data byte where an event should begin. */
    NO_STATUS,
    /** Track {number} has an event cut short of its data bytes. */
    SHORT_EVENT,
    /** 'mode' must be one of melody, full. */
    MODE,
    /** The file has no notes. */
    NO_NOTES,
    /** The file has no part with pitched notes, only drums. */
    NO_PITCHED_NOTES,
    /**
     * '{widget}' is track {number}, which is drums: drums have no tune to sing. The file's tracks: {listing}.
     * Arguments: widget, number, then number/name/family triplets for the listing.
     */
    DRUM_TRACK,
    /**
     * '{widget}' is track {number}, but this file has {count} tracks with notes: {listing}.
     * Arguments: widget, number, count, then number/name/family triplets for the listing.
     */
    TRACK_NUMBER,
    /** Instrument and vocal are both track {number}; one part cannot be both lines. */
    SAME_TRACK,
    /** The file runs past {limit} beats, longer than any song. */
    TOO_LONG,
    /** These events cannot be written as a native score. */
    NOTATION
}

internal fun fail(code: ErrorCode, vararg args: Any?): Nothing = throw MidiError(code, args.map { it.toString() })
