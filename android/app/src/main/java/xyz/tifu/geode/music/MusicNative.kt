package xyz.tifu.geode.music

/** A song's progress, called by the engine on the thread that runs [MusicNative.makeSong]. */
fun interface SongProgressListener {
    /** [stage]: SongStage's ordinal (MusicStage); [done] of [total] the stage's units (tokens, ODE evaluations, VAE
     *  tiles - total may be a ceiling); [elapsed] seconds since the song began. */
    fun onProgress(stage: Int, chunk: Int, chunks: Int, done: Long, total: Long, elapsed: Double)
}

/** The stages of engine/src/yue2_song.hpp's SongStage, in its order. */
enum class MusicStage { LOAD, SCORE, CODES, PREFIX, ACOUSTIC, DECODE, MASTER, WRITE, JOIN }   // (yue2::SongStage, in its order)

/**
 * YuE2's songs (engine/src/yue2_song.hpp) through engine/android/music_jni.cpp: libgeode_music.so, loaded only in the
 * :music process (MusicService) - never in the app's, which keeps the chat's engine and the UI.
 */
object MusicNative {
    init {
        System.loadLibrary("geode_music")
    }

    /** A whole song from [request] (the JSON yue2::song_request_from_json reads): blocks until the file is written,
     *  cancelled or failed. The result JSON (yue2::song_result_json). */
    external fun makeSong(request: String, listener: SongProgressListener?): String

    /** From any thread: the song running (or the next one, until [reset]) stops at its next check and [makeSong]
     *  returns {"cancelled": true}. */
    external fun cancel()

    /** The cancel flag cleared: before a song starts (MusicService checks its own cancels after this). */
    external fun reset()
}
