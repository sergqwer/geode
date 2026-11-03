package xyz.tifu.geode.media

import android.Manifest
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/** The microphone, for a voice message: 16 kHz mono, as the model hears it, kept on the phone. */
object Microphone {
    class Failed(message: String) : IllegalStateException(message)

    /**
     * Records until [stop] is set or the clip is full ([AudioStore.MAX_SAMPLES]); [progress] gets the samples so
     * far, every tenth of a second. Cancelled: the recording ends, nothing kept.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun record(stop: AtomicBoolean, progress: (Int) -> Unit): FloatArray = withContext(Dispatchers.IO) {
        val rate = AudioStore.RATE
        val min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) throw Failed("the microphone does not record at 16 kHz")
        // the source speech recognition uses: no effects that change the sound with the room
        val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, rate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(min, rate / 2))
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            throw Failed("the microphone could not be opened")
        }
        val out = FloatArray(AudioStore.MAX_SAMPLES)
        val buf = ShortArray(rate / 10)
        var n = 0
        try {
            rec.startRecording()
            while (currentCoroutineContext().isActive && !stop.get() && n < out.size) {
                val r = rec.read(buf, 0, minOf(buf.size, out.size - n))
                if (r < 0) throw Failed("the microphone stopped ($r)")
                for (i in 0 until r) out[n + i] = buf[i] / 32768f
                n += r
                progress(n)
            }
        } finally {
            runCatching { rec.stop() }
            rec.release()
        }
        out.copyOf(n)
    }
}

/** A clip played back (one at a time): the samples as they are, 16 kHz mono. */
class ClipPlayer {
    private var track: AudioTrack? = null

    /**
     * Plays [pcm]; [done] when it ends (not when it is stopped). False: the sound could not be played (the audio
     * system refused the track: busy, or an output switching) - nothing plays, nothing is left open.
     */
    fun play(pcm: FloatArray, done: () -> Unit): Boolean {
        stop()
        if (pcm.isEmpty()) {
            done()
            return true
        }
        var t: AudioTrack? = null
        return try {
            val made = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(AudioStore.RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STATIC)
                .setBufferSizeInBytes(pcm.size * 4)
                .build()
            t = made
            val written = made.write(pcm, 0, pcm.size, AudioTrack.WRITE_BLOCKING)
            if (written != pcm.size) throw IllegalStateException("the track took $written of ${pcm.size} samples")
            made.notificationMarkerPosition = pcm.size
            made.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack) {
                    if (this@ClipPlayer.track === track) stop()
                    done()
                }

                override fun onPeriodicNotification(track: AudioTrack) = Unit
            })
            made.play()
            track = made
            true
        } catch (e: Exception) {
            t?.release()
            false
        }
    }

    fun stop() {
        track?.let {
            runCatching { it.stop() }
            it.release()
        }
        track = null
    }
}
