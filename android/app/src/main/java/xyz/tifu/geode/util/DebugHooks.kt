package xyz.tifu.geode.util

import kotlinx.coroutines.flow.MutableSharedFlow

/**
 * What debug builds drive from adb (LinuxDebugReceiver, in the debug source set only): nothing sends here in a
 * release build, and the app listens only when BuildConfig.DEBUG.
 */
object DebugHooks {
    /** An image (ImageStore's name) to attach to the open chat's next question, as the photo picker would. */
    val attach = MutableSharedFlow<String>(extraBufferCapacity = 4)

    /** A sound clip (AudioStore's name) to attach to the open chat's next question, as the sound picker would. */
    val attachAudio = MutableSharedFlow<String>(extraBufferCapacity = 4)

    /** A video file to attach to the open chat's next question, as the video picker would (its Uri). */
    val attachVideo = MutableSharedFlow<android.net.Uri>(extraBufferCapacity = 4)
}
