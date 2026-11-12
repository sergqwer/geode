// music_jni.cpp - libgeode_music.so: YuE2 songs (engine/src/yue2_song.hpp) for xyz.tifu.geode.music.MusicNative,
// loaded only by the app's :music process (MusicService). One song at a time: makeSong blocks its thread (the
// service's worker) until the file is written, cancelled or failed; progress comes back on that thread.
#include <jni.h>

#include <atomic>
#include <string>

#include "../src/yue2_song.hpp"

using namespace geode;

namespace {
std::atomic<bool> g_cancel{false};

std::string text_of(JNIEnv *env, jstring s) {
    if (!s) return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}
}  // namespace

extern "C" {

// request: the JSON yue2::song_request_from_json reads (pack_dir, npu_dir and output included); listener: an object
// with onProgress(stage: Int, chunk: Int, chunks: Int, done: Long, total: Long, elapsed: Double). The result JSON
// (yue2::song_result_json).
JNIEXPORT jstring JNICALL Java_xyz_tifu_geode_music_MusicNative_makeSong(JNIEnv *env, jclass, jstring request,
                                                                         jobject listener) {
    // (the flag is not cleared here: a cancel that came after MusicNative.reset is this song's)
    yue2::SongRequest q;
    std::string err;
    yue2::SongResult r;
    if (!yue2::song_request_from_json(text_of(env, request), q, err)) {
        r.error = "song request: " + err;
        return env->NewStringUTF(yue2::song_result_json(r).c_str());
    }
    jmethodID on = nullptr;
    if (listener) {
        jclass cls = env->GetObjectClass(listener);
        on = env->GetMethodID(cls, "onProgress", "(IIIJJD)V");
        env->DeleteLocalRef(cls);
        if (!on) env->ExceptionClear();
    }
    yue2::SongCallbacks cb;
    cb.progress = [&](const yue2::SongProgress &p) {
        if (!on) return;
        env->CallVoidMethod(listener, on, static_cast<jint>(p.stage), static_cast<jint>(p.chunk), static_cast<jint>(p.chunks),
                            static_cast<jlong>(p.done), static_cast<jlong>(p.total), static_cast<jdouble>(p.elapsed));
        if (env->ExceptionCheck()) env->ExceptionClear();
    };
    cb.cancelled = [] { return g_cancel.load(std::memory_order_relaxed); };
    r = yue2::make_song(q, cb);
    return env->NewStringUTF(yue2::song_result_json(r).c_str());
}

// from any thread: the running song stops at its next check (a token, a piece of a prompt, an ODE evaluation, a
// VAE tile) and makeSong returns {"cancelled": true}
JNIEXPORT void JNICALL Java_xyz_tifu_geode_music_MusicNative_cancel(JNIEnv *, jclass) { g_cancel = true; }

// before a song: the flag cleared (MusicService looks at its own cancels after this, so none falls in between)
JNIEXPORT void JNICALL Java_xyz_tifu_geode_music_MusicNative_reset(JNIEnv *, jclass) { g_cancel = false; }

}  // extern "C"
