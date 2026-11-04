// jni.cpp - JNI bindings for xyz.tifu.geode.engine.GeodeNative (the app's :engine and :probe processes).
#include <jni.h>

#include <string>
#include <vector>

#include "geode/geode.h"

namespace {

struct Callback {
    JNIEnv *env;
    jobject listener;
    jmethodID on_token;
};

int on_token(void *user, int32_t token) {
    auto *cb = static_cast<Callback *>(user);
    const jboolean stop = cb->env->CallBooleanMethod(cb->listener, cb->on_token, static_cast<jint>(token));
    if (cb->env->ExceptionCheck()) {
        cb->env->ExceptionClear();
        return 1;
    }
    return stop ? 1 : 0;
}

jstring utf(JNIEnv *env, const char *s) { return env->NewStringUTF(s ? s : ""); }

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_xyz_tifu_geode_engine_GeodeNative_version(JNIEnv *env, jobject) {
    return utf(env, geode_version());
}

// Returns a handle (> 0) or 0; on failure the message is written into errOut[0].
JNIEXPORT jlong JNICALL Java_xyz_tifu_geode_engine_GeodeNative_open(JNIEnv *env, jobject, jstring dir,
                                                                    jlong memoryBudget, jint computeThreads,
                                                                    jint ioThreads, jint helperThreads,
                                                                    jint contextLength, jboolean prefetch,
                                                                    jboolean gpu, jfloat gpuShare,
                                                                    jint expertsPerToken, jfloat routePrior,
                                                                    jint tailSkip, jboolean npu, jint mode,
                                                                    jint off, jobjectArray errOut) {
    const char *d = env->GetStringUTFChars(dir, nullptr);
    geode_config cfg{};
    cfg.model_dir = d;
    cfg.memory_budget = static_cast<uint64_t>(memoryBudget);
    cfg.compute_threads = computeThreads;
    cfg.io_threads = ioThreads;
    cfg.context_length = contextLength;
    cfg.helper_threads = helperThreads;
    cfg.no_prefetch = prefetch ? 0 : 1;
    cfg.gpu = gpu ? 1 : 0;
    cfg.gpu_share = gpuShare;
    cfg.experts_per_token = expertsPerToken;
    cfg.route_prior = routePrior;
    cfg.tail_skip = tailSkip;
    cfg.npu = npu ? 1 : 0;
    cfg.mode = mode;   // GEODE_MODE_*: the units that run the model (auto: the settings above)
    // what the user turned off (GeodeNative.OFF_*): bit 0 speculative decoding, 1 the vision encoder, 2 the audio one
    cfg.no_draft = off & 1;
    cfg.no_vision = (off >> 1) & 1;
    cfg.no_audio = (off >> 2) & 1;
    geode_model *model = nullptr;
    char err[512] = {0};
    const int rc = geode_open(&cfg, &model, err, sizeof err);
    env->ReleaseStringUTFChars(dir, d);
    if (rc != GEODE_OK) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
        return 0;
    }
    return reinterpret_cast<jlong>(model);
}

// Returns a GEODE_* code; on failure the message is written into errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_generate(
    JNIEnv *env, jobject, jlong handle, jintArray prompt, jfloat temperature, jfloat topP, jint topK,
    jfloat presencePenalty, jlong seed, jint maxTokens, jintArray stopTokens, jobject listener, jobjectArray errOut) {
    auto *model = reinterpret_cast<geode_model *>(handle);
    std::vector<int32_t> p(static_cast<size_t>(env->GetArrayLength(prompt)));
    if (!p.empty()) env->GetIntArrayRegion(prompt, 0, static_cast<jsize>(p.size()), p.data());
    std::vector<int32_t> stops(stopTokens ? static_cast<size_t>(env->GetArrayLength(stopTokens)) : 0);
    if (!stops.empty()) env->GetIntArrayRegion(stopTokens, 0, static_cast<jsize>(stops.size()), stops.data());
    geode_sampling s{};
    s.temperature = temperature;
    s.top_p = topP;
    s.top_k = topK;
    s.presence_penalty = presencePenalty;
    s.seed = static_cast<uint64_t>(seed);
    s.max_tokens = maxTokens;
    s.stop_tokens = stops.data();
    s.n_stop_tokens = static_cast<int32_t>(stops.size());
    Callback cb{env, listener, env->GetMethodID(env->GetObjectClass(listener), "onToken", "(I)Z")};
    char err[512] = {0};
    const int rc = geode_generate(model, p.data(), static_cast<int32_t>(p.size()), &s, on_token, &cb, err, sizeof err);
    if (rc != GEODE_OK && errOut && env->GetArrayLength(errOut) > 0)
        env->SetObjectArrayElement(errOut, 0, utf(env, err));
    return rc;
}

// An image for the next prompt (geode_add_image): rgb is h rows of w pixels, 3 bytes each; cache: its feature file
// (null: none). Returns its image_pad tokens, or -(a GEODE_* code) with the message in errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_addImage(JNIEnv *env, jobject, jlong handle,
                                                                       jbyteArray rgb, jint w, jint h, jstring cache,
                                                                       jobjectArray errOut) {
    const jsize n = env->GetArrayLength(rgb);
    if (static_cast<int64_t>(n) != static_cast<int64_t>(w) * h * 3) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, "bad image size"));
        return -GEODE_E_ARG;
    }
    jbyte *p = env->GetByteArrayElements(rgb, nullptr);
    const char *c = cache ? env->GetStringUTFChars(cache, nullptr) : nullptr;
    char err[512] = {0};
    int32_t tokens = 0;
    const int rc = geode_add_image(reinterpret_cast<geode_model *>(handle), reinterpret_cast<const uint8_t *>(p), w, h,
                                   c, &tokens, err, sizeof err);
    if (c) env->ReleaseStringUTFChars(cache, c);
    env->ReleaseByteArrayElements(rgb, p, JNI_ABORT);
    if (rc != GEODE_OK) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
        return -rc;
    }
    return tokens;
}

// A video's frame for the next prompt (geode_add_video_frame), as addImage. Returns its <|video|> tokens, or -(a
// GEODE_* code) with the message in errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_addVideoFrame(JNIEnv *env, jobject, jlong handle,
                                                                            jbyteArray rgb, jint w, jint h,
                                                                            jstring cache, jobjectArray errOut) {
    const jsize n = env->GetArrayLength(rgb);
    if (static_cast<int64_t>(n) != static_cast<int64_t>(w) * h * 3) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, "bad frame size"));
        return -GEODE_E_ARG;
    }
    jbyte *p = env->GetByteArrayElements(rgb, nullptr);
    const char *c = cache ? env->GetStringUTFChars(cache, nullptr) : nullptr;
    char err[512] = {0};
    int32_t tokens = 0;
    const int rc = geode_add_video_frame(reinterpret_cast<geode_model *>(handle), reinterpret_cast<const uint8_t *>(p),
                                         w, h, c, &tokens, err, sizeof err);
    if (c) env->ReleaseStringUTFChars(cache, c);
    env->ReleaseByteArrayElements(rgb, p, JNI_ABORT);
    if (rc != GEODE_OK) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
        return -rc;
    }
    return tokens;
}

// A sound clip for the next prompt (geode_add_audio): 16 kHz mono floats in [-1, 1]. Returns its <|audio|> tokens,
// or -(a GEODE_* code) with the message in errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_addAudio(JNIEnv *env, jobject, jlong handle,
                                                                       jfloatArray pcm, jobjectArray errOut) {
    const jsize n = pcm ? env->GetArrayLength(pcm) : 0;
    if (n <= 0) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, "an empty clip"));
        return -GEODE_E_ARG;
    }
    jfloat *p = env->GetFloatArrayElements(pcm, nullptr);
    char err[512] = {0};
    int32_t tokens = 0;
    const int rc = geode_add_audio(reinterpret_cast<geode_model *>(handle), p, n, &tokens, err, sizeof err);
    env->ReleaseFloatArrayElements(pcm, p, JNI_ABORT);
    if (rc != GEODE_OK) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
        return -rc;
    }
    return tokens;
}

// geode_audio_tokens: the <|audio|> tokens a clip of n samples takes (0: no audio encoder, or too short a clip).
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_audioTokens(JNIEnv *, jobject, jlong handle,
                                                                          jlong samples) {
    return geode_audio_tokens(reinterpret_cast<geode_model *>(handle), samples);
}

// geode_cache_image: the image's features into its cache file only. A GEODE_* code.
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_cacheImage(JNIEnv *env, jobject, jlong handle,
                                                                         jbyteArray rgb, jint w, jint h, jstring cache) {
    if (!rgb || !cache || static_cast<int64_t>(env->GetArrayLength(rgb)) != static_cast<int64_t>(w) * h * 3)
        return GEODE_E_ARG;
    jbyte *p = env->GetByteArrayElements(rgb, nullptr);
    const char *c = env->GetStringUTFChars(cache, nullptr);
    char err[512] = {0};
    const int rc = geode_cache_image(reinterpret_cast<geode_model *>(handle), reinterpret_cast<const uint8_t *>(p), w,
                                     h, c, err, sizeof err);
    env->ReleaseStringUTFChars(cache, c);
    env->ReleaseByteArrayElements(rgb, p, JNI_ABORT);
    return rc;
}

// geode_prefill: the prompt into the conversation, nothing generated. A GEODE_* code; on failure the message is
// written into errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_prefill(JNIEnv *env, jobject, jlong handle, jintArray prompt,
                                                                      jobjectArray errOut) {
    std::vector<int32_t> p(static_cast<size_t>(env->GetArrayLength(prompt)));
    if (!p.empty()) env->GetIntArrayRegion(prompt, 0, static_cast<jsize>(p.size()), p.data());
    char err[512] = {0};
    const int rc = geode_prefill(reinterpret_cast<geode_model *>(handle), p.data(), static_cast<int32_t>(p.size()), err,
                                 sizeof err);
    if (rc != GEODE_OK && errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
    return rc;
}

// geode_state_save / geode_state_load: the conversation's length in tokens, or -(a GEODE_* code) with the message in
// errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_stateSave(JNIEnv *env, jobject, jlong handle, jstring path,
                                                                        jboolean append, jobjectArray errOut) {
    const char *p = env->GetStringUTFChars(path, nullptr);
    char err[512] = {0};
    int32_t pos = 0;
    const int rc = geode_state_save(reinterpret_cast<geode_model *>(handle), p, append ? 1 : 0, &pos, err, sizeof err);
    env->ReleaseStringUTFChars(path, p);
    if (rc != GEODE_OK) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
        return -rc;
    }
    return pos;
}

JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_stateLoad(JNIEnv *env, jobject, jlong handle, jstring path,
                                                                        jobjectArray errOut) {
    const char *p = env->GetStringUTFChars(path, nullptr);
    char err[512] = {0};
    int32_t pos = 0;
    const int rc = geode_state_load(reinterpret_cast<geode_model *>(handle), p, &pos, err, sizeof err);
    env->ReleaseStringUTFChars(path, p);
    if (rc != GEODE_OK) {
        if (errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
        return -rc;
    }
    return pos;
}

// geode_reserve_memory: the bytes the reserve holds now, or -(a GEODE_* code) with the message in errOut[0]; a
// reserve smaller than asked is no failure (its message is in errOut[0] too).
JNIEXPORT jlong JNICALL Java_xyz_tifu_geode_engine_GeodeNative_reserveMemory(JNIEnv *env, jobject, jlong handle,
                                                                             jlong bytes, jobjectArray errOut) {
    char err[512] = {0};
    uint64_t held = 0;
    const int rc = geode_reserve_memory(reinterpret_cast<geode_model *>(handle), static_cast<uint64_t>(bytes < 0 ? 0 : bytes),
                                        &held, err, sizeof err);
    if (err[0] && errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
    return rc != GEODE_OK ? -static_cast<jlong>(rc) : static_cast<jlong>(held);
}

// geode_set_experts: the routed experts a token takes from now on. A GEODE_* code; the message in errOut[0].
JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_setExperts(JNIEnv *env, jobject, jlong handle, jint k,
                                                                         jobjectArray errOut) {
    char err[512] = {0};
    const int rc = geode_set_experts(reinterpret_cast<geode_model *>(handle), k, err, sizeof err);
    if (rc != GEODE_OK && errOut && env->GetArrayLength(errOut) > 0) env->SetObjectArrayElement(errOut, 0, utf(env, err));
    return rc;
}

// geode_profile's text (where the time went, and the GPU path's state on its last line)
JNIEXPORT jstring JNICALL Java_xyz_tifu_geode_engine_GeodeNative_profile(JNIEnv *env, jobject, jlong handle) {
    std::vector<char> out(16384);
    geode_profile(reinterpret_cast<geode_model *>(handle), out.data(), out.size());
    return utf(env, out.data());
}

JNIEXPORT void JNICALL Java_xyz_tifu_geode_engine_GeodeNative_stop(JNIEnv *, jobject, jlong handle) {
    geode_stop(reinterpret_cast<geode_model *>(handle));
}

JNIEXPORT void JNICALL Java_xyz_tifu_geode_engine_GeodeNative_reset(JNIEnv *, jobject, jlong handle) {
    geode_reset(reinterpret_cast<geode_model *>(handle));
}

JNIEXPORT void JNICALL Java_xyz_tifu_geode_engine_GeodeNative_close(JNIEnv *, jobject, jlong handle) {
    geode_close(reinterpret_cast<geode_model *>(handle));
}

// [memory_used, expert_cache_bytes, expert_hits, expert_misses, flash_bytes_read, context_used, context_length,
//  the two speeds x1000, context_allocated, expert_ahead_hits, expert_reads_ahead, experts_per_token] as longs.
JNIEXPORT jlongArray JNICALL Java_xyz_tifu_geode_engine_GeodeNative_stats(JNIEnv *env, jobject, jlong handle) {
    geode_stats s{};
    geode_get_stats(reinterpret_cast<geode_model *>(handle), &s);
    const jlong v[13] = {static_cast<jlong>(s.memory_used), static_cast<jlong>(s.expert_cache_bytes),
                         static_cast<jlong>(s.expert_hits), static_cast<jlong>(s.expert_misses),
                         static_cast<jlong>(s.flash_bytes_read), s.context_used, s.context_length,
                         static_cast<jlong>(s.prompt_tokens_per_s * 1000), static_cast<jlong>(s.decode_tokens_per_s * 1000),
                         s.context_allocated, static_cast<jlong>(s.expert_ahead_hits),
                         static_cast<jlong>(s.expert_reads_ahead), s.experts_per_token};
    jlongArray out = env->NewLongArray(13);
    env->SetLongArrayRegion(out, 0, 13, v);
    return out;
}

JNIEXPORT jint JNICALL Java_xyz_tifu_geode_engine_GeodeNative_memoryProbe(JNIEnv *env, jobject, jstring statusPath,
                                                                          jlong maxBytes, jlong stepBytes,
                                                                          jint pauseMs) {
    const char *p = env->GetStringUTFChars(statusPath, nullptr);
    const int rc = geode_memory_probe(p, static_cast<uint64_t>(maxBytes), static_cast<uint64_t>(stepBytes), pauseMs);
    env->ReleaseStringUTFChars(statusPath, p);
    return rc;
}

// A benchmark's JSON (geode_bench), or an empty string for an unknown test.
JNIEXPORT jstring JNICALL Java_xyz_tifu_geode_engine_GeodeNative_bench(JNIEnv *env, jobject, jstring what, jstring arg) {
    const char *w = env->GetStringUTFChars(what, nullptr);
    const char *a = env->GetStringUTFChars(arg, nullptr);
    std::vector<char> out(1 << 20);
    const int rc = geode_bench(w, a, out.data(), out.size());
    env->ReleaseStringUTFChars(what, w);
    env->ReleaseStringUTFChars(arg, a);
    return utf(env, rc == GEODE_OK ? out.data() : "");
}

}  // extern "C"
