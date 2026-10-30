// npu/host/npu_app.c - the NPU checks (probe.c) from inside the app (debug builds: NpuDebugReceiver), to learn
// whether an app process (SELinux domain untrusted_app) reaches the cDSP at all. FastRPC's library is the vendor's
// public libcdsprpc.so (the manifest's <uses-native-library>); the DSP finds the module's skel through
// ADSP_LIBRARY_PATH, which must name the app's native library folder before the first FastRPC call.
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "geode_npu.h"
#include "probe.h"

typedef struct {
    char *buf;
    size_t len, cap;
} Text;

static void to_text(void *ctx, const char *line) {
    Text *t = (Text *)ctx;
    const size_t n = strlen(line);
    if (t->len + n + 1 > t->cap) return;
    memcpy(t->buf + t->len, line, n + 1);
    t->len += n;
}

JNIEXPORT jstring JNICALL Java_xyz_tifu_geode_npu_NpuDebugReceiver_probe(JNIEnv *env, jclass cls, jstring lib_dir,
                                                                          jint reps) {
    (void)cls;
    static char buf[16384];
    Text t = {buf, 0, sizeof buf};
    buf[0] = 0;
    const char *dir = (*env)->GetStringUTFChars(env, lib_dir, NULL);
    char path[1024];
    snprintf(path, sizeof path, "%s;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp", dir);
    (*env)->ReleaseStringUTFChars(env, lib_dir, dir);
    setenv("ADSP_LIBRARY_PATH", path, 1);
    to_text(&t, "ADSP_LIBRARY_PATH=");
    to_text(&t, path);
    to_text(&t, "\n");
    remote_handle64 h = npu_open(to_text, &t);
    if (h) {
        const int bad = npu_bench(h, reps, to_text, &t);
        char line[64];
        snprintf(line, sizeof line, "failures: %d\n", bad);
        to_text(&t, line);
        geode_npu_close(h);
    }
    return (*env)->NewStringUTF(env, buf);
}
