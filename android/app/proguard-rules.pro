# The JNI layer calls back into Kotlin by name: keep the native bridge and the token listener.
-keep class xyz.tifu.geode.engine.GeodeNative { *; }
-keep interface xyz.tifu.geode.engine.TokenListener { *; }
# sherpa-onnx (Geode's own voice): its JNI reads the config classes' fields and calls back by name
-keep class com.k2fsa.sherpa.onnx.** { *; }
# ... and the callback it calls by name: invoke(float[]): Integer
-keepclassmembers class xyz.tifu.geode.media.NeuralVoice$Feed { java.lang.Integer invoke(float[]); }
