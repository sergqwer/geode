# The JNI layer calls back into Kotlin by name: keep the native bridge and the token listener.
-keep class xyz.tifu.geode.engine.GeodeNative { *; }
-keep interface xyz.tifu.geode.engine.TokenListener { *; }
