# BoringSSL (btls-sys) for Android arm64 through the NDK's own toolchain: given here, btls-sys leaves the compiler to
# it (its own .cmd compiler wrapper made CMake take the build for Windows)
set(ANDROID_ABI arm64-v8a)
set(ANDROID_PLATFORM android-30)
set(ANDROID_STL c++_static)
include("D:/android-tools/sdk/ndk/29.0.14206865/build/cmake/android.toolchain.cmake")
