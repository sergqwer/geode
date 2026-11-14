// Root build script — only plugin declarations live here. Per-module configuration is in app/build.gradle.kts.
plugins {
    alias(libs.plugins.android.application) apply false
    // not applied (AGP 9 builds Kotlin itself): only puts this Kotlin version on the build's classpath
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
