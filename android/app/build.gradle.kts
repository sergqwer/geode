import java.io.FileInputStream
import java.security.SecureRandom
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

// The upload key from a gitignored keystore.properties. Absent on a machine
// without the key: the release stays unsigned.
val keystorePropsFile = rootProject.file("keystore.properties")
val keystoreProps = Properties().apply {
    if (keystorePropsFile.exists()) FileInputStream(keystorePropsFile).use { load(it) }
}

// The key the app signs its requests to geode.tifu.xyz with (SiteApi.kt; the server's GEODE_SITE_KEY): the file
// keystore.properties names (siteKeyFile, kept outside the repository). In the build as two byte arrays - a random mask
// and the key under it - never as one string. Absent: the requests go unsigned (the server refuses device reports).
val siteKey: ByteArray = keystoreProps.getProperty("siteKeyFile")?.let { file(it) }?.takeIf { it.isFile }
    ?.readText()?.trim()?.chunked(2)?.map { it.toInt(16).toByte() }?.toByteArray() ?: ByteArray(0)
val siteMask = ByteArray(siteKey.size).also { SecureRandom().nextBytes(it) }
fun javaBytes(b: ByteArray) = b.joinToString(",", "new byte[]{", "}") { it.toString() }

android {
    namespace = "xyz.tifu.geode"
    compileSdk = 36
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "xyz.tifu.geode"
        // 30: ActivityManager.getHistoricalProcessExitReasons for the memory test
        minSdk = 30
        targetSdk = 36
        versionCode = 13
        versionName = "0.9.12"
        buildConfigField("byte[]", "SITE_MASK", javaBytes(siteMask))
        buildConfigField("byte[]", "SITE_KEY",
            javaBytes(ByteArray(siteKey.size) { (siteKey[it].toInt() xor siteMask[it].toInt()).toByte() }))
        ndk {
            // the engine needs ARMv8.2 dotprod; a 35B model on a 32-bit phone makes no sense (PLAN.md)
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_static", "-DGEODE_TOOLS=OFF")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("../../engine/CMakeLists.txt")
            version = "4.1.2"
        }
    }

    signingConfigs {
        create("release") {
            if (keystorePropsFile.exists()) {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // the engine's symbols in the bundle: Play symbolicates native crashes with them
            ndk { debugSymbolLevel = "FULL" }
        }
        // The release build (R8, the same rules) under its own id and the debug key: installs beside a debug
        // build without touching its models, for a smoke test on a phone
        create("rc") {
            initWith(getByName("release"))
            applicationIdSuffix = ".rc"
            versionNameSuffix = "-rc"
            signingConfig = signingConfigs.getByName("debug")
            // run-as works on it: a test model goes into its files over adb (it never ships)
            isDebuggable = true
            matchingFallbacks += "release"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // the agent's Linux image, delivered by Google Play with the app
    assetPacks += listOf(":linux_pack")
    packaging {
        // native libraries extracted at install (nativeLibraryDir): the agent's Linux execs proot from there, the one
        // place an app targeting Android 10+ may exec files from (tools/linux_tools.py)
        jniLibs { useLegacyPackaging = true }
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    androidResources {
        generateLocaleConfig = false
    }
    testOptions {
        // TokenizerTest and GemmaTokenizerTest read what tools/tokbin.py wrote
        unitTests.all {
            it.systemProperty("geode.tokdir", rootProject.file("../work/tok").absolutePath)
            it.inputs.files(rootProject.fileTree("../work/tok"))
            it.systemProperty("geode.gemmatokdir", rootProject.file("../work/tok-gemma").absolutePath)
            it.inputs.files(rootProject.fileTree("../work/tok-gemma"))
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.documentfile)
    // the web chat's QR sign-in: the code drawn (server) and read from the camera (app), all on the phone
    implementation(libs.zxing.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    // the agent's Linux image unpacked on the phone (.tar.xz; pure Java)
    implementation(libs.xz)
    debugImplementation(libs.androidx.ui.tooling)
    testImplementation(libs.junit)
    testImplementation(libs.org.json)
}
