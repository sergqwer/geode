package xyz.tifu.geode.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// The geode's crystal violet: without Material You, or with it turned off.
private val Light = lightColorScheme(primary = Color(0xFF6A4FE0), secondary = Color(0xFF7C6F64))
private val Dark = darkColorScheme(primary = Color(0xFFB39DFF), secondary = Color(0xFFC9BCB0))

/** themeMode: 0 = follow the system, 1 = light, 2 = dark; dynamicColor: the wallpaper's colours (Android 12+). */
@Composable
fun GeodeTheme(themeMode: Int, dynamicColor: Boolean = true, content: @Composable () -> Unit) {
    val dark = when (themeMode) {
        1 -> false
        2 -> true
        else -> isSystemInDarkTheme()
    }
    val context = LocalContext.current
    val scheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> Dark
        else -> Light
    }
    MaterialTheme(colorScheme = scheme, content = content)
}
