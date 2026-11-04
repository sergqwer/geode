package xyz.tifu.geode.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

/**
 * Something is being done: three dots, one lit after another. In place of a spinning progress indicator where the wait
 * can be long - the prompt read, a tool run, the context compressed: a spinner draws the screen at its full rate
 * (60 frames a second) the whole time, and on a phone whose cores the model is using that is taken from the model.
 * These step 2.5 times a second, and only the dots' layers change (no composition, no layout).
 */
@Composable
fun Dots(modifier: Modifier = Modifier, dot: Dp = 6.dp, color: Color = MaterialTheme.colorScheme.primary) {
    var lit by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(STEP_MS)
            lit = (lit + 1) % 3
        }
    }
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(dot * 0.6f), verticalAlignment = Alignment.CenterVertically) {
        for (i in 0 until 3) {
            Box(Modifier.size(dot).graphicsLayer { alpha = if (i == lit) 1f else 0.3f }.background(color, CircleShape))
        }
    }
}

private const val STEP_MS = 400L
