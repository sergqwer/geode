package xyz.tifu.geode.util

import java.util.Locale

/** 1536 MiB -> "1.5 GB" (binary units, the way phones report storage). */
fun formatBytes(bytes: Long): String {
    val units = arrayOf("B", "KB", "MB", "GB", "TB")
    var v = bytes.toDouble()
    var i = 0
    while (v >= 1024 && i < units.size - 1) {
        v /= 1024
        i++
    }
    return if (i == 0) "$bytes B" else String.format(Locale.getDefault(), "%.1f %s", v, units[i])
}

fun formatMb(mb: Int): String = formatBytes(mb.toLong() shl 20)
