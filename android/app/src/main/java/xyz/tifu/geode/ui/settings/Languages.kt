package xyz.tifu.geode.ui.settings

/**
 * The app's languages: English (res/values, what a phone in any other language gets) and the translations, each by
 * its BCP 47 tag (res/xml/locales_config.xml lists the same) and its name in itself. Indonesian's folder is values-in
 * (Android's old code) and Filipino's values-tl, as Android's own apps have them.
 */
object Languages {
    val ALL: List<Pair<String, String>> = listOf(
        "en" to "English",
        "zh-Hans" to "简体中文",
        "zh-Hant" to "繁體中文",
        "hi" to "हिन्दी",
        "es" to "Español",
        "ar" to "العربية",
        "fr" to "Français",
        "bn" to "বাংলা",
        "pt" to "Português",
        "ru" to "Русский",
        "id" to "Bahasa Indonesia",
        "ur" to "اردو",
        "de" to "Deutsch",
        "ja" to "日本語",
        "mr" to "मराठी",
        "vi" to "Tiếng Việt",
        "te" to "తెలుగు",
        "ha" to "Hausa",
        "tr" to "Türkçe",
        "pa" to "ਪੰਜਾਬੀ",
        "sw" to "Kiswahili",
        "fil" to "Filipino",
        "ta" to "தமிழ்",
        "fa" to "فارسی",
        "ko" to "한국어",
        "th" to "ไทย",
        "jv" to "Basa Jawa",
        "it" to "Italiano",
        "gu" to "ગુજરાતી",
        "kn" to "ಕನ್ನಡ",
        "uk" to "Українська",
    )

    /** The tag of ALL a picked tag is ("" and unknown: "") */
    fun canonical(tag: String): String {
        val first = tag.substringBefore(',').trim()
        if (first.isEmpty()) return ""
        ALL.firstOrNull { it.first.equals(first, ignoreCase = true) }?.let { return it.first }
        // "id" may come back as Android's old "in", "fil" as "tl"; a region the system added ("de-DE") is dropped;
        // Chinese by its script, else its region
        val parts = first.split('-')
        val lang = parts[0].lowercase()
        if (lang == "zh") {
            val trad = parts.drop(1).any { it.equals("Hant", true) || it.uppercase() in setOf("TW", "HK", "MO") }
            return if (trad) "zh-Hant" else "zh-Hans"
        }
        val alias = mapOf("in" to "id", "tl" to "fil")[lang] ?: lang
        return ALL.firstOrNull { it.first == alias }?.first ?: ""
    }

    fun name(tag: String): String? = canonical(tag).takeIf { it.isNotEmpty() }?.let { t -> ALL.first { it.first == t }.second }
}
