// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/query.py: the search syntax in a query (RawTextQuery) — `<3` a timeout, `:uk` a language, `!gh` an engine
//! or `!it` a category, `!!` "feeling lucky"; `!!x` external bangs are not known here (searx/external_bang.py is
//! left out: they only redirect a browser), so such a word stays in the query. Plus geode-search's rule for the
//! language when the query names none (geode_search.py's): from the query's script, else the phone's.

use crate::locales;
use crate::settings::{self, Engine};

/// A selected engine and the category it was selected for (searx.search.models.EngineRef).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineRef {
    pub name: &'static str,
    pub category: String,
}

#[derive(Debug, Default)]
pub struct RawTextQuery {
    pub enginerefs: Vec<EngineRef>,
    pub languages: Vec<String>,
    pub timeout_limit: Option<f64>,
    pub specific: bool,
    pub redirect_to_first_result: bool,
    user_query_parts: Vec<String>,
}

/// searx.webutils.VALID_LANGUAGE_CODE: ^[a-z]{2,3}(-[a-zA-Z]{2})?$
pub fn valid_language_code(v: &str) -> bool {
    let (l, r) = match v.split_once('-') {
        Some((l, r)) => (l, Some(r)),
        None => (v, None),
    };
    (2..=3).contains(&l.len())
        && l.bytes().all(|b| b.is_ascii_lowercase())
        && r.is_none_or(|r| r.len() == 2 && r.bytes().all(|b| b.is_ascii_alphabetic()))
}

impl RawTextQuery {
    /// `disabled`: the engines left out unless named (a category bang does not bring them in).
    pub fn parse(query: &str, disabled: &[&str]) -> RawTextQuery {
        let mut q = RawTextQuery::default();
        // re.split(r'(\s+)', query): the words; the whitespace between them is dropped by getQuery
        for part in query.split(char::is_whitespace).filter(|p| !p.is_empty()) {
            let special = if let Some(v) = part.strip_prefix('<') {
                q.timeout(v)
            } else if let Some(v) = part.strip_prefix(':') {
                q.language(v)
            } else if part.starts_with("!!") && part.chars().count() > 2 {
                false // ExternalBangParser: no external bangs known
            } else if part.starts_with('!') && part != "!!" {
                q.bang(&part[1..], disabled)
            } else if part == "!!" {
                q.redirect_to_first_result = true;
                true
            } else {
                false
            };
            if !special {
                q.user_query_parts.push(part.to_string());
            }
        }
        q
    }

    /// The query without its search syntax.
    pub fn query(&self) -> String {
        self.user_query_parts.join(" ")
    }

    fn timeout(&mut self, v: &str) -> bool {
        if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        let Ok(n) = v.parse::<u64>() else { return false };
        // below 100 seconds (<3), from 100 on milliseconds (<850)
        self.timeout_limit = Some(if n < 100 { n as f64 } else { n as f64 / 1000.0 });
        true
    }

    fn language(&mut self, raw: &str) -> bool {
        let value = raw.to_lowercase().replace('_', "-");
        if value.is_empty() {
            return false;
        }
        let mut found = false;
        for lc in locales::sxng_locales() {
            let [id, name, country, english] = lc.clone().map(|s| s.to_lowercase());
            if (value == id || value == name || value == english || value.replace('-', " ") == country)
                && !self.languages.contains(&value)
            {
                found = true;
                let parts: Vec<&str> = id.split('-').collect();
                self.languages.push(if parts.len() == 2 {
                    format!("{}-{}", parts[0], parts[1].to_uppercase())
                } else {
                    id.clone()
                });
                if value == id {
                    break;
                }
            }
        }
        // a valid, yet not selectable language
        if valid_language_code(&value) || value == "auto" {
            let v = match value.split_once('-') {
                Some((l, r)) => format!("{}-{}", l.to_lowercase(), r.to_uppercase()),
                None => value.clone(),
            };
            if !self.languages.contains(&v) {
                self.languages.push(v);
                found = true;
            }
        }
        found
    }

    fn bang(&mut self, raw: &str, disabled: &[&str]) -> bool {
        let value = raw.replace(['-', '_'], " ").to_lowercase();
        if value.is_empty() {
            return false;
        }
        let name = settings::by_shortcut(&value).map(|e| e.name.to_string()).unwrap_or(value);
        let found = if let Some(e) = settings::engine(&name) {
            self.enginerefs.push(EngineRef { name: e.name, category: "none".into() });
            true
        } else if settings::is_category(&name) {
            let engines: Vec<&Engine> = settings::category(&name);
            self.enginerefs.extend(
                engines
                    .into_iter()
                    .filter(|e| !disabled.contains(&e.name))
                    .map(|e| EngineRef { name: e.name, category: name.clone() }),
            );
            true
        } else {
            false
        };
        if found {
            self.specific = true;
        }
        found
    }
}

/// A query's script tells its language well enough for an engine's region (DuckDuckGo answers "all regions" with a
/// captcha); Latin text says nothing, so it goes by the phone's language (`phone`, from $GEODE_SEARCH_LANG) or en.
pub fn guess_lang(query: &str, phone: Option<&str>) -> String {
    const SCRIPTS: &[(u32, u32, &str)] = &[
        (0x0400, 0x04FF, "ru"),
        (0x0370, 0x03FF, "el"),
        (0x0590, 0x05FF, "he"),
        (0x0600, 0x06FF, "ar"),
        (0x0900, 0x097F, "hi"),
        (0x0E00, 0x0E7F, "th"),
        (0x3040, 0x30FF, "ja"),
        (0xAC00, 0xD7AF, "ko"),
        (0x4E00, 0x9FFF, "zh"),
    ];
    if query.chars().any(|c| "іїєґІЇЄҐ".contains(c)) {
        return "uk".into();
    }
    for c in query.chars() {
        let o = c as u32;
        if let Some((_, _, l)) = SCRIPTS.iter().find(|(lo, hi, _)| (*lo..=*hi).contains(&o)) {
            return (*l).into();
        }
    }
    phone.filter(|p| !p.is_empty()).unwrap_or("en").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DISABLED: &[&str] = &["google", "bing", "startpage"];

    #[test]
    fn plain_query() {
        let q = RawTextQuery::parse("  python   asyncio timeout ", DISABLED);
        assert_eq!(q.query(), "python asyncio timeout");
        assert!(q.enginerefs.is_empty() && q.languages.is_empty() && !q.specific);
    }

    #[test]
    fn engine_bang_by_shortcut_and_name() {
        let q = RawTextQuery::parse("!gh deepseek harness", DISABLED);
        assert_eq!(q.query(), "deepseek harness");
        assert_eq!(q.enginerefs, vec![EngineRef { name: "github", category: "none".into() }]);
        assert!(q.specific);
        // a disabled engine named by its bang is asked
        let q = RawTextQuery::parse("!go test", DISABLED);
        assert_eq!(q.enginerefs[0].name, "google");
        let q = RawTextQuery::parse("!stackoverflow x", DISABLED);
        assert_eq!(q.enginerefs[0].name, "stackoverflow");
    }

    #[test]
    fn category_bang_leaves_disabled_out() {
        let q = RawTextQuery::parse("!general x", DISABLED);
        let names: Vec<_> = q.enginerefs.iter().map(|r| r.name).collect();
        assert_eq!(names, ["duckduckgo", "brave", "mojeek", "wikipedia"]);
        let q = RawTextQuery::parse("!it pandas", DISABLED);
        let names: Vec<_> = q.enginerefs.iter().map(|r| r.name).collect();
        assert_eq!(names, ["github", "stackoverflow"]);
        assert_eq!(q.enginerefs[0].category, "it");
    }

    #[test]
    fn unknown_bangs_stay() {
        let q = RawTextQuery::parse("!nope !!g x", DISABLED);
        assert_eq!(q.query(), "!nope !!g x");
        let q = RawTextQuery::parse("!! x", DISABLED);
        assert!(q.redirect_to_first_result);
        assert_eq!(q.query(), "x");
    }

    #[test]
    fn languages() {
        assert_eq!(RawTextQuery::parse(":uk Київ", DISABLED).languages, ["uk"]);
        assert_eq!(RawTextQuery::parse(":de-at x", DISABLED).languages, ["de-AT"]);
        assert_eq!(RawTextQuery::parse(":de_at x", DISABLED).languages, ["de-AT"]);
        let l = RawTextQuery::parse(":german x", DISABLED).languages;
        assert_eq!(l[0], "de");
        assert!(l.len() > 1 && l.iter().all(|x| x.starts_with("de")));
        // valid, though not one of SearXNG's
        assert_eq!(RawTextQuery::parse(":xx x", DISABLED).languages, ["xx"]);
        let q = RawTextQuery::parse(":Ukrainian x", DISABLED);
        // by name every locale of the language matches, as in SearXNG; the first one is the search's
        assert_eq!(q.languages, ["uk", "uk-UA"]);
        assert_eq!(q.query(), "x");
        // not a language: stays in the query
        assert_eq!(RawTextQuery::parse(":) x", DISABLED).query(), ":) x");
    }

    #[test]
    fn timeouts() {
        assert_eq!(RawTextQuery::parse("<3 x", DISABLED).timeout_limit, Some(3.0));
        assert_eq!(RawTextQuery::parse("<850 x", DISABLED).timeout_limit, Some(0.85));
        assert_eq!(RawTextQuery::parse("<x y", DISABLED).query(), "<x y");
    }

    #[test]
    fn guessing_the_language() {
        assert_eq!(guess_lang("Київ погода", None), "uk");
        assert_eq!(guess_lang("погода москва", None), "ru");
        assert_eq!(guess_lang("東京 天気", None), "zh");
        assert_eq!(guess_lang("テスト", None), "ja");
        assert_eq!(guess_lang("python asyncio", None), "en");
        assert_eq!(guess_lang("python asyncio", Some("de")), "de");
    }
}
