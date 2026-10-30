// SPDX-License-Identifier: AGPL-3.0-or-later
//! SearXNG's locales without Babel: src/data/locales.json holds what SearXNG's own code answers for every locale a
//! search can name (tools/search/gen_rust_data.py runs searx.locales / EngineTraits / the engines' helpers over
//! the kept engines' traits): the Accept-Language header and each engine's region and language values. Also
//! SearXNG's list of locales, which the `:lang` query syntax matches against.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct Data {
    sxng_locales: Vec<[String; 4]>,
    fields: Vec<String>,
    locales: HashMap<String, Vec<Option<String>>>,
}

fn data() -> &'static Data {
    static DATA: OnceLock<Data> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("data/locales.json")).expect("locales.json"))
}

/// SearXNG's locales (searx/sxng_locales.py): tag, native name, country, English name.
pub fn sxng_locales() -> &'static [[String; 4]] {
    &data().sxng_locales
}

/// One locale's values, as SearXNG's code gave them.
#[derive(Clone, Debug)]
pub struct Row {
    /// the tag the values are for (the one asked, its language, or "all")
    pub tag: String,
    values: &'static [Option<String>],
}

impl Row {
    pub fn get(&self, field: &str) -> Option<&'static str> {
        let i = data().fields.iter().position(|f| f == field)?;
        self.values.get(i)?.as_deref()
    }

    /// the field's value; "" for a None (Python's `None` where a string is formatted)
    pub fn str(&self, field: &str) -> &'static str {
        self.get(field).unwrap_or("")
    }
}

/// The values for a SearXNG locale tag ("all", "uk", "de-AT", ...): the tag itself, else its spelling with the
/// region in capitals, else its language, else "all" (what an engine's traits answer for a locale they don't know
/// is their default, which is what they answer for "all").
pub fn row(tag: &str) -> Row {
    let locales = &data().locales;
    let mut tries = vec![tag.to_string()];
    if let Some((l, r)) = tag.split_once('-') {
        tries.push(format!("{}-{}", l.to_lowercase(), r.to_uppercase()));
        tries.push(l.to_lowercase());
    } else {
        tries.push(tag.to_lowercase());
    }
    tries.push("all".into());
    for t in tries {
        if let Some((k, v)) = locales.get_key_value(t.as_str()) {
            return Row { tag: k.clone(), values: v.as_slice() };
        }
    }
    unreachable!("locales.json has \"all\"")
}

/// OnlineProcessor.get_params: the Accept-Language header for the search's locale. A tag Babel knows is in the
/// table; for another valid one the same rule (language, language-territory) is applied.
pub fn accept_language(tag: &str) -> String {
    if let Some(v) = data().locales.get(tag)
        && let Some(Some(al)) = v.first()
    {
        return al.clone();
    }
    if tag == "all" || !crate::query::valid_language_code(tag) {
        return "en-US,en;q=0.9".into();
    }
    let (l, t) = match tag.split_once('-') {
        Some((l, t)) => (l.to_lowercase(), t.to_uppercase()),
        None => (tag.to_lowercase(), tag.to_lowercase()),
    };
    format!("{l},{l}-{t};q=0.7,en;q=0.3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_from_searxng() {
        let all = row("all");
        assert_eq!(all.str("ddg"), "wt-wt");
        assert_eq!(all.str("g_hl"), "ZZ");
        let uk = row("uk");
        assert_eq!(uk.str("ddg"), "ua-uk");
        assert_eq!(uk.str("wp"), "uk.wikipedia.org");
        assert_eq!(row("de-at").tag, "de-AT");
        assert_eq!(row("de-AT").str("g_cr"), "countryAT");
        // an unknown region falls back to its language
        assert_eq!(row("uk-XX").tag, "uk");
        assert_eq!(row("qq").tag, "all");
    }

    #[test]
    fn accept_language_header() {
        assert_eq!(accept_language("all"), "en-US,en;q=0.9");
        assert_eq!(accept_language("uk"), "uk,uk-uk;q=0.7,en;q=0.3");
        assert_eq!(accept_language("de-AT"), "de,de-AT;q=0.7,en;q=0.3");
    }

    #[test]
    fn searxng_locale_list() {
        assert!(sxng_locales().iter().any(|l| l[0] == "uk" && l[3] == "Ukrainian"));
    }
}
