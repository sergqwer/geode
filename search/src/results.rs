// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/results.py and searx/result_types: what an engine gives back, and the ResultContainer that merges the
//! engines' results (a URL found by several engines is one result), scores them (engine weights, the positions,
//! how many engines found it) and orders them (score first, then results of a kind grouped).

use crate::date::DateTime;
use crate::settings;
use crate::url::{Url, normalize_result_url};
use crate::util::normalize_text_field;
use std::collections::HashMap;

/// MAXIMUM_TITLE_LENGTH_CHARS / MAXIMUM_CONTENT_LENGTH_CHARS
const MAX_TITLE: usize = 200;
const MAX_CONTENT: usize = 1200;

/// A main result (MainResult, or a legacy dict result) as an engine gives it.
#[derive(Clone, Debug, Default)]
pub struct MainResult {
    pub url: String,
    pub title: String,
    pub content: String,
    /// default.html, packages.html (GitHub), paper.html (arXiv), videos.html (Brave's videos)
    pub template: &'static str,
    pub thumbnail: String,
    pub img_src: String,
    pub published: Option<DateTime>,
    // set by the container
    pub engine: &'static str,
    pub engines: Vec<&'static str>,
    pub positions: Vec<usize>,
    pub score: f64,
    pub category: &'static str,
    parsed: Url,
}

impl MainResult {
    pub fn new(url: impl Into<String>, title: impl Into<String>, content: impl Into<String>) -> MainResult {
        MainResult {
            url: url.into(),
            title: title.into(),
            content: content.into(),
            template: "default.html",
            ..Default::default()
        }
    }

    /// MainResult.__hash__: template, the URL without its scheme, img_src
    fn key(&self) -> String {
        let u = &self.parsed;
        format!("{}|{}|{}|{}|{}|{}|{}", self.template, u.netloc, u.path, u.params, u.query, u.fragment, self.img_src)
    }
}

/// An infobox (Wikipedia's summary of the page the query names).
#[derive(Clone, Debug, Default)]
pub struct Infobox {
    pub title: String,
    /// the page's URL, which identifies the infobox
    pub id: String,
    pub content: String,
    pub img_src: String,
    pub urls: Vec<(String, String)>,
    pub engine: &'static str,
    pub engines: Vec<&'static str>,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // a few dozen per search
pub enum Item {
    Main(MainResult),
    Answer { answer: String, url: Option<String> },
    Infobox(Infobox),
    Suggestion(String),
    Correction(String),
}

#[derive(Clone, Debug)]
pub struct Answer {
    pub answer: String,
    pub url: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unresponsive {
    pub engine: String,
    pub reason: String,
    pub suspended: bool,
}

#[derive(Default)]
pub struct ResultContainer {
    main: Vec<MainResult>,
    index: HashMap<String, usize>,
    pub infoboxes: Vec<Infobox>,
    pub suggestions: Vec<String>,
    pub corrections: Vec<String>,
    pub answers: Vec<Answer>,
    pub unresponsive: Vec<Unresponsive>,
    pub timings: Vec<(String, f64)>,
    closed: bool,
}

fn weight(engine: &str) -> f64 {
    settings::engine(engine).map(|e| e.weight).unwrap_or(1.0)
}

impl ResultContainer {
    pub fn new() -> ResultContainer {
        ResultContainer::default()
    }

    /// ResultContainer.extend: one engine's results, in its order (the positions count its main results).
    pub fn extend(&mut self, engine: &'static str, items: Vec<Item>) {
        if self.closed {
            return;
        }
        let mut main_count = 0;
        for item in items {
            match item {
                Item::Main(mut r) => {
                    if r.url.is_empty() {
                        continue; // a result without a URL cannot be told apart from another: dropped
                    }
                    r.engine = engine;
                    normalize(&mut r);
                    main_count += 1;
                    self.merge_main(r, main_count);
                }
                Item::Answer { answer, url } => {
                    // AnswerSet: one per answer text
                    if !self.answers.iter().any(|a| a.answer == answer) {
                        self.answers.push(Answer { answer, url });
                    }
                }
                Item::Infobox(mut b) => {
                    b.engine = engine;
                    b.engines = vec![engine];
                    b.title = normalize_text_field(&b.title, MAX_TITLE);
                    b.content = normalize_text_field(&b.content, MAX_CONTENT);
                    if !b.id.is_empty() {
                        b.id = normalize_result_url(&b.id).0;
                    }
                    self.merge_infobox(b);
                }
                Item::Suggestion(s) => {
                    if !self.suggestions.contains(&s) {
                        self.suggestions.push(s);
                    }
                }
                Item::Correction(s) => {
                    if !self.corrections.contains(&s) {
                        self.corrections.push(s);
                    }
                }
            }
        }
    }

    fn merge_main(&mut self, r: MainResult, position: usize) {
        let key = r.key();
        match self.index.get(&key) {
            None => {
                let mut r = r;
                r.positions = vec![position];
                self.index.insert(key, self.main.len());
                self.main.push(r);
            }
            Some(&i) => {
                let origin = &mut self.main[i];
                merge_two_main_results(origin, &r);
                origin.positions.push(position);
            }
        }
    }

    /// _merge_infobox / merge_two_infoboxes
    fn merge_infobox(&mut self, b: Infobox) {
        if !b.id.is_empty()
            && let Some(o) = self.infoboxes.iter_mut().find(|o| o.id == b.id)
        {
            let (w1, w2) = (weight(o.engine), weight(b.engine));
            if w2 > w1 {
                o.engine = b.engine;
            }
            for e in &b.engines {
                if !o.engines.contains(e) {
                    o.engines.push(e);
                }
            }
            for u in b.urls {
                if !o.urls.iter().any(|x| x.1 == u.1) {
                    o.urls.push(u);
                }
            }
            if !b.img_src.is_empty() && (o.img_src.is_empty() || w2 > w1) {
                o.img_src = b.img_src;
            }
            if b.content.chars().count() > o.content.chars().count() {
                o.content = b.content;
            }
            return;
        }
        self.infoboxes.push(b);
    }

    pub fn add_unresponsive(&mut self, engine: &str, reason: &str, suspended: bool) {
        if self.closed {
            return;
        }
        let u = Unresponsive { engine: engine.into(), reason: reason.into(), suspended };
        if !self.unresponsive.contains(&u) {
            self.unresponsive.push(u);
        }
    }

    /// close: infoboxes with an image first, then every result's score.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.infoboxes.sort_by_key(|b| b.img_src.is_empty());
        for r in &mut self.main {
            r.score = calculate_score(r);
        }
    }

    /// get_ordered_results: by score, then a result joins the group of its kind (category, template, picture or
    /// not) when that group is near (8 per group, 20 places at most).
    pub fn ordered(&mut self) -> Vec<MainResult> {
        self.close();
        let mut results = self.main.clone();
        // sorted(..., reverse=True) keeps equal scores in their order
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut out: Vec<MainResult> = Vec::with_capacity(results.len());
        let mut groups: Vec<(String, usize, i32)> = Vec::new(); // (category, index, count)
        const MAX_COUNT: i32 = 8;
        const MAX_DISTANCE: usize = 20;
        for mut res in results {
            res.category = settings::engine(res.engine).and_then(|e| e.categories.first().copied()).unwrap_or("");
            let has_img = !res.thumbnail.is_empty() || !res.img_src.is_empty();
            let category = format!("{}:{}:{}", res.category, res.template, if has_img { "img_src" } else { "" });
            let grp = groups.iter().position(|g| g.0 == category);
            match grp {
                Some(gi) if groups[gi].2 > 0 && out.len() - groups[gi].1 < MAX_DISTANCE => {
                    let index = groups[gi].1;
                    out.insert(index, res);
                    for g in groups.iter_mut() {
                        if g.1 >= index {
                            g.1 += 1;
                        }
                    }
                    groups[gi].2 -= 1;
                }
                _ => {
                    out.push(res);
                    let entry = (category, out.len(), MAX_COUNT);
                    match grp {
                        Some(gi) => groups[gi] = entry,
                        None => groups.push(entry),
                    }
                }
            }
        }
        out
    }
}

/// MainResult.normalize_result_fields: the URL (IDN host, http if no scheme), whitespace and length of title and
/// content (no content that only repeats the title), the engine among the result's engines.
fn normalize(r: &mut MainResult) {
    let (url, parsed) = normalize_result_url(&r.url);
    r.url = url;
    r.parsed = parsed;
    r.title = normalize_text_field(&r.title, MAX_TITLE);
    r.content = normalize_text_field(&r.content, MAX_CONTENT);
    if r.content == r.title {
        r.content.clear();
    }
    if !r.engines.contains(&r.engine) {
        r.engines.push(r.engine);
    }
}

/// merge_two_main_results: the longer content and title win, the engines are joined, https wins over http.
fn merge_two_main_results(origin: &mut MainResult, other: &MainResult) {
    if other.content.chars().count() > origin.content.chars().count() {
        origin.content = other.content.clone();
    }
    if other.title.chars().count() > origin.title.chars().count() {
        origin.title = other.title.clone();
    }
    if !origin.engines.contains(&other.engine) {
        origin.engines.push(other.engine);
    }
    if !origin.parsed.scheme.ends_with('s') && other.parsed.scheme.ends_with('s') {
        origin.parsed.scheme = other.parsed.scheme.clone();
        origin.url = origin.parsed.geturl();
    }
}

/// calculate_score (priority "" for every result here: no hostnames plugin)
fn calculate_score(r: &MainResult) -> f64 {
    let mut w = 1.0;
    for e in &r.engines {
        w *= weight(e);
    }
    w *= r.positions.len() as f64;
    r.positions.iter().map(|p| w / *p as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(url: &str, title: &str) -> Item {
        Item::Main(MainResult::new(url, title, ""))
    }

    #[test]
    fn merging_and_scoring() {
        let mut c = ResultContainer::new();
        c.extend("brave", vec![m("https://a.org/x", "A"), m("https://b.org/", "B"), m("https://c.org/", "C")]);
        c.extend("duckduckgo", vec![m("http://c.org/", "C longer"), m("https://a.org/x", "A")]);
        let r = c.ordered();
        let urls: Vec<&str> = r.iter().map(|x| x.url.as_str()).collect();
        // a: 2 engines, positions 1 and 2 -> 2/1 + 2/2 = 3; c: 2/3 + 2/1 = 2.67; b: 1/2
        assert_eq!(urls, ["https://a.org/x", "https://c.org/", "https://b.org/"]);
        assert_eq!(r[1].title, "C longer");
        assert_eq!(r[1].engines, ["brave", "duckduckgo"]);
        assert!((r[0].score - 3.0).abs() < 1e-9);
    }

    #[test]
    fn grouping_by_kind() {
        // GitHub results (it:packages.html with a picture) gather after the first of them
        let mut c = ResultContainer::new();
        let mut items = vec![];
        for i in 0..3 {
            let mut g = MainResult::new(format!("https://github.com/{i}"), format!("g{i}"), "");
            g.template = "packages.html";
            g.thumbnail = "https://avatars/x".into();
            items.push(Item::Main(g));
        }
        c.extend("github", items);
        c.extend("stackoverflow", vec![m("https://so.com/1", "s1"), m("https://so.com/2", "s2")]);
        let r = c.ordered();
        let t: Vec<&str> = r.iter().map(|x| x.title.as_str()).collect();
        // scores g0 1, s1 1, g1 .5, s2 .5, g2 .33: each GitHub result joins the group its first one opened
        assert_eq!(t, ["g0", "g1", "g2", "s1", "s2"]);
    }

    #[test]
    fn text_fields_and_answers() {
        let mut c = ResultContainer::new();
        c.extend(
            "x",
            vec![
                Item::Main(MainResult::new("https://xn--strung-xxa.de/", " same\n title ", "same title")),
                Item::Answer { answer: "42".into(), url: None },
                Item::Answer { answer: "42".into(), url: None },
                Item::Suggestion("s".into()),
            ],
        );
        let r = c.ordered();
        assert_eq!(r[0].url, "https://störung.de/");
        assert_eq!(r[0].title, "same title");
        assert_eq!(r[0].content, "");
        assert_eq!(c.answers.len(), 1);
        assert_eq!(c.suggestions, ["s"]);
    }
}
