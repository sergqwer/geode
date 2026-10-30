// SPDX-License-Identifier: AGPL-3.0-or-later
//! The output contract (the app and the agent's web_search plugin read it): one JSON object —
//! {"query", "answers": [str], "results": [{"url", "title", "content", "engines", "publishedDate"?}],
//! "infoboxes"?: [...], "suggestions"?: [...], "unresponsive": [[engine, reason]]}. Snippets are cut to 300
//! characters, titles to 200; an infobox's summary is also one of the answers (an agent reads answers first).

use crate::results::{MainResult, ResultContainer};
use crate::util::clip;
use serde_json::{Value, json};

pub const SNIPPET: usize = 300;

pub fn build(query: &str, rc: &mut ResultContainer, max: usize) -> Value {
    let ordered: Vec<MainResult> = rc.ordered();
    let mut answers: Vec<String> = Vec::new();
    for a in &rc.answers {
        let mut s = clip(&a.answer, 600);
        if let Some(u) = &a.url {
            s += &format!(" ({u})");
        }
        answers.push(s);
    }
    for b in &rc.infoboxes {
        if !b.content.is_empty() {
            let mut s = clip(&format!("{}: {}", b.title, b.content), 800);
            if !b.id.is_empty() {
                s += &format!(" ({})", b.id);
            }
            answers.push(s);
        }
    }
    let results: Vec<Value> = ordered
        .iter()
        .take(max)
        .map(|r| {
            let mut engines: Vec<&str> = r.engines.clone();
            engines.sort();
            let mut o = json!({"url": r.url, "title": clip(&r.title, 200), "content": clip(&r.content, SNIPPET), "engines": engines});
            if let Some(d) = r.published {
                o["publishedDate"] = json!(d.isoformat());
            }
            o
        })
        .collect();
    let mut out = json!({"query": query, "answers": answers, "results": results});
    if !rc.infoboxes.is_empty() {
        out["infoboxes"] = rc
            .infoboxes
            .iter()
            .map(|b| {
                let mut o =
                    json!({"title": b.title, "content": clip(&b.content, 800), "url": b.id, "engines": b.engines});
                if !b.img_src.is_empty() {
                    o["img_src"] = json!(b.img_src);
                }
                o
            })
            .collect();
    }
    if !rc.suggestions.is_empty() {
        let mut s = rc.suggestions.clone();
        s.sort();
        out["suggestions"] = json!(s);
    }
    let mut un: Vec<(String, String)> = rc.unresponsive.iter().map(|u| (u.engine.clone(), u.reason.clone())).collect();
    un.sort();
    un.dedup();
    out["unresponsive"] = json!(un.into_iter().map(|(e, r)| [e, r]).collect::<Vec<_>>());
    out
}
