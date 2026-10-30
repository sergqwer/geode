// SPDX-License-Identifier: AGPL-3.0-or-later
//! The port against SearXNG itself: tests/fixtures/parity.json is what SearXNG's Python (its engine parsers and
//! ResultContainer, tools/search/parity.py over the slim tree) makes of the same saved pages; the Rust merge must
//! give the same results, in the same order, with the same scores.

use geode_search::engines::{arxiv, brave, duckduckgo, github, mojeek, stackexchange};
use geode_search::http::HttpResponse;
use geode_search::results::{Item, ResultContainer};
use serde_json::Value;

fn fixture(name: &str) -> HttpResponse {
    let t = std::fs::read_to_string(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let (head, body) = t.split_once('\n').unwrap();
    let parts: Vec<&str> = head.trim_start_matches("<!-- ").trim_end_matches(" -->").split(' ').collect();
    HttpResponse {
        status: parts[0].parse().unwrap(),
        url: parts[3].to_string(),
        text: body.to_string(),
        ..Default::default()
    }
}

fn expected(case: &str) -> Value {
    let t = std::fs::read_to_string(format!("{}/tests/fixtures/parity.json", env!("CARGO_MANIFEST_DIR"))).unwrap();
    serde_json::from_str::<Value>(&t).unwrap()[case].clone()
}

fn compare(case: &str, mut rc: ResultContainer) {
    let want = expected(case);
    let got = rc.ordered();
    let want_r = want["results"].as_array().unwrap();
    assert_eq!(got.len(), want_r.len(), "{case}: how many");
    for (i, (g, w)) in got.iter().zip(want_r).enumerate() {
        let mut engines: Vec<&str> = g.engines.clone();
        engines.sort();
        assert_eq!(g.url, w["url"].as_str().unwrap(), "{case} #{i}: url");
        assert_eq!(g.title, w["title"].as_str().unwrap(), "{case} #{i}: title");
        assert_eq!(
            engines,
            w["engines"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect::<Vec<_>>(),
            "{case} #{i}: engines"
        );
        assert!(
            (g.score - w["score"].as_f64().unwrap()).abs() < 1e-5,
            "{case} #{i}: score {} vs {}",
            g.score,
            w["score"]
        );
    }
    let mut sugg = rc.suggestions.clone();
    sugg.sort();
    assert_eq!(
        sugg,
        want["suggestions"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>(),
        "{case}: suggestions"
    );
}

#[test]
fn general_search_as_searxng_merges_it() {
    let mut rc = ResultContainer::new();
    rc.extend("brave", brave::response(&fixture("brave_en.html")).unwrap());
    rc.extend("duckduckgo", duckduckgo::response(&fixture("ddg_en.html")).unwrap().0);
    rc.extend("mojeek", mojeek::response(&fixture("mojeek_en.html")).unwrap());
    compare("general", rc);
}

#[test]
fn it_category_as_searxng_merges_it() {
    let mut rc = ResultContainer::new();
    rc.extend("stackoverflow", stackexchange::response(&fixture("stackoverflow.json"), "stackoverflow").unwrap());
    rc.extend("github", github::response(&fixture("github.json")).unwrap());
    compare("it", rc);
}

#[test]
fn arxiv_as_searxng_reads_it() {
    let mut rc = ResultContainer::new();
    let items: Vec<Item> = arxiv::response(&fixture("arxiv.xml")).unwrap();
    rc.extend("arxiv", items);
    compare("arxiv", rc);
}
