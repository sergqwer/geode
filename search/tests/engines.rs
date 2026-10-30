// SPDX-License-Identifier: AGPL-3.0-or-later
//! The engines' parsers on pages the phone fetched (tests/fixtures: geode-search --dump on a Redmi Note 9 Pro,
//! Chrome-on-Android identity, 2026-10-28), and the whole merge of a general search from them.

use geode_search::engines::{
    Params, arxiv, bing, brave, duckduckgo, github, google, mojeek, pypi, stackexchange, startpage, wikipedia,
};
use geode_search::http::{EngineError, HttpResponse, raise_for_httperror};
use geode_search::results::{Item, MainResult, ResultContainer};

/// A dump: its first line `<!-- status url -> final url -->`, then the body.
fn fixture(name: &str) -> HttpResponse {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let t = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (head, body) = t.split_once('\n').unwrap();
    let parts: Vec<&str> = head.trim_start_matches("<!-- ").trim_end_matches(" -->").split(' ').collect();
    HttpResponse {
        status: parts[0].parse().unwrap(),
        url: parts[3].to_string(),
        text: body.to_string(),
        ..Default::default()
    }
}

fn mains(items: &[Item]) -> Vec<&MainResult> {
    items.iter().filter_map(|i| if let Item::Main(m) = i { Some(m) } else { None }).collect()
}

#[test]
fn duckduckgo_results_and_vqd() {
    let (items, vqd) = duckduckgo::response(&fixture("ddg_en.html")).unwrap();
    let m = mains(&items);
    assert_eq!(m.len(), 10);
    assert!(m.iter().all(|r| r.url.starts_with("http") && !r.title.is_empty()));
    assert!(m.iter().any(|r| r.url == "https://docs.python.org/3/library/asyncio-task.html"));
    assert!(m.iter().filter(|r| !r.content.is_empty()).count() >= 8);
    assert!(vqd.is_some_and(|v| !v.is_empty()));
    let (items, _) = duckduckgo::response(&fixture("ddg_uk.html")).unwrap();
    assert!(mains(&items).iter().any(|r| r.url.contains("sinoptik") || r.title.contains("Погода")));
}

#[test]
fn duckduckgo_request() {
    let p = Params::new("uk", "general");
    let r = duckduckgo::request("!w Київ", &p).unwrap();
    assert_eq!(r.url, "https://html.duckduckgo.com/html/");
    assert_eq!(
        r.form,
        [("q".to_string(), "'!w' Київ".to_string()), ("b".into(), "".into()), ("kl".into(), "ua-uk".into())]
    );
    assert_eq!(r.cookies, [("kl".to_string(), "ua-uk".to_string())]);
    assert!(duckduckgo::request(&"x".repeat(500), &p).is_none());
}

#[test]
fn duckduckgo_captcha_is_not_suspended() {
    let resp = HttpResponse {
        status: 200,
        text: "<html><form id=\"challenge-form\"></form></html>".into(),
        ..Default::default()
    };
    let e = duckduckgo::response(&resp).unwrap_err();
    assert_eq!(e.reason(), "SearxEngineCaptchaException");
    assert_eq!(e.suspend_seconds(), Some(0));
}

#[test]
fn brave_payload() {
    let items = brave::response(&fixture("brave_en.html")).unwrap();
    let m = mains(&items);
    let web: Vec<_> = m.iter().filter(|r| r.template == "default.html").collect();
    assert!(web.len() >= 10, "{}", web.len());
    assert!(web.iter().all(|r| r.url.starts_with("https://") && !r.title.is_empty()));
    assert!(web.iter().any(|r| r.url.contains("docs.python.org")));
    assert!(web.iter().filter(|r| r.published.is_some()).count() >= 3);
}

#[test]
fn brave_request_cookies() {
    let r = brave::request("rust", &Params::new("de-AT", "general"));
    assert_eq!(r.url, "https://search.brave.com/search?q=rust&source=web");
    let c: Vec<(&str, &str)> = r.cookies.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    assert!(c.contains(&("country", "at")) && c.contains(&("ui_lang", "de-de")) && c.contains(&("safesearch", "off")));
}

#[test]
fn mojeek_results() {
    let items = mojeek::response(&fixture("mojeek_en.html")).unwrap();
    let m = mains(&items);
    assert!(m.len() >= 8, "{}", m.len());
    assert!(m.iter().all(|r| r.url.starts_with("http") && !r.title.is_empty()));
    // a page Mojeek has no snippet for (its first result here) comes without one
    assert!(m.iter().filter(|r| !r.content.is_empty()).count() >= m.len() - 2);
    assert_eq!(m[1].title, "python - Why asyncio.TimeoutError is raised? - Stack Overflow");
    // "Sorry, Mojeek doesn't currently search pages in this language": an answer with nothing in it
    let items = mojeek::response(&fixture("mojeek_uk.html")).unwrap();
    assert!(mains(&items).is_empty());
}

#[test]
fn mojeek_challenge_shape() {
    let v = fixture("mojeek_challenge.json");
    let j: serde_json::Value = serde_json::from_str(&v.text).unwrap();
    let p = &j["parameters"];
    // the solver reads these fields; the real search solved it on the phone
    for k in ["nonce", "salt", "keyPrefix", "cost", "keyLength", "algorithm"] {
        assert!(!p[k].is_null(), "{k}");
    }
    assert_eq!(p["algorithm"], "PBKDF2/SHA-256");
}

#[test]
fn wikipedia_infobox_and_missing_page() {
    let items = wikipedia::response(&fixture("wikipedia_kyiv.json")).unwrap();
    let Item::Infobox(b) = &items[0] else { panic!("{items:?}") };
    assert_eq!(b.title, "Kyiv");
    assert_eq!(b.id, "https://en.wikipedia.org/wiki/Kyiv");
    assert!(b.content.starts_with("Kyiv, also spelt Kiev, is the capital"));
    assert!(wikipedia::response(&fixture("wikipedia_404.json")).unwrap().is_empty());
    // a lower case query is title-cased, on the Wikipedia of the locale
    let r = wikipedia::request("київ", &Params::new("uk", "general"));
    assert_eq!(r.url, "https://uk.wikipedia.org/api/rest_v1/page/summary/%D0%9A%D0%B8%D1%97%D0%B2");
}

#[test]
fn github_api() {
    let items = github::response(&fixture("github.json")).unwrap();
    let m = mains(&items);
    assert_eq!(m.len(), 30);
    assert_eq!(m[0].url, "https://github.com/deepseek-ai/deepseek-harness");
    assert_eq!(m[0].template, "packages.html");
    assert!(!m[0].thumbnail.is_empty() && m[0].published.is_some());
}

#[test]
fn stackoverflow_api() {
    let items = stackexchange::response(&fixture("stackoverflow.json"), "stackoverflow").unwrap();
    let m = mains(&items);
    assert_eq!(m.len(), 10);
    assert!(m[0].url.starts_with("https://stackoverflow.com/q/"));
    assert!(m[0].content.starts_with('[') && m[0].content.contains("// score: "));
}

#[test]
fn arxiv_feed() {
    let items = arxiv::response(&fixture("arxiv.xml")).unwrap();
    let m = mains(&items);
    assert_eq!(m.len(), 10);
    assert!(m[0].url.starts_with("http://arxiv.org/abs/"));
    assert_eq!(m[0].template, "paper.html");
    assert!(m[0].published.unwrap().offset.is_none());
    assert!(m[0].content.len() > 100);
}

#[test]
fn pypi_json() {
    let items = pypi::response(&fixture("pypi.json")).unwrap();
    let m = mains(&items);
    assert_eq!(m[0].title, "curl-cffi 0.17.0");
    assert_eq!(m[0].url, "https://pypi.org/project/curl-cffi/");
    assert_eq!(m[0].published.unwrap().isoformat(), "2026-10-26T04:12:42.896562+00:00");
    let missing = HttpResponse { status: 404, ..Default::default() };
    assert!(pypi::response(&missing).unwrap().is_empty());
}

#[test]
fn bing_mobile_markup() {
    let items = bing::response(&fixture("bing.html")).unwrap();
    let m = mains(&items);
    assert!(m.len() >= 4, "{}", m.len());
    assert_eq!(m[0].url, "https://www.python.org/");
    assert_eq!(m[0].title, "Welcome to Python.org");
    assert!(m[0].content.starts_with("Experienced programmers"));
}

#[test]
fn google_403_is_access_denied() {
    let e = raise_for_httperror(&fixture("google_403.html")).unwrap_err();
    assert_eq!(e.suspend_seconds(), Some(86400));
    let sorry = HttpResponse { status: 302, url: "https://www.google.com/sorry/index".into(), ..Default::default() };
    assert!(matches!(google::response(&sorry), Err(EngineError::Captcha { .. })));
    let r = google::request("rust", &Params::new("de-AT", "general"));
    assert_eq!(
        r.url,
        "https://www.google.com/wml/search?q=rust&sca_esv=1&hl=de&lr=lang_de&cr=countryAT&ie=utf8&oe=utf8"
    );
}

#[test]
fn startpage_anubis_page() {
    let t = fixture("startpage_anubis.html").text;
    assert!(t.contains("id=\"anubis_challenge\""));
    let raw = startpage::extr(&t, "<script id=\"anubis_challenge\" type=\"application/json\">", "</script>");
    let v: serde_json::Value = serde_json::from_str(raw).unwrap();
    // difficulty 6 > 4: SearXNG gives up (Anubis difficulty too high)
    assert!(v["rules"]["difficulty"].as_u64().unwrap() > 4);
}

#[test]
fn a_general_search_merged() {
    // the four default engines' pages for "python asyncio timeout", merged as the search does
    let mut rc = ResultContainer::new();
    let (ddg, _) = duckduckgo::response(&fixture("ddg_en.html")).unwrap();
    rc.extend("brave", brave::response(&fixture("brave_en.html")).unwrap());
    rc.extend("duckduckgo", ddg);
    rc.extend("mojeek", mojeek::response(&fixture("mojeek_en.html")).unwrap());
    let out = geode_search::output::build("python asyncio timeout", &mut rc, 10);
    let results = out["results"].as_array().unwrap();
    assert_eq!(results.len(), 10);
    // the page two engines found comes first
    assert!(results[0]["engines"].as_array().unwrap().len() >= 2, "{}", results[0]);
    for r in results {
        assert!(r["content"].as_str().unwrap().chars().count() <= 300);
        assert!(r["title"].as_str().unwrap().chars().count() <= 200);
    }
}
