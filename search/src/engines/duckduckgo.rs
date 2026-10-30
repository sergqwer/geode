// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/duckduckgo.py: DuckDuckGo's no-JavaScript page (html.duckduckgo.com/html), a POST of the search
//! form. The region goes in the form (kl) and a cookie; the page's vqd token is kept for the query (a next page
//! would need it). A captcha page suspends nothing (DuckDuckGo does not block the address).

use super::Params;
use crate::html::{attr, parse, sel, text, text_all};
use crate::http::{EngineError, HttpRequest, HttpResponse, Method};
use crate::results::{Item, MainResult};

const DDG_URL: &str = "https://html.duckduckgo.com/html/";

/// quote_ddg_bangs: every !word quoted, so DuckDuckGo searches instead of redirecting (its bang list is left out
/// of geode-search, so any !word counts)
pub fn quote_ddg_bangs(query: &str) -> String {
    query
        .split_whitespace()
        .map(|w| if w.starts_with('!') { format!("'{w}'") } else { w.to_string() })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn request(query: &str, p: &Params) -> Option<HttpRequest> {
    if query.chars().count() >= 500 {
        return None; // DDG does not accept queries with more than 499 characters
    }
    let query = quote_ddg_bangs(query);
    let region = p.row.str("ddg");
    let mut req = p.request(DDG_URL);
    req.method = Method::Post;
    for (k, v) in [
        ("Sec-Fetch-Dest", "document"),
        ("Sec-Fetch-Mode", "navigate"),
        ("Sec-Fetch-Site", "same-origin"),
        ("Sec-Fetch-User", "?1"),
    ] {
        req.set_header(k, v);
    }
    req.form.push(("q".into(), query));
    req.form.push(("b".into(), String::new()));
    req.form.push(("kl".into(), region.to_string()));
    if region != "wt-wt" {
        req.cookies.push(("kl".into(), region.to_string()));
    }
    req.set_header("Content-Type", "application/x-www-form-urlencoded");
    req.set_header("Referer", DDG_URL);
    Some(req)
}

/// The query as the form carries it (the vqd is kept for it).
pub fn form_query(req: &HttpRequest) -> String {
    req.form.iter().find(|f| f.0 == "q").map(|f| f.1.clone()).unwrap_or_default()
}

/// set_vqd's key: the query and the User-Agent the token was given to, hashed (EngineCache.secret_hash)
pub fn vqd_key(query: &str, user_agent: &str) -> String {
    use sha2::{Digest, Sha256};
    crate::util::hex(&Sha256::digest(format!("{query}//{user_agent}").as_bytes())[..16])
}

/// The results, the zero-click answer, and the page's vqd token.
pub fn response(resp: &HttpResponse) -> Result<(Vec<Item>, Option<String>), EngineError> {
    let mut items = Vec::new();
    if resp.status == 303 {
        return Ok((items, None));
    }
    let doc = parse(&resp.text);
    if doc.select(&sel("form#challenge-form")).next().is_some() {
        return Err(EngineError::captcha("CAPTCHA", Some(0)));
    }
    let vqd = doc.select(&sel("input[name=\"vqd\"]")).next().and_then(|i| attr(i, "value"));
    for div in doc.select(&sel("div#links > div[class*=\"web-result\"]")) {
        let a: Vec<_> = div.select(&sel("h2 > a")).collect();
        let Some(href) = a.first().and_then(|a| attr(*a, "href")) else {
            return Err(EngineError::Parse("IndexError: result without a link".into()));
        };
        let content = div.select(&sel("a[class*=\"result__snippet\"]")).next().map(text).unwrap_or_default();
        items.push(Item::Main(MainResult::new(href, text_all(a), content)));
    }
    let zero = text_all(doc.select(&sel("div#zero_click_abstract")));
    if !zero.is_empty()
        && !zero.contains("Your IP address is")
        && !zero.contains("Your user agent:")
        && !zero.contains("URL Decoded:")
        && let Some(url) = doc.select(&sel("div#zero_click_abstract > a")).next().and_then(|a| attr(a, "href"))
    {
        items.push(Item::Answer { answer: zero, url: Some(url) });
    }
    Ok((items, vqd))
}
