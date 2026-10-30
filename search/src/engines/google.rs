// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/google.py: Google's WML page (www.google.com/wml/search), with the language (hl, lr) and country
//! (cr) get_google_info derives from the locale. Upstream sends it with a Nokia User-Agent; geode-search keeps
//! its one browser identity (Chrome on Android) for every request. A "sorry" page is a captcha.

use super::Params;
use crate::html::{attr, parse, sel, text, text_all};
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::results::{Item, MainResult};
use crate::url::{unquote, urlencode, urlparse};

pub fn request(query: &str, p: &Params) -> HttpRequest {
    let r = &p.row;
    let mut args = vec![("q", query.to_string()), ("sca_esv", "1".into())];
    args.push(("hl", r.str("g_hl").into()));
    args.push(("lr", if p.locale == "all" { String::new() } else { r.str("g_lr").into() }));
    args.push(("cr", r.str("g_cr").into()));
    args.push(("ie", "utf8".into()));
    args.push(("oe", "utf8".into()));
    let start = (p.pageno - 1) * 10;
    if start > 0 {
        args.push(("start", start.to_string()));
    }
    if p.safesearch > 0 {
        args.push(("safe", ["off", "medium", "high"][p.safesearch as usize].into()));
    }
    p.request(format!("https://www.google.com/wml/search?{}", urlencode(&args)))
}

/// detect_google_sorry
fn detect_sorry(resp: &HttpResponse) -> Result<(), EngineError> {
    let u = urlparse(&resp.url);
    if u.hostname() == "sorry.google.com"
        || u.path.starts_with("/sorry")
        || resp.status == 302
        || (resp.text.len() < 2000 && resp.text.contains("/sorry/"))
    {
        return Err(EngineError::captcha("CAPTCHA", None));
    }
    Ok(())
}

pub fn unwrap_google_url(raw: &str) -> String {
    match raw.strip_prefix("/url?q=") {
        Some(r) => unquote(r.split("&sa=U").next().unwrap_or(r)),
        None => raw.to_string(),
    }
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    detect_sorry(resp)?;
    let mut t = resp.text.as_str();
    if t.trim_start().starts_with("<?xml") {
        t = t.split_once("?>").map(|x| x.1).unwrap_or(t);
    }
    let doc = parse(t);
    let mut items = Vec::new();
    for result in doc.select(&sel("div[class*=\"zMzFAb\"]")) {
        let Some(title) = result.select(&sel("a[class*=\"fuLhoc\"] span[class*=\"CVA68e\"]")).next() else { continue };
        let Some(raw) = result.select(&sel("a[class*=\"fuLhoc\"]")).next().and_then(|a| attr(a, "href")) else {
            continue;
        };
        let content = text_all(result.select(&sel("div[class*=\"taTFJ\"] span[class*=\"FrIlee\"]")));
        let mut m = MainResult::new(unwrap_google_url(&raw), text(title), content);
        m.thumbnail =
            result.select(&sel("img[src*=\"encrypted-tbn\"]")).next().and_then(|i| attr(i, "src")).unwrap_or_default();
        items.push(Item::Main(m));
    }
    for s in doc.select(&sel("table[class*=\"HExoMb\"] a[class*=\"ZWRArf\"]")) {
        items.push(Item::Suggestion(text(s)));
    }
    Ok(items)
}
