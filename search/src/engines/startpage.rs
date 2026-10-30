// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/startpage.py (web): Startpage's search form POSTed as its own page does, with the "sc" code
//! scraped from its home page (kept an hour) and the preferences cookie; an Anubis proof-of-work page in front of
//! either is solved (SHA-256, difficulty up to 4) and its pass cookie kept four minutes. The results are a JSON
//! object in the page's React call.

use super::{Ctx, Params, js};
use crate::date::{DateTime, from_unix, parse_day_month_year};
use crate::html::{attr, html_to_text, parse, sel};
use crate::http::{EngineError, HttpRequest, HttpResponse, Method};
use crate::results::{Item, MainResult};
use crate::url::urlencode;
use crate::util::hex;
use sha2::{Digest, Sha256};

const BASE_URL: &str = "https://www.startpage.com";
const SEARCH_URL: &str = "https://www.startpage.com/sp/search";
const SC_CODE_CACHE_SEC: u64 = 3600;
const MAX_DIFFICULTY: u32 = 4;

pub async fn search(ctx: &Ctx<'_>, name: &str, query: &str, p: &Params) -> Result<Vec<Item>, EngineError> {
    let sc = get_sc_code(ctx, name, p).await?;
    let auth = ctx.cache.get(name, "SPCHAL_AUTH");
    let req = request(query, p, &sc, auth.as_deref());
    let mut resp = ctx.send(name, req.clone(), true).await?;
    if resp.text.contains("id=\"anubis_challenge\"") {
        let auth = solve_anubis(ctx, name, p, &resp).await?;
        let req = req.cookie("spchal-auth", auth);
        resp = ctx.send(name, req, true).await?;
        if resp.text.contains("id=\"anubis_challenge\"") {
            return Err(EngineError::captcha("CAPTCHA", None));
        }
    }
    response(&resp)
}

pub fn request(query: &str, p: &Params, sc: &str, auth: Option<&str>) -> HttpRequest {
    let region = p.row.get("sp_r").unwrap_or("en_US");
    let language = p.row.get("sp_l").unwrap_or("english");
    let safe = ["none", "moderate", "heavy"][p.safesearch.min(2) as usize];
    let mut form: Vec<(String, String)> = [
        ("query", query),
        ("cat", "web"),
        ("t", "device"),
        ("sc", sc),
        ("with_date", ""),
        ("abd", "1"),
        ("abe", "1"),
        ("qsr", "all"),
        ("qadf", safe),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    if !language.is_empty() {
        form.push(("language".into(), language.into()));
        form.push(("lui".into(), language.into()));
    }
    form.push(("segment".into(), "startpage.udog".into()));
    if p.pageno > 1 {
        form.push(("page".into(), p.pageno.to_string()));
    }
    let mut cookie: Vec<(&str, String)> = vec![
        ("date_time", "world".into()),
        ("disable_family_filter", safe.into()),
        ("disable_open_in_new_window", "0".into()),
        ("enable_post_method", "1".into()),
        ("enable_proxy_safety_suggest", "1".into()),
        ("enable_stay_control", "1".into()),
        ("instant_answers", "1".into()),
        ("lang_homepage", "s/device/en/".into()),
        ("num_of_results", "10".into()),
        ("suggestions", "1".into()),
        ("wt_unit", "celsius".into()),
    ];
    if !language.is_empty() {
        cookie.push(("language", language.into()));
        cookie.push(("language_ui", language.into()));
    }
    if !region.is_empty() {
        cookie.push(("search_results_region", region.into()));
    }
    let prefs = cookie.iter().map(|(k, v)| format!("{k}EEE{v}")).collect::<Vec<_>>().join("N1N");
    let mut req = p.request(SEARCH_URL).header("Origin", BASE_URL).header("Referer", format!("{BASE_URL}/"));
    req.method = Method::Post;
    req.form = form;
    req = req.cookie("preferences", prefs);
    if let Some(a) = auth {
        req = req.cookie("spchal-auth", a);
    }
    req
}

/// get_sc_code: the "sc" argument of Startpage's search form, from its home page
async fn get_sc_code(ctx: &Ctx<'_>, name: &str, p: &Params) -> Result<String, EngineError> {
    if let Some(sc) = ctx.cache.get(name, "SC_CODE") {
        return Ok(sc);
    }
    let mut req = p.request(format!("{BASE_URL}/"));
    req.follow_redirects = true;
    let mut resp = ctx.send(name, req.clone(), false).await?;
    if resp.text.contains("id=\"anubis_challenge\"") {
        let auth = solve_anubis(ctx, name, p, &resp).await?;
        resp = ctx.send(name, req.cookie("spchal-auth", auth), false).await?;
    }
    if resp.url.starts_with("https://www.startpage.com/sp/captcha") {
        return Err(EngineError::captcha("get_sc_code: got redirected to https://www.startpage.com/sp/captcha", None));
    }
    let doc = parse(&resp.text);
    let sc =
        doc.select(&sel("form#search input[name=\"sc\"]")).next().and_then(|i| attr(i, "value")).ok_or_else(|| {
            EngineError::captcha(
                format!("get_sc_code: [PR-695] querying new sc timestamp failed! ({})", resp.url),
                None,
            )
        })?;
    ctx.cache.set(name, "SC_CODE", &sc, Some(SC_CODE_CACHE_SEC));
    Ok(sc)
}

/// The nonce whose SHA-256 of randomData + nonce starts with `difficulty` zeros (None after 8x the average).
pub fn anubis_nonce(random_data: &str, difficulty: u32) -> Option<(u64, String)> {
    let prefix = "0".repeat(difficulty as usize);
    for nonce in 0..16u64.pow(difficulty) * 8 {
        let digest = hex(&Sha256::new()
            .chain_update(random_data.as_bytes())
            .chain_update(nonce.to_string().as_bytes())
            .finalize());
        if digest.starts_with(&prefix) {
            return Some((nonce, digest));
        }
    }
    None
}

/// _solve_anubis: the pass cookie for the challenge in a page
async fn solve_anubis(ctx: &Ctx<'_>, name: &str, p: &Params, resp: &HttpResponse) -> Result<String, EngineError> {
    let raw = extr(&resp.text, "<script id=\"anubis_challenge\" type=\"application/json\">", "</script>");
    let payload: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| EngineError::Parse(format!("JSONDecodeError: {e}")))?;
    let challenge = payload.get("challenge").cloned().unwrap_or_default();
    let difficulty = payload
        .get("rules")
        .and_then(|r| r.get("difficulty"))
        .and_then(|d| d.as_u64().or_else(|| d.as_str().and_then(|s| s.parse().ok())))
        .unwrap_or(99) as u32;
    if difficulty > MAX_DIFFICULTY {
        return Err(EngineError::captcha("startpage: Anubis difficulty too high", None));
    }
    let (nonce, digest) = anubis_nonce(&js(&challenge, "randomData"), difficulty)
        .ok_or_else(|| EngineError::captcha("startpage: Anubis failed", None))?;
    let args = [
        ("id", js(&challenge, "id")),
        ("response", digest),
        ("nonce", nonce.to_string()),
        ("redir", resp.url.clone()),
        ("elapsedTime", "1".into()),
    ];
    let mut req = p.request(format!("{BASE_URL}/.within.website/x/cmd/anubis/api/pass-challenge?{}", urlencode(&args)));
    req.cookies = resp.cookies.clone();
    let pass = ctx.send(name, req, false).await?;
    let auth = pass
        .cookie("spchal-auth")
        .ok_or_else(|| EngineError::captcha("startpage: Anubis pass-challenge failed", None))?
        .to_string();
    ctx.cache.set(name, "SPCHAL_AUTH", &auth, Some(240));
    Ok(auth)
}

/// searx.utils.extr: the text between two markers ("" if either is missing)
pub fn extr<'a>(txt: &'a str, begin: &str, end: &str) -> &'a str {
    let Some(i) = txt.find(begin) else { return "" };
    let first = i + begin.len();
    match txt[first..].find(end) {
        Some(j) => &txt[first..first + j],
        None => "",
    }
}

/// _parse_published_date: "5 Mar 2024 ... " or "3 days ago ... " in front of a snippet
fn parse_published_date(content: &str) -> (String, Option<DateTime>) {
    let Some(pos) = content.find(" ... ") else { return (content.to_string(), None) };
    let head = &content[..pos];
    let rest = content[pos + 5..].to_string();
    let words: Vec<&str> = head.split(' ').collect();
    if words.len() == 3
        && words[0].parse::<u32>().is_ok_and(|d| (1..=31).contains(&d))
        && words[1].len() == 3
        && words[1].chars().next().is_some_and(char::is_uppercase)
        && words[2].len() == 4
        && words[2].bytes().all(|b| b.is_ascii_digit())
    {
        return (rest, parse_day_month_year(head));
    }
    if words.len() == 3
        && words[0].bytes().all(|b| b.is_ascii_digit())
        && (words[1] == "day" || words[1] == "days")
        && words[2] == "ago"
    {
        let days: i64 = words[0].parse().unwrap_or(0);
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
        return (rest, Some(from_unix(now - days * 86400)));
    }
    (content.to_string(), None)
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    let raw = format!("{{{}}}}}", extr(&resp.text, "React.createElement(UIStartpage.AppSerpWeb, {", "}})"));
    if resp.header("location").is_some_and(|l| l.starts_with("https://www.startpage.com/sp/captcha")) {
        return Err(EngineError::captcha("CAPTCHA", None));
    }
    let v: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| EngineError::Parse(format!("JSONDecodeError: {e}")))?;
    let regions =
        v.get("render").and_then(|r| r.get("presenter")).and_then(|p| p.get("regions")).cloned().unwrap_or_default();
    let mut items = Vec::new();
    for categ in regions.get("mainline").and_then(|m| m.as_array()).into_iter().flatten() {
        if js(categ, "display_type") != "web-google" {
            continue; // news and images: not asked for in the web category
        }
        for item in categ.get("results").and_then(|r| r.as_array()).into_iter().flatten() {
            let (content, date) = parse_published_date(&html_to_text(&js(item, "description")));
            let mut m = MainResult::new(js(item, "clickUrl"), html_to_text(&js(item, "title")), content);
            m.published = date;
            items.push(Item::Main(m));
        }
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_in_snippets() {
        let (c, d) = parse_published_date("5 Mar 2024 ... the text");
        assert_eq!(c, "the text");
        assert_eq!(d.unwrap().isoformat(), "2024-03-05T00:00:00");
        let (c, d) = parse_published_date("no date ... here");
        assert_eq!(c, "no date ... here");
        assert!(d.is_none());
    }

    #[test]
    fn anubis() {
        let (n, d) = anubis_nonce("abc", 2).unwrap();
        assert!(d.starts_with("00"));
        assert_eq!(hex(&Sha256::digest(format!("abc{n}").as_bytes())), d);
    }
}
