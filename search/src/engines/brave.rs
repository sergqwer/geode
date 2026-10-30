// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/brave.py (category "search"): Brave's results page carries its data as a JavaScript literal in a
//! SvelteKit `data: [...]` array, partly as immediately invoked functions whose arguments fill the object; it is
//! made JSON (searx.utils' JS helpers, src/js.rs) and read. Country and UI language go in cookies.

use super::{Params, js};
use crate::date::parse_iso;
use crate::html::html_to_text;
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::js::{find_matching_bracket, find_variable_positions, js_obj_to_json, split_args};
use crate::results::{Item, MainResult};
use crate::url::urlencode;
use serde_json::Value;

const BASE_URL: &str = "https://search.brave.com/";

pub fn request(query: &str, p: &Params) -> HttpRequest {
    let mut args = vec![("q", query.to_string()), ("source", "web".into())];
    if p.pageno > 1 {
        args.push(("offset", (p.pageno - 1).to_string()));
    }
    let safesearch = match p.safesearch {
        2 => "strict",
        1 => "moderate",
        _ => "off",
    };
    p.request(format!("{BASE_URL}search?{}", urlencode(&args)))
        .cookie("safesearch", safesearch)
        .cookie("useLocation", "0")
        .cookie("summarizer", "0")
        .cookie("country", p.row.str("br_c"))
        .cookie("ui_lang", p.row.str("br_ui"))
}

/// IIFE_RE at `pos`: (function ( params ) { return {  — the params and where the object starts
fn match_iife(c: &[char], pos: usize) -> Option<(String, usize)> {
    let mut i = pos;
    let lit = |i: &mut usize, s: &str| -> bool {
        let n = s.chars().count();
        if *i + n <= c.len() && c[*i..*i + n].iter().copied().eq(s.chars()) {
            *i += n;
            true
        } else {
            false
        }
    };
    let ws = |i: &mut usize| {
        while *i < c.len() && c[*i].is_whitespace() {
            *i += 1;
        }
    };
    if !lit(&mut i, "(function") {
        return None;
    }
    ws(&mut i);
    if !lit(&mut i, "(") {
        return None;
    }
    let start = i;
    while i < c.len() && c[i] != ')' {
        i += 1;
    }
    let params: String = c[start..i].iter().collect();
    if !lit(&mut i, ")") {
        return None;
    }
    ws(&mut i);
    if !lit(&mut i, "{") {
        return None;
    }
    ws(&mut i);
    if !lit(&mut i, "return") {
        return None;
    }
    ws(&mut i);
    (i < c.len() && c[i] == '{').then_some((params, i))
}

/// extract_iife_object: the object literal with the call's arguments put in for the parameters
fn extract_iife_object(c: &[char], params: &str, body_start: usize) -> Result<(String, usize), String> {
    let body_end = find_matching_bracket(c, body_start, '{', '}')?;
    let mut obj: Vec<char> = c[body_start..=body_end].to_vec();
    let call_start =
        c[body_end..].iter().position(|x| *x == '(').map(|p| p + body_end).ok_or("IIFE missing call arguments")?;
    let call_end = find_matching_bracket(c, call_start, '(', ')')?;
    let args = split_args(&c[call_start + 1..call_end]);
    let names: Vec<String> = params.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
    let vars = find_variable_positions(&obj);
    for (name, pos) in vars.into_iter().rev() {
        if let Some(i) = names.iter().position(|n| *n == name)
            && let Some(value) = args.get(i)
        {
            let n = name.chars().count();
            obj.splice(pos..pos + n, value.chars());
        }
    }
    Ok((obj.into_iter().collect(), call_end + 1))
}

/// extract_json_data: the SvelteKit data payload of the page, as JSON
pub fn extract_json_data(text: &str) -> Result<Value, String> {
    let c: Vec<char> = text.chars().collect();
    let marker: Vec<char> = "data: [".chars().collect();
    let start = c
        .windows(marker.len())
        .position(|w| w == marker.as_slice())
        .ok_or("Brave data payload not found (no 'data: [')")?;
    let array_start = start + marker.len() - 1;
    let array_end = find_matching_bracket(&c, array_start, '[', ']')?;
    let src = &c[array_start + 1..array_end];
    let mut objects: Vec<String> = Vec::new();
    let (mut pos, n) = (0usize, src.len());
    while pos < n {
        while pos < n && " \t\n\r,".contains(src[pos]) {
            pos += 1;
        }
        if pos >= n {
            break;
        }
        if let Some((params, body)) = match_iife(src, pos) {
            let (obj, next) = extract_iife_object(src, &params, body)?;
            objects.push(obj);
            pos = next;
            while pos < n && " \t\n\r)".contains(src[pos]) {
                pos += 1;
            }
            continue;
        }
        if src[pos] == '{' {
            let end = find_matching_bracket(src, pos, '{', '}')?;
            objects.push(src[pos..=end].iter().collect());
            pos = end + 1;
            continue;
        }
        let near: String = src[pos..(pos + 40).min(n)].iter().collect();
        return Err(format!("unexpected token in data array at offset {pos}: {near:?}"));
    }
    if objects.is_empty() {
        return Err("no data objects found inside Brave payload".into());
    }
    let json = js_obj_to_json(&format!("{{\"data\":[{}]}}", objects.join(",")))?;
    serde_json::from_str(&json).map_err(|e| format!("JSONDecodeError: {e}"))
}

fn published(v: &Value, key: &str) -> Option<crate::date::DateTime> {
    v.get(key).and_then(Value::as_str).and_then(parse_iso)
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    let data = extract_json_data(&resp.text).map_err(EngineError::Parse)?;
    let mut items = Vec::new();
    // _get_response_data(json_data, "search")
    let d = data
        .get("data")
        .and_then(|a| a.get(1))
        .and_then(|x| x.get("data"))
        .ok_or_else(|| EngineError::Response("Unexpected Brave JSON structure for category 'search'".into()))?;
    if d.get("noResults").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(items);
    }
    let body = d
        .get("body")
        .and_then(|b| b.get("response"))
        .ok_or_else(|| EngineError::Response("Unexpected Brave JSON structure for category 'search'".into()))?;
    let web = body
        .get("web")
        .ok_or_else(|| EngineError::Response("Unexpected Brave JSON structure for category 'search'".into()))?;
    let results =
        web.get("results").and_then(Value::as_array).ok_or_else(|| EngineError::Parse("KeyError: 'results'".into()))?;
    for r in results {
        // parse_search_result
        let mut m = MainResult::new(js(r, "url"), js(r, "title"), html_to_text(&js(r, "description")));
        m.published = published(r, "page_age");
        if let Some(t) = r.get("thumbnail").filter(|t| t.is_object())
            && !t.get("logo").and_then(Value::as_bool).unwrap_or(false)
        {
            m.thumbnail = js(t, "src");
        }
        items.push(Item::Main(m));
    }
    // _parse_secondary_items: the videos in the page, and Brave's related queries as suggestions
    if let Some(videos) = body.get("videos").and_then(|v| v.get("results")).and_then(Value::as_array) {
        for r in videos {
            let mut m = MainResult::new(js(r, "url"), js(r, "title"), js(r, "description"));
            m.template = "videos.html";
            m.published = published(r, "age");
            m.thumbnail = r.get("thumbnail").map(|t| js(t, "src")).unwrap_or_default();
            items.push(Item::Main(m));
        }
    }
    if let Some(rq) = body.get("query").and_then(|q| q.get("related_queries")).and_then(Value::as_array) {
        for x in rq {
            if let Some(parts) = x.as_array() {
                let s: Vec<String> =
                    parts.iter().filter_map(|v| v.get(1).and_then(Value::as_str).map(String::from)).collect();
                items.push(Item::Suggestion(s.join(" ")));
            }
        }
    }
    Ok(items)
}
