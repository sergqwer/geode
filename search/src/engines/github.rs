// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/github.py: GitHub's repository search API, the most starred first.

use super::{Params, js};
use crate::date::parse_iso;
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::results::{Item, MainResult};
use crate::url::urlencode;

pub fn request(query: &str, p: &Params) -> HttpRequest {
    p.request(format!(
        "https://api.github.com/search/repositories?sort=stars&order=desc&{}",
        urlencode(&[("q", query)])
    ))
    .header("Accept", "application/vnd.github.preview.text-match+json")
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    let v = resp.json()?;
    let mut items = Vec::new();
    for item in v.get("items").and_then(|i| i.as_array()).into_iter().flatten() {
        let content: Vec<String> =
            ["language", "description"].iter().map(|k| js(item, k)).filter(|s| !s.is_empty()).collect();
        let mut m = MainResult::new(js(item, "html_url"), js(item, "full_name"), content.join(" / "));
        m.template = "packages.html";
        m.thumbnail = item.get("owner").map(|o| js(o, "avatar_url")).unwrap_or_default();
        let date = Some(js(item, "updated_at")).filter(|s| !s.is_empty()).unwrap_or_else(|| js(item, "created_at"));
        m.published = parse_iso(&date);
        items.push(Item::Main(m));
    }
    Ok(items)
}
