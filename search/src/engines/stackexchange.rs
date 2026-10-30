// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/stackexchange.py: the Stack Exchange API's advanced search on one site (stackoverflow here,
//! sorted by relevance as geode's settings ask).

use super::{Params, js};
use crate::html::unescape;
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::results::{Item, MainResult};
use crate::url::urlencode;

const PAGESIZE: u32 = 10;
const SORT: &str = "relevance";

pub fn request(query: &str, p: &Params, site: &str) -> HttpRequest {
    let args = [
        ("q", query.to_string()),
        ("page", p.pageno.to_string()),
        ("pagesize", PAGESIZE.to_string()),
        ("site", site.to_string()),
        ("sort", SORT.to_string()),
        ("order", "desc".to_string()),
    ];
    p.request(format!("https://api.stackexchange.com/2.3/search/advanced?{}", urlencode(&args)))
}

pub fn response(resp: &HttpResponse, site: &str) -> Result<Vec<Item>, EngineError> {
    let v = resp.json()?;
    let mut items = Vec::new();
    for r in v.get("items").and_then(|i| i.as_array()).ok_or_else(|| EngineError::Parse("KeyError: 'items'".into()))? {
        let tags: Vec<&str> =
            r.get("tags").and_then(|t| t.as_array()).into_iter().flatten().filter_map(|t| t.as_str()).collect();
        let mut content = format!("[{}]", tags.join(", "));
        content += &format!(" {}", r.get("owner").map(|o| js(o, "display_name")).unwrap_or_default());
        if r.get("is_answered").and_then(|a| a.as_bool()).unwrap_or(false) {
            content += " // is answered";
        }
        content += &format!(" // score: {}", r.get("score").map(|s| s.to_string()).unwrap_or_default());
        let id = r.get("question_id").map(|q| q.to_string()).unwrap_or_default();
        items.push(Item::Main(MainResult::new(
            format!("https://{site}.com/q/{id}"),
            unescape(&js(r, "title")),
            unescape(&content),
        )));
    }
    Ok(items)
}
