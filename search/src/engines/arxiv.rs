// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/arxiv.py: arXiv's API (an Atom feed), every field searched.

use super::Params;
use crate::date::parse_iso;
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::results::{Item, MainResult};
use crate::url::urlencode;

const MAX_RESULTS: u32 = 10;
const ATOM: &str = "http://www.w3.org/2005/Atom";

pub fn request(query: &str, p: &Params) -> HttpRequest {
    let args = [
        ("search_query", format!("all:{query}")),
        ("start", ((p.pageno - 1) * MAX_RESULTS).to_string()),
        ("max_results", MAX_RESULTS.to_string()),
    ];
    p.request(format!("https://export.arxiv.org/api/query?{}", urlencode(&args)))
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    let doc = roxmltree::Document::parse(&resp.text).map_err(|e| EngineError::Parse(format!("XMLSyntaxError: {e}")))?;
    let mut items = Vec::new();
    for entry in doc.descendants().filter(|n| n.has_tag_name((ATOM, "entry"))) {
        let field = |name: &str| -> Option<String> {
            entry.descendants().find(|n| n.has_tag_name((ATOM, name))).map(|n| n.text().unwrap_or("").to_string())
        };
        let (Some(title), Some(url), Some(summary), Some(published)) =
            (field("title"), field("id"), field("summary"), field("published"))
        else {
            return Err(EngineError::Parse("IndexError: an entry without title, id, summary or date".into()));
        };
        let mut m = MainResult::new(url, title, summary);
        m.template = "paper.html";
        // datetime.fromisoformat(text.rstrip("Z")): a naive time
        m.published = parse_iso(published.trim_end_matches('Z')).map(|mut d| {
            d.offset = None;
            d
        });
        items.push(Item::Main(m));
    }
    Ok(items)
}
