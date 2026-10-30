// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/wikipedia.py (display_type infobox): the REST summary of the page the query names, on the
//! Wikipedia of the search's language, as an infobox (a page that is not a standard article, as a result).
//!
//! One change from upstream: the summary API answers a redirect page (Kiev -> Kyiv) with a 302, which SearXNG,
//! not following redirects there, fails on; here up to two redirects are followed (its soft_max_redirects).

use super::{Params, js};
use crate::html::html_to_text;
use crate::http::{EngineError, HttpRequest, HttpResponse, raise_for_httperror};
use crate::results::{Infobox, Item, MainResult};
use crate::url::quote;
use crate::util::{py_islower, py_title};

pub fn request(query: &str, p: &Params) -> HttpRequest {
    let q = if py_islower(query) { py_title(query) } else { query.to_string() };
    let netloc = p.row.get("wp").unwrap_or("en.wikipedia.org");
    let mut req = p.request(format!("https://{netloc}/api/rest_v1/page/summary/{}", quote(&q, "/")));
    req.follow_redirects = true;
    req
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    if resp.status == 404 {
        return Ok(vec![]);
    }
    if resp.status == 400
        && let Ok(v) = resp.json()
        && js(&v, "type") == "https://mediawiki.org/wiki/HyperSwitch/errors/bad_request"
        && js(&v, "detail") == "title-invalid-characters"
    {
        return Ok(vec![]);
    }
    raise_for_httperror(resp)?;
    let v = resp.json()?;
    let display = v
        .get("titles")
        .and_then(|t| t.get("display"))
        .and_then(|d| d.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .unwrap_or_else(|| js(&v, "title"));
    let title = html_to_text(&display);
    let link = v
        .get("content_urls")
        .and_then(|c| c.get("desktop"))
        .and_then(|d| d.get("page"))
        .and_then(|p| p.as_str())
        .ok_or_else(|| EngineError::Parse("KeyError: 'content_urls'".into()))?
        .to_string();
    let mut items = Vec::new();
    let standard = js(&v, "type") == "standard";
    if !standard {
        items.push(Item::Main(MainResult::new(link.clone(), title.clone(), js(&v, "description"))));
    } else {
        items.push(Item::Infobox(Infobox {
            title,
            id: link.clone(),
            content: js(&v, "extract"),
            img_src: v.get("thumbnail").map(|t| js(t, "source")).unwrap_or_default(),
            urls: vec![("Wikipedia".into(), link)],
            ..Default::default()
        }));
    }
    Ok(items)
}
