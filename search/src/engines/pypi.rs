// SPDX-License-Identifier: AGPL-3.0-or-later
//! geode-search's engines/pypi_json.py: PyPI's search page answers scripts with a JavaScript challenge, so the
//! package the query names (its first word) is read from the JSON API: summary, latest version and its date,
//! home page.

use super::{Params, js};
use crate::date::parse_iso;
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::results::{Item, MainResult};
use crate::url::quote;

pub fn request(query: &str, p: &Params) -> Option<HttpRequest> {
    let name = query.split_whitespace().next().unwrap_or("");
    // an empty name would ask for /pypi//json
    Some(p.request(format!("https://pypi.org/pypi/{}/json", quote(name, ""))))
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    if resp.status != 200 {
        return Ok(vec![]);
    }
    let v = resp.json()?;
    let info = v.get("info").cloned().unwrap_or_default();
    let (name, version) = (js(&info, "name"), js(&info, "version"));
    if name.is_empty() {
        return Ok(vec![]);
    }
    let mut content = js(&info, "summary");
    let home = info
        .get("project_urls")
        .map(|u| js(u, "Homepage"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| js(&info, "home_page"));
    if !home.is_empty() {
        content += &format!(" (home: {home})");
    }
    let url = Some(js(&info, "package_url"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("https://pypi.org/project/{name}/"));
    let mut m = MainResult::new(url, format!("{name} {version}").trim().to_string(), content);
    m.published = v
        .get("urls")
        .and_then(|u| u.as_array())
        .and_then(|f| f.first())
        .map(|f| js(f, "upload_time_iso_8601"))
        .and_then(|d| parse_iso(&d));
    Ok(vec![Item::Main(m)])
}
