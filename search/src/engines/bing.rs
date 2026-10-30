// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/bing.py: Bing's results page. The market (language-country) goes in setlang / cc; Bing's
//! tracking links (/ck/a?...&u=a1<base64>) are unwrapped to the page they lead to. Bing serves Chrome on Android
//! its mobile markup, where the title's link wraps the <h2> instead of sitting in it: both are read.

use super::Params;
use crate::html::{attr, parse, sel};
use crate::http::{EngineError, HttpRequest, HttpResponse};
use crate::results::{Item, MainResult};
use crate::url::{query_values, urlencode, urlparse};
use crate::util::{b64decode, squeeze};
use scraper::{ElementRef, Node};

const BASE_URL: &str = "https://www.bing.com";

pub fn request(query: &str, p: &Params) -> HttpRequest {
    let region = p.row.str("bi");
    let adlt = match p.safesearch {
        2 => "strict",
        1 => "moderate",
        _ => "off",
    };
    let mut args = vec![("q", query.to_string()), ("adlt", adlt.to_string())];
    if !region.is_empty() && region != "clear" {
        let (lang, cc) = region.split_once('-').unwrap_or((region, ""));
        args.push(("setlang", lang.to_string()));
        if !cc.is_empty() && !["us", "cn", "ru"].contains(&cc) {
            args.push(("cc", cc.to_string())); // bing just sends junk for these
        }
    }
    p.request(format!("{BASE_URL}/search?{}", urlencode(&args)))
}

/// The text of an element, the icons Bing puts in a snippet (span.algoSlug_icon) left out.
fn text_without_icons(el: ElementRef) -> String {
    let mut s = String::new();
    for node in el.descendants() {
        if let Node::Text(t) = node.value() {
            let in_icon = node
                .ancestors()
                .filter_map(ElementRef::wrap)
                .take_while(|a| a.id() != el.id())
                .any(|a| a.value().name() == "span" && a.value().attr("class") == Some("algoSlug_icon"));
            if !in_icon {
                s.push_str(t);
            }
        }
    }
    squeeze(&s)
}

pub fn unwrap_url(href: &str) -> String {
    if href.starts_with("https://www.bing.com/ck/a?") {
        let q = urlparse(href).query;
        if let Some(u) = query_values(&q, "u").first()
            && let Some(enc) = u.strip_prefix("a1")
            && let Some(b) = b64decode(enc)
        {
            return String::from_utf8_lossy(&b).into_owned();
        }
    }
    href.to_string()
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    let doc = parse(&resp.text);
    let mut items = Vec::new();
    for item in doc.select(&sel("ol#b_results > li[class*=\"b_algo\"]")) {
        // desktop markup: <h2><a href>title</a></h2>; the mobile page (Chrome on Android) has <a href><h2>title</h2></a>
        let link = item
            .select(&sel("h2 > a"))
            .next()
            .or_else(|| item.select(&sel("a")).find(|a| a.select(&sel("h2")).next().is_some()));
        let Some(link) = link else { continue };
        let href = attr(link, "href").unwrap_or_default();
        let title = crate::html::text(link);
        if href.is_empty() || title.is_empty() {
            continue;
        }
        let content: String = item.select(&sel("p")).map(text_without_icons).collect::<String>().trim().to_string();
        items.push(Item::Main(MainResult::new(unwrap_url(&href), title, content)));
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_links() {
        let href = "https://www.bing.com/ck/a?!&&p=x&u=a1aHR0cHM6Ly93d3cucHl0aG9uLm9yZy8&ntb=1";
        assert_eq!(unwrap_url(href), "https://www.python.org/");
        assert_eq!(unwrap_url("https://example.com/"), "https://example.com/");
    }
}
