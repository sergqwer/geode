// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx.utils' text helpers: html_to_text (an HTML snippet as plain text, the way its HTMLTextExtractor on
//! Python's HTMLParser does it) and extract_text (an element's text, as lxml's method="text" gives it), plus CSS
//! selection standing in for the engines' XPath expressions.

use crate::util::squeeze;
use scraper::{ElementRef, Html, Selector};

/// A CSS selector (the engines' selectors are fixed: a bad one is a bug).
pub fn sel(css: &str) -> Selector {
    Selector::parse(css).unwrap_or_else(|_| panic!("selector {css}"))
}

pub fn parse(html: &str) -> Html {
    Html::parse_document(html)
}

/// extract_text(element): every text node under it, whitespace squeezed.
pub fn text(el: ElementRef) -> String {
    squeeze(&el.text().collect::<String>())
}

/// extract_text(list): each element's text, concatenated, stripped.
pub fn text_all<'a>(els: impl IntoIterator<Item = ElementRef<'a>>) -> String {
    els.into_iter().map(text).collect::<String>().trim().to_string()
}

pub fn attr(el: ElementRef, name: &str) -> Option<String> {
    el.value().attr(name).map(str::to_string)
}

pub fn parent(el: ElementRef) -> Option<ElementRef> {
    el.parent().and_then(ElementRef::wrap)
}

/// The child elements named `tag` (XPath's `./tag`).
pub fn children<'a>(el: ElementRef<'a>, tag: &'a str) -> impl Iterator<Item = ElementRef<'a>> + 'a {
    el.children().filter_map(ElementRef::wrap).filter(move |c| c.value().name() == tag)
}

/// html_to_text: tags dropped (a <br> becomes a space), script and style left out, character references decoded;
/// an end tag that does not close the open one stays as text (HTMLTextExtractor.handle_endtag).
pub fn html_to_text(html: &str) -> String {
    if html.is_empty() {
        return String::new();
    }
    let html = squeeze(&html.replace(['\n', '\r'], " "));
    let mut out = String::new();
    let mut tags: Vec<String> = Vec::new();
    let blocked = |tags: &Vec<String>| tags.last().is_some_and(|t| t == "script" || t == "style");
    let b = html.as_bytes();
    let mut i = 0;
    let mut data_start = 0;
    let flush = |out: &mut String, tags: &Vec<String>, s: &str| {
        if !blocked(tags) {
            out.push_str(&unescape(s));
        }
    };
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        let rest = &html[i..];
        // inside script/style only the matching end tag ends the raw text
        if blocked(&tags) {
            let end = format!("</{}", tags.last().unwrap());
            if !rest.to_lowercase().starts_with(&end) {
                i += 1;
                continue;
            }
        }
        if rest.starts_with("<!--") {
            flush(&mut out, &tags, &html[data_start..i]);
            let end = rest.find("-->").map(|e| i + e + 3).unwrap_or(b.len());
            i = end;
            data_start = i;
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            flush(&mut out, &tags, &html[data_start..i]);
            i = rest.find('>').map(|e| i + e + 1).unwrap_or(b.len());
            data_start = i;
            continue;
        }
        let is_end = rest.starts_with("</");
        let name_start = if is_end { 2 } else { 1 };
        let name: String =
            rest[name_start..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == ':').collect();
        if name.is_empty() || !rest[name_start..].starts_with(|c: char| c.is_ascii_alphabetic()) {
            i += 1; // a "<" that starts no tag is text
            continue;
        }
        let Some(close) = tag_end(rest) else {
            i += 1;
            continue;
        };
        flush(&mut out, &tags, &html[data_start..i]);
        let name = name.to_ascii_lowercase();
        let self_closing = rest[..close].ends_with("/>");
        if is_end {
            if let Some(last) = tags.last() {
                if *last == name {
                    tags.pop();
                } else {
                    out.push_str(&format!("</{name}>"));
                }
            }
        } else {
            tags.push(name.clone());
            if name == "br" {
                out.push(' ');
            }
            if self_closing {
                tags.pop();
            }
        }
        i += close;
        data_start = i;
    }
    flush(&mut out, &tags, &html[data_start..]);
    out.trim().to_string()
}

/// The length of a tag up to its closing ">", quotes respected.
fn tag_end(s: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (i, c) in s.char_indices().skip(1) {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '>' => return Some(i + 1),
            None => {}
        }
    }
    None
}

/// html.unescape for the references result snippets carry.
pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let body: String = rest[1..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '#').take(32).collect();
        let semi = rest[1 + body.len()..].starts_with(';');
        let decoded = if let Some(num) = body.strip_prefix('#') {
            let v = if let Some(h) = num.strip_prefix(['x', 'X']) {
                u32::from_str_radix(h, 16).ok()
            } else {
                num.parse::<u32>().ok()
            };
            v.map(|v| match v {
                0 | 0xD800..=0xDFFF => '\u{FFFD}',
                v if v > 0x10FFFF => '\u{FFFD}',
                v => char::from_u32(v).unwrap_or('\u{FFFD}'),
            })
            .map(String::from)
        } else {
            named(&body).map(String::from)
        };
        match decoded {
            Some(d) if !body.is_empty() => {
                out.push_str(&d);
                rest = &rest[1 + body.len() + usize::from(semi)..];
            }
            _ => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn named(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" | "AMP" => "&",
        "lt" | "LT" => "<",
        "gt" | "GT" => ">",
        "quot" | "QUOT" => "\"",
        "apos" => "'",
        "nbsp" => "\u{a0}",
        "ndash" => "–",
        "mdash" => "—",
        "hellip" => "…",
        "laquo" => "«",
        "raquo" => "»",
        "lsquo" => "‘",
        "rsquo" => "’",
        "sbquo" => "‚",
        "ldquo" => "“",
        "rdquo" => "”",
        "bdquo" => "„",
        "copy" => "©",
        "reg" => "®",
        "trade" => "™",
        "middot" => "·",
        "bull" => "•",
        "times" => "×",
        "divide" => "÷",
        "deg" => "°",
        "euro" => "€",
        "pound" => "£",
        "yen" => "¥",
        "cent" => "¢",
        "sect" => "§",
        "para" => "¶",
        "shy" => "\u{ad}",
        "zwj" => "\u{200d}",
        "zwnj" => "\u{200c}",
        "thinsp" => "\u{2009}",
        "ensp" => "\u{2002}",
        "emsp" => "\u{2003}",
        "larr" => "←",
        "rarr" => "→",
        "uarr" => "↑",
        "darr" => "↓",
        "plusmn" => "±",
        "frac12" => "½",
        "frac14" => "¼",
        "frac34" => "¾",
        "iexcl" => "¡",
        "iquest" => "¿",
        "acute" => "´",
        "uml" => "¨",
        "micro" => "µ",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_text_as_searxng() {
        // the doctests of searx.utils.html_to_text
        assert_eq!(html_to_text("Example <span id=\"42\">#2</span>"), "Example #2");
        assert_eq!(html_to_text("<style>.span { color: red; }</style><span>Example</span>"), "Example");
        assert_eq!(html_to_text("regexp: (?&lt;![a-zA-Z]"), "regexp: (?<![a-zA-Z]");
        assert_eq!(html_to_text("<p><b>Lorem ipsum </i>dolor sit amet</p>"), "Lorem ipsum </i>dolor sit amet</p>");
        assert_eq!(html_to_text("&#x3e &#x3c &#97"), "> < a");
        assert_eq!(html_to_text("a<br>b"), "a b");
        assert_eq!(html_to_text("<strong>Python</strong> 3 &amp; asyncio"), "Python 3 & asyncio");
    }

    #[test]
    fn element_text() {
        let d = parse("<div id=x> a <b>b\n c</b><script>s</script> </div>");
        let el = d.select(&sel("#x")).next().unwrap();
        assert_eq!(text(el), "a b cs");
    }
}
