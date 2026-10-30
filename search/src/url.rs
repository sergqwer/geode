// SPDX-License-Identifier: AGPL-3.0-or-later
//! Python's urllib.parse as SearXNG uses it on result URLs: urlparse, geturl, urljoin and urlencode, with their
//! exact rules (a result's identity across engines is built from these pieces), and the IDNA decoding of
//! `xn--` host names (searx.result_types._base._normalize_url_fields).

const SCHEME_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";
const USES_PARAMS: &[&str] = &[
    "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtspu", "sip", "sips", "mms", "sftp",
    "tel",
];
const USES_NETLOC: &[&str] = &[
    "",
    "ftp",
    "http",
    "gopher",
    "nntp",
    "telnet",
    "imap",
    "wais",
    "file",
    "mms",
    "https",
    "shttp",
    "snews",
    "prospero",
    "rtsp",
    "rtsps",
    "rtspu",
    "rsync",
    "svn",
    "svn+ssh",
    "sftp",
    "nfs",
    "git",
    "git+ssh",
    "ws",
    "wss",
    "itms-services",
];
const USES_RELATIVE: &[&str] = &[
    "", "ftp", "http", "gopher", "nntp", "imap", "wais", "file", "https", "shttp", "mms", "prospero", "rtsp", "rtsps",
    "rtspu", "sftp", "svn", "svn+ssh", "ws", "wss",
];

/// urllib.parse.ParseResult
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Url {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub params: String,
    pub query: String,
    pub fragment: String,
}

/// urllib.parse.urlsplit (Python 3.12+), the parts as (scheme, netloc, path, query, fragment).
pub fn urlsplit(url: &str) -> (String, String, String, String, String) {
    // leading C0 controls and spaces stripped; tabs and newlines removed anywhere
    let url: String =
        url.trim_start_matches(|c: char| c <= ' ').chars().filter(|c| !matches!(c, '\t' | '\r' | '\n')).collect();
    let mut rest = url.as_str();
    let mut scheme = String::new();
    if let Some(i) = rest.find(':') {
        let first = rest.chars().next().unwrap();
        if i > 0 && first.is_ascii_alphabetic() && rest[..i].chars().all(|c| SCHEME_CHARS.contains(c)) {
            scheme = rest[..i].to_lowercase();
            rest = &rest[i + 1..];
        }
    }
    let mut netloc = String::new();
    if let Some(r) = rest.strip_prefix("//") {
        let end = r.find(['/', '?', '#']).unwrap_or(r.len());
        netloc = r[..end].to_string();
        rest = &r[end..];
    }
    let (mut path, mut fragment) = (rest.to_string(), String::new());
    if let Some((p, f)) = path.clone().split_once('#') {
        path = p.to_string();
        fragment = f.to_string();
    }
    let mut query = String::new();
    if let Some((p, q)) = path.clone().split_once('?') {
        path = p.to_string();
        query = q.to_string();
    }
    (scheme, netloc, path, query, fragment)
}

/// urllib.parse.urlparse
pub fn urlparse(url: &str) -> Url {
    let (scheme, netloc, mut path, query, fragment) = urlsplit(url);
    let mut params = String::new();
    if USES_PARAMS.contains(&scheme.as_str()) && path.contains(';') {
        // _splitparams: the ';' of the last path segment
        let start = path.rfind('/').unwrap_or(0);
        if let Some(i) = path[start..].find(';') {
            let i = start + i;
            params = path[i + 1..].to_string();
            path.truncate(i);
        }
    }
    Url { scheme, netloc, path, params, query, fragment }
}

impl Url {
    /// ParseResult.geturl (urlunparse / urlunsplit of Python 3.12+)
    pub fn geturl(&self) -> String {
        let mut url = self.path.clone();
        if !self.params.is_empty() {
            url = format!("{url};{}", self.params);
        }
        if !self.netloc.is_empty() {
            if !url.is_empty() && !url.starts_with('/') {
                url = format!("/{url}");
            }
            url = format!("//{}{url}", self.netloc);
        } else if url.starts_with("//")
            || (!self.scheme.is_empty()
                && USES_NETLOC.contains(&self.scheme.as_str())
                && (url.is_empty() || url.starts_with('/')))
        {
            url = format!("//{url}");
        }
        if !self.scheme.is_empty() {
            url = format!("{}:{url}", self.scheme);
        }
        if !self.query.is_empty() {
            url = format!("{url}?{}", self.query);
        }
        if !self.fragment.is_empty() {
            url = format!("{url}#{}", self.fragment);
        }
        url
    }

    pub fn hostname(&self) -> String {
        let h = self.netloc.rsplit('@').next().unwrap_or("");
        let h = if h.starts_with('[') {
            h.split(']').next().unwrap_or("").trim_start_matches('[')
        } else {
            h.split(':').next().unwrap_or("")
        };
        h.to_lowercase()
    }
}

/// searx.result_types._base._normalize_url_fields: a result URL as SearXNG keeps it (IDN host decoded, http when
/// no scheme) and its parts.
pub fn normalize_result_url(url: &str) -> (String, Url) {
    let mut u = urlparse(url);
    if u.netloc.starts_with("xn--") {
        u.netloc = idna_decode(&u.netloc);
    }
    if u.scheme.is_empty() {
        u.scheme = "http".into();
    }
    (u.geturl(), u)
}

/// The "idna" codec's decode: every `xn--` label from Punycode.
pub fn idna_decode(host: &str) -> String {
    host.split('.')
        .map(|label| {
            if label.len() > 4 && label[..4].eq_ignore_ascii_case("xn--") {
                punycode_decode(&label[4..]).unwrap_or_else(|| label.to_string())
            } else {
                label.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// RFC 3492
fn punycode_decode(input: &str) -> Option<String> {
    const BASE: u32 = 36;
    let (mut out, rest): (Vec<char>, &str) = match input.rfind('-') {
        Some(i) => (input[..i].chars().collect(), &input[i + 1..]),
        None => (Vec::new(), input),
    };
    let (mut n, mut i, mut bias) = (128u32, 0u32, 72u32);
    let mut it = rest.bytes().peekable();
    while it.peek().is_some() {
        let old_i = i;
        let mut w = 1u32;
        let mut k = BASE;
        loop {
            let c = it.next()?;
            let digit = match c {
                b'a'..=b'z' => (c - b'a') as u32,
                b'A'..=b'Z' => (c - b'A') as u32,
                b'0'..=b'9' => (c - b'0') as u32 + 26,
                _ => return None,
            };
            i = i.checked_add(digit.checked_mul(w)?)?;
            let t = if k <= bias {
                1
            } else if k >= bias + 26 {
                26
            } else {
                k - bias
            };
            if digit < t {
                break;
            }
            w = w.checked_mul(BASE - t)?;
            k += BASE;
        }
        let len = out.len() as u32 + 1;
        // adapt
        let mut delta = if old_i == 0 { (i - old_i) / 700 } else { (i - old_i) / 2 };
        delta += delta / len;
        let mut k = 0;
        while delta > ((BASE - 1) * 26) / 2 {
            delta /= BASE - 1;
            k += BASE;
        }
        bias = k + (BASE * delta) / (delta + 38);
        n = n.checked_add(i / len)?;
        i %= len;
        out.insert(i as usize, char::from_u32(n)?);
        i += 1;
    }
    Some(out.into_iter().collect())
}

/// urllib.parse.urljoin
pub fn urljoin(base: &str, url: &str) -> String {
    if base.is_empty() {
        return url.to_string();
    }
    if url.is_empty() {
        return base.to_string();
    }
    let b = urlparse(base);
    let mut u = urlparse(url);
    if u.scheme.is_empty() {
        u.scheme = b.scheme.clone();
    }
    if u.scheme != b.scheme || !USES_RELATIVE.contains(&u.scheme.as_str()) {
        return url.to_string();
    }
    if USES_NETLOC.contains(&u.scheme.as_str()) {
        if !u.netloc.is_empty() {
            return u.geturl();
        }
        u.netloc = b.netloc.clone();
    }
    if u.path.is_empty() && u.params.is_empty() {
        u.path = b.path.clone();
        u.params = b.params.clone();
        if u.query.is_empty() {
            u.query = b.query.clone();
        }
        return u.geturl();
    }
    let base_parts: Vec<&str> = b.path.split('/').collect();
    let mut base_parts: Vec<String> = base_parts.iter().map(|s| s.to_string()).collect();
    if base_parts.last().is_some_and(|s| !s.is_empty()) {
        base_parts.pop();
    }
    let segments: Vec<String> = if u.path.starts_with('/') {
        u.path.split('/').map(String::from).collect()
    } else {
        let mut s = base_parts;
        s.extend(u.path.split('/').map(String::from));
        // filter out elements that would cause redundant slashes on re-joining
        let n = s.len();
        let mut f: Vec<String> = Vec::new();
        for (idx, seg) in s.into_iter().enumerate() {
            if idx == 0 || idx == n - 1 || !seg.is_empty() {
                f.push(seg);
            }
        }
        f
    };
    let mut resolved: Vec<String> = Vec::new();
    for seg in &segments {
        if seg == ".." {
            resolved.pop();
        } else if seg != "." {
            resolved.push(seg.clone());
        }
    }
    if segments.last().is_some_and(|s| s == "." || s == "..") {
        resolved.push(String::new());
    }
    let path = resolved.join("/");
    u.path = if path.is_empty() { "/".into() } else { path };
    u.geturl()
}

/// urllib.parse.quote_plus (safe "") of one value
pub fn quote_plus(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// urllib.parse.quote (safe "/" by default)
pub fn quote(s: &str, safe: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            _ if b < 128 && safe.contains(b as char) => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// urllib.parse.urlencode
pub fn urlencode<K: AsRef<str>, V: AsRef<str>>(pairs: &[(K, V)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", quote_plus(k.as_ref()), quote_plus(v.as_ref())))
        .collect::<Vec<_>>()
        .join("&")
}

/// urllib.parse.unquote
pub fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = ((b[i + 1] as char).to_digit(16), (b[i + 2] as char).to_digit(16))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// urllib.parse.parse_qs: the values of one key
pub fn query_values(query: &str, key: &str) -> Vec<String> {
    query
        .split(['&', ';'])
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            (unquote(&k.replace('+', " ")) == key && !v.is_empty()).then(|| unquote(&v.replace('+', " ")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_unparse() {
        let u = urlparse("HTTPS://Example.com/a/b;p?q=1#f");
        assert_eq!(
            (
                u.scheme.as_str(),
                u.netloc.as_str(),
                u.path.as_str(),
                u.params.as_str(),
                u.query.as_str(),
                u.fragment.as_str()
            ),
            ("https", "Example.com", "/a/b", "p", "q=1", "f")
        );
        assert_eq!(u.geturl(), "https://Example.com/a/b;p?q=1#f");
        assert_eq!(urlparse("https://example.com?").geturl(), "https://example.com");
        // no scheme, no netloc: Python's geturl gives "http:" + path
        assert_eq!(normalize_result_url("example.com/x").0, "http:example.com/x");
        assert_eq!(normalize_result_url("//example.com/x").0, "http://example.com/x");
    }

    #[test]
    fn idn() {
        assert_eq!(idna_decode("xn--strung-xxa.de"), "störung.de");
        assert_eq!(normalize_result_url("https://xn--strung-xxa.de/a").0, "https://störung.de/a");
        assert_eq!(idna_decode("xn--80aswg.xn--p1ai"), "сайт.рф");
    }

    #[test]
    fn join() {
        assert_eq!(urljoin("https://example.com", "/path?a=1"), "https://example.com/path?a=1");
        assert_eq!(urljoin("https://example.com/a/b", "c"), "https://example.com/a/c");
        assert_eq!(urljoin("https://example.com/a/b", "../c"), "https://example.com/c");
        assert_eq!(urljoin("https://example.com/a", "//other.org/x"), "https://other.org/x");
    }

    #[test]
    fn encode() {
        assert_eq!(urlencode(&[("q", "python asyncio"), ("x", "a&b/é")]), "q=python+asyncio&x=a%26b%2F%C3%A9");
        assert_eq!(quote("Kyiv (city)", "/"), "Kyiv%20%28city%29");
        assert_eq!(unquote("https%3A%2F%2Fa.b%2F%C3%A9"), "https://a.b/é");
        assert_eq!(query_values("u=a1aHR0&x=2&u=b", "u"), ["a1aHR0", "b"]);
    }
}
