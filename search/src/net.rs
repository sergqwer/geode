// SPDX-License-Identifier: AGPL-3.0-or-later
//! The network (searx/network): one client for the whole search — Chrome 149 on Android's TLS ClientHello,
//! HTTP/2 settings, header order, User-Agent and client hints (wreq with wreq-util's emulation), connections
//! reused across engines — and the cookies a response sets kept for the rest of the search (by host).

use crate::http::{BoxFuture, EngineError, Fetch, HttpRequest, HttpResponse, Method};
use crate::url::{urlencode, urlparse};
use std::cell::RefCell;
use std::time::Duration;
use wreq::redirect::Policy;
use wreq_util::{Emulation, Platform, Profile};

/// The User-Agent wreq-util's Chrome149 / Android emulation sends (checked from the phone against a
/// fingerprinting service); DuckDuckGo's tokens are keyed by it.
pub const USER_AGENT: &str =
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Mobile Safari/537.36";

pub struct Net {
    client: wreq::Client,
    /// (domain, name, value): a response's cookies, sent again to that domain and its subdomains
    jar: RefCell<Vec<(String, String, String)>>,
    timeout: Duration,
}

impl Net {
    pub fn new(timeout: Duration) -> Result<Net, String> {
        let emulation = Emulation::builder().profile(Profile::Chrome149).platform(Platform::Android).build();
        let client = wreq::Client::builder().emulation(emulation).build().map_err(|e| e.to_string())?;
        Ok(Net { client, jar: RefCell::new(Vec::new()), timeout })
    }

    fn cookies_for(&self, host: &str, own: &[(String, String)]) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for (domain, name, value) in self.jar.borrow().iter() {
            if (host == domain || host.ends_with(&format!(".{domain}"))) && !own.iter().any(|c| &c.0 == name) {
                out.retain(|c| &c.0 != name);
                out.push((name.clone(), value.clone()));
            }
        }
        out.extend(own.iter().cloned());
        out
    }

    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, EngineError> {
        let host = urlparse(&req.url).hostname();
        let mut b = match req.method {
            Method::Get => self.client.get(&req.url),
            Method::Post => self.client.post(&req.url),
        };
        b = b.redirect(if req.follow_redirects { Policy::limited(10) } else { Policy::none() }).timeout(self.timeout);
        for (k, v) in &req.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        let cookies = self.cookies_for(&host, &req.cookies);
        if !cookies.is_empty() {
            b = b.header("cookie", cookies.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("; "));
        }
        if !req.form.is_empty() {
            b = b.body(urlencode(&req.form));
        } else if let Some((ct, body)) = req.body {
            b = b.header("content-type", ct).body(body);
        }
        let resp = b.send().await.map_err(net_error)?;
        let status = resp.status().as_u16();
        let url = resp.uri().to_string();
        let final_host = urlparse(&url).hostname();
        let mut headers = Vec::new();
        let mut set = Vec::new();
        for (k, v) in resp.headers() {
            let v = String::from_utf8_lossy(v.as_bytes()).into_owned();
            if k.as_str() == "set-cookie"
                && let Some((n, val)) = v.split(';').next().and_then(|nv| nv.split_once('='))
            {
                let domain = v
                    .split(';')
                    .skip(1)
                    .filter_map(|a| a.trim().split_once('='))
                    .find(|(k, _)| k.eq_ignore_ascii_case("domain"))
                    .map(|(_, d)| d.trim_start_matches('.').to_lowercase())
                    .unwrap_or_else(|| final_host.clone());
                set.push((domain, n.trim().to_string(), val.trim().to_string()));
            }
            headers.push((k.as_str().to_string(), v));
        }
        let text = resp.text().await.map_err(net_error)?;
        let cookies: Vec<(String, String)> = set.iter().map(|(_, n, v)| (n.clone(), v.clone())).collect();
        {
            let mut jar = self.jar.borrow_mut();
            for (d, n, v) in set {
                jar.retain(|c| !(c.0 == d && c.1 == n));
                jar.push((d, n, v));
            }
        }
        Ok(HttpResponse { status, url, headers, text, cookies })
    }
}

/// curl_cffi's exception names, as SearXNG reports them
fn net_error(e: wreq::Error) -> EngineError {
    let kind = if e.is_timeout() {
        "Timeout"
    } else if e.is_tls() {
        "SSLError"
    } else if e.is_connect() || e.is_dns() || e.is_connection_reset() {
        "ConnectionError"
    } else if e.is_redirect() {
        "TooManyRedirects"
    } else {
        "RequestException"
    };
    EngineError::Network(kind.to_string())
}

impl Fetch for Net {
    fn fetch(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, EngineError>> {
        Box::pin(self.send(req))
    }

    fn user_agent(&self) -> &str {
        USER_AGENT
    }
}
