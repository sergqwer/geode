// SPDX-License-Identifier: AGPL-3.0-or-later
//! What an engine asks of the network and gets back (searx/search/processors/online.py's request parameters,
//! searx/network's response), the errors an engine ends with (searx/exceptions.py) and how an HTTP status
//! becomes one (searx/network/raise_for_httperror.py). The network itself is behind the `Fetch` trait: the
//! wreq client on the phone (net.rs), saved pages in the tests.

use crate::settings::*;
use std::future::Future;
use std::pin::Pin;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub cookies: Vec<(String, String)>,
    /// application/x-www-form-urlencoded body
    pub form: Vec<(String, String)>,
    /// any other body: (content type, bytes)
    pub body: Option<(String, Vec<u8>)>,
    /// the engines' own requests do not follow redirects (default_request_params: allow_redirects False);
    /// searx.network.get does
    pub follow_redirects: bool,
}

impl HttpRequest {
    pub fn get(url: impl Into<String>) -> HttpRequest {
        HttpRequest {
            method: Method::Get,
            url: url.into(),
            headers: vec![],
            cookies: vec![],
            form: vec![],
            body: None,
            follow_redirects: false,
        }
    }

    pub fn header(mut self, k: &str, v: impl Into<String>) -> Self {
        self.set_header(k, v);
        self
    }

    pub fn set_header(&mut self, k: &str, v: impl Into<String>) {
        let v = v.into();
        if let Some(h) = self.headers.iter_mut().find(|h| h.0.eq_ignore_ascii_case(k)) {
            h.1 = v;
        } else {
            self.headers.push((k.to_string(), v));
        }
    }

    pub fn cookie(mut self, k: &str, v: impl Into<String>) -> Self {
        self.cookies.retain(|c| c.0 != k);
        self.cookies.push((k.to_string(), v.into()));
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct HttpResponse {
    pub status: u16,
    /// the final URL
    pub url: String,
    /// names in lower case
    pub headers: Vec<(String, String)>,
    pub text: String,
    /// the response's Set-Cookie name=value pairs
    pub cookies: Vec<(String, String)>,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|h| h.0.eq_ignore_ascii_case(name)).map(|h| h.1.as_str())
    }

    pub fn cookie(&self, name: &str) -> Option<&str> {
        self.cookies.iter().rev().find(|c| c.0 == name).map(|c| c.1.as_str())
    }

    pub fn json(&self) -> Result<serde_json::Value, EngineError> {
        serde_json::from_str(&self.text).map_err(|e| EngineError::Parse(format!("JSONDecodeError: {e}")))
    }
}

/// How an engine's search ended when it gave no results (searx/exceptions.py and the network's errors).
#[derive(Clone, Debug, PartialEq)]
pub enum EngineError {
    /// SearxEngineCaptchaException; the suspension in seconds (None: the setting's default)
    Captcha { message: String, suspend: Option<u64> },
    /// SearxEngineAccessDeniedException
    AccessDenied { message: String, suspend: Option<u64> },
    /// SearxEngineTooManyRequestsException
    TooManyRequests,
    /// SearxEngineAPIException
    Api(String),
    /// SearxEngineResponseException: the page was not what the engine expects
    Response(String),
    /// an HTTP error status the engine does not handle (curl_cffi's HTTPError)
    Http(u16),
    /// the request failed (connection, TLS, protocol): curl_cffi's RequestException
    Network(String),
    /// the engine could not read what came back (an exception in its code)
    Parse(String),
}

impl EngineError {
    /// The reason in "unresponsive" (the exception's name, as SearXNG names the error).
    pub fn reason(&self) -> String {
        match self {
            EngineError::Captcha { .. } => "SearxEngineCaptchaException".into(),
            EngineError::AccessDenied { .. } => "SearxEngineAccessDeniedException".into(),
            EngineError::TooManyRequests => "SearxEngineTooManyRequestsException".into(),
            EngineError::Api(_) => "SearxEngineAPIException".into(),
            EngineError::Response(_) => "SearxEngineResponseException".into(),
            EngineError::Http(_) => "HTTPError".into(),
            EngineError::Network(kind) => kind.clone(),
            EngineError::Parse(_) => "ParseError".into(),
        }
    }

    pub fn message(&self) -> String {
        match self {
            EngineError::Captcha { message, .. } | EngineError::AccessDenied { message, .. } => message.clone(),
            EngineError::TooManyRequests => "Too many request".into(),
            EngineError::Api(m) | EngineError::Response(m) | EngineError::Network(m) | EngineError::Parse(m) => {
                m.clone()
            }
            EngineError::Http(s) => format!("HTTP error {s}"),
        }
    }

    /// OnlineProcessor.search / handle_exception: how long the engine is suspended after this error (None: not).
    /// The access-denied family carries its time (its setting's default when not given); a network error or an
    /// HTTP error suspends for min(ban_time_on_fail, max_ban_time_on_fail); an exception in the engine's own code
    /// does not suspend.
    pub fn suspend_seconds(&self) -> Option<u64> {
        match self {
            EngineError::Captcha { suspend, .. } => Some(suspend.unwrap_or(SUSPEND_CAPTCHA)),
            EngineError::AccessDenied { suspend, .. } => Some(suspend.unwrap_or(SUSPEND_ACCESS_DENIED)),
            EngineError::TooManyRequests => Some(SUSPEND_TOO_MANY_REQUESTS),
            EngineError::Network(_) | EngineError::Http(_) => Some(BAN_TIME_ON_FAIL.min(MAX_BAN_TIME_ON_FAIL)),
            EngineError::Api(_) | EngineError::Response(_) | EngineError::Parse(_) => None,
        }
    }

    pub fn captcha(message: impl Into<String>, suspend: Option<u64>) -> EngineError {
        EngineError::Captcha { message: message.into(), suspend }
    }
}

/// raise_for_httperror: Cloudflare's and Google's challenges, 402/403 access denied, 429 too many requests,
/// any other status from 400 on an HTTP error.
pub fn raise_for_httperror(resp: &HttpResponse) -> Result<(), EngineError> {
    let s = resp.status;
    if s < 400 {
        return Ok(());
    }
    if resp.header("server").is_some_and(|v| v.starts_with("cloudflare")) {
        let t = &resp.text;
        let challenge = ((s == 429 || s == 503)
            && (t.contains("__cf_chl_jschl_tk__=")
                || (t.contains("/cdn-cgi/challenge-platform/")
                    && t.contains("orchestrate/jsch/v1")
                    && t.contains("window._cf_chl_enter("))))
            || (s == 403 && t.contains("__cf_chl_captcha_tk__="));
        if challenge {
            return Err(EngineError::captcha("Cloudflare CAPTCHA", Some(SUSPEND_CF_CAPTCHA)));
        }
        if s == 403 && t.contains("<span class=\"cf-error-code\">1020</span>") {
            return Err(EngineError::AccessDenied {
                message: "Cloudflare Firewall".into(),
                suspend: Some(SUSPEND_CF_ACCESS_DENIED),
            });
        }
    }
    if s == 503 && resp.text.contains("\"https://www.google.com/recaptcha/") {
        return Err(EngineError::captcha("ReCAPTCHA", Some(SUSPEND_RECAPTCHA)));
    }
    if s == 402 || s == 403 {
        return Err(EngineError::AccessDenied { message: format!("HTTP error {s}"), suspend: None });
    }
    if s == 429 {
        return Err(EngineError::TooManyRequests);
    }
    Err(EngineError::Http(s))
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// The network as engines see it.
pub trait Fetch {
    fn fetch(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, EngineError>>;
    /// The browser's User-Agent the requests carry (DuckDuckGo keys its vqd tokens by it).
    fn user_agent(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(status: u16, text: &str) -> HttpResponse {
        HttpResponse { status, text: text.into(), ..Default::default() }
    }

    #[test]
    fn status_mapping() {
        assert!(raise_for_httperror(&resp(200, "")).is_ok());
        assert!(raise_for_httperror(&resp(302, "")).is_ok());
        assert_eq!(raise_for_httperror(&resp(403, "")).unwrap_err().suspend_seconds(), Some(86400));
        assert_eq!(raise_for_httperror(&resp(429, "")).unwrap_err(), EngineError::TooManyRequests);
        assert_eq!(raise_for_httperror(&resp(500, "")).unwrap_err(), EngineError::Http(500));
        assert_eq!(
            raise_for_httperror(&resp(503, "x \"https://www.google.com/recaptcha/ y")).unwrap_err().suspend_seconds(),
            Some(604800)
        );
        let mut cf = resp(403, "<span class=\"cf-error-code\">1020</span>");
        cf.headers.push(("server".into(), "cloudflare".into()));
        assert_eq!(raise_for_httperror(&cf).unwrap_err().reason(), "SearxEngineAccessDeniedException");
        assert_eq!(EngineError::Network("ConnectionError".into()).suspend_seconds(), Some(5));
        assert_eq!(EngineError::Parse("x".into()).suspend_seconds(), None);
    }
}
