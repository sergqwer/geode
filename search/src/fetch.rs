// SPDX-License-Identifier: AGPL-3.0-or-later
//! One page fetched for the agent's web_fetch (geode-search --fetch URL), with the same Chrome-on-Android client as
//! the searches. The rules are DeepSeek Harness's anonymous fetch backend (@deepseek-ai/dsh-web-fetch-http, MIT):
//! http(s) only, no credentials, public destinations only (every address a name resolves to checked, the
//! connection pinned to them), same-origin redirects only (at most 5), text and HTML only, 5 MB at most, 30 s.
//! Its errors are worded as the harness words them.

use futures_util::StreamExt;
use serde_json::json;
use crate::fetchrules::{is_public, kind_of, looks_like_text, may_be_text};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::time::Duration;
use wreq::redirect::Policy;
use wreq_util::{Emulation, Platform, Profile};

const MAX_URL_LENGTH: usize = 2048;
const MAX_REDIRECTS: usize = 5;
const MAX_RESPONSE_BYTES: usize = 5_000_000;
const TIMEOUT: Duration = Duration::from_secs(30);
const ACCEPT: &str = "text/html,application/xhtml+xml,text/*;q=0.9,application/json;q=0.8";

struct Target {
    url: url::Url,
    host: String,
    addrs: Vec<SocketAddr>,
}

fn validate(input: &str) -> Result<Target, String> {
    if input.len() > MAX_URL_LENGTH {
        return Err(format!("URL exceeds the maximum length of {MAX_URL_LENGTH}"));
    }
    let url = url::Url::parse(input).map_err(|_| format!("invalid URL: {input}"))?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(format!("unsupported URL scheme \"{}:\" (only http and https are allowed)", url.scheme()));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("credentials in URLs are not allowed".into());
    }
    let host = url.host_str().ok_or_else(|| format!("invalid URL: {input}"))?.trim_matches(['[', ']']).to_string();
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs: Vec<SocketAddr> = match host.parse::<IpAddr>() {
        Ok(ip) => vec![SocketAddr::new(ip, port)],
        Err(_) => (host.as_str(), port).to_socket_addrs().map_err(|e| format!("web fetch failed: Error: getaddrinfo {} {host}", dns_code(&e)))?.collect(),
    };
    if addrs.is_empty() {
        return Err(format!("hostname \"{host}\" resolved to no addresses"));
    }
    for a in &addrs {
        if !is_public(a.ip()) {
            return Err(if host.parse::<IpAddr>().is_ok() {
                format!("URL hostname \"{host}\" is a non-public IP address")
            } else {
                format!("URL hostname \"{host}\" resolves to a non-public IP address")
            });
        }
    }
    Ok(Target { url, host, addrs })
}

fn dns_code(e: &std::io::Error) -> &'static str {
    let m = e.to_string().to_ascii_lowercase();
    if m.contains("temporary") || m.contains("try again") { "EAI_AGAIN" } else { "ENOTFOUND" }
}

fn origin(u: &url::Url) -> String {
    format!("{}://{}", u.scheme(), u.host_str().unwrap_or("")) + &u.port().map(|p| format!(":{p}")).unwrap_or_default()
}

fn net_error(e: wreq::Error) -> String {
    if e.is_timeout() { "web fetch timed out".into() } else { format!("web fetch failed: {e}") }
}

/// The page: {url, statusCode, kind (html|text), content, truncated} as JSON text.
pub async fn fetch(input: &str) -> Result<String, String> {
    let mut target = validate(input)?;
    let mut redirects = 0;
    loop {
        let emulation = Emulation::builder().profile(Profile::Chrome149).platform(Platform::Android).build();
        let client = wreq::Client::builder()
            .emulation(emulation)
            .redirect(Policy::none())
            .resolve_to_addrs(target.host.clone(), target.addrs.clone())
            .timeout(TIMEOUT)
            .build()
            .map_err(|e| format!("web fetch failed: {e}"))?;
        let resp = client.get(target.url.as_str()).header("accept", ACCEPT).send().await.map_err(net_error)?;
        let status = resp.status().as_u16();
        if (300..400).contains(&status) && status != 304 {
            let location = resp.headers().get("location").and_then(|v| v.to_str().ok()).map(String::from);
            let Some(location) = location else {
                return Err(format!("redirect response (HTTP {status}) without a Location header"));
            };
            if redirects >= MAX_REDIRECTS {
                return Err(format!("exceeded the maximum of {MAX_REDIRECTS} redirects"));
            }
            let next = target.url.join(&location).map_err(|_| format!("invalid redirect Location \"{location}\""))?;
            let validated = validate(next.as_str())?;
            if origin(&validated.url) != origin(&target.url) {
                return Err(format!(
                    "cross-origin redirect to {} is not followed automatically; retry against that URL directly",
                    validated.url
                ));
            }
            target = validated;
            redirects += 1;
            continue;
        }
        let content_type = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).map(String::from);
        // a file served as bytes (GitHub's release assets: application/octet-stream) is read if it turns out to be text
        let sniff = may_be_text(content_type.as_deref());
        let Some(kind) = kind_of(content_type.as_deref()).or(sniff.then_some("text")) else {
            return Err(format!("unsupported content type \"{}\"", content_type.as_deref().unwrap_or("unknown")));
        };
        if resp.content_length().is_some_and(|l| l as usize > MAX_RESPONSE_BYTES) {
            return Err(format!("response exceeds the maximum of {MAX_RESPONSE_BYTES} bytes"));
        }
        let final_url = resp.uri().to_string();
        let mut body: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(net_error)?;
            if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
                return Err(format!("response exceeds the maximum of {MAX_RESPONSE_BYTES} bytes"));
            }
            body.extend_from_slice(&chunk);
        }
        let charset = content_type
            .as_deref()
            .and_then(|ct| ct.split(';').skip(1).find_map(|p| p.trim().strip_prefix("charset=").map(|c| c.trim_matches('"').to_string())));
        let encoding = match charset {
            Some(c) => encoding_rs::Encoding::for_label(c.as_bytes()).ok_or_else(|| format!("unsupported charset \"{c}\""))?,
            None => encoding_rs::UTF_8,
        };
        if sniff && !looks_like_text(&body) {
            return Err(format!(
                "a binary file (content type \"{}\"), not text",
                content_type.as_deref().unwrap_or("unknown")
            ));
        }
        let (text, _, _) = encoding.decode(&body);
        return Ok(json!({"url": final_url, "statusCode": status, "kind": kind, "content": text, "truncated": false}).to_string());
    }
}
