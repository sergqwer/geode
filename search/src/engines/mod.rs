// SPDX-License-Identifier: AGPL-3.0-or-later
//! The engines (searx/engines/*.py) and the request flow every online engine goes through
//! (searx/search/processors/online.py: the base parameters, the engine's request, the HTTP call, the status
//! check, the engine's response). Each engine's request and response are plain functions of what goes in and
//! comes back, so they are tested against pages saved from the phone.

pub mod arxiv;
pub mod bing;
pub mod brave;
pub mod duckduckgo;
pub mod github;
pub mod google;
pub mod mojeek;
pub mod pypi;
pub mod stackexchange;
pub mod startpage;
pub mod wikipedia;

use crate::cache::Cache;
use crate::http::{EngineError, Fetch, HttpRequest, HttpResponse, raise_for_httperror};
use crate::locales::{self, Row};
use crate::results::Item;
use crate::settings::{Engine, Module};
use std::cell::Cell;
use std::path::PathBuf;

/// RequestParams: what the search hands every engine.
#[derive(Clone, Debug)]
pub struct Params {
    /// searxng_locale: "all", "uk", "de-AT" ...
    pub locale: String,
    /// SearXNG's answers for that locale (each engine's region and language values)
    pub row: Row,
    /// the Accept-Language header (send_accept_language_header)
    pub accept_language: String,
    pub pageno: u32,
    pub safesearch: u8,
    pub category: String,
}

impl Params {
    pub fn new(locale: &str, category: &str) -> Params {
        Params {
            locale: locale.to_string(),
            row: locales::row(locale),
            accept_language: locales::accept_language(locale),
            pageno: 1,
            safesearch: 0,
            category: category.to_string(),
        }
    }

    /// A request carrying the base headers (OnlineProcessor.get_params).
    pub fn request(&self, url: impl Into<String>) -> HttpRequest {
        HttpRequest::get(url).header("Accept-Language", self.accept_language.clone())
    }
}

/// What an engine's search runs with: the network, the cache, and (debug) a folder for the raw responses.
pub struct Ctx<'a> {
    pub fetch: &'a dyn Fetch,
    pub cache: &'a Cache,
    pub dump: Option<PathBuf>,
    seq: Cell<u32>,
}

impl<'a> Ctx<'a> {
    pub fn new(fetch: &'a dyn Fetch, cache: &'a Cache, dump: Option<PathBuf>) -> Ctx<'a> {
        Ctx { fetch, cache, dump, seq: Cell::new(0) }
    }

    /// One HTTP call; `raise` checks the status as raise_for_httperror does.
    pub async fn send(&self, engine: &str, req: HttpRequest, raise: bool) -> Result<HttpResponse, EngineError> {
        let url = req.url.clone();
        let resp = self.fetch.fetch(req).await?;
        if let Some(dir) = &self.dump {
            let n = self.seq.get() + 1;
            self.seq.set(n);
            let _ = std::fs::create_dir_all(dir);
            let head = format!("<!-- {} {} -> {} -->\n", resp.status, url, resp.url);
            let _ = std::fs::write(dir.join(format!("{engine}-{n}.txt")), head + &resp.text);
        }
        if raise {
            raise_for_httperror(&resp)?;
        }
        Ok(resp)
    }
}

/// One engine's search: its request (None: nothing to ask, as DuckDuckGo for a query of 500 characters and more),
/// the call, its response.
pub async fn search(engine: &Engine, ctx: &Ctx<'_>, query: &str, p: &Params) -> Result<Vec<Item>, EngineError> {
    let name = engine.name;
    match engine.module {
        Module::DuckDuckGo => {
            let Some(req) = duckduckgo::request(query, p) else { return Ok(vec![]) };
            let form_q = duckduckgo::form_query(&req);
            let resp = ctx.send(name, req, true).await?;
            let (items, vqd) = duckduckgo::response(&resp)?;
            if let Some(vqd) = vqd {
                ctx.cache.set(name, &duckduckgo::vqd_key(&form_q, ctx.fetch.user_agent()), &vqd, Some(3600));
            }
            Ok(items)
        }
        Module::Brave => {
            let resp = ctx.send(name, brave::request(query, p), true).await?;
            brave::response(&resp)
        }
        Module::Google => {
            let resp = ctx.send(name, google::request(query, p), true).await?;
            google::response(&resp)
        }
        Module::Bing => {
            let resp = ctx.send(name, bing::request(query, p), true).await?;
            bing::response(&resp)
        }
        Module::Mojeek => mojeek::search(ctx, name, query, p).await,
        Module::Startpage => startpage::search(ctx, name, query, p).await,
        Module::Wikipedia => {
            let resp = ctx.send(name, wikipedia::request(query, p), false).await?;
            wikipedia::response(&resp)
        }
        Module::GitHub => {
            let resp = ctx.send(name, github::request(query, p), true).await?;
            github::response(&resp)
        }
        Module::StackExchange => {
            let resp = ctx.send(name, stackexchange::request(query, p, "stackoverflow"), true).await?;
            stackexchange::response(&resp, "stackoverflow")
        }
        Module::Arxiv => {
            let resp = ctx.send(name, arxiv::request(query, p), true).await?;
            arxiv::response(&resp)
        }
        Module::PypiJson => {
            let Some(req) = pypi::request(query, p) else { return Ok(vec![]) };
            let resp = ctx.send(name, req, false).await?;
            pypi::response(&resp)
        }
    }
}

/// A JSON value's string, "" when missing or not a string.
pub(crate) fn js(v: &serde_json::Value, key: &str) -> String {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("").to_string()
}
