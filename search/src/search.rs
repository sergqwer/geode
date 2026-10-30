// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/search/__init__.py (Search) as geode-search runs it: the engines the query selects, each one's request
//! sent at once on the shared client, all waited for up to the search's timeout (the longest of the engines'
//! own, at most max_request_timeout, or the query's `<N`), the answers merged as they arrive; an engine that
//! failed is listed as unresponsive and, for the errors SearXNG suspends on, suspended (kept in the cache folder).

use crate::cache::Cache;
use crate::engines::{self, Ctx, Params};
use crate::http::EngineError;
use crate::output;
use crate::query::{EngineRef, RawTextQuery, guess_lang};
use crate::results::ResultContainer;
use crate::settings::{self, ENGINES, MAX_REQUEST_TIMEOUT};
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use serde_json::Value;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Options {
    pub query: String,
    pub max: usize,
    pub lang: Option<String>,
    pub engines: Option<Vec<String>>,
    pub categories: Option<Vec<String>>,
    pub cache: Option<PathBuf>,
    pub dump: Option<PathBuf>,
    /// $GEODE_SEARCH_LANG: the phone's language
    pub phone_lang: Option<String>,
}

pub struct Outcome {
    pub json: Option<Value>,
    pub code: i32,
    pub message: Option<String>,
    /// per engine: name, milliseconds, the error if any
    pub timings: Vec<(String, u128, Option<String>)>,
}

fn fail(code: i32, message: &str) -> Outcome {
    Outcome { json: None, code, message: Some(format!("geode-search: {message}")), timings: vec![] }
}

/// The engines to ask: the query's bangs, else --engines, else the categories' (general by default) without the
/// disabled ones.
fn select(raw: &RawTextQuery, opts: &Options) -> Vec<EngineRef> {
    let mut refs: Vec<EngineRef> = if !raw.enginerefs.is_empty() {
        raw.enginerefs.clone()
    } else if let Some(names) = &opts.engines {
        names
            .iter()
            .filter_map(|n| settings::engine(n.trim()))
            .map(|e| EngineRef { name: e.name, category: e.categories.first().unwrap_or(&"general").to_string() })
            .collect()
    } else {
        let cats = opts.categories.clone().unwrap_or_else(|| vec!["general".into()]);
        cats.iter()
            .flat_map(|c| {
                settings::category(c.trim())
                    .into_iter()
                    .filter(|e| !e.disabled)
                    .map(move |e| EngineRef { name: e.name, category: c.trim().to_string() })
            })
            .collect()
    };
    let mut seen = Vec::new();
    refs.retain(|r| {
        let new = !seen.contains(&r.name);
        seen.push(r.name);
        new
    });
    refs
}

pub async fn run(opts: Options, fetch: &dyn crate::http::Fetch) -> Outcome {
    let started = Instant::now();
    let disabled: Vec<&str> = ENGINES.iter().filter(|e| e.disabled).map(|e| e.name).collect();
    let raw = RawTextQuery::parse(&opts.query, &disabled);
    let query = raw.query();
    if query.trim().is_empty() {
        return fail(2, "empty query");
    }
    let refs = select(&raw, &opts);
    if refs.is_empty() {
        return fail(2, "no engine to ask");
    }
    let lang = raw
        .languages
        .first()
        .cloned()
        .or(opts.lang.clone())
        .filter(|l| !l.is_empty() && l != "auto")
        .unwrap_or_else(|| guess_lang(&query, opts.phone_lang.as_deref()));

    let cache = match &opts.cache {
        Some(dir) => Cache::open(dir),
        None => Cache::memory(),
    };
    let ctx = Ctx::new(fetch, &cache, opts.dump.clone());
    let mut rc = ResultContainer::new();

    // _get_requests: the suspended engines are reported, the others asked; the timeout
    let mut asked = Vec::new();
    let mut default_timeout: f64 = 0.0;
    for r in &refs {
        let engine = settings::engine(r.name).expect("selected engines exist");
        if let Some(s) = cache.suspended(r.name) {
            rc.add_unresponsive(r.name, &s.reason, true);
            continue;
        }
        default_timeout = default_timeout.max(engine.timeout);
        asked.push((engine, Params::new(&lang, &r.category)));
    }
    let actual_timeout = match raw.timeout_limit {
        None => default_timeout.min(MAX_REQUEST_TIMEOUT),
        Some(q) => q.min(MAX_REQUEST_TIMEOUT),
    };
    let deadline = tokio::time::Instant::from_std(started) + Duration::from_secs_f64(actual_timeout);

    let mut pending: FuturesUnordered<_> = asked
        .iter()
        .map(|(engine, params)| {
            let ctx = &ctx;
            let query = &query;
            async move {
                let t = Instant::now();
                let r = tokio::time::timeout_at(deadline, engines::search(engine, ctx, query, params)).await;
                (*engine, t.elapsed().as_millis(), r)
            }
        })
        .collect();
    let mut timings = Vec::new();
    let mut answered = 0;
    while let Some((engine, ms, r)) = pending.next().await {
        match r {
            Ok(Ok(items)) => {
                answered += 1;
                rc.extend(engine.name, items);
                cache.resume(engine.name);
                timings.push((engine.name.to_string(), ms, None));
            }
            Ok(Err(e)) => {
                record_error(&mut rc, &cache, engine.name, &e);
                timings.push((engine.name.to_string(), ms, Some(format!("{}: {}", e.reason(), e.message()))));
            }
            Err(_) => {
                rc.add_unresponsive(engine.name, "timeout", false);
                timings.push((engine.name.to_string(), ms, Some("timeout".into())));
            }
        }
    }
    drop(pending);
    cache.save();

    let json = output::build(&query, &mut rc, opts.max);
    let empty = json["results"].as_array().is_none_or(|a| a.is_empty())
        && json["answers"].as_array().is_none_or(|a| a.is_empty());
    if empty && answered == 0 {
        let why: Vec<String> = rc.unresponsive.iter().map(|u| format!("{} ({})", u.engine, u.reason)).collect();
        return Outcome {
            json: Some(json),
            code: 1,
            message: Some(format!("geode-search: no engine answered: {}", why.join(", "))),
            timings,
        };
    }
    Outcome { json: Some(json), code: 0, message: None, timings }
}

/// handle_exception: the engine listed with its error, and suspended when SearXNG would
fn record_error(rc: &mut ResultContainer, cache: &Cache, engine: &str, e: &EngineError) {
    let reason = e.reason();
    rc.add_unresponsive(engine, &reason, false);
    if let Some(secs) = e.suspend_seconds() {
        cache.suspend(engine, secs, &reason);
    }
}
