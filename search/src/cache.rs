// SPDX-License-Identifier: AGPL-3.0-or-later
//! What an engine keeps between searches (searx.enginelib.EngineCache: DuckDuckGo's vqd tokens, Mojeek's captcha
//! cookie, Startpage's sc code and Anubis pass) and which engines are suspended (searx.search.processors.abstract
//! SuspendedStatus) — in one JSON file in the folder given by --cache (the app passes files/search), or only in
//! memory for one search without it.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// ExpireCacheCfg.MAXHOLD_TIME: a value set without an expiry is held 7 days
pub const MAXHOLD_TIME: u64 = 60 * 60 * 24 * 7;

#[derive(Default, Serialize, Deserialize)]
struct State {
    /// engine -> key -> (value, expires at, unix seconds)
    #[serde(default)]
    values: BTreeMap<String, BTreeMap<String, (String, u64)>>,
    /// engine -> suspension
    #[serde(default)]
    suspended: BTreeMap<String, Suspended>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Suspended {
    pub until: u64,
    pub reason: String,
    pub continuous_errors: u32,
}

pub struct Cache {
    path: Option<PathBuf>,
    state: RefCell<State>,
    dirty: RefCell<bool>,
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Cache {
    pub fn memory() -> Cache {
        Cache { path: None, state: RefCell::new(State::default()), dirty: RefCell::new(false) }
    }

    /// The cache in `dir` (created if missing); a file that cannot be read is started afresh.
    pub fn open(dir: impl Into<PathBuf>) -> Cache {
        let dir: PathBuf = dir.into();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("geode-search.json");
        let mut state: State =
            std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        // drop what has expired
        let now = unix_now();
        for kv in state.values.values_mut() {
            kv.retain(|_, (_, exp)| *exp > now);
        }
        state.values.retain(|_, kv| !kv.is_empty());
        Cache { path: Some(path), state: RefCell::new(state), dirty: RefCell::new(false) }
    }

    pub fn get(&self, engine: &str, key: &str) -> Option<String> {
        let now = unix_now();
        let st = self.state.borrow();
        let (v, exp) = st.values.get(engine)?.get(key)?;
        (*exp > now).then(|| v.clone())
    }

    /// EngineCache.set: `expire` seconds (None: MAXHOLD_TIME)
    pub fn set(&self, engine: &str, key: &str, value: &str, expire: Option<u64>) {
        let exp = unix_now() + expire.unwrap_or(MAXHOLD_TIME).min(MAXHOLD_TIME);
        self.state
            .borrow_mut()
            .values
            .entry(engine.to_string())
            .or_default()
            .insert(key.to_string(), (value.to_string(), exp));
        *self.dirty.borrow_mut() = true;
    }

    /// The suspension still in force for an engine.
    pub fn suspended(&self, engine: &str) -> Option<Suspended> {
        let st = self.state.borrow();
        // whole seconds: a zero suspension (DuckDuckGo's captcha) is over at once
        st.suspended.get(engine).filter(|s| s.until > unix_now()).cloned()
    }

    /// SuspendedStatus.suspend
    pub fn suspend(&self, engine: &str, seconds: u64, reason: &str) {
        let mut st = self.state.borrow_mut();
        let s = st.suspended.entry(engine.to_string()).or_default();
        s.continuous_errors += 1;
        s.until = unix_now() + seconds;
        s.reason = reason.to_string();
        *self.dirty.borrow_mut() = true;
    }

    /// SuspendedStatus.resume: after an answer
    pub fn resume(&self, engine: &str) {
        if self.state.borrow_mut().suspended.remove(engine).is_some() {
            *self.dirty.borrow_mut() = true;
        }
    }

    /// Written aside and renamed into place, only if something changed.
    pub fn save(&self) {
        let Some(path) = &self.path else { return };
        if !*self.dirty.borrow() {
            return;
        }
        let tmp = path.with_extension("json.tmp");
        if let Ok(b) = serde_json::to_vec(&*self.state.borrow())
            && std::fs::write(&tmp, b).is_ok()
        {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_suspensions_survive_a_save() {
        let dir = std::env::temp_dir().join(format!("geode-search-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let c = Cache::open(&dir);
            c.set("duckduckgo", "vqd//q", "4-123", Some(3600));
            c.set("x", "gone", "v", Some(0));
            c.suspend("google", 86400, "SearxEngineAccessDeniedException");
            c.suspend("mojeek", 0, "x");
            c.save();
        }
        let c = Cache::open(&dir);
        assert_eq!(c.get("duckduckgo", "vqd//q").as_deref(), Some("4-123"));
        assert_eq!(c.get("x", "gone"), None);
        assert_eq!(c.suspended("google").unwrap().reason, "SearxEngineAccessDeniedException");
        // a zero suspension (DuckDuckGo's captcha) is over at once
        assert!(c.suspended("mojeek").is_none());
        c.resume("google");
        assert!(c.suspended("google").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
