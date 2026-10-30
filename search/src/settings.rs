// SPDX-License-Identifier: AGPL-3.0-or-later
//! The engines and the limits of geode-search's settings.yml (tools/search/settings.yml), with the defaults
//! SearXNG's searx/engines/__init__.py (ENGINE_DEFAULT_ARGS) and settings_defaults.py give what it leaves out.

/// The engine module an entry runs (several entries could share one, as stackoverflow does stackexchange).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Module {
    DuckDuckGo,
    Brave,
    Google,
    Bing,
    Startpage,
    Mojeek,
    Wikipedia,
    GitHub,
    StackExchange,
    Arxiv,
    PypiJson,
}

#[derive(Debug)]
pub struct Engine {
    pub name: &'static str,
    pub module: Module,
    pub shortcut: &'static str,
    /// The first one is the result's category when the list is grouped (ResultContainer.get_ordered_results).
    pub categories: &'static [&'static str],
    /// Not asked unless named (a bang for it, or --engines).
    pub disabled: bool,
    /// Seconds; the search waits for the longest of the engines it asks (up to MAX_REQUEST_TIMEOUT).
    pub timeout: f64,
    pub weight: f64,
}

/// outgoing.request_timeout
pub const REQUEST_TIMEOUT: f64 = 3.0;
/// outgoing.max_request_timeout
pub const MAX_REQUEST_TIMEOUT: f64 = 6.0;
/// search.ban_time_on_fail / search.max_ban_time_on_fail: the suspension after a network error
pub const BAN_TIME_ON_FAIL: u64 = 5;
pub const MAX_BAN_TIME_ON_FAIL: u64 = 120;
/// search.suspended_times (settings_defaults.py)
pub const SUSPEND_ACCESS_DENIED: u64 = 86400;
pub const SUSPEND_CAPTCHA: u64 = 86400;
pub const SUSPEND_TOO_MANY_REQUESTS: u64 = 3600;
pub const SUSPEND_CF_CAPTCHA: u64 = 1296000;
pub const SUSPEND_CF_ACCESS_DENIED: u64 = 86400;
pub const SUSPEND_RECAPTCHA: u64 = 604800;

/// categories_as_tabs: an engine in none of these gets "other" appended (engines/__init__.py load_engine)
pub const TABS: &[&str] = &["general", "it", "science"];

/// settings.yml's engines, in its order (the order engines are asked in and listed for a category).
pub static ENGINES: &[Engine] = &[
    Engine {
        name: "duckduckgo",
        module: Module::DuckDuckGo,
        shortcut: "ddg",
        categories: &["general", "web"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "brave",
        module: Module::Brave,
        shortcut: "br",
        categories: &["general", "web"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    // by bang only (!go !bi !sp): Google answers scripts with 403 / captchas on many addresses, Bing's ranking
    // dilutes the merged list, Startpage puts a proof-of-work wall (Anubis) in front of its results
    Engine {
        name: "google",
        module: Module::Google,
        shortcut: "go",
        categories: &["general", "web"],
        disabled: true,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "bing",
        module: Module::Bing,
        shortcut: "bi",
        categories: &["general", "web"],
        disabled: true,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "startpage",
        module: Module::Startpage,
        shortcut: "sp",
        categories: &["general", "web"],
        disabled: true,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "mojeek",
        module: Module::Mojeek,
        shortcut: "mjk",
        categories: &["general", "web"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "wikipedia",
        module: Module::Wikipedia,
        shortcut: "wp",
        categories: &["general"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    // by bang (!gh !so !arx !pypi) or category (!it !science)
    Engine {
        name: "github",
        module: Module::GitHub,
        shortcut: "gh",
        categories: &["it", "repos"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "stackoverflow",
        module: Module::StackExchange,
        shortcut: "so",
        categories: &["it", "q&a"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
    Engine {
        name: "arxiv",
        module: Module::Arxiv,
        shortcut: "arx",
        categories: &["science", "scientific publications"],
        disabled: false,
        timeout: 4.0,
        weight: 1.0,
    },
    // PyPI's search page wants JavaScript: the package the query names, from the JSON API; bang only
    Engine {
        name: "pypi",
        module: Module::PypiJson,
        shortcut: "pypi",
        categories: &["packages", "other"],
        disabled: false,
        timeout: REQUEST_TIMEOUT,
        weight: 1.0,
    },
];

pub fn engine(name: &str) -> Option<&'static Engine> {
    ENGINES.iter().find(|e| e.name == name)
}

pub fn by_shortcut(shortcut: &str) -> Option<&'static Engine> {
    ENGINES.iter().find(|e| e.shortcut == shortcut)
}

/// The engines of a category, in settings order (searx.engines.categories).
pub fn category(name: &str) -> Vec<&'static Engine> {
    ENGINES.iter().filter(|e| e.categories.contains(&name)).collect()
}

pub fn is_category(name: &str) -> bool {
    ENGINES.iter().any(|e| e.categories.contains(&name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_has_a_tab_category() {
        for e in ENGINES {
            assert!(e.categories.iter().any(|c| TABS.contains(c) || *c == "other"), "{}", e.name);
        }
    }

    #[test]
    fn shortcuts_are_unique() {
        for e in ENGINES {
            assert_eq!(ENGINES.iter().filter(|x| x.shortcut == e.shortcut).count(), 1, "{}", e.shortcut);
        }
    }
}
