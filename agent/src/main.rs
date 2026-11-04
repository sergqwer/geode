//! geode-agent: the ACP agent the app runs (`libgeodeagent.so`), configured by the environment Agent.kt sets.

use geode_agent::acp::{Out, Server};
use geode_agent::config::Config;
use geode_agent::linux::{HostShell, LinuxGate, Proot, Shell};
use geode_agent::model::HttpModel;
use geode_agent::paths::Namespace;
use geode_agent::tools::web::GeodeSearch;
use std::io::BufReader;
use std::sync::Arc;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--version") {
        println!("geode-agent {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let cfg = Config::from_env();
    let ns = Namespace::new(&cfg.chat, cfg.rootfs.as_deref(), cfg.skills.as_deref()).with_folders(&cfg.folders);
    // the app makes room for the Linux before its first process (geode/linux); it waits at most this long
    let gate = LinuxGate::new(std::time::Duration::from_secs(5));
    let shell: Box<dyn Shell> = match std::env::var_os("GEODE_HOST_SHELL") {
        Some(bash) => Box::new(HostShell { bash: bash.into(), ns: ns.clone(), tmp: cfg.tmp() }),
        None => Box::new(Proot::new(cfg.clone(), ns.clone(), gate.clone())),
    };
    let web = GeodeSearch {
        exe: cfg.libdir.clone().unwrap_or_default().join("libgeodesearch.so"),
        cache: cfg.search_cache.clone(),
        lang: std::env::var("GEODE_SEARCH_LANG").ok(),
    };
    let model = HttpModel { base_url: cfg.base_url.clone(), key: cfg.key.clone() };
    let out = Arc::new(Out::new(Box::new(std::io::stdout())));
    let server = Server::new(cfg, ns, shell, Box::new(web), Box::new(model), out);
    gate.attach(server.peer());
    server.serve(BufReader::new(std::io::stdin()));
}
