//! Commands in the chat's Linux, run only when the agent runs one: proot from the app's native library folder with
//! exactly Geode's command line (Linux.kt start()), the chat's folders bound over the shared image. Output is kept as
//! dsh-bash-local keeps it: the last 64 000 bytes of each stream in memory, the whole stream in a file once it is
//! longer.

use crate::config::Config;
use crate::paths::Namespace;
use crate::util::{hex, random_bytes};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const MAX_OUTPUT_BYTES: usize = 64_000;
pub const DEFAULT_TIMEOUT_MS: u64 = 120_000;
pub const MAX_TIMEOUT_MS: u64 = 600_000;
const PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// A turn's cancellation: set by session/cancel, checked by everything the turn runs.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Captured {
    pub text: String,
    pub truncated: bool,
    /// The whole stream's file, as the model reaches it (a guest path).
    pub spill_path: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    pub stdout: Captured,
    pub stderr: Captured,
    pub exit_code: Option<i32>,
    pub signal: Option<String>,
    pub timed_out: bool,
    pub timeout_ms: u64,
    pub aborted: bool,
}

/// Runs the agent's commands (proot on the phone; a recording in tests).
pub trait Shell: Send + Sync {
    fn run(&self, command: &str, cwd: &str, timeout_ms: u64, cancel: &Cancel) -> Result<Outcome, String>;
    /// A long-running program with its stdio piped (an MCP server).
    fn spawn(&self, argv: &[String], env: &[(String, String)], cwd: &str) -> Result<Child, String>;
    /// The Linux held for a long-running program about to be spawned, released when the hold is dropped (the
    /// program ended). None: this shell starts no Linux.
    fn hold(&self) -> Option<LinuxHold> {
        None
    }
}

/// The app's side of the Linux running: it makes room before the first Linux process (the model gives back memory)
/// and takes it back after the last.
pub trait LinuxClient: Send + Sync {
    /// The Linux is about to start: returns once the app answered, or after [timeout] (it then starts anyway).
    fn starting(&self, timeout: Duration);
    /// No Linux process is left.
    fn stopped(&self);
}

/// How many Linux (proot) processes run - bash commands and MCP servers - and the app told when the first starts
/// and the last ends.
pub struct LinuxGate {
    running: Mutex<usize>,
    client: Mutex<Option<Arc<dyn LinuxClient>>>,
    timeout: Duration,
}

/// One running Linux process's share of the gate; dropping it counts the process out.
pub struct LinuxHold(Arc<LinuxGate>);

impl LinuxGate {
    pub fn new(timeout: Duration) -> Arc<LinuxGate> {
        Arc::new(LinuxGate { running: Mutex::new(0), client: Mutex::new(None), timeout })
    }

    /// Who is told (the ACP client, once the server is up).
    pub fn attach(&self, client: Arc<dyn LinuxClient>) {
        *self.client.lock().unwrap() = Some(client);
    }

    /// A process about to start: the first one waits for the app to make room. Starts that come together wait for
    /// the same answer (the count is locked while the app is asked).
    pub fn hold(self: &Arc<Self>) -> LinuxHold {
        let mut running = self.running.lock().unwrap();
        if *running == 0 {
            let client = self.client.lock().unwrap().clone();
            if let Some(c) = client {
                c.starting(self.timeout);
            }
        }
        *running += 1;
        LinuxHold(self.clone())
    }

    pub fn running(&self) -> usize {
        *self.running.lock().unwrap()
    }
}

impl Drop for LinuxHold {
    fn drop(&mut self) {
        let mut running = self.0.running.lock().unwrap();
        *running = running.saturating_sub(1);
        if *running == 0 {
            let client = self.0.client.lock().unwrap().clone();
            if let Some(c) = client {
                c.stopped();
            }
        }
    }
}

pub struct Proot {
    cfg: Config,
    ns: Namespace,
    gate: Arc<LinuxGate>,
}

impl Proot {
    pub fn new(cfg: Config, ns: Namespace, gate: Arc<LinuxGate>) -> Proot {
        Proot { cfg, ns, gate }
    }

    fn ready(&self) -> Result<(PathBuf, PathBuf), String> {
        let lib = self.cfg.libdir.clone().ok_or("the agent's Linux is not available (no native library folder)")?;
        let rootfs = self.cfg.rootfs.clone().ok_or("the agent's Linux is not installed")?;
        if !rootfs.join(".geode-rootfs").is_file() && !rootfs.join("bin").exists() {
            return Err("the agent's Linux is not installed: set it up in Geode first".into());
        }
        Ok((lib, rootfs))
    }

    /// proot's command line for [argv] in [cwd], as Linux.kt builds it.
    fn command(&self, argv: &[String], env: &[(String, String)], cwd: &str) -> Result<Command, String> {
        let (lib, rootfs) = self.ready()?;
        let chat = &self.cfg.chat;
        for d in ["workspace", "home", "local", "tmp"] {
            let _ = std::fs::create_dir_all(chat.join(d));
        }
        let mut c = Command::new(lib.join("libproot.so"));
        c.args(["--kill-on-exit", "--link2symlink", "--sysvipc", "-0", "-r"]).arg(&rootfs);
        c.args(["-b", "/dev", "-b", "/proc", "-b", "/sys", "-b", "/proc/self/fd:/dev/fd"]);
        let bind = |c: &mut Command, host: PathBuf, guest: &str| {
            c.arg("-b").arg(format!("{}:{guest}", host.display()));
        };
        bind(&mut c, chat.join("workspace"), "/workspace");
        bind(&mut c, chat.join("home"), "/root");
        bind(&mut c, chat.join("local"), "/usr/local");
        bind(&mut c, chat.join("tmp"), "/tmp");
        if let Some(s) = &self.cfg.skills {
            let _ = std::fs::create_dir_all(chat.join("home/.agents/skills"));
            bind(&mut c, s.clone(), "/root/.agents/skills");
        }
        c.args(["-w", cwd, "/usr/bin/env", "-i"]);
        c.args(["HOME=/root", "USER=root", "LANG=C.UTF-8", "TERM=dumb", "SHELL=/bin/bash", "TMPDIR=/tmp"]);
        c.arg(format!("PATH={PATH}"));
        for (k, v) in env {
            c.arg(format!("{k}={v}"));
        }
        c.args(argv);
        c.env_clear();
        c.env("PROOT_LOADER", lib.join("libproot-loader.so"));
        let tmp = self.cfg.proot_tmp.clone().unwrap_or_else(|| self.cfg.tmp());
        let _ = std::fs::create_dir_all(&tmp);
        c.env("PROOT_TMP_DIR", tmp);
        c.env("PATH", "/system/bin");
        Ok(c)
    }
}

const BASH_ENV: [(&str, &str); 3] = [("NO_COLOR", "1"), ("PAGER", "cat"), ("GIT_PAGER", "cat")];

impl Shell for Proot {
    fn run(&self, command: &str, cwd: &str, timeout_ms: u64, cancel: &Cancel) -> Result<Outcome, String> {
        let env: Vec<(String, String)> = BASH_ENV.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let argv = vec!["/bin/bash".to_string(), "-c".to_string(), command.to_string()];
        let mut c = self.command(&argv, &env, cwd)?;
        c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        // the Linux counted in for the command's whole run
        let _hold = self.gate.hold();
        let child = c.spawn().map_err(|e| format!("could not start the Linux: {e}"))?;
        Ok(supervise(child, timeout_ms, cancel, &self.cfg.tmp(), &self.ns))
    }

    fn hold(&self) -> Option<LinuxHold> {
        Some(self.gate.hold())
    }

    fn spawn(&self, argv: &[String], env: &[(String, String)], cwd: &str) -> Result<Child, String> {
        let mut c = self.command(argv, env, cwd)?;
        c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        c.spawn().map_err(|e| format!("could not start the Linux: {e}"))
    }
}

/// One stream kept as dsh keeps it: its last [MAX_OUTPUT_BYTES] in memory, all of it in a file once it is longer.
struct Keep {
    tail: Vec<u8>,
    total: usize,
    file: Option<(File, PathBuf)>,
    dir: PathBuf,
    name: String,
}

impl Keep {
    fn push(&mut self, chunk: &[u8]) {
        self.total += chunk.len();
        if self.file.is_none() && self.tail.len() + chunk.len() > MAX_OUTPUT_BYTES {
            let _ = std::fs::create_dir_all(&self.dir);
            let path = self.dir.join(&self.name);
            if let Ok(mut f) = File::create(&path) {
                let _ = f.write_all(&self.tail);
                self.file = Some((f, path));
            }
        }
        if let Some((f, _)) = &mut self.file {
            let _ = f.write_all(chunk);
        }
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > MAX_OUTPUT_BYTES * 2 {
            let cut = self.tail.len() - MAX_OUTPUT_BYTES;
            self.tail.drain(..cut);
        }
    }

    fn finish(mut self, ns: &Namespace) -> Captured {
        let truncated = self.total > MAX_OUTPUT_BYTES;
        if self.tail.len() > MAX_OUTPUT_BYTES {
            let cut = self.tail.len() - MAX_OUTPUT_BYTES;
            self.tail.drain(..cut);
        }
        let mut start = 0;
        if truncated {
            // a cut inside a character starts at the next whole one
            while start < self.tail.len() && (self.tail[start] & 0xC0) == 0x80 {
                start += 1;
            }
        }
        let text = String::from_utf8_lossy(&self.tail[start..]).into_owned();
        let spill_path = self.file.take().and_then(|(f, p)| {
            drop(f);
            ns.guest_of(&p)
        });
        Captured { text, truncated, spill_path }
    }
}

/// Waits for [child]: its output read as it comes, killed on timeout or cancel.
pub fn supervise(mut child: Child, timeout_ms: u64, cancel: &Cancel, tmp: &std::path::Path, ns: &Namespace) -> Outcome {
    let dir = tmp.join(format!("geode-subprocess-{}", hex(&random_bytes::<3>())));
    let tag = format!("geode-subprocess-{}-{}", child.id(), hex(&random_bytes::<6>()));
    let reader = |stream: Option<Box<dyn Read + Send>>, kind: &str| {
        let keep = Arc::new(Mutex::new(Keep {
            tail: Vec::new(),
            total: 0,
            file: None,
            dir: dir.clone(),
            name: format!("{tag}-{kind}.log"),
        }));
        let k = keep.clone();
        let handle = stream.map(|mut s| {
            thread::spawn(move || {
                let mut buf = [0u8; 16384];
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    k.lock().unwrap().push(&buf[..n]);
                }
            })
        });
        (keep, handle)
    };
    let (out, out_h) = reader(child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>), "stdout");
    let (err, err_h) = reader(child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>), "stderr");
    let started = Instant::now();
    let mut timed_out = false;
    let mut aborted = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) => {}
            Err(_) => break None,
        }
        if cancel.is_cancelled() {
            aborted = true;
            let _ = child.kill();
            break child.wait().ok();
        }
        if started.elapsed() >= Duration::from_millis(timeout_ms) {
            timed_out = true;
            let _ = child.kill();
            break child.wait().ok();
        }
        thread::sleep(Duration::from_millis(15));
    };
    // the output's readers end with the pipes; a descendant left holding them (it outlived a kill) is not waited for
    let deadline = Instant::now() + Duration::from_millis(if timed_out || aborted { 500 } else { 2000 });
    for h in [out_h, err_h].into_iter().flatten() {
        while !h.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if h.is_finished() {
            let _ = h.join();
        }
    }
    let take = |k: Arc<Mutex<Keep>>| match Arc::try_unwrap(k) {
        Ok(m) => m.into_inner().unwrap(),
        Err(a) => {
            let mut g = a.lock().unwrap();
            Keep {
                tail: std::mem::take(&mut g.tail),
                total: g.total,
                file: g.file.take(),
                dir: g.dir.clone(),
                name: g.name.clone(),
            }
        }
    };
    let (exit_code, signal) = match status {
        Some(s) => (s.code(), signal_name(&s)),
        None => (None, None),
    };
    Outcome {
        stdout: take(out).finish(ns),
        stderr: take(err).finish(ns),
        exit_code: if timed_out || aborted { None } else { exit_code },
        signal: if timed_out || aborted { None } else { signal },
        timed_out,
        timeout_ms,
        aborted,
    }
}

#[cfg(unix)]
fn signal_name(s: &std::process::ExitStatus) -> Option<String> {
    use std::os::unix::process::ExitStatusExt;
    s.signal().map(|n| match n {
        9 => "SIGKILL".into(),
        15 => "SIGTERM".into(),
        2 => "SIGINT".into(),
        6 => "SIGABRT".into(),
        11 => "SIGSEGV".into(),
        _ => format!("SIG{n}"),
    })
}

#[cfg(not(unix))]
fn signal_name(_: &std::process::ExitStatus) -> Option<String> {
    None
}

/// A host shell for development off the phone (GEODE_HOST_SHELL: a bash to run commands with, no proot).
pub struct HostShell {
    pub bash: PathBuf,
    pub ns: Namespace,
    pub tmp: PathBuf,
}

impl Shell for HostShell {
    fn run(&self, command: &str, cwd: &str, timeout_ms: u64, cancel: &Cancel) -> Result<Outcome, String> {
        let dir = self.ns.host_of(cwd).ok_or_else(|| format!("no such directory: {cwd}"))?;
        let child = Command::new(&self.bash)
            .args(["-c", command])
            .current_dir(dir)
            .envs(BASH_ENV)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        Ok(supervise(child, timeout_ms, cancel, &self.tmp, &self.ns))
    }

    fn spawn(&self, argv: &[String], env: &[(String, String)], cwd: &str) -> Result<Child, String> {
        let dir = self.ns.host_of(cwd).unwrap_or_else(|| PathBuf::from("."));
        Command::new(&argv[0])
            .args(&argv[1..])
            .envs(env.iter().cloned())
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bash() -> Option<PathBuf> {
        ["C:/Program Files/Git/bin/bash.exe", "/bin/bash"].iter().map(PathBuf::from).find(|p| p.exists())
    }

    fn shell(dir: &std::path::Path) -> Option<HostShell> {
        let ns = Namespace::new(&dir.join("chat"), None, None);
        std::fs::create_dir_all(dir.join("chat/workspace")).ok()?;
        Some(HostShell { bash: bash()?, ns, tmp: dir.join("chat/tmp") })
    }

    #[test]
    fn output_exit_timeout_cancel() {
        let dir = std::env::temp_dir().join(format!("geode-agent-linux-{}", crate::util::uuid()));
        let Some(sh) = shell(&dir) else { return };
        let c = Cancel::new();
        let o = sh.run("echo out; echo err >&2; exit 3", "/workspace", 10_000, &c).unwrap();
        assert_eq!((o.stdout.text.as_str(), o.stderr.text.as_str(), o.exit_code), ("out\n", "err\n", Some(3)));
        let t = Instant::now();
        let o = sh.run("sleep 5", "/workspace", 300, &c).unwrap();
        assert!(o.timed_out && t.elapsed() < Duration::from_secs(3));
        let c2 = Cancel::new();
        let c3 = c2.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            c3.cancel();
        });
        let t = Instant::now();
        let o = sh.run("sleep 5", "/workspace", 10_000, &c2).unwrap();
        assert!(o.aborted && t.elapsed() < Duration::from_secs(3));
        let o = sh.run("seq 1 40000", "/workspace", 20_000, &Cancel::new()).unwrap();
        assert!(o.stdout.truncated && o.stdout.text.starts_with("334\n29335\n"));
        assert!(o.stdout.spill_path.as_deref().unwrap().starts_with("/tmp/geode-subprocess-"));
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    /// A client that answers after [delay] and records what it was told.
    struct Recorder {
        delay: Duration,
        events: Mutex<Vec<&'static str>>,
    }

    impl LinuxClient for Recorder {
        fn starting(&self, _timeout: Duration) {
            self.events.lock().unwrap().push("starting");
            thread::sleep(self.delay);
            self.events.lock().unwrap().push("started");
        }
        fn stopped(&self) {
            self.events.lock().unwrap().push("stopped");
        }
    }

    fn gate(delay: Duration) -> (Arc<LinuxGate>, Arc<Recorder>) {
        let g = LinuxGate::new(Duration::from_secs(5));
        let r = Arc::new(Recorder { delay, events: Mutex::new(Vec::new()) });
        g.attach(r.clone());
        (g, r)
    }

    #[test]
    fn first_in_asks_last_out_tells() {
        let (g, r) = gate(Duration::ZERO);
        let mcp = g.hold(); // a stdio MCP server
        let bash = g.hold(); // a command while it runs
        assert_eq!(g.running(), 2);
        drop(bash);
        assert_eq!(*r.events.lock().unwrap(), ["starting", "started"]);
        drop(mcp);
        assert_eq!(*r.events.lock().unwrap(), ["starting", "started", "stopped"]);
        // the next command asks again
        drop(g.hold());
        assert_eq!(*r.events.lock().unwrap(), ["starting", "started", "stopped", "starting", "started", "stopped"]);
    }

    #[test]
    fn starts_together_wait_for_one_answer() {
        let (g, r) = gate(Duration::from_millis(150));
        let holds: Vec<_> = (0..8)
            .map(|_| {
                let g = g.clone();
                thread::spawn(move || {
                    let h = g.hold();
                    // every process starts only after the app made room
                    assert!(h.0.running() >= 1);
                    h
                })
            })
            .map(|t| t.join().unwrap())
            .collect();
        assert_eq!(g.running(), 8);
        assert_eq!(*r.events.lock().unwrap(), ["starting", "started"]);
        drop(holds);
        assert_eq!(g.running(), 0);
        assert_eq!(*r.events.lock().unwrap(), ["starting", "started", "stopped"]);
    }

    #[test]
    fn without_a_client_nothing_waits() {
        let g = LinuxGate::new(Duration::from_secs(5));
        let t = Instant::now();
        drop(g.hold());
        assert!(t.elapsed() < Duration::from_millis(50));
        assert_eq!(g.running(), 0);
    }
}
