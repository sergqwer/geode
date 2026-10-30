//! The chat's Linux as its file tools see it, without proot: guest paths (what the model reads and writes) mapped to
//! the phone's own folders, exactly as Geode's proot command binds them - /workspace, /root, /usr/local and /tmp are
//! the chat's, the user's shared skills appear at /root/.agents/skills, everything else is the image. Symbolic links
//! resolve inside that namespace: an absolute target starts again from the guest root, so nothing leads out.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Mount {
    pub guest: String,
    pub host: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Namespace {
    /// Longest guest prefix first.
    mounts: Vec<Mount>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// The canonical guest path (no `.`, `..` or symbolic links in its directories).
    pub guest: String,
    pub host: PathBuf,
}

const MAX_LINKS: usize = 40;

/// Split a guest path into its components ("/a//b/" -> ["a", "b"]).
fn components(p: &str) -> Vec<&str> {
    p.split('/').filter(|c| !c.is_empty()).collect()
}

pub fn join_guest(base: &str, rel: &str) -> String {
    if rel.starts_with('/') {
        rel.to_string()
    } else if base.ends_with('/') {
        format!("{base}{rel}")
    } else {
        format!("{base}/{rel}")
    }
}

impl Namespace {
    pub fn new(chat: &Path, rootfs: Option<&Path>, skills: Option<&Path>) -> Namespace {
        let mut mounts = vec![
            Mount { guest: "/workspace".into(), host: chat.join("workspace") },
            Mount { guest: "/root".into(), host: chat.join("home") },
            Mount { guest: "/usr/local".into(), host: chat.join("local") },
            Mount { guest: "/tmp".into(), host: chat.join("tmp") },
        ];
        if let Some(s) = skills {
            mounts.push(Mount { guest: "/root/.agents/skills".into(), host: s.to_path_buf() });
        }
        if let Some(r) = rootfs {
            mounts.push(Mount { guest: "/".into(), host: r.to_path_buf() });
        }
        mounts.sort_by_key(|m| std::cmp::Reverse(m.guest.len()));
        Namespace { mounts }
    }

    pub fn mounts(&self) -> &[Mount] {
        &self.mounts
    }

    /// The host path of a canonical guest path (None: outside every mount, as when the Linux is not installed).
    pub fn host_of(&self, guest: &str) -> Option<PathBuf> {
        for m in &self.mounts {
            let rest = if m.guest == "/" {
                Some(guest.trim_start_matches('/'))
            } else if guest == m.guest {
                Some("")
            } else {
                guest.strip_prefix(&m.guest).and_then(|r| r.strip_prefix('/'))
            };
            if let Some(rest) = rest {
                let mut p = m.host.clone();
                for c in components(rest) {
                    p.push(c);
                }
                return Some(p);
            }
        }
        None
    }

    /// The guest path of a host path inside a mount (the reverse of [host_of]).
    pub fn guest_of(&self, host: &Path) -> Option<String> {
        let mut best: Option<(usize, String)> = None;
        for m in &self.mounts {
            if let Ok(rest) = host.strip_prefix(&m.host) {
                let depth = m.host.components().count();
                let rel: Vec<String> =
                    rest.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
                let guest = if rel.is_empty() {
                    m.guest.clone()
                } else if m.guest == "/" {
                    format!("/{}", rel.join("/"))
                } else {
                    format!("{}/{}", m.guest, rel.join("/"))
                };
                if best.as_ref().is_none_or(|(d, _)| depth > *d) {
                    best = Some((depth, guest));
                }
            }
        }
        best.map(|(_, g)| g)
    }

    /// Resolve a path the model gave (absolute, or relative to [cwd]) to its canonical guest path and the host path
    /// behind it. Every directory on the way is followed through its symbolic links; [follow_last] follows the last
    /// component too (reading); a missing component ends the walk, the rest kept as written.
    pub fn resolve(&self, path: &str, cwd: &str, follow_last: bool) -> Result<Resolved, String> {
        let abs = join_guest(cwd, path);
        let mut pending: Vec<String> = components(&abs).into_iter().rev().map(String::from).collect();
        let mut done: Vec<String> = Vec::new();
        let mut hops = 0;
        while let Some(c) = pending.pop() {
            match c.as_str() {
                "." => continue,
                ".." => {
                    done.pop();
                    continue;
                }
                _ => {}
            }
            let candidate = format!("/{}", [done.as_slice(), std::slice::from_ref(&c)].concat().join("/"));
            let last = pending.is_empty();
            if last && !follow_last {
                done.push(c);
                break;
            }
            let Some(host) = self.host_of(&candidate) else {
                done.push(c);
                continue;
            };
            match std::fs::symlink_metadata(&host) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    hops += 1;
                    if hops > MAX_LINKS {
                        return Err(format!("cannot resolve \"{abs}\": too many levels of symbolic links"));
                    }
                    let target = std::fs::read_link(&host).map_err(|e| format!("cannot resolve \"{abs}\": {e}"))?;
                    let target = target.to_string_lossy().replace('\\', "/");
                    if target.starts_with('/') {
                        // a link proot left (an emulated hard link) names a host path: back into the guest
                        let target = self.guest_of(Path::new(&target)).unwrap_or(target);
                        done.clear();
                        for t in components(&target).into_iter().rev() {
                            pending.push(t.to_string());
                        }
                    } else {
                        for t in components(&target).into_iter().rev() {
                            pending.push(t.to_string());
                        }
                    }
                }
                Ok(_) => done.push(c),
                Err(_) => {
                    // missing: the rest is taken as written (a file to create, a path not found)
                    done.push(c);
                    while let Some(rest) = pending.pop() {
                        match rest.as_str() {
                            "." => {}
                            ".." => {
                                done.pop();
                            }
                            _ => done.push(rest),
                        }
                    }
                }
            }
        }
        let guest = format!("/{}", done.join("/"));
        let host = self
            .host_of(&guest)
            .ok_or_else(|| format!("cannot resolve \"{abs}\": the agent's Linux is not installed"))?;
        Ok(Resolved { guest, host })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(dir: &Path) -> Namespace {
        std::fs::create_dir_all(dir.join("chat/workspace/sub")).unwrap();
        std::fs::create_dir_all(dir.join("chat/home")).unwrap();
        std::fs::create_dir_all(dir.join("rootfs/usr/bin")).unwrap();
        std::fs::create_dir_all(dir.join("skills/hello")).unwrap();
        Namespace::new(&dir.join("chat"), Some(&dir.join("rootfs")), Some(&dir.join("skills")))
    }

    #[test]
    fn mapping() {
        let t = tempdir("paths-map");
        let n = ns(&t);
        let r = n.resolve("sub/../a.txt", "/workspace", true).unwrap();
        assert_eq!(r.guest, "/workspace/a.txt");
        assert_eq!(r.host, t.join("chat").join("workspace").join("a.txt"));
        assert_eq!(
            n.resolve("/usr/bin/python3", "/", true).unwrap().host,
            t.join("rootfs").join("usr").join("bin").join("python3")
        );
        assert_eq!(n.resolve("/root/.agents/skills/hello", "/", true).unwrap().host, t.join("skills").join("hello"));
        assert_eq!(n.resolve("/root/x", "/", true).unwrap().host, t.join("chat").join("home").join("x"));
        assert_eq!(n.guest_of(&t.join("chat").join("tmp").join("o.txt")).unwrap(), "/tmp/o.txt");
        assert_eq!(n.resolve("../../..", "/workspace/sub", true).unwrap().guest, "/");
    }

    #[test]
    fn no_linux() {
        let t = tempdir("paths-nolinux");
        let n = Namespace::new(&t.join("chat"), None, None);
        assert!(n.resolve("/etc/hosts", "/workspace", true).is_err());
        assert!(n.resolve("/workspace/x", "/", true).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn links_stay_inside() {
        let t = tempdir("paths-links");
        let n = ns(&t);
        std::os::unix::fs::symlink("/usr/bin", t.join("chat/workspace/bin")).unwrap();
        let r = n.resolve("/workspace/bin/python3", "/", true).unwrap();
        assert_eq!(r.guest, "/usr/bin/python3");
        assert_eq!(r.host, t.join("rootfs/usr/bin/python3"));
    }

    pub fn tempdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("geode-agent-test-{name}-{}", crate::util::hex(&crate::util::random_bytes::<4>())));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
