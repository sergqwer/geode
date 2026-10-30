#!/bin/bash
# Geode's agent Linux: the official Debian trixie arm64 root filesystem and nothing added to it - the agent installs
# what a task needs itself (apt install python3, nodejs, git...; pip and npm once those are in). Built once on a
# developer's phone, inside the app's Linux, from the root filesystem freshly installed (LinuxDebugReceiver op
# install), and shipped whole as an archive (pack.sh).
#
#   adb shell am broadcast -n xyz.tifu.geode/.agent.LinuxDebugReceiver --es op run --el timeout 1800 \
#       --es cmd "'bash /workspace/build.sh > /workspace/build.log 2>&1'"
#
# What a chat's agent writes lives in the chat's own folders mounted over this image (/workspace, /root, /usr/local,
# /tmp: geode-agent); what it installs with apt goes into the system every chat's Linux shares.
set -euo pipefail

# apt and dpkg under proot: no sandbox user (proot's root is the app), no pty, no recommends, no translations
cat > /etc/apt/apt.conf.d/99geode <<'EOF'
APT::Sandbox::User "root";
Dpkg::Use-Pty "0";
APT::Install-Recommends "false";
APT::Install-Suggests "false";
Acquire::Languages "none";
EOF
# what a phone does not need of a package: its docs and translations (the licences stay)
cat > /etc/dpkg/dpkg.cfg.d/geode-slim <<'EOF'
path-exclude=/usr/share/doc/*
path-include=/usr/share/doc/*/copyright
path-exclude=/usr/share/man/*
path-exclude=/usr/share/info/*
path-exclude=/usr/share/locale/*
path-include=/usr/share/locale/locale.alias
path-exclude=/usr/share/lintian/*
EOF
rm -rf /usr/share/doc/*/[!c]* /usr/share/man/* /usr/share/info/* /usr/share/locale/[!l]*

# no package lists or caches: the agent's first install fetches the lists
apt-get clean
rm -rf /var/lib/apt/lists/* /var/log/apt/* /var/log/dpkg.log /var/cache/debconf/*-old /tmp/* /var/tmp/*
mkdir -p /etc/geode
echo "Geode agent Linux $(date -u +%Y-%m-%d), Debian $(cat /etc/debian_version)" > /etc/geode/image
cat /etc/geode/image
grep -h '^URIs' /etc/apt/sources.list.d/*.sources 2>/dev/null || cat /etc/apt/sources.list
du -sh / --exclude=/proc --exclude=/sys --exclude=/dev --exclude=/workspace 2>/dev/null | tail -1
