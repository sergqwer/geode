#!/bin/bash
# Packs the finished image (build.sh) into the archive Geode ships: the root filesystem as it is - its package sources
# on, the agent installs from them - the phone's own mounts and the chats' folders left out. Run inside the app's
# Linux; the tar lands in /workspace/geode-linux.tar (the base system has no xz: tools/linux/publish.py compresses it
# on the PC).
set -euo pipefail
# no package lists, caches or logs (build.sh's last step, again: harmless when done)
apt-get clean
rm -rf /var/lib/apt/lists/* /var/log/apt/* /var/log/dpkg.log /root/.cache /tmp/* /var/tmp/*
echo "Geode agent Linux $(date -u +%Y-%m-%d), Debian $(cat /etc/debian_version)" > /etc/geode/image
cat /etc/geode/image
# the image's version: a chat whose Linux last ran another one runs this image's /etc/geode/migrate first (Agent.kt)
date -u +%Y%m%dT%H%M%SZ > /etc/geode/version
echo "version: $(cat /etc/geode/version)"
echo "image: $(du -sh / --exclude=/proc --exclude=/sys --exclude=/dev --exclude=/workspace 2>/dev/null | tail -1 | cut -f1)"
cd /
tar --numeric-owner --one-file-system -cf /workspace/geode-linux.tar \
    --exclude=./proc --exclude=./sys --exclude=./dev --exclude=./workspace --exclude=./tmp --exclude=./.geode-rootfs \
    --exclude=./root --exclude=./usr/local .
echo "tar: $(du -h /workspace/geode-linux.tar | cut -f1)"
