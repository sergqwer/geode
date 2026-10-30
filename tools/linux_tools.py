"""The agent's Linux runner: proot and the libraries it needs, from Termux's package repository, made ready to ship in
the APK's native library directory (the only place an app targeting Android 10+ may exec files from).

    .venv/Scripts/python.exe tools/linux_tools.py [--out android/app/src/main/jniLibs/arm64-v8a]

Writes libproot.so (the proot executable), libproot-loader.so (its static loader: proot execs it, it maps the guest
program), libtalloc.so and libandroid-shmem.so. Each ELF's RUNPATH becomes $ORIGIN (its own directory: the system
linker finds the others there, no LD_LIBRARY_PATH leaking into the guest) and proot's NEEDED libtalloc.so.2 becomes
libtalloc.so (an APK packs lib*.so only). Both rewrites are in place: the new strings are no longer than the old.

proot is GPL-2.0 (its source: github.com/termux/proot), talloc LGPL-3.0, libandroid-shmem BSD-3-Clause.
"""
import argparse
import io
import lzma
import pathlib
import re
import struct
import tarfile
import urllib.request

REPO = "https://packages.termux.dev/apt/termux-main/pool/main/"
PACKAGES = {  # package -> {path in the package: name in the APK}
    "p/proot": {"bin/proot": "libproot.so", "libexec/proot/loader": "libproot-loader.so"},
    "libt/libtalloc": {"lib/libtalloc.so.2.5.0": "libtalloc.so"},
    "liba/libandroid-shmem": {"lib/libandroid-shmem.so": "libandroid-shmem.so"},
}
PREFIX = "data/data/com.termux/files/usr/"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "geode-tools"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def latest_deb(pkg: str) -> str:
    html = fetch(REPO + pkg + "/").decode()
    debs = re.findall(r'href="([^"]+_aarch64\.deb)"', html)
    if not debs:
        raise SystemExit(f"no aarch64 package in {pkg}")
    return debs[-1]


def deb_data(deb: bytes) -> tarfile.TarFile:
    """The data.tar.* member of a .deb (an ar archive)."""
    assert deb[:8] == b"!<arch>\n"
    at = 8
    while at < len(deb):
        name = deb[at:at + 16].decode().strip().rstrip("/")
        size = int(deb[at + 48:at + 58].decode().strip())
        body = deb[at + 60:at + 60 + size]
        if name.startswith("data.tar"):
            raw = lzma.decompress(body) if name.endswith(".xz") else body
            return tarfile.open(fileobj=io.BytesIO(raw))
        at += 60 + size + (size & 1)
    raise SystemExit("no data.tar in the package")


def patch_elf(b: bytearray, needed: dict) -> None:
    """RUNPATH -> $ORIGIN, NEEDED renamed per [needed]; strings rewritten in place (padded with NULs)."""
    if b[:4] != b"\x7fELF":
        raise SystemExit("not an ELF")
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, _ = struct.unpack_from("<HHH", b, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]
    dyn = next((s for s in secs if s[1] == 6), None)
    if dyn is None:
        return  # static (the loader)
    strtab = secs[dyn[6]][4]

    def rewrite(off: int, new: str) -> None:
        o = strtab + off
        old_len = b.index(b"\0", o) - o
        enc = new.encode()
        assert len(enc) <= old_len, (new, old_len)
        b[o:o + old_len] = enc + b"\0" * (old_len - len(enc))

    for i in range(dyn[5] // 16):
        tag, val = struct.unpack_from("<qQ", b, dyn[4] + i * 16)
        o = strtab + val
        name = bytes(b[o:b.index(b"\0", o)]).decode() if tag in (1, 29) else ""
        if tag == 29:
            rewrite(val, "$ORIGIN")
        elif tag == 1 and name in needed:
            rewrite(val, needed[name])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="android/app/src/main/jniLibs/arm64-v8a")
    out = pathlib.Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for pkg, files in PACKAGES.items():
        deb = latest_deb(pkg)
        data = deb_data(fetch(REPO + pkg + "/" + deb))
        members = {m.name.lstrip("./"): m for m in data.getmembers()}
        for src, dst in files.items():
            m = members[PREFIX + src]
            b = bytearray(data.extractfile(m).read())
            patch_elf(b, {"libtalloc.so.2": "libtalloc.so"})
            (out / dst).write_bytes(b)
            print(f"{deb}: {src} -> {out / dst} ({len(b)} bytes)")


if __name__ == "__main__":
    main()
