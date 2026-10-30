"""Puts a packed agent image into the app: python tools/linux/publish.py <geode-linux.tar | geode-linux.tar.xz>

A plain tar (pack.sh's: the base system has no xz) is compressed here first (xz, preset 9e, CRC64 - what the app's
XZInputStream reads). The archive goes into the install-time asset pack (android/linux_pack, gitignored) with its
version beside it - the first 16 hex digits of its SHA-256. A phone whose Linux carries another version unpacks this
one again when the agent next starts (Agent.outdated); the chats' own folders stay as they are.
"""
import hashlib
import lzma
import pathlib
import shutil
import sys

ASSETS = pathlib.Path(__file__).resolve().parents[2] / "android" / "linux_pack" / "src" / "main" / "assets"


def main() -> None:
    src = pathlib.Path(sys.argv[1])
    ASSETS.mkdir(parents=True, exist_ok=True)
    dst = ASSETS / "geode-linux.tar.xz"
    if src.suffix == ".tar":
        with src.open("rb") as i, lzma.open(dst, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                                            preset=9 | lzma.PRESET_EXTREME) as o:
            shutil.copyfileobj(i, o, 1 << 20)
    elif src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    h = hashlib.sha256()
    with dst.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    version = h.hexdigest()[:16]
    (ASSETS / "geode-linux.version").write_text(version + "\n", encoding="ascii")
    print(f"{dst} ({dst.stat().st_size / 1e6:.1f} MB), version {version}")


if __name__ == "__main__":
    main()
