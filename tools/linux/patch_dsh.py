"""Geode's fixes to the installed DeepSeek Harness (python3 patch_dsh.py <dsh package dir>), run by build.sh after npm.

Each patch names the file (under node_modules, wherever npm put the package) and an anchor that must occur exactly
once: a dsh update that moves the code fails the build here instead of shipping an unpatched agent.
"""
import pathlib
import sys

PATCHES = [
    # A new file is published with link(temp, target) - a create that never replaces - and the staging folder removed.
    # Android does not let an app make hard links; proot fakes one by moving the data to .l2s.* in the staging folder
    # and leaving symbolic links to it, so removing the folder left the new file a dangling link. No-replace by lstat,
    # then rename: the agent is alone in its folder.
    ("@deepseek-ai/dsh-fs-local/lib/index.js",
     "\t\t\tawait linkFile(tempPath, absolutePath);\n",
     "\t\t\tawait lstat(absolutePath).then(() => {\n"
     "\t\t\t\tthrow Object.assign(new Error(`EEXIST: file already exists, '${absolutePath}'`), { code: \"EEXIST\" });\n"
     "\t\t\t}, (error) => {\n"
     "\t\t\t\tif (error?.code !== \"ENOENT\") throw error;\n"
     "\t\t\t});\n"
     "\t\t\tawait rename(tempPath, absolutePath);\n"),
]


def find(root: pathlib.Path, rel: str) -> list[pathlib.Path]:
    """Every copy of the package file under root (npm may nest one package in another's node_modules)."""
    return [p for p in root.rglob(pathlib.PurePosixPath(rel).name) if p.as_posix().endswith("node_modules/" + rel)]


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    for rel, old, new in PATCHES:
        files = find(root, rel)
        if not files:
            sys.exit(f"patch_dsh: {rel} not found under {root}")
        for f in files:
            s = f.read_text(encoding="utf-8")
            if new in s:
                print(f"patch_dsh: {f} already patched")
                continue
            if s.count(old) != 1:
                sys.exit(f"patch_dsh: anchor found {s.count(old)} times in {f}")
            f.write_text(s.replace(old, new), encoding="utf-8")
            print(f"patch_dsh: patched {f}")


if __name__ == "__main__":
    main()
