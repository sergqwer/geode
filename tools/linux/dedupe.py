"""Deduplicates a node_modules tree in place: every nested copy of the same package@version but one becomes a symbolic
link to it (npm keeps one copy per dependent when versions differ from the hoisted one: DeepSeek Harness's plugins
each carry their own zod 4.4.3 beside a hoisted 4.6.7, ~6.5 MB fifty times). Node resolves the links to the one real
directory, so the dependents share it - as a deduplicated install would have them.

    python3 dedupe.py /opt/node/lib/node_modules/@deepseek-ai/dsh/node_modules
"""
import json
import os
import shutil
import sys


def packages(root):
    """(path, name, version) of each package directory below a node_modules (scoped ones included)."""
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) != "node_modules":
            continue
        for d in list(dirnames):
            names = [os.path.join(d, s) for s in sorted(os.listdir(os.path.join(dirpath, d)))] if d.startswith("@") else [d]
            for n in names:
                p = os.path.join(dirpath, n)
                if os.path.islink(p) or not os.path.isdir(p):
                    continue
                try:
                    v = json.load(open(os.path.join(p, "package.json"), encoding="utf-8")).get("version")
                except (OSError, ValueError):
                    continue
                if v:
                    yield p, n.replace(os.sep, "/"), v


def size(p):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs
               if not os.path.islink(os.path.join(a, f)))


def main():
    # real paths throughout (/opt/node is itself a link): a path through a link made below is seen as one
    root = os.path.realpath(sys.argv[1])
    first = {}
    saved = 0
    links = 0
    # shallow paths first: the copy kept is the least nested one
    for p, name, v in sorted(packages(root), key=lambda t: t[0].count(os.sep)):
        key = (name, v)
        if key not in first:
            first[key] = p
            continue
        if os.path.realpath(p) != p:
            continue   # reached through a link made above: it is the kept copy's own
        keep = first[key]
        if p.startswith(keep + os.sep):
            continue   # inside the copy kept
        saved += size(p)
        shutil.rmtree(p)
        os.symlink(os.path.relpath(keep, os.path.dirname(p)), p)
        links += 1
    print(f"{links} copies linked, {saved / 2**20:.0f} MB freed")


if __name__ == "__main__":
    main()
