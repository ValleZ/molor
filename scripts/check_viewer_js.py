"""Catch unterminated strings and unbalanced brackets in a built viewer page.

There is no JS engine on this machine, so the page's script is checked by
scanning it. An earlier version of this check tracked only double-quoted
strings, and a bare apostrophe inside a single-quoted one -- `'model's training
set'` -- slipped past it and shipped. A parse error means the whole script never
runs, the window error handler never installs, and the page renders blank: the
worst failure mode available, and invisible to every check that reads the
document rather than executing it.

So this walks the script once as a lexer would, tracking single quotes, double
quotes, template literals, regex literals, line comments and block comments,
and reports an unterminated string with the line it began on. Bracket balance
is checked on the code outside strings and comments.

Run it on every built page before publishing:

    python scripts/check_viewer_js.py artifacts/molor_public_fp3.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BS = chr(92)


def scan(src: str):
    """Return (errors, bracket_counts) for one script body."""
    errors = []
    depth = {"(": 0, "[": 0, "{": 0}
    close = {")": "(", "]": "[", "}": "{"}
    i, n, line = 0, len(src), 1
    # Previous significant character decides whether a slash starts a regex or
    # is a division sign. Good enough for this page's style.
    prev = ""
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                errors.append((line, "unterminated block comment"))
                break
            line += src.count("\n", i, j)
            i = j + 2
            continue
        if c in "'\"`":
            start, q = line, c
            i += 1
            closed = False
            while i < n:
                ch = src[i]
                if ch == BS:
                    i += 2
                    continue
                if ch == "\n":
                    line += 1
                    if q != "`":
                        break          # neither quote type spans a line here
                    i += 1
                    continue
                if ch == q:
                    closed = True
                    i += 1
                    break
                i += 1
            if not closed:
                errors.append((start, f"unterminated {q}-quoted string"))
            prev = "str"
            continue
        if c == "/" and prev in ("", "op"):
            # regex literal
            i += 1
            while i < n and src[i] != "\n":
                if src[i] == BS:
                    i += 2
                    continue
                if src[i] == "/":
                    i += 1
                    break
                i += 1
            while i < n and src[i].isalpha():
                i += 1
            prev = "str"
            continue
        if c in depth:
            depth[c] += 1
            prev = "op"
            i += 1
            continue
        if c in close:
            depth[close[c]] -= 1
            if depth[close[c]] < 0:
                errors.append((line, f"unmatched {c}"))
            prev = "val"
            i += 1
            continue
        prev = "op" if c in "=+-*%<>!&|,;:?" else "val"
        i += 1
    for b, d in depth.items():
        if d:
            errors.append((0, f"{b} unbalanced by {d:+d}"))
    return errors, depth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+")
    args = ap.parse_args()

    bad = 0
    for p in args.pages:
        h = Path(p).read_text(encoding="utf-8")
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", h, re.S)
        # Skip the data blob: it is JSON in a script tag, not code.
        bodies = [s for s in scripts if "function" in s]
        print(f"{p}  ({len(scripts)} script blocks, {len(bodies)} with code)")
        if not bodies:
            print("   no code block found")
            bad += 1
            continue
        for s in bodies:
            errors, _ = scan(s)
            if errors:
                bad += 1
                for ln, msg in errors:
                    where = f"line {ln} of the script" if ln else "overall"
                    print(f"   FAIL  {where}: {msg}")
            else:
                print(f"   ok    {len(s)} chars, strings and brackets balanced")
        first = h.splitlines()[0][:40] if h.splitlines() else ""
        print(f"   charset first line: {first}")
        if "�" in h:
            print("   FAIL  replacement character present (encoding damage)")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
