"""Turn a filled-in strength sheet into a column the dose conversion can use.

The sheet records, per strip, whether a material smelled stronger or weaker
than the rest of the strips when every one of them was prepared at its own
recorded use level. A bin of "+" means the material delivered more smell than
its use level implies, so its strength estimate goes up and its dose in a
blend goes down.

Each bin is one step of a factor of three. That is coarse on purpose: the
compiled threshold figures span eight orders of magnitude and correlate with
the perfumer's own use levels at only +0.38, so a factor-of-three correction
measured by the nose that will smell the result is worth more than a precise
number from an unknown source. Two steps either way covers an order of
magnitude, which is the scale of the error being corrected.

The blind duplicates are the point of the exercise and are reported first. If
a material lands two bins apart from itself, this session cannot resolve one
bin, and the script says so rather than writing the numbers anyway.

    python scripts/apply_strength.py data/inventory/strength_session1.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: One bin is a factor of three in strength. "+" means it smelled stronger
#: than its use level implies, so the strength multiplier rises.
STEP = 3.0
BINS = {"--": -2, "-": -1, "0": 0, "+": 1, "++": 2,
        "-2": -2, "-1": -1, "1": 1, "2": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--out", default="data/inventory/strength.csv")
    ap.add_argument("--force", action="store_true",
                    help="write even when the blind duplicates disagree")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    path = Path(args.sheet)
    path = path if path.is_absolute() else ROOT / path
    rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
    # The sheet is nameless so it can sit on the bench; the names are in the
    # key file beside it.
    keypath = path.with_name(path.stem + "_key.csv")
    if not keypath.is_file():
        raise SystemExit(f"{keypath.name} is missing: it maps strip numbers "
                         f"to materials and is written beside the sheet.")
    names = {r["strip"]: r["material"]
             for r in csv.DictReader(io.open(keypath, encoding="utf-8-sig"))}
    for r in rows:
        r["material"] = names.get(str(r.get("strip", "")).strip(), "")
    seen = defaultdict(list)
    blank = 0
    for r in rows:
        b = (r.get("bin") or "").strip()
        if not b:
            blank += 1
            continue
        if b not in BINS:
            raise SystemExit(f"strip {r.get('strip')}: {b!r} is not one of "
                             f"-- - 0 + ++")
        seen[r["material"].strip()].append(BINS[b])

    print(f"[strength] {len(rows)} strips, {blank} not filled in, "
          f"{len(seen)} materials judged")
    if not seen:
        raise SystemExit("nothing filled in yet")

    dups = {n: v for n, v in seen.items() if len(v) > 1}
    worst = 0
    if dups:
        print("\n[strength] the blind duplicates, which are this session's "
              "error bar:")
        for n, v in sorted(dups.items()):
            spread = max(v) - min(v)
            worst = max(worst, spread)
            verdict = ("same bin twice" if spread == 0
                       else "one bin apart" if spread == 1
                       else f"{spread} bins apart")
            print(f"   {n:30s} {v}  {verdict}")
        if worst == 0:
            print("   -> a one-bin difference between materials is meaningful.")
        elif worst == 1:
            print("   -> one bin is within your own noise; read two-bin "
                  "differences only.")
        else:
            print(f"   -> {worst} bins of self-disagreement. This session "
                  f"cannot resolve the thing it is measuring.")
    else:
        print("\n[strength] no duplicates filled in, so this session has no "
              "error bar. The numbers below are unguarded.")

    if worst >= 2 and not args.force:
        raise SystemExit("\nNot writing strength.csv: the duplicates "
                         "disagree by two bins or more, which is as large as "
                         "the effect being measured. Smell them again on a "
                         "different day, or pass --force to write anyway.")

    out_rows = []
    for n, v in sorted(seen.items()):
        step = sum(v) / len(v)
        out_rows.append({"name": n,
                         "strength_multiplier": round(STEP ** step, 4),
                         "bin_mean": round(step, 3),
                         "judgements": len(v),
                         "self_spread": (max(v) - min(v)) if len(v) > 1 else ""})

    out = ROOT / args.out
    write_header = not out.is_file()
    existing = {}
    if out.is_file():
        for r in csv.DictReader(io.open(out, encoding="utf-8-sig")):
            existing[r["name"]] = r
    for r in out_rows:
        existing[r["name"]] = r
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        for n in sorted(existing):
            w.writerow(existing[n])

    print(f"\n[strength] {'wrote' if write_header else 'updated'} {args.out}")
    print(f"  {'material':30s} {'bin':>5s}  {'multiplier':>10s}")
    for r in out_rows:
        arrow = ("dose less" if r["bin_mean"] > 0.01
                 else "dose more" if r["bin_mean"] < -0.01 else "unchanged")
        print(f"  {r['name'][:30]:30s} {r['bin_mean']:5.1f}  "
              f"{r['strength_multiplier']:10.2f}   {arrow}")
    print("\n[strength] re-run scripts/export_simplify_data.py to put these "
          "in the page, where they take precedence over the compiled "
          "threshold figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
