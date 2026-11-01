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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: One bin is a factor of three in strength. "+" means it smelled stronger
#: than its use level implies, so the strength multiplier rises.
STEP = 3.0
BINS = {"--": -2, "-": -1, "0": 0, "+": 1, "++": 2,
        "-2": -2, "-1": -1, "1": 1, "2": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", nargs="+")
    ap.add_argument("--out", default="data/inventory/strength.csv")
    ap.add_argument("--force", action="store_true",
                    help="write even when the blind duplicates disagree")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    # One sheet per session. Pooling several is the point: a single pass gives
    # the duplicates as its only error bar, while two passes also give the
    # same blotter judged twice, which is the stricter read on repeatability
    # because the preparation is held fixed and only the day changed.
    sessions = []
    for sp in args.sheet:
        path = Path(sp)
        path = path if path.is_absolute() else ROOT / path
        rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
        per_blotter, blank = {}, 0
        for r in rows:
            b = (r.get("bin") or "").strip()
            if not b:
                blank += 1
                continue
            if b not in BINS:
                raise SystemExit(f"{path.name} blotter {r.get('blotter')}: "
                                 f"{b!r} is not one of -- - 0 + ++")
            per_blotter[int(r["blotter"])] = (r["material"].strip(), BINS[b])
        sessions.append((path.name, len(rows), blank, per_blotter))

    # The sheets carry the material names, because preparing a blotter
    # requires knowing what goes on it: the separate key file that an earlier
    # version wrote was unusable theatre.
    seen = defaultdict(list)
    for _, _, _, per_blotter in sessions:
        for mat, v in per_blotter.values():
            seen[mat].append(v)

    for name, n_rows, blank, _ in sessions:
        print(f"[strength] {name}: {n_rows} strips, {blank} not filled in")
    print(f"[strength] {len(sessions)} session(s), {len(seen)} materials, "
          f"{sum(len(v) for v in seen.values())} judgements")
    if not seen:
        raise SystemExit("nothing filled in yet")

    # Error bar 1: the same material prepared twice within one session.
    worst = 0
    print("\n[strength] within a session, the same material on two blotters:")
    any_dup = False
    for name, _, _, per_blotter in sessions:
        per_mat = defaultdict(list)
        for b in sorted(per_blotter):
            mat, v = per_blotter[b]
            per_mat[mat].append((b, v))
        for n, bv in sorted(per_mat.items()):
            if len(bv) < 2:
                continue
            any_dup = True
            vals = [v for _, v in bv]
            spread = max(vals) - min(vals)
            worst = max(worst, spread)
            where = ", ".join(f"#{b}:{v:+d}" for b, v in bv)
            print(f"   {name:26s} {n:22s} {where:24s} "
                  f"{'same bin' if not spread else str(spread) + ' apart'}")
    if not any_dup:
        print("   none filled in.")

    # Error bar 2: the same blotter judged again in another session. Identical
    # strip, identical dilution; only the nose and the day changed.
    if len(sessions) > 1:
        base = sessions[0][3]
        print("\n[strength] between sessions, the same blotter judged again:")
        for name, _, _, other in sessions[1:]:
            common = sorted(set(base) & set(other))
            if not common:
                continue
            diffs = [abs(base[b][1] - other[b][1]) for b in common]
            exact = sum(1 for d in diffs if d == 0)
            within1 = sum(1 for d in diffs if d <= 1)
            a = np.array([base[b][1] for b in common], float)
            c = np.array([other[b][1] for b in common], float)
            r = (float(np.corrcoef(a, c)[0, 1])
                 if a.std() and c.std() else float("nan"))
            print(f"   {sessions[0][0]} vs {name}: {len(common)} blotters, "
                  f"{exact} identical ({exact / len(common):.0%}), "
                  f"{within1} within one bin ({within1 / len(common):.0%})")
            print(f"   mean disagreement {np.mean(diffs):.2f} bins, "
                  f"correlation r = {r:+.2f}")
            worst = max(worst, int(max(diffs)))
            for b in common:
                if abs(base[b][1] - other[b][1]) >= 2:
                    print(f"      blotter {b:2d} {base[b][0]:22s} "
                          f"{base[b][1]:+d} then {other[b][1]:+d}")

    if worst == 0:
        print("\n   -> a one-bin difference between materials is meaningful.")
    elif worst == 1:
        print("\n   -> one bin is within your own noise; read two-bin "
              "differences only.")
    else:
        print(f"\n   -> up to {worst} bins of self-disagreement somewhere. "
              f"Each material below carries its own spread, and only the "
              f"ones that held still are worth using.")

    # Refusing the whole session because one material misbehaved throws away
    # the materials that did hold still, which is most of them. So the
    # unresolved ones are held back by name and the rest are written.
    unresolved = sorted(n for n, v in seen.items()
                        if len(v) > 1 and max(v) - min(v) >= 2)
    if unresolved and not args.force:
        print(f"\n[strength] holding back {len(unresolved)} material(s) whose "
              f"own judgements span two bins or more, which is as large as the "
              f"effect being measured:")
        for n in unresolved:
            print(f"   {n:24s} {seen[n]}")
        print("   Re-smell these; the rest are written below. --force writes "
              "them anyway, averaged.")
        seen = {n: v for n, v in seen.items() if n not in unresolved}
        if not seen:
            raise SystemExit("nothing left to write")

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
        sp = r["self_spread"]
        flag = "" if sp == "" or sp <= 1 else f"   (spread {sp}, unresolved)"
        print(f"  {r['name'][:30]:30s} {r['bin_mean']:5.1f}  "
              f"{r['strength_multiplier']:10.2f}   {arrow}{flag}")
    print("\n[strength] re-run scripts/export_simplify_data.py to put these "
          "in the page, where they take precedence over the compiled "
          "threshold figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
