"""Turn filled-in strength sheets into a column the dose conversion can use.

A sheet records, per blotter, whether a material smelled stronger or weaker
than the rest of the blotters when every one of them was prepared at its own
recorded use level. A bin of "+" means the material delivered more smell than
its use level implies, so its strength estimate goes up and its dose in a
blend goes down. Each bin is one step of a factor of three.

Pass as many sheets as there are sessions:

    python scripts/apply_strength.py data/inventory/strength_round1.csv \
        data/inventory/strength_round2.csv data/inventory/strength_retest.csv

Two things in the first version of this script were wrong, and both mattered.

*It compared sessions by blotter number.* Blotter 3 is Isobornyl Acetate on
the first sheet and Hedione on the re-test sheet, so the "same blotter judged
again" table was comparing different materials and reporting invented
disagreement. Sheets are now matched on the material name as well as the
number, and only blotters holding the same material in both are compared.

*It vetoed a material whose judgements spanned two bins.* The span of a set of
numbers grows with how many of them there are, so that rule punished a
material for being measured more often. Four materials were re-tested, came
back repeating themselves exactly within the new session -- Hedione twice at
--, Ethylene Brassylate twice at --, Anisaldehyde twice at 0 -- and the extra
evidence made all four look *worse* by that rule. A statistic that degrades as
evidence accumulates is the wrong statistic.

So nothing is vetoed now. Instead the duplicate pairs give a pooled estimate
of how noisy a single judgement is, that gives each material a standard error
from its own count of judgements, and each estimate is shrunk toward "no
correction" in proportion to its noise. A material judged twice with a
consistent answer keeps most of its correction; one judged twice with a
contradictory answer is pulled most of the way back to its compiled figure,
which is the honest thing to do with a measurement that did not converge. No
afternoon of smelling is thrown away, and no single noisy pair moves a formula
on its own.
"""

from __future__ import annotations

import argparse
import csv
import io
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
#: Per-judgement noise, in bins, assumed when no sheet has a duplicate pair to
#: estimate it from. One bin: a session without duplicates has shown nothing
#: about its own repeatability, so its estimates shrink hard.
DEFAULT_NOISE = 1.0


def read_sheet(path: Path):
    """{blotter -> (material, bin)} for the rows that were filled in."""
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
    return per_blotter, len(rows), blank


def judgement_noise(sessions):
    """Pooled sd of one judgement, from within-session duplicate pairs.

    A duplicate pair is the same material prepared twice in one session, so
    its difference is close to pure judging noise: the material, the day and
    the nose are held fixed. The sd of a difference of two independent
    judgements is sqrt(2) times the sd of one, hence the division.
    """
    diffs = []
    for _, per_blotter in sessions:
        per_mat = defaultdict(list)
        for mat, v in per_blotter.values():
            per_mat[mat].append(v)
        for vs in per_mat.values():
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    diffs.append(vs[i] - vs[j])
    if not diffs:
        return DEFAULT_NOISE, 0
    d = np.asarray(diffs, float)
    return float(np.sqrt((d ** 2).mean() / 2.0)) or DEFAULT_NOISE, len(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", nargs="+")
    ap.add_argument("--out", default="data/inventory/strength.csv")
    ap.add_argument("--no-shrink", action="store_true",
                    help="write the raw means, unshrunk; for inspection only")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    sessions = []
    for sp in args.sheet:
        path = Path(sp)
        path = path if path.is_absolute() else ROOT / path
        per_blotter, n_rows, blank = read_sheet(path)
        sessions.append((path.name, per_blotter))
        print(f"[strength] {path.name}: {n_rows} strips, "
              f"{blank} not filled in")

    # The sheets carry the material names, because preparing a blotter
    # requires knowing what goes on it: the separate key file an earlier
    # version wrote was unusable theatre.
    seen = defaultdict(list)
    for _, per_blotter in sessions:
        for mat, v in per_blotter.values():
            seen[mat].append(v)
    if not seen:
        raise SystemExit("nothing filled in yet")
    print(f"[strength] {len(sessions)} session(s), {len(seen)} materials, "
          f"{sum(len(v) for v in seen.values())} judgements")

    print("\n[strength] within a session, the same material on two blotters:")
    any_dup = False
    for name, per_blotter in sessions:
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
            where = ", ".join(f"#{b}:{v:+d}" for b, v in bv)
            print(f"   {name:26s} {n:22s} {where:24s} "
                  f"{'same bin' if not spread else str(spread) + ' apart'}")
    if not any_dup:
        print("   none filled in.")

    # The same blotter judged again in a later session: identical strip,
    # identical dilution, only the nose and the day changed. Matched on the
    # material as well as the number, because a re-test sheet renumbers.
    if len(sessions) > 1:
        base_name, base = sessions[0]
        print("\n[strength] the same blotter judged again in a later session:")
        for name, other in sessions[1:]:
            shared = set(base) & set(other)
            common = sorted(b for b in shared if base[b][0] == other[b][0])
            skipped = len(shared) - len(common)
            if not common:
                print(f"   {base_name} vs {name}: no blotter number holds the "
                      f"same material in both, so there is nothing to "
                      f"compare ({skipped} skipped).")
                continue
            diffs = [abs(base[b][1] - other[b][1]) for b in common]
            exact = sum(1 for d in diffs if d == 0)
            within1 = sum(1 for d in diffs if d <= 1)
            a = np.array([base[b][1] for b in common], float)
            c = np.array([other[b][1] for b in common], float)
            r = (float(np.corrcoef(a, c)[0, 1])
                 if a.std() and c.std() else float("nan"))
            print(f"   {base_name} vs {name}: {len(common)} comparable "
                  f"blotters, {exact} identical ({exact / len(common):.0%}), "
                  f"{within1} within one bin ({within1 / len(common):.0%})")
            print(f"   mean disagreement {np.mean(diffs):.2f} bins, "
                  f"correlation r = {r:+.2f}")
            if skipped:
                print(f"   ({skipped} blotter numbers hold different "
                      f"materials in the two sheets and were skipped)")

    noise, n_pairs = judgement_noise(sessions)
    if n_pairs:
        print(f"\n[strength] {n_pairs} duplicate pairs give a per-judgement "
              f"sd of {noise:.2f} bins: one judgement on its own is worth "
              f"about a factor of {STEP ** noise:.1f}.")
    else:
        print(f"\n[strength] no duplicate pairs, so the noise is assumed to "
              f"be {noise:.2f} bins and every estimate shrinks hard.")

    names = sorted(seen)
    raw = np.array([np.mean(seen[n]) for n in names], float)
    cnt = np.array([len(seen[n]) for n in names], float)
    se = noise / np.sqrt(cnt)

    # Empirical Bayes. The spread of the raw means holds both real differences
    # between materials and judging noise; subtracting the noise leaves the
    # real part, and each material keeps the fraction of its correction that
    # its own evidence supports.
    var_obs = float(raw.var(ddof=1)) if len(raw) > 1 else 0.0
    var_true = max(var_obs - float((se ** 2).mean()), 1e-6)
    weight = var_true / (var_true + se ** 2)
    if args.no_shrink:
        weight = np.ones_like(weight)
    est = weight * raw
    if var_obs > 0 and not args.no_shrink:
        print(f"[strength] of the spread between materials, "
              f"{min(var_true / var_obs, 1.0):.0%} is real and the rest is "
              f"noise, so corrections keep {weight.min():.0%} to "
              f"{weight.max():.0%} of their raw size.")

    out_rows = []
    for i, n in enumerate(names):
        v = seen[n]
        out_rows.append({
            "name": n,
            "strength_multiplier": round(STEP ** est[i], 4),
            "bin_mean": round(float(est[i]), 3),
            "bin_raw": round(float(raw[i]), 3),
            "raw_multiplier": round(STEP ** raw[i], 4),
            "standard_error_bins": round(float(se[i]), 3),
            "shrinkage": round(float(weight[i]), 3),
            "judgements": len(v),
            "self_spread": (max(v) - min(v)) if len(v) > 1 else "",
        })

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
    print(f"  {'material':24s} {'n':>2s} {'raw':>5s} {'se':>5s} "
          f"{'used':>5s} {'x strength':>10s}")
    print("  " + "-" * 64)
    for r in sorted(out_rows, key=lambda r: r["bin_mean"]):
        arrow = ("dose less" if r["bin_mean"] > 0.05
                 else "dose more" if r["bin_mean"] < -0.05 else "unchanged")
        print(f"  {r['name'][:24]:24s} {r['judgements']:2d} "
              f"{r['bin_raw']:+5.2f} {r['standard_error_bins']:5.2f} "
              f"{r['bin_mean']:+5.2f} {r['strength_multiplier']:10.2f}   "
              f"{arrow}")
    print("\n[strength] a multiplier above 1 means the material delivers more "
          "smell per gram than its use level implies, so the dose conversion "
          "divides its threshold by this figure.")
    print("[strength] re-run scripts/export_simplify_data.py to put these in "
          "the page, where they take precedence over the compiled threshold "
          "figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
