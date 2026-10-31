"""How many materials does it take to reach the mud? And a pair to prove it.

Random blends collapse towards the centre of percept space as ingredients are
added: a single material sits a median 1.22 from the library's average point, a
three-material blend 0.81, ten 0.47, thirty 0.20. The published olfactory-white
result says that at around thirty components, spanning odour space at equal
perceived intensity, mixtures stop smelling of anything in particular and start
smelling like each other. This asks at what n *this model* says the same, and
then writes two blends to make so the claim can be smelled rather than argued.

The measurement needs a yardstick, because "converged" is meaningless without a
scale, and cosine is the wrong one. Two independent thirty-material blends have
a centred cosine near zero even though they are nearly the same point -- cosine
normalises away exactly the shrinkage that is the phenomenon. Absolute distance
is the right measure, and the scale it is read against is the distance between
a material and its nearest neighbour in the library: pairs at that distance are
the ones a perfumer finds hard to tell apart (the musk chemotypes sit there).
So the criterion is that two independent blends be closer to each other than a
typical material is to its nearest neighbour.

Two protocols are compared, because which materials go in matters as much as
how many. Random draws are what a careless mixture looks like. Spanning draws
-- farthest-point sampling, so the ingredients are spread across the space --
are what the published protocol asked for, and they should converge faster.

What this cannot capture: equal weights stand in for equal perceived intensity,
which is what the real experiment controlled for and what this model has no way
to compute. A blend of equal masses is not a blend of equal intensities, and
the gap is as large as the threshold spread -- six orders of magnitude. So the
n reported here is a model prediction about *proportions*, and the bench will
disagree to the extent that intensity matters, which is the honest reason to
run it.

    python scripts/olfactory_white.py --owned --emit
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixtures import combine  # noqa: E402


def use_levels() -> dict[str, float]:
    path = ROOT / "data" / "inventory" / "inventory.csv"
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or r.get("material") or "").strip()
        if not n:
            continue
        try:
            out[n] = float((r.get("top_pct") or "").strip())
        except ValueError:
            out[n] = float("nan")
    return out


def farthest_point(X: np.ndarray, pool: list[int], n: int, rng) -> list[int]:
    """Spread n picks across the space, starting from a random seed."""
    chosen = [int(rng.choice(pool))]
    d = np.linalg.norm(X[pool] - X[chosen[0]], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        chosen.append(pool[k])
        d = np.minimum(d, np.linalg.norm(X[pool] - X[pool[k]], axis=1))
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--rule", default="weighted-mean",
                    choices=["weighted-mean", "logit-mean", "noisy-or"])
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--owned", action="store_true",
                    help="only materials on the shelf, so the pair is makeable")
    ap.add_argument("--min-use", type=float, default=0.5,
                    help="skip materials the perfumer never takes this high, "
                         "in percent; at n=25 each ingredient is 4% of the "
                         "blend and something capped at 0.1% cannot go there")
    ap.add_argument("--emit", action="store_true",
                    help="write the blends to make")
    ap.add_argument("--out", default="data/formulas/queries/white")
    ap.add_argument("--emit-n", type=int, default=None,
                    help="materials per demonstration blend; the default is "
                         "the first size whose spanning gap is under a third "
                         "of the yardstick")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    names = list(pj["molecules"])
    X = np.asarray(pj["probs"], float)
    mu = X.mean(0)

    pool = list(range(len(names)))
    lv = use_levels()
    if args.owned:
        pool = [i for i, n in enumerate(names)
                if n in lv and not math.isnan(lv[n]) and lv[n] >= args.min_use]
    print(f"[white] {len(pool)} materials available, rule {args.rule}")

    # The yardstick: how far apart are the pairs a nose finds hard to separate?
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    yard = float(np.median(nn))
    print(f"[white] a material's nearest neighbour sits {yard:.3f} away "
          f"(median); 10th percentile {np.percentile(nn, 10):.3f}")
    print(f"[white] a material sits {np.median(np.linalg.norm(X - mu, axis=1)):.3f} "
          f"from the library's average point\n")

    rng = np.random.default_rng(args.seed)
    print(f"  {'n':>4s} {'random: gap between two':>24s} {'<yard':>7s} "
          f"{'spanning: gap':>15s} {'<yard':>7s} {'to centre':>10s}")
    print("  " + "-" * 74)
    curve, crossed = [], {}
    for n in (2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50):
        if n > len(pool):
            break
        w = np.full(n, 1.0 / n)
        gaps, gaps_s, cen = [], [], []
        for _ in range(args.pairs):
            a = combine(args.rule, X[rng.choice(pool, n, replace=False)], w)
            b = combine(args.rule, X[rng.choice(pool, n, replace=False)], w)
            gaps.append(float(np.linalg.norm(a - b)))
            cen.append(float(np.linalg.norm(a - mu)))
        for _ in range(max(args.pairs // 10, 20)):
            ia = farthest_point(X, pool, n, rng)
            ib = farthest_point(X, pool, n, rng)
            gaps_s.append(float(np.linalg.norm(
                combine(args.rule, X[ia], w) - combine(args.rule, X[ib], w))))
        g, gs = np.array(gaps), np.array(gaps_s)
        row = {"n": n, "gap": float(g.mean()), "under": float((g < yard).mean()),
               "gap_span": float(gs.mean()),
               "under_span": float((gs < yard).mean()),
               "to_centre": float(np.mean(cen))}
        curve.append(row)
        for key, frac in (("random", row["under"]), ("spanning", row["under_span"])):
            if frac >= 0.5 and key not in crossed:
                crossed[key] = n
        print(f"  {n:4d} {g.mean():24.3f} {row['under']:6.0%} "
              f"{gs.mean():15.3f} {row['under_span']:6.0%} "
              f"{row['to_centre']:10.3f}")

    print("")
    for key in ("random", "spanning"):
        if key in crossed:
            print(f"[white] {key} blends: half of all independent pairs are "
                  f"within one nearest-neighbour distance of each other at "
                  f"n = {crossed[key]}")
        else:
            print(f"[white] {key} blends never got half the pairs under the "
                  f"yardstick in this range")

    if args.emit:
        # The crossing size answers the question; it makes a poor demonstration.
        # At n = 8 the predicted gap is only just under the yardstick and under
        # twice the control's, which a nose could easily call a draw. The pair
        # to make is therefore taken well past the crossing, at the smallest n
        # whose spanning gap is under a third of the yardstick, so that a
        # failure on the bench means something.
        n = args.emit_n or next(
            (r["n"] for r in curve if r["gap_span"] < yard / 3), 25)
        outdir = ROOT / args.out
        outdir.mkdir(parents=True, exist_ok=True)
        # The demonstration pair: two spanning blends at the crossing size,
        # chosen from the many draws as the pair with the fewest shared
        # ingredients, because the point is two different recipes that smell
        # the same. A small-n control goes with them: if the control pair
        # smells the same too, the experiment has shown nothing about n.
        # Disjoint by construction: the pool is split in half at random and a
        # spanning set is drawn from each half, so the two blends cannot share
        # an ingredient. Two spanning draws from the whole pool would overlap
        # heavily -- farthest-point sampling keeps choosing the same extremes --
        # and "the same smell from the same materials" demonstrates nothing.
        if 2 * n > len(pool):
            n = len(pool) // 2
            print(f"[white] pool allows at most {n} per blend if the two are "
                  f"to share nothing; using that")
        # Weights proportional to each material's own maximum use level, not
        # equal. This started as a bug fix -- an equal-weight blend of twenty
        # puts every ingredient at 5%, and the filter was admitting materials
        # the perfumer caps at 2%, so the sheet asked for five percent of
        # blackcurrant mercaptan, which would have annihilated the experiment
        # and told us nothing about mixtures. Weighting by use level fixes it
        # for free: a share of 100 * level / sum(levels) is at or below every
        # material's own ceiling whenever the levels sum past 100, which they
        # do well before twenty ingredients.
        #
        # It is also the better protocol. The published experiment controlled
        # for equal perceived *intensity*, which this model cannot compute; a
        # perfumer's maximum use level is an inverse strength estimate made by
        # a nose, and is the closest thing available. So the emitted pair is
        # nearer the real experiment than an equal-weight blend would be, and
        # the predicted gap below is computed at the weights on the sheet.
        def level_weights(idx):
            v = np.array([lv.get(names[i], float("nan")) for i in idx], float)
            v = np.where(np.isfinite(v) & (v > 0), v, np.nanmedian(v))
            return v / v.sum()

        best = None
        for _ in range(400):
            perm = rng.permutation(pool)
            half_a = [int(i) for i in perm[: len(perm) // 2]]
            half_b = [int(i) for i in perm[len(perm) // 2:]]
            ia = farthest_point(X, half_a, n, rng)
            ib = farthest_point(X, half_b, n, rng)
            gap = float(np.linalg.norm(
                combine(args.rule, X[ia], level_weights(ia))
                - combine(args.rule, X[ib], level_weights(ib))))
            if best is None or gap < best[0]:
                best = (gap, ia, ib)
        gap, ia, ib = best
        share = len(set(ia) & set(ib))
        print(f"\n[white] the pair to make: {n} materials each, "
              f"{share} ingredient(s) in common, predicted gap {gap:.3f} "
              f"against a yardstick of {yard:.3f}")
        jobs = [(f"white_{n}_a", ia), (f"white_{n}_b", ib)]
        # The control has three ingredients, so each is about a third of the
        # blend and only materials the perfumer takes that high can be used --
        # otherwise the control is the one sheet in the set that cannot be
        # made, which is the opposite of a control.
        ctrl = 3
        big = [i for i in pool
               if lv.get(names[i], 0) == lv.get(names[i], 0)
               and lv.get(names[i], 0) >= 100 / ctrl]
        if len(big) < 2 * ctrl:
            big = sorted(pool, key=lambda i: -(lv.get(names[i], 0) or 0))[: 2 * ctrl]
            print(f"[white] only {len(big)} materials can sit at "
                  f"{100 / ctrl:.0f}%; the control uses the highest available")
        pick = rng.permutation(big)
        jobs += [(f"control_{ctrl}_a", [int(i) for i in pick[:ctrl]]),
                 (f"control_{ctrl}_b", [int(i) for i in pick[ctrl:2 * ctrl]])]
        over = []
        for tag, idx in jobs:
            path = outdir / f"{tag}.csv"
            wts = level_weights(idx)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                cw = csv.writer(fh)
                cw.writerow(["material", "dilution", "weight"])
                for i, wi in zip(idx, wts):
                    cw.writerow([names[i], 100, round(100 * float(wi), 2)])
                    cap = lv.get(names[i], float("nan"))
                    # Tolerance, because use-level-proportional weighting puts
                    # each material exactly at its own ceiling by construction
                    # and the written share is rounded to two places; without
                    # it every ingredient is reported as an overshoot of 0.05%.
                    if cap == cap and 100 * wi > cap * 1.02 + 0.02:
                        over.append((names[i], 100 * float(wi), cap))
            print(f"    {path.relative_to(ROOT)}: "
                  + ", ".join(names[i] for i in idx[:6])
                  + (" ..." if len(idx) > 6 else ""))
        if over:
            print("    [!] above the perfumer's own use level: "
                  + ", ".join(f"{n_} at {s:.1f}% vs {c:g}%"
                              for n_, s, c in over))
        cg = float(np.linalg.norm(
            combine(args.rule, X[jobs[2][1]], level_weights(jobs[2][1]))
            - combine(args.rule, X[jobs[3][1]], level_weights(jobs[3][1]))))
        print(f"    the control pair's predicted gap is {cg:.3f}, "
              f"{cg / max(gap, 1e-9):.1f}x the big pair's -- they should be "
              f"obviously different")
        (outdir / "prediction.json").write_text(json.dumps(
            {"rule": args.rule, "n": n, "yardstick": yard,
             "pair_gap": gap, "shared_ingredients": share,
             "control_n": ctrl, "control_gap": cg,
             "claim": "the two white blends should be hard to tell apart; "
                      "the two control blends should be obviously different",
             "caveat": "equal weights stand in for equal perceived intensity, "
                       "which this model cannot compute",
             "blends": {tag: [names[i] for i in idx] for tag, idx in jobs}},
            indent=2), encoding="utf-8")

    (ROOT / "artifacts" / "olfactory_white.json").write_text(json.dumps(
        {"rule": args.rule, "yardstick": yard, "owned_only": bool(args.owned),
         "curve": curve, "crossing": crossed}, indent=2), encoding="utf-8")
    print(f"\n[white] wrote artifacts/olfactory_white.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
