"""Given a target smell, propose materials for it -- and check the proposal.

This is the direction that has already failed once. The earlier accord work
built a target in receptor space, searched for a recipe with greedy selection
and invented weights, and the suggestions did not survive a blind test. Two
things are different here, and neither is a guarantee.

*The target is a descriptor profile, not a receptor profile.* Receptor space
correlates with perception only weakly, so a receptor-space target was asking
the search to match the wrong thing.

*The match is scored against a matched null.* The forward direction showed that
raw descriptor values are dominated by GS-LF label frequency: a random
ten-material blend already scores floral 0.901 and sweet 0.945, so a search
that maximises raw cosine to a target containing common labels will happily
return ten arbitrary materials. Here a candidate blend is standardised against
random blends of the same size and weights, and the objective is agreement
between that lift profile and the target. What the search optimises is
therefore "unusually melon for a blend of this size", not "melon".

The combination rule is a fraction-weighted mean. On the one formula where both
were checked it ranked descriptors identically to noisy-OR (Spearman +1.000),
and it does not saturate as ingredients are added, which matters when the
search is comparing blends of different sizes.

What this cannot do, and what no amount of care here will fix: the model has no
potency, volatility or threshold axis. Weights are abstract proportions. A
material the search wants at 0.2 may belong at 0.02% on the bench, and the
materials that matter most in a real accord are often the ones used at trace
level precisely because they are enormously strong -- which is the one thing
this objective cannot see.

``--recover`` runs the search against a known formula's own descriptor target
and reports how many of its real ingredients come back, which is the only
validation available without smelling anything.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

NULL_DRAWS = 600


def parse_target(spec: str, desc: list[str]) -> np.ndarray:
    """"melon=1,cucumber=0.6" -> a unit vector over the descriptor space."""
    t = np.zeros(len(desc))
    idx = {d: i for i, d in enumerate(desc)}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip().lower()
        if k not in idx:
            raise SystemExit(f"'{k}' is not one of the {len(desc)} descriptors")
        t[idx[k]] = float(v) if v.strip() else 1.0
    if not t.any():
        raise SystemExit("empty target")
    return t / np.linalg.norm(t)


def blend(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w[:, None] * P).sum(axis=0) / w.sum()


def null_stats(LM: np.ndarray, w: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of a random blend of len(w) materials at these weights."""
    n = len(w)
    draws = np.stack([blend(LM[rng.choice(len(LM), n, replace=False)], w)
                      for _ in range(NULL_DRAWS)])
    return draws.mean(0), np.maximum(draws.std(0), 1e-9)


def score(P_sel: np.ndarray, w: np.ndarray, target: np.ndarray,
          mu: np.ndarray, sd: np.ndarray) -> float:
    z = (blend(P_sel, w) - mu) / sd
    n = np.linalg.norm(z)
    return float(z @ target / n) if n > 1e-12 else 0.0


def search(LM: np.ndarray, names: list[str], target: np.ndarray, *,
           size: int, pool: list[int], rng, refine: bool = True):
    chosen: list[int] = []
    trace = []
    for step in range(size):
        w = np.ones(len(chosen) + 1)
        mu, sd = null_stats(LM, w, rng)
        best = (-2.0, None)
        for c in pool:
            if c in chosen:
                continue
            s = score(LM[chosen + [c]], w, target, mu, sd)
            if s > best[0]:
                best = (s, c)
        chosen.append(best[1])
        trace.append({"step": step + 1, "added": names[best[1]],
                      "score": round(best[0], 4)})

    w = np.ones(len(chosen))
    mu, sd = null_stats(LM, w, rng)
    if refine:
        # Coordinate descent on a log-spaced grid. Weights are proportions, not
        # doses; the grid is coarse because the objective does not deserve more.
        grid = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        for _ in range(4):
            for i in range(len(chosen)):
                cur, bestw = -2.0, w[i]
                for g in grid:
                    w2 = w.copy()
                    w2[i] = g
                    mu2, sd2 = null_stats(LM, w2, rng)
                    s = score(LM[chosen], w2, target, mu2, sd2)
                    if s > cur:
                        cur, bestw = s, g
                w[i] = bestw
        mu, sd = null_stats(LM, w, rng)
    return chosen, w, score(LM[chosen], w, target, mu, sd), trace, (mu, sd)


def owned_names(library: set[str]) -> set[str]:
    path = ROOT / "data" / "inventory" / "inventory.csv"
    if not path.is_file():
        return set()
    out = set()
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        n = (r.get("material") or r.get("name") or "").strip()
        if n in library:
            out.add(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                    help='e.g. "melon=1,cucumber=0.6,green=0.5"')
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--owned", action="store_true",
                    help="restrict to materials the perfumer already has")
    ap.add_argument("--recover", default=None,
                    help="a formula CSV: search its target, report recall")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    names = list(pj["molecules"])
    LM = np.stack([np.asarray(r, float) for r in pj["probs"]])
    rng = np.random.default_rng(args.seed)

    pool = list(range(len(names)))
    if args.owned:
        own = owned_names(set(names))
        pool = [i for i, n in enumerate(names) if n in own]
        print(f"[reverse] pool restricted to {len(pool)} materials on hand")

    if args.recover:
        from parse_formulas import classify
        import predict_formula as pf

        rows = pf.read_formula(ROOT / args.recover)
        ing: list[tuple[str, float]] = []
        for r in rows:
            k, n = classify(r["name"], set(names))
            if k == "material" and n in names:
                ing.append((n, r["neat"]))
        truth = {n for n, _ in ing}
        # Target: the formula's own lift profile at its real mass fractions,
        # thresholded to its clear peaks. Giving the search the raw profile
        # would hand it the answer; using equal weights would describe a
        # different blend from the one the perfumer actually made.
        sel = [names.index(n) for n, _ in ing]
        wt = np.array([x for _, x in ing], float)
        mu, sd = null_stats(LM, wt, rng)
        z = (blend(LM[sel], wt) - mu) / sd
        keep = np.argsort(-z)[:6]
        target = np.zeros(len(desc))
        target[keep] = z[keep]
        target /= np.linalg.norm(target)
        print(f"[reverse] recovering {Path(args.recover).name}: "
              f"{len(truth)} known ingredients")
        print("  target taken from its own top-6 lift descriptors: "
              + ", ".join(desc[i] for i in keep))
        pool = [i for i in pool]
        chosen, w, s, trace, _ = search(LM, names, target, size=len(truth),
                                        pool=pool, rng=rng)
        got = {names[i] for i in chosen}
        hit = got & truth
        print(f"\n  recovered {len(hit)}/{len(truth)}: "
              + (", ".join(sorted(hit)) or "none"))
        print(f"  also proposed: " + ", ".join(sorted(got - truth)))
        base = len(truth) / len(pool)
        print(f"  chance recall for {len(truth)} picks from {len(pool)}: "
              f"{base:.1%} per pick, ~{len(truth) * base:.2f} expected hits")
        return 0

    if not args.target:
        raise SystemExit("give --target or --recover")
    target = parse_target(args.target, desc)
    named = [desc[i] for i in np.argsort(-target) if target[i] > 0]
    print(f"[reverse] target: {', '.join(named)}")
    print(f"  searching {args.size} materials from a pool of {len(pool)}\n")

    chosen, w, s, trace, (mu, sd) = search(LM, names, target, size=args.size,
                                           pool=pool, rng=rng)
    print("  greedy build-up:")
    for t in trace:
        print(f"    {t['step']}. {t['added']:34s} match {t['score']:+.3f}")

    order = np.argsort(-w)
    print(f"\n  proposed accord (match {s:+.3f}; proportions, NOT doses):")
    tot = w.sum()
    for k in order:
        print(f"    {100 * w[k] / tot:5.1f}%  {names[chosen[k]]}")

    z = (blend(LM[chosen], w) - mu) / sd
    print("\n  what the blend is predicted to be unusually high in:")
    for i in np.argsort(-z)[:10]:
        star = " <- asked for" if target[i] > 0 else ""
        print(f"    {desc[i]:14s} z {z[i]:+6.2f}{star}")
    miss = [desc[i] for i in np.argsort(-target) if target[i] > 0
            and z[i] < 1.0]
    if miss:
        print(f"\n  asked for but not delivered (z < 1): {', '.join(miss)}")

    if args.out:
        rep = {"target": args.target, "size": args.size,
               "owned_only": bool(args.owned), "match": round(s, 4),
               "accord": [{"name": names[chosen[k]],
                           "proportion": round(float(w[k] / tot), 4)}
                          for k in order],
               "trace": trace,
               "profile": [{"name": desc[i], "z": round(float(z[i]), 3)}
                           for i in np.argsort(-z)[:20]]}
        (ROOT / args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[reverse] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
