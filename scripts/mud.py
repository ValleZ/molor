"""The fewest materials that land on the mud point, chosen on purpose.

Twenty random ingredients approximate the centre of percept space, which is
unimpressive: anyone can mix twenty things badly. The interesting question is
the minimum. If three materials chosen deliberately sit closer to the average
point than thirty random ones, then "olfactory white" is not a property of
having many ingredients at all -- it is a location, and a location can be
reached directly.

So this is a different computation from ``olfactory_white.py``. That one asked
when independent *random* blends stop being distinguishable. This one solves
for the blend of k materials whose predicted percept vector is nearest the
library's average point:

    min ||Bᵀw − μ||  subject to w ≥ 0, Σw = 1

for each small k, which is the projection of μ onto the convex hull of the
chosen k. Pairs are solved exactly in closed form over every pair in the
library. From three upwards the subset choice is a local search from several
starts -- greedy, then swap each member against every candidate until nothing
improves -- so the results are good but not proven optimal, and the script says
so rather than claiming a minimum it has not established.

The yardstick is the same one: the median distance from a material to its
nearest neighbour in the library, which is the scale at which a perfumer
struggles to tell two things apart. A blend inside that distance of μ is, by
the model's reckoning, at the mud point rather than near it.

    python scripts/mud.py --owned --emit
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
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def thresholds() -> dict[str, float]:
    """Measured water-phase detection thresholds in ppb, by material name.

    Only measured values: the predicted ones were withdrawn from this project
    after validating at Spearman 0.14 against the perfumer's own use levels.
    """
    path = ROOT / "data" / "inventory" / "inventory.csv"
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or r.get("material") or "").strip()
        v = (r.get("water_threshold_ppb") or "").strip()
        if n and v:
            try:
                out[n] = float(v)
            except ValueError:
                pass
    return out


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


def masses(shares: np.ndarray, th: np.ndarray, exponent: float) -> np.ndarray:
    """Convert shares of odour activity into mass fractions."""
    m = shares * th ** exponent
    s = m.sum()
    return m / s if s > 0 else np.full_like(m, 1.0 / len(m))


def fit(B: np.ndarray, target: np.ndarray, lam: float = 80.0):
    """Best convex blend of the rows of B for one target. Returns (w, residual)."""
    k = B.shape[0]
    if k == 1:
        return np.ones(1), float(np.linalg.norm(B[0] - target))
    if k == 2:
        a, b = B
        d = a - b
        denom = float(d @ d)
        t = 0.5 if denom < 1e-12 else float((target - b) @ d / denom)
        t = min(1.0, max(0.0, t))
        w = np.array([t, 1 - t])
        return w, float(np.linalg.norm(w @ B - target))
    A = np.vstack([B.T, lam * np.ones((1, k))])
    y = np.concatenate([target, [lam]])
    w = nnls(A, y)[0]
    s = w.sum()
    if s <= 0:
        w = np.full(k, 1.0 / k)
    else:
        w = w / s
    return w, float(np.linalg.norm(w @ B - target))


def best_pair(X: np.ndarray, pool: list[int], target: np.ndarray):
    """Exhaustive over every pair, closed form per pair."""
    P = X[pool]
    best = (np.inf, None, None)
    for i in range(len(pool)):
        d = P[i] - P                                  # (m, D)
        denom = np.einsum("ij,ij->i", d, d)
        num = np.einsum("ij,ij->i", target - P, d)
        t = np.clip(np.divide(num, np.maximum(denom, 1e-12)), 0.0, 1.0)
        mix = t[:, None] * P[i] + (1 - t)[:, None] * P
        r = np.linalg.norm(mix - target, axis=1)
        r[i] = np.inf
        j = int(np.argmin(r))
        if r[j] < best[0]:
            best = (float(r[j]), [pool[i], pool[j]],
                    np.array([t[j], 1 - t[j]]))
    return best


def search(X: np.ndarray, pool: list[int], target: np.ndarray, k: int,
           rng, restarts: int = 8, feasible=None):
    """Greedy then swap-until-stable, from several random starts.

    ``feasible`` rejects a subset whose implied doses cannot be weighed. Without
    it the odour-activity conversion returns arithmetically correct nonsense:
    thresholds span orders of magnitude, so the solver happily proposes a blend
    that is 96% cinnamyl alcohol and 0.0% pistachio thiazole, which is one
    material with contamination rather than a five-material white.
    """
    def score(idx):
        w, res = fit(X[idx], target)
        if feasible is not None and not feasible(idx, w):
            return w, np.inf
        return w, res

    best = (np.inf, None, None)
    for r in range(restarts):
        if r == 0:
            chosen: list[int] = []
            for _ in range(k):
                cand = [c for c in pool if c not in chosen]
                # Partial subsets are scored without the feasibility test,
                # which only makes sense on a complete blend.
                scores = [(fit(X[chosen + [c]], target)[1], c) for c in cand]
                chosen.append(min(scores)[1])
        else:
            chosen = [int(i) for i in rng.choice(pool, k, replace=False)]
        w, res = score(chosen)
        improved = True
        while improved:
            improved = False
            for slot in range(k):
                for c in pool:
                    if c in chosen:
                        continue
                    trial = list(chosen)
                    trial[slot] = c
                    w2, r2 = score(trial)
                    if r2 < res - 1e-9:
                        chosen, w, res, improved = trial, w2, r2, True
        if res < best[0]:
            best = (res, chosen, w)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--owned", action="store_true")
    ap.add_argument("--target", default="centroid",
                    choices=["centroid", "flat", "dull"],
                    help="which white. centroid: the library's average smell, "
                         "which is NOT a flat profile -- it is sweet 0.45, "
                         "fruity 0.44, floral 0.38, green 0.35, so it has "
                         "character by construction and calling it mud was "
                         "wrong. flat: every descriptor equally applicable at "
                         "--level, the equal-energy white. dull: the blend "
                         "that smells least of anything, the origin")
    ap.add_argument("--level", type=float, default=None,
                    help="for --target flat: the level every descriptor is "
                         "asked to sit at. The default is the centroid's own "
                         "mean, 0.05, so the two whites are the same "
                         "brightness and differ only in shape")
    ap.add_argument("--exclude", action="append", default=None,
                    help="a material to keep out; repeatable")
    ap.add_argument("--potency", default="mass",
                    choices=["mass", "threshold"],
                    help="mass: the solved weights are mass fractions, which "
                         "is what the first attempt assumed and what the "
                         "bench refuted. threshold: the solved weights are "
                         "odour-activity shares and the sheet converts them "
                         "to masses by multiplying each by its measured "
                         "detection threshold, so a powerful material is "
                         "dosed down")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--emit-k", type=int, action="append", default=None)
    ap.add_argument("--out", default="data/formulas/queries/mud")
    ap.add_argument("--mass-floor", type=float, default=2.0,
                    help="smallest share of the blend any ingredient may end "
                         "up at, in percent, once doses are derived from "
                         "thresholds")
    ap.add_argument("--mass-ceiling", type=float, default=60.0,
                    help="largest share any one ingredient may end up at")
    ap.add_argument("--oav-exponent", type=float, default=1.0,
                    help="mass goes as activity share times threshold to this "
                         "power. 1.0 is straight odour-activity dosing. The "
                         "bench wanted methyl benzoate at 5%% where 1.0 gave "
                         "7%%, which is about 1.2, so the correction may need "
                         "to be slightly stronger than linear")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    names = list(pj["molecules"])
    X = np.asarray(pj["probs"], float)
    mu = X.mean(0)
    lv = use_levels()

    thr = thresholds()
    pool = list(range(len(names)))
    if args.owned:
        pool = [i for i, n in enumerate(names) if n in lv]
    if args.exclude:
        drop = {e.strip().lower() for e in args.exclude}
        before = len(pool)
        pool = [i for i in pool if names[i].lower() not in drop]
        missing = drop - {names[i].lower() for i in range(len(names))}
        print(f"[mud] excluded {before - len(pool)} material(s)"
              + (f"; not in the library: {sorted(missing)}" if missing else ""))
    if args.potency == "threshold":
        # A material with no measured threshold cannot be dosed by one, and
        # the predicted thresholds are not trustworthy enough to stand in --
        # so the pool shrinks rather than the method being fudged.
        before = len(pool)
        pool = [i for i in pool if names[i] in thr]
        print(f"[mud] odour-activity dosing: {len(pool)} of {before} "
              f"materials have a measured threshold")
    rng = np.random.default_rng(args.seed)

    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    yard = float(np.median(D.min(1)))
    if args.target == "centroid":
        target, tname = mu, "the library's average smell"
    elif args.target == "flat":
        lvl = args.level if args.level is not None else float(mu.mean())
        target = np.full_like(mu, lvl)
        tname = f"every descriptor equally applicable at {lvl:.3f}"
    else:
        target = np.zeros_like(mu)
        tname = "the origin: smelling as little of anything as possible"
    print(f"[mud] {len(pool)} materials, target is {tname}, "
          f"yardstick {yard:.3f}")
    print(f"[mud] for reference the centroid is NOT flat: mean {mu.mean():.3f} "
          f"but sd {mu.std():.3f} across descriptors, and its top four are "
          f"sweet {mu[pj['descriptors'].index('sweet')]:.2f}, fruity, floral, "
          f"green -- which is why it has character")

    dist_mu = np.linalg.norm(X - target, axis=1)

    print(f"[mud] for comparison, random blends reach: 3 materials 0.737, "
          f"12 materials 0.407, 20 materials 0.283, 50 materials 0.191\n")

    # What can be weighed. Only applies when doses are derived from
    # thresholds, since mass weighting puts every ingredient where the solver
    # put it by definition.
    feasible = None
    if args.potency == "threshold":
        def feasible(idx, w):
            m = masses(np.asarray(w, float),
                       np.array([thr[names[i]] for i in idx]),
                       args.oav_exponent)
            return (100 * m.min() >= args.mass_floor
                    and 100 * m.max() <= args.mass_ceiling)
        print(f"[mud] doses must land between {args.mass_floor:g}% and "
              f"{args.mass_ceiling:g}% of the blend to count as weighable")

    rows = []
    for k in range(1, args.max_k + 1):
        if k == 1:
            i = int(min(pool, key=lambda j: dist_mu[j]))
            res, chosen, w = float(dist_mu[i]), [i], np.ones(1)
        elif k == 2:
            res, chosen, w = best_pair(X, pool, target)
        else:
            res, chosen, w = search(X, pool, target, k, rng, args.restarts,
                                    feasible=feasible)
        mix = w @ X[chosen]
        # What single material is nearest the blend? If the answer is "nothing
        # close", the blend is somewhere no material is, which is the point.
        dn = np.linalg.norm(X - mix, axis=1)
        near = int(np.argmin(dn))
        desc = pj["descriptors"]
        o = np.argsort(-mix)
        rows.append({"k": k, "residual": res, "names": [names[i] for i in chosen],
                     "profile_mean": float(mix.mean()),
                     "profile_sd": float(mix.std()),
                     "top": [[desc[int(i)], round(float(mix[i]), 3)]
                             for i in o[:4]],
                     "weights": [round(float(x), 4) for x in w],
                     "nearest": names[near], "nearest_d": float(dn[near])})
        tag = "  <- inside the yardstick" if res < yard else ""
        print(f"  k={k}  distance to the mud point {res:.3f}{tag}")
        for i, x in sorted(zip(chosen, w), key=lambda p: -p[1]):
            cap = lv.get(names[i], float("nan"))
            lim = "" if cap != cap else f"   [you use it up to {cap:g}%]"
            print(f"        {100 * x:5.1f}%  {names[i]}{lim}")
        print(f"        nearest single material to this blend: "
              f"{names[near]} at {dn[near]:.3f}")
        print(f"        its profile: mean {mix.mean():.3f}, sd "
              f"{mix.std():.3f}, led by "
              + ", ".join(f"{desc[int(i)]} {mix[i]:.2f}" for i in o[:4]))

    print("\n[mud] the search above is exact for k=2 and a local search from "
          f"{args.restarts} starts for k>=3, so these are upper bounds on the "
          "distance, not proven minima")

    if args.emit:
        outdir = ROOT / args.out
        outdir.mkdir(parents=True, exist_ok=True)
        for k in (args.emit_k or [3, args.max_k]):
            row = next(r for r in rows if r["k"] == k)
            suffix = "_oav" if args.potency == "threshold" else ""
            stem = {"centroid": "mud", "flat": "white_flat",
                    "dull": "white_dull"}[args.target]
            path = outdir / f"{stem}_{k}{suffix}.csv"
            over = []
            with open(path, "w", encoding="utf-8", newline="") as fh:
                cw = csv.writer(fh)
                cw.writerow(["material", "dilution", "weight"])
                shares = np.asarray(row["weights"], float)
                if args.potency == "threshold":
                    # The solved weights are shares of odour activity. Mass is
                    # activity times threshold: to contribute the same amount
                    # of smell, a material detectable at 0.5 ppb needs six
                    # times less mass than one detectable at 3 ppb. This is
                    # the correction the first mud blend lacked, where methyl
                    # benzoate at a third of the mass had six times the
                    # threshold advantage and took the blend over.
                    shares = masses(
                        shares, np.array([thr[n_] for n_ in row["names"]]),
                        args.oav_exponent)
                for n_, x in zip(row["names"], shares):
                    cw.writerow([n_, 100, round(100 * float(x), 2)])
                    cap = lv.get(n_, float("nan"))
                    if cap == cap and 100 * x > cap * 1.02 + 0.02:
                        over.append((n_, 100 * float(x), cap))
            if args.potency == "threshold":
                print("      doses are odour-activity corrected: "
                      + ", ".join(f"{n_} {100 * s:.1f}% (activity share "
                                  f"{100 * w_:.0f}%)"
                                  for n_, s, w_ in zip(row["names"], shares,
                                                       row["weights"])))
            print(f"[mud] wrote {path.relative_to(ROOT)}  "
                  f"(distance {row['residual']:.3f})")
            if over:
                print("      [!] above your own use level, so smell it "
                      "diluted: "
                      + ", ".join(f"{n_} at {s:.1f}% vs {c:g}%"
                                  for n_, s, c in over))
        (outdir / "prediction.json").write_text(json.dumps(
            {"target": "library average point in percept space",
             "yardstick": yard, "owned_only": bool(args.owned),
             "claim": "these blends should smell of nothing in particular, "
                      "and the k=3 one should be about as characterless as a "
                      "blend of thirty random materials",
             "blends": rows}, indent=2), encoding="utf-8")

    (ROOT / "artifacts" / "mud.json").write_text(json.dumps(
        {"yardstick": yard, "owned_only": bool(args.owned), "blends": rows},
        indent=2), encoding="utf-8")
    print("[mud] wrote artifacts/mud.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
