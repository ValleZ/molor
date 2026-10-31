"""Could one fixed set of materials reproduce everything? The gamut question.

Choosing a few materials per target is one claim. A *fixed* palette -- twenty
five materials that between them reach any smell -- is a far stronger one, and
it is the olfactory version of asking whether three phosphors can make every
colour. The answer there is no, and the reason transfers exactly.

Under any averaging rule, a blend of a fixed palette is a convex combination of
its points, so the reachable set is the **convex hull** of the palette. Three
consequences follow before any computation:

1. A hull cannot reach outside itself. Whatever is more extreme than every
   palette member on some axis is unreachable at any weights. In colour this is
   the gamut; the extreme materials here are things like fenchone and the
   pinenes, which sit twice as far from the cloud's centre as a median material.
2. To have any volume in d effective dimensions the palette needs at least
   d + 1 members. Percept space needs 13 components for 80% of its variance and
   22 for 90%, so a palette below about 14 is degenerate by construction, no
   matter which materials it holds.
3. The useful palette members are therefore the *extremes*, not the typical
   ones -- the opposite of what a "representative selection" would pick.

Noisy-OR has a gamut too, in the opposite direction: mixing can only raise each
coordinate, so nothing below the palette's own minimum on an axis is reachable.
You cannot make a blend less floral than your least floral primary.

This script measures the gamut. For a palette B and a target t it solves

    min ||Bᵀw − t||  subject to w ≥ 0, Σw = 1

which is the projection of t onto the palette's hull, and reports how close the
best achievable blend gets. Palettes are grown greedily: at each step it adds
the material that most reduces the mean residual over all targets. Selection
uses the unconstrained projection, which is cheap enough to evaluate every
candidate at every step; the reported numbers use the constrained one.

    python scripts/palette.py --max-size 40
    python scripts/palette.py --owned --targets oils

What this cannot tell you: whether a blend that matches a target vector smells
like it. The vector is this model's opinion, so a high score is "indistinguish-
able to the model", and the model is wrong in known ways -- it cannot see a
trace component, and it has no time axis at all.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def owned_names() -> set[str]:
    path = ROOT / "data" / "inventory" / "inventory.csv"
    if not path.is_file():
        return set()
    out = set()
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or r.get("material") or "").strip()
        if n:
            out.add(n)
    return out


def hull_fit(B: np.ndarray, T: np.ndarray, lam: float = 50.0):
    """Project each row of T onto the convex hull of the rows of B.

    Solved as non-negative least squares on the system augmented with a heavily
    weighted row enforcing the weights' sum, which is the standard way to get a
    simplex-constrained fit out of NNLS. lam is large relative to the data so
    the sum constraint binds; the residual reported excludes that row.
    """
    k = B.shape[0]
    A = np.vstack([B.T, lam * np.ones((1, k))])
    W = np.zeros((T.shape[0], k))
    for i, t in enumerate(T):
        y = np.concatenate([t, [lam]])
        W[i] = nnls(A, y)[0]
    fit = W @ B
    return W, fit


def centred_cos(A: np.ndarray, Bv: np.ndarray, mu: np.ndarray) -> np.ndarray:
    a, b = A - mu, Bv - mu
    na = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    nb = np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    return np.einsum("ij,ij->i", a, b) / (na * nb)


def greedy(X: np.ndarray, T: np.ndarray, pool: list[int], kmax: int,
           report=None):
    """Grow a palette by the material that most reduces unconstrained residual.

    The unconstrained projection is an optimistic bound on the constrained one,
    but it is a *monotone* one and it can be evaluated for every candidate at
    every step with one pseudo-inverse each, which the constrained version
    cannot. The final palettes are then scored constrained.
    """
    chosen: list[int] = []
    for step in range(1, kmax + 1):
        best, best_err = None, np.inf
        for c in pool:
            if c in chosen:
                continue
            B = X[chosen + [c]]
            # least-squares residual of every target against this basis
            sol, *_ = np.linalg.lstsq(B.T, T.T, rcond=None)
            err = float(np.square(T.T - B.T @ sol).sum())
            if err < best_err:
                best, best_err = c, err
        chosen.append(best)
        if report:
            report(step, chosen, best)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--max-size", type=int, default=30)
    ap.add_argument("--owned", action="store_true",
                    help="palette restricted to materials on the shelf")
    ap.add_argument("--targets", default="materials",
                    choices=["materials", "oils", "both"])
    ap.add_argument("--out", default="artifacts/palette.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    names = list(pj["molecules"])
    X = np.asarray(pj["probs"], float)

    import predict_formula as pf
    oils = pf.load_oils("library")
    oil_names = list(oils)
    OX = (np.stack([np.asarray(oils[o]["percept"], float) for o in oil_names])
          if oil_names else np.zeros((0, X.shape[1])))

    if args.targets == "materials":
        T, tnames = X, names
    elif args.targets == "oils":
        T, tnames = OX, oil_names
    else:
        T = np.vstack([X, OX])
        tnames = names + oil_names

    pool = list(range(len(names)))
    if args.owned:
        own = owned_names()
        pool = [i for i, n in enumerate(names) if n in own]
    mu = X.mean(0)
    dist = np.linalg.norm(X - mu, axis=1)

    print(f"[palette] {len(pool)} candidate materials, {len(T)} targets, "
          f"{X.shape[1]} descriptors")
    print(f"[palette] a single material sits a median {np.median(dist):.3f} "
          f"from the average point; the extremes are "
          + ", ".join(f"{names[i]} {dist[i]:.2f}"
                      for i in np.argsort(-dist)[:4]))

    chosen = greedy(X, T, pool, args.max_size)
    print("\n[palette] greedy order, and what each size can reach "
          "(constrained to the hull):\n")
    print(f"  {'k':>3s} {'added':28s} {'median cos':>10s} {'>=0.90':>8s} "
          f"{'>=0.95':>8s} {'worst cos':>10s} {'median resid':>12s}")
    print("  " + "-" * 82)
    curve = []
    for k in range(1, len(chosen) + 1):
        if k > 3 and k % 2 and k != len(chosen):
            continue
        B = X[chosen[:k]]
        _, fit = hull_fit(B, T)
        cos = centred_cos(T, fit, mu)
        resid = np.linalg.norm(T - fit, axis=1)
        # A palette member is its own perfect match, so leaving the members in
        # the target set inflates every statistic by whatever fraction of the
        # targets the palette is -- 8% at k=25. They are excluded.
        if args.targets in ("materials", "both"):
            keep = np.array([tnames[i] not in {names[c] for c in chosen[:k]}
                             for i in range(len(T))])
            cos, resid = cos[keep], resid[keep]
        row = {"k": k, "added": names[chosen[k - 1]],
               "median_cos": float(np.median(cos)),
               "frac90": float((cos >= 0.90).mean()),
               "frac95": float((cos >= 0.95).mean()),
               "worst_cos": float(cos.min()),
               "median_resid": float(np.median(resid))}
        curve.append(row)
        print(f"  {k:3d} {row['added'][:28]:28s} {row['median_cos']:10.3f} "
              f"{row['frac90']:7.0%} {row['frac95']:7.0%} "
              f"{row['worst_cos']:10.3f} {row['median_resid']:12.3f}")

    B = X[chosen]
    _, fit = hull_fit(B, T)
    cos = centred_cos(T, fit, mu)
    resid = np.linalg.norm(T - fit, axis=1)
    print(f"\n[palette] the {len(chosen)} chosen, by how much weight they "
          f"carry across all targets:")
    W, _ = hull_fit(B, T)
    use = W.sum(0) / W.sum()
    for j in np.argsort(-use):
        i = chosen[j]
        print(f"    {use[j]:6.1%}  {names[i]:28s} "
              f"(sits {dist[i]:.2f} from the average point)")

    print(f"\n[palette] worst-reproduced targets at k={len(chosen)} -- the "
          f"gamut's edge:")
    for i in np.argsort(cos)[:10]:
        print(f"    cos {cos[i]:+.3f}  resid {resid[i]:.3f}  {tnames[i]}")

    never = [tnames[i] for i in range(len(T)) if cos[i] < 0.5]
    print(f"\n[palette] {len(never)}/{len(T)} targets stay below cosine 0.5 "
          f"with the whole palette available"
          + (": " + ", ".join(never[:12]) if never else ""))

    (ROOT / args.out).write_text(json.dumps(
        {"owned_only": bool(args.owned), "targets": args.targets,
         "palette": [names[i] for i in chosen], "curve": curve,
         "worst": [{"name": tnames[i], "cos": float(cos[i])}
                   for i in np.argsort(cos)[:20]]}, indent=2), encoding="utf-8")
    print(f"\n[palette] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
