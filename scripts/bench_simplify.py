"""Does the simplifier beat keeping the biggest ingredients? And does it find
the constituents a natural is actually made of?

Nobody publishes a formula next to its official simplification, so there is no
list of correct answers to score against. There are two things better than
nothing, and both are available here.

*The heuristic anyone would use by hand.* Asked to cut a formula to five
materials, a perfumer keeps the five largest and rebalances. If the search
cannot beat that, it is an expensive way to sort by weight. The comparison is
made fair by giving the heuristic its best possible convex weights too, so what
is being compared is the *choice of materials* and not the dosing.

*The naturals, where the answer is known.* An oil built from a published GC
composition has a true simplification: its own major constituents. Asked to
reproduce bergamot oil from two materials without being allowed bergamot oil
itself, a search that proposes linalyl acetate and limonene has rediscovered
what bergamot is; one that proposes two unrelated esters has found a numerical
coincidence. This is scored as recovery: how many of the oil's k largest
constituents appear in the proposal.

Both numbers are reported per target and in aggregate, with a random baseline
underneath so the scale is visible.

    python scripts/bench_simplify.py --max-k 5
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixtures import combine  # noqa: E402
from parse_formulas import classify  # noqa: E402


def fit_simplex(X, idx, t, iters=400):
    """Best convex combination of X[idx] for target t."""
    B = X[idx]
    if len(idx) == 1:
        return np.ones(1)
    G = B @ B.T
    b = B @ t
    L = max(float(np.trace(G)), 1e-9)
    w = np.full(len(idx), 1.0 / len(idx))
    for _ in range(iters):
        v = w - (G @ w - b) / L
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        k = np.arange(1, len(u) + 1)
        rho = np.nonzero(u - (css - 1) / k > 0)[0]
        th = (css[rho[-1]] - 1) / (rho[-1] + 1) if len(rho) else 0.0
        w = np.maximum(v - th, 0.0)
        s = w.sum()
        w = w / s if s > 0 else np.full(len(idx), 1.0 / len(idx))
    return w


def err(X, idx, t):
    w = fit_simplex(X, idx, t)
    return float(np.linalg.norm(w @ X[idx] - t)), w


def greedy(X, pool, t, k, G, b, tt, seed=None):
    """Greedy on the unconstrained residual, then one swap pass.

    ``seed`` starts the swap pass from a given subset instead of from the
    greedy pick. The benchmark is the reason this exists: greedy selection
    optimises the *unconstrained* residual because that is what a Gram matrix
    makes cheap, while the answer is finally fitted under a simplex
    constraint, and the two disagree often enough that the search lost to
    "keep the biggest k ingredients" on 64% of the oils. Running the swap pass
    from the heuristic as well, and keeping whichever ends up closer, makes
    the search no worse than the obvious thing by construction.
    """
    def resid(S):
        A = G[np.ix_(S, S)] + 1e-9 * np.eye(len(S))
        try:
            w = np.linalg.solve(A, b[S])
        except np.linalg.LinAlgError:
            return np.inf
        return max(float(tt - w @ b[S]), 0.0)

    chosen: list[int] = list(seed) if seed else []
    while len(chosen) < k:
        best, bestr = -1, np.inf
        for c in pool:
            if c in chosen:
                continue
            r = resid(chosen + [c])
            if r < bestr:
                bestr, best = r, c
        if best < 0:
            break
        chosen.append(best)
    for slot in range(len(chosen)):
        cur, keep = resid(chosen), chosen[slot]
        for c in pool:
            if c in chosen:
                continue
            trial = list(chosen)
            trial[slot] = c
            r = resid(trial)
            if r < cur - 1e-12:
                cur, keep = r, c
        chosen[slot] = keep
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--random-draws", type=int, default=40)
    ap.add_argument("--out", default="artifacts/bench_simplify.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    names = list(pj["molecules"])
    P = {n: np.asarray(r, float) for n, r in zip(names, pj["probs"])}

    import predict_formula as pf
    oils = pf.load_oils("library")
    for o, v in oils.items():
        P[o] = np.asarray(v["percept"], float)
        names.append(o)
    X = np.stack([P[n] for n in names])
    at = {n: i for i, n in enumerate(names)}
    lib = set(names)

    # Targets: every oil, with its own composition as the known answer, and
    # every formula we hold, where the known answer is its largest ingredients.
    targets = []
    for o, v in oils.items():
        comp = sorted(v["composition_used"].items(), key=lambda kv: -kv[1])
        ing = [(n, w) for n, w in comp if n in at]
        if len(ing) >= 3:
            targets.append({"name": o, "kind": "oil", "ing": ing,
                            "exclude": [at[o]]})

    def add_formula(label, pairs):
        ing = []
        for nm, amt in pairs:
            kind, res = classify(nm, lib)
            if kind == "material" and res in at:
                ing.append((res, float(amt)))
            else:
                hit, _ = pf.match_oil(nm, list(oils))
                if hit:
                    ing.append((hit, float(amt)))
        agg: dict[str, float] = {}
        for n, a in ing:
            agg[n] = agg.get(n, 0.0) + a
        ing = sorted(agg.items(), key=lambda kv: -kv[1])
        if len(ing) >= 4:
            targets.append({"name": label, "kind": "formula", "ing": ing,
                            "exclude": []})

    for f in sorted(glob.glob(str(ROOT / "data/formulas/published/*.json"))):
        blob = json.loads(Path(f).read_text(encoding="utf-8"))
        for fm in blob.get("formulas", []):
            rows = fm.get("ingredients") or []
            pairs = []
            for r in rows:
                if isinstance(r, dict):
                    nm = r.get("material") or r.get("name") or ""
                    amt = r.get("parts") or r.get("amount") or r.get("weight") or 0
                else:
                    continue
                if nm and float(amt or 0) > 0:
                    pairs.append((nm, amt))
            if pairs:
                add_formula(fm.get("name", Path(f).stem), pairs)

    for csvf in sorted(glob.glob(str(ROOT / "data/formulas/queries/*.csv"))):
        rows = pf.read_formula(Path(csvf))
        add_formula("query: " + Path(csvf).stem,
                    [(r["name"], r["neat"]) for r in rows])

    print(f"[bench] {len(targets)} targets: "
          f"{sum(1 for t in targets if t['kind'] == 'oil')} oils, "
          f"{sum(1 for t in targets if t['kind'] == 'formula')} formulas\n")

    rng = np.random.default_rng(0)
    rows = []
    for tgt in targets:
        ing = tgt["ing"]
        idx_all = [at[n] for n, _ in ing]
        w_all = np.array([w for _, w in ing], float)
        w_all = w_all / w_all.sum()
        t = combine("weighted-mean", X[idx_all], w_all)
        pool = [i for i in range(len(X)) if i not in set(tgt["exclude"])]
        G = X[pool] @ X[pool].T
        # G and b indexed in pool space for the greedy
        Gp = np.zeros((len(X), len(X)))
        Gp[np.ix_(pool, pool)] = G
        b = X @ t
        tt = float(t @ t)
        for k in range(2, args.max_k + 1):
            if k >= len(ing):
                continue
            # the hand heuristic: the k biggest, given their best weights
            topk = idx_all[:k]
            e_top, _ = err(X, topk, t)
            cands = [greedy(X, pool, t, k, Gp, b, tt),
                     greedy(X, pool, t, k, Gp, b, tt, seed=topk),
                     topk]
            scored = [(err(X, c, t)[0], c) for c in cands]
            e_ours, ours = min(scored, key=lambda z: z[0])
            e_greedy = scored[0][0]
            ds = []
            for _ in range(args.random_draws):
                pick = list(rng.choice(pool, k, replace=False))
                ds.append(err(X, pick, t)[0])
            e_rand = float(np.median(ds))
            recovered = len(set(ours) & set(topk))
            rows.append({"target": tgt["name"], "kind": tgt["kind"], "k": k,
                         "ours": e_ours, "greedy_only": e_greedy,
                         "topk": e_top, "random": e_rand,
                         "recovered": recovered,
                         "picked": [names[i] for i in ours],
                         "true_topk": [names[i] for i in topk]})

    def agg(kind, k):
        sub = [r for r in rows if r["kind"] == kind and r["k"] == k]
        if not sub:
            return None
        o = np.array([r["ours"] for r in sub])
        tp = np.array([r["topk"] for r in sub])
        rd = np.array([r["random"] for r in sub])
        gd = np.array([r["greedy_only"] for r in sub])
        return {"n": len(sub), "ours": float(np.median(o)),
                "greedy_only": float(np.median(gd)),
                "greedy_beats_topk": float((gd < tp).mean()),
                "topk": float(np.median(tp)), "random": float(np.median(rd)),
                "beats_topk": float((o < tp - 1e-9).mean()),
                "never_worse": float((o <= tp + 1e-9).mean()),
                "recovery": float(np.mean([r["recovered"] / r["k"] for r in sub]))}

    for kind in ("oil", "formula"):
        print(f"=== {kind}s: median distance to the target, lower is better")
        print(f"  {'k':>2s} {'n':>4s} {'ours':>8s} {'biggest k':>10s} "
              f"{'random':>8s} {'better/not worse':>17s} {'recovery':>9s}")
        for k in range(2, args.max_k + 1):
            a = agg(kind, k)
            if not a:
                continue
            print(f"  {k:2d} {a['n']:4d} {a['ours']:8.3f} {a['topk']:10.3f} "
                  f"{a['random']:8.3f} {a['beats_topk']:4.0%}/{a['never_worse']:.0%} "
                  f"{a['recovery']:8.0%}   (search alone "
                  f"{a['greedy_only']:.3f}, beat the heuristic "
                  f"{a['greedy_beats_topk']:.0%} of the time)")
        print("")

    print("oils at k=3, what it picked against what the oil is made of:")
    for r in [x for x in rows if x["kind"] == "oil" and x["k"] == 3][:14]:
        print(f"  {r['target'][:34]:34s} ours {r['ours']:.3f} vs "
              f"biggest {r['topk']:.3f}")
        print(f"      picked: {', '.join(r['picked'])}")
        print(f"      real  : {', '.join(r['true_topk'])}")

    (ROOT / args.out).write_text(json.dumps(
        {"rows": rows,
         "summary": {kind: {k: agg(kind, k) for k in range(2, args.max_k + 1)}
                     for kind in ("oil", "formula")}}, indent=2),
        encoding="utf-8")
    print(f"\n[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
