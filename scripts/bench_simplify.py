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

**Two mixture rules, and the benchmark scores both.** The page mixes a blend by
the **mean of logits** (`logit(mix) = sum_i w_i logit(p_i)`); the first table
validates that rule, because a benchmark that scores something the page does
not do is worthless. The second table re-fits every choice of materials under
**noisy-OR**, the union, which is the alternative the atlas and the oils used
before November 2026 and which stays reachable in both.

Neither rule is the accurate one, and that is a result, not a hedge:
`rule_bakeoff.py` scores five rules on 19 oils against the model's own
vocabulary and cannot separate them (p > 0.2). The mean of logits was adopted
because mixing in log-odds is linear, so the weight fit is exact and a formula
reproduced at its own size comes back exactly; because it is the rule the atlas
places a blend by, so the page and the map agree; and because it does not
saturate with ingredient count. It does dilute a one-sided character by 1/N,
which the union does not -- see the page's own caveats.

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


def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def fit_simplex(X, idx, t, iters=400):
    """Weights for the mean-of-logits rule, by the same closed-form solve the
    page uses: mixing is additive in log-odds, so this is least squares there
    with the answer projected onto the simplex."""
    if len(idx) == 1:
        return np.ones(1)
    Q = _logit(X[idx])
    tq = _logit(t)
    A = Q @ Q.T + 1e-10 * np.eye(len(idx))
    try:
        w = np.linalg.solve(A, Q @ tq)
    except np.linalg.LinAlgError:
        return np.full(len(idx), 1.0 / len(idx))
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(len(idx), 1.0 / len(idx))


def err(X, idx, t):
    """Distance of the best log-odds blend of X[idx] from t, in probability
    space -- the number the page reports."""
    w = fit_simplex(X, idx, t)
    mix = 1.0 / (1.0 + np.exp(-(w @ _logit(X[idx]))))
    return float(np.linalg.norm(mix - t)), w


def fit_noisy(X, idx, t, sweeps=24):
    """Best convex weights for a noisy-OR target, by coordinate descent.

    Seeded with the linear fit, which is a good starting point even though the
    objective is not, then moved one coordinate at a time and renormalised.
    The step halves whenever a sweep finds nothing better, so this converges to
    a local optimum of the rule the page would have to adopt to match the CLI.
    """
    def dist_no(w):
        return float(np.linalg.norm(combine("noisy-or", X[idx], w) - t))

    w = np.array(fit_simplex(X, idx, t), float)
    s = w.sum()
    w = w / s if s > 0 else np.full(len(idx), 1.0 / len(idx))
    best = dist_no(w)
    step = 0.25
    for _ in range(sweeps):
        moved = False
        for j in range(len(w)):
            for sgn in (1.0, -1.0):
                trial = w.copy()
                trial[j] = max(trial[j] + sgn * step, 0.0)
                st = trial.sum()
                if st <= 0:
                    continue
                trial = trial / st
                dv = dist_no(trial)
                if dv < best - 1e-12:
                    best, w, moved = dv, trial, True
        if not moved:
            step /= 2.0
            if step < 1e-3:
                break
    return w, best


def greedy(X, Q, pool, t, k, seed=None):
    """Greedy on the unconstrained log-odds residual, then one swap pass.

    ``seed`` starts the swap pass from a given subset instead of from the
    greedy pick. The benchmark is the reason this exists: greedy selection
    optimises the *unconstrained* residual because that is what a Gram matrix
    makes cheap, while the answer is finally fitted under a simplex
    constraint, and the two disagree often enough that the search lost to
    "keep the biggest k ingredients" on 64% of the oils. Running the swap pass
    from the heuristic as well, and keeping whichever ends up closer, makes
    the search no worse than the obvious thing by construction.
    """
    tq = _logit(t)

    def resid(S):
        A = Q[S] @ Q[S].T + 1e-10 * np.eye(len(S))
        try:
            w = np.linalg.solve(A, Q[S] @ tq)
        except np.linalg.LinAlgError:
            return np.inf
        return max(float(tq @ tq - w @ (Q[S] @ tq)), 0.0)

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
    ap.add_argument("--noisy-draws", type=int, default=12,
                    help="random blends re-fitted under noisy-OR; each one is "
                         "an iterative fit, so fewer than --random-draws")
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
    rule_gap: list[float] = []
    Q = _logit(X)          # every material in log-odds, once
    for tgt in targets:
        ing = tgt["ing"]
        idx_all = [at[n] for n, _ in ing]
        w_all = np.array([w for _, w in ing], float)
        w_all = w_all / w_all.sum()
        # The page's rule, and the one this table validates.
        t = combine("logit-mean", X[idx_all], w_all)
        # The union, which the atlas and the oils used before November 2026.
        # Reported so the two can be compared; the gap is in log-odds because
        # that is where the page's rule is linear.
        t_no = combine("noisy-or", X[idx_all], w_all)
        rule_gap.append(float(np.linalg.norm(_logit(t) - _logit(t_no))))
        pool = [i for i in range(len(X)) if i not in set(tgt["exclude"])]
        for k in range(2, args.max_k + 1):
            if k >= len(ing):
                continue
            # the hand heuristic: the k biggest, given their best weights
            topk = idx_all[:k]
            e_top, _ = err(X, topk, t)
            cands = [greedy(X, Q, pool, t, k),
                     greedy(X, Q, pool, t, k, seed=topk),
                     topk]
            scored = [(err(X, c, t)[0], c) for c in cands]
            e_ours, ours = min(scored, key=lambda z: z[0])
            e_greedy = scored[0][0]
            picks = []
            ds = []
            for _ in range(args.random_draws):
                pick = list(rng.choice(pool, k, replace=False))
                picks.append(pick)
                ds.append(err(X, pick, t)[0])
            e_rand = float(np.median(ds))

            # The same three choices, scored under noisy-OR with weights fitted
            # under noisy-OR. The choice of materials is what is being tested;
            # the dosing is re-fitted so the rule, not the fit, is what differs.
            _, ours_no = fit_noisy(X, ours, t_no)
            _, topk_no = fit_noisy(X, topk, t_no)
            random_no = float(np.median(
                [fit_noisy(X, p, t_no)[1] for p in picks[:args.noisy_draws]]))

            recovered = len(set(ours) & set(topk))
            rows.append({"target": tgt["name"], "kind": tgt["kind"], "k": k,
                         "ours": e_ours, "greedy_only": e_greedy,
                         "topk": e_top, "random": e_rand,
                         "ours_no": ours_no, "topk_no": topk_no,
                         "random_no": random_no,
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
        on = np.array([r["ours_no"] for r in sub])
        tn = np.array([r["topk_no"] for r in sub])
        rn = np.array([r["random_no"] for r in sub])
        return {"n": len(sub), "ours": float(np.median(o)),
                "greedy_only": float(np.median(gd)),
                "greedy_beats_topk": float((gd < tp).mean()),
                "topk": float(np.median(tp)), "random": float(np.median(rd)),
                "beats_topk": float((o < tp - 1e-9).mean()),
                "never_worse": float((o <= tp + 1e-9).mean()),
                "recovery": float(np.mean([r["recovered"] / r["k"] for r in sub])),
                # the same proposals, weights re-fitted under noisy-OR
                "ours_no": float(np.median(on)),
                "topk_no": float(np.median(tn)),
                "random_no": float(np.median(rn)),
                "beats_topk_no": float((on < tn - 1e-9).mean()),
                "never_worse_no": float((on <= tn + 1e-9).mean()),
                "random_ratio": float(np.median(o / np.maximum(rd, 1e-9))),
                "random_ratio_no": float(np.median(on / np.maximum(rn, 1e-9)))}

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

    print("=== the same choices, scored under the union (noisy-OR)")
    print("    weights re-fitted under noisy-OR; only the rule differs")
    print(f"  {'k':>2s} {'n':>4s} {'ours':>8s} {'biggest k':>10s} "
          f"{'random':>8s} {'ours/random':>12s} {'better/not worse':>17s}")
    for kind in ("oil", "formula"):
        for k in range(2, args.max_k + 1):
            a = agg(kind, k)
            if not a:
                continue
            print(f"  {k:2d} {a['n']:4d} {a['ours_no']:8.3f} {a['topk_no']:10.3f} "
                  f"{a['random_no']:8.3f} {a['random_ratio_no']:11.2f}x "
                  f"{a['beats_topk_no']:4.0%}/{a['never_worse_no']:.0%}   ({kind})")
    print("")
    print("=== the page's rule (mean of logits), ours/random at the same k")
    for kind in ("oil", "formula"):
        for k in range(2, args.max_k + 1):
            a = agg(kind, k)
            if a:
                print(f"  {k:2d} {a['n']:4d} {a['ours']:8.3f} {a['topk']:10.3f} "
                      f"{a['random']:8.3f} {a['random_ratio']:11.2f}x "
                      f"{a['beats_topk']:4.0%}/{a['never_worse']:.0%}   ({kind})")
    if rule_gap:
        g = np.array(rule_gap)
        print(f"\n[bench] the two targets for one blend sit a median {np.median(g):.3f} "
              f"apart (max {g.max():.3f}); the page's yardstick is 0.488")

    print("\noils at k=3, what it picked against what the oil is made of:")
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
