"""Combine receptor vectors into a mixture profile, and search for a recipe.

Two separate things happen here, and only one of them is grounded in the model.

**Combining.** The model scores one molecule against one receptor and returns
P(this receptor responds). It has no concentration axis at all, so there is no
principled way to say what a mixture does. The default rule here is a noisy-OR,
``1 - prod(1 - w_i * p_i)``, which reads as "the probability that at least one
component activates this receptor", treating components as independent. That is
semantically coherent with what the scores mean, and it is still a heuristic.
Real odorant mixtures are not independent: competitive binding saturates a
receptor rather than summing, and mixture *suppression*, where one odorant
lowers another's receptor response, is well documented. No combination rule
built on single-molecule scores can express that. ``--rule max`` gives a
conservative alternative that assumes the strongest component dominates.

**Searching.** Given a target profile, greedy forward selection picks the
material that most improves the match, then refines the weights by coordinate
descent. The objective is cosine similarity after receptor-wise standardisation
against the library, because raw cosine is dominated by the shared "general
responsiveness" axis every molecule loads on (~0.95 correlation with the panel
mean) and barely discriminates.

What comes out is a receptor-profile match, not a perfumery formula. The
receptor vectors themselves correlate with perception only weakly
(p = 0.006 on one perfumer's notes), so treat any recipe as a hypothesis to
smell, not a prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.store import Library  # noqa: E402

EPS = 1e-6


def combine(scores: np.ndarray, weights: np.ndarray, rule: str = "noisy-or") -> np.ndarray:
    """Collapse component score rows into one mixture profile."""
    w = np.clip(weights, 0.0, 1.0)[:, None]
    s = np.clip(scores, 0.0, 1.0 - EPS)
    if rule == "noisy-or":
        return 1.0 - np.prod(1.0 - w * s, axis=0)
    if rule == "max":
        return (w * s).max(axis=0)
    if rule == "mean":
        tot = w.sum()
        return (w * s).sum(axis=0) / (tot if tot > 0 else 1.0)
    raise SystemExit(f"unknown rule {rule!r}")


class Space:
    """Comparison space for two receptor profiles.

    ``zscore`` divides by each receptor's spread, which gives every receptor
    equal say -- but a receptor that barely varies across the library then has
    its noise amplified, which is why its residuals can read as tens of sd.
    ``center`` only removes the shared baseline that every molecule loads on,
    leaving naturally variable receptors dominant. ``raw`` does neither and is
    inflated by that shared axis. The recipe should not depend much on this
    choice; if it does, that is worth knowing.
    """

    def __init__(self, scores: np.ndarray, metric: str = "zscore"):
        self.metric = metric
        self.mu = scores.mean(axis=0)
        self.sd = scores.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0

    def z(self, v: np.ndarray) -> np.ndarray:
        if self.metric == "raw":
            return v
        if self.metric == "center":
            return v - self.mu
        return (v - self.mu) / self.sd

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        za, zb = self.z(a), self.z(b)
        na, nb = np.linalg.norm(za), np.linalg.norm(zb)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(za @ zb / (na * nb))


def greedy(target, pool_scores, space, max_n, rule, grid, seed=None):
    """Forward selection; returns (indices, weights, similarity trace)."""
    chosen: list[int] = list(seed[0]) if seed else []
    weights: list[float] = list(seed[1]) if seed else []
    trace = []
    best_sim = (space.sim(target, combine(pool_scores[chosen], np.array(weights), rule))
                if chosen else -2.0)
    while len(chosen) < max_n:
        best = None
        for c in range(pool_scores.shape[0]):
            if c in chosen:
                continue
            for w in grid:
                idx = chosen + [c]
                wt = np.array(weights + [w])
                sim = space.sim(target, combine(pool_scores[idx], wt, rule))
                if best is None or sim > best[0]:
                    best = (sim, c, w)
        if best is None or best[0] <= best_sim + 1e-5:
            break
        best_sim, c, w = best
        chosen.append(c)
        weights.append(w)
        trace.append(best_sim)
    return chosen, np.array(weights), trace


def refine(target, pool_scores, chosen, weights, space, rule, grid, rounds=4):
    """Coordinate descent on the weights of the selected materials."""
    w = weights.copy()
    for _ in range(rounds):
        moved = False
        for k in range(len(chosen)):
            base = w[k]
            best_w, best_s = base, space.sim(target, combine(pool_scores[chosen], w, rule))
            for cand in grid:
                w[k] = cand
                s = space.sim(target, combine(pool_scores[chosen], w, rule))
                if s > best_s + 1e-6:
                    best_s, best_w = s, cand
            w[k] = best_w
            if best_w != base:
                moved = True
        if not moved:
            break
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="library_mpnn_expanded")
    ap.add_argument("--target", required=True,
                    help="comma-separated material names forming the target mixture")
    ap.add_argument("--weights", default=None,
                    help="comma-separated weights for the target (default: equal)")
    ap.add_argument("--rule", default="noisy-or", choices=["noisy-or", "max", "mean"])
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--seed", default=None,
                    help="materials forced into the blend before the search (';' separated)")
    ap.add_argument("--metric", default="center",
                    choices=["center", "zscore", "raw"])
    ap.add_argument("--space", default="receptor", choices=["receptor", "percept"],
                    help="optimise receptor activation, or predicted odour descriptors")
    ap.add_argument("--exclude-target", action="store_true", default=True)
    ap.add_argument("--allow-target", dest="exclude_target", action="store_false")
    args = ap.parse_args()

    entries, _, scores = Library(name=args.library).load()
    names = [e.name for e in entries]
    if args.space == "percept":
        # Optimise predicted smell rather than receptor activation. Justified
        # by the formula check: on the one case with an external verdict,
        # descriptor distance matched the perfumer and receptor distance did not.
        import json as _j

        pj = _j.loads((ROOT / "artifacts" / "percepts_expanded.json").read_text(encoding="utf-8"))
        pm = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
        miss = [n for n in names if n not in pm]
        if miss:
            raise SystemExit(f"no percept prediction for {miss[:3]}")
        scores = np.stack([pm[n] for n in names])
        print(f"[space] optimising {scores.shape[1]} odour descriptors")
    lower = {n.lower(): i for i, n in enumerate(names)}
    space = Space(scores, args.metric)

    # Material names contain commas ("Ethyl Acetate, 10% DPG"), so ";" wins
    # as the separator whenever it appears.
    sep = ";" if ";" in args.target else ","
    want = [t.strip() for t in args.target.split(sep) if t.strip()]
    tgt_idx = []
    for t in want:
        i = lower.get(t.lower())
        if i is None:
            cand = [n for n in names if t.lower() in n.lower()]
            if len(cand) == 1:
                i = lower[cand[0].lower()]
            else:
                raise SystemExit(f"cannot resolve {t!r}" + (f"; did you mean {cand[:4]}" if cand else ""))
        tgt_idx.append(i)

    tw = (np.array([float(x) for x in args.weights.replace(";", ",").split(",")])
          if args.weights else np.ones(len(tgt_idx)))
    tw = tw / tw.max()
    target = combine(scores[tgt_idx], tw, args.rule)

    print(f"TARGET MIXTURE ({args.rule}, {len(tgt_idx)} components)")
    for i, w in zip(tgt_idx, tw):
        print(f"   {w:5.2f}  {names[i]}")
    print(f"   profile: max {target.max():.3f}, "
          f"receptors>0.1 {(target > 0.1).sum()}, sum {target.sum():.1f}")

    banned = set(tgt_idx) if args.exclude_target else set()
    pool = [i for i in range(len(names)) if i not in banned]
    pool_scores = scores[pool]

    # Baseline: the closest single material, for context.
    singles = sorted(
        ((space.sim(target, scores[i]), i) for i in pool), reverse=True)[:5]
    print("\nCLOSEST SINGLE MATERIALS (nothing mixed)")
    for s, i in singles:
        print(f"   {s:+.4f}  {names[i]}")

    grid = [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    seeded, seed_w = [], []
    if args.seed:
        for t in args.seed.split(";"):
            t = t.strip()
            if not t:
                continue
            gi = lower.get(t.lower())
            if gi is None:
                cand = [n for n in names if t.lower() in n.lower()]
                if len(cand) != 1:
                    raise SystemExit(f"cannot resolve seed {t!r}; candidates {cand[:4]}")
                gi = lower[cand[0].lower()]
            if gi in banned:
                raise SystemExit(f"seed {names[gi]!r} is part of the target")
            seeded.append(pool.index(gi))
            seed_w.append(1.0)
        print("\nSEEDED WITH: " + ", ".join(names[pool[c]] for c in seeded))
    chosen, w, trace = greedy(target, pool_scores, space, args.max_n, args.rule, grid,
                              seed=(seeded, seed_w))
    w = refine(target, pool_scores, chosen, w, space, args.rule, grid)
    final = space.sim(target, combine(pool_scores[chosen], w, args.rule))

    print(f"\nBEST BLEND FOUND  (cosine after receptor standardisation: {final:+.4f})")
    order = np.argsort(-w)
    for k in order:
        gi = pool[chosen[k]]
        print(f"   {w[k]:5.2f}  {names[gi]}")
    print("\n   similarity as components were added:")
    print("   " + "  ".join(f"{i+1}:{t:+.3f}" for i, t in enumerate(trace)))

    mix = combine(pool_scores[chosen], w, args.rule)

    # Perception check. The percept model scores single molecules, so the same
    # noisy-OR is applied to its descriptor probabilities -- the same heuristic,
    # with the same caveats, but in a space that can be read as smell.
    import json as _json

    pfile = ROOT / "artifacts" / "percepts_expanded.json"
    if pfile.is_file():
        pj = _json.loads(pfile.read_text(encoding="utf-8"))
        desc = pj["descriptors"]
        pmap = {n: np.asarray(r, dtype=float) for n, r in zip(pj["molecules"], pj["probs"])}
        if all(names[i] in pmap for i in tgt_idx):
            tp = combine(np.stack([pmap[names[i]] for i in tgt_idx]), tw, args.rule)
            bp = combine(np.stack([pmap[names[pool[c]]] for c in chosen]), w, args.rule)
            print("\n   PREDICTED ODOUR  (noisy-OR over the percept model)")
            print(f"     {'descriptor':16s} {'target':>8s} {'blend':>8s}")
            for j in np.argsort(-tp)[:10]:
                print(f"     {desc[j]:16s} {tp[j]:8.2f} {bp[j]:8.2f}")
            only_blend = [j for j in np.argsort(-bp)[:10] if j not in set(np.argsort(-tp)[:10])]
            if only_blend:
                print("     blend adds: " + ", ".join(
                    f"{desc[j]} ({bp[j]:.2f})" for j in only_blend[:5]))
            ca = tp - tp.mean(); cb = bp - bp.mean()
            print(f"     descriptor-space cosine: "
                  f"{float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))):+.4f}")

    resid = space.z(mix) - space.z(target)
    print("\n   worst receptor mismatches (blend minus target, standardised)")
    import csv as _csv

    rec = list(_csv.DictReader(
        open(ROOT / "artifacts" / "receptors.tsv", encoding="utf-8"), delimiter="\t"))
    for j in np.argsort(-np.abs(resid))[:6]:
        print(f"     {rec[j]['gene']:9s} target {target[j]:.3f}  blend {mix[j]:.3f}"
              f"  ({resid[j]:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
