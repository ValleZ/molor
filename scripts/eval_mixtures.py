"""Does combining percept vectors predict what a mixture smells like?

This has never been tested. The accord work combined vectors with noisy-OR and
weights that were invented, produced suggestions, and those suggestions failed
their blind test. But the blind test judged the *suggestions*, not the
combination rule, so the rule itself has never been checked against a mixture
anyone has smelled.

The perfumer's formula records allow that check: each one pairs a composition
with a written verdict on the result. The verdict's descriptors are the target,
and the question is whether a predicted mixture vector ranks them above the 130
or so descriptors the perfumer did not mention.

Two baselines decide whether any of this is worth doing, and both are more
demanding than chance.

*The base-rate prior.* Predicting GS-LF label frequencies while ignoring the
formula entirely scores about 0.84 AUC against the perfumer's notes on single
materials, because odour labels are unbalanced and people mention common
qualities. Any mixture rule has to beat that.

*The dominant ingredient.* Using only the largest ingredient's vector and
discarding the rest is the null hypothesis for combination specifically. If
noisy-OR does no better, then combining adds nothing and the accord machinery
should be retired, whatever its absolute score.

Four rules are compared. Noisy-OR is what the accord work used. A
fraction-weighted mean is the obvious linear alternative. Max is what one would
use if the strongest contributor of each quality wins outright. Dominant is the
baseline above.
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
sys.path.insert(0, str(ROOT / "scripts"))

from check_impressions import VOCAB  # noqa: E402
from notes_vs_models import DROPPED, TAG_TO_GSLF, auc_within, base_rates  # noqa: E402

PARSED = ROOT / "data" / "formulas" / "parsed.json"
EPS = 1e-6


def verdict_descriptors(text: str, valid: set[str]) -> set[str]:
    low = text.lower()
    tags = {d for d, keys in VOCAB.items() if any(k in low for k in keys)}
    tags -= DROPPED
    return {TAG_TO_GSLF.get(t, t) for t in tags} & valid


def combine(rule: str, V: np.ndarray, w: np.ndarray) -> np.ndarray:
    """V: (n_ingredients, n_descriptors) probabilities. w: fractions summing to 1."""
    if rule == "noisy-or":
        # The accord work's rule: each ingredient independently fails to deliver
        # the quality, scaled by its share.
        ww = np.clip(w / w.max(), 0, 1)[:, None]
        return 1.0 - np.prod(1.0 - ww * np.clip(V, 0, 1 - EPS), axis=0)
    if rule == "weighted-mean":
        return (w[:, None] * V).sum(axis=0)
    if rule == "max":
        return V.max(axis=0)
    if rule == "dominant":
        return V[int(np.argmax(w))]
    raise ValueError(rule)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--min-descriptors", type=int, default=2,
                    help="a verdict must name at least this many descriptors")
    ap.add_argument("--out", default="artifacts/mixture_eval.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    idx = {d: i for i, d in enumerate(desc)}
    P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
    prior = base_rates(desc)

    parsed = json.loads(PARSED.read_text(encoding="utf-8"))["usable"]
    rules = ["noisy-or", "weighted-mean", "max", "dominant"]
    rows, scores = [], {r: [] for r in rules}
    prior_scores = []

    for f in parsed:
        tags = verdict_descriptors(f["verdict"], set(idx))
        if len(tags) < args.min_descriptors:
            continue
        mats = [(r["name"], r["fraction"]) for r in f["ingredients"]
                if r["kind"] == "material" and r["name"] in P
                and r.get("fraction", 0) > 0]
        if len(mats) < 3:
            continue
        V = np.stack([P[n] for n, _ in mats])
        w = np.array([x for _, x in mats], float)
        w = w / w.sum()
        mask = np.zeros(len(desc), dtype=bool)
        for t in tags:
            mask[idx[t]] = True

        row = {"formula": f["name"], "n_ingredients": len(mats),
               "descriptors": sorted(tags)}
        for rule in rules:
            a = auc_within(combine(rule, V, w), mask)
            if a is None:
                break
            row[rule] = round(a, 4)
            scores[rule].append(a)
        pa = auc_within(prior, mask)
        row["base-rate"] = round(pa, 4)
        prior_scores.append(pa)
        rows.append(row)

    if not rows:
        raise SystemExit("no formula had both a resolvable composition and a "
                         "verdict naming enough descriptors")

    print(f"[mix] {len(rows)} formulas with a resolvable composition and a "
          f"verdict naming >={args.min_descriptors} descriptors\n")
    print(f"  {'formula':32s} {'ing':>3s} " + " ".join(f"{r:>13s}" for r in rules)
          + f" {'base-rate':>10s}  descriptors")
    print("  " + "-" * 118)
    for r in rows:
        print(f"  {r['formula'][:32]:32s} {r['n_ingredients']:3d} "
              + " ".join(f"{r[k]:13.3f}" for k in rules)
              + f" {r['base-rate']:10.3f}  {', '.join(r['descriptors'])[:34]}")

    print(f"\n  {'rule':16s} {'mean AUC':>9s} {'vs base-rate':>13s} "
          f"{'vs dominant':>12s}")
    base = float(np.mean(prior_scores))
    dom = float(np.mean(scores["dominant"]))
    summary = {"n": len(rows), "base_rate_auc": round(base, 4)}
    for rule in rules:
        m = float(np.mean(scores[rule]))
        summary[rule] = {"auc": round(m, 4), "vs_base_rate": round(m - base, 4),
                         "vs_dominant": round(m - dom, 4)}
        print(f"  {rule:16s} {m:9.4f} {m - base:+13.4f} {m - dom:+12.4f}")
    print(f"  {'base rate':16s} {base:9.4f}")

    # Paired bootstrap over formulas for the comparison that matters.
    rng = np.random.default_rng(0)
    print("\n  paired bootstrap over formulas, 95% CI:")
    for rule in ("noisy-or", "weighted-mean", "max"):
        a = np.array(scores[rule]); b = np.array(scores["dominant"])
        d = [np.mean(a[i] - b[i]) for i in
             (rng.integers(0, len(a), len(a)) for _ in range(4000))]
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"    {rule:14s} minus dominant  {np.mean(d):+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {sig}")
        summary[rule]["ci_vs_dominant"] = [round(float(lo), 4),
                                           round(float(hi), 4)]
    for rule in rules:
        a = np.array(scores[rule]); b = np.array(prior_scores)
        d = [np.mean(a[i] - b[i]) for i in
             (rng.integers(0, len(a), len(a)) for _ in range(4000))]
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"    {rule:14s} minus base rate {np.mean(d):+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {sig}")
        summary[rule]["ci_vs_base_rate"] = [round(float(lo), 4),
                                            round(float(hi), 4)]

    (ROOT / args.out).write_text(json.dumps(
        {"summary": summary, "per_formula": rows}, indent=2), encoding="utf-8")
    print(f"\n[mix] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
