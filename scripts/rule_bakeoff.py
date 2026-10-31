"""Which mixture rule is right? Decide it on the oils, which have ground truth.

Five rules combine ingredient vectors into a blend vector, and until now the
choice was made on shape rather than on evidence. The earlier attempt compared
them against the perfumer's written verdicts on 30 of their own formulas, where
noisy-OR won but the margin depended on four closely related formulas being
included -- an n of 30 with a fragile margin.

The essential oils are a better test set and were sitting unused. Each one is a
mixture whose composition is published, and 19 of them have their own name in
the percept model's 138-word vocabulary: bergamot, lavender, geranium, mint,
clove, cinnamon, lemon, vetiver, grapefruit, orangeflower. So each rule can be
asked the same blind question -- build this oil from its constituents and say
what it smells of -- and scored on where the oil's own name lands. No rule is
fitted to anything here; the compositions and the labels both come from outside
the model.

Two details make the comparison fair.

*The null uses the rule under test.* Raw descriptor values are dominated by
GS-LF label frequency, so every rule is standardised against random blends of
the same size and weights combined by that same rule. Otherwise the comparison
would measure how much each rule inflates common labels.

*Rank, not z.* The rules produce differently scaled z-distributions, so the
score is the rank of the oil's own descriptor among 138, which is comparable
across rules and is also what a reader of the atlas actually consumes.

    python scripts/rule_bakeoff.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_naturals import self_word  # noqa: E402
from eval_mixtures import combine  # noqa: E402

RULES = ["noisy-or", "weighted-mean", "logit-mean", "max", "dominant"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--null", type=int, default=2000)
    ap.add_argument("--out", default="artifacts/rule_bakeoff.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    names = list(pj["molecules"])
    P = {n: np.asarray(r, float) for n, r in zip(names, pj["probs"])}
    LM = np.stack([P[n] for n in names])

    spec = json.loads((ROOT / "data" / "naturals" / "compositions.json")
                      .read_text(encoding="utf-8"))["naturals"]

    cases = []
    for oil, meta in spec.items():
        word = self_word(oil)
        if not word or word not in desc:
            continue
        res = [(n, v) for n, v in meta["composition"].items() if n in P]
        if len(res) < 2:
            continue
        w = np.array([v for _, v in res], float)
        cases.append((oil, word, np.stack([P[n] for n, _ in res]), w / w.sum()))
    print(f"[bakeoff] {len(cases)} oils have a testable name and a usable "
          f"composition\n")

    results = {r: [] for r in RULES}
    for oil, word, V, w in cases:
        i = desc.index(word)
        line = f"  {oil[:34]:34s} {word:12s}"
        for rule in RULES:
            rng = np.random.default_rng(0)      # same draws for every rule
            mix = combine(rule, V, w)
            null = np.stack([
                combine(rule, LM[rng.choice(len(LM), len(w), replace=False)], w)
                for _ in range(args.null)])
            z = (mix - null.mean(0)) / np.maximum(null.std(0), 1e-9)
            rank = int(np.argsort(np.argsort(-z))[i]) + 1
            results[rule].append(rank)
            line += f" {rule[:4]}:{rank:>4d}"
        print(line)

    print(f"\n[bakeoff] rank of each oil's own name among {len(desc)} "
          f"descriptors, lower is better\n")
    print(f"  {'rule':14s} {'median':>7s} {'mean':>7s} {'top-5':>7s} "
          f"{'top-20':>7s} {'worst':>7s}")
    print("  " + "-" * 56)
    table = {}
    for rule in RULES:
        r = np.array(results[rule], float)
        table[rule] = {"median": float(np.median(r)), "mean": float(r.mean()),
                       "top5": int((r <= 5).sum()), "top20": int((r <= 20).sum()),
                       "worst": int(r.max()), "ranks": results[rule]}
        print(f"  {rule:14s} {np.median(r):7.1f} {r.mean():7.1f} "
              f"{(r <= 5).sum():5d}/{len(r)} {(r <= 20).sum():5d}/{len(r)} "
              f"{int(r.max()):7d}")
    print(f"\n  chance would put a name at rank {len(desc) / 2:.0f} on average, "
          f"and in the top 5 about {5 / len(desc) * len(cases):.1f} times")

    # Paired comparison against the best rule: with 19 oils, a difference in
    # means says little on its own, so the sign test on paired ranks is
    # reported too. It asks only how often one rule beats another, which is
    # what survives this sample size.
    best = min(RULES, key=lambda r: table[r]["median"])
    print(f"\n[bakeoff] paired against {best}, oil by oil:")
    from scipy.stats import binomtest, wilcoxon
    for rule in RULES:
        if rule == best:
            continue
        a = np.array(results[best], float)
        b = np.array(results[rule], float)
        wins = int((a < b).sum())
        ties = int((a == b).sum())
        n = len(a) - ties
        p = binomtest(wins, n, 0.5).pvalue if n else 1.0
        try:
            pw = float(wilcoxon(a, b).pvalue) if n else 1.0
        except ValueError:
            pw = 1.0
        print(f"  vs {rule:14s} {best} wins {wins}/{n} (ties {ties})  "
              f"sign p {p:.3f}  wilcoxon p {pw:.3f}")
        table[rule]["lost_to_best"] = wins
        table[rule]["p_sign"] = float(p)

    (ROOT / args.out).write_text(json.dumps(
        {"oils": [c[0] for c in cases], "words": [c[1] for c in cases],
         "best_by_median": best, "rules": table}, indent=2), encoding="utf-8")
    print(f"\n[bakeoff] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
