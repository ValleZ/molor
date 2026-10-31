"""Blends whose smell would tell the combination rules apart. A bench protocol.

The rules cannot be chosen from the data in this repository. Scored on the 19
oils whose own name is in the model's vocabulary, all five -- noisy-OR,
weighted mean, logit mean, max, and simply taking the biggest ingredient --
land on a median rank of 2 or 3 out of 138 and are statistically
indistinguishable (``rule_bakeoff.py``). The reason is diagnostic rather than
reassuring: if *dominant* ties with everything, the test set has no mixture
structure to speak of. Those oils are one or two constituents plus decoration,
so every rule returns the same answer and the test has no power.

What has power is a blend the rules *disagree* about, and finding none at
first was informative. The search was originally set up as noisy-OR against
the weighted mean, on the assumption that a union lets a minor ingredient
through at close to full strength. It does not, in this implementation: the
rule scales each ingredient by ``w / w.max()``, so an ingredient at 3% of a
four-material blend enters at about 9% of its own probability, not 100%. Across
40,000 random blends there was not one where noisy-OR carried a minority
descriptor the mean buried. That is worth stating plainly -- noisy-OR here is a
weighted union, and it is no more trace-sensitive than a mean.

The rule that is trace-sensitive is ``max``, which ignores weights entirely. So
the dispute worth taking to the bench is max against the weighted mean, and it
is exactly the question every trace-material failure in this project turns on:
does three percent of something with a strong character impose that character
on the blend, or not? Nootkatone is 0.1% of grapefruit oil and is the whole
grapefruit note; Calone was 0.47% of a formula built around its aquatic note
and the model saw nothing. Make such a blend, smell it, and the disagreement
resolves itself -- one descriptor is there or it is not.

This searches the owned shelf for those blends and writes a sheet to make. The
selection is deliberately conservative:

* Only materials the perfumer stocks, so the sheet can be made tonight.
* Weights are real proportions, with one ingredient held to a few percent --
  that minority ingredient is the whole experiment.
* A blend is kept only when the rules' top lift descriptors differ *and* the
  descriptor in dispute is one the other rule actively ranks low, so the
  question put to the nose is "do you smell X, yes or no" rather than a
  judgement between two shades of the same thing.
* The disputed descriptor must be a word a perfumer would use unprompted.

What this still cannot settle: the model's own error. If a blend's disputed
descriptor is absent on the bench, that may mean the union rule is wrong, or
that the percept vector for the minority ingredient is wrong. The design
reduces the second risk by preferring minority ingredients that are in the
percept model's training set, where its vector is partly recall of a human
label rather than a generalisation.

    python scripts/discriminating_blends.py --count 8
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

from eval_mixtures import combine  # noqa: E402

#: Words a perfumer would use at the bench without being coached. The model's
#: vocabulary also holds terms like "ketonic" and "weedy", which are not
#: questions a nose can answer cleanly.
USABLE = {
    "green", "herbal", "minty", "fresh", "citrus", "lemon", "orange",
    "grapefruit", "bergamot", "floral", "rose", "jasmin", "lavender",
    "violet", "powdery", "sweet", "vanilla", "caramellic", "coconut",
    "creamy", "fruity", "apple", "pear", "banana", "berry", "melon",
    "tropical", "woody", "cedar", "sandalwood", "vetiver", "earthy",
    "balsamic", "amber", "musk", "animal", "leathery", "smoky", "phenolic",
    "spicy", "cinnamon", "clove", "anisic", "almond", "nutty", "buttery",
    "cheesy", "fatty", "waxy", "metallic", "sulfurous", "medicinal",
    "camphoreous", "pine", "mushroom", "tobacco", "honey", "winey", "rummy",
}


def owned_use_levels() -> dict[str, float]:
    """Each stocked material's own maximum use level, in percent.

    The search has no potency axis, so without this it will happily propose a
    blend that is a third dimethyl sulfide -- proportions the model finds
    unremarkable and a nose finds unusable. The perfumer's own column is the
    best available guard: a material they never take above 0.1% does not
    belong in a majority slot, whatever the vector says. Materials with no
    recorded level are admitted only as the minority ingredient.
    """
    path = ROOT / "data" / "inventory" / "inventory.csv"
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or r.get("material") or "").strip()
        if not n:
            continue
        v = (r.get("top_pct") or "").strip()
        try:
            out[n] = float(v)
        except ValueError:
            out[n] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--draws", type=int, default=40000)
    ap.add_argument("--minor", type=float, default=0.03,
                    help="weight of the ingredient the rules argue about")
    ap.add_argument("--null", type=int, default=600)
    ap.add_argument("--bulk-use", type=float, default=5.0,
                    help="a majority ingredient must be one the perfumer uses "
                         "at least this high, in percent")
    ap.add_argument("--claims", default="max",
                    choices=["noisy-or", "weighted-mean", "max"],
                    help="the rule that says the minority descriptor is there")
    ap.add_argument("--denies", default="weighted-mean",
                    choices=["noisy-or", "weighted-mean", "max"],
                    help="the rule that says it is not")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/formulas/queries/rule_test")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    names = list(pj["molecules"])
    LM = np.asarray(pj["probs"], float)
    usable = [i for i, d in enumerate(desc) if d in USABLE]

    use = owned_use_levels()
    import math
    bulk = [i for i, n in enumerate(names)
            if n in use and not math.isnan(use[n]) and use[n] >= args.bulk_use]
    minor = [i for i, n in enumerate(names) if n in use]
    print(f"[rules] searching {args.draws} blends of {args.size}: "
          f"{len(bulk)} materials usable at {100 / (args.size - 1):.0f}% as "
          f"bulk (use level >= {args.bulk_use}%), {len(minor)} usable as the "
          f"minority ingredient")
    if len(bulk) < args.size:
        raise SystemExit("too few materials with a high enough use level")

    rng = np.random.default_rng(args.seed)
    # One null per weight vector, and the weight vector is fixed by design, so
    # the null is computed once and reused -- which also means every candidate
    # is standardised identically.
    w = np.full(args.size, (1.0 - args.minor) / (args.size - 1))
    w[-1] = args.minor
    nulls = {}
    for rule in ("noisy-or", "weighted-mean", "max"):
        n = np.stack([combine(rule, LM[rng.choice(len(LM), args.size,
                                                  replace=False)], w)
                      for _ in range(args.null)])
        nulls[rule] = (n.mean(0), np.maximum(n.std(0), 1e-9))

    found = []
    for _ in range(args.draws):
        pick = np.concatenate([
            rng.choice(bulk, args.size - 1, replace=False),
            rng.choice(minor, 1)])
        if len(set(pick.tolist())) != args.size:
            continue
        V = LM[pick]
        z = {}
        for rule in ("noisy-or", "weighted-mean", "max"):
            mu, sd = nulls[rule]
            z[rule] = (combine(rule, V, w) - mu) / sd
        # The dispute: a usable descriptor the union rule puts near the top and
        # the mean rule does not rate at all.
        cand = [i for i in usable
                if z[args.claims][i] >= 3.0 and z[args.denies][i] <= 0.75]
        if not cand:
            continue
        i = max(cand, key=lambda j: z[args.claims][j] - z[args.denies][j])
        gap = float(z[args.claims][i] - z[args.denies][i])
        # The minority ingredient has to be the reason, or the blend tests
        # nothing: its own probability for the disputed descriptor must lead.
        if int(np.argmax(V[:, i])) != args.size - 1:
            continue
        found.append({"gap": gap, "desc": desc[i], "pick": pick.tolist(),
                      "z_or": float(z["noisy-or"][i]),
                      "z_mean": float(z["weighted-mean"][i]),
                      "z_max": float(z["max"][i]),
                      "p_minor": float(V[-1, i])})

    found.sort(key=lambda r: -r["gap"])
    # One blend per disputed descriptor, so the evening tests several claims
    # rather than one claim eight times.
    seen, keep = set(), []
    for r in found:
        if r["desc"] in seen:
            continue
        seen.add(r["desc"])
        keep.append(r)
        if len(keep) >= args.count:
            break

    print(f"[rules] {len(found)} candidate blends, {len(keep)} kept, one per "
          f"disputed descriptor\n")
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(keep, 1):
        ing = [names[j] for j in r["pick"]]
        print(f"  blend {k}: is it {r['desc'].upper()}?   "
              f"max says z {r['z_max']:+.1f}, weighted mean says "
              f"{r['z_mean']:+.1f}, noisy-OR says {r['z_or']:+.1f}")
        for n_, wt in zip(ing, w):
            lvl = use.get(n_)
            tag = ("  <- the ingredient in dispute" if n_ == ing[-1]
                   else "")
            lv = ("" if lvl is None or lvl != lvl
                  else f"   [you use it up to {lvl:g}%]")
            print(f"      {100 * wt:5.1f}%  {n_}{tag}{lv}")
        path = outdir / f"blend{k}_{r['desc']}.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            cw = csv.writer(fh)
            cw.writerow(["material", "dilution", "weight"])
            for n_, wt in zip(ing, w):
                cw.writerow([n_, 100, round(100 * wt, 2)])
        print(f"      -> {path.relative_to(ROOT)}")
    (outdir / "predictions.json").write_text(json.dumps(
        {"size": args.size, "minor_weight": args.minor,
         "blends": [{"n": k, "disputed": r["desc"],
                     "ingredients": [names[j] for j in r["pick"]],
                     "z_noisy_or": round(r["z_or"], 2),
                     "z_weighted_mean": round(r["z_mean"], 2),
                     "z_max": round(r["z_max"], 2)}
                    for k, r in enumerate(keep, 1)]}, indent=2), encoding="utf-8")
    print(f"\n[rules] predictions recorded in {args.out}/predictions.json "
          f"before anything is smelled, which is the point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
