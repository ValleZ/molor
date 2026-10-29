"""Rank additions to the gooseberry formula, keeping only stable ones.

The weighting sweep showed that one target produces three largely disjoint
shortlists depending on odour-activity weights that were guessed rather than
measured. So a single ranking is not trustworthy. This scores every candidate
under both percept models and all three weightings -- six runs -- and reports
how often each lands in the top five. Only picks that survive most of that are
worth acting on.

Candidates are restricted to materials the perfumer already owns (the curated
128), minus anything already in the formula.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TARGET = ["(Z)-3-Hexenal", "(E)-2-Hexenal", "Methyl Butyrate",
          "Ethyl Acetate, 10% DPG", "Ethyl Butyrate", "Hexenol-3-CIS",
          "Methyl Acetate", "Heptan-2-one"]

WEIGHTINGS = {
    "equal": [1, 1, 1, 1, 1, 1, 1, 1],
    "mild": [2.0, 0.245, 0.130, 0.014, 1.0, 0.118, 0.022, 0.084],
    "strong": [4.0, 0.06, 0.017, 0.0002, 1.0, 0.014, 0.0005, 0.007],
}

# The formula as it stands, with dilutions resolved to neat parts.
FORMULA = [
    ("Hexenyl-3-CIS Acetate", 300), ("Paradisamide", 200), ("Ethyl Linalool", 200),
    ("Cassifix", 80), ("Verdox HC", 70), ("Fructone", 50), ("Hexenol-3-CIS", 35),
    ("Blackcurrant mercaptan, 0.1% DPG", 0.03), ("Ambrettolide", 25),
    ("Ethyl 2-methyl butyrate", 1.5),
]

ASKED_ABOUT = ["Ethyl Acetoacetate", "Hexyl Caprylate", "Allyl Amyl Glycolate",
               "Trans-2,Cis-6-Nonadienal, 1% DPG"]
EPS = 1e-6


def nor(rows, w):
    w = np.asarray(w, float)
    w = np.clip(w / w.max(), 0, 1)[:, None]
    return 1.0 - np.prod(1.0 - w * np.clip(rows, 0, 1 - EPS), axis=0)


def main() -> int:
    owned = set()
    for r in csv.DictReader(open(ROOT / "library/input/expanded_materials.csv",
                                 encoding="utf-8")):
        if r.get("source") == "curated":
            owned.add(r["name"])
    print(f"materials you own: {len(owned)}")

    results = defaultdict(list)
    top5 = defaultdict(int)
    desc_of = {}
    runs = 0

    for mname, pfile in (("GS-LF", "percepts_expanded.json"),
                         ("OpenPOM", "percepts_openpom.json")):
        path = ROOT / "artifacts" / pfile
        if not path.is_file():
            continue
        pj = json.loads(path.read_text(encoding="utf-8"))
        desc = pj["descriptors"]
        P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
        mu = np.stack(list(P.values())).mean(axis=0)

        def cos(a, b):
            ca, cb = a - mu, b - mu
            return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))

        in_formula = {n for n, _ in FORMULA}
        pool = sorted((owned | set(ASKED_ABOUT)) - in_formula - set(TARGET))
        pool = [p for p in pool if p in P]
        base_rows = np.stack([P[n] for n, _ in FORMULA])
        base_w = np.array([w for _, w in FORMULA])

        for wname, tw in WEIGHTINGS.items():
            runs += 1
            tgt = nor(np.stack([P[n] for n in TARGET]), tw)
            base = cos(nor(base_rows, base_w), tgt)
            gains = []
            for cand in pool:
                best = max(
                    cos(nor(np.vstack([base_rows, P[cand][None]]),
                            np.append(base_w, w * base_w.max())), tgt)
                    for w in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
                )
                gains.append((best - base, cand))
                desc_of.setdefault(cand, {})[mname] = ", ".join(
                    f"{desc[j]}({P[cand][j]:.2f})" for j in np.argsort(-P[cand])[:3])
            gains.sort(reverse=True)
            for g, c in gains:
                results[c].append(g)
            for _, c in gains[:5]:
                top5[c] += 1

    print(f"{runs} runs (2 percept models x 3 weightings)\n")
    print("THE TWO YOU ASKED ABOUT")
    for c in ASKED_ABOUT:
        if c not in results:
            print(f"  {c}: not in the library")
            continue
        g = results[c]
        own = "you own it" if c in owned else "NOT in your 128 - would need buying"
        rank = sorted(results, key=lambda k: -np.median(results[k])).index(c) + 1
        print(f"  {c}  ({own})")
        print(f"    median gain {np.median(g):+.4f}   rank {rank}/{len(results)}   "
              f"top-5 in {top5[c]}/{runs} runs")
        for m, d in desc_of.get(c, {}).items():
            print(f"    {m:8s} {d}")

    print("\nSTABLE PICKS FROM WHAT YOU ALREADY OWN")
    print(f"  {'material':34s} {'median gain':>12s} {'worst':>9s} {'top5':>6s}")
    ranked = sorted(results, key=lambda k: (-top5[k], -np.median(results[k])))
    shown = 0
    for c in ranked:
        if c in ASKED_ABOUT and c not in owned:
            continue
        g = results[c]
        if top5[c] == 0:
            continue
        print(f"  {c[:33]:34s} {np.median(g):+12.4f} {min(g):+9.4f} {top5[c]:3d}/{runs}")
        for m, d in desc_of.get(c, {}).items():
            print(f"      {m:8s} {d}")
        shown += 1
        if shown >= 7:
            break
    if not shown:
        print("  nothing reached the top five in any run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
