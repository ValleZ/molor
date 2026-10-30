"""Draw a blind test set spanning the model's predicted range.

Four unblinded trials have matched the model's ordering, but the perfumer knew
the rankings before smelling, so expectation cannot be ruled out. This draws
candidates from across the predicted range -- some strongly predicted to help,
some middling, some predicted to do nothing or to hurt -- and prints them in
shuffled order with no scores.

The key is written to a file and never printed, so it does not appear in the
terminal the perfumer is reading.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6

TARGET = ["(Z)-3-Hexenal", "(E)-2-Hexenal", "Methyl Butyrate",
          "Ethyl Acetate, 10% DPG", "Ethyl Butyrate", "Hexenol-3-CIS",
          "Methyl Acetate", "Heptan-2-one"]
WEIGHTINGS = {
    "equal": [1] * 8,
    "mild": [2.0, 0.245, 0.130, 0.014, 1.0, 0.118, 0.022, 0.084],
    "strong": [4.0, 0.06, 0.017, 0.0002, 1.0, 0.014, 0.0005, 0.007],
}
FORMULA = [
    ("Hexenyl-3-CIS Acetate", 300), ("Paradisamide", 200), ("Ethyl Linalool", 200),
    ("Cassifix", 80), ("Verdox HC", 70), ("Fructone", 50), ("Hexenol-3-CIS", 35),
    ("Blackcurrant mercaptan, 0.1% DPG", 0.03), ("Ambrettolide", 25),
    ("Ethyl 2-methyl butyrate", 1.5),
]
#: Already tried, so they carry expectations and cannot be part of a blind test.
BURNED = {"Allyl Amyl Glycolate", "Trans-2,Cis-6-Nonadienal, 1% DPG",
          "Dimethyl Sulfide, 1% DPG", "Thiocineole, 0.1% DPG",
          "Isoamyl Acetate", "Manzanate", "Trans-4-Decenal, 10% DPG"}


def nor(rows, w):
    w = np.asarray(w, float)
    w = np.clip(w / w.max(), 0, 1)[:, None]
    return 1.0 - np.prod(1.0 - w * np.clip(rows, 0, 1 - EPS), axis=0)


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261029
    owned = {r["name"] for r in csv.DictReader(
        open(ROOT / "library/input/expanded_materials.csv", encoding="utf-8"))
        if r.get("source") == "curated"}

    scores, top5 = {}, {}
    runs = 0
    for pf in ("percepts_expanded.json", "percepts_openpom.json"):
        path = ROOT / "artifacts" / pf
        if not path.is_file():
            continue
        pj = json.loads(path.read_text(encoding="utf-8"))
        P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
        mu = np.stack(list(P.values())).mean(axis=0)

        def cos(a, b):
            return float((a - mu) @ (b - mu)
                         / (np.linalg.norm(a - mu) * np.linalg.norm(b - mu)))

        in_f = {n for n, _ in FORMULA}
        pool = sorted(n for n in owned - in_f - BURNED - set(TARGET) if n in P)
        br = np.stack([P[n] for n, _ in FORMULA])
        bw = np.array([w for _, w in FORMULA])
        for tw in WEIGHTINGS.values():
            runs += 1
            tgt = nor(np.stack([P[n] for n in TARGET]), tw)
            base = cos(nor(br, bw), tgt)
            gains = []
            for c in pool:
                g = max(cos(nor(np.vstack([br, P[c][None]]),
                               np.append(bw, w * bw.max())), tgt) - base
                        for w in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0))
                gains.append((g, c))
                scores.setdefault(c, []).append(g)
            gains.sort(reverse=True)
            for _, c in gains[:5]:
                top5[c] = top5.get(c, 0) + 1

    ranked = sorted(scores, key=lambda k: (-top5.get(k, 0), -np.median(scores[k])))
    n = len(ranked)
    picks = [ranked[0], ranked[1],                 # strongly predicted
             ranked[n // 2 - 1], ranked[n // 2],   # middling
             ranked[-2], ranked[-1]]               # predicted nothing or worse
    picks = list(dict.fromkeys(picks))

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(picks)))
    shuffled = [picks[i] for i in order]

    key = {
        "seed": seed,
        "runs": runs,
        "materials": {
            c: {"median_gain": round(float(np.median(scores[c])), 5),
                "top5_of_runs": top5.get(c, 0),
                "rank_of": f"{ranked.index(c) + 1}/{n}"}
            for c in picks
        },
        "presented_order": shuffled,
    }
    out = ROOT / "artifacts" / "blind_test_key.json"
    out.write_text(json.dumps(key, indent=2), encoding="utf-8")

    print("BLIND TEST SET - rank these on 'makes the accord more gooseberry'")
    print(f"({len(shuffled)} materials you own; predictions are hidden)\n")
    for i, c in enumerate(shuffled, 1):
        print(f"  {chr(64 + i)}.  {c}")
    print(f"\nkey written to {out.relative_to(ROOT)} - do not open it until you have ranked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
