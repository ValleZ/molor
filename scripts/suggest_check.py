"""Test proposed additions against a formula, in predicted-odour space.

For each candidate material, add it to the base formula at several weights and
report the best improvement in descriptor cosine to the target. This answers
"would this help?" rather than "what is this material like?".

Caveats that apply to every number below. The combination rule is a noisy-OR
over single-molecule predictions, so it cannot represent mixture suppression.
There is no potency or volatility axis anywhere in the model, so the weights
are abstract proportions, not doses -- a material that helps at weight 1.0 here
may be correct at 0.1% on the bench, or ruin the accord at 1%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.store import Library  # noqa: E402

EPS = 1e-6

# Target: gooseberry volatiles weighted by rough odour activity (1/threshold).
# Order-of-magnitude values, used to stop bulk esters dominating trace aldehydes.
TARGET = [
    ("(Z)-3-Hexenal", 4.0), ("Ethyl Butyrate", 1.0), ("(E)-2-Hexenal", 0.06),
    ("Methyl Butyrate", 0.017), ("Hexenol-3-CIS", 0.014), ("Heptan-2-one", 0.007),
    ("Methyl Acetate", 0.0005), ("Ethyl Acetate, 10% DPG", 0.0002),
]

# The formula with the two drydown materials removed, as agreed.
BASE = [
    ("Hexenyl-3-CIS Acetate", 300), ("Paradisamide", 200), ("Ethyl Linalool", 200),
    ("Fructone", 50), ("Hexenol-3-CIS", 35), ("Ambrettolide", 25),
    ("Ethyl 2-methyl butyrate", 1.5), ("Blackcurrant mercaptan, 0.1% DPG", 0.03),
]

SUGGESTED = [
    "Floropal", "Labienoxime", "Methyl Anthranilate", "Ethyl Caprylate",
    "Ethyl Octanoate", "Farnesol", "Geraniol", "D-Limonene", "Citronellol",
    "Benzaldehyde", "Cinnamyl Alcohol", "Ethyl Cinnamate", "Phenyl Propyl Alcohol",
    "Eucalyptol", "Terpineol Alpha", "Alpha Pinene", "Linalool Oxide",
    "Methyl Salicylate", "Borneol", "Lindenol", "Trans-2-Hexenol",
    "Trans-2-Octenal", "Aldehyde C-7", "Cis-3-Hexenyl Butyrate",
]


def nor(rows, w):
    w = np.asarray(w, float)
    w = np.clip(w / w.max(), 0, 1)[:, None]
    return 1.0 - np.prod(1.0 - w * np.clip(rows, 0, 1 - EPS), axis=0)


def main() -> int:
    import sys as _sys

    extra = [a for a in _sys.argv[1:] if not a.startswith("-")]
    if extra:
        SUGGESTED[:] = extra

    entries, _, _ = Library(name="library_mpnn_expanded").load()
    names = [e.name for e in entries]
    low = {n.lower(): n for n in names}

    pj = json.loads((ROOT / "artifacts" / "percepts_expanded.json").read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
    mu = np.stack([P[n] for n in names]).mean(axis=0)

    def cos(a, b):
        ca, cb = a - mu, b - mu
        return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))

    def resolve(q):
        if q.lower() in low:
            return low[q.lower()]
        hits = [n for n in names if q.lower() in n.lower()]
        return hits[0] if len(hits) == 1 else None

    tgt = nor(np.stack([P[n] for n, _ in TARGET]), [w for _, w in TARGET])
    base_rows = np.stack([P[n] for n, _ in BASE])
    base_w = np.array([w for _, w in BASE])
    base = nor(base_rows, base_w)
    base_sim = cos(base, tgt)

    print(f"base formula (Cassifix and Verdox HC removed): {base_sim:+.4f}")
    print(f"target profile: " + ", ".join(
        f"{desc[j]} {tgt[j]:.2f}" for j in np.argsort(-tgt)[:6]))
    print()

    print("WOULD IT HELP?  best descriptor-cosine gain when added")
    print(f"  {'material':30s} {'in lib':>7s} {'weight':>7s} {'new':>8s} {'gain':>8s}  top descriptors")
    rows = []
    for q in SUGGESTED:
        n = resolve(q)
        if n is None:
            rows.append((None, q, None, None, None))
            continue
        best = None
        for w in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            v = nor(np.vstack([base_rows, P[n][None]]), np.append(base_w, w * base_w.max()))
            s = cos(v, tgt)
            if best is None or s > best[0]:
                best = (s, w)
        rows.append((best[0] - base_sim, q, n, best[1], best[0]))
    rows.sort(key=lambda r: (r[0] is None, -(r[0] or 0)))
    for gain, q, n, w, new in rows:
        if gain is None:
            print(f"  {q:30s} {'NOT IN LIBRARY':>7s}")
            continue
        top = ", ".join(f"{desc[j]}({P[n][j]:.2f})" for j in np.argsort(-P[n])[:3])
        flag = "" if gain > 0 else "   (no help)"
        print(f"  {q[:29]:30s} {'yes':>7s} {w:7.2f} {new:+8.4f} {gain:+8.4f}  {top}{flag}")

    print("\nCOMBINED: base plus every suggestion that helps, at its best weight")
    add = [(n, w) for gain, q, n, w, _ in rows if gain and gain > 0]
    if add:
        rows_all = np.vstack([base_rows] + [P[n][None] for n, _ in add])
        w_all = np.append(base_w, [w * base_w.max() for _, w in add])
        v = nor(rows_all, w_all)
        print(f"  {len(add)} additions -> {cos(v, tgt):+.4f} (from {base_sim:+.4f})")
        print("  " + ", ".join(n for n, _ in add))
        print("\n  resulting profile vs target")
        for j in np.argsort(-tgt)[:10]:
            print(f"    {desc[j]:14s} target {tgt[j]:.2f}  blend {v[j]:.2f}  ({v[j]-tgt[j]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
