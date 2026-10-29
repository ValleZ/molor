"""OpenPOM vs the shipped GS-LF baseline, judged against the perfumer's notes.

The two models have different vocabularies (138 vs 152 descriptors), so each is
scored on its own: a descriptor counts as ground truth for a material when its
name appears literally in that material's note. Literal matching is crude --
"MARZIPAN" will not match *almond* -- but it is crude in the same way for both
models, so the comparison is fair even though the absolute numbers understate
both.

Also compares them on the musk problem. If a percept model groups nitro,
macrocyclic and polycyclic musks together, it is a better odour axis than the
receptor space, and a better judge for any future evaluation.
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MUSKS = {
    "nitro": ["Musk Xylene", "Musk Ambrette", "Musk Ketone"],
    "macrocyclic": ["Exaltolide Total", "Ambrettolide", "Habanolide",
                    "Ethylene Brassylate", "Velvione", "Muscone"],
    "polycyclic": ["Galaxolide 50% IPM", "Tonalide", "Celestolide", "Cashmeran"],
}


def load(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    P = np.asarray(d["probs"], dtype=float)
    return d["descriptors"], d["molecules"], P


def main() -> int:
    models = {
        "GS-LF baseline": ROOT / "artifacts" / "percepts_expanded.json",
        "OpenPOM (x10)": ROOT / "artifacts" / "percepts_openpom.json",
    }
    with open(ROOT / "library/input/impressions.tsv", encoding="utf-8") as fh:
        notes = {r["material"]: r["impression"].lower()
                 for r in csv.DictReader(fh, delimiter="\t")}

    print("AGREEMENT WITH YOUR NOTES")
    print("  a descriptor counts as truth when its name appears in the note")
    print(f"  {'model':18s} {'scored':>7s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} "
          f"{'median rank':>12s}")
    store = {}
    for label, path in models.items():
        if not path.is_file():
            print(f"  {label}: missing {path}")
            continue
        desc, mols, P = load(path)
        store[label] = (desc, mols, P)
        pos = {m: i for i, m in enumerate(mols)}
        h1 = h3 = h5 = scored = 0
        ranks = []
        for name, note in notes.items():
            if name not in pos:
                continue
            truth = {j for j, d in enumerate(desc) if d.lower() in note}
            if not truth:
                continue
            scored += 1
            order = np.argsort(-P[pos[name]])
            best = min(int(np.where(order == j)[0][0]) for j in truth)
            ranks.append(best + 1)
            h1 += best < 1
            h3 += best < 3
            h5 += best < 5
        print(f"  {label:18s} {scored:7d} {h1/scored:7.1%} {h3/scored:7.1%} "
              f"{h5/scored:7.1%} {np.median(ranks):12.1f}")

    if len(store) < 2:
        return 0

    print("\nSIDE BY SIDE, your notes alongside")
    pick = ["Vanillin", "Ambrettolide", "Blackcurrant mercaptan, 0.1% DPG",
            "Isoamyl Acetate", "Para Cresyl Acetate, 50% in IPM", "Habanolide",
            "Helvetolide", "Stemone", "Labienoxime", "Oxane, 50% TEC"]
    for name in pick:
        if not all(name in s[1] for s in store.values()):
            continue
        print(f"\n  {name}")
        if name in notes:
            print(f"    you      : {notes[name][:70]}")
        for label, (desc, mols, P) in store.items():
            p = P[mols.index(name)]
            top = np.argsort(-p)[:5]
            print(f"    {label:16s}: " + ", ".join(f"{desc[j]}({p[j]:.2f})" for j in top))

    print("\nTHE MUSK PROBLEM, in each model's descriptor space")
    print("  (cosine after centring; 1.0 = identical profile)")
    for label, (desc, mols, P) in store.items():
        pos = {m: i for i, m in enumerate(mols)}
        C = P - P.mean(axis=0)
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        print(f"\n  -- {label} --")
        groups = {k: [m for m in v if m in pos] for k, v in MUSKS.items()}
        for ga, gb in combinations(groups, 2):
            vals = [float(C[pos[a]] @ C[pos[b]])
                    for a, b in product(groups[ga], groups[gb])]
            if vals:
                print(f"     {ga:12s} vs {gb:12s} median cosine {np.median(vals):+.3f}")
        for g, members in groups.items():
            if len(members) > 1:
                vals = [float(C[pos[a]] @ C[pos[b]]) for a, b in combinations(members, 2)]
                print(f"     within {g:34s} {np.median(vals):+.3f}")
        mi = [i for i, d in enumerate(desc) if d.lower() == "musk"]
        if mi:
            allm = [m for v in groups.values() for m in v]
            vals = [P[pos[m], mi[0]] for m in allm]
            print(f"     'musk' descriptor on {len(allm)} musks: "
                  f"median {np.median(vals):.2f}, min {min(vals):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
