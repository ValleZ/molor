"""The nitro-vs-macrocyclic musk case, and the same test judged by a human nose.

Two refinements on chem_vs_smell.py.

First, musk chemotypes are named explicitly rather than lumped, because the
aggregate hides the case that matters: nitro musks share no scaffold at all
with macrocyclic ones.

Second, the "smell" axis there was the GS-LF percept model -- itself a graph
network over the same molecules, so it may group musks by structure for the
same reasons the receptor model does, and two correlated models can agree while
both being wrong. The perfumer's own notes are the only odour data here that no
model produced, so the hard-case test is repeated against them.
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist, squareform

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402


def _require_notes(path):
    """Exit with an explanation rather than a traceback when the notes are absent.

    These are the perfumer's own written notes and are deliberately not in this repository; see .gitignore. Everything derived from them that was cleared for publication is in data/inventory/inventory.csv.
    """
    if not path.is_file():
        raise SystemExit(
            f"{path.name} is not in this repository: it is the perfumer's own "
            f"written smelling notes. Everything derived from them that was "
            f"cleared for publication is in data/inventory/inventory.csv. "
            f"This script measures something against those notes and cannot "
            f"run without them.")
    return path

CHEMOTYPE = {
    "nitro": ["Musk Xylene", "Musk Ambrette", "Musk Ketone"],
    "macrocyclic": ["Exaltolide Total", "Ambrettolide", "Habanolide",
                    "Ethylene Brassylate", "Velvione", "Muscone"],
    "polycyclic": ["Galaxolide 50% IPM", "Tonalide", "Celestolide", "Cashmeran"],
    "alicyclic/other": ["Helvetolide", "Edenolide", "Iso E Super",
                        "Ambroxan, 10% in Alcohol"],
}


def main() -> int:
    entries, logits, _ = Library(name="library_mpnn_expanded").load()
    names = [e.name for e in entries]
    idx = {n: i for i, n in enumerate(names)}
    smiles = {n: e.canonical_smiles for n, e in zip(names, entries)}

    red, _ = preprocess(logits, n_components=20)
    rec = squareform(pdist(red))
    flat = np.sort(pdist(red))
    pct = lambda v: 100.0 * np.searchsorted(flat, v) / len(flat)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {n: gen.GetFingerprint(Chem.MolFromSmiles(s)) for n, s in smiles.items()}

    print("MUSK CHEMOTYPES: do they recognise each other?")
    print(f"  {'pair':56s} {'tanimoto':>9s} {'receptor':>9s} {'pctile':>7s}")
    groups = {k: [m for m in v if m in idx] for k, v in CHEMOTYPE.items()}
    for ga, gb in combinations(groups, 2):
        rows = []
        for a, b in product(groups[ga], groups[gb]):
            rows.append((pct(rec[idx[a], idx[b]]),
                         DataStructs.TanimotoSimilarity(fp[a], fp[b]),
                         rec[idx[a], idx[b]], a, b))
        if not rows:
            continue
        rows.sort()
        med = np.median([r[0] for r in rows])
        print(f"\n  -- {ga} vs {gb}  (median percentile {med:.0f}%) --")
        for p, t, d, a, b in rows[:3]:
            print(f"  {(a[:26] + ' / ' + b[:26]):56s} {t:9.3f} {d:9.2f} {p:6.1f}%")
        if len(rows) > 3:
            p, t, d, a, b = rows[-1]
            print(f"  {'worst: ' + (a[:20] + ' / ' + b[:20]):56s} {t:9.3f} {d:9.2f} {p:6.1f}%")

    # within-chemotype, for scale
    print("\n  within a single chemotype, for comparison")
    for g, members in groups.items():
        if len(members) < 2:
            continue
        vals = [pct(rec[idx[a], idx[b]]) for a, b in combinations(members, 2)]
        print(f"    {g:18s} median percentile {np.median(vals):5.1f}%  (n={len(vals)})")

    # ---- the hard-case test, judged by the perfumer's notes ----
    notes_path = ROOT / "library" / "input" / "impressions.tsv"
    if not notes_path.is_file():
        return 0
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_impressions import tags_for

    with open(notes_path, encoding="utf-8") as fh:
        notes = {r["material"]: r["impression"]
                 for r in csv.DictReader(fh, delimiter="\t")}
    tagged = {n: tags_for(notes[n]) for n in names if n in notes and tags_for(notes[n])}
    keys = sorted(tagged)
    print(f"\nHARD-CASE TEST JUDGED BY YOUR NOTES ({len(keys)} materials)")

    hard, ctrl = [], []
    for a, b in combinations(keys, 2):
        t = DataStructs.TanimotoSimilarity(fp[a], fp[b])
        if t >= 0.25:
            continue
        ta, tb = tagged[a], tagged[b]
        j = len(ta & tb) / len(ta | tb)
        d = pct(rec[idx[a], idx[b]])
        (hard if j >= 0.5 else ctrl if j == 0 else None) is None or None
        if j >= 0.5:
            hard.append((d, a, b, j))
        elif j == 0:
            ctrl.append(d)
    if hard:
        print(f"  structurally unrelated but your notes agree (Jaccard>=0.5): "
              f"{len(hard)} pairs")
        print(f"    median receptor percentile {np.median([h[0] for h in hard]):.1f}%")
        print(f"  structurally unrelated and your notes share nothing: {len(ctrl)} pairs")
        print(f"    median receptor percentile {np.median(ctrl):.1f}%")
        print("\n  closest such pairs:")
        for d, a, b, j in sorted(hard)[:8]:
            print(f"    {d:5.1f}%  {a[:28]:30s} / {b[:28]:30s}")
            print(f"           {notes[a][:44]:46s} | {notes[b][:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
