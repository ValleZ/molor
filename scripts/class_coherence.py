"""Nearest-neighbour family coherence, restored from the perfumer's own labels.

This axis had gone dead. The README quotes 0.422 for the GCN, measured on an
older 128-material file that carried a chemical class per row;
expanded_materials.csv has no such column, so every material shared the empty
class and the statistic read 1.000 for both models and for chance alike --
which is to say it measured nothing.

The inventory sheet supplies a family per material for 128 of the 301. These
are odour/chemical families as a working perfumer assigns them ("Fruity ester",
"Rose alcohol", "Musk"), not a formal taxonomy, which is arguably the more
relevant grouping for this project than a structural class would be.

Granularity needs care. The raw labels are 94 distinct over 165 materials with
66 singletons, and a singleton can never have a same-family neighbour, so
including them only drags every model down by the same constant. Coarsening on
the first token before a slash gives 38 families with 7 singletons. Both are
reported, since the choice could otherwise be made to flatter a result.

Chance is the probability that two randomly chosen labelled materials share a
family, and the p-value comes from permuting which material carries which
label while holding the geometry fixed.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402

INV = ROOT / "data" / "inventory" / "inventory.csv"


def families(coarse: bool) -> dict[str, str]:
    out = {}
    if not INV.is_file():
        raise SystemExit(f"missing {INV}; run scripts/build_inventory.py")
    for r in csv.DictReader(io.open(INV, encoding="utf-8-sig")):
        fam = (r.get("family") or "").strip()
        if not fam:
            continue
        out[r["name"]] = fam.split("/")[0].strip() if coarse else fam
    return out


def load(name: str):
    entries, logits, _ = Library(name=name).load()
    names = [e.name for e in entries]
    red, _ = preprocess(logits, n_components=20)
    return names, squareform(pdist(red))


def score(names, D, fam, permutations, drop_singletons):
    have = [n for n in names if n in fam]
    counts: dict[str, int] = {}
    for n in have:
        counts[fam[n]] = counts.get(fam[n], 0) + 1
    if drop_singletons:
        have = [n for n in have if counts[fam[n]] > 1]
    if len(have) < 10:
        return None
    idx = {n: names.index(n) for n in have}

    nn = {}
    for n in have:
        i = idx[n]
        nn[n] = min((D[i, idx[m]], m) for m in have if m != n)[1]
    rate = float(np.mean([fam[n] == fam[nn[n]] for n in have]))

    pairs = [(a, b) for k, a in enumerate(have) for b in have[k + 1:]]
    chance = float(np.mean([fam[a] == fam[b] for a, b in pairs]))

    rng = np.random.default_rng(0)
    null = np.empty(permutations)
    for t in range(permutations):
        perm = list(have)
        rng.shuffle(perm)
        m = dict(zip(have, perm))
        null[t] = np.mean([fam[m[n]] == fam[m[nn[n]]] for n in have])
    p = float((null >= rate).mean())
    return {"materials": len(have), "n_families": len({fam[n] for n in have}),
            "nn_same_family": round(rate, 4), "chance": round(chance, 4),
            "p": round(p, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", default=[],
                    help="label=library_name; repeatable")
    ap.add_argument("--permutations", type=int, default=2000)
    args = ap.parse_args()
    stores = args.store or ["GCN=library", "MPNN=library_mpnn_clean"]

    loaded = {}
    for spec in stores:
        label, _, name = spec.partition("=")
        loaded[label] = load(name)

    for coarse in (False, True):
        fam = families(coarse)
        tag = "coarsened on the first token" if coarse else "raw labels"
        print(f"\n=== family coherence, {tag} "
              f"({len({v for v in fam.values()})} families over "
              f"{len(fam)} materials) ===")
        for drop in (False, True):
            note = ("excluding families with a single member"
                    if drop else "all labelled materials")
            print(f"  {note}")
            for label, (names, D) in loaded.items():
                r = score(names, D, fam, args.permutations, drop)
                if r is None:
                    print(f"    {label:6s} too few labelled materials")
                    continue
                print(f"    {label:6s} {r['materials']:3d} materials, "
                      f"{r['n_families']:2d} families   "
                      f"NN shares family {r['nn_same_family']:.3f}   "
                      f"chance {r['chance']:.3f}   p = {r['p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
