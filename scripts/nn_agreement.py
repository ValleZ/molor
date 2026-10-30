"""Nearest-neighbour agreement with the perfumer's notes.

check_impressions asks whether pairs sharing a descriptor are closer *on
average* than pairs that do not. That is a global measure, and it can be
dominated by the bulk of distant pairs. Reading a map is a local act: the eye
looks at what sits next to what. So this measures nearest-neighbour precision
instead -- for each material with a written impression, does its closest
neighbour share at least one descriptor with it?

The two measures can disagree, and on this library they do. The MPNN wins the
global test marginally (gap 1.92, p = 0.030) while the GCN wins this one
solidly: 0.554 against a 0.396 chance rate, permutation p = 0.005, where the
MPNN reaches 0.457 and is not distinguishable from chance (p = 0.178). The local
result is the one that matches what the viewer is used for, and it agrees with
the perfumer's own reading of the two maps.

Chance is the probability that two randomly chosen noted materials share a
descriptor, and the p-value comes from permuting which material carries which
note, holding the geometry fixed.
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
sys.path.insert(0, str(ROOT / "scripts"))

from check_impressions import tags_for  # noqa: E402
from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402


def load(name: str):
    entries, logits, _ = Library(name=name).load()
    names = [e.name for e in entries]
    red, _ = preprocess(logits, n_components=20)
    return names, squareform(pdist(red))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", default=[],
                    help="label=library_name; repeatable")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--flat", action="store_true",
                    help="fine tags only, without the coarse families")
    args = ap.parse_args()
    stores = args.store or ["GCN=library", "MPNN=library_mpnn_clean"]

    notes = {r["material"]: r["impression"] for r in csv.DictReader(
        io.open(ROOT / "library/input/impressions.tsv", encoding="utf-8"),
        delimiter="\t")}
    tags = {n: tags_for(t, families=not args.flat) for n, t in notes.items()}

    print(f"vocabulary: {'fine tags only' if args.flat else 'fine tags + families'}")
    for spec in stores:
        label, _, name = spec.partition("=")
        names, D = load(name)
        have = [n for n in names if tags.get(n)]
        idx = {n: names.index(n) for n in have}

        # Nearest neighbour among materials that also carry notes, so the
        # comparison is not decided by which materials happen to be described.
        nn = {}
        for n in have:
            i = idx[n]
            nn[n] = min(((D[i, idx[m]], m) for m in have if m != n))[1]
        rate = float(np.mean([bool(tags[n] & tags[nn[n]]) for n in have]))

        pairs = [(a, b) for k, a in enumerate(have) for b in have[k + 1:]]
        chance = float(np.mean([bool(tags[a] & tags[b]) for a, b in pairs]))

        rng = np.random.default_rng(0)
        null = np.empty(args.permutations)
        for t in range(args.permutations):
            perm = list(have)
            rng.shuffle(perm)
            m = dict(zip(have, perm))
            null[t] = np.mean([bool(tags[m[n]] & tags[m[nn[n]]]) for n in have])
        p = float((null >= rate).mean())

        print(f"  {label:6s} {name:22s} {len(have):3d} materials   "
              f"NN shares a descriptor {rate:.3f}   chance {chance:.3f}   "
              f"permutation p = {p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
