"""Compare the GCN and MPNN MolOR variants on the cases that looked wrong.

The GCN discards bond features entirely; the MPNN consumes them. If the GCN's
inability to separate an aryl acetate from a benzylic acetate is a
featurisation limit, the MPNN should do better here. If it is a training-data
limit, both will fail the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402

CASES = [
    ("Para Cresyl Acetate, 50% in IPM", "Styrallyl Acetate"),
    ("Para Cresyl Acetate, 50% in IPM", "Vanillin"),
    ("Styrallyl Acetate", "Vanillin"),
    ("Helvetolide", "Aldehyde C-10"),
    ("Helvetolide", "Edenolide"),
    ("Vanillin", "Ethyl Vanillin"),
    ("Methyl Benzoate", "Benzyl Acetate"),
]

SERIES = {
    "fatty aldehydes": ["Aldehyde C-8", "Aldehyde C-10", "Aldehyde C-11 Undecylenic"],
    "gamma lactones": ["Gamma Nonalactone", "Gamma Decalactone", "Gamma Undecalactone"],
    "macrocyclic musks": ["Exaltolide Total", "Habanolide", "Ambrettolide"],
    "ester musks": ["Helvetolide", "Edenolide"],
}


def load(name):
    lib = Library(name=name)
    entries, logits, scores = lib.load()
    names = [e.name for e in entries]
    classes = np.array([e.cls for e in entries])
    red, _ = preprocess(logits, n_components=20)
    D = squareform(pdist(red))
    flat = np.sort(pdist(red))
    return dict(names=names, idx={n: i for i, n in enumerate(names)}, classes=classes,
                D=D, flat=flat, scores=scores, logits=logits)


def pctile(m, d):
    return 100.0 * np.searchsorted(m["flat"], d) / len(m["flat"])


def nn_class_rate(m):
    D = m["D"].copy()
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    return float((m["classes"][nn] == m["classes"]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpnn", default="library_mpnn_s6")
    args = ap.parse_args()

    g = load("library")
    try:
        m = load(args.mpnn)
    except Exception as exc:
        raise SystemExit(f"could not load MPNN library {args.mpnn!r}: {exc}")

    common = [n for n in g["names"] if n in m["idx"]]
    print(f"GCN {len(g['names'])} molecules | MPNN {len(m['names'])} | common {len(common)}")

    print("\n" + "=" * 94)
    print("THE CASES IN QUESTION  (distance, and percentile among all pairs)")
    print("=" * 94)
    print(f"{'pair':52s} {'GCN':>18s} {'MPNN':>18s}")
    for a, b in CASES:
        if a not in g["idx"] or b not in g["idx"] or a not in m["idx"] or b not in m["idx"]:
            continue
        dg = g["D"][g["idx"][a], g["idx"][b]]
        dm = m["D"][m["idx"][a], m["idx"][b]]
        label = f"{a.split(',')[0][:24]} <-> {b.split(',')[0][:22]}"
        print(f"{label:52s} {dg:7.2f} ({pctile(g, dg):4.1f}%) {dm:7.2f} ({pctile(m, dm):4.1f}%)")

    print("\n" + "=" * 94)
    print("NEAREST NEIGHBOURS for the materials that looked misplaced")
    print("=" * 94)
    for who in ["Helvetolide", "Para Cresyl Acetate, 50% in IPM", "Galaxolide 50% IPM"]:
        if who not in g["idx"]:
            continue
        print(f"\n  {who}")
        for tag, mod in (("GCN ", g), ("MPNN", m)):
            if who not in mod["idx"]:
                continue
            D = mod["D"].copy()
            np.fill_diagonal(D, np.inf)
            i = mod["idx"][who]
            top = np.argsort(D[i])[:5]
            print(f"    {tag}: " + ", ".join(
                f"{mod['names'][j]} ({D[i, j]:.1f})" for j in top))

    print("\n" + "=" * 94)
    print("HOMOLOGOUS SERIES  (mean within-series distance as a percentile of all pairs)")
    print("=" * 94)
    print(f"{'series':24s} {'GCN':>12s} {'MPNN':>12s}")
    for label, members in SERIES.items():
        vals = []
        for mod in (g, m):
            ii = [mod["idx"][x] for x in members if x in mod["idx"]]
            if len(ii) < 2:
                vals.append(None)
                continue
            w = [mod["D"][a, b] for k, a in enumerate(ii) for b in ii[k + 1:]]
            vals.append(pctile(mod, float(np.mean(w))))
        f = lambda v: f"{v:11.1f}%" if v is not None else f"{'-':>12s}"
        print(f"{label:24s} {f(vals[0])} {f(vals[1])}")

    print("\n" + "=" * 94)
    print("GLOBAL BEHAVIOUR")
    print("=" * 94)
    ig = [g["idx"][n] for n in common]
    im = [m["idx"][n] for n in common]
    dg = squareform(g["D"][np.ix_(ig, ig)], checks=False)
    dm = squareform(m["D"][np.ix_(im, im)], checks=False)
    print(f"  distance-matrix agreement (Spearman)   {spearmanr(dg, dm).statistic:.4f}")
    print(f"  nearest neighbour shares class  GCN    {nn_class_rate(g):.3f}")
    print(f"                                  MPNN   {nn_class_rate(m):.3f}")
    counts = {c: int((g['classes'] == c).sum()) for c in set(g['classes'])}
    n = len(g["classes"])
    chance = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    print(f"                                  chance {chance:.3f}")
    for tag, mod in (("GCN ", g), ("MPNN", m)):
        s = mod["scores"]
        print(f"  {tag} score: median {np.median(s):.4f}  mean {s.mean():.4f}  "
              f"max {s.max():.4f}  frac>0.1 {np.mean(s > 0.1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
