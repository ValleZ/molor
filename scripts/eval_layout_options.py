"""Two ways the atlas might be improved on the prediction side, both tested.

*A percept-space layout.* The map's geometry is receptor space, which encodes
chemistry more than smell. Percept vectors are predictions of smell, so laying
the library out in that space instead is the obvious alternative -- and it gave
sensible neighbourhoods when spot-checked, putting Para Cresyl Acetate among
Para Cresol, Indolene and 6-Methyl Quinoline.

*Ensembling the percept models.* Six models score within noise of one another on
the graded panels, which is the classic setting for an average to beat every
member.

Both are measured here rather than assumed, on the two instruments that are not
the training target: nearest-neighbour agreement with the perfumer's written
notes, and rank agreement with the Dravnieks and Keller panels.

Neither works. The result is kept so the work is not repeated.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_impressions import tags_for  # noqa: E402
from eval_similarity import centred_cosine, load_percept  # noqa: E402
from eval_similarity_graded import graded_truth, spearman_profile_matrix  # noqa: E402
from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402

PERCEPTS = {
    "FP v3": "artifacts/percepts_fp3.json",
    "FP v1": "artifacts/percepts_fp.json",
    "FP v2": "artifacts/percepts_fp2.json",
    "OpenPOM": "artifacts/percepts_openpom.json",
    "DenseSense": "artifacts/percepts_densesense.json",
    "GS-LF MPNN": "artifacts/percepts_expanded.json",
}
STORES = {"receptor space, GCN": "library",
          "receptor space, MPNN": "library_mpnn_clean"}


def notes_tags():
    notes = {r["material"]: r["impression"] for r in csv.DictReader(
        io.open(ROOT / "library/input/impressions.tsv", encoding="utf-8"),
        delimiter="\t")}
    return {n: tags_for(t, families=True) for n, t in notes.items()}


def nn_rate(names, D, tags, permutations=2000):
    have = [n for n in names if tags.get(n)]
    if len(have) < 10:
        return None
    idx = {n: names.index(n) for n in have}
    nn = {n: min((D[idx[n], idx[m]], m) for m in have if m != n)[1]
          for n in have}
    rate = float(np.mean([bool(tags[n] & tags[nn[n]]) for n in have]))
    pairs = [(a, b) for k, a in enumerate(have) for b in have[k + 1:]]
    chance = float(np.mean([bool(tags[a] & tags[b]) for a, b in pairs]))
    rng = np.random.default_rng(0)
    null = np.empty(permutations)
    for t in range(permutations):
        perm = list(have)
        rng.shuffle(perm)
        m = dict(zip(have, perm))
        null[t] = np.mean([bool(tags[m[n]] & tags[m[nn[n]]]) for n in have])
    return {"materials": len(have), "nn_shares": round(rate, 4),
            "chance": round(chance, 4), "p": round(float((null >= rate).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/layout_options.json")
    args = ap.parse_args()

    tags = notes_tags()
    report = {"nn_agreement": {}, "graded": {}}

    print("nearest-neighbour agreement with the perfumer's notes")
    for label, store in STORES.items():
        entries, logits, _ = Library(name=store).load()
        names = [e.name for e in entries]
        red, _ = preprocess(logits, n_components=20)
        r = nn_rate(names, squareform(pdist(red)), tags)
        report["nn_agreement"][label] = r
        print(f"  {label:26s} {r['materials']:3d} mats  "
              f"NN shares {r['nn_shares']:.3f}  chance {r['chance']:.3f}  "
              f"p={r['p']:.4f}")

    V = {k: load_percept(ROOT / v) for k, v in PERCEPTS.items()}
    for label, vecs in V.items():
        names = list(vecs)
        P = np.stack([vecs[n] for n in names])
        X = P - P.mean(0)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        r = nn_rate(names, 1.0 - X @ X.T, tags)
        if r:
            report["nn_agreement"][f"percept space, {label}"] = r
            print(f"  percept space, {label:12s} {r['materials']:3d} mats  "
                  f"NN shares {r['nn_shares']:.3f}  chance {r['chance']:.3f}  "
                  f"p={r['p']:.4f}")

    # Ensemble over the models that share the 138-descriptor GS-LF ordering.
    dims = {k: len(next(iter(v.values()))) for k, v in V.items()}
    poolable = [k for k, d in dims.items() if d == 138]
    common = sorted(set.intersection(*[set(V[k]) for k in poolable]))
    V[f"ENSEMBLE (mean of {len(poolable)})"] = {
        n: np.mean([V[k][n] for k in poolable], axis=0) for n in common}
    print(f"\nensemble pools the {len(poolable)} models sharing the 138-descriptor "
          f"ordering: {poolable}")

    for which in ("dravnieks", "keller"):
        truth, _ = graded_truth(which, None)
        names = sorted(set(truth) & set.intersection(*[set(v) for v in V.values()]))
        T = spearman_profile_matrix(np.stack([truth[n] for n in names]))
        iu = np.triu_indices(len(names), k=1)
        res = {}
        for k, v in V.items():
            C = centred_cosine(np.stack([v[n] for n in names]))
            res[k] = round(float(spearmanr(C[iu], T[iu]).statistic), 4)
        report["graded"][which] = {"materials": len(names), "rho": res}
        print(f"\n{which}: {len(names)} materials")
        for k, r in sorted(res.items(), key=lambda kv: -kv[1]):
            print(f"  {k:24s} rho {r:+.4f}")

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[layout] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
