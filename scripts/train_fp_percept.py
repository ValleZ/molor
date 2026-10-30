"""Train an odour-descriptor model on fingerprints instead of a learned graph.

Three graph architectures -- a 2-layer GCN, an MPNN, and an MPNN augmented with
electron density -- all return effectively identical predictions for delta- and
gamma-decalactone. The cause is the readout: message passing converged to the
same multiset of atom embeddings, and every permutation-invariant readout maps
equal multisets to equal outputs. It is the 1-Weisfeiler-Leman limit, and ring
size is its textbook blind spot.

ECFP sidesteps it because nothing is learned and nothing converges: the bit
vector simply records which substructures are present, and a six-ring fragment
hashes differently from a five-ring one. Measured on the pairs that broke the
GNNs, ECFP4 Tanimoto runs 0.29-0.77 rather than 1.0.

So this trains the dullest possible model -- Morgan bits plus RDKit descriptors
into a multi-label MLP -- on the same GS-LF data OpenPOM uses, to find out what
the graph machinery was buying. Held-out AUC is reported against OpenPOM's
published 0.8872 (5-fold CV with 10-model ensembles), which is a kinder
protocol than the single split used here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def featurise(smiles: list[str]):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    desc_fns = [f for _, f in Descriptors._descList]
    rows, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = np.zeros(2048, dtype=np.float32)
        for b in gen.GetFingerprint(m).GetOnBits():
            fp[b] = 1.0
        d = np.array([fn(m) for fn in desc_fns], dtype=np.float64)
        d[~np.isfinite(d)] = 0.0
        rows.append(np.concatenate([fp, d]))
        ok.append(i)
    return np.vstack(rows), ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",
                    default="vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv")
    ap.add_argument("--predict-csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--out", default="artifacts/percepts_fp.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(ROOT / args.train_csv)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    print(f"[fp] training set {len(df)} molecules x {len(tasks)} descriptors")

    X, ok = featurise(df[smi_col].astype(str).tolist())
    Y = df[tasks].to_numpy(dtype=float)[ok]
    print(f"[fp] features {X.shape}")

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=args.seed)
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(512, 256), alpha=1e-3, batch_size=64,
                      learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                      n_iter_no_change=8, random_state=args.seed),
    )
    print("[fp] training ...", flush=True)
    clf.fit(Xtr, Ytr)

    Pte = clf.predict_proba(Xte)
    Pte = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.ravel(p)
                           for p in Pte]) if isinstance(Pte, list) else np.asarray(Pte)
    aucs = []
    for j in range(Y.shape[1]):
        if Yte[:, j].sum() >= 3 and Yte[:, j].sum() < len(Yte):
            aucs.append(roc_auc_score(Yte[:, j], Pte[:, j]))
    print(f"[fp] held-out mean ROC-AUC over {len(aucs)} labels: {np.mean(aucs):.4f}")
    print(f"     (OpenPOM reports 0.8872 under 5-fold CV with 10-model ensembles)")

    # refit on everything, then predict the perfumery library
    print("[fp] refitting on the full training set ...", flush=True)
    clf.fit(X, Y)

    import csv as _csv

    rows = list(_csv.DictReader(open(ROOT / args.predict_csv, newline="",
                                     encoding="utf-8-sig")))
    names = [r["name"] for r in rows]
    Xp, okp = featurise([r["smiles"] for r in rows])
    names = [names[i] for i in okp]
    Pp = clf.predict_proba(Xp)
    Pp = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.ravel(p)
                          for p in Pp]) if isinstance(Pp, list) else np.asarray(Pp)
    print(f"[fp] predicted {Pp.shape[0]} molecules, median {np.median(Pp):.4f}")

    (ROOT / args.out).write_text(json.dumps({
        "checkpoint": "fingerprint MLP (ECFP4 2048 + RDKit descriptors)",
        "heldout_mean_auc": round(float(np.mean(aucs)), 4),
        "descriptors": tasks,
        "molecules": names,
        "probs": [[round(float(v), 4) for v in row] for row in Pp],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"[fp] wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
