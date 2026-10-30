"""Five-fold evaluation of the fingerprint percept model, with per-label detail.

A note on what cannot be compared. OpenPOM's shipped checkpoints were trained
on the whole of curated_GS_LF_merged_4983, so scoring them on any subset of it
measures training-set performance. There is no held-out odour-labelled set that
both models are naive to, so the published 0.8872 and the number below are
*not* a head-to-head result: this is 5-fold CV of one model, quoted against a
figure obtained under a similar protocol by a different group. The only honest
head-to-head available is agreement with the perfumer's notes on materials
neither model trained on, which is reported separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_fp_percept import featurise  # noqa: E402

TRAIN = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"


def main() -> int:
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(ROOT / TRAIN)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    X, ok = featurise(df[smi_col].astype(str).tolist())
    Y = df[tasks].to_numpy(dtype=float)[ok]
    print(f"[cv] {X.shape[0]} molecules, {X.shape[1]} features, {len(tasks)} labels")

    oof = np.zeros_like(Y, dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(kf.split(X), 1):
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(512, 256), alpha=1e-3, batch_size=64,
                          learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                          n_iter_no_change=8, random_state=0),
        )
        clf.fit(X[tr], Y[tr])
        P = clf.predict_proba(X[te])
        P = (np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.ravel(p)
                              for p in P]) if isinstance(P, list) else np.asarray(P))
        oof[te] = P
        print(f"[cv]   fold {k}/5 done", flush=True)

    rows = []
    for j, t in enumerate(tasks):
        n = int(Y[:, j].sum())
        if 3 <= n < len(Y):
            rows.append((t, n, roc_auc_score(Y[:, j], oof[:, j]),
                         average_precision_score(Y[:, j], oof[:, j])))
    aucs = np.array([r[2] for r in rows])
    print(f"\n[cv] out-of-fold mean ROC-AUC over {len(rows)} labels: {aucs.mean():.4f}")
    print(f"     median {np.median(aucs):.4f}   worst {aucs.min():.4f}   best {aucs.max():.4f}")
    print(f"     labels above 0.80: {(aucs >= 0.80).sum()}/{len(aucs)}   "
          f"above 0.90: {(aucs >= 0.90).sum()}/{len(aucs)}")
    print("     OpenPOM reports 0.8872 (5-fold CV, 10-model ensembles, their own split)")

    rows.sort(key=lambda r: -r[2])
    print("\n  strongest labels")
    for t, n, a, ap in rows[:8]:
        print(f"    {t:16s} n={n:4d}  AUC {a:.3f}  AP {ap:.3f}")
    print("  weakest labels")
    for t, n, a, ap in rows[-8:]:
        print(f"    {t:16s} n={n:4d}  AUC {a:.3f}  AP {ap:.3f}")

    watch = ["green", "fruity", "musk", "lactonic", "sulfurous", "woody",
             "floral", "waxy", "fatty", "ethereal"]
    print("\n  labels that matter for this library")
    for t, n, a, ap in rows:
        if t in watch:
            print(f"    {t:16s} n={n:4d}  AUC {a:.3f}  AP {ap:.3f}")

    (ROOT / "artifacts" / "fp_cv_report.json").write_text(json.dumps({
        "folds": 5, "mean_auc": round(float(aucs.mean()), 4),
        "median_auc": round(float(np.median(aucs)), 4),
        "per_label": {t: {"n": n, "auc": round(a, 4), "ap": round(ap, 4)}
                      for t, n, a, ap in rows},
    }, indent=2), encoding="utf-8")
    print("\n[cv] wrote artifacts/fp_cv_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
