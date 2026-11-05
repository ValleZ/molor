"""Percept model v3: the configuration the ablations selected.

v1 had the best similarity geometry and the worst accuracy; v2 reversed both.
Neither trade was necessary. Two ablation rounds (artifacts/fp_ablation.json,
fp_ablation2.json) found that almost everything v2 added for accuracy was paid
for in geometry, and that the two biggest costs bought almost no accuracy:

  drop pos_weight        +0.079 Spearman, -0.008 AUC
  standardise all feats  +0.064 Spearman, -0.018 AUC
  drop dropout           +0.045 Spearman, -0.009 AUC
  drop batchnorm         needed with the above; harmful on its own
  narrow to (512, 256)   v1's widths

while the things v2 was right about -- ensembling and long training -- cost no
geometry at all and are kept here. So v3 is v1's architecture trained v2's way.

Standardising the binary fingerprint bits deserves a note, because v2 removed it
deliberately as a bug. Dividing a binary column by its standard deviation
multiplies rare bits by a large factor, so a molecule carrying an unusual
substructure gets a correspondingly large input. For per-label ranking that is
noise, which is why it costs AUC. For telling two molecules apart it is signal,
and it is the second largest term above.

Scored on both axes rather than one: out-of-fold AUC over GS-LF, and Spearman
against human label overlap on the 26106 library pairs GS-LF covers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_similarity import (centred_cosine, gslf_truth, jaccard_matrix,  # noqa: E402
                             material_keys)
from train_fp_percept import featurise  # noqa: E402

TRAIN = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"

#: Selected by ablation. Every departure from v2 is justified in the docstring.
HIDDEN = (512, 256)
BATCHNORM = False
DROPOUT = 0.0
POS_WEIGHT = False
SCALE_ALL = True


def build(d_in: int, n_tasks: int) -> nn.Sequential:
    layers, prev = [], d_in
    for h in HIDDEN:
        layers.append(nn.Linear(prev, h))
        if BATCHNORM:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if DROPOUT > 0:
            layers.append(nn.Dropout(DROPOUT))
        prev = h
    layers.append(nn.Linear(prev, n_tasks))
    return nn.Sequential(*layers)


def fit(Xtr, Ytr, Xva, Yva, seed, device, epochs=180, lr=1e-3):
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    model = build(Xtr.shape[1], Ytr.shape[1]).to(device)
    if POS_WEIGHT:
        pos = Ytr.sum(axis=0)
        w = np.clip((len(Ytr) - pos) / np.maximum(pos, 1.0), 1.0, 50.0)
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        lossf = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
    xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    best, best_state, patience = -1.0, None, 0
    n, bs = len(xt), 128
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            lossf(model(xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(xv)).cpu().numpy()
            a = [roc_auc_score(Yva[:, j], pv[:, j]) for j in range(Yva.shape[1])
                 if 3 <= Yva[:, j].sum() < len(Yva)]
            m = float(np.mean(a)) if a else 0.0
            if m > best:
                best, patience = m, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(models, X, device):
    out = []
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        for m in models:
            out.append(torch.sigmoid(m(xt)).cpu().numpy())
    return np.mean(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--predict-csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--train-csv", default=TRAIN,
                    help="label set to train on; data/percept/"
                         "gslf_hierarchical.csv is GS-LF with its implied "
                         "hierarchy propagated")
    ap.add_argument("--out", default="artifacts/percepts_fp3.json")
    ap.add_argument("--pos-weight", action="store_true",
                    help="enable per-label pos_weight (capped at 50). The "
                         "ablations dropped it for GEOMETRY (it costs ~+0.08 "
                         "Spearman) and accepted -0.008 AUC, but they never "
                         "measured what it does to the predicted "
                         "distribution -- which is what decides whether a "
                         "rare facet has any material above 0.7 at all")
    args = ap.parse_args()
    if args.pos_weight:
        globals()["POS_WEIGHT"] = True
        print("[v3] pos_weight ENABLED (per-label neg/pos, capped at 50)")

    import csv as _csv

    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v3] device {device}")

    df = pd.read_csv(ROOT / args.train_csv)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    Xraw, ok = featurise(df[smi_col].astype(str).tolist())
    Y = df[tasks].to_numpy(dtype=float)[ok]

    rows = list(_csv.DictReader(
        open(ROOT / args.predict_csv, newline="", encoding="utf-8-sig")))
    names = [r["name"] for r in rows]
    Xpraw, okp = featurise([r["smiles"] for r in rows])
    names = [names[i] for i in okp]
    print(f"[v3] {Xraw.shape[0]} training molecules x {len(tasks)} labels, "
          f"{len(names)} library materials")

    start = 0 if SCALE_ALL else 2048
    blk = Xraw[:, start:]
    mu, sd = blk.mean(0), blk.std(0)
    sd[sd < 1e-9] = 1.0
    X = np.clip(np.hstack([Xraw[:, :start], (blk - mu) / sd]
                          ).astype(np.float32), -10, 10)
    Xp = np.clip(np.hstack([Xpraw[:, :start], (Xpraw[:, start:] - mu) / sd]
                           ).astype(np.float32), -10, 10)

    oof = np.zeros_like(Y, dtype=float)
    for k, (tr, te) in enumerate(
            KFold(args.folds, shuffle=True, random_state=0).split(X), 1):
        cut = int(len(tr) * 0.9)
        models = [fit(X[tr[:cut]], Y[tr[:cut]], X[tr[cut:]], Y[tr[cut:]],
                      100 * k + s, device) for s in range(args.ensemble)]
        oof[te] = predict(models, X[te], device)
        print(f"[v3]   fold {k}/{args.folds} done", flush=True)

    aucs = np.array([roc_auc_score(Y[:, j], oof[:, j]) for j in range(Y.shape[1])
                     if 3 <= Y[:, j].sum() < len(Y)])
    print(f"\n[v3] out-of-fold mean ROC-AUC: {aucs.mean():.4f}")
    print(f"     v1 0.7912   v2 0.8818   OpenPOM reports 0.8872")
    print(f"     median {np.median(aucs):.4f}  above 0.80: {(aucs >= 0.80).sum()}"
          f"/{len(aucs)}  above 0.90: {(aucs >= 0.90).sum()}/{len(aucs)}")

    print("[v3] refitting on everything ...", flush=True)
    cut = int(len(X) * 0.9)
    final = [fit(X[:cut], Y[:cut], X[cut:], Y[cut:], s, device)
             for s in range(args.ensemble)]
    P = predict(final, Xp, device)

    truth, _ = gslf_truth()
    keys = material_keys()
    scorable = {n: truth[k] for n, k in keys.items() if k in truth}
    idx = [i for i, n in enumerate(names) if n in scorable]
    # The similarity check needs prediction-set molecules that carry human
    # labels, and a --predict-csv of novel structures may have none: scoring
    # petroleum alkanes against GS-LF crashed here on an empty stack. The
    # model and its output are unaffected, so the right behaviour is to say
    # the check could not run and still write the file.
    if len(idx) >= 3:
        J = jaccard_matrix(np.stack([scorable[names[i]] for i in idx]))
        iu = np.triu_indices(len(idx), k=1)
        rho = float(spearmanr(centred_cosine(P[idx])[iu], J[iu]).statistic)
        print(f"\n[v3] label-grounded similarity over {len(iu[0])} pairs: "
              f"rho {rho:+.4f}")
        print(f"     v1 +0.5505   v2 +0.3731   OpenPOM +0.4476   "
              f"Uni-Mol +0.1508")
    else:
        rho = float("nan")
        print(f"\n[v3] label-grounded similarity not computed: only "
              f"{len(idx)} of {len(names)} prediction molecules carry a "
              f"GS-LF label")
        print(f"     the out-of-fold AUC above is unaffected; it is measured "
              f"on the training set")

    (ROOT / args.out).write_text(json.dumps({
        "checkpoint": f"fingerprint MLP v3 (ablation-selected, {args.ensemble}x ensemble)",
        "oof_mean_auc": round(float(aucs.mean()), 4),
        "similarity_spearman": (round(rho, 4) if rho == rho else None),
        "config": {"hidden": list(HIDDEN), "batchnorm": BATCHNORM,
                   "dropout": DROPOUT, "pos_weight": POS_WEIGHT,
                   "scale_all": SCALE_ALL, "ensemble": args.ensemble},
        "descriptors": tasks, "molecules": names,
        "probs": [[round(float(v), 4) for v in row] for row in P],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"[v3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
