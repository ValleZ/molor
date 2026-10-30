"""Second ablation round: close the gap between the rebuilt model and v1.

Round one found pos_weight to be the main destroyer of similarity structure
(+0.373 -> +0.452 when dropped, costing 0.008 AUC) and killed the idea that the
penultimate layer holds extra geometry -- it scores slightly worse than the
output. Ensembling and training length did nothing.

That leaves roughly 0.10 of Spearman between the best rebuilt variant (+0.452)
and v1 (+0.551). Everything still differing is architectural, because v1 was an
sklearn MLPClassifier and v2 is hand-written torch:

1. *scaling the fingerprint bits* -- v1 put StandardScaler across all 2257
   features. Dividing a binary column by its standard deviation multiplies rare
   bits by a large factor, so a molecule with an unusual substructure gets a
   large, idiosyncratic input. v2 deliberately stopped doing this because it
   looked like a bug. It may be what preserves molecule-specific detail.
2. *batch normalisation* -- v1 had none. BatchNorm re-centres each hidden unit
   per batch, which is exactly the kind of operation that would strip the
   per-molecule offsets similarity depends on.
3. *dropout* -- v1 had none; v2 uses 0.3, which forces redundant, distributed
   representations rather than sharp ones.
4. *layer widths* -- v1 was (512, 256) against v2's (1024, 512).

pos_weight is off for every variant here, since round one settled that. The
ensemble is one model and training is capped at 60 epochs, both shown not to
matter, which keeps the sweep affordable.
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
MATERIALS = "library/input/expanded_materials.csv"
N_BITS = 2048


def build(d_in, n_tasks, hidden, bn, p):
    layers, prev = [], d_in
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if p > 0:
            layers.append(nn.Dropout(p))
        prev = h
    layers.append(nn.Linear(prev, n_tasks))
    return nn.Sequential(*layers)


#: name -> (hidden, batchnorm, dropout, scale_all_features)
VARIANTS = {
    "round-1 best":        ((1024, 512), True,  0.3, False),
    "+ scale all feats":   ((1024, 512), True,  0.3, True),
    "+ no batchnorm":      ((1024, 512), False, 0.3, False),
    "+ no dropout":        ((1024, 512), True,  0.0, False),
    "+ narrow (512,256)":  ((512, 256),  True,  0.3, False),
    "v1 architecture":     ((512, 256),  False, 0.0, True),
    "v1 arch, keep BN":    ((512, 256),  True,  0.0, True),
}


def fit(Xtr, Ytr, Xva, Yva, seed, device, hidden, bn, p, epochs=60, lr=1e-3):
    torch.manual_seed(seed)
    model = build(Xtr.shape[1], Ytr.shape[1], hidden, bn, p).to(device)
    lossf = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
    xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    from sklearn.metrics import roc_auc_score
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


def probs(model, X, device):
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default="artifacts/fp_ablation2.json")
    args = ap.parse_args()

    import csv as _csv

    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablate2] device {device}, {args.folds} folds")

    df = pd.read_csv(ROOT / TRAIN)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    Xraw, ok = featurise(df[smi_col].astype(str).tolist())
    Y = df[tasks].to_numpy(dtype=float)[ok]

    rows = list(_csv.DictReader(
        open(ROOT / MATERIALS, newline="", encoding="utf-8-sig")))
    lib_names = [r["name"] for r in rows]
    Xpraw, okp = featurise([r["smiles"] for r in rows])
    lib_names = [lib_names[i] for i in okp]

    truth, _ = gslf_truth()
    keys = material_keys()
    scorable = {n: truth[k] for n, k in keys.items() if k in truth}
    sim_idx = [i for i, n in enumerate(lib_names) if n in scorable]
    J = jaccard_matrix(np.stack([scorable[lib_names[i]] for i in sim_idx]))
    iu = np.triu_indices(len(sim_idx), k=1)
    print(f"[ablate2] {Xraw.shape[0]} molecules, similarity on {len(sim_idx)} "
          f"materials / {len(iu[0])} pairs\n")

    def prep(scale_all):
        """Standardise either every column or only the descriptor block."""
        start = 0 if scale_all else N_BITS
        blk = Xraw[:, start:]
        mu, sd = blk.mean(0), blk.std(0)
        sd[sd < 1e-9] = 1.0
        X = np.hstack([Xraw[:, :start], (blk - mu) / sd])
        Xp = np.hstack([Xpraw[:, :start], (Xpraw[:, start:] - mu) / sd])
        return (np.clip(X.astype(np.float32), -10, 10),
                np.clip(Xp.astype(np.float32), -10, 10))

    kf = list(KFold(args.folds, shuffle=True, random_state=0).split(Xraw))
    report = {}
    for label, (hidden, bn, p, scale_all) in VARIANTS.items():
        X, Xp = prep(scale_all)
        oof = np.zeros_like(Y, dtype=float)
        for k, (tr, te) in enumerate(kf, 1):
            cut = int(len(tr) * 0.9)
            m = fit(X[tr[:cut]], Y[tr[:cut]], X[tr[cut:]], Y[tr[cut:]],
                    100 * k, device, hidden, bn, p)
            oof[te] = probs(m, X[te], device)
        aucs = np.array([roc_auc_score(Y[:, j], oof[:, j]) for j in range(Y.shape[1])
                         if 3 <= Y[:, j].sum() < len(Y)])

        cut = int(len(X) * 0.9)
        final = fit(X[:cut], Y[:cut], X[cut:], Y[cut:], 0, device, hidden, bn, p)
        V = probs(final, Xp, device)[sim_idx]
        rho = float(spearmanr(centred_cosine(V)[iu], J[iu]).statistic)

        report[label] = {"auc": round(float(aucs.mean()), 4),
                         "spearman": round(rho, 4), "hidden": list(hidden),
                         "batchnorm": bn, "dropout": p, "scale_all": scale_all}
        print(f"  {label:22s} AUC {aucs.mean():.4f}   rho {rho:+.4f}", flush=True)

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ablate2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
