"""Fingerprint percept model, second attempt, fixing four setup flaws.

The first version reached 0.791 out-of-fold against OpenPOM's published 0.8872.
Rather than conclude the representation is at fault, this tests whether the gap
was mine:

1. *class imbalance* -- sklearn's log-loss treats a label with 1902 positives
   and one with 42 alike. The weakest labels in v1 were all rare ones, so this
   is the prime suspect. Here each label gets its own pos_weight.
2. *single model vs a ten-model ensemble* -- OpenPOM averages ten; ensembling
   alone is usually worth a couple of AUC points.
3. *scaling the fingerprint bits* -- StandardScaler divides binary features by
   their standard deviation, which inflates rare bits. Only the continuous
   RDKit descriptors are scaled now.
4. *undertraining* -- v1 stopped after at most 120 sklearn iterations.

Runs on CUDA when available: this is plain torch with no DGL, so unlike the
receptor model it is not stuck on CPU.
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

from train_fp_percept import featurise  # noqa: E402

TRAIN = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
N_BITS = 2048


class MLP(nn.Module):
    def __init__(self, d_in: int, n_tasks: int, hidden=(1024, 512), p=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, n_tasks))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_one(Xtr, Ytr, Xva, Yva, seed, device, epochs=180, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], Ytr.shape[1]).to(device)
    # One pos_weight per label, so rare descriptors are not drowned out.
    pos = Ytr.sum(axis=0)
    w = np.clip((len(Ytr) - pos) / np.maximum(pos, 1.0), 1.0, 50.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, dtype=torch.float32, device=device))
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
            from sklearn.metrics import roc_auc_score
            a = [roc_auc_score(Yva[:, j], pv[:, j]) for j in range(Yva.shape[1])
                 if 3 <= Yva[:, j].sum() < len(Yva)]
            m = float(np.mean(a)) if a else 0.0
            if m > best:
                best, patience = m, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


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
    ap.add_argument("--predict-csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--out", default="artifacts/percepts_fp2.json")
    args = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v2] device {device}")

    df = pd.read_csv(ROOT / TRAIN)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    X, ok = featurise(df[smi_col].astype(str).tolist())
    Y = df[tasks].to_numpy(dtype=float)[ok]
    print(f"[v2] {X.shape[0]} molecules x {X.shape[1]} features, {len(tasks)} labels")

    # Scale only the continuous descriptor block; leave the binary bits alone.
    desc = X[:, N_BITS:]
    mu, sd = desc.mean(0), desc.std(0)
    sd[sd < 1e-9] = 1.0
    X = np.hstack([X[:, :N_BITS], (desc - mu) / sd]).astype(np.float32)
    X = np.clip(X, -10, 10)

    oof = np.zeros_like(Y, dtype=float)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X), 1):
        cut = int(len(tr) * 0.9)
        tr_i, va_i = tr[:cut], tr[cut:]
        models = []
        for s in range(args.ensemble):
            m, b = fit_one(X[tr_i], Y[tr_i], X[va_i], Y[va_i], seed=100 * k + s, device=device)
            models.append(m)
        oof[te] = predict(models, X[te], device)
        print(f"[v2]   fold {k}/5 done", flush=True)

    aucs = np.array([roc_auc_score(Y[:, j], oof[:, j]) for j in range(Y.shape[1])
                     if 3 <= Y[:, j].sum() < len(Y)])
    print(f"\n[v2] out-of-fold mean ROC-AUC: {aucs.mean():.4f}  (v1 was 0.7912)")
    print(f"     median {np.median(aucs):.4f}  above 0.80: {(aucs >= 0.80).sum()}/{len(aucs)}"
          f"  above 0.90: {(aucs >= 0.90).sum()}/{len(aucs)}")
    print(f"     OpenPOM reports 0.8872")

    print("[v2] refitting on everything ...", flush=True)
    cut = int(len(X) * 0.9)
    final = [fit_one(X[:cut], Y[:cut], X[cut:], Y[cut:], seed=s, device=device)[0]
             for s in range(args.ensemble)]

    import csv as _csv
    rows = list(_csv.DictReader(open(ROOT / args.predict_csv, newline="", encoding="utf-8-sig")))
    names = [r["name"] for r in rows]
    Xp, okp = featurise([r["smiles"] for r in rows])
    names = [names[i] for i in okp]
    dp = (Xp[:, N_BITS:] - mu) / sd
    Xp = np.clip(np.hstack([Xp[:, :N_BITS], dp]).astype(np.float32), -10, 10)
    P = predict(final, Xp, device)

    (ROOT / args.out).write_text(json.dumps({
        "checkpoint": f"fingerprint MLP v2 (pos_weight, {args.ensemble}x ensemble)",
        "oof_mean_auc": round(float(aucs.mean()), 4),
        "descriptors": tasks, "molecules": names,
        "probs": [[round(float(v), 4) for v in row] for row in P],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"[v2] wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
