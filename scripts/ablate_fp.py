"""Find out why the worst-AUC model has the best similarity structure.

Fingerprint v1 scores 0.79 out-of-fold and +0.55 on label-grounded similarity.
Fingerprint v2 scores 0.88 and +0.37. Something about fitting the labels harder
destroys the geometry, and v1-vs-v2 changed four things at once, so the pairing
does not say which.

Four candidate mechanisms, each testable:

1. *pos_weight* -- upweighting rare descriptors makes the model assert them
   widely, which adds a shared component to every vector and washes out the
   differences between molecules.
2. *sigmoid saturation* -- a confident model pushes probabilities to 0 and 1,
   and cosine distance between near-binary vectors is coarser than between
   graded ones. If so, the logits keep the structure the probabilities lose.
3. *ensembling* -- averaging five models regresses each prediction towards the
   consensus, which is the class-typical profile.
4. *training length* -- more epochs simply means more of whatever 1-3 do.

And one candidate fix that is not an ablation: read the *penultimate layer*
instead of the output. The 512-dim hidden state is never squashed through a
per-label sigmoid, so it can retain molecule-specific information that the
138-dim descriptor output discards. If that wins on both axes, the right answer
is to keep the accurate model and stop using its output as an embedding.

Each variant is scored on both axes at once: out-of-fold AUC over the GS-LF
folds, and Spearman correlation against human label overlap on the 229 library
materials GS-LF covers.
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
from train_fp_v2 import MLP  # noqa: E402

TRAIN = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
MATERIALS = "library/input/expanded_materials.csv"
N_BITS = 2048

#: name -> (pos_weight, ensemble, epochs, readout)
VARIANTS = {
    "v2 full (probs)":      (True,  5, 180, "probs"),
    "v2 logits":            (True,  5, 180, "logits"),
    "v2 penultimate":       (True,  5, 180, "hidden"),
    "no pos_weight":        (False, 5, 180, "probs"),
    "single model":         (True,  1, 180, "probs"),
    "short training":       (True,  5,  25, "probs"),
    "v1-like (all four)":   (False, 1,  25, "probs"),
}


def fit(Xtr, Ytr, Xva, Yva, seed, device, use_pw, epochs, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], Ytr.shape[1]).to(device)
    if use_pw:
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


def read_out(models, X, device, mode):
    """Average the ensemble in the space the variant is scored in."""
    out = []
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        for m in models:
            if mode == "hidden":
                # Everything except the final Linear.
                h = nn.Sequential(*list(m.net.children())[:-1])(xt)
                out.append(h.cpu().numpy())
            elif mode == "logits":
                out.append(m(xt).cpu().numpy())
            else:
                out.append(torch.sigmoid(m(xt)).cpu().numpy())
    return np.mean(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default="artifacts/fp_ablation.json")
    args = ap.parse_args()

    import csv as _csv

    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablate] device {device}, {args.folds} folds")

    df = pd.read_csv(ROOT / TRAIN)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    X, ok = featurise(df[smi_col].astype(str).tolist())
    Y = df[tasks].to_numpy(dtype=float)[ok]

    desc = X[:, N_BITS:]
    mu, sd = desc.mean(0), desc.std(0)
    sd[sd < 1e-9] = 1.0
    X = np.clip(np.hstack([X[:, :N_BITS], (desc - mu) / sd]).astype(np.float32), -10, 10)
    print(f"[ablate] {X.shape[0]} molecules x {X.shape[1]} features, {len(tasks)} labels")

    # The library, featurised the same way.
    rows = list(_csv.DictReader(
        open(ROOT / MATERIALS, newline="", encoding="utf-8-sig")))
    lib_names = [r["name"] for r in rows]
    Xp, okp = featurise([r["smiles"] for r in rows])
    lib_names = [lib_names[i] for i in okp]
    Xp = np.clip(np.hstack([Xp[:, :N_BITS], (Xp[:, N_BITS:] - mu) / sd]
                           ).astype(np.float32), -10, 10)

    # Ground truth for the similarity axis.
    truth, _ = gslf_truth()
    keys = material_keys()
    scorable = {n: truth[k] for n, k in keys.items() if k in truth}
    sim_idx = [i for i, n in enumerate(lib_names) if n in scorable]
    Ytrue = np.stack([scorable[lib_names[i]] for i in sim_idx])
    J = jaccard_matrix(Ytrue)
    iu = np.triu_indices(len(sim_idx), k=1)
    print(f"[ablate] similarity axis: {len(sim_idx)} materials, {len(iu[0])} pairs\n")

    kf = list(KFold(args.folds, shuffle=True, random_state=0).split(X))
    report = {}
    for label, (use_pw, ens, epochs, mode) in VARIANTS.items():
        oof = np.zeros_like(Y, dtype=float)
        for k, (tr, te) in enumerate(kf, 1):
            cut = int(len(tr) * 0.9)
            tr_i, va_i = tr[:cut], tr[cut:]
            models = [fit(X[tr_i], Y[tr_i], X[va_i], Y[va_i], 100 * k + s,
                          device, use_pw, epochs) for s in range(ens)]
            # AUC is always measured on probabilities, whatever the readout is.
            oof[te] = read_out(models, X[te], device, "probs")
        aucs = np.array([roc_auc_score(Y[:, j], oof[:, j]) for j in range(Y.shape[1])
                         if 3 <= Y[:, j].sum() < len(Y)])

        # Refit on everything for the library representation.
        cut = int(len(X) * 0.9)
        final = [fit(X[:cut], Y[:cut], X[cut:], Y[cut:], s, device, use_pw, epochs)
                 for s in range(ens)]
        V = read_out(final, Xp, device, mode)[sim_idx]
        rho = float(spearmanr(centred_cosine(V)[iu], J[iu]).statistic)

        report[label] = {"auc": round(float(aucs.mean()), 4),
                         "spearman": round(rho, 4), "dim": int(V.shape[1]),
                         "pos_weight": use_pw, "ensemble": ens,
                         "epochs": epochs, "readout": mode}
        print(f"  {label:22s} AUC {aucs.mean():.4f}   rho {rho:+.4f}   "
              f"dim {V.shape[1]}", flush=True)

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ablate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
