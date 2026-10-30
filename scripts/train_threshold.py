"""Predict odour detection threshold, and prefer the measured value where it exists.

This replaces the intensity model as the dosing number, because that model was
trained on the wrong quantity. keller_2016 records perceived strength at a fixed
dilution: linalool rates 57.6 out of 100 there, mid-range, which is what a naive
panel smells at 1e-3 in paraffin oil. Its measured detection threshold puts it
near the bottom of this library instead, which is where the bench puts it --
linalool is dosed in percentages, Aldehyde C-12 in traces. The two quantities
correlate at only r 0.59 over the 28 shared materials, so one cannot stand in
for the other.

So the target here is Abraham's Log(1/ODT): higher means a lower threshold and a
more potent material. 268 molecules, measured.

The first rule is not to predict what is known. 28 of the 301 library materials
have a measured threshold, and those are reported as measured; the model only
fills the remaining 273. Predicted and measured values are kept in separate
columns so nothing downstream can confuse them.

268 training molecules is small, so the honest expectation is modest. Two
baselines say whether structure is doing anything: molecular weight and logP
alone -- the "small and volatile is potent" prior -- and the training mean.
Folds are grouped on InChIKey, which here is one molecule per group, so this is
plain cross-validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_fp_percept import featurise  # noqa: E402

DATA = ROOT / "data" / "percept" / "abraham_thresholds.csv"
TARGET = "Log (1/ODT)"
N_BITS = 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--predict-csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--out", default="artifacts/threshold.json")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}; run scripts/build_percept_datasets.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA)
    df = df[df[TARGET].notna() & df["smiles"].notna()].copy()
    X, ok = featurise(df["smiles"].astype(str).tolist())
    df = df.iloc[ok].reset_index(drop=True)
    y = df[TARGET].to_numpy(float)
    print(f"[odt] {len(df)} molecules with a measured threshold, device {device}")
    print(f"[odt] target {TARGET}: mean {y.mean():.2f} sd {y.std():.2f} "
          f"range {y.min():.2f}..{y.max():.2f}")

    desc = X[:, N_BITS:]
    mu, sd = desc.mean(0), desc.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = np.clip(np.hstack([X[:, :N_BITS], (desc - mu) / sd]), -10, 10).astype(np.float32)

    class MLP(nn.Module):
        def __init__(self, d_in, hidden=(256, 128), p=0.1):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def fit(tr, seed, epochs=200):
        torch.manual_seed(seed)
        m = MLP(Xs.shape[1]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        xt = torch.tensor(Xs[tr], device=device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
        lossf = nn.MSELoss()
        n, bs = len(xt), 32
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                lossf(m(xt[idx]), yt[idx]).backward()
                opt.step()
        m.eval()
        return m

    def predict(models, Z):
        with torch.no_grad():
            zt = torch.tensor(Z, device=device)
            return np.mean([m(zt).cpu().numpy() for m in models], axis=0)

    kf = list(KFold(args.folds, shuffle=True, random_state=0).split(Xs))
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(kf, 1):
        models = [fit(tr, 100 * k + s) for s in range(args.ensemble)]
        oof[te] = predict(models, Xs[te])
    print()

    def score(pred, label):
        r = pearsonr(pred, y).statistic
        rho = spearmanr(pred, y).statistic
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        print(f"  {label:32s} r {r:+.4f}  rho {rho:+.4f}  RMSE {rmse:.2f}")
        return {"pearson": round(float(r), 4), "spearman": round(float(rho), 4),
                "rmse": round(rmse, 3)}

    res = {"model (ECFP4 + descriptors)": score(oof, "ECFP4 + descriptors")}

    from rdkit.Chem import Descriptors
    dn = [n for n, _ in Descriptors._descList]
    cols = [N_BITS + dn.index(c) for c in ("MolWt", "MolLogP") if c in dn]
    if len(cols) == 2:
        Z = X[:, cols]
        pred = np.zeros(len(y))
        for tr, te in kf:
            pred[te] = Ridge(alpha=1.0).fit(Z[tr], y[tr]).predict(Z[te])
        res["volatility proxy (MW, logP)"] = score(pred, "volatility proxy (MW, logP)")
    pred = np.zeros(len(y))
    for tr, te in kf:
        pred[te] = y[tr].mean()
    res["training mean"] = score(pred, "training mean")

    # Library: measured where known, predicted otherwise.
    import csv as _csv
    rows = list(_csv.DictReader(
        open(ROOT / args.predict_csv, newline="", encoding="utf-8-sig")))
    names = [r["name"] for r in rows]
    Xp, okp = featurise([r["smiles"] for r in rows])
    names = [names[i] for i in okp]
    dp = (Xp[:, N_BITS:] - mu) / sd
    Xp = np.clip(np.hstack([Xp[:, :N_BITS], dp]), -10, 10).astype(np.float32)

    final = [fit(np.arange(len(y)), s) for s in range(args.ensemble)]
    pred_lib = predict(final, Xp)

    # Applicability domain. Abraham measured small volatiles -- MW up to 232,
    # 95th percentile 158 -- while this library runs to 305 with a median of
    # 170, so much of it sits in or beyond the sparse tail of the training
    # distribution. Predictions there are not merely noisier, they are
    # systematically too potent: the materials heavier than anything measured
    # average 2.49 against 1.81 for the rest, which is how near-odourless
    # fixatives like cinnamyl cinnamate and benzyl salicylate ended up at the
    # top of the ranking. Flagged rather than silently shipped.
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    train_mols = [Chem.MolFromSmiles(s) for s in df["smiles"].astype(str)]
    train_fps = [gen.GetFingerprint(m) for m in train_mols if m is not None]
    train_mw = np.array([Descriptors.MolWt(m) for m in train_mols if m is not None])
    mw_max = float(train_mw.max())

    smi_by_name = {r["name"]: r["smiles"] for r in rows}
    measured = {r["in_library"]: float(r[TARGET])
                for _, r in df.iterrows() if str(r["in_library"])}

    out_vals, out_src, out_mw, out_sim = [], [], [], []
    for n, p in zip(names, pred_lib):
        mol = Chem.MolFromSmiles(smi_by_name[n])
        mw = float(Descriptors.MolWt(mol))
        sim = max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(mol),
                                                     train_fps))
        out_mw.append(round(mw, 1))
        out_sim.append(round(float(sim), 3))
        if n in measured:
            out_vals.append(round(measured[n], 3))
            out_src.append("measured")
        elif mw > mw_max or sim < 0.30:
            out_vals.append(round(float(p), 3))
            out_src.append("out-of-domain")
        else:
            out_vals.append(round(float(p), 3))
            out_src.append("predicted")
    n_meas = out_src.count("measured")
    n_ood = out_src.count("out-of-domain")
    print(f"\n[odt] training MW up to {mw_max:.0f}; library: {n_meas} measured, "
          f"{out_src.count('predicted')} predicted in-domain, "
          f"{n_ood} out-of-domain (heavier than anything measured, or "
          f"ECFP4 similarity below 0.30 to every training molecule)")

    # Rank only what is trustworthy: measured values and in-domain predictions.
    trust = [i for i, s_ in enumerate(out_src) if s_ != "out-of-domain"]
    order = sorted(trust, key=lambda i: -out_vals[i])
    print("  most potent (measured or in-domain only):")
    for i in order[:12]:
        print(f"    {out_vals[i]:6.2f}  [{out_src[i]:9s}]  {names[i]}")
    print("  least potent:")
    for i in order[-8:]:
        print(f"    {out_vals[i]:6.2f}  [{out_src[i]:9s}]  {names[i]}")
    ood = sorted((i for i, s_ in enumerate(out_src) if s_ == "out-of-domain"),
                 key=lambda i: -out_vals[i])
    print(f"  withheld as out-of-domain ({len(ood)}), predictions shown but "
          f"not to be trusted:")
    for i in ood[:8]:
        print(f"    {out_vals[i]:6.2f}  MW {out_mw[i]:6.1f}  "
              f"max sim {out_sim[i]:.2f}  {names[i]}")

    (ROOT / args.out).write_text(json.dumps({
        "target": "Abraham 2012 Log(1/ODT); higher = lower threshold = more potent",
        "training_molecules": int(len(y)),
        "cv": {"folds": args.folds, "ensemble": args.ensemble},
        "scores": res,
        "applicability_domain": {
            "train_mw_max": round(mw_max, 1),
            "rule": "out-of-domain if MW exceeds the heaviest measured molecule "
                    "or ECFP4 similarity to every training molecule is below 0.30",
        },
        "molecules": names,
        "log_inv_odt": out_vals,
        "source": out_src,
        "mol_weight": out_mw,
        "max_train_similarity": out_sim,
    }, indent=2), encoding="utf-8")
    print(f"\n[odt] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
