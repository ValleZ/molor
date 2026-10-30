"""Predict perceived odour intensity, which nothing else in this project does.

Every model here predicts *character* -- what a material smells of. None
predicts how strong it smells, so every dosing judgement has been guesswork:
the OAV weights in the accord work were invented, and the perfumer had to
correct them from the bench ("2-methylbutyrate is super potent").

keller_2016 supplies the missing target. 474 molecules were presented at two or
more dilutions in paraffin oil and rated 0-100 for strength by 55 subjects.
Intensity at a fixed dilution is essentially potency, which is what dosing
needs.

Three things decide whether a number here means anything.

*The split must group by molecule.* Each molecule appears at several
concentrations. Splitting stimuli at random would put the same molecule in train
and test and measure memorisation, so folds are grouped on InChIKey.

*Concentration must be a baseline, not a feature to be credited.* Intensity
rises steeply with dose -- mean 20 at 1e-7 against 58 at 1e-1 -- so a model
knowing only the dilution already predicts well. The question is whether
structure adds anything on top, so log concentration alone is scored as its own
baseline. A volatility proxy (molecular weight and logP) is scored too, since
"small and volatile smells stronger" is the obvious chemical prior.

*The panel's own agreement is the ceiling.* Between-subject standard deviation
averages 26.3 on a mean near 40, so the target is noisy. Split-half reliability
over subjects, corrected by Spearman-Brown, says how well the panel mean
predicts itself; a model near that number has extracted what is there, and no
model can exceed it except by fitting noise.
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

DATA = ROOT / "data" / "percept" / "keller.csv"
N_BITS = 2048


def noise_ceiling() -> dict:
    """Read the panel ceiling computed by build_percept_datasets.

    It needs the raw per-subject ratings, which live behind pyrfume in a
    different environment, so it is computed once at dataset build time and
    carried in the manifest rather than recomputed here.
    """
    man = ROOT / "data" / "percept" / "manifest.json"
    if not man.is_file():
        return {"error": "manifest.json missing; run build_percept_datasets.py"}
    m = json.loads(man.read_text(encoding="utf-8"))
    c = m.get("keller_intensity_ceiling")
    return c or {"error": "manifest has no keller_intensity_ceiling; rebuild it"}


def build_features(df: pd.DataFrame):
    X, ok = featurise(df["smiles"].astype(str).tolist())
    df = df.iloc[ok].reset_index(drop=True)
    logc = np.log10(df["concentration"].to_numpy(float))[:, None]
    return X, logc, df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--predict-csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--predict-conc", type=float, default=1e-3,
                    help="dilution at which to predict the library")
    ap.add_argument("--out", default="artifacts/intensity.json")
    ap.add_argument("--skip-ceiling", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from scipy.stats import pearsonr, spearmanr
    from sklearn.model_selection import GroupKFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA)
    df = df[df["intensity"].notna() & df["smiles"].notna()].copy()
    print(f"[int] {len(df)} stimuli, {df['inchikey'].nunique()} molecules, "
          f"device {device}")

    X, logc, df = build_features(df)
    y = df["intensity"].to_numpy(float)
    groups = df["inchikey"].to_numpy()
    print(f"[int] features {X.shape}, target mean {y.mean():.1f} sd {y.std():.1f}")

    ceiling = {} if args.skip_ceiling else noise_ceiling()
    if ceiling.get("spearman_brown_full_panel_r"):
        print(f"[int] panel noise ceiling: split-half r "
              f"{ceiling['split_half_r']:.3f}, full-panel r "
              f"{ceiling['spearman_brown_full_panel_r']:.3f} "
              f"({ceiling['n_subjects']} subjects)")
    elif ceiling.get("error"):
        print(f"[int] noise ceiling unavailable: {ceiling['error']}")

    # Standardise descriptors; leave fingerprint bits alone. Concentration is
    # kept as its own column so the baselines can use it in isolation.
    desc = X[:, N_BITS:]
    mu, sd = desc.mean(0), desc.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = np.clip(np.hstack([X[:, :N_BITS], (desc - mu) / sd]), -10, 10)
    Xfull = np.hstack([Xs, logc]).astype(np.float32)

    class MLP(nn.Module):
        def __init__(self, d_in, hidden=(512, 256)):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def fit_predict(tr, te, seed):
        torch.manual_seed(seed)
        m = MLP(Xfull.shape[1]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.tensor(Xfull[tr], device=device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
        lossf = nn.MSELoss()
        n, bs = len(xt), 64
        for ep in range(120):
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
        with torch.no_grad():
            return m(torch.tensor(Xfull[te], device=device)).cpu().numpy(), m

    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, te) in enumerate(gkf.split(Xfull, y, groups), 1):
        preds = [fit_predict(tr, te, 100 * k + s)[0] for s in range(args.ensemble)]
        oof[te] = np.mean(preds, axis=0)
        print(f"[int]   fold {k}/{args.folds} ({len(te)} held-out stimuli)",
              flush=True)

    def score(pred, label):
        r = pearsonr(pred, y).statistic
        rho = spearmanr(pred, y).statistic
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        print(f"  {label:34s} r {r:+.4f}   rho {rho:+.4f}   RMSE {rmse:5.1f}")
        return {"pearson": round(float(r), 4), "spearman": round(float(rho), 4),
                "rmse": round(rmse, 2)}

    print("\n[int] out-of-fold, grouped by molecule")
    res = {"model (structure + concentration)": score(oof, "structure + concentration")}

    # Baselines, each fitted the same grouped way.
    from sklearn.linear_model import LinearRegression
    base_sets = {
        "concentration only": logc,
        "volatility proxy (MW, logP)": None,
        "global mean": np.zeros((len(y), 1)),
    }
    descnames = None
    try:
        from rdkit.Chem import Descriptors
        descnames = [n for n, _ in Descriptors._descList]
    except Exception:  # noqa: BLE001
        pass
    if descnames and "MolWt" in descnames and "MolLogP" in descnames:
        cols = [N_BITS + descnames.index("MolWt"), N_BITS + descnames.index("MolLogP")]
        base_sets["volatility proxy (MW, logP)"] = np.hstack([X[:, cols], logc])
    else:
        base_sets.pop("volatility proxy (MW, logP)")

    for label, feats in base_sets.items():
        if feats is None:
            continue
        pred = np.zeros(len(y))
        for tr, te in gkf.split(feats, y, groups):
            lr = LinearRegression().fit(feats[tr], y[tr])
            pred[te] = lr.predict(feats[te])
        res[label] = score(pred, label)

    if ceiling.get("spearman_brown_full_panel_r"):
        print(f"\n  panel ceiling (full-panel split-half r): "
              f"{ceiling['spearman_brown_full_panel_r']:+.4f}")

    # Predict the library at one dilution, for dosing guidance.
    import csv as _csv
    rows = list(_csv.DictReader(
        open(ROOT / args.predict_csv, newline="", encoding="utf-8-sig")))
    names = [r["name"] for r in rows]
    Xp, okp = featurise([r["smiles"] for r in rows])
    names = [names[i] for i in okp]
    dp = (Xp[:, N_BITS:] - mu) / sd
    Xp = np.clip(np.hstack([Xp[:, :N_BITS], dp]), -10, 10)
    Xp = np.hstack([Xp, np.full((len(Xp), 1), np.log10(args.predict_conc))]
                   ).astype(np.float32)

    allidx = np.arange(len(y))
    models = [fit_predict(allidx, allidx[:1], s)[1] for s in range(args.ensemble)]
    with torch.no_grad():
        xt = torch.tensor(Xp, device=device)
        pred = np.mean([m(xt).cpu().numpy() for m in models], axis=0)

    order = np.argsort(-pred)
    print(f"\n[int] predicted intensity for the library at {args.predict_conc:g} "
          f"dilution (0-100)")
    print("  strongest:")
    for i in order[:10]:
        print(f"    {pred[i]:5.1f}  {names[i]}")
    print("  weakest:")
    for i in order[-10:]:
        print(f"    {pred[i]:5.1f}  {names[i]}")

    (ROOT / args.out).write_text(json.dumps({
        "target": "keller_2016 perceived intensity, 0-100",
        "noise_ceiling": ceiling,
        "cv": {"folds": args.folds, "grouped_by": "inchikey",
               "ensemble": args.ensemble},
        "scores": res,
        "predicted_at_concentration": args.predict_conc,
        "molecules": names,
        "predicted_intensity": [round(float(v), 2) for v in pred],
    }, indent=2), encoding="utf-8")
    print(f"\n[int] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
