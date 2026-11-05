"""Learn a space that encodes smell from the 386 receptor responses.

The finding in `scripts/receptor_importance.py` is that single receptors
predict single panel descriptors at median |rho| 0.38 while the whole
386-dimensional distance predicts panel similarity at +0.03. The information
is there and the geometry is what loses it. So stop measuring distance in
receptor space and learn a readout out of it: 386 -> hidden -> 138 descriptors,
and use the 138-dimensional output as the space, where distance means odour
similarity by construction because the axes are odour words.

**The honest constraint, stated first.** Receptor vectors exist for the 396
library molecules only. GS-LF's 4983 would need MolOR run over 4983 x 386
pairs, which is the expensive thing this project has not spent. So this cannot
be compared with the deployed percept model, which saw 4987 molecules. What it
*can* do is settle the question that decides whether that expense is worth it:

    on the same molecules, same labels, same folds, same architecture --
    do receptor features carry odour information that fingerprints do not?

Three feature sets, identical treatment:

    receptors    386 z-scored panel logits
    fingerprint  2048-bit Morgan r=2 + 209 RDKit descriptors, standardised
    both         the two concatenated

If `both` beats `fingerprint`, the panel adds something and computing the 4983
is justified. If it does not, then the receptor model contributes nothing to
predicting smell that structure does not already give, which would be the
cleanest statement of this project's central negative result.

Scored two ways, and the second is the one that counts:

1. out-of-fold ROC-AUC per descriptor, averaged -- in-domain, same labels;
2. Spearman between predicted-vector cosine and **human panel similarity** on
   held-out molecules, using Dravnieks 1985 and Keller 2016, which nothing
   here is fitted on. This is the same external yardstick
   `scripts/receptor_metric.py` used, so the numbers sit beside each other.

Expect little. The partial correlation of receptor distance with smell holding
structure fixed is -0.113, and roughly 374 molecules against 386 features is a
regime where a readout can fit anything and generalise nothing. Ridge with a
cross-validated penalty is used rather than an MLP for exactly that reason; a
`--mlp` flag runs the small network too, for the reader who wants to see it
overfit.

    python scripts/receptor_readout.py
    python scripts/receptor_readout.py --mlp
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit_ridge(Xtr, Ytr, Xte, alphas=ALPHAS, folds=4):
    """Multi-output ridge with the penalty chosen inside the training fold.

    Chosen inside, not once outside, because selecting a penalty on the test
    molecules is the quiet version of the mistake this README keeps finding.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score

    best, best_a = -np.inf, alphas[0]
    inner = KFold(folds, shuffle=True, random_state=0)
    for a in alphas:
        scores = []
        for itr, ite in inner.split(Xtr):
            m = Ridge(alpha=a).fit(Xtr[itr], Ytr[itr])
            p = m.predict(Xtr[ite])
            aucs = [roc_auc_score(Ytr[ite][:, j], p[:, j])
                    for j in range(Ytr.shape[1])
                    if 0 < Ytr[ite][:, j].sum() < len(ite)]
            if aucs:
                scores.append(float(np.mean(aucs)))
        if scores and np.mean(scores) > best:
            best, best_a = float(np.mean(scores)), a
    m = Ridge(alpha=best_a).fit(Xtr, Ytr)
    return m.predict(Xte), best_a


def fit_mlp(Xtr, Ytr, Xte, epochs=300, hidden=256, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(Ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(hidden, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(xt), yt).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(
            net(torch.tensor(Xte, dtype=torch.float32))).numpy()


def flat_keys(entries, smiles_by_name):
    """Stereochemistry-free InChIKeys, matching GS-LF and the panel files.

    Both the GS-LF key column and the ``inchikey`` column of the Dravnieks and
    Keller files are flat -- their second block is the standard no-stereo hash
    ``UHFFFAOYSA``. A store entry's own ``inchikey`` carries stereochemistry,
    so matching on it silently drops every chiral or E/Z molecule: it found
    302 of 396 library molecules in GS-LF where 374 is the true figure, and
    the panel overlaps were understated the same way.
    """
    from eval_similarity import flat_key

    out = []
    for e in entries:
        k = flat_key(smiles_by_name.get(e.name, "") or "")
        out.append(k if k else e.inchikey)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="library")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mlp", action="store_true",
                    help="also run MLPs at each --hidden width")
    ap.add_argument("--hidden", default="64,256,1024",
                    help="hidden widths to sweep when --mlp is given")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import RDLogger
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    from eval_similarity import gslf_truth
    from orvector.config import resolve_paths
    from orvector.store import Library
    from receptor_metric import PANELS, centred_cos, load_panel
    from train_fp_percept import featurise

    RDLogger.DisableLog("rdApp.*")
    paths = resolve_paths()
    entries, logits, _ = Library(paths, name=args.store).load()
    names = [e.name for e in entries]
    smiles = {r["name"].strip(): (r.get("smiles") or "").strip()
              for r in csv.DictReader(io.open(
                  ROOT / "library" / "input" / "expanded_materials.csv",
                  encoding="utf-8"))}
    keys = flat_keys(entries, smiles)
    truth, tasks = gslf_truth()

    use = [i for i, k in enumerate(keys)
           if k in truth and smiles.get(names[i])]
    Y = np.stack([truth[keys[i]] for i in use])
    R = logits[use]
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    F, ok = featurise([smiles[names[i]] for i in use])
    if len(ok) != len(use):
        use = [use[i] for i in ok]
        Y, R = Y[ok], R[ok]
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    print(f"[readout] {len(use)} library molecules with GS-LF labels")
    print(f"           receptors {R.shape[1]}, fingerprint {F.shape[1]}, "
          f"labels {Y.shape[1]}")
    keep = [j for j in range(Y.shape[1]) if 8 <= Y[:, j].sum()]
    Y = Y[:, keep]
    print(f"           {len(keep)} descriptors with at least 8 positives "
          f"kept of {len(tasks)}")

    # "both" appears twice on purpose. A single ridge penalty shared across
    # two blocks of very different width is not a fair test of whether the
    # smaller block helps: 386 receptor columns against 2257 fingerprint ones
    # share one alpha, so the panel can be buried by arithmetic rather than by
    # irrelevance. The balanced variant scales each block by sqrt of its own
    # width first, so the penalty falls on the two equally.
    FEATS = {"receptors": R, "fingerprint": F,
             "both": np.hstack([R, F]),
             "both balanced": np.hstack([R / np.sqrt(R.shape[1]),
                                         F / np.sqrt(F.shape[1])])}

    panels = {}
    for nm, rel in PANELS.items():
        pan, cols = load_panel(ROOT / rel)
        idx = [a for a, i in enumerate(use) if keys[i] in pan]
        if len(idx) < 20:
            continue
        P = np.stack([pan[keys[use[a]]] for a in idx])
        panels[nm] = (np.array(idx), centred_cos(P))
        print(f"           {nm}: {len(idx)} of these have panel data")

    folds = list(KFold(args.folds, shuffle=True, random_state=0)
                 .split(np.arange(len(use))))
    fitters = [("ridge (linear, no hidden layer)", fit_ridge)]
    if args.mlp:
        # Swept rather than assumed. The first version of this script reported
        # one hidden width, 256, which on 374 molecules is already about 99k
        # parameters for the receptor block and 578k for the fingerprint one
        # -- so "the network was too small" was never a plausible reading of
        # the result, and the sweep is here to say so with numbers.
        for h in [int(x) for x in args.hidden.split(",")]:
            fitters.append(
                (f"mlp {h} hidden",
                 (lambda hh: lambda a, b, c: (fit_mlp(a, b, c, hidden=hh),
                                              None))(h)))

    for fname, fitter in fitters:
        print(f"\n=== {fname} ===")
        pred = {k: np.zeros_like(Y, dtype=float) for k in FEATS}
        alphas = {k: [] for k in FEATS}
        for tr, te in folds:
            for k, X in FEATS.items():
                p, a = fitter(X[tr], Y[tr], X[te])
                pred[k][te] = p
                if a is not None:
                    alphas[k].append(a)
        print(f"  {'features':12s} {'OOF AUC':>9s} "
              + "".join(f"{('rho ' + n):>16s}" for n in panels))
        rows = {}
        for k in FEATS:
            aucs = [roc_auc_score(Y[:, j], pred[k][:, j])
                    for j in range(Y.shape[1])
                    if 0 < Y[:, j].sum() < len(Y)]
            line = f"  {k:12s} {np.mean(aucs):9.4f} "
            rho = {}
            for nm, (idx, S) in panels.items():
                V = pred[k][idx]
                C = centred_cos(V)
                iu = np.triu_indices(len(idx), 1)
                rho[nm] = float(spearmanr(C[iu], S[iu]).statistic)
                line += f"{rho[nm]:+16.4f}"
            rows[k] = (float(np.mean(aucs)), rho)
            print(line)
        if alphas["receptors"]:
            print(f"  penalties chosen inside the training folds: "
                  + ", ".join(f"{k} {alphas[k]}" for k in FEATS))

        a_f, r_f = rows["fingerprint"]
        a_r, r_r = rows["receptors"]
        print(f"\n  Does the panel add anything to structure?")
        for which in ("both", "both balanced"):
            a_b, r_b = rows[which]
            print(f"    AUC  {which:14s} - fingerprint  {a_b - a_f:+.4f}")
            for nm in panels:
                print(f"    rho {nm:10s} {which:14s} - fingerprint  "
                      f"{r_b[nm] - r_f[nm]:+.4f}")
        print(f"  And on its own, against structure alone?")
        print(f"    AUC  receptors - fingerprint  {a_r - a_f:+.4f}")
        for nm in panels:
            print(f"    rho {nm:10s} receptors - fingerprint  "
                  f"{r_r[nm] - r_f[nm]:+.4f}")
        print(f"\n  For scale, the raw 386-dim receptor DISTANCE reaches "
              f"+0.03 and +0.02 on\n  these panels "
              f"(scripts/receptor_metric.py), and a readout is being asked "
              f"to beat that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
