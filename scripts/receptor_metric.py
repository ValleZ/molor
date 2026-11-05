"""Can reweighting the 386 receptors recover odour structure the global
distance discards? Pre-registered, and judged by panels nothing here fitted.

``scripts/musk_receptor.py`` shows that OR5AN1 is the panel's best musk
detector and that deleting it changes every pairwise distance by 0.09%. One
informative column cannot survive an unweighted Euclidean norm over 386. The
obvious response is to weight the columns. This script tests whether that
works, and is written to be able to fail.

Why it might not. To weight receptors you need to know which ones matter, and
that needs odour supervision. Fit the weights against odour labels and the
labels may be doing all the work, in which case you have built a worse percept
model with 386 extra steps. The README was caught by exactly this once: a
7.7-vs-52.8 percentile result that turned out to be two structure-based models
agreeing with each other. So:

* **Weights are learned from GS-LF label overlap only** -- human binary tags.
* **Judgement is by graded human panels that nothing here was fitted on**:
  Dravnieks 1985 (percent applicability over a panel, 146 descriptors, 43
  molecules in this library) and Keller 2016 (mean 0-100 ratings from ~57
  subjects, 20 descriptors, 99 molecules here). Published psychophysics, not
  one person's bench notes.
* **The split is on materials, not on pairs.** A test material's pairs never
  informed the weights. Splitting pairs would leak: material A's weight
  contribution would be learned from A's other pairs.

Pre-registered pass condition, fixed before the first run: a learned weighting
must beat the unweighted baseline on held-out materials, judged by the panel,
**on both Dravnieks and Keller**, by more than the bootstrap spread. One of two
is not a result; this project has been burned by picking the dataset that
agreed.

Also reported, because it is the number that decides whether any of this
matters: ECFP4 Tanimoto on the same held-out pairs. The receptor panel earns
its place only by beating structure, which it is not currently expected to do
-- the partial correlation of receptor distance with smell holding structure
fixed is -0.113, and that is the ceiling on everything here.

    python scripts/receptor_metric.py
    python scripts/receptor_metric.py --store library     # the GCN check
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

PANELS = {"dravnieks": "data/percept/dravnieks.csv",
          "keller": "data/percept/keller.csv"}

#: Columns in a panel file that are not descriptor ratings.
PANEL_META = {"inchikey", "smiles", "name", "concentration", "in_library",
              "solvent_", "n_subjects", "intensity", "pleasantness",
              "familiarity"}


def load_panel(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """InChIKey -> graded descriptor vector, averaged over concentrations.

    A molecule appears once per concentration. Averaging is the honest default
    here: this script compares *molecules*, and nothing in the receptor vector
    knows about concentration, so picking one level would be an arbitrary
    choice dressed as a measurement.
    """
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    cols = [c for c in rows[0]
            if c not in PANEL_META and not c.endswith("__sd")]
    acc: dict[str, list[np.ndarray]] = {}
    for r in rows:
        key = (r.get("inchikey") or "").strip()
        if not key:
            continue
        v = np.array([float(r[c]) if (r.get(c) or "").strip() else 0.0
                      for c in cols], float)
        acc.setdefault(key, []).append(v)
    return {k: np.mean(vs, axis=0) for k, vs in acc.items()}, cols


def centred_cos(V: np.ndarray) -> np.ndarray:
    """Cosine after centring each descriptor across molecules.

    Uncentred, every panel vector correlates with every other through the
    shared base rate -- the same reason library percepts are read as lift.
    """
    V = V - V.mean(0, keepdims=True)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return (V / n) @ (V / n).T


def upper(M: np.ndarray) -> np.ndarray:
    i = np.triu_indices(M.shape[0], 1)
    return M[i]


def fit_weights(Z: np.ndarray, S: np.ndarray, mode: str) -> np.ndarray:
    """Per-receptor weights from label similarity on the training materials.

    ``corr``: how much each receptor's absolute difference tracks label
    dissimilarity, kept where the sign is right. Cheap, convex, no tuning.

    ``metric``: a diagonal metric fitted by gradient descent to maximise the
    (linear) correlation between weighted distance and label dissimilarity.
    Weights are squared so they stay non-negative.
    """
    n = Z.shape[0]
    iu = np.triu_indices(n, 1)
    d = np.abs(Z[iu[0]] - Z[iu[1]])            # (pairs, receptors)
    y = 1.0 - S[iu]                            # label dissimilarity
    yc = y - y.mean()
    if mode == "corr":
        dc = d - d.mean(0, keepdims=True)
        num = dc.T @ yc
        den = np.linalg.norm(dc, axis=0) * np.linalg.norm(yc) + 1e-12
        w = np.maximum(num / den, 0.0)
        return w / (w.sum() + 1e-12) * len(w)
    if mode == "metric":
        import torch

        D = torch.tensor(d ** 2, dtype=torch.float32)
        Y = torch.tensor(yc, dtype=torch.float32)
        Y = Y / (Y.norm() + 1e-12)
        # Initialised at 1, not 0. With w = a**2, the gradient at a = 0 is
        # exactly 0, so a zero init never moves and the "learned" metric
        # silently stays uniform -- which is what the first run of this script
        # reported, as a delta of +0.0000 against the baseline on both panels.
        a = torch.ones(Z.shape[1], requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            w = a ** 2 + 1e-6
            dist = torch.sqrt(D @ w + 1e-9)
            dist = dist - dist.mean()
            loss = -(dist @ Y) / (dist.norm() + 1e-12)
            loss.backward()
            opt.step()
        w = (a.detach() ** 2 + 1e-6).numpy()
        return w / (w.sum() + 1e-12) * len(w)
    raise ValueError(mode)


def weighted_dist(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    V = Z * np.sqrt(w)[None, :]
    g = (V * V).sum(1)
    D = g[:, None] + g[None, :] - 2 * V @ V.T
    return np.sqrt(np.maximum(D, 0.0))


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
    ap.add_argument("--reps", type=int, default=40,
                    help="random half splits of the materials")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs
    from scipy.stats import spearmanr

    from eval_similarity import gslf_truth, jaccard_matrix
    from orvector.config import resolve_paths
    from orvector.store import Library

    RDLogger.DisableLog("rdApp.*")
    paths = resolve_paths()
    entries, logits, _ = Library(paths, name=args.store).load()
    names = [e.name for e in entries]
    smiles = {r["name"].strip(): (r.get("smiles") or "").strip()
              for r in csv.DictReader(io.open(
                  ROOT / "library" / "input" / "expanded_materials.csv",
                  encoding="utf-8"))}
    keys = flat_keys(entries, smiles)
    Zall = (logits - logits.mean(0)) / (logits.std(0) + 1e-12)
    print(f"[metric] {args.store}: {len(names)} materials x "
          f"{logits.shape[1]} receptors")

    truth, _ = gslf_truth()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    for panel_name, rel in PANELS.items():
        pan, pcols = load_panel(ROOT / rel)
        # Materials that have a panel vector AND a GS-LF label (needed to
        # learn the weights) AND a parseable structure (for the control).
        use = [i for i, k in enumerate(keys)
               if k in pan and k in truth and smiles.get(names[i])]
        if len(use) < 20:
            print(f"\n[{panel_name}] only {len(use)} usable materials, skipped")
            continue
        Z = Zall[np.ix_(use, np.arange(Zall.shape[1]))]
        P = np.stack([pan[keys[i]] for i in use])
        Y = np.stack([truth[keys[i]] for i in use])
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(smiles[names[i]]))
               for i in use]
        Spanel = centred_cos(P)
        Slabel = jaccard_matrix(Y)
        n = len(use)
        print(f"\n[{panel_name}] {n} materials in common, "
              f"{n * (n - 1) // 2} pairs, {len(pcols)} panel descriptors")

        # Repeated random halves rather than 5-fold. Only pairs with BOTH
        # materials held out can be scored, so a 5-fold split leaves about a
        # fifth of the pairs and, on Dravnieks, 189 of them -- too few to
        # separate anything. Halving gives the largest held-out pair set a
        # clean split allows, and repeating it 40 times gives a paired sample
        # to compare methods on.
        METHODS = ("unweighted", "corr", "metric", "tanimoto")
        per_split = {k: [] for k in METHODS}
        rng = np.random.default_rng(0)
        npairs = []
        for rep in range(args.reps):
            perm = rng.permutation(n)
            tr, te = perm[: n // 2], perm[n // 2:]
            Ztr, Zte = Z[tr], Z[te]
            Str = Slabel[np.ix_(tr, tr)]
            ws = {"unweighted": np.ones(Z.shape[1]),
                  "corr": fit_weights(Ztr, Str, "corr"),
                  "metric": fit_weights(Ztr, Str, "metric")}
            iu = np.triu_indices(len(te), 1)
            y = Spanel[np.ix_(te, te)][iu]
            npairs.append(len(y))
            for k in METHODS:
                if k == "tanimoto":
                    tan = np.array([[DataStructs.TanimotoSimilarity(
                        fps[te[a]], fps[te[b]]) for b in range(len(te))]
                        for a in range(len(te))])
                    x = 1.0 - tan[iu]
                else:
                    x = upper(weighted_dist(Zte, ws[k]))
                # Negated: these are distances and the panel gives
                # similarity, so a working metric gives a NEGATIVE Spearman.
                # Reported positive throughout, so larger is always better.
                per_split[k].append(-spearmanr(x, y).statistic)
        print(f"  {args.reps} random half splits, "
              f"{int(np.mean(npairs))} held-out pairs each "
              f"(both materials outside the half the weights were fitted on)")

        print(f"\n  {'metric':12s} {'median Spearman':>16s} "
              f"{'2.5-97.5 over splits':>23s}")
        for k in METHODS:
            v = np.array(per_split[k])
            lo, hi = np.percentile(v, [2.5, 97.5])
            tag = ("  <- current pipeline" if k == "unweighted" else
                   "  <- structure only" if k == "tanimoto" else "")
            print(f"  {k:12s} {np.median(v):+16.4f} "
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>23s}{tag}")

        # Paired across splits: each method saw exactly the same halves, so
        # the per-split difference is the comparison with the least noise in
        # it. This is what the pass condition is read off.
        print(f"\n  paired against the unweighted baseline, "
              f"per split:")
        b = np.array(per_split["unweighted"])
        for k in ("corr", "metric", "tanimoto"):
            d = np.array(per_split[k]) - b
            lo, hi = np.percentile(d, [2.5, 97.5])
            wins = 100.0 * float((d > 0).mean())
            verdict = ("better" if lo > 0 else
                       "worse" if hi < 0 else "indistinguishable")
            print(f"    {k:12s} mean delta {d.mean():+.4f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  wins {wins:3.0f}% of splits"
                  f"  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
