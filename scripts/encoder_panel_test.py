"""GCN against MPNN, judged by human panels the earlier six attempts lacked.

The GCN-vs-MPNN section of the README ends "unresolved after six attempts".
Those six were nearest-neighbour agreement with one perfumer's notes, global
mean distance over descriptor-sharing pairs, family coherence, homologous-series
tightness, a 12-trial blind forced choice, and encoder health -- and every one
of them used either GS-LF-derived labels or a single nose. The two graded panels
in `data/percept/` were built afterwards and have never been pointed at this
question. Dravnieks 1985 and Keller 2016 are ~57-subject psychophysics that
nothing in this repository is fitted on, so they are a seventh attempt with a
judge the first six did not have.

Today's readout experiment produced a hint in passing: receptor-only agreement
with Dravnieks was higher from the GCN store at every readout width. That was
not what the experiment was built to test, the margins sat inside a bootstrap
spread of +-0.05 to +-0.10, and picking it out afterwards is how this project
has fooled itself before. So this script asks the question directly and with
the comparison paired, which is where the power is: both stores hold the same
molecules, so each can be scored on exactly the same pairs and the difference
bootstrapped over molecules.

Two probes, and which is primary is fixed here before running:

    PRIMARY   readout. Ridge 386 -> descriptors, out-of-fold, then cosine
              between predicted descriptor vectors. This measures how much
              odour information the store CONTAINS, which is the question,
              and is not hostage to the global-distance geometry that
              `receptor_metric.py` showed discards most of it.
    SECONDARY distance. PCA-20 of the z-scored logits, Euclidean -- the
              deployed atlas pipeline exactly. This measures what the atlas
              actually shows a viewer.

PASS CONDITION, registered before the first run: one encoder is better than
the other only if the paired bootstrap difference excludes zero **on both
panels** for the primary probe. Anything else is reported as unresolved, which
would make it seven attempts.

Bootstrap resamples MOLECULES, not pairs. Pairs from a shared molecule are
dependent, so resampling pairs would understate the interval -- the same
mistake that makes 1225 pairs from 50 molecules look like 1225 observations.

    python scripts/encoder_panel_test.py
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

STORES = (("MPNN", "library_mpnn_clean"), ("GCN", "library"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import RDLogger
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold

    from eval_similarity import flat_key, gslf_truth
    from orvector.config import resolve_paths
    from orvector.store import Library
    from receptor_metric import PANELS, centred_cos, load_panel
    from receptor_readout import fit_ridge

    RDLogger.DisableLog("rdApp.*")
    paths = resolve_paths()
    smiles = {r["name"].strip(): (r.get("smiles") or "").strip()
              for r in csv.DictReader(io.open(
                  ROOT / "library" / "input" / "expanded_materials.csv",
                  encoding="utf-8"))}
    truth, _ = gslf_truth()

    loaded = {}
    for label, store in STORES:
        entries, logits, _ = Library(paths, name=store).load()
        keys = [flat_key(smiles.get(e.name, "") or "") or e.inchikey
                for e in entries]
        loaded[label] = ([e.name for e in entries], keys, logits)
        print(f"[enc] {label:5s} {store:20s} {logits.shape[0]} x "
              f"{logits.shape[1]}")

    # The two stores must hold the same molecules in the same order for the
    # comparison to be paired at all.
    n0, k0, _ = loaded["MPNN"]
    n1, k1, _ = loaded["GCN"]
    if n0 != n1:
        common = [n for n in n0 if n in set(n1)]
        print(f"      stores differ; using the {len(common)} in both")
        for label in loaded:
            nm, kk, lg = loaded[label]
            idx = [nm.index(c) for c in common]
            loaded[label] = (common, [kk[i] for i in idx], lg[idx])
    else:
        print(f"      both stores hold the same {len(n0)} molecules in order")

    names, keys, _ = loaded["MPNN"]

    def z(lg):
        return (lg - lg.mean(0)) / (lg.std(0) + 1e-12)

    def dist_probe(lg, sel):
        red = PCA(n_components=20, random_state=0).fit_transform(
            z(lg)[sel] - z(lg)[sel].mean(0))
        return squareform(pdist(red))

    # The readout is fitted once, on every labelled molecule in the library,
    # and only then restricted to a panel's molecules. Fitting it on the
    # panel subset alone is the version this script ran first, and it is much
    # weaker: Dravnieks overlaps 50 molecules, of which only 10 descriptors
    # clear 8 positives, so the primary probe was being starved before it was
    # asked anything. Out-of-fold predictions keep it honest either way.
    lab = [i for i, k in enumerate(keys) if k in truth]
    Yall = np.stack([truth[keys[i]] for i in lab])
    Yall = Yall[:, [j for j in range(Yall.shape[1]) if Yall[:, j].sum() >= 8]]
    print(f"      readout fitted on {len(lab)} labelled molecules, "
          f"{Yall.shape[1]} descriptors, out-of-fold")
    at = {m: a for a, m in enumerate(lab)}

    def readout_pred(lg):
        X = z(lg)[lab]
        pred = np.zeros_like(Yall, dtype=float)
        for tr, te in KFold(args.folds, shuffle=True,
                            random_state=0).split(np.arange(len(lab))):
            p, _ = fit_ridge(X[tr], Yall[tr], X[te])
            pred[te] = p
        return pred

    def readout_probe(pred, sel):
        rows = [at[i] for i in sel]
        return -centred_cos(pred[rows])  # negate: a distance, like the other

    preds = {label: readout_pred(loaded[label][2]) for label, _ in STORES}

    rng = np.random.default_rng(0)
    verdict = {}
    for panel, rel in PANELS.items():
        pan, _ = load_panel(ROOT / rel)
        sel = [i for i, k in enumerate(keys) if k in pan and i in at]
        if len(sel) < 20:
            print(f"\n[{panel}] only {len(sel)} molecules, skipped")
            continue
        P = np.stack([pan[keys[i]] for i in sel])
        S = centred_cos(P)
        m = len(sel)
        print(f"\n[{panel}] {m} molecules in both stores and the panel, "
              f"{m * (m - 1) // 2} pairs")

        D = {}
        for label, _ in STORES:
            _, _, lg = loaded[label]
            D[(label, "distance")] = dist_probe(lg, sel)
            D[(label, "readout")] = readout_probe(preds[label], sel)

        iu = np.triu_indices(m, 1)
        print(f"   {'probe':10s} {'MPNN':>9s} {'GCN':>9s} "
              f"{'GCN - MPNN':>12s} {'paired 95% over molecules':>28s}")
        for probe in ("readout", "distance"):
            a = -spearmanr(D[("MPNN", probe)][iu], S[iu]).statistic
            b = -spearmanr(D[("GCN", probe)][iu], S[iu]).statistic
            diffs = []
            for _ in range(args.boot):
                bs = rng.integers(0, m, m)
                # Drop self-pairs created by resampling with replacement.
                ii, jj = np.triu_indices(m, 1)
                pi, pj = bs[ii], bs[jj]
                ok = pi != pj
                sy = S[pi[ok], pj[ok]]
                if sy.std() < 1e-12:
                    continue
                ra = -spearmanr(D[("MPNN", probe)][pi[ok], pj[ok]],
                                sy).statistic
                rb = -spearmanr(D[("GCN", probe)][pi[ok], pj[ok]],
                                sy).statistic
                diffs.append(rb - ra)
            diffs = np.array(diffs)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            mark = "" if lo <= 0 <= hi else ("  GCN" if lo > 0 else "  MPNN")
            print(f"   {probe:10s} {a:+9.4f} {b:+9.4f} {b - a:+12.4f} "
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>28s}{mark}")
            if probe == "readout":
                verdict[panel] = (lo, hi)

    print(f"\nPass condition was: primary probe (readout) excludes zero on "
          f"BOTH panels.")
    for panel, (lo, hi) in verdict.items():
        side = ("GCN" if lo > 0 else "MPNN" if hi < 0 else "neither")
        print(f"   {panel:10s} [{lo:+.3f}, {hi:+.3f}]  -> {side}")
    sides = {("GCN" if lo > 0 else "MPNN" if hi < 0 else None)
             for lo, hi in verdict.values()}
    if len(verdict) == len(PANELS) and len(sides) == 1 and None not in sides:
        print(f"\n   RESOLVED in favour of {sides.pop()}.")
    else:
        print(f"\n   UNRESOLVED. That makes seven attempts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
