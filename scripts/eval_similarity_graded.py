"""Rank representations against graded human profiles, not the training labels.

Every similarity number in this project so far was measured against GS-LF
Jaccard overlap, which has two defects that flatter and distort in unknown
proportions:

*It is the training target.* Each percept model was fitted on these labels, so
scoring their geometry against the same labels rewards fitting them, not
smelling right.

*The labels are binary and their hierarchy is unenforced.* 165 molecules carry
a fruit tag without the fruity tag, so Jaccard treats banana and apple as no
more similar than banana and woody, and cannot distinguish faint from intense.

Dravnieks and Keller fix both. Neither was used to train anything here, both are
graded, and both come from different panels with different vocabularies, so
agreement with them is evidence about smell rather than about GS-LF. They are
smaller -- 43 and 99 of the library's 301 materials -- so the pair counts are
lower and the noise higher; that is the price of an honest instrument.

Ground truth is the correlation between two molecules' continuous descriptor
profiles. Spearman is used rather than cosine because the panels' scales are not
linear: a descriptor at 40% applicability is not twice one at 20%.

Molecules measured at more than one concentration are averaged by default, since
the models have no concentration input and cannot distinguish the two. The
--concentration flag scores a single level instead, which matters because odour
character genuinely changes with dilution.
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

from eval_similarity import (centred_cosine, jaccard_matrix, load_ecfp,  # noqa: E402
                             load_npz, load_percept, material_keys)

DATA = ROOT / "data" / "percept"
#: Columns that are not descriptors.
KELLER_DROP = {"intensity", "pleasantness", "familiarity", "n_subjects"}


def graded_truth(which: str, concentration: str | None):
    """Material name -> continuous descriptor profile, from a human panel."""
    path = DATA / f"{which}.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run scripts/build_percept_datasets.py")
    df = pd.read_csv(path)
    df = df[df["in_library"].notna() & (df["in_library"].astype(str) != "")]
    if concentration:
        df = df[df["concentration"].astype(str) == concentration]
    drop = {"inchikey", "smiles", "name", "concentration", "in_library",
            "solvent_"} | KELLER_DROP
    desc = [c for c in df.columns if c not in drop and not c.endswith("__sd")]
    if not desc:
        raise SystemExit(f"no descriptor columns found in {path.name}")
    # One profile per material, averaged over concentrations and duplicates.
    prof = df.groupby("in_library")[desc].mean()
    return {str(k): v.to_numpy(float) for k, v in prof.iterrows()}, desc


def spearman_profile_matrix(P: np.ndarray) -> np.ndarray:
    """Rank-correlation between every pair of profiles."""
    from scipy.stats import rankdata

    R = np.vstack([rankdata(row) for row in P])
    R = R - R.mean(axis=1, keepdims=True)
    n = np.linalg.norm(R, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    R = R / n
    return R @ R.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percept", action="append", default=[])
    ap.add_argument("--embedding", action="append", default=[])
    ap.add_argument("--ecfp", action="store_true")
    ap.add_argument("--truth", default="dravnieks,keller")
    ap.add_argument("--concentration", default=None,
                    help="score one concentration level instead of averaging")
    ap.add_argument("--bootstrap", type=int, default=400,
                    help="resamples over materials for the CI")
    ap.add_argument("--out", default="artifacts/similarity_graded.json")
    args = ap.parse_args()

    from scipy.stats import spearmanr

    sources: list[tuple[str, dict]] = []
    for spec in args.percept:
        label, _, p = spec.partition("=")
        sources.append((label, load_percept(ROOT / p)))
    for spec in args.embedding:
        label, _, p = spec.partition("=")
        sources.append((label, load_npz(ROOT / p)))
    if args.ecfp:
        sources.append(("ECFP4 (raw)", load_ecfp()))

    report = {}
    for which in [t.strip() for t in args.truth.split(",") if t.strip()]:
        truth, desc = graded_truth(which, args.concentration)
        # Score every model on the same materials. The percept files were built
        # at different times, and the ones predating the library clean-up use
        # the old dilution-bearing names, so they cover fewer materials. Letting
        # each model use whatever it happens to cover would compare different
        # pair sets and could invent a winner.
        common = sorted(set(truth) & set.intersection(
            *[set(v) for _, v in sources]) if sources else [])
        print(f"\n=== {which}: {len(truth)} library materials have profiles, "
              f"{len(common)} covered by every model, "
              f"{len(desc)} graded descriptors"
              + (f", concentration {args.concentration}" if args.concentration else
                 ", concentrations averaged") + " ===")
        missing = {label: sorted(set(truth) - set(v))[:4]
                   for label, v in sources if set(truth) - set(v)}
        for label, ex in missing.items():
            print(f"    {label} lacks {len(set(truth) - dict(sources)[label].keys())}"
                  f" of them, e.g. {ex}")
        report[which] = {"materials_with_profiles": len(truth),
                         "materials_scored": len(common),
                         "descriptors": len(desc)}

        rng = np.random.default_rng(0)
        for label, vecs in sources:
            names = [n for n in common if n in vecs]
            if len(names) < 8:
                print(f"  {label:24s} only {len(names)} overlap; skipped")
                continue
            T = spearman_profile_matrix(np.stack([truth[n] for n in names]))
            V = centred_cosine(np.stack([vecs[n] for n in names]))
            iu = np.triu_indices(len(names), k=1)
            rho = float(spearmanr(V[iu], T[iu]).statistic)

            # Pairs are not independent -- 30 materials generate 435 of them --
            # so resample materials, not pairs, or the interval is far too tight.
            boot = []
            for _ in range(args.bootstrap):
                idx = rng.integers(0, len(names), len(names))
                sub = np.ix_(idx, idx)
                keep = np.triu_indices(len(idx), k=1)
                t, v = T[sub][keep], V[sub][keep]
                ok = np.isfinite(t) & np.isfinite(v)
                if ok.sum() > 10 and np.std(t[ok]) > 0 and np.std(v[ok]) > 0:
                    boot.append(spearmanr(v[ok], t[ok]).statistic)
            lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20
                      else (np.nan, np.nan))
            report[which][label] = {"materials": len(names),
                                    "pairs": int(len(iu[0])),
                                    "spearman": round(rho, 4),
                                    "ci95": [round(float(lo), 4), round(float(hi), 4)]}
            print(f"  {label:24s} {len(names):4d} mats  {len(iu[0]):6d} pairs  "
                  f"rho {rho:+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[graded] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
