"""Strength of each word on each material, beside the percept model.

    .venv\\Scripts\\python.exe scripts\\build_strength.py

The percept model (artifacts/percepts_fp4.json) answers "would a describer use
this word for this material". It is trained on yes/no word lists and knows
nothing of how strong a facet is: tobacco on Damascone Beta reads 0.98 and is a
faint nuance. This script leaves that model alone and writes a second matrix,
artifacts/strength_fp4.json, with the same molecules and words.

Strength is fitted on the only graded truth here, the Dravnieks panel (percent
applicability, 0 to 100, of each word on 127 molecules that also have a GS-LF
row). Inputs per cell:
  - the model's opinion of the cell. For a molecule with a GS-LF row this is
    its OUT-OF-FOLD prediction (artifacts/percepts_fp4_oof.npz, written by
    train_fp_v3.py --pos-weight --oof-out): the shipped number has memorised
    the row and says 0.98 for every labelled word;
  - whether the row has the word;
  - votes for the word from the other sources (data/percept/gslf_votes.csv).
A molecule without a row has no label and no votes, so it gets a second fit on
the model's opinion alone. How common the word is in GS-LF is an optional
fourth input (--commonness), off by default: see that flag.

artifacts/strength_fp4.json holds the expected panel rating (the panel's
strongest words on a material rate 30 to 60; under 5 is a trace).
artifacts/percepts_fp4s.json holds the same on a 0-to-1 scale in the percept
schema, and is what the atlas and the simplifier read since 2026-11-11
(docs/PERCEPT_EVIDENCE.md, "fp4s").
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_similarity import flat_key  # noqa: E402

GSLF = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"


KNEE, CEIL = 0.7, 0.98


def soft_cap(v):
    """Identity up to KNEE, then a smooth bend that approaches CEIL: order
    is kept and nothing reaches 1."""
    v = np.asarray(v, dtype=float)
    span = CEIL - KNEE
    return np.where(v <= KNEE, v, KNEE + span * (1 - np.exp(-(v - KNEE) / span)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def model(cst):
    from sklearn.ensemble import HistGradientBoostingRegressor
    # early_stopping=False: left on "auto" it holds out a random tenth of the
    # cells once there are more than 10,000 (there are 10,414), and two runs
    # of this script then disagreed by up to 9 rating points on a cell.
    return HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                         learning_rate=0.05, monotonic_cst=cst,
                                         early_stopping=False, random_state=0)


def panel_cells(voc, row_of):
    """(gslf row, word index, panel rating) for every panel molecule with a
    GS-LF row and every vocabulary word the Dravnieks crosswalk reaches."""
    dra = pd.read_csv(ROOT / "data/percept/dravnieks.csv")
    dra["flat"] = [flat_key(s) for s in dra["smiles"]]
    cw = pd.read_csv(ROOT / "data/percept/crosswalk_dravnieks_to_vocab.csv")
    by_word: dict[str, list[str]] = {}
    for s, v in zip(cw["dravnieks"], cw["vocab"]):
        c = [c for c in dra.columns if c.strip().lower() == s.strip().lower()]
        assert c, s
        by_word.setdefault(v, []).append(c[0])
    mean = dra.dropna(subset=["flat"]).groupby("flat")[
        sorted({c for cs in by_word.values() for c in cs})].mean()
    col = {w: j for j, w in enumerate(voc)}
    out = []
    for k, r in mean.iterrows():
        if k in row_of:
            for v, cs in by_word.items():
                out.append((row_of[k], col[v], max(r[c] for c in cs)))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp4.json")
    ap.add_argument("--oof", default="artifacts/percepts_fp4_oof.npz")
    ap.add_argument("--out", default="artifacts/strength_fp4.json")
    ap.add_argument("--percepts-out", default="artifacts/percepts_fp4s.json",
                    help="the same strengths as a percept file (0 to 1, the "
                         "schema every --percepts consumer reads)")
    ap.add_argument("--commonness", action="store_true",
                    help="add how common the word is as an input. It fits the "
                         "held-out panel better (0.62 against 0.37) because it "
                         "learns the panel's habit of rating everyday words "
                         "(sweet, green, woody) high and rare ones low, and "
                         "for the same reason it sinks specific facets such as "
                         "black currant. Off since 2026-11-11: the perfumer "
                         "chose the fit without it by eye on the atlas")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    d = json.load(open(ROOT / args.percepts))
    names, voc = d["molecules"], d["descriptors"]
    P = np.array(d["probs"], dtype=float)
    gs = pd.read_csv(ROOT / GSLF)
    assert [c for c in gs.columns if c not in ("nonStereoSMILES", "descriptors")] == voc
    Y = gs[voc].to_numpy(float)
    V = pd.read_csv(ROOT / "data/percept/gslf_votes.csv")[voc].to_numpy(float)
    z = np.load(ROOT / args.oof)
    assert (z["rows"] == np.arange(len(Y))).all()
    O = z["oof"].astype(float)
    common = np.log(Y.mean(0) + 1e-4)
    row_of = {}
    for i, s in enumerate(gs["nonStereoSMILES"].astype(str)):
        k = flat_key(s)
        if k:
            row_of.setdefault(k, i)

    def feats(op, lab, vot, j):
        return np.column_stack([logit(op), lab, vot, common[j]])

    c = panel_cells(voc, row_of)
    i, j, y = c[:, 0].astype(int), c[:, 1].astype(int), c[:, 2]
    X = feats(O[i, j], Y[i, j], V[i, j], j)
    full, full_cst, bare, bare_cst = [0, 1, 2, 3], [1, 1, 1, 0], [0, 3], [1, 0]
    if not args.commonness:
        full, full_cst, bare, bare_cst = [0, 1, 2], [1, 1, 1], [0], [1]

    # held-out check, by molecule
    pf, pb = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(10).split(X, y, i):
        pf[te] = model(full_cst).fit(X[tr][:, full], y[tr]).predict(X[te][:, full])
        pb[te] = model(bare_cst).fit(X[tr][:, bare], y[tr]).predict(X[te][:, bare])

    def within(score, mask):
        r = []
        for g in np.unique(i):
            m = (i == g) & mask
            if m.sum() >= 4 and y[m].std() > 0 and score[m].std() > 0:
                r.append(spearmanr(score[m], y[m]).statistic)
        return float(np.mean(r))

    shown = O[i, j] > 0.5
    check = {
        "panel_molecules": int(len(np.unique(i))), "cells": int(len(y)),
        "what": "mean within-molecule Spearman against the panel rating, over "
                "the words the model shows (out-of-fold above 0.5), molecules "
                "held out of the fit",
        "percept_number": round(within(O[i, j], shown), 3),
        "strength_with_row": round(within(pf, shown), 3),
        "strength_without_row": round(within(pb, shown), 3),
    }
    print(json.dumps(check, indent=1))

    mf = model(full_cst).fit(X[:, full], y)
    mb = model(bare_cst).fit(X[:, bare], y)
    smi = {r["name"]: r["smiles"] for r in csv.DictReader(
        open(ROOT / "library/input/expanded_materials.csv", encoding="utf-8-sig"))}
    S = np.zeros_like(P)
    has_row = []
    jj = np.arange(len(voc))
    for n, name in enumerate(names):
        k = flat_key(smi[name]) if name in smi else None
        g = row_of.get(k)
        has_row.append(g is not None)
        if g is None:
            S[n] = mb.predict(np.column_stack([logit(P[n]), common])[:, :len(bare)])
        else:
            S[n] = mf.predict(feats(O[g], Y[g], V[g], jj)[:, full])
    S = np.clip(S, 0, 100)
    (ROOT / args.out).write_text(json.dumps({
        "what": "expected Dravnieks panel rating (percent applicability, "
                "0-100) of each word on each material; a companion to the "
                "percept file, not a replacement",
        "percepts": args.percepts, "check": check,
        "word_commonness_input": args.commonness,
        "descriptors": voc, "molecules": names, "has_gslf_row": has_row,
        "strength": [[round(float(v), 2) for v in r] for r in S],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.out}: {S.shape[0]} x {S.shape[1]}, "
          f"{sum(has_row)} molecules with a GS-LF row")

    # The percept file. One unit = the strongest word of a typical material
    # (the median over materials of the largest rating), so a leading facet
    # reads near 1. Above KNEE the value bends toward CEIL instead of being
    # cut at 1: a hard cut ties every strong word of a material at 1.0
    # (pear = sweet = fruity on Hexyl Acetate), and the engine mixes in
    # log-odds, where 1.0 is 13.8 and outvotes everything.
    unit = float(np.median(S.max(1)))
    Q = soft_cap(S / unit)
    meta = {k: d[k] for k in ("oof_mean_auc", "similarity_spearman") if k in d}
    (ROOT / args.percepts_out).write_text(json.dumps({
        "checkpoint": "fp4 strength (%s; panel-fitted weights on fp4)"
                      % d.get("checkpoint", args.percepts),
        **meta,
        "config": {"base_percepts": args.percepts, "strength": args.out,
                   "unit_panel_rating": round(unit, 3), "knee": KNEE,
                   "ceil": CEIL, "word_commonness_input": args.commonness},
        "descriptors": voc, "molecules": names,
        "probs": [[round(float(v), 4) for v in r] for r in Q],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.percepts_out}: unit {unit:.2f}, mean {Q.mean():.4f}, "
          f"max {Q.max():.4f}, words above 0.5 per material "
          f"{(Q > 0.5).sum(1).mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
