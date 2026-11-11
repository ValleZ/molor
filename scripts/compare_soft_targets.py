"""Report numbers for the soft-target candidates (steps 6 and 7 of the task).

Run with .venv:  python scripts/compare_soft_targets.py
Reads artifacts/percepts_fp4.json and the candidates percepts_fp6{a,b,c}.json,
artifacts/temp/fp4_seedshift.json (fp4's recipe with --seed-offset 7000), the out-of-fold dumps
artifacts/temp/oof_*.npz written by train_fp_v3.py --oof-out, the targets, and
the votes written by scripts/build_soft_targets.py.
Writes artifacts/temp/soft_targets_compare.md. It judges nothing: every figure
is measured and printed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_similarity import flat_key  # noqa: E402

MODELS = {"fp4": "artifacts/percepts_fp4.json",
          "A": "artifacts/percepts_fp6a.json",
          "B": "artifacts/percepts_fp6b.json",
          "C": "artifacts/percepts_fp6c.json"}
OOF = {"fp4": "oof_fp4", "A": "oof_fp6a", "B": "oof_fp6b", "C": "oof_fp6c"}
TARGET = {"A": "gslf_soft_targets.csv", "B": "gslf_soft_targets_B.csv",
          "C": "gslf_soft_targets_B.csv"}
GATE = ["Damascone Beta", "Cassifix", "Phenethyl Alcohol", "Hedione",
        "Iso E Super", "Linalool", "Vanillin", "Ambroxan",
        "Gamma Undecalactone", "Galaxolide"]
NOSE = {
    "Damascone Beta": "berry, fruity, tea, black currant present; dry plum "
                      "somewhat; tobacco about 0.25 if at all",
    "Cassifix": "woody plus amber",
    "Phenethyl Alcohol": "a complex smell of many flowers",
}
SEED = "artifacts/temp/fp4_seedshift.json"
GSLF = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
BANDS = ((0, 0, "0"), (1, 1, "1"), (2, 99, "2+"))
WBINS = ((1, 5, "under 5"), (5, 15, "5 to 15"), (15, 35, "15 to 35"),
         (35, 49.99, "35 to 50"), (49.99, 51, "at the cap 50"))


def load(p):
    d = json.load(open(ROOT / p))
    return d, np.array(d["probs"], dtype=float)


def main() -> int:
    ds, P = {}, {}
    for k, p in MODELS.items():
        ds[k], P[k] = load(p)
    dseed, S = load(SEED)
    names, voc = ds["fp4"]["molecules"], ds["fp4"]["descriptors"]
    for k in P:
        assert ds[k]["molecules"] == names and ds[k]["descriptors"] == voc
    ix = {n: i for i, n in enumerate(names)}
    col = {w: j for j, w in enumerate(voc)}
    cand = [k for k in P if k != "fp4"]
    L = []
    w = L.append

    def row(label, f, with_seed=True):
        cells = [f(ds[k], P[k]) for k in P]
        if with_seed:
            cells.append(f(dseed, S))
        w(f"| {label} | " + " | ".join(cells) + " |")

    # ---- numbers -----------------------------------------------------------
    w("### Aggregate numbers (measured)\n")
    w("| | " + " | ".join(P) + " | fp4, other seeds |\n|" + "---|" * (len(P) + 2))
    row("out-of-fold mean AUC", lambda d, p: f"{d['oof_mean_auc']:.4f}")
    row("label-grounded similarity rho",
        lambda d, p: f"{d['similarity_spearman']:+.4f}")
    row("mean prediction (703 x 138)", lambda d, p: f"{p.mean():.4f}")
    row("descriptors with >= 5 library materials above 0.7",
        lambda d, p: f"{int(((p > 0.7).sum(0) >= 5).sum())} of 138")
    row("descriptors with >= 5 library materials above 0.5",
        lambda d, p: f"{int(((p > 0.5).sum(0) >= 5).sum())} of 138")
    row("mean number of words above 0.5 per material",
        lambda d, p: f"{(p > 0.5).sum(1).mean():.2f}")
    row("mean number of words above 0.3 per material",
        lambda d, p: f"{(p > 0.3).sum(1).mean():.2f}")
    row("materials with >= 10 words above 0.5",
        lambda d, p: f"{int(((p > 0.5).sum(1) >= 10).sum())}")
    row("cells above 0.9", lambda d, p: f"{int((p > 0.9).sum())}")

    # ---- training data on the library molecules ----------------------------
    gs = pd.read_csv(ROOT / GSLF)
    Y = gs[voc].to_numpy(float)
    votes = np.load(ROOT / "artifacts/temp/votes.npy")
    src_v = np.load(ROOT / "artifacts/temp/src_votes.npy")
    first_v = np.load(ROOT / "artifacts/temp/first_votes.npy")
    covered = np.load(ROOT / "artifacts/temp/covered.npy")
    gkeys = {}
    for i, s in enumerate(gs["nonStereoSMILES"].astype(str)):
        k = flat_key(s)
        if k:
            gkeys.setdefault(k, i)
    smi = {r["name"]: r["smiles"] for r in csv.DictReader(
        open(ROOT / "library/input/expanded_materials.csv", encoding="utf-8-sig"))}
    g_of = {}      # library index -> gslf row
    for n, i in ix.items():
        k = flat_key(smi[n]) if n in smi else None
        if k in gkeys:
            g_of[i] = gkeys[k]

    # ---- cells moved -------------------------------------------------------
    sd = np.abs(S - P["fp4"])
    floors = {"mean": float(sd.mean()), "p95": float(np.percentile(sd, 95)),
              "p99": float(np.percentile(sd, 99))}
    w("\n### Cells moved against fp4 (measured)\n")
    w("Floor: fp4's recipe with every seed shifted by 7000 "
      "(`--seed-offset 7000`); its cells differ from fp4 by "
      + ", ".join(f"{k} {v:.3f}" for k, v in floors.items())
      + f", max {sd.max():.3f}.\n")
    w("| moved by more than | " + " | ".join(f"{k} up | {k} down" for k in cand)
      + " | seeds up | seeds down |\n|" + "---|" * (2 * len(cand) + 3))
    for t in (floors["mean"], floors["p95"], floors["p99"], 0.3):
        r = []
        for k in cand:
            dlt = P[k] - P["fp4"]
            r += [int((dlt > t).sum()), int((dlt < -t).sum())]
        dlt = S - P["fp4"]
        r += [int((dlt > t).sum()), int((dlt < -t).sum())]
        w(f"| {t:.3f} | " + " | ".join(map(str, r)) + " |")

    # what the cells that rose by more than 0.3 are
    Lab = np.full(P["fp4"].shape, -1.0)
    Vot = np.zeros(P["fp4"].shape)
    for i, g in g_of.items():
        Lab[i], Vot[i] = Y[g], votes[g]
    w("\nCells that rose by more than 0.3, by what the training data says of "
      "them:\n")
    w("| | " + " | ".join(cand) + " | seeds |\n|" + "---|" * (len(cand) + 2))
    kinds = (("labelled", lambda u: u & (Lab == 1)),
             ("not labelled, has a vote", lambda u: u & (Lab == 0) & (Vot > 0)),
             ("not labelled, no vote", lambda u: u & (Lab == 0) & (Vot == 0)),
             ("molecule has no GS-LF row", lambda u: u & (Lab < 0)))
    for nm, f in kinds:
        w(f"| {nm} | " + " | ".join(
            str(int(f((q - P["fp4"]) > 0.3).sum()))
            for q in [P[k] for k in cand] + [S]) + " |")

    # ---- out-of-fold, per class, all training rows --------------------------
    O = {}
    for k, f in OOF.items():
        z = np.load(ROOT / f"artifacts/temp/{f}.npz")
        assert (z["rows"] == np.arange(len(Y))).all()
        O[k] = z["oof"].astype(float)
    Tg = {k: pd.read_csv(ROOT / "data/percept" / f)[voc].to_numpy(float)
          for k, f in TARGET.items()}
    cov2 = np.broadcast_to(covered[:, None], Y.shape)
    pos = Y.sum(0)
    W = np.broadcast_to(np.clip((len(Y) - pos) / np.maximum(pos, 1), 1, 50),
                        Y.shape)
    classes = []
    for lo, hi, nm in BANDS:
        classes.append((f"label 1, covered, {nm} votes",
                        (Y == 1) & cov2 & (votes >= lo) & (votes <= hi)))
    classes.append(("label 1, not covered, not first-listed",
                    (Y == 1) & ~cov2 & (first_v == 0)))
    classes.append(("label 1, not covered, first-listed",
                    (Y == 1) & ~cov2 & (first_v == 1)))
    for lo, hi, nm in BANDS:
        classes.append((f"label 0, {nm} votes",
                        (Y == 0) & (votes >= lo) & (votes <= hi)))
    w("\n### Mean prediction per target class, out of fold (measured)\n")
    w(f"All {Y.shape[0]} training rows x 138 words; every prediction is made "
      "by models that did not train on the row. Target A / B is what the "
      "class was trained toward (C uses B's).\n")
    w("| class | cells | target A | target B | " + " | ".join(O) + " |\n|"
      + "---|" * (4 + len(O)))
    for nm, m in classes:
        w(f"| {nm} | {int(m.sum())} | {Tg['A'][m].mean():.2f} | "
          f"{Tg['B'][m].mean():.2f} | "
          + " | ".join(f"{O[k][m].mean():.3f}" for k in O) + " |")
    a = (Y == 1) & cov2 & (votes >= 2)
    b = (Y == 1) & cov2 & (votes == 0)
    w("\nLabelled, covered cells: mean prediction with 2+ votes minus with 0 "
      "votes: " + ", ".join(f"{k} {O[k][a].mean() - O[k][b].mean():+.3f}"
                            for k in O) + ".")

    w("\n### The same, by how rare the word is (out of fold, measured)\n")
    w("pos_weight of the word = unlabelled rows / labelled rows, capped at 50; "
      f"{int((W[0] >= 50).sum())} of 138 words sit at the cap.\n")
    for title, m0 in (("Labelled cells with no vote and not first-listed",
                       (Y == 1) & (votes == 0)),
                      ("Labelled cells with 2+ votes", (Y == 1) & (votes >= 2)),
                      ("Unlabelled cells with a vote", (Y == 0) & (votes >= 1))):
        w(f"\n{title}:\n")
        w("| pos_weight | cells | " + " | ".join(O) + " |\n|" + "---|" * (2 + len(O)))
        for lo, hi, nm in WBINS:
            m = m0 & (W >= lo) & (W < hi)
            if m.sum():
                w(f"| {nm} | {int(m.sum())} | "
                  + " | ".join(f"{O[k][m].mean():.3f}" for k in O) + " |")

    # ---- gate sheet ---------------------------------------------------------
    w("\n### Gate sheet: every word above 0.5 in any of the models\n")
    w("Values are measured. `data` is what the training data says of the "
      "cell: L = GS-LF label, a digit = source votes, F = first-listed, "
      "`-` = nothing. **new** marks a word with no label and no vote that "
      "rose by more than 0.3 over fp4 in some candidate.\n")
    for n in GATE:
        if n not in ix:
            w(f"- {n}: not in the prediction set")
            continue
        i = ix[n]
        g = g_of.get(i)
        w(f"\n#### {n}" + (f" (perfumer: {NOSE[n]})" if n in NOSE else ""))
        if g is None:
            w("No GS-LF row: nothing in the targets for this molecule.\n")
        else:
            w(f"GS-LF row: {', '.join(v for v, y in zip(voc, Y[g]) if y > 0)}. "
              + ("Covered by at least one of the four sources."
                 if covered[g] else "In none of the four sources.") + "\n")
        words = sorted({voc[j] for k in P for j in np.where(P[k][i] > 0.5)[0]},
                       key=lambda x: -P["fp4"][i][col[x]])
        w("| word | data | " + " | ".join(P) + " |\n|" + "---|" * (2 + len(P)))
        for x in words:
            j = col[x]
            tag = ""
            if g is not None:
                tag = (("L" if Y[g, j] else "")
                       + (str(int(src_v[g, j])) if src_v[g, j] else "")
                       + ("F" if first_v[g, j] else "")) or "-"
                if (tag == "-" and max(P[k][i, j] for k in cand)
                        - P["fp4"][i, j] > 0.3):
                    tag = "**new**"
            w(f"| {x} | {tag} | " + " | ".join(
                f"{P[k][i][j]:.2f}" for k in P) + " |")
        w("| *words above 0.5* | | " + " | ".join(
            str(int((P[k][i] > 0.5).sum())) for k in P) + " |")

    # ---- Damascone Beta: the cell the task is about -------------------------
    i = ix["Damascone Beta"]
    w("\n### Damascone Beta, the words the perfumer placed (measured)\n")
    w("| | " + " | ".join(P) + " |\n|" + "---|" * (1 + len(P)))
    for x in ("tobacco", "plum", "tea", "black currant", "berry", "fruity"):
        w(f"| {x} | " + " | ".join(f"{P[k][i][col[x]]:.2f}" for k in P) + " |")
    w("| tobacco's rank among the 138 words | " + " | ".join(
        str(int((P[k][i] > P[k][i][col['tobacco']]).sum()) + 1) for k in P) + " |")
    w("| tobacco divided by fruity | " + " | ".join(
        f"{P[k][i][col['tobacco']] / P[k][i][col['fruity']]:.2f}" for k in P) + " |")

    out = ROOT / "artifacts/temp/soft_targets_compare.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
