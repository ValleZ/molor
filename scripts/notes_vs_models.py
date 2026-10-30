"""Score each percept model against the perfumer's own smelling notes.

Every other evaluation here asks about *geometry* -- whether materials that
smell alike sit near each other. This asks the more direct question: when the
perfumer writes that something smells green and waxy, does the model put green
and waxy near the top of its 138 predicted descriptors for that molecule?

These notes are the most relevant ground truth available for this library. They
cover the actual materials in use, they are perceptual rather than structural,
and no model here was trained on them. They are also the weakest instrument in
the project: one nose, free text, and a keyword vocabulary that was written to
match this person's wording rather than a standard lexicon. Treat differences of
a few points as noise.

The metric is per-material ranking. For one molecule, the model emits 138
probabilities; the perfumer's note names a handful of descriptors. AUC asks
whether those named descriptors are ranked above the unnamed ones, which is
invariant to a model's overall calibration -- a model that predicts everything
at 0.9 is not rewarded.

The baseline is the important part. Odour descriptors are wildly unbalanced
(fruity 38%, green 29%, chamomile 0.6%), and these notes mention common
qualities more often than rare ones, so a model that ignores the molecule
entirely and always predicts the GS-LF base rates will score well above 0.5.
That prior, not chance, is what a model has to beat to have learned anything
about these particular molecules.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_impressions import VOCAB  # noqa: E402

#: The note vocabulary was written against this person's wording, so six of its
#: tags are not GS-LF descriptor names. Five map cleanly. "marine" (seaweed,
#: kelp, salty water) is dropped: GS-LF's nearest is "ozone", which is a
#: different quality, and inventing the correspondence would manufacture
#: agreement or disagreement that is not in the data.
TAG_TO_GSLF = {
    "animalic": "animal",
    "dusty": "powdery",
    "anise": "anisic",
    "spice": "spicy",
    "sugary": "sweet",
}
DROPPED = {"marine"}

MODELS = [
    ("FP v1", "artifacts/percepts_fp.json"),
    ("FP v2", "artifacts/percepts_fp2.json"),
    ("FP v3", "artifacts/percepts_fp3.json"),
    ("FP v3 hierarchical", "artifacts/percepts_fp3h.json"),
    ("OpenPOM", "artifacts/percepts_openpom.json"),
    ("DenseSense", "artifacts/percepts_densesense.json"),
    ("GS-LF MPNN", "artifacts/percepts_expanded.json"),
]


def note_descriptors() -> dict[str, set[str]]:
    """Material -> the GS-LF descriptor names its note implies."""
    path = ROOT / "library" / "input" / "impressions.tsv"
    out = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8"), delimiter="\t"):
        low = r["impression"].lower()
        tags = {d for d, keys in VOCAB.items() if any(k in low for k in keys)}
        tags -= DROPPED
        mapped = {TAG_TO_GSLF.get(t, t) for t in tags}
        if mapped:
            out[r["material"]] = mapped
    return out


def base_rates(descriptors: list[str]) -> np.ndarray:
    """GS-LF label frequencies, as a molecule-independent prediction."""
    import pandas as pd

    path = ("vendor/openpom/openpom/data/curated_datasets/"
            "curated_GS_LF_merged_4983.csv")
    df = pd.read_csv(ROOT / path)
    return np.array([float(df[d].mean()) if d in df.columns else 0.0
                     for d in descriptors])


def auc_within(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Rank-AUC of the named descriptors against the unnamed, one molecule."""
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U, ties counted as half.
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-descriptor", action="store_true",
                    help="also break results down by descriptor")
    ap.add_argument("--out", default="artifacts/notes_vs_models.json")
    args = ap.parse_args()

    notes = note_descriptors()
    print(f"[notes] {len(notes)} materials have a note implying at least one "
          f"GS-LF descriptor")
    used = sorted({d for v in notes.values() for d in v})
    print(f"[notes] vocabulary in play ({len(used)}): {used}")

    loaded = []
    for label, path in MODELS:
        p = ROOT / path
        if not p.is_file():
            continue
        pj = json.loads(p.read_text(encoding="utf-8"))
        loaded.append((label, pj,
                       {n: np.asarray(r, float)
                        for n, r in zip(pj["molecules"], pj["probs"])}))

    # Score every model on the same materials. The percept files were produced
    # at different times -- Dense Sense only covers molecules with precomputed
    # electron densities, and the files predating the library clean-up use the
    # old dilution-bearing names -- so coverage ranges from 88 to 117 materials.
    # Letting each model use its own set would compare different questions.
    common = sorted(set(notes) & set.intersection(*[set(P) for _, _, P in loaded]))
    print(f"[notes] {len(common)} of them are covered by every model, "
          f"which is the set scored below")

    report, rows = {}, []
    for label, pj, P in loaded:
        desc = pj["descriptors"]
        idx = {d: i for i, d in enumerate(desc)}

        aucs, precs, n_mat = [], [], 0
        per_desc: dict[str, list[float]] = {}
        for mat in common:
            tags = notes[mat]
            have = [t for t in tags if t in idx]
            if not have:
                continue
            mask = np.zeros(len(desc), dtype=bool)
            for t in have:
                mask[idx[t]] = True
            a = auc_within(P[mat], mask)
            if a is None:
                continue
            aucs.append(a)
            n_mat += 1
            k = len(have)
            topk = set(np.argsort(-P[mat])[:k].tolist())
            precs.append(len(topk & {idx[t] for t in have}) / k)
            for t in have:
                r = int((P[mat] > P[mat][idx[t]]).sum()) + 1
                per_desc.setdefault(t, []).append(r)

        if not aucs:
            continue
        rows.append((label, n_mat, float(np.mean(aucs)), float(np.mean(precs))))
        report[label] = {"materials": n_mat,
                         "mean_auc": round(float(np.mean(aucs)), 4),
                         "mean_precision_at_k": round(float(np.mean(precs)), 4)}
        if args.per_descriptor:
            report[label]["median_rank_by_descriptor"] = {
                t: int(np.median(v)) for t, v in sorted(per_desc.items())}

    # Molecule-independent prior, evaluated identically.
    pj = json.loads((ROOT / MODELS[2][1]).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    idx = {d: i for i, d in enumerate(desc)}
    prior = base_rates(desc)
    aucs, precs = [], []
    for mat in common:
        tags = notes[mat]
        have = [t for t in tags if t in idx]
        if not have:
            continue
        mask = np.zeros(len(desc), dtype=bool)
        for t in have:
            mask[idx[t]] = True
        a = auc_within(prior, mask)
        if a is not None:
            aucs.append(a)
            k = len(have)
            topk = set(np.argsort(-prior)[:k].tolist())
            precs.append(len(topk & {idx[t] for t in have}) / k)
    rows.append(("base rates (no molecule)", len(aucs), float(np.mean(aucs)),
                 float(np.mean(precs))))
    report["base_rates"] = {"materials": len(aucs),
                            "mean_auc": round(float(np.mean(aucs)), 4),
                            "mean_precision_at_k": round(float(np.mean(precs)), 4)}

    print(f"\n  {'model':26s} {'mats':>5s} {'mean AUC':>9s} {'prec@k':>8s}")
    for label, n, a, pr in sorted(rows, key=lambda r: -r[2]):
        print(f"  {label:26s} {n:5d} {a:9.4f} {pr:8.3f}")
    print("\n  chance is 0.500; the base-rate row is what a model must beat to")
    print("  have learned anything molecule-specific.")

    if args.per_descriptor:
        best = max((r for r in rows if "base rates" not in r[0]), key=lambda r: r[2])
        md = report[best[0]].get("median_rank_by_descriptor", {})
        print(f"\n  median rank out of {len(desc)} descriptors, {best[0]}:")
        for t, r in sorted(md.items(), key=lambda kv: kv[1]):
            print(f"    {t:12s} {r:4d}")

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[notes] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
