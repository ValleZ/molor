"""Per-descriptor prior calibration: give a starved facet as many carriers as
human evidence says it should have.

The model's ranking is right (label=1 materials score far above label=0) but
its scale is compressed for rare labels, so a facet can end up with zero
materials above the 0.7 carrier threshold while the labels name several. This
applies a MONOTONE per-descriptor transform:

    m   = expected carriers, from the label prevalence among library materials
    v_m = the m-th largest prediction for that descriptor
    v >= v_m  ->  0.72 + 0.23 * (v - v_m) / (v_max - v_m)
    v <  v_m  ->  0.72 * v / v_m

monotone and continuous, so within-descriptor ordering is preserved -- which is
what every previous attempt destroyed by flattening to a constant.

Only descriptors that are BOTH starved (fewer than --min-carriers above 0.7)
and evidenced (at least --min-expected carriers) are touched. A healthy facet
is left exactly as it is: that is what keeps the working briefs working.

Run with an interpreter that has rdkit (e.g. .venv-pyrfume).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
GSLF = ("vendor/openpom/openpom/data/curated_datasets/"
        "curated_GS_LF_merged_4983.csv")
MATERIALS = "library/input/expanded_materials.csv"
SKIP = {"nonStereoSMILES", "descriptors", "name", "IsomericSMILES", ""}
TARGET = 0.72          # just above the engine's 0.7 carrier threshold
TOP = 0.95             # value for the strongest carrier


def flat_key(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    key = Chem.MolToInchiKey(m)
    return key.split("-")[0] if key else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--gslf", default=GSLF)
    ap.add_argument("--materials", default=MATERIALS)
    ap.add_argument("--out", default="artifacts/percepts_fp3_calibrated.json")
    ap.add_argument("--min-carriers", type=int, default=5,
                    help="a descriptor with this many materials above 0.7 or "
                         "more is considered healthy and is left alone")
    ap.add_argument("--min-expected", type=int, default=2,
                    help="require the labels to name at least this many "
                         "library materials before a starved facet is lifted")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    descs = list(pj["descriptors"])
    mols = list(pj["molecules"])
    P = np.asarray(pj["probs"], dtype=float)
    di = {d: i for i, d in enumerate(descs)}

    with (ROOT / args.gslf).open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c not in SKIP]
        labels = {}
        for r in rd:
            k = flat_key(r.get("nonStereoSMILES"))
            if k:
                labels.setdefault(k, r)
    smiles = {}
    with (ROOT / args.materials).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            nm = r.get("name") or ""
            smi = r.get("smiles") or ""
            if nm and smi:
                smiles[nm] = smi

    matched = [(i, labels[flat_key(smiles[n])])
               for i, n in enumerate(mols)
               if n in smiles and flat_key(smiles[n]) in labels]
    scale = len(mols) / max(1, len(matched))
    print("[cal] %d molecules, %d matched to GS-LF (scale %.2f)"
          % (len(mols), len(matched), scale))

    calibrated, summary = [], []
    for d in descs:
        if d not in cols:
            continue
        j = di[d]
        m_true = sum(1 for _, row in matched
                     if str(row.get(d, "0")).strip().lower() in
                     ("1", "1.0", "true"))
        m = int(round(m_true * scale))
        carriers = int((P[:, j] > 0.7).sum())
        if m_true < args.min_expected or carriers >= args.min_carriers:
            continue
        order = np.argsort(-P[:, j], kind="stable")
        v_m = float(P[order[min(m, len(mols)) - 1], j])
        v_max = float(P[order[0], j])
        if v_m <= 0.0:
            continue
        col = P[:, j].copy()
        hi = col >= v_m
        new = np.empty_like(col)
        new[hi] = TARGET + (TOP - TARGET) * (col[hi] - v_m) / max(v_max - v_m,
                                                                  1e-9)
        new[~hi] = TARGET * col[~hi] / v_m
        P[:, j] = np.clip(new, 0.0, 1.0)
        calibrated.append(d)
        summary.append((d, m_true, m, carriers,
                        int((P[:, j] > 0.7).sum())))
    print("[cal] calibrated %d descriptors" % len(calibrated))
    print("%-12s %7s %5s %8s %8s"
          % ("facet", "lab=1", "m", "carriers", "carriers'"))
    for d, mt, m, c0, c1 in sorted(summary, key=lambda t: -t[1]):
        print("%-12s %7d %5d %8d %8d" % (d, mt, m, c0, c1))

    if args.dry_run:
        return 0
    out = dict(pj)
    out["checkpoint"] = str(pj.get("checkpoint", "")) + "+prior-calibrated"
    out["probs"] = [[round(float(v), 4) for v in row] for row in P]
    out["calibration"] = {
        "rule": "monotone per-descriptor; m from label prevalence",
        "target": TARGET, "top": TOP,
        "min_carriers": args.min_carriers,
        "min_expected": args.min_expected,
        "descriptors": {d: {"label_ones": mt, "expected_carriers": m,
                            "carriers_before": c0, "carriers_after": c1}
                        for d, mt, m, c0, c1 in summary},
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("[cal] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
