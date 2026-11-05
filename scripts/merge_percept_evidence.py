"""Fuse human GS-LF labels into a percept artifact.  Labels first, raise-only.

The percept vectors are a model's predictions. Where a human label exists it is
better evidence than the model's hedged prediction -- but GS-LF labels are
binary, and a 0 there means "the panellist did not use that word", not
"intensity zero" (sandalwood is 0.7% of the training set, so absence of the
word is the normal case). So this merge:

  label == 1  -> raise the cell to the descriptor's P95 over the library
  label == 0  -> leave the model value alone
  no GS-LF row -> leave the model value alone

Raising is what fixes the downstream problem: the mixer can only voice a facet
that many materials carry at >=0.7, and the model never emits >=0.5 for a rare
label, so those facets have no carriers at all.

Output keeps the percept-file schema (checkpoint, descriptors, molecules, probs)
so every `--percepts` consumer works unchanged, plus an "evidence" block and a
sibling .manifest.json recording provenance and the counts.

Run with an interpreter that has rdkit (e.g. .venv-pyrfume).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
GSLF = ("vendor/openpom/openpom/data/curated_datasets/"
        "curated_GS_LF_merged_4983.csv")
MATERIALS = "library/input/expanded_materials.csv"
SKIP = {"nonStereoSMILES", "descriptors", "name", "IsomericSMILES", ""}
WATCH = ["sandalwood", "orris", "peach", "rummy", "cucumber", "ozone",
         "tobacco", "smoky", "vanilla", "clean", "floral", "green", "woody"]


def flat_key(smiles: str) -> str | None:
    """Stereochemistry-free InChIKey, as GS-LF's nonStereoSMILES requires."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    key = Chem.MolToInchiKey(m)
    return key.split("-")[0] if key else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(path: Path) -> tuple[list[str], dict[str, dict]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c not in SKIP]
        rows = list(rd)
    lab = {}
    for r in rows:
        k = flat_key(r.get("nonStereoSMILES"))
        if k:
            lab.setdefault(k, r)
    return cols, lab


def load_smiles(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            name = r.get("name") or r.get("Name")
            smi = r.get("smiles") or r.get("SMILES") or ""
            if name and smi:
                out[name] = smi
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_expanded.json")
    ap.add_argument("--gslf", default=GSLF)
    ap.add_argument("--materials", default=MATERIALS)
    ap.add_argument("--out", default="artifacts/percepts_merged.json")
    ap.add_argument("--present", default="p95",
                    help="value a label=1 cell is raised to: 'p95' or a float")
    args = ap.parse_args()

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    descs = list(pj["descriptors"])
    mols = list(pj["molecules"])
    probs = [list(map(float, row)) for row in pj["probs"]]
    di = {d: i for i, d in enumerate(descs)}

    label_cols, labels = load_labels(ROOT / args.gslf)
    smiles = load_smiles(ROOT / args.materials)
    common = [c for c in label_cols if c in di]
    print(f"[merge] percept {len(mols)} molecules x {len(descs)} descriptors; "
          f"GS-LF {len(labels)} keys, {len(common)} descriptors shared")

    # present value per descriptor: P95 of the library distribution, or a float
    if args.present == "p95":
        present = {}
        for d in common:
            col = sorted(probs[r][di[d]] for r in range(len(mols)))
            present[d] = col[min(len(col) - 1, int(0.95 * len(col)))]
    else:
        val = float(args.present)
        present = {d: val for d in common}
    print("[merge] raise target: %s (%s)"
          % (args.present,
             ", ".join("%s %.2f" % (d, present[d]) for d in WATCH
                       if d in present)[:150]))

    before = {d: sum(1 for r in range(len(mols)) if probs[r][di[d]] > 0.7)
              for d in common}
    raised = {d: 0 for d in common}
    matched = 0
    for r, name in enumerate(mols):
        key = flat_key(smiles.get(name, ""))
        row = labels.get(key) if key else None
        if row is None:
            continue
        matched += 1
        for d in common:
            if str(row.get(d, "0")).strip().lower() not in ("1", "1.0", "true"):
                continue
            j = di[d]
            if probs[r][j] < present[d]:
                probs[r][j] = present[d]
                raised[d] += 1
    after = {d: sum(1 for r in range(len(mols)) if probs[r][di[d]] > 0.7)
             for d in common}

    print("[merge] materials matched to a GS-LF row: %d, cells raised: %d"
          % (matched, sum(raised.values())))
    print("%-12s %7s %7s %7s" % ("facet", "raised", ">0.7was", ">0.7now"))
    for d in WATCH:
        if d in common:
            print("%-12s %7d %7d %7d"
                  % (d, raised[d], before[d], after[d]))

    out = dict(pj)
    out["checkpoint"] = str(pj.get("checkpoint", "")) + "+gslf-labels"
    out["probs"] = probs
    out["evidence"] = {
        "rule": "label==1 -> max(model, %s); label==0 -> model untouched"
                % args.present,
        "source_labels": args.gslf,
        "materials_matched": matched,
        "cells_raised": sum(raised.values()),
        "raised_by_descriptor": {d: raised[d] for d in common if raised[d]},
        "present_values": {d: round(present[d], 6) for d in common},
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/merge_percept_evidence.py",
        "inputs": {k: {"path": v, "sha256": sha256(ROOT / v),
                       "bytes": (ROOT / v).stat().st_size}
                   for k, v in (("percepts", args.percepts),
                                ("gslf", args.gslf),
                                ("materials", args.materials))},
        "output": {"path": args.out, "bytes": outp.stat().st_size},
        "shapes": {"molecules": len(mols), "descriptors": len(descs),
                   "shared_descriptors": len(common)},
        "rule": {"present": args.present,
                 "raise_only": True,
                 "zero_label": "model value kept (absence of a word is not "
                               "absence of a smell)"},
        "counts": {"materials_matched": matched,
                   "cells_raised": sum(raised.values()),
                   "carriers_above_0.7_before": before,
                   "carriers_above_0.7_after": after},
        "not_covered": "naturals and captives with no GS-LF row keep model "
                       "values: the merge helps molecules, not oils",
    }
    mpath = outp.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[merge] wrote %s and %s" % (args.out, mpath.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
