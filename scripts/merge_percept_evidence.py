"""Fuse human evidence into a percept artifact, best evidence first.

Three sources, in order of precision, applied per CELL:

  1. Dravnieks 1985 panel ratings (percent applicability, 2146 distinct values)
     -- graded, so it can RAISE or LOWER. Only positive ratings are applied: a
     blank means the word was not applied, not that the smell is absent.
  2. GS-LF binary labels -- raise-only, to --present (default 0.9). A 1 means
     "this material does smell of it"; a 0 means the panellist did not use the
     word, so the model value is kept (sandalwood's base rate is 0.7%, so
     absence of the word is the normal case). Skipped where graded evidence
     already spoke for the cell.
  3. The model prediction, untouched, everywhere else (all oils and captives
     have no GS-LF row and are unaffected).

Why raise at all: the model never emits >= 0.5 for a rare label, so facets like
peach/orris/rummy have no materials above 0.7 and the mixer -- which averages
logits -- cannot voice them from a formula at all.

Output keeps the percept-file schema (checkpoint, descriptors, molecules,
probs) so every `--percepts` consumer works unchanged, plus an "evidence"
block, and writes a sibling .manifest.json with provenance and counts.

Run with an interpreter that has rdkit (e.g. .venv-pyrfume).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
GSLF = ("vendor/openpom/openpom/data/curated_datasets/"
        "curated_GS_LF_merged_4983.csv")
MATERIALS = "library/input/expanded_materials.csv"
DRAVNIEKS = "data/percept/dravnieks.csv"
CROSSWALK = "data/percept/crosswalk_dravnieks_to_vocab.csv"
SKIP = {"nonStereoSMILES", "descriptors", "name", "IsomericSMILES", ""}
META = {"inchikey", "smiles", "name", "concentration", "in_library"}
WATCH = ["sandalwood", "orris", "peach", "rummy", "cucumber", "ozone",
         "tobacco", "smoky", "vanilla", "clean", "cedar", "floral", "woody"]


def flat_key(smiles: str) -> str | None:
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


def load_gslf(path: Path) -> tuple[list[str], dict[str, dict]]:
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


def load_graded(path: Path, cross: Path, known: set[str]):
    """flatkey -> {vocab descriptor: value 0-1}, max over panel/levels."""
    if not path.is_file() or not cross.is_file():
        return {}, 0
    xwalk: dict[str, list[str]] = {}
    with cross.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            a = (r.get("dravnieks") or "").strip().upper()
            v = (r.get("vocab") or "").strip()
            if a and v and v in known:
                xwalk.setdefault(a, []).append(v)
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = {c.strip().upper(): c for c in rd.fieldnames
                if c.strip().upper() not in {m.upper() for m in META}}
        pairs = [(cols[a], vs) for a, vs in xwalk.items() if a in cols]
        for r in rd:
            k = (r.get("inchikey") or "").split("-")[0]
            if not k:
                continue
            for col, vocabs in pairs:
                try:
                    v = float(r.get(col))
                except (TypeError, ValueError):
                    continue
                if v <= 0.0:
                    continue          # not applied is not the same as zero
                for d in vocabs:
                    slot = out.setdefault(k, {})
                    slot[d] = max(slot.get(d, 0.0), min(1.0, v / 100.0))
    return out, len(pairs)


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
    ap.add_argument("--dravnieks", default=DRAVNIEKS)
    ap.add_argument("--crosswalk", default=CROSSWALK)
    ap.add_argument("--out", default="artifacts/percepts_merged.json")
    ap.add_argument("--present", default=0.9,
                    help="value a GS-LF label=1 cell is raised to")
    ap.add_argument("--no-graded", action="store_true",
                    help="skip the Dravnieks panel stage (labels only)")
    ap.add_argument("--only-starved", type=int, default=None, metavar="N",
                    help="apply evidence ONLY to descriptors with fewer than N "
                         "carriers above 0.7 in this artifact")
    ap.add_argument("--max-mean", type=float, default=None, metavar="M",
                    help="apply evidence ONLY to descriptors whose MEAN "
                         "prediction is below M -- the signature of label "
                         "compression. Preferred over --only-starved, whose "
                         "carrier counts are computed on this artifact alone "
                         "and so call citrus/spicy starved when their carriers "
                         "are the oils the database adds.")
    args = ap.parse_args()

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    descs = list(pj["descriptors"])
    mols = list(pj["molecules"])
    probs = [list(map(float, row)) for row in pj["probs"]]
    di = {d: i for i, d in enumerate(descs)}

    label_cols, labels = load_gslf(ROOT / args.gslf)
    smiles = load_smiles(ROOT / args.materials)
    common = [c for c in label_cols if c in di]
    graded, n_pairs = ({}, 0) if args.no_graded else load_graded(
        ROOT / args.dravnieks, ROOT / args.crosswalk, set(common))
    print("[merge] percept %d x %d; GS-LF %d keys (%d shared); "
          "graded: %d crosswalk columns, %d molecules"
          % (len(mols), len(descs), len(labels), len(common), n_pairs,
             len(graded)))

    before = {d: sum(1 for r in range(len(mols)) if probs[r][di[d]] > 0.7)
              for d in common}
    allow = None
    if args.only_starved is not None:
        allow = {d for d in common if before.get(d, 0) < args.only_starved}
        print("[merge] starved-only: %d of %d descriptors eligible (<%d "
              "carriers)" % (len(allow), len(common), args.only_starved))
    if args.max_mean is not None:
        mean = {d: sum(probs[r][di[d]] for r in range(len(mols))) / len(mols)
                for d in common}
        allow = {d for d in common if mean[d] < args.max_mean}
        print("[merge] compressed-only (mean < %.3f): %d of %d descriptors: "
              "%s" % (args.max_mean, len(allow), len(common),
                       sorted(allow)))
    g_up = {d: 0 for d in common}
    g_dn = {d: 0 for d in common}
    lab_up = {d: 0 for d in common}
    matched = g_matched = 0
    present = float(args.present)

    for r, name in enumerate(mols):
        key = flat_key(smiles.get(name, ""))
        g = graded.get(key) if key else None
        if g:
            g_matched += 1
            for d, v in g.items():
                if allow is not None and d not in allow:
                    continue
                j = di[d]
                if probs[r][j] < v:
                    g_up[d] += 1
                elif probs[r][j] > v:
                    g_dn[d] += 1
                probs[r][j] = v
        row = labels.get(key) if key else None
        if row is None:
            continue
        matched += 1
        for d in common:
            if g and d in g:
                continue          # graded evidence already spoke for this cell
            if allow is not None and d not in allow:
                continue          # descriptor is healthy; leave it alone
            if str(row.get(d, "0")).strip().lower() not in ("1", "1.0", "true"):
                continue
            j = di[d]
            if probs[r][j] < present:
                probs[r][j] = present
                lab_up[d] += 1

    after = {d: sum(1 for r in range(len(mols)) if probs[r][di[d]] > 0.7)
             for d in common}
    print("[merge] materials: %d graded, %d GS-LF; cells: %d graded-up, "
          "%d graded-down, %d label-raised"
          % (g_matched, matched, sum(g_up.values()), sum(g_dn.values()),
             sum(lab_up.values())))
    print("%-12s %7s %7s %8s %8s %8s"
          % ("facet", "gr-up", "gr-down", "lab-up", ">0.7 was", ">0.7 now"))
    for d in WATCH:
        if d in common:
            print("%-12s %7d %7d %8d %8d %8d"
                  % (d, g_up[d], g_dn[d], lab_up[d], before[d], after[d]))

    out = dict(pj)
    out["checkpoint"] = str(pj.get("checkpoint", "")) + "+evidence"
    out["probs"] = probs
    out["evidence"] = {
        "precedence": ["dravnieks_panel", "gslf_label", "model"],
        "graded_rule": "positive panel value / 100 applied in both directions",
        "label_rule": "label==1 -> max(model, %.2f); label==0 -> model kept"
                      % present,
        "materials_matched": {"graded": g_matched, "gslf": matched},
        "cells": {"graded_raised": sum(g_up.values()),
                  "graded_lowered": sum(g_dn.values()),
                  "label_raised": sum(lab_up.values())},
        "graded_raised_by_descriptor": {d: g_up[d] for d in common if g_up[d]},
        "graded_lowered_by_descriptor": {d: g_dn[d] for d in common if g_dn[d]},
        "label_raised_by_descriptor": {d: lab_up[d] for d in common
                                       if lab_up[d]},
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    inputs = {"percepts": args.percepts, "gslf": args.gslf,
              "materials": args.materials}
    if not args.no_graded:
        inputs.update({"dravnieks": args.dravnieks,
                       "crosswalk": args.crosswalk})
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/merge_percept_evidence.py",
        "inputs": {k: {"path": v, "sha256": sha256(ROOT / v),
                       "bytes": (ROOT / v).stat().st_size}
                   for k, v in inputs.items()},
        "output": {"path": args.out, "bytes": outp.stat().st_size},
        "shapes": {"molecules": len(mols), "descriptors": len(descs),
                   "shared_descriptors": len(common),
                   "graded_crosswalk_columns": n_pairs},
        "rule": {"precedence": ["dravnieks_panel", "gslf_label", "model"],
                 "present": present,
                 "graded_applies_positive_only": True,
                 "zero_label": "model value kept (absence of a word is not "
                               "absence of a smell)"},
        "counts": {"materials_graded": g_matched, "materials_gslf": matched,
                   "cells_graded_raised": sum(g_up.values()),
                   "cells_graded_lowered": sum(g_dn.values()),
                   "cells_label_raised": sum(lab_up.values()),
                   "carriers_above_0.7_before": before,
                   "carriers_above_0.7_after": after},
        "not_covered": "naturals and captives with no GS-LF row keep model "
                       "values: the merge helps molecules, not oils",
        "vocabulary_gap": "Dravnieks names 37 of our 138 descriptors exactly "
                          "and ~30 more through the crosswalk; sandalwood, "
                          "orris, ozone, clean, rummy and green are absent from "
                          "the ASTM word list entirely",
    }
    mpath = outp.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[merge] wrote %s and %s" % (args.out, mpath.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
