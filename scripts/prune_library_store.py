#!/usr/bin/env python3
"""
Reconcile artifacts/library with library/input/expanded_materials.csv.

The receptor store is append-only (`orvector library add` has no prune), so a row
edited or removed in the CSV leaves the store stale. Gate 3 in
ADDING_MATERIALS.md is equality between provenance.json's n_molecules and the CSV
row count; this restores it without rescoring 700 molecules (~45 minutes).

    .venv/Scripts/python.exe scripts/prune_library_store.py --dry-run
    .venv/Scripts/python.exe scripts/prune_library_store.py

Reconciliation is by MOLECULE IDENTITY, not by name. A name-keyed prune cannot
cope with a corrected structure: when "Rose Oxide" was fixed from the 5-methyl
regiomer (ZBCCUWZRJMGIBB) to the real racemate (CZCBTSFUTPZVKJ), the store held
the new structure under the old row's sibling name, and a name prune would have
kept the wrong isomer and dropped the right one. Here the expected id->name map
comes from the CSV's own SMILES, so a rename and a re-identification both fall out
of one rule: keep rows whose id is expected, drop the rest, repair the name.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STORE = ROOT / "artifacts" / "library"
LIBRARY = ROOT / "library" / "input" / "expanded_materials.csv"


def csv_identities() -> dict[str, str]:
    """InChIKey -> name, straight from the CSV's SMILES."""
    identities: dict[str, str] = {}
    with io.open(LIBRARY, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            mol = Chem.MolFromSmiles((row.get("smiles") or "").strip())
            if mol is None:
                continue
            try:
                identities[Chem.MolToInchiKey(mol)] = row["name"].strip()
            except Exception:  # noqa: BLE001
                continue
    return identities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    expected = csv_identities()
    print(f"csv molecules   : {len(expected)} (by InChIKey)")

    lines = (STORE / "molecules.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0]
    rows = [line for line in lines[1:] if line.strip()]
    parsed = list(csv.DictReader(io.StringIO("\n".join([header] + rows)), delimiter="\t"))

    keep, drop, rename = [], [], []
    for index, row in enumerate(parsed):
        mid = row["molecule_id"]
        if mid not in expected:
            drop.append((index, row["name"], mid))
            continue
        keep.append(index)
        if row["name"].strip() != expected[mid]:
            rename.append((index, row["name"], expected[mid]))

    print(f"store molecules : {len(parsed)}")
    print(f"  keep          : {len(keep)}")
    print(f"  drop          : {len(drop)}")
    for _, name, mid in drop:
        print(f"    {name:<32} {mid}")
    print(f"  rename        : {len(rename)}")
    for _, old, new in rename:
        print(f"    {old!r} -> {new!r}")

    data = np.load(STORE / "vectors.npz", allow_pickle=True)
    ids = np.asarray(data["molecule_id"]).astype(str)
    if set(ids.tolist()) != {row["molecule_id"] for row in parsed}:
        raise SystemExit("ABORT: vectors.npz ids differ from molecules.tsv - rebuild the store instead")

    n_receptors = len(np.asarray(data["receptor_ids"]))
    mol_axis, rec_axis = [], []
    for key in data.keys():
        arr = data[key]
        if arr.shape and arr.shape[0] == len(ids):
            mol_axis.append(key)
        elif arr.shape and arr.shape[0] == n_receptors:
            rec_axis.append(key)
        else:
            raise SystemExit(f"ABORT: {key} shape {arr.shape} matches neither axis")
    print(f"  molecule axis : {mol_axis}")
    print(f"  receptor axis : {rec_axis}")

    mask = np.isin(ids, list(expected.keys()))
    if int(mask.sum()) != len(expected):
        raise SystemExit(f"ABORT: {int(mask.sum())} survivors vs {len(expected)} csv identities")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if not drop and not rename:
        print("\nstore already reconciles")
        return 0

    backup = STORE / f"_backup_{time.strftime('%Y%m%dT%H%M%S')}"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STORE / "molecules.tsv", backup / "molecules.tsv")
    shutil.copy2(STORE / "vectors.npz", backup / "vectors.npz")

    rename_map = {parsed[i]["molecule_id"]: new for i, _, new in rename}
    out_rows = [header]
    for i in keep:
        fields = rows[i].split("\t")
        mid = parsed[i]["molecule_id"]
        if mid in rename_map:
            fields[1] = rename_map[mid]
        out_rows.append("\t".join(fields))
    (STORE / "molecules.tsv").write_text("\n".join(out_rows) + "\n", encoding="utf-8")

    np.savez(
        STORE / "vectors.npz",
        **{k: (data[k][mask] if k in mol_axis else data[k]) for k in data.keys()},
    )

    provenance = json.loads((STORE / "provenance.json").read_text(encoding="utf-8"))
    provenance["n_molecules"] = int(mask.sum())
    provenance["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    (STORE / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    vec = np.load(STORE / "vectors.npz", allow_pickle=True)
    after_ids = np.asarray(vec["molecule_id"]).astype(str)
    after_rows = [ln for ln in (STORE / "molecules.tsv").read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    print(f"\ndropped {len(drop)}, renamed {len(rename)}")
    print(f"  molecules.tsv : {len(after_rows)}")
    print(f"  vectors.npz   : {len(after_ids)} ids, logits {vec['logits'].shape}")
    print(f"  backup        : {backup}")
    if len(after_ids) != len(expected) or len(after_rows) != len(expected):
        raise SystemExit("ABORT-post: did not reconcile")
    if set(after_ids.tolist()) != set(expected.keys()):
        raise SystemExit("ABORT-post: ids disagree with the csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
