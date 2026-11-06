#!/usr/bin/env python3
"""
Catch molecules added in the catalogue batch that duplicate a row already there,
comparing on the InChIKey *skeleton* rather than the full key.

Why this exists: the fetch gate in fetch_catalog_structures.py compares full
InChIKeys. That misses the case it most needs to catch - the library stores many
constituents as 2D SMILES with unspecified stereochemistry, so their key ends
-UHFFFAOYSA-N, while PubChem returns the stereospecific structure and the same
molecule ends -CUZKYEQNSA-N. Patchoulol arrived twice this way: once as
"Patchoulol" (GC constituent reference) and once as "Clearwood (R) PRISMA"
(Fraterworks sells both under CAS 5986-55-0).

The percept model reads ECFP4 radius 2 and cannot see stereochemistry at all, so
two rows that differ only in stereo get bit-identical vectors: they are the same
material twice, and they inflate every count downstream.

    .venv/Scripts/python.exe scripts/audit_new_duplicates.py
    .venv/Scripts/python.exe scripts/audit_new_duplicates.py --fix
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIBRARY = ROOT / "library" / "input" / "expanded_materials.csv"
PROPOSED = ROOT / "artifacts" / "proposed_materials.csv"
ALIASES = ROOT / "artifacts" / "proposed_aliases.csv"

NOISE_RE = re.compile(r"[\u00ae\u2122\u00a9\u2120]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(value: str) -> str:
    text = NOISE_RE.sub(" ", (value or "").lower()).replace("&", " and ")
    return " ".join(NON_ALNUM.sub(" ", text).split())


def skeleton(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="remove the duplicate appended rows and record aliases")
    args = parser.parse_args()

    with io.open(PROPOSED, newline="", encoding="utf-8-sig") as handle:
        appended_names = {norm(r["name"]): r["name"] for r in csv.DictReader(handle)}

    text = LIBRARY.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))[1:]
    rows = [r for r in rows if r]

    original: dict[str, str] = {}
    batch: dict[str, list[str]] = {}
    for row in rows:
        name, smiles = row[0], (row[1] if len(row) > 1 else "")
        key = skeleton(smiles)
        if not key:
            continue
        if norm(name) in appended_names:
            batch.setdefault(key, []).append(name)
        else:
            original.setdefault(key, name)

    collisions = {key: names for key, names in batch.items() if key in original}
    internal = {key: names for key, names in batch.items() if key not in original and len(names) > 1}

    print(f"appended rows            : {len(appended_names)}")
    print(f"same skeleton as an existing row : {len(collisions)}")
    for key, names in collisions.items():
        print(f"  {key}  {original[key]!r}  <-  {names}")
    print(f"duplicated inside the batch      : {len(internal)}")
    for key, names in internal.items():
        print(f"  {key}  {names}")

    if not args.fix:
        print("\n--fix not passed: nothing changed")
        return 0

    # Only cross-collisions are removed: those rows restate a material the
    # library already had. Duplicates *within* the batch are reported and left
    # alone - Rose Oxide HC (CAS 16409-43-1) and Rose Oxide Laevo (3033-23-6) are
    # distinct commercial grades, and export_simplify_data already flags the
    # bit-identical percept vectors that result. Deleting one would be a product
    # decision, not a data-repair one.
    doomed = {name for names in collisions.values() for name in names}

    alias_rows = []
    for key, names in collisions.items():
        for name in names:
            alias_rows.append({"supplier_name": name, "library_name": original[key], "why": f"skeleton {key}"})
    if internal:
        print(f"\nkeeping {len(internal)} stereo/grade pair(s) inside the batch - reported, not removed")

    doomed_norm = {norm(d) for d in doomed}
    lines = text.splitlines(keepends=True)
    kept, removed = [], 0
    for line in lines:
        if not line.strip():
            kept.append(line)
            continue
        parsed = next(csv.reader([line.rstrip("\r\n")]))
        if parsed and norm(parsed[0]) in doomed_norm:
            removed += 1
            continue
        kept.append(line)

    LIBRARY.write_text("".join(kept), encoding="utf-8")

    with ALIASES.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["supplier_name", "library_name", "why"])
        writer.writerows(alias_rows)

    after = len([r for r in csv.reader(io.StringIO(LIBRARY.read_text(encoding="utf-8-sig"))) if r]) - 1
    print(f"\nremoved {removed} duplicate row(s); library now {after} rows")
    print(f"recorded {len(alias_rows)} alias(es) in {ALIASES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
