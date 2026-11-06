#!/usr/bin/env python3
"""
Append artifacts/proposed_materials.csv rows to library/input/expanded_materials.csv.

    .venv/Scripts/python.exe scripts/apply_proposed_materials.py --dry-run
    .venv/Scripts/python.exe scripts/apply_proposed_materials.py

APPEND-ONLY, deliberately. An earlier version of this script rewrote the whole
file through csv.DictWriter, and because expanded_materials.csv contains at
least one row with more fields than the header, DictWriter raised *after* the
file had already been truncated to 309 of 428 lines. Nothing about the append
needs a rewrite, so this version never touches existing bytes.

Gates before writing: every proposed SMILES parses in RDKit, no name collides
with an existing row, and the header is exactly name,smiles,source. Idempotent:
proposals already present by normalised name are skipped.
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

NOISE_RE = re.compile(r"[\u00ae\u2122\u00a9\u2120]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(value: str) -> str:
    text = NOISE_RE.sub(" ", (value or "").lower()).replace("&", " and ")
    return " ".join(NON_ALNUM.sub(" ", text).split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    text = LIBRARY.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    if header != ["name", "smiles", "source"]:
        raise SystemExit(f"unexpected header: {header}")

    rag = [(index + 2, row) for index, row in enumerate(body) if len(row) != 3]
    if rag:
        print(f"note: {len(rag)} pre-existing row(s) with != 3 fields (left untouched):")
        for line_no, row in rag[:5]:
            print(f"  line {line_no}: {len(row)} fields - {row[0][:50]!r}")

    taken = {norm(row[0]) for row in body if row}

    with io.open(PROPOSED, newline="", encoding="utf-8-sig") as handle:
        proposed = list(csv.DictReader(handle))

    added, skipped, bad = [], [], []
    for row in proposed:
        name = (row.get("name") or "").strip()
        smiles = (row.get("smiles") or "").strip()
        if not name or not smiles:
            bad.append((name, "empty"))
            continue
        if norm(name) in taken:
            skipped.append(name)
            continue
        if Chem.MolFromSmiles(smiles) is None:
            bad.append((name, "smiles"))
            continue
        taken.add(norm(name))
        added.append({"name": name, "smiles": smiles, "source": row["source"]})

    print(f"existing rows : {len(body)}")
    print(f"proposed      : {len(proposed)}")
    print(f"  to append   : {len(added)}")
    print(f"  skipped     : {len(skipped)} (already present)")
    print(f"  rejected    : {len(bad)}")
    for name, why in bad[:10]:
        print(f"    {name:<40} {why}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if bad:
        raise SystemExit("refusing to write: rejected rows present")
    if not added:
        print("\nnothing to append")
        return 0

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["name", "smiles", "source"], lineterminator="\n")
    writer.writerows(added)
    payload = buffer.getvalue()

    if not text.endswith("\n"):
        payload = "\n" + payload

    with LIBRARY.open("a", encoding="utf-8", newline="") as handle:
        handle.write(payload)

    after = len(list(csv.reader(io.StringIO(LIBRARY.read_text(encoding="utf-8-sig"))))) - 1
    print(f"\nappended {len(added)} rows")
    print(f"library now : {after} rows (expected {len(body) + len(added)})")
    if after != len(body) + len(added):
        raise SystemExit("row count does not reconcile - check the file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
