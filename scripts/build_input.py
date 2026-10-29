"""Merge the raw material list with corrections and derive a chemical class.

Writes library/input/final_materials.csv, the curated input for
``python -m orvector library add``. Keeps an audit trail: every row records
whether its structure came from the original list or a correction.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "library" / "input"

SMARTS = {
    "nitrile": "[NX1]#[CX2]",
    "thiol": "[SX2H]",
    "sulfide": "[#6][SX2][#6]",
    "imine": "[CX3]=[NX2]",
    "amide": "[NX3][CX3](=O)",
    "lactone": "[C;R](=O)[O;R]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "phenol": "[OX2H]c",
    "alcohol": "[OX2H][CX4]",
    "acetal": "[CX4]([OX2][#6])[OX2][#6]",
    "ether": "[#6][OX2][#6]",
    "aromatic": "c",
    "nitrogen": "[#7]",
    "sulfur": "[#16]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}

#: Priority order for assigning a single class label.
PRIORITY = [
    "sulfur_note",
    "nitrile",
    "schiff_base",
    "amide",
    "macrocyclic_lactone",
    "lactone",
    "heterocycle",
    "phenolic",
    "aromatic_ester",
    "aromatic_aldehyde",
    "aromatic_ketone",
    "aromatic_alcohol",
    "aromatic_other",
    "terpenoid_ketone",
    "aliphatic_ester",
    "aliphatic_aldehyde",
    "aliphatic_ketone",
    "aliphatic_alcohol",
    "acetal",
    "hydrocarbon",
    "other",
]


def largest_ring(mol) -> int:
    ri = mol.GetRingInfo()
    return max((len(r) for r in ri.AtomRings()), default=0)


def has(mol, key) -> bool:
    p = PATS[key]
    return p is not None and mol.HasSubstructMatch(p)


def classify(mol) -> str:
    arom = has(mol, "aromatic")
    ring = largest_ring(mol)
    heteroaromatic = any(
        a.GetIsAromatic() and a.GetSymbol() != "C" for a in mol.GetAtoms()
    )

    if has(mol, "thiol") or has(mol, "sulfide") or has(mol, "sulfur"):
        return "sulfur_note"
    if has(mol, "nitrile"):
        return "nitrile"
    if has(mol, "imine"):
        return "schiff_base"
    if has(mol, "amide"):
        return "amide"
    if has(mol, "lactone"):
        return "macrocyclic_lactone" if ring >= 10 else "lactone"
    if heteroaromatic or (ring >= 5 and has(mol, "ether") and not arom and not has(mol, "ester")):
        return "heterocycle"
    if has(mol, "phenol"):
        return "phenolic"
    if arom:
        if has(mol, "ester"):
            return "aromatic_ester"
        if has(mol, "aldehyde"):
            return "aromatic_aldehyde"
        if has(mol, "ketone"):
            return "aromatic_ketone"
        if has(mol, "alcohol"):
            return "aromatic_alcohol"
        return "aromatic_other"
    if has(mol, "ketone") and ring and ring <= 7:
        return "terpenoid_ketone"
    if has(mol, "ester"):
        return "aliphatic_ester"
    if has(mol, "aldehyde"):
        return "aliphatic_aldehyde"
    if has(mol, "ketone"):
        return "aliphatic_ketone"
    if has(mol, "alcohol"):
        return "aliphatic_alcohol"
    if has(mol, "acetal"):
        return "acetal"
    if not any(a.GetSymbol() not in ("C", "H") for a in mol.GetAtoms()):
        return "hydrocarbon"
    return "other"


def load(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["material"].strip()] = row["smiles"].strip()
    return out


def main() -> int:
    original = load(INPUT / "my_materials.tsv")
    corrections = load(INPUT / "corrections.tsv")

    #: Structures that are genuinely identical to another entry. Keeping both
    #: would plot two labels on one point and imply false precision.
    DROP = {
        "Auralva": "structure identical to Aurantiol; Auralva's own structure is unconfirmed",
    }

    rows = []
    skipped_mixture = []
    for name, smi in original.items():
        if smi.lower() == "mixture":
            skipped_mixture.append(name)
            continue
        if name in DROP:
            continue
        source = "original"
        if name in corrections:
            smi = corrections[name]
            source = "corrected"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  SKIP unparseable: {name}", file=sys.stderr)
            continue
        canonical = Chem.MolToSmiles(mol)
        rows.append(
            {
                "name": name,
                "smiles": canonical,
                "class": classify(mol),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "source": source,
                "input_smiles": smi,
            }
        )

    rows.sort(key=lambda r: r["name"])
    out = INPUT / "final_materials.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["name", "smiles", "class", "formula", "heavy_atoms",
                        "source", "input_smiles"],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)

    from collections import Counter

    print(f"wrote {out}  ({len(rows)} materials)")
    print(f"  corrected structures: {sum(1 for r in rows if r['source'] == 'corrected')}")
    print(f"  mixtures excluded   : {len(skipped_mixture)}")
    print(f"  dropped duplicates  : {list(DROP)}")
    print("\nclass distribution:")
    for cls, n in Counter(r["class"] for r in rows).most_common():
        print(f"  {cls:22s} {n:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
