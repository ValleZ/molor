"""Normalise the perfumer's own inventory sheet into the project.

Exported from the perfumery-inventory spreadsheet. It carries several things
this project had been missing or guessing at:

*top %* -- the maximum level the perfumer actually uses a material at. Against
the 11 materials that also have a measured air-phase threshold, this correlates
at rho 0.756, better than anything modelled here. It is not pure potency, since
it also reflects IFRA limits, cost, solubility and how much of the character is
wanted, but as a practical dosing guide it is the best signal available.

*family* -- an odour/chemical family per material, which restores an evaluation
axis that had gone dead: expanded_materials.csv has no class column, so
nearest-neighbour class coherence was reading 1.000 for every model and for
chance alike.

*water thresholds in ppb* -- assembled from many sources, and the perfumer
flagged it as rough. Testing bears that out: against Abraham's measured
air-phase values it reaches only rho 0.451 over 11 materials, and it agrees with
the sheet's own top % at just rho 0.386. Part of that is legitimate -- a water
threshold is a different quantity from an air threshold, since the molecule must
also partition out of solution -- but it is kept as reference rather than as
ground truth.

*type, Log drop per 0.3%, stability, IFRA limits, logP, boiling point* --
volatility class, tenacity (19 of 165 measured rather than estimated),
regulatory ceilings and physical properties.

Names are cleaned with the same rules as the library so the two join, and
mixtures and naturals are kept but marked, since they have no single structure.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clean_library import clean_name  # noqa: E402

SRC = ROOT / "data" / "inventory" / "perfumery_inventory.csv"
OUT = ROOT / "data" / "inventory" / "inventory.csv"

#: Sheet column -> output column.
COLS = {
    "type": "volatility_class",
    "family": "family",
    "top %": "top_pct",
    "IFRA Cat 4": "ifra_cat4",
    "IFRA Cat 9": "ifra_cat9",
    "stability": "stability",
    "Log drop per 0.3%": "log_drop",
    "log drop measured or estimated": "log_drop_source",
    "water thresholds in ppb": "water_threshold_ppb",
    "logP": "logp",
    "Boiling point": "boiling_point_c",
}


def num(x: str | None) -> float | None:
    if not isinstance(x, str):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", x.replace(",", ""))
    return float(m.group(0)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    rows = list(csv.DictReader(io.open(args.src, encoding="utf-8-sig")))
    print(f"[inv] {len(rows)} rows from the sheet")

    out, n_mix, n_bad = [], 0, 0
    for r in rows:
        name = clean_name(r["material"])
        smi = (r.get("smiles") or "").strip()
        key, is_single = "", False
        if smi and smi.lower() != "mixture":
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                key = Chem.MolToInchiKey(mol)
                smi = Chem.MolToSmiles(mol)
                is_single = True
            else:
                n_bad += 1
        else:
            n_mix += 1
        rec = {"name": name, "inchikey": key, "smiles": smi if is_single else "",
               "single_molecule": int(is_single)}
        for src_col, dst in COLS.items():
            v = r.get(src_col, "")
            if dst in ("top_pct", "log_drop", "water_threshold_ppb", "logp",
                       "boiling_point_c"):
                n = num(v)
                rec[dst] = "" if n is None else n
            else:
                rec[dst] = (v or "").strip()
        out.append(rec)

    print(f"[inv] {sum(r['single_molecule'] for r in out)} single molecules, "
          f"{n_mix} mixtures or naturals, {n_bad} unparseable SMILES")

    # Join rate against the library, which is what makes the sheet useful.
    lib_names = {r["name"] for r in csv.DictReader(
        io.open(ROOT / "library/input/expanded_materials.csv",
                encoding="utf-8-sig"))}
    joined = [r for r in out if r["name"] in lib_names]
    print(f"[inv] {len(joined)} of {len(out)} join the 301-material library by name")
    missing = sorted(r["name"] for r in out
                     if r["name"] not in lib_names and r["single_molecule"])
    if missing:
        print(f"[inv] single molecules in the sheet but not the library "
              f"({len(missing)}): {missing[:8]}")

    fields = (["name", "inchikey", "smiles", "single_molecule"]
              + list(COLS.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"[inv] wrote {args.out}")

    fams = {}
    for r in out:
        if r["family"]:
            fams[r["family"]] = fams.get(r["family"], 0) + 1
    meta = {
        "source": "perfumery-inventory spreadsheet, exported by the perfumer",
        "rows": len(out),
        "single_molecules": int(sum(r["single_molecule"] for r in out)),
        "join_library": len(joined),
        "top_pct_filled": int(sum(1 for r in out if r["top_pct"] != "")),
        "family_filled": int(sum(1 for r in out if r["family"])),
        "families": len(fams),
        "log_drop_measured": int(sum(1 for r in out
                                     if r["log_drop_source"] == "m")),
        "caveats": {
            "top_pct": "perfumer's own maximum use level; rho 0.756 against "
                       "measured air thresholds over 11 materials, but also "
                       "reflects IFRA limits, cost and desired character",
            "water_threshold_ppb": "assembled from mixed sources and flagged "
                                   "rough by the perfumer; rho 0.451 against "
                                   "Abraham air thresholds (n=11) and 0.386 "
                                   "against the sheet's own top %; water and "
                                   "air thresholds are different quantities",
        },
    }
    (Path(args.out).parent / "manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[inv] {meta['family_filled']} family labels across "
          f"{meta['families']} families; {meta['log_drop_measured']} measured "
          f"log-drop values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
