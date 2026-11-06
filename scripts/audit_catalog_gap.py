#!/usr/bin/env python3
"""
What do we actually know about a supplier catalogue that is not in the library?

Joins a Fraterworks gap list (molor/data/fraterworks_missing.json) against every
data source the library already owns, and reports per material which columns
would be inherited, which would be imputed, and which would be empty.

    python scripts/audit_catalog_gap.py
    python scripts/audit_catalog_gap.py --gap ../valkon.us/molor/data/fraterworks_missing.json

Sources, and what each one can contribute:

  data/inventory/inventory.csv          boiling_point_c, logp, volatility_class,
                                        family, ifra_cat4/cat9, top_pct,
                                        water_threshold_ppb, stability
  library/input/expanded_materials.csv  name + smiles (already a library molecule)
  data/ifra/materials.csv               ifra_cat4_pct + ifra_status
                                        (the resolved verdicts, not a seed)
  tools/longevity.csv     (perfumer)    substantivity_hours
  perfumer/docs/usable-materials-cas.csv  the CAS the library already knows

The inventory has no CAS column - it keys on name and InChIKey - so the join
from a supplier CAS goes through the perfumer's name/CAS list, then by name.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PERFUMER = ROOT.parent / "perfumer"

DEFAULT_GAP = ROOT.parent / "valkon.us" / "molor" / "data" / "fraterworks_missing.json"
INVENTORY = ROOT / "data" / "inventory" / "inventory.csv"
LIBRARY = ROOT / "library" / "input" / "expanded_materials.csv"
IFRA = ROOT / "data" / "ifra" / "materials.csv"
LONGEVITY = PERFUMER / "tools" / "longevity.csv"
USABLE = PERFUMER / "docs" / "usable-materials-cas.csv"

CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
NOISE_RE = re.compile(r"[\u00ae\u2122\u00a9\u2120]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")

BLANK = {"", "-", "\u2014", "—", "n/a", "na", "none"}


def norm(value: str) -> str:
    text = NOISE_RE.sub(" ", (value or "").lower()).replace("&", " and ")
    return " ".join(NON_ALNUM.sub(" ", text).split())


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with io.open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def has_value(row: dict, field: str) -> bool:
    return (row.get(field) or "").strip().lower() not in BLANK


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", default=str(DEFAULT_GAP))
    parser.add_argument("--out", default=str(HERE.parent / "artifacts" / "catalog_gap_audit.csv"))
    args = parser.parse_args()

    gap_doc = json.loads(Path(args.gap).read_text(encoding="utf-8"))
    gap = gap_doc.get("missing") or []

    inventory = read_csv(INVENTORY)
    library = read_csv(LIBRARY)
    # The IFRA column used to come from the perfumer's hand-kept seed. That
    # seed is gone (2026-11): the figures are resolved here in the library, and
    # reading the old path silently reported 0 rows -- a coverage column that
    # reads "0 / 384" looks like a fact about the catalogue rather than a
    # broken join, which is exactly the failure this script exists to avoid.
    ifra = read_csv(IFRA)
    if not ifra:
        print(f"WARNING: {IFRA} missing or empty; the ifra_limit_pct column "
              f"is meaningless. Run scripts/import_ifra.py and "
              f"scripts/resolve_ifra.py.")
    longevity = read_csv(LONGEVITY)
    usable = read_csv(USABLE)

    inv_by_name = {norm(r.get("name", "")): r for r in inventory}
    inv_by_key = {(r.get("inchikey") or "").strip(): r for r in inventory if r.get("inchikey")}
    lib_by_name = {norm(r.get("name", "")): r for r in library}
    cas_to_name = {(r.get("cas") or "").strip(): (r.get("name") or "") for r in usable}
    ifra_by_name = {norm(r.get("name", "")): r for r in ifra}
    ifra_by_cas = {(r.get("cas") or "").strip(): r for r in ifra if r.get("cas")}
    long_by_name = {norm(r.get("name", "")): r for r in longevity}

    rows = []
    for material in gap:
        title = material.get("title") or ""
        cas_list = material.get("cas") or []
        cas = cas_list[0] if cas_list else ""
        key = norm(title)

        # A supplier name may reach the inventory through the CAS list the
        # perfumer already holds, so try the resolved name too.
        usable_name = cas_to_name.get(cas, "")
        inv = inv_by_name.get(key) or (usable_name and inv_by_name.get(norm(usable_name))) or None
        if inv is None and cas:
            inv = next((r for r in inventory if (r.get("smiles") or "") and cas and False), None)

        ifra_row = ifra_by_cas.get(cas) or ifra_by_name.get(key)
        long_row = long_by_name.get(key)

        tags = material.get("tags") or []
        supplier_regs = sorted(
            t for t in tags if any(w in t.lower() for w in ("ifra", "regs", "spec", "hs_code", "sds", "documents"))
        )

        rows.append(
            {
                "name": title,
                "cas": cas,
                "product_type": material.get("product_type"),
                "in_inventory": "yes" if inv else "no",
                "in_library": "yes" if key in lib_by_name else "no",
                "boiling_point_c": (inv.get("boiling_point_c") if inv and has_value(inv, "boiling_point_c") else ""),
                "logp": (inv.get("logp") if inv and has_value(inv, "logp") else ""),
                "volatility_class": (inv.get("volatility_class") if inv and has_value(inv, "volatility_class") else ""),
                "family": (inv.get("family") if inv and has_value(inv, "family") else ""),
                "ifra_cat4": (inv.get("ifra_cat4") if inv and has_value(inv, "ifra_cat4") else ""),
                "ifra_limit_pct": ((ifra_row or {}).get("cat4_pct") or ""),
                "ifra_status": ((ifra_row or {}).get("status") or ""),
                "top_pct": (inv.get("top_pct") if inv and has_value(inv, "top_pct") else ""),
                "threshold_ppb": (
                    inv.get("water_threshold_ppb") if inv and has_value(inv, "water_threshold_ppb") else ""
                ),
                "longevity_h": ((long_row or {}).get("substantivity_hours") or ""),
                "supplier_regs_tags": ";".join(supplier_regs),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    have = lambda field: sum(1 for r in rows if str(r[field]).strip() not in BLANK)  # noqa: E731
    print(f"gap materials                    : {n}")
    print(f"inventory rows available         : {len(inventory)}")
    print(f"library molecules available      : {len(library)}")
    print(f"data/ifra/materials.csv rows     : {len(ifra)}")
    print(f"longevity.csv rows               : {len(longevity)}")
    print()
    print(f"  in inventory (inherits columns) : {have('in_inventory') if False else sum(1 for r in rows if r['in_inventory'] == 'yes')}")
    print(f"  already in library              : {sum(1 for r in rows if r['in_library'] == 'yes')}")
    print(f"  has CAS                         : {sum(1 for r in rows if r['cas'])}")
    print()
    print("column coverage across the gap:")
    for field in (
        "boiling_point_c",
        "logp",
        "volatility_class",
        "family",
        "ifra_cat4",
        "ifra_limit_pct",
        "top_pct",
        "threshold_ppb",
        "longevity_h",
    ):
        count = have(field)
        print(f"  {field:<20} {count:4d} / {n}  ({100 * count / n:5.1f}%)")

    already = [r["name"] for r in rows if r["in_library"] == "yes"]
    if already:
        print(f"\nalready in the library - these are differ false negatives, fix with an ALIAS not a row:")
        for name in already:
            print(f"  {name}")

    print("\nsupplier regs tags (what Fraterworks says it has):")
    for value, count in Counter(r["supplier_regs_tags"] for r in rows).most_common(8):
        print(f"  {count:4d}  {value or '(none)'}")

    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
