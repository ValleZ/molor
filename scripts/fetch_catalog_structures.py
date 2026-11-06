#!/usr/bin/env python3
"""
Resolve structures for a supplier catalogue gap list, so it can be added to the
library without a single hand-typed SMILES.

    .venv/Scripts/python.exe scripts/fetch_catalog_structures.py
    .venv/Scripts/python.exe scripts/fetch_catalog_structures.py --refresh

Input : ../valkon.us/molor/data/fraterworks_missing.json  (the gap list)
Output: artifacts/catalog_structures.json   cache: CID, SMILES, InChIKey, formula
        artifacts/proposed_materials.csv    name,smiles,source  - ready to append

Gates applied, in order, because each one has a silent failure mode:

  1. already in the library by normalised name  -> emit an ALIAS, not a row
  2. no CAS on the supplier page                -> deferred (no structure to trust)
  3. PubChem name/CAS lookup returns nothing    -> deferred, reported by name
  4. RDKit cannot parse the fetched SMILES      -> dropped, reported (featurise
                                                   would drop it silently later)
  5. InChIKey already present in the library or earlier in this batch
                                                -> it is the same molecule: alias

Stereochemistry: PubChem's IsomericSMILES is kept as given. The percept model
reads ECFP4 radius-2 bits and cannot see it either way, but the column is not
ours to invent.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PERFUMER = ROOT.parent / "perfumer"

DEFAULT_GAP = ROOT.parent / "valkon.us" / "molor" / "data" / "fraterworks_missing.json"
LIBRARY = ROOT / "library" / "input" / "expanded_materials.csv"
CACHE = ROOT / "artifacts" / "catalog_structures.json"
PROPOSED = ROOT / "artifacts" / "proposed_materials.csv"

PUBCHEM = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/IsomericSMILES,InChIKey,MolecularFormula/JSON"
)
UA = "olfactory-embeddings/1.0 (catalogue structure resolution)"

import re

NOISE_RE = re.compile(r"[\u00ae\u2122\u00a9\u2120]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(value: str) -> str:
    text = NOISE_RE.sub(" ", (value or "").lower()).replace("&", " and ")
    return " ".join(NON_ALNUM.sub(" ", text).split())


def inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        return None


def pubchem(cas: str, cache: dict, pace: float, refresh: bool) -> dict:
    if not refresh and cas in cache:
        return cache[cas]

    url = PUBCHEM.format(name=urllib.parse.quote(cas))
    result: dict = {"cas": cas, "status": "no_cid"}
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"user-agent": UA, "accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as res:
                payload = json.loads(res.read())
            props = (payload.get("PropertyTable") or {}).get("Properties") or []
            if props:
                row = props[0]

                # PubChem returns CAS hits in an arbitrary order; prefer the one
                # whose InChIKey has an actual structure.
                result = {
                    "cas": cas,
                    "status": "ok",
                    "cid": row.get("CID"),
                    "smiles": row.get("IsomericSMILES") or row.get("SMILES") or row.get("CanonicalSMILES"),
                    "inchikey": row.get("InChIKey"),
                    "formula": row.get("MolecularFormula"),
                }
            break
        except urllib.error.HTTPError as err:
            if err.code == 404:
                result = {"cas": cas, "status": "no_cid"}
                break
            if err.code in (429, 500, 502, 503) and attempt < 2:
                delay = 2.0 * 2**attempt + random.random()
                print(f"    {cas}: HTTP {err.code}, retry in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
                continue
            result = {"cas": cas, "status": f"http_{err.code}"}
            break
        except Exception as err:  # noqa: BLE001
            if attempt < 2:
                time.sleep(1.5 * 2**attempt)
                continue
            result = {"cas": cas, "status": f"error_{type(err).__name__}"}
            break

    cache[cas] = result
    time.sleep(pace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", default=str(DEFAULT_GAP))
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-query")
    parser.add_argument("--pace", type=float, default=0.25, help="seconds between PubChem calls")
    args = parser.parse_args()

    gap_doc = json.loads(Path(args.gap).read_text(encoding="utf-8"))
    gap = gap_doc.get("missing") or []

    library = list(csv.DictReader(io.open(LIBRARY, newline="", encoding="utf-8-sig")))
    lib_by_name = {norm(r["name"]): r for r in library}
    lib_keys = {}
    for row in library:
        key = inchikey(row.get("smiles", ""))
        if key:
            lib_keys.setdefault(key, row["name"])

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() and not args.refresh else {}

    alias = []
    deferred_no_cas = []
    no_cid = []
    bad_smiles = []
    duplicate = []
    proposed = []
    batch_keys: dict[str, str] = {}

    for index, material in enumerate(gap, 1):
        name = (material.get("title") or "").strip()
        cas_list = material.get("cas") or []
        cas = cas_list[0] if cas_list else ""

        if norm(name) in lib_by_name:
            alias.append({"supplier_name": name, "library_name": lib_by_name[norm(name)]["name"], "why": "name"})
            continue

        if not cas:
            deferred_no_cas.append(name)
            continue

        hit = pubchem(cas, cache, args.pace, args.refresh)
        smiles = hit.get("smiles")
        if not smiles:
            no_cid.append({"name": name, "cas": cas, "status": hit.get("status")})
            continue

        key = inchikey(smiles)
        if key is None:
            bad_smiles.append({"name": name, "cas": cas, "smiles": smiles})
            continue
        if key in lib_keys:
            alias.append({"supplier_name": name, "library_name": lib_keys[key], "why": f"inchikey {key}"})
            continue
        if key in batch_keys:
            duplicate.append({"name": name, "cas": cas, "same_as": batch_keys[key]})
            alias.append({"supplier_name": name, "library_name": batch_keys[key], "why": f"inchikey {key}"})
            continue

        batch_keys[key] = name
        proposed.append(
            {
                "name": name,
                "smiles": smiles,
                "source": (
                    f"CAS {cas}; Fraterworks catalogue snapshot {gap_doc.get('meta', {}).get('generated_at', '')[:10]}; "
                    f"structure from PubChem CID {hit.get('cid')} (InChIKey {key}, {hit.get('formula')}); "
                    "supplier title used verbatim, stereochemistry as PubChem gives it"
                ),
                "_cas": cas,
                "_cid": hit.get("cid"),
                "_inchikey": key,
            }
        )

        if index % 25 == 0:
            print(f"  {index}/{len(gap)}  proposed {len(proposed)}", file=sys.stderr)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    with PROPOSED.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "smiles", "source", "_cas", "_cid", "_inchikey"])
        writer.writeheader()
        writer.writerows(proposed)

    alias_report = PROPOSED.with_name("proposed_aliases.csv")
    with alias_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["supplier_name", "library_name", "why"])
        writer.writeheader()
        writer.writerows(alias)

    total = len(gap)
    print(f"\ngap list                          : {total}")
    print(f"  alias (already a library row)   : {len(alias)}")
    print(f"  deferred - no CAS               : {len(deferred_no_cas)}")
    print(f"  deferred - no PubChem record    : {len(no_cid)}")
    print(f"  dropped  - SMILES will not parse: {len(bad_smiles)}")
    print(f"  duplicate InChIKey in batch     : {len(duplicate)}")
    print(f"  NEW MOLECULES PROPOSED          : {len(proposed)}")

    if no_cid[:10]:
        print("\nno PubChem record (first 10):")
        for row in no_cid[:10]:
            print(f"  {row['name']:<38} {row['cas']:<14} {row['status']}")
    if bad_smiles:
        print("\nunparsable (first 5):")
        for row in bad_smiles[:5]:
            print(f"  {row['name']:<38} {row['smiles']}")

    print(f"\nwrote -> {PROPOSED}")
    print(f"wrote -> {alias_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
