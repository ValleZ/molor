"""Normalise material names and SMILES, and merge duplicate compounds.

Three problems, all of which reach the model or the viewer:

1. *Dilution in the name.* Many materials are stocked as solutions -- "Oxane,
   50% TEC", "Furfural 5% DPG" -- but the vector is for the molecule, not the
   bottle, so the strength and solvent do not belong in the identity. Note the
   traps: "Benzyl Alcohol" and "Matsutake Alcohol" are compound names rather
   than dilutions, and DEP, IPM and TEC appear both as solvent abbreviations
   and as materials in their own right.

2. *Duplicate compounds.* "Nootkatone" and "Nootkatone Crystals" are one
   compound entered twice, differing only in whether the SMILES specifies
   stereochemistry. They occupy two points on a map that claims to show
   molecules.

3. *Bracket atoms changing the featurisation.* Chirality must be written in
   brackets, as in [C@H], and RDKit then records that atom's hydrogens as
   explicit, which zeroes its implicit valence. dgllife's
   CanonicalAtomFeaturizer feeds implicit valence to the model, so the same
   molecule featurises differently depending only on how its SMILES was typed.
   Measured on this library: 14 of the 15 bracket-atom materials.

For (3) the fix normalises hydrogen bookkeeping while *keeping* the chiral tags,
so the stored SMILES stays the full description of the molecule and a future
stereo-aware model loses nothing. Both current models ignore stereochemistry
regardless -- verified, not assumed -- so this changes no chemistry, only the
notation artefact.

Run with --apply to write; the default is a dry run that reports every change.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "library/input/expanded_materials.csv"

#: A trailing dilution: optional comma, a percentage, and an optional solvent.
#: Anchored to the end so an interior percentage cannot be eaten by accident.
DILUTION = re.compile(
    r"[,\s]+\d+(?:\.\d+)?\s*%\s*(?:in\s+)?[A-Za-z][A-Za-z ]*$|"
    r"[,\s]+\d+(?:\.\d+)?\s*%\s*$",
    re.I)

#: Grade and form qualifiers that are not part of the compound's identity.
#: "Crystals" is only stripped for these reviewed names: elsewhere it can be
#: part of a trade name (Oranger Crystals), where removing it invents a
#: material that no supplier lists.
FORM_SUFFIX = re.compile(r",?\s*(crystals|cryst\.?)\s*$", re.I)
FORM_OK = {
    "Benzyl Isoeugenol, crystals", "Camphor Crystals", "Cedrol Crystals",
    "Cedryl Acetate, Crystals", "Coumarin, crystals", "Nootkatone Crystals",
    "Thymol Crystals",
}
#: Trailing supplier abbreviation or grade, e.g. "Diethyl Phthalate (DEP)".
PAREN_ABBR = re.compile(r"\s*\((?:DEP|IPM|TEC|DPG|PG|PEA)\)\s*$", re.I)
GRADE = re.compile(r",\s*Natural\s*$", re.I)


def clean_name(name: str) -> str:
    out = name.strip()
    out = PAREN_ABBR.sub("", out)
    out = GRADE.sub("", out)
    if name in FORM_OK:
        out = FORM_SUFFIX.sub("", out)
    prev = None
    while prev != out:            # a name may carry both a grade and a strength
        prev = out
        out = DILUTION.sub("", out).strip()
    return out.strip(" ,")


def normalise_mol(smi: str):
    """Make hydrogen bookkeeping notation-independent, keeping stereochemistry.

    Returns (mol, inchikey) or (None, None) if unparseable. The returned mol is
    what should be featurised; the canonical SMILES of it is unchanged from the
    input, because RDKit writes a chiral atom in brackets either way. The
    difference is internal and shows up only in the atom features, which is
    exactly where the artefact was.
    """
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    before = Chem.MolToInchiKey(m)
    for a in m.GetAtoms():
        if a.GetNumExplicitHs():
            # Hand the hydrogens back to the implicit count; the chiral tag is
            # stored on the atom and survives.
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(m)
    except Exception:             # noqa: BLE001 - keep the original on failure
        return Chem.MolFromSmiles(smi), before
    if Chem.MolToInchiKey(m) != before:
        # Never silently alter the compound; keep the original instead.
        return Chem.MolFromSmiles(smi), before
    return m, before


def stereo_specified(smi: str) -> bool:
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return any(str(e.specified) == "Specified" for e in Chem.FindPotentialStereo(m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--csv", default=TARGET)
    args = ap.parse_args()

    from dgllife.utils import (CanonicalAtomFeaturizer, CanonicalBondFeaturizer,
                               mol_to_bigraph)
    from rdkit import Chem, RDLogger

    import numpy as np
    RDLogger.DisableLog("rdApp.*")
    af, bf = CanonicalAtomFeaturizer(), CanonicalBondFeaturizer(self_loop=False)

    def feats(smi):
        m = Chem.MolFromSmiles(smi)
        return mol_to_bigraph(m, node_featurizer=af, edge_featurizer=bf,
                              add_self_loop=False).ndata["h"].numpy()

    path = ROOT / args.csv
    rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
    print(f"[clean] {len(rows)} rows from {args.csv}\n")

    def feats_mol(m):
        return mol_to_bigraph(m, node_featurizer=af, edge_featurizer=bf,
                              add_self_loop=False).ndata["h"].numpy()

    renamed, refeat, bad = [], [], []
    for r in rows:
        n2 = clean_name(r["name"])
        m, key = normalise_mol(r["smiles"])
        if m is None:
            bad.append(r["name"])
            continue
        if n2 != r["name"]:
            renamed.append((r["name"], n2))
        # The artefact is invisible in the SMILES string, so compare features:
        # the normalised mol against the molecule as originally written, and
        # against the same molecule written without any stereo notation.
        f_norm, f_orig = feats_mol(m), feats(r["smiles"])
        flat = Chem.MolFromSmiles(Chem.MolToSmiles(
            Chem.MolFromSmiles(r["smiles"]), isomericSmiles=False))
        fixed = (flat is not None
                 and np.array_equal(np.sort(f_norm, axis=0),
                                    np.sort(feats_mol(flat), axis=0)))
        if not np.array_equal(f_norm, f_orig):
            refeat.append((n2, r["smiles"], fixed))
        r["_name"], r["_smiles"] = n2, Chem.MolToSmiles(m)
        r["_key"], r["_conn"] = key, key[:14]
        r["_stereo"] = stereo_specified(r["smiles"])

    print(f"[1] names cleaned: {len(renamed)}")
    for a, b in renamed:
        print(f"      {a:42s} -> {b}")
    if bad:
        print(f"\n    UNPARSEABLE (left untouched): {bad}")

    print(f"\n[2] materials whose atom features change once H bookkeeping is "
          f"normalised: {len(refeat)}")
    for n, s, fixed in refeat:
        print(f"      {n:34s} matches notation-free form: {fixed}   {s}")

    # Duplicates. Grouping on the full InChIKey misses the case that matters:
    # a compound entered twice, once with stereochemistry specified and once
    # without, has two different keys. Group on the connectivity block instead,
    # then merge only where one entry leaves stereochemistry unspecified -- that
    # is a less precise record of the same molecule. Two entries that each
    # specify *different* stereochemistry are genuinely different compounds
    # (geraniol and nerol) and must survive as two points.
    byconn: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("_conn"):
            byconn.setdefault(r["_conn"], []).append(r)

    mergeable, genuine = {}, {}
    for k, v in byconn.items():
        if len(v) < 2:
            continue
        spec = [r for r in v if r["_stereo"]]
        unspec = [r for r in v if not r["_stereo"]]
        if unspec and len(spec) <= 1:
            mergeable[k] = v
        else:
            genuine[k] = v

    print(f"\n[3] same compound entered twice (one without stereochemistry): "
          f"{len(mergeable)}")
    for k, v in mergeable.items():
        for r in v:
            print(f"      {r['_name']:34s} stereo={r['_stereo']}  "
                  f"source={r.get('source','')}  {r['_smiles']}")
    print(f"\n[4] distinct stereoisomers, kept separate: {len(genuine)}")
    for k, v in genuine.items():
        print(f"      {' / '.join(r['_name'] for r in v)}")

    # Merge each mergeable group into its richest record.
    drop = set()
    for k, v in mergeable.items():
        spec = [r for r in v if r["_stereo"]]
        keep = spec[0] if spec else v[0]
        for r in v:
            if r is keep:
                continue
            if r.get("source") == "curated":
                keep["source"] = "curated"
            if len(r["_name"]) < len(keep["_name"]):
                keep["_name"] = r["_name"]
            drop.add(id(r))

    merged = [r for r in rows if r.get("_key") and id(r) not in drop]

    byname: dict[str, list[dict]] = {}
    for r in merged:
        byname.setdefault(r["_name"], []).append(r)
    clash = {k: v for k, v in byname.items() if len(v) > 1}
    print(f"\n[5] remaining name collisions: {len(clash)}")
    for k, v in clash.items():
        print(f"      {k}: {len(v)} distinct compounds -- needs a manual name")

    print(f"\n[6] {len(rows)} rows -> {len(merged)} compounds "
          f"({len(rows) - len(merged)} merged away)")
    if clash:
        print("    refusing to write while names collide")
        return 1

    if not args.apply:
        print("\n[clean] dry run; nothing written. Re-run with --apply.")
        return 0

    out = [{"name": r["_name"], "smiles": r["_smiles"],
            "source": r.get("source", "")} for r in merged]
    out.sort(key=lambda r: r["name"].lower())
    bak = path.with_suffix(".csv.bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"[clean] original preserved at {bak.name}")
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "smiles", "source"])
        w.writeheader()
        w.writerows(out)
    print(f"[clean] wrote {len(out)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
