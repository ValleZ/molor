"""Add bench-written descriptor labels to GS-LF, without touching GS-LF.

The percept model trains on a vendored upstream file,
vendor/openpom/.../curated_GS_LF_merged_4983.csv. Appending to that in place
would be wrong twice over: it would be overwritten by any update to the
vendored package, and "the GS-LF 4983" appears throughout this repository as a
fixed reference that other numbers are quoted against. So this writes a
separate augmented file and leaves the original alone.

What gets added, and why it is legitimate. Of the library's 22 molecules with
no GS-LF label, 16 are reference structures nobody formulates with. Five of
the rest have well-established published descriptions, and the model was
guessing badly on at least one of them: rose oxide came out `solvent, mint,
celery, herbal`, which is wrong about a material nobody mistakes for
anything else. The sixth, Cassifix, is argued out below.

What this does NOT do is fix a named molecule. Measured over the 378 library
molecules that carry a human label, the model puts a labelled descriptor in
its own top 5 about 45% of the time and its top 20 about 87% of the time,
and these new rows land in the same place as the upstream ones -- median
rank 10 against 6. Rose oxide still reads celery and mint, with `rose`
itself at 106 of 138. Three retrains differing by one or two of these rows
moved mean ROC-AUC over 0.8659-0.8675 and label-grounded rho over
0.474-0.538, which is re-initialisation noise and larger than any one row.
A label is one of 4988 and does not outvote a structure.

Each label is a published description reduced to the model's own 138-word
vocabulary. A published word with no descriptor is dropped rather than
approximated -- lychee for rose oxide, pistachio for the thiazole, iris for
the methyl ionone, jammy for the maltol ester -- which is the same rule the
modifier prompt asks of a language model, and the script says which it
dropped.

    python scripts/augment_labels.py
    python scripts/train_fp_v3.py --train-csv data/percept/gslf_plus_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: name -> (descriptors, the published description it was reduced from)
#: Taken from supplier and reference descriptions, not from this project's own
#: smelling notes, so it can live in a public repository.
BENCH_LABELS = {
    "Rose Oxide": (
        ["rose", "geranium", "green", "metallic", "fresh", "floral"],
        "green, rosy, geranium-like, metallic, with a lychee top"),
    "Maltol Isobutyrate": (
        ["caramellic", "sweet", "fruity", "strawberry", "berry"],
        "fruity-caramellic, jammy, strawberry; a slower, less sugary maltol"),
    "Pistachio Thiazole": (
        ["nutty", "roasted", "sulfurous", "savory", "hazelnut", "green"],
        "nutty, roasted, pistachio, with a sulfurous green edge"),
    "Methyl Ionone Gamma Coeur": (
        ["orris", "violet", "woody", "powdery", "floral"],
        "iris, violet, woody, powdery -- the classic methyl ionone profile"),
    # Settled at the blotter rather than guessed. `black currant` is one of
    # the 138, spelled with a space -- two descriptors are, it and `fruit
    # skin`, and a check for `blackcurrant` and `cassis` finds neither.
    #
    # Cassifix is described by suppliers the same way (fruity, green, cassis,
    # sparkle) and is deliberately NOT here, because that label lost an
    # argument with the structure and then with a nose.
    #
    # Labelling it from the supplier copy put `green` dead last of 138 in its
    # own prediction. The stored SMILES, CC1=CCC(C23CCCC(C)(C2)OC3)C1(C)C,
    # carries the campholenic cyclopentene shared by the sandalwood
    # synthetics, so the model reads amber, woody, cedar, sandalwood. A
    # blotter agreed with the structure rather than with the copy. Supplier
    # text for a specialty like this describes the effect it is sold for in a
    # cassis accord, not the smell of the molecule, and those are not the
    # same claim. Left unlabelled: the prediction is the better record here.
    "Thiocineole": (
        ["sulfurous", "green", "grapefruit", "black currant", "woody",
         "herbal"],
        "powerful sulphurous-green, blackcurrant and grapefruit, with "
        "subtle woody and herbal facets"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train",
        default="vendor/openpom/openpom/data/curated_datasets/"
                "curated_GS_LF_merged_4983.csv")
    ap.add_argument("--materials",
                    default="library/input/expanded_materials.csv")
    ap.add_argument("--out", default="data/percept/gslf_plus_bench.csv")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    src = ROOT / args.train
    rows = list(csv.DictReader(io.open(src, encoding="utf-8")))
    cols = list(rows[0].keys())
    desc = [c for c in cols if c not in ("nonStereoSMILES", "descriptors")]
    print(f"[aug] {src.name}: {len(rows)} rows, {len(desc)} descriptors")

    mats = {r["name"].strip(): (r.get("smiles") or "").strip()
            for r in csv.DictReader(io.open(ROOT / args.materials,
                                            encoding="utf-8"))}

    def flat(smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        # The key column is stereochemistry-stripped by name and by
        # construction; matching with stereochemistry misses every chiral or
        # E/Z molecule, which once reported 94 unlabelled where 22 was true.
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)

    have = {k for k in (flat((r.get("nonStereoSMILES") or "").strip())
                        for r in rows) if k}

    added = []
    for name, (words, note) in BENCH_LABELS.items():
        smi = mats.get(name)
        if not smi:
            print(f"  !! {name}: not in {args.materials}")
            continue
        key = flat(smi)
        if key is None:
            print(f"  !! {name}: SMILES will not parse")
            continue
        if key in have:
            print(f"  -- {name}: already labelled upstream, skipped")
            continue
        bad = [w for w in words if w not in desc]
        keep = [w for w in words if w in desc]
        if bad:
            print(f"  {name}: dropped {', '.join(bad)} -- no such descriptor")
        row = {c: "0" for c in cols}
        row["nonStereoSMILES"] = key
        row["descriptors"] = ";".join(keep)
        for w in keep:
            row[w] = "1"
        added.append((name, row, keep, note))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        for _, row, _, _ in added:
            w.writerow(row)

    print(f"\n[aug] wrote {args.out}: {len(rows)} upstream + {len(added)} "
          f"bench = {len(rows) + len(added)}")
    print(f"[aug] {src.name} itself is unchanged\n")
    for name, _, keep, note in added:
        print(f"  {name}")
        print(f"    {', '.join(keep)}")
        print(f"    from: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
