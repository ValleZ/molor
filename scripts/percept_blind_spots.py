"""Library materials the percept model cannot tell apart, and why.

Found while reading an SDS: the percept model gives every n-alkane from C7 to
C28 the same 2048-bit ECFP4 fingerprint -- literally the same eight on-bits.
A circular fingerprint of radius 2 sees a chain of CH2 groups as the same
handful of substructures no matter how long the chain is, so chain length is
invisible to it. The model's only remaining handle is the 209-descriptor block
(molecular weight, cLogP and so on).

That generalises past alkanes, and it reaches real materials. This script
groups the library by exact fingerprint and reports what the model then does
with each group. Two kinds of collision come out, and they are not equally bad:

  CHAIN LENGTH -- the aliphatic aldehydes C-7 to C-12, the ethyl esters, the
  gamma lactones. Molecular weight differs, so the descriptor block separates
  them a little, and the model does produce different vectors. Whether it
  separates them ENOUGH is the question this measures: against the 0.439
  just-noticeable yardstick the project uses elsewhere, most neighbouring
  pairs in these series land below it.

  GEOMETRY AND STEREOCHEMISTRY -- geraniol against nerol, geranyl against
  neryl acetate, menthone against isomenthone. Here the fingerprint is
  identical AND so are molecular weight and almost every constitutional
  descriptor, so the model's output is identical to the last decimal. Not
  approximately: the vectors are the same array. These materials smell
  different and the model has no way to say so.

The second kind matters for the deployed simplifier, which will offer either
member of such a pair as a perfect substitute for the other at zero cost. That
is a real defect rather than a curiosity, so it is checked here and the page
warns about it.

    python scripts/percept_blind_spots.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--materials",
                    default="library/input/expanded_materials.csv")
    ap.add_argument("--yardstick", type=float, default=0.439)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    names, desc = list(pj["molecules"]), list(pj["descriptors"])
    P = np.asarray(pj["probs"], float)
    Z = (P - P.mean(0)) / (P.std(0) + 1e-9)
    pos = {n: i for i, n in enumerate(names)}

    smi = {r["name"].strip(): (r.get("smiles") or "").strip()
           for r in csv.DictReader(io.open(ROOT / args.materials,
                                           encoding="utf-8"))}

    # The demonstration that started this: a homologous series collapsing.
    print("The fingerprint cannot see chain length at all.")
    print("  n-alkane ECFP4 radius 2, 2048 bits:")
    ref = None
    for k in (7, 10, 16, 23, 28):
        m = Chem.MolFromSmiles("C" * k)
        b = frozenset(gen.GetFingerprint(m).GetOnBits())
        same = "" if ref is None else ("  identical" if b == ref
                                       else "  DIFFERENT")
        print(f"    C{k:<3} {len(b)} on-bits  MW {Descriptors.MolWt(m):6.1f}"
              f"{same}")
        ref = b

    groups: dict[frozenset, list[str]] = {}
    for n in names:
        s = smi.get(n)
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        groups.setdefault(
            frozenset(gen.GetFingerprint(m).GetOnBits()), []).append(n)
    coll = {k: v for k, v in groups.items() if len(v) > 1}
    affected = sum(len(v) for v in coll.values())
    print(f"\n{len(coll)} groups of library materials share one fingerprint, "
          f"{affected} materials in all, of {len(names)}.")

    identical, nearly = [], []
    print(f"\n{'group':52s} {'min':>7s} {'max':>7s}  verdict")
    print("-" * 86)
    for v in sorted(coll.values(), key=lambda x: -len(x)):
        idx = [pos[n] for n in v if n in pos]
        if len(idx) < 2:
            continue
        V = P[idx]
        d = [float(np.linalg.norm(V[a] - V[b]))
             for a in range(len(idx)) for b in range(a + 1, len(idx))]
        mn, mx = min(d), max(d)
        # Molecular weight is what the descriptor block has left to work with.
        mws = [Descriptors.MolWt(Chem.MolFromSmiles(smi[n])) for n in v]
        same_mw = max(mws) - min(mws) < 0.01
        if mx == 0.0:
            verdict = "IDENTICAL output"
            identical.append(v)
        elif mx < args.yardstick:
            verdict = f"all below the {args.yardstick} yardstick"
            nearly.append(v)
        else:
            verdict = "separated, by molecular weight alone"
        tag = "  [same MW too]" if same_mw else ""
        label = ", ".join(sorted(v))
        print(f"{label[:52]:52s} {mn:7.4f} {mx:7.4f}  {verdict}{tag}")

    if identical:
        print(f"\n{len(identical)} group(s) get a bit-identical percept "
              f"vector. The model cannot")
        print(f"distinguish these at all, and the simplifier will offer "
              f"either as a perfect")
        print(f"substitute for the other:")
        for v in identical:
            i = pos[sorted(v)[0]]
            top = ", ".join(desc[j] for j in np.argsort(-Z[i])[:4])
            print(f"   {' / '.join(sorted(v)):44s} both read: {top}")
        print("\nThese are geometric or stereo isomers, so neither the "
              "fingerprint nor the\nconstitutional descriptors differ. "
              "Fixing it needs a featurisation that\ncarries geometry -- "
              "which is what the MPNN's bond features were for, and that\n"
              "encoder annihilates them (see the GCN vs MPNN section).")
    if nearly:
        print(f"\n{len(nearly)} further group(s) differ by less than one "
              f"just-noticeable step,")
        print(f"so the model separates them only nominally:")
        for v in nearly:
            print(f"   {', '.join(sorted(v))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
