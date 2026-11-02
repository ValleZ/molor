"""Write pasteable requests for the boiling points the inventory is missing.

232 of the 358 proposable molecules have no boiling point, and boiling point
is the best predictor of volatility tier in this data -- rho +0.774 against
+0.544 for molecular weight, and weight adds nothing once boiling point is
known (partial +0.001). So the gap is worth filling.

The danger is not missing data, it is confident wrong data. A language model
asked for 232 boiling points will produce 232 plausible numbers whether or not
it knows any of them, and a fabricated 180 C is indistinguishable from a real
one once it is in a CSV. Two defences are built in here.

**Controls.** Each batch carries a handful of molecules whose boiling point
this inventory already records, mixed in and unlabelled. On merge their
answers are scored against the known values, which measures the error rate of
that particular session rather than assuming one. A batch whose controls come
back wrong is discarded rather than argued with.

**Pressure.** Fragrance materials are routinely distilled under vacuum because
they decompose at atmospheric pressure, so published figures are a mix of the
two and the reduced-pressure ones are far lower. Rhodinol is listed at 132 C
at 4 mmHg and boils near 230 C at atmospheric; taken at face value it moves
from Mid-Base to Top. The request therefore demands the pressure with every
value and the merge step drops anything not atmospheric rather than
converting it, since the Clausius-Clapeyron correction needs an enthalpy of
vaporisation that is not available either.

    python scripts/make_bp_request.py
    python scripts/make_bp_request.py --batch-size 40
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "inventory" / "bp_requests"

HEAD = """# Boiling points, batch {n} of {total}

I need **published atmospheric-pressure boiling points** for the fragrance
materials in the table below. Please look them up rather than recalling them.

## Rules, in order of importance

1. **If you cannot find a published value, write `UNKNOWN`.** A missing row
   costs me nothing. A guessed row corrupts a dataset I cannot easily audit,
   because a wrong boiling point looks exactly like a right one. Do not
   estimate from structure, do not interpolate from similar molecules, and do
   not infer from a homologous series.
2. **Give the pressure for every value.** Many of these are distilled under
   vacuum, and a figure at 4 mmHg is not comparable to one at 760 mmHg. If the
   source states no pressure, write `unstated` rather than assuming.
3. **Report the value as the source gives it**, including the unit. Do not
   convert Fahrenheit to Celsius and do not average a range -- I will do both,
   so that a mistake is mine and traceable.
4. **Name the source** for each value: PubChem CID, a supplier's technical data
   sheet, The Good Scents Company, a Sigma-Aldrich listing, a paper. One short
   identifier is enough.
5. If several sources disagree, give the one you judge most reliable and note
   the disagreement in the notes column.

`INCHIKEY` is the identifier to search on -- it is exact, where a trade name
often is not. PubChem accepts it directly:
`https://pubchem.ncbi.nlm.nih.gov/#query=<INCHIKEY>`

## Output format

One CSV block, these columns exactly, one row per input row, same order, no
commentary before or after:

```
name,inchikey,bp_as_published,unit,pressure,source,notes
```

Example of each kind of answer:

```
Vanillin,MWOOGOJBHIARFG-UHFFFAOYSA-N,285-286,C,760 mmHg,PubChem CID 1183,
2-Acetyl Thiazole,XRIZQKKTQVRBHM-UHFFFAOYSA-N,89-91,C,12 mmHg,PubChem CID 520108,vacuum only; no atmospheric figure found
Some Captive,AAAAAAAAAAAAAA-UHFFFAOYSA-N,UNKNOWN,,,,"supplier TDS gives flash point only"
```

## The materials

| name | INCHIKEY | SMILES |
|---|---|---|
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--controls", type=int, default=6,
                    help="known-answer rows mixed into each batch")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    known = {}
    for r in csv.DictReader(io.open(ROOT / "data/inventory/inventory.csv",
                                    encoding="utf-8-sig")):
        v = (r.get("boiling_point_c") or "").strip()
        if v:
            try:
                known[r["name"].strip()] = float(v)
            except ValueError:
                pass

    smiles = {r["name"].strip(): (r.get("smiles") or "").strip()
              for r in csv.DictReader(io.open(
                  ROOT / "library/input/expanded_materials.csv",
                  encoding="utf-8"))}

    pay = json.loads((ROOT / "artifacts/simplify_data.json")
                     .read_text(encoding="utf-8"))
    want = [m["n"] for m in pay["materials"]
            if not m.get("noUse") and not m.get("unver") and not m.get("oil")
            and m["n"] not in known]

    def key(n):
        m = Chem.MolFromSmiles(smiles.get(n, "") or "")
        return (Chem.MolToInchiKey(m), Chem.MolToSmiles(m)) if m else (None, None)

    rows, nokey = [], []
    for n in want:
        ik, smi = key(n)
        if ik:
            rows.append((n, ik, smi))
        else:
            nokey.append(n)

    # Controls, spread across the whole boiling range so a session that is
    # only wrong about heavy materials is still caught.
    rng = random.Random(args.seed)
    ctrl_pool = sorted(known.items(), key=lambda kv: kv[1])
    picks, step = [], max(1, len(ctrl_pool) // 40)
    for i in range(0, len(ctrl_pool), step):
        n = ctrl_pool[i][0]
        ik, smi = key(n)
        if ik:
            picks.append((n, ik, smi))
    rng.shuffle(picks)

    batches = [rows[i:i + args.batch_size]
               for i in range(0, len(rows), args.batch_size)]
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("batch_*.md"):
        f.unlink()

    ci = 0
    for b, batch in enumerate(batches, 1):
        use = list(batch)
        for _ in range(args.controls):
            if ci < len(picks):
                use.append(picks[ci]); ci += 1
        rng.shuffle(use)
        body = "".join(f"| {n} | {ik} | {smi} |\n" for n, ik, smi in use)
        txt = HEAD.format(n=b, total=len(batches)) + body
        (OUT / f"batch_{b:02d}.md").write_text(txt, encoding="utf-8")

    print(f"[bp] {len(want)} proposable molecules have no boiling point")
    print(f"[bp] {len(rows)} have a usable InChIKey; {len(nokey)} do not"
          + (f" ({', '.join(nokey[:4])})" if nokey else ""))
    print(f"[bp] wrote {len(batches)} batches to "
          f"data/inventory/bp_requests/, {args.batch_size} materials each")
    print(f"[bp] plus {min(ci, len(picks))} control rows with answers this "
          f"repository already holds, mixed in unlabelled")
    print(f"\nPaste one batch per message. Save each reply as")
    print(f"  data/inventory/bp_requests/reply_01.csv  (matching the batch "
          f"number)")
    print(f"then run scripts/merge_bp.py, which scores the controls before it "
          f"merges anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
