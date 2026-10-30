"""Propagate GS-LF's implied hierarchy, which the labels do not enforce.

165 molecules in GS-LF carry a fruit tag -- banana, pear, cherry -- without the
fruity tag. 162 carry a floral child without floral, and 112 a citrus child
without citrus. So the label set behaves as 138 unrelated dimensions: banana and
apple are no more related than banana and woody, and a molecule described as
"banana" is not described as fruity at all.

That has two costs. Any similarity computed over these labels understates how
alike two molecules are when they are described at different levels of
specificity. And a model is trained to predict "fruity" as a thing separate from
every fruit, which is incoherent.

The fix is to set a parent wherever a child is set. The judgement is in which
relations count, and the list below is deliberately conservative: only
near-universally accepted hypernyms, and nothing where the parent names a
different quality rather than a broader one. Specifically excluded, with
reasons:

  musk under animal   -- musk animalic and indolic animalic are different
                         qualities, and the two cluster separately in every
                         representation measured here
  lavender under floral -- usually read as aromatic/herbal in perfumery
  hay under green     -- dry rather than green
  cheesy under dairy  -- fermented rather than dairy
  creamy under dairy  -- used for lactones, vanilla and sandalwood as much as
                         for dairy
  grapefruit under fruity -- it is a citrus, and citrus is not placed under
                         fruity either, since in perfumery practice they are
                         sibling families rather than parent and child

Propagation is transitive but the tree here is only one level deep, so no
cascade arises.

Whether this helps is an empirical question and is not assumed. The effect is
reported on the labels themselves, and the retrained model must then be scored
against the graded panels and the perfumer's notes -- none of which it is
trained on -- because scoring it against the labels it was just given would
only confirm that it learned them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ("vendor/openpom/openpom/data/curated_datasets/"
       "curated_GS_LF_merged_4983.csv")
OUT = ROOT / "data" / "percept" / "gslf_hierarchical.csv"

#: parent -> children. Only clear hypernyms; see the module docstring for what
#: is deliberately left out.
HIERARCHY = {
    "fruity": ["apple", "apricot", "banana", "berry", "black currant", "cherry",
               "grape", "melon", "peach", "pear", "pineapple", "plum",
               "raspberry", "strawberry", "tropical"],
    "citrus": ["bergamot", "grapefruit", "lemon", "orange"],
    "floral": ["geranium", "hyacinth", "jasmin", "lily", "muguet",
               "orangeflower", "rose", "violet", "hawthorn"],
    "woody": ["cedar", "sandalwood", "vetiver"],
    "green": ["grassy", "leafy", "weedy"],
    "sulfurous": ["alliaceous", "garlic", "onion", "cabbage"],
    # "creamy" is deliberately absent: it is used freely for lactones,
    # vanilla and sandalwood, so routing it to dairy tripled that label
    # (123 -> 372) on relations that do not hold.
    "dairy": ["buttery", "milky"],
    "nutty": ["almond", "hazelnut"],
    "spicy": ["cinnamon", "clove"],
    "roasted": ["coffee", "popcorn"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    df = pd.read_csv(ROOT / SRC)
    smi = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi, "descriptors")]
    Y = df[tasks].to_numpy(dtype=float).copy()
    idx = {t: i for i, t in enumerate(tasks)}

    missing = {p: [c for c in ch if c not in idx]
               for p, ch in HIERARCHY.items() if any(c not in idx for c in ch)}
    missing = {p: m for p, m in missing.items() if m}
    if missing:
        print(f"[hier] descriptors named in the hierarchy but absent from "
              f"GS-LF: {missing}")
    bad_parents = [p for p in HIERARCHY if p not in idx]
    if bad_parents:
        raise SystemExit(f"parent descriptors not in GS-LF: {bad_parents}")

    before = Y.sum()
    report = {}
    for parent, children in HIERARCHY.items():
        ch = [c for c in children if c in idx]
        if not ch:
            continue
        pi = idx[parent]
        child_any = Y[:, [idx[c] for c in ch]].max(axis=1)
        inconsistent = int(((child_any == 1) & (Y[:, pi] == 0)).sum())
        Y[:, pi] = np.maximum(Y[:, pi], child_any)
        report[parent] = {
            "children": ch,
            "molecules_gaining_the_parent": inconsistent,
            "parent_count_before": int(df[parent].sum()),
            "parent_count_after": int(Y[:, pi].sum()),
        }
        print(f"  {parent:10s} +{inconsistent:4d} molecules  "
              f"({report[parent]['parent_count_before']} -> "
              f"{report[parent]['parent_count_after']})  from {len(ch)} children")

    added = int(Y.sum() - before)
    print(f"\n[hier] {added} labels added across {len(report)} parents "
          f"({100 * added / before:.1f}% more positive labels)")
    print(f"[hier] mean labels per molecule {before / len(df):.2f} -> "
          f"{Y.sum() / len(df):.2f}")

    out = pd.DataFrame(Y, columns=tasks)
    out.insert(0, smi, df[smi].to_numpy())
    if "descriptors" in df.columns:
        out["descriptors"] = df["descriptors"].to_numpy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[hier] wrote {args.out}")

    (Path(args.out).parent / "gslf_hierarchy.json").write_text(json.dumps({
        "source": SRC,
        "hierarchy": {p: v["children"] for p, v in report.items()},
        "excluded": {
            "musk under animal": "different qualities; they cluster separately",
            "lavender under floral": "read as aromatic/herbal in perfumery",
            "hay under green": "dry rather than green",
            "cheesy under dairy": "fermented rather than dairy",
            "citrus under fruity": "sibling families in practice, not parent "
                                   "and child",
        },
        "labels_added": added,
        "per_parent": report,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
