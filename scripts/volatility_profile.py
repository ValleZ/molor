"""Split a formula by volatility and predict each tier separately.

The percept model returns one vector for a blend, as though every material
evaporated at once. For a formula built the way perfumes are actually built --
a small, intense, volatile top over a large, quiet, persistent base -- that
single vector is the base, and it can contradict the formula's whole intent.

The case that prompted this. A tomato-leaf accord containing cis-3-hexenol,
cis-3-hexenyl acetate, Triplal, Stemone and trans-2-hexenal came back as
cedar, amber and dry, with no green note at all. Nothing was wrong with the
arithmetic: the formula is 44% Iso E Super, which has a compiled threshold of
0.71 ppb and carries 84% of the blend's odour activity. The engine was
describing the drydown correctly and saying nothing about the first two
minutes, which is where the entire brief lives.

The data to fix it was already in the repository. The inventory records a
volatility class for 165 materials -- Top, Top-Mid, Mid, Mid-Base, Base -- and
every green material in that formula is Top while Iso E Super is Base. An oil
has no class of its own, so it inherits one from its constituents, weighted by
its GC composition, which is the same trick that makes oils predictable at
all here.

What this is not. It is not a model of evaporation. There is no time axis, no
substantivity, no rate: it is the same percept model run on three subsets of
the same formula. A top-note tier that reads green tells you the volatile
fraction is green, not how long that lasts. Nothing here measures persistence,
and the tiers come from a perfumer's judgement rather than from vapour
pressure, which this project looked for and could not source.

    python scripts/volatility_profile.py data/formulas/queries/query-3.csv
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from parse_formulas import classify  # noqa: E402

#: The tiers, in the order a blotter reveals them.
TIERS = ("Top", "Top-Mid", "Mid", "Mid-Base", "Base")
#: Collapsed for reporting, because five tiers over a dozen ingredients gives
#: tiers with one material in them.
GROUP = {"Top": "top", "Top-Mid": "top", "Mid": "middle",
         "Mid-Base": "base", "Base": "base"}
DOSE_EXP = 0.6


def volatility_classes():
    """name -> one of TIERS, for everything that has one."""
    out = {}
    path = ROOT / "data" / "inventory" / "inventory.csv"
    if path.is_file():
        for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
            n = (r.get("name") or "").strip()
            v = (r.get("volatility_class") or "").strip()
            if n and v in TIERS:
                out[n] = v
    return out


def oil_classes(vol, naturals_path="artifacts/naturals.json"):
    """An oil inherits a tier from its constituents, weighted by GC share.

    Petitgrain is mostly linalyl acetate and linalool, both Top-Mid, so it
    behaves as a top note; vetiver is sesquiterpenoids and behaves as a base.
    Without this every oil would be unclassified, and oils are often most of
    a formula's weight.
    """
    path = ROOT / naturals_path
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    rank = {t: i for i, t in enumerate(TIERS)}
    out = {}
    for oil, v in blob.get("naturals", {}).items():
        comp = v.get("composition_used") or {}
        num = den = 0.0
        for con, share in comp.items():
            if con in vol:
                num += share * rank[vol[con]]
                den += share
        if den > 0:
            out[oil] = TIERS[int(round(num / den))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("formula", help="a formula CSV, or - for a pasted sheet")
    ap.add_argument("--payload", default="artifacts/simplify_data.json")
    ap.add_argument("--weight", choices=("activity", "mass"),
                    default="activity")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    d = json.loads((ROOT / args.payload).read_text(encoding="utf-8"))
    desc, mats = d["descriptors"], d["materials"]
    names = [m["n"] for m in mats]
    lib = set(names)
    X = np.asarray([m["p"] for m in mats], float)
    mean = np.asarray(d["mean"], float)
    sd = X.std(0) + 1e-9
    thr = {m["n"]: m.get("thr") for m in mats}

    import predict_formula as pf
    oils = [m["n"] for m in mats if m.get("oil")]

    vol = volatility_classes()
    vol.update(oil_classes(vol))

    rows = []
    if args.formula == "-":
        text = sys.stdin.read()
        for line in text.splitlines():
            parts = [p.strip() for p in line.replace("\t", "|").split("|")
                     if p.strip()]
            if len(parts) >= 2:
                try:
                    rows.append((parts[0], float(parts[-1])))
                except ValueError:
                    pass
    else:
        for r in pf.read_formula(ROOT / args.formula
                                 if not Path(args.formula).is_absolute()
                                 else Path(args.formula)):
            rows.append((r["name"], r["neat"]))

    res, miss, unclassed = [], [], []
    for nm, amt in rows:
        kind, out = classify(nm, lib)
        if kind == "solvent":
            continue
        if not (kind == "material" and out in lib):
            hit, _ = pf.match_oil(nm, oils)
            out = hit
        if not out:
            miss.append(nm)
            continue
        tier = vol.get(out)
        if tier is None:
            unclassed.append(out)
            continue
        res.append((out, float(amt), GROUP[tier], tier))

    if not res:
        raise SystemExit("nothing resolved with a volatility class")

    print(f"[vol] {len(res)} ingredients placed, {len(unclassed)} with no "
          f"volatility class, {len(miss)} unresolved")
    # How much of the formula the tiers below do NOT cover. This matters more
    # than it looks: on a coffee accord the unclassified ingredients were the
    # pyrazines, so every tier was missing the roast and the tier view was as
    # blind to the subject as the single vector had been.
    placed = sum(a for _, a, _, _ in res)
    skipped = sum(float(a) for nm, a in rows
                  if nm in unclassed or nm in miss)
    if unclassed or miss:
        print(f"      the tiers below cover {100 * placed / (placed + skipped):.0f}% "
              f"of the resolved weight; the rest has no class and appears in "
              f"no tier")
    if unclassed:
        print(f"      no class: {', '.join(sorted(set(unclassed)))}")
    if miss:
        print(f"      unresolved: {', '.join(miss)}")

    total = sum(a for _, a, _, _ in res)
    print(f"\n  {'tier':8s} {'share of mass':>14s} {'ingredients':>12s}")
    for g in ("top", "middle", "base"):
        sub = [r for r in res if r[2] == g]
        if not sub:
            continue
        print(f"  {g:8s} {100 * sum(a for _, a, _, _ in sub) / total:13.1f}% "
              f"{len(sub):12d}   "
              + ", ".join(n.split(' (')[0][:18] for n, _, _, _ in sub)[:76])

    print(f"\n  predicted character per tier, weighted by "
          f"{args.weight}:\n")
    for g in ("top", "middle", "base"):
        sub = [r for r in res if r[2] == g]
        if not sub:
            continue
        idx = [names.index(n) for n, _, _, _ in sub]
        w = np.asarray([a for _, a, _, _ in sub], float)
        if args.weight == "activity":
            w = np.asarray([a / (thr[n] ** DOSE_EXP)
                            for n, a, _, _ in sub], float)
        w = w / w.sum()
        z = ((w @ X[idx]) - mean) / sd
        top = ", ".join(f"{desc[j]} {z[j]:+.1f}" for j in np.argsort(-z)[:8])
        print(f"  {g:8s} {top}")

    # And the whole blend, which is what the rest of the project reports.
    idx = [names.index(n) for n, _, _, _ in res]
    w = np.asarray([a for _, a, _, _ in res], float)
    if args.weight == "activity":
        w = np.asarray([a / (thr[n] ** DOSE_EXP) for n, a, _, _ in res], float)
    w = w / w.sum()
    z = ((w @ X[idx]) - mean) / sd
    print(f"\n  {'all':8s} "
          + ", ".join(f"{desc[j]} {z[j]:+.1f}" for j in np.argsort(-z)[:8]))
    print("\n  The single vector is what every other script here reports, and "
          "on a formula\n  with a large base it is the base. Read the tiers "
          "when the brief is a top note.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
