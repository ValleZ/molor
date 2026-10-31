"""A one-session bench sheet that measures relative strength, not thresholds.

A detection threshold is the wrong thing to measure for this project, and
measuring one properly is an ascending forced-choice staircase per material.
What the dose conversion actually needs is relative strength at working
concentration -- how much smell a material delivers per unit of mass -- and
there is a far cheaper way to get it.

The trick is that the perfumer's own maximum use levels are already an inverse
strength estimate made by a nose. If those levels were perfect, a strip of each
material prepared *at its own use level* would smell about equally strong. They
are not perfect, so the strips will not match, and the mismatch is exactly the
correction wanted. That turns an eight-order-of-magnitude measurement into a
small one: every judgement compares strips already within a factor of a few of
each other, which is the range a nose compares well in.

So the session is one strip per material, each at that material's own use
level, sorted into five bins from clearly weaker than the group to clearly
stronger. One pass, no dilution ladder, no staircase, no reference material to
keep fresh. Twenty materials is about forty sniffs with rests: one sitting.

Three things make the result trustworthy rather than merely quick.

*The order is randomised*, so the sheet is not sorted by anything a nose could
infer.

*Blind duplicates.* Two or three materials appear twice under different strip
numbers, never within a few strips of each other. The disagreement between a
pair is this session's own error bar, and without it there is no way to tell a
real one-bin difference from noise.

*Anchors.* A few materials whose thresholds were genuinely measured rather than
compiled are included, so the resulting scale can be tied to absolute units
later if that ever matters.

    python scripts/strength_worksheet.py --top 20
    # smell, fill in the bin column, then:
    python scripts/apply_strength.py data/inventory/strength_session1.csv
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

#: Drops of diluent each strip is made up in.
DILUENT_DROPS = 20


def inventory():
    out = {}
    path = ROOT / "data" / "inventory" / "inventory.csv"
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or "").strip()
        if not n:
            continue

        def num(k):
            v = (r.get(k) or "").strip()
            try:
                return float(v)
            except ValueError:
                return None

        out[n] = {"use": num("top_pct"), "thr": num("water_threshold_ppb"),
                  "logp": num("logp")}
    return out


def measured_anchors():
    """Materials whose threshold was measured rather than compiled."""
    path = ROOT / "artifacts" / "threshold.json"
    if not path.is_file():
        return set()
    tj = json.loads(path.read_text(encoding="utf-8"))
    return {n for n, s in zip(tj["molecules"], tj["source"]) if s == "measured"}


def dilution_plan(use_pct: float):
    """Stock strength and drop count that land on a material's use level.

    Drops because a bench has droppers, and the ratios only need to be right
    to within a few percent: the whole measurement is a five-bin judgement, so
    weighing to a milligram would be false precision.

    d drops of a stock into DILUENT_DROPS of diluent give
    stock * d / (d + DILUENT_DROPS), not stock * d / DILUENT_DROPS. The first
    version of this treated the 20 drops as the total, which is 5% out at one
    drop and a third out at ten -- half a bin.
    """
    # Fewest drops wins among the options that are close enough. Minimising
    # the error alone chose twenty drops of a 10% stock over two drops of a
    # 100% one for a 5% target, which is ten times the pipetting for a 5%
    # gain in accuracy that a five-bin judgement cannot see.
    TOL = 0.12
    options = []
    for stock in (100.0, 10.0, 1.0, 0.1):
        if stock <= use_pct:
            continue
        want = DILUENT_DROPS * use_pct / (stock - use_pct)
        drops = max(1, int(round(want)))
        got = stock * drops / (drops + DILUENT_DROPS)
        err = abs(got / use_pct - 1.0)
        if 1 <= drops <= 20:
            options.append((stock, drops, err, got))
    close = [o for o in options if o[2] <= TOL]
    best = (min(close, key=lambda o: (o[1], o[2])) if close
            else (min(options, key=lambda o: o[2]) if options else None))
    if best is None:
        stock, drops = 0.1, 1
        best = (stock, drops, 0.0, stock * drops / (drops + DILUENT_DROPS))
    return best[0], best[1], best[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20,
                    help="how many materials from the influence worklist")
    ap.add_argument("--duplicates", type=int, default=3,
                    help="blind repeats, which give the session its error bar")
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worklist", default="artifacts/threshold_worklist.json")
    ap.add_argument("--out", default="data/inventory/strength_session1.csv")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    inv = inventory()
    anchors = measured_anchors()
    wl_path = ROOT / args.worklist
    if not wl_path.is_file():
        raise SystemExit(f"{args.worklist} is missing; run "
                         f"scripts/threshold_worklist.py first")
    wl = json.loads(wl_path.read_text(encoding="utf-8"))

    picked, seen = [], set()
    for r in wl:
        n = r["name"]
        if n in seen or n not in inv or not inv[n]["use"]:
            continue
        seen.add(n)
        picked.append(n)
        if len(picked) >= args.top:
            break
    # Anchors join even if they are low on the worklist: they are what would
    # let this scale be converted to absolute units.
    for n in sorted(anchors):
        if len(picked) >= args.top + args.anchors:
            break
        if n in inv and inv[n]["use"] and n not in seen:
            seen.add(n)
            picked.append(n)

    rng = np.random.default_rng(args.seed)
    dups = [str(x) for x in rng.choice(picked,
                                       min(args.duplicates, len(picked)),
                                       replace=False)]
    strips = picked + dups
    # A blind duplicate next to itself is not blind: a first version of this
    # sheet put the same material on strips 2 and 3.
    gap = max(4, len(strips) // 6)
    order = rng.permutation(len(strips))
    for _ in range(5000):
        cand = rng.permutation(len(strips))
        pos, ok = {}, True
        for slot, k in enumerate(cand):
            n = strips[k]
            if n in pos and slot - pos[n] < gap:
                ok = False
                break
            pos[n] = slot
        if ok:
            order = cand
            break

    rows, key = [], []
    for pos, k in enumerate(order, 1):
        n = strips[k]
        use = inv[n]["use"]
        stock, drops, got = dilution_plan(use)
        rows.append({"strip": pos, "stock_pct": stock,
                     "drops_of_stock": drops,
                     "drops_of_diluent": DILUENT_DROPS,
                     "gives_pct": round(got, 4), "bin": "", "notes": ""})
        key.append({"strip": pos, "material": n,
                    "use_level_pct": round(use, 4),
                    "gives_pct": round(got, 4),
                    "anchor": int(n in anchors)})

    # The sheet to fill in carries no material names, so it can sit open on
    # the bench without spoiling the blind duplicates. The key is a separate
    # file that apply_strength.py joins back.
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keypath = out.with_name(out.stem + "_key.csv")
    with open(keypath, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)

    print("=" * 74)
    print("STRENGTH SESSION 1".center(74))
    print("=" * 74)
    print("""
WHAT YOU ARE DOING

  Each strip is one material, diluted to the level you already said you use it
  at. If your use levels were perfect every strip would smell about as strong
  as every other. They will not. Sorting that mismatch into five bins is the
  whole measurement.

WHAT YOU NEED

  Blotter strips, a dropper, your usual diluent. Nothing else.

HOW TO PREPARE

  Per strip: the stated drops of that material's stock into 20 drops of
  diluent, mix, dip a strip, label it with the strip NUMBER only.""")
    print(f"  Do not write the material on the strip -- {len(dups)} of the "
          f"{len(strips)} strips repeat")
    print("  a material, and knowing which would spoil it.")
    print("""
HOW TO SMELL

  In number order. For each strip ask only one question: is this stronger or
  weaker than most of the strips in front of me? Mark one bin.

    --  clearly weaker than most        -   a little weaker
    0   about the same as most          +   a little stronger
    ++  clearly stronger than most

  Pause 30 to 60 seconds between strips. If everything starts smelling the
  same, stop and finish another day -- that is adaptation, not data. Half a
  session is still useful.

  Go round twice if you can. The second pass is quick once you know the range,
  and it is where most of the corrections happen.
""")
    print(f"WHY IT IS WORTH THE HOUR\n")
    print(f"  These {args.top} materials were chosen because a tenfold error "
          f"in their strength")
    print("  moves your formulas further than the smallest difference a nose")
    print("  can detect. They carry roughly two thirds of everything the dose")
    print("  conversion rests on. The rest of the inventory can keep its")
    print("  compiled figure indefinitely.\n")
    print("-" * 74)
    print("  strip   drops of stock   into   material (leave blank)")
    print("-" * 74)
    for r in rows:
        print(f"  {r['strip']:5d}   {r['drops_of_stock']:3d} of "
              f"{r['stock_pct']:>5g}%      20     {'_' * 30}")
    print("-" * 74)
    print(f"\n  {len(strips)} strips, {len(picked)} distinct materials, "
          f"{len(dups)} repeated blind and never within {gap} strips")
    print("  of each other.")
    print(f"\n  Fill in the bin column of {args.out}")
    print(f"  then run: python scripts/apply_strength.py {args.out}")
    print(f"\n  That file has no material names in it either. The mapping is "
          f"in\n  {keypath.relative_to(ROOT)}, which you never need to open.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
