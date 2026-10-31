"""A one-session bench sheet that measures relative strength, not thresholds.

A detection threshold is the wrong thing to measure for this project, and
measuring one properly is an ascending forced-choice staircase per material.
What the dose conversion needs is relative strength at working concentration --
how much smell a material delivers per unit of mass -- and there is a much
cheaper way to get it.

The perfumer's own maximum use levels are already an inverse strength estimate
made by a nose. If those levels were right, a blotter of each material
prepared *at its own use level* would smell about equally strong. They are not
right, so the blotters will not match, and the mismatch is the correction
wanted. That turns an eight-order-of-magnitude measurement into a small one:
every judgement compares blotters already within a factor of a few of each
other, which is the range a nose compares well in.

Two things in the first version of this sheet were wrong and are worth
recording, because both came from an imagined bench rather than a real one.

*It asked for drops.* A perfumer who weighs formulas has a balance, and a drop
is a bad unit: its volume depends on viscosity, so the same dropper gives very
different amounts of ethyl butyrate and of Galaxolide. Everything is in grams
now, with the diluent fixed at 2.00 g so the amount of stock to weigh is
around a tenth of a gram -- two decimal places on any bench balance.

*It hid the material names to keep the duplicates blind, and that is not
possible.* Preparing a blotter requires knowing what goes on it. A solo
perfumer cannot be blind during preparation, and pretending otherwise with a
separate key file just made the sheet unusable. So the names are on the
preparation sheet where they are needed, and the blinding that is actually
available is applied to the judging instead: the blotters are judged in a
different order from the one they were made in, printed separately, so the
repeated materials are not adjacent in either pass and nothing is judged in
the order it was mixed.

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

#: Grams of diluent per blotter. Chosen so the stock to weigh lands near a
#: tenth of a gram, which any bench balance reads to two decimals, and so that
#: 26 blotters need about 55 g of DPG in total.
DILUENT_G = 2.00
#: Below this the balance is guessing, so a weaker stock is used instead.
MIN_WEIGHABLE_G = 0.03


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
    path = ROOT / "artifacts" / "threshold.json"
    if not path.is_file():
        return set()
    tj = json.loads(path.read_text(encoding="utf-8"))
    return {n for n, s in zip(tj["molecules"], tj["source"]) if s == "measured"}


def weigh_plan(use_pct: float):
    """Grams of which stock, into DILUENT_G of DPG, to reach the use level.

    m grams of an s% stock in d grams of diluent gives s * m / (m + d), so
    m = d * c / (s - c). The strongest stock that still needs a weighable
    amount wins, because fewer dilution steps means less compounding error.
    """
    best = None
    for stock in (100.0, 10.0, 1.0, 0.1):
        if stock <= use_pct:
            continue
        m = DILUENT_G * use_pct / (stock - use_pct)
        g = round(m, 2)
        if g < MIN_WEIGHABLE_G:
            continue
        got = stock * g / (g + DILUENT_G)
        if best is None:
            best = (stock, g, got)
    if best is None:
        # Nothing weighable: use the weakest stock and say what it gives.
        stock = 0.1
        g = MIN_WEIGHABLE_G
        best = (stock, g, stock * g / (g + DILUENT_G))
    return best


def spread_order(strips, rng, tries=5000):
    """A permutation in which no material appears twice close together."""
    gap = max(4, len(strips) // 6)
    order = rng.permutation(len(strips))
    for _ in range(tries):
        cand = rng.permutation(len(strips))
        pos, ok = {}, True
        for slot, k in enumerate(cand):
            n = strips[k]
            if n in pos and slot - pos[n] < gap:
                ok = False
                break
            pos[n] = slot
        if ok:
            return cand, gap
    return order, gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--duplicates", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diluent", default="DPG")
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
    mix_order, gap = spread_order(strips, rng)
    # A second, different order for the judging pass. Preparation cannot be
    # blind, so this is the blinding that is actually available: nothing is
    # smelled in the order it was mixed.
    judge_order, _ = spread_order(strips, rng)
    while list(judge_order) == list(mix_order):
        judge_order, _ = spread_order(strips, rng)

    number = {}
    rows = []
    for pos, k in enumerate(mix_order, 1):
        n = strips[k]
        use = inv[n]["use"]
        stock, grams, got = weigh_plan(use)
        number[k] = pos
        rows.append({"blotter": pos, "material": n, "stock_pct": stock,
                     "grams_of_stock": grams, "grams_of_diluent": DILUENT_G,
                     "gives_pct": round(got, 4),
                     "your_use_level_pct": round(use, 4),
                     "anchor": int(n in anchors), "bin": "", "notes": ""})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_num = {r["blotter"]: r for r in rows}
    total_dil = DILUENT_G * len(rows)
    print("=" * 76)
    print(f"STRENGTH SESSION 1 — {len(rows)} blotters, "
          f"{len(picked)} materials".center(76))
    print("=" * 76)
    print(f"""
WHAT YOU NEED

  {len(rows)} blotter strips
  {len(rows)} small vials (or a tray with {len(rows)} wells) — one dilution per blotter
  {args.diluent}, about {total_dil:.0f} g in total
  a balance reading 0.01 g, and a pencil

  Not water. Water is for threshold work, where the material has to be in
  solution; on a blotter it evaporates and leaves the material neat, which is
  the opposite of a controlled dilution. {args.diluent} is odourless and stays put.

STEP 1 — MIX (table A below, in blotter order)

  For each row: weigh the stated grams of that material's stock into a vial,
  add {DILUENT_G:.2f} g of {args.diluent}, stir or swirl. One vial per row.
  Dip one blotter, write only the BLOTTER NUMBER on it in pencil.
  Set it aside and move on. About 45 minutes for all {len(rows)}.

  Each dilution is that material at the level you told me you use it at. If
  your use levels were right every blotter would smell about equally strong.

STEP 2 — SMELL (table B below, a different order)

  Judge them in the order given in table B, not the order you mixed them.
  For each blotter, one question only:

      is this stronger or weaker than most of the blotters in front of me?

      --  clearly weaker than most        -   a little weaker
      0   about the same as most          +   a little stronger
      ++  clearly stronger than most

  Write the answer against that blotter number. Pause 30 to 60 seconds between
  blotters. If they all start smelling the same, stop and finish another day:
  that is adaptation, not data. Half a session is still useful.

  Go round twice if you can. The second pass is quick once you know the range,
  and it is where most of the corrections happen.

STEP 3 — TYPE IT IN

  Put each answer in the bin column of
      {args.out}
  next to its blotter number, then run
      python scripts/apply_strength.py {args.out}

ABOUT THE REPEATS

  {len(dups)} of the {len(rows)} blotters are a material that appears twice, never within
  {gap} positions of each other in either table. You cannot be blind while
  mixing -- you have to know what you are weighing -- so these do not test
  that. They test whether your judging is consistent, which is why table B is
  in a different order from table A. If a material lands two bins from itself,
  the session cannot resolve one bin and apply_strength.py will refuse to
  write the numbers.

WHY THESE {args.top} MATERIALS

  A tenfold error in their strength moves your formulas further than the
  smallest difference a nose can detect. They carry about two thirds of
  everything the dose conversion rests on. The other 137 figures in the
  inventory can keep their compiled values indefinitely.
""")
    print("=" * 76)
    print("TABLE A — MIX IN THIS ORDER")
    print("=" * 76)
    print(f"  {'no.':>4s}  {'weigh':>7s}  {'of stock':>9s}  "
          f"{'into':>6s}  material")
    print("  " + "-" * 72)
    for r in rows:
        star = " *" if r["anchor"] else "  "
        print(f"  {r['blotter']:4d}  {r['grams_of_stock']:7.2f} g  "
              f"{r['stock_pct']:>8g}%  {DILUENT_G:5.2f} g  "
              f"{r['material']}{star}")
    print("  " + "-" * 72)
    print(f"  * = one of the {sum(r['anchor'] for r in rows)} anchors, a "
          f"material whose threshold was really measured")
    print(f"\n  Total {args.diluent}: {total_dil:.0f} g. "
          f"Total stock weighed: {sum(r['grams_of_stock'] for r in rows):.2f} g.")

    print("\n" + "=" * 76)
    print("TABLE B — SMELL IN THIS ORDER, AND RECORD")
    print("=" * 76)
    print(f"  {'step':>4s}  {'blotter':>7s}   bin  (-- / - / 0 / + / ++)")
    print("  " + "-" * 56)
    for step, k in enumerate(judge_order, 1):
        print(f"  {step:4d}  {number[k]:7d}   ____")
    print("  " + "-" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
