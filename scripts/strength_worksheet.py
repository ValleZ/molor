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
import re
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
#: Below this a 0.01 g balance quantises badly -- 0.04 g is plus or minus 12%
#: -- and neat aldehydes are unpleasant to handle in such amounts. A weaker
#: stock is used instead, which makes the weighed mass larger.
MIN_WEIGHABLE_G = 0.10
#: The weighed mass this aims for when it has a choice.
TARGET_G = 0.35


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


def stocked_dilutions():
    """Which dilutions of each material the perfumer actually keeps.

    The inventory sheet has a free-text column for this -- "1%, 0.1%",
    "10% out fridge", "R-6.6 10%" -- and reading it matters: without it this
    sheet asked for neat methyl benzoate and neat Undecavertol where a 10%
    stock is on the shelf, and for a 10% aldehyde C-10 where only 1% and 0.1%
    exist. That sheet could not be followed.

    Two materials are stocked *only* as a dilution, and that is recorded in the
    material's own name -- "Cinnamaldehyde, 10% DPG" -- not in the dilutions
    column. Reading only the column asked for neat cinnamaldehyde and neat
    strawberry furanone from a bench that has neither. So the name is parsed
    too, and a material whose every row carries a strength is marked as having
    no neat stock at all.

    Returns ({name: [pcts]}, {names with no neat stock}, have_sheet).

    The file holding this is the perfumer's own writing and is not in the
    repository, so its absence is normal and simply means every decade is
    assumed available.
    """
    path = ROOT / "data" / "inventory" / "perfumery_inventory.csv"
    if not path.is_file():
        return {}, set(), False

    def pcts_in(text):
        found = []
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text or ""):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if 0 < v < 100:
                found.append(v)
        return found

    out, has_neat, known = {}, set(), set()
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        raw = (r.get("material") or "").strip()
        if not raw:
            continue
        # "Cinnamaldehyde, 10% DPG" is one shelf item: the base name, stocked
        # at 10%. "Anisaldehyde, Natural" is a qualifier, not a strength.
        name_pcts = pcts_in(raw)
        n = re.split(r",", raw)[0].strip() if name_pcts else raw
        known.add(n)
        if not name_pcts:
            has_neat.add(n)
        out.setdefault(n, []).extend(name_pcts + pcts_in(r.get("dilutions")))
    out = {n: sorted(set(v), reverse=True) for n, v in out.items() if v}
    return out, (known - has_neat), True


def measured_anchors():
    path = ROOT / "artifacts" / "threshold.json"
    if not path.is_file():
        return set()
    tj = json.loads(path.read_text(encoding="utf-8"))
    return {n for n, s in zip(tj["molecules"], tj["source"]) if s == "measured"}


def weigh_plan(use_pct: float, available, has_neat=True):
    """Grams of which stock, into DILUENT_G of DPG, to reach the use level.

    m grams of an s% stock in d grams of diluent gives s * m / (m + d), so
    m = d * c / (s - c). Among the stocks that exist on the shelf and give a
    weighable amount, the one landing nearest TARGET_G wins: a larger weighed
    mass is a smaller relative error on a 0.01 g balance, and it keeps neat
    aldehydes out of the session whenever a dilution of them exists.

    A stock that already sits at the use level needs no vial: the blotter goes
    straight into it. Cinnamaldehyde is stocked at 10% and used at 10%, and
    without this case it fell through every branch and was dropped.
    """
    for stock in available:
        if abs(stock - use_pct) <= 0.02 * use_pct:
            return stock, 0.0, stock, None
    options = []
    for stock in available:
        if stock <= use_pct:
            continue
        g = round(DILUENT_G * use_pct / (stock - use_pct), 2)
        if g < MIN_WEIGHABLE_G or g > 5.0:
            continue
        got = stock * g / (g + DILUENT_G)
        options.append((abs(g - TARGET_G), stock, g, got))
    if options:
        _, stock, g, got = min(options)
        return stock, g, got, None
    # Nothing on the shelf reaches it in one step. Dropping the material is
    # the wrong answer -- ethyl butyrate is third on the influence list and
    # vanished from one draft of this sheet -- so make an intermediate first,
    # which is what a bench would do anyway.
    if not has_neat:
        # No neat material to make an intermediate from.
        return None, None, None, None
    for mid in (10.0, 1.0, 0.1):
        if mid <= use_pct:
            continue
        g = round(DILUENT_G * use_pct / (mid - use_pct), 2)
        if g < MIN_WEIGHABLE_G or g > 5.0:
            continue
        # The intermediate itself: 2 g of it, from neat.
        mid_g = round(2.0 * mid / 100.0, 2)
        if mid_g < MIN_WEIGHABLE_G:
            continue
        got = mid * g / (g + DILUENT_G)
        return mid, g, got, (mid, mid_g, round(2.0 - mid_g, 2))
    return None, None, None, None


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
    stocked, no_neat, have_sheet = stocked_dilutions()
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
    rows, unreachable = [], []
    for pos, k in enumerate(mix_order, 1):
        n = strips[k]
        use = inv[n]["use"]
        # Neat plus whatever dilutions the inventory records. With no sheet to
        # read, assume the usual decades.
        has_neat = n not in no_neat
        avail = ([100.0] if has_neat else []) + (
            stocked.get(n, []) if have_sheet else [10.0, 1.0, 0.1])
        stock, grams, got, premix = weigh_plan(
            use, sorted(set(avail), reverse=True), has_neat)
        if stock is None:
            unreachable.append(n)
            continue
        number[k] = pos
        rows.append({"blotter": pos, "material": n, "stock_pct": stock,
                     "grams_of_stock": grams,
                     "grams_of_diluent": DILUENT_G if grams else 0.0,
                     "gives_pct": round(got, 4),
                     "your_use_level_pct": round(use, 4),
                     "premix_pct": premix[0] if premix else "",
                     "premix_neat_g": premix[1] if premix else "",
                     "premix_dpg_g": premix[2] if premix else "",
                     "anchor": int(n in anchors), "bin": "", "notes": ""})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Renumber contiguously: the filter above can leave gaps, and a sheet
    # that skips from 11 to 14 while table B still lists 12 is unusable.
    renum = {}
    for newno, r in enumerate(rows, 1):
        renum[r["blotter"]] = newno
        r["blotter"] = newno
    number = {k: renum[v] for k, v in number.items() if v in renum}
    judge_order = [k for k in judge_order if k in number]
    by_num = {r["blotter"]: r for r in rows}
    total_dil = sum(r["grams_of_diluent"] for r in rows)
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
    premixed = {}
    for r in rows:
        star = " *" if r["anchor"] else "  "
        if r["premix_pct"] and r["material"] in premixed:
            # A repeated material needs its intermediate made once, not twice.
            print(f"  {r['blotter']:4d}  use the {r['premix_pct']:g}% of "
                  f"{r['material']} you made at {premixed[r['material']]}")
        elif r["premix_pct"]:
            premixed[r["material"]] = r["blotter"]
        if r["premix_pct"] and r["blotter"] == premixed.get(r["material"]):
            print(f"  {r['blotter']:4d}  first make {r['premix_pct']:g}% of "
                  f"{r['material']}: {r['premix_neat_g']:.2f} g neat + "
                  f"{r['premix_dpg_g']:.2f} g {args.diluent}")
        if not r["grams_of_stock"]:
            print(f"  {r['blotter']:4d}  {'--':>7s}    "
                  f"{r['stock_pct']:>8g}%  {'no DPG':>6s}  "
                  f"{r['material']}{star}  <- dip straight into the stock")
            continue
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
