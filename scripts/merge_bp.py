"""Merge fetched boiling points, after checking the ones we could already check.

Reads ``data/inventory/bp_requests/reply_NN.csv`` as saved from a web session,
scores each batch against the control rows that batch carried -- materials
whose boiling point this inventory already records, mixed in unlabelled -- and
merges only the batches that pass.

Why a gate rather than a warning. A fabricated boiling point is invisible once
it is in a CSV: it has the right shape, the right magnitude, and no way to be
distinguished from a real one except by looking it up again, which is the work
being delegated. The controls make that detectable per session. A batch whose
controls are wrong is not evidence about the other fifty rows in it.

Two values are dropped rather than rescued:

*Reduced pressure.* Fragrance materials are often distilled under vacuum
because they decompose at atmospheric, so published figures mix the two.
Rhodinol is listed at 132 C at 4 mmHg and boils near 230 C at atmospheric,
which would move it from Mid-Base to Top. Converting needs an enthalpy of
vaporisation this project does not have, and a guessed one is worse than a
missing value.

*Unstated pressure* is kept but in its own tier, because the request asked for
the pressure explicitly and a source that gives none is weaker evidence than
one that says 760 mmHg.

Output is ``data/inventory/boiling_points.csv``, separate from inventory.csv
so the 127 hand-compiled figures stay distinguishable from fetched ones -- the
same reason thresholds carry `compiled` against `imputed`.

    python scripts/merge_bp.py
    python scripts/merge_bp.py --force     # merge a batch that failed its controls
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "data" / "inventory" / "bp_requests"
OUT = ROOT / "data" / "inventory" / "boiling_points.csv"

#: A control is wrong if it misses by more than this many degrees.
CTRL_TOL = 12.0
#: A batch fails if its median control error exceeds this, or if any single
#: control is off by more than CTRL_GROSS -- one 150-degree miss matters more
#: than a good median.
CTRL_MEDIAN_MAX = 10.0
CTRL_GROSS = 50.0


def parse_value(raw, unit):
    """A published figure to degrees Celsius, or None."""
    if raw is None:
        return None
    t = raw.strip().strip('"')
    if not t or t.upper() in ("UNKNOWN", "N/A", "NA", "-", ""):
        return None
    t = t.replace(",", "")
    # The range separator in "374-378" is not a minus sign. Reading it as one
    # averaged 374 with -378 to get -2, which then converted to -18.9 C and
    # looked like a plausible figure for a volatile material. Collapse a
    # hyphen or dash BETWEEN digits to a space first, so a genuine leading
    # minus still survives for the rare material that boils below zero.
    t2 = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " ", t)
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t2)]
    if not nums:
        return None
    v = float(np.mean(nums[:2])) if len(nums) >= 2 else nums[0]
    # The unit column decides, not the text: a value string can contain an F
    # for other reasons and guessing from it is how the above happened.
    u = (unit or "").strip().upper()
    if u.startswith("F") or (not u and re.search(r"F", t.upper())):
        v = (v - 32.0) * 5.0 / 9.0
    return v


def parse_pressure(raw):
    """Return ('atmospheric' | 'unstated' | 'reduced', as-given)."""
    t = (raw or "").strip().lower()
    if not t or t in ("unstated", "none", "-", "n/a"):
        return "unstated", t
    if "atm" in t and not re.search(r"\d", t.replace("1", "")):
        return "atmospheric", t
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if not nums:
        return "unstated", t
    v = nums[0]
    if "kpa" in t:
        return ("atmospheric" if 95 <= v <= 108 else "reduced"), t
    if "bar" in t:
        return ("atmospheric" if 0.95 <= v <= 1.08 else "reduced"), t
    if "atm" in t:
        return ("atmospheric" if 0.95 <= v <= 1.08 else "reduced"), t
    # default unit is mmHg / torr
    return ("atmospheric" if 700 <= v <= 800 else "reduced"), t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    known = {}
    for r in csv.DictReader(io.open(ROOT / "data/inventory/inventory.csv",
                                    encoding="utf-8-sig")):
        v = (r.get("boiling_point_c") or "").strip()
        if v:
            try:
                known[r["name"].strip()] = float(v)
            except ValueError:
                pass

    replies = sorted(REQ.glob("reply_*.csv"))
    if not replies:
        print(f"no replies found in {REQ.relative_to(ROOT)}")
        print("save each web answer as reply_01.csv, reply_02.csv, ...")
        return 1

    kept, stats = [], []
    for path in replies:
        rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
        ctrl, data = [], []
        for r in rows:
            n = (r.get("name") or "").strip()
            v = parse_value(r.get("bp_as_published"), r.get("unit"))
            tier, praw = parse_pressure(r.get("pressure"))
            rec = {"name": n, "inchikey": (r.get("inchikey") or "").strip(),
                   "bp_c": v, "ptier": tier, "pressure": praw,
                   "source": (r.get("source") or "").strip(),
                   "notes": (r.get("notes") or "").strip(),
                   "batch": path.name}
            (ctrl if n in known else data).append(rec)

        errs = [(abs(c["bp_c"] - known[c["name"]]), c) for c in ctrl
                if c["bp_c"] is not None and c["ptier"] != "reduced"]
        e = np.array([x for x, _ in errs]) if errs else np.array([])
        med = float(np.median(e)) if len(e) else float("nan")
        gross = [c["name"] for x, c in errs if x > CTRL_GROSS]
        ok = len(e) >= 3 and med <= CTRL_MEDIAN_MAX and not gross
        stats.append((path.name, len(rows), len(ctrl), len(e), med, gross, ok))
        print(f"\n{path.name}: {len(rows)} rows, {len(ctrl)} controls, "
              f"{len(e)} scorable")
        if len(e):
            print(f"  control error: median {med:.1f} C, worst "
                  f"{e.max():.1f} C, within {CTRL_TOL:.0f} C: "
                  f"{100 * float((e <= CTRL_TOL).mean()):.0f}%")
            for x, c in sorted(errs, key=lambda t: -t[0])[:3]:
                print(f"    {c['name'][:30]:30s} said "
                      f"{c['bp_c']:6.1f}  known {known[c['name']]:6.1f}  "
                      f"off {x:5.1f}")
        if gross:
            print(f"  GROSS MISS on {', '.join(gross)} -- batch rejected")
        if not ok and not args.force:
            print(f"  not merged. Re-run this batch, or pass --force.")
            continue
        if not ok:
            print(f"  FORCED in despite failing controls.")
        for d in data:
            if d["bp_c"] is None:
                continue
            if d["ptier"] == "reduced":
                continue
            kept.append(d)

    if not kept:
        print("\nnothing merged.")
        return 1

    # Dedupe, preferring a stated atmospheric pressure over an unstated one.
    best = {}
    for d in kept:
        cur = best.get(d["name"])
        if cur is None or (cur["ptier"] == "unstated"
                           and d["ptier"] == "atmospheric"):
            best[d["name"]] = d

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "inchikey", "bp_c", "pressure_tier", "pressure",
                    "source", "notes", "batch"])
        for n in sorted(best):
            d = best[n]
            w.writerow([n, d["inchikey"], round(d["bp_c"], 1), d["ptier"],
                        d["pressure"], d["source"], d["notes"], d["batch"]])
    atm = sum(1 for d in best.values() if d["ptier"] == "atmospheric")
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(best)} materials "
          f"({atm} at a stated atmospheric pressure, {len(best) - atm} with "
          f"the pressure unstated)")

    # A last sanity check that needs no external data: boiling point should
    # track molecular weight at about +0.7 on this kind of set, as it does on
    # the 127 already held. A much lower figure would say the merged column is
    # noise even though every row looked plausible.
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from scipy.stats import spearmanr
    RDLogger.DisableLog("rdApp.*")
    smi = {r["name"].strip(): (r.get("smiles") or "").strip()
           for r in csv.DictReader(io.open(
               ROOT / "library/input/expanded_materials.csv",
               encoding="utf-8"))}
    xs, ys = [], []
    for n, d in best.items():
        m = Chem.MolFromSmiles(smi.get(n, "") or "")
        if m is not None:
            xs.append(Descriptors.MolWt(m)); ys.append(d["bp_c"])
    if len(xs) >= 20:
        print(f"sanity: merged boiling point vs molecular weight "
              f"rho {spearmanr(xs, ys).statistic:+.3f} over {len(xs)} "
              f"(the 127 already held give +0.705)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
