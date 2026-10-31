"""Which thresholds are worth measuring, ranked by what they would change.

Measuring a detection threshold properly is an afternoon per material: an
ascending forced-choice series, several triads per step, rest intervals long
enough to avoid adaptation. There are 157 figures in the inventory. Nobody is
going to measure 157 of anything, and they would not need to: most of them
never move an answer.

So this ranks them by influence rather than by suspicion. For every formula in
the repository, each ingredient's threshold is perturbed by a factor of ten
and the resulting shift in the blend's predicted percept vector is measured --
the same sensitivity analysis that correctly named the facets a perfumer
reported sticking out of three blends. A material's score is the largest shift
it causes anywhere, times the number of formulas it appears in.

Two things raise a material up this list: being present in a lot of formulas,
and carrying a lot of odour activity for its mass. Two things keep it off:
never being used, and being weak enough that a tenfold error changes nothing.

What the list is for: deciding where an afternoon goes. The top twenty here
cover most of what the dose conversion actually rests on, and the long tail can
keep its compiled figure indefinitely.

    python scripts/threshold_worklist.py --top 25
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixtures import combine  # noqa: E402
from parse_formulas import classify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="artifacts/threshold_worklist.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    names = list(pj["molecules"])
    P = {n: np.asarray(r, float) for n, r in zip(names, pj["probs"])}

    import predict_formula as pf
    oils = pf.load_oils("library")
    for o, v in oils.items():
        P[o] = np.asarray(v["percept"], float)

    inv = {}
    for r in csv.DictReader(io.open(ROOT / "data/inventory/inventory.csv",
                                    encoding="utf-8-sig")):
        n = (r.get("name") or "").strip()
        v = (r.get("water_threshold_ppb") or "").strip()
        lp = (r.get("logp") or "").strip()
        if n:
            try:
                inv[n] = (float(v) if v else None,
                          float(lp) if lp else None,
                          float(r.get("top_pct") or 0) or None)
            except ValueError:
                inv[n] = (None, None, None)

    lib = set(P)
    forms = []
    for csvf in sorted(glob.glob(str(ROOT / "data/formulas/queries/**/*.csv"),
                                 recursive=True)):
        rows = pf.read_formula(Path(csvf))
        ing = []
        for r in rows:
            kind, nm = classify(r["name"], lib)
            if kind == "material" and nm in P:
                ing.append((nm, r["neat"]))
            else:
                hit, _ = pf.match_oil(r["name"], list(oils))
                if hit:
                    ing.append((hit, r["neat"]))
        agg: dict[str, float] = defaultdict(float)
        for n, a in ing:
            agg[n] += a
        if len(agg) >= 3:
            forms.append((Path(csvf).stem, sorted(agg.items(),
                                                  key=lambda kv: -kv[1])))
    print(f"[worklist] {len(forms)} formulas with three or more resolved "
          f"ingredients\n")

    shift = defaultdict(float)
    appears = defaultdict(int)
    for label, ing in forms:
        nm = [n for n, _ in ing]
        m = np.array([a for _, a in ing], float)
        m = m / m.sum()
        th = np.array([(inv.get(n, (None, None, None))[0] or np.nan)
                       for n in nm])
        if np.isnan(th).all():
            continue
        th = np.where(np.isnan(th), np.nanmedian(th), th)
        V = np.stack([P[n] for n in nm])
        act = m / th
        act = act / act.sum()
        base = combine("weighted-mean", V, act)
        for i, n in enumerate(nm):
            appears[n] += 1
            for f in (10.0, 0.1):
                t2 = th.copy()
                t2[i] *= f
                a2 = m / t2
                a2 = a2 / a2.sum()
                d = float(np.linalg.norm(
                    combine("weighted-mean", V, a2) - base))
                if d > shift[n]:
                    shift[n] = d

    rows = []
    for n in shift:
        thr, lp, use = inv.get(n, (None, None, None))
        rows.append({"name": n, "worst_shift": shift[n],
                     "formulas": appears[n],
                     "score": shift[n] * appears[n],
                     "threshold": thr, "logp": lp, "use": use})
    rows.sort(key=lambda r: -r["score"])

    print(f"  {'material':30s} {'score':>6s} {'worst':>6s} {'in':>3s} "
          f"{'ppb now':>9s} {'logP':>5s}  water series?")
    print("  " + "-" * 82)
    for r in rows[: args.top]:
        # Above logP 5 a water series stops being meaningful at the top of the
        # range: the material is no longer dissolving, so the nominal
        # concentration and the headspace stop tracking each other. At
        # threshold levels almost everything dissolves, which is why only a
        # couple of these are flagged.
        warn = ("solubility -- use a solvent series"
                if (r["logp"] or 0) > 5 else "yes")
        print(f"  {r['name'][:30]:30s} {r['score']:6.2f} {r['worst_shift']:6.3f} "
              f"{r['formulas']:3d} {(r['threshold'] if r['threshold'] else 0):9g} "
              f"{(r['logp'] if r['logp'] else 0):5.1f}  {warn}")

    tot = sum(r["score"] for r in rows) or 1.0
    cum = sum(r["score"] for r in rows[: args.top]) / tot
    print(f"\n[worklist] the top {args.top} of {len(rows)} carry "
          f"{cum:.0%} of the total influence")
    (ROOT / args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[worklist] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
