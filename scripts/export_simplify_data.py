"""Everything the in-browser formula simplifier needs, and nothing else.

The atlas payload already carries the full 361 x 138 descriptor matrix, so the
forward direction was always computable in a browser. This writes the subset a
simplifier needs -- about 0.3 MB against the atlas's 4.1 -- so the page loads
fast and holds no receptor data, no projections and no distance matrices.

What goes in, and why each piece is needed rather than nice to have:

*The descriptor rows.* The target to match and the candidates to match it with.

*The alias table and the solvent list.* A pasted formula says "Hedione HC",
"E. Brass." and "DPG", and resolving those is the difference between reading a
formula and reading a quarter of it. 157 aliases and 19 solvents, which is 4 KB.

*Thresholds.* To turn shares of odour activity back into masses. These are a
compilation from published and web sources with some converted from air-phase
figures, not measurements, and the page says so where it uses them. Where a
blotter session has measured a material's relative strength, the compiled
figure is divided by that multiplier before it goes in the payload, so the
page and the CLI dose a formula identically without the page needing to know
anything about bins or standard errors.

*Use levels.* To keep a proposal inside what the perfumer would actually dose,
and to warn when it does not.

*Isolation.* The mean distance from a material to its five nearest neighbours.
A material with no near-equivalents reads as itself at any dose -- Calone, rose
oxide and raspberry ketone all stuck out of blends the solver was happy with --
so the simplifier excludes the isolated ones by default.

*The oils' own names and constituent lists.* So "Bergamot EO" resolves, and so
the page can say what an oil was built from.
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

from parse_formulas import ALIAS, SOLVENT  # noqa: E402
from strength import load_strength  # noqa: E402


def inventory():
    """name -> (use level %, threshold ppb). Both may be None."""
    path = ROOT / "data" / "inventory" / "inventory.csv"
    out: dict[str, tuple] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or "").strip()
        if not n:
            continue

        def num(key):
            v = (r.get(key) or "").strip()
            try:
                return float(v)
            except ValueError:
                return None

        out[n] = (num("top_pct"), num("water_threshold_ppb"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--out", default="artifacts/simplify_data.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    names = list(pj["molecules"])
    X = np.asarray(pj["probs"], float)

    import predict_formula as pf
    oils = pf.load_oils("library")
    oil_names = list(oils)
    if oil_names:
        X = np.vstack([X] + [np.asarray(oils[o]["percept"], float)[None, :]
                             for o in oil_names])
        names = names + oil_names

    # Isolation, computed over everything on the page so an oil is judged by
    # the same rule as a molecule.
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    iso = np.sort(D, axis=1)[:, :5].mean(1)
    iso_pct = 100.0 * np.argsort(np.argsort(iso)) / max(len(iso) - 1, 1)
    # The just-noticeable scale everything is reported against.
    yardstick = float(np.median(D.min(1)))

    inv = inventory()
    strength = load_strength()
    mats = []
    n_measured = 0
    for i, n in enumerate(names):
        lvl, thr = inv.get(n, (None, None))
        o = oils.get(n)
        st = strength.get(n)
        # Odour activity is mass over threshold, so "delivers more smell per
        # gram than its use level implies" is the same statement as "has a
        # lower effective threshold". Correcting the threshold here means
        # every dose calculation downstream picks it up for free.
        if st and thr:
            thr = round(thr / st["mult"], 6)
            n_measured += 1
        mats.append({
            "n": n,
            "p": [round(float(v), 3) for v in X[i]],
            "iso": round(float(iso_pct[i]), 1),
            "use": lvl,
            "thr": thr,
            "str": round(st["mult"], 3) if st else None,
            "strN": st["n"] if st else None,
            "own": bool(n in inv),
            "oil": bool(o),
            "built": (", ".join(
                f"{k} {100 * v:.0f}%" for k, v in
                sorted(o["composition_used"].items(), key=lambda kv: -kv[1])[:5])
                if o else None),
        })

    payload = {
        "descriptors": desc,
        "materials": mats,
        "aliases": {k: v for k, v in ALIAS.items()},
        "solvents": sorted(SOLVENT),
        # Which oil to take when several chemotypes of one plant tie. Without
        # it the page drops "Lavender Maillette France" for matching three
        # lavenders equally well.
        "oil_defaults": pf.oil_defaults(),
        "yardstick": round(yardstick, 4),
        "mean": [round(float(v), 4) for v in X.mean(0)],
        "meta": {
            "n_materials": len(mats),
            "n_oils": len(oil_names),
            "n_owned": sum(1 for m in mats if m["own"]),
            "n_thresholds": sum(1 for m in mats if m["thr"]),
            "n_strength_measured": n_measured,
            "n_strength": len(strength),
            "percept_model": pj.get("checkpoint", Path(args.percepts).stem),
            "percept_auc": pj.get("oof_mean_auc"),
        },
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    mb = out.stat().st_size / 1_048_576
    print(f"wrote {out}  ({mb:.2f} MB)")
    print(f"  {len(mats)} materials ({len(oil_names)} of them oils), "
          f"{payload['meta']['n_owned']} owned, "
          f"{payload['meta']['n_thresholds']} with a threshold")
    print(f"  {len(desc)} descriptors, {len(ALIAS)} aliases, "
          f"yardstick {yardstick:.3f}")
    if strength:
        print(f"  {n_measured} of {len(strength)} measured strengths applied "
              f"to a compiled threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
