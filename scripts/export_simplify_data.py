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
from thresholds import describe as describe_thr, impute  # noqa: E402


#: Materials that must never be proposed for a fragrance, whoever owns them.
#: The bar for this list is a regulatory prohibition or a supplier saying so
#: in as many words -- not an opinion about how hard something is to dose.
#:
#: An earlier version of this list also held blackcurrant mercaptan, furfuryl
#: mercaptan, geosmin and the dimethyl sulfides, on the reasoning that a
#: material used at parts per billion cannot be dosed from a percept vector.
#: That reasoning was wrong twice over. It confuses "needs a small dose" with
#: "is not a material": blackcurrant mercaptan is in this perfumer's own
#: inventory at a 2% use level and is one of the most useful things on the
#: shelf. And the dosing worry was already handled better elsewhere -- a
#: candidate with no threshold figure is skipped when activity semantics are
#: on, which is the precise rule rather than a blanket one.
NEVER_PROPOSE = {
    "Musk Ambrette": "IFRA prohibited",
    "Musk Xylene": "IFRA prohibited",
    "2,6-Dibromophenol": "not for fragrance use per the supplier data; it is "
                         "here as a reference molecule for the iodine note "
                         "of shellfish",
    # Butylphenyl methylpropional. Prohibited in EU cosmetics from 1 March
    # 2022 as reprotoxic 1B, and prohibited by IFRA. It is still in older
    # formulas and in literature accords, so a pasted formula containing it
    # resolves and is flagged rather than refused -- what changes is that it
    # is never proposed as a substitution.
    "Lilial": "prohibited in EU cosmetics since March 2022 and by IFRA",
}

#: Files whose entries were added because a real formula asked for them by
#: name. A material someone put in a formula is evidence of availability in
#: the same way an inventory row is.
FORMULA_SOURCED = ("add_supplier_names.csv", "add_coffee_roast.csv")

#: Provenance strings that mark a library entry as having arrived for the
#: model's sake rather than from a formulating palette. Most are the GS-LF
#: training set.
#:
#: This is a label, not a filter, and the distinction was paid for. Excluding
#: them from proposals looked safe and was wrong: 2-isobutyl quinoline is
#: Pyralone and geosmin is sold by the gram, and both arrived here through the
#: training set, so both were being silently withheld. Nothing available here
#: can tell a perfumery material from a food volatile across 169 molecules,
#: and an inventory cannot stand in for it either -- that is one shelf, and
#: matching the inventory's own names through the alias table recovers only
#: three more, because the rest of its unmatched rows are naturals and bases.
#:
#: So they are offered and marked. The reader decides, which is the honest
#: place for a judgement this file cannot make.
REFERENCE_SOURCES = ("gs-lf", "constituent reference", "gc constituent",
                     "natural constituent")


def provenance():
    """name -> the source string recorded when the molecule was added."""
    path = ROOT / "library" / "input" / "expanded_materials.csv"
    out = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        n = (r.get("name") or "").strip()
        if n:
            out[n] = (r.get("source") or "").strip()
    return out


def formula_usage():
    """name -> how many formulas in this repository use it.

    Commonality has to come from evidence rather than from one shelf, and
    this is the evidence available: the repository's own query formulas and
    the published accords it benchmarks against. A material nobody here has
    been seen using is one to propose with a warning, not with confidence.
    """
    import glob

    from parse_formulas import classify as _classify

    lib = set()
    counts: dict[str, int] = {}
    pj = ROOT / "artifacts" / "percepts_fp3.json"
    if pj.is_file():
        lib = set(json.loads(pj.read_text(encoding="utf-8"))["molecules"])

    def bump(names):
        for n in set(names):
            counts[n] = counts.get(n, 0) + 1

    for cf in glob.glob(str(ROOT / "data/formulas/queries/**/*.csv"),
                        recursive=True):
        got = []
        for r in csv.DictReader(io.open(cf, encoding="utf-8-sig")):
            nm = (r.get("name") or r.get("material") or "").strip()
            if nm:
                kind, res = _classify(nm, lib)
                if kind == "material" and res in lib:
                    got.append(res)
        bump(got)

    for jf in glob.glob(str(ROOT / "data/formulas/published/*.json")):
        blob = json.loads(Path(jf).read_text(encoding="utf-8"))
        for fm in blob.get("formulas", []):
            got = []
            for r in (fm.get("ingredients") or []):
                if not isinstance(r, dict):
                    continue
                nm = (r.get("material") or r.get("name") or "").strip()
                if nm:
                    kind, res = _classify(nm, lib)
                    if kind == "material" and res in lib:
                        got.append(res)
            bump(got)
    return counts


def formula_sourced():
    """Names added to the library because a pasted formula named them."""
    out = set()
    for fn in FORMULA_SOURCED:
        path = ROOT / "library" / "input" / fn
        if not path.is_file():
            continue
        for r in csv.DictReader(io.open(path, encoding="utf-8")):
            n = (r.get("name") or "").strip()
            if n:
                out.add(n)
    return out


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
    prov = provenance()
    used = formula_usage()
    from_formula = formula_sourced()
    strength = load_strength()
    n_blocked = 0
    n_unverified = 0
    mats = []
    n_measured = 0
    for i, n in enumerate(names):
        lvl, thr = inv.get(n, (None, None))
        o = oils.get(n)
        st = strength.get(n)
        if st:
            n_measured += 1
        # Why a material may not be proposed. Being a reference molecule is
        # a soft reason; the named list is a hard one.
        why_not = NEVER_PROPOSE.get(n)
        unverified = False
        if not why_not and not o and n not in inv:
            src = prov.get(n, "").lower()
            unverified = any(k in src for k in REFERENCE_SOURCES)
        if why_not:
            n_blocked += 1
        if unverified:
            n_unverified += 1
        mats.append({
            "n": n,
            "noUse": why_not,
            # True when the molecule reached this library through the percept
            # model's training set rather than a formulating palette. It may
            # be an excellent material or it may be a food volatile; nothing
            # here knows which, so the page says so and offers it anyway.
            "unver": unverified or None,
            # Commonly used, on the evidence available: stocked by a working
            # perfumer, built from a GC composition, or named in a formula
            # here. Without one of those nobody in this project has been seen
            # using it, and a proposal is a guess about availability as well
            # as about smell.
            "com": bool(n in inv or o or used.get(n) or n in from_formula),
            "nf": used.get(n) or None,
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

    # Thresholds, filled and labelled. A material with no figure could not be
    # dosed and so could not be proposed at all, which is a worse answer than
    # a figure that is a factor of three out.
    raw_thr = [m["thr"] for m in mats]
    filled, tsrc, treport = impute(X, raw_thr)
    for m, v, srcv in zip(mats, filled, tsrc):
        st = strength.get(m["n"])
        # The measured correction still applies on top of an imputed figure:
        # it is a statement about the material, not about the figure.
        m["thr"] = (round(v / st["mult"], 6) if (st and v) else
                    (round(v, 6) if v else None))
        m["thrSrc"] = srcv
    print("  " + describe_thr(treport))

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
            "n_thresholds_compiled": treport["compiled"],
            "n_thresholds_imputed": treport["imputed"],
            "threshold_imputation": treport,
            "n_strength_measured": n_measured,
            "n_not_for_use": n_blocked,
            "n_unverified": n_unverified,
            "n_common": sum(1 for m in mats if m["com"]),
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
    print(f"  {n_blocked} materials may never be proposed; "
          f"{n_unverified} are offered but unverified as perfumery materials")
    print(f"  {sum(1 for m in mats if m['com'])} are commonly used "
          f"(stocked, built from GC, or named in one of "
          f"{len(set(used))} materials' worth of formulas here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
