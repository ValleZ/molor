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

from parse_formulas import ALIAS, SOLVENT, library_name  # noqa: E402
from strength import load_strength  # noqa: E402
from thresholds import assemble as assemble_thr
from thresholds import describe as describe_thr  # noqa: E402


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
    # Isovaleraldehyde. A malt/apple-scented trace volatile of fermentation
    # and of essential-oil constituents -- sold (if at all) as a flavor or
    # analytical reagent, not as a perfumery formulating material. The
    # formula engine kept picking it for dark-rose/leather briefs on a
    # small sweet/earthy lift; in those accords it is a trace constituent,
    # not a bottle on the shelf.
    "3-Methylbutanal": "trace volatile, not purchasable as a formulating material",
    # The leaf aldehydes. Volatile green-note trace constituents of
    # essential oils and fruit (cis-3-hexenal = "the smell of cut grass"),
    # sold (if at all) as flavor/aroma chemicals in tiny quantities, not
    # as perfumery formulating materials. Both are fleeting top-notes
    # with high-impact green/fatty characters that the fit over-weights
    # in herbal/green briefs.
    "(E)-2-Hexenal": "trace volatile leaf aldehyde, not a formulating material",
    "(Z)-3-Hexenal": "trace volatile leaf aldehyde, not a formulating material",
    # 2-Heptanone. A banana/crest-malty trace volatile of essential oils,
    # cheese and fruit -- a flavor/aroma chemical, not a perfumery
    # formulating material. High-impact fruity/fatty notes that the fit
    # over-weights wherever a brief asks for fruity lift.
    "Heptan-2-one": "trace volatile ketone, not a formulating material",
    # Styrene. A trace constituent of styrax balsam (0.5% in Hafioglu's
    # direct GC; the distilled oil is 70-93% of it, but that is a
    # different material and the resinoid keeps only traces). It enters
    # the library so Styrax Resinoid can be built from its cited GC
    # composition; as a bottle it is a plastics monomer, flammable and
    # not something anyone formulates with, and its solvent/plastic
    # vector is exactly the kind of high-impact note the fit over-weights.
    "Styrene": "trace volatile monomer of styrax, not a formulating material",
    # The styrax/benzoin rebuild constituents. They enter the library so
    # the resinoids can be built from their cited GC compositions, but as
    # bottles they are not sold as perfumery materials: they are balsam
    # trace constituents / reagent chemicals. Cinnamic alcohol and benzyl
    # cinnamate ARE shelf materials and stay proposable; these four are not.
    "Cinnamic Acid": "balsam constituent (styrax), not sold as a formulating material",
    "3-Phenylpropyl Cinnamate": "styrax trace constituent, not sold as a formulating material",
    "Coniferyl Benzoate": "benzoin resin carrier (the smell IS the resin), not sold neat",
    "Benzoic Acid": "benzoin constituent and preservative reagent, not a perfumery material",
    # The guaiacwood rebuild constituents. Guaiacwood EO IS the shelf
    # form (the oil is the sold product); the isolated molecules are
    # obscure reagent/naturals-isolate items a home perfumer cannot
    # practically buy, and the engine kept dosing Guaiol at 40 parts
    # (a vector duplicate of the oil) in woody/smoky briefs.
    "Guaiol": "guaiacwood isolate, sold only as a niche naturals reference — use the EO instead",
    "Bulnesol": "guaiacwood isolate, sold only as a niche naturals reference — use the EO instead",
    "Guaiene Beta": "guaiacwood trace hydrocarbon constituent, not sold as a formulating material",
    "Eudesmol Beta": "guaiacwood constituent and reagent chemical, not sold as a formulating material",
    # Vetiver. Khusimol is the sesquiterpene alcohol that carries the
    # vetiver root note, and it entered this library as a vetiver GC
    # constituent with an IMPUTED threshold and no CAS number or IFRA
    # figure. The engine kept proposing it for woody/earthy briefs: a
    # vector duplicate of the oil with none of the dosing data. Flagged
    # rather than proposed; vetiver is voiced through the EO.
    "Khusimol": "vetiver constituent with an imputed threshold and no CAS/IFRA data — use the vetiver EO instead",
    # Labdanum. Labdane is the diterpene skeleton of labdanum resin: it is what
    # the resin is built from, not a bottle. It reaches the library as a
    # labdanum GC constituent with an IMPUTED threshold and no CAS or IFRA
    # figure, and the engine proposed it at 10 parts in a lilac formula on a
    # small balsamic lift. Labdanum is voiced through the absolute.
    "Labdane": "labdanum diterpene constituent, sold only as a reference standard — use Labdanum Absolute instead",
    # Solvents and carriers. These entered the library from the Fraterworks
    # sweep, where they are sold under product_type "Aromachem" like any
    # other material, so they became formulable candidates: the engine
    # proposed DPG in a lilac formula at 0.2% and it satisfied residual
    # cheaply because its percept vector is near-flat - which also appears
    # to terminate the greedy growth early (n=12 returned 5 materials).
    # They are things you DISSOLVE a material in, never things you smell;
    # the material sweep should have excluded them at import.
    "Dipropylene Glycol (DPG)": "carrier solvent, not a perfumery material",
    "Carbitol®": "glycol ether solvent, not a perfumery material",
    "Triethyl Citrate": "carrier/plasticiser solvent, not a perfumery material",
    "Diethyl Phthalate": "carrier solvent (DEP), not a perfumery material",
    "Isopropyl Myristate": "carrier/emollient ester, not a perfumery material",
    # The myrrh rebuild constituents. None is a shelf material: the
    # furanosesquiterpenes are the smell of myrrh but exist in commerce
    # only as phytochemical reference standards -- the Fraterworks
    # catalogue carries no myrrh constituent and no curzerene-type
    # material either, and the perfumer's inventory holds no myrrh row.
    # Myrrh is voiced through the "Myrrh EO" reconstruction only.
    "Furanoeudesma-1,3-diene": "the myrrh smell itself, but exists only as a myrrh constituent; sold as an analytical reference standard, not a formulating material",
    "Curzerene": "myrrh furanoelemane sold only as a phytochemical reagent, not a formulating material",
    "Lindestrene": "myrrh furanoeudesmane, exists only as a myrrh constituent, not a formulating material",
    "Furanodiene": "myrrh/zedoary furanogermacradiene, thermally labile reagent chemical, not a formulating material",
    "2-Acetoxyfuranodiene": "myrrh resin constituent, not sold as a formulating material",
    "2-Methoxyfuranodiene": "myrrh trace constituent, not sold as a formulating material",
    "Germacrone": "turmeric/geranium sesquiterpene ketone and phytochemical reagent, not a perfumery material",
    "Elemene Beta": "pharmaceutical phytochemical (the Chinese elemene anticancer adjuvant), not a perfumery material",
    "Germacrene B": "reference sesquiterpene hydrocarbon, thermally labile, not sold as a formulating material",
    # The oakmoss rebuild constituents (af81a65 lichen acids). None is a
    # shelf material: lichen metabolites exist in commerce as phytochemical
    # reference standards, and orcinol is a known skin sensitizer. They
    # enter the library so Oakmoss Absolute can be built from its cited
    # composition; Oakmoss Absolute itself stays proposable.
    "Orcinol": "oakmoss lichen metabolite and sensitizer, exists only as a reference standard, not a formulating material",
    "Methyl Beta-Orcinol Carboxylate": "oakmoss lichen acid, not sold as a formulating material",
    "Methyl Orsellinate": "oakmoss lichen acid, not sold as a formulating material",
    "Ethyl Everninate": "oakmoss lichen acid, not sold as a formulating material",
    # The rest of the GC constituents of the vetiver, patchouli, cedar and
    # sandalwood rebuilds. Same rule as the myrrh and oakmoss blocks above:
    # they enter the library so an oil can be rebuilt from a cited
    # composition, and none of them is a bottle -- if they are sold at all it
    # is as a phytochemical reference standard. The 2026-11 IFRA pass flagged
    # exactly these for having no CAS anywhere; the CAS is now known (PubChem)
    # and the material is still unbuyable, so they are blocked by name rather
    # than by an absence of data, which was only ever a symptom.
    "Khusimone": "vetiver GC constituent; the ketone is a reference standard, and vetiver is voiced through the EO",
    "Isovalencenol": "vetiver GC constituent, sold only as a phytochemical reference standard",
    "Vetiselinenol": "vetiver GC constituent, sold only as a phytochemical reference standard",
    "Vetivone Alpha": "vetiver GC constituent, a reference standard rather than a stock item",
    "Vetivone Beta": "vetiver GC constituent, a reference standard rather than a stock item",
    "Zizanoic Acid": "vetiver GC acid; a phytochemical reagent, and vetiver is voiced through the EO",
    "Patchoulol": "patchouli GC constituent and the oil's own note -- sold as the EO, not as a bottle of the alcohol",
    "Norpatchoulenol": "patchouli GC constituent, sold only as a reference standard",
    "Pogostol": "patchouli GC constituent, sold only as a reference standard",
    "Seychellene": "patchouli GC hydrocarbon, sold only as a reference standard",
    "Patchoulene Beta": "patchouli GC hydrocarbon, sold only as a reference standard",
    "Guaiene Alpha": "patchouli/guaiac GC hydrocarbon, a reference standard rather than a stock item",
    "Bulnesene Alpha": "patchouli/guaiac GC hydrocarbon, a reference standard rather than a stock item",
    "Cuparene": "patchouli/cedar GC hydrocarbon, a reference standard rather than a stock item",
    "Cedrene Beta": "cedarwood GC hydrocarbon; beta-cedrene is a reference standard, cedarwood is voiced through the EO",
    "Thujopsene": "cedarwood GC hydrocarbon, sold only as a reference standard",
    "Widdrol": "cedarwood GC sesquiterpene alcohol, sold only as a reference standard",
    "Elemol": "elemi GC sesquiterpene alcohol, sold only as a reference standard",
    "Ethyl Tiglate": "trace fermentation/fruit ester, not stocked by any supplier on this perfumer's list",
    # The perfumer's own do-not-use list, moved here so ONE list decides what
    # exists on every surface. It used to live only in
    # perfumer/tools/ifra_limits.csv, which reached the simplifier through
    # build_db.py but never the atlas -- so prohibited and unbuyable materials
    # stayed visible on the published map. Curation belongs to the dataset: a
    # material nobody can buy is not a point on a map of what can be bought.
    #
    # The reasons keep the same bar as the rest of this dict (a regulatory
    # prohibition, or nobody selling it as a perfumery material), which is why
    # they read alike. They are shown to a user whose formula line is refused,
    # so they are written to be true rather than tidy.
    "Lyral": "IFRA prohibited (HICC) from the 49th amendment and banned in EU cosmetics, never a formulating material",
    "l-Carvone": "not stocked as a perfumery material by any supplier on this perfumer's list",
    "Evernic Acid": "oakmoss lichen acid, not sold as a formulating material",
    "Atranorin": "oakmoss lichen acid, not sold as a formulating material",
    "Chloroatranorin": "oakmoss lichen acid, not sold as a formulating material",
    "Triethyl Citrate": "carrier and solvent, odourless, so it has no odour to contribute",
    "Butylated Hydroxytoluene (BHT)": "antioxidant preservative, not a perfumery material",
    "Avobenzone": "sunscreen UV filter, not a perfumery material",
    "Octinoxate": "sunscreen UV filter, not a perfumery material",
    "3,3-Dimethyl-1-Indanone": "no supplier carries it as a fragrance material",
    "Alpha Santalene": "sandalwood sesquiterpene hydrocarbon, not sold as a formulating material",
    "Alpha Santalol": "sandalwood constituent; the pure isomer is a research reference, and the sold products are the oil or an alpha/beta blend",
    "Castoramine": "castoreum alkaloid, exists only as a reference standard, not a formulating material",
    "Ledol": "labrador-tea sesquiterpene, not sold as a formulating material",
    "Nerolidol": "not purchasable as a formulating material",
    "(E,E)-2,4-Decadienal": "not purchasable as a formulating material",
    "2,6-Dimethylpyrazine": "not purchasable as a formulating material",
    "2-Ethyl-3,5-Dimethylpyrazine": "not purchasable as a formulating material",
    "2-Methyl Pyrazine": "not purchasable as a formulating material",
    "2-Octen-4-One": "not purchasable as a formulating material",
    "3-Octen-2-One": "not purchasable as a formulating material",
    "Dimethyl Disulfide": "not purchasable as a formulating material",
    "Dimethyl Trisulfide": "not purchasable as a formulating material",
    "Isopropyl 2-Methylbutyrate": "not purchasable as a formulating material",
    # Catalogue additives the perfumer engine's brief search reached once it
    # searched the whole library on the brief's own objective (2026-11-08):
    # their predicted percept rows happened to fit a sulfurous brief.
    "Oxynex® ST": "antioxidant and photostabiliser, not a perfumery material",
    "Carbitol™": "solvent (ethoxydiglycol), not a perfumery material",
    "Squalane": "odourless emollient, not a perfumery material",
    "Diethyl Hexyl Adipate": "plasticiser and emollient, odourless, not a perfumery material",
    "Chloroxylenol": "antiseptic (PCMX), not a perfumery material",
}

#: Files whose entries were added because a real formula asked for them by
#: name. A material someone put in a formula is evidence of availability in
#: the same way an inventory row is.
FORMULA_SOURCED = ("add_supplier_names.csv", "add_coffee_roast.csv")

#: Provenance strings that mark a library entry as a structure added so an
#: essential oil could be built from its constituents, rather than a material
#: anyone formulates with. Thujopsene, khusimol, the santalols and the rest
#: came from PubChem for that purpose.
#:
#: "GS-LF" is deliberately NOT in this list, and that mistake cost a day. The
#: source column records where a *structure* came from, not where the material
#: came from, and most of the GS-LF-sourced entries are a supplier's catalogue
#: with their SMILES looked up in the GS-LF dataset. Treating that column as
#: material provenance labelled 155 materials "unverified as a perfumery
#: material" -- Aldehyde C-7, Allyl Hexanoate, Acetophenone, Pyralone,
#: prenyl acetate -- and withheld or warned about every one of them.
REFERENCE_SOURCES = ("constituent reference", "gc constituent",
                     "natural constituent")

#: A supplier's own catalogue, already in this repository: 371 product names
#: from a house that sells to hobbyists and small studios. This is the
#: commonality signal that does not depend on one person's shelf or on which
#: formulas happen to be checked in here -- somebody sells it.
SUPPLIER_CATALOGUES = ("fraterworks_raw.txt", "fraterworks_resolved.csv")


def supplier_catalogue():
    """Normalised names a supplier lists as products."""
    import re as _re

    def norm(x):
        return " ".join(_re.sub(r"[^a-z0-9]+", " ", x.lower()).split())

    out = set()
    for fn in SUPPLIER_CATALOGUES:
        path = ROOT / "library" / "input" / fn
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            for r in csv.DictReader(io.open(path, encoding="utf-8")):
                n = (r.get("name") or "").strip()
                if n:
                    out.add(norm(n))
        else:
            for line in io.open(path, encoding="utf-8"):
                t = line.strip()
                if t:
                    out.add(norm(t))
    return out


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


def inventory(resolve=lambda n: n):
    """name -> compiled threshold ppb, or None.

    Keyed by the library name the shelf name refers to. `resolve` is
    `parse_formulas.library_name` bound to this page's own material and oil
    names: without it the join was exact string equality, and the perfumer's
    shelf -- which says "Orange Sweet EO", "Oakmoss IFRA43", "Patchouli
    Dark" -- matched 2 of its 47 oils. Every other oil lost its compiled threshold in
    silence. The sheet's top % is not read: inventory.csv no longer carries it.
    """
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

        out[resolve(n)] = num("water_threshold_ppb")
    return out


#: Fitted to the inventory's own 159 tier assignments by grid search, not
#: chosen here, and reproducing 74% of them. Shared with the atlas, where the
#: same numbers derive the same tiers; a tier recorded by hand always wins.
BP_TOP, BP_BASE = 216.0, 276.0


def volatility(resolve=lambda n: n):
    """name -> (tier, 'recorded' | 'derived'), collapsed to three.

    The simplifier matches a single percept vector, and the README records
    what that costs: a tomato-leaf accord of five green materials read
    `cedar, amber, dry` because it is 44% Iso E Super. One vector describes a
    base-heavy formula's drydown, so a proposal can match on paper and have a
    completely different arc. Tiers are carried here so the page can say when
    that has happened.
    """
    from volatility_profile import GROUP, oil_classes, volatility_classes

    rec, bps = {}, {}
    ip = ROOT / "data" / "inventory" / "inventory.csv"
    if ip.is_file():
        for r in csv.DictReader(io.open(ip, encoding="utf-8-sig")):
            g = GROUP.get((r.get("volatility_class") or "").strip())
            if g:
                rec[resolve((r.get("name") or "").strip())] = g
    bp = ROOT / "data" / "inventory" / "boiling_points.csv"
    if bp.is_file():
        for r in csv.DictReader(io.open(bp, encoding="utf-8-sig")):
            try:
                bps[resolve((r.get("name") or "").strip())] = float(r["bp_c"])
            except (ValueError, KeyError):
                pass
    out = {}
    for n, g in rec.items():
        out[n] = (g, "recorded")
    for n, v in bps.items():
        if n in out:
            continue
        out[n] = (("top" if v < BP_TOP else
                   "middle" if v < BP_BASE else "base"), "derived")
    # Oils inherit a tier from their constituents weighted by GC share, where
    # enough of the oil is accounted for. An oil has no boiling point -- a
    # distillate boils over a range -- so this is the only route, and the
    # coverage floor is what stops it inventing one: vetiver was coming out
    # Mid on 1% of its weight.
    oil_t, oil_cov = oil_classes(volatility_classes(), with_cover=True)
    for n, t in oil_t.items():
        out.setdefault(n, (GROUP[t], "inherited"))
    return out, oil_cov


def main() -> int:
    ap = argparse.ArgumentParser()
    # THE SWITCH. Which percept artifact the whole downstream stack uses:
    # this export -> artifacts/simplify_data.json -> perfumer materials.db.
    # fp4 (retrained with pos_weight) is current; revert to fp3 by passing
    # --percepts artifacts/percepts_fp3.json, or point at fp5 when it exists.
    ap.add_argument("--percepts", default="artifacts/percepts_fp4.json")
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
    # The just-noticeable scale everything is reported against. One definition,
    # in scripts/yardstick.py, so no caller carries a stale literal.
    import yardstick as ys
    yardstick = ys.of_distances(D)

    # One resolver for every sheet-vs-library join on this page, so the use
    # level, the recorded tier and the compiled threshold all agree about which
    # material a shelf name means -- and about which one a pasted formula
    # means, since library_name is what the parser uses.
    lib_set = set(names)

    def resolve(n):
        return library_name(n, lib_set, oil_names)

    inv = inventory(resolve)
    prov = provenance()
    used = formula_usage()
    from_formula = formula_sourced()
    catalogue = supplier_catalogue()

    import re as _re

    def _norm(x):
        return " ".join(_re.sub(r"[^a-z0-9]+", " ", x.lower()).split())
    strength = load_strength()
    vol, oil_cov = volatility(resolve)
    # IFRA: one resolved row per material, from the official standards
    # workbook (scripts/import_ifra.py -> scripts/resolve_ifra.py). The page
    # had no IFRA column at all until 2026-11, which meant the simplifier could
    # offer a material at a dose the regulator forbids and say nothing about
    # it. `status` is what keeps "no restriction" apart from "not identified".
    ifra = {}
    ifra_path = ROOT / "data" / "ifra" / "materials.csv"
    if ifra_path.is_file():
        for r in csv.DictReader(io.open(ifra_path, encoding="utf-8-sig")):
            ifra[(r.get("name") or "").strip()] = r
    else:
        print(f"  WARNING: {ifra_path} missing; the page will carry no IFRA "
              f"limits (run scripts/import_ifra.py and scripts/resolve_ifra.py)")
    n_blocked = 0
    n_unverified = 0
    mats = []
    n_measured = 0
    for i, n in enumerate(names):
        thr = inv.get(n)
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
            # How widely a material is used is NOT recorded here, and three
            # attempts to infer it all failed in the same way. An inventory
            # is one shelf. The formulas checked in here are a handful. A
            # supplier's catalogue is one house's range: built on it, camphor,
            # borneol, thymol, evernyl, bacdanol and methyl jasmonate all came
            # out "rarely used". Each attempt mislabelled obviously common
            # materials, so no commonality claim is published. What is
            # published is where the entry came from, which is a fact.
            "nf": used.get(n) or None,
            "cat": bool(_norm(n) in catalogue) or None,
            "p": [round(float(v), 3) for v in X[i]],
            "iso": round(float(iso_pct[i]), 1),
            # IFRA category 4 (fine fragrance), % of the finished product, and
            # the status that says whether a missing number means "unlimited"
            # or "unknown". Rendered as "IFRA 4.7%" / "no restriction" /
            # "IFRA unknown" -- three different claims.
            # `not in (None, "")` rather than truthiness: a cap of exactly 0
            # is a prohibition and is the one value that must never be read as
            # "no cap".
            "ifra": (float(ifra[n]["cat4_pct"])
                     if n in ifra
                     and ifra[n].get("cat4_pct") not in (None, "") else None),
            "ifraStat": (ifra[n].get("status") or None) if n in ifra else None,
            "ifraWhy": (ifra[n].get("evidence") or None) if n in ifra else None,
            "thr": thr,
            "str": round(st["mult"], 3) if st else None,
            "strN": st["n"] if st else None,
            "own": bool(n in inv),
            "vol": vol.get(n, (None, None))[0],
            "volSrc": vol.get(n, (None, None))[1],
            # How much of an oil's GC weight the tier rests on, so the page
            # can say "inherited from 71% of the oil" instead of implying a
            # measurement.
            "volCov": (round(100 * oil_cov[n]) if n in oil_cov else None),
            "oil": bool(o),
            "built": (", ".join(
                f"{k} {100 * v:.0f}%" for k, v in
                sorted(o["composition_used"].items(), key=lambda kv: -kv[1])[:5])
                if o else None),
        })

    # Materials whose percept vectors are bit-identical. ECFP4 radius 2 plus
    # constitutional descriptors cannot separate an E/Z or stereo pair, so
    # geraniol and nerol come out as the same array -- and a substitution
    # engine scored on percept distance would then offer either as a perfect
    # swap for the other at a reported cost of zero. The page says so instead.
    # Equality is tested on the vector rather than on the fingerprint, since
    # it is identical OUTPUT that misleads, whatever produced it.
    twins = {}
    seen = {}
    for i, n in enumerate(names):
        key = X[i].tobytes()
        seen.setdefault(key, []).append(n)
    for group in seen.values():
        if len(group) > 1:
            for n in group:
                twins[n] = [o for o in group if o != n]
    if twins:
        print(f"  {len(twins)} materials share a percept vector with another: "
              + "; ".join(sorted(f"{a} = {'/'.join(b)}"
                                 for a, b in twins.items())[:3]))
    for m in mats:
        m["twin"] = twins.get(m["n"]) or None

    # Thresholds, filled and labelled. A material with no figure could not be
    # dosed and so could not be proposed at all, which is a worse answer than
    # a figure that is a factor of three out.
    # One threshold column, assembled in scripts/thresholds.py so the atlas
    # and this page cannot grow separate notions of the word. Order, tiers and
    # the reasons for both are documented there.
    tvals, tsrc, textra, treport = assemble_thr(
        [m["n"] for m in mats], X, strength=strength,
        not_anchors=[m["n"] for m in mats if m.get("oil")],
        oils=[m["n"] for m in mats if m.get("oil")])
    for m, v, srcv in zip(mats, tvals, tsrc):
        m["thr"] = round(v, 6) if v else None
        m["thrSrc"] = srcv
        ex = textra.get(m["n"])
        if ex:
            m["thrN"] = ex["n"]
            m["thrLo"] = round(ex["lo"], 6)
            m["thrHi"] = round(ex["hi"], 6)
            m["thrWide"] = ex["wide"] or None
    print(f"  KFO-Atlas supplied {treport.get('kfo', 0)} thresholds that would "
          f"otherwise have been imputed")
    print("  " + describe_thr(treport))
    if ifra:
        status_n: dict[str, int] = {}
        for m in mats:
            key = m.get("ifraStat") or "missing"
            status_n[key] = status_n.get(key, 0) + 1
        print(f"  IFRA: {sum(1 for m in mats if m.get('ifra') is not None)} "
              f"with a cat-4 cap -- " + ", ".join(
                  f"{k} {v}" for k, v in
                  sorted(status_n.items(), key=lambda kv: -kv[1])))
        oils_uncapped = [m["n"] for m in mats
                         if m.get("oil") and m.get("ifra") is None]
        print(f"  IFRA: {47 - len(oils_uncapped)}/47 oils carry a derived cap "
              f"from their restricted constituents")

    payload = {
        "descriptors": desc,
        "materials": mats,
        "aliases": {k: v for k, v in ALIAS.items()},
        "solvents": sorted(SOLVENT),
        # Which oil to take when several chemotypes of one plant tie. Without
        # it the page drops "Lavender Maillette France" for matching three
        # lavenders equally well.
        "oil_defaults": pf.oil_defaults(),
        "volMeta": {
            "recorded": sum(1 for v in vol.values() if v[1] == "recorded"),
            "derived": sum(1 for v in vol.values() if v[1] == "derived"),
            "inherited": sum(1 for v in vol.values() if v[1] == "inherited"),
            "agreement": 0.74, "bpTop": BP_TOP, "bpBase": BP_BASE,
        },
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
            "n_catalogue": sum(1 for m in mats if m.get("cat")),
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
    print(f"  {sum(1 for m in mats if m.get('cat'))} appear in a supplier "
          f"catalogue in this repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
