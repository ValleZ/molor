"""Parse the perfumer's formula records into resolvable ingredient lists.

The records are prose, not data. Amounts appear as grams, milligrams, parts and
percentages; some lines are table rows and some are bare "Material 350"; many
ingredients name a stock dilution whose strength has to come from the inventory
("Blackcurrant Mercaptan 0.1% 30" is 30 parts of a 0.1% solution, so 0.03 parts
of material). Names are the perfumer's own shorthand -- DMBCA, PEA, cis-3-hexenyl
acetate for what the library calls Hexenyl-3-CIS Acetate.

So this resolves what it can and reports what it cannot, rather than guessing.
Anything unresolved is named in the output, because a formula that leans on an
unresolvable ingredient cannot be predicted and must be excluded from any later
analysis rather than silently scored on its remainder.

Three kinds of ingredient are never resolvable to a single structure, and are
labelled rather than dropped quietly:

  solvent    DPG, benzyl benzoate as filler, perfumer's alcohol -- no odour
             contribution to model, so these are excluded from the normalised
             composition but do count toward the total
  natural    Orange EO, Bergamot BF, Ylang Ylang Complete -- mixtures
  base       Cassis Base 345B, Dewberry FO, an accord used as an ingredient

Each formula keeps the verdict the perfumer wrote underneath it, which is the
only reason the records are worth parsing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clean_library import clean_name  # noqa: E402

DOC = ROOT / "data" / "formulas" / "recipes.md"

#: The perfumer's shorthand -> the library's name.
ALIAS = {
    "dmbca": "Dimethyl Benzyl Carbinyl Acetate",
    "pea": "Phenethyl Alcohol",
    "phenylethyl alcohol": "Phenethyl Alcohol",
    "cis-3-hexenyl acetate": "Hexenyl-3-CIS Acetate",
    "hexenyl-3-cis acetate": "Hexenyl-3-CIS Acetate",
    "cis-3-hexenol": "Hexenol-3-CIS",
    "cis-3-hexenyl salicylate": "Cis-3-Hexenyl Salicylate",
    "ethyl 2-methylbutyrate": "Ethyl 2-methyl butyrate",
    "blackcurrant mercaptan": "Blackcurrant mercaptan",
    "calone": "Calone 1951",
    # Matched to the library by InChIKey from the CAS number on the stock, not
    # by name: the library spells these differently. "heliotropine" and
    # "terpineol" are already below; terpineol is 8000-41-7, the isomer
    # mixture, against the library's alpha isomer alone -- an approximation
    # rather than an identity.
    "anisic aldehyde": "Anisaldehyde",         # 123-11-5, p-anisaldehyde
    "dihydro ionone beta": "Dihydro Beta Ionone",   # 17283-81-7
    "nonadienal": "Trans-2,Cis-6-Nonadienal",
    "pharone": "Pharaone",                     # 313973-37-4, as the sheet spells it
    "padma": "Viridine",                       # 101-48-4, the acronym for it
    "phenyl acetaldehyde dimethyl acetal": "Viridine",
    "l-menthol": "Laevo Menthol",
    "menthol": "Laevo Menthol",
    "l-carvone": "l-Carvone",
    # Trade and systematic names used by the patent-era published formulas.
    # Only where it is the same molecule: bornyl acetate is NOT Isobornyl
    # Acetate, acetyl cedrene is NOT Cedrene Alpha and methyl eugenol is NOT
    # Methyl Isoeugenol, so those stay unmapped and are dropped instead.
    "methyl dihydrojasmonate": "Hedione",
    "alpha-ionone": "Ionone Alpha",
    "beta-ionone": "Ionone Beta",
    "isoraldeine": "Methyl Ionone Gamma Coeur",
    "alpha-isomethyl ionone": "Methyl Ionone Gamma Coeur",
    "alpha-hexyl cinnamaldehyde": "Hexyl Cinnamic Aldehyde",
    "geraniol": "Geraniol 95",
    "rhodinol": "Rhodinol Ex Geranium Bourbon",
    "indole": "Indolene 50",
    "isobutyl quinoline": "2-Isobutyl Quinoline",
    "eugenol": "Eugenol ex Clove",
    "alpha-terpineol": "Terpineol Alpha",
    "musk xylol": "Musk Xylene",
    "dihydromyrcenol": "Dihydro Myrcenol",
    "dihydro myrcenol": "Dihydro Myrcenol",
    "dihydromircenol": "Dihydro Myrcenol",
    "iso e super": "Iso E Super",
    "ionone alpha": "Ionone Alpha",
    "ionone beta": "Ionone Beta",
    "methyl ionone gamma": "Methyl Ionone Gamma Coeur",
    "terpineol": "Terpineol Alpha",
    "cinnamic alcohol": "Cinnamyl Alcohol",
    "cinnamyl alcohol": "Cinnamyl Alcohol",
    "heliotropin repl x24": "Heliotropin",
    "heliotropine": "Heliotropin",
    "phenylacetaldehyde": "Phenyl Acetaldehyde",
    "phenyl acetaldehyde": "Phenyl Acetaldehyde",
    "isoeugenol replacer": "Isoeugenol",
    "gamma decalactone": "Gamma Decalactone",
    "gamma -decalactone": "Gamma Decalactone",
    "delta decalactone": "Delta Decalactone",
    "gamma nonalactone": "Gamma Nonalactone",
    "gamma undecalactone": "Gamma Undecalactone",
    "galaxolide": "Galaxolide",
    "orange crystals": "Oranger Crystals",
    "oranger crystals": "Oranger Crystals",
    "benzaldehyde": "Benzaldehyde",
    "vanillin": "Vanillin",
    "ethyl vanillin": "Ethyl Vanillin",
    "ethyl maltol": "Ethyl Maltol",
    "ethyl furaneol": "Ethyl Furaneol",
    "strawberry furanone": "Strawberry Furanone",
    "damascenone": "Damascenone Total",
    "damascenone total": "Damascenone Total",
    "damascone beta": "Damascone Beta",
    "damascone alpha": "Damascone Alpha",
    "melonal": "Melonal",
    "methyl laitone": "Methyl Laitone",
    "coumarin": "Coumarin",
    "ambroxan": "Ambroxan",
    "cedarwood virginia": "Cedarwood Virginia",
    "hexyl cinnamaldehyde": "Hexyl Cinnamic Aldehyde",
    "hexyl cinnamal": "Hexyl Cinnamic Aldehyde",
    "aldehyde c-8": "Aldehyde C-8",
    "aldehyde c-10": "Aldehyde C-10",
    "aldehyde c-12": "Aldehyde C-12 MNA",
    "citral": "Citral",
    "rose oxide": "Rose Oxide",
    "para cresyl acetate": "Para Cresyl Acetate",
    "geraniol": "Geraniol 95",
    "linalyl acetate": "Linalyl Acetate",
    "tetrahydro linalool": "Tetrahydro Linalool",
    "anisaldehyde": "Anisaldehyde",
    "isoamyl acetate": "Isoamyl Acetate",
    "hexyl acetate": "Hexyl Acetate",
    "ethyl butyrate": "Ethyl Butyrate",
    "methyl benzoate": "Methyl Benzoate",
    "styrallyl acetate": "Styrallyl Acetate",
    "benzyl acetate": "Benzyl Acetate",
    "hexyl salicylate": "Hexyl Salicylate",
    "hydroxycitronellal": "Hydroxycitronellal",
    "citronellol": "Citronellol",
    "helional": "Helional",
    "hedione": "Hedione",
    "dihydromyrcenol": "Dihydro Myrcenol",
    "lemon fcf": "Lemon FCF",
    # Hedione HC is the cis-enriched grade of the same molecule, and the
    # percept model's fingerprint is stereo-blind, so it is the same input.
    # The grades do not smell the same on a blotter; nothing here can tell
    # them apart, and pretending otherwise by dropping the ingredient is worse.
    "hedione hc": "Hedione",
    # Sylvamber and Timbersilk are both the OTNE isomer family, CAS
    # 54464-57-2, which is what Iso E Super is. Sylvamber is an enriched
    # fraction of it and is perceptibly different on the bench; constitutionally
    # it is the same material, which is all the model reads.
    "sylvamber": "Iso E Super",
    "timbersilk": "Iso E Super",
    "e. brass.": "Ethylene Brassylate",
    "e brass": "Ethylene Brassylate",
    "palisandin": "Palissandin",
    "edenolide": "Edenolide",
    "velvione": "Velvione",
    "habanolide": "Habanolide",
    "ambrettolide": "Ambrettolide",
    "ethylene brassylate": "Ethylene Brassylate",
    "helvetolide": "Helvetolide",
    "sandalore": "Sandalore",
    "ebanol": "Ebanol",
    "cashmeran": "Cashmeran",
    "methyl diantilis": "Methyl Diantilis",
    "undecavertol": "Undecavertol",
    "florhydral": "Florhydral",
    "indolene 50": "Indolene 50",
    "cyclamen aldehyde": "Cyclamen Aldehyde",
    "d-limonene": "D-Limonene",
    "linalool": "Linalool",
    "ethyl linalool": "Ethyl Linalool",
    "fructone": "Fructone",
    "verdox hc": "Verdox HC",
    "cassifix": "Cassifix",
    "paradisamide": "Paradisamide",
    "manzanate": "Manzanate",
    "raspberry ketone": "Raspberry Ketone",
    "strawberry glycidate": "Strawberry Glycidate",
    "phenethyl acetate": "Phenethyl Acetate",
    "methyl pamplemousse": "Methyl Pamplemousse",
    "lemonile": "Lemonile",
    "peonile": "Peonile",
    "isobornyl acetate": "Isobornyl Acetate",
    "thiocineole": "Thiocineole",
    "liffarome": "Liffarome",
    "benzyl alcohol": "Benzyl Alcohol",
    "benzyl salicylate": "Benzyl Salicylate",
    "toscanol": "Toscanol",
    "nympheal": "Nympheal",
    'verdox': 'Verdox HC',
    'limonene': 'D-Limonene',
    'cinnamal': 'Cinnamaldehyde',
    'cinnamaldehyde': 'Cinnamaldehyde',
    'ethyl hexanoate': 'Ethyl Caproate',
    'allyl heptanoate': 'Allyl Heptanoate',
    'hexyl cinnamal': 'Hexyl Cinnamic Aldehyde',
    'patchouli alcohol': 'Patchouli Dark',
    'terpineol alpha': 'Terpineol Alpha',
    'heliotropine': 'Heliotropin',
    'methyl latione': 'Methyl Laitone',
    'oranger': 'Oranger Crystals',
    'orangers': 'Oranger Crystals',
    'pinene': 'Alpha Pinene',
    'alpha-pinene': 'Alpha Pinene',
    'beta-pinene': 'Beta Pinene',
    'eugenol': 'Eugenol ex Clove',
    'isoeugenol': 'Isoeugenol',
    'linalyl acetate': 'Linalyl Acetate',
    'citronellol': 'Citronellol',
    'geranyl acetate': 'Geranyl Acetate',
    'nerol': 'Nerol',
    'eucalyptol': 'Eucalyptol',
    'beta-caryophyllene': 'Caryophyllene Beta',
    'caryophyllene': 'Caryophyllene Beta',
    'triplal': 'Triplal',
    'myrcene': 'Myrcene',
    'safrole': 'Safrole',
    'coumarin': 'Coumarin',
    'hexyl salicylate': 'Hexyl Salicylate',
    'otne': 'Iso E Super',
    'ethyl linalool': 'Ethyl Linalool',
    'dihydromyrcenol': 'Dihydro Myrcenol',
}

#: Lines that are arithmetic or bookkeeping, not ingredients. A "Total 1000"
#: row parsed as an ingredient adds 1000 parts of an unknown and makes an
#: otherwise fully resolved formula look 70% unresolvable, which is what it did
#: to four of the gooseberries.
NOT_INGREDIENT = re.compile(
    r"^(total|totals|sum|q\.?s\.?|balance|per\s+1\s*g.*|material\b.*|"
    r"materials\b.*|component\b.*|components\b.*|units?|parts?|mg|µg|"
    r"%\s*of\s+compound|"
    r"normalized\s*%|intended\s+role|role|note|notes|status|job|why|ov|"
    r"ifra|trial(?:,\s*mg)?)$", re.I)

SOLVENT = {"dpg", "dipg", "dipropylene glycol", "ethanol", "etoh",
           "benzyl benzoate", "perfumer's alcohol", "witch hazel",
           "glycerin", "peg-7", "allantoin", "bisabolol", "ipm",
           "isopropyl myristate", "diethyl phthalate", "dep",
           "triethyl citrate", "tec", "water"}
NATURAL_HINT = re.compile(
    r"\b(eo|oil|absolute|resinoid|complete|terpenes|bf|fcf|ylang|petitgrain|"
    r"patchouli|vetiver java|cananga|oud|tangerine|guaiacwood)\b", re.I)
BASE_HINT = re.compile(r"\b(base|fo\b|accord|fragrance oil|ripple|345b|repl)\b",
                       re.I)

#: An amount with an optional stock strength in front of it.
AMOUNT = re.compile(
    r"^(?P<name>.*?)[\s—\-–:|]*"
    r"(?:(?P<strength>\d+(?:\.\d+)?)\s*%\s*(?:in\s+)?(?:DPG|PG|IPM|TEC|"
    r"benzyl\s+alcohol|alcohol)?\s*[,]?\s*)?"
    r"(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g|mg|%|parts?|p|µg|units?)?\s*"
    r"(?:\(.*?\))?\s*$", re.I)


#: The document writes lactones and ionones with Greek letters; the library
#: spells them out.
GREEK = {"γ": "gamma ", "δ": "delta ",
         "α": "alpha ", "β": "beta "}


def norm(name: str) -> str:
    n = name.strip().strip("*|").strip()
    for g, word in GREEK.items():
        n = n.replace(g + "-", word).replace(g, word)
    # Leading query marks are the perfumer's own doubt about an ingredient, and
    # stray escapes and tildes come from the markdown export.
    n = re.sub(r"^[?\s]+", "", n)
    n = n.replace("\\~", "").replace("~", "")
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"\s*\((?:[^()]*)\)\s*$", "", n)       # trailing parenthetical
    n = re.sub(r"[,—\-–]\s*$", "", n).strip()
    return n


def classify(name: str, library: set[str]):
    """Return (kind, resolved_name). kind is material|solvent|natural|base|unknown."""
    raw = norm(name)
    # strip trailing qualifiers that describe the stock, not the material
    raw = re.sub(r",?\s*(undiluted|neat|crystals?|natural|total)\s*$", "", raw,
                 flags=re.I).strip() or raw
    low = raw.lower()
    if NOT_INGREDIENT.match(low):
        return "skip", raw
    # strip a stock strength out of the name itself
    low = re.sub(r"\b\d+(?:\.\d+)?\s*%\s*(?:in\s+)?"
                 r"(?:dpg|pg|ipm|tec|benzyl alcohol|alcohol)?\b", "", low).strip()
    low = re.sub(r"\s+", " ", low).strip(" ,-—")
    if not low:
        return "unknown", raw
    if low in SOLVENT:
        return "solvent", raw
    if low in ALIAS:
        return "material", ALIAS[low]
    cleaned = clean_name(raw)
    if cleaned in library:
        return "material", cleaned
    for cand in (low.title(), low.capitalize()):
        if cand in library:
            return "material", cand
    hit = [m for m in library if m.lower() == low]
    if hit:
        return "material", hit[0]
    if BASE_HINT.search(raw):
        return "base", raw
    if NATURAL_HINT.search(raw):
        return "natural", raw
    return "unknown", raw


def parse(text: str, library: set[str]):
    formulas = []
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            title = re.sub(r"^#+\s*", "", s).strip().rstrip(":")
            title = re.sub(r"\\", "", title)
            if cur:
                formulas.append(cur)
            cur = {"name": title, "lines": [], "verdict": []}
            continue
        if cur is None or not s:
            continue
        if set(s) <= set("|-: "):
            continue
        body = s.strip("|").strip()
        body = re.sub(r"\*\*", "", body)
        if "|" in body:
            cells = [c.strip() for c in body.split("|") if c.strip()]
            if len(cells) >= 2 and re.search(r"\d", cells[1]):
                body = cells[0] + " " + cells[1]
            else:
                continue
        # "Isoamyl Acetate 1-1.5" is a range the perfumer bracketed; take the
        # low end rather than letting the high end be read as the amount and
        # the low end as part of the name.
        body = re.sub(r"(\d+(?:\.\d+)?)\s*[–—-]\s*\d+(?:\.\d+)?\s*$",
                      r"\1", body)
        # A weight-first line, as in the Clinique Happy transcription.
        lead = re.match(r"^(\d+(?:\.\d+)?)\s*(?:g|mg|parts?|p)?\s+"
                        r"(?=[A-Za-z])(.+?)(?:\s+[A-Z]-\d[\d.]*)?$", body)
        if lead and not re.search(r"\d\s*(?:g|mg|%|parts?)?\s*$", body):
            cur["lines"].append((lead.group(2), None,
                                 float(lead.group(1)), None))
            continue
        m = AMOUNT.match(body)
        if m and norm(m.group("name")):
            cur["lines"].append((m.group("name"), m.group("strength"),
                                 float(m.group("amount")), m.group("unit")))
        else:
            cur["verdict"].append(s)
    if cur:
        formulas.append(cur)

    out = []
    for f in formulas:
        if not f["lines"]:
            continue
        rows, issues = [], []
        for nm, strength, amt, unit in f["lines"]:
            kind, resolved = classify(nm, library)
            if kind == "skip":
                continue
            # a stock strength scales the delivered amount
            frac = (float(strength) / 100.0) if strength else 1.0
            rows.append({"raw": norm(nm), "kind": kind, "name": resolved,
                         "amount": amt, "unit": (unit or "").lower(),
                         "strength": frac, "delivered": amt * frac})
            if kind in ("unknown", "natural", "base"):
                issues.append(f"{kind}: {resolved}")
        actives = [r for r in rows if r["kind"] == "material"]
        total = sum(r["delivered"] for r in rows if r["kind"] != "solvent")
        unresolved = sum(r["delivered"] for r in rows
                         if r["kind"] in ("unknown", "natural", "base"))
        for r in actives:
            r["fraction"] = r["delivered"] / total if total else 0.0
        out.append({
            "name": f["name"],
            "verdict": " ".join(f["verdict"]).strip(),
            "ingredients": rows,
            "n_materials": len(actives),
            "unresolved_share": round(unresolved / total, 3) if total else 1.0,
            "issues": sorted(set(issues)),
        })
    return out


#: "hexyl cinnamal / Verdox / allyl heptanoate 2-5% ea" -- several materials
#: sharing one declared range, each at that range.
EACH = re.compile(r"\bea\b\.?$", re.I)
RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")
LONE = re.compile(r"<?\s*(\d+(?:\.\d+)?)\s*(%|g|mg|parts?|p)\b", re.I)
RATIO = re.compile(r"(\d+(?:\.\d+)?(?::\d+(?:\.\d+)?)+)\s*$")


def parse_mixes(path: Path, library: set[str]):
    """Parse the spreadsheet's ingredient strings, one formula per row.

    A different shape from the document: comma-separated on one line, with
    declared ranges rather than weights for the commercial oils. A range is
    taken at its midpoint, and a trailing ratio like "1:1:0.3" is applied to
    the names listed before it.

    Commercial fragrance oils are marked partial. Their published compositions
    list only the regulated and notable constituents, so a large and unstated
    share of the formula is missing, and a prediction from the declared part is
    weaker evidence than one from a formula the perfumer weighed out.
    """
    out = []
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        text = (r.get("Ingredients") or "").strip()
        verdict = (r.get("impression") or "").strip()
        if not text or not verdict or len(text) < 12:
            continue
        is_fo = (r.get("family") or "").strip().lower() == "fragrance oil"
        rows, issues = [], []

        ratio = None
        m = RATIO.search(text)
        if m:
            ratio = [float(x) for x in m.group(1).split(":")]
            text = text[:m.start()].strip().rstrip(",")

        parts = [p.strip() for p in re.split(r",(?![^()]*\))", text) if p.strip()]
        for pi, part in enumerate(parts):
            each = bool(EACH.search(part))
            part_clean = EACH.sub("", part).strip()
            rm = RANGE.search(part_clean)
            if rm:
                amount = (float(rm.group(1)) + float(rm.group(2))) / 2.0
                names_blob = part_clean[:rm.start()].strip()
            else:
                # Some records put the weight first: "300 Hexyl Cinnamaldehyde".
                lead = re.match(r"^(\d+(?:\.\d+)?)\s*(?:g|mg|parts?|p)?\s+"
                                r"(?=[A-Za-z])(.+)$", part_clean)
                lm = LONE.search(part_clean)
                if lead and not lm:
                    amount = float(lead.group(1))
                    names_blob = lead.group(2).strip()
                elif lm:
                    amount = float(lm.group(1))
                    names_blob = part_clean[:lm.start()].strip()
                elif ratio is not None and pi < len(ratio):
                    amount = ratio[pi]
                    names_blob = part_clean
                else:
                    continue
            names = ([n.strip() for n in names_blob.split("/") if n.strip()]
                     if each or "/" in names_blob else [names_blob])
            for nm in names:
                kind, resolved = classify(nm, library)
                if kind == "skip":
                    continue
                strength = None
                sm = re.search(r"(\d+(?:\.\d+)?)\s*%", nm)
                if sm and float(sm.group(1)) <= 50:
                    strength = float(sm.group(1)) / 100.0
                frac = strength if strength else 1.0
                rows.append({"raw": norm(nm), "kind": kind, "name": resolved,
                             "amount": amount, "unit": "",
                             "strength": frac, "delivered": amount * frac})
                if kind in ("unknown", "natural", "base"):
                    issues.append(f"{kind}: {resolved}")

        actives = [x for x in rows if x["kind"] == "material"]
        total = sum(x["delivered"] for x in rows if x["kind"] != "solvent")
        unres = sum(x["delivered"] for x in rows
                    if x["kind"] in ("unknown", "natural", "base"))
        for x in actives:
            x["fraction"] = x["delivered"] / total if total else 0.0
        out.append({
            "name": r["material"].strip(),
            "verdict": verdict,
            "source": "mixes.csv",
            "partial_composition": is_fo,
            "ingredients": rows,
            "n_materials": len(actives),
            "unresolved_share": round(unres / total, 3) if total else 1.0,
            "issues": sorted(set(issues)),
        })
    return out


def main() -> int:
    # Some verdicts are in Russian and this console is cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=str(DOC))
    ap.add_argument("--out", default="data/formulas/parsed.json")
    ap.add_argument("--max-unresolved", type=float, default=0.10,
                    help="a formula is usable if at most this share of its "
                         "non-solvent mass is unresolvable")
    args = ap.parse_args()

    library = {r["name"] for r in csv.DictReader(
        io.open(ROOT / "library/input/expanded_materials.csv",
                encoding="utf-8-sig"))}
    text = io.open(args.doc, encoding="utf-8").read()
    formulas = parse(text, library)
    for f in formulas:
        f.setdefault("source", "recipes.md")
        f.setdefault("partial_composition", False)
    mixes_csv = ROOT / "data" / "formulas" / "mixes.csv"
    if mixes_csv.is_file():
        mixes = parse_mixes(mixes_csv, library)
        print(f"[parse] {len(mixes)} rows from mixes.csv with an ingredient "
              f"list and an impression")
        formulas += mixes

    usable = [f for f in formulas
              if f["n_materials"] >= 3
              and f["unresolved_share"] <= args.max_unresolved]
    print(f"[parse] {len(formulas)} formulas with amounts, "
          f"{len(usable)} usable (>=3 known materials, "
          f"<={args.max_unresolved:.0%} unresolvable mass)\n")
    print(f"  {'formula':34s} {'mats':>4s} {'unres':>6s}  verdict / blockers")
    print("  " + "-" * 96)
    for f in sorted(formulas, key=lambda x: (-x["n_materials"])):
        flag = "" if f in usable else "  <-- excluded"
        note = (f["verdict"][:46] if f["verdict"]
                else "; ".join(f["issues"])[:46])
        print(f"  {f['name'][:34]:34s} {f['n_materials']:4d} "
              f"{f['unresolved_share']:6.0%}  {note}{flag}")

    unknown: dict[str, int] = {}
    for f in formulas:
        for r in f["ingredients"]:
            if r["kind"] == "unknown":
                unknown[r["name"]] = unknown.get(r["name"], 0) + 1
    if unknown:
        print(f"\n  unresolved ingredient names ({len(unknown)}), most common "
              f"first -- these need an alias or are genuinely not in the library:")
        for n, c in sorted(unknown.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {c:2d}x  {n}")

    (ROOT / args.out).write_text(json.dumps(
        {"usable": usable, "all": formulas}, indent=2), encoding="utf-8")
    print(f"\n[parse] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
