"""Resolve perfumery trade names to validated SMILES using the GS-LF table.

GS-LF ships 5862 molecules with names and isomeric SMILES, so it is a far
safer source than writing SMILES by hand -- the 166-material list done that way
needed 36 corrections. Anything GS-LF cannot resolve is reported rather than
guessed.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.config import resolve_paths  # noqa: E402

#: Trade names whose GS-LF equivalent is a systematic name. Only entries that
#: are unambiguous in the trade are listed; anything doubtful is left to fail.
SYNONYMS = {
    "aldehyde c-7": "heptanal", "aldehyde c-8": "octanal", "aldehyde c-9": "nonanal",
    "aldehyde c-10": "decanal", "aldehyde c-11 undecylic": "undecanal",
    "aldehyde c-11 undecylenic": "10-undecenal",
    "aldehyde c-12 lauric": "dodecanal", "aldehyde c-12 mna": "2-methylundecanal",
    "alcohol c-6": "hexanol", "alcohol c-7": "heptanol", "alcohol c-8": "octanol",
    "alcohol c-9": "nonanol", "alcohol c-10": "decanol",
    "alcohol c-11 undecylenic": "10-undecen-1-ol", "alcohol c-12 lauric": "dodecanol",
    "acetate c-9": "nonyl acetate",
    "amyl acetate": "isoamyl acetate",
    "anisic aldehyde": "anisaldehyde",
    "cis-3-hexenol": "cis-3-hexen-1-ol",
    "cis 3 hexenyl acetate": "cis-3-hexenyl acetate",
    "citronellol 98": "citronellol", "geraniol extra": "geraniol",
    "camphor": "camphor", "laevo menthol": "menthol",
    "eucalyptol": "eucalyptol", "cis jasmone": "cis-jasmone",
    "d-limonene": "limonene", "beta pinene": "beta-pinene",
    "alpha pinene": "alpha-pinene", "caryophyllene beta": "beta-caryophyllene",
    "cedrene alpha": "alpha-cedrene", "farnesene alpha": "alpha-farnesene",
    "fenchone alpha": "fenchone", "terpinene gamma": "gamma-terpinene",
    "terpineol alpha": "alpha-terpineol", "ionone alpha": "alpha-ionone",
    "ionone beta": "beta-ionone", "damascone alpha": "alpha-damascone",
    "damascone beta": "beta-damascone", "damascone delta": "delta-damascone",
    "damascone gamma": "gamma-damascone", "damascenone": "beta-damascenone",
    "dihydro alpha ionone": "dihydro-alpha-ionone",
    "dihydro beta ionone": "dihydro-beta-ionone",
    "methyl ionone alpha extra": "alpha-methylionone",
    "methyl ionone gamma coeur": "gamma-methylionone",
    "delta decalactone": "delta-decalactone", "delta dodecalactone": "delta-dodecalactone",
    "delta nonalactone": "delta-nonalactone", "delta octalactone": "delta-octalactone",
    "delta undecalactone": "delta-undecalactone",
    "gamma decalactone": "gamma-decalactone", "gamma dodecalactone": "gamma-dodecalactone",
    "gamma heptalactone": "gamma-heptalactone", "gamma hexalactone": "gamma-hexalactone",
    "gamma nonalactone": "gamma-nonalactone", "gamma octalactone": "gamma-octalactone",
    "gamma undecalactone": "gamma-undecalactone",
    "alpha angelica lactone": "alpha-angelica lactone",
    "phenyl ethyl alcohol": "phenethyl alcohol",
    "phenyl ethyl acetate": "phenethyl acetate",
    "phenyl acetaldehyde": "phenylacetaldehyde",
    "phenyl acetic acid": "phenylacetic acid",
    "para cresol": "p-cresol", "para cresyl acetate": "p-cresyl acetate",
    "para cresyl methyl ether": "p-cresyl methyl ether",
    "para methyl acetophenone": "4'-methylacetophenone",
    "para tolyl aldehyde": "p-tolualdehyde",
    "heliotropin": "piperonal", "anethole": "anethole",
    "methyl heptenone": "6-methyl-5-hepten-2-one",
    "methyl nonyl ketone": "2-undecanone",
    "hydratropic aldehyde": "2-phenylpropanal",
    "benzyl phenylacetate": "benzyl phenyl acetate",
    "strawberry furanone": "furaneol",
    "ethyl furaneol": "homofuraneol",
    "ethyl caproate": "ethyl hexanoate",
    "diacetyl": "2,3-butanedione",
    "acetoin": "acetoin", "maltol": "maltol", "ethyl maltol": "ethyl maltol",
    "guaiacol": "guaiacol", "creosol": "creosol",
    "4-ethyl guaiacol": "4-ethylguaiacol",
    "2-isopropyl phenol": "2-isopropylphenol",
    "2-acetyl pyridine": "2-acetylpyridine",
    "2-acetyl thiazole": "2-acetylthiazole",
    "tetramethyl pyrazine": "tetramethylpyrazine",
    "methyl chavicol": "estragole",
    "caprylic acid": "octanoic acid", "myristic acid": "tetradecanoic acid",
    "palmitic acid": "hexadecanoic acid",
    "matsutake alcohol": "1-octen-3-ol",
    "methyl heptine carbonate": "methyl 2-octynoate",
    "methyl octine carbonate": "methyl 2-nonynoate",
    "raspberry ketone": "raspberry ketone",
    "trans-2-hexenol": "trans-2-hexen-1-ol",
    "trans-2-nonenal": "trans-2-nonenal",
    "trans-2-octenal": "trans-2-octenal",
    "trans-2-decenal": "trans-2-decenal",
    "trans-4-decenal": "trans-4-decenal",
    "trans-2,cis-6-nonadienal": "trans-2-cis-6-nonadienal",
    "trans-2,cis-6-nonadienol": "trans-2-cis-6-nonadien-1-ol",
    "cuminaldehyde": "cuminaldehyde",
    "isobutyl quinoline": "6-isobutylquinoline",
    "2-isobutyl quinoline": "6-isobutylquinoline",
    "6-methyl quinoline": "6-methylquinoline",
    "veratraldehyde": "veratraldehyde",
    "vanillin acetate": "vanillyl acetate",
    "nootkatone": "nootkatone", "valencene": "valencene",
    "geosmin": "geosmin", "safranal": "safranal",
    "zingerone": "zingerone", "menthone": "menthone",
    "piperitone": "piperitone", "fenchol": "fenchol", "nerol": "nerol",
    "nerolidol": "nerolidol", "farnesol": "farnesol", "myrcene": "myrcene",
    "ocimene": "beta-ocimene", "indole": "indole", "coumarin": "coumarin",
    "eugenol": "eugenol", "isoeugenol": "isoeugenol",
    "dihydroeugenol": "dihydroeugenol", "carvacrol": "carvacrol",
    "benzophenone": "benzophenone", "acetophenone": "acetophenone",
    "thymol": "thymol", "citral": "citral", "citronellal": "citronellal",
    "linalool": "linalool", "linalool oxide": "linalool oxide",
    "benzyl alcohol": "benzyl alcohol", "benzyl benzoate": "benzyl benzoate",
}

NOISE = re.compile(
    r"\s*\(.*?\)|\s*\d+(\.\d+)?\s*%.*$|,?\s*(natural|crystals?|liquid|total|extra|"
    r"signature|rectified|coeur|neo|super|hc|dm|mip|ex\s+.*|98|95|50|70)\s*$",
    re.I,
)


def normalise(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("’", "'")
    prev = None
    while prev != s:
        prev = s
        s = NOISE.sub("", s).strip(" ,-")
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> int:
    paths = resolve_paths()
    import pandas as pd

    df = pd.read_csv(paths.gs_lf_csv)
    lookup: dict[str, str] = {}
    for nm, smi in zip(df["name"], df["IsomericSMILES"]):
        if isinstance(nm, str):
            lookup.setdefault(normalise(nm), smi)
    # also index IUPAC names, which catch many systematic trade names
    if "IUPACName" in df.columns:
        for nm, smi in zip(df["IUPACName"], df["IsomericSMILES"]):
            if isinstance(nm, str):
                lookup.setdefault(normalise(nm), smi)
    print(f"GS-LF lookup: {len(lookup)} distinct names")

    raw = [l.strip() for l in open(ROOT / "library/input/fraterworks_raw.txt",
                                  encoding="utf-8") if l.strip()]
    hits, misses = [], []
    for name in raw:
        key = normalise(name)
        smi = lookup.get(key)
        if smi is None and key in SYNONYMS:
            smi = lookup.get(normalise(SYNONYMS[key]))
        if smi is None:
            # last resort: drop leading/trailing descriptors and retry
            alt = re.sub(r"^(alpha|beta|gamma|delta|cis|trans|iso)\s+", r"\1-", key)
            smi = lookup.get(alt)
        if smi:
            hits.append((name, smi))
        else:
            misses.append(name)

    out = ROOT / "library/input/fraterworks_resolved.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["name", "smiles", "source"])
        for n, s in hits:
            w.writerow([n, s, "GS-LF"])
    print(f"resolved {len(hits)}/{len(raw)} ({100*len(hits)/len(raw):.0f}%) -> {out}")
    print(f"unresolved {len(misses)}")
    for m in misses[:60]:
        print(f"   {m}")
    if len(misses) > 60:
        print(f"   ... and {len(misses)-60} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
