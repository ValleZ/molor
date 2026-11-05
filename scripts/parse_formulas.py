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
#: recipes.md is the perfumer's own formula book, with their written verdict on
#: each one. It is deliberately not in this repository: it was the input to the
#: mixture-rule comparison and is nobody else's business. The published
#: formulas it was compared against are in data/formulas/published/.

#: The perfumer's shorthand -> the library's name.
ALIAS = {
    # Requested by name, and most of them were already in the
    # library under another house's spelling.
    'homofuronol': 'Ethyl Furaneol',
    'homofuraneol': 'Ethyl Furaneol',
    'homofuronol plus': 'Ethyl Furaneol',
    'maltyl isobutyrate': 'Maltol Isobutyrate',
    'ethyl methyl-2-butyrate': 'Ethyl 2-methyl butyrate',
    'ethyl 2-methylbutyrate': 'Ethyl 2-methyl butyrate',
    'ethyl methyl 2 butyrate': 'Ethyl 2-methyl butyrate',
    'acetyl-2 pyridine': '2-Acetyl Pyridine',
    'acetyl 2 pyridine': '2-Acetyl Pyridine',
    '2-acetylpyridine': '2-Acetyl Pyridine',
    'dimethyl sulphide': 'Dimethyl Sulfide',
    'dimethylsulfide': 'Dimethyl Sulfide',
    'dimethylsulphide': 'Dimethyl Sulfide',
    'dimethyl disulphide': 'Dimethyl Disulfide',
    'dimethyl trisulphide': 'Dimethyl Trisulfide',
    'cosmone': 'Cosmone',
    'methyl corylone': 'Methyl Corylone',
    'methyl cyclopentenolone': 'Corylone',
    'dimethyl cyclopentenolone': 'Methyl Corylone',

    'isovaleraldehyde': '3-Methylbutanal',
    'isovaleric aldehyde': '3-Methylbutanal',
    '3-methyl butanal': '3-Methylbutanal',
    'aldehyde c-5 iso': '3-Methylbutanal',
    '2-furfurylthiol': 'Furfuryl Mercaptan',
    'furfuryl thiol': 'Furfuryl Mercaptan',
    '2-furanmethanethiol': 'Furfuryl Mercaptan',
    'beta-ionone': 'Ionone Beta',
    'beta ionone': 'Ionone Beta',
    '2,3,5,6-tetramethylpyrazine': 'Tetramethyl Pyrazine',

    # The pyrazines, whose trade names say roasted and whose
    # structures differ. A bare "dimethyl pyrazine" is deliberately
    # NOT aliased: the 2,5- and 2,6- isomers are both sold and are
    # not the same molecule.
    'torref pyrazine': '2,6-Dimethylpyrazine',
    'torrefie pyrazine': '2,6-Dimethylpyrazine',
    '2,6-dimethyl pyrazine': '2,6-Dimethylpyrazine',
    '2,6-dimethylpyrazine': '2,6-Dimethylpyrazine',
    'ligustrazine': 'Tetramethyl Pyrazine',
    '2,3,5,6-tetramethylpyrazine': 'Tetramethyl Pyrazine',
    'tetramethylpyrazine': 'Tetramethyl Pyrazine',
    '2,3,5-trimethyl pyrazine': '2,3,5-Trimethylpyrazine',
    '2,3,5-trimethylpyrazine': '2,3,5-Trimethylpyrazine',
    'trimethyl pyrazine': '2,3,5-Trimethylpyrazine',
    'trimethylpyrazine': '2,3,5-Trimethylpyrazine',

    # The roast and coffee materials, under the names a flavour
    # supplier prints rather than the ones this library uses.
    'civetone': 'Civettone',
    'cyclohexadecenone': 'Civettone',
    '2-acetylpyrazine': '2-Acetyl Pyrazine',
    'acetyl pyrazine': '2-Acetyl Pyrazine',
    '2-methylpyrazine': '2-Methyl Pyrazine',
    'methyl pyrazine': '2-Methyl Pyrazine',
    'ethyl dimethyl pyrazine': '2-Ethyl-3,5-Dimethylpyrazine',
    '2-ethyl-3,5-dimethylpyrazine': '2-Ethyl-3,5-Dimethylpyrazine',
    'ethyl dimethylpyrazine': '2-Ethyl-3,5-Dimethylpyrazine',
    'furfuryl thiol': 'Furfuryl Mercaptan',
    '2-furfurylthiol': 'Furfuryl Mercaptan',
    'furfurylmercaptan': 'Furfuryl Mercaptan',
    'coffee mercaptan': 'Furfuryl Mercaptan',
    'methyl cyclopentenolone': 'Corylone',
    'cyclotene': 'Corylone',
    'methylcyclopentenolone': 'Corylone',
    '3-methyl-1,2-cyclopentanedione': 'Corylone',
    'isoeugenol acetate': 'Isoeugenyl Acetate',
    '3,3-dimethyl-1-indanone': '3,3-Dimethyl-1-Indanone',
    'dimethyl indanone': '3,3-Dimethyl-1-Indanone',

    # Supplier names for the twelve molecules added from pasted
    # formulas. Isoraldeine and Irisone are one molecule under two
    # houses' names, and 'Cetone Alpha' is its grade.
    'isopropyl methyl-2-butyrate': 'Isopropyl 2-Methylbutyrate',
    'isopropyl methyl 2 butyrate': 'Isopropyl 2-Methylbutyrate',
    'isopropyl 2 methylbutyrate': 'Isopropyl 2-Methylbutyrate',
    'isoraldeine': 'Isomethyl Ionone Alpha',
    'isoraldeine 70': 'Isomethyl Ionone Alpha',
    'isoraldeine cetone alpha': 'Isomethyl Ionone Alpha',
    'irisone': 'Isomethyl Ionone Alpha',
    'irisone alpha': 'Isomethyl Ionone Alpha',
    'alpha isomethyl ionone': 'Isomethyl Ionone Alpha',
    'isomethyl ionone': 'Isomethyl Ionone Alpha',
    'methyl ionone alpha iso': 'Isomethyl Ionone Alpha',
    'butylphenyl methylpropional': 'Lilial',
    'lysmeral': 'Lilial',
    'cuminaldehyde': 'Cuminic Aldehyde',
    'cumin aldehyde': 'Cuminic Aldehyde',
    'p-isopropyl benzaldehyde': 'Cuminic Aldehyde',
    '4-methoxyacetophenone': 'Acetanisole',
    'cyclopentadecanone': 'Exaltone',
    'phenirat': 'Phenoxyethyl Isobutyrate',
    'anisyl acetate (4-methoxybenzyl acetate)': 'Anisyl Acetate',

    'ambrofix': 'Ambroxan',
    'ambrofix (ambroxan)': 'Ambroxan',
    # Trade and chemical synonyms for molecules already in the
    # library, each checked against it rather than inferred from a
    # near neighbour.
    'decanal': 'Aldehyde C-10',
    'aldehyde c-10 decanal': 'Aldehyde C-10',
    'allyl caproate': 'Allyl Hexanoate',
    'aldehyde c-18': 'Gamma Nonalactone',
    'aldehyde c-18 coconut': 'Gamma Nonalactone',
    'coconut aldehyde': 'Gamma Nonalactone',
    'aldehyde c-16 strawberry': 'Strawberry Glycidate',
    'aldehyde c-14 peach': 'Gamma Undecalactone',
    'exaltolide': 'Exaltolide Total',
    'pentadecanolide': 'Exaltolide Total',
    'dextro limonene': 'D-Limonene',
    'dextro-limonene': 'D-Limonene',
    'd limonene': 'D-Limonene',
    'orange terpenes': 'D-Limonene',
    'pyralone': '2-Isobutyl Quinoline',
    'isobutyl quinoline': '2-Isobutyl Quinoline',
    'iso butyl quinoline': '2-Isobutyl Quinoline',

    "dmbca": "Dimethyl Benzyl Carbinyl Acetate",
    "pea": "Phenethyl Alcohol",
    "phenylethyl alcohol": "Phenethyl Alcohol",
    "phenyl ethyl alcohol": "Phenethyl Alcohol",
    # Aldehyde C-16 and C-14 are perfumery names for molecules this
    # library holds under their own: the strawberry glycidate and the
    # peach lactone. Neither is an aldehyde.
    "aldehyde c-16": "Strawberry Glycidate",
    "aldehyde c-14": "Gamma Undecalactone",
    "aldehyde c-18": "Gamma Nonalactone",
    "trans-2-hexenal": "(E)-2-Hexenal",
    # Ribes is the blackcurrant genus, so this is the same thiol the
    # library already holds under the English name.
    "ribes mercaptan": "Blackcurrant mercaptan",
    "cassis mercaptan": "Blackcurrant mercaptan",
    "amyl cinnamic aldehyde": "Amyl Cinnamic Aldehyde",
    "alpha-amyl cinnamaldehyde": "Amyl Cinnamic Aldehyde",
    "amyl cinnamaldehyde": "Amyl Cinnamic Aldehyde",
    "trans 2 hexenal": "(E)-2-Hexenal",
    "cis-3-hexenal": "(Z)-3-Hexenal",
    "galaxolide 50": "Galaxolide",
    "galaxolide 50 ipm": "Galaxolide",
    "galaxolide 50 dep": "Galaxolide",
    "beta phenylethyl alcohol": "Phenethyl Alcohol",
    "phenylethanol": "Phenethyl Alcohol",
    "p-cresyl methyl ether": "p-Cresyl Methyl Ether",
    "para cresyl methyl ether": "p-Cresyl Methyl Ether",
    "cresyl methyl ether": "p-Cresyl Methyl Ether",
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
    "cedarwood virginia": "Cedarwood EO (Virginia)",
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
    "lemon fcf": "Lemon EO (Sicilian, cold-pressed)",
    # Hedione HC is the cis-enriched grade of the same molecule, and the
    # percept model's fingerprint is stereo-blind, so it is the same input.
    # The grades do not smell the same on a blotter; nothing here can tell
    # them apart, and pretending otherwise by dropping the ingredient is worse.
    "hedione hc": "Hedione",
    "hedione high cis": "Hedione",
    "hedione hc high cis": "Hedione",
    "veramoss": "Evernyl",
    "methyl atrarate": "Evernyl",
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
    # Vertofix Coeur is IFF's grade name for acetylcedrene; supplier sheets
    # and GC tables write the chemical name, methyl cedryl ketone, and both
    # spellings have to reach this library's one entry for the skeleton.
    "vertofix coeur": "Vertofix Coeur",
    "methyl cedryl ketone": "Vertofix Coeur",
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
    'patchouli alcohol': 'Patchoulol',
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
    'coumarin': 'Coumarin',
    'hexyl salicylate': 'Hexyl Salicylate',
    'otne': 'Iso E Super',
    'ethyl linalool': 'Ethyl Linalool',
    'dihydromyrcenol': 'Dihydro Myrcenol',
    # Inventory/paste spellings: same material, origin or grade qualifier.
    # The library's lime is the Persian expressed build; the perfumer's
    # bottle says "Lime Oil, Distilled" (the usual trade name).
    'lime oil, distilled': 'Lime EO (Persian, expressed)',
    'lime oil': 'Lime EO (Persian, expressed)',
    'lime eo': 'Lime EO (Persian, expressed)',
    'rosemary tunisia': 'Rosemary EO (cineole type)',
    'rosemary': 'Rosemary EO (cineole type)',
    # Tobacarol / Guaiacwood spellings a pasted formula will use.
    'tobacarol': 'Tobacarol',
    'caryophyllene epoxide': 'Tobacarol',
    'beta caryophyllene epoxide': 'Tobacarol',
    'guaiacwood': 'Guaiacwood EO',
    'guaiacwood oil': 'Guaiacwood EO',
    'guaiac wood': 'Guaiacwood EO',
    'guaiol': 'Guaiol',
    'bulnesol': 'Bulnesol',
    # Myrrh spellings a pasted formula will use. "Myrrh EO", "Myrrh
    # Resinoid" and "Myrrh Absolute" carry their own natural hint words
    # and resolve through match_oil on the head word; bare "myrrh" and
    # "myrrh resin" carry none and need the alias.
    'myrrh': 'Myrrh EO',
    'myrrh oil': 'Myrrh EO',
    'myrrh resin': 'Myrrh EO',
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
           # Carrier oils. A formula built for skin application is mostly
           # carrier, and reporting 88% of it as "not in the library" told
           # the reader nothing: it is not meant to smell.
           "mct oil", "mct", "fractionated coconut oil", "coconut oil",
           "jojoba", "jojoba oil", "sweet almond oil", "grapeseed oil",
           "caprylic capric triglyceride", "cyclomethicone",
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
    # Suppliers' sheets arrive with typographic quotes around a marketing
    # grade -- Vanillin "Signature" -- and those dropped the ingredient.
    n = (n.replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u2013", "-").replace("\u2014", "-"))
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"\s*\((?:[^()]*)\)\s*$", "", n)       # trailing parenthetical
    n = re.sub(r"[,—\-–]\s*$", "", n).strip()
    return n


def classify(name: str, library: set[str], _inner: bool = False):
    """Return (kind, resolved_name). kind is material|solvent|natural|base|unknown.

    A supplier often puts the chemical name in brackets after their own trade
    name -- "Ambrofix (Ambroxan)", "Pyralone (Isobutyl Quinoline)" -- so when
    the outer name is unknown, what is inside the brackets is tried as a name
    in its own right. `_inner` stops that recursing.
    """
    raw = norm(name)
    if not _inner:
        for inner in re.findall(r"\(([^()]*)\)", name):
            if not inner.strip():
                continue
            kind, res = classify(inner, library, _inner=True)
            if kind in ("material", "solvent"):
                return kind, res
    # strip trailing qualifiers that describe the stock, not the material
    raw = re.sub(r",?\s*(undiluted|neat|crystals?|natural|total)\s*$", "", raw,
                 flags=re.I).strip() or raw
    low = raw.lower()
    if NOT_INGREDIENT.match(low):
        return "skip", raw
    # strip a stock strength out of the name itself
    low = re.sub(r"\b\d+(?:\.\d+)?\s*%\s*(?:in\s+)?"
                 r"(?:dpg|pg|ipm|tec|benzyl alcohol|alcohol)?\b", "", low).strip()
    # A strength written with its solvent and no percent sign, as in
    # "Galaxolide 50 DEP" or "Hedione 10 DPG", which dropped the whole
    # ingredient -- a tenth of one formula's weight. The solvent token is what
    # makes it unambiguous. A bare "Galaxolide 50" is a trade name rather
    # than a dilution and is aliased explicitly instead, and "Aldehyde
    # C-12" has no solvent after the number and must not be touched.
    low = re.sub(r"\s+\d+(?:\.\d+)?\s*%?\s*"
                 r"(?:dep|dpg|ipm|tec|bb|dowanol|triethyl citrate|"
                 r"benzyl benzoate|ethanol|alcohol)\s*$", "", low).strip()
    # A trailing percentage with no solvent named: "Calone 1%",
    # "2-Acetylpyrazine 1%". Only at the end of the string, so
    # "Galaxolide 50" and "Aldehyde C-12" are untouched.
    low = re.sub(r"[\s,]+\d+(?:\.\d+)?\s*%\s*$", "", low)
    low = re.sub(r"\s+", " ", low).strip(" ,-—")
    # "Aldehyde C14" and "Aldehyde C-14" are the same material written two
    # ways, and the library uses the hyphen. Without this the unhyphenated
    # form was dropped.
    low = re.sub(r"\baldehyde\s+c\s*-?\s*(\d+)", r"aldehyde c-\1", low)
    # A marketing grade in quotes: Vanillin "Signature", Peru Balsam
    # "Signature" Resinoid, Tonka Bean "Luxe" Absolute. The quotes survive
    # norm(); what is inside them never names the material.
    low = re.sub(r'"[^"]*"', " ", low)
    # A grade or origin written as a bare suffix. Closed list, so it cannot
    # eat a real name: "Geraniol Extra", "Guaiacol (Natural)" once the
    # bracket is gone, "Ambrinol 95", "Citronellol 98".
    low = re.sub(r"[\s,]+(?:signature|superior|luxe|super premium|"
                 r"multi-origin|wild|extra|total)\b", " ", low)
    low = re.sub(r"\s+\d{2,3}\s*$", "", low)
    low = re.sub(r"\s+", " ", low).strip(" ,-")
    if not low:
        return "unknown", raw
    if low in SOLVENT:
        return "solvent", raw
    if low in ALIAS:
        return "material", ALIAS[low]
    # A stereo or isomer word at the front, where this library puts it at the
    # back: "Beta Ionone" against "Ionone Beta", "Alpha Ionone", "Delta
    # Octalactone". Reordering is tried before any fuzzy matching, and only
    # for this closed list of words, so it cannot invent a match.
    GREEK = ("alpha", "beta", "gamma", "delta", "epsilon", "iso", "cis",
             "trans", "laevo", "levo", "dextro", "para", "ortho", "meta")
    parts = low.replace("-", " ").split()
    if len(parts) > 1 and parts[0] in GREEK:
        moved = " ".join(parts[1:] + [parts[0]])
        if moved in ALIAS:
            return "material", ALIAS[moved]
        for cand in (moved.title(), moved.capitalize()):
            if cand in library:
                return "material", cand
        hit = [m for m in library if m.lower() == moved]
        if hit:
            return "material", hit[0]
    if len(parts) > 1 and parts[-1] in GREEK:
        moved = " ".join([parts[-1]] + parts[:-1])
        if moved in ALIAS:
            return "material", ALIAS[moved]
        hit = [m for m in library if m.lower() == moved]
        if hit:
            return "material", hit[0]
    # "Gamma-Undecalactone" against "Gamma Undecalactone".
    dehyphen = low.replace("-", " ")
    if dehyphen != low:
        if dehyphen in ALIAS:
            return "material", ALIAS[dehyphen]
        hit = [m for m in library if m.lower() == dehyphen]
        if hit:
            return "material", hit[0]
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


def library_name(name: str, library: set[str], oils=()) -> str:
    """The library's name for a written name, or the written name unchanged.

    One resolution, in one place: the alias table and the normalisation above,
    then `match_oil`'s head-word match for a mixture. Three scripts had already
    written these four lines by hand -- `bench_simplify.py`,
    `threshold_worklist.py`, `volatility_profile.py` -- because a name written
    the way a perfumer writes it is not the name the library stores.

    The join that lacked it is the inventory. `inventory.csv` is keyed by the
    name on the bottle and the library by the name the chemistry has, and the
    two were matched by exact string equality, so "Orange Sweet EO" never met
    "Orange EO (sweet, Florida)": 35 of 166 shelf names matched nothing, and
    each of those materials silently lost its use level, family, volatility
    class and **compiled** threshold, falling back to imputation.

    `oils` is the list of mixture names to match against; without it, only the
    alias table and exact names resolve. The name is returned unchanged when
    nothing matches, so a caller compares the result with its input rather than
    testing a sentinel.
    """
    kind, res = classify(name, library)
    if kind == "material" and res in library:
        return res
    if oils:
        # Deferred: predict_formula imports this module for `classify`, and the
        # head-word matcher lives there because it is a question about mixtures.
        from predict_formula import match_oil
        hit, _ = match_oil(name, list(oils))
        if hit:
            return hit
    return name


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
