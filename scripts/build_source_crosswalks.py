"""Write data/percept/crosswalk_<source>_to_vocab.csv for the four extra label
sources and Good Scents (first-position vote only), plus unmapped_<source>.csv.

The mappings are written by hand below, in the format of
crosswalk_dravnieks_to_vocab.csv (source word, vocab word, note). A source
word is mapped only when it means the same smell as the vocabulary word; one
source word may map to several vocab words. Everything else is left out and
listed in unmapped_<source>.csv with its frequency, so it can be reviewed.

Traps recorded in the project: vocabulary "ethereal" is the ether/solvent
smell; "berry" is the fruit; "green" is an odour, not a colour or a grade;
"aromatic" is not "herbal". A specific word with no vocabulary word of its own
(lilac, mandarin, patchouli, walnut, lime ...) is NOT lifted to its family word.

Left out after a check against the GS-LF labels (scripts/build_soft_targets.py
writes data/percept/crosswalk_audit.csv): Arctander "gassy" (36 training
molecules, 1 of them labelled gassy: not the same smell), Flavornet "gasoline"
(7 molecules, none labelled solvent), Flavornet "tart lime" (was lifted to
citrus against the rule above).

Run with .venv-pyrfume (needs pyrfume for the frequencies).
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "percept"
GSLF = ROOT / "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"

# word -> "vocab" or "vocab1|vocab2"  (optional note after "#")
ARCTANDER = """
acid=sour # acid/pungent-sour smell of the lower acids
aldehydic=aldehydic
almond=almond
ambre=amber
animal=animal
anisic=anisic
apple=apple
apricot=apricot
balsamic=balsamic
banana=banana
berry=berry
brandy=brandy
buttery=buttery
camphoraceous=camphoreous
caramelic=caramellic
citrus=citrus
coco=coconut
coconut=coconut
creamy=creamy
earthy=earthy
ethereal=ethereal # ether-like, diffusive; same as the vocabulary sense
fatty=fatty
floral=floral
fruity=fruity
geranium=geranium
grape=grape
green=green
hay=hay
herbal=herbal
honey=honey
hyacinth=hyacinth
jasmin=jasmin
leafy=leafy
leather=leathery
lily=lily
medicinal=medicinal
metallic=metallic
minty=mint
mushroom=mushroom
musky=musk
musty=musty
nutty=nutty
oily=oily
orange=orange
orange-blossom=orangeflower
orris=orris
peach=peach
pear=pear
phenolic=phenolic
pine=pine
pineapple=pineapple
plum=plum
powdery=powdery
rose=rose
sandalwood=sandalwood
smoky=smoky
sour=sour
spicy=spicy
sulfuraceous=sulfurous
tea=tea
tobacco=tobacco
vanilla=vanilla
vanillin=vanilla # vanillin is the vanilla smell
violet=violet
waxy=waxy
winey=winey
woody=woody
"""

IFRA = """
acidic=sour
aldehydic=aldehydic
almond=almond
amber=amber
animal like=animal
anisic=anisic
apple=apple
apricot=apricot
balsamic=balsamic
banana=banana
bergamot=bergamot
berry=berry
blackcurrant=black currant
blueberry=berry # a berry fruit
buttery=buttery
camphoraceous=camphoreous
caramel=caramellic
cedarwood=cedar
chamomile=chamomile
cheesy=cheesy
cherry=cherry
chocolate=chocolate
cinnamon=cinnamon
citrus=citrus
clean=clean
clove=clove
cocoa=cocoa
coconut=coconut
coffee=coffee
cooling=cooling
creamy=creamy
cucumber=cucumber
dairy=dairy
dry=dry
earthy=earthy
ethereal=ethereal
fatty=fatty
fermented=fermented
licorice=anisic # as in the Dravnieks crosswalk
floral=floral
fresh=fresh
fruity=fruity
garlic=garlic
geranium=geranium
grape=grape
grapefruit=grapefruit
grassy=grassy
green=green
hawthorn=hawthorn
hay=hay
herbal=herbal
honey=honey
honeydew=melon # a melon
hyacinth=hyacinth
jasmin=jasmin
juicy=juicy
lactonic=lactonic
lavender=lavender
leafy=leafy
leathery=leathery
lemon=lemon
lily=lily
medicinal=medicinal
melon=melon
metallic=metallic
milky=milky
minty=mint
mushroom=mushroom
musk like=musk
musty=musty
muguet=muguet
nutty=nutty
oily=oily
onion=onion
orange=orange
orris=orris
ozonic=ozone
peach=peach
pear=pear
phenolic=phenolic
pine=pine
pineapple=pineapple
plum=plum
popcorn=popcorn
potato=potato
powdery=powdery
pungent=pungent
raspberry=raspberry
roasted=roasted
rose=rose
rum=rummy
sandalwood=sandalwood
savoury=savory
sharp=sharp
smoky=smoky
soapy=soapy
sour=sour
spicy=spicy
strawberry=strawberry
sulfurous=sulfurous
sweet=sweet
tea=tea
terpenic=terpenic
tobacco=tobacco
tobacco like=tobacco
tropical-fruit=tropical
vanilla=vanilla
vegetable=vegetable
vetiver=vetiver
violet=violet
warm=warm
waxy=waxy
woody=woody
bitter=bitter
burnt=burnt
brandy=brandy
coumarin=coumarinic
celery=celery
neroli=orangeflower # neroli is the orange-blossom smell
"""

SIGMA = """
alcohol=alcoholic
almond=almond
alliaceous=alliaceous
alliaceous (onion, garlic)=alliaceous|onion|garlic # the descriptor names both
anise=anisic
animal=animal
apple=apple
apricot=apricot
balsam=balsamic
balsamic=balsamic
banana=banana
beef=beefy
berry=berry
blueberry=berry # a berry fruit
butter=buttery
cabbage=cabbage
camphoraceous=camphoreous
cantaloupe=melon
caramel=caramellic
cedar=cedar
celery=celery
cheese=cheesy
cherry=cherry
chocolate=chocolate
cinnamon=cinnamon
citrus=citrus
clove=clove
coffee=coffee
coconut=coconut
coumarin=coumarinic
creamy=creamy
cucumber=cucumber
earthy=earthy
ethereal=ethereal
fatty=fatty
fishy=fishy
floral=floral
fruity=fruity
gardenia=
geranium=geranium
grape=grape
grapefruit=grapefruit
green=green
hawthorne=hawthorn
hazelnut=hazelnut
herbaceous=herbal
honey=honey
hyacinth=hyacinth
iris=orris # iris root is orris
jasmine=jasmin
lavender=lavender
leafy=leafy
lemon=lemon
lily=lily
medicinal=medicinal
melon=melon
meaty=meaty
minty=mint
mossy=
mushroom=mushroom
musty=musty
nutty=nutty
oily=oily
orange=orange
peach=peach
pear=pear
pineapple=pineapple
plum=plum
potato=potato
raspberry=raspberry
rose=rose
rum=rummy
smoky=smoky
soapy=soapy
sour=sour
spicy=spicy
strawberry=strawberry
sulfurous=sulfurous
sweet=sweet
tobacco=tobacco
tropical=tropical
vanilla=vanilla
vegetable=vegetable
violet=violet
warm=warm
waxy=waxy
wine-like=winey
winelike=winey
woody=woody
brandy=brandy
"""

FLAVORNET = """
acid=sour
alcohol=alcoholic
almond=almond
anise=anisic
apple=apple
apple peel=apple
apple, rose=apple|rose
apricot=apricot
balsamic=balsamic
banana=banana
bitter=bitter
bitter almond=almond
black currant=black currant
boiled vegetable=vegetable|cooked
burnt=burnt
burnt sugar=burnt|caramellic
butter=buttery
cabbage=cabbage
camomile=chamomile
camphor=camphoreous
caramel=caramellic
cheese=cheesy
citrus=citrus
cinnamon=cinnamon
clove=clove
cocoa=cocoa
coconut=coconut
coffee=coffee
cognac=cognac
cooked meat=cooked|meaty
cooked potato=cooked|potato
cooked vegetable=vegetable|cooked
coumarin=coumarinic
cream=creamy
cucumber=cucumber
earth=earthy
ether=ethereal # ether smell, same as the vocabulary sense
fat=fatty
fish=fishy
flower=floral
fresh=fresh
fruit=fruity
garlic=garlic
geranium=geranium
grape=grape
grapefruit=grapefruit
grass=grassy
green=green
green leaf=green|leafy
green bean=vegetable
green tea=tea
hawthorne=hawthorn
hazelnut=hazelnut
herb=herbal
honey=honey
hot milk=milky
hyacinth=hyacinth
jasmine=jasmin
lactone=lactonic
lavender=lavender
leaf=leafy
lemon=lemon
licorice=anisic # as in the Dravnieks crosswalk
lily=lily
malt=malty
meat=meaty
meat broth=meaty
medicine=medicinal
melon=melon
metal=metallic
mildew=musty
mint=mint
mold=musty
mothball=camphoreous # as in the Dravnieks crosswalk
muguet=muguet
mushroom=mushroom
must=musty
nut=nutty
oil=oily
onion=onion
orange=orange
orange peel=orange
orris=orris
peach=peach
pear=pear
peppermint=mint
phenol=phenolic
pine=pine
pineapple=pineapple
plum=plum
popcorn=popcorn
potato=potato
pungent=pungent
raspberry=raspberry
roast=roasted
roast beef=roasted|meaty|beefy
roasted meat=roasted|meaty
roasted nut=roasted|nutty
rose=rose
sharp=sharp
smoke=smoky
soap=soapy
solvent=solvent
sour=sour
spearmint=mint
spice=spicy
strawberry=strawberry
sulfur=sulfurous
sweat=sweaty
sweet=sweet
tea=tea
tobacco=tobacco
tomato=tomato
tomato leaf=tomato
vanilla=vanilla
violet=violet
warm=warm
wax=waxy
wine=winey
wood=woody
"""

# Good Scents: only the first-listed word is used (the vote of step 3), so only
# words that can appear first need a map. Words that are exactly a vocabulary
# word map to themselves (added automatically); these are the extras.
GOODSCENTS_EXTRA = """
minty=mint
anise=anisic
licorice=anisic
acidic=sour
sulfury=sulfurous
camphor=camphoreous
camphoraceous=camphoreous
lily of the valley=muguet
balsam=balsamic
blueberry=berry # a berry fruit
currant bud black currant bud=black currant
coffee roasted coffee=coffee
hazelnut roasted hazelnut=hazelnut
meaty roasted meaty=meaty
neroli=orangeflower # neroli is the orange-blossom smell
coumarin=coumarinic
tropical fruit=tropical
tropical-fruit=tropical
musky=musk
ambre=amber
leather=leathery
orange blossom=orangeflower
orange-blossom=orangeflower
jasmine=jasmin
herbaceous=herbal
winey=winey
"""


def parse(block: str) -> dict[str, tuple[list[str], str]]:
    out = {}
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        src, rest = line.split("=", 1)
        note = ""
        if "#" in rest:
            rest, note = rest.split("#", 1)
        voc = [v.strip() for v in rest.split("|") if v.strip()]
        if voc:
            out[src.strip().lower()] = (voc, note.strip())
    return out


def vocab() -> list[str]:
    with open(GSLF, newline="", encoding="utf-8") as fh:
        head = next(csv.reader(fh))
    return [c for c in head if c not in ("nonStereoSMILES", "descriptors")]


def load_crosswalk(source: str) -> dict[str, list[str]]:
    """source word -> vocab words, as written by main()."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    with open(OUT / f"crosswalk_{source}_to_vocab.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r[source].strip().lower()].append(r["vocab"])
    return dict(out)


def main() -> int:
    voc = vocab()
    assert len(voc) == 138, len(voc)
    sys.path.insert(0, str(ROOT / "scripts"))
    from soft_target_sources import load_all
    res = load_all()
    maps = {"arctander": parse(ARCTANDER), "ifra": parse(IFRA),
            "sigma": parse(SIGMA), "flavornet": parse(FLAVORNET)}
    gs = parse(GOODSCENTS_EXTRA)
    for w in res["goodscents"]["words"].values():
        for x in w:
            if x in voc and x not in gs:
                gs[x] = ([x], "identical word")
    maps["goodscents"] = gs

    summary = []
    for src, m in maps.items():
        bad = [(s, v) for s, (vs, _) in m.items() for v in vs if v not in voc]
        assert not bad, (src, bad)
        freq = collections.Counter(
            w for ws in res[src]["words"].values() for w in ws)
        if src == "goodscents":   # only first-listed words matter here
            freq = collections.Counter(
                w for ws in res[src]["first"].values() for w in ws)
        with open(OUT / f"crosswalk_{src}_to_vocab.csv", "w", newline="",
                  encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow([src, "vocab", "note"])
            for s, (vs, note) in sorted(m.items()):
                if s not in freq:
                    continue
                for v in vs:
                    wr.writerow([s, v, note])
        unm = [(w, n) for w, n in freq.most_common() if w not in m]
        with open(OUT / f"unmapped_{src}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow([src, "molecules"])
            wr.writerows(unm)
        mapped = [w for w in freq if w in m]
        summary.append((src, len(freq), len(mapped),
                        sum(freq[w] for w in mapped), sum(freq.values())))
    print("source, distinct words, mapped, word-uses mapped, word-uses total")
    for s in summary:
        print(*s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
