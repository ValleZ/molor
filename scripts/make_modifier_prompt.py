"""Generate self-contained prompts that turn prose, or an image, into modifiers.

The percept model cannot read text or see, and a hosted page cannot carry an
API key for a free public project. So the language step runs wherever the
reader already has a model: paste a prompt plus a description or a photograph
into any chat, get back a modifier list, paste that into
scripts/modifiers_to_formula.py.

That split has a second benefit. Each prompt is the whole specification of its
step, in one file, in plain words -- so the same input can go to several models
and the answers be compared, which is the only way to find out whether a small
model is enough.

Two prompts, because the two inputs fail differently.

Prose is already a selection: whoever wrote it chose what mattered and marked
the subject grammatically. A photograph is not. It carries sand colour, sky,
composition, a parked car and a sunscreen bottle, nearly all of which has no
smell, and nothing in it says which object the picture is *about*. So the image
prompt spends its length on what to ignore, and asks for the visible-and-
odorous things to be listed before any descriptor is assigned, which makes the
selection step auditable rather than implicit.

The image prompt also has to forbid one specific mistake that the prose one
never provokes: reading colour and light as smell. Blue is not marine, warm
light is not amber, and a red thing does not smell of berries. That is
synaesthesia, and it produces a confident modifier list with no evidence under
it.

    python scripts/make_modifier_prompt.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VOCAB_SECTION = """## The only words you may use

These {n} descriptors are the complete output vocabulary of the model this
feeds. You may not use any other word, and you may not invent one.

{vocab}
"""

FORMAT_SECTION = """## What to return

A fenced block containing one modifier per line, tab-separated:

    descriptor<TAB>intensity<TAB>what it came from

Then a line beginning `MISSING:` listing every quality the input calls for that
has no descriptor above. If nothing is missing, write `MISSING: none`.

Return the block and nothing else. No preamble, no explanation after it.

## How to choose the intensities

Intensity runs from 0 to 1 and means **how much of the finished smell this
quality should be** -- not how confident you are that it belongs.

{weighting}

Use between 8 and 30 modifiers. Fewer than 8 and the target is too vague to
match; more than 30 and everything averages to nothing in particular.

## Where a word is missing, say so

Several ordinary smell words are absent from the vocabulary above: there is no
`marine`, no `seaweed`, no `iodine`, no `mineral`. Some things are not smells
at all -- `salty` has no odour, since salt is not volatile, and `wet`, `dusty`
and `feathery` are textures rather than smells.

When the input calls for one of these, do two things. Put the nearest
available descriptors in the list, and name the missing quality on the
`MISSING:` line. Both halves matter: the approximation is what gets made, and
the missing line is what stops the reader believing it was captured.
"""

PROSE_HEAD = """# Describe a smell as modifiers

You turn a description of a smell, a place, a memory or an idea into a list of
odour descriptors with intensities, for a model that predicts what a mixture of
perfumery materials will smell like.

Your hardest job is not finding words for a smell. It is resisting the obvious
word for a smell you have only ever read about -- see "Named products are
accords" below.
"""

PROSE_WEIGHTING = """Weight the subject of the description above its setting. A sentence about
drinking coffee on a beach is mostly coffee: the coffee descriptors belong near
1.0 and the beach near 0.3, because that is the proportion a person would
smell. Flattening everything to the same intensity is the most common mistake
and produces a muddle with no subject.

Work through the input element by element -- each object, place, material,
weather, time of day -- and give each element its own descriptors. Something
with no smell contributes nothing: birds, a stick, a colour, a sound, a
feeling. Do not invent descriptors for them."""

PROSE_EXAMPLE = """## Worked example

Input:

> drinking freshly brewed coffee while sitting next to waves on a black beach
> north of san francisco while watching birds and using a driftwood stick to
> draw on the wet pebbles

Output:

```
coffee\t1.00\tfreshly brewed coffee
roasted\t0.90\tfreshly brewed coffee
ozone\t0.80\twaves, standing in for marine
burnt\t0.55\tcoffee
cocoa\t0.50\tcoffee
nutty\t0.45\tcoffee
smoky\t0.45\tcoffee
caramellic\t0.35\tcoffee
bitter\t0.30\tcoffee
fresh\t0.55\tsea air
aldehydic\t0.45\tsea air
cucumber\t0.25\tsea spray, standing in for marine
melon\t0.20\tsea spray
fishy\t0.15\ttide line, standing in for seaweed
earthy\t0.55\twet pebbles
musty\t0.35\twet pebbles
metallic\t0.30\twet black stone
mushroom\t0.20\twet stone, standing in for mineral
woody\t0.50\tdriftwood
dry\t0.40\tdriftwood, sun-bleached
cedar\t0.25\tdriftwood
phenolic\t0.20\tsalt-weathered wood
cooling\t0.30\tcold northern air
pine\t0.20\tcoastal conifers
terpenic\t0.20\tcoastal conifers
green\t0.20\tcoastal scrub
animal\t0.15\tbirds
powdery\t0.15\tdry sand
MISSING: marine, salty, seaweed, iodine, mineral, wet, stone, feathery
```

Note what the example does and does not do. The birds get `animal 0.15`,
because a bird smells faintly of animal and nothing else. The stick and the
drawing get nothing. "North of San Francisco" becomes cold and coniferous,
which is knowledge about the place rather than anything in the words. And the
sea is approximated and declared, not quietly dropped.

## Now do this one

"""

IMAGE_HEAD = """# Describe a photograph as modifiers

You look at an image and write what the scene in it would smell like, as a
list of odour descriptors with intensities, for a model that predicts what a
mixture of perfumery materials will smell like.

## First, before any descriptor: say what you can smell

Begin by listing, to yourself, every thing in the image that has a smell at
all. This is most of the work, because a photograph is mostly things that do
not.

Include: plants, food, drink, wood, earth, stone, water, smoke, rain, snow,
leather, metal, rust, fabric, skin, animals, fuel, paint, cut grass, decay,
cooking, the sea, a shop interior, weather, season, time of day, and whether
surfaces are wet or dry.

Exclude: colour, light, shadow, composition, framing, grain, people's faces,
expressions, clothing style, writing, signage, screens, vehicles at a distance,
architecture as shape rather than material, and the photograph's mood.

## Do not read colour as smell

This is the mistake to avoid, and it is specific. **A colour is not a smell.**
Blue is not marine. Warm light is not amber. Something red does not smell of
berries, and a pink sunset does not smell floral. Green paint has no green
note. If your reason for a descriptor is a hue, a brightness or an atmosphere,
drop it -- you are describing the picture, not the air in it.

A smell needs a physical source that volatilises. Name the source in the third
column of every line, and if you cannot name one, the line does not belong.
"""

IMAGE_WEIGHTING = """Weight by what the picture is of, not by how much of the frame a thing fills.
The subject is usually what is in focus, closest, central, or what the
photograph was obviously taken for. A close-up of a coffee cup on a beach is
mostly coffee; a wide shot of the same beach with a cup in the corner is mostly
beach. Sky and distance are nearly always setting, not subject.

Then the season, the weather and whether things are wet. These change a scene's
smell more than any object in it, and a photograph shows them where prose often
forgets to mention them.

Flattening everything to the same intensity is the most common mistake and
produces a muddle with no subject."""

IMAGE_EXAMPLE = """## Worked example

For a photograph of a wet cobbled street at dusk in late autumn, a lit bakery
window on the left, fallen leaves banked in the gutter, a parked bicycle:

```
earthy\t0.70\twet stone and soil between the cobbles
musty\t0.60\tfallen leaves, part-rotted
green\t0.30\tleaves still turning
hay\t0.35\tdry leaf litter at the edges
mushroom\t0.30\tdecay under the leaf bank
woody\t0.35\twet wood, the tree and the bicycle frame
caramellic\t0.65\tbaking, through the bakery window
sweet\t0.60\tbaking
vanilla\t0.45\tbaking
buttery\t0.45\tbaking
malty\t0.40\tbread crust
roasted\t0.30\tbread crust
cooling\t0.35\tcold damp air at dusk
fresh\t0.30\train-washed air
metallic\t0.25\tthe bicycle, wet steel
phenolic\t0.15\twet tarmac at the kerb
ozone\t0.20\tafter rain
MISSING: bready, petrichor, wet, stone, cold
```

Note what the example does and does not do. The bakery is the subject because
it is lit and the photograph was taken for it, so its descriptors run highest
even though the street fills more of the frame. The leaves get `musty` and
`mushroom` from decay, not from being orange. The bicycle contributes
`metallic` because wet steel smells, not because bicycles are evocative. Dusk
contributes nothing by itself -- the cold damp air does. And `bready` and
`petrichor` are named as missing rather than approximated in silence; the
`caramellic`/`malty`/`sweet` group is the approximation, and it is not the
same thing.

Every line in that block names a descriptor from the vocabulary. The smell of
baking bread has no descriptor of its own, so it appears only on the `MISSING:`
line, never as a modifier. There is no such thing as a zero-intensity row, or a
row naming a word outside the list: a line you cannot write is a line that goes
on `MISSING:`.

## If the image has little to smell

Say so rather than inventing. A screenshot, a diagram, a night sky, a page of
text or a plain wall has almost no odour content. Return the few modifiers you
can actually justify and put the rest on the `MISSING:` line. A short honest
list is worth more than a long invented one.

## Now do this image

"""


PRODUCT_RULE = """## Named products are accords, not their advertising

When the input names a product or a cultural shorthand -- sunscreen, clean
laundry, a new car, old books, a hospital, a swimming pool, play-dough -- give
the descriptors of what it actually emits, not of what it is associated with.
These are composed smells, and the association is usually one loud ingredient
that may not be in the thing at all.

The case that taught this rule: a brief saying "a distinct sunscreen vibe" was
read by three different models as coconut and creamy, the suntan-lotion
cliche. The formula it described contains no coconut and nothing creamy. Its
sunscreen quality comes from salicylates, which is what sun cream is actually
built on -- the UV filters themselves are salicylates and benzoates -- and in
this vocabulary those read:

    balsamic, herbal, aromatic, green, floral

So "sunscreen" is `balsamic` with `herbal` and `aromatic` beside it, and
`coconut` only when the product in question is a coconut-scented one. The same
test applies to every shorthand: ask what molecules leave the thing and reach
a nose, not what the packaging shows.

Two more, by the same logic. "Clean laundry" is `musk`, `soapy`, `aldehydic`,
`floral` -- not `fresh` alone, and nothing about cotton. "Old books" is
`musty`, `woody`, `vanilla`, `dry` -- vanilla because degrading paper really
does emit vanillin.
"""


def build(desc, head, weighting, example):
    lines = []
    for i in range(0, len(desc), 6):
        lines.append("  " + "  ".join(desc[i:i + 6]))
    vocab = "\n".join(lines)
    parts = [
        head,
        VOCAB_SECTION.format(n=len(desc), vocab=vocab),
        FORMAT_SECTION.format(weighting=weighting),
        PRODUCT_RULE,
        example,
    ]
    return "\n".join(parts).replace("\\t", "\t")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--out", default="data/percept/modifier_prompt.md")
    ap.add_argument("--out-image",
                    default="data/percept/modifier_prompt_image.md")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = sorted(pj["descriptors"])

    for path, head, weighting, example in (
            (args.out, PROSE_HEAD, PROSE_WEIGHTING, PROSE_EXAMPLE),
            (args.out_image, IMAGE_HEAD, IMAGE_WEIGHTING, IMAGE_EXAMPLE)):
        text = build(desc, head, weighting, example)
        out = ROOT / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {path}: {len(text)} characters, "
              f"about {round(len(text) / 3.6)} tokens")
    print(f"  {len(desc)} descriptors in both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
