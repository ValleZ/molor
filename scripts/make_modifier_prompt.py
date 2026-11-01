"""Generate a self-contained prompt that turns prose into a modifier list.

The percept model cannot read text, and a hosted page cannot carry an API key
for a free public project. So the language step runs wherever the reader
already has a model: paste this prompt plus a description into any chat, get
back a modifier list, paste that into the solver.

That split has a second benefit. The prompt is the whole specification of the
language step, in one file, in plain words -- so it can be pasted into several
models and the answers compared, which is the only way to find out whether a
small model is enough for this.

Why the vocabulary is in the prompt rather than described: the model may use
only these 138 words, because they are exactly what the percept model emits. A
quality with no descriptor cannot be requested however well it is described,
and the prompt asks for those to be listed separately instead of being
approximated silently -- which is the difference between a tool that says "the
sea is not in my vocabulary" and one that quietly drops it.

    python scripts/make_modifier_prompt.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = """# Describe a smell as modifiers

You turn a description of a smell, a place, a memory or an idea into a list of
odour descriptors with intensities, for a model that predicts what a mixture of
perfumery materials will smell like.

## The only words you may use

These {n} descriptors are the complete output vocabulary of the model this
feeds. You may not use any other word, and you may not invent one.

{vocab}

## What to return

A fenced block containing one modifier per line, tab-separated:

    descriptor<TAB>intensity<TAB>what in the input it came from

Then a line beginning `MISSING:` listing every quality the input asks for that
has no descriptor above. If nothing is missing, write `MISSING: none`.

Return the block and nothing else. No preamble, no explanation after it.

## How to choose the intensities

Intensity runs from 0 to 1 and means **how much of the finished smell this
quality should be** -- not how confident you are that it belongs.

Weight the subject of the description above its setting. A sentence about
drinking coffee on a beach is mostly coffee: the coffee descriptors belong near
1.0 and the beach near 0.3, because that is the proportion a person would
smell. Flattening everything to the same intensity is the most common mistake
and produces a muddle with no subject.

Work through the input element by element -- each object, place, material,
weather, time of day -- and give each element its own descriptors. Something
with no smell contributes nothing: birds, a stick, a colour, a sound, a
feeling. Do not invent descriptors for them.

Use between 8 and 30 modifiers. Fewer than 8 and the target is too vague to
match; more than 30 and everything averages to nothing in particular.

## Where a word is missing, say so

Several ordinary smell words are absent from the vocabulary above: there is no
`marine`, no `seaweed`, no `iodine`, no `mineral`, no `smoke` except `smoky`.
Some things are not smells at all -- `salty` has no odour, since salt is not
volatile, and `wet`, `dusty` and `feathery` are textures.

When the input asks for one of these, do two things. Put the nearest available
descriptors in the list, and name the missing quality on the `MISSING:` line.
Both halves matter: the approximation is what gets made, and the missing line
is what stops the reader believing it was captured.

## Worked example

Input:

> drinking freshly brewed coffee while sitting next to waves on a black beach
> north of san francisco while watching birds and using a driftwood stick to
> draw on the wet pebbles

Output:

```
coffee\t1.00\tfreshly brewed coffee
roasted\t0.90\tfreshly brewed coffee
ozone\t0.80\twaves, standing in for marine
cocoa\t0.50\tcoffee
burnt\t0.55\tcoffee
smoky\t0.45\tcoffee
nutty\t0.45\tcoffee
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--out", default="data/percept/modifier_prompt.md")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = sorted(pj["descriptors"])

    # Six to a line: short enough to scan, few enough lines to paste.
    lines = []
    for i in range(0, len(desc), 6):
        lines.append("  " + "  ".join(f"{w}" for w in desc[i:i + 6]))
    vocab = "\n".join(lines)

    text = TEMPLATE.format(n=len(desc), vocab=vocab).replace("\\t", "\t")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    chars = len(text)
    print(f"wrote {args.out}")
    print(f"  {len(desc)} descriptors, {chars} characters, "
          f"about {round(chars / 3.6)} tokens")
    print(f"  paste it into any chat, then the description underneath")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
