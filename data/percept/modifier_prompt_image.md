# Describe a photograph as modifiers

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

## The only words you may use

These 138 descriptors are the complete output vocabulary of the model this
feeds. You may not use any other word, and you may not invent one.

  alcoholic  aldehydic  alliaceous  almond  amber  animal
  anisic  apple  apricot  aromatic  balsamic  banana
  beefy  bergamot  berry  bitter  black currant  brandy
  burnt  buttery  cabbage  camphoreous  caramellic  cedar
  celery  chamomile  cheesy  cherry  chocolate  cinnamon
  citrus  clean  clove  cocoa  coconut  coffee
  cognac  cooked  cooling  cortex  coumarinic  creamy
  cucumber  dairy  dry  earthy  ethereal  fatty
  fermented  fishy  floral  fresh  fruit skin  fruity
  garlic  gassy  geranium  grape  grapefruit  grassy
  green  hawthorn  hay  hazelnut  herbal  honey
  hyacinth  jasmin  juicy  ketonic  lactonic  lavender
  leafy  leathery  lemon  lily  malty  meaty
  medicinal  melon  metallic  milky  mint  muguet
  mushroom  musk  musty  natural  nutty  odorless
  oily  onion  orange  orangeflower  orris  ozone
  peach  pear  phenolic  pine  pineapple  plum
  popcorn  potato  powdery  pungent  radish  raspberry
  ripe  roasted  rose  rummy  sandalwood  savory
  sharp  smoky  soapy  solvent  sour  spicy
  strawberry  sulfurous  sweaty  sweet  tea  terpenic
  tobacco  tomato  tropical  vanilla  vegetable  vetiver
  violet  warm  waxy  weedy  winey  woody

## What to return

A fenced block containing one modifier per line, tab-separated:

    descriptor<TAB>intensity<TAB>what it came from

Then a line beginning `MISSING:` listing every quality the input calls for that
has no descriptor above. If nothing is missing, write `MISSING: none`.

Return the block and nothing else. No preamble, no explanation after it.

## How to choose the intensities

Intensity runs from 0 to 1 and means **how much of the finished smell this
quality should be** -- not how confident you are that it belongs.

Weight by what the picture is of, not by how much of the frame a thing fills.
The subject is usually what is in focus, closest, central, or what the
photograph was obviously taken for. A close-up of a coffee cup on a beach is
mostly coffee; a wide shot of the same beach with a cup in the corner is mostly
beach. Sky and distance are nearly always setting, not subject.

Then the season, the weather and whether things are wet. These change a scene's
smell more than any object in it, and a photograph shows them where prose often
forgets to mention them.

Flattening everything to the same intensity is the most common mistake and
produces a muddle with no subject.

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

## Named products are accords, not their advertising

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

## Worked example

For a photograph of a wet cobbled street at dusk in late autumn, a lit bakery
window on the left, fallen leaves banked in the gutter, a parked bicycle:

```
earthy	0.70	wet stone and soil between the cobbles
musty	0.60	fallen leaves, part-rotted
green	0.30	leaves still turning
hay	0.35	dry leaf litter at the edges
mushroom	0.30	decay under the leaf bank
woody	0.35	wet wood, the tree and the bicycle frame
caramellic	0.65	baking, through the bakery window
sweet	0.60	baking
vanilla	0.45	baking
buttery	0.45	baking
malty	0.40	bread crust
roasted	0.30	bread crust
cooling	0.35	cold damp air at dusk
fresh	0.30	rain-washed air
metallic	0.25	the bicycle, wet steel
phenolic	0.15	wet tarmac at the kerb
ozone	0.20	after rain
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

