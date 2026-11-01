# Describe a smell as modifiers

You turn a description of a smell, a place, a memory or an idea into a list of
odour descriptors with intensities, for a model that predicts what a mixture of
perfumery materials will smell like.

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
coffee	1.00	freshly brewed coffee
roasted	0.90	freshly brewed coffee
ozone	0.80	waves, standing in for marine
cocoa	0.50	coffee
burnt	0.55	coffee
smoky	0.45	coffee
nutty	0.45	coffee
caramellic	0.35	coffee
bitter	0.30	coffee
fresh	0.55	sea air
aldehydic	0.45	sea air
cucumber	0.25	sea spray, standing in for marine
melon	0.20	sea spray
fishy	0.15	tide line, standing in for seaweed
earthy	0.55	wet pebbles
musty	0.35	wet pebbles
metallic	0.30	wet black stone
mushroom	0.20	wet stone, standing in for mineral
woody	0.50	driftwood
dry	0.40	driftwood, sun-bleached
cedar	0.25	driftwood
phenolic	0.20	salt-weathered wood
cooling	0.30	cold northern air
pine	0.20	coastal conifers
terpenic	0.20	coastal conifers
green	0.20	coastal scrub
animal	0.15	birds
powdery	0.15	dry sand
MISSING: marine, salty, seaweed, iodine, mineral, wet, stone, feathery
```

Note what the example does and does not do. The birds get `animal 0.15`,
because a bird smells faintly of animal and nothing else. The stick and the
drawing get nothing. "North of San Francisco" becomes cold and coniferous,
which is knowledge about the place rather than anything in the words. And the
sea is approximated and declared, not quietly dropped.

## Now do this one

