# Where a threshold figure came from, where it matters

`inventory.csv`'s `water_threshold_ppb` column was assembled from published
and web sources, with some values approximated from air-phase figures. It
tracks the 28 Abraham measurements at Spearman +0.39 where a perfumer's own
working use levels manage +0.76, so the column is the weakest input the dose
conversion has. This file records the cases where a particular figure was
questioned and what the answer turned out to be, so the same ground is not
re-argued.

## Iso E Super, 0.5 ppb — questioned, and the figure stands

It carries 84–87% of predicted odour activity in three of five formulas
written by a language model, at 24–44% of their mass, which made every one of
them read `cedar, amber, dry` whatever the brief said. It is also 4th on
`threshold_worklist.py`'s influence ranking, so it moves more formulas than
almost anything else here.

**The claim made against it, which was wrong.** That 0.5 ppb looked like an
air-phase number in a water-phase column, because nobody doses a 0.5 ppb
material at 25% of a formula.

**Why that reasoning fails.** Commercial Iso E Super is a mixture of isomers,
and the potency does not belong to the bulk of it.

| | threshold in air | |
|---|---|---|
| major isomer, 40–60% of the mixture | 500 ng/L = 0.052 ppm | weak |
| minor isomer (Arborone), about 5% | 5 pg/L = 5.2 × 10⁻⁷ ppm | 100,000× more potent |

The minor isomer carries essentially all of the woody-amber radiance. At 5% of
the mixture, the mixture itself becomes detectable around 0.10 ng/L — roughly
**5,000× more potent than its own major isomer**. You dose it at 25% *because*
the bulk is weak, and it dominates anyway because a twentieth of it is
detectable at five picograms per litre. Both facts are true at once, and the
second is the one that governs dosing.

Against its neighbours in the same column, 0.5 ppb sits between damascenone at
0.002 and cashmeran at 5. For a material of that kind it is defensible, and if
anything too weak rather than too potent.

**What follows.** The engine's reading of those formulas was right and the
objection was mine. A formula that is 44% Iso E Super really is woody-amber in
the drydown; the brief lives in the top notes, which is what
`scripts/volatility_profile.py` was built to show. The threshold was a red
herring.

**What was not done, and why.** The figure was not replaced with the air-phase
number. Converting 0.10 ng/L in air to a water-phase ppb needs a Henry's law
constant per molecule, which this project does not have, and an unmarked
conversion is how the column became untrustworthy in the first place. An
air-phase figure in a water-phase column would make it worse, not better. The
isomer data is recorded here instead, as the provenance of a value that was
previously an unexplained outlier.

*Isomer thresholds supplied by the perfumer, 2026-11-01.*

## Galaxolide, 10 ppb — the bench puts it near 12

**The comparison.** Sniffed against Hedione on 2026-11-06, the bottle was
clearly about 3x louder, and about level with Ethyl Maltol at 1%. An earlier
memory of it as weaker than Hedione was wrong, which is why it was checked.

**The number.** The inventory's bottle is Galaxolide 50% IPM and the library's
rows are neat, so 3x as the solution is about 6x neat. That gives an effective
threshold of 72.7 / 6 = 12.1 ppb, a multiplier of 0.826 on the compiled 10 ppb.
If the bottle sniffed was neat, it should be 24.2 ppb instead. `judgements` is 0
because this was a bottle comparison, not a blotter session, and the row has to
be re-entered if `apply_strength.py` regenerates `strength.csv`.

**What it does not settle.** Published work puts the active (4S) isomers at
1 ng/L in air or below, seen second-hand, not in the primary paper (Frater,
Muller and Kraft, Helv. Chim. Acta, 1999). With ECHA's vapour pressure
(0.0727 Pa) and solubility (1.75 mg/L) that is about 0.2 ppb in water. It was
not used: a detection figure answers where a material becomes noticeable, and
this bench comparison answers how strong it is at a working dose. An unmarked
conversion is also how this column became untrustworthy.

*Comparison supplied by the perfumer, 2026-11-06.*

## Cedarwood EO (Virginia), 200 ppb — a balance test against Calone puts it at 2.0

**The figure had no source.** 45 of the 47 naturals carry a threshold with no
source and no boiling point, and this one sat at 200 ppb, which made it one of
the weakest bodies in the library. On the strips of 2026-11-07 it ranked among
the loudest materials on the shelf, level with Ylang Ylang EO and below only
Vetiver EO.

**The number.** In five compounded sea mixes (2026-11-06), cedarwood and
Calone took over from each other at roughly similar doses, within a factor of
about 2. That puts cedarwood's effective threshold near 10 ppb (5 to 20), about
20x below the stored figure. The row in `strength.csv` is 3 bins (27x), so
7.4 ppb, the upper of the two whole-bin choices: a neat strip saturates, so
the mixes are the better guide to dosing, and they sit between 2 and 3 bins.
`judgements` is 0 because this came from mixes, not a blotter session, and the
row has to be re-entered if `apply_strength.py` regenerates `strength.csv`.

**The balance test (2026-11-07), which replaces the estimate above.** Calone
1951 at 4.7 parts with cedarwood at 1, 2, 3, 4.5 and 6.5 parts in DPG, smelled
at sea-mix strength (let down to 10%). Cedarwood at 2 parts was the most neutral
and at 3 it led, so the balance is about 2.3 parts. On the linear model, equal
loudness means 2.3 / t = 4.7 / 4.11, so t = 2.01 ppb, a multiplier of 99.4
(4.19 bins). That replaces the 3-bin row. It is relative to Calone's 4.11
ppb, which itself rests on a blotter multiplier measured at top % (see the last
section), so if Calone moves, cedarwood moves with it.

**What it does not settle.** Ylang Ylang EO (359, imputed) and Vetiver EO (60)
ranked as loud or louder on the same strips, but no mix measured their size,
so they are left as they are until one does.

*Mixes and strip ranking supplied by the perfumer, 2026-11-06 and 2026-11-07.*

## The blotter-session multipliers rest on top %, which is no longer used

On 2026-11-07 the perfumer asked that nothing use the sheet's top % column.
`build_inventory.py` no longer copies it, so it is gone from `inventory.csv`
and everything downstream, and `strength_worksheet.py`, which prepared each
blotter at that level, was removed.

The 23 rows in `strength.csv` with `judgements` above 0 came from those
sessions. Each strip was made at the material's top %, and a bin means
"louder or quieter than its use level implies". So the multipliers still carry
top % inside them. They stay for now because they are the only measurement
for those materials, but they are suspect. Calone 1951's 0.243 (judged two
bins weak) is the one that matters most: it held Calone to 1.23 parts, and the
sea mixes say it needs far more.

New corrections come from direct comparisons instead, entered by hand with
`judgements` 0, like Galaxolide and Cedarwood EO above: a balance test between
two materials at working strength, or a ranking of strips at stated dilutions.
`apply_strength.py` would overwrite those hand rows if re-run on the old
session sheets, so don't.

## Ylang Ylang EO (extra) and Vetiver EO (Haiti) — set from the strip ranking

On the neat strips of 2026-11-07, Ylang Ylang EO ranked level with Cedarwood
EO and Vetiver EO above both. Cedarwood is the anchor, at 2.01 ppb from the
balance test against Calone. So Ylang Ylang (359 ppb, imputed, no source) is
set equal to it, 2.01 ppb (x178.6), and Vetiver (60 ppb, no source) one bin
louder, 0.67 ppb (x89.5). The ranks are measured; one bin for "louder than"
is the smallest step a ranking can express, so Vetiver could be stronger
still. A balance test against cedarwood, as with Calone, would pin it.
`judgements` is 0 for both, so `apply_strength.py` must not regenerate the file.

*Strip ranking supplied by the perfumer, 2026-11-07.*

## Strip ranking of the formula materials (2026-11-07), and a fit over both sessions

The perfumer ranked, loudest first and all neat except the labdanum:
Vetiver EO, Hexyl Acetate, Lemonile, Labdanum Resinoid 50% TEC, Cedarwood
EO, Cis-3-Hexenyl Salicylate, Aurantiol, Vertofix Coeur, Floralozone,
Helional, Florol, Hedione. (The vetiver on the shelf is Fraterworks Vetiver
Oil Java, CAS 8016-96-4, matched to the library's Vetiver EO (Haiti) row:
same plant, a different origin's grade.)

**The fit.** Effective thresholds were fitted to this session and the earlier
merged ranking together, as the least weighted change in log threshold that
puts every strip in the order the nose gave, with "louder" at least a quarter
bin (1.32x) and ties equal. Loudness on a strip is taken as concentration over
threshold. Cedarwood EO is held at its balance-test 2.01 ppb. Weights: 3 for a
blotter-session or bench figure, 1 for a compiled one, 0.3 for an imputed or
unsourced one, so measured figures move least. The result goes in
`strength.csv`'s `strip_factor` column, which `strength.py` multiplies onto
the blotter multiplier, so the session numbers stay as measured.

**What moved most.** Floralozone 1 -> 45 ppb (ranked below Vertofix neat),
Helional 10 -> 59, Cis-3-Hexenyl Salicylate 257 (imputed) -> 20, Aurantiol
100 -> 26, Lemonile 5 -> 1.5, Hedione 73 -> 183, Galaxolide 12 -> 53. Ethyl
Maltol (to 0.8) and Styrallyl Acetate (to 0.9) move about 100x because the
earlier ranking had Ethyl Maltol at 1% above Galaxolide at 50%, and a linear
reading of concentration turns that into potency; their hand ceilings in the
engine still bound them.

**What it does not settle.** Labdanum Resinoid is not in the library. Its
strip was dropped, and the library's Labdanum Absolute, which the shelf row
had been matched to, is no longer marked as owned (`predict_formula.oil_form`
now refuses to match two different forms of one plant). A strip ranking gives
order, not size; a step between neighbours may be larger than a quarter bin.

*Strip ranking supplied by the perfumer, 2026-11-07.*

**Additions to the formula session (later on 2026-11-07).** Galaxolide (50% in
IPM) between Florol and Hedione; Paradisamide about level with Vertofix Coeur
and Floralozone (no gap forced to either); Verdox HC between Cedarwood and
Cis-3-Hexenyl Salicylate; Patchouli (Dark Indonesia, CAS 8014-09-3, matched to
the library's Patchouli EO (Javanese)) between Cedarwood and the labdanum, so
below Lemonile. Refitted from the pre-strip thresholds with unsourced naturals
weighted 0.3: Patchouli 30 -> 1.5 ppb, Verdox HC 100 -> 14, Paradisamide
14 -> 32, Florol 100 -> 82; the rest moved under 1.3x.

**Ethyl Maltol as a dilution pair, and Tetrahydro Linalool (later on
2026-11-07).** Strips, quietest first: Hedione ~ Ethyl Maltol 1% (level, maybe
a little above) < Tetrahydro Linalool < Galaxolide 50% < Ethyl Maltol 10% <
Florol. Ten times more Ethyl Maltol moved it only a few places, which is the
usual compressive growth of loudness with concentration. Under any
monotonic loudness curve of concentration over threshold, it also means
Hedione's threshold is at least 10x Florol's, where the column had 2.3x. The
refit puts Hedione at 615 ppb (blotter-weighted, so it moved least per unit
but carried the stretch), Florol 47, Ethyl Maltol 6.1, Tetrahydro Linalool
106, and the materials between them about 1.6x stronger.

The earlier ranking had Ethyl Maltol 1% above Galaxolide 50%; per the
engine's hand-ceiling note that was a bottle comparison, where headspace
favours the volatile material. Today's strips put it the other way, so the
bottle result is kept on record but left out of the strip fit; Ethyl Maltol's
previous 0.8 ppb came from it.
