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
