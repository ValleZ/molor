# Soft targets for the percept model: report, second run (2026-11-10)

Task: `docs/TASK_SOFT_TARGETS.md`. This is the second run; the first, of the same day, was reviewed, found wrong in places and redone (see "What changed from the first run"). No default changed, `percepts_fp4.json` and `naturals.json` untouched, nothing exported, `C:\Valle\dev\perfumer` not touched. **No recommendation to adopt is made; only the perfumer decides, by nose.** Every number is *measured*; nothing is estimated.

## Plainly

- **No candidate meets the goal.** A and B (the task's loss) keep tobacco on Damascone Beta at 0.94 and 0.96 and have fewer words per material than fp4 (10.3 and 12.7 above 0.5 against 15.7; other seeds of fp4 give 16.1). C (a third loss, added after A and B, that settles at the target) brings tobacco to 0.50 but leaves 4.5 words above 0.5 per material and 40 of 138 descriptors with five carriers above 0.7. The task defines fewer facets as failure.
- **The votes cannot tell tobacco from the words the perfumer confirmed on Damascone Beta.** Tobacco, berry, black currant and fruity are all labelled with no source vote and not first-listed, so every vote-based target treats them alike. The words the sources do raise are plum and floral (one source each, and first-listed) and rose (one source), and the perfumer finds plum only "somewhat". In A and B the confirmed fruity falls more than tobacco (tobacco / fruity 1.05 in fp4, 1.48 in A, 1.28 in B).
- **Soft targets move levels, not separation.** Out of fold, fp4 already predicts labelled cells with 2+ votes 0.174 above those with none; A +0.154, B +0.172, C +0.177.
- **A and B still create words nothing supports**: 86 (A) and 172 (B) cells with no label and no vote rose by more than 0.3; other seeds give 43, C gives 6. Musk on Gamma Undecalactone (0.05 to 0.65 / 0.75) and cherry on Vanillin (0.30 to 0.70 / 0.76) are marked **new** in the gate sheet for the nose.
- AUC and similarity rho are in the table and nothing is concluded from them.

## What changed from the first run

- Crosswalks: Arctander `gassy` removed (36 training molecules, 1 labelled gassy: a different smell; it had put gassy 0.43 on Phenethyl Alcohol), Flavornet `gasoline`->solvent removed (7 molecules, none labelled), Flavornet `tart lime`->citrus removed (lime is unmapped elsewhere). Every mapping is now checked against the labels in `data/percept/crosswalk_audit.csv`.
- `build_source_crosswalks.py` loads the sources itself (it read a pickle only a temp script wrote).
- Step 4 is compared like for like (the first report's "same cell count" was false).
- Targets separate rows no source covers (2730 of 4983 rows, 10322 of 24312 labelled cells): there "0 votes" is no information.
- `train_fp_v3.py` gained `--oof-out` (out-of-fold dump) and `--soft-weight cell` (candidate C). Without `--soft-targets` the path is unchanged: the control equals fp4 exactly.
- The per-class table is out of fold on all 4983 rows, and the seed-shift run is a column in the tables.

## Deliverables

- crosswalks `data/percept/crosswalk_{arctander,ifra,sigma,flavornet,goodscents}_to_vocab.csv`, unmapped words `data/percept/unmapped_<source>.csv` (`scripts/build_source_crosswalks.py`)
- votes `data/percept/gslf_votes.csv` and the mapping check `data/percept/crosswalk_audit.csv` (`scripts/build_soft_targets.py`, loader `scripts/soft_target_sources.py`). The targets `gslf_soft_targets.csv` (A) and `gslf_soft_targets_B.csv` (B) are rebuilt by the same script and are not in git
- candidates `artifacts/percepts_fp6a.json` (A), `percepts_fp6b.json` (B), `percepts_fp6c.json` (C: B's targets, `--soft-weight cell`)
- `scripts/train_fp_v3.py`: new options `--soft-targets`, `--soft-weight`, `--oof-out`, `--seed-offset`; with none of them given the output is fp4, cell for cell
- `scripts/compare_soft_targets.py` prints the tables of steps 6 and 7 below

## Step 0: control

`train_fp_v3.py --pos-weight` reproduces `percepts_fp4.json` exactly (max and mean |diff| 0.0, AUC 0.8688, rho +0.3557), also after this run's changes to the script. Training is deterministic, so the usable floor is fp4's recipe with every seed shifted by 7000 (`--seed-offset 7000`): mean |diff| 0.028 (under the 0.05 stop line), p95 0.119, p99 0.195, max 0.425.

## Step 1: sources

Matching is by flat InChIKey only. Arctander through `stimuli.new_CID`, IFRA and Flavornet by CID, Good Scents through `stimuli.CID`, Sigma's positive stimulus ids (negative ones are CAS-only mixtures and are skipped).

| source | molecules in file | with a word | match GS-LF rows | match library (of 699 keyed) | distinct words | words mapped | earlier name+synonym |
|---|---|---|---|---|---|---|---|
| arctander | 2751 | 2488 | 1365 | 346 | 76 | 67 | 66 |
| ifra | 1060 | 1026 | 891 | 328 | 184 | 119 | 105 |
| sigma | 867 | 857 | 822 | 258 | 116 | 88 | 78 |
| flavornet | 716 | 653 | 619 | 203 | 195 | 120 | 75 |
| goodscents (first-listed words only) | 4565 | 4077 | 3943 | 625 | 335 | 129 | - |

Union of the four: 2253 of 4983 training rows, 538 of 699 library molecules. The earlier Arctander library count was 355 against 346 here; not chased.

## Step 2: unmapped words (molecules in the file)

- arctander (9): gassy 68, tarry 28, lilac 25, rooty 23, mossy 19, aromatic 18, narcissus 17, mimosa 15, pepper 15
- ifra (65): gourmand 50, food like 24, aromatic 17, marine 15, soft 12, ylang 9, patchouli 6, galbanum 6, mossy 5, watery 5, peppery 5, mandarin 5
- sigma (28): peanut 8, walnut 7, pepper 7, maple 6, jam 5, carnation 5, plastic 4, lime 4, horseradish 4, blossom 3, gardenia 3, mossy 3
- flavornet (75): alkane 20, turpentine 12, rancid 10, gasoline 8, plastic 7, rubber 6, tallow 5, resin 4, walnut 4, cotton candy 4, chemical 3, caraway 3
- goodscents (206 first-listed): bland 90, mustard 14, mild 12, rhubarb 9, mentholic 9, plastic 9, peppery 9, ambergris 9, marine 9, greasy 8, mandarin 8, fried 8

Judgement calls kept, for the perfumer to check: `acid`/`acidic`->sour, `neroli`->orangeflower, `iris`->orris, `vanillin`->vanilla, `honeydew`/`cantaloupe`->melon, `blueberry`->berry, `licorice`->anisic, `mothball`->camphoreous, `ethereal`->ethereal in all sources. Share of voted training molecules that carry the label, by source: Good Scents first-listed 99%, Flavornet 79%, IFRA 76%, Arctander 72%, Sigma 57%. Lowest kept mappings (5+ molecules): Sigma grapefruit 0 of 9, Flavornet malt 0 of 5, Flavornet soap 2 of 10, Sigma anise 5 of 21, Arctander berry 9 of 35. Same word on both sides, so kept; a low share can also mean a missing label.

## Step 3: votes

votes = sources applying the word (0 to 4) plus one if first-listed in Good Scents or IFRA (one extra vote when either holds). Label 1 by votes 0..5: 15884, 6170, 1692, 460, 94, 12. Label 0 by votes 0..3: 660459, 2650, 215, 18.

What the count hides:
- 4529 first-listed votes: 3696 from Good Scents alone, 689 IFRA alone, 144 both. GS-LF is built from Good Scents, so that vote says "prominent", not "a second source agrees".
- 1060 of the 2258 labelled cells with 2+ votes are one source plus first-listed; 2875 of the 6170 with one vote are first-listed only.
- 10322 labelled cells sit in rows none of the four sources covers.

## Step 4: votes against the Dravnieks panel

138 panel molecules, 127 in GS-LF, 116 of them covered by a source; 82 mapped words. Rating = mean over concentrations, max over the panel descriptors mapping to a word; present = 10 or more. The 2026-11-09 table was the 116 covered molecules with source votes only (9512 cells), so this is the like-for-like comparison:

| GS-LF label | source votes | cells | mean rating | present | 2026-11-09: cells / mean / present |
|---|---|---|---|---|---|
| 1 | 0 | 309 | 11.4 | 39% | 335 / 12.3 / 42% |
| 1 | 1 | 177 | 16.7 | 58% | 175 / 17.1 / 58% |
| 1 | 2+ | 102 | 23.6 | 66% | 78 / 23.7 / 64% |
| 0 | 0 | 8742 | 2.8 | 7% | 8754 / 2.9 / 7% |
| 0 | 1 | 170 | 8.1 | 31% | 161 / 7.3 / 28% |
| 0 | 2+ | 12 | 10.5 | 42% | 9 / 11.5 / 44% |

The ordering holds. Labelled cells by source vote and first-listed: no source, not first 34% (284 cells); no source, first 67% (61); source, not first 53% (192); source and first 78% (87). First-listed is worth about as much as a source vote.

## Step 5: targets

Target = the panel's present rate in the class, rounded to 0.05; B = A / 0.75.

| class | panel cells | present | A | B | training cells |
|---|---|---|---|---|---|
| label 1, covered, 0 votes | 256 | 33% | 0.35 | 0.47 | 7220 |
| label 1, covered, 1 vote | 185 | 55% | 0.55 | 0.73 | 4512 |
| label 1, covered, 2+ votes | 147 | 71% | 0.70 | 0.93 | 2258 |
| label 1, not covered, not first-listed | 476 | 42% | 0.40 | 0.53 | 8664 |
| label 1, not covered, first-listed | 148 | 74% | 0.75 | 1.00 | 1658 |
| label 0, 0 votes | 9608 | 7% | 0.00 | 0.00 | 660459 |
| label 0, 1 vote | 165 | 31% | 0.30 | 0.40 | 2650 |
| label 0, 2+ votes | 17 | 41% | 0.40 | 0.53 | 233 |

For rows not covered the sources are unknown, so the rate is taken over all labelled panel cells with that first-listed value (my departure from the task's single 0-vote class). Label 0 with 2+ votes rests on 17 panel cells.

## Steps 6 and 7

All three use fp4's recipe (`--pos-weight`, 5x ensemble, 5 folds); pos_weight, early stopping and every AUC use the binary labels. A and B use the task's loss, which settles at p = w t / (w t + 1 - t): 0.96 for t = 0.35 at tobacco's weight 47.9, and equally 0.96 for an unlabelled cell with one vote (t = 0.3) at weight 50. C weights the whole cell where t > 0 (weight w) and settles at p = t; on binary targets it is fp4's loss. C is the "third loss" the task allows after A and B; pos_weight is kept.

### Aggregate numbers (measured)

| | fp4 | A | B | C | fp4, other seeds |
|---|---|---|---|---|---|
| out-of-fold mean AUC | 0.8688 | 0.8676 | 0.8678 | 0.8620 | 0.8694 |
| label-grounded similarity rho | +0.3557 | +0.3452 | +0.3454 | +0.3500 | +0.3550 |
| mean prediction (703 x 138) | 0.1546 | 0.1258 | 0.1407 | 0.1125 | 0.1578 |
| descriptors with >= 5 library materials above 0.7 | 133 of 138 | 126 of 138 | 129 of 138 | 40 of 138 | 132 of 138 |
| descriptors with >= 5 library materials above 0.5 | 137 of 138 | 132 of 138 | 135 of 138 | 99 of 138 | 136 of 138 |
| mean number of words above 0.5 per material | 15.70 | 10.32 | 12.73 | 4.50 | 16.10 |
| mean number of words above 0.3 per material | 25.35 | 19.30 | 22.49 | 14.97 | 25.96 |
| materials with >= 10 words above 0.5 | 512 | 355 | 442 | 34 | 531 |
| cells above 0.9 | 2759 | 1058 | 1612 | 41 | 2845 |

### Cells moved against fp4 (measured)

Floor: fp4's recipe with every seed shifted by 7000 (`--seed-offset 7000`); its cells differ from fp4 by mean 0.028, p95 0.119, p99 0.195, max 0.425.

| moved by more than | A up | A down | B up | B down | C up | C down | seeds up | seeds down |
|---|---|---|---|---|---|---|---|---|
| 0.028 | 7227 | 28221 | 10002 | 23481 | 16525 | 27158 | 15850 | 13160 |
| 0.119 | 1292 | 12140 | 2004 | 7118 | 883 | 16736 | 2864 | 1987 |
| 0.195 | 538 | 5178 | 876 | 1826 | 183 | 11783 | 595 | 376 |
| 0.300 | 243 | 872 | 373 | 129 | 33 | 6472 | 52 | 27 |

Cells that rose by more than 0.3, by what the training data says of them:

| | A | B | C | seeds |
|---|---|---|---|---|
| labelled | 0 | 0 | 0 | 2 |
| not labelled, has a vote | 157 | 194 | 25 | 3 |
| not labelled, no vote | 86 | 172 | 6 | 43 |
| molecule has no GS-LF row | 0 | 7 | 2 | 4 |

### Mean prediction per target class, out of fold (measured)

All 4983 training rows x 138 words; every prediction is made by models that did not train on the row. Target A / B is what the class was trained toward (C uses B's).

| class | cells | target A | target B | fp4 | A | B | C |
|---|---|---|---|---|---|---|---|
| label 1, covered, 0 votes | 7220 | 0.35 | 0.47 | 0.584 | 0.454 | 0.507 | 0.346 |
| label 1, covered, 1 votes | 4512 | 0.55 | 0.73 | 0.656 | 0.519 | 0.582 | 0.428 |
| label 1, covered, 2+ votes | 2258 | 0.70 | 0.93 | 0.758 | 0.608 | 0.679 | 0.523 |
| label 1, not covered, not first-listed | 8664 | 0.40 | 0.53 | 0.634 | 0.488 | 0.544 | 0.383 |
| label 1, not covered, first-listed | 1658 | 0.75 | 1.00 | 0.631 | 0.514 | 0.566 | 0.448 |
| label 0, 0 votes | 660459 | 0.00 | 0.00 | 0.134 | 0.102 | 0.117 | 0.094 |
| label 0, 1 votes | 2650 | 0.30 | 0.40 | 0.442 | 0.357 | 0.406 | 0.293 |
| label 0, 2+ votes | 233 | 0.40 | 0.53 | 0.598 | 0.478 | 0.542 | 0.408 |

Labelled, covered cells: mean prediction with 2+ votes minus with 0 votes: fp4 +0.174, A +0.154, B +0.172, C +0.177.

### The same, by how rare the word is (out of fold, measured)

pos_weight of the word = unlabelled rows / labelled rows, capped at 50; 73 of 138 words sit at the cap.


Labelled cells with no vote and not first-listed:

| pos_weight | cells | fp4 | A | B | C |
|---|---|---|---|---|---|
| under 5 | 3062 | 0.624 | 0.393 | 0.479 | 0.416 |
| 5 to 15 | 4105 | 0.648 | 0.492 | 0.552 | 0.392 |
| 15 to 35 | 3495 | 0.653 | 0.542 | 0.591 | 0.387 |
| 35 to 50 | 1811 | 0.632 | 0.537 | 0.576 | 0.358 |
| at the cap 50 | 3411 | 0.502 | 0.416 | 0.451 | 0.273 |

Labelled cells with 2+ votes:

| pos_weight | cells | fp4 | A | B | C |
|---|---|---|---|---|---|
| under 5 | 977 | 0.740 | 0.524 | 0.622 | 0.528 |
| 5 to 15 | 590 | 0.767 | 0.635 | 0.698 | 0.519 |
| 15 to 35 | 366 | 0.783 | 0.697 | 0.742 | 0.533 |
| 35 to 50 | 150 | 0.791 | 0.737 | 0.770 | 0.544 |
| at the cap 50 | 175 | 0.748 | 0.687 | 0.722 | 0.464 |

Unlabelled cells with a vote:

| pos_weight | cells | fp4 | A | B | C |
|---|---|---|---|---|---|
| under 5 | 521 | 0.499 | 0.343 | 0.415 | 0.355 |
| 5 to 15 | 750 | 0.508 | 0.391 | 0.447 | 0.333 |
| 15 to 35 | 723 | 0.460 | 0.388 | 0.434 | 0.299 |
| 35 to 50 | 329 | 0.371 | 0.349 | 0.385 | 0.253 |
| at the cap 50 | 560 | 0.383 | 0.341 | 0.374 | 0.243 |

### Gate sheet: every word above 0.5 in any of the models

Values are measured. `data` is what the training data says of the cell: L = GS-LF label, a digit = source votes, F = first-listed, `-` = nothing. **new** marks a word with no label and no vote that rose by more than 0.3 over fp4 in some candidate.


#### Damascone Beta (perfumer: berry, fruity, tea, black currant present; dry plum somewhat; tobacco about 0.25 if at all)
GS-LF row: berry, black currant, floral, fruity, honey, mint, plum, rose, tobacco, woody. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| berry | L | 0.98 | 0.91 | 0.94 | 0.49 |
| tobacco | L | 0.98 | 0.94 | 0.96 | 0.50 |
| tea | - | 0.98 | 0.91 | 0.94 | 0.59 |
| plum | L1F | 0.97 | 0.95 | 0.97 | 0.87 |
| black currant | L | 0.97 | 0.87 | 0.92 | 0.47 |
| fruity | L | 0.93 | 0.64 | 0.75 | 0.64 |
| floral | L1F | 0.91 | 0.66 | 0.80 | 0.61 |
| rose | L1 | 0.90 | 0.77 | 0.86 | 0.60 |
| tropical | - | 0.90 | 0.71 | 0.80 | 0.49 |
| woody | L | 0.90 | 0.59 | 0.69 | 0.38 |
| mint | L | 0.88 | 0.57 | 0.68 | 0.31 |
| natural | - | 0.86 | 0.67 | 0.76 | 0.33 |
| honey | L | 0.86 | 0.73 | 0.79 | 0.42 |
| raspberry | - | 0.81 | 0.77 | 0.83 | 0.39 |
| dry | - | 0.80 | 0.55 | 0.60 | 0.41 |
| herbal | - | 0.74 | 0.58 | 0.66 | 0.49 |
| apple | 1 | 0.70 | 0.69 | 0.74 | 0.44 |
| sweet | - | 0.69 | 0.46 | 0.58 | 0.47 |
| apricot | - | 0.67 | 0.60 | 0.64 | 0.46 |
| powdery | - | 0.64 | 0.45 | 0.48 | 0.31 |
| spicy | - | 0.57 | 0.32 | 0.36 | 0.32 |
| earthy | - | 0.52 | 0.28 | 0.31 | 0.25 |
| hay | - | 0.51 | 0.23 | 0.29 | 0.15 |
| green | - | 0.51 | 0.32 | 0.36 | 0.35 |
| grape | - | 0.44 | 0.43 | 0.53 | 0.31 |
| *words above 0.5* | | 24 | 18 | 20 | 5 |

#### Cassifix (perfumer: woody plus amber)
No GS-LF row: nothing in the targets for this molecule.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| woody |  | 0.97 | 0.89 | 0.94 | 0.71 |
| cedar |  | 0.96 | 0.82 | 0.89 | 0.56 |
| amber |  | 0.92 | 0.74 | 0.81 | 0.53 |
| sandalwood |  | 0.75 | 0.57 | 0.68 | 0.18 |
| dry |  | 0.71 | 0.52 | 0.58 | 0.47 |
| spicy |  | 0.64 | 0.26 | 0.38 | 0.25 |
| musk |  | 0.57 | 0.26 | 0.39 | 0.19 |
| berry |  | 0.52 | 0.46 | 0.49 | 0.30 |
| *words above 0.5* | | 8 | 5 | 5 | 3 |

#### Phenethyl Alcohol (perfumer: a complex smell of many flowers)
GS-LF row: dry, floral, fresh, honey, rose, sweet. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| rose | L3 | 0.99 | 0.97 | 0.99 | 0.84 |
| floral | L | 0.98 | 0.88 | 0.93 | 0.78 |
| honey | L3 | 0.97 | 0.88 | 0.92 | 0.70 |
| hyacinth | 1 | 0.95 | 0.94 | 0.95 | 0.58 |
| balsamic | - | 0.88 | 0.74 | 0.77 | 0.52 |
| geranium | - | 0.71 | 0.46 | 0.61 | 0.34 |
| green | 1 | 0.68 | 0.47 | 0.58 | 0.45 |
| sweet | LF | 0.68 | 0.44 | 0.54 | 0.44 |
| jasmin | - | 0.66 | 0.70 | 0.71 | 0.52 |
| dry | L | 0.63 | 0.39 | 0.45 | 0.31 |
| leafy | - | 0.57 | 0.41 | 0.47 | 0.31 |
| waxy | - | 0.56 | 0.29 | 0.36 | 0.25 |
| metallic | - | 0.56 | 0.40 | 0.48 | 0.31 |
| fresh | L | 0.55 | 0.36 | 0.41 | 0.28 |
| cinnamon | - | 0.52 | 0.37 | 0.39 | 0.26 |
| musty | - | 0.46 | 0.49 | 0.57 | 0.40 |
| earthy | 1 | 0.44 | 0.52 | 0.60 | 0.34 |
| lily | - | 0.41 | 0.52 | 0.57 | 0.28 |
| *words above 0.5* | | 15 | 8 | 12 | 6 |

#### Hedione
GS-LF row: floral, fruity, green, jasmin, lactonic, natural, oily, tropical. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| jasmin | L4 | 0.99 | 0.99 | 0.99 | 0.90 |
| natural | LF | 0.97 | 0.96 | 0.97 | 0.83 |
| lactonic | L | 0.90 | 0.78 | 0.79 | 0.39 |
| floral | L2F | 0.85 | 0.67 | 0.81 | 0.69 |
| fruity | L1 | 0.85 | 0.47 | 0.63 | 0.52 |
| oily | L | 0.82 | 0.49 | 0.54 | 0.28 |
| tropical | L | 0.74 | 0.43 | 0.52 | 0.32 |
| green | L | 0.70 | 0.36 | 0.47 | 0.36 |
| waxy | - | 0.65 | 0.32 | 0.39 | 0.28 |
| peach | - | 0.63 | 0.45 | 0.55 | 0.21 |
| fresh | 1 | 0.62 | 0.54 | 0.63 | 0.43 |
| herbal | - | 0.60 | 0.49 | 0.54 | 0.37 |
| *words above 0.5* | | 12 | 5 | 10 | 4 |

#### Iso E Super
GS-LF row: amber, cedar, dry, ketonic, phenolic, woody. In none of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| woody | L | 0.98 | 0.87 | 0.93 | 0.58 |
| cedar | L | 0.97 | 0.90 | 0.94 | 0.50 |
| dry | LF | 0.93 | 0.90 | 0.92 | 0.85 |
| amber | L | 0.90 | 0.82 | 0.86 | 0.52 |
| ketonic | L | 0.89 | 0.80 | 0.84 | 0.47 |
| camphoreous | - | 0.85 | 0.68 | 0.72 | 0.58 |
| phenolic | L | 0.82 | 0.50 | 0.61 | 0.34 |
| tobacco | - | 0.80 | 0.61 | 0.67 | 0.36 |
| spicy | - | 0.63 | 0.32 | 0.43 | 0.28 |
| musk | - | 0.62 | 0.49 | 0.55 | 0.35 |
| powdery | - | 0.61 | 0.48 | 0.53 | 0.36 |
| mint | - | 0.59 | 0.38 | 0.46 | 0.37 |
| earthy | - | 0.57 | 0.30 | 0.39 | 0.27 |
| leathery | - | 0.56 | 0.34 | 0.44 | 0.20 |
| herbal | - | 0.51 | 0.28 | 0.38 | 0.31 |
| *words above 0.5* | | 15 | 7 | 10 | 5 |

#### Linalool
GS-LF row: citrus, floral, fresh, green, lavender, natural, rose, sweet, woody. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| citrus | L1 | 0.98 | 0.91 | 0.95 | 0.75 |
| lavender | L2 | 0.95 | 0.92 | 0.94 | 0.77 |
| natural | LF | 0.92 | 0.84 | 0.88 | 0.62 |
| woody | L2 | 0.90 | 0.72 | 0.79 | 0.59 |
| floral | L4F | 0.89 | 0.81 | 0.88 | 0.69 |
| lemon | - | 0.86 | 0.64 | 0.71 | 0.42 |
| fresh | L1 | 0.86 | 0.73 | 0.79 | 0.57 |
| rose | L | 0.83 | 0.75 | 0.83 | 0.44 |
| mint | - | 0.83 | 0.59 | 0.68 | 0.42 |
| lily | 1 | 0.83 | 0.81 | 0.85 | 0.36 |
| herbal | - | 0.77 | 0.67 | 0.75 | 0.62 |
| bergamot | - | 0.74 | 0.70 | 0.75 | 0.36 |
| sweet | LF | 0.72 | 0.54 | 0.67 | 0.53 |
| weedy | - | 0.70 | 0.55 | 0.62 | 0.39 |
| bitter | - | 0.56 | 0.38 | 0.46 | 0.23 |
| warm | - | 0.49 | 0.46 | 0.54 | 0.30 |
| berry | - | 0.47 | 0.55 | 0.62 | 0.37 |
| camphoreous | - | 0.44 | 0.49 | 0.56 | 0.40 |
| *words above 0.5* | | 15 | 15 | 17 | 8 |

#### Vanillin
GS-LF row: chocolate, creamy, sweet, vanilla. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| vanilla | L4 | 1.00 | 0.99 | 1.00 | 0.81 |
| creamy | L1 | 0.99 | 0.96 | 0.98 | 0.69 |
| phenolic | - | 0.97 | 0.81 | 0.88 | 0.38 |
| chocolate | L1 | 0.94 | 0.91 | 0.93 | 0.61 |
| spicy | - | 0.94 | 0.70 | 0.80 | 0.49 |
| milky | - | 0.94 | 0.80 | 0.84 | 0.42 |
| sweet | L1F | 0.94 | 0.74 | 0.86 | 0.71 |
| powdery | 1 | 0.93 | 0.82 | 0.88 | 0.46 |
| caramellic | 1 | 0.87 | 0.88 | 0.91 | 0.54 |
| clove | - | 0.87 | 0.68 | 0.75 | 0.43 |
| smoky | - | 0.86 | 0.65 | 0.74 | 0.45 |
| warm | - | 0.85 | 0.66 | 0.74 | 0.36 |
| dairy | - | 0.80 | 0.51 | 0.58 | 0.27 |
| almond | - | 0.79 | 0.68 | 0.75 | 0.31 |
| balsamic | - | 0.72 | 0.73 | 0.78 | 0.46 |
| burnt | - | 0.71 | 0.36 | 0.43 | 0.26 |
| sharp | - | 0.66 | 0.44 | 0.54 | 0.26 |
| bitter | - | 0.65 | 0.46 | 0.50 | 0.30 |
| pungent | - | 0.64 | 0.43 | 0.49 | 0.31 |
| cinnamon | - | 0.60 | 0.38 | 0.47 | 0.23 |
| medicinal | - | 0.55 | 0.37 | 0.46 | 0.31 |
| woody | - | 0.55 | 0.25 | 0.31 | 0.26 |
| aromatic | - | 0.53 | 0.38 | 0.37 | 0.33 |
| hay | - | 0.50 | 0.39 | 0.46 | 0.30 |
| cherry | **new** | 0.30 | 0.70 | 0.76 | 0.30 |
| *words above 0.5* | | 24 | 16 | 18 | 5 |

#### Ambroxan
GS-LF row: amber, cedar, dry, green, musk, pine, sweet, tea, woody. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| amber | L2F | 1.00 | 0.99 | 0.99 | 0.91 |
| woody | L2 | 1.00 | 0.97 | 0.99 | 0.77 |
| cedar | L | 1.00 | 0.98 | 0.99 | 0.55 |
| pine | L | 0.98 | 0.92 | 0.95 | 0.46 |
| dry | LF | 0.97 | 0.96 | 0.97 | 0.84 |
| musk | L | 0.93 | 0.80 | 0.85 | 0.33 |
| tea | L | 0.93 | 0.87 | 0.90 | 0.41 |
| balsamic | - | 0.85 | 0.58 | 0.66 | 0.47 |
| camphoreous | - | 0.83 | 0.62 | 0.72 | 0.40 |
| tobacco | - | 0.81 | 0.62 | 0.68 | 0.22 |
| earthy | - | 0.70 | 0.33 | 0.43 | 0.29 |
| herbal | 1 | 0.68 | 0.56 | 0.70 | 0.37 |
| spicy | - | 0.67 | 0.39 | 0.51 | 0.33 |
| vetiver | - | 0.59 | 0.48 | 0.49 | 0.24 |
| natural | - | 0.58 | 0.44 | 0.44 | 0.26 |
| mint | - | 0.55 | 0.36 | 0.49 | 0.17 |
| powdery | - | 0.54 | 0.38 | 0.40 | 0.27 |
| sandalwood | - | 0.51 | 0.34 | 0.37 | 0.20 |
| *words above 0.5* | | 18 | 11 | 12 | 4 |

#### Gamma Undecalactone
GS-LF row: aldehydic, apricot, coconut, creamy, fatty, fruity, ketonic, lactonic, peach, sweet. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| coconut | L | 1.00 | 0.95 | 0.97 | 0.65 |
| lactonic | L | 0.99 | 0.91 | 0.94 | 0.49 |
| creamy | L | 0.99 | 0.92 | 0.95 | 0.62 |
| fatty | L | 0.99 | 0.86 | 0.91 | 0.68 |
| peach | L1 | 0.99 | 0.96 | 0.97 | 0.71 |
| buttery | - | 0.98 | 0.89 | 0.92 | 0.57 |
| milky | - | 0.97 | 0.87 | 0.91 | 0.51 |
| dairy | - | 0.96 | 0.81 | 0.86 | 0.44 |
| waxy | - | 0.93 | 0.73 | 0.81 | 0.50 |
| coumarinic | - | 0.86 | 0.63 | 0.73 | 0.26 |
| aldehydic | L | 0.83 | 0.65 | 0.72 | 0.39 |
| oily | - | 0.82 | 0.67 | 0.73 | 0.52 |
| metallic | - | 0.78 | 0.51 | 0.59 | 0.35 |
| apricot | L1 | 0.77 | 0.75 | 0.78 | 0.51 |
| ketonic | LF | 0.74 | 0.63 | 0.72 | 0.43 |
| sweet | L1F | 0.66 | 0.50 | 0.60 | 0.51 |
| nutty | - | 0.65 | 0.42 | 0.52 | 0.39 |
| fruity | L | 0.61 | 0.39 | 0.50 | 0.47 |
| jasmin | - | 0.58 | 0.52 | 0.57 | 0.34 |
| celery | - | 0.58 | 0.42 | 0.43 | 0.25 |
| cheesy | - | 0.57 | 0.43 | 0.48 | 0.31 |
| tropical | - | 0.56 | 0.24 | 0.31 | 0.23 |
| fresh | - | 0.55 | 0.22 | 0.27 | 0.25 |
| caramellic | - | 0.52 | 0.28 | 0.31 | 0.25 |
| musty | 1 | 0.43 | 0.45 | 0.55 | 0.34 |
| soapy | **new** | 0.26 | 0.52 | 0.58 | 0.30 |
| musk | **new** | 0.05 | 0.65 | 0.75 | 0.31 |
| *words above 0.5* | | 24 | 19 | 21 | 10 |

#### Galaxolide
GS-LF row: floral, musk, sweet. Covered by at least one of the four sources.

| word | data | fp4 | A | B | C |
|---|---|---|---|---|---|
| musk | L2F | 0.99 | 0.98 | 0.99 | 0.94 |
| amber | - | 0.82 | 0.64 | 0.74 | 0.50 |
| woody | 1 | 0.49 | 0.43 | 0.52 | 0.27 |
| *words above 0.5* | | 2 | 2 | 3 | 1 |

### Damascone Beta, the words the perfumer placed (measured)

| | fp4 | A | B | C |
|---|---|---|---|---|
| tobacco | 0.98 | 0.94 | 0.96 | 0.50 |
| plum | 0.97 | 0.95 | 0.97 | 0.87 |
| tea | 0.98 | 0.91 | 0.94 | 0.59 |
| black currant | 0.97 | 0.87 | 0.92 | 0.47 |
| berry | 0.98 | 0.91 | 0.94 | 0.49 |
| fruity | 0.93 | 0.64 | 0.75 | 0.64 |
| tobacco's rank among the 138 words | 2 | 2 | 2 | 6 |
| tobacco divided by fruity | 1.05 | 1.48 | 1.28 | 0.78 |

## To reproduce

```
.venv-pyrfume\Scripts\python.exe scriptsuild_source_crosswalks.py
.venv-pyrfume\Scripts\python.exe scriptsuild_soft_targets.py
set T=artifacts/temp
.venv\Scripts\python.exe scripts	rain_fp_v3.py --pos-weight --out %T%/fp4_repro.json --oof-out %T%/oof_fp4.npz
.venv\Scripts\python.exe scripts	rain_fp_v3.py --pos-weight --seed-offset 7000 --out %T%/fp4_seedshift.json
.venv\Scripts\python.exe scripts	rain_fp_v3.py --pos-weight --soft-targets data/percept/gslf_soft_targets.csv --out artifacts/percepts_fp6a.json --oof-out %T%/oof_fp6a.npz
.venv\Scripts\python.exe scripts	rain_fp_v3.py --pos-weight --soft-targets data/percept/gslf_soft_targets_B.csv --out artifacts/percepts_fp6b.json --oof-out %T%/oof_fp6b.npz
.venv\Scripts\python.exe scripts	rain_fp_v3.py --pos-weight --soft-targets data/percept/gslf_soft_targets_B.csv --soft-weight cell --out artifacts/percepts_fp6c.json --oof-out %T%/oof_fp6c.npz
.venv\Scripts\python.exe scripts\compare_soft_targets.py
```

About two minutes per training run on the GPU. The intermediate files land in `artifacts/temp/`, which git ignores.

## Not done or unsure

- The perfumer has not seen any of this; the gate sheet above is for that.
- C was run only on B's targets. On A's it cannot exceed 0.75 by construction.
- Nothing between B and C was tried (for example cell weighting on labelled cells only). Any such variant still cannot separate tobacco from berry on Damascone Beta, because the votes do not.
- Crosswalk judgements are mine and unreviewed. The first-listed vote was counted once when in either source; the other reading was not run.
- fp4's count of descriptors with five materials above 0.7 is 133 here; the 129 in the task is from an earlier build of fp4 (rho 0.3582 then, 0.3557 now).
- The 69 library molecules without a GS-LF row get nothing from the targets.
- `.idea/caches/deviceStreaming.xml` was modified before this work; not mine.
