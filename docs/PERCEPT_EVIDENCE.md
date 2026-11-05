# Fusing human labels into the percept model

## Why

The library's `percept_vector` is a **model prediction**, not a measurement:
`GS_LF_baseline_gcn.pth` (a graph-conv net over molecular structure) trained on
`curated_GS_LF_merged_4983.csv` — 4983 molecules × **138 binary** odour tags.
Naturals have no structure, so their vectors are composed from their
constituents' (`build_naturals.py`).

That model is **ranked correctly but badly calibrated for rare labels**. For the
262 library materials that have a human GS-LF row, comparing the label with the
prediction on that same material:

| facet | materials with label=1 | model ≥0.5 on them | mean pred (label=1) | mean pred (label=0) |
|---|---|---|---|---|
| sandalwood | 2 | 1 | 0.50 | 0.00 |
| orris | 8 | **0** | 0.11 | 0.01 |
| peach | 13 | **0** | 0.17 | 0.01 |
| rummy | 10 | **0** | 0.12 | 0.01 |
| cucumber | 6 | 2 | 0.29 | 0.01 |
| ozone | 5 | 0 | 0.14 | 0.01 |
| tobacco | 11 | 0 | 0.07 | 0.01 |
| cedar | 7 | 0 | 0.09 | 0.00 |
| clean | 15 | 0 | 0.04 | 0.02 |
| vanilla | 10 | 6 | 0.48 | 0.02 |
| floral | 126 | 59 | 0.45 | 0.15 |
| sweet | 166 | 62 | 0.44 | 0.28 |

The model still *separates* the classes (label=1 ≫ label=0 almost everywhere —
`clean` is the exception), but it refuses to say **how much**: it essentially
never emits ≥0.5 for a rare label. Base rates explain it: `sandalwood` is 0.7%
of the training set, `floral` 22%; Spearman(base rate, mean prediction) = 0.69.

Downstream this is fatal, not cosmetic. The engine mixes by
`sigmoid(Σ wᵢ·logit(pᵢ))` with a 40% per-material cap, so a facet needs **many
materials above ~0.7** to be voiced at all — that is a "cluster". `floral` has
92 such materials; `orris`, `peach`, `rummy`, `ozone`, `cucumber` have **0**.
That is why briefs in those families can't be built, and it is *not* a defect in
the search or the mixture rule.

## The rule

**Labels first, per cell, raise-only.**

Three sources in order of precision, applied **per cell**:

| precedence | source | direction | coverage |
|---|---|---|---|
| 1 | **Dravnieks 1985** panel rating ÷ 100 | **raises or lowers** | 43 / 474 materials, 80 crosswalk columns |
| 2 | GS-LF `label == 1` | raise to **0.9** | 262 / 474 materials, 133 of 138 terms |
| 3 | GS-LF `label == 0` | model value kept | — |
| 4 | no row | model value kept | oils, captives |

Stage 1 applies **only positive panel ratings**: a blank or 0 in Dravnieks means
the word was not applied, not that the smell is absent. Where graded evidence
speaks for a cell, stage 2 skips it.

Stage 1 needs `data/percept/crosswalk_dravnieks_to_vocab.csv`, because the two
vocabularies barely intersect: Dravnieks speaks ASTM DS61 (`FRUITY,CITRUS`,
`LEMON`, `WOODY, RESINOUS`, `BURNT,SMOKY`) and matches only **37 of our 138**
terms by name, ~73 through the crosswalk. Absent from the ASTM list entirely:
`sandalwood`, `orris`, `ozone`, `clean`, `rummy`, `green`. Those facets are
label-only, which is why `sandalwood` reaches just 2 carriers.

Measured with the full cascade (43 materials graded, 262 labelled; 2064 cells
graded up, **672 graded down**, 1796 label-raised): `peach` 0→12 carriers,
`tobacco` 0→9, `cedar` 0→6, `cucumber` 0→6, `smoky` 0→2, `vanilla` 5→7,
`floral` 16→111, `woody` 14→56. The panel raises 32 `peach` cells but only 12
clear 0.7 — graded evidence is more conservative than a flat 0.9, which is the
precision it was added for.

Why raise-only: a GS-LF `0` means *the panellist did not use that word*, not
"intensity zero". With base rates of 0.7–2%, absence of the word is the normal
case even for materials that obviously carry the quality. Treating `0` as `0.0`
would delete true structure — Rhodinol's 0.77 fruity, Sandalore's 0.46 cedar —
and turn every matched vector into a binary vector, which is precisely the
"no gradation" flaw already documented for GS-LF (`build_percept_datasets.py`).

Why a fixed value and not the descriptor's own percentile: **P95 was tried
first and it fails**, which is worth recording because it is the intuitive
choice. The percentile is taken from the very distribution that is compressed,
so for a rare facet it is tiny (`sandalwood` P95 = 0.02, `orris` 0.04, `peach`
0.05, `rummy` 0.09): raising a labelled cell to it changes nothing and the
carrier count stays at 0. Measured (`--present p95`, 1395 cells raised):
`sandalwood` 0→0, `orris` 0→0, `peach` 0→0, `rummy` 0→0 carriers above 0.7,
while only the already-loud facets moved (`floral` 16→126). A label of 1 means
"this material does smell of X" — a semantic statement, not a quantile — so
the target is fixed at **0.9** (below the logit clamp at `1 - ODDS_EPS`, ≈13.8,
so one material cannot saturate a blend).

Measured with `--present 0.9` (2017 cells raised over 262 materials), carriers
above 0.7 before → after:

| facet | before | after |
|---|---|---|
| peach | 0 | **13** |
| tobacco | 0 | **11** |
| rummy | 0 | **10** |
| orris | 0 | **8** |
| cucumber | 0 | **6** |
| ozone / smoky | 0 | **5** |
| clean | 0 | **15** |
| vanilla | 5 | 11 |
| sandalwood | 0 | **2** (thin — only 2 labelled materials exist) |
| floral / green / woody | 16 / 19 / 14 | 126 / 106 / 72 |

The last row is the risk: loud facets gain far more than rare ones, so the
merged matrix is *more* dominated by floral/green/woody than before. Whether
that harms the working briefs is exactly what validation item 3 measures.

### Alternatives rejected

* **Per material instead of per cell** (matched → labels only): destroys
  gradation and manufactures ties; rejected above.
* **Zero out on `label == 0`**: treats absence of a word as absence of a smell.
  Deletes real structure and contradicts the base rates.
* **Hand-editing signature vectors** ("Sandalore should say sandalwood 0.9"):
  that is inventing sensory evidence. The labels *are* that evidence, with a
  provenance, and they already say what needs saying.
* **A global recalibration of all 138 descriptors**: redistributes the same
  sparse signal for the metric's benefit without adding evidence; it cannot fix
  a facet whose strongest carrier is real but unlabelled.

## Where it lives

The output is *a percept artifact for the library*, so it lives with the other
ones and keeps their schema — that is what makes it consumable by every
pipeline with no code change:

```
artifacts/percepts_merged.json        # same keys: checkpoint, descriptors,
                                      # molecules, probs   (+ "evidence" block)
artifacts/percepts_merged.manifest.json   # provenance, rule, counts, hashes
```

* Producers of percept files already write this shape (`run_percept.py`,
  `run_openpom.py`, `run_densesense.py`, `build_naturals.py`).
* Consumers already take `--percepts <path>`: `bench_simplify`,
  `build_naturals`, `descriptor_list`, `discriminating_blends`, `eval_mixtures`,
  `export_simplify_data`, `export_viewer_data`, `mud`, `olfactory_white`,
  `palette`, `percept_blind_spots`, `predict_formula`, `reverse_accord`,
  `rule_bakeoff`, `threshold_worklist`, `try_formula`; plus `ORV_PERCEPTS` in
  `formula_check`, `mixture`, `suggest_check`.
* The perfumer app reads it through `export_simplify_data.py --percepts`, which
  is what writes `materials.db`'s `percept_vector` column. **No schema change**,
  no new directory, no consumer edits — one flag.

Regenerate:

```bash
.venv-pyrfume/Scripts/python scripts/merge_percept_evidence.py \
    --percepts artifacts/percepts_expanded.json \
    --out artifacts/percepts_merged.json
# then, on the perfumer side:
python scripts/export_simplify_data.py --percepts artifacts/percepts_merged.json
```

## Validation (must all pass before the default changes)

1. **Agreement with human evidence does not regress.** `eval_similarity.py`
   (Spearman vs Jaccard overlap of human descriptor sets) and
   `notes_vs_models.py` (AUC over the perfumer's own notes) on merged vs
   unmerged. Raising rare labels should improve both; if it does not, the rule
   is wrong.
2. **The dead facets gain carriers.** Per facet, count materials above 0.7
   before/after. Target: `peach`, `orris`, `rummy`, `smoky`, `cucumber`,
   `tobacco`, `ozone` ≥ 5; `sandalwood` and `malty` will stay thin (2 and 0
   labelled materials) and that must be stated, not hidden.
3. **Working briefs do not regress.** The perfumer's five sample briefs are
   measured on fit and facet landing before/after; nothing may drop.
4. **Naturals and captives are unchanged.** They have no GS-LF row, so they keep
   model values — and they are where the model is weakest. The merge helps
   molecules, not oils. Say so in the manifest.

## Validation result (measured 2026-XX-XX) — REJECTED

The A/B was run the way the gate requires: build a scratch `materials.db` from
the merged artifact (`ab_evidence.py --build`, 266 materials updated, 208 oils
and captives untouched), then generate the five shipped sample briefs plus two
previously-impossible ones from both libraries.

| brief | shipped library | merged evidence |
|---|---|---|
| sandalwood (impossible before) | fit 0.700, sandalwood 0.10 | **fit 0.924, sandalwood 0.36** |
| peach (impossible before) | fit 0.897, peach 0.32 | fit 0.926, landing 1→2/5 |
| garden soil | fit 0.869, 3/6 landed | fit 0.971, 3/6 |
| cedar & amber | fit 0.894, 3/6 | fit 0.915, **1/6** |
| cinnamon & clove | fit 0.943, 5/6 | fit 0.874, **3/6, `clove` silent** |
| lilac & cold cream | fit 0.950, 4/5 | fit 0.896, **2/5** |
| bitter citrus peel | fit 0.890, citrus 0.61 | fit 0.694, **citrus 0.18** |

**It fails validation item 3** (working briefs must not regress), so the
shipped library is unchanged and no consumer default moves.

Two lessons worth keeping:

1. **Whole-matrix merging destroys contrast.** Raising 2017 label cells and
   2736 graded cells lifted `floral` from 16 to 111 carriers; the library became
   loud on everything, the fit flattened, and every working brief regressed.
2. **Filtering to the compressed facets is not enough.** Restricting the merge
   to descriptors whose mean prediction is below 0.06 (`--max-mean`, 119 of 133)
   left `floral`/`green`/`woody`/`sweet` untouched and recovered *some* briefs,
   but still broke others — `citrus` itself was never modified, yet it fell from
   0.61 to 0.18 because `orange`, `grapefruit` and `lemon` were, and the
   co-facets re-rank the candidate pool. The model's compression is
   **load-bearing for the current fits**: the engine's ridge, share caps and
   seed were all tuned against this matrix. Replacing the data without
   re-benchmarking the engine is not a drop-in change.

What would be needed to ship it: re-tune `_RIDGE_W`, `_MAX_SHARE` and the
specialist seed **against the merged matrix**, then re-run the whole sample
benchmark and the test suite — i.e. the merge is a data change that implies an
engine re-fit, not a data refresh. Not done.

Scratch build `materials_cascade.db` (perfumer repo root, untracked) is left
for inspection; it is not shipped and no service reads it.

### Correction: production is fp3, not the GCN baseline

`materials.db` records `percept_model = fingerprint MLP v3 (ablation-selected,
5x ensemble)`, built from `artifacts/percepts_fp3.json`. The first A/B merged
evidence into `percepts_expanded.json` (the GCN baseline), so it compared
"baseline + evidence" against an fp3 library — a confounded result. Every
measurement below after that point was redone on fp3.

fp3 is **more compressed than the baseline**, not less: of 138 descriptors,
`fresh` has 3 materials above 0.7, `powdery` 0, `dry` 1, `earthy` 1, `leafy` 0.
108 of 138 qualify as starved. That is why a "starved-only" filter barely
filters anything on this model.

### Attempt 3: monotone prior calibration on fp3 (`calibrate_percepts.py`)

Per-descriptor monotone transform (the m-th largest prediction maps to 0.72,
the max to 0.95, everything below scales proportionally), with m taken from the
label prevalence among matched library materials — so within-descriptor
ordering is preserved and a facet gets exactly as many carriers as the labels
warrant. 108 descriptors calibrated.

| brief | shipped fp3 | fp3h (hierarchy) | evidence merge | calibrate |
|---|---|---|---|---|
| sandalwood | 0.700, sw 0.10 | 0.825, sw 0.22 | 0.941, sw 0.44 | 0.937, sw 0.42 |
| peach | 0.897, 1/5 | 0.880, 3/5 | 0.890, 1/5 | 0.895, 1/5 |
| lilac | 0.950, 4/5 | 0.892, 3/5 | 0.937, 3/5 | 0.978, 3/5 |
| citrus | 0.890, 2/5 | 0.902, 1/5 | 0.797, 1/5 | 0.836, 3/5 |
| garden soil | 0.869, 3/6 | 0.869, 3/6 | 0.791, 3/6 | 0.962, 3/6 |
| cedar & amber | 0.894, 3/6 | 0.902, 2/6 | 0.884, 3/6 | 0.913, 2/6 |
| cinnamon & clove | 0.943, 5/6 | 0.900, 2/6 | 0.824, 2/6 | 0.884, 2/6 |

**All three data-side attempts fail the same gate: the working briefs lose
facets.** The conclusion is now well supported rather than assumed:

> The engine's parameters (`_RIDGE_W`, `_MAX_SHARE`, the specialist seed) were
tuned *against this matrix's compression*. No data-side change — more evidence,
hierarchy labels, or prior calibration — passes the sample gate on its own.
Making rare facets voiceable requires re-fitting the engine to the new matrix,
which means a full re-benchmark (five samples, the reference formula, the
identity round trip, the suite), not a swap.

The tools are all in place for that project: `merge_percept_evidence.py`
(evidence cascade), `calibrate_percepts.py` (prior calibration), `percepts_fp3h`
(hierarchy), `eval_fp_percept.py` / `eval_similarity.py` / `notes_vs_models.py`
for model-level scoring, and the seven-brief A/B harness for the engine level.

## Status

Script: `scripts/merge_percept_evidence.py`. Candidate artifacts:

| file | rule | verdict |
|---|---|---|
| `artifacts/percepts_merged.json` | labels, `--present p95` | rejected (see above) |
| `artifacts/percepts_merged90.json` | labels, `--present 0.9` | superseded |
| **`artifacts/percepts_merged_cascade.json`** | panel → label → model | **candidate** |
| matching `.manifest.json` each | provenance, hashes, per-facet counts | |

`--no-graded` rebuilds the labels-only variant for comparison.

**No consumer default has been changed.** The candidate is inert until a
`--percepts artifacts/percepts_merged90.json` flag is passed somewhere, and
`export_simplify_data.py` — the step that writes `materials.db` — has not been
run against it. Validation items 1–4 in this document are the gate.
