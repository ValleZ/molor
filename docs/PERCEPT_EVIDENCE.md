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
  `build_naturals`, `descriptor_list`, `eval_mixtures`,
  `export_simplify_data`, `export_viewer_data`, `mud`,
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

### Attempt 4: retrain with pos_weight (`percepts_fp4.json`)

The ablations had rejected `pos_weight` on AUC/geometry ("drop pos_weight:
+0.079 Spearman, -0.008 AUC"), but they never measured the predicted
*distribution*, which is what decides whether a rare facet has any carrier at
all. Retrained with `train_fp_v3.py --pos-weight` (same recipe otherwise, 5
folds x 5 ensemble, ~2 min on GPU):

| | fp3 (shipped) | fp4 (pos_weight) |
|---|---|---|
| descriptors with >=5 carriers above 0.7 | **25 / 138 (18%)** | **129 / 138 (93%)** |
| sandalwood / orris / peach / rummy / clean / tobacco | 0 / 0 / 0 / 0 / 0 / 0 | **10 / 18 / 37 / 38 / 37 / 33** |
| powdery / dry / leafy / fresh | 0 / 1 / 0 / 3 | **81 / 70 / 32 / 100** |
| out-of-fold mean AUC | 0.8672 | **0.8688** |
| label-grounded similarity (Spearman) | 0.4999 | 0.3582 |

**This solves the supply problem outright** — every facet now has materials to
build a formula from, and AUC does not suffer. The cost is the geometry the
ablation predicted.

Downstream (7-brief A/B on an fp4 scratch database): sandalwood 0.700 -> 0.863
with the facet voiced at 0.41 and landing 0 -> 2/5; lilac holds 4/5; citrus
holds 2/5. But cedar & amber 3/6 -> 1/6, cinnamon 5/6 -> 3/6, garden soil
3/6 -> 2/6.

Re-fitting the engine on fp4 (`_RIDGE_W` 0.15/0.10/0.05/0.02/0.01 x
`_SEED_CARRIER` 0.6/0.45) does **not** recover them: the best setting keeps
**2 of 7** briefs at their shipped landing+fit (ridge 0.10: lilac 4/5,
sandalwood 2/5; ridge 0.05: sandalwood 3/5 but cinnamon 1/6). `cedar` never
recovers at any setting.

So the conclusion is stronger than "the engine was tuned to the compression":
**the shipped sample briefs were themselves selected against the fp3 matrix.**
A better matrix makes different materials win, and the five samples we chose
from fp3 no longer land. Adopting fp4 is therefore a product decision, not a
data refresh: it means re-selecting the samples against fp4, re-tuning the
engine, and re-running the whole suite against the new database.

## Publishing the atlas and the simplifier

Neither is a JSON file on the site: both are **self-contained HTML** with the
payload inlined, so exporting `viewer_data.json` / `simplify_data.json` never
changes what a visitor sees. Two builders publish them, and both live upstream:

```powershell
build_naturals      --percepts artifacts/percepts_fp4.json   # oils have no structure
scripts/export_simplify_data.py --percepts artifacts/percepts_fp4.json
export_viewer_data  --percepts artifacts/percepts_fp4.json
build_simplify.py simplify_data.json C:/Valle/dev/valkon.us/molor/simplify/index.html --mixpanel <token>
build_viewer.py   viewer_data.json   C:/Valle/dev/valkon.us/molor/index.html          --mixpanel <token>
```

The `--mixpanel` token is the one value a builder cannot derive; pass nothing and
the published page silently ships with analytics stripped.

Two rules learned the hard way:

* **Recompose the oils on the same model.** Oils have no molecular structure, so
their vectors are composed from their constituents; leaving them on the old model
makes 47 of 474 materials inconsistent with the model the database advertises.
* **No atlas trimming (perfumer's ruling).** `build_naturals` used to hide an oil
whose vector collapsed onto a single constituent — Orange EO at 0.9998 to
D-Limonene — which kept 13 of 47 off the map. `--max-nearest-cosine` now defaults
to 1.01 so every oil earns a point; the trim survives as an explicit flag.

## Status

Script: `scripts/merge_percept_evidence.py`. Candidate artifacts:

| file | rule | verdict |
|---|---|---|
| `artifacts/percepts_merged.json` | labels, `--present p95` | rejected (see above) |
| `artifacts/percepts_merged90.json` | labels, `--present 0.9` | superseded |
| **`artifacts/percepts_merged_cascade.json`** | panel → label → model | **candidate** |
| matching `.manifest.json` each | provenance, hashes, per-facet counts | |

`--no-graded` rebuilds the labels-only variant for comparison.

### ADOPTED: percept v4 (fp4) is the active model

The evidence-merge candidates above stay rejected. What shipped is the
**retrain**: `artifacts/percepts_fp4.json` (fingerprint MLP, pos_weight, 5x
ensemble) is now the default in `scripts/export_simplify_data.py`, so the whole
stack follows it — export -> `artifacts/simplify_data.json` ->
`perfumer/tools/build_db.py` -> `materials.db`.

Switching models is now a flag, not an edit: pass a different `--percepts` and
re-run the export + build. `build_db.py` derives `percept_model` / `percept_auc`
from that file's meta, so provenance can never again hardcode a model name.
Reverting to fp3 is the same two commands with
`--percepts artifacts/percepts_fp3.json` (the old database is kept at
`perfumer/materials.db.fp3`).

What it took to ship, all measured:

* the five demo samples were **re-selected against fp4** (the old set was
  chosen on fp3 and no longer landed): lilac 0.966 5/5, musk 0.928 4/5, garden
  soil 0.905 2/6, cinnamon 0.877 4/6, orange blossom 0.859 2/5;
* the engine needed **no** re-tune: ridge 0.10 already scores 5/5 on that set
  (0.15 also 5/5, 0.05 4/5, 0.02 3/5);
* `percept.build_target_vector`'s floor had to stop being capped at the
  quietest ask — fp4 lifts the library mean above `logit(0.5)` on two axes, so
  the cap was replacing the background target with the ask;
* `tools/fetch_cas.py`, `fetch_ifra.py` and `fetch_longevity.py` must run
  **after** `build_db.py` — a bare rebuild wipes CAS/IFRA/substantivity;
* two fp3-era test assumptions were replaced by data-independent invariants
  (a two-material 50/50 accord is now possible; the scale assertion recomputed
  a formula that the model's sd profile decides).

Suite: **88 passed** against the fp4 database.

## fp5 without `pos_weight`: REJECTED by the perfumer (2026-11-09)

**Verdict: fp4 stays. Dropping or weakening `pos_weight` is not to be tried
again on the strength of label-based numbers.**

The candidate was fp3's recipe (no `pos_weight`) on today's 703 molecules. On
every number below it beat fp4, and the perfumer's nose overruled them on the
material vectors themselves:

| material | fp4 | candidate | perfumer |
|---|---|---|---|
| Damascone Beta | 24 words above 0.5: berry 0.98, tobacco 0.98, tea 0.98, plum 0.97, black currant 0.97 | 6 above 0.5: fruity 0.93, floral 0.79, woody 0.70, berry 0.62, plum 0.32 | the rich profile is right: tea and black currant are there, berry and fruit for sure, dry plum somewhat; tobacco about 0.25 if at all |
| Cassifix (no label row) | woody 0.97, cedar 0.96, amber 0.92 | woody 0.93, cedar 0.33, amber 0.25 | woody plus something special that is called amber; fp4 closer |
| Phenethyl Alcohol | 15 above 0.5, with hyacinth 0.95, balsamic 0.88, geranium 0.71 | floral 0.93, rose 0.84, sweet 0.66 | a complex smell of many flowers; fp4 closer |

Why the numbers pointed the wrong way:

1. **Materials really are that rich.** A label row lists seven or so words and
   each is a real facet. Without `pos_weight` the loss is dominated by the
   zeros, so the model learns to mention a rare word faintly even where the
   row has it (only 24% of labelled cells reach 0.7). That is under-calling,
   and fp4's "over-prediction" was mostly the correction of it. The panel
   agrees in hindsight: it rates 10.3% of cells as present, the candidate's
   mean prediction is 0.044 and fp4's 0.155. The truth lies between, and the
   candidate is further from it.
2. **Label support is circular.** It rewards picking materials whose row holds
   the asked word, and a model without `pos_weight` mostly repeats its
   training rows. It cannot tell a model that copies labels from one that
   rightly reads a facet into a material whose row lacks the word (Cassifix
   has no row at all, and fp4's reading of it was the better one).
3. **The pick tables were judged by name, not by smell.** "Cedryl Acetate for
   a cedar brief" looks right on paper to someone who has not smelled the
   alternatives.

What survives: fp4 does overstate some cells (tobacco 0.98 on Damascone Beta
against the perfumer's 0.25), and the graded targets of to-do items 5 and 6
are the way to address that, trained **with** `pos_weight`. Fit still cannot
compare two models. And the only gate that counts is the perfumer reading a
material's vector against the material.

The measurements, kept for the record:

Retrained at five strengths of `pos_weight` (the per-label weight raised to a
power: 0 = the candidate, 1 = fp4), and run through eight briefs at eight
materials from the whole library:

| `pos_weight` power | 0 | 0.25 | 0.5 | 0.75 | 1 (fp4) |
|---|---|---|---|---|---|
| out-of-fold AUC | 0.867 | 0.867 | 0.871 | 0.870 | 0.869 |
| similarity vs label overlap | **0.503** | 0.440 | 0.417 | 0.387 | 0.356 |
| mean prediction (label rate 0.051) | 0.044 | 0.066 | 0.091 | 0.120 | 0.155 |
| engine's own fit, mean of 8 briefs | 0.526 | 0.633 | 0.763 | 0.839 | 0.871 |
| label support for the asked facets | **0.66** | 0.62 | 0.62 | 0.52 | 0.48 |
| label support for the hero facet | **0.84** | 0.73 | 0.76 | 0.62 | 0.56 |

*Label support* is the share of the smell that comes from picks whose own
GS-LF row uses the asked word, among the picks that have a row (asked facets
weighted by their ask). It does not come from the model under test, though a
model trained without `pos_weight` follows its training labels closely, so
part of this is the model repeating them.

The engine's fit moves the other way: **fit is the model grading itself and
cannot compare two models.** The gates above (fit, facets landing) all had
this flaw. Label support turned out to have its own (see the verdict).

What the picks look like (judged by name only; not smelled):

| brief | fp4 | no `pos_weight` |
|---|---|---|
| cedar and amber | Cassifix, Hedione, Cycloverdyl Propionate, Boisambrene Forte, Bornafix, 2-Ethylhexyl Salicylate, Florex, Trisamber | Vertofix Coeur, Cinnamyl Cinnamate, Cedryl Acetate, Caryophyllene Oxide, Isobornyl Acetate, Cedrenyl Acetate, Ambroxan, Amboryl Acetate |
| peach | Phenethyl Alcohol, Apritone, Benzyl Acetate, Gamma Nonalactone, Delta Decalactone, Fruitate, Gamma Octalactone, Blackcurrant mercaptan | Apritone, Jasminlactone, Nectaryl, Lactojasmone, Gamma Decalactone, Delta Decalactone, Gamma Nonalactone, Delta Undecalactone |
| lilac and cold cream | Isobutavan, Phytol, Benzyl Alcohol, Apritone, Geranyl Acetone, Bourgeonal, Starfleur 40, Cis Jasmone | Veratraldehyde, Hedione, Lactojasmone, Helional, Canthoxal, Vanillin Acetate, Canthoxalia, Anisyl Alcohol |
| berry | Apritone, Fructone, Rhubofix, Dimethyl Benzyl Carbinyl Acetate, Ethyl 2-methyl butyrate, Styrallyl Acetate, Grape Butyrate, Ethyl Butyrate | Fructone, Benzyl Isobutyrate, Dihydroactinidiolide, Ethyl 2-methyl butyrate, Ethyl Isobutyrate, Methyl Isobutyrate, Damascone Alpha, Damascone Beta |

### What the Dravnieks panel says about labels and the model

127 of the 138 panel molecules have a GS-LF row, giving 10,414 cells where a
panel rating, a label and an out-of-fold prediction can be compared. "Present"
is a rating of 9.8 or more, the median rating of a labelled cell.

| predictor of "present" (cross-validated by molecule) | AUC |
|---|---|
| label alone | 0.629 |
| out-of-fold model alone (no `pos_weight`) | 0.793 |
| label and model together | 0.802 |

- A label of 1 is worth about 2.4x in odds; the model's out-of-fold opinion is
  worth more. A missing word is weak evidence of absence, as the raise-only
  rule assumed, and a present word is weaker evidence than a flat 0.9.
- Half of the labelled cells rate under 10 on the panel; 7.5% of the
  unlabelled ones rate 10 or more.
- **Position in the label row carries almost no information.** The first word
  of `descriptors` rates 15.7 on average, the sixth and later 13.2. The merged
  file's order is not prominence.

A panel-calibrated matrix was built from this (value = probability of
"present" given label, out-of-fold prediction and how common the descriptor
is; labelled molecules use their out-of-fold prediction, the rest the model's).
It makes the result nearly independent of `pos_weight` (label support 0.67 to
0.69, hero 0.83 to 0.88) and beats the plain model by little (0.69 / 0.87
against 0.66 / 0.84). Not worth its machinery yet.

The scripts for all of the above were one-off and are not in the repository,
and neither is the training change (an exponent on the weight in
`train_fp_v3.py`'s `fit`, and a dump of the out-of-fold predictions).

### To do

In order. Each item says what to do and how to tell whether it worked.

1. ~~Decide fp5 by nose.~~ Done 2026-11-09: rejected, see the verdict.
2. ~~Ship fp5 = fp3's recipe.~~ Dropped with it.
3. **A gate the perfumer can run in five minutes.** Neither fit nor label
   support can judge a model. For any new candidate, print the top words of
   ten materials the perfumer knows well (Damascone Beta, Cassifix, Phenethyl
   Alcohol, Hedione, Iso E Super and five more of their choosing), old model
   beside new, and let the nose decide before anything else is measured.
4. **The perfumer's own yes/no answers: as a check only, never as training.**
   One nose is not the consensus the labels describe (Damascone Beta is
   labelled plum and does not smell of plum to the perfumer), so a personal
   "no" must not overwrite a label. If collected at all, the answers are a
   private test set: which model disagrees with this nose less.
5. **More label sources for the same molecules: checked 2026-11-09, worth
   doing.** The pyrfume archive holds all four, and votes across sources grade
   the labels. On the Dravnieks panel molecules:

   | GS-LF label | votes from the 4 extra sources | cells | mean panel rating | rated 10 or more |
   |---|---|---|---|---|
   | 1 | 0 | 335 | 12.3 | 42% |
   | 1 | 1 | 175 | 17.1 | 58% |
   | 1 | 2 or more | 78 | 23.7 | 64% |
   | 0 | 0 | 8754 | 2.9 | 7% |
   | 0 | 1 | 161 | 7.3 | 28% |
   | 0 | 2 or more | 9 | 11.5 | 44% |

   Library coverage (of 699 keyed molecules): Arctander 355, IFRA glossary
   328, Sigma 258, Flavornet 203. They add only 8 of the 69 molecules that
   have no GS-LF row, so this grades labels and does not fill gaps.
   - The check matched words by name plus a short synonym list (66 of
     Arctander's 76 words map, 105 of IFRA's 184, 78 of Sigma's 114, 75 of
     Flavornet's 195). Write a proper crosswalk per source, as for Dravnieks.
   - Build a soft training target per cell from label and votes, with values
     taken from the panel table above (for example 0.4 / 0.6 / 0.65 for a
     label with 0 / 1 / 2+ votes, 0.3 for one vote without a label).
   - Train on it **with** `pos_weight` (fp4's recipe) and put the result
     through the gate of item 3. The aim is fp4's richness with the overstated
     cells graded down (tobacco on Damascone Beta), not fewer facets.
6. **Descriptor order: checked 2026-11-09, the first word counts.** The merged
   file's order is uninformative, but the raw sources keep theirs. Mean panel
   rating by position: Good Scents 21.3 / 15.9 / 16.4 / 14.4 (first, second,
   third, later), IFRA glossary 21.3 / 14.6 / 12.2. Only the first position
   stands out. Add "first-listed in Good Scents or IFRA" as one more vote in
   the target of item 5; do not grade the later positions.
7. **Labels for the oils and the 69 unlabelled molecules.** They have no row,
   and the oils are composed from constituents.
   - Take each one's supplier description (Fraterworks pages are already
     fetched by `fetch_fraterworks.py`; The Good Scents has oils too).
   - Turn the prose into vocabulary words with the same mapper that turns a
     brief into facets (`llm.py` in the perfumer repo).
   - The perfumer checks ten by hand. If they hold, write them as label rows
     and treat them like any other.
   - For oils, compare with the composed vector. Where they disagree, the
     description wins for what the oil smells of, and the disagreement is a
     finding about the composition, in this order of likelihood:
     1. the oil is composed by mass share (`build_naturals.py`), while the
        engine mixes by share of the smell. A potent trace (damascenone in
        rose, pyrazines in galbanum) is nearly absent by mass and leads by
        smell. Compose oils with the engine's own loudness rule first, then
        compare again;
     2. a constituent is missing from the stated composition or from the
        library;
     3. only then is the description treated as the odd one out.
   - Keep the composed vector for gradation and raise the described facets on
     it, as the label merge does for molecules.
8. **Ensemble disagreement.** Save the standard deviation over the five
   models beside the mean. Try scoring a material at mean minus half its
   spread, and see whether label support rises on the unlabelled picks (about
   one per sheet today).
9. **Panel calibration, later.** Revisit once items 4 to 7 have added
   evidence: the calibration is the natural place to combine label, vote
   count, model and spread into one value, and the panel is the only graded
   truth to fit it on.
