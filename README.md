# orvector

SMILES → predicted responsiveness score for every functional human olfactory
receptor, as a fixed-order vector.

Scores are continuous sigmoid outputs of the model's raw logits. They are not
binned, thresholded, or normalised, and they are not concentration-specific
activation strengths.

## Two things to look at

**[The receptor atlas](https://valkon.us/molor)** — 375 points: 352 raw
materials, 22 essential oils rebuilt from published GC compositions, and the
average of everything, laid out by predicted receptor response. Six layouts,
colouring by predicted descriptor or by the perfumer's own odour families,
and a per-material card with predicted odour, measured threshold where one
exists, and whether the percept model was trained on that molecule.

**[The formula simplifier](https://valkon.us/molor/simplify)** — paste a
formula, get the smallest blend of materials whose predicted character is
closest to it. Runs entirely in the page. Closeness is reported against what
a random blend of the same size achieves, against the nearest single
material, and against the floor the largest allowed blend reaches, because a
distance on its own means nothing. For 23 materials its doses now rest on a
blotter measurement rather than on a compiled threshold figure, and it names
which ones.

Both are single self-contained HTML files built from the library store by
`scripts/export_viewer_data.py` + `scripts/build_viewer.py` and
`scripts/export_simplify_data.py` + `scripts/build_simplify.py`. The deployed
copies live in a separate site repository; `--no-notes` withholds the
perfumer's written smelling impressions from any published build, and every
deploy is checked for that.

The rest of this document is the record of what was measured, what was
retracted, and what is still wrong. The corrections table immediately below
is the shortest honest summary of the whole project.

## Corrections

Several findings in this document were later overturned by their own
re-examination. Each section carries the correction inline; collected here so a
reader does not act on a retracted claim:

| original claim | status |
|---|---|
| bond features change the MPNN's answer | **false** — zeroing every edge feature changes its output by exactly 0.0 |
| isomer blindness is a 1-Weisfeiler-Leman readout limit | **false** — it is GRU saturation in that one checkpoint; the GCN separates the same isomers |
| the MPNN beats the GCN on chemical-class coherence, 0.617 vs 0.422 | **not comparable** — different material sets (302 vs 128); re-measured, 0.262 vs 0.271, indistinguishable |
| fingerprint v1 has the best similarity structure | **artefact** — measured against the training labels; on independent panels no trained model separates from another |
| the ablation selected v3 on similarity grounds | **unsupported** — the metric it was selected on does not predict independent panel data |
| predicted odour thresholds are usable for dosing | **withdrawn** — Spearman 0.14 against the perfumer's use levels in-domain, −0.17 outside it |
| patchouli and vetiver cannot be built from their GC compositions | **false** — a shelf limit, not a model limit; thirteen sesquiterpenoids added as reference molecules and vetiver now puts `vetiver` 3rd of 138 by lift |
| a blend's position on the atlas means something | **only under the right rule** — noisy-OR is a union, so it placed a 28-ingredient formula 62 from the cloud's centre where materials sit 14.6; the distance was measuring ingredient count |
| noisy-OR lets a trace ingredient through where a mean buries it | **false** — it scales by `w / w.max()`, so 3% of a four-material blend enters at ~9% strength; in 40,000 random blends it never once carried a descriptor the mean did not |
| the inventory's water-phase thresholds are measured values | **false** — the perfumer compiled them from published and web sources and approximated some from air-phase figures, so a value may be out by 10x or more. Every script that called them "measured" has been corrected. At that uncertainty a 10x error moves a five-material blend 0.4–0.7 in percept space, against a just-noticeable yardstick of 0.525, so threshold-derived doses are dominated by the input error rather than by the solver |
| Iso E Super's 0.5 ppb threshold is wrong and distorts every woody formula | **retracted** -- mine, argued over two messages and supported by none of the data. The reasoning was that nobody doses a 0.5 ppb material at 25%; the mixture's potency belongs to a minor isomer at about 5%, detectable at 5 pg/L in air, which makes the commercial mixture roughly 5,000x more potent than its own major isomer. You dose it at 25% *because* the bulk is weak. The figure stands, the engine's reading of those formulas was right, and the real gap was the missing volatility axis -- see `data/inventory/threshold_provenance.md` |
| one vector describes what a blend smells like | **incomplete** -- on a base-heavy formula it describes the drydown. A tomato-leaf accord containing five green materials read `cedar, amber, dry` because it is 44% Iso E Super; split by volatility its top tier reads `cortex, grassy, leafy, melon`. The volatility classes were in the inventory unused. `scripts/volatility_profile.py` |
| coverage by odorous weight says how much of a formula was understood | **misleading when the drops were dilutions** -- a coffee accord read 96% covered with all three of its coffee materials dropped, because they were 10% and 1% stocks worth 2 neat parts of 70. The page now names dropped dilutions and says the figure understates what is missing |
| a material whose judgements span two bins cannot be used | **false** -- the span of a set of numbers grows with its size, so that rule flagged the four materials a re-test had just resolved. Replaced by shrinkage against a measured per-judgement noise of 0.89 bins |
| Calone sticks out of blends because it is powerful | **false** -- measured at x0.24 strength, among the weakest per gram in the set. It sticks out because it is isolated (74th percentile), not because it is strong; the two had been conflated |
| musks do not cluster because OR5AN1's learned signal is weak, so "musk" was never in the training data | **false, and the opposite** — OR5AN1's logit spread is the 97th percentile of the panel, the three nitro musks are the top three materials in the library on it, and of all 386 receptors it ranks **first** on size-controlled musk preference. Deleting it from the panel changes every pairwise distance by 0.09% (Spearman +1.0000), so the global distance cannot see it. **But see the row below: this one overclaims too.** `scripts/musk_receptor.py` |
| upweighting OR5AN1 would therefore make musks cluster | **false, and backwards** — nitro musks sit at +4.4 z on it and macrocyclics at +1.8, so the axis that detects the class also separates its members; sweeping the weight to 10,000x moves those pairs from the 84th to the 94th percentile. A class detector is not a similarity axis. Against Keller's own MUSKY rating over 115 molecules OR5AN1 ranks 228 of 386, which does not confirm the identification either. What does hold: single receptors predict individual panel descriptors at median |rho| 0.38, an order of magnitude better than the whole 386-dim distance predicts similarity (+0.03) — but 50 different receptors are best for 146 descriptors, so importance is per question, not per receptor, and no single reweighting can use it. `scripts/receptor_importance.py` |
| a fixed palette of ~25 materials could reach any smell | **false for single molecules** — the reachable set is the palette's convex hull: 25 materials capture 95% of the space's energy and still reproduce only 45% of the library to cosine 0.90 |
| the centroid of percept space is the olfactory white point, the "mud" | **false, and the perfumer caught it on the bench** — the centroid is the *average* smell, not a flat one: sweet 0.45, fruity 0.44, floral 0.38, green 0.35, sd 0.075 across descriptors. It has character by construction. Their description was "bright and sharp, like bright light", which is a better name for it |

The pattern is worth stating: every number that looked good on its own
cross-validation failed when checked against something it had not been fitted
to. Bench observations caught four of these before any metric did.

The same pattern runs the other way once: the formula work below contains one
prediction that was stated in advance and then confirmed on a blotter, and it
is the only external validation here that no metric could have supplied.

## What this is built on

| Component | Pin |
|---|---|
| Upstream code | [`microsoft/olfaction`](https://github.com/microsoft/olfaction) @ `0b8bd710509d50310e8c45396301e22eebbc8f69` |
| Checkpoint | `m2or_MolOR_esm_650m_unweighted.pth`, md5 `98b3bcc6f0bae4c8faddb40ffb22c7d5` |
| Checkpoint source | [Zenodo 13765978](https://zenodo.org/records/13765978) |
| Protein encoder | ESM-2 650M (`esm2_t33_650M_UR50D`), layer 33 |
| Model | `MolORPredictor` (GCN + odorant–receptor cross-attention) |

The model architecture and the embedding routine are imported from the pinned
upstream checkout rather than reimplemented, so they cannot silently diverge
from the trained weights. `scripts/generate_OR_predictions_pseudogenes.py` is
used as a reference, not invoked.

## How it works, end to end

Everything below this section is a finding — something that was measured and
sometimes corrected. This section is the plumbing those findings sit on, in
order, with the widths of every array, because nothing else here explains it
from the start.

### Two models, and only one of them is trained here

This is the answer to "why are there checkpoints if you train a model
yourself". There are two models. They share a library of molecules and nothing
else.

|  | **receptor model** | **percept model** |
|---|---|---|
| the question it answers | does olfactory receptor OR1A1 respond to this molecule? | what does this molecule smell like? |
| where it comes from | a pretrained checkpoint, Zenodo 13765978, published with the MolOR paper | trained here, `scripts/train_fp_v3.py`, about five minutes on a GPU |
| what it was fitted on | M2OR, a database of receptor assays | GS-LF, 4983 molecules with human odour descriptors |
| input | one (molecule, receptor) pair | one molecule |
| output | **one** number: a responsiveness logit | **138** numbers: one per odour descriptor |
| used for | **where** a point sits on the atlas | **what colour** it is, and everything the simplifier does |

So the atlas is two models at once: position from receptors, colour from
percepts. They are never averaged into one vector, and the distances quoted in
either space mean different things. `artifacts/library/` holds the receptor
vectors; `artifacts/percepts_fp3.json` holds the percept vectors.

Nothing here retrains the receptor model. It is run in inference mode against
frozen weights, which is why the checkpoint's md5 is checked on every load: a
different checkpoint would not match the architecture, and a silently different
architecture would produce plausible numbers that mean nothing.

### What "encoder" means

A molecule has a variable number of atoms; a protein has a variable number of
residues. A neural network needs a fixed-size input. An encoder is the learned
function that closes that gap — variable-sized structured object in, fixed-size
vector out — and it is trained jointly with whatever consumes its output rather
than designed by hand. A fingerprint is the hand-designed alternative: fixed
rules, no training. Both appear in this project, the fingerprint in the percept
model and two learned encoders in the receptor model.

### Pipeline 1 — one SMILES to 386 receptor responses

`orvector predict` (`src/orvector/inference.py`). Widths are from
`src/orvector/config.py`, where each one is frozen with its reason.

1. **Parse.** RDKit reads the SMILES. Hydrogens are normalised, because
   chirality written in brackets (`[C@H]`) makes RDKit record that atom's
   hydrogens as explicit, which zeroes its implicit valence — and implicit
   valence is a feature block. Without this step the same molecule featurises
   differently depending only on how it was typed. The InChIKey is compared
   before and after, so the compound cannot be altered silently.
2. **Atom features.** `CanonicalAtomFeaturizer` gives each atom **74** numbers
   (element one-hot, degree, formal charge, hybridisation, aromaticity, total
   hydrogens). Molecules are padded to **98** atom slots — the largest molecule
   in GS-LF. This budget is not cosmetic: padded slots are zeroed only *after*
   `mol_norm`, and `LayerNorm` maps a zero vector to its own bias, so the pad
   leaks into the protein-side attention. Widening it moves the logits, which
   is why it is frozen and tested.
3. **Molecular encoder.** 74 per atom → **256** per atom. This is the GCN or
   the MPNN; see below.
4. **Receptor encoder.** A receptor enters the model as nothing but its
   amino-acid sequence. ESM-2 650M (`esm2_t33_650M_UR50D`, layer 33) turns one
   sequence into **443 × 1280** — one 1280-vector per residue position, padded
   to 443. This is the one genuinely expensive step, so it runs once for the
   whole panel and is cached as `artifacts/panel_embeddings.npy`, 386 × 443 ×
   1280 float32 ≈ 0.9 GB, memmapped and never loaded whole.
5. **Cross-attention.** Atoms attend over residues and residues over atoms
   (`OdorantReceptorCrossAttention`). The two sides are pooled and
   concatenated: 1280 + 256 = **1536**.
6. **Head.** An MLP, 1536 → 128 → **1**. One logit, meaning *this molecule
   against this receptor*.
7. **The panel.** Repeat for all **386** functional human receptors and stack
   the logits. That vector is what the atlas lays out.

#### Why a protein encoder at all

Because the model's unit of prediction is a *pair*, not a molecule. The obvious
alternative design is one output head per receptor — 386 outputs, molecule in,
386 numbers out, and no protein encoder anywhere. That architecture works and
is simpler.

What it cannot do is score a receptor that was not in its training set. Head
number 387 does not exist and cannot be created without retraining. Feeding the
receptor in as a sequence instead buys three things:

- **A receptor the model never saw can still be scored**, because the sequence
  is input rather than an index. This is the whole point: assay coverage is
  thin and uneven, and the 386-receptor human panel scored here is not the
  set the checkpoint was fitted on.
- **The panel is swappable.** Changing which receptors are scored is a change
  of input data, not of architecture — which is also why `panel.py` can
  recompute the functional/pseudogene split and assert it against upstream's.
- **Similarity between receptors is learned rather than assumed.** Two
  receptors with similar sequences get similar embeddings for free, so an assay
  on one is evidence about the other.

The cost is that a receptor is represented *only* by its sequence. No structure,
no binding pocket, no conformation. Which is the honest frame for the whole
receptor model: it is a sequence-conditioned statistical model of assay
outcomes, not physics. That claim is taken apart properly under "Why this is a
learned model and not a physics simulation".

#### The two molecular encoders, in one sentence each

Upstream ships two, as two separate Zenodo checkpoints, and this repository
keeps both stores so they can be compared.

- **GCN** (`m2or_MolOR_esm_650m_unweighted.pth`, 25 MB): each atom replaces its
  own feature vector with a normalised average of its neighbours', twice, 256
  wide, with residual connections. Bonds appear only as the adjacency used for
  the averaging — `edge_featurizer=None` — so the encoder is blind to bond
  order and to E/Z.
- **MPNN** (`m2or_MolOR_MPNN_esm_650m_unweighted.pth`, 60 MB): each bond
  carries a *message*, computed by a small network from that bond's own 13
  features (6 of which are stereo), and a shared GRU accumulates the messages
  over several passes. It is handed the bond information the GCN discards.

That difference ought to settle which is better, and does not. The MPNN
checkpoint's encoder is numerically saturated to the point that zeroing every
bond feature changes its output by exactly zero, and six attempts to separate
the two spaces perceptually came out mixed. The atlas therefore ships both and
defaults to the MPNN, for reasons given in full under "GCN vs MPNN".

### Pipeline 2 — 386 dimensions down to three

`src/orvector/project.py` and `build_model_block` in
`scripts/export_viewer_data.py`. The naive version of this step — PCA the raw
vectors to three axes — produces a picture of the wrong thing, because every
molecule correlates about 0.95 with the panel mean and that shared component is
essentially *how hard does this molecule hit receptors at all*. Projected raw,
the map draws potency and buries chemistry.

So:

1. **Logits, not sigmoid scores.** The scores are saturated near zero, so a
   distance computed on them is decided by a handful of receptors.
2. **Z-score each receptor** across the library, removing the per-receptor
   baseline so that broadly-tuned generalists stop dominating every distance.
3. **Optionally drop the first principal component** — the potency axis. This
   is the atlas's "potency removed" view; both are exported, neither is
   privileged.
4. **PCA to 20 components.** With 386 receptors and a few hundred molecules
   this is firmly the n < p regime, so a nonlinear step on the full width would
   be fitting noise.
5. **Euclidean distance matrix** over those 20 components. This matrix is what
   the viewer's "distance to" numbers come from, and its median is the
   `medianDistance` the page quotes against. (The simplifier's separate
   yardstick, 0.439, is a percept-space figure and not this one.)
6. **Metric MDS to 3 axes** on the precomputed distances (`n_init=6`,
   `max_iter=800`, fixed seed). MDS minimises the disagreement between the 3D
   distances and the 20-D ones directly, which is what we want; PCA maximises
   retained variance, which is not the same objective. Both are exported.
7. **Axis-align.** MDS returns an arbitrary orientation and these clouds are
   about three to one elongated, so an SVD rotates the widest spread onto x and
   the next onto y, with a fixed sign convention so repeated builds do not
   mirror. A rotation preserves every pairwise distance, so this costs nothing.
8. **Report the distortion.** Every projection carries a **Shepard rank
   correlation** between its own pairwise distances and the 20-D ones, and the
   page prints it. 3D reaches 0.99 against 0.98 for 2D on global distances, and
   the gap on *local* neighbourhoods is much larger — 0.70 against 0.51 on
   top-10 overlap — which is why the atlas is 3D. Each point also carries its
   own neighbour-preservation score, so a point that landed in the wrong
   neighbourhood can say so.

### Pipeline 3 — one SMILES to 138 descriptors

`scripts/train_fp_v3.py`. This model is small and entirely conventional; the
interesting part is what it is scored on, not how it is built.

- **Input: a fingerprint, not a learned encoder.** 2048-bit Morgan radius 2
  (ECFP4: a hash of every atom's circular neighbourhood out to two bonds) plus
  all 209 RDKit molecular descriptors, standardised — **2257** numbers.
- **Model:** an MLP, 2257 → 512 → 256 → 138, no batchnorm, no dropout.
- **Fitting:** 5-fold cross-validation for an honest out-of-fold score, then a
  refit on everything as an ensemble of 5 seeds. 180 epochs, cosine schedule.
- **Output:** 138 probabilities, one per GS-LF descriptor (`fruity`, `woody`,
  `black currant`, …).

Two things about reading that output. First, **always as lift against the
library mean, never raw**: base rates dominate, `floral` and `fruity` score on
almost everything, and the z-scored lift is the only readable signal — see
"Read lift against a null, never the raw values". Second, the model is scored
on *two* axes that pull against each other, per-descriptor ranking accuracy and
whether two molecules that smell alike land near each other; optimising either
one damages the other, which is the subject of "Predicting smell directly".

### Pipeline 4 — the simplifier

`scripts/export_simplify_data.py` builds a payload; `viewer/simplify.html`
does the arithmetic in the page with no server. Given a formula or a
description, it answers: what are the fewest materials that smell like this?

**The target.** A pasted formula's lines are resolved to library materials
(272 aliases, dilution and solvent parsing, oils matched separately), and the
target percept vector is the ingredient vectors mixed by weight. A pasted
*description* instead scores only the descriptors actually named, weighted —
a formula is matched on all 138, a description on the words it used.

**The pool — which materials may be proposed.** 435 entries, 39 of them
essential oils. Three exclusions, in descending order of how well they are
justified:

- **4 are never proposed at all**, because they are regulatory prohibitions or
  restrictions: Musk Ambrette, Musk Xylene, 2,6-Dibromophenol, Lilial. This
  list was once longer and included materials excluded on my judgement of
  whether they were dosable. That was wrong and they were put back; it is now
  only things that are actually prohibited.
- **34 are flagged "unverified"** and hidden behind a checkbox, because their
  provenance says they entered the library as constituent structures so an
  essential oil could be built from them — khusimol, the vetivones,
  patchoulol. This rests on a recorded fact about where the entry came from,
  not on a guess about availability.
- **Materials with no threshold** are excluded when dosing is on, because a
  material with no threshold cannot be converted to a mass. This is why the
  thresholds are imputed rather than left blank.

What is deliberately *not* in the pool logic is how commonly a material is
used. Three attempts to infer that — one perfumer's inventory, the formulas in
this repository, a supplier's catalogue — each confidently mislabelled obvious
staples, and the claim was withdrawn. See "Read lift against a null".

**The search.** For every k from 1 up, two candidate subsets are built: a
greedy-plus-swap search over the pool, and the plain heuristic of keeping the
k heaviest ingredients of the original. Whichever lands closer wins, which
makes the proposal no worse than the obvious thing by construction — the
search alone lost to the heuristic on 64% of oils. Weights are then fitted by
simplex-constrained least squares (unconstrained solve, then projected
gradient). The fidelity curve over all k is computed once and is forced
monotone, because a greedy search is not monotone in k and at one point nine
materials fitted worse than eight, which made the curve a lie.

**Thresholds, and the dose.** A detection threshold is the concentration at
which a material becomes smellable; materials differ over eight orders of
magnitude, so equal weights do not mean equal loudness.

- 125 of 435 materials have a **compiled** figure, assembled from published
  sources.
- The other 310 are **imputed**: k=5 nearest neighbours in the percept model's
  own 138-dimensional space, regressing log10(threshold). Leave-one-out median
  error 3.0x, 77% within a decade, against a spread of 1.29 log units.
  Better than nothing, and not much better — and because imputation
  regresses to the mean, a genuinely potent outlier comes out *over*-dosed.
- Every figure carries its tier, and the page says which it used. A number that
  will be put on a scale has to be able to say what it rests on.
- 23 materials additionally carry a **measured strength correction** from a
  blotter session, applied as `effective threshold = threshold / multiplier` —
  on top of an imputed figure as readily as a compiled one, since it is a
  statement about the material and not about the figure. See "Measuring
  strength instead of thresholds".

The weights the solver returns are shares of *odour activity*. Converting them
to mass multiplies by `threshold ^ 0.6`. The exponent is 1.0 for straight
odour-activity dosing; 0.6 is where one bench comparison preferred it, which is
a single observation and is why it is a named constant rather than a tuned
parameter. Then the sheet is scaled so parts sum to 100 and each material is
assigned a stock strength from the ladder 100 / 10 / 1 / 0.1%.

The honest limit on all of this: the dose error is larger than the thing being
optimised. That has its own section, and it is the reason the simplifier
reports an ordering rather than a weighing.

## Setup

Requires Python 3.10–3.12. **Python 3.13 will not work**: DGL publishes no
wheels for it. DGL also publishes no Windows CUDA build, so the graph model
runs on CPU; the GPU is still used for the one-time ESM-2 embedding pass.

```bash
# 1. Environment (uv shown; conda/venv equivalent is fine)
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch==2.3.0" "numpy==1.26.4"
uv pip install --python .venv/Scripts/python.exe \
  "dgl==2.2.1" "dgllife==0.3.2" "rdkit==2023.3.1" "fair-esm==2.0.0" \
  "pandas==2.0.3" "scipy==1.11.4" "scikit-learn==1.3.2" \
  "pyyaml" "psutil" "pydantic" "requests" "tqdm" "pytest"
uv pip install --python .venv/Scripts/python.exe --no-deps "torchdata==0.9.0"
uv pip install --python .venv/Scripts/python.exe --no-deps -e .

# 2. Upstream code at the pinned commit
git clone https://github.com/microsoft/olfaction vendor/olfaction
git -C vendor/olfaction checkout 0b8bd710509d50310e8c45396301e22eebbc8f69

# 3. Checkpoint (and the reference logits used by the parity test)
mkdir -p vendor/zenodo
curl -L -o vendor/zenodo/m2or_MolOR_esm_650m_unweighted.pth \
  "https://zenodo.org/api/records/13765978/files/m2or_MolOR_esm_650m_unweighted.pth/content"
curl -L -o vendor/zenodo/olfactory_subgenome_OR_logits.pt \
  "https://zenodo.org/api/records/13765978/files/olfactory_subgenome_OR_logits.pt/content"
```

### Version pins that are not arbitrary

* `torch==2.3.0` — DGL 2.2.1 bundles `graphbolt` DLLs for torch ≤ 2.3.0. 2.3.1
  fails to import.
* `torchdata==0.9.0` — DGL 2.2.1 imports `torchdata.datapipes`, removed in 0.10.
* `numpy<2` — upstream-era code.
* `pandas==2.0.3` — `Series.value_counts()` tie ordering defines the panel
  order. `prepare` verifies the recomputed order against upstream's shipped
  index files and fails loudly if it differs, so a bad pandas cannot silently
  reorder the panel.

## Usage

```bash
python -m orvector prepare --device cuda      # or --device cpu
python -m orvector predict \
  --smiles 'COC(=O)c1ccccc1' \
  --output methyl_benzoate.npz \
  --csv-out methyl_benzoate.csv \
  --device cpu
```

Multiple molecules:

```bash
python -m orvector predict --csv molecules.csv --output batch.npz --device cpu
```

`--csv` auto-detects a `smiles` / `SMILES` / `IsomericSMILES` column (override
with `--smiles-column`) and an optional `id` / `name` / `CID` column.

`python -m orvector info` prints the resolved config, artifact status and manifest.

### Output

`predict` writes an `.npz` containing:

* `scores` — `(n_molecules, n_receptors)` float32, sigmoid of the logits (the default vector)
* `logits` — `(n_molecules, n_receptors)` float32, raw model output
* `receptor_ids`, `genes`, `upstream_seq_index` — the frozen panel order
* `smiles_input`, `smiles_canonical`, `molecule_ids`
* `panel_hash`, `checkpoint_md5`, `model_config_hash`, `encoder_config_hash`,
  `upstream_commit`, `orvector_version`, `seq_pad_len`, `max_node_len`

`--csv-out` additionally writes one row per (molecule, receptor).

After `prepare`, prediction is fully offline — covered by a test that disables
socket access.

## The receptor panel

**386 receptors.** Derived from upstream's HORDE gene table (857 rows → 845
unique sequences) and upstream's own functional/pseudogene split: 386
functional, 459 pseudogenes. Pseudogenes and fragments are excluded.

Ordering is ascending `upstream_seq_index` (upstream's 845-sequence
`value_counts` order, filtered to functional entries). That keeps a direct
correspondence to the columns of upstream's pre-generated logits, which is what
makes the parity test possible. The order and sequence content are covered by
`panel_hash`, recorded in `artifacts/manifest.json`.

Two sequences are shared by multiple genes — `OR4F29`/`OR4F16`/`OR4F3` and
`OR2A42`/`OR2A1` are genuine identical-protein paralogs. One sequence is kept,
with the other symbols recorded in `gene_aliases`.

### One repaired entry

`OR2G6`'s `Conceptual Sequence` field in `genes.csv` is malformed:
`"|Strand=+|Length=MEETNNSS…"`. Upstream's cleaning (strip `trand=+`, then
`split("|")[1]`) reduces it to the single residue `"S"` — so upstream's own
HORDE run embedded a 1-residue "receptor" here. The real 316 aa protein is
present in the field and matches reviewed UniProt **Q5TZ20 (OR2G6_HUMAN)**
exactly, so it is recovered and marked in the `source` column.

Repairs are allow-listed in `panel.EXPECTED_REPAIRS` and verified at build
time: an unexpected repair, a missing expected one, or a length that disagrees
with UniProt all fail the build.

## Two frozen constants

Both are pinned in `config.py`, recorded in the manifest, and included in the
embedding cache key — but for different reasons, and the difference was
measured rather than assumed.

**`MAX_NODE_LEN = 98`** — atom slots, the max atom count over the 5862 GS-LF
molecules. **This one changes the numbers.** In
`OdorantReceptorCrossAttention` the padded atom slots are zeroed only *after*
`mol_norm`, and `LayerNorm` maps a zero vector to its own `bias`, so
`Linear(bias)` enters the protein-side attention as keys and values even though
`fixed_size_tensor2[node_mask == 0] = 0` drops them from the molecule
aggregation. Widening the budget shifts the logits;
`test_atom_slot_budget_changes_scores` asserts it.

A molecule with more than 98 heavy atoms is rejected with a clear error rather
than silently truncated.

**`SEQ_PAD_LEN = 443`** — the residue width every sequence is padded to.
Upstream takes `token_representations[1:-1]`, dropping BOS and the *batch's*
last token rather than slicing to each sequence's true length; a common width
is what makes that extraction coherent at all. Upstream's width is the max over
all 845 cleaned sequences (443), not the 357 the 386 functional ones alone
would give, and matching it keeps our embeddings directly comparable to theirs.

It is frozen because it *defines the cached artefact*, not because it moves the
scores. The symmetric leak does exist on the protein side, but it runs through
`torch.relu(scores)`, which is zero for the inputs tested — shortening the
padded tail leaves the logits unchanged. `test_residue_padding_tail_leaks`
records that as an `xfail` probe rather than a claim, so if residue padding
ever does become load-bearing the suite says so.

## Notable deviations from the upstream script

* **`mol2prot_dim=True`.** The checkpoint's `cross_attn` projections are
  `Linear(1280→1280)` and `Linear(256→1280)` with `linear1` of `(1, 1280)`,
  which only occurs on the `mol2prot=True` branch. Upstream's argparse defaults
  it to `False`; building with the default fails strict loading. `model.py`
  derives the architecture from the checkpoint's own tensor shapes and refuses
  to run if it disagrees with the frozen config.
* **The `if len(smiles) == 1: continue` guard is removed.** It exists to dodge
  batch-norm instability during *training*. The MLP head's `BatchNorm1d` uses
  stored running statistics under `eval()`, so a one-pair batch is well
  defined; keeping the guard would emit no predictions for a single molecule.
  `test_single_and_batched_agree` covers this.
* **No hard-coded `.cuda()`.** Device is explicit throughout.
* **Weights load with `strict=True`**; missing or unexpected parameters raise.
* **`torch.autograd.set_detect_anomaly(False)`** is restored after model
  construction — `MolORPredictor.__init__` switches that global flag on.
* Inference runs under `model.eval()` and `torch.inference_mode()`.

## Tests

```bash
./.venv/Scripts/python.exe -m pytest tests -v
```

The six acceptance criteria, plus guards on the assumptions above:

1. Finite score for every panel receptor; scores continuous, not binned or normalised.
2. Single vs batched inference agree; molecules unaffected by batch composition.
3. Repeated runs bit-identical; panel order and hash stable; hash matches manifest.
4. Invalid SMILES raise a clear, input-naming error; oversized molecules rejected.
5. A reference batch matches **upstream's inference path** — upstream's own
   `predict_OR_feat`, model and weights, driven one pair at a time — to within
   2e-4 of our batched loop. See the note below on the *published* logits.
6. Prediction works with all socket access disabled.

Also asserted: per-sequence embedding caching is valid because ESM-2 output is
invariant to batch grouping; padding length is load-bearing; the OR2G6 repair
matches UniProt.

## GCN vs MPNN: bond features reach the encoder and are then destroyed

> **This section's original claim was wrong and is corrected below.** It read
> "bond features change the answer". They do not reach the MPNN's output at
> all. Measured with `scripts/encoder_health.py`: zeroing *every* edge feature
> changes the encoder's output by exactly `0.0`. They do reach the NNConv layer
> — zeroing them moves *its* output by 2.4e4 — and are then annihilated.

Upstream ships two MolOR encoders. The GCN variant (the default checkpoint)
runs with `edge_featurizer=None` and so discards bond information entirely;
the MPNN variant is handed `CanonicalBondFeaturizer(self_loop=True)`, 13 dims,
of which 6 are bond stereo.

**The MPNN checkpoint's encoder is numerically degenerate.** NNConv emits values
up to 4.8e4, which pins the GRU that follows it: 49.6% of its 256 output
dimensions sit at exactly ±1, and the whole 2816-cell representation takes only
82–100 distinct values. Saturation is complete after a *single* message-passing
step and does not change through six, so it is a property of the published
weights rather than of how they are run, and it cannot be repaired — rescaling
to unsaturate would produce a function the checkpoint was never trained for.
Upstream appears to have known: `gcn_or_predictor.py` carries the comment
`## problem causing NaNs is here` on the line that calls this encoder.

An earlier version of this section reported that "the encoder reaches a fixed
point by step 3, so 3 and 6 give bit-identical output" and treated that as a
convenient invariance. It was the saturation, observed and misread.

The GCN encoder is healthy on every measure: no pinned outputs, 490–626
distinct values, and it responds to perturbation (Δ 526 from zeroing one atom's
features). It is blind to E/Z, but legitimately so, having no bond features to
lose.

**Which encoder lays the library out better is unresolved after six attempts:**

| measure | favours | strength |
|---|---|---|
| nearest-neighbour agreement with the perfumer's notes | GCN | 0.554 vs 0.457, p = 0.005 |
| global mean distance over descriptor-sharing pairs | MPNN | p = 0.030, fails its two-descriptor variant |
| family coherence (`scripts/class_coherence.py`) | GCN | +0.008 to +0.014, CI spans ±0.1 |
| homologous-series tightness | GCN | 0.0–0.1 vs 8.8–10.8 pct, may be over-clustering by chain length |
| blind forced choice, 12 trials | — | 8–3 raw, but Fisher p = 0.49 once the slot preference is controlled |
| encoder health | GCN | mechanism, not perception |

**The two spaces agree only at Spearman 0.255.** Same data, same panel, same
ESM embeddings, same attention head — only the molecular encoder differs. The
deployment stays on the MPNN because nothing has shown the GCN clearly better,
not because the MPNN is sound.

Bench cases point both ways and are worth more than any single statistic. The
GCN puts Styrallyl Acetate next to Dimethyl Benzyl Carbinyl Acetate where the
MPNN chose Strawberry Glycidate; the MPNN puts Oxane next to Thiocineole where
the GCN chose Citral. Neither is explained by fingerprint similarity.

An earlier table here quoted chemical-class coherence as 0.422 for the GCN
against 0.617 for the MPNN. Those came from a 128-material file with a `class`
column that `expanded_materials.csv` does not have, and the two stores held
different molecule sets (128 against 302), so they were never comparable.
Re-measured on the same cleaned 301 with the perfumer's own family labels:
GCN 0.271, MPNN 0.262, chance 0.047 — both far above chance, neither
distinguishable from the other.

Two implementation notes. `num_step_message_passing` cannot be recovered from
the weights (one layer is reused each step, the GRU is shared), and the shipped
config disagrees with the checkpoint on `node_out_feats` (128 vs 256) and
`edge_hidden_feats` (64 vs 128). It does not matter, for the reason above.
Separately, `run_mpnn.py` hoists the GNN out of the receptor loop, since node
features do not depend on the receptor; it asserts bit-identity against
upstream's own forward before using it, turning hours of compute into minutes.

## Validation against perceptual notes: the vectors do not predict smell

`library/input/impressions.tsv` holds one perfumer's smelling notes for all 128
materials; `scripts/check_impressions.py` tests them against receptor distance.

Pairs sharing a perceptual descriptor **are** closer in receptor space — for
the MPNN:

| model | ≥1 shared descriptor |
|---|---|
| GCN | gap +0.46, p = 0.31 |
| MPNN | gap **+2.24, p = 0.0060** |

**The MPNN carries real perceptual signal; the GCN does not.** So bond features
buy better chemistry *and* better correspondence to smell.

This result depends on the descriptor vocabulary, and the first version of it
was broken. Using only the fine-grained tags, "BANANA" and "ripened fruits"
scored as sharing nothing, and both models came out null (MPNN p = 0.21, GCN
p = 0.45). Rolling fine descriptors up into coarse families (`FAMILIES` in the
script, so banana/pear/peach/berry all imply FRUIT) is what reveals the signal.
Run with `--flat` to reproduce the broken version. The lesson is that a null
result here can be an artefact of the instrument rather than of the model.

Caveat: one nose, 128 materials, free text, a hand-written vocabulary. The
GS-LF set (5862 molecules, 152 descriptors, already in the repo) remains the
properly powered test.

### The MPNN returns identical vectors for structural isomers — because its encoder saturates

> **The observations below stand; the original explanation did not.** This was
> attributed to a Weisfeiler-Leman expressiveness limit. It is the GRU
> saturation described above, and it is specific to the MPNN checkpoint. The
> GCN encoder separates the same isomers cleanly.

The notes exposed a concrete pathology. The MPNN's closest pairs are largely
structural isomers with unrelated smells:

| pair | GCN max\|Δlogit\| | MPNN max\|Δlogit\| |
|---|---|---|
| δ-decalactone ("coconut, herbal cream") vs γ-decalactone ("dirty peach skin") | 0.55 | **9.5e-07** |
| benzyl acetate ("acetone, paint") vs ethyl benzoate ("sharp ripe fruits") | 2.23 | **9.5e-07** |
| ethyl 2-methylbutyrate ("ripened fruits") vs isoamyl acetate ("BANANA") | 2.57 | **1.9e-06** |
| α-damascone ("pear, eucalyptus") vs α-ionone ("classical violet") | 2.62 | **9.5e-07** |

1e-6 is float32 noise: the MPNN returns the *same vector* for molecules that
smell nothing alike. For δ/γ-decalactone:

* the two graphs differ in exactly one feature, `is_in_ring` (12 vs 10 directed
  ring-bond edges), and that feature carries real learned weight;
* per-atom GNN outputs do differ (row-wise max 18.2);
* but the **sorted multisets of atom embeddings are bit-identical**, and the
  encoder resolves only 6 distinct atom representations among 12 atoms;
* `MolORPredictor`'s readout is permutation-invariant, so the same bag of
  embeddings in a different order gives the same prediction.

The fourth point is why the vectors coincide. The third is the cause of the
second, and it is not an expressiveness limit — it is the saturated GRU
collapsing 12 atoms onto 6 representations. A 1-Weisfeiler-Leman argument would
predict that *any* message-passing model with a permutation-invariant readout
fails here. The GCN does not: on the same pair its summed readout differs by
**796**, and it shows no saturation. Ring size is well within reach of these
architectures.

That matters because the 1-WL reading drove a lot of subsequent work — the
fingerprint models, a search for published fixes to "the readout bottleneck",
and the Dense Sense detour — on the premise of a fundamental limit that was in
fact one checkpoint's numerics.

**So "the MPNN is better" needs qualifying**, and so does "the GCN is better".
Neither predicts smell, and the six measures above do not separate them.

## The limit that matters: chemistry, not smell

The descriptor test above shows a weak but real perceptual signal. A sharper
test asks what happens where chemistry and smell *disagree* — two materials
from unrelated chemical families that smell alike. That is the case a perfumer
actually needs a receptor-level map for, and it is where this fails.

`scripts/chem_vs_smell.py` and `scripts/musk_test.py` measure three independent
things per pair: Morgan (ECFP4) Tanimoto for structure, GS-LF descriptor cosine
for smell, and MPNN receptor distance.

### Musk chemotypes do not recognise each other

Musks are the textbook case: four unrelated scaffolds, one odour class.

| | median receptor percentile |
|---|---|
| **within** nitro | 6.5% |
| **within** alicyclic | 8.0% |
| **within** macrocyclic | 18.7% |
| nitro vs macrocyclic | **86%** |
| nitro vs polycyclic | **91%** |
| macrocyclic vs polycyclic | **91%** |

Musk Xylene and Galaxolide — both unmistakably musks — sit at the **96th
percentile** of all 45,150 pairs. The model holds a tight, correct cluster per
chemotype and has no concept uniting them.

#### Correction: it is not a gap in the training data. The musk receptor works, and the metric discards it

This section used to explain the result by blaming the data: "OR5AN1 is the
known musk receptor and its learned signal is weak, so there is no training
data from which `musk` could be learned as one thing." That was never
measured. Measured now, with `scripts/musk_receptor.py`, every part of it is
wrong.

**The receptor is not weak.** OR5AN1 is column 12 of the panel, and its logit
spread is the **97th percentile** of all 386 receptors — one of the panel's
most discriminating columns, not a dead one.

**It responds to musks, across unrelated scaffolds.** The three nitro musks
are the top three materials in the entire library on it, and five macrocyclics
are in its top forty:

| | OR5AN1 z | rank in the library |
|---|---|---|
| Musk Xylene (nitro) | +4.95 | **1** of 396 |
| Musk Ketone (nitro) | +4.24 | **2** |
| Musk Ambrette (nitro) | +4.12 | **3** |
| Civettone (macrocyclic) | +2.27 | 6 |
| Ambrettolide (macrocyclic) | +2.11 | 8 |
| Velvione (macrocyclic) | +2.06 | 9 |
| Habanolide (macrocyclic) | +1.90 | 14 |
| Muscone (macrocyclic) | +1.24 | 39 |
| Tonalide (polycyclic) | −0.41 | 271 |
| Cashmeran (polycyclic) | −1.34 | 366 |

A nitroarene and a C15 macrolactone share no scaffold. On this one receptor
they agree. Mean z over the 14 musks is +1.54 against −0.06 for the other 382,
one-sided p = 0.0012.

**It is not merely a size detector**, which is the obvious objection since
musks are large and the response correlates +0.45 with molecular weight.
Regress the response on rank molecular weight and compare musks to everything
else on the residual: +0.79 against −0.03, one-sided p = 0.057 — marginal on
14 musks. So the right comparison is not against zero but against the panel.
Computing that same size-controlled musk preference for all 386 receptors puts
OR5AN1 **first of 386** (effect +0.824, which is the panel maximum; next are
OR2T27, OR2W5, OR13C9). Of all the receptors the model could have made the
musk receptor, it made the right one.

**And it makes no difference, because of the metric.** Delete OR5AN1 from the
panel entirely and relayout: every pairwise distance agrees with the original
at Spearman **+1.0000**, median change 0.09%. One column of 386 inside a
Euclidean norm over PCA-20 cannot move anything, however correct it is.

So the musk failure is not a deficiency of M2OR or of the checkpoint. The
knowledge is present, in the right receptor, at the top of the panel. It is
destroyed on the way out, by reading a 386-dimensional panel through an
unweighted global distance. Unlike a hole in someone else's dataset, that is
ours, and it suggests the obvious experiment: a receptor-weighted or
receptor-selected distance. That experiment has now been run, and it does not
work — see immediately below.

#### The reweighting was tried, and it failed its own pass condition

`scripts/receptor_metric.py`. If one informative receptor is drowned by 385
others, weight the columns. The reason this might not work is that choosing
weights needs odour supervision, and a weighting fitted to odour labels may
just be the labels doing the work — the same circularity that produced the
retracted 7.7-vs-52.8 result below. So the design was fixed before the first
run:

- weights learned **only** from GS-LF label overlap, which is human binary tags;
- judged **only** by graded human panels nothing here was fitted on —
  Dravnieks 1985 (146 descriptors, 46 molecules in common) and Keller 2016
  (20 descriptors, ~57 subjects, 115 in common);
- split **on materials**, not on pairs, and scored only on pairs where *both*
  materials were held out, because a material's weight contribution would
  otherwise be learned from its own other pairs;
- **pass condition, registered in the script's docstring before running it:
  beat the unweighted baseline on *both* panels.** One of two is not a result.

Two weightings were fitted — a cheap per-receptor correlation weighting, and a
diagonal metric learned by gradient descent — over 40 random half splits of
the materials. Spearman against the panel, signed so larger is better:

| | Dravnieks | Keller |
|---|---|---|
| unweighted (the current pipeline) | +0.017 | +0.011 |
| correlation weighting | +0.018 | +0.013 |
| learned diagonal metric | **+0.057** | +0.009 |
| ECFP4 Tanimoto (structure only) | +0.064 | +0.017 |

Paired per split against the baseline, the learned metric wins 85% of splits
on Dravnieks (mean Δ +0.032) and **25%** on Keller (mean Δ −0.006). Every
confidence interval spans zero. So it fails the pass condition, and it fails it
in precisely the way the pass condition was written to catch: had only
Dravnieks been reported, this would read as a modest success.

The more useful observation is the size of the whole table. Nothing in it is
distinguishable from nothing. The unweighted receptor distance predicts
independent panel similarity at +0.017 and +0.011; structure itself manages
+0.064 and +0.017. At this magnitude the methods are being compared inside the
noise, and the comparison is not capable of resolving receptors from structure,
let alone one weighting from another.

What that does and does not establish. It does **not** exclude a small effect:
46 and 115 molecules, and 20 coarse descriptors in Keller's case, are weak
instruments. It does exclude a *large* one — and a large one is what the musk
case needs, since uniting nitro with macrocyclic musks means moving a pair from
the 85th percentile to somewhere near the 10th. That move is not available from
reweighting this panel.

So the correction above stands as a correction — the knowledge is in OR5AN1 and
the global distance discards it — without licensing the repair it suggested.
The honest summary is that the panel contains at least one genuinely
odour-relevant axis and not enough of them to build a metric on, which is a
sharper version of the same conclusion the next section reaches by a different
route.

One bug found and fixed on the way, recorded because the first run of this
script reported a clean null that was an artefact: the learned metric
parameterised its weights as `w = a²` and initialised `a` at zero, where the
gradient of `w` is also exactly zero. The optimiser never moved, the "learned"
metric stayed uniform, and it reported a difference from the baseline of
exactly +0.0000 on both panels. A null result that is *too* clean is a bug
report.

Two honest limits on this correction. OR5AN1 unites nitro and macrocyclic
musks and does *not* pick up the polycyclic or alicyclic ones — Tonalide,
Celestolide, Cashmeran and Helvetolide all sit at or below the library mean —
so even the right receptor accounts for two of four chemotypes, which is what
the receptor literature would predict rather than a failure. And 14 musks is a
small sample; the rank-1-of-386 result is what carries the claim, not the
p-value.

#### Correction to that correction: a class detector is not a similarity axis, and importance is per descriptor

A reader asked the obvious follow-up. If OR5AN1 is a specific, reliable musk
receptor the way rods matter more than cones for seeing in the dark, scale the
important receptors up — and why did training not do that already?
`scripts/receptor_importance.py` answers all three parts, and the first answer
corrects what the section above implied.

**Scaling it up makes musks worse, not better.** Sweeping the weight on OR5AN1
and watching where nitro-vs-macrocyclic musk pairs sit:

| weight on OR5AN1 | musk pair percentile | ρ vs Dravnieks | ρ vs Keller |
|---|---|---|---|
| 1 (unweighted) | 84% | +0.032 | +0.022 |
| 10 | 85% | +0.032 | +0.022 |
| 100 | 89% | +0.031 | +0.019 |
| 10 000 | 94% | +0.024 | +0.013 |

Monotonically worse on the thing it was supposed to fix, and slightly worse on
both panels as well. The reason is in the numbers the section above reported
without reading them properly: nitro musks sit at +4.44 z on OR5AN1 and
macrocyclics at +1.83, both far above the library — and **2.60 z apart from
each other**, while the spread *within* each group is 0.37. So OR5AN1 answers
"are these musks?" extremely well and "are these two alike?" not at all.
Upweighting an axis on which two materials differ pushes them apart. A class
**detector** is not a similarity **axis**, and `musk_receptor.py` measured only
the first while the section above implied the second.

**An external check does not confirm the musk claim either.** Against Keller's
own MUSKY rating over 115 molecules, OR5AN1 ranks **228 of 386** (|ρ| 0.050);
the best are OR2T11 at 0.346, OR2T35 at 0.298, OR2T2 at 0.294.
`musk_receptor.py` asked a narrower question — are *this library's* 14 musks
elevated on it with size controlled — and got rank 1 of 386. Both can be true,
but 115 molecules of external panel data outweigh 14 in-library contrasts, and
the honest reading is that the musk-receptor identification is weaker than that
section claims.

**But the reader's underlying intuition is right, and it is the useful finding
here.** Some receptors do carry much more odour information than others — per
descriptor. For every descriptor in the two panels, the single best-predicting
receptor:

| | median best-single-receptor \|ρ\| | max | distinct receptors |
|---|---|---|---|
| Dravnieks, 146 descriptors | **0.378** | 0.590 (MINTY → OR2T11) | 50 |
| Keller, 20 descriptors | **0.298** | 0.409 (AMMONIA → OR9A4) | 16 |

Compare those with +0.032 and +0.022 for the whole 386-dimensional distance
against panel similarity. **One receptor predicts a descriptor an order of
magnitude better than the entire panel predicts similarity.** The information
is there in quantity; the global distance is what loses it.

And this is why the reweighting experiment above could not have worked. The
receptor that serves one descriptor is not the one that serves the next — 50
distinct receptors for 146 Dravnieks descriptors. A single weight per receptor,
applied to every comparison, would have to be right for "is this minty?" and
"is this musky?" at once, and those want different receptors. The rod analogy
breaks in exactly the same place: rods beat cones for detection in dim light
and are useless for telling two hues apart. Importance in a sensory system is
relative to the question, not a property of the receptor.

**Why training did not produce it.** Because nothing in the checkpoint's loss
ever compares two molecules. It predicts one binary outcome per (molecule,
receptor) pair, and the 386-vector is a stack of 386 independent predictions
assembled afterwards, here, by this project — it is not a learned embedding.
No term in the objective has any opinion about the geometry of that stack, so
no importance weighting over it could emerge; it was never asked for. Contrast
the percept model, whose output dimensions *are* odour descriptors, so
similarity in its output space is similarity in odour-label terms by
construction. That is also why the percept model, not the receptor panel,
drives everything in the simplifier.

Which names the experiment that follows from all of this, and it is not a
distance at all: **learn a supervised readout per descriptor from the receptor
vector**, and see whether it beats the fingerprint the percept model currently
uses. The table above is a one-receptor lower bound on what that could reach.
Untried.

### A validation that looked like a pass, and was not

Scored across all 301 materials using the **percept model** as the smell axis,
structurally unrelated but similar-smelling pairs sat at the 7.7th percentile
versus 52.8th for a control — an apparent success.

That axis is itself a graph network over the same molecules. Two structure-based
models can agree while sharing a blind spot. Repeating the test with the
perfumer's own notes as judge — the only odour data here no model produced:

| judged by the perfumer's notes | median receptor percentile |
|---|---|
| structurally unrelated, notes agree (426 pairs) | 50.4% |
| structurally unrelated, notes share nothing (3920 pairs) | 52.6% |

**No signal.** The 7.7-vs-52.8 result was two correlated models agreeing with
each other. Any evaluation that uses one model to judge another must be treated
as circular until checked against data outside both.

Partial correlations over all pairs: receptor~structure with smell held fixed
is −0.188; receptor~smell with structure held fixed is −0.113. Both survive, so
there is odour information beyond structure — but it is weaker, and it vanishes
on the hard cases.

### Practical consequence

**The vectors are a chemistry representation with a weak odour correlate, not
an odour representation.** They succeed where chemistry and smell already agree
and add nothing where they diverge.

When using this to pick materials, sort candidates by Tanimoto to the query:

* **Tanimoto ≥ 0.25** — reliable regime, but the answers are mostly ones a
  perfumer would reach anyway (for a green-C6 target it suggests more green C6).
* **Tanimoto < 0.25** — blind spot. Both positive and negative verdicts here
  are uninformative, and the negatives are the worst of all: the noisy-OR
  combination rule only ever adds, so it structurally undervalues any material
  whose job is to cut, sharpen or contrast.

So this is a bookkeeping and gap-finding tool over a material library, not a
discovery engine. It reliably reports what is missing in descriptor terms and
what is chemically adjacent to fill it. It will not find the unexpected
material from an unrelated family.

## Why this is a learned model and not a physics simulation

A reasonable question: protein structures are predictable now, molecular
simulation is mature, so why interpolate from sparse measurements instead of
computing binding from first principles? People have tried, for about 25 years.
It is worth recording why it has not displaced the data-driven approach, and
where the physics route might still pay off.

**What has been tried.** Before AlphaFold, olfactory receptors were modelled by
homology to bovine rhodopsin — the only GPCR structure from 2000 until the
β2-adrenergic receptor in 2007. ORs share roughly 20% sequence identity with
rhodopsin, enough for an approximate seven-helix backbone and not nearly enough
for binding-pocket side chains. Many docking campaigns were built on those
models. They rationalised known ligands more often than they predicted new ones.

**Four reasons it does not produce usable predictions:**

1. *Structure prediction gives the wrong state.* GPCRs are conformational
   ensembles; predictions skew toward inactive conformations because the PDB
   does. Agonists stabilise the active state, so docking an agonist into an
   inactive model is the wrong exercise, and confidence scores do not flag it.
2. *The signal sits below the scoring noise floor.* Separating a strong from a
   weak odorant is 1–2 kcal/mol. Docking scoring functions carry 2–3 kcal/mol
   error, because they were fit against nanomolar drug binding while odorants
   bind at micromolar to millimolar.
3. *Entropy and water dominate.* Odorant binding is largely desolvation and
   displacement of ordered water from a shallow pocket — exactly what docking
   scores approximate worst.
4. *Promiscuity is the function.* A receptor that binds a hundred ligands
   weakly has no single well-defined pose to find.

**The deeper reason is that the validation loop never closed.** Simulation
becomes accurate where predictions can be checked against measurements and
corrected. ORs barely express in heterologous cells; the chaperone trick
(RTP1/RTP2) that made screening possible at all arrived only in 2004, and
throughput stayed low. With almost no ground truth, a failed prediction cannot
be attributed to the structure, the conformational state, the scoring function
or assay noise — so the method cannot be improved. That, more than any single
inaccuracy, is what stalled the field.

### Free-energy methods: the part worth watching

Alchemical free-energy methods — FEP (free energy perturbation) and TI
(thermodynamic integration) — are the one physics approach that genuinely works
in industry, reaching roughly 1 kcal/mol on *relative* binding affinity. They
handle the entropy and explicit-water problems that defeat docking scores,
because they sample the full thermodynamic cycle rather than scoring a single
pose.

Three conditions gate them, and olfaction currently fails all three:

- a reliable structure of the actual target, ideally experimental;
- a reference ligand with measured affinity to perturb *from* (FEP predicts
  ΔΔG between similar compounds, not absolute affinity);
- chemical similarity between the reference and the query.

They also cost hours to days of GPU time per ligand–receptor pair, so 400
receptors × a material library is not a feasible campaign today.

Where they could become useful here: not for building a whole receptor panel,
but for *one* well-characterised receptor where a question is worth the compute.
OR51E2 now has a cryo-EM structure (Billesbølle et al., Nature 2023, bound to
propionate), so a congeneric series of short-chain acids against OR51E2 is the
kind of question FEP could answer today. The results would also be real
training data — which is the scarce resource. Note that OR51E2 is atypical: it
binds a small carboxylic acid and is class-I-like, so its pocket may not
generalise to the bulky flexible odorants perfumery cares about.

A cheaper middle path, if this is ever pursued: use predicted structures plus
pocket-residue features as *additional model inputs* rather than as a simulation
substrate, letting the learned model decide how much to trust them.

## Known limitation: the published logits are not reproducible

`olfactory_subgenome_OR_logits.pt` on Zenodo **cannot be reproduced
numerically** from the published artefacts, and this is a property of the
upstream release rather than of this implementation.

`scripts/generate_OR_predictions_pseudogenes.py` as frozen cannot have
produced it. The script reuses one batched graph `bg` across its loop over all
845 receptors, while `utils.predict_OR_feat` calls `bg.ndata.pop('h')` —
which *removes* the feature. The second receptor therefore raises
`KeyError: 'h'`. The tensor must have come from a different revision whose
configuration is not recoverable from the repository.

What was checked before reaching that conclusion:

* Our cached ESM-2 embeddings are **bit-identical** (max |Δ| = 0.0) to what
  upstream's own `esm_embed` produces for the exact consecutive chunk-of-5
  grouping it uses over the 845-sequence ordering.
* Weights load with `strict=True`, and the architecture is re-derived from the
  checkpoint's tensor shapes.
* Molecule rows align: for a fixed receptor, our logits correlate 0.59–0.81
  with the published column across 200 GS-LF molecules.
* All eight combinations of {unweighted, weighted} checkpoint ×
  {self-loops, none} × {unweighted, weighted} published logits were compared.
  The configuration used here is the closest of them (corr 0.69); none matches
  numerically.
* Sweeping `max_node_len` over {22, 98, 100, 110} moves correlation only
  0.61 → 0.69, so the atom-slot budget does not explain the gap.

Accordingly, acceptance test 5 verifies equivalence against upstream's
inference path — identical weights, featurisation and embeddings — which is
the claim that can actually be established, and a separate test asserts
structural (per-receptor, across-molecule) agreement with the published
tensor.

A further upstream bug is worked around rather than inherited:
`data/m2or.esm_embed(device=...)` moves the *tokens* to the requested device
but calls `setup_esm()` without forwarding it, so the model lands on
`setup_esm`'s own `cuda:0` default whenever CUDA is visible. Requesting CPU
embeddings on a CUDA machine would fail with a device mismatch;
`embeddings._place_encoder` pins the cached encoder explicitly.

## Library and 2D maps

```bash
python -m orvector library add --csv library/input/final_materials.csv --device cpu
python -m orvector library list
python -m orvector project --methods pca,mds --out-dir artifacts/maps
```

The library (`artifacts/library/`) stores **raw** logits and scores keyed by
InChIKey. Normalisation is set-dependent, so storing it would invalidate every
vector whenever a molecule is added; it is applied at analysis time instead.
Appends are refused if the panel hash, checkpoint or encoder config differs
from the store's provenance, since those vectors would not be comparable.

`project` writes a PNG plus a coordinates CSV per method, and reports a
**Shepard rank correlation** — Spearman between 2D pairwise distances and the
true 386-dim ones — so you can tell a result from a decoration.

### Does receptor space encode chemistry?

Measured on the 128-material library this section was written against, with the
GCN checkpoint. The class labels came from that file's `class` column, which
`expanded_materials.csv` does not carry; re-measuring on the cleaned 301 needs
the perfumer's family labels instead, and gives 0.271 against a chance rate of
0.047 (`scripts/class_coherence.py`). The figures below are therefore historical
and not comparable with anything measured on the current library.

| check | value |
|---|---|
| nearest neighbour shares chemical class | **0.422** vs 0.074 by chance (5.7x, permutation p < 0.0005) |
| same, with the potency axis removed | **0.430** — so it is not merely a size effect |
| within- / between-class distance ratio | 0.785 |
| PCA 2D fidelity (Shepard) | 0.978 all components, 0.783 potency-removed |

Homologous series land among the closest pairs in the whole library: fatty
aldehydes C8/C10/C11 and γ-lactones C9/C10/C11 both at the **0.1st percentile**
of all pairwise distances, damascones at the 1st, the linalool family at the
2.6th, macrocyclic musks at the 5.7th.

Mutual nearest neighbours are chemically coherent — Aldehyde C-10 ↔ C-11,
Ambrettolide ↔ Habanolide, γ-decalactone ↔ γ-undecalactone, vanillin ↔ ethyl
vanillin, α-ionone ↔ β-ionone, dihydroeugenol ↔ methyl diantilis.

Caveat: the ionones as a *group* sit at the 49th percentile, because methyl
ionone γ falls far from α/β-ionone. Not every intuitive family is tight.

## Predicting smell directly: five models, two axes

Receptor vectors encode chemistry, not smell (above). So the library is also
scored by models trained on human odour descriptors — the GS-LF set
(GoodScents + Leffingwell, 4983 molecules × 138 descriptors) that OpenPOM uses.

The important discovery is that **the usual accuracy metric and the thing this
project needs are not the same measurement**, and optimising one damages the
other.

- *Out-of-fold ROC-AUC* asks whether a model ranks molecules correctly **within
  one descriptor** — is this molecule more `fruity` than that one?
- *Label-grounded similarity* asks whether two molecules sit near each other
  **across all descriptors**. This is what the viewer's layout, the
  nearest-neighbour suggestions and the accord substitution work all rely on.
  Measured as the Spearman correlation between predicted cosine similarity and
  the Jaccard overlap of human descriptor sets, over the 26 106 pairs of the 229
  library materials GS-LF covers (`scripts/eval_similarity.py`).

| model | OOF AUC | similarity ρ |
|---|---|---|
| fingerprint **v3** (ablation-selected) | 0.8672 | **+0.4894** |
| fingerprint v2 (`pos_weight`, ensemble) | **0.8818** | +0.3731 |
| fingerprint v1 (first attempt) | 0.7912 | **+0.5505** |
| OpenPOM (published checkpoints) | 0.8872\* | +0.4476 |
| Dense Sense (adds electron density) | — | +0.3731 |
| GS-LF MPNN | — | +0.3355 |
| Uni-Mol (pretrained, no odour labels) | — | +0.1508 |
| raw ECFP4 (control) | — | +0.1804 |

\* OpenPOM's own published figure under a similar protocol, not a head-to-head:
its checkpoints were trained on all of GS-LF, so it cannot be scored on a
held-out part of it. See `scripts/eval_fp_percept.py`.

### Adding a label does not fix a molecule

Twenty-two library molecules carry no GS-LF label and are predicted from
structure alone. Five have well-established published descriptions, so those
were reduced to the 138-word vocabulary and added as training rows
(`scripts/augment_labels.py`, which writes `data/percept/gslf_plus_bench.csv`
and leaves the vendored GS-LF file untouched). The question was whether a
label, once added, makes the model say it.

Mostly not. Over the 378 library molecules that carry a human label, the model
puts a labelled descriptor in its own top 5 of 138 about 45% of the time and
its top 20 about 87% of the time; the median rank is 6. The five new rows land
in the same place as the upstream ones — median rank 10 — so they are not
being ignored, they are being treated exactly like every other label. Rose
oxide, given `rose, geranium, green, metallic, fresh, floral`, still reads
celery, malty, mint, with `rose` itself at 106 of 138. Thiocineole took
cleanly: `grapefruit` first, `black currant` fifth.

The size of the effect is also below the noise floor. Three retrains differing
by one or two of these rows gave mean ROC-AUC 0.8659, 0.8670 and 0.8675, and
label-grounded ρ +0.474, +0.538 and +0.497. That spread is re-initialisation,
not data — which means **the fourth decimal of every AUC in the table above is
not a real digit**, and one row among 4988 cannot outvote a structural
neighbourhood.

A corollary worth the trouble: where a label and a structure disagree loudly,
suspect the label. Cassifix is sold as a fruity-green cassis note, and
labelling it that way put `green` dead last of 138 in its own prediction. Its
stored SMILES carries the campholenic cyclopentene of the sandalwood
synthetics, the model reads amber, woody, cedar, sandalwood, and a blotter
agreed with the structure. Supplier copy for a specialty often describes the
effect it is sold for in an accord rather than the smell of the molecule. It
is left unlabelled.

### Correction: the similarity column above is measured against the training labels

The ρ column in that table is agreement with GS-LF label overlap — the same
labels every model there was fitted on. Two independent graded datasets
(`scripts/build_percept_datasets.py`, `data/percept/`) were later used to check
it, and **the ranking does not survive**.

Dravnieks (30 materials in common, 146 graded descriptors) and Keller (88
materials, 20 descriptors) are human panel data, continuous rather than binary,
and neither trained anything here:

| model | GS-LF ρ | Dravnieks ρ | Keller ρ |
|---|---|---|---|
| fingerprint v1 | **+0.5505** | +0.2103 | +0.0993 |
| fingerprint v2 | +0.3731 | +0.2895 | +0.1091 |
| fingerprint v3 | +0.4894 | +0.2851 | **+0.1250** |
| OpenPOM | +0.4476 | **+0.3161** | +0.1152 |
| Dense Sense | +0.3731 | +0.2974 | +0.0938 |
| GS-LF MPNN | +0.3355 | +0.2816 | +0.1098 |
| Uni-Mol | +0.1508 | +0.1224 | +0.0653 |
| raw ECFP4 | +0.1804 | +0.1691 | +0.0490 |

v1's large lead inverts: it is last among the trained models on Dravnieks and
second-last on Keller. A paired bootstrap over materials — resampling both
models on the same draw, which is the right test when the pairs are not
independent — returns **no significant difference between any two trained
models** on either dataset. The single significant result anywhere is v3 over
raw ECFP4 on Keller, +0.072 with CI [+0.017, +0.129].

So three things follow, and they undercut the section above.

1. **The trained models are indistinguishable on independent human data.** The
   v1-vs-v2-vs-v3 ordering was measuring fit to GS-LF, not agreement with
   smell.
2. **The ablation table below therefore rests on a discredited metric.** The
   "drop pos_weight, gain +0.079 of similarity" result is real as a statement
   about GS-LF Jaccard and unsupported as a statement about smell. v3 remains a
   sensible configuration, but it was not *selected* on evidence that holds up.
3. **Absolute agreement is low everywhere** — ρ ≈ 0.1–0.3. None of these models
   reproduces human descriptor profiles well. The honest summary is that they
   beat raw chemistry and are otherwise hard to tell apart.

What the graded data is for, going forward: it is the evaluation set, because it
is not the training target; and Keller carries concentration, perceived
intensity and between-subject spread, none of which GS-LF can express.

Two results are worth stating plainly.

**Uni-Mol scores below raw fingerprints.** It is a transformer pretrained on
209M 3D conformers and has never seen an odour label; its CLS embedding reaches
+0.151 against raw ECFP4's +0.180. A general-purpose 3D chemical prior does not
encode smell similarity, which is evidence the odour labels carry information
chemistry alone does not — and means fine-tuning is the only route by which a
3D model could help here.

**The best-AUC model had the worst geometry, and the trade was avoidable.** Two
ablation rounds (`scripts/ablate_fp.py`, `ablate_fp2.py`) found that nearly
everything v2 added for accuracy was paid for in geometry, and that the two
costliest additions bought almost nothing:

| change | Δ similarity | Δ AUC |
|---|---|---|
| drop per-label `pos_weight` | **+0.079** | −0.008 |
| standardise all features, bits included | **+0.064** | −0.018 |
| drop dropout | +0.045 | −0.009 |

`pos_weight` upweights rare descriptors, so the model asserts them widely; that
adds a shared component to every vector and washes out the differences between
molecules. Standardising the binary fingerprint bits — which v2 removed as a
bug — divides each column by its standard deviation, so a molecule carrying a
rare substructure gets a correspondingly large input. For per-label ranking
that is noise; for telling two molecules apart it is signal.

Caveat on precision: a single-model v3 scores 0.8537/+0.4263, worse on **both**
axes than the five-model ensemble, which puts roughly ±0.03 of seed noise on the
single-model Spearman values above. Only the first two rows are safely outside
it; the dropout term is marginal and the batchnorm and layer-width terms were
within noise.

v3 is therefore v1's architecture — (512, 256), no batchnorm, no dropout, all
features standardised, no `pos_weight` — trained v2's way, with a five-model
ensemble and long cosine-annealed training, both of which cost no geometry.
Nothing measured here dominates it on both axes.

```bash
python scripts/train_fp_v3.py                      # writes artifacts/percepts_fp3.json
python scripts/eval_similarity.py --percept "v3=artifacts/percepts_fp3.json" --ecfp
```

### What none of them did

A blind test (`scripts/blind_set.py`) drew six materials spanning the predicted
range — some predicted to improve a gooseberry accord, some predicted to do
nothing — and presented them shuffled, with the key written to a file rather
than the terminal. The perfumer's rankings did not match the predictions: of
the six, only one of the strongly-predicted materials helped. Four earlier
*unblinded* trials had appeared to confirm the model, which is the expected
shape of expectation bias. Treat the suggestion ranking as a search heuristic
over a 302-material library, not as a prediction.

## Evaluation data that nothing here is trained on

The lesson of the corrections above is that a metric built from the training
labels will flatter the model fitted to them. These are the instruments that do
not have that defect. `scripts/build_percept_datasets.py` writes them to
`data/percept/` with a manifest recording provenance and every judgement made.

| dataset | content | in the library |
|---|---|---|
| `dravnieks.csv` | 144 stimuli, 146 descriptors, percent applicability with 2146 distinct values | 43 |
| `keller.csv` | 956 stimuli over 474 molecules at two dilutions, 20 descriptors plus intensity, pleasantness and familiarity, 0–100 per subject with between-subject spread | 99 |
| `abraham_thresholds.csv` | 268 **measured** air-phase detection thresholds, `Log(1/ODT)` | 28 |
| `gslf_hierarchical.csv` | GS-LF with its implied hierarchy propagated, 951 labels added | — |

Two interpretation decisions in the Keller build are recorded rather than
hidden. Blank descriptor ratings are read as zero: the blank rate is 83% and
barely moves with intensity, so they mean "not applied". A blank *intensity* is
also zero, but for a different reason — it coincides with the subject reporting
they smelled nothing in 97% of such trials — and averaging only over subjects
who did smell something would overstate every molecule.

`data/inventory/` holds the perfumer's own sheet, normalised by
`scripts/build_inventory.py`: 166 rows, 129 single molecules, 128 joining the
library. Its `top %` column — the working ceiling they actually dose at — tracks
the measured thresholds at Spearman 0.756, better than anything modelled here,
and is the reason the predicted thresholds were withdrawn. Its `family` column
restored the class axis. Its assembled water-threshold column is kept as
reference only: rho 0.451 against Abraham, and 0.386 against the sheet's own
use levels.

### The evaluation scripts

| script | asks |
|---|---|
| `eval_similarity_graded.py` | does a representation's geometry match graded human panels? Paired bootstrap over materials, since pairs are not independent |
| `notes_vs_models.py` | does a model rank the perfumer's own descriptors highly? Scored against a base-rate prior, which alone reaches 0.841 AUC |
| `nn_agreement.py` | does a material's nearest neighbour share a descriptor with it? The local measure, which is what reading a map actually is |
| `class_coherence.py` | same question for odour family |
| `encoder_health.py` | saturation, representation diversity, input sensitivity, and two isomer probes |
| `probe_pair.py` | where every representation puts one specific pair, with the chemistry baseline and what each dataset actually says |
| `blind_encoder_test.py` | forced choice between two layouts, key written to a file and never printed |

A caution carried by `notes_vs_models`: predicting GS-LF base rates while
ignoring the molecule entirely scores 0.841 AUC against the perfumer's notes,
where the best model reaches 0.905. Most of an impressive-looking AUC on
unbalanced odour labels is the prior.

## Formulas: reading a blend, and the one prediction a nose has checked

Everything above this section predicts for one molecule at a time. This part
predicts for a formula, and it contains the only claim in the whole repository
that was stated in advance and then confirmed by smelling something.

### The throwaway path

```bash
python scripts/try_formula.py pasted.txt --owned
python scripts/try_formula.py v1.txt v2.txt --label "v1" --label "v2"
```

`try_formula.py` takes a formula sheet exactly as a perfumer writes one -- name
first, amount last, tabs or spaces, percent signs optional, a stock strength in
the name read as a dilution so `Calone 10%  40` contributes four parts of
calone and not forty -- and produces a local atlas with the blend on it as a
star. Several formulas go on one page, fitted together, so the distance between
their stars is a distance in the same space as everything else.

Nothing is added to the library and nothing is deployed: the intermediate files
go to `artifacts/temp/`, which is gitignored, and the page gets no analytics
because no token is passed. A formula is not a material and has no business in
the molecule store.

### Naturals are built from GC compositions, and they check themselves

A formula written by a perfumer is mostly naturals, so dropping them costs more
coverage than anything else. `scripts/build_naturals.py` builds each oil from a
sourced GC composition in `data/naturals/compositions.json`, in both percept
and receptor space, by the same combination rule a formula uses.

The validation is close to free and it is blind: the percept model's
138-descriptor vocabulary contains *bergamot*, *lavender*, *geranium*, *mint*,
*clove*, *cinnamon*, *lemon*, *vetiver* and a dozen more, so an oil can be
scored without being told its name. The question is where its own name lands by
lift against a null of random blends of the same size. **14 of the 19
checkable oils put their own name in the top 5 of 138.** Four lavender
chemotypes span cosine 0.72 to 0.91 and the model ranks them as different
materials, which is the clearest demonstration here that it reads proportions
and not just ingredients.

An oil within 0.98 centred cosine of a single library material is built but
kept **off** the atlas: its point would say nothing the molecule beside it
already says. Ten of 28 are in that position -- sweet orange and tangerine sit
exactly on D-Limonene, lemongrass 0.992 from Citral. They remain available to
formulas, because "useless as a point" and "necessary as an ingredient" are
different questions, and dropping them from the file made any formula that said
`Lemon EO` lose that weight.

### The library now holds reference molecules, not only shelf stock

`vetiver and patchouli cannot be built` was wrong, and wrong in an instructive
way. Against a published vetiver composition the library resolved **1.4%** --
caryophyllene and a trace of farnesol. The limit was never the model: it was
that no vetiver sesquiterpenoid was in the library. Thirteen were added as
reference molecules with PubChem structures (khusimol, the vetivones,
vetiselinenol, isovalencenol, zizanoic acid, khusimone; patchoulol, the
guaienes, seychellene, β-patchoulene, norpatchoulenol), which is consistent
with what the library already was -- 321 materials of which about 129 are on
the perfumer's actual shelf.

| oil | resolves | blind self-check |
|---|---|---|
| Patchouli (Javanese) | 82% of the oil | no vocabulary word; lift reads earthy +2.9, camphoreous +2.5, woody +1.8 |
| Vetiver (Haiti) | ~46% of the oil | **`vetiver` 3rd of 138, z +1.81** |

Vetiver naming itself is worth more than it looks: khusimone is the only
constituent GS-LF ever labelled *vetiver*, and it is 2.6% of the build, so the
other 97% had to carry the word by structural similarity alone. It will also
stay the weakest build here for a reason no addition fixes -- vetiver has no
majority constituent, it is ~150 sesquiterpenoids, and the named ones total
under half the oil.

What stays out of reach is different in kind. Labdanum is C20 labdane acids
plus an unassigned resin fraction: **5.7%** of a published composition resolves
even now, and the volatiles that carry its smell are mostly unidentified in the
literature. Adding molecules cannot fix a composition nobody has written down.

### The citrus cluster: the ordering is right, the scale is 15x too small

Asked to compare the three points the model cannot separate, the perfumer
reported that tangerine, orange and D-Limonene are indeed very similar, with
tangerine "milder and more silent", orange "more bright and rich", and
D-Limonene standing to orange oil as this project's six-constituent petitgrain
stood to real petitgrain -- a skeleton of it.

That names an axis of completeness rather than of character, and the model has
it. By amplitude -- the norm of the predicted percept vector, which centred
cosine discards -- the ordering is D-Limonene 2.023, tangerine 2.030, orange
2.048. Exactly the reported ordering, unprompted.

The magnitudes are the problem. Those differences are 0.010 and 0.035 in
absolute distance, against a just-noticeable yardstick of 0.525: the model has
the sign right and is fifteen to fifty times short on the scale, so by its own
reckoning it would call all three indistinguishable. Switching the trim
criterion from cosine to absolute distance would not rescue them either, for
the same reason.

What the model does separate, at near-yardstick scale, is the oils whose
composition genuinely diverges from limonene: mandarin sits 0.199 from orange
and lemon 0.449, because they carry 15% gamma-terpinene and 14% beta-pinene
respectively. The oils it cannot separate are the ones whose difference is
carried entirely in traces -- the sinensals, valencene, methyl
N-methylanthranilate -- which is the same wall as nootkatone in grapefruit,
Calone in the Acqua di Gio formula, ethyl maltol in the mojito, and the spice
in petitgrain. Four of those were found on a bench and one in the data; none
was found by a metric.

### Read lift against a null, never the raw values

The raw combined descriptor values are dominated by GS-LF label frequency. A
random ten-material blend already scores floral 0.90 and sweet 0.95, so a
formula scoring floral 0.99 has said almost nothing. `predict_formula.py`
therefore standardises against a null of random blends **of the same size at
the same weights**, and the lift is the only informative reading. On a 32-
ingredient fresh-woody formula the raw list was floral / woody / fruity /
green; the lift was `jasmin +6.5` (Hedione at 14.5%, read correctly), then
natural, vetiver, amber, dry, cedar, lavender, sandalwood, bergamot.

What the null cannot fix: formula and null are combined by the same rule, so a
bias in the rule survives the subtraction.

### A blend's position on the atlas is a property of the combination rule

This was a bug, found by a perfumer asking why a blend was not in the middle of
the cloud when its ingredients were. Noisy-OR is a union, not an average:
`1 − Π(1 − wᵢpᵢ)` is ≥ every ingredient on every receptor, so a blend cannot
land between its ingredients, it gets pushed outward. A 28-ingredient formula
fired 89 receptors above 0.5 where the median single material fires none, and
its total activation was 2nd highest of 325 points. Its distance from the
centre was measuring ingredient count, not character.

| rule | distance from the cloud's centre | nearest point |
|---|---|---|
| noisy-OR (union) | 62.2 | Cedrene Alpha |
| mean of probabilities | 31.3 | Lime EO |
| **mean of logits** | **14.5** (median material 14.6) | **Lavender EO (French)** |

The atlas is built by taking logits and reducing them, so averaging in that
space is averaging in the space the picture is drawn in. `--place logit-mean`
is now the default for an injected blend, and the inspector card states which
rule placed the star. The oils are mostly unharmed by the union rule -- with
five to eight constituents the inflation is small -- with one exception worth
knowing: **Cedarwood EO is the single most extreme point on the published
atlas** at 152.8 total activation against a library median of 5.5.

### Olfactory white sits at the centroid

Published work reports that mixtures of ~30 equal-intensity components spanning
odour space converge on a common percept. This model reproduces the
convergence, and the measurement is only useful if the two mechanisms are kept
apart.

| blend | distance from the average point, by mean | by noisy-OR | nearest single material |
|---|---|---|---|
| single material (median) | 1.22 | 1.22 | — |
| random 3 | 0.81 | 0.90 | +0.99 |
| random 10 | 0.47 | 2.58 | +0.74 |
| random 30, three independent draws | **0.20, 0.20, 0.31** | 4.2, 4.9, 5.6 | +0.55, +0.55, +0.60 |
| a real 12-material formula | 0.61 | 1.95 | +0.81 |

Under a mean, large random blends collapse **into** the centroid -- four to six
times closer in than any material. Under noisy-OR they shoot **outward** to the
saturation corner, and two unrelated 30-material blends then reach cosine
+0.94 with each other purely by both sitting against the ceiling. Only the
first of those is the phenomenon; the second is an artefact, and cosine is
misleading in both directions, so `--compare` now prints absolute distance from
the average point alongside it and refuses to compare two blends combined by
different rules.

On the atlas the three random 30-material blends sit 1.1 to 4.4 from the
cloud's centre where the median material sits 14.9 out -- the 0th to 1st
percentile. The mud is in the middle, exactly. The single materials nearest
that centre are Pharaone, Undecavertol, Nympheal, diethyl phthalate and methyl
benzoate; the furthest out are α- and β-pinene and fenchone.

A real formula is also pulled well inward (0.61, versus 1.22 for a material),
which is the uncomfortable part: by this measure a twelve-material accord is
already a third of the way to white.

### Three different whites, because the centroid is not flat

Calling the centre of percept space "the mud" was wrong, and the error was
caught by smelling it rather than by any measurement. The centroid is the
*average* smell: sweet 0.45, fruity 0.44, floral 0.38, green 0.35, with a
standard deviation of 0.075 across the 138 descriptors. Nothing about that is
characterless. The perfumer's verdict on a three-material blend aimed at it was
that it is "the bright white where all the receptors do fire ... bright and
sharp like the bright light", which is a better description than mud and a
different thing from what the published olfactory-white experiment produced.

So `mud.py --target` now offers three, and they are measurably different:

| target | what it asks for | best 5-material blend reaches | profile mean | profile sd |
|---|---|---|---|---|
| `centroid` | the library's average smell | 0.211 | 0.051 | 0.079 |
| `flat` | every descriptor equally applicable | 0.546 | 0.041 | 0.046 |
| `dull` | the origin: smelling of as little as possible | 0.647 | 0.031 | 0.045 |

The ordering is the useful part. The centroid is the brightest and the most
peaked; the flat target is dimmer and genuinely more even; the origin is dimmest
of all. And only the centroid is reachable -- 0.211 is inside the
nearest-neighbour yardstick of 0.525 while the other two are not, so from five
materials an equal-energy white cannot be built at all, only approached.

Dosing had to be fixed first, twice. The blend aimed at the centroid failed on
the bench because methyl benzoate dominated it, and of the three potency
signals in this repository only the measured detection thresholds predicted
that (0.52 ppb against undecavertol's 3.0; the perfumer's use levels put all
three materials at 2% and the predicted-intensity model ranked methyl benzoate
*weaker*, though it answers a different question -- intensity at a fixed dose
rather than threshold). `--potency threshold` makes the solved weights shares
of odour activity and converts them to masses, which drops methyl benzoate from
33.5% to 7.0%; the bench then preferred 5%, so `--oav-exponent` exists and 1.2
fits that single observation, which is one parameter fitted to one data point
and is labelled as such.

The second fix: converting activity to mass with thresholds spanning orders of
magnitude produces arithmetically correct nonsense -- a blend 96% cinnamyl
alcohol and 0.0% pistachio thiazole, which is one material with contamination.
The search now rejects any subset whose implied doses fall outside
`--mass-floor` to `--mass-ceiling`, so what it proposes can be weighed.

### The dose error is larger than the thing being optimised

Three blends aimed at the centre of percept space, from three disjoint material
sets, came back with one verdict: "all of them give the same base smell but all
different, and some facet sticking out slightly". That is the white point half
confirmed -- a shared base from different recipes is the metamer claim -- and
the facets turn out to be quantitatively predictable.

The perfumer also disclosed that the inventory's water-phase thresholds are a
compilation from published and web sources, with some approximated from
air-phase figures, and may be out by a factor of ten or more. Every script here
called them "measured". They are not, and the magnitudes below are why that is
not a pedantic correction.

Perturbing a single threshold by 10x moves a five-material blend's percept
vector by **0.4 to 0.7**. The dose optimisation works at a scale of about 0.2,
and the just-noticeable yardstick is 0.525. The input error is therefore larger
than the quantity being optimised, and larger than the smallest difference a
nose can detect. Over 4000 draws of independent threshold error spanning a
decade either way:

| blend | nominal | median under error | 90th percentile | worst 1% |
|---|---|---|---|---|
| five materials, nominal-optimised | 0.320 | 0.534 | 0.765 | 0.895 |
| five materials, `--robust` optimised | 0.350 | **0.473** | **0.592** | **0.700** |
| four materials (`mud_4_oav`) | 0.178 | 0.412 | 0.616 | 0.937 |

The median realisation sits at or above the yardstick: a blend as actually
weighed is typically as far from its target as two just-noticeably different
materials are from each other. That is "same base, a different facet every
time", reached from the bench and from the arithmetic independently.

`--robust` optimises the 90th percentile under that noise instead of the
nominal doses. It costs almost nothing nominally and improves the tail from
0.765 to 0.592, and it gets there by refusing to depend on potent traces: the
robust blend's masses span 8% to 31%, where the nominal one leaned on rose
oxide at 3.7% of the mass but 36% of the predicted odour activity. A
sensitivity ranking of the blend the perfumer actually made named hexyl
acetate, gamma undecalactone and rose oxide as its three most error-sensitive
ingredients; the perfumer, without having seen that ranking, named rose oxide
and gamma undecalactone as the facets that stuck out.

So the conclusion is about where effort belongs. A better solver cannot help.
Either the thresholds get measured, or dosing stays an ordering heuristic and
blends are chosen for insensitivity to it.

**The thresholds now have been, for the materials that matter** -- not as
thresholds, which was the wrong target. See "Measuring strength instead of
thresholds" below.

A fourth blend, the robust one, came back with the same base again and two new
facets: "calone and raspberry characters slightly sticking out". That separates
two mechanisms which had been conflated. Gamma undecalactone, which stuck out of
the previous blend, sits in the **1st percentile** of isolation -- the mean
distance from a material to its five nearest neighbours -- so it has plenty of
near-equivalents and should blend away; it was the second most
threshold-sensitive ingredient in that blend, so it was mis-dosed. Calone, rose
oxide and raspberry ketone sit at the **74th, 75th and 59th** percentiles of
isolation. They were not mis-dosed; they have no near-equivalents in this
library, so they read as themselves at any dose. A blend can fail either way and
the two want different fixes.

`--max-isolation` refuses materials above a chosen percentile. At the 60th with
seven materials the blend reaches 0.642 against the unconstrained robust
blend's 0.592, so insisting on ingredients that blend away costs about 0.05 --
cheap. Its masses also span 5% to 27% with no traces at all. The caveat is that
isolation measures this library's coverage as much as the molecule: with twenty
more marine materials, Calone would not be isolated.

And the perfumer named the shared base: it "suspiciously reminds of Dihydro
Myrcenol". By distance the model disagrees flatly -- Dihydro Myrcenol sits 1.37
from the average point, rank 233 of 340, and 1.33 from Undecavertol, which is
the 2nd nearest material to it. By **direction** it agrees: for the blend they
had just made, the centred cosine ranks Dihydro Myrcenol **9th of 340**. The
two measures answer different questions and this project had been mixing them.
Absolute distance from the centroid asks "is this neutral", and penalises
amplitude; centred cosine asks "what does this remind me of", and ignores it.
Dihydro Myrcenol has almost exactly Undecavertol's profile at two to two and a
half times the magnitude -- floral 0.90 against 0.34, citrus 0.81 against 0.37 --
so it is the loud archetype of the quiet region these blends sit in, which is
precisely what a nose would reach for a name with.

### Oversimplification works in the model's space, and is untested on a nose

Effective dimensionality of the two spaces, measured on this library:

| space | components for 80% of variance | for 90% |
|---|---|---|
| percept (138 descriptors) | 13 | 22 |
| receptor (386 ORs) | 2 | 4 |

The receptor figure is flattered -- its first component is essentially total
potency, which is why the atlas offers a "potency removed" view. But 13 to 22
effective axes for smell means a blend of k materials with k−1 free proportions
should be able to reach any point in it for quite small k. That is the formal
version of a metamer: different mixtures, one percept, exactly as different
spectra give one colour.

`reverse_accord.py --match` tests it against ground truth:

```bash
python scripts/reverse_accord.py --match Petitgrain --size 2 --owned
```

| materials allowed | proposal | centred cosine to petitgrain |
|---|---|---|
| 2 | Linalyl Acetate 60%, Neobergamate Forte 40% | **+0.948** |
| 6 | + Undecavertol, Linalyl Propionate, Calone, Hexyl Acetate | +0.975 |

The best single material in the library reaches about +0.78 of petitgrain. Two
materials reach +0.95. And the search **did not rediscover the composition**:
petitgrain is linalyl acetate 69%, linalool 22%, α-terpineol 4%, geranyl
acetate 2%, geraniol 2%, nerol 1%, and it found 1 of those 6 -- though it found
the one that is 69% of the oil, against 0.19 hits expected by chance. For the
rest it substituted Neobergamate Forte and Undecavertol: a *different* mixture
with the same predicted profile, which is a metamer claim the model cannot
check about itself.

**Open bench test:** linalyl acetate 60 / Neobergamate Forte 40 against a
petitgrain bottle. Both materials are on the shelf. Unlike every reverse-search
result before it, this one has a ground truth sitting next to it.

### The prediction that was checked, and what it got wrong

A perfumer's 12-material Acqua di Gio attempt resolved at 100%. Asked which
materials *on their own shelf* the blend was nearest in percept space, the
model answered -- before anything was smelled:

```
* +0.809  Bergamot EO            <- an ingredient, 17% of the blend
  +0.778  Lavandin EO (Grosso)     not an ingredient
  +0.671  Petitgrain EO            not an ingredient
  +0.551  Linalyl Acetate
```

Bench verdict: *"It does smell like citrusy obviously but also like petitgrain
(more) and like lavandin."*

Both non-ingredient materials, top two of 142, named in advance. The **set** was
right and the **order** was wrong -- the model ranked lavandin above petitgrain
and the nose put petitgrain first. That is the strongest external validation in
this repository, and it is worth being precise about its size: one formula, one
nose, one session, and the model's own ranking within the pair was inverted.

It also failed on the thing that formula exists for. **Neither this blend nor
the 32-material version shows the aquatic note at all**: `ozone` comes out at
z +0.15, rank 37 of 138, and there is no `marine` or `watery` descriptor in the
vocabulary. Calone is 0.47% of the blend by mass, so a mass-weighted union
cannot see it -- the same failure as nootkatone in grapefruit oil (0.1% of the
oil, and the whole grapefruit note) and the same failure as the mojito. The
model reads blends whose character is in their **bulk** and is blind to blends
whose character is in their **traces**.

Odour-activity weighting, which exists to fix exactly this, made it worse:
weighting by neat share over measured threshold flipped the ranking to woody /
cedar / amber, because Iso E Super and Ambroxan have very low thresholds and
both citrus oils were dropped for having no measured threshold at all. That is
the third formula on which OAV weighting has failed to generalise, against one
on which it helped.

### Could one fixed palette reproduce everything? No, and the reason is geometric

Choosing materials per target is one claim; a *fixed* palette that reaches any
smell is a much stronger one, and it is the olfactory version of asking whether
three phosphors make every colour. The answer transfers: under any averaging
rule a blend of a fixed palette is a convex combination of its members, so the
reachable set is the **convex hull** of the palette. A hull cannot reach outside
itself, needs at least d + 1 members to have volume in d dimensions, and is
spanned by its *extreme* members rather than its typical ones. Noisy-OR has a
gamut too, pointing the other way: mixing only raises each coordinate, so
nothing below the palette's own minimum on an axis is reachable -- you cannot
make a blend less floral than your least floral primary.

`scripts/palette.py` measures it. Palettes are grown greedily, and targets are
scored by the best blend of the palette that can be made (NNLS with the weights
constrained to a simplex), excluding palette members from their own statistics.

Targets = the 321 single materials:

| palette size | best possible k directions | those k materials, free weights | as an actual blend |
|---|---|---|---|
| 3 | 63.9% of the energy | 57.3% | median cosine 0.380, 3% within 0.90 |
| 10 | 85.4% | 80.1% | 0.739, 20% within 0.90 |
| 14 | 89.6% | 85.1% | 0.809, 26% within 0.90 |
| 25 | 95.0% | 92.3% | 0.886, **45%** within 0.90, 21% within 0.95 |
| 45 | — | — | 0.914, 62% within 0.90, **22%** within 0.95 |

Twenty-five materials capture 95% of the energy in the space and still
reproduce fewer than half the library to a cosine of 0.90. The gap between
those two numbers is exactly what non-negativity costs: a subspace is not a
hull. Doubling the palette to 45 moves the strict criterion from 21% to 22%.

The gamut's edge is chemically coherent -- phenyl acetaldehyde, methyl
anthranilate, 2-isobutyl quinoline, 2-acetyl thiazole, 2-acetyl pyridine,
veloutone. Nitrogen and sulfur heterocycles occupy descriptor combinations
nothing else approaches, so like a saturated spectral colour they can only be
reached by being in the palette.

And then the opposite result. Targets = the 28 built oils, which are mixtures:

| palette size | median cosine | within 0.90 | worst |
|---|---|---|---|
| 6 | 0.911 | 64% | 0.498 |
| 10 | 0.966 | 96% | 0.885 |
| 14 | 0.977 | **100%** | 0.907 |

Fourteen fixed materials reach every oil. The palette the search chose is
D-Limonene, Linalyl Acetate, α-Pinene, Linalool, Citronellol, Citral, Cedrol,
Menthone, Cinnamaldehyde, Eugenol, Thymol, Patchoulol and l-Menthol -- which is
very nearly the list of major essential-oil constituents, arrived at without
being told that.

**That result is partly circular and must not be quoted without the caveat.**
The oils' vectors were *built* from library molecules, so reproducing them from
a palette of library molecules is close to tautological. What it does show is
the structural asymmetry, which is not circular: single molecules are the
*vertices* of the cloud and a hull can only reach a vertex by containing it,
whereas mixtures are *interior* points -- measurably so, since blends collapse
toward the centroid -- and interiors are exactly what a hull covers. Since
"any scent" in a perfumer's sense means accords and finished formulas rather
than single chemicals, the strong claim may well hold for the things people
actually want to make, while being provably false for the things the library is
made of. Testing that properly needs target vectors for real perfumes that were
not built out of the palette, which this project does not have.

### The mixture rule cannot be chosen from any data here

`scripts/rule_bakeoff.py` scores five rules on the 19 oils whose own name is in
the model's vocabulary -- a blind question with outside ground truth, where each
rule is standardised against a null combined by that same rule.

| rule | median rank of the oil's own name (of 138) | mean | top-5 | worst |
|---|---|---|---|---|
| noisy-OR | 2.0 | 5.0 | 14/19 | 34 |
| weighted mean | 2.0 | 5.1 | 14/19 | 34 |
| logit mean | 2.0 | 5.8 | 13/19 | 38 |
| max | 3.0 | 4.9 | 14/19 | **18** |
| **dominant ingredient** | **2.0** | 6.1 | **14/19** | 34 |

Nothing separates. Paired sign tests against noisy-OR give p = 0.057 for max
and p > 0.2 for everything else. The finding that matters is the last row:
**taking the single biggest ingredient and ignoring the rest does as well as
every mixture rule.** That is not a point in favour of simplicity, it is a
verdict on the test set -- these oils are one or two constituents plus
decoration, so no rule can distinguish itself, and it explains why the earlier
comparison on 30 formulas also produced a margin that depended on which
formulas were included.

Where the rules *do* differ is instructive: `max` wins exactly on the two oils
whose identity sits in minor constituents (grapefruit, rank 1 against 9;
neroli, 2 against 34) and loses on the bulk-carried ones.

Weighting by potency does not fix it. Odour-activity weighting generalises as a
one-parameter family, `w ∝ (share / threshold)^n`, with n = 0 equal weights and
n = 1 full OAV. On the five oils with measured thresholds for most of their
constituents the self-check rank degrades monotonically with n (median 2 → 10),
and **equal weights beats mass weighting** (5/5 in the top five against 3/5,
mean rank 2.6 against 8.8) because it is the crudest possible way of
up-weighting a trace. Five oils is a hint, not a result.

One correction to an assumption that was load-bearing for a while: **noisy-OR
as implemented is not trace-sensitive.** It scales each ingredient by
`w / w.max()`, so an ingredient at 3% of a four-material blend enters at about
9% of its own probability. Across 40,000 random blends there was not a single
case where noisy-OR carried a minority descriptor that the weighted mean
buried. The only trace-sensitive rule here is `max`.

So the fork worth taking to a bench is max against the weighted mean, and it is
the same question every trace failure in this project turns on: does three
percent of something with a strong character impose that character?
`scripts/discriminating_blends.py` searches the owned shelf for blends where the
two rules disagree about a descriptor a perfumer would name unprompted, guards
the proportions against the perfumer's own maximum use levels so the sheet is
makeable, writes one CSV per blend, and records every prediction to
`predictions.json` **before** anything is smelled.

### What is missing, in order of how soon it bites

1. **Time.** A perfume is a trajectory; these vectors are static. There is no
   volatility axis anywhere in the model, so a blend matched at t=0 says
   nothing about t=1h -- which is the first thing a simplified formula loses.
2. **Dose.** Detection thresholds span six orders of magnitude. A metamer
   matched at one concentration is not one at another, and the weights
   `reverse_accord` prints are proportions, not doses.
3. **Suppression.** Real mixtures are sub-additive. Both combination rules here
   (a union and a mean) were chosen for their shape, not fitted to mixture
   data, and the choice moves a blend's position by a factor of four.
4. **Hedonics.** No axis for richness, naturalness or whether something is
   good. The perfumer's "fresh but not as nice as the real deal" is invisible
   to every number here, and the usual explanation -- that richness comes from
   dozens of components below any weight the model notices -- is precisely the
   blind spot above.

## Measuring strength instead of thresholds

A detection threshold is the wrong thing to measure for this project, and
measuring one properly is an ascending forced-choice staircase per material:
157 figures, an afternoon each, and nobody does 157 afternoons. What the dose
conversion needs is not a threshold but **relative strength at working
concentration** -- how much smell a material delivers per unit of mass -- and
there is a much cheaper way to get it.

The perfumer's own maximum use levels are already an inverse strength estimate
made by a nose. If those levels were right, a blotter of each material
prepared *at its own use level* would smell about equally strong. They are not
right, so the blotters do not match, and the mismatch is exactly the correction
wanted. That turns an eight-order-of-magnitude measurement into a small one:
every judgement compares blotters already within a factor of a few of each
other, which is the range a nose compares well in.

`scripts/threshold_worklist.py` chose the materials by influence rather than by
suspicion: for every formula in the repository, each ingredient's threshold was
perturbed tenfold and the resulting shift in the blend's percept vector
measured. The top 20 carry about two thirds of everything the dose conversion
rests on. `scripts/strength_worksheet.py` turns that list into a bench sheet,
`scripts/apply_strength.py` turns the filled sheet into multipliers, and
`scripts/strength.py` applies them.

### What came back

Three sessions, 62 judgements, 23 materials. One bin is a factor of three.

| | |
|---|---|
| per-judgement sd, from 10 duplicate pairs | **0.89 bins**, i.e. one judgement alone is worth a factor of 2.7 |
| the same blotter judged again a day later | 46% identical, **88% within one bin**, r = +0.65 |
| of the spread between materials | **68% is real**, the rest is judging noise |

The largest corrections, after shrinkage:

| material | x strength | |
|---|---|---|
| Aldehyde C-8 | 3.81 | dose less |
| Cinnamaldehyde | 2.91 | dose less |
| Aldehyde C-10, Methyl Benzoate, Strawberry Furanone | 2.03 | dose less |
| Calone 1951, Undecavertol | 0.24 | dose more |
| Hedione, Ethylene Brassylate | 0.28 | dose more |

**Two of these confirm bench notes written months earlier, before anything was
measured.** Methyl benzoate came back at x2.03, and it is the material that
overpowered the third mud blend and forced that experiment to be reframed.
Cinnamaldehyde came back at x2.91, and the perfumer had written "I have
cinnamaldehyde only in 10% and it's already fairly strong". Neither note was in
front of the nose during the session. That is the closest thing to external
validation this measurement can get.

**Calone at x0.24 separates two things the project had been conflating.**
Calone stuck out of a blend, and the natural reading was that it is powerful.
It is not: it is *distinctive*. It sits at the 74th percentile of isolation,
with no near-equivalents in the library, so it reads as itself at any dose
while delivering little smell per gram. Sticking out of a blend and being
strong are different properties, and until this was measured the dose
conversion had no way to tell them apart.

### Two statistics that were wrong on the way

*Comparing sessions by blotter number.* Blotter 3 is Isobornyl Acetate on the
first sheet and Hedione on the re-test sheet, which renumbers. The
"same blotter judged again" table was comparing different materials and
reporting 1.20 bins of invented disagreement. Sheets are now matched on the
material name as well as the number.

*Vetoing a material whose judgements spanned two bins.* The span of a set of
numbers grows with how many of them there are, so this rule punished a material
for being measured more often. Four materials were re-tested, came back
repeating themselves exactly within the new session -- Hedione twice at `--`,
Ethylene Brassylate twice at `--`, Anisaldehyde twice at `0` -- and the extra
evidence made all four look *worse* by that rule. A statistic that degrades as
evidence accumulates is the wrong statistic.

It was replaced by the standard treatment: the duplicate pairs give a pooled
per-judgement noise, each material gets a standard error from its own count of
judgements, and each estimate is shrunk toward no-correction in proportion to
that error. A consistent pair keeps most of its correction; a contradictory one
is pulled back toward its compiled figure, which is the honest thing to do with
a measurement that did not converge. Nothing is discarded and no single noisy
pair can move a formula.

### How it enters a dose

Odour activity is mass over threshold, so "delivers more smell per gram than
its use level implies" is the same statement as "has a lower effective
threshold":

    effective threshold = compiled threshold / strength multiplier

That is the only place the measurement is used. It does not replace the
compiled figures; it corrects the 23 a nose has now checked, and the other 102
stand unchanged. `scripts/strength.py` is shared by the CLI and by the exporter
that builds the browser page, so a formula dosed in one and dosed in the other
get the same answer. The simplifier names the corrected materials under its
dose table, and the atlas is unaffected: it displays measured thresholds on the
material card but never uses them to place a point.

### What is still missing

The sheet measures strength at one concentration on a blotter in still air.
It says nothing about volatility, so nothing here knows that a blend will smell
different in an hour. The three remaining materials on the influence list with
no measurement are the ones whose use levels the inventory does not record.

## Layout

```
src/orvector/
  config.py       frozen constants, hashes, paths
  upstream.py     controlled import of the vendored upstream modules
  panel.py        panel construction, repair audit, checksum, TSV I/O
  embeddings.py   ESM-2 cache keyed by sequence + encoder config
  model.py        checkpoint inspection and strict load
  inference.py    featurisation and the prediction loop
  cli.py          prepare / predict / info
  store.py        the library store: molecules, logits, scores, provenance

scripts/          everything measured in this document, one script per claim
  train_fp_v3.py            the percept model that colours the atlas
  build_naturals.py         essential oils from GC compositions, self-checking
  predict_formula.py        a formula's predicted character and neighbours
  try_formula.py            paste a formula, get a local atlas with it on it
  reverse_accord.py         a descriptor target or a material to reproduce
  palette.py                the gamut of a fixed palette
  rule_bakeoff.py           five mixture rules against the oils
  mud.py / olfactory_white.py   the white point, and how few materials reach it
  musk_test.py              do the four musk chemotypes recognise each other
  musk_receptor.py          OR5AN1 does; the global distance discards it
  receptor_metric.py        can reweighting get it back? pre-registered, no
  receptor_importance.py    a class detector is not a similarity axis
  bench_simplify.py         the simplifier against the hand heuristic
  discriminating_blends.py  blends whose smell would decide the mixture rule
  export_viewer_data.py / build_viewer.py          the atlas
  export_simplify_data.py / build_simplify.py      the simplifier
  check_viewer_js.py        lexes a built page before it ships

viewer/           template.html (atlas), simplify.html (simplifier)
data/
  inventory/      the perfumer's materials, use levels, threshold figures,
                  and strength.csv, the measured corrections to them
  naturals/       sourced GC compositions, with what each omits and why
  formulas/       published formulas, and the queries run against them
  percept/        GS-LF and Keller label sets, which nothing here is fitted to
library/input/    what went into the store, one CSV per addition
artifacts/        receptors.tsv, manifest.json, the library stores, esm_cache/
vendor/olfaction  pinned upstream checkout
vendor/zenodo     checkpoint and reference logits
```

`artifacts/panel_embeddings.npy` is ~0.9 GB (386 × 443 × 1280 float32) and is
memmapped, never loaded whole.
