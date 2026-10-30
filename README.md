# orvector

SMILES → predicted responsiveness score for every functional human olfactory
receptor, as a fixed-order vector.

Scores are continuous sigmoid outputs of the model's raw logits. They are not
binned, thresholded, or normalised, and they are not concentration-specific
activation strengths.

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

The pattern is worth stating: every number that looked good on its own
cross-validation failed when checked against something it had not been fitted
to. Bench observations caught four of these before any metric did.

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
chemotype and has no concept uniting them. Expected, given the deorphanization
gap: OR5AN1 is the known musk receptor and its learned signal is weak, so there
is no training data from which "musk" could be learned as one thing.

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
artifacts/        receptors.tsv, manifest.json, esm_cache/, panel_embeddings.npy
vendor/olfaction  pinned upstream checkout
vendor/zenodo     checkpoint and reference logits
```

`artifacts/panel_embeddings.npy` is ~0.9 GB (386 × 443 × 1280 float32) and is
memmapped, never loaded whole.
