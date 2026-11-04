# Adding a material to the library

The recipe, end to end, from one molecule or one cited natural to a shipped
`materials.db`. Written after doing it for Vertofix Coeur, Styrax Resinoid and
Benzoin Siam Resinoid (commit history around November 2026); every command
below is the one that was actually run. The precedent commit for naturals
rebuilt from constituents is `af81a65` (oakmoss, labdanum, castoreum).

There are two kinds of addition, and they share every step except 3:

* **A single molecule** (Vertofix Coeur): one CSV row, one percept row, done.
* **A natural rebuilt from a cited composition** (an absolute, resinoid or oil
  the library does not hold as a bottle): add its signature constituents as
  molecules first, then compose the natural from them by weight.

The library is a store, not a database: `library/input/expanded_materials.csv`
is the molecule set, `artifacts/percepts_fp3.json` is its percept model output,
`artifacts/naturals.json` is the naturals store, and
`artifacts/simplify_data.json` is the build product the perfumer service and
the simplifier page both read. `simplify_data.json` is gitignored — it is
regenerated in seconds from the stores — so the *commits* are the CSV, the
two JSON stores and the retrained `percepts_fp3.json`.

## 0. Before you start

Work in the `olfactory-embeddings` checkout (the `.venv` has torch + rdkit +
CUDA; the percept retrain below takes about two minutes on a GPU). The
perfumer service repo (`C:\Valle\dev\perfumer`) reads this checkout
**read-only** through `tools/build_db.py`; nothing there ever writes back.

Check the working tree is clean apart from what you intend to change, and
check the inventory (`data/inventory/inventory.csv`) for the material's
shelf name first: a library name that matches an inventory row exactly
inherits its use level, IFRA cells, family, logP, boiling point and
**compiled** detection threshold for free. "Vertofix Coeur" and
"Benzoin Siam Resinoid" match; "Styrax Resinoid" does not match the shelf's
"Styrax Resinoid EAL, Wild", so that row carries an *imputed* threshold and
no use level — the same situation "Oakmoss Absolute" has been in since
`af81a65` (the shelf row there is "Oakmoss IFRA43").

## 1. Molecules: `library/input/expanded_materials.csv`

One row per molecule, three columns: `name,smiles,source`. The name is the
canonical name every alias resolves to; the SMILES is the structure; the
source column is provenance in one line, written for a reader who will want
to know how certain the entry is.

Conventions that matter:

* A material added only so a natural can be built from it is tagged
  `natural-rebuild` and nothing else — the citations live in the naturals
  store, not here.
* A real formulating material gets its CAS and one honest sentence, e.g.
  `Vertofix Coeur,CC1CCC2C13CC(C2(C)C)C(=C(C3)C(=O)C)C,"CAS 32388-55-9;
  methyl cedryl ketone (acetylcedrene), the skeleton the IFF grade is
  mostly an isomer mix of; in the perfumer's inventory; structure from
  PubChem CID 107065, stereochemistry unspecified because the percept
  fingerprint is stereo-blind"`.
* Stereochemistry may be left unspecified — the percept model reads ECFP4
  radius-2 Morgan bits, which cannot see it — but say so in the source
  column, the way the Cosmone and Hedione rows do.

**Gate: RDKit must accept every SMILES.** `featurise` drops rows that do not
parse, *silently*, so a typo costs a molecule without an error:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'scripts'); from train_fp_percept import featurise; import csv, io; rows=list(csv.DictReader(io.open('library/input/expanded_materials.csv',newline='',encoding='utf-8-sig'))); X,ok=featurise([r['smiles'] for r in rows]); print(len(rows), len(ok), len(ok)==len(rows))"
```

For styrax and benzoin this meant five new rebuild molecules: Cinnamic Acid,
3-Phenylpropyl Cinnamate, Styrene, Coniferyl Benzoate, Benzoic Acid — every
structure taken from PubChem and checked against its formula.

## 2. Retrain the percept ensemble

```powershell
.\.venv\Scripts\python.exe scripts\train_fp_v3.py 2>&1 | Tee-Object -FilePath artifacts\fp3_413.log
```

This is Pipeline 3 in the README: a 2048-bit ECFP4 + RDKit-descriptor MLP,
5-fold CV with a 5-model ensemble per fold, then a final refit that predicts
the 138 GS-LF descriptors for every CSV row. It writes
`artifacts/percepts_fp3.json` (committed — it is the reproducibility-critical
output) and prints the number that gates the step:

```
[v3] 4983 training molecules x 138 labels, 413 library materials
[v3] out-of-fold mean ROC-AUC: 0.8672
```

**Gate: the OOF ROC-AUC must be in line with the baseline.** The AUC is
measured on the GS-LF training set, which the library does not touch, so
adding prediction-set molecules cannot move it by design; 0.8672 has come
out of every v3 run since the 301-material library, and a materially
different number means the training data or config changed, not the
library. Also check the molecule count in the log line — 407 → 413 here —
because a silent `featurise` drop shows up as a count that did not move.

The run takes ~2 minutes on CUDA, ~10x that on CPU.

## 3. Naturals: `artifacts/naturals.json`

Single molecules skip this step.

A natural is a *composition-weighted reconstruction*: published GC analyses
give the percentages, the library's own percept rows give the vectors, and
the entry is honest about both. Two entry kinds live in the same
`naturals` object:

* **Built oils** (39 of them) are generated by `scripts/build_naturals.py`
  from `data/naturals/compositions.json` and carry percept + receptor
  vectors, self-checks and lift tables.
* **Reconstruction-only entries** — Birch Tar, Oakmoss, Labdanum, Castoreum,
  Styrax, Benzoin — are added by hand, because they are absolutes and
  resinoids with no receptor vector, no atlas point and no self-check to
  run. `af81a65` added the first four as metadata only; since the simplifier
  export began to need a `percept` for every entry they also carry a
  composed percept vector and `"atlas": false` (the same semantics as a
  trimmed oil: resolvable in formulas, available to the simplifier, no
  point on the atlas).

The hand-entry format, with the styrax entry as the example:

```json
"Styrax Resinoid": {
  "composition_used": {
    "Cinnamyl Cinnamate": 0.5703,
    "3-Phenylpropyl Cinnamate": 0.2072,
    "Cinnamic Acid": 0.1347,
    "Cinnamyl Alcohol": 0.0387,
    "Benzyl Cinnamate": 0.038,
    "Styrene": 0.0079,
    "Vanillin": 0.0032
  },
  "coverage": 1.0,
  "omitted_from_library": [ "prose notes on what is NOT voiced", ... ],
  "source": "citations with every published value and the medians used",
  "note": "how the percept was composed and what the entry may not be used for",
  "percept": [ 138 values ],
  "atlas": false
}
```

**The composition must come from published GC analyses, and where they
disagree, use the median.** The convention the store uses: list every
published value (a range contributes its midpoint) per constituent in the
`source` string, take the median as the working figure, renormalise the
in-library constituents to sum to 1.0 in `composition_used`, and put the
unvoiceable fraction (storesin, triterpene acids, minor esters) in
`omitted_from_library` as prose. Cite by author and year; name the
material you analysed (balsam vs distilled oil is a 70% styrene
difference); say "reconstruction only — not for use" in the source string.

**The percept is the composition-weighted mean of the constituents' fp3
rows** — the same rule the simplifier page states for its own blending
("the blend is a weighted mean of its parts") and the same arithmetic the
`af81a65` rows were composed with. Compute it after the retrain, from the
`percepts_fp3.json` you just wrote:

```python
w = np.array(list(comp.values()), float); w /= w.sum()
V = np.stack([P[n] for n in comp])            # P: name -> fp3 row
percept = [round(float(v), 4) for v in (w[:, None] * V).sum(0)]
```

Sanity-check the composed vector before committing it: styrax came out
balsamic 0.94 / cinnamon 0.43, benzoin balsamic 0.76 / vanilla 0.39, and a
vector that does not say the material's name is a composition error, not a
model quirk. Verify the file still parses and every entry has a 138-long
percept before moving on.

**Landmine:** `scripts/build_naturals.py` rebuilds this file from
`compositions.json` from scratch and will wipe every hand entry. Do not run
it after adding one; if an oil's GC profile genuinely needs rebuilding,
re-apply the reconstruction-only entries afterwards.

## 4. The simplifier payload

```powershell
.\.venv\Scripts\python.exe scripts\export_simplify_data.py
```

This regenerates `artifacts/simplify_data.json` (gitignored, rebuilt in
seconds) from the stores: molecules + oils with their vectors, isolation
percentiles, the alias table, use levels, and the threshold column. It
prints its own summary; the numbers to read are:

```
458 materials (45 of them oils), 458 with a threshold
138 descriptors, 279 aliases, yardstick 0.485
9 materials may never be proposed; 34 are offered but unverified
```

If the export crashes on `oils[o]["percept"]`, a naturals entry is missing
its composed vector — see step 3.

Aliases: if the material has spellings a pasted formula will use, add them
to the `ALIAS` dict in `scripts/parse_formulas.py` (lowercase spelling →
canonical name). Vertofix needed two:

```python
"vertofix coeur": "Vertofix Coeur",
"methyl cedryl ketone": "Vertofix Coeur",
```

Naturals do not need aliases — `classify` routes resinoid/absolute names to
`match_oil`, which resolves them (and grade-qualified spellings like
"Styrax Resinoid EAL, Wild") by head word.

## 5. Thresholds

`export_simplify_data.py` calls `scripts/thresholds.py`, which assembles one
threshold column for everything: **compiled** figures from the inventory's
`water_threshold_ppb`, **kfo** medians from the Keller et al. atlas, and
**k-NN imputation** (5 neighbours in percept space, leave-one-out median
error ~3x) for the rest. Oils never anchor anyone else's figure. A row
with no figure at all cannot be dosed, so every row gets one.

The check for the new rows is that the number is *sensible for the
material*, and that the provenance label is the honest one:

* Vertofix Coeur 100 ppb `compiled` (its own inventory row),
* Benzoin Siam Resinoid 50 ppb `compiled` (same),
* Styrax Resinoid 684 ppb `imputed` — anchored on the cinnamate family
  (cinnamyl alcohol 500, benzyl isoeugenol 5000): a heavy-ester resinoid
  genuinely is a weak smeller, and the label says the figure is an
  interpolation, not a measurement.

## 6. NEVER_PROPOSE (blocking a material from being proposed)

The `NEVER_PROPOSE` dict in `scripts/export_simplify_data.py` is the hard
block: a material it names resolves in pasted formulas but is never
*proposed* by the simplifier or the perfumer engine. The bar is a
regulatory prohibition, a supplier saying "not for fragrance use", or —
established by three commits in November 2026 — a *trace volatile that is
not purchasable as a formulating material* (3-methylbutanal, the leaf
aldehydes, heptan-2-one, and now Styrene, which enters the library only as
a styrax constituent and is a plastics monomer as a bottle).

The flow, end to end, because it spans both repos:

1. `NEVER_PROPOSE["Styrene"] = "trace volatile monomer of styrax, not a
   formulating material"` in `scripts/export_simplify_data.py`;
2. the export writes it into the row as `noUse`;
3. the perfumer's `tools/build_db.py` turns it into a notes string
   `"not for use: <reason>"`;
4. the perfumer's `materials.py::_parse_notes` reads that marker into
   `not_for_use`, and the engine keeps such materials out of its candidate
   pool while still resolving them in user formulas.

## 7. The perfumer database

Commit the upstream changes **first** — `build_db.py` records the source
commit and dirty flag in the `meta` table, and the db should name the
commit it was built from. Then, from `C:\Valle\dev\perfumer`:

```powershell
python tools/build_db.py --source C:/Valle/dev/olfactory-embeddings --out materials.db
python tools/fetch_cas.py         # PubChem by InChIKey; cached; safe offline
python tools/fetch_ifra.py        # merges tools/ifra_limits.csv, fill-or-tighten
python tools/fetch_longevity.py   # imports tools/longevity.csv (a stub today)
python tools/verify_db.py
```

`verify_db.py` prints counts, per-column null rates and vector sanity; the
counts move with the library (435 → 439 with the af81a65 absolutes → 458
with vertofix, styrax, benzoin and their constituents). The two assertions
in `tests/test_engine.py` that pin the material count must move with them
(`== 458` and `z.shape == (458,)` — update the comment too).

**Gate:** `python -m pytest tests/ -q` in the perfumer repo — 66 tests,
all green, against the real materials.db.

`materials.db` is **gitignored** and never committed. The only perfumer
commits a library change produces are code ones: the test count assertion
above, or a tools/ change.

## 8. Deploy (for reference — do not do this without review)

`materials.db` ships separately from code; see
[`perfumer/docs/DEPLOYMENT.md`](../perfumer/docs/DEPLOYMENT.md) for the
full procedure. The two commands, after review:

```powershell
scp materials.db root@phonecallfor.me:/opt/perfumer/materials.db
ssh root@phonecallfor.me "systemctl restart perfumer"
```

Code changes (if any) go with `.\deploy.ps1` from the perfumer repo, which
syncs the *.py files + tools/ and restarts the service. The upstream
stores themselves deploy only through this db — no olfactory-embeddings
file is read by the server.

## Acceptance gates, in one list

1. Every new SMILES parses in RDKit (`featurise` keeps all rows; the
   retrain log reports the expected molecule count).
2. OOF ROC-AUC in line with the v3 baseline (0.8672; the number is
   training-set-only, so it should not move).
3. `naturals.json` parses; every entry has a 138-long percept; composed
   vectors say the material's name (balsamic/cinnamon for styrax,
   balsamic/vanilla for benzoin).
4. `export_simplify_data.py` runs clean and the new rows have thresholds
   with honest provenance labels.
5. `tools/verify_db.py` passes in the perfumer repo with the expected
   counts.
6. `python -m pytest tests/ -q` green in perfumer (66 tests).
7. Nothing pushed, nothing deployed, until reviewed.
