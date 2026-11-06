# Adding a material to the library

The recipe, end to end, from one molecule or one cited natural to a shipped
`materials.db`. Written after doing it for Vertofix Coeur, Styrax Resinoid and
Benzoin Siam Resinoid, then for Tobacarol, Guaiacwood EO and Myrrh EO (commit
history around November 2026); every command below is the one that was actually
run. The precedent commit for naturals rebuilt from constituents is `af81a65`
(oakmoss, labdanum, castoreum).

There are two kinds of addition, and they share every step except 4:

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

Work in the `olfactory-embeddings` checkout. The `.venv` has torch + rdkit and
CUDA; the percept retrain in §2 uses the GPU and takes about two minutes there,
while the receptor stores in §3 run on CPU because DGL is the CPU build — see
that section. The perfumer service repo (`C:\Valle\dev\perfumer`) reads this
checkout **read-only** through `tools/build_db.py`; nothing there ever writes
back.

Check the working tree is clean apart from what you intend to change, and
check the inventory (`data/inventory/inventory.csv`) for the material's
shelf name first: a library name that **resolves** to an inventory row
inherits its use level, IFRA cells, family, logP, boiling point and
**compiled** detection threshold for free.

The join is `parse_formulas.library_name` — the alias table, the
normalisation in `classify`, and the head-word match a pasted formula gets —
so "Styrax Resinoid" reaches the shelf's "Styrax Resinoid EAL, Wild" and
"Oakmoss Absolute" reaches "Oakmoss IFRA43". **This used to be exact string
equality**, and the cost was silent: 35 of 166 shelf names matched nothing, so
every oil except two ("Benzoin Siam Resinoid" and "Tangerine EO", spelled the
same way in both files) fell back to an imputed threshold with no dosing
ceiling. If a new material's shelf row still does not attach, add the spelling
to `ALIAS` in `scripts/parse_formulas.py` — the one alias table there is — and
not a second one.

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
.\.venv\Scripts\python.exe scripts\train_fp_v3.py 2>&1 | Tee-Object -FilePath artifacts\fp3_retrain.log
```

This is Pipeline 3 in the README: a 2048-bit ECFP4 + RDKit-descriptor MLP,
5-fold CV with a 5-model ensemble per fold, then a final refit that predicts
the 138 GS-LF descriptors for every CSV row. It writes
`artifacts/percepts_fp3.json` (committed — it is the reproducibility-critical
output) and prints the number that gates the step:

```
[v3] 4983 training molecules x 138 labels, 427 library materials
[v3] out-of-fold mean ROC-AUC: 0.8672
```

**Gate: the OOF ROC-AUC must be in line with the baseline.** The AUC is
measured on the GS-LF training set, which the library does not touch, so
adding prediction-set molecules cannot move it by design; 0.8672 has come
out of every v3 run since the 301-material library, and a materially
different number means the training data or config changed, not the
library. Also check the molecule count in the log line — it must equal the CSV
row count, 427 today — because a silent `featurise` drop shows up as a count
that did not move.

The run takes ~2 minutes on CUDA, ~10x that on CPU.

## 3. Receptor vectors: `artifacts/library/`

The atlas places a molecule by its 386 receptor responses, not by its percept.
A molecule in the CSV and in `percepts_fp3.json` but not in the store has no
point on the atlas, and nothing anywhere says so — the page simply draws fewer
points. The scorer is append-only, so growing the library costs the new
molecules rather than all 427 × 386 pairs, and the panel embeddings come from
`artifacts/panel_embeddings.npy`, so ESM-2 does not re-run.

One store, one space: the MPNN store was removed in November 2026 and the GCN
is the atlas's chemistry layout. The removal took no dependency with it —
`dgl`, `dgllife`, `rdkit` and `torch` are the GCN's dependencies as much as
they were the MPNN's, and both encoders came from the same MolOR install.

```powershell
.venv/Scripts/python.exe -m orvector library add --csv library/input/expanded_materials.csv --device cpu
```

**Gate: the store must hold every row of the CSV.** Nothing in this repository
compares them, and the failure is silent. On 4 November 2026 the stores were
left a day behind the CSV: 31 molecules — Vertofix Coeur, Tobacarol, Guaiol,
the nine myrrh furanosesquiterpenes — were absent from a page that had been
built, checked and reviewed, and it looked correct.

```powershell
.venv/Scripts/python.exe -c "import csv,io,json;n=len(list(csv.DictReader(io.open('library/input/expanded_materials.csv',newline='',encoding='utf-8-sig'))));print('csv',n);print('store',json.load(open('artifacts/library/provenance.json'))['n_molecules'])"
```

Both must agree — 427 today. 31 new molecules took 129 s.

`--device cpu` is not a compromise, it is the only option: DGL in `.venv` is
the CPU wheel, so any graph moved to the GPU fails with `Device API cuda is not
enabled` however CUDA-ready torch reports itself. MolOR is a DGL model.

## 4. Naturals: `artifacts/naturals.json`

Single molecules skip this step.

A natural is a *composition-weighted reconstruction*: published GC analyses
give the percentages, the library's own vectors give the rest, and the entry
is honest about both.

**The store is generated, one way, for all 47 oils.** `data/naturals/
compositions.json` is the spec and `scripts/build_naturals.py` writes
`artifacts/naturals.json` from it. Never edit the store: an entry written by
hand is an entry nothing regenerates. Eight of them
had accumulated by November 2026 — Birch Tar, Oakmoss, Labdanum, Castoreum,
Styrax, Benzoin, Guaiacwood, Myrrh — carrying a percept composed by a
*different rule* from the other 39, and nothing compared the two. When they
were moved into the spec and rebuilt, myrrh's descriptor row moved 0.48 and
castoreum's 0.57. The rule below is the one that survived; there is now one
code path, one rule, and no hand entries.

The spec entry, with styrax as the example:

```json
"Styrax Resinoid": {
  "source": "citations: every published value, and the median used",
  "note": "which rule composed the vectors, and what the entry may not be used for",
  "composition": {
    "Cinnamyl Cinnamate": 57.03,
    "3-Phenylpropyl Cinnamate": 20.72,
    "Cinnamic Acid": 13.47,
    "Cinnamyl Alcohol": 3.87,
    "Benzyl Cinnamate": 3.8,
    "Styrene": 0.79,
    "Vanillin": 0.32
  },
  "omitted": "prose: what the reconstruction cannot voice, and why"
}
```

`composition` is in the scale the analysis used — percentages, mostly — and
is renormalised over the constituents the library actually holds. Two optional
keys: `coverage`, a citation statistic (the fraction of one published
chromatogram those constituents resolve; without it the build records its own
resolved fraction, which is a different and less interesting number), and
`default_for`, the words an unqualified formula means (`["lavender"]`).
Everything else in the store — `composition_used`, `percept`, `receptors`,
`lift_top`, `nearest_materials`, `nearest_cosine`, `collapsed_onto`,
`self_check`, `atlas` — is computed. Do not write it.

**The composition must come from published GC analyses, and where they
disagree, use the median.** List every published value (a range contributes
its midpoint) per constituent in `source`, take the median as the working
figure, and put the unvoiceable fraction (storesin, triterpene acids, minor
esters) in `omitted` as prose. Cite by author and year; name the material you
analysed (balsam vs distilled oil is a 70% styrene difference); say
"reconstruction only — not for use" in `source`.

**Both vectors are the composition-weighted mean over the constituents**, in
both spaces, and that is `--compose weighted-mean`, the default.
`build_naturals.py` takes the constituents' `percepts_fp3.json` rows for the
percept and their MolOR scores for the receptor vector, and averages them by
the stated fractions. An oil therefore *sits among its own constituents* rather
than outside them.

The rule was not chosen on a score, and it is worth saying so plainly. On
character, `rule_bakeoff.py` cannot separate the mean from the union or even
from the dominant ingredient (p > 0.2). What decided it was the rendered map:
under the union an oil is pushed outwards by construction, and Cedarwood EO was
the single most extreme point on the published atlas (percept sum 9.84 against
a material median near 5, 152.8 total receptor activation against 5.5).
Recomposed by the mean it reads 4.18 and sits with cedrol and cedrene, which is
what a perfumer looking at the map expects. The union stays reachable as
`--compose noisy-or` for the comparison.

Two rules remain, and they answer different questions rather than disagreeing:

| object | rule | why |
|---|---|---|
| an **oil** (a material you can buy) | weighted mean of its parts | it is a thing, and its vector should describe the thing |
| a **blend's position** on the map | mean of logits | the map is drawn in log-odds, so the centre and the placement must be computed there |
| the **white point** | mean of logits | it is the map's origin, not a material: 0.3–1.1 from the cloud's centre, against 5–14 for the probability mean |

**Then build the store:**

```powershell
.venv/Scripts/python.exe scripts/build_naturals.py --store library --max-nearest-cosine 1.01
```

`--max-nearest-cosine 1.01` is the current policy: an oil whose percept sits
within 0.98 of a single library material used to be trimmed off the atlas as a
point that repeats one already drawn, which left twelve of them invisible —
sweet orange at cosine 0.9998 to D-Limonene, and Birch Tar, Oakmoss and Styrax
with it. A material the perfumer can smell and cannot find on the page is the
worse of the two failures, so every oil carries a point and the card says when
a build sits on top of its own constituent.

**Gates.** `47 built, 47 on the atlas`, nothing `TRIM`-ed; every entry carries a
138-long `percept` and a 386-long `receptors`. Verify the composed vectors still say the
material's name: styrax came out balsamic 0.94 / cinnamon 0.43, benzoin
balsamic 0.76 / vanilla 0.39, and a vector that does not is a composition
error, not a model quirk. Under the mean these numbers are lower than they were
under the union — that is the rule doing its job, not the composition breaking;
check the *ranking* of the descriptors, not the absolute value.

When you add one oil, the other 46 are recomputed too — the build is a pure
function of the spec, the percept model and the receptor stores. Compare the
file against the previous one and expect movement only if one of those inputs
moved; on this rebuild all 39 built oils shifted, because their stored vectors
had been computed while the receptor store was missing 31 molecules, so
constituents that could not be scored were silently left out of the weights.
That is the class of drift a generated store catches and a hand-maintained one
hides. See [`DEPLOY_MOLOR.md`](DEPLOY_MOLOR.md) for the deploy.

## 5. The simplifier payload

```powershell
.\.venv\Scripts\python.exe scripts\export_simplify_data.py
```

This regenerates `artifacts/simplify_data.json` (gitignored, rebuilt in
seconds) from the stores: molecules + oils with their vectors, isolation
percentiles, the alias table, use levels, and the threshold column. It
prints its own summary; the numbers to read are:

```
143 compiled thresholds, 435 imputed (leave-one-out median error 3.63x)
750 materials (47 of them oils), 149 owned, 750 with a threshold
138 descriptors, 289 aliases, yardstick 1.099
56 materials may never be proposed; 34 are offered but unverified
```

The materials count is the one to watch: it must equal the CSV rows plus
every oil in the naturals file — 703 + 47 today — because that is what the
perfumer's `materials.db` will hold. `compiled` is the inventory meeting the
library by name (§6); if it drops, the join is failing, not the data.

`yardstick` is the **median distance from a material to its nearest neighbour**
in percept space, defined once in [`scripts/yardstick.py`](scripts/yardstick.py).
It is a *ruler*, not a measurement: it is what "close enough" is read against,
and it is the scale at which two materials are as similar as typical neighbours
are — not a perceptual threshold, and not «what a random blend achieves» (this
doc said that for months; it was never true of the code).

It is **model-dependent**, which is the trap. The same formula over the same
library gives 0.506 on fp3 and 1.221 on fp4, because `pos_weight` lifts the mean
probability from 0.047 to 0.164 and everything sits further apart. The 0.451 long
quoted here was an fp3-era figure that survived the fp4 adoption (`b8d08f9`); the
formula has not changed since the first commit. So: quote the number the export
prints, and never hard-code it — a literal 0.439 in `percept_blind_spots.py` did
survive the model change and made that script's verdicts ~2.5× too strict.

If the export crashes on `oils[o]["percept"]`, a naturals entry is missing
its composed vector — see §4.

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

## 6. Thresholds

`export_simplify_data.py` calls `scripts/thresholds.py`, which assembles one
threshold column for everything: **compiled** figures from the inventory's
`water_threshold_ppb`, **kfo** medians from the Keller et al. atlas, and
**k-NN imputation** (5 neighbours in percept space, leave-one-out median
error ~3x) for the rest. Oils never anchor anyone else's figure. A row
with no figure at all cannot be dosed, so every row gets one.

The inventory side of that column goes through `parse_formulas.library_name`,
same as §0. It did not, and the count said so once it was fixed: `compiled`
went 127 → 142 while `imputed` fell 222 → 208, because 15 materials had a
published figure in the sheet the exact-match join could not see. A change in
that pair of numbers is now worth reading; it was previously invisible.

The check for the new rows is that the number is *sensible for the
material*, and that the provenance label is the honest one:

* Vertofix Coeur 100 ppb `compiled` (its own inventory row),
* Benzoin Siam Resinoid 50 ppb `compiled` (same),
* Styrax Resinoid 684 ppb `imputed` — anchored on the cinnamate family
  (cinnamyl alcohol 500, benzyl isoeugenol 5000): a heavy-ester resinoid
  genuinely is a weak smeller, and the label says the figure is an
  interpolation, not a measurement.

**The KFO tier is generated, not curated, and nothing regenerates it.** It is
`scripts/import_kfo.py` read against the published workbook, so it goes stale
the moment the library grows. After the 278-molecule catalogue commit the sheet
had never been re-read, so the batch was imputed over a 180-material match set.
Re-reading it matched 230 and moved 48 materials from guess to published — 36
of them from the batch, 12 recovered from materials already in the library —
taking `kfo` 124 → 172 and `imputed` 483 → 435:

```powershell
.\.venv\Scripts\python.exe scripts\import_kfo.py vendor\zenodo\kfo_2282.xlsx --no-fetch
```

The workbook is gitignored under `vendor/zenodo/`; the CID → InChIKey cache in
`data/percept/` makes `--no-fetch` enough and the run takes seconds. It prints
its own agreement check against the compiled water column, and that is the
number to hold it to: **2.4x median error, 83% within a decade**, unchanged by
the regeneration because KFO is a different sample of the same quantity, not a
sharper instrument. **These are water thresholds** — that was tested, not
assumed; the argument is in the script's docstring, not here.

Two rules decide which rows count, both tested before believed:

* **An oil or fat measurement is dropped, not converted**, on the same argument
  as the air-phase rows. "Nutty Pyrazine" rested on one 50 µg/mL peanut-oil row
  — 50 000 ppb, against a k-NN guess of 2.1 and a supplier selling it
  pre-diluted at 1% because it is potent. It falls back to `imputed`.
* **Stereochemistry is matched at the skeleton, never at the name.** Library
  structures are stereo-unspecified where the sheet holds one isomer, so a
  full-InChIKey join missed Geosmin (imputed 2.95 ppb against a published 0.11)
  and 13 others. The `match` column records `full` or `skeleton`. Name matching
  is the wrong shortcut: the sheet's "Creosol" is guaiacol, the library's is not.

## 6a. IFRA limits: one source, one verdict per material

Every perfumery material carries an IFRA verdict, and it comes from the
documents IFRA publishes rather than from transcription. Two steps, then a
consumer:

```powershell
.venv/Scripts/python.exe scripts/import_ifra.py     # workbooks -> data/ifra/*.csv
.venv/Scripts/python.exe scripts/resolve_ifra.py    # tables  -> data/ifra/materials.csv
```

**The source.** `IFRA 51st Amendment - IFRA Standards Overview (XLSX)` from
`ifrafragrance.org/.../ifra-standards-documentation`, plus the same amendment's
*Annex on contributions from other sources*. Both go in `vendor/ifra/`
(gitignored -- IFRA restricts redistribution); the derived tables are committed,
the way `data/percept/kfo_thresholds.csv` is. 263 standards × 18 categories,
Maximum Acceptable Concentration as % w/w of the finished product, which is the
convention the engine already uses, so nothing converts units.

**What the tables keep that a naive parse loses.** Blank ≠ `No Restriction`:
86 standards are prohibitions or specifications only and their cells are empty
because no limit exists to state, while `No Restriction` is a separate explicit
value in 182 cells. Folding them together invents a 100% cap for 68 prohibited
substances. Likewise 16 cells are comma decimals and 13 carry an annotation
(`0.0015 (5-MOP)`), so a bare `float()` drops 142 of 2 914 usable figures.

**The verdict is five-valued, not a nullable number**, because the engine used
to read "no restriction", "nobody looked", and "we cannot identify it" as the
same 100%:

| status | meaning |
|---|---|
| `restricted` | a numeric cat-4 cap applies |
| `prohibited` | forbidden by an IFRA standard or a regulator; cap 0 |
| `specification_only` | a standard exists but sets a purity spec, not a use limit |
| `unrestricted` | identified, and no standard or rule claims it |
| `unknown` | identity not established -- **not** a licence to dose freely |

**Routes, all recorded, tightest wins** (a permissive figure never survives
contact with a stricter one):

1. **direct** — CAS → standard, then name/synonym → standard.
2. **class** — a curated family rule from `data/ifra/classes.csv`. 259 of 263
   standards say their scope "includes but is not limited to" the CAS numbers
   listed, so a CAS miss is not proof of non-restriction.
3. **contribution** — the material *contains* a restricted constituent:
   `cap = standard limit / level`. Covers Schiff bases (Aurantiol: 2.1% ÷
   56.4% = 3.72%) and every natural (Clove bud: 2.5% ÷ 82% eugenol = 3.05%).
4. **inventory** — the perfumer's own figure, only when nothing above capped it.
5. **house cap** — `data/ifra/house_caps.csv`, for caps that are not IFRA at
   all (Lyral 0 is the EU cosmetics ban, not an IFRA figure: IFRA restricts
   HICC at 0.2%).

**Oils are mixtures and must never be given a limit by name.** Their cap is
derived from the constituents the annex reports in them, which is why the 47
oils need `data/ifra/ncs_alias.csv` (oil → the annex's 301 natural complex
substances) — 30 of 47 now carry a derived cap, the other 17 either have their
own standard (Oakmoss) or no restricted constituent in the annex.

**Curated vs generated**, the distinction that keeps this auditable:

| file | | 
|---|---|
| `standards.csv`, `limits.csv`, `contributions.csv` | generated by `import_ifra.py` |
| `materials.csv`, `material_cas.csv` | generated by `resolve_ifra.py` |
| `classes.csv`, `house_caps.csv`, `ncs_alias.csv`, `cas_fixes.csv` | **curated -- the only judgement in the pipeline** |

Gates `resolve_ifra.py` prints: every material gets a row, a prohibition branch
fires, and at least one cap comes from the contents route. It also prints a
sample (Farnesol 1.2%, Aurantiol 3.72%, Habanolide unrestricted, Linalool
specification-only) so a regression is visible rather than inferred.

**The 52nd amendment** has no public bulk file: consultation ran 12 Dec 2025 –
12 Jun 2026 and the drafts went out through IFRA's SharePoint. It is notified
around January 2027 and binds new creations ~11 months later. Until an official
file exists the overlay is `data/ifra/limits_<amendment>.csv` (same schema, its
own `amendment` column), which `resolve_ifra.py` merges automatically and caps
on the tighter of. Absence from that overlay means "not yet known for 52",
never "unrestricted": absence of evidence for one amendment is not evidence
about the material.

## 7. NEVER_PROPOSE (blocking a material from being proposed)

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

## 8. The perfumer database

Commit the upstream changes **first** — `build_db.py` records the source
commit and dirty flag in the `meta` table, and the db should name the
commit it was built from. Then, from `C:\Valle\dev\perfumer`:

```powershell
python tools/build_db.py --source C:/Valle/dev/olfactory-embeddings --out materials.db
python tools/fetch_cas.py         # PubChem by InChIKey; cached; safe offline
python tools/fetch_longevity.py   # imports tools/longevity.csv (a stub today)
python tools/verify_db.py

`tools/fetch_ifra.py` and `tools/ifra_limits.csv` are gone as of the 2026-11
IFRA rework: the limits now come from `data/ifra/` in the library checkout,
read by `build_db.py` itself (see §6a), so there is no second merge step and
no hand-kept seed to drift.
```

`verify_db.py` prints counts, per-column null rates and vector sanity; the
counts move with the library (435 → 439 with the af81a65 absolutes → 458 with
vertofix, styrax and benzoin → 474 with the myrrh and guaiacwood rebuilds).
The number to hold it to is the one §5 prints — CSV rows + oils, 474 today;
this doc cannot check `build_db.py`'s own arithmetic, so confirm the two agree
before pinning anything. The two assertions in `tests/test_engine.py` that fix
the material count must move with them (a bare `== N` and
`z.shape == (N,)` — update the comment too).

**Gate:** `python -m pytest tests/ -q` in the perfumer repo — 66 tests,
all green, against the real materials.db.

`materials.db` is **gitignored** and never committed. The only perfumer
commits a library change produces are code ones: the test count assertion
above, or a tools/ change.

## 9. Deploy (for reference — do not do this without review)

Two deployments follow a library change, and they are independent: the two
public pages ([`DEPLOY_MOLOR.md`](DEPLOY_MOLOR.md)), and `materials.db` for
the perfumer service. Nothing below touches the pages, and `DEPLOY_MOLOR.md`
never reads `materials.db`.

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
3. **The receptor store holds every CSV row** (§3) — `provenance.json`'s
   `n_molecules` equals the CSV row count for `library/`. This is the gate that
   has actually failed: the percept and the naturals were rebuilt on
   4 November 2026 and the store was not, so 31 molecules had no atlas point on
   a page that passed every
   other check here.
4. Both naturals files parse, carry the same oils, and every entry has a
   138-long percept **and** a 386-long `receptors` array; composed vectors
   say the material's name (balsamic/cinnamon for styrax, balsamic/vanilla
   for benzoin); `compose_hand_naturals.py --dry-run` reports nothing left to
   compose.
5. `export_viewer_data.py` prints a molecule count equal to CSV rows + oils + 1
   (475 today) — every oil gets a point, so the count moves with the naturals
   file — and the two pages build and pass the gates in
   [`DEPLOY_MOLOR.md`](DEPLOY_MOLOR.md) §5.
6. `export_simplify_data.py` runs clean, its materials count equals CSV rows +
   oils, and the new rows have thresholds with honest provenance labels.
7. `tools/verify_db.py` passes in the perfumer repo with the expected
   counts.
8. `python -m pytest tests/ -q` green in perfumer (66 tests).
9. Nothing pushed, nothing deployed, until reviewed.
