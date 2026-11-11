# Deploying the atlas and the simplifier

The recipe for the two published pages, end to end: from a library store to
`https://valkon.us/molor` and `https://valkon.us/molor/simplify`. Written after
doing it by hand for the November 2026 rebuild (427 molecules on the atlas, 474
materials in the simplifier); every command below is the one that was actually
run, and the numbers in [§7](#7-what-the-november-2026-build-measured) are what
that run printed, there to be compared against rather than copied.

Neither page is server-side. Each is one self-contained HTML file with its JSON
payload inlined in a `<script type="application/json">` block, so "deploy" means
copy two files into the site repository and push it. There is no CI in that
repository and no build step: the committed HTML *is* the page.

## 0. What ships, and to where

| page | built by | local artifact | committed as | served at |
|---|---|---|---|---|
| receptor atlas | `export_viewer_data.py` → `build_viewer.py` | `artifacts/receptor_atlas_deploy.html` | `molor/index.html` | https://valkon.us/molor |
| formula simplifier | `export_simplify_data.py` → `build_simplify.py` | `artifacts/simplify_deploy.html` | `molor/simplify/index.html` | https://valkon.us/molor/simplify |

Two different checkouts are involved and they have opposite roles:

* `C:\Valle\dev\olfactory-embeddings` — this repository. Holds the stores, the
  templates and the scripts. **Its build outputs are gitignored**
  (`artifacts/receptor_atlas*.html`, `artifacts/simplify*.html`,
  `artifacts/viewer_data*.json`, `artifacts/simplify_data.json`), so nothing
  built here is versioned here.
* `C:\Valle\dev\valkon.us` — the site repository
  (`git@github.com:ValleZ/web.valkon.us.git`, root `CNAME`, pushed to `main`).
  It is the **version of record for the built pages**. A deploy that is not
  committed there has not happened; the artifact in `artifacts/` will be
  overwritten by the next local build without a trace.

The gitignore rule is deliberate rather than incidental — see the comment above
`artifacts/receptor_atlas*.html` in `.gitignore`: keeping a second copy in this
repository only doubles the churn, because the JSON payload embeds a full
coordinate set that changes on every export.

## 1. Before you start

Both pages are pure Python over numpy/scipy/rdkit, no GPU. Use the project venv
from the repository root:

```powershell
cd C:\Valle\dev\olfactory-embeddings
Test-Path .\.venv\Scripts\python.exe      # must be True
git status --short                        # only what you intend to ship
git log -1 --format='%H %s'               # put this hash in the site commit
```

Record that source hash. The site repository's commit message is the only
place the provenance of a published build lives, and a rebuild six weeks later
that cannot name its input is a rebuild nobody can audit.

**The analytics token is not in this repository, on purpose** (`build_viewer.py`
says why: a token in the repository is a token that outlives the reason it was
added). It is public — a Mixpanel client-side write key, meant to be visible in
the page — so read it back from what is already published:

```powershell
$mixpanel = (Select-String -Path C:\Valle\dev\valkon.us\molor\index.html `
  -Pattern 'mixpanel\.init\("([^"]+)"' -AllMatches).Matches[0].Groups[1].Value
$mixpanel
```

If that comes back empty, the currently published atlas was built without
analytics and the token has to be recovered from the Mixpanel project instead.

## 2. Rebuild the stores first, if the library changed

The two exports read stores; they do not build them. If a material, a natural,
a threshold or the percept model moved since the last deploy, the stores must be
regenerated first, and [`ADDING_MATERIALS.md`](ADDING_MATERIALS.md) is that
recipe (its §1–§7: CSV row, percept retrain, receptor vectors, naturals, alias
table, thresholds, never-propose). Its §8 lands in `materials.db` for the
perfumer service, which is a **separate deployment from this one** — the pages
below do not use `materials.db` at all. Two of those steps are the ones this
page cannot do without: the receptor stores (§3) and the hand-composed oils'
vectors (§4).

The four inputs the exports actually open, and how each is rebuilt:

| input | rebuilt by | used for |
|---|---|---|
| `artifacts/library/` (vectors + logits) | `python -m orvector library add --csv library/input/expanded_materials.csv --device cpu` | the chemistry space, and the reference molecule set |
| `artifacts/naturals.json` | `build_naturals.py --store library --max-nearest-cosine 1.01` from `data/naturals/compositions.json` and `data/naturals/oil_descriptions.json` | the oils, on the atlas and in the simplifier |
| `artifacts/percepts_fp3.json` | `train_fp_v3.py` | every colour, and the simplifier's arithmetic |

### Gate: the store must hold every row of the CSV

The receptor store is what goes stale unnoticed, because the exports do not
complain: `export_viewer_data.py` builds the atlas from whatever the store
holds, so a store one commit behind simply draws fewer points, and the page
looks correct. `export_simplify_data.py` reads the CSV and the
naturals directly, so it is current while the atlas is not — the two pages then
disagree by exactly the missing molecules.

That is not hypothetical. The 4 November 2026 rebuild retrained
`percepts_fp3.json` and rebuilt `naturals.json` on the day, and left both
receptor stores at 1 November: 31 molecules added by `af81a65`, `02dca7e`,
`f47075e` and `2a2d5fa` — Vertofix Coeur, Tobacarol, Guaiol, the nine myrrh
furanosesquiterpenes — had no atlas point, and the page was 427 points where it
should have been 458. Both pages built cleanly and passed every check here,
because none of them compares the store to the CSV.

So compare them explicitly, before exporting:

```powershell
.venv/Scripts/python.exe -c "import csv,io,json;n=len(list(csv.DictReader(io.open('library/input/expanded_materials.csv',newline='',encoding='utf-8-sig'))));p=json.load(open('artifacts/library/provenance.json'));print('csv',n,'store',p['n_molecules'],p['updated_utc'])"
```

Pass condition: `store` equals `csv` (`≥` is acceptable if the store holds a
molecule the CSV no longer does — the export aligns by name and drops extras;
`<` never is), and `updated_utc` is later than the last commit that touched the
CSV:

```powershell
git log -1 --format='%ci %s' -- library/input/expanded_materials.csv
```

If they do not, the command in the table above is the fix. It is append-only —
it skips InChIKeys already in the store, so growing the library costs the new
molecules, not all 427 × 386 pairs — and the panel embeddings come from
`artifacts/panel_embeddings.npy`, so ESM-2 does not re-run. 31 new molecules
took 129 s.

**`--device cpu` is not a compromise, it is the only option.** DGL in `.venv` is
the CPU wheel: torch reports `2.3.0+cu121` and `cuda True`, but any DGL graph
moved to the GPU fails with `Check failed: allow_missing: Device API cuda is not
enabled. Please install the cuda version of dgl.` No other venv here has a CUDA
DGL either (`.venv-pom` is CPU torch, `.venv-unimol` has no dgl, `.venv-pyrfume`
no torch). Torch being CUDA-enabled is not enough — the model is a DGL model.

`run_mpnn.py` is gone with the MPNN space (November 2026), so the landmine it
carried is too: it used to default to `library/input/final_materials.csv`, a
legacy 128-row file with dilution-qualified names, and would score that instead
and exit 0. If a second space is ever added back, give its entry point an
explicit `--csv` default before anything else.

**The receptor store is versioned in this repository**, unlike the pages.
After a rebuild `git status` shows three modified files — `molecules.tsv`,
`provenance.json` and `vectors.npz` under `artifacts/library/` — and they belong
in a commit *here*, before the site deploy, so that the source hash the site
commit names is a commit that actually contains the molecule set the pages were
drawn from. A store rebuilt and left uncommitted means no commit reproduces the
published page.

**Landmine: `build_naturals.py` is not part of a deploy.** It rebuilds
`artifacts/naturals.json` from `compositions.json` from scratch and will wipe
every hand-composed oil. A deploy needs the naturals file as it stands, not a
fresh one.

### Gate: an oil composed by hand has no receptor vector until one is composed

**The naturals store is generated, never written.** `data/naturals/
compositions.json` is the spec (47 oils); `build_naturals.py` composes every
entry from it — percept as the weighted mean of the constituents' fp3 rows,
receptor vector as the weighted mean of their MolOR scores — and writes one
file per space. `--compose` selects the rule; `weighted-mean` is the default
and the adopted one, `noisy-or` reproduces the pre-November builds, and
`logit-mean` matches the atlas's placement rule. `load_naturals` reads
`v["receptors"]` for each oil it places, so an entry without that key cannot
be on the atlas at all.

Eight oils used to be written by hand instead, and that is the defect this gate
exists to catch: they were written straight into the file, composed by hand,
while the other 39 went through `build_naturals.py`, and nothing compared them.
Their note said "No receptor vector, so no atlas point", which was true only
while the store was missing their constituents. They are in the spec now and
all 47 go through one path; myrrh's descriptor row moved 0.48 and castoreum's
0.57 when the hand composition was replaced by the generator's.

Since 2026-11-10 the percept row has a second input: the words of
`data/naturals/oil_descriptions.json` are raised on the composed vector
(`docs/PERCEPT_EVIDENCE.md`, "Oil descriptions"). `percept` is the raised
vector, `percept_composed` the composition alone, and `described` the record.
The receptor vector, and so the oil's position on the atlas, is untouched.
`--percepts` defaults to fp4 now; it was fp3 until that day, so this command
as written recomposed the oils on the retired model.

If the spec, the descriptions or the store changed, rebuild the file:

```powershell
.venv/Scripts/python.exe scripts/build_naturals.py --store library --max-nearest-cosine 1.01
```

**`--max-nearest-cosine 1.01` is the current policy.** The default (0.98) trims
an oil whose percept sits within that cosine of a single library material, on
the argument that the point repeats one already drawn — sound for sweet orange
at 0.9998 to D-Limonene, and it left 12 oils invisible including Birch Tar Oil
(0.985 to Guaiacol), Oakmoss Absolute (0.982) and Styrax Resinoid (0.991). All
47 carry a point now: a material the perfumer can smell and cannot find on the
page is the worse failure, and the card already says its piece —
`collapsed_onto` travels into the payload and renders as *"Not a
Distinguishable oil: this build sits on top of X…"*. The measurement is kept,
not deleted, so re-trimming is a flag away.

**Expect the built oils to move when their inputs move.** The build is a pure
function of the spec, the percept model and the two receptor stores, so
re-running it after any of those changes rewrites entries you did not touch —
on this run all 39 built oils shifted, because their stored vectors had been
computed while the store was 31 molecules short and unscorable constituents
were silently dropped from the weights. Compare against the previous file and
know why before accepting it; a silent drift here is what hand-maintenance hid.

If nothing has changed in the library, this section is a no-op and §3–§6 alone
will reproduce the published pages to within the projection coordinates.

## 3. The atlas

Two commands, about two and a half minutes, almost all of it in the MDS fits.
Run from the repository root.

```powershell
# the payload: score matrix, projections, distances, per-material cards
.\.venv\Scripts\python.exe scripts\export_viewer_data.py --no-notes --out artifacts\viewer_data_deploy.json

# the page: template + payload + analytics
.\.venv\Scripts\python.exe scripts\build_viewer.py artifacts\viewer_data_deploy.json artifacts\receptor_atlas_deploy.html --mixpanel $mixpanel
```

`--no-notes` is not optional for a published build. It withholds the perfumer's
written smelling impressions, the odour tag list and the tag-derived colour map
— `"note": ""` on every molecule, `"odours": []`, `"odourColour": {}` — and
nothing else. The inventory fields (use level, threshold, family, volatility)
are published and stay published; the perfumer asked for those.

Two other flags exist and neither belongs in a deploy:

* `--query FILE` injects blends from `predict_formula.py --viewer-point` as
  stars. It exists for `scripts/try_formula.py`, is documented "never passed
  for a published build", and would publish someone's formula.
* `--no-white-point` drops the average-of-everything reference point. It is a
  local experiment; the published atlas has it.

The costs of a mistake here are asymmetric: a build without `--no-notes` leaks
private writing onto a public page, and it looks exactly like a correct build
until someone reads one.

Note what this export is *not* allowed to decide: `MODELS[0]` (the GCN store)
fixes the molecule set and the naturals file everything aligns to, and
`modelOrder` separately decides which space the page opens on. Those are two
different questions and the code keeps them apart on purpose. That distinction
was learned the hard way when the GCN atlas plotted its oils in MPNN
coordinates; with one chemistry space the specific failure cannot recur, but
the index still means "the molecule set", not "the first tab".

## 4. The simplifier

Two commands, about two seconds.

```powershell
# the payload: descriptor rows, aliases, solvents, thresholds, use levels, isolation
.\.venv\Scripts\python.exe scripts\export_simplify_data.py --out artifacts\simplify_data.json

# the page
.\.venv\Scripts\python.exe scripts\build_simplify.py artifacts\simplify_data.json artifacts\simplify_deploy.html --mixpanel $mixpanel
```

Keep the payload's **default** name. `artifacts/simplify_data.json` is not only
the simplifier page's input: the perfumer service reads it too, through
`tools/build_db.py`. Redirecting it to a `_deploy` name is harmless for the page
and leaves the perfumer reading a stale file.

The simplifier has no `--no-notes` flag because it carries no notes; the export
selects its fields explicitly (see its module docstring) and there is nothing
private in the list.

## 5. The gates, before anything is copied

There is no JS engine in this project, so a built page is checked by lexing it.
A parse error means the whole script never runs, the error handler never
installs, and the page renders blank — invisible to every check that reads the
document instead of executing it. This is the step whose absence once shipped a
blank page:

```powershell
.\.venv\Scripts\python.exe scripts\check_viewer_js.py artifacts\receptor_atlas_deploy.html
.\.venv\Scripts\python.exe scripts\check_viewer_js.py artifacts\simplify_deploy.html
```

Expected, per page: `4 script blocks, 3 with code`, then `ok` on each of the
three, plus the charset line. Anything else — `FAIL`, an unterminated string
with a line number — stops the deploy.

Then the leak check, on the artifact that is about to be copied:

```powershell
$a = [IO.File]::ReadAllText('artifacts\receptor_atlas_deploy.html')
"odours empty:      " + $a.Contains('"odours":[]')
"odourColour empty: " + $a.Contains('"odourColour":{}')
"empty notes:       " + ([regex]::Matches($a,'"note":""').Count)
"NON-EMPTY notes:   " + ([regex]::Matches($a,'"note":"[^"]').Count)
"mixpanel inits:    " + ([regex]::Matches($a,'mixpanel\.init\(').Count)
```

Pass condition: the first two `True`, non-empty notes `0`, empty notes equal to
the molecule count the export printed, and `mixpanel inits` `1`. A count of `2`
means the snippet was injected twice; `0` means the token was not taken and the
build is a local one.

Finally, open the file directly. It is self-contained — no server, no network
for the payload — so `Start-Process artifacts\receptor_atlas_deploy.html` gives
the real thing:

```powershell
Start-Process artifacts\receptor_atlas_deploy.html
Start-Process artifacts\simplify_deploy.html
```

Check that the atlas opens on the space it should, that the legend colours
read as smells, that clicking a material shows a card with a threshold and no
note, and that the simplifier resolves a pasted formula (a line or two of a
known one) rather than reporting most of it unresolved.

## 6. Publish

Copy, look at what changed, commit, push. All four steps, no shortcuts:

```powershell
Copy-Item artifacts\receptor_atlas_deploy.html C:\Valle\dev\valkon.us\molor\index.html -Force
Copy-Item artifacts\simplify_deploy.html      C:\Valle\dev\valkon.us\molor\simplify\index.html -Force

git -C C:\Valle\dev\valkon.us status --short
git -C C:\Valle\dev\valkon.us diff --stat
```

`status` should show exactly two modified files and nothing else. `diff --stat`
will show two enormous line changes (each page is one long minified line or a
handful of them) — that is normal and is why the gates in §5, not the diff, are
what a review reads.

Commit in the site repository, naming the source commit from §1:

```powershell
git -C C:\Valle\dev\valkon.us add molor/index.html molor/simplify/index.html
git -C C:\Valle\dev\valkon.us commit -m "molor: atlas and simplifier rebuilt at <molecules> materials"
git -C C:\Valle\dev\valkon.us push origin main
```

The `molor: ` prefix is the convention every commit in that repository uses, and
the message should say what a reader of the site would notice (a count, a new
filter, a fixed number), not which script ran. Recent history is the model:
`molor: 396 materials`, `molor: one threshold column, shared by both pages`,
`molor: a single published value is not a median of one`.

After the push, hard-refresh both live URLs (`Ctrl+F5`) — an unrefreshed tab
will show you the previous build and look like a failed deploy — and check the
counts in the page header against what the exports printed.

`sitemap.xml` in the site repository does not list `/molor` or
`/molor/simplify`; a rebuild does not change that, and adding it is a separate
decision about what the site advertises rather than part of this recipe.

## 7. What the November 2026 build measured

Not gates — a record. If a rebuild prints numbers far from these, the store
changed under you and that is worth understanding before it is published.

**Atlas export** (`export_viewer_data.py --no-notes`), 157 s:

```
white point added: the average of 427 materials
48 essential oils built from GC composition
386 of 475 materials are in the percept model's training set
inventory: 147 of 475 materials carry a shelf row's use level
thresholds (shared with the simplifier): 126 compiled, 124 KFO-Atlas, 177 imputed
full descriptor matrix: 475 molecules x 138
wrote artifacts/viewer_data_deploy.json  (8.19 MB)
  molecules 475  receptors 386  models ['gcn', 'percept']
  gcn   mds3_full      shepard 0.989
  percept mds3_full    shepard 0.940
```

`475 = 427 molecules + 47 oils + the white point`; the receptor store holds
427, and the export prints 48 oils because that count includes the white point.
Two spaces, not three: the chemistry space and the percept space. The MPNN
space was removed in November 2026 — it tied the GCN on the panel comparison
and its GRU was pinned to 82–100 distinct values across a 2816-cell
representation — and the build got a third smaller with it (8.28 MB → 5.54 MB).

Shepard above 0.93 on the opening space is the number to watch; a projection
that drops below about 0.9 is drawing a picture of something other than what it
claims, and the README's Pipeline 2 section is why (the panel mean dominates raw
vectors, hence preprocess → PCA-20 → MDS).

**Simplifier export**, 1.4 s:

```
compiled 143, KFO-Atlas 124, imputed 484 (leave-one-out median error 3.63x)
752 materials (47 of them oils), 149 owned, 752 with a threshold
138 descriptors, 289 aliases, yardstick 1.101
53 may never be proposed; 34 offered but unverified
```

`yardstick 1.101` is the median distance from a material to its nearest
neighbour in percept space ([`scripts/yardstick.py`](scripts/yardstick.py)), and
the page reports closeness against it. It is a property of the data **and of the
percept model**: the identical formula gives 0.506 on fp3 and 1.221 on fp4,
because `pos_weight` lifts the mean probability from 0.047 to 0.164. The 0.451
quoted here for months was an fp3 figure retired when fp4 became active
(`b8d08f9`); the definition never changed. Read the number the export prints
rather than a literal — `percept_blind_spots.py` hard-coded 0.439 and stayed
silently wrong after the model change.

If `compiled` looks small and `owned` looks low, the inventory join is not
resolving: both counts are the sheet meeting the library by name, and that join
is `parse_formulas.library_name` (alias table + head-word match). It was exact
string equality once, and the visible cost was 15 materials carrying an imputed
threshold while a published one was in the sheet.

**Built pages**: atlas 8.28 MB, simplifier 0.57 MB. The atlas
carries three spaces × six projections × 475 points plus a 475 × 138 descriptor
matrix, so it is a large page by design and 8 MB is the expected size, not a
mistake.

## Acceptance gates, in one list

1. Working tree in this repository clean apart from intended changes; source
   commit hash recorded.
2. Stores current (§2), **checked, not assumed**: `provenance.json`'s
   `n_molecules` for `library/` equals the CSV row count, and it is newer than
   the last CSV commit. `naturals.json` and
   `percepts_fp3.json` are current. Nothing in the export printed a `SKIPPED`
   model, and the molecule count the export prints equals CSV rows + oils + 1
   (475 today).
2b. `naturals.json` was **generated** from `data/naturals/compositions.json`
   (`build_naturals.py --store library --max-nearest-cosine 1.01`): 47 oils,
   every entry with a 138-long `percept` and a 386-long `receptors`, and one
   identical key set across all 47. Any movement in oils you did not touch is
   explained by an input that moved — say which.
3. Atlas exported with `--no-notes`, built with the token; simplifier exported
   to its default `artifacts/simplify_data.json`.
4. `check_viewer_js.py` prints `ok` for all three code blocks of both pages.
5. Leak check: `"odours":[]`, `"odourColour":{}`, `0` non-empty notes, `1`
   `mixpanel.init` per page.
6. Both pages opened from disk and looked at — atlas opens on the intended
   space, a material card has no note, the simplifier resolves a known formula.
7. Shepard and the export counts in line with §7.
8. Copied to `molor/index.html` and `molor/simplify/index.html`; `git status`
   in the site repository shows exactly those two files.
9. Committed with a `molor: ` message that names the source commit, and pushed
   to `origin main`.
10. Both live URLs hard-refreshed and their counts checked against the exports.
