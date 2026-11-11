TASK: graded training labels for the percept model from four extra sources
Repo: C:\Valle\dev\olfactory-embeddings. Read docs/PERCEPT_EVIDENCE.md first,
from "fp5 without pos_weight: REJECTED" to the end. This is to-do items 5 and 6.

GOAL
The active model (artifacts/percepts_fp4.json) is trained on binary GS-LF
labels with pos_weight. Its rich profiles are right (the perfumer confirmed
this by nose), but some cells are overstated: Damascone Beta reads tobacco
0.98 where the perfumer says about 0.25. Build a soft target per
(molecule, descriptor) cell from the GS-LF label plus votes from four other
sources, retrain with fp4's recipe, and produce a candidate the perfumer can
judge. Success is fp4's richness with overstated cells lower. A model with
fewer facets is a failure, whatever its metrics say.

HARD RULES
- Do not change any default, overwrite percepts_fp4.json, rebuild
  naturals.json, export pages, touch C:\Valle\dev\perfumer, commit, or push.
  Your output is a candidate file and a report.
- Do not call the candidate better. Only the perfumer decides, by reading
  vectors of materials they know. Fit, AUC and "label support" have each
  misjudged a model in this project; report them, conclude nothing from them.
- Never drop pos_weight or weaken it as the main result (rejected 2026-11-09).
- The vocabulary is the 138 column names of
  vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv.
  Never add, rename or reorder words.
- Match molecules by flat InChIKey only: scripts/eval_similarity.py flat_key().
  Matching by name or stereo SMILES has silently lost chiral molecules here
  three times.

STEP 0. Control. Reproduce fp4 before changing anything:
  .venv\Scripts\python.exe scripts\train_fp_v3.py --pos-weight --out artifacts/temp/fp4_repro.json
Compare probs with artifacts/percepts_fp4.json and report the max and mean
absolute difference. This is the noise floor for every later comparison. If
the mean difference is above 0.05, stop and report; the recipe is not what
you think it is.

STEP 1. Load the four sources with .venv-pyrfume\Scripts\python.exe:
  pyrfume.load_data("<archive>/molecules.csv" | "stimuli.csv" | "behavior.csv")
Archives: arctander_1960 (behavior_1.csv), ifra_2019, sigma_2014, flavornet.
Also goodscents, for word order only (step 3). scripts/build_percept_datasets.py
shows the loading pattern. For each archive report: molecules, how many match
a GS-LF training row by flat InChIKey, how many match
library/input/expanded_materials.csv, and its distinct descriptor words.
Votes are needed for all 4983 training molecules, not only the library.

STEP 2. One crosswalk per source, in the format of
data/percept/crosswalk_dravnieks_to_vocab.csv (source word, vocab word, note):
  data/percept/crosswalk_<source>_to_vocab.csv
- Map a source word only when it means the same smell. One source word may
  map to several vocab words on several rows.
- Leave a word unmapped when unsure and list it in the report with its
  frequency. A wrong mapping is worse than a missing one.
- Traps already hit in this project: "ethereal" in the vocabulary is the
  ether/solvent smell; "berry" is the fruit; "green" is an odour, not a
  colour or grade; "aromatic" is not "herbal".
- Earlier name-plus-synonym matching reached 66 of Arctander's 76 words,
  105 of IFRA's 184, 78 of Sigma's 114, 75 of Flavornet's 195. Report yours.

STEP 3. Votes. For each training molecule and vocab word, votes = number of
the four sources that apply the word (0 to 4). Add one vote if the word is
the FIRST listed descriptor for that molecule in Good Scents or in the IFRA
glossary (first position only; later positions carry no information).

STEP 4. Check the votes against the panel before training. Rebuild this table
on the Dravnieks molecules (data/percept/dravnieks.csv with its crosswalk;
"present" is a rating of 10 or more) and compare with the 2026-11-09 one:
  label 1, votes 0 / 1 / 2+ : mean rating 12.3 / 17.1 / 23.7, present 42 / 58 / 64 %
  label 0, votes 0 / 1 / 2+ : mean rating  2.9 /  7.3 / 11.5, present  7 / 28 / 44 %
Your proper crosswalks will change the counts. If the ordering by votes does
not hold, stop and report.

STEP 5. Targets. Write data/percept/gslf_soft_targets.csv with the same
columns as the GS-LF file. Take values from YOUR step 4 table. Starting point:
  label 1: 0.40 / 0.60 / 0.65 for 0 / 1 / 2+ votes
  label 0: 0.00 / 0.30 / 0.45 for 0 / 1 / 2+ votes
Also write a second file with the same values divided by the largest, so the
best-supported class is 1.0 (variant B).

STEP 6. Train. Add a --soft-targets <csv> option to scripts/train_fp_v3.py
without changing its behaviour when the option is absent:
- the loss uses the soft target;
- pos_weight is still computed from the BINARY labels (the code sums Ytr;
  with soft targets that sum is wrong);
- every roc_auc_score call (early stopping and out-of-fold) uses the BINARY
  labels; it fails or misleads on soft ones.
Know this before reading results: BCE with pos_weight w and target t settles
at p = w*t / (w*t + 1 - t). With w = 50 and t = 0.4 that is 0.97, so soft
targets alone may grade almost nothing down on rare labels. Train both:
  A: targets as in step 5      -> artifacts/percepts_fp6a.json
  B: targets rescaled to 1.0   -> artifacts/percepts_fp6b.json
and report, per vote class, the mean prediction on training cells for fp4, A
and B. If neither separates the classes, say so plainly; that is a result.
Do not invent a third loss without reporting A and B first.

STEP 7. The gate sheet (to-do item 3). For Damascone Beta, Cassifix,
Phenethyl Alcohol, Hedione, Iso E Super and five more well-known library
materials of different families, print every word above 0.5 with its value
for fp4, A and B side by side, plus tobacco on Damascone Beta explicitly.
The perfumer's known readings, to print beside them:
  Damascone Beta: berry, fruity, tea, black currant present; dry plum
    somewhat; tobacco about 0.25 if at all
  Cassifix: woody plus amber
  Phenethyl Alcohol: a complex smell of many flowers
Also report, as numbers only: out-of-fold AUC, mean prediction, descriptors
with at least 5 library materials above 0.7 (fp4: 129 of 138), and how many
cells moved by more than the step 0 noise floor, up and down.

DELIVERABLES
- the four crosswalk CSVs and the list of unmapped words per source
- both soft-target CSVs, the two candidate percept files
- the train_fp_v3.py change as an uncommitted diff
- a report with the tables of steps 0, 1, 4, 6 and 7, and a list of anything
  you could not do or were unsure about. Mark every number as measured or
  estimated. Do not write a recommendation to adopt.
