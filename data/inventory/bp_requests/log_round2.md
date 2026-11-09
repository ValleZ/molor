# Boiling points, round 2 — batches 06–12

342 materials had no boiling point; `scripts/make_bp_request.py --first
formula_priority.txt` wrote them into `batch_06.md` … `batch_12.md`, 50
materials plus 6 unlabelled controls each (48 + 6 in the last). Every row was
looked up and transcribed; nothing was estimated, converted or averaged.

## How each row was looked up

1. **PubChem**, by InChIKey, ≤ 1 request/second.
   Note for anyone repeating this: `…/pug_view/data/compound/<CID>/JSON?heading=Boiling+Point`
   returns `PUGVIEW.NotFound` for many records that *do* hold a boiling point
   (e.g. CID 10050, "Camphor, (1S)-"). The heading filter was therefore
   dropped and the whole `Chemical and Physical Properties` subtree fetched,
   with the `Boiling Point` section located locally. 139 rows had a value that
   way.
2. **Same-skeleton PubChem record.** Where the exact CID had no boiling point,
   the truncated InChIKey (first block) was used to list CIDs sharing the
   constitution, and the lowest-numbered one was read. This rescued 14 rows,
   all of them cases where the inventory's SMILES gave a thin stereo record
   while a better-attested stereoisomer exists (camphor, borneol, geosmin…).
   Each is marked `stereo-isomer CID <n>` in `notes`.
3. **NIST Chemistry WebBook**, CAS taken from PubChem synonyms, phase-change
   page (`ID=C<digits>&Mask=4`), `T boil` in kelvin, ~3 s between requests.
   Six rows.
4. **Vigon product pages**, ~3 s between requests, property table read off the
   page and the item number recorded (`Vigon item 5xxxxx`). 77 rows. The pages
   state no pressure, so those rows are `unstated`.

Total: 384 rows, 236 with a number (139 + 14 + 6 + 77).

## Per batch

| reply | rows | with a number | UNKNOWN | at reduced pressure | merge verdict | control error |
|---|---|---|---|---|---|---|
| reply_06 | 56 | 39 | 17 | 9 | **pass** | 5 scorable, median 2.0 °C, worst 34.0 °C |
| reply_07 | 56 | 28 | 28 | 4 | unverified | 2 scorable, median 6.0 °C, worst 6.0 °C |
| reply_08 | 56 | 44 | 12 | 11 | **pass** | 6 scorable, median 0.8 °C, worst 43.5 °C |
| reply_09 | 56 | 35 | 21 | 12 | **pass** | 5 scorable, median 1.5 °C, worst 16.0 °C |
| reply_10 | 56 | 35 | 21 | 8 | **pass** | 5 scorable, median 0.0 °C, worst 4.0 °C |
| reply_11 | 56 | 29 | 27 | 7 | **pass** | 3 scorable, median 1.4 °C, worst 10.0 °C |
| reply_12 | 48 | 26 | 22 | 5 | **pass** | 4 scorable, median 3.8 °C, worst 13.0 °C |
| **total** | **384** | **236** | **148** | **56** | none failed | |

Of the 236 values, 53 state an atmospheric pressure (760 mmHg, 755 mmHg),
127 state none at all, and 56 are at reduced pressure and will be dropped by
`merge_bp.py` as designed. 6 values are in K (NIST) and 1 in °F (camphor,
CAMEO); both units are left as published for the merge to convert.

`reply_07` is `unverified` because its six controls are trade materials whose
honest answer is UNKNOWN (Melonal's only figure is at 100 mmHg, Doremox,
Stemone, Verdox HC and Iso E Super), leaving two scorable — both within 6 °C.
It is left as is, per the brief.

`merge_bp.py` merged all seven replies. `data/inventory/boiling_points.csv`
went from 155 to 296 materials; merged boiling point against molecular weight
rho +0.734 over 296 (the 127 hand-compiled figures give +0.705).

## Where sources disagree by more than 10 °C

| material | values found | what was recorded |
|---|---|---|
| Ethyl Benzoate | PAC 213.4 °C @760, JECFA 212 °C, HMDB **75.50 °C @760** | 213.4 °C. The HMDB string is 137 °C low and looks corrupted (a dropped "1" in 175.5 still would not fit); noted. |
| Maltol | Vigon **285 °C**, CAMEO 199 °F (93 °C) at 760 mmHg "sublimes", HSDB "sublimes at 93 °C" | 285 °C. About 190 °C apart; the low figures are sublimation points, noted. |
| Methyl Myristate | HMDB 323 °C @760, HSDB 295 °C (no pressure) | 323 °C, 28 °C apart; noted. |
| Isobutavan | HMDB 165–166 °C @2 mmHg, JECFA 130–132 °C @2 mmHg | JECFA, 35 °C apart **at the same stated pressure**; noted. Both are vacuum, so the merge drops the row either way. |
| Oranger Crystals | HMDB 300–301 °C @760, JECFA 300 °C | 300–301 °C. This is a control and the inventory holds 257 °C — 43.5 °C. The published figure was kept. |
| Velvione | Vigon 296 °C, inventory 330 °C | 296 °C (control miss of 34 °C). |
| Strawberry Furanone | HSDB 216 °C, inventory 232 °C | 216 °C (control miss of 16 °C). |
| Ethyl Linalool | Vigon 214 °C, inventory 199 °C | 214 °C (control miss of 15 °C). |
| Florhydral | Vigon 257 °C, inventory 270 °C | 257 °C (control miss of 13 °C). |
| Hexyl Salicylate | Vigon 290 °C, inventory 300 °C | 290 °C (control miss of 10 °C). |
| Camphor | CAMEO 399 °F @760 (= 203.9 °C), DrugBank 204 °C, HSDB 209 °C | 399 °F, the only entry with a stated pressure; the three agree within 5 °C. |
| Irotyl | JECFA 210–211 °C, HMDB 81 °C @24 mmHg | JECFA; different pressures, so not a contradiction, but 130 °C apart as printed. |
| Cis-3-Hexenyl Tiglate | HMDB 140 °C @30 mmHg, JECFA 105 °C @5 mmHg | 140 °C (higher pressure), noted. |
| Aldehyde C-12 Lauric | JECFA 249 °C, HMDB 185 °C @100 mmHg | 249 °C, noted. |
| Bisabolene | Vigon 261 °C (unspecified) and 274 °C (β-) | 274 °C for β-bisabolene, noted. |
| Zingerone | HSDB 187–188 °C @14 mmHg, HMDB reads "141.00 to 0.50 °C @ 0.00 mm Hg" | HSDB; the HMDB string is internally impossible. |

The five inventory-vs-published rows (Oranger Crystals, Velvione, Strawberry
Furanone, Ethyl Linalool, Florhydral, Hexyl Salicylate) are worth a look: in
each case the figure transcribed here is from a named published source and the
hand-compiled value in `inventory.csv` is what differs. I did not touch
`inventory.csv`.

## Things I was unsure of

- **Stereo-isomer stand-ins (14).** The exact InChIKey resolved to a CID with
  no physical data, and the value comes from a CID sharing the same first
  InChIKey block: Methyl Jasmonate (5281929), Cedryl Acetate (240121), Geranyl
  Acetone (19633), Civettone (11334208), Camphor
  (2537), Borneol (64685), Geosmin (1213), Phytol (5280435), β-Santalol
  (5281532), Geranyl Acetone (19633), Hydratropic Aldehyde (7146),
  Trans-2-Octenal (16900), Trans-2,Cis-6-Nonadienol (34134), Matsutake Alcohol
  (18827), Cinnamyl Alcohol (5315892). Marked in `notes`.
- **Maltol Isobutyrate** has no PubChem record at all for its InChIKey and none
  for its SMILES; its 176 °C is a supplier figure only.
- **Name/structure mismatches to be aware of:**
  - Pistachio Thiazole — Vigon returns two different thiazoles (92 °C and
    176 °C) and neither can be tied to the InChIKey, so it stays UNKNOWN.
  - Almond Pyrazine — the CAS search returned 2-methoxy-3-methyl- (158 °C) and
    2-methoxy-3-isobutylpyrazine (48 °C); the methyl one was used.
  - Jasmal® — two Vigon products; the jasmonyl one (251 °C) was used.
  - Maltol — the name search also returns Maltol Isobutyrate (176 °C); the
    "maltol FCC" entry (285 °C) was used for Maltol.
  - Vertosine is recorded from Vigon's methyl-anthranilate/Triplal Schiff base
    page (405 °C), which matches the SMILES but is a trade-grade product.
  - Piconia is recorded from Vigon's isolongifolanone page (296 °C).
- **Supplier values rejected (8).** Vigon lists figures for Dupical (110 °C),
  Magnolan (134 °C), γ-Dodecalactone (131 °C), Phenyl Ethyl Pivalate (100 °C),
  Rose Oxide Laevo (230 °C), Sandalore (95 °C) and Traseolide (178 °C) that
  cannot be atmospheric boiling points for those materials — they look like
  the flash-point column or a vacuum figure carried over. Rose Oxide Laevo at
  230 °C also contradicts the racemic material's published 182 °C by far more
  than a stereoisomer can. Those rows stay UNKNOWN and each `notes` cell says
  what the page stated and why it was not used.
- **Pressure.** Everything from Vigon is `unstated`: the pages print a boiling
  point without a pressure. Everything from NIST is `unstated` for the same
  reason (the `T boil` table does not print one). Neither was assumed to be
  760 mmHg.
- **UNKNOWN rows.** All 148 say what was checked. The large majority are trade
  captives whose only PubChem record carries no physical properties at all and
  which NIST does not hold. `thegoodscentscompany.com`, where many of these
  figures live, was not consulted (robots.txt blocks it), and no page behind a
  login was used.

## Not done, on purpose

No export, no rebuild, no commit. `batch_01..05.md` and `reply_01..05.csv`
untouched; `scripts/` and other data files unchanged.
