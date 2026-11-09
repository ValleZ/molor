# Boiling points, batch 11 of 12

I need **published atmospheric-pressure boiling points** for the fragrance
materials in the table below. Please look them up rather than recalling them.

## Rules, in order of importance

1. **If you cannot find a published value, write `UNKNOWN`.** A missing row
   costs me nothing. A guessed row corrupts a dataset I cannot easily audit,
   because a wrong boiling point looks exactly like a right one. Do not
   estimate from structure, do not interpolate from similar molecules, and do
   not infer from a homologous series.
2. **Give the pressure for every value.** Many of these are distilled under
   vacuum, and a figure at 4 mmHg is not comparable to one at 760 mmHg. If the
   source states no pressure, write `unstated` rather than assuming.
3. **Report the value as the source gives it**, including the unit. Do not
   convert Fahrenheit to Celsius and do not average a range -- I will do both,
   so that a mistake is mine and traceable.
4. **Name the source** for each value: PubChem CID, a supplier's technical data
   sheet, The Good Scents Company, a Sigma-Aldrich listing, a paper. One short
   identifier is enough.
5. If several sources disagree, give the one you judge most reliable and note
   the disagreement in the notes column.

`INCHIKEY` is the identifier to search on -- it is exact, where a trade name
often is not. PubChem accepts it directly:
`https://pubchem.ncbi.nlm.nih.gov/#query=<INCHIKEY>`

## Output format

One CSV block, these columns exactly, one row per input row, same order, no
commentary before or after:

```
name,inchikey,bp_as_published,unit,pressure,source,notes
```

Example of each kind of answer:

```
Vanillin,MWOOGOJBHIARFG-UHFFFAOYSA-N,285-286,C,760 mmHg,PubChem CID 1183,
2-Acetyl Thiazole,XRIZQKKTQVRBHM-UHFFFAOYSA-N,89-91,C,12 mmHg,PubChem CID 520108,vacuum only; no atmospheric figure found
Some Captive,AAAAAAAAAAAAAA-UHFFFAOYSA-N,UNKNOWN,,,,"supplier TDS gives flash point only"
```

## The materials

| name | INCHIKEY | SMILES |
|---|---|---|
| Methyl Tuberate | LOQFOBUZYRBURV-UHFFFAOYSA-N | CCCCCC1OC(=O)CC1C |
| Methyl Cyclogeranate | NVAGRWXDIIKUBU-UHFFFAOYSA-N | COC(=O)C1C(C)=CCCC1(C)C |
| Lactojasmone | ALWUKGXLBSQSMA-UHFFFAOYSA-N | CCCCCCC1(C)CCC(=O)O1 |
| Oudhbois | DSYVYGLBPDTBKH-UHFFFAOYSA-N | CCC(=O)N(CC)c1cccc(C)c1 |
| Operanide | QFQQCCDAGARSEN-UHFFFAOYSA-N | CC1C(C)(C)C2=C(C3OC(C)(C)OC3CC2)C1(C)C |
| Irone Alpha | JZQOJFLIJNRDHK-CMDGGOBGSA-N | CC(=O)/C=C/C1C(C)=CCC(C)C1(C)C |
| Onestolide® | LOKPJYNMYCVCRM-UHFFFAOYSA-N | O=C1CCCCCCCCCCCCCCCO1 |
| Passifloran | GSJSVAFGVJLTNQ-UHFFFAOYSA-N | CCCC(CCOC(C)=O)SC(C)=O |
| Mysore Acetate | TXFBASMXWNIAGS-UHFFFAOYSA-N | CC(=O)OCC1CCC2C3CCC(C3)C12 |
| Lilyflore® | UWSPWQQZFOSTHS-UHFFFAOYSA-N | Cc1ccc2c(c1)CC(C)(CO)C2 |
| Muguissimo™ | MHWUIEVFBRHXMA-PKNBQFBNSA-N | C/C(=C\C1CCCCC1)CC(C)C=O |
| Iraldeine® Delta | NSSHGPBKKVJJMM-UHFFFAOYSA-N | CC(=O)C(C)=CC1=C(C)CCCC1(C)C |
| Orcinyl 3 | NOTCZLKDULMKBR-UHFFFAOYSA-N | COc1cc(C)cc(O)c1 |
| Nerolin Fragarol | XOHIHZHSDMWWMS-UHFFFAOYSA-N | CC(C)COc1ccc2ccccc2c1 |
| Mintonat | OIVWFAFCHQDCCG-UHFFFAOYSA-N | CC(=O)OC1CC(C)CC(C)(C)C1 |
| Irotyl | YXAGIRHBJJLWHW-UHFFFAOYSA-N | CCCCC(CC)C(=O)OCC |
| Labdamate | JAGZUIGGHGTFHO-UHFFFAOYSA-N | CCOC(=O)CCc1ccccc1 |
| Ketoisophorone | AYJXHIDNNLJQDT-UHFFFAOYSA-N | CC1=CC(=O)CC(C)(C)C1=O |
| Nutty Furan | KEFJLCGVTHRGAH-UHFFFAOYSA-N | CC(=O)c1ccc(C)o1 |
| Hydroxyambran | WKHTUDYDJUHYMK-UHFFFAOYSA-N | CC(CO)C1CCCCCCCCCCC1 |
| Nutty Pyrazine | YZEFQPIMXZVPKP-UHFFFAOYSA-N | CC1CCc2nccnc21 |
| Myroxide | LIMXJCIGROLRED-UHFFFAOYSA-N | C=CC(C)=CCC1OC1(C)C |
| Ivy Dioxolane | QSWPBUXWYKVPMT-UHFFFAOYSA-N | CC1=CC2CC(C3OCCO3)C1CC2C(C)C |
| Jasmolactone® Delta | NBCMACYORPIYNY-NSCUHMNNSA-N | C/C=C/CCC1CCCC(=O)O1 |
| Majantol® | FYMOBFDUZIDKMI-UHFFFAOYSA-N | Cc1cccc(CC(C)(C)CO)c1 |
| Hexyl Salicylate | DUKPKQFHJQGTGU-UHFFFAOYSA-N | CCCCCCOC(=O)c1ccccc1O |
| Benzyl Salicylate | ZCTQGTTXIYCGGC-UHFFFAOYSA-N | O=C(OCc1ccccc1)c1ccccc1O |
| Mesifuran | SIMKGHMLPVDSJE-UHFFFAOYSA-N | COC1=C(C)OC(C)C1=O |
| Indolarome | CZSXBBWOROMVEW-UHFFFAOYSA-N | c1ccc2c(c1)CC1COCOC21 |
| Mandarin Aldehyde Total | SSNZFFBDIMUILS-ZHACJKMWSA-N | CCCCCCCCC/C=C/C=O |
| Milk Lactone | WXBXVVIUZANZAU-CMDGGOBGSA-N | CCCCCCC/C=C/C(=O)O |
| Orange Flower Ether | YWJHQHJWHJRTAB-UHFFFAOYSA-N | COC(C)(C)C1CC=C(C)CC1 |
| Ethyl Benzoate | MTZQAGJQAFMTAQ-UHFFFAOYSA-N | CCOC(=O)c1ccccc1 |
| Pandanol | CQLYXIUHVFRXLT-UHFFFAOYSA-N | COCCc1ccccc1 |
| Neo Patchouli | BGEIOYHSLPDHJC-UHFFFAOYSA-N | CC12CCCC(OC=O)(C1)C1CC(C)(C)C1CC2 |
| Orivone® | DCSKAMGZSIRJAQ-UHFFFAOYSA-N | CCC(C)(C)C1CCC(=O)CC1 |
| Maritima | OVJCOXQJBRJIRJ-UHFFFAOYSA-N | CC(C)=CCCC(C)=CCCc1ccncc1 |
| Pistachio Thiazole | ZOMLUJGPZRNYBF-UHFFFAOYSA-N | Cc1ccsc1C(C)C |
| Musk R1 | MKEIDVFLAWJKMY-UHFFFAOYSA-N | O=C1CCCCCCCCCOCCCCCO1 |
| Methyl Myristate | ZAZKJZBWRNNLDS-UHFFFAOYSA-N | CCCCCCCCCCCCCC(=O)OC |
| Microcare PEHG | QCDWFXQBSFUVSP-UHFFFAOYSA-N | OCCOc1ccccc1 |
| Edenolide | QISVACSCBGQBEZ-UHFFFAOYSA-N | CCOC(=O)CC(=O)OC(C)C1CCCC(C)(C)C1 |
| Pamplefleur® | SXGYZCMGVZKIPJ-UHFFFAOYSA-N | CC(CO)CC(C)c1ccccc1 |
| MMB | MFKRHJVUCZRDTF-UHFFFAOYSA-N | COC(C)(C)CCO |
| Mimosal™ | LBKHGAIELUNYML-ZRDIBKRKSA-N | C/C(=C\c1ccc(C)cc1)CCC=O |
| Isojasmone B-11 | VGECIEOJXLMWGO-UHFFFAOYSA-N | CCCCCCC1=CCCC1=O |
| Jasminlactone | XPPALVZZCMPTIV-ARJAWSKDSA-N | CC/C=C\CC1CCCC(=O)O1 |
| Madranol | HMENPFAAAQWASO-UHFFFAOYSA-N | CC(O)C(C)CC1C(C)CCCC1(C)C |
| Iris Nitrile | WOVJAWMZNOWDII-FPLPWBNLSA-N | CCCCCC/C=C\C#N |
| Osyrol | ZJVRYPHKSDHLTC-UHFFFAOYSA-N | COC(C)(C)CCCC(C)C(C)O |
| Iraldeine® Beta | LMWNGLDCJDIIBR-CMDGGOBGSA-N | CCC(=O)/C=C/C1=C(C)CCCC1(C)C |
| Metal Oxazole | LUVGCXAXOHXTEI-UHFFFAOYSA-N | c1nc2c(o1)CCCCCCCCCC2 |
| Muguet Aldehyde | LMETVDMCIJNNKH-UHFFFAOYSA-N | CC(C)=CCCC(C)CCOCC=O |
| Vertofix Coeur | YBUIAJZFOGJGLJ-UHFFFAOYSA-N | CC(=O)C1=C(C)C2CC3(C1)C(C)CCC3C2(C)C |
| Isomuscone | LXJDKGYSHYYKFJ-UHFFFAOYSA-N | O=C1CCCCCCCCCCCCCCC1 |
| Kohinool® | XEFKRPWKRQTXDA-UHFFFAOYSA-N | CC(O)C(C)C(C)C(C)C(C)(C)C |
