# Boiling points, batch 5 of 5

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
| (E,E)-2,4-Decadienal | JZQKTMZYLHNFPL-BLHCBFLLSA-N | CCCCC/C=C/C=C/C=O |
| Exaltone | OSOIQJGOYGSIMF-UHFFFAOYSA-N | O=C1CCCCCCCCCCCCCC1 |
| Corylone | CFAKWWQIUFSQFU-UHFFFAOYSA-N | CC1=C(O)C(=O)CC1 |
| Anisyl Acetate | HFNGYHHRRMSKEU-UHFFFAOYSA-N | COc1ccc(COC(C)=O)cc1 |
| Isopropyl 2-Methylbutyrate | DIRDKDDFAMNBNY-UHFFFAOYSA-N | CCC(C)C(=O)OC(C)C |
| Allyl Cyclohexyl Propionate | TWXUTZNBHUWMKJ-UHFFFAOYSA-N | C=CCOC(=O)CCC1CCCCC1 |
| Beta Santalol | OJYKYCDSGQGTRJ-UHFFFAOYSA-N | C=C1C2CCC(C2)C1(C)CCC=C(C)CO |
| Cosmone | ZFGBKXBPHOUSJX-CLFYSBASSA-N | CC1C/C=C\CCCCCCCCC(=O)C1 |
| Cuminic Aldehyde | WTWBUQJHJGUZCY-UHFFFAOYSA-N | CC(C)c1ccc(C=O)cc1 |
| 2-Ethyl-3,5-Dimethylpyrazine | WHMWOHBXYIZFPF-UHFFFAOYSA-N | CCc1nc(C)cnc1C |
| Methyl Jasmonate | GEWDNTWNSAZUDX-UHFFFAOYSA-N | CCC=CCC1C(=O)CCC1CC(=O)OC |
| Methyl Corylone | RLEJFFGSMZQXJX-UHFFFAOYSA-N | CC1=C(O)C(=O)C(C)C1 |
| Phenyl Ethyl Phenyl Acetate | ZOZIRNMDEZKZHM-UHFFFAOYSA-N | O=C(Cc1ccccc1)OCCc1ccccc1 |
| 3,3-Dimethyl-1-Indanone | QWZAOSKLFKAEOK-UHFFFAOYSA-N | CC1(C)CC(=O)c2ccccc21 |
| Civettone | ZKVZSBSZTMPBQR-UPHRSURJSA-N | O=C1CCCCCCC/C=C\CCCCCCC1 |
| Damascenone Total | POIARNZEYGURDG-FNORWQNLSA-N | C/C=C/C(=O)C1=C(C)C=CCC1(C)C |
| Phytol | BOTWFXYSPFMFNR-UHFFFAOYSA-N | CC(=CCO)CCCC(C)CCCC(C)CCCC(C)C |
| Liffarome | BLOXMGXSDAAJGX-PLNGDYQASA-N | CC/C=C\CCOC(=O)OC |
| Dimethyl Disulfide | WQOXQRCZOLPYPM-UHFFFAOYSA-N | CSSC |
| Isomethyl Ionone Alpha | JRJBVWJSTHECJK-UHFFFAOYSA-N | CC(=O)C(C)=CC1C(C)=CCCC1(C)C |
| Tetramethyl Pyrazine | FINHMKGKINIASC-UHFFFAOYSA-N | Cc1nc(C)c(C)nc1C |
| Furfuryl Mercaptan | ZFFTZDQKIXPDAF-UHFFFAOYSA-N | SCc1ccco1 |
| Phenoxyethyl Isobutyrate | MJTPMXWJHPOWGH-UHFFFAOYSA-N | CC(C)C(=O)OCCOc1ccccc1 |
| Alpha Santalol | PDEQKAVEYSOLJX-UHFFFAOYSA-N | CC(=CCCC1(C)C2CC3C(C2)C31C)CO |
| Neobergamate Forte | DCXXKSXLKWAZNO-UHFFFAOYSA-N | C=CC(=C)CCCC(C)(C)OC(C)=O |
| Methyl Ionone Gamma Coeur | PSLHSYRVWGKZHP-UHFFFAOYSA-N | CC(=O)C=C(C)C1C(C)=CCCC1(C)C |
| Aldehyde C-8 | NUJGJRNETVAIRJ-UHFFFAOYSA-N | CCCCCCCC=O |
| Alpha Santalene | KWFJIXPIFLVMPM-UHFFFAOYSA-N | CC(C)=CCCC1(C)C2CC3C(C2)C31C |
| 2-Methyl Pyrazine | CAWHJQAVHZEVTJ-UHFFFAOYSA-N | Cc1cnccn1 |
| Manzanate | HZPKNSYIDSNZKW-UHFFFAOYSA-N | CCCC(C)C(=O)OCC |
| Mefranal | DFJMIMVMOIFPQG-UHFFFAOYSA-N | CC(CC=O)CCc1ccccc1 |
| 2,6-Dimethylpyrazine | HJFZAYHYIWGLNL-UHFFFAOYSA-N | Cc1cncc(C)n1 |
| 3-Methylbutanal | YGHRJJRRZDOVPD-UHFFFAOYSA-N | CC(C)CC=O |
| Dimethyl Trisulfide | YWHLKYXPLRWGSE-UHFFFAOYSA-N | CSSSC |
| Isoeugenyl Acetate | IUSBVFZKQJGVEP-SNAWJCMRSA-N | C/C=C/c1ccc(OC(C)=O)c(OC)c1 |
| 2,3,5-Trimethylpyrazine | IAEGWXHKWJGQAZ-UHFFFAOYSA-N | Cc1cnc(C)c(C)n1 |
| 2-Acetyl Pyrazine | DBZAKQWXICEWNW-UHFFFAOYSA-N | CC(=O)c1cnccn1 |
| Acetanisole | NTPLXRHDUXRPNE-UHFFFAOYSA-N | COc1ccc(C(C)=O)cc1 |
