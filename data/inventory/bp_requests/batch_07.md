# Boiling points, batch 7 of 12

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
| Nopyl Acetate | AWNOGHRWORTNEI-UHFFFAOYSA-N | CC(=O)OCCC1=CCC2CC1C2(C)C |
| Dihydroterpinyl Acetate | HBNHCGDYYBMKJN-UHFFFAOYSA-N | CC(=O)OC(C)(C)C1CCC(C)CC1 |
| Herbanate® | HJEFAEQTNTXLHL-WRWGMCAJSA-N | CCOC(=O)[C@H]1[C@H](C(C)C)[C@@H]2C=C[C@H]1C2 |
| Acalea | VHBOIZSCWFNPBM-UHFFFAOYSA-N | CCCCCC(C=O)=Cc1ccc(C)cc1 |
| Okoumal | JEPWTUCYPWOCQV-UHFFFAOYSA-N | CC1COC(C)(c2ccc3c(c2)C(C)(C)CCC3(C)C)O1 |
| Exaltone | OSOIQJGOYGSIMF-UHFFFAOYSA-N | O=C1CCCCCCCCCCCCCC1 |
| Firsantol® | DGHXZJZALPECTJ-UHFFFAOYSA-N | C=C(CC(C)CO)C1CC=C(C)C1(C)C |
| Violiff | LDHHCYCOENSXIM-IHWYPQMZSA-N | COC(=O)OC1CC/C=C\CCC1 |
| Melafleur | AQJANVUPNABWRU-UHFFFAOYSA-N | CC1(C)CCCC2=C1CC(C=O)CC2 |
| Jasmal® | VSRVCSJJKWDZSH-UHFFFAOYSA-N | CCCCCC1COCCC1OC(C)=O |
| Civettone | ZKVZSBSZTMPBQR-UPHRSURJSA-N | O=C1CCCCCCC/C=C\CCCCCCC1 |
| Magnolan | SHWFPOIJJLMZKA-UHFFFAOYSA-N | CC1OC(C)C2Cc3ccccc3C2O1 |
| Allyl Ionone | FXCYGAGBPZQRJE-ZHACJKMWSA-N | C=CCCC(=O)/C=C/C1C(C)=CCCC1(C)C |
| Cyclopidene | ZDBCSZOFHPULAS-UHFFFAOYSA-N | COC(=O)C=C1CCCC1 |
| Methyl Diantilis | FNEWGEWRECZWQM-UHFFFAOYSA-N | CCOc1cc(COC)ccc1O |
| Iso E Super | FVUGZKDGWGKCFE-UHFFFAOYSA-N | CC(=O)C1(C)CC2=C(CCCC2(C)C)CC1C |
| Melonal | YGFGZTXGYTUXBA-UHFFFAOYSA-N | CC(C)=CCCC(C)C=O |
| Isocyclocitral | AEXLFXYEFLKADK-UHFFFAOYSA-N | CC1=CC(C)C(C)C(C=O)C1.CC1=CC(C)C(C=O)C(C)C1 |
| Cyclemax™ | RLEFOSDUWZYGOS-UHFFFAOYSA-N | CC(C)c1ccc(CCC=O)cc1 |
| Violettyne MIP | VJCOQIJZYGQPKM-FNORWQNLSA-N | C=C/C=C/C#CCCCCC |
| Spirambrene | ORHSGDMSYGKJJY-GRYCIOLGSA-N | C[C@@H]1CC[C@H]2[C@H](C2(C)C)C12COC(C)(C)OC2 |
| Nectarate | KLEHDRXJJZQKPQ-UHFFFAOYSA-N | CCCC(=O)OC1C=CC2C3CCC(C3)C12 |
| Gamma Dodecalactone | WGPCZPLRVAWXPW-UHFFFAOYSA-N | CCCCCCCCC1CCC(=O)O1 |
| Floramat | ZHWLEUGSDGROJS-UHFFFAOYSA-N | CCOC(=O)OC1CCCCC1C(C)(C)C |
| Trifernal | MYHGOWDLVRDUFA-UHFFFAOYSA-N | CC(CC=O)c1ccccc1 |
| Oxyoctaline Formate | DAWDTDPJINNEJM-UHFFFAOYSA-N | CC1=CCCC2(C)C(OC=O)C(C)CCC12C |
| Acetyl Propionyl | TZMFJUDUGYTVRY-UHFFFAOYSA-N | CCC(=O)C(C)=O |
| Precyclemone B | MBVBLQFHVRGNLW-UHFFFAOYSA-N | CC(C)=CCCC1=CCCC(C)(C=O)C1 |
| Poirenate | PZUDDXJZNSJESK-UHFFFAOYSA-N | CCOC(=O)C(C)C1CCCCC1 |
| Bourgeonal® | FZJUFJKVIYFBSY-UHFFFAOYSA-N | CC(C)(C)c1ccc(CCC=O)cc1 |
| Verdox HC | FINOAUDUYKVGDS-UHFFFAOYSA-N | CC(=O)OC1CCCCC1C(C)(C)C |
| Isoeugenyl Acetate | IUSBVFZKQJGVEP-SNAWJCMRSA-N | C/C=C/c1ccc(OC(C)=O)c(OC)c1 |
| Dimyrcetol | ONXALCWZXZZQSG-UHFFFAOYSA-N | C=CC(C)CCCC(C)(C)OC=O |
| Doremox | GDAVABNCFOTAOD-UHFFFAOYSA-N | CC1CCOC(c2ccccc2)C1 |
| Silvial® | YLIXVKUWWOQREC-UHFFFAOYSA-N | CC(C)Cc1ccc(CC(C)C=O)cc1 |
| Dimethyl Benzyl Carbinol | RIWRBSMFKVOJMN-UHFFFAOYSA-N | CC(C)(O)Cc1ccccc1 |
| Maltol Isobutyrate | LPNUBNQKJRGCSC-UHFFFAOYSA-N | Cc1c(OC(=O)C(C)C)occc1=O |
| Troenan | LODLFUCEWWWXPR-UHFFFAOYSA-N | CCCC(C)C1OCC(C)(CCC)CO1 |
| Melozone® | USPJNXWHVJTDJW-UHFFFAOYSA-N | O=CC1CCC2C3CCC(C3)C12 |
| Camphor | DSSYKIVIOFKYAU-OMNKOJBGSA-N | CC1(C)C2CC[C@]1(C)C(=O)C2 |
| Dupical | WFEISWUNAJPLRX-ONNFQVAWSA-N | O=CCC/C=C1\CC2CC1C1CCCC21 |
| Cyclohexyl Ethyl Alcohol | QJQZRLXDLORINA-UHFFFAOYSA-N | OCCC1CCCCC1 |
| Berryflor | ODGMPRNNVXITJQ-UHFFFAOYSA-N | CCOC(=O)CCCCCOC(C)=O |
| Delphone | VNWOJVJCRAHBJJ-UHFFFAOYSA-N | CCCCCC1CCCC1=O |
| Pharaone | WJWBVJLPGOYBGM-UHFFFAOYSA-N | C=CCCC(=O)C(=C)C1CCCCC1 |
| Aldehyde C-11 MOA | LBICMZLDYMBIGA-UHFFFAOYSA-N | CCCCCCCCC(C)C=O |
| Cis-3-Hexenyl Cis-3-Hexenoate | UZJQQWFHPLYECS-SFECMWDFSA-N | CC/C=C\CCOC(=O)C/C=C\CC |
| Cedrenyl Acetate | AICQDCHSUWFHCC-ZUFFMMDNSA-N | CC(=O)OCC1=CC[C@@]23C[C@@H]1C(C)(C)[C@@H]2CC[C@H]3C |
| Trans-2,Cis-6-Nonadienol | AMXYRHBJZOVHOL-ODYTWBPASA-N | CC/C=C\CC/C=C/CO |
| Stemone | CBVWMGCJNPPAAR-UHFFFAOYSA-N | CCC(CC(C)CC)=NO |
| Safranal | SGAWOGXMMPSZPB-UHFFFAOYSA-N | CC1=C(C=O)C(C)(C)CC=C1 |
| Anisyl Acetate | HFNGYHHRRMSKEU-UHFFFAOYSA-N | COc1ccc(COC(C)=O)cc1 |
| Zenolide | GJJSUPSPZIZYPM-UHFFFAOYSA-N | O=C1CCCCCCCCCCC(=O)OCCO1 |
| Boisambrene Forte | VQNUNMBDOKEZHS-UHFFFAOYSA-N | CCOCOC1CCCCCCCCCCC1 |
| Isobutavan | BGKAKRUFBSTALK-UHFFFAOYSA-N | COc1cc(C=O)ccc1OC(=O)C(C)C |
| Labienoxime | MEJYWDUBOCZFFS-FENWIEIGSA-N | C=C/C(C)=C/CC(C)(C)/C(=N\O)C(C)C |
