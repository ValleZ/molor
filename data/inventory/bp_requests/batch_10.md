# Boiling points, batch 10 of 12

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
| Boronal | FJCQUJKUMKZEMH-UHFFFAOYSA-N | CC(C=O)=CCC1=C(C)CCCC1(C)C |
| Cycloverdyl Propionate | BLBJUGKATXCWET-UHFFFAOYSA-N | CCC(=O)OC1CC2CC1C1CC=CC21 |
| Damascol 4 | VEZKFOIEZCNFBT-UHFFFAOYSA-N | CCC(=O)CC(C)(C)c1ccccc1 |
| Cinnamyl Alcohol | OOCCDEMITAIZTP-QPJJXVBHSA-N | OC/C=C/c1ccccc1 |
| Cetylic Alcohol | BXWNKGSJHAJOGX-UHFFFAOYSA-N | CCCCCCCCCCCCCCCCO |
| Fruitate | MGQPKOBPXHENPO-LSCVPOLPSA-N | CCOC(=O)[C@@]12CCC[C@@H]1[C@@H]1CC[C@H]2C1 |
| Delphol HC | MWYGSCHWFNRSET-UHFFFAOYSA-N | CCCCCC1CCCC1O |
| Dimethyl Hydroquinone | OHBQPCCCRFSCAX-UHFFFAOYSA-N | COc1ccc(OC)cc1 |
| Chamomile Isobutyrate | JJWWUTCHOAKZPR-UHFFFAOYSA-N | C=C(C)CC(C)OC(=O)C(C)C |
| Coumarone | YUTFQTAITWWGFH-UHFFFAOYSA-N | CC(=O)c1cc2ccccc2o1 |
| Claritone | CTLDWNVYXLHMAS-UHFFFAOYSA-N | CC(C)=CCC(C)(C)C(=O)C(C)C |
| Coffee Pentenolone | JHWFWLUAUPZUCP-UHFFFAOYSA-N | CCC1=C(O)C(=O)CC1 |
| Civettal | XOKMRXSMOHCNIX-UHFFFAOYSA-N | Cc1ccc2c(c1)CCCN2 |
| Brahmanol | CYVGAJHMMVDTDZ-UHFFFAOYSA-N | CC1=CCC(CCC(C)CO)C1(C)C |
| Firascone | OENXQYZXKNBMLM-UHFFFAOYSA-N | COC(=O)C1C(C)C=CCC1(C)C |
| Triplal | MZZRKEIUNOYYDF-UHFFFAOYSA-N | CC1=CC(C)C(C=O)CC1 |
| Centifolether | YRELUSXNOZNOQI-UHFFFAOYSA-N | COCC(C)Cc1ccccc1 |
| Dulcinyl | TZJLGGWGVLADDN-UHFFFAOYSA-N | CC(=O)CCc1ccc2c(c1)OCO2 |
| Damascone Alpha | CRIGTVCBMUKRSL-FNORWQNLSA-N | C/C=C/C(=O)C1C(C)=CCCC1(C)C |
| Caramel Acetate | VPKIUOQJQJVLRW-UHFFFAOYSA-N | CC(=O)OC1=C(C)OC(C)C1=O |
| Geraldehyde | MMLYERLRGHVBEK-UHFFFAOYSA-N | CC(C)=CCCC(C)=CCCC=O |
| Herboxane | QUBNQDBPPTZYJL-UHFFFAOYSA-N | CCCCC1OC(C)CC(C)(C)O1 |
| Canthoxalia | DCVGBQPSFNIPPI-UHFFFAOYSA-N | COC(=O)c1ccccc1N=CC(C)Cc1ccc(OC)cc1 |
| Celeriax | NGSZDVVHIGAMOJ-UHFFFAOYSA-N | CCC=C1OC(=O)c2ccccc21 |
| Cetonal® | ACVNYUMGXUDACR-UHFFFAOYSA-N | CC1=CCCC(C)(C)C1C(C)CCC=O |
| Bisabolene | YHBUQBJHSRGZNF-UHFFFAOYSA-N | CC(C)=CCC=C(C)C1CC=C(C)CC1 |
| Caryophyllene Acetate | SJDDHMSVZMBJPH-YQFWSFKMSA-N | CC(=O)O[C@@]12CCC[C@@](C)(CC[C@@H]3[C@@H]1CC3(C)C)C2 |
| Cis-3-Hexenyl Propionate | LGTLDEUQCOJGFP-WAYWQWQTSA-N | CC/C=C\CCOC(=O)CC |
| Toscanol | SHLSJFDJYSSUIP-UHFFFAOYSA-N | COc1ccc(CC2CC2)cc1 |
| Hydroxycitronellal | WPFVBOQKRVRMJB-UHFFFAOYSA-N | CC(CC=O)CCCC(C)(C)O |
| Hyacinth Body | QQDGMPOYFGNLMT-UHFFFAOYSA-N | CCOC(C)OCCc1ccccc1 |
| Dihydroambrate | KOSQMCNVEGXSCQ-UHFFFAOYSA-N | C=CC1(OC(C)=O)CCCCC1C(C)CC |
| Hexyl Acetate | AOGQPLXWSUTHQB-UHFFFAOYSA-N | CCCCCCOC(C)=O |
| Cinnamoxal™ | KKVZAVRSVHUSPL-GQCTYLIASA-N | COc1ccccc1/C=C/C=O |
| Filbertone | ARJWAURHQDJJAC-GQCTYLIASA-N | C/C=C/C(=O)C(C)CC |
| Dihydroactinidiolide | IMKHDCBNRDRUEB-LLVKDONJSA-N | CC1(C)CCC[C@@]2(C)OC(=O)C=C12 |
| Fraistone | GSIXJEIRJVOUFB-UHFFFAOYSA-N | CCOC(=O)CC1(C)OCC(C)O1 |
| Cocoa Hexenal | YURDCJXYOLERLO-LCYFTJDESA-N | CC(C)C/C=C(/C=O)c1ccccc1 |
| Boisiris® | LUNAXRIJWWBQTE-UHFFFAOYSA-N | C=C1C2CCC(C)(OCC)C1CCC2(C)C |
| Dihydroanethol | KBHWKXNXTURZCD-UHFFFAOYSA-N | CCCc1ccc(OC)cc1 |
| Dihydromyrcenyl Acetate | BEARMGATPGLSKG-UHFFFAOYSA-N | C=CC(C)CCCC(C)(C)OC(C)=O |
| Cantryl® | YYJOTFQPYNOYAB-UHFFFAOYSA-N | CC1=CCC(CC#N)C1(C)C |
| Dihydrorose Oxide | LETLQGKRPMMKKC-UHFFFAOYSA-N | CC(C)CC1CC(C)CCO1 |
| Grape Butyrate | OMSUIQOIVADKIM-UHFFFAOYSA-N | CCOC(=O)CC(C)O |
| Diethyl Tartrate (DET) | YSAVZVORKRDODB-PHDIDXHHSA-N | CCOC(=O)[C@H](O)[C@@H](O)C(=O)OCC |
| Dihydro Gamma Ionone | ZLPHULOLXDKCND-UHFFFAOYSA-N | C=C1CCCC(C)(C)C1CCC(C)=O |
| Ethyl Undecylenate | FXNFFCMITPHEIT-UHFFFAOYSA-N | C=CCCCCCCCCC(=O)OCC |
| Globanone® | ZGEHHVDYDNXYMW-UHFFFAOYSA-N | O=C1CCCCCCC=CCCCCCCC1 |
| Herbac® | DXIWBWIDAYBUDF-UHFFFAOYSA-N | CC(=O)C1CCCC(C)(C)C1 |
| Cyclozonal | POLSEKWMVQFBBA-UHFFFAOYSA-N | CC1(C)CCCC2=C1CCC(C=O)C2 |
| Corps Racine | JJJPNTQYUJPWGQ-UHFFFAOYSA-N | c1ccc(CCCc2ccccn2)cc1 |
| Grenoble Pyrazine | OXQOBQJCDNLAPO-UHFFFAOYSA-N | Cc1nccnc1C |
| Datilat | ZNPGEERBEOWCEF-UHFFFAOYSA-N | CC=CC(=O)OC(C)C1CCCCC1 |
| Cistulate | ZVLGEOXMDTXXEI-UHFFFAOYSA-N | COC(=O)C1CC2CC1C(C)(C)C2 |
| Carvacryl Methyl Ether | YVLHTQPPMZOCOW-UHFFFAOYSA-N | COc1cc(C(C)C)ccc1C |
| Floralate | LICUDMOFXVHDPB-UHFFFAOYSA-N | CC(=O)OCC1CCC(C)=CC1C |
