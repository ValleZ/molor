# Boiling points, batch 8 of 12

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
| Neofolione | ZWNPUELCBZVMDA-CMDGGOBGSA-N | CCCCCC/C=C/C(=O)OC |
| Dihydrolinalool | JRTBBCBDKSRRCY-UHFFFAOYSA-N | CCC(C)(O)CCC=C(C)C |
| Rose Oxide | CZCBTSFUTPZVKJ-UHFFFAOYSA-N | CC(C)=CC1CC(C)CCO1 |
| Rosyrane Super | SWOPLXXJAVYFPY-UHFFFAOYSA-N | CC1=CCOC(c2ccccc2)C1 |
| Tabanon | CBQXHTWJSZXYSK-UHFFFAOYSA-N | CC=CC=C1C(C)=CC(=O)CC1(C)C |
| Vernaldehyde | MVTYXAVPKZRAMW-UHFFFAOYSA-N | CC(C)CCCC1=CCC(C)(C=O)CC1 |
| 2-Acetyl Thiazole | MOMFXATYAINJML-UHFFFAOYSA-N | CC(=O)c1nccs1 |
| Myrac Aldehyde | MQBIZQLCHSZBOI-UHFFFAOYSA-N | CC(C)=CCCC1=CCC(C=O)CC1 |
| Ethyl Safranate | KYEZYLFPHRVFBF-UHFFFAOYSA-N | CCOC(=O)C1=C(C)C=CCC1(C)C |
| Safraleine® | KZSUMHASCAWKLE-UHFFFAOYSA-N | CC1C(=O)c2ccccc2C1(C)C |
| Para Cresyl Caprylate | ALRYNTSLFYRKGF-UHFFFAOYSA-N | CCCCCCCC(=O)Oc1ccc(C)cc1 |
| Terrasol | KIPCKEJKGCXRGA-UHFFFAOYSA-N | CCC1(O)C2(C)CCC(C2)C1(C)C |
| Apritone | ZNSALEJHPSBXDK-JLHYYAGUSA-N | CC(C)=CCC/C(C)=C/CC1CCCC1=O |
| Veloutone | PUKWIVZFEZFVAT-UHFFFAOYSA-N | CCCCCC1(C)CCC(C)(C)C1=O |
| Veratraldehyde | WJUFSDZVCOTFON-UHFFFAOYSA-N | COc1ccc(C=O)cc1OC |
| Fructalate | KRJHRNUTLDTSKY-UHFFFAOYSA-N | CCOC(=O)C1CCC(C(=O)OCC)CC1 |
| Indocolore | HWCUCNKPBMGSSC-UHFFFAOYSA-N | C=C(OC(C)=O)c1ccccc1 |
| Karanal | DASQRZJTRKBKPP-UHFFFAOYSA-N | CCC(C)C1(C)COC(C2CCC(C)=CC2C)OC1 |
| 2-Acetyl Pyrazine | DBZAKQWXICEWNW-UHFFFAOYSA-N | CC(=O)c1cnccn1 |
| Romascone | SMJXJPQSPQXVMU-UHFFFAOYSA-N | C=C1CCCC(C)(C)C1C(=O)OC |
| Gamma Heptalactone | VLSVVMPLPMNWBH-UHFFFAOYSA-N | CCCC1CCC(=O)O1 |
| Gamma Decalactone | IFYYFLINQYPWGJ-UHFFFAOYSA-N | CCCCCCC1CCC(=O)O1 |
| Honeyflor | XEAIHUDTEINXFG-UHFFFAOYSA-N | CCC=CCC(=O)OC |
| Herbavert | JVBNHJDWWBSWLE-UHFFFAOYSA-N | CCOC1CC(C)CC(C)(C)C1 |
| Piconia | VCOCESNMLNDPLX-UHFFFAOYSA-N | CC1(C)C2CCC3(C2)C1C(=O)CCC3(C)C |
| Dimethyl Sulfide | QMMFVYPAHWMCMS-UHFFFAOYSA-N | CSC |
| Clonal™ | VXCUURYYWGCLIH-UHFFFAOYSA-N | CCCCCCCCCCCC#N |
| Corylone | CFAKWWQIUFSQFU-UHFFFAOYSA-N | CC1=C(O)C(=O)CC1 |
| Rhumacetal | MWUDABUKTZAZCX-UHFFFAOYSA-N | CCOC1(OCC)CCCCC1 |
| Caryophyllene Oxide | NVEQFIOZRFFVFW-RGCMKSIDSA-N | C=C1CC[C@H]2O[C@]2(C)CC[C@@H]2[C@@H]1CC2(C)C |
| Guaiyl Acetate | DRFSOBZVMGLICQ-SGMGOOAPSA-N | CC(=O)OC(C)(C)[C@@H]1CC[C@H](C)C2=C(C1)[C@@H](C)CC2 |
| Anisaldehyde | ZRSNZINYAWTAHE-UHFFFAOYSA-N | COc1ccc(C=O)cc1 |
| Cis-3-Hexenyl Tiglate | JNWQKXUWZWKUAY-JQEGGOPCSA-N | C/C=C(\C)C(=O)OCC/C=C\CC |
| Phytol | BOTWFXYSPFMFNR-UHFFFAOYSA-N | CC(=CCO)CCCC(C)CCCC(C)CCCC(C)C |
| Freskomenthe | RQXTZKGDMNIWJF-UHFFFAOYSA-N | CCC(C)C1CCCCC1=O |
| Verdol | DLTWBMHADAJAAZ-UHFFFAOYSA-N | CC(C)(C)C1CCCCC1O |
| Borneol | DTGKSKDOIYIVQL-NQMVMOMDSA-N | CC1(C)[C@@H]2CC[C@]1(C)[C@H](O)C2 |
| Gamma Hexalactone | JBFHTYHTHYHCDJ-UHFFFAOYSA-N | CCC1CCC(=O)O1 |
| Rosethyl | DEYIKSBEIOIJCD-UHFFFAOYSA-N | CCOCc1ccccc1OC |
| Peranat | UMFWRNLJDQMTFR-UHFFFAOYSA-N | CCCC(C)COC(=O)C(C)CCC |
| Verdone | ZRYDPLOWJSFQAE-UHFFFAOYSA-N | CC(C)(C)C1CCCCC1=O |
| Isopropyl Quinoline | NKCQEIXYLHACJC-UHFFFAOYSA-N | CC(C)c1ccc2ncccc2c1 |
| Hindinol® | MTVBNJVZZAQKRV-UHFFFAOYSA-N | CC(=CCC1CC=C(C)C1(C)C)CO |
| Norlimbanol® | BVDMQAQCEBGIJR-UHFFFAOYSA-N | CCCC(O)CCC1C(C)CCCC1(C)C |
| 2-Ethylhexyl Salicylate | FMRHJJZUHUTGKE-UHFFFAOYSA-N | CCCCC(CC)COC(=O)c1ccccc1O |
| Delta Dodecalactone | QRPLZGZHJABGRS-UHFFFAOYSA-N | CCCCCCCC1CCCC(=O)O1 |
| Skatole | ZFRKQXVRDFCRJG-UHFFFAOYSA-N | Cc1c[nH]c2ccccc12 |
| Peomosa | RUGISKODRCWQNE-UHFFFAOYSA-N | Cc1ccccc1CCO |
| Amyl Benzoate | QKNZNUNCDJZTCH-UHFFFAOYSA-N | CCCCCOC(=O)c1ccccc1 |
| Hydratropic Aldehyde | IQVAERDLDAZARL-QMMMGPOBSA-N | C[C@@H](C=O)c1ccccc1 |
| Levistamel | IXYLIUKQQQXXON-UHFFFAOYSA-N | Cc1cc(C)oc(=O)c1 |
| Pyroprunat | IBCFURUUOKXNFW-UHFFFAOYSA-N | CC=CC(=O)OC1CCCC1C1CCCC1 |
| Trans-2-Nonenal | BSAIUMLZVGUGKX-BQYQJAHWSA-N | CCCCCC/C=C/C=O |
| Oranger Crystals | XSAYZAUNJMRRIR-UHFFFAOYSA-N | CC(=O)c1ccc2ccccc2c1 |
| Nerolin Bromelia | GUMOJENFFHZAFP-UHFFFAOYSA-N | CCOc1ccc2ccccc2c1 |
| Plicatone | VWFXPROOUAWFEE-UHFFFAOYSA-N | CC1CC2C3CCC(C3)C2CC1=O |
