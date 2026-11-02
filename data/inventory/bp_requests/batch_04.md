# Boiling points, batch 4 of 5

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
| Magnolan | SHWFPOIJJLMZKA-UHFFFAOYSA-N | CC1OC(C)C2Cc3ccccc3C2O1 |
| Trans-2-Hexenol | ZCHHRLHTBGRGOT-SNAWJCMRSA-N | CCC/C=C/CO |
| Ethyl Butyrate | OBNCKNCVKJNDBV-UHFFFAOYSA-N | CCCC(=O)OCC |
| Helvetolide | LSTSBZKIQFOPHA-UHFFFAOYSA-N | CCC(=O)OCC(C)(C)OC(C)C1CCCC(C)(C)C1 |
| Dimethyl Benzyl Carbinyl Butyrate | SHSGYHAHMQLYRB-UHFFFAOYSA-N | CCCC(=O)OC(C)(C)Cc1ccccc1 |
| Vanillin Acetate | PZSJOBKRSVRODF-UHFFFAOYSA-N | COc1cc(C=O)ccc1OC(C)=O |
| Prenyl Acetate | XXIKYCPRDXIMQM-UHFFFAOYSA-N | CC(=O)OCC=C(C)C |
| Theaspirane | GYUZHTWCNKINPY-UHFFFAOYSA-N | CC1=CCCC(C)(C)C12CCC(C)O2 |
| Trans-2-Octenal | LVBXEMGDVWVTGY-VOTSOKGWSA-N | CCCCC/C=C/C=O |
| Mahagonate | NXAJZPARDMDLKH-UHFFFAOYSA-N | COC(=O)C1CC2(C(C)C)C=CC1(C)CC2 |
| Rhuboflor | UBALJTRHSDXCPY-UHFFFAOYSA-N | CC=C1CC2CC1C1CCCOC21 |
| Phenyl Propyl Alcohol | VAJVDSVGBWFCLW-UHFFFAOYSA-N | OCCCc1ccccc1 |
| Rhodinyl Formate | VMBKZHMLQXCNCP-UHFFFAOYSA-N | C=C(C)CCCC(C)CCOC=O |
| Pharaone | WJWBVJLPGOYBGM-UHFFFAOYSA-N | C=CCCC(=O)C(=C)C1CCCCC1 |
| Thymol | MGSRCZKZVOBKFT-UHFFFAOYSA-N | Cc1ccc(C(C)C)c(O)c1 |
| Sabinene | NDVASEGYNIMXJL-UHFFFAOYSA-N | C=C1CCC2(C(C)C)CC12 |
| Trans-2-Decenal | MMFCJPPRCYDLLZ-CMDGGOBGSA-N | CCCCCCC/C=C/C=O |
| l-Carvone | ULDHMXUKGWMISQ-SECBINFHSA-N | C=C(C)[C@@H]1CC=C(C)C(=O)C1 |
| Amyl Cinnamic Aldehyde | HMKKIXGYKWDQSV-UHFFFAOYSA-N | CCCCCC(C=O)=Cc1ccccc1 |
| Isobutavan | BGKAKRUFBSTALK-UHFFFAOYSA-N | COc1cc(C=O)ccc1OC(=O)C(C)C |
| Scentenal | UOGMZEGKCNHMBF-UHFFFAOYSA-N | COC1CC2CC1C1CC(C=O)CC21 |
| Evernyl | UUQHKWMIDYRWHH-UHFFFAOYSA-N | COC(=O)c1c(C)cc(O)c(C)c1O |
| Terpinyl Acetate | IGODOXYLBBXFDW-UHFFFAOYSA-N | CC(=O)OC(C)(C)C1CC=C(C)CC1 |
| Trans-2-Nonenal | BSAIUMLZVGUGKX-BQYQJAHWSA-N | CCCCCC/C=C/C=O |
| Tonalide | DNRJTBAOUJJKDY-UHFFFAOYSA-N | CC(=O)c1cc2c(cc1C)C(C)(C)C(C)CC2(C)C |
| Cedramber | HRGPYCVTDOECMG-UHFFFAOYSA-N | COC1(C)CCC23CC1C(C)(C)C2CCC3C |
| Oceanol | GDNUTKRFRLPQAB-UHFFFAOYSA-N | COC(=O)c1ccc(O)c(C)c1O |
| p-Cymene | HFPZCAJZSCWRBC-UHFFFAOYSA-N | Cc1ccc(C(C)C)cc1 |
| Terpinolene, Rectified | MOYAFQVGZZPNRA-UHFFFAOYSA-N | CC1=CCC(=C(C)C)CC1 |
| Terpinene Gamma | YKFLAYDHMOASIY-UHFFFAOYSA-N | CC1=CCC(C(C)C)=CC1 |
| Ethyl Vanillin | CBOQJANXLMLOSS-UHFFFAOYSA-N | CCOc1cc(C=O)ccc1O |
| Isobutyl Isovalerate | KEBDNKNVCHQIJU-UHFFFAOYSA-N | CC(C)COC(=O)CC(C)C |
| Ambrinol | GPVOTKFXWGURGP-UHFFFAOYSA-N | CC1(O)CCC2C(=CCCC2(C)C)C1 |
| Rhodinol Ex Geranium Bourbon | JGQFVRIQXUFPAH-JTQLQIEISA-N | C=C(C)CCC[C@H](C)CCO |
| Ambrocenide | YJVLDTISLGRQTJ-UHFFFAOYSA-N | CC1CCC2C(C)(C)C3CC12CC1OC(C)(C)OC13C |
| Fructalate | KRJHRNUTLDTSKY-UHFFFAOYSA-N | CCOC(=O)C1CCC(C(=O)OCC)CC1 |
| Triethyl Citrate | DOOTYTYQINUNNV-UHFFFAOYSA-N | CCOC(=O)CC(O)(CC(=O)OCC)C(=O)OCC |
| Zingerone | OJYLAHXKWMRDGS-UHFFFAOYSA-N | COc1cc(CCC(C)=O)ccc1O |
| Khusimol | OOYRHNIVDZZGQV-UHFFFAOYSA-N | C=C1C2CCC(CO)C23CCC(C3)C1(C)C |
| Piperitone | YSTPAHQEHQSRJD-UHFFFAOYSA-N | CC1=CC(=O)C(C(C)C)CC1 |
| Piperonyl Acetate | PFWYHTORQZAGCA-UHFFFAOYSA-N | CC(=O)OCc1ccc2c(c1)OCO2 |
| 3-Octen-2-One | ZCFOBLITZWHNNC-UHFFFAOYSA-N | CCCCC=CC(C)=O |
| D-Limonene | XMGQYMWWDOXHJM-SNVBAGLBSA-N | C=C(C)[C@@H]1CC=C(C)CC1 |
| Melafleur | AQJANVUPNABWRU-UHFFFAOYSA-N | CC1(C)CCCC2=C1CC(C=O)CC2 |
| Allyl Amyl Glycolate | XCWPXUNHSPOFGV-UHFFFAOYSA-N | C=CCOC(=O)COCCC(C)C |
| 2-Octen-4-One | FMDLEUPBHMCPQV-UHFFFAOYSA-N | CC=CC(=O)CCCC |
| Delta Octalactone | FYTRVXSHONWYNE-UHFFFAOYSA-N | CCCC1CCCC(=O)O1 |
| Hexyl 2-Methyl Butyrate | YUECNVSODFDKOQ-UHFFFAOYSA-N | CCCCCCOC(=O)C(C)CC |
| Whiskey Lactone | WNVCMFHPRIBNCW-UHFFFAOYSA-N | CCCCC1OC(=O)CC1C |
| Palissandin | WDSOYUINLRIGNV-UHFFFAOYSA-N | COC1CCCCCCCCCCC1 |
| Phenethyl Acetate | MDHYEMXUFSJLGV-UHFFFAOYSA-N | CC(=O)OCCc1ccccc1 |
| Viridine | WNJSKZBEWNVKGU-UHFFFAOYSA-N | COC(Cc1ccccc1)OC |
| Rhodinyl Acetate | WNXJCQNXNOOMDJ-UHFFFAOYSA-N | C=C(C)CCCC(C)CCOC(C)=O |
| Safranal | SGAWOGXMMPSZPB-UHFFFAOYSA-N | CC1=C(C=O)C(C)(C)CC=C1 |
| Cascalone | RFSYHLFFESXTQI-UHFFFAOYSA-N | CC(C)c1ccc2c(c1)OCC(=O)CO2 |
| Terpinen-4-ol | WRYLYDPHFGVWKC-UHFFFAOYSA-N | CC1=CCC(O)(C(C)C)CC1 |
