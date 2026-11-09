# Boiling points, batch 9 of 12

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
| Biocyclamol | OOOOFOPLSIWRAR-UHFFFAOYSA-N | CC(C)=CCCC(C)=CCCC(C)CCO |
| Vanillin Acetate | PZSJOBKRSVRODF-UHFFFAOYSA-N | COc1cc(C=O)ccc1OC(C)=O |
| Strawberry Furanone | INAXVXBDKKUCGI-UHFFFAOYSA-N | CC1=C(O)C(=O)C(C)O1 |
| Linalyl Propionate | WAQIIHCCEMGYKP-UHFFFAOYSA-N | C=CC(C)(CCC=C(C)C)OC(=O)CC |
| Bicyclononalactone | MSFLYJIWLHSQLG-UHFFFAOYSA-N | O=C1CCC2CCCCC2O1 |
| Palissandin | WDSOYUINLRIGNV-UHFFFAOYSA-N | COC1CCCCCCCCCCC1 |
| Damascone Gamma | IXLLBXDECOMIBP-FNORWQNLSA-N | C=C1CCCC(C)(C)C1C(=O)/C=C/C |
| Rhuboflor | UBALJTRHSDXCPY-UHFFFAOYSA-N | CC=C1CC2CC1C1CCCOC21 |
| Zingerone | OJYLAHXKWMRDGS-UHFFFAOYSA-N | COc1cc(CCC(C)=O)ccc1O |
| Creosol | NAPGLDHBHXOYMM-UHFFFAOYSA-N | COc1c(C)cc(C)cc1O |
| Geosmin | JLPUXFOGCDVKGO-GRYCIOLGSA-N | C[C@@H]1CCC[C@]2(C)CCCC[C@@]12O |
| Allyl Anthranilate | UCANFCXAKYMFGA-UHFFFAOYSA-N | C=CCOC(=O)c1ccccc1N |
| Ethyl Myristate | MMKRHZKQPFCLLS-UHFFFAOYSA-N | CCCCCCCCCCCCCC(=O)OCC |
| Ambrocenide | YJVLDTISLGRQTJ-UHFFFAOYSA-N | CC1CCC2C(C)(C)C3CC12CC1OC(C)(C)OC13C |
| Theaspirane | GYUZHTWCNKINPY-UHFFFAOYSA-N | CC1=CCCC(C)(C)C12CCC(C)O2 |
| Alcohol C-10 | MWKFXSUHUHTGQN-UHFFFAOYSA-N | CCCCCCCCCCO |
| Trans-2-Octenal | LVBXEMGDVWVTGY-VOTSOKGWSA-N | CCCCC/C=C/C=O |
| Delta Octalactone | FYTRVXSHONWYNE-UHFFFAOYSA-N | CCCC1CCCC(=O)O1 |
| Andrane | HZRFVTRTTXBHSE-UHFFFAOYSA-N | CC1CCC2C(C)(C)C3CC12CC1OC13C |
| Biomuguet | ITBYWGRSPHMAEE-HNRFISLBSA-N | CC(C)=CCC/C(C)=C/CC[C@H](C)CC=O |
| Mefranal | DFJMIMVMOIFPQG-UHFFFAOYSA-N | CC(CC=O)CCc1ccccc1 |
| Aldambre | MVOSYKNQRRHGKX-UHFFFAOYSA-N | O=C1CCCCCCCCCCO1 |
| Piperonyl Acetate | PFWYHTORQZAGCA-UHFFFAOYSA-N | CC(=O)OCc1ccc2c(c1)OCO2 |
| Whiskey Lactone | WNVCMFHPRIBNCW-UHFFFAOYSA-N | CCCCC1OC(=O)CC1C |
| Trans-2-Decenal | MMFCJPPRCYDLLZ-CMDGGOBGSA-N | CCCCCCC/C=C/C=O |
| Beta Santalol | OJYKYCDSGQGTRJ-UHFFFAOYSA-N | C=C1C2CCC(C2)C1(C)CCC=C(C)CO |
| Aldron® | TUCJTYGBRWODLS-UHFFFAOYSA-N | CC1CC(CC2C3CCC(C3)C2(C)C)CCC1=O |
| Acetal CD | ZPENOSKWEKGDCX-UHFFFAOYSA-N | OCC1COC(Cc2ccccc2)O1 |
| Matsutake Alcohol | VSMOENVRRABVKN-MRVPVSSYSA-N | C=C[C@@H](O)CCCCC |
| Isophytol | KEVYVLWNCKMXJX-UHFFFAOYSA-N | C=CC(C)(O)CCCC(C)CCCC(C)CCCC(C)C |
| Eugenol ex Clove | RRAFCDWBNXTKKO-UHFFFAOYSA-N | C=CCc1ccc(O)c(OC)c1 |
| Apo Patchone Coeur | DKKRDMLKVSKFMJ-UHFFFAOYSA-N | CC(C)C1CCC(O)CC1 |
| Ambermax® 50 | MNHPIPXJWPEIDB-UHFFFAOYSA-N | CC(CO)C1=C2C(C)(C)C3CCC2(C3)C(C)(C)CC1 |
| Delta Undecalactone | YZRXRLLRSPQHDK-UHFFFAOYSA-N | CCCCCCC1CCCC(=O)O1 |
| AmberXtreme | HUYXPANWJVOYCI-UHFFFAOYSA-N | CC1C(C)(C)C2CCC3CC(C)(C)OC3C2C1(C)C |
| Amboryl Acetate | ANPXCNWSCGYEMA-UHFFFAOYSA-N | CC(=O)OCC1CCC(C)(C)C23CC4CC2C13C4(C)C |
| Cedramber | HRGPYCVTDOECMG-UHFFFAOYSA-N | COC1(C)CCC23CC1C(C)(C)C2CCC3C |
| Oceanol | GDNUTKRFRLPQAB-UHFFFAOYSA-N | COC(=O)c1ccc(O)c(C)c1O |
| Tobacarol | RBQQAPZKUGJMCR-UHFFFAOYSA-N | CC1CCC2OC2(C)CCC2C1CC2(C)C |
| Benzyl Isovalerate | HVJKZICIMIWFCP-UHFFFAOYSA-N | CC(C)CC(=O)OCc1ccccc1 |
| Cedrene Alpha | IRAQOCYXUMOFCW-YKURLNKLSA-N | CC1=CC[C@@]23CC1C(C)(C)[C@@H]2CC[C@H]3C |
| Anisyl Alcohol | MSHFRERJPWKJFX-UHFFFAOYSA-N | COc1ccc(CO)cc1 |
| Ethyl Linalool | KRLBLPBPZSSIGH-UHFFFAOYSA-N | C=CC(C)(O)CCC=C(C)CC |
| Amber Core | GQBVHGLNSHPKPG-UHFFFAOYSA-N | CCC(O)COC1CCCCC1C(C)(C)C |
| Ethyl 2-methyl butyrate | HCRBXQFHJMCTLF-UHFFFAOYSA-N | CCOC(=O)C(C)CC |
| Antillone™ | DZECLZDTSQJBSU-UHFFFAOYSA-N | C=CCCCCCCC(C)=O |
| Aquaflora | BFTAXJRKNXWMMX-UHFFFAOYSA-N | O=CCC1CC2CC1C1CCCC21 |
| Anisyl Acetone | PCBSXBYCASFXTM-UHFFFAOYSA-N | COc1ccc(CCC(C)=O)cc1 |
| Mahagonate | NXAJZPARDMDLKH-UHFFFAOYSA-N | COC(=O)C1CC2(C(C)C)C=CC1(C)CC2 |
| Methyl Corylone | RLEJFFGSMZQXJX-UHFFFAOYSA-N | CC1=C(O)C(=O)C(C)C1 |
| Cassifix | RTIYSHLNESDZSF-UHFFFAOYSA-N | CC1=CCC(C23CCCC(C)(C2)OC3)C1(C)C |
| Cascalone | RFSYHLFFESXTQI-UHFFFAOYSA-N | CC(C)c1ccc2c(c1)OCC(=O)CO2 |
| Ethyl Levulinate | GMEONFUTDYJSNV-UHFFFAOYSA-N | CCOC(=O)CCC(C)=O |
| Hexyl Tiglate | JTCIUOKKVACNCK-BJMVGYQFSA-N | C/C=C(\C)C(=O)OCCCCCC |
| Almond Pyrazine | VKJIAEQRKBQLLA-UHFFFAOYSA-N | COc1nccnc1C |
| Alcohol C-12 Lauric | LQZZUXJYWNFBMV-UHFFFAOYSA-N | CCCCCCCCCCCCO |
