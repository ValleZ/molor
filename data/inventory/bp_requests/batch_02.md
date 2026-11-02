# Boiling points, batch 2 of 5

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
| Fenchol | IAIHUHQCLTYTSF-UHFFFAOYSA-N | CC12CCC(C1)C(C)(C)C2O |
| Guaiacol | LHGVFZTZFXWLCP-UHFFFAOYSA-N | COc1ccccc1O |
| Ethyl Octanoate | YYZUSRORWSJGET-UHFFFAOYSA-N | CCCCCCCC(=O)OCC |
| Citronellyl Formate | DZNVIZQPWLDQHI-UHFFFAOYSA-N | CC(C)=CCCC(C)CCOC=O |
| Eucalyptol | WEEGYLXZBRQIMU-UHFFFAOYSA-N | CC12CCC(CC1)C(C)(C)O2 |
| Ethyl Tiglate | OAPHLAAOJMTMLY-GQCTYLIASA-N | C/C=C(\C)C(=O)OCC |
| Dihydrocarvyl Acetate | TUSIZTVSUSBSQI-UHFFFAOYSA-N | C=C(C)C1CCC(C)C(OC(C)=O)C1 |
| Ethyl Heptanoate | TVQGDYNRXLTQAP-UHFFFAOYSA-N | CCCCCCC(=O)OCC |
| Gamma Dodecalactone | WGPCZPLRVAWXPW-UHFFFAOYSA-N | CCCCCCCCC1CCC(=O)O1 |
| Dihydrocarveol | KRCZYMFUWVJCLI-UHFFFAOYSA-N | C=C(C)C1CCC(C)C(O)C1 |
| Fenchone Alpha | LHXDLQBQYFFVNW-UHFFFAOYSA-N | CC12CCC(C1)C(C)(C)C2=O |
| Ethyl Decanoate | RGXWDWUGBIJHDO-UHFFFAOYSA-N | CCCCCCCCCC(=O)OCC |
| Damascone Gamma | IXLLBXDECOMIBP-FNORWQNLSA-N | C=C1CCCC(C)(C)C1C(=O)/C=C/C |
| Ethyl Formate | WBJINCZRORDGAQ-UHFFFAOYSA-N | CCOC=O |
| Kephalis | YLNYLLVKHRZLGO-UHFFFAOYSA-N | C=C(OCC)C1C(C)(C)CC(=O)CC1(C)C |
| Velvione | ABRIMXGLNHCLIP-UHFFFAOYSA-N | O=C1CCCC=CCCCCCCCCCC1 |
| Ethyl Furaneol | QJYOEDXNPLUUAR-UHFFFAOYSA-N | CCC1=C(O)C(=O)C(C)O1 |
| Ethyl Acetoacetate | XYIBRDXRRQCHLP-UHFFFAOYSA-N | CCOC(=O)CC(C)=O |
| Ethyl Isobutyrate | WDAXFOBOLVPGLV-UHFFFAOYSA-N | CCOC(=O)C(C)C |
| Citronellal | NEHNMFOYXAPHSD-UHFFFAOYSA-N | CC(C)=CCCC(C)CC=O |
| Hexyl Caprylate | PBGWNXWNCSSXCO-UHFFFAOYSA-N | CCCCCCCC(=O)OCCCCCC |
| Diacetyl | QSJXEFYPDANLFS-UHFFFAOYSA-N | CC(=O)C(C)=O |
| Diethyl Malonate | IYXGSMUGOJNHAZ-UHFFFAOYSA-N | CCOC(=O)CC(=O)OCC |
| Furfural | HYBBIBNJHNGZAN-UHFFFAOYSA-N | O=Cc1ccco1 |
| Geranyl Propionate | BYCHQEILESTMQU-FMIVXFBMSA-N | CCC(=O)OC/C=C(\C)CCC=C(C)C |
| Gamma Heptalactone | VLSVVMPLPMNWBH-UHFFFAOYSA-N | CCCC1CCC(=O)O1 |
| Hexyl Benzoate | UUGLJVMIFJNVFH-UHFFFAOYSA-N | CCCCCCOC(=O)c1ccccc1 |
| Dihydro Beta Ionone | QJJDNZGPQDGNDX-UHFFFAOYSA-N | CC(=O)CCC1=C(C)CCCC1(C)C |
| Farnesene Alpha | CXENHBSYCFFKJS-VDQVFBMKSA-N | C=C/C(C)=C/C/C=C(\C)CCC=C(C)C |
| Damascone Delta | XEJGJTYRUWUFFD-FNORWQNLSA-N | C/C=C/C(=O)C1C(C)C=CCC1(C)C |
| Thiocineole | DXKUHGIKFQWKIT-UHFFFAOYSA-N | CC12CCC(CC1)SC2(C)C |
| Delta Dodecalactone | QRPLZGZHJABGRS-UHFFFAOYSA-N | CCCCCCCC1CCCC(=O)O1 |
| Geranyl Butyrate | ZSBOMYJPSRFZAL-JLHYYAGUSA-N | CCCC(=O)OC/C=C(\C)CCC=C(C)C |
| Geranyl Acetate | HIGQPQRQIQDZMP-FLIBITNWSA-N | CC(=O)OC/C=C(/C)CCC=C(C)C |
| Gamma Octalactone | IPBFYZQJXZJBFQ-UHFFFAOYSA-N | CCCCC1CCC(=O)O1 |
| Cis-3-Hexenyl Isovalerate | AIQLNKITFBJPFO-WAYWQWQTSA-N | CC/C=C\CCOC(=O)CC(C)C |
| Ethyl Caproate | SHZIWNPUGXLXDT-UHFFFAOYSA-N | CCCCCC(=O)OCC |
| Furfuryl Acetate | CKOYRRWBOKMNRG-UHFFFAOYSA-N | CC(=O)OCc1ccco1 |
| Dihydroterpinyl Acetate | HBNHCGDYYBMKJN-UHFFFAOYSA-N | CC(=O)OC(C)(C)C1CCC(C)CC1 |
| Cis-3-Hexenyl Isobutyrate | OSMAJVWUIUORGC-WAYWQWQTSA-N | CC/C=C\CCOC(=O)C(C)C |
| Dihydro Alpha Ionone | JHJCHCSUEGPIGE-UHFFFAOYSA-N | CC(=O)CCC1C(C)=CCCC1(C)C |
| Hexyl Butyrate | XAPCMTMQBXLDBB-UHFFFAOYSA-N | CCCCCCOC(=O)CCC |
| Ethyl Levulinate | GMEONFUTDYJSNV-UHFFFAOYSA-N | CCOC(=O)CCC(C)=O |
| Delta Nonalactone | PXRBWNLUQYZAAX-UHFFFAOYSA-N | CCCCC1CCCC(=O)O1 |
| Ethyl Laurate | MMXKVMNBHPAILY-UHFFFAOYSA-N | CCCCCCCCCCCC(=O)OCC |
| Geosmin | JLPUXFOGCDVKGO-GRYCIOLGSA-N | C[C@@H]1CCC[C@]2(C)CCCC[C@@]12O |
| Ethyl Myristate | MMKRHZKQPFCLLS-UHFFFAOYSA-N | CCCCCCCCCCCCCC(=O)OCC |
| Delta Undecalactone | YZRXRLLRSPQHDK-UHFFFAOYSA-N | CCCCCCC1CCCC(=O)O1 |
| Heptan-2-one | CATSNJVOTSVZJV-UHFFFAOYSA-N | CCCCCC(C)=O |
| Linalool | CDOSHBSSFJOMGT-UHFFFAOYSA-N | C=CC(C)(O)CCC=C(C)C |
| Farnesol | CRDAMVZIKSXKFV-YFVJMOTDSA-N | CC(C)=CCC/C(C)=C/CC/C(C)=C/CO |
| Ethyl Salicylate | GYCKQBWUSACYIF-UHFFFAOYSA-N | CCOC(=O)c1ccccc1O |
| Diethyl Succinate | DKMROQRQHGEIOW-UHFFFAOYSA-N | CCOC(=O)CCC(=O)OCC |
| Floropal | IEZPIUQRQRWIFE-UHFFFAOYSA-N | CC1CC(C)(c2ccccc2)OC(C)O1 |
| Geranyl Formate | FQMZVFJYMPNUCT-YRNVUSSQSA-N | CC(C)=CCC/C(C)=C/COC=O |
| Diethyl Phthalate | FLKPEMZONWLCSK-UHFFFAOYSA-N | CCOC(=O)c1ccccc1C(=O)OCC |
