# Boiling points, batch 6 of 12

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
| Phenoxanol® | OXYRENDGHPGWKV-UHFFFAOYSA-N | CC(CCO)CCc1ccccc1 |
| Bornafix® | YCOHHRPARVZBHK-UHFFFAOYSA-N | CC(CO)COC1CC2CCC1(C)C2(C)C |
| Allyl Cyclohexyl Propionate | TWXUTZNBHUWMKJ-UHFFFAOYSA-N | C=CCOC(=O)CCC1CCCCC1 |
| Methyl Octine Carbonate | NTLJTUMJJWVCTL-UHFFFAOYSA-N | CCCCCCC#CC(=O)OC |
| Cedroxyde® | PUXIGTZUWRBNMW-RRMOSLQNSA-N | CC1(C)C/C=C\CC/C=C\CCC2OC21C |
| Givescone® | CQHUPYQUERYPML-UHFFFAOYSA-N | CCOC(=O)C1C(CC)=CCCC1(C)C |
| Ambrinol | GPVOTKFXWGURGP-UHFFFAOYSA-N | CC1(O)CCC2C(=CCCC2(C)C)C1 |
| Rhodinol Ex Geranium Bourbon | JGQFVRIQXUFPAH-JTQLQIEISA-N | C=C(C)CCC[C@H](C)CCO |
| Trisamber | YFHKETGXYQOMGB-UHFFFAOYSA-N | CC1CC2CCC3C(C2O1)C(C)(C)C(C)C3(C)C |
| Aldehyde C-10 | KSMVZQYAVGTKIV-UHFFFAOYSA-N | CCCCCCCCCC=O |
| Rhubofix | VKPRTBDRPNWOGL-UHFFFAOYSA-N | CC1=CCC2C3CC(C2C1)C1(C3)OC1C |
| Cervolide® | MRMOPGVGWFNHIN-UHFFFAOYSA-N | O=C1CCCCCCCCCCOCCCCO1 |
| Canthoxal | VLFBSPUPYFTTNF-UHFFFAOYSA-N | COc1ccc(CC(C)C=O)cc1 |
| Velvione | ABRIMXGLNHCLIP-UHFFFAOYSA-N | O=C1CCCC=CCCCCCCCCCC1 |
| Nerolin Yara Yara | LUZDYPLAQQGJEA-UHFFFAOYSA-N | COc1ccc2ccccc2c1 |
| Isomethyl Ionone Alpha | JRJBVWJSTHECJK-UHFFFAOYSA-N | CC(=O)C(C)=CC1C(C)=CCCC1(C)C |
| Vertenex | MBZRJSQZCBXRGK-UHFFFAOYSA-N | CC(=O)OC1CCC(C(C)(C)C)CC1 |
| Methyl Pamplemousse | RDHNTAXPFZIMDN-UHFFFAOYSA-N | COC(OC)C(C)(C)CC=C(C)C |
| Aldehyde C-12 Lauric | HFJRKMMYBMWEAD-UHFFFAOYSA-N | CCCCCCCCCCCC=O |
| Phenoxyethyl Isobutyrate | MJTPMXWJHPOWGH-UHFFFAOYSA-N | CC(C)C(=O)OCCOc1ccccc1 |
| Methyl Jasmonate | GEWDNTWNSAZUDX-UHFFFAOYSA-N | CCC=CCC1C(=O)CCC1CC(=O)OC |
| Damascone Delta | XEJGJTYRUWUFFD-FNORWQNLSA-N | C/C=C/C(=O)C1C(C)C=CCC1(C)C |
| Polysantol® | QZFSNJAQFWEXEA-MDZDMXLPSA-N | CC1=CCC(/C=C/C(C)(C)C(C)O)C1(C)C |
| Nectaryl | KNHGOYVXAHUDHP-UHFFFAOYSA-N | CC1=CCC(C(C)CC2CCCC2=O)CC1 |
| Romandolide® | PDUFSBFWOJELLV-UHFFFAOYSA-N | CCC(=O)OCC(=O)OC(C)C1CCCC(C)(C)C1 |
| Exaltenone | JIYIHILAGIZIMD-CLFYSBASSA-N | O=C1CC/C=C\CCCCCCCCCC1 |
| Habanolide | ZYXGECMFJMLZNA-UHFFFAOYSA-N | O=C1CCCCCCCCCC=CCCCO1 |
| 2-Isobutyl Quinoline | YKGUUBIPVHRERN-UHFFFAOYSA-N | CC(C)Cc1ccc2ncccc2c1 |
| Cedryl Acetate | HQKQRXZEXPXXIG-GXTNNVKYSA-N | CC(=O)OC1(C)CCC23CC1C(C)(C)[C@@H]2CC[C@H]3C |
| Lemonile | DHJVLXVXNFUSMU-UHFFFAOYSA-N | CCC(C)=CCCC(C)=CC#N |
| Cosmone | ZFGBKXBPHOUSJX-CLFYSBASSA-N | CC1C/C=C\CCCCCCCCC(=O)C1 |
| Vertoliff | CYGGTUUCLIFNIU-UHFFFAOYSA-N | CC1C=CCCC1(C)C=O |
| Diphenyl Oxide | USIUVYZYUHIAEV-UHFFFAOYSA-N | c1ccc(Oc2ccccc2)cc1 |
| Aphermate | NFASPEPDTMCBEN-UHFFFAOYSA-N | CC(OC=O)C1CCCC(C)(C)C1 |
| Koavone® | IXIYWQIFBRZMNR-HJWRWDBZSA-N | CC(=O)/C(C)=C(/C)C(C)C(C)(C)C |
| Dimetol® | HGDVHRITTGWMJK-UHFFFAOYSA-N | CC(C)CCCC(C)(C)O |
| Muscenone® | ZLICVGXIPCPPMS-UHFFFAOYSA-N | CC1=CCCCCCCCCCCCCC1=O |
| Geranyl Acetone | HNZUNIKWNYHEJJ-FMIVXFBMSA-N | CC(=O)CC/C=C(\C)CCC=C(C)C |
| Vetiver Acetate 53281 | UAVFEMBKDRODDE-UHFFFAOYSA-N | CC(=O)OC1C=C(C)C2CC(=C(C)C)CC2C(C)C1 |
| Iralia® | VPKMGDRERYMTJX-CMDGGOBGSA-N | CCC(=O)/C=C/C1C(C)=CCCC1(C)C |
| Scentenal | UOGMZEGKCNHMBF-UHFFFAOYSA-N | COC1CC2CC1C1CC(C=O)CC21 |
| Florex® | LLVFKEOMLYDSGT-FARCUNLSSA-N | C/C=C1\CC2CC1C1CCC(=O)OC21 |
| Coranol | KZZASWGRLOTITL-UHFFFAOYSA-N | CC(C)(O)CCC1CCCCC1 |
| Sclareolate® | KPUAQAYQMYWKBV-VIFPVBQESA-N | CCCOC(=O)[C@H](C)OC(C)(C)CC |
| Cis-3-Hexenyl Benzoate | BCOXBEHFBZOJJZ-ARJAWSKDSA-N | CC/C=C\CCOC(=O)c1ccccc1 |
| Fleuramone™ | PJXHBTZLHITWFX-UHFFFAOYSA-N | CCCCCCCC1CCCC1=O |
| Aldehyde C-11 Undecylenic | OFHHDSQXFXLTKC-UHFFFAOYSA-N | C=CCCCCCCCCC=O |
| Adoxal | UEGBWDUVDAKUGA-UHFFFAOYSA-N | CC(C)=CCCC(C)CCCC(C)C=O |
| Carbinol Muguet | YXVSKJDFNJFXAJ-UHFFFAOYSA-N | CC(C)(O)CCc1ccccc1 |
| Phenyl Ethyl Phenyl Acetate | ZOZIRNMDEZKZHM-UHFFFAOYSA-N | O=C(Cc1ccccc1)OCCc1ccccc1 |
| Dynascone | OEVIJAZJVZDBQL-UHFFFAOYSA-N | C=CCCC(=O)C1=CCCC(C)(C)C1 |
| Benzyl Acetone | AKGGYBADQZYZPD-UHFFFAOYSA-N | CC(=O)CCc1ccccc1 |
| Evernyl | UUQHKWMIDYRWHH-UHFFFAOYSA-N | COC(=O)c1c(C)cc(O)c(C)c1O |
| Spirogalbanone | WPFFGMCNILGTEA-UHFFFAOYSA-N | C=CCCC(=O)C1=CCCC2(CCCC2)C1 |
| Musk Ketone | WXCMHFPAUCOJIG-UHFFFAOYSA-N | CC(=O)c1c(C)c([N+](=O)[O-])c(C(C)(C)C)c([N+](=O)[O-])c1C |
| Maltol | XPCTZQVDEJYUGT-UHFFFAOYSA-N | Cc1occc(=O)c1O |
