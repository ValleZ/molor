# Boiling points, batch 1 of 5

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
| 6-Methyl Quinoline | LUYISICIYVKBTA-UHFFFAOYSA-N | Cc1ccc2ncccc2c1 |
| Cinnamyl Acetate | WJSDHUCWMSHDCR-VMPITWQZSA-N | CC(=O)OC/C=C/c1ccccc1 |
| Acetoin | ROWKJAVDOGWPAT-UHFFFAOYSA-N | CC(=O)C(C)O |
| Allyl Hexanoate | RCSBILYQLVXLJG-UHFFFAOYSA-N | C=CCOC(=O)CCCCC |
| Amyl Salicylate | RANVDUNFZBMTBK-UHFFFAOYSA-N | CCCCCOC(=O)c1ccccc1O |
| Alcohol C-11 Undecylenic | GIEMHYCMBGELGY-UHFFFAOYSA-N | C=CCCCCCCCCCO |
| Celestolide | IKTHMQYJOWTSJO-UHFFFAOYSA-N | CC(=O)c1cc(C(C)(C)C)cc2c1CCC2(C)C |
| 2-Acetyl Thiazole | MOMFXATYAINJML-UHFFFAOYSA-N | CC(=O)c1nccs1 |
| Beta Pinene | WTARULDDTDQWMU-UHFFFAOYSA-N | C=C1CCC2CC1C2(C)C |
| Benzaldehyde | HUMNYLRZRPPJDN-UHFFFAOYSA-N | O=Cc1ccccc1 |
| Cis-3-Hexenyl Butyrate | ZCHOPXVYTWUHDS-WAYWQWQTSA-N | CC/C=C\CCOC(=O)CCC |
| Cis-3-Hexenyl Benzoate | BCOXBEHFBZOJJZ-ARJAWSKDSA-N | CC/C=C\CCOC(=O)c1ccccc1 |
| Cinnamyl Propionate | KGDJMNKPBUNHGY-RMKNXTFCSA-N | CCC(=O)OC/C=C/c1ccccc1 |
| Benzyl Salicylate | ZCTQGTTXIYCGGC-UHFFFAOYSA-N | O=C(OCc1ccccc1)c1ccccc1O |
| Benzophenone | RWCCWEUUXYIKHB-UHFFFAOYSA-N | O=C(c1ccccc1)c1ccccc1 |
| Acetophenone | KWOLFJPFCHCOCG-UHFFFAOYSA-N | CC(=O)c1ccccc1 |
| Acetate C-9 | GJQIMXVRFNLMTB-UHFFFAOYSA-N | CCCCCCCCCOC(C)=O |
| Dimethyl Octenone | WKQWDIFHKYWRTA-UHFFFAOYSA-N | CCC(=O)C(C)CC=C(C)C |
| 2-Acetyl Pyridine | AJKVQEKCUACUMD-UHFFFAOYSA-N | CC(=O)c1ccccn1 |
| Aldehyde C-11 Undecylic | KMPQYAYAQWNLME-UHFFFAOYSA-N | CCCCCCCCCCC=O |
| Benzyl Formate | UYWQUFXKFGHYNT-UHFFFAOYSA-N | O=COCc1ccccc1 |
| Alcohol C-6 | ZSIAUFGUXNUGDI-UHFFFAOYSA-N | CCCCCCO |
| Aldehyde C-7 | FXHGMKSSBGDXIY-UHFFFAOYSA-N | CCCCCCC=O |
| Cedrene Alpha | IRAQOCYXUMOFCW-YKURLNKLSA-N | CC1=CC[C@@]23CC1C(C)(C)[C@@H]2CC[C@H]3C |
| Cis Jasmone | XMLSXPIVAXONDL-PLNGDYQASA-N | CC/C=C\CC1=C(C)CCC1=O |
| Borneol | DTGKSKDOIYIVQL-NQMVMOMDSA-N | CC1(C)[C@@H]2CC[C@]1(C)[C@H](O)C2 |
| Helional | BOPPSUHPZARXTH-UHFFFAOYSA-N | CC(C=O)Cc1ccc2c(c1)OCO2 |
| Amyl Acetate | PGMYKACGEOXYJE-UHFFFAOYSA-N | CCCCCOC(C)=O |
| Benzyl Cinnamate | NGHOLYJTSCBCGC-VAWYXSNFSA-N | O=C(/C=C/c1ccccc1)OCc1ccccc1 |
| Isobutyl Phenyl Acetate | RJASFPFZACBKBE-UHFFFAOYSA-N | CC(C)COC(=O)Cc1ccccc1 |
| Benzyl Isovalerate | HVJKZICIMIWFCP-UHFFFAOYSA-N | CC(C)CC(=O)OCc1ccccc1 |
| Caryophyllene Beta | NPNUFJAVOOONJE-GFUGXAQUSA-N | C=C1CC/C=C(\C)CC[C@@H]2[C@@H]1CC2(C)C |
| Benzyl Alcohol | WVDDGKGOMKODPV-UHFFFAOYSA-N | OCc1ccccc1 |
| Cedrenyl Acetate | AICQDCHSUWFHCC-ZUFFMMDNSA-N | CC(=O)OCC1=CC[C@@]23C[C@@H]1C(C)(C)[C@@H]2CC[C@H]3C |
| Carvacrol | RECUKUPTGUEGMW-UHFFFAOYSA-N | Cc1ccc(C(C)C)cc1O |
| Benzyl Benzoate | SESFRYSPDFLNCH-UHFFFAOYSA-N | O=C(OCc1ccccc1)c1ccccc1 |
| 2-Isopropyl Phenol | CRBJBYGJVIBWIY-UHFFFAOYSA-N | CC(C)c1ccccc1O |
| Caprylic Acid | WWZKQHOCKIZLMA-UHFFFAOYSA-N | CCCCCCCC(=O)O |
| 2-Ethylhexyl Salicylate | FMRHJJZUHUTGKE-UHFFFAOYSA-N | CCCCC(CC)COC(=O)c1ccccc1O |
| (Z)-3-Hexenal | GXANMBISFKBPEX-ARJAWSKDSA-N | CC/C=C\CC=O |
| Allyl Heptanoate | SJWKGDGUQTWDRV-UHFFFAOYSA-N | C=CCOC(=O)CCCCCC |
| Benzyl Propionate | VHOMAPWVLKRQAZ-UHFFFAOYSA-N | CCC(=O)OCc1ccccc1 |
| Aldehyde C-12 Lauric | HFJRKMMYBMWEAD-UHFFFAOYSA-N | CCCCCCCCCCCC=O |
| Alpha Angelica Lactone | QOTQFLOTGBBMEX-UHFFFAOYSA-N | CC1=CCC(=O)O1 |
| Aldehyde C-9 | GYHFUZHODSMOHU-UHFFFAOYSA-N | CCCCCCCCC=O |
| Cinnamyl Cinnamate | NQBWNECTZUOWID-MZXMXVKLSA-N | O=C(/C=C/c1ccccc1)OC/C=C/c1ccccc1 |
| Undecavertol | WSTQLNQRVZNEDV-UHFFFAOYSA-N | CCC=C(C)C(O)CCCCC |
| Camphor | DSSYKIVIOFKYAU-OMNKOJBGSA-N | CC1(C)C2CC[C@]1(C)C(=O)C2 |
| Benzyl Butyrate | VONGZNXBKCOUHB-UHFFFAOYSA-N | CCCC(=O)OCc1ccccc1 |
| Allyl Anthranilate | UCANFCXAKYMFGA-UHFFFAOYSA-N | C=CCOC(=O)c1ccccc1N |
| Cedrol | SVURIXNDRWRAFU-OGMFBOKVSA-N | C[C@@H]1CC[C@H]2C(C)(C)[C@H]3C[C@@]12CC[C@@]3(C)O |
| Benzyl Isobutyrate | UIKJRDSCEYGECG-UHFFFAOYSA-N | CC(C)C(=O)OCc1ccccc1 |
| 2-Isobutyl Quinoline | YKGUUBIPVHRERN-UHFFFAOYSA-N | CC(C)Cc1ccc2ncccc2c1 |
| Cedryl Acetate | HQKQRXZEXPXXIG-GXTNNVKYSA-N | CC(=O)OC1(C)CCC23CC1C(C)(C)[C@@H]2CC[C@H]3C |
| (E)-2-Hexenal | MBDOYVRWFFCFHM-SNAWJCMRSA-N | CCC/C=C/C=O |
| Benzyl Phenylacetate | MIYFJEKZLFWKLZ-UHFFFAOYSA-N | O=C(Cc1ccccc1)OCc1ccccc1 |
