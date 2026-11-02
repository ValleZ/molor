# Boiling points, batch 3 of 5

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
| Nerol | GLZPCOQZEFWAFX-YFHOEESVSA-N | CC(C)=CCC/C(C)=C\CO |
| Methyl Isoeugenol | NNWHUJCUHAELCL-UHFFFAOYSA-N | CC=Cc1ccc(OC)c(OC)c1 |
| Hexenol-3-CIS | UFLHIIWVXFIJGU-ARJAWSKDSA-N | CC/C=C\CCO |
| Linalyl Formate | JZOCDHMHLGUPFI-UHFFFAOYSA-N | C=CC(C)(CCC=C(C)C)OC=O |
| Matsutake Alcohol | VSMOENVRRABVKN-MRVPVSSYSA-N | C=C[C@@H](O)CCCCC |
| Methyl Isobutyrate | BHIWKHZACMWKOJ-UHFFFAOYSA-N | COC(=O)C(C)C |
| Para Tolyl Aldehyde | FXLOVSHXALFLKQ-UHFFFAOYSA-N | Cc1ccc(C=O)cc1 |
| Lyral | ORMHZBNNECIKOH-UHFFFAOYSA-N | CC(C)(O)CCCC1=CCC(C=O)CC1 |
| Palmitic Acid | IPCSVZSSVZVIGE-UHFFFAOYSA-N | CCCCCCCCCCCCCCCC(=O)O |
| Methyl Chavicol | ZFMSMUAANRJZFM-UHFFFAOYSA-N | C=CCc1ccc(OC)cc1 |
| Hexyl Isovalerate | RSDDTPVXLMVLQE-UHFFFAOYSA-N | CCCCCCOC(=O)CC(C)C |
| Hexyl Isobutyrate | CYHBDKTZDLSRMY-UHFFFAOYSA-N | CCCCCCOC(=O)C(C)C |
| Isophytol | KEVYVLWNCKMXJX-UHFFFAOYSA-N | C=CC(C)(O)CCCC(C)CCCC(C)CCCC(C)C |
| Citral | WTEVQBCEXWBHNA-JXMROGBWSA-N | CC(C)=CCC/C(C)=C/C=O |
| Muscone | ALHUZKCOMYUFRB-UHFFFAOYSA-N | CC1CCCCCCCCCCCCC(=O)C1 |
| Methyl Heptine Carbonate | FRLZQXRXIKQFNS-UHFFFAOYSA-N | CCCCCC#CC(=O)OC |
| Maltol Isobutyrate | LPNUBNQKJRGCSC-UHFFFAOYSA-N | Cc1c(OC(=O)C(C)C)occc1=O |
| Nerolidol | FQTLCLSUCSAZDY-ATGUSINASA-N | C=C[C@@](C)(O)CC/C=C(\C)CCC=C(C)C |
| Isoamyl Salicylate | PMGCQNGBLMMXEW-UHFFFAOYSA-N | CC(C)CCOC(=O)c1ccccc1O |
| Hydrocinnamic Acid | XMIIGOLPHOKFCH-UHFFFAOYSA-N | O=C(O)CCc1ccccc1 |
| Methyl Palmitate | FLIACVVOZYBSBS-UHFFFAOYSA-N | CCCCCCCCCCCCCCCC(=O)OC |
| Methyl Anthranilate | VAMXMNNIEUEQDV-UHFFFAOYSA-N | COC(=O)c1ccccc1N |
| Methyl Octine Carbonate | NTLJTUMJJWVCTL-UHFFFAOYSA-N | CCCCCCC#CC(=O)OC |
| Isobutyl Salicylate | PTXDBYSCVQQBNF-UHFFFAOYSA-N | CC(C)COC(=O)c1ccccc1O |
| Methyl Nicotinate | YNBADRVTZLEFNH-UHFFFAOYSA-N | COC(=O)c1cccnc1 |
| Dimethyl Sulfide | QMMFVYPAHWMCMS-UHFFFAOYSA-N | CSC |
| Labienoxime | MEJYWDUBOCZFFS-FENWIEIGSA-N | C=C/C(C)=C/CC(C)(C)/C(=N\O)C(C)C |
| Isobutyl Benzoate | KYZHGEFMXZOSJN-UHFFFAOYSA-N | CC(C)COC(=O)c1ccccc1 |
| Ocimene | IHPKGUQCSIINRJ-UHFFFAOYSA-N | C=CC(C)=CCC=C(C)C |
| Menthone | NFLGAXVYCFJBMK-BDAKNGLRSA-N | CC(C)[C@@H]1CC[C@@H](C)CC1=O |
| Methyl Acetate | KXKVLQRXCPHEJC-UHFFFAOYSA-N | COC(C)=O |
| Dimethyl Benzyl Carbinyl Acetate | FLUWAIIVLCVEKF-UHFFFAOYSA-N | CC(=O)OC(C)(C)Cc1ccccc1 |
| Linalyl Cinnamate | DPFUEXLIKDHJNB-BUHFOSPRSA-N | C=CC(C)(CCC=C(C)C)OC(=O)/C=C/c1ccccc1 |
| Methyl Nonyl Ketone | KYWIYKKSMDLRDC-UHFFFAOYSA-N | CCCCCCCCCC(C)=O |
| Hexyl Tiglate | JTCIUOKKVACNCK-BJMVGYQFSA-N | C/C=C(\C)C(=O)OCCCCCC |
| Methyl Salicylate | OSWPMRLSEDHDFF-UHFFFAOYSA-N | COC(=O)c1ccccc1O |
| Isoamyl Isovalerate | XINCECQTMHSORG-UHFFFAOYSA-N | CC(C)CCOC(=O)CC(C)C |
| Hydratropic Aldehyde | IQVAERDLDAZARL-QMMMGPOBSA-N | C[C@@H](C=O)c1ccccc1 |
| Methyl Heptenone | UHEPJGULSIKKTP-UHFFFAOYSA-N | CC(=O)CCC=C(C)C |
| Isoamyl Benzoate | MLLAPOCBLWUFAP-UHFFFAOYSA-N | CC(C)CCOC(=O)c1ccccc1 |
| Musk Ketone | WXCMHFPAUCOJIG-UHFFFAOYSA-N | CC(=O)c1c(C)c([N+](=O)[O-])c(C(C)(C)C)c([N+](=O)[O-])c1C |
| Methyl Linoleate | WTTJVINHCBCLGX-NQLNTKRDSA-N | CCCCC/C=C\C/C=C\CCCCCCCC(=O)OC |
| Methyl Butyrate | UUIQMZJEGPQKFD-UHFFFAOYSA-N | CCCC(=O)OC |
| Nopyl Acetate | AWNOGHRWORTNEI-UHFFFAOYSA-N | CC(=O)OCCC1=CCC2CC1C2(C)C |
| Phenyl Acetic Acid | WLJVXDMOQOGPHL-UHFFFAOYSA-N | O=C(O)Cc1ccccc1 |
| Methyl Propionate | RJUFJBKOKNCXHH-UHFFFAOYSA-N | CCC(=O)OC |
| Myristic Acid | TUNFSRHWOTWDNC-UHFFFAOYSA-N | CCCCCCCCCCCCCC(=O)O |
| Para Methyl Acetophenone | GNKZMNRKLCTJAY-UHFFFAOYSA-N | CC(=O)c1ccc(C)cc1 |
| Isoeugenol | BJIOGJUNALELMI-ONEGZZNKSA-N | C/C=C/c1ccc(O)c(OC)c1 |
| Isopropyl Myristate | AXISYYRBXTVTFY-UHFFFAOYSA-N | CCCCCCCCCCCCCC(=O)OC(C)C |
| Para Cresol | IWDCLRJOBJJRNH-UHFFFAOYSA-N | Cc1ccc(O)cc1 |
| Levulinic Acid | JOOXCMJARBKPKM-UHFFFAOYSA-N | CC(=O)CCC(=O)O |
| Myrcene | UAHWPYUMFXYFJY-UHFFFAOYSA-N | C=CC(=C)CCC=C(C)C |
| Indolene 50 | SIKJAQJRHWYJAI-UHFFFAOYSA-N | c1ccc2[nH]ccc2c1 |
| Laevo Menthol | NOOLISFMXDJSKH-UHFFFAOYSA-N | CC1CCC(C(C)C)C(O)C1 |
| Maltol | XPCTZQVDEJYUGT-UHFFFAOYSA-N | Cc1occc(=O)c1O |
