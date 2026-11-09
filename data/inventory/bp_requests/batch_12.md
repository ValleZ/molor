# Boiling points, batch 12 of 12

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
| Syringa Aldehyde 50% PEA | CIXAYNMKFFQEFU-UHFFFAOYSA-N | Cc1ccc(CC=O)cc1 |
| Ultravanil | ZWQBCCVEOIYNDL-UHFFFAOYSA-N | CCOc1cc(C)ccc1O |
| Rose Crystals | JKRWZLOCPLZZEI-UHFFFAOYSA-N | CC(=O)OC(c1ccccc1)C(Cl)(Cl)Cl |
| Tropicalia™ | FCCGTJAGEHZPBF-UHFFFAOYSA-N | CCOC(=O)C1CC2C=CC1C2 |
| Tomato Thiazole | CMPVUVUNJQERIT-UHFFFAOYSA-N | CC(C)Cc1nccs1 |
| Symroxane | VIHCRGZIINXRHA-UHFFFAOYSA-N | COC1C=C2C(C)(C)C3CCC2(C3)C(C)(C)C1 |
| Syringol | KLIDCXVFHGNTTM-UHFFFAOYSA-N | COc1cccc(OC)c1O |
| Raspberry Glycidate | UMSHXRVBYUKRDF-UHFFFAOYSA-N | CCOC(=O)C1OC1c1ccc(C)cc1 |
| Ylanganate | WVWZECQNFWFVFW-UHFFFAOYSA-N | COC(=O)c1ccccc1C |
| Vetiverol ex Vetiver Haiti | INIOTLARNNSXAE-UHFFFAOYSA-N | CC1=CC(O)CC(C)C2CC(=C(C)C)CC12 |
| Patchone | CCOQPGVQAWPUPE-UHFFFAOYSA-N | CC(C)(C)C1CCC(O)CC1 |
| Phenyl Ethyl Pivalate | DPVYDTACPLLHCF-UHFFFAOYSA-N | CC(C)(C)C(=O)OCCc1ccccc1 |
| Starfleur™ 40 | XUTZHJNXAOHUTM-UHFFFAOYSA-N | CC(C)CC1CCC(CCC=O)CC1 |
| Phenyl Ethyl Tiglate | KVMWYGAYARXPOL-QDEBKDIKSA-N | C/C=C(\C)C(=O)OCCc1ccccc1 |
| Florol | YVSNOTITPICPTB-UHFFFAOYSA-N | CC(C)CC1CC(C)(O)CCO1 |
| Vetikone® | PFDVPCSDZXZDMF-UHFFFAOYSA-N | CC(=O)CC(C)(C)c1ccccc1 |
| Vanillin | MWOOGOJBHIARFG-UHFFFAOYSA-N | COc1cc(C=O)ccc1O |
| Tetrahydrogeraniol | PRNCMAKCNVRZFX-UHFFFAOYSA-N | CC(C)CCCC(C)CCO |
| Phenethyl Alcohol | WRMNZCZEMHIOCP-UHFFFAOYSA-N | OCCc1ccccc1 |
| Pinoacetaldehyde | DDFGFKGJBOILQZ-UHFFFAOYSA-N | CC1(C)C2CC=C(CCC=O)C1C2 |
| Syvertal | LQSZQAHPRJMRET-UHFFFAOYSA-N | CCCCC(CC)C1OCCO1 |
| Woodol® | VJCRIDPUJYUCNH-UHFFFAOYSA-N | CC1(C)C2CCC3(C2)C1C(O)CCC3(C)C |
| Phenyl Ethyl Propionate | HVGZQCSMLUDISR-UHFFFAOYSA-N | CCC(=O)OCCc1ccccc1 |
| Florhydral | OHRBQTOZYGEWCJ-UHFFFAOYSA-N | CC(C)c1cccc(C(C)CC=O)c1 |
| TCD Alcohol DM | OTLDLKLSNZMTTA-UHFFFAOYSA-N | OCC1CC2CC1C1CCC(CO)C21 |
| Sinensal Natural ex Orange | NOPLRNXKHZRXHT-YFVJMOTDSA-N | C=CC(=C)CC/C=C(\C)CC/C=C(\C)C=O |
| Ysamber® K | RTPMDRYHLHGFGT-XEZPLFJOSA-N | CC1(C)[C@H]2CC[C@]3(C2)[C@H]1C1(CCC3(C)C)OCCO1 |
| Praslin Pyrazine | LCZUOKDVTBMCMX-UHFFFAOYSA-N | Cc1cnc(C)cn1 |
| Vetimoss | ALVJDUNBMKMTDC-UHFFFAOYSA-N | COc1ccc(OC)c(C(C)(C)C)c1 |
| Vanitrope | RADIRXJQODWKGQ-HWKANZROSA-N | C/C=C/c1ccc(OCC)c(O)c1 |
| Traseolide | IMRYETFJNLKUHK-UHFFFAOYSA-N | CC(=O)c1cc2c(cc1C)C(C)(C)C(C)C2C(C)C |
| Theobromal | KCDXJAYRVLXPFO-UHFFFAOYSA-N | COc1cc(C=O)cc(OC)c1O |
| Rose Oxide Laevo | CZCBTSFUTPZVKJ-NXEZZACHSA-N | CC(C)=C[C@@H]1C[C@H](C)CCO1 |
| Velberry™ | ZFFGZCTVZNNVLP-UHFFFAOYSA-N | CC(=O)OCC(C)(C)COC(C)=O |
| Vertosine | KDSJAOIHFZMUNK-SDNWHVSQSA-N | COC(=O)c1ccccc1N/C=C1\CCC(C)=CC1C |
| Z11™ 10 MIP | PHNCACYNYORRNS-UHFFFAOYSA-N | CC12CCC3C(CCC4C(C)(C)CCCC43C)(CO1)O2 |
| Tamisone | YINCMLNASBORJF-UHFFFAOYSA-N | O=C1CCCC2C3C=CC(C3)C12 |
| Sulfurol (Milky) | BKAWJIRCKVUVED-UHFFFAOYSA-N | Cc1ncsc1CCO |
| Sotolone | UNYNVICDCJHOPO-UHFFFAOYSA-N | CC1=C(O)C(=O)OC1C |
| Sandalore | NGYMOTOXXHCHOC-UHFFFAOYSA-N | CC1=CCC(CCC(C)C(C)O)C1(C)C |
| Phenyl Ethyl Cinnamate | MJQVZIANGRDJBT-VAWYXSNFSA-N | O=C(/C=C/c1ccccc1)OCCc1ccccc1 |
| Vetikolacetat® | MXIGGIZQUGDKAP-UHFFFAOYSA-N | CC(=O)OC(C)CC(C)(C)c1ccccc1 |
| Benzyl Isoeugenol | YKSSSKBJDZDZTD-XVNBXDOJSA-N | C/C=C/c1ccc(OCc2ccccc2)c(OC)c1 |
| Pastry Carbaldehyde | OUDFNZMQXZILJD-UHFFFAOYSA-N | Cc1ccc(C=O)o1 |
| Sakura Salicylate | KFBPVHFUHYKWIF-UHFFFAOYSA-N | CC(C)OCCOC(=O)c1ccccc1O |
| Sublimolide™ | QILMAYXCYBTEDM-IWQZZHSRSA-N | O=C1CCCCCCC/C=C\CCCCCCO1 |
| Valencene | QEBNYNLSCGVZOH-NFAWXSAZSA-N | C=C(C)[C@@H]1CCC2=CCC[C@@H](C)[C@]2(C)C1 |
| Salicynile® | UYBGOPSMJQAVEH-KPKJPENVSA-N | CCC/C=C(\C#N)c1ccccc1 |
