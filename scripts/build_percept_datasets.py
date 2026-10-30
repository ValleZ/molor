"""Prepare graded odour datasets to replace or augment GS-LF.

GS-LF is 4983 molecules x 138 *binary* tags, and three things about it limit
what any model trained on it can do:

1. *No gradation.* A tag is present or absent, so "faintly fruity" and
   "overwhelmingly fruity" are the same training signal.
2. *No concentration.* Every molecule is described once, at an unstated
   strength, although smell character changes with dilution.
3. *An unenforced hierarchy.* 165 molecules carry a fruit tag (banana, pear)
   without the fruity tag; 162 do the same for floral. So "banana" and "fruity"
   are unrelated dimensions, and any similarity computed over these labels
   treats banana and apple as no more alike than banana and woody.

Two Pyrfume archives fix different parts of this, and neither replaces GS-LF
alone:

*dravnieks_1985* -- 160 stimuli over 138 molecules, 146 descriptors, graded as
percent applicability averaged over a panel. Fine-grained (LEMON, GRAPEFRUIT
and ORANGE are separate descriptors) and genuinely continuous, but small.

*keller_2016* -- 480 molecules at two dilutions, rated 0-100 by ~57 subjects
with replicates, on 19 descriptors plus intensity, pleasantness and
familiarity. Coarse descriptors, but it is the only source here with
concentration, perceived intensity, and per-subject spread, so it can say how
much a panel agrees.

Both carry concentration explicitly, which GS-LF cannot express at all.

Two judgements are recorded rather than hidden:

*Blank descriptor ratings in keller_2016 are read as zero.* A subject marks the
two or three descriptors that apply and leaves the other sixteen empty. The
blank rate is 83% and barely moves with intensity (88.5% at intensity 0-5,
82.5% at 60-100), so blanks are "this descriptor does not apply", not missing
data. This follows the DREAM challenge's own convention.

*A blank intensity means the subject smelled nothing, so it is read as zero.*
Every trial carries a can-or-cannot-smell answer, so none are incomplete and
none are dropped. But intensity is blank in 97% of the trials where the subject
reported smelling nothing, and averaging only over the subjects who did smell
something would overstate how strong a molecule is. Pleasantness and
familiarity are left missing for those trials instead of zeroed, since there was
no odour to judge and zero is a position on those scales rather than an absence.

Writes to data/percept/ with a manifest recording provenance and shapes. Runs
under .venv-pyrfume (pyrfume + rdkit).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "percept"

KELLER_META = {"THE ODOR IS:", "CAN OR CAN'T SMELL", "KNOW OR DON'T KNOW THE SMELL"}
KELLER_SCALES = {"HOW STRONG IS THE SMELL?": "intensity",
                 "HOW PLEASANT IS THE SMELL?": "pleasantness",
                 "HOW FAMILIAR IS THE SMELL?": "familiarity"}


def inchikey(smiles: str) -> str | None:
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m) if m is not None else None


def library_keys() -> dict[str, str]:
    import csv

    out = {}
    path = ROOT / "library" / "input" / "expanded_materials.csv"
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            k = inchikey(r["smiles"])
            if k:
                out[k] = r["name"]
    return out


def build_dravnieks(lib: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    import pyrfume

    beh = pyrfume.load_data("dravnieks_1985/behavior_1.csv")   # applicability
    sti = pyrfume.load_data("dravnieks_1985/stimuli.csv")
    mol = pyrfume.load_data("dravnieks_1985/molecules.csv")

    descriptors = list(beh.columns)
    df = beh.join(sti[["CID", "Conc", "Name"]], how="inner")
    df = df.merge(mol[["IsomericSMILES", "name"]], left_on="CID",
                  right_index=True, how="left")
    df = df.rename(columns={"IsomericSMILES": "smiles", "Conc": "concentration"})
    df = df[df["smiles"].notna()].copy()
    df["inchikey"] = [inchikey(s) for s in df["smiles"]]
    df = df[df["inchikey"].notna()].copy()
    df["in_library"] = [lib.get(k, "") for k in df["inchikey"]]

    cols = ["inchikey", "smiles", "name", "concentration", "in_library"] + descriptors
    df = df[cols]
    v = df[descriptors].to_numpy(float)
    meta = {
        "source": "pyrfume dravnieks_1985 (ASTM DS61, doi:10.1520/DS61-EB)",
        "values": "percent applicability, averaged over the panel",
        "stimuli": int(len(df)),
        "molecules": int(df["inchikey"].nunique()),
        "descriptors": len(descriptors),
        "value_min": float(np.nanmin(v)), "value_max": float(np.nanmax(v)),
        "distinct_values": int(len(np.unique(np.round(v[np.isfinite(v)], 4)))),
        "concentration_levels": sorted(map(str, df["concentration"].unique())),
        "in_library_molecules": int(df.loc[df["in_library"] != "", "inchikey"].nunique()),
    }
    return df, meta


def build_keller(lib: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    import pyrfume

    beh = pyrfume.load_data("keller_2016/behavior.csv")
    sti = pyrfume.load_data("keller_2016/stimuli.csv")
    mol = pyrfume.load_data("keller_2016/molecules.csv")

    beh = beh.copy()
    beh["num"] = pd.to_numeric(beh["Value"], errors="coerce")

    # A trial counts as completed if the subject answered the can-smell question
    # or gave an intensity; otherwise it is absent, not zero.
    smelled = beh[beh["MeasurementValue"] == "CAN OR CAN'T SMELL"][
        ["Stimulus", "Subject"]].drop_duplicates()
    intens = beh[(beh["MeasurementValue"] == "HOW STRONG IS THE SMELL?")
                 & beh["num"].notna()][["Stimulus", "Subject"]].drop_duplicates()
    done = pd.concat([smelled, intens]).drop_duplicates()
    done["_done"] = True
    n_trials_total = beh[["Stimulus", "Subject"]].drop_duplicates().shape[0]

    desc_rows = beh[~beh["MeasurementValue"].isin(KELLER_META | set(KELLER_SCALES))]
    desc_rows = desc_rows.merge(done, on=["Stimulus", "Subject"], how="inner")
    # Blank means the descriptor was not applied.
    desc_rows["num"] = desc_rows["num"].fillna(0.0)

    scale_rows = beh[beh["MeasurementValue"].isin(KELLER_SCALES)].merge(
        done, on=["Stimulus", "Subject"], how="inner")
    scale_rows["MeasurementValue"] = scale_rows["MeasurementValue"].map(KELLER_SCALES)

    # A blank intensity marks a subject who smelled nothing, so it is a zero.
    # Dropping those rows instead would average intensity only over the
    # subjects who perceived the odorant and overstate every molecule.
    nothing = beh[(beh["MeasurementValue"] == "CAN OR CAN'T SMELL")
                  & (beh["Value"] == "I can't smell anything")][
        ["Stimulus", "Subject"]].drop_duplicates()
    nothing["_nothing"] = True
    scale_rows = scale_rows.merge(nothing, on=["Stimulus", "Subject"], how="left")
    zero_it = (scale_rows["MeasurementValue"] == "intensity") \
        & scale_rows["num"].isna() & scale_rows["_nothing"].eq(True)
    n_zeroed = int(zero_it.sum())
    scale_rows.loc[zero_it, "num"] = 0.0
    scale_rows = scale_rows[scale_rows["num"].notna()].drop(columns=["_nothing"])
    print(f"[data] keller: {n_zeroed} blank intensities read as 0 "
          f"(subject reported smelling nothing)")

    both = pd.concat([desc_rows, scale_rows], ignore_index=True)
    agg = both.groupby(["Stimulus", "MeasurementValue"])["num"].agg(
        ["mean", "std", "count"]).reset_index()
    mean = agg.pivot(index="Stimulus", columns="MeasurementValue", values="mean")
    sd = agg.pivot(index="Stimulus", columns="MeasurementValue", values="std")
    n = agg.pivot(index="Stimulus", columns="MeasurementValue", values="count")
    sd.columns = [f"{c}__sd" for c in sd.columns]

    descriptors = [c for c in mean.columns if c not in KELLER_SCALES.values()]
    out = mean.join(sd)
    out["n_subjects"] = n.max(axis=1)

    sti = sti.set_index("Stimulus")
    out = out.join(sti[["CIDs", "Concentration", "Solvent"]], how="inner")

    # Most stimuli name a PubChem CID, but a few odorants have none and are
    # identified by CAS instead, so resolve on whichever the row carries.
    out["_cid"] = pd.to_numeric(out["CIDs"], errors="coerce")
    by_cid = mol.drop_duplicates("CID").set_index("CID")[
        ["CanonicalSMILES", "OdorName"]]
    by_cas = mol.dropna(subset=["CAS"]).drop_duplicates("CAS").set_index("CAS")[
        ["CanonicalSMILES", "OdorName"]]
    resolved = out.merge(by_cid, left_on="_cid", right_index=True, how="left")
    miss = resolved["CanonicalSMILES"].isna()
    if miss.any():
        alt = out.loc[miss].merge(by_cas, left_on="CIDs", right_index=True,
                                  how="left")
        for col in ("CanonicalSMILES", "OdorName"):
            resolved.loc[miss, col] = alt[col].to_numpy()
        n_cas = int(resolved.loc[miss, "CanonicalSMILES"].notna().sum())
        print(f"[data] keller: {int(miss.sum())} stimuli lacked a CID; "
              f"{n_cas} resolved by CAS")
    out = resolved.drop(columns=["_cid"])
    out = out.rename(columns={"CanonicalSMILES": "smiles", "OdorName": "name",
                              "Concentration": "concentration"})
    out = out[out["smiles"].notna()].copy()
    out["inchikey"] = [inchikey(s) for s in out["smiles"]]
    out = out[out["inchikey"].notna()].copy()
    out["in_library"] = [lib.get(k, "") for k in out["inchikey"]]

    lead = ["inchikey", "smiles", "name", "concentration", "solvent_",
            "n_subjects", "in_library"]
    out = out.rename(columns={"Solvent": "solvent_"})
    scales = list(KELLER_SCALES.values())
    cols = ([c for c in lead if c in out.columns] + scales + descriptors
            + [f"{c}__sd" for c in scales + descriptors if f"{c}__sd" in out.columns])
    out = out[[c for c in cols if c in out.columns]]

    v = out[descriptors].to_numpy(float)
    meta = {
        "source": "pyrfume keller_2016 (doi:10.1186/s12868-016-0287-2)",
        "values": "mean subject rating 0-100; '__sd' columns are between-subject "
                  "standard deviations",
        "blank_descriptors": "read as 0 (not applied), per DREAM convention",
        "trials_total": int(n_trials_total),
        "trials_completed": int(len(done)),
        "stimuli": int(len(out)),
        "molecules": int(out["inchikey"].nunique()),
        "descriptors": len(descriptors),
        "scales": scales,
        "value_min": float(np.nanmin(v)), "value_max": float(np.nanmax(v)),
        "concentration_levels": sorted(map(str, out["concentration"].unique())),
        "in_library_molecules": int(out.loc[out["in_library"] != "", "inchikey"].nunique()),
    }
    return out, meta


def build_abraham(lib: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Measured human odour detection thresholds.

    This is the quantity perfumery dosing actually turns on, and it is measured
    rather than modelled, so it should be used in preference to any prediction
    wherever it exists. Perceived intensity at a fixed dilution -- what
    keller_2016 records -- is a different variable: linalool rates 57.6 out of
    100 at a 1e-3 dilution there, which reads as mid-range, while its measured
    threshold puts it near the bottom of this library. The second agrees with
    the bench and the first does not.

    The column is Log(1/ODT), so larger means a lower threshold and a more
    potent material. The file also carries Abraham's solvation descriptors
    (E, S, A, B, L, V), which are the inputs to his published linear
    free-energy model; they are experimental parameters rather than anything
    RDKit computes, so they are kept for reference but cannot be derived for a
    new molecule.
    """
    import pyrfume

    beh = pyrfume.load_data("abraham_2012/behavior.csv")
    sti = pyrfume.load_data("abraham_2012/stimuli.csv")
    mol = pyrfume.load_data("abraham_2012/molecules.csv")

    s2c = (dict(zip(sti.index, sti["CID"])) if sti.index.name == "Stimulus"
           else dict(zip(sti["Stimulus"], sti["CID"])))
    cid2smi = dict(zip(mol.index, mol["IsomericSMILES"]))
    cid2name = dict(zip(mol.index, mol["name"]))

    keep = [c for c in ("Log (1/ODT)", "E", "S", "A", "B", "L", "V") if c in beh]
    rows = []
    for stim, r in beh.iterrows():
        odt = r.get("Log (1/ODT)")
        cid = s2c.get(stim)
        smi = cid2smi.get(cid)
        if not isinstance(smi, str) or not np.isfinite(odt):
            continue
        key = inchikey(smi)
        if key is None:
            continue
        rows.append({"inchikey": key, "smiles": smi,
                     "name": cid2name.get(cid, ""),
                     "in_library": lib.get(key, ""),
                     **{c: r.get(c) for c in keep}})
    df = pd.DataFrame(rows).drop_duplicates("inchikey")
    v = df["Log (1/ODT)"].to_numpy(float)
    meta = {
        "source": "pyrfume abraham_2012 (doi:10.1093/chemse/bjr094)",
        "values": "Log(1/ODT): higher means a lower detection threshold, "
                  "i.e. a more potent material. MEASURED, not predicted.",
        "molecules": int(len(df)),
        "value_min": round(float(v.min()), 3),
        "value_max": round(float(v.max()), 3),
        "in_library_molecules": int((df["in_library"] != "").sum()),
        "note": "the right target for dosing; keller_2016 intensity is "
                "perceived strength at a fixed dilution, a different variable "
                "(the two correlate at r 0.59 over the 28 shared materials)",
    }
    return df, meta


def keller_intensity_ceiling(seed: int = 0, splits: int = 50) -> dict:
    """How well the panel's mean intensity predicts itself.

    Between-subject spread on this scale averages 26 points out of 100, so the
    target is noisy and a model cannot be judged against a perfect score. Half
    the subjects are averaged and correlated against the other half; the
    Spearman-Brown correction turns that into the reliability of the whole
    panel, which is the most any model could reach without fitting noise.

    Computed here because it needs the raw per-subject ratings, and this is the
    environment that can reach them. The result travels in the manifest so the
    training script can quote it without pyrfume.
    """
    import pyrfume
    from scipy.stats import pearsonr

    raw = pyrfume.load_data("keller_2016/behavior.csv")
    beh = raw[raw["MeasurementValue"] == "HOW STRONG IS THE SMELL?"].copy()
    beh["num"] = pd.to_numeric(beh["Value"], errors="coerce")
    nothing = raw[(raw["MeasurementValue"] == "CAN OR CAN'T SMELL")
                  & (raw["Value"] == "I can't smell anything")][
        ["Stimulus", "Subject"]].drop_duplicates()
    nothing["_n"] = True
    beh = beh.merge(nothing, on=["Stimulus", "Subject"], how="left")
    beh.loc[beh["num"].isna() & beh["_n"].eq(True), "num"] = 0.0
    beh = beh[beh["num"].notna()]

    subjects = np.array(sorted(beh["Subject"].unique()))
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        ma = beh[beh["Subject"].isin(set(perm[:half]))].groupby(
            "Stimulus")["num"].mean()
        mb = beh[beh["Subject"].isin(set(perm[half:]))].groupby(
            "Stimulus")["num"].mean()
        j = pd.concat([ma, mb], axis=1, join="inner").dropna()
        if len(j) > 20:
            rs.append(float(pearsonr(j.iloc[:, 0], j.iloc[:, 1]).statistic))
    r_half = float(np.mean(rs))
    return {"n_subjects": int(len(subjects)),
            "split_half_r": round(r_half, 4),
            "spearman_brown_full_panel_r": round(2 * r_half / (1 + r_half), 4),
            "splits": len(rs),
            "note": "ceiling for predicting the panel mean intensity"}


def gslf_stats() -> dict:
    path = ("vendor/openpom/openpom/data/curated_datasets/"
            "curated_GS_LF_merged_4983.csv")
    df = pd.read_csv(ROOT / path)
    smi = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi, "descriptors")]
    Y = df[tasks].to_numpy(float)
    return {
        "source": f"openpom {Path(path).name}",
        "values": "binary presence flags",
        "molecules": int(len(df)), "descriptors": len(tasks),
        "distinct_values": 2,
        "concentration_levels": [],
        "labels_per_molecule_mean": round(float(Y.sum(1).mean()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lib = library_keys()
    print(f"[data] library has {len(lib)} materials")

    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "datasets": {}}

    dra, dmeta = build_dravnieks(lib)
    dra.to_csv(out / "dravnieks.csv", index=False, encoding="utf-8")
    manifest["datasets"]["dravnieks"] = dmeta
    print(f"[data] dravnieks: {dmeta['stimuli']} stimuli, {dmeta['molecules']} molecules, "
          f"{dmeta['descriptors']} descriptors, {dmeta['distinct_values']} distinct values, "
          f"{dmeta['in_library_molecules']} in library")

    kel, kmeta = build_keller(lib)
    kel.to_csv(out / "keller.csv", index=False, encoding="utf-8")
    manifest["datasets"]["keller"] = kmeta
    print(f"[data] keller: {kmeta['stimuli']} stimuli, {kmeta['molecules']} molecules, "
          f"{kmeta['descriptors']} descriptors + {len(kmeta['scales'])} scales, "
          f"{kmeta['trials_completed']}/{kmeta['trials_total']} trials used, "
          f"{kmeta['in_library_molecules']} in library")

    abr, ameta = build_abraham(lib)
    abr.to_csv(out / "abraham_thresholds.csv", index=False, encoding="utf-8")
    manifest["datasets"]["abraham_thresholds"] = ameta
    print(f"[data] abraham: {ameta['molecules']} molecules with a MEASURED "
          f"threshold, range {ameta['value_min']} to {ameta['value_max']}, "
          f"{ameta['in_library_molecules']} in library")

    print("[data] computing the keller intensity noise ceiling ...", flush=True)
    manifest["keller_intensity_ceiling"] = keller_intensity_ceiling()
    c = manifest["keller_intensity_ceiling"]
    print(f"[data] panel ceiling: split-half r {c['split_half_r']:.4f}, "
          f"full-panel r {c['spearman_brown_full_panel_r']:.4f} "
          f"over {c['n_subjects']} subjects")

    manifest["datasets"]["gslf"] = gslf_stats()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(f"[data] wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
