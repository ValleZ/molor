"""Where does every representation put a specific pair, and why?

Built for the cases the perfumer raises from the bench. The interesting ones are
pairs that are far apart chemically but close perceptually -- Para Cresyl
Acetate smells of horse, Habanolide of pony, and nothing in their structures
says so. A model that only follows the fingerprint must fail them, so they
separate "encodes smell" from "encodes chemistry" better than any aggregate.

For each pair this reports:

* **ECFP4 Tanimoto** -- the chemistry baseline, computed from the molecules
  alone, so it says how much a structure-following model could possibly know.
* **rank in each receptor store** -- out of all other materials, where the pair
  sits. Rank matters more than distance, since the scales differ per model.
* **percept-model agreement** -- cosine between predicted descriptor vectors,
  which is a different question from receptor distance.
* **the labels each dataset actually holds** -- because if the training data
  never links the two, no model trained on it can, and that is a data failure
  rather than a model failure. Checked in GS-LF, Dravnieks and Keller.
* **the perfumer's own notes**, which are the reason the pair is interesting.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402

STORES = [("GCN", "library"), ("MPNN", "library_mpnn_clean")]
PERCEPTS = [("FP v3", "artifacts/percepts_fp3.json"),
            ("OpenPOM", "artifacts/percepts_openpom.json")]


def materials() -> dict[str, str]:
    return {r["name"]: r["smiles"] for r in csv.DictReader(
        io.open(ROOT / "library/input/expanded_materials.csv",
                encoding="utf-8-sig"))}


def notes() -> dict[str, str]:
    p = ROOT / "library/input/impressions.tsv"
    if not p.is_file():
        return {}
    return {r["material"]: r["impression"] for r in csv.DictReader(
        io.open(p, encoding="utf-8"), delimiter="\t")}


def gslf_labels(smi_a: str, smi_b: str):
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    path = ("vendor/openpom/openpom/data/curated_datasets/"
            "curated_GS_LF_merged_4983.csv")
    df = pd.read_csv(ROOT / path)
    col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (col, "descriptors")]
    keys = {}
    for i, s in enumerate(df[col].astype(str)):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            keys.setdefault(Chem.MolToInchiKey(m), i)
    out = {}
    for tag, smi in (("a", smi_a), ("b", smi_b)):
        m = Chem.MolFromSmiles(smi)
        i = keys.get(Chem.MolToInchiKey(m)) if m is not None else None
        out[tag] = (None if i is None else
                    [t for t in tasks if df[t].iloc[i] == 1])
    return out


def graded_labels(which: str, name_a: str, name_b: str, top: int = 6):
    import pandas as pd
    p = ROOT / "data" / "percept" / f"{which}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df = df[df["in_library"].astype(str) != ""]
    drop = {"inchikey", "smiles", "name", "concentration", "in_library",
            "solvent_", "n_subjects", "intensity", "pleasantness", "familiarity"}
    desc = [c for c in df.columns if c not in drop and not c.endswith("__sd")]
    out = {}
    for tag, nm in (("a", name_a), ("b", name_b)):
        sub = df[df["in_library"] == nm]
        if sub.empty:
            out[tag] = None
            continue
        prof = sub[desc].mean()
        out[tag] = [(d, round(float(prof[d]), 1))
                    for d in prof.sort_values(ascending=False).index[:top]
                    if prof[d] > 0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--neighbours", type=int, default=5)
    args = ap.parse_args()
    a, b = args.pair

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    mats, note = materials(), notes()
    for n in (a, b):
        if n not in mats:
            raise SystemExit(f"{n!r} is not in the library")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {n: gen.GetFingerprint(Chem.MolFromSmiles(mats[n])) for n in (a, b)}
    tan = DataStructs.TanimotoSimilarity(fps[a], fps[b])

    print(f"=== {a}  vs  {b} ===")
    print(f"  {a}: {mats[a]}")
    print(f"  {b}: {mats[b]}")
    print(f"\n  ECFP4 Tanimoto {tan:.3f}  "
          f"<- how much a structure-following model could know")
    for n in (a, b):
        if n in note:
            print(f"  note, {n}: {note[n]}")

    print("\n  receptor stores (rank of each as the other's neighbour)")
    for label, store in STORES:
        try:
            entries, logits, _ = Library(name=store).load()
        except Exception as exc:  # noqa: BLE001
            print(f"    {label}: could not load {store}: {exc}")
            continue
        names = [e.name for e in entries]
        if a not in names or b not in names:
            print(f"    {label}: pair not both present")
            continue
        red, _ = preprocess(logits, n_components=20)
        D = squareform(pdist(red))
        i, j = names.index(a), names.index(b)
        di = D[i].copy(); di[i] = np.inf
        dj = D[j].copy(); dj[j] = np.inf
        print(f"    {label:5s} distance {D[i, j]:7.2f}   "
              f"{b} is #{int((di < di[j]).sum()) + 1}/{len(names) - 1} from {a}   "
              f"{a} is #{int((dj < dj[i]).sum()) + 1}/{len(names) - 1} from {b}")
        for src, dd in ((a, di), (b, dj)):
            near = np.argsort(dd)[:args.neighbours]
            print(f"       {src} nearest: "
                  + ", ".join(f"{names[k]} ({dd[k]:.1f})" for k in near))

    print("\n  percept models (cosine between predicted descriptor vectors)")
    import json
    for label, path in PERCEPTS:
        p = ROOT / path
        if not p.is_file():
            continue
        pj = json.loads(p.read_text(encoding="utf-8"))
        P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
        if a not in P or b not in P:
            print(f"    {label}: pair not both predicted")
            continue
        M = np.stack(list(P.values())).mean(axis=0)
        x, y = P[a] - M, P[b] - M
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        desc = pj["descriptors"]
        topa = [desc[k] for k in np.argsort(-P[a])[:5]]
        topb = [desc[k] for k in np.argsort(-P[b])[:5]]
        print(f"    {label:8s} centred cosine {cos:+.3f}")
        print(f"       {a}: {topa}")
        print(f"       {b}: {topb}")

    print("\n  what the training and panel data actually say")
    g = gslf_labels(mats[a], mats[b])
    for tag, nm in (("a", a), ("b", nm_b := b)):
        v = g[tag]
        print(f"    GS-LF, {nm}: " + ("not in the dataset" if v is None else str(v)))
    if g["a"] and g["b"]:
        shared = sorted(set(g["a"]) & set(g["b"]))
        print(f"    GS-LF shared labels: {shared or 'NONE'}")
    for which in ("dravnieks", "keller"):
        gl = graded_labels(which, a, b)
        if gl is None:
            continue
        for tag, nm in (("a", a), ("b", b)):
            v = gl[tag]
            print(f"    {which}, {nm}: "
                  + ("not measured" if v is None else str(v)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
