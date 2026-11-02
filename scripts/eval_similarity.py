"""Score representations on how well their geometry matches labelled smell.

Out-of-fold AUC asks a different question from the one this project needs.
AUC rewards ranking molecules correctly *within* one descriptor; the viewer and
the accord work both ask whether two molecules sit near each other *across* all
descriptors. Those come apart: fingerprint v1 has the worst AUC of any model
here and the best similarity structure, because optimising per-label ranking
pushes every prediction towards the class-typical profile, which flattens the
distances between molecules.

So this measures the second axis directly. For every pair of library materials
that appears in GS-LF, the true similarity is the Jaccard overlap of their
human descriptor sets; the predicted similarity is the cosine between their
representation vectors, mean-centred so that a shared baseline profile does not
inflate every pair alike. The reported number is the Spearman correlation
between the two over all pairs.

A representation can be a percept model's descriptor probabilities, a raw
fingerprint, or any learned embedding -- anything that yields one vector per
molecule. That makes unsupervised models comparable with trained ones, which
matters because Uni-Mol and friends emit embeddings, not descriptors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GSLF = "vendor/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
MATERIALS = "library/input/expanded_materials.csv"


def flat_key(smiles: str) -> str | None:
    """A stereochemistry-independent InChIKey for matching against GS-LF.

    The training file's key column is ``nonStereoSMILES`` -- flat by name and
    by construction. A full InChIKey carries stereochemistry in its second
    block, so keying the library on ``Chem.MolToInchiKey`` of its own
    (stereo-bearing) SMILES failed to match every chiral or E/Z molecule:
    citral, limonene, cinnamaldehyde, the cis-3-hexenyl esters, the ionones.
    It reported 302 of 396 library molecules as GS-LF-covered where 374 is
    true, and the same mistake once reported 94 unlabelled molecules against a
    real figure of 22.

    Stereochemistry is removed before the key is taken, so a stereoisomer
    matches the flat training row that describes it. That is the right
    behaviour here precisely because the labels themselves are non-stereo:
    GS-LF does not claim to tell (R)- from (S)-carvone.
    """
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToInchiKey(m)


def gslf_truth() -> tuple[dict[str, np.ndarray], list[str]]:
    """Flat InChIKey -> ground-truth label vector, from the annotated set."""
    import pandas as pd
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(ROOT / GSLF)
    smi_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    tasks = [c for c in df.columns if c not in (smi_col, "descriptors")]
    Y = df[tasks].to_numpy(dtype=float)
    truth = {}
    for i, s in enumerate(df[smi_col].astype(str)):
        k = flat_key(s)
        if k is not None:
            truth.setdefault(k, Y[i])
    return truth, tasks


def material_keys() -> dict[str, str]:
    """Library material name -> flat InChIKey, matching :func:`gslf_truth`."""
    out = {}
    with open(ROOT / MATERIALS, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            k = flat_key(r["smiles"])
            if k is not None:
                out[r["name"]] = k
    return out


def jaccard_matrix(Y: np.ndarray) -> np.ndarray:
    inter = Y @ Y.T
    tot = Y.sum(axis=1)
    union = tot[:, None] + tot[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def centred_cosine(V: np.ndarray) -> np.ndarray:
    V = V - V.mean(axis=0, keepdims=True)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    V = V / n
    return V @ V.T


def load_percept(path: Path) -> dict[str, np.ndarray]:
    j = json.loads(path.read_text(encoding="utf-8"))
    return {n: np.asarray(p, dtype=float) for n, p in zip(j["molecules"], j["probs"])}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """An embedding bundle: names array plus a matching matrix."""
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    return {n: v for n, v in zip(names, np.asarray(z["emb"], dtype=float))}


def load_ecfp() -> dict[str, np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = {}
    with open(ROOT / MATERIALS, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            v = np.zeros(2048)
            for b in gen.GetFingerprint(m).GetOnBits():
                v[b] = 1.0
            out[r["name"]] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percept", action="append", default=[],
                    help="label=path to a percepts JSON")
    ap.add_argument("--embedding", action="append", default=[],
                    help="label=path to an .npz with 'names' and 'emb'")
    ap.add_argument("--ecfp", action="store_true", help="include raw ECFP4 as a control")
    ap.add_argument("--out", default="artifacts/similarity_report.json")
    args = ap.parse_args()

    from scipy.stats import spearmanr

    truth, _ = gslf_truth()
    keys = material_keys()
    # Only materials whose smell is actually recorded by humans can be scored.
    scorable = {n: truth[k] for n, k in keys.items() if k in truth}
    print(f"[sim] {len(scorable)} of {len(keys)} library materials appear in GS-LF")

    sources: list[tuple[str, dict[str, np.ndarray]]] = []
    for spec in args.percept:
        label, _, p = spec.partition("=")
        sources.append((label, load_percept(ROOT / p)))
    for spec in args.embedding:
        label, _, p = spec.partition("=")
        sources.append((label, load_npz(ROOT / p)))
    if args.ecfp:
        sources.append(("ECFP4 (raw)", load_ecfp()))

    report = {}
    for label, vecs in sources:
        names = sorted(n for n in scorable if n in vecs)
        Y = np.stack([scorable[n] for n in names])
        V = np.stack([vecs[n] for n in names])
        J = jaccard_matrix(Y)
        C = centred_cosine(V)
        iu = np.triu_indices(len(names), k=1)
        rho = float(spearmanr(C[iu], J[iu]).statistic)
        report[label] = {"materials": len(names), "pairs": int(len(iu[0])),
                         "spearman": round(rho, 4), "dim": int(V.shape[1])}
        print(f"  {label:24s} {len(names):4d} mats  {len(iu[0]):6d} pairs  "
              f"dim {V.shape[1]:5d}  rho {rho:+.4f}")

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[sim] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
