"""Predict odour descriptors with the Dense Sense ensemble.

Dense Sense augments the OpenPOM-style network with a second message-passing
branch over an electron-density graph: node features are the diagonal of the
localization/delocalization matrix and edge features its off-diagonal entries.
The two branches are read out separately and concatenated before the head, so
the QM graph never has to align atom-for-atom with the molecular graph.

That matters for this project because the receptor model returns bit-identical
vectors for structural isomers -- the atom-type multiset is the same for
delta- and gamma-decalactone -- whereas their LDM matrices differ. This is the
first representation we have that could separate them.

The catch: the LDM matrices are shipped precomputed, positionally indexed
against upstream's 4812-molecule CSV, and nothing in the repository computes
them for a new molecule. So predictions are only possible for materials that
appear in that set, matched by InChIKey. Everything else is skipped rather
than guessed.

Runs under .venv-pom (DeepChem + openpom + dgl).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Hyperparameters copied from the cell that trained 30_ensemble_models.
MODEL_KWARGS = dict(
    batch_size=25, learning_rate=0.001, class_imbalance_ratio=None,
    loss_aggr_type="sum", node_out_feats=25, edge_hidden_feats=75,
    edge_out_feats=25, num_step_message_passing=5, mpnn_residual=True,
    message_aggregator_type="max", mode="classification",
    number_atom_features=133, number_bond_features=14, n_classes=1,
    readout_type="set2set", num_step_set2set=3, num_layer_set2set=2,
    ffn_hidden_list=[512, 512], ffn_embeddings=256, ffn_activation="relu",
    ffn_dropout_p=0.154493949, ffn_dropout_at_input_no_act=False,
    weight_decay=1.39e-6, self_loop=False, optimizer_name="adam",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="vendor/densesense")
    ap.add_argument("--csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--out", default="artifacts/percepts_densesense.json")
    ap.add_argument("--models", type=int, default=10)
    ap.add_argument("--checkpoint", default="checkpoint2.pt")
    args = ap.parse_args()
    repo = ROOT / args.repo

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import deepchem as dc
    import pandas as pd
    from deepchem.feat import DMPNNFeaturizer

    from densesense_model import MPNNPOMModel, get_graphs

    their_csv = repo / "DMPNNpruned_without hydrogen_curated_GS_LF_merged_4812_QM_cleaned.csv"
    df = pd.read_csv(their_csv)
    tasks = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]
    print(f"[densesense] {len(tasks)} descriptors, {len(df)} reference molecules")

    mtx = np.load(repo / "DMPNNpruned_graph_data_cleaned.npz", allow_pickle=True)["mtx"]
    if len(mtx) != len(df):
        raise SystemExit(f"{len(mtx)} matrices vs {len(df)} rows - misaligned")

    # InChIKey -> (row index, their SMILES). Featurising their SMILES keeps both
    # branches in the same canonical form the matrices were produced from.
    ref = {}
    for i, s in enumerate(df["nonStereoSMILES"]):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            ref.setdefault(Chem.MolToInchiKey(m), (i, str(s)))

    rows = list(csv.DictReader(open(ROOT / args.csv, newline="", encoding="utf-8-sig")))
    names, smis, idxs, skipped = [], [], [], []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        hit = ref.get(Chem.MolToInchiKey(m)) if m is not None else None
        if hit is None:
            skipped.append(r["name"])
            continue
        names.append(r["name"])
        smis.append(hit[1])
        idxs.append(hit[0])
    print(f"[densesense] {len(names)} of {len(rows)} materials have precomputed LDM; "
          f"{len(skipped)} skipped")

    feat = DMPNNFeaturizer()
    X_mol = feat.featurize(smis)
    ok = [k for k, g in enumerate(X_mol) if hasattr(g, "node_features")]
    if len(ok) != len(names):
        for k in set(range(len(names))) - set(ok):
            print(f"  featurisation failed: {names[k]}")
    names = [names[k] for k in ok]
    X = np.asarray([(X_mol[k], get_graphs(np.asarray(mtx[idxs[k]]))) for k in ok],
                   dtype=object)
    ds = dc.data.NumpyDataset(X=X, ids=np.array(names, dtype=object))

    preds = []
    for i in range(args.models):
        d = repo / "30_ensemble_models" / f"experiments_{i + 1}"
        ckpt = d / args.checkpoint
        if not ckpt.is_file():
            print(f"  missing {ckpt}")
            continue
        model = MPNNPOMModel(n_tasks=len(tasks), model_dir=str(d),
                             device_name="cpu", **MODEL_KWARGS)
        model.restore(str(ckpt))
        p = np.asarray(model.predict(ds))
        if p.ndim == 3:
            p = p[:, :, -1]
        preds.append(p)
        print(f"[densesense]   model {i + 1}/{args.models}  "
              f"range {p.min():.3f}..{p.max():.3f}", flush=True)

    if not preds:
        raise SystemExit("no checkpoints loaded")
    P = np.mean(preds, axis=0)
    print(f"[densesense] ensemble of {len(preds)}: {P.shape}, "
          f"median {np.median(P):.4f}, max {P.max():.4f}")

    (ROOT / args.out).write_text(json.dumps({
        "checkpoint": f"densesense 30_ensemble_models/experiments_1..{len(preds)}",
        "descriptors": tasks,
        "molecules": names,
        "skipped_no_ldm": sorted(skipped),
        "probs": [[round(float(v), 4) for v in row] for row in P],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"[densesense] wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
