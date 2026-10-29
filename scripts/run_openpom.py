"""Predict odour descriptors with the OpenPOM ensemble.

OpenPOM is the open replication of the Principal Odor Map (Lee et al., Science
2023): a 2.09M-parameter message-passing network over its own 134-dim atom /
6-dim bond featurisation, shipped as a 10-model ensemble. That is roughly 8x
the parameters of ``GS_LF_baseline_gcn`` (256K) and a far stronger architecture
for the same job, so it is the natural upgrade for the descriptor side of this
project -- which is the side that every validation so far has favoured.

Runs under ``.venv-pom``, a separate environment: OpenPOM needs DeepChem and
its own dependency set, and the main environment is pinned to a working
torch/dgl/dgllife combination that is not worth disturbing.

The task list is parsed out of upstream's own ``predict_odors.py`` rather than
retyped, because 138 descriptor names in the wrong order would silently
mislabel every prediction.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_tasks(predict_py: Path) -> list[str]:
    src = predict_py.read_text(encoding="utf-8")
    m = re.search(r"TASKS\s*=\s*(\[.*?\])", src, re.S)
    if not m:
        raise SystemExit(f"could not find the TASKS list in {predict_py}")
    tasks = ast.literal_eval(m.group(1))
    if len(tasks) != 138:
        print(f"  note: {len(tasks)} tasks, expected 138")
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="vendor/openpom")
    ap.add_argument("--csv", default="library/input/expanded_materials.csv")
    ap.add_argument("--out", default="artifacts/percepts_openpom.json")
    ap.add_argument("--models", type=int, default=10)
    args = ap.parse_args()

    repo = ROOT / args.repo
    tasks = load_tasks(repo / "predict_odors.py")
    print(f"[openpom] {len(tasks)} descriptors")

    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphConvConstants, GraphFeaturizer
    from openpom.models.mpnn_pom import MPNNPOMModel

    rows = list(csv.DictReader(open(ROOT / args.csv, newline="", encoding="utf-8-sig")))
    names = [r["name"] for r in rows]
    smiles = [r["smiles"] for r in rows]

    feat = GraphFeaturizer()
    X = feat.featurize(smiles)
    ok = [i for i, x in enumerate(X) if hasattr(x, "node_features")]
    if len(ok) != len(smiles):
        for i in set(range(len(smiles))) - set(ok):
            print(f"  featurisation failed: {names[i]}")
    names = [names[i] for i in ok]
    ds = dc.data.NumpyDataset(X=X[ok], ids=np.array(names, dtype=object))
    print(f"[openpom] featurised {len(names)} molecules")

    preds = []
    for i in range(args.models):
        d = repo / "models" / "ensemble_models" / f"experiments_{i + 1}"
        ckpt = d / "checkpoint2.pt"
        if not ckpt.is_file():
            print(f"  missing {ckpt}")
            continue
        # Hyperparameters from upstream's ensemble_benchmark notebook, and
        # cross-checked against the checkpoint's own tensor shapes: node 100
        # (project_node_feats 100x134), edge hidden 75 (edge_func 75x6), edge
        # out 100, set2set 2 layers (lstm l0/l1 at hidden 200), ffn [392, 392]
        # then 256 then 138. Only num_step_message_passing is unverifiable from
        # weights, since the layer is reused each step.
        model = MPNNPOMModel(
            n_tasks=len(tasks), batch_size=128, class_imbalance_ratio=None,
            loss_aggr_type="sum",
            node_out_feats=100, edge_hidden_feats=75, edge_out_feats=100,
            num_step_message_passing=5, mpnn_residual=True,
            message_aggregator_type="sum", mode="classification",
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1, readout_type="set2set", num_layer_set2set=2,
            ffn_hidden_list=[392, 392], ffn_embeddings=256,
            ffn_activation="relu", ffn_dropout_p=0.0,
            model_dir=str(d), device_name="cpu")
        model.restore(str(ckpt))
        p = np.asarray(model.predict(ds))
        # DeepChem may return (n, tasks, classes) for multi-class heads; the
        # positive-class column is the odour probability.
        if p.ndim == 3:
            p = p[:, :, -1]
        preds.append(p)
        print(f"[openpom]   model {i + 1}/{args.models}  range "
              f"{p.min():.3f}..{p.max():.3f}", flush=True)

    if not preds:
        raise SystemExit("no models loaded")
    P = np.mean(preds, axis=0)
    print(f"[openpom] ensemble of {len(preds)}: {P.shape}, "
          f"median {np.median(P):.4f}, max {P.max():.4f}")

    (ROOT / args.out).write_text(json.dumps({
        "checkpoint": "openpom ensemble_models/experiments_1..%d" % len(preds),
        "descriptors": tasks,
        "molecules": names,
        "probs": [[round(float(v), 4) for v in row] for row in P],
    }, separators=(",", ":")), encoding="utf-8")
    print(f"[openpom] wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
