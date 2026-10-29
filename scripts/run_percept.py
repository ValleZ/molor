"""Predict GS-LF odour descriptors for the material library.

``GS_LF_baseline_gcn.pth`` maps a molecule straight to 152 odour descriptors
with no receptor information at all, so it is the direct molecule -> perception
route rather than the receptor route the rest of this project takes. That makes
it a useful independent check: its vocabulary is professional rather than
keyword-derived, and its predictions can be compared against the perfumer's
own notes.

Descriptor order is the task-column order of
``NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv`` after dropping the
metadata columns, which is how upstream's ``GS_LF`` dataset builds its label
matrix. Get that order wrong and every label is silently mislabelled, so the
column set is asserted to be exactly 152 wide.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.config import resolve_paths  # noqa: E402
from orvector.upstream import disable_anomaly_detection  # noqa: E402

META_COLS = ["IsomericSMILES", "Stimulus", "CID", "IUPACName", "MolecularWeight", "name"]


def descriptor_names(csv_path: Path) -> list[str]:
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=1)
    names = [c for c in df.columns if c not in META_COLS]
    if len(names) != 152:
        raise SystemExit(f"expected 152 descriptor columns, found {len(names)}")
    return names


def build(state: dict):
    from dgllife.model import GCNPredictor

    n_tasks = int(state["predict.predict.4.weight"].shape[0])
    hidden = []
    i = 0
    while f"gnn.gnn_layers.{i}.graph_conv.weight" in state:
        hidden.append(int(state[f"gnn.gnn_layers.{i}.graph_conv.weight"].shape[1]))
        i += 1
    in_feats = int(state["gnn.gnn_layers.0.graph_conv.weight"].shape[0])
    residual = "gnn.gnn_layers.0.res_connection.weight" in state
    batchnorm = "gnn.gnn_layers.0.bn_layer.weight" in state
    model = GCNPredictor(
        in_feats=in_feats,
        hidden_feats=hidden,
        residual=[residual] * len(hidden),
        batchnorm=[batchnorm] * len(hidden),
        dropout=[0.0] * len(hidden),
        predictor_hidden_feats=int(state["predict.predict.1.weight"].shape[0]),
        predictor_dropout=0.0,
        n_tasks=n_tasks,
    )
    disable_anomaly_detection()
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, n_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="GS_LF_baseline_gcn.pth")
    ap.add_argument("--out", default="artifacts/percepts_baseline.json")
    ap.add_argument("--csv", default=None,
                    help="score a CSV of name,smiles instead of the stored library")
    args = ap.parse_args()

    paths = resolve_paths()
    names = descriptor_names(paths.gs_lf_csv)
    print(f"[percept] {len(names)} descriptors, first five: {names[:5]}")

    blob = torch.load(paths.weights / args.checkpoint, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if "model_state_dict" in blob else blob
    model, n_tasks = build(state)
    print(f"[percept] strict load ok: {n_tasks} tasks, "
          f"{sum(v.numel() for v in state.values()):,} parameters")

    import dgl
    from dgllife.utils import CanonicalAtomFeaturizer, SMILESToBigraph

    s2g = SMILESToBigraph(add_self_loop=True, node_featurizer=CanonicalAtomFeaturizer())

    if args.csv:
        from types import SimpleNamespace

        entries = []
        with open(ROOT / args.csv, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                entries.append(SimpleNamespace(name=r["name"], canonical_smiles=r["smiles"]))
    else:
        from orvector.store import Library

        entries, _, _ = Library(paths).load()
    graphs, keep = [], []
    for e in entries:
        g = s2g(e.canonical_smiles)
        if g is None:
            print(f"  skip {e.name}")
            continue
        graphs.append(g)
        keep.append(e)

    with torch.inference_mode():
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        logits = model(bg, bg.ndata.pop("h"))
    probs = torch.sigmoid(logits).numpy().astype(np.float32)
    print(f"[percept] predicted {probs.shape[0]} molecules x {probs.shape[1]} descriptors")
    print(f"[percept] prob range {probs.min():.4f}..{probs.max():.4f}, "
          f"median {np.median(probs):.4f}")

    out = {
        "checkpoint": args.checkpoint,
        "descriptors": names,
        "molecules": [e.name for e in keep],
        "probs": [[round(float(v), 4) for v in row] for row in probs],
    }
    Path(ROOT / args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"[percept] wrote {ROOT / args.out}")

    # --- does it agree with the perfumer's notes? ---
    notes_path = ROOT / "library" / "input" / "impressions.tsv"
    if not notes_path.is_file():
        return 0
    with open(notes_path, encoding="utf-8") as fh:
        notes = {r["material"]: r["impression"].lower()
                 for r in csv.DictReader(fh, delimiter="\t")}

    print("\n" + "=" * 86)
    print("PREDICTED DESCRIPTORS vs YOUR NOTES")
    print("=" * 86)
    hits = 0
    scored = 0
    for i, e in enumerate(keep):
        note = notes.get(e.name)
        if not note:
            continue
        scored += 1
        top = np.argsort(-probs[i])[:5]
        labels = [names[t] for t in top]
        hit = [l for l in labels if l.lower() in note]
        if hit:
            hits += 1
        if i < 26:
            mark = "  <- " + ",".join(hit) if hit else ""
            print(f"\n  {e.name[:34]:36s} {', '.join(f'{names[t]}({probs[i,t]:.2f})' for t in top)}{mark}")
            print(f"  {'':36s} your note: {note[:62]}")
    print(f"\n  top-5 prediction literally appears in the note: {hits}/{scored} "
          f"({100 * hits / scored:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
