"""Score the material library with the MPNN-encoder MolOR checkpoint.

The MPNN variant is the interesting comparison because it actually consumes
bond features (``CanonicalBondFeaturizer(self_loop=True)``, 13 dims), which the
GCN variant discards entirely -- upstream builds the GCN with
``edge_featurizer=None``. If the GCN's inability to tell an aryl acetate from a
benzylic acetate is a featurisation problem rather than a training-data
problem, this is where it should show up.

Architecture is derived from the checkpoint's own tensor shapes, not from
``data/configures/M2OR_Pairs/MolOR_MPNN_canonical.json``, which disagrees with
the weights on ``node_out_feats`` (128 vs 256) and ``edge_hidden_feats``
(64 vs 128). ``num_step_message_passing`` cannot be read back from the weights
at all -- one message-passing layer is reused every step and the GRU is shared
-- so it is an explicit argument and the caller records which value was used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.config import ESM_DIM, MAX_NODE_LEN, SEQ_PAD_LEN, resolve_paths  # noqa: E402
from orvector.embeddings import load_panel_embeddings, panel_residue_masks  # noqa: E402
from orvector.panel import panel_hash, read_panel  # noqa: E402
from orvector.store import Entry, Library  # noqa: E402
from orvector.upstream import disable_anomaly_detection, molor_predictor_cls  # noqa: E402

BOND_FEATS = 13


def derive_mpnn_config(sd: dict) -> dict:
    """Read the MPNN architecture back out of the weights."""
    in_feats = int(sd["gnn.project_node_feats.0.weight"].shape[1])
    node_out = int(sd["gnn.project_node_feats.0.weight"].shape[0])
    edge_in = int(sd["gnn.gnn_layer.edge_func.0.weight"].shape[1])
    edge_hidden = int(sd["gnn.gnn_layer.edge_func.0.weight"].shape[0])
    edge_out = int(sd["gnn.gnn_layer.edge_func.2.weight"].shape[0])
    if edge_out != node_out * node_out:
        raise SystemExit(
            f"edge network outputs {edge_out}, expected {node_out}^2 = {node_out ** 2}"
        )
    prot = int(sd["cross_attn.query_transform_tensor1.weight"].shape[1])
    q1_out = int(sd["cross_attn.query_transform_tensor1.weight"].shape[0])
    agg_in = int(sd["cross_attn.linear1.weight"].shape[1])
    mlp_in = int(sd["predict.predict.1.weight"].shape[1])
    return {
        "in_feats": in_feats,
        "node_out_feats": node_out,
        "edge_in_feats": edge_in,
        "edge_hidden_feats": edge_hidden,
        "prot_feats": prot,
        "add_feats": prot,
        "mol2_prot": q1_out == prot and agg_in == prot,
        "gnn_attended_feats": mlp_in - prot,
        "predictor_hidden_feats": int(sd["predict.predict.1.weight"].shape[0]),
        "n_tasks": int(sd["predict.predict.4.weight"].shape[0]),
    }


def build(sd: dict, steps: int, device: str):
    cfg = derive_mpnn_config(sd)
    if cfg["edge_in_feats"] != BOND_FEATS:
        raise SystemExit(f"checkpoint expects {cfg['edge_in_feats']} bond features")
    model = molor_predictor_cls()(
        in_feats=cfg["in_feats"],
        node_out_feats=cfg["node_out_feats"],
        edge_in_feats=cfg["edge_in_feats"],
        edge_hidden_feats=cfg["edge_hidden_feats"],
        num_step_message_passing=steps,
        model_encoder="MPNN",
        add_feats=cfg["add_feats"],
        prot_feats=cfg["prot_feats"],
        gnn_attended_feats=None,
        mol2_prot=cfg["mol2_prot"],
        max_seq_len=SEQ_PAD_LEN,
        max_node_len=MAX_NODE_LEN,
        predictor_hidden_feats=cfg["predictor_hidden_feats"],
        predictor_dropout=0.05,
        n_tasks=cfg["n_tasks"],
        activation=[F.relu],
    )
    disable_anomaly_detection()
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(torch.device(device))
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6,
                    help="num_step_message_passing (not recoverable from weights)")
    ap.add_argument("--csv", default="library/input/final_materials.csv")
    ap.add_argument("--name", default=None, help="library directory under artifacts/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="recompute molecules already stored")
    args = ap.parse_args()

    name = args.name or f"library_mpnn_s{args.steps}"
    paths = resolve_paths()
    ckpt = paths.weights / "m2or_MolOR_MPNN_esm_650m_unweighted.pth"
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    model, cfg = build(sd, args.steps, args.device)
    print(f"[mpnn] derived config: {cfg}")
    print(f"[mpnn] num_step_message_passing={args.steps} (assumed, see module docstring)")
    print("[mpnn] strict weight load succeeded")

    receptors = read_panel(paths.receptors_tsv)
    embeddings = load_panel_embeddings(paths, n_receptors=len(receptors))
    masks = panel_residue_masks(receptors)

    from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors

    from orvector.inference import mol_to_graph

    RDLogger.DisableLog("rdApp.*")
    _node_f = CanonicalAtomFeaturizer()
    _edge_f = CanonicalBondFeaturizer(self_loop=True)

    def s2g(mol):
        """As SMILESToBigraph, but from a molecule so hydrogen bookkeeping can
        be normalised first -- see orvector.inference.normalise_hydrogens."""
        return mol_to_graph(mol, add_self_loop=True,
                            node_featurizer=_node_f, edge_featurizer=_edge_f)

    # Append-only: existing molecules are kept and only new InChIKeys computed,
    # so growing the library costs the new molecules rather than all of them.
    existing_entries, existing_logits = [], None
    lib_pre = Library(paths, name=name)
    if lib_pre.exists() and not args.force:
        existing_entries, existing_logits, _ = lib_pre.load()
        print(f"[mpnn] library already holds {len(existing_entries)} molecules")
    known = {e.inchikey for e in existing_entries}

    rows = list(csv.DictReader(open(ROOT / args.csv, newline="", encoding="utf-8-sig")))
    mols, entries = [], []
    for r in rows:
        smi = r["smiles"].strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  skip unparseable {r['name']}")
            continue
        can = Chem.MolToSmiles(mol)
        key = Chem.MolToInchiKey(mol)
        if key in known:
            continue
        known.add(key)
        g = s2g(mol)
        if g is None or g.num_nodes() > MAX_NODE_LEN:
            print(f"  skip {r['name']}")
            continue
        mols.append(g)
        entries.append(Entry(
            molecule_id=Chem.MolToInchiKey(mol), name=r["name"], cls=r.get("class", ""),
            input_smiles=smi, canonical_smiles=can, inchikey=Chem.MolToInchiKey(mol),
            formula=rdMolDescriptors.CalcMolFormula(mol), n_atoms=g.num_nodes(),
            added_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ))

    import dgl

    dev = torch.device(args.device)
    R = len(receptors)
    logits = np.zeros((len(mols), R), dtype=np.float32)

    def full_forward(g, lo, hi):
        """Upstream's own forward, one receptor block at a time."""
        n = hi - lo
        bg = dgl.batch([g] * n)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg = bg.to(dev)
        nf = bg.ndata.pop("h").to(dev)
        ef = bg.edata.pop("e").to(dev)
        out = model(
            bg, nf,
            torch.from_numpy(np.stack([np.asarray(embeddings[k]) for k in range(lo, hi)])).to(dev),
            torch.from_numpy(masks[lo:hi]).to(dev),
            torch.from_numpy(np.stack([node_mask(g)] * n)).to(dev),
            dev, edge_feats=ef,
        )
        return out.reshape(n).float().cpu().numpy()

    def node_mask(g):
        nm = np.zeros(MAX_NODE_LEN, dtype=np.float32)
        nm[: g.num_nodes()] = 1.0
        return nm

    def fast_forward(g, lo, hi):
        """Same computation with the GNN hoisted out of the receptor loop.

        ``node_feats`` depends only on the molecule, but the model recomputes
        it for every receptor block. For 386 receptors that is 24x of wasted
        message passing, and the MPNN's edge network (128 -> 256x256 per edge,
        six steps) makes it the dominant cost. Hoisting it is exact, not an
        approximation: the tiled rows are identical to what ``dgl.unbatch``
        would produce from repeated copies of one graph.
        """
        n = hi - lo
        sg = dgl.batch([g]).to(dev)
        nf = sg.ndata.pop("h").to(dev)
        ef = sg.edata.pop("e").to(dev)
        node_feats = model.gnn(sg, nf, ef)                      # (n_atoms, F)
        padded = torch.zeros((MAX_NODE_LEN, node_feats.shape[1]), device=dev)
        padded[: node_feats.shape[0]] = node_feats
        batch_node_feats = padded.unsqueeze(0).expand(n, -1, -1).contiguous()

        add_feats = torch.from_numpy(
            np.stack([np.asarray(embeddings[k]) for k in range(lo, hi)])
        ).to(dev)
        seq_mask = torch.from_numpy(masks[lo:hi]).to(dev)
        nmask = torch.from_numpy(np.stack([node_mask(g)] * n)).to(dev)

        add_feats = model.prot_norm(add_feats)
        batch_node_feats = model.mol_norm(batch_node_feats)
        feats = model.cross_attn(add_feats, batch_node_feats, seq_mask, nmask)
        return model.predict(model.feat_norm(feats)).reshape(n).float().cpu().numpy()

    # Prove the hoisted path matches upstream's forward before relying on it.
    with torch.inference_mode():
        ref = full_forward(mols[0], 0, 8)
        got = fast_forward(mols[0], 0, 8)
    delta = float(np.abs(ref - got).max())
    print(f"[mpnn] hoisted-GNN check vs upstream forward: max|diff| = {delta:.3e}")
    if delta > 1e-4:
        raise SystemExit("hoisted path diverges from upstream forward; aborting")

    if not mols:
        print("[mpnn] nothing new to compute")
        return 0
    print(f"[mpnn] scoring {len(mols)} new molecules x {R} receptors on {args.device}")
    with torch.inference_mode():
        for mi, g in enumerate(mols):
            for s in range(0, R, args.batch_size):
                hi = min(s + args.batch_size, R)
                logits[mi, s:hi] = fast_forward(g, s, hi)
            if (mi + 1) % 10 == 0 or mi + 1 == len(mols):
                print(f"[mpnn]   {mi + 1}/{len(mols)}", flush=True)

    if not np.isfinite(logits).all():
        raise SystemExit("non-finite logits")
    scores = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)

    if existing_entries:
        entries = existing_entries + entries
        logits = np.vstack([existing_logits, logits])
        scores = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)

    lib = Library(paths, name=name)
    lib._write(entries, logits, scores, panel_hash(receptors),
               [r.receptor_id for r in receptors], [r.gene for r in receptors])
    print(f"[mpnn] stored {len(entries)} molecules -> {lib.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
