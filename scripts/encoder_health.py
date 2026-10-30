"""Check whether a MolOR molecular encoder is numerically healthy.

Written after discovering that the MPNN checkpoint's encoder is degenerate. Its
NNConv layer emits values around 5e4, which pins the following GRU: 49.6% of the
256 output dimensions sit at exactly +-1 after a *single* message-passing step,
and the whole representation takes only ~82 distinct values. Message passing
therefore contributes nothing but saturation, and the encoder collapses to
roughly a per-atom linear map of the atom features.

That is not repairable from the weights. Rescaling the inputs to unsaturate the
GRU would produce a function the checkpoint was never trained for -- healthier
numbers that mean less. Upstream appears to have known: gcn_or_predictor.py
carries the comment "problem causing NaNs is here" on the line that calls the
MPNN encoder.

Three measurements decide health:
  saturation  -- fraction of outputs pinned at an activation bound
  diversity   -- distinct values in the representation
  sensitivity -- whether perturbing the input changes the output at all

and two probe pairs show what the degeneracy costs:
  geraniol / nerol               differ only in bond stereo (E/Z)
  gamma- / delta-decalactone     differ only in ring size

The second pair is the important one. The GCN encoder separates it cleanly,
which contradicts this project's earlier explanation of the model's isomer
blindness as a 1-Weisfeiler-Leman readout limit. Saturation is the simpler
cause, and it is specific to the MPNN checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PROBES = {
    "geraniol": "CC(C)=CCC/C(C)=C/CO",
    "nerol": "CC(C)=CCC/C(C)=C\\CO",
    "vanillin": "O=Cc1ccc(O)c(OC)c1",
    "gamma-decalactone": "CCCCCCC1CCC(=O)O1",
    "delta-decalactone": "CCCCCC1CCCC(=O)O1",
}
PAIRS = [("geraniol", "nerol", "bond stereo (E/Z)"),
         ("gamma-decalactone", "delta-decalactone", "ring size")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("gcn", "mpnn"), default="gcn")
    ap.add_argument("--steps", type=int, default=6,
                    help="MPNN only; saturation is complete at 1 so this "
                         "changes nothing, which is itself the finding")
    args = ap.parse_args()

    from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
    from rdkit import Chem, RDLogger

    from orvector.config import resolve_paths
    from orvector.inference import mol_to_graph

    RDLogger.DisableLog("rdApp.*")
    paths = resolve_paths()
    nf = CanonicalAtomFeaturizer()
    use_edges = args.encoder == "mpnn"
    efz = CanonicalBondFeaturizer(self_loop=True) if use_edges else None

    if use_edges:
        from run_mpnn import build
        sd = torch.load(paths.weights / "m2or_MolOR_MPNN_esm_650m_unweighted.pth",
                        map_location="cpu", weights_only=False)["model_state_dict"]
        model, _ = build(sd, args.steps, "cpu")
    else:
        from orvector.model import build_model
        model = build_model(paths=paths, device="cpu")
    model.eval()
    enc = model.gnn
    print(f"[health] {args.encoder.upper()} encoder: {type(enc).__name__}"
          + (f", {args.steps} message-passing steps" if use_edges else ""))

    def graph(smi):
        return mol_to_graph(Chem.MolFromSmiles(smi), add_self_loop=True,
                            node_featurizer=nf, edge_featurizer=efz)

    def run(g):
        with torch.no_grad():
            return (enc(g, g.ndata["h"], g.edata["e"]) if use_edges
                    else enc(g, g.ndata["h"])).numpy()

    outs = {}
    print("\n  molecule             shape      min        max       "
          "at +-1   distinct")
    for name, smi in PROBES.items():
        v = run(graph(smi))
        outs[name] = v
        pinned = float((np.abs(np.abs(v) - 1) < 1e-6).mean())
        print(f"  {name:19s} {str(v.shape):10s} {v.min():+9.3f} {v.max():+10.3f} "
              f"{pinned:8.3f} {len(np.unique(np.round(v, 6))):9d}")

    print("\n  probe pairs (the model's readout is permutation-invariant, so the"
          "\n  summed column is the one that reaches the logits)")
    for a, b, why in PAIRS:
        x, y = outs[a], outs[b]
        m = min(len(x), len(y))
        print(f"  {a} vs {b}  --  {why}")
        print(f"     rows identical {np.array_equal(x[:m], y[:m])}, "
              f"multiset identical "
              f"{np.array_equal(np.sort(x[:m], 0), np.sort(y[:m], 0))}, "
              f"summed max|delta| {np.abs(x.sum(0) - y.sum(0)).max():.4e}")

    g = graph(PROBES["geraniol"])
    base = run(g)
    h2 = g.ndata["h"].clone()
    h2[0] = 0.0
    with torch.no_grad():
        pert = (enc(g, h2, g.edata["e"]) if use_edges else enc(g, h2)).numpy()
    print(f"\n  sensitivity control, zeroing atom 0's features: "
          f"max|delta| {np.abs(base - pert).max():.4e}")
    if use_edges:
        e2 = torch.zeros_like(g.edata["e"])
        with torch.no_grad():
            ze = enc(g, g.ndata["h"], e2).numpy()
        print(f"  zeroing every edge feature:                     "
              f"max|delta| {np.abs(base - ze).max():.4e}"
              f"   <- 0 means bond features never reach the output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
