"""SMILES -> per-receptor responsiveness scores.

Differences from ``scripts/generate_OR_predictions_pseudogenes.py``, which is
used as a reference rather than invoked:

* The ``if len(smiles) == 1: continue`` guard is removed. It exists upstream to
  dodge batch-norm instability *during training*; the MLP head's
  ``BatchNorm1d`` uses its stored running statistics under ``eval()``, so a
  single-pair batch is well defined. Keeping the guard would silently emit no
  predictions for a one-molecule request.
* Hard-coded ``.cuda()`` calls are replaced by an explicit device argument.
* Molecules and receptors are paired generically, so one molecule against the
  whole panel and a CSV of molecules take the same path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_NODE_LEN, SEQ_PAD_LEN
from .panel import Receptor


class InvalidSMILESError(ValueError):
    """The input could not be parsed as a molecule by RDKit."""


class MoleculeTooLargeError(ValueError):
    """The molecule has more heavy atoms than the frozen atom-slot budget."""


@dataclass(frozen=True)
class Molecule:
    """A parsed input molecule."""

    input_smiles: str
    canonical_smiles: str
    n_atoms: int
    graph: object


def normalise_hydrogens(mol):
    """Make hydrogen bookkeeping independent of how the SMILES was written.

    Chirality must be written in brackets, as in ``[C@H]``, and RDKit then
    records that atom's hydrogens as *explicit*, which sets its implicit valence
    to zero. ``CanonicalAtomFeaturizer`` includes implicit valence as a feature
    block, so the same molecule featurises differently depending only on its
    notation. Measured on this library, 14 of the 15 materials written with
    bracket atoms were affected, and their receptor vectors were therefore not
    comparable with the other 287.

    Handing the hydrogens back to the implicit count fixes it. The chiral tag
    lives on the atom, not on the hydrogen count, so stereochemistry survives --
    which matters because bond stereo *is* a bond feature, so discarding stereo
    notation wholesale would change far more than this artefact. The InChIKey is
    checked before and after and the original kept on any mismatch, so the
    compound itself can never be altered silently.
    """
    from rdkit import Chem

    if not any(a.GetNumExplicitHs() for a in mol.GetAtoms()):
        return mol
    before = Chem.MolToInchiKey(mol)
    patched = Chem.Mol(mol)
    for a in patched.GetAtoms():
        if a.GetNumExplicitHs():
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(patched)
    except Exception:  # noqa: BLE001 - a failed normalisation must not lose the molecule
        return mol
    return patched if Chem.MolToInchiKey(patched) == before else mol


def mol_to_graph(mol, *, add_self_loop=True, node_featurizer=None,
                 edge_featurizer=None):
    """Featurise a molecule, normalising hydrogen bookkeeping first.

    Equivalent to ``SMILESToBigraph`` with the same arguments -- which is
    ``smiles_to_bigraph``, i.e. ``MolFromSmiles`` then ``mol_to_bigraph`` with
    ``canonical_atom_order=True`` -- except that it takes a molecule, so the
    normalisation above can be applied before the atom features are read.
    """
    from dgllife.utils import mol_to_bigraph

    return mol_to_bigraph(normalise_hydrogens(mol),
                          add_self_loop=add_self_loop,
                          node_featurizer=node_featurizer,
                          edge_featurizer=edge_featurizer,
                          canonical_atom_order=True,
                          explicit_hydrogens=False,
                          num_virtual_nodes=0)


def _featurizer():
    """Upstream's molecular featurisation, as used to train the checkpoint."""
    from dgllife.utils import CanonicalAtomFeaturizer

    def build(smiles: str):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol_to_graph(mol, add_self_loop=True,
                            node_featurizer=CanonicalAtomFeaturizer())

    return build


def canonicalize(smiles: str) -> str:
    """RDKit canonical SMILES, matching upstream's dataset preparation."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidSMILESError("Empty SMILES input.")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise InvalidSMILESError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def parse_molecule(smiles: str, *, max_atoms: int = MAX_NODE_LEN) -> Molecule:
    """Parse and featurise one SMILES string."""
    canonical = canonicalize(smiles)
    graph = _featurizer()(canonical)
    if graph is None:
        raise InvalidSMILESError(
            f"SMILES {smiles!r} parsed but could not be converted to a graph."
        )
    n_atoms = graph.num_nodes()
    if n_atoms == 0:
        raise InvalidSMILESError(f"SMILES {smiles!r} produced a graph with no atoms.")
    if n_atoms > max_atoms:
        raise MoleculeTooLargeError(
            f"SMILES {smiles!r} has {n_atoms} heavy atoms but the model's atom-slot "
            f"budget (max_node_len) is {max_atoms}. That budget is part of the "
            f"frozen configuration the checkpoint was trained with, so the molecule "
            f"cannot be scored without changing the model's numerics."
        )
    return Molecule(
        input_smiles=smiles.strip(),
        canonical_smiles=canonical,
        n_atoms=n_atoms,
        graph=graph,
    )


def _node_mask(n_atoms: int, max_atoms: int = MAX_NODE_LEN) -> np.ndarray:
    mask = np.zeros(max_atoms, dtype=np.float32)
    mask[:n_atoms] = 1.0
    return mask


def predict_matrix(
    model,
    molecules: list[Molecule],
    receptors: list[Receptor],
    embeddings,
    residue_masks: np.ndarray,
    *,
    device: str = "cpu",
    pair_batch_size: int = 16,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every (molecule, receptor) pair.

    Returns ``(logits, scores)``, each ``(n_molecules, n_receptors)`` float32,
    where ``scores = sigmoid(logits)``.
    """
    import dgl
    import torch

    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be >= 1")
    if len(molecules) == 0:
        raise ValueError("No molecules to score.")
    if len(receptors) == 0:
        raise ValueError("Receptor panel is empty.")
    if residue_masks.shape != (len(receptors), SEQ_PAD_LEN):
        raise ValueError(
            f"residue_masks has shape {residue_masks.shape}; expected "
            f"{(len(receptors), SEQ_PAD_LEN)}"
        )

    torch_device = torch.device(device)
    n_mol, n_rec = len(molecules), len(receptors)
    logits = np.full((n_mol, n_rec), np.nan, dtype=np.float32)

    node_masks = {i: _node_mask(m.n_atoms) for i, m in enumerate(molecules)}
    pairs = [(mi, ri) for mi in range(n_mol) for ri in range(n_rec)]
    total = len(pairs)
    done = 0

    with torch.inference_mode():
        for start in range(0, total, pair_batch_size):
            chunk = pairs[start : start + pair_batch_size]

            graphs = [molecules[mi].graph for mi, _ in chunk]
            bg = dgl.batch(graphs)
            bg.set_n_initializer(dgl.init.zero_initializer)
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg = bg.to(torch_device)
            node_feats = bg.ndata.pop("h").to(torch_device)

            add_feats = torch.from_numpy(
                np.stack([np.asarray(embeddings[ri], dtype=np.float32) for _, ri in chunk])
            ).to(torch_device)
            seq_mask = torch.from_numpy(
                np.stack([residue_masks[ri] for _, ri in chunk])
            ).to(torch_device)
            node_mask = torch.from_numpy(
                np.stack([node_masks[mi] for mi, _ in chunk])
            ).to(torch_device)

            out = model(bg, node_feats, add_feats, seq_mask, node_mask, torch_device)
            out = out.reshape(len(chunk)).to(torch.float32).cpu().numpy()

            for (mi, ri), value in zip(chunk, out):
                logits[mi, ri] = value

            done += len(chunk)
            if progress:
                print(f"[predict]   {done}/{total} pairs", flush=True)

    if not np.isfinite(logits).all():
        bad = int((~np.isfinite(logits)).sum())
        raise RuntimeError(f"{bad} of {logits.size} predictions were not finite.")

    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return logits, scores.astype(np.float32)
