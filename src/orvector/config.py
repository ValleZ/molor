"""Frozen configuration for the orvector receptor panel and model.

Every constant here was derived from the pinned upstream repository or read
directly out of the official checkpoint, not guessed. See ``PROVENANCE`` for
how each one was established.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Software identity
# --------------------------------------------------------------------------

ORVECTOR_VERSION = "0.1.0"

#: Commit of https://github.com/microsoft/olfaction this build is pinned to.
UPSTREAM_COMMIT = "0b8bd710509d50310e8c45396301e22eebbc8f69"

#: Author-recommended receptor-inference checkpoint (Zenodo record 13765978).
CHECKPOINT_NAME = "m2or_MolOR_esm_650m_unweighted.pth"
CHECKPOINT_MD5 = "98b3bcc6f0bae4c8faddb40ffb22c7d5"
CHECKPOINT_URL = (
    "https://zenodo.org/api/records/13765978/files/"
    "m2or_MolOR_esm_650m_unweighted.pth/content"
)

#: Pre-generated upstream HORDE logits, used only by the parity test.
REFERENCE_LOGITS_NAME = "olfactory_subgenome_OR_logits.pt"
REFERENCE_LOGITS_MD5 = "d5f9af2640d50a5ad94f58313dfa6f3e"

# --------------------------------------------------------------------------
# Protein encoder
# --------------------------------------------------------------------------

#: Upstream ``setup_esm`` loads ``esm.pretrained.esm2_t33_650M_UR50D``.
ESM_MODEL = "esm2_t33_650M_UR50D"

#: Upstream ``esm_embed`` extracts ``repr_layers=[33]`` for the 650M model.
ESM_LAYER = 33

#: Embedding width of ESM-2 650M.
ESM_DIM = 1280

#: Upstream ``esm_embed`` embeds in ``divide_chunks(sequences, 5)`` groups.
ESM_CHUNK = 5

#: Residue positions every panel sequence is padded to with the literal
#: ``"<pad>"`` token before embedding.
#:
#: Upstream computes ``max_seq_len = len(max(top_seqs, key=len))`` over *all*
#: 845 cleaned HORDE sequences, which is 443 -- not the 357 that the 386
#: functional sequences alone would give.
#:
#: It is frozen because it *defines the cached artefact*, not because it shifts
#: the scores. Upstream's ``esm_embed`` slices ``[1:-1]`` off the whole padded
#: batch rather than each sequence's true length, so a common width is what
#: makes the extraction coherent at all, and an embedding computed at another
#: width is a different array of a different shape. Matching upstream's 443
#: also keeps our embeddings directly comparable to theirs.
#:
#: Measured, not assumed: shortening the padded tail leaves the logits
#: unchanged. The padded residues do enter the molecule-side attention as keys
#: and values (they survive ``prot_norm`` as ``Linear(0) = bias``), but that
#: path runs through ``torch.relu(scores)`` in
#: ``OdorantReceptorCrossAttention``, which is zero for the inputs tested. See
#: ``test_residue_padding_tail_leaks``, which keeps probing this. Contrast
#: :data:`MAX_NODE_LEN`, where the equivalent leak *is* active.
SEQ_PAD_LEN = 443

# --------------------------------------------------------------------------
# Molecule featurisation / model geometry
# --------------------------------------------------------------------------

#: Atom slots the molecule side is padded to, i.e. upstream
#: ``GS_LF_OR.max_node_len`` = max atom count over the 5862 GS-LF molecules.
#:
#: This one genuinely changes the numbers. Padded atom slots are zeroed only
#: after ``mol_norm``, and ``LayerNorm`` maps a zero vector to its own ``bias``,
#: so ``Linear(bias)`` feeds the protein-side attention as keys and values even
#: though ``fixed_size_tensor2[node_mask == 0] = 0`` drops them from the
#: molecule aggregation. Widening the budget shifts the logits; see
#: ``test_atom_slot_budget_changes_scores``.
MAX_NODE_LEN = 98

#: ``MolORPredictor`` keyword arguments. Derived from the checkpoint's own
#: tensor shapes and ``data/configures/M2OR_Pairs/MolOR_canonical.json``.
MODEL_CONFIG: dict[str, object] = {
    # in_feats: CanonicalAtomFeaturizer().feat_size()
    #           == gnn.gnn_layers.0.graph_conv.weight.shape[0] == 74
    "in_feats": 74,
    # two 256-wide GCN layers (gnn_hidden_feats x num_gnn_layers)
    "hidden_feats": [256, 256],
    "residual": [True, True],
    "batchnorm": [False, False],
    "dropout": [0.05, 0.05],
    # prot_feats: cross_attn.query_transform_tensor1 is (1280, 1280)
    "add_feats": ESM_DIM,
    "prot_feats": ESM_DIM,
    # None -> gnn.hidden_feats[-1] == 256; predict.predict.1 is (128, 1536)
    # and 1536 == 1280 + 256, which confirms it.
    "gnn_attended_feats": None,
    # mol2_prot=True is the single most important inference: with it False the
    # Q/K/V shapes would be (256, 1280) and linear1 would be (1, 256). The
    # checkpoint has (1280, 1280) / (1280, 256) and (1, 1280), so the weights
    # were trained with mol2prot_dim=True -- the opposite of the upstream
    # argparse default. Building with the default fails strict loading.
    "mol2_prot": True,
    "model_encoder": "GCN",
    "max_seq_len": SEQ_PAD_LEN,
    "max_node_len": MAX_NODE_LEN,
    "predictor_hidden_feats": 128,
    "predictor_dropout": 0.05,
    # predict.predict.4 is (1, 128): a single responsiveness logit per pair.
    "n_tasks": 1,
}

PROVENANCE = {
    "upstream_repo": "https://github.com/microsoft/olfaction",
    "upstream_commit": UPSTREAM_COMMIT,
    "checkpoint": CHECKPOINT_NAME,
    "checkpoint_md5": CHECKPOINT_MD5,
    "checkpoint_source": "Zenodo record 13765978 (doi:10.1101/2024.09.16.613334)",
    "panel_source": "data/datasets/genes.csv + data/datasets/HORDE/functional_indices.json",
    "esm_model": ESM_MODEL,
    "esm_layer": ESM_LAYER,
    "seq_pad_len": SEQ_PAD_LEN,
    "max_node_len": MAX_NODE_LEN,
}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Resolved on-disk layout."""

    root: Path
    upstream: Path
    weights: Path
    artifacts: Path

    @property
    def genes_csv(self) -> Path:
        return self.upstream / "data" / "datasets" / "genes.csv"

    @property
    def functional_indices(self) -> Path:
        return self.upstream / "data" / "datasets" / "HORDE" / "functional_indices.json"

    @property
    def pseudogene_indices(self) -> Path:
        return self.upstream / "data" / "datasets" / "HORDE" / "pseudogene_indices.json"

    @property
    def checkpoint(self) -> Path:
        return self.weights / CHECKPOINT_NAME

    @property
    def reference_logits(self) -> Path:
        return self.weights / REFERENCE_LOGITS_NAME

    @property
    def gs_lf_csv(self) -> Path:
        return (
            self.upstream
            / "data"
            / "datasets"
            / "NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv"
        )

    @property
    def receptors_tsv(self) -> Path:
        return self.artifacts / "receptors.tsv"

    @property
    def manifest(self) -> Path:
        return self.artifacts / "manifest.json"

    @property
    def embedding_cache(self) -> Path:
        return self.artifacts / "esm_cache"

    @property
    def panel_embeddings(self) -> Path:
        return self.artifacts / "panel_embeddings.npy"


def resolve_paths(root: str | os.PathLike[str] | None = None) -> Paths:
    """Resolve the project layout, honouring ``ORVECTOR_ROOT``."""
    if root is None:
        root = os.environ.get("ORVECTOR_ROOT")
    if root is None:
        # src/orvector/config.py -> project root
        root = Path(__file__).resolve().parents[2]
    root = Path(root).resolve()
    return Paths(
        root=root,
        upstream=root / "vendor" / "olfaction",
        weights=root / "vendor" / "zenodo",
        artifacts=root / "artifacts",
    )


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def encoder_config_hash() -> str:
    """Hash of everything that determines a cached per-residue embedding."""
    payload = {
        "esm_model": ESM_MODEL,
        "layer": ESM_LAYER,
        "dim": ESM_DIM,
        "pad_len": SEQ_PAD_LEN,
        "chunk": ESM_CHUNK,
        "slice": "1:-1",
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sequence_hash(sequence: str) -> str:
    """Stable hash of an unpadded residue string."""
    return hashlib.sha256(sequence.encode()).hexdigest()


def model_config_hash() -> str:
    blob = json.dumps(MODEL_CONFIG, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def software_version() -> dict[str, str]:
    """Versions of everything that can move the numbers."""
    import numpy
    import pandas
    import rdkit
    import torch

    import dgl

    return {
        "orvector": ORVECTOR_VERSION,
        "python": ".".join(str(x) for x in __import__("sys").version_info[:3]),
        "torch": torch.__version__,
        "dgl": dgl.__version__,
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def config_dict() -> dict[str, object]:
    return {
        "provenance": dict(PROVENANCE),
        "model_config": {k: v for k, v in MODEL_CONFIG.items()},
        "model_config_hash": model_config_hash(),
        "encoder_config_hash": encoder_config_hash(),
    }


__all__ = [n for n in dir() if not n.startswith("_")]
