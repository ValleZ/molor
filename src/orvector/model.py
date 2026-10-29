"""Load the official MolOR receptor-inference checkpoint with strict matching."""

from __future__ import annotations

from pathlib import Path

from .config import (
    CHECKPOINT_MD5,
    CHECKPOINT_URL,
    ESM_DIM,
    MODEL_CONFIG,
    Paths,
    file_md5,
    resolve_paths,
)
from .upstream import disable_anomaly_detection, molor_predictor_cls


class CheckpointError(RuntimeError):
    """The checkpoint is missing, corrupt, or does not match the architecture."""


def checkpoint_path(paths: Paths | None = None) -> Path:
    paths = paths or resolve_paths()
    path = paths.checkpoint
    if not path.is_file():
        raise CheckpointError(
            f"Checkpoint not found at {path}.\nDownload it from Zenodo:\n"
            f"  curl -L -o {path} '{CHECKPOINT_URL}'\n"
            f"Expected md5: {CHECKPOINT_MD5}"
        )
    return path


def load_state_dict(paths: Paths | None = None, *, verify_md5: bool = True) -> dict:
    """Read the checkpoint and return its model state dict."""
    import torch

    path = checkpoint_path(paths)
    if verify_md5:
        actual = file_md5(path)
        if actual != CHECKPOINT_MD5:
            raise CheckpointError(
                f"Checkpoint md5 mismatch for {path}:\n"
                f"  expected {CHECKPOINT_MD5}\n  actual   {actual}\n"
                "Re-download the file; a different checkpoint will not match "
                "the frozen model configuration."
            )

    # weights_only=False is required: the file is a dict holding the state dict
    # plus a 'timestep' entry. The checksum above is what establishes trust.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    if isinstance(blob, dict):
        return blob
    raise CheckpointError(f"Unrecognised checkpoint structure in {path}: {type(blob)}")


def derive_config(state_dict: dict) -> dict:
    """Read the architecture back out of the weights.

    This exists so the frozen :data:`MODEL_CONFIG` is *checked* against the
    checkpoint rather than merely asserted. The decisive one is ``mol2_prot``:
    with ``mol2prot_dim=True`` the cross-attention projections are
    ``Linear(1280, 1280)`` and ``Linear(256, 1280)`` with ``linear1`` of
    ``(1, 1280)``; with it False they would be ``Linear(1280, 256)`` and
    ``(1, 256)``. The upstream argparse defaults it to False, but these
    weights were trained with it True.
    """
    try:
        in_feats = state_dict["gnn.gnn_layers.0.graph_conv.weight"].shape[0]
        layer_widths = []
        i = 0
        while f"gnn.gnn_layers.{i}.graph_conv.weight" in state_dict:
            layer_widths.append(int(state_dict[f"gnn.gnn_layers.{i}.graph_conv.weight"].shape[1]))
            i += 1
        prot_feats = int(state_dict["cross_attn.query_transform_tensor1.weight"].shape[1])
        q1_out = int(state_dict["cross_attn.query_transform_tensor1.weight"].shape[0])
        mol_in = int(state_dict["cross_attn.query_transform_tensor2.weight"].shape[1])
        agg_in = int(state_dict["cross_attn.linear1.weight"].shape[1])
        mlp_in = int(state_dict["predict.predict.1.weight"].shape[1])
        predictor_hidden = int(state_dict["predict.predict.1.weight"].shape[0])
        n_tasks = int(state_dict["predict.predict.4.weight"].shape[0])
    except KeyError as exc:
        raise CheckpointError(f"Checkpoint is missing expected tensor {exc}.") from exc

    mol2_prot = q1_out == prot_feats and agg_in == prot_feats
    residual = "gnn.gnn_layers.0.res_connection.weight" in state_dict
    batchnorm = "gnn.gnn_layers.0.bn_layer.weight" in state_dict

    return {
        "in_feats": int(in_feats),
        "hidden_feats": layer_widths,
        "residual": [residual] * len(layer_widths),
        "batchnorm": [batchnorm] * len(layer_widths),
        "prot_feats": prot_feats,
        "add_feats": prot_feats,
        "mol2_prot": mol2_prot,
        "gnn_attended_feats": mlp_in - prot_feats,
        "predictor_hidden_feats": predictor_hidden,
        "n_tasks": n_tasks,
        "_mol_feats": mol_in,
        "_mlp_in": mlp_in,
    }


def _check_consistency(derived: dict) -> None:
    cfg = MODEL_CONFIG
    problems: list[str] = []

    def cmp(name, got, want):
        if got != want:
            problems.append(f"  {name}: checkpoint implies {got!r}, config says {want!r}")

    cmp("in_feats", derived["in_feats"], cfg["in_feats"])
    cmp("hidden_feats", derived["hidden_feats"], cfg["hidden_feats"])
    cmp("residual", derived["residual"], cfg["residual"])
    cmp("batchnorm", derived["batchnorm"], cfg["batchnorm"])
    cmp("prot_feats", derived["prot_feats"], cfg["prot_feats"])
    cmp("mol2_prot", derived["mol2_prot"], cfg["mol2_prot"])
    cmp("predictor_hidden_feats", derived["predictor_hidden_feats"], cfg["predictor_hidden_feats"])
    cmp("n_tasks", derived["n_tasks"], cfg["n_tasks"])

    if derived["prot_feats"] != ESM_DIM:
        problems.append(
            f"  prot_feats {derived['prot_feats']} != ESM-2 650M width {ESM_DIM}"
        )
    expected_attended = cfg["gnn_attended_feats"] or cfg["hidden_feats"][-1]
    cmp("gnn_attended_feats", derived["gnn_attended_feats"], expected_attended)
    if derived["_mol_feats"] != cfg["hidden_feats"][-1]:
        problems.append(
            f"  molecule-side width {derived['_mol_feats']} != last GCN width "
            f"{cfg['hidden_feats'][-1]}"
        )

    if problems:
        raise CheckpointError(
            "Checkpoint architecture does not match the frozen model "
            "configuration:\n" + "\n".join(problems)
        )


def build_model(
    *,
    device: str = "cpu",
    paths: Paths | None = None,
    verify_md5: bool = True,
    state_dict: dict | None = None,
):
    """Build ``MolORPredictor`` and strict-load the official weights.

    Returns the model in ``eval()`` mode on ``device``. Loading is strict: any
    missing or unexpected parameter raises instead of being ignored.
    """
    import torch

    if state_dict is None:
        state_dict = load_state_dict(paths, verify_md5=verify_md5)

    derived = derive_config(state_dict)
    _check_consistency(derived)

    kwargs = {k: v for k, v in MODEL_CONFIG.items()}
    import torch.nn.functional as F

    kwargs["activation"] = [F.relu] * len(kwargs["hidden_feats"])

    model = molor_predictor_cls()(**kwargs)
    # MolORPredictor.__init__ flips on global autograd anomaly detection.
    disable_anomaly_detection()

    incompatible = model.load_state_dict(state_dict, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise CheckpointError(
            f"Strict load reported missing={missing} unexpected={unexpected}"
        )

    model.eval()
    target = torch.device(device)
    model.to(target)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_summary(state_dict: dict) -> dict:
    derived = derive_config(state_dict)
    return {
        "n_tensors": len(state_dict),
        "n_parameters": int(sum(v.numel() for v in state_dict.values())),
        "derived": {k: v for k, v in derived.items() if not k.startswith("_")},
    }
