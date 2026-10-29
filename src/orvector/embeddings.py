"""ESM-2 650M per-residue embedding, cached by sequence and encoder config.

Faithfulness
------------
The embeddings are produced by upstream's own ``data/m2or.esm_embed`` with
``per_residue=True``, so the extraction layer (33), the special-token handling
and the ``[1:-1]`` slice are upstream's, not a reimplementation.

That slice is the subtle part. Upstream takes
``token_representations[i, 1:-1]`` -- dropping BOS and the *batch's* final
token -- rather than slicing to each sequence's true length. That is only
coherent because every sequence is first padded with the literal ``"<pad>"``
token string up to a common length, so each one yields exactly ``SEQ_PAD_LEN``
positions. We reproduce that padding, which is also why the cache key includes
the pad length: an embedding computed at a different pad length is a different
artefact and must not be reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    ESM_CHUNK,
    ESM_DIM,
    ESM_LAYER,
    ESM_MODEL,
    SEQ_PAD_LEN,
    Paths,
    encoder_config_hash,
    resolve_paths,
    sequence_hash,
)
from .panel import Receptor
from .upstream import esm_embed as upstream_esm_embed

PAD_TOKEN = "<pad>"


def pad_sequence(sequence: str, pad_len: int = SEQ_PAD_LEN) -> str:
    """Pad with the literal ``"<pad>"`` token string, as upstream does."""
    if len(sequence) > pad_len:
        raise ValueError(
            f"Sequence of length {len(sequence)} exceeds pad length {pad_len}."
        )
    return sequence + PAD_TOKEN * (pad_len - len(sequence))


def residue_mask(seq_len: int, pad_len: int = SEQ_PAD_LEN) -> np.ndarray:
    """1.0 for real residues, 0.0 for padding -- upstream's ``seq_mask``."""
    mask = np.zeros(pad_len, dtype=np.float32)
    mask[:seq_len] = 1.0
    return mask


def panel_residue_masks(receptors: list[Receptor], pad_len: int = SEQ_PAD_LEN) -> np.ndarray:
    return np.stack([residue_mask(r.seq_len, pad_len) for r in receptors])


def cache_dir(paths: Paths | None = None) -> Path:
    paths = paths or resolve_paths()
    return paths.embedding_cache / encoder_config_hash()


def cache_path_for(sequence: str, paths: Paths | None = None) -> Path:
    return cache_dir(paths) / f"{sequence_hash(sequence)}.npy"


def _resolve_device(device: str):
    import torch

    if device in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _place_encoder(torch_device) -> None:
    """Pin upstream's cached ESM model to ``torch_device``.

    Upstream's ``esm_embed(device=...)`` moves the *tokens* to ``device`` but
    calls ``setup_esm()`` without forwarding it, so the model lands on
    ``setup_esm``'s own default -- ``cuda:0`` whenever CUDA is visible. Asking
    for CPU embeddings on a CUDA machine therefore fails with a device
    mismatch, and the module-level ``esm_model`` global means the first call
    wins for the rest of the process. We call ``setup_esm`` with the device we
    want and then move the cached global explicitly, so repeated calls on
    different devices behave.
    """
    from .upstream import m2or as _m2or

    module = _m2or()
    module.setup_esm(device=torch_device, random_weights=False, esm_model_version="650m")
    if module.esm_model is not None:
        module.esm_model.to(torch_device)
        module.esm_model.eval()


def embed_sequences(
    sequences: list[str],
    *,
    device: str = "auto",
    batch_size: int = 8,
    pad_len: int = SEQ_PAD_LEN,
) -> np.ndarray:
    """Embed unpadded sequences, returning ``(n, pad_len, ESM_DIM)`` float32.

    ``batch_size`` controls how many sequences are handed to upstream per
    call; upstream sub-chunks internally at :data:`ESM_CHUNK`. Because every
    sequence is padded to the same width and ESM-2 masks pad tokens out of
    attention, the result does not depend on how the work is grouped -- which
    the test suite asserts rather than assumes.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    torch_device = _resolve_device(device)
    _place_encoder(torch_device)
    embed = upstream_esm_embed()

    out = np.empty((len(sequences), pad_len, ESM_DIM), dtype=np.float32)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        padded = [pad_sequence(s, pad_len) for s in batch]
        with torch.inference_mode():
            reps = embed(
                padded,
                device=torch_device,
                per_residue=True,
                random_weights=False,
                esm_model_version="650m",
            )
        for offset, rep in enumerate(reps):
            arr = np.asarray(rep, dtype=np.float32)
            if arr.shape != (pad_len, ESM_DIM):
                raise RuntimeError(
                    f"Unexpected embedding shape {arr.shape}; expected "
                    f"{(pad_len, ESM_DIM)}. Upstream's [1:-1] slice assumes "
                    f"every sequence is padded to {pad_len} tokens."
                )
            out[start + offset] = arr
    return out


def build_cache(
    receptors: list[Receptor],
    *,
    paths: Paths | None = None,
    device: str = "auto",
    batch_size: int = 8,
    force: bool = False,
) -> dict[str, int]:
    """Populate the per-sequence embedding cache and the packed panel array."""
    paths = paths or resolve_paths()
    cdir = cache_dir(paths)
    cdir.mkdir(parents=True, exist_ok=True)

    missing = [r for r in receptors if force or not cache_path_for(r.sequence, paths).is_file()]
    stats = {"total": len(receptors), "computed": 0, "reused": len(receptors) - len(missing)}

    if missing:
        print(
            f"[embed] computing {len(missing)} embedding(s) on {device} "
            f"(reusing {stats['reused']}), pad_len={SEQ_PAD_LEN}"
        )
        for start in range(0, len(missing), batch_size):
            group = missing[start : start + batch_size]
            arrs = embed_sequences(
                [r.sequence for r in group], device=device, batch_size=batch_size
            )
            for r, arr in zip(group, arrs):
                final = cache_path_for(r.sequence, paths)
                tmp = final.with_name(final.name + ".tmp")
                # Write through a file handle: np.save() would otherwise append
                # a second ".npy" to a path that does not already end in it.
                with open(tmp, "wb") as fh:
                    np.save(fh, arr)
                tmp.replace(final)
                stats["computed"] += 1
            print(
                f"[embed]   {min(start + len(group), len(missing))}/{len(missing)}",
                flush=True,
            )
    else:
        print(f"[embed] all {len(receptors)} embeddings already cached")

    _pack_panel(receptors, paths)
    (cdir / "encoder.json").write_text(
        json.dumps(
            {
                "esm_model": ESM_MODEL,
                "layer": ESM_LAYER,
                "dim": ESM_DIM,
                "pad_len": SEQ_PAD_LEN,
                "upstream_chunk": ESM_CHUNK,
                "slice": "1:-1",
                "config_hash": encoder_config_hash(),
            },
            indent=2,
        )
    )
    return stats


def _pack_panel(receptors: list[Receptor], paths: Paths) -> None:
    """Assemble the per-sequence cache into one memmappable array.

    Prediction streams receptor slices out of this file, so it is written
    incrementally and never held in memory in full (386 x 443 x 1280 float32
    is ~0.9 GB).
    """
    target = paths.panel_embeddings
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".npy.tmp")
    packed = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(receptors), SEQ_PAD_LEN, ESM_DIM)
    )
    try:
        for r in receptors:
            path = cache_path_for(r.sequence, paths)
            if not path.is_file():
                raise FileNotFoundError(f"Embedding cache miss for {r.gene} at {path}")
            packed[r.panel_index] = np.load(path, mmap_mode="r")
        packed.flush()
    finally:
        del packed
    tmp.replace(target)
    print(f"[embed] packed panel embeddings -> {target}")


def load_panel_embeddings(paths: Paths | None = None, n_receptors: int | None = None):
    """Memmap the packed panel embeddings (read-only)."""
    paths = paths or resolve_paths()
    if not paths.panel_embeddings.is_file():
        raise FileNotFoundError(
            f"Packed panel embeddings not found at {paths.panel_embeddings}. "
            "Run `python -m orvector prepare` first."
        )
    arr = np.load(paths.panel_embeddings, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[1:] != (SEQ_PAD_LEN, ESM_DIM):
        raise RuntimeError(
            f"Packed embeddings have shape {arr.shape}; expected "
            f"(n, {SEQ_PAD_LEN}, {ESM_DIM})."
        )
    if n_receptors is not None and arr.shape[0] != n_receptors:
        raise RuntimeError(
            f"Packed embeddings hold {arr.shape[0]} receptors but the panel has "
            f"{n_receptors}. Re-run `prepare`."
        )
    return arr
