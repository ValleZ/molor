"""Controlled access to the vendored upstream ``microsoft/olfaction`` code.

The upstream repository is a flat script collection, not an installable
package: ``data/m2or.py`` does ``from utils import ROOT_DIR``, and there is no
``data/__init__.py``. Rather than reimplement the model or the embedding
routine -- which would risk silently diverging from the trained weights -- we
put the pinned checkout on ``sys.path`` and import the real modules.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

from .config import UPSTREAM_COMMIT, resolve_paths

_LOCK = threading.Lock()
_CACHE: dict[str, ModuleType] = {}


class UpstreamUnavailable(RuntimeError):
    """The vendored upstream checkout is missing or at the wrong commit."""


def upstream_root() -> Path:
    path = resolve_paths().upstream
    if not (path / "gcn_or_predictor.py").is_file():
        raise UpstreamUnavailable(
            f"Vendored upstream checkout not found at {path}.\n"
            f"Clone it and check out the pinned commit:\n"
            f"  git clone https://github.com/microsoft/olfaction {path}\n"
            f"  git -C {path} checkout {UPSTREAM_COMMIT}"
        )
    return path


def _ensure_on_path() -> Path:
    root = upstream_root()
    entry = str(root)
    if entry not in sys.path:
        # Appended, not prepended: upstream has generic module names such as
        # `utils` and `data`, and we do not want them shadowing anything of
        # ours or the stdlib's.
        sys.path.append(entry)
    return root


def _load_file_module(name: str, relative: str) -> ModuleType:
    """Import a single upstream .py file under an explicit module name."""
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        root = _ensure_on_path()
        path = root / relative
        if not path.is_file():
            raise UpstreamUnavailable(f"Expected upstream file missing: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise UpstreamUnavailable(f"Could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        # Register before exec so intra-module circular imports resolve.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _CACHE[name] = module
        return module


def gcn_or_predictor() -> ModuleType:
    """Upstream ``gcn_or_predictor`` module (defines ``MolORPredictor``)."""
    return _load_file_module("_orvector_upstream_gcn_or_predictor", "gcn_or_predictor.py")


def m2or() -> ModuleType:
    """Upstream ``data/m2or`` module (defines ``esm_embed`` / ``setup_esm``)."""
    return _load_file_module("_orvector_upstream_m2or", "data/m2or.py")


def utils() -> ModuleType:
    """Upstream ``utils`` module (defines ``predict_OR_feat``)."""
    return _load_file_module("_orvector_upstream_utils", "utils.py")


def predict_or_feat():
    """Upstream's own pair-prediction entry point.

    Used by the equivalence test to drive the model exactly as upstream does.
    Note that it calls ``bg.ndata.pop('h')``, which *removes* the feature, so
    each call needs a freshly batched graph -- the published
    ``generate_OR_predictions_pseudogenes.py`` reuses one ``bg`` across its
    receptor loop and therefore raises ``KeyError`` on its second receptor.
    """
    return utils().predict_OR_feat


def molor_predictor_cls():
    return gcn_or_predictor().MolORPredictor


def esm_embed():
    """The upstream per-residue embedding function, unmodified."""
    return m2or().esm_embed


def upstream_commit() -> str:
    """The commit actually checked out, for the manifest."""
    import subprocess

    root = upstream_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def disable_anomaly_detection() -> None:
    """Undo upstream's global ``set_detect_anomaly(True)``.

    ``MolORPredictor.__init__`` switches on autograd anomaly detection as a
    side effect. It is a process-wide flag that only slows things down and has
    no place in an inference path, so we clear it after construction.
    """
    import torch

    torch.autograd.set_detect_anomaly(False)
