"""
The yardstick: the scale every "how close is close enough" number is read against.

    yardstick = median, over materials, of the distance from a material to its
                nearest neighbour in percept space.

It is a RULER, not a measurement. When the simplifier says a proposal sits 0.015
from the target "against a yardstick of 0.44", it means the proposal is much
closer to its target than two typical neighbours are to each other -- not that
any nose has been consulted. `percept_blind_spots.py` reaches the same way: a
pair "below the yardstick" means "closer than typical neighbours", never
"provably indistinguishable".

It is also MODEL-DEPENDENT, and that is where the documentation went wrong. The
same formula over different percept artifacts:

    percepts_fp3.json          mean prob 0.047   yardstick 0.506
    percepts_fp4.json (427)    mean prob 0.164   yardstick 1.221
    percepts_fp4 + 47 oils     (what the export ships)      1.101

`pos_weight` in fp4 lifts the mean probability about 3x, so every vector sits
further from every other and the ruler stretches with it. ADDING_MATERIALS.md and
DEPLOY_MOLOR.md still quote 0.451, which is an fp3-era figure retired when fp4
became the active model (commit b8d08f9); nothing about the formula changed -- it
has been median-nearest-neighbour since the first commit (d8ed44d).

So: read the live value with `from_payload()` rather than hard-coding a literal.
A hard-coded 0.439 in percept_blind_spots.py survived the model change and made
that script's verdicts silently wrong by a factor of ~2.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = ROOT / "artifacts" / "simplify_data.json"


def matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances with the diagonal set to +inf."""
    X = np.asarray(X, float)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return D


def nn_distances_from(D: np.ndarray) -> np.ndarray:
    """Per-material distance to its nearest neighbour, from a distance matrix."""
    return np.asarray(D, float).min(1)


def nn_distances(X: np.ndarray) -> np.ndarray:
    """Per-material distance to its nearest neighbour."""
    return nn_distances_from(matrix(X))


def of_distances(D: np.ndarray) -> float:
    """The yardstick from a precomputed distance matrix."""
    return float(np.median(nn_distances_from(D)))


def of(X: np.ndarray) -> float:
    """The yardstick for a set of materials."""
    return of_distances(matrix(X))


def from_payload(path: Path | str = DEFAULT_PAYLOAD) -> float:
    """The yardstick the shipped simplifier payload carries.

    Preferred over any literal: the number moves when the material set or the
    percept model does, and the payload is regenerated with it.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return float(doc["yardstick"])
    except (KeyError, TypeError) as exc:  # pragma: no cover
        raise KeyError(
            f"{path} has no 'yardstick' - rerun scripts/export_simplify_data.py"
        ) from exc
