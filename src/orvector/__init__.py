"""orvector: SMILES -> human olfactory receptor responsiveness vectors.

Built on the pinned ``microsoft/olfaction`` MolOR cross-attention model and the
official ``m2or_MolOR_esm_650m_unweighted`` checkpoint.
"""

from .config import ORVECTOR_VERSION as __version__

__all__ = ["__version__"]
