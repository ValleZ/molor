import json

import numpy as np
import pytest

from orvector.config import resolve_paths
from orvector.embeddings import load_panel_embeddings, panel_residue_masks
from orvector.model import build_model
from orvector.panel import read_panel

METHYL_BENZOATE = "COC(=O)c1ccccc1"


@pytest.fixture(scope="session")
def paths():
    return resolve_paths()


@pytest.fixture(scope="session")
def prepared(paths):
    """Skip the suite cleanly if `prepare` has not been run."""
    missing = [
        p.name
        for p in (paths.receptors_tsv, paths.panel_embeddings, paths.manifest)
        if not p.is_file()
    ]
    if missing:
        pytest.skip(f"run `python -m orvector prepare` first; missing {missing}")
    return paths


@pytest.fixture(scope="session")
def receptors(prepared):
    return read_panel(prepared.receptors_tsv)


@pytest.fixture(scope="session")
def manifest(prepared):
    return json.loads(prepared.manifest.read_text())


@pytest.fixture(scope="session")
def embeddings(prepared, receptors):
    return load_panel_embeddings(prepared, n_receptors=len(receptors))


@pytest.fixture(scope="session")
def masks(receptors):
    return panel_residue_masks(receptors)


@pytest.fixture(scope="session")
def model(prepared):
    return build_model(device="cpu", paths=prepared)


@pytest.fixture(scope="session")
def gs_lf_molecules(prepared):
    """GS-LF molecules in the row order that indexes the upstream logits."""
    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not prepared.gs_lf_csv.is_file():
        pytest.skip(f"missing {prepared.gs_lf_csv}")
    df = pd.read_csv(prepared.gs_lf_csv)
    return [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df["IsomericSMILES"]]
