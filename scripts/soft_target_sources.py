"""Load the four extra odour-label sources (+ Good Scents order) keyed by flat InChIKey.

Run under .venv-pyrfume. Returns, per source, {flat_key: {words}} and, for
Good Scents and IFRA, {flat_key: {first-listed word}}. Words are lower-cased,
stripped; multi-word descriptors are kept whole.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_similarity import flat_key  # noqa: E402


def _clean(w) -> str | None:
    if not isinstance(w, str):
        return None
    w = w.strip().lower()
    return w or None


def _cid_keys(mol: pd.DataFrame) -> dict:
    out = {}
    for cid, smi in zip(mol.index, mol["IsomericSMILES"]):
        k = flat_key(smi) if isinstance(smi, str) else None
        if k:
            out[int(cid)] = k
    return out


def load_all() -> dict:
    import pyrfume

    res = {}

    # arctander: binary matrix; Stimulus -> new_CID
    mol = pyrfume.load_data("arctander_1960/molecules.csv")
    sti = pyrfume.load_data("arctander_1960/stimuli.csv")
    beh = pyrfume.load_data("arctander_1960/behavior_1.csv")
    ck = _cid_keys(mol)
    words, n_stim = defaultdict(set), 0
    for stim, row in beh.iterrows():
        cid = sti.loc[stim, "new_CID"]
        if cid != cid or int(cid) not in ck:
            continue
        n_stim += 1
        for w in beh.columns[(row.to_numpy() > 0)]:
            words[ck[int(cid)]].add(_clean(w))
    res["arctander"] = {"words": dict(words), "first": {}, "n_mol": len(mol),
                        "n_stim": len(sti)}

    # ifra: three descriptor columns, repeated rows per stimulus (= CID)
    mol = pyrfume.load_data("ifra_2019/molecules.csv")
    beh = pyrfume.load_data("ifra_2019/behavior.csv")
    ck = _cid_keys(mol)
    words, first = defaultdict(set), defaultdict(set)
    for cid, r in beh.iterrows():
        if int(cid) not in ck:
            continue
        k = ck[int(cid)]
        for j, c in enumerate(beh.columns):
            w = _clean(r[c])
            if w:
                words[k].add(w)
                if j == 0:
                    first[k].add(w)
    res["ifra"] = {"words": dict(words), "first": dict(first), "n_mol": len(mol),
                   "n_stim": beh.index.nunique()}

    # sigma: boolean matrix; Stimulus > 0 is a CID, negatives are CAS-only mixtures
    mol = pyrfume.load_data("sigma_2014/molecules.csv")
    beh = pyrfume.load_data("sigma_2014/behavior.csv")
    ck = _cid_keys(mol)
    words = defaultdict(set)
    for stim, r in beh.iterrows():
        if int(stim) not in ck:
            continue
        for w in ast.literal_eval(r["descriptors"]) if isinstance(r["descriptors"], str) else []:
            if _clean(w):
                words[ck[int(stim)]].add(_clean(w))
    res["sigma"] = {"words": dict(words), "first": {}, "n_mol": len(mol),
                    "n_stim": len(beh)}

    # flavornet: ';'-separated
    mol = pyrfume.load_data("flavornet/molecules.csv")
    beh = pyrfume.load_data("flavornet/behavior.csv")
    ck = _cid_keys(mol)
    words = defaultdict(set)
    for cid, r in beh.iterrows():
        if int(cid) not in ck:
            continue
        for w in str(r["Descriptors"]).split(";"):
            if _clean(w):
                words[ck[int(cid)]].add(_clean(w))
    res["flavornet"] = {"words": dict(words), "first": {}, "n_mol": len(mol),
                        "n_stim": len(beh)}

    # goodscents: ordered, ';'-separated; Stimulus -> CID
    mol = pyrfume.load_data("goodscents/molecules.csv")
    sti = pyrfume.load_data("goodscents/stimuli.csv")
    beh = pyrfume.load_data("goodscents/behavior.csv")
    ck = _cid_keys(mol)
    words, first = defaultdict(set), defaultdict(set)
    for stim, r in beh.iterrows():
        cid = sti.loc[stim, "CID"]
        if isinstance(cid, pd.Series):
            cid = cid.iloc[0]
        if cid != cid or int(cid) not in ck:
            continue
        k = ck[int(cid)]
        for j, w in enumerate(str(r["Descriptors"]).split(";")):
            w = _clean(w)
            if w:
                words[k].add(w)
                if j == 0:
                    first[k].add(w)
    res["goodscents"] = {"words": dict(words), "first": dict(first),
                         "n_mol": len(mol), "n_stim": len(beh)}
    return res
