"""Acceptance tests for the orvector CLI.

Covers the six acceptance criteria, plus three guards on assumptions that the
pipeline's reproducibility depends on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from orvector.config import MAX_NODE_LEN, SEQ_PAD_LEN
from orvector.inference import (
    InvalidSMILESError,
    MoleculeTooLargeError,
    canonicalize,
    parse_molecule,
    predict_matrix,
)
from orvector.panel import build_panel, panel_hash, upstream_clean

from .conftest import METHYL_BENZOATE


# ---------------------------------------------------------------------------
# 1. Single-molecule inference produces a finite score for every receptor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def methyl_benzoate_result(model, receptors, embeddings, masks):
    mol = parse_molecule(METHYL_BENZOATE)
    logits, scores = predict_matrix(
        model, [mol], receptors, embeddings, masks, device="cpu", pair_batch_size=16
    )
    return mol, logits, scores


def test_finite_score_for_every_receptor(methyl_benzoate_result, receptors):
    _, logits, scores = methyl_benzoate_result
    assert logits.shape == (1, len(receptors))
    assert scores.shape == (1, len(receptors))
    assert np.isfinite(logits).all(), "non-finite logits"
    assert np.isfinite(scores).all(), "non-finite scores"
    # Scores are sigmoid outputs: continuous, strictly inside (0, 1).
    assert (scores > 0).all() and (scores < 1).all()
    # Not thresholded or normalised.
    assert not np.allclose(scores, np.round(scores)), "scores look binarised"
    assert abs(scores.sum() - 1.0) > 1e-3, "scores look normalised to a simplex"


def test_scores_are_sigmoid_of_logits(methyl_benzoate_result):
    _, logits, scores = methyl_benzoate_result
    expected = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Single and batched inference agree within numerical tolerance
# ---------------------------------------------------------------------------


def test_single_and_batched_agree(model, receptors, embeddings, masks):
    """Pair-batch size must not change the numbers.

    This is the test that justifies deleting upstream's
    ``if len(smiles) == 1: continue`` guard: a batch of one is as valid as a
    batch of many once the model is in eval mode.
    """
    mol = parse_molecule(METHYL_BENZOATE)
    subset = receptors[:24]
    sub_emb = np.asarray(embeddings[:24])
    sub_masks = masks[:24]

    one, _ = predict_matrix(
        model, [mol], subset, sub_emb, sub_masks, device="cpu", pair_batch_size=1
    )
    many, _ = predict_matrix(
        model, [mol], subset, sub_emb, sub_masks, device="cpu", pair_batch_size=24
    )
    np.testing.assert_allclose(one, many, rtol=0, atol=2e-4)


def test_multi_molecule_matches_individual(model, receptors, embeddings, masks):
    smiles = [METHYL_BENZOATE, "CCO", "CC(C)=CCCC(C)=CCO"]
    mols = [parse_molecule(s) for s in smiles]
    subset = receptors[:12]
    sub_emb = np.asarray(embeddings[:12])
    sub_masks = masks[:12]

    together, _ = predict_matrix(
        model, mols, subset, sub_emb, sub_masks, device="cpu", pair_batch_size=7
    )
    for i, m in enumerate(mols):
        alone, _ = predict_matrix(
            model, [m], subset, sub_emb, sub_masks, device="cpu", pair_batch_size=5
        )
        np.testing.assert_allclose(
            together[i : i + 1], alone, rtol=0, atol=2e-4,
            err_msg=f"molecule {smiles[i]} differs in a mixed batch",
        )


# ---------------------------------------------------------------------------
# 3. Repeated runs preserve receptor ordering and scores
# ---------------------------------------------------------------------------


def test_repeated_runs_are_identical(model, receptors, embeddings, masks):
    mol = parse_molecule(METHYL_BENZOATE)
    subset = receptors[:16]
    sub_emb = np.asarray(embeddings[:16])
    sub_masks = masks[:16]
    first, fs = predict_matrix(
        model, [mol], subset, sub_emb, sub_masks, device="cpu", pair_batch_size=8
    )
    second, ss = predict_matrix(
        model, [mol], subset, sub_emb, sub_masks, device="cpu", pair_batch_size=8
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(fs, ss)


def test_panel_order_and_hash_are_stable(prepared, receptors):
    rebuilt = build_panel(prepared)
    assert [r.receptor_id for r in rebuilt] == [r.receptor_id for r in receptors]
    assert [r.sequence for r in rebuilt] == [r.sequence for r in receptors]
    assert panel_hash(rebuilt) == panel_hash(receptors)


def test_panel_hash_matches_manifest(manifest, receptors):
    assert manifest["panel_hash"] == panel_hash(receptors)
    assert manifest["n_receptors"] == len(receptors)


def test_panel_is_intact_protein_coding(receptors):
    assert len(receptors) >= 380
    for r in receptors:
        assert r.seq_len >= 250, f"{r.gene} looks like a fragment ({r.seq_len} aa)"
        assert r.seq_len <= SEQ_PAD_LEN
        assert set(r.sequence) <= set("ACDEFGHIKLMNPQRSTVWYX"), r.gene
    genes = [r.gene for r in receptors]
    assert len(set(genes)) == len(genes), "duplicate gene symbols in panel"


# ---------------------------------------------------------------------------
# 4. Invalid SMILES fail clearly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not_a_smiles", "C(((", "C1CC", "[Xx]", "COC(=O)c1ccccc1)"],
)
def test_invalid_smiles_raise(bad):
    with pytest.raises(InvalidSMILESError) as exc:
        parse_molecule(bad)
    assert str(exc.value), "error message must not be empty"


def test_invalid_smiles_message_names_the_input():
    with pytest.raises(InvalidSMILESError, match="not_a_smiles"):
        parse_molecule("not_a_smiles")


def test_oversized_molecule_is_rejected_clearly():
    big = "C" * (MAX_NODE_LEN + 5)
    with pytest.raises(MoleculeTooLargeError, match=str(MAX_NODE_LEN)):
        parse_molecule(big)


def test_canonicalisation_is_idempotent():
    once = canonicalize("c1ccccc1C(=O)OC")
    assert canonicalize(once) == once
    assert once == canonicalize(METHYL_BENZOATE)


# ---------------------------------------------------------------------------
# 5. A small reference batch matches the upstream inference path
# ---------------------------------------------------------------------------


def test_matches_upstream_inference_path(model, receptors, embeddings, masks):
    """Our batched loop must equal upstream's own ``predict_OR_feat``.

    This is the real equivalence test for the refactor: identical weights,
    identical molecular featurisation and identical cached embeddings, driven
    once through upstream's entry point one pair at a time, and once through
    our batched path. Any disagreement is a bug in our loop.

    It also exercises the removal of upstream's ``len(smiles) == 1`` skip:
    every upstream call here is a single-molecule batch.
    """
    import dgl
    import torch

    from orvector.upstream import predict_or_feat

    upstream_predict = predict_or_feat()
    args = {
        "device": torch.device("cpu"),
        "edge_featurizer": None,
        "featurizer_type": "canonical",
        "model": "MolOR",
    }

    mol = parse_molecule(METHYL_BENZOATE)
    subset = receptors[:24]
    node_mask = np.zeros(MAX_NODE_LEN, dtype=np.float32)
    node_mask[: mol.n_atoms] = 1.0

    reference = np.zeros(len(subset), dtype=np.float32)
    with torch.inference_mode():
        for i, r in enumerate(subset):
            # A fresh batch per call: predict_OR_feat pops 'h' off the graph.
            bg = dgl.batch([mol.graph])
            assert "h" in bg.ndata, "batched graph lost its node features"
            out = upstream_predict(
                args,
                model,
                bg,
                torch.from_numpy(np.asarray(embeddings[r.panel_index])[None]),
                torch.from_numpy(masks[r.panel_index][None]),
                torch.from_numpy(node_mask[None]),
            )
            reference[i] = float(out.reshape(-1)[0])

    ours, _ = predict_matrix(
        model,
        [mol],
        subset,
        np.stack([np.asarray(embeddings[r.panel_index]) for r in subset]),
        np.stack([masks[r.panel_index] for r in subset]),
        device="cpu",
        pair_batch_size=8,
    )

    diff = np.abs(ours[0] - reference)
    print(f"\n[upstream-path] n={len(subset)} max|diff|={diff.max():.3e}")
    np.testing.assert_allclose(ours[0], reference, rtol=0, atol=2e-4)


def test_structural_agreement_with_published_logits(
    prepared, model, receptors, embeddings, masks, gs_lf_molecules
):
    """Sanity-check against upstream's published HORDE logits.

    ``olfactory_subgenome_OR_logits.pt`` is a (5862, 845) tensor whose rows
    follow the GS-LF CSV order and whose columns follow the 845-sequence
    ``value_counts`` order, so ``receptor.upstream_seq_index`` indexes it.

    This asserts structure, not equality, deliberately. The published
    ``generate_OR_predictions_pseudogenes.py`` cannot have produced that file:
    it reuses a single batched graph across its receptor loop while
    ``predict_OR_feat`` calls ``bg.ndata.pop('h')``, so the second receptor
    raises ``KeyError``. The tensor therefore came from a different, working
    revision whose exact configuration is not recoverable from the repository,
    and no combination of the published checkpoints, graph settings or
    atom-slot budgets reproduces it numerically (the configuration used here is
    the closest of those tried). What does hold, and is checked here, is that
    for a fixed receptor our scores track upstream's across molecules.

    OR2G6 is excluded: upstream's cleaning reduced its malformed gene-table
    field to the single residue "S" whereas this panel carries the repaired
    316 aa sequence, so the two are not comparable there.
    """
    import torch

    if not prepared.reference_logits.is_file():
        pytest.skip(f"missing {prepared.reference_logits}")

    reference = torch.load(prepared.reference_logits, map_location="cpu").numpy()
    assert reference.shape == (5862, 845)

    mol_rows = list(range(60))
    rec_subset = [r for r in receptors if "repaired" not in r.source][:6]

    mols = [parse_molecule(gs_lf_molecules[i]) for i in mol_rows]
    sub_emb = np.stack([np.asarray(embeddings[r.panel_index]) for r in rec_subset])
    sub_masks = np.stack([masks[r.panel_index] for r in rec_subset])

    logits, _ = predict_matrix(
        model, mols, rec_subset, sub_emb, sub_masks, device="cpu", pair_batch_size=24
    )
    expected = np.stack(
        [[reference[row, r.upstream_seq_index] for r in rec_subset] for row in mol_rows]
    )

    # Per-receptor correlation across molecules is the meaningful axis: it says
    # our model ranks odorants for a given receptor the way upstream's does.
    for k, r in enumerate(rec_subset):
        corr = np.corrcoef(logits[:, k], expected[:, k])[0, 1]
    per_receptor = []
    for k, r in enumerate(rec_subset):
        corr = np.corrcoef(logits[:, k], expected[:, k])[0, 1]
        per_receptor.append(corr)
        print(f"[published] {r.gene:10s} corr over {len(mol_rows)} molecules = {corr:.4f}")

    overall = np.corrcoef(logits.ravel(), expected.ravel())[0, 1]
    median = float(np.median(per_receptor))
    print(f"[published] median per-receptor corr = {median:.4f}, overall = {overall:.4f}")

    # Deliberately loose. Per-receptor agreement spans roughly 0.37-0.77, so the
    # bars catch gross breakage (a reordered panel, a wrong checkpoint, a broken
    # featuriser) without pretending to a numerical parity that, as the
    # docstring explains, the published artefacts cannot support.
    assert min(per_receptor) > 0.2, f"weakest receptor corr {min(per_receptor):.3f}"
    assert median > 0.4, f"median per-receptor corr {median:.3f}"
    assert overall > 0.45, f"overall corr {overall:.3f}"


# ---------------------------------------------------------------------------
# 6. After setup, prediction works offline
# ---------------------------------------------------------------------------


def test_prediction_works_offline(prepared, receptors, embeddings, masks, monkeypatch):
    """Build the model and predict with all socket access disabled."""
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("prediction attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    from orvector.model import build_model

    offline_model = build_model(device="cpu", paths=prepared)
    mol = parse_molecule(METHYL_BENZOATE)
    logits, scores = predict_matrix(
        offline_model,
        [mol],
        receptors[:8],
        np.asarray(embeddings[:8]),
        masks[:8],
        device="cpu",
        pair_batch_size=4,
    )
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Guards on assumptions the pipeline relies on
# ---------------------------------------------------------------------------


def test_upstream_clean_reproduces_known_cases():
    # Stop codon asterisk is stripped (OR10D3 case).
    assert upstream_clean("MKL*") == "MKL"
    # The OR2G6 malformed-field case that upstream reduces to one residue.
    assert upstream_clean("|Strand=+|Length=MEETNNSS") == "S"


def test_or2g6_repair_matches_uniprot(receptors):
    """The one repaired sequence must equal reviewed UniProt Q5TZ20."""
    or2g6 = next((r for r in receptors if r.gene == "OR2G6"), None)
    assert or2g6 is not None, "OR2G6 missing from panel"
    assert or2g6.uniprot == "Q5TZ20"
    assert or2g6.seq_len == 316
    assert or2g6.sequence.startswith("MEETNNSSEKGFLLLGFSDQPQLERFLF")
    assert or2g6.sequence.endswith("VKGALRTLILGSAAGQSHKD")


def test_atom_slot_budget_changes_scores(receptors, embeddings, masks):
    """``MAX_NODE_LEN`` is load-bearing, so it must stay frozen.

    Padded atom slots are zeroed only *after* ``mol_norm``, and ``LayerNorm``
    maps a zero vector to its own ``bias``. ``Linear(bias)`` is non-zero, so the
    padded slots enter the protein-side attention as keys and values even though
    ``fixed_size_tensor2[node_mask == 0] = 0`` removes them from the molecule
    aggregation. Changing the budget therefore changes the scores, which is why
    it is pinned to upstream's 98 and recorded in the manifest.
    """
    import dgl
    import torch
    import torch.nn.functional as F

    from orvector.config import MODEL_CONFIG
    from orvector.model import load_state_dict
    from orvector.upstream import disable_anomaly_detection, molor_predictor_cls

    state = load_state_dict(verify_md5=False)
    mol = parse_molecule(METHYL_BENZOATE)
    subset = receptors[:4]
    emb = torch.from_numpy(np.ascontiguousarray(np.asarray(embeddings[:4])))
    msk = torch.from_numpy(np.ascontiguousarray(masks[:4]))

    def run(budget):
        kw = dict(MODEL_CONFIG)
        kw["max_node_len"] = budget
        kw["activation"] = [F.relu] * len(kw["hidden_feats"])
        m = molor_predictor_cls()(**kw)
        disable_anomaly_detection()
        m.load_state_dict(state, strict=True)
        m.eval()
        nm = np.zeros(budget, dtype=np.float32)
        nm[: mol.n_atoms] = 1.0
        with torch.inference_mode():
            bg = dgl.batch([mol.graph] * len(subset))
            feats = bg.ndata.pop("h")
            out = m(
                bg, feats, emb, msk,
                torch.from_numpy(np.stack([nm] * len(subset))),
                torch.device("cpu"),
            )
        return out.reshape(len(subset)).numpy()

    frozen = run(MAX_NODE_LEN)
    wider = run(MAX_NODE_LEN + 40)
    delta = np.abs(frozen - wider).max()
    # ASCII only: the Windows console encodes as cp1252.
    print(f"\n[atom-slots] {MAX_NODE_LEN} vs {MAX_NODE_LEN + 40}: max abs delta={delta:.4f}")
    assert delta > 1e-3, (
        "Changing the atom-slot budget did not change the scores. If this ever "
        "passes, the sensitivity documented in config.MAX_NODE_LEN no longer "
        "holds and that comment should be revised."
    )


@pytest.mark.xfail(
    reason="The residue-padding leak is gated by torch.relu(scores) in "
           "OdorantReceptorCrossAttention, which is zero for these inputs, so "
           "shortening the padded tail leaves the logits unchanged. Kept as a "
           "live probe: if it ever XPASSes, residue padding has become "
           "load-bearing for some input and SEQ_PAD_LEN's rationale changes.",
    strict=False,
)
def test_residue_padding_tail_leaks(model, receptors, embeddings, masks):
    """Probe whether the padded residue tail affects the scores."""
    import dgl
    import torch

    mol = parse_molecule(METHYL_BENZOATE)
    subset = receptors[:4]
    full_emb = np.asarray(embeddings[:4])
    full_masks = masks[:4]
    node_mask = np.zeros(MAX_NODE_LEN, dtype=np.float32)
    node_mask[: mol.n_atoms] = 1.0

    def run(emb, msk):
        """Call the model directly: predict_matrix pins the residue width."""
        with torch.inference_mode():
            bg = dgl.batch([mol.graph] * len(subset))
            feats = bg.ndata.pop("h")
            out = model(
                bg,
                feats,
                torch.from_numpy(np.ascontiguousarray(emb)),
                torch.from_numpy(np.ascontiguousarray(msk)),
                torch.from_numpy(np.stack([node_mask] * len(subset))),
                torch.device("cpu"),
            )
        return out.reshape(len(subset)).numpy()

    full = run(full_emb, full_masks)
    # Every real residue is retained (the longest of these is well under 360);
    # only the padded tail is shortened.
    short_len = 360
    assert max(r.seq_len for r in subset) < short_len
    short = run(full_emb[:, :short_len, :], full_masks[:, :short_len])

    print(f"\n[padding] full vs tail-truncated logits differ by "
          f"{np.abs(full - short).max():.6f}")
    assert not np.allclose(full, short, atol=1e-3)


def test_embedding_is_invariant_to_batch_grouping(receptors):
    """Per-sequence caching is only valid if grouping does not matter.

    Upstream slices ``[1:-1]`` off the whole padded batch, so cached
    per-sequence embeddings are reusable only if a sequence's representation is
    independent of which sequences share its forward pass. ESM-2 masks pad
    tokens out of attention, so it should be -- this asserts it.
    """
    from orvector.embeddings import embed_sequences

    seqs = [receptors[0].sequence, receptors[1].sequence, receptors[2].sequence]
    grouped = embed_sequences(seqs, device="cpu", batch_size=3)
    alone = np.stack(
        [embed_sequences([s], device="cpu", batch_size=1)[0] for s in seqs]
    )
    np.testing.assert_allclose(grouped, alone, rtol=0, atol=1e-4)


def test_manifest_records_reproducibility_keys(manifest):
    for key in (
        "panel_hash",
        "checkpoint_md5",
        "model_config_hash",
        "encoder_config_hash",
        "seq_pad_len",
        "max_node_len",
        "software",
    ):
        assert key in manifest, f"manifest missing {key}"
    assert manifest["seq_pad_len"] == SEQ_PAD_LEN
    assert manifest["max_node_len"] == MAX_NODE_LEN
    assert manifest["checkpoint_md5"] == "98b3bcc6f0bae4c8faddb40ffb22c7d5"
