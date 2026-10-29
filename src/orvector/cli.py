"""``python -m orvector`` command line interface."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as cfg
from .config import (
    MAX_NODE_LEN,
    ORVECTOR_VERSION,
    SEQ_PAD_LEN,
    encoder_config_hash,
    file_md5,
    model_config_hash,
    resolve_paths,
    software_version,
)
from .embeddings import build_cache, load_panel_embeddings, panel_residue_masks
from .inference import InvalidSMILESError, MoleculeTooLargeError, parse_molecule, predict_matrix
from .model import build_model, load_state_dict, model_summary
from .panel import build_panel, panel_hash, read_panel, write_panel


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] project root: {paths.root}")
    print(f"[prepare] upstream:     {paths.upstream}")

    receptors = build_panel(paths)
    phash = panel_hash(receptors)
    write_panel(receptors, paths.receptors_tsv)
    repaired = [r for r in receptors if "repaired" in r.source]
    print(
        f"[panel] {len(receptors)} intact functional receptors "
        f"-> {paths.receptors_tsv}"
    )
    print(f"[panel] sequence lengths: {min(r.seq_len for r in receptors)}"
          f"-{max(r.seq_len for r in receptors)}")
    print(f"[panel] repaired entries: {[r.gene for r in repaired] or 'none'}")
    print(f"[panel] panel_hash: {phash}")

    # Validate the checkpoint now rather than at first prediction.
    state_dict = load_state_dict(paths, verify_md5=not args.no_verify_md5)
    summary = model_summary(state_dict)
    print(
        f"[model] checkpoint ok: {summary['n_tensors']} tensors, "
        f"{summary['n_parameters']:,} parameters"
    )
    build_model(device="cpu", paths=paths, state_dict=state_dict)
    print("[model] strict weight load succeeded")

    stats = build_cache(
        receptors,
        paths=paths,
        device=args.device,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(f"[embed] {stats['computed']} computed, {stats['reused']} reused")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "orvector_version": ORVECTOR_VERSION,
        "panel_hash": phash,
        "n_receptors": len(receptors),
        "receptors_tsv": paths.receptors_tsv.name,
        "repaired_receptors": [
            {"gene": r.gene, "uniprot": r.uniprot, "seq_len": r.seq_len} for r in repaired
        ],
        "checkpoint_md5": file_md5(paths.checkpoint),
        "model_config_hash": model_config_hash(),
        "encoder_config_hash": encoder_config_hash(),
        "seq_pad_len": SEQ_PAD_LEN,
        "max_node_len": MAX_NODE_LEN,
        "software": software_version(),
        **cfg.config_dict(),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[prepare] manifest -> {paths.manifest}")
    print("[prepare] done; prediction now works offline")
    return 0


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------


def _read_smiles_csv(path: Path, column: str | None) -> list[tuple[str, str]]:
    """Return ``(identifier, smiles)`` rows from a CSV."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV {path} has no header row.")
        fields = reader.fieldnames
        if column is None:
            for candidate in ("smiles", "SMILES", "Smiles", "canonical_smiles",
                              "IsomericSMILES"):
                if candidate in fields:
                    column = candidate
                    break
            else:
                raise SystemExit(
                    f"Could not find a SMILES column in {path}. "
                    f"Columns are {fields}; pass --smiles-column."
                )
        elif column not in fields:
            raise SystemExit(f"Column {column!r} not in {path}; columns are {fields}.")

        id_col = next(
            (c for c in ("id", "ID", "name", "Name", "CID", "cid") if c in fields), None
        )
        rows: list[tuple[str, str]] = []
        for i, row in enumerate(reader):
            smiles = (row.get(column) or "").strip()
            if not smiles:
                continue
            ident = (row.get(id_col) or "").strip() if id_col else ""
            rows.append((ident or f"row{i}", smiles))
    if not rows:
        raise SystemExit(f"No SMILES found in {path} (column {column!r}).")
    return rows


def cmd_predict(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)

    if bool(args.smiles) == bool(args.csv):
        raise SystemExit("Provide exactly one of --smiles or --csv.")

    receptors = read_panel(paths.receptors_tsv)
    phash = panel_hash(receptors)
    if paths.manifest.is_file():
        manifest = json.loads(paths.manifest.read_text())
        if manifest.get("panel_hash") not in (None, phash):
            raise SystemExit(
                "Panel hash does not match the manifest written by `prepare`:\n"
                f"  receptors.tsv: {phash}\n"
                f"  manifest:      {manifest.get('panel_hash')}\n"
                "Re-run `python -m orvector prepare`."
            )
        if manifest.get("encoder_config_hash") not in (None, encoder_config_hash()):
            raise SystemExit(
                "Encoder configuration changed since `prepare`; cached embeddings "
                "are not reusable. Re-run `python -m orvector prepare`."
            )
    else:
        manifest = {}

    if args.csv:
        entries = _read_smiles_csv(Path(args.csv), args.smiles_column)
    else:
        entries = [("input0", args.smiles)]

    molecules = []
    identifiers = []
    for ident, smiles in entries:
        try:
            molecules.append(parse_molecule(smiles))
        except (InvalidSMILESError, MoleculeTooLargeError) as exc:
            raise SystemExit(f"error [{ident}]: {exc}") from exc
        identifiers.append(ident)

    embeddings = load_panel_embeddings(paths, n_receptors=len(receptors))
    masks = panel_residue_masks(receptors)
    model = build_model(device=args.device, paths=paths, verify_md5=not args.no_verify_md5)

    print(
        f"[predict] {len(molecules)} molecule(s) x {len(receptors)} receptors "
        f"on {args.device}"
    )
    logits, scores = predict_matrix(
        model,
        molecules,
        receptors,
        embeddings,
        masks,
        device=args.device,
        pair_batch_size=args.batch_size,
        progress=args.progress,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=scores,
        logits=logits,
        receptor_ids=np.array([r.receptor_id for r in receptors], dtype=object),
        genes=np.array([r.gene for r in receptors], dtype=object),
        upstream_seq_index=np.array([r.upstream_seq_index for r in receptors], dtype=np.int32),
        molecule_ids=np.array(identifiers, dtype=object),
        smiles_input=np.array([m.input_smiles for m in molecules], dtype=object),
        smiles_canonical=np.array([m.canonical_smiles for m in molecules], dtype=object),
        panel_hash=phash,
        checkpoint_md5=manifest.get("checkpoint_md5", file_md5(paths.checkpoint)),
        model_config_hash=model_config_hash(),
        encoder_config_hash=encoder_config_hash(),
        upstream_commit=cfg.UPSTREAM_COMMIT,
        orvector_version=ORVECTOR_VERSION,
        seq_pad_len=SEQ_PAD_LEN,
        max_node_len=MAX_NODE_LEN,
    )
    print(f"[predict] wrote {out_path}  scores{scores.shape} logits{logits.shape}")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(
                ["molecule_id", "smiles_canonical", "panel_index", "receptor_id",
                 "gene", "logit", "score"]
            )
            for mi, mol in enumerate(molecules):
                for r in receptors:
                    writer.writerow(
                        [
                            identifiers[mi],
                            mol.canonical_smiles,
                            r.panel_index,
                            r.receptor_id,
                            r.gene,
                            f"{logits[mi, r.panel_index]:.6f}",
                            f"{scores[mi, r.panel_index]:.6f}",
                        ]
                    )
        print(f"[predict] wrote {csv_path}")

    top = int(np.argmax(scores[0]))
    print(
        f"[predict] {molecules[0].canonical_smiles}: "
        f"score range {scores[0].min():.4f}-{scores[0].max():.4f}, "
        f"top receptor {receptors[top].gene} ({scores[0, top]:.4f})"
    )
    return 0


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------


def cmd_library(args: argparse.Namespace) -> int:
    from .store import Library

    paths = resolve_paths(args.root)
    lib = Library(paths)

    if args.action == "add":
        if not args.csv:
            raise SystemExit("library add requires --csv")
        lib.add_from_csv(
            Path(args.csv),
            device=args.device,
            pair_batch_size=args.batch_size,
            force=args.force,
            progress=args.progress,
        )
        return 0

    entries, logits, _ = lib.load()
    if not entries:
        print(f"Library at {lib.root} is empty. Add molecules with:\n"
              f"  python -m orvector library add --csv <file.csv>")
        return 0
    prov = json.loads(lib.provenance_json.read_text())
    print(f"{len(entries)} molecules x {logits.shape[1]} receptors")
    print(f"panel_hash      {prov['panel_hash']}")
    print(f"checkpoint_md5  {prov['checkpoint_md5']}")
    print(f"updated         {prov.get('updated_utc')}")
    from collections import Counter

    print("\nclass distribution:")
    for cls, n in Counter(e.cls or "(none)" for e in entries).most_common():
        print(f"  {cls:24s} {n:3d}")
    if args.verbose:
        print()
        for e in entries:
            print(f"  {e.name:46s} {e.formula:12s} {e.canonical_smiles}")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    from .project import project, render
    from .store import Library

    paths = resolve_paths(args.root)
    lib = Library(paths, name=args.library)
    entries, logits, scores = lib.load()
    if len(entries) < 3:
        raise SystemExit(
            f"Need at least 3 molecules in the library (have {len(entries)}). "
            "Add some with `python -m orvector library add --csv ...`."
        )

    blob = np.load(lib.vectors_npz, allow_pickle=True)
    genes = [str(x) for x in blob["genes"]]
    names = [e.name for e in entries]
    classes = [e.cls or "unclassified" for e in entries]
    potency = scores.sum(axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",")
    variants = [(False, "all-components")] if args.keep_pc1 else \
               [(False, "all-components"), (True, "pc1-removed")]

    summary = []
    for method in methods:
        for drop, tag in variants:
            proj = project(
                logits, genes, method=method.strip(), drop_pc1=drop,
                n_components=args.n_components, edge_k=args.edge_k,
            )
            stem = f"receptor_map_{method.strip()}_{tag}"
            png = out_dir / f"{stem}.png"
            render(
                proj, names, classes, potency, png,
                title=f"Aromachemical library in olfactory receptor space "
                      f"({len(entries)} materials x {logits.shape[1]} receptors, "
                      f"{args.library.replace('library_', '').replace('library', 'GCN').upper()} encoder)",
                figsize=(args.width, args.height), dpi=args.dpi,
            )
            with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["name", "class", "x", "y", "total_activation", "canonical_smiles"])
                for i, e in enumerate(entries):
                    w.writerow([e.name, e.cls, f"{proj.coords[i, 0]:.6f}",
                                f"{proj.coords[i, 1]:.6f}", f"{potency[i]:.4f}",
                                e.canonical_smiles])
            summary.append((method.strip(), tag, proj.shepard, proj.stress))

    print("\nfidelity summary (higher Shepard = 2D distances more faithful)")
    print(f"  {'method':8s} {'variant':16s} {'shepard':>8s} {'stress':>8s}")
    for method, tag, sh, st in sorted(summary, key=lambda r: -r[2]):
        print(f"  {method:8s} {tag:16s} {sh:8.3f} "
              + (f"{st:8.4f}" if st is not None else f"{'-':>8s}"))
    best = max(summary, key=lambda r: r[2])
    print(f"\nmost faithful: {best[0]} / {best[1]} (Shepard {best[2]:.3f})")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    print(json.dumps(
        {
            "root": str(paths.root),
            "artifacts_present": {
                "receptors_tsv": paths.receptors_tsv.is_file(),
                "panel_embeddings": paths.panel_embeddings.is_file(),
                "manifest": paths.manifest.is_file(),
                "checkpoint": paths.checkpoint.is_file(),
            },
            "config": cfg.config_dict(),
            "software": software_version(),
            "manifest": json.loads(paths.manifest.read_text())
            if paths.manifest.is_file()
            else None,
        },
        indent=2,
        default=str,
    ))
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orvector",
        description="Predict human olfactory receptor responsiveness from SMILES.",
    )
    parser.add_argument("--root", default=None, help="Project root (default: auto).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Freeze the panel and cache ESM embeddings.")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto (default: auto)")
    p.add_argument("--batch-size", type=int, default=8, help="Sequences per embedding batch.")
    p.add_argument("--force", action="store_true", help="Recompute cached embeddings.")
    p.add_argument("--no-verify-md5", action="store_true", help="Skip checkpoint checksum.")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("predict", help="Score molecules against the frozen panel.")
    p.add_argument("--smiles", help="A single SMILES string.")
    p.add_argument("--csv", help="CSV of molecules (multi-molecule input).")
    p.add_argument("--smiles-column", default=None, help="SMILES column name in --csv.")
    p.add_argument("--output", required=True, help="Output .npz path.")
    p.add_argument("--csv-out", default=None, help="Also write a readable CSV.")
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu | cuda (default: cpu; cuda needs a CUDA-enabled DGL build)",
    )
    p.add_argument("--batch-size", type=int, default=16, help="(molecule, receptor) pairs per batch.")
    p.add_argument("--progress", action="store_true", help="Print per-batch progress.")
    p.add_argument("--no-verify-md5", action="store_true", help="Skip checkpoint checksum.")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("library", help="Manage the persistent vector library.")
    p.add_argument("action", choices=["add", "list"], help="What to do.")
    p.add_argument("--csv", default=None, help="Input CSV for `add`.")
    p.add_argument("--device", default="cpu", help="cpu | cuda (default: cpu)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--force", action="store_true", help="Recompute molecules already stored.")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--verbose", action="store_true", help="List every molecule.")
    p.set_defaults(func=cmd_library)

    p = sub.add_parser("project", help="Project the library to 2D and render maps.")
    p.add_argument("--methods", default="pca,mds",
                   help="Comma-separated: pca, mds, tsne, umap (default: pca,mds)")
    p.add_argument("--out-dir", default="artifacts/maps")
    p.add_argument("--n-components", type=int, default=20,
                   help="PCs retained before the 2D step (default: 20)")
    p.add_argument("--keep-pc1", action="store_true",
                   help="Only render the all-components variant.")
    p.add_argument("--width", type=float, default=22.0)
    p.add_argument("--height", type=float, default=15.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--edge-k", type=int, default=4,
                   help="Mutual-kNN depth for the true-neighbour links (0 = none).")
    p.add_argument("--library", default="library",
                   help="Library directory under artifacts/ (default: library)")
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("info", help="Show configuration and artifact status.")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
