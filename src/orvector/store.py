"""A durable, append-only library of receptor-activation vectors.

Design decisions that matter for reuse:

* **Raw logits and scores are stored, never normalised ones.** Any useful
  normalisation (receptor z-scoring, whitening) depends on the molecule set, so
  baking it in would invalidate every stored vector each time a molecule is
  added. Normalisation happens at analysis time instead.
* **InChIKey is the identity key**, not the SMILES string, so re-adding a
  compound written a different way is a no-op rather than a duplicate point.
* **Appends are refused across configurations.** If the panel hash, checkpoint
  or encoder config differs from the store's provenance, the vectors would not
  be comparable, so the store refuses rather than silently mixing them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (
    ORVECTOR_VERSION,
    Paths,
    encoder_config_hash,
    file_md5,
    model_config_hash,
    resolve_paths,
    software_version,
)


@dataclass(frozen=True)
class Entry:
    """One stored molecule."""

    molecule_id: str
    name: str
    cls: str
    input_smiles: str
    canonical_smiles: str
    inchikey: str
    formula: str
    n_atoms: int
    added_utc: str

    @staticmethod
    def columns() -> list[str]:
        return [
            "molecule_id", "name", "class", "input_smiles", "canonical_smiles",
            "inchikey", "formula", "n_atoms", "added_utc",
        ]

    def as_row(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("cls")
        return d

    @classmethod
    def from_row(cls, row: dict) -> "Entry":
        return cls(
            molecule_id=row["molecule_id"],
            name=row["name"],
            cls=row.get("class", ""),
            input_smiles=row["input_smiles"],
            canonical_smiles=row["canonical_smiles"],
            inchikey=row["inchikey"],
            formula=row.get("formula", ""),
            n_atoms=int(row["n_atoms"]),
            added_utc=row.get("added_utc", ""),
        )


class StoreConfigMismatch(RuntimeError):
    """The store was built under a different model/panel configuration."""


class Library:
    """Append-only vector store rooted at ``artifacts/library``."""

    def __init__(self, paths: Paths | None = None, name: str = "library"):
        self.paths = paths or resolve_paths()
        self.root = self.paths.artifacts / name
        self.molecules_tsv = self.root / "molecules.tsv"
        self.vectors_npz = self.root / "vectors.npz"
        self.provenance_json = self.root / "provenance.json"

    # -- provenance ------------------------------------------------------

    def current_provenance(self, panel_hash: str) -> dict:
        return {
            "panel_hash": panel_hash,
            "checkpoint_md5": file_md5(self.paths.checkpoint),
            "model_config_hash": model_config_hash(),
            "encoder_config_hash": encoder_config_hash(),
            "orvector_version": ORVECTOR_VERSION,
            "software": software_version(),
        }

    def _check_provenance(self, panel_hash: str) -> None:
        if not self.provenance_json.is_file():
            return
        stored = json.loads(self.provenance_json.read_text())
        current = self.current_provenance(panel_hash)
        for key in ("panel_hash", "checkpoint_md5", "model_config_hash",
                    "encoder_config_hash"):
            if stored.get(key) != current[key]:
                raise StoreConfigMismatch(
                    f"Library at {self.root} was built with {key}="
                    f"{stored.get(key)} but this run has {current[key]}.\n"
                    "Vectors from different configurations are not comparable. "
                    "Either restore the original configuration or start a new "
                    "library directory."
                )

    # -- io --------------------------------------------------------------

    def exists(self) -> bool:
        return self.molecules_tsv.is_file() and self.vectors_npz.is_file()

    def load(self) -> tuple[list[Entry], np.ndarray, np.ndarray]:
        if not self.exists():
            return [], np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
        with open(self.molecules_tsv, newline="", encoding="utf-8") as fh:
            entries = [Entry.from_row(r) for r in csv.DictReader(fh, delimiter="\t")]
        blob = np.load(self.vectors_npz, allow_pickle=True)
        logits, scores = blob["logits"], blob["scores"]
        order = [str(x) for x in blob["molecule_id"]]
        if order != [e.molecule_id for e in entries]:
            raise RuntimeError(
                f"{self.vectors_npz} and {self.molecules_tsv} disagree on "
                "molecule order; the library is inconsistent."
            )
        if len(entries) != logits.shape[0]:
            raise RuntimeError("Row count mismatch between molecules and vectors.")
        return entries, logits, scores

    def _write(self, entries: list[Entry], logits: np.ndarray, scores: np.ndarray,
               panel_hash: str, receptor_ids: list[str], genes: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.molecules_tsv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=Entry.columns(), delimiter="\t",
                               lineterminator="\n")
            w.writeheader()
            for e in entries:
                w.writerow(e.as_row())
        np.savez_compressed(
            self.vectors_npz,
            logits=logits.astype(np.float32),
            scores=scores.astype(np.float32),
            molecule_id=np.array([e.molecule_id for e in entries], dtype=object),
            receptor_ids=np.array(receptor_ids, dtype=object),
            genes=np.array(genes, dtype=object),
        )
        prov = self.current_provenance(panel_hash)
        prov["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prov["n_molecules"] = len(entries)
        self.provenance_json.write_text(json.dumps(prov, indent=2, default=str))

    # -- mutation --------------------------------------------------------

    def add_from_csv(
        self,
        csv_path: Path,
        *,
        device: str = "cpu",
        pair_batch_size: int = 16,
        force: bool = False,
        progress: bool = False,
    ) -> dict:
        """Compute and append vectors for every molecule in ``csv_path``."""
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")

        from .embeddings import load_panel_embeddings, panel_residue_masks
        from .inference import InvalidSMILESError, MoleculeTooLargeError, parse_molecule, predict_matrix
        from .model import build_model
        from .panel import panel_hash, read_panel

        receptors = read_panel(self.paths.receptors_tsv)
        phash = panel_hash(receptors)
        self._check_provenance(phash)

        existing, logits, scores = self.load()
        known = {e.inchikey for e in existing}

        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"{csv_path} has no rows.")

        pending, skipped, failed = [], [], []
        seen_now: set[str] = set()
        for i, row in enumerate(rows):
            smi = (row.get("smiles") or row.get("SMILES") or "").strip()
            name = (row.get("name") or row.get("material") or f"row{i}").strip()
            if not smi:
                continue
            try:
                mol = parse_molecule(smi)
            except (InvalidSMILESError, MoleculeTooLargeError) as exc:
                failed.append((name, str(exc)))
                continue
            rd = Chem.MolFromSmiles(mol.canonical_smiles)
            key = Chem.MolToInchiKey(rd)
            if key in seen_now:
                skipped.append((name, "duplicate within input file"))
                continue
            if key in known and not force:
                skipped.append((name, "already in library"))
                continue
            seen_now.add(key)
            pending.append(
                (
                    Entry(
                        molecule_id=key,
                        name=name,
                        cls=(row.get("class") or "").strip(),
                        input_smiles=smi,
                        canonical_smiles=mol.canonical_smiles,
                        inchikey=key,
                        formula=rdMolDescriptors.CalcMolFormula(rd),
                        n_atoms=mol.n_atoms,
                        added_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ),
                    mol,
                )
            )

        stats = {
            "input_rows": len(rows),
            "to_compute": len(pending),
            "skipped": len(skipped),
            "failed": len(failed),
            "already": len(existing),
        }
        for name, why in failed:
            print(f"[library] FAILED {name}: {why}")
        for name, why in skipped:
            print(f"[library] skip {name}: {why}")

        if not pending:
            print("[library] nothing new to compute")
            if existing:
                self._write(existing, logits, scores, phash,
                            [r.receptor_id for r in receptors], [r.gene for r in receptors])
            return stats

        embeddings = load_panel_embeddings(self.paths, n_receptors=len(receptors))
        masks = panel_residue_masks(receptors)
        model = build_model(device=device, paths=self.paths)

        print(f"[library] computing {len(pending)} x {len(receptors)} receptors on {device}")
        new_logits = np.zeros((len(pending), len(receptors)), dtype=np.float32)
        new_scores = np.zeros_like(new_logits)
        for i, (entry, mol) in enumerate(pending):
            lg, sc = predict_matrix(
                model, [mol], receptors, embeddings, masks,
                device=device, pair_batch_size=pair_batch_size,
            )
            new_logits[i] = lg[0]
            new_scores[i] = sc[0]
            if progress or (i + 1) % 10 == 0 or i + 1 == len(pending):
                print(f"[library]   {i + 1}/{len(pending)}  {entry.name}", flush=True)

        if existing:
            if logits.shape[1] != new_logits.shape[1]:
                raise RuntimeError("Receptor count changed; refusing to append.")
            all_entries = existing + [e for e, _ in pending]
            all_logits = np.vstack([logits, new_logits])
            all_scores = np.vstack([scores, new_scores])
        else:
            all_entries = [e for e, _ in pending]
            all_logits, all_scores = new_logits, new_scores

        self._write(all_entries, all_logits, all_scores, phash,
                    [r.receptor_id for r in receptors], [r.gene for r in receptors])
        stats["total"] = len(all_entries)
        print(f"[library] stored {len(all_entries)} molecules -> {self.root}")
        return stats
