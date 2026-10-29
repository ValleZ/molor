"""Construction and loading of the frozen human olfactory receptor panel.

The panel is derived from the pinned upstream HORDE gene table
(``data/datasets/genes.csv``) and upstream's own functional/pseudogene index
split (``data/datasets/HORDE/functional_indices.json``).

Ordering
--------
Receptors are emitted in ascending ``upstream_seq_index`` -- that is, in the
order of upstream's ``Conceptual Sequence`` ``value_counts()`` listing, filtered
to the functional entries. Keeping that order means column *i* of a panel score
vector corresponds directly to a known column of upstream's pre-generated
``olfactory_subgenome_OR_logits.pt``, which is what makes the parity test
possible. The order is frozen and covered by :func:`panel_hash`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .config import Paths, SEQ_PAD_LEN, resolve_paths

#: Shortest sequence we will accept as an intact receptor. Real human ORs are
#: ~300-360 aa; the panel's 1st percentile is 302. Anything far below this is a
#: fragment or a parsing artefact.
MIN_INTACT_LEN = 250

#: A contiguous run of standard residue letters, used to recover the protein
#: out of a malformed gene-table field.
_RESIDUE_RUN = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{%d,}" % MIN_INTACT_LEN)

#: Repairs we expect to perform, each verified against UniProt. If the upstream
#: table changes such that a repair appears or disappears, panel construction
#: fails loudly rather than silently shipping a different panel.
#:
#: OR2G6's ``Conceptual Sequence`` field is malformed in genes.csv -- it reads
#: ``"|Strand=+|Length=MEETNNSS..."``. Upstream's cleaning (strip ``trand=+``,
#: then ``split("|")[1]``) reduces it to the single letter ``"S"``, so
#: upstream's own HORDE run embedded a 1-residue "receptor" here. The real
#: 316 aa protein is present in the field and matches reviewed UniProt entry
#: Q5TZ20 (OR2G6_HUMAN) exactly.
EXPECTED_REPAIRS: dict[str, tuple[str, int]] = {
    "OR2G6": ("Q5TZ20", 316),
}

SOURCE_HORDE = "HORDE:genes.csv"
SOURCE_REPAIRED = "HORDE:genes.csv(malformed-field-repaired;UniProt-verified)"


@dataclass(frozen=True)
class Receptor:
    """One panel entry."""

    panel_index: int
    receptor_id: str
    gene: str
    gene_aliases: str
    horde_id: str
    family: str
    subfamily: str
    chromosome: str
    seq_len: int
    source: str
    uniprot: str
    upstream_seq_index: int
    sequence: str


TSV_COLUMNS = [f.name for f in fields(Receptor)]


def upstream_clean(sequence: str) -> str:
    """Upstream's sequence cleaning, reproduced exactly.

    Mirrors ``scripts/generate_OR_predictions_pseudogenes.py``: strip the
    gap/stop characters, strip the ``trand=+`` fragment, then -- if an ``=``
    survives -- take the second pipe-delimited field.
    """
    sequence = sequence.replace("\\", "")
    sequence = sequence.replace("/", "")
    sequence = sequence.replace("*", "")
    sequence = sequence.replace("trand=+", "")
    if len(sequence.split("=")) != 1:
        sequence = sequence.split("|")[1]
    return sequence


def _repair(raw: str) -> str | None:
    """Recover a plausible protein from a malformed gene-table field."""
    runs = _RESIDUE_RUN.findall(raw)
    if not runs:
        return None
    return max(runs, key=len)


def _ordered_sequences(df) -> list[str]:
    """Upstream's sequence ordering: ``value_counts()`` over the raw column."""
    return df["Conceptual Sequence"].value_counts().keys().tolist()


def _str(value) -> str:
    import pandas as pd

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def build_panel(paths: Paths | None = None, *, strict_indices: bool = True) -> list[Receptor]:
    """Build the frozen panel from the pinned upstream data files."""
    import pandas as pd

    paths = paths or resolve_paths()
    for required in (paths.genes_csv, paths.functional_indices, paths.pseudogene_indices):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required upstream data file missing: {required}. "
                "Is the vendored checkout complete?"
            )

    df = pd.read_csv(paths.genes_csv, delimiter="\t")
    top_seqs = _ordered_sequences(df)

    saved_functional = json.loads(paths.functional_indices.read_text())
    saved_pseudo = json.loads(paths.pseudogene_indices.read_text())

    # Recompute upstream's split and insist it reproduces the shipped files.
    # value_counts() tie-breaking is pandas-version dependent, so this is a
    # real guard against a silently reordered panel, not a formality.
    flag_by_seq = (
        df.drop_duplicates(subset=["Conceptual Sequence"])
        .set_index("Conceptual Sequence")["Pseduo Gene"]
        .astype(int)
    )
    recomputed_functional = [
        i for i, s in enumerate(top_seqs) if int(flag_by_seq.get(s, 0)) == 0
    ]
    recomputed_pseudo = [i for i, s in enumerate(top_seqs) if int(flag_by_seq.get(s, 0)) == 1]

    if strict_indices and (
        recomputed_functional != saved_functional or recomputed_pseudo != saved_pseudo
    ):
        raise RuntimeError(
            "Recomputed HORDE functional/pseudogene indices do not match the "
            "shipped index files. This usually means the installed pandas "
            "orders Series.value_counts() ties differently than the version "
            "used upstream, which would silently reorder the panel.\n"
            f"  recomputed functional: {len(recomputed_functional)} "
            f"(saved {len(saved_functional)})\n"
            f"  recomputed pseudogene: {len(recomputed_pseudo)} "
            f"(saved {len(saved_pseudo)})"
        )

    # Row order within a shared sequence decides the primary gene symbol.
    row_order = {seq: idx for idx, seq in enumerate(df["Conceptual Sequence"])}
    by_sequence = df.groupby("Conceptual Sequence", sort=False)

    receptors: list[Receptor] = []
    repaired: dict[str, int] = {}
    dropped: list[tuple[int, str, int]] = []

    for raw_index in saved_functional:
        raw = top_seqs[raw_index]
        sequence = upstream_clean(raw)
        source = SOURCE_HORDE
        uniprot = ""

        group = by_sequence.get_group(raw)
        # Deterministic: genes.csv row order, which is stable in the file.
        group = group.sort_index()
        symbols: list[str] = [_str(s) for s in group["Symbol"] if _str(s)]
        primary = symbols[0] if symbols else f"seq{raw_index}"
        aliases = ";".join(symbols[1:])

        if len(sequence) < MIN_INTACT_LEN:
            candidate = _repair(raw)
            if candidate is None:
                dropped.append((raw_index, primary, len(sequence)))
                continue
            sequence = candidate
            source = SOURCE_REPAIRED
            repaired[primary] = len(sequence)
            if primary in EXPECTED_REPAIRS:
                uniprot = EXPECTED_REPAIRS[primary][0]

        row = group.iloc[0]
        receptors.append(
            Receptor(
                panel_index=len(receptors),
                receptor_id=f"HORDE:{_str(row.get('Horde Id'))}",
                gene=primary,
                gene_aliases=aliases,
                horde_id=_str(row.get("Horde Id")),
                family=_str(row.get("Family Id")),
                subfamily=_str(row.get("Sub Family")),
                chromosome=_str(row.get("Chromosome")),
                seq_len=len(sequence),
                source=source,
                uniprot=uniprot,
                upstream_seq_index=raw_index,
                sequence=sequence,
            )
        )

    # Audit the repair set against expectations.
    if strict_indices:
        unexpected = {g: n for g, n in repaired.items() if g not in EXPECTED_REPAIRS}
        if unexpected:
            raise RuntimeError(
                f"Unexpected sequence repairs {unexpected}. Each repair must be "
                "reviewed and verified against UniProt before being added to "
                "EXPECTED_REPAIRS."
            )
        for gene, (accession, length) in EXPECTED_REPAIRS.items():
            if gene not in repaired:
                raise RuntimeError(
                    f"Expected to repair {gene} (UniProt {accession}, {length} aa) "
                    "but the malformed field was not found. The upstream gene "
                    "table may have changed."
                )
            if repaired[gene] != length:
                raise RuntimeError(
                    f"Repaired {gene} to {repaired[gene]} aa but UniProt "
                    f"{accession} is {length} aa."
                )

    longest = max(r.seq_len for r in receptors)
    if longest > SEQ_PAD_LEN:
        raise RuntimeError(
            f"Panel contains a {longest} aa sequence but SEQ_PAD_LEN is "
            f"{SEQ_PAD_LEN}; padding would truncate it."
        )

    if dropped:
        # Not an error: a genuinely unrecoverable fragment is excluded by design.
        print(f"[panel] excluded {len(dropped)} unrecoverable fragment(s): {dropped}")

    return receptors


def panel_hash(receptors: list[Receptor]) -> str:
    """Checksum over the frozen panel: order, identity and sequence content."""
    h = hashlib.sha256()
    for r in receptors:
        h.update(f"{r.panel_index}\t{r.receptor_id}\t{r.gene}\t{r.sequence}\n".encode())
    return h.hexdigest()


def write_panel(receptors: list[Receptor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for r in receptors:
            writer.writerow(asdict(r))


def read_panel(path: Path) -> list[Receptor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Receptor panel not found at {path}. Run `python -m orvector prepare` first."
        )
    out: list[Receptor] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out.append(
                Receptor(
                    panel_index=int(row["panel_index"]),
                    receptor_id=row["receptor_id"],
                    gene=row["gene"],
                    gene_aliases=row["gene_aliases"],
                    horde_id=row["horde_id"],
                    family=row["family"],
                    subfamily=row["subfamily"],
                    chromosome=row["chromosome"],
                    seq_len=int(row["seq_len"]),
                    source=row["source"],
                    uniprot=row["uniprot"],
                    upstream_seq_index=int(row["upstream_seq_index"]),
                    sequence=row["sequence"],
                )
            )
    if not out:
        raise RuntimeError(f"Receptor panel at {path} is empty.")
    for i, r in enumerate(out):
        if r.panel_index != i:
            raise RuntimeError(
                f"Receptor panel at {path} is not in panel_index order "
                f"(row {i} has panel_index {r.panel_index}); ordering is frozen."
            )
    return out
