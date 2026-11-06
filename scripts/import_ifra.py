"""IFRA limits as tables, read from the standards IFRA actually publishes.

The perfumer's ``tools/ifra_limits.csv`` is a hand-entered seed of 43 cat-4
figures, and its docstring says there is no official machine-readable IFRA
dataset. That is out of date: IFRA publishes an **overview workbook** and points
people at it in its own FAQ, because it ships no software.

    IFRA 51st Amendment - IFRA Standards Overview (XLSX). 263 standards.
    https://ifrafragrance.org/initiatives-positions/safe-use-fragrance-science/
        ifra-standards/ifra-standards-documentation
    (files served from d3t14p1xronwr0.cloudfront.net/docs/Standards-Documentation/)

One row per Standard, one column per IFRA product category -- 18 of them, since
5, 7, 10 and 11 are lettered -- holding the Maximum Acceptable Concentration as
**% w/w of the finished consumer product**. That is the same convention the
engine already assumes (``CONCENTRATE_PCT`` scales it to the concentrate), so
nothing here converts units.

Three things about the sheet decide this importer's shape:

*Blank is not "no restriction".* 86 of the 263 standards are prohibitions or
specifications only, and their 18 category cells are empty because there is no
limit to state. "No Restriction" is a separate, explicit value that appears in
136 + 46 cells -- 132 of them in Category 12. A parser that folds the two
together invents a 100% cap for 68 prohibited substances.

*The scope is open by default.* 259 of 263 standards say "the scope of this
Standard includes, but is not limited to, the CAS numbers indicated", so a CAS
list is evidence of membership, never proof of absence. That is why this script
emits tables and refuses to decide anything: membership and verdicts belong to
the resolver, where each rule can be read and argued with.

*The seed is not a ceiling, it is a source of errors.* Of its 43 rows, 15
disagree with the workbook. Farnesol is the one arbitrated against the official
PDF: the standard says 1.2%, the seed says 0.4%. Benzyl acetate is capped at 9.4
by the seed and has **no standard at all** in the 51st -- 9.4 is the benzyl
benzoate figure. So this table REPLACES the seed; it is never merged into it.
The merge rule that was there ("never loosen") would have preserved the wrong
strict values forever.

Column offsets are read from the header row by name, because getting them wrong
is silent: column 8 is "CAS numbers" and column 9 is "CAS numbers (comment)",
and reading the comment column yields CAS for 3 rows out of 263.

Usage:
    .venv/Scripts/python.exe scripts/import_ifra.py
    .venv/Scripts/python.exe scripts/import_ifra.py --overview path/to.xlsx

Gates printed at the end, all of which must hold:
    * 263 standards, 4734 limit rows (263 x 18)
    * 0 unparsed cells
    * 86 standards with no category limits at all, 177 with numbers
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERVIEW = (ROOT / "vendor" / "ifra"
                    / "ifra-51st-amendment-ifra-standards-overview.xlsx")
DEFAULT_ANNEX = (ROOT / "vendor" / "ifra"
                 / "ifra-51st-amendment-annex-on-contributions-from-other-sources.xlsx")
OUT = ROOT / "data" / "ifra"

SOURCE_URL = ("https://ifrafragrance.org/initiatives-positions/safe-use-"
              "fragrance-science/ifra-standards/ifra-standards-documentation")

CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
CATEGORY_RE = re.compile(r"^category\s+(.+?)\s*\(%\)$", re.I)
NO_RESTRICTION_RE = re.compile(r"^no\s*restriction", re.I)
NUMBER_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(.*)$")

#: The 18 IFRA product categories, in the order the workbook gives them.
CATEGORIES = ["1", "2", "3", "4", "5A", "5B", "5C", "5D", "6", "7A", "7B",
              "8", "9", "10A", "10B", "11A", "11B", "12"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_cell(value) -> tuple[str, float | None, str]:
    """(state, pct, note) for one category cell.

    The text values in the sheet are not noise: "No Restriction" is a claim,
    blank is the absence of a claim, and "See Notebox" is a figure we have not
    read yet. Sixteen cells are comma decimals ("0,10") and thirteen carry an
    annotation ("0.0015 (5-MOP)"), so a bare float() throws away a sixth of a
    standard.
    """
    if value is None or str(value).strip() == "":
        return "not_applicable", None, ""
    if isinstance(value, (int, float)):
        return "limit", float(value), ""
    text = str(value).strip()
    if NO_RESTRICTION_RE.match(text):
        return "no_restriction", None, ""
    if re.search(r"note", text, re.I):
        return "see_notebox", None, text
    m = NUMBER_RE.match(text)
    if m:
        return "limit", float(m.group(1).replace(",", ".")), m.group(2).strip()
    return "unparsed", None, text


def split_list(cell) -> list[str]:
    """Newline-separated cells -> list. CAS and synonyms both arrive this way."""
    return [v.strip() for v in str(cell or "").split("\n") if v.strip()]


def column_map(header: list, path: Path) -> dict:
    """Header name -> index, matched by prefix so a re-issued sheet still reads.

    Also the guard against the off-by-one that motivated this function: CAS
    lives in "CAS numbers" and is followed by "CAS numbers (comment)", and the
    comment column only contains a CAS number for 3 of 263 rows.
    """
    wanted = {
        "id": "key",
        "amendment": "amendment number",
        "prev_year": "year of previous publication",
        "last_year": "year of last publication",
        "deadline_existing": "implementation deadline for existing",
        "deadline_new": "implementation deadline for new",
        "name": "name of the ifra standard",
        "cas": "cas numbers",
        "cas_comment": "cas numbers (comment)",
        "synonyms": "synonyms",
        "type": "ifra standard type",
        "property": "intrinsic property",
        "phototox_note": "restricted ingredients due to phototoxic",
        "restricted_note": "restricted ingredients: notes",
        "spec_note": "specified ingredients: notes",
        "other_sources": "contributions from other sources",
    }
    map_: dict[str, int] = {}
    lowered = [str(h or "").strip().lower() for h in header]
    for key, prefix in wanted.items():
        for i, h in enumerate(lowered):
            if h.startswith(prefix):
                map_[key] = i
                break
        else:
            raise SystemExit(f"{path.name}: header has no column starting "
                             f"{prefix!r}; header is {lowered}")
    for i, h in enumerate(lowered):
        m = CATEGORY_RE.match(str(header[i] or "").strip())
        if m and str(m.group(1)).upper() in CATEGORIES:
            map_[f"cat:{str(m.group(1)).upper()}"] = i
    missing = [c for c in CATEGORIES if f"cat:{c}" not in map_]
    if missing:
        raise SystemExit(f"{path.name}: category columns missing {missing}")
    return map_


def read_standards(path: Path) -> tuple[list[dict], dict]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Row 1 is a provenance sentence, row 2 is empty, row 3 is the header.
    header_idx = next(i for i, r in enumerate(rows[:6])
                      if any("ifra standard type" in str(c or "").lower()
                             for c in r))
    header = list(rows[header_idx])
    col = column_map(header, path)

    standards = []
    for r in rows[header_idx + 1:]:
        if not r or not r[col["id"]]:
            continue
        cells = {c: parse_cell(r[col[f"cat:{c}"]]) for c in CATEGORIES}
        standards.append({
            "standard_id": str(r[col["id"]]).strip(),
            "name": str(r[col["name"]] or "").strip(),
            "type": str(r[col["type"]] or "").strip(),
            "amendment": str(r[col["amendment"]] or "").strip(),
            "last_publication_year": str(r[col["last_year"]] or "").strip(),
            "deadline_existing": str(r[col["deadline_existing"]] or "").strip(),
            "deadline_new": str(r[col["deadline_new"]] or "").strip(),
            "intrinsic_property": str(r[col["property"]] or "").strip(),
            "cas_numbers": split_list(r[col["cas"]]),
            "synonyms": split_list(r[col["synonyms"]]),
            "scope_open": "not limited to" in str(r[col["cas_comment"]] or "").lower(),
            "notes_json": json.dumps({
                "phototoxicity": str(r[col["phototox_note"]] or "").strip(),
                "restricted": str(r[col["restricted_note"]] or "").strip(),
                "specified": str(r[col["spec_note"]] or "").strip(),
                "other_sources": str(r[col["other_sources"]] or "").strip(),
            }, ensure_ascii=False),
            "cells": cells,
        })
    return standards, {"header_row": header_idx + 1}


def read_contributions(path: Path) -> list[dict]:
    """ONE table for "this material contains that restricted thing at X%".

    Two sheets of the annex are the same shape and were kept apart for no
    reason: a natural complex substance carries restricted constituents, and a
    Schiff base carries its restricted aldehyde. Both allow the one derivation
    a material with no standard of its own needs:

        cap = limit(constituent cat-4, %) / level(constituent, %)

    Aurantiol is the worked example: hydroxycitronellal at 2.1% in the standard
    and 56.4% of the Schiff base gives 3.72%. Lemongrass oil is the same
    arithmetic with citral.
    """
    if not path.is_file():
        return []
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []

    def level_of(cell) -> float | None:
        _, pct, _ = parse_cell(cell)
        return pct

    # --- naturals: NCS -> its restricted constituents ----------------------
    if "Natural contributions" in wb.sheetnames:
        ws = wb["Natural contributions"]
        rows = list(ws.iter_rows(values_only=True))
        start = next((i for i, r in enumerate(rows[:12])
                      if any("NCS NAME" in str(c or "").upper() for c in r)), 6)
        for r in rows[start + 1:]:
            if not r or len(r) < 10 or not r[5] or not r[7]:
                continue
            cid = CAS_RE.findall(str(r[7] or ""))
            if not cid:
                continue
            ncs_cas = CAS_RE.findall(str(r[3] or "")) + CAS_RE.findall(str(r[4] or ""))
            out.append({
                "kind": "ncs",
                "parent_name": str(r[5] or "").strip(),
                "parent_cas": "; ".join(ncs_cas),
                "parent_extra": str(r[6] or "").strip(),   # botanical name
                "constituent_name": str(r[8] or "").strip(),
                "constituent_cas": cid[0],
                "level_pct": ("" if level_of(r[9]) is None
                              else f"{level_of(r[9]):g}"),
                "level_raw": str(r[9] or "").strip(),
            })

    # --- Schiff bases: base -> its restricted aldehyde ---------------------
    if "Schiff bases" in wb.sheetnames:
        ws = wb["Schiff bases"]
        rows = list(ws.iter_rows(values_only=True))
        start = next(i for i, r in enumerate(rows[:8])
                     if any("Schiff base" in str(c or "") for c in r))
        for r in rows[start + 1:]:
            if not r or not r[0]:
                continue
            ald = CAS_RE.findall(str(r[1] or ""))
            base = CAS_RE.findall(str(r[3] or ""))
            if not ald or not base:
                continue
            out.append({
                "kind": "schiff",
                "parent_name": str(r[2] or "").strip(),
                "parent_cas": "; ".join(base),
                "parent_extra": "",
                "constituent_name": str(r[0] or "").strip(),
                "constituent_cas": ald[0],
                "level_pct": ("" if level_of(r[4]) is None
                              else f"{level_of(r[4]):g}"),
                "level_raw": str(r[4] or "").strip(),
            })
    wb.close()
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    ap.add_argument("--annex", type=Path, default=DEFAULT_ANNEX)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    for p in (args.overview,):
        if not p.is_file():
            raise SystemExit(f"{p} not found. It is gitignored: download "
                             f"'IFRA 51st Amendment - IFRA Standards Overview "
                             f"(XLS)' from {SOURCE_URL}")

    standards, meta = read_standards(args.overview)
    contributions = read_contributions(args.annex)

    limit_rows = []
    for s in standards:
        for c in CATEGORIES:
            state, pct, note = s["cells"][c]
            limit_rows.append({
                "standard_id": s["standard_id"],
                "category": c,
                "amendment": s["amendment"],
                "state": state,
                "pct": "" if pct is None else f"{pct:g}",
                "note": note,
            })

    write_csv(args.out / "standards.csv", [
        {**{k: v for k, v in s.items() if k not in ("cells", "cas_numbers", "synonyms")},
         "cas_numbers": "; ".join(s["cas_numbers"]),
         "synonyms": "; ".join(s["synonyms"])}
        for s in standards
    ], ["standard_id", "name", "type", "amendment", "last_publication_year",
        "deadline_existing", "deadline_new", "intrinsic_property", "scope_open",
        "cas_numbers", "synonyms", "notes_json"])
    write_csv(args.out / "limits.csv", limit_rows,
              ["standard_id", "category", "amendment", "state", "pct", "note"])
    if contributions:
        write_csv(args.out / "contributions.csv", contributions,
                  ["kind", "parent_name", "parent_cas", "parent_extra",
                   "constituent_name", "constituent_cas", "level_pct",
                   "level_raw"])

    # ---- gates -------------------------------------------------------------
    states = Counter(r["state"] for r in limit_rows)
    all_blank = [s["standard_id"] for s in standards
                 if all(s["cells"][c][0] == "not_applicable" for c in CATEGORIES)]
    any_number = [s["standard_id"] for s in standards
                  if any(s["cells"][c][0] == "limit" for c in CATEGORIES)]
    cat4 = sum(1 for s in standards if s["cells"]["4"][0] == "limit")

    print(f"[ifra] {len(standards)} standards -> {args.out.relative_to(ROOT)}")
    print(f"[ifra] {len(limit_rows)} limit rows "
          f"({len(standards)} x {len(CATEGORIES)})")
    print(f"[ifra] cell states: "
          + ", ".join(f"{k} {v}" for k, v in states.most_common()))
    print(f"[ifra] {len(all_blank)} standards with no category limits "
          f"(prohibition/specification only), {len(any_number)} with numbers, "
          f"{cat4} with a category-4 figure")
    if contributions:
        by_kind = Counter(c["kind"] for c in contributions)
        print(f"[ifra] {len(contributions)} contributions "
              + ", ".join(f"{k} {v}" for k, v in by_kind.most_common())
              + ", "
              + str(sum(1 for c in contributions if c["level_pct"]))
              + " with a level")
    open_scope = sum(1 for s in standards if s["scope_open"])
    print(f"[ifra] {open_scope}/{len(standards)} standards declare an open scope "
          f"(\"includes but is not limited to\") -- absence from a CAS list is "
          f"not evidence of absence")

    ok = True
    if states["unparsed"]:
        ok = False
        print(f"[ifra] GATE FAILED: {states['unparsed']} unparsed cells:")
        for r in limit_rows:
            if r["state"] == "unparsed":
                print(f"         {r['standard_id']} cat {r['category']}: "
                      f"{r['note']!r}")
    if len(limit_rows) != len(standards) * len(CATEGORIES):
        ok = False
        print("[ifra] GATE FAILED: limit rows do not cover every standard x category")
    if len(all_blank) != 86 or cat4 == 0:
        ok = False
        print(f"[ifra] GATE FAILED: expected 86 standards with no limits, "
              f"got {len(all_blank)}")

    manifest = {
        "source": {
            "overview": args.overview.name,
            "overview_sha256": sha256(args.overview),
            "annex": args.annex.name if args.annex.is_file() else None,
            "annex_sha256": sha256(args.annex) if args.annex.is_file() else None,
            "url": SOURCE_URL,
            # Not one number: the 263 standards were last touched by 16
            # different amendments, and each row keeps its own.
            "amendments_represented": sorted(
                {s["amendment"] for s in standards if s["amendment"]},
                key=lambda a: int(a) if a.isdigit() else 0),
            "imported": date.today().isoformat(),
        },
        "meta": meta,
        "counts": {
            "standards": len(standards),
            "limit_rows": len(limit_rows),
            "states": dict(states),
            "standards_without_limits": len(all_blank),
            "standards_with_numbers": len(any_number),
            "category4_figures": cat4,
            "contributions": len(contributions),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ifra] manifest: {args.out.relative_to(ROOT) / 'manifest.json'} "
          f"sha256 {manifest['source']['overview_sha256'][:16]}...")
    print("[ifra] gates " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
