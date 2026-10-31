"""Paste a formula, get an atlas with it on it. One command, nothing committed.

This is the throwaway path. A formula is not a material and has no business in
the library store or in a published build, but the question worth asking of a
formula -- where does it land, and next to what -- can only be answered by
laying it out with everything else. So each blend is injected at export time,
drawn as a star, and written to a build nobody deploys.

    python scripts/try_formula.py data/formulas/queries/query-3.csv
    python scripts/try_formula.py v1.txt v2.txt --label "v1" --label "v2"

Several formulas go on one atlas, which is how two versions get compared: the
layout is fitted with both stars in it, so the distance between them is a
distance in the same space as everything else, and each blend after the first
is also measured against the one before it.

Input is either a ``material,dilution,weight`` CSV or a pasted formula sheet:
one ingredient per line, name first and amount last, tabs or spaces between,
percent signs optional. A stock strength written into the name is read as the
dilution, so ``Calone 10%  40`` contributes four parts of calone rather than
forty.

Nothing here touches the library, the percept model or any published artifact.
The intermediate files go to artifacts/temp/, which is ignored by git, and the
page carries no analytics because no token is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / "artifacts" / "temp"

#: "Indole 10% in DPG", "Galaxolide 50%", "Hedione (at 50%)" -- a stock
#: strength written into the name, which is a dilution rather than part of the
#: material's identity.
STOCK = re.compile(
    r"[\s(@]+(?:at\s*)?(\d+(?:\.\d+)?)\s*%?\s*(?:in\s+\w+)?\)?\s*$", re.I)


def parse_sheet(path: Path):
    """Rows of (name, dilution, weight) from a CSV or a pasted sheet."""
    text = path.read_text(encoding="utf-8-sig")
    head = text.splitlines()[0].lower() if text.strip() else ""
    if "material" in head and "weight" in head:
        rows = []
        for r in csv.DictReader(text.splitlines()):
            if (r.get("material") or "").strip():
                rows.append((r["material"].strip(),
                             float(r.get("dilution") or 100),
                             float(r.get("weight") or 0)))
        return rows

    rows = []
    for line in text.splitlines():
        s = line.strip().strip("|").strip()
        if not s or s.startswith("#"):
            continue
        # Name first, amount last. Any separator: a tab, runs of spaces, or a
        # pipe from a pasted markdown table.
        m = re.match(r"^(.*?)[\s|]+(\d+(?:[.,]\d+)?)\s*%?\s*$", s)
        if not m:
            continue
        name = m.group(1).strip(" |\t")
        amount = float(m.group(2).replace(",", "."))
        dil = 100.0
        # Only treat a trailing number in the name as a stock strength when it
        # carries a percent sign or an "in <solvent>": "Aldehyde C-12" and
        # "Galaxolide 50" are names, and reading the 50 as a dilution would
        # quietly divide the ingredient by two.
        hit = STOCK.search(name)
        if hit and ("%" in name[hit.start():]
                    or re.search(r"\bin\s+\w+", name[hit.start():], re.I)):
            dil = float(hit.group(1))
            name = name[: hit.start()].strip(" ,-(")
        if name:
            rows.append((name, dil, amount))
    return rows


def run(cmd: list[str]) -> None:
    print("  $ " + " ".join(str(c) for c in cmd[1:]), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"step failed with code {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("formula", nargs="+",
                    help="one or more material,dilution,weight CSVs or pasted "
                         "formula sheets; several go on one atlas")
    ap.add_argument("--label", action="append", default=None,
                    help="name for the star; repeat once per formula")
    ap.add_argument("--owned", action="store_true",
                    help="also list the nearest materials on the shelf, which "
                         "are the only ones a blotter can check")
    ap.add_argument("--rule", default="noisy-or",
                    choices=["noisy-or", "weighted-mean", "max", "dominant"])
    ap.add_argument("--place", default="logit-mean",
                    choices=["logit-mean", "union", "mean"],
                    help="how the star is positioned; see predict_formula.py. "
                         "union is noisy-OR, which puts any many-ingredient "
                         "blend on the rim of the cloud whatever it smells of")
    ap.add_argument("--both-encoders", action="store_true",
                    help="export both layouts; the default exports the GCN "
                         "one only, which is half the work for a throwaway")
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--keep-notes", action="store_true",
                    help="include the private smelling notes; this build is "
                         "local and never deployed, so they are on by default")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    TEMP.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    stores = "library,library_mpnn_clean" if args.both_encoders else "library"
    labels = args.label or []

    jobs = []
    for k, name in enumerate(args.formula):
        src = Path(name)
        src = src if src.is_absolute() else ROOT / src
        if not src.is_file():
            raise SystemExit(f"no such formula file: {src}")
        rows = parse_sheet(src)
        if len(rows) < 3:
            raise SystemExit(f"read only {len(rows)} ingredients from "
                             f"{src.name}; expected 'name  amount' per line")
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", src.stem)
        csv_path = TEMP / f"{stem}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["material", "dilution", "weight"])
            w.writerows(rows)
        total = sum(r[2] * r[1] / 100.0 for r in rows)
        label = labels[k] if k < len(labels) else f"BLEND: {src.stem}"
        print(f"[try] {len(rows)} ingredients, {total:.2f} parts neat "
              f"material -> {csv_path.relative_to(ROOT)}   [{label}]")
        jobs.append((stem, csv_path, label, TEMP / f"{stem}_point.json"))

    stem0 = jobs[0][0]
    data = TEMP / f"{stem0}_viewer.json"
    page = TEMP / f"{stem0}_atlas.html"

    for k, (stem, csv_path, label, point) in enumerate(jobs, 1):
        print("")
        print(f"[try] 1/3 predicting {label}  ({k} of {len(jobs)})")
        cmd = [py, "scripts/predict_formula.py", csv_path, "--rule", args.rule,
               "--percepts", args.percepts, "--viewer-point", point,
               "--place", args.place, "--label", label, "--stores", stores]
        if args.owned:
            cmd.append("--owned")
        # Each formula after the first is measured against the one before it,
        # which is the question two versions of a formula actually raise.
        if k > 1:
            cmd += ["--compare", jobs[k - 2][3]]
        run(cmd)

    # One query file for the export, so every blend lands on one atlas and the
    # layout is fitted with all of them in it. Comparing stars across two
    # separately fitted layouts would compare two different pictures.
    merged = TEMP / f"{stem0}_points.json"
    queries = []
    for _, _, _, point in jobs:
        queries += json.loads(point.read_text(encoding="utf-8"))["queries"]
    merged.write_text(json.dumps({"queries": queries}, indent=2),
                      encoding="utf-8")

    print("")
    print(f"[try] 2/3 exporting the atlas with {len(queries)} blend(s) on it")
    export = [py, "scripts/export_viewer_data.py", "--out", data,
              "--percepts", args.percepts, "--query", merged]
    if not args.both_encoders:
        export += ["--models", "gcn=library"]
    if not args.keep_notes:
        export += ["--no-notes"]
    run(export)

    print("")
    print("[try] 3/3 building the page")
    run([py, "scripts/build_viewer.py", data, page])
    run([py, "scripts/check_viewer_js.py", page])

    print("")
    print(f"[try] open {page}")
    print("[try] stars: " + ", ".join(j[2] for j in jobs))
    print("[try] nothing was added to the library and nothing here is "
          "deployed; delete artifacts/temp when done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
