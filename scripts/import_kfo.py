"""Odour thresholds from the KFO-Atlas, as a tier between compiled and imputed.

The KFO-Atlas (Leibniz-LSB, 2282 aroma profiles / 896 key food odorants)
publishes an odour threshold per compound alongside the concentrations it was
found at, keyed by PubChem CID. 179 of this library's 396 molecules match it,
and 125 of those currently carry a k-nearest-neighbour *guess* rather than a
published figure.

**Whether these are water thresholds was tested rather than assumed**, because
the first read of this file assumed they were food-matrix values needing
conversion and that was wrong. Three checks:

*The variation is not matrix-driven.* Of 331 compounds appearing across three
or more food categories, only 50 have one value per category -- what a matrix
effect looks like -- while 281 vary *within* a single category. All wines,
different numbers. That is different source papers citing different literature
thresholds, not the matrix acting on the compound. 389 of 890 compounds use a
single value across every food they appear in.

*There is no systematic offset, so nothing to convert.* Against this
inventory's own compiled water thresholds the median ratio is 1.13x, and the
ratio tracks cLogP at only Spearman +0.285. A real fat-matrix effect would be
strongly logP-dependent and biased high; this is neither.

*And the agreement is as good as the column it would join.* Median absolute
error against compiled is 2.4x with 81% inside a decade, where this project's
own imputation manages 3.0x and 77%. So the KFO median is ranked ABOVE
imputed, not below compiled -- it is the same quantity at comparable quality.

What the tests did NOT clear, and why compiled still wins outright:

    Damascenone Total   compiled 0.002 ppb   KFO 0.05    25x high
    Ionone Beta         compiled 0.007       KFO 0.09    13x high
    Trans-4-Decenal     compiled 0.1         KFO 25     250x high

On the potent end KFO's error rises to 3.8x and it misses *upward*, because
food-chemistry papers report thresholds near the concentrations compounds
occur at rather than at their detection limit. A threshold 250x too high means
dosing 250x too much of a material detectable at a tenth of a part per
billion. So a compiled figure is never overridden, and every KFO value travels
with its row count and full spread so the page can refuse to look confident.

    python scripts/import_kfo.py <path to kfo_2282.xlsx>
    python scripts/import_kfo.py <path> --no-fetch   # use the cached CID map
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "percept" / "kfo_thresholds.csv"
CACHE = ROOT / "data" / "percept" / "kfo_cid_inchikey.tsv"

#: Aqueous and mass units, to parts per billion. Only units whose conversion
#: is exact for a dilute aqueous solution are here; everything else is
#: reported and skipped rather than approximated.
TO_PPB = {
    "μg/kg": 1, "ug/kg": 1, "μg/l": 1, "ug/l": 1, "ng/g": 1, "ng/ml": 1,
    "ppb": 1, "μg/g": 1000, "ug/g": 1000, "μg/ml": 1000, "ug/ml": 1000,
    "mg/kg": 1000, "mg/l": 1000, "ppm": 1000, "mg/g": 1e6,
    "ng/kg": 0.001, "ng/l": 0.001,
}
#: Air thresholds are a different quantity and are dropped, not converted --
#: the same rule the boiling-point merge applies to vacuum distillations.
AIR = {"mg/m³", "mg/m3", "μg/m³", "ug/m3", "μg/m3", "ng/l air"}
#: A spread this wide across the published values makes the median a label
#: rather than a measurement, and the page says so.
WIDE_SPREAD = 100.0


def fetch_keys(cids, use_cache=True, fetch=True):
    """PubChem CID -> InChIKey, cached in the repository.

    Cached because the import should be reproducible without network, and
    because 896 identifiers are not something to re-ask for on every run.
    """
    have = {}
    if use_cache and CACHE.is_file():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                c, k = line.split("\t", 1)
                have[c.strip()] = k.strip()
    missing = [c for c in cids if c not in have]
    if missing and fetch:
        print(f"[kfo] resolving {len(missing)} CIDs at PubChem")
        for i in range(0, len(missing), 50):
            chunk = missing[i:i + 50]
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                   + ",".join(chunk) + "/property/InChIKey/CSV")
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    txt = r.read().decode()
            except Exception as exc:  # noqa: BLE001
                print(f"  chunk at {i} failed: {str(exc)[:60]}")
                continue
            for row in csv.DictReader(io.StringIO(txt)):
                c = str(row.get("CID", "")).strip()
                k = (row.get("InChIKey") or "").strip()
                if c and k:
                    have[c] = k
            time.sleep(0.25)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text("".join(f"{c}\t{k}\n" for c, k in sorted(have.items())),
                         encoding="utf-8")
        print(f"[kfo] cache now holds {len(have)} identifiers")
    elif missing:
        print(f"[kfo] {len(missing)} CIDs are not cached and --no-fetch is set")
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="KFO-Atlas aroma-profiles workbook")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import openpyxl
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    # Row 0 is a provenance sentence from the publishers, row 1 the header.
    hdr = [str(c) if c else "" for c in rows[1]]
    col = {h: i for i, h in enumerate(hdr)}
    need = ["PubChem CIDs", "Compound names", "Odor thresholds"]
    miss = [n for n in need if n not in col]
    if miss:
        raise SystemExit(f"columns not found: {miss}; header is {hdr}")
    # The published header misspells this one.
    ucol = col.get("Oodr threshold units", col.get("Odor threshold units"))

    vals = defaultdict(list)
    names = {}
    air = 0
    unknown = Counter()
    for r in rows[2:]:
        if not r or not r[col["PubChem CIDs"]]:
            continue
        raw = r[col["Odor thresholds"]]
        if raw is None:
            continue
        cid = str(r[col["PubChem CIDs"]]).strip()
        if not cid.isdigit():
            continue
        unit = str(r[ucol] or "").strip().lower() if ucol is not None else ""
        if unit in AIR:
            air += 1
            continue
        f = TO_PPB.get(unit)
        if f is None:
            unknown[unit] += 1
            continue
        try:
            v = float(raw) * f
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        vals[cid].append(v)
        names.setdefault(cid, str(r[col["Compound names"]] or "").strip())

    print(f"[kfo] {len(vals)} compounds with an aqueous threshold")
    print(f"[kfo] dropped {air} air-phase rows (a different quantity) and "
          f"{sum(unknown.values())} rows in units not convertible exactly")
    if unknown:
        print(f"       e.g. {', '.join(f'{u} x{c}' for u, c in unknown.most_common(5))}")

    keys = fetch_keys(sorted(vals), fetch=not args.no_fetch)
    lib = {}
    for r in csv.DictReader(io.open(
            ROOT / "library/input/expanded_materials.csv", encoding="utf-8")):
        m = Chem.MolFromSmiles((r.get("smiles") or "").strip())
        if m is not None:
            lib.setdefault(Chem.MolToInchiKey(m), r["name"].strip())
    k2c = {}
    for c, k in keys.items():
        k2c.setdefault(k, c)

    pay = json.loads((ROOT / "artifacts/simplify_data.json")
                     .read_text(encoding="utf-8"))
    M = {m["n"]: m for m in pay["materials"]}

    out, cross = [], []
    for k, name in sorted(lib.items(), key=lambda kv: kv[1]):
        cid = k2c.get(k)
        if cid is None or cid not in vals:
            continue
        v = np.asarray(vals[cid], float)
        med = float(np.median(v))
        rec = {"name": name, "inchikey": k, "cid": cid,
               "kfo_name": names.get(cid, ""),
               "ppb": round(med, 6), "n": int(v.size),
               "min_ppb": round(float(v.min()), 6),
               "max_ppb": round(float(v.max()), 6),
               "spread": round(float(v.max() / max(v.min(), 1e-12)), 1),
               "wide": int(v.max() / max(v.min(), 1e-12) > WIDE_SPREAD)}
        out.append(rec)
        cur = M.get(name, {})
        if cur.get("thrSrc") == "compiled" and cur.get("thr"):
            cross.append((name, cur["thr"], med))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    wide = sum(r["wide"] for r in out)
    print(f"\n[kfo] wrote {OUT.relative_to(ROOT)}: {len(out)} library "
          f"materials, {wide} of them with a spread over {WIDE_SPREAD:.0f}x")
    bysrc = Counter(M.get(r["name"], {}).get("thrSrc") for r in out)
    print(f"[kfo] they currently hold: "
          + ", ".join(f"{v} {k}" for k, v in bysrc.items()))

    if cross:
        e = np.array([abs(np.log10(m / w)) for _, w, m in cross])
        from scipy.stats import spearmanr
        cw = [w for _, w, _ in cross]
        cm = [m for _, _, m in cross]
        print(f"\n[kfo] against the {len(cross)} compiled water thresholds:")
        print(f"       Spearman {spearmanr(cw, cm).statistic:+.3f}, "
              f"median error {10 ** np.median(e):.1f}x, "
              f"within a decade {100 * float(np.mean(e < 1)):.0f}%")
        pot = [(n, w, m) for n, w, m in cross if w <= 1.0]
        if pot:
            ep = np.array([abs(np.log10(m / w)) for _, w, m in pot])
            print(f"       on the {len(pot)} compounds at or below 1 ppb, "
                  f"median error {10 ** np.median(ep):.1f}x -- which is why a "
                  f"compiled figure is never overridden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
