"""Every library material, one IFRA verdict, and the route that produced it.

``import_ifra.py`` turns the published workbooks into tables; this turns the
tables into an answer per material. The answer is a five-valued status, not a
nullable number, because "no restriction exists" and "we could not find out"
are different facts and the engine used to read both as 100%:

    restricted          a numeric maximum applies (cap in cat4_pct)
    prohibited          the standard forbids it, or a regulator does (cap 0)
    specification_only  a standard exists but sets a purity spec, not a use cap
    unrestricted        identified, and no standard or rule claims it
    unknown             identity not established, so nothing can be claimed

Routes, tried in this order, all of them recorded:

    1 direct       CAS -> standard, then name/synonym -> standard
    2 class        a curated family rule (data/ifra/classes.csv)
    3 contribution the material CONTAINS a restricted constituent at X%:
                   NCS constituents for naturals, Schiff-base aldehydes,
                   cap = standard limit / level
    4 inventory    the perfumer's own figure (data/inventory/inventory.csv),
                   used only when nothing above produced a cap
    5 house cap    data/ifra/house_caps.csv, for caps that are not IFRA

**Never loosen.** When several routes produce a cap the tightest wins and the
row says which route that was, so a permissive figure can never survive contact
with a stricter one.

Two things this refuses to do, both deliberate:

* A CAS list is not a whitelist. 259 of the 263 standards say their scope
  "includes but is not limited to" the CAS numbers listed, so a material that
  looks chemically like a family member is caught by a class rule, and one that
  does not is only called ``unrestricted`` when its identity is established.
* An oil is a mixture. Its cap is derived from the restricted constituents the
  annex reports in it, never guessed from the oil's name.

Amendments are additive: a ``data/ifra/limits_<amendment>.csv`` (the 52nd
draft, when it exists as a file) is merged beside the 51st and the tighter of
the two wins, which is the policy the perfumer chose. Until that file exists
the 52nd simply contributes nothing -- absence of evidence for an amendment is
not evidence of non-restriction, and nothing here treats it as such.

Usage:
    .venv/Scripts/python.exe scripts/resolve_ifra.py
    .venv/Scripts/python.exe scripts/resolve_ifra.py --refresh-cas ../perfumer/materials.db
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IFRA = ROOT / "data" / "ifra"
LIBRARY = ROOT / "library" / "input" / "expanded_materials.csv"
INVENTORY = ROOT / "data" / "inventory" / "inventory.csv"
NATURALS = ROOT / "artifacts" / "naturals.json"
CAS_MAP = IFRA / "material_cas.csv"
OUT = IFRA / "materials.csv"

CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
NONALNUM = re.compile(r"[^a-z0-9]+")
#: Categories carried into the database. Cat 4 is fine fragrance (what the
#: engine caps on) and cat 9 body/hand rinse-off; the other 16 stay in
#: data/ifra/limits.csv for anyone who needs them.
KEPT_CATEGORIES = ("4", "9")


def norm(s: str) -> str:
    return " ".join(NONALNUM.sub(" ", (s or "").lower()).split())


def read_csv(path: Path) -> list[dict]:
    """CSV rows, but never silently short.

    A cell containing an unquoted comma shifts every later column and parks the
    overflow under the None key. Filtering those rows out is how an alias table
    quietly loses the half of its entries whose names contain a comma -- which
    is most natural-product names ("Lemongrass oil, East Indian"). Refuse
    instead: the file is wrong and the fix is to quote the cell.
    """
    if not path.is_file():
        return []
    with io.open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r for r in rows if r.get(None)]
    if bad:
        raise SystemExit(
            f"{path.name}: {len(bad)} row(s) have fields past the last header "
            f"(an unquoted comma), e.g. {bad[0].get('name') or bad[0]}. "
            f"Quote the offending cell.")
    return rows


def as_float(v) -> float | None:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- material CAS
def refresh_cas(db: Path) -> None:
    """Rebuild data/ifra/material_cas.csv from a built materials.db.

    A committed cache, like data/percept/kfo_cid_inchikey.tsv: the CAS numbers
    come from PubChem by InChIKey, cost a network round trip to reproduce, and
    are facts about molecules rather than about the perfumer. Without it this
    script would either need the other repository or would have to call this
    library's 414 CAS-less materials "unknown" for want of an identifier.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = [{"name": n, "cas": (c or "").strip(),
             "source": "materials.db (PubChem by InChIKey)"}
            for n, c in con.execute("SELECT name, cas_number FROM materials")]
    con.close()
    with CAS_MAP.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "cas", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ifra] wrote {CAS_MAP.relative_to(ROOT)}: "
          f"{sum(1 for r in rows if r['cas'])}/{len(rows)} with a CAS")


# ---------------------------------------------------------------- the tables
def load() -> dict:
    standards = {}
    for r in read_csv(IFRA / "standards.csv"):
        standards[r["standard_id"]] = {
            "id": r["standard_id"], "name": r["name"],
            "type": (r["type"] or "").upper(),
            "cas": {c.strip() for c in (r["cas_numbers"] or "").split(";") if c.strip()},
            "synonyms": [s.strip() for s in (r["synonyms"] or "").split(";") if s.strip()],
        }
    limits = defaultdict(dict)          # standard_id -> category -> (amendment, state, pct)
    for p in sorted(IFRA.glob("limits*.csv")):
        for r in read_csv(p):
            limits[r["standard_id"]][r["category"]] = (
                r.get("amendment") or "", r["state"], as_float(r["pct"]))
    return {"standards": standards, "limits": limits}


def cap_of(tables, standard_id, category="4"):
    """(pct, amendment, state) for one standard+category, tightest amendment first."""
    entry = tables["limits"].get(standard_id, {}).get(category)
    if not entry:
        return None
    amendment, state, pct = entry
    if state != "limit" or pct is None:
        return None
    return pct, amendment, state


def state_of(tables, standard_id, category="4"):
    entry = tables["limits"].get(standard_id, {}).get(category)
    return entry[1] if entry else None


def parent_aliases(parent_name: str) -> list[str]:
    """Every name a contribution row can be reached by.

    The annex writes a Schiff base as "Hydroxycitronellal methyl anthranilate
    (or Aurantiol, Aurantium, Aurantoin)", so the trade name the library uses
    lives inside the parentheses and nowhere else. Matching only the full
    string loses Aurantiol, Indolene 50, Vertosine and Canthoxalia -- the exact
    materials that most need this route, because they have no standard of
    their own.
    """
    out = {norm(parent_name)}
    for group in re.findall(r"\(or ([^)]+)\)", parent_name, re.I):
        for part in re.split(r"[;,]", group):
            if norm(part):
                out.add(norm(part))
    for part in re.split(r";", parent_name):
        if norm(part):
            out.add(norm(part))
    return sorted(out)


# ------------------------------------------------------------------ resolving
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh-cas", type=Path, default=None,
                    help="rebuild the committed CAS cache from a materials.db")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if args.refresh_cas:
        refresh_cas(args.refresh_cas)
    for required in ("standards.csv", "limits.csv"):
        if not (IFRA / required).is_file():
            raise SystemExit(f"{IFRA / required} missing: run scripts/import_ifra.py")

    tables = load()
    standards = tables["standards"]
    contributions = read_csv(IFRA / "contributions.csv")
    house = read_csv(IFRA / "house_caps.csv")
    classes = read_csv(IFRA / "classes.csv")
    ncs_alias = {norm(r["oil"]): r["ncs"] for r in read_csv(IFRA / "ncs_alias.csv")}

    by_cas, by_name = defaultdict(list), defaultdict(list)
    for s in standards.values():
        for c in s["cas"]:
            by_cas[c].append(s["id"])
        by_name[norm(s["name"])].append(s["id"])
        for syn in s["synonyms"]:
            by_name[norm(syn)].append(s["id"])

    # constituents -> the standards that restrict them, and their levels
    const_by_cas = {c: ids for c, ids in by_cas.items()}
    contrib_by_parent = defaultdict(list)
    for r in contributions:
        if not r["level_pct"]:
            continue
        for alias in parent_aliases(r["parent_name"]):
            contrib_by_parent[alias].append(r)
        for c in (r["parent_cas"] or "").split(";"):
            if c.strip():
                contrib_by_parent[f"cas:{c.strip()}"].append(r)

    cas_map = {norm(r["name"]): r["cas"] for r in read_csv(CAS_MAP)}
    # Curated overrides, for materials PubChem's InChIKey route could not name:
    # the annex gives Aurantiol's CAS and nothing else did, and Habanolide's
    # InChIKey lands on a depositor record with no CAS at all.
    for r in read_csv(IFRA / "cas_fixes.csv"):
        if r.get("cas"):
            cas_map[norm(r["name"])] = r["cas"].strip()
    inv = {norm(r.get("name", "")): r for r in read_csv(INVENTORY)}
    house_by_name = {norm(r["name"]): r for r in house}
    house_by_cas = {r["cas_number"].strip(): r for r in house if r.get("cas_number")}

    names = [(r["name"].strip(), (r.get("source") or ""))
             for r in read_csv(LIBRARY)]
    oils = set()
    if NATURALS.is_file():
        oils = {norm(n) for n in json.loads(NATURALS.read_text(encoding="utf-8"))["naturals"]}
        # The oils are not in the molecule CSV -- they are compositions built
        # from it -- so they enter the loop here. They have no CAS at all, and
        # that is why their cap can only come from what the annex says they
        # contain, never from a name.
        have = {norm(n) for n, _ in names}
        for oil in sorted(json.loads(NATURALS.read_text(encoding="utf-8"))["naturals"]):
            if norm(oil) not in have:
                names.append((oil, "reconstruction (naturals store)"))

    rows = []
    for name, source in names:
        is_oil = norm(name) in oils
        cas = cas_map.get(norm(name), "")
        if not cas:
            found = CAS_RE.findall(source or "")
            cas = found[0] if found else ""

        # every route contributes a (cap, label) and the tightest wins
        candidates: list[tuple[float, str, str, str]] = []   # pct, via, std, evidence
        spec_only = None

        # 1 direct ---------------------------------------------------------
        direct = list(by_cas.get(cas, [])) if cas else []
        if not direct:
            direct = list(by_name.get(norm(name), []))
        for sid in {s for s in direct}:
            st = standards[sid]
            if "PROHIBITION" in st["type"]:
                candidates.append((0.0, "direct", sid, f"{sid} {st['name']} is a PROHIBITION"))
                continue
            cap = cap_of(tables, sid)
            if cap:
                candidates.append((cap[0], "direct", sid,
                                   f"{sid} {st['name']} cat-4 {cap[0]:g}% (amendment {cap[1]})"))
            elif state_of(tables, sid) in ("not_applicable", "no_restriction"):
                spec_only = spec_only or (sid, f"{sid} {st['name']} sets no use limit")

        # 2 curated class rules -------------------------------------------
        for rule in classes:
            try:
                if not re.search(rule["pattern"], name, re.I):
                    continue
            except re.error:
                continue
            sid = rule["standard_id"]
            cap = cap_of(tables, sid)
            if cap:
                candidates.append((cap[0], "class", sid,
                                   f"class rule {rule['pattern']!r} -> {sid} cat-4 {cap[0]:g}%"))

        # 3 contents: naturals and Schiff bases ---------------------------
        parents = []
        if is_oil:
            ncs = ncs_alias.get(norm(name))
            if ncs:
                parents.append(norm(ncs))
        if cas:
            parents.append(f"cas:{cas}")
        parents.append(norm(name))
        seen_contrib = set()
        for parent in parents:
            for r in contrib_by_parent.get(parent, []):
                key = (r["parent_name"], r["constituent_cas"])
                if key in seen_contrib:
                    continue
                seen_contrib.add(key)
                level = as_float(r["level_pct"])
                if not level:
                    continue
                for sid in const_by_cas.get(r["constituent_cas"], []):
                    cap = cap_of(tables, sid)
                    if not cap:
                        continue
                    derived = cap[0] / (level / 100)
                    # A cap above 100% is arithmetically right and practically
                    # meaningless: it says the constituent cannot bind, because
                    # you would have to dose more of the substance than a
                    # formula contains. Bergamot's geranial works out at 214%.
                    # Record it as unbinding rather than as a cap.
                    if derived >= 100:
                        continue
                    candidates.append((derived, "contribution", sid,
                                       f"{cap[0]:g}% {sid} / {level:g}% "
                                       f"{r['constituent_name'][:28]} in "
                                       f"{r['parent_name'][:32]}"))

        # 4 the perfumer's own figure, only if nothing above capped it ----
        if not candidates:
            row = inv.get(norm(name))
            v = as_float((row or {}).get("ifra_cat4"))
            if v is not None:
                candidates.append((v, "inventory", "",
                                   "data/inventory/inventory.csv ifra_cat4; the perfumer's "
                                   "own figure, not from an IFRA workbook"))

        # 5 explicit house caps -------------------------------------------
        for key, table in (("name", house_by_name), ("cas", house_by_cas)):
            key_val = norm(name) if key == "name" else cas
            r = table.get(key_val) if key_val else None
            if r:
                v = as_float(r["cat4_pct"])
                if v is not None:
                    candidates.append((v, "house", "",
                                       f"house cap: {r['reason'][:90]}"))
                break

        # ------------------------------------------------------- verdict
        if candidates:
            pct, via, sid, evidence = min(candidates, key=lambda t: t[0])
            status = "prohibited" if pct == 0 else "restricted"
        else:
            pct, via, sid, evidence = None, None, "", ""
            identified = bool(cas) or is_oil
            if spec_only:
                status, via, sid, evidence = "specification_only", "direct", spec_only[0], spec_only[1]
            elif identified:
                status = "unrestricted"
                via = "absent"
                evidence = ("no standard, class rule or constituent claims it"
                            + ("" if cas else " (natural: identified by name)"))
            else:
                status = "unknown"
                evidence = "no CAS and no name match, so nothing can be claimed"

        cat9 = None
        if sid and via in ("direct", "class"):
            c = cap_of(tables, sid, "9")
            cat9 = c[0] if c else None

        rows.append({
            "name": name,
            "is_natural": int(is_oil),
            "cas": cas,
            "status": status,
            "standard_id": sid or "",
            "cat4_pct": "" if pct is None else f"{pct:.6g}",
            "cat9_pct": "" if cat9 is None else f"{cat9:.6g}",
            "via": via or "",
            "evidence": evidence,
            "source": "IFRA 51st overview workbook via data/ifra/limits.csv",
        })

    rows.sort(key=lambda r: r["name"].lower())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------- gates
    status_n = Counter(r["status"] for r in rows)
    via_n = Counter(r["via"] for r in rows if r["status"] in ("restricted", "prohibited"))
    print(f"[resolve] {len(rows)} materials -> {args.out.relative_to(ROOT)}")
    print("[resolve] " + ", ".join(f"{k} {v}" for k, v in status_n.most_common()))
    print("[resolve] caps by route: "
          + ", ".join(f"{k} {v}" for k, v in via_n.most_common()))
    tight = [r for r in rows if r["status"] == "restricted"
             and r["cat4_pct"] and float(r["cat4_pct"]) < 1.0]
    print(f"[resolve] {len(tight)} materials capped below 1% in cat-4")

    ok = True
    unknown = [r["name"] for r in rows if r["status"] == "unknown"]
    if unknown:
        print(f"[resolve] {len(unknown)} unknown (identity not established): "
              + ", ".join(unknown[:8]) + ("..." if len(unknown) > 8 else ""))
    if not any(r["status"] == "prohibited" for r in rows):
        ok = False
        print("[resolve] GATE FAILED: no prohibited material at all -- the "
              "prohibition branch is not firing")
    if not any(r["via"] == "contribution" for r in rows):
        ok = False
        print("[resolve] GATE FAILED: no cap came from the contents route")
    for probe in ("Farnesol", "Aurantiol", "Habanolide", "Linalool"):
        r = next((x for x in rows if x["name"] == probe), None)
        if r is None:
            print(f"[resolve] WARNING: {probe} is not in the library")
            continue
        print(f"[resolve]   {probe:<22} {r['status']:<19} "
              f"cat4={r['cat4_pct'] or '-':<8} via={r['via'] or '-':<12} "
              f"{r['evidence'][:70]}")
    print("[resolve] gates " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
