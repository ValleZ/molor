"""Votes, the Dravnieks check, and soft targets for GS-LF (to-do items 5 and 6).

Run with .venv-pyrfume after scripts/build_source_crosswalks.py.

votes[mol, word] = number of {Arctander, IFRA, Sigma, Flavornet} that apply the
word to the molecule (0-4), plus 1 if the word is the first-listed descriptor
in Good Scents or in the IFRA glossary (one extra vote, however many of the
two say so). Molecules are matched by flat InChIKey only.

A row is COVERED when at least one of the four sources lists the molecule.
2730 of the 4983 training rows are not covered: for them "0 source votes" is
no information, not four sources staying silent, so their labelled cells are
a class of their own (see targets()).

Writes:
  data/percept/gslf_votes.csv              integer votes, GS-LF columns
  data/percept/gslf_soft_targets.csv       variant A
  data/percept/gslf_soft_targets_B.csv     variant B (A / its largest value)
  artifacts/temp/soft_targets_report.json  numbers for the report
  data/percept/crosswalk_audit.csv         each mapping against the GS-LF labels
  artifacts/temp/{votes,src_votes,first_votes,covered}.npy
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_source_crosswalks import GSLF, OUT, load_crosswalk, vocab  # noqa: E402
from soft_target_sources import flat_key, load_all  # noqa: E402

SOURCES = ("arctander", "ifra", "sigma", "flavornet")
PRESENT = 10.0   # panel rating counted as "present"
BANDS = ((0, 0, "0"), (1, 1, "1"), (2, 99, "2+"))


def vote_matrix(res, voc, keys):
    """Source votes, the first-listed vote, and the covered flag per row."""
    col = {w: j for j, w in enumerate(voc)}
    row_of: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        if k:
            row_of.setdefault(k, []).append(i)
    src_votes = np.zeros((len(keys), len(voc)), dtype=np.int8)
    first_vote = np.zeros_like(src_votes)
    covered = np.zeros(len(keys), dtype=bool)
    for s in SOURCES:
        cw = load_crosswalk(s)
        for k, words in res[s]["words"].items():
            for i in row_of.get(k, []):
                covered[i] = True
                hit = {v for w in words for v in cw.get(w, [])}
                for v in hit:
                    src_votes[i, col[v]] += 1
    for s in ("goodscents", "ifra"):
        cw = load_crosswalk(s)
        for k, words in res[s]["first"].items():
            for i in row_of.get(k, []):
                for w in words:
                    for v in cw.get(w, []):
                        first_vote[i, col[v]] = 1
    return src_votes, first_vote, covered


def audit(res, voc, keys, Y):
    """Every crosswalk row against the labels: of the training molecules the
    source word is applied to, how many carry the vocabulary label. A low
    share is not proof of a wrong mapping (the labels are incomplete), but
    near zero on many molecules is (Arctander "gassy": 1 of 36)."""
    col = {w: j for j, w in enumerate(voc)}
    row = {k: i for i, k in enumerate(keys) if k}
    out = []
    for s in SOURCES + ("goodscents",):
        src = res[s]["first"] if s == "goodscents" else res[s]["words"]
        for sw, vws in sorted(load_crosswalk(s).items()):
            rows = [row[k] for k, ws in src.items() if sw in ws and k in row]
            for vw in vws:
                a = int(sum(Y[i, col[vw]] for i in rows))
                out.append((s, sw, vw, len(rows), a,
                            round(a / len(rows), 3) if rows else ""))
    with open(OUT / "crosswalk_audit.csv", "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["source", "source_word", "vocab", "training_molecules",
                     "with_label", "share"])
        wr.writerows(out)
    return out


def dravnieks_cells(voc, Y, keys, src_v, first_v, covered):
    """One row per (Dravnieks molecule in GS-LF, mapped word):
    label, source votes, first-listed vote, covered, panel rating."""
    dra = pd.read_csv(ROOT / "data/percept/dravnieks.csv")
    dra["flat"] = [flat_key(s) for s in dra["smiles"]]
    dra = dra[dra["flat"].notna()]
    cw = load_crosswalk("dravnieks")
    cols = {s: [c for c in dra.columns if c.strip().lower() == s] for s in cw}
    missing = [s for s, c in cols.items() if not c]
    assert not missing, missing
    mean = dra.groupby("flat")[[c[0] for c in cols.values()]].mean()
    row_of = {}
    for i, k in enumerate(keys):
        if k:
            row_of.setdefault(k, i)
    col = {w: j for j, w in enumerate(voc)}
    by_word: dict[str, list[str]] = {}
    for s, vs in cw.items():
        for v in vs:
            by_word.setdefault(v, []).append(cols[s][0])
    out = []
    for k, r in mean.iterrows():
        i = row_of.get(k)
        if i is None:
            continue
        for v, cs in by_word.items():
            j = col[v]
            out.append((int(Y[i, j]), int(src_v[i, j]), int(first_v[i, j]),
                        int(covered[i]), float(np.nanmax([r[c] for c in cs]))))
    n_in = len({k for k in mean.index if k in row_of})
    n_cov = len({k for k in mean.index if k in row_of and covered[row_of[k]]})
    return np.array(out), mean.shape[0], n_in, n_cov


def stat(r):
    return {"cells": int(len(r)),
            "mean_rating": round(float(r.mean()), 1) if len(r) else None,
            "present_pct": round(100 * float((r >= PRESENT).mean()), 0)
            if len(r) else None}


def table(cells, votes):
    """label x votes (0 / 1 / 2+) over the given cells."""
    rows = []
    for lab in (1, 0):
        for lo, hi, name in BANDS:
            m = (cells[:, 0] == lab) & (votes >= lo) & (votes <= hi)
            rows.append({"label": lab, "votes": name, **stat(cells[m, 4])})
    return rows


def r05(pct):
    return round(round(pct / 100 / 0.05) * 0.05, 2)


def targets(cells):
    """Target per class = the panel's present rate in that class, rounded to
    0.05. Classes:
      label 1, covered row:      votes 0 / 1 / 2+   (votes = sources + first)
      label 1, row not covered:  first-listed no / yes, the sources unknown,
                                 so the rate is taken over ALL labelled panel
                                 cells with that first-listed value
      label 0:                   votes 1 / 2+; no vote stays 0
    """
    lab, src, first, cov, rate = (cells[:, i] for i in range(5))
    v = src + first
    cls, val = {}, {}
    for lo, hi, name in BANDS:
        m = (lab == 1) & (cov == 1) & (v >= lo) & (v <= hi)
        cls[f"label1_covered_votes{name}"] = stat(rate[m])
    for f in (0, 1):
        m = (lab == 1) & (first == f)
        cls[f"label1_uncovered_first{f}"] = stat(rate[m])
    for lo, hi, name in BANDS:
        m = (lab == 0) & (v >= lo) & (v <= hi)
        cls[f"label0_votes{name}"] = stat(rate[m])
    for k, s in cls.items():
        val[k] = r05(s["present_pct"])
    val["label0_votes0"] = 0.0
    a = [val[f"label1_covered_votes{n}"] for _, _, n in BANDS]
    b = [val[f"label0_votes{n}"] for _, _, n in BANDS]
    assert a == sorted(a) and b == sorted(b), val
    assert val["label1_uncovered_first0"] <= val["label1_uncovered_first1"], val
    return cls, val


def main() -> int:
    voc = vocab()
    res = load_all()
    df = pd.read_csv(GSLF)
    smi = df["nonStereoSMILES"].astype(str).tolist()
    keys = [flat_key(s) for s in smi]
    Y = df[voc].to_numpy(dtype=float)
    assert not np.isnan(Y).any()
    gk = {k for k in keys if k}

    lib = {flat_key(r["smiles"]) for r in csv.DictReader(
        open(ROOT / "library/input/expanded_materials.csv", encoding="utf-8-sig"))}
    lib.discard(None)

    report = {"step1": {}}
    for s in SOURCES + ("goodscents",):
        ks = set(res[s]["words"])
        report["step1"][s] = {
            "molecules_in_file": res[s]["n_mol"],
            "with_a_word": len(ks),
            "match_gslf_flat_keys": len(ks & gk),
            "match_library_flat_keys": len(ks & lib),
            "distinct_words": len({w for v in res[s]["words"].values() for w in v}),
            "distinct_first_words": len({w for v in res[s]["first"].values() for w in v}),
            "crosswalk_source_words": len(load_crosswalk(s)),
        }
    allk = set().union(*[set(res[s]["words"]) for s in SOURCES])
    report["step1"]["union_of_four"] = {
        "with_a_word": len(allk), "match_gslf_flat_keys": len(allk & gk),
        "match_library_flat_keys": len(allk & lib)}
    report["step1"]["library_keyed"] = len(lib)
    report["step1"]["gslf_flat_keys"] = len(gk)
    report["step1"]["gslf_rows"] = len(keys)

    src_v, first_v, covered = vote_matrix(res, voc, keys)
    votes = src_v + first_v
    cov2 = np.broadcast_to(covered[:, None], Y.shape)

    # where the first-listed vote comes from
    def first_only(s):
        cw = load_crosswalk(s)
        col = {w: j for j, w in enumerate(voc)}
        M = np.zeros_like(first_v)
        row = {k: i for i, k in enumerate(keys) if k}
        for k, words in res[s]["first"].items():
            if k in row:
                for w in words:
                    for v in cw.get(w, []):
                        M[row[k], col[v]] = 1
        return M
    f_gs, f_ifra = first_only("goodscents"), first_only("ifra")

    report["votes"] = {
        "rows_covered_by_a_source": int(covered.sum()),
        "rows_with_any_vote": int((votes.sum(1) > 0).sum()),
        "cells_by_votes_label1": [int(((Y == 1) & (votes == v)).sum()) for v in range(6)],
        "cells_by_votes_label0": [int(((Y == 0) & (votes == v)).sum()) for v in range(6)],
        "label1_cells_in_uncovered_rows": int(((Y == 1) & ~cov2).sum()),
        "label1_covered_no_source_vote": int(((Y == 1) & cov2 & (src_v == 0)).sum()),
        "label1_with_source_vote": int(((Y == 1) & (src_v > 0)).sum()),
        "first_vote_cells": int(first_v.sum()),
        "first_vote_goodscents_only": int(((f_gs == 1) & (f_ifra == 0)).sum()),
        "first_vote_ifra_only": int(((f_gs == 0) & (f_ifra == 1)).sum()),
        "first_vote_both": int(((f_gs == 1) & (f_ifra == 1)).sum()),
        "first_vote_on_label0": int(((Y == 0) & (first_v > 0)).sum()),
        "label1_votes2plus": int(((Y == 1) & (votes >= 2)).sum()),
        "label1_votes2plus_from_one_source_plus_first":
            int(((Y == 1) & (votes >= 2) & (src_v == 1)).sum()),
        "label1_votes1": int(((Y == 1) & (votes == 1)).sum()),
        "label1_votes1_first_only": int(((Y == 1) & (votes == 1) & (src_v == 0)).sum()),
    }
    pd.concat([df[["nonStereoSMILES"]], pd.DataFrame(votes, columns=voc)],
              axis=1).to_csv(OUT / "gslf_votes.csv", index=False)

    au = audit(res, voc, keys, Y)
    report["audit_lowest"] = [list(r) for r in sorted(
        (r for r in au if r[3] >= 5 and r[5] != "" and r[5] < 0.3),
        key=lambda r: r[5])]

    cells, n_dra, n_in, n_cov = dravnieks_cells(voc, Y, keys, src_v, first_v, covered)
    cv = cells[cells[:, 3] == 1]
    report["step4"] = {
        "dravnieks_molecules": int(n_dra), "in_gslf": int(n_in),
        "in_gslf_and_covered": int(n_cov), "cells": int(len(cells)),
        # the 2026-11-09 table: covered molecules, the four sources only
        "like_for_like_covered_source_votes": table(cv, cv[:, 1]),
        # the task's rule on every panel molecule in GS-LF
        "all_votes_with_first": table(cells, cells[:, 1] + cells[:, 2]),
        "label1_by_source_and_first": {
            f"src{'0' if s == 0 else '1plus'}_first{f}": stat(cells[
                (cells[:, 0] == 1) & ((cells[:, 1] > 0) == bool(s))
                & (cells[:, 2] == f), 4]) for s in (0, 1) for f in (0, 1)},
        "label1_uncovered_rows": stat(cells[(cells[:, 0] == 1) & (cells[:, 3] == 0), 4]),
    }
    for name in ("like_for_like_covered_source_votes", "all_votes_with_first"):
        t = report["step4"][name]
        for a in (t[:3], t[3:]):
            assert [r["mean_rating"] for r in a] == sorted(r["mean_rating"] for r in a), \
                f"ordering by votes fails in {name}"

    cls, val = targets(cells)
    report["step5"] = {"classes": cls, "targets_A": val}
    top = max(val.values())
    T = np.zeros_like(Y)
    for lo, hi, name in BANDS:
        band = (votes >= lo) & (votes <= hi)
        T[(Y == 1) & cov2 & band] = val[f"label1_covered_votes{name}"]
        T[(Y == 0) & band] = val[f"label0_votes{name}"]
    for f in (0, 1):
        T[(Y == 1) & ~cov2 & (first_v == f)] = val[f"label1_uncovered_first{f}"]
    assert (T[Y == 1] > 0).all()
    for name, M in (("gslf_soft_targets.csv", T),
                    ("gslf_soft_targets_B.csv", T / top)):
        out = df.copy()
        out[voc] = np.round(M, 4)
        out.to_csv(OUT / name, index=False)
    report["step5"]["targets_B"] = {k: round(v / top, 4) for k, v in val.items()}
    report["step5"]["variant_B_divisor"] = top
    report["step5"]["mean_target_A"] = round(float(T.mean()), 4)
    report["step5"]["mean_target_B"] = round(float((T / top).mean()), 4)
    report["step5"]["mean_label_rate"] = round(float(Y.mean()), 4)

    print(json.dumps(report, indent=1))
    (ROOT / "artifacts/temp/soft_targets_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    np.save(ROOT / "artifacts/temp/votes.npy", votes)
    np.save(ROOT / "artifacts/temp/src_votes.npy", src_v)
    np.save(ROOT / "artifacts/temp/first_votes.npy", first_v)
    np.save(ROOT / "artifacts/temp/covered.npy", covered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
