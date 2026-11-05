"""Fill the missing detection thresholds, and say where every value came from.

310 of the library's 435 materials have no threshold figure, and the dose
conversion skips a material that has none: it cannot turn a share of odour
activity into a mass without one. So the materials with no number were not
merely dosed badly, they were excluded from being proposed at all. A figure
that is a factor of three out is more use than that, which is the argument
for imputing the rest.

Imputation is k-nearest-neighbour regression on log10(threshold), with the
percept model's own 138-dimensional space as the metric. Checked leave-one-out
on the 125 materials that have a figure:

    k=1    median error 0.70 log units   (5.0x)   66% within 10x
    k=3    median error 0.63             (4.2x)   71%
    k=5    median error 0.48             (3.0x)   77%   <- used
    k=10   median error 0.58             (3.8x)   74%
    null, predict the median for all:    0.70     (5.0x)

So k=5 is better than nothing and not much better than that. Two things about
that number are worth keeping in view.

It is error against the compilation, not against truth. The 125 figures were
assembled from published and web sources with some converted from air-phase
values, and they track the 28 really-measured Abraham thresholds at Spearman
+0.39 where a perfumer's own working use levels manage +0.76. An imputation
fitted on them reproduces them faithfully, bias included, and cannot be better
than its source.

And the spread being imputed is enormous: the targets run over eight orders of
magnitude. A 3x median error against a 1.29-log spread is informative, but a
dose derived from it is an ordering, not a weighing.

Hence the source label. Every value carries one, and anything downstream that
prints a weighable mass has to be able to say which tier it rests on.

    compiled   a figure from the inventory's own column         125
    imputed    k=5 neighbours in percept space                  310
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Neighbours used, chosen by the leave-one-out check in the docstring.
K = 5


def impute(X: np.ndarray, thr: list, names=None) -> tuple[list, list, dict]:
    """(values, sources, report) for every row of X.

    X is the percept matrix, one row per material. thr holds a figure or None
    per material, in the same order. Values are returned in the same units as
    the ones given.
    """
    thr = list(thr)
    have = [i for i, t in enumerate(thr) if t and t > 0]
    if len(have) < K + 1:
        return thr, ["compiled" if t else None for t in thr], {
            "imputed": 0, "compiled": len(have),
            "note": "too few known thresholds to impute from"}

    P = X[have]
    y = np.array([np.log10(thr[i]) for i in have], float)
    anchor_names = [names[i] for i in have] if names else None

    out = list(thr)
    src = ["compiled" if (t and t > 0) else None for t in thr]
    n_imp = 0
    for i, t in enumerate(thr):
        if t and t > 0:
            continue
        d = np.linalg.norm(P - X[i], axis=1)
        # Ties break on the anchor's NAME, not its row index. Without that the
        # answer depends on the order the caller happened to pass: Isomenthone
        # and Menthone have bit-identical percept vectors, so they sit at
        # distance exactly 0 from the same neighbours, and the atlas and the
        # simplifier -- which list materials in different orders -- imputed
        # different thresholds for three materials from identical inputs.
        if anchor_names is not None:
            near = sorted(range(len(d)),
                          key=lambda j: (d[j], anchor_names[j]))[:K]
        else:
            near = np.argsort(d)[:K]
        out[i] = float(10.0 ** y[near].mean())
        src[i] = "imputed"
        n_imp += 1

    # The leave-one-out error, recomputed here rather than quoted, so the
    # figure in the report is the one this data actually gives.
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    pred = np.array([y[np.argsort(D[j])[:K]].mean() for j in range(len(y))])
    err = np.abs(pred - y)
    report = {
        "compiled": len(have),
        "imputed": n_imp,
        "k": K,
        "loo_median_log10": round(float(np.median(err)), 3),
        "loo_median_factor": round(float(10 ** np.median(err)), 2),
        "loo_within_decade": round(float((err <= 1).mean()), 3),
        "target_sd_log10": round(float(y.std()), 3),
    }
    return out, src, report


def describe(report: dict) -> str:
    if not report.get("imputed"):
        return f"{report.get('compiled', 0)} thresholds, none imputed"
    return (f"{report['compiled']} compiled thresholds, {report['imputed']} "
            f"imputed from {report['k']} neighbours in percept space "
            f"(leave-one-out median error {report['loo_median_factor']}x, "
            f"{100 * report['loo_within_decade']:.0f}% within a decade)")


def assemble(names, X, strength=None, not_anchors=(), oils=()):
    """The whole threshold column, one place, so two pages cannot disagree.

    The atlas and the simplifier had grown separate notions of "threshold".
    The simplifier carried ppb in three provenance tiers -- compiled from the
    inventory, published in the KFO-Atlas, k-nearest-neighbour imputed -- while
    the atlas carried log(1/ODT) from Abraham's air-phase measurements for 28
    materials and withheld the rest. Two different quantities under one word,
    on two pages built from one library. This is the ppb column, and both now
    read it from here.

    Order is not cosmetic. KFO is applied AFTER imputation rather than as an
    anchor for it: feeding its values into the k-nearest-neighbour step moved
    the leave-one-out error from 3.0x to 4.76x, because interpolation is only
    as good as the points it interpolates from and KFO carries 2.4x of its
    own. And a compiled figure is never replaced, because on the compounds at
    or below 1 ppb KFO misses by 3.8x and misses upward -- 250x high on
    trans-4-decenal -- which is the direction that over-doses.

    Returns (values, sources, extras, report). ``extras`` holds the KFO row
    count and spread per material, so a median of 696 disagreeing papers
    cannot present itself as a measurement.
    """
    import csv as _csv
    import io as _io

    from parse_formulas import library_name

    # Shelf names resolve to library names by the alias table and the head-word
    # match a formula gets (parse_formulas.library_name), not by string
    # equality: the inventory says "Orange Sweet EO" and the library says
    # "Orange EO (sweet, Florida)", and an exact join dropped the compiled
    # figure -- and the use level -- for 35 shelf names without a word.
    library = set(names)
    inv, kfo = {}, {}
    ip = ROOT / "data" / "inventory" / "inventory.csv"
    if ip.is_file():
        for r in _csv.DictReader(_io.open(ip, encoding="utf-8-sig")):
            v = (r.get("water_threshold_ppb") or "").strip()
            if v:
                try:
                    # Keyed by the library name the shelf name refers to; without
                    # that, every oil and every grade-qualified name falls through
                    # to imputation while a compiled figure sits in the inventory.
                    inv[library_name(r.get("name") or "", library, oils)] = float(v)
                except ValueError:
                    pass
    kp = ROOT / "data" / "percept" / "kfo_thresholds.csv"
    if kp.is_file():
        for r in _csv.DictReader(_io.open(kp, encoding="utf-8-sig")):
            try:
                kfo[r["name"].strip()] = (float(r["ppb"]), int(r["n"]),
                                          float(r["min_ppb"]),
                                          float(r["max_ppb"]),
                                          r.get("wide") == "1")
            except (ValueError, KeyError):
                pass

    # A mixture is not an anchor. Tangerine EO carries a recorded threshold,
    # and because the simplifier lists oils while the atlas does not, it was a
    # k-nearest-neighbour anchor on one page and absent on the other -- three
    # molecules then got different imputed thresholds from identical inputs.
    # An oil's threshold is a property of the blend in any case, so using it
    # to interpolate a single molecule's is mixing categories. Excluded names
    # keep their own figure and stop anchoring anyone else's.
    skip = set(not_anchors)
    raw = [None if n in skip else inv.get(n) for n in names]
    vals, src, report = impute(X, raw, names=list(names))
    for i, n in enumerate(names):
        if n in skip and inv.get(n):
            vals[i] = inv[n]
            src[i] = "compiled"
    report["compiled"] = sum(1 for s_ in src if s_ == "compiled")
    extras = {}
    n_kfo = 0
    for i, n in enumerate(names):
        if src[i] == "compiled":
            continue
        k = kfo.get(n)
        if not k:
            continue
        vals[i] = k[0]
        src[i] = "kfo"
        extras[n] = {"n": k[1], "lo": k[2], "hi": k[3], "wide": k[4]}
        n_kfo += 1
    report["kfo"] = n_kfo
    report["imputed"] = sum(1 for s_ in src if s_ == "imputed")
    # The bench correction is a statement about the material, not about the
    # figure, so it applies on top of whichever tier supplied it.
    if strength:
        for i, n in enumerate(names):
            st = strength.get(n)
            if st and vals[i]:
                vals[i] = vals[i] / st["mult"]
    return vals, src, extras, report
