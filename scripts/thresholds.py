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

import numpy as np

#: Neighbours used, chosen by the leave-one-out check in the docstring.
K = 5


def impute(X: np.ndarray, thr: list) -> tuple[list, list, dict]:
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

    out = list(thr)
    src = ["compiled" if (t and t > 0) else None for t in thr]
    n_imp = 0
    for i, t in enumerate(thr):
        if t and t > 0:
            continue
        d = np.linalg.norm(P - X[i], axis=1)
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
