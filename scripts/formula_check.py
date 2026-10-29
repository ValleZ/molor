"""Compare a working perfumery formula against a target volatile profile.

The perfumer reports that their gooseberry accord does not smell like
gooseberry. That is a labelled negative, which is more informative than another
positive: if the model also puts the formula far from the target, the model and
the nose agree; if the model puts it close, the model is wrong in a way we can
point at.

Dilutions are handled explicitly. "Blackcurrant Mercaptan 0.1%" at 30 parts is
0.03 parts of material, not 30, and treating it as 30 would hand a trace
component the same weight as the base. Note the opposite failure too: weighting
by amount assumes receptor activation scales with how much you add, and the
model has no potency axis, so a thiol detectable at parts per billion is
underweighted at its true dose. Both weightings are therefore reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.store import Library  # noqa: E402

EPS = 1e-6

TARGET = [
    ("(Z)-3-Hexenal", 1.0), ("(E)-2-Hexenal", 1.0), ("Methyl Butyrate", 1.0),
    ("Ethyl Acetate, 10% DPG", 1.0), ("Ethyl Butyrate", 1.0),
    ("Hexenol-3-CIS", 1.0), ("Methyl Acetate", 1.0), ("Heptan-2-one", 1.0),
]

# (library name, parts in the formula, dilution of the material as supplied)
FORMULA = [
    ("Hexenyl-3-CIS Acetate", 300, 1.0),
    ("Paradisamide", 200, 1.0),
    ("Ethyl Linalool", 200, 1.0),
    ("Cassifix", 80, 1.0),
    ("Verdox HC", 70, 1.0),
    ("Fructone", 50, 1.0),
    ("Hexenol-3-CIS", 35, 1.0),
    ("Blackcurrant mercaptan, 0.1% DPG", 30, 0.001),
    ("Ambrettolide", 25, 1.0),
    ("Ethyl 2-methyl butyrate", 15, 0.10),
]

BLEND_CENTER = [
    ("Cis-3-Hexenyl Isobutyrate", 1.0), ("Ethyl Formate", 1.0),
    ("Aldehyde C-7", 1.0), ("Eucalyptol", 1.0),
    ("Methyl Propionate", 1.0), ("Nympheal", 0.5),
]
BLEND_ZSCORE = [
    ("Ethyl Formate", 1.0), ("Aldehyde C-7", 1.0), ("Methyl Propionate", 0.85),
    ("Blackcurrant mercaptan, 0.1% DPG", 0.5), ("Nympheal", 0.3),
    ("Helvetolide", 0.15),
]


def noisy_or(rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, 1)[:, None]
    return 1.0 - np.prod(1.0 - w * np.clip(rows, 0, 1 - EPS), axis=0)


def cos_centered(a, b, mu):
    ca, cb = a - mu, b - mu
    return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))


def main() -> int:
    entries, _, scores = Library(name="library_mpnn_expanded").load()
    names = [e.name for e in entries]
    idx = {n: i for i, n in enumerate(names)}
    rmu = scores.mean(axis=0)

    pj = json.loads((ROOT / "artifacts" / "percepts_expanded.json").read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}
    pmu = np.stack(list(P.values())).mean(axis=0)

    missing = [n for n, *_ in TARGET + FORMULA + BLEND_CENTER + BLEND_ZSCORE
               if n not in idx]
    if missing:
        raise SystemExit(f"not in library: {sorted(set(missing))}")

    def profiles(spec, weights):
        rows_r = np.stack([scores[idx[n]] for n, *_ in spec])
        rows_p = np.stack([P[n] for n, *_ in spec])
        w = np.asarray(weights, float)
        w = w / w.max()
        return noisy_or(rows_r, w), noisy_or(rows_p, w)

    tgt_r, tgt_p = profiles(TARGET, [w for _, w in TARGET])

    amt = np.array([p * d for _, p, d in FORMULA])
    sets = {
        "your formula (by amount)": (FORMULA, amt),
        "your formula (equal weights)": (FORMULA, np.ones(len(FORMULA))),
        "my blend / centred metric": (BLEND_CENTER, [w for _, w in BLEND_CENTER]),
        "my blend / z-scored metric": (BLEND_ZSCORE, [w for _, w in BLEND_ZSCORE]),
    }

    print("DISTANCE TO THE GOOSEBERRY VOLATILE TARGET")
    print(f"  {'':30s} {'receptor cos':>13s} {'descriptor cos':>15s}")
    results = {}
    for label, (spec, w) in sets.items():
        pr, pp = profiles(spec, w)
        rc, pc = cos_centered(pr, tgt_r, rmu), cos_centered(pp, tgt_p, pmu)
        results[label] = (pr, pp)
        print(f"  {label:30s} {rc:+13.4f} {pc:+15.4f}")

    print("\nPREDICTED ODOUR: target vs your formula (by amount)")
    yp = results["your formula (by amount)"][1]
    print(f"  {'descriptor':16s} {'target':>8s} {'yours':>8s} {'gap':>8s}")
    for j in np.argsort(-tgt_p)[:14]:
        print(f"  {desc[j]:16s} {tgt_p[j]:8.2f} {yp[j]:8.2f} {yp[j]-tgt_p[j]:+8.2f}")
    extra = [j for j in np.argsort(-yp)[:14] if tgt_p[j] < 0.35]
    if extra:
        print("  present in yours, absent from the target:")
        for j in extra[:8]:
            print(f"    {desc[j]:16s} {yp[j]:.2f}  (target {tgt_p[j]:.2f})")

    print("\nWHAT EACH COMPONENT OF YOUR FORMULA CONTRIBUTES")
    print(f"  {'material':34s} {'parts':>7s} {'neat':>8s} {'top predicted odour'}")
    for n, p, d in FORMULA:
        top = np.argsort(-P[n])[:3]
        print(f"  {n[:33]:34s} {p:7.0f} {p*d:8.2f}  "
              + ", ".join(f"{desc[t]}({P[n][t]:.2f})" for t in top))

    print("\nTARGET COMPONENTS, FOR COMPARISON")
    for n, _ in TARGET:
        top = np.argsort(-P[n])[:3]
        print(f"  {n[:33]:34s} {'':16s}" + ", ".join(f"{desc[t]}({P[n][t]:.2f})" for t in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
