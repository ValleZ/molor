"""Turn a pasted modifier list into a formula.

The other half of scripts/make_modifier_prompt.py. Paste that prompt and a
description into any chat, save what comes back, and run it through here.

Why the objective is weighted. Matching a described smell is not the same
problem as matching a formula, and using the same objective for both produced
a blend that satisfied the distance metric with none of the subject in it: a
coffee-and-sea description came back as a woody-aromatic accord, 0.332 from
the target against a just-noticeable yardstick of 0.508, with the coffee still
two standard deviations short. Uniform L2 over 138 descriptors trades one
large shortfall against many small ones, which is right for "match this
formula" and wrong for "make this smell". So only the descriptors the
description actually named are scored, each by the intensity it was given.

Why the isolated materials are in. The simplifier excludes them by default,
because a material with no near-equivalents reads as itself and sticks out of
a blend. That is the correct default when the job is to substitute. It is
exactly backwards here: a material is isolated *because* it is the only thing
on the shelf that smells of that, so the carriers of any nameable note are the
isolated ones. 2-acetyl pyrazine is the roast, and nothing else is.

    python scripts/modifiers_to_formula.py my_modifiers.txt --k 8
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
#: The exponent on the threshold, as used in the page and the CLI. A straight
#: activity-to-mass conversion overshot on the bench.
DOSE_EXP = 0.6


def parse_modifiers(text: str, desc: list[str]):
    """(modifiers, unavailable, unknown) from a pasted block."""
    low = {d.lower(): d for d in desc}
    mods, unavailable, unknown = [], [], []
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        if not line:
            continue
        if line.upper().startswith("MISSING:"):
            rest = line.split(":", 1)[1].strip()
            if rest and rest.lower() != "none":
                unavailable = [w.strip() for w in rest.split(",") if w.strip()]
            continue
        # tab, pipe, or two-or-more spaces, so a hand-edited list still reads
        parts = [p.strip() for p in re.split(r"\t|\s*\|\s*|\s{2,}", line)
                 if p.strip()]
        if len(parts) < 2:
            continue
        name = parts[0].lower()
        try:
            val = float(parts[1].rstrip("%"))
        except ValueError:
            continue
        if val > 1.0:
            val /= 100.0
        src = parts[2] if len(parts) > 2 else ""
        if name in low:
            mods.append((low[name], max(0.0, min(1.0, val)), src))
        else:
            unknown.append(parts[0])
    return mods, unavailable, unknown


def fit(A, b, iters=1500):
    """min ||A.T w - b|| over w >= 0, sum w = 1."""
    k = A.shape[0]
    G, c = A @ A.T, A @ b
    w = np.full(k, 1.0 / k)
    try:
        u = np.linalg.lstsq(A.T, b, rcond=None)[0]
        if u.min() >= 0 and u.sum() > 0:
            w = u / u.sum()
    except np.linalg.LinAlgError:
        pass
    L = max(np.linalg.eigvalsh(G).max(), 1e-9)
    for _ in range(iters):
        w = w - (G @ w - c) / L
        s = np.sort(w)[::-1]
        css = np.cumsum(s) - 1.0
        rho = np.nonzero(s - css / (np.arange(len(s)) + 1) > 0)[0][-1]
        w = np.maximum(w - css[rho] / (rho + 1), 0.0)
    return w, float(np.linalg.norm(A.T @ w - b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modifiers", help="file holding the pasted block, or - "
                                      "for stdin")
    ap.add_argument("--payload", default="artifacts/simplify_data.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--include-reference", action="store_true",
                    help="allow the structures that exist only to build oils")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    d = json.loads((ROOT / args.payload).read_text(encoding="utf-8"))
    desc, mat = d["descriptors"], d["materials"]
    names = [m["n"] for m in mat]
    X = np.asarray([m["p"] for m in mat], float)
    mean = np.asarray(d["mean"], float)
    sd = X.std(0) + 1e-9
    thr = [m.get("thr") for m in mat]
    tsrc = [m.get("thrSrc") for m in mat]

    text = (sys.stdin.read() if args.modifiers == "-"
            else io.open(args.modifiers, encoding="utf-8").read())
    mods, unavailable, unknown = parse_modifiers(text, desc)
    if len(mods) < 3:
        raise SystemExit(f"only {len(mods)} modifiers parsed; expected a block "
                         f"of 'descriptor<tab>intensity' lines")

    print(f"[mod] {len(mods)} modifiers read")
    for name, v, src in sorted(mods, key=lambda t: -t[1]):
        print(f"   {name:15s} {v:4.2f}  {src[:52]}")
    if unknown:
        print(f"\n[mod] {len(unknown)} line(s) named something outside the "
              f"138-word vocabulary and were dropped: {', '.join(unknown)}")
    if unavailable:
        print(f"[mod] the model reported no descriptor for: "
              f"{', '.join(unavailable)}")
        print("      those are absent from the target, not approximated into "
              "it, so nothing below delivers them.")

    w = np.zeros(len(desc))
    for name, v, _ in mods:
        w[desc.index(name)] = v
    target = np.clip(mean + 2.0 * w * sd, 0.0, 1.0)
    scale = np.sqrt(w) / sd
    Xs, ts = X * scale, target * scale

    pool = [i for i in range(len(mat))
            if not mat[i].get("noUse")
            and (args.include_reference or not mat[i].get("unver"))]
    print(f"\n[mod] searching {len(pool)} materials")

    chosen = []
    for k in range(1, args.k + 1):
        best, bi = None, None
        for i in pool:
            if i in chosen:
                continue
            _, e = fit(Xs[chosen + [i]], ts, iters=150)
            if best is None or e < best:
                best, bi = e, i
        chosen.append(bi)
        _, e = fit(Xs[chosen], ts)
        print(f"   k={k:2d}  weighted error {e:5.2f}")

    act, err = fit(Xs[chosen], ts)
    keep = [j for j in range(len(chosen)) if act[j] > 0.004]
    dropped = len(chosen) - len(keep)
    chosen = [chosen[j] for j in keep]
    act = act[keep] / act[keep].sum()
    raw = np.asarray([act[j] * (thr[i] ** DOSE_EXP)
                      for j, i in enumerate(chosen)])
    mass = raw / raw.sum()

    print(f"\n[mod] formula, {len(chosen)} materials"
          + (f" ({dropped} of the {args.k} earned nothing and were dropped)"
             if dropped else ""))
    print(f"  {'material':30s} {'parts':>8s} {'%smell':>7s} {'thr':>8s} "
          f"{'tier':>8s}  flags")
    for j in np.argsort(-mass):
        i = chosen[j]
        flags = []
        if mat[i].get("oil"):
            flags.append("EO")
        if tsrc[i] == "imputed":
            flags.append("threshold imputed")
        print(f"  {names[i][:30]:30s} {100 * mass[j]:8.3f} "
              f"{100 * act[j]:6.1f}% {thr[i]:8.3g} {tsrc[i]:>8s}  "
              + "; ".join(flags))

    n_imp = sum(1 for i in chosen if tsrc[i] == "imputed")
    if n_imp:
        print(f"\n  {n_imp} of {len(chosen)} doses rest on an imputed "
              f"threshold. Those run high on potent materials, so a very "
              f"strong ingredient is dosed too heavily; treat the parts as an "
              f"ordering and smell before trusting the weight.")

    got = act @ X[chosen]
    gap = (target - got) / sd
    print("\n[mod] modifier by modifier:")
    for name, v, _ in sorted(mods, key=lambda t: -abs(gap[desc.index(t[0])])):
        g = gap[desc.index(name)]
        tag = "SHORT" if g > 0.5 else ("over" if g < -0.5 else "met")
        print(f"   {name:15s} asked {v:4.2f}   off {g:+5.2f} sd   {tag}")
    met = sum(1 for n, _, _ in mods if abs(gap[desc.index(n)]) <= 0.5)
    print(f"\n  {met} of {len(mods)} modifiers met within half a standard "
          f"deviation (yardstick {d['yardstick']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
