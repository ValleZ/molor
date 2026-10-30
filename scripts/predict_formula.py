"""Predict what a formula smells like, and place it among the raw materials.

This answers two different questions, and they deserve different amounts of
trust.

*What does it smell of?* Combine the ingredients' predicted percept vectors.
The combination rule has been tested once, in ``eval_mixtures.py``, against the
perfumer's own written verdicts on 30 of their formulas. Noisy-OR beat the
dominant-ingredient baseline there, but the margin was significant only with
four closely related formulas included, so the rule is better than nothing and
not much more than that. Read the descriptor list as a ranking, not as
intensities.

*What is it near in the library?* This is on firmer ground, because it asks
only for a distance between vectors the model produced directly. Two clouds are
reported. Percept space is the space of predicted smell, so a neighbour there
is a material predicted to smell similar. Receptor space is the model's own
output space, and a mixture has no receptor vector of its own -- the one used
here assumes a receptor fires if any ingredient fires it, which is the right
shape at low occupancy and wrong once ligands compete for the same pocket. It
is reported because it places the blend on the published atlas, not because it
is more reliable than the percept neighbours.

The model cannot name a perfume. Nothing in its training data pairs a
composition with a name, so any name here would be a human gloss on the
descriptor ranking, not a prediction.

Input is a CSV with columns ``material,dilution,weight``: the stock strength as
a percentage and the weight of that stock, exactly as a formula sheet records
them. Neat parts are computed from the two, so diluted ingredients are weighted
by how much material they actually contribute.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixtures import combine  # noqa: E402
from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402
from parse_formulas import classify  # noqa: E402


def read_formula(path: Path):
    rows = []
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        name = (r.get("material") or "").strip()
        if not name:
            continue
        dil = float(r.get("dilution") or 100.0)
        wt = float(r.get("weight") or 0.0)
        rows.append({"name": name, "dilution": dil, "weight": wt,
                     "neat": wt * dil / 100.0})
    return rows


def centred_cosine_to(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine of q to each row of P, after centring both on P's mean."""
    mu = P.mean(0)
    X = P - mu
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    y = q - mu
    y /= max(float(np.linalg.norm(y)), 1e-12)
    return X @ y


def thresholds() -> dict[str, float]:
    """Measured water-phase detection thresholds in ppb, by InChIKey block."""
    path = ROOT / "data" / "inventory" / "inventory.csv"
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        v = (r.get("water_threshold_ppb") or "").strip()
        k = (r.get("inchikey") or "").strip()
        if v and k:
            try:
                out[k[:14]] = float(v)
            except ValueError:
                pass
    return out


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("formula", help="CSV with material,dilution,weight")
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--store", default="library")
    ap.add_argument("--rule", default="noisy-or",
                    choices=["noisy-or", "weighted-mean", "max", "dominant"])
    ap.add_argument("--weight", default="mass", choices=["mass", "oav"],
                    help="mass: neat share. oav: neat share / measured "
                         "detection threshold, so traces of strong materials "
                         "count for what they do")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}

    entries, logits, scores = Library(name=args.store).load()
    lib_names = [e.name for e in entries]
    lib_set = set(lib_names)
    S = {n: s for n, s in zip(lib_names, scores)}

    fpath = Path(args.formula)
    rows = read_formula(fpath if fpath.is_absolute() else ROOT / fpath)
    total_neat = sum(r["neat"] for r in rows)

    thr = thresholds()
    block = {e.name: e.inchikey[:14] for e in entries}
    resolved, missing = [], []
    for r in rows:
        kind, name = classify(r["name"], lib_set)
        if kind == "material" and name in P:
            r["resolved"] = name
            r["threshold"] = thr.get(block.get(name, ""))
            resolved.append(r)
        else:
            r["resolved"] = None
            r["threshold"] = None
            missing.append(r)

    cov = sum(r["neat"] for r in resolved) / total_neat
    print(f"[formula] {fpath.name}: {len(rows)} ingredients, "
          f"{total_neat:.2f} g neat material")
    print(f"  resolved {len(resolved)}, covering {cov:.1%} of the neat weight\n")
    print(f"  {'ingredient':28s} {'stock':>6s} {'g':>8s} {'parts/1000':>11s}  "
          f"resolved to")
    print("  " + "-" * 88)
    for r in sorted(rows, key=lambda x: -x["neat"]):
        tag = r["resolved"] or "-- not in the library, dropped --"
        print(f"  {r['name'][:28]:28s} {r['dilution']:5.0f}% {r['weight']:8.3f} "
              f"{1000 * r['neat'] / total_neat:11.2f}  {tag}")
    if missing:
        lost = sum(r["neat"] for r in missing) / total_neat
        print(f"\n  dropped {len(missing)} ingredient(s), {lost:.1%} of the "
              f"neat weight")

    if len(resolved) < 3:
        raise SystemExit("too few resolvable ingredients to combine")

    w = np.array([r["neat"] for r in resolved], float)
    if args.weight == "oav":
        # Odour activity: neat share divided by the measured detection
        # threshold, so a trace of something enormously strong counts for what
        # it does rather than for what it weighs. The thresholds are water-phase
        # and assembled from mixed sources, which is the wrong phase for
        # sniffing a blend and not very accurate either; and detection
        # threshold is not perceived contribution at the dose used, which is
        # what the rule actually wants. Ethyl Maltol shows the gap: a weak
        # threshold, yet a few percent of it is plainly sweet on the bench.
        have = [r for r in resolved if r["threshold"] is not None]
        if len(have) < len(resolved):
            lost = [r["resolved"] for r in resolved if r["threshold"] is None]
            print(f"\n  [oav] no measured threshold for {', '.join(lost)}; "
                  f"dropped from the odour-activity weighting")
        if len(have) < 3:
            raise SystemExit("too few ingredients have a measured threshold")
        resolved = have
        w = np.array([r["neat"] / r["threshold"] for r in resolved], float)
    w = w / w.sum()
    V = np.stack([P[r["resolved"]] for r in resolved])
    mix = combine(args.rule, V, w)

    order = np.argsort(-mix)[: args.top]
    print(f"\n  predicted descriptors ({args.rule}, ranking only):")
    for i in order:
        bar = "#" * int(round(28 * mix[i] / max(mix.max(), 1e-9)))
        print(f"    {desc[i]:16s} {mix[i]:.3f}  {bar}")

    print("\n  what drives the top descriptors:")
    for i in order[:5]:
        share = w * V[:, i]
        j = np.argsort(-share)[:3]
        who = ", ".join(f"{resolved[k]['resolved']} ({V[k, i]:.2f})" for k in j)
        print(f"    {desc[i]:16s} <- {who}")

    pnames = list(pj["molecules"])
    PM = np.stack([P[n] for n in pnames])
    sim = centred_cosine_to(PM, mix)
    own = {r["resolved"] for r in resolved}
    print("\n  nearest raw materials in percept space (predicted smell):")
    for k in np.argsort(-sim)[: args.neighbours]:
        mark = " *" if pnames[k] in own else "  "
        print(f"   {mark} {sim[k]:+.3f}  {pnames[k]}")
    print("    * = an ingredient of this formula")

    probs = np.stack([S[r["resolved"]] for r in resolved])
    ww = np.clip(w / w.max(), 0, 1)[:, None]
    rec_mix = 1.0 - np.prod(1.0 - ww * np.clip(probs, 0, 1 - 1e-6), axis=0)
    red, _ = preprocess(logit(np.vstack([scores, rec_mix[None, :]])),
                        n_components=20)
    D = squareform(pdist(red))[-1, :-1]
    med = float(np.median(pdist(red[:-1])))
    print(f"\n  nearest raw materials in receptor space (median library "
          f"separation {med:.2f}):")
    for k in np.argsort(D)[: args.neighbours]:
        mark = " *" if lib_names[k] in own else "  "
        print(f"   {mark} {D[k]:6.2f}  {lib_names[k]}")

    if args.out:
        rep = {"formula": fpath.name, "rule": args.rule,
               "coverage": round(cov, 4),
               "ingredients": [{"name": r["name"], "resolved": r["resolved"],
                                "parts_per_1000": round(
                                    1000 * r["neat"] / total_neat, 2)}
                               for r in rows],
               "descriptors": [{"name": desc[i], "p": round(float(mix[i]), 4)}
                               for i in np.argsort(-mix)[:30]],
               "percept_neighbours": [
                   {"name": pnames[k], "cos": round(float(sim[k]), 4)}
                   for k in np.argsort(-sim)[:15]],
               "receptor_neighbours": [
                   {"name": lib_names[k], "d": round(float(D[k]), 3)}
                   for k in np.argsort(D)[:15]]}
        (ROOT / args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[formula] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
