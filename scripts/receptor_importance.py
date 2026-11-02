"""Are some receptors more important than others? Yes -- but per descriptor,
which is why a single reweighting cannot use it.

The question came from a reader: if OR5AN1 is a reliable, specific musk
receptor the way rods are more important than cones for seeing in the dark,
then surely you just scale the important receptors up -- and why did training
not already do that?

Three measurements, in the order that makes the answer clear. The first
refutes the fix. The second explains why. The third says why training could
never have produced it.

A. SCALE IT UP AND SEE. Sweep the weight on OR5AN1 and watch where
   nitro-vs-macrocyclic musk pairs sit in the distance distribution. They get
   *worse*, monotonically, 84% to 94%. The reason is in the numbers: both
   groups sit far above the library on that receptor (+4.4 and +1.8 z) and
   2.6 z apart from each other, while the spread *within* each group is 0.37.
   So OR5AN1 is an excellent class DETECTOR and a terrible similarity AXIS
   for the same class. A receptor that says "these are all musks" need not say
   "these are alike", and `scripts/musk_receptor.py` measured only the first.

   This is a correction to that script's conclusion, which implied the signal
   was merely being diluted and could be recovered by weighting it up.

B. IMPORTANCE IS REAL AND IT IS PER DESCRIPTOR. For every descriptor in the
   two human panels, find the single receptor that best predicts it. Median
   best-single-receptor |rho| is 0.38 on Dravnieks and 0.30 on Keller, with
   peaks near 0.59 -- against +0.03 and +0.02 for the whole 386-dimensional
   distance against panel similarity. So the panel carries a great deal more
   odour information than the global distance expresses. But the receptor that
   serves one descriptor is not the one that serves the next: 50 distinct
   receptors are best for the 146 Dravnieks descriptors, 16 for Keller's 20.

   One weight per receptor, applied to every comparison, cannot express that.
   It has to be the right weighting for "is this minty?" and "is this musky?"
   simultaneously, and those want different receptors. The rod analogy breaks
   in the same place: rods beat cones for detection in dim light and are
   useless for telling two hues apart. Importance in a sensory system is
   relative to the question, not a property of the receptor.

C. WHY TRAINING DID NOT DO IT. Nothing in the checkpoint's loss ever compares
   two molecules. It predicts one binary outcome per (molecule, receptor)
   pair, and the 386-vector is a stack of 386 independent predictions
   assembled afterwards, here, by us. No term in the objective has any opinion
   about the geometry of that stack, so no importance weighting over it could
   emerge -- it was never asked for. Contrast the percept model, whose output
   dimensions ARE odour descriptors, so similarity in its output space is
   similarity in odour-label terms by construction.

   That also names the fix this suggests, which is not a weighted distance at
   all: learn a supervised readout per descriptor FROM the receptor vector.
   Measurement B is a one-receptor lower bound on how well that could work.

    python scripts/receptor_importance.py
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

NITRO = ["Musk Xylene", "Musk Ambrette", "Musk Ketone"]
MACRO = ["Exaltolide Total", "Ambrettolide", "Habanolide", "Civettone",
         "Muscone", "Velvione"]
SWEEP = (1, 3, 10, 30, 100, 1000, 10000)


def gene_order(paths) -> list[str]:
    rows = list(csv.DictReader(
        io.open(paths.artifacts / "receptors.tsv", encoding="utf-8"),
        delimiter="\t"))
    key = "gene" if "gene" in rows[0] else list(rows[0])[0]
    return [r[key] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="library_mpnn_clean")
    ap.add_argument("--receptor", default="OR5AN1")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    from orvector.config import resolve_paths
    from orvector.store import Library
    from receptor_metric import PANELS, centred_cos, load_panel

    paths = resolve_paths()
    entries, logits, _ = Library(paths, name=args.store).load()
    names = [e.name for e in entries]
    keys = [e.inchikey for e in entries]
    Z = (logits - logits.mean(0)) / (logits.std(0) + 1e-12)
    genes = gene_order(paths)
    if args.receptor not in genes:
        raise SystemExit(f"{args.receptor} not in the panel")
    col = genes.index(args.receptor)
    pos = {n: i for i, n in enumerate(names)}
    print(f"[imp] {args.store}: {len(names)} materials x {len(genes)} "
          f"receptors; {args.receptor} is column {col}")

    panels = {}
    for nm, rel in PANELS.items():
        pan, cols = load_panel(ROOT / rel)
        use = [i for i, k in enumerate(keys) if k in pan]
        P = np.stack([pan[keys[i]] for i in use])
        panels[nm] = (use, P, centred_cos(P), cols)
        print(f"      {nm}: {len(use)} molecules in common, "
              f"{len(cols)} descriptors")

    def dist(w):
        V = Z * np.sqrt(w)[None, :]
        g = (V * V).sum(1)
        return np.sqrt(np.maximum(g[:, None] + g[None, :] - 2 * V @ V.T, 0.0))

    pairs = [(pos[a], pos[b]) for a in NITRO for b in MACRO
             if a in pos and b in pos]

    print(f"\nA. Scale it up: weight on {args.receptor} alone, "
          f"{len(pairs)} nitro-vs-macrocyclic musk pairs")
    print(f"   {'weight':>8s} {'musk pair pctile':>17s} "
          f"{'rho Dravnieks':>14s} {'rho Keller':>11s}")
    for mult in SWEEP:
        w = np.ones(Z.shape[1])
        w[col] = mult
        D = dist(w)
        flat = squareform(D, checks=False)
        pct = np.median([100.0 * float((flat < D[a, b]).mean())
                         for a, b in pairs])
        out = []
        for nm in ("dravnieks", "keller"):
            use, P, S, _ = panels[nm]
            sub = D[np.ix_(use, use)]
            iu = np.triu_indices(len(use), 1)
            out.append(-spearmanr(sub[iu], S[iu]).statistic)
        print(f"   {mult:8d} {pct:16.0f}% {out[0]:+14.4f} {out[1]:+11.4f}")

    z = Z[:, col]
    nit = np.array([z[pos[n]] for n in NITRO if n in pos])
    mac = np.array([z[pos[n]] for n in MACRO if n in pos])
    gap = np.mean([abs(a - b) for a in nit for b in mac])
    print(f"\n   Why it gets worse. Both groups sit far above the library on "
          f"{args.receptor}")
    print(f"   (nitro {nit.mean():+.2f} z, macrocyclic {mac.mean():+.2f} z, "
          f"library 0 by construction),")
    print(f"   but the mean gap BETWEEN them on that same axis is {gap:.2f} z "
          f"while the spread")
    print(f"   WITHIN each group is only {nit.std():.2f} and {mac.std():.2f}. "
          f"A good class detector,")
    print(f"   not a similarity axis. Upweighting it separates the class it "
          f"detects.")

    print(f"\nB. Is importance global? Best single receptor per panel "
          f"descriptor")
    for nm in ("dravnieks", "keller"):
        use, P, S, cols = panels[nm]
        Zu = Z[use]
        best = []
        for j, c in enumerate(cols):
            y = P[:, j]
            if y.std() < 1e-9:
                continue
            r = np.array([abs(spearmanr(Zu[:, k], y).statistic)
                          for k in range(Zu.shape[1])])
            r[~np.isfinite(r)] = 0.0
            k = int(np.argmax(r))
            best.append((float(r[k]), c, genes[k]))
        best.sort(reverse=True)
        uniq = len({g for _, _, g in best})
        print(f"\n   {nm}: {len(best)} descriptors, served best by {uniq} "
              f"DIFFERENT receptors")
        print(f"   median best-single-receptor |rho| "
              f"{np.median([r for r, _, _ in best]):.3f}, "
              f"max {best[0][0]:.3f}")
        for r, c, g in best[:6]:
            print(f"     {c[:32]:32s} {g:9s} |rho| {r:.3f}")

    # The external check on the musk claim, which does not confirm it.
    use, P, S, cols = panels["keller"]
    if "MUSKY" in cols:
        Zu = Z[use]
        y = P[:, cols.index("MUSKY")]
        r = np.array([abs(spearmanr(Zu[:, k], y).statistic)
                      for k in range(Zu.shape[1])])
        r[~np.isfinite(r)] = 0.0
        rank = int((r > r[col]).sum()) + 1
        top = np.argsort(-r)[:3]
        print(f"\n   An external check that does NOT confirm the musk claim. "
              f"Against Keller's")
        print(f"   own MUSKY rating over {len(use)} molecules, "
              f"{args.receptor} ranks {rank} of {len(genes)} "
              f"(|rho| {r[col]:.3f}).")
        print(f"   Best are {', '.join(f'{genes[k]} {r[k]:.3f}' for k in top)}.")
        print(f"   musk_receptor.py asked a different question -- are this "
              f"library's 14 musks")
        print(f"   elevated on it, size controlled -- and got rank 1 of 386. "
              f"Both can hold, but")
        print(f"   {len(use)} molecules of external panel data outweigh 14 "
              f"in-library contrasts.")

    print(f"\nC. Why training did not do this by itself: the checkpoint's loss "
          f"is one binary")
    print(f"   outcome per (molecule, receptor) pair. It never compares two "
          f"molecules, so it")
    print(f"   has no opinion about the geometry of the {len(genes)}-vector "
          f"we stack afterwards.")
    print(f"   No weighting over that stack could emerge from it. B is the "
          f"one-receptor lower")
    print(f"   bound on what a supervised per-descriptor readout might "
          f"instead recover.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
