"""Does receptor space follow chemistry or smell where the two disagree?

The earlier test asked whether materials sharing a descriptor are closer on
average. That can pass while failing on the cases that matter: two materials
from unrelated chemical families that smell alike. A nitro musk and a
macrocyclic musk share almost no structure, so if the vectors encode smell they
should still be neighbours, and if they encode chemistry they will not be.

Three independent measures per pair:

* **structure** -- Morgan (ECFP4) Tanimoto, computed from the molecule alone.
  Nothing to do with either model, so it is a clean chemistry axis.
* **smell** -- cosine over GS-LF descriptor predictions. A different checkpoint,
  a different architecture and a different training target from the receptor
  model, so it is at least not circular. It is still a model, not a nose.
* **receptor** -- distance in the MPNN receptor space being evaluated.

The decisive quantity is the partial correlation of receptor distance with
smell once structure is held fixed. If that collapses toward zero, receptor
space carries no odour information beyond what structure already gave it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402


def partial_spearman(a, b, c):
    """Spearman(a, b) with c held fixed, via ranks and linear residuals."""
    from scipy.stats import rankdata

    ra, rb, rc = (rankdata(x) for x in (a, b, c))
    ra, rb, rc = (x - x.mean() for x in (ra, rb, rc))
    ra_r = ra - rc * (ra @ rc) / (rc @ rc)
    rb_r = rb - rc * (rb @ rc) / (rc @ rc)
    return float(ra_r @ rb_r / np.sqrt((ra_r @ ra_r) * (rb_r @ rb_r)))


def main() -> int:
    entries, _, _ = Library(name="library_mpnn_expanded").load()
    _, logits, _ = Library(name="library_mpnn_expanded").load()
    names = [e.name for e in entries]
    smiles = [e.canonical_smiles for e in entries]
    n = len(names)

    pj = json.loads((ROOT / "artifacts" / "percepts_expanded.json").read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    P = np.stack([np.asarray(r, float) for r in pj["probs"]])
    pnames = pj["molecules"]
    if pnames != names:
        pos = {m: i for i, m in enumerate(pnames)}
        P = P[[pos[x] for x in names]]

    # chemistry: ECFP4 Tanimoto
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    tan = np.ones((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        tan[i] = sims

    # smell: centred cosine over descriptor predictions
    pc = P - P.mean(axis=0)
    norms = np.linalg.norm(pc, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    smell = (pc / norms) @ (pc / norms).T

    # receptors
    red, _ = preprocess(logits, n_components=20)
    rec = squareform(pdist(red))

    iu = np.triu_indices(n, 1)
    t, s, r = tan[iu], smell[iu], rec[iu]

    print(f"{n} materials, {len(t):,} pairs\n")
    print("HOW DOES RECEPTOR DISTANCE RELATE TO EACH AXIS?")
    print(f"  vs structure (Tanimoto)     Spearman {spearmanr(r, t).statistic:+.4f}")
    print(f"  vs smell (descriptor cos)   Spearman {spearmanr(r, s).statistic:+.4f}")
    print(f"  structure vs smell          Spearman {spearmanr(t, s).statistic:+.4f}")
    print()
    print("  partial correlations (the decisive ones)")
    print(f"    receptor~structure, smell held fixed  {partial_spearman(r, t, s):+.4f}")
    print(f"    receptor~smell, structure held fixed  {partial_spearman(r, s, t):+.4f}")

    # the hard quadrant
    far_chem = t < 0.25
    near_smell = s > 0.90
    hard = far_chem & near_smell
    print(f"\nTHE HARD CASES: structurally unrelated (Tanimoto<0.25) but "
          f"smell alike (cos>0.90)\n  {hard.sum():,} pairs of {len(t):,}")
    if hard.sum():
        pctl = 100.0 * np.searchsorted(np.sort(r), r[hard]) / len(r)
        print(f"  their receptor distance sits at the {np.median(pctl):.1f}th percentile "
              f"(median)\n  if receptor space tracked smell this would be low; "
              f"50 means indistinguishable from random pairs")
        ctrl = far_chem & (s < 0.5)
        pc2 = 100.0 * np.searchsorted(np.sort(r), r[ctrl]) / len(r)
        print(f"  control, equally unrelated chemically but smelling different: "
              f"{np.median(pc2):.1f}th percentile")

    # musks, the case the question was really about
    musk_i = desc.index("musk")
    musky = [i for i in range(n) if P[i, musk_i] >= 0.25]
    print(f"\nMUSKS (predicted musk >= 0.25): {len(musky)}")
    for i in sorted(musky, key=lambda k: -P[k, musk_i]):
        print(f"  {names[i][:34]:36s} musk {P[i, musk_i]:.2f}   {smiles[i][:44]}")

    if len(musky) > 1:
        print("\n  cross-chemotype musk pairs (Tanimoto < 0.25)")
        print(f"  {'pair':52s} {'tanimoto':>9s} {'smell':>7s} {'receptor':>9s} {'pctile':>7s}")
        rows = []
        for a in range(len(musky)):
            for b in range(a + 1, len(musky)):
                i, j = musky[a], musky[b]
                if tan[i, j] >= 0.25:
                    continue
                p = 100.0 * np.searchsorted(np.sort(r), rec[i, j]) / len(r)
                rows.append((p, names[i], names[j], tan[i, j], smell[i, j], rec[i, j]))
        rows.sort()
        for p, ni, nj, tt, ss, rr in rows[:14]:
            print(f"  {(ni[:24] + ' / ' + nj[:24]):52s} {tt:9.3f} {ss:7.3f} "
                  f"{rr:9.2f} {p:6.1f}%")
        if rows:
            print(f"\n  median receptor percentile across {len(rows)} cross-chemotype "
                  f"musk pairs: {np.median([x[0] for x in rows]):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
