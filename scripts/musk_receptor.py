"""Did the model learn a musk receptor? Yes, and the readout throws it away.

``scripts/musk_test.py`` shows that whole-panel distance puts nitro and
macrocyclic musks at the 85th percentile of all pairs -- structurally
unrelated materials that smell the same, placed as far apart as strangers. The
README explained that as a gap in the training data: OR5AN1 is the known musk
receptor, its learned signal is weak, so "musk" was never learnable as one
thing.

That explanation was never measured, and it is wrong. This script measures it.

OR5AN1 is column 12 of the panel. Its logit spread is the 97th percentile of
all 386 receptors, so it is one of the panel's *most* discriminating columns,
not a dead one. The three nitro musks are the top three materials in the whole
library on it, and four macrocyclics sit in its top fifteen -- two chemical
families with no scaffold in common, agreeing on one receptor.

The obvious objection is size. Musks are large, OR5AN1's response correlates
+0.45 with molecular weight, and a receptor that merely liked big molecules
would look like this. So the test controls for it: regress the response on
rank molecular weight and compare musks to everything else on the residual.
It survives, and then the comparison worth making is against the panel rather
than against zero -- the same residual effect is computed for all 386
receptors, and OR5AN1 comes first of 386.

So the signal is there, it is specific, and it is in the right receptor. What
destroys it is the distance: Euclidean over PCA-20 of the z-scored 386-vector,
in which one column is one column. Deleting OR5AN1 from the panel entirely
leaves every pairwise distance intact at Spearman +1.0000. A receptor cannot
unite a perceptual class when the metric averages it against 385 others.

This is not a deficiency of M2OR. It is a deficiency of reading the panel with
an unweighted global norm, and unlike a dataset gap it is ours to fix.

    python scripts/musk_receptor.py
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

#: Column of OR5AN1 in the panel ordering, asserted against the gene list.
OR5AN1 = "OR5AN1"

#: The musks in this library, by chemotype. Scaffolds are unrelated across
#: groups, which is the whole point: nothing but smell connects a nitroarene
#: to a C15 macrolactone.
CHEMO = {
    "nitro": ["Musk Xylene", "Musk Ambrette", "Musk Ketone"],
    "macrocyclic": ["Exaltolide Total", "Ambrettolide", "Habanolide",
                    "Globalide", "Civettone", "Muscone", "Velvione"],
    "polycyclic": ["Galaxolide 50% IPM", "Tonalide", "Celestolide",
                   "Cashmeran"],
    "alicyclic": ["Helvetolide", "Edenolide", "Romandolide", "Applelide"],
}


def gene_order(paths) -> list[str]:
    """The panel's gene symbols, in the column order the stores use."""
    tsv = paths.artifacts / "receptors.tsv"
    rows = list(csv.DictReader(io.open(tsv, encoding="utf-8"), delimiter="\t"))
    key = "gene" if "gene" in rows[0] else list(rows[0])[0]
    return [r[key] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="library")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from scipy.spatial.distance import pdist
    from scipy.stats import mannwhitneyu, rankdata, spearmanr
    from sklearn.decomposition import PCA

    from orvector.config import resolve_paths
    from orvector.store import Library

    RDLogger.DisableLog("rdApp.*")
    paths = resolve_paths()
    entries, logits, _ = Library(paths, name=args.store).load()
    names = [e.name for e in entries]
    genes = gene_order(paths)
    if len(genes) != logits.shape[1]:
        raise SystemExit(
            f"{len(genes)} gene symbols for {logits.shape[1]} columns")
    if OR5AN1 not in genes:
        raise SystemExit(f"{OR5AN1} is not in the panel")
    col = genes.index(OR5AN1)
    print(f"[musk] {args.store}: {len(names)} materials x {len(genes)} "
          f"receptors; {OR5AN1} is column {col}")

    Z = (logits - logits.mean(0)) / (logits.std(0) + 1e-12)
    z = Z[:, col]
    pos = {n: i for i, n in enumerate(names)}

    sd = logits.std(0)
    pct_sd = 100.0 * float(np.argsort(np.argsort(sd))[col]) / (len(sd) - 1)
    print(f"\n1. Is the receptor dead? Its logit sd is {sd[col]:.3f}, the "
          f"{pct_sd:.0f}th percentile of the panel")
    print(f"   (panel min {sd.min():.3f}, median {np.median(sd):.3f}, "
          f"max {sd.max():.3f}) -- it is one of the most discriminating "
          f"columns, not a dead one")

    rank_in_lib = np.argsort(np.argsort(z))
    print(f"\n2. Does it respond to musks?  ({OR5AN1} z, and the material's "
          f"rank in the library on it)")
    found = {}
    for grp, ms in CHEMO.items():
        for m in ms:
            if m not in pos:
                continue
            i = pos[m]
            found.setdefault(grp, []).append((m, i))
            print(f"   {m[:30]:30s} {z[i]:+6.2f}   "
                  f"#{len(names) - int(rank_in_lib[i])} of {len(names)}"
                  f"   [{grp}]")
    absent = [m for ms in CHEMO.values() for m in ms if m not in pos]
    if absent:
        print(f"   not in this library: {', '.join(absent)}")

    musk = np.zeros(len(names), bool)
    for v in found.values():
        for _, i in v:
            musk[i] = True
    p_raw = mannwhitneyu(z[musk], z[~musk], alternative="greater").pvalue
    print(f"\n   mean z over {int(musk.sum())} musks {z[musk].mean():+.2f} "
          f"against {z[~musk].mean():+.2f} over the other {int((~musk).sum())}"
          f", one-sided p = {p_raw:.4f}")

    smi = {r["name"].strip(): (r.get("smiles") or "").strip()
           for r in csv.DictReader(io.open(
               ROOT / "library" / "input" / "expanded_materials.csv",
               encoding="utf-8"))}
    mw = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        m = Chem.MolFromSmiles(smi.get(n, "") or "")
        if m is not None:
            mw[i] = Descriptors.MolWt(m)
    ok = ~np.isnan(mw)
    print(f"\n3. Or is it just a size detector? z vs molecular weight is "
          f"Spearman {spearmanr(z[ok], mw[ok]).statistic:+.3f}")
    print(f"   (the panel mean does {spearmanr(Z.mean(1)[ok], mw[ok]).statistic:+.3f},"
          f" so it is only somewhat more size-driven than average)")

    mrank = rankdata(mw[ok])

    def residual(c):
        v = Z[ok, c]
        return v - np.polyval(np.polyfit(mrank, v, 1), mrank)

    r = residual(col)
    mk = musk[ok]
    p_res = mannwhitneyu(r[mk], r[~mk], alternative="greater").pvalue
    print(f"   size-controlled: musks {r[mk].mean():+.2f} against "
          f"{r[~mk].mean():+.2f}, one-sided p = {p_res:.4f}")

    eff = np.array([residual(c)[mk].mean() - residual(c)[~mk].mean()
                    for c in range(Z.shape[1])])
    place = int((eff > eff[col]).sum()) + 1
    print(f"\n   Against zero that p is marginal on {int(mk.sum())} musks, so "
          f"compare it to the panel instead:")
    print(f"   the same size-controlled musk preference computed for all "
          f"{len(genes)} receptors puts {OR5AN1} at **#{place} of "
          f"{len(genes)}** (effect {eff[col]:+.3f}, panel max "
          f"{eff.max():+.3f})")
    runners = [genes[c] for c in np.argsort(-eff)[1:4]]
    print(f"   next strongest: {', '.join(runners)}")

    red = PCA(n_components=20, random_state=0).fit_transform(Z - Z.mean(0))
    Zk = Z.copy()
    Zk[:, col] = 0.0
    redk = PCA(n_components=20, random_state=0).fit_transform(Zk - Zk.mean(0))
    d, dk = pdist(red), pdist(redk)
    print(f"\n4. So why does the map not show it? Delete {OR5AN1} from the "
          f"panel and relayout:")
    print(f"   every pairwise distance agrees at Spearman "
          f"{spearmanr(d, dk).statistic:+.4f}")
    print(f"   median change {np.median(np.abs(d - dk)):.4f} on a median "
          f"distance of {np.median(d):.2f}, i.e. "
          f"{100 * np.median(np.abs(d - dk)) / np.median(d):.2f}%")
    print(f"\n   The receptor that knows what a musk is contributes nothing "
          f"measurable to the\n   metric everything is read in. That is the "
          f"finding: not a missing signal, a\n   discarded one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
