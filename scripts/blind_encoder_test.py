"""A blind forced-choice test between the GCN and MPNN receptor layouts.

Four aggregate measures have now been tried on this question and none settles
it. Nearest-neighbour agreement with the perfumer's notes favours the GCN, 0.554
against 0.457 at p = 0.005. Global mean distance over descriptor-sharing pairs
favours the MPNN, marginally, at p = 0.030 and failing its stricter variant.
Family coherence favours the GCN by +0.008 to +0.014, with intervals spanning
ten times that. Homologous-series tightness favours the GCN heavily, but that
may be over-clustering by carbon count rather than a virtue.

Meanwhile the two bench observations that actually moved opinion pointed
opposite ways: the GCN puts Styrallyl Acetate next to Dimethyl Benzyl Carbinyl
Acetate where the MPNN chose Strawberry Glycidate, and the MPNN puts Oxane next
to Thiocineole where the GCN chose Citral.

So the deciding evidence has to be the nose, and it has to be blind, because
every unblinded comparison so far has been read in the direction of whatever
was expected. The design is forced choice: for one probe material, the two
encoders each nominate their nearest neighbour; the perfumer smells all three
and says which neighbour is closer to the probe. The listing order is shuffled
per trial so position carries no signal, and the key naming which model
nominated which candidate is written to a file and never printed.

Trials are chosen where the models disagree most -- each one's nominee is remote
under the other -- because an agreed pair tests nothing. Only materials the
perfumer owns are used, and anything already discussed is excluded, since those
carry expectations.

Reading the result: under the null that both encoders are equally good, wins
follow a fair coin. With 12 trials, 10 or more is p = 0.019 one-sided, 9 is
0.073, 8 is 0.194. So a clear outcome is informative and a 7-5 split is not.
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

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402

#: Already examined with the models' answers visible, so they carry
#: expectations and cannot be part of a blind test.
BURNED = {
    "Styrallyl Acetate", "Dimethyl Benzyl Carbinyl Acetate",
    "Strawberry Glycidate", "Oxane", "Thiocineole", "Citral",
    "Rose Oxide", "Para Cresyl Acetate", "Habanolide", "Dimethyl Sulfide",
    "Geraniol 95", "Nerol", "Nootkatone",
}


def load(name: str):
    entries, logits, _ = Library(name=name).load()
    names = [e.name for e in entries]
    red, _ = preprocess(logits, n_components=20)
    return names, squareform(pdist(red))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--gcn", default="library")
    ap.add_argument("--mpnn", default="library_mpnn_clean")
    ap.add_argument("--out", default="artifacts/blind_encoder_key.json")
    args = ap.parse_args()

    owned = {r["name"] for r in csv.DictReader(
        io.open(ROOT / "library/input/expanded_materials.csv",
                encoding="utf-8-sig")) if r.get("source") == "curated"}
    owned -= BURNED
    print(f"[blind] {len(owned)} owned materials available after exclusions")

    gn, GD = load(args.gcn)
    mn, MD = load(args.mpnn)
    pool = sorted(owned & set(gn) & set(mn))
    gi = {n: gn.index(n) for n in pool}
    mi = {n: mn.index(n) for n in pool}

    def nearest(dm, idx, probe):
        cand = [(dm[idx[probe], idx[c]], c) for c in pool if c != probe]
        cand.sort()
        return cand[0][1], {c: r for r, (_, c) in enumerate(cand, 1)}

    rows = []
    for p in pool:
        g_nn, g_rank = nearest(GD, gi, p)
        m_nn, m_rank = nearest(MD, mi, p)
        if g_nn == m_nn:
            continue
        # Disagreement: each model's nominee should be remote under the other.
        score = min(m_rank[g_nn], g_rank[m_nn])
        rows.append({"probe": p, "gcn": g_nn, "mpnn": m_nn,
                     "gcn_nominee_rank_under_mpnn": m_rank[g_nn],
                     "mpnn_nominee_rank_under_gcn": g_rank[m_nn],
                     "disagreement": score})
    rows.sort(key=lambda r: -r["disagreement"])
    print(f"[blind] {len(rows)} probes where the two models nominate "
          f"different neighbours")

    # Keep each material in at most one trial, so no two trials share a smell.
    picked, used = [], set()
    for r in rows:
        trio = {r["probe"], r["gcn"], r["mpnn"]}
        if trio & used:
            continue
        picked.append(r)
        used |= trio
        if len(picked) == args.trials:
            break
    print(f"[blind] {len(picked)} trials selected, no material reused")

    rng = np.random.default_rng(args.seed)
    key = {"seed": args.seed, "design": "forced choice, 1 probe vs 2 nominees",
           "gcn_store": args.gcn, "mpnn_store": args.mpnn, "trials": []}

    print("\n" + "=" * 68)
    print("BLIND TEST - which neighbour smells closer to the probe?")
    print("=" * 68)
    print("For each trial, smell the probe, then both candidates. Say whether")
    print("(a) or (b) is closer in smell to the probe. If you genuinely cannot")
    print("choose, record a tie rather than guessing.\n")

    for t, r in enumerate(picked, 1):
        flip = bool(rng.integers(0, 2))
        a, b = (r["mpnn"], r["gcn"]) if flip else (r["gcn"], r["mpnn"])
        key["trials"].append({
            "trial": t, "probe": r["probe"],
            "a": a, "b": b,
            "a_model": "MPNN" if flip else "GCN",
            "b_model": "GCN" if flip else "MPNN",
            "gcn_nominee_rank_under_mpnn": r["gcn_nominee_rank_under_mpnn"],
            "mpnn_nominee_rank_under_gcn": r["mpnn_nominee_rank_under_gcn"],
        })
        print(f" {t:2d}.  probe:  {r['probe']}")
        print(f"        (a)  {a}")
        print(f"        (b)  {b}\n")

    out = ROOT / args.out
    out.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print("=" * 68)
    print(f"key written to {args.out} - do not open it until you have answered")
    print("record answers as, for example:  1a 2b 3b 4a 5tie ...")
    print("\nUnder the null that the encoders are equally good, wins are a fair")
    print(f"coin. With {len(picked)} trials: 10+ is p = 0.019 one-sided, 9 is")
    print("0.073, 8 is 0.194. A 7-5 split means nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
