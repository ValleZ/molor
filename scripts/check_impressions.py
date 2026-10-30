"""Test the receptor vectors against the user's own smelling notes.

Everything so far has been scored against chemical-class labels. These notes
are perceptual, written by one person on one set of materials, so they are the
first chance to ask whether receptor-space distance tracks *smell* rather than
structure.

The descriptor vocabulary below is deliberately coarse and was written to match
this person's wording ("ringing", "weed", "pony"), not a standard odour
lexicon. Treat it as a noisy instrument: one nose, 128 materials, free text.
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402

# descriptor -> substrings that signal it in these notes
VOCAB = {
    "rose": ["rose"],
    "citrus": ["citrus", "lemon", "lime", "orange", "grapefruit", "bergamot", "zest"],
    "green": ["green", "grass", "lawn", "meadow", "leaves", "stems", "hay", "weed",
              "herb", "reed"],
    "peach": ["peach", "apricot"],
    "pear": ["pear"],
    "coconut": ["coconut"],
    "musk": ["musk"],
    "woody": ["wood", "sandal", "cedar", "pine", "lumber", "sawdust", "fir"],
    "floral": ["floral", "flower", "lily", "lilac", "violet", "jasmine", "peon",
               "blossom", "muguet", "iris"],
    "vanilla": ["vanill"],
    "berry": ["berry", "berries", "raspberry", "strawberry", "blackcurrant",
              "blackberr", "currant", "gooseberry"],
    "metallic": ["metallic", "aliminium", "foil", "mineral"],
    "animalic": ["animalistic", "fecal", "horse", "leather", "fur", "pony", "skin"],
    "dusty": ["dust", "dried", "powdery"],
    "anise": ["anise", "sambuca"],
    "spice": ["clove", "cinnamon"],
    "melon": ["watermelon", "melon", "cucumber"],
    "medicinal": ["medicinal", "iodine", "chemical", "acetone", "paint"],
    "honey": ["honey"],
    "apple": ["apple"],
    "banana": ["banana"],
    "pineapple": ["pineapple"],
    "marine": ["seaweed", "kelp", "salty water"],
    "sugary": ["sugar", "candy", "caramel", "sweet"],
    "aldehydic": ["aldehyd", "ringing", "waxy"],
    "fruity": ["fruit", "ripe"],
}


#: Coarse families. A first pass used only the fine tags above, which scored
#: "BANANA" and "ripened fruits" as sharing nothing -- an artefact of the
#: vocabulary, not of the model. These roll fine descriptors up so that related
#: smells count as related.
FAMILIES = {
    "FRUIT": ["banana", "pear", "peach", "apple", "berry", "pineapple",
              "fruity", "melon", "citrus"],
    "GREEN": ["green"],
    "FLORAL": ["floral", "rose"],
    "SWEET": ["sugary", "vanilla", "honey"],
    "WOODY": ["woody"],
    "ANIMAL": ["musk", "animalic"],
}


def tags_for(text: str, families: bool = True) -> set[str]:
    low = text.lower()
    t = {d for d, keys in VOCAB.items() if any(k in low for k in keys)}
    if families:
        for fam, members in FAMILIES.items():
            if t & set(members):
                t.add(fam)
    return t


def load(name):
    lib = Library(name=name)
    entries, logits, _ = lib.load()
    names = [e.name for e in entries]
    red, _ = preprocess(logits, n_components=20)
    D = squareform(pdist(red))
    return names, {n: i for i, n in enumerate(names)}, D, np.sort(pdist(red))


def main() -> int:
    notes = {
        r["material"]: r["impression"]
        for r in csv.DictReader(
            open(ROOT / "library/input/impressions.tsv", encoding="utf-8"), delimiter="\t"
        )
    }

    # Store names are passed in rather than fixed, because the libraries get
    # rebuilt under new names whenever the featurisation or the material list
    # changes, and silently scoring a stale store is the easy mistake here.
    gcn_name = "library"
    mpnn_name = "library_mpnn_s6"
    for i, a in enumerate(sys.argv):
        if a == "--gcn" and i + 1 < len(sys.argv):
            gcn_name = sys.argv[i + 1]
        if a == "--mpnn" and i + 1 < len(sys.argv):
            mpnn_name = sys.argv[i + 1]
    print(f"GCN store: {gcn_name}   MPNN store: {mpnn_name}")

    models = {}
    for key, libname in (("GCN", gcn_name), ("MPNN", mpnn_name)):
        names, idx, D, flat = load(libname)
        models[key] = dict(names=names, idx=idx, D=D, flat=flat)

    names = models["MPNN"]["names"]
    use_fam = "--flat" not in sys.argv
    tags = {n: tags_for(notes[n], families=use_fam) for n in names if n in notes}
    print("vocabulary: " + ("fine tags + coarse families" if use_fam else "fine tags only"))
    usable = [n for n in names if tags.get(n)]
    print(f"materials with at least one descriptor: {len(usable)}/{len(names)}")
    from collections import Counter

    cnt = Counter(t for n in usable for t in tags[n])
    print("descriptor frequency: " + ", ".join(f"{t}:{c}" for t, c in cnt.most_common(12)))

    def pct(model, d):
        f = models[model]["flat"]
        return 100.0 * np.searchsorted(f, d) / len(f)

    # ---- the real test: do shared descriptors predict receptor distance? ----
    print("\n" + "=" * 86)
    print("DOES SHARED SMELL PREDICT RECEPTOR-SPACE PROXIMITY?")
    print("=" * 86)
    rng = np.random.default_rng(0)
    for key in ("GCN", "MPNN"):
        m = models[key]
        for min_shared in (1, 2):
            shared, other = [], []
            for a, b in combinations(usable, 2):
                d = m["D"][m["idx"][a], m["idx"][b]]
                (shared if len(tags[a] & tags[b]) >= min_shared else other).append(d)
            shared, other = np.array(shared), np.array(other)
            obs = other.mean() - shared.mean()

            # Permutation: reshuffle which material owns which descriptor set.
            pool = [tags[n] for n in usable]
            null = []
            for _ in range(2000):
                perm = rng.permutation(len(pool))
                st = {n: pool[perm[i]] for i, n in enumerate(usable)}
                s2, o2 = [], []
                for a, b in combinations(usable, 2):
                    d = m["D"][m["idx"][a], m["idx"][b]]
                    (s2 if len(st[a] & st[b]) >= min_shared else o2).append(d)
                null.append(np.mean(o2) - np.mean(s2))
            null = np.array(null)
            p = float((null >= obs).mean())
            print(f"  {key:5s} >={min_shared} shared descriptor(s): "
                  f"{len(shared):5d} pairs, mean dist {shared.mean():6.2f} vs "
                  f"{other.mean():6.2f} others  (gap {obs:+.2f}, p = {p:.4f})")

    # ---- spot check A: what the model calls closest ----
    print("\n" + "=" * 86)
    print("THE MODEL'S 16 CLOSEST PAIRS (MPNN) vs YOUR NOTES")
    print("=" * 86)
    m = models["MPNN"]
    pairs = sorted(
        ((m["D"][m["idx"][a], m["idx"][b]], a, b) for a, b in combinations(names, 2))
    )[:16]
    for d, a, b in pairs:
        share = tags.get(a, set()) & tags.get(b, set())
        mark = "  shared: " + ",".join(sorted(share)) if share else "  NO SHARED DESCRIPTOR"
        print(f"\n  {d:5.2f} ({pct('MPNN', d):4.1f}%)  {a}  <->  {b}{mark}")
        print(f"         {notes.get(a, '')[:76]}")
        print(f"         {notes.get(b, '')[:76]}")

    # ---- spot check B: perceptually close, model far ----
    print("\n" + "=" * 86)
    print("YOU SAY SIMILAR, MODEL SAYS FAR (>=2 shared descriptors, distance > median)")
    print("=" * 86)
    misses = []
    for a, b in combinations(usable, 2):
        sh = tags[a] & tags[b]
        if len(sh) < 2:
            continue
        d = m["D"][m["idx"][a], m["idx"][b]]
        misses.append((pct("MPNN", d), d, a, b, sh))
    misses.sort(reverse=True)
    for p_, d, a, b, sh in misses[:8]:
        print(f"\n  {d:5.2f} ({p_:4.1f}%)  {a}  <->  {b}   shared: {','.join(sorted(sh))}")
        print(f"         {notes.get(a, '')[:76]}")
        print(f"         {notes.get(b, '')[:76]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
