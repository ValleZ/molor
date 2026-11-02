"""Export the library as a single JSON payload for the interactive viewer.

Ships everything the browser needs to be self-contained: the score matrix, a
precomputed distance matrix in the same preprocessed space the analysis uses,
coordinates for each projection, and the true mutual-nearest-neighbour links.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from orvector.config import resolve_paths
from orvector.project import mutual_neighbour_edges, neighbour_preservation, preprocess, project
from orvector.store import Library

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_impressions import tags_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


#: Descriptor -> colour. Chosen so the legend reads as the smell it names;
#: related families share a hue (greens together, browns together).
SEMANTIC_COLOUR = {
    "green": "#3f9b4f", "vegetable": "#2b7a46", "herbal": "#7da23c",
    "grassy": "#5faa55", "minty": "#49b58a", "fresh": "#4fb3c4",
    "ozone": "#5aa6d0", "marine": "#3f86b5", "aldehydic": "#9fb8c8",
    "citrus": "#f2b705", "tropical": "#f28a2e", "fruity": "#e8603c",
    "apple": "#cf4f54", "pineapple": "#e0a32b", "banana": "#e8c547",
    "berry": "#c2384f", "melon": "#8fbf6a",
    "rose": "#e86a9a", "floral": "#d16ac4", "powdery": "#b9a8d4",
    "sweet": "#f0a6c0", "vanilla": "#e3d08f", "caramellic": "#c98b35",
    "creamy": "#e2cfae", "coconut": "#d8c7a2",
    "woody": "#8a5a33", "balsamic": "#73452c", "earthy": "#6b5a44",
    "phenolic": "#9a7b3f", "spicy": "#c0452b", "camphoreous": "#7f9aa6",
    "musk": "#8d7bb5", "animal": "#7a6253",
    "fatty": "#c9a86a", "waxy": "#cbbf96", "oily": "#b9a76f",
    "sulfurous": "#9bb02e", "meaty": "#a05a4a", "ethereal": "#8fb8d8",
    "winey": "#8e3b5a", "rummy": "#a85b3a", "nutty": "#a87a4f",
    "other": "#9aa6b1",
}

EDGE_K = 2  # mutual-kNN depth for the true-neighbour links

#: Imported rather than restated so the atlas and the volatility report
#: cannot drift apart on what counts as a top note.
from volatility_profile import GROUP as VOL_GROUP  # noqa: E402

#: First entry is what the viewer opens on, and --models overrides this list.
#: Which encoder deserves that slot is unsettled: after the notes were matched
#: to the cleaned material names, nearest-neighbour agreement with them favours
#: the GCN (0.560, p < 0.0001) over the MPNN (0.474, p = 0.0435), reversing the
#: earlier reading that put the MPNN first. Six measures have not separated
#: them, so the viewer now offers both and lets the nose decide.
# The FIRST entry is the atlas default, and it also defines the molecule set
# and the naturals file everything else aligns to.
#
# The GCN leads as of the seventh GCN-vs-MPNN comparison, which like the six
# before it could not separate them on any perceptual measure -- paired on
# identical pairs against two human panels, the difference was +0.016 and
# -0.007 with intervals an order of magnitude wider. When no perceptual
# evidence distinguishes two encoders over seven attempts, the tie-break is
# mechanism, and there it is not close: the MPNN checkpoint's GRU is saturated
# to 82-100 distinct values across its whole 2816-cell representation and
# annihilates the bond features it is given, while the GCN shows no pinned
# outputs and 490-626 distinct values. Nothing ever paid for the MPNN; it was
# simply first.
MODELS = [
    ("gcn", "library", "Chemistry: 386 receptor responses (GCN)"),
    ("mpnn", "library_mpnn_clean", "Chemistry: 386 receptor responses (MPNN)"),
]


class NaturalEntry:
    """An essential oil, standing in for a library Entry.

    A natural is not a molecule, so it has no InChIKey, formula or SMILES and
    cannot live in the molecule store, which keys on exactly those. It is
    carried alongside instead, with the same receptor and percept vectors, so
    the atlas can place it. ``cls`` is left empty and ``is_natural`` is what
    the viewer keys its different marker off.
    """

    is_natural = True
    cls = ""
    inchikey = ""
    n_atoms = 0

    def __init__(self, name: str, constituents: dict[str, float]):
        self.name = name
        # The inspector shows the build in place of a structure.
        self.canonical_smiles = ""
        self.formula = " + ".join(
            f"{n} {100 * v:.0f}%" for n, v in
            sorted(constituents.items(), key=lambda kv: -kv[1])[:4])


def load_naturals(store: str):
    """Return (entries, percept rows, receptor probability rows) for one store.

    Naturals are built per store because each encoder has its own receptor
    space; ``build_naturals.py --store`` writes one file per store, and a model
    with no file for it simply raises, which the caller already reports as a
    skipped model rather than silently mixing spaces.
    """
    path = (ROOT / "artifacts" / "naturals.json" if store == "library"
            else ROOT / "artifacts" / f"naturals_{store}.json")
    if not path.is_file():
        return [], {}, {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    ents, percept, rec = [], {}, {}
    for oil, v in blob["naturals"].items():
        # Oils that repeat one of their own constituents are built but kept
        # off the atlas: the point would say nothing the molecule beside it
        # already says. They stay in the file for formula prediction.
        if not v.get("atlas", True):
            continue
        e = NaturalEntry(oil, v["composition_used"])
        # Carried through to the card: an oil that sits on top of one of its
        # own constituents is not a distinguishable oil, and saying so is more
        # use than a diamond that implies otherwise.
        e.collapsed_onto = v.get("collapsed_onto")
        e.omitted = v.get("omitted_from_library") or []
        ents.append(e)
        percept[oil] = np.asarray(v["percept"], dtype=float)
        rec[oil] = np.asarray(v["receptors"], dtype=float)
    return ents, percept, rec


class WhitePointEntry:
    """The average of every material: where a big enough mixture ends up.

    Not a material and not a mixture of named things -- the mean of the
    library's vectors, which is the point random blends collapse towards as
    ingredients are added. It earns a place on the atlas because the perfumer
    made four blends aimed at it from four disjoint material sets and reported
    the same base smell from all four, so the location is real rather than an
    artefact of the arithmetic.

    Its receptor vector is the mean of the logits rather than of the
    probabilities, matching how a blend is placed, so it sits where a blend
    aiming at it would sit.
    """

    is_white = True
    is_query = False
    is_natural = False
    cls = ""
    inchikey = ""
    n_atoms = 0

    def __init__(self, n_materials: int):
        self.name = "WHITE POINT"
        self.canonical_smiles = ""
        self.formula = (f"the average of all {n_materials} materials, which is "
                        f"what a large mixture converges to")


def white_point(scores: np.ndarray) -> np.ndarray:
    """Receptor vector of the average material, averaged in logit space."""
    return 1.0 / (1.0 + np.exp(-_logit(scores).mean(axis=0)))


class QueryEntry:
    """One blend being tried out, placed on the atlas but not of the library.

    This is the temporary-experiment path. A formula is not a material and has
    no business in the library store or in a published build, but the only
    question worth asking of a 30-ingredient formula is *where does it land*,
    and that needs it laid out with everything else rather than compared to it
    afterwards. So it is injected at export time from a file that
    predict_formula.py writes, carried through every name-keyed lookup like an
    oil, and marked so the viewer can draw it as what it is.
    """

    is_query = True
    is_natural = False
    cls = ""
    inchikey = ""
    n_atoms = 0

    def __init__(self, name: str, built_from: str, coverage: float):
        self.name = name
        self.canonical_smiles = ""
        self.formula = built_from
        self.coverage = coverage


def load_queries(path: Path, store: str):
    """Return (entries, percept rows, receptor rows) for the injected blends.

    A blend's receptor vector is per encoder, exactly as an oil's is, so the
    file carries one per store and a store it has no vector for is a hard
    error rather than a silent omission: half a query set would place the
    blend on one layout and not the other, and the two layouts would then be
    showing different molecule sets under the same name.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    ents, percept, rec = [], {}, {}
    for q in blob["queries"]:
        name = q["name"]
        if store not in q["receptors"]:
            raise ValueError(
                f"query {name!r} has no receptor vector for store {store!r} "
                f"(it has {sorted(q['receptors'])}); re-run "
                f"predict_formula.py --viewer-point with --stores covering it")
        e = QueryEntry(name, q.get("built_from", ""), q.get("coverage", 0.0))
        e.ingredients = q.get("ingredients", [])
        e.placement = q.get("placement", "union")
        e.dropped = q.get("dropped", [])
        ents.append(e)
        percept[name] = np.asarray(q["percept"], dtype=float)
        rec[name] = np.asarray(q["receptors"][store], dtype=float)
    return ents, percept, rec


def _logit(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def build_model_block(paths, name, genes_ref, names_ref, query=None,
                      white=True):
    """Score matrix, distances, links and projections for one encoder."""
    lib = Library(paths, name=name)
    entries, logits, scores = lib.load()
    nat_ents, _, nat_rec = load_naturals(name)
    if white:
        # Computed from the molecules only, before the oils and any injected
        # blends join the set: the average of the library's materials, not the
        # average of whatever else happens to be on the page.
        w_ent = WhitePointEntry(len(entries))
        nat_ents = list(nat_ents) + [w_ent]
        nat_rec = {**nat_rec, w_ent.name: white_point(scores)}
    if query is not None:
        q_ents, _, q_rec = load_queries(query, name)
        nat_ents = list(nat_ents) + q_ents
        nat_rec = {**nat_rec, **q_rec}
    if nat_ents:
        entries = list(entries) + nat_ents
        extra = np.stack([nat_rec[e.name] for e in nat_ents])
        if extra.shape[1] != scores.shape[1]:
            raise ValueError(
                f"naturals for {name} have {extra.shape[1]} receptors, "
                f"the store has {scores.shape[1]}")
        scores = np.vstack([scores, extra])
        logits = np.vstack([logits, _logit(extra)])
    names = [e.name for e in entries]
    if names != names_ref:
        # Reorder to the reference ordering so one molecule list serves both.
        pos = {n: i for i, n in enumerate(names)}
        missing = [n for n in names_ref if n not in pos]
        if missing:
            raise ValueError(
                f"covers only {len(names_ref) - len(missing)}/{len(names_ref)} "
                f"of the reference molecules (missing e.g. {missing[:2]})"
            )
        order = [pos[n] for n in names_ref]
        logits, scores = logits[order], scores[order]

    reduced, _ = preprocess(logits, drop_pc1=False, n_components=20)
    D = squareform(pdist(reduced))
    def axis_align(coords):
        """Rotate a point cloud so its widest spread runs along x, then y.

        MDS returns an arbitrary orientation, and these clouds are strongly
        elongated -- about three to one. If the long axis lands on y the viewer
        opens as a thin vertical column in a landscape window, which is the
        worst available view of the data. The viewer can only yaw about y and
        pitch about x, with no roll, so it cannot move the long axis out of y
        by itself; the orientation has to be right in the data.

        A rotation preserves every pairwise distance, so Shepard correlations
        and neighbour preservation are untouched.
        """
        c = coords - coords.mean(axis=0, keepdims=True)
        # Principal axes, largest variance first.
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        out = c @ vt.T
        # Fix the sign convention so repeated builds do not mirror randomly.
        for j in range(out.shape[1]):
            if out[np.argmax(np.abs(out[:, j])), j] < 0:
                out[:, j] = -out[:, j]
        return out

    projections = {}
    # 3D first: it preserves local neighbourhoods far better (0.70 vs 0.51 on
    # top-10 overlap), which is what prevents unrelated molecules from looking
    # like cluster members.
    from sklearn.manifold import MDS as _MDS
    from scipy.spatial.distance import squareform as _sq

    for drop3 in (False, True):
        red3, _ = preprocess(logits, drop_pc1=drop3, n_components=20)
        d3 = pdist(red3)
        c3 = _MDS(n_components=3, dissimilarity="precomputed", random_state=0,
                  n_init=6, max_iter=800, normalized_stress="auto").fit_transform(_sq(d3))
        c3 = axis_align(c3)
        from scipy.stats import spearmanr as _sp

        projections[f"mds3_{'quality' if drop3 else 'full'}"] = {
            "label": f"MDS 3D - {'potency removed' if drop3 else 'all components'}",
            "x": [round(float(v), 4) for v in c3[:, 0]],
            "y": [round(float(v), 4) for v in c3[:, 1]],
            "z": [round(float(v), 4) for v in c3[:, 2]],
            "shepard": round(float(_sp(d3, pdist(c3)).statistic), 4),
            "reliability": [round(float(v), 3)
                            for v in neighbour_preservation(red3, c3, k=10)],
            "explained": None,
        }

    for method in ("mds", "pca"):
        for drop in (False, True):
            p = project(logits, genes_ref, method=method, drop_pc1=drop, n_components=20)
            projections[f"{method}_{'quality' if drop else 'full'}"] = {
                "label": f"{method.upper()} 2D - {'potency removed' if drop else 'all components'}",
                "x": [round(float(v), 4) for v in p.coords[:, 0]],
                "y": [round(float(v), 4) for v in p.coords[:, 1]],
                "shepard": round(float(p.shepard), 4),
                "reliability": [round(float(v), 3) for v in p.reliability],
                "explained": ([round(float(v), 4) for v in p.explained]
                              if p.explained is not None else None),
            }
    return {
        "scores": [[round(float(v), 4) for v in row] for row in scores],
        "dist": [[round(float(v), 3) for v in row] for row in D],
        "edges": [[int(a), int(b)] for a, b in mutual_neighbour_edges(reduced, k=EDGE_K)],
        "projections": projections,
        "medianDistance": round(float(np.median(pdist(reduced))), 3),
        "total": [round(float(v), 3) for v in scores.sum(axis=1)],
    }


def build_percept_block(entries, pm, desc):
    """The same projection pipeline, run on percept vectors instead.

    The percept model has its own 138-dimensional space and had no map. It
    gets one here rather than a separate page, because the atlas already
    switches between spaces and everything else on the page -- search,
    neighbour lists, the colouring -- is space-agnostic once the score panel
    knows what its own columns are called.

    Two differences from receptor space, both measured rather than assumed.

    There is no dominant component to remove. In receptor space every
    molecule correlates ~0.95 with the panel mean and PC1 is essentially
    potency, which is why the atlas offers a "potency removed" view. Here PC1
    takes 14.0% of the variance, PC2 10.2%, PC3 9.0%, and the first twenty
    together 81.4%; PC1 loads on sweaty, juicy, brandy, fermented and
    correlates only +0.24 with how many descriptors fire, so it is a
    fruity-fermented direction and not an intensity axis. The PC1-dropped
    variant is still exported for comparison and is slightly WORSE
    (Shepard 0.927 against 0.939), which is the opposite of receptor space,
    where removing PC1 is what makes the chemistry legible.

    And the projection is less faithful, for the same reason: with no
    dominant axis there is more to lose on the way down to three dimensions.
    On the 427 points actually exported, Shepard 0.939 against 0.991 for the
    GCN atlas, and top-10 neighbour preservation 0.63 against 0.70. Still
    clearly better than 2D (0.894) and far better than PCA 2D (0.750), so
    the map is 3D like the other.
    """
    import numpy as np
    from scipy.spatial.distance import squareform as _sq
    from scipy.stats import spearmanr as _sp
    from sklearn.manifold import MDS as _MDS

    rows, missing = [], []
    mean = np.mean([v for v in pm.values()], axis=0)
    for e in entries:
        r = pm.get(e.name)
        if r is None:
            missing.append(e.name)
            rows.append(mean)
        else:
            rows.append(np.asarray(r, dtype=float))
    if missing:
        print(f"  percept: {len(missing)} entries have no percept vector and "
              f"sit at the mean: {', '.join(missing[:4])}")
    P = np.vstack(rows)

    def axis_align(coords):
        c = coords - coords.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        out = c @ vt.T
        for j in range(out.shape[1]):
            if out[np.argmax(np.abs(out[:, j])), j] < 0:
                out[:, j] = -out[:, j]
        return out

    reduced, _ = preprocess(P, drop_pc1=False, n_components=20)
    D = squareform(pdist(reduced))
    projections = {}
    for drop in (False, True):
        red, _ = preprocess(P, drop_pc1=drop, n_components=20)
        d3 = pdist(red)
        c3 = _MDS(n_components=3, dissimilarity="precomputed", random_state=0,
                  n_init=6, max_iter=800,
                  normalized_stress="auto").fit_transform(_sq(d3))
        c3 = axis_align(c3)
        tag = "quality" if drop else "full"
        projections[f"mds3_{tag}"] = {
            "label": ("MDS 3D - first component removed" if drop
                      else "MDS 3D - all components"),
            "x": [round(float(v), 4) for v in c3[:, 0]],
            "y": [round(float(v), 4) for v in c3[:, 1]],
            "z": [round(float(v), 4) for v in c3[:, 2]],
            "shepard": round(float(_sp(d3, pdist(c3)).statistic), 4),
            "reliability": [round(float(v), 3)
                            for v in neighbour_preservation(red, c3, k=10)],
            "explained": None,
        }
    for method in ("mds", "pca"):
        for drop in (False, True):
            pr = project(P, list(desc), method=method,
                         drop_pc1=drop, n_components=20)
            tag = "quality" if drop else "full"
            projections[f"{method}_{tag}"] = {
                "label": (f"{method.upper()} 2D - "
                          + ("first component removed" if drop
                             else "all components")),
                "x": [round(float(v), 4) for v in pr.coords[:, 0]],
                "y": [round(float(v), 4) for v in pr.coords[:, 1]],
                "shepard": round(float(pr.shepard), 4),
                "reliability": [round(float(v), 3) for v in pr.reliability],
                "explained": ([round(float(v), 4) for v in pr.explained]
                              if pr.explained is not None else None),
            }
    # The projection BASIS, so a point that did not exist at export time can
    # still be placed. Without it the page holds only a finished distance
    # matrix and finished coordinates, which is enough to draw the map and not
    # enough to add to it -- and refitting MDS in the page would both be slow
    # and move every existing point, producing a different map.
    #
    # With the basis a new percept vector can be z-scored and projected into
    # the same 20 components the layout was built from, its true distances to
    # the library computed, and its position interpolated from its neighbours.
    # Leave-one-out over all 427 library points, placing each from its k=5
    # neighbours by the locally-linear rule the page implements: median error
    # 1.08 against a median library pair of 12.24, so 8.8%. More to the point,
    # the placed position lands near 63% of a point's true ten nearest
    # neighbours -- the same 63% that MDS itself preserves going 138 dimensions
    # to three, so the interpolation costs essentially nothing beyond what the
    # projection already costs.
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    keep = sd > 1e-9
    from sklearn.decomposition import PCA as _PCA
    Zk = (P[:, keep] - mu[keep]) / sd[keep]
    _p = _PCA(n_components=min(20, Zk.shape[0] - 1, Zk.shape[1]),
              random_state=0).fit(Zk)
    return {
        "scores": [[round(float(v), 4) for v in row] for row in P],
        "dist": [[round(float(v), 3) for v in row] for row in D],
        "edges": [[int(a), int(b)]
                  for a, b in mutual_neighbour_edges(reduced, k=EDGE_K)],
        "projections": projections,
        "medianDistance": round(float(np.median(pdist(reduced))), 3),
        "total": [round(float(v), 3) for v in P.sum(axis=1)],
        "axes": list(desc),
        "axisWord": "descriptor",
        "axisScale": "probability 0-1",
        "basis": {
            "keep": [int(b) for b in keep],
            "mean": [round(float(v), 6) for v in mu],
            "sd": [round(float(v), 6) for v in sd],
            "centre": [round(float(v), 6) for v in _p.mean_],
            "components": [[round(float(v), 6) for v in row]
                           for row in _p.components_],
            "reduced": [[round(float(v), 4) for v in row] for row in reduced],
            "k": 5,
            # Both in 3D map units, so they can be quoted against each other.
            "looMedianError": 1.08,
            "mapMedianPair": round(float(np.median(pdist(
                np.column_stack([projections["mds3_full"]["x"],
                                 projections["mds3_full"]["y"],
                                 projections["mds3_full"]["z"]])))), 2),
            "looNeighbourKeep": 0.63,
        },
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated key=library pairs; first is the reference set")
    ap.add_argument("--out", default="artifacts/viewer_data.json")
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json",
                    help="the percept model that colours the points. The "
                         "baseline file covers 128 molecules of a 410-material "
                         "library, which left two thirds of the atlas grey.")
    ap.add_argument("--no-white-point", action="store_true",
                    help="omit the average-of-everything reference point")
    ap.add_argument("--query", default=None,
                    help="a file written by predict_formula.py --viewer-point: "
                         "its blends are laid out with the library and marked "
                         "as queries. For a local experiment; never passed for "
                         "a published build")
    ap.add_argument("--no-notes", action="store_true",
                    help="omit the perfumer's private smelling notes (use for public builds)")
    args = ap.parse_args()
    global MODELS
    if args.models:
        MODELS = []
        for part in args.models.split(","):
            k, _, v = part.partition("=")
            MODELS.append((k.strip(), v.strip(),
                           {"mpnn": "Chemistry: 386 receptor responses (MPNN)",
                            "gcn": "Chemistry: 386 receptor responses (GCN)"
                            }.get(k.strip(), k.strip())))

    paths = resolve_paths()
    # The first model's library defines the molecule set everything aligns to.
    lib = Library(paths, name=MODELS[0][1])
    entries, logits, scores = lib.load()
    blob = np.load(lib.vectors_npz, allow_pickle=True)
    genes = [str(x) for x in blob["genes"]]

    rec_rows = list(csv.DictReader(open(paths.receptors_tsv, encoding="utf-8"), delimiter="\t"))
    families = [r["family"] for r in rec_rows]

    rec_rows2 = rec_rows
    # Essential oils join the molecule set here, so every lookup keyed on a
    # name picks them up and every one they have no value for falls through to
    # its default -- they have no notes, no inventory row and no measured
    # threshold, which is correct rather than missing.
    nat_ents, nat_percept, _ = load_naturals(MODELS[0][1])
    if not args.no_white_point:
        _pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
        _pm = {n: np.asarray(r, float)
               for n, r in zip(_pj["molecules"], _pj["probs"])}
        _rows = np.stack([_pm[e.name] for e in entries if e.name in _pm])
        _w = WhitePointEntry(len(_rows))
        nat_ents = list(nat_ents) + [_w]
        nat_percept = {**nat_percept, _w.name: _rows.mean(axis=0)}
        print(f"  white point added: the average of {len(_rows)} materials")
    if nat_ents:
        entries = list(entries) + nat_ents
        print(f"  {len(nat_ents)} essential oils built from GC composition")
    query_path = (ROOT / args.query) if args.query else None
    if query_path is not None:
        q_ents, q_percept, _ = load_queries(query_path, MODELS[0][1])
        entries = list(entries) + q_ents
        nat_percept = {**nat_percept, **q_percept}
        print(f"  {len(q_ents)} query blend(s) injected from {args.query}: "
              + ", ".join(e.name for e in q_ents))
    names_ref = [e.name for e in entries]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    classes = sorted({e.cls for e in entries})
    class_colour = {
        c: "#%02x%02x%02x" % tuple(int(255 * v) for v in cmap(i % 20)[:3])
        for i, c in enumerate(classes)
    }

    # Odour families from the perfumer's own notes, for colouring and filtering.
    notes_path = ROOT / "library" / "input" / "impressions.tsv"
    notes, odour_tags = {}, {}
    if notes_path.is_file():
        import csv as _csv

        with open(notes_path, encoding="utf-8") as fh:
            for row in _csv.DictReader(fh, delimiter="	"):
                notes[row["material"]] = row["impression"]
        for e in entries:
            if e.name in notes:
                # Keep only the coarse families: fine tags are too sparse to filter on.
                t = {x for x in tags_for(notes[e.name]) if x.isupper()}
                if t:
                    odour_tags[e.name] = t
    from collections import Counter as _Counter

    freq = _Counter(t for v in odour_tags.values() for t in v)
    odour_list = [t for t, _ in freq.most_common()]
    pal = plt.get_cmap("tab10")
    # The note families were coloured from a categorical palette, which put
    # GREEN on orange and FLORAL on green -- the same complaint that was fixed
    # for the predicted descriptors and missed here. They now reuse the
    # semantic map, so a family reads as the smell it names.
    FAMILY_COLOUR = {
        "GREEN": SEMANTIC_COLOUR["green"],
        "FRUIT": SEMANTIC_COLOUR["fruity"],
        "FLORAL": SEMANTIC_COLOUR["floral"],
        "SWEET": SEMANTIC_COLOUR["sweet"],
        "WOODY": SEMANTIC_COLOUR["woody"],
        "ANIMAL": SEMANTIC_COLOUR["animal"],
    }
    odour_colour = {
        t: FAMILY_COLOUR.get(
            t, SEMANTIC_COLOUR.get(
                t.lower(),
                "#%02x%02x%02x" % tuple(int(255 * v) for v in pal(i % 10)[:3])))
        for i, t in enumerate(odour_list)
    }
    print(f"  odour families: {dict(freq)}")

    # Which materials the percept model was trained on. 228 of the 301 are in
    # GS-LF, so for most of the library a "predicted" descriptor is partly
    # recall of a human label rather than a generalisation. The gain over a
    # base-rate prior survives on unseen materials (+0.052 against +0.061 of
    # AUC), but precision at the top of the list does not (0.284 against
    # 0.377) -- and the top of the list is what the ranking view shows. So the
    # distinction is marked rather than left for the reader to assume.
    in_train: set[str] = set()
    _gslf = ROOT / ("vendor/openpom/openpom/data/curated_datasets/"
                    "curated_GS_LF_merged_4983.csv")
    if _gslf.is_file():
        import pandas as _pd
        from rdkit import Chem as _Chem, RDLogger as _RDL
        _RDL.DisableLog("rdApp.*")
        _df = _pd.read_csv(_gslf)
        _col = ("nonStereoSMILES" if "nonStereoSMILES" in _df.columns
                else _df.columns[0])
        # Match on the InChIKey connectivity block, not the full key. GS-LF
        # stores nonStereoSMILES, so a stereo-specified library material never
        # matches the full key even when the model was trained on its label
        # row -- and the percept model's input fingerprint is stereo-blind
        # anyway, so the two are the same input to it. Matching the full key
        # undercounted by 67 of 303, which understated memorisation badly.
        _keys = set()
        for _s in _df[_col].astype(str):
            _m = _Chem.MolFromSmiles(_s)
            if _m is not None:
                _keys.add(_Chem.MolToInchiKey(_m)[:14])
        for _e in entries:
            _k = getattr(_e, "inchikey", None)
            if _k and _k[:14] in _keys:
                in_train.add(_e.name)
        print(f"  {len(in_train)} of {len(entries)} materials are in the "
              f"percept model's training set")

    # Predicted perceived intensity, if it has been trained. This is a
    # different quantity from everything else on the page: receptor scores are
    # probabilities that a receptor responds and descriptor scores are
    # probabilities that a quality applies, whereas this is how strong the
    # material smells at a stated dilution. It is the only number here that
    # bears on dosing, so the percentile is carried too -- 70 out of 100 means
    # little without knowing where the library sits.
    intensity, intensity_meta = {}, {}
    ipath = ROOT / "artifacts" / "intensity.json"
    if ipath.is_file():
        ij = json.loads(ipath.read_text(encoding="utf-8"))
        vals = np.asarray(ij["predicted_intensity"], dtype=float)
        order = np.argsort(np.argsort(vals))
        pctl = 100.0 * order / max(len(vals) - 1, 1)
        intensity = {n: (round(float(v), 1), round(float(p), 1))
                     for n, v, p in zip(ij["molecules"], vals, pctl)}
        sc = ij.get("scores", {}).get("model (structure + concentration)", {})
        intensity_meta = {
            "concentration": ij.get("predicted_at_concentration"),
            "pearson": sc.get("pearson"),
            "ceiling": (ij.get("noise_ceiling") or {}).get(
                "spearman_brown_full_panel_r"),
            "source": ij.get("target"),
        }
        print(f"  intensity available for {len(intensity)} molecules")

    # Measured or predicted detection threshold, which is the quantity dosing
    # turns on. Carried with its provenance per material, because 28 values are
    # measured, 163 are in-domain predictions, and 110 are extrapolations beyond
    # anything Abraham measured and must not read as equivalent.
    thresh, thresh_meta = {}, {}
    tpath = ROOT / "artifacts" / "threshold.json"
    if tpath.is_file():
        tj = json.loads(tpath.read_text(encoding="utf-8"))
        tv = np.asarray(tj["log_inv_odt"], dtype=float)
        trust = np.array([s_ != "out-of-domain" for s_ in tj["source"]])
        # Percentile is computed over the trustworthy subset only, so an
        # extrapolation cannot shift where everything else appears to sit.
        ref = np.sort(tv[trust]) if trust.any() else np.sort(tv)
        # Only measured values are carried. The predictions validated badly
        # against the perfumer's own use levels -- Spearman 0.14 in-domain and
        # -0.17 out-of-domain, against 0.76 for the measured values -- so the
        # model has no external validity on this library despite scoring 0.82
        # in cross-validation on Abraham's own chemical space. Shipping a
        # number that anticorrelates with practice is worse than shipping none.
        for n, v, src in zip(tj["molecules"], tv, tj["source"]):
            if src != "measured":
                continue
            pc = 100.0 * float(np.searchsorted(ref, v)) / max(len(ref) - 1, 1)
            thresh[n] = (round(float(v), 2), src, round(min(pc, 100.0), 1))
        thresh_meta = {
            "n_measured": len(thresh),
            "predictions_withheld": int(len(tj["molecules"]) - len(thresh)),
            "why_withheld": "model predictions reached Spearman 0.14 against "
                            "the perfumer's own maximum use levels in-domain "
                            "and -0.17 out-of-domain, where the measured "
                            "values reach 0.76",
        }
        print(f"  thresholds: {len(thresh)} measured kept, "
              f"{thresh_meta['predictions_withheld']} predictions withheld")

    # The perfumer's own maximum use level, which is the best dosing signal
    # available: it tracks measured thresholds at rho 0.76 where both exist.
    usage = {}
    ipath = ROOT / "data" / "inventory" / "inventory.csv"
    if ipath.is_file():
        import csv as _c
        for r in _c.DictReader(open(ipath, encoding="utf-8-sig")):
            if r.get("top_pct"):
                usage[r["name"]] = (float(r["top_pct"]), r.get("family", ""),
                                    r.get("volatility_class", ""))
        print(f"  inventory: {len(usage)} materials with a use level")

    # Boiling points, and a volatility tier derived from them where the
    # inventory never assigned one. The cut-points are not chosen here: they
    # are fitted to the owner's own 159 tier assignments by grid search, which
    # lands on top below 216 C, middle below 276 C, base above, and reproduces
    # 74% of those labels. That is three in four, not a rule -- the misses are
    # systematic, because a tier encodes perceived tenacity and a boiling
    # point does not. Hedione boils at 300 and is used as a middle note.
    #
    # So a derived tier is marked as derived and never overwrites a recorded
    # one. That also keeps two bad source values out: Ambroxan is listed at
    # 78 C, which is its melting point rather than its boiling point, and
    # Benzaldehyde at 200 where it boils at 179. Both already carry a tier,
    # so neither is used here.
    BP_TOP, BP_BASE = 216.0, 276.0
    bps = {}
    bpath = ROOT / "data" / "inventory" / "boiling_points.csv"
    if bpath.is_file():
        import csv as _c2
        for r in _c2.DictReader(open(bpath, encoding="utf-8-sig")):
            try:
                bps[r["name"].strip()] = float(r["bp_c"])
            except (ValueError, KeyError):
                pass
        print(f"  boiling points: {len(bps)} fetched values available")

    def tier_of(name):
        """(group, provenance) -- recorded beats derived, derived beats none."""
        rec = VOL_GROUP.get(usage.get(name, [None, None, None])[2])
        if rec:
            return rec, "recorded"
        bp = bps.get(name)
        if bp is None:
            return None, None
        g = "top" if bp < BP_TOP else "middle" if bp < BP_BASE else "base"
        return g, "derived"

    # Measured relative strength, where a blotter session has one. A
    # multiplier above 1 means the material delivers more smell per gram than
    # its own use level implies. This is the only measured quantity on the
    # card other than the 28 real thresholds.
    from strength import load_strength
    strength = load_strength()
    if strength:
        print(f"  strength: {len(strength)} materials measured on a blotter")

    # Predicted GS-LF odour descriptors, from whichever model --percepts names.
    percept_path = ROOT / args.percepts
    percept_tags, percept_list, percept_colour, percept_top = {}, [], {}, {}
    percept_row: dict[str, list[float]] = {}
    percept_all_names: list[str] = []
    percept_meta = {}
    if percept_path.is_file():
        pj = json.loads(percept_path.read_text(encoding="utf-8"))
        desc = pj["descriptors"]
        # Carried into the payload so the page states which model coloured it
        # and how well that model scores, rather than naming one in hard-coded
        # prose that goes stale the next time the model is swapped.
        percept_meta = {
            "name": pj.get("checkpoint", percept_path.stem),
            "n_descriptors": len(desc),
            "auc": pj.get("oof_mean_auc"),
            "similarity": pj.get("similarity_spearman"),
            "config": pj.get("config"),
        }
        pm = {n: np.asarray(r) for n, r in zip(pj["molecules"], pj["probs"])}
        # The oils' percept vectors are combined from their constituents', so
        # they belong in the same map and get tags, a top-6 and a full row by
        # the same rules as everything else.
        pm.update(nat_percept)
        for e in entries:
            r = pm.get(e.name)
            if r is None:
                continue
            order = np.argsort(-r)
            # Tags: anything confident, else just the single best call.
            keep = [int(j) for j in order[:4] if r[j] >= 0.25] or [int(order[0])]
            percept_tags[e.name] = [desc[j] for j in keep]
            percept_top[e.name] = [[desc[int(j)], round(float(r[j]), 3)] for j in order[:6]]
        percept_row = {n: [round(float(v), 3) for v in pm[n]] for n in pm}
        c2 = _Counter(t for v in percept_tags.values() for t in v)
        percept_list = [t for t, n in c2.most_common(22)]
        keepset = set(percept_list)
        for k in percept_tags:
            percept_tags[k] = [t for t in percept_tags[k] if t in keepset] or ["other"]
        if any("other" in v for v in percept_tags.values()):
            percept_list = percept_list + ["other"]
        # Semantic colours: a descriptor should look like what it smells of,
        # so green is green and woody is brown. Anything unlisted falls back to
        # a categorical palette.
        pal2 = plt.get_cmap("tab20")
        percept_colour = {
            t: SEMANTIC_COLOUR.get(
                t, "#%02x%02x%02x" % tuple(int(255 * v) for v in pal2(i % 20)[:3]))
            for i, t in enumerate(percept_list)
        }
        percept_all_names = list(desc)
        print(f"  percept descriptors in legend: {percept_list}")
        print(f"  full descriptor matrix: {len(percept_row)} molecules x "
              f"{len(percept_all_names)}")

    models = {}
    for key, libname, label in MODELS:
        try:
            models[key] = build_model_block(paths, libname, genes, names_ref,
                                            query=query_path,
                                            white=not args.no_white_point)
            models[key]["label"] = label
            # Every model says what its own score columns are, so the page
            # does not have to assume they are receptors.
            models[key]["axes"] = list(genes)
            models[key]["axisWord"] = "receptor"
            models[key]["axisScale"] = "responsiveness 0-1"
            print(f"  model {key:5s} <- artifacts/{libname}")
        except Exception as exc:
            print(f"  model {key:5s} SKIPPED ({exc})")

    if not models:
        raise SystemExit("no libraries could be loaded")

    # The percept model's own space, as a third selectable map. It is built
    # from the same aligned dict the colouring uses, so naturals and the white
    # point are on it too.
    if percept_row and percept_path.is_file():
        try:
            pj2 = json.loads(percept_path.read_text(encoding="utf-8"))
            pm2 = {n: np.asarray(r, dtype=float)
                   for n, r in zip(pj2["molecules"], pj2["probs"])}
            pm2.update(nat_percept)
            models["percept"] = build_percept_block(entries, pm2,
                                                    pj2["descriptors"])
            models["percept"]["label"] = "Smell: 138 predicted odour words"
            print(f"  model percept <- {args.percepts}")
        except Exception as exc:  # noqa: BLE001
            print(f"  model percept SKIPPED ({exc})")

    prov = json.loads(lib.provenance_json.read_text())
    payload = {
        "meta": {
            "n_molecules": len(entries),
            "n_receptors": len(genes),
            "panel_hash": prov["panel_hash"][:16],
            "checkpoint_md5": prov["checkpoint_md5"],
            "percept_model": percept_meta,
            "intensity_model": intensity_meta,
            "threshold_model": thresh_meta,
            "n_strength": sum(1 for e in entries if e.name in strength),
        },
        "receptors": [{"gene": g, "family": f} for g, f in zip(genes, families)],
        "classes": classes,
        "classColour": class_colour,
        "molecules": [
            {"name": e.name, "cls": e.cls, "formula": e.formula,
             "smiles": e.canonical_smiles, "atoms": e.n_atoms,
             "odour": [] if args.no_notes else sorted(odour_tags.get(e.name, [])),
             "percept": percept_tags.get(e.name, []),
             "perceptTop": percept_top.get(e.name, []),
             # The full descriptor row, so the page can rank the library on any
             # single quality. The top-6 list above cannot: asking "what else
             # smells animalic" needs every material's animal score, not each
             # material's own strongest six. Para Cresyl Acetate and Habanolide
             # are 3rd and 6th on that axis while their whole-vector cosine is
             # only +0.26, because 137 other dimensions drown the one they
             # share -- so a per-descriptor ranking answers a question that
             # similarity cannot.
             "perceptAll": percept_row.get(e.name, []),
             "isNatural": int(getattr(e, "is_natural", False)),
             "isQuery": int(getattr(e, "is_query", False)),
             "isWhite": int(getattr(e, "is_white", False)),
             "queryCoverage": getattr(e, "coverage", None),
             "queryIngredients": getattr(e, "ingredients", None),
             "queryPlacement": getattr(e, "placement", None),
             "queryDropped": getattr(e, "dropped", None),
             "collapsedOnto": getattr(e, "collapsed_onto", None),
             "inTrain": int(e.name in in_train),
             "intensity": intensity.get(e.name, [None, None])[0],
             "intensityPct": intensity.get(e.name, [None, None])[1],
             "odt": thresh.get(e.name, [None, None, None])[0],
             "odtSrc": thresh.get(e.name, [None, None, None])[1],
             "odtPct": thresh.get(e.name, [None, None, None])[2],
             # Inventory fields are published: the perfumer asked for the
             # table's data to be included. --no-notes withholds the written
             # smelling impressions and the tags derived from them, and nothing
             # else.
             "strength": (round(strength[e.name]["mult"], 2)
                          if e.name in strength else None),
             "strengthN": (strength[e.name]["n"]
                           if e.name in strength else None),
             "strengthSE": (round(strength[e.name]["se"], 2)
                            if e.name in strength
                            and strength[e.name]["se"] is not None else None),
             "topPct": usage.get(e.name, [None, None, None])[0],
             "family": usage.get(e.name, [None, None, None])[1],
             "volatility": usage.get(e.name, [None, None, None])[2],
             # Collapsed to three, using the same map volatility_profile.py
             # uses and for the same reason it gives there: five tiers over
             # a small set leaves tiers holding one material. On this atlas
             # Top-Mid is 18 points and Mid-Base is 12, so the five-way
             # split fragments exactly where the comment warns.
             "vol": tier_of(e.name)[0],
             "volSrc": tier_of(e.name)[1],
             "note": "" if args.no_notes else notes.get(e.name, "")}
            for e in entries
        ],
        "odours": [] if args.no_notes else odour_list,
        "odourColour": {} if args.no_notes else odour_colour,
        "percepts": percept_list,
        "perceptAllNames": percept_all_names,
        "perceptColour": percept_colour,
        "models": models,
        # The smell map leads, because it is the one that answers the
        # question most people arrive with. The encoder maps follow. The
        # FIRST entry is what the page opens on; MODELS[0] separately
        # decides the molecule set and the naturals file, and that stays
        # the GCN store.
        "modelOrder": ((["percept"] if "percept" in models else [])
                       + [k for k, _, _ in MODELS if k in models]),
    }

    out = ROOT / args.out
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    mb = out.stat().st_size / 1_048_576
    print(f"wrote {out}  ({mb:.2f} MB)")
    print(f"  molecules {len(entries)}  receptors {len(genes)}  models {list(models)}")
    for mk, mv in models.items():
        for pk, pv in mv["projections"].items():
            print(f"  {mk:5s} {pk:14s} shepard {pv['shepard']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
