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

#: First entry is what the viewer opens on. The MPNN leads because it is the
#: better model on both axes measured so far: chemical-class coherence
#: (0.617 vs 0.422) and agreement with perceptual notes (p = 0.006 vs 0.31).
MODELS = [
    ("mpnn", "library_mpnn_s6", "MPNN encoder (uses bond features)"),
    ("gcn", "library", "GCN encoder (no bond features)"),
]


def build_model_block(paths, name, genes_ref, names_ref):
    """Score matrix, distances, links and projections for one encoder."""
    lib = Library(paths, name=name)
    entries, logits, scores = lib.load()
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


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated key=library pairs; first is the reference set")
    ap.add_argument("--out", default="artifacts/viewer_data.json")
    ap.add_argument("--percepts", default="artifacts/percepts_baseline.json")
    ap.add_argument("--no-notes", action="store_true",
                    help="omit the perfumer's private smelling notes (use for public builds)")
    args = ap.parse_args()
    global MODELS
    if args.models:
        MODELS = []
        for part in args.models.split(","):
            k, _, v = part.partition("=")
            MODELS.append((k.strip(), v.strip(),
                           {"mpnn": "MPNN encoder (uses bond features)",
                            "gcn": "GCN encoder (no bond features)"}.get(k.strip(), k.strip())))

    paths = resolve_paths()
    # The first model's library defines the molecule set everything aligns to.
    lib = Library(paths, name=MODELS[0][1])
    entries, logits, scores = lib.load()
    blob = np.load(lib.vectors_npz, allow_pickle=True)
    genes = [str(x) for x in blob["genes"]]

    rec_rows = list(csv.DictReader(open(paths.receptors_tsv, encoding="utf-8"), delimiter="\t"))
    families = [r["family"] for r in rec_rows]

    rec_rows2 = rec_rows
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
    odour_colour = {
        t: "#%02x%02x%02x" % tuple(int(255 * v) for v in pal(i % 10)[:3])
        for i, t in enumerate(odour_list)
    }
    print(f"  odour families: {dict(freq)}")

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
        for n, v, src in zip(tj["molecules"], tv, tj["source"]):
            pc = 100.0 * float(np.searchsorted(ref, v)) / max(len(ref) - 1, 1)
            thresh[n] = (round(float(v), 2), src, round(min(pc, 100.0), 1))
        sc = tj.get("scores", {}).get("model (ECFP4 + descriptors)", {})
        thresh_meta = {
            "pearson": sc.get("pearson"),
            "training_molecules": tj.get("training_molecules"),
            "n_measured": int(sum(1 for x in tj["source"] if x == "measured")),
            "n_out_of_domain": int(sum(1 for x in tj["source"]
                                       if x == "out-of-domain")),
            "domain": tj.get("applicability_domain", {}),
        }
        print(f"  thresholds: {thresh_meta['n_measured']} measured, "
              f"{thresh_meta['n_out_of_domain']} out-of-domain")

    # Predicted GS-LF odour descriptors, from whichever model --percepts names.
    percept_path = ROOT / args.percepts
    percept_tags, percept_list, percept_colour, percept_top = {}, [], {}, {}
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
        for e in entries:
            r = pm.get(e.name)
            if r is None:
                continue
            order = np.argsort(-r)
            # Tags: anything confident, else just the single best call.
            keep = [int(j) for j in order[:4] if r[j] >= 0.25] or [int(order[0])]
            percept_tags[e.name] = [desc[j] for j in keep]
            percept_top[e.name] = [[desc[int(j)], round(float(r[j]), 3)] for j in order[:6]]
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
        print(f"  percept descriptors in legend: {percept_list}")

    models = {}
    for key, libname, label in MODELS:
        try:
            models[key] = build_model_block(paths, libname, genes, names_ref)
            models[key]["label"] = label
            print(f"  model {key:5s} <- artifacts/{libname}")
        except Exception as exc:
            print(f"  model {key:5s} SKIPPED ({exc})")

    if not models:
        raise SystemExit("no libraries could be loaded")

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
             "intensity": intensity.get(e.name, [None, None])[0],
             "intensityPct": intensity.get(e.name, [None, None])[1],
             "odt": thresh.get(e.name, [None, None, None])[0],
             "odtSrc": thresh.get(e.name, [None, None, None])[1],
             "odtPct": thresh.get(e.name, [None, None, None])[2],
             "note": "" if args.no_notes else notes.get(e.name, "")}
            for e in entries
        ],
        "odours": [] if args.no_notes else odour_list,
        "odourColour": {} if args.no_notes else odour_colour,
        "percepts": percept_list,
        "perceptColour": percept_colour,
        "models": models,
        "modelOrder": [k for k, _, _ in MODELS if k in models],
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
