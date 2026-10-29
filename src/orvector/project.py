"""Project the receptor-activation library to 2D and render a labelled map.

Preprocessing matters more than the choice of projection here. The raw vectors
carry a dominant shared component -- every molecule correlates ~0.95 with the
panel mean, and after centring the first principal component still explains the
large majority of the variance. That axis is essentially potency/size, so any
projection of the raw vectors mostly draws "how strongly does this molecule hit
receptors at all" and buries the chemical-quality structure.

The pipeline is therefore:

1. logits, not sigmoid scores (the scores are saturated near zero, so distances
   on them are decided by a handful of receptors);
2. receptor-wise z-scoring across the library, which removes the per-receptor
   baseline so broadly-tuned generalists stop dominating every distance;
3. optional removal of the first principal component (the potency axis);
4. reduction to a modest number of PCs before any nonlinear step, since with
   386 receptors and ~10^2 molecules we are firmly in the n < p regime.

Every map reports a **Shepard rank correlation** between the 2D pairwise
distances and the true high-dimensional ones. That number is what separates a
result from a decoration, so it is printed on the figure itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Projection:
    """A 2D embedding plus the diagnostics needed to judge it."""

    coords: np.ndarray           # (n, 2)
    method: str
    shepard: float               # Spearman(2D distances, high-dim distances)
    stress: float | None         # MDS only
    explained: np.ndarray | None  # PCA only
    loadings: list[tuple[str, list[tuple[str, float]]]] | None
    note: str = ""
    #: Set by :func:`project` after construction.
    high: np.ndarray | None = None
    edges: list[tuple[int, int]] | None = None
    reliability: np.ndarray | None = None


def preprocess(
    logits: np.ndarray,
    *,
    drop_pc1: bool = False,
    n_components: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reduced, zscored)`` representations of the library."""
    from sklearn.decomposition import PCA

    X = np.asarray(logits, dtype=np.float64)
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 molecules to project.")

    # Receptor-wise z-scoring. Receptors with no variance across the library
    # carry no information and would produce division by zero.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    keep = (sd > 1e-9).ravel()
    Z = (X[:, keep] - mu[:, keep]) / sd[:, keep]

    if drop_pc1:
        p1 = PCA(n_components=1).fit(Z)
        Z = Z - p1.inverse_transform(p1.transform(Z))

    k = int(min(n_components, Z.shape[0] - 1, Z.shape[1]))
    reduced = PCA(n_components=k, random_state=0).fit_transform(Z)
    return reduced, Z


def _pairwise(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    from scipy.spatial.distance import pdist

    return pdist(X, metric=metric)


def shepard_correlation(high: np.ndarray, coords: np.ndarray) -> float:
    """Spearman correlation between high-dim and 2D pairwise distances."""
    from scipy.stats import spearmanr

    d_hi = _pairwise(high)
    d_lo = _pairwise(coords)
    return float(spearmanr(d_hi, d_lo).statistic)


def mutual_neighbour_edges(high: np.ndarray, k: int = 4) -> list[tuple[int, int]]:
    """Pairs that are mutually among each other's ``k`` nearest in the TRUE space.

    Drawn on the map so genuine relationships stay visible even where the 2D
    placement distorts them. A long edge is distortion made explicit rather
    than hidden.
    """
    from scipy.spatial.distance import squareform, pdist

    D = squareform(pdist(high))
    np.fill_diagonal(D, np.inf)
    nn = {i: set(np.argsort(D[i])[:k]) for i in range(D.shape[0])}
    return [(i, j) for i in nn for j in nn[i] if i < j and i in nn[j]]


def neighbour_preservation(high: np.ndarray, coords: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-point Jaccard overlap of true vs plotted ``k``-nearest neighbours.

    Global Shepard correlation can look excellent while individual points are
    badly placed, so this reports placement reliability point by point.
    """
    from scipy.spatial.distance import squareform, pdist

    Dh = squareform(pdist(high))
    Dl = squareform(pdist(coords))
    np.fill_diagonal(Dh, np.inf)
    np.fill_diagonal(Dl, np.inf)
    out = np.zeros(Dh.shape[0])
    for i in range(Dh.shape[0]):
        a = set(np.argsort(Dh[i])[:k])
        b = set(np.argsort(Dl[i])[:k])
        out[i] = len(a & b) / len(a | b)
    return out


def _loadings(Z: np.ndarray, genes: list[str], keep_mask: np.ndarray, top: int = 6):
    from sklearn.decomposition import PCA

    p = PCA(n_components=2, random_state=0).fit(Z)
    kept_genes = [g for g, k in zip(genes, keep_mask) if k]
    out = []
    for axis in range(2):
        comp = p.components_[axis]
        idx = np.argsort(-np.abs(comp))[:top]
        out.append((f"PC{axis + 1}", [(kept_genes[i], float(comp[i])) for i in idx]))
    return out


def project(
    logits: np.ndarray,
    genes: list[str],
    *,
    method: str = "pca",
    drop_pc1: bool = False,
    n_components: int = 20,
    random_state: int = 0,
    edge_k: int = 4,
) -> Projection:
    """Project to 2D by ``method`` ∈ {pca, mds, umap, tsne}."""
    from sklearn.decomposition import PCA

    reduced, Z = preprocess(logits, drop_pc1=drop_pc1, n_components=n_components)
    n = reduced.shape[0]

    sd = np.asarray(logits, dtype=np.float64).std(axis=0)
    keep_mask = sd > 1e-9
    loadings = _loadings(Z, genes, keep_mask)
    explained = None
    stress = None

    if method == "pca":
        p = PCA(n_components=2, random_state=random_state)
        coords = p.fit_transform(Z)
        explained = p.explained_variance_ratio_
    elif method == "mds":
        from sklearn.manifold import MDS

        # Metric MDS directly minimises the discrepancy between 2D distances
        # and the distances we care about, which is the literal objective of
        # "preserve similarity as faithfully as possible on a plane".
        m = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=8,
            max_iter=1000,
            normalized_stress="auto",
        )
        from scipy.spatial.distance import squareform

        D = squareform(_pairwise(reduced))
        coords = m.fit_transform(D)
        stress = float(m.stress_)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
        coords = TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            random_state=random_state, max_iter=1500,
        ).fit_transform(reduced)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit(
                "umap-learn is not installed. Install it or use --method pca|mds."
            ) from exc
        coords = umap.UMAP(
            n_components=2,
            n_neighbors=int(max(5, min(20, n / 6))),
            min_dist=0.25,
            random_state=random_state,
        ).fit_transform(reduced)
    else:
        raise ValueError(f"Unknown method {method!r}")

    coords = np.asarray(coords, dtype=np.float64)
    proj = Projection(
        coords=coords,
        method=method,
        shepard=shepard_correlation(reduced, coords),
        stress=stress,
        explained=explained,
        loadings=loadings,
        note="PC1 (potency axis) removed" if drop_pc1 else "all components retained",
    )
    proj.high = reduced
    proj.edges = mutual_neighbour_edges(reduced, k=edge_k)
    proj.reliability = neighbour_preservation(reduced, coords, k=10)
    return proj


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render(
    proj: Projection,
    names: list[str],
    classes: list[str],
    potency: np.ndarray,
    out_path,
    *,
    title: str = "Olfactory receptor activation space",
    figsize: tuple[float, float] = (22.0, 15.0),
    dpi: int = 200,
    label_all: bool = True,
) -> None:
    """Render a large labelled scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    coords = proj.coords
    uniq = sorted(set(classes))
    cmap = plt.get_cmap("tab20" if len(uniq) <= 20 else "gist_ncar")
    colour = {c: cmap(i % cmap.N if len(uniq) <= 20 else i / max(1, len(uniq) - 1))
              for i, c in enumerate(uniq)}

    # Marker area scaled by total activation, so size reads as potency.
    p = np.asarray(potency, dtype=float)
    span = p.max() - p.min()
    sizes = 90 + 520 * ((p - p.min()) / span if span > 0 else np.zeros_like(p))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Keep the side panel at a roughly fixed physical width so enlarging the
    # figure gives the extra room to the plot rather than to the legend.
    side_in = min(6.0, max(4.0, figsize[0] * 0.14))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[figsize[0] - side_in, side_in], wspace=0.02
    )
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])
    side.axis("off")

    # True mutual-nearest-neighbour links, drawn underneath the points. A long
    # link means the plane has pulled a genuine pair apart: distortion you can
    # see rather than distortion you cannot.
    if proj.edges:
        for i, j in proj.edges:
            ax.plot(
                [coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color="0.55", lw=0.7, alpha=0.5, zorder=2,
            )

    rel = proj.reliability if proj.reliability is not None else np.ones(len(names))
    for c in uniq:
        m = np.array([cl == c for cl in classes])
        # Well-placed points get a solid dark outline; poorly-placed ones a
        # pale outline, so unreliable positions are visually discounted.
        good = m & (rel >= 0.25)
        poor = m & (rel < 0.25)
        if good.any():
            ax.scatter(
                coords[good, 0], coords[good, 1], s=sizes[good], c=[colour[c]],
                edgecolors="black", linewidths=0.6, alpha=0.9, zorder=3, label=c,
            )
        if poor.any():
            ax.scatter(
                coords[poor, 0], coords[poor, 1], s=sizes[poor], c=[colour[c]],
                edgecolors="0.55", linewidths=1.6, linestyle=":", alpha=0.55,
                zorder=3, label=None if good.any() else c,
            )

    texts = []
    if label_all:
        for i, nm in enumerate(names):
            texts.append(
                ax.annotate(
                    nm, (coords[i, 0], coords[i, 1]),
                    fontsize=7.4, zorder=4,
                    ha="center", va="center",
                )
            )
        try:
            from adjustText import adjust_text

            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.55, alpha=0.85),
                expand=(1.18, 1.3),
                force_text=(0.4, 0.55),
                max_move=28,
            )
        except Exception as exc:  # pragma: no cover - cosmetic only
            print(f"[plot] label adjustment skipped: {exc}")

    xlab, ylab = "dimension 1", "dimension 2"
    if proj.explained is not None:
        xlab = f"PC1  ({proj.explained[0]:.1%} of variance)"
        ylab = f"PC2  ({proj.explained[1]:.1%} of variance)"
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title(
        f"{title}\n{proj.method.upper()}  |  Shepard rank corr = {proj.shepard:.3f}"
        f"  |  {proj.note}",
        fontsize=14, pad=14,
    )
    ax.grid(alpha=0.18, zorder=0)
    ax.axhline(0, color="0.8", lw=0.8, zorder=1)
    ax.axvline(0, color="0.8", lw=0.8, zorder=1)

    # ---- side panel: legend, interpretation, method notes ----
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=9,
               markerfacecolor=colour[c], markeredgecolor="black", label=c)
        for c in uniq
    ]
    leg = side.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
        frameon=True, fontsize=9, title="chemical class", title_fontsize=10,
    )
    side.add_artist(leg)

    lines = []
    if proj.loadings:
        lines.append("Axis drivers (receptor loadings)")
        for axis, items in proj.loadings:
            lines.append(f"  {axis}:")
            for gene, w in items:
                lines.append(f"     {gene:9s} {w:+.3f}")
        lines.append("")
    quality = ("faithful" if proj.shepard >= 0.9 else
               "usable" if proj.shepard >= 0.75 else
               "indicative only" if proj.shepard >= 0.6 else "unreliable")
    med_rel = float(np.median(rel))
    n_poor = int((rel < 0.25).sum())
    lines += [
        "Reading this map",
        f"  Shepard rank corr {proj.shepard:.3f} ({quality}):",
        "  how well 2D distances reproduce",
        "  the 386-dim ones. This is a GLOBAL",
        "  measure and oversells local detail.",
        f"  Median local neighbour overlap is",
        f"  only {med_rel:.2f} (top-10).",
        "",
        "  Grey links join pairs that are",
        "  mutual nearest neighbours in the",
        "  TRUE 386-dim space. A long link =",
        "  visible 2D distortion.",
        f"  Dotted pale outline = badly placed",
        f"  ({n_poor} points); ignore their",
        "  position, trust their links.",
        "  Marker size = total activation",
        "  (a potency proxy).",
        "",
        "Pipeline",
        "  raw logits",
        "  -> receptor-wise z-score",
        f"  -> {proj.note}",
        f"  -> {proj.method.upper()} to 2D",
        "",
        "Caveats",
        "  Stereochemistry is invisible to",
        "  the model, so enantiomers and",
        "  E/Z isomers coincide exactly.",
        "  Scores are P(responsive) from",
        "  binary screens, not intensity",
        "  at any given concentration.",
    ]
    if proj.stress is not None:
        lines.insert(0, f"MDS normalised stress: {proj.stress:.4f}")
        lines.insert(1, "")
    side.text(
        0.0, 0.0, "\n".join(lines), transform=side.transAxes,
        fontsize=8.4, va="bottom", ha="left", family="monospace",
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
