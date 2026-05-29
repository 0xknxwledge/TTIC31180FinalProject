"""Graph visualizations: regime heatmaps, change-network, structure-over-time.

Uses a headless (Agg) matplotlib backend so figures render without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402


def top_edges(delta: np.ndarray, names: list[str], k: int) -> list[tuple[str, str, float]]:
    """Top-k off-diagonal edges of `delta` by |weight|, as (source, target, weight)."""

    d = delta.shape[0]
    cand = [(i, j) for i in range(d) for j in range(d) if i != j]
    cand.sort(key=lambda ij: abs(float(delta[ij])), reverse=True)
    return [(names[i], names[j], float(delta[i, j])) for i, j in cand[:k]]


def change_network(delta: np.ndarray, names: list[str], k: int) -> nx.DiGraph:
    """Directed graph of the top-k change edges (edge attr `weight`)."""

    g = nx.DiGraph()
    g.add_nodes_from(names)
    for src, dst, w in top_edges(delta, names, k):
        g.add_edge(src, dst, weight=w)
    return g


def _heatmap(ax, matrix: np.ndarray, names: list[str], title: str):
    m = np.asarray(matrix, dtype=float)
    lim = float(np.abs(m).max()) or 1.0
    im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title(title, fontsize=10)
    return im


def plot_regime_heatmaps(W_ordinary, W_event, names: list[str], path: str | Path) -> None:
    """3-panel heatmap: W_ordinary, W_event, and Delta = event - ordinary."""

    delta = np.asarray(W_event, dtype=float) - np.asarray(W_ordinary, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, m, title in zip(axes, [W_ordinary, W_event, delta],
                            ["W ordinary", "W event", "Delta (event - ordinary)"]):
        im = _heatmap(ax, m, names, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_change_network(delta, names: list[str], path: str | Path, k: int = 20) -> None:
    """Node-link diagram of the top-k change edges (red = +, blue = -, width = |Delta|)."""

    g = change_network(np.asarray(delta, dtype=float), names, k)
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    wmax = max((abs(w) for w in weights), default=1.0) or 1.0
    fig, ax = plt.subplots(figsize=(11, 11))
    pos = nx.circular_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=350, node_color="#dddddd", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrowsize=10, connectionstyle="arc3,rad=0.08",
        edge_color=["#d62728" if w > 0 else "#1f77b4" for w in weights],
        width=[1.0 + 4.0 * abs(w) / wmax for w in weights],
    )
    ax.set_title(f"Top {k} change edges (Delta_W); red = strengthen, blue = weaken")
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_structure_over_time(matrices: list[np.ndarray], titles: list[str],
                             names: list[str], path: str | Path) -> None:
    """Grid of W heatmaps (e.g. one per contiguous time block) on a shared scale."""

    n = len(matrices)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    vmax = max((float(np.abs(m).max()) for m in matrices), default=1.0) or 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for idx, (m, title) in enumerate(zip(matrices, titles)):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(np.asarray(m, dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"Contemporaneous structure W over time ({len(names)} assets)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
