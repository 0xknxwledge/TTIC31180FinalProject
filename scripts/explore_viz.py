"""Prototype time-resolved visualizations of how the contemporaneous DAG W
evolves across the ~2-year panel.

Fits one contemporaneous W per ~2-month contiguous time block (the
"duplicate-as-2-regimes, gamma=0" trick) and renders two NEW figures:

  1) outputs/figures/edge_persistence.png   -- edge persistence heatmap
        rows  = union of each block's top-k edges, cols = blocks,
        color = signed W weight; shows which edges persist vs are transient.
  2) outputs/figures/top_edges_per_block.png -- per-block top-edge node-link
        diagrams (one circular graph per block) so the reader can read off the
        dominant contemporaneous links and watch them rotate through time.

These do NOT overwrite existing figures and live only in this script (no
edits to frtdbn/viz.py). Run:

  python scripts/explore_viz.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.model import FitConfig, fit_fr_tdbn  # noqa: E402
from frtdbn.panel import DEFAULT_PANEL, build_lagged_design  # noqa: E402
from frtdbn.splitting import time_block_indices  # noqa: E402
from frtdbn.viz import top_edges  # noqa: E402

FIGDIR = Path("outputs/figures")
N_BLOCKS = 8           # ~2-month blocks over the ~2-year panel
TOP_K_PER_BLOCK = 8    # edges used per block for the union / per-block graphs


# --- single-W block fit (duplicate-as-2-regimes, gamma=0) ---------------------

def _block_config() -> FitConfig:
    return FitConfig(
        p=1, solver="admm", loss="student_t",
        gamma_w=0.0, gamma_a=0.0, lambda_w=0.05, lambda_a=0.05,
        lbfgs_max_iter=20, outer_max_iter=6, t_admm=1, h_tol=1e-4, seed=0,
    )


def fit_block_W(target: np.ndarray, lags: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Fit one contemporaneous W on the rows `idx` of the panel."""

    tb, lb = target[idx], [lag[idx] for lag in lags]
    fb = fit_fr_tdbn([tb, tb], [lb, lb], _block_config())
    return fb.W[0]


def fit_blocks(target, lags, ts, n_blocks: int):
    """Return (W_blocks, block_labels) for `n_blocks` contiguous time blocks."""

    W_blocks, labels = [], []
    for idx in time_block_indices(target.shape[0], n_blocks):
        W_blocks.append(fit_block_W(target, lags, idx))
        a, b = pd.Timestamp(ts[idx[0]]).date(), pd.Timestamp(ts[idx[-1]]).date()
        labels.append(f"{a}\n..{b}")
    return W_blocks, labels


# --- figure 1: edge-persistence heatmap --------------------------------------

def plot_edge_persistence(W_blocks, block_labels, names, path, top_k: int) -> None:
    """Rows = union of per-block top-k edges, cols = blocks, color = signed W.

    Edges are ordered by how persistent they are (mean |weight| across blocks),
    so the eye separates always-on backbone edges from one-off transient ones.
    """

    # Union of each block's top-k off-diagonal edges (by |weight|).
    edge_set: dict[tuple[int, int], None] = {}
    d = len(names)
    for W in W_blocks:
        cand = [(i, j) for i in range(d) for j in range(d) if i != j]
        cand.sort(key=lambda ij: abs(float(W[ij])), reverse=True)
        for ij in cand[:top_k]:
            edge_set.setdefault(ij, None)
    edges = list(edge_set)

    # Weight matrix: edges x blocks.
    M = np.array([[float(W[i, j]) for W in W_blocks] for (i, j) in edges])
    persistence = np.abs(M).mean(axis=1)          # mean |weight| across blocks
    order = np.argsort(-persistence)              # most persistent at top
    M, edges = M[order], [edges[i] for i in order]
    row_labels = [f"{names[j]} -> {names[i]}" for (i, j) in edges]  # j drives i

    lim = float(np.abs(M).max()) or 1.0
    fig, ax = plt.subplots(figsize=(1.15 * len(W_blocks) + 4, 0.30 * len(edges) + 2.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(block_labels)))
    ax.set_xticklabels(block_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xlabel("time block")
    ax.set_title(
        f"Edge persistence: signed W weight across {len(W_blocks)} time blocks\n"
        f"(rows = union of each block's top-{top_k} edges, sorted by mean |weight|)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="W weight (j -> i)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- figure 2: per-block top-edge node-link grid -----------------------------

def plot_top_edges_per_block(W_blocks, block_labels, names, path, top_k: int) -> None:
    """One circular node-link diagram per block of its top-k edges.

    Shared node layout + shared edge-width scale across panels, so the reader
    compares structure across time at a glance (red = +, blue = -)."""

    n = len(W_blocks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    # Shared circular layout (one graph with all nodes) + shared width scale.
    g_layout = nx.DiGraph()
    g_layout.add_nodes_from(names)
    pos = nx.circular_layout(g_layout)
    wmax = max((abs(w) for W in W_blocks for _, _, w in top_edges(W, names, top_k)), default=1.0) or 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for idx, (W, label) in enumerate(zip(W_blocks, block_labels)):
        ax = axes[idx // cols][idx % cols]
        g = nx.DiGraph()
        g.add_nodes_from(names)
        for src, dst, w in top_edges(W, names, top_k):
            g.add_edge(src, dst, weight=w)
        weights = [g[u][v]["weight"] for u, v in g.edges()]
        nx.draw_networkx_nodes(g, pos, node_size=140, node_color="#e8e8e8",
                               edgecolors="#999999", linewidths=0.4, ax=ax)
        nx.draw_networkx_labels(g, pos, font_size=5.5, ax=ax)
        nx.draw_networkx_edges(
            g, pos, ax=ax, arrowsize=7, connectionstyle="arc3,rad=0.08",
            edge_color=["#d62728" if w > 0 else "#1f77b4" for w in weights],
            width=[0.6 + 3.5 * abs(w) / wmax for w in weights],
        )
        ax.set_title(label.replace("\n", " "), fontsize=8)
        ax.axis("off")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(
        f"Top-{top_k} contemporaneous edges per time block "
        f"(red = +, blue = -, width = |W|)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    target, lags, names, ts = build_lagged_design(frames, DEFAULT_PANEL)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print(f"panel: {target.shape[0]} rows x {len(names)} assets, "
          f"{pd.Timestamp(ts[0]).date()} .. {pd.Timestamp(ts[-1]).date()}")
    print(f"fitting one W per block over {N_BLOCKS} blocks ...")
    W_blocks, block_labels = fit_blocks(target, lags, ts, N_BLOCKS)

    p1 = FIGDIR / "edge_persistence.png"
    p2 = FIGDIR / "top_edges_per_block.png"
    plot_edge_persistence(W_blocks, block_labels, names, p1, TOP_K_PER_BLOCK)
    plot_top_edges_per_block(W_blocks, block_labels, names, p2, TOP_K_PER_BLOCK)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
