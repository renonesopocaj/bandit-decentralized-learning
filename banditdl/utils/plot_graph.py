"""Render a 2D weighted graph of the worker network from a saved run.

Two edge-weight modes:

- `sampler_probability`: uses `sampler_probabilities_final.npy`, the per-worker
  final-round sampler distribution. Edge (i, j) weight = (P[i, j] + P[j, i]) / 2.
- `neighbor_disagreement`: uses `pairwise_model_distance_final.npy`. Edge weight
  is `1 / (1 + dist)` so closer (more agreeing) workers get heavier edges —
  matching the bandit reward semantics.

When the run's heterogeneity is `grouped_classes`, nodes are colored by group
and laid out on concentric clusters; otherwise a spring layout is used.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from omegaconf import OmegaConf

WeightSource = Literal["sampler_probability", "neighbor_disagreement"]


def _hydra_cfg(run_dir: pathlib.Path):
    for candidate in (run_dir / ".hydra" / "config.yaml", run_dir.parent / ".hydra" / "config.yaml"):
        if candidate.is_file():
            return OmegaConf.load(candidate)
    return None


def _worker_groups(cfg, n_honest: int) -> np.ndarray | None:
    """Return per-worker group id for `grouped_classes` partitions, else None."""
    if cfg is None:
        return None
    het = cfg.get("heterogeneity", {})
    if str(het.get("method", "")) != "pathological":
        return None
    if str(het.get("partition", "")) != "grouped_classes":
        return None
    nb_groups = int(het.get("nb_groups", 1))
    sizes = [n_honest // nb_groups] * nb_groups
    for i in range(n_honest % nb_groups):
        sizes[i] += 1
    assignment = np.empty(n_honest, dtype=int)
    cursor = 0
    for g, size in enumerate(sizes):
        assignment[cursor : cursor + size] = g
        cursor += size
    return assignment


def _load_weights(
    run_dir: pathlib.Path, weight_source: WeightSource, n_honest: int | None
) -> np.ndarray:
    """Return a symmetric (N, N) honest-worker weight matrix."""
    if weight_source == "sampler_probability":
        path = run_dir / "sampler_probabilities_final.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        prob = np.load(path)  # (n_honest, n_total)
        n = prob.shape[0]
        honest_block = prob[:, :n]  # restrict to honest-honest edges
        return 0.5 * (honest_block + honest_block.T)
    if weight_source == "neighbor_disagreement":
        path = run_dir / "pairwise_model_distance_final.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        dist = np.load(path)
        if n_honest is not None:
            dist = dist[:n_honest, :n_honest]
        return 1.0 / (1.0 + dist)
    raise ValueError(f"Unknown weight_source: {weight_source!r}")


def _grouped_layout(groups: np.ndarray) -> dict[int, tuple[float, float]]:
    """Place each group on its own ring; workers within a group spread evenly."""
    unique_groups = sorted(set(int(g) for g in groups))
    nb_groups = len(unique_groups)
    pos: dict[int, tuple[float, float]] = {}
    for gi, g in enumerate(unique_groups):
        members = [int(i) for i, gg in enumerate(groups) if int(gg) == g]
        cx = np.cos(2 * np.pi * gi / nb_groups)
        cy = np.sin(2 * np.pi * gi / nb_groups)
        if len(members) == 1:
            pos[members[0]] = (cx, cy)
            continue
        for mi, worker_id in enumerate(members):
            theta = 2 * np.pi * mi / len(members)
            r = 0.32
            pos[worker_id] = (cx + r * np.cos(theta), cy + r * np.sin(theta))
    return pos


def _node_colors(groups: np.ndarray | None, n: int):
    if groups is None:
        return ["tab:blue"] * n
    cmap = cm.get_cmap("tab10")
    return [cmap(int(g) % 10) for g in groups]


def plot_clustering_graph(
    run_dir: pathlib.Path,
    output_path: pathlib.Path,
    *,
    weight_source: WeightSource = "sampler_probability",
    top_edges_per_node: int | None = None,
    layout: Literal["auto", "spring", "group"] = "auto",
    title: str | None = None,
) -> pathlib.Path:
    """Render and save the weighted clustering graph for `run_dir`.

    `top_edges_per_node` keeps only the k strongest outgoing edges per node so
    the plot stays readable for dense N. Pass None to keep all edges.
    """
    run_dir = pathlib.Path(run_dir)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _hydra_cfg(run_dir)
    n_honest = None
    if cfg is not None:
        nb_workers = int(OmegaConf.select(cfg, "topology.nodes"))
        nb_byz = int(OmegaConf.select(cfg, "adversary.byzcount") or 0)
        n_honest = nb_workers - nb_byz

    weights = _load_weights(run_dir, weight_source, n_honest)
    n = weights.shape[0]
    np.fill_diagonal(weights, 0.0)

    if top_edges_per_node is not None and top_edges_per_node < n - 1:
        keep = np.zeros_like(weights, dtype=bool)
        for i in range(n):
            order = np.argsort(weights[i])[::-1]
            keep[i, order[:top_edges_per_node]] = True
        keep = keep | keep.T
        weights = weights * keep

    graph = nx.from_numpy_array(weights)
    groups = _worker_groups(cfg, n) if cfg is not None else None
    if layout == "group" or (layout == "auto" and groups is not None):
        if groups is None:
            pos = nx.spring_layout(graph, seed=0, weight="weight")
        else:
            pos = _grouped_layout(groups)
    else:
        pos = nx.spring_layout(graph, seed=0, weight="weight")

    edges = list(graph.edges(data="weight"))
    edge_weights = np.array([w for _, _, w in edges]) if edges else np.array([])
    if edges and edge_weights.max() > 0:
        norm = mcolors.Normalize(vmin=0.0, vmax=float(edge_weights.max()))
        edge_colors = cm.viridis(norm(edge_weights))
        edge_widths = 0.4 + 3.5 * edge_weights / edge_weights.max()
    else:
        edge_colors = "lightgray"
        edge_widths = 0.5
        norm = None

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.85,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=_node_colors(groups, n),
        node_size=260,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)

    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(_edge_label(weight_source))

    if title is None:
        title = f"{run_dir.name} — {weight_source}"
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _edge_label(weight_source: WeightSource) -> str:
    if weight_source == "sampler_probability":
        return "Symmetrized sampler probability (final round)"
    return "1 / (1 + final model L2 distance)"
