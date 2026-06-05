"""Fixed-graph generation and Byzantine robustness checks."""

import random

import networkx as nx


def generate_connected_graph(n, e, seed=None):
    """Generate a random connected graph with n nodes and e edges.

    Args:
        n: Number of nodes
        e: Number of edges
        seed: Random seed

    Returns:
        Connected networkx Graph
    """
    if e < n - 1 or e > n * (n - 1) // 2:
        raise ValueError("Invalid number of edges for a simple connected graph.")

    # Generate a random spanning tree
    G = nx.random_labeled_tree(n, seed=seed)
    G = nx.Graph(G)  # Ensure it's an undirected graph

    # Add extra edges randomly until the edge budget is met
    existing_edges = set(G.edges())
    possible_edges = set(
        (i, j) for i in range(n) for j in range(i + 1, n)
    ) - existing_edges

    extra_edges_needed = e - (n - 1)
    random.seed(seed)
    extra_edges = random.sample(sorted(possible_edges), extra_edges_needed)
    G.add_edges_from(extra_edges)

    return G


def graph_byz_robust(G, byz):
    """Verify if a graph is Byzantine robust.

    A graph is Byzantine robust if no honest node has a Byzantine majority.

    Args:
        G: networkx Graph
        byz: List of Byzantine node indices

    Returns:
        Tuple (is_robust, corrupted_nodes)
    """
    corrupted_nodes = []
    is_robust = True
    for node in G.nodes():
        if node in byz:
            continue
        neighbors = list(G.neighbors(node))
        byz_neighbors = len([n for n in neighbors if n in byz])
        honest_neighbors = len(neighbors) - byz_neighbors + 1
        if byz_neighbors >= honest_neighbors:
            corrupted_nodes.append(node)
            is_robust = False
    return is_robust, corrupted_nodes
