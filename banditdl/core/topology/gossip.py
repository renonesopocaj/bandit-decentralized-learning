"""Laplacian-based gossip matrix construction."""

import networkx as nx
import numpy as np


def LaplacianGossipMatrix(G):
    """Construct a gossip matrix based on the Laplacian of the graph.

    Args:
        G: networkx Graph

    Returns:
        Gossip mixing matrix W (ndarray)
    """
    max_degree = max([G.degree(node) for node in G.nodes()])
    sorted_nodes = sorted(G.nodes())
    W = np.eye(G.number_of_nodes()) - 1/max_degree * nx.laplacian_matrix(G, nodelist=sorted_nodes).toarray()
    return W
