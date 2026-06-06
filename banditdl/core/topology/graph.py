"""Graph and communication network implementation."""

import networkx as nx
import numpy as np
import torch


class CommunicationNetwork(nx.Graph):
    """Communication Network of the Honest subgraph.

    The implementation here provides gossip derived from the Laplacian matrix of the graph.
    """

    def __init__(self, incoming_graph_data, weights_method="metropolis", device="cpu"):
        super().__init__(incoming_graph_data)
        self.construct_weights(weights_method)

        self.device = device
        self.laplacian = nx.laplacian_matrix(self).astype(np.float32).todense()
        self.adjacency_matrix = nx.adjacency_matrix(self).astype(np.float32).todense()

    def weights(self, j):
        """Return the weights associated with neighbors of node j from the adjacency matrix."""
        res = list(np.asarray(self.adjacency_matrix[j,:]).flatten())
        weight_sum = sum(res)
        if weight_sum < 1:
            res[j] = 1 - weight_sum
        elif weight_sum > 1:
            # Normalize to 1
            res = [w / weight_sum for w in res]
        return torch.tensor(res, dtype=torch.float32).to(self.device)

    def construct_weights(self, weights_method):
        """Construct edge weights based on the specified method."""
        if weights_method == "metropolis":
            for e in self.edges:
                self.edges[e]['weight'] = 1 / (max(self.degree[e[0]], self.degree[e[1]]) + 1)
        elif weights_method == "unitary":
            for e in self.edges:
                self.edges[e]['weight'] = 1


graph_types = ["fully_connected", "Erdos_Renyi", "lattice", "two_worlds", "random_geometric"]


def create_graph(name, size, hyper=None, seed=None, *args, **kwargs):
    """Create a communication graph of the specified type.

    Args:
        name: Graph type (fully_connected, Erdos_Renyi, lattice, two_worlds, random_geometric)
        size: Number of nodes
        hyper: Hyperparameter (depends on graph type)
        seed: Random seed
        *args, **kwargs: Additional arguments passed to CommunicationNetwork

    Returns:
        CommunicationNetwork instance
    """
    if name == "fully_connected":
        net = nx.complete_graph(size)
    elif name == "Erdos_Renyi":
        net = nx.erdos_renyi_graph(size, hyper, seed=seed)
    elif name == "lattice":
        net = nx.grid_graph(dim=[int(size**(1/hyper)) for i in range(hyper)], periodic=True)
    elif name == "two_worlds":
        c1 = nx.complete_graph(size//2)
        c2 = nx.complete_graph(size - size//2)
        c2 = nx.relabel_nodes(c2, {i: i+size//2 for i in range(size-size//2)}, copy=False)
        net = nx.union(c1, c2)

        for i in range(size//2):
            for k in range(int(hyper)):
                net.add_edge(i, (i + k) % (size//2) + size//2)
    elif name == "random_geometric":
        net = nx.random_geometric_graph(size, radius=hyper, seed=seed, dim=2, p=2)
    else:
        raise ValueError(name + " is not a possible graph")

    return CommunicationNetwork(net, *args, **kwargs)
