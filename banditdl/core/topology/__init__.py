"""Topology module for network communication graphs and gossip matrices."""

from .fxgraph import generate_connected_graph, graph_byz_robust
from .gossip import LaplacianGossipMatrix
from .graph import CommunicationNetwork, create_graph

__all__ = [
    "CommunicationNetwork",
    "LaplacianGossipMatrix",
    "create_graph",
    "generate_connected_graph",
    "graph_byz_robust",
]
