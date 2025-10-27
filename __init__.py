"""
AlgoGraph - Graph Data Structures and Algorithms

A companion library to AlgoTree for working with graph structures.
Provides immutable graph operations, algorithms, and interoperability with trees.

Core Components:
- Vertex: Immutable graph vertices with attributes
- Edge: Immutable directed/undirected edges with weights
- Graph: Immutable graph container with algorithms
- interop: Conversion functions between trees and graphs

Example:
    >>> from AlgoGraph import Vertex, Edge, Graph
    >>> v1, v2 = Vertex('A'), Vertex('B')
    >>> e = Edge('A', 'B', weight=5.0)
    >>> g = Graph({v1, v2}, {e})
    >>> g.has_edge('A', 'B')
    True

Integration with AlgoTree:
    >>> from AlgoTree import Node, Tree
    >>> from AlgoGraph import tree_to_graph, graph_to_tree
    >>> tree = Tree(Node('root', Node('child1'), Node('child2')))
    >>> graph = tree_to_graph(tree)
    >>> recovered_tree = graph_to_tree(graph, 'root')
"""

from .vertex import Vertex
from .edge import Edge
from .graph import Graph
from .interop import (
    tree_to_graph,
    node_to_graph,
    graph_to_tree,
    flat_dict_to_graph,
    graph_to_flat_dict,
    tree_to_flat_dict,
    flat_dict_to_tree,
)
from .serialization import (
    graph_to_json,
    graph_from_json,
    save_graph,
    load_graph,
)

# Algorithms are available in submodule
# from AlgoGraph.algorithms import dfs, bfs, dijkstra, etc.

__version__ = "1.0.0"

__all__ = [
    'Vertex',
    'Edge',
    'Graph',
    'tree_to_graph',
    'node_to_graph',
    'graph_to_tree',
    'flat_dict_to_graph',
    'graph_to_flat_dict',
    'tree_to_flat_dict',
    'flat_dict_to_tree',
    'graph_to_json',
    'graph_from_json',
    'save_graph',
    'load_graph',
]
