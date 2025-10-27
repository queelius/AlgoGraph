"""
Examples of graph algorithms in AlgoGraph.

Demonstrates traversal, shortest path, connectivity, and spanning tree algorithms.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.algorithms import (
    # Traversal
    dfs, bfs, topological_sort, find_path,
    # Shortest path
    dijkstra, shortest_path, shortest_path_length,
    # Connectivity
    connected_components, is_connected, is_bipartite,
    # Spanning tree
    kruskal, prim, total_weight
)


def example_traversal():
    """Example: Graph traversal algorithms."""
    print("=" * 60)
    print("Example 1: Graph Traversal")
    print("=" * 60)

    # Build a graph
    g = Graph(
        {Vertex('A'), Vertex('B'), Vertex('C'), Vertex('D'), Vertex('E')},
        {
            Edge('A', 'B'),
            Edge('A', 'C'),
            Edge('B', 'D'),
            Edge('C', 'D'),
            Edge('D', 'E')
        }
    )

    print("\nGraph:")
    print("  A -> B, C")
    print("  B -> D")
    print("  C -> D")
    print("  D -> E")

    print("\n1. Depth-First Search from A:")
    print(f"   {dfs(g, 'A')}")

    print("\n2. Breadth-First Search from A:")
    print(f"   {bfs(g, 'A')}")

    print("\n3. Find path from A to E:")
    path = find_path(g, 'A', 'E')
    print(f"   {' -> '.join(path)}")

    print("\n4. Topological Sort:")
    order = topological_sort(g)
    print(f"   {' -> '.join(order)}")


def example_shortest_path():
    """Example: Shortest path algorithms."""
    print("\n" + "=" * 60)
    print("Example 2: Shortest Paths")
    print("=" * 60)

    # Build weighted graph (road network)
    g = Graph(
        {
            Vertex('NYC', attrs={'pop': 8000000}),
            Vertex('Boston', attrs={'pop': 700000}),
            Vertex('DC', attrs={'pop': 700000}),
            Vertex('Philly', attrs={'pop': 1500000})
        },
        {
            Edge('NYC', 'Boston', weight=215.0),  # miles
            Edge('NYC', 'Philly', weight=95.0),
            Edge('NYC', 'DC', weight=225.0),
            Edge('Philly', 'DC', weight=140.0),
            Edge('Boston', 'Philly', weight=310.0)
        }
    )

    print("\nRoad Network (distances in miles):")
    print("  NYC -> Boston: 215")
    print("  NYC -> Philly: 95")
    print("  NYC -> DC: 225")
    print("  Philly -> DC: 140")
    print("  Boston -> Philly: 310")

    print("\n1. Shortest path from NYC to DC:")
    path = shortest_path(g, 'NYC', 'DC')
    length = shortest_path_length(g, 'NYC', 'DC')
    print(f"   Route: {' -> '.join(path)}")
    print(f"   Distance: {length} miles")

    print("\n2. All distances from NYC (Dijkstra):")
    distances, _ = dijkstra(g, 'NYC')
    for city, dist in sorted(distances.items()):
        if city != 'NYC':
            print(f"   To {city}: {dist} miles")


def example_connectivity():
    """Example: Connectivity analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Connectivity Analysis")
    print("=" * 60)

    # Build social network
    g = Graph(
        {Vertex('Alice'), Vertex('Bob'), Vertex('Charlie'),
         Vertex('Diana'), Vertex('Eve'), Vertex('Frank')},
        {
            Edge('Alice', 'Bob', directed=False),
            Edge('Bob', 'Charlie', directed=False),
            Edge('Diana', 'Eve', directed=False),
            Edge('Eve', 'Frank', directed=False)
        }
    )

    print("\nSocial Network (friendships):")
    print("  Alice -- Bob -- Charlie")
    print("  Diana -- Eve -- Frank")

    print("\n1. Is network connected?", is_connected(g))

    print("\n2. Connected components:")
    components = connected_components(g)
    for i, comp in enumerate(components, 1):
        print(f"   Group {i}: {', '.join(sorted(comp))}")

    # Check bipartiteness (can we 2-color the graph?)
    g2 = Graph(
        {Vertex('A'), Vertex('B'), Vertex('C'), Vertex('D')},
        {
            Edge('A', 'B', directed=False),
            Edge('B', 'C', directed=False),
            Edge('C', 'D', directed=False),
            Edge('D', 'A', directed=False)
        }
    )

    print("\n3. Bipartite check (square graph):")
    is_bip, coloring = is_bipartite(g2)
    print(f"   Is bipartite: {is_bip}")
    if is_bip:
        group1 = [v for v, c in coloring.items() if c == 0]
        group2 = [v for v, c in coloring.items() if c == 1]
        print(f"   Group 1: {', '.join(group1)}")
        print(f"   Group 2: {', '.join(group2)}")


def example_spanning_tree():
    """Example: Minimum spanning tree."""
    print("\n" + "=" * 60)
    print("Example 4: Minimum Spanning Tree")
    print("=" * 60)

    # Build weighted graph (network cable costs)
    g = Graph(
        {Vertex('A'), Vertex('B'), Vertex('C'), Vertex('D'), Vertex('E')},
        {
            Edge('A', 'B', weight=4.0, directed=False),
            Edge('A', 'C', weight=2.0, directed=False),
            Edge('B', 'C', weight=1.0, directed=False),
            Edge('B', 'D', weight=5.0, directed=False),
            Edge('C', 'D', weight=8.0, directed=False),
            Edge('C', 'E', weight=10.0, directed=False),
            Edge('D', 'E', weight=2.0, directed=False)
        }
    )

    print("\nNetwork Cable Costs:")
    for edge in g.edges:
        print(f"  {edge.source} -- {edge.target}: ${edge.weight}")

    print(f"\nTotal if we connect all cables: ${total_weight(g)}")

    print("\n1. Minimum Spanning Tree (Kruskal):")
    mst_kruskal = kruskal(g)
    print(f"   Edges needed: {mst_kruskal.edge_count}")
    print("   Connections:")
    for edge in sorted(mst_kruskal.edges, key=lambda e: e.weight):
        print(f"      {edge.source} -- {edge.target}: ${edge.weight}")
    print(f"   Total cost: ${total_weight(mst_kruskal)}")

    print("\n2. Minimum Spanning Tree (Prim):")
    mst_prim = prim(g)
    print(f"   Total cost: ${total_weight(mst_prim)}")
    print(f"   Same as Kruskal? {total_weight(mst_prim) == total_weight(mst_kruskal)}")


def example_comparison_with_tree():
    """Example: When to use graphs vs trees."""
    print("\n" + "=" * 60)
    print("Example 5: Graphs vs Trees")
    print("=" * 60)

    print("\nUse Trees when:")
    print("  - Hierarchical structure (org charts, file systems)")
    print("  - Single path between any two nodes")
    print("  - Parent-child relationships")
    print("  - No cycles")

    print("\nUse Graphs when:")
    print("  - Network structure (social, roads, internet)")
    print("  - Multiple paths between nodes")
    print("  - Cycles are meaningful (feedback loops)")
    print("  - Need to track edge weights/properties")

    print("\nExample: File system vs Symlinks")
    print("  Tree: Directory hierarchy (no cycles)")
    print("  Graph: With symlinks (can create cycles)")

    # Show conversion
    from AlgoTree import Node, Tree
    from AlgoGraph import tree_to_graph

    print("\nConverting a tree to graph:")
    tree = Tree(Node('root', Node('a'), Node('b', Node('c'))))
    graph = tree_to_graph(tree)
    print(f"  Tree nodes: {len(list(tree.walk()))}")
    print(f"  Graph vertices: {graph.vertex_count}")
    print(f"  Graph edges: {graph.edge_count}")
    print(f"  Graph is connected: {is_connected(graph)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AlgoGraph Algorithm Examples")
    print("=" * 60)

    example_traversal()
    example_shortest_path()
    example_connectivity()
    example_spanning_tree()
    example_comparison_with_tree()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
    AlgoGraph provides comprehensive graph algorithms:

    1. Traversal: DFS, BFS, topological sort, path finding
    2. Shortest Paths: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
    3. Connectivity: Components, bipartite checking, bridges
    4. Spanning Trees: Kruskal, Prim algorithms

    All algorithms work with immutable graph structures
    and integrate seamlessly with AlgoTree.
    """)


if __name__ == '__main__':
    main()
