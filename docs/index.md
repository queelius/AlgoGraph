# AlgoGraph

**Immutable graph data structures and algorithms library with optional AlgoTree integration**

AlgoGraph is a powerful, functional graph library for Python that provides immutable data structures and a comprehensive collection of graph algorithms. Built with principles of immutability and composability, AlgoGraph makes it easy to work with complex graph structures while maintaining code clarity and correctness.

## Key Features

### Immutable Data Structures
- **Vertex**: Immutable graph vertices with customizable attributes
- **Edge**: Immutable directed/undirected edges with weights and attributes
- **Graph**: Immutable graph container with efficient operations

### 30+ Graph Algorithms
Organized into four comprehensive categories:

- **Traversal**: DFS, BFS, topological sort, cycle detection, path finding
- **Shortest Path**: Dijkstra, Bellman-Ford, Floyd-Warshall, A* search
- **Connectivity**: Connected components, strongly connected components, bipartite checking, bridges, articulation points
- **Spanning Tree**: Kruskal's algorithm, Prim's algorithm, MST operations

### Interactive Shell
A VFS-like interface for exploring graphs interactively:

- Navigate graphs like a file system (`cd`, `ls`, `pwd`)
- Query paths, components, and neighbors
- Load and save graphs in JSON format
- Tab completion and command history
- Support for absolute paths and quoted vertex names

### Serialization & Interoperability
- JSON import/export for persistent storage
- Optional AlgoTree integration for tree-graph conversions
- Flat dictionary representation for data exchange

## Quick Example

```python
from AlgoGraph import Vertex, Edge, Graph
from AlgoGraph.algorithms import dijkstra, connected_components

# Create vertices
vertices = {
    Vertex('A', attrs={'city': 'Boston'}),
    Vertex('B', attrs={'city': 'New York'}),
    Vertex('C', attrs={'city': 'Philadelphia'}),
}

# Create weighted edges
edges = {
    Edge('A', 'B', weight=215),  # miles
    Edge('B', 'C', weight=95),
    Edge('A', 'C', weight=310),
}

# Build graph
graph = Graph(vertices, edges)

# Find shortest path
distances, predecessors = dijkstra(graph, 'A')
print(f"Distance from A to C: {distances['C']} miles")

# Check connectivity
components = connected_components(graph)
print(f"Connected components: {len(components)}")
```

## Why AlgoGraph?

### Functional & Immutable
AlgoGraph follows functional programming principles with immutable data structures. Every operation returns a new graph, preserving the original. This makes your code easier to reason about and prevents subtle bugs from shared mutable state.

### Clean Separation of Concerns
Graph data structures are separate from algorithms. This composable design lets you easily combine different algorithms and operations while keeping your code modular and testable.

### Production Ready
- Comprehensive test coverage
- Well-documented API
- Type hints throughout
- Performance optimized
- Battle-tested algorithms

### Developer Friendly
- Clear, consistent API design
- Rich examples and cookbook recipes
- Interactive shell for exploration
- Helpful error messages
- Extensive documentation

## What's Next?

- **New to AlgoGraph?** Start with the [Installation Guide](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md)
- **Want to learn the concepts?** Read the [User Guide](user-guide/core-concepts.md)
- **Looking for specific functionality?** Check the [API Reference](api/vertex.md)
- **Need examples?** Browse the [Examples & Cookbook](examples/social-networks.md)
- **Want to explore interactively?** Try the [Interactive Shell](shell/overview.md)

## Design Philosophy

AlgoGraph is built on three core principles:

1. **Immutability**: All data structures are immutable by default, preventing bugs and enabling safe concurrent operations
2. **Composability**: Clean separation between data and algorithms allows flexible combination of operations
3. **Simplicity**: Clear, consistent API that's easy to learn and use correctly

Learn more in the [Design & Philosophy](design/immutability.md) section.

## License

AlgoGraph is open source software. See the repository for license details.
