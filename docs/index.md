# AlgoGraph

**Immutable graph data structures and algorithms library with functional transformers, declarative selectors, and lazy views**

AlgoGraph is a powerful, functional graph library for Python that provides immutable data structures and a comprehensive collection of 56+ graph algorithms. Version 2.0.0 brings AlgoTree-level API elegance with pipe-based transformers, declarative selectors, and lazy views—achieving ~90% code reduction for common operations.

## Installation

```bash
pip install AlgoGraph
```

## Key Features

### Immutable Data Structures
- **Vertex**: Immutable graph vertices with customizable attributes
- **Edge**: Immutable directed/undirected edges with weights and attributes
- **Graph**: Immutable graph container with efficient operations
- **GraphBuilder**: Fluent API for graph construction

### 56+ Graph Algorithms
Organized into eight comprehensive categories:

- **Traversal**: DFS, BFS, topological sort, cycle detection, path finding
- **Shortest Path**: Dijkstra, Bellman-Ford, Floyd-Warshall, A* search
- **Connectivity**: Connected components, SCC, bipartite checking, bridges, articulation points
- **Spanning Tree**: Kruskal's algorithm, Prim's algorithm, MST operations
- **Centrality**: PageRank, betweenness, closeness, degree, eigenvector centrality
- **Flow Networks**: Max flow, min cut, Edmonds-Karp, Ford-Fulkerson
- **Matching**: Hopcroft-Karp, maximum bipartite matching
- **Graph Coloring**: Welsh-Powell, DSatur, chromatic number

### Advanced Features (v2.0.0)

#### Transformer Pipelines
Compose graph operations using the `|` pipe operator:

```python
from AlgoGraph.transformers import filter_vertices, largest_component, stats

result = (graph
    | filter_vertices(lambda v: v.get('active'))
    | largest_component()
    | stats())
```

#### Declarative Selectors
Query vertices and edges with logical operators:

```python
from AlgoGraph.graph_selectors import vertex as v

power_users = graph.select_vertices(
    v.attrs(active=True) & v.degree(min_degree=10)
)
```

#### Lazy Views
Memory-efficient filtering without copying:

```python
from AlgoGraph.views import filtered_view

view = filtered_view(graph, vertex_filter=lambda v: v.get('active'))
small_graph = view.materialize()  # Only copy when needed
```

### Interactive Shell
A VFS-like interface for exploring graphs interactively:

- Navigate graphs like a file system (`cd`, `ls`, `pwd`)
- Query paths, components, and neighbors
- Load and save graphs in JSON format
- Tab completion and command history

### Serialization & Interoperability
- JSON import/export for persistent storage
- Optional AlgoTree integration for tree-graph conversions
- Flat dictionary representation for data exchange

## Quick Example

```python
from AlgoGraph import Graph
from AlgoGraph.algorithms import dijkstra, pagerank
from AlgoGraph.transformers import filter_vertices, stats
from AlgoGraph.graph_selectors import vertex as v

# Build graph with fluent API
network = (Graph.builder()
    .add_vertex('Alice', followers=1000, active=True)
    .add_vertex('Bob', followers=500, active=True)
    .add_vertex('Charlie', followers=2000, active=False)
    .add_edge('Alice', 'Bob', weight=5, directed=False)
    .add_edge('Bob', 'Charlie', weight=3, directed=False)
    .build())

# Run algorithms
pr = pagerank(network)
distances = dijkstra(network, 'Alice')

# Use transformers
analysis = network | filter_vertices(lambda v: v.get('active')) | stats()

# Use selectors
influencers = network.select_vertices(
    v.attrs(followers=lambda f: f > 800) & v.attrs(active=True)
)
```

## Why AlgoGraph?

### Functional & Immutable
AlgoGraph follows functional programming principles with immutable data structures. Every operation returns a new graph, preserving the original. This makes your code easier to reason about and prevents subtle bugs from shared mutable state.

### Composable & Declarative
The v2.0.0 transformer and selector patterns let you express complex operations declaratively. Chain transformers with `|`, combine selectors with `&`, `|`, `~`—write *what* you want, not *how* to do it.

### Production Ready
- 213 comprehensive tests
- Well-documented API with type hints
- Performance optimized
- Battle-tested algorithms from peer-reviewed literature

### Developer Friendly
- Clear, consistent API design
- Rich examples and cookbook recipes
- Interactive shell for exploration
- Helpful error messages

## What's Next?

- **New to AlgoGraph?** Start with the [Installation Guide](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md)
- **Want to learn the concepts?** Read the [User Guide](user-guide/core-concepts.md)
- **Looking for v2.0.0 features?** Check [Transformers](user-guide/transformers.md), [Selectors](user-guide/selectors.md), and [Views](user-guide/views.md)
- **Looking for specific functionality?** Check the [API Reference](api/vertex.md)
- **Need examples?** Browse the [Examples & Cookbook](examples/social-networks.md)
- **Want to explore interactively?** Try the [Interactive Shell](shell/overview.md)

## Design Philosophy

AlgoGraph is built on five core principles:

1. **Immutability**: All data structures are immutable by default
2. **Composability**: Chain operations with `|` pipe operator
3. **Declarative**: Express *what*, not *how* (selectors vs lambdas)
4. **Lazy evaluation**: Views defer computation until needed
5. **Type safety**: Full type hints throughout

Learn more in the [Design & Philosophy](design/immutability.md) section.

## License

MIT License - AlgoGraph is open source software.
