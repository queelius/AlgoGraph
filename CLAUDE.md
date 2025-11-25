# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlgoGraph is an immutable graph library with 56+ algorithms, functional transformers, declarative selectors, and lazy views. Version 2.0.0 brings AlgoTree-level API elegance to graph programming with pipe-based composition and ~90% code reduction for common operations.

**Primary Interface:** The Python API (Graph, Vertex, Edge, transformers, selectors) is the recommended approach for scripting, automation, and programmatic use.

**Secondary Interface:** The interactive shell is for exploration and terminal-based workflows - not for scripting or automation.

## Development Commands

### Environment Setup
```bash
# From the AlgoGraph directory or parent
export PYTHONPATH=/path/to/released:$PYTHONPATH
```

### Running Tests
```bash
# All tests (213 tests)
python -m pytest test/

# Specific test file
python -m pytest test/test_algorithms.py
python -m pytest test/test_phase3_features.py

# Single test
python -m pytest test/test_algorithms.py::TestTraversal::test_dfs_simple

# With coverage
python -m pytest --cov=. --cov-report=html test/
```

### Building & Publishing
```bash
# Build distribution (uses pyproject.toml)
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

### Interactive Shell (For Exploration Only)
```bash
python -m AlgoGraph.shell.shell              # Sample graph
python -m AlgoGraph.shell.shell graph.json   # Load from file
algograph                                     # CLI entry point (if installed)
```

**Note:** For scripting and automation, always use the Python API, not the shell.

## Architecture

### Core Layer (Immutable Data Structures)

```
vertex.py     → Vertex: Immutable vertices with arbitrary attributes
edge.py       → Edge: Directed/undirected edges with weights and attributes
graph.py      → Graph: Immutable container with query/modification methods
builder.py    → GraphBuilder: Fluent API for graph construction
```

**Key Pattern**: All operations return new objects. Chain with `g.add_vertex(v).add_edge(e)`.

### Algorithm Layer (56+ Algorithms)

```
algorithms/
├── traversal.py      → DFS, BFS, topological sort, cycle detection, path finding
├── shortest_path.py  → Dijkstra, Bellman-Ford, Floyd-Warshall, A*
├── connectivity.py   → Components, SCC, bipartite, bridges, articulation points
├── spanning_tree.py  → Kruskal, Prim, MST utilities
├── centrality.py     → PageRank, betweenness, closeness, eigenvector, degree
├── flow.py           → Edmonds-Karp, Ford-Fulkerson, max flow, min cut
├── matching.py       → Hopcroft-Karp, maximum bipartite matching
└── coloring.py       → Welsh-Powell, DSatur, chromatic number, edge coloring
```

### Advanced Features Layer (v2.0.0)

```
transformers.py      → Pipe-based composition with | operator
                       filter_vertices, map_edges, largest_component, stats, etc.

graph_selectors.py   → Declarative queries with logical operators (&, |, ~, ^)
                       vertex.attrs(), vertex.degree(), edge.weight(), etc.

views.py             → Lazy evaluation without copying
                       FilteredView, SubGraphView, NeighborhoodView, etc.
```

**Pipe Pattern**: `graph | filter_vertices(pred) | largest_component() | stats()`

**Selector Pattern**: `graph.select_vertices(v.attrs(active=True) & v.degree(min_degree=5))`

**View Pattern**: `filtered_view(graph, vertex_filter=pred).materialize()`

### Supporting Layers

```
serialization.py  → JSON save/load: save_graph(), load_graph()
interop.py        → AlgoTree conversion (optional): tree_to_graph(), graph_to_tree()

shell/
├── context.py    → Immutable navigation state (GraphContext)
├── commands.py   → Command classes with execute() method
├── shell.py      → REPL with readline tab completion
└── cli.py        → Entry point for algograph command
```

## Key Design Patterns

### Immutability
All graph operations return new Graph instances:
```python
g2 = g.add_vertex(Vertex('A'))  # g unchanged, g2 is new
g3 = g2.add_edge(Edge('A', 'B'))  # g2 unchanged
```

### Selector Type Inheritance
Logical combinators dynamically inherit from VertexSelector/EdgeSelector to pass isinstance() checks:
```python
# In graph_selectors.py, AndSelector.__init__ does:
if isinstance(left, VertexSelector):
    self.__class__ = type('AndVertexSelector', (AndSelector, VertexSelector), {})
```

### Transformer Pipe Operator
Transformers implement `__ror__` for right-hand pipe support:
```python
def __ror__(self, other: T) -> S:
    return self(other)  # Enables: graph | transformer
```

## Test Organization

```
test/
├── test_algorithms.py        → Core algorithm tests (27 tests)
├── test_builder.py           → GraphBuilder fluent API tests (33 tests)
├── test_interop.py           → AlgoTree conversion tests (13 tests, skipped without AlgoTree)
├── test_shell.py             → Shell navigation/commands (38 tests)
├── test_phase2_algorithms.py → Centrality, flow, matching, coloring (36 tests)
├── test_phase3_features.py   → Transformers, selectors, views (40 tests)
└── test_phase3_coverage.py   → Additional coverage tests (30 tests)
```

Total: 213 tests passing (13 skipped when AlgoTree unavailable)

## Module Naming Note

`graph_selectors.py` is named to avoid conflict with Python's built-in `selectors` module (I/O multiplexing). Do not rename to `selectors.py`.

## Graph Construction Patterns

```python
# Direct construction
g = Graph({Vertex('A'), Vertex('B')}, {Edge('A', 'B')})

# Fluent builder
g = (Graph.builder()
     .add_vertex('A', age=30)
     .add_edge('A', 'B', weight=5.0)
     .add_path('B', 'C', 'D')
     .build())

# From edges (auto-creates vertices)
g = Graph.from_edges(('A', 'B'), ('B', 'C'))
```

## Algorithm Usage

```python
from AlgoGraph.algorithms import dijkstra, pagerank, connected_components

distances = dijkstra(graph, source='A')
pr = pagerank(social_network)
components = connected_components(graph)
```

## Advanced Features Usage

```python
# Transformers
from AlgoGraph.transformers import filter_vertices, largest_component, stats
result = graph | filter_vertices(lambda v: v.get('active')) | stats()

# Selectors
from AlgoGraph.graph_selectors import vertex as v, edge as e
matches = graph.select_vertices(v.attrs(role='admin') | v.degree(min_degree=10))

# Views
from AlgoGraph.views import filtered_view, neighborhood_view
view = neighborhood_view(graph, center='A', k=2)
subgraph = view.materialize()
```

## Important Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
