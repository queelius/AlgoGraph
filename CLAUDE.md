# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlgoGraph is an immutable graph data structures and algorithms library with optional AlgoTree integration. It provides directed/undirected graphs with weighted edges, vertex/edge attributes, and 30+ graph algorithms organized by category.

## Development Environment

### PYTHONPATH Setup
AlgoGraph requires the parent `released/` directory in PYTHONPATH:
```bash
cd /home/spinoza/github/released
export PYTHONPATH=.
```

This enables both standalone use and optional AlgoTree interop features.

### Running Tests
```bash
# From AlgoGraph directory
cd /home/spinoza/github/released/AlgoGraph

# Run all tests
pytest test/

# Run specific test file
pytest test/test_algorithms.py
pytest test/test_interop.py
pytest test/test_shell.py

# Run with verbose output
pytest -v test/

# Run specific test class or method
pytest test/test_algorithms.py::TestTraversal::test_dfs_simple

# Run with coverage
pytest --cov=. --cov-report=html test/
```

### Graph Serialization
```bash
# Save/load graphs to JSON
from AlgoGraph import Graph, Vertex, Edge, save_graph, load_graph

# Save graph
save_graph(graph, 'my_graph.json')

# Load graph
graph = load_graph('my_graph.json')
```

### Interactive Shell
```bash
cd /home/spinoza/github/released
export PYTHONPATH=.

# Start with sample graph
python -m AlgoGraph.shell.shell

# Or load a graph from file
python -m AlgoGraph.shell.shell my_graph.json
```

The shell provides a VFS-like interface where you can `cd` into vertices, `ls` their attributes, navigate neighbors, and run graph queries (`path`, `shortest`, `components`, `bfs`).

**Features:**
- File I/O: Load graphs from JSON files, save with `save` command
- Tab completion: Press TAB to complete commands and vertex names
- Command history: Use UP/DOWN arrows to recall commands
- Absolute paths: Use `cd /vertex` to jump directly to any vertex
- Quoted names: Use `cd "Alice Smith"` for vertex names with spaces

## Architecture

### Core Immutable Data Structures

1. **Vertex** (`vertex.py`): Immutable graph vertices with arbitrary attributes
   - Methods: `with_attrs()`, `without_attrs()`, `get()`
   - Hashable by ID for set/dict usage

2. **Edge** (`edge.py`): Immutable edges connecting vertices
   - Supports directed/undirected
   - Optional weight and attributes
   - Methods: `with_weight()`, `reversed()`, `connects()`

3. **Graph** (`graph.py`): Immutable graph container
   - Stores sets of vertices and edges
   - All operations return new Graph instances
   - Query methods: `neighbors()`, `degree()`, `has_vertex()`, `has_edge()`
   - Construction: `add_vertex()`, `add_edge()`, `remove_vertex()`, `subgraph()`

### Algorithm Organization

Algorithms are in `algorithms/` and imported via `algorithms/__init__.py`:

- **Traversal** (`traversal.py`): DFS, BFS, topological sort, cycle detection, path finding
- **Shortest Path** (`shortest_path.py`): Dijkstra, Bellman-Ford, Floyd-Warshall, A*
- **Connectivity** (`connectivity.py`): Connected components, SCC, bipartite checking, bridges, articulation points
- **Spanning Tree** (`spanning_tree.py`): Kruskal, Prim, MST utilities

### Serialization Layer

`serialization.py` provides JSON file I/O:
- `graph_to_json()` / `graph_from_json()`: Convert graphs to/from JSON strings
- `save_graph()` / `load_graph()`: Save/load graphs to/from JSON files
- Preserves all graph data: vertices, edges, attributes, weights, directed/undirected
- Human-readable JSON format

### Interoperability Layer

**Optional dependency**: Requires AlgoTree in PYTHONPATH

`interop.py` provides bidirectional conversion:
- `tree_to_graph()` / `node_to_graph()`: Convert tree hierarchies to graphs
- `graph_to_tree()`: Extract spanning tree from graph
- `flat_dict_to_graph()` / `graph_to_flat_dict()`: Interchange format compatible with AlgoTree's flat exporter

### Interactive Shell

Multi-file design in `shell/`:
- `context.py`: Immutable navigation state (GraphContext)
- `commands.py`: Command classes with execute() method (including SaveCommand)
- `shell.py`: REPL with readline integration for tab completion and history
- `cli.py`: Entry point with file loading support
- `serialization.py`: Graph save/load functionality

Navigation model treats graphs as filesystems:
- `/` = root (all vertices)
- `/vertex_id` = at a vertex (shows attributes + neighbors/)
- `/vertex_id/neighbors` = neighbor view mode

**Command Parsing:**
- Uses `shlex.split()` for proper quote handling
- Supports vertex names with spaces via quotes: `cd "Alice Smith"`
- Tab completion for commands and vertex names (context-aware)
- Command history via readline (UP/DOWN arrows)

**Path Navigation:**
- Relative: `cd vertex` - navigate to neighbor or any vertex
- Absolute: `cd /vertex` - jump directly to any vertex from anywhere
- Special: `cd ..` (up), `cd /` (root), `cd neighbors` (neighbors mode)

## Key Design Principles

1. **Immutability**: All graph operations return new objects; no in-place mutation
2. **Composability**: Operations chain naturally (e.g., `g.add_vertex(v).add_edge(e)`)
3. **Separation**: Data structures (Graph/Vertex/Edge) are separate from algorithms
4. **Type Safety**: Full type hints throughout
5. **Functional Style**: Prefer pure functions in algorithms module

## Testing Strategy

Test files mirror the module structure:
- `test/test_algorithms.py`: Algorithm correctness tests organized by category
- `test/test_interop.py`: Tree-graph conversion tests (requires AlgoTree)
- `test/test_shell.py`: Shell navigation and command tests (38 tests)
  - Original shell functionality (29 tests)
  - Serialization (2 tests)
  - Save command (2 tests)
  - Absolute paths (3 tests)
  - Quoted vertex names (2 tests)

Tests use pytest and follow pattern: arrange graph → run algorithm → assert expected result.

**Coverage:** All shell improvements are fully tested with 100% test success rate.

## Common Patterns

### Creating Graphs
```python
from AlgoGraph import Graph, Vertex, Edge

# Method 1: Construct with sets
g = Graph({Vertex('A'), Vertex('B')}, {Edge('A', 'B')})

# Method 2: Build incrementally (returns new graphs)
g = Graph().add_vertex(Vertex('A')).add_edge(Edge('A', 'B'))
```

### Running Algorithms
```python
from AlgoGraph.algorithms import dijkstra, bfs, connected_components

distances = dijkstra(graph, source='A')
order = bfs(graph, start='A')
components = connected_components(graph)
```

### AlgoTree Integration
Only import interop functions when AlgoTree integration is needed:
```python
from AlgoGraph import tree_to_graph, graph_to_tree
# Requires: export PYTHONPATH=/path/to/released:$PYTHONPATH
```
