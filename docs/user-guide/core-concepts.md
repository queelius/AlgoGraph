# Core Concepts

Understanding AlgoGraph's core concepts will help you use the library effectively. This guide explains the fundamental principles and design decisions behind AlgoGraph.

## Immutability

The most important concept in AlgoGraph is **immutability**: all data structures are immutable by default.

### What is Immutability?

An immutable object cannot be changed after it's created. Instead of modifying an object, you create a new one with the desired changes.

```python
from AlgoGraph import Vertex

# Create a vertex
v1 = Vertex('A', attrs={'value': 10})

# "Modify" it (actually creates a new vertex)
v2 = v1.with_attrs(value=20)

# Original is unchanged
print(v1.get('value'))  # 10
print(v2.get('value'))  # 20
```

### Why Immutability?

1. **Prevents Bugs**: No accidental modifications to shared data
2. **Thread-Safe**: Safe to use in concurrent code without locks
3. **Easier to Reason About**: Data flow is explicit and predictable
4. **Enables Time Travel**: Keep history of all graph states
5. **Simplifies Testing**: No hidden state changes to track

### Immutability in Practice

All three core types are immutable:

```python
from AlgoGraph import Vertex, Edge, Graph

# Vertices are immutable
v1 = Vertex('A')
v2 = v1.with_attrs(x=10)  # New vertex
v3 = v1.with_id('B')      # New vertex

# Edges are immutable
e1 = Edge('A', 'B', weight=5)
e2 = e1.with_weight(10)         # New edge
e3 = e1.to_undirected()         # New edge

# Graphs are immutable
g1 = Graph({v1}, {e1})
g2 = g1.add_vertex(v2)    # New graph
g3 = g1.add_edge(e2)      # New graph
```

## The Three Core Types

AlgoGraph has three fundamental data types:

### Vertex

A **Vertex** (or node) represents a point in the graph. Each vertex has:

- **ID** (string): Unique identifier
- **Attributes** (dict): Arbitrary key-value pairs

```python
# Simple vertex
v = Vertex('London')

# Vertex with attributes
v = Vertex('London', attrs={
    'population': 9000000,
    'country': 'UK',
    'coordinates': (51.5074, -0.1278)
})

# Access attributes
v.get('population')           # 9000000
v.get('timezone', 'UTC')      # 'UTC' (default)
```

Vertices are compared by ID and attributes:

```python
v1 = Vertex('A', attrs={'x': 1})
v2 = Vertex('A', attrs={'x': 1})
v3 = Vertex('A', attrs={'x': 2})

v1 == v2  # True (same ID and attributes)
v1 == v3  # False (different attributes)
```

### Edge

An **Edge** connects two vertices. Each edge has:

- **Source** (string): Source vertex ID
- **Target** (string): Target vertex ID
- **Directed** (bool): Whether the edge is directed (default: True)
- **Weight** (float): Edge weight (default: 1.0)
- **Attributes** (dict): Arbitrary key-value pairs

```python
# Simple directed edge
e = Edge('A', 'B')

# Undirected edge
e = Edge('A', 'B', directed=False)

# Weighted edge
e = Edge('A', 'B', weight=5.0)

# Edge with attributes
e = Edge('A', 'B', weight=100, attrs={
    'road_type': 'highway',
    'speed_limit': 65
})
```

Directed vs. undirected edges:

```python
directed = Edge('A', 'B', directed=True)
directed.connects('A', 'B')  # True
directed.connects('B', 'A')  # False

undirected = Edge('A', 'B', directed=False)
undirected.connects('A', 'B')  # True
undirected.connects('B', 'A')  # True (order doesn't matter)
```

### Graph

A **Graph** is a collection of vertices and edges. It provides:

- **Vertices**: Set of Vertex objects
- **Edges**: Set of Edge objects
- **Operations**: Methods for querying and transforming the graph

```python
# Empty graph
g = Graph()

# Graph with vertices and edges
g = Graph(
    vertices={Vertex('A'), Vertex('B'), Vertex('C')},
    edges={Edge('A', 'B'), Edge('B', 'C')}
)

# Query the graph
g.vertex_count  # 3
g.edge_count    # 2
g.has_vertex('A')      # True
g.has_edge('A', 'B')   # True
g.neighbors('A')       # {'B'}
g.degree('B')          # 2 (one in, one out)
```

## Separation of Data and Algorithms

AlgoGraph separates **data structures** from **algorithms**:

- **Data structures** (`Vertex`, `Edge`, `Graph`) live in the main module
- **Algorithms** (DFS, BFS, Dijkstra, etc.) live in `AlgoGraph.algorithms`

This separation provides several benefits:

1. **Modularity**: Import only what you need
2. **Testability**: Easy to test data and algorithms separately
3. **Extensibility**: Add new algorithms without changing data structures
4. **Clarity**: Clear distinction between "what" (data) and "how" (algorithms)

```python
# Data structures
from AlgoGraph import Vertex, Edge, Graph

# Algorithms (separate import)
from AlgoGraph.algorithms import dijkstra, bfs, connected_components

# Use together
g = Graph(...)
distances, predecessors = dijkstra(g, 'A')
components = connected_components(g)
```

## Graph Modifications

Since graphs are immutable, "modifications" return new graphs:

```python
g1 = Graph({Vertex('A')}, {})

# Each operation returns a new graph
g2 = g1.add_vertex(Vertex('B'))
g3 = g2.add_edge(Edge('A', 'B'))
g4 = g3.remove_vertex('A')

# Original is unchanged
print(g1.vertex_count)  # 1
print(g4.vertex_count)  # 1 (just 'B' now)

# Can chain operations
g5 = (Graph()
    .add_vertex(Vertex('A'))
    .add_vertex(Vertex('B'))
    .add_edge(Edge('A', 'B'))
)
```

## Type System

AlgoGraph uses Python's type hints throughout:

```python
from typing import Set, Dict, Optional

def process_graph(
    graph: Graph,
    start: str,
    threshold: float = 1.0
) -> Optional[Set[str]]:
    """Process graph and return matching vertices."""
    # Type hints help IDEs and type checkers
    ...
```

Benefits:

- **IDE Support**: Better autocomplete and error detection
- **Documentation**: Types document expected inputs/outputs
- **Type Checking**: Use `mypy` to catch errors before runtime

## Vertex and Edge Identity

### Vertex Identity

Vertices are identified by their **ID**. Two vertices with the same ID are considered the same vertex (even with different attributes):

```python
v1 = Vertex('A', attrs={'x': 1})
v2 = Vertex('A', attrs={'x': 2})

# Same ID, so same hash (for use in sets)
hash(v1) == hash(v2)  # True

# But different equality (attributes differ)
v1 == v2  # False
```

This means you can't have two different vertices with the same ID in a graph:

```python
g = Graph({
    Vertex('A', attrs={'x': 1}),
    Vertex('A', attrs={'x': 2}),  # This replaces the first one
})
g.vertex_count  # 1 (not 2)
```

### Edge Identity

Edges are identified by their **source, target, and direction**:

```python
e1 = Edge('A', 'B', directed=True)
e2 = Edge('A', 'B', directed=True)
e3 = Edge('A', 'B', directed=False)

# Directed edges: same source/target = same identity
hash(e1) == hash(e2)  # True
hash(e1) == hash(e3)  # False (different direction)

# Undirected edges: order doesn't matter for hash
e4 = Edge('A', 'B', directed=False)
e5 = Edge('B', 'A', directed=False)
hash(e4) == hash(e5)  # True
```

## Graph Types

AlgoGraph supports various graph types:

### Directed vs. Undirected

```python
# Directed graph (default)
directed = Graph(
    vertices={Vertex('A'), Vertex('B')},
    edges={Edge('A', 'B', directed=True)}
)

# Undirected graph
undirected = Graph(
    vertices={Vertex('A'), Vertex('B')},
    edges={Edge('A', 'B', directed=False)}
)

# Mixed graph (both types of edges)
mixed = Graph(
    vertices={Vertex('A'), Vertex('B'), Vertex('C')},
    edges={
        Edge('A', 'B', directed=True),
        Edge('B', 'C', directed=False),
    }
)
```

### Weighted vs. Unweighted

```python
# Unweighted (all weights = 1.0)
unweighted = Graph(edges={
    Edge('A', 'B'),  # weight defaults to 1.0
    Edge('B', 'C'),
})

# Weighted
weighted = Graph(edges={
    Edge('A', 'B', weight=5.0),
    Edge('B', 'C', weight=3.0),
})
```

### Special Graph Types

You can model any graph type using AlgoGraph:

- **Simple graphs**: No self-loops, no parallel edges
- **Multigraphs**: Multiple edges between same vertices (use edge attributes to distinguish)
- **Trees**: Connected acyclic graphs
- **DAGs**: Directed acyclic graphs
- **Bipartite graphs**: Two disjoint vertex sets
- **Complete graphs**: Every vertex connected to every other

## Attributes

Both vertices and edges can have arbitrary attributes:

```python
# Vertex attributes
person = Vertex('Alice', attrs={
    'age': 30,
    'email': 'alice@example.com',
    'tags': ['developer', 'team-lead'],
    'metadata': {'joined': '2020-01-01'}
})

# Edge attributes
road = Edge('Boston', 'NYC', weight=215, attrs={
    'road_type': 'Interstate',
    'route': 'I-90',
    'toll': True,
    'scenic_rating': 7
})
```

Attributes are stored as dictionaries and can be any JSON-serializable type.

## Common Patterns

### Building Graphs Incrementally

```python
# Start empty
g = Graph()

# Add vertices one by one
for name in ['A', 'B', 'C']:
    g = g.add_vertex(Vertex(name))

# Add edges
g = g.add_edge(Edge('A', 'B'))
g = g.add_edge(Edge('B', 'C'))
```

### Building from Collections

```python
# From lists
vertices = [Vertex(name) for name in ['A', 'B', 'C']]
edges = [Edge(src, tgt) for src, tgt in [('A', 'B'), ('B', 'C')]]
g = Graph(set(vertices), set(edges))

# From comprehensions
g = Graph(
    vertices={Vertex(str(i)) for i in range(5)},
    edges={Edge(str(i), str(i+1)) for i in range(4)}
)
```

### Transforming Graphs

```python
# Add attributes to all vertices
def label_vertices(graph):
    new_graph = graph
    for v in graph.vertices:
        updated = v.with_attrs(label=f"Node {v.id}")
        new_graph = new_graph.update_vertex(updated)
    return new_graph

# Filter vertices by predicate
high_degree = graph.find_vertices(lambda v: graph.degree(v.id) > 3)

# Extract subgraph
vertex_ids = {v.id for v in high_degree}
subgraph = graph.subgraph(vertex_ids)
```

## Next Steps

Now that you understand the core concepts, learn about:

- [Working with Vertices](vertices.md)
- [Working with Edges](edges.md)
- [Building Graphs](graphs.md)
- [Graph Algorithms](algorithms.md)
