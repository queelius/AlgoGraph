# Immutability

Immutability is the cornerstone of AlgoGraph's design. This document explains why we chose immutability and how it benefits your code.

## What is Immutability?

**Immutable objects cannot be changed after creation.** Instead of modifying an object, you create a new one with the desired changes.

```python
from AlgoGraph import Vertex

# Create vertex
v1 = Vertex('A', attrs={'value': 10})

# "Modify" it - actually creates new vertex
v2 = v1.with_attrs(value=20)

# Original unchanged
print(v1.get('value'))  # 10
print(v2.get('value'))  # 20
```

## Why Immutability?

### 1. No Unexpected Side Effects

With mutable objects, functions can modify data unexpectedly:

```python
# Mutable approach (NOT how AlgoGraph works)
def bad_add_edge(graph, edge):
    graph.edges.add(edge)  # Modifies original!
    return graph

g1 = create_graph()
g2 = bad_add_edge(g1, edge)
# g1 was modified - surprise!
```

With immutability, the original is always safe:

```python
# Immutable approach (how AlgoGraph works)
g1 = Graph(...)
g2 = g1.add_edge(edge)  # Returns new graph
# g1 is unchanged - guaranteed!
```

### 2. Thread Safety

Immutable objects are inherently thread-safe:

```python
from concurrent.futures import ThreadPoolExecutor
from AlgoGraph.algorithms import dijkstra, bfs

graph = load_graph('network.json')

# Safe to use same graph in multiple threads
with ThreadPoolExecutor() as executor:
    future1 = executor.submit(dijkstra, graph, 'A')
    future2 = executor.submit(bfs, graph, 'B')
    future3 = executor.submit(connected_components, graph)

# No locks needed - graph can't be modified
results = [f.result() for f in [future1, future2, future3]]
```

### 3. Easier Debugging

With immutability, data doesn't change unexpectedly:

```python
# Create graph
g1 = create_initial_graph()
print(f"Initial: {g1.vertex_count} vertices")  # 5

# Transform it
g2 = add_some_vertices(g1)
print(f"After adding: {g2.vertex_count} vertices")  # 8

# Original still intact for debugging
print(f"Original still: {g1.vertex_count} vertices")  # 5

# Can compare states
print(f"Added {g2.vertex_count - g1.vertex_count} vertices")
```

### 4. Time Travel / Undo

Keep history of all states:

```python
history = []

# Build graph step by step
g = Graph()
history.append(g)

g = g.add_vertex(Vertex('A'))
history.append(g)

g = g.add_vertex(Vertex('B'))
history.append(g)

g = g.add_edge(Edge('A', 'B'))
history.append(g)

# "Undo" by going back in history
previous_state = history[-2]
print(f"Before last operation: {previous_state.edge_count} edges")
```

### 5. Referential Transparency

Same inputs always produce same outputs:

```python
from AlgoGraph.algorithms import dijkstra

# Pure function - no side effects
distances1, _ = dijkstra(graph, 'A')
distances2, _ = dijkstra(graph, 'A')

# Results are always identical
assert distances1 == distances2

# Graph is unchanged
assert graph == graph  # Always true
```

## Implementation Techniques

### Frozen Dataclasses

AlgoGraph uses Python's frozen dataclasses:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Vertex:
    id: str
    attrs: dict

# Attempts to modify raise errors
v = Vertex('A', {'x': 1})
v.id = 'B'  # FrozenInstanceError!
```

### Immutable Collections

Sets and dicts are copied, not shared:

```python
@dataclass(frozen=True)
class Graph:
    vertices: Set[Vertex]
    edges: Set[Edge]

    def add_vertex(self, vertex):
        # Create new set (doesn't modify original)
        new_vertices = self.vertices | {vertex}
        return Graph(new_vertices, self.edges)
```

### Copy-on-Write

Only copy what changes:

```python
def add_edge(self, edge):
    # Only vertices and edges change
    # All other graph properties remain shared
    new_edges = self.edges | {edge}
    return Graph(self.vertices, new_edges)
```

## Performance Considerations

### Memory Overhead

Immutability can use more memory:

```python
# Creating many versions
g1 = Graph(...)
g2 = g1.add_vertex(v1)  # New graph
g3 = g2.add_vertex(v2)  # Another new graph
g4 = g3.add_vertex(v3)  # Yet another new graph

# All four graphs exist in memory
```

**Mitigation:** Python's garbage collector reclaims unused graphs automatically.

### Structural Sharing

AlgoGraph uses structural sharing to reduce overhead:

```python
g1 = Graph(vertices_set, edges_set)
g2 = g1.add_vertex(new_vertex)

# g2 shares edges with g1 (sets are immutable)
# Only vertices set is new
```

### When to Be Careful

Large batch operations can be expensive:

```python
# Inefficient - creates many intermediate graphs
g = Graph()
for i in range(1000):
    g = g.add_vertex(Vertex(str(i)))
```

**Better approach:**

```python
# Efficient - create all vertices at once
vertices = {Vertex(str(i)) for i in range(1000)}
g = Graph(vertices)
```

## Working with Immutability

### Building Graphs

When building large graphs, create collections first:

```python
# Good - create sets then build graph
vertices = {Vertex(name) for name in names}
edges = {Edge(src, tgt, weight=w) for src, tgt, w in edge_data}
graph = Graph(vertices, edges)

# Less efficient - many intermediate graphs
graph = Graph()
for name in names:
    graph = graph.add_vertex(Vertex(name))
for src, tgt, w in edge_data:
    graph = graph.add_edge(Edge(src, tgt, weight=w))
```

### Transforming Graphs

Collect changes then apply:

```python
# Collect updates
updates = {}
for v in graph.vertices:
    if needs_update(v):
        updates[v.id] = compute_new_attrs(v)

# Apply all at once
new_graph = graph
for vid, new_attrs in updates.items():
    v = graph.get_vertex(vid)
    updated = v.with_attrs(**new_attrs)
    new_graph = new_graph.update_vertex(updated)
```

### Algorithms

Algorithms work with immutable graphs but can use mutable internal state:

```python
def dijkstra(graph, source):
    # Graph is immutable
    # But algorithm uses mutable data structures internally
    distances = {}  # Mutable dict (internal)
    visited = set()  # Mutable set (internal)

    # ... algorithm logic ...

    # Returns immutable result
    return distances, predecessors
```

## Comparison with Mutable Approaches

### NetworkX (Mutable)

```python
import networkx as nx

# Mutable graph
G = nx.Graph()
G.add_node('A', value=10)
G.add_edge('A', 'B')

# Modifies in place
G.nodes['A']['value'] = 20  # Original changed!

# Not thread-safe
# Need locks for concurrent access
```

### AlgoGraph (Immutable)

```python
from AlgoGraph import Graph, Vertex, Edge

# Immutable graph
g1 = Graph({Vertex('A', attrs={'value': 10})}, {Edge('A', 'B')})

# Returns new graph
v = g1.get_vertex('A')
updated_v = v.with_attrs(value=20)
g2 = g1.update_vertex(updated_v)

# Original unchanged
print(g1.get_vertex('A').get('value'))  # 10
print(g2.get_vertex('A').get('value'))  # 20

# Thread-safe by default
# No locks needed
```

## Best Practices

### 1. Embrace Immutability

Don't fight it - work with it:

```python
# Don't try to "modify" in loops
# BAD
g = Graph()
for item in items:
    g = g.add_vertex(Vertex(item))  # Many intermediate graphs

# Good - build collections first
vertices = {Vertex(item) for item in items}
g = Graph(vertices)
```

### 2. Use Chaining

Chain operations for clarity:

```python
result = (graph
    .add_vertex(Vertex('A'))
    .add_vertex(Vertex('B'))
    .add_edge(Edge('A', 'B'))
)
```

### 3. Keep References Minimal

Don't keep unnecessary graph versions:

```python
# BAD - keeps all intermediate states
states = []
g = Graph()
for i in range(1000):
    g = g.add_vertex(Vertex(str(i)))
    states.append(g)  # Lots of memory!

# Good - keep only final state
g = Graph({Vertex(str(i)) for i in range(1000)})
```

### 4. Leverage Structural Sharing

Take advantage of shared structure:

```python
base_graph = create_large_graph()  # Expensive

# These are cheap - share most structure with base
variant1 = base_graph.add_vertex(Vertex('X'))
variant2 = base_graph.add_vertex(Vertex('Y'))
variant3 = base_graph.add_edge(Edge('A', 'B'))
```

## Conclusion

Immutability makes AlgoGraph:

- **Safer**: No unexpected modifications
- **Simpler**: Easier to reason about
- **More Testable**: Predictable behavior
- **Thread-Safe**: Use in concurrent code without locks
- **Debuggable**: Keep history of all states

While there's a learning curve and some performance considerations, the benefits far outweigh the costs for most applications.

## Further Reading

- [Core Concepts](../user-guide/core-concepts.md)
- [Composability](composability.md)
- [Separation of Concerns](separation.md)
