# API Comparison: Current vs Proposed

This document shows side-by-side comparisons of common operations in AlgoGraph v1.1.0 (current) versus the proposed v2.0.0 API.

---

## 1. Building a Simple Graph

### Current (v1.1.0) - Verbose
```python
from AlgoGraph import Graph, Vertex, Edge

# Create vertices individually
alice = Vertex('Alice', attrs={'age': 30, 'role': 'engineer'})
bob = Vertex('Bob', attrs={'age': 25, 'role': 'designer'})
carol = Vertex('Carol', attrs={'age': 35, 'role': 'manager'})

# Create edges individually
friendship1 = Edge('Alice', 'Bob', directed=False, weight=0.9, attrs={'type': 'friend'})
friendship2 = Edge('Bob', 'Carol', directed=False, weight=0.7, attrs={'type': 'colleague'})
reports_to = Edge('Alice', 'Carol', directed=True, weight=1.0, attrs={'type': 'reports_to'})

# Build graph
company = Graph(
    vertices={alice, bob, carol},
    edges={friendship1, friendship2, reports_to}
)

# Total: 14 lines, many intermediate variables
```

### Proposed (v2.0.0) - Fluent
```python
from AlgoGraph import graph, vertex, edge

# Build graph declaratively
company = graph(
    vertex('Alice', age=30, role='engineer',
        edge('Bob', type='friend', weight=0.9, directed=False),
        edge('Carol', type='reports_to', weight=1.0)
    ),
    vertex('Bob', age=25, role='designer',
        edge('Carol', type='colleague', weight=0.7, directed=False)
    ),
    vertex('Carol', age=35, role='manager')
)

# Total: 10 lines, no intermediate variables
```

**Improvement:** 30% fewer lines, clearer intent, no namespace pollution

---

## 2. Loading from Data

### Current (v1.1.0) - Manual Construction
```python
from AlgoGraph import Graph, Vertex, Edge

# Load data from dict
data = {
    'A': {'value': 10, 'neighbors': [('B', 5.0), ('C', 10.0)]},
    'B': {'value': 20, 'neighbors': [('C', 3.0)]},
    'C': {'value': 30, 'neighbors': []}
}

# Manually construct graph
vertices = set()
edges = set()

for node_id, node_data in data.items():
    # Extract attributes
    attrs = {k: v for k, v in node_data.items() if k != 'neighbors'}
    vertices.add(Vertex(node_id, attrs=attrs))

    # Extract edges
    for neighbor_id, weight in node_data.get('neighbors', []):
        edges.add(Edge(node_id, neighbor_id, weight=weight))

graph = Graph(vertices, edges)

# Total: 17 lines, imperative style
```

### Proposed (v2.0.0) - Declarative Factory
```python
from AlgoGraph import Graph

# Load data from dict
data = {
    'A': {'value': 10, 'neighbors': [('B', 5.0), ('C', 10.0)]},
    'B': {'value': 20, 'neighbors': [('C', 3.0)]},
    'C': {'value': 30, 'neighbors': []}
}

# Factory method handles construction
graph = Graph.from_dict(data, neighbors_key='neighbors')

# Total: 8 lines, declarative style
```

**Improvement:** 50% fewer lines, no manual iteration, clearer intent

---

## 3. Filtering Vertices

### Current (v1.1.0) - Procedural
```python
from AlgoGraph import Graph, Vertex, Edge

# Build graph
g = Graph({
    Vertex('A', attrs={'active': True, 'priority': 10}),
    Vertex('B', attrs={'active': False, 'priority': 5}),
    Vertex('C', attrs={'active': True, 'priority': 15}),
    Vertex('D', attrs={'active': True, 'priority': 3})
}, set())

# Filter vertices manually
high_priority_active = set()
for v in g.vertices:
    if v.get('active', False) and v.get('priority', 0) > 5:
        high_priority_active.add(v.id)

# Create subgraph
filtered_graph = g.subgraph(high_priority_active)

# Total: 16 lines, manual filtering, two-step process
```

### Proposed (v2.0.0) - Declarative Selectors
```python
from AlgoGraph import Graph, vertex
from AlgoGraph.selectors import vertex as v

# Build graph (using builder)
g = Graph.from_dict({
    'A': {'active': True, 'priority': 10},
    'B': {'active': False, 'priority': 5},
    'C': {'active': True, 'priority': 15},
    'D': {'active': True, 'priority': 3}
})

# Filter with selectors (composable, reusable)
high_priority_active = v.attrs(active=True) & v.attrs(priority=lambda p: p > 5)
filtered_graph = g.filter_vertices(high_priority_active)

# Total: 13 lines, declarative, one-step process
```

**Improvement:** 20% fewer lines, composable selectors, more readable

---

## 4. Graph Transformations

### Current (v1.1.0) - Step-by-step
```python
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.algorithms import connected_components

# Load graph
g = load_graph('data.json')

# Step 1: Filter high-degree vertices
high_degree_ids = {v.id for v in g.vertices if g.degree(v.id) > 5}
g = g.subgraph(high_degree_ids)

# Step 2: Get largest component
components = connected_components(g)
largest_component = max(components, key=len)
g = g.subgraph(largest_component)

# Step 3: Normalize attributes
normalized_vertices = set()
for v in g.vertices:
    degree = g.degree(v.id)
    norm_degree = degree / g.vertex_count
    normalized_vertices.add(v.with_attrs(norm_degree=norm_degree))

g = Graph(normalized_vertices, g.edges)

# Step 4: Export to dict
result = graph_to_dict(g)

# Total: 20 lines, many intermediate variables, imperative
```

### Proposed (v2.0.0) - Fluent Pipeline
```python
from AlgoGraph import Graph
from AlgoGraph.transformers import (
    filter_vertices, largest_component, annotate, to_dict
)

# Load and transform in one pipeline
result = (Graph.from_file('data.json')
    | filter_vertices(lambda v: v.degree() > 5)
    | largest_component()
    | annotate(lambda g, v: {
        'norm_degree': v.degree() / g.vertex_count
    })
    | to_dict())

# Total: 10 lines, no intermediate variables, declarative
```

**Improvement:** 50% fewer lines, fluent style, highly composable

---

## 5. Shortest Path Analysis

### Current (v1.1.0) - Manual Reconstruction
```python
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.algorithms import dijkstra

# Build graph
g = Graph(edges={
    Edge('A', 'B', weight=10),
    Edge('B', 'C', weight=20),
    Edge('A', 'C', weight=35),
    Edge('C', 'D', weight=15)
})

# Run Dijkstra
distances, predecessors = dijkstra(g, 'A')

# Manually reconstruct path
def reconstruct_path(start, end, predecessors):
    path = []
    current = end
    while current:
        path.append(current)
        current = predecessors.get(current)
    return list(reversed(path))

path = reconstruct_path('A', 'D', predecessors)
distance = distances['D']

print(f"Path: {' -> '.join(path)}")
print(f"Distance: {distance}")

# Total: 24 lines, manual path reconstruction
```

### Proposed (v2.0.0) - Integrated API
```python
from AlgoGraph import Graph
from AlgoGraph.algorithms import shortest_path

# Build graph from edge list
g = Graph.from_edge_list([
    ('A', 'B', 10),
    ('B', 'C', 20),
    ('A', 'C', 35),
    ('C', 'D', 15)
])

# Get complete shortest path result
result = shortest_path(g, 'A', 'D')

print(f"Path: {' -> '.join(result['path'])}")
print(f"Distance: {result['distance']}")

# Total: 14 lines, integrated path reconstruction
```

**Improvement:** 40% fewer lines, no manual reconstruction, cleaner API

---

## 6. Complex Queries

### Current (v1.1.0) - Nested Lambdas
```python
from AlgoGraph import Graph, Vertex, Edge

# Build social network
g = Graph({
    Vertex('Alice', attrs={'age': 30, 'city': 'NYC', 'active': True}),
    Vertex('Bob', attrs={'age': 25, 'city': 'SF', 'active': True}),
    Vertex('Carol', attrs={'age': 35, 'city': 'NYC', 'active': False}),
    Vertex('Dave', attrs={'age': 28, 'city': 'NYC', 'active': True})
}, {
    Edge('Alice', 'Bob', directed=False),
    Edge('Alice', 'Dave', directed=False),
    Edge('Bob', 'Carol', directed=False)
})

# Find active users in NYC with more than 1 friend
# Requires nested condition checking
result = []
for v in g.vertices:
    if (v.get('active', False) and
        v.get('city') == 'NYC' and
        g.degree(v.id) > 1):
        result.append(v)

# Total: 20 lines, imperative, hard to read
```

### Proposed (v2.0.0) - Composable Selectors
```python
from AlgoGraph import Graph
from AlgoGraph.selectors import vertex as v

# Build social network (cleaner)
g = Graph.from_dict({
    'Alice': {'age': 30, 'city': 'NYC', 'active': True},
    'Bob': {'age': 25, 'city': 'SF', 'active': True},
    'Carol': {'age': 35, 'city': 'NYC', 'active': False},
    'Dave': {'age': 28, 'city': 'NYC', 'active': True}
}, edges=[
    ('Alice', 'Bob'),
    ('Alice', 'Dave'),
    ('Bob', 'Carol')
])

# Query with composable selectors
result = g.select_vertices(
    v.attrs(active=True) &
    v.attrs(city='NYC') &
    v.degree() > 1
)

# Total: 19 lines, declarative, much more readable
```

**Improvement:** Similar line count but vastly more readable and composable

---

## 7. Graph Analytics Pipeline

### Current (v1.1.0) - Not Possible
```python
# Can't chain analytics operations in v1.1.0
# Must do step-by-step with intermediate variables

from AlgoGraph import Graph
from AlgoGraph.algorithms import dijkstra, connected_components, kruskal

g = load_graph('network.json')

# Step 1: Compute centrality (manually)
centrality = {}
for v in g.vertices:
    distances, _ = dijkstra(g, v.id)
    reachable = [d for d in distances.values() if d != float('inf')]
    if reachable:
        avg_dist = sum(reachable) / len(reachable)
        centrality[v.id] = 1 / avg_dist if avg_dist > 0 else 0
    else:
        centrality[v.id] = 0

# Step 2: Filter by centrality
high_centrality_ids = {
    v_id for v_id, c in centrality.items() if c > 0.5
}
g = g.subgraph(high_centrality_ids)

# Step 3: Get components
components = connected_components(g)

# Step 4: Compute MST for largest component
largest = max(components, key=len)
subgraph = g.subgraph(largest)
mst = kruskal(subgraph)

# Total: 28 lines, many steps, complex
```

### Proposed (v2.0.0) - Fluent Analytics
```python
from AlgoGraph import Graph
from AlgoGraph.algorithms.centrality import closeness
from AlgoGraph.transformers import (
    annotate_centrality, filter_vertices, largest_component, mst
)

# Fluent analytics pipeline
result = (Graph.from_file('network.json')
    | annotate_centrality(closeness)
    | filter_vertices(lambda v: v.get('centrality', 0) > 0.5)
    | largest_component()
    | mst())

# Total: 10 lines, single pipeline, declarative
```

**Improvement:** 65% fewer lines, much clearer intent, composable

---

## 8. Type Safety

### Current (v1.1.0) - No Generic Types
```python
from AlgoGraph import Graph, Vertex, Edge

# No type information about vertex/edge data
def process_graph(g: Graph):
    for v in g.vertices:
        # Type checker doesn't know what's in attrs
        age = v.get('age')  # Type is Any
        if age > 30:  # No type checking
            print(f"{v.id} is over 30")

# Building graph with typos - no compile-time detection
g = Graph({
    Vertex('Alice', attrs={'ag': 30}),  # Typo: 'ag' instead of 'age'
    Vertex('Bob', attrs={'age': 25})
})

process_graph(g)  # Runtime error, not caught by type checker
```

### Proposed (v2.0.0) - Generic Types
```python
from AlgoGraph import Graph, Vertex
from dataclasses import dataclass

# Define typed vertex data
@dataclass
class Person:
    age: int
    name: str
    city: str

# Graph with typed vertices
def process_graph(g: Graph[Person, None]):
    for v in g.vertices:
        # Type checker knows v.data is Person
        person = v.data
        age = person.age  # Type is int
        if age > 30:  # Type safe
            print(f"{v.id} is over 30")

# Building graph with types
g = Graph[Person, None]()
g = g.add_vertex(Vertex('Alice', Person(age=30, name='Alice', city='NYC')))

# Typo caught at type-check time!
# g = g.add_vertex(Vertex('Bob', Person(ag=25, ...)))  # Type error!
```

**Improvement:** Full type safety, compile-time error detection, IDE support

---

## 9. Exporting and Visualization

### Current (v1.1.0) - JSON Only
```python
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.serialization import save_graph

# Build graph
g = Graph({
    Vertex('A'), Vertex('B'), Vertex('C')
}, {
    Edge('A', 'B', weight=1.0),
    Edge('B', 'C', weight=2.0),
    Edge('A', 'C', weight=3.0)
})

# Only JSON export available
save_graph(g, 'graph.json')

# To visualize, must manually convert to DOT format
# (not provided by library)
with open('graph.dot', 'w') as f:
    f.write('digraph G {\n')
    for v in g.vertices:
        f.write(f'  {v.id};\n')
    for e in g.edges:
        f.write(f'  {e.source} -> {e.target} [label="{e.weight}"];\n')
    f.write('}\n')

# Total: 18 lines for export, manual DOT generation
```

### Proposed (v2.0.0) - Multiple Formats
```python
from AlgoGraph import Graph
from AlgoGraph.exporters import to_json, to_dot, to_graphml
from AlgoGraph.viz import render

# Build graph (using edge list)
g = Graph.from_edge_list([
    ('A', 'B', 1.0),
    ('B', 'C', 2.0),
    ('A', 'C', 3.0)
])

# Export to multiple formats
to_json(g, 'graph.json')
to_dot(g, 'graph.dot', layout='neato')
to_graphml(g, 'graph.graphml')

# Visualize directly
render(g, format='png', output='graph.png', layout='fdp')

# Total: 11 lines, multiple formats, direct visualization
```

**Improvement:** 40% fewer lines, native multi-format support, direct viz

---

## 10. Functional Composition

### Current (v1.1.0) - Limited Composition
```python
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.algorithms import bfs, dijkstra, connected_components

# Can't compose operations functionally
# Must use intermediate variables

g = load_graph('data.json')

# Separate function calls, no composition
bfs_order = bfs(g, 'A')
distances, _ = dijkstra(g, 'A')
components = connected_components(g)

# Can't create reusable pipelines
# Can't curry or partially apply
```

### Proposed (v2.0.0) - Full Functional Composition
```python
from AlgoGraph import Graph
from AlgoGraph.transformers import (
    filter_vertices, map_vertices, largest_component,
    pipeline
)
from functools import partial

# Create reusable transformers
remove_inactive = filter_vertices(lambda v: v.get('active', True))
normalize_attrs = map_vertices(lambda v: v.with_attrs(
    normalized=v.get('value', 0) / 100
))
get_core = largest_component()

# Compose into pipeline
core_analysis = pipeline(
    remove_inactive,
    normalize_attrs,
    get_core
)

# Reuse pipeline on different graphs
g1_core = g1 | core_analysis
g2_core = g2 | core_analysis
g3_core = g3 | core_analysis

# Curry for partial application
filter_by_degree = partial(filter_vertices, lambda v: v.degree() > 3)
high_degree_analysis = pipeline(filter_by_degree, core_analysis)
```

**Improvement:** Full functional composition, reusable pipelines, currying support

---

## Summary: Line Count Comparison

| Operation | v1.1.0 Lines | v2.0.0 Lines | Reduction |
|-----------|--------------|--------------|-----------|
| Build simple graph | 14 | 10 | 30% |
| Load from data | 17 | 8 | 50% |
| Filter vertices | 16 | 13 | 20% |
| Transform pipeline | 20 | 10 | 50% |
| Shortest path | 24 | 14 | 40% |
| Complex query | 20 | 19 | 5% |
| Analytics pipeline | 28 | 10 | 65% |
| Export & visualize | 18 | 11 | 40% |
| **Average** | **19.6** | **11.9** | **39%** |

**Overall Result:** Proposed v2.0 API reduces code by ~40% on average while significantly improving readability, composability, and type safety.

---

## Key Improvements Summary

1. **Fluent Builders**: 30-50% less code for graph construction
2. **Transformer Pattern**: 50-65% less code for complex operations
3. **Selector DSL**: More readable and composable queries
4. **Generic Types**: Compile-time type safety
5. **Multiple Formats**: Native support for DOT, GraphML, etc.
6. **Functional Composition**: Reusable, composable pipelines
7. **Integrated APIs**: Less manual work (path reconstruction, etc.)

The proposed v2.0 API makes AlgoGraph:
- **More concise** (39% fewer lines)
- **More readable** (declarative vs imperative)
- **More composable** (pipelines, selectors, transformers)
- **More type-safe** (generic types, protocols)
- **More powerful** (rich ecosystem of operations)
