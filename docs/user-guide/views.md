# Views

Views provide lazy evaluation for graph operations—filtering and transforming graphs without copying data until you need it. This is essential for working efficiently with large graphs.

## Basic Usage

```python
from AlgoGraph import Graph
from AlgoGraph.views import filtered_view

# Create a large graph
graph = create_large_graph()  # 1 million vertices

# Create view WITHOUT copying
view = filtered_view(
    graph,
    vertex_filter=lambda v: v.get('active')
)

# Iterate lazily (no full copy created)
for vertex in view.vertices():
    process(vertex)

# Only copy when you need a concrete Graph
small_graph = view.materialize()
```

## View Types

### FilteredView

Filter vertices and/or edges without copying:

```python
from AlgoGraph.views import filtered_view

# Filter vertices only
active_view = filtered_view(
    graph,
    vertex_filter=lambda v: v.get('active')
)

# Filter edges only
heavy_view = filtered_view(
    graph,
    edge_filter=lambda e: e.weight > 100
)

# Filter both
combined = filtered_view(
    graph,
    vertex_filter=lambda v: v.get('active'),
    edge_filter=lambda e: e.weight > 10
)
```

Note: When filtering vertices, edges connecting to removed vertices are automatically excluded.

### SubGraphView

View a specific subset of vertices:

```python
from AlgoGraph.views import subgraph_view

# View only specific vertices
view = subgraph_view(graph, vertex_ids={'A', 'B', 'C', 'D'})

# Only edges between included vertices are visible
for edge in view.edges():
    print(edge)  # Only edges within {A, B, C, D}
```

### ReversedView

View the graph with reversed edge directions:

```python
from AlgoGraph.views import reversed_view

# Create reversed view
rev = reversed_view(graph)

# A -> B becomes B -> A
for edge in rev.edges():
    print(f"{edge.source} -> {edge.target}")
```

Undirected edges remain unchanged.

### UndirectedView

View all edges as undirected:

```python
from AlgoGraph.views import undirected_view

# Convert directed edges to undirected
undir = undirected_view(graph)

for edge in undir.edges():
    assert not edge.directed
```

### NeighborhoodView

View the k-hop neighborhood around a vertex:

```python
from AlgoGraph.views import neighborhood_view

# 1-hop neighbors
local = neighborhood_view(graph, center='Alice', k=1)

# 2-hop neighborhood (friends of friends)
extended = neighborhood_view(graph, center='Alice', k=2)

# Just the center vertex
just_center = neighborhood_view(graph, center='Alice', k=0)
```

The view includes all vertices within k hops and edges between them.

### ComponentView

View a specific connected component:

```python
from AlgoGraph.views import component_view
from AlgoGraph.algorithms import connected_components

# Get components
components = connected_components(graph)
largest = max(components, key=len)

# View largest component
main = component_view(graph, vertex_ids=largest)
```

## View Properties

All views provide these properties and methods:

```python
# Properties
view.vertex_count   # Number of vertices (computed lazily)
view.edge_count     # Number of edges (computed lazily)
view.graph          # Reference to underlying graph

# Iteration (lazy)
for v in view.vertices():
    process(v)

for e in view.edges():
    process(e)

# Materialization
concrete_graph = view.materialize()
```

## Lazy Evaluation

Views compute results lazily—nothing is copied or computed until you iterate or materialize:

```python
# These are instant (O(1))
view1 = filtered_view(huge_graph, vertex_filter=pred1)
view2 = filtered_view(view1.materialize(), edge_filter=pred2)

# This iterates through vertices (O(n))
for v in view1.vertices():
    if should_stop(v):
        break  # Can stop early!

# This computes everything (O(n + m))
count = view1.vertex_count  # Must iterate all vertices
```

## Composing Views

You can compose views for complex operations:

```python
from AlgoGraph.views import filtered_view, neighborhood_view

# Start with neighborhood
local = neighborhood_view(graph, center='Alice', k=2)

# Filter the neighborhood
active_local = filtered_view(
    local.materialize(),
    vertex_filter=lambda v: v.get('active')
)

# Get final result
result = active_local.materialize()
```

## Views vs Transformers

| Aspect | Views | Transformers |
|--------|-------|--------------|
| Evaluation | Lazy | Eager |
| Memory | Minimal until materialize | Creates new Graph |
| Use case | Large graphs, early termination | Pipelines, composition |
| Syntax | Function call | Pipe operator `\|` |

**Use Views when:**
- Working with large graphs
- You might not need all results
- Memory is a concern
- You want to iterate without copying

**Use Transformers when:**
- Building reusable pipelines
- Chaining multiple operations
- You need the result as a Graph
- Code clarity is priority

## Common Patterns

### Efficient Filtering

```python
from AlgoGraph.views import filtered_view

def find_first_matching(graph, predicate):
    """Find first vertex matching predicate without full scan."""
    view = filtered_view(graph, vertex_filter=predicate)
    for v in view.vertices():
        return v  # Return immediately
    return None

# Stops at first match, doesn't scan entire graph
admin = find_first_matching(huge_graph, lambda v: v.get('role') == 'admin')
```

### Memory-Efficient Analysis

```python
from AlgoGraph.views import filtered_view

def analyze_segment(graph, segment_filter):
    """Analyze a segment without copying the whole graph."""
    view = filtered_view(graph, vertex_filter=segment_filter)

    total = 0
    count = 0
    for v in view.vertices():
        total += v.get('value', 0)
        count += 1

    return total / count if count > 0 else 0

# Analyze without creating intermediate graphs
avg_value = analyze_segment(huge_graph, lambda v: v.get('region') == 'US')
```

### Exploring Large Graphs

```python
from AlgoGraph.views import neighborhood_view

def explore_from(graph, start, max_depth=3):
    """Explore graph from a starting point."""
    for depth in range(1, max_depth + 1):
        view = neighborhood_view(graph, center=start, k=depth)
        print(f"Depth {depth}: {view.vertex_count} vertices, {view.edge_count} edges")

        if view.vertex_count > 10000:
            print("Graph is large, stopping exploration")
            break

    return view.materialize()
```

### Conditional Materialization

```python
from AlgoGraph.views import filtered_view

def get_subgraph_if_small(graph, predicate, max_size=1000):
    """Only materialize if result is small enough."""
    view = filtered_view(graph, vertex_filter=predicate)

    # Check size first
    if view.vertex_count > max_size:
        raise ValueError(f"Result too large: {view.vertex_count} vertices")

    # Safe to materialize
    return view.materialize()
```

## Performance Tips

1. **Iterate when possible**: Use `for v in view.vertices()` instead of `view.vertex_count` if you don't need the count.

2. **Materialize once**: If you need to use the result multiple times, materialize once and reuse.

3. **Filter early**: Apply the most restrictive filter first to reduce work.

4. **Use appropriate view**: `subgraph_view` with known IDs is faster than `filtered_view` with predicates.

5. **Avoid nested iteration**: Don't iterate a view while iterating another view on the same graph.
