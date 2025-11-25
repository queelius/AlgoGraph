# Transformers

Transformers are composable operations that transform graphs using the `|` pipe operator. They enable functional-style programming with graphs, making complex workflows readable and maintainable.

## Basic Usage

```python
from AlgoGraph import Graph
from AlgoGraph.transformers import filter_vertices, filter_edges, stats

# Create a graph
graph = (Graph.builder()
    .add_vertex('A', active=True, score=90)
    .add_vertex('B', active=False, score=75)
    .add_vertex('C', active=True, score=60)
    .add_edge('A', 'B', weight=5)
    .add_edge('B', 'C', weight=10)
    .build())

# Apply single transformer
active_graph = graph | filter_vertices(lambda v: v.get('active'))

# Chain multiple transformers
result = (graph
    | filter_vertices(lambda v: v.get('active'))
    | filter_edges(lambda e: e.weight > 3)
    | stats())
```

## Available Transformers

### Filtering

#### `filter_vertices(predicate)`
Keep only vertices matching the predicate.

```python
from AlgoGraph.transformers import filter_vertices

# Keep active vertices
active = graph | filter_vertices(lambda v: v.get('active'))

# Keep high-degree vertices
hubs = graph | filter_vertices(lambda v: graph.degree(v.id) > 5)
```

#### `filter_edges(predicate)`
Keep only edges matching the predicate.

```python
from AlgoGraph.transformers import filter_edges

# Keep heavy edges
heavy = graph | filter_edges(lambda e: e.weight > 10)

# Keep undirected edges
undirected = graph | filter_edges(lambda e: not e.directed)
```

### Mapping

#### `map_vertices(fn)`
Transform vertex attributes.

```python
from AlgoGraph.transformers import map_vertices

# Add computed attribute
with_score = graph | map_vertices(
    lambda v: v.with_attrs(score=v.get('value', 0) * 2)
)

# Normalize names
normalized = graph | map_vertices(
    lambda v: Vertex(v.id.lower(), attrs=v.attrs)
)
```

#### `map_edges(fn)`
Transform edge attributes.

```python
from AlgoGraph.transformers import map_edges

# Scale weights
scaled = graph | map_edges(lambda e: e.with_weight(e.weight * 1000))

# Add labels
labeled = graph | map_edges(
    lambda e: Edge(e.source, e.target, weight=e.weight,
                   attrs={**e.attrs, 'label': f'{e.source}->{e.target}'})
)
```

### Structure Transformations

#### `reverse()`
Reverse all directed edges.

```python
from AlgoGraph.transformers import reverse

reversed_graph = graph | reverse()
# A -> B becomes B -> A
```

#### `to_undirected()`
Convert all edges to undirected.

```python
from AlgoGraph.transformers import to_undirected

undirected = graph | to_undirected()
```

#### `subgraph(vertex_ids)`
Extract a subgraph containing specific vertices.

```python
from AlgoGraph.transformers import subgraph

subset = graph | subgraph({'A', 'B', 'C'})
```

### Algorithm-Based Transformers

#### `largest_component()`
Extract the largest connected component.

```python
from AlgoGraph.transformers import largest_component

# Get biggest component
main = graph | largest_component()
```

#### `minimum_spanning_tree()`
Compute the minimum spanning tree.

```python
from AlgoGraph.transformers import minimum_spanning_tree

mst = graph | minimum_spanning_tree()
```

### Export Transformers

#### `to_dict()`
Convert graph to dictionary representation.

```python
from AlgoGraph.transformers import to_dict

data = graph | to_dict()
# {'vertices': [...], 'edges': [...]}
```

#### `to_adjacency_list()`
Export as adjacency list.

```python
from AlgoGraph.transformers import to_adjacency_list

adj = graph | to_adjacency_list()
# {'A': ['B', 'C'], 'B': ['C'], ...}
```

#### `stats()`
Compute graph statistics.

```python
from AlgoGraph.transformers import stats

info = graph | stats()
# {
#     'vertex_count': 100,
#     'edge_count': 250,
#     'density': 0.05,
#     'is_connected': True,
#     'components': 1
# }
```

## Pipeline Composition

Transformers can be composed into reusable pipelines:

```python
from AlgoGraph.transformers import Pipeline, filter_vertices, largest_component, stats

# Create reusable pipeline
analyze_active = Pipeline(
    filter_vertices(lambda v: v.get('active')),
    largest_component(),
    stats()
)

# Apply to any graph
result = graph | analyze_active
```

You can also compose transformers directly:

```python
# Compose with |
pipeline = filter_vertices(pred1) | filter_edges(pred2) | stats()

# Apply pipeline
result = graph | pipeline
```

## Conditional Transformers

Apply transformers conditionally:

```python
from AlgoGraph.transformers import filter_vertices

# Only filter if condition is met
transform = filter_vertices(lambda v: v.get('active')).when(
    lambda g: g.vertex_count > 100
)

result = graph | transform
# If vertex_count <= 100, graph passes through unchanged
```

## Common Patterns

### Data Cleaning Pipeline

```python
clean_pipeline = (
    filter_vertices(lambda v: v.id is not None)
    | filter_edges(lambda e: e.weight > 0)
    | map_vertices(lambda v: v.with_attrs(
        name=v.get('name', '').strip().lower()
    ))
)

clean_graph = raw_graph | clean_pipeline
```

### Analysis Pipeline

```python
from AlgoGraph.transformers import filter_vertices, largest_component, stats

def analyze_active_users(graph, min_score=50):
    return (graph
        | filter_vertices(lambda v: v.get('active'))
        | filter_vertices(lambda v: v.get('score', 0) >= min_score)
        | largest_component()
        | stats())

report = analyze_active_users(social_network, min_score=75)
```

### ETL Pipeline

```python
# Extract, Transform, Load pattern
result = (raw_data
    | filter_vertices(is_valid)           # Extract
    | map_vertices(normalize)             # Transform
    | map_edges(add_computed_weight)      # Transform
    | to_dict())                          # Load
```

## Performance Considerations

- Transformers create new Graph objects (immutability)
- Chain operations to minimize intermediate graphs
- For large graphs, consider using [Views](views.md) for lazy evaluation
- `stats()` computes connectivity which may be expensive for huge graphs
