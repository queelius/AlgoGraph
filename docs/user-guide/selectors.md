# Selectors

Selectors provide declarative pattern matching for vertices and edges using logical operators. Instead of writing lambda functions, you express *what* you're looking for with a composable query syntax.

## Basic Usage

```python
from AlgoGraph import Graph
from AlgoGraph.graph_selectors import vertex as v, edge as e

# Create a graph
graph = (Graph.builder()
    .add_vertex('Alice', age=30, role='admin', active=True)
    .add_vertex('Bob', age=25, role='user', active=True)
    .add_vertex('Charlie', age=35, role='user', active=False)
    .add_edge('Alice', 'Bob', weight=5)
    .add_edge('Bob', 'Charlie', weight=10)
    .build())

# Select vertices
admins = graph.select_vertices(v.attrs(role='admin'))
active_users = graph.select_vertices(v.attrs(active=True))

# Select edges
heavy_edges = graph.select_edges(e.weight(min_weight=7))
```

## Vertex Selectors

### `vertex.id(pattern)`
Match vertices by ID using glob or regex patterns.

```python
from AlgoGraph.graph_selectors import vertex as v

# Glob pattern
users = graph.select_vertices(v.id('user_*'))

# Regex pattern
numbered = graph.select_vertices(v.id(r'node_\d+'))

# Exact match
alice = graph.select_vertices(v.id('Alice'))
```

### `vertex.attrs(**attrs)`
Match vertices by attribute values.

```python
# Exact value match
admins = graph.select_vertices(v.attrs(role='admin'))

# Multiple attributes (AND)
active_admins = graph.select_vertices(v.attrs(role='admin', active=True))

# Callable predicates
adults = graph.select_vertices(v.attrs(age=lambda a: a >= 18))
high_score = graph.select_vertices(v.attrs(score=lambda s: s > 90))
```

### `vertex.degree(min_degree=None, max_degree=None, exact=None)`
Match vertices by their degree (number of connections).

```python
# Minimum degree
hubs = graph.select_vertices(v.degree(min_degree=10))

# Maximum degree
leaves = graph.select_vertices(v.degree(max_degree=1))

# Exact degree
isolated = graph.select_vertices(v.degree(exact=0))

# Range
moderate = graph.select_vertices(v.degree(min_degree=3, max_degree=7))
```

### `vertex.has_neighbor(selector)`
Match vertices that have neighbors matching another selector.

```python
# Has admin neighbor
connected_to_admin = graph.select_vertices(
    v.has_neighbor(v.attrs(role='admin'))
)

# Has high-degree neighbor
near_hubs = graph.select_vertices(
    v.has_neighbor(v.degree(min_degree=10))
)
```

## Edge Selectors

### `edge.weight(min_weight=None, max_weight=None, exact=None)`
Match edges by weight.

```python
from AlgoGraph.graph_selectors import edge as e

# Heavy edges
heavy = graph.select_edges(e.weight(min_weight=100))

# Light edges
light = graph.select_edges(e.weight(max_weight=10))

# Exact weight
unit = graph.select_edges(e.weight(exact=1.0))

# Range
medium = graph.select_edges(e.weight(min_weight=10, max_weight=50))
```

### `edge.directed(is_directed)`
Match edges by directionality.

```python
# Only directed edges
directed = graph.select_edges(e.directed(True))

# Only undirected edges
undirected = graph.select_edges(e.directed(False))
```

### `edge.source(vertex_selector)` / `edge.target(vertex_selector)`
Match edges by their source or target vertex.

```python
# Edges from admins
from_admin = graph.select_edges(e.source(v.attrs(role='admin')))

# Edges to high-degree vertices
to_hubs = graph.select_edges(e.target(v.degree(min_degree=10)))

# Both endpoints
admin_to_user = graph.select_edges(
    e.source(v.attrs(role='admin')) & e.target(v.attrs(role='user'))
)
```

### `edge.attrs(**attrs)`
Match edges by attribute values.

```python
# By type
highways = graph.select_edges(e.attrs(type='highway'))

# By multiple attributes
important = graph.select_edges(e.attrs(priority='high', active=True))

# With callable
recent = graph.select_edges(e.attrs(timestamp=lambda t: t > cutoff))
```

## Logical Operators

Combine selectors with logical operators for complex queries.

### AND (`&`)
Both conditions must match.

```python
# Active AND admin
active_admins = graph.select_vertices(
    v.attrs(active=True) & v.attrs(role='admin')
)

# High degree AND has specific attribute
power_users = graph.select_vertices(
    v.degree(min_degree=10) & v.attrs(verified=True)
)
```

### OR (`|`)
Either condition matches.

```python
# Admin OR moderator
staff = graph.select_vertices(
    v.attrs(role='admin') | v.attrs(role='moderator')
)

# High degree OR high score
important = graph.select_vertices(
    v.degree(min_degree=10) | v.attrs(score=lambda s: s > 90)
)
```

### NOT (`~`)
Negation of a condition.

```python
# NOT banned
active = graph.select_vertices(~v.attrs(banned=True))

# High degree but NOT admin
non_admin_hubs = graph.select_vertices(
    v.degree(min_degree=10) & ~v.attrs(role='admin')
)
```

### XOR (`^`)
Exactly one condition matches (exclusive or).

```python
# Either admin OR high-degree, but not both
either_or = graph.select_vertices(
    v.attrs(role='admin') ^ v.degree(min_degree=10)
)
```

## Complex Queries

Combine operators for sophisticated queries:

```python
# Active power users who are not banned
power_users = graph.select_vertices(
    v.attrs(active=True) &
    v.degree(min_degree=10) &
    ~v.attrs(banned=True)
)

# Staff or VIPs, excluding suspended accounts
privileged = graph.select_vertices(
    (v.attrs(role='admin') | v.attrs(role='moderator') | v.attrs(vip=True)) &
    ~v.attrs(suspended=True)
)

# Heavy edges from admins to regular users
admin_traffic = graph.select_edges(
    e.source(v.attrs(role='admin')) &
    e.target(~v.attrs(role='admin')) &
    e.weight(min_weight=100)
)
```

## Selector Methods

### `selector.first(graph)`
Get the first matching element, or None.

```python
admin = v.attrs(role='admin').first(graph)
if admin:
    print(f"Found admin: {admin.id}")
```

### `selector.count(graph)`
Count matching elements.

```python
num_admins = v.attrs(role='admin').count(graph)
print(f"Found {num_admins} admins")
```

### `selector.exists(graph)`
Check if any element matches.

```python
if v.attrs(role='superadmin').exists(graph):
    print("Has superadmin")
```

### `selector.ids(graph)` (VertexSelector only)
Get set of matching vertex IDs.

```python
admin_ids = v.attrs(role='admin').ids(graph)
# {'alice', 'bob', 'charlie'}
```

## Integration with Transformers

Selectors work seamlessly with transformers:

```python
from AlgoGraph.transformers import filter_vertices, subgraph, stats

# Use selector to get IDs, then create subgraph
admin_ids = v.attrs(role='admin').ids(graph)
admin_subgraph = graph | subgraph(admin_ids) | stats()

# Or use selector predicate directly
active = graph | filter_vertices(lambda vtx: v.attrs(active=True).matches(vtx, graph))
```

## Common Patterns

### Finding Anomalies

```python
# Isolated vertices (no connections)
isolated = graph.select_vertices(v.degree(exact=0))

# Vertices with unusual patterns
suspicious = graph.select_vertices(
    v.degree(min_degree=100) & v.attrs(account_age=lambda a: a < 7)
)
```

### Network Analysis

```python
# Bridge nodes (connect different groups)
bridges = graph.select_vertices(
    v.has_neighbor(v.attrs(group='A')) &
    v.has_neighbor(v.attrs(group='B'))
)

# Influencer candidates
influencers = graph.select_vertices(
    v.degree(min_degree=50) &
    v.attrs(followers=lambda f: f > 10000)
)
```

### Data Validation

```python
# Missing required attributes
invalid = graph.select_vertices(
    ~v.attrs(email=lambda e: e is not None)
)

# Inconsistent edges
broken = graph.select_edges(
    e.weight(max_weight=0)  # Negative or zero weight
)
```
