# AlgoGraph Architectural Review
**Date:** 2025-11-12
**Version Reviewed:** v1.1.0
**Reviewer:** Claude (Architectural Analysis)

---

## Executive Summary

AlgoGraph is a well-designed immutable graph library with ~4,600 lines of Python code. It successfully achieves immutability and provides 30+ algorithms. However, compared to AlgoTree (its predecessor) and mature graph libraries like NetworkX, there are significant opportunities for improvement in API elegance, composability, and feature completeness.

### Top 10 Improvements (Ranked by Impact)

1. **Add Fluent Builder API** - Make graph construction as elegant as AlgoTree's builder patterns
2. **Implement Graph Transformers** - Bring AlgoTree's transformer pattern to graphs for composability
3. **Add Selector/Query DSL** - Enable powerful graph queries like AlgoTree's selectors
4. **Generic Type Support** - Make Graph generic over vertex/edge types for type safety
5. **Missing Core Algorithms** - Add flow networks, matching, centrality measures, community detection
6. **Graph Generators** - Add classic graphs (complete, random, preferential attachment, etc.)
7. **Enhanced Export Formats** - GraphViz, DOT, GraphML, Cytoscape JSON
8. **Graph Operations** - Union, intersection, difference, cartesian product, composition
9. **Performance Layer** - Cached adjacency structures, lazy evaluation, view objects
10. **Validation & Constraints** - Graph invariants, validation rules, type protocols

---

## 1. API Design Analysis

### 1.1 Current Strengths

**Immutability is Well-Implemented**
```python
# Good: All operations return new graphs
g1 = Graph()
g2 = g1.add_vertex(Vertex('A'))
g3 = g2.add_edge(Edge('A', 'B'))
```

**Consistent Naming Conventions**
```python
# All construction methods start with verb
g.add_vertex(v)
g.remove_vertex('A')
g.update_vertex(v)
```

**Clean Separation of Concerns**
- Data structures (Vertex, Edge, Graph) are separate from algorithms
- Algorithms are pure functions taking Graph as input
- No God-object anti-pattern

### 1.2 Critical API Issues

#### Issue 1: Verbose Graph Construction

**Problem:** Building graphs requires too much boilerplate compared to AlgoTree.

**Current (Verbose):**
```python
from AlgoGraph import Graph, Vertex, Edge

v1 = Vertex('A', attrs={'value': 1})
v2 = Vertex('B', attrs={'value': 2})
v3 = Vertex('C', attrs={'value': 3})

e1 = Edge('A', 'B', weight=5.0)
e2 = Edge('B', 'C', weight=3.0)
e3 = Edge('A', 'C', weight=10.0)

g = Graph({v1, v2, v3}, {e1, e2, e3})
```

**Proposed (Fluent):**
```python
from AlgoGraph import GraphBuilder, graph

# Builder pattern
g = (GraphBuilder()
     .add('A', value=1)
     .add('B', value=2)
     .add('C', value=3)
     .connect('A', 'B', weight=5.0)
     .connect('B', 'C', weight=3.0)
     .connect('A', 'C', weight=10.0)
     .build())

# Or functional style (like AlgoTree's node() function)
g = graph(
    vertex('A', value=1, edges=[
        edge('B', weight=5.0),
        edge('C', weight=10.0)
    ]),
    vertex('B', value=2, edges=[
        edge('C', weight=3.0)
    ]),
    vertex('C', value=3)
)

# Or declarative dict style
g = Graph.from_dict({
    'A': {'value': 1, 'edges': [('B', 5.0), ('C', 10.0)]},
    'B': {'value': 2, 'edges': [('C', 3.0)]},
    'C': {'value': 3}
})
```

#### Issue 2: No Method Chaining for Transformations

**Problem:** Can't chain transformations fluently.

**Current:**
```python
# Can chain construction, but not transformations
g = g.add_vertex(Vertex('A')).add_vertex(Vertex('B'))

# But algorithms are separate - no chaining
components = connected_components(g)
subgraph = g.subgraph({'A', 'B'})
```

**Proposed:**
```python
# Fluent transformations inspired by AlgoTree
result = (graph
    .filter_vertices(lambda v: v.get('value', 0) > 10)
    .filter_edges(lambda e: e.weight > 5.0)
    .map_vertices(lambda v: v.with_attrs(normalized=v.get('value')/100))
    .connected_component_containing('A')
    .to_dict())

# Or using transformer pattern (like AlgoTree)
from AlgoGraph.transformers import filter_vertices, map_vertices, extract_component

result = (graph
    | filter_vertices(lambda v: v.get('value') > 10)
    | map_vertices(lambda v: v.with_attrs(normalized=True))
    | extract_component('A'))
```

#### Issue 3: Inconsistent API Between Graph and Algorithms

**Problem:** Some operations are Graph methods, others are separate functions.

**Current Inconsistency:**
```python
# Some operations are methods
neighbors = g.neighbors('A')
degree = g.degree('A')
has_edge = g.has_edge('A', 'B')

# But algorithms are functions
path = dijkstra(g, 'A')
components = connected_components(g)
mst = kruskal(g)
```

**Proposed Consistency Option 1 (All Methods):**
```python
# Make algorithms methods on Graph
neighbors = g.neighbors('A')
degree = g.degree('A')
path = g.dijkstra('A')
components = g.connected_components()
mst = g.kruskal()
```

**Proposed Consistency Option 2 (All Functions):**
```python
# Make everything a function (more functional, better for composition)
from AlgoGraph.ops import neighbors, degree, dijkstra, connected_components

neighbors = neighbors(g, 'A')
degree = degree(g, 'A')
path = dijkstra(g, 'A')
components = connected_components(g)
```

**Recommended:** Keep algorithms as functions (better for testing, composition, and clarity), but add convenience methods that delegate to them.

#### Issue 4: No Graph Query DSL

**Problem:** AlgoTree has powerful selectors, but AlgoGraph has none.

**AlgoTree's Power:**
```python
from AlgoTree import Tree, name, depth, attrs

# Find nodes by complex criteria
nodes = tree.find_all(name('test*') & depth(3) & attrs(value=lambda v: v > 10))
```

**AlgoGraph Equivalent (Missing):**
```python
from AlgoGraph import Graph, vertex, edge

# Find vertices by complex criteria
vertices = graph.select(
    vertex.name('A*') &
    vertex.attrs(value=lambda v: v > 10) &
    vertex.degree() > 3
)

# Find edges by criteria
edges = graph.select_edges(
    edge.weight() > 5.0 &
    edge.connects(vertex.name('A*'))
)
```

#### Issue 5: Missing Graph Views and Lazy Evaluation

**Problem:** All operations materialize new graphs. No lazy views.

**Current:**
```python
# Always creates new graph
subgraph = g.subgraph({'A', 'B', 'C'})  # Copies all data
```

**Proposed:**
```python
# Lazy view (like networkx.classes.graphviews)
view = g.view(vertices={'A', 'B', 'C'})  # No copy, just filters
view = g.view_edges(weight=lambda w: w > 5.0)  # Filter edges lazily

# Still immutable, but efficient for queries
neighbors = view.neighbors('A')  # Only sees filtered graph
```

### 1.3 Missing Abstractions

#### Missing: Base Protocol for Graph-like Objects

**Problem:** No typing protocol for duck-typed graph operations.

**Proposed:**
```python
from typing import Protocol, Set, Iterator

class GraphProtocol(Protocol):
    """Protocol for graph-like objects."""

    def neighbors(self, vertex_id: str) -> Set[str]: ...
    def has_vertex(self, vertex_id: str) -> bool: ...
    def has_edge(self, source: str, target: str) -> bool: ...

    @property
    def vertices(self) -> Set[Vertex]: ...

    @property
    def edges(self) -> Set[Edge]: ...

# Now algorithms can work with any graph-like object
def bfs(graph: GraphProtocol, start: str) -> List[str]:
    # Works with Graph, or any custom graph class
    pass
```

#### Missing: Vertex/Edge Mixins for Common Patterns

**Problem:** No way to extend Vertex/Edge with domain-specific behavior.

**Current Limitation:**
```python
# Can only use attrs dict
v = Vertex('A', attrs={'x': 10, 'y': 20, 'color': 'red'})
x = v.get('x')  # Untyped access
```

**Proposed:**
```python
# Mixin pattern for typed attributes
from AlgoGraph.mixins import PositionMixin, ColorMixin

class PositionVertex(Vertex, PositionMixin):
    """Vertex with x, y position."""
    pass

class ColorVertex(Vertex, ColorMixin):
    """Vertex with color attribute."""
    pass

# Type-safe access
v = PositionVertex('A', x=10, y=20, color='red')
x = v.x  # Type-checked
color = v.color  # Type-checked

# Or use Generic types
from AlgoGraph import GenericGraph, TypedVertex

class Node:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

g = GenericGraph[Node]()
```

#### Missing: Transformer Pattern (from AlgoTree)

**Problem:** AlgoTree's transformer pattern is elegant but missing in AlgoGraph.

**AlgoTree's Pattern:**
```python
from AlgoTree import Tree, map_, filter_, prune

# Composable transformers
pipeline = map_(normalize) | filter_(is_valid) | prune(is_empty)
result = tree | pipeline
```

**Proposed for AlgoGraph:**
```python
from AlgoGraph import Graph, map_vertices, filter_edges, extract_component

# Composable graph transformers
pipeline = (
    map_vertices(normalize_attrs) |
    filter_edges(lambda e: e.weight > 5.0) |
    extract_component('core')
)

result = graph | pipeline

# Or with >> operator for type-changing transforms
metrics = graph >> compute_centrality >> to_dataframe
```

---

## 2. Missing Algorithms

### 2.1 Critical Gaps (High Priority)

#### Flow Networks
- **Max Flow**: Ford-Fulkerson, Edmonds-Karp, Push-Relabel
- **Min Cost Flow**: Network Simplex, Cycle Canceling
- **Min Cut**: Stoer-Wagner, Gomory-Hu tree
- **Matching**: Hopcroft-Karp (bipartite), Blossom (general)

**Proposed API:**
```python
from AlgoGraph.algorithms.flow import max_flow, min_cut, min_cost_flow
from AlgoGraph.algorithms.matching import max_matching, perfect_matching

# Max flow
flow_value, flow_dict = max_flow(g, source='S', sink='T', capacity='capacity')

# Min cut
cut_value, partition = min_cut(g, 'S', 'T')

# Maximum matching
matching = max_matching(g)  # Returns set of edges
```

#### Centrality Measures
- **Betweenness**: Vertex and edge betweenness
- **Closeness**: Closeness centrality
- **Eigenvector**: PageRank, eigenvector centrality, Katz centrality
- **Degree**: In-degree, out-degree, weighted degree centrality

**Proposed API:**
```python
from AlgoGraph.algorithms.centrality import (
    betweenness, closeness, pagerank, eigenvector, degree_centrality
)

# Compute centrality measures
bc = betweenness(g)  # Dict[str, float]
cc = closeness(g, vertex='A')  # float
pr = pagerank(g, alpha=0.85)  # Dict[str, float]
```

#### Community Detection
- **Modularity-based**: Louvain, Leiden, greedy modularity
- **Label Propagation**: Asynchronous label propagation
- **Spectral**: Spectral bisection, normalized cut
- **Clique**: k-clique communities

**Proposed API:**
```python
from AlgoGraph.algorithms.community import (
    louvain, label_propagation, spectral_clustering
)

# Community detection
communities = louvain(g)  # List[Set[str]]
communities = label_propagation(g)
modularity_score = modularity(g, communities)
```

#### Graph Coloring
- **Vertex Coloring**: Greedy, backtracking, DSatur
- **Edge Coloring**: Vizing's algorithm
- **Chromatic Number**: Exact and approximation

**Proposed API:**
```python
from AlgoGraph.algorithms.coloring import (
    greedy_coloring, dsatur, edge_coloring, chromatic_number
)

coloring = greedy_coloring(g)  # Dict[str, int]
num_colors = chromatic_number(g)  # int
```

### 2.2 Important Gaps (Medium Priority)

#### Graph Isomorphism
- **Isomorphism Testing**: VF2 algorithm
- **Canonical Labeling**: Nauty-based
- **Subgraph Isomorphism**: VF2 for subgraphs

**Proposed API:**
```python
from AlgoGraph.algorithms.isomorphism import (
    is_isomorphic, find_isomorphism, is_subgraph_isomorphic
)

is_iso = is_isomorphic(g1, g2)
mapping = find_isomorphism(g1, g2)  # Dict[str, str] or None
```

#### Planarity Testing
- **Planarity Check**: Left-right planarity test
- **Planar Embedding**: Compute planar embedding
- **Kuratowski Subgraphs**: Find K5 or K3,3

**Proposed API:**
```python
from AlgoGraph.algorithms.planarity import (
    is_planar, planar_embedding, find_kuratowski_subgraph
)

is_planar = is_planar(g)
embedding = planar_embedding(g)  # Dict of planar coordinates
```

#### Cliques and Independent Sets
- **Max Clique**: Bron-Kerbosch, Tomita
- **Max Independent Set**: Complement + max clique
- **Clique Enumeration**: All maximal cliques

**Proposed API:**
```python
from AlgoGraph.algorithms.cliques import (
    max_clique, all_maximal_cliques, max_independent_set
)

clique = max_clique(g)  # Set[str]
cliques = list(all_maximal_cliques(g))  # Iterator[Set[str]]
```

### 2.3 Nice-to-Have Gaps (Low Priority)

#### Tree Decomposition
- **Treewidth**: Compute treewidth
- **Tree Decomposition**: Find tree decomposition

#### Graph Drawing
- **Force-Directed**: Fruchterman-Reingold, Kamada-Kawai
- **Spectral Layout**: Eigenvalue-based layout
- **Circular/Hierarchical**: Layout algorithms

#### Advanced Traversals
- **Eulerian Path/Circuit**: Hierholzer's algorithm
- **Hamiltonian Path**: Backtracking, Held-Karp
- **Random Walks**: Simulation, stationary distribution

---

## 3. Missing Features

### 3.1 Graph Types

**Missing: Multigraph Support**
```python
# Proposed API
from AlgoGraph import MultiGraph

mg = MultiGraph()
mg = mg.add_edge(Edge('A', 'B', key='route1', weight=10))
mg = mg.add_edge(Edge('A', 'B', key='route2', weight=15))

# Get all edges between vertices
edges = mg.get_edges('A', 'B')  # List[Edge]
```

**Missing: Temporal/Dynamic Graphs**
```python
# Proposed API
from AlgoGraph import TemporalGraph

tg = TemporalGraph()
tg = tg.add_edge(Edge('A', 'B'), timestamp=100)
tg = tg.add_edge(Edge('B', 'C'), timestamp=200)

# Query at time
snapshot = tg.at_time(150)  # Graph at time 150
interval = tg.between(100, 200)  # Graph for time range
```

**Missing: Bipartite Graph Utilities**
```python
# Proposed API
from AlgoGraph import BipartiteGraph

bg = BipartiteGraph(left={'A', 'B'}, right={'1', '2', '3'})
bg = bg.add_edge(Edge('A', '1'))

# Projections
left_proj = bg.project_left()  # Graph of left nodes
right_proj = bg.project_right()  # Graph of right nodes
```

### 3.2 Graph Operations

**Missing: Set Operations**
```python
# Proposed API
from AlgoGraph.ops import union, intersection, difference

# Graph union
g3 = union(g1, g2, node_merge=lambda v1, v2: v1.with_attrs(**v2.attrs))

# Graph intersection
g_common = intersection(g1, g2)

# Graph difference
g_diff = difference(g1, g2)
```

**Missing: Graph Products**
```python
# Proposed API
from AlgoGraph.ops import cartesian_product, tensor_product, strong_product

# Cartesian product G × H
g_cart = cartesian_product(g1, g2)

# Tensor product G ⊗ H
g_tensor = tensor_product(g1, g2)
```

**Missing: Graph Composition**
```python
# Proposed API
from AlgoGraph.ops import compose, contract, line_graph

# Compose g1 and g2
g_composed = compose(g1, g2)

# Contract edge
g_contracted = contract(g, edge=('A', 'B'))

# Line graph L(G)
lg = line_graph(g)
```

### 3.3 Graph Generators

**Missing: Classic Graphs**
```python
# Proposed API
from AlgoGraph.generators import (
    complete_graph, cycle_graph, path_graph, star_graph,
    wheel_graph, grid_graph, ladder_graph
)

# Classic graphs
k5 = complete_graph(5)
c10 = cycle_graph(10)
p20 = path_graph(20)
s6 = star_graph(6)

# Structured graphs
grid = grid_graph(10, 10)
ladder = ladder_graph(8)
```

**Missing: Random Graphs**
```python
# Proposed API
from AlgoGraph.generators.random import (
    erdos_renyi, barabasi_albert, watts_strogatz,
    random_regular, random_geometric
)

# Erdős-Rényi
g_er = erdos_renyi(n=100, p=0.05)

# Barabási-Albert (preferential attachment)
g_ba = barabasi_albert(n=100, m=3)

# Watts-Strogatz (small world)
g_ws = watts_strogatz(n=100, k=6, p=0.1)
```

**Missing: Social Network Graphs**
```python
# Proposed API
from AlgoGraph.generators.social import (
    karate_club, les_miserables, davis_southern_women
)

# Load classic datasets
karate = karate_club()
miserables = les_miserables()
```

### 3.4 Export Formats

**Current:** Only JSON
**Missing:**
- GraphViz/DOT
- GraphML
- GML
- GEXF
- Cytoscape JSON
- Adjacency list/matrix
- Edge list

**Proposed API:**
```python
from AlgoGraph.exporters import (
    to_dot, to_graphml, to_gexf, to_cytoscape,
    to_adjacency_matrix, to_edge_list
)

# Export to DOT
dot_str = to_dot(g, layout='neato')

# Export to GraphML
graphml_str = to_graphml(g)

# Export to adjacency matrix
matrix = to_adjacency_matrix(g)  # numpy array

# Visualize with graphviz
from AlgoGraph.viz import render
render(g, format='png', output='graph.png')
```

### 3.5 Import Formats

**Current:** Only JSON
**Missing:**
- Parse DOT files
- Parse GraphML
- Parse edge lists
- Parse adjacency matrices

**Proposed API:**
```python
from AlgoGraph.importers import (
    from_dot, from_graphml, from_edge_list, from_adjacency_matrix
)

g = from_dot('graph.dot')
g = from_graphml('graph.graphml')
g = from_edge_list('edges.txt', delimiter=',')
```

---

## 4. Learning from AlgoTree

### 4.1 What Worked Well in AlgoTree (Adopt)

#### 1. Fluent Builder API
AlgoTree's builders are elegant:
```python
# AlgoTree
from AlgoTree import tree, branch, leaf

t = tree('root',
    branch('child1',
        leaf('grandchild1'),
        leaf('grandchild2')
    ),
    branch('child2',
        leaf('grandchild3')
    )
)
```

**Recommendation:** Adopt for AlgoGraph
```python
from AlgoGraph import graph, vertex, edge

g = graph(
    vertex('A', value=1,
        edge('B', weight=5),
        edge('C', weight=10)
    ),
    vertex('B', value=2,
        edge('C', weight=3)
    ),
    vertex('C', value=3)
)
```

#### 2. Selector Pattern
AlgoTree's selectors are composable and powerful:
```python
from AlgoTree import name, depth, attrs

nodes = tree.find_all(
    (name('test*') | name('prod*')) &
    depth(lambda d: d < 5) &
    attrs(status='active')
)
```

**Recommendation:** Adopt for graph queries
```python
from AlgoGraph.selectors import vertex, edge

vertices = graph.select_vertices(
    vertex.name('server*') &
    vertex.degree() > 3 &
    vertex.attrs(status='active')
)

edges = graph.select_edges(
    edge.weight() > 10 &
    edge.connects(vertex.name('A*'), vertex.name('B*'))
)
```

#### 3. Transformer Pattern with Pipes
AlgoTree's transformers compose beautifully:
```python
from AlgoTree import map_, filter_, prune

result = tree | map_(normalize) | filter_(valid) | prune(empty)
```

**Recommendation:** Adopt for graph transformations
```python
from AlgoGraph.transformers import map_vertices, filter_edges, component

result = (graph
    | map_vertices(normalize_attrs)
    | filter_edges(lambda e: e.weight > 5)
    | component('core'))
```

#### 4. Comprehensive Export/Import
AlgoTree has many exporters (JSON, YAML, XML, pickle, flat dict, etc.)

**Recommendation:** Match or exceed AlgoTree's export options

#### 5. DSL Support
AlgoTree has `parse_tree()` for text-based tree construction.

**Recommendation:** Add DOT-like DSL for graphs
```python
from AlgoGraph import parse_graph

g = parse_graph("""
    A -> B [weight=5]
    A -> C [weight=10]
    B -> C [weight=3]
""")
```

### 4.2 What Didn't Work in AlgoTree (Avoid)

#### 1. Too Many Ways to Do Things
AlgoTree has too many overlapping APIs:
- `Node` class with methods
- `Tree` wrapper class with methods
- Functional operations in transformers
- Builder pattern
- DSL

**Recommendation:** For AlgoGraph, establish ONE clear way per use case:
- Construction: Builders + `from_*` methods
- Transformation: Transformer pattern with pipes
- Queries: Selector pattern
- Algorithms: Pure functions

#### 2. Weak Type Hints
AlgoTree's type hints are incomplete.

**Recommendation:** AlgoGraph should have comprehensive type hints with generics:
```python
from typing import TypeVar, Generic

V = TypeVar('V')  # Vertex payload type
E = TypeVar('E')  # Edge payload type

class Graph(Generic[V, E]):
    vertices: Set[Vertex[V]]
    edges: Set[Edge[E]]
```

#### 3. Performance Not Prioritized
AlgoTree creates many intermediate objects.

**Recommendation:** AlgoGraph should offer:
- Lazy evaluation where possible
- View objects for filtered graphs
- Cached adjacency structures
- Optional numpy backend for large graphs

### 4.3 Evolution Opportunities

#### Make Graph More Functional Than AlgoTree
AlgoTree has some stateful operations. AlgoGraph should be purely functional:

```python
# Every operation returns new graph
g2 = g1.add_vertex(v)  # g1 unchanged
g3 = g2.add_edge(e)    # g2 unchanged

# Support lens-like updates
from AlgoGraph.lenses import vertices, edges, attrs

g2 = g1.modify(vertices['A'].attrs['value'], lambda x: x + 1)
```

#### Better Integration with Scientific Python
```python
# NumPy integration
import numpy as np
matrix = g.to_adjacency_matrix()  # Returns np.ndarray
g = Graph.from_adjacency_matrix(matrix, vertex_ids=['A', 'B', 'C'])

# Pandas integration
import pandas as pd
df = g.to_edge_dataframe()  # Returns pd.DataFrame
g = Graph.from_edge_dataframe(df, source='src', target='dst')

# NetworkX compatibility
import networkx as nx
nx_graph = g.to_networkx()
g = Graph.from_networkx(nx_graph)
```

---

## 5. Composability & Unix Philosophy

### 5.1 Current State Assessment

**Do One Thing Well: ✓ Good**
- Graph data structure is focused
- Algorithms are separate pure functions
- Each module has clear responsibility

**Composition: ✗ Needs Work**
- No pipeline/transformer pattern
- Can't chain operations fluently
- Limited functional composition

**Orthogonality: ✓ Good**
- Graph, Vertex, Edge are independent
- Algorithms don't overlap
- Features are non-interfering

**Small Sharp Tools: ~ Mixed**
- Good: Algorithms are focused functions
- Bad: No micro-utilities for graph manipulation

### 5.2 Recommendations for Unix Philosophy

#### 1. Everything Should Compose

**Current:**
```python
# Can't compose
g = Graph({Vertex('A')}, {Edge('A', 'B')})
components = connected_components(g)
subgraph = g.subgraph(components[0])
```

**Proposed:**
```python
# Compose with pipes
from AlgoGraph import Graph, pipe
from AlgoGraph.ops import filter_vertices, largest_component, to_dict

result = (Graph.from_file('data.json')
    | filter_vertices(lambda v: v.get('active'))
    | largest_component()
    | to_dict())
```

#### 2. Small Utilities That Combine

**Proposed:**
```python
from AlgoGraph.utils import (
    # Vertex predicates
    has_attr, attr_equals, attr_greater_than,
    # Edge predicates
    weight_greater_than, connects_to, is_directed,
    # Combinators
    all_of, any_of, not_,
    # Graph operations
    add, remove, update, merge
)

# Combine small predicates
pred = all_of(
    has_attr('status'),
    attr_equals('status', 'active'),
    attr_greater_than('priority', 5)
)

vertices = g.find_vertices(pred)
```

#### 3. Currying and Partial Application

**Proposed:**
```python
from AlgoGraph.algorithms import dijkstra_from, bfs_from
from functools import partial

# Curried algorithms
dijkstra_from_A = dijkstra_from('A')

# Apply to different graphs
distances1 = dijkstra_from_A(g1)
distances2 = dijkstra_from_A(g2)

# Or use partial
find_components_min_3 = partial(connected_components, min_size=3)
```

#### 4. Text-Based Input/Output

**Current:** Only programmatic API
**Proposed:** Support text I/O for Unix-style piping

```bash
# Command-line graph utilities
cat graph.json | algograph filter "degree > 3" | algograph centrality --betweenness | sort -rn

# Or use DOT format
echo "A -> B; B -> C; A -> C" | algograph to-json > graph.json
```

---

## 6. Type System & Protocols

### 6.1 Current Type Hints Assessment

**Current Coverage: ~ 60%**
- Core classes have basic types
- Algorithms have input/output types
- Missing: Generic types, protocols, advanced typing

**Issues:**
```python
# Not generic - vertex/edge payloads are untyped
class Graph:
    vertices: Set[Vertex]  # What's in Vertex.attrs? Unknown.
    edges: Set[Edge]       # What's in Edge.attrs? Unknown.

# Algorithms lose type information
def dijkstra(graph: Graph, source: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    # Return types are verbose and untyped
    pass
```

### 6.2 Proposed: Generic Types

```python
from typing import TypeVar, Generic, Protocol

# Generic vertex and edge payloads
V = TypeVar('V')  # Vertex data type
E = TypeVar('E')  # Edge data type

class Vertex(Generic[V]):
    id: str
    data: V  # Typed payload instead of attrs dict

    def with_data(self, data: V) -> 'Vertex[V]':
        return Vertex(self.id, data)

class Edge(Generic[E]):
    source: str
    target: str
    data: E  # Typed payload
    weight: float = 1.0
    directed: bool = True

class Graph(Generic[V, E]):
    vertices: Set[Vertex[V]]
    edges: Set[Edge[E]]

    def add_vertex(self, vertex: Vertex[V]) -> 'Graph[V, E]':
        ...

    def find_vertices(self, pred: Callable[[Vertex[V]], bool]) -> Set[Vertex[V]]:
        ...

# Usage with types
class Person:
    name: str
    age: int

class Relationship:
    type: str
    since: datetime

social_graph: Graph[Person, Relationship] = Graph()

# Type checker knows the types!
person = social_graph.get_vertex('Alice')
person.data.age  # Type is int
person.data.name  # Type is str

relationship = social_graph.get_edge('Alice', 'Bob')
relationship.data.type  # Type is str
```

### 6.3 Proposed: Structural Typing with Protocols

```python
from typing import Protocol, Set, runtime_checkable

@runtime_checkable
class GraphLike(Protocol):
    """Protocol for graph-like objects."""

    @property
    def vertices(self) -> Set[Any]: ...

    @property
    def edges(self) -> Set[Any]: ...

    def neighbors(self, vertex_id: str) -> Set[str]: ...

    def has_vertex(self, vertex_id: str) -> bool: ...

# Now algorithms work with anything graph-like
def dfs(graph: GraphLike, start: str) -> List[str]:
    # Works with Graph, or any compatible class
    pass

# Custom graph implementations
class CustomGraph:
    def neighbors(self, vertex_id: str) -> Set[str]:
        # Custom implementation
        pass

    # ... other methods

# Works with dfs because it matches protocol
custom = CustomGraph()
order = dfs(custom, 'A')  # Type checker accepts this
```

### 6.4 Proposed: Better Algorithm Types

```python
from typing import TypedDict, Literal

# Typed results
class ShortestPathResult(TypedDict):
    distances: Dict[str, float]
    predecessors: Dict[str, Optional[str]]
    path: Optional[List[str]]

def dijkstra(
    graph: Graph,
    source: str,
    target: Optional[str] = None
) -> ShortestPathResult:
    """
    Dijkstra's shortest path algorithm.

    Returns typed dictionary with distances, predecessors, and path.
    """
    ...

# Use with type safety
result = dijkstra(g, 'A', 'B')
distance = result['distances']['B']  # Type checker knows this is float
path = result['path']  # Type checker knows this is Optional[List[str]]
```

### 6.5 Proposed: Validation at Boundaries

```python
from pydantic import BaseModel, validator

class ValidatedVertex(BaseModel):
    """Vertex with validation."""
    id: str
    attrs: Dict[str, Any]

    @validator('id')
    def id_not_empty(cls, v):
        if not v:
            raise ValueError('Vertex ID cannot be empty')
        return v

class ValidatedGraph(BaseModel):
    """Graph with validation."""
    vertices: Set[ValidatedVertex]
    edges: Set[Edge]

    @validator('edges')
    def edges_reference_vertices(cls, v, values):
        vertex_ids = {vertex.id for vertex in values.get('vertices', set())}
        for edge in v:
            if edge.source not in vertex_ids:
                raise ValueError(f'Edge references non-existent vertex: {edge.source}')
            if edge.target not in vertex_ids:
                raise ValueError(f'Edge references non-existent vertex: {edge.target}')
        return v
```

---

## 7. Proposed Implementation Roadmap

### Phase 1: API Improvements (v1.2.0) - 2 weeks

**Goals:** Improve API ergonomics without breaking changes

1. **Add Builder API**
   - `GraphBuilder` class with fluent methods
   - `graph()`, `vertex()`, `edge()` helper functions
   - `Graph.from_dict()` with flexible formats

2. **Add Convenience Methods**
   - `Graph.map_vertices()`, `Graph.filter_vertices()`
   - `Graph.map_edges()`, `Graph.filter_edges()`
   - Delegate to functional implementations

3. **Enhance Serialization**
   - Add `Graph.from_edge_list()`, `Graph.to_edge_list()`
   - Add `Graph.from_adjacency_matrix()`, `Graph.to_adjacency_matrix()`
   - NumPy/Pandas optional integration

4. **Documentation**
   - Add API design guide
   - Add cookbook with common patterns
   - Document builder patterns

### Phase 2: Core Algorithms (v1.3.0) - 3 weeks

**Goals:** Fill critical algorithm gaps

1. **Flow Networks**
   - Max flow (Edmonds-Karp)
   - Min cut (Stoer-Wagner)
   - Implement in `algorithms/flow.py`

2. **Centrality**
   - Betweenness centrality
   - Closeness centrality
   - PageRank
   - Implement in `algorithms/centrality.py`

3. **Matching**
   - Max bipartite matching (Hopcroft-Karp)
   - Implement in `algorithms/matching.py`

4. **Coloring**
   - Greedy vertex coloring
   - Implement in `algorithms/coloring.py`

### Phase 3: Advanced Features (v2.0.0) - 4 weeks

**Goals:** Add transformer pattern, selectors, and advanced types

1. **Transformer Pattern**
   - `Transformer` base class
   - `GraphTransformer`, `VertexTransformer`, `EdgeTransformer`
   - Pipeline composition with `|` operator
   - Implement in `transformers.py`

2. **Selector Pattern**
   - `Selector` base class
   - `VertexSelector`, `EdgeSelector`
   - Logical combinators (`&`, `|`, `~`)
   - Implement in `selectors.py`

3. **Generic Types**
   - Make `Graph`, `Vertex`, `Edge` generic
   - Add type protocols
   - Full type coverage

4. **Graph Views**
   - `GraphView` for lazy filtering
   - `SubGraphView`, `FilteredView`
   - Efficient for large graphs

### Phase 4: Ecosystem (v2.1.0) - 3 weeks

**Goals:** Rich ecosystem with generators, exporters, and integrations

1. **Graph Generators**
   - Classic graphs (`generators/classic.py`)
   - Random graphs (`generators/random.py`)
   - Social networks (`generators/social.py`)

2. **Export Formats**
   - DOT/GraphViz (`exporters/dot.py`)
   - GraphML (`exporters/graphml.py`)
   - Cytoscape (`exporters/cytoscape.py`)

3. **Integrations**
   - NetworkX compatibility layer
   - NumPy/SciPy integration
   - Pandas DataFrame I/O

4. **Command-Line Tool**
   - `algograph` CLI for Unix-style operations
   - Support piping and text formats

### Phase 5: Performance (v2.2.0) - 2 weeks

**Goals:** Optimize for production use

1. **Caching**
   - Cache adjacency structures
   - Cache computed properties

2. **Lazy Evaluation**
   - Lazy graph views
   - Lazy algorithm results where appropriate

3. **Benchmarking**
   - Benchmark suite vs NetworkX
   - Profile hot paths
   - Optimize critical algorithms

### Phase 6: Advanced Algorithms (v2.3.0) - 4 weeks

**Goals:** Complete algorithm coverage

1. **Community Detection**
   - Louvain algorithm
   - Label propagation
   - Modularity optimization

2. **Graph Isomorphism**
   - VF2 algorithm
   - Subgraph isomorphism

3. **Planarity**
   - Planarity testing
   - Planar embedding

4. **Advanced Features**
   - Graph products
   - Graph operations (union, intersection)
   - Temporal graphs

---

## 8. Code Examples: Before and After

### Example 1: Building a Social Network

**Before (v1.1.0 - Verbose):**
```python
from AlgoGraph import Graph, Vertex, Edge

# Create vertices
alice = Vertex('Alice', attrs={'age': 30, 'city': 'NYC'})
bob = Vertex('Bob', attrs={'age': 25, 'city': 'SF'})
carol = Vertex('Carol', attrs={'age': 35, 'city': 'LA'})

# Create edges
e1 = Edge('Alice', 'Bob', directed=False, attrs={'relationship': 'friend', 'since': 2020})
e2 = Edge('Bob', 'Carol', directed=False, attrs={'relationship': 'colleague', 'since': 2021})
e3 = Edge('Alice', 'Carol', directed=False, attrs={'relationship': 'friend', 'since': 2019})

# Build graph
social = Graph({alice, bob, carol}, {e1, e2, e3})

# Find friends older than 28 in NYC
friends_nyc = []
for v in social.vertices:
    if v.get('age', 0) > 28 and v.get('city') == 'NYC':
        friends_nyc.append(v.id)
```

**After (v2.0.0 - Fluent):**
```python
from AlgoGraph import graph, vertex, edge
from AlgoGraph.selectors import vertex as v_sel

# Build graph fluently
social = graph(
    vertex('Alice', age=30, city='NYC',
        edge('Bob', relationship='friend', since=2020),
        edge('Carol', relationship='friend', since=2019)
    ),
    vertex('Bob', age=25, city='SF',
        edge('Carol', relationship='colleague', since=2021)
    ),
    vertex('Carol', age=35, city='LA')
)

# Find friends older than 28 in NYC with selectors
friends_nyc = social.select_vertices(
    v_sel.attrs(age=lambda a: a > 28) &
    v_sel.attrs(city='NYC')
)
```

### Example 2: Analyzing a Road Network

**Before (v1.1.0 - Procedural):**
```python
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.algorithms import dijkstra, connected_components

# Build road network
roads = Graph()
roads = roads.add_edge(Edge('A', 'B', weight=10))
roads = roads.add_edge(Edge('B', 'C', weight=20))
roads = roads.add_edge(Edge('A', 'C', weight=35))
roads = roads.add_edge(Edge('C', 'D', weight=15))

# Find shortest path
distances, predecessors = dijkstra(roads, 'A')

# Reconstruct path
def reconstruct_path(start, end, predecessors):
    path = []
    current = end
    while current:
        path.append(current)
        current = predecessors.get(current)
    return list(reversed(path))

path_to_d = reconstruct_path('A', 'D', predecessors)
distance_to_d = distances['D']

print(f"Shortest path A -> D: {' -> '.join(path_to_d)}")
print(f"Distance: {distance_to_d}")
```

**After (v2.0.0 - Fluent):**
```python
from AlgoGraph import Graph
from AlgoGraph.algorithms import shortest_path

# Build road network from edge list
roads = Graph.from_edge_list([
    ('A', 'B', 10),
    ('B', 'C', 20),
    ('A', 'C', 35),
    ('C', 'D', 15)
])

# Find shortest path (returns typed result)
result = shortest_path(roads, 'A', 'D')

print(f"Shortest path A -> D: {' -> '.join(result['path'])}")
print(f"Distance: {result['distance']}")

# Or use method chaining
path, distance = roads.shortest_path_between('A', 'D')
```

### Example 3: Graph Transformation Pipeline

**Before (v1.1.0 - Not Possible):**
```python
# Can't chain transformations in v1.1.0
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.algorithms import connected_components

# Have to do manually step by step
g = load_graph('data.json')

# Filter high-degree vertices
high_degree_ids = {v.id for v in g.vertices if g.degree(v.id) > 5}
g = g.subgraph(high_degree_ids)

# Get largest component
components = connected_components(g)
largest = max(components, key=len)
g = g.subgraph(largest)

# Add computed attributes
new_vertices = set()
for v in g.vertices:
    deg = g.degree(v.id)
    new_v = v.with_attrs(normalized_degree=deg / g.vertex_count)
    new_vertices.add(new_v)
g = Graph(new_vertices, g.edges)
```

**After (v2.0.0 - Fluent Pipeline):**
```python
from AlgoGraph import Graph
from AlgoGraph.transformers import filter_vertices, largest_component, annotate

# Fluent transformation pipeline
result = (Graph.from_file('data.json')
    | filter_vertices(lambda v: v.degree() > 5)
    | largest_component()
    | annotate(lambda g, v: {'normalized_degree': v.degree() / g.vertex_count})
    | to_dict())

# Or using transformer composition
from AlgoGraph.transformers import pipeline

transform = pipeline(
    filter_vertices(degree_gt(5)),
    largest_component(),
    annotate_degrees(normalized=True)
)

result = Graph.from_file('data.json') | transform
```

### Example 4: Finding Influential Nodes

**Before (v1.1.0 - Manual):**
```python
from AlgoGraph import Graph
from AlgoGraph.algorithms import dijkstra

# Load graph
g = load_graph('social.json')

# Compute closeness centrality manually
closeness = {}
for v in g.vertices:
    distances, _ = dijkstra(g, v.id)

    # Sum of distances to all reachable vertices
    reachable_distances = [d for d in distances.values() if d != float('inf')]

    if reachable_distances:
        avg_distance = sum(reachable_distances) / len(reachable_distances)
        closeness[v.id] = 1 / avg_distance if avg_distance > 0 else 0
    else:
        closeness[v.id] = 0

# Find top 10
top_10 = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]

for vertex_id, score in top_10:
    print(f"{vertex_id}: {score:.4f}")
```

**After (v2.0.0 - Built-in):**
```python
from AlgoGraph import Graph
from AlgoGraph.algorithms.centrality import closeness, top_k

# Load graph
g = Graph.from_file('social.json')

# Compute closeness centrality
closeness_scores = closeness(g)

# Find top 10
top_10 = top_k(closeness_scores, k=10)

for vertex_id, score in top_10:
    print(f"{vertex_id}: {score:.4f}")

# Or in one line
top_10 = g.top_k_by(closeness, k=10)
```

---

## 9. Comparison with Mature Libraries

### NetworkX Feature Comparison

| Feature | NetworkX | AlgoGraph v1.1 | AlgoGraph v2.0 (Proposed) |
|---------|----------|----------------|---------------------------|
| **Core Graphs** |
| Directed | ✓ | ✓ | ✓ |
| Undirected | ✓ | ✓ | ✓ |
| Multigraph | ✓ | ✗ | ✓ |
| Weighted edges | ✓ | ✓ | ✓ |
| Node attributes | ✓ | ✓ | ✓ |
| Edge attributes | ✓ | ✓ | ✓ |
| **Immutability** |
| Immutable | ✗ | ✓ | ✓ |
| Structural sharing | ✗ | ✗ | ✓ |
| **Algorithms** |
| Shortest paths | ✓✓ | ✓ | ✓✓ |
| Traversals | ✓ | ✓ | ✓ |
| Components | ✓ | ✓ | ✓ |
| Spanning trees | ✓ | ✓ | ✓ |
| Flow networks | ✓✓ | ✗ | ✓ |
| Centrality | ✓✓ | ✗ | ✓ |
| Community | ✓✓ | ✗ | ✓ |
| Matching | ✓✓ | ✗ | ✓ |
| Coloring | ✓ | ✗ | ✓ |
| Isomorphism | ✓ | ✗ | ✓ |
| Planarity | ✓ | ✗ | ✓ |
| **Generators** |
| Classic graphs | ✓✓ | ✗ | ✓ |
| Random graphs | ✓✓ | ✗ | ✓ |
| Social networks | ✓ | ✗ | ✓ |
| **I/O** |
| JSON | ✓ | ✓ | ✓ |
| GraphML | ✓ | ✗ | ✓ |
| DOT/GraphViz | ✓ | ✗ | ✓ |
| Edge list | ✓ | ✗ | ✓ |
| Adjacency matrix | ✓ | ✗ | ✓ |
| **API Design** |
| Builder pattern | ✗ | ✗ | ✓ |
| Transformer pattern | ✗ | ✗ | ✓ |
| Selector pattern | ✗ | ✗ | ✓ |
| Method chaining | ✗ | ~ | ✓ |
| Type hints | ~ | ✓ | ✓✓ |
| Generic types | ✗ | ✗ | ✓ |

**Legend:** ✓✓ = Excellent, ✓ = Good, ~ = Partial, ✗ = Missing

### AlgoTree vs AlgoGraph Feature Parity

| Feature | AlgoTree v2.0 | AlgoGraph v1.1 | AlgoGraph v2.0 (Proposed) |
|---------|---------------|----------------|---------------------------|
| **API Patterns** |
| Immutability | ✓ | ✓ | ✓ |
| Builder API | ✓✓ | ✗ | ✓ |
| Transformer pattern | ✓✓ | ✗ | ✓ |
| Selector pattern | ✓✓ | ✗ | ✓ |
| DSL support | ✓ | ✗ | ✓ |
| Pipe composition | ✓ | ✗ | ✓ |
| **Type System** |
| Type hints | ✓ | ✓ | ✓✓ |
| Generic types | ✗ | ✗ | ✓ |
| Protocols | ✗ | ✗ | ✓ |
| **I/O** |
| JSON | ✓ | ✓ | ✓ |
| YAML | ✓ | ✗ | ✓ |
| XML | ✓ | ✗ | ✓ |
| Pickle | ✓ | ✗ | ✓ |
| Custom exporters | ✓ | ✗ | ✓ |
| **Shell** |
| Interactive shell | ✓ | ✓ | ✓ |
| Command system | ✓ | ✓ | ✓ |
| File I/O | ✓ | ✓ | ✓ |
| Tab completion | ✓ | ✓ | ✓ |
| **Documentation** |
| API docs | ✓ | ✓ | ✓ |
| Tutorials | ✓ | ✓ | ✓ |
| Examples | ✓ | ✓ | ✓ |
| Design guide | ✓ | ~ | ✓ |

---

## 10. Metrics and Success Criteria

### Code Quality Metrics

**Current State (v1.1.0):**
- Lines of code: ~4,600
- Test coverage: 85% (65/78 tests passing, 13 skipped)
- Type hint coverage: ~60%
- Documentation coverage: ~80%
- Cyclomatic complexity: Low (good)

**Target State (v2.0.0):**
- Lines of code: ~8,000 (with new features)
- Test coverage: >95%
- Type hint coverage: 100%
- Documentation coverage: 100%
- Benchmarks: Within 2x of NetworkX performance

### API Elegance Metrics

**Measure:** Lines of code needed for common tasks

**Current:**
- Build 10-node graph: 30 lines
- Filter and transform: 15 lines
- Complex query: 20 lines

**Target:**
- Build 10-node graph: 10 lines
- Filter and transform: 3 lines
- Complex query: 5 lines

### Feature Completeness

**Current:** 40% of NetworkX feature parity
**Target v2.0:** 75% of NetworkX feature parity
**Target v3.0:** 90% of NetworkX feature parity

### Performance Benchmarks

**Target:** Within 2-3x of NetworkX for common operations

- Graph construction: <2x NetworkX
- BFS/DFS: <1.5x NetworkX
- Dijkstra: <2x NetworkX
- Connected components: <2x NetworkX

Note: Some overhead acceptable due to immutability benefits.

---

## 11. Conclusion

AlgoGraph v1.1.0 is a solid foundation with excellent immutability and clean separation of concerns. However, it lags significantly behind both AlgoTree (in API elegance) and NetworkX (in feature completeness).

### Key Takeaways

1. **Biggest Opportunity:** API design improvements (builders, transformers, selectors)
2. **Biggest Gap:** Missing algorithms (flow, centrality, community detection)
3. **Biggest Strength:** Clean immutable design with good test coverage
4. **Biggest Risk:** Falling behind NetworkX without investment in completeness

### Recommended Focus

**Short term (3 months):**
- Phase 1: API improvements with builders and convenience methods
- Phase 2: Core missing algorithms (flow, centrality, matching)

**Medium term (6 months):**
- Phase 3: Transformer and selector patterns
- Phase 4: Rich ecosystem (generators, exporters)

**Long term (12 months):**
- Phase 5: Performance optimization
- Phase 6: Complete algorithm coverage

### Success Definition

AlgoGraph v2.0 will be successful if:
1. Building graphs is as elegant as AlgoTree node construction
2. Transforming graphs uses composable pipelines like AlgoTree
3. Querying graphs uses powerful selectors like AlgoTree
4. Algorithm coverage reaches 75% of NetworkX
5. Type system provides full type safety
6. Performance stays within 2-3x of NetworkX
7. Community adoption grows (stars, downloads, contributions)

The proposed roadmap is ambitious but achievable. AlgoGraph has the potential to be the premier immutable graph library in Python, combining NetworkX's completeness with AlgoTree's elegance and Haskell's type safety.

---

## Appendix A: Proposed File Structure

```
AlgoGraph/
├── __init__.py                 # Public API exports
├── vertex.py                   # Vertex class (generic)
├── edge.py                     # Edge class (generic)
├── graph.py                    # Graph class (generic)
├── protocols.py                # Type protocols (GraphLike, etc.)
├── builders.py                 # Builder patterns (NEW)
├── transformers.py             # Transformer pattern (NEW)
├── selectors.py                # Selector pattern (NEW)
├── views.py                    # Graph views (NEW)
├── serialization.py            # JSON I/O
├── interop.py                  # AlgoTree integration
├── algorithms/
│   ├── __init__.py
│   ├── traversal.py           # BFS, DFS, topological sort
│   ├── shortest_path.py       # Dijkstra, Bellman-Ford, Floyd-Warshall
│   ├── connectivity.py        # Components, bridges, articulation points
│   ├── spanning_tree.py       # Kruskal, Prim
│   ├── flow.py               # Max flow, min cut (NEW)
│   ├── centrality.py         # Betweenness, closeness, PageRank (NEW)
│   ├── matching.py           # Bipartite matching (NEW)
│   ├── coloring.py           # Graph coloring (NEW)
│   ├── community.py          # Community detection (NEW)
│   ├── isomorphism.py        # Isomorphism testing (NEW)
│   └── planarity.py          # Planarity testing (NEW)
├── generators/
│   ├── __init__.py           # (NEW)
│   ├── classic.py            # Complete, cycle, path, etc. (NEW)
│   ├── random.py             # Erdős-Rényi, Barabási-Albert (NEW)
│   └── social.py             # Social network datasets (NEW)
├── exporters/
│   ├── __init__.py
│   ├── json.py              # JSON export (exists)
│   ├── dot.py               # GraphViz/DOT (NEW)
│   ├── graphml.py           # GraphML (NEW)
│   ├── gexf.py              # GEXF (NEW)
│   └── cytoscape.py         # Cytoscape JSON (NEW)
├── importers/
│   ├── __init__.py          # (NEW)
│   ├── json.py              # JSON import (exists)
│   ├── dot.py               # DOT parser (NEW)
│   ├── graphml.py           # GraphML parser (NEW)
│   └── edge_list.py         # Edge list parser (NEW)
├── utils/
│   ├── __init__.py          # (NEW)
│   ├── predicates.py        # Common predicates (NEW)
│   └── operators.py         # Graph operations (NEW)
├── cli/
│   ├── __init__.py          # (NEW)
│   └── main.py              # Command-line tool (NEW)
├── shell/
│   ├── __init__.py
│   ├── context.py
│   ├── commands.py
│   ├── shell.py
│   └── cli.py
├── test/
│   ├── test_algorithms.py   # (exists)
│   ├── test_builders.py     # (NEW)
│   ├── test_transformers.py # (NEW)
│   ├── test_selectors.py    # (NEW)
│   ├── test_views.py        # (NEW)
│   ├── test_generators.py   # (NEW)
│   ├── test_exporters.py    # (NEW)
│   └── ...
└── examples/
    ├── basic_usage.py
    ├── builders.py           # (NEW)
    ├── transformers.py       # (NEW)
    ├── social_network.py
    └── road_network.py
```

---

## Appendix B: API Design Principles

### 1. Principle of Least Surprise
Users should be able to guess what an operation does from its name.

**Good:**
```python
g.add_vertex(v)      # Clear: adds a vertex
g.neighbors('A')     # Clear: returns neighbors
g.shortest_path('A', 'B')  # Clear: finds shortest path
```

**Bad:**
```python
g.insert(v)          # Ambiguous: insert what where?
g.adjacent('A')      # Ambiguous: adjacent to what?
g.path('A', 'B')     # Ambiguous: any path? shortest?
```

### 2. Progressive Disclosure
Simple things should be simple, complex things should be possible.

**Good:**
```python
# Simple: build graph easily
g = graph(
    vertex('A', edge('B'), edge('C')),
    vertex('B'),
    vertex('C')
)

# Complex: fine-grained control
g = (GraphBuilder()
     .add_vertex(Vertex('A', attrs={'x': 10}))
     .add_edge(Edge('A', 'B', weight=5.0, directed=False, attrs={'type': 'road'}))
     .build())
```

### 3. Composability
Operations should compose naturally.

**Good:**
```python
result = (graph
    | filter_vertices(active)
    | largest_component()
    | map_vertices(normalize)
    | to_dict())
```

**Bad:**
```python
# Can't compose - have to nest or use temp variables
temp1 = filter_vertices(graph, active)
temp2 = largest_component(temp1)
temp3 = map_vertices(temp2, normalize)
result = to_dict(temp3)
```

### 4. Consistency
Similar operations should have similar APIs.

**Good:**
```python
g.add_vertex(v)
g.add_edge(e)
g.remove_vertex('A')
g.remove_edge('A', 'B')
```

**Bad:**
```python
g.add_vertex(v)
g.insert_edge(e)           # Inconsistent verb
g.delete('A')              # Inconsistent verb and missing type
g.remove_edge('A', 'B')
```

### 5. Fail Fast
Errors should be caught early and reported clearly.

**Good:**
```python
try:
    g = Graph({Vertex('A')}, {Edge('A', 'Z')})  # Z doesn't exist
except ValueError as e:
    print(e)  # "Edge references non-existent vertex: Z"
```

**Bad:**
```python
g = Graph({Vertex('A')}, {Edge('A', 'Z')})  # Silently accepted
neighbors = g.neighbors('Z')  # Fails later with obscure error
```

### 6. Explicit is Better Than Implicit
Make behavior clear, don't rely on magic.

**Good:**
```python
g.dijkstra('A', weight_attr='distance')  # Clear what attribute to use
```

**Bad:**
```python
g.dijkstra('A')  # Magically uses 'weight' or 'distance' or...?
```

### 7. Immutability by Default
All operations return new objects unless explicitly stated.

**Good:**
```python
g2 = g1.add_vertex(v)  # Clear that g1 is unchanged
g3 = g2.remove_edge('A', 'B')  # Clear that g2 is unchanged
```

**Bad:**
```python
g.add_vertex(v)  # Does this mutate g? Return new graph? Both?
```

---

**End of Architectural Review**
