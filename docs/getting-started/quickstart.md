# Quick Start

This guide will get you up and running with AlgoGraph in just a few minutes.

## Your First Graph

Let's create a simple social network graph:

```python
from AlgoGraph import Vertex, Edge, Graph

# Create vertices representing people
alice = Vertex('Alice', attrs={'age': 30, 'city': 'NYC'})
bob = Vertex('Bob', attrs={'age': 25, 'city': 'Boston'})
charlie = Vertex('Charlie', attrs={'age': 35, 'city': 'NYC'})

# Create edges representing friendships (undirected)
friendship1 = Edge('Alice', 'Bob', directed=False)
friendship2 = Edge('Bob', 'Charlie', directed=False)

# Build the graph
graph = Graph(
    vertices={alice, bob, charlie},
    edges={friendship1, friendship2}
)

print(graph)  # Graph with 3 vertices and 2 edges
```

## Basic Graph Operations

### Checking Vertex and Edge Existence

```python
# Check if a vertex exists
print(graph.has_vertex('Alice'))  # True
print(graph.has_vertex('Diana'))  # False

# Check if an edge exists
print(graph.has_edge('Alice', 'Bob'))  # True
print(graph.has_edge('Alice', 'Charlie'))  # False
```

### Getting Vertex Information

```python
# Get a specific vertex
alice = graph.get_vertex('Alice')
print(alice.id)  # 'Alice'
print(alice.get('age'))  # 30
print(alice.get('city'))  # 'NYC'

# Get all neighbors of a vertex
neighbors = graph.neighbors('Bob')
print(neighbors)  # {'Alice', 'Charlie'}

# Get vertex degree
print(graph.degree('Bob'))  # 2
```

## Modifying Graphs (Immutably)

AlgoGraph uses immutable data structures. When you "modify" a graph, you get a new graph back:

```python
# Add a new vertex
diana = Vertex('Diana', attrs={'age': 28, 'city': 'Seattle'})
graph2 = graph.add_vertex(diana)

print(graph.vertex_count)   # 3 (original unchanged)
print(graph2.vertex_count)  # 4 (new graph)

# Add a new edge
edge = Edge('Charlie', 'Diana', directed=False)
graph3 = graph2.add_edge(edge)

# Chain operations
graph4 = (graph
    .add_vertex(Vertex('Eve'))
    .add_edge(Edge('Diana', 'Eve', directed=False))
)
```

## Running Algorithms

AlgoGraph provides 30+ graph algorithms in separate modules:

### Path Finding

```python
from AlgoGraph.algorithms import find_path, shortest_path

# Find any path between two vertices
path = find_path(graph, 'Alice', 'Charlie')
print(path)  # ['Alice', 'Bob', 'Charlie']

# Find shortest path (considers edge weights)
shortest = shortest_path(graph, 'Alice', 'Charlie')
print(shortest)  # ['Alice', 'Bob', 'Charlie']
```

### Breadth-First Search

```python
from AlgoGraph.algorithms import bfs

# Perform BFS from a starting vertex
order = bfs(graph, 'Alice')
print(order)  # ['Alice', 'Bob', 'Charlie'] (visit order)
```

### Connected Components

```python
from AlgoGraph.algorithms import connected_components, is_connected

# Find all connected components
components = connected_components(graph)
print(components)  # [{'Alice', 'Bob', 'Charlie'}]

# Check if graph is connected
print(is_connected(graph))  # True
```

## Working with Weighted Graphs

Create a road network with distances:

```python
# Create a weighted directed graph
cities = {
    Vertex('Boston'),
    Vertex('NYC'),
    Vertex('Philadelphia'),
    Vertex('Washington DC')
}

roads = {
    Edge('Boston', 'NYC', weight=215),      # miles
    Edge('NYC', 'Philadelphia', weight=95),
    Edge('Philadelphia', 'Washington DC', weight=140),
    Edge('Boston', 'Philadelphia', weight=310),
}

road_network = Graph(cities, roads)

# Find shortest path using Dijkstra's algorithm
from AlgoGraph.algorithms import dijkstra

distances, predecessors = dijkstra(road_network, 'Boston')

print(f"Boston to NYC: {distances['NYC']} miles")
print(f"Boston to Washington DC: {distances['Washington DC']} miles")
```

## Saving and Loading Graphs

```python
from AlgoGraph import save_graph, load_graph

# Save to JSON
save_graph(graph, 'my_social_network.json')

# Load from JSON
loaded_graph = load_graph('my_social_network.json')

print(loaded_graph.vertex_count)  # Same as original
```

## Interactive Exploration

Launch the interactive shell to explore graphs:

```bash
python3 -m AlgoGraph.shell.shell my_social_network.json
```

In the shell, you can navigate the graph like a file system:

```
graph(3v):/$ ls
Alice/  [2 neighbors]
Bob/  [2 neighbors]
Charlie/  [1 neighbors]

graph(3v):/$ cd Alice

graph(3v):/Alice$ ls
Attributes:
  age = 30
  city = NYC

neighbors/  [2 vertices]

graph(3v):/Alice$ cd neighbors

graph(3v):/Alice/neighbors$ ls
Bob/  <-> [weight: 1.0]
Charlie/  <-> [weight: 1.0]

graph(3v):/Alice/neighbors$ cd Bob

graph(3v):/Bob$ info
Vertex: Bob
Degree: 2

Attributes:
  age = 25
  city = Boston
```

## Common Patterns

### Creating Graphs from Data

```python
# From a list of edges
edge_list = [
    ('A', 'B', 5),   # (source, target, weight)
    ('B', 'C', 3),
    ('C', 'D', 7),
]

vertices = {Vertex(v) for edge in edge_list for v in edge[:2]}
edges = {Edge(src, tgt, weight=w) for src, tgt, w in edge_list}
graph = Graph(vertices, edges)

# From an adjacency dictionary
adj_dict = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['D'],
    'D': []
}

vertices = {Vertex(v) for v in adj_dict.keys()}
edges = {
    Edge(src, tgt)
    for src, targets in adj_dict.items()
    for tgt in targets
}
graph = Graph(vertices, edges)
```

### Filtering Vertices and Edges

```python
# Find vertices with specific attributes
nyc_residents = graph.find_vertices(
    lambda v: v.get('city') == 'NYC'
)

# Find edges with weight > threshold
heavy_edges = graph.find_edges(
    lambda e: e.weight > 100
)

# Create subgraph
subgraph = graph.subgraph({'Alice', 'Bob'})
```

### Updating Attributes

```python
# Update a vertex's attributes (returns new graph)
alice = graph.get_vertex('Alice')
updated_alice = alice.with_attrs(age=31, job='Engineer')
graph2 = graph.update_vertex(updated_alice)

# Update an edge's weight
edge = graph.get_edge('Alice', 'Bob')
updated_edge = edge.with_weight(2.5)
graph3 = graph.update_edge(updated_edge)
```

## What's Next?

Now that you understand the basics, you can:

- Learn about [Core Concepts](../user-guide/core-concepts.md) in depth
- Explore all available [Graph Algorithms](../user-guide/algorithms.md)
- See more [Examples](examples.md)
- Browse the [API Reference](../api/vertex.md)
- Try the [Interactive Shell](../shell/overview.md)
