# Graph Algorithms

AlgoGraph provides 30+ graph algorithms organized into four categories. All algorithms work with immutable graphs and return new data structures without modifying the original graph.

## Algorithm Categories

### Traversal Algorithms
- DFS (Depth-First Search)
- BFS (Breadth-First Search)
- Topological Sort
- Cycle Detection
- Path Finding

### Shortest Path Algorithms
- Dijkstra's Algorithm
- Bellman-Ford Algorithm
- Floyd-Warshall Algorithm
- A* Search

### Connectivity Algorithms
- Connected Components
- Strongly Connected Components
- Bipartite Checking
- Bridge Finding
- Articulation Points
- Diameter Calculation

### Spanning Tree Algorithms
- Kruskal's Algorithm
- Prim's Algorithm
- Minimum Spanning Tree

## Importing Algorithms

All algorithms are in the `AlgoGraph.algorithms` module:

```python
# Import specific algorithms
from AlgoGraph.algorithms import dfs, bfs, dijkstra

# Import from submodules
from AlgoGraph.algorithms.traversal import topological_sort
from AlgoGraph.algorithms.shortest_path import bellman_ford
from AlgoGraph.algorithms.connectivity import connected_components
from AlgoGraph.algorithms.spanning_tree import kruskal

# Import all from a category
from AlgoGraph.algorithms.traversal import *
```

## Traversal Algorithms

### Depth-First Search (DFS)

Explore as far as possible along each branch before backtracking:

```python
from AlgoGraph import Graph, Edge, Vertex
from AlgoGraph.algorithms import dfs

g = Graph(
    vertices={Vertex('A'), Vertex('B'), Vertex('C'), Vertex('D')},
    edges={Edge('A', 'B'), Edge('A', 'C'), Edge('B', 'D')}
)

# Get DFS traversal order
order = dfs(g, 'A')
print(order)  # ['A', 'B', 'D', 'C'] (exact order may vary)

# DFS with callback
def visit(vertex_id):
    print(f"Visiting {vertex_id}")

dfs(g, 'A', visit=visit)
```

**Variants:**
- `dfs_recursive`: Recursive implementation
- `dfs_iterative`: Iterative implementation using explicit stack

### Breadth-First Search (BFS)

Explore all neighbors before moving to the next level:

```python
from AlgoGraph.algorithms import bfs, bfs_levels

# Get BFS traversal order
order = bfs(g, 'A')
print(order)  # ['A', 'B', 'C', 'D']

# Get BFS by levels
levels = bfs_levels(g, 'A')
print(levels)  # [['A'], ['B', 'C'], ['D']]
```

### Topological Sort

Order vertices in a DAG so all edges go from earlier to later vertices:

```python
from AlgoGraph.algorithms import topological_sort, has_cycle

# Check if graph is a DAG
if not has_cycle(g):
    order = topological_sort(g)
    print(f"Topological order: {order}")
else:
    print("Graph has cycles - cannot topologically sort")
```

**Use cases:** Task scheduling, build systems, course prerequisites

### Path Finding

Find paths between vertices:

```python
from AlgoGraph.algorithms import find_path, find_all_paths

# Find any path
path = find_path(g, 'A', 'D')
print(path)  # ['A', 'B', 'D']

# Find all paths (can be expensive!)
all_paths = find_all_paths(g, 'A', 'D')
for path in all_paths:
    print(' -> '.join(path))
```

### Cycle Detection

Check if a graph contains cycles:

```python
from AlgoGraph.algorithms import has_cycle

if has_cycle(g):
    print("Graph contains at least one cycle")
else:
    print("Graph is acyclic (DAG)")
```

## Shortest Path Algorithms

### Dijkstra's Algorithm

Find shortest paths from a source vertex (non-negative weights only):

```python
from AlgoGraph.algorithms import dijkstra, shortest_path, shortest_path_length

# Get distances and predecessors
distances, predecessors = dijkstra(g, 'A')

print(f"Distance from A to D: {distances['D']}")
print(f"Predecessor of D: {predecessors['D']}")

# Get the actual shortest path
path = shortest_path(g, 'A', 'D')
print(f"Shortest path: {' -> '.join(path)}")

# Just get the distance
distance = shortest_path_length(g, 'A', 'D')
print(f"Shortest distance: {distance}")
```

**Time Complexity:** O((V + E) log V) with binary heap

**Use cases:** Road networks, network routing, map applications

### Bellman-Ford Algorithm

Find shortest paths allowing negative edge weights (detects negative cycles):

```python
from AlgoGraph.algorithms import bellman_ford

# Can handle negative weights
distances, predecessors = bellman_ford(g, 'A')

if distances is None:
    print("Negative cycle detected!")
else:
    print(f"Distances: {distances}")
```

**Time Complexity:** O(VE)

**Use cases:** Currency arbitrage, graphs with negative weights

### Floyd-Warshall Algorithm

Find shortest paths between all pairs of vertices:

```python
from AlgoGraph.algorithms import floyd_warshall, all_shortest_paths

# Get distance matrix
distances = floyd_warshall(g)

# Distance from any vertex to any other
print(f"Distance from A to D: {distances['A']['D']}")
print(f"Distance from B to C: {distances['B']['C']}")

# Get all shortest paths from A to D
paths = all_shortest_paths(g, 'A', 'D')
for path in paths:
    print(' -> '.join(path))
```

**Time Complexity:** O(V³)

**Use cases:** All-pairs distances, transitive closure

### A* Search

Heuristic-based shortest path algorithm:

```python
from AlgoGraph.algorithms import a_star

def heuristic(v1, v2):
    """Estimate distance between vertices."""
    # Example: Euclidean distance for points
    p1 = graph.get_vertex(v1).get('coords', (0, 0))
    p2 = graph.get_vertex(v2).get('coords', (0, 0))
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

path = a_star(g, 'A', 'D', heuristic)
print(f"A* path: {' -> '.join(path)}")
```

**Use cases:** Pathfinding in games, GPS navigation

## Connectivity Algorithms

### Connected Components

Find all connected components in an undirected graph:

```python
from AlgoGraph.algorithms import connected_components, is_connected

# Get all components
components = connected_components(g)
print(f"Number of components: {len(components)}")

for i, component in enumerate(components, 1):
    print(f"Component {i}: {component}")

# Check if graph is fully connected
if is_connected(g):
    print("Graph is connected")
else:
    print("Graph has multiple components")
```

**Use cases:** Network analysis, clustering, social networks

### Strongly Connected Components

Find strongly connected components in directed graphs:

```python
from AlgoGraph.algorithms import strongly_connected_components, is_strongly_connected

# Get SCCs
sccs = strongly_connected_components(g)
print(f"Number of SCCs: {len(sccs)}")

# Check if strongly connected
if is_strongly_connected(g):
    print("Graph is strongly connected")
```

**Use cases:** Web page ranking, dependency analysis

### Bipartite Checking

Check if a graph is bipartite (2-colorable):

```python
from AlgoGraph.algorithms import is_bipartite

bipartite, partition = is_bipartite(g)

if bipartite:
    set1, set2 = partition
    print(f"Graph is bipartite!")
    print(f"Set 1: {set1}")
    print(f"Set 2: {set2}")
else:
    print("Graph is not bipartite")
```

**Use cases:** Matching problems, scheduling, assignment problems

### Bridges and Articulation Points

Find critical edges and vertices:

```python
from AlgoGraph.algorithms import find_bridges, find_articulation_points

# Find bridges (edges whose removal disconnects the graph)
bridges = find_bridges(g)
print(f"Bridges: {bridges}")

# Find articulation points (vertices whose removal disconnects the graph)
articulation_points = find_articulation_points(g)
print(f"Articulation points: {articulation_points}")
```

**Use cases:** Network reliability, vulnerability analysis

### Tree and Diameter

Check if graph is a tree and find its diameter:

```python
from AlgoGraph.algorithms import is_tree, diameter

# Check if graph is a tree
if is_tree(g):
    print("Graph is a tree")

    # Find diameter (longest shortest path)
    d = diameter(g)
    print(f"Diameter: {d}")
```

## Spanning Tree Algorithms

### Kruskal's Algorithm

Find minimum spanning tree using edge-based approach:

```python
from AlgoGraph.algorithms import kruskal, total_weight

# Get MST
mst = kruskal(g)
print(f"MST has {mst.edge_count} edges")
print(f"Total weight: {total_weight(mst)}")

# Print MST edges
for edge in mst.edges:
    print(f"{edge.source} - {edge.target}: {edge.weight}")
```

**Time Complexity:** O(E log E)

### Prim's Algorithm

Find minimum spanning tree using vertex-based approach:

```python
from AlgoGraph.algorithms import prim

# Get MST starting from vertex 'A'
mst = prim(g, 'A')
print(f"Total weight: {total_weight(mst)}")
```

**Time Complexity:** O(E log V)

### General MST Functions

```python
from AlgoGraph.algorithms import minimum_spanning_tree, is_spanning_tree

# Get MST (uses best algorithm automatically)
mst = minimum_spanning_tree(g)

# Check if a graph is a spanning tree of another
if is_spanning_tree(mst, g):
    print("Valid spanning tree")
```

## Algorithm Complexity Reference

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| DFS | O(V + E) | O(V) | Recursive uses call stack |
| BFS | O(V + E) | O(V) | Uses queue |
| Topological Sort | O(V + E) | O(V) | Only for DAGs |
| Dijkstra | O((V + E) log V) | O(V) | Non-negative weights |
| Bellman-Ford | O(VE) | O(V) | Handles negative weights |
| Floyd-Warshall | O(V³) | O(V²) | All-pairs shortest paths |
| A* | O(E) | O(V) | Depends on heuristic |
| Connected Components | O(V + E) | O(V) | Undirected graphs |
| SCC (Tarjan's) | O(V + E) | O(V) | Directed graphs |
| Bipartite Check | O(V + E) | O(V) | Uses BFS/DFS |
| Bridges | O(V + E) | O(V) | Uses DFS |
| Articulation Points | O(V + E) | O(V) | Uses DFS |
| Kruskal's MST | O(E log E) | O(V) | Union-find |
| Prim's MST | O(E log V) | O(V) | Priority queue |

## Choosing the Right Algorithm

### For Finding Paths

- **Any path:** Use `find_path` (DFS-based)
- **Shortest path (unweighted):** Use `bfs`
- **Shortest path (weighted, non-negative):** Use `dijkstra`
- **Shortest path (with negative weights):** Use `bellman_ford`
- **All-pairs shortest paths:** Use `floyd_warshall`
- **Heuristic search:** Use `a_star`

### For Graph Structure Analysis

- **Check connectivity:** Use `is_connected` or `connected_components`
- **Directed graph connectivity:** Use `strongly_connected_components`
- **Find critical points:** Use `find_bridges` and `find_articulation_points`
- **Check if tree:** Use `is_tree`
- **Check 2-colorability:** Use `is_bipartite`

### For Traversal

- **Explore depth-first:** Use `dfs`
- **Explore breadth-first:** Use `bfs`
- **Level-by-level traversal:** Use `bfs_levels`
- **Topological ordering:** Use `topological_sort`

### For Spanning Trees

- **Minimum spanning tree:** Use `kruskal` or `prim`
- **Just need total weight:** Use `total_weight`

## Performance Tips

1. **Choose the right algorithm:** Don't use Floyd-Warshall for single-source shortest paths
2. **Pre-filter graphs:** Remove unnecessary vertices/edges before running algorithms
3. **Cache results:** Many algorithms can be expensive - cache results when possible
4. **Use appropriate data structures:** BFS is faster than Dijkstra for unweighted graphs
5. **Check preconditions:** Ensure graph meets algorithm requirements (e.g., no cycles for topological sort)

## Next Steps

- See [Examples](../examples/social-networks.md) for real-world use cases
- Check the [API Reference](../api/algorithms/traversal.md) for detailed algorithm documentation
- Try algorithms in the [Interactive Shell](../shell/queries.md)
