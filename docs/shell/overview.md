# Interactive Shell Overview

The AlgoGraph interactive shell provides a VFS-like interface for exploring graphs. Navigate through graphs as if they were a file system, run queries, and visualize graph structure interactively.

## Starting the Shell

### With a Graph File

```bash
python3 -m AlgoGraph.shell.shell path/to/graph.json
```

### With Sample Data

```bash
python3 -m AlgoGraph.shell.shell
```

This loads a sample social network for exploration.

### Programmatically

```python
from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.shell import GraphShell, GraphContext

# Create your graph
graph = Graph(
    vertices={Vertex('A'), Vertex('B'), Vertex('C')},
    edges={Edge('A', 'B'), Edge('B', 'C')}
)

# Start shell
context = GraphContext(graph)
shell = GraphShell(context)
shell.run()
```

## Shell Basics

### The Prompt

The shell prompt shows the current location and graph size:

```
graph(5v):/$
```

- `graph`: You're in a graph shell
- `(5v)`: Graph has 5 vertices
- `/`: Current location (root in this case)
- `$`: Prompt character

### Navigation Hierarchy

The shell has three levels:

1. **Root (`/`)**: View all vertices
2. **Vertex (`/Alice`)**: View a specific vertex and its attributes
3. **Neighbors (`/Alice/neighbors`)**: View a vertex's neighbors

```
/                    # Root - all vertices
├── Alice/           # Vertex level
│   └── neighbors/   # Neighbors of Alice
├── Bob/
│   └── neighbors/
└── Charlie/
    └── neighbors/
```

## Basic Commands

### pwd - Print Working Directory

Shows your current location:

```bash
graph(5v):/$ pwd
/

graph(5v):/Alice$ pwd
/Alice
```

### ls - List Contents

List what's at the current location:

```bash
# At root - shows all vertices
graph(5v):/$ ls
Alice/  [2 neighbors]
Bob/  [3 neighbors]
Charlie/  [1 neighbors]

# At vertex - shows attributes and neighbors/
graph(5v):/Alice$ ls
Attributes:
  age = 30
  city = NYC

neighbors/  [2 vertices]

# In neighbors mode - shows neighbors
graph(5v):/Alice/neighbors$ ls
Bob/  <-> [weight: 1.0]
Charlie/  <-> [weight: 1.0]
```

### cd - Change Directory

Navigate to different locations:

```bash
# Go to a vertex
graph(5v):/$ cd Alice
Now at: /Alice

# Enter neighbors mode
graph(5v):/Alice$ cd neighbors
Now at: /Alice/neighbors

# Navigate to a neighbor
graph(5v):/Alice/neighbors$ cd Bob
Now at: /Bob

# Go up one level
graph(5v):/Bob$ cd ..
Now at: /

# Go to root
graph(5v):/Alice/neighbors$ cd /
Now at: /

# Use absolute paths
graph(5v):/$ cd /Alice
Now at: /Alice
```

## Information Commands

### info - Show Information

Display details about current location:

```bash
# At root - shows graph statistics
graph(5v):/$ info
Graph Information:
  Vertices: 5
  Edges: 7
  Type: Undirected

# At vertex - shows vertex details
graph(5v):/Alice$ info
Vertex: Alice
Degree: 2

Attributes:
  age = 30
  city = NYC
```

### neighbors - Show Neighbors

Quick view of a vertex's neighbors (alternative to `cd neighbors` + `ls`):

```bash
graph(5v):/Alice$ neighbors
Neighbors of Alice:
  Bob <-> (weight: 1.0)
  Charlie <-> (weight: 1.0)
```

### find - Find a Vertex

Search for a vertex anywhere in the graph:

```bash
graph(5v):/$ find Charlie
Found: Charlie
Degree: 1
Attributes:
  age = 35
  city = NYC
```

## Query Commands

### path - Find Any Path

Find a path between two vertices:

```bash
graph(5v):/$ path Alice Diana
Path found: Alice -> Bob -> Diana
Length: 2 edges
```

### shortest - Find Shortest Path

Find the shortest weighted path:

```bash
graph(5v):/$ shortest Alice Diana
Shortest path: Alice -> Bob -> Diana
Distance: 2.0
```

### components - Show Connected Components

Analyze graph connectivity:

```bash
graph(5v):/$ components
Connected components: 1

Component 1 (5 vertices):
  Alice, Bob, Charlie, Diana, Eve
```

### bfs - Breadth-First Search

Run BFS from a vertex:

```bash
graph(5v):/Alice$ bfs
BFS from Alice:
Alice -> Bob -> Charlie -> Diana -> Eve

Visited 5 vertices

# Or specify starting vertex
graph(5v):/$ bfs Bob
BFS from Bob:
Bob -> Alice -> Charlie -> Diana -> Eve

Visited 5 vertices
```

## File Operations

### save - Save Graph

Save the current graph to a JSON file:

```bash
graph(5v):/$ save my_graph.json
Graph saved to my_graph.json
```

## Shell Features

### Tab Completion

Press TAB to autocomplete:

```bash
graph(5v):/$ cd Al<TAB>
graph(5v):/$ cd Alice
```

Works for:
- Command names
- Vertex names
- Special keywords (`neighbors`, `..`, `/`)

### Command History

Use UP/DOWN arrow keys to navigate command history:

```bash
# Press UP to recall previous commands
# Press DOWN to go forward in history
```

### Quoted Names

Use quotes for vertex names with spaces or special characters:

```bash
graph(5v):/$ cd "Alice Smith"
Now at: /Alice Smith

graph(5v):/Alice Smith$ cd ..
Now at: /
```

### Absolute vs. Relative Paths

```bash
# Relative path (from current location)
graph(5v):/$ cd Alice
graph(5v):/Alice$ cd neighbors

# Absolute path (from root)
graph(5v):/Alice/neighbors$ cd /Bob
Now at: /Bob
```

## Example Session

Here's a complete example session exploring a social network:

```bash
$ python3 -m AlgoGraph.shell.shell

AlgoGraph Shell
Type 'help' for available commands, 'exit' to quit

Graph loaded: 5 vertices, 4 edges

graph(5v):/$ ls
Alice/  [2 neighbors]
Bob/  [2 neighbors]
Charlie/  [1 neighbors]
Diana/  [2 neighbors]
Eve/  [1 neighbors]

graph(5v):/$ cd Alice

graph(5v):/Alice$ info
Vertex: Alice
Degree: 2

Attributes:
  age = 30
  city = NYC

graph(5v):/Alice$ neighbors
Neighbors of Alice:
  Bob <-> (weight: 1.0)
  Charlie <-> (weight: 1.0)

graph(5v):/Alice$ cd neighbors

graph(5v):/Alice/neighbors$ ls
Bob/  <-> [weight: 1.0]
Charlie/  <-> [weight: 1.0]

graph(5v):/Alice/neighbors$ cd Bob

graph(5v):/Bob$ info
Vertex: Bob
Degree: 2

Attributes:
  age = 25
  city = Boston

graph(5v):/Bob$ cd /

graph(5v):/$ path Alice Eve
Path found: Alice -> Bob -> Diana -> Eve
Length: 3 edges

graph(5v):/$ shortest Alice Eve
Shortest path: Alice -> Bob -> Diana -> Eve
Distance: 3.0

graph(5v):/$ components
Connected components: 1

Component 1 (5 vertices):
  Alice, Bob, Charlie, Diana, Eve

graph(5v):/$ save social_network.json
Graph saved to social_network.json

graph(5v):/$ exit
Goodbye!
```

## Getting Help

### help Command

Show available commands:

```bash
graph(5v):/$ help

AlgoGraph Shell Commands:

Navigation:
  cd <vertex>     - Navigate to a vertex (relative)
  cd /vertex      - Navigate to a vertex (absolute path)
  cd neighbors    - View neighbors of current vertex
  cd ..           - Go up one level
  cd / or cd      - Go to root
  ls              - List contents of current location
  pwd             - Print current path

Information:
  info            - Show info about current vertex or graph
  neighbors       - Show neighbors of current vertex
  help            - Show this help message

Graph Queries:
  find <vertex>   - Find vertex in graph
  path <v1> <v2>  - Find path between vertices
  shortest <v1> <v2> - Find shortest path (weighted)
  components      - Show connected components
  bfs [start]     - Breadth-first search from vertex

File Operations:
  save <file>     - Save graph to JSON file

Other:
  exit, quit      - Exit the shell

Tips:
  - Use quotes for vertex names with spaces: cd "Alice Smith"
  - Press TAB for command/vertex name completion
  - Use UP/DOWN arrows for command history
```

## Exiting the Shell

Multiple ways to exit:

```bash
graph(5v):/$ exit
Goodbye!

# Or use quit
graph(5v):/$ quit
Goodbye!

# Or just q
graph(5v):/$ q
Goodbye!

# Or Ctrl+D
graph(5v):/$ <Ctrl+D>
Goodbye!
```

## Next Steps

- Learn about [Navigation Commands](navigation.md) in detail
- Explore [Query Commands](queries.md) for graph analysis
- Check out [Advanced Features](advanced.md) like tab completion and scripting
