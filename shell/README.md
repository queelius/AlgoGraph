# AlgoGraph Shell

Interactive shell for exploring graphs with a filesystem-like interface.

## Overview

The AlgoGraph shell provides a VFS (Virtual File System) interface for navigating and exploring graphs. You can "cd" into vertices, "ls" their attributes and neighbors, and navigate the graph structure as if it were a filesystem.

This design is inspired by the observation that filesystems are graphs (due to symbolic links, hard links, etc.), so treating graphs as filesystems is a natural fit.

## Quick Start

```bash
# Start interactive shell
cd /home/spinoza/github/released
export PYTHONPATH=.
python -m AlgoGraph.shell.shell

# You'll see a prompt like:
# graph(5v):/$
```

## Navigation Model

The shell treats graphs like a filesystem:

- **Root (`/`)**: Shows all vertices in the graph
- **Vertex (`/A`)**: Shows vertex attributes and a `neighbors/` pseudo-directory
- **Neighbors mode (`/A/neighbors`)**: Shows neighboring vertices you can navigate to

### Special Locations

- `/` - Graph root (lists all vertices)
- `/vertex_id` - At a specific vertex
- `/vertex_id/neighbors` - Viewing neighbors of a vertex

## Core Commands

### Navigation

```bash
# Go to root
cd /
cd

# Navigate to a vertex
cd Alice

# View neighbors
cd neighbors

# Go up one level
cd ..

# Navigate from neighbors to a neighbor
cd neighbors
cd Bob  # Navigate to Bob (must be a neighbor)
```

### Listing Contents

```bash
# At root - shows all vertices
ls
# Output:
# Alice/  [2 neighbors]
# Bob/    [1 neighbors]
# Charlie/  [2 neighbors]

# At a vertex - shows attributes and neighbors/
cd Alice
ls
# Output:
# Attributes:
#   age = 30
#   city = NYC
#
# neighbors/  [2 vertices]

# In neighbors mode - shows neighboring vertices
cd neighbors
ls
# Output:
# Bob/  <-> [weight: 1.0]
# Charlie/  <->
```

### Information

```bash
# Print working directory
pwd
# Output: /Alice

# Show info about current location
info
# At root, shows graph statistics
# At vertex, shows vertex details

# Show neighbors (alternative to cd neighbors + ls)
neighbors
```

## Graph Query Commands

### Finding Vertices

```bash
find Alice
# Output:
# Found: Alice
# Degree: 2
# Attributes:
#   age = 30
#   city = NYC
```

### Path Finding

```bash
# Find any path
path Alice Charlie
# Output:
# Path found: Alice -> Bob -> Charlie
# Length: 2 edges

# Find shortest weighted path
shortest Alice Charlie
# Output:
# Shortest path: Alice -> Bob -> Charlie
# Distance: 5.0
```

### Connectivity Analysis

```bash
# Show connected components
components
# Output:
# Connected components: 1
#
# Component 1 (5 vertices):
#   Alice, Bob, Charlie, Diana, Eve

# Breadth-first search
cd Alice
bfs
# Output:
# BFS from Alice:
# Alice -> Bob -> Charlie -> Diana -> Eve
#
# Visited 5 vertices

# Or specify start vertex
bfs Alice
```

## Example Session

```bash
graph(5v):/$ ls
Alice/  [2 neighbors]
Bob/  [3 neighbors]
Charlie/  [2 neighbors]
Diana/  [2 neighbors]
Eve/  [1 neighbors]

graph(5v):/$ cd Alice
Now at: /Alice

graph(5v):/Alice$ ls
Attributes:
  age = 30
  city = NYC

neighbors/  [2 vertices]

graph(5v):/Alice$ info
Vertex: Alice
Degree: 2
In-degree: 2
Out-degree: 2

Attributes:
  age = 30
  city = NYC

graph(5v):/Alice$ neighbors
Neighbors of Alice:
  Bob <->
  Charlie <->

graph(5v):/Alice$ cd neighbors
Now at: /Alice/neighbors

graph(5v):/Alice/neighbors$ ls
Bob/  <->
Charlie/  <->

graph(5v):/Alice/neighbors$ cd Bob
Now at: /Bob

graph(5v):/Bob$ pwd
/Bob

graph(5v):/Bob$ path Alice Eve
Path found: Alice -> Bob -> Diana -> Eve
Length: 3 edges

graph(5v):/Bob$ cd /
Now at: /

graph(5v):/$ components
Connected components: 1

Component 1 (5 vertices):
  Alice, Bob, Charlie, Diana, Eve

graph(5v):/$ exit
Goodbye!
```

## Comparison with AlgoTree Shell

| Feature | AlgoTree Shell | AlgoGraph Shell |
|---------|---------------|-----------------|
| Navigation | Tree hierarchy (parent/child) | Graph structure (vertices/neighbors) |
| `cd` | Navigate to child | Navigate to any vertex |
| `ls` at node | Shows children | Shows attributes + neighbors/ |
| `cd neighbors` | N/A | Special mode to view neighbors |
| Special dirs | None | neighbors/ pseudo-directory |
| Graph cycles | N/A | Fully supported |
| Bidirectional | Parent → child only | Any direction via neighbors |

## VFS Design Philosophy

The shell treats the graph as a virtual filesystem where:

1. **Vertices are directories**: You can "cd" into them
2. **Attributes are files**: Shown by "ls"
3. **Neighbors are a special directory**: Accessible via `cd neighbors`
4. **Edges are implicit links**: Navigate by cd'ing to neighbors

This is a natural fit because:
- Filesystems with symlinks/hardlinks ARE graphs
- The `cd` metaphor works well for graph navigation
- `ls` naturally shows "what's here" (attributes) and "where can I go" (neighbors)
- The neighbors/ pseudo-directory makes neighbor exploration intuitive

## Design Principles

1. **Immutability**: All navigation operations return new context
2. **No modification**: Shell is read-only (graph cannot be modified)
3. **Stateful navigation**: Current vertex is maintained between commands
4. **Mode-based**: Different "ls" behavior at root vs vertex vs neighbors
5. **Filesystem metaphor**: Leverages familiar cd/ls/pwd commands

## Implementation Details

- **GraphContext**: Immutable navigation state (current vertex, mode)
- **Commands**: Each command is a class with `execute()` method
- **CommandResult**: Encapsulates success, output, new context, error
- **GraphShell**: REPL that maintains context and dispatches commands

## Future Enhancements

Potential additions:
- Graph modification commands (add vertex, add edge)
- Save/load graph state
- Visualization integration
- Tab completion for vertex names
- History navigation (back/forward)
- Bookmarks (mark vertex, return to marked vertex)
- Path highlighting in visual output
