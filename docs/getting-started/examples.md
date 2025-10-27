# Basic Examples

This page provides practical examples to help you get started with AlgoGraph.

## Example 1: Social Network Analysis

A simple social network where we analyze friend connections:

```python
from AlgoGraph import Vertex, Edge, Graph
from AlgoGraph.algorithms import (
    connected_components,
    shortest_path,
    shortest_path_length,
    diameter
)

# Create people with attributes
people = {
    Vertex('Alice', attrs={'age': 30, 'interests': ['reading', 'hiking']}),
    Vertex('Bob', attrs={'age': 25, 'interests': ['gaming', 'coding']}),
    Vertex('Carol', attrs={'age': 28, 'interests': ['hiking', 'photography']}),
    Vertex('Dave', attrs={'age': 35, 'interests': ['reading', 'cooking']}),
    Vertex('Eve', attrs={'age': 32, 'interests': ['coding', 'music']}),
}

# Create friendships (undirected, all weighted equally)
friendships = {
    Edge('Alice', 'Bob', directed=False),
    Edge('Alice', 'Carol', directed=False),
    Edge('Bob', 'Eve', directed=False),
    Edge('Carol', 'Dave', directed=False),
    Edge('Dave', 'Eve', directed=False),
}

social_network = Graph(people, friendships)

# Analysis 1: Find friend-of-friend connections
print("Alice's direct friends:", social_network.neighbors('Alice'))
# {'Bob', 'Carol'}

# Analysis 2: Shortest path between people
path = shortest_path(social_network, 'Alice', 'Eve')
print(f"Shortest path from Alice to Eve: {' -> '.join(path)}")
# Alice -> Bob -> Eve

# Analysis 3: Find degree of separation
separation = shortest_path_length(social_network, 'Alice', 'Dave')
print(f"Alice and Dave are {separation} connections apart")
# 2

# Analysis 4: Check network connectivity
components = connected_components(social_network)
print(f"Network has {len(components)} connected component(s)")
# 1 (everyone is connected)

# Analysis 5: Find the network diameter (longest shortest path)
network_diameter = diameter(social_network)
print(f"Network diameter: {network_diameter}")
# Maximum separation in the network

# Find people with shared interests
def find_people_by_interest(graph, interest):
    return {
        v.id for v in graph.find_vertices(
            lambda v: interest in v.get('interests', [])
        )
    }

hikers = find_people_by_interest(social_network, 'hiking')
print(f"People interested in hiking: {hikers}")
# {'Alice', 'Carol'}
```

## Example 2: Task Dependency Graph

Model project tasks with dependencies:

```python
from AlgoGraph import Vertex, Edge, Graph
from AlgoGraph.algorithms import (
    topological_sort,
    has_cycle,
    find_path
)

# Define tasks with estimated hours
tasks = {
    Vertex('design', attrs={'hours': 8, 'assignee': 'Alice'}),
    Vertex('setup_env', attrs={'hours': 2, 'assignee': 'Bob'}),
    Vertex('backend', attrs={'hours': 16, 'assignee': 'Carol'}),
    Vertex('frontend', attrs={'hours': 12, 'assignee': 'Dave'}),
    Vertex('testing', attrs={'hours': 8, 'assignee': 'Eve'}),
    Vertex('deployment', attrs={'hours': 4, 'assignee': 'Bob'}),
}

# Define dependencies (A -> B means B depends on A)
dependencies = {
    Edge('design', 'backend'),
    Edge('design', 'frontend'),
    Edge('setup_env', 'backend'),
    Edge('setup_env', 'frontend'),
    Edge('backend', 'testing'),
    Edge('frontend', 'testing'),
    Edge('testing', 'deployment'),
}

project = Graph(tasks, dependencies)

# Check for circular dependencies
if has_cycle(project):
    print("ERROR: Circular dependency detected!")
else:
    print("No circular dependencies found")

# Get execution order
execution_order = topological_sort(project)
print("\nTask execution order:")
for i, task_id in enumerate(execution_order, 1):
    task = project.get_vertex(task_id)
    print(f"{i}. {task_id} ({task.get('hours')}h, {task.get('assignee')})")

# Find critical path to deployment
critical_path = find_path(project, 'design', 'deployment')
total_hours = sum(
    project.get_vertex(task).get('hours')
    for task in critical_path
)
print(f"\nCritical path: {' -> '.join(critical_path)}")
print(f"Minimum project duration: {total_hours} hours")

# Find which tasks can run in parallel
def find_parallel_tasks(graph, order):
    """Group tasks that can run simultaneously."""
    levels = []
    completed = set()

    for task in order:
        # Check if all dependencies are completed
        deps = {e.source for e in graph.edges if e.target == task}
        if deps.issubset(completed):
            # Can start immediately
            if not levels or task in completed:
                levels.append([task])
            else:
                levels[-1].append(task)
        else:
            # Must wait for dependencies
            levels.append([task])
        completed.add(task)

    return levels

parallel_groups = find_parallel_tasks(project, execution_order)
print("\nParallel execution phases:")
for i, phase in enumerate(parallel_groups, 1):
    print(f"Phase {i}: {', '.join(phase)}")
```

## Example 3: City Road Network

Shortest path in a weighted graph:

```python
from AlgoGraph import Vertex, Edge, Graph
from AlgoGraph.algorithms import (
    dijkstra,
    shortest_path,
    all_shortest_paths
)

# Cities with population data
cities = {
    Vertex('Boston', attrs={'population': 675000}),
    Vertex('NYC', attrs={'population': 8336000}),
    Vertex('Philadelphia', attrs={'population': 1584000}),
    Vertex('Washington DC', attrs={'population': 705000}),
    Vertex('Baltimore', attrs={'population': 585000}),
}

# Roads with distances in miles
roads = {
    Edge('Boston', 'NYC', weight=215),
    Edge('NYC', 'Philadelphia', weight=95),
    Edge('Philadelphia', 'Baltimore', weight=100),
    Edge('Baltimore', 'Washington DC', weight=40),
    Edge('Philadelphia', 'Washington DC', weight=140),
    Edge('NYC', 'Baltimore', weight=185),
}

road_network = Graph(cities, roads)

# Find shortest routes from Boston
distances, predecessors = dijkstra(road_network, 'Boston')

print("Distances from Boston:")
for city in sorted(distances.keys()):
    if city != 'Boston':
        print(f"  {city}: {distances[city]} miles")

# Get the actual shortest path
path_to_dc = shortest_path(road_network, 'Boston', 'Washington DC')
print(f"\nShortest route to Washington DC:")
print(' -> '.join(path_to_dc))
print(f"Total distance: {distances['Washington DC']} miles")

# Find all shortest paths (if multiple exist)
all_paths = all_shortest_paths(road_network, 'NYC', 'Washington DC')
print(f"\nAll shortest paths from NYC to Washington DC:")
for path in all_paths:
    print(f"  {' -> '.join(path)}")

# Find cities within a certain distance
def cities_within_distance(graph, start, max_distance):
    """Find all cities within max_distance from start."""
    distances, _ = dijkstra(graph, start)
    return {
        city: dist
        for city, dist in distances.items()
        if dist <= max_distance and city != start
    }

nearby = cities_within_distance(road_network, 'Boston', 300)
print(f"\nCities within 300 miles of Boston:")
for city, dist in sorted(nearby.items(), key=lambda x: x[1]):
    print(f"  {city}: {dist} miles")
```

## Example 4: Bipartite Matching

Model a bipartite graph (e.g., students and courses):

```python
from AlgoGraph import Vertex, Edge, Graph
from AlgoGraph.algorithms import is_bipartite

# Create students and courses
students = {
    Vertex('Alice', attrs={'type': 'student', 'year': 3}),
    Vertex('Bob', attrs={'type': 'student', 'year': 2}),
    Vertex('Carol', attrs={'type': 'student', 'year': 4}),
}

courses = {
    Vertex('CS101', attrs={'type': 'course', 'capacity': 30}),
    Vertex('CS201', attrs={'type': 'course', 'capacity': 25}),
    Vertex('CS301', attrs={'type': 'course', 'capacity': 20}),
}

# Create enrollments (undirected edges)
enrollments = {
    Edge('Alice', 'CS101', directed=False),
    Edge('Alice', 'CS201', directed=False),
    Edge('Bob', 'CS101', directed=False),
    Edge('Bob', 'CS301', directed=False),
    Edge('Carol', 'CS201', directed=False),
    Edge('Carol', 'CS301', directed=False),
}

enrollment_graph = Graph(students | courses, enrollments)

# Verify it's bipartite (students never connect to students,
# courses never connect to courses)
bipartite, partition = is_bipartite(enrollment_graph)
print(f"Is bipartite: {bipartite}")

if bipartite:
    set1, set2 = partition
    print(f"Partition 1: {set1}")
    print(f"Partition 2: {set2}")

# Find students in a specific course
def students_in_course(graph, course_id):
    return graph.neighbors(course_id)

print(f"\nStudents in CS101: {students_in_course(enrollment_graph, 'CS101')}")

# Find courses for a student
def courses_for_student(graph, student_id):
    return graph.neighbors(student_id)

print(f"Alice's courses: {courses_for_student(enrollment_graph, 'Alice')}")

# Find students enrolled in multiple courses
def students_with_multiple_courses(graph, min_courses):
    student_vertices = graph.find_vertices(
        lambda v: v.get('type') == 'student'
    )
    return {
        v.id for v in student_vertices
        if graph.degree(v.id) >= min_courses
    }

overloaded = students_with_multiple_courses(enrollment_graph, 2)
print(f"\nStudents taking 2+ courses: {overloaded}")
```

## Example 5: Graph Transformation

Modify graphs immutably:

```python
from AlgoGraph import Vertex, Edge, Graph

# Start with a simple graph
original = Graph(
    vertices={Vertex('A'), Vertex('B'), Vertex('C')},
    edges={Edge('A', 'B'), Edge('B', 'C')}
)

# Add vertex attributes
def add_labels(graph):
    """Add labels to all vertices."""
    new_graph = graph
    for v in graph.vertices:
        updated = v.with_attrs(label=f"Vertex {v.id}")
        new_graph = new_graph.update_vertex(updated)
    return new_graph

labeled = add_labels(original)

# Convert directed to undirected
def to_undirected(graph):
    """Convert all edges to undirected."""
    new_edges = {e.to_undirected() for e in graph.edges}
    return Graph(vertices=graph.vertices, edges=new_edges)

undirected = to_undirected(original)

# Scale all edge weights
def scale_weights(graph, factor):
    """Multiply all edge weights by a factor."""
    new_edges = {e.with_weight(e.weight * factor) for e in graph.edges}
    return Graph(vertices=graph.vertices, edges=new_edges)

weighted = Graph(
    vertices={Vertex('A'), Vertex('B')},
    edges={Edge('A', 'B', weight=10)}
)
scaled = scale_weights(weighted, 2.5)
print(scaled.get_edge('A', 'B').weight)  # 25.0

# Filter graph by predicate
def filter_by_degree(graph, min_degree):
    """Keep only vertices with degree >= min_degree."""
    valid_vertices = {
        v.id for v in graph.vertices
        if graph.degree(v.id) >= min_degree
    }
    return graph.subgraph(valid_vertices)

filtered = filter_by_degree(original, 2)
print(f"Vertices with degree >= 2: {[v.id for v in filtered.vertices]}")
```

## Next Steps

These examples demonstrate common patterns in AlgoGraph. For more advanced use cases:

- Browse the [Cookbook](../examples/social-networks.md) for real-world scenarios
- Learn about all [Available Algorithms](../user-guide/algorithms.md)
- Explore the [API Reference](../api/vertex.md) for detailed documentation
- Try the [Interactive Shell](../shell/overview.md) to explore graphs hands-on
