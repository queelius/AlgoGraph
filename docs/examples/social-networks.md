# Social Networks

Learn how to model and analyze social networks using AlgoGraph.

## Basic Social Network

Let's create a simple friend network:

```python
from AlgoGraph import Vertex, Edge, Graph
from AlgoGraph.algorithms import (
    shortest_path,
    connected_components,
    diameter,
    find_bridges
)

# Create people with attributes
people = {
    Vertex('Alice', attrs={
        'age': 30,
        'city': 'NYC',
        'interests': ['reading', 'hiking'],
        'join_date': '2020-01-15'
    }),
    Vertex('Bob', attrs={
        'age': 25,
        'city': 'Boston',
        'interests': ['gaming', 'coding'],
        'join_date': '2020-03-20'
    }),
    Vertex('Carol', attrs={
        'age': 28,
        'city': 'NYC',
        'interests': ['hiking', 'photography'],
        'join_date': '2020-02-10'
    }),
    Vertex('Dave', attrs={
        'age': 35,
        'city': 'Chicago',
        'interests': ['reading', 'cooking'],
        'join_date': '2019-11-05'
    }),
    Vertex('Eve', attrs={
        'age': 32,
        'city': 'Boston',
        'interests': ['coding', 'music'],
        'join_date': '2020-04-18'
    }),
}

# Create friendships (undirected)
friendships = {
    Edge('Alice', 'Bob', directed=False, attrs={'since': '2020-02-01'}),
    Edge('Alice', 'Carol', directed=False, attrs={'since': '2020-02-15'}),
    Edge('Bob', 'Eve', directed=False, attrs={'since': '2020-04-20'}),
    Edge('Carol', 'Dave', directed=False, attrs={'since': '2020-03-10'}),
    Edge('Dave', 'Eve', directed=False, attrs={'since': '2020-05-01'}),
}

network = Graph(people, friendships)
```

## Analysis 1: Degrees of Separation

Find how connected people are:

```python
# Friend-of-a-friend analysis
def analyze_separation(network, person1, person2):
    """Analyze connection between two people."""
    path = shortest_path(network, person1, person2)

    if not path:
        return f"{person1} and {person2} are not connected"

    separation = len(path) - 1

    if separation == 0:
        return f"{person1} is {person2}"
    elif separation == 1:
        return f"{person1} and {person2} are direct friends"
    elif separation == 2:
        middle = path[1]
        return f"{person1} and {person2} are friends-of-friends (through {middle})"
    else:
        return f"{person1} and {person2} are {separation} connections apart: {' -> '.join(path)}"

# Test it
print(analyze_separation(network, 'Alice', 'Eve'))
# Alice and Eve are friends-of-friends (through Bob)

print(analyze_separation(network, 'Alice', 'Dave'))
# Alice and Dave are friends-of-friends (through Carol)
```

## Analysis 2: Find Influencers

Identify well-connected people:

```python
def find_influencers(network, min_friends=3):
    """Find people with many connections."""
    influencers = []

    for person in network.vertices:
        degree = network.degree(person.id)
        if degree >= min_friends:
            influencers.append({
                'name': person.id,
                'friends': degree,
                'city': person.get('city'),
            })

    # Sort by number of friends
    return sorted(influencers, key=lambda x: x['friends'], reverse=True)

influencers = find_influencers(network, min_friends=2)
for person in influencers:
    print(f"{person['name']}: {person['friends']} friends ({person['city']})")
```

## Analysis 3: Community Detection

Find groups of closely connected people:

```python
# Find connected components
components = connected_components(network)

print(f"Found {len(components)} community(ies):")
for i, component in enumerate(components, 1):
    members = sorted(component)
    print(f"\nCommunity {i} ({len(members)} members):")

    # Show members and their details
    for member in members:
        person = network.get_vertex(member)
        print(f"  - {member} ({person.get('age')}, {person.get('city')})")

    # Find common interests
    interests = set()
    for member in members:
        person = network.get_vertex(member)
        interests.update(person.get('interests', []))

    print(f"  Common interests: {', '.join(sorted(interests))}")
```

## Analysis 4: Critical Connections

Find friendships that bridge communities:

```python
# Find bridge edges (removing them disconnects the network)
bridges = find_bridges(network)

if bridges:
    print("Critical friendships (bridges):")
    for bridge in bridges:
        edge = network.get_edge(bridge[0], bridge[1])
        since = edge.get('since', 'unknown') if edge else 'unknown'
        print(f"  {bridge[0]} <-> {bridge[1]} (friends since {since})")
else:
    print("No critical friendships - network is well-connected")
```

## Analysis 5: Friend Recommendations

Suggest new friends based on mutual connections:

```python
def recommend_friends(network, person_id, max_recommendations=3):
    """Recommend friends based on mutual connections."""
    if not network.has_vertex(person_id):
        return []

    # Get direct friends
    direct_friends = network.neighbors(person_id)

    # Get friends-of-friends
    recommendations = {}
    for friend in direct_friends:
        friends_of_friend = network.neighbors(friend)
        for potential_friend in friends_of_friend:
            # Skip self and existing friends
            if potential_friend == person_id or potential_friend in direct_friends:
                continue

            # Count mutual friends
            if potential_friend not in recommendations:
                recommendations[potential_friend] = {
                    'mutual_friends': [],
                    'score': 0
                }

            recommendations[potential_friend]['mutual_friends'].append(friend)
            recommendations[potential_friend]['score'] += 1

    # Sort by number of mutual friends
    sorted_recs = sorted(
        recommendations.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )

    # Format results
    results = []
    for person, data in sorted_recs[:max_recommendations]:
        person_vertex = network.get_vertex(person)
        results.append({
            'name': person,
            'mutual_friends': data['mutual_friends'],
            'mutual_count': data['score'],
            'age': person_vertex.get('age'),
            'city': person_vertex.get('city'),
            'interests': person_vertex.get('interests', [])
        })

    return results

# Get recommendations for Alice
recommendations = recommend_friends(network, 'Alice', max_recommendations=3)

print(f"\nFriend recommendations for Alice:")
for rec in recommendations:
    mutuals = ', '.join(rec['mutual_friends'])
    interests = ', '.join(rec['interests'])
    print(f"\n{rec['name']} ({rec['age']}, {rec['city']})")
    print(f"  {rec['mutual_count']} mutual friend(s): {mutuals}")
    print(f"  Interests: {interests}")
```

## Analysis 6: Shared Interests Network

Create a bipartite graph of people and interests:

```python
from AlgoGraph.algorithms import is_bipartite

# Create interest vertices
all_interests = set()
for person in network.vertices:
    all_interests.update(person.get('interests', []))

interest_vertices = {
    Vertex(interest, attrs={'type': 'interest'})
    for interest in all_interests
}

# Create person-interest edges
person_interest_edges = set()
for person in network.vertices:
    for interest in person.get('interests', []):
        person_interest_edges.add(
            Edge(person.id, interest, directed=False)
        )

# Create bipartite graph
people_vertices = {
    v.with_attrs(type='person')
    for v in network.vertices
}

bipartite_graph = Graph(
    vertices=people_vertices | interest_vertices,
    edges=person_interest_edges
)

# Verify it's bipartite
is_bip, partition = is_bipartite(bipartite_graph)
print(f"Is bipartite: {is_bip}")

# Find people who share interests
def find_shared_interests(person1_id, person2_id):
    """Find interests shared between two people."""
    interests1 = bipartite_graph.neighbors(person1_id)
    interests2 = bipartite_graph.neighbors(person2_id)
    return interests1 & interests2

shared = find_shared_interests('Alice', 'Dave')
print(f"\nAlice and Dave share interests: {shared}")
```

## Weighted Network: Interaction Strength

Model relationship strength with edge weights:

```python
# Create network with interaction weights
weighted_edges = {
    Edge('Alice', 'Bob', directed=False, weight=0.9, attrs={
        'messages': 150,
        'last_interaction': '2025-10-25'
    }),
    Edge('Alice', 'Carol', directed=False, weight=0.8, attrs={
        'messages': 120,
        'last_interaction': '2025-10-24'
    }),
    Edge('Bob', 'Eve', directed=False, weight=0.7, attrs={
        'messages': 90,
        'last_interaction': '2025-10-23'
    }),
    # Low-weight connections (weak ties)
    Edge('Carol', 'Dave', directed=False, weight=0.3, attrs={
        'messages': 15,
        'last_interaction': '2025-09-10'
    }),
}

weighted_network = Graph(people, weighted_edges)

# Find strong connections
def find_strong_connections(network, min_weight=0.7):
    """Find strong relationships."""
    strong = network.find_edges(lambda e: e.weight >= min_weight)

    results = []
    for edge in strong:
        results.append({
            'connection': f"{edge.source} <-> {edge.target}",
            'strength': edge.weight,
            'messages': edge.get('messages', 0),
            'last_contact': edge.get('last_interaction', 'unknown')
        })

    return sorted(results, key=lambda x: x['strength'], reverse=True)

strong_ties = find_strong_connections(weighted_network)
print("\nStrong connections:")
for tie in strong_ties:
    print(f"{tie['connection']}: strength {tie['strength']}")
    print(f"  {tie['messages']} messages, last: {tie['last_contact']}")
```

## Network Metrics

Calculate important network statistics:

```python
from AlgoGraph.algorithms import diameter

def network_statistics(network):
    """Calculate key network metrics."""
    stats = {
        'total_people': network.vertex_count,
        'total_friendships': network.edge_count,
        'components': len(connected_components(network)),
        'is_connected': len(connected_components(network)) == 1,
    }

    # Average degree
    degrees = [network.degree(v.id) for v in network.vertices]
    stats['avg_friends'] = sum(degrees) / len(degrees) if degrees else 0
    stats['max_friends'] = max(degrees) if degrees else 0
    stats['min_friends'] = min(degrees) if degrees else 0

    # Diameter (if connected)
    if stats['is_connected']:
        stats['diameter'] = diameter(network)
    else:
        stats['diameter'] = None

    return stats

stats = network_statistics(network)
print("\nNetwork Statistics:")
print(f"People: {stats['total_people']}")
print(f"Friendships: {stats['total_friendships']}")
print(f"Average friends: {stats['avg_friends']:.1f}")
print(f"Max friends: {stats['max_friends']}")
print(f"Network diameter: {stats['diameter']}")
```

## Saving and Loading

Persist your social network:

```python
from AlgoGraph import save_graph, load_graph

# Save network
save_graph(network, 'social_network.json')

# Load it back
loaded_network = load_graph('social_network.json')

# Verify
assert loaded_network.vertex_count == network.vertex_count
assert loaded_network.edge_count == network.edge_count
```

## Interactive Exploration

Use the shell to explore interactively:

```bash
$ python3 -m AlgoGraph.shell.shell social_network.json

graph(5v):/$ ls
Alice/  [2 neighbors]
Bob/  [2 neighbors]
Carol/  [2 neighbors]
Dave/  [2 neighbors]
Eve/  [2 neighbors]

graph(5v):/$ cd Alice

graph(5v):/Alice$ info
Vertex: Alice
Degree: 2

Attributes:
  age = 30
  city = NYC
  interests = ['reading', 'hiking']

graph(5v):/Alice$ neighbors
Neighbors of Alice:
  Bob <-> (weight: 1.0)
  Carol <-> (weight: 1.0)

graph(5v):/Alice$ path Alice Eve
Path found: Alice -> Bob -> Eve
Length: 2 edges
```

## Next Steps

- See [Road Networks](road-networks.md) for spatial graphs
- Check [Dependency Graphs](dependency-graphs.md) for DAG examples
- Explore more in [Real-World Use Cases](use-cases.md)
