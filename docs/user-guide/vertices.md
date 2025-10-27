# Working with Vertices

Vertices (also called nodes) are the fundamental building blocks of graphs. This guide covers everything you need to know about creating and using vertices in AlgoGraph.

## Creating Vertices

### Basic Vertices

The simplest vertex has just an ID:

```python
from AlgoGraph import Vertex

v = Vertex('A')
print(v.id)  # 'A'
```

Vertex IDs must be strings:

```python
# Valid
v1 = Vertex('node1')
v2 = Vertex('Alice')
v3 = Vertex('city_42')

# If you have non-string IDs, convert them
v4 = Vertex(str(42))  # Integer to string
v5 = Vertex(str(uuid.uuid4()))  # UUID to string
```

### Vertices with Attributes

Add arbitrary attributes as a dictionary:

```python
# Person vertex
person = Vertex('Alice', attrs={
    'age': 30,
    'city': 'NYC',
    'occupation': 'Engineer'
})

# City vertex
city = Vertex('NYC', attrs={
    'population': 8336000,
    'country': 'USA',
    'coordinates': (40.7128, -74.0060)
})

# Generic data vertex
data = Vertex('node1', attrs={
    'value': 42,
    'timestamp': '2025-01-15T10:30:00',
    'tags': ['important', 'verified'],
    'metadata': {'source': 'api', 'confidence': 0.95}
})
```

!!! note "Attribute Types"
    Attributes can be any JSON-serializable type: strings, numbers, booleans, lists, dicts, null. Avoid non-serializable types like functions or custom objects if you plan to save graphs to JSON.

## Accessing Vertex Data

### Get the ID

```python
v = Vertex('London', attrs={'country': 'UK'})
print(v.id)  # 'London'
```

### Get Attributes

Use the `get()` method (like a dictionary):

```python
v = Vertex('Alice', attrs={'age': 30, 'city': 'NYC'})

# Get attribute value
age = v.get('age')  # 30
city = v.get('city')  # 'NYC'

# Get with default value
email = v.get('email', 'N/A')  # 'N/A' (not present)

# Access the full attrs dict
print(v.attrs)  # {'age': 30, 'city': 'NYC'}
```

!!! warning "Don't Modify attrs Directly"
    Even though you can access `v.attrs`, don't modify it directly:
    ```python
    # BAD - will raise an error (frozen dataclass)
    v.attrs['age'] = 31  # Error!

    # GOOD - create new vertex with updated attributes
    v2 = v.with_attrs(age=31)
    ```

## Modifying Vertices (Immutably)

Since vertices are immutable, "modifying" a vertex creates a new one:

### Add or Update Attributes

```python
v1 = Vertex('Alice', attrs={'age': 30})

# Add new attribute
v2 = v1.with_attrs(city='NYC')
print(v2.attrs)  # {'age': 30, 'city': 'NYC'}

# Update existing attribute
v3 = v1.with_attrs(age=31)
print(v3.attrs)  # {'age': 31}

# Add multiple attributes
v4 = v1.with_attrs(age=31, city='NYC', job='Engineer')
print(v4.attrs)  # {'age': 31, 'city': 'NYC', 'job': 'Engineer'}

# Original is unchanged
print(v1.attrs)  # {'age': 30}
```

### Remove Attributes

```python
v1 = Vertex('Alice', attrs={'age': 30, 'city': 'NYC', 'job': 'Engineer'})

# Remove one attribute
v2 = v1.without_attrs('job')
print(v2.attrs)  # {'age': 30, 'city': 'NYC'}

# Remove multiple attributes
v3 = v1.without_attrs('city', 'job')
print(v3.attrs)  # {'age': 30}

# Remove non-existent attribute (no error)
v4 = v1.without_attrs('email')  # No change
```

### Change Vertex ID

```python
v1 = Vertex('Alice', attrs={'age': 30})

# Create vertex with new ID (keeps attributes)
v2 = v1.with_id('Alice_Smith')
print(v2.id)     # 'Alice_Smith'
print(v2.attrs)  # {'age': 30}
```

## Vertex Comparison

### Equality

Vertices are equal if they have the same ID **and** attributes:

```python
v1 = Vertex('A', attrs={'x': 1})
v2 = Vertex('A', attrs={'x': 1})
v3 = Vertex('A', attrs={'x': 2})
v4 = Vertex('B', attrs={'x': 1})

v1 == v2  # True (same ID and attrs)
v1 == v3  # False (same ID, different attrs)
v1 == v4  # False (different ID)
```

### Hashing

Vertices are hashable (can be used in sets and as dict keys). The hash is based on the ID only:

```python
v1 = Vertex('A', attrs={'x': 1})
v2 = Vertex('A', attrs={'x': 2})

# Same hash (same ID)
hash(v1) == hash(v2)  # True

# Can put in sets
vertices = {v1, v2}
len(vertices)  # 1 (only one 'A' vertex in set)
```

!!! warning "Set Behavior"
    When vertices with the same ID are in a set, only one is kept:
    ```python
    vertices = {
        Vertex('A', attrs={'x': 1}),
        Vertex('A', attrs={'x': 2}),  # Replaces the first one
    }
    # The set contains just one vertex, but which attrs it has is undefined
    ```

## String Representation

Vertices have helpful string representations:

```python
v1 = Vertex('A')
v2 = Vertex('Alice', attrs={'age': 30, 'city': 'NYC'})

# repr() shows full details
print(repr(v1))  # Vertex('A')
print(repr(v2))  # Vertex('Alice', age=30, city='NYC')

# str() shows just the ID
print(str(v1))  # A
print(str(v2))  # Alice

# Useful in f-strings
print(f"Processing {v2}")  # Processing Alice
```

## Common Patterns

### Creating Vertices from Data

```python
# From a list of names
names = ['Alice', 'Bob', 'Charlie']
vertices = {Vertex(name) for name in names}

# From a dictionary
data = {
    'Alice': {'age': 30, 'city': 'NYC'},
    'Bob': {'age': 25, 'city': 'Boston'},
}
vertices = {Vertex(name, attrs=attrs) for name, attrs in data.items()}

# From CSV or database
import csv

with open('people.csv') as f:
    reader = csv.DictReader(f)
    vertices = {
        Vertex(row['id'], attrs={
            'name': row['name'],
            'age': int(row['age']),
        })
        for row in reader
    }
```

### Filtering Vertices

```python
from AlgoGraph import Graph

g = Graph({
    Vertex('Alice', attrs={'age': 30, 'city': 'NYC'}),
    Vertex('Bob', attrs={'age': 25, 'city': 'Boston'}),
    Vertex('Carol', attrs={'age': 35, 'city': 'NYC'}),
})

# Find vertices by attribute
nyc_residents = g.find_vertices(lambda v: v.get('city') == 'NYC')
# {Vertex('Alice', ...), Vertex('Carol', ...)}

# Find vertices by age
seniors = g.find_vertices(lambda v: v.get('age', 0) >= 30)

# Find vertices with specific attribute present
has_email = g.find_vertices(lambda v: 'email' in v.attrs)
```

### Transforming Vertex Attributes

```python
# Add a computed attribute to all vertices
def add_generation(vertex):
    """Add generation based on age."""
    age = vertex.get('age', 0)
    if age < 30:
        generation = 'young'
    elif age < 50:
        generation = 'middle'
    else:
        generation = 'senior'
    return vertex.with_attrs(generation=generation)

# Apply to all vertices in graph
new_graph = graph
for v in graph.vertices:
    updated = add_generation(v)
    new_graph = new_graph.update_vertex(updated)
```

### Batch Vertex Operations

```python
# Create many vertices at once
vertices = {
    Vertex(f'node_{i}', attrs={'value': i * 10})
    for i in range(100)
}

# Update multiple vertices
graph = Graph(vertices)
updated_graph = graph
for v in graph.vertices:
    if v.get('value', 0) > 500:
        updated = v.with_attrs(priority='high')
        updated_graph = updated_graph.update_vertex(updated)
```

### Using Vertices with Special Names

Vertex IDs can contain any characters, including spaces and special characters:

```python
# Spaces in names
v1 = Vertex('Alice Smith')
v2 = Vertex('New York City')

# Special characters
v3 = Vertex('item-42')
v4 = Vertex('user@example.com')
v5 = Vertex('node_with_underscores')

# Unicode
v6 = Vertex('François')
v7 = Vertex('東京')
```

When using the interactive shell with special names, use quotes:

```bash
cd "Alice Smith"
cd "New York City"
```

## Best Practices

### 1. Choose Meaningful IDs

Use descriptive IDs that make sense in your domain:

```python
# Good
Vertex('London')
Vertex('user_12345')
Vertex('transaction_abc123')

# Less ideal
Vertex('1')
Vertex('x')
Vertex('temp')
```

### 2. Keep Attributes Flat When Possible

Flat attributes are easier to query:

```python
# Good
Vertex('Alice', attrs={
    'first_name': 'Alice',
    'last_name': 'Smith',
    'age': 30,
    'city': 'NYC'
})

# Can work but harder to query
Vertex('Alice', attrs={
    'name': {'first': 'Alice', 'last': 'Smith'},
    'location': {'city': 'NYC', 'zip': '10001'}
})
```

### 3. Use Consistent Attribute Names

Standardize attribute names across vertices:

```python
# Good - consistent naming
Vertex('Alice', attrs={'age': 30, 'city': 'NYC'})
Vertex('Bob', attrs={'age': 25, 'city': 'Boston'})

# Confusing - inconsistent naming
Vertex('Alice', attrs={'age': 30, 'city': 'NYC'})
Vertex('Bob', attrs={'years_old': 25, 'location': 'Boston'})
```

### 4. Validate Attribute Values

Add validation when creating vertices from external data:

```python
def create_person_vertex(id: str, age: int, city: str) -> Vertex:
    """Create a person vertex with validation."""
    if not id:
        raise ValueError("ID cannot be empty")
    if age < 0 or age > 150:
        raise ValueError(f"Invalid age: {age}")
    if not city:
        raise ValueError("City cannot be empty")

    return Vertex(id, attrs={'age': age, 'city': city})
```

## Next Steps

- Learn about [Working with Edges](edges.md)
- See how to [Build Graphs](graphs.md)
- Explore [Graph Algorithms](algorithms.md)
