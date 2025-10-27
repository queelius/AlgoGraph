# Vertex API Reference

Complete API documentation for the `Vertex` class.

## Class Definition

```python
@dataclass(frozen=True)
class Vertex:
    """Immutable graph vertex with attributes."""
    id: str
    attrs: Dict[str, Any] = field(default_factory=dict)
```

## Constructor

### `Vertex(id, attrs={})`

Create a new vertex.

**Parameters:**

- `id` (str): Unique identifier for the vertex
- `attrs` (Dict[str, Any], optional): Dictionary of vertex attributes

**Returns:** Vertex instance

**Example:**

```python
from AlgoGraph import Vertex

# Simple vertex
v1 = Vertex('A')

# Vertex with attributes
v2 = Vertex('London', attrs={
    'population': 9000000,
    'country': 'UK'
})
```

## Attributes

### `id`

**Type:** `str` (read-only)

The unique identifier of the vertex.

```python
v = Vertex('Alice')
print(v.id)  # 'Alice'
```

### `attrs`

**Type:** `Dict[str, Any]` (read-only)

Dictionary of vertex attributes. While you can access this dict, you cannot modify it directly due to the frozen dataclass.

```python
v = Vertex('A', attrs={'x': 10, 'y': 20})
print(v.attrs)  # {'x': 10, 'y': 20}
```

## Methods

### `get(key, default=None)`

Get an attribute value by key.

**Parameters:**

- `key` (str): Attribute key to look up
- `default` (Any, optional): Default value if key not found (default: None)

**Returns:** Attribute value or default

**Example:**

```python
v = Vertex('A', attrs={'value': 42})

v.get('value')           # 42
v.get('missing')         # None
v.get('missing', 0)      # 0
```

---

### `with_attrs(**kwargs)`

Create a new vertex with added or updated attributes.

**Parameters:**

- `**kwargs`: Keyword arguments for attributes to add/update

**Returns:** New Vertex instance with updated attributes

**Example:**

```python
v1 = Vertex('A', attrs={'x': 10})

v2 = v1.with_attrs(y=20)
print(v2.attrs)  # {'x': 10, 'y': 20}

v3 = v1.with_attrs(x=15, z=30)
print(v3.attrs)  # {'x': 15, 'z': 30}

# Original unchanged
print(v1.attrs)  # {'x': 10}
```

---

### `without_attrs(*keys)`

Create a new vertex with specified attributes removed.

**Parameters:**

- `*keys` (str): Attribute keys to remove

**Returns:** New Vertex instance without specified attributes

**Example:**

```python
v1 = Vertex('A', attrs={'x': 10, 'y': 20, 'z': 30})

v2 = v1.without_attrs('y')
print(v2.attrs)  # {'x': 10, 'z': 30}

v3 = v1.without_attrs('x', 'z')
print(v3.attrs)  # {'y': 20}

# Removing non-existent key is safe
v4 = v1.without_attrs('missing')  # No error
```

---

### `with_id(new_id)`

Create a new vertex with a different ID (preserves attributes).

**Parameters:**

- `new_id` (str): New vertex ID

**Returns:** New Vertex instance with updated ID

**Example:**

```python
v1 = Vertex('A', attrs={'value': 10})

v2 = v1.with_id('B')
print(v2.id)     # 'B'
print(v2.attrs)  # {'value': 10}
```

## Special Methods

### `__repr__()`

Detailed string representation showing ID and attributes.

**Returns:** str

**Example:**

```python
v1 = Vertex('A')
print(repr(v1))  # Vertex('A')

v2 = Vertex('A', attrs={'x': 10, 'y': 20})
print(repr(v2))  # Vertex('A', x=10, y=20)
```

---

### `__str__()`

Simple string representation (just the ID).

**Returns:** str

**Example:**

```python
v = Vertex('Alice', attrs={'age': 30})
print(str(v))  # 'Alice'
print(f"User: {v}")  # 'User: Alice'
```

---

### `__hash__()`

Hash based on vertex ID only (not attributes).

**Returns:** int

**Example:**

```python
v1 = Vertex('A', attrs={'x': 1})
v2 = Vertex('A', attrs={'x': 2})

# Same hash (same ID)
hash(v1) == hash(v2)  # True

# Can use in sets/dicts
vertices = {v1, v2}  # Set with one element
```

---

### `__eq__(other)`

Equality based on both ID and attributes.

**Parameters:**

- `other`: Object to compare with

**Returns:** bool

**Example:**

```python
v1 = Vertex('A', attrs={'x': 1})
v2 = Vertex('A', attrs={'x': 1})
v3 = Vertex('A', attrs={'x': 2})
v4 = Vertex('B', attrs={'x': 1})

v1 == v2  # True (same ID and attrs)
v1 == v3  # False (same ID, different attrs)
v1 == v4  # False (different ID)
v1 == 'A'  # False (different type)
```

## Type Annotations

The Vertex class is fully type-annotated:

```python
from typing import Any, Dict

class Vertex:
    id: str
    attrs: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any: ...
    def with_attrs(self, **kwargs) -> 'Vertex': ...
    def without_attrs(self, *keys: str) -> 'Vertex': ...
    def with_id(self, new_id: str) -> 'Vertex': ...
```

## Usage Patterns

### Creating Vertices from Data

```python
# From a list
names = ['Alice', 'Bob', 'Charlie']
vertices = [Vertex(name) for name in names]

# From a dict
data = {
    'Alice': {'age': 30},
    'Bob': {'age': 25}
}
vertices = [
    Vertex(name, attrs=attrs)
    for name, attrs in data.items()
]

# From CSV/database
import csv
with open('data.csv') as f:
    reader = csv.DictReader(f)
    vertices = [
        Vertex(row['id'], attrs=row)
        for row in reader
    ]
```

### Transforming Vertices

```python
# Add computed attribute
v1 = Vertex('A', attrs={'value': 10})
v2 = v1.with_attrs(doubled=v1.get('value') * 2)

# Conditional update
v = Vertex('A', attrs={'score': 85})
if v.get('score') > 80:
    v = v.with_attrs(grade='A')

# Remove sensitive data
v1 = Vertex('User1', attrs={'name': 'Alice', 'password': 'secret'})
v2 = v1.without_attrs('password')
```

### Batch Operations

```python
# Update many vertices
vertices = [Vertex(f'v{i}', attrs={'value': i}) for i in range(10)]

# Apply transformation
updated = [
    v.with_attrs(value=v.get('value') * 2)
    for v in vertices
]

# Filter by attribute
high_value = [
    v for v in vertices
    if v.get('value', 0) > 5
]
```

## See Also

- [Edge API Reference](edge.md)
- [Graph API Reference](graph.md)
- [Working with Vertices Guide](../user-guide/vertices.md)
