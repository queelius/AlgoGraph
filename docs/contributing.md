# Contributing to AlgoGraph

Thank you for your interest in contributing to AlgoGraph! This guide will help you get started.

## Getting Started

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/released/AlgoGraph.git
cd AlgoGraph

# Set up PYTHONPATH
export PYTHONPATH=/path/to/released:$PYTHONPATH

# Install development dependencies
pip install pytest pytest-cov pytest-benchmark
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=AlgoGraph

# Run only unit tests
pytest test/

# Run benchmarks
pytest test/ --benchmark-only
```

### 3. Make Your Changes

Follow these guidelines:

- **Maintain immutability**: All data structures should remain immutable
- **Add tests**: Include tests for new features or bug fixes
- **Document code**: Add docstrings and type hints
- **Follow style**: Use consistent style with existing code

## Code Style

### Python Style

AlgoGraph follows PEP 8 with these conventions:

```python
# Use type hints
def dijkstra(graph: Graph, source: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Dijkstra's shortest path algorithm.

    Args:
        graph: Graph with non-negative edge weights
        source: Source vertex ID

    Returns:
        Tuple of (distances, predecessors)
    """
    ...

# Use descriptive variable names
distances = {}  # Good
d = {}  # Avoid

# Prefer comprehensions where appropriate
vertices = {Vertex(name) for name in names}
```

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """
    Short one-line description.

    Longer description with more details if needed.
    Can span multiple lines.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Example:
        >>> example_function(42, "test")
        True
    """
    ...
```

## Testing Guidelines

### Write Comprehensive Tests

```python
def test_vertex_creation():
    """Test basic vertex creation."""
    v = Vertex('A', attrs={'x': 10})

    assert v.id == 'A'
    assert v.get('x') == 10
    assert v.get('y') is None

def test_vertex_immutability():
    """Test that vertices are immutable."""
    v1 = Vertex('A', attrs={'x': 10})
    v2 = v1.with_attrs(y=20)

    # Original unchanged
    assert v1.get('y') is None
    assert v2.get('y') == 20
    assert v2.get('x') == 10
```

### Test Edge Cases

```python
def test_empty_graph():
    """Test operations on empty graph."""
    g = Graph()

    assert g.vertex_count == 0
    assert g.edge_count == 0
    assert not g.has_vertex('A')

def test_single_vertex():
    """Test graph with single vertex."""
    g = Graph({Vertex('A')})

    assert g.vertex_count == 1
    assert g.neighbors('A') == set()
```

### Test Performance

Add benchmarks for critical operations:

```python
def test_large_graph_creation(benchmark):
    """Benchmark large graph creation."""
    def create_large_graph():
        vertices = {Vertex(str(i)) for i in range(1000)}
        edges = {Edge(str(i), str(i+1)) for i in range(999)}
        return Graph(vertices, edges)

    graph = benchmark(create_large_graph)
    assert graph.vertex_count == 1000
```

## Adding New Features

### 1. New Algorithm

When adding a new algorithm:

```python
# In AlgoGraph/algorithms/category.py

def new_algorithm(graph: Graph, param: str) -> Result:
    """
    Brief description of what the algorithm does.

    Time Complexity: O(...)
    Space Complexity: O(...)

    Args:
        graph: Input graph
        param: Parameter description

    Returns:
        Description of return value

    Example:
        >>> from AlgoGraph import Graph, Vertex, Edge
        >>> g = Graph({Vertex('A'), Vertex('B')}, {Edge('A', 'B')})
        >>> result = new_algorithm(g, 'A')
        >>> print(result)
        ...
    """
    # Implementation
    ...
```

Don't forget to:

1. Add to `__all__` in the module
2. Export from `algorithms/__init__.py`
3. Add tests
4. Update documentation

### 2. New Data Structure Feature

When adding methods to existing classes:

```python
class Graph:
    def new_method(self, param: ParamType) -> ReturnType:
        """
        Description of what the method does.

        Args:
            param: Parameter description

        Returns:
            Description of return value

        Example:
            >>> g = Graph(...)
            >>> result = g.new_method(param)
            >>> print(result)
            ...
        """
        # Maintain immutability!
        new_vertices = ...
        new_edges = ...
        return Graph(new_vertices, new_edges)
```

### 3. New Shell Command

When adding shell commands:

```python
# In AlgoGraph/shell/commands.py

class NewCommand(Command):
    """Brief description of command."""

    def execute(self, context: GraphContext, args: List[str]) -> CommandResult:
        # Validate args
        if not args:
            return CommandResult(
                success=False,
                output="",
                context=context,
                error="Usage: newcommand <arg>"
            )

        # Implement logic
        result = do_something(context, args[0])

        return CommandResult(
            success=True,
            output=result,
            context=context
        )
```

Then register in `shell.py`:

```python
self.commands = {
    ...
    'newcommand': NewCommand,
}
```

## Documentation

### Update Documentation

When adding features, update:

1. **Docstrings**: In the code itself
2. **User Guide**: In `docs/user-guide/`
3. **API Reference**: In `docs/api/`
4. **Examples**: Add examples to `docs/examples/`

### Build Documentation Locally

```bash
# Install MkDocs
pip install mkdocs mkdocs-material mkdocstrings

# Build docs
cd /path/to/AlgoGraph
mkdocs serve

# Open http://127.0.0.1:8000 in browser
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Test Everything

```bash
# Run tests
pytest

# Check coverage
pytest --cov=AlgoGraph --cov-report=html

# Run benchmarks
pytest --benchmark-only
```

### 4. Commit Changes

```bash
git add .
git commit -m "Add feature: description"
```

Use clear commit messages:

- `Add feature: description` - New feature
- `Fix bug: description` - Bug fix
- `Update docs: description` - Documentation
- `Refactor: description` - Code refactoring
- `Test: description` - Test additions

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a pull request on GitHub.

## Code Review

Pull requests will be reviewed for:

- **Correctness**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it documented?
- **Style**: Does it follow conventions?
- **Immutability**: Are data structures immutable?
- **Performance**: Is it reasonably efficient?

## Areas for Contribution

### High Priority

- Additional graph algorithms
- Performance optimizations
- Documentation improvements
- Bug fixes

### Medium Priority

- More examples and tutorials
- Shell command improvements
- Visualization tools
- Import/export formats

### Ideas Welcome

- Integration with other libraries
- Benchmarking suite
- Additional data structures
- Your ideas!

## Questions?

- Open an issue on GitHub
- Check existing documentation
- Look at similar code in the project

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Thank You!

Your contributions make AlgoGraph better for everyone. Thank you for taking the time to contribute!
