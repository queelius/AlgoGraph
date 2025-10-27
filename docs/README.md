# AlgoGraph Documentation

This directory contains the source files for AlgoGraph's documentation, built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## Building the Documentation

### Prerequisites

Install MkDocs and required plugins:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

### Local Development

To build and serve the documentation locally:

```bash
# From the AlgoGraph root directory
cd /path/to/AlgoGraph

# Serve docs locally (auto-reloads on changes)
mkdocs serve

# Open in browser: http://127.0.0.1:8000
```

### Build Static Site

To build the documentation as a static website:

```bash
# Build to site/ directory
mkdocs build

# Build with strict mode (warnings become errors)
mkdocs build --strict
```

The generated HTML will be in the `site/` directory.

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md         # Installation guide
│   ├── quickstart.md           # Quick start tutorial
│   └── examples.md             # Basic examples
├── user-guide/
│   ├── core-concepts.md        # Core concepts and design
│   ├── vertices.md             # Working with vertices
│   ├── edges.md                # Working with edges
│   ├── graphs.md               # Building graphs
│   ├── algorithms.md           # Using algorithms
│   ├── serialization.md        # JSON I/O
│   └── algotree-integration.md # AlgoTree interop
├── shell/
│   ├── overview.md             # Shell introduction
│   ├── navigation.md           # Navigation commands
│   ├── queries.md              # Query commands
│   └── advanced.md             # Advanced features
├── api/
│   ├── vertex.md               # Vertex API
│   ├── edge.md                 # Edge API
│   ├── graph.md                # Graph API
│   ├── algorithms/
│   │   ├── traversal.md        # Traversal algorithms
│   │   ├── shortest_path.md    # Shortest path algorithms
│   │   ├── connectivity.md     # Connectivity algorithms
│   │   └── spanning_tree.md    # Spanning tree algorithms
│   ├── serialization.md        # Serialization API
│   ├── interop.md              # Interop API
│   └── shell.md                # Shell API
├── examples/
│   ├── social-networks.md      # Social network examples
│   ├── road-networks.md        # Road network examples
│   ├── dependency-graphs.md    # Dependency graph examples
│   └── use-cases.md            # Real-world use cases
├── design/
│   ├── immutability.md         # Immutability philosophy
│   ├── composability.md        # Composability
│   └── separation.md           # Separation of concerns
└── contributing.md             # Contribution guidelines
```

## Writing Documentation

### Style Guide

1. **Use clear, concise language**
   - Write for beginners and experts
   - Define technical terms on first use
   - Use active voice

2. **Include code examples**
   ```python
   # Good - shows actual usage
   from AlgoGraph import Vertex
   v = Vertex('A', attrs={'value': 10})
   ```

3. **Use admonitions for important notes**
   ```markdown
   !!! note
       This is an important note

   !!! warning
       This is a warning

   !!! tip
       This is a helpful tip
   ```

4. **Link to related content**
   ```markdown
   See [Core Concepts](../user-guide/core-concepts.md) for more information.
   ```

### Code Blocks

Use syntax highlighting:

````markdown
```python
from AlgoGraph import Graph

g = Graph()
```
````

### Admonitions

Available admonition types:

```markdown
!!! note
    Regular note

!!! tip
    Helpful tip

!!! warning
    Important warning

!!! danger
    Critical warning

!!! example
    Example usage

!!! question
    Question or FAQ
```

## Configuration

Documentation configuration is in `mkdocs.yml` in the project root:

```yaml
site_name: AlgoGraph Documentation
theme:
  name: material
  # ... theme configuration

nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      # ... more pages
```

### Adding New Pages

1. Create the markdown file in appropriate directory
2. Add to `nav` section in `mkdocs.yml`
3. Link from related pages

## Deployment

### GitHub Pages

To deploy to GitHub Pages:

```bash
# Build and deploy
mkdocs gh-deploy

# This builds the site and pushes to gh-pages branch
```

### Manual Deployment

1. Build the site: `mkdocs build`
2. Upload `site/` directory to web server

## Testing Documentation

Before committing documentation changes:

1. **Build locally and verify**
   ```bash
   mkdocs serve
   # Check all pages render correctly
   ```

2. **Test code examples**
   ```bash
   # Extract and run code examples
   python -m doctest docs/examples/*.md
   ```

3. **Check for broken links**
   ```bash
   mkdocs build --strict
   # Fails on warnings (including broken links)
   ```

4. **Verify navigation**
   - All pages reachable from home
   - Breadcrumbs work correctly
   - Search finds relevant content

## Common Tasks

### Adding a New Algorithm

1. Document in user guide: `docs/user-guide/algorithms.md`
2. Add API reference: `docs/api/algorithms/<category>.md`
3. Add example: `docs/examples/`
4. Update navigation in `mkdocs.yml`

### Adding a New Example

1. Create markdown file: `docs/examples/new-example.md`
2. Add to navigation in `mkdocs.yml`
3. Link from related pages

### Updating API Reference

1. Update docstrings in source code
2. Update corresponding API doc page
3. Add examples if needed

## Markdown Extensions

Available Markdown extensions (configured in `mkdocs.yml`):

- **Admonitions**: Note/warning/tip boxes
- **Code highlighting**: Syntax highlighted code blocks
- **Tables**: Markdown tables
- **Task lists**: Checkbox lists
- **Tabs**: Tabbed content blocks
- **Snippets**: Include file contents
- **Superfences**: Advanced code fences

See [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/) for details.

## Troubleshooting

### MkDocs Not Found

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

### Theme Not Loading

Ensure Material theme is installed:

```bash
pip install mkdocs-material
```

### Code Examples Not Rendering

Check that code blocks use triple backticks and language identifier:

````markdown
```python
# Your code here
```
````

### Navigation Not Working

Verify `mkdocs.yml` navigation structure matches file paths.

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Python Markdown Extensions](https://python-markdown.github.io/extensions/)

## Questions?

For documentation-related questions:

1. Check this README
2. See [Contributing Guide](contributing.md)
3. Open an issue on GitHub
