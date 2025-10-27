# Installation

This guide will help you install and configure AlgoGraph on your system.

## Requirements

- Python 3.8 or higher
- (Optional) AlgoTree library for tree-graph interoperability

## Installation Steps

### Step 1: Clone the Repository

Currently, AlgoGraph is available from the source repository:

```bash
git clone https://github.com/released/AlgoGraph.git
cd AlgoGraph
```

### Step 2: Set Up PYTHONPATH

AlgoGraph requires you to configure your `PYTHONPATH` to include the parent directory of the repository.

#### On Linux/macOS

Add the following to your `~/.bashrc` or `~/.zshrc`:

```bash
export PYTHONPATH=/path/to/released:$PYTHONPATH
```

For example, if you cloned the repository to `/home/user/github/released/AlgoGraph`:

```bash
export PYTHONPATH=/home/user/github/released:$PYTHONPATH
```

Then reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

#### On Windows

**Using Command Prompt:**

```cmd
set PYTHONPATH=C:\path\to\released;%PYTHONPATH%
```

**Using PowerShell:**

```powershell
$env:PYTHONPATH = "C:\path\to\released;$env:PYTHONPATH"
```

**Permanent Configuration:**

1. Right-click on "This PC" or "My Computer"
2. Select "Properties" → "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `PYTHONPATH`
6. Variable value: `C:\path\to\released`

### Step 3: Verify Installation

Test that AlgoGraph is correctly installed:

```python
python3 -c "from AlgoGraph import Vertex, Edge, Graph; print('AlgoGraph installed successfully!')"
```

You should see:
```
AlgoGraph installed successfully!
```

## Optional: Install AlgoTree

For tree-graph interoperability features, install AlgoTree:

```bash
# Clone AlgoTree (if in same parent directory)
cd /path/to/released
git clone https://github.com/released/AlgoTree.git
```

Update your `PYTHONPATH` as described above to include both libraries.

Verify AlgoTree integration:

```python
python3 -c "from AlgoGraph.interop import tree_to_graph; print('AlgoTree integration available!')"
```

!!! note "AlgoTree is Optional"
    AlgoTree integration is completely optional. All core AlgoGraph features work without it. You only need AlgoTree if you want to convert between tree and graph representations.

## Interactive Shell

To use the interactive graph shell:

```bash
cd /path/to/released/AlgoGraph
python3 -m AlgoGraph.shell.shell
```

Or load a specific graph file:

```bash
python3 -m AlgoGraph.shell.shell path/to/graph.json
```

## Development Installation

If you plan to contribute to AlgoGraph or run tests:

```bash
# Install development dependencies
pip install pytest pytest-benchmark

# Run tests
cd /path/to/released/AlgoGraph
pytest

# Run with coverage
pytest --cov=AlgoGraph

# Run benchmarks
pytest test/ --benchmark-only
```

## Troubleshooting

### Import Error: No module named 'AlgoGraph'

**Problem**: Python cannot find the AlgoGraph module.

**Solution**: Verify your `PYTHONPATH` is set correctly:

```bash
echo $PYTHONPATH  # Linux/macOS
echo %PYTHONPATH%  # Windows CMD
$env:PYTHONPATH    # Windows PowerShell
```

Make sure it includes the parent directory of AlgoGraph (the `released` directory, not `AlgoGraph` itself).

### Import Error: No module named 'AlgoTree'

**Problem**: AlgoTree features are being used but AlgoTree is not installed.

**Solution**: Either install AlgoTree (see above) or avoid using interop functions like `tree_to_graph` and `graph_to_tree`.

### Shell Commands Don't Have Tab Completion

**Problem**: Tab completion doesn't work in the interactive shell.

**Solution**: Install readline support:

```bash
# Linux/macOS (usually included)
pip install readline

# Windows
pip install pyreadline3
```

## Next Steps

Now that you have AlgoGraph installed, proceed to:

- [Quick Start Guide](quickstart.md) - Learn the basics
- [Basic Examples](examples.md) - See common use cases
- [Core Concepts](../user-guide/core-concepts.md) - Understand the fundamentals
