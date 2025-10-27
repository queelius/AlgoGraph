#!/usr/bin/env python3
"""
Manual test script to demonstrate shell improvements.
"""

import sys
import os
import tempfile

# Add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlgoGraph import Graph, Vertex, Edge, save_graph, load_graph
from AlgoGraph.shell import GraphContext, GraphShell

def test_serialization():
    """Test 1: File I/O for graphs"""
    print("=" * 60)
    print("TEST 1: Graph Serialization (Save/Load)")
    print("=" * 60)

    # Create a graph with spaces in vertex names
    vertices = {
        Vertex('Alice Smith', attrs={'age': 30, 'city': 'NYC'}),
        Vertex('Bob Jones', attrs={'age': 25, 'city': 'Boston'}),
        Vertex('Charlie Brown', attrs={'age': 35, 'city': 'Seattle'}),
    }

    edges = {
        Edge('Alice Smith', 'Bob Jones', directed=False, weight=5.0),
        Edge('Bob Jones', 'Charlie Brown', directed=False, weight=3.0),
    }

    graph = Graph(vertices, edges)

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filename = f.name

    print(f"Original graph: {graph.vertex_count} vertices, {graph.edge_count} edges")
    print(f"Saving to {filename}...")
    save_graph(graph, filename)
    print("✓ Graph saved successfully")

    # Load it back
    print(f"Loading from {filename}...")
    loaded = load_graph(filename)
    print(f"Loaded graph: {loaded.vertex_count} vertices, {loaded.edge_count} edges")

    # Verify
    assert loaded.vertex_count == graph.vertex_count
    assert loaded.edge_count == graph.edge_count

    # Check a vertex with attributes
    alice = loaded.get_vertex('Alice Smith')
    assert alice.attrs['age'] == 30
    print(f"✓ Verified vertex 'Alice Smith' with attrs: {alice.attrs}")

    # Check an edge with weight
    edge = loaded.get_edge('Alice Smith', 'Bob Jones')
    assert edge.weight == 5.0
    print(f"✓ Verified edge 'Alice Smith' <-> 'Bob Jones' with weight: {edge.weight}")

    print("✓ All serialization tests passed!\n")

    # Cleanup
    os.unlink(filename)

    return graph

def test_command_parsing():
    """Test 2: Command parsing with spaces"""
    print("=" * 60)
    print("TEST 2: Command Parsing (Spaces in Vertex Names)")
    print("=" * 60)

    # Create graph with spaces
    vertices = {
        Vertex('Alice Smith'),
        Vertex('Bob Jones'),
    }
    edges = {Edge('Alice Smith', 'Bob Jones', directed=False)}
    graph = Graph(vertices, edges)

    context = GraphContext(graph)
    shell = GraphShell(context)

    # Test navigation with quoted names
    print("Testing: cd 'Alice Smith' (should work with shlex)")
    result = shell.execute_command("cd 'Alice Smith'")
    assert result  # Should continue
    assert shell.context.current_vertex == 'Alice Smith'
    print(f"✓ Successfully navigated to: {shell.context.current_vertex}")

    print("Testing: cd 'Bob Jones'")
    result = shell.execute_command("cd 'Bob Jones'")
    assert shell.context.current_vertex == 'Bob Jones'
    print(f"✓ Successfully navigated to: {shell.context.current_vertex}")

    print("✓ Command parsing with spaces works!\n")

def test_absolute_paths():
    """Test 3: Absolute path navigation"""
    print("=" * 60)
    print("TEST 3: Absolute Path Navigation")
    print("=" * 60)

    vertices = {Vertex('A'), Vertex('B'), Vertex('C'), Vertex('D')}
    edges = {
        Edge('A', 'B', directed=False),
        Edge('B', 'C', directed=False),
        Edge('C', 'D', directed=False),
    }
    graph = Graph(vertices, edges)

    context = GraphContext(graph)
    shell = GraphShell(context)

    # Start at A
    shell.execute_command("cd A")
    print(f"Started at: {shell.context.get_path()}")

    # Jump to D with absolute path
    print("Testing: cd /D (absolute path)")
    shell.execute_command("cd /D")
    assert shell.context.current_vertex == 'D'
    print(f"✓ Jumped to: {shell.context.get_path()}")

    # Jump back to A from D
    print("Testing: cd /A (absolute path from D)")
    shell.execute_command("cd /A")
    assert shell.context.current_vertex == 'A'
    print(f"✓ Jumped to: {shell.context.get_path()}")

    # Go to neighbors mode, then jump with absolute path
    shell.execute_command("cd neighbors")
    print(f"Entered neighbors mode: {shell.context.get_path()}")

    print("Testing: cd /C (absolute path from neighbors mode)")
    shell.execute_command("cd /C")
    assert shell.context.current_vertex == 'C'
    print(f"✓ Jumped to: {shell.context.get_path()}")

    print("✓ Absolute paths work from any location!\n")

def test_save_command():
    """Test 4: Save command in shell"""
    print("=" * 60)
    print("TEST 4: Save Command in Shell")
    print("=" * 60)

    vertices = {Vertex('X'), Vertex('Y')}
    edges = {Edge('X', 'Y', weight=10.0)}
    graph = Graph(vertices, edges)

    context = GraphContext(graph)
    shell = GraphShell(context)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filename = f.name

    print(f"Testing: save {filename}")
    shell.execute_command(f"save {filename}")

    assert os.path.exists(filename)
    print(f"✓ File created: {filename}")

    # Verify we can load it
    loaded = load_graph(filename)
    assert loaded.vertex_count == 2
    assert loaded.edge_count == 1
    print("✓ Saved graph can be loaded successfully")

    # Cleanup
    os.unlink(filename)
    print("✓ Save command works!\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AlgoGraph Shell Improvements - Manual Testing")
    print("=" * 60 + "\n")

    try:
        # Run tests
        graph = test_serialization()
        test_command_parsing()
        test_absolute_paths()
        test_save_command()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNew features available:")
        print("  1. ✓ File I/O: Load/save graphs with JSON")
        print("  2. ✓ Quoted names: Use 'Alice Smith' for vertex names with spaces")
        print("  3. ✓ Absolute paths: Use cd /vertex to jump anywhere")
        print("  4. ✓ Tab completion: Press TAB to complete commands and vertex names")
        print("  5. ✓ Command history: Use UP/DOWN arrows to recall commands")
        print("  6. ✓ Save command: Use 'save filename.json' to save current graph")
        print("\nTo try the interactive shell:")
        print("  python -m AlgoGraph.shell.shell [graph_file.json]")
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
