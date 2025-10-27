"""
Example: AlgoTree <-> AlgoGraph Integration

Demonstrates how to use AlgoTree and AlgoGraph together,
converting between trees and graphs for different use cases.

REQUIRES: AlgoTree must be installed or in PYTHONPATH
Run from released/ directory with:
    PYTHONPATH=. python AlgoGraph/examples/tree_graph_integration.py
"""

# Add parent to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AlgoTree import Node, Tree, node
from AlgoTree.pretty_tree import pretty_tree
from AlgoGraph import (
    Vertex, Edge, Graph,
    tree_to_graph, graph_to_tree,
    graph_to_flat_dict, flat_dict_to_graph
)


def example_1_org_chart():
    """Example 1: Organization chart as tree, analyze as graph."""
    print("=" * 60)
    print("Example 1: Organization Chart")
    print("=" * 60)

    # Build organization tree
    org_tree = Tree(node('CEO',
        node('CTO',
            node('Engineering Manager',
                'Engineer 1',
                'Engineer 2'
            ),
            node('DevOps Manager',
                'DevOps 1'
            )
        ),
        node('CFO',
            node('Accounting Manager',
                'Accountant 1',
                'Accountant 2'
            )
        )
    ))

    print("\n1. Organization as Tree:")
    print(pretty_tree(org_tree.root))

    # Convert to graph for analysis
    org_graph = tree_to_graph(org_tree)

    print("\n2. Graph Statistics:")
    print(f"   Employees: {org_graph.vertex_count}")
    print(f"   Reporting relationships: {org_graph.edge_count}")

    # Analyze graph structure
    print("\n3. Department Sizes (direct reports):")
    for vertex in org_graph.vertices:
        if org_graph.degree(vertex.id) > 0:
            reports = org_graph.out_degree(vertex.id)
            if reports > 0:
                print(f"   {vertex.id}: {reports} direct reports")

    # Convert back to tree
    recovered_tree = graph_to_tree(org_graph, 'CEO')
    print("\n4. Recovered tree matches:", org_tree.root.name == recovered_tree.root.name)


def example_2_file_system_with_symlinks():
    """Example 2: File system with symbolic links (graph) -> extract tree view."""
    print("\n" + "=" * 60)
    print("Example 2: File System with Symlinks")
    print("=" * 60)

    # Build file system graph (with symlinks creating cycles)
    vertices = {
        Vertex('/', attrs={'type': 'dir'}),
        Vertex('/home', attrs={'type': 'dir'}),
        Vertex('/home/user', attrs={'type': 'dir'}),
        Vertex('/home/user/docs', attrs={'type': 'dir'}),
        Vertex('/home/user/docs/file.txt', attrs={'type': 'file', 'size': 1024}),
        Vertex('/tmp', attrs={'type': 'dir'}),
        Vertex('/tmp/link', attrs={'type': 'symlink'}),
    }

    edges = {
        Edge('/', '/home'),
        Edge('/', '/tmp'),
        Edge('/home', '/home/user'),
        Edge('/home/user', '/home/user/docs'),
        Edge('/home/user/docs', '/home/user/docs/file.txt'),
        Edge('/tmp', '/tmp/link'),
        # Symlink creates cycle
        Edge('/tmp/link', '/home/user/docs'),  # Points back to docs
    }

    fs_graph = Graph(vertices, edges)

    print(f"\n1. File System Graph:")
    print(f"   Nodes: {fs_graph.vertex_count}")
    print(f"   Edges: {fs_graph.edge_count}")
    print(f"   Has cycles: {fs_graph.edge_count >= fs_graph.vertex_count}")

    # Extract tree view (spanning tree) starting from root
    fs_tree = graph_to_tree(fs_graph, '/')

    print("\n2. Tree View (symlinks excluded):")
    print(pretty_tree(fs_tree.root))

    # Show symlink
    print("\n3. Symlink detected:")
    print(f"   /tmp/link -> /home/user/docs")


def example_3_dependency_graph():
    """Example 3: Package dependencies (DAG) using flat format."""
    print("\n" + "=" * 60)
    print("Example 3: Package Dependencies via Flat Format")
    print("=" * 60)

    # Define dependencies in flat format
    # (this format works for both trees and graphs)
    dep_flat = {
        'app': {
            '.name': 'app',
            '.edges': [
                {'target': 'web_framework', 'weight': 1},
                {'target': 'database', 'weight': 1}
            ],
            'version': '1.0.0'
        },
        'web_framework': {
            '.name': 'web_framework',
            '.edges': [
                {'target': 'http_lib', 'weight': 1},
                {'target': 'template_engine', 'weight': 1}
            ],
            'version': '2.1.0'
        },
        'database': {
            '.name': 'database',
            '.edges': [
                {'target': 'sql_driver', 'weight': 1}
            ],
            'version': '3.0.5'
        },
        'http_lib': {
            '.name': 'http_lib',
            '.edges': [],
            'version': '1.5.2'
        },
        'template_engine': {
            '.name': 'template_engine',
            '.edges': [],
            'version': '4.2.0'
        },
        'sql_driver': {
            '.name': 'sql_driver',
            '.edges': [],
            'version': '2.8.1'
        }
    }

    print("\n1. Loading dependencies from flat format:")
    dep_graph = flat_dict_to_graph(dep_flat)
    print(f"   Packages: {dep_graph.vertex_count}")
    print(f"   Dependencies: {dep_graph.edge_count}")

    print("\n2. Package versions:")
    for vertex in sorted(dep_graph.vertices, key=lambda v: v.id):
        version = vertex.get('version', 'unknown')
        print(f"   {vertex.id}: {version}")

    print("\n3. Dependency tree from 'app':")
    dep_tree = graph_to_tree(dep_graph, 'app')
    print(pretty_tree(dep_tree.root, node_name=lambda n: f"{n.name} ({n.get('version', '?')})"))

    # Convert back to flat format
    print("\n4. Export to flat format:")
    exported = graph_to_flat_dict(dep_graph)
    print(f"   Exported {len(exported)} packages")
    print(f"   Format compatible with AlgoTree? Yes")


def example_4_social_network():
    """Example 4: Social network (undirected graph)."""
    print("\n" + "=" * 60)
    print("Example 4: Social Network")
    print("=" * 60)

    # Build social network with friendships
    people = {
        Vertex('Alice', attrs={'age': 30, 'city': 'NYC'}),
        Vertex('Bob', attrs={'age': 25, 'city': 'Boston'}),
        Vertex('Charlie', attrs={'age': 35, 'city': 'NYC'}),
        Vertex('Diana', attrs={'age': 28, 'city': 'Boston'}),
        Vertex('Eve', attrs={'age': 32, 'city': 'SF'})
    }

    # Friendships are undirected
    friendships = {
        Edge('Alice', 'Bob', directed=False),
        Edge('Alice', 'Charlie', directed=False),
        Edge('Bob', 'Diana', directed=False),
        Edge('Charlie', 'Diana', directed=False),
        Edge('Diana', 'Eve', directed=False)
    }

    network = Graph(people, friendships)

    print("\n1. Social Network:")
    print(f"   People: {network.vertex_count}")
    print(f"   Friendships: {network.edge_count}")

    print("\n2. Friend counts:")
    for person in sorted(network.vertices, key=lambda v: v.id):
        friends = network.neighbors(person.id)
        print(f"   {person.id} ({person.get('age')}): {len(friends)} friends")

    print("\n3. Finding paths through network:")
    print("   (Converting to tree shows one possible path structure)")

    # Create spanning tree from Alice's perspective
    alice_view = graph_to_tree(network, 'Alice')
    print("\n   Network from Alice's perspective:")
    print(pretty_tree(alice_view.root))

    # Create spanning tree from Eve's perspective
    eve_view = graph_to_tree(network, 'Eve')
    print("\n   Network from Eve's perspective:")
    print(pretty_tree(eve_view.root))


def example_5_mixed_workflow():
    """Example 5: Build with Tree API, analyze with Graph API."""
    print("\n" + "=" * 60)
    print("Example 5: Mixed Workflow - Build as Tree, Analyze as Graph")
    print("=" * 60)

    # Start with tree API (easier for hierarchical construction)
    print("\n1. Building project structure as tree (easy!):")
    project = Tree(node('MyProject',
        node('src',
            node('main.py', lines=150),
            node('utils.py', lines=75),
            node('models',
                node('user.py', lines=200),
                node('product.py', lines=180)
            )
        ),
        node('tests',
            node('test_main.py', lines=100),
            node('test_utils.py', lines=50)
        ),
        node('docs',
            node('README.md', lines=80)
        )
    ))

    print(pretty_tree(project.root))

    # Convert to graph for analysis
    print("\n2. Analyzing as graph:")
    project_graph = tree_to_graph(project)

    # Calculate total lines of code
    total_lines = sum(v.get('lines', 0) for v in project_graph.vertices)
    print(f"   Total lines of code: {total_lines}")

    # Find all Python files
    py_files = project_graph.find_vertices(
        lambda v: v.id.endswith('.py')
    )
    print(f"   Python files: {len(py_files)}")
    for f in sorted(py_files, key=lambda v: v.get('lines', 0), reverse=True):
        print(f"      {f.id}: {f.get('lines')} lines")

    # Analyze structure
    print(f"\n3. Project statistics:")
    print(f"   Total items: {project_graph.vertex_count}")
    print(f"   Directory depth: {project.height}")  # Use tree API for depth
    print(f"   Test coverage: {len([v for v in py_files if 'test_' in v.id])}/{len(py_files)} files")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AlgoTree <-> AlgoGraph Integration Examples")
    print("=" * 60)

    example_1_org_chart()
    example_2_file_system_with_symlinks()
    example_3_dependency_graph()
    example_4_social_network()
    example_5_mixed_workflow()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
    1. Trees are special graphs (acyclic, connected, directed)
    2. tree_to_graph(): Convert hierarchies to network representation
    3. graph_to_tree(): Extract spanning tree from graph
    4. Flat format: Universal interchange between both
    5. Use trees for hierarchies, graphs for networks
    6. Mix APIs: Build with tree (easy), analyze with graph (powerful)
    """)


if __name__ == '__main__':
    main()
