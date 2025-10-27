"""
Demo of AlgoGraph Shell capabilities.

This script demonstrates how to use the shell programmatically
and shows example interactions.
"""

from AlgoGraph import Graph, Vertex, Edge
from AlgoGraph.shell import GraphContext, GraphShell


def create_social_network():
    """Create a sample social network graph."""
    vertices = {
        Vertex('Alice', attrs={'age': 30, 'city': 'NYC'}),
        Vertex('Bob', attrs={'age': 25, 'city': 'Boston'}),
        Vertex('Charlie', attrs={'age': 35, 'city': 'NYC'}),
        Vertex('Diana', attrs={'age': 28, 'city': 'Seattle'}),
        Vertex('Eve', attrs={'age': 32, 'city': 'Boston'}),
    }

    edges = {
        Edge('Alice', 'Bob', directed=False),
        Edge('Alice', 'Charlie', directed=False),
        Edge('Bob', 'Diana', directed=False),
        Edge('Charlie', 'Diana', directed=False),
        Edge('Diana', 'Eve', directed=False),
    }

    return Graph(vertices, edges)


def demo_navigation():
    """Demo navigation commands."""
    print("=" * 60)
    print("Demo 1: Navigation")
    print("=" * 60)

    graph = create_social_network()
    ctx = GraphContext(graph)
    shell = GraphShell(ctx)

    commands = [
        ("ls", "List all vertices"),
        ("cd Alice", "Navigate to Alice"),
        ("pwd", "Show current location"),
        ("ls", "Show Alice's attributes and neighbors"),
        ("cd neighbors", "Enter neighbors mode"),
        ("ls", "Show Alice's neighbors"),
        ("cd Bob", "Navigate to Bob"),
        ("pwd", "Show we're now at Bob"),
        ("cd ..", "Go back to root"),
    ]

    for cmd, description in commands:
        print(f"\n>>> {cmd}  # {description}")
        shell.execute_command(cmd)
        print()


def demo_queries():
    """Demo graph query commands."""
    print("=" * 60)
    print("Demo 2: Graph Queries")
    print("=" * 60)

    graph = create_social_network()
    ctx = GraphContext(graph)
    shell = GraphShell(ctx)

    commands = [
        ("find Alice", "Find Alice in the graph"),
        ("info", "Show graph statistics"),
        ("cd Alice", "Navigate to Alice"),
        ("info", "Show Alice's details"),
        ("neighbors", "Show Alice's neighbors"),
        ("path Alice Eve", "Find path from Alice to Eve"),
        ("shortest Alice Eve", "Find shortest path"),
        ("components", "Show connected components"),
        ("bfs Alice", "Breadth-first search from Alice"),
    ]

    for cmd, description in commands:
        print(f"\n>>> {cmd}  # {description}")
        shell.execute_command(cmd)
        print()


def demo_weighted_graph():
    """Demo with a weighted road network."""
    print("=" * 60)
    print("Demo 3: Weighted Graph (Road Network)")
    print("=" * 60)

    # Create road network
    cities = {
        Vertex('NYC', attrs={'population': 8000000}),
        Vertex('Boston', attrs={'population': 700000}),
        Vertex('Philly', attrs={'population': 1600000}),
        Vertex('DC', attrs={'population': 700000}),
    }

    roads = {
        Edge('NYC', 'Boston', weight=215.0, directed=False),
        Edge('NYC', 'Philly', weight=95.0, directed=False),
        Edge('NYC', 'DC', weight=225.0, directed=False),
        Edge('Philly', 'DC', weight=140.0, directed=False),
        Edge('Boston', 'Philly', weight=310.0, directed=False),
    }

    graph = Graph(cities, roads)
    ctx = GraphContext(graph)
    shell = GraphShell(ctx)

    commands = [
        ("ls", "List all cities"),
        ("cd NYC", "Go to NYC"),
        ("ls", "Show NYC details"),
        ("cd neighbors", "View roads from NYC"),
        ("ls", "Show connected cities with distances"),
        ("cd /", "Back to root"),
        ("shortest NYC DC", "Find shortest route from NYC to DC"),
        ("path NYC Boston", "Find any route from NYC to Boston"),
    ]

    for cmd, description in commands:
        print(f"\n>>> {cmd}  # {description}")
        shell.execute_command(cmd)
        print()


if __name__ == '__main__':
    demo_navigation()
    print("\n" * 2)
    demo_queries()
    print("\n" * 2)
    demo_weighted_graph()

    print("\n" * 2)
    print("=" * 60)
    print("Try it yourself!")
    print("=" * 60)
    print()
    print("Run: python -m AlgoGraph.shell.shell")
    print("Or:  PYTHONPATH=. python AlgoGraph/shell/shell.py")
