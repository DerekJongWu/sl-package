from typing import List, Dict, Tuple, Union, Sequence
from .game_node import Node

class Game:
    """Represents a game theory structure with players, actions, and payoffs.

    Attributes:
        root: The root node of the game tree
        current_nodes: List of current leaf nodes for expansion
        players: Ordered list of player names
        player_indices: Mapping from player names to their indices in payoff tuples
    """
    def __init__(self) -> None:
        self.root: Node = Node()
        self.current_nodes: List[Node] = [self.root]  # Track leaf nodes for expansion
        self.players: List[str] = []  # List to track players in the order they're added
        self.player_indices: Dict[str, int] = {}  # Maps player names to their indices

    def add_player(self, player: str) -> int:
        """Add a player to the game if not already present.

        Args:
            player: Name of the player to add

        Returns:
            The index of the player in payoff tuples
        """
        if player not in self.player_indices:
            self.players.append(player)
            self.player_indices[player] = len(self.players) - 1
        return self.player_indices[player]

    def get_player_index(self, player: str) -> int:
        """Return the index of the player in payoff tuples.

        Args:
            player: Name of the player

        Returns:
            The index of the player

        Raises:
            ValueError: If player is not found in the game
        """
        if player not in self.player_indices:
            raise ValueError(f"Player {player} not found in game")
        return self.player_indices[player]

    def add_moves(self, player: str, actions: Sequence[str]) -> None:
        """Adds moves for a player at all current leaf nodes.

        Args:
            player: Name of the player making the moves
            actions: Sequence of action names available to the player
        """
        # Add player to the tracking system if not already added
        self.add_player(player)

        new_nodes = []
        for node in self.current_nodes:
            node.players.add(player)
            for action in actions:
                child_node = Node()
                node.add_action(action, child_node)
                new_nodes.append(child_node)
        self.current_nodes = new_nodes

    def add_outcomes(self, outcomes: Sequence[Tuple[Union[int, float], ...]]) -> None:
        """Assigns payoffs to the current leaf nodes.

        Args:
            outcomes: Sequence of payoff tuples, one per terminal node

        Raises:
            ValueError: If number of outcomes doesn't match number of terminal nodes
        """
        if len(outcomes) != len(self.current_nodes):
            raise ValueError("Number of outcomes must match the number of terminal nodes.")
        for node, payoff in zip(self.current_nodes, outcomes):
            node.payoff = payoff

    def get_terminal_nodes(self) -> List[Node]:
        """Get all terminal nodes in the game tree.

        Returns:
            List of all terminal nodes (nodes with no actions)
        """
        terminal_nodes = []

        def traverse(node: Node) -> None:
            if not node.actions:
                terminal_nodes.append(node)
            else:
                for child in node.actions.values():
                    traverse(child)

        traverse(self.root)
        return terminal_nodes

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the game tree.

        Returns:
            List of all nodes in the tree
        """
        all_nodes = []

        def traverse(node: Node) -> None:
            all_nodes.append(node)
            for child in node.actions.values():
                traverse(child)

        traverse(self.root)
        return all_nodes

    def get_player_actions(self, player: str) -> List[str]:
        """Get all unique actions available to a player across the entire game tree.

        Args:
            player: Name of the player

        Returns:
            List of all unique action names available to the player
        """
        actions = set()

        def traverse(node: Node) -> None:
            if player in node.players:
                actions.update(node.actions.keys())
            for child in node.actions.values():
                traverse(child)

        traverse(self.root)
        return list(actions)

    def __repr__(self) -> str:
        """Return a string representation of the game."""
        num_players = len(self.players)
        num_terminal = len(self.get_terminal_nodes())
        num_nodes = len(self.get_all_nodes())
        return (f"Game(players={num_players}, "
                f"nodes={num_nodes}, "
                f"terminal_nodes={num_terminal})")

    def display_tree(self):
        """Recursively prints the game tree."""
        def recurse(node, depth=0):
            payoff_text = f", Payoff: {node.payoff}" if node.payoff is not None else ""
            print("  " * depth + f"Players: {node.players}{payoff_text}")
            for action, child in node.actions.items():
                print("  " * depth + f"Action: {action}")
                recurse(child, depth + 1)
        recurse(self.root)
    
    def visualize_tree(self):
        """Visualizes the game tree with improved spacing using Graphviz."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with 'pip install sl_package[viz]'")
        try:
            import networkx as nx
            from networkx.drawing.nx_agraph import graphviz_layout
        except ImportError:
            raise ImportError("networkx is required for visualization. Install with 'pip install sl_package[viz]'")
        graph = nx.DiGraph()
        node_labels = {}
        
        def add_edges(node, parent=None, action_label=None):
            """Recursively add nodes and edges to the graph."""
            node_id = id(node)  # Unique identifier
            label = f"{', '.join(node.players)}" if node.actions else f"Payoff: {node.payoff}"
            node_labels[node_id] = label

            if parent is not None:
                graph.add_edge(parent, node_id, action=action_label)

            for action, child in node.actions.items():
                add_edges(child, node_id, action)

        add_edges(self.root)

        # Use Graphviz DOT layout for better hierarchy
        pos = graphviz_layout(graph, prog="dot")

        # Draw graph
        plt.figure(figsize=(10, 6))
        nx.draw(graph, pos, with_labels=True, labels=node_labels, node_color="lightblue", edge_color="black", 
                node_size=3000, font_size=8, font_weight="bold", arrowsize=15)

        # Add action labels on edges
        edge_labels = {(u, v): data["action"] for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, 
                                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

        plt.title("Game Tree Visualization")
        plt.show()