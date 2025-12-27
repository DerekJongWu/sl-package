from typing import Optional, Set, Dict, Tuple, Union

class Node:
    """Represents a game state, supporting sequential moves.

    Attributes:
        players: Set of player names active at this node
        actions: Dictionary mapping action names to child nodes
        payoff: Tuple of payoffs for terminal nodes, None otherwise
    """
    def __init__(self, players: Optional[Set[str]] = None) -> None:
        self.players: Set[str] = players if players else set()
        self.actions: Dict[str, 'Node'] = {}  # Maps action names to child nodes
        self.payoff: Optional[Tuple[Union[int, float], ...]] = None  # Stores outcome if terminal

    def add_action(self, action: str, child_node: 'Node') -> None:
        """Adds an action leading to a child node.

        Args:
            action: Name of the action
            child_node: The node reached by taking this action
        """
        self.actions[action] = child_node

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        players_str = f"players={self.players}" if self.players else "no players"
        actions_str = f"{len(self.actions)} actions" if self.actions else "terminal"
        payoff_str = f"payoff={self.payoff}" if self.payoff is not None else "no payoff"
        return f"Node({players_str}, {actions_str}, {payoff_str})"

    def is_terminal(self) -> bool:
        """Check if this is a terminal node.

        Returns:
            True if the node has no actions (is a leaf node)
        """
        return len(self.actions) == 0
