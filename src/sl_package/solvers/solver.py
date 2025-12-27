from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List
from ..game import Game


class Solver(ABC):
    """Abstract base class for game theory solvers.

    All solver implementations must inherit from this class and implement
    the solve() method.

    Attributes:
        game: The game instance to solve
        equilibrium: The computed equilibrium (format depends on solver type)
    """

    def __init__(self, game: Game) -> None:
        """Initialize with a game instance.

        Args:
            game: The game instance to solve

        Raises:
            TypeError: If game is not an instance of Game
        """
        if not isinstance(game, Game):
            raise TypeError("Solver expects an instance of Game.")
        self.game: Game = game
        self.equilibrium: Union[Dict, List] = {}

    @abstractmethod
    def solve(self) -> Any:
        """Solve the game and return the equilibrium.

        This method must be implemented by all subclasses.

        Returns:
            The computed equilibrium (format depends on solver type)

        Raises:
            NotImplementedError: If not implemented in subclass
        """
        raise NotImplementedError("Solve method must be implemented in subclasses.")

    def get_equilibrium(self) -> Union[Dict, List]:
        """Return the computed equilibrium.

        Returns:
            The computed equilibrium, or empty dict if not yet computed
        """
        return self.equilibrium
    
