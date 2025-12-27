"""
sl_package: A comprehensive game theory library for Python.

This package provides tools for modeling and solving game-theoretic problems,
including both simultaneous and sequential games.

Main Components:
    - Node: Represents a state in the game tree
    - Game: Represents a complete game with players, actions, and payoffs
    - Solver: Abstract base class for equilibrium solvers
    - PureStrategyNashSolver: Finds pure strategy Nash equilibria
    - MixedStrategySolver: Computes mixed strategy Nash equilibria
    - BackwardInductionSolver: Solves sequential games via backward induction

Example Usage:
    >>> from sl_package import Game, PureStrategyNashSolver
    >>>
    >>> # Create a simple game
    >>> game = Game()
    >>> game.add_moves('Player1', ['Cooperate', 'Defect'])
    >>> game.add_moves('Player2', ['Cooperate', 'Defect'])
    >>> game.add_outcomes([
    ...     (3, 3),  # Both cooperate
    ...     (0, 5),  # P1 cooperates, P2 defects
    ...     (5, 0),  # P1 defects, P2 cooperates
    ...     (1, 1)   # Both defect
    ... ])
    >>>
    >>> # Find Nash equilibria
    >>> solver = PureStrategyNashSolver(game)
    >>> equilibria = solver.solve()
    >>> print(equilibria)

For more information, see the documentation at:
https://github.com/yourusername/sl-package
"""

__version__ = '0.1.0'

from .game import Game
from .game_node import Node
from .solvers.backward_induction import BackwardInductionSolver
from .solvers.mixed_strategy import MixedStrategySolver
from .solvers.pure_strategy import PureStrategyNashSolver
from .solvers.solver import Solver

__all__ = [
    'Node',
    'Game',
    'MixedStrategySolver',
    'PureStrategyNashSolver',
    'BackwardInductionSolver',
    'Solver',
]
