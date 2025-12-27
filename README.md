# SL Package

A Python package for game theory analysis with various solvers including Nash equilibrium, backward induction, and mixed strategy solvers.

## Installation

You can install the package using pip:

```bash
pip install sl-package
```

Or install from source:

```bash
git clone https://github.com/yourusername/sl-package.git
cd sl-package
pip install -e .
```

## Usage

```python
from sl_package import Game, Node, MixedStrategySolver, PureStrategyNashSolver, BackwardInductionSolver

# Create a game
game = Game()

# Use different solvers
mixed_solver = MixedStrategySolver()
pure_solver = PureStrategyNashSolver()
backward_solver = BackwardInductionSolver()

# Solve the game
solution = mixed_solver.solve(game)
```

## Features

- **Mixed Strategy Solver**: Find mixed strategy Nash equilibria
- **Pure Strategy Solver**: Find pure strategy Nash equilibria  
- **Backward Induction Solver**: Solve sequential games using backward induction
- **Game Representation**: Flexible game tree representation with Node class

## Development

To set up the development environment:

```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 