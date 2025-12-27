"""
Comprehensive Test Suite for Game Theory Package

This test suite validates all core functionality of the sl_package game theory library,
including game construction, node operations, and all solver algorithms.

Run with: python test_game_theory.py
"""

import unittest
import numpy as np
import sys
from io import StringIO

# Import all components from the package
from src.sl_package import (
    Node, Game,
    PureStrategyNashSolver,
    MixedStrategySolver,
    BackwardInductionSolver,
    Solver
)
from src.sl_package import sampling


class TestNodeClass(unittest.TestCase):
    """Tests for the Node class."""

    def test_node_creation_empty(self):
        """Test creating a node with no players."""
        node = Node()
        self.assertIsInstance(node.players, set)
        self.assertEqual(len(node.players), 0)
        self.assertEqual(len(node.actions), 0)
        self.assertIsNone(node.payoff)

    def test_node_creation_with_players(self):
        """Test creating a node with players specified."""
        players = {'Player1', 'Player2'}
        node = Node(players=players)
        self.assertEqual(node.players, players)

    def test_add_action(self):
        """Test adding actions to a node."""
        node = Node()
        child = Node()
        node.add_action('Left', child)

        self.assertIn('Left', node.actions)
        self.assertEqual(node.actions['Left'], child)

    def test_add_multiple_actions(self):
        """Test adding multiple actions to a node."""
        node = Node()
        child1 = Node()
        child2 = Node()

        node.add_action('Up', child1)
        node.add_action('Down', child2)

        self.assertEqual(len(node.actions), 2)
        self.assertEqual(node.actions['Up'], child1)
        self.assertEqual(node.actions['Down'], child2)

    def test_payoff_assignment(self):
        """Test assigning payoffs to terminal nodes."""
        node = Node()
        node.payoff = (3, 2)
        self.assertEqual(node.payoff, (3, 2))

        # Test with multiple players
        node.payoff = (1, 2, 3)
        self.assertEqual(node.payoff, (1, 2, 3))


class TestGameClass(unittest.TestCase):
    """Tests for the Game class."""

    def test_game_initialization(self):
        """Test game initialization creates empty root."""
        game = Game()
        self.assertIsNotNone(game.root)
        self.assertIsInstance(game.root, Node)
        self.assertEqual(len(game.players), 0)
        self.assertEqual(len(game.current_nodes), 1)

    def test_add_player(self):
        """Test adding players to a game."""
        game = Game()
        idx1 = game.add_player('Alice')
        idx2 = game.add_player('Bob')

        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 1)
        self.assertEqual(len(game.players), 2)
        self.assertIn('Alice', game.players)
        self.assertIn('Bob', game.players)

    def test_add_player_duplicate(self):
        """Test adding the same player twice returns same index."""
        game = Game()
        idx1 = game.add_player('Alice')
        idx2 = game.add_player('Alice')

        self.assertEqual(idx1, idx2)
        self.assertEqual(len(game.players), 1)

    def test_get_player_index(self):
        """Test getting player index from name."""
        game = Game()
        game.add_player('Alice')
        game.add_player('Bob')

        self.assertEqual(game.get_player_index('Alice'), 0)
        self.assertEqual(game.get_player_index('Bob'), 1)

    def test_get_player_index_not_found(self):
        """Test getting index for non-existent player raises error."""
        game = Game()
        game.add_player('Alice')

        with self.assertRaises(ValueError):
            game.get_player_index('Charlie')

    def test_add_moves_single_player(self):
        """Test adding moves for a single player."""
        game = Game()
        game.add_moves('Player1', ['Left', 'Right'])

        self.assertEqual(len(game.current_nodes), 2)
        self.assertIn('Player1', game.players)

    def test_add_moves_multiple_players(self):
        """Test adding moves for multiple players creates correct tree."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_moves('Player2', ['Left', 'Right'])

        # Should create 2x2 = 4 terminal nodes
        self.assertEqual(len(game.current_nodes), 4)
        self.assertEqual(len(game.players), 2)

    def test_add_outcomes_correct_count(self):
        """Test adding outcomes with correct number of nodes."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_moves('Player2', ['Left', 'Right'])

        outcomes = [(3, 3), (0, 5), (5, 0), (1, 1)]
        game.add_outcomes(outcomes)

        # Verify payoffs are assigned
        for node, expected_payoff in zip(game.current_nodes, outcomes):
            self.assertEqual(node.payoff, expected_payoff)

    def test_add_outcomes_wrong_count(self):
        """Test adding outcomes with wrong number raises error."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_moves('Player2', ['Left', 'Right'])

        # Wrong number of outcomes
        outcomes = [(3, 3), (0, 5)]

        with self.assertRaises(ValueError):
            game.add_outcomes(outcomes)

    def test_display_tree(self):
        """Test display_tree executes without error."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_outcomes([(1, 0), (0, 1)])

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            game.display_tree()
            output = sys.stdout.getvalue()
            self.assertIn('Payoff', output)
        finally:
            sys.stdout = old_stdout


class TestPrisonersDilemma(unittest.TestCase):
    """Tests using the classic Prisoner's Dilemma game."""

    def setUp(self):
        """Create a Prisoner's Dilemma game for testing."""
        self.game = Game()
        self.game.add_moves('Prisoner1', ['Cooperate', 'Defect'])
        self.game.add_moves('Prisoner2', ['Cooperate', 'Defect'])
        # Payoffs: (C,C)=(-1,-1), (C,D)=(-3,0), (D,C)=(0,-3), (D,D)=(-2,-2)
        self.game.add_outcomes([
            (-1, -1),  # Both cooperate
            (-3, 0),   # P1 cooperates, P2 defects
            (0, -3),   # P1 defects, P2 cooperates
            (-2, -2)   # Both defect
        ])

    def test_prisoners_dilemma_pure_strategy(self):
        """Test finding pure strategy Nash equilibrium in Prisoner's Dilemma."""
        solver = PureStrategyNashSolver(self.game)
        equilibria = solver.solve()

        # Should find exactly one equilibrium: (Defect, Defect)
        self.assertEqual(len(equilibria), 1)
        self.assertEqual(equilibria[0]['Prisoner1'], 'Defect')
        self.assertEqual(equilibria[0]['Prisoner2'], 'Defect')

    def test_prisoners_dilemma_mixed_strategy(self):
        """Test mixed strategy solver on Prisoner's Dilemma."""
        solver = MixedStrategySolver(self.game)
        equilibrium = solver.solve()

        # Should return a valid equilibrium
        self.assertIsNotNone(equilibrium)
        self.assertIn('Prisoner1', equilibrium)
        self.assertIn('Prisoner2', equilibrium)


class TestMatchingPennies(unittest.TestCase):
    """Tests using the Matching Pennies game (no pure Nash equilibrium)."""

    def setUp(self):
        """Create a Matching Pennies game."""
        self.game = Game()
        self.game.add_moves('Player1', ['Heads', 'Tails'])
        self.game.add_moves('Player2', ['Heads', 'Tails'])
        # P1 wins if same, P2 wins if different
        self.game.add_outcomes([
            (1, -1),   # Both heads
            (-1, 1),   # P1 heads, P2 tails
            (-1, 1),   # P1 tails, P2 heads
            (1, -1)    # Both tails
        ])

    def test_matching_pennies_no_pure_nash(self):
        """Test that Matching Pennies has no pure strategy Nash equilibrium."""
        solver = PureStrategyNashSolver(self.game)
        equilibria = solver.solve()

        # Should find no pure strategy equilibria
        self.assertEqual(len(equilibria), 0)

    def test_matching_pennies_mixed_strategy(self):
        """Test mixed strategy Nash equilibrium is 50-50 for both players."""
        solver = MixedStrategySolver(self.game)
        equilibrium = solver.solve()

        # Extract probabilities (they're formatted as strings like "50.0%")
        p1_heads = float(equilibrium['Player1']['Heads'].rstrip('%'))
        p2_heads = float(equilibrium['Player2']['Heads'].rstrip('%'))

        # Should be close to 50-50 for both players
        self.assertAlmostEqual(p1_heads, 50.0, delta=1.0)
        self.assertAlmostEqual(p2_heads, 50.0, delta=1.0)


class TestPureStrategyNashSolver(unittest.TestCase):
    """Tests for the Pure Strategy Nash Solver."""

    def setUp(self):
        """Create a simple coordination game."""
        self.game = Game()
        self.game.add_moves('Player1', ['A', 'B'])
        self.game.add_moves('Player2', ['A', 'B'])
        # Coordination game: (A,A)=(2,2), (A,B)=(0,0), (B,A)=(0,0), (B,B)=(1,1)
        self.game.add_outcomes([
            (2, 2),  # Both choose A
            (0, 0),  # P1 A, P2 B
            (0, 0),  # P1 B, P2 A
            (1, 1)   # Both choose B
        ])

    def test_coordination_game_multiple_equilibria(self):
        """Test finding multiple Nash equilibria in coordination game."""
        solver = PureStrategyNashSolver(self.game)
        equilibria = solver.solve()

        # Should find two equilibria: (A, A) and (B, B)
        self.assertEqual(len(equilibria), 2)

        # Check both equilibria are present
        strategies = [(eq['Player1'], eq['Player2']) for eq in equilibria]
        self.assertIn(('A', 'A'), strategies)
        self.assertIn(('B', 'B'), strategies)

    def test_strategy_profile_generation(self):
        """Test that all strategy profiles are generated."""
        solver = PureStrategyNashSolver(self.game)
        solver._generate_strategy_profiles()

        # Should generate 2x2 = 4 strategy profiles
        self.assertEqual(len(solver.strategy_profiles), 4)

    def test_payoff_computation(self):
        """Test payoff computation for all strategy profiles."""
        solver = PureStrategyNashSolver(self.game)
        solver._generate_strategy_profiles()
        solver._compute_payoffs()

        # All strategy profiles should have payoffs
        self.assertEqual(len(solver.payoff_matrix), len(solver.strategy_profiles))

    def test_print_equilibria(self):
        """Test print_equilibria executes without error."""
        solver = PureStrategyNashSolver(self.game)

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            solver.print_equilibria()
            output = sys.stdout.getvalue()
            self.assertIn('Nash Equilibria', output)
        finally:
            sys.stdout = old_stdout

    def test_record_equilibrium(self):
        """Test record_equilibrium returns first equilibrium."""
        solver = PureStrategyNashSolver(self.game)
        solver.solve()
        result = solver.record_equilibrium()

        self.assertIsInstance(result, dict)
        self.assertIn('Player1', result)
        self.assertIn('Player2', result)


class TestMixedStrategySolver(unittest.TestCase):
    """Tests for the Mixed Strategy Solver."""

    def test_2x2_game_closed_form(self):
        """Test that 2x2 games use closed-form solution."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_moves('Player2', ['Left', 'Right'])
        game.add_outcomes([
            (3, 3), (0, 5),
            (5, 0), (1, 1)
        ])

        solver = MixedStrategySolver(game)
        equilibrium = solver.solve()

        # Should return formatted equilibrium
        self.assertIsNotNone(equilibrium)
        self.assertIn('Player1', equilibrium)
        self.assertIn('Player2', equilibrium)

    def test_2x2_mixed_strategy_calculation(self):
        """Test the static 2x2 mixed strategy calculation method."""
        payoff_A = np.array([[3, 0], [0, 3]])
        payoff_B = np.array([[3, 0], [0, 3]])

        p, q = MixedStrategySolver._solve_2_player_mixed_strategy(payoff_A, payoff_B)

        # Probabilities should be in [0, 1]
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)
        self.assertGreaterEqual(q, 0)
        self.assertLessEqual(q, 1)

    def test_zero_division_handling(self):
        """Test that zero division is handled gracefully."""
        # Create a game where one strategy strictly dominates
        payoff_A = np.array([[5, 5], [0, 0]])
        payoff_B = np.array([[5, 0], [5, 0]])

        p, q = MixedStrategySolver._solve_2_player_mixed_strategy(payoff_A, payoff_B)

        # Should not crash and return valid probabilities
        self.assertIsNotNone(p)
        self.assertIsNotNone(q)

    def test_solver_validates_min_players(self):
        """Test solver raises error for single-player games."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_outcomes([(1,), (0,)])

        solver = MixedStrategySolver(game)

        with self.assertRaises(ValueError):
            solver.solve()

    def test_get_equilibrium_method(self):
        """Test get_equilibrium method returns computed equilibrium."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_moves('Player2', ['Left', 'Right'])
        game.add_outcomes([(3, 3), (0, 5), (5, 0), (1, 1)])

        solver = MixedStrategySolver(game)
        solver.solve()

        equilibrium = solver.get_equilibrium()
        self.assertIsNotNone(equilibrium)


class TestBackwardInductionSolver(unittest.TestCase):
    """Tests for the Backward Induction Solver."""

    def setUp(self):
        """Create a simple sequential game."""
        self.game = Game()
        self.game.add_moves('Player1', ['Left', 'Right'])
        self.game.add_moves('Player2', ['Accept', 'Reject'])
        # Sequential game tree structure
        self.game.add_outcomes([
            (2, 2),  # Left, Accept
            (0, 0),  # Left, Reject
            (1, 3),  # Right, Accept
            (3, 1)   # Right, Reject
        ])

    def test_backward_induction_solve(self):
        """Test backward induction finds optimal actions."""
        solver = BackwardInductionSolver(self.game)
        optimal_actions = solver.solve()

        # Should find optimal actions for nodes in the tree
        self.assertIsInstance(optimal_actions, dict)
        self.assertGreater(len(optimal_actions), 0)

    def test_get_subgame_perfect_equilibrium(self):
        """Test extracting subgame perfect equilibrium."""
        solver = BackwardInductionSolver(self.game)
        solver.solve()
        equilibrium = solver.get_subgame_perfect_equilibrium()

        # Should return equilibrium strategies by player
        self.assertIsInstance(equilibrium, dict)

    def test_node_values_computed(self):
        """Test that node values are computed during backward induction."""
        solver = BackwardInductionSolver(self.game)
        solver.solve()

        # Node values should be stored
        self.assertGreater(len(solver.node_values), 0)

    def test_print_equilibrium(self):
        """Test print_equilibrium executes without error."""
        solver = BackwardInductionSolver(self.game)

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            solver.print_equilibrium()
            output = sys.stdout.getvalue()
            self.assertIn('Equilibrium', output)
        finally:
            sys.stdout = old_stdout

    def test_record_equilibrium(self):
        """Test record_equilibrium returns dict of actions."""
        solver = BackwardInductionSolver(self.game)
        result = solver.record_equilibrium()

        self.assertIsInstance(result, dict)


class TestThreePlayerGame(unittest.TestCase):
    """Tests for three-player games."""

    def setUp(self):
        """Create a three-player game."""
        self.game = Game()
        self.game.add_moves('Player1', ['A', 'B'])
        self.game.add_moves('Player2', ['C', 'D'])
        self.game.add_moves('Player3', ['E', 'F'])

        # Create 2x2x2 = 8 outcomes
        self.game.add_outcomes([
            (1, 1, 1), (2, 0, 0),  # P1=A, P2=C, P3=E/F
            (0, 2, 0), (3, 3, 3),  # P1=A, P2=D, P3=E/F
            (0, 0, 2), (1, 1, 1),  # P1=B, P2=C, P3=E/F
            (2, 2, 2), (0, 1, 1)   # P1=B, P2=D, P3=E/F
        ])

    def test_three_player_pure_strategy(self):
        """Test pure strategy solver on three-player game."""
        solver = PureStrategyNashSolver(self.game)
        equilibria = solver.solve()

        # Should complete without error
        self.assertIsInstance(equilibria, list)

    def test_three_player_payoff_tuples(self):
        """Test that three-player payoff tuples are handled correctly."""
        solver = PureStrategyNashSolver(self.game)
        solver._generate_strategy_profiles()
        solver._compute_payoffs()

        # Check that payoffs are 3-tuples
        for payoff in solver.payoff_matrix.values():
            if payoff is not None:
                self.assertEqual(len(payoff), 3)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    def test_negative_payoffs(self):
        """Test games with negative payoffs."""
        game = Game()
        game.add_moves('Player1', ['Up', 'Down'])
        game.add_moves('Player2', ['Left', 'Right'])
        game.add_outcomes([
            (-5, -5), (-10, 0),
            (0, -10), (-8, -8)
        ])

        solver = PureStrategyNashSolver(game)
        equilibria = solver.solve()

        # Should handle negative payoffs
        self.assertIsInstance(equilibria, list)

    def test_zero_payoffs(self):
        """Test games with all zero payoffs."""
        game = Game()
        game.add_moves('Player1', ['A', 'B'])
        game.add_moves('Player2', ['C', 'D'])
        game.add_outcomes([
            (0, 0), (0, 0),
            (0, 0), (0, 0)
        ])

        solver = PureStrategyNashSolver(game)
        equilibria = solver.solve()

        # All strategies are Nash equilibria when payoffs are equal
        self.assertEqual(len(equilibria), 4)

    def test_float_payoffs(self):
        """Test games with floating-point payoffs."""
        game = Game()
        game.add_moves('Player1', ['X', 'Y'])
        game.add_moves('Player2', ['Z', 'W'])
        game.add_outcomes([
            (1.5, 2.7), (0.3, 5.1),
            (4.8, 0.2), (1.1, 1.1)
        ])

        solver = PureStrategyNashSolver(game)
        equilibria = solver.solve()

        # Should handle float payoffs
        self.assertIsInstance(equilibria, list)

    def test_asymmetric_game(self):
        """Test asymmetric games (different number of strategies per player)."""
        game = Game()
        game.add_moves('Player1', ['A', 'B', 'C'])
        game.add_moves('Player2', ['X', 'Y'])

        # 3x2 = 6 outcomes
        game.add_outcomes([
            (1, 1), (0, 2),  # A, X/Y
            (2, 0), (3, 3),  # B, X/Y
            (1, 2), (2, 1)   # C, X/Y
        ])

        solver = PureStrategyNashSolver(game)
        equilibria = solver.solve()

        # Should handle asymmetric games
        self.assertIsInstance(equilibria, list)


class TestSamplingModule(unittest.TestCase):
    """Tests for the sampling utility functions."""

    def test_generate_distribution_kde(self):
        """Test generating distribution with KDE method."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        x, density, dist = sampling.generate_distribution(data, method='kde')

        self.assertEqual(len(x), 1000)
        self.assertEqual(len(density), 1000)
        self.assertIsNotNone(dist)

    def test_generate_distribution_gaussian(self):
        """Test generating distribution with gaussian method."""
        np.random.seed(42)
        data = np.random.normal(5, 2, 100)

        x, density, dist = sampling.generate_distribution(data, method='gaussian')

        self.assertEqual(len(x), 1000)
        self.assertEqual(len(density), 1000)
        self.assertIsNotNone(dist)

    def test_generate_distribution_custom_points(self):
        """Test generating distribution with custom number of points."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)

        x, density, dist = sampling.generate_distribution(data, method='kde', num_points=500)

        self.assertEqual(len(x), 500)
        self.assertEqual(len(density), 500)

    def test_generate_samples_kde(self):
        """Test generating samples from KDE distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        x, density, kde_dist = sampling.generate_distribution(data, method='kde')

        samples = sampling.generate_samples(kde_dist, num_samples=50, method='kde')

        self.assertEqual(len(samples), 50)

    def test_generate_samples_gaussian(self):
        """Test generating samples from gaussian distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        x, density, gaussian_dist = sampling.generate_distribution(data, method='gaussian')

        samples = sampling.generate_samples(gaussian_dist, num_samples=50, method='gaussian')

        self.assertEqual(len(samples), 50)

    def test_sample_from_distribution(self):
        """Test simple normal distribution sampling."""
        np.random.seed(42)
        samples = sampling.sample_from_distribution(mean=5, std_dev=2, num_samples=100)

        self.assertEqual(len(samples), 100)
        # Check mean is approximately correct
        self.assertAlmostEqual(np.mean(samples), 5, delta=1)

    def test_sample_from_distribution_single(self):
        """Test sampling single value from distribution."""
        np.random.seed(42)
        sample = sampling.sample_from_distribution(mean=10, std_dev=1, num_samples=1)

        self.assertEqual(len(sample), 1)


class TestSolverBaseClass(unittest.TestCase):
    """Tests for the base Solver class."""

    def test_solver_initialization(self):
        """Test that Solver base class can be initialized via concrete implementation."""
        game = Game()
        game.add_moves('Player1', ['A', 'B'])
        game.add_moves('Player2', ['C', 'D'])
        game.add_outcomes([(1, 1), (0, 0), (0, 0), (1, 1)])

        # Use a concrete solver to test base class initialization
        solver = PureStrategyNashSolver(game)

        self.assertEqual(solver.game, game)
        self.assertEqual(solver.equilibrium, {})

    def test_solver_get_equilibrium(self):
        """Test get_equilibrium method on base class."""
        game = Game()
        game.add_moves('Player1', ['A', 'B'])
        game.add_moves('Player2', ['C', 'D'])
        game.add_outcomes([(1, 1), (0, 0), (0, 0), (1, 1)])

        # Use a concrete solver to test base class method
        solver = PureStrategyNashSolver(game)

        # Initially should return empty dict
        self.assertEqual(solver.get_equilibrium(), {})

        # After setting equilibrium
        solver.equilibrium = {'test': 'value'}
        self.assertEqual(solver.get_equilibrium(), {'test': 'value'})


class TestDeterministicBehavior(unittest.TestCase):
    """Tests to ensure deterministic behavior with random seeds."""

    def test_mixed_strategy_deterministic(self):
        """Test that mixed strategy solver is deterministic with same seed."""
        # Create game
        game1 = Game()
        game1.add_moves('P1', ['A', 'B'])
        game1.add_moves('P2', ['C', 'D'])
        game1.add_outcomes([(1, -1), (-1, 1), (-1, 1), (1, -1)])

        game2 = Game()
        game2.add_moves('P1', ['A', 'B'])
        game2.add_moves('P2', ['C', 'D'])
        game2.add_outcomes([(1, -1), (-1, 1), (-1, 1), (1, -1)])

        # For 2x2 games, the solver uses closed-form solution (deterministic)
        solver1 = MixedStrategySolver(game1)
        result1 = solver1.solve()

        solver2 = MixedStrategySolver(game2)
        result2 = solver2.solve()

        # Results should be identical
        self.assertEqual(result1, result2)

    def test_sampling_deterministic(self):
        """Test that sampling functions are deterministic with same seed."""
        np.random.seed(42)
        sample1 = sampling.sample_from_distribution(5, 1, 10)

        np.random.seed(42)
        sample2 = sampling.sample_from_distribution(5, 1, 10)

        np.testing.assert_array_equal(sample1, sample2)


class TestBattleOfSexes(unittest.TestCase):
    """Test with the classic Battle of the Sexes game."""

    def setUp(self):
        """Create a Battle of the Sexes game."""
        self.game = Game()
        self.game.add_moves('Player1', ['Opera', 'Football'])
        self.game.add_moves('Player2', ['Opera', 'Football'])
        # P1 prefers Opera together, P2 prefers Football together
        # Both prefer being together over being apart
        self.game.add_outcomes([
            (2, 1),  # Both Opera (P1's favorite)
            (0, 0),  # P1 Opera, P2 Football
            (0, 0),  # P1 Football, P2 Opera
            (1, 2)   # Both Football (P2's favorite)
        ])

    def test_battle_of_sexes_two_pure_equilibria(self):
        """Test that Battle of Sexes has two pure Nash equilibria."""
        solver = PureStrategyNashSolver(self.game)
        equilibria = solver.solve()

        # Should have exactly 2 pure equilibria
        self.assertEqual(len(equilibria), 2)

        strategies = [(eq['Player1'], eq['Player2']) for eq in equilibria]
        self.assertIn(('Opera', 'Opera'), strategies)
        self.assertIn(('Football', 'Football'), strategies)

    def test_battle_of_sexes_mixed_strategy(self):
        """Test mixed strategy equilibrium exists."""
        solver = MixedStrategySolver(self.game)
        equilibrium = solver.solve()

        # Should find a mixed strategy equilibrium
        self.assertIsNotNone(equilibrium)
        self.assertIn('Player1', equilibrium)
        self.assertIn('Player2', equilibrium)


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNodeClass))
    suite.addTests(loader.loadTestsFromTestCase(TestGameClass))
    suite.addTests(loader.loadTestsFromTestCase(TestPrisonersDilemma))
    suite.addTests(loader.loadTestsFromTestCase(TestMatchingPennies))
    suite.addTests(loader.loadTestsFromTestCase(TestPureStrategyNashSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestMixedStrategySolver))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardInductionSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestThreePlayerGame))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSamplingModule))
    suite.addTests(loader.loadTestsFromTestCase(TestSolverBaseClass))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicBehavior))
    suite.addTests(loader.loadTestsFromTestCase(TestBattleOfSexes))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)