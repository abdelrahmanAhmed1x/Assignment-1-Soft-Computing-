import math
import unittest
from unittest.mock import Mock
from ga.config import GAConfig
from ga.engine import GeneticAlgorithm
from operators.selection import TournamentSelection
from operators.crossover import MultiPointCrossover
from operators.mutation import BitFlipMutation, SwapMutation, UniformFloatMutation
from operators.replacement import ElitistReplacement, GenerationalReplacement
from examples.simple_function import BinaryProblem, IntegerProblem, FloatProblem


def engine(problem=None, **kwargs):
    defaults = dict(problem=problem or BinaryProblem(), selection=TournamentSelection(),
                    crossover=MultiPointCrossover(), mutation=BitFlipMutation(),
                    replacement=ElitistReplacement(), config=GAConfig(7, 5, seed=42, minimize=False))
    defaults.update(kwargs)
    return GeneticAlgorithm(**defaults)


class EngineTests(unittest.TestCase):
    def test_three_representations_reproducibility_and_history(self):
        for problem, mutation, minimize in [(BinaryProblem(), BitFlipMutation(), False),
                                           (IntegerProblem(), SwapMutation(), False),
                                           (FloatProblem(), UniformFloatMutation(-5., 5.), True)]:
            ga = engine(problem, mutation=mutation, config=GAConfig(15, 20, minimize=minimize, seed=42))
            result = ga.run()
            self.assertEqual(result, ga.run())
            self.assertTrue(problem.feasible(result.best_chromosome))
            self.assertEqual(result.best_fitness, problem.fitness(result.best_chromosome))
            self.assertEqual(len(result.fitness_history), 21)
            self.assertEqual(result.generations, 20)
            self.assertEqual(result.best_generation, result.fitness_history.index(result.best_fitness))
            history = result.fitness_history
            self.assertTrue(all(a >= b if minimize else a <= b for a, b in zip(history, history[1:])))

    def test_injected_strategies_odd_size_and_crossover_rates(self):
        for rate in [0, 1]:
            selection = Mock(wraps=TournamentSelection())
            crossover = Mock(wraps=MultiPointCrossover())
            mutation = Mock(wraps=BitFlipMutation())
            replacement = Mock(wraps=GenerationalReplacement())
            result = engine(selection=selection, crossover=crossover, mutation=mutation,
                            replacement=replacement, config=GAConfig(7, 3, rate, 0, False, 1)).run()
            self.assertEqual(selection.select.call_count, 24)
            self.assertEqual(crossover.cross.call_count, 12 if rate else 0)
            self.assertEqual(mutation.mutate.call_count, 21)
            for call in mutation.mutate.call_args_list:
                self.assertEqual(call.args[2], 0)
            for call in selection.select.call_args_list:
                self.assertEqual(len(call.args[0]), 7)
            for call in replacement.replace.call_args_list:
                self.assertEqual(len(call.args[0]), 7)
                self.assertEqual(len(call.args[1]), 7)
            self.assertEqual(result.generations, 3)

    def test_initial_best_survives_generational_loss(self):
        problem = Mock(generate=Mock(return_value=(1, 1, 1)), feasible=lambda c: True,
                       fitness=lambda c: float(sum(c)))
        mutation = Mock(mutate=lambda c, rng, rate: (0, 0, 0))
        result = engine(problem, mutation=mutation, replacement=GenerationalReplacement()).run()
        self.assertEqual(result.best_fitness, 3)
        self.assertEqual(result.best_generation, 0)
        self.assertEqual(result.fitness_history, [3] * 6)

    def test_new_record_survives_later_loss(self):
        problem = Mock(generate=Mock(return_value=(0, 0, 0)), feasible=lambda c: True,
                       fitness=lambda c: float(sum(c)))
        mutation = Mock()
        mutation.mutate.side_effect = [(1, 1, 1)] * 7 + [(0, 0, 0)] * 28
        result = engine(problem, mutation=mutation, replacement=GenerationalReplacement()).run()
        self.assertEqual(result.best_fitness, 3)
        self.assertEqual(result.best_generation, 1)

    def test_rejection_and_feasibility_before_fitness(self):
        problem = Mock()
        problem.generate.side_effect = [(2,), (0,), (1,)]
        problem.feasible.side_effect = lambda c: c != (2,)
        problem.fitness.side_effect = lambda c: float(c[0]) if c != (2,) else self.fail('Invalid fitness evaluated')
        mutation = Mock()
        mutation.mutate.side_effect = [(2,), (1,), (0,)]
        result = engine(problem, mutation=mutation, config=GAConfig(2, 1, 0, 0, False, 1)).run()
        self.assertEqual(problem.generate.call_count, 3)
        self.assertEqual(result.best_fitness, 1)

    def test_retry_limits_and_nonfinite(self):
        problem = Mock(generate=lambda rng: (0,), feasible=lambda c: False)
        with self.assertRaisesRegex(RuntimeError, 'Initialization'):
            engine(problem, config=GAConfig(2, 1)).run()
        problem.feasible = lambda c: c == (0,)
        problem.fitness = lambda c: 0.
        with self.assertRaisesRegex(RuntimeError, 'Offspring'):
            engine(problem, mutation=Mock(mutate=lambda *args: (1,)), config=GAConfig(2, 1, 0)).run()
        for value in [math.nan, math.inf, -math.inf]:
            problem.fitness = lambda c: value
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                engine(problem).run()

    def test_config_and_bad_replacement(self):
        for kwargs in [dict(population_size=1), dict(population_size=2.5), dict(max_generations=0),
                       dict(crossover_rate=-1), dict(mutation_rate=1.1), dict(mutation_rate=math.nan)]:
            with self.assertRaises(ValueError):
                GAConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, 'contract'):
            engine(replacement=Mock(replace=lambda *args: [])).run()
