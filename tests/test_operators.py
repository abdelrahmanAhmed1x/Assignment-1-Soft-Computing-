import math
import random
import unittest
from unittest.mock import Mock
from ga.models import Individual
from operators.selection import RouletteWheelSelection, TournamentSelection
from operators.crossover import MultiPointCrossover, OrderCrossover
from operators.mutation import BitFlipMutation, SwapMutation, InversionMutation, UniformFloatMutation
from operators.replacement import GenerationalReplacement, ElitistReplacement


def population(*values):
    return [Individual((i,), v, True) for i, v in enumerate(values)]


class SelectionTests(unittest.TestCase):
    def test_roulette_intervals(self):
        pop = population(1, 2, 7)
        for draw, index in [(0., 0), (.099, 0), (.1, 1), (.299, 1), (.3, 2), (.999, 2)]:
            self.assertIs(RouletteWheelSelection().select(pop, False, Mock(random=lambda: draw)), pop[index])
        rng = random.Random(9)
        counts = [0, 0, 0]
        for _ in range(10000):
            counts[pop.index(RouletteWheelSelection().select(pop, False, rng))] += 1
        for count, expected in zip(counts, (1000, 2000, 7000)):
            self.assertLess(abs(count - expected), 180)

    def test_equal_zero_negative_minimization(self):
        for values in [(0, 0), (5, 5), (-3, -3)]:
            pop = population(*values)
            rng = Mock(choice=lambda p: p[1])
            self.assertIs(RouletteWheelSelection().select(pop, False, rng), pop[1])
        pop = population(-3, -1, 0)
        self.assertIs(RouletteWheelSelection().select(pop, False, Mock(random=lambda: 0)), pop[1])
        self.assertIs(RouletteWheelSelection().select(pop, True, Mock(random=lambda: 0)), pop[0])
        for strategy in [RouletteWheelSelection(), TournamentSelection()]:
            for value in [math.inf, math.nan]:
                with self.assertRaises(ValueError):
                    strategy.select(population(0, value), True, random.Random())

    def test_tournament_known_competitors(self):
        pop = population(4, 1, 8)
        rng = Mock(sample=lambda p, size: [pop[0], pop[2]])
        self.assertIs(TournamentSelection().select(pop, True, rng), pop[0])
        self.assertIs(TournamentSelection().select(pop, False, rng), pop[2])
        with self.assertRaises(ValueError):
            TournamentSelection(4).select(pop, True, random.Random())


class CrossoverTests(unittest.TestCase):
    def test_alternating_segments_and_immutability(self):
        a, b = (0,) * 7, (1,) * 7
        children = MultiPointCrossover().cross(a, b, Mock(sample=lambda seq, k: [2, 5]))
        self.assertEqual(children, ((0, 0, 1, 1, 1, 0, 0), (1, 1, 0, 0, 0, 1, 1)))
        self.assertEqual(a, (0,) * 7)
        self.assertEqual(b, (1,) * 7)

    def test_lecture_ox_wraparound(self):
        a = tuple(range(1, 10))
        b = (9, 3, 7, 8, 2, 6, 5, 1, 4)
        children = OrderCrossover().cross(a, b, Mock(sample=lambda seq, k: [3, 6]))
        self.assertEqual(children[0], (3, 8, 2, 4, 5, 6, 7, 1, 9))
        self.assertEqual(children[1], (3, 4, 7, 8, 2, 6, 5, 9, 1))
        self.assertEqual(a, tuple(range(1, 10)))
        self.assertEqual(b, (9, 3, 7, 8, 2, 6, 5, 1, 4))
        for seed in range(100):
            for child in OrderCrossover().cross(a, b, random.Random(seed)):
                self.assertEqual(sorted(child), list(a))

    def test_small_and_invalid(self):
        self.assertEqual(OrderCrossover().cross((1,), (1,), random.Random()), ((1,), (1,)))
        self.assertEqual(OrderCrossover().cross((), (), random.Random()), ((), ()))
        for a, b in [((1, 1), (1, 2)), ((1, 2), (1, 3)), ((1., 2), (1, 2))]:
            with self.assertRaises(ValueError):
                OrderCrossover().cross(a, b, random.Random())
        with self.assertRaises(ValueError):
            MultiPointCrossover().cross((1, 2), (3, 4), random.Random())
        with self.assertRaises(ValueError):
            MultiPointCrossover(0)


class MutationTests(unittest.TestCase):
    def test_bit_flip_probabilities(self):
        op = BitFlipMutation()
        a = (0, 1, 0, 1)
        self.assertEqual(op.mutate(a, Mock(random=lambda: 0), 0), a)
        self.assertEqual(op.mutate(a, random.Random(2), 1), (1, 0, 1, 0))
        rng = Mock()
        rng.random.side_effect = [.1, .6, .5, .9]
        self.assertEqual(op.mutate(a, rng, .5), (1, 1, 1, 1))
        self.assertEqual(a, (0, 1, 0, 1))

    def test_swap_inversion_one_check(self):
        a = (1, 2, 3, 4, 5, 6)
        for op, expected in [(SwapMutation(), (1, 5, 3, 4, 2, 6)),
                             (InversionMutation(), (1, 5, 4, 3, 2, 6))]:
            rng = Mock(random=Mock(return_value=.2), sample=Mock(return_value=[1, 4]))
            self.assertEqual(op.mutate(a, rng, .3), expected)
            self.assertEqual(rng.random.call_count, 1)
            self.assertEqual(sorted(expected), list(a))
            self.assertEqual(op.mutate(a, random.Random(), 0), a)
            self.assertEqual(op.mutate((1,), random.Random(), 1), (1,))
            self.assertEqual(sorted(op.mutate(a, random.Random(1), 1)), list(a))
        self.assertEqual(a, (1, 2, 3, 4, 5, 6))

    def test_float_displacement_math(self):
        rng = Mock()
        rng.random.side_effect = [.1, .5, .2, .8]  # gate, direction, gate, direction
        rng.uniform.side_effect = [1.5, 2.]
        a = (4., 4.)
        result = UniformFloatMutation((1., 0.), (10., 10.)).mutate(a, rng, .5)
        self.assertEqual(result, (2.5, 6.))
        self.assertEqual([call.args for call in rng.uniform.call_args_list], [(0, 3.), (0, 6.)])
        self.assertEqual(a, (4., 4.))

    def test_float_bounds_rates_and_per_gene(self):
        op = UniformFloatMutation(0., 10.)
        self.assertEqual(op.mutate((0., 10.), random.Random(), 0), (0., 10.))
        for seed in range(100):
            self.assertTrue(all(0 <= x <= 10 for x in op.mutate((0., 10., 5.), random.Random(seed), 1)))
        rng = Mock()
        rng.random.side_effect = [.8, .1, .2]
        rng.uniform.return_value = 1
        self.assertEqual(op.mutate((5., 5.), rng, .5), (5., 4.))
        for op in [BitFlipMutation(), SwapMutation(), InversionMutation(), UniformFloatMutation(0., 1.)]:
            with self.assertRaises(ValueError):
                op.mutate((0, 1), random.Random(), math.nan)


class ReplacementTests(unittest.TestCase):
    def test_generational(self):
        parents, children = population(0, 1), population(10, 11)
        result = GenerationalReplacement().replace(parents, children, 2, True)
        self.assertEqual(result, children)
        self.assertIsNot(result, children)
        with self.assertRaises(ValueError):
            GenerationalReplacement().replace(parents, [], 2, True)

    def test_mixed_both_directions(self):
        a, b = population(2, 8), population(1, 9)
        for minimize, expected in [(True, [1, 2]), (False, [9, 8])]:
            result = ElitistReplacement().replace(a, b, 2, minimize)
            self.assertEqual([i.fitness for i in result], expected)
