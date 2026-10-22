import itertools
import math
import unittest
from dataclasses import replace
from ga.config import GAConfig
from examples.run_cvrp import load_instance, solve
from problems.cvrp.models import Customer, Vehicle, CVRPInstance
from problems.cvrp.problem import CVRPProblem


def small(demands=(1, 1, 1), capacities=(2, 2), matrix=None):
    n = len(demands) + 1
    matrix = matrix or tuple(tuple(abs(i - j) for j in range(n)) for i in range(n))
    return CVRPInstance(0, tuple(Customer(i, d) for i, d in enumerate(demands, 1)),
                        tuple(Vehicle(i, c) for i, c in enumerate(capacities)), matrix)


class CVRPTests(unittest.TestCase):
    def assert_routes(self, problem, chromosome):
        routes = problem.decode(chromosome)
        self.assertIsNotNone(routes)
        self.assertEqual(sorted(c for r in routes for c in r.customers), sorted(problem.ids))
        self.assertLessEqual(len(routes), len(problem.instance.vehicles))
        self.assertEqual(len({r.vehicle_id for r in routes}), len(routes))
        capacities = {v.id: v.capacity for v in problem.instance.vehicles}
        total = 0
        for route in routes:
            self.assertEqual(route.load, sum(problem.demands[c] for c in route.customers))
            self.assertLessEqual(route.load, capacities[route.vehicle_id])
            indices = [0] + [problem.indices[c] for c in route.customers] + [0]
            cost = sum(problem.instance.distance_matrix[a][b] for a, b in zip(indices, indices[1:]))
            self.assertEqual(route.distance, cost)
            total += cost
        self.assertEqual(problem.fitness(chromosome), total)
        return routes

    def test_manual_distance_optimum_and_coverage(self):
        problem = CVRPProblem(small())
        self.assertEqual(problem.fitness((1, 2, 3)), 8)
        self.assertEqual(min(problem.fitness(c) for c in itertools.permutations(problem.ids)), 8)
        self.assert_routes(problem, (1, 2, 3))
        for bad in [(1, 2), (1, 1, 3), (1, 2, 9), (1., 2, 3)]:
            self.assertFalse(problem.feasible(bad))
            with self.assertRaises(ValueError):
                problem.fitness(bad)

    def test_dp_improves_greedy_cost(self):
        problem = CVRPProblem(small())
        # Greedy [1,2] [3] costs 4+6=10; DP [1] [2,3] costs 8.
        self.assertLess(problem.fitness((1, 2, 3)), 10)

    def test_dp_matches_all_contiguous_partitions(self):
        for capacities in [(2, 2), (1, 3), (3, 1)]:
            problem = CVRPProblem(small(capacities=capacities))
            for chromosome in itertools.permutations(problem.ids):
                costs = []
                for split in range(4):
                    chunks = [chromosome[:split], chromosome[split:]]
                    if all(len(chunk) <= cap for chunk, cap in zip(chunks, capacities)):
                        cost = 0
                        for chunk in chunks:
                            if chunk:
                                path = (0,) + chunk + (0,)
                                cost += sum(abs(a - b) for a, b in zip(path, path[1:]))
                        costs.append(cost)
                self.assertEqual(problem.fitness(chromosome), min(costs))
                self.assert_routes(problem, chromosome)

    def test_heterogeneous_skip_and_order_limitation(self):
        problem = CVRPProblem(small((6, 4), (3, 10)))
        routes = self.assert_routes(problem, (1, 2))
        self.assertEqual([r.vehicle_id for r in routes], [1])
        problem = CVRPProblem(small((4, 3), (3, 4)))
        self.assertFalse(problem.feasible((1, 2)))
        self.assertTrue(problem.feasible((2, 1)))

    def test_unsatisfiable_packing(self):
        problem = CVRPProblem(small((4, 4, 4), (7, 7)))
        self.assertFalse(problem.feasible((1, 2, 3)))
        with self.assertRaisesRegex(RuntimeError, 'Initialization'):
            solve(problem, GAConfig(2, 1, seed=1))

    def test_instance_validation(self):
        base = small()
        invalid = [dict(vehicles=(Vehicle(0, 0),)), dict(vehicles=(Vehicle(0, math.inf),)),
                   dict(customers=(Customer(1, -1),)), dict(customers=(Customer(1, math.nan),)),
                   dict(customers=(Customer(1, 1), Customer(1, 1))), dict(depot=1),
                   dict(distance_matrix=((0,),)),
                   dict(distance_matrix=tuple((math.nan,) * 4 for _ in range(4))),
                   dict(distance_matrix=tuple((-1,) * 4 for _ in range(4))),
                   dict(customers=(Customer(1, 9),)), dict(vehicles=(Vehicle(0, 2),))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(base, **kwargs)

    def test_arbitrary_ids_asymmetric_distances_and_zero_demand(self):
        instance = CVRPInstance(99, (Customer(7, 0), Customer(-2, 2)), (Vehicle(8, 2),),
                                ((0, 2, 8), (5, 0, 3), (4, 9, 0)))
        problem = CVRPProblem(instance)
        self.assertEqual(problem.fitness((7, -2)), 9)
        self.assert_routes(problem, (7, -2))

    def test_seeded_generic_engine(self):
        problem = CVRPProblem(small())
        config = GAConfig(10, 15, seed=123)
        result = solve(problem, config)
        self.assertEqual(result, solve(problem, config))
        self.assertEqual(result.best_fitness, 8)
        self.assert_routes(problem, result.best_chromosome)

    def test_official_data_and_small_run(self):
        problem = CVRPProblem(load_instance())
        self.assertEqual(len(problem.ids), 16)
        self.assertEqual(sum(problem.demands.values()), 60)
        official_order = (4, 3, 1, 7, 14, 16, 10, 9, 12, 11, 15, 13, 8, 2, 6, 5)
        self.assertEqual(problem.fitness(official_order), 6208)
        result = solve(problem, GAConfig(10, 5, seed=42))
        self.assert_routes(problem, result.best_chromosome)

    def test_saved_three_seed_experiments(self):
        import json
        from pathlib import Path
        records = json.loads((Path(__file__).resolve().parents[1] / 'data/cvrp/experiments.json').read_text())
        self.assertEqual([r['config']['seed'] for r in records], [42, 123, 2026])
        problem = CVRPProblem(load_instance())
        for record in records:
            result = record['result']
            routes = self.assert_routes(problem, tuple(result['best_chromosome']))
            self.assertEqual([r.distance for r in routes], [r['distance'] for r in record['routes']])
            self.assertEqual(sum(r.distance for r in routes), result['best_fitness'])
            history = result['fitness_history']
            self.assertEqual(len(history), result['generations'] + 1)
            self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))
            self.assertEqual(result['best_generation'], history.index(result['best_fitness']))
            self.assertEqual(len(routes), 4)
            self.assertTrue(all(r.load == 15 for r in routes))
