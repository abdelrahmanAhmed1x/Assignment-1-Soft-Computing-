"""Permutation encoding with minimum-distance contiguous route splitting."""
import math
import random
from ga.models import Chromosome
from ga.representations import generate_permutation, validate_permutation
from problems.cvrp.models import CVRPInstance, Route


class CVRPProblem:
    def __init__(self, instance: CVRPInstance) -> None:
        self.instance = instance
        self.ids = tuple(c.id for c in instance.customers)
        self.demands = {c.id: c.demand for c in instance.customers}
        self.indices = {identifier: i for i, identifier in enumerate((instance.depot,) + self.ids)}

    def generate(self, rng: random.Random) -> Chromosome:
        return generate_permutation(self.ids, rng)

    def decode(self, chromosome: Chromosome) -> list[Route] | None:
        if not validate_permutation(chromosome, self.ids):
            return None
        n, k = len(chromosome), len(self.instance.vehicles)
        matrix = self.instance.distance_matrix
        # segment [start:end] includes both depot arcs. O(N^2) precomputation.
        loads = [[0.0] * (n + 1) for _ in range(n)]
        costs = [[0.0] * (n + 1) for _ in range(n)]
        for start in range(n):
            load, distance, previous = 0.0, 0.0, 0
            for end in range(start + 1, n + 1):
                identifier = chromosome[end - 1]
                index = self.indices[identifier]
                load += self.demands[identifier]
                distance += matrix[previous][index]
                loads[start][end] = load
                costs[start][end] = distance + matrix[index][0]
                previous = index
        # dp[v][j]: first j customers, using a subset of the first v vehicles.
        dp = [[math.inf] * (n + 1) for _ in range(k + 1)]
        back = [[-1] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0.0
        for v, vehicle in enumerate(self.instance.vehicles, 1):
            for end in range(n + 1):
                dp[v][end] = dp[v - 1][end]  # unused vehicle
                back[v][end] = end
                for start in range(end):
                    if loads[start][end] <= vehicle.capacity:
                        cost = dp[v - 1][start] + costs[start][end]
                        if cost < dp[v][end]:
                            dp[v][end], back[v][end] = cost, start
        if not math.isfinite(dp[k][n]):
            return None
        routes = []
        end = n
        for v in range(k, 0, -1):
            start = back[v][end]
            if start != end:
                routes.append(Route(self.instance.vehicles[v - 1].id,
                                    tuple(chromosome[start:end]),
                                    loads[start][end], costs[start][end]))
            end = start
        routes.reverse()
        return routes

    def feasible(self, chromosome: Chromosome) -> bool:
        # Coverage, vehicle uniqueness, loads, valid references and depot arcs
        # follow by construction from validated permutations and the DP.
        return self.decode(chromosome) is not None

    def fitness(self, chromosome: Chromosome) -> float:
        routes = self.decode(chromosome)
        if routes is None:
            raise ValueError("No feasible route partition for this chromosome")
        return sum(route.distance for route in routes)
