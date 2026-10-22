"""Full offspring replacement and best-of-parents-plus-offspring mixing."""
import math
from ga.models import Individual


def _check(candidates: list[Individual], size: int) -> None:
    if size < 1 or len(candidates) < size:
        raise ValueError("Not enough candidates for replacement")
    if any(not x.feasible or not math.isfinite(x.fitness) for x in candidates):
        raise ValueError("Replacement requires feasible, finite evaluations")


class GenerationalReplacement:
    def replace(self, parents: list[Individual], offspring: list[Individual],
                population_size: int, minimize: bool) -> list[Individual]:
        _check(offspring, population_size)
        return offspring[:population_size]


class ElitistReplacement:
    def replace(self, parents: list[Individual], offspring: list[Individual],
                population_size: int, minimize: bool) -> list[Individual]:
        candidates = parents + offspring
        _check(candidates, population_size)
        return sorted(candidates, key=lambda x: x.fitness, reverse=not minimize)[:population_size]
