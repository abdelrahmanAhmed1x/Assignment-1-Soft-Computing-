"""Lecture 4 selection strategies."""
import math
import random
from dataclasses import dataclass
from ga.models import Individual


def _validate(population: list[Individual]) -> None:
    if not population or any(not x.feasible or not math.isfinite(x.fitness) for x in population):
        raise ValueError("Selection requires a nonempty, feasible, finite population")


class RouletteWheelSelection:
    def select(self, population: list[Individual], minimize: bool,
               rng: random.Random) -> Individual:
        _validate(population)
        values = [x.fitness for x in population]
        # Scale first to prevent overflow; positive maximization stays proportional.
        scale = max(abs(x) for x in values) or 1.0
        values = [x / scale for x in values]
        if minimize:
            worst = max(values)
            weights = [worst - x for x in values]
        else:
            shift = min(0.0, min(values))
            weights = [x - shift for x in values]
        total = sum(weights)
        if total == 0 or max(values) == min(values):
            return rng.choice(population)
        target = rng.random() * total
        cumulative = 0.0
        for individual, weight in zip(population, weights):
            cumulative += weight
            if target < cumulative:
                return individual
        return population[max(i for i, w in enumerate(weights) if w > 0)]


@dataclass(frozen=True)
class TournamentSelection:
    size: int = 2

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size < 2:
            raise ValueError("Tournament size must be at least two")

    def select(self, population: list[Individual], minimize: bool,
               rng: random.Random) -> Individual:
        _validate(population)
        if self.size > len(population):
            raise ValueError("Tournament size exceeds population")
        contestants = rng.sample(population, self.size)
        return (min if minimize else max)(contestants, key=lambda x: x.fitness)
