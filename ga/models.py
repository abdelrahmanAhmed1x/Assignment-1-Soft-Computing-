"""Small shared evaluation and result models."""
from dataclasses import dataclass
from typing import TypeAlias

Chromosome: TypeAlias = tuple[int | float, ...]


@dataclass
class Individual:
    chromosome: Chromosome
    fitness: float
    feasible: bool


@dataclass
class GAResult:
    best_chromosome: Chromosome
    best_fitness: float
    generations: int
    fitness_history: list[float]
    best_generation: int
