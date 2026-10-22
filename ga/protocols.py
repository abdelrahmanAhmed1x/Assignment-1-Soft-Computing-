"""Structural interfaces: strategies need no common base class."""
import random
from typing import Protocol
from ga.models import Chromosome, Individual


class Problem(Protocol):
    def generate(self, rng: random.Random) -> Chromosome: ...
    def fitness(self, chromosome: Chromosome) -> float: ...
    def feasible(self, chromosome: Chromosome) -> bool: ...


class Selection(Protocol):
    def select(self, population: list[Individual], minimize: bool,
               rng: random.Random) -> Individual: ...


class Crossover(Protocol):
    def cross(self, parent1: Chromosome, parent2: Chromosome,
              rng: random.Random) -> tuple[Chromosome, Chromosome]: ...


class Mutation(Protocol):
    def mutate(self, chromosome: Chromosome, rng: random.Random,
               mutation_rate: float) -> Chromosome: ...


class Replacement(Protocol):
    def replace(self, parents: list[Individual], offspring: list[Individual],
                population_size: int, minimize: bool) -> list[Individual]: ...
