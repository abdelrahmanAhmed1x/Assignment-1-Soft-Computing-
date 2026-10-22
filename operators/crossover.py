"""Lecture 4 alternating segments and cyclic order-1 crossover."""
import random
from dataclasses import dataclass
from ga.models import Chromosome
from ga.representations import validate_permutation


@dataclass(frozen=True)
class MultiPointCrossover:
    points: int = 2

    def __post_init__(self) -> None:
        if type(self.points) is not int or self.points < 1:
            raise ValueError("Crossover points must be a positive integer")

    def cross(self, parent1: Chromosome, parent2: Chromosome,
              rng: random.Random) -> tuple[Chromosome, Chromosome]:
        if len(parent1) != len(parent2) or len(parent1) <= self.points:
            raise ValueError("Equal lengths with at least points + 1 genes required")
        cuts = [0] + sorted(rng.sample(range(1, len(parent1)), self.points)) + [len(parent1)]
        child1, child2 = [], []
        for i, (start, end) in enumerate(zip(cuts, cuts[1:])):
            a, b = (parent1, parent2) if i % 2 == 0 else (parent2, parent1)
            child1.extend(a[start:end])
            child2.extend(b[start:end])
        return tuple(child1), tuple(child2)


class OrderCrossover:
    def cross(self, parent1: Chromosome, parent2: Chromosome,
              rng: random.Random) -> tuple[Chromosome, Chromosome]:
        if not validate_permutation(parent1, parent1) or not validate_permutation(parent2, parent1):
            raise ValueError("OX1 requires permutations of the same integer genes")
        if len(parent1) < 2:
            return tuple(parent1), tuple(parent2)
        left, right = sorted(rng.sample(range(len(parent1)), 2))

        def child(a: Chromosome, b: Chromosome) -> Chromosome:
            result = list(a)
            copied = set(a[left:right + 1])
            position = (right + 1) % len(a)
            for offset in range(len(b)):
                gene = b[(right + 1 + offset) % len(b)]
                if gene not in copied:
                    result[position] = gene
                    position = (position + 1) % len(a)
            return tuple(result)

        return child(parent1, parent2), child(parent2, parent1)
