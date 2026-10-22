"""Three representations using the same engine."""
import random
from dataclasses import dataclass
from ga.models import Chromosome
from ga.config import GAConfig
from ga.engine import GeneticAlgorithm
from ga.representations import BinaryRepresentation, IntegerRepresentation, FloatRepresentation
from operators.selection import TournamentSelection
from operators.crossover import MultiPointCrossover
from operators.mutation import BitFlipMutation, SwapMutation, UniformFloatMutation
from operators.replacement import ElitistReplacement


@dataclass
class BinaryProblem:
    representation: BinaryRepresentation = BinaryRepresentation(8)

    def generate(self, rng: random.Random) -> Chromosome:
        return self.representation.generate(rng)

    def feasible(self, chromosome: Chromosome) -> bool:
        return self.representation.validate(chromosome)

    def fitness(self, chromosome: Chromosome) -> float:
        return float(sum(chromosome))


@dataclass
class IntegerProblem:
    representation: IntegerRepresentation = IntegerRepresentation(6, 0, 9)

    def generate(self, rng: random.Random) -> Chromosome:
        return self.representation.generate(rng)

    def feasible(self, chromosome: Chromosome) -> bool:
        return self.representation.validate(chromosome)

    def fitness(self, chromosome: Chromosome) -> float:
        # Ordering objective; swap does not create new integer values.
        return float(sum((i + 1) * x for i, x in enumerate(chromosome)))


@dataclass
class FloatProblem:
    representation: FloatRepresentation = FloatRepresentation(3, -5.0, 5.0)

    def generate(self, rng: random.Random) -> Chromosome:
        return self.representation.generate(rng)

    def feasible(self, chromosome: Chromosome) -> bool:
        return self.representation.validate(chromosome)

    def fitness(self, chromosome: Chromosome) -> float:
        return float(sum(x * x for x in chromosome))


def main() -> None:
    for problem, mutation, minimize in [(BinaryProblem(), BitFlipMutation(), False),
                                         (IntegerProblem(), SwapMutation(), False),
                                         (FloatProblem(), UniformFloatMutation(-5.0, 5.0), True)]:
        result = GeneticAlgorithm(problem, TournamentSelection(), MultiPointCrossover(),
                                  mutation, ElitistReplacement(),
                                  GAConfig(30, 60, seed=42, minimize=minimize)).run()
        print(type(problem).__name__, result)


if __name__ == "__main__":
    main()
