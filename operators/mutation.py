"""Each operator owns its probability check; the engine never gates mutation."""
import random
from dataclasses import dataclass
from ga.models import Chromosome
from ga.representations import BinaryRepresentation, FloatRepresentation, float_bounds


def _rate(rate: float) -> None:
    if not 0 <= rate <= 1:
        raise ValueError("Mutation rate must be between zero and one")


def _occurs(rng: random.Random, rate: float) -> bool:
    # Explicit zero guarantees no mutation even with a controlled RNG returning 0.
    return rate > 0 and rng.random() <= rate


class BitFlipMutation:
    def mutate(self, chromosome: Chromosome, rng: random.Random,
               mutation_rate: float) -> Chromosome:
        _rate(mutation_rate)
        if not chromosome or not BinaryRepresentation(len(chromosome)).validate(chromosome):
            raise ValueError("Bit-flip requires binary genes")
        return tuple(1 - x if _occurs(rng, mutation_rate) else x for x in chromosome)


class SwapMutation:
    def mutate(self, chromosome: Chromosome, rng: random.Random,
               mutation_rate: float) -> Chromosome:
        _rate(mutation_rate)
        result = list(chromosome)
        if _occurs(rng, mutation_rate) and len(result) >= 2:
            a, b = rng.sample(range(len(result)), 2)
            result[a], result[b] = result[b], result[a]
        return tuple(result)


class InversionMutation:
    def mutate(self, chromosome: Chromosome, rng: random.Random,
               mutation_rate: float) -> Chromosome:
        _rate(mutation_rate)
        result = list(chromosome)
        if _occurs(rng, mutation_rate) and len(result) >= 2:
            a, b = sorted(rng.sample(range(len(result)), 2))
            result[a:b + 1] = reversed(result[a:b + 1])
        return tuple(result)


@dataclass(frozen=True)
class UniformFloatMutation:
    lower: float | tuple[float, ...]
    upper: float | tuple[float, ...]

    def mutate(self, chromosome: Chromosome, rng: random.Random,
               mutation_rate: float) -> Chromosome:
        _rate(mutation_rate)
        if not FloatRepresentation(len(chromosome), self.lower, self.upper).validate(chromosome):
            raise ValueError("Floating genes must be finite and within bounds")
        result = []
        for x, (lo, hi) in zip(chromosome, float_bounds(len(chromosome), self.lower, self.upper)):
            if _occurs(rng, mutation_rate):
                if rng.random() <= 0.5:
                    x -= rng.uniform(0, x - lo)
                else:
                    x += rng.uniform(0, hi - x)
                x = min(hi, max(lo, x))  # roundoff protection
            result.append(x)
        return tuple(result)
