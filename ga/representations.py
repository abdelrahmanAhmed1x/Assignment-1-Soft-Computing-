"""Generation and validation; bool is deliberately not an integer gene."""
import math
import random
from dataclasses import dataclass
from ga.models import Chromosome


def _length(length: int) -> None:
    if type(length) is not int or length < 1:
        raise ValueError("Chromosome length must be a positive integer")


def float_bounds(length: int, lower: float | tuple[float, ...],
                 upper: float | tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    _length(length)
    lows = lower if isinstance(lower, tuple) else (lower,) * length
    highs = upper if isinstance(upper, tuple) else (upper,) * length
    if len(lows) != length or len(highs) != length:
        raise ValueError("One bound per gene is required")
    pairs = tuple(zip(lows, highs))
    for lo, hi in pairs:
        if (type(lo) not in (int, float) or type(hi) not in (int, float)
                or not math.isfinite(lo) or not math.isfinite(hi) or lo > hi
                or not math.isfinite(hi - lo)):
            raise ValueError("Bounds must be finite, ordered, and have finite width")
    return pairs


@dataclass(frozen=True)
class BinaryRepresentation:
    length: int

    def __post_init__(self) -> None:
        _length(self.length)

    def generate(self, rng: random.Random) -> Chromosome:
        return tuple(rng.randint(0, 1) for _ in range(self.length))

    def validate(self, chromosome: Chromosome) -> bool:
        return (len(chromosome) == self.length
                and all(type(x) is int and x in (0, 1) for x in chromosome))


@dataclass(frozen=True)
class IntegerRepresentation:
    length: int
    lower: int
    upper: int

    def __post_init__(self) -> None:
        _length(self.length)
        if type(self.lower) is not int or type(self.upper) is not int or self.lower > self.upper:
            raise ValueError("Integer bounds must be ordered integers")

    def generate(self, rng: random.Random) -> Chromosome:
        return tuple(rng.randint(self.lower, self.upper) for _ in range(self.length))

    def validate(self, chromosome: Chromosome) -> bool:
        return (len(chromosome) == self.length
                and all(type(x) is int and self.lower <= x <= self.upper for x in chromosome))


@dataclass(frozen=True)
class FloatRepresentation:
    length: int
    lower: float | tuple[float, ...]
    upper: float | tuple[float, ...]

    def __post_init__(self) -> None:
        float_bounds(self.length, self.lower, self.upper)

    def generate(self, rng: random.Random) -> Chromosome:
        return tuple(rng.uniform(lo, hi) for lo, hi in float_bounds(self.length, self.lower, self.upper))

    def validate(self, chromosome: Chromosome) -> bool:
        if len(chromosome) != self.length:
            return False
        return all(type(x) in (int, float) and math.isfinite(x) and lo <= x <= hi
                   for x, (lo, hi) in zip(chromosome, float_bounds(self.length, self.lower, self.upper)))


def generate_permutation(ids: tuple[int, ...], rng: random.Random) -> Chromosome:
    if any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Permutation IDs must be unique integers")
    return tuple(rng.sample(ids, len(ids)))


def validate_permutation(chromosome: Chromosome, ids: tuple[int, ...]) -> bool:
    return (all(type(x) is int for x in ids) and len(set(ids)) == len(ids)
            and len(chromosome) == len(ids) and all(type(x) is int for x in chromosome)
            and set(chromosome) == set(ids))
