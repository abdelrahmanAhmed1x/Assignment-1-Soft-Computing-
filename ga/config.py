from dataclasses import dataclass


@dataclass
class GAConfig:
    population_size: int = 100
    max_generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    minimize: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.population_size) is not int or self.population_size < 2:
            raise ValueError("Population size must be an integer >= 2")
        if type(self.max_generations) is not int or self.max_generations <= 0:
            raise ValueError("Maximum generations must be a positive integer")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("Crossover and mutation rates must be in [0, 1]")
        if type(self.minimize) is not bool:
            raise ValueError("minimize must be a boolean")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")
