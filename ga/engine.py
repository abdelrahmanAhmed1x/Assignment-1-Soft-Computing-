"""One canonical GA for all problems and representations."""
import math
import random
from ga.config import GAConfig
from ga.models import Chromosome, GAResult, Individual
from ga.protocols import Crossover, Mutation, Problem, Replacement, Selection


class GeneticAlgorithm:
    def __init__(self, problem: Problem, selection: Selection, crossover: Crossover,
                 mutation: Mutation, replacement: Replacement, config: GAConfig) -> None:
        self.problem = problem
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.config = config

    def _evaluate(self, chromosome: Chromosome) -> Individual | None:
        if not self.problem.feasible(chromosome):
            return None
        fitness = self.problem.fitness(chromosome)
        if not math.isfinite(fitness):
            raise ValueError("Problem returned nonfinite fitness")
        return Individual(chromosome, fitness, True)

    def run(self) -> GAResult:
        self.config.__post_init__()  # validate even if caller edited the dataclass
        rng = random.Random(self.config.seed)
        size = self.config.population_size
        limit = 1000 * size
        population = []
        for _ in range(limit):
            individual = self._evaluate(self.problem.generate(rng))
            if individual is not None:
                population.append(individual)
            if len(population) == size:
                break
        else:
            raise RuntimeError("Initialization retry limit reached: cannot fill a feasible population")
        choose = min if self.config.minimize else max
        best = choose(population, key=lambda x: x.fitness)
        history = [best.fitness]
        best_generation = 0
        for generation in range(1, self.config.max_generations + 1):
            offspring = []
            for _ in range(limit):
                p1 = self.selection.select(population, self.config.minimize, rng)
                p2 = self.selection.select(population, self.config.minimize, rng)
                draw = rng.random()
                if self.config.crossover_rate > 0 and draw <= self.config.crossover_rate:
                    children = self.crossover.cross(p1.chromosome, p2.chromosome, rng)
                else:
                    children = (tuple(p1.chromosome), tuple(p2.chromosome))
                for chromosome in children:
                    mutated = self.mutation.mutate(chromosome, rng, self.config.mutation_rate)
                    individual = self._evaluate(mutated)
                    if individual is not None:
                        offspring.append(individual)
                    if len(offspring) == size:
                        break
                if len(offspring) == size:
                    break
            else:
                raise RuntimeError(f"Offspring retry limit reached at generation {generation}")
            # Inspect offspring before replacement so even a custom policy cannot lose a record.
            candidate = choose(offspring + [best], key=lambda x: x.fitness)
            if candidate.fitness != best.fitness:
                best, best_generation = candidate, generation
            population = self.replacement.replace(population, offspring, size, self.config.minimize)
            if len(population) != size or any(not x.feasible or not math.isfinite(x.fitness) for x in population):
                raise ValueError("Replacement violated the population contract")
            history.append(best.fitness)
        return GAResult(best.chromosome, best.fitness, self.config.max_generations,
                        history, best_generation)
