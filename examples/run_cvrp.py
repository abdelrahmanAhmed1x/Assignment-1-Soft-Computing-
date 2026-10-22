"""Run the assignment-linked data; no solver libraries are used."""
import argparse
import json
from pathlib import Path
from ga.config import GAConfig
from ga.models import GAResult
from ga.engine import GeneticAlgorithm
from operators.selection import TournamentSelection
from operators.crossover import OrderCrossover
from operators.mutation import InversionMutation
from operators.replacement import ElitistReplacement
from problems.cvrp.models import Customer, Vehicle, CVRPInstance
from problems.cvrp.problem import CVRPProblem

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/cvrp/google_example.json"


def load_instance(path: Path = DEFAULT_DATA) -> CVRPInstance:
    data = json.loads(path.read_text())
    return CVRPInstance(data["depot"], tuple(Customer(**c) for c in data["customers"]),
                        tuple(Vehicle(**v) for v in data["vehicles"]),
                        tuple(tuple(row) for row in data["distance_matrix"]))


def solve(problem: CVRPProblem, config: GAConfig) -> GAResult:
    return GeneticAlgorithm(problem, TournamentSelection(size=2), OrderCrossover(),
                            InversionMutation(), ElitistReplacement(), config).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    args = parser.parse_args()
    try:
        problem = CVRPProblem(load_instance(args.data))
        for seed in args.seeds:
            config = GAConfig(args.population, args.generations, args.crossover_rate,
                              args.mutation_rate, True, seed)
            result = solve(problem, config)
            routes = problem.decode(result.best_chromosome)
            if routes is None:
                raise RuntimeError("Best result could not be decoded")
            print("=" * 40 + "\nCAPACITATED VEHICLE ROUTING PROBLEM\n" + "=" * 40)
            print(f"Seed: {seed}\nPopulation Size: {config.population_size}")
            print(f"Maximum Generations: {config.max_generations}\nTotal Generations: {result.generations}")
            print("Selection: Tournament (2)\nCrossover: Order-1\nMutation: Inversion\nReplacement: Elitist")
            print(f"Initial Best Distance: {result.fitness_history[0]:g}")
            print(f"Final Best Distance: {result.best_fitness:g}\nBest Generation: {result.best_generation}")
            capacities = {v.id: v.capacity for v in problem.instance.vehicles}
            for route in routes:
                print(f"\nVehicle {route.vehicle_id}:")
                print("  Depot -> " + " -> ".join(map(str, route.customers)) + " -> Depot")
                print(f"  Load: {route.load:g} / {capacities[route.vehicle_id]:g}")
                print(f"  Distance: {route.distance:g}")
            print(f"\nTotal Distance: {sum(r.distance for r in routes):g}")
            print(f"Total Load: {sum(r.load for r in routes):g}\nFeasible: YES\n")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"CVRP error: {error}\n")


if __name__ == "__main__":
    main()
