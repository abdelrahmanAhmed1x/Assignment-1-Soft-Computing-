# Soft Computing Assignment 1

A reusable genetic algorithm (GA) toolbox and a capacitated vehicle routing problem
(CVRP) application. Python 3.12 or newer, standard library only. There is one GA
engine, four strategy Protocols and one Problem Protocol. No CVRP solver library is
used. All six supplied PDFs were available and inspected, including the relevant
operator diagrams and formulas.

## Run and test

From the directory containing the PDFs:

```bash
cd ga-assignment
python3 --version
# Optional isolated environment; no packages need installing:
python3 -m venv .venv
. .venv/bin/activate
python -m examples.run_cvrp
python -m examples.run_cvrp --seeds 42 123 2026
python -m examples.simple_function
python -m unittest discover -s tests -v
```

Use a Python 3.12+ interpreter for `python3`. Validation was executed on Python
3.14.4. Running from the project directory requires no installation or network.
The CVRP default run uses 100 individuals, 500 generations, crossover rate 0.8,
mutation rate 0.1 and seed 42. Other CLI options:

```bash
python -m examples.run_cvrp --population 60 --generations 200 --mutation-rate 0.2 --crossover-rate 0.9 --seeds 123
python -m examples.run_cvrp --data data/cvrp/google_example.json
```

## Files and architecture

| Location | Responsibility |
|---|---|
| `ga/protocols.py` | Problem, Selection, Crossover, Mutation, Replacement interfaces |
| `ga/models.py` | Chromosome tuple alias, Individual and GAResult dataclasses |
| `ga/representations.py` | Binary, integer, floating-point generation/validation; permutation functions |
| `ga/config.py` | One configurable GAConfig with validated defaults |
| `ga/engine.py` | Initialization, reproduction, rejection, replacement, best tracking |
| `operators/` | The ten concrete strategies, grouped into four files |
| `problems/cvrp/` | Customer, Vehicle, CVRPInstance, Route and CVRPProblem |
| `examples/` | CVRP CLI and three small representation demonstrations |
| `tests/` | Deterministic algorithm checks and integration tests |
| `data/cvrp/` | Official example input and recorded three-seed experiments |

The engine calls `Problem.generate(rng)`, `Problem.feasible(chromosome)` and
`Problem.fitness(chromosome)`. Selection chooses evaluated parents; crossover
creates two chromosomes; mutation changes one chromosome; replacement builds the
next population. Protocols describe those methods without requiring inheritance.
The engine imports no concrete strategy and knows nothing about CVRP.

Each run creates one `random.Random(seed)` and passes it to every random operation.
It evaluates the initial population, then repeats selection, crossover, mutation,
feasibility, fitness and replacement. Both initialization and each generation have
a limit of `1000 * population_size` attempts (an offspring attempt is one parent
pair). Rejected children cause another ordinary reproduction attempt. Exhaustion
raises a clear error; it does not silently inject a different search algorithm.
Nonfinite fitness raises ValueError. A bad replacement size/evaluation is rejected.

Only feasible candidates enter populations. Odd population sizes consume only the
needed child, with no mutation/evaluation of a surplus child. The engine records
best-so-far fitness starting at generation 0, including newly evaluated offspring
before replacement. A later population can lose a good candidate under full
replacement, but the final result retains it. Ties keep the earliest best generation.

## Representations and operators

Chromosomes are immutable `tuple[int | float, ...]` values. The type alias is only a
hint; runtime validators enforce representation rules. Binary and ordinary integer
validators require actual integer genes (rejecting bool and float). Integer bounds
are inclusive and duplicates are allowed. Floating values must be finite and within
scalar or per-gene tuple bounds. Integer-valued real numbers are accepted by the
floating validator; generated values are floats. Permutations require exactly the
specified unique integer IDs. Operators never modify their parents. A tuple copy
can share an immutable object safely.

| Strategy | Lecture | Procedure and compatible representations |
|---|---|---|
| RouletteWheelSelection | 4 | Accumulate nonnegative proportional weights and spin once. Any representation with finite scalar fitness. |
| TournamentSelection | 4 | Sample two distinct population entries and choose the better fitness; size is configurable. Any representation. |
| MultiPointCrossover | 4 | Choose N distinct internal boundaries, sort, exchange alternating segments. Binary, ordinary integer, float. Default N=2. |
| OrderCrossover | 4 | OX1 copies an inclusive segment, scans the other parent cyclically after its end, skips copied genes and fills cyclically. Integer permutations only. |
| BitFlipMutation | 3 | Independently test each bit and replace 0 with 1 or 1 with 0. Binary only. |
| SwapMutation | 5 | One probability check, then swap two distinct positions. Integer permutations and other sequences, including repeated integers. |
| InversionMutation | 5 | One probability check, then reverse the inclusive segment between two distinct positions. Permutations and other sequences. |
| UniformFloatMutation | 5 | Independently test each gene, select left/right with equal probability, sample displacement within available distance to that bound. Bounded real chromosomes. |
| GenerationalReplacement | 3 | Use the evaluated offspring, with no parent survival guarantee. Any representation. |
| ElitistReplacement | 2 | Combine parents and offspring, sort and retain the best population_size individuals. Any representation. |

For example, multi-point crossover of `00|000|00` and `11|111|11` at boundaries
2 and 5 gives `00|111|00` and `11|000|11`. The lecture OX1 example copies
`4,5,6,7` from `(1,2,3,4,5,6,7,8,9)`. Scanning the second parent
`(9,3,7,8,2,6,5,1,4)` after that segment gives unused genes `1,9,3,8,2`;
the first child is `(3,8,2,4,5,6,7,1,9)`.

For floating mutation at x=4 with bounds [1,10], a left move samples displacement
in [0,3] and subtracts it. A right move samples in [0,6] and adds it. The engine
calls mutation once per child; it adds no probability gate. Bit/float probabilities
apply per gene; swap/inversion probabilities apply once to the chromosome. Rate 0
never changes a gene, and rate 1 always applies the appropriate check/action
(a swap of equal genes or a zero displacement can still leave values unchanged).

Roulette uses raw proportional fitness for nonnegative maximization. For negative
maximization, subtract the population minimum. For minimization, use
`weight = maximum_cost - cost`. Equal values or all-zero weights select uniformly.
A common normalization by maximum absolute fitness prevents overflow and leaves
these probabilities unchanged. The worst candidate may have zero weight.
Tournament selection compares raw objectives and is used for CVRP because it
handles minimization directly without this transformation.

Multi-point crossover needs equal lengths and at least N+1 genes. OX1 validates
matching sets and returns empty/singleton permutations unchanged. Swap and
inversion on fewer than two genes are harmless no-ops. Floating mutation validates
bounds at use time, allowing one scalar interval or one interval per gene.

## Source decisions and lecture ambiguities

The PDFs remain in the parent directory: the assignment and GA Lectures 1-5.
Lecture 2's simple-GA diagram and Lecture 3's canonical-GA pseudocode determine
the lifecycle. Lecture 2 discusses mixing generations but does not prescribe a
complete survivor policy. Keeping the best parents plus offspring is the explicit
elitist interpretation requested here, rather than a claim that its exact policy
appears in the slide.

Lecture 5's uniform floating-point slide opens with wording suggesting uniform
sampling across the entire interval, but its detailed directions/formulas specify
equal-probability left/right bounded displacement. These distributions differ
unless x is the midpoint. This project follows the detailed displacement algorithm
and the user's explicit requirement. Direction <=0.5 means left.

Lecture 4 describes roulette using R in [0,S-1], suited to integer fitness, and also
shows a normalized real spin. We use a real spin in [0,S) for arbitrary fitness.
Tournament sampling without replacement within one tournament is a simple choice;
individuals may participate again in later tournaments. Fitness ties keep the
first sampled contestant. Crossover/mutation comparisons use <= as taught, with
an explicit rate-zero guard to ensure a zero random draw cannot trigger rate 0.
OX1 cuts and inversion endpoints are inclusive; multi-point cuts lie between genes.

The requested homogeneous example where greedy capacity splitting fails but a
contiguous alternative succeeds is mathematically impossible with nonnegative
demands: taking the longest capacity-feasible prefix leaves no more work than
any shorter first route, and repeating this minimizes the number of routes.
We therefore test a **cost** counterexample instead: demand (1,1,1), capacity 2,
two vehicles, locations 0,1,2,3 on a line. Greedy `[1,2] [3]` costs 10; DP
`[1] [2,3]` costs 8. We also test genuinely infeasible packing and a heterogeneous
case where feasibility depends on chromosome order. No impossible test is claimed.

## CVRP data and decoding

`google_example.json` contains the assignment-linked
[Google CVRP example data](https://developers.google.com/optimization/routing/cvrp):
16 customers, depot 0, four vehicles of capacity 15, total demand 60, and the
17-by-17 distance matrix in metres. This is the official linked example, not a
substitute. Only the problem data is used; no OR-Tools code or package is imported.
The JSON has `depot`, `customers` (id/demand), `vehicles` (id/capacity), and
`distance_matrix`; `source` records provenance. Matrix order is depot first,
then customers in their listed order. Customer IDs need not equal matrix indices.

A chromosome is a permutation of customer IDs. The decoder precomputes every
contiguous segment's demand and depot-to-segment-to-depot distance. Then
`dp[v][j]` is the cheapest way to serve the first j customers using a subset of
the first v vehicles. A transition either skips vehicle v or assigns it a
capacity-feasible segment ending at j. Backtracking constructs actual Route
objects. Complexity is O(K*N^2) time and O(N^2+K*N) space.

For a homogeneous fleet this finds the minimum-distance contiguous split using
at most K vehicles. Heterogeneous capacities are respected in vehicle input order;
vehicles can be skipped. It does not search every permutation of heterogeneous
vehicle assignments for a fixed chromosome. Evolution searches customer orders.
A valid permutation can still have no feasible split under this decoder model.

Routes store customer sequences without depot sentinels: both depot arcs are
included in every route distance and displayed explicitly in the CLI. Every
customer belongs to exactly one segment; each vehicle appears at most once.
Fitness is the sum of decoded route distances. Infeasible chromosomes have no
fitness and raise ValueError when evaluated directly. Feasibility and fitness
each decode independently for simplicity; there is no cache.

Instance validation checks unique integer IDs, positive finite capacities,
nonnegative finite demands/distances, square dimensions, individual demand versus
maximum capacity, and total demand versus total capacity (also a sound check for
heterogeneous fleets). These checks do not prove packability. For example,
three demands of 4 cannot fit into two vehicles of capacity 7. The decoder rejects
such orders; initialization eventually reports its retry-limit error. This reports
failure to find a feasible population, not a mathematical infeasibility certificate.

## Reuse the engine

A new problem needs just three methods; it does not inherit a base class:

```python
import random
from ga.models import Chromosome
from ga.representations import BinaryRepresentation
from ga.config import GAConfig
from ga.engine import GeneticAlgorithm
from operators.selection import RouletteWheelSelection
from operators.crossover import MultiPointCrossover
from operators.mutation import BitFlipMutation
from operators.replacement import GenerationalReplacement

class CountOnes:
    representation = BinaryRepresentation(8)

    def generate(self, rng: random.Random) -> Chromosome:
        return self.representation.generate(rng)

    def feasible(self, chromosome: Chromosome) -> bool:
        return self.representation.validate(chromosome)

    def fitness(self, chromosome: Chromosome) -> float:
        return float(sum(chromosome))

result = GeneticAlgorithm(
    CountOnes(), RouletteWheelSelection(), MultiPointCrossover(2),
    BitFlipMutation(), GenerationalReplacement(),
    GAConfig(population_size=30, max_generations=60, minimize=False, seed=42),
).run()
print(result.best_chromosome, result.best_fitness)
```

`examples.simple_function` also demonstrates repeated bounded integer genes with
an ordering objective and swap mutation, plus real sphere minimization with
multi-point crossover and bounded displacement mutation. Swap rearranges existing
integer values; it cannot introduce arbitrary new values. Crossover can recombine
values present in parents. There is deliberately no fifth mutation strategy.

## Measured results

All runs use the unchanged defaults, 100 individuals and 500 generations.
`data/cvrp/experiments.json` records configurations, full 501-entry histories,
chromosomes, routes and feasibility. Tests independently recompute route costs,
coverage, capacity and best-generation consistency from these records.

| Seed | Initial distance | Final distance | Improvement | Best generation | Routes / feasible |
|---|---:|---:|---:|---:|---|
| 42 | 7804 | 6208 | 20.5% | 19 | 4 / yes |
| 123 | 9244 | 6208 | 32.8% | 46 | 4 / yes |
| 2026 | 9132 | 6208 | 32.0% | 31 | 4 / yes |

Seed 42 final routes (depot 0 at both ends):

| Vehicle | Route | Load / capacity | Distance (m) |
|---|---|---|---:|
| 0 | 0 -> 14 -> 16 -> 10 -> 9 -> 0 | 15 / 15 | 1552 |
| 1 | 0 -> 5 -> 6 -> 2 -> 8 -> 0 | 15 / 15 | 1552 |
| 2 | 0 -> 7 -> 3 -> 4 -> 1 -> 0 | 15 / 15 | 1552 |
| 3 | 0 -> 13 -> 15 -> 11 -> 12 -> 0 | 15 / 15 | 1552 |

Total distance: **6208 m**. Total load: **60**. All three seeds match the
linked page's published 6208 m example solution. This is a comparison with a
published feasible solution, not an independent proof of the global optimum.
No hyperparameter tuning or local search was needed.

## Grading coverage audit

Statuses refer to verified implementation, not a prediction of awarded marks.
There are 34 passing unittest methods; several check many inputs/seeds. The only
requested test replaced by an alternative is the impossible homogeneous greedy
feasibility counterexample discussed above.

| Grading criterion | Implementation location | Relevant tests | Status |
|---|---|---|---|
| Toolbox design and structure (6) | `ga/protocols.py`, `ga/engine.py`, package layout | `EngineTests.test_injected_strategies_odd_size_and_crossover_rates`, `test_three_representations_reproducibility_and_history`; import audit | Verified: one engine, four strategy Protocols, one Problem; no concrete/CVRP imports in engine |
| Chromosome representations (6) | `ga/representations.py`, `examples/simple_function.py` | `RepresentationTests` (all 5); engine three-representation test | Verified: binary, repeated integers, bounded finite reals and permutation utilities |
| GA configuration and implementation (8) | `ga/config.py`, `ga/engine.py`, `ga/models.py` | `EngineTests` (all 7) | Verified: parameters, both directions, odd sizes, rejection, retries, finite fitness, history, best retention and reproducibility |
| Selection (10) | `operators/selection.py` | `SelectionTests` (all 3) | Verified: roulette known intervals/frequencies, equal/zero/negative fitness and pairwise tournament in both directions |
| Crossover (10) | `operators/crossover.py` | `CrossoverTests` (all 3) | Verified: alternating segments, exact lecture OX1 wraparound, validity, immutability and short/invalid inputs |
| Mutation (12) | `operators/mutation.py` | `MutationTests` (all 4) | Verified: four methods, exact displacement arguments/results, probability once per gene/chromosome, bounds, rates 0/1 and immutability |
| Replacement (8) | `operators/replacement.py` | `ReplacementTests` (both); engine best-retention tests | Verified: full offspring replacement and combined elitist sorting, both directions and constant size |
| Problem modeling and decoding (10) | `problems/cvrp/models.py`, `problem.py` | `CVRPTests.test_dp_matches_all_contiguous_partitions`, `test_dp_improves_greedy_cost`, `test_heterogeneous_skip_and_order_limitation` | Verified: dataclasses, customer permutation, reconstructable minimum-cost split; documented heterogeneous vehicle-order limit |
| Constraint handling (10) | `CVRPInstance.__post_init__`, `CVRPProblem.feasible/decode` | CVRP coverage, validation, unsatisfiable packing and seeded tests; `assert_routes` independently checks every route | Verified: IDs, coverage, finite inputs, capacity, fleet, both depot arcs and infeasible outcomes; impossible requested greedy test replaced explicitly |
| Fitness evaluation (10) | `CVRPProblem.fitness/decode` | CVRP manual optimum, asymmetric distances, exhaustive partition comparison and official 6208-cost check | Verified: exact sum of all route arcs including depot departure/return; invalid orders raise |
| Correct toolbox usage (6) | `examples/run_cvrp.py:solve`, CVRP Problem methods | CVRP seeded engine execution; engine reuse test; import audit | Verified: CVRP injects Tournament/OX1/Inversion/Elitist into the same engine |
| Performance and solution quality (4) | `examples/run_cvrp.py`, `data/cvrp/experiments.json` | `CVRPTests.test_saved_three_seed_experiments`, three executed full runs | Verified: all seeds improve, 6208 final cost, 500 generations, feasible routes and loads recorded |

The tests establish the small line instance's optimum by enumerating its six
customer permutations and compare DP against all contiguous splits on small
homogeneous and heterogeneous instances. They do not exhaustively solve the
16-customer benchmark. Sparse feasible spaces can exhaust bounded retries even
when an instance has a solution; the resulting error says so. No submission zip
is generated because the required TA/group/member naming details were not supplied.
