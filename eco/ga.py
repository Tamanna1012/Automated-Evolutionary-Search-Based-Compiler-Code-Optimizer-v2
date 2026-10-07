"""The evolutionary search engine: population init, fitness evaluation,
tournament selection + elitism, mutation, crossover and the generational
loop, per the assignment's Evolutionary Algorithm Design section."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .individual import Individual
from .metrics import evaluate
from .optimizations import OPTIMIZATION_NAMES, apply_genome
from .pareto import pareto_front
from .tac import TAC

DEFAULT_WEIGHTS = {"exec_time": 0.25, "instr_count": 0.25, "code_size": 0.25, "arith_ops": 0.25}
INVALID_PENALTY = 1e6
MAX_GENOME_LEN = 5


@dataclass
class GAConfig:
    pop_size: int = 30
    num_generations: int = 30
    elitism_frac: float = 0.10
    tournament_k: int = 3
    mutation_add_prob: float = 0.7
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    seed: int = 0


@dataclass
class GenerationRecord:
    generation: int
    best_cost: float
    avg_cost: float
    pareto_front_size: int
    valid_rate: float
    elite_genomes: List[List[str]]
    costs: List[float] = field(default_factory=list)  # whole-population costs, for distribution plots


@dataclass
class GAResult:
    history: List[GenerationRecord]
    final_population: List[Individual]
    best_individual: Individual
    final_pareto_front: List[Individual]
    generations_to_convergence: int
    population_snapshots: List[List[Individual]] = field(default_factory=list)


def _random_genome(rng: random.Random) -> List[str]:
    length = rng.randint(1, len(OPTIMIZATION_NAMES))
    return rng.sample(OPTIMIZATION_NAMES, length)


def _mutate(genome: Sequence[str], rng: random.Random, add_prob: float) -> List[str]:
    genome = list(genome)
    if rng.random() < add_prob:
        if len(genome) < MAX_GENOME_LEN:
            pos = rng.randint(0, len(genome))
            genome.insert(pos, rng.choice(OPTIMIZATION_NAMES))
    else:
        if genome:
            genome.pop(rng.randrange(len(genome)))
    return genome


def _crossover(g1: Sequence[str], g2: Sequence[str], rng: random.Random) -> List[str]:
    if not g1 or not g2:
        return list(g1) if g1 else list(g2)
    cut = rng.randint(0, min(len(g1), len(g2)))
    child = list(g1[:cut]) + list(g2[cut:])
    return child[:MAX_GENOME_LEN]


def _tournament_select(population: List[Individual], rng: random.Random, k: int) -> Individual:
    contenders = rng.sample(population, min(k, len(population)))
    return min(contenders, key=lambda ind: ind.cost)


def build_individual(original_tac: TAC, genome: List[str], test_input_sets, expected_outputs,
                      baseline_fitness, weights: Dict[str, float]) -> Individual:
    tac = apply_genome(original_tac, genome)
    result = evaluate(tac, test_input_sets, expected_outputs)
    ind = Individual(genome=genome, tac=tac, fitness=result.fitness, valid=result.valid)
    ind.cost = weighted_cost(ind, baseline_fitness, weights)
    return ind


def weighted_cost(ind: Individual, baseline_fitness, weights: Dict[str, float]) -> float:
    if not ind.valid:
        return INVALID_PENALTY
    bt = baseline_fitness.as_tuple()
    it = ind.fitness_tuple()
    names = ["exec_time", "instr_count", "code_size", "arith_ops"]
    total = 0.0
    for name, base_v, v in zip(names, bt, it):
        base_v = base_v if base_v else 1.0
        total += weights[name] * (v / base_v)
    return total


def run_ga(original_tac: TAC, test_input_sets, expected_outputs, config: GAConfig = None) -> GAResult:
    config = config or GAConfig()
    rng = random.Random(config.seed)

    baseline_eval = evaluate(original_tac, test_input_sets, expected_outputs)
    baseline_fitness = baseline_eval.fitness

    def make(genome: List[str]) -> Individual:
        return build_individual(original_tac, genome, test_input_sets, expected_outputs,
                                 baseline_fitness, config.weights)

    population: List[Individual] = [make([])]  # original always kept
    while len(population) < config.pop_size:
        population.append(make(_random_genome(rng)))

    n_elite = max(1, math.ceil(config.pop_size * config.elitism_frac))
    history: List[GenerationRecord] = []
    population_snapshots: List[List[Individual]] = []
    best_cost_seen = math.inf
    generations_to_convergence = 0

    for gen in range(config.num_generations):
        population.sort(key=lambda ind: ind.cost)
        valid_pop = [ind for ind in population if ind.valid]
        costs = [ind.cost for ind in valid_pop] or [ind.cost for ind in population]
        best_cost = min(costs)
        avg_cost = sum(costs) / len(costs)
        front = pareto_front(valid_pop) if valid_pop else []
        elites = population[:n_elite]
        population_snapshots.append(list(population))

        history.append(GenerationRecord(
            generation=gen + 1,
            best_cost=best_cost,
            avg_cost=avg_cost,
            pareto_front_size=len(front),
            valid_rate=len(valid_pop) / len(population),
            elite_genomes=[list(e.genome) for e in elites],
            costs=[ind.cost for ind in population],
        ))

        if best_cost < best_cost_seen - 1e-9:
            best_cost_seen = best_cost
            generations_to_convergence = gen + 1

        if gen == config.num_generations - 1:
            break  # no need to breed a next generation after the last eval

        next_population: List[Individual] = list(elites)
        while len(next_population) < config.pop_size:
            p1 = _tournament_select(population, rng, config.tournament_k)
            p2 = _tournament_select(population, rng, config.tournament_k)
            g1 = _mutate(p1.genome, rng, config.mutation_add_prob)
            g2 = _mutate(p2.genome, rng, config.mutation_add_prob)
            child_genome = _crossover(g1, g2, rng)
            next_population.append(make(child_genome))
        population = next_population

    population.sort(key=lambda ind: ind.cost)
    valid_final = [ind for ind in population if ind.valid]
    best_individual = min(valid_final, key=lambda ind: ind.cost) if valid_final else population[0]
    final_front = pareto_front(valid_final) if valid_final else []

    return GAResult(
        history=history,
        final_population=population,
        best_individual=best_individual,
        final_pareto_front=final_front,
        generations_to_convergence=generations_to_convergence,
        population_snapshots=population_snapshots,
    )
