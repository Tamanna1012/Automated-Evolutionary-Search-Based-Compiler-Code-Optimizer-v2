"""Builds the per-program and aggregate reports described in the
assignment's "Expected Output" and "Testing & Metrics" sections."""
from __future__ import annotations

from typing import List

from .benchmark_generator import BenchmarkProgram
from .ga import GAResult
from .pareto import describe_tradeoff, distinct_fitness_points
from .tac import tac_to_text

SELECTED_GENERATIONS = [1, 5, 10, 20, 50]


def _pick_generations(history, num_generations: int):
    picks = sorted({g for g in SELECTED_GENERATIONS if g <= num_generations} | {num_generations})
    return [(g, history[g - 1]) for g in picks]


def program_report(program: BenchmarkProgram, result: GAResult) -> dict:
    best = result.best_individual
    baseline = program.baseline_fitness
    improvement_pct = 0.0
    if baseline.exec_time:
        improvement_pct = 100.0 * (baseline.exec_time - best.fitness.exec_time) / baseline.exec_time

    front = result.final_pareto_front
    # Many genomes (different pass orders/repetitions) can land on the exact
    # same objective-space point; collapse those into one entry with a tie
    # count so the report reads as distinct trade-offs, not repeats.
    by_point = {}
    for ind in front:
        key = ind.fitness_tuple()
        by_point.setdefault(key, []).append(ind)
    pareto_entries = [
        {
            "genome": inds[0].genome,
            "fitness": inds[0].fitness.as_dict(),
            "tradeoff": describe_tradeoff(inds[0], front),
            "tied_genome_count": len(inds),
        }
        for inds in sorted(by_point.values(), key=lambda inds: inds[0].fitness_tuple())
    ]

    checkpoints = [
        {"generation": g, "best_cost": rec.best_cost, "avg_cost": rec.avg_cost}
        for g, rec in _pick_generations(result.history, len(result.history))
    ]

    return {
        "program": program.name,
        "num_test_sets": len(program.test_input_sets),
        "baseline_fitness": baseline.as_dict(),
        "checkpoints": checkpoints,
        "pareto_front": pareto_entries,
        "pareto_front_raw_size": len(front),
        "best_overall": {
            "genome": best.genome,
            "fitness": best.fitness.as_dict(),
            "cost": best.cost,
            "tac": tac_to_text(best.tac),
        },
        "improvement_pct_exec_time": improvement_pct,
        "generations_to_convergence": result.generations_to_convergence,
        "verdict": "PASS" if best.valid else "FAIL",
    }


def format_program_report_text(report: dict) -> str:
    lines = [f"=== {report['program']} ==="]
    lines.append(f"Baseline fitness: {report['baseline_fitness']}")
    lines.append("Best/avg cost at checkpoint generations:")
    for cp in report["checkpoints"]:
        lines.append(f"  gen {cp['generation']:>3}: best={cp['best_cost']:.4f}  avg={cp['avg_cost']:.4f}")
    lines.append(f"Generations to convergence: {report['generations_to_convergence']}")
    lines.append(f"Exec-time improvement vs baseline: {report['improvement_pct_exec_time']:.1f}%")
    lines.append(f"Pareto front: {report['pareto_front_raw_size']} non-dominated individuals, "
                 f"{len(report['pareto_front'])} distinct objective-space point(s):")
    for entry in report["pareto_front"]:
        tie_note = f"  ({entry['tied_genome_count']} genomes tie here)" if entry["tied_genome_count"] > 1 else ""
        lines.append(f"  genome={entry['genome']}  fitness={entry['fitness']}  "
                     f"-> {entry['tradeoff']}{tie_note}")
    lines.append("Best-overall (weighted) individual:")
    lines.append(f"  genome: {report['best_overall']['genome']}")
    lines.append(f"  fitness: {report['best_overall']['fitness']}")
    lines.append(f"  cost: {report['best_overall']['cost']:.4f}")
    lines.append("  optimized TAC:")
    for ln in report["best_overall"]["tac"].splitlines():
        lines.append(f"    {ln}")
    lines.append(f"Correctness verdict: {report['verdict']}")
    return "\n".join(lines)


def aggregate_metrics(programs: List[BenchmarkProgram], results: List[GAResult]) -> dict:
    n = len(results)
    valid_rates = [r.history[-1].valid_rate for r in results]
    pareto_sizes = [len(r.final_pareto_front) for r in results]
    distinct_sizes = [distinct_fitness_points(r.final_pareto_front) for r in results]
    convergences = [r.generations_to_convergence for r in results]

    improvements = []
    for p, r in zip(programs, results):
        base = p.baseline_fitness.exec_time
        if base and r.best_individual.valid:
            improvements.append(100.0 * (base - r.best_individual.fitness.exec_time) / base)

    pass_count = sum(1 for r in results if r.best_individual.valid)

    return {
        "num_programs": n,
        "validity_rate_pct": 100.0 * sum(valid_rates) / n if n else 0.0,
        "avg_pareto_front_size": sum(pareto_sizes) / n if n else 0.0,
        "avg_distinct_pareto_points": sum(distinct_sizes) / n if n else 0.0,
        "avg_best_cost_improvement_pct": sum(improvements) / len(improvements) if improvements else 0.0,
        "avg_generations_to_convergence": sum(convergences) / n if n else 0.0,
        "pass_rate_pct": 100.0 * pass_count / n if n else 0.0,
    }
