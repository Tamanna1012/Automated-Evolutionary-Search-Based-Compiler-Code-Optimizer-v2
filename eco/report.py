"""Builds the per-program and aggregate reports described in the
assignment's "Expected Output" and "Testing & Metrics" sections."""
from __future__ import annotations

import csv
import os
from typing import List

from .benchmark_generator import BenchmarkProgram
from .ga import GAResult
from .pareto import describe_tradeoff, distinct_fitness_points
from .tac import tac_to_text

SELECTED_GENERATIONS = [1, 5, 10, 20, 50]


def _pick_generations(history, num_generations: int):
    picks = sorted({g for g in SELECTED_GENERATIONS if g <= num_generations} | {num_generations})
    return [(g, history[g - 1]) for g in picks]


def program_report(program: BenchmarkProgram, result: GAResult, runtime: dict = None) -> dict:
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
        "kernel": getattr(program, "kernel", "synthetic"),
        "source_code": program.source_code,
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
        "runtime": runtime,
        "generations_to_convergence": result.generations_to_convergence,
        "verdict": "PASS" if best.valid else "FAIL",
    }


def format_program_report_text(report: dict) -> str:
    lines = [f"=== {report['program']} ({report['kernel']}) ==="]
    lines.append("Original source:")
    lines.extend(f"    {ln}" for ln in report["source_code"].splitlines())
    lines.append(f"Baseline fitness: {report['baseline_fitness']}")
    lines.append("Best/avg cost at checkpoint generations:")
    for cp in report["checkpoints"]:
        lines.append(f"  gen {cp['generation']:>3}: best={cp['best_cost']:.4f}  avg={cp['avg_cost']:.4f}")
    lines.append(f"Generations to convergence: {report['generations_to_convergence']}")
    lines.append(f"Simulated exec-time improvement vs baseline (cycle model): {report['improvement_pct_exec_time']:.1f}%")
    rt = report.get("runtime")
    if rt:
        lines.append(f"Measured runtime (real, Python backend): baseline {rt['baseline_ns']:.0f} ns/call -> "
                     f"best {rt['best_ns']:.0f} ns/call = {rt['measured_improvement_pct']:.1f}% faster "
                     f"({rt['measured_speedup_x']:.2f}x); optimized outputs correct: "
                     f"{'yes' if rt['outputs_correct'] else 'NO'}")
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


def aggregate_metrics(programs: List[BenchmarkProgram], results: List[GAResult],
                      runtimes: dict = None, correlations: dict = None) -> dict:
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

    measured = {}
    if runtimes:
        rts = [runtimes[p.name] for p in programs]
        imps = [r["measured_improvement_pct"] for r in rts]
        speedups = [r["measured_speedup_x"] for r in rts]
        total_base = sum(r["baseline_ns"] for r in rts)
        total_best = sum(r["best_ns"] for r in rts)
        measured = {
            "avg_measured_improvement_pct": sum(imps) / len(imps),
            "avg_measured_speedup_x": sum(speedups) / len(speedups),
            "total_time_speedup_x": total_base / total_best if total_best else 1.0,
            "measured_outputs_correct_pct": 100.0 * sum(1 for r in rts if r["outputs_correct"]) / len(rts),
            "programs_measurably_faster_pct":
                100.0 * sum(1 for r in rts if r["measured_improvement_pct"] > 2.0) / len(rts),
        }
    if correlations:
        measured["correlation"] = correlations

    return {
        **measured,
        "num_programs": n,
        "validity_rate_pct": 100.0 * sum(valid_rates) / n if n else 0.0,
        "avg_pareto_front_size": sum(pareto_sizes) / n if n else 0.0,
        "avg_distinct_pareto_points": sum(distinct_sizes) / n if n else 0.0,
        "avg_best_cost_improvement_pct": sum(improvements) / len(improvements) if improvements else 0.0,
        "avg_generations_to_convergence": sum(convergences) / n if n else 0.0,
        "pass_rate_pct": 100.0 * pass_count / n if n else 0.0,
    }


SUMMARY_COLUMNS = [
    "program", "kernel", "baseline_cost", "best_cost", "baseline_exec_time", "best_exec_time",
    "improvement_pct", "measured_baseline_ns", "measured_best_ns", "measured_improvement_pct",
    "measured_speedup_x", "generations_to_convergence", "pareto_front_size",
    "distinct_points", "verdict",
]


def write_all_program_reports(programs: List[BenchmarkProgram], results: List[GAResult],
                              out_dir: str, runtimes: dict = None) -> List[dict]:
    """Write one detailed text report per program plus ``all_programs_summary.csv``.

    ``baseline_cost`` / ``best_cost`` are the weighted, baseline-normalized cost
    the GA minimizes (so the baseline is 1.0 by construction);
    ``improvement_pct`` is the exec_time improvement used everywhere else.
    Returns the summary rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for program, result in zip(programs, results):
        rt = (runtimes or {}).get(program.name)
        report = program_report(program, result, rt)
        with open(os.path.join(out_dir, f"{program.name}_report.txt"), "w", encoding="utf-8") as f:
            f.write(format_program_report_text(report))
        rows.append({
            "program": program.name,
            "kernel": report["kernel"],
            "baseline_cost": 1.0,
            "best_cost": round(report["best_overall"]["cost"], 6),
            "baseline_exec_time": round(report["baseline_fitness"]["exec_time"], 4),
            "best_exec_time": round(report["best_overall"]["fitness"]["exec_time"], 4),
            "improvement_pct": round(report["improvement_pct_exec_time"], 2),
            "measured_baseline_ns": round(rt["baseline_ns"], 1) if rt else "",
            "measured_best_ns": round(rt["best_ns"], 1) if rt else "",
            "measured_improvement_pct": round(rt["measured_improvement_pct"], 2) if rt else "",
            "measured_speedup_x": round(rt["measured_speedup_x"], 3) if rt else "",
            "generations_to_convergence": report["generations_to_convergence"],
            "pareto_front_size": report["pareto_front_raw_size"],
            "distinct_points": len(report["pareto_front"]),
            "verdict": report["verdict"],
        })
    with open(os.path.join(out_dir, "all_programs_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
