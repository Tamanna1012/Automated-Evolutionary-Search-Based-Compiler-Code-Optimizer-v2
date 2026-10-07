"""End-to-end experiment driver.

1. Generates the benchmark dataset (200-500 synthetic TAC programs).
2. Runs the full evolutionary search (with per-generation logging) on a
   "deep-dive" subset of programs, producing the detailed per-program
   reports the assignment asks for.
3. Runs the same search across the *entire* dataset to compute the
   aggregate Testing & Metrics table (validity rate, Pareto front size,
   best-cost improvement, generations to convergence).
4. Renders all six required plots and writes text reports to disk.
"""
from __future__ import annotations

import json
import os
import time

from .dataset import generate_dataset, dataset_summary
from .ga import GAConfig, run_ga
from .pareto import distinct_fitness_points, pareto_front
from .report import aggregate_metrics, format_program_report_text, program_report
from .visualize import (
    plot_convergence,
    plot_elite_heatmap,
    plot_elite_heatmap_multi,
    plot_fitness_boxplot,
    plot_improvement_bar,
    plot_multi_convergence,
    plot_pareto_2d,
    plot_pareto_3d,
)

HERE = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(HERE, "outputs", "plots")
REPORTS_DIR = os.path.join(HERE, "outputs", "reports")

N_PROGRAMS = 300
DEEP_DIVE_N = 20
POP_SIZE = 30
NUM_GENERATIONS = 30
SEED = 42


def main():
    t_start = time.time()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Generating {N_PROGRAMS} benchmark programs...")
    programs = generate_dataset(N_PROGRAMS, seed=SEED)
    print("  ", dataset_summary(programs))

    config = GAConfig(pop_size=POP_SIZE, num_generations=NUM_GENERATIONS, seed=SEED)

    print(f"Running evolutionary search on all {N_PROGRAMS} programs "
          f"(pop={POP_SIZE}, gens={NUM_GENERATIONS})...")
    t0 = time.time()
    results = [run_ga(p.tac, p.test_input_sets, p.expected_outputs, config) for p in programs]
    print(f"  done in {time.time() - t0:.1f}s")

    # Deep-dive subset: the 20 largest baseline programs, so there's the
    # most room to observe optimizations and trade-offs firing.
    order = sorted(range(len(programs)), key=lambda i: -programs[i].baseline_fitness.instr_count)
    deep_idx = order[:DEEP_DIVE_N]
    deep_programs = [programs[i] for i in deep_idx]
    deep_results = [results[i] for i in deep_idx]
    flagship_i = deep_idx[0]
    flagship_program, flagship_result = programs[flagship_i], results[flagship_i]

    print(f"Flagship program: {flagship_program.name} "
          f"(baseline instr_count={flagship_program.baseline_fitness.instr_count})")

    # ---- Per-program reports (deep-dive subset) ----
    print(f"Writing per-program reports for {DEEP_DIVE_N} deep-dive programs...")
    all_reports = []
    for p, r in zip(deep_programs, deep_results):
        rep = program_report(p, r)
        all_reports.append(rep)
        with open(os.path.join(REPORTS_DIR, f"{p.name}_report.txt"), "w") as f:
            f.write(format_program_report_text(rep))
    with open(os.path.join(REPORTS_DIR, "deep_dive_reports.json"), "w") as f:
        json.dump(all_reports, f, indent=2)

    # ---- Aggregate metrics across the full dataset ----
    print("Computing aggregate Testing & Metrics table...")
    agg = aggregate_metrics(programs, results)
    with open(os.path.join(REPORTS_DIR, "aggregate_metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)

    summary_lines = [
        "# Aggregate Testing & Metrics",
        f"- Programs evaluated: {agg['num_programs']}",
        f"- Validity Rate: {agg['validity_rate_pct']:.1f}%  (target > 90%)",
        f"- Avg. Pareto Front Size: {agg['avg_pareto_front_size']:.2f} "
        f"(avg. {agg['avg_distinct_pareto_points']:.2f} distinct objective-space points)",
        f"- Best Cost Improvement (avg exec_time vs baseline): "
        f"{agg['avg_best_cost_improvement_pct']:.1f}%  (target > 30%)",
        f"- Avg. Generations to Convergence: {agg['avg_generations_to_convergence']:.1f}",
        f"- Correctness PASS rate: {agg['pass_rate_pct']:.1f}%",
    ]
    with open(os.path.join(REPORTS_DIR, "aggregate_metrics.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

    # ---- Plots ----
    print("Rendering plots...")
    plot_convergence(flagship_result, os.path.join(PLOTS_DIR, "1_convergence_curve.png"),
                      title=f"Convergence Curve — {flagship_program.name}")
    plot_multi_convergence({p.name: r for p, r in zip(deep_programs, deep_results)},
                            os.path.join(PLOTS_DIR, "1b_convergence_curves_20programs.png"))

    # For the 2D/3D Pareto scatter plots we want a snapshot with genuine
    # trade-off variety. Because all six optimizations here are simplifying
    # (they never trade one objective for another), a fully-converged final
    # population collapses onto a single dominant optimum - real variety is
    # visible *mid-search*, before elitism converges everyone onto it. Scan
    # the deep-dive programs' early generations and pick whichever snapshot
    # has the richest (most distinct-point) Pareto front to plot.
    best_choice = None  # (distinct_points, program, gen_idx, population, front)
    for p, r in zip(deep_programs, deep_results):
        for gen_idx, snap in enumerate(r.population_snapshots[:10]):
            valid_snap = [ind for ind in snap if ind.valid]
            front = pareto_front(valid_snap)
            score = distinct_fitness_points(front)
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, p, gen_idx, valid_snap, front)
    _, pareto_program, pareto_gen_idx, pareto_pop, pareto_front_snap = best_choice
    print(f"  Pareto scatter snapshot: {pareto_program.name}, generation {pareto_gen_idx + 1} "
          f"({best_choice[0]} distinct objective-space points on the front)")

    plot_pareto_2d(pareto_pop, pareto_front_snap, os.path.join(PLOTS_DIR, "2_pareto_front_2d.png"),
                    title=f"Pareto Front (exec_time vs instr_count) — {pareto_program.name}, "
                          f"gen {pareto_gen_idx + 1}")
    plot_fitness_boxplot(flagship_result, os.path.join(PLOTS_DIR, "3_fitness_boxplot.png"),
                          title=f"Cost Distribution Across Generations — {flagship_program.name}")

    improvement_pcts = [rep["improvement_pct_exec_time"] for rep in all_reports]
    plot_improvement_bar([p.name for p in deep_programs], improvement_pcts,
                          os.path.join(PLOTS_DIR, "4_improvement_bar.png"))

    plot_elite_heatmap_multi(deep_results, os.path.join(PLOTS_DIR, "5_elite_optimization_heatmap.png"))
    plot_pareto_3d(pareto_pop, pareto_front_snap, os.path.join(PLOTS_DIR, "6_pareto_front_3d.png"),
                   title=f"Pareto Front 3D — {pareto_program.name}, gen {pareto_gen_idx + 1}")

    print(f"All done in {time.time() - t_start:.1f}s. "
          f"Plots -> {PLOTS_DIR}, reports -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()
