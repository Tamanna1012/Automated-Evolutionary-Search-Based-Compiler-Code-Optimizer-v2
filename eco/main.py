"""End-to-end experiment driver.

1. Generates the benchmark dataset (200-500 programs: instances of real
   hand-written kernels compiled to TAC by the ast front end; ``--synthetic``
   switches to the old random TAC generator).
2. Runs the full evolutionary search (with per-generation logging) on a
   "deep-dive" subset of programs, producing the detailed per-program
   reports the assignment asks for.
3. Runs the same search across the *entire* dataset to compute the
   aggregate Testing & Metrics table (validity rate, Pareto front size,
   best-cost improvement, generations to convergence).
4. Writes a detailed report for every program (plus a summary CSV).
5. Renders all six required plots and writes text reports to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .dataset import generate_dataset, dataset_summary
from .ga import GAConfig, run_ga
from .pareto import distinct_fitness_points
from .report import (
    aggregate_metrics,
    format_program_report_text,
    program_report,
    write_all_program_reports,
)
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
ALL_REPORTS_DIR = os.path.join(REPORTS_DIR, "all_programs")

N_PROGRAMS = 300
DEEP_DIVE_N = 20
POP_SIZE = 30
NUM_GENERATIONS = 50
SEED = 42


def main(kind: str = "kernels"):
    t_start = time.time()
    # The default (real-kernel) run writes to eco/outputs; the optional
    # synthetic run writes to eco/outputs_synthetic so it never overwrites it.
    out_root = "outputs" if kind == "kernels" else "outputs_synthetic"
    PLOTS_DIR = os.path.join(HERE, out_root, "plots")
    REPORTS_DIR = os.path.join(HERE, out_root, "reports")
    ALL_REPORTS_DIR = os.path.join(REPORTS_DIR, "all_programs")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Generating {N_PROGRAMS} benchmark programs (kind={kind})...")
    programs = generate_dataset(N_PROGRAMS, seed=SEED, kind=kind)
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

    # ---- Per-program reports for every program + one summary CSV ----
    print(f"Writing reports for all {N_PROGRAMS} programs -> {ALL_REPORTS_DIR}")
    write_all_program_reports(programs, results, ALL_REPORTS_DIR)

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

    # The 2D/3D Pareto scatter plots use the FINAL population of the
    # deep-dive program whose final Pareto front is richest (most distinct
    # objective-space points). With the trade-off optimizations
    # (strength_reduction vs multiply_fusion) the final front is no longer a
    # single point; if some program still collapses to one point we say so.
    pareto_program, pareto_result = max(
        zip(deep_programs, deep_results),
        key=lambda pr: (distinct_fitness_points(pr[1].final_pareto_front), len(pr[1].final_pareto_front)),
    )
    pareto_pop = [ind for ind in pareto_result.final_population if ind.valid]
    pareto_front_final = pareto_result.final_pareto_front
    final_gen = len(pareto_result.history)
    n_distinct = distinct_fitness_points(pareto_front_final)
    pareto_label = (f"{pareto_program.name}, final population (gen {final_gen}), "
                    f"{n_distinct} distinct front point{'s' if n_distinct != 1 else ''}"
                    f"\nrichest final front among the {DEEP_DIVE_N} deep-dive programs")
    print(f"  Pareto scatter: {pareto_program.name}, final population (gen {final_gen}): "
          f"{n_distinct} distinct objective-space points on the front")

    plot_pareto_2d(pareto_pop, pareto_front_final, os.path.join(PLOTS_DIR, "2_pareto_front_2d.png"),
                    title=f"Pareto Front (exec_time vs instr_count)\n{pareto_label}")
    plot_fitness_boxplot(flagship_result, os.path.join(PLOTS_DIR, "3_fitness_boxplot.png"),
                          title=f"Cost Distribution Across Generations — {flagship_program.name}")

    improvement_pcts = [rep["improvement_pct_exec_time"] for rep in all_reports]
    plot_improvement_bar([p.name for p in deep_programs], improvement_pcts,
                          os.path.join(PLOTS_DIR, "4_improvement_bar.png"))

    plot_elite_heatmap_multi(deep_results, os.path.join(PLOTS_DIR, "5_elite_optimization_heatmap.png"))
    plot_pareto_3d(pareto_pop, pareto_front_final, os.path.join(PLOTS_DIR, "6_pareto_front_3d.png"),
                   title=f"Pareto Front 3D\n{pareto_label}")

    print(f"All done in {time.time() - t_start:.1f}s. "
          f"Plots -> {PLOTS_DIR}, reports -> {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary compiler-optimizer experiment")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the old random synthetic TAC generator instead of the real kernels")
    args = parser.parse_args()
    main("synthetic" if args.synthetic else "kernels")
