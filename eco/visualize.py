"""The six required plots: convergence curve, 2D Pareto scatter, box plot
of fitness distribution, improvement bar chart, elite-optimization heatmap
and 3D Pareto scatter."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from .ga import GAResult
from .optimizations import OPTIMIZATION_NAMES

BOX_GENERATIONS = [1, 5, 10, 20, 50]


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(result: GAResult, out_path: str, title: str = "Convergence Curve"):
    gens = [r.generation for r in result.history]
    best = [r.best_cost for r in result.history]
    avg = [r.avg_cost for r in result.history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gens, best, marker="o", label="best cost", color="#1f6f43")
    ax.plot(gens, avg, marker="s", label="avg cost", color="#7a7a7a", alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weighted cost (lower is better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    _savefig(fig, out_path)


def plot_multi_convergence(results: Dict[str, GAResult], out_path: str,
                            title: str = "Convergence Curves (multiple programs)"):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, result in results.items():
        gens = [r.generation for r in result.history]
        best = [r.best_cost for r in result.history]
        ax.plot(gens, best, alpha=0.6, linewidth=1)
    # highlight the mean best-cost curve across programs
    max_len = max(len(r.history) for r in results.values())
    means = []
    for i in range(max_len):
        vals = [r.history[i].best_cost for r in results.values() if i < len(r.history)]
        means.append(sum(vals) / len(vals))
    ax.plot(range(1, max_len + 1), means, color="#1f6f43", linewidth=2.5, label="mean best cost")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best weighted cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    _savefig(fig, out_path)


def plot_pareto_2d(population, front, out_path: str, title: str = "Pareto Front (exec_time vs instr_count)"):
    """`population`/`front` are lists of (valid) Individual - typically a
    GAResult's final_population/final_pareto_front, or an earlier
    population_snapshots[gen] paired with pareto_front() of that snapshot."""
    pop = [ind for ind in population if ind.valid]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs_all = [ind.fitness.exec_time for ind in pop]
    ys_all = [ind.fitness.instr_count for ind in pop]
    ax.scatter(xs_all, ys_all, color="#b0b0b0", alpha=0.6, label="dominated", s=35)

    front_sorted = sorted(front, key=lambda i: i.fitness.exec_time)
    xs_f = [ind.fitness.exec_time for ind in front_sorted]
    ys_f = [ind.fitness.instr_count for ind in front_sorted]
    ax.plot(xs_f, ys_f, color="#c0392b", linewidth=1, linestyle="--", zorder=2)
    ax.scatter(xs_f, ys_f, color="#c0392b", label="Pareto front", s=55, zorder=3)

    ax.set_xlabel("exec_time (cycles)")
    ax.set_ylabel("instr_count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    _savefig(fig, out_path)


def plot_pareto_3d(population, front, out_path: str,
                    title: str = "Pareto Front 3D (exec_time, instr_count, code_size)"):
    pop = [ind for ind in population if ind.valid]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([i.fitness.exec_time for i in pop], [i.fitness.instr_count for i in pop],
               [i.fitness.code_size for i in pop], color="#b0b0b0", alpha=0.5, label="dominated")
    ax.scatter([i.fitness.exec_time for i in front], [i.fitness.instr_count for i in front],
               [i.fitness.code_size for i in front], color="#c0392b", s=55, label="Pareto front")
    ax.set_xlabel("exec_time")
    ax.set_ylabel("instr_count")
    ax.set_zlabel("code_size")
    ax.set_title(title)
    ax.legend()
    _savefig(fig, out_path)


def plot_fitness_boxplot(result: GAResult, out_path: str,
                          title: str = "Fitness (cost) Distribution Across Generations"):
    num_gens = len(result.history)
    gens = sorted({g for g in BOX_GENERATIONS if g <= num_gens} | {num_gens})
    data = [result.history[g - 1].costs for g in gens]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=[str(g) for g in gens], showmeans=True)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weighted cost")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    _savefig(fig, out_path)


def plot_improvement_bar(program_names: List[str], improvements_pct: List[float], out_path: str,
                          title: str = "Best-Individual Improvement vs Baseline"):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f6f43" if v >= 0 else "#c0392b" for v in improvements_pct]
    ax.bar(range(len(program_names)), improvements_pct, color=colors)
    ax.set_xticks(range(len(program_names)))
    ax.set_xticklabels(program_names, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("exec_time improvement vs baseline (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    _savefig(fig, out_path)


def plot_elite_heatmap_multi(results: List[GAResult], out_path: str,
                              title: str = "Optimization Frequency in Elite Individuals (across programs)"):
    num_gens = min(len(r.history) for r in results)
    matrix = []
    for name in OPTIMIZATION_NAMES:
        row = []
        for gen_idx in range(num_gens):
            total = 0
            count = 0
            for r in results:
                rec = r.history[gen_idx]
                total += sum(len(g) for g in rec.elite_genomes)
                count += sum(g.count(name) for g in rec.elite_genomes)
            row.append(count / total if total else 0.0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(6, num_gens * 0.35), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(OPTIMIZATION_NAMES)))
    ax.set_yticklabels(OPTIMIZATION_NAMES)
    ax.set_xticks(range(num_gens))
    ax.set_xticklabels([str(i + 1) for i in range(num_gens)], fontsize=7, rotation=90)
    ax.set_xlabel("Generation")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="share of elite genome slots")
    _savefig(fig, out_path)


def plot_elite_heatmap(result: GAResult, out_path: str,
                        title: str = "Optimization Frequency in Elite Individuals"):
    num_gens = len(result.history)
    matrix = []
    for name in OPTIMIZATION_NAMES:
        row = []
        for rec in result.history:
            total = sum(len(g) for g in rec.elite_genomes) or 1
            count = sum(g.count(name) for g in rec.elite_genomes)
            row.append(count / total)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(6, num_gens * 0.35), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(OPTIMIZATION_NAMES)))
    ax.set_yticklabels(OPTIMIZATION_NAMES)
    ax.set_xticks(range(num_gens))
    ax.set_xticklabels([str(r.generation) for r in result.history], fontsize=7, rotation=90)
    ax.set_xlabel("Generation")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="share of elite genome slots")
    _savefig(fig, out_path)
