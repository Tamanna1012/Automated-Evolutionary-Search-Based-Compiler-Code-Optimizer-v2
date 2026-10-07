"""Benchmark dataset assembly: 200-500 synthetic programs, each carrying
source code, TAC, multiple test-input sets, expected outputs and baseline
metrics, per the assignment's Dataset Requirements section."""
from __future__ import annotations

import random
from typing import List

from .benchmark_generator import BenchmarkProgram, generate_program


def generate_dataset(n_programs: int = 300, seed: int = 0) -> List[BenchmarkProgram]:
    if not (200 <= n_programs <= 500):
        raise ValueError("n_programs must be between 200 and 500 per the spec")
    master_rng = random.Random(seed)
    programs = []
    for i in range(n_programs):
        # derive an independent stream per program so any subset is reproducible
        prog_rng = random.Random(master_rng.randrange(2**32))
        programs.append(generate_program(i, prog_rng))
    return programs


def dataset_summary(programs: List[BenchmarkProgram]) -> dict:
    sizes = [p.baseline_fitness.instr_count for p in programs]
    return {
        "num_programs": len(programs),
        "avg_instr_count": sum(sizes) / len(sizes),
        "min_instr_count": min(sizes),
        "max_instr_count": max(sizes),
    }
