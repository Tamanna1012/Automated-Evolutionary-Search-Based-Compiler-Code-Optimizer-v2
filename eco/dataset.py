"""Benchmark dataset assembly: 200-500 programs, each carrying source code,
TAC, multiple test-input sets, expected outputs and baseline metrics, per the
assignment's Dataset Requirements section.

By default the programs are instantiations of real hand-written kernels
(``kernels.py``) compiled by the ``ast`` front end, with ground-truth outputs
from running the original Python source. The old random TAC generator is
still available as ``kind="synthetic"``."""
from __future__ import annotations

import random
from typing import List

from .benchmark_generator import BenchmarkProgram, generate_program
from .kernels import generate_kernel_program

DATASET_KINDS = ("kernels", "synthetic")


def generate_dataset(n_programs: int = 300, seed: int = 0, kind: str = "kernels") -> List[BenchmarkProgram]:
    if kind not in DATASET_KINDS:
        raise ValueError(f"kind must be one of {DATASET_KINDS}, got {kind!r}")
    if not (200 <= n_programs <= 500):
        raise ValueError("n_programs must be between 200 and 500 per the spec")
    master_rng = random.Random(seed)
    programs = []
    for i in range(n_programs):
        # derive an independent stream per program so any subset is reproducible
        prog_rng = random.Random(master_rng.randrange(2**32))
        build = generate_kernel_program if kind == "kernels" else generate_program
        programs.append(build(i, prog_rng))
    return programs


def dataset_summary(programs: List[BenchmarkProgram]) -> dict:
    sizes = [p.baseline_fitness.instr_count for p in programs]
    return {
        "num_programs": len(programs),
        "avg_instr_count": sum(sizes) / len(sizes),
        "min_instr_count": min(sizes),
        "max_instr_count": max(sizes),
    }
