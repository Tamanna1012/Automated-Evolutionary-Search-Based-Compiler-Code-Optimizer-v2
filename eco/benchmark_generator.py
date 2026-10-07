"""Synthesizes a single random straight-line arithmetic TAC benchmark
program, complete with pseudo-C source, multiple test-input sets, expected
(ground-truth) outputs and baseline metrics.

Programs are generated with deliberate redundancy (repeated subexpressions
for CSE), dead temporaries (for DCE), identity-friendly constants like 0/1
(for algebraic simplification) and copy chains (for copy propagation), so
every one of the six optimizations has real opportunities to fire.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .metrics import Fitness, evaluate
from .tac import TAC, render_source

OPS = ["+", "-", "*", "/"]
NUM_TEST_SETS = 5


@dataclass
class BenchmarkProgram:
    id: int
    name: str
    source_code: str
    tac: TAC
    input_names: List[str]
    test_input_sets: List[Dict[str, int]]
    expected_outputs: List[List[int]]
    baseline_fitness: Fitness


def _gen_test_input_sets(rng: random.Random, input_names: List[str]) -> List[Dict[str, int]]:
    sets = []
    for _ in range(NUM_TEST_SETS):
        sets.append({name: rng.randint(-15, 15) for name in input_names})
    return sets


def generate_program(prog_id: int, rng: random.Random) -> BenchmarkProgram:
    num_inputs = rng.randint(2, 6)
    input_names = [f"in{i}" for i in range(num_inputs)]

    tac: TAC = []
    vars_pool: List[str] = []

    for i, name in enumerate(input_names):
        dest = f"v{i}"
        tac.append(("input", dest, name))
        vars_pool.append(dest)

    num_consts = rng.randint(2, 5)
    for i in range(num_consts):
        dest = f"c{i}"
        # bias toward 0/1 sometimes so algebraic simplification has real targets
        value = rng.choice([0, 1]) if rng.random() < 0.35 else rng.randint(-10, 10)
        tac.append(("const", dest, value))
        vars_pool.append(dest)

    num_temps = rng.randint(20, 45)
    last_bin_key = None
    last_copy_dest = None
    last_const_chain_dest = None  # tail of a deliberately deep foldable-constant chain
    const_pool = [f"c{i}" for i in range(num_consts)]
    for i in range(num_temps):
        dest = f"t{i}"
        roll = rng.random()
        if roll < 0.20 and vars_pool:
            # copy chain -> feeds copy_propagation / dead_code_elimination.
            # Half the time chain off the *previous* copy so multi-hop
            # chains form (each hop needs its own copy_propagation slot
            # to fully collapse - see optimizations.py).
            if last_copy_dest is not None and rng.random() < 0.8:
                src = last_copy_dest
            else:
                src = rng.choice(vars_pool)
            tac.append(("copy", dest, src))
            last_copy_dest = dest
        elif roll < 0.40 and last_bin_key is not None:
            # deliberately repeat the previous arithmetic expression -> CSE fodder
            op, a, b = last_bin_key
            tac.append(("bin", dest, op, a, b))
        elif roll < 0.55 and const_pool:
            # deliberately deep chain of purely-constant arithmetic -> feeds
            # constant_folding. Chaining off the previous link (rather than
            # always two fresh constants) builds multi-hop foldable chains
            # that compete with the copy chains above for genome budget.
            op = rng.choice(OPS)
            a = last_const_chain_dest if (last_const_chain_dest and rng.random() < 0.85) else rng.choice(const_pool)
            b = rng.choice(const_pool)
            tac.append(("bin", dest, op, a, b))
            last_const_chain_dest = dest
        else:
            op = rng.choice(OPS)
            a = rng.choice(vars_pool)
            b = rng.choice(vars_pool)
            tac.append(("bin", dest, op, a, b))
            last_bin_key = (op, a, b)
        vars_pool.append(dest)

    num_outputs = rng.randint(1, 3)
    candidates = vars_pool[num_inputs + num_consts:]  # prefer outputs that depend on real work
    if not candidates:
        candidates = vars_pool
    chosen = rng.sample(candidates, min(num_outputs, len(candidates)))
    seen = set()
    for var in chosen:
        if var not in seen:
            tac.append(("output", var))
            seen.add(var)

    test_input_sets = _gen_test_input_sets(rng, input_names)
    base_eval = evaluate(tac, test_input_sets)
    name = f"bench_{prog_id:04d}"
    source_code = render_source(tac, input_names, func_name=name)

    return BenchmarkProgram(
        id=prog_id,
        name=name,
        source_code=source_code,
        tac=tac,
        input_names=input_names,
        test_input_sets=test_input_sets,
        expected_outputs=base_eval.outputs_per_test,
        baseline_fitness=base_eval.fitness,
    )
