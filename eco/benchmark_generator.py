"""Synthesizes a single random straight-line arithmetic TAC benchmark
program, complete with pseudo-C source, multiple test-input sets, expected
(ground-truth) outputs and baseline metrics.

Programs are generated with deliberate redundancy (repeated subexpressions
for CSE), dead temporaries (for DCE), identity-friendly constants like 0/1
(for algebraic simplification) and copy chains (for copy propagation), so
every one of the optimizations has real opportunities to fire. A final
"scaling" block adds multiply-by-small-literal and repeated-addition
patterns, which strength_reduction and multiply_fusion trade off against
each other (see optimizations.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .metrics import Fitness, evaluate
from .tac import TAC, render_source

OPS = ["+", "-", "*", "/"]
NUM_TEST_SETS = 5
SCALE_FACTORS = [2, 3, 3, 4, 4]  # 3 and 4 trade speed for size when strength-reduced


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


def _append_scaling_block(tac: TAC, prog_id: int, input_names: List[str], vars_pool: List[str]):
    """Append multiply-by-literal and repeated-add fodder (plus outputs).

    Uses its own RNG stream keyed on the program id so the rest of the
    program (and its test inputs) is generated exactly as before.
    """
    frng = random.Random(prog_id * 7919 + 17)
    sources = [f"v{i}" for i in range(len(input_names))] + [v for v in vars_pool if v.startswith("t")]

    for j in range(frng.randint(2, 4)):  # x * k  -> strength_reduction targets
        k = frng.choice(SCALE_FACTORS)
        tac.append(("const", f"k{j}", k))
        tac.append(("bin", f"s{j}", "*", frng.choice(sources), f"k{j}"))
        tac.append(("output", f"s{j}"))

    for j in range(frng.randint(1, 3)):  # (x+x)+x or (x+x)+(x+x) -> multiply_fusion targets
        x = frng.choice(sources)
        tac.append(("bin", f"u{j}", "+", x, x))
        if frng.random() < 0.7:
            tac.append(("bin", f"w{j}", "+", f"u{j}", x))
        else:
            tac.append(("bin", f"w{j}", "+", f"u{j}", f"u{j}"))
        tac.append(("output", f"w{j}"))


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

    _append_scaling_block(tac, prog_id, input_names, vars_pool)

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
