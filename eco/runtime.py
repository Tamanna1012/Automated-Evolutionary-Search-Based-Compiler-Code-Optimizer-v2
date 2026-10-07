"""Real (measured) execution time for TAC programs.

The GA keeps optimizing the deterministic cycle model in ``interpreter.py``.
This module is the *validation* side: it translates a TAC program into a real
Python function, runs it many times and times it with
``time.perf_counter_ns``, so we can check how well the simulated cycles
predict actual speed.

Backend: Python only. A C backend (gcc + ctypes) was considered but no C
compiler is guaranteed to be installed, and a ctypes call has a fixed
overhead that would swamp these tiny straight-line kernels, so the clean
fallback is the only backend. What is measured is therefore CPython's
execution of the generated function - assignments, arithmetic and (for
``/``) a call to the same truncating-division helper the interpreter uses -
not native machine code.

Noise control: each measurement is ``rounds`` timed batches (after a warm-up
batch) with the garbage collector paused; the reported value is the
*minimum* batch time per call, the usual low-noise estimator for micro
benchmarks. Results are cached by a hash of the TAC (+ argument names and
test inputs), so identical programs are timed once and always agree.
"""
from __future__ import annotations

import gc
import hashlib
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .interpreter import _safe_div
from .tac import Instr

DEFAULT_REPS = 200      # passes over the whole test-input list per timed batch
DEFAULT_ROUNDS = 7      # timed batches; the minimum is reported

_CACHE: Dict[str, float] = {}


def clear_cache() -> None:
    _CACHE.clear()


def tac_hash(tac: Sequence[Instr]) -> str:
    return hashlib.sha1(repr(list(tac)).encode("utf-8")).hexdigest()


def tac_to_python(tac: Sequence[Instr], input_names: Sequence[str], func_name: str = "prog") -> str:
    """Render a TAC program as the source text of a Python function.

    Parameters are named ``p_<input>`` and every ``input`` instruction
    copies its parameter into the TAC variable, so the generated code does
    the same amount of work as the TAC (one statement per instruction).
    The function returns the tuple of ``output`` values.
    """
    params = ", ".join(f"p_{n}" for n in input_names)
    lines = [f"def {func_name}({params}):"]
    outputs: List[str] = []
    for instr in tac:
        op = instr[0]
        if op == "input":
            lines.append(f"    {instr[1]} = p_{instr[2]}")
        elif op == "const":
            lines.append(f"    {instr[1]} = {int(instr[2])}")
        elif op == "copy":
            lines.append(f"    {instr[1]} = {instr[2]}")
        elif op == "bin":
            _, dest, o, a, b = instr
            if o == "/":
                lines.append(f"    {dest} = _div({a}, {b})")
            else:
                lines.append(f"    {dest} = {a} {o} {b}")
        elif op == "output":
            outputs.append(instr[1])
        else:
            raise ValueError(f"unknown opcode {op}")
    lines.append(f"    return ({', '.join(outputs)}{',' if len(outputs) == 1 else ''})")
    return "\n".join(lines) + "\n"


def compile_tac(tac: Sequence[Instr], input_names: Sequence[str]) -> Callable:
    """Compile a TAC program into a real Python function taking positional ints."""
    namespace = {"_div": _safe_div}
    exec(compile(tac_to_python(tac, input_names), "<tac>", "exec"), namespace)  # noqa: S102
    return namespace["prog"]


def outputs_match(tac: Sequence[Instr], input_names: Sequence[str],
                  test_input_sets: Sequence[Dict[str, int]],
                  expected_outputs: Sequence[Sequence[int]]) -> bool:
    """True iff the compiled function reproduces the expected outputs on every test set."""
    fn = compile_tac(tac, input_names)
    for inputs, want in zip(test_input_sets, expected_outputs):
        if list(fn(*[inputs[n] for n in input_names])) != list(want):
            return False
    return True


def measure_runtime_ns(tac: Sequence[Instr], input_names: Sequence[str],
                       test_input_sets: Sequence[Dict[str, int]],
                       reps: int = DEFAULT_REPS, rounds: int = DEFAULT_ROUNDS) -> float:
    """Low-noise real runtime of one call (minimum over timed batches), in nanoseconds.

    One "call" is averaged over the test-input sets, matching how the cycle
    model's ``exec_time`` is averaged. Always positive.
    """
    key = "|".join([tac_hash(tac), ",".join(input_names), repr([sorted(t.items()) for t in test_input_sets]),
                    str(reps), str(rounds)])
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    fn = compile_tac(tac, input_names)
    arg_lists: List[Tuple[int, ...]] = [tuple(t[n] for n in input_names) for t in test_input_sets]
    calls_per_batch = reps * len(arg_lists)

    def batch():
        for _ in range(reps):
            for args in arg_lists:
                fn(*args)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        batch()  # warm-up
        best = math.inf
        for _ in range(rounds):
            t0 = time.perf_counter_ns()
            batch()
            elapsed = time.perf_counter_ns() - t0
            best = min(best, elapsed)
    finally:
        if was_enabled:
            gc.enable()
    value = max(best, 1) / calls_per_batch
    _CACHE[key] = value
    return value


# ----------------------------------------------------------------- statistics
def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(_ranks(xs), _ranks(ys))


# ---------------------------------------------------------- dataset measuring
def measure_program(program, result) -> dict:
    """Real runtime of the original TAC and of the final best individual."""
    base_ns = measure_runtime_ns(program.tac, program.input_names, program.test_input_sets)
    best = result.best_individual
    best_ns = measure_runtime_ns(best.tac, program.input_names, program.test_input_sets)
    correct = outputs_match(best.tac, program.input_names, program.test_input_sets, program.expected_outputs)
    base_cycles = program.baseline_fitness.exec_time
    return {
        "baseline_ns": base_ns,
        "best_ns": best_ns,
        "measured_improvement_pct": 100.0 * (base_ns - best_ns) / base_ns if base_ns else 0.0,
        "measured_speedup_x": base_ns / best_ns if best_ns else 1.0,
        "baseline_cycles": base_cycles,
        "best_cycles": best.fitness.exec_time,
        "outputs_correct": correct,
    }


def measure_dataset(programs, results) -> Dict[str, dict]:
    """Measured runtimes for every program: ``{program name: measure_program(...)}``."""
    return {p.name: measure_program(p, r) for p, r in zip(programs, results)}


def final_population_points(program, result, max_points: int = 12) -> List[Tuple[float, float]]:
    """(simulated cycles, measured ns) for distinct valid programs in the final population."""
    seen = {}
    for ind in result.final_population:
        if ind.valid:
            seen.setdefault(tac_hash(ind.tac), ind)
    inds = list(seen.values())[:max_points]
    return [(i.fitness.exec_time,
             measure_runtime_ns(i.tac, program.input_names, program.test_input_sets)) for i in inds]


def correlation_summary(programs, results, runtimes: Dict[str, dict]) -> dict:
    """How well do simulated cycles predict measured time? Three views, all honest.

    * ``pooled``: every measured program (original and best of each program),
      cycles vs ns. Inflated by program size - big programs are slow in both.
    * ``improvement``: per program, simulated improvement % vs measured
      improvement % - does the model predict *how much* an optimization helps?
    * ``within_program``: per program, rank correlation across the distinct
      programs in its final population - does the model rank variants of the
      *same* program correctly? Reported as the mean Spearman rho.
    """
    cyc, ns = [], []
    sim_imp, real_imp = [], []
    for p in programs:
        rt = runtimes[p.name]
        cyc += [rt["baseline_cycles"], rt["best_cycles"]]
        ns += [rt["baseline_ns"], rt["best_ns"]]
        if rt["baseline_cycles"]:
            sim_imp.append(100.0 * (rt["baseline_cycles"] - rt["best_cycles"]) / rt["baseline_cycles"])
            real_imp.append(rt["measured_improvement_pct"])

    within = []
    for p, r in zip(programs, results):
        pts = final_population_points(p, r)
        rho = spearman([c for c, _ in pts], [t for _, t in pts]) if len({c for c, _ in pts}) >= 3 else None
        if rho is not None:
            within.append(rho)

    return {
        "pooled_pearson": pearson(cyc, ns),
        "pooled_spearman": spearman(cyc, ns),
        "pooled_n": len(cyc),
        "improvement_pearson": pearson(sim_imp, real_imp),
        "improvement_spearman": spearman(sim_imp, real_imp),
        "improvement_n": len(sim_imp),
        "within_program_mean_spearman": sum(within) / len(within) if within else None,
        "within_program_n_programs": len(within),
    }
