"""Live optimization of a user-supplied function, for the dashboard.

``optimize_source`` is the exact same pipeline the experiment uses -
``ast`` front end, ground truth from running the original Python source,
the genetic algorithm, Pareto analysis, output verification - packaged as a
single call that returns plain JSON-serialisable data. The dashboard runs
this file inside Pyodide (Python compiled to WebAssembly) so a visitor's
function is compiled, optimized and checked in their own browser; nothing
is sent to a server.

It deliberately imports only modules with no third-party dependencies.
"""
from __future__ import annotations

import json
import random
from typing import List

from .frontend import UnsupportedSyntax, compile_function, run_source
from .ga import GAConfig, run_ga
from .interpreter import run
from .metrics import evaluate
from .pareto import describe_tradeoff
from .tac import tac_to_text

MAX_POP = 50
MAX_GENS = 50
NUM_TESTS = 5


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def optimize_source(source: str, lo: int = 1, hi: int = 20, pop: int = 30,
                    gens: int = 30, seed: int = 42) -> dict:
    """Compile ``source`` (one straight-line integer function), evolve optimized
    versions of it and return everything the dashboard shows."""
    pop = max(10, min(int(pop), MAX_POP))
    gens = max(2, min(int(gens), MAX_GENS))
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo

    try:
        compiled = compile_function(source)
    except UnsupportedSyntax as exc:
        return _error(f"Unsupported code: {exc}")
    except SyntaxError as exc:
        return _error(f"Syntax error: {exc.msg} (line {exc.lineno})")
    if not compiled.arg_names:
        return _error("The function needs at least one parameter.")

    rng = random.Random(seed)
    tests = [{a: rng.randint(lo, hi) for a in compiled.arg_names} for _ in range(NUM_TESTS)]

    expected: List[List[int]] = []
    for inputs in tests:
        try:
            expected.append(run_source(source, compiled.name, inputs))
        except Exception as exc:  # noqa: BLE001 - report any failure of the user's function
            return _error(f"Your function failed on {inputs}: {type(exc).__name__}: {exc}")
    for inputs, want in zip(tests, expected):
        got = run(compiled.tac, inputs)
        if not got.ok or got.outputs != want:
            return _error(
                f"The compiled code disagrees with Python on {inputs} (Python: {want}, compiled: {got.outputs}). "
                "This usually means '//' was applied to a negative number (Python floors, the optimizer "
                "truncates toward zero) - use a positive input range.")

    baseline = evaluate(compiled.tac, tests, expected).fitness
    result = run_ga(compiled.tac, tests, expected, GAConfig(pop_size=pop, num_generations=gens, seed=seed))
    best = result.best_individual

    front = result.final_pareto_front
    front_ids = {id(i) for i in front}
    by_point = {}
    for ind in front:
        by_point.setdefault(ind.fitness_tuple(), []).append(ind)
    pareto = [{"genome": inds[0].genome, "fitness": inds[0].fitness.as_dict(),
               "label": describe_tradeoff(inds[0], front), "tied": len(inds)}
              for inds in sorted(by_point.values(), key=lambda g: g[0].fitness_tuple())]

    checks = []
    for inputs, want in zip(tests, expected):
        got = run(best.tac, inputs)
        checks.append({"inputs": inputs, "expected": want, "optimized": got.outputs,
                       "match": bool(got.ok and got.outputs == want)})

    imp = 0.0
    if baseline.exec_time:
        imp = 100.0 * (baseline.exec_time - best.fitness.exec_time) / baseline.exec_time
    return {
        "ok": True,
        "function": compiled.name,
        "args": compiled.arg_names,
        "settings": {"lo": lo, "hi": hi, "pop": pop, "gens": gens, "seed": seed},
        "tac_before": tac_to_text(compiled.tac).splitlines(),
        "tac_after": tac_to_text(best.tac).splitlines(),
        "genome": list(best.genome),
        "baseline": baseline.as_dict(),
        "best": best.fitness.as_dict(),
        "improvement_pct": imp,
        "best_cost": best.cost,
        "generations_to_convergence": result.generations_to_convergence,
        "hist_best": [round(r.best_cost, 5) for r in result.history],
        "hist_avg": [round(r.avg_cost, 5) for r in result.history],
        "valid_rate_final": result.history[-1].valid_rate,
        "front_size": len(front),
        "distinct_points": len(pareto),
        "population": [[i.fitness.exec_time, i.fitness.instr_count, i.fitness.code_size,
                        i.fitness.arith_ops, 1 if id(i) in front_ids else 0, 1 if i.valid else 0]
                       for i in result.final_population],
        "pareto": pareto,
        "checks": checks,
        "verdict": "PASS" if best.valid and all(c["match"] for c in checks) else "FAIL",
    }


def optimize_json(params_json: str) -> str:
    """JSON in, JSON out: the single entry point the browser calls."""
    p = json.loads(params_json)
    try:
        out = optimize_source(p["source"], p.get("lo", 1), p.get("hi", 20), p.get("pop", 30),
                              p.get("gens", 30), p.get("seed", 42))
    except RecursionError:
        out = _error("The expression is too deeply nested.")
    return json.dumps(out)
