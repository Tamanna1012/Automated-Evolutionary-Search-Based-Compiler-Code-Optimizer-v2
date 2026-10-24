"""Live comparison of our optimizer with other ways of optimizing the same program.

Every method below runs on the user's program each time Optimize is pressed; nothing here is a
stored result. Methods:

  a) no optimization (the baseline)
  b) each technique on its own, repeated until it stops changing anything
  c) one sweep of the techniques in textbook order, with no repetition
  d) local only: the straight-line passes inside each basic block, no global data flow
  e) the best pass order the project's genetic algorithm finds (small fixed budget, fixed seed)
  f) our method: global analysis, repeated until nothing changes

Each result is lowered to the TAC the shared cost model reads (``metrics.evaluate``), so all four
numbers are comparable. A method is marked verified only if it prints the same values as the
original on every test input; the check also uses hold-out inputs that the genetic algorithm never
saw, and a symbolic proof when one exists. A method that fails is flagged and cannot be "best".

For a program with branches or loops the techniques are the global ones in ``cfg.py`` (the
optimizer has no global version of algebraic simplification or copy propagation).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .cfg import GLOBAL_TECHNIQUES, PASSES, execute, global_optimize, has_control
from .code_optimizer import BY_NAME, MAX_ROUNDS, TECHNIQUES, Stmt, from_tac, local_optimize, optimize_stmts, stmts_to_tac
from .equivalence import prove_equivalent
from .ga import GAConfig, run_ga
from .interpreter import run
from .metrics import evaluate

# The genetic algorithm gets a small, fixed budget so the comparison is fast and repeatable in the browser.
GA_POPULATION = 16
GA_GENERATIONS = 12
GA_SEED = 0

METRICS = [("instr_count", "Instructions"), ("exec_time", "Simulated cycles"),
           ("arith_ops", "Arithmetic operations"), ("code_size", "Code size")]
EPS = 1e-9


def _alone(stmts: Sequence[Stmt], name: str, control: bool) -> List[Stmt]:
    """One technique by itself, repeated until it changes nothing."""
    fn = dict(PASSES)[name] if control else BY_NAME[name]
    current = list(stmts)
    for _ in range(MAX_ROUNDS):
        current, events = fn(current)
        if not events:
            break
    return current


def _one_sweep(stmts: Sequence[Stmt], control: bool) -> List[Stmt]:
    """Every technique once, in order, with no second round."""
    if not control:
        return optimize_stmts(list(stmts), passes=list(TECHNIQUES))[0]
    current = list(stmts)
    for _, fn in PASSES:
        current, _ = fn(current)
    return current


def _verify_stmts(original: Sequence[Stmt], candidate: Sequence[Stmt], inputs: Sequence[dict], want: Sequence[list]) -> dict:
    for t, expected in zip(inputs, want):
        try:
            got = execute(candidate, t)
        except Exception as exc:  # noqa: BLE001 - any failure to run is a failed check
            return {"status": "failed", "detail": f"on {t}: could not run ({exc})"}
        if got != expected:
            return {"status": "failed", "detail": f"on {t}: original printed {expected}, this method printed {got}"}
    return _proof(original, candidate, len(inputs))


def _verify_tac(original: Sequence[Stmt], tac, inputs: Sequence[dict], want: Sequence[list]) -> dict:
    for t, expected in zip(inputs, want):
        res = run(tac, t)
        if not res.ok or res.outputs != expected:
            return {"status": "failed", "detail": f"on {t}: original printed {expected}, this method printed {res.outputs if res.ok else res.error}"}
    try:
        candidate = from_tac(tac)
    except Exception:  # noqa: BLE001
        return {"status": "tested", "detail": f"printed the same values on all {len(inputs)} test inputs; no symbolic proof"}
    return _proof(original, candidate, len(inputs))


def _proof(original, candidate, n_tests: int) -> dict:
    try:
        proved = bool(prove_equivalent(original, candidate)["proved"])
    except Exception:  # noqa: BLE001
        proved = False
    if proved:
        return {"status": "proved", "detail": f"symbolically proved equal, and printed the same values on all {n_tests} test inputs"}
    return {"status": "tested", "detail": f"printed the same values on all {n_tests} test inputs; no symbolic proof"}


def _row(key: str, label: str, group: str, description: str, tac, metric_inputs, check: dict, ours: bool = False, extra: Optional[dict] = None) -> dict:
    fit = evaluate(tac, metric_inputs).fitness.as_dict()
    row = {"key": key, "label": label, "group": group, "description": description, "is_ours": ours,
           "instr_count": fit["instr_count"], "exec_time": round(fit["exec_time"], 4),
           "arith_ops": fit["arith_ops"], "code_size": fit["code_size"],
           "status": check["status"], "verified": check["status"] != "failed", "detail": check["detail"]}
    if extra:
        row.update(extra)
    return row


def _wins(methods: List[dict]) -> List[dict]:
    """For every metric: is ours best, tied or behind, against which method, and by how much."""
    ours = next(m for m in methods if m["is_ours"])
    others = [m for m in methods if not m["is_ours"] and m["verified"]]
    out = []
    for key, label in METRICS:
        best = min(m[key] for m in others)
        rivals = [m["label"] for m in others if abs(m[key] - best) < EPS]
        diff = round(best - ours[key], 4)               # other - ours: positive means ours is lower (better)
        pct = round(100.0 * (best - ours[key]) / best, 2) if best else None
        if abs(diff) < EPS:
            verdict = "tied"
        else:
            verdict = "best" if diff > 0 else "behind"
        base = next(m for m in methods if m["group"] == "baseline")
        out.append({"metric": key, "label": label, "verdict": verdict, "ours": ours[key], "other": best,
                    "compared_with": rivals, "diff": diff, "pct": pct,
                    "vs_baseline": round(base[key] - ours[key], 4),
                    "vs_baseline_pct": round(100.0 * (base[key] - ours[key]) / base[key], 2) if base[key] else None})
    return out


def build_candidates(tac, original: Sequence[Stmt], runnable: Sequence[dict]) -> List[dict]:
    """Every method's resulting program, before any measuring. Each entry has ``key``, ``label``,
    ``group``, ``description`` and either ``stmts`` (a statement list) or ``tac`` (the genetic
    algorithm's code); the genetic algorithm's entry also has its ``genome``."""
    control = has_control(original)
    names = GLOBAL_TECHNIQUES if control else TECHNIQUES
    out = [{"key": "baseline", "label": "No optimization", "group": "baseline", "description": "The generated code as it is.", "tac": tac}]
    for name in names:
        out.append({"key": f"only:{name}", "label": f"{name} only", "group": "single",
                    "description": f"{name} applied alone, repeated until it changes nothing.", "stmts": _alone(original, name, control)})
    out.append({"key": "sweep", "label": "One sweep", "group": "sweep",
                "description": "Every technique once, in textbook order, with no repetition." if not control else
                               "Every global technique once, in the optimizer's pass order, with no repetition.",
                "stmts": _one_sweep(original, control)})
    out.append({"key": "local", "label": "Local only", "group": "local",
                "description": "The straight-line passes run inside each basic block; no data flow between blocks.",
                "stmts": local_optimize(list(original))})
    # The GA's passes assume straight-line code, so on a loop some orderings never terminate. A correct
    # variant never needs 20x the original's cycles, so that is the cap; beyond it a variant is invalid.
    cap = 20 * max([run(tac, t).cycles for t in runnable] + [1]) + 100
    ga = run_ga(tac, runnable, [run(tac, t).outputs for t in runnable],
                GAConfig(pop_size=GA_POPULATION, num_generations=GA_GENERATIONS, seed=GA_SEED, max_steps=cap))
    out.append({"key": "ga", "label": "Genetic algorithm", "group": "ga",
                "description": f"Best pass order found by the project's genetic algorithm (population {GA_POPULATION}, "
                               f"{GA_GENERATIONS} generations, seed {GA_SEED}).",
                "tac": ga.best_individual.tac, "genome": list(ga.best_individual.genome)})
    out.append({"key": "ours", "label": "Our method", "group": "ours",
                "description": "Global analysis, all techniques repeated until a full round changes nothing.",
                "stmts": global_optimize(original)[0] if control else optimize_stmts(list(original))[0]})
    return out


def mark_best(methods: List[dict]) -> None:
    """Flag, per metric, every verified method that has the lowest value. A failed method is never best."""
    for key, _ in METRICS:
        best_value = min(m[key] for m in methods if m["verified"])
        for m in methods:
            m.setdefault("best", {})[key] = m["verified"] and abs(m[key] - best_value) < EPS


def compare_methods(tac, original: Sequence[Stmt], inputs: Sequence[dict], holdout: Sequence[dict] = ()) -> dict:
    """Run every method on the same program and report its four metrics and whether it stayed correct.

    ``tac`` is the front end's code for the program, ``original`` the same program as statements,
    ``inputs`` the test inputs the cost model averages over, and ``holdout`` extra inputs used only
    to verify (the genetic algorithm never sees them).
    """
    control = has_control(original)
    runnable = [t for t in inputs if run(tac, t).ok]
    check_inputs = list(runnable) + [t for t in holdout if run(tac, t).ok and t not in runnable]
    want = [execute(original, t) for t in check_inputs]
    methods: List[dict] = []
    for c in build_candidates(tac, original, runnable):
        if c["group"] == "baseline":
            check = {"status": "proved", "detail": "this is the original program"}
            program = c["tac"]
        elif "stmts" in c:
            check = _verify_stmts(original, c["stmts"], check_inputs, want)
            program = stmts_to_tac(c["stmts"])
        else:
            check = _verify_tac(original, c["tac"], check_inputs, want)
            program = c["tac"]
        methods.append(_row(c["key"], c["label"], c["group"], c["description"], program, inputs, check,
                            ours=c["group"] == "ours", extra={"genome": c["genome"]} if "genome" in c else None))
    mark_best(methods)
    return {"metrics": [{"key": k, "label": l} for k, l in METRICS], "methods": methods, "wins": _wins(methods),
            "scope": "global" if control else "straight-line",
            "techniques": list(GLOBAL_TECHNIQUES if control else TECHNIQUES),
            "ga": {"population": GA_POPULATION, "generations": GA_GENERATIONS, "seed": GA_SEED},
            "verified_on": len(check_inputs)}
