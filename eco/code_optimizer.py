"""Classic code optimization for the dashboard's "Code Optimization" tab.

Where the rest of the project *searches* for good optimization sequences with a
genetic algorithm (``optimizations.py``), this module applies the standard
compiler techniques themselves, one after another, and records what changed and
why. It is deliberately separate from ``optimizations.py``: those passes work on
the front end's variable-only TAC and only need to return a new program, while
these work on textbook 3-address statements (literals inline, temporaries named
``t1``, ``t2``) and must log every change with a reason for the user to read.

It works on the intermediate code the front end (``frontend.py``) generates:

    Python source -> TAC (front end) -> 3-address code -> CODE OPTIMIZATION
        -> optimized 3-address code -> 4-address code (quadruples)

Techniques (applied repeatedly until nothing changes):

  1. constant propagation      a = 10; b = a + 5      ->  b = 10 + 5
  2. constant folding          t1 = 10 * 5            ->  t1 = 50
  3. algebraic simplification  a + 0, a - 0, a * 1, a / 1, a * 0
  4. copy propagation          a = b; c = a + 5       ->  c = b + 5
  5. common subexpression elimination   t2 = a + b    ->  t2 = t1
  6. dead code elimination     unused definitions are removed

Safety rules: only the generated code is changed (never the user's source);
division is folded only when exact and the divisor is not zero; expressions
are reused only while their operands are unchanged; the program's inputs and
outputs are never removed; and the optimized code is executed against the
original on random inputs (``checks`` / ``behaviour_preserved``, used by the
tests) to prove it behaves identically.

Everything here uses only the standard library so the dashboard can run it in
the browser (Pyodide) on a visitor's own program.
"""
from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .frontend import UnsupportedSyntax, compile_function, run_source
from .interpreter import _safe_div, run
from .tac import dest_of, tac_to_text

Operand = Union[str, int, None]
COMMUTATIVE = {"+", "*"}
MAX_ROUNDS = 12

TECHNIQUES = [
    "Constant Propagation", "Constant Folding", "Algebraic Simplification",
    "Copy Propagation", "Common Subexpression Elimination", "Dead Code Elimination",
]


# ------------------------------------------------------------------ statements
@dataclass
class Stmt:
    """One textbook three-address statement.

    ``op`` is one of ``input``, ``=``, ``+``, ``-``, ``*``, ``/`` or ``print``.
    Operands are variable names (``str``) or integer literals (``int``).
    ``origin`` is the statement's number in the original listing; it stays
    attached to the statement through optimization so changes can be traced.
    """
    dest: Optional[str]
    op: str
    a: Operand = None
    b: Operand = None
    origin: int = 0

    def text(self) -> str:
        if self.op == "input":
            return f"{self.dest} = input({self.a})"
        if self.op == "=":
            return f"{self.dest} = {self.a}"
        if self.op == "print":
            return f"print {self.a}"
        return f"{self.dest} = {self.a} {self.op} {self.b}"

    def uses(self) -> List[str]:
        if self.op == "input":
            return []
        return [x for x in (self.a, self.b) if isinstance(x, str)]

    def copy_with(self, **kw) -> "Stmt":
        d = {"dest": self.dest, "op": self.op, "a": self.a, "b": self.b, "origin": self.origin}
        d.update(kw)
        return Stmt(**d)


def from_tac(tac) -> List[Stmt]:
    """Convert the front end's TAC into textbook 3-address statements.

    Literal constants the front end pooled (``_k1 = 5``) become inline operands
    (``t1 = a + 5``) and ``_e1`` temporaries are renamed ``t1``; variables the
    user named are untouched.
    """
    literals = {i[1]: i[2] for i in tac if i[0] == "const" and i[1].startswith("_k")}
    taken = {dest_of(i) for i in tac if dest_of(i) is not None}
    rename: Dict[str, str] = {}
    for name in sorted(taken):
        if name.startswith("_e") and name[2:].isdigit():
            cand = "t" + name[2:]
            while cand in taken or cand in rename.values():
                cand += "_"
            rename[name] = cand

    def operand(v):
        if v in literals:
            return literals[v]
        return rename.get(v, v)

    out: List[Stmt] = []
    for ins in tac:
        op = ins[0]
        if op == "input":
            out.append(Stmt(rename.get(ins[1], ins[1]), "input", ins[2]))
        elif op == "const":
            if ins[1] in literals:
                continue  # inlined at its uses
            out.append(Stmt(rename.get(ins[1], ins[1]), "=", int(ins[2])))
        elif op == "copy":
            out.append(Stmt(rename.get(ins[1], ins[1]), "=", operand(ins[2])))
        elif op == "bin":
            _, d, o, a, b = ins
            out.append(Stmt(rename.get(d, d), o, operand(a), operand(b)))
        elif op == "output":
            out.append(Stmt(None, "print", operand(ins[1])))
    for n, s in enumerate(out, 1):
        s.origin = n
    return out


def run_stmts(stmts: Sequence[Stmt], inputs: Dict[str, int]) -> List[int]:
    """Execute 3-address statements (same arithmetic as the interpreter)."""
    env: Dict[str, int] = {}
    outputs: List[int] = []

    def val(x):
        return x if isinstance(x, int) else env[x]

    for s in stmts:
        if s.op == "input":
            env[s.dest] = int(inputs[s.a])
        elif s.op == "=":
            env[s.dest] = val(s.a)
        elif s.op == "print":
            outputs.append(val(s.a))
        else:
            x, y = val(s.a), val(s.b)
            env[s.dest] = x + y if s.op == "+" else x - y if s.op == "-" else x * y if s.op == "*" else _safe_div(x, y)
    return outputs


# ------------------------------------------------------------------ passes
def _event(technique: str, before: Stmt, after: Optional[Stmt], reason: str) -> dict:
    return {"technique": technique, "original": before.text(),
            "optimized": after.text() if after is not None else "(removed)",
            "reason": reason, "origin": before.origin}


def constant_propagation(stmts: List[Stmt]):
    env: Dict[str, int] = {}
    out, events = [], []
    for s in stmts:
        new = s
        if s.op != "input":
            a = env[s.a] if isinstance(s.a, str) and s.a in env else s.a
            b = env[s.b] if isinstance(s.b, str) and s.b in env else s.b
            if (a, b) != (s.a, s.b):
                new = s.copy_with(a=a, b=b)
                known = [f"{x} = {env[x]}" for x in (s.a, s.b) if isinstance(x, str) and x in env]
                events.append(_event("Constant Propagation", s, new, "Known constant value substituted (" + ", ".join(known) + ")"))
        if new.dest is not None:
            if new.op == "=" and isinstance(new.a, int):
                env[new.dest] = new.a
            else:
                env.pop(new.dest, None)
        out.append(new)
    return out, events


def constant_folding(stmts: List[Stmt]):
    out, events = [], []
    for s in stmts:
        new = s
        if s.op in "+-*/" and len(s.op) == 1 and isinstance(s.a, int) and isinstance(s.b, int):
            value = None
            if s.op == "+":
                value = s.a + s.b
            elif s.op == "-":
                value = s.a - s.b
            elif s.op == "*":
                value = s.a * s.b
            elif s.b != 0 and s.a % s.b == 0:  # only exact, non-zero divisions
                value = s.a // s.b
            if value is not None:
                new = s.copy_with(op="=", a=value, b=None)
                events.append(_event("Constant Folding", s, new, f"Evaluated constant expression {s.a} {s.op} {s.b} at compile time"))
        out.append(new)
    return out, events


def algebraic_simplification(stmts: List[Stmt]):
    out, events = [], []
    for s in stmts:
        new, why = s, None
        if s.op == "+":
            if s.b == 0 and not isinstance(s.b, str):
                new, why = s.copy_with(op="=", a=s.a, b=None), "Adding zero does not change the value"
            elif s.a == 0 and not isinstance(s.a, str):
                new, why = s.copy_with(op="=", a=s.b, b=None), "Adding zero does not change the value"
        elif s.op == "-":
            if s.b == 0 and not isinstance(s.b, str):
                new, why = s.copy_with(op="=", a=s.a, b=None), "Subtracting zero does not change the value"
        elif s.op == "*":
            if (s.b == 0 and not isinstance(s.b, str)) or (s.a == 0 and not isinstance(s.a, str)):
                new, why = s.copy_with(op="=", a=0, b=None), "Multiplying by zero always gives zero"
            elif s.b == 1 and not isinstance(s.b, str):
                new, why = s.copy_with(op="=", a=s.a, b=None), "Multiplying by one does not change the value"
            elif s.a == 1 and not isinstance(s.a, str):
                new, why = s.copy_with(op="=", a=s.b, b=None), "Multiplying by one does not change the value"
        elif s.op == "/":
            if s.b == 1 and not isinstance(s.b, str):
                new, why = s.copy_with(op="=", a=s.a, b=None), "Dividing by one does not change the value"
        if why:
            events.append(_event("Algebraic Simplification", s, new, why))
        out.append(new)
    return out, events


def copy_propagation(stmts: List[Stmt]):
    copies: Dict[str, str] = {}   # variable -> variable it is a plain copy of
    out, events = [], []
    for s in stmts:
        new = s
        if s.op != "input":
            a = copies.get(s.a, s.a) if isinstance(s.a, str) else s.a
            b = copies.get(s.b, s.b) if isinstance(s.b, str) else s.b
            if (a, b) != (s.a, s.b):
                new = s.copy_with(a=a, b=b)
                used = [f"{x} is a copy of {copies[x]}" for x in (s.a, s.b) if isinstance(x, str) and x in copies]
                events.append(_event("Copy Propagation", s, new, "Copy replaced by its source (" + ", ".join(used) + ")"))
        d = new.dest
        if d is not None:
            for k in [k for k, v in copies.items() if k == d or v == d]:
                del copies[k]
            if new.op == "=" and isinstance(new.a, str) and new.a != d:
                copies[d] = copies.get(new.a, new.a)
        out.append(new)
    return out, events


def common_subexpression_elimination(stmts: List[Stmt]):
    def key(s):
        ops = (s.a, s.b)
        if s.op in COMMUTATIVE:
            ops = tuple(sorted(ops, key=lambda x: (isinstance(x, str), str(x))))
        return (s.op,) + ops

    available: Dict[tuple, str] = {}
    out, events = [], []
    for s in stmts:
        new = s
        is_bin = s.op in ("+", "-", "*", "/")
        if is_bin:
            k = key(s)
            if k in available:
                new = s.copy_with(op="=", a=available[k], b=None)
                events.append(_event("Common Subexpression Elimination", s, new,
                                     f"Expression {s.a} {s.op} {s.b} was already calculated in {available[k]}"))
        d = new.dest
        if d is not None:
            for k2 in [k2 for k2, v in available.items() if v == d or d in k2[1:]]:
                del available[k2]
            if is_bin and new.op != "=" and d not in s.uses():
                available[key(s)] = d
        out.append(new)
    return out, events


def dead_code_elimination(stmts: List[Stmt]):
    live = set()
    kept: List[Stmt] = []
    events = []
    for s in reversed(stmts):
        if s.op == "print":
            live.update(s.uses())
            kept.append(s)
        elif s.op == "input":
            live.discard(s.dest)
            kept.append(s)  # parameters are never removed
        elif s.dest not in live:
            events.append(_event("Dead Code Elimination", s, None, f"{s.dest} is never used, so the statement has no effect on the output"))
        else:
            live.discard(s.dest)
            live.update(s.uses())
            kept.append(s)
    kept.reverse()
    events.reverse()
    return kept, events


PIPELINE = [constant_propagation, constant_folding, algebraic_simplification,
            copy_propagation, common_subexpression_elimination, dead_code_elimination]


def optimize_stmts(stmts: List[Stmt]) -> Tuple[List[Stmt], List[dict]]:
    """Run all techniques repeatedly until a full round changes nothing."""
    current = [s.copy_with() for s in stmts]
    log: List[dict] = []
    for rnd in range(1, MAX_ROUNDS + 1):
        changed = False
        for pass_fn in PIPELINE:
            current, events = pass_fn(current)
            for e in events:
                e["round"] = rnd
            if events:
                changed = True
                log.extend(events)
        if not changed:
            break
    return current, log


# ------------------------------------------------------------------ tables
def _show(x) -> str:
    return "-" if x is None else str(x)


def three_address_rows(stmts: Sequence[Stmt]) -> List[dict]:
    rows = []
    for n, s in enumerate(stmts, 1):
        rows.append({"sno": n, "stmt": s.text(), "op": s.op, "arg1": _show(s.a), "arg2": _show(s.b), "origin": s.origin})
    return rows


def quadruple_rows(stmts: Sequence[Stmt]) -> List[dict]:
    """4-address code: (operation, arg1, arg2, result)."""
    rows = []
    for n, s in enumerate(stmts, 1):
        rows.append({"sno": n, "op": s.op, "arg1": _show(s.a), "arg2": _show(s.b), "result": _show(s.dest)})
    return rows


# ------------------------------------------------------------------ entry points
def _error(stage: str, message: str) -> dict:
    return {"ok": False, "stage": stage, "error": message}


def optimize_program(source: str, lo: int = 1, hi: int = 20, seed: int = 7, num_tests: int = 5) -> dict:
    """Run the whole workflow on a user's function and return everything the tab shows."""
    if not source or not source.strip():
        return _error("Input", "The source code is empty. Write a function and press Optimize.")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return _error("Syntax analysis", f"Syntax error: {exc.msg} (line {exc.lineno})")
    try:
        compiled = compile_function(source)
    except UnsupportedSyntax as exc:
        stage = "Semantic analysis" if "undefined" in str(exc) else "Syntax analysis"
        return _error(stage, str(exc))

    original = from_tac(compiled.tac)
    optimized, log = optimize_stmts(original)

    lo, hi = (int(lo), int(hi)) if int(lo) <= int(hi) else (int(hi), int(lo))
    rng = random.Random(seed)
    tests = [{a: rng.randint(lo, hi) for a in compiled.arg_names} for _ in range(num_tests)]
    checks = []
    for t in tests:
        try:
            python_out = run_source(source, compiled.name, t)
        except Exception as exc:  # noqa: BLE001
            return _error("Execution", f"Your function failed on {t}: {type(exc).__name__}: {exc}")
        tac_out = run(compiled.tac, t).outputs
        orig_out = run_stmts(original, t)
        opt_out = run_stmts(optimized, t)
        checks.append({"inputs": t, "original": orig_out, "optimized": opt_out, "python": python_out,
                       "match": orig_out == opt_out == tac_out})
        if tac_out != python_out:
            return _error("Execution", f"The generated code disagrees with Python on {t} (Python: {python_out}, generated: {tac_out}). "
                                       "This usually means // was used on a negative number; use a positive input range.")
    n_orig, n_opt = len(original), len(optimized)
    removed = n_orig - n_opt
    by_tech: Dict[str, int] = {}
    for e in log:
        by_tech[e["technique"]] = by_tech.get(e["technique"], 0) + 1
    return {
        "ok": True,
        "function": compiled.name,
        "args": compiled.arg_names,
        "source": source,
        "intermediate": tac_to_text(compiled.tac).splitlines(),   # raw front-end TAC (the "Intermediate Code" stage)
        "original": three_address_rows(original),
        "optimized": three_address_rows(optimized),
        "quadruples": quadruple_rows(optimized),
        "log": [dict(e, sno=i) for i, e in enumerate(log, 1)],
        "summary": {
            "original_instructions": n_orig,
            "optimized_instructions": n_opt,
            "removed": removed,
            "optimizations": len(log),
            "techniques": [t for t in TECHNIQUES if t in by_tech],
            "by_technique": [{"technique": t, "count": by_tech[t]} for t in TECHNIQUES if t in by_tech],
            "reduction_pct": round(100.0 * removed / n_orig, 2) if n_orig else 0.0,
            "rounds": max([e["round"] for e in log] + [0]),
        },
        "no_optimization": not log,
        "checks": checks,
        "behaviour_preserved": all(c["match"] for c in checks),
        "settings": {"lo": lo, "hi": hi, "seed": seed, "tests": num_tests},
    }


def optimize_program_json(params_json: str) -> str:
    """JSON in, JSON out: the entry point the browser calls."""
    p = json.loads(params_json)
    return json.dumps(optimize_program(p["source"], p.get("lo", 1), p.get("hi", 20), p.get("seed", 7)))


SAMPLES = {
    "Mixed example (all six techniques)": (1, 20, """def demo(a, b):
    x = 10 * 5
    k = 10
    m = k + 5
    n = m + 0
    unused = 100
    t1 = a + b
    t2 = a + b
    scaled = t1 * 1
    copy_of_b = b
    shifted = copy_of_b + 5
    return n, t2, scaled, shifted
"""),
    "Constant folding and propagation": (1, 20, """def consts(p):
    a = 10
    b = a + 5
    c = 10 * 5
    d = b * c
    return d + p
"""),
    "Common subexpressions": (1, 20, """def common(a, b, c):
    t1 = a * b
    t2 = a * b
    t3 = b * a
    r = t1 + t2 + t3 + c
    return r
"""),
    "Already optimal (nothing to change)": (1, 20, """def plain(a, b):
    s = a + b
    return s
"""),
}
DEFAULT_SOURCE = next(iter(SAMPLES.values()))[2]
