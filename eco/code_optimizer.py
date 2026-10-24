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
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .cfg import CONTROL, GLOBAL_TECHNIQUES, build_cfg, cfg_graph, execute as run_stmts, flow_report, global_optimize, has_control
from .equivalence import prove_equivalent
from .frontend import UnsupportedSyntax, compile_function, run_source
from .interpreter import CYCLE_COST, run
from .metrics import evaluate
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
        if self.op == "label":
            return f"{self.a}:"
        if self.op == "goto":
            return f"goto {self.a}"
        if self.op == "branch":
            return f"if {self.a} goto {self.b[0]} else {self.b[1]}"
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
        elif op == "label":
            out.append(Stmt(None, "label", ins[1]))
        elif op == "jump":
            out.append(Stmt(None, "goto", ins[1]))
        elif op == "branch":
            out.append(Stmt(None, "branch", rename.get(ins[1], ins[1]), (ins[2], ins[3])))
    for n, s in enumerate(out, 1):
        s.origin = n
    return out


def stmts_to_tac(stmts: Sequence[Stmt]) -> TAC:
    """Lower 3-address statements back to the TAC the shared metrics code runs.

    Integer literal operands are pooled into one ``const`` each, as the front end does
    for straight-line code. With control flow every use gets its own ``const``, since a
    constant defined in one block does not exist on the path into another.
    """
    control = has_control(stmts)
    tac: TAC = []
    pool: Dict[int, str] = {}
    counter = [0]

    def operand(x):
        if isinstance(x, str):
            return x
        if not control and x in pool:
            return pool[x]
        counter[0] += 1
        name = f"_lit{counter[0]}"
        if not control:
            pool[x] = name
        tac.append(("const", name, x))
        return name

    for s in stmts:
        if s.op == "input":
            tac.append(("input", s.dest, s.a))
        elif s.op == "=":
            tac.append(("const", s.dest, s.a) if isinstance(s.a, int) else ("copy", s.dest, s.a))
        elif s.op == "print":
            tac.append(("output", operand(s.a)))
        elif s.op == "label":
            tac.append(("label", s.a))
        elif s.op == "goto":
            tac.append(("jump", s.a))
        elif s.op == "branch":
            tac.append(("branch", operand(s.a), s.b[0], s.b[1]))
        else:
            tac.append(("bin", s.dest, s.op, operand(s.a), operand(s.b)))
    return tac


def _stmt_cycles(s: Stmt) -> int:
    if s.op == "input":
        return CYCLE_COST["input"]
    if s.op == "=":
        return CYCLE_COST["copy"] if isinstance(s.a, str) else CYCLE_COST["const"]
    if s.op == "print":
        return CYCLE_COST["output"]
    if s.op == "label":
        return CYCLE_COST["label"]
    if s.op == "goto":
        return CYCLE_COST["jump"]
    if s.op == "branch":
        return CYCLE_COST["branch"]
    return CYCLE_COST["bin" + s.op]


def _change_pct(old: float, new: float) -> float:
    return round(100.0 * (new - old) / old, 2) if old else 0.0


# ------------------------------------------------------------------ passes
AS_RULES = {
    "Adding zero does not change the value": ("x + 0 = x", "the other operand is the literal 0"),
    "Subtracting zero does not change the value": ("x - 0 = x", "the subtrahend is the literal 0"),
    "Multiplying by zero always gives zero": ("x * 0 = 0", "one operand is the literal 0"),
    "Multiplying by one does not change the value": ("x * 1 = x", "the other operand is the literal 1"),
    "Dividing by one does not change the value": ("x / 1 = x", "the divisor is the literal 1"),
}


def _event(technique: str, before: Stmt, after: Optional[Stmt], reason: str, rule: str, precondition: str,
           merged_into: Optional[int] = None) -> dict:
    """One logged change. ``origin`` is the original line number; ``merged_into`` is the original
    line whose result a common-subexpression rewrite reuses, when the pass knows it."""
    return {"technique": technique, "original": before.text(),
            "optimized": after.text() if after is not None else "(removed)",
            "reason": reason, "rule": rule, "precondition": precondition, "origin": before.origin,
            "merged_into": merged_into,
            "cycles_saved": _stmt_cycles(before) - (_stmt_cycles(after) if after is not None else 0),
            "instrs_removed": 1 if after is None else 0}


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
                events.append(_event("Constant Propagation", s, new, "Known constant value substituted (" + ", ".join(known) + ")",
                                     "if x = c and x is used later, replace x by c",
                                     "x is assigned only once, to a literal, and that assignment reaches this use"))
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
                events.append(_event("Constant Folding", s, new, f"Evaluated constant expression {s.a} {s.op} {s.b} at compile time",
                                     f"{s.a} {s.op} {s.b} = {value} (evaluated with the same arithmetic the program uses)",
                                     "both operands are integer literals; a division is folded only when it is exact and the divisor is not zero"))
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
            events.append(_event("Algebraic Simplification", s, new, why, *AS_RULES[why]))
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
                events.append(_event("Copy Propagation", s, new, "Copy replaced by its source (" + ", ".join(used) + ")",
                                     "if y = x and y is used later, replace y by x",
                                     "y is a copy of x, and neither y nor x is reassigned before this use"))
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
    holder_line: Dict[tuple, int] = {}   # expression -> original line that computed it
    out, events = [], []
    for s in stmts:
        new = s
        is_bin = s.op in ("+", "-", "*", "/")
        if is_bin:
            k = key(s)
            if k in available:
                new = s.copy_with(op="=", a=available[k], b=None)
                events.append(_event("Common Subexpression Elimination", s, new,
                                     f"Expression {s.a} {s.op} {s.b} was already calculated in {available[k]}",
                                     f"{s.a} {s.op} {s.b} was already computed into {available[k]}, so reuse {available[k]}",
                                     "same operator and operands (order ignored for + and *), and no operand is reassigned in between",
                                     merged_into=holder_line.get(k)))
        d = new.dest
        if d is not None:
            for k2 in [k2 for k2, v in available.items() if v == d or d in k2[1:]]:
                del available[k2]
            if is_bin and new.op != "=" and d not in s.uses():
                available[key(s)] = d
                holder_line[key(s)] = s.origin
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
            events.append(_event("Dead Code Elimination", s, None, f"{s.dest} is never used, so the statement has no effect on the output",
                                 "a definition whose value is never read has no effect on the output",
                                 f"{s.dest} is not read by any later statement and is not printed"))
        else:
            live.discard(s.dest)
            live.update(s.uses())
            kept.append(s)
    kept.reverse()
    events.reverse()
    return kept, events


PIPELINE = [constant_propagation, constant_folding, algebraic_simplification,
            copy_propagation, common_subexpression_elimination, dead_code_elimination]
NAME_OF = dict(zip(PIPELINE, TECHNIQUES))
BY_NAME = {name: fn for fn, name in NAME_OF.items()}


def optimize_stmts(stmts: List[Stmt], passes: Optional[List[str]] = None) -> Tuple[List[Stmt], List[dict], List[dict]]:
    """Run all techniques repeatedly until a full round changes nothing.

    Returns the optimized statements, the change log, and a trace with one entry per
    pass per round. Only dead code elimination removes statements, so a pass's
    statement-count drop is credited to that pass. With ``passes`` given (manual mode),
    exactly those techniques run once each, in that order.
    """
    if passes is None:
        pipeline, rounds = PIPELINE, MAX_ROUNDS
    else:
        pipeline, rounds = [BY_NAME[name] for name in passes], 1
    current = [s.copy_with() for s in stmts]
    log: List[dict] = []
    trace: List[dict] = []
    for rnd in range(1, rounds + 1):
        changed = False
        for pass_fn in pipeline:
            before = len(current)
            current, events = pass_fn(current)
            for e in events:
                e["round"] = rnd
            trace.append({"round": rnd, "technique": NAME_OF[pass_fn],
                          "statements_before": before, "statements_after": len(current), "events": len(events)})
            if events:
                changed = True
                log.extend(events)
        if not changed:
            break
    return current, log, trace


def selection_views(log: List[dict], trace: List[dict], techniques: List[str] = TECHNIQUES) -> Dict[str, dict]:
    """Chart data and log rows for every non-empty combination of techniques.

    Keyed by a bitmask over TECHNIQUES, so the browser only looks up a view and never
    computes one. A 'combined' bar is the union of the selected techniques.
    """
    per = {t: {"removed": sum(x["statements_before"] - x["statements_after"] for x in trace if x["technique"] == t),
               "changes": sum(1 for e in log if e["technique"] == t)} for t in techniques}
    views = {}
    for mask in range(1, 1 << len(techniques)):
        sel = [t for i, t in enumerate(techniques) if mask >> i & 1]
        views[str(mask)] = {
            "techniques": sel,
            "bars": [dict(label=t, **per[t]) for t in sel],
            "combined": {"removed": sum(per[t]["removed"] for t in sel), "changes": sum(per[t]["changes"] for t in sel)},
            "sno": [e["sno"] for e in log if e["technique"] in sel],
        }
    return views


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


# ------------------------------------------------------------------ changed lines
# Reasons the passes log start with a capital letter; these first words are plain English, so a
# reason that begins with one can be lowercased to read as a clause. Anything else (a variable name)
# is left alone.
_CLAUSE_WORDS = {"Adding", "Subtracting", "Multiplying", "Dividing", "Known", "Copy", "Expression", "Evaluated"}


def _clause(text: str) -> str:
    first = text.split(" ", 1)[0]
    return text[0].lower() + text[1:] if first in _CLAUSE_WORDS else text


def _because(step: dict) -> str:
    """The plain-English 'because ...' clause for one logged change."""
    tech, reason = step["technique"], step["reason"]
    if tech == "Constant Folding":
        if step["after"].startswith("goto"):
            return "the branch condition is a known constant, so the branch always goes the same way"
        return "both operands are constants, so the result can be computed once at compile time"
    if tech in ("Constant Propagation", "Global Constant Propagation"):
        detail = reason[reason.find("(") + 1:reason.rfind(")")] if "(" in reason else reason
        return f"it reads a variable that always holds a known constant ({detail}), so the constant is used directly"
    if tech == "Copy Propagation":
        detail = reason[reason.find("(") + 1:reason.rfind(")")] if "(" in reason else reason
        return f"it reads a copy of another variable ({detail}), so the original variable is used instead"
    return _clause(reason)


def _sentence(step: dict) -> str:
    if step["after"] == "(removed)":
        return f"{step['before']} was removed because {_because(step)}"
    return f"{step['before']} became {step['after']} because {_because(step)}"


def line_change_rows(original: Sequence[Stmt], log: Sequence[dict]) -> List[dict]:
    """One row per original line the optimizer touched, built only from the change log.

    A line touched by several passes keeps its whole chain (original -> intermediate -> final).
    ``origin`` is the line's number in the original listing, which is its position in the
    "Original 3-Address Code" table. The front end does not track Python source lines, so there
    is no source-line column.
    """
    text_of = {s.origin: s.text() for s in original}
    block_of = {s.origin: b.id for b in build_cfg(original) for s in b.stmts}
    chains: Dict[int, List[dict]] = {}
    for e in log:
        chains.setdefault(e["origin"], []).append(
            {"sno": e["sno"], "technique": e["technique"], "before": e["original"], "after": e["optimized"],
             "round": e["round"], "reason": e["reason"], "merged_into": e.get("merged_into")})
    rows = []
    for origin in sorted(chains):
        steps = chains[origin]
        last = steps[-1]
        removed = last["after"] == "(removed)"
        rows.append({
            "line": origin,
            "block": block_of.get(origin, 0),
            "original": text_of.get(origin, steps[0]["before"]),
            "final": None if removed else last["after"],
            "removed": removed,
            # the original line whose result a common-subexpression step reused, even if the line was deleted later
            "merged_into": next((st["merged_into"] for st in reversed(steps) if st["merged_into"]), None),
            "techniques": list(dict.fromkeys(st["technique"] for st in steps)),
            "steps": steps,
            "sentences": [_sentence(st) for st in steps],
            "reason": "; then ".join(_sentence(st) for st in steps),
        })
    return rows


def cfg_views(original: Sequence[Stmt], optimized: Sequence[Stmt], rows: Sequence[dict]) -> dict:
    """The original and the optimized control-flow graph, with changed and removed parts marked.

    A statement keeps its ``origin`` through optimization, so ``rows`` (the changed lines) say
    which drawn statements were rewritten or deleted. A block of the original is 'removed' when the
    optimized program has no block with the same label.
    """
    changed = {r["line"] for r in rows if not r["removed"]}
    deleted = {r["line"] for r in rows if r["removed"]}
    before, after = cfg_graph(original), cfg_graph(optimized)
    kept = {b["label"] for b in after["blocks"] if b["label"] is not None}
    for b in before["blocks"]:
        for st in b["stmts"]:
            st["status"] = "removed" if st["origin"] in deleted else ("changed" if st["origin"] in changed else "same")
        gone = b["label"] is not None and b["label"] not in kept
        b["status"] = "removed" if gone else ("changed" if any(st["status"] != "same" for st in b["stmts"]) else "same")
    for b in after["blocks"]:
        for st in b["stmts"]:
            st["status"] = "changed" if st["origin"] in changed else "same"
        b["status"] = "changed" if any(st["status"] != "same" for st in b["stmts"]) else "same"
    return {"original": before, "optimized": after,
            "removed_blocks": sum(1 for b in before["blocks"] if b["status"] == "removed")}


# ------------------------------------------------------------------ entry points
def local_optimize(stmts: List[Stmt]) -> List[Stmt]:
    """What a block-local optimizer can do: the straight-line passes, run inside each basic block.

    Dead code elimination and algebraic simplification are left out here: the first needs
    liveness across blocks, and the second's constant tracking assumes single assignment.
    """
    out: List[Stmt] = []
    passes = [constant_propagation, constant_folding, copy_propagation, common_subexpression_elimination]
    for block in build_cfg(stmts):
        head = [st for st in block.stmts if st.op == "label"]
        tail = [st for st in block.stmts if st.op in ("goto", "branch")]
        body = [st for st in block.stmts if st.op not in CONTROL]
        changed = True
        while changed:
            changed = False
            for fn in passes:
                body, events = fn(body)
                changed = changed or bool(events)
        out.extend(head + body + tail)
    return out


FLOW_EXAMPLE = """def example(n, k):
    x = 2 * 3
    if n < k:
        y = x + n
    else:
        y = x + n
    z = x + n
    unused = 99
    return y + z
"""


def _error(stage: str, message: str) -> dict:
    return {"ok": False, "stage": stage, "error": message}


def _boundary_inputs(names: Sequence[str], allow_negative: bool) -> List[Dict[str, int]]:
    values = [0, 1] + ([-1] if allow_negative else []) + [1_000_000]
    return [{a: v for a in names} for v in values]


def _verdict(checks: List[dict], proved: bool, bound: Optional[int] = None) -> dict:
    bad = next((c for c in checks if not c["match"]), None)
    passed, total = sum(c["match"] for c in checks), len(checks)
    if bad:
        return {"level": "fail", "text": f"NOT equivalent: first mismatch at {bad['inputs']}: "
                                         f"original {bad['original']}, optimized {bad['optimized']}"}
    if proved and bound:
        return {"level": "proved", "text": f"Equivalent: proved for every input whose loops finish within {bound} iterations, "
                                           f"and passed {passed}/{total} compile-time tests"}
    if proved:
        return {"level": "proved", "text": f"Equivalent: proved symbolically and passed {passed}/{total} compile-time tests"}
    if total:
        return {"level": "tested", "text": f"Equivalent on {passed}/{total} compile-time tests; not proved symbolically"}
    return {"level": "unverified", "text": "Not verified: no compile-time tests ran"}


def optimize_program(source: str, lo: int = 1, hi: int = 20, seed: int = 7, num_tests: int = 5,
                     passes: Optional[List[str]] = None, compare: bool = True) -> dict:
    """Run the whole workflow on a user's function and return everything the tab shows.

    ``compare`` runs every other optimization method on the same program for the comparison card;
    the manual-vs-automatic call below turns it off because it only needs two numbers.
    """
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

    if passes is not None and any(name not in BY_NAME for name in passes):
        return _error("Input", f"Unknown optimization in {passes}")
    original = from_tac(compiled.tac)
    control = has_control(original)
    if control:
        if passes is not None:
            return _error("Input", "The manual pipeline applies to straight-line code only")
        optimized, records, trace = global_optimize(original)
        log = [dict(_event(r["technique"], r["before"], r["after"], r["reason"], r["rule"], r["precondition"]),
                    round=r["round"]) for r in records]
    else:
        optimized, log, trace = optimize_stmts(original, passes)
    techniques = GLOBAL_TECHNIQUES if control else TECHNIQUES
    log = [dict(e, sno=i) for i, e in enumerate(log, 1)]
    line_changes = line_change_rows(original, log)

    lo, hi =(int(lo), int(hi)) if int(lo) <= int(hi) else (int(hi), int(lo))
    rng = random.Random(seed)
    has_division = any(i[0] == "bin" and i[2] == "/" for i in compiled.tac)
    tests = [("random", {a: rng.randint(lo, hi) for a in compiled.arg_names}) for _ in range(num_tests)]
    tests += [("boundary", t) for t in _boundary_inputs(compiled.arg_names, allow_negative=True)]
    checks = []
    started = time.perf_counter()
    for kind, t in tests:
        try:
            orig_out = run_stmts(original, t)
            opt_out = run_stmts(optimized, t)
            tac_res = run(compiled.tac, t)
            if not tac_res.ok:
                raise RuntimeError(tac_res.error)
        except RuntimeError as exc:  # the program does not finish on this input
            if kind == "boundary":
                continue
            return _error("Execution", f"Your function did not finish on {t}: {exc}")
        # Python floors on // while the generated code truncates, so for negative inputs the
        # reference is the original generated code; the page says so.
        python_out = None
        if not (has_division and any(v < 0 for v in t.values())):
            try:
                python_out = run_source(source, compiled.name, t)
            except Exception as exc:  # noqa: BLE001
                if kind == "boundary":
                    continue  # e.g. division by zero at a boundary input: not a valid test point
                return _error("Execution", f"Your function failed on {t}: {type(exc).__name__}: {exc}")
        tac_out = tac_res.outputs
        checks.append({"kind": kind, "inputs": t, "original": orig_out, "optimized": opt_out, "python": python_out,
                       "match": orig_out == opt_out == tac_out})
        if python_out is not None and tac_out != python_out:
            return _error("Execution", f"The generated code disagrees with Python on {t} (Python: {python_out}, generated: {tac_out}). "
                                       "This usually means // was used on a negative number; use a positive input range.")
    check_ms = (time.perf_counter() - started) * 1000
    n_orig = sum(1 for s in original if s.op != "label")
    n_opt = sum(1 for s in optimized if s.op != "label")
    removed = n_orig - n_opt
    by_tech: Dict[str, int] = {}
    for e in log:
        by_tech[e["technique"]] = by_tech.get(e["technique"], 0) + 1

    before = evaluate(compiled.tac, [t for _, t in tests]).fitness.as_dict()
    after_tac = stmts_to_tac(optimized)
    after = evaluate(after_tac, [t for _, t in tests]).fitness.as_dict()
    temps = lambda tac: len({dest_of(i) for i in tac if dest_of(i) is not None})
    metrics = [{"key": k, "objective": label, "original": before[k], "optimized": after[k],
                "change_pct": _change_pct(before[k], after[k])}
               for k, label in (("exec_time", "exec_time (simulated cycles)"), ("instr_count", "instr_count"),
                                ("code_size", "code_size"), ("arith_ops", "arith_ops"))]
    metrics.append({"key": "temporaries", "objective": "temporaries/variables used",
                    "original": temps(compiled.tac), "optimized": temps(after_tac),
                    "change_pct": _change_pct(temps(compiled.tac), temps(after_tac))})
    equivalence = prove_equivalent(original, optimized)
    per_technique = [{"technique": t,
                      "changes": sum(1 for e in log if e["technique"] == t),
                      "instructions_removed": sum(e["instrs_removed"] for e in log if e["technique"] == t),
                      "cycles_saved": sum(e["cycles_saved"] for e in log if e["technique"] == t)}
                     for t in techniques]
    result = {
        "ok": True,
        "function": compiled.name,
        "args": compiled.arg_names,
        "source": source,
        "intermediate": tac_to_text(compiled.tac).splitlines(),   # raw front-end TAC (the "Intermediate Code" stage)
        "original": three_address_rows(original),
        "optimized": three_address_rows(optimized),
        "quadruples": quadruple_rows(optimized),
        "log": log,
        "line_changes": line_changes,
        "cfg": cfg_views(original, optimized, line_changes),
        "pass_trace": trace,
        "technique_order": techniques,
        "selection_views": selection_views(log, trace, techniques),
        "flow": ({"control_flow": True, "original": flow_report(original), "optimized": flow_report(optimized),
                  "local_count": sum(1 for st in local_optimize(original) if st.op != "label")}
                 if control else {"control_flow": False}),
        "summary": {
            "original_instructions": n_orig,
            "optimized_instructions": n_opt,
            "removed": removed,
            "optimizations": len(log),
            "lines_changed": len(line_changes),
            "techniques": [t for t in techniques if t in by_tech],
            "by_technique": [{"technique": t, "count": by_tech[t]} for t in techniques if t in by_tech],
            "reduction_pct": round(100.0 * removed / n_orig, 2) if n_orig else 0.0,
            "rounds": max([e["round"] for e in log] + [0]),
        },
        "no_optimization": not log,
        "equivalence": equivalence,
        "compile_check": {"ms": round(check_ms, 3), "passed": sum(c["match"] for c in checks), "total": len(checks),
                          "saved_cycles_per_call": round(before["exec_time"] - after["exec_time"], 2),
                          "note": "one-time compile-time validation; the optimized program itself is never slowed down by this"},
        "verdict": _verdict(checks, equivalence["proved"], equivalence["bound"]),
        "division_warning": (
            "This program uses //. Python floors negative quotients while the generated code truncates, "
            f"so the {sum(1 for c in checks if c['python'] is None)} negative-input checks were compared with the original "
            "generated code, not with Python. Python is not a reference for them."
            if has_division and any(c["python"] is None for c in checks) else None),
        "metrics": metrics,
        "benefit_pct": 0.0 - _change_pct(before["exec_time"], after["exec_time"]),
        "per_technique": per_technique,
        "checks": checks,
        "behaviour_preserved": all(c["match"] for c in checks),
        "settings": {"lo": lo, "hi": hi, "seed": seed, "tests": num_tests},
    }
    result["mode"] = "manual" if passes is not None else "automatic"
    result["passes"] = list(passes) if passes is not None else None
    if compare:
        from .comparison import compare_methods   # imported here: comparison.py imports this module
        hold = random.Random(seed + 1000)   # hold-out inputs: used only to verify, never to search
        holdout = [{a: hold.randint(lo - 3, hi + 3) for a in compiled.arg_names} for _ in range(6)]
        result["method_comparison"] = compare_methods(compiled.tac, original, [t for _, t in tests], holdout)
    if passes is not None:
        auto = optimize_program(source, lo, hi, seed, num_tests, compare=False)
        result["comparison"] = {
            "custom_benefit_pct": result["benefit_pct"], "custom_optimized_instructions": n_opt,
            "automatic_benefit_pct": auto["benefit_pct"], "automatic_optimized_instructions": auto["summary"]["optimized_instructions"],
        }
    return result


def optimize_program_json(params_json: str) -> str:
    """JSON in, JSON out: the entry point the browser calls."""
    p = json.loads(params_json)
    return json.dumps(optimize_program(p["source"], p.get("lo", 1), p.get("hi", 20), p.get("seed", 7),
                                       passes=p.get("passes")))


SAMPLES = {
    "Integer division (shows the Python-floor note)": (1, 20, """def quotient(a, b):
    half = a // 2
    k = 10
    m = k + 5
    same = a // 2
    return half + same + m + b
"""),
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
