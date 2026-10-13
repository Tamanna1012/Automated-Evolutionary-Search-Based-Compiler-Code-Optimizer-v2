"""Symbolic equivalence check for straight-line code and for branches without loops.

Each output is rewritten as a polynomial over the input variables, with terms
sorted and coefficients merged. ``+``, ``-`` and ``*`` are exact in this form. A
division is not expanded: it becomes an opaque atom built from the canonical
forms of its operands. Division by the literal 0 or 1 and division of two literals
are evaluated exactly, using the same semantics as the interpreter.

A comparison is an opaque atom too. Where control flow joins, a variable whose
two arms differ becomes an if-then-else atom on the branch condition. Loops are
not handled: a program with a loop has no symbolic proof.

Two programs are reported PROVED only when every output has an identical
canonical form. Anything else is NOT PROVED, never a false proof.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .cfg import COMPARE, backward_jumps, build_cfg, reachable, unroll_loops
from .interpreter import _safe_div, compare

Poly = Dict[Tuple[str, ...], int]


def _clean(p: Poly) -> Poly:
    return {m: c for m, c in p.items() if c}


def _const(c: int) -> Poly:
    return _clean({(): c})


def _add(p: Poly, q: Poly, sign: int = 1) -> Poly:
    r = dict(p)
    for m, c in q.items():
        r[m] = r.get(m, 0) + sign * c
    return _clean(r)


def _mul(p: Poly, q: Poly) -> Poly:
    r: Poly = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(sorted(m1 + m2))
            r[m] = r.get(m, 0) + c1 * c2
    return _clean(r)


def _is_const(p: Poly) -> bool:
    return all(m == () for m in p)


def _const_value(p: Poly) -> int:
    return p.get((), 0)


def canon(p: Poly) -> str:
    if not p:
        return "0"
    terms = sorted(p.items(), key=lambda t: (len(t[0]), t[0]))
    out = []
    for i, (m, c) in enumerate(terms):
        body = "*".join(m)
        if not m:
            mag = str(abs(c))
        elif abs(c) == 1:
            mag = body
        else:
            mag = f"{abs(c)}*{body}"
        if i == 0:
            out.append(("-" if c < 0 else "") + mag)
        else:
            out.append(f" {'-' if c < 0 else '+'} {mag}")
    return "".join(out)


def _binop(op: str, p: Poly, q: Poly) -> Poly:
    if op == "+":
        return _add(p, q)
    if op == "-":
        return _add(p, q, -1)
    if op == "*":
        return _mul(p, q)
    if _is_const(q):
        d = _const_value(q)
        if d == 0:
            return {}
        if d == 1:
            return p
        if _is_const(p):
            return _const(_safe_div(_const_value(p), d))
    return {(f"div[{canon(p)}|{canon(q)}]",): 1}


def _compare(op: str, p: Poly, q: Poly) -> Poly:
    if _is_const(p) and _is_const(q):
        return _const(compare(op, _const_value(p), _const_value(q)))
    return {(f"cmp[{op}|{canon(p)}|{canon(q)}]",): 1}


def _ite(cond: str, a: Poly, b: Poly) -> Poly:
    if canon(a) == canon(b):
        return a
    return {(f"ite[{cond}|{canon(a)}|{canon(b)}]",): 1}


def _and(a: str, b: str) -> str:
    if a == "1":
        return b
    if b == "1":
        return a
    if a == "0" or b == "0":
        return "0"
    return f"({a} & {b})"


def _or(a: str, b: str) -> str:
    if a == "0":
        return b
    if b == "0":
        return a
    if a == "1" or b == "1":
        return "1"
    return f"({a} | {b})"


def _edge_condition(out_env: dict, src, kind: str) -> str:
    """The condition under which control leaves ``src`` along the edge of this kind."""
    if kind == "":
        return "1"
    cond = src.stmts[-1].a
    value = out_env[cond] if isinstance(cond, str) else _const(cond)
    if _is_const(value):
        taken = _const_value(value) != 0
        return "1" if taken == (kind == "true") else "0"
    text = canon(value)
    return text if kind == "true" else f"!({text})"


def symbolic_program(stmts: Sequence) -> Optional[List[str]]:
    """Canonical forms of the printed values, or None when the program has a loop
    or prints somewhere other than its last block.

    Every block gets a path condition (the input conditions under which it runs).
    A join becomes a chain of if-then-else terms over the incoming path conditions;
    for each input exactly one incoming path is taken, so the last one is the default.
    """
    blocks = build_cfg(stmts)
    if any(t <= b.id for b in blocks for t, _ in b.succ):
        return None
    reach = reachable(blocks)
    preds: Dict[int, List[Tuple[int, str]]] = {b.id: [] for b in blocks}
    for b in blocks:
        for t, k in b.succ:
            preds[t].append((b.id, k))
    last = blocks[-1].id
    out_env: Dict[int, dict] = {}
    out_pc: Dict[int, str] = {}
    outputs: List[str] = []
    for b in blocks:
        if b.id not in reach:
            continue
        if b.id == 0:
            env, pc = {}, "1"
        else:
            incoming = []
            for p, k in preds[b.id]:
                if p not in out_env:
                    continue
                ec = _and(out_pc[p], _edge_condition(out_env[p], blocks[p], k))
                if ec != "0":
                    incoming.append((ec, out_env[p]))
            if not incoming:
                return None
            pc = "0"
            for ec, _ in incoming:
                pc = _or(pc, ec)
            if len(incoming) == 1:
                env = dict(incoming[0][1])
            else:
                env = {}
                for var in set.intersection(*[set(e) for _, e in incoming]):
                    v = incoming[-1][1][var]
                    for ec, e in reversed(incoming[:-1]):
                        v = _ite(ec, e[var], v)
                    env[var] = v

        def val(x, env=env):
            return _const(x) if isinstance(x, int) else env[x]

        try:
            for s in b.stmts:
                if s.op == "input":
                    env[s.dest] = {(s.a,): 1}
                elif s.op == "=":
                    env[s.dest] = val(s.a)
                elif s.op == "print":
                    if b.id != last:
                        return None
                    outputs.append(canon(val(s.a)))
                elif s.op in ("label", "goto", "branch"):
                    continue
                elif s.op in COMPARE:
                    env[s.dest] = _compare(s.op, val(s.a), val(s.b))
                else:
                    env[s.dest] = _binop(s.op, val(s.a), val(s.b))
        except KeyError:
            return None
        out_env[b.id] = env
        out_pc[b.id] = pc
    return outputs


LOOP_BOUND = 8


def prove_equivalent(original: Sequence, optimized: Sequence, bound: int = LOOP_BOUND) -> dict:
    control = any(s.op in ("label", "goto", "branch") for s in list(original) + list(optimized))
    loops = bool(backward_jumps(list(original)) or backward_jumps(list(optimized)))
    if loops:
        original = unroll_loops(original, bound)
        optimized = unroll_loops(optimized, bound) if original is not None else None
    a = symbolic_program(original) if original is not None else None
    b = symbolic_program(optimized) if optimized is not None else None
    if a is None or b is None:
        return {"proved": False, "same_output_count": False, "outputs": [], "control_flow": control, "bound": None,
                "reason": "no symbolic proof: the unrolled program is too large, or a print sits inside a branch"}
    rows = [{"index": i + 1, "original": x, "optimized": y, "match": x == y}
            for i, (x, y) in enumerate(zip(a, b))]
    same_count = len(a) == len(b) and len(a) > 0
    proved = same_count and all(r["match"] for r in rows)
    return {"proved": proved, "same_output_count": len(a) == len(b), "outputs": rows, "control_flow": control,
            "bound": bound if loops else None}
