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

from .cfg import COMPARE, build_cfg, dominators, reachable
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


def _merge(bid: int, preds: List[Tuple[int, str]], out_env: Dict[int, dict], blocks, dom) -> Optional[dict]:
    live = [(p, k) for p, k in preds if p in out_env]
    if not live:
        return None
    if len(live) == 1:
        return dict(out_env[live[0][0]])
    branches = [d for d in dom.get(bid, []) if d != bid and blocks[d].stmts
                and blocks[d].stmts[-1].op == "branch"]
    if not branches or len(live) != 2:
        return None
    d = max(branches)
    cond_stmt = blocks[d].stmts[-1]
    cond = out_env[d][cond_stmt.a] if isinstance(cond_stmt.a, str) else _const(cond_stmt.a)
    t_side, f_side = blocks[d].succ[0][0], blocks[d].succ[1][0]
    sides: Dict[str, dict] = {}
    for p, k in live:
        if p == d:
            side = k
        elif t_side in dom.get(p, []):
            side = "true"
        elif f_side in dom.get(p, []):
            side = "false"
        else:
            return None
        sides[side] = out_env[p]
    if set(sides) != {"true", "false"}:
        return None
    tv, fv = sides["true"], sides["false"]
    if _is_const(cond):
        return dict(tv if _const_value(cond) else fv)
    text = canon(cond)
    return {var: _ite(text, tv[var], fv[var]) for var in tv.keys() & fv.keys()}


def symbolic_program(stmts: Sequence) -> Optional[List[str]]:
    """Canonical forms of the printed values, or None when the program has a loop
    or prints somewhere other than its last block."""
    blocks = build_cfg(stmts)
    if any(t <= b.id for b in blocks for t, _ in b.succ):
        return None
    reach = reachable(blocks)
    dom = dominators(blocks)
    preds: Dict[int, List[Tuple[int, str]]] = {b.id: [] for b in blocks}
    for b in blocks:
        for t, k in b.succ:
            preds[t].append((b.id, k))
    last = blocks[-1].id
    out_env: Dict[int, dict] = {}
    outputs: List[str] = []
    for b in blocks:
        if b.id not in reach:
            continue
        env = {} if b.id == 0 else _merge(b.id, preds[b.id], out_env, blocks, dom)
        if env is None:
            return None

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
    return outputs


def prove_equivalent(original: Sequence, optimized: Sequence) -> dict:
    a = symbolic_program(original)
    b = symbolic_program(optimized)
    control = any(s.op in ("label", "goto", "branch") for s in list(original) + list(optimized))
    if a is None or b is None:
        return {"proved": False, "same_output_count": False, "outputs": [], "control_flow": control,
                "reason": "no symbolic proof: the program has a loop, or prints inside a branch"}
    rows = [{"index": i + 1, "original": x, "optimized": y, "match": x == y}
            for i, (x, y) in enumerate(zip(a, b))]
    same_count = len(a) == len(b) and len(a) > 0
    proved = same_count and all(r["match"] for r in rows)
    return {"proved": proved, "same_output_count": len(a) == len(b), "outputs": rows, "control_flow": control}
