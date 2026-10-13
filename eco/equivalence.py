"""Symbolic equivalence check for straight-line 3-address statements.

Each output is rewritten as a polynomial over the input variables, with terms
sorted and coefficients merged. ``+``, ``-`` and ``*`` are exact in this form. A
division is not expanded: it becomes an opaque atom built from the canonical
forms of its operands, so two divisions are identified only when their operands
are identical. Division by the literal 0 or 1 and division of two literals are
evaluated exactly, using the same semantics as the interpreter.

Two programs are reported PROVED only when every output has an identical
canonical form. Anything else is NOT PROVED, never a false proof.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .interpreter import _safe_div

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


def symbolic_outputs(stmts: Sequence) -> List[str]:
    """Canonical polynomial text of every printed value, in print order."""
    env: Dict[str, Poly] = {}
    outputs: List[str] = []

    def val(x) -> Poly:
        return _const(x) if isinstance(x, int) else env[x]

    for s in stmts:
        if s.op == "input":
            env[s.dest] = {(s.a,): 1}
        elif s.op == "=":
            env[s.dest] = val(s.a)
        elif s.op == "print":
            outputs.append(canon(val(s.a)))
        else:
            env[s.dest] = _binop(s.op, val(s.a), val(s.b))
    return outputs


def prove_equivalent(original: Sequence, optimized: Sequence) -> dict:
    a = symbolic_outputs(original)
    b = symbolic_outputs(optimized)
    rows = [{"index": i + 1, "original": x, "optimized": y, "match": x == y}
            for i, (x, y) in enumerate(zip(a, b))]
    same_count = len(a) == len(b) and len(a) > 0
    proved = same_count and all(r["match"] for r in rows)
    return {"proved": proved, "same_output_count": len(a) == len(b), "outputs": rows}
