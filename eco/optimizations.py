"""The six TAC-level optimization passes the evolutionary search chooses from.

Every benchmark program is generated in single-assignment form (each
temporary is defined exactly once), which keeps the dataflow reasoning
below correct without a full reaching-definitions analysis: a variable's
defining instruction is unique, so "is this operand currently known to be
constant" can be answered with a single forward scan.

Each pass has the signature ``tac -> tac`` (pure, returns a new list) so a
genome (an ordered list of pass names) can simply be *replayed* from the
original program to re-derive any individual's TAC.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .tac import Instr, TAC, dest_of, uses_of

COMMUTATIVE = {"+", "*"}


def constant_folding(tac: Sequence[Instr]) -> TAC:
    """Fold arithmetic whose operands are already known literals.

    Single-hop by design, like constant_propagation and copy_propagation
    below: it only "sees" ``const`` instructions that existed before this
    pass started, so a chain of dependent folds (e.g. ``t1 = c1+c2``, then
    ``t2 = t1+c3``) needs constant_folding applied more than once - once
    per link in the chain - to fully collapse. Combined with the tight
    genome-length cap, this means a program with both a deep constant
    chain *and* a deep copy chain cannot always fully clean up both within
    one genome's budget, which is exactly the kind of scarcity that
    produces genuinely non-dominated (Pareto) trade-offs rather than a
    single dominant optimum.
    """
    const_env: Dict[str, int] = {i[1]: i[2] for i in tac if i[0] == "const"}
    out: TAC = []
    for instr in tac:
        op = instr[0]
        if op == "bin" and instr[3] in const_env and instr[4] in const_env:
            _, dest, o, a, b = instr
            va, vb = const_env[a], const_env[b]
            if o == "+":
                value = va + vb
            elif o == "-":
                value = va - vb
            elif o == "*":
                value = va * vb
            else:  # division - fold only when it divides evenly to stay simple/safe
                if vb == 0 or va % vb != 0:
                    out.append(instr)
                    continue
                value = va // vb
            out.append(("const", dest, value))
        else:
            out.append(instr)
    return out


def constant_propagation(tac: Sequence[Instr]) -> TAC:
    """Replace uses of known-constant variables with fresh literal defs.

    Deliberately a *single-hop* rewrite (like a real optimizer pass run once
    in a fixpoint loop): it only sees variables that were already literal
    ``const`` instructions before this pass started, so a chain such as
    ``c1 = 5; a = c1; b = a`` needs this pass applied more than once (i.e.
    more than one genome slot, across mutation/crossover) to fully collapse
    - which is what gives the genome's pass *order and repetition* real
    consequences instead of every pass reaching its fixpoint in one shot.
    """
    const_env: Dict[str, int] = {i[1]: i[2] for i in tac if i[0] == "const"}
    out: TAC = []
    fresh_id = [0]

    def fresh_const(value: int) -> str:
        fresh_id[0] += 1
        name = f"_pc{fresh_id[0]}"
        out.append(("const", name, value))
        return name

    for instr in tac:
        op = instr[0]
        if op == "copy" and instr[2] in const_env:
            out.append(("const", instr[1], const_env[instr[2]]))
        elif op == "bin":
            _, dest, o, a, b = instr
            na = fresh_const(const_env[a]) if a in const_env else a
            nb = fresh_const(const_env[b]) if b in const_env else b
            out.append(("bin", dest, o, na, nb))
        else:
            out.append(instr)
    return out


def copy_propagation(tac: Sequence[Instr]) -> TAC:
    """Forward-substitute copies so later uses reference the copy's source.

    Single-hop by design (see constant_propagation's docstring): a copy
    chain of depth k needs this pass applied k times across the genome to
    fully collapse, so genome order/repetition - not just which passes are
    present - genuinely changes the result.
    """
    direct = {i[1]: i[2] for i in tac if i[0] == "copy"}

    out: TAC = []
    for instr in tac:
        op = instr[0]
        if op == "copy":
            out.append(instr)
        elif op == "bin":
            _, dest, o, a, b = instr
            out.append(("bin", dest, o, direct.get(a, a), direct.get(b, b)))
        elif op == "output":
            out.append(("output", direct.get(instr[1], instr[1])))
        else:
            out.append(instr)
    return out


def common_subexpression_elimination(tac: Sequence[Instr]) -> TAC:
    """Reuse an earlier temp when an identical arithmetic expression recurs."""
    seen: Dict[tuple, str] = {}
    out: TAC = []
    for instr in tac:
        if instr[0] == "bin":
            _, dest, o, a, b = instr
            key = (o, a, b)
            if o in COMMUTATIVE:
                key = (o,) + tuple(sorted((a, b)))
            if key in seen:
                out.append(("copy", dest, seen[key]))
            else:
                seen[key] = dest
                out.append(instr)
        else:
            out.append(instr)
    return out


def dead_code_elimination(tac: Sequence[Instr]) -> TAC:
    """Backward liveness sweep: drop definitions nothing ever uses."""
    live = set()
    for instr in tac:
        if instr[0] == "output":
            live.update(uses_of(instr))

    out: TAC = []
    for instr in reversed(tac):
        d = dest_of(instr)
        if d is not None and d not in live:
            continue  # dead definition, drop it
        live.update(uses_of(instr))
        out.append(instr)
    out.reverse()
    return out


def algebraic_simplification(tac: Sequence[Instr]) -> TAC:
    """Simplify identities such as x+0, x*1, x*0, x-0, x/1, x-x."""
    const_env: Dict[str, int] = {}
    out: TAC = []
    for instr in tac:
        op = instr[0]
        if op == "const":
            const_env[instr[1]] = instr[2]
            out.append(instr)
            continue
        if op != "bin":
            out.append(instr)
            continue

        _, dest, o, a, b = instr
        ca = const_env.get(a)
        cb = const_env.get(b)
        if o == "+":
            if cb == 0:
                out.append(("copy", dest, a))
            elif ca == 0:
                out.append(("copy", dest, b))
            else:
                out.append(instr)
        elif o == "-":
            if cb == 0:
                out.append(("copy", dest, a))
            elif a == b:
                const_env[dest] = 0
                out.append(("const", dest, 0))
            else:
                out.append(instr)
        elif o == "*":
            if cb == 0 or ca == 0:
                const_env[dest] = 0
                out.append(("const", dest, 0))
            elif cb == 1:
                out.append(("copy", dest, a))
            elif ca == 1:
                out.append(("copy", dest, b))
            else:
                out.append(instr)
        elif o == "/":
            if cb == 1:
                out.append(("copy", dest, a))
            elif a == b and cb != 0:
                const_env[dest] = 1
                out.append(("const", dest, 1))
            else:
                out.append(instr)
        else:
            out.append(instr)
    return out


OPTIMIZATIONS = {
    "constant_folding": constant_folding,
    "constant_propagation": constant_propagation,
    "copy_propagation": copy_propagation,
    "cse": common_subexpression_elimination,
    "dead_code_elimination": dead_code_elimination,
    "algebraic_simplification": algebraic_simplification,
}

OPTIMIZATION_NAMES: List[str] = list(OPTIMIZATIONS.keys())


def apply_genome(original_tac: Sequence[Instr], genome: Sequence[str]) -> TAC:
    """Re-derive a TAC by replaying a genome's optimizations from scratch."""
    tac: TAC = list(original_tac)
    for name in genome:
        tac = OPTIMIZATIONS[name](tac)
    return tac
