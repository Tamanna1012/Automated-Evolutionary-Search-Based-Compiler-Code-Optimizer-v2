"""The eight TAC-level optimization passes the evolutionary search chooses from.

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
    """Value-numbering CSE: reuse an earlier temp when the same value recurs.

    Expressions are compared by the *values* of their operands, not their
    names. A forward scan keeps a canonical name for every variable:

    * ``d = copy s`` makes ``d`` an alias of ``s`` (looks through copies),
    * a ``const`` equal to an earlier ``const`` aliases that earlier one,
    * an expression recomputed from the same canonical operands is replaced
      by a copy of the first result (commutative ops compare unordered).

    Because aliases chain, a whole nested repeat such as
    ``(((a+b)+c)+d)`` computed twice collapses in a single application:
    the inner repeat becomes a copy, which makes the next level's operands
    canonical-equal too, and so on. (The older name-based version rewrote
    only one level per application, so deep repeats needed many genome
    slots.) The now-redundant copies are left for copy propagation and dead
    code elimination, as before.

    If a name is ever defined twice (the passes can emit that - see
    constant_propagation) every table is reset, which is always safe.
    """
    canon: Dict[str, str] = {}      # variable -> canonical variable holding the same value
    const_first: Dict[int, str] = {}
    seen: Dict[tuple, str] = {}     # (op, canon a, canon b) -> variable computing it
    defined = set()
    out: TAC = []

    def c(name: str) -> str:
        return canon.get(name, name)

    for instr in tac:
        d = dest_of(instr)
        if d is not None:
            if d in defined:  # redefinition: forget everything, stay correct
                canon.clear()
                const_first.clear()
                seen.clear()
            defined.add(d)

        op = instr[0]
        if op == "copy":
            canon[instr[1]] = c(instr[2])
            out.append(instr)
        elif op == "const":
            first = const_first.setdefault(instr[2], instr[1])
            canon[instr[1]] = first
            out.append(instr)
        elif op == "bin":
            _, dest, o, a, b = instr
            ca, cb = c(a), c(b)
            key = (o,) + (tuple(sorted((ca, cb))) if o in COMMUTATIVE else (ca, cb))
            if key in seen:
                out.append(("copy", dest, seen[key]))
                canon[dest] = c(seen[key])
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


def _name_source(tac: Sequence[Instr], prefix: str):
    """Return a callable yielding variable names not defined anywhere in ``tac``."""
    taken = {dest_of(i) for i in tac if dest_of(i) is not None}
    counter = [0]

    def fresh() -> str:
        while True:
            counter[0] += 1
            name = f"{prefix}{counter[0]}"
            if name not in taken:
                taken.add(name)
                return name

    return fresh


STRENGTH_REDUCTION_FACTORS = (2, 3, 4)


def strength_reduction(tac: Sequence[Instr]) -> TAC:
    """Replace ``x * k`` (k = 2, 3, 4) with a short chain of additions.

    ``x*2 -> x+x``, ``x*3 -> (x+x)+x``, ``x*4 -> (x+x)+(x+x)``. Additions
    are cheaper than a multiply in the cycle model, so ``exec_time`` drops -
    but for k = 3, 4 the multiply becomes *two* arithmetic instructions, so
    ``arith_ops`` and ``code_size`` go up. That is a genuine speed-for-size
    trade-off (k = 2 is a pure win). The now-unused constant is left for
    dead_code_elimination to remove.
    """
    fresh = _name_source(tac, "_sr")
    const_env: Dict[str, int] = {}  # forward scan: value of each name *at this point*
    out: TAC = []
    for instr in tac:
        if instr[0] == "bin" and instr[2] == "*":
            _, dest, _, a, b = instr
            ka, kb = const_env.get(a), const_env.get(b)
            if kb in STRENGTH_REDUCTION_FACTORS and ka is None:
                x, k = a, kb
            elif ka in STRENGTH_REDUCTION_FACTORS and kb is None:
                x, k = b, ka
            else:
                x = None
            if x is not None:
                if k == 2:
                    out.append(("bin", dest, "+", x, x))
                elif k == 3:
                    t = fresh()
                    out.append(("bin", t, "+", x, x))
                    out.append(("bin", dest, "+", t, x))
                else:
                    t = fresh()
                    out.append(("bin", t, "+", x, x))
                    out.append(("bin", dest, "+", t, t))
                const_env.pop(dest, None)
                continue
        d = dest_of(instr)
        if d is not None:
            if instr[0] == "const":
                const_env[d] = instr[2]
            else:
                const_env.pop(d, None)
        out.append(instr)
    return out


def multiply_fusion(tac: Sequence[Instr]) -> TAC:
    """Fuse repeated additions of the same value back into one multiply.

    The size-oriented inverse of strength_reduction: ``(x+x)+x -> x*3`` and
    ``(x+x)+(x+x) -> x*4``. It halves ``arith_ops`` for the chain and shrinks
    ``code_size`` (once the leftover add is dead-code-eliminated) but a
    multiply costs more cycles than the adds it replaces, so ``exec_time``
    rises - the opposite corner of the speed/size trade-off. The needed
    literal is reused if one is already defined, otherwise a fresh ``const``
    is emitted.
    """
    fresh = _name_source(tac, "_mf")
    const_env: Dict[str, int] = {}
    doubled: Dict[str, str] = {}  # t -> x for every live ``t = x + x``
    out: TAC = []

    def const_for(k: int) -> str:
        for name, value in const_env.items():
            if value == k:
                return name
        name = fresh()
        out.append(("const", name, k))
        const_env[name] = k
        return name

    def kill(name: str):
        doubled.pop(name, None)
        for t in [t for t, x in doubled.items() if x == name]:
            del doubled[t]

    for instr in tac:
        if instr[0] == "bin" and instr[2] == "+":
            _, dest, _, p, q = instr
            x, k = None, None
            if p == q and p in doubled:
                x, k = doubled[p], 4
            elif p in doubled and doubled[p] == q:
                x, k = q, 3
            elif q in doubled and doubled[q] == p:
                x, k = p, 3
            if x is not None and x != dest:
                kname = const_for(k)
                kill(dest)
                const_env.pop(dest, None)
                out.append(("bin", dest, "*", x, kname))
                continue
        d = dest_of(instr)
        if d is not None:
            kill(d)
            if instr[0] == "const":
                const_env[d] = instr[2]
            else:
                const_env.pop(d, None)
            if instr[0] == "bin" and instr[2] == "+" and instr[3] == instr[4] != d:
                doubled[d] = instr[3]
        out.append(instr)
    return out


OPTIMIZATIONS = {
    "constant_folding": constant_folding,
    "constant_propagation": constant_propagation,
    "copy_propagation": copy_propagation,
    "cse": common_subexpression_elimination,
    "dead_code_elimination": dead_code_elimination,
    "algebraic_simplification": algebraic_simplification,
    "strength_reduction": strength_reduction,
    "multiply_fusion": multiply_fusion,
}

OPTIMIZATION_NAMES: List[str] = list(OPTIMIZATIONS.keys())


def apply_genome(original_tac: Sequence[Instr], genome: Sequence[str]) -> TAC:
    """Re-derive a TAC by replaying a genome's optimizations from scratch."""
    tac: TAC = list(original_tac)
    for name in genome:
        tac = OPTIMIZATIONS[name](tac)
    return tac
