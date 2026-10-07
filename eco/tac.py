"""Three-Address Code (TAC) representation used throughout the optimizer.

An instruction is a plain tuple so it is cheap to create, copy and hash
during the millions of GA evaluations performed across the population:

    ("input", dest, name)     dest = <parameter name>
    ("const", dest, value)    dest = <int literal>
    ("copy",  dest, src)      dest = src
    ("bin",   dest, op, a, b) dest = a <op> b        op in {'+','-','*','/'}
    ("output", src)           emit(src)

A TAC "program" is simply a list of such instructions, always in
single-assignment-friendly straight-line form (no control flow), which is
enough to exercise all six classic optimizations the assignment asks for.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

Instr = Tuple
TAC = List[Instr]

ARITH_OPS = ("+", "-", "*", "/")


def dest_of(instr: Instr):
    """Return the variable an instruction defines, or None (e.g. output)."""
    op = instr[0]
    if op in ("input", "const", "copy", "bin"):
        return instr[1]
    return None


def uses_of(instr: Instr) -> Tuple[str, ...]:
    """Return the variables an instruction reads."""
    op = instr[0]
    if op == "copy":
        return (instr[2],)
    if op == "bin":
        return (instr[3], instr[4])
    if op == "output":
        return (instr[1],)
    return ()


def clone(tac: Sequence[Instr]) -> TAC:
    return list(tac)


def render_source(tac: Sequence[Instr], input_names: Sequence[str], func_name: str = "prog") -> str:
    """Render a pseudo-C rendition of the TAC for the "source code" field."""
    lines = [f"int {func_name}({', '.join('int ' + n for n in input_names)}) {{"]
    for instr in tac:
        op = instr[0]
        if op == "input":
            _, dest, name = instr
            lines.append(f"    int {dest} = {name};")
        elif op == "const":
            _, dest, value = instr
            lines.append(f"    int {dest} = {value};")
        elif op == "copy":
            _, dest, src = instr
            lines.append(f"    int {dest} = {src};")
        elif op == "bin":
            _, dest, o, a, b = instr
            lines.append(f"    int {dest} = {a} {o} {b};")
        elif op == "output":
            _, src = instr
            lines.append(f"    output({src});")
    lines.append("}")
    return "\n".join(lines)


def tac_to_text(tac: Sequence[Instr]) -> str:
    """Compact one-instruction-per-line TAC listing, used in reports."""
    out = []
    for instr in tac:
        op = instr[0]
        if op == "input":
            out.append(f"{instr[1]} = INPUT {instr[2]}")
        elif op == "const":
            out.append(f"{instr[1]} = {instr[2]}")
        elif op == "copy":
            out.append(f"{instr[1]} = {instr[2]}")
        elif op == "bin":
            out.append(f"{instr[1]} = {instr[3]} {instr[2]} {instr[4]}")
        elif op == "output":
            out.append(f"OUTPUT {instr[1]}")
    return "\n".join(out)
