"""A tiny TAC interpreter used both to check correctness and to simulate
execution time via a per-opcode cycle cost model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .tac import Instr

# Cycle cost model: add/sub are single-cycle ALU ops while multiply/divide
# are modeled as costlier, matching typical instruction-latency intuitions
# (add 1, multiply ~3). This gap is what makes strength reduction
# (x*3 -> x+x+x) a genuine speed-for-size trade-off rather than a free win.
# Comparisons and branches take one cycle; labels are not instructions.
CYCLE_COST = {
    "input": 1,
    "const": 1,
    "copy": 1,
    "output": 1,
    "label": 0,
    "jump": 1,
    "branch": 1,
    "bin+": 1,
    "bin-": 1,
    "bin*": 3,
    "bin/": 4,
    "bin<": 1,
    "bin<=": 1,
    "bin>": 1,
    "bin>=": 1,
    "bin==": 1,
    "bin!=": 1,
}

MAX_STEPS = 200_000  # a loop that never ends is reported as an error, not run forever


@dataclass
class RunResult:
    ok: bool
    outputs: List[int] = field(default_factory=list)
    cycles: int = 0
    error: str = ""


def _safe_div(a: int, b: int) -> int:
    """Integer division truncated toward zero (C semantics); div-by-zero -> 0."""
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def compare(op: str, a: int, b: int) -> int:
    if op == "<":
        return int(a < b)
    if op == "<=":
        return int(a <= b)
    if op == ">":
        return int(a > b)
    if op == ">=":
        return int(a >= b)
    if op == "==":
        return int(a == b)
    return int(a != b)


def run(tac: Sequence[Instr], input_values: Dict[str, int], max_steps: int = MAX_STEPS) -> RunResult:
    """Execute a TAC program and return outputs + cycle count.

    ``max_steps`` caps the work: a run that needs more is reported as failed, never run on."""
    labels = {ins[1]: i for i, ins in enumerate(tac) if ins[0] == "label"}
    env: Dict[str, int] = {}
    outputs: List[int] = []
    cycles = 0
    pc = 0
    steps = 0
    try:
        while pc < len(tac):
            steps += 1
            if steps > max_steps:
                raise RuntimeError("step limit exceeded (the program may loop forever)")
            instr = tac[pc]
            pc += 1
            op = instr[0]
            if op == "input":
                _, dest, name = instr
                env[dest] = int(input_values[name])
                cycles += CYCLE_COST["input"]
            elif op == "const":
                _, dest, value = instr
                env[dest] = int(value)
                cycles += CYCLE_COST["const"]
            elif op == "copy":
                _, dest, src = instr
                env[dest] = env[src]
                cycles += CYCLE_COST["copy"]
            elif op == "bin":
                _, dest, o, a, b = instr
                va, vb = env[a], env[b]
                if o == "+":
                    env[dest] = va + vb
                elif o == "-":
                    env[dest] = va - vb
                elif o == "*":
                    env[dest] = va * vb
                elif o == "/":
                    env[dest] = _safe_div(va, vb)
                elif o in ("<", "<=", ">", ">=", "==", "!="):
                    env[dest] = compare(o, va, vb)
                else:
                    raise ValueError(f"unknown operator {o}")
                cycles += CYCLE_COST["bin" + o]
            elif op == "output":
                _, src = instr
                outputs.append(env[src])
                cycles += CYCLE_COST["output"]
            elif op == "label":
                continue
            elif op == "jump":
                pc = labels[instr[1]]
                cycles += CYCLE_COST["jump"]
            elif op == "branch":
                _, cond, then_l, else_l = instr
                pc = labels[then_l] if env[cond] else labels[else_l]
                cycles += CYCLE_COST["branch"]
            else:
                raise ValueError(f"unknown opcode {op}")
        return RunResult(ok=True, outputs=outputs, cycles=cycles)
    except Exception as exc:  # noqa: BLE001 - any interpreter fault means invalid genome
        return RunResult(ok=False, outputs=[], cycles=cycles, error=str(exc))
