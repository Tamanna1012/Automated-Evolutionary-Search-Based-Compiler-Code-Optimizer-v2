"""A tiny TAC interpreter used both to check correctness and to simulate
execution time via a per-opcode cycle cost model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .tac import Instr

# Cycle cost model: multiply/divide are modeled as costlier than add/sub,
# matching typical instruction-latency intuitions used in real cost models.
CYCLE_COST = {
    "input": 1,
    "const": 1,
    "copy": 1,
    "output": 1,
    "bin+": 2,
    "bin-": 2,
    "bin*": 3,
    "bin/": 4,
}


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
    q = a / b
    return int(q) if q >= 0 else -int(-q)


def run(tac: Sequence[Instr], input_values: Dict[str, int]) -> RunResult:
    """Execute a straight-line TAC program and return outputs + cycle count."""
    env: Dict[str, int] = {}
    outputs: List[int] = []
    cycles = 0
    try:
        for instr in tac:
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
                else:
                    raise ValueError(f"unknown operator {o}")
                cycles += CYCLE_COST["bin" + o]
            elif op == "output":
                _, src = instr
                outputs.append(env[src])
                cycles += CYCLE_COST["output"]
            else:
                raise ValueError(f"unknown opcode {op}")
        return RunResult(ok=True, outputs=outputs, cycles=cycles)
    except Exception as exc:  # noqa: BLE001 - any interpreter fault means invalid genome
        return RunResult(ok=False, outputs=[], cycles=cycles, error=str(exc))
