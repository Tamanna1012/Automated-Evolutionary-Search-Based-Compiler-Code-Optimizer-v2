"""Multi-objective fitness metrics for a TAC program.

Four objectives are tracked, all "lower is better":
  * exec_time   - average simulated cycles across the benchmark's test-input sets
  * instr_count - static number of TAC instructions
  * code_size   - static weighted "bytes" a naive code generator would emit
  * arith_ops   - static number of arithmetic ('bin') instructions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .interpreter import run
from .tac import Instr

# Weighted "bytes" per opcode, standing in for a naive bytecode emitter.
CODE_SIZE_WEIGHT = {
    "input": 2,
    "const": 4,
    "copy": 3,
    "bin": 6,
    "output": 3,
}


@dataclass
class Fitness:
    exec_time: float
    instr_count: int
    code_size: int
    arith_ops: int

    def as_tuple(self):
        return (self.exec_time, self.instr_count, self.code_size, self.arith_ops)

    def as_dict(self):
        return {
            "exec_time": self.exec_time,
            "instr_count": self.instr_count,
            "code_size": self.code_size,
            "arith_ops": self.arith_ops,
        }


@dataclass
class EvalResult:
    fitness: Fitness
    valid: bool
    outputs_per_test: List[List[int]] = field(default_factory=list)


def static_metrics(tac: Sequence[Instr]):
    instr_count = len(tac)
    arith_ops = sum(1 for i in tac if i[0] == "bin")
    code_size = sum(CODE_SIZE_WEIGHT[i[0]] for i in tac)
    return instr_count, arith_ops, code_size


def evaluate(tac: Sequence[Instr], test_input_sets: Sequence[Dict[str, int]],
             expected_outputs: Sequence[List[int]] = None) -> EvalResult:
    """Run the TAC on every test-input set and compute its fitness record.

    `valid` is True iff the program runs without error on every test set and
    (when `expected_outputs` is given) reproduces those outputs exactly -
    this is the correctness check for evolved/optimized programs.
    """
    instr_count, arith_ops, code_size = static_metrics(tac)

    total_cycles = 0
    outputs_per_test: List[List[int]] = []
    valid = True
    for idx, test_inputs in enumerate(test_input_sets):
        result = run(tac, test_inputs)
        if not result.ok:
            valid = False
            outputs_per_test.append([])
            continue
        total_cycles += result.cycles
        outputs_per_test.append(result.outputs)
        if expected_outputs is not None and result.outputs != list(expected_outputs[idx]):
            valid = False

    exec_time = total_cycles / len(test_input_sets) if test_input_sets else 0.0
    fitness = Fitness(exec_time=exec_time, instr_count=instr_count,
                       code_size=code_size, arith_ops=arith_ops)
    return EvalResult(fitness=fitness, valid=valid, outputs_per_test=outputs_per_test)
