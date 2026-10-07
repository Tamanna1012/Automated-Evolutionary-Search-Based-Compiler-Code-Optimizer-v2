"""Candidate program representation ("individual") for the GA population."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .metrics import Fitness
from .tac import TAC


@dataclass
class Individual:
    genome: List[str]          # ordered optimizations applied to the original TAC
    tac: TAC                   # the resulting program
    fitness: Fitness           # (exec_time, instr_count, code_size, arith_ops)
    valid: bool                # True iff outputs match the original on every test set
    cost: float = 0.0          # scalar weighted cost, filled in by the GA for ranking

    def fitness_tuple(self):
        return self.fitness.as_tuple()
