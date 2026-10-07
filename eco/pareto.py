"""Pareto dominance and non-dominated front extraction over the four
objectives (exec_time, instr_count, code_size, arith_ops), all minimized."""
from __future__ import annotations

from typing import List

from .individual import Individual


def dominates(a: Individual, b: Individual) -> bool:
    """True if `a` is at least as good as `b` on every objective and
    strictly better on at least one (i.e. a dominates b)."""
    ta, tb = a.fitness_tuple(), b.fitness_tuple()
    at_least_as_good = all(x <= y for x, y in zip(ta, tb))
    strictly_better = any(x < y for x, y in zip(ta, tb))
    return at_least_as_good and strictly_better


def distinct_fitness_points(front: List[Individual]) -> int:
    """How many *distinct* objective-space points the front covers - many
    genomes can tie on the exact same fitness, so this is a more honest
    measure of trade-off variety than the raw front size."""
    return len({ind.fitness_tuple() for ind in front})


def pareto_front(individuals: List[Individual]) -> List[Individual]:
    """Non-dominated set among the (valid) individuals given."""
    front = []
    for cand in individuals:
        if any(dominates(other, cand) for other in individuals if other is not cand):
            continue
        front.append(cand)
    return front


_LABELS = {
    "exec_time": "fastest (lowest exec_time)",
    "instr_count": "fewest instructions",
    "code_size": "smallest code size",
    "arith_ops": "fewest arithmetic ops",
}


def describe_tradeoff(ind: Individual, front: List[Individual]) -> str:
    """Label what an individual on the Pareto front is best at, relative
    to its front-mates."""
    labels = []
    names = ["exec_time", "instr_count", "code_size", "arith_ops"]
    for i, name in enumerate(names):
        value = ind.fitness_tuple()[i]
        if value == min(o.fitness_tuple()[i] for o in front):
            labels.append(_LABELS[name])
    if not labels:
        return "balanced trade-off"
    return ", ".join(labels)
