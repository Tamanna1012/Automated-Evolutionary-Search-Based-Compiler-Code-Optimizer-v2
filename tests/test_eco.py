"""Sanity/regression tests for the evolutionary compiler optimizer."""
import itertools
import random

from eco.benchmark_generator import generate_program
from eco.dataset import generate_dataset
from eco.ga import GAConfig, run_ga
from eco.interpreter import run
from eco.metrics import evaluate
from eco.optimizations import OPTIMIZATION_NAMES, apply_genome
from eco.pareto import dominates, pareto_front


SAMPLE_TAC = [
    ("input", "a", "x"),
    ("input", "b", "y"),
    ("const", "t1", 0),
    ("bin", "t2", "+", "a", "t1"),
    ("bin", "t3", "+", "a", "b"),
    ("bin", "t4", "+", "a", "b"),
    ("copy", "t5", "t4"),
    ("const", "c1", 1),
    ("bin", "t6", "*", "t5", "c1"),
    ("output", "t2"),
    ("output", "t6"),
]
SAMPLE_TESTS = [{"x": 3, "y": 4}, {"x": -2, "y": 5}, {"x": 0, "y": 0}]


def test_interpreter_basic_arithmetic():
    result = run(SAMPLE_TAC, {"x": 3, "y": 4})
    assert result.ok
    assert result.outputs == [3, 7]


def test_interpreter_div_by_zero_is_safe_not_a_crash():
    tac = [("input", "a", "x"), ("input", "b", "y"), ("bin", "c", "/", "a", "b"), ("output", "c")]
    result = run(tac, {"x": 10, "y": 0})
    assert result.ok
    assert result.outputs == [0]


def test_all_short_genomes_preserve_correctness():
    base = evaluate(SAMPLE_TAC, SAMPLE_TESTS)
    for r in range(1, 4):
        for genome in itertools.permutations(OPTIMIZATION_NAMES, r):
            opt_tac = apply_genome(SAMPLE_TAC, list(genome))
            res = evaluate(opt_tac, SAMPLE_TESTS, base.outputs_per_test)
            assert res.valid, f"genome {genome} broke correctness"


def test_optimizations_never_increase_cost_when_all_applied():
    base = evaluate(SAMPLE_TAC, SAMPLE_TESTS)
    full = apply_genome(SAMPLE_TAC, list(OPTIMIZATION_NAMES))
    opt_eval = evaluate(full, SAMPLE_TESTS, base.outputs_per_test)
    assert opt_eval.valid
    assert opt_eval.fitness.instr_count <= base.fitness.instr_count
    assert opt_eval.fitness.exec_time <= base.fitness.exec_time


def test_pareto_dominance_basic():
    from eco.individual import Individual
    from eco.metrics import Fitness

    a = Individual([], [], Fitness(1, 1, 1, 1), True)
    b = Individual([], [], Fitness(2, 2, 2, 2), True)
    c = Individual([], [], Fitness(1, 1, 1, 1), True)  # tie with a
    assert dominates(a, b)
    assert not dominates(b, a)
    assert not dominates(a, c)  # equal, not strictly better
    front = pareto_front([a, b, c])
    assert b not in front
    assert a in front and c in front


def test_benchmark_generator_produces_valid_correct_programs():
    rng = random.Random(123)
    for i in range(15):
        prog = generate_program(i, rng)
        result = evaluate(prog.tac, prog.test_input_sets, prog.expected_outputs)
        assert result.valid
        assert prog.baseline_fitness.instr_count == len(prog.tac)


def test_dataset_size_bounds_enforced():
    try:
        generate_dataset(50, seed=0)
        assert False, "should have rejected n_programs < 200"
    except ValueError:
        pass
    progs = generate_dataset(200, seed=0)
    assert len(progs) == 200


def test_ga_improves_over_baseline_and_stays_correct():
    prog_rng = random.Random(7)
    prog = generate_program(0, prog_rng)
    config = GAConfig(pop_size=20, num_generations=15, seed=1)
    result = run_ga(prog.tac, prog.test_input_sets, prog.expected_outputs, config)

    assert result.best_individual.valid
    assert result.best_individual.fitness.exec_time <= prog.baseline_fitness.exec_time
    # best cost must never regress generation-to-generation (elitism)
    best_costs = [rec.best_cost for rec in result.history]
    assert all(b1 >= b2 - 1e-9 for b1, b2 in zip(best_costs, best_costs[1:]))


def test_pareto_front_contains_only_mutually_nondominated_individuals():
    prog_rng = random.Random(99)
    prog = generate_program(1, prog_rng)
    config = GAConfig(pop_size=25, num_generations=20, seed=2)
    result = run_ga(prog.tac, prog.test_input_sets, prog.expected_outputs, config)

    front = result.final_pareto_front
    for a in front:
        for b in front:
            if a is not b:
                assert not dominates(a, b)


# ---------------------------------------------------------------------------
# Trade-off optimizations: strength_reduction / multiply_fusion
# ---------------------------------------------------------------------------
from eco.optimizations import OPTIMIZATIONS, multiply_fusion, strength_reduction

SCALE_INPUTS = [{"x": v} for v in (-7, -1, 0, 1, 2, 5, 13, 1000)]


def _scale_tac(k):
    return [
        ("input", "a", "x"),
        ("const", "k", k),
        ("bin", "d", "*", "a", "k"),
        ("output", "d"),
    ]


def _same_outputs(tac_a, tac_b, inputs=SCALE_INPUTS):
    for inp in inputs:
        ra, rb = run(tac_a, inp), run(tac_b, inp)
        assert ra.ok and rb.ok
        assert ra.outputs == rb.outputs


def test_new_optimizations_are_registered():
    assert "strength_reduction" in OPTIMIZATION_NAMES
    assert "multiply_fusion" in OPTIMIZATION_NAMES
    assert len(OPTIMIZATION_NAMES) == len(set(OPTIMIZATION_NAMES)) == 8


def test_strength_reduction_preserves_output_for_each_factor():
    for k in (2, 3, 4):
        tac = _scale_tac(k)
        reduced = strength_reduction(tac)
        assert not any(i[0] == "bin" and i[2] == "*" for i in reduced)
        _same_outputs(tac, reduced)


def test_strength_reduction_also_handles_constant_on_the_left():
    tac = [("input", "a", "x"), ("const", "k", 3), ("bin", "d", "*", "k", "a"), ("output", "d")]
    _same_outputs(tac, strength_reduction(tac))


def test_strength_reduction_leaves_other_multiplies_alone():
    tac = _scale_tac(7)
    assert strength_reduction(tac) == tac
    both_vars = [("input", "a", "x"), ("input", "b", "y"), ("bin", "d", "*", "a", "b"), ("output", "d")]
    assert strength_reduction(both_vars) == both_vars


def test_strength_reduction_is_a_real_tradeoff_for_k3():
    """Faster, but more arithmetic instructions: genuinely non-dominated."""
    tac = _scale_tac(3)
    reduced = apply_genome(tac, ["strength_reduction", "dead_code_elimination"])
    base, opt = evaluate(tac, SCALE_INPUTS).fitness, evaluate(reduced, SCALE_INPUTS).fitness
    assert opt.exec_time < base.exec_time
    assert opt.arith_ops > base.arith_ops
    assert opt.code_size > base.code_size


def test_multiply_fusion_preserves_output():
    for chain in (
        [("bin", "u", "+", "a", "a"), ("bin", "d", "+", "u", "a")],
        [("bin", "u", "+", "a", "a"), ("bin", "d", "+", "a", "u")],
        [("bin", "u", "+", "a", "a"), ("bin", "d", "+", "u", "u")],
    ):
        tac = [("input", "a", "x")] + chain + [("output", "d")]
        fused = multiply_fusion(tac)
        assert any(i[0] == "bin" and i[2] == "*" for i in fused)
        _same_outputs(tac, fused)


def test_multiply_fusion_trades_time_for_size():
    tac = [("input", "a", "x"), ("bin", "u", "+", "a", "a"), ("bin", "d", "+", "u", "a"), ("output", "d")]
    fused = apply_genome(tac, ["multiply_fusion", "dead_code_elimination"])
    base, opt = evaluate(tac, SCALE_INPUTS).fitness, evaluate(fused, SCALE_INPUTS).fitness
    assert opt.exec_time > base.exec_time
    assert opt.arith_ops < base.arith_ops
    assert opt.code_size < base.code_size


def test_multiply_fusion_ignores_unrelated_adds():
    tac = [("input", "a", "x"), ("input", "b", "y"), ("bin", "u", "+", "a", "a"),
           ("bin", "d", "+", "u", "b"), ("output", "d")]
    assert multiply_fusion(tac) == tac


def test_new_passes_do_not_clash_with_existing_names():
    tac = [("input", "a", "x"), ("const", "_sr1", 3), ("bin", "d", "*", "a", "_sr1"), ("output", "d")]
    reduced = strength_reduction(tac)
    defs = [i[1] for i in reduced if i[0] != "output"]
    assert len(defs) == len(set(defs))
    _same_outputs(tac, reduced)


def test_every_pass_pair_with_new_passes_preserves_correctness_on_benchmarks():
    rng = random.Random(2024)
    for i in range(10):
        prog = generate_program(i, rng)
        for first, second in itertools.product(OPTIMIZATION_NAMES, repeat=2):
            opt = apply_genome(prog.tac, [first, second])
            assert evaluate(opt, prog.test_input_sets, prog.expected_outputs).valid, (prog.name, first, second)


def test_new_passes_compose_in_both_orders_without_changing_output():
    rng = random.Random(5)
    prog = generate_program(3, rng)
    for genome in (
        ["strength_reduction", "multiply_fusion"],
        ["multiply_fusion", "strength_reduction"],
        ["strength_reduction", "dead_code_elimination", "multiply_fusion", "dead_code_elimination"],
    ):
        opt = apply_genome(prog.tac, genome)
        assert evaluate(opt, prog.test_input_sets, prog.expected_outputs).valid


def test_benchmarks_contain_tradeoff_fodder():
    prog = generate_program(0, random.Random(1))
    assert strength_reduction(prog.tac) != prog.tac
    assert multiply_fusion(prog.tac) != prog.tac
