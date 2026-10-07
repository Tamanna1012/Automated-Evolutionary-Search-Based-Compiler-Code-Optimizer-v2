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


# ---------------------------------------------------------------------------
# Reports for all programs
# ---------------------------------------------------------------------------
import csv

from eco import main as eco_main
from eco.report import SUMMARY_COLUMNS, aggregate_metrics, write_all_program_reports


def test_report_written_for_all_300_programs_plus_summary_csv(tmp_path):
    programs = generate_dataset(300, seed=3)
    config = GAConfig(pop_size=6, num_generations=3, seed=0)  # tiny GA: this tests reporting, not search
    results = [run_ga(p.tac, p.test_input_sets, p.expected_outputs, config) for p in programs]

    out_dir = tmp_path / "all_programs"
    rows = write_all_program_reports(programs, results, str(out_dir))

    assert len(rows) == 300
    txt_files = sorted(f.name for f in out_dir.glob("bench_*_report.txt"))
    assert txt_files == sorted(f"{p.name}_report.txt" for p in programs)

    with open(out_dir / "all_programs_summary.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SUMMARY_COLUMNS
        csv_rows = list(reader)
    assert len(csv_rows) == 300
    assert [r["program"] for r in csv_rows] == [p.name for p in programs]
    for r, res in zip(csv_rows, results):
        assert r["verdict"] in ("PASS", "FAIL")
        assert int(r["pareto_front_size"]) == len(res.final_pareto_front)
        assert 1 <= int(r["distinct_points"]) <= int(r["pareto_front_size"])
        assert float(r["baseline_cost"]) == 1.0
        assert float(r["best_cost"]) <= 1.0 + 1e-9


def test_report_contents_cover_required_sections(tmp_path):
    prog = generate_program(0, random.Random(11))
    res = run_ga(prog.tac, prog.test_input_sets, prog.expected_outputs, GAConfig(pop_size=10, num_generations=6, seed=1))
    write_all_program_reports([prog], [res], str(tmp_path))
    text = (tmp_path / f"{prog.name}_report.txt").read_text(encoding="utf-8")
    for needle in ("Best/avg cost at checkpoint generations", "Pareto front:", "genome=",
                   "Best-overall (weighted) individual", "genome:", "Correctness verdict: PASS"):
        assert needle in text


def test_main_runs_50_generations_and_reports_dir_is_under_reports():
    assert eco_main.NUM_GENERATIONS == 50
    assert eco_main.ALL_REPORTS_DIR.replace("\\", "/").endswith("outputs/reports/all_programs")


# ---------------------------------------------------------------------------
# Real-kernel dataset: ast front end, ground truth from the Python source
# ---------------------------------------------------------------------------
import pytest

from eco.frontend import UnsupportedSyntax, compile_function, run_source
from eco.kernels import TEMPLATE_NAMES, TEMPLATES, generate_kernel_program


def _agree(source, name, inputs_list):
    comp = compile_function(source)
    for inputs in inputs_list:
        res = run(comp.tac, inputs)
        assert res.ok
        assert res.outputs == run_source(source, name, inputs), (source, inputs)
    return comp


def test_frontend_matches_python_for_arithmetic_and_multiple_returns():
    src = "def f(a, b, c):\n    t = a * b + c\n    u = (a - b) * (a - b) - 3\n    return t, u, a + 7\n"
    _agree(src, "f", [{"a": a, "b": b, "c": c} for a, b, c in [(1, 2, 3), (-4, 5, 0), (9, -9, 2)]])


def test_frontend_integer_division_and_unary_minus_and_augassign():
    src = ("def g(x, y):\n    q = (x * 10) // y\n    n = -x\n    m = -(x + y)\n"
           "    q += 5\n    q = q * 2\n    return q, n, m\n")
    _agree(src, "g", [{"x": x, "y": y} for x, y in [(3, 2), (20, 7), (1, 1)]])


def test_frontend_is_single_assignment_even_when_python_reassigns():
    src = "def h(x):\n    r = x\n    r = r * 2\n    r = r + r\n    return r\n"
    comp = _agree(src, "h", [{"x": v} for v in (-3, 0, 5)])
    dests = [i[1] for i in comp.tac if i[0] != "output"]
    assert len(dests) == len(set(dests))


def test_frontend_pools_literals_and_keeps_source_redundancy():
    comp = compile_function("def f(c):\n    a = c * 9 + 1\n    b = c * 9 + 2\n    return a, b\n")
    nines = [i for i in comp.tac if i[0] == "const" and i[2] == 9]
    assert len(nines) == 1                      # one const per distinct literal
    muls = [i for i in comp.tac if i[0] == "bin" and i[2] == "*"]
    assert len(muls) == 2                       # the repeated c*9 is NOT pre-eliminated
    assert len(apply_genome(comp.tac, ["cse"])) == len(comp.tac)  # same count, copy replaces mul
    assert sum(1 for i in apply_genome(comp.tac, ["cse"]) if i[0] == "bin" and i[2] == "*") == 1


@pytest.mark.parametrize("src", [
    "def f(x):\n    return x / 2\n",            # true division
    "def f(x):\n    return x ** 2\n",
    "def f(x):\n    return x % 2\n",
    "def f(x):\n    return x * 1.5\n",
    "def f(x):\n    if x:\n        return 1\n    return 2\n",
    "def f(x):\n    return y\n",                 # undefined name
    "def f(x):\n    x = 1\n",                    # no return
    "x = 1\n",                                   # no function
])
def test_frontend_rejects_unsupported_code(src):
    with pytest.raises(UnsupportedSyntax):
        compile_function(src)


def test_there_are_30_to_50_kernel_templates():
    assert 30 <= len(TEMPLATE_NAMES) <= 50


def test_every_template_compiles_and_matches_source_across_seeds():
    for name, make in TEMPLATES.items():
        for seed in range(12):
            inst = make(random.Random(seed))
            lo, hi = inst.domain
            rng = random.Random(seed + 100)
            inputs = [{a: rng.randint(lo, hi) for a in inst.arg_names} for _ in range(6)]
            _agree(inst.source, inst.kernel, inputs)


def test_kernel_program_record_is_complete_and_ground_truth_comes_from_source():
    for pid in range(len(TEMPLATE_NAMES)):
        prog = generate_kernel_program(pid, random.Random(pid))
        assert prog.kernel == TEMPLATE_NAMES[pid]
        assert prog.source_code.startswith("def ")
        assert len(prog.test_input_sets) == 5 == len(prog.expected_outputs)
        for ins, want in zip(prog.test_input_sets, prog.expected_outputs):
            assert want == run_source(prog.source_code, prog.kernel, ins)
        assert evaluate(prog.tac, prog.test_input_sets, prog.expected_outputs).valid
        assert prog.baseline_fitness.instr_count == len(prog.tac)


def test_default_dataset_is_real_kernels_and_in_spec_range():
    progs = generate_dataset(300, seed=1)
    assert 200 <= len(progs) <= 500
    assert {p.kernel for p in progs} == set(TEMPLATE_NAMES)   # every template is used
    assert len({p.source_code for p in progs}) > len(TEMPLATE_NAMES)  # instances differ (constants/sizes)
    assert len({len(p.input_names) for p in progs}) > 3        # varied argument counts


def test_synthetic_generator_remains_available_behind_a_flag():
    progs = generate_dataset(200, seed=1, kind="synthetic")
    assert all(p.kernel == "synthetic" for p in progs)
    with pytest.raises(ValueError):
        generate_dataset(200, seed=1, kind="nonsense")


def test_ga_on_a_real_kernel_stays_correct_and_improves():
    prog = generate_kernel_program(TEMPLATE_NAMES.index("naive_poly"), random.Random(4))
    res = run_ga(prog.tac, prog.test_input_sets, prog.expected_outputs, GAConfig(pop_size=20, num_generations=15, seed=2))
    assert res.best_individual.valid
    assert res.best_individual.fitness.exec_time < prog.baseline_fitness.exec_time


# ---------------------------------------------------------------------------
# Real measured runtime (validation). These only check that measurement works
# and stays correct - never exact timings, so they cannot be flaky.
# ---------------------------------------------------------------------------
from eco.report import program_report, format_program_report_text
from eco.runtime import (
    clear_cache, compile_tac, correlation_summary, measure_dataset, measure_program,
    measure_runtime_ns, outputs_match, pearson, spearman, tac_hash, tac_to_python,
)

FAST = dict(reps=5, rounds=3)


def test_runtime_compiled_function_matches_interpreter_including_division():
    tac = [("input", "a", "x"), ("input", "b", "y"), ("bin", "q", "/", "a", "b"),
           ("bin", "r", "-", "a", "b"), ("output", "q"), ("output", "r")]
    fn = compile_tac(tac, ["x", "y"])
    for x, y in [(7, 2), (-7, 2), (5, 0), (0, 3), (-9, -4)]:
        assert list(fn(x, y)) == run(tac, {"x": x, "y": y}).outputs


def test_runtime_source_has_one_statement_per_instruction():
    src = tac_to_python(SAMPLE_TAC, ["x", "y"])
    assert src.startswith("def prog(p_x, p_y):")
    assert len([ln for ln in src.splitlines() if ln.startswith("    ")]) == len(SAMPLE_TAC) - 2 + 1


def test_runtime_measurement_returns_positive_numbers_and_is_cached():
    clear_cache()
    first = measure_runtime_ns(SAMPLE_TAC, ["x", "y"], SAMPLE_TESTS, **FAST)
    assert first > 0
    assert measure_runtime_ns(SAMPLE_TAC, ["x", "y"], SAMPLE_TESTS, **FAST) == first  # cache hit, same value
    assert tac_hash(SAMPLE_TAC) == tac_hash(list(SAMPLE_TAC))


def test_every_kernel_template_runs_correctly_as_real_python():
    for pid in range(len(TEMPLATE_NAMES)):
        prog = generate_kernel_program(pid, random.Random(pid + 7))
        assert outputs_match(prog.tac, prog.input_names, prog.test_input_sets, prog.expected_outputs)
        assert measure_runtime_ns(prog.tac, prog.input_names, prog.test_input_sets, **FAST) > 0


def test_optimized_code_stays_correct_when_really_run_and_report_has_measurement():
    prog = generate_kernel_program(TEMPLATE_NAMES.index("binomial_expansion"), random.Random(3))
    res = run_ga(prog.tac, prog.test_input_sets, prog.expected_outputs, GAConfig(pop_size=15, num_generations=8, seed=1))
    m = measure_program(prog, res)
    assert m["outputs_correct"] is True
    assert m["baseline_ns"] > 0 and m["best_ns"] > 0 and m["measured_speedup_x"] > 0
    rep = program_report(prog, res, m)
    text = format_program_report_text(rep)
    assert "Measured runtime" in text and "Simulated exec-time improvement" in text
    assert "Original source:" in text


def test_report_without_runtime_still_works():
    prog = generate_kernel_program(0, random.Random(1))
    res = run_ga(prog.tac, prog.test_input_sets, prog.expected_outputs, GAConfig(pop_size=8, num_generations=3, seed=1))
    assert "Measured runtime" not in format_program_report_text(program_report(prog, res))


def test_correlation_statistics_on_known_data():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)  # monotone, not linear
    assert pearson([1, 1, 1], [1, 2, 3]) is None        # constant input has no correlation
    assert spearman([1, 2], [1, 2]) is None             # too few points


def test_dataset_measurement_and_correlation_summary_are_well_formed(tmp_path):
    progs = [generate_kernel_program(i, random.Random(i)) for i in range(0, 42, 3)]
    config = GAConfig(pop_size=10, num_generations=5, seed=0)
    results = [run_ga(p.tac, p.test_input_sets, p.expected_outputs, config) for p in progs]
    runtimes = measure_dataset(progs, results)
    assert set(runtimes) == {p.name for p in progs}
    assert all(r["baseline_ns"] > 0 and r["best_ns"] > 0 and r["outputs_correct"] for r in runtimes.values())

    corr = correlation_summary(progs, results, runtimes)
    assert corr["pooled_n"] == 2 * len(progs)
    for key in ("pooled_pearson", "pooled_spearman", "improvement_pearson", "within_program_mean_spearman"):
        assert corr[key] is None or -1.0 <= corr[key] <= 1.0

    rows = write_all_program_reports(progs, results, str(tmp_path), runtimes)
    assert all(r["measured_speedup_x"] != "" for r in rows)
    agg = aggregate_metrics(progs, results, runtimes, corr)
    assert agg["measured_outputs_correct_pct"] == 100.0
    assert agg["avg_measured_speedup_x"] > 0


# ---------------------------------------------------------------------------
# CSE that looks through copies and equal constants (value numbering)
# ---------------------------------------------------------------------------
from eco.optimizations import common_subexpression_elimination as cse_pass


def _bins(tac):
    return sum(1 for i in tac if i[0] == "bin")


def test_cse_looks_through_copies():
    tac = [("input", "a", "x"), ("input", "b", "y"), ("copy", "u", "a"),
           ("bin", "t1", "+", "a", "b"), ("bin", "t2", "+", "u", "b"),
           ("output", "t1"), ("output", "t2")]
    out = cse_pass(tac)
    assert _bins(out) == 1 and ("copy", "t2", "t1") in out
    _same_outputs_multi(tac, out, ["x", "y"])


def test_cse_treats_equal_constants_as_the_same_value():
    tac = [("input", "a", "x"), ("const", "k1", 5), ("const", "k2", 5),
           ("bin", "t1", "*", "a", "k1"), ("bin", "t2", "*", "k2", "a"),
           ("output", "t1"), ("output", "t2")]
    out = cse_pass(tac)
    assert _bins(out) == 1
    _same_outputs_multi(tac, out, ["x"])


def test_cse_collapses_a_nested_repeat_in_a_single_application():
    # (((a+b)+c)+d) computed twice, the second time from scratch
    tac = [("input", "a", "xa"), ("input", "b", "xb"), ("input", "c", "xc"), ("input", "d", "xd"),
           ("bin", "p1", "+", "a", "b"), ("bin", "p2", "+", "p1", "c"), ("bin", "p3", "+", "p2", "d"),
           ("bin", "q1", "+", "a", "b"), ("bin", "q2", "+", "q1", "c"), ("bin", "q3", "+", "q2", "d"),
           ("output", "p3"), ("output", "q3")]
    out = cse_pass(tac)
    assert _bins(out) == 3
    _same_outputs_multi(tac, out, ["xa", "xb", "xc", "xd"])


def test_cse_does_not_merge_non_commutative_swaps_or_different_operators():
    tac = [("input", "a", "x"), ("input", "b", "y"),
           ("bin", "t1", "-", "a", "b"), ("bin", "t2", "-", "b", "a"),
           ("bin", "t3", "/", "a", "b"), ("bin", "t4", "/", "b", "a"),
           ("bin", "t5", "+", "a", "b"), ("bin", "t6", "*", "a", "b"),
           ("output", "t1"), ("output", "t2"), ("output", "t3"), ("output", "t4"),
           ("output", "t5"), ("output", "t6")]
    assert _bins(cse_pass(tac)) == 6


def test_cse_is_safe_when_a_name_is_defined_twice():
    tac = [("input", "a", "x"), ("const", "k", 2), ("bin", "t", "+", "a", "k"),
           ("const", "k", 9), ("bin", "u", "+", "a", "k"), ("output", "t"), ("output", "u")]
    out = cse_pass(tac)
    _same_outputs_multi(tac, out, ["x"])


def test_cse_is_correct_on_every_kernel_and_after_other_passes():
    for pid in range(len(TEMPLATE_NAMES)):
        prog = generate_kernel_program(pid, random.Random(pid + 31))
        for genome in (["cse"], ["copy_propagation", "cse"], ["cse", "copy_propagation", "dead_code_elimination"]):
            opt = apply_genome(prog.tac, genome)
            assert evaluate(opt, prog.test_input_sets, prog.expected_outputs).valid, (prog.kernel, genome)
            assert len(opt) <= len(prog.tac)


def test_value_numbering_cse_helps_the_inline_variance_kernel_a_lot():
    # a kernel that recomputes its mean inline has many nested repeats
    for seed in range(40):
        prog = generate_kernel_program(TEMPLATE_NAMES.index("variance_of_values"), random.Random(seed))
        if "mean = (" in prog.source_code.split("\n")[1] and "var = ((x0 - (" in prog.source_code:
            break
    opt = apply_genome(prog.tac, ["cse", "copy_propagation", "dead_code_elimination"])
    assert evaluate(opt, prog.test_input_sets, prog.expected_outputs).valid
    assert len(opt) < 0.6 * len(prog.tac)


def _same_outputs_multi(tac_a, tac_b, names):
    rng = random.Random(0)
    for _ in range(8):
        inputs = {n: rng.randint(-9, 9) for n in names}
        ra, rb = run(tac_a, inputs), run(tac_b, inputs)
        assert ra.ok and rb.ok and ra.outputs == rb.outputs


# ---------------------------------------------------------------------------
# Spelling labels (tidy / naive / plain)
# ---------------------------------------------------------------------------
ALWAYS_NAIVE = {"difference_of_squares", "triangle_area_twice", "weighted_average",
                "temperature_conversions", "linear_interpolation", "cross_product_3d"}
RANDOM_SPELLING = {"distance_sq_2d", "variance_of_values", "simple_interest", "income_tax",
                   "speed_distance_time", "mean_of_values", "kinetic_and_potential_energy"}


def test_every_program_has_a_spelling_label_matching_its_kernel_class():
    progs = generate_dataset(300, seed=42)
    seen = {}
    for p in progs:
        assert p.spelling in ("plain", "tidy", "naive")
        seen.setdefault(p.kernel, set()).add(p.spelling)
    for kernel, labels in seen.items():
        if kernel in ALWAYS_NAIVE:
            assert labels == {"naive"}, kernel
        elif kernel in RANDOM_SPELLING:
            assert labels <= {"tidy", "naive"} and labels, kernel
        else:
            assert labels == {"plain"}, kernel
    assert any(len(seen[k]) == 2 for k in RANDOM_SPELLING)  # both spellings really occur


def test_spelling_reaches_reports_csv_and_aggregate(tmp_path):
    progs = [generate_kernel_program(i, random.Random(i)) for i in range(0, 42, 5)]
    results = [run_ga(p.tac, p.test_input_sets, p.expected_outputs, GAConfig(pop_size=8, num_generations=3, seed=0))
               for p in progs]
    rows = write_all_program_reports(progs, results, str(tmp_path))
    assert [r["spelling"] for r in rows] == [p.spelling for p in progs]
    first = (tmp_path / f"{progs[0].name}_report.txt").read_text(encoding="utf-8").splitlines()[0]
    assert f"{progs[0].spelling} spelling" in first
    agg = aggregate_metrics(progs, results)
    assert sum(v["programs"] for v in agg["improvement_by_spelling"].values()) == len(progs)
