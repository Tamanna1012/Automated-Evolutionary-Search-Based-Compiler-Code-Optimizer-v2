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


# ---------------------------------------------------------------------------
# Results dashboard (self-contained HTML)
# ---------------------------------------------------------------------------
import json as _json

from eco.dashboard import PLACEHOLDER, TEMPLATE_PATH, build_dashboard_data, write_dashboard


def _small_run(n=6):
    progs = [generate_kernel_program(i * 7, random.Random(i)) for i in range(n)]
    results = [run_ga(p.tac, p.test_input_sets, p.expected_outputs, GAConfig(pop_size=8, num_generations=4, seed=1))
               for p in progs]
    runtimes = measure_dataset(progs, results)
    agg = aggregate_metrics(progs, results, runtimes, correlation_summary(progs, results, runtimes))
    return progs, results, runtimes, agg


def test_dashboard_data_has_everything_the_page_needs():
    progs, results, runtimes, agg = _small_run()
    data = build_dashboard_data(progs, results, runtimes, agg, {"dataset": "t", "population_size": 8, "generations": 4, "seed": 1})
    _json.dumps(data)  # must be JSON-serialisable
    assert len(data["programs"]) == len(progs)
    for item, prog, res in zip(data["programs"], progs, results):
        assert item["name"] == prog.name and item["kernel"] == prog.kernel
        assert len(item["hist_best"]) == len(item["hist_avg"]) == len(res.history) == 4
        assert len(item["population"]) == len(res.final_population)
        assert item["verdict"] == "PASS" and item["runtime"]["outputs_correct"]
        assert item["tac_before"] and item["tac_after"] and item["source"].startswith("def ")
        assert sum(r[4] for r in item["population"]) == len(res.final_pareto_front)
    assert data["aggregate"]["num_programs"] == len(progs)


def test_dashboard_html_is_self_contained_and_embeds_the_data(tmp_path):
    progs, results, runtimes, agg = _small_run(3)
    data = build_dashboard_data(progs, results, runtimes, agg, {"dataset": "t", "population_size": 8, "generations": 4, "seed": 1})
    # a hostile string must not be able to close the script tag
    data["programs"][0]["source"] += "\n# </script><b>x</b>"
    out = write_dashboard(data, str(tmp_path / "d" / "dashboard.html"))
    html = open(out, encoding="utf-8").read()
    assert PLACEHOLDER not in html
    assert progs[0].name in html and "Program explorer" in html
    assert "</script><b>x" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    # the only external URL is the on-demand Python engine for the live tab (no static script/link tags)
    import re as _re
    assert set(_re.findall(r"https://[^\"'` ]+", html)) == {"https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"}
    assert "<script src" not in html and "<link" not in html


def test_dashboard_template_has_the_data_placeholder():
    assert PLACEHOLDER in open(TEMPLATE_PATH, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# Modules shipped to the browser for the Code Optimization tab
# ---------------------------------------------------------------------------
from eco.dashboard import LIVE_FILES, live_sources













def test_live_modules_are_self_contained_for_the_browser():
    """Everything shipped to Pyodide may import only the standard library and each other."""
    import ast as _ast
    allowed_std = {"__future__", "json", "random", "math", "dataclasses", "typing", "ast", "time"}
    shipped = set(LIVE_FILES)
    src = live_sources()
    assert set(src) == {n + ".py" for n in LIVE_FILES}
    for name, text in src.items():
        for node in _ast.walk(_ast.parse(text)):
            if isinstance(node, _ast.ImportFrom) and node.level == 1:
                assert node.module in shipped, (name, node.module)       # relative import of a shipped module
            elif isinstance(node, _ast.Import):
                assert all(a.name.split(".")[0] in allowed_std for a in node.names), (name, node.names)
            elif isinstance(node, _ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed_std, (name, node.module)




# ---------------------------------------------------------------------------
# Code Optimization workbench (classic techniques with a log of every change)
# ---------------------------------------------------------------------------
from eco.code_optimizer import (
    SAMPLES, TECHNIQUES, Stmt, algebraic_simplification, common_subexpression_elimination,
    constant_folding, constant_propagation, copy_propagation, dead_code_elimination, from_tac,
    optimize_program, optimize_program_json, optimize_stmts, quadruple_rows, run_stmts,
)


def _texts(stmts):
    return [s.text() for s in stmts]


def test_constant_folding_evaluates_constant_expressions():
    out, ev = constant_folding([Stmt("t1", "*", 10, 5), Stmt("t2", "+", 2, 3), Stmt("t3", "-", 2, 9)])
    assert _texts(out) == ["t1 = 50", "t2 = 5", "t3 = -7"]
    assert [e["technique"] for e in ev] == ["Constant Folding"] * 3


def test_constant_folding_never_changes_behaviour_for_division():
    out, _ = constant_folding([Stmt("a", "/", 10, 5), Stmt("b", "/", 7, 2), Stmt("c", "/", 4, 0)])
    assert _texts(out) == ["a = 2", "b = 7 / 2", "c = 4 / 0"]   # inexact and divide-by-zero are left alone


def test_constant_propagation_replaces_known_constants_only():
    out, ev = constant_propagation([Stmt("a", "=", 10), Stmt("b", "+", "a", 5), Stmt("c", "+", "x", 1)])
    assert _texts(out) == ["a = 10", "b = 10 + 5", "c = x + 1"]
    assert len(ev) == 1 and "a = 10" in ev[0]["reason"]


def test_constant_propagation_forgets_a_variable_that_is_reassigned():
    out, _ = constant_propagation([Stmt("a", "=", 10), Stmt("a", "+", "q", 1), Stmt("b", "+", "a", 5)])
    assert out[2].text() == "b = a + 5"


def test_algebraic_simplification_identities():
    stmts = [Stmt("r1", "+", "a", 0), Stmt("r2", "+", 0, "a"), Stmt("r3", "-", "a", 0), Stmt("r4", "*", "a", 1),
             Stmt("r5", "*", 1, "a"), Stmt("r6", "/", "a", 1), Stmt("r7", "*", "a", 0), Stmt("r8", "*", 0, "a")]
    out, ev = algebraic_simplification(stmts)
    assert _texts(out) == ["r1 = a", "r2 = a", "r3 = a", "r4 = a", "r5 = a", "r6 = a", "r7 = 0", "r8 = 0"]
    assert len(ev) == 8


def test_algebraic_simplification_does_not_touch_non_identities():
    stmts = [Stmt("r", "-", 0, "a"), Stmt("s", "/", 1, "a"), Stmt("t", "+", "a", "b"), Stmt("u", "*", "a", 2)]
    out, ev = algebraic_simplification(stmts)
    assert _texts(out) == _texts(stmts) and ev == []


def test_cse_reuses_available_expressions_including_commutative_ones():
    out, ev = common_subexpression_elimination([Stmt("t1", "+", "a", "b"), Stmt("t2", "+", "a", "b"), Stmt("t3", "+", "b", "a")])
    assert _texts(out) == ["t1 = a + b", "t2 = t1", "t3 = t1"]
    assert len(ev) == 2


def test_cse_respects_order_for_non_commutative_ops_and_redefinitions():
    out, _ = common_subexpression_elimination([Stmt("t1", "-", "a", "b"), Stmt("t2", "-", "b", "a")])
    assert _texts(out) == ["t1 = a - b", "t2 = b - a"]
    out, _ = common_subexpression_elimination([Stmt("t1", "+", "a", "b"), Stmt("a", "=", 7), Stmt("t2", "+", "a", "b")])
    assert out[2].text() == "t2 = a + b"          # a changed in between, so the old result cannot be reused


def test_copy_propagation_and_its_invalidation():
    out, _ = copy_propagation([Stmt("a", "=", "b"), Stmt("c", "+", "a", 5)])
    assert _texts(out) == ["a = b", "c = b + 5"]
    out, _ = copy_propagation([Stmt("a", "=", "b"), Stmt("b", "=", 9), Stmt("c", "+", "a", 5)])
    assert out[2].text() == "c = a + 5"           # b changed after the copy, so a is no longer b


def test_dead_code_elimination_keeps_outputs_inputs_and_live_values():
    stmts = [Stmt("a", "input", "a"), Stmt("c", "+", "a", 1), Stmt("d", "=", 100), Stmt(None, "print", "c")]
    out, ev = dead_code_elimination(stmts)
    assert _texts(out) == ["a = input(a)", "c = a + 1", "print c"]
    assert len(ev) == 1 and ev[0]["original"] == "d = 100"
    out, _ = dead_code_elimination([Stmt("p", "input", "p")])
    assert _texts(out) == ["p = input(p)"]        # an unused parameter is still a parameter


def test_the_assignment_examples_work_end_to_end():
    r = optimize_program("def f(a):\n    x = 10\n    y = x + 5\n    z = 10 * 5\n    d = 100\n    return y + a\n")
    assert r["ok"] and r["behaviour_preserved"]
    names = {e["technique"] for e in r["log"]}
    assert {"Constant Propagation", "Constant Folding", "Dead Code Elimination"} <= names
    texts = [row["stmt"] for row in r["optimized"]]
    assert any("15" in t for t in texts)
    assert not any(t.startswith("z =") or t.startswith("d =") for t in texts)


def test_summary_numbers_are_computed_not_hardcoded():
    results = {name: optimize_program(src, lo, hi) for name, (lo, hi, src) in SAMPLES.items()}
    seen = set()
    for r in results.values():
        s = r["summary"]
        assert s["original_instructions"] == len(r["original"]) and s["optimized_instructions"] == len(r["optimized"])
        assert s["removed"] == s["original_instructions"] - s["optimized_instructions"]
        assert s["optimizations"] == len(r["log"])
        assert s["reduction_pct"] == round(100.0 * s["removed"] / s["original_instructions"], 2)
        assert set(s["techniques"]) <= set(TECHNIQUES)
        assert sum(t["count"] for t in s["by_technique"]) == s["optimizations"]
        seen.add((s["original_instructions"], s["optimized_instructions"], s["optimizations"]))
    assert len(seen) == len(results)               # different programs give different numbers


def test_no_optimization_is_reported_and_code_is_left_unchanged():
    r = optimize_program("def plain(a, b):\n    s = a + b\n    return s\n")
    assert r["ok"] and r["no_optimization"] and r["log"] == []
    assert [x["stmt"] for x in r["original"]] == [x["stmt"] for x in r["optimized"]]
    assert r["summary"]["removed"] == 0 and r["summary"]["reduction_pct"] == 0.0 and r["summary"]["techniques"] == []
    assert r["behaviour_preserved"]


def test_optimizer_preserves_behaviour_on_every_kernel_template():
    for pid in range(len(TEMPLATE_NAMES)):
        prog = generate_kernel_program(pid, random.Random(pid + 77))
        lo, hi = (1, 20) if "//" in prog.source_code else (-15, 15)
        r = optimize_program(prog.source_code, lo, hi)
        assert r["ok"], (prog.kernel, r.get("error"))
        assert r["behaviour_preserved"], prog.kernel
        assert r["summary"]["optimized_instructions"] <= r["summary"]["original_instructions"]
        assert all(c["original"] == c["optimized"] == c["python"] for c in r["checks"] if c["python"] is not None)
        assert all(c["original"] == c["optimized"] for c in r["checks"])


def test_optimizer_output_matches_the_original_on_many_random_inputs():
    prog = generate_kernel_program(TEMPLATE_NAMES.index("variance_of_values"), random.Random(5))
    comp = compile_function(prog.source_code)
    original = from_tac(comp.tac)
    optimized, _, _ = optimize_stmts(original)
    rng = random.Random(1)
    for _ in range(200):
        inputs = {a: rng.randint(1, 50) for a in comp.arg_names}
        assert run_stmts(original, inputs) == run_stmts(optimized, inputs) == run(comp.tac, inputs).outputs


def test_textbook_conversion_inlines_literals_and_keeps_user_names():
    comp = compile_function("def f(a):\n    k = 10\n    t1 = a * 5\n    return t1 + k\n")
    texts = _texts(from_tac(comp.tac))
    assert "k = 10" in texts and "t1 = a * 5" in texts   # user variables kept, literal 5 inlined
    assert not any("_k" in t or "_e" in t for t in texts)


def test_four_address_code_comes_from_the_optimized_code():
    r = optimize_program("def f(a):\n    x = 3 * 4\n    return x + a\n")
    assert r["ok"] and all(set(q) == {"sno", "op", "arg1", "arg2", "result"} for q in r["quadruples"])
    assert len(r["quadruples"]) == len(r["optimized"])
    assert any(q["op"] == "+" and q["arg1"] == "12" for q in r["quadruples"])


def test_optimizer_reports_errors_by_stage():
    assert optimize_program("def f(x:\n    return x\n")["stage"] == "Syntax analysis"
    assert optimize_program("def f(x):\n    return y\n")["stage"] == "Semantic analysis"
    assert optimize_program("def f(x):\n    return x / 2\n")["ok"] is False
    assert optimize_program("def f(x):\n    return x // 0\n")["stage"] == "Execution"
    assert optimize_program("def f(x):\n    return x // 2\n", -15, 15)["ok"] is True   # negatives checked against generated code


def test_optimizer_json_entry_point_and_dashboard_embedding(tmp_path):
    out = _json.loads(optimize_program_json(_json.dumps({"source": "def f(a):\n    return a + 0\n", "lo": 1, "hi": 9})))
    assert out["ok"] and out["summary"]["optimizations"] >= 1
    assert all(optimize_program(src, lo, hi)["ok"] for lo, hi, src in SAMPLES.values())
    progs, results, runtimes, agg = _small_run(2)
    data = build_dashboard_data(progs, results, runtimes, agg, {"dataset": "t", "population_size": 8, "generations": 4, "seed": 1})
    html = open(write_dashboard(data, str(tmp_path / "d.html")), encoding="utf-8").read()
    assert 'data-tab="optimize" class="active"' in html and "Optimization Summary" in html and "4-Address Code" in html
    assert html.index('data-tab="optimize"') < html.index('data-tab="overview"')      # primary tab comes first
    assert "def demo(a, b)" in html                                                  # the built-in example is embedded
    for section in ("Original 3-Address Code", "Optimization Analysis", "Optimized 3-Address Code", "No optimizations detected"):
        assert section in html
    assert "Behaviour check" not in html and "Front-end analysis" not in html        # only the requested sections


def test_results_metrics_use_the_shared_cost_model():
    r = optimize_program("def f(a, b):\n    x = 10 * 5\n    t1 = a + b\n    t2 = a + b\n    return t1 + t2 + x\n", 1, 9)
    by_key = {m["key"]: m for m in r["metrics"]}
    assert by_key["instr_count"]["optimized"] < by_key["instr_count"]["original"]
    assert by_key["exec_time"]["optimized"] < by_key["exec_time"]["original"]
    assert r["benefit_pct"] > 0


def test_per_technique_rows_cover_every_technique_and_sum_instruction_removals():
    r = optimize_program(SAMPLES["Mixed example (all six techniques)"][2], 1, 20)
    assert [t["technique"] for t in r["per_technique"]] == TECHNIQUES
    removed = sum(t["instructions_removed"] for t in r["per_technique"])
    assert removed == r["summary"]["removed"] or removed == sum(e["instrs_removed"] for e in r["log"])


def test_no_change_gives_zero_benefit():
    r = optimize_program("def plain(a, b):\n    s = a + b\n    return s\n")
    assert r["benefit_pct"] == 0
    assert all(m["change_pct"] == 0 for m in r["metrics"])


def test_pass_trace_sums_equal_the_total_reduction():
    r = optimize_program(SAMPLES["Mixed example (all six techniques)"][2], 1, 20)
    removed_by_trace = sum(t["statements_before"] - t["statements_after"] for t in r["pass_trace"])
    assert removed_by_trace == r["summary"]["removed"]
    assert {t["technique"] for t in r["pass_trace"]} == set(TECHNIQUES)


def test_selection_views_cover_all_subsets_and_union_is_additive():
    r = optimize_program(SAMPLES["Mixed example (all six techniques)"][2], 1, 20)
    views = r["selection_views"]
    assert len(views) == 2 ** len(TECHNIQUES) - 1
    single = {t: views[str(1 << i)]["bars"][0] for i, t in enumerate(TECHNIQUES)}
    pair = views[str((1 << 0) | (1 << 5))]
    assert pair["combined"]["removed"] == single["Constant Propagation"]["removed"] + single["Dead Code Elimination"]["removed"]
    assert pair["combined"]["changes"] == single["Constant Propagation"]["changes"] + single["Dead Code Elimination"]["changes"]
    assert len(pair["sno"]) == pair["combined"]["changes"]


from eco.equivalence import prove_equivalent
from eco.frontend import compile_function


def _stmts(src):
    return from_tac(compile_function(src).tac)


def test_prover_proves_known_equal_programs():
    a = _stmts("def f(a, b):\n    x = a + b\n    return x * 2\n")
    b = _stmts("def g(a, b):\n    return a * 2 + b * 2\n")
    assert prove_equivalent(a, b)["proved"] is True


def test_prover_rejects_known_different_programs():
    a = _stmts("def f(a, b):\n    return a + b\n")
    b = _stmts("def g(a, b):\n    return a - b\n")
    assert prove_equivalent(a, b)["proved"] is False


def test_prover_catches_a_deliberately_wrong_rewrite():
    original = _stmts("def f(a, b):\n    t1 = a * b\n    t2 = a * b\n    return t1 + t2\n")
    good, _, _ = optimize_stmts(original)
    assert prove_equivalent(original, good)["proved"] is True
    wrong = [s.copy_with(op="-") if s.op == "+" else s for s in good]
    result = prove_equivalent(original, wrong)
    assert result["proved"] is False
    assert any(not row["match"] for row in result["outputs"])


def test_prover_treats_division_as_opaque_but_folds_exact_constants():
    assert prove_equivalent(_stmts("def f(a, b):\n    return a // b\n"),
                            _stmts("def g(a, b):\n    return a // b\n"))["proved"] is True
    assert prove_equivalent(_stmts("def f(a, b):\n    return a // b\n"),
                            _stmts("def g(a, b):\n    return b // a\n"))["proved"] is False
    assert prove_equivalent(_stmts("def f(a):\n    return 10 // 5 + a\n"),
                            _stmts("def g(a):\n    return 2 + a\n"))["proved"] is True
    assert prove_equivalent(_stmts("def f(a):\n    return a // 1\n"),
                            _stmts("def g(a):\n    return a\n"))["proved"] is True


def test_optimized_mixed_sample_is_proved_equivalent():
    r = optimize_program(SAMPLES["Mixed example (all six techniques)"][2], 1, 20)
    assert r["equivalence"]["proved"] is True
    assert r["verdict"]["level"] == "proved"


def test_every_log_event_has_a_rule_and_a_checked_precondition():
    r = optimize_program(SAMPLES["Mixed example (all six techniques)"][2], 1, 20)
    assert r["log"]
    assert all(e["rule"] and e["precondition"] for e in r["log"])


def test_compile_time_check_includes_boundary_inputs_and_timing():
    r = optimize_program("def f(a, b):\n    return a * b + 0\n")
    kinds = {c["kind"] for c in r["checks"]}
    assert {"random", "boundary"} <= kinds
    assert r["compile_check"]["total"] == len(r["checks"])
    assert r["compile_check"]["ms"] >= 0
    assert r["verdict"]["level"] == "proved"


def test_division_programs_skip_negative_boundaries_and_stay_ok():
    r = optimize_program("def f(a, b):\n    return a // b\n")
    assert r["ok"] is True
    negatives = [c for c in r["checks"] if min(c["inputs"].values()) < 0]
    assert negatives and all(c["python"] is None and c["match"] for c in negatives)
    assert all(c["python"] is not None for c in r["checks"] if min(c["inputs"].values()) >= 0)


def test_verdict_reports_failure_with_first_mismatch():
    from eco.code_optimizer import _verdict
    checks = [{"inputs": {"a": 1}, "original": [2], "optimized": [2], "match": True},
              {"inputs": {"a": 3}, "original": [6], "optimized": [5], "match": False}]
    v = _verdict(checks, proved=True)
    assert v["level"] == "fail" and "{'a': 3}" in v["text"]
    assert _verdict(checks[:1], proved=False)["level"] == "tested"


def test_manual_empty_pipeline_changes_nothing_and_is_marked_manual():
    r = optimize_program(SAMPLES["Mixed example (all six techniques)"][2], 1, 20, passes=[])
    assert r["mode"] == "manual" and r["no_optimization"] is True
    assert r["comparison"]["custom_optimized_instructions"] == r["summary"]["original_instructions"]


def test_manual_order_matters_and_every_order_stays_equivalent():
    src = "def f(a, b):\n    k = 10\n    m = k + 5\n    c = m + 0\n    d = c * 1\n    return d + a\n"
    forward = optimize_program(src, passes=["Constant Propagation", "Constant Folding", "Algebraic Simplification",
                                            "Copy Propagation", "Dead Code Elimination"])
    reverse = optimize_program(src, passes=["Dead Code Elimination", "Copy Propagation", "Algebraic Simplification",
                                            "Constant Folding", "Constant Propagation"])
    assert forward["ok"] and reverse["ok"]
    assert forward["equivalence"]["proved"] and reverse["equivalence"]["proved"]
    assert forward["summary"]["optimized_instructions"] <= reverse["summary"]["optimized_instructions"]


def test_manual_comparison_reports_the_automatic_pipeline():
    src = SAMPLES["Mixed example (all six techniques)"][2]
    r = optimize_program(src, 1, 20, passes=["Dead Code Elimination"])
    auto = optimize_program(src, 1, 20)
    assert r["comparison"]["automatic_optimized_instructions"] == auto["summary"]["optimized_instructions"]
    assert r["comparison"]["custom_optimized_instructions"] >= auto["summary"]["optimized_instructions"]


def test_manual_rejects_unknown_optimization_names():
    assert optimize_program("def f(a):\n    return a + 0\n", passes=["Loop Unrolling"])["ok"] is False


# ---------------------------------------------------------------------------
# Control flow: front end, CFG, data-flow analysis and global optimizations
# ---------------------------------------------------------------------------
import random as _random

from eco.cfg import build_cfg, dominators, execute, flow_report, global_optimize, liveness, natural_loops, reaching_definitions, available_expressions
from eco.frontend import UnsupportedSyntax as _Unsupported, run_source as _run_source
from eco.interpreter import run as _run_tac

IF_SRC = "def f(n, k):\n    x = 2 * 3\n    if n < k:\n        y = x + n\n    else:\n        y = x + n\n    z = x + n\n    unused = 99\n    return y + z\n"
WHILE_SRC = "def g(n):\n    s = 0\n    while n > 0:\n        s = s + n\n        n = n - 1\n    return s\n"
NESTED_SRC = ("def h(a, b):\n    acc = 0\n    i = 0\n    while i < a:\n        if i < b:\n            acc = acc + i * 2\n"
              "        else:\n            acc = acc - 1\n        i = i + 1\n    return acc, i\n")


def _grid(names, lo=0, hi=7):
    return [dict(zip(names, vals)) for vals in __import__("itertools").product(range(lo, hi), repeat=len(names))]


@pytest.mark.parametrize("src,names", [(IF_SRC, ["n", "k"]), (WHILE_SRC, ["n"]), (NESTED_SRC, ["a", "b"])])
def test_control_flow_compiles_and_matches_python_on_every_input(src, names):
    compiled = compile_function(src)
    stmts = from_tac(compiled.tac)
    for inputs in _grid(names):
        assert execute(stmts, inputs) == _run_source(src, compiled.name, inputs)
        assert _run_tac(compiled.tac, inputs).outputs == _run_source(src, compiled.name, inputs)


@pytest.mark.parametrize("src", [
    "def f(a, b):\n    if a < b and b < 3:\n        return a\n    return b\n",
    "def f(a):\n    while a > 0:\n        return a\n    return a\n",
    "def f(a):\n    if a < 2:\n        return a\n    return 0\n",
    "def f(a):\n    while a < 2:\n        a = a + 1\n    else:\n        a = 0\n    return a\n",
])
def test_unsupported_control_flow_is_rejected_with_a_reason(src):
    with pytest.raises(_Unsupported):
        compile_function(src)


def test_cfg_shapes_for_straight_if_and_while():
    straight = from_tac(compile_function("def f(a):\n    b = a + 1\n    return b\n").tac)
    assert len(build_cfg(straight)) == 1
    if_blocks = build_cfg(from_tac(compile_function(IF_SRC).tac))
    branching = [b for b in if_blocks if len(b.succ) == 2]
    assert len(branching) == 1 and {k for _, k in branching[0].succ} == {"true", "false"}
    while_blocks = build_cfg(from_tac(compile_function(WHILE_SRC).tac))
    assert any(t <= b.id for b in while_blocks for t, _ in b.succ), "a loop needs a back edge"


def test_reaching_definitions_merge_both_arms_of_the_if():
    blocks = build_cfg(from_tac(compile_function(IF_SRC).tac))
    rin, _ = reaching_definitions(blocks)
    merge = blocks[3].id
    assert {v for v, _ in rin[merge]} >= {"y", "x", "n"}
    assert sum(1 for v, _ in rin[merge] if v == "y") == 2   # one y from each arm


def test_liveness_and_available_expressions_on_the_if_example():
    blocks = build_cfg(from_tac(compile_function(IF_SRC).tac))
    lin, lout = liveness(blocks)
    assert {"n", "x"} <= lout[0]
    ain, aout = available_expressions(blocks)
    assert ain[blocks[3].id] and any(v == "y" for _, v in ain[blocks[3].id])


def test_global_cse_reuses_a_value_computed_on_every_path():
    r = optimize_program(IF_SRC, 1, 9)
    cse = [e for e in r["log"] if e["technique"] == "Global Common Subexpression Elimination"]
    assert cse and cse[0]["optimized"] == "z = y"
    assert r["flow"]["control_flow"] is True and r["verdict"]["level"] == "proved"


def test_global_constant_propagation_crosses_the_branch_merge():
    r = optimize_program(IF_SRC, 1, 9)
    assert any(e["technique"] == "Global Constant Propagation" and e["optimized"] == "y = 6 + n" for e in r["log"])
    assert r["summary"]["removed"] >= 2


def test_unreachable_arm_of_a_constant_branch_is_removed():
    src = "def f(a):\n    if 1 < 2:\n        y = a + 1\n    else:\n        y = a + 2\n    return y\n"
    r = optimize_program(src, 1, 9)
    assert r["ok"] is True and r["behaviour_preserved"] is True
    techniques = {e["technique"] for e in r["log"]}
    assert "Unreachable Block Removal" in techniques and "Constant Folding" in techniques
    assert not any(row["op"] == "branch" for row in r["optimized"])


@pytest.mark.parametrize("src,names", [(IF_SRC, ["n", "k"]), (WHILE_SRC, ["n"]), (NESTED_SRC, ["a", "b"])])
def test_global_optimizations_preserve_behaviour_on_random_inputs(src, names):
    compiled = compile_function(src)
    original = from_tac(compiled.tac)
    optimized, records, trace = global_optimize(original)
    rng = _random.Random(3)
    for _ in range(40):
        inputs = {n: rng.randint(0, 9) for n in names}
        assert execute(optimized, inputs) == execute(original, inputs) == _run_source(src, compiled.name, inputs)
    assert all(t["statements_after"] <= t["statements_before"] for t in trace)


def test_control_flow_programs_pass_the_compile_time_checks():
    for src in (IF_SRC, WHILE_SRC, NESTED_SRC):
        r = optimize_program(src, 0, 6)
        assert r["ok"] is True and r["behaviour_preserved"] is True
        assert r["equivalence"]["control_flow"] is True
    assert optimize_program(IF_SRC, 0, 6)["equivalence"]["proved"] is True
    assert optimize_program(WHILE_SRC, 0, 6)["equivalence"]["proved"] is True
    assert optimize_program(WHILE_SRC, 0, 6)["equivalence"]["bound"] == 8
    assert optimize_program(NESTED_SRC, 0, 6)["equivalence"]["proved"] is True


def test_infinite_loop_is_reported_not_run_forever():
    tac = compile_function("def f(a):\n    while a < 100:\n        b = a\n    return a\n").tac
    result = _run_tac(tac, {"a": 5})
    assert result.ok is False and "step limit" in result.error


def test_dominators_and_natural_loops_of_a_while_loop():
    blocks = build_cfg(from_tac(compile_function(WHILE_SRC).tac))
    dom = dominators(blocks)
    assert all(0 in ds for ds in dom.values())            # the entry dominates every reachable block
    loops = natural_loops(blocks)
    assert len(loops) == 1
    assert loops[0]["header"] in loops[0]["body"] and loops[0]["back_edge"] in loops[0]["body"]
    assert len(loops[0]["body"]) >= 2


def test_if_has_no_loop_and_arms_are_dominated_by_the_test_block():
    blocks = build_cfg(from_tac(compile_function(IF_SRC).tac))
    assert natural_loops(blocks) == []
    dom = dominators(blocks)
    assert all(0 in dom[b] for b in dom)
    merge = blocks[3].id
    assert dom[merge] == [0, merge]                        # neither arm dominates the merge point


def test_flow_report_is_empty_for_straight_line_code():
    assert optimize_program("def f(a):\n    return a + 1\n")["flow"] == {"control_flow": False}
    assert flow_report(from_tac(compile_function("def f(a):\n    return a + 1\n").tac))["control_flow"] is False


def test_local_optimizer_misses_what_the_global_passes_find():
    from eco.code_optimizer import FLOW_EXAMPLE, local_optimize
    r = optimize_program(FLOW_EXAMPLE, 1, 9)
    original = from_tac(compile_function(FLOW_EXAMPLE).tac)
    local_text = [st.text() for st in local_optimize(original)]
    assert "y = x + n" in local_text          # x is defined in another block, so local passes cannot see its value
    assert r["flow"]["local_count"] > r["summary"]["optimized_instructions"]


def test_prover_proves_equal_branches_and_catches_swapped_arms():
    a = _stmts("def f(a, b):\n    if a < b:\n        y = a + 1\n    else:\n        y = b + 1\n    return y\n")
    same = _stmts("def g(a, b):\n    if a < b:\n        y = 1 + a\n    else:\n        y = 1 + b\n    return y\n")
    swapped = _stmts("def h(a, b):\n    if a < b:\n        y = b + 1\n    else:\n        y = a + 1\n    return y\n")
    assert prove_equivalent(a, same)["proved"] is True
    assert prove_equivalent(a, swapped)["proved"] is False


def test_loops_are_proved_up_to_the_bound_and_wrong_loop_rewrites_are_caught():
    from eco.equivalence import LOOP_BOUND
    loop = _stmts("def f(n):\n    s = 0\n    while n > 0:\n        s = s + n\n        n = n - 1\n    return s\n")
    wrong = _stmts("def f(n):\n    s = 0\n    while n > 0:\n        s = s - n\n        n = n - 1\n    return s\n")
    assert prove_equivalent(loop, loop)["proved"] is True and prove_equivalent(loop, loop)["bound"] == LOOP_BOUND
    assert prove_equivalent(loop, wrong)["proved"] is False


BOOL_SRC = ("def f(a, b, c):\n    if a < b and not c == 0 or b > 6:\n        r = a + b\n    else:\n"
            "        r = a - b\n    if c:\n        r = r + 1\n    return r\n")
CALL_SRC = ("def sq(x):\n    t = x * x\n    return t + 1\n\n"
            "def g(a, b):\n    u = sq(a) + sq(b)\n    if sq(a) < u:\n        v = sq(u)\n    else:\n        v = 0\n    return v, u\n")


@pytest.mark.parametrize("src,names", [(BOOL_SRC, ["a", "b", "c"]), (CALL_SRC, ["a", "b"])])
def test_boolean_conditions_and_helper_calls_match_python(src, names):
    compiled = compile_function(src)
    stmts = from_tac(compiled.tac)
    for inputs in _grid(names, 0, 5):
        assert execute(stmts, inputs) == _run_source(src, compiled.name, inputs)
    r = optimize_program(src, 0, 5)
    assert r["ok"] is True and r["behaviour_preserved"] is True


def test_recursive_helper_is_rejected_with_a_reason():
    with pytest.raises(_Unsupported, match="recursive"):
        compile_function("def f(x):\n    return f(x)\n\ndef g(a):\n    return f(a)\n")


ARRAY_SRC = ("def f(a, b):\n    xs = [a, b, a + b]\n    xs[1] = xs[0] * 2\n    s = 0\n    if a < b:\n"
             "        s = xs[2]\n    else:\n        s = xs[1]\n    return s, xs[0]\n")


def test_constant_index_arrays_match_python():
    compiled = compile_function(ARRAY_SRC)
    stmts = from_tac(compiled.tac)
    for inputs in _grid(["a", "b"], 0, 5):
        assert execute(stmts, inputs) == _run_source(ARRAY_SRC, compiled.name, inputs)
    assert optimize_program(ARRAY_SRC, 0, 5)["behaviour_preserved"] is True


@pytest.mark.parametrize("src", [
    "def f(a, i):\n    xs = [1, 2]\n    return xs[i]\n",
    "def f(a):\n    xs = [1, 2]\n    return xs[5]\n",
    "def f(a):\n    xs = [1, 2]\n    return xs\n",
    "def f(xs):\n    return xs[0]\n",
])
def test_unsupported_array_uses_are_rejected(src):
    with pytest.raises(_Unsupported):
        compile_function(src)


def test_division_program_with_negative_checks_carries_a_warning():
    r = optimize_program("def f(x):\n    return x // 2\n", -15, 15)
    assert r["ok"] is True and r["division_warning"] and "floors" in r["division_warning"]
    assert optimize_program("def f(x):\n    return x + 2\n", -15, 15)["division_warning"] is None


# ---------------------------------------------------------------------------
# Control-flow graph data, comparison with other methods, details of changed lines
# ---------------------------------------------------------------------------
import json as _json
import subprocess
import sys as _sys

from eco import comparison as _cmp
from eco.cfg import cfg_graph
from eco.code_optimizer import DEFAULT_SOURCE, FLOW_EXAMPLE
from eco.interpreter import MAX_STEPS

STRAIGHT_SRC = SAMPLES["Mixed example (all six techniques)"][2]
DEAD_BRANCH_SRC = "def f(a):\n    if 1 < 2:\n        y = a + 1\n    else:\n        y = a + 2\n    return y\n"
ARGS = {STRAIGHT_SRC: ["a", "b"], IF_SRC: ["n", "k"], WHILE_SRC: ["n"], NESTED_SRC: ["a", "b"], ARRAY_SRC: ["a", "b"], DEAD_BRANCH_SRC: ["a"]}
EVERY_PROGRAM = [v[2] for v in SAMPLES.values()] + [FLOW_EXAMPLE, IF_SRC, WHILE_SRC, NESTED_SRC, ARRAY_SRC, DEAD_BRANCH_SRC]
METRIC_KEYS = [k for k, _ in _cmp.METRICS]


def _back_edges(graph):
    return {(b["id"], e["to"]) for b in graph["blocks"] for e in b["succ"] if e["back"]}


def test_cfg_data_for_straight_line_code_is_one_drawn_block():
    r = optimize_program(STRAIGHT_SRC)
    for g, rows in ((r["cfg"]["original"], r["original"]), (r["cfg"]["optimized"], r["optimized"])):
        assert g["control_flow"] is False and len(g["blocks"]) == 1
        (b,) = g["blocks"]
        assert b["entry"] and b["exit"] and b["reachable"] and b["succ"] == []
        assert [st["text"] for st in b["stmts"]] == [row["stmt"] for row in rows]
    assert r["cfg"]["removed_blocks"] == 0


def test_cfg_data_for_if_else_has_a_branch_and_a_merge():
    g = optimize_program(IF_SRC)["cfg"]["original"]
    assert g["control_flow"] is True and len(g["blocks"]) == 4
    branch = [b for b in g["blocks"] if len(b["succ"]) == 2]
    assert len(branch) == 1 and branch[0]["entry"]
    assert {e["kind"] for e in branch[0]["succ"]} == {"true", "false"}
    merge = [b for b in g["blocks"] if sum(e["to"] == b["id"] for x in g["blocks"] for e in x["succ"]) == 2]
    assert len(merge) == 1 and merge[0]["exit"] and merge[0]["succ"] == []
    assert not _back_edges(g)
    assert sum(b["entry"] for b in g["blocks"]) == 1 and sum(b["exit"] for b in g["blocks"]) == 1


@pytest.mark.parametrize("src", [WHILE_SRC, NESTED_SRC])
def test_cfg_data_for_while_loops_marks_the_back_edge(src):
    r = optimize_program(src)
    for g, stmts in ((r["cfg"]["original"], from_tac(compile_function(src).tac)), ):
        blocks = build_cfg(stmts)
        from_loops = {(l["back_edge"], l["header"]) for l in natural_loops(blocks)}
        assert from_loops and _back_edges(g) == from_loops          # same loops as the dominator analysis finds
        dom = dominators(blocks)
        assert all(h in dom[t] for t, h in _back_edges(g))          # a back edge goes to a dominator
    assert _back_edges(r["cfg"]["optimized"])                       # the loop survives optimization


def test_cfg_marks_changed_and_removed_statements_and_blocks():
    r = optimize_program(DEAD_BRANCH_SRC)
    before, after = r["cfg"]["original"], r["cfg"]["optimized"]
    removed = [b for b in before["blocks"] if b["status"] == "removed"]
    assert len(removed) == 1 == r["cfg"]["removed_blocks"]
    assert removed[0]["label"] not in {b["label"] for b in after["blocks"]}
    assert all(st["status"] == "removed" for st in removed[0]["stmts"])
    rows = {x["line"]: x for x in r["line_changes"]}
    kept = {st["origin"] for b in after["blocks"] for st in b["stmts"]}
    for b in before["blocks"]:
        for st in b["stmts"]:
            assert (st["status"] == "removed") == (rows.get(st["origin"], {}).get("removed", False))
            assert (st["origin"] in kept) == (st["status"] != "removed")
    for b in after["blocks"]:
        for st in b["stmts"]:
            assert (st["status"] == "changed") == (st["origin"] in rows)


@pytest.mark.parametrize("src", EVERY_PROGRAM)
def test_cfg_is_recomputed_for_every_program(src):
    r = optimize_program(src)
    assert r["ok"]
    for g in (r["cfg"]["original"], r["cfg"]["optimized"]):
        ids = {b["id"] for b in g["blocks"]}
        assert g["blocks"][0]["entry"] and all(e["to"] in ids for b in g["blocks"] for e in b["succ"])


@pytest.mark.parametrize("src", EVERY_PROGRAM)
def test_every_compared_method_is_equivalent_to_the_original(src):
    r = optimize_program(src)
    mc = r["method_comparison"]
    assert [m["verified"] for m in mc["methods"]] == [True] * len(mc["methods"])
    assert sum(m["is_ours"] for m in mc["methods"]) == 1 and mc["methods"][0]["group"] == "baseline"
    # independent of the report: rebuild every method's program and run it against the original
    compiled = compile_function(src)
    original = from_tac(compiled.tac)
    names = ARGS.get(src) or compiled.arg_names
    grid = _grid(names, 0, 6)
    runnable = [t for t in grid if _run_tac(compiled.tac, t).ok]
    for cand in _cmp.build_candidates(compiled.tac, original, runnable):
        for t in runnable:
            want = execute(original, t)
            got = _run_tac(cand["tac"], t).outputs if "tac" in cand else execute(cand["stmts"], t)
            assert got == want, (cand["label"], t)


def test_every_kernel_template_compares_cleanly_and_ours_never_loses_to_no_optimization():
    for name, make in TEMPLATES.items():
        inst = make(random.Random(1))
        r = optimize_program(inst.source, *inst.domain)
        mc = r["method_comparison"]
        assert all(m["verified"] for m in mc["methods"]), name
        base = next(m for m in mc["methods"] if m["group"] == "baseline")
        ours = next(m for m in mc["methods"] if m["is_ours"])
        for k in METRIC_KEYS:
            assert ours[k] <= base[k] + 1e-9, (name, k, ours[k], base[k])


@pytest.mark.parametrize("src", EVERY_PROGRAM)
def test_ours_is_never_worse_than_no_optimization(src):
    r = optimize_program(src)
    mc = r["method_comparison"]
    base = next(m for m in mc["methods"] if m["group"] == "baseline")
    ours = next(m for m in mc["methods"] if m["is_ours"])
    for k in METRIC_KEYS:
        assert ours[k] <= base[k] + 1e-9, (k, ours[k], base[k])
    # the comparison's baseline and ours are the numbers in the Results table, not a second measurement
    results = {m["key"]: m for m in r["metrics"]}
    for k in METRIC_KEYS:
        assert base[k] == pytest.approx(results[k]["original"], abs=1e-3)
        assert ours[k] == pytest.approx(results[k]["optimized"], abs=1e-3)


@pytest.mark.parametrize("src", [STRAIGHT_SRC, IF_SRC, WHILE_SRC])
def test_comparison_is_deterministic(src):
    first = _json.dumps(optimize_program(src)["method_comparison"], sort_keys=True)
    random.seed(1)
    second = _json.dumps(optimize_program(src)["method_comparison"], sort_keys=True)
    random.seed(99)
    third = _json.dumps(optimize_program(src)["method_comparison"], sort_keys=True)
    assert first == second == third


def test_comparison_covers_every_method_and_reports_live_numbers():
    mc = optimize_program(STRAIGHT_SRC)["method_comparison"]
    groups = [m["group"] for m in mc["methods"]]
    assert groups == ["baseline"] + ["single"] * 6 + ["sweep", "local", "ga", "ours"]
    assert [m["label"] for m in mc["methods"] if m["group"] == "single"] == [f"{t} only" for t in TECHNIQUES]
    # a different program gives different numbers: nothing is stored
    other = optimize_program(SAMPLES["Common subexpressions"][2])["method_comparison"]
    assert [m["instr_count"] for m in other["methods"]] != [m["instr_count"] for m in mc["methods"]]
    ga = next(m for m in mc["methods"] if m["group"] == "ga")
    assert all(g in OPTIMIZATION_NAMES for g in ga["genome"])
    control = optimize_program(IF_SRC)["method_comparison"]
    assert control["scope"] == "global" and sum(m["group"] == "single" for m in control["methods"]) == 5


def test_a_method_that_changes_the_output_is_flagged_and_can_never_be_best():
    original = _stmts("def f(a, b):\n    x = a + b\n    y = x * 2\n    return y\n")
    inputs = [{"a": 1, "b": 2}, {"a": 5, "b": 3}]
    want = [execute(original, t) for t in inputs]
    broken = [s.copy_with(op="+") if s.op == "*" else s for s in original]          # * became +
    bad = _cmp._verify_stmts(original, broken, inputs, want)
    assert bad["status"] == "failed" and "original printed" in bad["detail"]
    assert _cmp._verify_stmts(original, original, inputs, want)["status"] in ("proved", "tested")
    numbers = lambda v: dict(instr_count=v, exec_time=float(v), arith_ops=v, code_size=v)
    methods = [dict(key="a", label="a", is_ours=False, group="baseline", verified=True, **numbers(9)),
               dict(key="b", label="b", is_ours=False, group="ga", verified=False, **numbers(1)),   # lowest numbers, but it breaks the program
               dict(key="c", label="c", is_ours=True, group="ours", verified=True, **numbers(5))]
    _cmp.mark_best(methods)
    assert methods[1]["best"] == dict.fromkeys(METRIC_KEYS, False)
    assert methods[2]["best"] == dict.fromkeys(METRIC_KEYS, True)
    for w in _cmp._wins(methods):                        # ours is compared with "a", never with the failed "b"
        assert w["verdict"] == "best" and w["compared_with"] == ["a"] and w["other"] == 9 and w["diff"] == 4


@pytest.mark.parametrize("src", EVERY_PROGRAM)
def test_wins_summary_is_computed_from_the_table(src):
    mc = optimize_program(src)["method_comparison"]
    ours = next(m for m in mc["methods"] if m["is_ours"])
    base = next(m for m in mc["methods"] if m["group"] == "baseline")
    others = [m for m in mc["methods"] if not m["is_ours"] and m["verified"]]
    assert [w["metric"] for w in mc["wins"]] == METRIC_KEYS
    for w in mc["wins"]:
        k = w["metric"]
        best = min(m[k] for m in others)
        assert w["ours"] == ours[k] and w["other"] == best
        assert w["diff"] == pytest.approx(best - ours[k], abs=1e-3)
        assert w["pct"] == (pytest.approx(100 * (best - ours[k]) / best, abs=0.01) if best else None)
        assert w["verdict"] == ("tied" if abs(best - ours[k]) < 1e-9 else ("best" if ours[k] < best else "behind"))
        assert set(w["compared_with"]) == {m["label"] for m in others if abs(m[k] - best) < 1e-9}
        assert w["vs_baseline"] == pytest.approx(base[k] - ours[k], abs=1e-3)


def test_wins_summary_reports_a_loss_plainly():
    wins = {w["metric"]: w for w in optimize_program(FLOW_EXAMPLE)["method_comparison"]["wins"]}
    assert wins["instr_count"]["verdict"] == "behind" and wins["instr_count"]["diff"] < 0 and wins["instr_count"]["pct"] < 0
    assert wins["arith_ops"]["verdict"] == "tied" and wins["arith_ops"]["diff"] == 0


def test_comparison_handles_code_with_nothing_to_optimize():
    r = optimize_program(SAMPLES["Already optimal (nothing to change)"][2])
    assert r["no_optimization"] and r["summary"]["lines_changed"] == 0 and r["line_changes"] == []
    assert all(w["verdict"] == "tied" for w in r["method_comparison"]["wins"])
    assert len({m["instr_count"] for m in r["method_comparison"]["methods"]}) == 1
    assert [b["status"] for b in r["cfg"]["original"]["blocks"]] == ["same"]


@pytest.mark.parametrize("src", EVERY_PROGRAM)
def test_lines_changed_count_matches_the_summary_and_the_panel_rows(src):
    r = optimize_program(src)
    rows = r["line_changes"]
    assert r["summary"]["lines_changed"] == len(rows)                       # the count the page shows is the panel's row count
    assert len(rows) == len({e["origin"] for e in r["log"]})                # one row per original line the log touched
    assert [x["line"] for x in rows] == sorted({x["line"] for x in rows})   # no line listed twice
    assert len(rows) <= r["summary"]["optimizations"]                       # a line can be rewritten many times, never the reverse
    listing = {row["sno"]: row["stmt"] for row in r["original"]}
    for x in rows:
        assert x["original"] == listing[x["line"]]
        assert x["steps"][0]["before"] == x["original"]                      # the chain starts at the original instruction
        assert all(a["after"] == b["before"] for a, b in zip(x["steps"], x["steps"][1:]))   # and every step continues the last
        assert x["removed"] == (x["steps"][-1]["after"] == "(removed)") == (x["final"] is None)
        assert x["final"] is None or x["final"] == x["steps"][-1]["after"]
        assert len(x["sentences"]) == len(x["steps"]) and x["techniques"] == list(dict.fromkeys(st["technique"] for st in x["steps"]))
        assert set(x["techniques"]) <= set(r["technique_order"])
    assert sorted(sno for x in rows for sno in (st["sno"] for st in x["steps"])) == [e["sno"] for e in r["log"]]   # every logged change is in exactly one chain


def test_lines_changed_in_manual_mode_also_matches():
    r = optimize_program(STRAIGHT_SRC, passes=["Constant Folding", "Dead Code Elimination"])
    assert r["summary"]["lines_changed"] == len(r["line_changes"]) == len({e["origin"] for e in r["log"]})


def test_a_line_touched_by_several_techniques_shows_its_whole_chain():
    r = optimize_program("def f(a):\n    k = 10\n    m = k + 5\n    n = m + 0\n    return n + a\n")
    row = next(x for x in r["line_changes"] if x["original"] == "m = k + 5")
    assert row["techniques"] == ["Constant Propagation", "Constant Folding"] or "Constant Folding" in row["techniques"]
    assert [st["after"] for st in row["steps"]][:2] == ["m = 10 + 5", "m = 15"]
    assert "because both operands are constants" in row["reason"]
    assert "because it reads a variable that always holds a known constant (k = 10)" in row["reason"]


def test_common_subexpression_rewrites_name_the_line_they_were_merged_into():
    ops = lambda text: [c for c in "+-*/" if f" {c} " in text]
    r = optimize_program(SAMPLES["Common subexpressions"][2])
    listing = {row["sno"]: row["stmt"] for row in r["original"]}
    merged = [x for x in r["line_changes"] if x["merged_into"]]
    assert merged and all(x["removed"] for x in merged)                      # copy propagation + dead code removal finish the job
    for x in merged:
        assert all(k < x["line"] and ops(listing[k]) == ops(listing[x["line"]]) == ["*"] for k in x["merged_into"])
    # with only CSE the merged line stays, and the row says what it became and where it was merged
    r = optimize_program(SAMPLES["Common subexpressions"][2], passes=["Common Subexpression Elimination"])
    kept = [x for x in r["line_changes"] if x["merged_into"]]
    assert kept and all(not x["removed"] and x["final"].startswith(x["original"].split(" = ")[0] + " = t") for x in kept)


def test_global_cse_names_every_branch_line_that_computed_the_reused_value():
    r = optimize_program(IF_SRC)
    listing = {row["sno"]: row["stmt"] for row in r["original"]}
    (row,) = [x for x in r["line_changes"] if x["techniques"] == ["Global Common Subexpression Elimination"]]
    assert row["original"] == "z = x + n" and row["final"] == "z = y"
    assert len(row["merged_into"]) == 2                                      # y was computed in both arms of the if
    assert all(listing[k] == "y = x + n" and k < row["line"] for k in row["merged_into"])


def test_step_cap_stops_a_runaway_program_without_changing_the_default():
    loop = [("input", "n", "n"), ("label", "L"), ("jump", "L"), ("output", "n")]
    assert _run_tac(loop, {"n": 1}).ok is False
    res = _run_tac(loop, {"n": 1}, max_steps=50)
    assert res.ok is False and "step limit" in res.error
    ok = compile_function("def f(a):\n    return a + 1\n").tac
    assert _run_tac(ok, {"a": 1}, max_steps=3).ok is False and _run_tac(ok, {"a": 1}).ok is True
    assert GAConfig().max_steps == MAX_STEPS


def test_the_bundle_for_the_browser_runs_the_whole_comparison_on_its_own(tmp_path):
    """Everything the page ships to Pyodide must be enough for optimize_program, comparison included."""
    for name in ("individual", "pareto", "optimizations", "ga", "comparison"):
        assert name in LIVE_FILES
    pkg = tmp_path / "eco"
    pkg.mkdir()
    for name, text in live_sources().items():
        (pkg / name).write_text(text, encoding="utf-8")
    code = ("import sys, json; sys.path.insert(0, %r)\n"
            "from eco.code_optimizer import optimize_program_json\n"
            "r = json.loads(optimize_program_json(json.dumps({'source': %r})))\n"
            "assert r['ok'] and len(r['method_comparison']['methods']) == 10 and r['cfg'] and 'lines_changed' in r['summary']\n"
            "print('bundle ok')\n") % (str(tmp_path), IF_SRC)
    out = subprocess.run([_sys.executable, "-I", "-c", code], cwd=str(tmp_path), capture_output=True, text=True)
    assert out.returncode == 0 and "bundle ok" in out.stdout, out.stderr


def test_precomputed_dashboard_results_carry_the_new_sections(tmp_path):
    r = optimize_program(DEFAULT_SOURCE)
    assert {"line_changes", "cfg", "method_comparison"} <= set(r)
    html = open(TEMPLATE_PATH, encoding="utf-8").read()
    for text in ("Control flow graph", "Comparison with other methods", "Where ours wins", "Lines changed", "aria-expanded", "data-bit"):
        assert text in html
