"""A library of real, hand-written straight-line arithmetic kernels.

Each template below is an ordinary Python function written the way a person
would write it for a real purpose (Horner polynomial evaluation, squared
distance, 3x3 determinant, compound interest, BMI, kinetic energy, ...).
Nothing is added for the optimizer's benefit: the redundancy the optimizer
finds is the redundancy normal code has - a repeated ``(x2 - x1)``, a literal
chain like ``2 * 314``, a ``0 - b`` negation used twice, an alias such as
``inv_a = d``.

A template is a function ``rng -> KernelInstance`` that picks constants,
sizes (degree / vector length / number of years) and therefore the number
of arguments, and renders the source text. ``generate_kernel_program``
compiles an instance with the ``ast`` front end, runs the ORIGINAL Python
source on each test input set to get ground-truth outputs, and checks that
the TAC interpreter agrees before the program is admitted to the dataset.

Spelling labels. Some kernels exist in two spellings of the same function
(a "tidy" one that names an intermediate and a "naive" one that recomputes it
inline), chosen at random per instance; six kernels were written with extra
redundant lines in every instance ("naive"); the rest have a single form
("plain"). The label travels with every program (``BenchmarkProgram.spelling``)
into the reports and the summary CSV so results can be split by it.

Integer-division kernels use ``//`` and positive input domains with
non-negative numerators, where Python's floor division and the TAC
interpreter's truncating division coincide.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from .benchmark_generator import NUM_TEST_SETS, BenchmarkProgram
from .frontend import compile_function, run_source
from .interpreter import run
from .metrics import evaluate

SIGNED = (-15, 15)
POSITIVE = (1, 20)


@dataclass
class KernelInstance:
    kernel: str
    source: str
    arg_names: List[str]
    domain: Tuple[int, int]
    style: str = "plain"   # "plain" | "tidy" | "naive"


def _lit(v: int) -> str:
    return str(v) if v >= 0 else f"({v})"


def _coef(rng: random.Random) -> int:
    """Polynomial-style coefficient: sometimes 0 or 1, as real polynomials often have."""
    r = rng.random()
    if r < 0.15:
        return 0
    if r < 0.30:
        return 1
    return rng.randint(-9, 9)


def _fn(kernel: str, args: Sequence[str], lines: Sequence[str], ret: str,
        domain: Tuple[int, int] = SIGNED, style: str = "plain") -> KernelInstance:
    body = "".join(f"    {ln}\n" for ln in lines)
    source = f"def {kernel}({', '.join(args)}):\n{body}    return {ret}\n"
    return KernelInstance(kernel, source, list(args), domain, style)


def _names(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i}" for i in range(n)]


# ---------------------------------------------------------------- polynomials
def horner_poly(rng):
    n = rng.randint(3, 9)
    c = [rng.randint(1, 9)] + [_coef(rng) for _ in range(n)]
    lines = [f"r = {_lit(c[0])}"] + [f"r = r * x + {_lit(k)}" for k in c[1:]]
    return _fn("horner_poly", ["x"], lines, "r")


def naive_poly(rng):
    n = rng.randint(3, 7)
    terms = [str(_lit(_coef(rng) or 1))]
    for k in range(1, n + 1):
        power = "*".join(["x"] * k)
        terms.append(f"{_lit(_coef(rng) or 2)} * ({power})" if k > 1 else f"{_lit(_coef(rng) or 2)} * x")
    return _fn("naive_poly", ["x"], [f"r = {' + '.join(terms)}"], "r")


def poly_with_derivative(rng):
    c0, c1, c2, c3 = (rng.randint(1, 9) for _ in range(4))
    lines = [
        "x2 = x * x",
        "x3 = x2 * x",
        f"value = {c0} + {c1} * x + {c2} * x2 + {c3} * x3",
        f"slope = {c1} + 2 * {c2} * x + 3 * {c3} * x2",
    ]
    return _fn("poly_with_derivative", ["x"], lines, "value, slope")


def quadratic_value_and_slope(rng):
    lines = ["y = a * x * x + b * x + c", "dy = 2 * a * x + b"]
    return _fn("quadratic_value_and_slope", ["a", "b", "c", "x"], lines, "y, dy")


def quadratic_discriminant(rng):
    k = rng.choice([4, 4, 4, 2, 8])
    lines = [
        f"disc = b * b - {k} * a * c",
        "two_a = 2 * a",
        "root_hi = -b + disc",
        "root_lo = -b - disc",
    ]
    return _fn("quadratic_discriminant", ["a", "b", "c"], lines, "disc, root_hi, root_lo, two_a")


def binomial_expansion(rng):
    lines = ["square = (a + b) * (a + b)", "cube = (a + b) * (a + b) * (a + b)"]
    return _fn("binomial_expansion", ["a", "b"], lines, "square, cube")


def difference_of_squares(rng):
    lines = ["factored = (a + b) * (a - b)", "expanded = a * a - b * b",
             "residual = (a + b) * (a - b) - (a * a - b * b)"]  # identity self-check, always 0
    return _fn("difference_of_squares", ["a", "b"], lines, "factored, expanded, residual", style="naive")


# ------------------------------------------------------------------- geometry
def distance_sq_2d(rng):
    scale = rng.randint(2, 9)
    tidy = rng.random() < 0.5
    if tidy:
        lines = ["dx = x2 - x1", "dy = y2 - y1", "d2 = dx * dx + dy * dy"]
    else:
        lines = ["d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)"]
    lines.append(f"scaled = d2 * {scale}")
    return _fn("distance_sq_2d", ["x1", "y1", "x2", "y2"], lines, "d2, scaled", style="tidy" if tidy else "naive")


def distance_sq_3d(rng):
    lines = [
        "d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)",
    ]
    return _fn("distance_sq_3d", ["x1", "y1", "z1", "x2", "y2", "z2"], lines, "d2")


def midpoint_2d(rng):
    lines = ["mx = (x1 + x2) // 2", "my = (y1 + y2) // 2"]
    return _fn("midpoint_2d", ["x1", "y1", "x2", "y2"], lines, "mx, my", POSITIVE)


def triangle_area_twice(rng):
    lines = ["area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)",
             "side_ab_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)",
             "side_ac_sq = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1)"]
    return _fn("triangle_area_twice", ["x1", "y1", "x2", "y2", "x3", "y3"], lines,
               "area2, side_ab_sq, side_ac_sq", style="naive")


def rectangle_properties(rng):
    lines = ["area = w * h", "perimeter = 2 * (w + h)", "diagonal_sq = w * w + h * h"]
    return _fn("rectangle_properties", ["w", "h"], lines, "area, perimeter, diagonal_sq", POSITIVE)


def circle_properties(rng):
    pi100 = rng.choice([314, 314, 3142])
    scale = 100 if pi100 == 314 else 1000
    lines = [f"area = {pi100} * r * r // {scale}", f"circumference = 2 * {pi100} * r // {scale}"]
    return _fn("circle_properties", ["r"], lines, "area, circumference", POSITIVE)


def sphere_properties(rng):
    lines = ["volume = 4 * 314 * r * r * r // 300", "surface = 4 * 314 * r * r // 100"]
    return _fn("sphere_properties", ["r"], lines, "volume, surface", POSITIVE)


def cross_product_3d(rng):
    lines = ["cx = ay * bz - az * by", "cy = az * bx - ax * bz", "cz = ax * by - ay * bx",
             "norm_sq = (ay * bz - az * by) * (ay * bz - az * by) + cy * cy + cz * cz"]
    return _fn("cross_product_3d", ["ax", "ay", "az", "bx", "by", "bz"], lines, "cx, cy, cz, norm_sq",
               style="naive")


def dot_product(rng):
    n = rng.randint(2, 8)
    a, b = _names("a", n), _names("b", n)
    expr = " + ".join(f"{x} * {y}" for x, y in zip(a, b))
    return _fn("dot_product", a + b, [f"s = {expr}"], "s")


def cosine_similarity_parts(rng):
    n = rng.randint(2, 4)
    a, b = _names("a", n), _names("b", n)
    lines = [
        "dot = " + " + ".join(f"{x} * {y}" for x, y in zip(a, b)),
        "norm_a = " + " + ".join(f"{x} * {x}" for x in a),
        "norm_b = " + " + ".join(f"{y} * {y}" for y in b),
    ]
    return _fn("cosine_similarity_parts", a + b, lines, "dot, norm_a, norm_b")


def determinant_3x3(rng):
    args = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    if rng.random() < 0.5:
        lines = ["det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)"]
    else:
        lines = ["det = a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h"]
    return _fn("determinant_3x3", args, lines, "det")


def inverse_2x2_parts(rng):
    lines = ["det = a * d - b * c", "inv_a = d", "inv_b = -b", "inv_c = -c", "inv_d = a"]
    return _fn("inverse_2x2_parts", ["a", "b", "c", "d"], lines, "det, inv_a, inv_b, inv_c, inv_d")


def matrix_vector(rng):
    n = rng.randint(2, 3)
    cols = _names("v", n)
    args, lines, outs = [], [], []
    for r in range(n):
        row = [f"m{r}{c}" for c in range(n)]
        args += row
        lines.append(f"y{r} = " + " + ".join(f"{m} * {v}" for m, v in zip(row, cols)))
        outs.append(f"y{r}")
    return _fn("matrix_vector", args + cols, lines, ", ".join(outs))


def trace_and_frobenius(rng):
    names = [f"m{r}{c}" for r in range(3) for c in range(3)]
    lines = ["trace = m00 + m11 + m22", "frobenius_sq = " + " + ".join(f"{m} * {m}" for m in names)]
    return _fn("trace_and_frobenius", names, lines, "trace, frobenius_sq")


def affine_transform_2d(rng):
    a, b, c, d = (rng.randint(-4, 4) for _ in range(4))
    tx, ty = rng.randint(-9, 9), rng.randint(-9, 9)
    lines = [f"x2 = {_lit(a)} * x + {_lit(b)} * y + {_lit(tx)}", f"y2 = {_lit(c)} * x + {_lit(d)} * y + {_lit(ty)}"]
    return _fn("affine_transform_2d", ["x", "y"], lines, "x2, y2")


# ----------------------------------------------------------------- statistics
def mean_of_values(rng):
    n = rng.randint(3, 8)
    xs = _names("x", n)
    total = " + ".join(xs)
    lines = [f"total = {total}", f"mean = total // {n}"]
    tidy = rng.random() >= 0.5   # keeps the original random stream (inline sum was < 0.5)
    if tidy:
        lines.append(f"remainder = total - mean * {n}")
    else:
        lines.append(f"remainder = {total} - mean * {n}")
    return _fn("mean_of_values", xs, lines, "mean, remainder", POSITIVE, style="tidy" if tidy else "naive")


def variance_of_values(rng):
    n = rng.randint(3, 6)
    xs = _names("x", n)
    total = " + ".join(xs)
    tidy = rng.random() < 0.5
    if tidy:
        dev = " + ".join(f"({x} - mean) * ({x} - mean)" for x in xs)
        lines = [f"total = {total}", f"mean = total // {n}", f"var = ({dev}) // {n}"]
    else:  # recompute the mean inline instead of naming it
        dev = " + ".join(f"({x} - ({total}) // {n}) * ({x} - ({total}) // {n})" for x in xs)
        lines = [f"mean = ({total}) // {n}", f"var = ({dev}) // {n}"]
    return _fn("variance_of_values", xs, lines, "mean, var", POSITIVE, style="tidy" if tidy else "naive")


def weighted_average(rng):
    n = rng.randint(2, 5)
    xs, ws = _names("x", n), _names("w", n)
    num = " + ".join(f"{w} * {x}" for w, x in zip(ws, xs))
    wsum = " + ".join(ws)
    lines = [f"avg = ({num}) // ({wsum})", f"total_weight = {wsum}",
             f"scaled = ({num}) // total_weight"]
    return _fn("weighted_average", xs + ws, lines, "avg, total_weight, scaled", POSITIVE, style="naive")


def sum_and_sum_of_squares(rng):
    n = rng.randint(3, 7)
    xs = _names("x", n)
    lines = [f"s = {' + '.join(xs)}", f"q = {' + '.join(f'{x} * {x}' for x in xs)}"]
    return _fn("sum_and_sum_of_squares", xs, lines, "s, q")


def regression_slope_parts(rng):
    n = rng.randint(3, 5)
    xs, ys = _names("x", n), _names("y", n)
    lines = [
        f"sx = {' + '.join(xs)}",
        f"sy = {' + '.join(ys)}",
        "sxy = " + " + ".join(f"{x} * {y}" for x, y in zip(xs, ys)),
        "sxx = " + " + ".join(f"{x} * {x}" for x in xs),
        f"numerator = {n} * sxy - sx * sy",
        f"denominator = {n} * sxx - sx * sx",
    ]
    return _fn("regression_slope_parts", xs + ys, lines, "numerator, denominator")


def moving_average(rng):
    n = rng.randint(3, 6)
    xs = _names("x", n)
    lines = [f"window = {' + '.join(xs)}", f"avg = window // {n}", f"prev = (window - x0 + {xs[-1]}) // {n}"]
    return _fn("moving_average", xs, lines, "avg, prev", POSITIVE)


# -------------------------------------------------------------------- finance
def compound_interest(rng):
    years = rng.randint(2, 6)
    rate = rng.randint(2, 12)
    lines = [f"p{y + 1} = p{y} * (100 + {rate}) // 100" for y in range(years)]
    lines.insert(0, "p0 = principal")
    lines.append("gain = p%d - principal" % years)
    return _fn("compound_interest", ["principal"], lines, "p%d, gain" % years, POSITIVE)


def simple_interest(rng):
    months = rng.choice([6, 12, 12, 24])
    tidy = rng.random() < 0.5
    if tidy:
        lines = ["interest = principal * rate * years // 100", "total = principal + interest",
                 f"monthly = total // {months}"]
    else:
        lines = ["interest = principal * rate * years // 100",
                 "total = principal + principal * rate * years // 100",
                 f"monthly = (principal + principal * rate * years // 100) // {months}"]
    return _fn("simple_interest", ["principal", "rate", "years"], lines, "interest, total, monthly", POSITIVE,
               style="tidy" if tidy else "naive")


def income_tax(rng):
    rate, surcharge = rng.randint(5, 30), rng.randint(1, 10)
    lines = [f"tax = income * {rate} // 100 + income * {surcharge} // 100"]
    tidy = rng.random() < 0.5
    if tidy:
        lines.append("net = income - tax")
    else:
        lines.append(f"net = income - (income * {rate} // 100 + income * {surcharge} // 100)")
    return _fn("income_tax", ["income"], lines, "tax, net", POSITIVE, style="tidy" if tidy else "naive")


def discount_chain(rng):
    d1, d2 = rng.randint(5, 30), rng.randint(5, 20)
    lines = [f"after_first = price * (100 - {d1}) // 100", f"final = after_first * (100 - {d2}) // 100",
             "savings = price - final"]
    return _fn("discount_chain", ["price"], lines, "final, savings", POSITIVE)


# ------------------------------------------------------- health and unit work
def body_mass_index(rng):
    lines = ["bmi = weight * 10000 // (height * height)", "area = height * height",
             "ideal = 22 * area // 10000"]
    return _fn("body_mass_index", ["weight", "height"], lines, "bmi, ideal", (30, 200))


def temperature_conversions(rng):
    lines = ["fahrenheit = c * 9 // 5 + 32", "kelvin = c + 273", "reaumur = c * 4 // 5",
             "rankine = c * 9 // 5 + 32 + 460"]
    return _fn("temperature_conversions", ["c"], lines, "fahrenheit, kelvin, reaumur, rankine", POSITIVE,
               style="naive")


def length_conversions(rng):
    lines = ["meters = km * 1000", "centimeters = km * 1000 * 100", "millimeters = km * 1000 * 100 * 10"]
    return _fn("length_conversions", ["km"], lines, "meters, centimeters, millimeters", POSITIVE)


def speed_distance_time(rng):
    tidy = rng.random() < 0.5
    if tidy:
        lines = ["speed = distance // time", "later = speed * extra", "total = distance + later"]
    else:
        lines = ["speed = distance // time", "later = distance // time * extra",
                 "total = distance + distance // time * extra"]
    return _fn("speed_distance_time", ["distance", "time", "extra"], lines, "speed, total", POSITIVE,
               style="tidy" if tidy else "naive")


# -------------------------------------------------------------------- physics
def kinetic_and_potential_energy(rng):
    g = rng.choice([10, 10, 98])
    div = 1 if g == 10 else 10
    lines = ["ke = m * v * v // 2", f"pe = m * {g} * h // {div}"]
    tidy = rng.random() < 0.5
    lines.append("mechanical = ke + pe" if tidy else f"mechanical = m * v * v // 2 + m * {g} * h // {div}")
    return _fn("kinetic_and_potential_energy", ["m", "v", "h"], lines, "mechanical, ke", POSITIVE,
               style="tidy" if tidy else "naive")


def projectile_position(rng):
    lines = ["x = v * t", "y = v * t - 5 * t * t"]
    return _fn("projectile_position", ["v", "t"], lines, "x, y")


def ohms_law_power(rng):
    lines = ["voltage = current * resistance", "power = current * current * resistance",
             "check = voltage * current"]
    return _fn("ohms_law_power", ["current", "resistance"], lines, "voltage, power, check", POSITIVE)


def uniform_acceleration(rng):
    lines = ["s = u * t + a * t * t // 2", "v_final = u + a * t", "v_sq = u * u + 2 * a * s"]
    return _fn("uniform_acceleration", ["u", "a", "t"], lines, "s, v_final, v_sq", POSITIVE)


def linear_interpolation(rng):
    lines = ["y = (y0 * (20 - t) + y1 * t) // 20", "mirror = (y1 * (20 - t) + y0 * t) // 20"]
    return _fn("linear_interpolation", ["y0", "y1", "t"], lines, "y, mirror", POSITIVE, style="naive")


def rgb_to_gray(rng):
    lines = ["gray = (299 * r + 587 * g + 114 * b) // 1000"]
    return _fn("rgb_to_gray", ["r", "g", "b"], lines, "gray", POSITIVE)


TEMPLATES: Dict[str, Callable[[random.Random], KernelInstance]] = {
    fn.__name__: fn for fn in (
        horner_poly, naive_poly, poly_with_derivative, quadratic_value_and_slope,
        quadratic_discriminant, binomial_expansion, difference_of_squares,
        distance_sq_2d, distance_sq_3d, midpoint_2d, triangle_area_twice,
        rectangle_properties, circle_properties, sphere_properties, cross_product_3d,
        dot_product, cosine_similarity_parts, determinant_3x3, inverse_2x2_parts,
        matrix_vector, trace_and_frobenius, affine_transform_2d,
        mean_of_values, variance_of_values, weighted_average, sum_and_sum_of_squares,
        regression_slope_parts, moving_average,
        compound_interest, simple_interest, income_tax, discount_chain,
        body_mass_index, temperature_conversions, length_conversions, speed_distance_time,
        kinetic_and_potential_energy, projectile_position, ohms_law_power,
        uniform_acceleration, linear_interpolation, rgb_to_gray,
    )
}
TEMPLATE_NAMES: List[str] = list(TEMPLATES)


def generate_kernel_program(prog_id: int, rng: random.Random) -> BenchmarkProgram:
    """Instantiate template ``prog_id mod len(TEMPLATES)`` and build its benchmark record.

    Raises ``ValueError`` if the compiled TAC disagrees with the original
    Python source on any test input (it never should; this guards the
    ground truth).
    """
    name = TEMPLATE_NAMES[prog_id % len(TEMPLATE_NAMES)]
    inst = TEMPLATES[name](rng)
    compiled = compile_function(inst.source)
    assert compiled.arg_names == inst.arg_names

    lo, hi = inst.domain
    test_input_sets = [{a: rng.randint(lo, hi) for a in inst.arg_names} for _ in range(NUM_TEST_SETS)]

    expected = [run_source(inst.source, inst.kernel, ins) for ins in test_input_sets]  # ground truth
    for ins, want in zip(test_input_sets, expected):
        got = run(compiled.tac, ins)
        if not got.ok or got.outputs != want:
            raise ValueError(f"front end disagrees with Python source for {inst.kernel} on {ins}: "
                             f"TAC={got.outputs} source={want}")

    base_eval = evaluate(compiled.tac, test_input_sets, expected)
    return BenchmarkProgram(
        id=prog_id,
        name=f"bench_{prog_id:04d}",
        source_code=inst.source,
        tac=compiled.tac,
        input_names=inst.arg_names,
        test_input_sets=test_input_sets,
        expected_outputs=expected,
        baseline_fitness=base_eval.fitness,
        kernel=inst.kernel,
        spelling=inst.style,
    )
