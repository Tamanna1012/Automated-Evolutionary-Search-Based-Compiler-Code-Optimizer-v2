# Automated Evolutionary Search-Based Compiler Code Optimizer

A fully automated system that maintains a population of alternative program
versions, evolves them across generations using selection, mutation and
crossover, and discovers Pareto-optimal optimized programs that balance
four competing compiler metrics: **execution time**, **instruction count**,
**code size**, and **arithmetic-operation count**.

## Quick start

```bash
pip install -r requirements.txt
python -m eco.main          # runs the full experiment end-to-end (real kernels)
python -m eco.main --synthetic   # same, on the old random synthetic TAC (-> eco/outputs_synthetic/)
python -m pytest tests/ -q  # unit / correctness tests
```

`python -m eco.main` will:

1. Generate a 300-program benchmark suite by instantiating 42 real hand-written
   kernels and compiling their Python source to TAC with an `ast` front end.
2. Run the evolutionary search (population 30, 50 generations) on **every**
   program to compute the aggregate Testing & Metrics table.
3. Produce detailed per-program reports for the 20 largest ("deep-dive")
   programs, including a generation-by-generation cost trace, the final
   Pareto-optimal solutions and the best-overall (weighted) individual.
4. Write the same style of detailed report for **all 300 programs** into
   `eco/outputs/reports/all_programs/`, plus one `all_programs_summary.csv`.
5. Render all six required plots to `eco/outputs/plots/` and write text /
   JSON reports to `eco/outputs/reports/`.

A full run (300 programs x 50 generations) takes about 1-2 minutes on a single core.

## System design

### TAC and the interpreter (`eco/tac.py`, `eco/interpreter.py`)

Each benchmark "program" is straight-line three-address code built from five
instruction kinds: `input`, `const`, `copy`, `bin` (`+ - * /`), `output`.
A tiny interpreter (`eco/interpreter.py`) executes a program against a set
of input values, returning both its outputs (for correctness checking) and
a simulated cycle count (`+`/`-` = 1 cycle, `*` = 3, `/` = 4, everything
else = 1) that stands in for `exec_time`. Add/sub being much cheaper than a
multiply is what lets strength reduction trade speed against size.

### The eight optimizations (`eco/optimizations.py`)

| # | Optimization | What it does |
|---|---|---|
| 1 | `constant_folding` | Evaluates arithmetic whose operands are already literal constants |
| 2 | `constant_propagation` | Substitutes known-constant variables into later uses |
| 3 | `copy_propagation` | Forward-substitutes `copy` targets so uses reference the original variable |
| 4 | `cse` | Reuses an earlier temp when an identical expression recurs |
| 5 | `dead_code_elimination` | Backward liveness sweep that drops unused definitions |
| 6 | `algebraic_simplification` | Simplifies identities: `x+0`, `x*1`, `x*0`, `x-0`, `x/1`, `x-x` |
| 7 | `strength_reduction` | Rewrites `x*2 -> x+x`, `x*3 -> (x+x)+x`, `x*4 -> (x+x)+(x+x)`. Adds are cheaper than a multiply, so **`exec_time` drops**, but for `k=3,4` one multiply becomes two adds, so **`arith_ops` and `code_size` rise** (the leftover constant is for `dead_code_elimination` to remove) |
| 8 | `multiply_fusion` | The size-oriented inverse: `(x+x)+x -> x*3`, `(x+x)+(x+x) -> x*4` (reusing an existing literal or emitting one). **`arith_ops` and `code_size` drop**, but a multiply costs more cycles than the adds it replaces, so **`exec_time` rises** |

Passes 1-6 only ever *simplify* (they improve every objective together or
leave it unchanged). Passes 7 and 8 are the first that **trade objectives
against each other**: neither dominates the other, so a genome that includes
`strength_reduction` and one that includes `multiply_fusion` land at
different, mutually non-dominated points. Both are exact, semantics
preserving rewrites over unbounded integers (`x*k == x+...+x`), checked by
the validity test on every individual. I did not add a third "rematerialize
instead of reuse a temp" pass: under the four tracked objectives
recomputing an expression instead of copying its temp is never better on
any of them (it only helps register pressure, which is not measured here),
so it could never appear on a Pareto front.

`constant_folding`, `constant_propagation` and `copy_propagation` are
deliberately **single-hop**: each only resolves one link of a dependency
chain (mirroring how a real optimizing compiler runs a pass repeatedly in
a fixpoint loop rather than making one pass all-powerful). This means a
program's genome — the *sequence* of optimizations applied — genuinely
matters: order and repetition change the result, not just which passes are
present. It's also what gives the evolutionary search real work to do
(see "A note on the Pareto front" below) rather than converging in one
generation.

### Individual representation (`eco/individual.py`)

Each individual carries: its resulting `tac`, a `genome` (the ordered list
of optimization names applied to the *original* program to reach it), a
`fitness` record (`exec_time`, `instr_count`, `code_size`, `arith_ops`), and
a `valid` flag (True iff it reproduces the original program's outputs on
every test-input set). A genome is always replayed from scratch against the
original TAC, so mutation/crossover only ever manipulate the genome, never
the derived program directly.

### Evolutionary algorithm (`eco/ga.py`)

- **Initialization**: `P=30` individuals; the unmodified original program is
  always included, the rest apply a random subset of the eight optimizations.
- **Fitness evaluation**: every individual is executed against all of the
  program's test-input sets; the four metrics and a correctness `valid`
  flag are computed per `eco/metrics.py`.
- **Selection**: tournament selection (3 random individuals, keep the
  lowest-cost) plus elitism (top 10% carried unchanged into the next
  generation).
- **Mutation**: with probability 0.7, insert one random optimization into
  the genome; with probability 0.3, remove one.
- **Crossover**: one-point cut of two (already-mutated) parent genomes,
  producing an offspring genome whose TAC is re-derived from the original
  program.
- **Generational loop**: evaluate → keep elites → select → mutate →
  crossover → form next population, logging best/average weighted cost
  each generation, for `NUM_GENERATIONS=50`.

Cost, for selection/elitism/reporting, is a weighted sum of each objective
normalized against the program's own baseline (equal 0.25 weights by
default); invalid individuals get a large penalty so incorrect programs are
naturally selected against without being removed from the population.

### Pareto-front analysis (`eco/pareto.py`)

Individual A dominates B if A is at least as good on all four objectives
and strictly better on at least one. `pareto_front()` extracts the
non-dominated set; `describe_tradeoff()` labels what each front member is
best at (fastest / fewest instructions / smallest code / fewest arithmetic
ops / balanced).

### Benchmark dataset (`eco/kernels.py`, `eco/frontend.py`, `eco/dataset.py`)

`generate_dataset(n, seed)` builds `200 <= n <= 500` (default 300) programs.
See "Dataset and measurement methodology" below for how they are made; the
old random generator (`eco/benchmark_generator.py`) is kept behind
`generate_dataset(..., kind="synthetic")` / `python -m eco.main --synthetic`.

## Testing & Metrics (last full run, 300 programs)

| Metric | Result | Target |
|---|---|---|
| Validity Rate | 100.0% | > 90% |
| Avg. Pareto Front Size | 24.9 individuals (avg. 1.04 distinct objective-space points) | identify trade-offs |
| Best Cost Improvement (simulated exec_time vs. baseline) | **13.1%** | > 30% (**not met** on real kernels, see below) |
| Avg. Generations to Convergence | 9.1 | tracked per program |
| Measured improvement (real runtime, original vs best) | **9.7%** (avg speedup 1.13x; 1.12x on total time; 9.2-9.7% over three runs) | validation of the cycle model |
| Optimized code correct when really run | 100.0% | — |
| Simulated vs measured improvement % (300 programs) | Pearson r = 0.83, Spearman rho = 0.88 (0.77-0.83 / 0.83-0.88 over three runs) | model should predict speedups |
| Correctness PASS rate | 100.0% | — |

Regenerate this table with `python -m eco.main` (see
`eco/outputs/reports/aggregate_metrics.md`/`.json`).

### Honest reading of these numbers (real kernels)

These are the results on the **real kernels**, and two of them are weaker than
the earlier results on the random synthetic programs (54.5% improvement and
1.65 distinct Pareto points per program with `--synthetic`):

* **Improvement is 13.1%, below the 30% target.** The distribution is very
  uneven: 100 of 300 programs improve by exactly 0% (kernels such as
  `dot_product`, `determinant_3x3`, `matrix_vector`, `trace_and_frobenius`,
  `rgb_to_gray` are already tight - there is nothing to remove), 39 programs
  improve by 30% or more (best: `difference_of_squares` 40%, `binomial_expansion`
  39%, `length_conversions` 36%, `naive_poly` 32%), and the maximum is 47.8%.
  Per-kernel averages are in `all_programs/all_programs_summary.csv`.
* **Where redundancy exists the search still leaves some on the table.** For
  example a variance kernel that recomputes its mean inline reaches only ~8%
  although most of its TAC is redundant, because CSE rewrites a repeat as a
  copy and `copy_propagation` / `cse` / `constant_*` are single-hop passes, so
  deeply nested repeats need many rounds. This is a limit of the pass set, not
  of the genome length: raising the genome cap from 5 to 15 in an experiment
  moved the average only from 13.8% to 15.2%. A value-numbering CSE that looks
  through copies would be the real fix; it is not part of this change.
* **Pareto fronts are mostly a single point (avg. 1.04; 13 of 300 programs
  have more than one).** `strength_reduction` / `multiply_fusion` only trade
  objectives when a program multiplies by 2, 3 or 4 or adds the same value
  repeatedly, which real kernels rarely do. The two trade-off passes are
  still correct and tested; they simply have few targets in real code.

`eco/outputs/plots/2_pareto_front_2d.png` and `6_pareto_front_3d.png` plot
the **final population** of the deep-dive program with the richest final
front; the title names the program, the generation and the number of
distinct front points (and says so when the richest front is a single point).

## Plots (`eco/outputs/plots/`)

1. `1_convergence_curve.png` — best/avg cost per generation for the
   flagship (largest) program — the convergence curve.
2. `1b_convergence_curves_20programs.png` — convergence curves overlaid
   across all 20 deep-dive programs plus their mean.
3. `2_pareto_front_2d.png` — Pareto front in 2D (`exec_time` vs
   `instr_count`), final population of the richest-front deep-dive program.
4. `3_fitness_boxplot.png` — cost distribution across generations 1, 5,
   10, 20, 50.
5. `4_improvement_bar.png` — best-individual improvement vs. original
   baseline across the 20 deep-dive programs, with **two bars per program**:
   the simulated improvement (cycle model) and the measured improvement
   (real runtime).
6. `5_elite_optimization_heatmap.png` — which optimizations appear most
   often in elite individuals, per generation, aggregated across the 20
   deep-dive programs.
7. `6_pareto_front_3d.png` — Pareto front in 3D (`exec_time`,
   `instr_count`, `code_size`), same program/population as plot 3.
8. `7_cycles_vs_measured_runtime.png` — scatter of simulated cycles against
   measured nanoseconds for all 600 measured programs (original and best of
   each), titled with the Pearson and Spearman correlation.

## Expected output (`eco/outputs/reports/`)

- `<program>_report.txt` / `deep_dive_reports.json` — per program: best/avg
  cost at generations 1, 5, 10, 20, 50, the final Pareto-optimal solutions
  and what each represents, the best-overall (weighted) individual with its
  genome and optimized TAC, the original source, the simulated **and measured**
  improvement, and a PASS/FAIL correctness verdict.
- `all_programs/<program>_report.txt` — the same report for **every one of
  the 300 programs** (best/avg cost at generations 1, 5, 10, 20, 50, final
  Pareto solutions with what each is best at, best weighted individual with
  its genome and optimized TAC, PASS/FAIL verdict), kept in their own folder
  to avoid 300 loose files.
- `all_programs/all_programs_summary.csv` — one row per program: `program`,
  `baseline_cost` / `best_cost` (the weighted, baseline-normalized cost the
  GA minimizes, so the baseline is 1.0), `baseline_exec_time` /
  `best_exec_time`, `improvement_pct` (simulated exec_time vs baseline),
  `measured_baseline_ns`, `measured_best_ns`, `measured_improvement_pct`,
  `measured_speedup_x` (real runtime),
  `generations_to_convergence`, `pareto_front_size`, `distinct_points`,
  `verdict`.
- `aggregate_metrics.md` / `.json` — the Testing & Metrics table above,
  computed across the full dataset.

## Project layout

```
eco/
  frontend.py             ast front end: Python source -> TAC
  runtime.py              real runtime: TAC -> Python function, timing, correlation
  kernels.py              42 real kernel templates + dataset-record builder
  tac.py                 TAC instruction representation + pretty-printers
  interpreter.py          TAC interpreter (correctness + simulated cycles)
  metrics.py               instr_count / arith_ops / code_size / exec_time
  optimizations.py        the eight optimization passes + genome replay
  individual.py            candidate program representation
  pareto.py                 dominance + Pareto front + trade-off labels
  ga.py                     population init, selection, mutation, crossover,
                            generational loop
  benchmark_generator.py   (optional) random synthetic program generator
  dataset.py                 assembles the 200-500 program dataset
  report.py                  per-program, all-program + aggregate report builders
  visualize.py                the six required plots
  main.py                      end-to-end experiment driver
  outputs/
    plots/                  generated PNGs
    reports/                generated text/JSON/Markdown reports
      all_programs/         report for each of the 300 programs + summary CSV
tests/
  test_eco.py              correctness + regression tests
```

## Dataset and measurement methodology

### Where the programs come from

The default dataset is **real source code, not random TAC**. `eco/kernels.py`
holds 42 hand-written Python functions from real domains:

* polynomials: Horner and naive evaluation, polynomial with derivative,
  quadratic value/slope and discriminant, `(a+b)^2/^3`, difference of squares
* geometry: squared distance 2D/3D, midpoint, triangle area, rectangle,
  circle and sphere formulas, 3D cross product, 2D affine transform
* linear algebra: dot product, cosine-similarity parts, 3x3 determinant
  (cofactor and full expansion), 2x2 inverse parts, matrix-vector product,
  trace/Frobenius norm
* statistics: mean, variance, weighted average, sum/sum of squares,
  regression-slope parts, moving average
* finance: compound interest, simple interest, income tax, discount chain
* health, units, physics: BMI, temperature and length conversions, speed,
  kinetic/potential energy, projectile position, Ohm's law power, uniform
  acceleration, linear interpolation, RGB to gray

Each template is a function of a random generator that picks constants,
sizes (polynomial degree, vector length, number of years) and therefore the
number of arguments, and renders ordinary source such as
`r = r * x + 5`. The 300 programs are the templates instantiated round-robin
(about 7 instances each) with different constants, sizes and argument counts.
The source is written the way a person writes it - repeated subexpressions,
literal chains such as `2 * 314`, a negation used twice, an alias like
`inv_a = d` - and some kernels randomly choose between a tidy and a naive
spelling (for example naming the mean versus recomputing it inline). Nothing
was added to the code purely to give the optimizer something to do, and many
kernels (see above) have nothing to optimize.

### Front end (`eco/frontend.py`)

A small compiler built on Python's `ast` module turns one function into TAC.
It supports exactly what is needed: assignments and augmented assignments to
names, integer literals, `+ - * //`, unary minus, and `return` of one or more
values (each becomes an `output`). Anything else (true division, `**`, `%`,
floats, branches, loops, undefined names) raises `UnsupportedSyntax`.
`//` maps to the TAC `/`. Code generation is naive on purpose: no expression
value is ever reused, so source-level redundancy survives for the optimizer;
like any real compiler it pools literals (one `const` per distinct value) and
it renames reassigned variables so the TAC stays in single-assignment form.

### Ground truth

For every program the expected outputs are produced by **running the original
Python function** on each of 5 random input sets (`run_source`). The TAC
interpreter must reproduce them exactly before the program is admitted;
generation raises an error otherwise. Python's `//` floors while the TAC
interpreter truncates toward zero, so kernels that divide are given positive
input domains with non-negative numerators, where the two agree.

### Fitness during the search

The GA's `exec_time` objective is a deterministic **simulated cycle count**
(add/sub 1, multiply 3, divide 4, everything else 1). It is used because it
is fast and reproducible, so tests and results are stable. It is not a
hardware measurement; real time is measured separately, as validation.

### Measured runtime (`eco/runtime.py`)

For every program the original TAC and the final best individual are
translated to a real Python function (one statement per TAC instruction;
division calls the same truncating helper the interpreter uses), run in a
loop and timed with `time.perf_counter_ns`:

* a warm-up batch, then 7 timed batches of 200 passes over the 5 test-input
  sets, with the garbage collector paused;
* the reported time is the **minimum** batch time per call (the usual
  low-noise estimator), averaged over the test inputs like the cycle model;
* results are cached by a SHA-1 of the TAC (plus argument names and inputs),
  so an identical program is timed once and always gets the same value;
* the compiled optimized code is also checked against the ground-truth
  outputs, so a speedup can never come from wrong code (100.0% correct).

Only the Python backend exists. A C backend was not built: no C compiler is
guaranteed to be installed (none was available here), and a ctypes call has
a fixed overhead that would swamp kernels this small. So "measured" means
CPython executing the generated function, not native machine code.

Timing is noisy by nature. Three consecutive full runs gave measured
improvements of 9.2%, 9.3% and 9.7% (committed outputs: 9.7%) and
correlations that moved by a few hundredths (improvement-correlation Pearson
0.77, 0.83, 0.83; within-program Spearman 0.50, 0.45, 0.56), which is why
ranges are quoted below. The simulated numbers are deterministic.

### How well do simulated cycles predict real time?

Three views, all reported in `aggregate_metrics.md/.json`:

| View | Result | Reading |
|---|---|---|
| Pooled: cycles vs ns over all 600 measured programs (original + best) | Pearson r = 0.70, Spearman rho = 0.66 (0.69-0.70 / 0.64-0.66) | **Moderate**, and flattered by program size: bigger programs are slower under both measures |
| Per program: simulated improvement % vs measured improvement % (300 programs) | Pearson r = 0.83, Spearman rho = 0.88 (0.77-0.83 / 0.83-0.88) | **Good** for deciding how much an optimization helps |
| Within one program: rank agreement across the distinct programs in its final population (170 programs with >= 3 distinct points) | mean Spearman rho = 0.56 (0.45-0.56) | **Weak to moderate**: the model only partly ranks close variants of the same program correctly |

So the cycle model is a decent predictor of *whether and roughly how much*
the search helps, but a poor fine-grained ranker of near-identical variants.
Plausible reasons: CPython charges roughly a constant cost per statement
regardless of operator while the model charges multiply 3 and divide 4;
division additionally pays for a function call in the generated code; and
`const`/`copy` statements are cheap in both but not in the same proportion.
These mismatches are the likely reason the measured improvement (about 9-10%) is
smaller than the simulated one (13.1%); this was not isolated by a separate
experiment. The trade-off passes (`strength_reduction`, `multiply_fusion`)
rely on a multiply-vs-add cost gap that CPython probably does not have (an
extra statement costs about as much as the multiply it replaces), so their
benefit should be expected in the model more than in measured time.

## Limitations

* The kernels are **straight-line integer arithmetic**: no loops, branches,
  memory, floating point or function calls, so none of the classic
  loop/inlining/vectorization trade-offs exist here.
* **GA fitness uses a cycle model.** Real time is measured only afterwards, as
  validation, and only through a Python backend (no C/native code); the model
  and CPython disagree on per-operation costs (see above).
* Real kernels are often already tight: the average simulated improvement is
  13.1% (about 9-10% measured) and the 30% target is **not met** on this dataset.
  The earlier 54.5% was obtained on random synthetic programs written to
  contain redundancy (`--synthetic`).
* Timing is noisy; numbers vary by a few percent between runs.
* The search is a single seed (42); results are not averaged over seeds.
