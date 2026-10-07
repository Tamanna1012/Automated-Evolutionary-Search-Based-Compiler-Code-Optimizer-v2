# Automated Evolutionary Search-Based Compiler Code Optimizer

A fully automated system that maintains a population of alternative program
versions, evolves them across generations using selection, mutation and
crossover, and discovers Pareto-optimal optimized programs that balance
four competing compiler metrics: **execution time**, **instruction count**,
**code size**, and **arithmetic-operation count**.

## Quick start

```bash
pip install -r requirements.txt
python -m eco.main          # runs the full experiment end-to-end
python -m pytest tests/ -q  # unit / correctness tests
```

`python -m eco.main` will:

1. Generate a 300-program synthetic TAC benchmark suite.
2. Run the evolutionary search (population 30, 50 generations) on **every**
   program to compute the aggregate Testing & Metrics table.
3. Produce detailed per-program reports for the 20 largest ("deep-dive")
   programs, including a generation-by-generation cost trace, the final
   Pareto-optimal solutions and the best-overall (weighted) individual.
4. Render all six required plots to `eco/outputs/plots/` and write text /
   JSON reports to `eco/outputs/reports/`.

A full run (300 programs x 50 generations) takes about 5 minutes on a single core.

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

### Benchmark dataset (`eco/benchmark_generator.py`, `eco/dataset.py`)

`generate_dataset(n, seed)` synthesizes `200 <= n <= 500` (default 300)
straight-line arithmetic programs (38-78 instructions each), each with:
pseudo-C source, its TAC, 5 randomized test-input sets, ground-truth
expected outputs (from executing the un-optimized TAC), and baseline
metrics. Programs deliberately contain redundant/repeated subexpressions
(CSE fodder), dead temporaries (DCE fodder), multi-hop copy chains (copy
propagation fodder), deep constant-arithmetic chains (constant folding
fodder), identity-friendly literals like 0/1 (algebraic simplification
fodder), and a final "scaling" block of `x*k` multiplies (`k` = 2, 3, 4) and
repeated-add chains (`strength_reduction` / `multiply_fusion` fodder) — so
every optimization has genuine opportunities to fire. The scaling block uses
its own RNG stream, so the rest of each program and its test inputs are
generated exactly as before.

## Testing & Metrics (last full run, 300 programs)

| Metric | Result | Target |
|---|---|---|
| Validity Rate | 100.0% | > 90% |
| Avg. Pareto Front Size | 21.5 individuals (avg. 1.65 distinct objective-space points) | identify trade-offs |
| Best Cost Improvement (exec_time vs. baseline) | 54.5% | > 30% |
| Avg. Generations to Convergence | 15.2 | tracked per program |
| Correctness PASS rate | 100.0% | — |

Regenerate this table with `python -m eco.main` (see
`eco/outputs/reports/aggregate_metrics.md`/`.json`).

### A note on the Pareto front

Before the trade-off passes existed, all six optimizations were
simplifications, so the final front collapsed to a single distinct
objective-space point (avg. 1.01 per program). With `strength_reduction` and
`multiply_fusion` the final front now holds more than one point for most
programs (avg. 1.65; many genomes still tie on the same point, hence the
much larger raw front size). Fronts are still small because only the `x*k`
and repeated-add patterns trade objectives; everything else remains a pure
win that every good individual applies.

`eco/outputs/plots/2_pareto_front_2d.png` and `6_pareto_front_3d.png` plot
the **final population** of the deep-dive program whose final front has the
most distinct points; the plot title names that program, the generation and
the number of distinct front points (and says so if the richest front is
still a single point).

Note on comparing runs: the cost-model change (add/sub 2 -> 1 cycle) and the
extra scaling block in the benchmark generator change the baselines, so
absolute improvement percentages are not directly comparable with older runs.

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
   baseline across the 20 deep-dive programs.
6. `5_elite_optimization_heatmap.png` — which optimizations appear most
   often in elite individuals, per generation, aggregated across the 20
   deep-dive programs.
7. `6_pareto_front_3d.png` — Pareto front in 3D (`exec_time`,
   `instr_count`, `code_size`), same program/population as plot 3.

## Expected output (`eco/outputs/reports/`)

- `<program>_report.txt` / `deep_dive_reports.json` — per program: best/avg
  cost at generations 1, 5, 10, 20, 50, the final Pareto-optimal solutions
  and what each represents, the best-overall (weighted) individual with its
  genome and optimized TAC, and a PASS/FAIL correctness verdict.
- `aggregate_metrics.md` / `.json` — the Testing & Metrics table above,
  computed across the full dataset.

## Project layout

```
eco/
  tac.py                 TAC instruction representation + pretty-printers
  interpreter.py          TAC interpreter (correctness + simulated cycles)
  metrics.py               instr_count / arith_ops / code_size / exec_time
  optimizations.py        the eight optimization passes + genome replay
  individual.py            candidate program representation
  pareto.py                 dominance + Pareto front + trade-off labels
  ga.py                     population init, selection, mutation, crossover,
                            generational loop
  benchmark_generator.py   synthesizes one random benchmark program
  dataset.py                 assembles the 200-500 program dataset
  report.py                  per-program + aggregate report builders
  visualize.py                the six required plots
  main.py                      end-to-end experiment driver
  outputs/
    plots/                  generated PNGs
    reports/                generated text/JSON/Markdown reports
tests/
  test_eco.py              correctness + regression tests
```

