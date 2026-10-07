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
2. Run the evolutionary search (population 30, 30 generations) on **every**
   program to compute the aggregate Testing & Metrics table.
3. Produce detailed per-program reports for the 20 largest ("deep-dive")
   programs, including a generation-by-generation cost trace, the final
   Pareto-optimal solutions and the best-overall (weighted) individual.
4. Render all six required plots to `eco/outputs/plots/` and write text /
   JSON reports to `eco/outputs/reports/`.

A full run takes ~25 seconds on a single core.

## System design

### TAC and the interpreter (`eco/tac.py`, `eco/interpreter.py`)

Each benchmark "program" is straight-line three-address code built from five
instruction kinds: `input`, `const`, `copy`, `bin` (`+ - * /`), `output`.
A tiny interpreter (`eco/interpreter.py`) executes a program against a set
of input values, returning both its outputs (for correctness checking) and
a simulated cycle count (`+`/`-` = 2 cycles, `*` = 3, `/` = 4, everything
else = 1) that stands in for `exec_time`.

### The six optimizations (`eco/optimizations.py`)

| # | Optimization | What it does |
|---|---|---|
| 1 | `constant_folding` | Evaluates arithmetic whose operands are already literal constants |
| 2 | `constant_propagation` | Substitutes known-constant variables into later uses |
| 3 | `copy_propagation` | Forward-substitutes `copy` targets so uses reference the original variable |
| 4 | `cse` | Reuses an earlier temp when an identical expression recurs |
| 5 | `dead_code_elimination` | Backward liveness sweep that drops unused definitions |
| 6 | `algebraic_simplification` | Simplifies identities: `x+0`, `x*1`, `x*0`, `x-0`, `x/1`, `x-x` |

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
  always included, the rest apply a random subset of the six optimizations.
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
  each generation, for `NUM_GENERATIONS=30`.

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
straight-line arithmetic programs (28-58 instructions each), each with:
pseudo-C source, its TAC, 5 randomized test-input sets, ground-truth
expected outputs (from executing the un-optimized TAC), and baseline
metrics. Programs deliberately contain redundant/repeated subexpressions
(CSE fodder), dead temporaries (DCE fodder), multi-hop copy chains (copy
propagation fodder), deep constant-arithmetic chains (constant folding
fodder), and identity-friendly literals like 0/1 (algebraic simplification
fodder) — so every optimization has genuine opportunities to fire.

## Testing & Metrics (last full run, 300 programs)

| Metric | Result | Target |
|---|---|---|
| Validity Rate | 100.0% | > 90% |
| Avg. Pareto Front Size | 23.2 individuals (avg. 1.0 distinct objective-space points) | identify trade-offs |
| Best Cost Improvement (exec_time vs. baseline) | 85.6% | > 30% |
| Avg. Generations to Convergence | 4.5 | tracked per program |
| Correctness PASS rate | 100.0% | — |

Regenerate this table with `python -m eco.main` (see
`eco/outputs/reports/aggregate_metrics.md`/`.json`).

### A note on the Pareto front

For any single program, the final (fully-converged) Pareto front is almost
always a **single distinct objective-space point**, reached by many
different genomes. This is an honest and expected consequence of the six
chosen optimizations: on straight-line code, all six are *simplifications*
that never trade one objective for another — every one of them helps
`exec_time`, `instr_count`, `code_size` and `arith_ops` together (or leaves
them unchanged), never one at the expense of another. That's exactly what
"peephole"/local optimizations do in real compilers too; the classic
size-vs-speed *tensions* (loop unrolling, inlining, vectorization) require
control flow and function boundaries that this simplified straight-line TAC
model doesn't include.

The dominance/front machinery itself (`eco/pareto.py`) is fully general and
correctly handles genuine trade-offs whenever they exist — which is visible
*mid-search*, before elitism converges the population onto the single
dominant optimum. `eco/outputs/plots/2_pareto_front_2d.png` and
`6_pareto_front_3d.png` are therefore taken from an early-generation
population snapshot (chosen automatically as whichever deep-dive
program/generation has the richest front), clearly labeled with its
generation number, to actually show the spread of dominated vs.
non-dominated candidates the search is choosing between.

## Plots (`eco/outputs/plots/`)

1. `1_convergence_curve.png` — best/avg cost per generation for the
   flagship (largest) program — the convergence curve.
2. `1b_convergence_curves_20programs.png` — convergence curves overlaid
   across all 20 deep-dive programs plus their mean.
3. `2_pareto_front_2d.png` — Pareto front in 2D (`exec_time` vs
   `instr_count`).
4. `3_fitness_boxplot.png` — cost distribution across generations 1, 5,
   10, 20, 30.
5. `4_improvement_bar.png` — best-individual improvement vs. original
   baseline across the 20 deep-dive programs.
6. `5_elite_optimization_heatmap.png` — which optimizations appear most
   often in elite individuals, per generation, aggregated across the 20
   deep-dive programs.
7. `6_pareto_front_3d.png` — Pareto front in 3D (`exec_time`,
   `instr_count`, `code_size`).

## Expected output (`eco/outputs/reports/`)

- `<program>_report.txt` / `deep_dive_reports.json` — per program: best/avg
  cost at generations 1, 5, 10, 20, 30, the final Pareto-optimal solutions
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
  optimizations.py        the six optimization passes + genome replay
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

