# Aggregate Testing & Metrics
- Programs evaluated: 300
- Validity Rate: 100.0%  (target > 90%)
- Avg. Pareto Front Size: 24.94 (avg. 1.04 distinct objective-space points)
- Best Cost Improvement (avg exec_time vs baseline): 13.1%  (target > 30%)
- Avg. Generations to Convergence: 9.1
- Correctness PASS rate: 100.0%

## Real measured runtime (validation; Python backend, min over timed rounds)
- Avg. measured improvement (original vs best): 9.7%  (simulated: 13.1%)
- Avg. measured speedup: 1.13x per program; 1.12x on total time
- Programs measurably faster (> 2%): 59.7%
- Optimized code still gives the expected outputs when really run: 100.0%

## How well do simulated cycles predict real time?
- Pooled over 600 measured programs (original + best of each): Pearson r = 0.70, Spearman rho = 0.66 (inflated by program size)
- Simulated vs measured *improvement %* across 300 programs: Pearson r = 0.83, Spearman rho = 0.88
- Within one program (ranking its final-population variants), mean Spearman rho = 0.56 over 170 programs
