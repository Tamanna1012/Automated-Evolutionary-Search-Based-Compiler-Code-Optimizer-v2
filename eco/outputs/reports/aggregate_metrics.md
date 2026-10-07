# Aggregate Testing & Metrics
- Programs evaluated: 300
- Validity Rate: 100.0%  (target > 90%)
- Avg. Pareto Front Size: 25.43 (avg. 1.07 distinct objective-space points)
- Best Cost Improvement (avg exec_time vs baseline): 15.6%  (target > 30%)
- Avg. Generations to Convergence: 4.1
- Correctness PASS rate: 100.0%

## Real measured runtime (validation; Python backend, min over timed rounds)
- Avg. measured improvement (original vs best): 12.2%  (simulated: 15.6%)
- Avg. measured speedup: 1.20x per program; 1.21x on total time
- Programs measurably faster (> 2%): 61.7%
- Optimized code still gives the expected outputs when really run: 100.0%

## How well do simulated cycles predict real time?
- Pooled over 600 measured programs (original + best of each): Pearson r = 0.65, Spearman rho = 0.62 (inflated by program size)
- Simulated vs measured *improvement %* across 300 programs: Pearson r = 0.90, Spearman rho = 0.92
- Within one program (ranking its final-population variants), mean Spearman rho = 0.63 over 184 programs
