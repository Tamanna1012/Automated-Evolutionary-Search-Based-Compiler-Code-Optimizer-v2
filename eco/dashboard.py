"""Builds the self-contained results dashboard (``outputs/dashboard.html``).

The dashboard is a single HTML file: the experiment's real results are
embedded as JSON and rendered with plain JavaScript and inline SVG, so it
opens by double-click (no server, no internet, no libraries). Nothing in it
is hard-coded - every number comes from the run that wrote it.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .code_optimizer import DEFAULT_SOURCE, optimize_program
from .report import program_report
from .tac import tac_to_text

HERE = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(HERE, "dashboard_template.html")
PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"
LIVE_PLACEHOLDER = "/*__LIVE_SOURCES__*/null"
# The modules the Code Optimization tab runs in the browser (via Pyodide) on a visitor's
# own program. They have no third-party dependencies.
LIVE_FILES = ["__init__", "tac", "interpreter", "frontend", "code_optimizer"]


def live_sources() -> dict:
    """Source text of the eco modules shipped inside the page for the Code Optimization tab."""
    out = {}
    for name in LIVE_FILES:
        with open(os.path.join(HERE, name + ".py"), encoding="utf-8") as f:
            out[name + ".py"] = f.read().replace("\r\n", "\n")
    return out


def build_dashboard_data(programs, results, runtimes: Dict[str, dict], aggregate: dict, meta: dict) -> dict:
    """Collect everything the dashboard shows into one JSON-serialisable dict."""
    items: List[dict] = []
    for program, result in zip(programs, results):
        rep = program_report(program, result, runtimes.get(program.name))
        best = result.best_individual
        front_ids = {id(ind) for ind in result.final_pareto_front}
        base = program.baseline_fitness
        items.append({
            "name": program.name,
            "kernel": getattr(program, "kernel", "synthetic"),
            "spelling": getattr(program, "spelling", "plain"),
            "source": program.source_code,
            "tac_before": tac_to_text(program.tac).splitlines(),
            "tac_after": tac_to_text(best.tac).splitlines(),
            "genome": list(best.genome),
            "baseline": base.as_dict(),
            "best": best.fitness.as_dict(),
            "best_cost": best.cost,
            "improvement_pct": rep["improvement_pct_exec_time"],
            "generations_to_convergence": result.generations_to_convergence,
            "hist_best": [round(r.best_cost, 5) for r in result.history],
            "hist_avg": [round(r.avg_cost, 5) for r in result.history],
            "front_size": len(result.final_pareto_front),
            "distinct_points": len(rep["pareto_front"]),
            "pareto": [{"genome": e["genome"], "fitness": e["fitness"], "label": e["tradeoff"],
                        "tied": e["tied_genome_count"]} for e in rep["pareto_front"]],
            "population": [[ind.fitness.exec_time, ind.fitness.instr_count, ind.fitness.code_size,
                            ind.fitness.arith_ops, 1 if id(ind) in front_ids else 0, 1 if ind.valid else 0]
                           for ind in result.final_population],
            "runtime": runtimes.get(program.name),
            "verdict": rep["verdict"],
            "num_tests": len(program.test_input_sets),
        })
    return {"meta": meta, "aggregate": aggregate, "programs": items}


def write_dashboard(data: dict, out_path: str) -> str:
    """Embed ``data`` in the HTML template and write the dashboard file."""
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    if PLACEHOLDER not in template:
        raise ValueError("dashboard template is missing its data placeholder")
    # the Code Optimization tab opens with a precomputed example so it works before any engine is loaded
    data = dict(data)
    data.setdefault("code_optimization", optimize_program(DEFAULT_SOURCE))
    if LIVE_PLACEHOLDER not in template:
        raise ValueError("dashboard template is missing its live-sources placeholder")
    safe = lambda obj: json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")  # never close the script tag early
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    html = template.replace(PLACEHOLDER, safe(data), 1).replace(LIVE_PLACEHOLDER, safe(live_sources()), 1)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
