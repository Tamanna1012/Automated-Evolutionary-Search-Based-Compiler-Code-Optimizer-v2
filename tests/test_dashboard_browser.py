"""Drives the generated dashboard in a real browser. Optional: skipped unless Playwright and a
Chromium are installed (``pip install playwright``; see README).

The Python engine normally downloads Pyodide from a CDN. Here that download is replaced by a stub that
returns what the real Python optimizer computes for each program, so everything after the click
(rendering, the clickable "Lines changed" panel, the control-flow graphs, the comparison) is the page's
own code running on real optimizer output. It does not test Pyodide itself.
"""
import glob
import json
import os

import pytest

from eco.code_optimizer import FLOW_EXAMPLE, SAMPLES, optimize_program

sync_api = pytest.importorskip("playwright.sync_api")

PAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "index.html")
LOOP_SRC = ("def loopy(n):\n    k = 2 * 3\n    s = 0\n    i = 0\n    while i < n:\n        s = s + k\n        i = i + 1\n"
            "    unused = 99\n    return s\n")
PROGRAMS = {"straight-line": SAMPLES["Mixed example (all six techniques)"][2], "if/else": FLOW_EXAMPLE,
            "while loop": LOOP_SRC, "nothing to optimize": SAMPLES["Already optimal (nothing to change)"][2]}
STUB = """window.loadPyodide = async () => ({
  FS: {mkdirTree(){}, writeFile(){}}, runPython(){},
  globals: {get(){ return (s) => window.__RESULTS[JSON.parse(s).source.trim()]; }}
});"""


@pytest.fixture(scope="module")
def browser():
    if not os.path.exists(PAGE) or "Comparison with other methods" not in open(PAGE, encoding="utf-8").read():
        pytest.skip("docs/index.html is missing or stale: run python -m eco.main first")
    with sync_api.sync_playwright() as pw:
        try:
            br = pw.chromium.launch(args=["--no-sandbox"])
        except Exception:  # noqa: BLE001 - fall back to a Chromium installed elsewhere
            found = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")
            if not found:
                pytest.skip("no Chromium available for Playwright")
            br = pw.chromium.launch(executable_path=found[0], args=["--no-sandbox"])
        yield br
        br.close()


@pytest.mark.parametrize("name", list(PROGRAMS))
def test_new_sections_work_in_a_real_browser(browser, name):
    src = PROGRAMS[name]
    res = optimize_program(src)
    results = {src.strip(): json.dumps(res)}
    page = browser.new_page(viewport={"width": 1280, "height": 1000})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/pyodide.js", lambda r: r.fulfill(body=STUB, content_type="application/javascript"))
    page.add_init_script("window.__RESULTS = " + json.dumps(results) + ";")
    page.goto("file://" + PAGE)
    page.fill("textarea[aria-label='Source code']", src)
    page.click("button.btn")
    page.wait_for_function("!document.querySelector('button.btn').disabled", timeout=20000)

    # clickable, keyboard-accessible "Lines changed": the count is the number of rows
    n = res["summary"]["lines_changed"]
    box = page.locator("button.box")
    assert box.inner_text().split("\n")[0].strip() == str(n)
    box.focus()
    page.keyboard.press("Enter")
    assert box.get_attribute("aria-expanded") == "true"
    panel = page.locator("#lines-panel")
    assert panel.locator("tbody tr").count() == n
    for tag in page.locator("button.chip[data-bit]").all():       # the technique tags filter the panel; none selected = all
        tech = tag.inner_text()
        tag.click()
        assert panel.locator("tbody tr").count() == sum(tech in x["techniques"] for x in res["line_changes"])
        tag.click()
    assert panel.locator("tbody tr").count() == n

    # the control-flow graph is drawn for every program, original and optimized
    cfg = page.locator("h2", has_text="Control flow graph").locator("xpath=..")
    assert cfg.locator("svg").count() == 2
    cfg.locator("button.chip", has_text="Optimized").click()
    assert cfg.locator("svg").count() == 1

    # the comparison table lists every method; the chart has one bar per method per metric
    mc = res["method_comparison"]
    card = page.locator("h2", has_text="Comparison with other methods").locator("xpath=..")
    assert card.locator("table tbody tr").count() == len(mc["methods"])
    assert card.locator("svg .col").count() == 4 * len(mc["methods"])
    assert page.locator("h2", has_text="Where ours wins").locator("xpath=..").locator("tbody tr").count() == 4
    assert errors == []
    page.close()
