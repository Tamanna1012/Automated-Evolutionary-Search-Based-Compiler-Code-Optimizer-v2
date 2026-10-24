"""Control-flow graph, iterative data-flow analysis and global optimizations.

Works on the textbook statement list used by ``code_optimizer`` (any object with
``op``, ``dest``, ``a``, ``b``, ``origin``, ``copy_with`` and ``text``). Control flow is
a list of ``label``, ``goto`` (``a`` = target label) and ``branch`` (``a`` = condition,
``b`` = (then_label, else_label)) statements.

Basic blocks are maximal runs with one entry (a label, or the statement after a jump)
and one exit (a goto or branch, or the end). Three analyses run over the blocks until
nothing changes:

  * liveness (backward, union):           which variables may still be read
  * reaching definitions (forward, union): which assignments may reach a point
  * available expressions (forward, intersection): which expressions are already
    computed, and in which variable, on every path into a point

The global passes use them: constant propagation (reaching definitions), common
subexpression elimination (available expressions), dead code elimination (liveness),
constant folding (including branches on constant conditions) and removal of blocks
no path reaches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from .interpreter import _safe_div, compare

ARITH = ("+", "-", "*", "/")
COMPARE = ("<", "<=", ">", ">=", "==", "!=")
CONTROL = ("label", "goto", "branch")
COMMUTATIVE = ("+", "*")
MAX_ROUNDS = 12
MAX_STEPS = 200_000

GLOBAL_TECHNIQUES = [
    "Global Constant Propagation", "Constant Folding", "Global Common Subexpression Elimination",
    "Global Dead Code Elimination", "Unreachable Block Removal",
]

Site = Tuple[int, int]  # (block id, statement index inside the block)


def has_control(stmts) -> bool:
    return any(s.op in CONTROL for s in stmts)


def defined(s) -> Optional[str]:
    return s.dest if s.op in ARITH + COMPARE + ("=", "input") else None


def uses(s) -> List[str]:
    if s.op in ("input", "label", "goto"):
        return []
    if s.op in ("branch", "print"):
        return [s.a] if isinstance(s.a, str) else []
    return [x for x in (s.a, s.b) if isinstance(x, str)]


def expr_key(s) -> tuple:
    ops = (s.a, s.b)
    if s.op in COMMUTATIVE:
        ops = tuple(sorted(ops, key=lambda x: (isinstance(x, str), str(x))))
    return (s.op,) + ops


@dataclass
class Block:
    id: int
    stmts: list
    succ: List[Tuple[int, str]]  # (block id, "" | "true" | "false")
    label: Optional[str]


def build_cfg(stmts) -> List[Block]:
    n = len(stmts)
    leaders = {0}
    for i, s in enumerate(stmts):
        if s.op == "label":
            leaders.add(i)
        if s.op in ("goto", "branch") and i + 1 < n:
            leaders.add(i + 1)
    starts = sorted(leaders)
    blocks = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        body = list(stmts[start:end])
        label = body[0].a if body and body[0].op == "label" else None
        blocks.append(Block(id=k, stmts=body, succ=[], label=label))
    by_label = {b.label: b.id for b in blocks if b.label is not None}
    for b in blocks:
        last = b.stmts[-1] if b.stmts else None
        if last is not None and last.op == "goto":
            b.succ = [(by_label[last.a], "")]
        elif last is not None and last.op == "branch":
            b.succ = [(by_label[last.b[0]], "true"), (by_label[last.b[1]], "false")]
        elif b.id + 1 < len(blocks):
            b.succ = [(b.id + 1, "")]
    return blocks


def _preds(blocks: List[Block]) -> Dict[int, List[int]]:
    preds: Dict[int, List[int]] = {b.id: [] for b in blocks}
    for b in blocks:
        for t, _ in b.succ:
            preds[t].append(b.id)
    return preds


def reachable(blocks: List[Block]) -> set:
    seen, stack = set(), [0] if blocks else []
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        stack.extend(t for t, _ in blocks[b].succ)
    return seen


def liveness(blocks: List[Block]) -> Tuple[Dict[int, FrozenSet[str]], Dict[int, FrozenSet[str]]]:
    live_in = {b.id: frozenset() for b in blocks}
    live_out = {b.id: frozenset() for b in blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            out = frozenset().union(*[live_in[t] for t, _ in b.succ]) if b.succ else frozenset()
            live = set(out)
            for s in reversed(b.stmts):
                d = defined(s)
                if d is not None:
                    live.discard(d)
                live.update(uses(s))
            new_in = frozenset(live)
            if out != live_out[b.id] or new_in != live_in[b.id]:
                live_out[b.id], live_in[b.id] = out, new_in
                changed = True
    return live_in, live_out


def _reach_transfer(state: FrozenSet[Tuple[str, Site]], bid: int, stmts) -> FrozenSet[Tuple[str, Site]]:
    cur = set(state)
    for idx, s in enumerate(stmts):
        d = defined(s)
        if d is not None:
            cur = {p for p in cur if p[0] != d}
            cur.add((d, (bid, idx)))
    return frozenset(cur)


def reaching_definitions(blocks: List[Block]):
    preds = _preds(blocks)
    rin = {b.id: frozenset() for b in blocks}
    rout = {b.id: frozenset() for b in blocks}
    changed = True
    while changed:
        changed = False
        for b in blocks:
            new_in = frozenset().union(*[rout[p] for p in preds[b.id]]) if preds[b.id] else frozenset()
            new_out = _reach_transfer(new_in, b.id, b.stmts)
            if new_in != rin[b.id] or new_out != rout[b.id]:
                rin[b.id], rout[b.id] = new_in, new_out
                changed = True
    return rin, rout


def _avail_transfer(state: FrozenSet[Tuple[tuple, str]], s) -> FrozenSet[Tuple[tuple, str]]:
    d = defined(s)
    cur = set(state)
    if d is not None:
        cur = {(k, v) for (k, v) in cur if v != d and d not in k[1:]}
        if s.op in ARITH and d not in uses(s):
            cur.add((expr_key(s), d))
    return frozenset(cur)


def available_expressions(blocks: List[Block]):
    preds = _preds(blocks)
    universe = frozenset((expr_key(s), defined(s)) for b in blocks for s in b.stmts if s.op in ARITH)
    ain = {b.id: (frozenset() if b.id == 0 else universe) for b in blocks}
    aout = {b.id: universe for b in blocks}
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b.id == 0:
                new_in = frozenset()
            elif preds[b.id]:
                new_in = frozenset.intersection(*[aout[p] for p in preds[b.id]])
            else:
                new_in = frozenset()
            new_out = new_in
            for s in b.stmts:
                new_out = _avail_transfer(new_out, s)
            if new_in != ain[b.id] or new_out != aout[b.id]:
                ain[b.id], aout[b.id] = new_in, new_out
                changed = True
    return ain, aout


def _rule(technique, before, after, reason, rule, precondition, merged_into=()) -> dict:
    return {"technique": technique, "before": before, "after": after,
            "reason": reason, "rule": rule, "precondition": precondition, "merged_into": list(merged_into)}


def _unreachable(stmts):
    blocks = build_cfg(stmts)
    live = reachable(blocks)
    records, kept = [], []
    for b in blocks:
        if b.id in live:
            kept.extend(b.stmts)
            continue
        for s in b.stmts:
            if s.op != "label":
                records.append(_rule("Unreachable Block Removal", s, None,
                                     f"block {b.id} cannot be reached from the entry of the function",
                                     "code that no path from the entry reaches has no effect on the output",
                                     "no reachable goto or branch leads to this block"))
    return kept, records


def _const_propagation(stmts):
    blocks = build_cfg(stmts)
    rin, _ = reaching_definitions(blocks)
    by_site = {(b.id, i): s for b in blocks for i, s in enumerate(b.stmts)}
    records, out = [], []
    for b in blocks:
        state = set(rin[b.id])
        for i, s in enumerate(b.stmts):
            new = s
            changes = []
            for field in ("a", "b"):
                x = getattr(s, field)
                if not isinstance(x, str) or s.op not in ARITH + COMPARE + ("=", "print", "branch"):
                    continue
                if field == "b" and s.op in ("=", "print", "branch"):
                    continue
                sites = [site for (v, site) in state if v == x]
                values = set()
                for site in sites:
                    d = by_site[site]
                    if d.op == "=" and isinstance(d.a, int):
                        values.add(d.a)
                    else:
                        values = None
                        break
                if values and len(values) == 1:
                    (c,) = values
                    new = new.copy_with(**{field: c})
                    changes.append((x, c))
            if changes:
                reason = ", ".join(f"{x} is always {c} here" for x, c in changes)
                rule = "; ".join(f"if every definition of {x} that reaches this point assigns {c}, replace {x} by {c}" for x, c in changes)
                pre = "; ".join(f"all reaching definitions of {x} are literal assignments of {c}" for x, c in changes)
                records.append(_rule("Global Constant Propagation", s, new, reason, rule, pre))
            out.append(new)
            d = defined(s)
            if d is not None:
                state = {p for p in state if p[0] != d}
                state.add((d, (b.id, i)))
    return out, records


def _fold(stmts):
    out, records = [], []
    for s in stmts:
        new = s
        if s.op == "branch" and isinstance(s.a, int):
            target = s.b[0] if s.a else s.b[1]
            new = s.copy_with(op="goto", a=target, b=None)
            records.append(_rule("Constant Folding", s, new,
                                 f"condition is the constant {s.a}, so the branch always goes one way",
                                 "a branch on a known condition is an unconditional jump",
                                 "the condition is a literal"))
        elif s.op in ARITH + COMPARE and isinstance(s.a, int) and isinstance(s.b, int):
            value = None
            if s.op in COMPARE:
                value = compare(s.op, s.a, s.b)
            elif s.op == "+":
                value = s.a + s.b
            elif s.op == "-":
                value = s.a - s.b
            elif s.op == "*":
                value = s.a * s.b
            elif s.b != 0 and s.a % s.b == 0:
                value = s.a // s.b
            if value is not None:
                new = s.copy_with(op="=", a=value, b=None)
                records.append(_rule("Constant Folding", s, new,
                                     f"evaluated constant expression {s.a} {s.op} {s.b}",
                                     f"{s.a} {s.op} {s.b} = {value}",
                                     "both operands are integer literals; a division is folded only when exact"))
        out.append(new)
    return out, records


def _cse(stmts):
    blocks = build_cfg(stmts)
    ain, _ = available_expressions(blocks)
    # original lines that compute an expression into a variable: a reused value may come from several
    # (one per branch), and those are the lines a rewrite is merged into
    computed_at: Dict[tuple, set] = {}
    for b in blocks:
        for s in b.stmts:
            if s.op in ARITH and defined(s) is not None:
                computed_at.setdefault((expr_key(s), defined(s)), set()).add(s.origin)
    records, out = [], []
    for b in blocks:
        state = ain[b.id]
        for s in b.stmts:
            new = s
            d = defined(s)
            if s.op in ARITH and d is not None:
                key = expr_key(s)
                holders = sorted(v for (k, v) in state if k == key and v != d)
                if holders:
                    v = holders[0]
                    new = s.copy_with(op="=", a=v, b=None)
                    records.append(_rule("Global Common Subexpression Elimination", s, new,
                                         f"{s.a} {s.op} {s.b} is already held in {v} on every path here",
                                         f"reuse {v} instead of computing {s.a} {s.op} {s.b} again",
                                         f"{v} holds the value of {s.a} {s.op} {s.b} on every path into this point and is not reassigned",
                                         merged_into=computed_at.get((key, v), ())))
            out.append(new)
            state = _avail_transfer(state, new)
    return out, records


def _dce(stmts):
    blocks = build_cfg(stmts)
    _, lout = liveness(blocks)
    records, dropped = [], set()
    for b in blocks:
        live = set(lout[b.id])
        for i in range(len(b.stmts) - 1, -1, -1):
            s = b.stmts[i]
            d = defined(s)
            if d is not None and s.op != "input" and d not in live:
                records.append(_rule("Global Dead Code Elimination", s, None,
                                     f"{d} is never read after this statement on any path",
                                     "a definition whose value is never read on any path has no effect on the output",
                                     f"{d} is not live after this statement"))
                dropped.add((b.id, i))
                continue
            if d is not None:
                live.discard(d)
            live.update(uses(s))
    out = [s for b in blocks for i, s in enumerate(b.stmts) if (b.id, i) not in dropped]
    return out, records


PASSES = [
    ("Unreachable Block Removal", _unreachable),
    ("Global Constant Propagation", _const_propagation),
    ("Constant Folding", _fold),
    ("Global Common Subexpression Elimination", _cse),
    ("Global Dead Code Elimination", _dce),
]


def global_optimize(stmts) -> Tuple[list, List[dict], List[dict]]:
    """Run the global passes until a full round changes nothing.

    Returns the new statements, the change records and a per-pass trace whose
    statement counts include labels.
    """
    current = [s.copy_with() for s in stmts]
    records: List[dict] = []
    trace: List[dict] = []
    for rnd in range(1, MAX_ROUNDS + 1):
        changed = False
        for name, fn in PASSES:
            before = len(current)
            current, events = fn(current)
            for e in events:
                e["round"] = rnd
            trace.append({"round": rnd, "technique": name, "statements_before": before,
                          "statements_after": len(current), "events": len(events)})
            if events:
                changed = True
                records.extend(events)
        if not changed:
            break
    return current, records, trace


def execute(stmts, inputs, max_steps: int = MAX_STEPS) -> List[int]:
    """Run a statement list (straight-line or with control flow) and return its outputs."""
    labels = {s.a: i for i, s in enumerate(stmts) if s.op == "label"}
    env: Dict[str, int] = {}
    out: List[int] = []
    pc, steps = 0, 0

    def val(x):
        return x if isinstance(x, int) else env[x]

    while pc < len(stmts):
        steps += 1
        if steps > max_steps:
            raise RuntimeError("step limit exceeded (the program may loop forever)")
        s = stmts[pc]
        pc += 1
        if s.op == "input":
            env[s.dest] = int(inputs[s.a])
        elif s.op == "=":
            env[s.dest] = val(s.a)
        elif s.op == "print":
            out.append(val(s.a))
        elif s.op == "label":
            continue
        elif s.op == "goto":
            pc = labels[s.a]
        elif s.op == "branch":
            pc = labels[s.b[0]] if val(s.a) else labels[s.b[1]]
        elif s.op in COMPARE:
            env[s.dest] = compare(s.op, val(s.a), val(s.b))
        else:
            x, y = val(s.a), val(s.b)
            env[s.dest] = x + y if s.op == "+" else x - y if s.op == "-" else x * y if s.op == "*" else _safe_div(x, y)
    return out


def dominators(blocks: List[Block]) -> Dict[int, List[int]]:
    """Dominator sets of the reachable blocks: b is in dom[n] when every path from the entry to n passes b."""
    reach = reachable(blocks)
    preds = _preds(blocks)
    ids = sorted(reach)
    dom = {b: set(ids) for b in ids}
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for b in ids[1:]:
            ps = [p for p in preds[b] if p in reach]
            new = (set.intersection(*[dom[p] for p in ps]) if ps else set(ids)) | {b}
            if new != dom[b]:
                dom[b] = new
                changed = True
    return {b: sorted(dom[b]) for b in ids}


def natural_loops(blocks: List[Block]) -> List[dict]:
    """Each back edge t -> h (h dominates t) gives a loop: h plus every block that reaches t without passing h."""
    reach = reachable(blocks)
    preds = _preds(blocks)
    dom = dominators(blocks)
    loops = []
    for t in blocks:
        if t.id not in reach:
            continue
        for h, _ in t.succ:
            if h not in dom.get(t.id, []):
                continue
            body, stack = {h}, [t.id]
            while stack:
                n = stack.pop()
                if n in body:
                    continue
                body.add(n)
                stack.extend(p for p in preds[n] if p in reach)
            loops.append({"header": h, "back_edge": t.id, "body": sorted(body)})
    return loops


def flow_report(stmts) -> dict:
    """Blocks with their statements, edges, and IN/OUT sets for every analysis."""
    blocks = build_cfg(stmts)
    reach = reachable(blocks)
    lin, lout = liveness(blocks)
    rin, rout = reaching_definitions(blocks)
    ain, aout = available_expressions(blocks)

    def site_text(site):
        return f"B{site[0]}: {blocks[site[0]].stmts[site[1]].text()}"

    def key_text(k):
        return f"{k[1]} {k[0]} {k[2]}"

    rows = []
    for b in blocks:
        rows.append({
            "id": b.id,
            "label": b.label,
            "stmts": [s.text() for s in b.stmts],
            "succ": [{"to": t, "kind": kind} for t, kind in b.succ],
            "reachable": b.id in reach,
            "reaching_in": sorted(f"{v} <- {site_text(site)}" for v, site in rin[b.id]),
            "reaching_out": sorted(f"{v} <- {site_text(site)}" for v, site in rout[b.id]),
            "live_in": sorted(lin[b.id]),
            "live_out": sorted(lout[b.id]),
            "avail_in": sorted(f"{key_text(k)} in {v}" for k, v in ain[b.id]),
            "avail_out": sorted(f"{key_text(k)} in {v}" for k, v in aout[b.id]),
        })
    dom = dominators(blocks)
    for row in rows:
        row["dominators"] = dom.get(row["id"], [])
    return {"blocks": rows, "control_flow": has_control(stmts), "loops": natural_loops(blocks)}


def cfg_graph(stmts) -> dict:
    """Drawable control-flow graph of any statement list, straight-line code included.

    Every block lists its instructions (the label that starts it is the block's title, not an
    instruction) with the statement's ``origin`` so a drawing can match a block of the optimized
    program to the original. An edge is a back edge when its target dominates its source, the same
    test ``natural_loops`` uses. Straight-line code is one block that is both entry and exit.
    """
    blocks = build_cfg(stmts)
    reach = reachable(blocks)
    dom = dominators(blocks)
    rows = []
    for b in blocks:
        rows.append({
            "id": b.id,
            "label": b.label,
            "stmts": [{"text": s.text(), "origin": s.origin} for s in b.stmts if s.op != "label"],
            "succ": [{"to": t, "kind": kind, "back": t in dom.get(b.id, [])} for t, kind in b.succ],
            "reachable": b.id in reach,
            "entry": b.id == 0,
            "exit": b.id in reach and not b.succ,
        })
    return {"blocks": rows, "control_flow": has_control(stmts)}


def backward_jumps(stmts) -> List[Tuple[int, int]]:
    """(header index, jump index) for every goto that jumps back to an earlier label, i.e. every loop."""
    pos = {s.a: i for i, s in enumerate(stmts) if s.op == "label"}
    return [(pos[s.a], j) for j, s in enumerate(stmts) if s.op == "goto" and s.a in pos and pos[s.a] < j]


def unroll_loops(stmts, bound: int, max_size: int = 4000) -> Optional[list]:
    """Replace every loop by ``bound`` copies of its body.

    Copy i jumps to copy i+1; after the last copy control goes to the loop's exit. The
    result has no back edges, so it can be analysed symbolically. Both programs are
    unrolled the same way, so equal results mean equal behaviour whenever every loop
    finishes within ``bound`` iterations.
    """
    stmts = list(stmts)
    while True:
        jumps = backward_jumps(stmts)
        if not jumps:
            return stmts
        h, j = min(jumps, key=lambda r: r[1] - r[0])     # innermost loop first
        body = stmts[h:j + 1]
        header = stmts[h].a
        exit_label = next(s.b[1] for s in body if s.op == "branch")
        labels = {s.a for s in body if s.op == "label"}
        copies = []
        for i in range(bound):
            def name(x, i=i):
                return x if i == 0 or x not in labels else f"{x}__u{i}"
            for k, s in enumerate(body):
                if s.op == "label":
                    copies.append(s.copy_with(a=name(s.a)))
                elif s.op == "goto":
                    last = k == len(body) - 1
                    target = (f"{header}__u{i + 1}" if i + 1 < bound else exit_label) if last else name(s.a)
                    copies.append(s.copy_with(a=target))
                elif s.op == "branch":
                    copies.append(s.copy_with(b=(name(s.b[0]), name(s.b[1]))))
                else:
                    copies.append(s.copy_with())
        stmts = stmts[:h] + copies + stmts[j + 1:]
        if len(stmts) > max_size:
            return None
