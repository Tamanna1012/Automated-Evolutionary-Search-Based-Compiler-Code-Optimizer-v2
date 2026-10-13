# Flow analysis in the compiler optimizer

This note explains how **control-flow analysis** and **global data-flow analysis** are used in this project. It is written for a reader who knows what a variable and an assignment are, but has not studied compilers before. The same text appears in the dashboard's *Flow Analysis* tab.

The optimizer works on a small language: one Python function with integers, `+ - * //`, `if`/`else` and `while` loops whose conditions are a single comparison (`< <= > >= == !=`). It turns the function into *three-address code* (TAC): a flat list where each line does one operation, such as `t1 = n < k`.

## 1. Control-flow analysis

### Basic blocks

Straight-line code runs from top to bottom. A **basic block** is a run of statements that always runs together: it is entered only at its first line and left only at its last line.

A new block starts at:

- the first statement of the function,
- every **label** (a place some jump can land on),
- the statement right after a `goto` or `branch`.

A block ends at a `goto`, a `branch`, or just before the next label.

### The control-flow graph

The **control-flow graph (CFG)** has one node per basic block and one edge for each way control can move from one block to another. A `branch` gives two edges, labelled **true** and **false**. A block that does not end in a jump falls through to the block below it.

### Worked example

We use this function throughout the note:

```python
def example(n, k):
    x = 2 * 3
    if n < k:
        y = x + n
    else:
        y = x + n
    z = x + n
    unused = 99
    return y + z
```

Its CFG has four blocks:

```
   B0  n = input(n), k = input(k), x = 2 * 3, t1 = n < k
       if t1 goto B1 else B2
          true /                  \ false
               v                   v
   B1  y = x + n              B2  y = x + n
       goto B3                    (falls through)
               \                   /
                v                 v
   B3  z = x + n, unused = 99, t2 = y + z, print t2
```

### Dominators, back edges and loops

A block **dominates** another if every path from the entry to the second block passes through the first. Dominators are the classic way to find loops. This project does **not** compute dominator sets. It finds loops through their **back edges**: an edge that goes to a block that appears earlier in the program. A `while` loop always creates one, because its body jumps back to the loop's test. For the analyses below, a back edge is enough, because the data-flow equations are solved by repetition until nothing changes, which works on any graph.

### Unreachable code

A block is **unreachable** when no path from the entry leads to it. For example, in `if 1 < 2: ... else: ...` the `else` arm can never run once the condition is known. The optimizer finds unreachable blocks by a simple search from the entry and deletes them.

## 2. Global data-flow analysis

Local optimization looks at one block at a time. Global data-flow analysis answers questions that cross block boundaries, such as "which assignments can reach this line?". Each question has the same four parts:

- **direction**: forward (follows the program) or backward (against it),
- **gen** and **kill** sets: what a block adds and removes,
- a **meet** operator: how the information from several incoming paths is combined,
- an **iterative algorithm**: keep updating every block's IN and OUT sets until nothing changes.

For a block `b` the equations are `IN[b] = meet over predecessors of OUT[p]` and `OUT[b] = gen[b] ∪ (IN[b] − kill[b])` (forward), with the directions swapped for liveness.

| Analysis | Direction | Question it answers | Meet | Gen | Kill | Used for |
|---|---|---|---|---|---|---|
| Reaching definitions | forward | which assignments may be the last one to a variable | union | the assignments in the block | other assignments to the same variable | global constant propagation |
| Live variables | backward | which variables may still be read later | union | variables read before being written in the block | variables written in the block | global dead code elimination |
| Available expressions | forward | which expressions are already computed, and in which variable, on **every** path | intersection | `a op b` computed in the block, if its variable is not redefined later | any expression that uses or produces a redefined variable | global common subexpression elimination |

### The iterative algorithm

1. Start with empty sets for every block, except that available expressions start from "everything" (the intersection needs a full set to start from), and the entry block starts empty.
2. Visit the blocks in order, computing IN from the neighbours, then OUT from IN.
3. Repeat step 2 until no set changes. Because the sets only grow (or only shrink, for available expressions), this always stops.

### IN and OUT sets for the worked example

These are the sets at the fixed point, for the **original** program.

| Block | Live IN | Live OUT | Available IN | Available OUT |
|---|---|---|---|---|
| B0 | — | n, x | — | 2 * 3 in x |
| B1 | n, x | n, x, y | 2 * 3 in x | 2 * 3 in x, n + x in y |
| B2 | n, x | n, x, y | 2 * 3 in x | 2 * 3 in x, n + x in y |
| B3 | n, x, y | — | 2 * 3 in x, n + x in y | 2 * 3 in x, n + x in y, n + x in z, y + z in t2 |

For reaching definitions, B3's IN contains **two** definitions of `y`: one from B1 and one from B2. Both reach the merge point, because either arm may have run. This is why a plain straight-line pass cannot know which `y` is live at the merge.

## 3. How each analysis enables an optimization

- **Global constant propagation** (reaching definitions). If every definition of `x` that reaches a use assigns the same constant, the use can be replaced by that constant. In the example, `x = 6` is the only definition of `x` reaching B1 and B2, so `y = x + n` becomes `y = 6 + n`.
- **Global common subexpression elimination** (available expressions). If `x + n` is available on every path into a point, and the variable that holds it has not changed, the new computation is replaced by a copy. In the example, both arms leave `x + n` in `y`, so `z = x + n` becomes `z = y`.
- **Global dead code elimination** (liveness). An assignment whose variable is not live afterwards has no effect on the output. After the rewrites above, `x = 6` and `unused = 99` are dead and are removed.
- **Constant folding and unreachable-block removal** (control-flow graph). A branch on a known condition becomes a plain jump, and the arm it skips becomes unreachable and is removed.

## 4. Local versus global

Running the straight-line passes **inside each block only** leaves all 12 statements of the example in place. Constant propagation in B1 cannot see that `x` is 6, because `x` was assigned in B0, and local passes reset their knowledge at every block boundary. The same passes applied to the whole program, with reaching definitions and liveness, remove two statements (`x = 6` and `unused = 99`) and rewrite three more (both `y` assignments and `z`): 10 statements remain.

The dashboard shows both counts side by side, so the difference is visible for any program you type in.

## 5. Limits and correctness

- **Supported now**: `and`, `or` and `not` in conditions, and calls to helper functions defined earlier in the same source (inlined at each call; recursion is rejected).
- **Arrays**: local arrays created with `[..]` and indexed by integer literals only (each element becomes a variable). Dynamic indexing, array parameters and pointers are rejected with a message, because they need a memory model that this project does not have.
- **Not handled**: aliasing (two names for one value), `return` inside a loop or branch, and recursion. Each is rejected with a message, never mis-compiled.
- **Symbolic proof**: every block gets a path condition, and a join point becomes a chain of if-then-else terms over the incoming path conditions. Two programs are proved equal only when their output forms are identical, so the method is sound but incomplete: an optimization that changes which branches exist can end in *not proved* even when the program is correct.
- **Loops (bounded proof)**: each loop is unrolled a fixed number of times (8) in both programs, and the unrolled programs are compared the same way. A proof therefore holds for every input whose loops finish within 8 iterations, and the dashboard says so. Longer runs are covered only by the compile-time tests.
- Every program is also checked by running the original and optimized code on test inputs (the compile-time check), so a failure is always reported as NOT equivalent with the first mismatching input.
- **Loops and termination**: the interpreter stops a run after a fixed number of steps and reports that the program did not finish. A loop whose trip count depends on the input is therefore tested on each input, not proved.
- **Integer division**: Python's `//` floors and the generated code truncates toward zero, so they disagree for negative operands. Negative boundary inputs are skipped for programs that divide.
- **Test coverage**: the compile-time check uses five random inputs plus boundary inputs (0, 1, a large value, and -1 where allowed). Passing these is evidence, not a proof, and the dashboard says so.
