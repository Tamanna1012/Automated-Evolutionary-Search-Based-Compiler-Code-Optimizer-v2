"""Front end: compiles a Python function into the project's TAC.

Uses only the standard ``ast`` module and supports:

  * one ``def`` whose parameters become ``input`` instructions
  * assignments (``x = expr``, ``x += expr`` ...) to plain names
  * expressions made of names, integer literals, ``+ - * //`` and unary ``-``
  * ``return expr`` or ``return a, b, ...`` (each value becomes an ``output``)
  * ``if``/``else`` and ``while`` whose condition is a single comparison
    (``< <= > >= == !=``); these need control flow, see below

``//`` maps to the TAC ``/`` operator. Python's ``//`` floors while the TAC
interpreter truncates toward zero, so they agree whenever the quotient's
operands are non-negative; the kernels that divide are therefore run on
positive input domains, and every program is checked against the original
Python source before it enters the dataset (see ``kernels.py``).

Straight-line code is compiled in single-assignment form (each TAC destination is
defined once) and literals are pooled, as an optimizing front end would do.

Control flow is compiled without renaming: a Python variable keeps one TAC name, so
the two arms of an ``if`` meet in the same variable. Literals are not pooled in that
mode, because a pooled constant defined in one block would not exist on the path
that reaches another block. New instructions:

    ("label", name)                 a jump target
    ("jump", name)                  unconditional transfer
    ("branch", cond, then, else)    transfer to ``then`` if cond is non-zero

A comparison is an ordinary ``bin`` instruction whose result is 0 or 1.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

from .tac import TAC

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "/"}
_CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}


class UnsupportedSyntax(ValueError):
    """Raised when the source uses a construct the front end does not support."""


@dataclass
class CompiledFunction:
    name: str
    arg_names: List[str]
    tac: TAC


class _Compiler:
    def __init__(self, control: bool = False):
        self.control = control
        self.tac: TAC = []
        self.env: Dict[str, str] = {}   # python variable -> current TAC variable
        self.defined = set()            # every TAC destination used so far
        self._temp = 0
        self._const = 0
        self._label = 0
        self._pool: Dict[int, str] = {}  # literal value -> the const variable holding it

    # -- naming -----------------------------------------------------------
    def _fresh_temp(self) -> str:
        self._temp += 1
        return f"_e{self._temp}"

    def _fresh_const(self) -> str:
        self._const += 1
        return f"_k{self._const}"

    def _fresh_label(self, hint: str) -> str:
        self._label += 1
        return f"L{self._label}_{hint}"

    def _dest_for(self, py_name: str) -> str:
        """First assignment keeps the source name; later ones get a suffix (SSA).

        With control flow the name is kept, so both arms of a branch write one variable.
        """
        if self.control:
            self.defined.add(py_name)
            return py_name
        if py_name not in self.defined:
            name = py_name
        else:
            n = 1
            while f"{py_name}_{n}" in self.defined:
                n += 1
            name = f"{py_name}_{n}"
        self.defined.add(name)
        return name

    def _define(self, dest: str):
        self.defined.add(dest)

    # -- expressions ------------------------------------------------------
    def expr(self, node: ast.AST, dest: Optional[str] = None) -> str:
        """Emit code for ``node`` and return the TAC variable holding its value.

        When ``dest`` is given the final instruction defines exactly that name.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise UnsupportedSyntax(f"only integer literals are supported, got {node.value!r}")
            return self._emit_const(node.value, dest)

        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise UnsupportedSyntax(f"use of undefined variable '{node.id}'")
            src = self.env[node.id]
            if dest is None:
                return src  # a bare read needs no instruction
            self.tac.append(("copy", dest, src))
            return dest

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return self.expr(node.operand, dest)
            if isinstance(node.op, ast.USub):
                if isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
                    return self._emit_const(-node.operand.value, dest)
                operand = self.expr(node.operand)
                zero = self._emit_const(0, None)
                out = dest or self._fresh_temp()
                self.tac.append(("bin", out, "-", zero, operand))
                return out
            raise UnsupportedSyntax(f"unsupported unary operator {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise UnsupportedSyntax(
                    f"unsupported operator {type(node.op).__name__} (use + - * //)")
            left = self.expr(node.left)
            right = self.expr(node.right)
            out = dest or self._fresh_temp()
            self.tac.append(("bin", out, op, left, right))
            return out

        raise UnsupportedSyntax(f"unsupported expression {type(node).__name__}")

    def _condition(self, node: ast.AST) -> str:
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS):
            raise UnsupportedSyntax(
                f"a condition must be a single comparison such as 'a < b' (got '{ast.unparse(node)}')")
        left = self.expr(node.left)
        right = self.expr(node.comparators[0])
        out = self._fresh_temp()
        self.tac.append(("bin", out, _CMPOPS[type(node.ops[0])], left, right))
        return out

    def _emit_const(self, value: int, dest: Optional[str]) -> str:
        value = int(value)
        if dest is None and not self.control and value in self._pool:
            return self._pool[value]
        name = dest or self._fresh_const()
        self.tac.append(("const", name, value))
        if not self.control:
            self._pool.setdefault(value, name)
        return name

    # -- statements -------------------------------------------------------
    def assign(self, py_name: str, value: ast.AST):
        # evaluate the right-hand side first (it may read the old binding)
        dest = self._dest_for(py_name)
        result = self.expr(value, dest)
        assert result == dest
        self.env[py_name] = dest

    def statement(self, stmt: ast.stmt, in_block: bool = False):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise UnsupportedSyntax("only simple 'name = expr' assignments are supported")
            self.assign(stmt.targets[0].id, stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise UnsupportedSyntax("only simple 'name op= expr' is supported")
            fake = ast.BinOp(left=ast.Name(id=stmt.target.id, ctx=ast.Load()), op=stmt.op, right=stmt.value)
            self.assign(stmt.target.id, fake)
        elif isinstance(stmt, ast.Return):
            if in_block:
                raise UnsupportedSyntax("'return' inside if/while is not supported; assign the result and return it at the end")
            if stmt.value is None:
                raise UnsupportedSyntax("a bare 'return' is not supported")
            values = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
            for v in values:
                self.tac.append(("output", self.expr(v)))
        elif isinstance(stmt, ast.If):
            self._if(stmt)
        elif isinstance(stmt, ast.While):
            self._while(stmt)
        elif isinstance(stmt, ast.Pass):
            return
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return  # docstring
        else:
            raise UnsupportedSyntax(f"unsupported statement {type(stmt).__name__}")

    def _block(self, stmts: List[ast.stmt]):
        for s in stmts:
            self.statement(s, in_block=True)

    def _if(self, stmt: ast.If):
        cond = self._condition(stmt.test)
        then_l, end_l = self._fresh_label("then"), self._fresh_label("end")
        else_l = self._fresh_label("else") if stmt.orelse else end_l
        self.tac.append(("branch", cond, then_l, else_l))
        before = dict(self.env)
        self.tac.append(("label", then_l))
        self._block(stmt.body)
        then_env = self.env
        if stmt.orelse:
            self.tac.append(("jump", end_l))
            self.tac.append(("label", else_l))
            self.env = dict(before)
            self._block(stmt.orelse)
            else_env = self.env
        else:
            else_env = before
        self.tac.append(("label", end_l))
        # a variable survives the if only when both arms leave it bound to the same name
        self.env = {k: v for k, v in then_env.items() if else_env.get(k) == v}

    def _while(self, stmt: ast.While):
        if stmt.orelse:
            raise UnsupportedSyntax("'while ... else' is not supported")
        head, body_l, end_l = self._fresh_label("head"), self._fresh_label("body"), self._fresh_label("exit")
        self.tac.append(("label", head))
        cond = self._condition(stmt.test)
        self.tac.append(("branch", cond, body_l, end_l))
        before = dict(self.env)
        self.tac.append(("label", body_l))
        self._block(stmt.body)
        self.tac.append(("jump", head))
        self.tac.append(("label", end_l))
        # variables first bound inside the loop are not visible after it
        self.env = before


def has_control_flow(source_or_tac) -> bool:
    if isinstance(source_or_tac, str):
        return any(isinstance(n, (ast.If, ast.While)) for n in ast.walk(ast.parse(source_or_tac)))
    return any(i[0] in ("label", "jump", "branch") for i in source_or_tac)


def compile_function(source: str) -> CompiledFunction:
    """Compile the single function defined in ``source`` to TAC."""
    module = ast.parse(source)
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or len(module.body) != 1:
        raise UnsupportedSyntax("source must contain exactly one function definition")
    fn = funcs[0]
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
        raise UnsupportedSyntax("only plain positional parameters are supported")
    for i, stmt in enumerate(fn.body):
        if isinstance(stmt, ast.Return) and i != len(fn.body) - 1:
            raise UnsupportedSyntax("'return' must be the last statement of the function")

    control = has_control_flow(source)
    comp = _Compiler(control)
    arg_names = [a.arg for a in fn.args.args]
    for name in arg_names:
        dest = comp._dest_for(name)
        comp.tac.append(("input", dest, name))
        comp.env[name] = dest
    for stmt in fn.body:
        comp.statement(stmt)
    if not any(i[0] == "output" for i in comp.tac):
        raise UnsupportedSyntax("function must return at least one value")
    return CompiledFunction(name=fn.name, arg_names=arg_names, tac=comp.tac)


def run_source(source: str, func_name: str, inputs: Dict[str, int]) -> List[int]:
    """Ground truth: execute the ORIGINAL Python source and return its outputs."""
    namespace: dict = {}
    exec(compile(source, f"<kernel {func_name}>", "exec"), {"__builtins__": {}}, namespace)  # noqa: S102
    result = namespace[func_name](**inputs)
    if isinstance(result, tuple):
        return [int(v) for v in result]
    return [int(result)]
