"""Front end: compiles a straight-line Python function into the project's TAC.

Uses only the standard ``ast`` module and supports exactly what the kernel
library needs:

  * one ``def`` whose parameters become ``input`` instructions
  * assignments (``x = expr``, ``x += expr`` ...) to plain names
  * expressions made of names, integer literals, ``+ - * //`` and unary ``-``
  * ``return expr`` or ``return a, b, ...`` (each value becomes an ``output``)

``//`` maps to the TAC ``/`` operator. Python's ``//`` floors while the TAC
interpreter truncates toward zero, so they agree whenever the quotient's
operands are non-negative; the kernels that divide are therefore run on
positive input domains, and every program is checked against the original
Python source before it enters the dataset (see ``kernels.py``).

The code generator is deliberately naive, like an unoptimized compiler:
no expression value is ever reused, so redundancy that is present in the
source (a repeated subexpression, a literal chain such as ``2 * 314``, an
alias copy) survives into the TAC for the optimizer to find. The one thing
it does like any real compiler is pool literals: each distinct integer
value gets a single ``const`` instruction that every later use shares
(otherwise two uses of ``9`` would be different variables and common
subexpression elimination could never match ``c * 9`` with ``c * 9``). Variables are renamed so
each TAC destination is assigned exactly once (single-assignment form),
which the optimization passes rely on.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .tac import TAC

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "/"}


class UnsupportedSyntax(ValueError):
    """Raised when the source uses a construct the front end does not support."""


@dataclass
class CompiledFunction:
    name: str
    arg_names: List[str]
    tac: TAC


class _Compiler:
    def __init__(self):
        self.tac: TAC = []
        self.env: Dict[str, str] = {}   # python variable -> current TAC variable
        self.defined = set()            # every TAC destination used so far
        self._temp = 0
        self._const = 0
        self._pool: Dict[int, str] = {}  # literal value -> the const variable holding it

    # -- naming -----------------------------------------------------------
    def _fresh_temp(self) -> str:
        self._temp += 1
        return f"_e{self._temp}"

    def _fresh_const(self) -> str:
        self._const += 1
        return f"_k{self._const}"

    def _dest_for(self, py_name: str) -> str:
        """First assignment keeps the source name; later ones get a suffix (SSA)."""
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

    def _emit_const(self, value: int, dest: Optional[str]) -> str:
        value = int(value)
        if dest is None and value in self._pool:
            return self._pool[value]
        name = dest or self._fresh_const()
        self.tac.append(("const", name, value))
        self._pool.setdefault(value, name)
        return name

    # -- statements -------------------------------------------------------
    def assign(self, py_name: str, value: ast.AST):
        # evaluate the right-hand side first (it may read the old binding)
        dest = self._dest_for(py_name)
        result = self.expr(value, dest)
        assert result == dest
        self.env[py_name] = dest

    def statement(self, stmt: ast.stmt):
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
            if stmt.value is None:
                raise UnsupportedSyntax("a bare 'return' is not supported")
            values = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
            for v in values:
                self.tac.append(("output", self.expr(v)))
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return  # docstring
        else:
            raise UnsupportedSyntax(f"unsupported statement {type(stmt).__name__}")


def compile_function(source: str) -> CompiledFunction:
    """Compile the single function defined in ``source`` to TAC."""
    module = ast.parse(source)
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or len(module.body) != 1:
        raise UnsupportedSyntax("source must contain exactly one function definition")
    fn = funcs[0]
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
        raise UnsupportedSyntax("only plain positional parameters are supported")

    comp = _Compiler()
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
