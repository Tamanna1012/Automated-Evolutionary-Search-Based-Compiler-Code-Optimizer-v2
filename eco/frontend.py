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
from typing import Dict, List, Optional, Tuple

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
    def __init__(self, control: bool = False, helpers: Optional[Dict[str, ast.FunctionDef]] = None):
        self.control = control
        self.helpers = helpers or {}
        self.prefix = ""
        self.arrays: Dict[str, int] = {}
        self._inline_depth = 0
        self._inline_count = 0
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

        if isinstance(node, ast.Subscript):
            name, k = self._element(node)
            return self.expr(ast.Name(id=f"{name}__{k}", ctx=ast.Load()), dest)

        if isinstance(node, ast.Name):
            if node.id in self.arrays:
                raise UnsupportedSyntax(f"array '{node.id}' must be indexed with an integer literal, e.g. {node.id}[0]")
            if node.id not in self.env:
                raise UnsupportedSyntax(f"use of undefined variable '{node.id}'")
            src = self.env[node.id]
            if dest is None:
                return src  # a bare read needs no instruction
            self.tac.append(("copy", dest, src))
            return dest

        if isinstance(node, ast.Call):
            return self._call(node, dest)

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
        """Emit code that leaves 1 (true) or 0 (false) in the returned variable."""
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
            left = self.expr(node.left)
            right = self.expr(node.comparators[0])
            out = self._fresh_temp()
            self.tac.append(("bin", out, _CMPOPS[type(node.ops[0])], left, right))
            return out
        if isinstance(node, ast.BoolOp):
            values = [self._condition(v) for v in node.values]
            out = values[0]
            for v in values[1:]:
                t = self._fresh_temp()
                if isinstance(node.op, ast.And):
                    self.tac.append(("bin", t, "*", out, v))      # 0/1 values: product is their 'and'
                else:
                    summed = self._fresh_temp()
                    self.tac.append(("bin", summed, "+", out, v))
                    zero = self._emit_const(0, None)
                    self.tac.append(("bin", t, "!=", summed, zero))
                out = t
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = self._condition(node.operand)
            zero = self._emit_const(0, None)
            out = self._fresh_temp()
            self.tac.append(("bin", out, "==", inner, zero))
            return out
        value = self.expr(node)
        zero = self._emit_const(0, None)
        out = self._fresh_temp()
        self.tac.append(("bin", out, "!=", value, zero))
        return out

    def _call(self, node: ast.Call, dest: Optional[str]) -> str:
        if not (isinstance(node.func, ast.Name) and node.func.id in self.helpers):
            raise UnsupportedSyntax(f"only calls to functions defined earlier in the source are supported "
                                    f"(got '{ast.unparse(node.func)}')")
        helper = self.helpers[node.func.id]
        params = [a.arg for a in helper.args.args]
        if node.keywords or len(node.args) != len(params):
            raise UnsupportedSyntax(f"'{helper.name}' takes {len(params)} positional arguments")
        if self._inline_depth >= 20:
            raise UnsupportedSyntax("recursive calls are not supported")
        if not (helper.body and isinstance(helper.body[-1], ast.Return) and helper.body[-1].value is not None):
            raise UnsupportedSyntax(f"helper '{helper.name}' must end with 'return <value>'")
        args = [self.expr(a) for a in node.args]
        saved = (self.env, self.prefix)
        self._inline_count += 1
        self.env = dict(zip(params, args))
        self.prefix = f"{helper.name}{self._inline_count}_"
        self._inline_depth += 1
        try:
            for st in helper.body[:-1]:
                self.statement(st, in_block=True)
            result = self.expr(helper.body[-1].value)
        finally:
            self._inline_depth -= 1
            self.env, self.prefix = saved
        if dest is None:
            return result
        self.tac.append(("copy", dest, result))
        return dest

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
        dest = self._dest_for(self.prefix + py_name)
        result = self.expr(value, dest)
        assert result == dest
        self.env[py_name] = dest

    def statement(self, stmt: ast.stmt, in_block: bool = False):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise UnsupportedSyntax("only simple 'name = expr' assignments are supported")
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.List):
                self._array_literal(target.id, stmt.value)
            elif isinstance(target, ast.Subscript):
                name, k = self._element(target)
                self.assign(f"{name}__{k}", stmt.value)
            elif isinstance(target, ast.Name):
                self.assign(target.id, stmt.value)
            else:
                raise UnsupportedSyntax("only simple 'name = expr' assignments are supported")
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

    def _array_literal(self, name: str, node: ast.List):
        if name in self.arrays and self.arrays[name] != len(node.elts):
            raise UnsupportedSyntax(f"array '{name}' changes length; arrays have a fixed size")
        self.arrays[name] = len(node.elts)
        for i, e in enumerate(node.elts):
            self.assign(f"{name}__{i}", e)

    def _element(self, node: ast.Subscript) -> Tuple[str, int]:
        if not (isinstance(node.value, ast.Name) and node.value.id in self.arrays):
            raise UnsupportedSyntax("only local arrays created with [..] can be indexed")
        if not (isinstance(node.slice, ast.Constant) and type(node.slice.value) is int):
            raise UnsupportedSyntax("array indices must be integer literals; dynamic indexing needs a memory model")
        k = node.slice.value
        if not 0 <= k < self.arrays[node.value.id]:
            raise UnsupportedSyntax(f"index {k} is out of range for array '{node.value.id}'")
        return node.value.id, k

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


def _check_function(fn: ast.FunctionDef):
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
        raise UnsupportedSyntax("only plain positional parameters are supported")
    for i, stmt in enumerate(fn.body):
        if isinstance(stmt, ast.Return) and i != len(fn.body) - 1:
            raise UnsupportedSyntax("'return' must be the last statement of the function")


def compile_function(source: str) -> CompiledFunction:
    """Compile the last function defined in ``source`` to TAC.

    Functions defined before it are helpers: a call to one is inlined at the call site.
    """
    module = ast.parse(source)
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if not funcs or len(funcs) != len(module.body):
        raise UnsupportedSyntax("source must contain only function definitions")
    for fn in funcs:
        _check_function(fn)
    fn = funcs[-1]
    helpers = {f.name: f for f in funcs[:-1]}

    control = has_control_flow(source)
    comp = _Compiler(control, helpers)
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
    namespace: dict = {"__builtins__": {}}
    exec(compile(source, f"<kernel {func_name}>", "exec"), namespace)  # noqa: S102
    result = namespace[func_name](**inputs)
    if isinstance(result, tuple):
        return [int(v) for v in result]
    return [int(result)]
