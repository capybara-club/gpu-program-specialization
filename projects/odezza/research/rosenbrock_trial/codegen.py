# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Restricted arithmetic -> FP64 (default) or FP32 RHS and state Jacobian.

This audit compiler accepts no Python statements, attributes, indexing or code.
It is deliberately independent of the production FP32 SASS/literal ABI.
"""
import ast
import math
import struct

ZERO = ('c', 0.0)
ONE = ('c', 1.0)


def parse(expression, states):
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float) and math.isfinite(n.value):
            return ('c', float(n.value))
        if isinstance(n, ast.Name) and n.id in states:
            return ('s', states.index(n.id))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            child = visit(n.operand)
            return ('neg', child) if isinstance(n.op, ast.USub) else child
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            return ({ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(n.op)], visit(n.left), visit(n.right))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ('sin', 'cos', 'exp', 'tanh') and len(n.args) == 1 and not n.keywords):
            return (n.func.id, visit(n.args[0]))
        raise ValueError('unsupported arithmetic: ' + ast.dump(n))
    return visit(ast.parse(expression, mode='eval').body)


def op(kind, a, b=None):
    # Simplify derivatives only; never rewrite the submitted RHS/domain.
    if kind == 'neg':
        return ZERO if a == ZERO else (kind, a)
    if kind == '*' and (a == ZERO or b == ZERO): return ZERO
    if kind in ('+', '-') and b == ZERO: return a
    if kind == '+' and a == ZERO: return b
    if kind == '-' and a == ZERO: return op('neg', b)
    if kind == '*' and a == ONE: return b
    if kind in ('*', '/') and b == ONE: return a
    if kind == '/' and a == ZERO: return ZERO
    return (kind, a, b)


def derivative(node, state):
    k = node[0]
    if k == 'c': return ZERO
    if k == 's': return ONE if node[1] == state else ZERO
    a = node[1]; da = derivative(a, state)
    if k == 'neg': return op('neg', da)
    if k == 'sin': return op('*', ('cos', a), da)
    if k == 'cos': return op('*', op('neg', ('sin', a)), da)
    if k == 'exp': return op('*', node, da)
    if k == 'tanh': return op('*', op('-', ONE, op('*', node, node)), da)
    b = node[2]; db = derivative(b, state)
    if k in ('+', '-'): return op(k, da, db)
    if k == '*': return op('+', op('*', da, b), op('*', a, db))
    return op('/', op('-', op('*', da, b), op('*', a, db)), op('*', b, b))


def function(name, nodes, precision='FP64'):
    dtype='float' if precision=='FP32' else 'double'
    lines = []; cache = {}
    def emit(n):
        if n in cache: return cache[n]
        if n[0] == 'c':
            value=float(n[1])
            if precision=='FP32':
                try:value=struct.unpack('f',struct.pack('f',value))[0]
                except OverflowError:raise ValueError('literal outside FP32 range')
                if not math.isfinite(value):raise ValueError('literal outside FP32 range')
            return value.hex()+('f' if precision=='FP32' else '')
        if n[0] == 's': return 'y[%d]' % n[1]
        args = [emit(a) for a in n[1:]]
        if n[0] in ('+', '-', '*', '/'): expr = '(%s %s %s)' % (args[0], n[0], args[1])
        elif n[0] == 'neg': expr = '(-%s)' % args[0]
        else: expr = '%s(%s)' % (n[0]+('f' if precision=='FP32' else ''), args[0])
        v = 'v%d' % len(cache); cache[n] = v
        lines.append('  const %s %s = %s;' % (dtype, v, expr))
        return v
    outputs = [emit(n) for n in nodes]
    lines.extend('  out[%d] = %s;' % (i, v) for i, v in enumerate(outputs))
    return 'HD static void %s(const %s* y, %s* out) {\n%s\n}\n' % (name, dtype, dtype, '\n'.join(lines))


def model(expressions, states, precision='FP64'):
    if precision not in ('FP32','FP64'):raise ValueError('unknown model precision')
    if not 1 <= len(states) <= 16 or len(states) != len(set(states)) or len(expressions) != len(states):
        raise ValueError('requires 1..16 distinct states and one autonomous RHS per state')
    nodes = [parse(e, states) for e in expressions]
    return 'struct Model { static constexpr int N = %d;\n%s%s};\n' % (
        len(states), function('rhs', nodes, precision),
        function('jac', [derivative(n, j) for n in nodes for j in range(len(states))], precision))
