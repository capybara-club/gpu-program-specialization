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
"""Independent scalar interpreter for the native postorder wire format."""
import math
import struct


def expression(program, state_names):
    """Readable resolved equation; exact replay continues to use the bytecode."""
    data = bytes.fromhex(program) if isinstance(program, str) else program
    stack, i = [], 0
    functions = {0x95: "sqrt", 0x97: "abs", 0x9b: "sin", 0x9c: "cos", 0xa0: "tanh", 0xa1: "exp", 0xa2: "log"}
    while i < len(data):
        op = data[i]; i += 1
        if op == 0x80: return stack[0][0]
        if op == 0x81:
            stack.append((state_names[data[i]], 100)); i += 1
        elif op == 0x83:
            value = struct.unpack_from("<f", data, i)[0]; i += 4
            stack.append((format(value, ".9g"), 100 if value >= 0 else 30))
        elif op == 0x94:
            text, precedence = stack.pop()
            stack.append(("-"+("("+text+")" if precedence <= 30 else text), 30))
        elif op in functions:
            text, _ = stack.pop(); stack.append((functions[op]+"("+text+")", 100))
        else:
            symbol, precedence = {0x90: (" + ", 10), 0x91: (" - ", 10), 0x92: (" * ", 20), 0x93: (" / ", 20)}[op]
            (right, rp), (left, lp) = stack.pop(), stack.pop()
            if lp < precedence: left = "("+left+")"
            if rp <= precedence: right = "("+right+")"
            stack.append((left+symbol+right, precedence))
    raise ValueError("Missing RETURN")


def describe(candidate, states):
    result = dict(candidate)
    result["equations"] = {state: expression(program, states) for state, program in zip(states, candidate["resolved_programs"])}
    named = {"const": {}, "rng": {}, "param": {}}
    for (namespace, name), value in zip(candidate["slots"], candidate["values"]): named[namespace][name] = value
    result["named_values"] = named
    return result


def evaluate(program, states, constants=(), permutation=0):
    if isinstance(program, str):
        program = bytes.fromhex(program)
    stack, i = [], 0
    unary = {0x94: lambda a: -a, 0x95: math.sqrt, 0x97: abs, 0x9b: math.sin,
             0x9c: math.cos, 0xa0: math.tanh, 0xa1: math.exp, 0xa2: math.log}
    while i < len(program):
        op = program[i]; i += 1
        if op == 0x80:
            if i != len(program) or len(stack) != 1:
                raise ValueError("Malformed native program")
            return stack[0]
        if op in (0x81, 0x82):
            stack.append((states if op == 0x81 else constants)[program[i]]); i += 1
        elif op == 0x83:
            stack.append(struct.unpack_from("<f", program, i)[0]); i += 4
        elif op in (0x84, 0x85):
            width = 1 if op == 0x84 else 2
            choice = sum(((permutation >> program[i+j]) & 1) << j for j in range(width))
            i += width; count = 1 << width
            values = stack[-count:]; del stack[-count:]; stack.append(values[choice])
        elif op in unary:
            stack.append(unary[op](stack.pop()))
        else:
            b, a = stack.pop(), stack.pop()
            stack.append({0x90: lambda: a+b, 0x91: lambda: a-b, 0x92: lambda: a*b, 0x93: lambda: a/b}[op]())
    raise ValueError("Missing RETURN")
