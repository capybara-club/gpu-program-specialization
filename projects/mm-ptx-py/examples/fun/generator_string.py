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
# SPDX-FileCopyrightText: 2026 Charles Durham
#
# SPDX-License-Identifier: MIT

import random

# Parameters (configurable)
MIN_TERMS = 2
MAX_TERMS = 5
MIN_CONST = -10.0
MAX_CONST = 10.0
MAX_DEPTH = 3
PROB_LEAF = 0.3  # Probability to choose leaf at each step (increases with depth)
PROB_CONST_IN_LEAF = 0.4  # Among leaves, prob to choose const vs var
PROB_UNARY = 0.2  # Prob to choose unary vs binary when not leaf
PROB_OUTER_CONST = 0.5  # Prob to have an outer constant multiplier for each term

class Node:
    def emit(self):
        pass

class Var(Node):
    def __init__(self, name):
        self.name = name  # 'x', 'y', 't'

    def emit(self):
        return [f"registry.{self.name}()"]

class Const(Node):
    def __init__(self, val):
        self.val = val

    def emit(self):
        return [f"Stack.f32.constant({self.val:.6f})"]

class Unary(Node):
    def __init__(self, op, child):
        self.op = op  # 'sin' or 'cos'
        self.child = child

    def emit(self):
        return self.child.emit() + [f"PtxInstruction.{self.op}_approx_ftz_f32()"]

class Binary(Node):
    def __init__(self, op, left, right):
        self.op = op  # 'add' or 'mul'
        self.left = left
        self.right = right

    def emit(self):
        return self.left.emit() + self.right.emit() + [f"PtxInstruction.{self.op}_ftz_f32()"]

def random_expr(depth=0):
    if depth >= MAX_DEPTH or random.random() < PROB_LEAF + (depth / MAX_DEPTH) * (1 - PROB_LEAF):
        if random.random() < PROB_CONST_IN_LEAF:
            val = random.uniform(MIN_CONST, MAX_CONST)
            return Const(val)
        else:
            var = random.choice(['x', 'y', 't'])
            return Var(var)
    else:
        r = random.random()
        if r < PROB_UNARY:
            op = random.choice(['sin', 'cos'])
            child = random_expr(depth + 1)
            return Unary(op, child)
        else:
            op = random.choice(['add', 'mul'])
            left = random_expr(depth + 1)
            right = random_expr(depth + 1)
            return Binary(op, left, right)

def generate_instructions():
    num_terms = random.randint(MIN_TERMS, MAX_TERMS)
    terms = []
    for _ in range(num_terms):
        inner = random_expr()
        trig_op = random.choice(['sin', 'cos'])
        term_expr = Unary(trig_op, inner)
        if random.random() < PROB_OUTER_CONST:
            outer_const = Const(random.uniform(MIN_CONST, MAX_CONST))
            term_expr = Binary('mul', term_expr, outer_const)
        terms.append(term_expr)
    
    # Emit instructions for sum of terms
    if not terms:
        return []
    instructions = terms[0].emit()
    for term in terms[1:]:
        instructions += term.emit()
        instructions += ["PtxInstruction.add_ftz_f32()"]
    return instructions

# Example usage: generate and print one set of instructions
if __name__ == "__main__":
    instrs = generate_instructions()
    print("instructions = [")
    for instr in instrs:
        print("    " + instr + ",")
    print("]")