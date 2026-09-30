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

class Generator:
    def __init__(
        self,
        add_instruction,
        mul_instruction,
        sin_instruction,
        cos_instruction,
        const_lambda,
        x_instruction,
        y_instruction,
        t_instruction,
        min_terms = 2,
        max_terms = 5,
        min_const = -2.0,
        max_const = 2.0,
        max_depth = 3,
        prob_leaf = 0.3,
        prob_const_in_leaf = 0.1,
        prob_unary = 0.2,
        prob_outer_const = 1.0
    ):
        self.add_instruction = add_instruction
        self.mul_instruction = mul_instruction
        self.sin_instruction = sin_instruction
        self.cos_instruction = cos_instruction
        self.const_lambda = const_lambda
        self.x_instruction = x_instruction
        self.y_instruction = y_instruction
        self.t_instruction = t_instruction

        self.min_terms = min_terms
        self.max_terms = max_terms
        self.min_const = min_const
        self.max_const = max_const
        self.max_depth = max_depth
        self.prob_leaf = prob_leaf
        self.prob_const_in_leaf = prob_const_in_leaf
        self.prob_unary = prob_unary
        self.prob_outer_const = prob_outer_const

    class Node:
        def emit(self, gen):
            pass

    class Var(Node):
        def __init__(self, name):
            self.name = name  # 'x', 'y', 't'

        def emit(self, gen):
            if self.name == 'x':
                return [gen.x_instruction]
            elif self.name == 'y':
                return [gen.y_instruction]
            elif self.name == 't':
                return [gen.t_instruction]
            else:
                raise ValueError(f"Unknown variable: {self.name}")

    class Const(Node):
        def __init__(self, val):
            self.val = val

        def emit(self, gen):
            return [gen.const_lambda(self.val)]

    class Unary(Node):
        def __init__(self, op, child):
            self.op = op  # 'sin' or 'cos'
            self.child = child

        def emit(self, gen):
            instr = gen.sin_instruction if self.op == 'sin' else gen.cos_instruction
            return self.child.emit(gen) + [instr]

    class Binary(Node):
        def __init__(self, op, left, right):
            self.op = op  # 'add' or 'mul'
            self.left = left
            self.right = right

        def emit(self, gen):
            instr = gen.add_instruction if self.op == 'add' else gen.mul_instruction
            return self.left.emit(gen) + self.right.emit(gen) + [instr]

    def random_expr(self, depth=0):
        if depth >= self.max_depth or random.random() < self.prob_leaf + (depth / self.max_depth) * (1 - self.prob_leaf):
            if random.random() < self.prob_const_in_leaf:
                val = random.uniform(self.min_const, self.max_const)
                return self.Const(val)
            else:
                var = random.choice(['x', 'y', 't'])
                return self.Var(var)
        else:
            r = random.random()
            if r < self.prob_unary:
                op = random.choice(['sin', 'cos'])
                child = self.random_expr(depth + 1)
                return self.Unary(op, child)
            else:
                op = random.choice(['add', 'mul'])
                left = self.random_expr(depth + 1)
                right = self.random_expr(depth + 1)
                return self.Binary(op, left, right)

    def generate_instructions(self):
        num_terms = random.randint(self.min_terms, self.max_terms)
        terms = []
        for _ in range(num_terms):
            inner = self.random_expr()
            trig_op = random.choice(['sin', 'cos'])
            term_expr = self.Unary(trig_op, inner)
            if random.random() < self.prob_outer_const:
                outer_const = self.Const(random.uniform(self.min_const, self.max_const))
                term_expr = self.Binary('mul', term_expr, outer_const)
            terms.append(term_expr)
        
        # Emit instructions for sum of terms
        if not terms:
            return []
        instructions = terms[0].emit(self)
        for term in terms[1:]:
            instructions += term.emit(self)
            instructions += [self.add_instruction]
        return instructions
