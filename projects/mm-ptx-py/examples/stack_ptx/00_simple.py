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

import sys
import os

import mm_ptx.stack_ptx as stack_ptx

# Use the upper directory helpers.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

# Describe which registers map to which Stack PTX stacks.
registry = stack_ptx.RegisterRegistry()
registry.add("in_0", Stack.f32)
registry.add("in_1", Stack.f32)
registry.add("out_0", Stack.f32)
registry.freeze()

# Instructions: out_0 = in_0 * in_1 + in_0
instructions = [
    registry.in_0,                  # Push in_0
    registry.in_1,                  # Push in_1
    PtxInstruction.mul_ftz_f32,     # Pop, Pop, Push in_0 * in_1
    registry.in_0,                  # Push in_0
    PtxInstruction.add_ftz_f32,     # Pop, Pop, Push in_0 * in_1 + in_0
]

# Request the top value of the f32 stack into out_0.
requests = [registry.out_0]

# Compile the Stack PTX program into a PTX stub.
ptx_stub = stack_ptx_compiler.compile(
    registry=registry,
    instructions=instructions,
    requests=requests,
    execution_limit=100,
    max_ast_size=100,
    max_ast_to_visit_stack_depth=20,
    stack_size=128,
    max_frame_depth=4
)

print(ptx_stub)
