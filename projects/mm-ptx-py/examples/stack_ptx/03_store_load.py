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

from enum import IntEnum 

import mm_ptx.stack_ptx as stack_ptx

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

registry = stack_ptx.RegisterRegistry()
registry.add("out_0",   Stack.u32)
registry.add("out_1",   Stack.u32)
registry.add("out_2",   Stack.u32)
registry.add("out_3",   Stack.u32)
registry.freeze()

# This helps name the store locations.
class Var(IntEnum):
    add = 0

# Describe the instructions we'd like to run as a list.
instructions = [
    Stack.u32.constant(1),
    Stack.u32.constant(2),
    PtxInstruction.add_u32,
    # Store the result of 1 + 2 for later. Pops the stack and saves the value
    Stack.u32.store(Var.add),   
    Stack.u32.constant(3),  # Push 3
    Stack.u32.constant(4),  # Push 4
    # Now load the result of 1 + 2 four times to show its
    # the only value output among the four values requested.
    Stack.load(Var.add),    
    Stack.load(Var.add),
    Stack.load(Var.add),
    Stack.load(Var.add),
]

requests = [registry.out_0, registry.out_1, registry.out_2, registry.out_3]

# Now we run the Stack PTX to grab the buffer.
ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4,
        # Can change this to increase the storage amount.
        # Each store element is 8 bytes.
        store_size=16
    )

print(ptx_stub)
