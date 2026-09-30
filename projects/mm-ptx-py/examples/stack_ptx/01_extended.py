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

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

registry = stack_ptx.RegisterRegistry()
registry.add("in_0",    Stack.u32)
registry.add("in_1",    Stack.u32)
registry.add("out_0",   Stack.u32)
registry.add("out_1",   Stack.u32)
registry.freeze()

# Describe the instructions we'd like to run as a list.
instructions = [
    # Push the constant '1' on to the u32 stack.
    Stack.u32.constant(1),
    # ...
    Stack.u32.constant(2),
    Stack.u32.constant(3),
    Stack.u32.constant(4),
    Stack.u32.constant(5),
    Stack.u32.constant(6),

    # Push the constant '3' on to the meta stack.
    Stack.meta_constant(3), 

    # Use the yank_dup meta instruction with the 3 being popped from the meta stack as a parameter,
    # This will go 3 deep in to the u32 stack and duplicate the value and push it on the top of the u32 stack.
    Stack.u32.yank_dup,   

    # Push register 'in_0' on to it's described stack (in this case u32).
    registry.in_0,

    # Push register 'in_1' on to it's described stack (in this case u32).
    registry.in_1,

    # Run the "add_u32" ptx instruction. This requires two values in the u32 stack which will be popped.
    # The result will be push on to the u32 stack. If the stack doesn't have the two values in the u32 stack
    # to use as operands, it will be skipped.
    PtxInstruction.add_u32,

    # ...
    PtxInstruction.add_u32,
]

# Now we make our requests. 
# out_0 will demand a value from the top of the u32 due to being declared a u32 value in the Register enum.
# if there is a value in the u32 stack, it will be popped and assigned to the register "out_0" in PTX.
# out_1 will demand the next value and if present will be assigned to the register "out_1" in PTX.
requests = [registry.out_0, registry.out_1]

# Now we run the Stack PTX to grab the buffer.
ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        # How many instructions to run before we halt.
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4
    )

print(ptx_stub)
