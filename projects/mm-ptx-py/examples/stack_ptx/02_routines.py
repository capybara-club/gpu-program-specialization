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

from enum import auto, unique

import mm_ptx.stack_ptx as stack_ptx

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

registry = stack_ptx.RegisterRegistry()
registry.add("in_0", Stack.u32)
registry.add("in_1", Stack.u32)
registry.add("out_0", Stack.u32)
registry.add("out_1", Stack.u32)
registry.freeze()

# Routines allow function calls in Stack PTX.
# We declare the routines just like we declare registers with auto().
# In this case we initially set the routines to an empty list. You can
# set it to actual instructions here but in this example we're going to have
# the routines call each other.
@unique
class Routine(stack_ptx.RoutineEnum):
    routine_0 = (auto(), [])
    routine_1 = (auto(), [])
    routine_2 = (auto(), [])

# Declare the routine instructions just like we 
# declare the Stack PTX instructions. It just 
# pushes 1 to the u32 stack.
Routine.routine_0.instructions = [
    Stack.u32.constant(1)
]

# We declare this routine, push 2 on the u32 
# and then call routine_0.
Routine.routine_1.instructions = [
    Stack.u32.constant(2),
    Routine.routine_0
]

# We declare this routine, push 3 on the u32 
# and then call routine_1.
Routine.routine_2.instructions = [
    Stack.u32.constant(3),
    Routine.routine_1
]


instructions = [
    # We call routine_2 which calls routine_1 which calls routine_0.
    Routine.routine_2,
    # We call routine_1 which calls routine_0.
    Routine.routine_1,
    # We call routine_0.
    Routine.routine_0,

    # On the u32 we should now have 
    # one 3.
    # two 2s.
    # three 1s.

    # We call a few add's so that the results all show up
    # in the PTX.
    PtxInstruction.add_u32,
    PtxInstruction.add_u32,
    PtxInstruction.add_u32,
    PtxInstruction.add_u32,
]

# We make two requests and we should see all of the constants
# pushed by the routines show up in the PTX.
requests = [registry.out_0, registry.out_1]

ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        routine_enum=Routine,
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4
    )

print(ptx_stub)
