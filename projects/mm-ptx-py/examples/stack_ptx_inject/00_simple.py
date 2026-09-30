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

# For this example we're now going to fuse both the Stack PTX and the PTX Inject systems in to one.
# We're going to declare a kernel with a PTX_INJECT declaration. We'll pull out the
# register names assigned to the cuda variables and then use them with Stack PTX to
# form valid PTX code. We'll compile the PTX and run it.

import sys
import os

import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx
from mm_ptx import get_ptx_inject_header

from cuda.core import LaunchConfig, launch

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler
from compiler_helper import NvCompilerHelper

ptx_header = get_ptx_inject_header().replace("\\", "/")

# Inline CUDA with a PTX_INJECT site we can patch from Python.
cuda_code = f"""
#include \"{ptx_header}\"

extern \"C\"
__global__
void
kernel() {{
    float x = 5;
    float y = 3;
    float z;
    for (int i = 0; i < 2; i++) {{
        PTX_INJECT(\"func\",
            PTX_IN (F32, x, x),
            PTX_MOD(F32, y, y),
            PTX_OUT(F32, z, z)
        );
    }}
    printf(\"%f\\n\", z);
}}
"""

# Compile CUDA to annotated PTX.
nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

# Parse inject sites and inspect their arguments.
inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func = inject["func"]

assert func["x"].mut_type == ptx_inject.MutType.IN
assert func["x"].data_type == "F32"

assert func["y"].mut_type == ptx_inject.MutType.MOD
assert func["y"].data_type == "F32"

assert func["z"].mut_type == ptx_inject.MutType.OUT
assert func["z"].data_type == "F32"

# Bind PTX register names to Stack PTX registers.
registry = stack_ptx.RegisterRegistry()
registry.add(func["x"].reg, Stack.f32, name="x")
registry.add(func["y"].reg, Stack.f32, name="y")
registry.add(func["z"].reg, Stack.f32, name="z")
registry.freeze()

# Emit instructions for z = (x + y) + (x + y) * x.
instructions = [
    registry.x,                     # Push x: [x]
    registry.y,                     # Push y: [x,y]
    PtxInstruction.add_ftz_f32,     # Pop x, y, Push x + y: [x + y]
    Stack.f32.dup,                  # Duplicate, Push (x + y): [x + y, x + y]
    registry.x,                     # Push x: [x + y, x + y, x]
    PtxInstruction.mul_ftz_f32,     # Pop two from top, Push... : [x + y, (x + y) * x]
    PtxInstruction.add_ftz_f32      # Pop two, Push... : [(x + y) + (x + y) * x]
]

print(instructions)

# Request values for z and y from the top of the stack.
requests = [registry.z, registry.y]

# Compile Stack PTX to a stub and inject it.
ptx_stub = stack_ptx_compiler.compile(
    registry=registry,
    instructions=instructions,
    requests=requests,
    execution_limit=100,
    max_ast_size=100,
    max_ast_to_visit_stack_depth=20,
    stack_size=128,
    max_frame_depth=4,
    store_size=16,
)

print(ptx_stub)

ptx_stubs = {
    "func": ptx_stub
}

rendered_ptx = inject.render_ptx(ptx_stubs)

# Compile injected PTX to a cubin and launch.
mod = nv_compiler.ptx_to_cubin(rendered_ptx)

ker = mod.get_kernel("kernel")

block = int(1)
grid = int(1)
config = LaunchConfig(grid=grid, block=block)
ker_args = ()

stream = nv_compiler.dev.default_stream

launch(stream, config, ker, *ker_args)

print("Should print 48.0000")
stream.sync()
