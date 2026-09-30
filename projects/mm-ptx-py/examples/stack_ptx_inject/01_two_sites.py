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

from cuda.core import LaunchConfig, launch
import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx
from mm_ptx import get_ptx_inject_header

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler_helper import NvCompilerHelper
from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

ptx_header = get_ptx_inject_header().replace("\\", "/")

cuda_code = f"""
#include \"{ptx_header}\"

extern \"C\"
__global__
void
kernel() {{
    float x = 5;
    float y = 3;
    float z = 0;
    float w = 0;

    PTX_INJECT(\"add\",
        PTX_IN (F32, x, x),
        PTX_IN (F32, y, y),
        PTX_OUT(F32, z, z)
    );

    PTX_INJECT(\"mul\",
        PTX_IN (F32, x, x),
        PTX_IN (F32, y, y),
        PTX_OUT(F32, w, w)
    );

    printf(\"%f %f\\n\", z, w);
}}
"""

nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func_add = inject["add"]
func_mul = inject["mul"]

registry_add = stack_ptx.RegisterRegistry()
registry_add.add(func_add["x"].reg, Stack.f32, name="x")
registry_add.add(func_add["y"].reg, Stack.f32, name="y")
registry_add.add(func_add["z"].reg, Stack.f32, name="z")
registry_add.freeze()

instructions_add = [
    registry_add.x,
    registry_add.y,
    PtxInstruction.add_ftz_f32,
]

stub_add = stack_ptx_compiler.compile(
    registry=registry_add,
    instructions=instructions_add,
    requests=[registry_add.z],
    execution_limit=100,
    max_ast_size=100,
    max_ast_to_visit_stack_depth=20,
    stack_size=128,
    max_frame_depth=4,
    store_size=16,
)

registry_mul = stack_ptx.RegisterRegistry()
registry_mul.add(func_mul["x"].reg, Stack.f32, name="x")
registry_mul.add(func_mul["y"].reg, Stack.f32, name="y")
registry_mul.add(func_mul["w"].reg, Stack.f32, name="w")
registry_mul.freeze()

instructions_mul = [
    registry_mul.x,
    registry_mul.y,
    PtxInstruction.mul_ftz_f32,
]

stub_mul = stack_ptx_compiler.compile(
    registry=registry_mul,
    instructions=instructions_mul,
    requests=[registry_mul.w],
    execution_limit=100,
    max_ast_size=100,
    max_ast_to_visit_stack_depth=20,
    stack_size=128,
    max_frame_depth=4,
    store_size=16,
)

ptx_stubs = {
    "add": stub_add,
    "mul": stub_mul,
}

rendered_ptx = inject.render_ptx(ptx_stubs)

mod = nv_compiler.ptx_to_cubin(rendered_ptx)

ker = mod.get_kernel("kernel")

block = int(1)
grid = int(1)
config = LaunchConfig(grid=grid, block=block)
ker_args = ()

stream = nv_compiler.dev.default_stream

launch(stream, config, ker, *ker_args)

print("Should print 8.0000 15.0000")
stream.sync()
