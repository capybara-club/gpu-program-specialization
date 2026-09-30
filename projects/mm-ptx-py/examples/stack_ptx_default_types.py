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

from mm_ptx.stack_ptx import (
    StackTypeEnum, 
    ArgTypeEnum, 
    create_instruction_enum,
    create_special_register_enum,
    StackPtx
)

from enum import auto, unique

@unique
class Stack(StackTypeEnum):
    # Fields: (auto-id, ptx literal prefix)
    f32 =   (auto(), "f32")
    s32 =   (auto(), "s32")
    u32 =   (auto(), "u32")

@unique
class ArgType(ArgTypeEnum):
    # Fields: (auto-id, stack type enum, optional num_vec_elems)
    f32 =   (auto(), Stack.f32)
    u32 =   (auto(), Stack.u32)

@unique
class PtxInstruction(create_instruction_enum(ArgType)):
    # Fields: (auto-id, ptx opcode string, [arg types], [ret types], optional aligned flag)
    add_u32 =               (auto(),    "add.u32",              [ArgType.u32, ArgType.u32],     [ArgType.u32])
    add_ftz_f32 =           (auto(),    "add.ftz.f32",          [ArgType.f32, ArgType.f32],     [ArgType.f32])
    mul_ftz_f32 =           (auto(),    "mul.ftz.f32",          [ArgType.f32, ArgType.f32],     [ArgType.f32])
    sin_approx_ftz_f32 =    (auto(),    "sin.approx.ftz.f32",   [ArgType.f32],                  [ArgType.f32])
    cos_approx_ftz_f32 =    (auto(),    "cos.approx.ftz.f32",   [ArgType.f32],                  [ArgType.f32])

@unique
class SpecialRegister(create_special_register_enum(ArgType)):
    # Fields: (auto-id, ptx register name, arg type)
    clock =     (auto(),    "clock",      ArgType.u32)
    tid_x =     (auto(),    "tid.x",      ArgType.u32)

compiler = \
    StackPtx(
        stack_enum=Stack,
        arg_enum=ArgType,
        ptx_instruction_enum=PtxInstruction, 
        special_register_enum=SpecialRegister
    )
