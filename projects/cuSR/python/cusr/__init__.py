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
from .ast import (
    ABS,
    ADD,
    COS,
    DIV,
    EX2,
    FMA,
    INSTRUCTION_SIZE,
    LG2,
    MAX,
    MAX_PROGRAM_INSTRUCTIONS,
    MIN,
    MUL,
    NEG,
    RCP,
    RETURN,
    RSQRT,
    SIN,
    SQRT,
    SUB,
    TANH,
    InstructionType,
    Op,
    decode_instruction,
    encode_constant,
    encode_constant_bits,
    encode_input,
    encode_op,
    encode_routine,
    encode_routine_arg,
    pack_program,
    pack_programs,
    verify_native_abi,
)
from .compiler import (
    CompiledTileStaticEvalCubin,
    CompiledTileStaticMseCubin,
    CubinCompileError,
    TileStaticEvalInstantiation,
    TileStaticMseInstantiation,
    compile_tile_static_eval_cubin,
    compile_tile_static_mse_cubin,
    make_tile_static_eval_instantiations,
    make_tile_static_mse_instantiations,
)
from .inspection import (
    CubinInspectError,
    CubinInspectErrorCode,
    CubinInspection,
    RegcountRecord,
    SassKernel,
    SassSite,
    inspect_cubin,
    inspect_cubin_file,
    print_cubin_inspection,
)
from .patching import PatchEpilogue, patch_cubin_in_place, prepare_patch_layout
from .reference import evaluate_program, evaluate_programs, evaluate_sse, evaluate_values
from .routines import DEFAULT_ROUTINES, Routine
from .runtime import TileStaticEvalModule, TileStaticMseModule

__all__ = (
    "ABS",
    "ADD",
    "COS",
    "DIV",
    "EX2",
    "FMA",
    "INSTRUCTION_SIZE",
    "LG2",
    "MAX",
    "MAX_PROGRAM_INSTRUCTIONS",
    "MIN",
    "MUL",
    "NEG",
    "RCP",
    "RETURN",
    "RSQRT",
    "SIN",
    "SQRT",
    "SUB",
    "TANH",
    "InstructionType",
    "Op",
    "PatchEpilogue",
    "decode_instruction",
    "encode_constant",
    "encode_constant_bits",
    "encode_input",
    "encode_op",
    "encode_routine",
    "encode_routine_arg",
    "pack_program",
    "pack_programs",
    "verify_native_abi",
    "CompiledTileStaticEvalCubin",
    "CompiledTileStaticMseCubin",
    "CubinCompileError",
    "TileStaticEvalInstantiation",
    "TileStaticMseInstantiation",
    "compile_tile_static_eval_cubin",
    "compile_tile_static_mse_cubin",
    "make_tile_static_eval_instantiations",
    "make_tile_static_mse_instantiations",
    "CubinInspectError",
    "CubinInspectErrorCode",
    "CubinInspection",
    "RegcountRecord",
    "SassKernel",
    "SassSite",
    "inspect_cubin",
    "inspect_cubin_file",
    "print_cubin_inspection",
    "patch_cubin_in_place",
    "prepare_patch_layout",
    "evaluate_program",
    "evaluate_programs",
    "evaluate_sse",
    "evaluate_values",
    "DEFAULT_ROUTINES",
    "Routine",
    "TileStaticEvalModule",
    "TileStaticMseModule",
)
