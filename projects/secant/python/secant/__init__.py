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
    Expression,
    InstructionType,
    Program,
    absolute,
    affine_stats,
    constant,
    constant_bits,
    cos,
    dynamic_column,
    dynamic_constant,
    dynamic_constant_or_column,
    dynamic_constant_sse,
    dynamic_leaf_sse,
    evaluate,
    exp,
    exp2,
    fma,
    gram_stats,
    input,
    log,
    log2,
    materialize,
    maximum,
    minimum,
    pack_programs,
    rcp,
    routine,
    routine_arg,
    rsqrt,
    sin,
    sqrt,
    sse,
    tanh,
    validate_program,
)
from .compiler import (
    AffineStatsRecipe,
    CompiledTemplate,
    CubinCompileError,
    DynamicConstantSSERecipe,
    DynamicLeafSSERecipe,
    GramStatsRecipe,
    MaterializeRecipe,
    SSERecipe,
    compile_affine_stats,
    compile_dynamic_constant_sse,
    compile_dynamic_leaf_sse,
    compile_gram_stats,
    compile_materialize,
    compile_sse,
)
from .routines import (
    DEFAULT_ROUTINES,
    SAFE_DIV_ROUTINE,
    SAFE_RSQRT_ROUTINE,
    SAFE_SQRT_ROUTINE,
    safe_div,
    safe_rsqrt,
    safe_sqrt,
)
from .runtime import (
    AffineStatsModule,
    BulkSSERunResult,
    DynamicConstantSSEModule,
    DynamicLeafSSEModule,
    GramStatsModule,
    MaterializeModule,
    ResidentSSE,
    SSEBulkRunner,
    SSEModule,
    SSERunResult,
)

__all__ = (
    "AffineStatsModule",
    "AffineStatsRecipe",
    "CompiledTemplate",
    "CubinCompileError",
    "DEFAULT_ROUTINES",
    "BulkSSERunResult",
    "DynamicConstantSSEModule",
    "DynamicLeafSSEModule",
    "DynamicConstantSSERecipe",
    "DynamicLeafSSERecipe",
    "Expression",
    "GramStatsModule",
    "GramStatsRecipe",
    "InstructionType",
    "MaterializeModule",
    "MaterializeRecipe",
    "Program",
    "ResidentSSE",
    "SAFE_DIV_ROUTINE",
    "SAFE_RSQRT_ROUTINE",
    "SAFE_SQRT_ROUTINE",
    "SSEModule",
    "SSEBulkRunner",
    "SSERecipe",
    "SSERunResult",
    "absolute",
    "affine_stats",
    "compile_affine_stats",
    "compile_dynamic_constant_sse",
    "compile_dynamic_leaf_sse",
    "compile_gram_stats",
    "compile_materialize",
    "compile_sse",
    "constant",
    "constant_bits",
    "cos",
    "dynamic_column",
    "dynamic_constant",
    "dynamic_constant_or_column",
    "dynamic_constant_sse",
    "dynamic_leaf_sse",
    "evaluate",
    "exp",
    "exp2",
    "fma",
    "gram_stats",
    "input",
    "log",
    "log2",
    "materialize",
    "maximum",
    "minimum",
    "pack_programs",
    "rcp",
    "routine",
    "routine_arg",
    "rsqrt",
    "safe_div",
    "safe_rsqrt",
    "safe_sqrt",
    "sin",
    "sqrt",
    "sse",
    "tanh",
    "validate_program",
)
