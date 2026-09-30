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
    NUM_BINARY_OPS,
    NUM_INPUTS,
    NUM_UNARY_OPS,
    BinaryAst,
    BinaryOp,
    UnaryOp,
    ast_arrays,
    binary_ast,
    deterministic_test_cohort,
    f32_constant_word,
    padding_ast,
    random_ast_cohort,
)
from .columns import (
    CompiledColumnKernels,
    CompiledSingleAstColumnKernels,
    compile_column_kernels,
    compile_single_ast_column_kernels,
)
from .gram import CompiledGramKernels, GramOutputs, KernelsPerModule, compile_gram_kernels
from .expr import (
    BinaryExpr,
    ConstantExpr,
    Expr,
    FeatureExpr,
    LeafBinding,
    UnaryExpr,
    abs_f32,
    as_expr,
    const,
    cube,
    cos,
    ex2,
    exp,
    expression_to_ast_and_leaf_settings,
    expressions_to_asts_and_settings,
    feature,
    log10,
    log2,
    maximum,
    minimum,
    neg,
    rcp,
    rsqrt,
    safe_div,
    safe_ex2,
    safe_exp,
    safe_log10,
    safe_log2,
    safe_rcp,
    safe_rsqrt,
    safe_sqrt,
    sin,
    square,
    sqrt,
)
from .solver import RidgeSolveOutputs, RidgeSolver, create_ridge_solver

__all__ = [
    "BinaryAst",
    "BinaryOp",
    "BinaryExpr",
    "CompiledGramKernels",
    "CompiledColumnKernels",
    "CompiledSingleAstColumnKernels",
    "ConstantExpr",
    "Expr",
    "FeatureExpr",
    "GramOutputs",
    "KernelsPerModule",
    "LeafBinding",
    "NUM_BINARY_OPS",
    "NUM_INPUTS",
    "NUM_UNARY_OPS",
    "RidgeSolveOutputs",
    "RidgeSolver",
    "UnaryExpr",
    "UnaryOp",
    "abs_f32",
    "ast_arrays",
    "as_expr",
    "binary_ast",
    "compile_column_kernels",
    "compile_gram_kernels",
    "compile_single_ast_column_kernels",
    "const",
    "create_ridge_solver",
    "cube",
    "cos",
    "deterministic_test_cohort",
    "ex2",
    "exp",
    "expression_to_ast_and_leaf_settings",
    "expressions_to_asts_and_settings",
    "f32_constant_word",
    "feature",
    "log10",
    "log2",
    "maximum",
    "minimum",
    "neg",
    "padding_ast",
    "random_ast_cohort",
    "rcp",
    "rsqrt",
    "safe_div",
    "safe_ex2",
    "safe_exp",
    "safe_log10",
    "safe_log2",
    "safe_rcp",
    "safe_rsqrt",
    "safe_sqrt",
    "sin",
    "square",
    "sqrt",
]
