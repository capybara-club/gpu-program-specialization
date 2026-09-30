/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef CUSR_AST_ROUTINES_H_INCLUDED
#define CUSR_AST_ROUTINES_H_INCLUDED

#include "cusr_ast.h"

#define CUSR_AST_ROUTINES_COUNT 11u

#ifndef CUSR_AST_ROUTINES_EPSILON_BITS
#define CUSR_AST_ROUTINES_EPSILON_BITS 0x322bcc77u /* 1.0e-8f */
#endif

#define CUSR_AST_ROUTINES_LOG2E_BITS 0x3fb8aa3bu
#define CUSR_AST_ROUTINES_LN2_BITS 0x3f317218u
#define CUSR_AST_ROUTINES_LOG10_2_BITS 0x3e9a209bu

typedef enum CusrAstRoutine {
    CUSR_AST_ROUTINE_EXP = 0,
    CUSR_AST_ROUTINE_SAFE_LOG = 1,
    CUSR_AST_ROUTINE_SAFE_LOG10 = 2,
    CUSR_AST_ROUTINE_SAFE_POW = 3,
    CUSR_AST_ROUTINE_SAFE_SQRT = 4,
    CUSR_AST_ROUTINE_SAFE_RSQRT = 5,
    CUSR_AST_ROUTINE_SAFE_DIV = 6,
    CUSR_AST_ROUTINE_SAFE_RCP = 7,
    CUSR_AST_ROUTINE_LOG = 8,
    CUSR_AST_ROUTINE_LOG10 = 9,
    CUSR_AST_ROUTINE_POW = 10,
    CUSR_AST_ROUTINE_NUM_ENUMS = 11
} CusrAstRoutine;

#ifdef __cplusplus
#if __cplusplus >= 202002L
#define CUSR_AST_ROUTINES_CONST constexpr
#else
#define CUSR_AST_ROUTINES_CONST const
#endif
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_exp = cusr_ast_encode_routine(CUSR_AST_ROUTINE_EXP, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_log = cusr_ast_encode_routine(CUSR_AST_ROUTINE_LOG, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_log = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_LOG, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_log10 = cusr_ast_encode_routine(CUSR_AST_ROUTINE_LOG10, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_log10 = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_LOG10, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_pow = cusr_ast_encode_routine(CUSR_AST_ROUTINE_POW, 2u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_pow = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_POW, 2u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_sqrt = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_SQRT, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_rsqrt = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_RSQRT, 1u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_div = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_DIV, 2u);
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_encode_safe_rcp = cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_RCP, 1u);
#else
#define CUSR_AST_ROUTINES_CONST const
#define cusr_ast_encode_exp cusr_ast_encode_routine(CUSR_AST_ROUTINE_EXP, 1u)
#define cusr_ast_encode_log cusr_ast_encode_routine(CUSR_AST_ROUTINE_LOG, 1u)
#define cusr_ast_encode_safe_log cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_LOG, 1u)
#define cusr_ast_encode_log10 cusr_ast_encode_routine(CUSR_AST_ROUTINE_LOG10, 1u)
#define cusr_ast_encode_safe_log10 cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_LOG10, 1u)
#define cusr_ast_encode_pow cusr_ast_encode_routine(CUSR_AST_ROUTINE_POW, 2u)
#define cusr_ast_encode_safe_pow cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_POW, 2u)
#define cusr_ast_encode_safe_sqrt cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_SQRT, 1u)
#define cusr_ast_encode_safe_rsqrt cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_RSQRT, 1u)
#define cusr_ast_encode_safe_div cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_DIV, 2u)
#define cusr_ast_encode_safe_rcp cusr_ast_encode_routine(CUSR_AST_ROUTINE_SAFE_RCP, 1u)
#endif

/* exp(x) = ex2(x * log2(e)) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_exp[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_LOG2E_BITS),
    cusr_ast_encode_mul,
    cusr_ast_encode_ex2,
    cusr_ast_encode_return
};

/* log(x) = lg2(abs(x) + eps) * ln(2) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_log[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_lg2,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_LN2_BITS),
    cusr_ast_encode_mul,
    cusr_ast_encode_return
};

/* log10(x) = lg2(abs(x) + eps) * log10(2) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_log10[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_lg2,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_LOG10_2_BITS),
    cusr_ast_encode_mul,
    cusr_ast_encode_return
};

/* pow(x, y) = ex2(y * lg2(abs(x) + eps)) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_pow[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_lg2,
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_mul,
    cusr_ast_encode_ex2,
    cusr_ast_encode_return
};

/* sqrt(x) = sqrt(abs(x) + eps) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_sqrt[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_sqrt,
    cusr_ast_encode_return
};

/* rsqrt(x) = rsqrt(abs(x) + eps) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_rsqrt[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_rsqrt,
    cusr_ast_encode_return
};

/* safe_div(x, y) = x * y / (y * y + eps) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_div[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_mul,
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_mul,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_div,
    cusr_ast_encode_return
};

/* safe_rcp(x) = x / (x * x + eps) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_safe_rcp[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_mul,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_EPSILON_BITS),
    cusr_ast_encode_add,
    cusr_ast_encode_div,
    cusr_ast_encode_return
};

/* log(x) = lg2(x) * ln(2) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_log[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_lg2,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_LN2_BITS),
    cusr_ast_encode_mul,
    cusr_ast_encode_return
};

/* log10(x) = lg2(x) * log10(2) */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_log10[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_lg2,
    cusr_ast_encode_constant_bits(CUSR_AST_ROUTINES_LOG10_2_BITS),
    cusr_ast_encode_mul,
    cusr_ast_encode_return
};

/* pow(x, y) = ex2(y * lg2(x)), for positive x */
static CUSR_AST_ROUTINES_CONST CusrAstInstruction cusr_ast_routine_pow[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_lg2,
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_mul,
    cusr_ast_encode_ex2,
    cusr_ast_encode_return
};

#ifdef __cplusplus
static CUSR_AST_ROUTINES_CONST const CusrAstInstruction* cusr_ast_routines[] = {
#else
static const CusrAstInstruction* const cusr_ast_routines[] = {
#endif
    cusr_ast_routine_exp,
    cusr_ast_routine_safe_log,
    cusr_ast_routine_safe_log10,
    cusr_ast_routine_safe_pow,
    cusr_ast_routine_safe_sqrt,
    cusr_ast_routine_safe_rsqrt,
    cusr_ast_routine_safe_div,
    cusr_ast_routine_safe_rcp,
    cusr_ast_routine_log,
    cusr_ast_routine_log10,
    cusr_ast_routine_pow
};

#undef CUSR_AST_ROUTINES_CONST

#endif /* CUSR_AST_ROUTINES_H_INCLUDED */
