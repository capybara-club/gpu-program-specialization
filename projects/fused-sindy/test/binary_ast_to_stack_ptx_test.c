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
#include "binary_ast_to_stack_ptx.h"
#include <stack_ptx_descriptions.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const StackPtxCompilerInfo test_compiler_info = {
    256u,
    256u,
    128u,
    8u,
    16u
};

static const StackPtxRegister test_registers[] = {
    { "_x1", STACK_PTX_STACK_TYPE_F32 },
    { "_x2", STACK_PTX_STACK_TYPE_F32 },
    { "_x3", STACK_PTX_STACK_TYPE_F32 },
    { "_x4", STACK_PTX_STACK_TYPE_F32 },
    { "_x5", STACK_PTX_STACK_TYPE_F32 },
    { "_x6", STACK_PTX_STACK_TYPE_F32 },
    { "_x7", STACK_PTX_STACK_TYPE_F32 },
    { "_x8", STACK_PTX_STACK_TYPE_F32 },
    { "_x0", STACK_PTX_STACK_TYPE_F32 },
};

static int
test_check(int condition, const char* label)
{
    if (condition) return 1;
    fprintf(stderr, "check failed: %s\n", label);
    return 0;
}

static void
test_make_add_tree(BinaryAST* ast)
{
    size_t i;
    memset(ast, 0, sizeof(*ast));
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        ast->binary[i] = BINARY_AST_BINARY_ADD_FTZ_F32;
    }
}

static void
test_make_extended_tree(BinaryAST* ast)
{
    memset(ast, 0, sizeof(*ast));
    ast->unary[0] = BINARY_AST_UNARY_NEG_FTZ_F32;
    ast->unary[1] = BINARY_AST_UNARY_ABS_FTZ_F32;
    ast->unary[2] = BINARY_AST_UNARY_RCP_APPROX_FTZ_F32;
    ast->unary[3] = BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32;
    ast->unary[4] = BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32;
    ast->unary[5] = BINARY_AST_UNARY_SIN_APPROX_FTZ_F32;
    ast->unary[6] = BINARY_AST_UNARY_COS_APPROX_FTZ_F32;
    ast->unary[7] = BINARY_AST_UNARY_EX2_APPROX_FTZ_F32;
    ast->unary[8] = BINARY_AST_UNARY_EXP_APPROX_FTZ_F32;
    ast->unary[9] = BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32;
    ast->unary[10] = BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32;
    ast->unary[11] = BINARY_AST_UNARY_SQUARE_F32;
    ast->unary[12] = BINARY_AST_UNARY_CUBE_F32;
    ast->unary[13] = BINARY_AST_UNARY_SAFE_RCP_F32;
    ast->unary[14] = BINARY_AST_UNARY_SAFE_EXP_F32;
    ast->binary[0] = BINARY_AST_BINARY_DIV_APPROX_FTZ_F32;
    ast->binary[1] = BINARY_AST_BINARY_MIN_FTZ_F32;
    ast->binary[2] = BINARY_AST_BINARY_MAX_FTZ_F32;
    ast->binary[3] = BINARY_AST_BINARY_SUB_FTZ_F32;
    ast->binary[4] = BINARY_AST_BINARY_MUL_FTZ_F32;
    ast->binary[5] = BINARY_AST_BINARY_ADD_FTZ_F32;
    ast->binary[6] = BINARY_AST_BINARY_SAFE_DIV_F32;
}

static size_t
test_count_instruction_type(
    const StackPtxInstruction* program,
    size_t count,
    StackPtxInstructionType instruction_type
) {
    size_t i;
    size_t matches = 0u;
    for (i = 0u; i < count; ++i) {
        if (program[i].instruction_type == instruction_type) {
            ++matches;
        }
    }
    return matches;
}

static int
test_compile_add_tree(void)
{
    BinaryAST feature;
    StackPtxInstruction program[32];
    size_t count = 0u;
    size_t measured = 0u;
    size_t workspace_bytes = 0u;
    void* workspace = NULL;
    char ptx[4096];
    size_t ptx_bytes = 0u;
    const size_t requests[] = { 8u };
    BinaryAstResult br;
    StackPtxResult sr;
    int ok = 0;

    test_make_add_tree(&feature);
    br = binary_ast_stack_ptx_instruction_count(&feature, &measured);
    if (!test_check(br == BINARY_AST_SUCCESS, "measure result")) return 0;
    if (!test_check(measured == 31u, "measure count")) return 0;

    br = binary_ast_write_stack_ptx(&feature, program, measured - 1u, &count);
    if (!test_check(br == BINARY_AST_ERROR_INSUFFICIENT_BUFFER, "insufficient buffer")) return 0;

    br = binary_ast_write_stack_ptx(&feature, program, sizeof(program) / sizeof(program[0]), &count);
    if (!test_check(br == BINARY_AST_SUCCESS, "write result")) return 0;
    if (!test_check(count == measured, "write count")) return 0;
    if (!test_check(
            test_count_instruction_type(program, count, STACK_PTX_INSTRUCTION_TYPE_ROUTINE) == 15u,
            "routine call count")) return 0;
    if (!test_check(
            test_count_instruction_type(program, count, STACK_PTX_INSTRUCTION_TYPE_PTX) == 7u,
            "ptx instruction count")) return 0;
    if (!test_check(program[count - 1u].instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN, "return")) return 0;

    sr = stack_ptx_compile_workspace_size(&test_compiler_info, &stack_ptx_stack_info, &workspace_bytes);
    if (!test_check(sr == STACK_PTX_SUCCESS, "stack_ptx workspace size")) return 0;
    workspace = malloc(workspace_bytes);
    if (!test_check(workspace != NULL, "workspace malloc")) return 0;

    sr = stack_ptx_compile(
        &test_compiler_info,
        &stack_ptx_stack_info,
        program,
        test_registers,
        sizeof(test_registers) / sizeof(test_registers[0]),
        binary_ast_stack_ptx_routines,
        binary_ast_stack_ptx_num_routines,
        requests,
        sizeof(requests) / sizeof(requests[0]),
        128u,
        workspace,
        workspace_bytes,
        ptx,
        sizeof(ptx),
        &ptx_bytes
    );
    if (!test_check(sr == STACK_PTX_SUCCESS, "stack_ptx compile")) goto cleanup;
    if (!test_check(ptx_bytes != 0u, "stack_ptx ptx bytes")) goto cleanup;
    ptx[ptx_bytes < sizeof(ptx) ? ptx_bytes : sizeof(ptx) - 1u] = '\0';
    if (!test_check(strstr(ptx, "add.ftz.f32") != NULL, "stack_ptx emitted add")) goto cleanup;
    ok = 1;

cleanup:
    free(workspace);
    return ok;
}

static int
test_compile_extended_tree(void)
{
    BinaryAST feature;
    StackPtxInstruction program[96];
    size_t count = 0u;
    size_t measured = 0u;
    size_t workspace_bytes = 0u;
    void* workspace = NULL;
    char ptx[8192];
    size_t ptx_bytes = 0u;
    const size_t requests[] = { 8u };
    BinaryAstResult br;
    StackPtxResult sr;
    int ok = 0;

    test_make_extended_tree(&feature);
    br = binary_ast_stack_ptx_instruction_count(&feature, &measured);
    if (!test_check(br == BINARY_AST_SUCCESS, "extended measure result")) return 0;

    br = binary_ast_write_stack_ptx(&feature, program, sizeof(program) / sizeof(program[0]), &count);
    if (!test_check(br == BINARY_AST_SUCCESS, "extended write result")) return 0;
    if (!test_check(count == measured, "extended write count")) return 0;

    sr = stack_ptx_compile_workspace_size(&test_compiler_info, &stack_ptx_stack_info, &workspace_bytes);
    if (!test_check(sr == STACK_PTX_SUCCESS, "extended stack_ptx workspace size")) return 0;
    workspace = malloc(workspace_bytes);
    if (!test_check(workspace != NULL, "extended workspace malloc")) return 0;

    sr = stack_ptx_compile(
        &test_compiler_info,
        &stack_ptx_stack_info,
        program,
        test_registers,
        sizeof(test_registers) / sizeof(test_registers[0]),
        binary_ast_stack_ptx_routines,
        binary_ast_stack_ptx_num_routines,
        requests,
        sizeof(requests) / sizeof(requests[0]),
        128u,
        workspace,
        workspace_bytes,
        ptx,
        sizeof(ptx),
        &ptx_bytes
    );
    if (!test_check(sr == STACK_PTX_SUCCESS, "extended stack_ptx compile")) goto cleanup;
    if (!test_check(ptx_bytes != 0u, "extended stack_ptx ptx bytes")) goto cleanup;
    ptx[ptx_bytes < sizeof(ptx) ? ptx_bytes : sizeof(ptx) - 1u] = '\0';
    if (!test_check(strstr(ptx, "neg.ftz.f32") != NULL, "extended emitted neg")) goto cleanup;
    if (!test_check(strstr(ptx, "abs.ftz.f32") != NULL, "extended emitted abs")) goto cleanup;
    if (!test_check(strstr(ptx, "rcp.approx.ftz.f32") != NULL, "extended emitted rcp")) goto cleanup;
    if (!test_check(strstr(ptx, "sqrt.approx.ftz.f32") != NULL, "extended emitted sqrt")) goto cleanup;
    if (!test_check(strstr(ptx, "rsqrt.approx.ftz.f32") != NULL, "extended emitted rsqrt")) goto cleanup;
    if (!test_check(strstr(ptx, "sin.approx.ftz.f32") != NULL, "extended emitted sin")) goto cleanup;
    if (!test_check(strstr(ptx, "cos.approx.ftz.f32") != NULL, "extended emitted cos")) goto cleanup;
    if (!test_check(strstr(ptx, "ex2.approx.ftz.f32") != NULL, "extended emitted ex2")) goto cleanup;
    if (!test_check(strstr(ptx, "lg2.approx.ftz.f32") != NULL, "extended emitted lg2")) goto cleanup;
    if (!test_check(strstr(ptx, "div.approx.ftz.f32") != NULL, "extended emitted div")) goto cleanup;
    if (!test_check(strstr(ptx, "min.ftz.f32") != NULL, "extended emitted min")) goto cleanup;
    if (!test_check(strstr(ptx, "max.ftz.f32") != NULL, "extended emitted max")) goto cleanup;
    ok = 1;

cleanup:
    free(workspace);
    return ok;
}

static int
test_keep_left(void)
{
    BinaryAST feature;
    size_t measured = 0u;
    BinaryAstResult br;

    test_make_add_tree(&feature);
    feature.binary[BINARY_AST_NUM_BINARY_OPS - 1u] = BINARY_AST_BINARY_KEEP_LEFT;
    br = binary_ast_stack_ptx_instruction_count(&feature, &measured);
    if (!test_check(br == BINARY_AST_SUCCESS, "keep-left measure")) return 0;
    if (!test_check(measured == 31u, "keep-left count")) return 0;
    return 1;
}

int
main(void)
{
    if (!test_compile_add_tree()) return 1;
    if (!test_compile_extended_tree()) return 1;
    if (!test_keep_left()) return 1;
    printf("PASS binary_ast_to_stack_ptx_test\n");
    return 0;
}
