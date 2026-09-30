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
#ifndef BINARY_AST_TO_STACK_PTX_H_INCLUDE
#define BINARY_AST_TO_STACK_PTX_H_INCLUDE

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define BINARY_AST_TO_STACK_PTX_PUBLIC_DEC extern "C"
#define BINARY_AST_TO_STACK_PTX_PUBLIC_DEF extern "C"
#else
#define BINARY_AST_TO_STACK_PTX_PUBLIC_DEC extern
#define BINARY_AST_TO_STACK_PTX_PUBLIC_DEF
#endif

#include "implicit_sindy.h"
#include "stack_ptx.h"

typedef enum {
    BINARY_AST_SUCCESS = 0,
    BINARY_AST_ERROR_INVALID_VALUE = 1,
    BINARY_AST_ERROR_INSUFFICIENT_BUFFER = 2
} BinaryAstResult;

BINARY_AST_TO_STACK_PTX_PUBLIC_DEC const StackPtxInstruction* binary_ast_stack_ptx_routines[];
BINARY_AST_TO_STACK_PTX_PUBLIC_DEC const size_t binary_ast_stack_ptx_num_routines;

BINARY_AST_TO_STACK_PTX_PUBLIC_DEC BinaryAstResult
binary_ast_stack_ptx_instruction_count(
    const BinaryAST* ast,
    size_t* count_out
);

BINARY_AST_TO_STACK_PTX_PUBLIC_DEC BinaryAstResult
binary_ast_write_stack_ptx(
    const BinaryAST* ast,
    StackPtxInstruction* out,
    size_t out_capacity,
    size_t* count_out
);

#endif /* BINARY_AST_TO_STACK_PTX_H_INCLUDE */

#ifdef BINARY_AST_TO_STACK_PTX_IMPLEMENTATION
#ifndef BINARY_AST_TO_STACK_PTX_IMPLEMENTATION_ONCE
#define BINARY_AST_TO_STACK_PTX_IMPLEMENTATION_ONCE

#include <stack_ptx_descriptions.h>

typedef enum {
    BINARY_AST_STACK_PTX_ROUTINE_IDENTITY = 0,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SQUARE = 1,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_CUBE = 2,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_EXP = 3,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_LOG10 = 4,
    BINARY_AST_STACK_PTX_ROUTINE_KEEP_LEFT = 5,
    BINARY_AST_STACK_PTX_ROUTINE_KEEP_RIGHT = 6,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_RCP = 7,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_SQRT = 8,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_RSQRT = 9,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_EX2 = 10,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_EXP = 11,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_LOG2 = 12,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_LOG10 = 13,
    BINARY_AST_STACK_PTX_ROUTINE_BINARY_SAFE_DIV = 14,
    BINARY_AST_STACK_PTX_ROUTINE_UNARY_ZERO = 15,
    BINARY_AST_STACK_PTX_ROUTINE_NUM_ENUMS = 16
} BinaryAstStackPtxRoutine;

enum {
    BINARY_AST_INTERNAL_NUM_AST_REGS = 15
};

#define BINARY_AST_STACK_PTX_LOG2_E_F32 1.4426950408889634f
#define BINARY_AST_STACK_PTX_LOG10_2_F32 0.3010299956639812f
#define BINARY_AST_STACK_PTX_SAFE_EPS_F32 1.0e-6f
#define BINARY_AST_STACK_PTX_SAFE_EX2_LO_F32 -126.0f
#define BINARY_AST_STACK_PTX_SAFE_EX2_HI_F32 126.0f
#define BINARY_AST_STACK_PTX_SAFE_EXP_LO_F32 -80.0f
#define BINARY_AST_STACK_PTX_SAFE_EXP_HI_F32 80.0f

#define BINARY_AST_STACK_PTX_ENCODE_MUL_F32 stack_ptx_encode_ptx_instruction_mul_ftz_f32
#define BINARY_AST_STACK_PTX_ENCODE_DIV_F32 stack_ptx_encode_ptx_instruction_div_approx_ftz_f32

static const StackPtxInstruction binary_ast_stack_ptx_routine_identity[] = {
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_square[] = {
    stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32),
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_cube[] = {
    stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32),
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_exp[] = {
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_LOG2_E_F32),
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_log10[] = {
    stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_LOG10_2_F32),
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_keep_left[] = {
    stack_ptx_encode_meta_constant(1u),
    stack_ptx_encode_meta_replace(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_keep_right[] = {
    stack_ptx_encode_meta_constant(1u),
    stack_ptx_encode_meta_drop(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_rcp[] = {
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EPS_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_ptx_instruction_rcp_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_sqrt[] = {
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EPS_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_rsqrt[] = {
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EPS_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_ptx_instruction_rsqrt_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_ex2[] = {
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EX2_LO_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EX2_HI_F32),
    stack_ptx_encode_ptx_instruction_min_ftz_f32,
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_exp[] = {
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EXP_LO_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EXP_HI_F32),
    stack_ptx_encode_ptx_instruction_min_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_LOG2_E_F32),
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_log2[] = {
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EPS_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_safe_log10[] = {
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EPS_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_LOG10_2_F32),
    BINARY_AST_STACK_PTX_ENCODE_MUL_F32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_binary_safe_div[] = {
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 0u),
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(BINARY_AST_STACK_PTX_SAFE_EPS_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_load(0u),
    BINARY_AST_STACK_PTX_ENCODE_DIV_F32,
    stack_ptx_encode_return
};

static const StackPtxInstruction binary_ast_stack_ptx_routine_unary_zero[] = {
    stack_ptx_encode_meta_drop(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_constant_f32(0.0f),
    stack_ptx_encode_return
};

BINARY_AST_TO_STACK_PTX_PUBLIC_DEF const StackPtxInstruction* binary_ast_stack_ptx_routines[] = {
    binary_ast_stack_ptx_routine_identity,
    binary_ast_stack_ptx_routine_unary_square,
    binary_ast_stack_ptx_routine_unary_cube,
    binary_ast_stack_ptx_routine_unary_exp,
    binary_ast_stack_ptx_routine_unary_log10,
    binary_ast_stack_ptx_routine_keep_left,
    binary_ast_stack_ptx_routine_keep_right,
    binary_ast_stack_ptx_routine_unary_safe_rcp,
    binary_ast_stack_ptx_routine_unary_safe_sqrt,
    binary_ast_stack_ptx_routine_unary_safe_rsqrt,
    binary_ast_stack_ptx_routine_unary_safe_ex2,
    binary_ast_stack_ptx_routine_unary_safe_exp,
    binary_ast_stack_ptx_routine_unary_safe_log2,
    binary_ast_stack_ptx_routine_unary_safe_log10,
    binary_ast_stack_ptx_routine_binary_safe_div,
    binary_ast_stack_ptx_routine_unary_zero
};

BINARY_AST_TO_STACK_PTX_PUBLIC_DEF const size_t binary_ast_stack_ptx_num_routines =
    BINARY_AST_STACK_PTX_ROUTINE_NUM_ENUMS;

typedef struct {
    StackPtxInstruction* out;
    size_t capacity;
    size_t count;
} BinaryAstToStackPtxWriter;

static StackPtxInstruction
binary_ast_stack_ptx_input(
    StackPtxIdx idx
) {
    StackPtxInstruction instruction = stack_ptx_encode_input(idx);
    return instruction;
}

static StackPtxInstruction
binary_ast_stack_ptx_return(void)
{
    StackPtxInstruction instruction = stack_ptx_encode_return;
    return instruction;
}

static StackPtxInstruction
binary_ast_stack_ptx_routine(
    StackPtxIdx idx
) {
    StackPtxInstruction instruction = stack_ptx_encode_routine(idx);
    return instruction;
}

static BinaryAstResult
binary_ast_stack_ptx_emit(
    BinaryAstToStackPtxWriter* writer,
    StackPtxInstruction instruction
) {
    if (writer == NULL) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    if (writer->out != NULL) {
        if (writer->count >= writer->capacity) {
            return BINARY_AST_ERROR_INSUFFICIENT_BUFFER;
        }
        writer->out[writer->count] = instruction;
    }
    writer->count++;
    return BINARY_AST_SUCCESS;
}

static BinaryAstResult
binary_ast_stack_ptx_emit_routine(
    BinaryAstToStackPtxWriter* writer,
    BinaryAstStackPtxRoutine routine
) {
    if (routine >= BINARY_AST_STACK_PTX_ROUTINE_NUM_ENUMS) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    return binary_ast_stack_ptx_emit(
        writer,
        binary_ast_stack_ptx_routine((StackPtxIdx)routine)
    );
}

static BinaryAstResult
binary_ast_stack_ptx_binary_instruction(
    BinaryAstBinaryOp op,
    StackPtxInstruction* instruction
) {
    if (instruction == NULL || op >= BINARY_AST_BINARY_NUM_ENUMS) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    switch (op) {
        case BINARY_AST_BINARY_ADD_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_add_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_BINARY_SUB_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_sub_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_BINARY_MUL_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_mul_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_BINARY_DIV_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_div_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_BINARY_MIN_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_min_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_BINARY_MAX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_max_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_BINARY_KEEP_LEFT:
        case BINARY_AST_BINARY_KEEP_RIGHT:
        case BINARY_AST_BINARY_SAFE_DIV_F32:
        case BINARY_AST_BINARY_NUM_ENUMS:
            break;
    }
    return BINARY_AST_ERROR_INVALID_VALUE;
}

static BinaryAstResult
binary_ast_stack_ptx_unary_instruction(
    BinaryAstUnaryOp op,
    StackPtxInstruction* instruction
) {
    if (instruction == NULL || op >= BINARY_AST_UNARY_NUM_ENUMS) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    switch (op) {
        case BINARY_AST_UNARY_NEG_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_neg_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_ABS_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_abs_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_RCP_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_rcp_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_rsqrt_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_SIN_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_sin_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_COS_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_cos_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_EX2_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32:
            *instruction = stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32;
            return BINARY_AST_SUCCESS;
        case BINARY_AST_UNARY_IDENTITY:
        case BINARY_AST_UNARY_SQUARE_F32:
        case BINARY_AST_UNARY_CUBE_F32:
        case BINARY_AST_UNARY_EXP_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_SAFE_RCP_F32:
        case BINARY_AST_UNARY_SAFE_SQRT_F32:
        case BINARY_AST_UNARY_SAFE_RSQRT_F32:
        case BINARY_AST_UNARY_SAFE_EX2_F32:
        case BINARY_AST_UNARY_SAFE_EXP_F32:
        case BINARY_AST_UNARY_SAFE_LOG2_F32:
        case BINARY_AST_UNARY_SAFE_LOG10_F32:
        case BINARY_AST_UNARY_ZERO_F32:
        case BINARY_AST_UNARY_NUM_ENUMS:
            break;
    }
    return BINARY_AST_ERROR_INVALID_VALUE;
}

static BinaryAstResult
binary_ast_stack_ptx_emit_unary(
    BinaryAstToStackPtxWriter* writer,
    const BinaryAST* ast,
    size_t unary_idx
) {
    BinaryAstUnaryOp op;
    StackPtxInstruction instruction;
    BinaryAstResult result;

    if (ast == NULL || unary_idx >= BINARY_AST_NUM_UNARY_OPS) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    op = ast->unary[unary_idx];
    switch (op) {
        case BINARY_AST_UNARY_IDENTITY:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_IDENTITY
            );
        case BINARY_AST_UNARY_SQUARE_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SQUARE
            );
        case BINARY_AST_UNARY_CUBE_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_CUBE
            );
        case BINARY_AST_UNARY_EXP_APPROX_FTZ_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_EXP
            );
        case BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_LOG10
            );
        case BINARY_AST_UNARY_SAFE_RCP_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_RCP
            );
        case BINARY_AST_UNARY_SAFE_SQRT_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_SQRT
            );
        case BINARY_AST_UNARY_SAFE_RSQRT_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_RSQRT
            );
        case BINARY_AST_UNARY_SAFE_EX2_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_EX2
            );
        case BINARY_AST_UNARY_SAFE_EXP_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_EXP
            );
        case BINARY_AST_UNARY_SAFE_LOG2_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_LOG2
            );
        case BINARY_AST_UNARY_SAFE_LOG10_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_SAFE_LOG10
            );
        case BINARY_AST_UNARY_ZERO_F32:
            return binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_UNARY_ZERO
            );
        case BINARY_AST_UNARY_NEG_FTZ_F32:
        case BINARY_AST_UNARY_ABS_FTZ_F32:
        case BINARY_AST_UNARY_RCP_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_SIN_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_COS_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_EX2_APPROX_FTZ_F32:
        case BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32:
            result = binary_ast_stack_ptx_unary_instruction(op, &instruction);
            if (result != BINARY_AST_SUCCESS) return result;
            return binary_ast_stack_ptx_emit(writer, instruction);
        case BINARY_AST_UNARY_NUM_ENUMS:
            break;
    }
    return BINARY_AST_ERROR_INVALID_VALUE;
}

static BinaryAstResult
binary_ast_stack_ptx_emit_leaf(
    const BinaryAST* ast,
    size_t leaf_idx,
    BinaryAstToStackPtxWriter* writer
) {
    StackPtxInstruction instruction;
    if (ast == NULL || leaf_idx >= BINARY_AST_NUM_INPUTS) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    instruction = binary_ast_stack_ptx_input((StackPtxIdx)leaf_idx);
    return binary_ast_stack_ptx_emit(writer, instruction);
}

static BinaryAstResult
binary_ast_stack_ptx_node_children(
    size_t ast_idx,
    size_t* binary_idx,
    size_t* lhs,
    size_t* rhs
) {
    if (binary_idx == NULL || lhs == NULL || rhs == NULL) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    if (ast_idx >= 8u && ast_idx < 12u) {
        const size_t local_idx = ast_idx - 8u;
        *binary_idx = local_idx;
        *lhs = local_idx * 2u;
        *rhs = *lhs + 1u;
        return BINARY_AST_SUCCESS;
    }
    if (ast_idx >= 12u && ast_idx < 14u) {
        const size_t local_idx = ast_idx - 12u;
        *binary_idx = 4u + local_idx;
        *lhs = 8u + local_idx * 2u;
        *rhs = *lhs + 1u;
        return BINARY_AST_SUCCESS;
    }
    if (ast_idx == 14u) {
        *binary_idx = 6u;
        *lhs = 12u;
        *rhs = 13u;
        return BINARY_AST_SUCCESS;
    }
    return BINARY_AST_ERROR_INVALID_VALUE;
}

static BinaryAstResult
binary_ast_stack_ptx_emit_node(
    const BinaryAST* ast,
    size_t ast_idx,
    BinaryAstToStackPtxWriter* writer
) {
    BinaryAstResult result;
    if (ast == NULL || writer == NULL || ast_idx >= BINARY_AST_INTERNAL_NUM_AST_REGS) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    if (ast_idx < BINARY_AST_NUM_INPUTS) {
        result = binary_ast_stack_ptx_emit_leaf(ast, ast_idx, writer);
        if (result != BINARY_AST_SUCCESS) return result;
        return binary_ast_stack_ptx_emit_unary(writer, ast, ast_idx);
    } else {
        size_t binary_idx = 0u;
        size_t lhs = 0u;
        size_t rhs = 0u;
        BinaryAstBinaryOp op;
        StackPtxInstruction instruction;

        result = binary_ast_stack_ptx_node_children(ast_idx, &binary_idx, &lhs, &rhs);
        if (result != BINARY_AST_SUCCESS) return result;
        op = ast->binary[binary_idx];
        if (op >= BINARY_AST_BINARY_NUM_ENUMS) return BINARY_AST_ERROR_INVALID_VALUE;

        result = binary_ast_stack_ptx_emit_node(ast, rhs, writer);
        if (result != BINARY_AST_SUCCESS) return result;
        result = binary_ast_stack_ptx_emit_node(ast, lhs, writer);
        if (result != BINARY_AST_SUCCESS) return result;

        if (op == BINARY_AST_BINARY_KEEP_LEFT) {
            result = binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_KEEP_LEFT
            );
        } else if (op == BINARY_AST_BINARY_KEEP_RIGHT) {
            result = binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_KEEP_RIGHT
            );
        } else if (op == BINARY_AST_BINARY_SAFE_DIV_F32) {
            result = binary_ast_stack_ptx_emit_routine(
                writer,
                BINARY_AST_STACK_PTX_ROUTINE_BINARY_SAFE_DIV
            );
        } else {
            result = binary_ast_stack_ptx_binary_instruction(op, &instruction);
            if (result != BINARY_AST_SUCCESS) return result;
            result = binary_ast_stack_ptx_emit(writer, instruction);
        }
        if (result != BINARY_AST_SUCCESS) return result;
        return binary_ast_stack_ptx_emit_unary(writer, ast, ast_idx);
    }
}

BINARY_AST_TO_STACK_PTX_PUBLIC_DEF
BinaryAstResult
binary_ast_stack_ptx_instruction_count(
    const BinaryAST* ast,
    size_t* count_out
) {
    BinaryAstToStackPtxWriter writer;
    BinaryAstResult result;

    if (ast == NULL || count_out == NULL) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    writer.out = NULL;
    writer.capacity = 0u;
    writer.count = 0u;
    result = binary_ast_stack_ptx_emit_node(
        ast,
        BINARY_AST_INTERNAL_NUM_AST_REGS - 1u,
        &writer
    );
    if (result != BINARY_AST_SUCCESS) return result;
    result = binary_ast_stack_ptx_emit(&writer, binary_ast_stack_ptx_return());
    if (result != BINARY_AST_SUCCESS) return result;
    *count_out = writer.count;
    return BINARY_AST_SUCCESS;
}

BINARY_AST_TO_STACK_PTX_PUBLIC_DEF
BinaryAstResult
binary_ast_write_stack_ptx(
    const BinaryAST* ast,
    StackPtxInstruction* out,
    size_t out_capacity,
    size_t* count_out
) {
    BinaryAstToStackPtxWriter writer;
    BinaryAstResult result;

    if (ast == NULL || out == NULL || count_out == NULL) {
        return BINARY_AST_ERROR_INVALID_VALUE;
    }
    writer.out = out;
    writer.capacity = out_capacity;
    writer.count = 0u;
    result = binary_ast_stack_ptx_emit_node(
        ast,
        BINARY_AST_INTERNAL_NUM_AST_REGS - 1u,
        &writer
    );
    if (result != BINARY_AST_SUCCESS) return result;
    result = binary_ast_stack_ptx_emit(&writer, binary_ast_stack_ptx_return());
    if (result != BINARY_AST_SUCCESS) return result;
    *count_out = writer.count;
    return BINARY_AST_SUCCESS;
}

#undef BINARY_AST_STACK_PTX_ENCODE_MUL_F32
#undef BINARY_AST_STACK_PTX_ENCODE_DIV_F32
#undef BINARY_AST_STACK_PTX_LOG2_E_F32
#undef BINARY_AST_STACK_PTX_LOG10_2_F32
#undef BINARY_AST_STACK_PTX_SAFE_EPS_F32
#undef BINARY_AST_STACK_PTX_SAFE_EX2_LO_F32
#undef BINARY_AST_STACK_PTX_SAFE_EX2_HI_F32
#undef BINARY_AST_STACK_PTX_SAFE_EXP_LO_F32
#undef BINARY_AST_STACK_PTX_SAFE_EXP_HI_F32

#endif /* BINARY_AST_TO_STACK_PTX_IMPLEMENTATION_ONCE */
#endif /* BINARY_AST_TO_STACK_PTX_IMPLEMENTATION */
