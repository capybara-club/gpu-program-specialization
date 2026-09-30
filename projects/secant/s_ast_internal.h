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
#ifndef SECANT_AST_INTERNAL_H_INCLUDED
#define SECANT_AST_INTERNAL_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SECANT_INTERNAL_AST_DYNAMIC_ARITY UINT8_MAX

static inline uint8_t
secant_internal_ast_instruction_num_args(
    SecantAstInstructionType instruction_type
) {
    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_S32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_U32:
            return SECANT_INTERNAL_AST_DYNAMIC_ARITY;
        case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RETURN_S32:
        case SECANT_AST_INSTRUCTION_TYPE_NEG_S32:
        case SECANT_AST_INSTRUCTION_TYPE_ABS_S32:
        case SECANT_AST_INSTRUCTION_TYPE_RETURN_U32:
        case SECANT_AST_INSTRUCTION_TYPE_NOT_U32:
            return 1u;
        case SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
        case SECANT_AST_INSTRUCTION_TYPE_ADD_S32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_S32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_S32:
        case SECANT_AST_INSTRUCTION_TYPE_DIV_S32:
        case SECANT_AST_INSTRUCTION_TYPE_MIN_S32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_S32:
        case SECANT_AST_INSTRUCTION_TYPE_ADD_U32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_U32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_U32:
        case SECANT_AST_INSTRUCTION_TYPE_DIV_U32:
        case SECANT_AST_INSTRUCTION_TYPE_MIN_U32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_U32:
        case SECANT_AST_INSTRUCTION_TYPE_AND_U32:
        case SECANT_AST_INSTRUCTION_TYPE_OR_U32:
        case SECANT_AST_INSTRUCTION_TYPE_XOR_U32:
        case SECANT_AST_INSTRUCTION_TYPE_SHL_U32:
        case SECANT_AST_INSTRUCTION_TYPE_SHR_U32:
            return 2u;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
        case SECANT_AST_INSTRUCTION_TYPE_FMA_S32:
        case SECANT_AST_INSTRUCTION_TYPE_FMA_U32:
            return 3u;
        case SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32:
            return 4u;
        default:
            return 0u;
    }
}

static inline int
secant_internal_ast_instruction_is_f32(
    SecantAstInstructionType instruction_type
) {
    return instruction_type == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32 ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32 ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32 ||
        (instruction_type >= SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 &&
         instruction_type <= SECANT_AST_INSTRUCTION_TYPE_TANH_F32) ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_EXP_F32 ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_LOG_F32;
}

static inline SecantResult
secant_internal_ast_instruction_write(
    SecantAstInstructionType instruction_type,
    SecantAstIdx idx,
    uint32_t bits,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* output_offset
) {
    size_t instruction_size;
    size_t offset;

    /* This legacy scalar writer has only one payload word. */
    if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32)
        return SECANT_ERROR_INVALID_VALUE;
    if (output == NULL || output_offset == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    offset = *output_offset;
    if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 ||
        instruction_type == SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32) {
        if (idx >= SECANT_AST_MAX_INPUTS) {
            return SECANT_ERROR_INVALID_VALUE;
        }
    }
    if ((instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ||
         instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32) &&
        (idx >= SECANT_AST_MAX_TOGGLE_BITS ||
         (instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32 &&
          (bits >= SECANT_AST_MAX_TOGGLE_BITS || bits == idx)))) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    {
        SecantAstInstruction encoded[SECANT_AST_MAX_INSTRUCTION_BYTES];

        encoded[0] = (SecantAstInstruction)instruction_type;
        instruction_size = secant_ast_instruction_size_get(encoded);
    }
    if (instruction_size == 0u) {
        return SECANT_ERROR_BAD_PROGRAM;
    }
    if (offset > output_size || instruction_size > output_size - offset) {
        return SECANT_ERROR_INSUFFICIENT_BUFFER;
    }
    output[offset] = (SecantAstInstruction)instruction_type;
    if (instruction_size == 2u) {
        output[offset + 1u] = idx;
    } else if (instruction_size == 3u) {
        output[offset + 1u] = idx;
        output[offset + 2u] = (uint8_t)bits;
    } else if (instruction_size == 5u) {
        output[offset + 1u] = (uint8_t)(bits >> 0u);
        output[offset + 2u] = (uint8_t)(bits >> 8u);
        output[offset + 3u] = (uint8_t)(bits >> 16u);
        output[offset + 4u] = (uint8_t)(bits >> 24u);
    }
    *output_offset = offset + instruction_size;
    return SECANT_SUCCESS;
}

static inline SecantResult
secant_internal_ast_constant_f32_write(
    float value,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* output_offset
) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return secant_internal_ast_instruction_write(
        SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32,
        0u,
        bits,
        output,
        output_size,
        output_offset);
}

#endif /* SECANT_AST_INTERNAL_H_INCLUDED */
