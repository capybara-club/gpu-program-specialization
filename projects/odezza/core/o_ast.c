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
#include "o_odezza_internal.h"

#include <string.h>

#define O_MAX_AST_INSTRUCTIONS 2048u
#define O_MAX_AST_STACK_DEPTH 128u

typedef struct OAstStackValue {
    unsigned char direct_leaf;
} OAstStackValue;

static size_t o_ast_payload_size(uint8_t opcode) {
    switch ((OdezzaAstOpcode)opcode) {
    case ODEZZA_AST_STATE_F32:
    case ODEZZA_AST_CONSTANT_F32:
    case ODEZZA_AST_TOGGLE2_F32:
        return 1u;
    case ODEZZA_AST_TOGGLE4_F32:
        return 2u;
    case ODEZZA_AST_LITERAL_F32:
        return 4u;
    default:
        return 0u;
    }
}

static size_t o_ast_arity(uint8_t opcode) {
    switch ((OdezzaAstOpcode)opcode) {
    case ODEZZA_AST_NEG_F32:
    case ODEZZA_AST_SQRT_F32:
    case ODEZZA_AST_RCP_F32:
    case ODEZZA_AST_ABS_F32:
    case ODEZZA_AST_SIN_F32:
    case ODEZZA_AST_COS_F32:
    case ODEZZA_AST_EX2_F32:
    case ODEZZA_AST_LG2_F32:
    case ODEZZA_AST_RSQRT_F32:
    case ODEZZA_AST_TANH_F32:
    case ODEZZA_AST_EXP_F32:
    case ODEZZA_AST_LOG_F32:
        return 1u;
    case ODEZZA_AST_ADD_F32:
    case ODEZZA_AST_SUB_F32:
    case ODEZZA_AST_MUL_F32:
    case ODEZZA_AST_DIV_F32:
    case ODEZZA_AST_MIN_F32:
    case ODEZZA_AST_MAX_F32:
        return 2u;
    case ODEZZA_AST_FMA_F32:
        return 3u;
    default:
        return 0u;
    }
}

OdezzaResult odezza_validate_ast_program(
    OdezzaAstProgram program,
    size_t state_count,
    uint32_t constant_count,
    uint32_t toggle_bit_count,
    int allow_toggles,
    OdezzaAstAnalysis *analysis_ret
) {
    OAstStackValue stack[O_MAX_AST_STACK_DEPTH];
    OdezzaAstAnalysis analysis;
    size_t offset = 0u;
    size_t depth = 0u;
    int returned = 0;

    if (program.bytes == NULL || program.byte_count == 0u || state_count == 0u || state_count > 255u || constant_count > 255u ||
        toggle_bit_count > ODEZZA_MAX_TOGGLE_BITS ||
        (allow_toggles != 0 && allow_toggles != 1)
    ) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    memset(&analysis, 0, sizeof(analysis));
    while (offset < program.byte_count) {
        uint8_t opcode;
        size_t payload_size;
        size_t arity;
        size_t index;
        if (analysis.instruction_count == O_MAX_AST_INSTRUCTIONS) return ODEZZA_ERROR_AST;
        if (returned) return ODEZZA_ERROR_AST;
        opcode = program.bytes[offset++];
        payload_size = o_ast_payload_size(opcode);
        if (payload_size > program.byte_count - offset) return ODEZZA_ERROR_AST;
        ++analysis.instruction_count;

        switch ((OdezzaAstOpcode)opcode) {
        case ODEZZA_AST_STATE_F32:
            if ((size_t)program.bytes[offset] >= state_count) return ODEZZA_ERROR_AST;
            if (depth == O_MAX_AST_STACK_DEPTH) return ODEZZA_ERROR_AST;
            stack[depth++].direct_leaf = 1u;
            break;
        case ODEZZA_AST_CONSTANT_F32:
            if ((uint32_t)program.bytes[offset] >= constant_count) return ODEZZA_ERROR_AST;
            if (depth == O_MAX_AST_STACK_DEPTH) return ODEZZA_ERROR_AST;
            stack[depth++].direct_leaf = 1u;
            break;
        case ODEZZA_AST_LITERAL_F32: {
            uint32_t bits = (uint32_t)program.bytes[offset] | ((uint32_t)program.bytes[offset + 1u] << 8u) | ((uint32_t)program.bytes[offset + 2u] << 16u) |
                            ((uint32_t)program.bytes[offset + 3u] << 24u);
            if ((bits & 0x7f800000u) == 0x7f800000u) return ODEZZA_ERROR_AST;
        }
            if (depth == O_MAX_AST_STACK_DEPTH) return ODEZZA_ERROR_AST;
            stack[depth++].direct_leaf = 1u;
            break;
        case ODEZZA_AST_TOGGLE2_F32: {
            uint32_t bit = program.bytes[offset];
            if (!allow_toggles || bit >= toggle_bit_count || depth < 2u) return ODEZZA_ERROR_AST;
            if (!stack[depth - 1u].direct_leaf || !stack[depth - 2u].direct_leaf) return ODEZZA_ERROR_AST;
            --depth;
            stack[depth - 1u].direct_leaf = 0u;
            analysis.contains_toggle = 1;
            if (analysis.required_toggle_bit_count < bit + 1u) analysis.required_toggle_bit_count = bit + 1u;
            break;
        }
        case ODEZZA_AST_TOGGLE4_F32: {
            uint32_t bit0 = program.bytes[offset];
            uint32_t bit1 = program.bytes[offset + 1u];
            if (!allow_toggles || bit0 >= toggle_bit_count || bit1 >= toggle_bit_count || bit0 == bit1 || depth < 4u) return ODEZZA_ERROR_AST;
            for (index = depth - 4u; index < depth; ++index) {
                if (!stack[index].direct_leaf) return ODEZZA_ERROR_AST;
            }
            depth -= 3u;
            stack[depth - 1u].direct_leaf = 0u;
            analysis.contains_toggle = 1;
            if (analysis.required_toggle_bit_count < bit0 + 1u) analysis.required_toggle_bit_count = bit0 + 1u;
            if (analysis.required_toggle_bit_count < bit1 + 1u) analysis.required_toggle_bit_count = bit1 + 1u;
            break;
        }
        case ODEZZA_AST_RETURN_F32:
            if (depth != 1u || payload_size != 0u) return ODEZZA_ERROR_AST;
            returned = 1;
            break;
        default:
            arity = o_ast_arity(opcode);
            if (arity == 0u || depth < arity) return ODEZZA_ERROR_AST;
            depth = depth - arity + 1u;
            stack[depth - 1u].direct_leaf = 0u;
            break;
        }
        offset += payload_size;
        if (depth > analysis.maximum_stack_depth) analysis.maximum_stack_depth = depth;
    }
    if (!returned || depth != 1u) return ODEZZA_ERROR_AST;
    if (analysis_ret != NULL) *analysis_ret = analysis;
    return ODEZZA_SUCCESS;
}
