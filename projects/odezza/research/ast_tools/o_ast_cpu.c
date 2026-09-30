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
#include "o_ast_tools.h"

#include <math.h>
#include <string.h>

typedef struct OAstCpuValue {
    float value;
    int direct_leaf;
} OAstCpuValue;

static size_t o_cpu_arity(uint8_t opcode) {
    switch ((OAstToolOpcode)opcode) {
    case O_AST_NEG_F32:
    case O_AST_SQRT_F32:
    case O_AST_RCP_F32:
    case O_AST_ABS_F32:
    case O_AST_SIN_F32:
    case O_AST_COS_F32:
    case O_AST_EX2_F32:
    case O_AST_LG2_F32:
    case O_AST_RSQRT_F32:
    case O_AST_TANH_F32:
    case O_AST_EXP_F32:
    case O_AST_LOG_F32:
        return 1u;
    case O_AST_ADD_F32:
    case O_AST_SUB_F32:
    case O_AST_MUL_F32:
    case O_AST_DIV_F32:
    case O_AST_MIN_F32:
    case O_AST_MAX_F32:
        return 2u;
    case O_AST_FMA_F32:
        return 3u;
    default:
        return 0u;
    }
}

static float o_cpu_operation(uint8_t opcode, const OAstCpuValue *arguments) {
    switch ((OAstToolOpcode)opcode) {
    case O_AST_ADD_F32:
        return arguments[0].value + arguments[1].value;
    case O_AST_SUB_F32:
        return arguments[0].value - arguments[1].value;
    case O_AST_MUL_F32:
        return arguments[0].value * arguments[1].value;
    case O_AST_DIV_F32:
        return arguments[0].value / arguments[1].value;
    case O_AST_NEG_F32:
        return -arguments[0].value;
    case O_AST_SQRT_F32:
        return sqrtf(arguments[0].value);
    case O_AST_RCP_F32:
        return 1.0f / arguments[0].value;
    case O_AST_ABS_F32:
        return fabsf(arguments[0].value);
    case O_AST_MIN_F32:
        return fminf(arguments[0].value, arguments[1].value);
    case O_AST_MAX_F32:
        return fmaxf(arguments[0].value, arguments[1].value);
    case O_AST_FMA_F32:
        return fmaf(arguments[0].value, arguments[1].value, arguments[2].value);
    case O_AST_SIN_F32:
        return sinf(arguments[0].value);
    case O_AST_COS_F32:
        return cosf(arguments[0].value);
    case O_AST_EX2_F32:
        return exp2f(arguments[0].value);
    case O_AST_LG2_F32:
        return log2f(arguments[0].value);
    case O_AST_RSQRT_F32:
        return 1.0f / sqrtf(arguments[0].value);
    case O_AST_TANH_F32:
        return tanhf(arguments[0].value);
    case O_AST_EXP_F32:
        return expf(arguments[0].value);
    case O_AST_LOG_F32:
        return logf(arguments[0].value);
    default:
        return 0.0f;
    }
}

OAstToolResult o_ast_cpu_evaluate(
    OAstProgramView program,
    const float *states,
    size_t state_count,
    const float *constants,
    size_t constant_count,
    uint32_t toggle_bit_count,
    uint32_t permutation,
    float *value_ret
) {
    OAstCpuValue stack[O_AST_TOOL_MAX_STACK_DEPTH];
    size_t depth = 0u;
    size_t offset = 0u;
    size_t instruction_count = 0u;
    int returned = 0;
    if (value_ret == NULL || program.bytes == NULL || program.byte_count == 0u || state_count == 0u || state_count > 255u || constant_count > 255u ||
        toggle_bit_count > 30u || states == NULL || (constant_count != 0u && constants == NULL) ||
        permutation >= (UINT32_C(1) << toggle_bit_count)
    ) {
        return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    while (offset < program.byte_count) {
        uint8_t opcode;
        if (returned || instruction_count == O_AST_TOOL_MAX_INSTRUCTIONS) return O_AST_TOOL_ERROR_PROGRAM;
        opcode = program.bytes[offset++];
        ++instruction_count;
        switch ((OAstToolOpcode)opcode) {
        case O_AST_STATE_F32:
            if (offset == program.byte_count || program.bytes[offset] >= state_count || depth == O_AST_TOOL_MAX_STACK_DEPTH) {
                return O_AST_TOOL_ERROR_PROGRAM;
            }
            stack[depth].value = states[program.bytes[offset++]];
            stack[depth++].direct_leaf = 1;
            break;
        case O_AST_CONSTANT_F32:
            if (offset == program.byte_count || program.bytes[offset] >= constant_count || depth == O_AST_TOOL_MAX_STACK_DEPTH) {
                return O_AST_TOOL_ERROR_PROGRAM;
            }
            stack[depth].value = constants[program.bytes[offset++]];
            stack[depth++].direct_leaf = 1;
            break;
        case O_AST_LITERAL_F32: {
            uint32_t bits;
            if (program.byte_count - offset < 4u || depth == O_AST_TOOL_MAX_STACK_DEPTH) return O_AST_TOOL_ERROR_PROGRAM;
            bits = (uint32_t)program.bytes[offset] | ((uint32_t)program.bytes[offset + 1u] << 8u) |
                   ((uint32_t)program.bytes[offset + 2u] << 16u) | ((uint32_t)program.bytes[offset + 3u] << 24u);
            if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return O_AST_TOOL_ERROR_PROGRAM;
            memcpy(&stack[depth].value, &bits, sizeof(bits));
            stack[depth++].direct_leaf = 1;
            offset += 4u;
            break;
        }
        case O_AST_TOGGLE2_F32: {
            uint32_t bit;
            size_t choice;
            if (offset == program.byte_count || depth < 2u) return O_AST_TOOL_ERROR_PROGRAM;
            bit = program.bytes[offset++];
            if (bit >= toggle_bit_count || !stack[depth - 2u].direct_leaf || !stack[depth - 1u].direct_leaf) return O_AST_TOOL_ERROR_PROGRAM;
            choice = (permutation >> bit) & 1u;
            stack[depth - 2u] = stack[depth - 2u + choice];
            --depth;
            stack[depth - 1u].direct_leaf = 0;
            break;
        }
        case O_AST_TOGGLE4_F32: {
            uint32_t bit0;
            uint32_t bit1;
            size_t choice;
            size_t index;
            if (program.byte_count - offset < 2u || depth < 4u) return O_AST_TOOL_ERROR_PROGRAM;
            bit0 = program.bytes[offset++];
            bit1 = program.bytes[offset++];
            if (bit0 >= toggle_bit_count || bit1 >= toggle_bit_count || bit0 == bit1) return O_AST_TOOL_ERROR_PROGRAM;
            for (index = depth - 4u; index < depth; ++index) {
                if (!stack[index].direct_leaf) return O_AST_TOOL_ERROR_PROGRAM;
            }
            choice = ((permutation >> bit0) & 1u) | (((permutation >> bit1) & 1u) << 1u);
            stack[depth - 4u] = stack[depth - 4u + choice];
            depth -= 3u;
            stack[depth - 1u].direct_leaf = 0;
            break;
        }
        case O_AST_RETURN_F32:
            if (depth != 1u || offset != program.byte_count) return O_AST_TOOL_ERROR_PROGRAM;
            *value_ret = stack[0].value;
            returned = 1;
            break;
        default: {
            size_t arity = o_cpu_arity(opcode);
            if (arity == 0u || depth < arity) return O_AST_TOOL_ERROR_PROGRAM;
            stack[depth - arity].value = o_cpu_operation(opcode, stack + depth - arity);
            stack[depth - arity].direct_leaf = 0;
            depth = depth - arity + 1u;
            break;
        }
        }
    }
    return returned ? O_AST_TOOL_SUCCESS : O_AST_TOOL_ERROR_PROGRAM;
}
