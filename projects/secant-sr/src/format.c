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
#include "internal.h"

#include <stdarg.h>
#include <stdio.h>

static SecantSRResult
secant_sr_format_write(char* output, size_t output_size, size_t* offset, const char* format, ...) {
    va_list args;
    va_list measure_args;
    int required;

    va_start(args, format);
    va_copy(measure_args, args);
    required = vsnprintf(NULL, 0u, format, measure_args);
    va_end(measure_args);
    if (required < 0 || (size_t)required > SIZE_MAX - *offset) {
        va_end(args);
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    if (output != NULL) {
        if (*offset >= output_size || (size_t)required >= output_size - *offset ||
            vsnprintf(output + *offset, output_size - *offset, format, args) != required) {
            va_end(args);
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
        }
    }
    va_end(args);
    *offset += (size_t)required;
    return SECANT_SR_SUCCESS;
}

static const char*
secant_sr_unary_name(SecantAstInstructionType type) {
    switch (type) {
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32: return "neg";
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: return "sqrt";
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: return "rcp";
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32: return "abs";
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: return "sin";
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32: return "cos";
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: return "exp2";
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: return "log2";
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: return "rsqrt";
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: return "tanh";
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32: return "exp";
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32: return "log";
        default: return NULL;
    }
}

static const char*
secant_sr_binary_symbol(SecantAstInstructionType type) {
    switch (type) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32: return "+";
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32: return "-";
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32: return "*";
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32: return "/";
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32: return "min";
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32: return "max";
        default: return NULL;
    }
}

static const SecantSRRoutine*
secant_sr_node_routine_get(
    const SecantSRIndividual* individual,
    const SecantAstInstruction* instruction,
    uint8_t arity
) {
    const size_t routine_idx = secant_ast_index_get(instruction);

    if (individual->routines == NULL || routine_idx >= individual->num_routines ||
        individual->routines[routine_idx].arity != arity ||
        individual->routines[routine_idx].name == NULL) {
        return NULL;
    }
    return individual->routines + routine_idx;
}

static SecantSRResult
secant_sr_node_format(
    const SecantSRIndividual* individual,
    size_t node_idx,
    char* output,
    size_t output_size,
    size_t* offset
) {
    const SecantSRNodeInfo* node = individual->nodes + node_idx;
    const SecantAstInstruction* instruction = individual->program + node->instruction_offset;
    const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);

    if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
        return secant_sr_format_write(output, output_size, offset, "x%u", (unsigned)secant_ast_index_get(instruction));
    }
    if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32) {
        return secant_sr_format_write(output, output_size, offset, "column[%u]", (unsigned)secant_ast_index_get(instruction));
    }
    if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32) {
        return secant_sr_format_write(output, output_size, offset, "constant[%u]", (unsigned)secant_ast_index_get(instruction));
    }
    if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
        return secant_sr_format_write(output, output_size, offset, "leaf[%u]", (unsigned)secant_ast_index_get(instruction));
    }
    if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
        return secant_sr_format_write(output, output_size, offset, "%.9g", secant_ast_constant_f32_get(instruction));
    }
    if (node->arity == 1u) {
        const SecantSRRoutine* routine = type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32
            ? secant_sr_node_routine_get(individual, instruction, 1u)
            : NULL;
        const char* name = routine != NULL ? routine->name : secant_sr_unary_name(type);

        if (name == NULL || node_idx == 0u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        _SECANT_SR_CHECK_RET(secant_sr_format_write(output, output_size, offset, "%s(", name));
        _SECANT_SR_CHECK_RET(secant_sr_node_format(individual, node_idx - 1u, output, output_size, offset));
        return secant_sr_format_write(output, output_size, offset, ")");
    }
    if (node->arity == 2u) {
        const size_t right_idx = node_idx - 1u;
        const size_t left_idx = individual->nodes[right_idx].subtree_first_node - 1u;
        const SecantSRRoutine* routine = type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32
            ? secant_sr_node_routine_get(individual, instruction, 2u)
            : NULL;
        const char* symbol = routine != NULL ? routine->name : secant_sr_binary_symbol(type);

        if (symbol == NULL || node_idx < 2u || individual->nodes[right_idx].subtree_first_node == 0u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (routine != NULL || type == SECANT_AST_INSTRUCTION_TYPE_MIN_F32 ||
            type == SECANT_AST_INSTRUCTION_TYPE_MAX_F32) {
            _SECANT_SR_CHECK_RET(secant_sr_format_write(output, output_size, offset, "%s(", symbol));
            _SECANT_SR_CHECK_RET(secant_sr_node_format(individual, left_idx, output, output_size, offset));
            _SECANT_SR_CHECK_RET(secant_sr_format_write(output, output_size, offset, ", "));
            _SECANT_SR_CHECK_RET(secant_sr_node_format(individual, right_idx, output, output_size, offset));
            return secant_sr_format_write(output, output_size, offset, ")");
        }
        _SECANT_SR_CHECK_RET(secant_sr_format_write(output, output_size, offset, "("));
        _SECANT_SR_CHECK_RET(secant_sr_node_format(individual, left_idx, output, output_size, offset));
        _SECANT_SR_CHECK_RET(secant_sr_format_write(output, output_size, offset, " %s ", symbol));
        _SECANT_SR_CHECK_RET(secant_sr_node_format(individual, right_idx, output, output_size, offset));
        return secant_sr_format_write(output, output_size, offset, ")");
    }
    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
}

SecantSRResult
secant_sr_individual_format(
    const SecantSRIndividual* individual,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
    size_t required = 0u;

    if (individual == NULL || individual->program == NULL || individual->nodes == NULL ||
        individual->num_nodes == 0u || required_size_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_node_format(
        individual, individual->num_nodes - 1u, output, output_size, &required));
    if (required == SIZE_MAX) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    ++required;
    *required_size_ret = required;
    if (output != NULL) {
        if (output_size < required) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
        }
        output[required - 1u] = '\0';
    }
    return SECANT_SR_SUCCESS;
}
