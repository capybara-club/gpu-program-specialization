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

uint8_t
secant_sr_instruction_arity(SecantAstInstructionType instruction_type) {
    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
            return 0u;
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
            return 1u;
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            return 2u;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            return 3u;
        default:
            return UINT8_MAX;
    }
}

uint16_t
secant_sr_instruction_complexity(SecantAstInstructionType instruction_type) {
    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
            return 3u;
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
            return 4u;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            return 2u;
        default:
            return 1u;
    }
}

int
secant_sr_instruction_is_unary(SecantAstInstructionType instruction_type) {
    return secant_sr_instruction_arity(instruction_type) == 1u;
}

int
secant_sr_instruction_is_binary(SecantAstInstructionType instruction_type) {
    return secant_sr_instruction_arity(instruction_type) == 2u;
}

static int
secant_sr_instruction_is_transcendental(SecantAstInstructionType instruction_type) {
    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
            return 1;
        default:
            return 0;
    }
}

static SecantSRResult
secant_sr_instruction_metadata_get(
    const SecantAstInstruction* instruction,
    const SecantSRRoutine* routines,
    size_t num_routines,
    uint8_t* arity_ret,
    uint16_t* complexity_ret,
    uint8_t* num_transcendental_operations_ret
) {
    const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);

    if (type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
        const size_t routine_idx = secant_ast_index_get(instruction);
        const SecantSRRoutine* routine;

        if (routines == NULL || routine_idx >= num_routines) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        routine = routines + routine_idx;
        if (routine->program == NULL || routine->name == NULL || routine->arity == 0u ||
            routine->arity > 2u || routine->complexity == 0u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        *arity_ret = routine->arity;
        *complexity_ret = routine->complexity;
        *num_transcendental_operations_ret = routine->num_transcendental_operations;
        return SECANT_SR_SUCCESS;
    }
    *arity_ret = secant_sr_instruction_arity(type);
    if (*arity_ret == UINT8_MAX) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    *complexity_ret = secant_sr_instruction_complexity(type);
    *num_transcendental_operations_ret = (uint8_t)secant_sr_instruction_is_transcendental(type);
    return SECANT_SR_SUCCESS;
}

static uint16_t
secant_sr_u64_popcount(uint64_t value) {
    uint16_t count = 0u;

    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

static void
secant_sr_program_features_update(
    SecantSRProgramFeatures* features,
    const SecantAstInstruction* instruction,
    SecantAstInstructionType type,
    uint8_t arity,
    uint8_t num_transcendental_operations
) {
    if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
        const uint8_t input_idx = secant_ast_index_get(instruction);

        features->static_column_mask[input_idx / 64u] |= UINT64_C(1) << (input_idx % 64u);
        ++features->num_static_column_occurrences;
    } else if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
        ++features->num_constants;
    }
    if (arity == 1u) {
        ++features->num_unary_operations;
    } else if (arity == 2u) {
        ++features->num_binary_operations;
    } else if (arity == 3u) {
        ++features->num_ternary_operations;
    }
    if (arity != 0u && type >= SECANT_AST_INSTRUCTION_TYPE_NONE &&
        type < SECANT_AST_INSTRUCTION_TYPE_ONE_PAST_LAST) {
        features->operation_mask |= UINT64_C(1) << (type - SECANT_AST_INSTRUCTION_TYPE_NONE);
    }
    features->num_transcendental_operations = (uint16_t)(
        features->num_transcendental_operations + num_transcendental_operations);
}

SecantSRResult
secant_sr_program_features_get(
    const SecantAstInstruction* program,
    size_t program_bytes,
    const SecantSRRoutine* routines,
    size_t num_routines,
    SecantSRProgramFeatures* features_ret
) {
    size_t stack_size = 0u;
    size_t offset = 0u;

    if (program == NULL || program_bytes == 0u || features_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    memset(features_ret, 0, sizeof(*features_ret));
    while (offset < program_bytes) {
        const SecantAstInstruction* instruction = program + offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);
        uint8_t arity;
        uint16_t instruction_complexity;
        uint8_t num_transcendental_operations;

        if (instruction_size == 0u || instruction_size > program_bytes - offset) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            if (stack_size != 1u || offset + instruction_size != program_bytes) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            features_ret->num_unique_static_columns = (uint16_t)(
                secant_sr_u64_popcount(features_ret->static_column_mask[0]) +
                secant_sr_u64_popcount(features_ret->static_column_mask[1]));
            return SECANT_SR_SUCCESS;
        }

        _SECANT_SR_CHECK_RET(secant_sr_instruction_metadata_get(
            instruction,
            routines,
            num_routines,
            &arity,
            &instruction_complexity,
            &num_transcendental_operations));
        (void)instruction_complexity;
        if (arity > stack_size ||
            stack_size - arity >= SECANT_AST_MAX_STACK_DEPTH) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 &&
            secant_ast_index_get(instruction) >= SECANT_AST_MAX_INPUTS) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if ((type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 ||
             type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 ||
             type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) &&
            secant_ast_index_get(instruction) >= SECANT_AST_MAX_DYNAMIC_LEAVES) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }

        if (features_ret->num_transcendental_operations >
            UINT16_MAX - num_transcendental_operations) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        secant_sr_program_features_update(
            features_ret,
            instruction,
            type,
            arity,
            num_transcendental_operations);
        stack_size -= arity;
        ++stack_size;
        offset += instruction_size;
    }
    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
}

static uint64_t
secant_sr_fingerprint(const SecantAstInstruction* program, size_t program_bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t idx;

    for (idx = 0u; idx < program_bytes; ++idx) {
        hash ^= program[idx];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

SecantSRResult
secant_sr_program_annotate(
    const SecantAstInstruction* program,
    size_t program_limit,
    const SecantSRRoutine* routines,
    size_t num_routines,
    size_t max_depth,
    uint32_t max_complexity,
    SecantSRNodeInfo* nodes,
    size_t node_capacity,
    size_t* program_bytes_ret,
    size_t* num_nodes_ret,
    size_t* num_leaves_ret,
    size_t* num_constants_ret,
    uint32_t* complexity_ret,
    uint16_t* depth_ret,
    uint64_t* fingerprint_ret
) {
    uint16_t stack[SECANT_AST_MAX_STACK_DEPTH];
    size_t stack_size = 0u;
    size_t offset = 0u;
    size_t num_nodes = 0u;
    size_t num_leaves = 0u;
    size_t num_constants = 0u;

    if (program == NULL || program_limit == 0u || nodes == NULL || node_capacity == 0u ||
        program_bytes_ret == NULL || num_nodes_ret == NULL || num_leaves_ret == NULL || num_constants_ret == NULL ||
        complexity_ret == NULL || depth_ret == NULL || fingerprint_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }

    while (offset < program_limit) {
        const SecantAstInstruction* instruction = program + offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);
        uint8_t arity;
        uint16_t instruction_complexity;
        uint8_t num_transcendental_operations;
        SecantSRNodeInfo* node;
        uint32_t complexity;
        uint16_t depth;
        uint16_t subtree_first_node;
        uint32_t subtree_offset;
        size_t arg_idx;

        if (instruction_size == 0u || instruction_size > program_limit - offset) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            const size_t program_bytes = offset + instruction_size;

            if (stack_size != 1u || num_nodes == 0u) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            *program_bytes_ret = program_bytes;
            *num_nodes_ret = num_nodes;
            *num_leaves_ret = num_leaves;
            *num_constants_ret = num_constants;
            *complexity_ret = nodes[stack[0]].complexity;
            *depth_ret = nodes[stack[0]].depth;
            *fingerprint_ret = secant_sr_fingerprint(program, program_bytes);
            return SECANT_SR_SUCCESS;
        }

        _SECANT_SR_CHECK_RET(secant_sr_instruction_metadata_get(
            instruction,
            routines,
            num_routines,
            &arity,
            &instruction_complexity,
            &num_transcendental_operations));
        (void)num_transcendental_operations;
        if (arity > stack_size || num_nodes >= node_capacity ||
            num_nodes >= UINT16_MAX || stack_size - arity >= SECANT_AST_MAX_STACK_DEPTH) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 &&
            secant_ast_index_get(instruction) >= SECANT_AST_MAX_INPUTS) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if ((type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 ||
             type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 ||
             type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) &&
            secant_ast_index_get(instruction) >= SECANT_AST_MAX_DYNAMIC_LEAVES) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }

        complexity = instruction_complexity;
        depth = 1u;
        subtree_first_node = (uint16_t)num_nodes;
        subtree_offset = (uint32_t)offset;
        for (arg_idx = stack_size - arity; arg_idx < stack_size; ++arg_idx) {
            const SecantSRNodeInfo* child = nodes + stack[arg_idx];

            if (complexity > UINT32_MAX - child->complexity) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
            }
            complexity += child->complexity;
            if (child->depth >= depth) {
                depth = (uint16_t)(child->depth + 1u);
            }
        }
        if (arity != 0u) {
            const SecantSRNodeInfo* first_child = nodes + stack[stack_size - arity];

            subtree_first_node = first_child->subtree_first_node;
            subtree_offset = first_child->subtree_offset;
        }
        if (complexity > max_complexity || complexity > UINT16_MAX || depth > max_depth ||
            offset > UINT32_MAX || offset + instruction_size - subtree_offset > UINT32_MAX) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }

        node = nodes + num_nodes;
        node->instruction_offset = (uint32_t)offset;
        node->subtree_offset = subtree_offset;
        node->subtree_bytes = (uint32_t)(offset + instruction_size - subtree_offset);
        node->subtree_first_node = subtree_first_node;
        node->subtree_nodes = (uint16_t)(num_nodes - subtree_first_node + 1u);
        node->depth = depth;
        node->complexity = (uint16_t)complexity;
        node->instruction_size = (uint8_t)instruction_size;
        node->arity = arity;

        if (arity == 0u) {
            ++num_leaves;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            ++num_constants;
        }

        stack_size -= arity;
        stack[stack_size++] = (uint16_t)num_nodes;
        ++num_nodes;
        offset += instruction_size;
    }
    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
}

SecantSRResult
secant_sr_program_leaf_replace_copy(
    const SecantAstInstruction* program,
    size_t program_bytes,
    size_t leaf_instruction_offset,
    SecantAstInstructionType replacement_type,
    SecantAstIdx replacement_idx,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* required_size_ret
) {
    size_t offset = 0u;
    size_t replaced_size = 0u;
    int found = 0;
    int returned = 0;

    if (program == NULL || program_bytes == 0u || required_size_ret == NULL ||
        replacement_idx >= SECANT_AST_MAX_DYNAMIC_LEAVES ||
        (replacement_type != SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 &&
         replacement_type != SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 &&
         replacement_type != SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    while (offset < program_bytes) {
        const SecantAstInstruction* instruction = program + offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u || instruction_size > program_bytes - offset || returned) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (offset == leaf_instruction_offset) {
            if (type != SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 &&
                type != SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 &&
                type != SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 &&
                type != SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32 &&
                type != SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            found = 1;
            replaced_size = instruction_size;
        }
        offset += instruction_size;
        returned = type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
    }
    if (!found || !returned || replaced_size < 2u) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    *required_size_ret = program_bytes - replaced_size + 2u;
    if (output == NULL) {
        return SECANT_SR_SUCCESS;
    }
    if (output_size < *required_size_ret) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    {
        const uintptr_t source_begin = (uintptr_t)program;
        const uintptr_t output_begin = (uintptr_t)output;

        if (program_bytes > UINTPTR_MAX - source_begin ||
            *required_size_ret > UINTPTR_MAX - output_begin ||
            (output_begin < source_begin + program_bytes &&
             source_begin < output_begin + *required_size_ret)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }
    memcpy(output, program, leaf_instruction_offset);
    output[leaf_instruction_offset] = (SecantAstInstruction)replacement_type;
    output[leaf_instruction_offset + 1u] = replacement_idx;
    memcpy(
        output + leaf_instruction_offset + 2u,
        program + leaf_instruction_offset + replaced_size,
        program_bytes - leaf_instruction_offset - replaced_size);
    return SECANT_SR_SUCCESS;
}
