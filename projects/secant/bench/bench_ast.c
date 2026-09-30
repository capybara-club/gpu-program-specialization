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
#include "bench_ast.h"
#include "s_ast_internal.h"

#include <stdint.h>
#include <string.h>

#define SECANT_PORTABLE_ALU_INCLUDE_AST
#include "corpus/portable_alu_v1.h"

static const SecantAstInstruction secant_bench_routine_safe_div[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(
        SECANT_F32_BITS_ONE_HUNDREDTH),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction secant_bench_routine_safe_sqrt[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32_bits(
        SECANT_F32_BITS_ONE_HUNDREDTH),
    secant_ast_encode_add_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction secant_bench_routine_safe_rsqrt[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32_bits(
        SECANT_F32_BITS_ONE_HUNDREDTH),
    secant_ast_encode_add_f32,
    secant_ast_encode_rsqrt_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const secant_bench_routines[] = {
    secant_bench_routine_safe_div,
    secant_bench_routine_safe_sqrt,
    secant_bench_routine_safe_rsqrt
};

static const char* const secant_bench_routine_names[] = {
    "safe_div",
    "safe_sqrt",
    "safe_rsqrt"
};

uint32_t
secant_bench_ast_hash32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

const char*
secant_bench_ast_corpus_name(SecantBenchAstMode mode) {
    if (mode == SECANT_BENCH_AST_MODE_ALU) {
        return SECANT_PORTABLE_ALU_CORPUS_NAME;
    }
    return "secant_generated_v1";
}

const char*
secant_bench_ast_corpus_definition_hash(SecantBenchAstMode mode) {
    if (mode == SECANT_BENCH_AST_MODE_ALU) {
        return SECANT_PORTABLE_ALU_CORPUS_HASH;
    }
    return "dynamic";
}

static SecantAstInstructionType
secant_bench_ast_binary(uint32_t selector, SecantBenchAstMode mode) {
    if (mode == SECANT_BENCH_AST_MODE_MUFU && (selector & 7u) == 0u) {
        return SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32;
    }

    switch (selector & 3u) {
        case 0u: return SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
        case 1u: return SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
        case 2u: return SECANT_AST_INSTRUCTION_TYPE_MIN_F32;
        default: return SECANT_AST_INSTRUCTION_TYPE_MAX_F32;
    }
}

static void
secant_bench_ast_emit(
    SecantAstInstruction* program,
    size_t* count,
    SecantAstInstructionType instruction_type,
    SecantAstIdx idx,
    uint32_t bits
) {
    (void)secant_internal_ast_instruction_write(
        instruction_type,
        idx,
        bits,
        program,
        SECANT_BENCH_AST_PROGRAM_STRIDE,
        count);
}

static void
secant_bench_ast_emit_discriminator(
    SecantAstInstruction* program,
    size_t* count,
    size_t kernel_ast_idx,
    uint32_t discriminator_class,
    SecantBenchCSEMode cse_mode,
    uint32_t* discriminator_site_idx
) {
    const uint32_t site_id = cse_mode == SECANT_BENCH_CSE_SHARED
        ? discriminator_class + 1u
        : (uint32_t)kernel_ast_idx * (SECANT_BENCH_AST_NUM_LEAVES * 2u - 1u) +
            *discriminator_site_idx + 1u;
    const uint32_t mantissa =
        (site_id * 0x5bd1e995u) & 0x007fffffu;

    secant_bench_ast_emit(
        program,
        count,
        SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32,
        0u,
        0x3f000000u | mantissa);
    secant_bench_ast_emit(
        program,
        count,
        (discriminator_class & 1u) == 0u
            ? SECANT_AST_INSTRUCTION_TYPE_ADD_F32
            : SECANT_AST_INSTRUCTION_TYPE_MUL_F32,
        0u,
        0u);
    *discriminator_site_idx += 1u;
}

static void
secant_bench_ast_emit_alu_program(
    SecantAstInstruction* program,
    size_t* count,
    size_t global_ast_idx,
    size_t kernel_ast_idx,
    size_t num_inputs,
    SecantBenchCSEMode cse_mode
) {
    const SecantAstInstruction* source =
        secant_portable_alu_asts[global_ast_idx % SECANT_PORTABLE_ALU_NUM_CASES];
    size_t source_offset = 0u;
    uint32_t discriminator_site_idx = 0u;
    size_t instruction_idx;

    if (cse_mode == SECANT_BENCH_CSE_DEFAULT) {
        memcpy(
            program,
            source,
            SECANT_PORTABLE_ALU_PROGRAM_BYTES * sizeof(*program));
        for (instruction_idx = 0u;
             instruction_idx < SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS;
             ++instruction_idx) {
            SecantAstInstruction* instruction = program + source_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);

            if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
                instruction[1] = (uint8_t)(secant_ast_index_get(instruction) % num_inputs);
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
            source_offset += secant_ast_instruction_size_get(instruction);
        }
        return;
    }
    for (instruction_idx = 0u;
         instruction_idx < SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction = source + source_offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);
        const SecantAstIdx input_idx = type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32
            ? (SecantAstIdx)(secant_ast_index_get(instruction) % num_inputs)
            : 0u;

        secant_bench_ast_emit(
            program,
            count,
            type,
            input_idx,
            0u);
        if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
            secant_bench_ast_emit_discriminator(
                program,
                count,
                kernel_ast_idx,
                input_idx,
                cse_mode,
                &discriminator_site_idx);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            break;
        }
        source_offset += instruction_size;
    }
}

static void
secant_bench_ast_emit_leaf(
    SecantAstInstruction* program,
    size_t* count,
    uint32_t ast_key,
    uint32_t leaf_idx,
    size_t num_inputs,
    SecantBenchAstMode mode,
    size_t kernel_ast_idx,
    SecantBenchCSEMode cse_mode,
    uint32_t* mufu_site_idx
) {
    const uint32_t selector = secant_bench_ast_hash32(
        ast_key + leaf_idx * 0x9e3779b9u);
    const uint32_t mufu_class = (selector >> 8) % 5u;

    secant_bench_ast_emit(
        program,
        count,
        SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
        (SecantAstIdx)(selector % (uint32_t)num_inputs),
        0u);
    if (mode != SECANT_BENCH_AST_MODE_MUFU) {
        return;
    }
    secant_bench_ast_emit_discriminator(
        program,
        count,
        kernel_ast_idx,
        mufu_class,
        cse_mode,
        mufu_site_idx);

    switch (mufu_class) {
        case 0u:
            secant_bench_ast_emit(
                program,
                count,
                SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
                0u,
                0u);
            break;
        case 1u:
            secant_bench_ast_emit(
                program,
                count,
                SECANT_AST_INSTRUCTION_TYPE_COS_F32,
                0u,
                0u);
            break;
        case 2u:
            secant_bench_ast_emit(
                program,
                count,
                SECANT_AST_INSTRUCTION_TYPE_EX2_F32,
                0u,
                0u);
            break;
        case 3u:
            secant_bench_ast_emit(
                program,
                count,
                SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32,
                SECANT_BENCH_ROUTINE_SAFE_SQRT,
                0u);
            break;
        default:
            secant_bench_ast_emit(
                program,
                count,
                SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32,
                SECANT_BENCH_ROUTINE_SAFE_RSQRT,
                0u);
            break;
    }
}

static void
secant_bench_ast_emit_tree(
    SecantAstInstruction* program,
    size_t* count,
    uint32_t ast_key,
    uint32_t first_leaf,
    uint32_t num_leaves,
    size_t num_inputs,
    SecantBenchAstMode mode,
    size_t kernel_ast_idx,
    SecantBenchCSEMode cse_mode,
    uint32_t* mufu_site_idx
) {
    const uint32_t left_leaves = num_leaves / 2u;
    const uint32_t right_leaves = num_leaves - left_leaves;

    if (num_leaves == 1u) {
        secant_bench_ast_emit_leaf(
            program,
            count,
            ast_key,
            first_leaf,
            num_inputs,
            mode,
            kernel_ast_idx,
            cse_mode,
            mufu_site_idx);
        return;
    }

    secant_bench_ast_emit_tree(
        program,
        count,
        ast_key,
        first_leaf,
        left_leaves,
        num_inputs,
        mode,
        kernel_ast_idx,
        cse_mode,
        mufu_site_idx);
    secant_bench_ast_emit_tree(
        program,
        count,
        ast_key,
        first_leaf + left_leaves,
        right_leaves,
        num_inputs,
        mode,
        kernel_ast_idx,
        cse_mode,
        mufu_site_idx);
    {
        const SecantAstInstructionType instruction_type = secant_bench_ast_binary(
            secant_bench_ast_hash32(
                ast_key ^
                first_leaf * 0x85ebca6bu ^
                num_leaves * 0xc2b2ae35u),
            mode);

        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
            secant_bench_ast_emit_discriminator(
                program,
                count,
                kernel_ast_idx,
                5u,
                cse_mode,
                mufu_site_idx);
        }
        secant_bench_ast_emit(
            program,
            count,
            instruction_type,
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32
                ? SECANT_BENCH_ROUTINE_SAFE_DIV
                : 0u,
            0u);
    }
}

static void
secant_bench_ast_make_program(
    SecantAstInstruction* program,
    size_t global_ast_idx,
    size_t kernel_ast_idx,
    size_t num_inputs,
    uint32_t seed,
    SecantBenchAstMode mode,
    SecantBenchCSEMode cse_mode
) {
    const uint32_t ast_key = secant_bench_ast_hash32(
        (uint32_t)global_ast_idx ^ seed);
    size_t count = 0u;
    uint32_t mufu_site_idx = 0u;

    memset(
        program,
        0,
        SECANT_BENCH_AST_PROGRAM_STRIDE * sizeof(*program));
    if (mode == SECANT_BENCH_AST_MODE_ALU) {
        secant_bench_ast_emit_alu_program(
            program,
            &count,
            global_ast_idx,
            kernel_ast_idx,
            num_inputs,
            cse_mode);
        return;
    }
    if (mode == SECANT_BENCH_AST_MODE_SIMPLE) {
        const float constant = 0.25f + (float)(ast_key & 7u) * 0.125f;
        uint32_t constant_bits;

        memcpy(&constant_bits, &constant, sizeof(constant_bits));
        secant_bench_ast_emit(
            program,
            &count,
            SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
            (SecantAstIdx)(ast_key % (uint32_t)num_inputs),
            0u);
        secant_bench_ast_emit(
            program,
            &count,
            SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32,
            0u,
            constant_bits);
        secant_bench_ast_emit(
            program,
            &count,
            (ast_key & 1u) != 0u
                ? SECANT_AST_INSTRUCTION_TYPE_ADD_F32
                : SECANT_AST_INSTRUCTION_TYPE_MUL_F32,
            0u,
            0u);
        secant_bench_ast_emit(
            program,
            &count,
            SECANT_AST_INSTRUCTION_TYPE_RETURN_F32,
            0u,
            0u);
        return;
    }

    secant_bench_ast_emit_tree(
        program,
        &count,
        ast_key,
        0u,
        SECANT_BENCH_AST_NUM_LEAVES,
        num_inputs,
        mode,
        kernel_ast_idx,
        cse_mode,
        &mufu_site_idx);
    secant_bench_ast_emit(
        program,
        &count,
        SECANT_AST_INSTRUCTION_TYPE_RETURN_F32,
        0u,
        0u);
}

int
secant_bench_ast_storage_sizes(
    size_t num_modules,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t* program_bytes_ret,
    size_t* pointer_bytes_ret
) {
    size_t num_asts;

    if (program_bytes_ret == NULL || pointer_bytes_ret == NULL ||
        (num_kernels != 0u &&
            asts_per_kernel > SIZE_MAX / num_kernels)) {
        return 0;
    }
    num_asts = num_kernels * asts_per_kernel;
    if (num_modules != 0u && num_asts > SIZE_MAX / num_modules) {
        return 0;
    }
    num_asts *= num_modules;
    if (num_asts > SIZE_MAX / SECANT_BENCH_AST_PROGRAM_STRIDE ||
        num_asts * SECANT_BENCH_AST_PROGRAM_STRIDE >
            SIZE_MAX / sizeof(SecantAstInstruction) ||
        num_asts > SIZE_MAX / sizeof(SecantAstInstruction*)) {
        return 0;
    }

    *program_bytes_ret = num_asts *
        SECANT_BENCH_AST_PROGRAM_STRIDE *
        sizeof(SecantAstInstruction);
    *pointer_bytes_ret = num_asts * sizeof(SecantAstInstruction*);
    return 1;
}

void
secant_bench_ast_fill(
    size_t num_modules,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    uint32_t seed,
    SecantBenchAstMode mode,
    SecantBenchCSEMode cse_mode,
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    const size_t num_asts = num_modules * num_kernels * asts_per_kernel;
    size_t ast_idx;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program = programs + ast_idx * SECANT_BENCH_AST_PROGRAM_STRIDE;

        secant_bench_ast_make_program(
            program,
            ast_idx,
            ast_idx % asts_per_kernel,
            num_inputs,
            seed,
            mode,
            cse_mode);
        asts[ast_idx] = program;
    }
}

static SecantAstInstructionType
secant_bench_ast_imbalanced_unary(uint32_t selector) {
    switch (selector % 3u) {
        case 0u: return SECANT_AST_INSTRUCTION_TYPE_SIN_F32;
        case 1u: return SECANT_AST_INSTRUCTION_TYPE_COS_F32;
        default: return SECANT_AST_INSTRUCTION_TYPE_EX2_F32;
    }
}

static int
secant_bench_ast_imbalanced_make_program(
    SecantAstInstruction* program,
    size_t global_ast_idx,
    size_t num_inputs,
    size_t num_nodes,
    uint32_t seed,
    SecantBenchAstMode mode
) {
    const uint32_t ast_key = secant_bench_ast_hash32((uint32_t)global_ast_idx ^ seed);
    const size_t num_leaves = mode == SECANT_BENCH_AST_MODE_MUFU
        ? (num_nodes + 3u) / 3u
        : (num_nodes + 1u) / 2u;
    const size_t num_binary = num_leaves - 1u;
    const size_t num_unary = num_nodes - num_leaves - num_binary;
    size_t output_offset = 0u;
    size_t leaf_idx;

    if ((mode != SECANT_BENCH_AST_MODE_ALU && mode != SECANT_BENCH_AST_MODE_MUFU) ||
        num_inputs == 0u || num_inputs > SECANT_AST_MAX_INPUTS || num_leaves < SECANT_BENCH_AST_NUM_LEAVES ||
        num_unary > num_leaves || num_nodes >= SECANT_AST_MAX_PROGRAM_INSTRUCTIONS) {
        return 0;
    }
    memset(program, 0, SECANT_BENCH_AST_PROGRAM_STRIDE * sizeof(*program));
    for (leaf_idx = 0u; leaf_idx < num_leaves; ++leaf_idx) {
        const uint32_t selector = secant_bench_ast_hash32(ast_key + (uint32_t)leaf_idx * UINT32_C(0x9e3779b9));
        SecantResult result = secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
            (SecantAstIdx)(selector % (uint32_t)num_inputs),
            0u,
            program,
            SECANT_BENCH_AST_PROGRAM_STRIDE,
            &output_offset);

        if (result != SECANT_SUCCESS) {
            return 0;
        }
        if (leaf_idx < num_unary) {
            const SecantAstInstructionType unary = mode == SECANT_BENCH_AST_MODE_MUFU
                ? secant_bench_ast_imbalanced_unary(selector >> 8u)
                : ((selector & 1u) != 0u
                    ? SECANT_AST_INSTRUCTION_TYPE_ABS_F32
                    : SECANT_AST_INSTRUCTION_TYPE_NEG_F32);

            if (secant_internal_ast_instruction_write(
                    unary, 0u, 0u, program, SECANT_BENCH_AST_PROGRAM_STRIDE, &output_offset) != SECANT_SUCCESS) {
                return 0;
            }
        }
        if (leaf_idx != 0u) {
            const SecantAstInstructionType binary = secant_bench_ast_binary(selector >> 16u, SECANT_BENCH_AST_MODE_ALU);

            if (secant_internal_ast_instruction_write(
                    binary, 0u, 0u, program, SECANT_BENCH_AST_PROGRAM_STRIDE, &output_offset) != SECANT_SUCCESS) {
                return 0;
            }
        }
    }
    return secant_internal_ast_instruction_write(
        SECANT_AST_INSTRUCTION_TYPE_RETURN_F32,
        0u,
        0u,
        program,
        SECANT_BENCH_AST_PROGRAM_STRIDE,
        &output_offset) == SECANT_SUCCESS;
}

int
secant_bench_ast_imbalanced_fill(
    size_t num_modules,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_nodes,
    uint32_t seed,
    SecantBenchAstMode mode,
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    size_t num_asts;
    size_t ast_idx;

    if (programs == NULL || asts == NULL || num_modules == 0u || num_kernels == 0u || asts_per_kernel == 0u ||
        num_kernels > SIZE_MAX / asts_per_kernel || num_modules > SIZE_MAX / (num_kernels * asts_per_kernel)) {
        return 0;
    }
    num_asts = num_modules * num_kernels * asts_per_kernel;
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program = programs + ast_idx * SECANT_BENCH_AST_PROGRAM_STRIDE;

        if (!secant_bench_ast_imbalanced_make_program(program, ast_idx, num_inputs, num_nodes, seed, mode)) {
            return 0;
        }
        asts[ast_idx] = program;
    }
    return 1;
}

int
secant_bench_ast_dynamic_constants_rewrite(
    size_t num_asts,
    size_t num_input_columns,
    size_t num_input_constants,
    SecantAstInstruction* programs
) {
    size_t ast_idx;

    if (programs == NULL || num_input_columns == 0u || num_input_constants == 0u ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        num_input_constants > SECANT_AST_MAX_INPUTS - num_input_columns) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program = programs + ast_idx * SECANT_BENCH_AST_PROGRAM_STRIDE;
        size_t offset = 0u;
        size_t instruction_idx;

        for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
            SecantAstInstruction* instruction = program + offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);

            if (instruction_size == 0u || offset > SECANT_BENCH_AST_PROGRAM_STRIDE - instruction_size) {
                return 0;
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
                const size_t input_idx = secant_ast_index_get(instruction);

                if (input_idx >= num_input_columns) {
                    if (input_idx - num_input_columns >= num_input_constants) {
                        return 0;
                    }
                    instruction[0] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32;
                    instruction[1] = (uint8_t)(input_idx - num_input_columns);
                }
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
            offset += instruction_size;
        }
        if (instruction_idx == SECANT_AST_MAX_PROGRAM_INSTRUCTIONS) {
            return 0;
        }
    }
    return 1;
}

int
secant_bench_ast_dynamic_leaves_rewrite(
    size_t num_asts,
    size_t num_dynamic_leaves,
    size_t num_dynamic_sites,
    int allow_static_columns,
    SecantAstInstruction* programs,
    size_t* max_dynamic_leaves_used_ret
) {
    size_t max_dynamic_leaves_used = 0u;
    size_t ast_idx;

    if (programs == NULL || max_dynamic_leaves_used_ret == NULL ||
        (allow_static_columns != 0 && allow_static_columns != 1) || num_dynamic_leaves == 0u ||
        num_dynamic_sites == 0u || num_dynamic_sites > num_dynamic_leaves ||
        num_dynamic_leaves > SECANT_AST_MAX_DYNAMIC_LEAVES) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction transformed[SECANT_BENCH_AST_PROGRAM_STRIDE];
        SecantAstInstruction* program = programs + ast_idx * SECANT_BENCH_AST_PROGRAM_STRIDE;
        size_t input_offset = 0u;
        size_t output_offset = 0u;
        size_t leaf_idx = 0u;
        size_t static_column_idx = 0u;
        size_t instruction_idx;

        memset(transformed, 0, sizeof(transformed));
        for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
            const SecantAstInstruction* instruction = program + input_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);
            size_t encoded_size = instruction_size;

            if (instruction_size == 0u || input_offset > SECANT_BENCH_AST_PROGRAM_STRIDE - instruction_size) {
                return 0;
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 &&
                (!allow_static_columns || static_column_idx < num_dynamic_sites)) {
                if (leaf_idx >= num_dynamic_leaves) {
                    return 0;
                }
                encoded_size = 2u;
                transformed[output_offset] =
                    (uint8_t)SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
                transformed[output_offset + 1u] = (uint8_t)leaf_idx;
                ++leaf_idx;
            } else {
                if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 ||
                    type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 ||
                    type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32 ||
                    output_offset > SECANT_BENCH_AST_PROGRAM_STRIDE - instruction_size) {
                    return 0;
                }
                memcpy(transformed + output_offset, instruction, instruction_size);
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
                ++static_column_idx;
            }
            output_offset += encoded_size;
            input_offset += instruction_size;
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
        }
        if (instruction_idx == SECANT_AST_MAX_PROGRAM_INSTRUCTIONS || leaf_idx == 0u ||
            (allow_static_columns && leaf_idx != num_dynamic_sites)) {
            return 0;
        }
        if (leaf_idx > max_dynamic_leaves_used) {
            max_dynamic_leaves_used = leaf_idx;
        }
        memcpy(program, transformed, sizeof(transformed));
    }
    *max_dynamic_leaves_used_ret = max_dynamic_leaves_used;
    return 1;
}

void
secant_bench_ast_get_routines(
    SecantBenchAstMode mode,
    const SecantAstInstruction* const** routines_ret,
    size_t* num_routines_ret,
    const char* const** routine_names_ret
) {
    if (mode == SECANT_BENCH_AST_MODE_ALU) {
#if SECANT_PORTABLE_ALU_NUM_ROUTINES > 0
        *routines_ret = secant_portable_alu_routines;
        *num_routines_ret = SECANT_PORTABLE_ALU_NUM_ROUTINES;
        *routine_names_ret = secant_portable_alu_routine_names;
#else
        *routines_ret = NULL;
        *num_routines_ret = 0u;
        *routine_names_ret = NULL;
#endif
    } else if (mode == SECANT_BENCH_AST_MODE_MUFU) {
        *routines_ret = secant_bench_routines;
        *num_routines_ret = SECANT_BENCH_ROUTINE_COUNT;
        *routine_names_ret = secant_bench_routine_names;
    } else {
        *routines_ret = NULL;
        *num_routines_ret = 0u;
        *routine_names_ret = NULL;
    }
}

uint64_t
secant_bench_ast_corpus_hash(
    const SecantAstInstruction* programs,
    size_t num_asts
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const size_t num_bytes = num_asts * SECANT_BENCH_AST_PROGRAM_STRIDE;
    size_t byte_idx;

    for (byte_idx = 0u; byte_idx < num_bytes; ++byte_idx) {
        hash ^= programs[byte_idx];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
