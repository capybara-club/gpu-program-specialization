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
    secant_ast_encode_constant_f32_bits(SECANT_BENCH_AST_ROUTINE_EPSILON_BITS),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction secant_bench_routine_safe_sqrt[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32_bits(SECANT_BENCH_AST_ROUTINE_EPSILON_BITS),
    secant_ast_encode_add_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction secant_bench_routine_safe_rsqrt[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32_bits(SECANT_BENCH_AST_ROUTINE_EPSILON_BITS),
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

static SecantAstInstruction
secant_bench_ast_binary(uint32_t selector, SecantBenchAstMode mode) {
    if (mode == SECANT_BENCH_AST_MODE_MUFU && (selector & 7u) == 0u) {
        return secant_ast_encode_routine_f32(SECANT_BENCH_ROUTINE_SAFE_DIV, 2u);
    }

    switch (selector & 3u) {
        case 0u: return secant_ast_encode_add_f32;
        case 1u: return secant_ast_encode_mul_f32;
        case 2u: return secant_ast_encode_min_f32;
        default: return secant_ast_encode_max_f32;
    }
}

static void
secant_bench_ast_emit_leaf(
    SecantAstInstruction* program,
    size_t* count,
    uint32_t ast_key,
    uint32_t leaf_idx,
    size_t num_inputs,
    SecantBenchAstMode mode
) {
    const uint32_t selector = secant_bench_ast_hash32(
        ast_key + leaf_idx * 0x9e3779b9u);

    program[(*count)++] = secant_ast_encode_input_f32(
        selector % (uint32_t)num_inputs);
    if (mode != SECANT_BENCH_AST_MODE_MUFU) {
        return;
    }

    switch ((selector >> 8) % 5u) {
        case 0u:
            program[(*count)++] = secant_ast_encode_sin_f32;
            break;
        case 1u:
            program[(*count)++] = secant_ast_encode_cos_f32;
            break;
        case 2u:
            program[(*count)++] = secant_ast_encode_constant_f32(0.125f);
            program[(*count)++] = secant_ast_encode_mul_f32;
            program[(*count)++] = secant_ast_encode_ex2_f32;
            break;
        case 3u:
            program[(*count)++] = secant_ast_encode_routine_f32(
                SECANT_BENCH_ROUTINE_SAFE_SQRT,
                1u);
            break;
        default:
            program[(*count)++] = secant_ast_encode_routine_f32(
                SECANT_BENCH_ROUTINE_SAFE_RSQRT,
                1u);
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
    SecantBenchAstMode mode
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
            mode);
        return;
    }

    secant_bench_ast_emit_tree(
        program,
        count,
        ast_key,
        first_leaf,
        left_leaves,
        num_inputs,
        mode);
    secant_bench_ast_emit_tree(
        program,
        count,
        ast_key,
        first_leaf + left_leaves,
        right_leaves,
        num_inputs,
        mode);
    program[(*count)++] = secant_bench_ast_binary(
        secant_bench_ast_hash32(
            ast_key ^
            first_leaf * 0x85ebca6bu ^
            num_leaves * 0xc2b2ae35u),
        mode);
}

static void
secant_bench_ast_make_program(
    SecantAstInstruction* program,
    size_t global_ast_idx,
    size_t num_inputs,
    uint32_t seed,
    SecantBenchAstMode mode
) {
    const uint32_t ast_key = secant_bench_ast_hash32(
        (uint32_t)global_ast_idx ^ seed);
    size_t count = 0u;

    memset(
        program,
        0,
        SECANT_BENCH_AST_PROGRAM_STRIDE * sizeof(*program));
    if (mode == SECANT_BENCH_AST_MODE_ALU) {
        memcpy(
            program,
            secant_portable_alu_asts[
                global_ast_idx % SECANT_PORTABLE_ALU_NUM_CASES],
            SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS * sizeof(*program));
        return;
    }
    if (mode == SECANT_BENCH_AST_MODE_SIMPLE) {
        program[count++] = secant_ast_encode_input_f32(
            ast_key % (uint32_t)num_inputs);
        program[count++] = secant_ast_encode_constant_f32(
            0.25f + (float)(ast_key & 7u) * 0.125f);
        program[count++] = (ast_key & 1u)
            ? secant_ast_encode_add_f32
            : secant_ast_encode_mul_f32;
        program[count++] = secant_ast_encode_return_f32;
        return;
    }

    secant_bench_ast_emit_tree(
        program,
        &count,
        ast_key,
        0u,
        SECANT_BENCH_AST_NUM_LEAVES,
        num_inputs,
        mode);
    program[count++] = secant_ast_encode_return_f32;
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
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    const size_t num_asts =
        num_modules * num_kernels * asts_per_kernel;
    size_t ast_idx;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program =
            programs + ast_idx * SECANT_BENCH_AST_PROGRAM_STRIDE;

        secant_bench_ast_make_program(
            program,
            ast_idx,
            num_inputs,
            seed,
            mode);
        asts[ast_idx] = program;
    }
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
    const size_t num_instructions =
        num_asts * SECANT_BENCH_AST_PROGRAM_STRIDE;
    size_t instruction_idx;

    for (instruction_idx = 0u;
         instruction_idx < num_instructions;
         ++instruction_idx) {
        const SecantAstInstruction instruction =
            programs[instruction_idx];
        const uint32_t fields[] = {
            instruction.instruction_type,
            instruction.aux,
            secant_ast_constant_f32_bits_get(&instruction)
        };
        size_t field_idx;

        for (field_idx = 0u;
             field_idx < sizeof(fields) / sizeof(fields[0]);
             ++field_idx) {
            uint32_t value = fields[field_idx];
            size_t byte_idx;

            for (byte_idx = 0u; byte_idx < sizeof(value); ++byte_idx) {
                hash ^= value & 0xffu;
                hash *= UINT64_C(1099511628211);
                value >>= 8u;
            }
        }
    }
    return hash;
}
