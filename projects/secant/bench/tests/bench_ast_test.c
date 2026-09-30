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
#include "bench_secant_api.h"
#include "secant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SECANT_PORTABLE_ALU_INCLUDE_AST
#define SECANT_PORTABLE_ALU_INCLUDE_NATIVE
#include "corpus/portable_alu_v1.h"

#define SECANT_PORTABLE_ALU_TEST_CASES 8u

static float
portable_expected(size_t ast_idx, const float* x) {
    switch (ast_idx) {
        case 0u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_0(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 1u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_1(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 2u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_2(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 3u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_3(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 4u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_4(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 5u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_5(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 6u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_6(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        default:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_7(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    }
}

static int
test_portable_corpus(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    float input[SECANT_PORTABLE_ALU_NUM_INPUTS];
    float output[SECANT_PORTABLE_ALU_TEST_CASES];
    size_t ast_idx;

    secant_bench_ast_fill(
        1u,
        1u,
        SECANT_PORTABLE_ALU_TEST_CASES,
        SECANT_PORTABLE_ALU_NUM_INPUTS,
        1234567u,
        SECANT_BENCH_AST_MODE_ALU,
        SECANT_BENCH_CSE_DEFAULT,
        programs,
        asts);
    if (strcmp(
            secant_bench_ast_corpus_name(SECANT_BENCH_AST_MODE_ALU),
            SECANT_PORTABLE_ALU_CORPUS_NAME) != 0 ||
        strcmp(
            secant_bench_ast_corpus_definition_hash(
                SECANT_BENCH_AST_MODE_ALU),
            SECANT_PORTABLE_ALU_CORPUS_HASH) != 0) {
        return 1;
    }
    for (ast_idx = 0u;
         ast_idx < SECANT_PORTABLE_ALU_TEST_CASES;
         ++ast_idx) {
        if (memcmp(
                asts[ast_idx],
                secant_portable_alu_asts[ast_idx],
                SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS *
                    sizeof(SecantAstInstruction)) != 0) {
            return 1;
        }
        input[ast_idx] = 0.25f + 0.125f * (float)ast_idx;
    }
    if (secant_bench_cpu_materialize_run(
            SECANT_PORTABLE_ALU_NUM_INPUTS,
            NULL,
            0u,
            asts,
            SECANT_PORTABLE_ALU_TEST_CASES,
            input,
            SECANT_PORTABLE_ALU_NUM_INPUTS,
            1u,
            1u,
            output,
            SECANT_PORTABLE_ALU_TEST_CASES,
            1u) != SECANT_SUCCESS) {
        return 1;
    }
    for (ast_idx = 0u;
         ast_idx < SECANT_PORTABLE_ALU_TEST_CASES;
         ++ast_idx) {
        if (fabsf(output[ast_idx] - portable_expected(ast_idx, input)) >
            1.0e-6f) {
            return 1;
        }
    }
    return 0;
}

static int
test_alu_discriminators(
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    int expect_distinct
) {
    uint32_t discriminator_bits[64];
    size_t num_discriminators = 0u;
    size_t num_discriminator_sites = 0u;
    size_t ast_idx;
    int saw_duplicate = 0;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        const SecantAstInstruction* program = asts[ast_idx];
        size_t instruction_offset = 0u;
        size_t instruction_idx;

        for (instruction_idx = 0u;
             instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
             ++instruction_idx) {
            const SecantAstInstruction* instruction = program + instruction_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);

            if (instruction_size == 0u) {
                return 0;
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
                const SecantAstInstruction* constant = instruction + instruction_size;
                const size_t constant_size = secant_ast_instruction_size_get(constant);
                const SecantAstInstruction* operation = constant + constant_size;
                const SecantAstInstructionType operation_type = secant_ast_instruction_type_get(operation);
                uint32_t bits;
                size_t prior_idx;

                if (secant_ast_instruction_type_get(constant) !=
                        SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 ||
                    (operation_type != SECANT_AST_INSTRUCTION_TYPE_ADD_F32 &&
                     operation_type != SECANT_AST_INSTRUCTION_TYPE_MUL_F32) ||
                    num_discriminators >= sizeof(discriminator_bits) / sizeof(discriminator_bits[0])) {
                    return 0;
                }
                bits = secant_ast_constant_f32_bits_get(constant);
                if ((bits & 0x7f800000u) != 0x3f000000u) {
                    return 0;
                }
                for (prior_idx = 0u; prior_idx < num_discriminators; ++prior_idx) {
                    if (discriminator_bits[prior_idx] == bits) {
                        saw_duplicate = 1;
                        break;
                    }
                }
                if (prior_idx == num_discriminators) {
                    discriminator_bits[num_discriminators++] = bits;
                }
                num_discriminator_sites += 1u;
                instruction_offset += constant_size + secant_ast_instruction_size_get(operation);
                instruction_idx += 2u;
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
            instruction_offset += instruction_size;
        }
    }
    if (num_discriminator_sites != num_asts * SECANT_BENCH_AST_NUM_LEAVES) {
        return 0;
    }
    return expect_distinct
        ? !saw_duplicate && num_discriminators == num_discriminator_sites
        : saw_duplicate && num_discriminators <= SECANT_PORTABLE_ALU_NUM_INPUTS;
}

static int
test_alu_cse(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    secant_bench_ast_fill(
        1u,
        1u,
        4u,
        SECANT_PORTABLE_ALU_NUM_INPUTS,
        1u,
        SECANT_BENCH_AST_MODE_ALU,
        SECANT_BENCH_CSE_DISTINCT,
        programs,
        asts);
    if (!test_alu_discriminators(asts, 4u, 1)) {
        return 1;
    }
    secant_bench_ast_fill(
        1u,
        1u,
        4u,
        SECANT_PORTABLE_ALU_NUM_INPUTS,
        1u,
        SECANT_BENCH_AST_MODE_ALU,
        SECANT_BENCH_CSE_SHARED,
        programs,
        asts);
    return !test_alu_discriminators(asts, 4u, 0);
}

static int
test_alu_input_remap(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    const SecantBenchCSEMode modes[] = {
        SECANT_BENCH_CSE_DEFAULT,
        SECANT_BENCH_CSE_SHARED,
        SECANT_BENCH_CSE_DISTINCT,
    };
    size_t mode_idx;

    for (mode_idx = 0u; mode_idx < sizeof(modes) / sizeof(modes[0]); ++mode_idx) {
        size_t ast_idx;

        secant_bench_ast_fill(1u, 1u, 4u, 3u, 1u, SECANT_BENCH_AST_MODE_ALU, modes[mode_idx], programs, asts);
        for (ast_idx = 0u; ast_idx < 4u; ++ast_idx) {
            size_t instruction_offset = 0u;
            size_t instruction_idx;

            for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
                const SecantAstInstruction* instruction = asts[ast_idx] + instruction_offset;
                const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
                const size_t instruction_size = secant_ast_instruction_size_get(instruction);

                if (instruction_size == 0u ||
                    (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 &&
                     secant_ast_index_get(instruction) >= 3u)) {
                    return 1;
                }
                if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                    break;
                }
                instruction_offset += instruction_size;
            }
            if (instruction_idx == SECANT_AST_MAX_PROGRAM_INSTRUCTIONS) {
                return 1;
            }
        }
    }
    return 0;
}

static int
test_mufu_discriminators(
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    int expect_distinct
) {
    uint32_t discriminator_bits[64];
    size_t num_discriminators = 0u;
    size_t ast_idx;
    int saw_duplicate = 0;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstructionType previous_type = SECANT_AST_INSTRUCTION_TYPE_NONE;
        SecantAstInstructionType prior_type = SECANT_AST_INSTRUCTION_TYPE_NONE;
        uint32_t prior_constant_bits = 0u;
        size_t instruction_offset = 0u;
        size_t instruction_idx;

        for (instruction_idx = 0u;
             instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
             ++instruction_idx) {
            const SecantAstInstruction* instruction = asts[ast_idx] + instruction_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);
            const int is_mufu_call =
                type == SECANT_AST_INSTRUCTION_TYPE_SIN_F32 ||
                type == SECANT_AST_INSTRUCTION_TYPE_COS_F32 ||
                type == SECANT_AST_INSTRUCTION_TYPE_EX2_F32 ||
                type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32;
            size_t prior_idx;

            if (instruction_size == 0u) {
                return 0;
            }
            if (is_mufu_call) {
                if (previous_type != SECANT_AST_INSTRUCTION_TYPE_ADD_F32 &&
                    previous_type != SECANT_AST_INSTRUCTION_TYPE_MUL_F32) {
                    return 0;
                }
                if (prior_type != SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 ||
                    (prior_constant_bits & 0x7f800000u) != 0x3f000000u ||
                    num_discriminators >= sizeof(discriminator_bits) / sizeof(discriminator_bits[0])) {
                    return 0;
                }
                for (prior_idx = 0u; prior_idx < num_discriminators; ++prior_idx) {
                    if (discriminator_bits[prior_idx] == prior_constant_bits) {
                        saw_duplicate = 1;
                        break;
                    }
                }
                if (prior_idx == num_discriminators) {
                    discriminator_bits[num_discriminators++] = prior_constant_bits;
                }
            }
            prior_type = previous_type;
            if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
                prior_constant_bits = secant_ast_constant_f32_bits_get(instruction);
            }
            previous_type = type;
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
            instruction_offset += instruction_size;
        }
    }
    return expect_distinct
        ? !saw_duplicate && num_discriminators >= num_asts * SECANT_BENCH_AST_NUM_LEAVES
        : saw_duplicate && num_discriminators <= 6u;
}

static int
test_run(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    size_t num_routines = 0u;
    size_t instruction_idx;
    float input[8];
    float output[24];
    size_t value_idx;
    size_t instruction_offset = 0u;
    int saw_rcp = 0;
    int saw_div = 0;

    secant_bench_ast_fill(
        2u,
        3u,
        4u,
        8u,
        7u,
        SECANT_BENCH_AST_MODE_MUFU,
        SECANT_BENCH_CSE_DISTINCT,
        programs,
        asts);
    secant_bench_ast_get_routines(
        SECANT_BENCH_AST_MODE_MUFU,
        &routines,
        &num_routines,
        &routine_names);
    if (routines == NULL ||
        routine_names == NULL ||
        num_routines != SECANT_BENCH_ROUTINE_COUNT) {
        return 1;
    }

    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction =
            routines[SECANT_BENCH_ROUTINE_SAFE_DIV] +
            instruction_offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            return 1;
        }
        saw_rcp |= type == SECANT_AST_INSTRUCTION_TYPE_RCP_F32;
        saw_div |= type == SECANT_AST_INSTRUCTION_TYPE_DIV_F32;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            break;
        }
        instruction_offset += instruction_size;
    }
    if (!saw_rcp || saw_div ||
        secant_bench_ast_corpus_hash(programs, 24u) == 0u ||
        !test_mufu_discriminators(asts, 4u, 1)) {
        return 1;
    }
    secant_bench_ast_fill(
        2u,
        3u,
        4u,
        8u,
        7u,
        SECANT_BENCH_AST_MODE_MUFU,
        SECANT_BENCH_CSE_SHARED,
        programs,
        asts);
    if (!test_mufu_discriminators(asts, 4u, 0)) {
        return 1;
    }
    for (value_idx = 0u; value_idx < 8u; ++value_idx) {
        input[value_idx] = 0.25f + 0.125f * (float)value_idx;
    }
    if (secant_bench_cpu_materialize_run(
            8u,
            routines,
            num_routines,
            asts,
            24u,
            input,
            8u,
            1u,
            1u,
            output,
            24u,
            1u) != SECANT_SUCCESS) {
        return 1;
    }
    for (value_idx = 0u; value_idx < 24u; ++value_idx) {
        if (!isfinite(output[value_idx])) {
            return 1;
        }
    }
    return 0;
}

static int
test_dynamic_leaf_projection(
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    size_t expected_dynamic_leaves,
    size_t expected_static_columns
) {
    size_t ast_idx;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        size_t instruction_offset = 0u;
        size_t expected_leaf = 0u;
        size_t num_constants = 0u;
        size_t num_static_columns = 0u;
        size_t instruction_idx;

        for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
            const SecantAstInstruction* instruction = asts[ast_idx] + instruction_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);

            if (instruction_size == 0u) {
                return 1;
            }
            num_constants += type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32;
            num_static_columns += type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32;
            if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
                if (secant_ast_index_get(instruction) != expected_leaf) {
                    return 1;
                }
                ++expected_leaf;
            }
            instruction_offset += instruction_size;
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
        }
        if (expected_leaf != expected_dynamic_leaves || num_static_columns != expected_static_columns ||
            num_constants == 0u) {
            return 1;
        }
    }
    return 0;
}

static int
test_dynamic_leaf_rewrite(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    size_t max_dynamic_leaves_used = 0u;

    secant_bench_ast_fill(
        1u, 1u, 4u, 8u, 19u, SECANT_BENCH_AST_MODE_ALU, SECANT_BENCH_CSE_DISTINCT, programs, asts);
    if (secant_bench_ast_dynamic_leaves_rewrite(4u, 4u, 4u, 0, programs, &max_dynamic_leaves_used)) {
        return 1;
    }
    secant_bench_ast_fill(
        1u, 1u, 4u, 8u, 19u, SECANT_BENCH_AST_MODE_ALU, SECANT_BENCH_CSE_DISTINCT, programs, asts);
    if (!secant_bench_ast_dynamic_leaves_rewrite(4u, 8u, 8u, 0, programs, &max_dynamic_leaves_used) ||
        max_dynamic_leaves_used != 8u || test_dynamic_leaf_projection(asts, 4u, 8u, 0u) != 0) {
        return 1;
    }
    secant_bench_ast_fill(
        1u, 1u, 4u, 8u, 19u, SECANT_BENCH_AST_MODE_ALU, SECANT_BENCH_CSE_DISTINCT, programs, asts);
    if (!secant_bench_ast_dynamic_leaves_rewrite(4u, 8u, 4u, 1, programs, &max_dynamic_leaves_used) ||
        max_dynamic_leaves_used != 4u || test_dynamic_leaf_projection(asts, 4u, 4u, 4u) != 0) {
        return 1;
    }
    secant_bench_ast_fill(
        1u, 1u, 4u, 8u, 23u, SECANT_BENCH_AST_MODE_MUFU, SECANT_BENCH_CSE_DISTINCT, programs, asts);
    if (!secant_bench_ast_dynamic_leaves_rewrite(4u, 8u, 4u, 1, programs, &max_dynamic_leaves_used) ||
        max_dynamic_leaves_used != 4u || test_dynamic_leaf_projection(asts, 4u, 4u, 4u) != 0) {
        return 1;
    }
    return 0;
}

static int
test_imbalanced_ast(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    const SecantBenchAstMode modes[] = {SECANT_BENCH_AST_MODE_ALU, SECANT_BENCH_AST_MODE_MUFU};
    size_t mode_idx;

    for (mode_idx = 0u; mode_idx < sizeof(modes) / sizeof(modes[0]); ++mode_idx) {
        size_t max_dynamic_leaves_used = 0u;
        size_t ast_idx;

        if (!secant_bench_ast_imbalanced_fill(1u, 1u, 4u, 16u, 128u, 37u, modes[mode_idx], programs, asts) ||
            !secant_bench_ast_dynamic_leaves_rewrite(4u, 8u, 8u, 1, programs, &max_dynamic_leaves_used) ||
            max_dynamic_leaves_used != 8u) {
            return 1;
        }
        for (ast_idx = 0u; ast_idx < 4u; ++ast_idx) {
            size_t instruction_offset = 0u;
            size_t instruction_count = 0u;
            size_t dynamic_leaves = 0u;
            size_t static_columns = 0u;

            for (;;) {
                const SecantAstInstruction* instruction = asts[ast_idx] + instruction_offset;
                const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
                const size_t instruction_size = secant_ast_instruction_size_get(instruction);

                if (instruction_size == 0u || instruction_offset > SECANT_BENCH_AST_PROGRAM_STRIDE - instruction_size) {
                    return 1;
                }
                if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                    break;
                }
                dynamic_leaves += type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
                static_columns += type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32;
                instruction_offset += instruction_size;
                ++instruction_count;
            }
            if (instruction_count != 128u || dynamic_leaves != 8u ||
                static_columns != (modes[mode_idx] == SECANT_BENCH_AST_MODE_ALU ? 56u : 35u)) {
                return 1;
            }
        }
    }
    return 0;
}

int
main(void) {
    const SecantAstInstruction** asts = NULL;
    SecantAstInstruction* programs = NULL;
    size_t program_bytes = 0u;
    size_t pointer_bytes = 0u;
    int result = 1;

    if (secant_bench_ast_storage_sizes(
            2u,
            3u,
            4u,
            &program_bytes,
            &pointer_bytes)) {
        programs = (SecantAstInstruction*)malloc(program_bytes);
        asts = (const SecantAstInstruction**)malloc(pointer_bytes);
        if (programs != NULL && asts != NULL) {
            result = test_portable_corpus(programs, asts);
            if (result == 0) {
                result = test_alu_cse(programs, asts);
            }
            if (result == 0) {
                result = test_alu_input_remap(programs, asts);
            }
            if (result == 0) {
                result = test_run(programs, asts);
            }
            if (result == 0) {
                result = test_dynamic_leaf_rewrite(programs, asts);
            }
            if (result == 0) {
                result = test_imbalanced_ast(programs, asts);
            }
        }
    }
    free(asts);
    free(programs);
    return result;
}
