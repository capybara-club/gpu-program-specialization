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
#include "cubin_evaluator.h"
#include "dataset.h"
#include "mse_reducer.h"
#include "secant_sr.h"
#include "secant_instructions.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
secant_sr_cubin_test_mse_entry_compare(const void* lhs_pointer, const void* rhs_pointer) {
    const SecantSRMSEEntry* lhs = lhs_pointer;
    const SecantSRMSEEntry* rhs = rhs_pointer;

    if (lhs->mse < rhs->mse) {
        return -1;
    }
    if (lhs->mse > rhs->mse) {
        return 1;
    }
    if (lhs->ast_idx < rhs->ast_idx) {
        return -1;
    }
    return lhs->ast_idx > rhs->ast_idx;
}

static double
secant_sr_cubin_test_target_ssd_get(const float* target, size_t num_rows) {
    double mean = 0.0;
    double ssd = 0.0;
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        mean += target[row];
    }
    mean /= (double)num_rows;
    for (row = 0u; row < num_rows; ++row) {
        const double difference = (double)target[row] - mean;

        ssd += difference * difference;
    }
    return ssd;
}

static int
secant_sr_cubin_test_reduced_entries_validate(
    const float* cpu_sse,
    size_t num_asts,
    size_t num_settings,
    size_t num_rows,
    double target_ssd,
    const SecantSRMSEEntry* actual,
    size_t top_k
) {
    SecantSRMSEEntry* expected = malloc(num_asts * sizeof(*expected));
    size_t ast_idx;
    int success = expected != NULL;

    for (ast_idx = 0u; success && ast_idx < num_asts; ++ast_idx) {
        size_t best_setting = 0u;
        float best_sse = INFINITY;
        double robustness = 0.0;
        size_t setting;

        for (setting = 0u; setting < num_settings; ++setting) {
            const float candidate = cpu_sse[ast_idx * num_settings + setting];

            if (isfinite(candidate) && candidate >= 0.0f &&
                (candidate < best_sse || (candidate == best_sse && setting < best_setting))) {
                best_sse = candidate;
                best_setting = setting;
            }
            if (isfinite(candidate) && candidate >= 0.0f) {
                double r2 = 1.0 - (double)candidate / target_ssd;

                if (r2 < 0.0) {
                    r2 = 0.0;
                } else if (r2 > 1.0) {
                    r2 = 1.0;
                }
                robustness += r2;
            }
        }
        expected[ast_idx].mse = best_sse / (float)num_rows;
        expected[ast_idx].ast_idx = (uint32_t)ast_idx;
        expected[ast_idx].setting_idx = (uint32_t)best_setting;
        expected[ast_idx].robustness = (float)(robustness / (double)num_settings);
    }
    if (success) {
        qsort(expected, num_asts, sizeof(*expected), secant_sr_cubin_test_mse_entry_compare);
    }
    for (ast_idx = 0u; success && ast_idx < top_k; ++ast_idx) {
        const float tolerance = 0.0005f * (1.0f + fabsf(expected[ast_idx].mse));
        const float robustness_tolerance = 0.0005f * (1.0f + fabsf(expected[ast_idx].robustness));

        if (actual[ast_idx].ast_idx != expected[ast_idx].ast_idx ||
            actual[ast_idx].setting_idx != expected[ast_idx].setting_idx ||
            !isfinite(actual[ast_idx].mse) || fabsf(actual[ast_idx].mse - expected[ast_idx].mse) > tolerance ||
            !isfinite(actual[ast_idx].robustness) ||
            fabsf(actual[ast_idx].robustness - expected[ast_idx].robustness) > robustness_tolerance) {
            fprintf(stderr,
                "reduced MSE mismatch rank=%zu expected=(%u,%u,%.9g) actual=(%u,%u,%.9g) tolerance=%.9g\n",
                ast_idx,
                expected[ast_idx].ast_idx,
                expected[ast_idx].setting_idx,
                expected[ast_idx].mse,
                actual[ast_idx].ast_idx,
                actual[ast_idx].setting_idx,
                actual[ast_idx].mse,
                tolerance);
            success = 0;
        }
    }
    free(expected);
    return success;
}

static int
secant_sr_cubin_test_run(const SecantSRCudaSession* session) {
    static const SecantAstInstruction square_f32[] = {
        secant_ast_encode_routine_arg_f32(0u),
        secant_ast_encode_routine_arg_f32(0u),
        secant_ast_encode_mul_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction cube_f32[] = {
        secant_ast_encode_routine_arg_f32(0u),
        secant_ast_encode_routine_arg_f32(0u),
        secant_ast_encode_mul_f32,
        secant_ast_encode_routine_arg_f32(0u),
        secant_ast_encode_mul_f32,
        secant_ast_encode_return_f32
    };
    static const SecantSRRoutine routines[] = {
        {square_f32, "square", 1u, 1u, 0u},
        {cube_f32, "cube", 1u, 1u, 0u}
    };
    static const SecantAstInstruction* const routine_programs[] = {
        square_f32,
        cube_f32
    };
    static const SecantAstInstructionType unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32
    };
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32
    };
    static const float constants[] = {-1.0f, 0.5f, 1.0f, 2.0f};
    const SecantSRSearchConfig config = {
        251u, 96u, 48u, 12u, 24u, 1u, 8u, 2u, 8u, 4u, 1.6,
        0.45, 0.30, 0.20, 0.15, 0.00005, UINT64_C(0x918273645), 0u
    };
    const size_t num_rows = 257u;
    SecantSRCubinEvaluator evaluator;
    SecantSRSearch search = NULL;
    const SecantSRIndividual* individuals = NULL;
    const SecantAstInstruction* const* asts = NULL;
    void* storage = NULL;
    float* input = NULL;
    float* target = NULL;
    float* cpu_sse = NULL;
    float* gpu_sse = NULL;
    size_t storage_size = 0u;
    size_t num_asts = 0u;
    size_t num_individuals = 0u;
    size_t ast_idx;
    SecantCpuSSERun cpu_run = secant_cpu_sse_run_init();
    int evaluator_active = 0;
    uint32_t routine_found_mask = 0u;
    int status = 1;

    if (secant_sr_search_storage_size(
            &config,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            sizeof(routines) / sizeof(routines[0]),
            sizeof(constants) / sizeof(constants[0]),
            &storage_size) != SECANT_SR_SUCCESS) {
        return 1;
    }
    storage = malloc(storage_size);
    input = malloc(num_rows * sizeof(*input));
    target = malloc(num_rows * sizeof(*target));
    cpu_sse = calloc(config.population_size, sizeof(*cpu_sse));
    gpu_sse = malloc(config.population_size * sizeof(*gpu_sse));
    if (storage != NULL && input != NULL && target != NULL && cpu_sse != NULL && gpu_sse != NULL &&
        secant_sr_search_init(
            &config,
            unary_ops,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            binary_ops,
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            routines,
            sizeof(routines) / sizeof(routines[0]),
            constants,
            sizeof(constants) / sizeof(constants[0]),
            storage,
            storage_size,
            &search) == SECANT_SR_SUCCESS &&
        secant_sr_search_asts_get(search, &asts, &num_asts) == SECANT_SR_SUCCESS &&
        secant_sr_search_individuals_get(search, &individuals, &num_individuals) == SECANT_SR_SUCCESS &&
        num_individuals == num_asts) {
        (void)secant_sr_dataset_nguyen1_fill(input, target, num_rows);
        evaluator_active = secant_sr_cubin_evaluator_create(
            8u, 32u, 1u, 256u, 128u, 1536u, 4u, 4u, input, target, config.population_size, num_rows, session, NULL,
            &evaluator);
        if (evaluator_active) {
            evaluator_active = secant_sr_cubin_evaluator_routines_set(
                &evaluator,
                routine_programs,
                sizeof(routine_programs) / sizeof(routine_programs[0]));
        }
        for (ast_idx = 0u; ast_idx < num_individuals && routine_found_mask != UINT32_C(3); ++ast_idx) {
            size_t offset = 0u;

            while (offset < individuals[ast_idx].program_bytes) {
                const SecantAstInstruction* instruction = individuals[ast_idx].program + offset;
                const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);

                if (type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
                    const size_t routine_idx = secant_ast_index_get(instruction);

                    if (routine_idx < 2u) {
                        routine_found_mask |= UINT32_C(1) << routine_idx;
                    }
                }
                offset += secant_ast_instruction_size_get(instruction);
            }
        }
    }
    cpu_run.programs.routines.items = routine_programs;
    cpu_run.programs.routines.count = sizeof(routine_programs) / sizeof(routine_programs[0]);
    cpu_run.programs.asts.items = asts;
    cpu_run.programs.asts.count = num_asts;
    cpu_run.num_inputs = 1u;
    cpu_run.num_targets = 1u;
    cpu_run.input = (SecantConstHostMatrixF32){input, num_rows, num_rows};
    cpu_run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
    cpu_run.num_rows = num_rows;
    cpu_run.output = (SecantHostMatrixF32){cpu_sse, num_asts, 1u};
    if (evaluator_active && routine_found_mask == UINT32_C(3) && secant_cpu_run_sse(&cpu_run) == SECANT_SUCCESS &&
        secant_sr_cubin_evaluator_run(&evaluator, asts, num_asts, gpu_sse)) {
        status = 0;
        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            const int matching_positive_infinity = isinf(cpu_sse[ast_idx]) && cpu_sse[ast_idx] > 0.0f &&
                isinf(gpu_sse[ast_idx]) && gpu_sse[ast_idx] > 0.0f;
            const float tolerance = isfinite(cpu_sse[ast_idx])
                ? 0.0005f * (1.0f + fabsf(cpu_sse[ast_idx]))
                : 0.0f;

            if (!matching_positive_infinity &&
                (!isfinite(cpu_sse[ast_idx]) || !isfinite(gpu_sse[ast_idx]) ||
                 fabsf(cpu_sse[ast_idx] - gpu_sse[ast_idx]) > tolerance)) {
                fprintf(stderr, "SSE mismatch ast=%zu cpu=%.9g gpu=%.9g tolerance=%.9g\n",
                    ast_idx, cpu_sse[ast_idx], gpu_sse[ast_idx], tolerance);
                status = 1;
                break;
            }
        }
    }
    if (evaluator_active) {
        secant_sr_cubin_evaluator_destroy(&evaluator);
    }
    free(gpu_sse);
    free(cpu_sse);
    free(target);
    free(input);
    free(storage);
    return status;
}

static int
secant_sr_cubin_dynamic_leaf_test_run(const SecantSRCudaSession* session, int mixed) {
    static const SecantAstInstruction dynamic_ast[] = {
        secant_ast_encode_dynamic_constant_or_column_input_f32(0),
        secant_ast_encode_dynamic_constant_or_column_input_f32(1),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction mixed_ast[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_dynamic_constant_or_column_input_f32(0),
        secant_ast_encode_add_f32,
        secant_ast_encode_dynamic_constant_or_column_input_f32(1),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    const SecantAstInstruction* ast = mixed ? mixed_ast : dynamic_ast;
    const size_t num_asts = 251u;
    const size_t num_rows = 257u;
    const size_t num_inputs = 9u;
    const size_t num_settings = 8u;
    const size_t leaf_stride = 2u;
    SecantSRCubinEvaluator evaluator;
    SecantSRMSEReducer reducer;
    const SecantAstInstruction** asts = NULL;
    uint32_t leaf_masks[8];
    uint32_t leaf_words[16];
    float* input = NULL;
    float* target = NULL;
    float* cpu_sse = NULL;
    float* gpu_sse = NULL;
    size_t ast_idx;
    size_t setting;
    SecantCpuDynamicLeafSSERun cpu_run = secant_cpu_dynamic_leaf_sse_run_init();
    int evaluator_active = 0;
    int reducer_active = 0;
    int status = 1;

    memset(&evaluator, 0, sizeof(evaluator));
    memset(&reducer, 0, sizeof(reducer));
    asts = malloc(num_asts * sizeof(*asts));
    input = malloc(num_inputs * num_rows * sizeof(*input));
    target = malloc(num_rows * sizeof(*target));
    cpu_sse = calloc(num_asts * num_settings, sizeof(*cpu_sse));
    gpu_sse = malloc(num_asts * num_settings * sizeof(*gpu_sse));
    if (asts != NULL && input != NULL && target != NULL && cpu_sse != NULL && gpu_sse != NULL) {
        size_t input_idx;
        size_t row;

        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            asts[ast_idx] = ast;
        }
        for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
            for (row = 0u; row < num_rows; ++row) {
                input[input_idx * num_rows + row] =
                    sinf((float)row * 0.03125f + (float)input_idx * 0.25f);
            }
        }
        for (row = 0u; row < num_rows; ++row) {
            target[row] = input[row] + input[8u * num_rows + row];
        }
        for (setting = 0u; setting < num_settings; ++setting) {
            const float constant = 0.25f * (float)(setting + 1u);

            leaf_masks[setting] = setting < 4u ? UINT32_C(3) : UINT32_C(1);
            leaf_words[setting * leaf_stride] = (uint32_t)setting;
            if ((leaf_masks[setting] & UINT32_C(2)) != 0u) {
                leaf_words[setting * leaf_stride + 1u] = (uint32_t)(8u - setting);
            } else {
                memcpy(leaf_words + setting * leaf_stride + 1u, &constant, sizeof(constant));
            }
        }
        evaluator_active = (mixed
            ? secant_sr_cubin_mixed_dynamic_leaf_evaluator_create
            : secant_sr_cubin_dynamic_leaf_evaluator_create)(
            8u,
            32u,
            2u,
            256u,
            128u,
            1536u,
            4u,
            4u,
            input,
            num_inputs,
            target,
            leaf_masks,
            leaf_words,
            leaf_stride,
            num_settings,
            num_asts,
            num_rows,
            session,
            NULL,
            &evaluator);
        reducer_active = evaluator_active && secant_sr_mse_reducer_create(num_asts, 7u, session, NULL, &reducer);
        cpu_run.programs.asts.items = asts;
        cpu_run.programs.asts.count = num_asts;
        cpu_run.num_dynamic_leaves = 2u;
        cpu_run.num_input_columns = num_inputs;
        cpu_run.num_static_input_columns = mixed ? num_inputs : 0u;
        cpu_run.num_targets = 1u;
        cpu_run.input = (SecantConstHostMatrixF32){input, num_inputs * num_rows, num_rows};
        cpu_run.leaf_masks = (SecantConstHostSpanU32){leaf_masks, num_settings};
        cpu_run.leaf_words =
            (SecantConstHostMatrixU32){leaf_words, num_settings * leaf_stride, leaf_stride};
        cpu_run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
        cpu_run.num_rows = num_rows;
        cpu_run.num_settings = num_settings;
        cpu_run.output = (SecantHostMatrixF32){cpu_sse, num_asts * num_settings, num_settings};
        if (evaluator_active && reducer_active && secant_cpu_run_dynamic_leaf_sse(&cpu_run) == SECANT_SUCCESS &&
            secant_sr_cubin_dynamic_leaf_evaluator_run(&evaluator, asts, num_asts, gpu_sse)) {
            SecantSRMSEEntry top_entries[7];
            size_t num_top_entries = 0u;

            status = 0;
            for (ast_idx = 0u; ast_idx < num_asts * num_settings; ++ast_idx) {
                const float tolerance = 0.0005f * (1.0f + fabsf(cpu_sse[ast_idx]));

                if (!isfinite(gpu_sse[ast_idx]) || fabsf(cpu_sse[ast_idx] - gpu_sse[ast_idx]) > tolerance) {
                    fprintf(stderr, "dynamic SSE mismatch idx=%zu cpu=%.9g gpu=%.9g tolerance=%.9g\n",
                        ast_idx, cpu_sse[ast_idx], gpu_sse[ast_idx], tolerance);
                    status = 1;
                    break;
                }
            }
            if (status == 0 &&
                (!secant_sr_mse_reducer_top_k_get(
                     &reducer,
                     evaluator.output,
                     num_asts,
                     num_settings,
                     num_settings,
                     num_rows,
                     secant_sr_cubin_test_target_ssd_get(target, num_rows),
                     7u,
                     top_entries,
                     7u,
                     &num_top_entries) ||
                 num_top_entries != 7u ||
                 !secant_sr_cubin_test_reduced_entries_validate(
                     cpu_sse,
                     num_asts,
                     num_settings,
                     num_rows,
                     secant_sr_cubin_test_target_ssd_get(target, num_rows),
                     top_entries,
                     num_top_entries))) {
                status = 1;
            }
            if (status == 0) {
                for (setting = 0u; setting < num_settings; ++setting) {
                    const float constant = -0.125f * (float)(setting + 1u);

                    leaf_masks[setting] = setting < 4u ? UINT32_C(2) : UINT32_C(3);
                    if ((leaf_masks[setting] & UINT32_C(1)) != 0u) {
                        leaf_words[setting * leaf_stride] = (uint32_t)(8u - setting);
                    } else {
                        memcpy(leaf_words + setting * leaf_stride, &constant, sizeof(constant));
                    }
                    leaf_words[setting * leaf_stride + 1u] = (uint32_t)setting;
                }
                memset(cpu_sse, 0, num_asts * num_settings * sizeof(*cpu_sse));
                if (!secant_sr_cubin_dynamic_leaf_settings_update(
                        &evaluator, leaf_masks, leaf_words, leaf_stride, num_settings) ||
                    secant_cpu_run_dynamic_leaf_sse(&cpu_run) != SECANT_SUCCESS ||
                    !secant_sr_cubin_dynamic_leaf_evaluator_run(&evaluator, asts, num_asts, gpu_sse)) {
                    status = 1;
                }
            }
            for (ast_idx = 0u; status == 0 && ast_idx < num_asts * num_settings; ++ast_idx) {
                const float tolerance = 0.0005f * (1.0f + fabsf(cpu_sse[ast_idx]));

                if (!isfinite(gpu_sse[ast_idx]) || fabsf(cpu_sse[ast_idx] - gpu_sse[ast_idx]) > tolerance) {
                    fprintf(stderr, "updated dynamic SSE mismatch idx=%zu cpu=%.9g gpu=%.9g tolerance=%.9g\n",
                        ast_idx, cpu_sse[ast_idx], gpu_sse[ast_idx], tolerance);
                    status = 1;
                }
            }
        }
    }
    if (reducer_active) {
        secant_sr_mse_reducer_destroy(&reducer);
    }
    if (evaluator_active) {
        secant_sr_cubin_evaluator_destroy(&evaluator);
    }
    free(gpu_sse);
    free(cpu_sse);
    free(target);
    free(input);
    free(asts);
    return status;
}

static int
secant_sr_cubin_dynamic_constant_test_run(const SecantSRCudaSession* session) {
    static const SecantAstInstruction ast[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_dynamic_constant_input_f32(0),
        secant_ast_encode_mul_f32,
        secant_ast_encode_dynamic_constant_input_f32(1),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    const size_t num_asts = 251u;
    const size_t num_rows = 257u;
    const size_t num_settings = 8u;
    SecantSRCubinEvaluator evaluator;
    SecantSRMSEReducer reducer;
    SecantCpuDynamicConstantSSERun cpu_run = secant_cpu_dynamic_constant_sse_run_init();
    const SecantAstInstruction** asts = NULL;
    float settings[2u * 8u];
    float* input = NULL;
    float* target = NULL;
    float* cpu_sse = NULL;
    float* gpu_sse = NULL;
    size_t ast_idx;
    size_t setting;
    int evaluator_active = 0;
    int reducer_active = 0;
    int status = 1;

    memset(&evaluator, 0, sizeof(evaluator));
    memset(&reducer, 0, sizeof(reducer));
    asts = malloc(num_asts * sizeof(*asts));
    input = malloc(num_rows * sizeof(*input));
    target = malloc(num_rows * sizeof(*target));
    cpu_sse = calloc(num_asts * num_settings, sizeof(*cpu_sse));
    gpu_sse = malloc(num_asts * num_settings * sizeof(*gpu_sse));
    if (asts != NULL && input != NULL && target != NULL && cpu_sse != NULL && gpu_sse != NULL) {
        size_t row;

        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            asts[ast_idx] = ast;
        }
        for (row = 0u; row < num_rows; ++row) {
            input[row] = -3.0f + 6.0f * (float)row / (float)(num_rows - 1u);
            target[row] = 2.0f * input[row] + 0.5f;
        }
        for (setting = 0u; setting < num_settings; ++setting) {
            settings[setting] = 0.5f * (float)setting;
            settings[num_settings + setting] = -1.0f + 0.5f * (float)setting;
        }
        evaluator_active = secant_sr_cubin_dynamic_constant_evaluator_create(
            8u,
            32u,
            1u,
            2u,
            256u,
            128u,
            1536u,
            4u,
            4u,
            input,
            target,
            settings,
            num_settings,
            num_settings,
            num_asts,
            num_rows,
            session,
            NULL,
            &evaluator);
        reducer_active = evaluator_active && secant_sr_mse_reducer_create(num_asts, 11u, session, NULL, &reducer);
        cpu_run.programs.asts.items = asts;
        cpu_run.programs.asts.count = num_asts;
        cpu_run.num_input_columns = 1u;
        cpu_run.num_input_constants = 2u;
        cpu_run.num_targets = 1u;
        cpu_run.input = (SecantConstHostMatrixF32){input, num_rows, num_rows};
        cpu_run.constant_settings =
            (SecantConstHostMatrixF32){settings, 2u * num_settings, num_settings};
        cpu_run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
        cpu_run.num_rows = num_rows;
        cpu_run.num_settings = num_settings;
        cpu_run.output = (SecantHostMatrixF32){cpu_sse, num_asts * num_settings, num_settings};
        if (evaluator_active && reducer_active && secant_cpu_run_dynamic_constant_sse(&cpu_run) == SECANT_SUCCESS &&
            secant_sr_cubin_dynamic_constant_evaluator_run(&evaluator, asts, num_asts, gpu_sse)) {
            SecantSRMSEEntry top_entries[11];
            size_t num_top_entries = 0u;

            status = 0;
            for (ast_idx = 0u; ast_idx < num_asts * num_settings; ++ast_idx) {
                const float tolerance = 0.0005f * (1.0f + fabsf(cpu_sse[ast_idx]));

                if (!isfinite(gpu_sse[ast_idx]) || fabsf(cpu_sse[ast_idx] - gpu_sse[ast_idx]) > tolerance) {
                    fprintf(stderr, "dynamic constant SSE mismatch idx=%zu cpu=%.9g gpu=%.9g tolerance=%.9g\n",
                        ast_idx, cpu_sse[ast_idx], gpu_sse[ast_idx], tolerance);
                    status = 1;
                    break;
                }
            }
            if (status == 0 &&
                (!secant_sr_mse_reducer_top_k_get(
                     &reducer,
                     evaluator.output,
                     num_asts,
                     num_settings,
                     num_settings,
                     num_rows,
                     secant_sr_cubin_test_target_ssd_get(target, num_rows),
                     11u,
                     top_entries,
                     11u,
                     &num_top_entries) ||
                 num_top_entries != 11u ||
                 !secant_sr_cubin_test_reduced_entries_validate(
                     cpu_sse,
                     num_asts,
                     num_settings,
                     num_rows,
                     secant_sr_cubin_test_target_ssd_get(target, num_rows),
                     top_entries,
                     num_top_entries))) {
                status = 1;
            }
        }
    }
    if (reducer_active) {
        secant_sr_mse_reducer_destroy(&reducer);
    }
    if (evaluator_active) {
        secant_sr_cubin_evaluator_destroy(&evaluator);
    }
    free(gpu_sse);
    free(cpu_sse);
    free(target);
    free(input);
    free(asts);
    return status;
}

static int
secant_sr_cubin_constant_optimizer_test_run(const SecantSRCudaSession* session) {
    static const SecantAstInstruction ast[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_dynamic_constant_input_f32(0),
        secant_ast_encode_mul_f32,
        secant_ast_encode_dynamic_constant_input_f32(1),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction* const asts[] = {ast, ast};
    enum { NUM_ASTS = 2, NUM_ROWS = 257, NUM_SETTINGS = 257, NUM_CONSTANTS = 2 };
    SecantSRCubinEvaluator evaluator;
    SecantCpuConstantOptimizerSSERun cpu_run = secant_cpu_constant_optimizer_sse_run_init();
    float input[NUM_ROWS];
    float target[NUM_ROWS];
    float initial_constants[NUM_ASTS * NUM_CONSTANTS] = {0.0f, 0.0f, 3.0f, -1.0f};
    float optimized_constants[NUM_ASTS * NUM_CONSTANTS];
    float cpu_sse[NUM_ASTS * NUM_SETTINGS];
    float gpu_sse[NUM_ASTS * NUM_SETTINGS];
    const uint64_t seed = UINT64_C(0x8c44a0bc77e2d491);
    const uint64_t generation = 5u;
    const uint64_t iteration = 2u;
    const uint64_t ast_index_base = 19u;
    const float scale = 4.0f;
    size_t row;
    size_t idx;
    int evaluator_active = 0;
    int success = 0;

    memset(&evaluator, 0, sizeof(evaluator));
    memset(cpu_sse, 0, sizeof(cpu_sse));
    for (row = 0u; row < NUM_ROWS; ++row) {
        input[row] = ((float)(int)(row % 41u) - 20.0f) * 0.1f;
        target[row] = 2.0f * input[row] + 1.0f;
    }
    cpu_run.programs.asts.items = asts;
    cpu_run.programs.asts.count = NUM_ASTS;
    cpu_run.num_input_columns = 1u;
    cpu_run.num_input_constants = NUM_CONSTANTS;
    cpu_run.input.data = input;
    cpu_run.input.num_elements = NUM_ROWS;
    cpu_run.input.leading_dimension = NUM_ROWS;
    cpu_run.current_constants.data = initial_constants;
    cpu_run.current_constants.num_elements = NUM_ASTS * NUM_CONSTANTS;
    cpu_run.current_constants.leading_dimension = NUM_CONSTANTS;
    cpu_run.target.data = target;
    cpu_run.target.num_elements = NUM_ROWS;
    cpu_run.target.leading_dimension = NUM_ROWS;
    cpu_run.num_rows = NUM_ROWS;
    cpu_run.num_settings = NUM_SETTINGS;
    cpu_run.seed = seed;
    cpu_run.generation = generation;
    cpu_run.iteration = iteration;
    cpu_run.ast_index_base = ast_index_base;
    cpu_run.perturbation_scale = scale;
    cpu_run.output.data = cpu_sse;
    cpu_run.output.num_elements = NUM_ASTS * NUM_SETTINGS;
    cpu_run.output.leading_dimension = NUM_SETTINGS;

    evaluator_active = secant_sr_cubin_constant_optimizer_evaluator_create(
        4u,
        1u,
        NUM_CONSTANTS,
        128u,
        128u,
        192u,
        2u,
        2u,
        input,
        target,
        NUM_SETTINGS,
        NUM_ASTS,
        NUM_ROWS,
        session,
        NULL,
        &evaluator);
    if (!evaluator_active || secant_cpu_run_constant_optimizer_sse(&cpu_run) != SECANT_SUCCESS ||
        !secant_sr_cubin_constant_optimizer_constants_upload(&evaluator, initial_constants, NUM_ASTS) ||
        !secant_sr_cubin_constant_optimizer_run(
            &evaluator, asts, NUM_ASTS, seed, generation, iteration, ast_index_base, scale, 0.5f, 1u) ||
        cuMemcpyDtoH(gpu_sse, evaluator.output, sizeof(gpu_sse)) != CUDA_SUCCESS ||
        !secant_sr_cubin_constant_optimizer_constants_download(
            &evaluator, optimized_constants, NUM_ASTS)) {
        goto cleanup;
    }
    for (idx = 0u; idx < NUM_ASTS * NUM_SETTINGS; ++idx) {
        const float tolerance = 2.0e-5f * (1.0f + fabsf(cpu_sse[idx]));

        if (!isfinite(gpu_sse[idx]) || fabsf(gpu_sse[idx] - cpu_sse[idx]) > tolerance) {
            goto cleanup;
        }
    }
    for (idx = 0u; idx < NUM_ASTS; ++idx) {
        float best_sse = cpu_sse[idx * NUM_SETTINGS];
        double reconstructed_sse = 0.0;
        size_t setting;

        for (setting = 1u; setting < NUM_SETTINGS; ++setting) {
            if (cpu_sse[idx * NUM_SETTINGS + setting] < best_sse) {
                best_sse = cpu_sse[idx * NUM_SETTINGS + setting];
            }
        }
        for (row = 0u; row < NUM_ROWS; ++row) {
            const double prediction =
                (double)input[row] * optimized_constants[idx * NUM_CONSTANTS] +
                optimized_constants[idx * NUM_CONSTANTS + 1u];
            const double difference = prediction - target[row];

            reconstructed_sse += difference * difference;
        }
        if (fabs(reconstructed_sse - best_sse) > 0.001 * (1.0 + fabs(best_sse)) ||
            best_sse >= cpu_sse[idx * NUM_SETTINGS]) {
            goto cleanup;
        }
    }
    success = 1;

cleanup:
    if (evaluator_active) {
        secant_sr_cubin_evaluator_destroy(&evaluator);
    }
    return success ? 0 : 1;
}

static int
secant_sr_cubin_packed_constant_optimizer_test_run(const SecantSRCudaSession* session) {
    static const SecantAstInstruction ast0[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_dynamic_constant_input_f32(0),
        secant_ast_encode_mul_f32,
        secant_ast_encode_dynamic_constant_input_f32(1),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction ast1[] = {
        secant_ast_encode_static_column_input_f32(1),
        secant_ast_encode_dynamic_constant_input_f32(0),
        secant_ast_encode_add_f32,
        secant_ast_encode_dynamic_constant_input_f32(1),
        secant_ast_encode_mul_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction* const asts[] = {ast0, ast1, ast0, ast1, ast0};
    enum {
        NUM_ASTS = 5,
        NUM_COLUMNS = 2,
        NUM_CONSTANTS = 2,
        NUM_ROWS = 257,
        NUM_SETTINGS = 257,
        NUM_ITERATIONS = 2,
        BEST_WIDTH = 1 + 4 * NUM_CONSTANTS
    };
    SecantSRCubinEvaluator evaluator;
    SecantCpuPackedConstantOptimizerSSERun cpu_run = secant_cpu_packed_constant_optimizer_sse_run_init();
    float input[NUM_COLUMNS * NUM_ROWS];
    float target[NUM_ROWS];
    float cpu_constants[NUM_ASTS * NUM_CONSTANTS];
    float gpu_constants[NUM_ASTS * NUM_CONSTANTS];
    float cpu_scales[NUM_ASTS * NUM_CONSTANTS];
    float gpu_scales[NUM_ASTS * NUM_CONSTANTS];
    float cpu_velocities[NUM_ASTS * NUM_CONSTANTS];
    float gpu_velocities[NUM_ASTS * NUM_CONSTANTS];
    float cpu_current_sse[NUM_ASTS];
    float gpu_current_sse[NUM_ASTS];
    float cpu_sse[NUM_ASTS * NUM_SETTINGS];
    float gpu_sse[NUM_ASTS * NUM_SETTINGS];
    float cpu_best[NUM_ASTS * BEST_WIDTH];
    size_t idx;
    int evaluator_active = 0;
    int success = 0;

    memset(&evaluator, 0, sizeof(evaluator));
    for (idx = 0u; idx < NUM_ROWS; ++idx) {
        const float x = ((float)(int)(idx % 61u) - 30.0f) * 0.1f;

        input[idx] = x;
        input[NUM_ROWS + idx] = 0.25f * x * x - 0.5f;
        target[idx] = 1.75f * x - 0.625f;
    }
    for (idx = 0u; idx < NUM_ASTS * NUM_CONSTANTS; ++idx) {
        const float center = ((float)(int)(idx % 7u) - 3.0f) * 0.25f;
        const float scale = 0.5f + 0.125f * (float)(idx % 5u);

        cpu_constants[idx] = center;
        gpu_constants[idx] = center;
        cpu_scales[idx] = scale;
        gpu_scales[idx] = scale;
        cpu_velocities[idx] = 0.0f;
        gpu_velocities[idx] = 0.0f;
    }
    for (idx = 0u; idx < NUM_ASTS; ++idx) {
        cpu_current_sse[idx] = INFINITY;
        gpu_current_sse[idx] = INFINITY;
    }
    cpu_run.programs.asts.items = asts;
    cpu_run.programs.asts.count = NUM_ASTS;
    cpu_run.programs.current_constants =
        (SecantHostMatrixF32){cpu_constants, NUM_ASTS * NUM_CONSTANTS, NUM_CONSTANTS};
    cpu_run.programs.current_constant_scales =
        (SecantHostMatrixF32){cpu_scales, NUM_ASTS * NUM_CONSTANTS, NUM_CONSTANTS};
    cpu_run.num_input_columns = NUM_COLUMNS;
    cpu_run.num_input_constants = NUM_CONSTANTS;
    cpu_run.input = (SecantConstHostMatrixF32){input, NUM_COLUMNS * NUM_ROWS, NUM_ROWS};
    cpu_run.target = (SecantConstHostMatrixF32){target, NUM_ROWS, NUM_ROWS};
    cpu_run.num_rows = NUM_ROWS;
    cpu_run.num_settings = NUM_SETTINGS;
    cpu_run.num_iterations = NUM_ITERATIONS;
    cpu_run.seed = UINT64_C(0xa1b2c3d4e5f60718);
    cpu_run.generation = 7u;
    cpu_run.iteration = 13u;
    cpu_run.current_constant_velocities =
        (SecantHostMatrixF32){cpu_velocities, NUM_ASTS * NUM_CONSTANTS, NUM_CONSTANTS};
    cpu_run.current_sse = (SecantHostSpanF32){cpu_current_sse, NUM_ASTS};
    cpu_run.momentum = 0.0f;
    cpu_run.scale_learning_rate = 0.25f;
    cpu_run.scale_failure_decay = 0.5f;
    cpu_run.minimum_scale = 1.0e-6f;
    cpu_run.maximum_scale = 1.0e6f;
    cpu_run.sse = (SecantHostMatrixF32){cpu_sse, NUM_ASTS * NUM_SETTINGS, NUM_SETTINGS};
    cpu_run.best = (SecantHostMatrixF32){cpu_best, NUM_ASTS * BEST_WIDTH, BEST_WIDTH};

    evaluator_active = secant_sr_cubin_packed_constant_optimizer_evaluator_create(
        2u,
        2u,
        NUM_COLUMNS,
        NUM_CONSTANTS,
        128u,
        128u,
        384u,
        2u,
        2u,
        input,
        target,
        NUM_SETTINGS,
        NUM_ASTS,
        NUM_ROWS,
        session,
        NULL,
        &evaluator);
    if (!evaluator_active || secant_cpu_run_packed_constant_optimizer_sse(&cpu_run) != SECANT_SUCCESS ||
        !secant_sr_cubin_packed_constant_optimizer_run(
            &evaluator,
            asts,
            NUM_ASTS,
            gpu_constants,
            gpu_scales,
            gpu_velocities,
            gpu_current_sse,
            cpu_run.seed,
            cpu_run.generation,
            cpu_run.iteration,
            NUM_ITERATIONS,
            cpu_run.momentum,
            cpu_run.scale_learning_rate,
            cpu_run.scale_failure_decay,
            cpu_run.minimum_scale,
            cpu_run.maximum_scale) ||
        evaluator.last_stats.modules_loaded != 4u ||
        cuMemcpyDtoH(gpu_sse, evaluator.output, sizeof(gpu_sse)) != CUDA_SUCCESS) {
        goto cleanup;
    }
    for (idx = 0u; idx < NUM_ASTS * NUM_SETTINGS; ++idx) {
        const float tolerance = 3.0e-4f * (1.0f + fabsf(cpu_sse[idx]));

        if (!isfinite(gpu_sse[idx]) || fabsf(gpu_sse[idx] - cpu_sse[idx]) > tolerance) {
            goto cleanup;
        }
    }
    for (idx = 0u; idx < NUM_ASTS * NUM_CONSTANTS; ++idx) {
        if (fabsf(gpu_constants[idx] - cpu_constants[idx]) > 3.0e-4f * (1.0f + fabsf(cpu_constants[idx])) ||
            fabsf(gpu_scales[idx] - cpu_scales[idx]) > 3.0e-4f * (1.0f + fabsf(cpu_scales[idx])) ||
            fabsf(gpu_velocities[idx] - cpu_velocities[idx]) > 3.0e-4f * (1.0f + fabsf(cpu_velocities[idx]))) {
            goto cleanup;
        }
    }
    for (idx = 0u; idx < NUM_ASTS; ++idx) {
        if (fabsf(gpu_current_sse[idx] - cpu_current_sse[idx]) >
            3.0e-4f * (1.0f + fabsf(cpu_current_sse[idx]))) {
            goto cleanup;
        }
    }
    success = 1;

cleanup:
    if (evaluator_active) {
        secant_sr_cubin_evaluator_destroy(&evaluator);
    }
    return success ? 0 : 1;
}

static int
secant_sr_cubin_mse_reducer_invalid_test_run(const SecantSRCudaSession* session) {
    const size_t num_asts = 17u;
    const size_t num_settings = 3u;
    float sse[17u * 3u];
    SecantSRMSEEntry entries[5];
    SecantSRMSEReducer reducer;
    CUdeviceptr device_sse = 0u;
    size_t num_entries = 0u;
    size_t idx;
    int reducer_active = 0;
    int success = 0;

    memset(&reducer, 0, sizeof(reducer));
    for (idx = 0u; idx < num_asts * num_settings; ++idx) {
        sse[idx] = INFINITY;
    }
    if (cuMemAlloc(&device_sse, sizeof(sse)) == CUDA_SUCCESS &&
        cuMemcpyHtoD(device_sse, sse, sizeof(sse)) == CUDA_SUCCESS) {
        reducer_active = secant_sr_mse_reducer_create(num_asts, 5u, session, NULL, &reducer);
    }
    if (reducer_active && secant_sr_mse_reducer_top_k_get(
            &reducer,
            device_sse,
            num_asts,
            num_settings,
            num_settings,
            257u,
            1.0,
            5u,
            entries,
            5u,
            &num_entries) &&
        num_entries == 0u) {
        success = 1;
    } else {
        fprintf(stderr,
            "invalid reducer execution failed active=%d num_entries=%zu\n",
            reducer_active,
            num_entries);
    }
    if (reducer_active) {
        secant_sr_mse_reducer_destroy(&reducer);
    }
    if (device_sse != 0u) {
        (void)cuMemFree(device_sse);
    }
    return success ? 0 : 1;
}

int
main(void) {
    SecantSRCudaSession session;
    int status;

    if (!secant_sr_cuda_session_create(&session)) {
        return 1;
    }
    status = secant_sr_cubin_test_run(&session) || secant_sr_cubin_dynamic_leaf_test_run(&session, 0) ||
        secant_sr_cubin_dynamic_leaf_test_run(&session, 1) ||
        secant_sr_cubin_dynamic_constant_test_run(&session) ||
        secant_sr_cubin_constant_optimizer_test_run(&session) ||
        secant_sr_cubin_packed_constant_optimizer_test_run(&session) ||
        secant_sr_cubin_mse_reducer_invalid_test_run(&session);
    secant_sr_cuda_session_destroy(&session);
    return status;
}
