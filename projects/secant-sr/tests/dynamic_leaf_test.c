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
#include "secant_sr.h"
#include "secant_instructions.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static SecantResult
test_cpu_sse_run(
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    const float* target,
    size_t num_rows,
    float* output
) {
    SecantCpuSSERun run = secant_cpu_sse_run_init();

    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.num_inputs = 2u;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, 2u * num_rows, num_rows};
    run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
    run.num_rows = num_rows;
    run.output = (SecantHostMatrixF32){output, num_asts, 1u};
    return secant_cpu_run_sse(&run);
}

static SecantResult
test_cpu_dynamic_leaf_sse_run(
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_stride,
    size_t num_settings,
    const float* target,
    size_t num_rows,
    float* output
) {
    SecantCpuDynamicLeafSSERun run = secant_cpu_dynamic_leaf_sse_run_init();

    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.num_dynamic_leaves = SECANT_AST_MAX_DYNAMIC_LEAVES;
    run.num_input_columns = 2u;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, 2u * num_rows, num_rows};
    run.leaf_masks = (SecantConstHostSpanU32){leaf_masks, num_settings};
    run.leaf_words = (SecantConstHostMatrixU32){leaf_words, num_settings * leaf_stride, leaf_stride};
    run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
    run.num_rows = num_rows;
    run.num_settings = num_settings;
    run.output = (SecantHostMatrixF32){output, num_asts * num_settings, num_settings};
    return secant_cpu_run_dynamic_leaf_sse(&run);
}

static double
target_build(float* input, float* target, size_t num_rows) {
    double mean = 0.0;
    double sum_squared_deviation = 0.0;
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        const float x = -1.0f + 2.0f * (float)row / (float)(num_rows - 1u);
        const float y = cosf((float)row * 0.03125f);

        input[row] = x;
        input[num_rows + row] = y;
        target[row] = x + 0.5f * y;
        mean += target[row];
    }
    mean /= (double)num_rows;
    for (row = 0u; row < num_rows; ++row) {
        const double difference = (double)target[row] - mean;

        sum_squared_deviation += difference * difference;
    }
    return sum_squared_deviation;
}

static int
projected_program_validate(const SecantAstInstruction* program) {
    size_t offset = 0u;
    size_t expected_leaf = 0u;

    for (;;) {
        const SecantAstInstructionType type = secant_ast_instruction_type_get(program + offset);
        const size_t instruction_size = secant_ast_instruction_size_get(program + offset);

        if (instruction_size == 0u ||
            type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 ||
            type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
            if (secant_ast_index_get(program + offset) != expected_leaf) {
                return 0;
            }
            ++expected_leaf;
        }
        offset += instruction_size;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return expected_leaf != 0u;
        }
    }
}

static int
dynamic_leaf_test_execute(
    SecantSRSearch search,
    const SecantSRSearchConfig* config,
    SecantAstInstruction* projected_storage,
    size_t projected_storage_capacity,
    const SecantAstInstruction** projected_asts,
    uint32_t* leaf_words,
    float* input,
    float* target,
    float* dynamic_sse,
    float* concrete_sse,
    float* source_sse,
    float* proposal_sse,
    uint32_t* best_setting_indices,
    size_t* proposal_program_sizes,
    uint64_t* source_fingerprints
) {
    const size_t num_rows = 257u;
    const size_t num_settings = 4u;
    const size_t leaf_stride = SECANT_AST_MAX_DYNAMIC_LEAVES;
    const uint32_t leaf_masks[4] = {UINT32_MAX, 0u, UINT32_MAX, 0u};
    size_t projected_storage_size = 0u;
    size_t num_asts = 0u;
    size_t max_dynamic_leaves = 0u;
    double target_ssd;
    size_t ast_idx;

    if (secant_sr_search_dynamic_leaf_programs_write(
            search,
            NULL,
            0u,
            NULL,
            0u,
            &projected_storage_size,
            &num_asts,
            &max_dynamic_leaves) != SECANT_SR_SUCCESS ||
        projected_storage_size > projected_storage_capacity || num_asts != config->population_size ||
        max_dynamic_leaves == 0u ||
        secant_sr_search_dynamic_leaf_programs_write(
            search,
            projected_storage,
            projected_storage_capacity,
            projected_asts,
            config->population_size,
            &projected_storage_size,
            &num_asts,
            &max_dynamic_leaves) != SECANT_SR_SUCCESS) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        if (!projected_program_validate(projected_asts[ast_idx])) {
            fprintf(stderr, "invalid projected AST %zu\n", ast_idx);
            return 0;
        }
    }
    for (ast_idx = 0u; ast_idx < num_settings * leaf_stride; ++ast_idx) {
        const size_t setting = ast_idx / leaf_stride;
        const size_t leaf = ast_idx % leaf_stride;

        if ((leaf_masks[setting] & (UINT32_C(1) << leaf)) != 0u) {
            leaf_words[ast_idx] = (uint32_t)((setting + leaf) % 2u);
        } else {
            const float value = setting == 1u ? 0.5f : -1.0f;

            memcpy(leaf_words + ast_idx, &value, sizeof(value));
        }
    }
    target_ssd = target_build(input, target, num_rows);
    if (test_cpu_dynamic_leaf_sse_run(
            projected_asts,
            num_asts,
            input,
            leaf_masks,
            leaf_words,
            leaf_stride,
            num_settings,
            target,
            num_rows,
            dynamic_sse) != SECANT_SUCCESS) {
        return 0;
    }
    {
        const SecantAstInstruction* const* source_asts;
        const SecantSRIndividual* source_individuals;
        size_t num_source_asts;
        size_t num_source_individuals;

        if (secant_sr_search_asts_get(search, &source_asts, &num_source_asts) != SECANT_SR_SUCCESS ||
            secant_sr_search_individuals_get(search, &source_individuals, &num_source_individuals) !=
                SECANT_SR_SUCCESS ||
            num_source_asts != num_asts || num_source_individuals != num_asts ||
            test_cpu_sse_run(
                source_asts,
                num_asts,
                input,
                target,
                num_rows,
                source_sse) != SECANT_SUCCESS ||
            secant_sr_search_scores_set(search, source_sse, num_asts, num_rows, target_ssd) != SECANT_SR_SUCCESS) {
            return 0;
        }
        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            source_fingerprints[ast_idx] = source_individuals[ast_idx].fingerprint;
        }
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        float best_sse = INFINITY;
        size_t setting;

        for (setting = 0u; setting < num_settings; ++setting) {
            const float candidate = dynamic_sse[ast_idx * num_settings + setting];

            if (isfinite(candidate) && candidate >= 0.0f && candidate < best_sse) {
                best_sse = candidate;
                best_setting_indices[ast_idx] = (uint32_t)setting;
            }
        }
    }
    if (secant_sr_search_dynamic_leaf_proposals_from_settings_write(
            search,
            leaf_masks,
            num_settings,
            leaf_words,
            num_settings * leaf_stride,
            leaf_stride,
            num_settings,
            best_setting_indices,
            num_asts,
            projected_storage,
            projected_storage_capacity,
            projected_asts,
            proposal_program_sizes,
            config->population_size,
            &projected_storage_size) != SECANT_SR_SUCCESS ||
        test_cpu_sse_run(
            projected_asts,
            num_asts,
            input,
            target,
            num_rows,
            proposal_sse) != SECANT_SUCCESS) {
        return 0;
    }
    proposal_sse[0] = INFINITY;
    if (secant_sr_search_proposals_apply(
            search,
            projected_asts,
            proposal_program_sizes,
            proposal_sse,
            NULL,
            num_asts,
            num_rows,
            target_ssd,
            SECANT_SR_ORIGIN_DYNAMIC_LEAF,
            NULL) != SECANT_SR_SUCCESS) {
        return 0;
    }
    {
        const SecantAstInstruction* const* concrete_asts;
        const SecantSRIndividual* individuals;
        size_t num_concrete_asts;
        size_t num_individuals;

        if (secant_sr_search_asts_get(search, &concrete_asts, &num_concrete_asts) != SECANT_SR_SUCCESS ||
            secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS ||
            num_concrete_asts != num_asts || num_individuals != num_asts ||
            test_cpu_sse_run(
                concrete_asts,
                num_concrete_asts,
                input,
                target,
                num_rows,
                concrete_sse) != SECANT_SUCCESS) {
            return 0;
        }
        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            const int promoted = isfinite(proposal_sse[ast_idx]) && proposal_sse[ast_idx] < source_sse[ast_idx];
            const float expected = promoted ? proposal_sse[ast_idx] : source_sse[ast_idx];

            if (!individuals[ast_idx].scored ||
                (promoted && individuals[ast_idx].origin != SECANT_SR_ORIGIN_DYNAMIC_LEAF) ||
                (!promoted && individuals[ast_idx].fingerprint != source_fingerprints[ast_idx]) ||
                fabsf(concrete_sse[ast_idx] - expected) > 1.0e-4f * (1.0f + fabsf(expected)) ||
                fabsf((float)individuals[ast_idx].fitness.sse - expected) > 1.0e-4f * (1.0f + fabsf(expected))) {
                fprintf(stderr, "compare-and-promote mismatch ast=%zu expected=%.9g actual=%.9g promoted=%d\n",
                    ast_idx, expected, (float)individuals[ast_idx].fitness.sse, promoted);
                return 0;
            }
        }
    }
    return 1;
}

static int
dynamic_leaf_test_run(void) {
    static const SecantAstInstructionType unary_ops[] = {SECANT_AST_INSTRUCTION_TYPE_NEG_F32};
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32
    };
    static const float constants[] = {-1.0f, 0.5f, 1.0f, 2.0f};
    const SecantSRSearchConfig config = {
        64u, 144u, 48u, 10u, 24u, 2u, 8u, 2u, 8u, 4u, 1.6,
        0.45, 0.30, 0.20, 0.25, 0.00005, UINT64_C(0x726f6a656374), 0u
    };
    const size_t num_rows = 257u;
    const size_t num_settings = 4u;
    const size_t leaf_stride = SECANT_AST_MAX_DYNAMIC_LEAVES;
    SecantSRSearch search = NULL;
    void* search_storage = NULL;
    SecantAstInstruction* projected_storage = NULL;
    const SecantAstInstruction** projected_asts = NULL;
    uint32_t* leaf_words = NULL;
    float* input = NULL;
    float* target = NULL;
    float* dynamic_sse = NULL;
    float* concrete_sse = NULL;
    float* source_sse = NULL;
    float* proposal_sse = NULL;
    uint32_t* best_setting_indices = NULL;
    size_t* proposal_program_sizes = NULL;
    uint64_t* source_fingerprints = NULL;
    size_t search_storage_size = 0u;
    size_t projected_storage_capacity;
    int status = 1;

    if (secant_sr_search_storage_size(
            &config,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            0u,
            sizeof(constants) / sizeof(constants[0]),
            &search_storage_size) != SECANT_SR_SUCCESS) {
        return 1;
    }
    search_storage = malloc(search_storage_size);
    projected_storage_capacity = config.population_size * config.max_program_bytes;
    projected_storage = malloc(projected_storage_capacity);
    input = malloc(2u * num_rows * sizeof(*input));
    target = malloc(num_rows * sizeof(*target));
    projected_asts = malloc(config.population_size * sizeof(*projected_asts));
    leaf_words = malloc(num_settings * leaf_stride * sizeof(*leaf_words));
    dynamic_sse = calloc(config.population_size * num_settings, sizeof(*dynamic_sse));
    concrete_sse = calloc(config.population_size, sizeof(*concrete_sse));
    source_sse = calloc(config.population_size, sizeof(*source_sse));
    proposal_sse = calloc(config.population_size, sizeof(*proposal_sse));
    best_setting_indices = calloc(config.population_size, sizeof(*best_setting_indices));
    proposal_program_sizes = malloc(config.population_size * sizeof(*proposal_program_sizes));
    source_fingerprints = malloc(config.population_size * sizeof(*source_fingerprints));
    if (search_storage != NULL && projected_storage != NULL && input != NULL && target != NULL &&
        projected_asts != NULL && leaf_words != NULL && dynamic_sse != NULL && concrete_sse != NULL &&
        source_sse != NULL && proposal_sse != NULL && best_setting_indices != NULL &&
        proposal_program_sizes != NULL && source_fingerprints != NULL &&
        secant_sr_search_init(
            &config,
            unary_ops,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            binary_ops,
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            NULL,
            0u,
            constants,
            sizeof(constants) / sizeof(constants[0]),
            search_storage,
            search_storage_size,
            &search) == SECANT_SR_SUCCESS &&
        dynamic_leaf_test_execute(
            search,
            &config,
            projected_storage,
            projected_storage_capacity,
            projected_asts,
            leaf_words,
            input,
            target,
            dynamic_sse,
            concrete_sse,
            source_sse,
            proposal_sse,
            best_setting_indices,
            proposal_program_sizes,
            source_fingerprints)) {
        status = 0;
    }
    free(source_fingerprints);
    free(proposal_program_sizes);
    free(proposal_sse);
    free(best_setting_indices);
    free(source_sse);
    free(concrete_sse);
    free(dynamic_sse);
    free(leaf_words);
    free(projected_asts);
    free(projected_storage);
    free(target);
    free(input);
    free(search_storage);
    return status;
}

int
main(void) {
    return dynamic_leaf_test_run();
}
