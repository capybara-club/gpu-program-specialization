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
#define _POSIX_C_SOURCE 200809L

#include "secant_sr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
seconds_get(void) {
    struct timespec value;

    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int
secant_sr_bench_run(size_t population_size, size_t generations, size_t max_nodes, uint64_t seed) {
    static const SecantAstInstructionType unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32,
        SECANT_AST_INSTRUCTION_TYPE_ABS_F32,
        SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_COS_F32
    };
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32,
        SECANT_AST_INSTRUCTION_TYPE_MIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_MAX_F32
    };
    static const float constants[] = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
    SecantSRSearchConfig config;
    SecantSRSearch search = NULL;
    const SecantSRIndividual* individuals;
    void* storage = NULL;
    float* sse = NULL;
    size_t storage_size = 0u;
    size_t num_individuals = 0u;
    size_t generation;
    double init_begin;
    double init_seconds;
    double generation_seconds = 0.0;
    int status = 1;

    memset(&config, 0, sizeof(config));
    config.population_size = population_size;
    config.max_program_bytes = max_nodes * 5u + 1u;
    config.max_nodes = max_nodes;
    config.max_depth = max_nodes < 24u ? max_nodes : 24u;
    config.max_complexity = (uint32_t)(max_nodes < 64u ? max_nodes : 64u);
    config.num_inputs = 8u;
    config.num_complexity_buckets = 12u;
    config.elites_per_bucket = 4u;
    config.elite_copies_per_generation = population_size < 32u ? population_size : 32u;
    config.tournament_size = 5u;
    config.complexity_factor = 1.5;
    config.crossover_probability = 0.45;
    config.subtree_mutation_probability = 0.30;
    config.point_mutation_probability = 0.20;
    config.constant_leaf_probability = 0.15;
    config.parsimony_coefficient = 0.00005;
    config.seed = seed;

    if (secant_sr_search_storage_size(
            &config,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            0u,
            sizeof(constants) / sizeof(constants[0]),
            &storage_size) != SECANT_SR_SUCCESS) {
        return 1;
    }
    storage = malloc(storage_size);
    sse = malloc(population_size * sizeof(*sse));
    if (storage != NULL && sse != NULL) {
        init_begin = seconds_get();
        if (secant_sr_search_init(
                &config,
                unary_ops,
                sizeof(unary_ops) / sizeof(unary_ops[0]),
                binary_ops,
                sizeof(binary_ops) / sizeof(binary_ops[0]),
                NULL,
                0u,
                constants,
                sizeof(constants) / sizeof(constants[0]),
                storage,
                storage_size,
                &search) == SECANT_SR_SUCCESS) {
            init_seconds = seconds_get() - init_begin;
            for (generation = 0u; generation < generations; ++generation) {
                size_t individual_idx;
                double begin;

                if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS) {
                    break;
                }
                for (individual_idx = 0u; individual_idx < num_individuals; ++individual_idx) {
                    sse[individual_idx] = (float)(individuals[individual_idx].complexity +
                        (individuals[individual_idx].fingerprint & UINT64_C(0xffff)) * (1.0 / 65536.0));
                }
                if (secant_sr_search_scores_set(search, sse, num_individuals, 1024u, 1024.0) != SECANT_SR_SUCCESS) {
                    break;
                }
                begin = seconds_get();
                if (secant_sr_search_generation_advance(search) != SECANT_SR_SUCCESS) {
                    break;
                }
                generation_seconds += seconds_get() - begin;
            }
            if (generation == generations) {
                printf(
                    "population=%zu generations=%zu max_nodes=%zu storage_bytes=%zu storage_mib=%.3f "
                    "init_seconds=%.9f init_asts_per_second=%.3f generation_seconds=%.9f "
                    "generated_asts_per_second=%.3f\n",
                    population_size,
                    generations,
                    max_nodes,
                    storage_size,
                    (double)storage_size / (1024.0 * 1024.0),
                    init_seconds,
                    (double)population_size / init_seconds,
                    generation_seconds,
                    (double)(population_size * generations) / generation_seconds);
                status = 0;
            }
        }
    }
    free(sse);
    free(storage);
    return status;
}

int
main(int argc, char** argv) {
    size_t population_size = 100000u;
    size_t generations = 10u;
    size_t max_nodes = 24u;
    uint64_t seed = UINT64_C(0x5eca7a123);
    int arg_idx;

    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--population") == 0 && arg_idx + 1 < argc) {
            population_size = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--generations") == 0 && arg_idx + 1 < argc) {
            generations = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--max-nodes") == 0 && arg_idx + 1 < argc) {
            max_nodes = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--seed") == 0 && arg_idx + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++arg_idx], NULL, 0);
        } else {
            fprintf(stderr, "usage: %s [--population N] [--generations N] [--max-nodes N] [--seed N]\n", argv[0]);
            return 1;
        }
    }
    if (population_size == 0u || generations == 0u || max_nodes == 0u || max_nodes >= 1024u) {
        return 1;
    }
    return secant_sr_bench_run(population_size, generations, max_nodes, seed);
}
