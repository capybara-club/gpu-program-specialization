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
#include "secant_sindy.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int size_parse(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int u32_parse(const char* text, uint32_t* value_ret) {
    size_t value;

    if (!size_parse(text, &value) || value > UINT32_MAX) {
        return 0;
    }
    *value_ret = (uint32_t)value;
    return 1;
}

static void usage(const char* executable) {
    fprintf(stderr,
            "usage:\n"
            "  %s target-stats [-o output.cu]\n"
            "  %s solver <cc-major> <cc-minor> [features targets stlsq-iterations] [-o output.cu]\n",
            executable, executable);
}

int main(int argc, char** argv) {
    SecantSindyCudaSolverRecipe recipe = secant_sindy_cuda_solver_recipe_init();
    const char* output_path = NULL;
    size_t source_size = 0u;
    char* allocated_source = NULL;
    const char* source;
    FILE* output = stdout;
    SecantSindyResult result;
    int solver;
    int arg_idx;

    if (argc < 2 || (strcmp(argv[1], "target-stats") != 0 && strcmp(argv[1], "solver") != 0)) {
        usage(argv[0]);
        return 2;
    }
    solver = strcmp(argv[1], "solver") == 0;
    recipe.feature_capacity = 32u;
    recipe.max_targets = 4u;
    recipe.max_stlsq_iterations = 32u;

    arg_idx = 2;
    if (solver) {
        if (argc - arg_idx < 2 || !u32_parse(argv[arg_idx], &recipe.compute_capability_major) ||
            !u32_parse(argv[arg_idx + 1], &recipe.compute_capability_minor)) {
            usage(argv[0]);
            return 2;
        }
        arg_idx += 2;
        if (arg_idx < argc && strcmp(argv[arg_idx], "-o") != 0) {
            if (argc - arg_idx < 3 || !size_parse(argv[arg_idx], &recipe.feature_capacity) ||
                !size_parse(argv[arg_idx + 1], &recipe.max_targets) ||
                !size_parse(argv[arg_idx + 2], &recipe.max_stlsq_iterations)) {
                usage(argv[0]);
                return 2;
            }
            arg_idx += 3;
        }
    }
    if (arg_idx < argc) {
        if (arg_idx + 2 != argc || strcmp(argv[arg_idx], "-o") != 0) {
            usage(argv[0]);
            return 2;
        }
        output_path = argv[arg_idx + 1];
    }

    if (solver) {
        result = secant_sindy_cuda_solver_source_size(&recipe, &source_size);
        if (result != SECANT_SINDY_SUCCESS) {
            fprintf(stderr, "source size failed: %s\n", secant_sindy_result_to_string(result));
            return 1;
        }
        allocated_source = (char*)malloc(source_size);
        if (allocated_source == NULL) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        result = secant_sindy_cuda_solver_source_write(&recipe, allocated_source, source_size);
        if (result != SECANT_SINDY_SUCCESS) {
            fprintf(stderr, "source write failed: %s\n", secant_sindy_result_to_string(result));
            free(allocated_source);
            return 1;
        }
        source = allocated_source;
    } else {
        source = secant_sindy_cuda_target_stats_source_get(&source_size);
    }
    if (output_path != NULL) {
        output = fopen(output_path, "wb");
        if (output == NULL) {
            fprintf(stderr, "failed to open %s\n", output_path);
            free(allocated_source);
            return 1;
        }
    }
    if (fwrite(source, 1u, source_size - 1u, output) != source_size - 1u) {
        fprintf(stderr, "failed to write CUDA source\n");
        if (output != stdout) {
            fclose(output);
        }
        free(allocated_source);
        return 1;
    }
    if (output != stdout && fclose(output) != 0) {
        fprintf(stderr, "failed to close %s\n", output_path);
        free(allocated_source);
        return 1;
    }
    free(allocated_source);
    return 0;
}
