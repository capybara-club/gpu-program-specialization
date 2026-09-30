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
#ifndef SECANT_BENCH_COMMON_H_INCLUDED
#define SECANT_BENCH_COMMON_H_INCLUDED

#include "bench_ast.h"

#define SECANT_PORTABLE_ALU_INCLUDE_NATIVE
#include "corpus/portable_alu_v1.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum SecantBenchShape {
    SECANT_BENCH_SHAPE_MATERIALIZE = 0,
    SECANT_BENCH_SHAPE_SSE = 1
} SecantBenchShape;

typedef enum SecantBenchBackend {
    SECANT_BENCH_BACKEND_CUDA = 0,
    SECANT_BENCH_BACKEND_PTX = 1,
    SECANT_BENCH_BACKEND_CUBIN = 2
} SecantBenchBackend;

typedef struct SecantBenchOptions {
    SecantBenchBackend backend;
    SecantBenchShape shape;
    SecantBenchAstMode ast_mode;
    size_t warmups;
    size_t iterations;
    size_t workers;
    size_t compile_scratch_size;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t num_streams;
    size_t patch_instructions_per_ast;
    size_t check_rows;
    size_t check_modules;
    size_t run_rows;
    size_t run_iterations;
    uint32_t seed;
    uint32_t source_sm;
    uint32_t target_sm;
    uint32_t opt_level;
    int device_ordinal;
} SecantBenchOptions;

static int
secant_bench_parse_size(const char* text, size_t* value_ret) {
    unsigned long long value;
    char* end = NULL;

    if (text == NULL || value_ret == NULL || text[0] == '\0' || text[0] == '-') {
        return 0;
    }
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
secant_bench_parse_u32(const char* text, uint32_t* value_ret) {
    size_t value;

    if (!secant_bench_parse_size(text, &value) || value > UINT32_MAX) {
        return 0;
    }
    *value_ret = (uint32_t)value;
    return 1;
}

static int
secant_bench_parse_int(const char* text, int* value_ret) {
    long value;
    char* end = NULL;

    if (text == NULL || value_ret == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    value = strtol(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value < INT32_MIN || value > INT32_MAX) {
        return 0;
    }
    *value_ret = (int)value;
    return 1;
}

static const char*
secant_bench_backend_name(SecantBenchBackend backend) {
    switch (backend) {
        case SECANT_BENCH_BACKEND_CUDA: return "cuda";
        case SECANT_BENCH_BACKEND_PTX: return "ptx";
        case SECANT_BENCH_BACKEND_CUBIN: return "cubin";
        default: return "unknown";
    }
}

static const char*
secant_bench_shape_name(SecantBenchShape shape) {
    return shape == SECANT_BENCH_SHAPE_SSE ? "sse" : "materialize";
}

static const char*
secant_bench_ast_mode_name(SecantBenchAstMode mode) {
    switch (mode) {
        case SECANT_BENCH_AST_MODE_SIMPLE: return "simple";
        case SECANT_BENCH_AST_MODE_ALU: return "alu";
        case SECANT_BENCH_AST_MODE_MUFU: return "mufu";
        default: return "unknown";
    }
}

static void
secant_bench_options_default(SecantBenchOptions* options) {
    memset(options, 0, sizeof(*options));
    options->backend = SECANT_BENCH_BACKEND_PTX;
    options->shape = SECANT_BENCH_SHAPE_MATERIALIZE;
    options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
    options->warmups = 2u;
    options->iterations = 16u;
    options->workers = 1u;
    options->compile_scratch_size = 64u * 1024u * 1024u;
    options->num_kernels = 8u;
    options->asts_per_kernel = 8u;
    options->num_inputs = 8u;
    options->num_targets = 1u;
    options->tile_rows = 4096u;
    options->threads_per_block = 128u;
    options->num_streams = 1u;
    options->patch_instructions_per_ast = 64u;
    options->check_rows = 257u;
    options->check_modules = 1u;
    options->run_rows = 1u << 20;
    options->run_iterations = 20u;
    options->seed = 1u;
    options->source_sm = 80u;
    options->target_sm = 120u;
    options->opt_level = 1u;
    options->device_ordinal = 0;
}

static int
secant_bench_parse_common_option(
    int argc,
    char** argv,
    int* arg_idx,
    SecantBenchOptions* options
) {
    const char* name = argv[*arg_idx];
    const char* value;

    if (*arg_idx + 1 >= argc) {
        return 0;
    }
    value = argv[++*arg_idx];
    if (strcmp(name, "--backend") == 0) {
        if (strcmp(value, "cuda") == 0) {
            options->backend = SECANT_BENCH_BACKEND_CUDA;
        } else if (strcmp(value, "ptx") == 0) {
            options->backend = SECANT_BENCH_BACKEND_PTX;
        } else if (strcmp(value, "cubin") == 0) {
            options->backend = SECANT_BENCH_BACKEND_CUBIN;
        } else {
            return 0;
        }
    } else if (strcmp(name, "--shape") == 0) {
        if (strcmp(value, "materialize") == 0) {
            options->shape = SECANT_BENCH_SHAPE_MATERIALIZE;
        } else if (strcmp(value, "sse") == 0) {
            options->shape = SECANT_BENCH_SHAPE_SSE;
        } else {
            return 0;
        }
    } else if (strcmp(name, "--ast-mode") == 0) {
        if (strcmp(value, "simple") == 0) {
            options->ast_mode = SECANT_BENCH_AST_MODE_SIMPLE;
        } else if (strcmp(value, "alu") == 0) {
            options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
        } else if (strcmp(value, "mufu") == 0) {
            options->ast_mode = SECANT_BENCH_AST_MODE_MUFU;
        } else {
            return 0;
        }
    } else if (strcmp(name, "--warmups") == 0) {
        return secant_bench_parse_size(value, &options->warmups);
    } else if (strcmp(name, "--iterations") == 0) {
        return secant_bench_parse_size(value, &options->iterations);
    } else if (strcmp(name, "--modules") == 0) {
        return secant_bench_parse_size(value, &options->iterations);
    } else if (strcmp(name, "--workers") == 0) {
        return secant_bench_parse_size(value, &options->workers);
    } else if (strcmp(name, "--compile-scratch-bytes") == 0) {
        return secant_bench_parse_size(value, &options->compile_scratch_size);
    } else if (strcmp(name, "--kernels") == 0) {
        return secant_bench_parse_size(value, &options->num_kernels);
    } else if (strcmp(name, "--asts-per-kernel") == 0) {
        return secant_bench_parse_size(value, &options->asts_per_kernel);
    } else if (strcmp(name, "--inputs") == 0) {
        return secant_bench_parse_size(value, &options->num_inputs);
    } else if (strcmp(name, "--targets") == 0) {
        return secant_bench_parse_size(value, &options->num_targets);
    } else if (strcmp(name, "--tile-rows") == 0) {
        return secant_bench_parse_size(value, &options->tile_rows);
    } else if (strcmp(name, "--threads") == 0) {
        return secant_bench_parse_size(value, &options->threads_per_block);
    } else if (strcmp(name, "--streams") == 0) {
        return secant_bench_parse_size(value, &options->num_streams);
    } else if (strcmp(name, "--patch-instructions-per-ast") == 0) {
        return secant_bench_parse_size(
            value,
            &options->patch_instructions_per_ast);
    } else if (strcmp(name, "--check-rows") == 0) {
        return secant_bench_parse_size(value, &options->check_rows);
    } else if (strcmp(name, "--check-modules") == 0) {
        return secant_bench_parse_size(value, &options->check_modules);
    } else if (strcmp(name, "--run-rows") == 0) {
        return secant_bench_parse_size(value, &options->run_rows);
    } else if (strcmp(name, "--run-iterations") == 0) {
        return secant_bench_parse_size(value, &options->run_iterations);
    } else if (strcmp(name, "--seed") == 0) {
        return secant_bench_parse_u32(value, &options->seed);
    } else if (strcmp(name, "--source-sm") == 0) {
        return secant_bench_parse_u32(value, &options->source_sm);
    } else if (strcmp(name, "--target-sm") == 0) {
        return secant_bench_parse_u32(value, &options->target_sm);
    } else if (strcmp(name, "--opt-level") == 0) {
        return secant_bench_parse_u32(value, &options->opt_level);
    } else if (strcmp(name, "--device") == 0) {
        return secant_bench_parse_int(value, &options->device_ordinal);
    } else {
        return 0;
    }
    return 1;
}

static int
secant_bench_options_valid(const SecantBenchOptions* options) {
    return options->iterations != 0u &&
        options->workers != 0u &&
        options->workers <= INT_MAX &&
        options->compile_scratch_size != 0u &&
        options->num_kernels != 0u &&
        options->asts_per_kernel != 0u &&
        options->num_inputs != 0u &&
        (options->ast_mode != SECANT_BENCH_AST_MODE_ALU ||
            options->num_inputs == 8u) &&
        options->num_targets != 0u &&
        options->tile_rows != 0u &&
        options->threads_per_block != 0u &&
        options->num_streams != 0u &&
        options->num_streams <= options->num_kernels &&
        options->patch_instructions_per_ast != 0u &&
        options->check_rows != 0u &&
        options->run_rows != 0u &&
        options->run_iterations != 0u &&
        options->source_sm >= 10u &&
        options->target_sm >= 10u &&
        options->opt_level <= 1u &&
        options->warmups <= SIZE_MAX - options->iterations &&
        options->asts_per_kernel <= SIZE_MAX / options->num_kernels &&
        options->num_inputs <= SIZE_MAX / options->run_rows &&
        options->num_targets <= SIZE_MAX / options->run_rows &&
        options->asts_per_kernel <= SIZE_MAX / options->run_rows &&
        options->asts_per_kernel <= SIZE_MAX / options->num_targets &&
        options->num_inputs <= SIZE_MAX / options->check_rows &&
        options->num_targets <= SIZE_MAX / options->check_rows &&
        options->asts_per_kernel <= SIZE_MAX / options->check_rows;
}

static float
secant_bench_input_value(size_t column, size_t row, uint32_t seed) {
    const uint32_t bits = secant_bench_ast_hash32(
        (uint32_t)column * 0x9e3779b9u ^
        (uint32_t)row * 0x85ebca6bu ^
        seed * 0xc2b2ae35u ^
        0x51ed270bu);

    return ((float)(bits & 0xffffu) / 32767.5f - 1.0f) * 1.5f;
}

static void
secant_bench_fill_input(
    float* input,
    size_t num_inputs,
    size_t rows,
    uint32_t seed
) {
    size_t input_idx;

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        size_t row;

        for (row = 0u; row < rows; ++row) {
            input[input_idx * rows + row] =
                secant_bench_input_value(input_idx, row, seed);
        }
    }
}

static void
secant_bench_fill_targets(
    float* targets,
    size_t num_targets,
    size_t rows,
    uint32_t seed
) {
    size_t target_idx;

    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        size_t row;

        for (row = 0u; row < rows; ++row) {
            const float x0 = secant_bench_input_value(0u, row, seed);
            const float x1 = secant_bench_input_value(1u, row, seed);
            const float x2 = secant_bench_input_value(2u, row, seed);
            const float x3 = secant_bench_input_value(3u, row, seed);
            const float x4 = secant_bench_input_value(4u, row, seed);
            const float x5 = secant_bench_input_value(5u, row, seed);
            const float x6 = secant_bench_input_value(6u, row, seed);
            const float x7 = secant_bench_input_value(7u, row, seed);

            targets[target_idx * rows + row] =
                SECANT_PORTABLE_ALU_SCALAR_TARGET(
                    x0, x1, x2, x3, x4, x5, x6, x7);
        }
    }
}

static int
secant_bench_float_close(
    SecantBenchAstMode ast_mode,
    float expected,
    float actual
) {
    const float difference = fabsf(expected - actual);
    const float scale = fmaxf(fabsf(expected), fabsf(actual));
    const float absolute_tolerance =
        ast_mode == SECANT_BENCH_AST_MODE_MUFU
            ? 2.0e-4f
            : 5.0e-5f;
    const float relative_tolerance =
        ast_mode == SECANT_BENCH_AST_MODE_MUFU
            ? 1.0e-3f
            : 2.0e-4f;

    /* Deep MUFU trees compose several approximate GPU operations. */
    return difference <=
        absolute_tolerance + relative_tolerance * scale;
}

static int
secant_bench_compare(
    SecantBenchAstMode ast_mode,
    const float* expected,
    const float* actual,
    size_t count,
    float* max_abs_error_ret,
    size_t* worst_idx_ret
) {
    float max_abs_error = 0.0f;
    size_t worst_idx = 0u;
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        const float error = fabsf(expected[idx] - actual[idx]);

        if (!secant_bench_float_close(
                ast_mode,
                expected[idx],
                actual[idx])) {
            if (max_abs_error_ret != NULL) {
                *max_abs_error_ret = error;
            }
            if (worst_idx_ret != NULL) {
                *worst_idx_ret = idx;
            }
            return 0;
        }
        if (error > max_abs_error) {
            max_abs_error = error;
            worst_idx = idx;
        }
    }
    if (max_abs_error_ret != NULL) {
        *max_abs_error_ret = max_abs_error;
    }
    if (worst_idx_ret != NULL) {
        *worst_idx_ret = worst_idx;
    }
    return 1;
}

#endif /* SECANT_BENCH_COMMON_H_INCLUDED */
