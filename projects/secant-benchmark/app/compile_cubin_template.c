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
#include "bench_cubin_template.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
compile_cubin_template_usage(const char* argv0) {
    fprintf(
        stderr,
        "usage: %s <kernel-shape> --sm NN [options] "
        "<kernels-per-module> <asts-per-kernel> -o FILE\n"
        "\n"
        "kernel shapes:\n"
        "  static-column-materialize\n"
        "  static-column-sse\n"
        "\n"
        "options:\n"
        "  --inputs N                       input columns (default: 8)\n"
        "  --targets N                      SSE targets (default: 1)\n"
        "  --tile-rows N                    SSE rows/CTA (default: 4096)\n"
        "  --threads N                      SSE threads/CTA (default: 128)\n"
        "  --patch-instructions-per-ast N   patch capacity (default: 64)\n",
        argv0);
}

static int
compile_cubin_template_size(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || text == end || *end != '\0' ||
        value == 0u || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
compile_cubin_template_u32(const char* text, uint32_t* value_ret) {
    size_t value;

    if (!compile_cubin_template_size(text, &value) ||
        value > UINT32_MAX) {
        return 0;
    }
    *value_ret = (uint32_t)value;
    return 1;
}

static int
compile_cubin_template_write(
    const char* output_path,
    const unsigned char* cubin,
    size_t cubin_size
) {
    FILE* output;
    int result = 0;

    output = fopen(output_path, "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", output_path);
        return 1;
    }
    if (fwrite(cubin, 1u, cubin_size, output) != cubin_size) {
        fprintf(stderr, "failed to write CUBIN: %s\n", output_path);
        result = 1;
    }
    if (fclose(output) != 0) {
        fprintf(stderr, "failed to close output file: %s\n", output_path);
        result = 1;
    }
    return result;
}

static int
compile_cubin_template_run(
    SecantBenchShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_instructions_per_ast,
    uint32_t target_sm,
    const char* output_path
) {
    char error[16384];
    unsigned char* cubin = NULL;
    size_t patch_capacity_instructions;
    size_t cubin_size = 0u;
    int exit_code = 1;

    if (asts_per_kernel >
        SIZE_MAX / patch_instructions_per_ast) {
        fprintf(stderr, "patch capacity overflows size_t\n");
        return 1;
    }
    patch_capacity_instructions =
        asts_per_kernel * patch_instructions_per_ast;
    if (!secant_bench_cubin_template_compile(
            shape,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            tile_rows,
            threads_per_block,
            SECANT_SSE_REDUCTION_MODE_ATOMIC,
            patch_capacity_instructions,
            target_sm,
            error,
            sizeof(error),
            &cubin,
            &cubin_size)) {
        fprintf(stderr, "template compilation failed: %s\n", error);
        return 1;
    }
    exit_code = compile_cubin_template_write(
        output_path,
        cubin,
        cubin_size);
    free(cubin);
    return exit_code;
}

int
main(int argc, char** argv) {
    SecantBenchShape shape;
    const char* positional[2];
    const char* output_path = NULL;
    size_t num_positional = 0u;
    size_t num_kernels = 0u;
    size_t asts_per_kernel = 0u;
    size_t num_inputs = 8u;
    size_t num_targets = 1u;
    size_t tile_rows = 4096u;
    size_t threads_per_block = 128u;
    size_t patch_instructions_per_ast = 64u;
    uint32_t target_sm = 0u;
    int arg_idx;

    if (argc < 2) {
        compile_cubin_template_usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "static-column-materialize") == 0) {
        shape = SECANT_BENCH_SHAPE_MATERIALIZE;
    } else if (strcmp(argv[1], "static-column-sse") == 0) {
        shape = SECANT_BENCH_SHAPE_SSE;
    } else {
        compile_cubin_template_usage(argv[0]);
        return 1;
    }

    for (arg_idx = 2; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--sm") == 0) {
            if (++arg_idx >= argc ||
                !compile_cubin_template_u32(
                    argv[arg_idx],
                    &target_sm)) {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--inputs") == 0) {
            if (++arg_idx >= argc ||
                !compile_cubin_template_size(
                    argv[arg_idx],
                    &num_inputs)) {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--targets") == 0) {
            if (++arg_idx >= argc ||
                !compile_cubin_template_size(
                    argv[arg_idx],
                    &num_targets)) {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--tile-rows") == 0) {
            if (++arg_idx >= argc ||
                !compile_cubin_template_size(
                    argv[arg_idx],
                    &tile_rows)) {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--threads") == 0) {
            if (++arg_idx >= argc ||
                !compile_cubin_template_size(
                    argv[arg_idx],
                    &threads_per_block)) {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(
                argv[arg_idx],
                "--patch-instructions-per-ast") == 0) {
            if (++arg_idx >= argc ||
                !compile_cubin_template_size(
                    argv[arg_idx],
                    &patch_instructions_per_ast)) {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "-o") == 0 ||
                   strcmp(argv[arg_idx], "--output") == 0) {
            if (++arg_idx >= argc || argv[arg_idx][0] == '\0') {
                compile_cubin_template_usage(argv[0]);
                return 1;
            }
            output_path = argv[arg_idx];
        } else if (num_positional < 2u) {
            positional[num_positional++] = argv[arg_idx];
        } else {
            compile_cubin_template_usage(argv[0]);
            return 1;
        }
    }
    if (target_sm == 0u || output_path == NULL ||
        num_positional != 2u ||
        !compile_cubin_template_size(
            positional[0],
            &num_kernels) ||
        !compile_cubin_template_size(
            positional[1],
            &asts_per_kernel)) {
        compile_cubin_template_usage(argv[0]);
        return 1;
    }
    return compile_cubin_template_run(
        shape,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        tile_rows,
        threads_per_block,
        patch_instructions_per_ast,
        target_sm,
        output_path);
}
