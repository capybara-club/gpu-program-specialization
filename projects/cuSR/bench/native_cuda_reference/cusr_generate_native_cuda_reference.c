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
#define CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION
#include <cusr_native_cuda_reference_gen.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _CUSR_GENERATE_ERROR_RET(ans) \
    do { \
        int cusr_generate_result = (ans); \
        return cusr_generate_result; \
    } while (0)

#define _CUSR_GENERATE_CHECK_RET(ans) \
    do { \
        int cusr_generate_check_ret = (ans); \
        if (cusr_generate_check_ret != 0) { \
            _CUSR_GENERATE_ERROR_RET(cusr_generate_check_ret); \
        } \
    } while (0)

static int
cusr_generate_parse_size(const char* text, size_t* value_ret)
{
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

static int
cusr_generate_write_file(const char* path, const char* data, size_t size)
{
    FILE* file = fopen(path, "wb");

    if (file == NULL) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    if (fwrite(data, 1u, size, file) != size) {
        fprintf(stderr, "failed to write %zu bytes to %s\n", size, path);
        fclose(file);
        return 0;
    }

    if (fclose(file) != 0) {
        fprintf(stderr, "failed to close %s\n", path);
        return 0;
    }

    return 1;
}

static int
cusr_generate_parse_expr_mode(const char* text, unsigned* expr_mode_ret)
{
    if (strcmp(text, "offset-minmax") == 0 || strcmp(text, "offset_minmax") == 0) {
        *expr_mode_ret = CUSR_NATIVE_CUDA_REFERENCE_EXPR_OFFSET_MINMAX;
        return 1;
    }

    if (strcmp(text, "mixed-reduce") == 0 || strcmp(text, "mixed_reduce") == 0 || strcmp(text, "alu-reduce") == 0 || strcmp(text, "alu_reduce") == 0) {
        *expr_mode_ret = CUSR_NATIVE_CUDA_REFERENCE_EXPR_MIXED_REDUCE;
        return 1;
    }

    if (strcmp(text, "depth3-alu") == 0 || strcmp(text, "depth3_alu") == 0) {
        *expr_mode_ret = CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_ALU;
        return 1;
    }

    if (strcmp(text, "depth3-mufu") == 0 || strcmp(text, "depth3_mufu") == 0) {
        *expr_mode_ret = CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_MUFU;
        return 1;
    }

    return 0;
}

static void
cusr_generate_usage(const char* argv0)
{
    fprintf(
        stderr,
        "usage: %s [-o output.cu] [--expr MODE] [--tile-rows 64|128|256] [--cta-threads 64|128|256] <kernels-per-file> <asts-per-kernel>\n"
        "\n"
        "expression modes: offset-minmax, mixed-reduce, depth3-alu, depth3-mufu\n"
        "\n"
        "example:\n"
        "  %s 2 4 > native_cuda_reference.cu\n"
        "  %s --expr depth3-alu --tile-rows 64 --cta-threads 128 -o native_cuda_reference.cu 2 4\n",
        argv0,
        argv0,
        argv0
    );
}

static int
cusr_generate_main_impl(int argc, char** argv, char** cuda_ret)
{
    size_t kernels_per_file = 0u;
    size_t asts_per_kernel = 0u;
    size_t tile_rows = 64u;
    size_t threads_per_cta = 128u;
    size_t cuda_size = 0u;
    int arg_idx = 1;
    const char* output_path = NULL;
    unsigned expr_mode = CUSR_NATIVE_CUDA_REFERENCE_EXPR_OFFSET_MINMAX;
    char* cuda_src;
    CusrNativeCudaReferenceGenResult result;

    *cuda_ret = NULL;

    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "-o") == 0) {
            if (arg_idx + 1 >= argc) {
                cusr_generate_usage(argv[0]);
                _CUSR_GENERATE_ERROR_RET(2);
            }
            output_path = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--expr") == 0) {
            if (arg_idx + 1 >= argc || !cusr_generate_parse_expr_mode(argv[arg_idx + 1], &expr_mode)) {
                cusr_generate_usage(argv[0]);
                _CUSR_GENERATE_ERROR_RET(2);
            }
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--tile-rows") == 0) {
            if (arg_idx + 1 >= argc || !cusr_generate_parse_size(argv[arg_idx + 1], &tile_rows)) {
                cusr_generate_usage(argv[0]);
                _CUSR_GENERATE_ERROR_RET(2);
            }
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--cta-threads") == 0) {
            if (arg_idx + 1 >= argc || !cusr_generate_parse_size(argv[arg_idx + 1], &threads_per_cta)) {
                cusr_generate_usage(argv[0]);
                _CUSR_GENERATE_ERROR_RET(2);
            }
            arg_idx += 2;
        } else {
            break;
        }
    }

    if (argc - arg_idx != 2) {
        cusr_generate_usage(argv[0]);
        _CUSR_GENERATE_ERROR_RET(2);
    }

    if (!cusr_generate_parse_size(argv[arg_idx], &kernels_per_file) || kernels_per_file == 0u) {
        fprintf(stderr, "invalid kernels-per-file: %s\n", argv[arg_idx]);
        _CUSR_GENERATE_ERROR_RET(2);
    }
    arg_idx += 1;

    if (!cusr_generate_parse_size(argv[arg_idx], &asts_per_kernel) || asts_per_kernel == 0u) {
        fprintf(stderr, "invalid asts-per-kernel: %s\n", argv[arg_idx]);
        _CUSR_GENERATE_ERROR_RET(2);
    }

    result = cusr_native_cuda_reference_gen_cuda_size(
        kernels_per_file,
        asts_per_kernel,
        tile_rows,
        threads_per_cta,
        expr_mode,
        &cuda_size);
    if (result != CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS) {
        fprintf(stderr, "measure failed: %s\n", cusr_native_cuda_reference_gen_result_to_string(result));
        _CUSR_GENERATE_ERROR_RET(1);
    }

    cuda_src = (char*)malloc(cuda_size);
    if (cuda_src == NULL) {
        fprintf(stderr, "failed to allocate %zu bytes\n", cuda_size);
        _CUSR_GENERATE_ERROR_RET(1);
    }

    *cuda_ret = cuda_src;

    result = cusr_native_cuda_reference_gen_cuda(
        cuda_src,
        cuda_size,
        kernels_per_file,
        asts_per_kernel,
        tile_rows,
        threads_per_cta,
        expr_mode,
        &cuda_size);
    if (result != CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS) {
        fprintf(stderr, "generate failed: %s\n", cusr_native_cuda_reference_gen_result_to_string(result));
        _CUSR_GENERATE_ERROR_RET(1);
    }

    if (output_path != NULL) {
        _CUSR_GENERATE_CHECK_RET(cusr_generate_write_file(output_path, cuda_src, cuda_size - 1u) ? 0 : 1);
    } else if (fwrite(cuda_src, 1u, cuda_size - 1u, stdout) != cuda_size - 1u || fflush(stdout) != 0) {
        fprintf(stderr, "failed to write generated CUDA to stdout\n");
        _CUSR_GENERATE_ERROR_RET(1);
    }

    _CUSR_GENERATE_ERROR_RET(0);
}

int
main(int argc, char** argv)
{
    char* cuda_src = NULL;
    const int result = cusr_generate_main_impl(argc, argv, &cuda_src);
    free(cuda_src);
    return result;
}

#undef _CUSR_GENERATE_CHECK_RET
#undef _CUSR_GENERATE_ERROR_RET
