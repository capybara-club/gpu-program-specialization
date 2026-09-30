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
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "implicit_sindy.h"
#include "stack_ptx_harness_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
test_write_file(
    const char* path,
    const void* data,
    size_t bytes
) {
    FILE* file;
    int ok = 0;

    if (path == NULL || data == NULL || bytes == 0u) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    if (fwrite(data, 1u, bytes, file) == bytes) ok = 1;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int
test_dump_cubins_if_requested(
    void* const* cubins,
    const size_t* cubin_sizes,
    size_t cubin_count,
    size_t kernels_per_module,
    size_t num_asts
) {
    const char* prefix = getenv("IMPLICIT_SINDY_TEST_DUMP_PREFIX");
    size_t i;

    if (prefix == NULL || prefix[0] == '\0') return 1;
    for (i = 0u; i < cubin_count; ++i) {
        char path[512];
        snprintf(
            path,
            sizeof(path),
            "%s_k%zu_n%zu_c%zu.cubin",
            prefix,
            kernels_per_module,
            num_asts,
            i
        );
        if (!test_write_file(path, cubins[i], cubin_sizes[i])) {
            fprintf(stderr, "failed to write cubin dump: %s\n", path);
            return 0;
        }
        fprintf(stderr, "wrote cubin dump: %s (%zu bytes)\n", path, cubin_sizes[i]);
    }
    return 1;
}

static int
test_check(
    int condition,
    const char* label
) {
    if (condition) return 1;
    fprintf(stderr, "check failed: %s\n", label);
    return 0;
}

static void
test_make_ast(
    size_t idx,
    BinaryAST* ast
) {
    size_t i;
    memset(ast, 0, sizeof(*ast));
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
        ast->unary[i] = (BinaryAstUnaryOp)((idx + i) % BINARY_AST_UNARY_NUM_ENUMS);
    }
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        BinaryAstBinaryOp op;
        switch ((idx + i) % BINARY_AST_BINARY_NUM_ENUMS) {
            case 1u:
                op = BINARY_AST_BINARY_SUB_FTZ_F32;
                break;
            case 2u:
                op = BINARY_AST_BINARY_MUL_FTZ_F32;
                break;
            case 3u:
                op = BINARY_AST_BINARY_KEEP_LEFT;
                break;
            case 4u:
                op = BINARY_AST_BINARY_KEEP_RIGHT;
                break;
            case 5u:
                op = BINARY_AST_BINARY_DIV_APPROX_FTZ_F32;
                break;
            case 6u:
                op = BINARY_AST_BINARY_MIN_FTZ_F32;
                break;
            case 7u:
                op = BINARY_AST_BINARY_MAX_FTZ_F32;
                break;
            default:
                op = BINARY_AST_BINARY_ADD_FTZ_F32;
                break;
        }
        ast->binary[i] = op;
    }
}

static int
test_compile_load_unload_case(
    size_t kernels_per_module,
    size_t num_asts,
    size_t expected_cubin_count
)
{
    enum {
        MAX_ASTS = 129,
        WORKER_COUNT = 2,
        SCRATCH_BYTES_PER_WORKER = 64 * 1024 * 1024
    };
    BinaryAST asts[MAX_ASTS];
    ImplicitSindyAstCompiler* compiler = NULL;
    ImplicitSindyGramModule* gram_module = NULL;
    void* modules[2];
    void* cubins[2];
    size_t cubin_sizes[2];
    void* workspace = NULL;
    size_t workspace_bytes = 0u;
    size_t cubin_count;
    CUdevice device;
    int owns_primary = 0;
    unsigned int sm_major = 0u;
    unsigned int sm_minor = 0u;
    ImplicitSindyResult result;
    size_t i;
    int ok = 0;

    memset(modules, 0, sizeof(modules));
    memset(cubins, 0, sizeof(cubins));
    memset(cubin_sizes, 0, sizeof(cubin_sizes));

    if (!test_check(num_asts <= MAX_ASTS, "num_asts limit")) return 0;

    for (i = 0u; i < num_asts; ++i) {
        test_make_ast(i, &asts[i]);
    }

    if (!stack_ptx_test_init_context(&device, &owns_primary)) return 0;
    if (!stack_ptx_test_device_sm(device, &sm_major, &sm_minor)) goto cleanup;

    result = implicit_sindy_ast_compiler_create(
        kernels_per_module,
        sm_major,
        sm_minor,
        NULL,
        0u,
        &compiler
    );
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "compiler create")) goto cleanup;

    result = implicit_sindy_ast_compile_workspace_size(
        compiler,
        num_asts,
        WORKER_COUNT,
        SCRATCH_BYTES_PER_WORKER,
        &workspace_bytes
    );
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "workspace size")) goto cleanup;
    workspace = malloc(workspace_bytes);
    if (!test_check(workspace != NULL, "workspace malloc")) goto cleanup;

    cubin_count = implicit_sindy_ast_cubin_count(
        num_asts,
        kernels_per_module
    );
    if (!test_check(cubin_count == expected_cubin_count, "case cubin count")) goto cleanup;

    result = implicit_sindy_ast_compile_cubins(
        compiler,
        asts,
        num_asts,
        sizeof(asts[0]),
        WORKER_COUNT,
        workspace,
        workspace_bytes,
        cubins,
        cubin_sizes
    );
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "compile cubins")) goto cleanup;
    for (i = 0u; i < cubin_count; ++i) {
        if (!test_check(cubins[i] != NULL && cubin_sizes[i] != 0u, "cubin")) goto cleanup;
    }
    if (!test_dump_cubins_if_requested(cubins, cubin_sizes, cubin_count, kernels_per_module, num_asts)) {
        goto cleanup;
    }

    result = implicit_sindy_load_cubin_modules(
        cubins,
        cubin_sizes,
        cubin_count,
        modules
    );
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "load cubin modules")) goto cleanup;
    for (i = 0u; i < cubin_count; ++i) {
        if (!test_check(modules[i] != NULL, "module handle")) goto cleanup;
    }

    result = implicit_sindy_gram_module_create(
        modules[0],
        kernels_per_module,
        &gram_module
    );
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "gram module create")) goto cleanup;
    result = implicit_sindy_gram_module_destroy(gram_module);
    gram_module = NULL;
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "gram module destroy")) goto cleanup;

    result = implicit_sindy_unload_modules(
        modules,
        cubin_count
    );
    if (!test_check(result == IMPLICIT_SINDY_SUCCESS, "unload modules")) goto cleanup;
    ok = 1;

cleanup:
    if (gram_module != NULL) (void)implicit_sindy_gram_module_destroy(gram_module);
    (void)implicit_sindy_unload_modules(modules, 2u);
    implicit_sindy_free_cubins(cubins, cubin_sizes, 2u);
    free(workspace);
    if (compiler != NULL) (void)implicit_sindy_ast_compiler_destroy(compiler);
    stack_ptx_test_release_context(device, owns_primary);
    return ok;
}

static int
test_compile_load_unload(void)
{
    if (!test_check(
            implicit_sindy_ast_cubin_count(33u, 1u) == 2u,
            "1-kernel cubin count")) return 0;
    if (!test_check(
            implicit_sindy_ast_cubin_count(129u, 4u) == 2u,
            "4-kernel cubin count")) return 0;
    if (!test_check(
            implicit_sindy_ast_cubin_count(97u, 3u) == 2u,
            "3-kernel cubin count")) return 0;
    if (!test_compile_load_unload_case(1u, 33u, 2u)) return 0;
    if (!test_compile_load_unload_case(4u, 33u, 1u)) return 0;
    if (!test_compile_load_unload_case(3u, 33u, 1u)) return 0;
    return 1;
}

int
main(void)
{
    if (!test_compile_load_unload()) return 1;
    printf("PASS implicit_sindy_ast_compile_api_test\n");
    return 0;
}
