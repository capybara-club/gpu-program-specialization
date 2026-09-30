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
#include "o_odezza_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define O_TRY(expression) \
    do { \
        OdezzaResult o_try_result = (expression); \
        if (o_try_result != ODEZZA_SUCCESS) return o_try_result; \
    } while (0)

static OdezzaResult o_expect(int condition, const char *message) {
    if (condition) return ODEZZA_SUCCESS;
    fprintf(stderr, "%s\n", message);
    return ODEZZA_ERROR_FORMAT;
}

static OdezzaResult o_generate_source(uint32_t constant_capacity, char **source_ret, size_t *source_size_ret) {
    char *source;
    size_t source_size;

    if (source_ret == NULL || source_size_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *source_ret = NULL;
    *source_size_ret = 0u;
    O_TRY(odezza_generate_scoring_cuda(
        4u,
        constant_capacity,
        8u,
        384u,
        384u,
        NULL,
        0u,
        &source_size
    ));
    source = (char *)malloc(source_size + 1u);
    if (source == NULL) return ODEZZA_ERROR_ALLOCATION;
    if (
        odezza_generate_scoring_cuda(
            4u,
            constant_capacity,
            8u,
            384u,
            384u,
            source,
            source_size + 1u,
            &source_size
        ) != ODEZZA_SUCCESS
    ) {
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }
    *source_ret = source;
    *source_size_ret = source_size;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_test_successful_compilation(const char *source, const char *const *options, size_t option_count) {
    OdezzaNvrtcCompilation *compilation = NULL;
    const OdezzaScoringCubinInspection *inspection = NULL;
    const char *native_string = NULL;
    OdezzaResult compilation_result;
    OdezzaResult result;
    unsigned char *cubin = NULL;
    unsigned char *short_cubin = NULL;
    void *inspection_arena = NULL;
    char *log = NULL;
    size_t cubin_size;
    size_t inspection_arena_size;
    size_t log_size;
    int native_result;

    result = odezza_nvrtc_compilation_create(source, "odezza-scoring.cu", options, option_count, &compilation);
    O_TRY(result);
    O_TRY(o_expect(compilation != NULL, "NVRTC create did not return a handle"));
    O_TRY(odezza_nvrtc_compilation_result(compilation, &compilation_result));
    O_TRY(o_expect(compilation_result == ODEZZA_SUCCESS, "generated scoring CUDA did not compile"));
    O_TRY(odezza_nvrtc_compilation_native_result(compilation, &native_result));
    O_TRY(o_expect(native_result == 0, "successful compilation has a failing native result"));
    O_TRY(odezza_nvrtc_compilation_native_result_string(compilation, &native_string));
    O_TRY(o_expect(native_string != NULL && strcmp(native_string, "NVRTC_SUCCESS") == 0, "unexpected native success string"));

    O_TRY(odezza_nvrtc_compilation_log_size(compilation, &log_size));
    O_TRY(o_expect(log_size >= 1u, "NVRTC log allocation size omitted its terminator"));
    log = (char *)malloc(log_size);
    if (log == NULL) return ODEZZA_ERROR_ALLOCATION;
    O_TRY(o_expect(odezza_nvrtc_compilation_write_log(compilation, log, log_size - 1u) == ODEZZA_ERROR_INSUFFICIENT_BUFFER,
                   "short NVRTC log buffer was accepted"));
    O_TRY(odezza_nvrtc_compilation_write_log(compilation, log, log_size));
    O_TRY(o_expect(log[log_size - 1u] == '\0', "NVRTC log is not NUL terminated"));

    O_TRY(odezza_nvrtc_compilation_cubin_size(compilation, &cubin_size));
    O_TRY(o_expect(cubin_size != 0u, "physical compilation returned an empty CUBIN"));
    short_cubin = (unsigned char *)malloc(cubin_size);
    if (short_cubin == NULL) return ODEZZA_ERROR_ALLOCATION;
    result = odezza_nvrtc_compilation_write_cubin(compilation, short_cubin, cubin_size - 1u);
    O_TRY(o_expect(result == ODEZZA_ERROR_INSUFFICIENT_BUFFER, "short CUBIN buffer was accepted"));
    cubin = (unsigned char *)malloc(cubin_size);
    if (cubin == NULL) return ODEZZA_ERROR_ALLOCATION;
    O_TRY(odezza_nvrtc_compilation_write_cubin(compilation, cubin, cubin_size));
    O_TRY(odezza_inspect_scoring_cubin(cubin, cubin_size, NULL, 0u, &inspection_arena_size, NULL));
    inspection_arena = malloc(inspection_arena_size);
    if (inspection_arena == NULL) return ODEZZA_ERROR_ALLOCATION;
    O_TRY(odezza_inspect_scoring_cubin(cubin, cubin_size, inspection_arena, inspection_arena_size, &inspection_arena_size, &inspection));
    O_TRY(o_expect(inspection != NULL && inspection->architecture == 89u, "compiled CUBIN inspection failed"));
    O_TRY(odezza_nvrtc_compilation_destroy(compilation));
    free(inspection_arena);
    free(cubin);
    free(short_cubin);
    free(log);
    printf(
        "NVRTC scoring compilation: %lu CUBIN bytes, %lu log bytes, %lu inspection-arena bytes\n",
        (unsigned long)cubin_size,
        (unsigned long)log_size,
        (unsigned long)inspection_arena_size
    );
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_test_failed_compilation(const char *const *options, size_t option_count) {
    static const char broken_source[] = "extern \"C\" __global__ void broken( {\n";
    OdezzaNvrtcCompilation *compilation = NULL;
    const char *native_string = NULL;
    OdezzaResult compilation_result;
    char *log;
    size_t cubin_size = 123u;
    size_t log_size;
    int native_result;

    O_TRY(odezza_nvrtc_compilation_create(broken_source, "broken.cu", options, option_count, &compilation));
    O_TRY(o_expect(compilation != NULL, "failed CUDA compilation did not retain its handle"));
    O_TRY(odezza_nvrtc_compilation_result(compilation, &compilation_result));
    O_TRY(o_expect(compilation_result == ODEZZA_ERROR_COMPILATION, "broken CUDA has the wrong mapped result"));
    O_TRY(odezza_nvrtc_compilation_native_result(compilation, &native_result));
    O_TRY(o_expect(native_result != 0, "broken CUDA has a successful native result"));
    O_TRY(odezza_nvrtc_compilation_native_result_string(compilation, &native_string));
    O_TRY(o_expect(native_string != NULL && strstr(native_string, "COMPILATION") != NULL, "wrong native compilation-error string"));
    O_TRY(odezza_nvrtc_compilation_log_size(compilation, &log_size));
    O_TRY(o_expect(log_size > 1u, "failed compilation did not retain diagnostics"));
    log = (char *)malloc(log_size);
    if (log == NULL) return ODEZZA_ERROR_ALLOCATION;
    O_TRY(odezza_nvrtc_compilation_write_log(compilation, log, log_size));
    O_TRY(o_expect(strstr(log, "error") != NULL, "failed compilation log contains no error"));
    O_TRY(o_expect(odezza_nvrtc_compilation_cubin_size(compilation, &cubin_size) == ODEZZA_ERROR_COMPILATION, "failed compilation exposed a CUBIN"));
    O_TRY(o_expect(cubin_size == 0u, "failed compilation reported nonzero CUBIN bytes"));
    O_TRY(odezza_nvrtc_compilation_destroy(compilation));
    free(log);
    printf("NVRTC failed-compilation diagnostics: %lu bytes retained\n", (unsigned long)log_size);
    return ODEZZA_SUCCESS;
}

int main(void) {
    static const char *const options[] = {"--std=c++17", "--gpu-architecture=sm_89", "--use_fast_math", "--ptxas-options=-O3"};
    OdezzaNvrtcCompilation *compilation = NULL;
    char *source = NULL;
    size_t source_size;
    OdezzaResult result;

    if (odezza_nvrtc_compilation_create(NULL, NULL, NULL, 0u, &compilation) != ODEZZA_ERROR_INVALID_ARGUMENT) return 1;
    if (odezza_nvrtc_compilation_destroy(NULL) != ODEZZA_ERROR_INVALID_ARGUMENT) return 1;
    result = o_generate_source(4u, &source, &source_size);
    if (result != ODEZZA_SUCCESS || source_size == 0u) {
        free(source);
        return 1;
    }
    result = o_test_successful_compilation(source, options, sizeof(options) / sizeof(options[0]));
    free(source);
    if (result != ODEZZA_SUCCESS) return 1;
    source = NULL;
    result = o_generate_source(0u, &source, &source_size);
    if (result != ODEZZA_SUCCESS || source_size == 0u) {
        free(source);
        return 1;
    }
    result = o_test_successful_compilation(source, options, sizeof(options) / sizeof(options[0]));
    free(source);
    if (result != ODEZZA_SUCCESS) return 1;
    result = o_test_failed_compilation(options, sizeof(options) / sizeof(options[0]));
    return result == ODEZZA_SUCCESS ? 0 : 1;
}
