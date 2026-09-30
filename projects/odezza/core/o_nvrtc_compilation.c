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
#include "o_odezza_internal.h"

#include <limits.h>

#include <nvrtc.h>

#include <stdlib.h>
#include <time.h>

struct OdezzaNvrtcCompilation {
    nvrtcProgram program;
    nvrtcResult native_result;
    OdezzaResult compilation_result;
    double compile_seconds;
};

static OdezzaResult o_nvrtc_result(nvrtcResult result) {
    switch (result) {
    case NVRTC_SUCCESS:
        return ODEZZA_SUCCESS;
    case NVRTC_ERROR_OUT_OF_MEMORY:
        return ODEZZA_ERROR_ALLOCATION;
    case NVRTC_ERROR_INVALID_INPUT:
    case NVRTC_ERROR_INVALID_PROGRAM:
    case NVRTC_ERROR_INVALID_OPTION:
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    case NVRTC_ERROR_COMPILATION:
        return ODEZZA_ERROR_COMPILATION;
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
    case NVRTC_ERROR_INTERNAL_ERROR:
        return ODEZZA_ERROR_COMPILER;
    default:
        return ODEZZA_ERROR_COMPILER;
    }
}

static OdezzaResult o_nvrtc_options_validate(const char *const *options, size_t option_count) {
    size_t index;

    if (option_count > (size_t)INT_MAX) return ODEZZA_ERROR_OVERFLOW;
    if (option_count != 0u && options == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (index = 0u; index < option_count; ++index) {
        if (options[index] == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_create(
    const char *source,
    const char *source_name,
    const char *const *options,
    size_t option_count,
    OdezzaNvrtcCompilation **compilation_ret
) {
    OdezzaNvrtcCompilation *compilation;
    nvrtcResult native_result;
    OdezzaResult result;
    struct timespec before = {0, 0}, after = {0, 0};

    if (source == NULL || compilation_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *compilation_ret = NULL;
    result = o_nvrtc_options_validate(options, option_count);
    if (result != ODEZZA_SUCCESS) return result;
    compilation = (OdezzaNvrtcCompilation *)malloc(sizeof(*compilation));
    if (compilation == NULL) return ODEZZA_ERROR_ALLOCATION;
    compilation->program = NULL;
    compilation->native_result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
    compilation->compilation_result = ODEZZA_ERROR_COMPILER;
    native_result = nvrtcCreateProgram(&compilation->program, source, source_name, 0, NULL, NULL);
    if (native_result != NVRTC_SUCCESS) {
        free(compilation);
        return o_nvrtc_result(native_result);
    }
    compilation->compile_seconds = 0.0;
    (void)clock_gettime(CLOCK_MONOTONIC, &before);
    native_result = nvrtcCompileProgram(compilation->program, (int)option_count, options);
    (void)clock_gettime(CLOCK_MONOTONIC, &after);
    compilation->compile_seconds = (double)(after.tv_sec - before.tv_sec)
        + (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
    compilation->native_result = native_result;
    compilation->compilation_result = o_nvrtc_result(native_result);
    *compilation_ret = compilation;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_nvrtc_compilation_seconds(const OdezzaNvrtcCompilation *compilation,
                                       double *seconds_ret) {
    if (!compilation || !seconds_ret) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *seconds_ret = compilation->compile_seconds;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_destroy(OdezzaNvrtcCompilation *compilation) {
    nvrtcResult native_result;

    if (compilation == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    native_result = nvrtcDestroyProgram(&compilation->program);
    free(compilation);
    return o_nvrtc_result(native_result);
}

OdezzaResult odezza_nvrtc_compilation_result(
    const OdezzaNvrtcCompilation *compilation,
    OdezzaResult *compilation_result_ret
) {
    if (compilation == NULL || compilation_result_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *compilation_result_ret = compilation->compilation_result;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_native_result(const OdezzaNvrtcCompilation *compilation, int *native_result_ret) {
    if (compilation == NULL || native_result_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *native_result_ret = (int)compilation->native_result;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_native_result_string(
    const OdezzaNvrtcCompilation *compilation,
    const char **native_result_string_ret
) {
    if (compilation == NULL || native_result_string_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *native_result_string_ret = nvrtcGetErrorString(compilation->native_result);
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_cubin_size(const OdezzaNvrtcCompilation *compilation, size_t *cubin_size_ret) {
    nvrtcResult native_result;

    if (compilation == NULL || cubin_size_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *cubin_size_ret = 0u;
    if (compilation->compilation_result != ODEZZA_SUCCESS) return compilation->compilation_result;
    native_result = nvrtcGetCUBINSize(compilation->program, cubin_size_ret);
    if (native_result != NVRTC_SUCCESS) return o_nvrtc_result(native_result);
    if (*cubin_size_ret == 0u) return ODEZZA_ERROR_UNSUPPORTED;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_write_cubin(
    const OdezzaNvrtcCompilation *compilation,
    void *buffer,
    size_t buffer_size
) {
    size_t required;
    nvrtcResult native_result;
    OdezzaResult result;

    if (compilation == NULL || buffer == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    result = odezza_nvrtc_compilation_cubin_size(compilation, &required);
    if (result != ODEZZA_SUCCESS) return result;
    if (buffer_size < required) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    native_result = nvrtcGetCUBIN(compilation->program, (char *)buffer);
    return o_nvrtc_result(native_result);
}

OdezzaResult odezza_nvrtc_compilation_log_size(const OdezzaNvrtcCompilation *compilation, size_t *log_size_ret) {
    nvrtcResult native_result;

    if (compilation == NULL || log_size_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *log_size_ret = 0u;
    native_result = nvrtcGetProgramLogSize(compilation->program, log_size_ret);
    if (native_result != NVRTC_SUCCESS) return o_nvrtc_result(native_result);
    if (*log_size_ret == 0u) return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_nvrtc_compilation_write_log(
    const OdezzaNvrtcCompilation *compilation,
    char *buffer,
    size_t buffer_size
) {
    size_t required;
    nvrtcResult native_result;
    OdezzaResult result;

    if (compilation == NULL || buffer == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    result = odezza_nvrtc_compilation_log_size(compilation, &required);
    if (result != ODEZZA_SUCCESS) return result;
    if (buffer_size < required) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    native_result = nvrtcGetProgramLog(compilation->program, buffer);
    return o_nvrtc_result(native_result);
}
