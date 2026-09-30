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
#include "cusr_tile_static_eval_nvrtc.h"

#include <nvrtc.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CUSR_TILE_STATIC_EVAL_TEMPLATE_PATH
#error "CUSR_TILE_STATIC_EVAL_TEMPLATE_PATH must name the CUDA template header"
#endif

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX cusr_eval_nvrtc_
#include <incbin.h>

INCBIN(tile_static_eval_template, CUSR_TILE_STATIC_EVAL_TEMPLATE_PATH);

#define CUSR_TILE_STATIC_EVAL_NVRTC_FIRST_MARKER 0x7fc0ffeeu
#define CUSR_TILE_STATIC_EVAL_NVRTC_MARKER_STRIDE 128u
#define CUSR_TILE_STATIC_EVAL_NVRTC_INSTANTIATION_BYTES 256u

#define _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(ans) \
    do { \
        CusrTileStaticEvalNvrtcResult cusr_tile_static_eval_nvrtc_result = (ans); \
        return cusr_tile_static_eval_nvrtc_result; \
    } while (0)

#define _CUSR_TILE_STATIC_EVAL_NVRTC_CHECK_RET(ans) \
    do { \
        CusrTileStaticEvalNvrtcResult cusr_tile_static_eval_nvrtc_check_ret = (ans); \
        if (cusr_tile_static_eval_nvrtc_check_ret != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS) { \
            _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(cusr_tile_static_eval_nvrtc_check_ret); \
        } \
    } while (0)

struct CusrTileStaticEvalNvrtcHandle {
    nvrtcProgram program;
    nvrtcResult compile_result;
};

static int
cusr_tile_static_eval_nvrtc_shape_is_valid(size_t ast_capacity, uint32_t tile_rows, uint32_t threads_per_cta)
{
    return
        (ast_capacity == 8u || ast_capacity == 16u || ast_capacity == 32u) &&
        (tile_rows == 64u || tile_rows == 128u || tile_rows == 256u) &&
        (threads_per_cta == 64u || threads_per_cta == 128u || threads_per_cta == 256u);
}

static int
cusr_tile_static_eval_nvrtc_arch_is_valid(uint32_t capability_major, uint32_t capability_minor)
{
    return
        (capability_major == 8u || capability_major == 9u || capability_major == 10u || capability_major == 12u) &&
        capability_minor < 10u;
}

static CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_make_source(
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t tile_rows,
    uint32_t threads_per_cta,
    char** source_ret)
{
    const size_t template_size = (size_t)cusr_eval_nvrtc_tile_static_eval_template_size;
    size_t source_capacity;
    size_t source_size;
    size_t kernel_idx;
    char* source;

    if (num_kernels > (SIZE_MAX - template_size - 2u) / CUSR_TILE_STATIC_EVAL_NVRTC_INSTANTIATION_BYTES) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_OVERFLOW);
    }

    source_capacity = template_size + num_kernels * CUSR_TILE_STATIC_EVAL_NVRTC_INSTANTIATION_BYTES + 2u;
    source = (char*)malloc(source_capacity);
    if (source == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_ALLOCATION);
    }

    memcpy(source, cusr_eval_nvrtc_tile_static_eval_template_data, template_size);
    source_size = template_size;
    source[source_size++] = '\n';

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        int written = snprintf(
            source + source_size,
            source_capacity - source_size,
            "CUSR_TILE_STATIC_EVAL_TEMPLATE_INSTANTIATE(cusr_tile_static_eval_f32_%03zu, %zuu, %zuu, %uu, %uu)\n",
            kernel_idx,
            kernel_idx,
            ast_capacity,
            tile_rows,
            threads_per_cta);

        if (written <= 0 || (size_t)written >= source_capacity - source_size) {
            free(source);
            _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_OVERFLOW);
        }

        source_size += (size_t)written;
    }

    source[source_size] = '\0';
    *source_ret = source;
    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

static CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_create_impl(
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t tile_rows,
    uint32_t threads_per_cta,
    uint32_t capability_major,
    uint32_t capability_minor,
    char** source_ret,
    CusrTileStaticEvalNvrtcHandle** handle_ret)
{
    char arch_option[64];
    const char* options[7];
    CusrTileStaticEvalNvrtcHandle* handle;
    nvrtcResult nvrtc_result;
    int written;

    if (num_kernels == 0u ||
        !cusr_tile_static_eval_nvrtc_shape_is_valid(ast_capacity, tile_rows, threads_per_cta) ||
        !cusr_tile_static_eval_nvrtc_arch_is_valid(capability_major, capability_minor) ||
        num_kernels > ((size_t)UINT32_MAX - CUSR_TILE_STATIC_EVAL_NVRTC_FIRST_MARKER) / CUSR_TILE_STATIC_EVAL_NVRTC_MARKER_STRIDE + 1u) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE);
    }

    handle = (CusrTileStaticEvalNvrtcHandle*)calloc(1u, sizeof(*handle));
    if (handle == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_ALLOCATION);
    }
    *handle_ret = handle;

    _CUSR_TILE_STATIC_EVAL_NVRTC_CHECK_RET(cusr_tile_static_eval_nvrtc_make_source(
        num_kernels,
        ast_capacity,
        tile_rows,
        threads_per_cta,
        source_ret));

    written = snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=sm_%u%u", capability_major, capability_minor);
    if (written <= 0 || (size_t)written >= sizeof(arch_option)) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_OVERFLOW);
    }

    options[0] = arch_option;
    options[1] = "--std=c++11";
    options[2] = "--device-as-default-execution-space";
    options[3] = "--ptxas-options=--opt-level=1";
    options[4] = "--ptxas-options=-v";
    options[5] = "--ptxas-options=-warn-spills";
    options[6] = "--ptxas-options=-Werror";

    nvrtc_result = nvrtcCreateProgram(&handle->program, *source_ret, "cusr_tile_static_eval_template.cu", 0, NULL, NULL);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC);
    }

    handle->compile_result = nvrtcCompileProgram(handle->program, 7, options);
    if (handle->compile_result != NVRTC_SUCCESS) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

const char*
cusr_tile_static_eval_nvrtc_result_to_string(CusrTileStaticEvalNvrtcResult result)
{
    switch (result) {
        case CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS: return "CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS";
        case CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE: return "CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE";
        case CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_OVERFLOW: return "CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_OVERFLOW";
        case CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_ALLOCATION: return "CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_ALLOCATION";
        case CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC: return "CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC";
        case CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INSUFFICIENT_BUFFER: return "CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INSUFFICIENT_BUFFER";
        default: return "CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_UNKNOWN";
    }
}

CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_create(
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t tile_rows,
    uint32_t threads_per_cta,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrTileStaticEvalNvrtcHandle** handle_ret)
{
    CusrTileStaticEvalNvrtcHandle* handle = NULL;
    char* source = NULL;
    CusrTileStaticEvalNvrtcResult result;

    if (handle_ret == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE);
    }
    *handle_ret = NULL;

    result = cusr_tile_static_eval_nvrtc_create_impl(
        num_kernels,
        ast_capacity,
        tile_rows,
        threads_per_cta,
        capability_major,
        capability_minor,
        &source,
        &handle);

    free(source);

    if (result != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS && (handle == NULL || handle->program == NULL)) {
        cusr_tile_static_eval_nvrtc_destroy(handle);
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(result);
    }

    *handle_ret = handle;
    if (result != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(result);
    }
    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_cubin_size(const CusrTileStaticEvalNvrtcHandle* handle, size_t* cubin_size_ret)
{
    nvrtcResult nvrtc_result;

    if (handle == NULL || handle->program == NULL || handle->compile_result != NVRTC_SUCCESS || cubin_size_ret == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE);
    }

    nvrtc_result = nvrtcGetCUBINSize(handle->program, cubin_size_ret);
    if (nvrtc_result != NVRTC_SUCCESS || *cubin_size_ret == 0u) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_get_cubin(const CusrTileStaticEvalNvrtcHandle* handle, void* cubin, size_t cubin_size)
{
    size_t required_size = 0u;
    nvrtcResult nvrtc_result;

    if (cubin == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_CHECK_RET(cusr_tile_static_eval_nvrtc_cubin_size(handle, &required_size));
    if (cubin_size < required_size) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INSUFFICIENT_BUFFER);
    }

    nvrtc_result = nvrtcGetCUBIN(handle->program, (char*)cubin);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_log_size(const CusrTileStaticEvalNvrtcHandle* handle, size_t* log_size_ret)
{
    nvrtcResult nvrtc_result;

    if (handle == NULL || handle->program == NULL || log_size_ret == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE);
    }

    nvrtc_result = nvrtcGetProgramLogSize(handle->program, log_size_ret);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

CusrTileStaticEvalNvrtcResult
cusr_tile_static_eval_nvrtc_get_log(const CusrTileStaticEvalNvrtcHandle* handle, char* log, size_t log_size)
{
    size_t required_size = 0u;
    nvrtcResult nvrtc_result;

    if (log == NULL) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INVALID_VALUE);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_CHECK_RET(cusr_tile_static_eval_nvrtc_log_size(handle, &required_size));
    if (log_size < required_size) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_INSUFFICIENT_BUFFER);
    }

    nvrtc_result = nvrtcGetProgramLog(handle->program, log);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_NVRTC);
    }

    _CUSR_TILE_STATIC_EVAL_NVRTC_ERROR_RET(CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS);
}

void
cusr_tile_static_eval_nvrtc_destroy(CusrTileStaticEvalNvrtcHandle* handle)
{
    if (handle == NULL) {
        return;
    }

    if (handle->program != NULL) {
        (void)nvrtcDestroyProgram(&handle->program);
    }
    free(handle);
}
