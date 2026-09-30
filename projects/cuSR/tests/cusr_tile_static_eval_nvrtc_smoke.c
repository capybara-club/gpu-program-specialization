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

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

static void
cusr_test_print_compile_log(const CusrTileStaticEvalNvrtcHandle* handle)
{
    size_t log_size = 0u;
    char* log;

    if (handle == NULL ||
        cusr_tile_static_eval_nvrtc_log_size(handle, &log_size) != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS ||
        log_size <= 1u) {
        return;
    }

    log = (char*)malloc(log_size);
    if (log == NULL) {
        return;
    }
    if (cusr_tile_static_eval_nvrtc_get_log(handle, log, log_size) == CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS) {
        fputs(log, stderr);
    }
    free(log);
}

static int
cusr_test_compile(
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrTileStaticEvalNvrtcHandle** handle_ret,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret)
{
    CusrTileStaticEvalNvrtcResult result;

    result = cusr_tile_static_eval_nvrtc_create(
        2u,
        8u,
        64u,
        128u,
        capability_major,
        capability_minor,
        handle_ret);
    if (result != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_eval_nvrtc_create failed: %s\n", cusr_tile_static_eval_nvrtc_result_to_string(result));
        cusr_test_print_compile_log(*handle_ret);
        return 0;
    }

    result = cusr_tile_static_eval_nvrtc_cubin_size(*handle_ret, cubin_size_ret);
    if (result != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_eval_nvrtc_cubin_size failed: %s\n", cusr_tile_static_eval_nvrtc_result_to_string(result));
        return 0;
    }

    *cubin_ret = (unsigned char*)malloc(*cubin_size_ret);
    if (*cubin_ret == NULL) {
        fputs("failed to allocate the eval cubin\n", stderr);
        return 0;
    }

    result = cusr_tile_static_eval_nvrtc_get_cubin(*handle_ret, *cubin_ret, *cubin_size_ret);
    if (result != CUSR_TILE_STATIC_EVAL_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_eval_nvrtc_get_cubin failed: %s\n", cusr_tile_static_eval_nvrtc_result_to_string(result));
        return 0;
    }
    if (*cubin_size_ret < 4u ||
        (*cubin_ret)[0] != 0x7fu ||
        (*cubin_ret)[1] != 'E' ||
        (*cubin_ret)[2] != 'L' ||
        (*cubin_ret)[3] != 'F') {
        fputs("NVRTC returned an invalid eval cubin\n", stderr);
        return 0;
    }

    return 1;
}

int
main(void)
{
    CusrTileStaticEvalNvrtcHandle* handle = NULL;
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;
    CUdevice device;
    int capability_major = 0;
    int capability_minor = 0;
    int ok = 0;

    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS) {
        fputs("failed to query CUDA device 0\n", stderr);
    } else {
        ok = cusr_test_compile(
            (uint32_t)capability_major,
            (uint32_t)capability_minor,
            &handle,
            &cubin,
            &cubin_size);
    }

    free(cubin);
    cusr_tile_static_eval_nvrtc_destroy(handle);
    return ok ? 0 : 1;
}
