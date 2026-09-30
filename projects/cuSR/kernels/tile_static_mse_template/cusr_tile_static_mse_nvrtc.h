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
#ifndef CUSR_TILE_STATIC_MSE_NVRTC_H_INCLUDED
#define CUSR_TILE_STATIC_MSE_NVRTC_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CusrTileStaticMseNvrtcResult {
    CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS = 0,
    CUSR_TILE_STATIC_MSE_NVRTC_ERROR_INVALID_VALUE = 1,
    CUSR_TILE_STATIC_MSE_NVRTC_ERROR_OVERFLOW = 2,
    CUSR_TILE_STATIC_MSE_NVRTC_ERROR_ALLOCATION = 3,
    CUSR_TILE_STATIC_MSE_NVRTC_ERROR_NVRTC = 4,
    CUSR_TILE_STATIC_MSE_NVRTC_ERROR_INSUFFICIENT_BUFFER = 5
} CusrTileStaticMseNvrtcResult;

typedef struct CusrTileStaticMseNvrtcHandle CusrTileStaticMseNvrtcHandle;

const char*
cusr_tile_static_mse_nvrtc_result_to_string(CusrTileStaticMseNvrtcResult result);

/*
 * Compiles one module with stable cusr_tile_static_mse_f32_%03d symbols and
 * retains the NVRTC program. If compilation fails after program creation,
 * handle_ret remains valid so the compiler log can be read before destruction.
 */
CusrTileStaticMseNvrtcResult
cusr_tile_static_mse_nvrtc_create(
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t tile_rows,
    uint32_t threads_per_cta,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrTileStaticMseNvrtcHandle** handle_ret
);

CusrTileStaticMseNvrtcResult
cusr_tile_static_mse_nvrtc_cubin_size(const CusrTileStaticMseNvrtcHandle* handle, size_t* cubin_size_ret);

CusrTileStaticMseNvrtcResult
cusr_tile_static_mse_nvrtc_get_cubin(const CusrTileStaticMseNvrtcHandle* handle, void* cubin, size_t cubin_size);

CusrTileStaticMseNvrtcResult
cusr_tile_static_mse_nvrtc_log_size(const CusrTileStaticMseNvrtcHandle* handle, size_t* log_size_ret);

CusrTileStaticMseNvrtcResult
cusr_tile_static_mse_nvrtc_get_log(const CusrTileStaticMseNvrtcHandle* handle, char* log, size_t log_size);

void
cusr_tile_static_mse_nvrtc_destroy(CusrTileStaticMseNvrtcHandle* handle);

#ifdef __cplusplus
}
#endif

#endif /* CUSR_TILE_STATIC_MSE_NVRTC_H_INCLUDED */
