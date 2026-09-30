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
#ifndef ELITE_NLE_STACK_PTX_INJECT_KERNEL_GEN_H
#define ELITE_NLE_STACK_PTX_INJECT_KERNEL_GEN_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    STACK_PTX_KERNEL_GEN_SUCCESS = 0,
    STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE = 1,
    STACK_PTX_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER = 2,
    STACK_PTX_KERNEL_GEN_ERROR_OUT_OF_MEMORY = 3,
    STACK_PTX_KERNEL_GEN_ERROR_FORMAT = 4
} StackPtxKernelGenResult;

#ifdef __cplusplus
extern "C" {
#endif

StackPtxKernelGenResult elite_nle_stack_ptx_inject_kernel_gen(
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
);

#ifdef __cplusplus
}
#endif

#endif // ELITE_NLE_STACK_PTX_INJECT_KERNEL_GEN_H
