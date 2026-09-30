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
/* Test-only LD_PRELOAD shim: fail exactly one 65-float upload in its own process.
 * Zero destination first to make accidental reuse deterministic. Never preload
 * this into a normal worker. Does not modify any driver or running service. */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes) {
    typedef CUresult (*Copy)(CUdeviceptr, const void *, size_t);
    typedef CUresult (*Zero)(CUdeviceptr, unsigned, size_t);
    static int injected;
    static void *driver;
    if (!driver) driver = dlopen("libcuda.so.1", RTLD_NOW);
    if (!driver) return CUDA_ERROR_UNKNOWN;
    Copy copy = (Copy)dlsym(driver, "cuMemcpyHtoD_v2");
    if (!copy) return CUDA_ERROR_UNKNOWN;
    if (!injected && bytes == 260 && getenv("ODEZZA_AUDIT_UPLOAD_FAILURE")) {
        Zero zero = (Zero)dlsym(driver, "cuMemsetD32_v2");
        if (!zero) return CUDA_ERROR_UNKNOWN;
        injected = 1;
        if (zero(dst, 0, 65) != CUDA_SUCCESS) return CUDA_ERROR_UNKNOWN;
        fputs("AUDIT: injected one 260-byte grid upload failure\n", stderr);
        return CUDA_ERROR_INVALID_VALUE;
    }
    return copy(dst, src, bytes);
}
