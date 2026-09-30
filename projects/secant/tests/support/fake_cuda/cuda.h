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
/* Test-only subset of the CUDA driver ABI. Never on a production include path. */
#ifndef SECANT_FAKE_CUDA_H
#define SECANT_FAKE_CUDA_H
#include <stddef.h>
#include <stdint.h>
typedef int CUresult;
typedef uint64_t CUdeviceptr;
typedef void *CUcontext;
typedef struct TestStream *CUstream;
typedef struct TestEvent *CUevent;
typedef struct TestModule *CUmodule;
typedef struct TestModule *CUfunction;
typedef int CUmoduleLoadingMode;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_UNKNOWN 999
#define CU_MODULE_EAGER_LOADING 1
#define CU_STREAM_NON_BLOCKING 1
#define CU_EVENT_DEFAULT 0
#define CU_EVENT_DISABLE_TIMING 2
CUresult cuCtxGetCurrent(CUcontext *);
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *);
CUresult cuStreamCreate(CUstream *, unsigned);
CUresult cuStreamDestroy(CUstream);
CUresult cuStreamSynchronize(CUstream);
CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned);
CUresult cuEventCreate(CUevent *, unsigned);
CUresult cuEventDestroy(CUevent);
CUresult cuEventRecord(CUevent, CUstream);
CUresult cuEventSynchronize(CUevent);
CUresult cuEventElapsedTime(float *, CUevent, CUevent);
CUresult cuModuleLoadData(CUmodule *, const void *);
CUresult cuModuleUnload(CUmodule);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                        CUstream, void **, void **);
CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned, size_t, size_t, CUstream);
#endif
