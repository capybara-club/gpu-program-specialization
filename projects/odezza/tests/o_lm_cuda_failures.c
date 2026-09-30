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
/* Linker-injected driver failures for the public native LM lifecycle test.
 * Real CUDA executes every successful call. No production test hooks exist. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "o_odezza_internal.h"
int o_test_fail_load, o_test_fail_launch, o_test_fail_event;
int o_test_fail_specialize;
OdezzaResult __real_o_lm_specialize(const void *, size_t, const OLmInspection *,
    const OdezzaAstProgram *, uint32_t, OLmExpressions *, OSassInstruction *, void *, uint32_t *);
OdezzaResult __wrap_o_lm_specialize(const void *image, size_t size, const OLmInspection *inspection,
    const OdezzaAstProgram *rhs, uint32_t bits, OLmExpressions *expressions,
    OSassInstruction *code, void *out, uint32_t *registers) {
    if (o_test_fail_specialize && --o_test_fail_specialize == 0)
        return ODEZZA_ERROR_REGISTER_PRESSURE;
    return __real_o_lm_specialize(image, size, inspection, rhs, bits, expressions, code, out, registers);
}
static int modules, streams;
CUresult __real_cuModuleLoadData(CUmodule *, const void *);
CUresult __real_cuModuleUnload(CUmodule);
CUresult __real_cuStreamCreate(CUstream *, unsigned int);
CUresult __real_cuStreamDestroy_v2(CUstream);
CUresult __real_cuEventRecord(CUevent, CUstream);
CUresult __real_cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
CUresult __wrap_cuModuleLoadData(CUmodule *module, const void *image) {
    CUresult r;
    if (o_test_fail_load && --o_test_fail_load == 0)
        return CUDA_ERROR_INVALID_IMAGE;
    r = __real_cuModuleLoadData(module, image);
    if (r == CUDA_SUCCESS)
        ++modules;
    return r;
}
CUresult __wrap_cuModuleUnload(CUmodule module) {
    CUresult r = __real_cuModuleUnload(module);
    if (r == CUDA_SUCCESS)
        --modules;
    return r;
}
CUresult __wrap_cuStreamCreate(CUstream *stream, unsigned int flags) {
    CUresult r = __real_cuStreamCreate(stream, flags);
    if (r == CUDA_SUCCESS)
        ++streams;
    return r;
}
CUresult __wrap_cuStreamDestroy_v2(CUstream stream) {
    CUresult r = __real_cuStreamDestroy_v2(stream);
    if (r == CUDA_SUCCESS)
        --streams;
    return r;
}
CUresult __wrap_cuEventRecord(CUevent event, CUstream stream) {
    if (o_test_fail_event && --o_test_fail_event == 0)
        return CUDA_ERROR_INVALID_VALUE;
    return __real_cuEventRecord(event, stream);
}
CUresult __wrap_cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                               unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                               CUstream stream, void **args, void **extra) {
    if (o_test_fail_launch && --o_test_fail_launch == 0)
        return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    return __real_cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
}
void o_test_check_resources(int expected_streams) {
    if (modules != 0 || streams != expected_streams) {
        fprintf(stderr, "native resource leak: %d modules, %d streams\n", modules, streams);
        exit(1);
    }
}
