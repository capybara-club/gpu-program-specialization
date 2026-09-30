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
/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cuda.h>

#include <check_result_helper.h>

__attribute__((unused))
static
void
get_device_capability(
    CUdevice device,
    int* compute_capability_major_out,
    int* compute_capability_minor_out
) {
    cuCheck( cuDeviceGetAttribute(compute_capability_major_out, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );
    cuCheck( cuDeviceGetAttribute(compute_capability_minor_out, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) );
}

__attribute__((unused))
static
CUresult
cuContextCreate(CUcontext* context, CUdevice device) {
    #ifdef CUDA_VERSION_13
        return cuCtxCreate(context, NULL, 0, device);
    #else
        return cuCtxCreate(context, 0, device);
    #endif
}

