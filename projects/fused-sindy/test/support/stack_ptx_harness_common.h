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
#ifndef STACK_PTX_HARNESS_COMMON_H
#define STACK_PTX_HARNESS_COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <cuda.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int
stack_ptx_test_parse_size(const char* text, size_t* out)
{
    unsigned long long value;
    char* end = NULL;
    if (text == NULL || text[0] == '\0' || out == NULL) return 0;
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text) return 0;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') ++end;
    if (*end != '\0' || value > (unsigned long long)SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int
stack_ptx_test_parse_sm(const char* text, unsigned int* major, unsigned int* minor)
{
    const char* p = text;
    size_t value = 0u;
    if (text == NULL || major == NULL || minor == NULL) return 0;
    if (strncmp(p, "sm_", 3u) == 0) p += 3u;
    else if (strncmp(p, "sm", 2u) == 0) p += 2u;
    else if (strncmp(p, "compute_", 8u) == 0) p += 8u;
    else if (strncmp(p, "compute", 7u) == 0) p += 7u;
    if (!stack_ptx_test_parse_size(p, &value) || value < 10u || value > 999u) return 0;
    *major = (unsigned int)(value / 10u);
    *minor = (unsigned int)(value % 10u);
    return 1;
}

static void
stack_ptx_test_sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static double
stack_ptx_test_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static const char*
stack_ptx_test_cu_name(CUresult result)
{
    const char* name = NULL;
    if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != NULL) return name;
    return "CUDA_ERROR_UNKNOWN";
}

static const char*
stack_ptx_test_cu_string(CUresult result)
{
    const char* string = NULL;
    if (cuGetErrorString(result, &string) == CUDA_SUCCESS && string != NULL) return string;
    return "";
}

static int
stack_ptx_test_init_context(CUdevice* out_device, int* out_owns_primary)
{
    CUcontext context = NULL;
    CUdevice device = 0;
    CUresult rc;

    if (out_device == NULL || out_owns_primary == NULL) return 0;
    *out_device = 0;
    *out_owns_primary = 0;

    rc = cuInit(0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
        return 0;
    }

    rc = cuCtxGetCurrent(&context);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxGetCurrent failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
        return 0;
    }

    if (context != NULL) {
        rc = cuCtxGetDevice(&device);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "cuCtxGetDevice failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
            return 0;
        }
    } else {
        rc = cuDeviceGet(&device, 0);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "cuDeviceGet(0) failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
            return 0;
        }
        rc = cuDevicePrimaryCtxRetain(&context, device);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "cuDevicePrimaryCtxRetain failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
            return 0;
        }
        rc = cuCtxSetCurrent(context);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "cuCtxSetCurrent failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
            cuDevicePrimaryCtxRelease(device);
            return 0;
        }
        *out_owns_primary = 1;
    }

    *out_device = device;
    return 1;
}

static void
stack_ptx_test_release_context(CUdevice device, int owns_primary)
{
    if (owns_primary) {
        cuCtxSetCurrent(NULL);
        cuDevicePrimaryCtxRelease(device);
    }
}

static int
stack_ptx_test_device_sm(CUdevice device, unsigned int* major, unsigned int* minor)
{
    int major_i = 0;
    int minor_i = 0;
    CUresult rc;
    if (major == NULL || minor == NULL) return 0;

    rc = cuDeviceGetAttribute(&major_i, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGetAttribute(major) failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
        return 0;
    }
    rc = cuDeviceGetAttribute(&minor_i, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGetAttribute(minor) failed: %s %s\n", stack_ptx_test_cu_name(rc), stack_ptx_test_cu_string(rc));
        return 0;
    }
    if (major_i <= 0 || minor_i < 0) return 0;
    *major = (unsigned int)major_i;
    *minor = (unsigned int)minor_i;
    return 1;
}

#endif /* STACK_PTX_HARNESS_COMMON_H */
