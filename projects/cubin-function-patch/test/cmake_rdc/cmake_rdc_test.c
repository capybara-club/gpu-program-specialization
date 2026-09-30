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
 * SPDX-FileCopyrightText: Copyright (c) 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 */

#define CUBIN_FUNCTION_PATCH_IMPLEMENTATION
#include "cubin_function_patch.h"

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern const unsigned char cfp_cmake_rdc_reserved_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_reserved_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_simple_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_simple_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_poly_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_poly_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_simple_rdc_cubin_data[];
extern const unsigned int cfp_cmake_rdc_simple_rdc_cubin_size;
extern const unsigned char cfp_cmake_rdc_poly_rdc_cubin_data[];
extern const unsigned int cfp_cmake_rdc_poly_rdc_cubin_size;
extern const unsigned char cfp_cmake_rdc_switch_reserved_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_switch_reserved_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_switch_replacement_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_switch_replacement_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_switch_replacement_rdc_cubin_data[];
extern const unsigned int cfp_cmake_rdc_switch_replacement_rdc_cubin_size;
extern const unsigned char cfp_cmake_rdc_panel_reserved_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_panel_reserved_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_panel_replacement_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_panel_replacement_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_panel_replacement_rdc_cubin_data[];
extern const unsigned int cfp_cmake_rdc_panel_replacement_rdc_cubin_size;
extern const unsigned char cfp_cmake_rdc_multi_panel_reserved_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_multi_panel_reserved_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_multi_panel_replacement_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_multi_panel_replacement_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_multi_panel_sparse_replacement_linked_cubin_data[];
extern const unsigned int cfp_cmake_rdc_multi_panel_sparse_replacement_linked_cubin_size;
extern const unsigned char cfp_cmake_rdc_multi_panel_replacement_rdc_cubin_data[];
extern const unsigned int cfp_cmake_rdc_multi_panel_replacement_rdc_cubin_size;
extern const unsigned char cfp_cmake_rdc_multi_panel_sparse_replacement_rdc_cubin_data[];
extern const unsigned int cfp_cmake_rdc_multi_panel_sparse_replacement_rdc_cubin_size;

typedef struct {
    void* data;
    size_t bytes;
} TestBlob;

static void
free_blob(
    TestBlob* blob
) {
    if (blob != NULL) {
        free(blob->data);
        blob->data = NULL;
        blob->bytes = 0u;
    }
}

static int
check_cuda(
    CUresult result,
    const char* expr
) {
    const char* name = NULL;
    const char* text = NULL;
    if (result == CUDA_SUCCESS) return 1;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &text);
    fprintf(stderr, "%s failed: %s: %s\n", expr, name != NULL ? name : "CUDA_ERROR", text != NULL ? text : "");
    return 0;
}

#define CHECK_CUDA(EXPR) check_cuda((EXPR), #EXPR)

static int
run_patch_kernel(
    const void* cubin,
    const float* input,
    float* output,
    size_t count
) {
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    unsigned long long n_arg = (unsigned long long)count;
    unsigned int block = 128u;
    unsigned int grid = (unsigned int)((count + block - 1u) / block);
    void* args[3];
    int ok = 0;

    if (cubin == NULL || input == NULL || output == NULL || count == 0u) return 0;
    if (!CHECK_CUDA(cuModuleLoadData(&module, cubin))) goto done;
    if (!CHECK_CUDA(cuModuleGetFunction(&kernel, module, "patch_kernel"))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_input, count * sizeof(float)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_output, count * sizeof(float)))) goto done;
    if (!CHECK_CUDA(cuMemcpyHtoD(d_input, input, count * sizeof(float)))) goto done;
    args[0] = &d_input;
    args[1] = &d_output;
    args[2] = &n_arg;
    if (!CHECK_CUDA(cuLaunchKernel(kernel, grid, 1u, 1u, block, 1u, 1u, 0u, NULL, args, NULL))) goto done;
    if (!CHECK_CUDA(cuCtxSynchronize())) goto done;
    if (!CHECK_CUDA(cuMemcpyDtoH(output, d_output, count * sizeof(float)))) goto done;
    ok = 1;

done:
    if (d_output != 0) (void)cuMemFree(d_output);
    if (d_input != 0) (void)cuMemFree(d_input);
    if (module != NULL) (void)cuModuleUnload(module);
    return ok;
}

static int
run_panel_kernel(
    const void* cubin,
    const float* input,
    float* output
) {
    enum {
        ROWS = 128,
        FEATURES = 32,
        LEAVES = 8
    };
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    CUdeviceptr d_leaf_ptrs = 0;
    CUdeviceptr d_leaf_strides = 0;
    CUdeviceptr h_leaf_ptrs[FEATURES * LEAVES];
    int h_leaf_strides[FEATURES * LEAVES];
    int active_rows = ROWS;
    void* args[4];
    size_t i;
    int ok = 0;

    if (cubin == NULL || input == NULL || output == NULL) return 0;
    if (!CHECK_CUDA(cuModuleLoadData(&module, cubin))) goto done;
    if (!CHECK_CUDA(cuModuleGetFunction(&kernel, module, "patch_panel_kernel"))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_input, ROWS * sizeof(float)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_output, ROWS * FEATURES * sizeof(float)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_leaf_ptrs, FEATURES * LEAVES * sizeof(CUdeviceptr)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_leaf_strides, FEATURES * LEAVES * sizeof(int)))) goto done;
    if (!CHECK_CUDA(cuMemcpyHtoD(d_input, input, ROWS * sizeof(float)))) goto done;
    for (i = 0u; i < FEATURES * LEAVES; ++i) {
        h_leaf_ptrs[i] = d_input;
        h_leaf_strides[i] = 1;
    }
    if (!CHECK_CUDA(cuMemcpyHtoD(d_leaf_ptrs, h_leaf_ptrs, sizeof(h_leaf_ptrs)))) goto done;
    if (!CHECK_CUDA(cuMemcpyHtoD(d_leaf_strides, h_leaf_strides, sizeof(h_leaf_strides)))) goto done;
    args[0] = &d_output;
    args[1] = &d_leaf_ptrs;
    args[2] = &d_leaf_strides;
    args[3] = &active_rows;
    if (!CHECK_CUDA(cuLaunchKernel(kernel, 1u, 1u, 1u, 128u, 1u, 1u, 0u, NULL, args, NULL))) goto done;
    if (!CHECK_CUDA(cuCtxSynchronize())) goto done;
    if (!CHECK_CUDA(cuMemcpyDtoH(output, d_output, ROWS * FEATURES * sizeof(float)))) goto done;
    ok = 1;

done:
    if (d_leaf_strides != 0) (void)cuMemFree(d_leaf_strides);
    if (d_leaf_ptrs != 0) (void)cuMemFree(d_leaf_ptrs);
    if (d_output != 0) (void)cuMemFree(d_output);
    if (d_input != 0) (void)cuMemFree(d_input);
    if (module != NULL) (void)cuModuleUnload(module);
    return ok;
}

static int
run_multi_panel_kernel(
    const void* cubin,
    size_t kernel_idx,
    const float* input,
    float* output
) {
    enum {
        ROWS = 128,
        FEATURES = 32,
        LEAVES = 8
    };
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    CUdeviceptr d_leaf_ptrs = 0;
    CUdeviceptr d_leaf_strides = 0;
    CUdeviceptr h_leaf_ptrs[FEATURES * LEAVES];
    int h_leaf_strides[FEATURES * LEAVES];
    int active_rows = ROWS;
    char kernel_name[64];
    void* args[4];
    size_t i;
    int ok = 0;

    if (cubin == NULL || input == NULL || output == NULL || kernel_idx >= 4u) return 0;
    snprintf(kernel_name, sizeof(kernel_name), "patch_multi_panel_kernel_%zu", kernel_idx);
    if (!CHECK_CUDA(cuModuleLoadData(&module, cubin))) goto done;
    if (!CHECK_CUDA(cuModuleGetFunction(&kernel, module, kernel_name))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_input, ROWS * sizeof(float)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_output, ROWS * FEATURES * sizeof(float)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_leaf_ptrs, FEATURES * LEAVES * sizeof(CUdeviceptr)))) goto done;
    if (!CHECK_CUDA(cuMemAlloc(&d_leaf_strides, FEATURES * LEAVES * sizeof(int)))) goto done;
    if (!CHECK_CUDA(cuMemcpyHtoD(d_input, input, ROWS * sizeof(float)))) goto done;
    for (i = 0u; i < FEATURES * LEAVES; ++i) {
        h_leaf_ptrs[i] = d_input;
        h_leaf_strides[i] = 1;
    }
    if (!CHECK_CUDA(cuMemcpyHtoD(d_leaf_ptrs, h_leaf_ptrs, sizeof(h_leaf_ptrs)))) goto done;
    if (!CHECK_CUDA(cuMemcpyHtoD(d_leaf_strides, h_leaf_strides, sizeof(h_leaf_strides)))) goto done;
    args[0] = &d_output;
    args[1] = &d_leaf_ptrs;
    args[2] = &d_leaf_strides;
    args[3] = &active_rows;
    if (!CHECK_CUDA(cuLaunchKernel(kernel, 1u, 1u, 1u, 128u, 1u, 1u, 0u, NULL, args, NULL))) goto done;
    if (!CHECK_CUDA(cuCtxSynchronize())) goto done;
    if (!CHECK_CUDA(cuMemcpyDtoH(output, d_output, ROWS * FEATURES * sizeof(float)))) goto done;
    ok = 1;

done:
    if (d_leaf_strides != 0) (void)cuMemFree(d_leaf_strides);
    if (d_leaf_ptrs != 0) (void)cuMemFree(d_leaf_ptrs);
    if (d_output != 0) (void)cuMemFree(d_output);
    if (d_input != 0) (void)cuMemFree(d_input);
    if (module != NULL) (void)cuModuleUnload(module);
    return ok;
}

static int
run_multi_panel_kernels(
    const void* cubin,
    const float* input,
    float* output
) {
    enum {
        PANEL_VALUES = 32 * 128
    };
    size_t i;
    if (cubin == NULL || input == NULL || output == NULL) return 0;
    for (i = 0u; i < 4u; ++i) {
        if (!run_multi_panel_kernel(cubin, i, input, output + i * PANEL_VALUES)) return 0;
    }
    return 1;
}

static int
compare_outputs(
    const char* label,
    const float* a,
    const float* b,
    size_t count
) {
    size_t i;
    float max_abs = 0.0f;
    for (i = 0u; i < count; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_abs) max_abs = diff;
    }
    printf("%s max_abs=%g\n", label, (double)max_abs);
    return max_abs < 1.0e-5f;
}

static int
expect_patch_result(
    CubinFunctionPatchResult result,
    CubinFunctionPatchResult expected,
    const char* label
) {
    if (result == expected) return 1;
    fprintf(
        stderr,
        "%s expected %s, got %s\n",
        label,
        cubin_function_patch_result_to_string(expected),
        cubin_function_patch_result_to_string(result)
    );
    return 0;
}

int
main(
    void
) {
    CUdevice device;
    CUcontext context = NULL;
    int ctx_created = 0;
    int sm_major = 0;
    int sm_minor = 0;
    int runtime_sm = 0;
    TestBlob simple_patched_cubin = {0};
    TestBlob poly_patched_cubin = {0};
    TestBlob switch_patched_cubin = {0};
    TestBlob panel_patched_cubin = {0};
    TestBlob multi_panel_patched_cubin = {0};
    TestBlob multi_panel_sparse_patched_cubin = {0};
    TestBlob failure_cubin = {0};
    void* handle_memory = NULL;
    size_t handle_memory_size = 0u;
    CubinFunctionPatchHandle* handle = NULL;
    const char* symbols[1] = {"patch_site"};
    const char* duplicate_symbols[2] = {"patch_site", "patch_site"};
    CubinFunctionPatchReport simple_report;
    size_t output_size = 0u;
    const size_t count = 257u;
    float input[257];
    float simple_ref[257];
    float simple_patch[257];
    float poly_ref[257];
    float poly_patch[257];
    float switch_ref[257];
    float switch_patch[257];
    float panel_ref[32 * 128];
    float panel_patch[32 * 128];
    float multi_panel_ref[4 * 32 * 128];
    float multi_panel_patch[4 * 32 * 128];
    float multi_panel_sparse_ref[4 * 32 * 128];
    float multi_panel_sparse_patch[4 * 32 * 128];
    size_t i;
    int ok = 0;

    for (i = 0u; i < count; ++i) {
        input[i] = (float)i * 0.03125f - 3.0f;
    }

    if (!CHECK_CUDA(cuInit(0))) goto done;
    if (!CHECK_CUDA(cuDeviceGet(&device, 0))) goto done;
    if (!CHECK_CUDA(cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))) goto done;
    if (!CHECK_CUDA(cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))) goto done;
    runtime_sm = sm_major * 10 + sm_minor;
    if (runtime_sm != CFP_CMAKE_RDC_TEST_SM) {
        printf("SKIP cmake_rdc test built for sm_%d, runtime device is sm_%d\n", CFP_CMAKE_RDC_TEST_SM, runtime_sm);
        ok = 1;
        goto done;
    }
    if (!CHECK_CUDA(cuDevicePrimaryCtxRetain(&context, device))) goto done;
    ctx_created = 1;
    if (!CHECK_CUDA(cuCtxSetCurrent(context))) goto done;
    printf("device sm_%d%d\n", sm_major, sm_minor);

    if (!expect_patch_result(
            cubin_function_patch_handle_size(duplicate_symbols, 2u, &handle_memory_size),
            CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL,
            "duplicate symbol check")) {
        goto done;
    }
    if (!expect_patch_result(
            cubin_function_patch_handle_size(symbols, 1u, &handle_memory_size),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "handle size")) {
        goto done;
    }
    handle_memory = malloc(handle_memory_size);
    if (handle_memory == NULL) goto done;
    if (!expect_patch_result(
            cubin_function_patch_create(cfp_cmake_rdc_reserved_linked_cubin_data, cfp_cmake_rdc_reserved_linked_cubin_size, symbols, 1u, handle_memory, handle_memory_size, &handle),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "create")) {
        goto done;
    }
    if (!expect_patch_result(cubin_function_patch_output_size(handle, &output_size), CUBIN_FUNCTION_PATCH_SUCCESS, "output size")) goto done;
    simple_patched_cubin.data = malloc(output_size);
    poly_patched_cubin.data = malloc(output_size);
    switch_patched_cubin.data = malloc(cfp_cmake_rdc_switch_reserved_linked_cubin_size);
    panel_patched_cubin.data = malloc(cfp_cmake_rdc_panel_reserved_linked_cubin_size);
    failure_cubin.data = malloc(output_size);
    if (simple_patched_cubin.data == NULL || poly_patched_cubin.data == NULL ||
        switch_patched_cubin.data == NULL || panel_patched_cubin.data == NULL ||
        failure_cubin.data == NULL) goto done;

    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "patch_site",
                cfp_cmake_rdc_simple_rdc_cubin_data,
                cfp_cmake_rdc_simple_rdc_cubin_size,
                simple_patched_cubin.data,
                output_size,
                &simple_patched_cubin.bytes,
                &simple_report),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "apply simple")) {
        goto done;
    }
    printf("patch_site reserved=%zu replacement=%zu\n", simple_report.reserved_size, simple_report.replacement_size);
    if (simple_report.replacement_size != simple_report.reserved_size) {
        size_t exact_bytes = 0u;
        if (!expect_patch_result(
                cubin_function_patch_apply_one_ex(
                    handle,
                    "patch_site",
                    cfp_cmake_rdc_simple_rdc_cubin_data,
                    cfp_cmake_rdc_simple_rdc_cubin_size,
                    CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE,
                    failure_cubin.data,
                    output_size,
                    &exact_bytes,
                    NULL),
                CUBIN_FUNCTION_PATCH_ERROR_TAIL_POLICY,
                "exact tail policy")) {
            goto done;
        }
    }
    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "missing_symbol",
                cfp_cmake_rdc_simple_rdc_cubin_data,
                cfp_cmake_rdc_simple_rdc_cubin_size,
                failure_cubin.data,
                output_size,
                &failure_cubin.bytes,
                NULL),
            CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND,
            "missing symbol")) {
        goto done;
    }
    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "patch_site",
                "bad",
                3u,
                failure_cubin.data,
                output_size,
                &failure_cubin.bytes,
                NULL),
            CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF,
            "invalid replacement elf")) {
        goto done;
    }
    if (!expect_patch_result(
            cubin_function_patch_apply_all(
                handle,
                cfp_cmake_rdc_poly_rdc_cubin_data,
                cfp_cmake_rdc_poly_rdc_cubin_size,
                poly_patched_cubin.data,
                output_size,
                &poly_patched_cubin.bytes,
                NULL,
                0u),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "apply poly")) {
        goto done;
    }

    if (!run_patch_kernel(cfp_cmake_rdc_simple_linked_cubin_data, input, simple_ref, count)) goto done;
    if (!run_patch_kernel(simple_patched_cubin.data, input, simple_patch, count)) goto done;
    if (!compare_outputs("simple patched vs linked", simple_ref, simple_patch, count)) goto done;
    if (!run_patch_kernel(cfp_cmake_rdc_poly_linked_cubin_data, input, poly_ref, count)) goto done;
    if (!run_patch_kernel(poly_patched_cubin.data, input, poly_patch, count)) goto done;
    if (!compare_outputs("poly patched vs linked", poly_ref, poly_patch, count)) goto done;

    free(handle_memory);
    handle_memory = NULL;
    handle = NULL;
    if (!expect_patch_result(
            cubin_function_patch_handle_size(symbols, 1u, &handle_memory_size),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "switch handle size")) {
        goto done;
    }
    handle_memory = malloc(handle_memory_size);
    if (handle_memory == NULL) goto done;
    if (!expect_patch_result(
            cubin_function_patch_create(cfp_cmake_rdc_switch_reserved_linked_cubin_data, cfp_cmake_rdc_switch_reserved_linked_cubin_size, symbols, 1u, handle_memory, handle_memory_size, &handle),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "switch create")) {
        goto done;
    }
    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "patch_site",
                cfp_cmake_rdc_switch_replacement_rdc_cubin_data,
                cfp_cmake_rdc_switch_replacement_rdc_cubin_size,
                switch_patched_cubin.data,
                cfp_cmake_rdc_switch_reserved_linked_cubin_size,
                &switch_patched_cubin.bytes,
                NULL),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "apply switch")) {
        goto done;
    }
    if (!run_patch_kernel(cfp_cmake_rdc_switch_replacement_linked_cubin_data, input, switch_ref, count)) goto done;
    if (!run_patch_kernel(switch_patched_cubin.data, input, switch_patch, count)) goto done;
    if (!compare_outputs("switch patched vs linked", switch_ref, switch_patch, count)) goto done;

    if (handle != NULL) cubin_function_patch_destroy(handle);
    free(handle_memory);
    handle_memory = NULL;
    handle = NULL;
    if (!expect_patch_result(
            cubin_function_patch_handle_size((const char* const[]){"patch_panel_site"}, 1u, &handle_memory_size),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "panel handle size")) {
        goto done;
    }
    handle_memory = malloc(handle_memory_size);
    if (handle_memory == NULL) goto done;
    if (!expect_patch_result(
            cubin_function_patch_create(cfp_cmake_rdc_panel_reserved_linked_cubin_data, cfp_cmake_rdc_panel_reserved_linked_cubin_size, (const char* const[]){"patch_panel_site"}, 1u, handle_memory, handle_memory_size, &handle),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "panel create")) {
        goto done;
    }
    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "patch_panel_site",
                cfp_cmake_rdc_panel_replacement_rdc_cubin_data,
                cfp_cmake_rdc_panel_replacement_rdc_cubin_size,
                panel_patched_cubin.data,
                cfp_cmake_rdc_panel_reserved_linked_cubin_size,
                &panel_patched_cubin.bytes,
                NULL),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "apply panel")) {
        goto done;
    }
    if (!run_panel_kernel(cfp_cmake_rdc_panel_replacement_linked_cubin_data, input, panel_ref)) goto done;
    if (!run_panel_kernel(panel_patched_cubin.data, input, panel_patch)) goto done;
    if (!compare_outputs("panel patched vs linked", panel_ref, panel_patch, 32u * 128u)) goto done;

    if (handle != NULL) cubin_function_patch_destroy(handle);
    free(handle_memory);
    handle_memory = NULL;
    handle = NULL;
    {
        const char* reversed_symbols[4] = {
            "patch_multi_panel_site_3",
            "patch_multi_panel_site_2",
            "patch_multi_panel_site_1",
            "patch_multi_panel_site_0"
        };
        CubinFunctionPatchReport reports[4];

        if (!expect_patch_result(
                cubin_function_patch_handle_size(reversed_symbols, 4u, &handle_memory_size),
                CUBIN_FUNCTION_PATCH_SUCCESS,
                "multi-panel handle size")) {
            goto done;
        }
        handle_memory = malloc(handle_memory_size);
        if (handle_memory == NULL) goto done;
        if (!expect_patch_result(
                cubin_function_patch_create(
                    cfp_cmake_rdc_multi_panel_reserved_linked_cubin_data,
                    cfp_cmake_rdc_multi_panel_reserved_linked_cubin_size,
                    reversed_symbols,
                    4u,
                    handle_memory,
                    handle_memory_size,
                    &handle),
                CUBIN_FUNCTION_PATCH_SUCCESS,
                "multi-panel create")) {
            goto done;
        }
        multi_panel_patched_cubin.data = malloc(cfp_cmake_rdc_multi_panel_reserved_linked_cubin_size);
        multi_panel_sparse_patched_cubin.data = malloc(cfp_cmake_rdc_multi_panel_reserved_linked_cubin_size);
        if (multi_panel_patched_cubin.data == NULL || multi_panel_sparse_patched_cubin.data == NULL) goto done;
        if (!expect_patch_result(
                cubin_function_patch_apply_all(
                    handle,
                    cfp_cmake_rdc_multi_panel_replacement_rdc_cubin_data,
                    cfp_cmake_rdc_multi_panel_replacement_rdc_cubin_size,
                    multi_panel_patched_cubin.data,
                    cfp_cmake_rdc_multi_panel_reserved_linked_cubin_size,
                    &multi_panel_patched_cubin.bytes,
                    reports,
                    4u),
                CUBIN_FUNCTION_PATCH_SUCCESS,
                "apply multi-panel dense")) {
            goto done;
        }
        printf(
            "multi-panel dense reports: %zu/%zu %zu/%zu %zu/%zu %zu/%zu\n",
            reports[0].replacement_size,
            reports[0].reserved_size,
            reports[1].replacement_size,
            reports[1].reserved_size,
            reports[2].replacement_size,
            reports[2].reserved_size,
            reports[3].replacement_size,
            reports[3].reserved_size
        );
        if (!run_multi_panel_kernels(cfp_cmake_rdc_multi_panel_replacement_linked_cubin_data, input, multi_panel_ref)) goto done;
        if (!run_multi_panel_kernels(multi_panel_patched_cubin.data, input, multi_panel_patch)) goto done;
        if (!compare_outputs("multi-panel dense patched vs linked", multi_panel_ref, multi_panel_patch, 4u * 32u * 128u)) goto done;

        if (!expect_patch_result(
                cubin_function_patch_apply_all(
                    handle,
                    cfp_cmake_rdc_multi_panel_sparse_replacement_rdc_cubin_data,
                    cfp_cmake_rdc_multi_panel_sparse_replacement_rdc_cubin_size,
                    multi_panel_sparse_patched_cubin.data,
                    cfp_cmake_rdc_multi_panel_reserved_linked_cubin_size,
                    &multi_panel_sparse_patched_cubin.bytes,
                    reports,
                    4u),
                CUBIN_FUNCTION_PATCH_SUCCESS,
                "apply multi-panel sparse")) {
            goto done;
        }
        printf(
            "multi-panel sparse reports: %zu/%zu %zu/%zu %zu/%zu %zu/%zu\n",
            reports[0].replacement_size,
            reports[0].reserved_size,
            reports[1].replacement_size,
            reports[1].reserved_size,
            reports[2].replacement_size,
            reports[2].reserved_size,
            reports[3].replacement_size,
            reports[3].reserved_size
        );
        if (!run_multi_panel_kernels(cfp_cmake_rdc_multi_panel_sparse_replacement_linked_cubin_data, input, multi_panel_sparse_ref)) goto done;
        if (!run_multi_panel_kernels(multi_panel_sparse_patched_cubin.data, input, multi_panel_sparse_patch)) goto done;
        if (!compare_outputs(
                "multi-panel sparse patched vs linked",
                multi_panel_sparse_ref,
                multi_panel_sparse_patch,
                4u * 32u * 128u)) goto done;
    }

    printf("cubin_function_patch_cmake_rdc_test PASS\n");
    ok = 1;

done:
    if (handle != NULL) cubin_function_patch_destroy(handle);
    free(handle_memory);
    free_blob(&simple_patched_cubin);
    free_blob(&poly_patched_cubin);
    free_blob(&switch_patched_cubin);
    free_blob(&panel_patched_cubin);
    free_blob(&multi_panel_patched_cubin);
    free_blob(&multi_panel_sparse_patched_cubin);
    free_blob(&failure_cubin);
    if (ctx_created) (void)cuDevicePrimaryCtxRelease(device);
    return ok ? 0 : 1;
}
