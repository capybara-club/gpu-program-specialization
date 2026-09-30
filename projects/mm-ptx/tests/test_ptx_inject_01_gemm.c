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

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_helper.h>

#include <cuda.h>
#include <cuda_helper.h>
#include <nvptx_helper.h>

#include <check_result_helper.h>
#include <mma_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

INCTXT(annotated_ptx, PTX_KERNEL);

#define STUB_BUFFER_SIZE 1000000ull

static const int k_m = 128;
static const int k_n = 128;
static const int k_k = 8;
static const int k_lda = 128;
static const int k_ldb = 128;
static const int k_ldc = 128;

static const float k_max_abs_tol = 1e-3f;

static
bool
has_cuda_device(void) {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

static
void
fill_inputs(float* h_a, float* h_b) {
    const int a_elems = k_m * k_k;
    const int b_elems = k_n * k_k;
    for (int idx = 0; idx < a_elems; ++idx) {
        h_a[idx] = (float)((idx % 7) - 3) * 0.25f;
    }
    for (int idx = 0; idx < b_elems; ++idx) {
        h_b[idx] = (float)(((idx + 5) % 11) - 5) * 0.2f;
    }
}

static
float
max_abs_diff(const float* a, const float* b, size_t count) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

static
void
run_custom_mma(
    int M, int N, int K,
    CUdeviceptr d_a, int lda,
    CUdeviceptr d_b, int ldb,
    CUdeviceptr d_c, int ldc,
    int device_compute_capability_major,
    int device_compute_capability_minor,
    const char* rendered_ptx,
    size_t rendered_ptx_num_bytes
) {
    void* sass =
        nvptx_compile(
            device_compute_capability_major,
            device_compute_capability_minor,
            rendered_ptx,
            rendered_ptx_num_bytes,
            NULL,
            false
        );

    CUmodule cu_module;
    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    free(sass);

    CUfunction cu_function;
    cuCheck( cuModuleGetFunction(&cu_function, cu_module, "gemm_nt") );

    static const unsigned int block_dim = 256;

    void* args[] = {
        (void*)&M,
        (void*)&N,
        (void*)&K,
        (void*)&d_a,
        (void*)&lda,
        (void*)&d_b,
        (void*)&ldb,
        (void*)&d_c,
        (void*)&ldc
    };

    cuCheck(
        cuLaunchKernel(
            cu_function,
            1, 1, 1,
            block_dim, 1, 1,
            0, 0,
            args,
            NULL
        )
    );

    cuCheck( cuCtxSynchronize() );
    cuCheck( cuModuleUnload(cu_module) );
}

int
main() {
    if (!has_cuda_device()) {
        fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }

    cuCheck( cuInit(0) );
    CUdevice device;
    cuCheck( cuDeviceGet(&device, 0) );

    int device_compute_capability_major = 0;
    int device_compute_capability_minor = 0;
    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    if (device_compute_capability_major < 8) {
        fprintf(stderr, "SKIP: requires sm80+ device\n");
        return 77;
    }

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    size_t num_injects = 0;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );
    ASSERT(num_injects == 2);

    size_t mma_func_idx = 0;
    size_t epilogue_func_idx = 0;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "mma", &mma_func_idx, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "epilogue", &epilogue_func_idx, NULL, NULL) );

    const char* mma_register_name_v_a = NULL;
    const char* mma_register_name_v_b = NULL;
    const char* mma_register_name_v_c = NULL;

    const char* epilogue_register_name_v_c_in = NULL;
    const char* epilogue_register_name_v_c_out = NULL;

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, mma_func_idx, "v_a", NULL, &mma_register_name_v_a, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, mma_func_idx, "v_b", NULL, &mma_register_name_v_b, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, mma_func_idx, "v_c", NULL, &mma_register_name_v_c, NULL, NULL, NULL) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c_in", NULL, &epilogue_register_name_v_c_in, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c_out", NULL, &epilogue_register_name_v_c_out, NULL, NULL, NULL) );

    char* mma_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    char* epilogue_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    ASSERT(mma_stub_buffer != NULL);
    ASSERT(epilogue_stub_buffer != NULL);

    snprintf(mma_stub_buffer, STUB_BUFFER_SIZE,
        "\tfma.rn.ftz.f32 %%%3$s, %%%2$s, %%%1$s, %%%3$s;",
        mma_register_name_v_a,
        mma_register_name_v_b,
        mma_register_name_v_c
    );

    snprintf(epilogue_stub_buffer, STUB_BUFFER_SIZE,
        "\tmov.f32 %%%2$s, %%%1$s;",
        epilogue_register_name_v_c_in,
        epilogue_register_name_v_c_out
    );

    const char** ptx_stubs = (const char**)calloc(num_injects, sizeof(*ptx_stubs));
    ASSERT(ptx_stubs != NULL);
    ptx_stubs[mma_func_idx] = mma_stub_buffer;
    ptx_stubs[epilogue_func_idx] = epilogue_stub_buffer;

    size_t num_bytes_written = 0;
    char* rendered_ptx = NULL;

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, device) );

    const size_t a_elems = (size_t)k_m * (size_t)k_k;
    const size_t b_elems = (size_t)k_n * (size_t)k_k;
    const size_t c_elems = (size_t)k_m * (size_t)k_n;

    float* h_a = (float*)malloc(a_elems * sizeof(float));
    float* h_b = (float*)malloc(b_elems * sizeof(float));
    float* h_c = (float*)malloc(c_elems * sizeof(float));
    float* h_c_gold = (float*)malloc(c_elems * sizeof(float));

    ASSERT(h_a != NULL);
    ASSERT(h_b != NULL);
    ASSERT(h_c != NULL);
    ASSERT(h_c_gold != NULL);

    fill_inputs(h_a, h_b);

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_c;
    cuCheck( cuMemAlloc(&d_a, a_elems * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_b, b_elems * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_c, c_elems * sizeof(float)) );

    cuCheck( cuMemcpyHtoD(d_a, h_a, a_elems * sizeof(float)) );
    cuCheck( cuMemcpyHtoD(d_b, h_b, b_elems * sizeof(float)) );

    float max_diff = 0.0f;

    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, num_injects, &num_bytes_written);
    run_custom_mma(
        k_m, k_n, k_k,
        d_a, k_lda,
        d_b, k_ldb,
        d_c, k_ldc,
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, c_elems * sizeof(float)) );

    gemm_gold(
        k_m, k_n, k_k,
        h_a, k_lda,
        h_b, k_ldb,
        h_c_gold, k_ldc
    );

    max_diff = max_abs_diff(h_c, h_c_gold, c_elems);
    ASSERT(max_diff < k_max_abs_tol);

    snprintf(mma_stub_buffer, STUB_BUFFER_SIZE,
        "\tsub.ftz.f32 %%%1$s, %%%2$s, %%%1$s;\n"
        "\tabs.ftz.f32 %%%1$s, %%%1$s;\n"
        "\tadd.ftz.f32 %%%3$s, %%%3$s, %%%1$s;",
        mma_register_name_v_a,
        mma_register_name_v_b,
        mma_register_name_v_c
    );

    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, num_injects, &num_bytes_written);
    run_custom_mma(
        k_m, k_n, k_k,
        d_a, k_lda,
        d_b, k_ldb,
        d_c, k_ldc,
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, c_elems * sizeof(float)) );

    l1_gold(
        k_m, k_n, k_k,
        h_a, k_lda,
        h_b, k_ldb,
        h_c_gold, k_ldc
    );

    max_diff = max_abs_diff(h_c, h_c_gold, c_elems);
    ASSERT(max_diff < k_max_abs_tol);

    snprintf(mma_stub_buffer, STUB_BUFFER_SIZE,
        "\tsub.ftz.f32 %%%1$s, %%%2$s, %%%1$s;\n"
        "\tmul.ftz.f32 %%%1$s, %%%1$s, %%%1$s;\n"
        "\tadd.ftz.f32 %%%3$s, %%%3$s, %%%1$s;",
        mma_register_name_v_a,
        mma_register_name_v_b,
        mma_register_name_v_c
    );

    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, num_injects, &num_bytes_written);
    run_custom_mma(
        k_m, k_n, k_k,
        d_a, k_lda,
        d_b, k_ldb,
        d_c, k_ldc,
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, c_elems * sizeof(float)) );

    l2_gold(
        k_m, k_n, k_k,
        h_a, k_lda,
        h_b, k_ldb,
        h_c_gold, k_ldc
    );

    max_diff = max_abs_diff(h_c, h_c_gold, c_elems);
    ASSERT(max_diff < k_max_abs_tol);

    free(mma_stub_buffer);
    free(epilogue_stub_buffer);
    free(ptx_stubs);
    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    cuCheck( cuMemFree(d_c) );
    cuCheck( cuMemFree(d_b) );
    cuCheck( cuMemFree(d_a) );

    cuCheck( cuCtxDestroy(cu_context) );

    free(h_c_gold);
    free(h_c);
    free(h_b);
    free(h_a);

    return 0;
}
