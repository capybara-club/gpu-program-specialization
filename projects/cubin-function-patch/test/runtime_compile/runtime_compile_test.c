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
#include <nvJitLink.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    void* data;
    size_t bytes;
} TestBlob;

extern const char cfp_test_patch_caller_cu_data[];
extern const unsigned int cfp_test_patch_caller_cu_size;
extern const char cfp_test_patch_reserved_cu_data[];
extern const unsigned int cfp_test_patch_reserved_cu_size;
extern const char cfp_test_patch_simple_cu_data[];
extern const unsigned int cfp_test_patch_simple_cu_size;
extern const char cfp_test_patch_poly_cu_data[];
extern const unsigned int cfp_test_patch_poly_cu_size;
extern const char cfp_test_patch_switch_reserved_cu_data[];
extern const unsigned int cfp_test_patch_switch_reserved_cu_size;
extern const char cfp_test_patch_switch_replacement_cu_data[];
extern const unsigned int cfp_test_patch_switch_replacement_cu_size;

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

static void
print_nvrtc_log(
    nvrtcProgram program
) {
    size_t log_size = 0u;
    char* log = NULL;
    if (nvrtcGetProgramLogSize(program, &log_size) != NVRTC_SUCCESS || log_size <= 1u) return;
    log = (char*)malloc(log_size);
    if (log == NULL) return;
    if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS && log[0] != '\0') {
        fprintf(stderr, "%s\n", log);
    }
    free(log);
}

static void
print_nvptx_log(
    nvPTXCompilerHandle compiler,
    int error_log
) {
    size_t log_size = 0u;
    char* log = NULL;
    if (error_log) {
        if (nvPTXCompilerGetErrorLogSize(compiler, &log_size) != NVPTXCOMPILE_SUCCESS) return;
    } else {
        if (nvPTXCompilerGetInfoLogSize(compiler, &log_size) != NVPTXCOMPILE_SUCCESS) return;
    }
    if (log_size <= 1u) return;
    log = (char*)malloc(log_size);
    if (log == NULL) return;
    if (error_log) {
        (void)nvPTXCompilerGetErrorLog(compiler, log);
    } else {
        (void)nvPTXCompilerGetInfoLog(compiler, log);
    }
    if (log[0] != '\0') fprintf(stderr, "%s\n", log);
    free(log);
}

static int
compile_cuda_to_ptx(
    const char* source,
    size_t source_bytes,
    const char* name,
    unsigned int sm_major,
    unsigned int sm_minor,
    TestBlob* ptx_out
) {
    nvrtcProgram program = NULL;
    nvrtcResult result;
    const char* options[5];
    char arch_option[64];
    size_t ptx_size = 0u;
    char* ptx = NULL;

    if (source == NULL || source_bytes == 0u || ptx_out == NULL) return 0;
    ptx_out->data = NULL;
    ptx_out->bytes = 0u;
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%u%u", sm_major, sm_minor);
    options[0] = arch_option;
    options[1] = "--std=c++17";
    options[2] = "--use_fast_math";
    options[3] = "--relocatable-device-code=true";
    options[4] = "--device-c";

    result = nvrtcCreateProgram(&program, source, name, 0, NULL, NULL);
    if (result != NVRTC_SUCCESS) return 0;
    result = nvrtcCompileProgram(program, 5, options);
    print_nvrtc_log(program);
    if (result != NVRTC_SUCCESS) {
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    if (nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS || ptx_size == 0u) {
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    ptx = (char*)malloc(ptx_size + 1u);
    if (ptx == NULL) {
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    if (nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) {
        free(ptx);
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    (void)nvrtcDestroyProgram(&program);
    ptx[ptx_size] = '\0';
    ptx_out->data = ptx;
    ptx_out->bytes = ptx_size;
    return 1;
}

static int
compile_ptx_to_rdc(
    const char* ptx,
    size_t ptx_bytes,
    unsigned int sm_major,
    unsigned int sm_minor,
    TestBlob* object_out
) {
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult result;
    const char* options[4];
    char arch_option[64];
    size_t image_size = 0u;
    void* image = NULL;

    if (ptx == NULL || ptx_bytes == 0u || object_out == NULL) return 0;
    object_out->data = NULL;
    object_out->bytes = 0u;
    snprintf(arch_option, sizeof(arch_option), "--gpu-name=sm_%u%u", sm_major, sm_minor);
    options[0] = arch_option;
    options[1] = "--compile-only";
    options[2] = "--opt-level=0";
    options[3] = "--allow-expensive-optimizations=false";

    result = nvPTXCompilerCreate(&compiler, ptx_bytes, ptx);
    if (result != NVPTXCOMPILE_SUCCESS) return 0;
    result = nvPTXCompilerCompile(compiler, 4, options);
    print_nvptx_log(compiler, 0);
    if (result != NVPTXCOMPILE_SUCCESS) {
        print_nvptx_log(compiler, 1);
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    if (nvPTXCompilerGetCompiledProgramSize(compiler, &image_size) != NVPTXCOMPILE_SUCCESS || image_size == 0u) {
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    image = malloc(image_size);
    if (image == NULL) {
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    if (nvPTXCompilerGetCompiledProgram(compiler, image) != NVPTXCOMPILE_SUCCESS) {
        free(image);
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    (void)nvPTXCompilerDestroy(&compiler);
    object_out->data = image;
    object_out->bytes = image_size;
    return 1;
}

static int
compile_cuda_to_rdc(
    const char* source,
    size_t source_bytes,
    const char* name,
    unsigned int sm_major,
    unsigned int sm_minor,
    TestBlob* object_out
) {
    TestBlob ptx = {0};
    int ok = 0;
    if (!compile_cuda_to_ptx(source, source_bytes, name, sm_major, sm_minor, &ptx)) return 0;
    if (!compile_ptx_to_rdc((const char*)ptx.data, ptx.bytes, sm_major, sm_minor, object_out)) goto done;
    ok = 1;

done:
    free_blob(&ptx);
    return ok;
}

static int
link_caller_and_site(
    const TestBlob* caller_object,
    const TestBlob* site_object,
    unsigned int sm_major,
    unsigned int sm_minor,
    TestBlob* cubin_out
) {
    nvJitLinkHandle link = NULL;
    nvJitLinkResult result;
    const char* options[3];
    char arch_option[64];
    size_t cubin_size = 0u;
    void* cubin = NULL;

    if (caller_object == NULL || site_object == NULL || cubin_out == NULL ||
        caller_object->data == NULL || site_object->data == NULL) {
        return 0;
    }
    cubin_out->data = NULL;
    cubin_out->bytes = 0u;
    snprintf(arch_option, sizeof(arch_option), "-arch=sm_%u%u", sm_major, sm_minor);
    options[0] = arch_option;
    options[1] = "-O0";
    options[2] = "-no-cache";

    result = nvJitLinkCreate(&link, 3, options);
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(link, NVJITLINK_INPUT_CUBIN, caller_object->data, caller_object->bytes, "caller_object");
    }
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(link, NVJITLINK_INPUT_CUBIN, site_object->data, site_object->bytes, "site_object");
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkComplete(link);
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubinSize(link, &cubin_size);
    if (result == NVJITLINK_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin == NULL) result = NVJITLINK_ERROR_INTERNAL;
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubin(link, cubin);
    (void)nvJitLinkDestroy(&link);
    if (result != NVJITLINK_SUCCESS || cubin == NULL || cubin_size == 0u) {
        free(cubin);
        fprintf(stderr, "nvJitLink failed: %d\n", (int)result);
        return 0;
    }
    cubin_out->data = cubin;
    cubin_out->bytes = cubin_size;
    return 1;
}

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
    TestBlob caller_object = {0};
    TestBlob reserved_object = {0};
    TestBlob simple_object = {0};
    TestBlob poly_object = {0};
    TestBlob switch_reserved_object = {0};
    TestBlob switch_replacement_object = {0};
    TestBlob reserved_cubin = {0};
    TestBlob simple_ref_cubin = {0};
    TestBlob poly_ref_cubin = {0};
    TestBlob switch_reserved_cubin = {0};
    TestBlob switch_ref_cubin = {0};
    TestBlob simple_patched_cubin = {0};
    TestBlob poly_patched_cubin = {0};
    TestBlob switch_patched_cubin = {0};
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
    size_t i;
    int ok = 0;

    for (i = 0u; i < count; ++i) {
        input[i] = (float)i * 0.03125f - 3.0f;
    }

    if (!CHECK_CUDA(cuInit(0))) goto done;
    if (!CHECK_CUDA(cuDeviceGet(&device, 0))) goto done;
    if (!CHECK_CUDA(cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))) goto done;
    if (!CHECK_CUDA(cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))) goto done;
    if (!CHECK_CUDA(cuDevicePrimaryCtxRetain(&context, device))) goto done;
    ctx_created = 1;
    if (!CHECK_CUDA(cuCtxSetCurrent(context))) goto done;
    printf("device sm_%d%d\n", sm_major, sm_minor);

    if (!compile_cuda_to_rdc(cfp_test_patch_caller_cu_data, cfp_test_patch_caller_cu_size - 1u, "patch_caller.cu", (unsigned int)sm_major, (unsigned int)sm_minor, &caller_object)) goto done;
    if (!compile_cuda_to_rdc(cfp_test_patch_reserved_cu_data, cfp_test_patch_reserved_cu_size - 1u, "patch_reserved.cu", (unsigned int)sm_major, (unsigned int)sm_minor, &reserved_object)) goto done;
    if (!compile_cuda_to_rdc(cfp_test_patch_simple_cu_data, cfp_test_patch_simple_cu_size - 1u, "patch_simple.cu", (unsigned int)sm_major, (unsigned int)sm_minor, &simple_object)) goto done;
    if (!compile_cuda_to_rdc(cfp_test_patch_poly_cu_data, cfp_test_patch_poly_cu_size - 1u, "patch_poly.cu", (unsigned int)sm_major, (unsigned int)sm_minor, &poly_object)) goto done;
    if (!compile_cuda_to_rdc(cfp_test_patch_switch_reserved_cu_data, cfp_test_patch_switch_reserved_cu_size - 1u, "patch_switch_reserved.cu", (unsigned int)sm_major, (unsigned int)sm_minor, &switch_reserved_object)) goto done;
    if (!compile_cuda_to_rdc(cfp_test_patch_switch_replacement_cu_data, cfp_test_patch_switch_replacement_cu_size - 1u, "patch_switch_replacement.cu", (unsigned int)sm_major, (unsigned int)sm_minor, &switch_replacement_object)) goto done;

    if (!link_caller_and_site(&caller_object, &reserved_object, (unsigned int)sm_major, (unsigned int)sm_minor, &reserved_cubin)) goto done;
    if (!link_caller_and_site(&caller_object, &simple_object, (unsigned int)sm_major, (unsigned int)sm_minor, &simple_ref_cubin)) goto done;
    if (!link_caller_and_site(&caller_object, &poly_object, (unsigned int)sm_major, (unsigned int)sm_minor, &poly_ref_cubin)) goto done;
    if (!link_caller_and_site(&caller_object, &switch_reserved_object, (unsigned int)sm_major, (unsigned int)sm_minor, &switch_reserved_cubin)) goto done;
    if (!link_caller_and_site(&caller_object, &switch_replacement_object, (unsigned int)sm_major, (unsigned int)sm_minor, &switch_ref_cubin)) goto done;

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
            cubin_function_patch_create(reserved_cubin.data, reserved_cubin.bytes, symbols, 1u, handle_memory, handle_memory_size, &handle),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "create")) {
        goto done;
    }
    if (!expect_patch_result(cubin_function_patch_output_size(handle, &output_size), CUBIN_FUNCTION_PATCH_SUCCESS, "output size")) goto done;
    simple_patched_cubin.data = malloc(output_size);
    poly_patched_cubin.data = malloc(output_size);
    switch_patched_cubin.data = malloc(switch_reserved_cubin.bytes);
    failure_cubin.data = malloc(output_size);
    if (simple_patched_cubin.data == NULL || poly_patched_cubin.data == NULL ||
        switch_patched_cubin.data == NULL || failure_cubin.data == NULL) goto done;

    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "patch_site",
                simple_object.data,
                simple_object.bytes,
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
                    simple_object.data,
                    simple_object.bytes,
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
                simple_object.data,
                simple_object.bytes,
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
                poly_object.data,
                poly_object.bytes,
                poly_patched_cubin.data,
                output_size,
                &poly_patched_cubin.bytes,
                NULL,
                0u),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "apply poly")) {
        goto done;
    }

    if (!run_patch_kernel(simple_ref_cubin.data, input, simple_ref, count)) goto done;
    if (!run_patch_kernel(simple_patched_cubin.data, input, simple_patch, count)) goto done;
    if (!compare_outputs("simple patched vs linked", simple_ref, simple_patch, count)) goto done;
    if (!run_patch_kernel(poly_ref_cubin.data, input, poly_ref, count)) goto done;
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
            cubin_function_patch_create(switch_reserved_cubin.data, switch_reserved_cubin.bytes, symbols, 1u, handle_memory, handle_memory_size, &handle),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "switch create")) {
        goto done;
    }
    if (!expect_patch_result(
            cubin_function_patch_apply_one(
                handle,
                "patch_site",
                switch_replacement_object.data,
                switch_replacement_object.bytes,
                switch_patched_cubin.data,
                switch_reserved_cubin.bytes,
                &switch_patched_cubin.bytes,
                NULL),
            CUBIN_FUNCTION_PATCH_SUCCESS,
            "apply switch")) {
        goto done;
    }
    if (!run_patch_kernel(switch_ref_cubin.data, input, switch_ref, count)) goto done;
    if (!run_patch_kernel(switch_patched_cubin.data, input, switch_patch, count)) goto done;
    if (!compare_outputs("switch patched vs linked", switch_ref, switch_patch, count)) goto done;

    printf("cubin_function_patch_driver_test PASS\n");
    ok = 1;

done:
    if (handle != NULL) cubin_function_patch_destroy(handle);
    free(handle_memory);
    free_blob(&caller_object);
    free_blob(&reserved_object);
    free_blob(&simple_object);
    free_blob(&poly_object);
    free_blob(&switch_reserved_object);
    free_blob(&switch_replacement_object);
    free_blob(&reserved_cubin);
    free_blob(&simple_ref_cubin);
    free_blob(&poly_ref_cubin);
    free_blob(&switch_reserved_cubin);
    free_blob(&switch_ref_cubin);
    free_blob(&simple_patched_cubin);
    free_blob(&poly_patched_cubin);
    free_blob(&switch_patched_cubin);
    free_blob(&failure_cubin);
    if (ctx_created) (void)cuDevicePrimaryCtxRelease(device);
    return ok ? 0 : 1;
}
