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
#define CUSR_AST_SASS_IMPLEMENTATION
#define CUSR_SASS_INSPECT_IMPLEMENTATION
#define CUSR_AST_SASS_PATCH_IMPLEMENTATION

#include <cusr_ast_sass_patch.h>
#include <cusr_ast_sass_cpu.h>
#include <cusr_settings.h>
#include <cusr_tile_static_mse_nvrtc.h>

#include <cuda.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define CUSR_TEST_ROWS 257u
#define CUSR_TEST_COLUMNS 8u
#define CUSR_TEST_SETTINGS 3u
#define CUSR_TEST_PROGRAMS 9u
#define CUSR_TEST_ACTIVE_PROGRAMS 4u
#define CUSR_TEST_CAPACITY_COMPARE_PROGRAMS 8u
#define CUSR_TEST_PROGRAM_STRIDE 16u
#define CUSR_TEST_FIRST_MARKER 0x7fc0ffeeu
#define CUSR_TEST_THREADS 128u
#define CUSR_TEST_LEAF_WORDS_STRIDE 8u
#define CUSR_TEST_ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CUSR_TEST_ERROR_RET(ans) do { int cusr_test_result = (ans); return cusr_test_result; } while (0)
#define CUSR_TEST_CHECK_RET(ans) do { if (!(ans)) { CUSR_TEST_ERROR_RET(0); } } while (0)

typedef enum CusrTestKernelFamily {
    CUSR_TEST_KERNEL_TILE_STATIC_MSE_128 = 0,
    CUSR_TEST_KERNEL_TILE_STATIC_MSE_256 = 1,
    CUSR_TEST_KERNEL_TILE_STATIC_MSE_64 = 2
} CusrTestKernelFamily;

typedef struct CusrTestCubin {
    unsigned char* data;
    size_t size;
} CusrTestCubin;

typedef enum CusrTestRoutine {
    CUSR_TEST_ROUTINE_SAFE_DIV = 0,
    CUSR_TEST_ROUTINE_SAFE_SQRT = 1,
    CUSR_TEST_ROUTINE_SAFE_RSQRT = 2,
    CUSR_TEST_ROUTINE_COUNT = 3
} CusrTestRoutine;

static uint32_t cusr_test_rng_state = 0x4d595df4u;

static const CusrAstInstruction cusr_test_routine_safe_div[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(0x3e800000u),
    cusr_ast_encode_add,
    cusr_ast_encode_div,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_routine_safe_sqrt[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(0x3e800000u),
    cusr_ast_encode_add,
    cusr_ast_encode_sqrt,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_routine_safe_rsqrt[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_abs,
    cusr_ast_encode_constant_bits(0x3f400000u),
    cusr_ast_encode_add,
    cusr_ast_encode_rsqrt,
    cusr_ast_encode_return
};

static const CusrAstInstruction* const cusr_test_routines[] = {
    cusr_test_routine_safe_div,
    cusr_test_routine_safe_sqrt,
    cusr_test_routine_safe_rsqrt
};

static const CusrAstInstruction cusr_test_program_sin_x[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_sin,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_cos_y[] = {
    cusr_ast_encode_input(1u),
    cusr_ast_encode_cos,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_safe_div[] = {
    cusr_ast_encode_input(2u),
    cusr_ast_encode_input(3u),
    cusr_ast_encode_routine(CUSR_TEST_ROUTINE_SAFE_DIV, 2u),
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_sin_cos_mul[] = {
    cusr_ast_encode_input(4u),
    cusr_ast_encode_sin,
    cusr_ast_encode_input(5u),
    cusr_ast_encode_cos,
    cusr_ast_encode_mul,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_sqrt_abs[] = {
    cusr_ast_encode_input(6u),
    cusr_ast_encode_routine(CUSR_TEST_ROUTINE_SAFE_SQRT, 1u),
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_rsqrt_abs[] = {
    cusr_ast_encode_input(7u),
    cusr_ast_encode_routine(CUSR_TEST_ROUTINE_SAFE_RSQRT, 1u),
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_minmax[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_input(1u),
    cusr_ast_encode_max,
    cusr_ast_encode_input(2u),
    cusr_ast_encode_min,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_fma[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_input(1u),
    cusr_ast_encode_input(2u),
    cusr_ast_encode_fma,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_test_program_tanh[] = {
    cusr_ast_encode_input(3u),
    cusr_ast_encode_tanh,
    cusr_ast_encode_return
};

static const CusrAstInstruction* const cusr_test_programs[] = {
    cusr_test_program_sin_x,
    cusr_test_program_cos_y,
    cusr_test_program_safe_div,
    cusr_test_program_sin_cos_mul,
    cusr_test_program_sqrt_abs,
    cusr_test_program_rsqrt_abs,
    cusr_test_program_minmax,
    cusr_test_program_fma,
    cusr_test_program_tanh
};

static uint32_t
cusr_test_rand_u32(void)
{
    cusr_test_rng_state = cusr_test_rng_state * 1664525u + 1013904223u;
    return cusr_test_rng_state;
}

static float
cusr_test_rand_unit_f32(void)
{
    return (float)(cusr_test_rand_u32() >> 8) * (1.0f / 16777216.0f);
}

static uint32_t
cusr_test_f32_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int
cusr_test_check_cuda(CUresult result, const char* label)
{
    if (result != CUDA_SUCCESS) {
        const char* name = "unknown";
        const char* description = "unknown";
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);
        fprintf(stderr, "%s failed: %s (%s)\n", label, name, description);
        return 0;
    }

    return 1;
}

static int
cusr_test_make_dir(const char* path)
{
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return 1;
    }

    fprintf(stderr, "failed to create output directory %s\n", path);
    return 0;
}

static int
cusr_test_path(char* buffer, size_t buffer_size, const char* output_dir, const char* filename)
{
    int written = snprintf(buffer, buffer_size, "%s/%s", output_dir, filename);
    return written >= 0 && (size_t)written < buffer_size;
}

static int
cusr_test_write_file(const char* path, const void* data, size_t size)
{
    FILE* file = fopen(path, "wb");
    int ok = 1;

    if (file == NULL) {
        fprintf(stderr, "failed to open %s for write\n", path);
        return 0;
    }

    if (fwrite(data, 1u, size, file) != size) {
        ok = 0;
    }

    if (fclose(file) != 0) {
        ok = 0;
    }

    if (!ok) {
        fprintf(stderr, "failed to write %s\n", path);
    }

    return ok;
}

static int
cusr_test_write_named_file(const char* output_dir, const char* filename, const void* data, size_t size)
{
    char path[1024];
    CUSR_TEST_CHECK_RET(cusr_test_path(path, sizeof(path), output_dir, filename));
    CUSR_TEST_CHECK_RET(cusr_test_write_file(path, data, size));
    CUSR_TEST_ERROR_RET(1);
}

static int
cusr_test_write_meta(const char* output_dir)
{
    char path[1024];
    char json[2048];
    int written;

    written = snprintf(
        json,
        sizeof(json),
        "{\n"
        "  \"num_rows\": %u,\n"
        "  \"num_columns\": %u,\n"
        "  \"num_settings\": %u,\n"
        "  \"num_programs\": %u,\n"
        "  \"active_programs\": %u,\n"
        "  \"program_stride\": %u,\n"
        "  \"leaf_words_stride\": %u,\n"
        "  \"programs\": [\"sin_x\", \"cos_y\", \"safe_div\", \"sin_cos_mul\", \"sqrt_abs\", \"rsqrt_abs\", \"minmax\", \"fma\", \"tanh\"]\n"
        "}\n",
        CUSR_TEST_ROWS,
        CUSR_TEST_COLUMNS,
        CUSR_TEST_SETTINGS,
        CUSR_TEST_PROGRAMS,
        CUSR_TEST_ACTIVE_PROGRAMS,
        CUSR_TEST_PROGRAM_STRIDE,
        CUSR_TEST_LEAF_WORDS_STRIDE
    );

    CUSR_TEST_CHECK_RET(written > 0 && (size_t)written < sizeof(json));
    CUSR_TEST_CHECK_RET(cusr_test_path(path, sizeof(path), output_dir, "meta.json"));
    CUSR_TEST_CHECK_RET(cusr_test_write_file(path, json, (size_t)written));
    CUSR_TEST_ERROR_RET(1);
}

static void
cusr_test_fill_data(float* x, float* target)
{
    uint32_t column;

    cusr_test_rng_state = 0x4d595df4u;

    for (column = 0u; column < CUSR_TEST_COLUMNS; ++column) {
        uint32_t row;

        for (row = 0u; row < CUSR_TEST_ROWS; ++row) {
            const float unit = cusr_test_rand_unit_f32();
            x[(size_t)column * CUSR_TEST_ROWS + row] = 0.30f + 0.90f * unit + 0.03f * (float)column;
        }
    }

    for (column = 0u; column < CUSR_TEST_ROWS; ++column) {
        const float x0 = x[column];
        const float x1 = x[(size_t)1u * CUSR_TEST_ROWS + column];
        target[column] = 0.50f + 0.25f * x0 + 0.125f * x1;
    }
}

static void
cusr_test_fill_settings(CusrSettingF32* settings, uint8_t* leaf_masks, uint32_t* leaf_words)
{
    static const uint8_t masks[CUSR_TEST_SETTINGS] = { 0xffu, 0x55u, 0xaau };
    uint32_t setting_idx;

    for (setting_idx = 0u; setting_idx < CUSR_TEST_SETTINGS; ++setting_idx) {
        uint32_t input_idx;

        settings[setting_idx].column_mask = masks[setting_idx];

        for (input_idx = 0u; input_idx < CUSR_NUM_INPUTS; ++input_idx) {
            const uint32_t column = (setting_idx * 2u + input_idx * 3u) & 7u;
            settings[setting_idx].column_indices[input_idx] = column;
            settings[setting_idx].constants[input_idx] = 1.0f + 0.125f * (float)input_idx;
            leaf_words[(size_t)setting_idx * CUSR_TEST_LEAF_WORDS_STRIDE + input_idx] =
                (masks[setting_idx] & (uint8_t)(1u << input_idx)) != 0u
                    ? column
                    : cusr_test_f32_bits(settings[setting_idx].constants[input_idx]);
        }
        leaf_masks[setting_idx] = masks[setting_idx];
    }
}

static int
cusr_test_compare_f32(const char* label, const float* actual, const float* expected, size_t count, float atol, float rtol)
{
    float max_abs = -1.0f;
    float max_rel = 0.0f;
    size_t worst_idx = 0u;
    size_t i;

    for (i = 0u; i < count; ++i) {
        const float actual_value = actual[i];
        const float expected_value = expected[i];

        if (!isfinite(actual_value) || !isfinite(expected_value)) {
            fprintf(
                stderr,
                "%s cpu compare failed: non-finite value at %zu actual=%g expected=%g\n",
                label,
                i,
                actual_value,
                expected_value
            );
            CUSR_TEST_ERROR_RET(0);
        }

        const float abs_error = fabsf(actual_value - expected_value);
        const float rel_error = abs_error / fmaxf(fabsf(expected_value), 1.0f);

        if (abs_error > max_abs) {
            max_abs = abs_error;
            max_rel = rel_error;
            worst_idx = i;
        }
    }

    if (max_abs > atol && max_rel > rtol) {
        fprintf(
            stderr,
            "%s cpu compare failed: max_abs=%g max_rel=%g worst=%zu actual=%g expected=%g\n",
            label,
            max_abs,
            max_rel,
            worst_idx,
            actual[worst_idx],
            expected[worst_idx]
        );
        CUSR_TEST_ERROR_RET(0);
    }

    printf("%s cpu compare ok max_abs=%g max_rel=%g worst=%zu\n", label, max_abs, max_rel, worst_idx);
    CUSR_TEST_ERROR_RET(1);
}

static void
cusr_test_fill_cpu_tile_static_mse_output(const float* cpu_eval_output, const float* target, float* cpu_mse_output)
{
    uint32_t program_idx;

    for (program_idx = 0u; program_idx < CUSR_TEST_PROGRAMS; ++program_idx) {
        uint32_t setting_idx;

        for (setting_idx = 0u; setting_idx < CUSR_TEST_SETTINGS; ++setting_idx) {
            float sse = 0.0f;
            uint32_t row;
            const size_t mse_index = (size_t)program_idx * CUSR_TEST_SETTINGS + setting_idx;

            for (row = 0u; row < CUSR_TEST_ROWS; ++row) {
                const size_t eval_index =
                    ((size_t)program_idx * CUSR_TEST_SETTINGS + setting_idx) * CUSR_TEST_ROWS + row;
                const float error = cpu_eval_output[eval_index] - target[row];
                sse = error * error + sse;
            }

            cpu_mse_output[mse_index] = sse / (float)CUSR_TEST_ROWS;
        }
    }
}

static int
cusr_test_device_arch(uint32_t* capability_major_ret, uint32_t* capability_minor_ret, CUcontext* context_ret)
{
    CUdevice device = 0;
    int major = 0;
    int minor = 0;

    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuInit(0u), "cuInit"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor"));
#if CUDA_VERSION >= 13000
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuCtxCreate(context_ret, NULL, 0u, device), "cuCtxCreate"));
#else
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuCtxCreate(context_ret, 0u, device), "cuCtxCreate"));
#endif

    *capability_major_ret = (uint32_t)major;
    *capability_minor_ret = (uint32_t)minor;
    CUSR_TEST_ERROR_RET(1);
}

static void
cusr_test_print_nvrtc_log(const CusrTileStaticMseNvrtcHandle* handle)
{
    size_t log_size = 0u;
    char* log;

    if (handle == NULL ||
        cusr_tile_static_mse_nvrtc_log_size(handle, &log_size) != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS ||
        log_size <= 1u) {
        return;
    }

    log = (char*)malloc(log_size);
    if (log != NULL) {
        if (cusr_tile_static_mse_nvrtc_get_log(handle, log, log_size) == CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
            fprintf(stderr, "%s\n", log);
        }
        free(log);
    }
}

static int
cusr_test_compile_template_impl(
    CusrTestKernelFamily family,
    size_t ast_capacity,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrTestCubin* cubin_ret,
    CusrTileStaticMseNvrtcHandle** handle_ret)
{
    const uint32_t tile_rows = family == CUSR_TEST_KERNEL_TILE_STATIC_MSE_64
        ? 64u
        : (family == CUSR_TEST_KERNEL_TILE_STATIC_MSE_256 ? 256u : 128u);
    CusrTileStaticMseNvrtcResult result;
    size_t cubin_size = 0u;
    unsigned char* cubin;

    result = cusr_tile_static_mse_nvrtc_create(
        1u,
        ast_capacity,
        tile_rows,
        CUSR_TEST_THREADS,
        capability_major,
        capability_minor,
        handle_ret);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
        cusr_test_print_nvrtc_log(*handle_ret);
        fprintf(stderr, "cusr_tile_static_mse_nvrtc_create failed: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_TEST_ERROR_RET(0);
    }

    result = cusr_tile_static_mse_nvrtc_cubin_size(*handle_ret, &cubin_size);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_mse_nvrtc_cubin_size failed: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_TEST_ERROR_RET(0);
    }

    cubin = (unsigned char*)malloc(cubin_size);
    CUSR_TEST_CHECK_RET(cubin != NULL);
    cubin_ret->data = cubin;
    cubin_ret->size = cubin_size;

    result = cusr_tile_static_mse_nvrtc_get_cubin(*handle_ret, cubin, cubin_size - 1u);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_ERROR_INSUFFICIENT_BUFFER) {
        fprintf(stderr, "undersized cubin buffer returned: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_TEST_ERROR_RET(0);
    }

    result = cusr_tile_static_mse_nvrtc_get_cubin(*handle_ret, cubin, cubin_size);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_mse_nvrtc_get_cubin failed: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_TEST_ERROR_RET(0);
    }

    CUSR_TEST_ERROR_RET(1);
}

static int
cusr_test_compile_template(
    CusrTestKernelFamily family,
    size_t ast_capacity,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrTestCubin* cubin_ret)
{
    CusrTileStaticMseNvrtcHandle* handle = NULL;
    int ok;

    cubin_ret->data = NULL;
    cubin_ret->size = 0u;
    ok = cusr_test_compile_template_impl(family, ast_capacity, capability_major, capability_minor, cubin_ret, &handle);
    cusr_tile_static_mse_nvrtc_destroy(handle);
    if (!ok) {
        free(cubin_ret->data);
        cubin_ret->data = NULL;
        cubin_ret->size = 0u;
    }
    return ok;
}

static int
cusr_test_patch_cubin_impl(CusrTestKernelFamily family, size_t ast_capacity, size_t num_programs, CusrTestCubin* cubin, void** workspace_ret)
{
    static const char* const function_names[] = { "cusr_tile_static_mse_f32_000" };
    size_t workspace_size = 0u;
    CusrSassInspectHandle inspect;
    CusrSassInspectResult inspect_result;
    CusrAstSassPatchResult patch_result;
    uint32_t capability_major = 0u;
    uint32_t capability_minor = 0u;

    (void)family;
    inspect_result = cusr_sass_inspect_workspace_size(
        cubin->data,
        cubin->size,
        function_names,
        1u,
        ast_capacity,
        CUSR_TEST_FIRST_MARKER,
        1u,
        &workspace_size
    );

    if (inspect_result != CUSR_SASS_INSPECT_SUCCESS) {
        fprintf(stderr, "inspect workspace failed: %s\n", cusr_sass_inspect_result_to_string(inspect_result));
        CUSR_TEST_ERROR_RET(0);
    }

    *workspace_ret = malloc(workspace_size == 0u ? 1u : workspace_size);
    CUSR_TEST_CHECK_RET(*workspace_ret != NULL);

    inspect_result = cusr_sass_inspect(
        cubin->data,
        cubin->size,
        function_names,
        1u,
        ast_capacity,
        CUSR_TEST_FIRST_MARKER,
        1u,
        *workspace_ret,
        workspace_size,
        &inspect
    );

    if (inspect_result != CUSR_SASS_INSPECT_SUCCESS) {
        fprintf(stderr, "inspect failed: %s\n", cusr_sass_inspect_result_to_string(inspect_result));
        CUSR_TEST_ERROR_RET(0);
    }

    capability_major = inspect.sass_arch / 10u;
    capability_minor = inspect.sass_arch % 10u;

    patch_result = cusr_ast_sass_patch_cubin(
        &inspect,
        capability_major,
        capability_minor,
        CUSR_AST_SASS_PATCH_EPILOGUE_SSE,
        cusr_test_routines,
        CUSR_TEST_ARRAY_COUNT(cusr_test_routines),
        cusr_test_programs,
        num_programs,
        cubin->data,
        cubin->size,
        NULL
    );

    if (patch_result != CUSR_AST_SASS_PATCH_SUCCESS) {
        fprintf(stderr, "patch failed: %s\n", cusr_ast_sass_patch_result_to_string(patch_result));
        CUSR_TEST_ERROR_RET(0);
    }

    CUSR_TEST_ERROR_RET(1);
}

static int
cusr_test_patch_cubin(CusrTestKernelFamily family, size_t ast_capacity, size_t num_programs, CusrTestCubin* cubin)
{
    void* workspace = NULL;
    int ok = cusr_test_patch_cubin_impl(family, ast_capacity, num_programs, cubin, &workspace);

    free(workspace);
    return ok;
}

static int
cusr_test_build_patched_cubin(
    CusrTestKernelFamily family,
    size_t ast_capacity,
    size_t num_programs,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrTestCubin* cubin_ret)
{
    int ok = cusr_test_compile_template(family, ast_capacity, capability_major, capability_minor, cubin_ret);

    if (ok) {
        ok = cusr_test_patch_cubin(family, ast_capacity, num_programs, cubin_ret);
    }

    return ok;
}

static int
cusr_test_run_tile_static_mse_impl(
    const CusrTestCubin* cubin,
    size_t tile_rows,
    uint32_t num_asts,
    const float* x,
    const float* target,
    const uint8_t* leaf_masks,
    const uint32_t* leaf_words,
    float* output,
    CUmodule* module_ret,
    CUdeviceptr* d_x_ret,
    CUdeviceptr* d_target_ret,
    CUdeviceptr* d_masks_ret,
    CUdeviceptr* d_words_ret,
    CUdeviceptr* d_output_ret)
{
    CUfunction kernel = NULL;
    size_t num_rows = CUSR_TEST_ROWS;
    uint32_t num_columns = CUSR_TEST_COLUMNS;
    size_t leading_dim = CUSR_TEST_ROWS;
    size_t leaf_words_stride = CUSR_TEST_LEAF_WORDS_STRIDE;
    uint32_t num_settings = CUSR_TEST_SETTINGS;
    unsigned int blocks = (unsigned int)((CUSR_TEST_ROWS + tile_rows - 1u) / tile_rows);
    uint32_t shared_stride = num_columns | 1u;
    unsigned int shared_bytes =
        (unsigned int)(tile_rows * shared_stride + tile_rows) * (unsigned int)sizeof(float);
    size_t x_bytes = (size_t)CUSR_TEST_ROWS * CUSR_TEST_COLUMNS * sizeof(float);
    size_t target_bytes = (size_t)CUSR_TEST_ROWS * sizeof(float);
    size_t mask_bytes = (size_t)CUSR_TEST_SETTINGS * sizeof(uint8_t);
    size_t words_bytes = (size_t)CUSR_TEST_SETTINGS * CUSR_TEST_LEAF_WORDS_STRIDE * sizeof(uint32_t);
    size_t output_count = (size_t)num_asts * CUSR_TEST_SETTINGS;
    size_t output_bytes = output_count * sizeof(float);
    void* args[] = {
        d_x_ret,
        d_target_ret,
        &num_rows,
        &num_columns,
        &leading_dim,
        d_masks_ret,
        d_words_ret,
        &leaf_words_stride,
        &num_settings,
        &num_asts,
        d_output_ret
    };
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuModuleLoadData(module_ret, cubin->data), "cuModuleLoadData tile_static_mse"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuModuleGetFunction(&kernel, *module_ret, "cusr_tile_static_mse_f32_000"), "cuModuleGetFunction tile_static_mse"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemAlloc(d_x_ret, x_bytes), "cuMemAlloc tile_static_mse x"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemAlloc(d_target_ret, target_bytes), "cuMemAlloc tile_static_mse target"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemAlloc(d_masks_ret, mask_bytes), "cuMemAlloc tile_static_mse masks"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemAlloc(d_words_ret, words_bytes), "cuMemAlloc tile_static_mse words"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemAlloc(d_output_ret, output_bytes), "cuMemAlloc tile_static_mse output"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemcpyHtoD(*d_x_ret, x, x_bytes), "cuMemcpyHtoD tile_static_mse x"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemcpyHtoD(*d_target_ret, target, target_bytes), "cuMemcpyHtoD tile_static_mse target"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemcpyHtoD(*d_masks_ret, leaf_masks, mask_bytes), "cuMemcpyHtoD tile_static_mse masks"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemcpyHtoD(*d_words_ret, leaf_words, words_bytes), "cuMemcpyHtoD tile_static_mse words"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemsetD8(*d_output_ret, 0u, output_bytes), "cuMemsetD8 tile_static_mse output"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuLaunchKernel(kernel, blocks, 1u, 1u, CUSR_TEST_THREADS, 1u, 1u, shared_bytes, NULL, args, NULL), "cuLaunchKernel tile_static_mse"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuCtxSynchronize(), "cuCtxSynchronize tile_static_mse"));
    CUSR_TEST_CHECK_RET(cusr_test_check_cuda(cuMemcpyDtoH(output, *d_output_ret, output_bytes), "cuMemcpyDtoH tile_static_mse output"));
    {
        size_t output_idx;
        const float inverse_num_rows = 1.0f / (float)CUSR_TEST_ROWS;

        for (output_idx = 0u; output_idx < output_count; ++output_idx) {
            output[output_idx] *= inverse_num_rows;
        }
    }

    CUSR_TEST_ERROR_RET(1);
}

static int
cusr_test_run_tile_static_mse(
    const CusrTestCubin* cubin,
    size_t tile_rows,
    uint32_t num_asts,
    const float* x,
    const float* target,
    const uint8_t* leaf_masks,
    const uint32_t* leaf_words,
    float* output)
{
    CUmodule module = NULL;
    CUdeviceptr d_x = 0u;
    CUdeviceptr d_target = 0u;
    CUdeviceptr d_masks = 0u;
    CUdeviceptr d_words = 0u;
    CUdeviceptr d_output = 0u;
    int ok = cusr_test_run_tile_static_mse_impl(cubin, tile_rows, num_asts, x, target, leaf_masks, leaf_words, output, &module, &d_x, &d_target, &d_masks, &d_words, &d_output);

    if (d_output != 0u) (void)cuMemFree(d_output);
    if (d_words != 0u) (void)cuMemFree(d_words);
    if (d_masks != 0u) (void)cuMemFree(d_masks);
    if (d_target != 0u) (void)cuMemFree(d_target);
    if (d_x != 0u) (void)cuMemFree(d_x);
    if (module != NULL) (void)cuModuleUnload(module);

    return ok;
}

static int
cusr_test_main_impl(const char* output_dir, CUcontext* context_ret)
{
    uint32_t capability_major = 0u;
    uint32_t capability_minor = 0u;
    CusrTestCubin mse_128_cubin = { NULL, 0u };
    CusrTestCubin mse_256_cubin = { NULL, 0u };
    CusrTestCubin mse_64_cubin = { NULL, 0u };
    CusrTestCubin mse_partial_cubin = { NULL, 0u };
    CusrTestCubin mse_capacity_8_cubin = { NULL, 0u };
    CusrTestCubin mse_capacity_32_cubin = { NULL, 0u };
    float* x = NULL;
    float* target = NULL;
    CusrSettingF32* settings = NULL;
    uint8_t* leaf_masks = NULL;
    uint32_t* leaf_words = NULL;
    float* cpu_eval_output = NULL;
    float* cpu_mse_output = NULL;
    float* mse_128_output = NULL;
    float* mse_256_output = NULL;
    float* mse_64_output = NULL;
    float* mse_partial_output = NULL;
    float* mse_capacity_8_output = NULL;
    float* mse_capacity_32_output = NULL;
    size_t x_bytes = (size_t)CUSR_TEST_ROWS * CUSR_TEST_COLUMNS * sizeof(float);
    size_t target_bytes = (size_t)CUSR_TEST_ROWS * sizeof(float);
    size_t settings_bytes = (size_t)CUSR_TEST_SETTINGS * sizeof(CusrSettingF32);
    size_t mask_bytes = (size_t)CUSR_TEST_SETTINGS * sizeof(uint8_t);
    size_t words_bytes = (size_t)CUSR_TEST_SETTINGS * CUSR_TEST_LEAF_WORDS_STRIDE * sizeof(uint32_t);
    size_t cpu_eval_output_bytes = (size_t)CUSR_TEST_PROGRAMS * CUSR_TEST_SETTINGS * CUSR_TEST_ROWS * sizeof(float);
    size_t mse_output_bytes = CUSR_TEST_PROGRAMS * CUSR_TEST_SETTINGS * sizeof(float);
    size_t partial_output_bytes = CUSR_TEST_ACTIVE_PROGRAMS * CUSR_TEST_SETTINGS * sizeof(float);
    size_t capacity_compare_output_bytes = CUSR_TEST_CAPACITY_COMPARE_PROGRAMS * CUSR_TEST_SETTINGS * sizeof(float);

    CUSR_TEST_CHECK_RET(cusr_test_make_dir(output_dir));
    CUSR_TEST_CHECK_RET(cusr_test_device_arch(&capability_major, &capability_minor, context_ret));
    CUSR_TEST_CHECK_RET(CUSR_TEST_ARRAY_COUNT(cusr_test_programs) == CUSR_TEST_PROGRAMS);
    CUSR_TEST_CHECK_RET(CUSR_TEST_ARRAY_COUNT(cusr_test_routines) == CUSR_TEST_ROUTINE_COUNT);

    x = (float*)malloc(x_bytes);
    target = (float*)malloc(target_bytes);
    settings = (CusrSettingF32*)malloc(settings_bytes);
    leaf_masks = (uint8_t*)malloc(mask_bytes);
    leaf_words = (uint32_t*)malloc(words_bytes);
    cpu_eval_output = (float*)malloc(cpu_eval_output_bytes);
    cpu_mse_output = (float*)malloc(mse_output_bytes);
    mse_128_output = (float*)malloc(mse_output_bytes);
    mse_256_output = (float*)malloc(mse_output_bytes);
    mse_64_output = (float*)malloc(mse_output_bytes);
    mse_partial_output = (float*)malloc(partial_output_bytes);
    mse_capacity_8_output = (float*)malloc(capacity_compare_output_bytes);
    mse_capacity_32_output = (float*)malloc(capacity_compare_output_bytes);

    CUSR_TEST_CHECK_RET(x != NULL && target != NULL && settings != NULL && leaf_masks != NULL && leaf_words != NULL &&
                        cpu_eval_output != NULL && cpu_mse_output != NULL &&
                        mse_128_output != NULL && mse_256_output != NULL && mse_64_output != NULL && mse_partial_output != NULL &&
                        mse_capacity_8_output != NULL && mse_capacity_32_output != NULL);

    cusr_test_fill_data(x, target);
    cusr_test_fill_settings(settings, leaf_masks, leaf_words);

    CUSR_TEST_CHECK_RET(cusr_test_build_patched_cubin(CUSR_TEST_KERNEL_TILE_STATIC_MSE_128, 16u, CUSR_TEST_PROGRAMS, capability_major, capability_minor, &mse_128_cubin));
    CUSR_TEST_CHECK_RET(cusr_test_build_patched_cubin(CUSR_TEST_KERNEL_TILE_STATIC_MSE_256, 16u, CUSR_TEST_PROGRAMS, capability_major, capability_minor, &mse_256_cubin));
    CUSR_TEST_CHECK_RET(cusr_test_build_patched_cubin(CUSR_TEST_KERNEL_TILE_STATIC_MSE_64, 16u, CUSR_TEST_PROGRAMS, capability_major, capability_minor, &mse_64_cubin));
    CUSR_TEST_CHECK_RET(cusr_test_build_patched_cubin(CUSR_TEST_KERNEL_TILE_STATIC_MSE_64, 16u, CUSR_TEST_ACTIVE_PROGRAMS, capability_major, capability_minor, &mse_partial_cubin));
    CUSR_TEST_CHECK_RET(cusr_test_build_patched_cubin(CUSR_TEST_KERNEL_TILE_STATIC_MSE_64, 8u, CUSR_TEST_CAPACITY_COMPARE_PROGRAMS, capability_major, capability_minor, &mse_capacity_8_cubin));
    CUSR_TEST_CHECK_RET(cusr_test_build_patched_cubin(CUSR_TEST_KERNEL_TILE_STATIC_MSE_64, 32u, CUSR_TEST_CAPACITY_COMPARE_PROGRAMS, capability_major, capability_minor, &mse_capacity_32_cubin));

    CUSR_TEST_CHECK_RET(cusr_test_run_tile_static_mse(
        &mse_128_cubin,
        128u,
        CUSR_TEST_PROGRAMS,
        x,
        target,
        leaf_masks,
        leaf_words,
        mse_128_output));
    CUSR_TEST_CHECK_RET(cusr_test_run_tile_static_mse(
        &mse_256_cubin,
        256u,
        CUSR_TEST_PROGRAMS,
        x,
        target,
        leaf_masks,
        leaf_words,
        mse_256_output));
    CUSR_TEST_CHECK_RET(cusr_test_run_tile_static_mse(
        &mse_64_cubin,
        64u,
        CUSR_TEST_PROGRAMS,
        x,
        target,
        leaf_masks,
        leaf_words,
        mse_64_output));
    CUSR_TEST_CHECK_RET(cusr_test_run_tile_static_mse(
        &mse_partial_cubin,
        64u,
        CUSR_TEST_ACTIVE_PROGRAMS,
        x,
        target,
        leaf_masks,
        leaf_words,
        mse_partial_output));
    CUSR_TEST_CHECK_RET(cusr_test_run_tile_static_mse(
        &mse_capacity_8_cubin,
        64u,
        CUSR_TEST_CAPACITY_COMPARE_PROGRAMS,
        x,
        target,
        leaf_masks,
        leaf_words,
        mse_capacity_8_output));
    CUSR_TEST_CHECK_RET(cusr_test_run_tile_static_mse(
        &mse_capacity_32_cubin,
        64u,
        CUSR_TEST_CAPACITY_COMPARE_PROGRAMS,
        x,
        target,
        leaf_masks,
        leaf_words,
        mse_capacity_32_output));

    CUSR_TEST_CHECK_RET(cusr_ast_sass_cpu_eval_program_ptrs_f32(
        x,
        CUSR_TEST_ROWS,
        CUSR_TEST_ROWS,
        settings,
        CUSR_TEST_SETTINGS,
        cusr_test_routines,
        CUSR_TEST_ARRAY_COUNT(cusr_test_routines),
        cusr_test_programs,
        CUSR_TEST_ARRAY_COUNT(cusr_test_programs),
        cpu_eval_output
    ) == CUSR_AST_SASS_CPU_SUCCESS);

    cusr_test_fill_cpu_tile_static_mse_output(cpu_eval_output, target, cpu_mse_output);

    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_128", mse_128_output, cpu_mse_output, mse_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_256", mse_256_output, cpu_mse_output, mse_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_64", mse_64_output, cpu_mse_output, mse_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_128_256", mse_256_output, mse_128_output, mse_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_128_64", mse_64_output, mse_128_output, mse_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_partial", mse_partial_output, cpu_mse_output, partial_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_capacity_8_active_8", mse_capacity_8_output, cpu_mse_output, capacity_compare_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_capacity_32_active_8", mse_capacity_32_output, cpu_mse_output, capacity_compare_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));
    CUSR_TEST_CHECK_RET(cusr_test_compare_f32("tile_static_mse_capacity_8_32", mse_capacity_32_output, mse_capacity_8_output, capacity_compare_output_bytes / sizeof(float), 5.0e-4f, 5.0e-5f));

    CUSR_TEST_CHECK_RET(cusr_test_write_meta(output_dir));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "x_f32.bin", x, x_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "target_f32.bin", target, target_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "settings_f32.bin", settings, settings_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "leaf_masks_u8.bin", leaf_masks, mask_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "leaf_words_u32.bin", leaf_words, words_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "tile_static_mse_128_output_f32.bin", mse_128_output, mse_output_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "tile_static_mse_256_output_f32.bin", mse_256_output, mse_output_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "tile_static_mse_64_output_f32.bin", mse_64_output, mse_output_bytes));
    CUSR_TEST_CHECK_RET(cusr_test_write_named_file(output_dir, "tile_static_mse_partial_output_f32.bin", mse_partial_output, partial_output_bytes));

    free(mse_partial_cubin.data);
    free(mse_capacity_32_cubin.data);
    free(mse_capacity_8_cubin.data);
    free(mse_64_cubin.data);
    free(mse_256_cubin.data);
    free(mse_128_cubin.data);
    free(mse_64_output);
    free(mse_partial_output);
    free(mse_capacity_32_output);
    free(mse_capacity_8_output);
    free(mse_256_output);
    free(mse_128_output);
    free(cpu_mse_output);
    free(cpu_eval_output);
    free(leaf_words);
    free(leaf_masks);
    free(settings);
    free(target);
    free(x);

    CUSR_TEST_ERROR_RET(1);
}

int
main(int argc, char** argv)
{
    const char* output_dir = argc > 1 ? argv[1] : "cusr_nvrtc_ast_sass_outputs";
    CUcontext context = NULL;
    int ok = cusr_test_main_impl(output_dir, &context);

    if (context != NULL) {
        (void)cuCtxDestroy(context);
    }

    if (!ok) {
        return 1;
    }

    printf("wrote %s\n", output_dir);
    return 0;
}
