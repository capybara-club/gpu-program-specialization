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

#include "cusr_ast_sass_patch.h"
#include "cusr_ast_sass_cpu.h"
#include "cusr_ast_routines.h"
#include "cusr_settings.h"
#include "cusr_tile_static_mse_embedded.h"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

static constexpr uint32_t CUSR_CPP_CHECK_ROWS = 128u;
static constexpr uint32_t CUSR_CPP_CHECK_COLUMNS = 8u;
static constexpr uint32_t CUSR_CPP_CHECK_SETTINGS = 2u;
static constexpr uint32_t CUSR_CPP_CHECK_THREADS = 128u;
static constexpr uint32_t CUSR_CPP_CHECK_FIRST_MARKER = 0x7fc0ffeeu;
static constexpr uint32_t CUSR_CPP_CHECK_TILE_ROWS = 64u;

static constexpr CusrAstInstruction cusr_ast_sass_cpp_check_program[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_safe_sqrt,
    cusr_ast_encode_input(1u),
    cusr_ast_encode_exp,
    cusr_ast_encode_add,
    cusr_ast_encode_input(2u),
    cusr_ast_encode_log10,
    cusr_ast_encode_add,
    cusr_ast_encode_input(3u),
    cusr_ast_encode_input(4u),
    cusr_ast_encode_safe_div,
    cusr_ast_encode_add,
    cusr_ast_encode_input(5u),
    cusr_ast_encode_input(6u),
    cusr_ast_encode_pow,
    cusr_ast_encode_add,
    cusr_ast_encode_return
};

static constexpr const CusrAstInstruction* cusr_ast_sass_cpp_check_programs[] = {
    cusr_ast_sass_cpp_check_program
};

static constexpr const CusrAstInstruction* cusr_ast_sass_cpp_check_module_programs[] = {
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program,
    cusr_ast_sass_cpp_check_program
};

static_assert(sizeof(CusrAstInstruction) == 8u, "AST instruction must stay 8 bytes");
static_assert(cusr_ast_sass_cpp_check_program[0].payload.idx == 0u, "input index encode failed");
static_assert(cusr_ast_sass_cpp_check_program[1].payload.idx == CUSR_AST_ROUTINE_SAFE_SQRT, "routine encode failed");
static_assert(cusr_ast_routine_exp[1].payload.bits == CUSR_AST_ROUTINES_LOG2E_BITS, "routine constant encode failed");
static_assert(cusr_ast_encode_tanh.payload.op == CUSR_AST_OP_TANH, "tanh encode failed");
static_assert(cusr_ast_encode_log.payload.idx != cusr_ast_encode_safe_log.payload.idx, "log encodes must remain distinct");
static_assert(cusr_ast_encode_log10.payload.idx != cusr_ast_encode_safe_log10.payload.idx, "log10 encodes must remain distinct");
static_assert(cusr_ast_encode_pow.payload.idx != cusr_ast_encode_safe_pow.payload.idx, "pow encodes must remain distinct");

struct CusrCppCheckContext {
    CUcontext context = nullptr;

    ~CusrCppCheckContext()
    {
        if (context != nullptr) {
            (void)cuCtxDestroy(context);
        }
    }
};

struct CusrCppCheckModule {
    CUmodule module = nullptr;

    ~CusrCppCheckModule()
    {
        if (module != nullptr) {
            (void)cuModuleUnload(module);
        }
    }
};

struct CusrCppCheckMemory {
    CUdeviceptr ptr = 0u;

    ~CusrCppCheckMemory()
    {
        if (ptr != 0u) {
            (void)cuMemFree(ptr);
        }
    }
};

static bool
cusr_cpp_check_cuda(CUresult result, const char* label)
{
    if (result != CUDA_SUCCESS) {
        const char* name = "unknown";
        const char* description = "unknown";
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);
        std::fprintf(stderr, "%s failed: %s (%s)\n", label, name, description);
        return false;
    }

    return true;
}

static bool
cusr_cpp_check_create_context(CusrCppCheckContext& context)
{
    CUdevice device = 0;
    int major = 0;
    int minor = 0;

    if (!cusr_cpp_check_cuda(cuInit(0u), "cuInit") ||
        !cusr_cpp_check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        !cusr_cpp_check_cuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major") ||
        !cusr_cpp_check_cuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor")) {
        return false;
    }

#if CUDA_VERSION >= 13000
    if (!cusr_cpp_check_cuda(cuCtxCreate(&context.context, nullptr, 0u, device), "cuCtxCreate")) {
#else
    if (!cusr_cpp_check_cuda(cuCtxCreate(&context.context, 0u, device), "cuCtxCreate")) {
#endif
        return false;
    }

    if (static_cast<unsigned>(major * 10 + minor) != cusr_tile_static_mse_embedded_sass_arch()) {
        std::fprintf(stderr, "embedded cubin is sm_%u but device is sm_%d%d\n", cusr_tile_static_mse_embedded_sass_arch(), major, minor);
        return false;
    }
    return true;
}

static bool
cusr_cpp_check_copy_embedded_cubin(std::vector<unsigned char>& cubin)
{
    size_t cubin_size = 0u;
    const unsigned char* embedded = cusr_tile_static_mse_embedded_cubin(8u, &cubin_size);

    if (embedded == nullptr || cubin_size == 0u) {
        std::fprintf(stderr, "embedded 8-kernel cubin is unavailable\n");
        return false;
    }

    cubin.assign(embedded, embedded + cubin_size);
    return true;
}

static bool
cusr_cpp_check_patch_cubin(std::vector<unsigned char>& cubin)
{
    static const char* const function_names[] = {
        "cusr_tile_static_mse_f32_000",
        "cusr_tile_static_mse_f32_001",
        "cusr_tile_static_mse_f32_002",
        "cusr_tile_static_mse_f32_003",
        "cusr_tile_static_mse_f32_004",
        "cusr_tile_static_mse_f32_005",
        "cusr_tile_static_mse_f32_006",
        "cusr_tile_static_mse_f32_007"
    };
    size_t workspace_size = 0u;
    CusrSassInspectHandle inspect;
    CusrSassInspectResult inspect_result = cusr_sass_inspect_workspace_size(
        cubin.data(),
        cubin.size(),
        function_names,
        std::size(function_names),
        CUSR_TILE_STATIC_MSE_EMBEDDED_AST_CAPACITY,
        CUSR_CPP_CHECK_FIRST_MARKER,
        1u,
        &workspace_size
    );

    if (inspect_result != CUSR_SASS_INSPECT_SUCCESS) {
        std::fprintf(stderr, "inspect workspace failed: %s\n", cusr_sass_inspect_result_to_string(inspect_result));
        return false;
    }

    std::vector<unsigned char> workspace(std::max<size_t>(workspace_size, 1u));
    inspect_result = cusr_sass_inspect(
        cubin.data(),
        cubin.size(),
        function_names,
        std::size(function_names),
        CUSR_TILE_STATIC_MSE_EMBEDDED_AST_CAPACITY,
        CUSR_CPP_CHECK_FIRST_MARKER,
        1u,
        workspace.data(),
        workspace_size,
        &inspect
    );

    if (inspect_result != CUSR_SASS_INSPECT_SUCCESS) {
        std::fprintf(stderr, "inspect failed: %s\n", cusr_sass_inspect_result_to_string(inspect_result));
        return false;
    }

    CusrAstSassPatchStats stats;
    const CusrAstSassPatchResult patch_result = cusr_ast_sass_patch_cubin(
        &inspect,
        inspect.sass_arch / 10u,
        inspect.sass_arch % 10u,
        CUSR_AST_SASS_PATCH_EPILOGUE_SSE,
        cusr_ast_routines,
        std::size(cusr_ast_routines),
        cusr_ast_sass_cpp_check_module_programs,
        std::size(cusr_ast_sass_cpp_check_module_programs),
        cubin.data(),
        cubin.size(),
        &stats
    );

    if (patch_result != CUSR_AST_SASS_PATCH_SUCCESS) {
        std::fprintf(stderr, "patch failed: %s\n", cusr_ast_sass_patch_result_to_string(patch_result));
        return false;
    }

    if (stats.sites_patched != std::size(function_names) || stats.asts_patched != std::size(function_names)) {
        std::fprintf(stderr, "expected eight patched sites and ASTs, got %zu and %zu\n", stats.sites_patched, stats.asts_patched);
        return false;
    }

    return true;
}

static void
cusr_cpp_check_fill_data(
    std::vector<float>& x,
    std::vector<float>& target,
    std::vector<CusrSettingF32>& settings,
    std::vector<uint8_t>& leaf_masks,
    std::vector<uint32_t>& leaf_words)
{
    for (uint32_t column = 0u; column < CUSR_CPP_CHECK_COLUMNS; ++column) {
        for (uint32_t row = 0u; row < CUSR_CPP_CHECK_ROWS; ++row) {
            x[static_cast<size_t>(column) * CUSR_CPP_CHECK_ROWS + row] =
                0.35f + 0.11f * static_cast<float>(column) + 0.001f * static_cast<float>(row);
        }
    }

    for (uint32_t row = 0u; row < CUSR_CPP_CHECK_ROWS; ++row) {
        target[row] = 0.45f + 0.002f * static_cast<float>(row);
    }

    for (uint32_t setting_idx = 0u; setting_idx < CUSR_CPP_CHECK_SETTINGS; ++setting_idx) {
        settings[setting_idx].column_mask = 0xffu;
        for (uint32_t input_idx = 0u; input_idx < CUSR_NUM_INPUTS; ++input_idx) {
            settings[setting_idx].column_indices[input_idx] = input_idx & 7u;
            settings[setting_idx].constants[input_idx] = 0.75f + 0.25f * static_cast<float>(input_idx);
            leaf_words[static_cast<size_t>(setting_idx) * CUSR_NUM_INPUTS + input_idx] =
                settings[setting_idx].column_indices[input_idx];
        }
        leaf_masks[setting_idx] = static_cast<uint8_t>(settings[setting_idx].column_mask);
    }
}

static bool
cusr_cpp_check_run_kernel(
    const std::vector<unsigned char>& cubin,
    const std::vector<float>& x,
    const std::vector<float>& target,
    const std::vector<uint8_t>& leaf_masks,
    const std::vector<uint32_t>& leaf_words,
    std::vector<float>& output)
{
    CusrCppCheckModule module;
    CusrCppCheckMemory d_x;
    CusrCppCheckMemory d_target;
    CusrCppCheckMemory d_leaf_masks;
    CusrCppCheckMemory d_leaf_words;
    CusrCppCheckMemory d_output;
    CUfunction kernel = nullptr;
    size_t num_rows = CUSR_CPP_CHECK_ROWS;
    uint32_t num_columns = CUSR_CPP_CHECK_COLUMNS;
    size_t leading_dim = CUSR_CPP_CHECK_ROWS;
    size_t leaf_words_stride = CUSR_NUM_INPUTS;
    uint32_t num_settings = CUSR_CPP_CHECK_SETTINGS;
    uint32_t num_asts = 1u;
    const unsigned int blocks = (CUSR_CPP_CHECK_ROWS + CUSR_CPP_CHECK_TILE_ROWS - 1u) / CUSR_CPP_CHECK_TILE_ROWS;
    const uint32_t shared_stride = CUSR_CPP_CHECK_COLUMNS | 1u;
    const unsigned int shared_bytes = CUSR_CPP_CHECK_TILE_ROWS * (shared_stride + 1u) * sizeof(float);
    const size_t x_bytes = x.size() * sizeof(float);
    const size_t target_bytes = target.size() * sizeof(float);
    const size_t leaf_masks_bytes = leaf_masks.size() * sizeof(uint8_t);
    const size_t leaf_words_bytes = leaf_words.size() * sizeof(uint32_t);
    const size_t output_bytes = output.size() * sizeof(float);

    if (!cusr_cpp_check_cuda(cuModuleLoadData(&module.module, cubin.data()), "cuModuleLoadData") ||
        !cusr_cpp_check_cuda(cuModuleGetFunction(&kernel, module.module, "cusr_tile_static_mse_f32_000"), "cuModuleGetFunction") ||
        !cusr_cpp_check_cuda(cuMemAlloc(&d_x.ptr, x_bytes), "cuMemAlloc x") ||
        !cusr_cpp_check_cuda(cuMemAlloc(&d_target.ptr, target_bytes), "cuMemAlloc target") ||
        !cusr_cpp_check_cuda(cuMemAlloc(&d_leaf_masks.ptr, leaf_masks_bytes), "cuMemAlloc leaf masks") ||
        !cusr_cpp_check_cuda(cuMemAlloc(&d_leaf_words.ptr, leaf_words_bytes), "cuMemAlloc leaf words") ||
        !cusr_cpp_check_cuda(cuMemAlloc(&d_output.ptr, output_bytes), "cuMemAlloc output") ||
        !cusr_cpp_check_cuda(cuMemcpyHtoD(d_x.ptr, x.data(), x_bytes), "cuMemcpyHtoD x") ||
        !cusr_cpp_check_cuda(cuMemcpyHtoD(d_target.ptr, target.data(), target_bytes), "cuMemcpyHtoD target") ||
        !cusr_cpp_check_cuda(cuMemcpyHtoD(d_leaf_masks.ptr, leaf_masks.data(), leaf_masks_bytes), "cuMemcpyHtoD leaf masks") ||
        !cusr_cpp_check_cuda(cuMemcpyHtoD(d_leaf_words.ptr, leaf_words.data(), leaf_words_bytes), "cuMemcpyHtoD leaf words") ||
        !cusr_cpp_check_cuda(cuMemsetD8(d_output.ptr, 0u, output_bytes), "cuMemsetD8 output")) {
        return false;
    }

    void* args[] = {
        &d_x.ptr,
        &d_target.ptr,
        &num_rows,
        &num_columns,
        &leading_dim,
        &d_leaf_masks.ptr,
        &d_leaf_words.ptr,
        &leaf_words_stride,
        &num_settings,
        &num_asts,
        &d_output.ptr
    };

    return cusr_cpp_check_cuda(cuLaunchKernel(kernel, blocks, 1u, 1u, CUSR_CPP_CHECK_THREADS, 1u, 1u, shared_bytes, nullptr, args, nullptr), "cuLaunchKernel") &&
        cusr_cpp_check_cuda(cuCtxSynchronize(), "cuCtxSynchronize") &&
        cusr_cpp_check_cuda(cuMemcpyDtoH(output.data(), d_output.ptr, output_bytes), "cuMemcpyDtoH output");
}

static bool
cusr_cpp_check_compare(const std::vector<float>& actual, const std::vector<float>& expected)
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst_idx = 0u;

    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "cpu compare failed: actual=%zu values expected=%zu values\n", actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0u; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr, "cpu compare failed: non-finite value at %zu actual=%g expected=%g\n", i, actual[i], expected[i]);
            return false;
        }

        const float abs_error = std::fabs(actual[i] - expected[i]);
        const float rel_error = abs_error / std::max(std::fabs(expected[i]), 1.0f);
        if (abs_error > max_abs) {
            max_abs = abs_error;
            max_rel = rel_error;
            worst_idx = i;
        }
    }

    if (max_abs > 2.0e-5f && max_rel > 2.0e-5f) {
        std::fprintf(stderr, "cpu compare failed: max_abs=%g max_rel=%g worst=%zu actual=%g expected=%g\n", max_abs, max_rel, worst_idx, actual[worst_idx], expected[worst_idx]);
        return false;
    }

    std::printf("cpp full pipeline ok max_abs=%g max_rel=%g worst=%zu\n", max_abs, max_rel, worst_idx);
    return true;
}

int
main()
{
    CusrCppCheckContext context;
    std::vector<unsigned char> cubin;
    std::vector<float> x(static_cast<size_t>(CUSR_CPP_CHECK_COLUMNS) * CUSR_CPP_CHECK_ROWS);
    std::vector<float> target(CUSR_CPP_CHECK_ROWS);
    std::vector<CusrSettingF32> settings(CUSR_CPP_CHECK_SETTINGS);
    std::vector<uint8_t> leaf_masks(CUSR_CPP_CHECK_SETTINGS);
    std::vector<uint32_t> leaf_words(static_cast<size_t>(CUSR_CPP_CHECK_SETTINGS) * CUSR_NUM_INPUTS);
    std::vector<float> gpu_output(CUSR_CPP_CHECK_SETTINGS);
    std::vector<float> cpu_eval(static_cast<size_t>(CUSR_CPP_CHECK_SETTINGS) * CUSR_CPP_CHECK_ROWS);
    std::vector<float> cpu_output(CUSR_CPP_CHECK_SETTINGS);

    if (!cusr_cpp_check_create_context(context) ||
        !cusr_cpp_check_copy_embedded_cubin(cubin) ||
        !cusr_cpp_check_patch_cubin(cubin)) {
        return 1;
    }

    cusr_cpp_check_fill_data(x, target, settings, leaf_masks, leaf_words);

    if (cusr_ast_sass_cpu_eval_program_ptrs_f32(
            x.data(),
            CUSR_CPP_CHECK_ROWS,
            CUSR_CPP_CHECK_ROWS,
            settings.data(),
            CUSR_CPP_CHECK_SETTINGS,
            cusr_ast_routines,
            std::size(cusr_ast_routines),
            cusr_ast_sass_cpp_check_programs,
            std::size(cusr_ast_sass_cpp_check_programs),
            cpu_eval.data()) != CUSR_AST_SASS_CPU_SUCCESS) {
        std::fprintf(stderr, "cusr_ast_sass_cpu_eval_program_ptrs_f32 failed\n");
        return 1;
    }

    for (uint32_t setting = 0u; setting < CUSR_CPP_CHECK_SETTINGS; ++setting) {
        float sse = 0.0f;
        for (uint32_t row = 0u; row < CUSR_CPP_CHECK_ROWS; ++row) {
            const float error = cpu_eval[static_cast<size_t>(setting) * CUSR_CPP_CHECK_ROWS + row] - target[row];
            sse += error * error;
        }
        cpu_output[setting] = sse;
    }

    if (!cusr_cpp_check_run_kernel(cubin, x, target, leaf_masks, leaf_words, gpu_output) ||
        !cusr_cpp_check_compare(gpu_output, cpu_output)) {
        return 1;
    }

    return 0;
}
