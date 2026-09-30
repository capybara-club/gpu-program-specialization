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
#include "secant.h"
#include "support/api_helpers.h"
#include "support/ast_stress.h"
#include "support/cuda_runtime.h"

#include <cuda.h>
#include <nvrtc.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STRESS_PROGRAM_CAPACITY SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY
#define STRESS_NUM_STREAMS 4u
#define STRESS_TILE_ROWS 128u
#define STRESS_THREADS_PER_BLOCK 128u

typedef struct StressState {
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    CUstream streams[STRESS_NUM_STREAMS];
    CUdeviceptr device_input;
    CUdeviceptr device_targets;
    CUdeviceptr device_materialize;
    CUdeviceptr device_sse;
    unsigned char* materialize_template;
    unsigned char* sse_template;
    unsigned char* materialize_cubin;
    unsigned char* sse_cubin;
    size_t materialize_cubin_size;
    size_t sse_cubin_size;
    void* materialize_workspace;
    void* sse_workspace;
    SecantCubinPlan* materialize_handle;
    SecantCubinPlan* sse_handle;
    SecantAstInstruction* programs;
    const SecantAstInstruction** asts;
    size_t* program_sizes;
    uint32_t* program_masks;
    size_t* program_mufu_counts;
    float* input;
    float* targets;
    float* expected_materialize;
    float* actual_materialize;
    float* expected_sse;
    float* actual_sse;
    int retained;
} StressState;

static int
stress_parse_size(const char* text, size_t* value_ret) {
    unsigned long long value;
    char* end = NULL;

    if (text == NULL || value_ret == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 0;
    }
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
stress_parse_u64(const char* text, uint64_t* value_ret) {
    unsigned long long value;
    char* end = NULL;

    if (text == NULL || value_ret == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 0;
    }
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *value_ret = (uint64_t)value;
    return 1;
}

static int
stress_parse_int(const char* text, int* value_ret) {
    long value;
    char* end = NULL;

    if (text == NULL || value_ret == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    value = strtol(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT32_MIN || value > INT32_MAX) {
        return 0;
    }
    *value_ret = (int)value;
    return 1;
}

static void
stress_usage(const char* program) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --iterations N\n"
        "  --kernels N\n"
        "  --asts-per-kernel N\n"
        "  --rows N\n"
        "  --inputs N\n"
        "  --targets N\n"
        "  --max-leaves N\n"
        "  --max-unary-depth N\n"
        "  --patch-instructions-per-ast N\n"
        "  --seed N\n"
        "  --device N\n",
        program);
}

static int
stress_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
stress_compile_template(
    int sse,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    uint32_t target_sm,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    SecantCubinMaterializeRecipe materialize_recipe = secant_cubin_materialize_recipe_init();
    SecantCubinSSERecipe sse_recipe = secant_cubin_sse_recipe_init();
    char architecture[64];
    const char* options[3];
    char* source = NULL;
    char* log = NULL;
    unsigned char* cubin = NULL;
    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_result = NVRTC_SUCCESS;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    size_t cubin_size = 0u;
    size_t log_size = 0u;
    int architecture_bytes;
    int success = 1;

    materialize_recipe.num_kernels = num_kernels;
    materialize_recipe.asts_per_kernel = asts_per_kernel;
    materialize_recipe.num_inputs = num_inputs;
    materialize_recipe.patch_capacity_instructions = patch_capacity_instructions;
    sse_recipe.num_kernels = num_kernels;
    sse_recipe.asts_per_kernel = asts_per_kernel;
    sse_recipe.num_inputs = num_inputs;
    sse_recipe.num_targets = num_targets;
    sse_recipe.tile_rows = STRESS_TILE_ROWS;
    sse_recipe.threads_per_block = STRESS_THREADS_PER_BLOCK;
    sse_recipe.patch_capacity_instructions = patch_capacity_instructions;
    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    result = sse
        ? secant_test_cubin_source_generate(
            &sse_recipe,
            NULL,
            0u,
            &source_size)
        : secant_test_cubin_source_generate(
            &materialize_recipe,
            NULL,
            0u,
            &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        fprintf(
            stderr,
            "%s CUDA source measurement failed: SecantResult(%d)\n",
            sse ? "SSE" : "materialize",
            (int)result);
        success = 0;
    }
    if (success) {
        source = (char*)malloc(source_size);
        if (source == NULL) {
            success = 0;
        }
    }
    if (success) {
        result = sse
            ? secant_test_cubin_source_generate(
                &sse_recipe,
                source,
                source_size,
                &written_size)
            : secant_test_cubin_source_generate(
                &materialize_recipe,
                source,
                source_size,
                &written_size);
        if (result != SECANT_SUCCESS || written_size != source_size) {
            fprintf(
                stderr,
                "%s CUDA source generation failed: SecantResult(%d)\n",
                sse ? "SSE" : "materialize",
                (int)result);
            success = 0;
        }
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-architecture=sm_%u",
        target_sm);
    if (architecture_bytes < 0 ||
        (size_t)architecture_bytes >= sizeof(architecture)) {
        success = 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    if (success) {
        nvrtc_result = nvrtcCreateProgram(
            &program,
            source,
            sse
                ? "secant_cubin_stress_sse.cu"
                : "secant_cubin_stress_materialize.cu",
            0,
            NULL,
            NULL);
        if (nvrtc_result == NVRTC_SUCCESS) {
            nvrtc_result = nvrtcCompileProgram(
                program,
                (int)(sizeof(options) / sizeof(options[0])),
                options);
        }
        if (nvrtc_result != NVRTC_SUCCESS) {
            if (program != NULL &&
                nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS &&
                log_size != 0u) {
                log = (char*)malloc(log_size);
                if (log != NULL &&
                    nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    fprintf(stderr, "%s\n", log);
                }
            }
            fprintf(
                stderr,
                "%s NVRTC compilation failed: %s\n",
                sse ? "SSE" : "materialize",
                nvrtcGetErrorString(nvrtc_result));
            success = 0;
        }
    }
    if (success &&
        (nvrtcGetCUBINSize(program, &cubin_size) != NVRTC_SUCCESS ||
         cubin_size == 0u)) {
        success = 0;
    }
    if (success) {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin == NULL ||
            nvrtcGetCUBIN(program, (char*)cubin) != NVRTC_SUCCESS) {
            success = 0;
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    free(log);
    free(source);
    if (!success) {
        free(cubin);
        return 0;
    }
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

static int
stress_cuda_create(
    StressState* state,
    int device_ordinal,
    uint32_t* target_sm_ret
) {
    int major = 0;
    int minor = 0;
    size_t stream_idx;

    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&state->device, device_ordinal) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(
            &state->context,
            state->device) != CUDA_SUCCESS) {
        return 0;
    }
    state->retained = 1;
    if (cuCtxGetCurrent(&state->previous_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(state->context) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            state->device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            state->device) != CUDA_SUCCESS) {
        return 0;
    }
    for (stream_idx = 0u; stream_idx < STRESS_NUM_STREAMS; ++stream_idx) {
        if (cuStreamCreate(
                state->streams + stream_idx,
                CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
            return 0;
        }
    }
    *target_sm_ret = (uint32_t)(major * 10 + minor);
    return 1;
}

static int
stress_inspect_templates(
    StressState* state,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions
) {
    SecantCubinMaterializeRecipe materialize_recipe = secant_cubin_materialize_recipe_init();
    SecantCubinSSERecipe sse_recipe = secant_cubin_sse_recipe_init();
    size_t workspace_size = 0u;
    SecantResult result;

    materialize_recipe.num_kernels = num_kernels;
    materialize_recipe.asts_per_kernel = asts_per_kernel;
    materialize_recipe.num_inputs = num_inputs;
    materialize_recipe.patch_capacity_instructions = patch_capacity_instructions;
    sse_recipe.num_kernels = num_kernels;
    sse_recipe.asts_per_kernel = asts_per_kernel;
    sse_recipe.num_inputs = num_inputs;
    sse_recipe.num_targets = num_targets;
    sse_recipe.tile_rows = STRESS_TILE_ROWS;
    sse_recipe.threads_per_block = STRESS_THREADS_PER_BLOCK;
    sse_recipe.patch_capacity_instructions = patch_capacity_instructions;
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        state->materialize_template,
        state->materialize_cubin_size,
        NULL,
        0u,
        &workspace_size,
        &state->materialize_handle);
    if (result != SECANT_SUCCESS || workspace_size == 0u) {
        fprintf(
            stderr,
            "materialize inspect measure failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    state->materialize_workspace = malloc(workspace_size);
    if (state->materialize_workspace == NULL) {
        return 0;
    }
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        state->materialize_template,
        state->materialize_cubin_size,
        state->materialize_workspace,
        workspace_size,
        &workspace_size,
        &state->materialize_handle);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize inspect failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    workspace_size = 0u;
    result = secant_test_cubin_inspect(
        &sse_recipe,
        state->sse_template,
        state->sse_cubin_size,
        NULL,
        0u,
        &workspace_size,
        &state->sse_handle);
    if (result != SECANT_SUCCESS || workspace_size == 0u) {
        fprintf(
            stderr,
            "SSE inspect measure failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    state->sse_workspace = malloc(workspace_size);
    if (state->sse_workspace == NULL) {
        return 0;
    }
    result = secant_test_cubin_inspect(
        &sse_recipe,
        state->sse_template,
        state->sse_cubin_size,
        state->sse_workspace,
        workspace_size,
        &workspace_size,
        &state->sse_handle);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "SSE inspect failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    return 1;
}


static int
stress_prepare(
    StressState* state,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t rows,
    size_t patch_instructions_per_ast,
    uint64_t seed,
    int device_ordinal
) {
    size_t num_asts;
    size_t patch_capacity_instructions;
    size_t input_count;
    size_t target_count;
    size_t materialize_count;
    size_t sse_count;
    size_t programs_count;
    uint32_t target_sm;

    if (!stress_checked_mul(
            num_kernels,
            asts_per_kernel,
            &num_asts) ||
        !stress_checked_mul(
            asts_per_kernel,
            patch_instructions_per_ast,
            &patch_capacity_instructions) ||
        !stress_checked_mul(num_inputs, rows, &input_count) ||
        !stress_checked_mul(num_targets, rows, &target_count) ||
        !stress_checked_mul(num_asts, rows, &materialize_count) ||
        !stress_checked_mul(num_asts, num_targets, &sse_count) ||
        !stress_checked_mul(
            num_asts,
            STRESS_PROGRAM_CAPACITY,
            &programs_count)) {
        return 0;
    }
    if (!stress_cuda_create(state, device_ordinal, &target_sm) ||
        !stress_compile_template(
            0,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions,
            target_sm,
            &state->materialize_template,
            &state->materialize_cubin_size) ||
        !stress_compile_template(
            1,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions,
            target_sm,
            &state->sse_template,
            &state->sse_cubin_size) ||
        !stress_inspect_templates(
            state,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions)) {
        return 0;
    }
    state->materialize_cubin = (unsigned char*)malloc(state->materialize_cubin_size);
    state->sse_cubin = (unsigned char*)malloc(state->sse_cubin_size);
    state->programs = (SecantAstInstruction*)malloc(
        programs_count * sizeof(*state->programs));
    state->asts = (const SecantAstInstruction**)malloc(
        num_asts * sizeof(*state->asts));
    state->program_sizes = (size_t*)malloc(
        num_asts * sizeof(*state->program_sizes));
    state->program_masks = (uint32_t*)malloc(
        num_asts * sizeof(*state->program_masks));
    state->program_mufu_counts = (size_t*)malloc(
        num_asts * sizeof(*state->program_mufu_counts));
    state->input = (float*)malloc(input_count * sizeof(*state->input));
    state->targets = (float*)malloc(target_count * sizeof(*state->targets));
    state->expected_materialize = (float*)malloc(
        materialize_count * sizeof(*state->expected_materialize));
    state->actual_materialize = (float*)malloc(
        materialize_count * sizeof(*state->actual_materialize));
    state->expected_sse = (float*)malloc(
        sse_count * sizeof(*state->expected_sse));
    state->actual_sse = (float*)malloc(
        sse_count * sizeof(*state->actual_sse));
    if (state->materialize_cubin == NULL ||
        state->sse_cubin == NULL ||
        state->programs == NULL ||
        state->asts == NULL ||
        state->program_sizes == NULL ||
        state->program_masks == NULL ||
        state->program_mufu_counts == NULL ||
        state->input == NULL ||
        state->targets == NULL ||
        state->expected_materialize == NULL ||
        state->actual_materialize == NULL ||
        state->expected_sse == NULL ||
        state->actual_sse == NULL) {
        return 0;
    }
    secant_test_ast_stress_data_fill(
        num_inputs,
        num_targets,
        rows,
        seed,
        state->input,
        state->targets);
    if (cuMemAlloc(
            &state->device_input,
            input_count * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(
            &state->device_targets,
            target_count * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(
            &state->device_materialize,
            materialize_count * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(
            &state->device_sse,
            sse_count * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(
            state->device_input,
            state->input,
            input_count * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(
            state->device_targets,
            state->targets,
            target_count * sizeof(float)) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}


static int
stress_iteration(
    StressState* state,
    size_t iteration,
    uint64_t seed,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t rows,
    size_t max_leaves,
    size_t max_unary_depth,
    uint32_t* instruction_mask,
    size_t* total_program_bytes,
    size_t* min_program_bytes,
    size_t* max_program_bytes,
    size_t* class_counts,
    float* max_materialize_absolute_errors,
    float* max_materialize_relative_errors,
    float* max_sse_absolute_errors,
    float* max_sse_relative_errors
) {
    const size_t num_asts = num_kernels * asts_per_kernel;
    const size_t materialize_count = num_asts * rows;
    const size_t sse_count = num_asts * num_targets;
    const size_t materialize_bytes = materialize_count * sizeof(float);
    const size_t sse_bytes = sse_count * sizeof(float);
    void* materialize_module = NULL;
    void* sse_module = NULL;
    void* materialize_functions[num_kernels];
    void* sse_functions[num_kernels];
    SecantResult cubin_result;
    SecantResult cuda_result;
    SecantResult cpu_result;
    size_t ast_idx;
    size_t kernel_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program = state->programs + ast_idx * STRESS_PROGRAM_CAPACITY;
        const int allow_mufu = ((iteration + ast_idx) & 3u) != 0u;
        uint32_t program_mask = 0u;

        if (!secant_test_ast_stress_program_generate(
                program,
                state->program_sizes + ast_idx,
                num_inputs,
                max_leaves,
                max_unary_depth,
                allow_mufu,
                seed,
                iteration,
                ast_idx,
                &program_mask)) {
            fprintf(
                stderr,
                "AST generation overflow seed=%" PRIu64
                " iteration=%zu ast=%zu\n",
                seed,
                iteration,
                ast_idx);
            return 0;
        }
        state->asts[ast_idx] = program;
        state->program_masks[ast_idx] = program_mask;
        state->program_mufu_counts[ast_idx] =
            secant_test_ast_stress_program_mufu_count(
                program,
                state->program_sizes[ast_idx]);
        ++class_counts[
            state->program_mufu_counts[ast_idx] == 0u ? 0u : 1u];
        *instruction_mask |= program_mask;
        *total_program_bytes += state->program_sizes[ast_idx];
        if (state->program_sizes[ast_idx] < *min_program_bytes) {
            *min_program_bytes = state->program_sizes[ast_idx];
        }
        if (state->program_sizes[ast_idx] > *max_program_bytes) {
            *max_program_bytes = state->program_sizes[ast_idx];
        }
    }
    cpu_result = secant_test_ast_stress_cpu_evaluate(
        num_inputs,
        num_targets,
        state->asts,
        num_asts,
        state->input,
        state->targets,
        rows,
        state->expected_materialize,
        state->expected_sse);
    if (cpu_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "CPU evaluation failed seed=%" PRIu64
            " iteration=%zu: %s\n",
            seed,
            iteration,
            secant_result_to_string(cpu_result));
        return 0;
    }
    memcpy(
        state->materialize_cubin,
        state->materialize_template,
        state->materialize_cubin_size);
    memcpy(
        state->sse_cubin,
        state->sse_template,
        state->sse_cubin_size);
    cubin_result = secant_test_cubin_specialize_into(
        state->materialize_handle,
        secant_test_ast_stress_routines,
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
        state->asts,
        num_asts,
        state->materialize_cubin,
        state->materialize_cubin_size);
    if (cubin_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize specialization failed seed=%" PRIu64
            " iteration=%zu: SecantResult(%d)\n",
            seed,
            iteration,
            (int)cubin_result);
        return 0;
    }
    cubin_result = secant_test_cubin_specialize_into(
        state->sse_handle,
        secant_test_ast_stress_routines,
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
        state->asts,
        num_asts,
        state->sse_cubin,
        state->sse_cubin_size);
    if (cubin_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "SSE specialization failed seed=%" PRIu64
            " iteration=%zu: SecantResult(%d)\n",
            seed,
            iteration,
            (int)cubin_result);
        return 0;
    }
    cuda_result = secant_test_cuda_module_load(
        state->materialize_cubin,
        state->materialize_cubin_size,
        num_kernels,
        &materialize_module,
        materialize_functions);
    if (cuda_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize module load failed seed=%" PRIu64
            " iteration=%zu: %s\n",
            seed,
            iteration,
            secant_test_cuda_result_to_string(cuda_result));
        return 0;
    }
    cuda_result = secant_test_cuda_module_load(
        state->sse_cubin,
        state->sse_cubin_size,
        num_kernels,
        &sse_module,
        sse_functions);
    if (cuda_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "SSE module load failed seed=%" PRIu64
            " iteration=%zu: %s\n",
            seed,
            iteration,
            secant_test_cuda_result_to_string(cuda_result));
        success = 0;
    }
    if (success &&
        (cuMemsetD8(
             state->device_materialize,
             0xffu,
             materialize_bytes) != CUDA_SUCCESS ||
         cuMemsetD8(
             state->device_sse,
             0u,
             sse_bytes) != CUDA_SUCCESS ||
         cuCtxSynchronize() != CUDA_SUCCESS)) {
        success = 0;
    }
    for (kernel_idx = 0u;
         kernel_idx < num_kernels && success;
         ++kernel_idx) {
        cuda_result = secant_test_cuda_run_static_column_materialize(
            materialize_functions[kernel_idx],
            num_inputs,
            asts_per_kernel,
            (const float*)(uintptr_t)state->device_input,
            num_inputs * rows,
            rows,
            rows,
            state->streams[kernel_idx % STRESS_NUM_STREAMS],
            (float*)(uintptr_t)(
                state->device_materialize +
                kernel_idx * asts_per_kernel * rows * sizeof(float)),
            asts_per_kernel * rows,
            rows);
        if (cuda_result == SECANT_SUCCESS) {
            cuda_result = secant_test_cuda_run_static_column_sse(
                sse_functions[kernel_idx],
                num_inputs,
                asts_per_kernel,
                num_targets,
                STRESS_TILE_ROWS,
                STRESS_THREADS_PER_BLOCK,
                (const float*)(uintptr_t)state->device_input,
                num_inputs * rows,
                rows,
                (const float*)(uintptr_t)state->device_targets,
                num_targets * rows,
                rows,
                rows,
                state->streams[kernel_idx % STRESS_NUM_STREAMS],
                (float*)(uintptr_t)(
                    state->device_sse +
                    kernel_idx * asts_per_kernel * num_targets *
                        sizeof(float)),
                asts_per_kernel * num_targets,
                num_targets);
        }
        if (cuda_result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "kernel launch failed seed=%" PRIu64
                " iteration=%zu kernel=%zu: %s\n",
                seed,
                iteration,
                kernel_idx,
                secant_test_cuda_result_to_string(cuda_result));
            success = 0;
        }
    }
    if (success &&
        (cuCtxSynchronize() != CUDA_SUCCESS ||
         cuMemcpyDtoH(
             state->actual_materialize,
             state->device_materialize,
             materialize_bytes) != CUDA_SUCCESS ||
         cuMemcpyDtoH(
             state->actual_sse,
             state->device_sse,
             sse_bytes) != CUDA_SUCCESS)) {
        fprintf(
            stderr,
            "CUDA synchronization or copy failed seed=%" PRIu64
            " iteration=%zu\n",
            seed,
            iteration);
        success = 0;
    }
    if (success) {
        success = secant_test_ast_stress_compare(
            "cubin",
            "materialize",
            state->expected_materialize,
            state->actual_materialize,
            materialize_count,
            rows,
            iteration,
            seed,
            state->asts,
            state->program_sizes,
            state->program_masks,
            state->program_mufu_counts,
            2.0e-6f,
            2.0e-6f,
            5.0e-5f,
            5.0e-4f,
            max_materialize_absolute_errors,
            max_materialize_relative_errors);
    }
    if (success) {
        success = secant_test_ast_stress_compare(
            "cubin",
            "SSE",
            state->expected_sse,
            state->actual_sse,
            sse_count,
            num_targets,
            iteration,
            seed,
            state->asts,
            state->program_sizes,
            state->program_masks,
            state->program_mufu_counts,
            2.0e-3f,
            8.0e-4f,
            1.0e-3f,
            5.0e-4f,
            max_sse_absolute_errors,
            max_sse_relative_errors);
    }
    if (sse_module != NULL &&
        cuModuleUnload((CUmodule)sse_module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (materialize_module != NULL &&
        cuModuleUnload((CUmodule)materialize_module) != CUDA_SUCCESS) {
        success = 0;
    }
    return success;
}

static void
stress_destroy(StressState* state) {
    size_t stream_idx;

    if (state->device_sse != 0u) {
        (void)cuMemFree(state->device_sse);
    }
    if (state->device_materialize != 0u) {
        (void)cuMemFree(state->device_materialize);
    }
    if (state->device_targets != 0u) {
        (void)cuMemFree(state->device_targets);
    }
    if (state->device_input != 0u) {
        (void)cuMemFree(state->device_input);
    }
    for (stream_idx = 0u;
         stream_idx < STRESS_NUM_STREAMS;
         ++stream_idx) {
        if (state->streams[stream_idx] != NULL) {
            (void)cuStreamDestroy(state->streams[stream_idx]);
        }
    }
    free(state->actual_sse);
    free(state->expected_sse);
    free(state->actual_materialize);
    free(state->expected_materialize);
    free(state->targets);
    free(state->input);
    free(state->program_mufu_counts);
    free(state->program_masks);
    free(state->program_sizes);
    free(state->asts);
    free(state->programs);
    free(state->sse_workspace);
    free(state->materialize_workspace);
    free(state->sse_cubin);
    free(state->materialize_cubin);
    free(state->sse_template);
    free(state->materialize_template);
    if (state->retained) {
        (void)cuCtxSetCurrent(state->previous_context);
        (void)cuDevicePrimaryCtxRelease(state->device);
    }
}

int
main(int argc, char** argv) {
    size_t iterations = 64u;
    size_t num_kernels = 4u;
    size_t asts_per_kernel = 16u;
    size_t rows = 257u;
    size_t num_inputs = 8u;
    size_t num_targets = 3u;
    size_t max_leaves = 12u;
    size_t max_unary_depth = 2u;
    size_t patch_instructions_per_ast = 192u;
    uint64_t seed = UINT64_C(0x5eca17);
    int device_ordinal = 0;
    StressState state;
    uint32_t instruction_mask = 0u;
    size_t total_program_bytes = 0u;
    size_t min_program_bytes = SIZE_MAX;
    size_t max_program_bytes = 0u;
    size_t class_counts[2] = { 0u, 0u };
    float max_materialize_absolute_errors[2] = { 0.0f, 0.0f };
    float max_materialize_relative_errors[2] = { 0.0f, 0.0f };
    float max_sse_absolute_errors[2] = { 0.0f, 0.0f };
    float max_sse_relative_errors[2] = { 0.0f, 0.0f };
    size_t arg_idx;
    size_t iteration;
    int success = 1;

    memset(&state, 0, sizeof(state));
    for (arg_idx = 1u; arg_idx < (size_t)argc; ++arg_idx) {
        const char* option = argv[arg_idx];
        const char* value;

        if (strcmp(option, "--help") == 0) {
            stress_usage(argv[0]);
            return 0;
        }
        if (arg_idx + 1u >= (size_t)argc) {
            stress_usage(argv[0]);
            return 2;
        }
        value = argv[++arg_idx];
        if (strcmp(option, "--iterations") == 0) {
            success = stress_parse_size(value, &iterations);
        } else if (strcmp(option, "--kernels") == 0) {
            success = stress_parse_size(value, &num_kernels);
        } else if (strcmp(option, "--asts-per-kernel") == 0) {
            success = stress_parse_size(value, &asts_per_kernel);
        } else if (strcmp(option, "--rows") == 0) {
            success = stress_parse_size(value, &rows);
        } else if (strcmp(option, "--inputs") == 0) {
            success = stress_parse_size(value, &num_inputs);
        } else if (strcmp(option, "--targets") == 0) {
            success = stress_parse_size(value, &num_targets);
        } else if (strcmp(option, "--max-leaves") == 0) {
            success = stress_parse_size(value, &max_leaves);
        } else if (strcmp(option, "--max-unary-depth") == 0) {
            success = stress_parse_size(value, &max_unary_depth);
        } else if (strcmp(option, "--patch-instructions-per-ast") == 0) {
            success = stress_parse_size(
                value,
                &patch_instructions_per_ast);
        } else if (strcmp(option, "--seed") == 0) {
            success = stress_parse_u64(value, &seed);
        } else if (strcmp(option, "--device") == 0) {
            success = stress_parse_int(value, &device_ordinal);
        } else {
            success = 0;
        }
        if (!success) {
            stress_usage(argv[0]);
            return 2;
        }
    }
    if (iterations == 0u ||
        num_kernels == 0u ||
        asts_per_kernel == 0u ||
        rows == 0u ||
        num_inputs == 0u ||
        num_inputs > UINT32_MAX ||
        num_targets == 0u ||
        max_leaves == 0u ||
        max_leaves > 16u ||
        max_unary_depth > 3u ||
        patch_instructions_per_ast == 0u ||
        (2u * max_leaves - 1u) * (max_unary_depth + 1u) + 2u >
            STRESS_PROGRAM_CAPACITY) {
        stress_usage(argv[0]);
        return 2;
    }
    success = stress_prepare(
        &state,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        rows,
        patch_instructions_per_ast,
        seed,
        device_ordinal);
    for (iteration = 0u;
         success && iteration < iterations;
         ++iteration) {
        success = stress_iteration(
            &state,
            iteration,
            seed,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            rows,
            max_leaves,
            max_unary_depth,
            &instruction_mask,
            &total_program_bytes,
            &min_program_bytes,
            &max_program_bytes,
            class_counts,
            max_materialize_absolute_errors,
            max_materialize_relative_errors,
            max_sse_absolute_errors,
            max_sse_relative_errors);
    }
    if (success) {
        const size_t num_asts = iterations * num_kernels * asts_per_kernel;

        printf(
            "cubin_stress status=pass seed=%" PRIu64
            " iterations=%zu kernels=%zu asts_per_kernel=%zu asts=%zu "
            "rows=%zu inputs=%zu targets=%zu program_bytes=[%zu,%zu] "
            "mean_program_bytes=%.2f instruction_mask=0x%08x\n"
            "  alu asts=%zu materialize_max_abs=%g "
            "materialize_max_rel=%g sse_max_abs=%g sse_max_rel=%g\n"
            "  mufu asts=%zu materialize_max_abs=%g "
            "materialize_max_rel=%g sse_max_abs=%g sse_max_rel=%g\n",
            seed,
            iterations,
            num_kernels,
            asts_per_kernel,
            num_asts,
            rows,
            num_inputs,
            num_targets,
            min_program_bytes,
            max_program_bytes,
            (double)total_program_bytes / (double)num_asts,
            instruction_mask,
            class_counts[0],
            (double)max_materialize_absolute_errors[0],
            (double)max_materialize_relative_errors[0],
            (double)max_sse_absolute_errors[0],
            (double)max_sse_relative_errors[0],
            class_counts[1],
            (double)max_materialize_absolute_errors[1],
            (double)max_materialize_relative_errors[1],
            (double)max_sse_absolute_errors[1],
            (double)max_sse_relative_errors[1]);
    }
    stress_destroy(&state);
    return success ? 0 : 1;
}
