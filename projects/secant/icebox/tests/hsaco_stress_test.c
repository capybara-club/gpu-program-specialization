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
#include "secant_cpu.h"
#include "secant_hsaco.h"
#include "support/ast_stress.h"
#include "support/hip_runtime.h"

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STRESS_PROGRAM_CAPACITY SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY
#define STRESS_NUM_STREAMS 4u
#define STRESS_TILE_ROWS 128u
#define STRESS_THREADS_PER_BLOCK 128u

typedef struct StressState {
    hipStream_t streams[STRESS_NUM_STREAMS];
    float* device_input;
    float* device_targets;
    float* device_materialize;
    float* device_sse;
    unsigned char* materialize_template;
    unsigned char* sse_template;
    unsigned char* materialize_hsaco;
    unsigned char* sse_hsaco;
    size_t materialize_hsaco_size;
    size_t sse_hsaco_size;
    void* materialize_workspace;
    void* sse_workspace;
    SecantHsacoPlan* materialize_plan;
    SecantHsacoPlan* sse_plan;
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
stress_architecture(int device_ordinal, uint32_t* gfx_arch_ret) {
    hipDeviceProp_t properties;
    char* end;
    unsigned long gfx_arch;

    if (hipSetDevice(device_ordinal) != hipSuccess ||
        hipGetDeviceProperties(
            &properties,
            device_ordinal) != hipSuccess ||
        strncmp(properties.gcnArchName, "gfx", 3u) != 0) {
        return 0;
    }
    gfx_arch = strtoul(properties.gcnArchName + 3u, &end, 10);
    if (end == properties.gcnArchName + 3u ||
        gfx_arch > UINT32_MAX) {
        return 0;
    }
    *gfx_arch_ret = (uint32_t)gfx_arch;
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
    uint32_t gfx_arch,
    unsigned char** hsaco_ret,
    size_t* hsaco_size_ret
) {
    const SecantHsacoMaterializeRecipe materialize_recipe = {
        num_kernels,
        asts_per_kernel,
        num_inputs,
        patch_capacity_instructions
    };
    const SecantHsacoSSERecipe sse_recipe = {
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        STRESS_TILE_ROWS,
        STRESS_THREADS_PER_BLOCK,
        patch_capacity_instructions,
        SECANT_SSE_REDUCTION_MODE_ATOMIC
    };
    char architecture[64];
    const char* options[3];
    char* source = NULL;
    char* log = NULL;
    unsigned char* hsaco = NULL;
    hiprtcProgram program = NULL;
    hiprtcResult hiprtc_result = HIPRTC_SUCCESS;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    size_t hsaco_size = 0u;
    size_t log_size = 0u;
    int architecture_bytes;
    int success = 1;

    *hsaco_ret = NULL;
    *hsaco_size_ret = 0u;
    result = sse
        ? secant_hsaco_sse_source_generate(
            &sse_recipe,
            NULL,
            0u,
            &source_size)
        : secant_hsaco_materialize_source_generate(
            &materialize_recipe,
            NULL,
            0u,
            &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        fprintf(
            stderr,
            "%s HIP source measurement failed: SecantResult(%d)\n",
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
            ? secant_hsaco_sse_source_generate(
                &sse_recipe,
                source,
                source_size,
                &written_size)
            : secant_hsaco_materialize_source_generate(
                &materialize_recipe,
                source,
                source_size,
                &written_size);
        if (result != SECANT_SUCCESS || written_size != source_size) {
            fprintf(
                stderr,
                "%s HIP source generation failed: SecantResult(%d)\n",
                sse ? "SSE" : "materialize",
                (int)result);
            success = 0;
        }
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--offload-arch=gfx%u",
        gfx_arch);
    if (architecture_bytes < 0 ||
        (size_t)architecture_bytes >= sizeof(architecture)) {
        success = 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "-O1";
    if (success) {
        hiprtc_result = hiprtcCreateProgram(
            &program,
            source,
            sse
                ? "secant_hsaco_stress_sse.hip"
                : "secant_hsaco_stress_materialize.hip",
            0,
            NULL,
            NULL);
        if (hiprtc_result == HIPRTC_SUCCESS) {
            hiprtc_result = hiprtcCompileProgram(
                program,
                (int)(sizeof(options) / sizeof(options[0])),
                options);
        }
        if (hiprtc_result != HIPRTC_SUCCESS) {
            if (program != NULL &&
                hiprtcGetProgramLogSize(program, &log_size) ==
                    HIPRTC_SUCCESS &&
                log_size != 0u) {
                log = (char*)malloc(log_size);
                if (log != NULL &&
                    hiprtcGetProgramLog(program, log) == HIPRTC_SUCCESS) {
                    fprintf(stderr, "%s\n", log);
                }
            }
            fprintf(
                stderr,
                "%s HIPRTC compilation failed: %s\n",
                sse ? "SSE" : "materialize",
                hiprtcGetErrorString(hiprtc_result));
            success = 0;
        }
    }
    if (success &&
        (hiprtcGetCodeSize(program, &hsaco_size) != HIPRTC_SUCCESS ||
         hsaco_size == 0u)) {
        success = 0;
    }
    if (success) {
        hsaco = (unsigned char*)malloc(hsaco_size);
        if (hsaco == NULL ||
            hiprtcGetCode(program, (char*)hsaco) != HIPRTC_SUCCESS) {
            success = 0;
        }
    }
    if (program != NULL) {
        (void)hiprtcDestroyProgram(&program);
    }
    free(log);
    free(source);
    if (!success) {
        free(hsaco);
        return 0;
    }
    *hsaco_ret = hsaco;
    *hsaco_size_ret = hsaco_size;
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
    const SecantHsacoMaterializeRecipe materialize_recipe = {
        num_kernels,
        asts_per_kernel,
        num_inputs,
        patch_capacity_instructions
    };
    const SecantHsacoSSERecipe sse_recipe = {
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        STRESS_TILE_ROWS,
        STRESS_THREADS_PER_BLOCK,
        patch_capacity_instructions,
        SECANT_SSE_REDUCTION_MODE_ATOMIC
    };
    size_t workspace_size = 0u;
    SecantResult result;

    result = secant_hsaco_materialize_inspect(
        &materialize_recipe,
        state->materialize_template,
        state->materialize_hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &state->materialize_plan);
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
    result = secant_hsaco_materialize_inspect(
        &materialize_recipe,
        state->materialize_template,
        state->materialize_hsaco_size,
        state->materialize_workspace,
        workspace_size,
        &workspace_size,
        &state->materialize_plan);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize inspect failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    workspace_size = 0u;
    result = secant_hsaco_sse_inspect(
        &sse_recipe,
        state->sse_template,
        state->sse_hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &state->sse_plan);
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
    result = secant_hsaco_sse_inspect(
        &sse_recipe,
        state->sse_template,
        state->sse_hsaco_size,
        state->sse_workspace,
        workspace_size,
        &workspace_size,
        &state->sse_plan);
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
    uint32_t gfx_arch;
    size_t stream_idx;

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
    if (!stress_architecture(device_ordinal, &gfx_arch) ||
        !stress_compile_template(
            0,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions,
            gfx_arch,
            &state->materialize_template,
            &state->materialize_hsaco_size) ||
        !stress_compile_template(
            1,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions,
            gfx_arch,
            &state->sse_template,
            &state->sse_hsaco_size) ||
        !stress_inspect_templates(
            state,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions)) {
        return 0;
    }
    state->materialize_hsaco =
        (unsigned char*)malloc(state->materialize_hsaco_size);
    state->sse_hsaco =
        (unsigned char*)malloc(state->sse_hsaco_size);
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
    if (state->materialize_hsaco == NULL ||
        state->sse_hsaco == NULL ||
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
    for (stream_idx = 0u; stream_idx < STRESS_NUM_STREAMS; ++stream_idx) {
        if (hipStreamCreateWithFlags(
                state->streams + stream_idx,
                hipStreamNonBlocking) != hipSuccess) {
            return 0;
        }
    }
    if (hipMalloc(
            (void**)&state->device_input,
            input_count * sizeof(float)) != hipSuccess ||
        hipMalloc(
            (void**)&state->device_targets,
            target_count * sizeof(float)) != hipSuccess ||
        hipMalloc(
            (void**)&state->device_materialize,
            materialize_count * sizeof(float)) != hipSuccess ||
        hipMalloc(
            (void**)&state->device_sse,
            sse_count * sizeof(float)) != hipSuccess ||
        hipMemcpy(
            state->device_input,
            state->input,
            input_count * sizeof(float),
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(
            state->device_targets,
            state->targets,
            target_count * sizeof(float),
            hipMemcpyHostToDevice) != hipSuccess) {
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
    size_t* total_program_instructions,
    size_t* min_program_instructions,
    size_t* max_program_instructions,
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
    SecantResult hsaco_result;
    SecantResult hip_result;
    SecantResult cpu_result;
    size_t ast_idx;
    size_t kernel_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program =
            state->programs + ast_idx * STRESS_PROGRAM_CAPACITY;
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
        *total_program_instructions += state->program_sizes[ast_idx];
        if (state->program_sizes[ast_idx] < *min_program_instructions) {
            *min_program_instructions = state->program_sizes[ast_idx];
        }
        if (state->program_sizes[ast_idx] > *max_program_instructions) {
            *max_program_instructions = state->program_sizes[ast_idx];
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
        state->materialize_hsaco,
        state->materialize_template,
        state->materialize_hsaco_size);
    memcpy(
        state->sse_hsaco,
        state->sse_template,
        state->sse_hsaco_size);
    hsaco_result = secant_hsaco_specialize_into(
        state->materialize_plan,
        secant_test_ast_stress_routines,
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
        state->asts,
        num_asts,
        state->materialize_hsaco,
        state->materialize_hsaco_size);
    if (hsaco_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize specialization failed seed=%" PRIu64
            " iteration=%zu: SecantResult(%d)\n",
            seed,
            iteration,
            (int)hsaco_result);
        return 0;
    }
    hsaco_result = secant_hsaco_specialize_into(
        state->sse_plan,
        secant_test_ast_stress_routines,
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
        state->asts,
        num_asts,
        state->sse_hsaco,
        state->sse_hsaco_size);
    if (hsaco_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "SSE specialization failed seed=%" PRIu64
            " iteration=%zu: SecantResult(%d)\n",
            seed,
            iteration,
            (int)hsaco_result);
        return 0;
    }
    hip_result = secant_test_hip_module_load(
        state->materialize_hsaco,
        state->materialize_hsaco_size,
        "secant_hsaco_materialize",
        num_kernels,
        &materialize_module,
        materialize_functions);
    if (hip_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize module load failed seed=%" PRIu64
            " iteration=%zu: %s\n",
            seed,
            iteration,
            secant_test_hip_result_to_string(hip_result));
        return 0;
    }
    hip_result = secant_test_hip_module_load(
        state->sse_hsaco,
        state->sse_hsaco_size,
        "secant_hsaco_sse",
        num_kernels,
        &sse_module,
        sse_functions);
    if (hip_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "SSE module load failed seed=%" PRIu64
            " iteration=%zu: %s\n",
            seed,
            iteration,
            secant_test_hip_result_to_string(hip_result));
        success = 0;
    }
    if (success &&
        (hipMemset(
             state->device_materialize,
             0xff,
             materialize_bytes) != hipSuccess ||
         hipMemset(
             state->device_sse,
             0,
             sse_bytes) != hipSuccess ||
         hipDeviceSynchronize() != hipSuccess)) {
        success = 0;
    }
    for (kernel_idx = 0u;
         kernel_idx < num_kernels && success;
         ++kernel_idx) {
        hip_result = secant_test_hip_run_static_column_materialize(
            materialize_functions[kernel_idx],
            num_inputs,
            asts_per_kernel,
            state->device_input,
            num_inputs * rows,
            rows,
            rows,
            state->streams[kernel_idx % STRESS_NUM_STREAMS],
            state->device_materialize +
                kernel_idx * asts_per_kernel * rows,
            asts_per_kernel * rows,
            rows);
        if (hip_result == SECANT_SUCCESS) {
            hip_result = secant_test_hip_run_static_column_sse(
                sse_functions[kernel_idx],
                num_inputs,
                asts_per_kernel,
                num_targets,
                STRESS_TILE_ROWS,
                STRESS_THREADS_PER_BLOCK,
                state->device_input,
                num_inputs * rows,
                rows,
                state->device_targets,
                num_targets * rows,
                rows,
                rows,
                state->streams[kernel_idx % STRESS_NUM_STREAMS],
                state->device_sse +
                    kernel_idx * asts_per_kernel * num_targets,
                asts_per_kernel * num_targets,
                num_targets);
        }
        if (hip_result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "kernel launch failed seed=%" PRIu64
                " iteration=%zu kernel=%zu: %s\n",
                seed,
                iteration,
                kernel_idx,
                secant_test_hip_result_to_string(hip_result));
            success = 0;
        }
    }
    if (success &&
        (hipDeviceSynchronize() != hipSuccess ||
         hipMemcpy(
             state->actual_materialize,
             state->device_materialize,
             materialize_bytes,
             hipMemcpyDeviceToHost) != hipSuccess ||
         hipMemcpy(
             state->actual_sse,
             state->device_sse,
             sse_bytes,
             hipMemcpyDeviceToHost) != hipSuccess)) {
        fprintf(
            stderr,
            "HIP synchronization or copy failed seed=%" PRIu64
            " iteration=%zu\n",
            seed,
            iteration);
        success = 0;
    }
    if (success) {
        success = secant_test_ast_stress_compare(
            "hsaco",
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
            "hsaco",
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
        hipModuleUnload((hipModule_t)sse_module) != hipSuccess) {
        success = 0;
    }
    if (materialize_module != NULL &&
        hipModuleUnload((hipModule_t)materialize_module) != hipSuccess) {
        success = 0;
    }
    return success;
}

static void
stress_destroy(StressState* state) {
    size_t stream_idx;

    if (state->device_sse != NULL) {
        (void)hipFree(state->device_sse);
    }
    if (state->device_materialize != NULL) {
        (void)hipFree(state->device_materialize);
    }
    if (state->device_targets != NULL) {
        (void)hipFree(state->device_targets);
    }
    if (state->device_input != NULL) {
        (void)hipFree(state->device_input);
    }
    for (stream_idx = 0u;
         stream_idx < STRESS_NUM_STREAMS;
         ++stream_idx) {
        if (state->streams[stream_idx] != NULL) {
            (void)hipStreamDestroy(state->streams[stream_idx]);
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
    free(state->sse_hsaco);
    free(state->materialize_hsaco);
    free(state->sse_template);
    free(state->materialize_template);
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
    size_t total_program_instructions = 0u;
    size_t min_program_instructions = SIZE_MAX;
    size_t max_program_instructions = 0u;
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
            &total_program_instructions,
            &min_program_instructions,
            &max_program_instructions,
            class_counts,
            max_materialize_absolute_errors,
            max_materialize_relative_errors,
            max_sse_absolute_errors,
            max_sse_relative_errors);
    }
    if (success) {
        const size_t num_asts =
            iterations * num_kernels * asts_per_kernel;

        printf(
            "hsaco_stress status=pass seed=%" PRIu64
            " iterations=%zu kernels=%zu asts_per_kernel=%zu asts=%zu "
            "rows=%zu inputs=%zu targets=%zu program_instructions=[%zu,%zu] "
            "mean_program_instructions=%.2f instruction_mask=0x%08x\n"
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
            min_program_instructions,
            max_program_instructions,
            (double)total_program_instructions / (double)num_asts,
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
