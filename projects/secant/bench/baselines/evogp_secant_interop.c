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

#include <cuda.h>
#include <nvrtc.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SECANT_EVOGP_SASS_INSTRUCTIONS_PER_AST 256u
#define SECANT_EVOGP_NORMALIZED_TOLERANCE 5.0e-4

typedef struct SecantEvoGPState {
    unsigned char* program_bytes;
    size_t program_bytes_size;
    uint64_t* program_offsets;
    size_t program_offsets_size;
    const SecantAstInstruction** asts;
    float* input;
    size_t input_size;
    float* evogp_output;
    size_t evogp_output_size;
    float* cpu_output;
    float* cubin_output;
    unsigned char* cubin;
    size_t cubin_size;
    void* plan_storage;
    SecantCubinPlan* plan;
    SecantCubinRunner runner;
    CUdevice device;
    CUcontext context;
    CUdeviceptr device_input;
    CUdeviceptr device_output;
    int primary_context_retained;
    char error[4096];
} SecantEvoGPState;

typedef struct SecantEvoGPErrorStats {
    double maximum_absolute;
    double maximum_normalized;
    size_t maximum_index;
    size_t nonfinite_mismatches;
} SecantEvoGPErrorStats;

static void
secant_evogp_error_set(
    SecantEvoGPState* state,
    const char* format,
    ...
) {
    va_list arguments;

    va_start(arguments, format);
    (void)vsnprintf(state->error, sizeof(state->error), format, arguments);
    va_end(arguments);
}

static int
secant_evogp_size_parse(
    const char* text,
    size_t* value_ret
) {
    char* end = NULL;
    unsigned long long value;

    if (text == NULL || text[0] == '\0' || value_ret == NULL) {
        return 0;
    }
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0u || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
secant_evogp_checked_mul(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
secant_evogp_checked_add(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static int
secant_evogp_file_read(
    SecantEvoGPState* state,
    const char* path,
    void** data_ret,
    size_t* size_ret
) {
    FILE* file;
    long file_size;
    void* data;
    size_t bytes_read;

    *data_ret = NULL;
    *size_ret = 0u;
    file = fopen(path, "rb");
    if (file == NULL) {
        secant_evogp_error_set(state, "failed to open %s", path);
        return 0;
    }
    if (fseek(file, 0L, SEEK_END) != 0 || (file_size = ftell(file)) <= 0L || fseek(file, 0L, SEEK_SET) != 0) {
        (void)fclose(file);
        secant_evogp_error_set(state, "failed to measure %s", path);
        return 0;
    }
    data = malloc((size_t)file_size);
    if (data == NULL) {
        (void)fclose(file);
        secant_evogp_error_set(state, "allocation failed while reading %s", path);
        return 0;
    }
    bytes_read = fread(data, 1u, (size_t)file_size, file);
    if (fclose(file) != 0 || bytes_read != (size_t)file_size) {
        free(data);
        secant_evogp_error_set(state, "failed to read %s", path);
        return 0;
    }
    *data_ret = data;
    *size_ret = (size_t)file_size;
    return 1;
}

static int
secant_evogp_inputs_read(
    SecantEvoGPState* state,
    const char* programs_path,
    const char* offsets_path,
    const char* input_path,
    const char* expected_path,
    size_t num_asts,
    size_t num_inputs,
    size_t num_rows
) {
    size_t expected_offsets_size;
    size_t input_elements;
    size_t output_elements;
    size_t expected_input_size;
    size_t expected_output_size;
    size_t num_offsets;
    size_t ast_idx;

    if (!secant_evogp_checked_add(num_asts, 1u, &num_offsets) ||
        !secant_evogp_checked_mul(num_offsets, sizeof(uint64_t), &expected_offsets_size) ||
        !secant_evogp_checked_mul(num_inputs, num_rows, &input_elements) ||
        !secant_evogp_checked_mul(num_asts, num_rows, &output_elements) ||
        !secant_evogp_checked_mul(input_elements, sizeof(float), &expected_input_size) ||
        !secant_evogp_checked_mul(output_elements, sizeof(float), &expected_output_size)) {
        secant_evogp_error_set(state, "input dimensions overflow size_t");
        return 0;
    }
    if (!secant_evogp_file_read(
            state,
            programs_path,
            (void**)&state->program_bytes,
            &state->program_bytes_size) ||
        !secant_evogp_file_read(
            state,
            offsets_path,
            (void**)&state->program_offsets,
            &state->program_offsets_size) ||
        !secant_evogp_file_read(state, input_path, (void**)&state->input, &state->input_size) ||
        !secant_evogp_file_read(
            state,
            expected_path,
            (void**)&state->evogp_output,
            &state->evogp_output_size)) {
        return 0;
    }
    if (state->program_offsets_size != expected_offsets_size || state->input_size != expected_input_size ||
        state->evogp_output_size != expected_output_size || state->program_offsets[0] != 0u ||
        state->program_offsets[num_asts] != state->program_bytes_size) {
        secant_evogp_error_set(state, "interop file sizes or terminal program offset are inconsistent");
        return 0;
    }
    state->asts = (const SecantAstInstruction**)malloc(num_asts * sizeof(*state->asts));
    state->cpu_output = (float*)malloc(expected_output_size);
    state->cubin_output = (float*)malloc(expected_output_size);
    if (state->asts == NULL || state->cpu_output == NULL || state->cubin_output == NULL) {
        secant_evogp_error_set(state, "interop host allocation failed");
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        const uint64_t begin = state->program_offsets[ast_idx];
        const uint64_t end = state->program_offsets[ast_idx + 1u];

        if (begin >= end || end > state->program_bytes_size) {
            secant_evogp_error_set(state, "invalid program offsets for AST %zu", ast_idx);
            return 0;
        }
        state->asts[ast_idx] = state->program_bytes + (size_t)begin;
    }
    return 1;
}

static int
secant_evogp_cpu_run(
    SecantEvoGPState* state,
    size_t num_asts,
    size_t num_inputs,
    size_t num_rows
) {
    SecantCpuMaterializeRun run = secant_cpu_materialize_run_init();
    SecantResult result;
    size_t input_elements;
    size_t output_elements;

    if (!secant_evogp_checked_mul(num_inputs, num_rows, &input_elements) ||
        !secant_evogp_checked_mul(num_asts, num_rows, &output_elements)) {
        secant_evogp_error_set(state, "CPU materialize dimensions overflow size_t");
        return 0;
    }
    run.programs.asts.items = state->asts;
    run.programs.asts.count = num_asts;
    run.num_inputs = num_inputs;
    run.input.data = state->input;
    run.input.num_elements = input_elements;
    run.input.leading_dimension = num_rows;
    run.num_rows = num_rows;
    run.output.data = state->cpu_output;
    run.output.num_elements = output_elements;
    run.output.leading_dimension = num_rows;
    result = secant_cpu_run(&run.header);
    if (result != SECANT_SUCCESS) {
        secant_evogp_error_set(state, "Secant CPU failed: %s", secant_result_to_string(result));
        return 0;
    }
    return 1;
}

static int
secant_evogp_cuda_initialize(
    SecantEvoGPState* state,
    uint32_t* target_sm_ret
) {
    int major;
    int minor;

    if (cuInit(0u) != CUDA_SUCCESS || cuDeviceGet(&state->device, 0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, state->device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, state->device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&state->context, state->device) != CUDA_SUCCESS) {
        secant_evogp_error_set(state, "CUDA primary-context initialization failed");
        return 0;
    }
    state->primary_context_retained = 1;
    if (cuCtxSetCurrent(state->context) != CUDA_SUCCESS) {
        secant_evogp_error_set(state, "failed to make the CUDA primary context current");
        return 0;
    }
    if (major < 0 || minor < 0 || major > 99 || minor > 9) {
        secant_evogp_error_set(state, "invalid CUDA compute capability %d.%d", major, minor);
        return 0;
    }
    *target_sm_ret = (uint32_t)(major * 10 + minor);
    return 1;
}

static int
secant_evogp_cubin_compile(
    SecantEvoGPState* state,
    const SecantCubinRecipeHeader* recipe,
    uint32_t target_sm
) {
    char architecture[64];
    const char* options[4];
    char* source = NULL;
    unsigned char* cubin = NULL;
    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_result;
    SecantResult result;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t log_size = 0u;
    int architecture_bytes;
    int success = 0;

    result = secant_cubin_source_size(recipe, &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        secant_evogp_error_set(state, "CUBIN source measurement failed: %s", secant_result_to_string(result));
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        secant_evogp_error_set(state, "CUBIN source allocation failed");
        return 0;
    }
    result = secant_cubin_source_write(recipe, source, source_size);
    if (result != SECANT_SUCCESS) {
        secant_evogp_error_set(state, "CUBIN source generation failed: %s", secant_result_to_string(result));
    }
    architecture_bytes = snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%u", target_sm);
    if (result == SECANT_SUCCESS &&
        (architecture_bytes < 0 || (size_t)architecture_bytes >= sizeof(architecture))) {
        secant_evogp_error_set(state, "invalid target SM %u", target_sm);
        result = SECANT_ERROR_INVALID_VALUE;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    nvrtc_result = result == SECANT_SUCCESS
        ? nvrtcCreateProgram(&program, source, "secant_evogp_materialize.cu", 0, NULL, NULL)
        : NVRTC_ERROR_INVALID_INPUT;
    if (nvrtc_result == NVRTC_SUCCESS) {
        nvrtc_result = nvrtcCompileProgram(program, (int)(sizeof(options) / sizeof(options[0])), options);
    }
    if (result == SECANT_SUCCESS && nvrtc_result != NVRTC_SUCCESS) {
        if (program != NULL && nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size != 0u) {
            char* log = (char*)malloc(log_size);

            if (log != NULL && nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                secant_evogp_error_set(state, "NVRTC failed:\n%s", log);
            } else {
                secant_evogp_error_set(state, "NVRTC failed: %s", nvrtcGetErrorString(nvrtc_result));
            }
            free(log);
        } else {
            secant_evogp_error_set(state, "NVRTC failed: %s", nvrtcGetErrorString(nvrtc_result));
        }
    } else if (result == SECANT_SUCCESS &&
        (nvrtcGetCUBINSize(program, &cubin_size) != NVRTC_SUCCESS || cubin_size == 0u)) {
        secant_evogp_error_set(state, "NVRTC returned an empty CUBIN");
    } else if (result == SECANT_SUCCESS) {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin == NULL) {
            secant_evogp_error_set(state, "CUBIN allocation failed");
        } else if (nvrtcGetCUBIN(program, (char*)cubin) != NVRTC_SUCCESS) {
            secant_evogp_error_set(state, "NVRTC CUBIN retrieval failed");
        } else {
            success = 1;
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    free(source);
    if (!success) {
        free(cubin);
        return 0;
    }
    state->cubin = cubin;
    state->cubin_size = cubin_size;
    return 1;
}

static int
secant_evogp_cubin_prepare(
    SecantEvoGPState* state,
    size_t num_asts,
    size_t num_inputs,
    uint32_t target_sm
) {
    SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    SecantResult result;
    size_t patch_capacity;
    size_t plan_storage_size = 0u;

    if (!secant_evogp_checked_mul(num_asts, SECANT_EVOGP_SASS_INSTRUCTIONS_PER_AST, &patch_capacity)) {
        secant_evogp_error_set(state, "patch capacity overflow");
        return 0;
    }
    recipe.num_kernels = 1u;
    recipe.asts_per_kernel = num_asts;
    recipe.num_inputs = num_inputs;
    recipe.patch_capacity_instructions = patch_capacity;
    if (!secant_evogp_cubin_compile(state, &recipe.header, target_sm)) {
        return 0;
    }
    result = secant_cubin_plan_storage_size(
        &recipe.header,
        state->cubin,
        state->cubin_size,
        &plan_storage_size);
    if (result != SECANT_SUCCESS) {
        secant_evogp_error_set(state, "Secant plan measurement failed: %s", secant_result_to_string(result));
        return 0;
    }
    state->plan_storage = malloc(plan_storage_size);
    if (state->plan_storage == NULL) {
        secant_evogp_error_set(state, "plan storage allocation failed");
        return 0;
    }
    result = secant_cubin_plan_init(
        &recipe.header,
        state->cubin,
        state->cubin_size,
        state->plan_storage,
        plan_storage_size,
        &state->plan);
    if (result != SECANT_SUCCESS) {
        secant_evogp_error_set(state, "Secant plan initialization failed: %s", secant_result_to_string(result));
        return 0;
    }
    options.num_workers = 1u;
    options.num_streams = 1u;
    result = secant_cubin_runner_create(
        state->plan,
        state->cubin,
        state->cubin_size,
        &options,
        &state->runner);
    if (result != SECANT_SUCCESS) {
        secant_evogp_error_set(state, "Secant runner creation failed: %s", secant_result_to_string(result));
        return 0;
    }
    return 1;
}

static int
secant_evogp_cubin_run(
    SecantEvoGPState* state,
    size_t num_asts,
    size_t num_inputs,
    size_t num_rows
) {
    SecantCubinMaterializeRun run = secant_cubin_materialize_run_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantResult result;
    size_t input_elements;
    size_t output_elements;
    size_t input_bytes;
    size_t output_bytes;

    if (!secant_evogp_checked_mul(num_inputs, num_rows, &input_elements) ||
        !secant_evogp_checked_mul(num_asts, num_rows, &output_elements) ||
        !secant_evogp_checked_mul(input_elements, sizeof(float), &input_bytes) ||
        !secant_evogp_checked_mul(output_elements, sizeof(float), &output_bytes)) {
        secant_evogp_error_set(state, "CUBIN materialize dimensions overflow size_t");
        return 0;
    }
    if (cuMemAlloc(&state->device_input, input_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&state->device_output, output_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(state->device_input, state->input, input_bytes) != CUDA_SUCCESS) {
        secant_evogp_error_set(state, "CUDA device allocation or input upload failed");
        return 0;
    }
    run.programs.asts.items = state->asts;
    run.programs.asts.count = num_asts;
    run.input.address = (uintptr_t)state->device_input;
    run.input.num_elements = input_elements;
    run.input.leading_dimension = num_rows;
    run.num_rows = num_rows;
    run.output.address = (uintptr_t)state->device_output;
    run.output.num_elements = output_elements;
    run.output.leading_dimension = num_rows;
    run.output_module_stride = 0u;
    result = secant_cubin_runner_run_materialize(state->runner, &run, &stats);
    if (result != SECANT_SUCCESS) {
        secant_evogp_error_set(state, "Secant CUBIN run failed: %s", secant_result_to_string(result));
        return 0;
    }
    if (cuMemcpyDtoH(state->cubin_output, state->device_output, output_bytes) != CUDA_SUCCESS) {
        secant_evogp_error_set(state, "CUDA output download failed");
        return 0;
    }
    printf(
        "secant_cubin modules=%zu asts=%zu runtime_seconds=%.9f total_seconds=%.9f\n",
        stats.num_modules,
        stats.num_asts,
        stats.runtime_seconds,
        stats.total_seconds);
    return 1;
}

static SecantEvoGPErrorStats
secant_evogp_compare(
    const float* actual,
    const float* reference,
    size_t count
) {
    SecantEvoGPErrorStats stats;
    size_t index;

    memset(&stats, 0, sizeof(stats));
    for (index = 0u; index < count; ++index) {
        double absolute;
        double normalized;

        if (actual[index] == reference[index]) {
            continue;
        }
        if (!isfinite(actual[index]) || !isfinite(reference[index])) {
            ++stats.nonfinite_mismatches;
            stats.maximum_absolute = INFINITY;
            stats.maximum_normalized = INFINITY;
            stats.maximum_index = index;
            continue;
        }
        absolute = fabs((double)actual[index] - (double)reference[index]);
        normalized = absolute / (1.0 + fabs((double)reference[index]));
        if (absolute > stats.maximum_absolute) {
            stats.maximum_absolute = absolute;
        }
        if (normalized > stats.maximum_normalized) {
            stats.maximum_normalized = normalized;
            stats.maximum_index = index;
        }
    }
    return stats;
}

static int
secant_evogp_execute(
    SecantEvoGPState* state,
    const char* programs_path,
    const char* offsets_path,
    const char* input_path,
    const char* expected_path,
    size_t num_asts,
    size_t num_inputs,
    size_t num_rows
) {
    SecantEvoGPErrorStats evogp_cpu;
    SecantEvoGPErrorStats cubin_cpu;
    SecantEvoGPErrorStats cubin_evogp;
    uint32_t target_sm;
    size_t marker_registers;
    size_t output_elements;

    if (!secant_evogp_checked_add(num_inputs, num_asts, &marker_registers) || marker_registers > 256u) {
        secant_evogp_error_set(state, "materialize marker registers require num_inputs + num_asts <= 256");
        return 0;
    }
    if (!secant_evogp_inputs_read(
            state,
            programs_path,
            offsets_path,
            input_path,
            expected_path,
            num_asts,
            num_inputs,
            num_rows) ||
        !secant_evogp_cpu_run(state, num_asts, num_inputs, num_rows) ||
        !secant_evogp_cuda_initialize(state, &target_sm) ||
        !secant_evogp_cubin_prepare(state, num_asts, num_inputs, target_sm) ||
        !secant_evogp_cubin_run(state, num_asts, num_inputs, num_rows)) {
        return 0;
    }
    if (!secant_evogp_checked_mul(num_asts, num_rows, &output_elements)) {
        secant_evogp_error_set(state, "comparison dimensions overflow size_t");
        return 0;
    }
    evogp_cpu = secant_evogp_compare(state->cpu_output, state->evogp_output, output_elements);
    cubin_cpu = secant_evogp_compare(state->cubin_output, state->cpu_output, output_elements);
    cubin_evogp = secant_evogp_compare(state->cubin_output, state->evogp_output, output_elements);
    printf(
        "comparison=evogp_vs_secant_cpu max_abs_error=%.9g max_normalized_error=%.9g "
        "worst_ast=%zu worst_row=%zu nonfinite_mismatches=%zu\n",
        evogp_cpu.maximum_absolute,
        evogp_cpu.maximum_normalized,
        evogp_cpu.maximum_index / num_rows,
        evogp_cpu.maximum_index % num_rows,
        evogp_cpu.nonfinite_mismatches);
    printf(
        "comparison=secant_cubin_vs_secant_cpu max_abs_error=%.9g max_normalized_error=%.9g "
        "worst_ast=%zu worst_row=%zu nonfinite_mismatches=%zu\n",
        cubin_cpu.maximum_absolute,
        cubin_cpu.maximum_normalized,
        cubin_cpu.maximum_index / num_rows,
        cubin_cpu.maximum_index % num_rows,
        cubin_cpu.nonfinite_mismatches);
    printf(
        "comparison=secant_cubin_vs_evogp max_abs_error=%.9g max_normalized_error=%.9g "
        "worst_ast=%zu worst_row=%zu nonfinite_mismatches=%zu\n",
        cubin_evogp.maximum_absolute,
        cubin_evogp.maximum_normalized,
        cubin_evogp.maximum_index / num_rows,
        cubin_evogp.maximum_index % num_rows,
        cubin_evogp.nonfinite_mismatches);
    if (evogp_cpu.maximum_normalized > SECANT_EVOGP_NORMALIZED_TOLERANCE ||
        cubin_cpu.maximum_normalized > SECANT_EVOGP_NORMALIZED_TOLERANCE ||
        cubin_evogp.maximum_normalized > SECANT_EVOGP_NORMALIZED_TOLERANCE ||
        evogp_cpu.nonfinite_mismatches != 0u || cubin_cpu.nonfinite_mismatches != 0u ||
        cubin_evogp.nonfinite_mismatches != 0u) {
        secant_evogp_error_set(
            state,
            "comparison exceeded normalized tolerance %.9g",
            SECANT_EVOGP_NORMALIZED_TOLERANCE);
        return 0;
    }
    return 1;
}

static void
secant_evogp_cleanup(
    SecantEvoGPState* state
) {
    if (state->context != NULL) {
        (void)cuCtxSetCurrent(state->context);
    }
    if (state->runner != NULL) {
        (void)secant_cubin_runner_destroy(state->runner);
    }
    if (state->device_output != 0u) {
        (void)cuMemFree(state->device_output);
    }
    if (state->device_input != 0u) {
        (void)cuMemFree(state->device_input);
    }
    if (state->primary_context_retained) {
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(state->device);
    }
    free(state->plan_storage);
    free(state->cubin);
    free(state->cubin_output);
    free(state->cpu_output);
    free(state->evogp_output);
    free(state->input);
    free(state->asts);
    free(state->program_offsets);
    free(state->program_bytes);
}

int
main(
    int argc,
    char** argv
) {
    SecantEvoGPState state;
    size_t num_asts;
    size_t num_inputs;
    size_t num_rows;
    int success = 0;

    memset(&state, 0, sizeof(state));
    if (argc != 8) {
        fprintf(
            stderr,
            "usage: %s <programs.bin> <offsets.u64> <input.f32> <evogp-output.f32> "
            "<num-asts> <num-inputs> <num-rows>\n",
            argv[0]);
    } else if (!secant_evogp_size_parse(argv[5], &num_asts) ||
        !secant_evogp_size_parse(argv[6], &num_inputs) ||
        !secant_evogp_size_parse(argv[7], &num_rows)) {
        fprintf(stderr, "invalid positive dimensions\n");
    } else {
        success = secant_evogp_execute(
            &state,
            argv[1],
            argv[2],
            argv[3],
            argv[4],
            num_asts,
            num_inputs,
            num_rows);
        if (!success) {
            fprintf(stderr, "secant EvoGP interoperability failed: %s\n", state.error);
        }
    }
    secant_evogp_cleanup(&state);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

#undef SECANT_EVOGP_NORMALIZED_TOLERANCE
#undef SECANT_EVOGP_SASS_INSTRUCTIONS_PER_AST
