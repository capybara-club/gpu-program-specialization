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
#define _POSIX_C_SOURCE 200809L

#include "bench_ast.h"
#include "bench_secant_api.h"
#include "secant.h"

#include <cuda.h>
#include <nvrtc.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_ERROR_BYTES 16384u

typedef struct BenchmarkOptions {
    SecantBenchAstMode ast_mode;
    SecantBenchCSEMode cse_mode;
    size_t num_modules;
    size_t num_workers;
    size_t num_streams;
    size_t kernels_per_module;
    size_t num_input_columns;
    size_t num_input_constants;
    size_t num_settings;
    size_t num_rows;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_instructions_per_ast;
    size_t optimizer_iterations;
    size_t warmups;
    size_t iterations;
    float perturbation_scale;
    float perturbation_decay;
    uint32_t seed;
    uint32_t target_sm;
    int device_ordinal;
} BenchmarkOptions;

static int
benchmark_checked_mul(size_t left, size_t right, size_t* result_ret) {
    if (left != 0u && right > SIZE_MAX / left) {
        return 0;
    }
    *result_ret = left * right;
    return 1;
}

static double
benchmark_seconds_get(void) {
    struct timespec value;

    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static float
benchmark_value(size_t outer, size_t inner, uint32_t seed) {
    const uint32_t hash = secant_bench_ast_hash32(
        (uint32_t)outer * UINT32_C(0x9e3779b9) ^
        (uint32_t)inner * UINT32_C(0x85ebca6b) ^ seed * UINT32_C(0xc2b2ae35));

    return ((float)(hash & UINT32_C(0xffff)) / 65535.0f) * 1.5f - 0.75f;
}

static void
benchmark_options_default(BenchmarkOptions* options) {
    memset(options, 0, sizeof(*options));
    options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
    options->cse_mode = SECANT_BENCH_CSE_DISTINCT;
    options->num_modules = 64u;
    options->num_workers = 24u;
    options->num_streams = 8u;
    options->kernels_per_module = 64u;
    options->num_input_columns = 4u;
    options->num_input_constants = 4u;
    options->num_settings = 8192u;
    options->num_rows = 10000u;
    options->tile_rows = 128u;
    options->threads_per_block = 128u;
    options->patch_instructions_per_ast = 64u;
    options->optimizer_iterations = 4u;
    options->warmups = 1u;
    options->iterations = 3u;
    options->perturbation_scale = 1.0f;
    options->perturbation_decay = 0.5f;
    options->seed = 1u;
    options->target_sm = 0u;
    options->device_ordinal = 0;
}

static void
benchmark_usage(const char* program) {
    fprintf(stderr,
        "usage: %s [--ast-mode simple|alu|mufu] [--cse shared|distinct] [--modules N] [--workers N] "
        "[--streams N] [--kernels N] [--columns N] [--constants N] [--settings N] [--rows N] "
        "[--tile-rows 128|256] [--threads N] [--patch-instructions-per-ast N] "
        "[--optimizer-iterations N] [--scale F] [--decay F] [--warmups N] [--iterations N] "
        "[--seed N] [--target-sm NN] [--device N]\n",
        program);
}

static int
benchmark_parse_size(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' || value == 0u || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
benchmark_parse_u32(const char* text, uint32_t* value_ret) {
    char* end = NULL;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' || value > UINT32_MAX) {
        return 0;
    }
    *value_ret = (uint32_t)value;
    return 1;
}

static int
benchmark_parse_int(const char* text, int* value_ret) {
    char* end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' || value < 0 || value > INT32_MAX) {
        return 0;
    }
    *value_ret = (int)value;
    return 1;
}

static int
benchmark_parse_float(const char* text, float* value_ret) {
    char* end = NULL;
    float value;

    errno = 0;
    value = strtof(text, &end);
    if (errno != 0 || text == end || *end != '\0' || !isfinite(value)) {
        return 0;
    }
    *value_ret = value;
    return 1;
}

static int
benchmark_options_parse(int argc, char** argv, BenchmarkOptions* options) {
    int arg_idx;

    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        const char* name = argv[arg_idx];
        const char* value;

        if (strcmp(name, "--help") == 0) {
            return -1;
        }
        if (arg_idx + 1 >= argc) {
            return 0;
        }
        value = argv[++arg_idx];
        if (strcmp(name, "--ast-mode") == 0) {
            if (strcmp(value, "simple") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_SIMPLE;
            } else if (strcmp(value, "alu") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
            } else if (strcmp(value, "mufu") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_MUFU;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--cse") == 0) {
            if (strcmp(value, "shared") == 0) {
                options->cse_mode = SECANT_BENCH_CSE_SHARED;
            } else if (strcmp(value, "distinct") == 0) {
                options->cse_mode = SECANT_BENCH_CSE_DISTINCT;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--modules") == 0) {
            if (!benchmark_parse_size(value, &options->num_modules)) return 0;
        } else if (strcmp(name, "--workers") == 0) {
            if (!benchmark_parse_size(value, &options->num_workers)) return 0;
        } else if (strcmp(name, "--streams") == 0) {
            if (!benchmark_parse_size(value, &options->num_streams)) return 0;
        } else if (strcmp(name, "--kernels") == 0) {
            if (!benchmark_parse_size(value, &options->kernels_per_module)) return 0;
        } else if (strcmp(name, "--columns") == 0) {
            if (!benchmark_parse_size(value, &options->num_input_columns)) return 0;
        } else if (strcmp(name, "--constants") == 0) {
            if (!benchmark_parse_size(value, &options->num_input_constants)) return 0;
        } else if (strcmp(name, "--settings") == 0) {
            if (!benchmark_parse_size(value, &options->num_settings)) return 0;
        } else if (strcmp(name, "--rows") == 0) {
            if (!benchmark_parse_size(value, &options->num_rows)) return 0;
        } else if (strcmp(name, "--tile-rows") == 0) {
            if (!benchmark_parse_size(value, &options->tile_rows)) return 0;
        } else if (strcmp(name, "--threads") == 0) {
            if (!benchmark_parse_size(value, &options->threads_per_block)) return 0;
        } else if (strcmp(name, "--patch-instructions-per-ast") == 0) {
            if (!benchmark_parse_size(value, &options->patch_instructions_per_ast)) return 0;
        } else if (strcmp(name, "--optimizer-iterations") == 0) {
            if (!benchmark_parse_size(value, &options->optimizer_iterations)) return 0;
        } else if (strcmp(name, "--scale") == 0) {
            if (!benchmark_parse_float(value, &options->perturbation_scale)) return 0;
        } else if (strcmp(name, "--decay") == 0) {
            if (!benchmark_parse_float(value, &options->perturbation_decay)) return 0;
        } else if (strcmp(name, "--warmups") == 0) {
            if (!benchmark_parse_size(value, &options->warmups)) return 0;
        } else if (strcmp(name, "--iterations") == 0) {
            if (!benchmark_parse_size(value, &options->iterations)) return 0;
        } else if (strcmp(name, "--seed") == 0) {
            if (!benchmark_parse_u32(value, &options->seed)) return 0;
        } else if (strcmp(name, "--target-sm") == 0) {
            if (!benchmark_parse_u32(value, &options->target_sm)) return 0;
        } else if (strcmp(name, "--device") == 0) {
            if (!benchmark_parse_int(value, &options->device_ordinal)) return 0;
        } else {
            return 0;
        }
    }
    return 1;
}

static int
benchmark_nvrtc_compile(
    const char* source,
    uint32_t target_sm,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret,
    char* error,
    size_t error_size
) {
    char architecture[64];
    const char* options[4];
    nvrtcProgram program = NULL;
    nvrtcResult result;
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;
    int architecture_size;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    architecture_size = snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%u", target_sm);
    if (architecture_size < 0 || (size_t)architecture_size >= sizeof(architecture)) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    result = nvrtcCreateProgram(&program, source, "secant_constant_optimizer_sse.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) result = nvrtcCompileProgram(program, 4, options);
    if (result == NVRTC_SUCCESS) result = nvrtcGetCUBINSize(program, &cubin_size);
    if (result == NVRTC_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin != NULL && nvrtcGetCUBIN(program, (char*)cubin) == NVRTC_SUCCESS) {
            success = 1;
        }
    }
    if (!success && program != NULL) {
        size_t log_size = 0u;

        if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1u) {
            char* log = malloc(log_size);

            if (log != NULL) {
                if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    (void)snprintf(error, error_size, "%s", log);
                }
                free(log);
            }
        }
    }
    if (program != NULL) (void)nvrtcDestroyProgram(&program);
    if (!success) {
        free(cubin);
        return 0;
    }
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

static int
benchmark_template_create(
    const SecantCubinConstantOptimizerSSERecipe* recipe,
    uint32_t target_sm,
    char** source_ret,
    size_t* source_size_ret,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret,
    char* error,
    size_t error_size
) {
    char* source = NULL;
    size_t source_size = 0u;
    size_t written_size = 0u;
    SecantResult result;

    result = secant_bench_cubin_source_generate(recipe, NULL, 0u, &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) return 0;
    source = malloc(source_size);
    if (source == NULL ||
        secant_bench_cubin_source_generate(recipe, source, source_size, &written_size) != SECANT_SUCCESS ||
        written_size != source_size ||
        !benchmark_nvrtc_compile(source, target_sm, cubin_ret, cubin_size_ret, error, error_size)) {
        free(source);
        return 0;
    }
    *source_ret = source;
    *source_size_ret = source_size;
    return 1;
}

static const char*
benchmark_ast_mode_name(SecantBenchAstMode mode) {
    if (mode == SECANT_BENCH_AST_MODE_SIMPLE) return "simple";
    if (mode == SECANT_BENCH_AST_MODE_ALU) return "alu";
    return "mufu";
}

int
main(int argc, char** argv) {
    BenchmarkOptions options;
    SecantCubinConstantOptimizerSSERecipe recipe = secant_cubin_constant_optimizer_sse_recipe_init();
    CUdevice device;
    CUcontext context = NULL;
    CUcontext previous_context = NULL;
    CUdeviceptr input_device = 0u;
    CUdeviceptr constants_device = 0u;
    CUdeviceptr target_device = 0u;
    CUdeviceptr output_device = 0u;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    void* plan_storage = NULL;
    char* source = NULL;
    unsigned char* cubin = NULL;
    SecantAstInstruction* programs = NULL;
    const SecantAstInstruction** asts = NULL;
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    float* input = NULL;
    float* constants = NULL;
    float* target = NULL;
    float* expected = NULL;
    float* actual = NULL;
    char error[BENCH_ERROR_BYTES] = {0};
    size_t num_asts;
    size_t num_inputs;
    size_t program_bytes;
    size_t pointer_bytes;
    size_t input_elements;
    size_t constant_elements;
    size_t output_elements;
    size_t plan_storage_size = 0u;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t num_routines = 0u;
    size_t verify_asts;
    size_t verify_rows;
    size_t verify_settings;
    size_t verify_output_elements;
    size_t idx;
    size_t benchmark_iteration;
    uint32_t target_sm;
    int major;
    int minor;
    int parse_result;
    int context_retained = 0;
    int success = 1;
    const char* stage = "validation";
    double template_seconds = 0.0;
    SecantRunnerStats totals = secant_runner_stats_init();

    benchmark_options_default(&options);
    parse_result = benchmark_options_parse(argc, argv, &options);
    if (parse_result < 0) {
        benchmark_usage(argv[0]);
        return 0;
    }
    num_inputs = options.num_input_columns + options.num_input_constants;
    if (parse_result == 0 || options.num_input_columns > 32u || options.num_input_constants > 32u ||
        num_inputs > SECANT_AST_MAX_INPUTS || options.num_streams > options.kernels_per_module ||
        options.threads_per_block > 1024u || options.perturbation_scale < 0.0f ||
        options.perturbation_decay <= 0.0f || options.perturbation_decay > 1.0f ||
        !benchmark_checked_mul(options.num_modules, options.kernels_per_module, &num_asts) ||
        !benchmark_checked_mul(options.num_input_columns, options.num_rows, &input_elements) ||
        !benchmark_checked_mul(num_asts, options.num_input_constants, &constant_elements) ||
        !benchmark_checked_mul(num_asts, options.num_settings, &output_elements) ||
        !secant_bench_ast_storage_sizes(
            options.num_modules, options.kernels_per_module, 1u, &program_bytes, &pointer_bytes) ||
        input_elements > SIZE_MAX / sizeof(float) || constant_elements > SIZE_MAX / sizeof(float) ||
        output_elements > SIZE_MAX / sizeof(float)) {
        benchmark_usage(argv[0]);
        return 1;
    }
    verify_asts = num_asts < 8u ? num_asts : 8u;
    verify_rows = options.num_rows < 257u ? options.num_rows : 257u;
    verify_settings = options.num_settings < 8u ? options.num_settings : 8u;
    if (!benchmark_checked_mul(verify_asts, options.num_settings, &verify_output_elements)) {
        return 1;
    }
    programs = malloc(program_bytes);
    asts = malloc(pointer_bytes);
    input = malloc(input_elements * sizeof(*input));
    constants = malloc(constant_elements * sizeof(*constants));
    target = malloc(options.num_rows * sizeof(*target));
    expected = calloc(verify_output_elements, sizeof(*expected));
    actual = malloc(verify_output_elements * sizeof(*actual));
    stage = "allocation";
    if (programs == NULL || asts == NULL || input == NULL || constants == NULL || target == NULL ||
        expected == NULL || actual == NULL) {
        (void)snprintf(error, sizeof(error), "host allocation failed");
        success = 0;
    }
    if (success) {
        stage = "AST generation";
        secant_bench_ast_fill(
            options.num_modules,
            options.kernels_per_module,
            1u,
            num_inputs,
            options.seed,
            options.ast_mode,
            options.cse_mode,
            programs,
            asts);
        success = secant_bench_ast_dynamic_constants_rewrite(
            num_asts, options.num_input_columns, options.num_input_constants, programs);
        if (!success) {
            (void)snprintf(error, sizeof(error), "AST dynamic-constant projection failed");
        }
        secant_bench_ast_get_routines(options.ast_mode, &routines, &num_routines, &routine_names);
    }
    if (success) {
        stage = "host data generation";
        for (idx = 0u; idx < input_elements; ++idx) {
            input[idx] = benchmark_value(idx / options.num_rows, idx % options.num_rows, options.seed);
        }
        for (idx = 0u; idx < constant_elements; ++idx) {
            constants[idx] = benchmark_value(idx / options.num_input_constants, idx % options.num_input_constants,
                options.seed + 17u);
        }
        for (idx = 0u; idx < options.num_rows; ++idx) {
            target[idx] = benchmark_value(0u, idx, options.seed + 31u);
        }
    }
    if (success) {
        stage = "CUDA context creation";
        if (cuInit(0u) != CUDA_SUCCESS || cuDeviceGet(&device, options.device_ordinal) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
            success = 0;
        } else {
            context_retained = 1;
            if (cuCtxGetCurrent(&previous_context) != CUDA_SUCCESS || cuCtxSetCurrent(context) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
                cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS) {
                success = 0;
            }
        }
    }
    if (success) {
        target_sm = options.target_sm != 0u ? options.target_sm : (uint32_t)(major * 10 + minor);
        recipe.num_kernels = options.kernels_per_module;
        recipe.num_input_columns = options.num_input_columns;
        recipe.num_input_constants = options.num_input_constants;
        recipe.tile_rows = options.tile_rows;
        recipe.threads_per_block = options.threads_per_block;
        recipe.patch_capacity_instructions = options.patch_instructions_per_ast;
    }
    if (success) {
        const double begin = benchmark_seconds_get();

        stage = "template compilation";
        success = benchmark_template_create(
            &recipe, target_sm, &source, &source_size, &cubin, &cubin_size, error, sizeof(error));
        template_seconds = benchmark_seconds_get() - begin;
    }
    if (success) {
        stage = "CUBIN inspection measure";
        if (secant_bench_cubin_inspect(
                &recipe, cubin, cubin_size, NULL, 0u, &plan_storage_size, &plan) != SECANT_SUCCESS ||
            plan_storage_size == 0u) {
            success = 0;
        }
    }
    if (success) {
        stage = "CUBIN inspection and runner creation";
        plan_storage = malloc(plan_storage_size);
        success = plan_storage != NULL &&
            secant_bench_cubin_inspect(
                &recipe, cubin, cubin_size, plan_storage, plan_storage_size, &plan_storage_size, &plan) == SECANT_SUCCESS &&
            secant_bench_cubin_runner_create(
                plan, cubin, cubin_size, options.num_workers, options.num_streams, &runner) == SECANT_SUCCESS;
    }
    if (success) {
        stage = "device allocation and upload";
        if (cuMemAlloc(&input_device, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&constants_device, constant_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&target_device, options.num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&output_device, output_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(input_device, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(constants_device, constants, constant_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(target_device, target, options.num_rows * sizeof(float)) != CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success) {
        SecantCpuConstantOptimizerSSERun cpu_run = secant_cpu_constant_optimizer_sse_run_init();
        SecantCubinConstantOptimizerSSERun gpu_run = secant_cubin_constant_optimizer_sse_run_init();
        SecantRunnerStats verify_stats = secant_runner_stats_init();

        stage = "CPU/GPU verification";
        cpu_run.programs.routines.items = routines;
        cpu_run.programs.routines.count = num_routines;
        cpu_run.programs.asts.items = asts;
        cpu_run.programs.asts.count = verify_asts;
        cpu_run.num_input_columns = options.num_input_columns;
        cpu_run.num_input_constants = options.num_input_constants;
        cpu_run.input.data = input;
        cpu_run.input.num_elements = input_elements;
        cpu_run.input.leading_dimension = options.num_rows;
        cpu_run.current_constants.data = constants;
        cpu_run.current_constants.num_elements = constant_elements;
        cpu_run.current_constants.leading_dimension = options.num_input_constants;
        cpu_run.target.data = target;
        cpu_run.target.num_elements = options.num_rows;
        cpu_run.target.leading_dimension = options.num_rows;
        cpu_run.num_rows = verify_rows;
        cpu_run.num_settings = verify_settings;
        cpu_run.seed = options.seed;
        cpu_run.generation = 7u;
        cpu_run.iteration = 11u;
        cpu_run.ast_index_base = 13u;
        cpu_run.perturbation_scale = options.perturbation_scale;
        cpu_run.output.data = expected;
        cpu_run.output.num_elements = verify_output_elements;
        cpu_run.output.leading_dimension = options.num_settings;

        gpu_run.programs = cpu_run.programs;
        gpu_run.input.address = (uintptr_t)input_device;
        gpu_run.input.num_elements = input_elements;
        gpu_run.input.leading_dimension = options.num_rows;
        gpu_run.current_constants.address = (uintptr_t)constants_device;
        gpu_run.current_constants.num_elements = constant_elements;
        gpu_run.current_constants.leading_dimension = options.num_input_constants;
        gpu_run.target.address = (uintptr_t)target_device;
        gpu_run.target.num_elements = options.num_rows;
        gpu_run.target.leading_dimension = options.num_rows;
        gpu_run.num_rows = verify_rows;
        gpu_run.num_settings = verify_settings;
        gpu_run.num_iterations = 1u;
        gpu_run.seed = cpu_run.seed;
        gpu_run.generation = cpu_run.generation;
        gpu_run.iteration = cpu_run.iteration;
        gpu_run.ast_index_base = cpu_run.ast_index_base;
        gpu_run.perturbation_scale = cpu_run.perturbation_scale;
        gpu_run.perturbation_decay = options.perturbation_decay;
        gpu_run.output.address = (uintptr_t)output_device;
        gpu_run.output.num_elements = output_elements;
        gpu_run.output.leading_dimension = options.num_settings;
        success = secant_cpu_run_constant_optimizer_sse(&cpu_run) == SECANT_SUCCESS &&
            secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &verify_stats) == SECANT_SUCCESS &&
            cuMemcpyDtoH(actual, output_device, verify_output_elements * sizeof(float)) == CUDA_SUCCESS;
        for (idx = 0u; success && idx < verify_asts; ++idx) {
            size_t setting;

            for (setting = 0u; setting < verify_settings; ++setting) {
                const size_t output_idx = idx * options.num_settings + setting;
                const float tolerance = (options.ast_mode == SECANT_BENCH_AST_MODE_MUFU ? 5.0e-2f : 1.0e-3f) *
                    (1.0f + fabsf(expected[output_idx]));

                if (!isfinite(expected[output_idx]) || !isfinite(actual[output_idx]) ||
                    fabsf(actual[output_idx] - expected[output_idx]) > tolerance) {
                    fprintf(stderr, "verify mismatch ast=%zu setting=%zu expected=%.9g actual=%.9g\n",
                        idx, setting, expected[output_idx], actual[output_idx]);
                    success = 0;
                    break;
                }
            }
        }
        if (success && cuMemcpyHtoD(constants_device, constants, constant_elements * sizeof(float)) != CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success) stage = "timed runner execution";
    for (benchmark_iteration = 0u;
         success && benchmark_iteration < options.warmups + options.iterations;
         ++benchmark_iteration) {
        SecantCubinConstantOptimizerSSERun run = secant_cubin_constant_optimizer_sse_run_init();
        SecantRunnerStats stats = secant_runner_stats_init();
        SecantResult result;

        run.programs.routines.items = routines;
        run.programs.routines.count = num_routines;
        run.programs.asts.items = asts;
        run.programs.asts.count = num_asts;
        run.input.address = (uintptr_t)input_device;
        run.input.num_elements = input_elements;
        run.input.leading_dimension = options.num_rows;
        run.current_constants.address = (uintptr_t)constants_device;
        run.current_constants.num_elements = constant_elements;
        run.current_constants.leading_dimension = options.num_input_constants;
        run.target.address = (uintptr_t)target_device;
        run.target.num_elements = options.num_rows;
        run.target.leading_dimension = options.num_rows;
        run.num_rows = options.num_rows;
        run.num_settings = options.num_settings;
        run.num_iterations = options.optimizer_iterations;
        run.seed = options.seed;
        run.generation = 17u + benchmark_iteration;
        run.iteration = 0u;
        run.ast_index_base = 0u;
        run.perturbation_scale = options.perturbation_scale;
        run.perturbation_decay = options.perturbation_decay;
        run.output.address = (uintptr_t)output_device;
        run.output.num_elements = output_elements;
        run.output.leading_dimension = options.num_settings;
        result = secant_cubin_runner_run_constant_optimizer_sse(runner, &run, &stats);
        if (result != SECANT_SUCCESS) {
            (void)snprintf(error, sizeof(error), "%s", secant_result_to_string(result));
            success = 0;
        } else if (benchmark_iteration >= options.warmups) {
            totals.compile_window_seconds += stats.compile_window_seconds;
            totals.compile_critical_seconds += stats.compile_critical_seconds;
            totals.compile_work_seconds += stats.compile_work_seconds;
            totals.module_load_seconds += stats.module_load_seconds;
            totals.completion_wait_seconds += stats.completion_wait_seconds;
            totals.module_unload_seconds += stats.module_unload_seconds;
            totals.runtime_seconds += stats.runtime_seconds;
            totals.total_seconds += stats.total_seconds;
        }
    }
    if (success &&
        (totals.compile_window_seconds <= 0.0 || totals.runtime_seconds <= 0.0 || totals.total_seconds <= 0.0)) {
        (void)snprintf(error, sizeof(error), "non-positive measured duration");
        success = 0;
    }
    if (success) {
        const double ast_compiles = (double)options.iterations * (double)num_asts;
        const double row_evals = ast_compiles * (double)options.num_settings * (double)options.num_rows *
            (double)options.optimizer_iterations;

        printf(
            "backend=cubin shape=constant_optimizer_sse topology=one_ast_per_kernel ast_mode=%s cse=%s "
            "modules=%zu workers=%zu streams=%zu kernels_per_module=%zu asts_per_kernel=1 asts=%zu "
            "columns=%zu constants=%zu settings=%zu rows=%zu tile_rows=%zu threads=%zu "
            "optimizer_iterations=%zu benchmark_iterations=%zu template_seconds=%.6f cubin_bytes=%zu "
            "compile_seconds=%.6f compile_asts_per_second=%.3f module_load_seconds=%.6f "
            "runtime_seconds=%.6f pipeline_seconds=%.6f row_evals=%.0f "
            "runtime_row_evals_per_second=%.3e pipeline_row_evals_per_second=%.3e verify=pass\n",
            benchmark_ast_mode_name(options.ast_mode),
            options.cse_mode == SECANT_BENCH_CSE_SHARED ? "shared" : "distinct",
            options.num_modules,
            options.num_workers,
            options.num_streams,
            options.kernels_per_module,
            num_asts,
            options.num_input_columns,
            options.num_input_constants,
            options.num_settings,
            options.num_rows,
            options.tile_rows,
            options.threads_per_block,
            options.optimizer_iterations,
            options.iterations,
            template_seconds,
            cubin_size,
            totals.compile_window_seconds,
            ast_compiles / totals.compile_window_seconds,
            totals.module_load_seconds,
            totals.runtime_seconds,
            totals.total_seconds,
            row_evals,
            row_evals / totals.runtime_seconds,
            row_evals / totals.total_seconds);
    } else {
        fprintf(stderr, "constant_optimizer_sse benchmark failed during %s: %s\n",
            stage, error[0] != '\0' ? error : "unknown error");
    }

    if (runner != NULL) (void)secant_cubin_runner_destroy(runner);
    if (output_device != 0u) (void)cuMemFree(output_device);
    if (target_device != 0u) (void)cuMemFree(target_device);
    if (constants_device != 0u) (void)cuMemFree(constants_device);
    if (input_device != 0u) (void)cuMemFree(input_device);
    free(actual);
    free(expected);
    free(target);
    free(constants);
    free(input);
    free(asts);
    free(programs);
    free(plan_storage);
    free(cubin);
    free(source);
    if (context_retained) {
        (void)cuCtxSetCurrent(previous_context);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return success ? 0 : 1;
}
