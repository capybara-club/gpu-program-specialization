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
#include "secant.h"
#include "backends/cuda/secant_cuda_runner.h"
#include "backends/ptx/secant_ptx.h"
#include "backends/ptx/secant_ptx_runner.h"

#include <cuda.h>
#include <nvrtc.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_PROGRAM_STRIDE 512u
#define BENCH_ERROR_BYTES 16384u
#define BENCH_COMPILE_SCRATCH_SIZE (64u * 1024u * 1024u)

typedef enum BenchmarkBackend {
    BENCHMARK_BACKEND_CUBIN = 0,
    BENCHMARK_BACKEND_CUDA = 1,
    BENCHMARK_BACKEND_PTX = 2
} BenchmarkBackend;

typedef struct BenchmarkOptions {
    BenchmarkBackend backend;
    SecantBenchAstMode ast_mode;
    size_t num_modules;
    size_t num_workers;
    size_t num_streams;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_input_columns;
    size_t num_input_constants;
    size_t num_settings;
    size_t num_rows;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_instructions_per_ast;
    size_t warmups;
    size_t iterations;
    size_t optimizer_iterations;
    SecantConstantOptimizerUpdateMode update_mode;
    size_t num_elites;
    float initial_scale;
    float momentum;
    float scale_learning_rate;
    float scale_failure_decay;
    float minimum_scale;
    float maximum_scale;
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
    options->backend = BENCHMARK_BACKEND_CUBIN;
    options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
    options->num_modules = 4u;
    options->num_workers = 24u;
    options->num_streams = 8u;
    options->num_kernels = 64u;
    options->asts_per_kernel = 32u;
    options->num_input_columns = 4u;
    options->num_input_constants = 4u;
    options->num_settings = 8192u;
    options->num_rows = 10000u;
    options->tile_rows = 128u;
    options->threads_per_block = 128u;
    options->patch_instructions_per_ast = 64u;
    options->warmups = 1u;
    options->iterations = 3u;
    options->optimizer_iterations = 4u;
    options->update_mode = SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER;
    options->num_elites = 1u;
    options->initial_scale = 1.0f;
    options->momentum = 0.0f;
    options->scale_learning_rate = 0.25f;
    options->scale_failure_decay = 0.5f;
    options->minimum_scale = 1.0e-6f;
    options->maximum_scale = 1.0e6f;
    options->seed = 1u;
    options->target_sm = 0u;
    options->device_ordinal = 0;
}

static void
benchmark_usage(const char* program) {
    fprintf(stderr,
        "usage: %s [--backend cubin|cuda|ptx] [--ast-mode simple|alu|mufu] "
        "[--modules N] [--workers N] [--streams N] "
        "[--kernels N] [--asts-per-kernel N] [--columns N] [--constants N] [--settings N] [--rows N] "
        "[--tile-rows N] [--threads N] [--patch-instructions-per-ast N] [--scale F] [--momentum F] "
        "[--update winner|elite] [--elites N] [--scale-learning-rate F] [--failure-decay F] "
        "[--minimum-scale F] [--maximum-scale F] "
        "[--warmups N] [--iterations N] [--optimizer-iterations N] [--seed N] [--target-sm NN] [--device N]\n",
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
    if (errno != 0 || text == end || *end != '\0' || !isfinite(value) || value < 0.0f) {
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
        if (strcmp(name, "--backend") == 0) {
            if (strcmp(value, "cubin") == 0) options->backend = BENCHMARK_BACKEND_CUBIN;
            else if (strcmp(value, "cuda") == 0) options->backend = BENCHMARK_BACKEND_CUDA;
            else if (strcmp(value, "ptx") == 0) options->backend = BENCHMARK_BACKEND_PTX;
            else return 0;
        } else if (strcmp(name, "--ast-mode") == 0) {
            if (strcmp(value, "simple") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_SIMPLE;
            } else if (strcmp(value, "alu") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
            } else if (strcmp(value, "mufu") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_MUFU;
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
            if (!benchmark_parse_size(value, &options->num_kernels)) return 0;
        } else if (strcmp(name, "--asts-per-kernel") == 0) {
            if (!benchmark_parse_size(value, &options->asts_per_kernel)) return 0;
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
        } else if (strcmp(name, "--scale") == 0) {
            if (!benchmark_parse_float(value, &options->initial_scale)) return 0;
        } else if (strcmp(name, "--update") == 0) {
            if (strcmp(value, "winner") == 0) {
                options->update_mode = SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER;
            } else if (strcmp(value, "elite") == 0) {
                options->update_mode = SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--elites") == 0) {
            if (!benchmark_parse_size(value, &options->num_elites)) return 0;
        } else if (strcmp(name, "--momentum") == 0) {
            if (!benchmark_parse_float(value, &options->momentum)) return 0;
        } else if (strcmp(name, "--scale-learning-rate") == 0) {
            if (!benchmark_parse_float(value, &options->scale_learning_rate)) return 0;
        } else if (strcmp(name, "--failure-decay") == 0) {
            if (!benchmark_parse_float(value, &options->scale_failure_decay)) return 0;
        } else if (strcmp(name, "--minimum-scale") == 0) {
            if (!benchmark_parse_float(value, &options->minimum_scale)) return 0;
        } else if (strcmp(name, "--maximum-scale") == 0) {
            if (!benchmark_parse_float(value, &options->maximum_scale)) return 0;
        } else if (strcmp(name, "--warmups") == 0) {
            if (!benchmark_parse_size(value, &options->warmups)) return 0;
        } else if (strcmp(name, "--iterations") == 0) {
            if (!benchmark_parse_size(value, &options->iterations)) return 0;
        } else if (strcmp(name, "--optimizer-iterations") == 0) {
            if (!benchmark_parse_size(value, &options->optimizer_iterations)) return 0;
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
benchmark_instruction_write(
    SecantAstInstruction* program,
    size_t* offset,
    SecantAstInstructionType type,
    uint32_t payload
) {
    const size_t size = type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32
        ? 5u
        : (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 ||
           type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 ? 2u : 1u);

    if (*offset > BENCH_PROGRAM_STRIDE - size) {
        return 0;
    }
    program[(*offset)++] = (uint8_t)type;
    if (size == 2u) {
        program[(*offset)++] = (uint8_t)payload;
    } else if (size == 5u) {
        program[(*offset)++] = (uint8_t)(payload >> 0u);
        program[(*offset)++] = (uint8_t)(payload >> 8u);
        program[(*offset)++] = (uint8_t)(payload >> 16u);
        program[(*offset)++] = (uint8_t)(payload >> 24u);
    }
    return 1;
}

static int
benchmark_ast_fill(
    const BenchmarkOptions* options,
    size_t num_asts,
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts,
    float* current_constants
) {
    size_t ast_idx;

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        SecantAstInstruction* program = programs + ast_idx * BENCH_PROGRAM_STRIDE;
        size_t offset = 0u;
        size_t constant_idx;

        memset(program, 0, BENCH_PROGRAM_STRIDE);
        for (constant_idx = 0u; constant_idx < options->num_input_constants; ++constant_idx) {
            const uint32_t hash = secant_bench_ast_hash32(
                options->seed ^ (uint32_t)ast_idx * UINT32_C(0x9e3779b9) ^
                (uint32_t)constant_idx * UINT32_C(0x85ebca6b));
            const float center = ((float)(hash & UINT32_C(0xffff)) / 65535.0f) * 4.0f - 2.0f;
            SecantAstInstructionType unary = SECANT_AST_INSTRUCTION_TYPE_NONE;

            current_constants[ast_idx * options->num_input_constants + constant_idx] = center;

            if (options->ast_mode == SECANT_BENCH_AST_MODE_ALU) {
                unary = (hash & 1u) != 0u
                    ? SECANT_AST_INSTRUCTION_TYPE_ABS_F32
                    : SECANT_AST_INSTRUCTION_TYPE_NEG_F32;
            } else if (options->ast_mode == SECANT_BENCH_AST_MODE_MUFU) {
                unary = (hash & 1u) != 0u
                    ? SECANT_AST_INSTRUCTION_TYPE_SIN_F32
                    : SECANT_AST_INSTRUCTION_TYPE_COS_F32;
            }
            if (!benchmark_instruction_write(
                    program, &offset, SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
                    (uint32_t)((ast_idx + constant_idx) % options->num_input_columns)) ||
                !benchmark_instruction_write(
                    program, &offset, SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
                    (uint32_t)constant_idx) ||
                !benchmark_instruction_write(program, &offset, SECANT_AST_INSTRUCTION_TYPE_MUL_F32, 0u) ||
                (unary != SECANT_AST_INSTRUCTION_TYPE_NONE &&
                 !benchmark_instruction_write(program, &offset, unary, 0u)) ||
                (constant_idx != 0u &&
                 !benchmark_instruction_write(program, &offset, SECANT_AST_INSTRUCTION_TYPE_ADD_F32, 0u))) {
                return 0;
            }
        }
        if (!benchmark_instruction_write(program, &offset, SECANT_AST_INSTRUCTION_TYPE_RETURN_F32, 0u)) {
            return 0;
        }
        asts[ast_idx] = program;
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
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%u", target_sm) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    result = nvrtcCreateProgram(&program, source, "secant_packed_constant_optimizer_sse.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) result = nvrtcCompileProgram(program, 4, options);
    if (result == NVRTC_SUCCESS) result = nvrtcGetCUBINSize(program, &cubin_size);
    if (result == NVRTC_SUCCESS && cubin_size != 0u) {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin != NULL && nvrtcGetCUBIN(program, (char*)cubin) == NVRTC_SUCCESS) {
            success = 1;
        }
    }
    if (!success && program != NULL) {
        size_t log_size = 0u;

        if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1u) {
            char* log = (char*)malloc(log_size);

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

static const char*
benchmark_ast_mode_name(SecantBenchAstMode mode) {
    if (mode == SECANT_BENCH_AST_MODE_SIMPLE) return "simple";
    if (mode == SECANT_BENCH_AST_MODE_ALU) return "alu";
    return "mufu";
}

int
main(int argc, char** argv) {
    BenchmarkOptions options;
    SecantCubinPackedConstantOptimizerSSERecipe recipe =
        secant_cubin_packed_constant_optimizer_sse_recipe_init();
    SecantCubinRunnerOptions runner_options = secant_cubin_runner_options_init();
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    SecantCUDARunner cuda_runner = NULL;
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXRunner ptx_runner = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUcontext previous_context = NULL;
    CUdeviceptr input_device = 0u;
    CUdeviceptr target_device = 0u;
    CUdeviceptr sse_device = 0u;
    CUdeviceptr best_device = 0u;
    SecantAstInstruction* programs = NULL;
    const SecantAstInstruction** asts = NULL;
    float* input = NULL;
    float* target = NULL;
    float* initial_constants = NULL;
    float* current_constants = NULL;
    float* current_scales = NULL;
    float* current_velocities = NULL;
    float* current_sse = NULL;
    char* source = NULL;
    unsigned char* cubin = NULL;
    void* plan_storage = NULL;
    char error[BENCH_ERROR_BYTES] = {0};
    size_t asts_per_module;
    size_t num_asts;
    size_t program_bytes;
    size_t pointer_bytes;
    size_t input_elements;
    size_t constant_elements;
    size_t sse_elements;
    size_t best_elements;
    size_t patch_capacity;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    size_t idx;
    size_t benchmark_iteration;
    uint32_t target_sm;
    int major = 0;
    int minor = 0;
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
    if (parse_result == 0 || options.num_input_columns > 32u || options.num_input_constants > 32u ||
        options.num_streams > options.num_kernels || options.threads_per_block > 1024u ||
        !benchmark_checked_mul(options.num_kernels, options.asts_per_kernel, &asts_per_module) ||
        !benchmark_checked_mul(options.num_modules, asts_per_module, &num_asts) ||
        !benchmark_checked_mul(num_asts, BENCH_PROGRAM_STRIDE, &program_bytes) ||
        !benchmark_checked_mul(num_asts, sizeof(*asts), &pointer_bytes) ||
        !benchmark_checked_mul(options.num_input_columns, options.num_rows, &input_elements) ||
        !benchmark_checked_mul(num_asts, options.num_input_constants, &constant_elements) ||
        !benchmark_checked_mul(num_asts, options.num_settings, &sse_elements) ||
        !benchmark_checked_mul(num_asts, 1u + 4u * options.num_input_constants, &best_elements) ||
        !benchmark_checked_mul(options.asts_per_kernel, options.patch_instructions_per_ast, &patch_capacity) ||
        input_elements > SIZE_MAX / sizeof(float) || sse_elements > SIZE_MAX / sizeof(float) ||
        best_elements > SIZE_MAX / sizeof(float) || constant_elements > SIZE_MAX / sizeof(float) ||
        options.initial_scale < options.minimum_scale || options.initial_scale > options.maximum_scale ||
        options.momentum > 1.0f || options.scale_learning_rate < 0.0f ||
        options.scale_learning_rate > 1.0f || options.scale_failure_decay <= 0.0f ||
        options.scale_failure_decay > 1.0f || options.minimum_scale <= 0.0f ||
        options.maximum_scale < options.minimum_scale) {
        benchmark_usage(argv[0]);
        return 1;
    }
    if ((options.update_mode == SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER && options.num_elites != 1u) ||
        (options.update_mode == SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION &&
         (options.num_elites < 2u || options.num_elites > 16u || options.num_elites > options.num_settings ||
          options.momentum != 0.0f))) {
        benchmark_usage(argv[0]);
        return 1;
    }
    programs = (SecantAstInstruction*)malloc(program_bytes);
    asts = (const SecantAstInstruction**)malloc(pointer_bytes);
    input = (float*)malloc(input_elements * sizeof(*input));
    target = (float*)malloc(options.num_rows * sizeof(*target));
    initial_constants = (float*)malloc(constant_elements * sizeof(*initial_constants));
    current_constants = (float*)malloc(constant_elements * sizeof(*current_constants));
    current_scales = (float*)malloc(constant_elements * sizeof(*current_scales));
    current_velocities = (float*)malloc(constant_elements * sizeof(*current_velocities));
    current_sse = (float*)malloc(num_asts * sizeof(*current_sse));
    if (programs == NULL || asts == NULL || input == NULL || target == NULL ||
        initial_constants == NULL || current_constants == NULL || current_scales == NULL ||
        current_velocities == NULL || current_sse == NULL ||
        !benchmark_ast_fill(&options, num_asts, programs, asts, initial_constants)) {
        (void)snprintf(error, sizeof(error), "host allocation or AST generation failed");
        success = 0;
    }
    if (success) {
        for (idx = 0u; idx < input_elements; ++idx) {
            input[idx] = benchmark_value(idx / options.num_rows, idx % options.num_rows, options.seed);
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
        recipe.num_kernels = options.num_kernels;
        recipe.asts_per_kernel = options.asts_per_kernel;
        recipe.num_input_columns = options.num_input_columns;
        recipe.num_input_constants = options.num_input_constants;
        recipe.tile_rows = options.tile_rows;
        recipe.threads_per_block = options.threads_per_block;
        recipe.patch_capacity_instructions = patch_capacity;
        runner_options.num_workers = options.num_workers;
        runner_options.num_streams = options.num_streams;
    }
    if (success) {
        const double begin = benchmark_seconds_get();

        stage = "template compilation";
        if (options.backend == BENCHMARK_BACKEND_CUBIN) {
            if (secant_cubin_source_size(&recipe.header, &source_size) != SECANT_SUCCESS ||
                (source = (char*)malloc(source_size)) == NULL ||
                secant_cubin_source_write(&recipe.header, source, source_size) != SECANT_SUCCESS ||
                !benchmark_nvrtc_compile(source, target_sm, &cubin, &cubin_size, error, sizeof(error))) {
                success = 0;
            }
        } else if (options.backend == BENCHMARK_BACKEND_PTX) {
            static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
            const uint32_t source_major = major >= 8 ? 8u : (uint32_t)major;
            const uint32_t source_minor = major >= 8 ? 0u : (uint32_t)minor;
            size_t log_size = 0u;
            SecantPTXResult ptx_result = secant_ptx_packed_constant_optimizer_sse_create(
                options.num_kernels,
                options.asts_per_kernel,
                options.num_input_columns,
                options.num_input_constants,
                options.tile_rows,
                options.threads_per_block,
                source_major,
                source_minor,
                nvrtc_options,
                2u,
                true,
                error,
                sizeof(error),
                &log_size,
                &ptx_handle);

            if (ptx_result != SECANT_PTX_SUCCESS) success = 0;
        }
        template_seconds = benchmark_seconds_get() - begin;
    }
    if (success) {
        static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
        static const char* const nvptx_options[] = { "--opt-level=1" };
        SecantResult create_result = SECANT_SUCCESS;

        stage = "runner creation";
        if (options.backend == BENCHMARK_BACKEND_CUBIN) {
            if (secant_cubin_plan_storage_size(&recipe.header, cubin, cubin_size, &plan_storage_size) != SECANT_SUCCESS ||
                (plan_storage = malloc(plan_storage_size)) == NULL ||
                secant_cubin_plan_init(
                    &recipe.header, cubin, cubin_size, plan_storage, plan_storage_size, &plan) != SECANT_SUCCESS) {
                create_result = SECANT_ERROR_BAD_BINARY;
            } else {
                create_result = secant_cubin_runner_create(plan, cubin, cubin_size, &runner_options, &runner);
            }
        } else if (options.backend == BENCHMARK_BACKEND_CUDA) {
            create_result = secant_cuda_packed_constant_optimizer_sse_runner_create(
                options.num_kernels,
                options.asts_per_kernel,
                options.num_input_columns,
                options.num_input_constants,
                options.tile_rows,
                options.threads_per_block,
                (uint32_t)major,
                (uint32_t)minor,
                nvrtc_options,
                2u,
                BENCH_COMPILE_SCRATCH_SIZE,
                options.num_workers,
                options.num_streams,
                &cuda_runner);
        } else {
            create_result = secant_ptx_packed_constant_optimizer_sse_runner_create(
                ptx_handle,
                (uint32_t)major,
                (uint32_t)minor,
                nvptx_options,
                1u,
                BENCH_COMPILE_SCRATCH_SIZE,
                options.num_workers,
                options.num_streams,
                &ptx_runner);
        }
        if (create_result != SECANT_SUCCESS) {
            (void)snprintf(error, sizeof(error), "%s", secant_result_to_string(create_result));
            success = 0;
        }
    }
    if (success) {
        stage = "device setup";
        if (cuMemAlloc(&input_device, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&target_device, options.num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&sse_device, sse_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&best_device, best_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(input_device, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(target_device, target, options.num_rows * sizeof(float)) != CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success) stage = "timed runner execution";
    for (benchmark_iteration = 0u;
         success && benchmark_iteration < options.warmups + options.iterations;
         ++benchmark_iteration) {
        SecantCubinPackedConstantOptimizerSSERun run = secant_cubin_packed_constant_optimizer_sse_run_init();
        SecantRunnerStats stats = secant_runner_stats_init();
        SecantResult result;

        memcpy(current_constants, initial_constants, constant_elements * sizeof(*current_constants));
        for (idx = 0u; idx < constant_elements; ++idx) {
            current_scales[idx] = options.initial_scale;
            current_velocities[idx] = 0.0f;
        }
        for (idx = 0u; idx < num_asts; ++idx) {
            current_sse[idx] = INFINITY;
        }
        run.programs.asts.items = asts;
        run.programs.asts.count = num_asts;
        run.programs.current_constants.data = current_constants;
        run.programs.current_constants.num_elements = constant_elements;
        run.programs.current_constants.leading_dimension = options.num_input_constants;
        run.programs.current_constant_scales.data = current_scales;
        run.programs.current_constant_scales.num_elements = constant_elements;
        run.programs.current_constant_scales.leading_dimension = options.num_input_constants;
        run.input.address = (uintptr_t)input_device;
        run.input.num_elements = input_elements;
        run.input.leading_dimension = options.num_rows;
        run.target.address = (uintptr_t)target_device;
        run.target.num_elements = options.num_rows;
        run.target.leading_dimension = options.num_rows;
        run.num_rows = options.num_rows;
        run.num_settings = options.num_settings;
        run.num_iterations = options.optimizer_iterations;
        run.seed = options.seed;
        run.generation = 17u;
        run.iteration = benchmark_iteration;
        run.update_mode = options.update_mode;
        run.num_elites = options.num_elites;
        run.current_constant_velocities.data = current_velocities;
        run.current_constant_velocities.num_elements = constant_elements;
        run.current_constant_velocities.leading_dimension = options.num_input_constants;
        run.current_sse.data = current_sse;
        run.current_sse.num_elements = num_asts;
        run.momentum = options.momentum;
        run.scale_learning_rate = options.scale_learning_rate;
        run.scale_failure_decay = options.scale_failure_decay;
        run.minimum_scale = options.minimum_scale;
        run.maximum_scale = options.maximum_scale;
        run.sse.address = (uintptr_t)sse_device;
        run.sse.num_elements = sse_elements;
        run.sse.leading_dimension = options.num_settings;
        run.best.address = (uintptr_t)best_device;
        run.best.num_elements = best_elements;
        run.best.leading_dimension = 1u + 4u * options.num_input_constants;
        if (options.backend == BENCHMARK_BACKEND_CUBIN) {
            result = secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &run, &stats);
        } else if (options.backend == BENCHMARK_BACKEND_CUDA) {
            result = secant_cuda_packed_constant_optimizer_sse_runner_run(cuda_runner, &run, NULL, &stats);
        } else {
            result = secant_ptx_packed_constant_optimizer_sse_runner_run(ptx_runner, &run, &stats);
        }
        if (result != SECANT_SUCCESS) {
            (void)snprintf(error, sizeof(error), "%s", secant_result_to_string(result));
            success = 0;
        }
        for (idx = 0u; success && idx < constant_elements; ++idx) {
            if (!isfinite(current_constants[idx]) || !isfinite(current_scales[idx]) ||
                current_scales[idx] < options.minimum_scale || current_scales[idx] > options.maximum_scale ||
                !isfinite(current_velocities[idx])) {
                (void)snprintf(error, sizeof(error), "invalid adaptive state");
                success = 0;
            }
        }
        for (idx = 0u; success && idx < num_asts; ++idx) {
            if (!(current_sse[idx] >= 0.0f)) {
                (void)snprintf(error, sizeof(error), "invalid incumbent SSE");
                success = 0;
            }
        }
        if (success && benchmark_iteration >= options.warmups) {
            totals.num_modules += stats.num_modules;
            totals.modules_loaded += stats.modules_loaded;
            totals.compile_window_seconds += stats.compile_window_seconds;
            totals.compile_critical_seconds += stats.compile_critical_seconds;
            totals.compile_work_seconds += stats.compile_work_seconds;
            totals.module_load_seconds += stats.module_load_seconds;
            totals.completion_wait_seconds += stats.completion_wait_seconds;
            totals.module_unload_seconds += stats.module_unload_seconds;
            totals.runtime_seconds += stats.runtime_seconds;
            totals.total_seconds += stats.total_seconds;
        } else if (success) {
            float best_sse;

            if (cuMemcpyDtoH(&best_sse, best_device, sizeof(best_sse)) != CUDA_SUCCESS || !isfinite(best_sse)) {
                (void)snprintf(error, sizeof(error), "non-finite reducer output");
                success = 0;
            }
        }
    }
    if (success &&
        (totals.compile_window_seconds <= 0.0 || totals.runtime_seconds <= 0.0 || totals.total_seconds <= 0.0)) {
        (void)snprintf(error, sizeof(error), "non-positive measured duration");
        success = 0;
    }
    if (success) {
        const double ast_compiles =
            (double)options.iterations * (double)options.optimizer_iterations * (double)num_asts;
        const double row_evals = ast_compiles * (double)options.num_settings * (double)options.num_rows;

        printf(
            "backend=%s shape=packed_constant_optimizer_sse topology=packed_shared_jitter ast_mode=%s "
            "modules=%zu workers=%zu streams=%zu kernels_per_module=%zu asts_per_kernel=%zu asts=%zu "
            "columns=%zu constants=%zu settings=%zu rows=%zu tile_rows=%zu threads=%zu "
            "benchmark_iterations=%zu optimizer_iterations=%zu update=%s elites=%zu initial_scale=%g momentum=%g "
            "scale_learning_rate=%g scale_failure_decay=%g minimum_scale=%g maximum_scale=%g "
            "template_seconds=%.6f cubin_bytes=%zu compile_window_seconds=%.6f "
            "compile_critical_seconds=%.6f compile_work_seconds=%.6f compile_window_asts_per_second=%.3f "
            "compile_parallel_asts_per_second=%.3f compile_core_asts_per_second=%.3f "
            "module_loads=%zu module_load_seconds=%.6f runtime_seconds=%.6f "
            "pipeline_seconds=%.6f row_evals=%.0f runtime_row_evals_per_second=%.3e "
            "pipeline_row_evals_per_second=%.3e reducer=adaptive_center_scale_velocity "
            "feedback=pinned_async_dtoh_respecialize verify=finite_state_bounds\n",
            options.backend == BENCHMARK_BACKEND_CUDA ? "cuda" :
                options.backend == BENCHMARK_BACKEND_PTX ? "ptx" : "cubin",
            benchmark_ast_mode_name(options.ast_mode),
            options.num_modules,
            options.num_workers,
            options.num_streams,
            options.num_kernels,
            options.asts_per_kernel,
            num_asts,
            options.num_input_columns,
            options.num_input_constants,
            options.num_settings,
            options.num_rows,
            options.tile_rows,
            options.threads_per_block,
            options.iterations,
            options.optimizer_iterations,
            options.update_mode == SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION ? "elite" : "winner",
            options.num_elites,
            options.initial_scale,
            options.momentum,
            options.scale_learning_rate,
            options.scale_failure_decay,
            options.minimum_scale,
            options.maximum_scale,
            template_seconds,
            cubin_size,
            totals.compile_window_seconds,
            totals.compile_critical_seconds,
            totals.compile_work_seconds,
            ast_compiles / totals.compile_window_seconds,
            ast_compiles / totals.compile_critical_seconds,
            ast_compiles / totals.compile_work_seconds,
            totals.modules_loaded,
            totals.module_load_seconds,
            totals.runtime_seconds,
            totals.total_seconds,
            row_evals,
            row_evals / totals.runtime_seconds,
            row_evals / totals.total_seconds);
    } else {
        fprintf(stderr, "packed constant optimizer benchmark failed at %s: %s\n", stage, error);
    }

    if (ptx_runner != NULL) (void)secant_ptx_runner_destroy(ptx_runner);
    if (ptx_handle != NULL) (void)secant_ptx_handle_destroy(ptx_handle);
    if (cuda_runner != NULL) (void)secant_cuda_runner_destroy(cuda_runner);
    if (runner != NULL) (void)secant_cubin_runner_destroy(runner);
    if (best_device != 0u) (void)cuMemFree(best_device);
    if (sse_device != 0u) (void)cuMemFree(sse_device);
    if (target_device != 0u) (void)cuMemFree(target_device);
    if (input_device != 0u) (void)cuMemFree(input_device);
    free(plan_storage);
    free(cubin);
    free(source);
    free(current_sse);
    free(current_velocities);
    free(current_scales);
    free(current_constants);
    free(initial_constants);
    free(target);
    free(input);
    free(asts);
    free(programs);
    if (context_retained) {
        (void)cuCtxSetCurrent(previous_context);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return success ? 0 : 1;
}
