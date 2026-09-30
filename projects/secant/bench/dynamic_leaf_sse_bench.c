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

#define BENCH_ERROR_BYTES 16384u
#define BENCH_COMPILE_SCRATCH_SIZE (64u * 1024u * 1024u)

typedef enum BenchmarkBackend {
    BENCHMARK_BACKEND_CUBIN = 0,
    BENCHMARK_BACKEND_CUDA = 1,
    BENCHMARK_BACKEND_PTX = 2
} BenchmarkBackend;

typedef enum BenchmarkDynamicValues {
    BENCHMARK_DYNAMIC_VALUES_MIXED = 0,
    BENCHMARK_DYNAMIC_VALUES_COLUMNS = 1
} BenchmarkDynamicValues;

typedef enum BenchmarkSettingOwner {
    BENCHMARK_SETTING_OWNER_THREAD = 0,
    BENCHMARK_SETTING_OWNER_WARP = 1
} BenchmarkSettingOwner;

typedef struct BenchmarkOptions {
    BenchmarkBackend backend;
    SecantBenchAstMode ast_mode;
    SecantBenchCSEMode cse_mode;
    BenchmarkDynamicValues dynamic_values;
    BenchmarkSettingOwner setting_owner;
    size_t num_modules;
    size_t num_workers;
    size_t num_streams;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_dynamic_leaves;
    size_t num_dynamic_sites;
    size_t ast_nodes;
    size_t num_input_columns;
    size_t num_input_column_capacity;
    size_t num_static_input_column_capacity;
    size_t num_targets;
    size_t num_settings;
    size_t settings_per_cta;
    size_t num_rows;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_instructions_per_ast;
    size_t warmups;
    size_t iterations;
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

static uint32_t
benchmark_f32_bits(float value) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void
benchmark_options_default(BenchmarkOptions* options) {
    memset(options, 0, sizeof(*options));
    options->backend = BENCHMARK_BACKEND_CUBIN;
    options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
    options->cse_mode = SECANT_BENCH_CSE_DISTINCT;
    options->setting_owner = BENCHMARK_SETTING_OWNER_THREAD;
    options->num_modules = 8u;
    options->num_workers = 24u;
    options->num_streams = 8u;
    options->num_kernels = 16u;
    options->asts_per_kernel = 32u;
    options->num_dynamic_leaves = 16u;
    options->num_input_columns = 8u;
    options->num_targets = 1u;
    options->num_settings = 4096u;
    options->num_rows = 1048576u;
    options->tile_rows = 64u;
    options->threads_per_block = 128u;
    options->patch_instructions_per_ast = 64u;
    options->warmups = 1u;
    options->iterations = 3u;
    options->seed = 1u;
    options->target_sm = 0u;
    options->device_ordinal = 0;
}

static void
benchmark_usage(const char* program) {
    fprintf(stderr,
        "usage: %s [--backend cubin|cuda|ptx] [--ast-mode simple|alu|mufu] "
        "[--cse shared|distinct] [--setting-owner thread|warp] [--modules N] [--workers N] "
        "[--streams N] [--kernels N] [--asts-per-kernel N] [--ast-nodes N] [--leaves N] [--columns N] "
        "[--dynamic-sites N] [--dynamic-values mixed|columns] [--column-capacity N] "
        "[--static-columns 0|column-capacity] [--targets N] "
        "[--settings N] [--settings-per-cta N] [--rows N] [--tile-rows N] [--threads N] "
        "[--patch-instructions-per-ast N] "
        "[--warmups N] [--iterations N] [--seed N] [--target-sm NN] [--device N]\n",
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
        } else if (strcmp(name, "--cse") == 0) {
            if (strcmp(value, "shared") == 0) {
                options->cse_mode = SECANT_BENCH_CSE_SHARED;
            } else if (strcmp(value, "distinct") == 0) {
                options->cse_mode = SECANT_BENCH_CSE_DISTINCT;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--setting-owner") == 0) {
            if (strcmp(value, "thread") == 0) {
                options->setting_owner = BENCHMARK_SETTING_OWNER_THREAD;
            } else if (strcmp(value, "warp") == 0) {
                options->setting_owner = BENCHMARK_SETTING_OWNER_WARP;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--modules") == 0) {
            if (!benchmark_parse_size(value, &options->num_modules)) {
                return 0;
            }
        } else if (strcmp(name, "--workers") == 0) {
            if (!benchmark_parse_size(value, &options->num_workers)) {
                return 0;
            }
        } else if (strcmp(name, "--streams") == 0) {
            if (!benchmark_parse_size(value, &options->num_streams)) {
                return 0;
            }
        } else if (strcmp(name, "--kernels") == 0) {
            if (!benchmark_parse_size(value, &options->num_kernels)) {
                return 0;
            }
        } else if (strcmp(name, "--asts-per-kernel") == 0) {
            if (!benchmark_parse_size(value, &options->asts_per_kernel)) {
                return 0;
            }
        } else if (strcmp(name, "--ast-nodes") == 0) {
            if (!benchmark_parse_size(value, &options->ast_nodes)) {
                return 0;
            }
        } else if (strcmp(name, "--leaves") == 0) {
            if (!benchmark_parse_size(value, &options->num_dynamic_leaves)) {
                return 0;
            }
        } else if (strcmp(name, "--dynamic-sites") == 0) {
            if (!benchmark_parse_size(value, &options->num_dynamic_sites)) {
                return 0;
            }
        } else if (strcmp(name, "--dynamic-values") == 0) {
            if (strcmp(value, "mixed") == 0) {
                options->dynamic_values = BENCHMARK_DYNAMIC_VALUES_MIXED;
            } else if (strcmp(value, "columns") == 0) {
                options->dynamic_values = BENCHMARK_DYNAMIC_VALUES_COLUMNS;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--columns") == 0) {
            if (!benchmark_parse_size(value, &options->num_input_columns)) {
                return 0;
            }
        } else if (strcmp(name, "--column-capacity") == 0) {
            if (!benchmark_parse_size(value, &options->num_input_column_capacity)) {
                return 0;
            }
        } else if (strcmp(name, "--static-columns") == 0) {
            uint32_t static_columns;

            if (!benchmark_parse_u32(value, &static_columns)) {
                return 0;
            }
            options->num_static_input_column_capacity = (size_t)static_columns;
        } else if (strcmp(name, "--targets") == 0) {
            if (!benchmark_parse_size(value, &options->num_targets)) {
                return 0;
            }
        } else if (strcmp(name, "--settings") == 0) {
            if (!benchmark_parse_size(value, &options->num_settings)) {
                return 0;
            }
        } else if (strcmp(name, "--settings-per-cta") == 0) {
            if (!benchmark_parse_size(value, &options->settings_per_cta)) {
                return 0;
            }
        } else if (strcmp(name, "--rows") == 0) {
            if (!benchmark_parse_size(value, &options->num_rows)) {
                return 0;
            }
        } else if (strcmp(name, "--tile-rows") == 0) {
            if (!benchmark_parse_size(value, &options->tile_rows)) {
                return 0;
            }
        } else if (strcmp(name, "--threads") == 0) {
            if (!benchmark_parse_size(value, &options->threads_per_block)) {
                return 0;
            }
        } else if (strcmp(name, "--patch-instructions-per-ast") == 0) {
            if (!benchmark_parse_size(value, &options->patch_instructions_per_ast)) {
                return 0;
            }
        } else if (strcmp(name, "--warmups") == 0) {
            if (!benchmark_parse_size(value, &options->warmups)) {
                return 0;
            }
        } else if (strcmp(name, "--iterations") == 0) {
            if (!benchmark_parse_size(value, &options->iterations)) {
                return 0;
            }
        } else if (strcmp(name, "--seed") == 0) {
            if (!benchmark_parse_u32(value, &options->seed)) {
                return 0;
            }
        } else if (strcmp(name, "--target-sm") == 0) {
            if (!benchmark_parse_u32(value, &options->target_sm)) {
                return 0;
            }
        } else if (strcmp(name, "--device") == 0) {
            if (!benchmark_parse_int(value, &options->device_ordinal)) {
                return 0;
            }
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
    int success = 0;
    int architecture_size;

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
    result = nvrtcCreateProgram(&program, source, "secant_dynamic_leaf_sse.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, 4, options);
    }
    if (result == NVRTC_SUCCESS) {
        result = nvrtcGetCUBINSize(program, &cubin_size);
    }
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
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
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
    const SecantCubinDynamicLeafSSERecipe* recipe,
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
    if (result != SECANT_SUCCESS || source_size == 0u) {
        return 0;
    }
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
    if (mode == SECANT_BENCH_AST_MODE_SIMPLE) {
        return "simple";
    }
    if (mode == SECANT_BENCH_AST_MODE_ALU) {
        return "alu";
    }
    return "mufu";
}

static SecantResult
benchmark_runner_run(
    const BenchmarkOptions* options,
    SecantCubinRunner cubin_runner,
    SecantCUDARunner cuda_runner,
    SecantPTXRunner ptx_runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_device_address,
    size_t input_elements,
    uintptr_t masks_device_address,
    uintptr_t words_device_address,
    size_t leaf_word_elements,
    size_t num_settings,
    uintptr_t targets_device_address,
    size_t target_elements,
    size_t num_rows,
    uintptr_t output_device_address,
    size_t output_elements,
    SecantRunnerStats* stats_ret
) {
    if (options->backend == BENCHMARK_BACKEND_CUDA) {
        return secant_cuda_dynamic_leaf_sse_runner_run_all(
            cuda_runner,
            routines,
            num_routines,
            routine_names,
            asts,
            num_asts,
            input_device_address,
            input_elements,
            options->num_input_columns,
            options->num_rows,
            masks_device_address,
            options->num_settings,
            words_device_address,
            leaf_word_elements,
            options->num_dynamic_leaves,
            num_settings,
            options->settings_per_cta < num_settings ? options->settings_per_cta : num_settings,
            options->num_targets,
            targets_device_address,
            target_elements,
            options->num_rows,
            num_rows,
            output_device_address,
            output_elements,
            options->num_settings,
            stats_ret);
    }
    if (options->backend == BENCHMARK_BACKEND_PTX) {
        return secant_ptx_dynamic_leaf_sse_runner_run_all(
            ptx_runner,
            routines,
            num_routines,
            asts,
            num_asts,
            input_device_address,
            input_elements,
            options->num_input_columns,
            options->num_rows,
            masks_device_address,
            options->num_settings,
            words_device_address,
            leaf_word_elements,
            options->num_dynamic_leaves,
            num_settings,
            options->settings_per_cta < num_settings ? options->settings_per_cta : num_settings,
            options->num_targets,
            targets_device_address,
            target_elements,
            options->num_rows,
            num_rows,
            output_device_address,
            output_elements,
            options->num_settings,
            stats_ret);
    }
    return secant_bench_cubin_dynamic_leaf_sse_runner_run(
        cubin_runner,
        routines,
        num_routines,
        asts,
        num_asts,
        input_device_address,
        input_elements,
        options->num_input_columns,
        options->num_rows,
        masks_device_address,
        options->num_settings,
        words_device_address,
        leaf_word_elements,
        options->num_dynamic_leaves,
        num_settings,
        options->num_targets,
        targets_device_address,
        target_elements,
        options->num_rows,
        num_rows,
        output_device_address,
        output_elements,
        options->num_settings,
        stats_ret);
}

int
main(int argc, char** argv) {
    BenchmarkOptions options;
    SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    CUdevice device;
    CUcontext context = NULL;
    CUcontext previous_context = NULL;
    CUdeviceptr input_device = 0u;
    CUdeviceptr masks_device = 0u;
    CUdeviceptr words_device = 0u;
    CUdeviceptr targets_device = 0u;
    CUdeviceptr output_device = 0u;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    SecantCUDARunner cuda_runner = NULL;
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXRunner ptx_runner = NULL;
    void* plan_storage = NULL;
    char* source = NULL;
    unsigned char* cubin = NULL;
    SecantAstInstruction* programs = NULL;
    const SecantAstInstruction** asts = NULL;
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    float* input = NULL;
    float* targets = NULL;
    float* expected = NULL;
    float* actual = NULL;
    uint32_t* leaf_masks = NULL;
    uint32_t* leaf_words = NULL;
    char error[BENCH_ERROR_BYTES] = {0};
    size_t num_asts_per_module;
    size_t num_asts;
    size_t program_bytes;
    size_t pointer_bytes;
    size_t input_elements;
    size_t target_elements;
    size_t leaf_word_elements;
    size_t output_elements;
    size_t output_module_elements;
    size_t plan_storage_size = 0u;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t max_dynamic_leaves_used = 0u;
    size_t num_routines = 0u;
    size_t idx;
    size_t iteration;
    uint32_t target_sm;
    int major;
    int minor;
    int parse_result;
    int context_retained = 0;
    int success = 1;
    const char* stage = "validation";
    double template_seconds = 0.0;
    double pipeline_seconds = 0.0;
    SecantRunnerStats totals = secant_runner_stats_init();

    benchmark_options_default(&options);
    parse_result = benchmark_options_parse(argc, argv, &options);
    if (parse_result < 0) {
        benchmark_usage(argv[0]);
        return 0;
    }
    if (options.num_input_column_capacity == 0u) {
        options.num_input_column_capacity = options.num_input_columns;
    }
    if (options.num_dynamic_sites == 0u) {
        options.num_dynamic_sites = options.num_dynamic_leaves;
    }
    if (options.settings_per_cta == 0u) {
        options.settings_per_cta = options.num_settings;
    }
    if (options.settings_per_cta > options.num_settings) {
        options.settings_per_cta = options.num_settings;
    }
    if (parse_result == 0 || (options.ast_nodes != 0u && options.ast_mode == SECANT_BENCH_AST_MODE_SIMPLE) ||
        options.num_dynamic_sites > options.num_dynamic_leaves ||
        options.num_dynamic_leaves > SECANT_AST_MAX_DYNAMIC_LEAVES ||
        options.num_input_columns > options.num_input_column_capacity ||
        options.num_input_column_capacity > SECANT_AST_MAX_INPUTS || options.threads_per_block > 1024u ||
        (options.setting_owner == BENCHMARK_SETTING_OWNER_WARP &&
         (options.backend == BENCHMARK_BACKEND_CUBIN || options.threads_per_block % 32u != 0u)) ||
        (options.backend == BENCHMARK_BACKEND_CUBIN &&
         options.settings_per_cta != options.num_settings) ||
        (options.num_static_input_column_capacity != 0u &&
         options.num_static_input_column_capacity != options.num_input_column_capacity) ||
        options.num_streams > options.num_kernels ||
        !benchmark_checked_mul(options.num_kernels, options.asts_per_kernel, &num_asts_per_module) ||
        !benchmark_checked_mul(options.num_modules, num_asts_per_module, &num_asts) ||
        !benchmark_checked_mul(options.num_input_columns, options.num_rows, &input_elements) ||
        !benchmark_checked_mul(options.num_targets, options.num_rows, &target_elements) ||
        !benchmark_checked_mul(options.num_settings, options.num_dynamic_leaves, &leaf_word_elements) ||
        !benchmark_checked_mul(num_asts, options.num_targets, &output_elements) ||
        !benchmark_checked_mul(output_elements, options.num_settings, &output_elements) ||
        !benchmark_checked_mul(num_asts_per_module, options.num_targets, &output_module_elements) ||
        !benchmark_checked_mul(output_module_elements, options.num_settings, &output_module_elements) ||
        !secant_bench_ast_storage_sizes(
            options.num_modules, options.num_kernels, options.asts_per_kernel, &program_bytes, &pointer_bytes) ||
        options.asts_per_kernel > SIZE_MAX / options.patch_instructions_per_ast ||
        input_elements > SIZE_MAX / sizeof(float) || target_elements > SIZE_MAX / sizeof(float) ||
        leaf_word_elements > SIZE_MAX / sizeof(uint32_t) || output_elements > SIZE_MAX / sizeof(float) ||
        output_module_elements > SIZE_MAX / sizeof(float) || options.num_settings > SIZE_MAX / sizeof(uint32_t)) {
        benchmark_usage(argv[0]);
        return 1;
    }
    programs = malloc(program_bytes);
    asts = malloc(pointer_bytes);
    input = malloc(input_elements * sizeof(*input));
    targets = malloc(target_elements * sizeof(*targets));
    leaf_masks = malloc(options.num_settings * sizeof(*leaf_masks));
    leaf_words = malloc(leaf_word_elements * sizeof(*leaf_words));
    expected = calloc(output_module_elements, sizeof(*expected));
    actual = malloc(output_module_elements * sizeof(*actual));
    stage = "allocation";
    if (programs == NULL || asts == NULL || input == NULL || targets == NULL || leaf_masks == NULL ||
        leaf_words == NULL || expected == NULL || actual == NULL) {
        (void)snprintf(error, sizeof(error), "host allocation failed");
        success = 0;
    }
    if (success) {
        stage = "AST projection";
        if (options.ast_nodes == 0u) {
            secant_bench_ast_fill(
                options.num_modules,
                options.num_kernels,
                options.asts_per_kernel,
                options.num_input_columns,
                options.seed,
                options.ast_mode,
                options.cse_mode,
                programs,
                asts);
        } else {
            success = secant_bench_ast_imbalanced_fill(
                options.num_modules,
                options.num_kernels,
                options.asts_per_kernel,
                options.num_input_columns,
                options.ast_nodes,
                options.seed,
                options.ast_mode,
                programs,
                asts);
        }
        success = success && secant_bench_ast_dynamic_leaves_rewrite(
            num_asts,
            options.num_dynamic_leaves,
            options.num_dynamic_sites,
            options.num_static_input_column_capacity != 0u,
            programs,
            &max_dynamic_leaves_used);
        if (!success) {
            (void)snprintf(error, sizeof(error), "AST projection exceeded the configured dynamic-leaf capacity");
        }
        secant_bench_ast_get_routines(options.ast_mode, &routines, &num_routines, &routine_names);
    }
    if (success) {
        stage = "host data generation";
        for (idx = 0u; idx < input_elements; ++idx) {
            input[idx] = benchmark_value(idx / options.num_rows, idx % options.num_rows, options.seed);
        }
        for (idx = 0u; idx < target_elements; ++idx) {
            targets[idx] = benchmark_value(idx / options.num_rows, idx % options.num_rows, options.seed + 31u);
        }
        for (idx = 0u; idx < leaf_word_elements; ++idx) {
            const size_t setting = idx / options.num_dynamic_leaves;
            const size_t leaf = idx % options.num_dynamic_leaves;
            const uint32_t hash = secant_bench_ast_hash32(
                (uint32_t)setting * UINT32_C(0x9e3779b9) ^ (uint32_t)leaf * UINT32_C(0x85ebca6b) ^ options.seed);

            if (leaf == 0u) {
                leaf_masks[setting] = options.dynamic_values == BENCHMARK_DYNAMIC_VALUES_COLUMNS
                    ? (options.num_dynamic_leaves == 32u
                        ? UINT32_MAX
                        : (UINT32_C(1) << options.num_dynamic_leaves) - UINT32_C(1))
                    : 0u;
            }
            if (options.dynamic_values == BENCHMARK_DYNAMIC_VALUES_COLUMNS ||
                setting == 0u || (setting > 1u && (hash & 1u) != 0u)) {
                leaf_masks[setting] |= UINT32_C(1) << leaf;
                leaf_words[idx] = hash % options.num_input_columns;
            } else {
                leaf_words[idx] = benchmark_f32_bits(benchmark_value(setting, leaf, options.seed + 17u));
            }
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
        recipe.num_input_columns = options.num_input_column_capacity;
        recipe.num_static_input_columns = options.num_static_input_column_capacity;
        recipe.num_dynamic_leaves = options.num_dynamic_leaves;
        recipe.num_targets = options.num_targets;
        recipe.tile_rows = options.tile_rows;
        recipe.threads_per_block = options.threads_per_block;
        recipe.patch_capacity_instructions = options.asts_per_kernel * options.patch_instructions_per_ast;
    }
    if (success) {
        const double begin = benchmark_seconds_get();

        stage = "template compilation";
        if (options.backend == BENCHMARK_BACKEND_CUBIN) {
            success = benchmark_template_create(
                &recipe, target_sm, &source, &source_size, &cubin, &cubin_size, error, sizeof(error));
        } else if (options.backend == BENCHMARK_BACKEND_PTX) {
            static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
            const uint32_t source_major = major >= 8 ? 8u : (uint32_t)major;
            const uint32_t source_minor = major >= 8 ? 0u : (uint32_t)minor;
            size_t log_size = 0u;
            const SecantPTXResult ptx_result = options.setting_owner == BENCHMARK_SETTING_OWNER_WARP
                ? secant_ptx_warp_dynamic_leaf_sse_create(
                    options.num_kernels,
                    options.asts_per_kernel,
                    options.num_input_column_capacity,
                    options.num_static_input_column_capacity,
                    options.num_dynamic_leaves,
                    options.num_targets,
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
                    &ptx_handle)
                : secant_ptx_dynamic_leaf_sse_create(
                    options.num_kernels,
                    options.asts_per_kernel,
                    options.num_input_column_capacity,
                    options.num_static_input_column_capacity,
                    options.num_dynamic_leaves,
                    options.num_targets,
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

            success = ptx_result == SECANT_PTX_SUCCESS;
        }
        template_seconds = benchmark_seconds_get() - begin;
    }
    if (success && options.backend == BENCHMARK_BACKEND_CUBIN) {
        stage = "CUBIN inspection measure";
        if (secant_bench_cubin_inspect(
                &recipe, cubin, cubin_size, NULL, 0u, &plan_storage_size, &plan) != SECANT_SUCCESS ||
            plan_storage_size == 0u) {
            success = 0;
        }
    }
    if (success) {
        static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
        static const char* const nvptx_options[] = { "--opt-level=1" };
        SecantResult create_result = SECANT_SUCCESS;

        stage = "runner creation";
        if (options.backend == BENCHMARK_BACKEND_CUBIN) {
            plan_storage = malloc(plan_storage_size);
            if (plan_storage == NULL ||
                secant_bench_cubin_inspect(
                    &recipe, cubin, cubin_size, plan_storage, plan_storage_size,
                    &plan_storage_size, &plan) != SECANT_SUCCESS) {
                create_result = SECANT_ERROR_BAD_BINARY;
            } else {
                create_result = secant_bench_cubin_runner_create(
                    plan, cubin, cubin_size, options.num_workers, options.num_streams, &runner);
            }
        } else if (options.backend == BENCHMARK_BACKEND_CUDA) {
            create_result = options.setting_owner == BENCHMARK_SETTING_OWNER_WARP
                ? secant_cuda_warp_dynamic_leaf_sse_runner_create(
                    options.num_kernels,
                    options.asts_per_kernel,
                    options.num_input_column_capacity,
                    options.num_static_input_column_capacity,
                    options.num_dynamic_leaves,
                    options.num_targets,
                    options.tile_rows,
                    options.threads_per_block,
                    (uint32_t)major,
                    (uint32_t)minor,
                    nvrtc_options,
                    2u,
                    BENCH_COMPILE_SCRATCH_SIZE,
                    options.num_workers,
                    options.num_streams,
                    &cuda_runner)
                : secant_cuda_dynamic_leaf_sse_runner_create(
                    options.num_kernels,
                    options.asts_per_kernel,
                    options.num_input_column_capacity,
                    options.num_static_input_column_capacity,
                    options.num_dynamic_leaves,
                    options.num_targets,
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
            create_result = secant_ptx_dynamic_leaf_sse_runner_create(
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
        stage = "device allocation and upload";
        if (cuMemAlloc(&input_device, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&masks_device, options.num_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemAlloc(&words_device, leaf_word_elements * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemAlloc(&targets_device, target_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&output_device, output_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(input_device, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(masks_device, leaf_masks, options.num_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(words_device, leaf_words, leaf_word_elements * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(targets_device, targets, target_elements * sizeof(float)) != CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success) {
        const size_t check_rows = options.num_rows < 257u ? options.num_rows : 257u;
        const size_t check_settings = options.num_settings < 4u ? options.num_settings : 4u;
        SecantRunnerStats verify_stats = secant_runner_stats_init();
        SecantResult result;

        stage = "CPU/GPU verification";
        result = secant_bench_cpu_dynamic_leaf_sse_run(
            options.num_dynamic_leaves,
            options.num_input_columns,
            options.num_static_input_column_capacity != 0u ? options.num_input_columns : 0u,
            options.num_targets,
            routines,
            num_routines,
            asts,
            num_asts_per_module,
            input,
            input_elements,
            options.num_rows,
            leaf_masks,
            options.num_settings,
            leaf_words,
            leaf_word_elements,
            options.num_dynamic_leaves,
            check_settings,
            targets,
            target_elements,
            options.num_rows,
            check_rows,
            expected,
            output_module_elements,
            options.num_settings);
        if (result != SECANT_SUCCESS) {
            (void)snprintf(error, sizeof(error), "CPU oracle: %s", secant_result_to_string(result));
            success = 0;
        }
        if (success) {
            result = benchmark_runner_run(
                &options,
                runner,
                cuda_runner,
                ptx_runner,
                routines,
                num_routines,
                routine_names,
                asts,
                num_asts,
                (uintptr_t)input_device,
                input_elements,
                (uintptr_t)masks_device,
                (uintptr_t)words_device,
                leaf_word_elements,
                check_settings,
                (uintptr_t)targets_device,
                target_elements,
                check_rows,
                (uintptr_t)output_device,
                output_elements,
                &verify_stats);
            if (result != SECANT_SUCCESS) {
                (void)snprintf(error, sizeof(error), "runner: %s", secant_result_to_string(result));
                success = 0;
            }
        }
        if (success && cuMemcpyDtoH(actual, output_device, output_module_elements * sizeof(float)) != CUDA_SUCCESS) {
            (void)snprintf(error, sizeof(error), "CUDA output copy failed");
            success = 0;
        }
        for (idx = 0u; success && idx < num_asts_per_module * options.num_targets; ++idx) {
            size_t setting;

            for (setting = 0u; setting < check_settings; ++setting) {
                const size_t output_idx = idx * options.num_settings + setting;
                const float tolerance = (options.ast_mode == SECANT_BENCH_AST_MODE_MUFU ? 5.0e-2f : 1.0e-3f) *
                    (1.0f + fabsf(expected[output_idx]));

                if (!isfinite(expected[output_idx]) || !isfinite(actual[output_idx]) ||
                    fabsf(actual[output_idx] - expected[output_idx]) > tolerance) {
                    fprintf(stderr, "verify mismatch output=%zu setting=%zu expected=%.9g actual=%.9g\n",
                        idx, setting, expected[output_idx], actual[output_idx]);
                    success = 0;
                    break;
                }
            }
        }
    }
    if (success) {
        stage = "timed runner execution";
    }
    for (iteration = 0u; success && iteration < options.warmups + options.iterations; ++iteration) {
        SecantRunnerStats stats = secant_runner_stats_init();
        const double pipeline_begin = benchmark_seconds_get();
        const SecantResult result = benchmark_runner_run(
            &options,
            runner,
            cuda_runner,
            ptx_runner,
            routines,
            num_routines,
            routine_names,
            asts,
            num_asts,
            (uintptr_t)input_device,
            input_elements,
            (uintptr_t)masks_device,
            (uintptr_t)words_device,
            leaf_word_elements,
            options.num_settings,
            (uintptr_t)targets_device,
            target_elements,
            options.num_rows,
            (uintptr_t)output_device,
            output_elements,
            &stats);
        const double pipeline_elapsed = benchmark_seconds_get() - pipeline_begin;

        if (result != SECANT_SUCCESS) {
            snprintf(error, sizeof(error), "%s", secant_result_to_string(result));
            success = 0;
        } else if (iteration >= options.warmups) {
            totals.compile_window_seconds += stats.compile_window_seconds;
            totals.compile_critical_seconds += stats.compile_critical_seconds;
            totals.compile_work_seconds += stats.compile_work_seconds;
            totals.module_load_seconds += stats.module_load_seconds;
            totals.completion_wait_seconds += stats.completion_wait_seconds;
            totals.module_unload_seconds += stats.module_unload_seconds;
            totals.runtime_seconds += stats.runtime_seconds;
            totals.total_seconds += stats.total_seconds;
            pipeline_seconds += pipeline_elapsed;
        }
    }
    if (success &&
        (totals.compile_window_seconds <= 0.0 || totals.runtime_seconds <= 0.0 || totals.total_seconds <= 0.0 ||
         pipeline_seconds <= 0.0)) {
        snprintf(error, sizeof(error), "non-positive measured duration");
        success = 0;
    }
    if (success) {
        const double ast_compiles = (double)options.iterations * (double)num_asts;
        const double row_evals = ast_compiles * (double)options.num_settings * (double)options.num_rows;

        printf(
            "backend=%s shape=dynamic_leaf_sse setting_owner=%s ast_mode=%s cse=%s dynamic_values=%s "
            "modules=%zu workers=%zu "
            "streams=%zu "
            "kernels=%zu asts_per_kernel=%zu asts=%zu ast_nodes=%zu leaves=%zu dynamic_sites=%zu "
            "leaves_used=%zu columns=%zu "
            "column_capacity=%zu static_columns=%zu static_column_capacity=%zu targets=%zu "
            "settings=%zu settings_per_cta=%zu setting_ctas=%zu rows=%zu tile_rows=%zu threads=%zu "
            "iterations=%zu template_seconds=%.6f "
            "compile_seconds=%.6f compile_asts_per_second=%.3f module_load_seconds=%.6f runtime_seconds=%.6f "
            "runner_pipeline_seconds=%.6f pipeline_seconds=%.6f runtime_row_evals_per_second=%.3e "
            "pipeline_row_evals_per_second=%.3e verify=pass\n",
            options.backend == BENCHMARK_BACKEND_CUDA ? "cuda" :
                options.backend == BENCHMARK_BACKEND_PTX ? "ptx" : "cubin",
            options.setting_owner == BENCHMARK_SETTING_OWNER_WARP ? "warp" : "thread",
            benchmark_ast_mode_name(options.ast_mode),
            options.cse_mode == SECANT_BENCH_CSE_SHARED ? "shared" : "distinct",
            options.dynamic_values == BENCHMARK_DYNAMIC_VALUES_COLUMNS ? "columns" : "mixed",
            options.num_modules,
            options.num_workers,
            options.num_streams,
            options.num_kernels,
            options.asts_per_kernel,
            num_asts,
            options.ast_nodes,
            options.num_dynamic_leaves,
            options.num_dynamic_sites,
            max_dynamic_leaves_used,
            options.num_input_columns,
            options.num_input_column_capacity,
            options.num_static_input_column_capacity != 0u ? options.num_input_columns : 0u,
            options.num_static_input_column_capacity,
            options.num_targets,
            options.num_settings,
            options.settings_per_cta,
            options.num_settings / options.settings_per_cta +
                (options.num_settings % options.settings_per_cta != 0u ? 1u : 0u),
            options.num_rows,
            options.tile_rows,
            options.threads_per_block,
            options.iterations,
            template_seconds,
            totals.compile_window_seconds,
            ast_compiles / totals.compile_window_seconds,
            totals.module_load_seconds,
            totals.runtime_seconds,
            totals.total_seconds,
            pipeline_seconds,
            row_evals / totals.runtime_seconds,
            row_evals / pipeline_seconds);
    } else {
        fprintf(stderr, "dynamic_leaf_sse benchmark failed during %s: %s\n",
            stage, error[0] != '\0' ? error : "unknown error");
    }
    if (ptx_runner != NULL) {
        (void)secant_ptx_runner_destroy(ptx_runner);
    }
    if (ptx_handle != NULL) {
        (void)secant_ptx_handle_destroy(ptx_handle);
    }
    if (cuda_runner != NULL) {
        (void)secant_cuda_runner_destroy(cuda_runner);
    }
    if (runner != NULL) {
        (void)secant_cubin_runner_destroy(runner);
    }
    if (output_device != 0u) {
        (void)cuMemFree(output_device);
    }
    if (targets_device != 0u) {
        (void)cuMemFree(targets_device);
    }
    if (words_device != 0u) {
        (void)cuMemFree(words_device);
    }
    if (masks_device != 0u) {
        (void)cuMemFree(masks_device);
    }
    if (input_device != 0u) {
        (void)cuMemFree(input_device);
    }
    free(actual);
    free(expected);
    free(leaf_words);
    free(leaf_masks);
    free(targets);
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
