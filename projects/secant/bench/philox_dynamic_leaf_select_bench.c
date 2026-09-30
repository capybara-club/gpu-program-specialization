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
#include "backends/cuda/secant_cuda.h"
#include "secant.h"

#include <cuda.h>

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_COMPILE_SCRATCH_BYTES (64u * 1024u * 1024u)
#define BENCH_LOG_BYTES 16384u

typedef struct BenchmarkOptions {
    SecantBenchAstMode ast_mode;
    SecantBenchCSEMode cse_mode;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_input_columns;
    size_t num_static_input_columns;
    size_t num_dynamic_leaves;
    size_t num_dynamic_sites;
    size_t num_rows;
    size_t tile_rows;
    size_t threads_per_block;
    size_t num_setting_blocks;
    size_t num_setting_passes;
    size_t num_streams;
    size_t warmups;
    size_t iterations;
    uint64_t seed;
    uint64_t epoch;
    uint64_t setting_offset;
    double column_probability;
    float constant_radius;
    int write_sse;
    int device_ordinal;
} BenchmarkOptions;

typedef struct BenchmarkPhilox4x32 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
} BenchmarkPhilox4x32;

static int
benchmark_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static double
benchmark_seconds_get(void) {
    struct timespec value;

    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static float
benchmark_value(size_t outer, size_t inner, uint64_t seed) {
    const uint32_t hash = secant_bench_ast_hash32(
        (uint32_t)outer * UINT32_C(0x9e3779b9) ^
        (uint32_t)inner * UINT32_C(0x85ebca6b) ^
        (uint32_t)seed ^ (uint32_t)(seed >> 32u));

    return ((float)(hash & UINT32_C(0xffff)) / 65535.0f) * 1.5f - 0.75f;
}

static uint32_t
benchmark_f32_bits(float value) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
benchmark_f32_from_bits(uint32_t bits) {
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static BenchmarkPhilox4x32
benchmark_philox_round(BenchmarkPhilox4x32 counter, uint32_t key0, uint32_t key1) {
    const uint64_t product0 = UINT64_C(0xd2511f53) * counter.x;
    const uint64_t product1 = UINT64_C(0xcd9e8d57) * counter.z;
    BenchmarkPhilox4x32 result;

    result.x = (uint32_t)(product1 >> 32u) ^ counter.y ^ key0;
    result.y = (uint32_t)product1;
    result.z = (uint32_t)(product0 >> 32u) ^ counter.w ^ key1;
    result.w = (uint32_t)product0;
    return result;
}

static BenchmarkPhilox4x32
benchmark_philox_random4(uint64_t setting, uint64_t seed, uint64_t epoch, uint32_t leaf) {
    BenchmarkPhilox4x32 counter;
    uint32_t key0 = (uint32_t)seed ^ (uint32_t)(epoch >> 32u) * UINT32_C(0x9e3779b9);
    uint32_t key1 = (uint32_t)(seed >> 32u) ^ (uint32_t)epoch * UINT32_C(0xbb67ae85);
    unsigned int round;

    counter.x = (uint32_t)setting;
    counter.y = (uint32_t)(setting >> 32u);
    counter.z = leaf;
    counter.w = (uint32_t)epoch;
    for (round = 0u; round < 10u; ++round) {
        counter = benchmark_philox_round(counter, key0, key1);
        key0 += UINT32_C(0x9e3779b9);
        key1 += UINT32_C(0xbb67ae85);
    }
    return counter;
}

static float
benchmark_philox_constant(uint32_t word, float radius) {
    return radius * ((float)(word >> 8u) * 1.1920928955078125e-7f - 1.0f);
}

static void
benchmark_setting_write(
    uint64_t setting,
    uint64_t seed,
    uint64_t epoch,
    uint32_t column_threshold,
    float constant_radius,
    size_t num_input_columns,
    size_t num_dynamic_leaves,
    uint32_t* mask_ret,
    uint32_t* words
) {
    uint32_t mask = 0u;
    size_t leaf;

    for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
        const BenchmarkPhilox4x32 random = benchmark_philox_random4(setting, seed, epoch, (uint32_t)leaf);
        const int is_column = random.x < column_threshold;

        if (is_column) {
            const uint32_t column = (uint32_t)(((uint64_t)random.y * (uint32_t)num_input_columns) >> 32u);

            mask |= UINT32_C(1) << leaf;
            words[leaf] = column;
        } else {
            words[leaf] = benchmark_f32_bits(benchmark_philox_constant(random.z, constant_radius));
        }
    }
    *mask_ret = mask;
}

static void
benchmark_options_default(BenchmarkOptions* options) {
    memset(options, 0, sizeof(*options));
    options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
    options->cse_mode = SECANT_BENCH_CSE_DISTINCT;
    options->num_kernels = 64u;
    options->asts_per_kernel = 32u;
    options->num_input_columns = 8u;
    options->num_static_input_columns = 8u;
    options->num_dynamic_leaves = 8u;
    options->num_dynamic_sites = 4u;
    options->num_rows = 256u;
    options->tile_rows = 256u;
    options->threads_per_block = 128u;
    options->num_setting_blocks = 16u;
    options->num_setting_passes = 4u;
    options->num_streams = 64u;
    options->warmups = 1u;
    options->iterations = 5u;
    options->seed = 1u;
    options->epoch = 0u;
    options->setting_offset = 0u;
    options->column_probability = 0.5;
    options->constant_radius = 8.0f;
    options->write_sse = 1;
    options->device_ordinal = 0;
}

static void
benchmark_usage(const char* program) {
    fprintf(stderr,
        "usage: %s [--ast-mode simple|alu|mufu] [--cse shared|distinct] "
        "[--kernels N] [--asts-per-kernel N] [--columns N] [--static-columns 0|N] "
        "[--leaves N] [--dynamic-sites N] [--rows N] [--tile-rows N] [--threads N] "
        "[--setting-blocks N] [--passes N] [--streams N] [--warmups N] [--iterations N] "
        "[--seed N] [--epoch N] [--setting-offset N] [--column-probability F] "
        "[--constant-radius F] [--write-sse 0|1] [--device N]\n",
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
benchmark_parse_u64(const char* text, uint64_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0') {
        return 0;
    }
    *value_ret = (uint64_t)value;
    return 1;
}

static int
benchmark_parse_double(const char* text, double* value_ret) {
    char* end = NULL;
    double value;

    errno = 0;
    value = strtod(text, &end);
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
        size_t* size_ret = NULL;

        if (strcmp(name, "--help") == 0) {
            return -1;
        }
        if (++arg_idx >= argc) {
            return 0;
        }
        value = argv[arg_idx];
        if (strcmp(name, "--kernels") == 0) size_ret = &options->num_kernels;
        else if (strcmp(name, "--asts-per-kernel") == 0) size_ret = &options->asts_per_kernel;
        else if (strcmp(name, "--columns") == 0) size_ret = &options->num_input_columns;
        else if (strcmp(name, "--leaves") == 0) size_ret = &options->num_dynamic_leaves;
        else if (strcmp(name, "--dynamic-sites") == 0) size_ret = &options->num_dynamic_sites;
        else if (strcmp(name, "--rows") == 0) size_ret = &options->num_rows;
        else if (strcmp(name, "--tile-rows") == 0) size_ret = &options->tile_rows;
        else if (strcmp(name, "--threads") == 0) size_ret = &options->threads_per_block;
        else if (strcmp(name, "--setting-blocks") == 0) size_ret = &options->num_setting_blocks;
        else if (strcmp(name, "--passes") == 0) size_ret = &options->num_setting_passes;
        else if (strcmp(name, "--streams") == 0) size_ret = &options->num_streams;
        else if (strcmp(name, "--warmups") == 0) size_ret = &options->warmups;
        else if (strcmp(name, "--iterations") == 0) size_ret = &options->iterations;
        if (size_ret != NULL) {
            if (!benchmark_parse_size(value, size_ret)) {
                return 0;
            }
        } else if (strcmp(name, "--static-columns") == 0) {
            uint64_t parsed;

            if (!benchmark_parse_u64(value, &parsed) || parsed > SIZE_MAX) return 0;
            options->num_static_input_columns = (size_t)parsed;
        } else if (strcmp(name, "--seed") == 0) {
            if (!benchmark_parse_u64(value, &options->seed)) return 0;
        } else if (strcmp(name, "--epoch") == 0) {
            if (!benchmark_parse_u64(value, &options->epoch)) return 0;
        } else if (strcmp(name, "--setting-offset") == 0) {
            if (!benchmark_parse_u64(value, &options->setting_offset)) return 0;
        } else if (strcmp(name, "--column-probability") == 0) {
            if (!benchmark_parse_double(value, &options->column_probability)) return 0;
        } else if (strcmp(name, "--constant-radius") == 0) {
            double parsed;

            if (!benchmark_parse_double(value, &parsed) || parsed <= 0.0 || parsed > (double)FLT_MAX) return 0;
            options->constant_radius = (float)parsed;
        } else if (strcmp(name, "--write-sse") == 0) {
            uint64_t parsed;

            if (!benchmark_parse_u64(value, &parsed) || parsed > 1u) return 0;
            options->write_sse = (int)parsed;
        } else if (strcmp(name, "--device") == 0) {
            uint64_t parsed;

            if (!benchmark_parse_u64(value, &parsed) || parsed > INT32_MAX) return 0;
            options->device_ordinal = (int)parsed;
        } else if (strcmp(name, "--ast-mode") == 0) {
            if (strcmp(value, "simple") == 0) options->ast_mode = SECANT_BENCH_AST_MODE_SIMPLE;
            else if (strcmp(value, "alu") == 0) options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
            else if (strcmp(value, "mufu") == 0) options->ast_mode = SECANT_BENCH_AST_MODE_MUFU;
            else return 0;
        } else if (strcmp(name, "--cse") == 0) {
            if (strcmp(value, "shared") == 0) options->cse_mode = SECANT_BENCH_CSE_SHARED;
            else if (strcmp(value, "distinct") == 0) options->cse_mode = SECANT_BENCH_CSE_DISTINCT;
            else return 0;
        } else {
            return 0;
        }
    }
    return 1;
}

static uint32_t
benchmark_column_threshold(double probability) {
    if (probability <= 0.0) {
        return 0u;
    }
    if (probability >= 1.0) {
        return UINT32_MAX;
    }
    return (uint32_t)(probability * 4294967296.0);
}

static int
benchmark_cpu_verify(
    const BenchmarkOptions* options,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    const float* input,
    const float* target,
    const uint32_t* gpu_indices,
    const float* gpu_sse,
    const uint32_t* gpu_ast_indices,
    const uint32_t* gpu_masks,
    const uint32_t* gpu_words,
    size_t verify_blocks
) {
    const size_t num_settings = verify_blocks * options->threads_per_block * options->num_setting_passes;
    const size_t num_outputs = verify_blocks * options->threads_per_block;
    const size_t num_asts = options->asts_per_kernel;
    uint32_t* masks = NULL;
    uint32_t* words = NULL;
    float* all_sse = NULL;
    SecantCpuDynamicLeafSSERun run = secant_cpu_dynamic_leaf_sse_run_init();
    uint32_t column_threshold = benchmark_column_threshold(options->column_probability);
    size_t setting;
    size_t output_idx;
    int success = 1;

    masks = (uint32_t*)malloc(num_settings * sizeof(*masks));
    words = (uint32_t*)malloc(num_settings * options->num_dynamic_leaves * sizeof(*words));
    all_sse = (float*)calloc(num_asts * num_settings, sizeof(*all_sse));
    if (masks == NULL || words == NULL || all_sse == NULL) {
        success = 0;
    }
    for (setting = 0u; success && setting < num_settings; ++setting) {
        benchmark_setting_write(
            options->setting_offset + setting,
            options->seed,
            options->epoch,
            column_threshold,
            options->constant_radius,
            options->num_input_columns,
            options->num_dynamic_leaves,
            &masks[setting],
            words + setting * options->num_dynamic_leaves);
    }
    if (success) {
        run.programs.routines.items = routines;
        run.programs.routines.count = num_routines;
        run.programs.asts.items = asts;
        run.programs.asts.count = num_asts;
        run.num_dynamic_leaves = options->num_dynamic_leaves;
        run.num_input_columns = options->num_input_columns;
        run.num_static_input_columns = options->num_static_input_columns;
        run.num_targets = 1u;
        run.input.data = input;
        run.input.num_elements = options->num_input_columns * options->num_rows;
        run.input.leading_dimension = options->num_rows;
        run.leaf_masks.data = masks;
        run.leaf_masks.num_elements = num_settings;
        run.leaf_words.data = words;
        run.leaf_words.num_elements = num_settings * options->num_dynamic_leaves;
        run.leaf_words.leading_dimension = options->num_dynamic_leaves;
        run.targets.data = target;
        run.targets.num_elements = options->num_rows;
        run.targets.leading_dimension = options->num_rows;
        run.num_rows = options->num_rows;
        run.num_settings = num_settings;
        run.output.data = all_sse;
        run.output.num_elements = num_asts * num_settings;
        run.output.leading_dimension = num_settings;
        success = secant_cpu_run_dynamic_leaf_sse(&run) == SECANT_SUCCESS;
    }
    for (output_idx = 0u; success && output_idx < num_outputs; ++output_idx) {
        const size_t setting_block = output_idx / options->threads_per_block;
        const size_t setting_thread = output_idx % options->threads_per_block;
        float best_sse = INFINITY;
        uint32_t best_index = UINT32_MAX;
        size_t pass;

        for (pass = 0u; pass < options->num_setting_passes; ++pass) {
            const size_t setting_idx = (setting_block * options->num_setting_passes + pass) *
                options->threads_per_block + setting_thread;
            size_t ast_idx;

            for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
                const float candidate = all_sse[ast_idx * num_settings + setting_idx];
                const uint32_t candidate_idx = (uint32_t)(pass * options->asts_per_kernel + ast_idx);

                if (isfinite(candidate) && candidate >= 0.0f &&
                    (candidate < best_sse || (candidate == best_sse && candidate_idx < best_index))) {
                    best_sse = candidate;
                    best_index = candidate_idx;
                }
            }
        }
        if (gpu_indices[output_idx] == UINT32_MAX) {
            fprintf(stderr, "GPU emitted no winner at output %zu\n", output_idx);
            success = 0;
        } else {
            const size_t gpu_pass = gpu_indices[output_idx] / options->asts_per_kernel;
            const size_t gpu_ast = gpu_indices[output_idx] % options->asts_per_kernel;
            const size_t gpu_setting = (setting_block * options->num_setting_passes + gpu_pass) *
                options->threads_per_block + setting_thread;
            const float cpu_gpu_candidate = all_sse[gpu_ast * num_settings + gpu_setting];
            const float tolerance_scale = options->ast_mode == SECANT_BENCH_AST_MODE_MUFU ? 5.0e-2f : 2.0e-3f;
            const float tolerance = tolerance_scale * (1.0f + fabsf(best_sse));
            const uint32_t expected_mask = masks[gpu_setting];
            size_t leaf;

            if (!isfinite(cpu_gpu_candidate) || cpu_gpu_candidate > best_sse + tolerance ||
                (gpu_sse != NULL && fabsf(gpu_sse[output_idx] - cpu_gpu_candidate) > tolerance) ||
                gpu_ast_indices[output_idx] != gpu_ast || gpu_masks[output_idx] != expected_mask) {
                fprintf(stderr,
                    "winner mismatch output=%zu cpu_best=%u gpu=%u cpu_sse=%.9g gpu_candidate=%.9g gpu_sse=%.9g\n",
                    output_idx,
                    best_index,
                    gpu_indices[output_idx],
                    best_sse,
                    cpu_gpu_candidate,
                    gpu_sse != NULL ? gpu_sse[output_idx] : cpu_gpu_candidate);
                success = 0;
            }
            for (leaf = 0u; success && leaf < options->num_dynamic_leaves; ++leaf) {
                const uint32_t expected_word = words[gpu_setting * options->num_dynamic_leaves + leaf];
                const uint32_t actual_word = gpu_words[output_idx * options->num_dynamic_leaves + leaf];

                if ((expected_mask & (UINT32_C(1) << leaf)) != 0u) {
                    if (actual_word != expected_word) {
                        success = 0;
                    }
                } else if (fabsf(benchmark_f32_from_bits(actual_word) - benchmark_f32_from_bits(expected_word)) >
                           2.0e-6f * (1.0f + fabsf(benchmark_f32_from_bits(expected_word)))) {
                    success = 0;
                }
                if (!success) {
                    fprintf(stderr,
                        "reconstruction mismatch output=%zu leaf=%zu expected=0x%08x actual=0x%08x\n",
                        output_idx,
                        leaf,
                        expected_word,
                        actual_word);
                }
            }
        }
    }
    free(all_sse);
    free(words);
    free(masks);
    return success;
}

int
main(int argc, char** argv) {
    static const char* const nvrtc_options[] = { "--restrict", "--no-cache", "--ptxas-options=--opt-level=1" };
    BenchmarkOptions options;
    CUdevice device;
    CUcontext context = NULL;
    CUcontext previous_context = NULL;
    CUmodule module = NULL;
    CUfunction reconstruct_function = NULL;
    CUfunction* functions = NULL;
    CUstream* streams = NULL;
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_target = 0u;
    CUdeviceptr device_indices = 0u;
    CUdeviceptr device_sse = 0u;
    CUdeviceptr device_ast_indices = 0u;
    CUdeviceptr device_masks = 0u;
    CUdeviceptr device_words = 0u;
    SecantCUDACompiled compiled = NULL;
    SecantAstInstruction* programs = NULL;
    const SecantAstInstruction** asts = NULL;
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    void* compile_scratch = NULL;
    float* input = NULL;
    float* target = NULL;
    uint32_t* verify_indices = NULL;
    float* verify_sse = NULL;
    uint32_t* verify_ast_indices = NULL;
    uint32_t* verify_masks = NULL;
    uint32_t* verify_words = NULL;
    const void* cubin = NULL;
    size_t cubin_size = 0u;
    size_t program_bytes;
    size_t pointer_bytes;
    size_t num_asts;
    size_t input_elements;
    size_t outputs_per_kernel;
    size_t output_elements;
    size_t word_elements;
    size_t num_routines = 0u;
    size_t max_dynamic_leaves_used = 0u;
    size_t kernel_idx;
    size_t iteration;
    size_t verify_blocks;
    uint32_t column_threshold;
    int major = 0;
    int minor = 0;
    int retained = 0;
    int parse_result;
    int success = 1;
    char log[BENCH_LOG_BYTES] = {0};
    size_t log_size = 0u;
    double compile_seconds = 0.0;
    double runtime_seconds = 0.0;
    const char* stage = "argument validation";

    benchmark_options_default(&options);
    parse_result = benchmark_options_parse(argc, argv, &options);
    if (parse_result < 0) {
        benchmark_usage(argv[0]);
        return 0;
    }
    if (parse_result == 0 || options.num_rows > options.tile_rows ||
        options.num_input_columns > 32u || options.num_dynamic_leaves > 32u ||
        (options.num_static_input_columns != 0u &&
         options.num_static_input_columns != options.num_input_columns) ||
        options.num_dynamic_sites > options.num_dynamic_leaves ||
        options.threads_per_block > 1024u || options.num_streams > options.num_kernels ||
        options.column_probability < 0.0 || options.column_probability > 1.0 ||
        options.num_setting_passes > UINT32_MAX / options.asts_per_kernel ||
        !benchmark_checked_mul(options.num_kernels, options.asts_per_kernel, &num_asts) ||
        !benchmark_checked_mul(options.num_input_columns, options.num_rows, &input_elements) ||
        !benchmark_checked_mul(options.num_setting_blocks, options.threads_per_block, &outputs_per_kernel) ||
        !benchmark_checked_mul(options.num_kernels, outputs_per_kernel, &output_elements) ||
        !benchmark_checked_mul(output_elements, options.num_dynamic_leaves, &word_elements) ||
        !secant_bench_ast_storage_sizes(1u, options.num_kernels, options.asts_per_kernel,
            &program_bytes, &pointer_bytes) ||
        input_elements > SIZE_MAX / sizeof(float) || output_elements > SIZE_MAX / sizeof(float) ||
        output_elements > SIZE_MAX / sizeof(uint32_t) || word_elements > SIZE_MAX / sizeof(uint32_t)) {
        benchmark_usage(argv[0]);
        return 1;
    }
    programs = (SecantAstInstruction*)malloc(program_bytes);
    asts = (const SecantAstInstruction**)malloc(pointer_bytes);
    compile_scratch = malloc(BENCH_COMPILE_SCRATCH_BYTES);
    input = (float*)malloc(input_elements * sizeof(*input));
    target = (float*)malloc(options.num_rows * sizeof(*target));
    functions = (CUfunction*)calloc(options.num_kernels, sizeof(*functions));
    streams = (CUstream*)calloc(options.num_streams, sizeof(*streams));
    if (programs == NULL || asts == NULL || compile_scratch == NULL || input == NULL || target == NULL ||
        functions == NULL || streams == NULL) {
        success = 0;
        stage = "host allocation";
    }
    if (success) {
        size_t idx;

        secant_bench_ast_fill(
            1u,
            options.num_kernels,
            options.asts_per_kernel,
            options.num_input_columns,
            (uint32_t)options.seed,
            options.ast_mode,
            options.cse_mode,
            programs,
            asts);
        success = secant_bench_ast_dynamic_leaves_rewrite(
            num_asts,
            options.num_dynamic_leaves,
            options.num_dynamic_sites,
            options.num_static_input_columns != 0u,
            programs,
            &max_dynamic_leaves_used);
        secant_bench_ast_get_routines(options.ast_mode, &routines, &num_routines, &routine_names);
        for (idx = 0u; idx < input_elements; ++idx) {
            input[idx] = benchmark_value(idx / options.num_rows, idx % options.num_rows, options.seed);
        }
        for (idx = 0u; idx < options.num_rows; ++idx) {
            target[idx] = benchmark_value(0u, idx, options.seed + 31u);
        }
        stage = "AST generation";
    }
    if (success) {
        stage = "CUDA context creation";
        if (cuInit(0u) != CUDA_SUCCESS || cuDeviceGet(&device, options.device_ordinal) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
            success = 0;
        } else {
            retained = 1;
            success = cuCtxGetCurrent(&previous_context) == CUDA_SUCCESS &&
                cuCtxSetCurrent(context) == CUDA_SUCCESS &&
                cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == CUDA_SUCCESS &&
                cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == CUDA_SUCCESS;
        }
    }
    if (success) {
        const double begin = benchmark_seconds_get();
        SecantCUDAResult result;

        stage = "native CUDA compilation";
        result = secant_cuda_philox_dynamic_leaf_select_compile(
            options.num_kernels,
            options.asts_per_kernel,
            options.num_input_columns,
            options.num_static_input_columns,
            options.num_dynamic_leaves,
            options.tile_rows,
            options.threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts,
            (uint32_t)major,
            (uint32_t)minor,
            nvrtc_options,
            3u,
            false,
            compile_scratch,
            BENCH_COMPILE_SCRATCH_BYTES,
            log,
            sizeof(log),
            &log_size,
            &compiled);
        compile_seconds = benchmark_seconds_get() - begin;
        if (result != SECANT_CUDA_SUCCESS) {
            fprintf(stderr, "compile_result=%s log_size=%zu\n", secant_cuda_result_to_string(result), log_size);
            if (log_size > 1u && log[0] != '\0') {
                fprintf(stderr, "%s\n", log);
            }
            success = 0;
        }
    }
    if (success) {
        success = secant_cuda_compiled_binary_get(compiled, &cubin, &cubin_size) == SECANT_CUDA_SUCCESS &&
            cuModuleLoadData(&module, cubin) == CUDA_SUCCESS &&
            cuModuleGetFunction(&reconstruct_function, module, "secant_philox_dynamic_leaf_reconstruct") == CUDA_SUCCESS;
        stage = "module load";
    }
    for (kernel_idx = 0u; success && kernel_idx < options.num_kernels; ++kernel_idx) {
        char name[96];

        if (snprintf(name, sizeof(name), "secant_philox_dynamic_leaf_select_%03zu", kernel_idx) < 0 ||
            cuModuleGetFunction(&functions[kernel_idx], module, name) != CUDA_SUCCESS) {
            success = 0;
            stage = "function lookup";
        }
    }
    for (kernel_idx = 0u; success && kernel_idx < options.num_streams; ++kernel_idx) {
        if (cuStreamCreate(&streams[kernel_idx], CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
            success = 0;
            stage = "stream creation";
        }
    }
    if (success) {
        stage = "device allocation";
        success = cuMemAlloc(&device_input, input_elements * sizeof(float)) == CUDA_SUCCESS &&
            cuMemAlloc(&device_target, options.num_rows * sizeof(float)) == CUDA_SUCCESS &&
            cuMemAlloc(&device_indices, output_elements * sizeof(uint32_t)) == CUDA_SUCCESS &&
            (!options.write_sse || cuMemAlloc(&device_sse, output_elements * sizeof(float)) == CUDA_SUCCESS) &&
            cuMemAlloc(&device_ast_indices, output_elements * sizeof(uint32_t)) == CUDA_SUCCESS &&
            cuMemAlloc(&device_masks, output_elements * sizeof(uint32_t)) == CUDA_SUCCESS &&
            cuMemAlloc(&device_words, word_elements * sizeof(uint32_t)) == CUDA_SUCCESS &&
            cuMemcpyHtoD(device_input, input, input_elements * sizeof(float)) == CUDA_SUCCESS &&
            cuMemcpyHtoD(device_target, target, options.num_rows * sizeof(float)) == CUDA_SUCCESS;
    }
    column_threshold = benchmark_column_threshold(options.column_probability);
    verify_blocks = options.num_setting_blocks < 2u ? options.num_setting_blocks : 2u;
    if (success) {
        size_t verify_outputs = verify_blocks * options.threads_per_block;
        CUdeviceptr output_indices = device_indices;
        CUdeviceptr output_sse = device_sse;
        void* args[] = {
            &device_input,
            &options.num_input_columns,
            &options.num_rows,
            &device_target,
            &options.num_rows,
            &options.asts_per_kernel,
            &options.seed,
            &options.epoch,
            &options.setting_offset,
            &column_threshold,
            &options.constant_radius,
            &options.num_setting_passes,
            &output_indices,
            &output_sse
        };
        void* reconstruct_args[] = {
            &device_indices,
            &verify_outputs,
            &verify_outputs,
            &options.num_input_columns,
            &options.seed,
            &options.epoch,
            &options.setting_offset,
            &column_threshold,
            &options.constant_radius,
            &options.num_setting_passes,
            &device_ast_indices,
            &device_masks,
            &device_words,
            &options.num_dynamic_leaves
        };
        const unsigned int reconstruct_blocks = (unsigned int)((verify_outputs + 255u) / 256u);

        stage = "GPU verification launch";
        success = cuLaunchKernel(
                functions[0],
                (unsigned int)verify_blocks, 1u, 1u,
                (unsigned int)options.threads_per_block, 1u, 1u,
                0u, streams[0], args, NULL) == CUDA_SUCCESS &&
            cuLaunchKernel(
                reconstruct_function,
                reconstruct_blocks, 1u, 1u,
                256u, 1u, 1u,
                0u, streams[0], reconstruct_args, NULL) == CUDA_SUCCESS &&
            cuStreamSynchronize(streams[0]) == CUDA_SUCCESS;
        if (success) {
            verify_indices = (uint32_t*)malloc(verify_outputs * sizeof(*verify_indices));
            verify_sse = options.write_sse ? (float*)malloc(verify_outputs * sizeof(*verify_sse)) : NULL;
            verify_ast_indices = (uint32_t*)malloc(verify_outputs * sizeof(*verify_ast_indices));
            verify_masks = (uint32_t*)malloc(verify_outputs * sizeof(*verify_masks));
            verify_words = (uint32_t*)malloc(verify_outputs * options.num_dynamic_leaves * sizeof(*verify_words));
            success = verify_indices != NULL && (!options.write_sse || verify_sse != NULL) &&
                verify_ast_indices != NULL && verify_masks != NULL && verify_words != NULL;
        }
        if (success) {
            success = cuMemcpyDtoH(verify_indices, device_indices, verify_outputs * sizeof(*verify_indices)) == CUDA_SUCCESS &&
                (!options.write_sse ||
                 cuMemcpyDtoH(verify_sse, device_sse, verify_outputs * sizeof(*verify_sse)) == CUDA_SUCCESS) &&
                cuMemcpyDtoH(verify_ast_indices, device_ast_indices,
                    verify_outputs * sizeof(*verify_ast_indices)) == CUDA_SUCCESS &&
                cuMemcpyDtoH(verify_masks, device_masks, verify_outputs * sizeof(*verify_masks)) == CUDA_SUCCESS &&
                cuMemcpyDtoH(verify_words, device_words,
                    verify_outputs * options.num_dynamic_leaves * sizeof(*verify_words)) == CUDA_SUCCESS;
        }
        if (success) {
            stage = "CPU/GPU verification";
            success = benchmark_cpu_verify(
                &options,
                routines,
                num_routines,
                asts,
                input,
                target,
                verify_indices,
                verify_sse,
                verify_ast_indices,
                verify_masks,
                verify_words,
                verify_blocks);
        }
    }
    for (iteration = 0u; success && iteration < options.warmups + options.iterations; ++iteration) {
        uint64_t iteration_epoch = options.epoch + iteration;
        const double begin = benchmark_seconds_get();

        stage = "timed execution";
        for (kernel_idx = 0u; success && kernel_idx < options.num_kernels; ++kernel_idx) {
            CUdeviceptr output_indices = device_indices + kernel_idx * outputs_per_kernel * sizeof(uint32_t);
            CUdeviceptr output_sse = options.write_sse
                ? device_sse + kernel_idx * outputs_per_kernel * sizeof(float)
                : 0u;
            void* args[] = {
                &device_input,
                &options.num_input_columns,
                &options.num_rows,
                &device_target,
                &options.num_rows,
                &options.asts_per_kernel,
                &options.seed,
                &iteration_epoch,
                &options.setting_offset,
                &column_threshold,
                &options.constant_radius,
                &options.num_setting_passes,
                &output_indices,
                &output_sse
            };

            if (cuLaunchKernel(
                    functions[kernel_idx],
                    (unsigned int)options.num_setting_blocks, 1u, 1u,
                    (unsigned int)options.threads_per_block, 1u, 1u,
                    0u, streams[kernel_idx % options.num_streams], args, NULL) != CUDA_SUCCESS) {
                success = 0;
            }
        }
        for (kernel_idx = 0u; success && kernel_idx < options.num_streams; ++kernel_idx) {
            if (cuStreamSynchronize(streams[kernel_idx]) != CUDA_SUCCESS) {
                success = 0;
            }
        }
        if (success && iteration >= options.warmups) {
            runtime_seconds += benchmark_seconds_get() - begin;
        }
    }
    if (success) {
        const double setting_evals = (double)options.iterations * options.num_kernels *
            options.num_setting_blocks * options.threads_per_block * options.num_setting_passes;
        const double ast_evals = setting_evals * options.asts_per_kernel;
        const double row_evals = ast_evals * options.num_rows;
        const double output_writes = (double)options.iterations * output_elements;

        printf(
            "backend=cuda shape=philox_dynamic_leaf_select ast_mode=%s cse=%s kernels=%zu "
            "asts_per_kernel=%zu columns=%zu static_columns=%zu dynamic_leaves=%zu dynamic_sites=%zu "
            "leaves_used=%zu rows=%zu tile_rows=%zu threads=%zu setting_blocks=%zu passes=%zu "
            "settings_per_kernel=%zu candidates_per_thread=%zu streams=%zu write_sse=%d iterations=%zu "
            "compile_seconds=%.6f cubin_bytes=%zu runtime_seconds=%.6f settings_per_second=%.3e "
            "ast_evals_per_second=%.3e row_evals_per_second=%.3e output_writes_per_second=%.3e verify=pass\n",
            options.ast_mode == SECANT_BENCH_AST_MODE_SIMPLE ? "simple" :
                options.ast_mode == SECANT_BENCH_AST_MODE_MUFU ? "mufu" : "alu",
            options.cse_mode == SECANT_BENCH_CSE_SHARED ? "shared" : "distinct",
            options.num_kernels,
            options.asts_per_kernel,
            options.num_input_columns,
            options.num_static_input_columns,
            options.num_dynamic_leaves,
            options.num_dynamic_sites,
            max_dynamic_leaves_used,
            options.num_rows,
            options.tile_rows,
            options.threads_per_block,
            options.num_setting_blocks,
            options.num_setting_passes,
            options.num_setting_blocks * options.threads_per_block * options.num_setting_passes,
            options.num_setting_passes * options.asts_per_kernel,
            options.num_streams,
            options.write_sse,
            options.iterations,
            compile_seconds,
            cubin_size,
            runtime_seconds,
            setting_evals / runtime_seconds,
            ast_evals / runtime_seconds,
            row_evals / runtime_seconds,
            output_writes / runtime_seconds);
    } else {
        fprintf(stderr, "philox dynamic-leaf selection benchmark failed during %s\n", stage);
    }

    free(verify_words);
    free(verify_masks);
    free(verify_ast_indices);
    free(verify_sse);
    free(verify_indices);
    if (device_words != 0u) (void)cuMemFree(device_words);
    if (device_masks != 0u) (void)cuMemFree(device_masks);
    if (device_ast_indices != 0u) (void)cuMemFree(device_ast_indices);
    if (device_sse != 0u) (void)cuMemFree(device_sse);
    if (device_indices != 0u) (void)cuMemFree(device_indices);
    if (device_target != 0u) (void)cuMemFree(device_target);
    if (device_input != 0u) (void)cuMemFree(device_input);
    if (streams != NULL) {
        for (kernel_idx = 0u; kernel_idx < options.num_streams; ++kernel_idx) {
            if (streams[kernel_idx] != NULL) (void)cuStreamDestroy(streams[kernel_idx]);
        }
    }
    if (module != NULL) (void)cuModuleUnload(module);
    if (compiled != NULL) (void)secant_cuda_compiled_destroy(compiled);
    if (retained) {
        (void)cuCtxSetCurrent(previous_context);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    free(streams);
    free(functions);
    free(target);
    free(input);
    free(compile_scratch);
    free(asts);
    free(programs);
    return success ? 0 : 1;
}
