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

#include "backends/cuda/secant_cuda.h"
#include "backends/ptx/secant_ptx.h"
#include "secant.h"

#include <cuda.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_COMPILE_SCRATCH_SIZE (64u * 1024u * 1024u)
#define BENCH_LOG_SIZE (256u * 1024u)

typedef enum BenchBackend {
    BENCH_BACKEND_CUDA = 0,
    BENCH_BACKEND_PTX = 1
} BenchBackend;

typedef struct BenchOptions {
    BenchBackend backend;
    int warp_owned;
    int mixed_bindings;
    int square_cube;
    size_t num_rows;
    size_t num_settings;
    size_t settings_per_cta;
    size_t num_parameters;
    size_t num_columns;
    size_t tile_rows;
    size_t threads;
    size_t verify_settings;
    size_t warmups;
    size_t iterations;
    int device_ordinal;
} BenchOptions;

static void
bench_usage(const char* program) {
    fprintf(stderr,
        "usage: %s [--backend cuda|ptx] [--setting-owner thread|warp] "
        "[--ast-mode linear|square-cube] "
        "[--bindings constants|mixed] [--rows N] [--settings N] "
        "[--settings-per-cta N] [--parameters 1..8] [--columns N] "
        "[--tile-rows N] [--threads N] [--verify-settings N] "
        "[--warmups N] [--iterations N] "
        "[--device N]\n",
        program);
}

static int
bench_size_parse(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
bench_options_parse(int argc, char** argv, BenchOptions* options) {
    int arg_idx;

    options->backend = BENCH_BACKEND_CUDA;
    options->warp_owned = 0;
    options->mixed_bindings = 0;
    options->square_cube = 0;
    options->num_rows = 1000000u;
    options->num_settings = 8u;
    options->settings_per_cta = 0u;
    options->num_parameters = 4u;
    options->num_columns = 8u;
    options->tile_rows = 128u;
    options->threads = 128u;
    options->verify_settings = 0u;
    options->warmups = 2u;
    options->iterations = 5u;
    options->device_ordinal = 0;

    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        const char* name = argv[arg_idx];
        const char* value;
        size_t parsed;

        if (arg_idx + 1 >= argc) return 0;
        value = argv[++arg_idx];
        if (strcmp(name, "--backend") == 0) {
            if (strcmp(value, "cuda") == 0) options->backend = BENCH_BACKEND_CUDA;
            else if (strcmp(value, "ptx") == 0) options->backend = BENCH_BACKEND_PTX;
            else return 0;
        } else if (strcmp(name, "--setting-owner") == 0) {
            if (strcmp(value, "thread") == 0) options->warp_owned = 0;
            else if (strcmp(value, "warp") == 0) options->warp_owned = 1;
            else return 0;
        } else if (strcmp(name, "--bindings") == 0) {
            if (strcmp(value, "constants") == 0) options->mixed_bindings = 0;
            else if (strcmp(value, "mixed") == 0) options->mixed_bindings = 1;
            else return 0;
        } else if (strcmp(name, "--ast-mode") == 0) {
            if (strcmp(value, "linear") == 0) options->square_cube = 0;
            else if (strcmp(value, "square-cube") == 0) options->square_cube = 1;
            else return 0;
        } else if (strcmp(name, "--device") == 0) {
            char* end = NULL;
            long ordinal;
            errno = 0;
            ordinal = strtol(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || ordinal < 0 || ordinal > INT32_MAX) {
                return 0;
            }
            options->device_ordinal = (int)ordinal;
        } else {
            if (!bench_size_parse(value, &parsed)) return 0;
            if (strcmp(name, "--rows") == 0) options->num_rows = parsed;
            else if (strcmp(name, "--settings") == 0) options->num_settings = parsed;
            else if (strcmp(name, "--settings-per-cta") == 0) options->settings_per_cta = parsed;
            else if (strcmp(name, "--parameters") == 0) options->num_parameters = parsed;
            else if (strcmp(name, "--columns") == 0) options->num_columns = parsed;
            else if (strcmp(name, "--tile-rows") == 0) options->tile_rows = parsed;
            else if (strcmp(name, "--threads") == 0) options->threads = parsed;
            else if (strcmp(name, "--verify-settings") == 0) options->verify_settings = parsed;
            else if (strcmp(name, "--warmups") == 0) options->warmups = parsed;
            else if (strcmp(name, "--iterations") == 0) options->iterations = parsed;
            else return 0;
        }
    }
    if (options->settings_per_cta == 0u) {
        options->settings_per_cta = options->num_settings;
    }
    if (options->settings_per_cta > options->num_settings) {
        options->settings_per_cta = options->num_settings;
    }
    if (options->verify_settings == 0u || options->verify_settings > options->num_settings) {
        options->verify_settings = options->num_settings;
    }
    return options->num_rows != 0u && options->num_settings != 0u &&
        options->settings_per_cta != 0u && options->tile_rows != 0u &&
        options->threads != 0u && options->threads <= 1024u &&
        (!options->warp_owned || options->threads % 32u == 0u) &&
        options->num_parameters != 0u &&
        options->num_parameters <= SECANT_CUDA_DYNAMIC_LEAF_LM_MAX_PARAMETERS &&
        options->num_columns != 0u && options->num_columns <= SECANT_AST_MAX_INPUTS &&
        options->num_columns + options->num_parameters <= SECANT_AST_MAX_INPUTS &&
        options->iterations != 0u;
}

static uint32_t
bench_f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
bench_bits_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void
bench_ast_z_emit(size_t parameter_idx, size_t num_columns, uint8_t* ast, size_t* offset) {
    ast[(*offset)++] = SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32;
    ast[(*offset)++] = (uint8_t)(parameter_idx % num_columns);
    ast[(*offset)++] = SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
    ast[(*offset)++] = (uint8_t)parameter_idx;
    ast[(*offset)++] = SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
}

static size_t
bench_ast_build(
    size_t num_parameters,
    size_t num_columns,
    int square_cube,
    uint8_t* ast
) {
    size_t offset = 0u;
    size_t parameter_idx;

    for (parameter_idx = 0u; parameter_idx < num_parameters; ++parameter_idx) {
        bench_ast_z_emit(parameter_idx, num_columns, ast, &offset);
        if (square_cube) {
            /* z + z^2 */
            bench_ast_z_emit(parameter_idx, num_columns, ast, &offset);
            bench_ast_z_emit(parameter_idx, num_columns, ast, &offset);
            ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
            ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
            /* + z^3 */
            bench_ast_z_emit(parameter_idx, num_columns, ast, &offset);
            bench_ast_z_emit(parameter_idx, num_columns, ast, &offset);
            ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
            bench_ast_z_emit(parameter_idx, num_columns, ast, &offset);
            ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
            ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
        }
        if (parameter_idx != 0u) {
            ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
        }
    }
    ast[offset++] = SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
    return offset;
}

static int
bench_cuda_ok(CUresult result, const char* operation) {
    const char* name = NULL;
    const char* text = NULL;
    if (result == CUDA_SUCCESS) return 1;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &text);
    fprintf(stderr, "%s failed: %s (%s)\n", operation,
        name != NULL ? name : "CUDA_ERROR", text != NULL ? text : "unknown");
    return 0;
}

static int
bench_compile(
    const BenchOptions* options,
    uint32_t major,
    uint32_t minor,
    const SecantAstInstruction* const* asts,
    void* scratch,
    char* log,
    const void** binary_ret,
    size_t* binary_size_ret,
    SecantCUDACompiled* cuda_compiled_ret,
    SecantPTXHandle* ptx_handle_ret,
    SecantPTXCompiled* ptx_compiled_ret
) {
    size_t log_size = 0u;

    *cuda_compiled_ret = NULL;
    *ptx_handle_ret = NULL;
    *ptx_compiled_ret = NULL;
    if (options->backend == BENCH_BACKEND_CUDA) {
        SecantCUDAResult result = options->warp_owned
            ? secant_cuda_warp_dynamic_leaf_lm_compile(
                1u, 1u, options->num_columns, options->num_columns,
                options->num_parameters, 1u, options->tile_rows, options->threads,
                NULL, 0u, asts, major, minor, NULL, 0u, true,
                scratch, BENCH_COMPILE_SCRATCH_SIZE,
                log, BENCH_LOG_SIZE, &log_size, cuda_compiled_ret)
            : secant_cuda_dynamic_leaf_lm_compile(
                1u, 1u, options->num_columns, options->num_columns,
                options->num_parameters, 1u, options->tile_rows, options->threads,
                NULL, 0u, asts, major, minor, NULL, 0u, true,
                scratch, BENCH_COMPILE_SCRATCH_SIZE,
                log, BENCH_LOG_SIZE, &log_size, cuda_compiled_ret);
        if (result != SECANT_CUDA_SUCCESS) {
            fprintf(stderr, "CUDA LM compile failed: %s\n%s\n",
                secant_cuda_result_to_string(result), log_size != 0u ? log : "");
            return 0;
        }
        result = secant_cuda_compiled_binary_get(
            *cuda_compiled_ret, binary_ret, binary_size_ret);
        return result == SECANT_CUDA_SUCCESS;
    } else {
        SecantPTXResult result = options->warp_owned
            ? secant_ptx_warp_dynamic_leaf_lm_create(
                1u, 1u, options->num_columns, options->num_columns,
                options->num_parameters, 1u, options->tile_rows, options->threads,
                major, minor, NULL, 0u, true,
                log, BENCH_LOG_SIZE, &log_size, ptx_handle_ret)
            : secant_ptx_dynamic_leaf_lm_create(
                1u, 1u, options->num_columns, options->num_columns,
                options->num_parameters, 1u, options->tile_rows, options->threads,
                major, minor, NULL, 0u, true,
                log, BENCH_LOG_SIZE, &log_size, ptx_handle_ret);
        if (result != SECANT_PTX_SUCCESS) {
            fprintf(stderr, "PTX LM template compile failed: %s\n%s\n",
                secant_ptx_result_to_string(result), log_size != 0u ? log : "");
            return 0;
        }
        memset(scratch, 0, BENCH_COMPILE_SCRATCH_SIZE);
        result = secant_ptx_compile(
            *ptx_handle_ret, NULL, 0u, asts,
            major, minor, NULL, 0u, true,
            scratch, BENCH_COMPILE_SCRATCH_SIZE,
            log, BENCH_LOG_SIZE, &log_size, ptx_compiled_ret);
        if (result != SECANT_PTX_SUCCESS) {
            fprintf(stderr, "PTX LM AST compile failed: %s\n%s\n",
                secant_ptx_result_to_string(result), log_size != 0u ? log : "");
            return 0;
        }
        result = secant_ptx_compiled_binary_get(
            *ptx_compiled_ret, binary_ret, binary_size_ret);
        return result == SECANT_PTX_SUCCESS;
    }
}

static void
bench_cpu_statistics(
    const BenchOptions* options,
    size_t num_statistics,
    const float* input,
    const uint32_t* masks,
    const uint32_t* words,
    const float* targets,
    double* statistics,
    size_t num_verify_settings
) {
    size_t verify_idx;
    size_t row;

    memset(statistics, 0, num_statistics * options->num_settings * sizeof(*statistics));
    for (verify_idx = 0u; verify_idx < num_verify_settings; ++verify_idx) {
        const size_t setting = num_verify_settings == options->num_settings
            ? verify_idx
            : num_verify_settings == 1u
                ? options->num_settings - 1u
                : verify_idx * (options->num_settings - 1u) / (num_verify_settings - 1u);
        for (row = 0u; row < options->num_rows; ++row) {
            double prediction = 0.0;
            double gradient[SECANT_CUDA_DYNAMIC_LEAF_LM_MAX_PARAMETERS];
            size_t parameter;
            size_t stat = 1u;
            size_t lhs;
            size_t rhs;

            for (parameter = 0u; parameter < options->num_parameters; ++parameter) {
                const double x = input[(parameter % options->num_columns) * options->num_rows + row];
                const int is_column = (int)((masks[setting] >> parameter) & 1u);
                const uint32_t word = words[setting * options->num_parameters + parameter];
                const double leaf = is_column
                    ? input[(word % options->num_columns) * options->num_rows + row]
                    : bench_bits_f32(word);
                const double z = x * leaf;
                if (options->square_cube) {
                    prediction += z + z * z + z * z * z;
                    gradient[parameter] = is_column
                        ? 0.0
                        : x * (1.0 + 2.0 * z + 3.0 * z * z);
                } else {
                    prediction += z;
                    gradient[parameter] = is_column ? 0.0 : x;
                }
            }
            {
                const double residual = prediction - targets[row];
                statistics[setting] += residual * residual;
                for (parameter = 0u; parameter < options->num_parameters; ++parameter, ++stat) {
                    statistics[stat * options->num_settings + setting] +=
                        gradient[parameter] * residual;
                }
                for (lhs = 0u; lhs < options->num_parameters; ++lhs) {
                    for (rhs = lhs; rhs < options->num_parameters; ++rhs, ++stat) {
                        statistics[stat * options->num_settings + setting] +=
                            gradient[lhs] * gradient[rhs];
                    }
                }
            }
        }
    }
}

int
main(int argc, char** argv) {
    BenchOptions options;
    uint8_t ast_storage[512];
    const SecantAstInstruction* asts[1];
    size_t ast_size;
    size_t num_statistics;
    size_t input_count;
    size_t word_count;
    size_t output_count;
    float* input = NULL;
    float* targets = NULL;
    uint32_t* masks = NULL;
    uint32_t* words = NULL;
    float* output = NULL;
    double* expected = NULL;
    void* scratch = NULL;
    char* log = NULL;
    CUdevice device;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr d_input = 0u;
    CUdeviceptr d_targets = 0u;
    CUdeviceptr d_masks = 0u;
    CUdeviceptr d_words = 0u;
    CUdeviceptr d_output = 0u;
    CUevent start = NULL;
    CUevent end = NULL;
    int major;
    int minor;
    int function_registers = 0;
    int function_local_bytes = 0;
    const void* binary = NULL;
    size_t binary_size = 0u;
    SecantCUDACompiled cuda_compiled = NULL;
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXCompiled ptx_compiled = NULL;
    size_t setting;
    size_t column;
    size_t row;
    size_t warmup;
    size_t iteration;
    size_t grid_x;
    size_t grid_y;
    size_t shared_floats;
    size_t shared_bytes;
    size_t num_input_columns;
    size_t input_ld;
    size_t words_ld;
    unsigned int num_settings;
    unsigned int settings_per_cta;
    size_t targets_ld;
    size_t num_rows;
    size_t num_asts = 1u;
    size_t num_targets = 1u;
    size_t output_ld;
    void* args[15];
    float elapsed_ms = 0.0f;
    double max_relative_error = 0.0;
    int checked = 0;
    int status = 1;

    if (!bench_options_parse(argc, argv, &options)) {
        bench_usage(argv[0]);
        return 2;
    }
    if (secant_cuda_dynamic_leaf_lm_statistics_count(
            options.num_parameters, &num_statistics) != SECANT_CUDA_SUCCESS ||
        options.num_columns > SIZE_MAX / options.num_rows ||
        options.num_settings > SIZE_MAX / options.num_parameters ||
        options.num_settings > UINT32_MAX ||
        num_statistics > SIZE_MAX / options.num_settings) {
        fprintf(stderr, "benchmark geometry overflow\n");
        return 2;
    }
    input_count = options.num_columns * options.num_rows;
    word_count = options.num_settings * options.num_parameters;
    output_count = num_statistics * options.num_settings;
    input = (float*)malloc(input_count * sizeof(*input));
    targets = (float*)malloc(options.num_rows * sizeof(*targets));
    masks = (uint32_t*)malloc(options.num_settings * sizeof(*masks));
    words = (uint32_t*)malloc(word_count * sizeof(*words));
    output = (float*)malloc(output_count * sizeof(*output));
    scratch = malloc(BENCH_COMPILE_SCRATCH_SIZE);
    log = (char*)malloc(BENCH_LOG_SIZE);
    if (input == NULL || targets == NULL || masks == NULL || words == NULL ||
        output == NULL || scratch == NULL || log == NULL) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    for (column = 0u; column < options.num_columns; ++column) {
        for (row = 0u; row < options.num_rows; ++row) {
            input[column * options.num_rows + row] =
                (float)((double)(column + 1u) * 0.01 +
                    ((double)(row % 251u) - 125.0) * 0.0005);
        }
    }
    for (row = 0u; row < options.num_rows; ++row) {
        targets[row] = 0.2f * input[row] - 0.03f;
    }
    for (setting = 0u; setting < options.num_settings; ++setting) {
        masks[setting] = 0u;
        for (column = 0u; column < options.num_parameters; ++column) {
            const int bind_column = options.mixed_bindings && ((setting + column) % 3u == 0u);
            if (bind_column) {
                masks[setting] |= 1u << column;
                words[setting * options.num_parameters + column] =
                    (uint32_t)((column * 3u + setting) % options.num_columns);
            } else {
                words[setting * options.num_parameters + column] = bench_f32_bits(
                    (float)(0.05 * (double)(column + 1u) + 0.001 * (double)setting));
            }
        }
    }
    ast_size = bench_ast_build(
        options.num_parameters, options.num_columns, options.square_cube, ast_storage);
    (void)ast_size;
    asts[0] = ast_storage;

    if (!bench_cuda_ok(cuInit(0u), "cuInit") ||
        !bench_cuda_ok(cuDeviceGet(&device, options.device_ordinal), "cuDeviceGet") ||
        !bench_cuda_ok(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain") ||
        !bench_cuda_ok(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
        !bench_cuda_ok(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "compute capability major") ||
        !bench_cuda_ok(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "compute capability minor")) {
        goto cleanup;
    }
    if (!bench_compile(
            &options, (uint32_t)major, (uint32_t)minor, asts,
            scratch, log, &binary, &binary_size,
            &cuda_compiled, &ptx_handle, &ptx_compiled)) {
        goto cleanup;
    }
    (void)binary_size;
    if (!bench_cuda_ok(cuModuleLoadData(&module, binary), "cuModuleLoadData") ||
        !bench_cuda_ok(cuModuleGetFunction(&function, module, "secant_dynamic_leaf_lm_000"), "cuModuleGetFunction") ||
        !bench_cuda_ok(cuFuncGetAttribute(
            &function_registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function), "query registers") ||
        !bench_cuda_ok(cuFuncGetAttribute(
            &function_local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function), "query local bytes") ||
        !bench_cuda_ok(cuMemAlloc(&d_input, input_count * sizeof(*input)), "cuMemAlloc input") ||
        !bench_cuda_ok(cuMemAlloc(&d_targets, options.num_rows * sizeof(*targets)), "cuMemAlloc targets") ||
        !bench_cuda_ok(cuMemAlloc(&d_masks, options.num_settings * sizeof(*masks)), "cuMemAlloc masks") ||
        !bench_cuda_ok(cuMemAlloc(&d_words, word_count * sizeof(*words)), "cuMemAlloc words") ||
        !bench_cuda_ok(cuMemAlloc(&d_output, output_count * sizeof(*output)), "cuMemAlloc output") ||
        !bench_cuda_ok(cuMemcpyHtoD(d_input, input, input_count * sizeof(*input)), "copy input") ||
        !bench_cuda_ok(cuMemcpyHtoD(d_targets, targets, options.num_rows * sizeof(*targets)), "copy targets") ||
        !bench_cuda_ok(cuMemcpyHtoD(d_masks, masks, options.num_settings * sizeof(*masks)), "copy masks") ||
        !bench_cuda_ok(cuMemcpyHtoD(d_words, words, word_count * sizeof(*words)), "copy words")) {
        goto cleanup;
    }
    grid_x = (options.num_rows + options.tile_rows - 1u) / options.tile_rows;
    grid_y = options.num_settings / options.settings_per_cta +
        (options.num_settings % options.settings_per_cta != 0u ? 1u : 0u);
    shared_floats = (options.warp_owned ? options.num_columns : (options.num_columns | 1u)) + 1u;
    if (shared_floats > SIZE_MAX / options.tile_rows ||
        shared_floats * options.tile_rows > SIZE_MAX / sizeof(float) ||
        grid_x > UINT32_MAX || grid_y > UINT32_MAX || options.threads > UINT32_MAX) {
        fprintf(stderr, "launch geometry overflow\n");
        goto cleanup;
    }
    shared_bytes = shared_floats * options.tile_rows * sizeof(float);
    num_input_columns = options.num_columns;
    input_ld = options.num_rows;
    words_ld = options.num_parameters;
    num_settings = (unsigned int)options.num_settings;
    settings_per_cta = (unsigned int)options.settings_per_cta;
    targets_ld = options.num_rows;
    num_rows = options.num_rows;
    output_ld = options.num_settings;
    args[0] = &d_input;
    args[1] = &num_input_columns;
    args[2] = &input_ld;
    args[3] = &d_masks;
    args[4] = &d_words;
    args[5] = &words_ld;
    args[6] = &num_settings;
    args[7] = &settings_per_cta;
    args[8] = &d_targets;
    args[9] = &targets_ld;
    args[10] = &num_rows;
    args[11] = &num_asts;
    args[12] = &num_targets;
    args[13] = &d_output;
    args[14] = &output_ld;

    if (!bench_cuda_ok(cuMemsetD8(d_output, 0, output_count * sizeof(*output)), "clear output") ||
        !bench_cuda_ok(cuLaunchKernel(
            function, (unsigned)grid_x, (unsigned)grid_y, 1u,
            (unsigned)options.threads, 1u, 1u,
            (unsigned)shared_bytes, 0, args, NULL), "correctness launch") ||
        !bench_cuda_ok(cuCtxSynchronize(), "correctness synchronize") ||
        !bench_cuda_ok(cuMemcpyDtoH(output, d_output, output_count * sizeof(*output)), "copy output")) {
        goto cleanup;
    }
    if (options.num_rows <= 65536u) {
        expected = (double*)malloc(output_count * sizeof(*expected));
        if (expected == NULL) goto cleanup;
        bench_cpu_statistics(
            &options, num_statistics, input, masks, words, targets, expected,
            options.verify_settings);
        for (setting = 0u; setting < options.verify_settings; ++setting) {
            const size_t checked_setting = options.verify_settings == options.num_settings
                ? setting
                : options.verify_settings == 1u
                    ? options.num_settings - 1u
                    : setting * (options.num_settings - 1u) / (options.verify_settings - 1u);
            size_t statistic;

            for (statistic = 0u; statistic < num_statistics; ++statistic) {
                const size_t output_idx = statistic * options.num_settings + checked_setting;
                const double difference = fabs((double)output[output_idx] - expected[output_idx]);
                const double scale = fmax(1.0, fabs(expected[output_idx]));
                const double relative = difference / scale;
                if (relative > max_relative_error) max_relative_error = relative;
            }
        }
        checked = 1;
        if (!(max_relative_error < 2.0e-3)) {
            fprintf(stderr, "LM statistics mismatch: max relative error %.9g\n", max_relative_error);
            goto cleanup;
        }
    }
    for (warmup = 0u; warmup < options.warmups; ++warmup) {
        if (!bench_cuda_ok(cuLaunchKernel(
                function, (unsigned)grid_x, (unsigned)grid_y, 1u,
                (unsigned)options.threads, 1u, 1u,
                (unsigned)shared_bytes, 0, args, NULL), "warmup launch")) goto cleanup;
    }
    if (!bench_cuda_ok(cuCtxSynchronize(), "warmup synchronize") ||
        !bench_cuda_ok(cuEventCreate(&start, CU_EVENT_DEFAULT), "create start event") ||
        !bench_cuda_ok(cuEventCreate(&end, CU_EVENT_DEFAULT), "create end event") ||
        !bench_cuda_ok(cuEventRecord(start, 0), "record start")) goto cleanup;
    for (iteration = 0u; iteration < options.iterations; ++iteration) {
        if (!bench_cuda_ok(cuLaunchKernel(
                function, (unsigned)grid_x, (unsigned)grid_y, 1u,
                (unsigned)options.threads, 1u, 1u,
                (unsigned)shared_bytes, 0, args, NULL), "timed launch")) goto cleanup;
    }
    if (!bench_cuda_ok(cuEventRecord(end, 0), "record end") ||
        !bench_cuda_ok(cuEventSynchronize(end), "event synchronize") ||
        !bench_cuda_ok(cuEventElapsedTime(&elapsed_ms, start, end), "elapsed time")) goto cleanup;
    elapsed_ms /= (float)options.iterations;
    printf(
        "backend=%s shape=dynamic_leaf_lm setting_owner=%s ast_mode=%s bindings=%s "
        "rows=%zu settings=%zu settings_per_cta=%zu setting_ctas=%zu parameters=%zu statistics=%zu "
        "tile_rows=%zu threads=%zu "
        "registers=%d local_bytes=%d "
        "milliseconds=%.6f grow_evals_per_s=%.6f checked=%s checked_settings=%zu "
        "max_relative_error=%.9g\n",
        options.backend == BENCH_BACKEND_CUDA ? "cuda" : "ptx",
        options.warp_owned ? "warp" : "thread",
        options.square_cube ? "square-cube" : "linear",
        options.mixed_bindings ? "mixed" : "constants",
        options.num_rows, options.num_settings, options.settings_per_cta, grid_y,
        options.num_parameters,
        num_statistics, options.tile_rows, options.threads,
        function_registers, function_local_bytes,
        elapsed_ms,
        (double)options.num_rows * (double)options.num_settings /
            ((double)elapsed_ms * 1.0e6),
        checked ? "yes" : "no", checked ? options.verify_settings : 0u,
        max_relative_error);
    status = 0;

cleanup:
    if (end != NULL) (void)cuEventDestroy(end);
    if (start != NULL) (void)cuEventDestroy(start);
    if (d_output != 0u) (void)cuMemFree(d_output);
    if (d_words != 0u) (void)cuMemFree(d_words);
    if (d_masks != 0u) (void)cuMemFree(d_masks);
    if (d_targets != 0u) (void)cuMemFree(d_targets);
    if (d_input != 0u) (void)cuMemFree(d_input);
    if (module != NULL) (void)cuModuleUnload(module);
    if (ptx_compiled != NULL) (void)secant_ptx_compiled_destroy(ptx_compiled);
    if (ptx_handle != NULL) (void)secant_ptx_handle_destroy(ptx_handle);
    if (cuda_compiled != NULL) (void)secant_cuda_compiled_destroy(cuda_compiled);
    if (context != NULL) (void)cuDevicePrimaryCtxRelease(device);
    free(expected);
    free(log);
    free(scratch);
    free(output);
    free(words);
    free(masks);
    free(targets);
    free(input);
    return status;
}
