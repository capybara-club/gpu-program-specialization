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
#ifndef SECANT_BENCH_CUBIN_TEMPLATE_H_INCLUDED
#define SECANT_BENCH_CUBIN_TEMPLATE_H_INCLUDED

#ifndef SECANT_BENCH_CUBIN_TEMPLATE_DYNAMIC_ONLY
#include "bench_common.h"
#endif
#include "secant.h"

#include <nvrtc.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SECANT_BENCH_CUBIN_TEMPLATE_DYNAMIC_ONLY
static int
secant_bench_cubin_template_compile(
    SecantBenchShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    uint32_t target_sm,
    char* error,
    size_t error_size,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char architecture[64];
    const char* options[4];
    char* source = NULL;
    unsigned char* cubin = NULL;
    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_result;
    SecantResult result;
    SecantCubinMaterializeRecipe materialize_recipe = secant_cubin_materialize_recipe_init();
    SecantCubinSSERecipe sse_recipe = secant_cubin_sse_recipe_init();
    const SecantCubinRecipeHeader* recipe;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t log_size = 0u;
    int architecture_bytes;
    int success = 0;

    if (error == NULL || error_size == 0u || cubin_ret == NULL ||
        cubin_size_ret == NULL) {
        return 0;
    }
    error[0] = '\0';
    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    materialize_recipe.num_kernels = num_kernels;
    materialize_recipe.asts_per_kernel = asts_per_kernel;
    materialize_recipe.num_inputs = num_inputs;
    materialize_recipe.patch_capacity_instructions = patch_capacity_instructions;
    sse_recipe.num_kernels = num_kernels;
    sse_recipe.asts_per_kernel = asts_per_kernel;
    sse_recipe.num_inputs = num_inputs;
    sse_recipe.num_targets = num_targets;
    sse_recipe.tile_rows = tile_rows;
    sse_recipe.threads_per_block = threads_per_block;
    sse_recipe.patch_capacity_instructions = patch_capacity_instructions;
    recipe = shape == SECANT_BENCH_SHAPE_MATERIALIZE ? &materialize_recipe.header : &sse_recipe.header;
    result = secant_cubin_source_size(recipe, &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        snprintf(
            error,
            error_size,
            "CUBIN CUDA measurement failed: SecantResult(%d)",
            (int)result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        snprintf(error, error_size, "CUBIN CUDA source allocation failed");
        return 0;
    }
    result = secant_cubin_source_write(recipe, source, source_size);
    if (result != SECANT_SUCCESS) {
        snprintf(
            error,
            error_size,
            "CUBIN CUDA generation failed: SecantResult(%d)",
            (int)result);
        free(source);
        return 0;
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-architecture=sm_%u",
        target_sm);
    if (architecture_bytes < 0 ||
        (size_t)architecture_bytes >= sizeof(architecture)) {
        snprintf(error, error_size, "invalid CUBIN target SM");
        free(source);
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    nvrtc_result = nvrtcCreateProgram(
        &program,
        source,
        "secant_cubin_template.cu",
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
            char* log = (char*)malloc(log_size);

            if (log != NULL &&
                nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                snprintf(error, error_size, "%s", log);
            } else {
                snprintf(
                    error,
                    error_size,
                    "%s",
                    nvrtcGetErrorString(nvrtc_result));
            }
            free(log);
        } else {
            snprintf(
                error,
                error_size,
                "%s",
                nvrtcGetErrorString(nvrtc_result));
        }
    } else if (nvrtcGetCUBINSize(program, &cubin_size) != NVRTC_SUCCESS ||
        cubin_size == 0u) {
        snprintf(error, error_size, "NVRTC returned an empty CUBIN");
    } else {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin == NULL) {
            snprintf(error, error_size, "CUBIN allocation failed");
        } else if (nvrtcGetCUBIN(program, (char*)cubin) != NVRTC_SUCCESS) {
            snprintf(error, error_size, "NVRTC CUBIN retrieval failed");
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
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}
#endif

static inline int
secant_bench_cubin_dynamic_constant_sse_template_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    uint32_t target_sm,
    char* error,
    size_t error_size,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    SecantCubinDynamicConstantSSERecipe recipe = secant_cubin_dynamic_constant_sse_recipe_init();
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

    if (error == NULL || error_size == 0u || cubin_ret == NULL ||
        cubin_size_ret == NULL) {
        return 0;
    }
    error[0] = '\0';
    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_input_columns = num_input_columns;
    recipe.num_input_constants = num_input_constants;
    recipe.num_targets = num_targets;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    result = secant_cubin_source_size(&recipe.header, &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        snprintf(
            error,
            error_size,
            "CUBIN CUDA measurement failed: SecantResult(%d)",
            (int)result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        snprintf(error, error_size, "CUBIN CUDA source allocation failed");
        return 0;
    }
    result = secant_cubin_source_write(&recipe.header, source, source_size);
    if (result != SECANT_SUCCESS) {
        snprintf(
            error,
            error_size,
            "CUBIN CUDA generation failed: SecantResult(%d)",
            (int)result);
        free(source);
        return 0;
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-architecture=sm_%u",
        target_sm);
    if (architecture_bytes < 0 ||
        (size_t)architecture_bytes >= sizeof(architecture)) {
        snprintf(error, error_size, "invalid CUBIN target SM");
        free(source);
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    nvrtc_result = nvrtcCreateProgram(
        &program,
        source,
        "secant_cubin_dynamic_constant_sse_template.cu",
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
            char* log = (char*)malloc(log_size);

            if (log != NULL &&
                nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                snprintf(error, error_size, "%s", log);
            } else {
                snprintf(
                    error,
                    error_size,
                    "%s",
                    nvrtcGetErrorString(nvrtc_result));
            }
            free(log);
        } else {
            snprintf(
                error,
                error_size,
                "%s",
                nvrtcGetErrorString(nvrtc_result));
        }
    } else if (nvrtcGetCUBINSize(program, &cubin_size) != NVRTC_SUCCESS ||
        cubin_size == 0u) {
        snprintf(error, error_size, "NVRTC returned an empty CUBIN");
    } else {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin == NULL) {
            snprintf(error, error_size, "CUBIN allocation failed");
        } else if (nvrtcGetCUBIN(program, (char*)cubin) != NVRTC_SUCCESS) {
            snprintf(error, error_size, "NVRTC CUBIN retrieval failed");
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
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

#endif /* SECANT_BENCH_CUBIN_TEMPLATE_H_INCLUDED */
