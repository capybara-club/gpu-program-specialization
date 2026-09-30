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
#ifndef SECANT_BENCH_HSACO_TEMPLATE_H_INCLUDED
#define SECANT_BENCH_HSACO_TEMPLATE_H_INCLUDED

#ifndef SECANT_BENCH_HSACO_TEMPLATE_DYNAMIC_ONLY
#include "bench_common.h"
#endif
#include "secant_hsaco.h"

#include <hip/hiprtc.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SECANT_BENCH_HSACO_TEMPLATE_DYNAMIC_ONLY
static int
secant_bench_hsaco_template_compile(
    SecantBenchShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    size_t patch_capacity_instructions,
    uint32_t opt_level,
    uint32_t gfx_arch,
    char* error,
    size_t error_size,
    unsigned char** hsaco_ret,
    size_t* hsaco_size_ret
) {
    char architecture[64];
    char optimization[16];
    const char* options[3];
    char* source = NULL;
    unsigned char* hsaco = NULL;
    hiprtcProgram program = NULL;
    hiprtcResult hiprtc_result;
    SecantResult result;
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
        tile_rows,
        threads_per_block,
        patch_capacity_instructions,
        reduction_mode
    };
    size_t source_size = 0u;
    size_t written_size = 0u;
    size_t hsaco_size = 0u;
    size_t log_size = 0u;
    int architecture_bytes;
    int optimization_bytes;
    int success = 0;

    if (error == NULL || error_size == 0u || hsaco_ret == NULL ||
        hsaco_size_ret == NULL) {
        return 0;
    }
    error[0] = '\0';
    *hsaco_ret = NULL;
    *hsaco_size_ret = 0u;
    result = shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? secant_hsaco_materialize_source_generate(
            &materialize_recipe,
            NULL,
            0u,
            &source_size)
        : secant_hsaco_sse_source_generate(
            &sse_recipe,
            NULL,
            0u,
            &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        snprintf(
            error,
            error_size,
            "HSACO HIP measurement failed: SecantResult(%d)",
            (int)result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        snprintf(error, error_size, "HSACO HIP source allocation failed");
        return 0;
    }
    result = shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? secant_hsaco_materialize_source_generate(
            &materialize_recipe,
            source,
            source_size,
            &written_size)
        : secant_hsaco_sse_source_generate(
            &sse_recipe,
            source,
            source_size,
            &written_size);
    if (result != SECANT_SUCCESS || written_size != source_size) {
        snprintf(
            error,
            error_size,
            "HSACO HIP generation failed: SecantResult(%d)",
            (int)result);
        free(source);
        return 0;
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--offload-arch=gfx%u",
        gfx_arch);
    if (architecture_bytes < 0 ||
        (size_t)architecture_bytes >= sizeof(architecture)) {
        snprintf(error, error_size, "invalid HSACO target architecture");
        free(source);
        return 0;
    }
    optimization_bytes = snprintf(
        optimization,
        sizeof(optimization),
        "-O%u",
        opt_level);
    if (optimization_bytes < 0 ||
        (size_t)optimization_bytes >= sizeof(optimization)) {
        snprintf(error, error_size, "invalid HSACO optimization level");
        free(source);
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = optimization;
    hiprtc_result = hiprtcCreateProgram(
        &program,
        source,
        "secant_hsaco_template.cu",
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
            hiprtcGetProgramLogSize(program, &log_size) == HIPRTC_SUCCESS &&
            log_size != 0u) {
            char* log = (char*)malloc(log_size);

            if (log != NULL &&
                hiprtcGetProgramLog(program, log) == HIPRTC_SUCCESS) {
                snprintf(error, error_size, "%s", log);
            } else {
                snprintf(
                    error,
                    error_size,
                    "%s",
                    hiprtcGetErrorString(hiprtc_result));
            }
            free(log);
        } else {
            snprintf(
                error,
                error_size,
                "%s",
                hiprtcGetErrorString(hiprtc_result));
        }
    } else if (hiprtcGetCodeSize(program, &hsaco_size) != HIPRTC_SUCCESS ||
        hsaco_size == 0u) {
        snprintf(error, error_size, "HIPRTC returned an empty HSACO");
    } else {
        hsaco = (unsigned char*)malloc(hsaco_size);
        if (hsaco == NULL) {
            snprintf(error, error_size, "HSACO allocation failed");
        } else if (hiprtcGetCode(program, (char*)hsaco) != HIPRTC_SUCCESS) {
            snprintf(error, error_size, "HIPRTC HSACO retrieval failed");
        } else {
            success = 1;
        }
    }
    if (program != NULL) {
        (void)hiprtcDestroyProgram(&program);
    }
    free(source);
    if (!success) {
        free(hsaco);
        return 0;
    }
    *hsaco_ret = hsaco;
    *hsaco_size_ret = hsaco_size;
    return 1;
}
#endif

static inline int
secant_bench_hsaco_dynamic_constant_sse_template_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    size_t patch_capacity_instructions,
    uint32_t opt_level,
    uint32_t gfx_arch,
    char* error,
    size_t error_size,
    unsigned char** hsaco_ret,
    size_t* hsaco_size_ret
) {
    const SecantHsacoDynamicConstantSSERecipe recipe = {
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_input_constants,
        num_targets,
        tile_rows,
        threads_per_block,
        patch_capacity_instructions,
        reduction_mode
    };
    char architecture[64];
    char optimization[16];
    const char* options[3];
    char* source = NULL;
    unsigned char* hsaco = NULL;
    hiprtcProgram program = NULL;
    hiprtcResult hiprtc_result;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    size_t hsaco_size = 0u;
    size_t log_size = 0u;
    int architecture_bytes;
    int optimization_bytes;
    int success = 0;

    if (error == NULL || error_size == 0u || hsaco_ret == NULL ||
        hsaco_size_ret == NULL) {
        return 0;
    }
    error[0] = '\0';
    *hsaco_ret = NULL;
    *hsaco_size_ret = 0u;
    result = secant_hsaco_dynamic_constant_sse_source_generate(
        &recipe,
        NULL,
        0u,
        &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        snprintf(
            error,
            error_size,
            "HSACO HIP measurement failed: SecantResult(%d)",
            (int)result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        snprintf(error, error_size, "HSACO HIP source allocation failed");
        return 0;
    }
    result = secant_hsaco_dynamic_constant_sse_source_generate(
        &recipe,
        source,
        source_size,
        &written_size);
    if (result != SECANT_SUCCESS || written_size != source_size) {
        snprintf(
            error,
            error_size,
            "HSACO HIP generation failed: SecantResult(%d)",
            (int)result);
        free(source);
        return 0;
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--offload-arch=gfx%u",
        gfx_arch);
    optimization_bytes = snprintf(
        optimization,
        sizeof(optimization),
        "-O%u",
        opt_level);
    if (architecture_bytes < 0 ||
        (size_t)architecture_bytes >= sizeof(architecture) ||
        optimization_bytes < 0 ||
        (size_t)optimization_bytes >= sizeof(optimization)) {
        snprintf(error, error_size, "invalid HSACO compiler option");
        free(source);
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = optimization;
    hiprtc_result = hiprtcCreateProgram(
        &program,
        source,
        "secant_hsaco_dynamic_constant_sse_template.cpp",
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
            hiprtcGetProgramLogSize(program, &log_size) == HIPRTC_SUCCESS &&
            log_size != 0u) {
            char* log = (char*)malloc(log_size);

            if (log != NULL &&
                hiprtcGetProgramLog(program, log) == HIPRTC_SUCCESS) {
                snprintf(error, error_size, "%s", log);
            } else {
                snprintf(
                    error,
                    error_size,
                    "%s",
                    hiprtcGetErrorString(hiprtc_result));
            }
            free(log);
        } else {
            snprintf(
                error,
                error_size,
                "%s",
                hiprtcGetErrorString(hiprtc_result));
        }
    } else if (hiprtcGetCodeSize(program, &hsaco_size) != HIPRTC_SUCCESS ||
        hsaco_size == 0u) {
        snprintf(error, error_size, "HIPRTC returned an empty HSACO");
    } else {
        hsaco = (unsigned char*)malloc(hsaco_size);
        if (hsaco == NULL) {
            snprintf(error, error_size, "HSACO allocation failed");
        } else if (hiprtcGetCode(program, (char*)hsaco) != HIPRTC_SUCCESS) {
            snprintf(error, error_size, "HIPRTC HSACO retrieval failed");
        } else {
            success = 1;
        }
    }
    if (program != NULL) {
        (void)hiprtcDestroyProgram(&program);
    }
    free(source);
    if (!success) {
        free(hsaco);
        return 0;
    }
    *hsaco_ret = hsaco;
    *hsaco_size_ret = hsaco_size;
    return 1;
}

#endif /* SECANT_BENCH_HSACO_TEMPLATE_H_INCLUDED */
