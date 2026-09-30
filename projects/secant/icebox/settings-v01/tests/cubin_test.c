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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ROWS 257u
#define TEST_INPUTS 4u
#define TEST_TARGETS 2u
#define TEST_KERNELS 2u
#define TEST_ASTS_PER_KERNEL 5u
#define TEST_ASTS (TEST_KERNELS * TEST_ASTS_PER_KERNEL)
#define TEST_RUNNER_MODULES 12u
#define TEST_RUNNER_ASTS (TEST_RUNNER_MODULES * TEST_ASTS)
#define TEST_PARTIAL_ASTS (TEST_ASTS + 3u)
#define TEST_PATCH_INSTRUCTIONS 64u
#define TEST_HIGH_CAPACITY_ASTS 96u
#define TEST_HIGH_CAPACITY_INPUTS 8u
#define TEST_DYNAMIC_CONSTANTS 2u
#define TEST_DYNAMIC_SETTINGS 7u
#define TEST_DYNAMIC_LEAVES 8u
#define TEST_MATRIX_ASTS (TEST_ASTS + 3u)
#define TEST_MATRIX_GUARD_ELEMENTS 11u
#define TEST_MATRIX_CANARY_BITS 0x4f13579bu
#define TEST_DYNAMIC_STRESS_ITERATIONS 8u
#define TEST_DYNAMIC_STRESS_KERNELS 3u
#define TEST_DYNAMIC_STRESS_ASTS_PER_KERNEL 9u
#define TEST_DYNAMIC_STRESS_ASTS (TEST_DYNAMIC_STRESS_KERNELS * TEST_DYNAMIC_STRESS_ASTS_PER_KERNEL)
#define TEST_DYNAMIC_STRESS_SETTINGS 11u
#define TEST_DYNAMIC_STRESS_PATCH_INSTRUCTIONS \
    (TEST_DYNAMIC_STRESS_ASTS_PER_KERNEL * 192u)
#define TEST_GRAM_ASTS_PER_COHORT 32u
#define TEST_GRAM_ASTS (TEST_KERNELS * TEST_GRAM_ASTS_PER_COHORT + 3u)
#define TEST_PDE_GRID_X 64u
#define TEST_PDE_GRID_T 64u
#define TEST_PDE_ROWS (TEST_PDE_GRID_X * TEST_PDE_GRID_T)
#define TEST_PDE_INPUTS 3u
#define TEST_PDE_FEATURES 8u

typedef enum TestMatrixShape {
    TEST_MATRIX_SHAPE_MATERIALIZE = 0,
    TEST_MATRIX_SHAPE_SSE,
    TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE,
    TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE,
    TEST_MATRIX_SHAPE_AFFINE_STATS,
    TEST_MATRIX_SHAPE_COUNT
} TestMatrixShape;

static const SecantAstInstruction test_safe_div[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(
        SECANT_F32_BITS_ONE_MILLIONTH),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_add[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sin_cos[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_cos_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_fma[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_fma_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_div[] = {
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_bad_routine_arity[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_zero_arg_routine[] = {
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_TWO),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_zero_arg_call[] = {
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sqrt_rsqrt[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_rsqrt_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_exp_log[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_exp_f32,
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_log_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_tanh_neg[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_neg_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_tanh_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_min_max[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_max_f32,
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_min_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_constant[] = {
    secant_ast_encode_constant_f32_bits(
        SECANT_F32_BITS_TWO_AND_HALF),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_identity[] = {
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_add[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_sin_mul[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_leaf_add[] = {
    secant_ast_encode_dynamic_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_leaf_identity[] = {
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_leaf_sin_mul[] = {
    secant_ast_encode_dynamic_constant_or_column_input_f32(2u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_dynamic_column_input_f32(3u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_mixed_leaf_add[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_balanced_eight_input[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_static_column_input_f32(4u),
    secant_ast_encode_static_column_input_f32(5u),
    secant_ast_encode_min_f32,
    secant_ast_encode_static_column_input_f32(6u),
    secant_ast_encode_static_column_input_f32(7u),
    secant_ast_encode_max_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_one[] = {
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_u[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_ux[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_uxx[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_u_squared[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_u_ux[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_u_uxx[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_pde_ux_squared[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const test_pde_asts[] = {
    test_pde_one,
    test_pde_u,
    test_pde_ux,
    test_pde_uxx,
    test_pde_u_squared,
    test_pde_u_ux,
    test_pde_u_uxx,
    test_pde_ux_squared
};

static const SecantAstInstruction* const test_routines[] = {
    test_safe_div
};

static const SecantAstInstruction* const test_asts[] = {
    test_add,
    test_sin_cos,
    test_fma,
    test_div,
    test_sqrt_rsqrt,
    test_exp_log,
    test_tanh_neg,
    test_min_max,
    test_constant,
    test_identity
};

static const SecantAstInstruction* const test_affine_asts[] = {
    test_add,
    test_sin_cos,
    test_fma,
    test_sqrt_rsqrt,
    test_exp_log,
    test_tanh_neg,
    test_min_max,
    test_constant,
    test_identity
};

typedef struct TestState {
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    CUstream streams[2];
    CUdeviceptr input;
    CUdeviceptr targets;
    CUdeviceptr materialize_output;
    CUdeviceptr sse_output;
    SecantCubinPlan* materialize_handle;
    SecantCubinPlan* sse_handle;
    void* materialize_workspace;
    void* sse_workspace;
    unsigned char* materialize_cubin;
    unsigned char* sse_cubin;
    size_t materialize_cubin_size;
    size_t sse_cubin_size;
    void* materialize_module;
    void* sse_module;
    void* materialize_functions[TEST_KERNELS];
    void* sse_functions[TEST_KERNELS];
    int retained;
} TestState;

static void
test_print_cubin_error(const char* operation, SecantResult result) {
    fprintf(
        stderr,
        "%s failed: SecantResult(%d)\n",
        operation,
        (int)result);
}

static int
test_close(float actual, float expected, float tolerance) {
    return isfinite(actual) && isfinite(expected) &&
        fabsf(actual - expected) <= tolerance * (1.0f + fabsf(expected));
}

static int
test_compare(
    const char* name,
    const float* actual,
    const float* expected,
    size_t count,
    float tolerance
) {
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        if (!test_close(actual[idx], expected[idx], tolerance)) {
            fprintf(
                stderr,
                "%s mismatch at %zu: %.9g != %.9g\n",
                name,
                idx,
                actual[idx],
                expected[idx]);
            return 0;
        }
    }
    return 1;
}

static const char*
test_matrix_shape_name(TestMatrixShape shape) {
    switch (shape) {
        case TEST_MATRIX_SHAPE_MATERIALIZE:
            return "materialize";
        case TEST_MATRIX_SHAPE_SSE:
            return "sse";
        case TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE:
            return "dynamic_constant_sse";
        case TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE:
            return "dynamic_leaf_sse";
        case TEST_MATRIX_SHAPE_AFFINE_STATS:
            return "affine_stats";
        default:
            return "invalid";
    }
}

static void
test_identity_program_set(
    SecantAstInstruction program[3],
    SecantAstInstructionType input_type,
    size_t input_idx
) {
    program[0] = (SecantAstInstruction)input_type;
    program[1] = (SecantAstInstruction)input_idx;
    program[2] = (SecantAstInstruction)SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
}

static int
test_guarded_output_reset(CUdeviceptr allocation, size_t num_elements) {
    return cuMemsetD32(allocation, TEST_MATRIX_CANARY_BITS, num_elements) == CUDA_SUCCESS &&
        cuCtxSynchronize() == CUDA_SUCCESS;
}

static int
test_guarded_output_read(
    const char* name,
    CUdeviceptr allocation,
    size_t allocation_elements,
    size_t active_elements,
    float* active_output
) {
    uint32_t* values = NULL;
    size_t idx;
    int success = 1;

    if (allocation == 0u ||
        allocation_elements < 2u * TEST_MATRIX_GUARD_ELEMENTS ||
        active_elements > allocation_elements - 2u * TEST_MATRIX_GUARD_ELEMENTS ||
        active_output == NULL) {
        return 0;
    }
    values = (uint32_t*)malloc(allocation_elements * sizeof(*values));
    if (values == NULL ||
        cuMemcpyDtoH(values, allocation, allocation_elements * sizeof(*values)) != CUDA_SUCCESS) {
        free(values);
        return 0;
    }
    memcpy(
        active_output,
        values + TEST_MATRIX_GUARD_ELEMENTS,
        active_elements * sizeof(*active_output));
    for (idx = 0u; idx < TEST_MATRIX_GUARD_ELEMENTS; ++idx) {
        if (values[idx] != TEST_MATRIX_CANARY_BITS) {
            fprintf(stderr, "%s prefix guard overwritten at %zu\n", name, idx);
            success = 0;
            break;
        }
    }
    for (idx = TEST_MATRIX_GUARD_ELEMENTS + active_elements; idx < allocation_elements; ++idx) {
        if (values[idx] != TEST_MATRIX_CANARY_BITS) {
            fprintf(
                stderr,
                "%s inactive output overwritten at %zu (active=%zu)\n",
                name,
                idx - TEST_MATRIX_GUARD_ELEMENTS,
                active_elements);
            success = 0;
            break;
        }
    }
    free(values);
    return success;
}

static int
test_compile_cuda_source(
    const char* source,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char architecture[64];
    const char* options[3];
    unsigned char* cubin = NULL;
    char* log = NULL;
    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_result;
    size_t cubin_size = 0u;
    size_t log_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (source == NULL ||
        snprintf(
            architecture,
            sizeof(architecture),
            "--gpu-architecture=sm_%d%d",
            major,
            minor) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    nvrtc_result = nvrtcCreateProgram(
        &program,
        source,
        "secant_cubin_test.cu",
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
    } else if (nvrtcGetCUBINSize(program, &cubin_size) == NVRTC_SUCCESS &&
        cubin_size != 0u) {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin != NULL &&
            nvrtcGetCUBIN(program, (char*)cubin) == NVRTC_SUCCESS) {
            success = 1;
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    free(log);
    if (!success) {
        free(cubin);
        return 0;
    }
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

static int
test_target_stats_module_create(
    int major,
    int minor,
    CUmodule* module_ret,
    CUfunction* function_ret
) {
    const char* source;
    unsigned char* cubin = NULL;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    int success = 0;

    *module_ret = NULL;
    *function_ret = NULL;
    source = secant_cuda_target_stats_f32_source_get(&source_size);
    if (source == NULL || source_size < 2u || source[source_size - 1u] != '\0' ||
        strlen(source) + 1u != source_size ||
        !test_compile_cuda_source(source, major, minor, &cubin, &cubin_size)) {
        return 0;
    }
    if (cuModuleLoadData(module_ret, cubin) == CUDA_SUCCESS &&
        cuModuleGetFunction(
            function_ret,
            *module_ret,
            "secant_cuda_target_stats_f32") == CUDA_SUCCESS) {
        success = 1;
    }
    if (!success && *module_ret != NULL) {
        (void)cuModuleUnload(*module_ret);
        *module_ret = NULL;
    }
    free(cubin);
    return success;
}

static int
test_target_stats_run(
    CUfunction function,
    CUdeviceptr targets,
    size_t targets_leading_dimension,
    size_t num_rows,
    size_t num_targets,
    CUdeviceptr target_stats,
    size_t target_stats_leading_dimension
) {
    const unsigned int threads = 128u;
    const unsigned int grid_x = (unsigned int)(num_rows / 4096u + (num_rows % 4096u != 0u));
    void* arguments[6];

    if (function == NULL || targets == 0u || target_stats == 0u || num_rows == 0u || num_targets == 0u ||
        targets_leading_dimension < num_rows ||
        target_stats_leading_dimension < SECANT_AFFINE_TARGET_STAT_COUNT_F32 ||
        cuMemsetD32(
            target_stats,
            0u,
            num_targets * target_stats_leading_dimension) != CUDA_SUCCESS) {
        return 0;
    }
    arguments[0] = &targets;
    arguments[1] = &targets_leading_dimension;
    arguments[2] = &num_rows;
    arguments[3] = &num_targets;
    arguments[4] = &target_stats;
    arguments[5] = &target_stats_leading_dimension;
    return cuLaunchKernel(
            function,
            grid_x,
            (unsigned int)num_targets,
            1u,
            threads,
            1u,
            1u,
            0u,
            NULL,
            arguments,
            NULL) == CUDA_SUCCESS &&
        cuCtxSynchronize() == CUDA_SUCCESS;
}

static void
test_target_stats_cpu(
    const float* targets,
    size_t targets_leading_dimension,
    size_t num_rows,
    size_t num_targets,
    float* target_stats,
    size_t target_stats_leading_dimension
) {
    size_t target_idx;

    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        float target_sum = 0.0f;
        float target_square_sum = 0.0f;
        size_t row;

        for (row = 0u; row < num_rows; ++row) {
            const float target = targets[target_idx * targets_leading_dimension + row];

            target_sum += target;
            target_square_sum += target * target;
        }
        target_stats[target_idx * target_stats_leading_dimension + SECANT_AFFINE_TARGET_STAT_SUM_F32] = target_sum;
        target_stats[target_idx * target_stats_leading_dimension + SECANT_AFFINE_TARGET_STAT_SUM_SQUARED_F32] =
            target_square_sum;
    }
}

static int
test_affine_coefficients_recover(
    const float* ast_stats,
    const float* target_stats,
    size_t target_stats_leading_dimension,
    size_t num_rows,
    size_t num_targets,
    float* coefficients
) {
    const double prediction_sum = ast_stats[SECANT_AFFINE_AST_STAT_SUM_PREDICTION_F32];
    const double prediction_square_sum = ast_stats[SECANT_AFFINE_AST_STAT_SUM_PREDICTION_SQUARED_F32];
    const double rows = (double)num_rows;
    const double denominator = rows * prediction_square_sum - prediction_sum * prediction_sum;
    size_t target_idx;

    if (!isfinite(denominator) || denominator == 0.0) {
        return 0;
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        const double target_sum =
            target_stats[target_idx * target_stats_leading_dimension + SECANT_AFFINE_TARGET_STAT_SUM_F32];
        const double prediction_target_sum =
            ast_stats[SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + target_idx];
        const double slope =
            (rows * prediction_target_sum - prediction_sum * target_sum) / denominator;
        const double intercept = (target_sum - slope * prediction_sum) / rows;

        if (!isfinite(slope) || !isfinite(intercept)) {
            return 0;
        }
        coefficients[2u * target_idx] = (float)slope;
        coefficients[2u * target_idx + 1u] = (float)intercept;
    }
    return 1;
}

static int
test_compile_cubin_template(
    int sse,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    SecantCubinMaterializeRecipe materialize_recipe = secant_cubin_materialize_recipe_init();
    SecantCubinSSERecipe sse_recipe = secant_cubin_sse_recipe_init();
    char* source = NULL;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;

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
        test_print_cubin_error("CUBIN CUDA measurement", result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        return 0;
    }
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
        test_print_cubin_error("CUBIN CUDA generation", result);
        free(source);
        return 0;
    }
    result = test_compile_cuda_source(
        source,
        major,
        minor,
        cubin_ret,
        cubin_size_ret)
        ? SECANT_SUCCESS
        : SECANT_ERROR_COMPILE_FAILED;
    free(source);
    return result == SECANT_SUCCESS;
}

static int
test_compile_dynamic_cubin_template(
    const SecantCubinDynamicConstantSSERecipe* recipe,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char* source = NULL;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    result = secant_test_cubin_source_generate(
        recipe,
        NULL,
        0u,
        &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        test_print_cubin_error(
            "dynamic CUBIN CUDA measurement",
            result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source != NULL) {
        result = secant_test_cubin_source_generate(
            recipe,
            source,
            source_size,
            &written_size);
        if (result == SECANT_SUCCESS && written_size == source_size) {
            success = test_compile_cuda_source(
                source,
                major,
                minor,
                cubin_ret,
                cubin_size_ret);
        } else {
            test_print_cubin_error(
                "dynamic CUBIN CUDA generation",
                result);
        }
    }
    free(source);
    return success;
}

static int
test_compile_affine_stats_cubin_template(
    const SecantCubinAffineStatsRecipe* recipe,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char* source = NULL;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    result = secant_test_cubin_source_generate(recipe, NULL, 0u, &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        test_print_cubin_error("affine-stats CUBIN CUDA measurement", result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source != NULL) {
        result = secant_test_cubin_source_generate(recipe, source, source_size, &written_size);
        if (result == SECANT_SUCCESS && written_size == source_size) {
            success = test_compile_cuda_source(source, major, minor, cubin_ret, cubin_size_ret);
        } else {
            test_print_cubin_error("affine-stats CUBIN CUDA generation", result);
        }
    }
    free(source);
    return success;
}

static int
test_compile_gram_stats_cubin_template(
    const SecantCubinGramStatsRecipe* recipe,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char* source = NULL;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    result = secant_test_cubin_source_generate(recipe, NULL, 0u, &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        test_print_cubin_error("Gram-stats CUBIN CUDA measurement", result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source != NULL) {
        result = secant_test_cubin_source_generate(recipe, source, source_size, &written_size);
        if (result == SECANT_SUCCESS && written_size == source_size) {
            success = test_compile_cuda_source(source, major, minor, cubin_ret, cubin_size_ret);
        } else {
            test_print_cubin_error("Gram-stats CUBIN CUDA generation", result);
        }
    }
    free(source);
    return success;
}

static int
test_compile_dynamic_leaf_cubin_template(
    const SecantCubinDynamicLeafSSERecipe* recipe,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char* source = NULL;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    result = secant_test_cubin_source_generate(
        recipe,
        NULL,
        0u,
        &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        test_print_cubin_error("dynamic leaf CUBIN CUDA measurement", result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source != NULL) {
        result = secant_test_cubin_source_generate(
            recipe,
            source,
            source_size,
            &written_size);
        if (result == SECANT_SUCCESS && written_size == source_size) {
            success = test_compile_cuda_source(source, major, minor, cubin_ret, cubin_size_ret);
        } else {
            test_print_cubin_error("dynamic leaf CUBIN CUDA generation", result);
        }
    }
    free(source);
    return success;
}

static int
test_cuda_create(TestState* state, int* major_ret, int* minor_ret) {
    memset(state, 0, sizeof(*state));
    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&state->device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&state->context, state->device) != CUDA_SUCCESS) {
        return 0;
    }
    state->retained = 1;
    if (cuCtxGetCurrent(&state->previous_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(state->context) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            major_ret,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            state->device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            minor_ret,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            state->device) != CUDA_SUCCESS ||
        cuStreamCreate(state->streams, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
        cuStreamCreate(state->streams + 1u, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

static void
test_state_destroy(TestState* state) {
    if (state->materialize_module != NULL) {
        (void)cuModuleUnload((CUmodule)state->materialize_module);
    }
    if (state->sse_module != NULL) {
        (void)cuModuleUnload((CUmodule)state->sse_module);
    }
    free(state->materialize_workspace);
    free(state->sse_workspace);
    free(state->materialize_cubin);
    free(state->sse_cubin);
    if (state->input != 0u) {
        (void)cuMemFree(state->input);
    }
    if (state->targets != 0u) {
        (void)cuMemFree(state->targets);
    }
    if (state->materialize_output != 0u) {
        (void)cuMemFree(state->materialize_output);
    }
    if (state->sse_output != 0u) {
        (void)cuMemFree(state->sse_output);
    }
    if (state->streams[0] != NULL) {
        (void)cuStreamDestroy(state->streams[0]);
    }
    if (state->streams[1] != NULL) {
        (void)cuStreamDestroy(state->streams[1]);
    }
    if (state->retained) {
        (void)cuCtxSetCurrent(state->previous_context);
        (void)cuDevicePrimaryCtxRelease(state->device);
    }
}

static int
test_create_templates(TestState* state, int major, int minor) {
    const size_t patch_capacity_instructions = TEST_ASTS_PER_KERNEL * TEST_PATCH_INSTRUCTIONS + 13u;
    SecantCubinMaterializeRecipe materialize_recipe = secant_cubin_materialize_recipe_init();
    SecantCubinSSERecipe sse_recipe = secant_cubin_sse_recipe_init();
    SecantCubinMaterializeRecipe invalid_recipe;
    size_t materialize_workspace_size = 0u;
    size_t sse_workspace_size = 0u;
    SecantCubinPlan* invalid_handle = NULL;
    unsigned char* malformed_cubin;
    unsigned char* misaligned_cubin;
    unsigned char* misaligned_plan_storage;
    char too_small_source[1];
    size_t measured_source_size = 0u;
    SecantResult result;

    materialize_recipe.num_kernels = TEST_KERNELS;
    materialize_recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    materialize_recipe.num_inputs = TEST_INPUTS;
    materialize_recipe.patch_capacity_instructions = patch_capacity_instructions;
    sse_recipe.num_kernels = TEST_KERNELS;
    sse_recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    sse_recipe.num_inputs = TEST_INPUTS;
    sse_recipe.num_targets = TEST_TARGETS;
    sse_recipe.tile_rows = 128u;
    sse_recipe.threads_per_block = 128u;
    sse_recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (secant_cubin_recipe_validate(&materialize_recipe.header) != SECANT_SUCCESS ||
        secant_cubin_recipe_validate(&sse_recipe.header) != SECANT_SUCCESS) {
        fprintf(stderr, "valid CUBIN recipe descriptor was rejected\n");
        return 0;
    }
    result = secant_cubin_source_size(&materialize_recipe.header, &measured_source_size);
    if (result != SECANT_SUCCESS || measured_source_size <= sizeof(too_small_source) ||
        secant_cubin_source_write(
            &materialize_recipe.header,
            too_small_source,
            sizeof(too_small_source)) != SECANT_ERROR_INSUFFICIENT_BUFFER) {
        fprintf(stderr, "split CUBIN source measure/write contract failed\n");
        return 0;
    }
    invalid_recipe = materialize_recipe;
    invalid_recipe.header.version = 0u;
    if (secant_cubin_recipe_validate(&invalid_recipe.header) != SECANT_ERROR_UNSUPPORTED_VERSION) {
        fprintf(stderr, "CUBIN recipe accepted an unsupported version\n");
        return 0;
    }
    measured_source_size = 1u;
    if (secant_cubin_source_size(&invalid_recipe.header, &measured_source_size) !=
            SECANT_ERROR_UNSUPPORTED_VERSION ||
        measured_source_size != 0u) {
        fprintf(stderr, "CUBIN source measurement did not independently reject an unsupported version\n");
        return 0;
    }
    invalid_recipe = materialize_recipe;
    invalid_recipe.header.struct_size = (uint32_t)(sizeof(invalid_recipe) - 1u);
    if (secant_cubin_recipe_validate(&invalid_recipe.header) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "CUBIN recipe accepted a truncated descriptor\n");
        return 0;
    }
    invalid_recipe = materialize_recipe;
    invalid_recipe.header.flags = 1u;
    if (secant_cubin_recipe_validate(&invalid_recipe.header) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "CUBIN recipe accepted unsupported flags\n");
        return 0;
    }
    invalid_recipe = materialize_recipe;
    invalid_recipe.header.shape = UINT32_MAX;
    if (secant_cubin_recipe_validate(&invalid_recipe.header) != SECANT_ERROR_UNSUPPORTED_SHAPE) {
        fprintf(stderr, "CUBIN recipe accepted an unknown shape\n");
        return 0;
    }
    {
        struct ExtendedRecipe {
            SecantCubinMaterializeRecipe recipe;
            uint64_t trailing;
        } extended_recipe;

        memset(&extended_recipe, 0, sizeof(extended_recipe));
        extended_recipe.recipe = materialize_recipe;
        extended_recipe.recipe.header.struct_size = (uint32_t)sizeof(extended_recipe);
        extended_recipe.trailing = UINT64_C(0x0123456789abcdef);
        if (secant_cubin_recipe_validate(&extended_recipe.recipe.header) != SECANT_SUCCESS ||
            extended_recipe.trailing != UINT64_C(0x0123456789abcdef)) {
            fprintf(stderr, "CUBIN recipe rejected or modified a compatible trailing extension\n");
            return 0;
        }
    }
    if (!test_compile_cubin_template(
            0,
            TEST_KERNELS,
            TEST_ASTS_PER_KERNEL,
            TEST_INPUTS,
            0u,
            0u,
            0u,
            patch_capacity_instructions,
            major,
            minor,
            &state->materialize_cubin,
            &state->materialize_cubin_size) ||
        !test_compile_cubin_template(
            1,
            TEST_KERNELS,
            TEST_ASTS_PER_KERNEL,
            TEST_INPUTS,
            TEST_TARGETS,
            128u,
            128u,
            patch_capacity_instructions,
            major,
            minor,
            &state->sse_cubin,
            &state->sse_cubin_size)) {
        return 0;
    }
    {
        SecantCubinDynamicConstantSSERecipe invalid_dynamic =
            secant_cubin_dynamic_constant_sse_recipe_init();

        invalid_dynamic.num_kernels = 1u;
        invalid_dynamic.asts_per_kernel = 1u;
        invalid_dynamic.num_input_columns = 33u;
        invalid_dynamic.num_input_constants = 1u;
        invalid_dynamic.num_targets = 1u;
        invalid_dynamic.tile_rows = 64u;
        invalid_dynamic.threads_per_block = 128u;
        invalid_dynamic.patch_capacity_instructions = 64u;
        materialize_workspace_size = 0u;
        if (secant_cubin_plan_storage_size(
                &invalid_dynamic.header,
                state->materialize_cubin,
                state->materialize_cubin_size,
                &materialize_workspace_size) != SECANT_ERROR_INVALID_VALUE ||
            materialize_workspace_size != 0u) {
            fprintf(stderr, "CUBIN inspection did not independently validate dynamic input limits\n");
            return 0;
        }
    }
    malformed_cubin = (unsigned char*)malloc(state->materialize_cubin_size);
    if (malformed_cubin == NULL) {
        return 0;
    }
    memcpy(
        malformed_cubin,
        state->materialize_cubin,
        state->materialize_cubin_size);
    malformed_cubin[0] ^= 1u;
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        malformed_cubin,
        state->materialize_cubin_size,
        NULL,
        0u,
        &materialize_workspace_size,
        &invalid_handle);
    free(malformed_cubin);
    if (result != SECANT_ERROR_PARSE_FAILED || invalid_handle != NULL) {
        test_print_cubin_error(
            "secant_test_cubin_inspect malformed ELF",
            result);
        return 0;
    }
    materialize_workspace_size = 0u;
    misaligned_cubin = (unsigned char*)malloc(
        state->materialize_cubin_size + 1u);
    if (misaligned_cubin == NULL) {
        return 0;
    }
    memcpy(
        misaligned_cubin + 1u,
        state->materialize_cubin,
        state->materialize_cubin_size);
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        misaligned_cubin + 1u,
        state->materialize_cubin_size,
        NULL,
        0u,
        &materialize_workspace_size,
        &invalid_handle);
    free(misaligned_cubin);
    if (result != SECANT_ERROR_INVALID_VALUE || invalid_handle != NULL) {
        test_print_cubin_error(
            "secant_test_cubin_inspect alignment",
            result);
        return 0;
    }
    materialize_workspace_size = 0u;
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        state->materialize_cubin,
        state->materialize_cubin_size,
        NULL,
        0u,
        &materialize_workspace_size,
        &state->materialize_handle);
    if (result != SECANT_SUCCESS || materialize_workspace_size == 0u) {
        test_print_cubin_error(
            "secant_test_cubin_inspect measure",
            result);
        return 0;
    }
    misaligned_plan_storage = (unsigned char*)malloc(materialize_workspace_size + 1u);
    if (misaligned_plan_storage == NULL) {
        return 0;
    }
    invalid_handle = NULL;
    result = secant_cubin_plan_init(
        &materialize_recipe.header,
        state->materialize_cubin,
        state->materialize_cubin_size,
        misaligned_plan_storage + 1u,
        materialize_workspace_size,
        &invalid_handle);
    free(misaligned_plan_storage);
    if (result != SECANT_SUCCESS || invalid_handle == NULL) {
        test_print_cubin_error("secant_cubin_plan_init odd storage alignment", result);
        return 0;
    }
    state->materialize_workspace = malloc(materialize_workspace_size);
    if (state->materialize_workspace == NULL) {
        return 0;
    }
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        state->materialize_cubin,
        state->materialize_cubin_size,
        state->materialize_workspace,
        materialize_workspace_size - 1u,
        &materialize_workspace_size,
        &state->materialize_handle);
    if (result != SECANT_ERROR_INSUFFICIENT_BUFFER ||
        state->materialize_handle != NULL) {
        test_print_cubin_error(
            "secant_test_cubin_inspect capacity",
            result);
        return 0;
    }
    result = secant_test_cubin_inspect(
        &materialize_recipe,
        state->materialize_cubin,
        state->materialize_cubin_size,
        state->materialize_workspace,
        materialize_workspace_size,
        &materialize_workspace_size,
        &state->materialize_handle);
    if (result != SECANT_SUCCESS) {
        test_print_cubin_error("secant_test_cubin_inspect", result);
        return 0;
    }
    result = secant_test_cubin_inspect(
        &sse_recipe,
        state->sse_cubin,
        state->sse_cubin_size,
        NULL,
        0u,
        &sse_workspace_size,
        &state->sse_handle);
    if (result != SECANT_SUCCESS || sse_workspace_size == 0u) {
        test_print_cubin_error("secant_test_cubin_inspect measure", result);
        return 0;
    }
    state->sse_workspace = malloc(sse_workspace_size);
    if (state->sse_workspace == NULL) {
        return 0;
    }
    result = secant_test_cubin_inspect(
        &sse_recipe,
        state->sse_cubin,
        state->sse_cubin_size,
        state->sse_workspace,
        sse_workspace_size,
        &sse_workspace_size,
        &state->sse_handle);
    if (result != SECANT_SUCCESS) {
        test_print_cubin_error("secant_test_cubin_inspect", result);
        return 0;
    }
    {
        SecantCubinPlanInfo info = secant_cubin_plan_info_init();
        SecantCubinMaterializeRecipe recovered = secant_cubin_materialize_recipe_init();
        SecantCubinSSERecipe wrong_shape = secant_cubin_sse_recipe_init();
        struct ExtendedPlanInfo {
            SecantCubinPlanInfo info;
            uint64_t trailing;
        } extended_info;
        struct ExtendedRecipe {
            SecantCubinMaterializeRecipe recipe;
            uint64_t trailing;
        } extended_recipe;

        result = secant_cubin_plan_info_get(state->materialize_handle, &info);
        if (result == SECANT_SUCCESS) {
            result = secant_cubin_plan_recipe_get(state->materialize_handle, &recovered.header);
        }
        if (result != SECANT_SUCCESS || info.shape != materialize_recipe.header.shape ||
            info.cubin_size != state->materialize_cubin_size ||
            recovered.num_kernels != materialize_recipe.num_kernels ||
            recovered.asts_per_kernel != materialize_recipe.asts_per_kernel ||
            recovered.num_inputs != materialize_recipe.num_inputs ||
            recovered.patch_capacity_instructions != materialize_recipe.patch_capacity_instructions) {
            test_print_cubin_error("secant_cubin_plan_info_get", result);
            return 0;
        }
        if (secant_cubin_plan_recipe_get(state->materialize_handle, &wrong_shape.header) !=
            SECANT_ERROR_UNSUPPORTED_SHAPE) {
            fprintf(stderr, "plan info accepted a mismatched concrete recipe\n");
            return 0;
        }
        info = secant_cubin_plan_info_init();
        info.version = 0u;
        info.shape = UINT32_MAX;
        if (secant_cubin_plan_info_get(state->materialize_handle, &info) !=
                SECANT_ERROR_UNSUPPORTED_VERSION ||
            info.shape != UINT32_MAX) {
            fprintf(stderr, "plan info unsupported-version handling modified output\n");
            return 0;
        }
        memset(&extended_info, 0, sizeof(extended_info));
        extended_info.info = secant_cubin_plan_info_init();
        extended_info.info.struct_size = (uint32_t)sizeof(extended_info);
        extended_info.trailing = UINT64_C(0xfedcba9876543210);
        if (secant_cubin_plan_info_get(state->materialize_handle, &extended_info.info) != SECANT_SUCCESS ||
            extended_info.trailing != UINT64_C(0xfedcba9876543210)) {
            fprintf(stderr, "plan info did not preserve a compatible trailing extension\n");
            return 0;
        }
        memset(&extended_recipe, 0, sizeof(extended_recipe));
        extended_recipe.recipe = secant_cubin_materialize_recipe_init();
        extended_recipe.recipe.header.struct_size = (uint32_t)sizeof(extended_recipe);
        extended_recipe.trailing = UINT64_C(0xa5a5a5a55a5a5a5a);
        if (secant_cubin_plan_recipe_get(state->materialize_handle, &extended_recipe.recipe.header) !=
                SECANT_SUCCESS ||
            extended_recipe.trailing != UINT64_C(0xa5a5a5a55a5a5a5a)) {
            fprintf(stderr, "plan recipe did not preserve a compatible trailing extension\n");
            return 0;
        }
    }
    return 1;
}

static int
test_write_cubins(TestState* state) {
    unsigned char* materialize_cubin = NULL;
    unsigned char* sse_cubin = NULL;
    const SecantAstInstruction* bad_asts[TEST_ASTS];
    const SecantAstInstruction* zero_arg_asts[TEST_ASTS];
    const SecantAstInstruction* zero_arg_routines[] = {
        test_zero_arg_routine
    };
    SecantResult result;
    size_t ast_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < TEST_ASTS; ++ast_idx) {
        bad_asts[ast_idx] = ast_idx == 0u ? test_bad_routine_arity : test_add;
        zero_arg_asts[ast_idx] = test_zero_arg_call;
    }
    materialize_cubin = (unsigned char*)malloc(
        state->materialize_cubin_size);
    sse_cubin = (unsigned char*)malloc(state->sse_cubin_size);
    if (materialize_cubin == NULL || sse_cubin == NULL) {
        success = 0;
    }
    if (success) {
        memcpy(
            materialize_cubin,
            state->materialize_cubin,
            state->materialize_cubin_size);
        memcpy(sse_cubin, state->sse_cubin, state->sse_cubin_size);
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            state->materialize_handle,
            zero_arg_routines,
            1u,
            zero_arg_asts,
            TEST_ASTS,
            materialize_cubin,
            state->materialize_cubin_size);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "secant_test_cubin_specialize_into zero-argument routine",
                result);
            success = 0;
        }
        memcpy(
            materialize_cubin,
            state->materialize_cubin,
            state->materialize_cubin_size);
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            state->materialize_handle,
            test_routines,
            1u,
            bad_asts,
            TEST_ASTS,
            materialize_cubin,
            state->materialize_cubin_size);
        if (result != SECANT_ERROR_STACK_UNDERFLOW) {
            test_print_cubin_error(
                "secant_test_cubin_specialize_into routine operands",
                result);
            success = 0;
        }
        memcpy(
            materialize_cubin,
            state->materialize_cubin,
            state->materialize_cubin_size);
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            state->materialize_handle,
            test_routines,
            1u,
            test_asts,
            TEST_ASTS,
            materialize_cubin,
            state->materialize_cubin_size - 1u);
        if (result != SECANT_ERROR_INSUFFICIENT_BUFFER) {
            test_print_cubin_error(
                "secant_test_cubin_specialize_into capacity",
                result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            state->materialize_handle,
            test_routines,
            1u,
            test_asts,
            TEST_ASTS,
            materialize_cubin,
            state->materialize_cubin_size);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("secant_test_cubin_specialize_into materialize", result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            state->sse_handle,
            test_routines,
            1u,
            test_asts,
            TEST_ASTS,
            sse_cubin,
            state->sse_cubin_size);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("secant_test_cubin_specialize_into SSE", result);
            success = 0;
        }
    }
    if (success &&
        (state->materialize_cubin_size < 4u ||
         state->sse_cubin_size < 4u ||
         memcmp(materialize_cubin, "\x7f" "ELF", 4u) != 0 ||
         memcmp(sse_cubin, "\x7f" "ELF", 4u) != 0)) {
        fprintf(stderr, "specialize did not preserve ELF CUBIN data\n");
        success = 0;
    }
    free(sse_cubin);
    free(materialize_cubin);
    return success;
}

static int
test_high_capacity_sse_patch(int major, int minor) {
    const SecantAstInstruction* asts[TEST_HIGH_CAPACITY_ASTS];
    SecantCubinPlan* handle = NULL;
    void* workspace = NULL;
    unsigned char* template_cubin = NULL;
    unsigned char* cubin = NULL;
    const size_t patch_capacity_instructions = TEST_HIGH_CAPACITY_ASTS * TEST_PATCH_INSTRUCTIONS + 11u;
    SecantCubinSSERecipe recipe = secant_cubin_sse_recipe_init();
    size_t template_cubin_size = 0u;
    size_t workspace_size = 0u;
    size_t ast_idx;
    SecantResult result;
    int success = 1;

    recipe.num_kernels = 1u;
    recipe.asts_per_kernel = TEST_HIGH_CAPACITY_ASTS;
    recipe.num_inputs = TEST_HIGH_CAPACITY_INPUTS;
    recipe.num_targets = TEST_TARGETS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    for (ast_idx = 0u; ast_idx < TEST_HIGH_CAPACITY_ASTS; ++ast_idx) {
        asts[ast_idx] = test_balanced_eight_input;
    }
    if (!test_compile_cubin_template(
            1,
            1u,
            TEST_HIGH_CAPACITY_ASTS,
            TEST_HIGH_CAPACITY_INPUTS,
            TEST_TARGETS,
            128u,
            128u,
            patch_capacity_instructions,
            major,
            minor,
            &template_cubin,
            &template_cubin_size)) {
        return 0;
    }
    result = secant_test_cubin_inspect(
        &recipe,
        template_cubin,
        template_cubin_size,
        NULL,
        0u,
        &workspace_size,
        &handle);
    if (result != SECANT_SUCCESS || workspace_size == 0u) {
        test_print_cubin_error("high-capacity inspect measure", result);
        success = 0;
    }
    if (success) {
        workspace = malloc(workspace_size);
        cubin = (unsigned char*)malloc(template_cubin_size);
        if (workspace == NULL || cubin == NULL) {
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            template_cubin,
            template_cubin_size,
            workspace,
            workspace_size,
            &workspace_size,
            &handle);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("high-capacity inspect", result);
            success = 0;
        }
    }
    if (success) {
        memcpy(cubin, template_cubin, template_cubin_size);
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            handle,
            NULL,
            0u,
            asts,
            TEST_HIGH_CAPACITY_ASTS,
            cubin,
            template_cubin_size);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "high-capacity secant_test_cubin_specialize_into",
                result);
            success = 0;
        }
    }
    free(template_cubin);
    free(cubin);
    free(workspace);
    return success;
}

static int
test_load_modules(TestState* state) {
    unsigned char* materialize_cubin;
    unsigned char* sse_cubin;
    SecantResult result;
    SecantResult cuda_result;
    int success = 1;

    materialize_cubin = (unsigned char*)malloc(
        state->materialize_cubin_size);
    sse_cubin = (unsigned char*)malloc(state->sse_cubin_size);
    if (materialize_cubin == NULL || sse_cubin == NULL) {
        free(sse_cubin);
        free(materialize_cubin);
        return 0;
    }
    memcpy(
        materialize_cubin,
        state->materialize_cubin,
        state->materialize_cubin_size);
    memcpy(sse_cubin, state->sse_cubin, state->sse_cubin_size);
    result = secant_test_cubin_specialize_into(
        state->materialize_handle,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        materialize_cubin,
        state->materialize_cubin_size);
    if (result != SECANT_SUCCESS) {
        test_print_cubin_error("secant_test_cubin_specialize_into materialize", result);
        success = 0;
    }
    if (success) {
        result = secant_test_cubin_specialize_into(
            state->sse_handle,
            test_routines,
            1u,
            test_asts,
            TEST_ASTS,
            sse_cubin,
            state->sse_cubin_size);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("secant_test_cubin_specialize_into SSE", result);
            success = 0;
        }
    }
    if (success) {
        cuda_result = secant_test_cuda_module_load(
            materialize_cubin,
            state->materialize_cubin_size,
            TEST_KERNELS,
            &state->materialize_module,
            state->materialize_functions);
        if (cuda_result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "secant_test_cuda_module_load materialize failed: %s\n",
                secant_test_cuda_result_to_string(cuda_result));
            success = 0;
        }
    }
    if (success) {
        cuda_result = secant_test_cuda_module_load(
            sse_cubin,
            state->sse_cubin_size,
            TEST_KERNELS,
            &state->sse_module,
            state->sse_functions);
        if (cuda_result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "secant_test_cuda_module_load SSE failed: %s\n",
                secant_test_cuda_result_to_string(cuda_result));
            success = 0;
        }
    }
    free(sse_cubin);
    free(materialize_cubin);
    return success;
}

static int
test_run_gpu(
    TestState* state,
    const float* input,
    const float* targets,
    float* materialize_output,
    float* sse_output
) {
    const size_t input_bytes = TEST_INPUTS * TEST_ROWS * sizeof(float);
    const size_t target_bytes = TEST_TARGETS * TEST_ROWS * sizeof(float);
    const size_t materialize_bytes = TEST_ASTS * TEST_ROWS * sizeof(float);
    const size_t sse_bytes = TEST_ASTS * TEST_TARGETS * sizeof(float);
    SecantResult result;
    size_t kernel_idx;

    if (cuMemAlloc(&state->input, input_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&state->targets, target_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&state->materialize_output, materialize_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&state->sse_output, sse_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(state->input, input, input_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(state->targets, targets, target_bytes) != CUDA_SUCCESS ||
        cuMemsetD8(state->materialize_output, 0u, materialize_bytes) != CUDA_SUCCESS ||
        cuMemsetD8(state->sse_output, 0u, sse_bytes) != CUDA_SUCCESS) {
        return 0;
    }
    for (kernel_idx = 0u; kernel_idx < TEST_KERNELS; ++kernel_idx) {
        result = secant_test_cuda_run_static_column_materialize(
            state->materialize_functions[kernel_idx],
            TEST_INPUTS,
            TEST_ASTS_PER_KERNEL,
            (const float*)(uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            state->streams[kernel_idx % 2u],
            (float*)(uintptr_t)(
                state->materialize_output +
                kernel_idx * TEST_ASTS_PER_KERNEL * TEST_ROWS *
                    sizeof(float)),
            TEST_ASTS_PER_KERNEL * TEST_ROWS,
            TEST_ROWS);
        if (result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "materialize launch failed: %s\n",
                secant_test_cuda_result_to_string(result));
            return 0;
        }
        result = secant_test_cuda_run_static_column_sse(
            state->sse_functions[kernel_idx],
            TEST_INPUTS,
            TEST_ASTS_PER_KERNEL,
            TEST_TARGETS,
            128u,
            128u,
            (const float*)(uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            (const float*)(uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            state->streams[kernel_idx % 2u],
            (float*)(uintptr_t)(
                state->sse_output +
                kernel_idx * TEST_ASTS_PER_KERNEL * TEST_TARGETS *
                    sizeof(float)),
            TEST_ASTS_PER_KERNEL * TEST_TARGETS,
            TEST_TARGETS);
        if (result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "SSE launch failed: %s\n",
                secant_test_cuda_result_to_string(result));
            return 0;
        }
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS ||
        cuMemcpyDtoH(
            materialize_output,
            state->materialize_output,
            materialize_bytes) != CUDA_SUCCESS ||
        cuMemcpyDtoH(
            sse_output,
            state->sse_output,
            sse_bytes) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

static int
test_materialize_respecialize_in_place(
    TestState* state,
    const float* input
) {
    static const char* phase_names[] = {
        "materialize re-specialize short",
        "materialize re-specialize long",
        "materialize re-specialize short again"
    };
    const SecantAstInstruction* short_asts[TEST_ASTS];
    const SecantAstInstruction* const* phase_asts[3];
    unsigned char* cubin = NULL;
    void* module = NULL;
    void* functions[TEST_KERNELS];
    float expected[TEST_ASTS * TEST_ROWS];
    float actual[TEST_ASTS * TEST_ROWS];
    CUdeviceptr output = 0u;
    SecantResult result;
    size_t ast_idx;
    size_t phase_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < TEST_ASTS; ++ast_idx) {
        short_asts[ast_idx] = test_identity;
    }
    phase_asts[0] = short_asts;
    phase_asts[1] = test_asts;
    phase_asts[2] = short_asts;
    cubin = (unsigned char*)malloc(state->materialize_cubin_size);
    if (cubin == NULL ||
        cuMemAlloc(&output, sizeof(actual)) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        memcpy(cubin, state->materialize_cubin, state->materialize_cubin_size);
    }
    for (phase_idx = 0u; phase_idx < 3u && success; ++phase_idx) {
        size_t kernel_idx;

        result = secant_test_cubin_specialize_into(
            state->materialize_handle,
            test_routines,
            1u,
            phase_asts[phase_idx],
            TEST_ASTS,
            cubin,
            state->materialize_cubin_size);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(phase_names[phase_idx], result);
            success = 0;
            break;
        }
        result = secant_test_cpu_materialize_run(
            TEST_INPUTS,
            test_routines,
            1u,
            phase_asts[phase_idx],
            TEST_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            TEST_ASTS * TEST_ROWS,
            TEST_ROWS);
        if (result != SECANT_SUCCESS ||
            secant_test_cuda_module_load(
                cubin,
                state->materialize_cubin_size,
                TEST_KERNELS,
                &module,
                functions) != SECANT_SUCCESS ||
            cuMemsetD8(output, 0u, sizeof(actual)) != CUDA_SUCCESS) {
            success = 0;
            break;
        }
        for (kernel_idx = 0u; kernel_idx < TEST_KERNELS; ++kernel_idx) {
            result = secant_test_cuda_run_static_column_materialize(
                functions[kernel_idx],
                TEST_INPUTS,
                TEST_ASTS_PER_KERNEL,
                (const float*)(uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                state->streams[kernel_idx % 2u],
                (float*)(uintptr_t)(
                    output + kernel_idx * TEST_ASTS_PER_KERNEL * TEST_ROWS * sizeof(float)),
                TEST_ASTS_PER_KERNEL * TEST_ROWS,
                TEST_ROWS);
            if (result != SECANT_SUCCESS) {
                success = 0;
                break;
            }
        }
        if (success &&
            (cuCtxSynchronize() != CUDA_SUCCESS ||
             cuMemcpyDtoH(actual, output, sizeof(actual)) != CUDA_SUCCESS ||
             !test_compare(
                 phase_names[phase_idx],
                 actual,
                 expected,
                 TEST_ASTS * TEST_ROWS,
                 3.0e-5f))) {
            success = 0;
        }
        if (module != NULL) {
            if (cuModuleUnload((CUmodule)module) != CUDA_SUCCESS) {
                success = 0;
            }
            module = NULL;
        }
    }
    if (module != NULL && cuModuleUnload((CUmodule)module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (output != 0u && cuMemFree(output) != CUDA_SUCCESS) {
        success = 0;
    }
    free(cubin);
    return success;
}

static int
test_run_bulk_runner(
    TestState* state,
    const float* input,
    const float* targets
) {
    const size_t materialize_elements = TEST_RUNNER_ASTS * TEST_ROWS;
    const size_t sse_elements = TEST_RUNNER_ASTS * TEST_TARGETS;
    const size_t materialize_bytes = materialize_elements * sizeof(float);
    const size_t sse_bytes = sse_elements * sizeof(float);
    const SecantAstInstruction* runner_asts[TEST_RUNNER_ASTS];
    float* expected_materialize = NULL;
    float* actual_materialize = NULL;
    float* expected_sse = NULL;
    float* actual_sse = NULL;
    CUdeviceptr materialize_output = 0u;
    CUdeviceptr sse_output = 0u;
    SecantCubinRunner materialize_runner = NULL;
    SecantCubinRunner sse_runner = NULL;
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantResult cpu_result;
    SecantResult result;
    size_t ast_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < TEST_RUNNER_ASTS; ++ast_idx) {
        runner_asts[ast_idx] = test_asts[ast_idx % TEST_ASTS];
    }
    expected_materialize = (float*)malloc(materialize_bytes);
    actual_materialize = (float*)malloc(materialize_bytes);
    expected_sse = (float*)calloc(sse_elements, sizeof(float));
    actual_sse = (float*)malloc(sse_bytes);
    if (expected_materialize == NULL ||
        actual_materialize == NULL ||
        expected_sse == NULL ||
        actual_sse == NULL) {
        success = 0;
    }
    if (success) {
        cpu_result = secant_test_cpu_materialize_run(
            TEST_INPUTS,
            test_routines,
            1u,
            runner_asts,
            TEST_RUNNER_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected_materialize,
            materialize_elements,
            TEST_ROWS);
        success = cpu_result == SECANT_SUCCESS;
    }
    if (success) {
        cpu_result = secant_test_cpu_sse_run(
            TEST_INPUTS,
            TEST_TARGETS,
            test_routines,
            1u,
            runner_asts,
            TEST_RUNNER_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected_sse,
            sse_elements,
            TEST_TARGETS);
        success = cpu_result == SECANT_SUCCESS;
    }
    if (success &&
        (cuMemAlloc(&materialize_output, materialize_bytes) !=
             CUDA_SUCCESS ||
         cuMemAlloc(&sse_output, sse_bytes) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        SecantCubinRunnerOptions invalid_options = secant_cubin_runner_options_init();

        invalid_options.version = 0u;
        materialize_runner = (SecantCubinRunner)(uintptr_t)1u;
        result = secant_cubin_runner_create(
            state->materialize_handle,
            state->materialize_cubin,
            state->materialize_cubin_size,
            &invalid_options,
            &materialize_runner);
        if (result != SECANT_ERROR_UNSUPPORTED_VERSION || materialize_runner != NULL) {
            fprintf(stderr, "CUBIN runner options unsupported-version contract failed\n");
            success = 0;
        }
        invalid_options = secant_cubin_runner_options_init();
        invalid_options.flags = 1u;
        materialize_runner = (SecantCubinRunner)(uintptr_t)1u;
        result = secant_cubin_runner_create(
            state->materialize_handle,
            state->materialize_cubin,
            state->materialize_cubin_size,
            &invalid_options,
            &materialize_runner);
        if (result != SECANT_ERROR_INVALID_VALUE || materialize_runner != NULL) {
            fprintf(stderr, "CUBIN runner options accepted unsupported flags\n");
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            state->materialize_handle,
            state->materialize_cubin,
            state->materialize_cubin_size,
            2u,
            2u,
            &materialize_runner);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "secant_test_cubin_runner_create",
                result);
            success = 0;
        }
    }
    if (success) {
        SecantCubinSSERun wrong_shape = secant_cubin_sse_run_init();
        SecantCubinMaterializeRun bad_header = secant_cubin_materialize_run_init();
        SecantRunnerStats invalid_stats = secant_runner_stats_init();
        struct ExtendedStats {
            SecantRunnerStats stats;
            uint64_t trailing;
        } extended_stats;

        stats.num_asts = 99u;
        if (secant_cubin_runner_run(materialize_runner, &wrong_shape.header, &stats) !=
                SECANT_ERROR_UNSUPPORTED_SHAPE ||
            stats.num_asts != 0u) {
            fprintf(stderr, "CUBIN runner accepted a mismatched run shape\n");
            success = 0;
        }
        bad_header.header.version = 0u;
        if (success && secant_cubin_runner_run(materialize_runner, &bad_header.header, NULL) !=
            SECANT_ERROR_UNSUPPORTED_VERSION) {
            fprintf(stderr, "CUBIN runner accepted an unsupported run version\n");
            success = 0;
        }
        bad_header = secant_cubin_materialize_run_init();
        bad_header.header.flags = 1u;
        if (success && secant_cubin_runner_run(materialize_runner, &bad_header.header, NULL) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "CUBIN runner accepted unsupported run flags\n");
            success = 0;
        }
        invalid_stats.version = 0u;
        if (success && secant_cubin_runner_run(materialize_runner, &bad_header.header, &invalid_stats) !=
            SECANT_ERROR_UNSUPPORTED_VERSION) {
            fprintf(stderr, "CUBIN runner accepted unsupported statistics version\n");
            success = 0;
        }
        invalid_stats = secant_runner_stats_init();
        invalid_stats.flags = 1u;
        if (success && secant_cubin_runner_run(materialize_runner, &bad_header.header, &invalid_stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "CUBIN runner accepted unsupported statistics flags\n");
            success = 0;
        }
        memset(&extended_stats, 0, sizeof(extended_stats));
        extended_stats.stats = secant_runner_stats_init();
        extended_stats.stats.struct_size = (uint32_t)sizeof(extended_stats);
        extended_stats.trailing = UINT64_C(0x55aa55aaaa55aa55);
        if (success && secant_cubin_runner_run(materialize_runner, &wrong_shape.header, &extended_stats.stats) !=
                SECANT_ERROR_UNSUPPORTED_SHAPE) {
            fprintf(stderr, "CUBIN runner rejected compatible extended statistics\n");
            success = 0;
        }
        if (success && (extended_stats.stats.struct_size != sizeof(extended_stats) ||
            extended_stats.trailing != UINT64_C(0x55aa55aaaa55aa55))) {
            fprintf(stderr, "CUBIN runner modified statistics extension metadata\n");
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_materialize_runner_run(
            materialize_runner,
            test_routines,
            1u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)materialize_output,
            materialize_elements,
            TEST_ROWS,
            TEST_ASTS * TEST_ROWS,
            &stats);
        if (result != SECANT_SUCCESS ||
            stats.num_modules != TEST_RUNNER_MODULES ||
            stats.modules_loaded != TEST_RUNNER_MODULES ||
            stats.num_asts != TEST_RUNNER_ASTS ||
            stats.compile_critical_seconds <= 0.0 ||
            stats.compile_critical_seconds >
                stats.compile_work_seconds) {
            test_print_cubin_error(
                "secant_test_cubin_materialize_runner_run",
                result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            state->sse_handle,
            state->sse_cubin,
            state->sse_cubin_size,
            2u,
            2u,
            &sse_runner);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "secant_test_cubin_runner_create",
                result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_sse_runner_run(
            sse_runner,
            test_routines,
            1u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)sse_output,
            sse_elements,
            TEST_TARGETS,
            &stats);
        if (result != SECANT_SUCCESS ||
            stats.num_modules != TEST_RUNNER_MODULES ||
            stats.modules_loaded != TEST_RUNNER_MODULES ||
            stats.num_asts != TEST_RUNNER_ASTS ||
            stats.compile_critical_seconds <= 0.0 ||
            stats.compile_critical_seconds >
                stats.compile_work_seconds) {
            test_print_cubin_error(
                "secant_test_cubin_sse_runner_run",
                result);
            success = 0;
        }
    }
    if (success &&
        (cuMemcpyDtoH(
             actual_materialize,
             materialize_output,
             materialize_bytes) != CUDA_SUCCESS ||
         cuMemcpyDtoH(
             actual_sse,
             sse_output,
             sse_bytes) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        success = test_compare(
            "bulk runner materialize",
            actual_materialize,
            expected_materialize,
            materialize_elements,
            3.0e-5f);
    }
    if (success) {
        success = test_compare(
            "bulk runner sse",
            actual_sse,
            expected_sse,
            sse_elements,
            2.0e-4f);
    }
    if (sse_runner != NULL &&
        secant_cubin_runner_destroy(sse_runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (secant_cubin_runner_destroy(NULL) != SECANT_SUCCESS) {
        success = 0;
    }
    if (materialize_runner != NULL) {
        if (cuCtxSetCurrent(NULL) != CUDA_SUCCESS) {
            success = 0;
        } else {
            if (secant_cubin_runner_destroy(materialize_runner) !=
                SECANT_ERROR_DRIVER_FAILED) {
                success = 0;
            }
            if (cuCtxSetCurrent(state->context) != CUDA_SUCCESS) {
                success = 0;
            } else if (
                secant_cubin_runner_destroy(materialize_runner) !=
                    SECANT_SUCCESS) {
                success = 0;
            }
        }
    }
    if (sse_output != 0u) {
        (void)cuMemFree(sse_output);
    }
    if (materialize_output != 0u) {
        (void)cuMemFree(materialize_output);
    }
    free(actual_sse);
    free(expected_sse);
    free(actual_materialize);
    free(expected_materialize);
    return success;
}

static int
test_run_partial_bulk_runner(
    TestState* state,
    const float* input,
    const float* targets
) {
    const size_t materialize_elements = TEST_PARTIAL_ASTS * TEST_ROWS;
    const size_t sse_elements = TEST_PARTIAL_ASTS;
    const size_t materialize_bytes = materialize_elements * sizeof(float);
    const size_t sse_bytes = sse_elements * sizeof(float);
    const SecantAstInstruction* asts[TEST_PARTIAL_ASTS];
    float* expected_materialize = NULL;
    float* actual_materialize = NULL;
    float* expected_sse = NULL;
    float* actual_sse = NULL;
    CUdeviceptr materialize_output = 0u;
    CUdeviceptr sse_output = 0u;
    SecantCubinRunner materialize_runner = NULL;
    SecantCubinRunner sse_runner = NULL;
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantResult result;
    size_t ast_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < TEST_PARTIAL_ASTS; ++ast_idx) {
        asts[ast_idx] = test_asts[ast_idx % TEST_ASTS];
    }
    expected_materialize = (float*)malloc(materialize_bytes);
    actual_materialize = (float*)malloc(materialize_bytes);
    expected_sse = (float*)calloc(sse_elements, sizeof(float));
    actual_sse = (float*)malloc(sse_bytes);
    if (expected_materialize == NULL || actual_materialize == NULL || expected_sse == NULL || actual_sse == NULL) {
        success = 0;
    }
    if (success) {
        result = secant_test_cpu_materialize_run(
            TEST_INPUTS,
            test_routines,
            1u,
            asts,
            TEST_PARTIAL_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected_materialize,
            materialize_elements,
            TEST_ROWS);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cpu_sse_run(
            TEST_INPUTS,
            1u,
            test_routines,
            1u,
            asts,
            TEST_PARTIAL_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected_sse,
            sse_elements,
            1u);
        success = result == SECANT_SUCCESS;
    }
    if (success &&
        (cuMemAlloc(&materialize_output, materialize_bytes) != CUDA_SUCCESS ||
         cuMemAlloc(&sse_output, sse_bytes) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            state->materialize_handle,
            state->materialize_cubin,
            state->materialize_cubin_size,
            2u,
            2u,
            &materialize_runner);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_materialize_runner_run(
            materialize_runner,
            test_routines,
            1u,
            asts,
            TEST_PARTIAL_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)materialize_output,
            materialize_elements,
            TEST_ROWS,
            TEST_ASTS * TEST_ROWS,
            &stats);
        success = result == SECANT_SUCCESS && stats.num_modules == 2u && stats.modules_loaded == 2u &&
            stats.num_asts == TEST_PARTIAL_ASTS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            state->sse_handle,
            state->sse_cubin,
            state->sse_cubin_size,
            2u,
            2u,
            &sse_runner);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_sse_runner_run(
            sse_runner,
            test_routines,
            1u,
            asts,
            TEST_PARTIAL_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_TARGETS + 1u,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)sse_output,
            sse_elements,
            1u,
            NULL);
        if (result != SECANT_ERROR_INVALID_VALUE) {
            test_print_cubin_error("partial runner target-capacity rejection", result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_sse_runner_run(
            sse_runner,
            test_routines,
            1u,
            asts,
            TEST_PARTIAL_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            1u,
            (uintptr_t)state->targets,
            TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)sse_output,
            sse_elements,
            1u,
            &stats);
        success = result == SECANT_SUCCESS && stats.num_modules == 2u && stats.modules_loaded == 2u &&
            stats.num_asts == TEST_PARTIAL_ASTS;
    }
    if (success &&
        (cuMemcpyDtoH(actual_materialize, materialize_output, materialize_bytes) != CUDA_SUCCESS ||
         cuMemcpyDtoH(actual_sse, sse_output, sse_bytes) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        success = test_compare(
            "partial bulk runner materialize",
            actual_materialize,
            expected_materialize,
            materialize_elements,
            3.0e-5f);
    }
    if (success) {
        success = test_compare(
            "partial bulk runner SSE",
            actual_sse,
            expected_sse,
            sse_elements,
            2.0e-4f);
    }
    if (sse_runner != NULL && secant_cubin_runner_destroy(sse_runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (materialize_runner != NULL && secant_cubin_runner_destroy(materialize_runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (sse_output != 0u) {
        (void)cuMemFree(sse_output);
    }
    if (materialize_output != 0u) {
        (void)cuMemFree(materialize_output);
    }
    free(actual_sse);
    free(expected_sse);
    free(actual_materialize);
    free(expected_materialize);
    return success;
}

static int
test_run_dynamic_constant_bulk_runner(
    TestState* state,
    int major,
    int minor,
    const float* input,
    const float* targets
) {
    const size_t output_elements =
        TEST_RUNNER_ASTS * TEST_TARGETS * TEST_DYNAMIC_SETTINGS;
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t patch_capacity_instructions =
        TEST_ASTS_PER_KERNEL * TEST_PATCH_INSTRUCTIONS + 13u;
    SecantCubinDynamicConstantSSERecipe recipe = secant_cubin_dynamic_constant_sse_recipe_init();
    const SecantAstInstruction* runner_asts[TEST_RUNNER_ASTS];
    float constants[TEST_DYNAMIC_CONSTANTS * TEST_DYNAMIC_SETTINGS];
    float* expected = NULL;
    float* actual = NULL;
    unsigned char* cubin = NULL;
    void* plan_storage = NULL;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdeviceptr device_constants = 0u;
    CUdeviceptr device_output = 0u;
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantResult result;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    size_t ast_idx;
    size_t setting;
    int success = 1;

    recipe.num_kernels = TEST_KERNELS;
    recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    recipe.num_input_columns = TEST_INPUTS;
    recipe.num_input_constants = TEST_DYNAMIC_CONSTANTS;
    recipe.num_targets = TEST_TARGETS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    for (ast_idx = 0u; ast_idx < TEST_RUNNER_ASTS; ++ast_idx) {
        runner_asts[ast_idx] = ast_idx % 2u == 0u
            ? test_dynamic_add
            : test_dynamic_sin_mul;
    }
    for (setting = 0u;
         setting < TEST_DYNAMIC_SETTINGS;
         ++setting) {
        constants[setting] = -0.4f + 0.1f * (float)setting;
        constants[TEST_DYNAMIC_SETTINGS + setting] =
            0.75f + 0.05f * (float)setting;
    }
    expected = (float*)calloc(output_elements, sizeof(float));
    actual = (float*)malloc(output_bytes);
    if (expected == NULL || actual == NULL) {
        success = 0;
    }
    if (success) {
        result = secant_test_cpu_dynamic_constant_sse_run(
            TEST_INPUTS,
            TEST_DYNAMIC_CONSTANTS,
            TEST_TARGETS,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            constants,
            TEST_DYNAMIC_CONSTANTS * TEST_DYNAMIC_SETTINGS,
            TEST_DYNAMIC_SETTINGS,
            TEST_DYNAMIC_SETTINGS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            output_elements,
            TEST_DYNAMIC_SETTINGS);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "dynamic CPU reference",
                result);
            success = 0;
        }
    }
    if (success) {
        success = test_compile_dynamic_cubin_template(
            &recipe,
            major,
            minor,
            &cubin,
            &cubin_size);
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            NULL,
            0u,
            &plan_storage_size,
            &plan);
        if (result != SECANT_SUCCESS || plan_storage_size == 0u) {
            test_print_cubin_error(
                "dynamic inspect measure",
                result);
            success = 0;
        }
    }
    if (success) {
        plan_storage = malloc(plan_storage_size);
        if (plan_storage == NULL) {
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            plan_storage,
            plan_storage_size,
            &plan_storage_size,
            &plan);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "dynamic inspect",
                result);
            success = 0;
        }
    }
    if (success &&
        (cuMemAlloc(
             &device_constants,
             sizeof(constants)) != CUDA_SUCCESS ||
         cuMemAlloc(
             &device_output,
             output_bytes) != CUDA_SUCCESS ||
         cuMemcpyHtoD(
             device_constants,
             constants,
             sizeof(constants)) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            plan,
            cubin,
            cubin_size,
            2u,
            2u,
            &runner);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error(
                "dynamic runner create",
                result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_dynamic_constant_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_constants,
            TEST_DYNAMIC_CONSTANTS * TEST_DYNAMIC_SETTINGS,
            TEST_DYNAMIC_SETTINGS,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            &stats);
        if (result != SECANT_SUCCESS ||
            stats.num_modules != TEST_RUNNER_MODULES ||
            stats.modules_loaded != TEST_RUNNER_MODULES ||
            stats.num_asts != TEST_RUNNER_ASTS) {
            test_print_cubin_error(
                "dynamic runner run all",
                result);
            success = 0;
        }
    }
    if (success &&
        cuMemcpyDtoH(
            actual,
            device_output,
            output_bytes) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        success = test_compare(
            "dynamic constant bulk runner",
            actual,
            expected,
            output_elements,
            4.0e-4f);
    }
    if (runner != NULL &&
        secant_cubin_runner_destroy(runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (device_output != 0u) {
        (void)cuMemFree(device_output);
    }
    if (device_constants != 0u) {
        (void)cuMemFree(device_constants);
    }
    free(plan_storage);
    free(cubin);
    free(actual);
    free(expected);
    return success;
}

static int
test_run_dynamic_leaf_bulk_runner(
    TestState* state,
    int major,
    int minor,
    const float* input,
    const float* targets,
    int mixed
) {
    const size_t output_elements = TEST_RUNNER_ASTS * TEST_TARGETS * TEST_DYNAMIC_SETTINGS;
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t patch_capacity_instructions = TEST_ASTS_PER_KERNEL * TEST_PATCH_INSTRUCTIONS + 13u;
    SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    const SecantAstInstruction* runner_asts[TEST_RUNNER_ASTS];
    SecantAstInstruction oversized_program[1000];
    uint32_t leaf_masks[TEST_DYNAMIC_SETTINGS];
    uint32_t leaf_words[TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES];
    float* expected = NULL;
    float* actual = NULL;
    unsigned char* cubin = NULL;
    void* plan_storage = NULL;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdeviceptr device_leaf_masks = 0u;
    CUdeviceptr device_leaf_words = 0u;
    CUdeviceptr device_output = 0u;
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantResult result;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    size_t ast_idx;
    size_t setting;
    int success = 1;

    recipe.num_kernels = TEST_KERNELS;
    recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    recipe.num_input_columns = TEST_INPUTS;
    recipe.num_static_input_columns = mixed ? TEST_INPUTS : 0u;
    recipe.num_dynamic_leaves = TEST_DYNAMIC_LEAVES;
    recipe.num_targets = TEST_TARGETS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    {
        SecantCubinDynamicLeafSSERecipe invalid_recipe = recipe;

        invalid_recipe.num_static_input_columns = 1u;
        if (secant_cubin_recipe_validate(&invalid_recipe.header) != SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "dynamic leaf accepted a partial static-column bank\n");
            return 0;
        }
    }
    memset(leaf_words, 0, sizeof(leaf_words));
    oversized_program[0] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
    oversized_program[1] = 0u;
    for (ast_idx = 2u; ast_idx + 1u < sizeof(oversized_program); ++ast_idx) {
        oversized_program[ast_idx] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_NEG_F32;
    }
    oversized_program[sizeof(oversized_program) - 1u] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
    for (ast_idx = 0u; ast_idx < TEST_RUNNER_ASTS; ++ast_idx) {
        switch (ast_idx % (mixed ? 4u : 3u)) {
            case 0u:
                runner_asts[ast_idx] = mixed ? test_mixed_leaf_add : test_dynamic_leaf_identity;
                break;
            case 1u:
                runner_asts[ast_idx] = test_dynamic_leaf_add;
                break;
            default:
                runner_asts[ast_idx] = test_dynamic_leaf_sin_mul;
                break;
        }
    }
    for (setting = 0u; setting < TEST_DYNAMIC_SETTINGS; ++setting) {
        const float constant1 = -0.35f + 0.08f * (float)setting;
        const float flexible2 = 0.45f + 0.03f * (float)setting;

        leaf_masks[setting] = (1u << 0u) | (1u << 3u) |
            (setting % 2u == 0u ? (1u << 2u) : 0u);
        leaf_words[setting * TEST_DYNAMIC_LEAVES + 0u] = 0u;
        memcpy(leaf_words + setting * TEST_DYNAMIC_LEAVES + 1u, &constant1, sizeof(constant1));
        if ((leaf_masks[setting] & (1u << 2u)) != 0u) {
            leaf_words[setting * TEST_DYNAMIC_LEAVES + 2u] = 2u;
        } else {
            memcpy(leaf_words + setting * TEST_DYNAMIC_LEAVES + 2u, &flexible2, sizeof(flexible2));
        }
        leaf_words[setting * TEST_DYNAMIC_LEAVES + 3u] = 1u;
    }
    expected = (float*)calloc(output_elements, sizeof(float));
    actual = (float*)malloc(output_bytes);
    if (expected == NULL || actual == NULL) {
        success = 0;
    }
    if (success) {
        result = secant_test_cpu_dynamic_leaf_sse_run(
            TEST_DYNAMIC_LEAVES,
            TEST_INPUTS,
            mixed ? TEST_INPUTS : 0u,
            TEST_TARGETS,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            output_elements,
            TEST_DYNAMIC_SETTINGS);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("dynamic leaf CPU reference", result);
            success = 0;
        }
    }
    if (success) {
        success = test_compile_dynamic_leaf_cubin_template(
            &recipe,
            major,
            minor,
            &cubin,
            &cubin_size);
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            NULL,
            0u,
            &plan_storage_size,
            &plan);
        if (result != SECANT_SUCCESS || plan_storage_size == 0u) {
            test_print_cubin_error("dynamic leaf inspect measure", result);
            success = 0;
        }
    }
    if (success) {
        plan_storage = malloc(plan_storage_size);
        if (plan_storage == NULL) {
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            plan_storage,
            plan_storage_size,
            &plan_storage_size,
            &plan);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("dynamic leaf inspect", result);
            success = 0;
        }
    }
    if (success) {
        SecantCubinDynamicLeafSSERecipe recovered_recipe = secant_cubin_dynamic_leaf_sse_recipe_init();

        result = secant_cubin_plan_recipe_get(plan, &recovered_recipe.header);
        if (result != SECANT_SUCCESS || recovered_recipe.num_input_columns != recipe.num_input_columns ||
            recovered_recipe.num_static_input_columns != recipe.num_static_input_columns ||
            recovered_recipe.num_dynamic_leaves != recipe.num_dynamic_leaves) {
            test_print_cubin_error("dynamic leaf recipe round trip", result);
            success = 0;
        }
    }
    if (success &&
        (cuMemAlloc(&device_leaf_masks, sizeof(leaf_masks)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_leaf_words, sizeof(leaf_words)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_output, output_bytes) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device_leaf_masks, leaf_masks, sizeof(leaf_masks)) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device_leaf_words, leaf_words, sizeof(leaf_words)) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            plan,
            cubin,
            cubin_size,
            2u,
            2u,
            &runner);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("dynamic leaf runner create", result);
            success = 0;
        }
    }
    if (success && mixed) {
        const SecantAstInstruction inactive_static_program[] = {
            (uint8_t)SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
            (uint8_t)(TEST_INPUTS - 1u),
            (uint8_t)SECANT_AST_INSTRUCTION_TYPE_RETURN_F32
        };
        const SecantAstInstruction* saved = runner_asts[0];

        result = secant_test_cubin_dynamic_leaf_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_INPUTS - 1u,
            TEST_ROWS,
            (uintptr_t)device_leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            (uintptr_t)device_leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            NULL);
        if (result != SECANT_SUCCESS ||
            cuMemcpyDtoH(actual, device_output, output_bytes) != CUDA_SUCCESS ||
            !test_compare("mixed leaf active column prefix", actual, expected, output_elements, 4.0e-4f)) {
            test_print_cubin_error("mixed leaf active column prefix", result);
            success = 0;
        }
        runner_asts[0] = inactive_static_program;
        result = secant_test_cubin_dynamic_leaf_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_INPUTS - 1u,
            TEST_ROWS,
            (uintptr_t)device_leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            (uintptr_t)device_leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            NULL);
        runner_asts[0] = saved;
        if (result != SECANT_ERROR_BAD_PROGRAM) {
            test_print_cubin_error("mixed leaf inactive static input rejection", result);
            success = 0;
        }
        result = secant_test_cubin_dynamic_leaf_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_INPUTS + 1u,
            TEST_ROWS,
            (uintptr_t)device_leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            (uintptr_t)device_leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            NULL);
        if (result != SECANT_ERROR_INVALID_VALUE) {
            test_print_cubin_error("mixed leaf column capacity rejection", result);
            success = 0;
        }
    }
    if (success && !mixed) {
        const SecantAstInstruction* saved = runner_asts[0];

        runner_asts[0] = test_add;
        result = secant_test_cubin_dynamic_leaf_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_INPUTS,
            TEST_ROWS,
            (uintptr_t)device_leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            (uintptr_t)device_leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            NULL);
        runner_asts[0] = saved;
        if (result != SECANT_ERROR_BAD_PROGRAM) {
            test_print_cubin_error("dynamic leaf static-input rejection", result);
            success = 0;
        }
    }
    if (success) {
        const SecantAstInstruction* saved = runner_asts[0];

        runner_asts[0] = oversized_program;
        result = secant_test_cubin_dynamic_leaf_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_INPUTS,
            TEST_ROWS,
            (uintptr_t)device_leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            (uintptr_t)device_leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            NULL);
        runner_asts[0] = saved;
        if (result != SECANT_ERROR_INSUFFICIENT_PATCH_SPACE) {
            test_print_cubin_error("dynamic leaf insufficient patch space", result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_dynamic_leaf_sse_runner_run(
            runner,
            NULL,
            0u,
            runner_asts,
            TEST_RUNNER_ASTS,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_INPUTS,
            TEST_ROWS,
            (uintptr_t)device_leaf_masks,
            TEST_DYNAMIC_SETTINGS,
            (uintptr_t)device_leaf_words,
            TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_SETTINGS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            TEST_DYNAMIC_SETTINGS,
            &stats);
        if (result != SECANT_SUCCESS || stats.num_modules != TEST_RUNNER_MODULES ||
            stats.modules_loaded != TEST_RUNNER_MODULES || stats.num_asts != TEST_RUNNER_ASTS) {
            test_print_cubin_error("dynamic leaf runner run all", result);
            success = 0;
        }
    }
    if (success && cuMemcpyDtoH(actual, device_output, output_bytes) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        success = test_compare("dynamic leaf bulk runner", actual, expected, output_elements, 4.0e-4f);
    }
    if (runner != NULL && secant_cubin_runner_destroy(runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (device_output != 0u) {
        (void)cuMemFree(device_output);
    }
    if (device_leaf_words != 0u) {
        (void)cuMemFree(device_leaf_words);
    }
    if (device_leaf_masks != 0u) {
        (void)cuMemFree(device_leaf_masks);
    }
    free(plan_storage);
    free(cubin);
    free(actual);
    free(expected);
    return success;
}

static SecantResult
test_matrix_runner_run(
    TestMatrixShape shape,
    SecantCubinRunner runner,
    TestState* state,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    size_t num_targets,
    CUdeviceptr device_constants,
    CUdeviceptr device_leaf_masks,
    CUdeviceptr device_leaf_words,
    CUdeviceptr device_output,
    size_t output_num_elements,
    SecantRunnerStats* stats_ret
) {
    switch (shape) {
        case TEST_MATRIX_SHAPE_MATERIALIZE:
            return secant_test_cubin_materialize_runner_run(
                runner,
                NULL,
                0u,
                asts,
                num_asts,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                output_num_elements,
                TEST_ROWS,
                TEST_ASTS * TEST_ROWS,
                stats_ret);
        case TEST_MATRIX_SHAPE_SSE:
            return secant_test_cubin_sse_runner_run(
                runner,
                NULL,
                0u,
                asts,
                num_asts,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                num_targets,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                output_num_elements,
                num_targets,
                stats_ret);
        case TEST_MATRIX_SHAPE_AFFINE_STATS:
            return secant_test_cubin_affine_stats_runner_run(
                runner,
                NULL,
                0u,
                asts,
                num_asts,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                num_targets,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                output_num_elements,
                SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + num_targets,
                stats_ret);
        case TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE:
            return secant_test_cubin_dynamic_constant_sse_runner_run(
                runner,
                NULL,
                0u,
                asts,
                num_asts,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_constants,
                TEST_DYNAMIC_CONSTANTS * TEST_DYNAMIC_SETTINGS,
                TEST_DYNAMIC_SETTINGS,
                TEST_DYNAMIC_SETTINGS,
                num_targets,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                output_num_elements,
                TEST_DYNAMIC_SETTINGS,
                stats_ret);
        case TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE:
            return secant_test_cubin_dynamic_leaf_sse_runner_run(
                runner,
                NULL,
                0u,
                asts,
                num_asts,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_INPUTS,
                TEST_ROWS,
                (uintptr_t)device_leaf_masks,
                TEST_DYNAMIC_SETTINGS,
                (uintptr_t)device_leaf_words,
                TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
                TEST_DYNAMIC_LEAVES,
                TEST_DYNAMIC_SETTINGS,
                num_targets,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                output_num_elements,
                TEST_DYNAMIC_SETTINGS,
                stats_ret);
        default:
            return SECANT_ERROR_INVALID_VALUE;
    }
}

static SecantResult
test_matrix_cpu_run(
    TestMatrixShape shape,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    size_t num_targets,
    const float* input,
    const float* targets,
    const float* constants,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    float* output,
    size_t output_num_elements
) {
    switch (shape) {
        case TEST_MATRIX_SHAPE_MATERIALIZE:
            return secant_test_cpu_materialize_run(
                TEST_INPUTS,
                NULL,
                0u,
                asts,
                num_asts,
                input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                output,
                output_num_elements,
                TEST_ROWS);
        case TEST_MATRIX_SHAPE_SSE:
            return secant_test_cpu_sse_run(
                TEST_INPUTS,
                num_targets,
                NULL,
                0u,
                asts,
                num_asts,
                input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                output,
                output_num_elements,
                num_targets);
        case TEST_MATRIX_SHAPE_AFFINE_STATS:
            return secant_test_cpu_affine_stats_run(
                TEST_INPUTS,
                num_targets,
                NULL,
                0u,
                asts,
                num_asts,
                input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                output,
                output_num_elements,
                SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + num_targets);
        case TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE:
            return secant_test_cpu_dynamic_constant_sse_run(
                TEST_INPUTS,
                TEST_DYNAMIC_CONSTANTS,
                num_targets,
                NULL,
                0u,
                asts,
                num_asts,
                input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                constants,
                TEST_DYNAMIC_CONSTANTS * TEST_DYNAMIC_SETTINGS,
                TEST_DYNAMIC_SETTINGS,
                TEST_DYNAMIC_SETTINGS,
                targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                output,
                output_num_elements,
                TEST_DYNAMIC_SETTINGS);
        case TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE:
            return secant_test_cpu_dynamic_leaf_sse_run(
                TEST_DYNAMIC_LEAVES,
                TEST_INPUTS,
                0u,
                num_targets,
                NULL,
                0u,
                asts,
                num_asts,
                input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                leaf_masks,
                TEST_DYNAMIC_SETTINGS,
                leaf_words,
                TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES,
                TEST_DYNAMIC_LEAVES,
                TEST_DYNAMIC_SETTINGS,
                targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                output,
                output_num_elements,
                TEST_DYNAMIC_SETTINGS);
        default:
            return SECANT_ERROR_INVALID_VALUE;
    }
}

static int
test_sse_trajectory_batch(
    TestState* state,
    SecantCubinRunner runner,
    const SecantAstInstruction* const* asts,
    const float* input,
    const float* targets
) {
    const size_t num_asts = TEST_MATRIX_ASTS;
    const size_t num_targets = TEST_TARGETS;
    const size_t num_output_elements = num_asts * num_targets;
    const size_t allocation_elements = num_output_elements + 2u * TEST_MATRIX_GUARD_ELEMENTS;
    const size_t num_rows[2] = { TEST_ROWS, 113u };
    const float weights[2] = { 0.25f, 1.75f };
    SecantCubinSSERun trajectory_runs[2];
    const SecantCubinRunHeader* run_headers[2];
    float expected[2][TEST_MATRIX_ASTS * TEST_TARGETS];
    float actual[2][TEST_MATRIX_ASTS * TEST_TARGETS];
    float expected_weighted[TEST_MATRIX_ASTS * TEST_TARGETS];
    float actual_weighted[TEST_MATRIX_ASTS * TEST_TARGETS];
    CUdeviceptr allocations[2] = { 0u, 0u };
    CUdeviceptr outputs[2] = { 0u, 0u };
    CUdeviceptr trajectory_input = 0u;
    CUdeviceptr trajectory_targets = 0u;
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantResult result = SECANT_SUCCESS;
    size_t trajectory_idx;
    size_t output_idx;
    int success = 1;

    memset(expected, 0, sizeof(expected));
    memset(actual, 0, sizeof(actual));
    if (cuMemAlloc(&trajectory_input, TEST_INPUTS * TEST_ROWS * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&trajectory_targets, TEST_TARGETS * TEST_ROWS * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(trajectory_input, input, TEST_INPUTS * TEST_ROWS * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(trajectory_targets, targets, TEST_TARGETS * TEST_ROWS * sizeof(float)) != CUDA_SUCCESS) {
        success = 0;
    }
    for (trajectory_idx = 0u; trajectory_idx < 2u && success; ++trajectory_idx) {
        trajectory_runs[trajectory_idx] = secant_cubin_sse_run_init();
        trajectory_runs[trajectory_idx].programs.routines.items = NULL;
        trajectory_runs[trajectory_idx].programs.routines.count = 0u;
        trajectory_runs[trajectory_idx].programs.asts.items = asts;
        trajectory_runs[trajectory_idx].programs.asts.count = num_asts;
        trajectory_runs[trajectory_idx].input.address = trajectory_idx == 0u
            ? (uintptr_t)state->input
            : (uintptr_t)trajectory_input;
        trajectory_runs[trajectory_idx].input.num_elements = TEST_INPUTS * TEST_ROWS;
        trajectory_runs[trajectory_idx].input.leading_dimension = TEST_ROWS;
        trajectory_runs[trajectory_idx].targets.address = trajectory_idx == 0u
            ? (uintptr_t)state->targets
            : (uintptr_t)trajectory_targets;
        trajectory_runs[trajectory_idx].targets.num_elements = TEST_TARGETS * TEST_ROWS;
        trajectory_runs[trajectory_idx].targets.leading_dimension = TEST_ROWS;
        trajectory_runs[trajectory_idx].num_rows = num_rows[trajectory_idx];
        trajectory_runs[trajectory_idx].num_targets = num_targets;
        run_headers[trajectory_idx] = &trajectory_runs[trajectory_idx].header;
        result = secant_test_cpu_sse_run(
            TEST_INPUTS,
            num_targets,
            NULL,
            0u,
            asts,
            num_asts,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            num_rows[trajectory_idx],
            expected[trajectory_idx],
            num_output_elements,
            num_targets);
        if (result != SECANT_SUCCESS ||
            cuMemAlloc(&allocations[trajectory_idx], allocation_elements * sizeof(float)) != CUDA_SUCCESS) {
            success = 0;
            break;
        }
        outputs[trajectory_idx] = allocations[trajectory_idx] + TEST_MATRIX_GUARD_ELEMENTS * sizeof(float);
        trajectory_runs[trajectory_idx].output.address = (uintptr_t)outputs[trajectory_idx];
        trajectory_runs[trajectory_idx].output.num_elements = num_output_elements;
        trajectory_runs[trajectory_idx].output.leading_dimension = num_targets;
        if (!test_guarded_output_reset(allocations[trajectory_idx], allocation_elements)) {
            success = 0;
            break;
        }
    }
    if (success) {
        result = secant_cubin_runner_run_batch(runner, run_headers, 2u, &stats);
        success = result == SECANT_SUCCESS && stats.num_modules == 2u && stats.modules_loaded == 2u &&
            stats.num_asts == num_asts;
        if (!success) {
            test_print_cubin_error("SSE trajectory batch", result);
        }
    }
    for (trajectory_idx = 0u; trajectory_idx < 2u && success; ++trajectory_idx) {
        success = test_guarded_output_read(
                "SSE trajectory batch",
                allocations[trajectory_idx],
                allocation_elements,
                num_output_elements,
                actual[trajectory_idx]) &&
            test_compare(
                "SSE trajectory batch",
                actual[trajectory_idx],
                expected[trajectory_idx],
                num_output_elements,
                2.0e-4f);
    }
    for (output_idx = 0u; output_idx < num_output_elements && success; ++output_idx) {
        expected_weighted[output_idx] =
            weights[0] * expected[0][output_idx] + weights[1] * expected[1][output_idx];
        actual_weighted[output_idx] = weights[0] * actual[0][output_idx] + weights[1] * actual[1][output_idx];
    }
    if (success) {
        success = test_compare(
            "weighted SSE trajectory reduction",
            actual_weighted,
            expected_weighted,
            num_output_elements,
            2.0e-4f);
    }
    if (success) {
        trajectory_runs[1].output.address = trajectory_runs[0].output.address;
        stats = secant_runner_stats_init();
        result = secant_cubin_runner_run_batch(runner, run_headers, 2u, &stats);
        success = result == SECANT_ERROR_INVALID_VALUE;
        if (!success) {
            fprintf(stderr, "SSE trajectory batch accepted overlapping output bins\n");
        }
    }
    if (success) {
        trajectory_runs[1].output.address = trajectory_runs[0].input.address;
        stats = secant_runner_stats_init();
        result = secant_cubin_runner_run_batch(runner, run_headers, 2u, &stats);
        success = result == SECANT_ERROR_INVALID_VALUE;
        if (!success) {
            fprintf(stderr, "SSE trajectory batch accepted output overlapping another run's input\n");
        }
    }
    if (success) {
        trajectory_runs[1].output.address = (uintptr_t)outputs[1];
        trajectory_runs[0].output.address = trajectory_runs[1].targets.address;
        stats = secant_runner_stats_init();
        result = secant_cubin_runner_run_batch(runner, run_headers, 2u, &stats);
        success = result == SECANT_ERROR_INVALID_VALUE;
        if (!success) {
            fprintf(stderr, "SSE trajectory batch accepted output overlapping another run's targets\n");
        }
        trajectory_runs[0].output.address = (uintptr_t)outputs[0];
    }
    for (trajectory_idx = 0u; trajectory_idx < 2u; ++trajectory_idx) {
        if (allocations[trajectory_idx] != 0u && cuMemFree(allocations[trajectory_idx]) != CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (trajectory_targets != 0u && cuMemFree(trajectory_targets) != CUDA_SUCCESS) {
        success = 0;
    }
    if (trajectory_input != 0u && cuMemFree(trajectory_input) != CUDA_SUCCESS) {
        success = 0;
    }
    return success;
}

static int
test_run_kernel_shape_matrix(
    TestState* state,
    int major,
    int minor,
    const float* input,
    const float* targets
) {
    static const size_t ast_counts[] = { 1u, 4u, 5u, 6u, 9u, 10u, 11u, 13u, 10u, 6u };
    const size_t max_output_elements = TEST_MATRIX_ASTS * TEST_ROWS;
    const size_t output_allocation_elements = max_output_elements + 2u * TEST_MATRIX_GUARD_ELEMENTS;
    const size_t output_num_elements = max_output_elements + TEST_MATRIX_GUARD_ELEMENTS;
    const size_t max_target_stats_elements = TEST_TARGETS * SECANT_AFFINE_TARGET_STAT_COUNT_F32;
    const size_t target_stats_allocation_elements = max_target_stats_elements + 2u * TEST_MATRIX_GUARD_ELEMENTS;
    const size_t patch_capacity_instructions = TEST_ASTS_PER_KERNEL * TEST_PATCH_INSTRUCTIONS + 13u;
    SecantCubinDynamicConstantSSERecipe dynamic_constant_recipe =
        secant_cubin_dynamic_constant_sse_recipe_init();
    SecantCubinDynamicLeafSSERecipe dynamic_leaf_recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    SecantCubinAffineStatsRecipe affine_stats_recipe = secant_cubin_affine_stats_recipe_init();
    SecantAstInstruction static_programs[TEST_INPUTS][3];
    SecantAstInstruction dynamic_constant_programs[TEST_INPUTS + TEST_DYNAMIC_CONSTANTS][3];
    SecantAstInstruction dynamic_leaf_programs[TEST_DYNAMIC_LEAVES][3];
    SecantAstInstruction invalid_programs[TEST_MATRIX_SHAPE_COUNT][3];
    const SecantAstInstruction* invalid_asts[TEST_MATRIX_SHAPE_COUNT];
    const SecantAstInstruction* shape_asts[TEST_MATRIX_SHAPE_COUNT][TEST_MATRIX_ASTS];
    float constants[TEST_DYNAMIC_CONSTANTS * TEST_DYNAMIC_SETTINGS];
    uint32_t leaf_masks[TEST_DYNAMIC_SETTINGS];
    uint32_t leaf_words[TEST_DYNAMIC_SETTINGS * TEST_DYNAMIC_LEAVES];
    float* expected = NULL;
    float* actual = NULL;
    float expected_target_stats[TEST_TARGETS * SECANT_AFFINE_TARGET_STAT_COUNT_F32];
    float actual_target_stats[TEST_TARGETS * SECANT_AFFINE_TARGET_STAT_COUNT_F32];
    float expected_affine_sse[TEST_MATRIX_ASTS * TEST_TARGETS];
    float actual_affine_sse[TEST_MATRIX_ASTS * TEST_TARGETS];
    unsigned char* dynamic_constant_cubin = NULL;
    unsigned char* dynamic_leaf_cubin = NULL;
    unsigned char* affine_stats_cubin = NULL;
    void* dynamic_constant_plan_storage = NULL;
    void* dynamic_leaf_plan_storage = NULL;
    void* affine_stats_plan_storage = NULL;
    SecantCubinPlan* dynamic_constant_plan = NULL;
    SecantCubinPlan* dynamic_leaf_plan = NULL;
    SecantCubinPlan* affine_stats_plan = NULL;
    SecantCubinRunner runners[TEST_MATRIX_SHAPE_COUNT] = { NULL };
    CUmodule target_stats_module = NULL;
    CUfunction target_stats_function = NULL;
    CUdeviceptr device_constants = 0u;
    CUdeviceptr device_leaf_masks = 0u;
    CUdeviceptr device_leaf_words = 0u;
    CUdeviceptr device_output_allocation = 0u;
    CUdeviceptr device_output;
    CUdeviceptr device_target_stats_allocation = 0u;
    CUdeviceptr device_target_stats;
    SecantResult result;
    size_t dynamic_constant_cubin_size = 0u;
    size_t dynamic_leaf_cubin_size = 0u;
    size_t affine_stats_cubin_size = 0u;
    size_t dynamic_constant_plan_storage_size = 0u;
    size_t dynamic_leaf_plan_storage_size = 0u;
    size_t affine_stats_plan_storage_size = 0u;
    size_t shape_idx;
    size_t ast_idx;
    size_t input_idx;
    size_t setting_idx;
    int success = 1;

    dynamic_constant_recipe.num_kernels = TEST_KERNELS;
    dynamic_constant_recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    dynamic_constant_recipe.num_input_columns = TEST_INPUTS;
    dynamic_constant_recipe.num_input_constants = TEST_DYNAMIC_CONSTANTS;
    dynamic_constant_recipe.num_targets = TEST_TARGETS;
    dynamic_constant_recipe.tile_rows = 128u;
    dynamic_constant_recipe.threads_per_block = 128u;
    dynamic_constant_recipe.patch_capacity_instructions = patch_capacity_instructions;
    dynamic_leaf_recipe.num_kernels = TEST_KERNELS;
    dynamic_leaf_recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    dynamic_leaf_recipe.num_input_columns = TEST_INPUTS;
    dynamic_leaf_recipe.num_static_input_columns = 0u;
    dynamic_leaf_recipe.num_dynamic_leaves = TEST_DYNAMIC_LEAVES;
    dynamic_leaf_recipe.num_targets = TEST_TARGETS;
    dynamic_leaf_recipe.tile_rows = 128u;
    dynamic_leaf_recipe.threads_per_block = 128u;
    dynamic_leaf_recipe.patch_capacity_instructions = patch_capacity_instructions;
    affine_stats_recipe.num_kernels = TEST_KERNELS;
    affine_stats_recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    affine_stats_recipe.num_inputs = TEST_INPUTS;
    affine_stats_recipe.num_targets = TEST_TARGETS;
    affine_stats_recipe.tile_rows = 128u;
    affine_stats_recipe.threads_per_block = 128u;
    affine_stats_recipe.patch_capacity_instructions = patch_capacity_instructions;
    memset(shape_asts, 0, sizeof(shape_asts));
    for (input_idx = 0u; input_idx < TEST_INPUTS; ++input_idx) {
        test_identity_program_set(
            static_programs[input_idx],
            SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
            input_idx);
        test_identity_program_set(
            dynamic_constant_programs[input_idx],
            SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
            input_idx);
    }
    for (input_idx = 0u; input_idx < TEST_DYNAMIC_CONSTANTS; ++input_idx) {
        test_identity_program_set(
            dynamic_constant_programs[TEST_INPUTS + input_idx],
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
            input_idx);
    }
    for (input_idx = 0u; input_idx < TEST_DYNAMIC_LEAVES; ++input_idx) {
        test_identity_program_set(
            dynamic_leaf_programs[input_idx],
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
            input_idx);
    }
    test_identity_program_set(
        invalid_programs[TEST_MATRIX_SHAPE_MATERIALIZE],
        SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
        TEST_INPUTS);
    test_identity_program_set(
        invalid_programs[TEST_MATRIX_SHAPE_SSE],
        SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
        TEST_INPUTS);
    test_identity_program_set(
        invalid_programs[TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE],
        SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
        TEST_DYNAMIC_CONSTANTS);
    test_identity_program_set(
        invalid_programs[TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE],
        SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
        TEST_DYNAMIC_LEAVES);
    test_identity_program_set(
        invalid_programs[TEST_MATRIX_SHAPE_AFFINE_STATS],
        SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
        TEST_INPUTS);
    for (shape_idx = 0u; shape_idx < TEST_MATRIX_SHAPE_COUNT; ++shape_idx) {
        invalid_asts[shape_idx] = invalid_programs[shape_idx];
    }
    for (ast_idx = 0u; ast_idx < TEST_MATRIX_ASTS; ++ast_idx) {
        shape_asts[TEST_MATRIX_SHAPE_MATERIALIZE][ast_idx] = static_programs[ast_idx % TEST_INPUTS];
        shape_asts[TEST_MATRIX_SHAPE_SSE][ast_idx] = static_programs[ast_idx % TEST_INPUTS];
        shape_asts[TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE][ast_idx] =
            dynamic_constant_programs[ast_idx % (TEST_INPUTS + TEST_DYNAMIC_CONSTANTS)];
        shape_asts[TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE][ast_idx] =
            dynamic_leaf_programs[ast_idx % TEST_DYNAMIC_LEAVES];
        shape_asts[TEST_MATRIX_SHAPE_AFFINE_STATS][ast_idx] =
            test_affine_asts[ast_idx % (sizeof(test_affine_asts) / sizeof(test_affine_asts[0]))];
    }
    for (setting_idx = 0u; setting_idx < TEST_DYNAMIC_SETTINGS; ++setting_idx) {
        constants[setting_idx] = -1.25f + 0.17f * (float)setting_idx;
        constants[TEST_DYNAMIC_SETTINGS + setting_idx] = 0.35f + 0.11f * (float)setting_idx;
        leaf_masks[setting_idx] = 0u;
        for (input_idx = 0u; input_idx < TEST_DYNAMIC_LEAVES; ++input_idx) {
            uint32_t* word = leaf_words + setting_idx * TEST_DYNAMIC_LEAVES + input_idx;

            if ((setting_idx + input_idx) % 3u != 0u) {
                leaf_masks[setting_idx] |= 1u << input_idx;
                *word = (uint32_t)((setting_idx + 3u * input_idx) % TEST_INPUTS);
            } else {
                const float value = -0.9f + 0.07f * (float)(setting_idx * TEST_DYNAMIC_LEAVES + input_idx);

                memcpy(word, &value, sizeof(value));
            }
        }
    }
    expected = (float*)calloc(max_output_elements, sizeof(*expected));
    actual = (float*)malloc(max_output_elements * sizeof(*actual));
    if (expected == NULL || actual == NULL) {
        success = 0;
    }
    if (success) {
        success = test_compile_dynamic_cubin_template(
            &dynamic_constant_recipe,
            major,
            minor,
            &dynamic_constant_cubin,
            &dynamic_constant_cubin_size);
    }
    if (success) {
        success = test_compile_dynamic_leaf_cubin_template(
            &dynamic_leaf_recipe,
            major,
            minor,
            &dynamic_leaf_cubin,
            &dynamic_leaf_cubin_size);
    }
    if (success) {
        success = test_compile_affine_stats_cubin_template(
            &affine_stats_recipe,
            major,
            minor,
            &affine_stats_cubin,
            &affine_stats_cubin_size);
    }
    if (success) {
        success = test_target_stats_module_create(
            major,
            minor,
            &target_stats_module,
            &target_stats_function);
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &dynamic_constant_recipe,
            dynamic_constant_cubin,
            dynamic_constant_cubin_size,
            NULL,
            0u,
            &dynamic_constant_plan_storage_size,
            &dynamic_constant_plan);
        success = result == SECANT_SUCCESS && dynamic_constant_plan_storage_size != 0u;
    }
    if (success) {
        dynamic_constant_plan_storage = malloc(dynamic_constant_plan_storage_size);
        success = dynamic_constant_plan_storage != NULL;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &dynamic_constant_recipe,
            dynamic_constant_cubin,
            dynamic_constant_cubin_size,
            dynamic_constant_plan_storage,
            dynamic_constant_plan_storage_size,
            &dynamic_constant_plan_storage_size,
            &dynamic_constant_plan);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &affine_stats_recipe,
            affine_stats_cubin,
            affine_stats_cubin_size,
            NULL,
            0u,
            &affine_stats_plan_storage_size,
            &affine_stats_plan);
        success = result == SECANT_SUCCESS && affine_stats_plan_storage_size != 0u;
    }
    if (success) {
        affine_stats_plan_storage = malloc(affine_stats_plan_storage_size);
        success = affine_stats_plan_storage != NULL;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &affine_stats_recipe,
            affine_stats_cubin,
            affine_stats_cubin_size,
            affine_stats_plan_storage,
            affine_stats_plan_storage_size,
            &affine_stats_plan_storage_size,
            &affine_stats_plan);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &dynamic_leaf_recipe,
            dynamic_leaf_cubin,
            dynamic_leaf_cubin_size,
            NULL,
            0u,
            &dynamic_leaf_plan_storage_size,
            &dynamic_leaf_plan);
        success = result == SECANT_SUCCESS && dynamic_leaf_plan_storage_size != 0u;
    }
    if (success) {
        dynamic_leaf_plan_storage = malloc(dynamic_leaf_plan_storage_size);
        success = dynamic_leaf_plan_storage != NULL;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &dynamic_leaf_recipe,
            dynamic_leaf_cubin,
            dynamic_leaf_cubin_size,
            dynamic_leaf_plan_storage,
            dynamic_leaf_plan_storage_size,
            &dynamic_leaf_plan_storage_size,
            &dynamic_leaf_plan);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            state->materialize_handle,
            state->materialize_cubin,
            state->materialize_cubin_size,
            2u,
            2u,
            &runners[TEST_MATRIX_SHAPE_MATERIALIZE]);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            affine_stats_plan,
            affine_stats_cubin,
            affine_stats_cubin_size,
            2u,
            2u,
            &runners[TEST_MATRIX_SHAPE_AFFINE_STATS]);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            state->sse_handle,
            state->sse_cubin,
            state->sse_cubin_size,
            2u,
            2u,
            &runners[TEST_MATRIX_SHAPE_SSE]);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            dynamic_constant_plan,
            dynamic_constant_cubin,
            dynamic_constant_cubin_size,
            2u,
            2u,
            &runners[TEST_MATRIX_SHAPE_DYNAMIC_CONSTANT_SSE]);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(
            dynamic_leaf_plan,
            dynamic_leaf_cubin,
            dynamic_leaf_cubin_size,
            2u,
            2u,
            &runners[TEST_MATRIX_SHAPE_DYNAMIC_LEAF_SSE]);
        success = result == SECANT_SUCCESS;
    }
    if (success &&
        (cuMemAlloc(&device_constants, sizeof(constants)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_leaf_masks, sizeof(leaf_masks)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_leaf_words, sizeof(leaf_words)) != CUDA_SUCCESS ||
         cuMemAlloc(
            &device_target_stats_allocation,
            target_stats_allocation_elements * sizeof(float)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_output_allocation, output_allocation_elements * sizeof(float)) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device_constants, constants, sizeof(constants)) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device_leaf_masks, leaf_masks, sizeof(leaf_masks)) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device_leaf_words, leaf_words, sizeof(leaf_words)) != CUDA_SUCCESS)) {
        success = 0;
    }
    device_output = device_output_allocation + TEST_MATRIX_GUARD_ELEMENTS * sizeof(float);
    device_target_stats = device_target_stats_allocation + TEST_MATRIX_GUARD_ELEMENTS * sizeof(float);
    for (shape_idx = 0u; shape_idx < TEST_MATRIX_SHAPE_COUNT && success; ++shape_idx) {
        const TestMatrixShape shape = (TestMatrixShape)shape_idx;
        size_t case_idx;

        success = test_guarded_output_reset(device_output_allocation, output_allocation_elements) &&
            test_guarded_output_reset(device_target_stats_allocation, target_stats_allocation_elements);
        result = test_matrix_runner_run(
            shape,
            runners[shape],
            state,
            shape_asts[shape],
            0u,
            1u,
            device_constants,
            device_leaf_masks,
            device_leaf_words,
            device_output,
            output_num_elements,
            NULL);
        if (!success || result != SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "%s accepted zero ASTs\n", test_matrix_shape_name(shape));
            success = 0;
            break;
        }
        result = test_matrix_runner_run(
            shape,
            runners[shape],
            state,
            invalid_asts + shape_idx,
            1u,
            1u,
            device_constants,
            device_leaf_masks,
            device_leaf_words,
            device_output,
            output_num_elements,
            NULL);
        if (result != SECANT_ERROR_BAD_PROGRAM) {
            fprintf(
                stderr,
                "%s accepted an out-of-range input index: %s\n",
                test_matrix_shape_name(shape),
                secant_result_to_string(result));
            success = 0;
            break;
        }
        if (shape == TEST_MATRIX_SHAPE_SSE) {
            const SecantAstInstruction* dynamic_ast[1] = { test_dynamic_leaf_identity };

            result = test_matrix_runner_run(
                shape,
                runners[shape],
                state,
                dynamic_ast,
                1u,
                1u,
                device_constants,
                device_leaf_masks,
                device_leaf_words,
                device_output,
                output_num_elements,
                NULL);
            if (result != SECANT_ERROR_BAD_PROGRAM) {
                fprintf(
                    stderr,
                    "static SSE accepted a dynamic input: %s\n",
                    secant_result_to_string(result));
                success = 0;
                break;
            }
        }
        if (shape != TEST_MATRIX_SHAPE_MATERIALIZE) {
            result = test_matrix_runner_run(
                shape,
                runners[shape],
                state,
                shape_asts[shape],
                1u,
                0u,
                device_constants,
                device_leaf_masks,
                device_leaf_words,
                device_output,
                output_num_elements,
                NULL);
            if (result != SECANT_ERROR_INVALID_VALUE) {
                fprintf(stderr, "%s accepted zero targets\n", test_matrix_shape_name(shape));
                success = 0;
                break;
            }
            result = test_matrix_runner_run(
                shape,
                runners[shape],
                state,
                shape_asts[shape],
                1u,
                TEST_TARGETS + 1u,
                device_constants,
                device_leaf_masks,
                device_leaf_words,
                device_output,
                output_num_elements,
                NULL);
            if (result != SECANT_ERROR_INVALID_VALUE) {
                fprintf(stderr, "%s accepted too many targets\n", test_matrix_shape_name(shape));
                success = 0;
                break;
            }
        }
        if (!test_guarded_output_read(
                test_matrix_shape_name(shape),
                device_output_allocation,
                output_allocation_elements,
                0u,
                actual) ||
            !test_guarded_output_read(
                test_matrix_shape_name(shape),
                device_target_stats_allocation,
                target_stats_allocation_elements,
                0u,
                actual_target_stats)) {
            success = 0;
            break;
        }
        for (case_idx = 0u; case_idx < sizeof(ast_counts) / sizeof(ast_counts[0]) && success; ++case_idx) {
            const size_t num_asts = ast_counts[case_idx];
            const size_t num_targets = shape == TEST_MATRIX_SHAPE_MATERIALIZE
                ? 0u
                : (case_idx % 2u == 0u ? 1u : TEST_TARGETS);
            const size_t output_groups =
                shape == TEST_MATRIX_SHAPE_MATERIALIZE || shape == TEST_MATRIX_SHAPE_AFFINE_STATS
                    ? 1u
                    : num_targets;
            const size_t output_inner = shape == TEST_MATRIX_SHAPE_MATERIALIZE
                ? TEST_ROWS
                : (shape == TEST_MATRIX_SHAPE_SSE
                    ? 1u
                    : (shape == TEST_MATRIX_SHAPE_AFFINE_STATS
                        ? SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + num_targets
                        : TEST_DYNAMIC_SETTINGS));
            const size_t active_output_elements = num_asts * output_groups * output_inner;
            const size_t active_target_stats_elements = shape == TEST_MATRIX_SHAPE_AFFINE_STATS
                ? num_targets * SECANT_AFFINE_TARGET_STAT_COUNT_F32
                : 0u;
            const size_t expected_modules = (num_asts + TEST_ASTS - 1u) / TEST_ASTS;
            const float tolerance = shape == TEST_MATRIX_SHAPE_MATERIALIZE ? 0.0f : 2.0e-4f;
            SecantRunnerStats stats = secant_runner_stats_init();
            char case_name[96];

            (void)snprintf(
                case_name,
                sizeof(case_name),
                "%s asts=%zu targets=%zu",
                test_matrix_shape_name(shape),
                num_asts,
                num_targets);

            memset(expected, 0, max_output_elements * sizeof(*expected));
            memset(expected_target_stats, 0, sizeof(expected_target_stats));
            memset(expected_affine_sse, 0, sizeof(expected_affine_sse));
            if (shape == TEST_MATRIX_SHAPE_AFFINE_STATS) {
                test_target_stats_cpu(
                    targets,
                    TEST_ROWS,
                    TEST_ROWS,
                    num_targets,
                    expected_target_stats,
                    SECANT_AFFINE_TARGET_STAT_COUNT_F32);
            }
            result = test_matrix_cpu_run(
                shape,
                shape_asts[shape],
                num_asts,
                num_targets,
                input,
                targets,
                constants,
                leaf_masks,
                leaf_words,
                expected,
                active_output_elements);
            if (result != SECANT_SUCCESS) {
                test_print_cubin_error("shape matrix CPU", result);
                success = 0;
                break;
            }
            if (shape == TEST_MATRIX_SHAPE_AFFINE_STATS) {
                result = secant_test_cpu_sse_run(
                    TEST_INPUTS,
                    num_targets,
                    NULL,
                    0u,
                    shape_asts[shape],
                    num_asts,
                    input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    targets,
                    TEST_TARGETS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    expected_affine_sse,
                    num_asts * num_targets,
                    num_targets);
                if (result != SECANT_SUCCESS) {
                    test_print_cubin_error("affine raw SSE oracle", result);
                    success = 0;
                    break;
                }
            }
            if (!test_guarded_output_reset(device_output_allocation, output_allocation_elements) ||
                !test_guarded_output_reset(device_target_stats_allocation, target_stats_allocation_elements)) {
                fprintf(stderr, "%s output guard reset failed\n", case_name);
                success = 0;
                break;
            }
            if (shape == TEST_MATRIX_SHAPE_AFFINE_STATS &&
                !test_target_stats_run(
                    target_stats_function,
                    state->targets,
                    TEST_ROWS,
                    TEST_ROWS,
                    num_targets,
                    device_target_stats,
                    SECANT_AFFINE_TARGET_STAT_COUNT_F32)) {
                fprintf(stderr, "%s target-statistics preprocessing failed\n", case_name);
                success = 0;
                break;
            }
            result = test_matrix_runner_run(
                shape,
                runners[shape],
                state,
                shape_asts[shape],
                num_asts,
                num_targets,
                device_constants,
                device_leaf_masks,
                device_leaf_words,
                device_output,
                output_num_elements,
                &stats);
            if (result != SECANT_SUCCESS || stats.num_modules != expected_modules ||
                stats.modules_loaded != expected_modules || stats.num_asts != num_asts) {
                test_print_cubin_error(test_matrix_shape_name(shape), result);
                success = 0;
                break;
            }
            if (!test_guarded_output_read(
                    case_name,
                    device_output_allocation,
                    output_allocation_elements,
                    active_output_elements,
                    actual) ||
                !test_compare(
                    case_name,
                    actual,
                    expected,
                    active_output_elements,
                    tolerance) ||
                !test_guarded_output_read(
                    case_name,
                    device_target_stats_allocation,
                    target_stats_allocation_elements,
                    active_target_stats_elements,
                    actual_target_stats) ||
                (shape == TEST_MATRIX_SHAPE_AFFINE_STATS &&
                 !test_compare(
                    case_name,
                    actual_target_stats,
                    expected_target_stats,
                    active_target_stats_elements,
                    tolerance))) {
                success = 0;
            }
            if (shape == TEST_MATRIX_SHAPE_AFFINE_STATS && success) {
                size_t affine_ast_idx;

                for (affine_ast_idx = 0u; affine_ast_idx < num_asts; ++affine_ast_idx) {
                    size_t target_idx;
                    const float* ast_statistics = actual + affine_ast_idx * output_inner;

                    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                        const float* target_statistics =
                            actual_target_stats + target_idx * SECANT_AFFINE_TARGET_STAT_COUNT_F32;

                        actual_affine_sse[affine_ast_idx * num_targets + target_idx] =
                            ast_statistics[SECANT_AFFINE_AST_STAT_SUM_PREDICTION_SQUARED_F32] -
                            2.0f * ast_statistics[
                                SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + target_idx] +
                            target_statistics[SECANT_AFFINE_TARGET_STAT_SUM_SQUARED_F32];
                    }
                }
                success = test_compare(
                    "affine reconstructed raw SSE",
                    actual_affine_sse,
                    expected_affine_sse,
                    num_asts * num_targets,
                    6.0e-4f);
            }
        }
    }
    if (success) {
        success = test_sse_trajectory_batch(
            state,
            runners[TEST_MATRIX_SHAPE_SSE],
            shape_asts[TEST_MATRIX_SHAPE_SSE],
            input,
            targets);
    }
    if (success) {
        static const SecantAstInstruction* const identity_asts[] = { test_identity };
        static const float expected_coefficients[] = { -2.0f, 1.9f, -1.4f, 0.92f };
        const size_t affine_stats_count = SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + TEST_TARGETS;
        float cpu_ast_stats[SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + TEST_TARGETS] = { 0.0f };
        float cpu_target_stats[TEST_TARGETS * SECANT_AFFINE_TARGET_STAT_COUNT_F32];
        float cpu_coefficients[2u * TEST_TARGETS];
        float cubin_coefficients[2u * TEST_TARGETS];

        test_target_stats_cpu(
            targets,
            TEST_ROWS,
            TEST_ROWS,
            TEST_TARGETS,
            cpu_target_stats,
            SECANT_AFFINE_TARGET_STAT_COUNT_F32);
        result = secant_test_cpu_affine_stats_run(
            TEST_INPUTS,
            TEST_TARGETS,
            NULL,
            0u,
            identity_asts,
            1u,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            cpu_ast_stats,
            affine_stats_count,
            affine_stats_count);
        success = result == SECANT_SUCCESS &&
            test_affine_coefficients_recover(
                cpu_ast_stats,
                cpu_target_stats,
                SECANT_AFFINE_TARGET_STAT_COUNT_F32,
                TEST_ROWS,
                TEST_TARGETS,
                cpu_coefficients) &&
            test_compare(
                "CPU known affine coefficients",
                cpu_coefficients,
                expected_coefficients,
                2u * TEST_TARGETS,
                3.0e-4f) &&
            test_guarded_output_reset(device_output_allocation, output_allocation_elements) &&
            test_guarded_output_reset(device_target_stats_allocation, target_stats_allocation_elements) &&
            test_target_stats_run(
                target_stats_function,
                state->targets,
                TEST_ROWS,
                TEST_ROWS,
                TEST_TARGETS,
                device_target_stats,
                SECANT_AFFINE_TARGET_STAT_COUNT_F32);
        if (success) {
            result = secant_test_cubin_affine_stats_runner_run(
                runners[TEST_MATRIX_SHAPE_AFFINE_STATS],
                NULL,
                0u,
                identity_asts,
                1u,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                TEST_TARGETS,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                affine_stats_count,
                affine_stats_count,
                NULL);
            success = result == SECANT_SUCCESS &&
                test_guarded_output_read(
                    "known affine AST statistics",
                    device_output_allocation,
                    output_allocation_elements,
                    affine_stats_count,
                    actual) &&
                test_guarded_output_read(
                    "known affine target statistics",
                    device_target_stats_allocation,
                    target_stats_allocation_elements,
                    TEST_TARGETS * SECANT_AFFINE_TARGET_STAT_COUNT_F32,
                    actual_target_stats) &&
                test_affine_coefficients_recover(
                    actual,
                    actual_target_stats,
                    SECANT_AFFINE_TARGET_STAT_COUNT_F32,
                    TEST_ROWS,
                    TEST_TARGETS,
                    cubin_coefficients) &&
                test_compare(
                    "CUBIN known affine coefficients",
                    cubin_coefficients,
                    expected_coefficients,
                    2u * TEST_TARGETS,
                    3.0e-4f);
        }
    }
    if (success) {
        const size_t affine_stats_count = SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + 1u;

        memset(expected, 0, max_output_elements * sizeof(*expected));
        result = secant_test_cpu_affine_stats_run(
            TEST_INPUTS,
            1u,
            NULL,
            0u,
            shape_asts[TEST_MATRIX_SHAPE_AFFINE_STATS],
            1u,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            affine_stats_count,
            affine_stats_count);
        if (result != SECANT_SUCCESS ||
            !test_guarded_output_reset(device_output_allocation, output_allocation_elements) ||
            !test_guarded_output_reset(
                device_target_stats_allocation,
                target_stats_allocation_elements)) {
            success = 0;
        }
        if (success) {
            result = secant_test_cubin_affine_stats_runner_run(
                runners[TEST_MATRIX_SHAPE_AFFINE_STATS],
                NULL,
                0u,
                shape_asts[TEST_MATRIX_SHAPE_AFFINE_STATS],
                1u,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                1u,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                affine_stats_count,
                affine_stats_count,
                NULL);
            success = result == SECANT_SUCCESS &&
                test_guarded_output_read(
                    "affine precomputed target moments",
                    device_output_allocation,
                    output_allocation_elements,
                    affine_stats_count,
                    actual) &&
                test_compare(
                    "affine precomputed target moments",
                    actual,
                    expected,
                    affine_stats_count,
                    2.0e-4f) &&
                test_guarded_output_read(
                    "affine skipped target moments",
                    device_target_stats_allocation,
                    target_stats_allocation_elements,
                    0u,
                    actual_target_stats);
        }
    }
    for (shape_idx = 0u; shape_idx < TEST_MATRIX_SHAPE_COUNT; ++shape_idx) {
        if (runners[shape_idx] != NULL && secant_cubin_runner_destroy(runners[shape_idx]) != SECANT_SUCCESS) {
            success = 0;
        }
    }
    if (device_output_allocation != 0u) {
        (void)cuMemFree(device_output_allocation);
    }
    if (target_stats_module != NULL && cuModuleUnload(target_stats_module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (device_target_stats_allocation != 0u) {
        (void)cuMemFree(device_target_stats_allocation);
    }
    if (device_leaf_words != 0u) {
        (void)cuMemFree(device_leaf_words);
    }
    if (device_leaf_masks != 0u) {
        (void)cuMemFree(device_leaf_masks);
    }
    if (device_constants != 0u) {
        (void)cuMemFree(device_constants);
    }
    free(dynamic_leaf_plan_storage);
    free(dynamic_constant_plan_storage);
    free(affine_stats_plan_storage);
    free(dynamic_leaf_cubin);
    free(dynamic_constant_cubin);
    free(affine_stats_cubin);
    free(actual);
    free(expected);
    return success;
}

static int
test_run_gram_stats(
    const TestState* state,
    int major,
    int minor,
    const float* input,
    const float* targets
) {
    const size_t asts_per_cohort = TEST_GRAM_ASTS_PER_COHORT;
    const size_t num_asts = TEST_GRAM_ASTS;
    const size_t gram_count = asts_per_cohort * asts_per_cohort;
    const size_t statistics_count = asts_per_cohort + gram_count + asts_per_cohort * TEST_TARGETS;
    const size_t statistics_leading_dimension = statistics_count + 3u;
    const size_t num_cohorts = (num_asts + asts_per_cohort - 1u) / asts_per_cohort;
    const size_t output_elements =
        (num_cohorts - 1u) * statistics_leading_dimension + statistics_count;
    const size_t reduced_statistics_count = asts_per_cohort + gram_count + asts_per_cohort;
    const size_t reduced_output_elements =
        (num_cohorts - 1u) * statistics_leading_dimension + reduced_statistics_count;
    const size_t patch_capacity_instructions = asts_per_cohort * TEST_PATCH_INSTRUCTIONS + 13u;
    SecantCubinGramStatsRecipe recipe = secant_cubin_gram_stats_recipe_init();
    const SecantAstInstruction* asts[TEST_GRAM_ASTS];
    float* expected = NULL;
    float* actual = NULL;
    unsigned char* cubin = NULL;
    void* plan_storage = NULL;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdeviceptr device_output = 0u;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    size_t ast_idx;
    SecantResult result;
    int success = 1;

    recipe.num_kernels = TEST_KERNELS;
    recipe.asts_per_kernel = asts_per_cohort;
    recipe.num_inputs = TEST_INPUTS;
    recipe.num_targets = TEST_TARGETS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        asts[ast_idx] = test_asts[ast_idx % TEST_ASTS];
    }
    expected = (float*)calloc(output_elements, sizeof(*expected));
    actual = (float*)malloc(output_elements * sizeof(*actual));
    if (expected == NULL || actual == NULL) {
        success = 0;
    }
    if (success) {
        result = secant_test_cpu_gram_stats_run(
            asts_per_cohort,
            TEST_INPUTS,
            TEST_TARGETS,
            test_routines,
            1u,
            asts,
            num_asts,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            output_elements,
            statistics_leading_dimension);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        success = test_compile_gram_stats_cubin_template(&recipe, major, minor, &cubin, &cubin_size);
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            NULL,
            0u,
            &plan_storage_size,
            &plan);
        if (result != SECANT_SUCCESS || plan_storage_size == 0u) {
            test_print_cubin_error("Gram-stats inspect measure", result);
            success = 0;
        }
    }
    if (success) {
        plan_storage = malloc(plan_storage_size);
        success = plan_storage != NULL;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            plan_storage,
            plan_storage_size,
            &plan_storage_size,
            &plan);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("Gram-stats inspect", result);
            success = 0;
        }
    }
    if (success) {
        result = secant_test_cubin_runner_create(plan, cubin, cubin_size, 2u, 2u, &runner);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("Gram-stats runner create", result);
            success = 0;
        }
    }
    if (success && cuMemAlloc(&device_output, output_elements * sizeof(*actual)) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        result = secant_test_cubin_gram_stats_runner_run(
            runner,
            test_routines,
            1u,
            asts,
            num_asts,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_TARGETS,
            (uintptr_t)state->targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            output_elements,
            statistics_leading_dimension,
            NULL);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("Gram-stats runner run", result);
            success = 0;
        }
    }
    if (success && cuMemcpyDtoH(actual, device_output, output_elements * sizeof(*actual)) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        success = test_compare(
            "Gram sufficient statistics",
            actual,
            expected,
            output_elements,
            3.0e-3f);
    }
    if (success) {
        size_t cohort_idx;

        for (cohort_idx = 0u; cohort_idx < num_cohorts && success; ++cohort_idx) {
            const float* cohort = actual + cohort_idx * statistics_leading_dimension;
            const size_t first_ast = cohort_idx * asts_per_cohort;
            size_t active_asts = num_asts - first_ast;
            size_t lhs;

            if (active_asts > asts_per_cohort) {
                active_asts = asts_per_cohort;
            }
            for (lhs = 0u; lhs < active_asts && success; ++lhs) {
                size_t rhs;

                for (rhs = 0u; rhs < active_asts; ++rhs) {
                    const float lhs_rhs = cohort[asts_per_cohort + lhs * asts_per_cohort + rhs];
                    const float rhs_lhs = cohort[asts_per_cohort + rhs * asts_per_cohort + lhs];

                    if (fabsf(lhs_rhs - rhs_lhs) > 1.0e-4f) {
                        fprintf(stderr, "Gram symmetry mismatch cohort=%zu lhs=%zu rhs=%zu\n", cohort_idx, lhs, rhs);
                        success = 0;
                        break;
                    }
                }
            }
        }
    }
    if (success) {
        memset(expected, 0, output_elements * sizeof(*expected));
        result = secant_test_cpu_gram_stats_run(
            asts_per_cohort,
            TEST_INPUTS,
            1u,
            test_routines,
            1u,
            asts,
            num_asts,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            targets,
            TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            reduced_output_elements,
            statistics_leading_dimension);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_gram_stats_runner_run(
            runner,
            test_routines,
            1u,
            asts,
            num_asts,
            (uintptr_t)state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            1u,
            (uintptr_t)state->targets,
            TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            (uintptr_t)device_output,
            reduced_output_elements,
            statistics_leading_dimension,
            NULL);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("Gram-stats reduced-target runner run", result);
            success = 0;
        }
    }
    if (success && cuMemcpyDtoH(
            actual,
            device_output,
            reduced_output_elements * sizeof(*actual)) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        success = test_compare(
            "Gram sufficient statistics with reduced targets",
            actual,
            expected,
            reduced_output_elements,
            3.0e-3f);
    }
    if (success) {
        SecantCubinGramStatsRun invalid_run = secant_cubin_gram_stats_run_init();

        invalid_run.programs.routines.items = test_routines;
        invalid_run.programs.routines.count = 1u;
        invalid_run.programs.asts.items = asts;
        invalid_run.programs.asts.count = num_asts;
        invalid_run.input.address = (uintptr_t)state->input;
        invalid_run.input.num_elements = TEST_INPUTS * TEST_ROWS;
        invalid_run.input.leading_dimension = TEST_ROWS;
        invalid_run.targets.address = (uintptr_t)state->targets;
        invalid_run.targets.num_elements = TEST_TARGETS * TEST_ROWS;
        invalid_run.targets.leading_dimension = TEST_ROWS;
        invalid_run.num_rows = TEST_ROWS;
        invalid_run.num_targets = TEST_TARGETS + 1u;
        invalid_run.statistics.address = (uintptr_t)device_output;
        invalid_run.statistics.num_elements = output_elements;
        invalid_run.statistics.leading_dimension = statistics_leading_dimension;
        success = secant_cubin_runner_run(runner, &invalid_run.header, NULL) == SECANT_ERROR_INVALID_VALUE;
    }
    if (runner != NULL && secant_cubin_runner_destroy(runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (device_output != 0u && cuMemFree(device_output) != CUDA_SUCCESS) {
        success = 0;
    }
    free(plan_storage);
    free(cubin);
    free(actual);
    free(expected);
    return success;
}

static int
test_pde_coefficients_solve(
    const float* statistics,
    double coefficients[TEST_PDE_FEATURES]
) {
    double augmented[TEST_PDE_FEATURES][TEST_PDE_FEATURES + 1u];
    const size_t gram_offset = TEST_PDE_FEATURES;
    const size_t cross_offset = gram_offset + TEST_PDE_FEATURES * TEST_PDE_FEATURES;
    size_t row;

    for (row = 0u; row < TEST_PDE_FEATURES; ++row) {
        size_t column;

        for (column = 0u; column < TEST_PDE_FEATURES; ++column) {
            augmented[row][column] =
                (double)statistics[gram_offset + row * TEST_PDE_FEATURES + column];
        }
        augmented[row][TEST_PDE_FEATURES] = (double)statistics[cross_offset + row];
    }
    for (row = 0u; row < TEST_PDE_FEATURES; ++row) {
        size_t pivot = row;
        size_t candidate;
        size_t eliminate_row;

        for (candidate = row + 1u; candidate < TEST_PDE_FEATURES; ++candidate) {
            if (fabs(augmented[candidate][row]) > fabs(augmented[pivot][row])) {
                pivot = candidate;
            }
        }
        if (!isfinite(augmented[pivot][row]) || fabs(augmented[pivot][row]) < 1.0e-12) {
            return 0;
        }
        if (pivot != row) {
            size_t column;

            for (column = row; column <= TEST_PDE_FEATURES; ++column) {
                const double temporary = augmented[row][column];

                augmented[row][column] = augmented[pivot][column];
                augmented[pivot][column] = temporary;
            }
        }
        for (eliminate_row = row + 1u; eliminate_row < TEST_PDE_FEATURES; ++eliminate_row) {
            const double factor = augmented[eliminate_row][row] / augmented[row][row];
            size_t column;

            for (column = row; column <= TEST_PDE_FEATURES; ++column) {
                augmented[eliminate_row][column] -= factor * augmented[row][column];
            }
        }
    }
    for (row = TEST_PDE_FEATURES; row-- > 0u;) {
        double value = augmented[row][TEST_PDE_FEATURES];
        size_t column;

        for (column = row + 1u; column < TEST_PDE_FEATURES; ++column) {
            value -= augmented[row][column] * coefficients[column];
        }
        coefficients[row] = value / augmented[row][row];
        if (!isfinite(coefficients[row])) {
            return 0;
        }
    }
    return 1;
}

static int
test_gram_stats_pde_solve(
    int major,
    int minor
) {
    const double pi = 3.1415926535897932384626433832795;
    const double advection_speed = 0.7;
    const double diffusivity = 0.08;
    const double amplitudes[4] = { 1.0, 0.45, -0.3, 0.2 };
    const double phases[4] = { 0.1, -0.35, 0.8, -1.1 };
    const double wavenumbers[4] = { 1.0, 2.0, 3.0, 5.0 };
    const double expected_coefficients[TEST_PDE_FEATURES] = {
        0.0, 0.0, -0.7, 0.08, 0.0, 0.0, 0.0, 0.0
    };
    const size_t gram_count = TEST_PDE_FEATURES * TEST_PDE_FEATURES;
    const size_t statistics_count = TEST_PDE_FEATURES + gram_count + TEST_PDE_FEATURES;
    SecantCubinGramStatsRecipe recipe = secant_cubin_gram_stats_recipe_init();
    float* input = NULL;
    float* target = NULL;
    unsigned char* cubin = NULL;
    void* plan_storage = NULL;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_target = 0u;
    CUdeviceptr device_statistics = 0u;
    float statistics[TEST_PDE_FEATURES + TEST_PDE_FEATURES * TEST_PDE_FEATURES + TEST_PDE_FEATURES];
    double coefficients[TEST_PDE_FEATURES] = { 0.0 };
    double target_square_sum = 0.0;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    size_t time_idx;
    SecantResult result;
    int success = 1;

    recipe.num_kernels = 1u;
    recipe.asts_per_kernel = TEST_PDE_FEATURES;
    recipe.num_inputs = TEST_PDE_INPUTS;
    recipe.num_targets = 1u;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = TEST_PDE_FEATURES * TEST_PATCH_INSTRUCTIONS + 13u;

    input = (float*)malloc(TEST_PDE_INPUTS * TEST_PDE_ROWS * sizeof(*input));
    target = (float*)malloc(TEST_PDE_ROWS * sizeof(*target));
    if (input == NULL || target == NULL) {
        success = 0;
    }
    /* This modal superposition exactly solves u_t = -c*u_x + nu*u_xx. */
    for (time_idx = 0u; time_idx < TEST_PDE_GRID_T && success; ++time_idx) {
        const double time = 1.25 * (double)time_idx / (double)(TEST_PDE_GRID_T - 1u);
        size_t space_idx;

        for (space_idx = 0u; space_idx < TEST_PDE_GRID_X; ++space_idx) {
            const double space = -pi + 2.0 * pi * (double)space_idx / (double)TEST_PDE_GRID_X;
            const size_t row = time_idx * TEST_PDE_GRID_X + space_idx;
            double u = 0.0;
            double ux = 0.0;
            double uxx = 0.0;
            double ut = 0.0;
            size_t mode;

            for (mode = 0u; mode < 4u; ++mode) {
                const double k = wavenumbers[mode];
                const double angle = k * (space - advection_speed * time) + phases[mode];
                const double decay = exp(-diffusivity * k * k * time);
                const double sine = sin(angle);
                const double cosine = cos(angle);

                u += amplitudes[mode] * sine * decay;
                ux += amplitudes[mode] * k * cosine * decay;
                uxx -= amplitudes[mode] * k * k * sine * decay;
                ut += amplitudes[mode] *
                    (-advection_speed * k * cosine - diffusivity * k * k * sine) * decay;
            }
            input[row] = (float)u;
            input[TEST_PDE_ROWS + row] = (float)ux;
            input[2u * TEST_PDE_ROWS + row] = (float)uxx;
            target[row] = (float)ut;
            target_square_sum += (double)target[row] * (double)target[row];
        }
    }
    if (success) {
        success = test_compile_gram_stats_cubin_template(&recipe, major, minor, &cubin, &cubin_size);
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            NULL,
            0u,
            &plan_storage_size,
            &plan);
        success = result == SECANT_SUCCESS && plan_storage_size != 0u;
    }
    if (success) {
        plan_storage = malloc(plan_storage_size);
        success = plan_storage != NULL;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            plan_storage,
            plan_storage_size,
            &plan_storage_size,
            &plan);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(plan, cubin, cubin_size, 2u, 2u, &runner);
        success = result == SECANT_SUCCESS;
    }
    if (success &&
        (cuMemAlloc(&device_input, TEST_PDE_INPUTS * TEST_PDE_ROWS * sizeof(*input)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_target, TEST_PDE_ROWS * sizeof(*target)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_statistics, statistics_count * sizeof(*statistics)) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success &&
        (cuMemcpyHtoD(
             device_input,
             input,
             TEST_PDE_INPUTS * TEST_PDE_ROWS * sizeof(*input)) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device_target, target, TEST_PDE_ROWS * sizeof(*target)) != CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        result = secant_test_cubin_gram_stats_runner_run(
            runner,
            NULL,
            0u,
            test_pde_asts,
            TEST_PDE_FEATURES,
            (uintptr_t)device_input,
            TEST_PDE_INPUTS * TEST_PDE_ROWS,
            TEST_PDE_ROWS,
            1u,
            (uintptr_t)device_target,
            TEST_PDE_ROWS,
            TEST_PDE_ROWS,
            TEST_PDE_ROWS,
            (uintptr_t)device_statistics,
            statistics_count,
            statistics_count,
            NULL);
        if (result != SECANT_SUCCESS) {
            test_print_cubin_error("PDE Gram-stats runner run", result);
            success = 0;
        }
    }
    if (success &&
        cuMemcpyDtoH(statistics, device_statistics, statistics_count * sizeof(*statistics)) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success) {
        success = test_pde_coefficients_solve(statistics, coefficients);
    }
    if (success) {
        size_t coefficient_idx;

        for (coefficient_idx = 0u; coefficient_idx < TEST_PDE_FEATURES; ++coefficient_idx) {
            if (fabs(coefficients[coefficient_idx] - expected_coefficients[coefficient_idx]) > 5.0e-4) {
                fprintf(
                    stderr,
                    "PDE coefficient mismatch at %zu: %.9g != %.9g\n",
                    coefficient_idx,
                    coefficients[coefficient_idx],
                    expected_coefficients[coefficient_idx]);
                success = 0;
                break;
            }
        }
    }
    if (success) {
        const size_t gram_offset = TEST_PDE_FEATURES;
        const size_t cross_offset = gram_offset + gram_count;
        double prediction_target_sum = 0.0;
        double prediction_square_sum = 0.0;
        double mse;
        size_t lhs;

        for (lhs = 0u; lhs < TEST_PDE_FEATURES; ++lhs) {
            size_t rhs;

            prediction_target_sum += coefficients[lhs] * (double)statistics[cross_offset + lhs];
            for (rhs = 0u; rhs < TEST_PDE_FEATURES; ++rhs) {
                prediction_square_sum += coefficients[lhs] * coefficients[rhs] *
                    (double)statistics[gram_offset + lhs * TEST_PDE_FEATURES + rhs];
            }
        }
        mse = (target_square_sum - 2.0 * prediction_target_sum + prediction_square_sum) /
            (double)TEST_PDE_ROWS;
        if (!isfinite(mse) || fabs(mse) > 1.0e-5) {
            fprintf(stderr, "PDE sufficient-statistics MSE mismatch: %.9g\n", mse);
            success = 0;
        }
    }
    if (runner != NULL && secant_cubin_runner_destroy(runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (device_statistics != 0u && cuMemFree(device_statistics) != CUDA_SUCCESS) {
        success = 0;
    }
    if (device_target != 0u && cuMemFree(device_target) != CUDA_SUCCESS) {
        success = 0;
    }
    if (device_input != 0u && cuMemFree(device_input) != CUDA_SUCCESS) {
        success = 0;
    }
    free(plan_storage);
    free(cubin);
    free(target);
    free(input);
    return success;
}

static uint32_t
test_dynamic_stress_random_u32(uint64_t* state) {
    uint64_t value = *state;

    value ^= value >> 12u;
    value ^= value << 25u;
    value ^= value >> 27u;
    *state = value;
    return (uint32_t)((value * UINT64_C(0x2545f4914f6cdd1d)) >> 32u);
}

static int
test_dynamic_stress_program_rewrite(
    SecantAstInstruction* program,
    size_t program_size
) {
    size_t offset = 0u;

    while (offset < program_size) {
        SecantAstInstruction* instruction = program + offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u || instruction_size > program_size - offset) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
            instruction[0] = SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
        }
        offset += instruction_size;
    }
    return offset == program_size;
}

static int
test_dynamic_leaf_random_stress(
    const TestState* state,
    int major,
    int minor,
    const float* input,
    const float* targets
) {
    const size_t num_asts = TEST_DYNAMIC_STRESS_ASTS;
    const size_t num_settings = TEST_DYNAMIC_STRESS_SETTINGS;
    const size_t output_count = num_asts * TEST_TARGETS * num_settings;
    const size_t program_bytes = num_asts * SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY;
    SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    SecantAstInstruction* program_storage = NULL;
    const SecantAstInstruction** asts = NULL;
    size_t* program_sizes = NULL;
    uint32_t* program_masks = NULL;
    size_t* program_mufu_counts = NULL;
    uint32_t* leaf_masks = NULL;
    uint32_t* leaf_words = NULL;
    float* expected = NULL;
    float* actual = NULL;
    unsigned char* cubin = NULL;
    void* plan_storage = NULL;
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdeviceptr device_masks = 0u;
    CUdeviceptr device_words = 0u;
    CUdeviceptr device_output = 0u;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    size_t iteration;
    float max_absolute_errors[2] = { 0.0f, 0.0f };
    float max_relative_errors[2] = { 0.0f, 0.0f };
    SecantResult result;
    int success = 1;

    recipe.num_kernels = TEST_DYNAMIC_STRESS_KERNELS;
    recipe.asts_per_kernel = TEST_DYNAMIC_STRESS_ASTS_PER_KERNEL;
    recipe.num_input_columns = TEST_INPUTS;
    recipe.num_static_input_columns = 0u;
    recipe.num_dynamic_leaves = TEST_DYNAMIC_LEAVES;
    recipe.num_targets = TEST_TARGETS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = TEST_DYNAMIC_STRESS_PATCH_INSTRUCTIONS;

    program_storage = (SecantAstInstruction*)malloc(program_bytes);
    asts = (const SecantAstInstruction**)malloc(num_asts * sizeof(*asts));
    program_sizes = (size_t*)malloc(num_asts * sizeof(*program_sizes));
    program_masks = (uint32_t*)malloc(num_asts * sizeof(*program_masks));
    program_mufu_counts = (size_t*)malloc(num_asts * sizeof(*program_mufu_counts));
    leaf_masks = (uint32_t*)malloc(num_settings * sizeof(*leaf_masks));
    leaf_words = (uint32_t*)malloc(num_settings * TEST_DYNAMIC_LEAVES * sizeof(*leaf_words));
    expected = (float*)malloc(output_count * sizeof(*expected));
    actual = (float*)malloc(output_count * sizeof(*actual));
    if (program_storage == NULL || asts == NULL || program_sizes == NULL || program_masks == NULL ||
        program_mufu_counts == NULL || leaf_masks == NULL || leaf_words == NULL || expected == NULL ||
        actual == NULL) {
        success = 0;
    }
    if (success) {
        success = test_compile_dynamic_leaf_cubin_template(&recipe, major, minor, &cubin, &cubin_size);
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            NULL,
            0u,
            &plan_storage_size,
            &plan);
        success = result == SECANT_SUCCESS && plan_storage_size != 0u;
    }
    if (success) {
        plan_storage = malloc(plan_storage_size);
        success = plan_storage != NULL;
    }
    if (success) {
        result = secant_test_cubin_inspect(
            &recipe,
            cubin,
            cubin_size,
            plan_storage,
            plan_storage_size,
            &plan_storage_size,
            &plan);
        success = result == SECANT_SUCCESS;
    }
    if (success) {
        result = secant_test_cubin_runner_create(plan, cubin, cubin_size, 2u, 2u, &runner);
        success = result == SECANT_SUCCESS;
    }
    if (success &&
        (cuMemAlloc(&device_masks, num_settings * sizeof(*leaf_masks)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_words, num_settings * TEST_DYNAMIC_LEAVES * sizeof(*leaf_words)) != CUDA_SUCCESS ||
         cuMemAlloc(&device_output, output_count * sizeof(*actual)) != CUDA_SUCCESS)) {
        success = 0;
    }

    for (iteration = 0u; iteration < TEST_DYNAMIC_STRESS_ITERATIONS && success; ++iteration) {
        const uint64_t seed = UINT64_C(0x5eca17d1) + iteration;
        uint64_t random_state = seed;
        size_t ast_idx;
        size_t setting;

        for (ast_idx = 0u; ast_idx < num_asts && success; ++ast_idx) {
            SecantAstInstruction* program = program_storage + ast_idx * SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY;
            const int allow_mufu = ((iteration + ast_idx) & 3u) != 0u;

            success = secant_test_ast_stress_program_generate(
                program,
                program_sizes + ast_idx,
                TEST_DYNAMIC_LEAVES,
                8u,
                3u,
                allow_mufu,
                seed,
                iteration,
                ast_idx,
                program_masks + ast_idx) &&
                test_dynamic_stress_program_rewrite(program, program_sizes[ast_idx]);
            asts[ast_idx] = program;
            program_mufu_counts[ast_idx] = secant_test_ast_stress_program_mufu_count(
                program,
                program_sizes[ast_idx]);
        }
        for (setting = 0u; setting < num_settings; ++setting) {
            const uint32_t valid_mask = (UINT32_C(1) << TEST_DYNAMIC_LEAVES) - 1u;
            size_t leaf_idx;

            leaf_masks[setting] = setting == 0u
                ? valid_mask
                : (setting == 1u ? 0u : test_dynamic_stress_random_u32(&random_state) & valid_mask);
            for (leaf_idx = 0u; leaf_idx < TEST_DYNAMIC_LEAVES; ++leaf_idx) {
                uint32_t word;

                if ((leaf_masks[setting] & (UINT32_C(1) << leaf_idx)) != 0u) {
                    word = test_dynamic_stress_random_u32(&random_state) % TEST_INPUTS;
                } else {
                    const float value =
                        ((float)(test_dynamic_stress_random_u32(&random_state) & 0xffffu) / 65535.0f - 0.5f) * 1.5f;

                    memcpy(&word, &value, sizeof(word));
                }
                leaf_words[setting * TEST_DYNAMIC_LEAVES + leaf_idx] = word;
            }
        }
        memset(expected, 0, output_count * sizeof(*expected));
        result = secant_test_cpu_dynamic_leaf_sse_run(
            TEST_DYNAMIC_LEAVES,
            TEST_INPUTS,
            0u,
            TEST_TARGETS,
            secant_test_ast_stress_routines,
            SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
            asts,
            num_asts,
            input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            leaf_masks,
            num_settings,
            leaf_words,
            num_settings * TEST_DYNAMIC_LEAVES,
            TEST_DYNAMIC_LEAVES,
            num_settings,
            targets,
            TEST_TARGETS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            expected,
            output_count,
            num_settings);
        success = result == SECANT_SUCCESS;
        if (success &&
            (cuMemcpyHtoD(device_masks, leaf_masks, num_settings * sizeof(*leaf_masks)) != CUDA_SUCCESS ||
             cuMemcpyHtoD(
                 device_words,
                 leaf_words,
                 num_settings * TEST_DYNAMIC_LEAVES * sizeof(*leaf_words)) != CUDA_SUCCESS)) {
            success = 0;
        }
        if (success) {
            result = secant_test_cubin_dynamic_leaf_sse_runner_run(
                runner,
                secant_test_ast_stress_routines,
                SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
                asts,
                num_asts,
                (uintptr_t)state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_INPUTS,
                TEST_ROWS,
                (uintptr_t)device_masks,
                num_settings,
                (uintptr_t)device_words,
                num_settings * TEST_DYNAMIC_LEAVES,
                TEST_DYNAMIC_LEAVES,
                num_settings,
                TEST_TARGETS,
                (uintptr_t)state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)device_output,
                output_count,
                num_settings,
                NULL);
            success = result == SECANT_SUCCESS;
        }
        if (success && cuMemcpyDtoH(actual, device_output, output_count * sizeof(*actual)) != CUDA_SUCCESS) {
            success = 0;
        }
        if (success) {
            success = secant_test_ast_stress_compare(
                "cubin",
                "dynamic-leaf SSE",
                expected,
                actual,
                output_count,
                TEST_TARGETS * num_settings,
                iteration,
                seed,
                asts,
                program_sizes,
                program_masks,
                program_mufu_counts,
                2.0e-3f,
                8.0e-4f,
                1.0e-3f,
                5.0e-4f,
                max_absolute_errors,
                max_relative_errors);
        }
    }

    if (runner != NULL && secant_cubin_runner_destroy(runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (device_output != 0u && cuMemFree(device_output) != CUDA_SUCCESS) {
        success = 0;
    }
    if (device_words != 0u && cuMemFree(device_words) != CUDA_SUCCESS) {
        success = 0;
    }
    if (device_masks != 0u && cuMemFree(device_masks) != CUDA_SUCCESS) {
        success = 0;
    }
    free(plan_storage);
    free(cubin);
    free(actual);
    free(expected);
    free(leaf_words);
    free(leaf_masks);
    free(program_mufu_counts);
    free(program_masks);
    free(program_sizes);
    free(asts);
    free(program_storage);
    return success;
}

int
main(void) {
    float input[TEST_INPUTS * TEST_ROWS];
    float targets[TEST_TARGETS * TEST_ROWS];
    float expected_materialize[TEST_ASTS * TEST_ROWS];
    float actual_materialize[TEST_ASTS * TEST_ROWS];
    float expected_sse[TEST_ASTS * TEST_TARGETS] = { 0.0f };
    float actual_sse[TEST_ASTS * TEST_TARGETS];
    TestState state;
    SecantResult cpu_result;
    size_t row;
    int major = 0;
    int minor = 0;
    int success;

    for (row = 0u; row < TEST_ROWS; ++row) {
        input[row] = 0.5f + 0.002f * (float)row;
        input[TEST_ROWS + row] = 1.1f + 0.001f * (float)row;
        input[2u * TEST_ROWS + row] = -0.4f + 0.0015f * (float)row;
        input[3u * TEST_ROWS + row] = 0.8f - 0.0005f * (float)row;
        targets[row] = 0.3f + 0.001f * (float)row;
        targets[TEST_ROWS + row] = -0.2f + 0.0007f * (float)row;
    }
    cpu_result = secant_test_cpu_materialize_run(
        TEST_INPUTS,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        input,
        TEST_INPUTS * TEST_ROWS,
        TEST_ROWS,
        TEST_ROWS,
        expected_materialize,
        TEST_ASTS * TEST_ROWS,
        TEST_ROWS);
    success = cpu_result == SECANT_SUCCESS;
    cpu_result = secant_test_cpu_sse_run(
        TEST_INPUTS,
        TEST_TARGETS,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        input,
        TEST_INPUTS * TEST_ROWS,
        TEST_ROWS,
        targets,
        TEST_TARGETS * TEST_ROWS,
        TEST_ROWS,
        TEST_ROWS,
        expected_sse,
        TEST_ASTS * TEST_TARGETS,
        TEST_TARGETS);
    success = cpu_result == SECANT_SUCCESS && success;
    success = test_cuda_create(&state, &major, &minor) && success;
    if (success) {
        success = test_create_templates(&state, major, minor);
    }
    if (success) {
        success = test_write_cubins(&state);
    }
    if (success) {
        success = test_high_capacity_sse_patch(major, minor);
    }
    if (success) {
        success = test_load_modules(&state);
    }
    if (success) {
        success = test_run_gpu(
            &state,
            input,
            targets,
            actual_materialize,
            actual_sse);
    }
    if (success) {
        success = test_compare(
            "materialize",
            actual_materialize,
            expected_materialize,
            TEST_ASTS * TEST_ROWS,
            3.0e-5f);
    }
    if (success) {
        success = test_compare(
            "sse",
            actual_sse,
            expected_sse,
            TEST_ASTS * TEST_TARGETS,
            2.0e-4f);
    }
    if (success) {
        success = test_materialize_respecialize_in_place(&state, input);
    }
    if (success) {
        success = test_run_bulk_runner(
            &state,
            input,
            targets);
    }
    if (success) {
        success = test_run_partial_bulk_runner(
            &state,
            input,
            targets);
    }
    if (success) {
        success = test_run_dynamic_constant_bulk_runner(
            &state,
            major,
            minor,
            input,
            targets);
    }
    if (success) {
        success = test_run_dynamic_leaf_bulk_runner(
            &state,
            major,
            minor,
            input,
            targets,
            0);
    }
    if (success) {
        success = test_run_dynamic_leaf_bulk_runner(
            &state,
            major,
            minor,
            input,
            targets,
            1);
    }
    if (success) {
        success = test_run_gram_stats(
            &state,
            major,
            minor,
            input,
            targets);
    }
    if (success) {
        success = test_gram_stats_pde_solve(major, minor);
    }
    if (success) {
        success = test_dynamic_leaf_random_stress(
            &state,
            major,
            minor,
            input,
            targets);
    }
    if (success) {
        success = test_run_kernel_shape_matrix(
            &state,
            major,
            minor,
            input,
            targets);
    }
    test_state_destroy(&state);
    return success ? 0 : 1;
}
