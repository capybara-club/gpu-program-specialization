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
#ifdef SECANT_TEST_SASS_LM_OPTIMIZER
#include "secant_cubin_lm_optimizer_runner.h"
#elif defined(SECANT_TEST_PTX_LM_OPTIMIZER)
#include "secant_ptx_runner.h"
#else
#include "secant_cuda_runner.h"
#endif

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TEST_ROWS
#define TEST_ROWS 257u
#endif
#ifndef TEST_COLUMNS
#define TEST_COLUMNS 2u
#endif
#ifndef TEST_CAPACITY_COLUMNS
#define TEST_CAPACITY_COLUMNS 3u
#endif
#ifndef TEST_ASTS
#define TEST_ASTS 2u
#endif
#ifndef TEST_SETTINGS
#define TEST_SETTINGS 37u
#endif
#define TEST_PARAMETERS SECANT_CUDA_LM_OPTIMIZER_PARAMETERS
#define TEST_STATISTICS SECANT_CUDA_LM_OPTIMIZER_STATISTICS
#ifndef TEST_TILE_ROWS
#define TEST_TILE_ROWS 64u
#endif
#ifndef TEST_THREADS
#define TEST_THREADS 96u
#endif
#ifndef TEST_SETTINGS_PER_CTA
#define TEST_SETTINGS_PER_CTA 7u
#endif
#ifndef TEST_ITERATIONS
#define TEST_ITERATIONS 6u
#endif
#ifndef TEST_ACTIVE_PARAMETERS
#define TEST_ACTIVE_PARAMETERS 2u
#endif
#ifndef TEST_SCRATCH_BYTES
#define TEST_SCRATCH_BYTES (32u * 1024u * 1024u)
#endif

#ifdef SECANT_TEST_SASS_LM_SMOOTH_OPS
static const SecantAstInstruction test_smooth_routine[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_return_f32
};

/* Every zero-weighted term exercises a dual-number rule without changing the
 * linear fit. Inputs to restricted-domain operations stay strictly positive. */
static const SecantAstInstruction test_static_binding_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_cos_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_ex2_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_lg2_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_rsqrt_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_tanh_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_exp_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_log_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_add_f32,
    secant_ast_encode_div_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,

    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_fma_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_neg_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_POSITIVE_ZERO),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};
#elif defined(SECANT_TEST_SASS_LM_EDGE_OPS)
static const SecantAstInstruction test_static_binding_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_sub_f32,
    secant_ast_encode_abs_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_min_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_sub_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_max_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_sub_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};
#else
static const SecantAstInstruction test_static_binding_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};
#endif

/* Index one is both a column-bound mixed leaf and an independent raw constant. */
static const SecantAstInstruction test_mixed_binding_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_or_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

typedef struct TestCUDA {
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    int retained;
    int major;
    int minor;
    CUdeviceptr allocations[11];
    size_t allocation_count;
} TestCUDA;

static int
test_cuda_create(TestCUDA* cuda) {
    memset(cuda, 0, sizeof(*cuda));
    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&cuda->device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuda->context, cuda->device) != CUDA_SUCCESS) {
        return 0;
    }
    cuda->retained = 1;
    return cuCtxGetCurrent(&cuda->previous_context) == CUDA_SUCCESS &&
        cuCtxSetCurrent(cuda->context) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(
            &cuda->major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            cuda->device) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(
            &cuda->minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            cuda->device) == CUDA_SUCCESS;
}

static CUdeviceptr
test_device_copy(TestCUDA* cuda, const void* data, size_t bytes) {
    CUdeviceptr address = 0u;

    if (cuda->allocation_count >= sizeof(cuda->allocations) / sizeof(cuda->allocations[0]) ||
        cuMemAlloc(&address, bytes) != CUDA_SUCCESS) {
        return 0u;
    }
    cuda->allocations[cuda->allocation_count++] = address;
    if (data != NULL && cuMemcpyHtoD(address, data, bytes) != CUDA_SUCCESS) {
        return 0u;
    }
    if (data == NULL && cuMemsetD8(address, 0u, bytes) != CUDA_SUCCESS) {
        return 0u;
    }
    return address;
}

static void
test_cuda_destroy(TestCUDA* cuda) {
    while (cuda->allocation_count != 0u) {
        --cuda->allocation_count;
        if (cuda->allocations[cuda->allocation_count] != 0u) {
            (void)cuMemFree(cuda->allocations[cuda->allocation_count]);
        }
    }
    (void)cuCtxSetCurrent(cuda->previous_context);
    if (cuda->retained) {
        (void)cuDevicePrimaryCtxRelease(cuda->device);
    }
}

int
main(void) {
#ifdef SECANT_TEST_LM_REPEATED_BINDING
#if TEST_ASTS != 4u
#error "SECANT_TEST_LM_REPEATED_BINDING requires TEST_ASTS=4"
#endif
    static const SecantAstInstruction* const asts[TEST_ASTS] = {
        test_mixed_binding_ast,
        test_mixed_binding_ast,
        test_mixed_binding_ast,
        test_mixed_binding_ast
    };
#else
    static const SecantAstInstruction* const asts[TEST_ASTS] = {
        test_static_binding_ast,
        test_mixed_binding_ast
    };
#endif
    static const char* const nvrtc_options[] = {
        "--restrict",
        "--use_fast_math"
    };
    const size_t mask_ld = TEST_SETTINGS + 3u;
    const size_t words_ld = TEST_PARAMETERS;
    const size_t constants_ld = TEST_PARAMETERS;
    const size_t statistics_ld = TEST_SETTINGS + 5u;
    const size_t state_ld = TEST_SETTINGS + 3u;
    const size_t setting_rows = TEST_ASTS * TEST_SETTINGS;
    float* input = NULL;
    float* target = NULL;
    uint32_t* masks = NULL;
    uint32_t* words = NULL;
    float* constants = NULL;
    float* damping = NULL;
    float* actual_constants = NULL;
    float* accepted = NULL;
    uint32_t* status = NULL;
#ifdef SECANT_TEST_PTX_LM_OPTIMIZER
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXLMOptimizerRunner runner = NULL;
#elif defined(SECANT_TEST_SASS_LM_OPTIMIZER)
    SecantCubinLMOptimizerRunner runner = NULL;
#else
    SecantCUDALMOptimizerRunner runner = NULL;
#endif
    SecantCUDALMOptimizerRun run = secant_cuda_lm_optimizer_run_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    TestCUDA cuda;
    size_t ast;
    size_t setting;
    size_t row;
    int success = 1;

    input = (float*)calloc(TEST_COLUMNS * TEST_ROWS, sizeof(*input));
    target = (float*)calloc(TEST_ROWS, sizeof(*target));
    masks = (uint32_t*)calloc(TEST_ASTS * mask_ld, sizeof(*masks));
    words = (uint32_t*)calloc(setting_rows * words_ld, sizeof(*words));
    constants = (float*)calloc(setting_rows * constants_ld, sizeof(*constants));
    damping = (float*)calloc(TEST_ASTS * state_ld, sizeof(*damping));
    actual_constants = (float*)calloc(setting_rows * constants_ld, sizeof(*actual_constants));
    accepted = (float*)calloc(
        TEST_ASTS * TEST_STATISTICS * statistics_ld,
        sizeof(*accepted));
    status = (uint32_t*)calloc(TEST_ASTS * state_ld, sizeof(*status));
    if (input == NULL || target == NULL || masks == NULL || words == NULL ||
        constants == NULL || damping == NULL || actual_constants == NULL ||
        accepted == NULL || status == NULL) {
        success = 0;
    }
    for (row = 0u; row < TEST_ROWS && success; ++row) {
        const float x0 = -1.0f + 2.0f * (float)row / (float)(TEST_ROWS - 1u);
        const float x1 = 0.35f * sinf(0.071f * (float)row) +
            0.15f * cosf(0.037f * (float)row);

        input[row] = x0;
        input[TEST_ROWS + row] = x1;
        target[row] = 2.5f * x0 + x1 + 0.75f;
    }
    for (ast = 0u; ast < TEST_ASTS && success; ++ast) {
        for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
            const size_t setting_row = ast * TEST_SETTINGS + setting;

            constants[setting_row * constants_ld] =
                -4.0f + 0.17f * (float)setting + 0.25f * (float)ast;
            constants[setting_row * constants_ld + 1u] =
                2.0f - 0.11f * (float)setting;
            damping[ast * state_ld + setting] = 1.0e-3f;
#ifdef SECANT_TEST_LM_REPEATED_BINDING
            masks[ast * mask_ld + setting] = 1u << 1u;
            words[setting_row * words_ld + 1u] = 1u;
#else
            if (ast == 1u) {
                masks[ast * mask_ld + setting] = 1u << 1u;
                words[setting_row * words_ld + 1u] = 1u;
            }
#endif
        }
    }

    memset(&cuda, 0, sizeof(cuda));
    if (success && !test_cuda_create(&cuda)) {
        success = 0;
    }
    if (success) {
#ifdef SECANT_TEST_PTX_LM_OPTIMIZER
        const SecantPTXResult create_result = secant_ptx_lm_optimizer_create(
            TEST_ASTS,
            TEST_CAPACITY_COLUMNS,
            TEST_CAPACITY_COLUMNS,
            TEST_TILE_ROWS,
            TEST_THREADS,
            (uint32_t)cuda.major,
            (uint32_t)cuda.minor,
            nvrtc_options,
            sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
            false,
            NULL,
            0u,
            NULL,
            &ptx_handle);

        if (create_result != SECANT_PTX_SUCCESS) {
            fprintf(
                stderr,
                "PTX LM template create failed: %s\n",
                secant_ptx_result_to_string(create_result));
            success = 0;
        }
#endif
    }
    if (success &&
#ifdef SECANT_TEST_SASS_LM_OPTIMIZER
        secant_cubin_lm_optimizer_runner_create(
            TEST_ASTS,
            TEST_CAPACITY_COLUMNS,
            TEST_CAPACITY_COLUMNS,
            TEST_ACTIVE_PARAMETERS,
            TEST_TILE_ROWS,
            TEST_THREADS,
            2048u,
            (uint32_t)cuda.major,
            (uint32_t)cuda.minor,
            nvrtc_options,
            sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
            TEST_SCRATCH_BYTES,
            TEST_ASTS,
            &runner)
#elif defined(SECANT_TEST_PTX_LM_OPTIMIZER)
        secant_ptx_lm_optimizer_runner_create(
            ptx_handle,
            (uint32_t)cuda.major,
            (uint32_t)cuda.minor,
            NULL,
            0u,
            TEST_SCRATCH_BYTES,
            TEST_ASTS,
            &runner)
#else
        secant_cuda_lm_optimizer_runner_create(
            TEST_ASTS,
            TEST_CAPACITY_COLUMNS,
            TEST_CAPACITY_COLUMNS,
            TEST_TILE_ROWS,
            TEST_THREADS,
            (uint32_t)cuda.major,
            (uint32_t)cuda.minor,
            nvrtc_options,
            sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
            TEST_SCRATCH_BYTES,
            TEST_ASTS,
            &runner)
#endif
        != SECANT_SUCCESS) {
        success = 0;
    }
    if (success) {
#ifdef SECANT_TEST_SASS_LM_SMOOTH_OPS
        static const SecantAstInstruction* const routines[] = {
            test_smooth_routine
        };

        run.routines.items = routines;
        run.routines.count = sizeof(routines) / sizeof(routines[0]);
#else
        run.routines.items = NULL;
        run.routines.count = 0u;
#endif
        run.asts.items = asts;
        run.asts.count = TEST_ASTS;
        run.input.address = test_device_copy(&cuda, input, TEST_COLUMNS * TEST_ROWS * sizeof(float));
        run.input.num_elements = TEST_COLUMNS * TEST_ROWS;
        run.input.leading_dimension = TEST_ROWS;
        run.num_input_columns = TEST_COLUMNS;
        run.leaf_masks.address = test_device_copy(
            &cuda, masks, TEST_ASTS * mask_ld * sizeof(uint32_t));
        run.leaf_masks.num_elements = TEST_ASTS * mask_ld;
        run.leaf_masks.leading_dimension = mask_ld;
        run.leaf_words.address = test_device_copy(
            &cuda, words, setting_rows * words_ld * sizeof(uint32_t));
        run.leaf_words.num_elements = setting_rows * words_ld;
        run.leaf_words.leading_dimension = words_ld;
        run.target.address = test_device_copy(&cuda, target, TEST_ROWS * sizeof(float));
        run.target.num_elements = TEST_ROWS;
        run.target.leading_dimension = TEST_ROWS;
        run.current_constants.address = test_device_copy(
            &cuda, constants, setting_rows * constants_ld * sizeof(float));
        run.current_constants.num_elements = setting_rows * constants_ld;
        run.current_constants.leading_dimension = constants_ld;
        run.proposal_constants.address = test_device_copy(
            &cuda, NULL, setting_rows * constants_ld * sizeof(float));
        run.proposal_constants.num_elements = setting_rows * constants_ld;
        run.proposal_constants.leading_dimension = constants_ld;
        run.evaluated_statistics.address = test_device_copy(
            &cuda, NULL, TEST_ASTS * TEST_STATISTICS * statistics_ld * sizeof(float));
        run.evaluated_statistics.num_elements =
            TEST_ASTS * TEST_STATISTICS * statistics_ld;
        run.evaluated_statistics.leading_dimension = statistics_ld;
        run.accepted_statistics.address = test_device_copy(
            &cuda, NULL, TEST_ASTS * TEST_STATISTICS * statistics_ld * sizeof(float));
        run.accepted_statistics.num_elements =
            TEST_ASTS * TEST_STATISTICS * statistics_ld;
        run.accepted_statistics.leading_dimension = statistics_ld;
        run.damping.address = test_device_copy(
            &cuda, damping, TEST_ASTS * state_ld * sizeof(float));
        run.damping.num_elements = TEST_ASTS * state_ld;
        run.damping.leading_dimension = state_ld;
        run.predicted_reduction.address = test_device_copy(
            &cuda, NULL, TEST_ASTS * state_ld * sizeof(float));
        run.predicted_reduction.num_elements = TEST_ASTS * state_ld;
        run.predicted_reduction.leading_dimension = state_ld;
        run.status.address = test_device_copy(
            &cuda, NULL, TEST_ASTS * state_ld * sizeof(uint32_t));
        run.status.num_elements = TEST_ASTS * state_ld;
        run.status.leading_dimension = state_ld;
        run.num_rows = TEST_ROWS;
        run.num_settings = TEST_SETTINGS;
        run.settings_per_cta = TEST_SETTINGS_PER_CTA;
        run.num_iterations = TEST_ITERATIONS;
        if (run.input.address == 0u || run.leaf_masks.address == 0u ||
            run.leaf_words.address == 0u || run.target.address == 0u ||
            run.current_constants.address == 0u ||
            run.proposal_constants.address == 0u ||
            run.evaluated_statistics.address == 0u ||
            run.accepted_statistics.address == 0u || run.damping.address == 0u ||
            run.predicted_reduction.address == 0u || run.status.address == 0u) {
            success = 0;
        }
    }
    if (success) {
        const SecantResult result =
#ifdef SECANT_TEST_SASS_LM_OPTIMIZER
            secant_cubin_lm_optimizer_runner_run(
#elif defined(SECANT_TEST_PTX_LM_OPTIMIZER)
            secant_ptx_lm_optimizer_runner_run(
#else
            secant_cuda_lm_optimizer_runner_run(
#endif
            runner,
            &run,
            &stats);

        if (result != SECANT_SUCCESS) {
            fprintf(stderr, "LM optimizer run failed: %s\n", secant_result_to_string(result));
            success = 0;
        }
    }
    if (success &&
        (cuMemcpyDtoH(
             actual_constants,
             (CUdeviceptr)run.current_constants.address,
             setting_rows * constants_ld * sizeof(float)) != CUDA_SUCCESS ||
         cuMemcpyDtoH(
             accepted,
             (CUdeviceptr)run.accepted_statistics.address,
             TEST_ASTS * TEST_STATISTICS * statistics_ld * sizeof(float)) != CUDA_SUCCESS ||
         cuMemcpyDtoH(
             status,
             (CUdeviceptr)run.status.address,
             TEST_ASTS * state_ld * sizeof(uint32_t)) != CUDA_SUCCESS)) {
        success = 0;
    }
    for (ast = 0u; ast < TEST_ASTS && success; ++ast) {
        for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
            const size_t setting_row = ast * TEST_SETTINGS + setting;
            const float slope = actual_constants[setting_row * constants_ld];
            const float intercept = actual_constants[setting_row * constants_ld + 1u];
            const float sse = accepted[ast * TEST_STATISTICS * statistics_ld + setting];

            if (!isfinite(slope) || !isfinite(intercept) || !isfinite(sse) ||
                fabsf(slope - 2.5f) > 2.0e-3f ||
                fabsf(intercept - 0.75f) > 2.0e-3f || sse > 2.0e-4f ||
                (status[ast * state_ld + setting] & 4u) != 0u) {
                fprintf(
                    stderr,
                    "bad fit ast=%zu setting=%zu slope=%.9g intercept=%.9g "
                    "sse=%.9g status=0x%x\n",
                    ast,
                    setting,
                    slope,
                    intercept,
                    sse,
                    status[ast * state_ld + setting]);
                success = 0;
                break;
            }
        }
    }
#ifdef SECANT_TEST_REPORT_FITS
    if (success) {
        for (ast = 0u; ast < TEST_ASTS; ++ast) {
            const size_t setting_row = ast * TEST_SETTINGS;
            const float sse =
                accepted[ast * TEST_STATISTICS * statistics_ld];

            printf(
                "ast=%zu binding=mixed[1]->column[1] "
                "start=(%.6g,%.6g) fit=(%.9g,%.9g) sse=%.9g\n",
                ast,
                constants[setting_row * constants_ld],
                constants[setting_row * constants_ld + 1u],
                actual_constants[setting_row * constants_ld],
                actual_constants[setting_row * constants_ld + 1u],
                sse);
        }
    }
#endif
    if (success &&
        (stats.modules_loaded != 1u || stats.num_asts != TEST_ASTS ||
         !(stats.runtime_seconds > 0.0))) {
        fprintf(stderr, "unexpected LM runner statistics\n");
        success = 0;
    }
#ifdef SECANT_TEST_REPORT
    if (success) {
        const double row_evaluations =
            (double)TEST_ASTS * (double)TEST_SETTINGS * (double)TEST_ROWS *
            (double)(TEST_ITERATIONS + 1u);

        printf(
            "backend=%s shape=lm_optimizer asts=%u rows=%u settings=%u "
            "settings_per_cta=%u tile_rows=%u threads=%u lm_iterations=%u "
            "compile_ms=%.6f runtime_ms=%.6f grow_evals_per_s=%.6f\n",
#ifdef SECANT_TEST_PTX_LM_OPTIMIZER
            "ptx",
#elif defined(SECANT_TEST_SASS_LM_OPTIMIZER)
            "sass",
#else
            "cuda",
#endif
            (unsigned)TEST_ASTS,
            (unsigned)TEST_ROWS,
            (unsigned)TEST_SETTINGS,
            (unsigned)TEST_SETTINGS_PER_CTA,
            (unsigned)TEST_TILE_ROWS,
            (unsigned)TEST_THREADS,
            (unsigned)TEST_ITERATIONS,
            stats.compile_window_seconds * 1.0e3,
            stats.runtime_seconds * 1.0e3,
            row_evaluations / stats.runtime_seconds * 1.0e-9);
    }
#endif
    if (runner != NULL &&
#ifdef SECANT_TEST_SASS_LM_OPTIMIZER
        secant_cubin_lm_optimizer_runner_destroy(runner)
#elif defined(SECANT_TEST_PTX_LM_OPTIMIZER)
        secant_ptx_lm_optimizer_runner_destroy(runner)
#else
        secant_cuda_lm_optimizer_runner_destroy(runner)
#endif
        != SECANT_SUCCESS) {
        success = 0;
    }
#ifdef SECANT_TEST_PTX_LM_OPTIMIZER
    if (ptx_handle != NULL &&
        secant_ptx_handle_destroy(ptx_handle) != SECANT_PTX_SUCCESS) {
        success = 0;
    }
#endif
    test_cuda_destroy(&cuda);
    free(status);
    free(accepted);
    free(actual_constants);
    free(damping);
    free(constants);
    free(words);
    free(masks);
    free(target);
    free(input);
    return success ? 0 : 1;
}
