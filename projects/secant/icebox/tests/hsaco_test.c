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
#include "secant_cpu.h"
#include "secant_hsaco.h"
#include "secant_hsaco_runner.h"
#include "support/hip_runtime.h"

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ROWS 257u
#define TEST_INPUTS 4u
#define TEST_TARGETS 2u
#define TEST_KERNELS 2u
#define TEST_ASTS_PER_KERNEL 4u
#define TEST_ASTS (TEST_KERNELS * TEST_ASTS_PER_KERNEL)
#define TEST_PATCH_INSTRUCTIONS 64u
#define TEST_EXPANSION_ROWS 131072u
#define TEST_EXPANSION_VALUES 64u
#define TEST_EXPANSION_INSTRUCTIONS \
    (TEST_EXPANSION_VALUES * 2u)
#define TEST_EXPANSION_PATCH_INSTRUCTIONS 256u

static const SecantAstInstruction test_safe_div[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32(1.0e-6f),
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
    secant_ast_encode_routine_f32(0u, 2u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_bad_routine_arity[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_routine_f32(0u, 1u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_zero_arg_routine[] = {
    secant_ast_encode_constant_f32(2.0f),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_zero_arg_call[] = {
    secant_ast_encode_routine_f32(0u, 0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sqrt_rsqrt[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32(1.0f),
    secant_ast_encode_add_f32,
    secant_ast_encode_rsqrt_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_ex2_lg2[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_ex2_f32,
    secant_ast_encode_static_column_input_f32(3u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32(1.0f),
    secant_ast_encode_add_f32,
    secant_ast_encode_lg2_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_tanh[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_sub_f32,
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
    secant_ast_encode_constant_f32(2.5f),
    secant_ast_encode_return_f32
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
    test_ex2_lg2,
    test_tanh,
    test_min_max
};

typedef struct TestState {
    hipStream_t streams[2];
    float* input;
    float* targets;
    float* materialize_output;
    float* sse_output;
    SecantHsacoPlan* materialize_handle;
    SecantHsacoPlan* sse_handle;
    void* materialize_workspace;
    void* sse_workspace;
    unsigned char* materialize_hsaco;
    unsigned char* sse_hsaco;
    size_t materialize_hsaco_size;
    size_t sse_hsaco_size;
    void* materialize_module;
    void* sse_module;
    void* materialize_functions[TEST_KERNELS];
    void* sse_functions[TEST_KERNELS];
} TestState;

typedef struct TestExpansionState {
    SecantHsacoPlan* handle;
    void* workspace;
    unsigned char* hsaco;
    size_t hsaco_size;
    void* module;
    void* functions[1];
    float* device_input;
    float* device_output;
    float* input;
    float* expected;
    float* actual;
} TestExpansionState;

static void
test_print_error(const char* operation, SecantResult result) {
    fprintf(
        stderr,
        "%s failed: %s\n",
        operation,
        secant_result_to_string(result));
}

static int
test_close(float actual, float expected, float tolerance) {
    return isfinite(actual) && isfinite(expected) &&
        fabsf(actual - expected) <=
            tolerance * (1.0f + fabsf(expected));
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

static int
test_contains_native_fma(
    const unsigned char* hsaco,
    size_t hsaco_size
) {
    size_t offset;

    for (offset = 0u;
         offset + sizeof(uint32_t) <= hsaco_size;
         offset += sizeof(uint32_t)) {
        uint32_t word;

        memcpy(&word, hsaco + offset, sizeof(word));
        if ((word & 0xffffff00u) == 0xd6130000u) {
            return 1;
        }
    }
    return 0;
}

static void
test_state_destroy(TestState* state) {
    if (state->materialize_module != NULL) {
        (void)hipModuleUnload((hipModule_t)state->materialize_module);
    }
    if (state->sse_module != NULL) {
        (void)hipModuleUnload((hipModule_t)state->sse_module);
    }
    free(state->materialize_workspace);
    free(state->sse_workspace);
    free(state->materialize_hsaco);
    free(state->sse_hsaco);
    if (state->input != NULL) {
        (void)hipFree(state->input);
    }
    if (state->targets != NULL) {
        (void)hipFree(state->targets);
    }
    if (state->materialize_output != NULL) {
        (void)hipFree(state->materialize_output);
    }
    if (state->sse_output != NULL) {
        (void)hipFree(state->sse_output);
    }
    if (state->streams[0] != NULL) {
        (void)hipStreamDestroy(state->streams[0]);
    }
    if (state->streams[1] != NULL) {
        (void)hipStreamDestroy(state->streams[1]);
    }
}

static int
test_architecture(uint32_t* gfx_arch_ret) {
    hipDeviceProp_t properties;
    char* end;
    unsigned long gfx_arch;

    if (hipSetDevice(0) != hipSuccess ||
        hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        strncmp(properties.gcnArchName, "gfx", 3u) != 0) {
        return 0;
    }
    gfx_arch = strtoul(properties.gcnArchName + 3u, &end, 10);
    if (end == properties.gcnArchName + 3u ||
        gfx_arch > UINT32_MAX) {
        return 0;
    }
    *gfx_arch_ret = (uint32_t)gfx_arch;
    return 1;
}

static int
test_compile_template(
    int sse,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    uint32_t gfx_arch,
    unsigned char** hsaco_ret,
    size_t* hsaco_size_ret
) {
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
        SECANT_SSE_REDUCTION_MODE_ATOMIC
    };
    char architecture[64];
    const char* options[3];
    char* source = NULL;
    unsigned char* hsaco = NULL;
    char* log = NULL;
    hiprtcProgram program = NULL;
    hiprtcResult hiprtc_result;
    SecantResult result;
    size_t source_size = 0u;
    size_t written_size = 0u;
    size_t hsaco_size = 0u;
    size_t log_size = 0u;
    int success = 0;

    *hsaco_ret = NULL;
    *hsaco_size_ret = 0u;
    result = sse
        ? secant_hsaco_sse_source_generate(
            &sse_recipe,
            NULL,
            0u,
            &source_size)
        : secant_hsaco_materialize_source_generate(
            &materialize_recipe,
            NULL,
            0u,
            &source_size);
    if (result != SECANT_SUCCESS || source_size == 0u) {
        test_print_error("HIP source measure", result);
        return 0;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        return 0;
    }
    result = sse
        ? secant_hsaco_sse_source_generate(
            &sse_recipe,
            source,
            source_size,
            &written_size)
        : secant_hsaco_materialize_source_generate(
            &materialize_recipe,
            source,
            source_size,
            &written_size);
    if (result != SECANT_SUCCESS ||
        written_size != source_size ||
        snprintf(
            architecture,
            sizeof(architecture),
            "--offload-arch=gfx%u",
            gfx_arch) < 0) {
        test_print_error("HIP source generation", result);
        free(source);
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "-O1";
    hiprtc_result = hiprtcCreateProgram(
        &program,
        source,
        "secant_hsaco_test.hip",
        0,
        NULL,
        NULL);
    if (hiprtc_result == HIPRTC_SUCCESS) {
        hiprtc_result = hiprtcCompileProgram(
            program,
            3,
            options);
    }
    if (hiprtc_result != HIPRTC_SUCCESS) {
        if (program != NULL &&
            hiprtcGetProgramLogSize(program, &log_size) ==
                HIPRTC_SUCCESS &&
            log_size != 0u) {
            log = (char*)malloc(log_size);
            if (log != NULL &&
                hiprtcGetProgramLog(program, log) == HIPRTC_SUCCESS) {
                fprintf(stderr, "%s\n", log);
            }
        }
    } else if (
        hiprtcGetCodeSize(program, &hsaco_size) != HIPRTC_SUCCESS ||
        hsaco_size == 0u) {
        fprintf(stderr, "HIPRTC returned an empty HSACO\n");
    } else {
        hsaco = (unsigned char*)malloc(hsaco_size);
        if (hsaco != NULL &&
            hiprtcGetCode(program, (char*)hsaco) == HIPRTC_SUCCESS) {
            success = 1;
        }
    }
    if (program != NULL) {
        (void)hiprtcDestroyProgram(&program);
    }
    free(log);
    free(source);
    if (!success) {
        free(hsaco);
        return 0;
    }
    *hsaco_ret = hsaco;
    *hsaco_size_ret = hsaco_size;
    return 1;
}

static int
test_inspect_templates(TestState* state, uint32_t gfx_arch) {
    const SecantHsacoMaterializeRecipe materialize_recipe = {
        TEST_KERNELS,
        TEST_ASTS_PER_KERNEL,
        TEST_INPUTS,
        TEST_PATCH_INSTRUCTIONS,
        SECANT_SSE_REDUCTION_MODE_ATOMIC
    };
    const SecantHsacoSSERecipe sse_recipe = {
        TEST_KERNELS,
        TEST_ASTS_PER_KERNEL,
        TEST_INPUTS,
        TEST_TARGETS,
        128u,
        128u,
        TEST_PATCH_INSTRUCTIONS,
        SECANT_SSE_REDUCTION_MODE_ATOMIC
    };
    SecantResult result;
    size_t workspace_size = 0u;
    SecantHsacoPlan* invalid_handle = NULL;
    unsigned char* malformed_hsaco;
    unsigned char* misaligned_hsaco;

    if (!test_compile_template(
            0,
            TEST_KERNELS,
            TEST_ASTS_PER_KERNEL,
            TEST_INPUTS,
            0u,
            0u,
            0u,
            TEST_PATCH_INSTRUCTIONS,
            gfx_arch,
            &state->materialize_hsaco,
            &state->materialize_hsaco_size) ||
        !test_compile_template(
            1,
            TEST_KERNELS,
            TEST_ASTS_PER_KERNEL,
            TEST_INPUTS,
            TEST_TARGETS,
            128u,
            128u,
            TEST_PATCH_INSTRUCTIONS,
            gfx_arch,
            &state->sse_hsaco,
            &state->sse_hsaco_size)) {
        return 0;
    }
    malformed_hsaco = (unsigned char*)malloc(state->materialize_hsaco_size);
    if (malformed_hsaco == NULL) {
        return 0;
    }
    memcpy(
        malformed_hsaco,
        state->materialize_hsaco,
        state->materialize_hsaco_size);
    malformed_hsaco[0] ^= 1u;
    result = secant_hsaco_materialize_inspect(
        &materialize_recipe,
        malformed_hsaco,
        state->materialize_hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &invalid_handle);
    free(malformed_hsaco);
    if (result != SECANT_ERROR_PARSE_FAILED || invalid_handle != NULL) {
        test_print_error("materialize inspect malformed ELF", result);
        return 0;
    }
    workspace_size = 0u;
    misaligned_hsaco = (unsigned char*)malloc(
        state->materialize_hsaco_size + 1u);
    if (misaligned_hsaco == NULL) {
        return 0;
    }
    memcpy(
        misaligned_hsaco + 1u,
        state->materialize_hsaco,
        state->materialize_hsaco_size);
    result = secant_hsaco_materialize_inspect(
        &materialize_recipe,
        misaligned_hsaco + 1u,
        state->materialize_hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &invalid_handle);
    free(misaligned_hsaco);
    if (result != SECANT_ERROR_INVALID_VALUE || invalid_handle != NULL) {
        test_print_error("materialize inspect alignment", result);
        return 0;
    }
    workspace_size = 0u;
    result = secant_hsaco_materialize_inspect(
        &materialize_recipe,
        state->materialize_hsaco,
        state->materialize_hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &state->materialize_handle);
    if (result != SECANT_SUCCESS || workspace_size == 0u) {
        test_print_error("materialize inspect measure", result);
        return 0;
    }
    state->materialize_workspace = malloc(workspace_size);
    if (state->materialize_workspace == NULL) {
        return 0;
    }
    result = secant_hsaco_materialize_inspect(
        &materialize_recipe,
        state->materialize_hsaco,
        state->materialize_hsaco_size,
        state->materialize_workspace,
        workspace_size,
        &workspace_size,
        &state->materialize_handle);
    if (result != SECANT_SUCCESS) {
        test_print_error("materialize inspect", result);
        return 0;
    }
    workspace_size = 0u;
    result = secant_hsaco_sse_inspect(
        &sse_recipe,
        state->sse_hsaco,
        state->sse_hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &state->sse_handle);
    if (result != SECANT_SUCCESS || workspace_size == 0u) {
        test_print_error("SSE inspect measure", result);
        return 0;
    }
    state->sse_workspace = malloc(workspace_size);
    if (state->sse_workspace == NULL) {
        return 0;
    }
    result = secant_hsaco_sse_inspect(
        &sse_recipe,
        state->sse_hsaco,
        state->sse_hsaco_size,
        state->sse_workspace,
        workspace_size,
        &workspace_size,
        &state->sse_handle);
    if (result != SECANT_SUCCESS) {
        test_print_error("SSE inspect", result);
        return 0;
    }
    return 1;
}

static int
test_specialize_templates(TestState* state) {
    const SecantAstInstruction* bad_asts[TEST_ASTS];
    const SecantAstInstruction* zero_arg_asts[TEST_ASTS];
    const SecantAstInstruction* zero_arg_routines[] = {
        test_zero_arg_routine
    };
    unsigned char* validation_hsaco;
    SecantResult result;
    size_t ast_idx;

    validation_hsaco = (unsigned char*)malloc(
        state->materialize_hsaco_size);
    if (validation_hsaco == NULL) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < TEST_ASTS; ++ast_idx) {
        bad_asts[ast_idx] =
            ast_idx == 0u ? test_bad_routine_arity : test_add;
        zero_arg_asts[ast_idx] = test_zero_arg_call;
    }
    memcpy(
        validation_hsaco,
        state->materialize_hsaco,
        state->materialize_hsaco_size);
    result = secant_hsaco_specialize_into(
        state->materialize_handle,
        test_routines,
        1u,
        bad_asts,
        TEST_ASTS,
        validation_hsaco,
        state->materialize_hsaco_size);
    if (result != SECANT_ERROR_INVALID_VALUE) {
        free(validation_hsaco);
        test_print_error("routine arity validation", result);
        return 0;
    }
    memcpy(
        validation_hsaco,
        state->materialize_hsaco,
        state->materialize_hsaco_size);
    result = secant_hsaco_specialize_into(
        state->materialize_handle,
        zero_arg_routines,
        1u,
        zero_arg_asts,
        TEST_ASTS,
        validation_hsaco,
        state->materialize_hsaco_size);
    free(validation_hsaco);
    if (result != SECANT_SUCCESS) {
        test_print_error("zero-argument routine", result);
        return 0;
    }

    if (test_contains_native_fma(
            state->materialize_hsaco,
            state->materialize_hsaco_size)) {
        fprintf(stderr, "template HSACO unexpectedly contains native FMA\n");
        return 0;
    }
    result = secant_hsaco_specialize_into(
        state->materialize_handle,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        state->materialize_hsaco,
        state->materialize_hsaco_size);
    if (result != SECANT_SUCCESS) {
        test_print_error("materialize specialize", result);
        return 0;
    }
    if (!test_contains_native_fma(
            state->materialize_hsaco,
            state->materialize_hsaco_size)) {
        fprintf(stderr, "materialize HSACO does not contain native FMA\n");
        return 0;
    }
    result = secant_hsaco_specialize_into(
        state->sse_handle,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        state->sse_hsaco,
        state->sse_hsaco_size);
    if (result != SECANT_SUCCESS) {
        test_print_error("SSE specialize", result);
        return 0;
    }
    return 1;
}

static void
test_expansion_state_destroy(TestExpansionState* state) {
    if (state->module != NULL) {
        (void)hipModuleUnload((hipModule_t)state->module);
    }
    if (state->device_input != NULL) {
        (void)hipFree(state->device_input);
    }
    if (state->device_output != NULL) {
        (void)hipFree(state->device_output);
    }
    free(state->actual);
    free(state->expected);
    free(state->input);
    free(state->workspace);
    free(state->hsaco);
}

static int
test_register_expansion_run(
    TestExpansionState* state,
    uint32_t gfx_arch
) {
    const SecantHsacoMaterializeRecipe recipe = {
        1u,
        1u,
        1u,
        TEST_EXPANSION_PATCH_INSTRUCTIONS
    };
    SecantAstInstruction program[TEST_EXPANSION_INSTRUCTIONS];
    const SecantAstInstruction* asts[] = { program };
    const size_t bytes = TEST_EXPANSION_ROWS * sizeof(float);
    size_t workspace_size = 0u;
    size_t idx;
    SecantResult result;
    SecantResult hip_result;

    for (idx = 0u; idx < TEST_EXPANSION_VALUES; ++idx) {
        program[idx] = secant_ast_encode_constant_f32(
            0.01f * (float)(idx + 1u));
    }
    for (idx = 0u; idx + 1u < TEST_EXPANSION_VALUES; ++idx) {
        program[TEST_EXPANSION_VALUES + idx] =
            secant_ast_encode_add_f32;
    }
    program[TEST_EXPANSION_INSTRUCTIONS - 1u] =
        secant_ast_encode_return_f32;
    state->input = (float*)calloc(
        TEST_EXPANSION_ROWS,
        sizeof(*state->input));
    state->expected = (float*)malloc(bytes);
    state->actual = (float*)malloc(bytes);
    if (state->input == NULL || state->expected == NULL ||
        state->actual == NULL) {
        return 0;
    }
    if (secant_cpu_materialize_run(
            1u,
            NULL,
            0u,
            asts,
            1u,
            state->input,
            TEST_EXPANSION_ROWS,
            TEST_EXPANSION_ROWS,
            TEST_EXPANSION_ROWS,
            state->expected,
            TEST_EXPANSION_ROWS,
            TEST_EXPANSION_ROWS) != SECANT_SUCCESS) {
        return 0;
    }
    if (!test_compile_template(
            0,
            1u,
            1u,
            1u,
            0u,
            0u,
            0u,
            TEST_EXPANSION_PATCH_INSTRUCTIONS,
            gfx_arch,
            &state->hsaco,
            &state->hsaco_size)) {
        return 0;
    }
    result = secant_hsaco_materialize_inspect(
        &recipe,
        state->hsaco,
        state->hsaco_size,
        NULL,
        0u,
        &workspace_size,
        &state->handle);
    if (result != SECANT_SUCCESS || workspace_size == 0u) {
        test_print_error("expansion inspect measure", result);
        return 0;
    }
    state->workspace = malloc(workspace_size);
    if (state->workspace == NULL) {
        return 0;
    }
    result = secant_hsaco_materialize_inspect(
        &recipe,
        state->hsaco,
        state->hsaco_size,
        state->workspace,
        workspace_size,
        &workspace_size,
        &state->handle);
    if (result != SECANT_SUCCESS) {
        test_print_error("expansion inspect", result);
        return 0;
    }
    result = secant_hsaco_specialize_into(
        state->handle,
        NULL,
        0u,
        asts,
        1u,
        state->hsaco,
        state->hsaco_size);
    if (result != SECANT_SUCCESS) {
        test_print_error("expansion specialize", result);
        return 0;
    }
    hip_result = secant_test_hip_module_load(
        state->hsaco,
        state->hsaco_size,
        "secant_hsaco_materialize",
        1u,
        &state->module,
        state->functions);
    if (hip_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "expansion load failed: %s\n",
            secant_test_hip_result_to_string(hip_result));
        return 0;
    }
    if (hipMalloc((void**)&state->device_input, bytes) != hipSuccess ||
        hipMalloc((void**)&state->device_output, bytes) != hipSuccess ||
        hipMemcpy(
            state->device_input,
            state->input,
            bytes,
            hipMemcpyHostToDevice) != hipSuccess) {
        return 0;
    }
    hip_result = secant_test_hip_run_static_column_materialize(
        state->functions[0],
        1u,
        1u,
        state->device_input,
        TEST_EXPANSION_ROWS,
        TEST_EXPANSION_ROWS,
        TEST_EXPANSION_ROWS,
        NULL,
        state->device_output,
        TEST_EXPANSION_ROWS,
        TEST_EXPANSION_ROWS);
    if (hip_result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "expansion run failed: %s\n",
            secant_test_hip_result_to_string(hip_result));
        return 0;
    }
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(
            state->actual,
            state->device_output,
            bytes,
            hipMemcpyDeviceToHost) != hipSuccess) {
        return 0;
    }
    return test_compare(
        "register expansion",
        state->actual,
        state->expected,
        TEST_EXPANSION_ROWS,
        1.0e-5f);
}

static int
test_register_expansion(uint32_t gfx_arch) {
    TestExpansionState state;
    int success;

    memset(&state, 0, sizeof(state));
    success = test_register_expansion_run(&state, gfx_arch);
    test_expansion_state_destroy(&state);
    return success;
}

static int
test_load(TestState* state) {
    SecantResult result;

    result = secant_test_hip_module_load(
        state->materialize_hsaco,
        state->materialize_hsaco_size,
        "secant_hsaco_materialize",
        TEST_KERNELS,
        &state->materialize_module,
        state->materialize_functions);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "materialize load failed: %s\n",
            secant_test_hip_result_to_string(result));
        return 0;
    }
    result = secant_test_hip_module_load(
        state->sse_hsaco,
        state->sse_hsaco_size,
        "secant_hsaco_sse",
        TEST_KERNELS,
        &state->sse_module,
        state->sse_functions);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "SSE load failed: %s\n",
            secant_test_hip_result_to_string(result));
        return 0;
    }
    return 1;
}

static int
test_run(
    TestState* state,
    const float* input,
    const float* targets,
    float* materialize_output,
    float* sse_output
) {
    const size_t input_bytes =
        TEST_INPUTS * TEST_ROWS * sizeof(float);
    const size_t target_bytes =
        TEST_TARGETS * TEST_ROWS * sizeof(float);
    const size_t materialize_bytes =
        TEST_ASTS * TEST_ROWS * sizeof(float);
    const size_t sse_bytes =
        TEST_ASTS * TEST_TARGETS * sizeof(float);
    SecantResult result;
    size_t kernel_idx;

    if (hipStreamCreateWithFlags(
            state->streams,
            hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(
            state->streams + 1u,
            hipStreamNonBlocking) != hipSuccess ||
        hipMalloc((void**)&state->input, input_bytes) != hipSuccess ||
        hipMalloc((void**)&state->targets, target_bytes) != hipSuccess ||
        hipMalloc(
            (void**)&state->materialize_output,
            materialize_bytes) != hipSuccess ||
        hipMalloc((void**)&state->sse_output, sse_bytes) != hipSuccess ||
        hipMemcpy(
            state->input,
            input,
            input_bytes,
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(
            state->targets,
            targets,
            target_bytes,
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(
            state->materialize_output,
            0,
            materialize_bytes) != hipSuccess ||
        hipMemset(state->sse_output, 0, sse_bytes) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
        return 0;
    }
    for (kernel_idx = 0u; kernel_idx < TEST_KERNELS; ++kernel_idx) {
        result = secant_test_hip_run_static_column_materialize(
            state->materialize_functions[kernel_idx],
            TEST_INPUTS,
            TEST_ASTS_PER_KERNEL,
            state->input,
            TEST_INPUTS * TEST_ROWS,
            TEST_ROWS,
            TEST_ROWS,
            state->streams[kernel_idx % 2u],
            state->materialize_output +
                kernel_idx * TEST_ASTS_PER_KERNEL * TEST_ROWS,
            TEST_ASTS_PER_KERNEL * TEST_ROWS,
            TEST_ROWS);
        if (result == SECANT_SUCCESS) {
            result = secant_test_hip_run_static_column_sse(
                state->sse_functions[kernel_idx],
                TEST_INPUTS,
                TEST_ASTS_PER_KERNEL,
                TEST_TARGETS,
                128u,
                128u,
                state->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                state->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                state->streams[kernel_idx % 2u],
                state->sse_output +
                    kernel_idx * TEST_ASTS_PER_KERNEL * TEST_TARGETS,
                TEST_ASTS_PER_KERNEL * TEST_TARGETS,
                TEST_TARGETS);
        }
        if (result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "kernel launch failed: %s\n",
                secant_test_hip_result_to_string(result));
            return 0;
        }
    }
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(
            materialize_output,
            state->materialize_output,
            materialize_bytes,
            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(
            sse_output,
            state->sse_output,
            sse_bytes,
            hipMemcpyDeviceToHost) != hipSuccess) {
        return 0;
    }
    return 1;
}

static int
test_runner_run(
    TestState* state,
    const float* expected_materialize,
    const float* expected_sse,
    float* actual_materialize,
    float* actual_sse
) {
    const size_t input_count = TEST_INPUTS * TEST_ROWS;
    const size_t target_count = TEST_TARGETS * TEST_ROWS;
    const size_t materialize_count = TEST_ASTS * TEST_ROWS;
    const size_t sse_count = TEST_ASTS * TEST_TARGETS;
    SecantRunnerStats stats;
    SecantResult result = SECANT_SUCCESS;
    size_t repeat;

    {
        SecantHsacoRunner materialize_runner = NULL;
        SecantHsacoRunner sse_runner = NULL;

        result = secant_hsaco_materialize_runner_create(
            state->materialize_handle,
            state->materialize_hsaco,
            state->materialize_hsaco_size,
            1u,
            2u,
            &materialize_runner);
        if (result == SECANT_SUCCESS) {
            result = secant_hsaco_sse_runner_create(
                state->sse_handle,
                state->sse_hsaco,
                state->sse_hsaco_size,
                1u,
                2u,
                &sse_runner);
        }
        for (repeat = 0u;
             repeat < 2u && result == SECANT_SUCCESS;
             ++repeat) {
            memset(&stats, 0, sizeof(stats));
            result = secant_hsaco_materialize_runner_run_all(
                materialize_runner,
                test_routines,
                1u,
                test_asts,
                TEST_ASTS,
                (uintptr_t)state->input,
                input_count,
                TEST_ROWS,
                TEST_ROWS,
                (uintptr_t)state->materialize_output,
                materialize_count,
                TEST_ROWS,
                0u,
                &stats);
            if (result == SECANT_SUCCESS &&
                (stats.num_modules != 1u ||
                 stats.modules_loaded != 1u ||
                 stats.compile_critical_seconds <= 0.0 ||
                 stats.compile_critical_seconds >
                    stats.compile_work_seconds)) {
                result = SECANT_ERROR_INVALID_STATE;
            }
            if (result == SECANT_SUCCESS) {
                memset(&stats, 0, sizeof(stats));
                result = secant_hsaco_sse_runner_run_all(
                    sse_runner,
                    test_routines,
                    1u,
                    test_asts,
                    TEST_ASTS,
                    (uintptr_t)state->input,
                    input_count,
                    TEST_ROWS,
                    (uintptr_t)state->targets,
                    target_count,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)state->sse_output,
                    sse_count,
                    TEST_TARGETS,
                    &stats);
            }
            if (result == SECANT_SUCCESS &&
                (stats.num_modules != 1u ||
                 stats.modules_loaded != 1u ||
                 stats.compile_critical_seconds <= 0.0 ||
                 stats.compile_critical_seconds >
                    stats.compile_work_seconds)) {
                result = SECANT_ERROR_INVALID_STATE;
            }
        }
        if (sse_runner != NULL) {
            const SecantResult destroy_result =
                secant_hsaco_runner_destroy(sse_runner);

            if (result == SECANT_SUCCESS) {
                result = destroy_result;
            }
        }
        if (materialize_runner != NULL) {
            const SecantResult destroy_result =
                secant_hsaco_runner_destroy(materialize_runner);

            if (result == SECANT_SUCCESS) {
                result = destroy_result;
            }
        }
    }
    if (result != SECANT_SUCCESS) {
        test_print_error("HSACO runner", result);
        return 0;
    }
    if (hipMemcpy(
            actual_materialize,
            state->materialize_output,
            materialize_count * sizeof(float),
            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(
            actual_sse,
            state->sse_output,
            sse_count * sizeof(float),
            hipMemcpyDeviceToHost) != hipSuccess) {
        return 0;
    }
    return test_compare(
            "HSACO runner materialize",
            actual_materialize,
            expected_materialize,
            materialize_count,
            8.0e-4f) &&
        test_compare(
            "HSACO runner SSE",
            actual_sse,
            expected_sse,
            sse_count,
            3.0e-3f);
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
    uint32_t gfx_arch = 0u;
    size_t row;
    int success;

    memset(&state, 0, sizeof(state));
    for (row = 0u; row < TEST_ROWS; ++row) {
        input[row] = 0.5f + 0.002f * (float)row;
        input[TEST_ROWS + row] = 1.1f + 0.001f * (float)row;
        input[2u * TEST_ROWS + row] =
            -0.4f + 0.0015f * (float)row;
        input[3u * TEST_ROWS + row] =
            0.8f - 0.0005f * (float)row;
        targets[row] = 0.3f + 0.001f * (float)row;
        targets[TEST_ROWS + row] =
            -0.2f + 0.0007f * (float)row;
    }
    success = secant_cpu_materialize_run(
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
        TEST_ROWS) == SECANT_SUCCESS;
    success =
        secant_cpu_sse_run(
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
            TEST_TARGETS) == SECANT_SUCCESS &&
        success;
    success = test_architecture(&gfx_arch) && success;
    success = test_inspect_templates(&state, gfx_arch) && success;
    success = test_specialize_templates(&state) && success;
    success = test_load(&state) && success;
    success = test_run(
        &state,
        input,
        targets,
        actual_materialize,
        actual_sse) && success;
    success = test_compare(
        "materialize",
        actual_materialize,
        expected_materialize,
        TEST_ASTS * TEST_ROWS,
        8.0e-4f) && success;
    success = test_compare(
        "sse",
        actual_sse,
        expected_sse,
        TEST_ASTS * TEST_TARGETS,
        3.0e-3f) && success;
    success = test_runner_run(
        &state,
        expected_materialize,
        expected_sse,
        actual_materialize,
        actual_sse) && success;
    success = test_register_expansion(gfx_arch) && success;
    test_state_destroy(&state);
    return success ? 0 : 1;
}
