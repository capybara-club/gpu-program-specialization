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
#include "secant_cuda.h"

#include <nvrtc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const SecantAstInstruction test_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction square_routine[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction routine_ast[] = {
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction dynamic_column_ast[] = {
    secant_ast_encode_dynamic_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction all_ops_ast[] = {
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_neg_f32,
    secant_ast_encode_abs_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_cos_f32,
    secant_ast_encode_ex2_f32,
    secant_ast_encode_lg2_f32,
    secant_ast_encode_rsqrt_f32,
    secant_ast_encode_tanh_f32,
    secant_ast_encode_exp_f32,
    secant_ast_encode_log_f32,
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_sub_f32,
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_div_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_min_f32,
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_max_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_fma_f32,
    secant_ast_encode_return_f32
};

static const char test_wrapper[] =
    "\nextern \"C\" __global__ void secant_ast_gradient_test(float* output) {\n"
    "    float value;\n"
    "    float gradient0;\n"
    "    float gradient1;\n"
    "    secant_test_gradient(2.0f, 3.0f, 4.0f, value, gradient0, gradient1);\n"
    "    output[0] = value;\n"
    "    output[1] = gradient0;\n"
    "    output[2] = gradient1;\n"
    "}\n";

static const char all_ops_wrapper[] =
    "\nextern \"C\" __global__ void secant_ast_all_ops_gradient_test(float* output) {\n"
    "    float value;\n"
    "    float gradient0;\n"
    "    float gradient1;\n"
    "    secant_test_all_ops_gradient(2.0f, 3.0f, 4.0f, value, gradient0, gradient1);\n"
    "    output[0] = value;\n"
    "    output[1] = gradient0;\n"
    "    output[2] = gradient1;\n"
    "}\n";

static int
source_generate(
    int gradient,
    const char* function_name,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char** source_ret,
    size_t* source_size_ret,
    uint32_t* num_variables_ret
) {
    SecantCUDAResult result;
    size_t required_size = 0u;
    size_t generated_size = 0u;
    char* source;

    result = gradient
        ? secant_cuda_ast_forward_gradient_source_generate(
            function_name,
            num_input_columns,
            num_dynamic_constants,
            routines,
            num_routines,
            ast,
            NULL,
            0u,
            &required_size,
            NULL)
        : secant_cuda_ast_primal_source_generate(
            function_name,
            num_input_columns,
            num_dynamic_constants,
            routines,
            num_routines,
            ast,
            NULL,
            0u,
            &required_size,
            NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 0;
    }
    source = (char*)malloc(required_size);
    if (source == NULL) {
        return 0;
    }
    result = gradient
        ? secant_cuda_ast_forward_gradient_source_generate(
            function_name,
            num_input_columns,
            num_dynamic_constants,
            routines,
            num_routines,
            ast,
            source,
            required_size,
            &generated_size,
            num_variables_ret)
        : secant_cuda_ast_primal_source_generate(
            function_name,
            num_input_columns,
            num_dynamic_constants,
            routines,
            num_routines,
            ast,
            source,
            required_size,
            &generated_size,
            num_variables_ret);
    if (result != SECANT_CUDA_SUCCESS || generated_size != required_size) {
        free(source);
        return 0;
    }
    *source_ret = source;
    *source_size_ret = generated_size;
    return 1;
}

static int
source_compile(
    const char* generated,
    size_t generated_size,
    const char* wrapper
) {
    const char* options[] = {
        "--std=c++17",
        "--gpu-architecture=compute_80"
    };
    const size_t wrapper_size = strlen(wrapper) + 1u;
    const size_t source_size = generated_size - 1u + wrapper_size;
    char* source = (char*)malloc(source_size);
    nvrtcProgram program = NULL;
    nvrtcResult result;
    int success = 0;

    if (source == NULL) {
        return 0;
    }
    memcpy(source, generated, generated_size - 1u);
    memcpy(source + generated_size - 1u, wrapper, wrapper_size);
    result = nvrtcCreateProgram(&program, source, "secant_ast_gradient_test.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, 2, options);
    }
    if (result != NVRTC_SUCCESS) {
        size_t log_size = 0u;

        if (program != NULL && nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size != 0u) {
            char* log = (char*)malloc(log_size);

            if (log != NULL) {
                if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    fprintf(stderr, "%s", log);
                }
                free(log);
            }
        }
    } else {
        success = 1;
    }
    if (program != NULL) {
        nvrtcDestroyProgram(&program);
    }
    free(source);
    return success;
}

int
main(int argc, char** argv) {
    const SecantAstInstruction* const routines[] = {square_routine};
    char* primal = NULL;
    char* gradient = NULL;
    char* routine_gradient = NULL;
    char* all_ops_gradient = NULL;
    size_t primal_size = 0u;
    size_t gradient_size = 0u;
    size_t routine_gradient_size = 0u;
    size_t all_ops_gradient_size = 0u;
    size_t ignored_size = 0u;
    uint32_t primal_variables = 0u;
    uint32_t gradient_variables = 0u;
    uint32_t routine_variables = 0u;
    SecantCUDAResult result;
    int status = 1;

    if (!source_generate(
            0,
            "secant_test_primal",
            1u,
            2u,
            NULL,
            0u,
            test_ast,
            &primal,
            &primal_size,
            &primal_variables) ||
        !source_generate(
            1,
            "secant_test_gradient",
            1u,
            2u,
            NULL,
            0u,
            test_ast,
            &gradient,
            &gradient_size,
            &gradient_variables) ||
        !source_generate(
            1,
            "secant_test_routine_gradient",
            0u,
            1u,
            routines,
            1u,
            routine_ast,
            &routine_gradient,
            &routine_gradient_size,
            &routine_variables) ||
        !source_generate(
            1,
            "secant_test_all_ops_gradient",
            1u,
            2u,
            NULL,
            0u,
            all_ops_ast,
            &all_ops_gradient,
            &all_ops_gradient_size,
            NULL)) {
        goto cleanup;
    }
    if (primal_variables != 3u || gradient_variables != 3u || routine_variables != 1u ||
        strstr(primal, "const float x0 = (input0 * constant0);") == NULL ||
        strstr(primal, "return x2;") == NULL ||
        strstr(gradient, "float& gradient0_ret") == NULL ||
        strstr(gradient, "float& gradient1_ret") == NULL ||
        strstr(gradient, "const float dx0_0 = fmaf(0.0f, constant0, input0 * 1.0f);") == NULL ||
        strstr(gradient, "secant_cuda_ast_cos_f32(x0) * dx0_0") == NULL ||
        strstr(gradient, "gradient1_ret = dx2_1;") == NULL ||
        strstr(gradient, "for (") != NULL ||
        strstr(routine_gradient, "const float x0 = (constant0 * constant0);") == NULL ||
        strstr(routine_gradient, "secant_cuda_routine") != NULL) {
        fprintf(stderr, "unexpected generated AST CUDA\n");
        goto cleanup;
    }
    if (!source_compile(gradient, gradient_size, test_wrapper) ||
        !source_compile(all_ops_gradient, all_ops_gradient_size, all_ops_wrapper)) {
        goto cleanup;
    }
    result = secant_cuda_ast_forward_gradient_source_generate(
        "bad_dynamic_column",
        0u,
        1u,
        NULL,
        0u,
        dynamic_column_ast,
        NULL,
        0u,
        &ignored_size,
        NULL);
    if (result != SECANT_CUDA_ERROR_UNSUPPORTED_OP) {
        goto cleanup;
    }

    if (argc == 2 && strcmp(argv[1], "--print") == 0) {
        printf("// Primal\n\n%s\n\n// Forward gradient\n\n%s", primal, gradient);
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--print]\n", argv[0]);
        goto cleanup;
    }
    status = 0;

cleanup:
    free(all_ops_gradient);
    free(routine_gradient);
    free(gradient);
    free(primal);
    return status;
}
