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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const SecantAstInstruction test_add[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sin[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_mul_add[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_max[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_max_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_mixed_leaf_add[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_mixed_leaf_mul[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_dynamic_constant_or_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_lm_mixed[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_or_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static size_t
test_count(const char* text, const char* pattern) {
    size_t count = 0u;
    const size_t pattern_size = strlen(pattern);

    while ((text = strstr(text, pattern)) != NULL) {
        ++count;
        text += pattern_size;
    }
    return count;
}

int
main(void) {
    const SecantAstInstruction* const asts[] = {
        test_add, test_sin, test_sin, test_add
    };
    const SecantAstInstruction* const dynamic_asts[] = {
        test_dynamic_mul_add,
        test_dynamic_max,
        test_dynamic_max,
        test_dynamic_mul_add
    };
    const SecantAstInstruction* const mixed_asts[] = {
        test_mixed_leaf_add,
        test_mixed_leaf_mul,
        test_mixed_leaf_mul,
        test_mixed_leaf_add
    };
    size_t required_size = 0u;
    size_t generated_size = 0u;
    char* cuda;
    SecantCUDAResult result;

    result = secant_cuda_materialize_source_generate(
        2u, 2u, 3u, NULL, 0u, NULL, asts, NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_materialize_source_generate(
        2u, 2u, 3u, NULL, 0u, NULL, asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || generated_size != required_size ||
        strstr(cuda, "void secant_static_column_materialize_000(") == NULL ||
        strstr(cuda, "void secant_static_column_materialize_001(") == NULL ||
        strstr(cuda, "const float input2 =") == NULL ||
        strstr(cuda, "const float input3 =") != NULL ||
        test_count(cuda, "\n    {\n") != 4u) {
        fprintf(stderr, "unexpected generated CUDA\n");
        free(cuda);
        return 1;
    }
    free(cuda);
    required_size = 0u;
    generated_size = 0u;
    result = secant_cuda_sse_source_generate(
        2u, 2u, 3u, 1u, 128u, 128u,
        NULL, 0u, NULL, asts, NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_sse_source_generate(
        2u, 2u, 3u, 1u, 128u, 128u,
        NULL, 0u, NULL, asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS ||
        generated_size != required_size ||
        strstr(
            cuda,
            "remaining_rows = tile_begin < num_rows"
        ) == NULL) {
        fprintf(stderr, "unexpected generated CUDA SSE\n");
        free(cuda);
        return 1;
    }
    free(cuda);

    required_size = 0u;
    generated_size = 0u;
    {
        const SecantAstInstruction* const lm_asts[] = { test_lm_mixed };

        result = secant_cuda_lm_optimizer_source_generate(
            1u, 3u, 0u, 128u, 128u,
            NULL, 0u, lm_asts,
            NULL, 0u, &required_size, NULL);
        if (result != SECANT_CUDA_ERROR_INVALID_VALUE) {
            fprintf(stderr, "CUDA LM accepted a dynamic-only template\n");
            return 1;
        }
        required_size = 0u;
        result = secant_cuda_lm_optimizer_source_generate(
            1u, 3u, 3u, 128u, 128u,
            NULL, 0u, lm_asts,
            NULL, 0u, &required_size, NULL);
        if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
            return 1;
        }
        cuda = (char*)malloc(required_size);
        if (cuda == NULL) {
            return 1;
        }
        result = secant_cuda_lm_optimizer_source_generate(
            1u, 3u, 3u, 128u, 128u,
            NULL, 0u, lm_asts,
            cuda, required_size, &generated_size, NULL);
        if (result != SECANT_CUDA_SUCCESS || generated_size != required_size ||
            strstr(cuda, "void secant_lm_statistics_000(") == NULL ||
            strstr(cuda, "void secant_lm_solve_f32(") == NULL ||
            strstr(cuda, "const float constant0") == NULL ||
            strstr(cuda, "const float mixed1") == NULL ||
            strstr(cuda, "const float mixed_gradient1") == NULL ||
            strstr(cuda, "const float input2 = input_tile[") == NULL ||
            strstr(cuda, "@p ld.shared.f32 %0, [%1]") == NULL ||
            strstr(cuda, "address7 += shared_stride_bytes;") == NULL ||
            strstr(cuda, "blockIdx.y * settings_per_cta") == NULL ||
            strstr(cuda, "float sum_7_7") == NULL ||
            strstr(cuda, "const float l_7_7") == NULL ||
            strstr(cuda, "accepted[statistic * statistics_leading_dimension") == NULL ||
            strstr(cuda, "float matrix[") != NULL) {
            fprintf(stderr, "unexpected generated CUDA LM optimizer\n");
            free(cuda);
            return 1;
        }
        free(cuda);
    }

    required_size = 0u;
    generated_size = 0u;
    result = secant_cuda_dynamic_constant_sse_source_generate(
        2u, 2u, 3u, 2u, 2u, 128u, 128u,
        NULL, 0u, NULL, dynamic_asts,
        NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_dynamic_constant_sse_source_generate(
        2u, 2u, 3u, 2u, 2u, 128u, 128u,
        NULL, 0u, NULL, dynamic_asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS ||
        generated_size != required_size ||
        strstr(cuda, "void secant_dynamic_constant_sse_000(") == NULL ||
        strstr(cuda, "void secant_dynamic_constant_sse_001(") == NULL ||
        strstr(
            cuda,
            "const float input3 = constant_settings["
        ) == NULL ||
        strstr(
            cuda,
            "const float input4 = constant_settings["
        ) == NULL ||
        strstr(cuda, "__shfl") != NULL ||
        strstr(
            cuda,
            "output_leading_dimension + setting"
        ) == NULL ||
        test_count(cuda, "const float target1 =") != 2u) {
        fprintf(stderr, "unexpected generated CUDA dynamic SSE\n");
        free(cuda);
        return 1;
    }
    free(cuda);

    result = secant_cuda_dynamic_constant_sse_source_generate(
        1u,
        1u,
        SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS + 1u,
        1u,
        1u,
        128u,
        128u,
        NULL,
        0u,
        NULL,
        dynamic_asts,
        NULL,
        0u,
        &required_size,
        NULL);
    if (result != SECANT_CUDA_ERROR_INVALID_VALUE) {
        return 1;
    }
    result = secant_cuda_dynamic_constant_sse_source_generate(
        1u,
        1u,
        1u,
        SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS + 1u,
        1u,
        128u,
        128u,
        NULL,
        0u,
        NULL,
        dynamic_asts,
        NULL,
        0u,
        &required_size,
        NULL);
    if (result != SECANT_CUDA_ERROR_INVALID_VALUE) {
        return 1;
    }

    required_size = 0u;
    generated_size = 0u;
    result = secant_cuda_warp_dynamic_leaf_sse_source_generate(
        2u, 2u, 3u, 3u, 2u, 1u, 128u, 128u,
        NULL, 0u, NULL, mixed_asts,
        NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_warp_dynamic_leaf_sse_source_generate(
        2u, 2u, 3u, 3u, 2u, 1u, 128u, 128u,
        NULL, 0u, NULL, mixed_asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || generated_size != required_size ||
        strstr(cuda, "const unsigned int warp = threadIdx.x >> 5u;") == NULL ||
        strstr(cuda, "blockIdx.y * settings_per_cta") == NULL ||
        strstr(cuda, "for (unsigned int setting_offset = warp;") == NULL ||
        strstr(cuda, "for (size_t eval_row = lane;") == NULL ||
        strstr(cuda, "__shfl_down_sync(0xffffffffu") == NULL ||
        strstr(cuda, "if (lane == 0u && 0u < num_asts") == NULL ||
        strstr(cuda, "input_tile[(size_t)word0 * 128u + eval_row]") == NULL ||
        strstr(cuda, "input_tile[eval_row * shared_stride") != NULL) {
        fprintf(stderr, "unexpected generated CUDA warp dynamic-leaf SSE\n");
        free(cuda);
        return 1;
    }
    free(cuda);
    result = secant_cuda_warp_dynamic_leaf_sse_source_generate(
        1u, 1u, 3u, 3u, 2u, 1u, 128u, 33u,
        NULL, 0u, NULL, mixed_asts,
        NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_ERROR_INVALID_VALUE) {
        return 1;
    }

    required_size = 0u;
    generated_size = 0u;
    result = secant_cuda_dynamic_leaf_lm_source_generate(
        2u, 2u, 3u, 3u, 2u, 1u, 128u, 128u,
        NULL, 0u, mixed_asts,
        NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_dynamic_leaf_lm_source_generate(
        2u, 2u, 3u, 3u, 2u, 1u, 128u, 128u,
        NULL, 0u, mixed_asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || generated_size != required_size ||
        strstr(cuda, "void secant_dynamic_leaf_lm_000(") == NULL ||
        strstr(cuda, "secant_dynamic_leaf_lm_eval_001_001(") == NULL ||
        strstr(cuda, "gradient0 = is_column0 ? 0.0f : gradient0;") == NULL ||
        strstr(cuda, "lm_000_000_005 = fmaf(gradient1, gradient1") == NULL ||
        strstr(cuda, "* 6u + 5u)") == NULL ||
        strstr(cuda, "blockIdx.y * settings_per_cta") == NULL ||
        strstr(cuda, "for (unsigned int setting_offset = threadIdx.x;") == NULL) {
        fprintf(stderr, "unexpected generated CUDA dynamic-leaf LM\n");
        free(cuda);
        return 1;
    }
    free(cuda);

    required_size = 0u;
    generated_size = 0u;
    result = secant_cuda_warp_dynamic_leaf_lm_source_generate(
        1u, 1u, 3u, 3u, 2u, 1u, 128u, 128u,
        NULL, 0u, mixed_asts,
        NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_warp_dynamic_leaf_lm_source_generate(
        1u, 1u, 3u, 3u, 2u, 1u, 128u, 128u,
        NULL, 0u, mixed_asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || generated_size != required_size ||
        strstr(cuda, "for (unsigned int setting_offset = warp;") == NULL ||
        strstr(cuda, "for (size_t eval_row = lane;") == NULL ||
        strstr(cuda, "lm_000_000_005 += __shfl_down_sync") == NULL ||
        strstr(cuda, "if (lane == 0u && 0u < num_asts") == NULL) {
        fprintf(stderr, "unexpected generated CUDA warp dynamic-leaf LM\n");
        free(cuda);
        return 1;
    }
    free(cuda);

    required_size = 0u;
    generated_size = 0u;
    result = secant_cuda_philox_dynamic_leaf_select_source_generate(
        2u, 2u, 3u, 3u, 2u, 128u, 128u,
        NULL, 0u, NULL, mixed_asts,
        NULL, 0u, &required_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_cuda_philox_dynamic_leaf_select_source_generate(
        2u, 2u, 3u, 3u, 2u, 128u, 128u,
        NULL, 0u, NULL, mixed_asts,
        cuda, required_size, &generated_size, NULL);
    if (result != SECANT_CUDA_SUCCESS || generated_size != required_size ||
        strstr(cuda, "void secant_philox_dynamic_leaf_select_000(") == NULL ||
        strstr(cuda, "void secant_philox_dynamic_leaf_select_001(") == NULL ||
        strstr(cuda, "void secant_philox_dynamic_leaf_reconstruct(") == NULL ||
        strstr(cuda, "secant_leaf_philox4x32_10") == NULL ||
        strstr(cuda, "candidate_index = pass_base") == NULL ||
        strstr(cuda, "output_best_sse != nullptr") == NULL) {
        fprintf(stderr, "unexpected generated CUDA Philox dynamic-leaf selection\n");
        free(cuda);
        return 1;
    }
    free(cuda);
    return 0;
}
