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
#include "secant_hip.h"

#include <stdlib.h>
#include <string.h>

static const SecantAstInstruction program[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

int
main(void) {
    const SecantAstInstruction* asts[] = { program };
    char* source = NULL;
    char* sse_source = NULL;
    char* dynamic_source = NULL;
    size_t source_size = 0u;
    size_t written_size = 0u;
    size_t sse_size = 0u;
    size_t dynamic_size = 0u;
    int result = 1;

    if (secant_hip_sse_source_generate(
            1u, 1u, 2u, 1u, 128u, 128u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC, NULL, 0u, NULL, asts,
            NULL, 0u, &sse_size, NULL) != SECANT_HIP_SUCCESS ||
        sse_size == 0u ||
        secant_hip_sse_source_generate(
            1u, 1u, 2u, 1u, 1024u, 1024u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC, NULL, 0u, NULL, asts,
            NULL, 0u, &sse_size, NULL) != SECANT_HIP_ERROR_INVALID_VALUE) {
        return 1;
    }
    sse_source = (char*)malloc(sse_size);
    if (sse_source == NULL ||
        secant_hip_sse_source_generate(
            1u, 1u, 2u, 1u, 128u, 128u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC, NULL, 0u, NULL, asts,
            sse_source, sse_size, &written_size, NULL) !=
                SECANT_HIP_SUCCESS ||
        written_size != sse_size ||
        strstr(
            sse_source,
            "remaining_rows = tile_begin < num_rows"
        ) == NULL) {
        free(sse_source);
        return 1;
    }
    free(sse_source);
    written_size = 0u;
    if (secant_hip_dynamic_constant_sse_source_generate(
            1u, 1u, 2u, 2u, 2u, 128u, 128u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC,
            NULL, 0u, NULL, asts,
            NULL, 0u, &dynamic_size, NULL) != SECANT_HIP_SUCCESS ||
        dynamic_size == 0u) {
        return 1;
    }
    dynamic_source = (char*)malloc(dynamic_size);
    if (dynamic_source == NULL ||
        secant_hip_dynamic_constant_sse_source_generate(
            1u, 1u, 2u, 2u, 2u, 128u, 128u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC,
            NULL, 0u, NULL, asts,
            dynamic_source,
            dynamic_size,
            &written_size,
            NULL) != SECANT_HIP_SUCCESS ||
        written_size != dynamic_size ||
        strstr(
            dynamic_source,
            "void secant_dynamic_constant_sse_000("
        ) == NULL ||
        strstr(
            dynamic_source,
            "const float input2 = constant_settings["
        ) == NULL ||
        strstr(dynamic_source, "__shfl") != NULL) {
        free(dynamic_source);
        return 1;
    }
    free(dynamic_source);
    written_size = 0u;
    if (secant_hip_materialize_source_generate(
            1u, 1u, 2u, NULL, 0u, NULL, asts,
            NULL, 0u, &source_size, NULL) != SECANT_HIP_SUCCESS ||
        source_size == 0u) {
        return 1;
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        return 1;
    }
    if (secant_hip_materialize_source_generate(
            1u, 1u, 2u, NULL, 0u, NULL, asts,
            source, source_size, &written_size, NULL) == SECANT_HIP_SUCCESS &&
        written_size == source_size &&
        strstr(source, "secant_static_column_materialize_000") != NULL &&
        strstr(source, "v_rcp_f32") != NULL &&
        strstr(source, "v_sin_f32") != NULL &&
        strstr(source, "v_cos_f32") != NULL &&
        strstr(source, "v_exp_f32") != NULL &&
        strstr(source, "v_rsq_f32") != NULL &&
        strstr(source, "return sinf(value)") == NULL &&
        strstr(source, "return cosf(value)") == NULL) {
        result = 0;
    }

    free(source);
    return result;
}
