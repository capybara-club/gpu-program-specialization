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
#include "secant_ptx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    size_t required_size = 0u;
    size_t generated_size = 0u;
    char* cuda;
    SecantPTXResult result;

    result = secant_ptx_materialize_source_generate(
        2u, 3u, 4u, NULL, 0u, &required_size);
    if (result != SECANT_PTX_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_ptx_materialize_source_generate(
        2u, 3u, 4u, cuda, required_size, &generated_size);
    if (result != SECANT_PTX_SUCCESS || generated_size != required_size ||
        strstr(cuda, "PTX_INJECT_START secant_expr_000_000") == NULL ||
        strstr(cuda, "PTX_INJECT_START secant_expr_001_002") == NULL ||
        strstr(cuda, "// _x4 i f32 F32 input3") == NULL ||
        test_count(cuda, "// PTX_INJECT_START") != 6u) {
        fprintf(stderr, "unexpected generated PTX template CUDA\n");
        free(cuda);
        return 1;
    }
    free(cuda);
    required_size = 0u;
    generated_size = 0u;
    result = secant_ptx_sse_source_generate(
        2u, 3u, 4u, 1u, 128u, 128u,
        SECANT_SSE_REDUCTION_MODE_ATOMIC,
        NULL, 0u, &required_size);
    if (result != SECANT_PTX_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_ptx_sse_source_generate(
        2u, 3u, 4u, 1u, 128u, 128u,
        SECANT_SSE_REDUCTION_MODE_ATOMIC,
        cuda, required_size, &generated_size);
    if (result != SECANT_PTX_SUCCESS ||
        generated_size != required_size ||
        strstr(
            cuda,
            "remaining_rows = tile_begin < num_rows"
        ) == NULL) {
        fprintf(stderr, "unexpected generated PTX SSE template CUDA\n");
        free(cuda);
        return 1;
    }
    free(cuda);

    required_size = 0u;
    generated_size = 0u;
    result = secant_ptx_dynamic_constant_sse_source_generate(
        2u, 3u, 4u, 3u, 2u, 128u, 128u,
        SECANT_SSE_REDUCTION_MODE_ATOMIC,
        NULL, 0u, &required_size);
    if (result != SECANT_PTX_SUCCESS || required_size == 0u) {
        return 1;
    }
    cuda = (char*)malloc(required_size);
    if (cuda == NULL) {
        return 1;
    }
    result = secant_ptx_dynamic_constant_sse_source_generate(
        2u, 3u, 4u, 3u, 2u, 128u, 128u,
        SECANT_SSE_REDUCTION_MODE_ATOMIC,
        cuda, required_size, &generated_size);
    if (result != SECANT_PTX_SUCCESS ||
        generated_size != required_size ||
        strstr(cuda, "void secant_dynamic_constant_sse_000(") == NULL ||
        strstr(cuda, "void secant_dynamic_constant_sse_001(") == NULL ||
        strstr(
            cuda,
            "const float input4 = constant_settings["
        ) == NULL ||
        strstr(cuda, "__shfl") != NULL ||
        test_count(cuda, "// PTX_INJECT_START") != 6u) {
        fprintf(stderr, "unexpected generated PTX dynamic SSE CUDA\n");
        free(cuda);
        return 1;
    }
    free(cuda);
    return 0;
}
