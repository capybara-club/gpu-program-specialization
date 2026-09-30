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
#include "o_odezza_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static OdezzaResult o_expect_result(OdezzaResult actual, OdezzaResult expected, const char *label) {
    if (label == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (actual != expected) {
        fprintf(stderr, "%s: got result %d, expected %d\n", label, (int)actual, (int)expected);
        return ODEZZA_ERROR_FORMAT;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_file(const char *path, const char *data, size_t data_size) {
    FILE *file;
    size_t written;

    if (path == NULL || data == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return ODEZZA_ERROR_FORMAT;
    }
    written = fwrite(data, 1u, data_size, file);
    if (fclose(file) != 0 || written != data_size) {
        return ODEZZA_ERROR_FORMAT;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_generate_default(char *buffer, size_t buffer_size, size_t *bytes_written_ret) {
    return odezza_generate_scoring_cuda(
        4u,
        4u,
        8u,
        384u,
        384u,
        buffer,
        buffer_size,
        bytes_written_ret
    );
}

static OdezzaResult o_test_generate_scoring_cuda(const char *output_path) {
    OdezzaResult result;
    char *source;
    char *second;
    size_t content_bytes;
    size_t reported;

    result = o_generate_default(NULL, 123u, &content_bytes);
    if (result != ODEZZA_SUCCESS || content_bytes == 0u || content_bytes == SIZE_MAX) {
        return ODEZZA_ERROR_FORMAT;
    }
    source = (char *)malloc(content_bytes + 1u);
    second = (char *)malloc(content_bytes + 1u);
    if (source == NULL || second == NULL) {
        free(second);
        free(source);
        return ODEZZA_ERROR_OVERFLOW;
    }

    reported = 0u;
    result = o_generate_default(source, content_bytes, &reported);
    if (o_expect_result(result, ODEZZA_ERROR_INSUFFICIENT_BUFFER, "buffer without NUL capacity") != ODEZZA_SUCCESS || reported != content_bytes) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }
    result = o_generate_default(source, content_bytes + 1u, &reported);
    if (result != ODEZZA_SUCCESS || reported != content_bytes || source[content_bytes] != '\0' || strlen(source) != content_bytes) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }
    result = o_generate_default(second, content_bytes + 1u, &reported);
    if (result != ODEZZA_SUCCESS || memcmp(source, second, content_bytes + 1u) != 0) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }
    if (strstr(source, "ODEZZA_STATE_CAPACITY 4") == NULL || strstr(source, "runtime_ragged_dense_state") == NULL ||
        strstr(source, "trajectory_offsets") == NULL || strstr(source, "toggle_bit_positions") == NULL ||
        strstr(source, "\"toggle_permutations\": \"runtime power of two\"") == NULL ||
        strstr(source, "const unsigned int permutation = (unsigned int)configuration") == NULL ||
        strstr(source, "configuration >> active_toggle_count") == NULL || strstr(source, "\"r\"(permutation)") == NULL ||
        strstr(source, "ODEZZA_TOGGLE_PERMUTATION_COUNT") != NULL ||
        strstr(source, "toggle_capacity") != NULL || strstr(source, "toggle_shifts") != NULL || strstr(source, "toggle_shift0") != NULL ||
        strstr(source, "state_names") != NULL ||
        strstr(source, "observation_interval") != NULL) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }

    reported = 77u;
    result = o_generate_default(NULL, 0u, NULL);
    if (o_expect_result(result, ODEZZA_ERROR_INVALID_ARGUMENT, "NULL byte-count output") != ODEZZA_SUCCESS) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }
    result = odezza_generate_scoring_cuda(
        0u,
        4u,
        8u,
        384u,
        384u,
        NULL,
        0u,
        &reported
    );
    if (o_expect_result(result, ODEZZA_ERROR_INVALID_ARGUMENT, "zero state capacity") != ODEZZA_SUCCESS || reported != 0u) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }
    result = odezza_generate_scoring_cuda(
        4u,
        4u,
        8u,
        384u,
        1u,
        NULL,
        0u,
        &reported
    );
    if (o_expect_result(result, ODEZZA_ERROR_INVALID_ARGUMENT, "undersized branch arena") != ODEZZA_SUCCESS || reported != 0u) {
        free(second);
        free(source);
        return ODEZZA_ERROR_FORMAT;
    }

    if (output_path != NULL) {
        result = o_write_file(output_path, source, content_bytes);
        if (result != ODEZZA_SUCCESS) {
            free(second);
            free(source);
            return result;
        }
    }
    printf(
        "C99 CUDA generator: %zu bytes written, %zu bytes for a C-string allocation\n",
        content_bytes,
        content_bytes + 1u
    );
    free(second);
    free(source);
    return ODEZZA_SUCCESS;
}

int main(int argc, char **argv) {
    OdezzaResult result;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [generated-cuda-output]\n", argv[0]);
        return 2;
    }
    result = o_test_generate_scoring_cuda(argc == 2 ? argv[1] : NULL);
    if (result != ODEZZA_SUCCESS) {
        fprintf(stderr, "C99 CUDA generator tests failed with result %d\n", (int)result);
        return 1;
    }
    return 0;
}
