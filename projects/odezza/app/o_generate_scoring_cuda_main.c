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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static OdezzaResult o_print_usage(FILE *file, const char *program) {
    int result;

    if (file == NULL || program == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    result = fprintf(file,
                     "Usage: %s [options]\n"
                     "\n"
                     "Generate the CUDA source for the Odezza scoring kernel.\n"
                     "With no options, the command emits the canonical reusable-capacity shape.\n"
                     "\n"
                     "Shape options:\n"
                     "  --state-capacity N              Maximum active states (default 4).\n"
                     "  --constant-capacity N           Maximum constants per system (default 8).\n"
                     "  --system-capacity N             Candidate branches per kernel (default 8).\n"
                     "  --shared-patch-capacity N       Shared SASS instructions (default 64).\n"
                     "  --system-patch-capacity N       SASS instructions per system (default 64).\n"
                     "\n"
                     "Output options:\n"
                     "  -o, --output PATH               Write PATH instead of stdout; '-' is stdout.\n"
                     "  -h, --help                      Show this help.\n",
                     program);
    return result < 0 ? ODEZZA_ERROR_IO : ODEZZA_SUCCESS;
}

static OdezzaResult o_next_argument(int argc, char **argv, int *index, const char **value_ret) {
    if (argv == NULL || index == NULL || value_ret == NULL || *index < 0 || *index >= argc) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (*index + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", argv[*index]);
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    ++*index;
    *value_ret = argv[*index];
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_parse_u32(const char *text, uint32_t *value_ret) {
    char *end;
    unsigned long long value;

    if (text == NULL || value_ret == NULL || text[0] == '\0' || text[0] == '-') {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    errno = 0;
    end = NULL;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    *value_ret = (uint32_t)value;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_output(const char *path, const char *source, size_t source_bytes) {
    FILE *file;
    size_t bytes_written;
    int close_result;

    if (source == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    file = path == NULL || strcmp(path, "-") == 0 ? stdout : fopen(path, "wb");
    if (file == NULL) {
        return ODEZZA_ERROR_IO;
    }
    bytes_written = fwrite(source, 1u, source_bytes, file);
    if (file == stdout) {
        close_result = fflush(file);
    } else {
        close_result = fclose(file);
    }
    if (bytes_written != source_bytes || close_result != 0) {
        return ODEZZA_ERROR_IO;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_generate_scoring_cuda_run(int argc, char **argv) {
    uint32_t state_capacity = 4u;
    uint32_t constant_capacity = 8u;
    uint32_t system_capacity = 8u;
    uint32_t shared_patch_capacity = 64u;
    uint32_t system_patch_capacity = 64u;
    const char *output_path = NULL;
    const char *value;
    char *source;
    size_t source_bytes;
    int index;
    OdezzaResult result;

    if (argc <= 0 || argv == NULL || argv[0] == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    for (index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        if (strcmp(argument, "-h") == 0 || strcmp(argument, "--help") == 0) {
            return o_print_usage(stdout, argv[0]);
        }
        if (strcmp(argument, "-o") == 0 || strcmp(argument, "--output") == 0) {
            result = o_next_argument(argc, argv, &index, &output_path);
            if (result != ODEZZA_SUCCESS) return result;
            continue;
        }

#define O_PARSE_U32_OPTION(option, variable) \
    if (strcmp(argument, option) == 0) { \
        result = o_next_argument(argc, argv, &index, &value); \
        if (result != ODEZZA_SUCCESS) return result; \
        result = o_parse_u32(value, &variable); \
        if (result != ODEZZA_SUCCESS) { \
            fprintf(stderr, "%s requires an unsigned 32-bit integer\n", option); \
            return result; \
        } \
        continue; \
    }

        O_PARSE_U32_OPTION("--state-capacity", state_capacity)
        O_PARSE_U32_OPTION("--constant-capacity", constant_capacity)
        O_PARSE_U32_OPTION("--system-capacity", system_capacity)
        O_PARSE_U32_OPTION("--shared-patch-capacity", shared_patch_capacity)
        O_PARSE_U32_OPTION("--system-patch-capacity", system_patch_capacity)

#undef O_PARSE_U32_OPTION

        fprintf(stderr, "unknown option: %s\n", argument);
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }

    result = odezza_generate_scoring_cuda(
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        NULL,
        0u,
        &source_bytes
    );
    if (result != ODEZZA_SUCCESS) {
        fprintf(stderr, "invalid scoring-kernel shape\n");
        return result;
    }
    if (source_bytes == SIZE_MAX) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    source = (char *)malloc(source_bytes + 1u);
    if (source == NULL) {
        return ODEZZA_ERROR_ALLOCATION;
    }
    result = odezza_generate_scoring_cuda(
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        source,
        source_bytes + 1u,
        &source_bytes
    );
    if (result == ODEZZA_SUCCESS) {
        result = o_write_output(output_path, source, source_bytes);
    }
    free(source);
    return result;
}

int main(int argc, char **argv) {
    OdezzaResult result = o_generate_scoring_cuda_run(argc, argv);
    if (result != ODEZZA_SUCCESS) {
        fprintf(stderr, "internal CUDA generator failed with result %d\n", (int)result);
        return 1;
    }
    return 0;
}
