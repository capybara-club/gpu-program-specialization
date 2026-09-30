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
#define CUSR_SASS_INSPECT_IMPLEMENTATION
#include <cusr_sass_inspect.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUSR_INSPECT_ERROR_RET(value) return (value)
#define CUSR_INSPECT_CHECK_RET(expr) do { if (!(expr)) { CUSR_INSPECT_ERROR_RET(0); } } while (0)

typedef struct CusrInspectFile {
    unsigned char* data;
    size_t size;
} CusrInspectFile;

static int
cusr_inspect_parse_size(const char* text, size_t* value_ret)
{
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > (unsigned long long)SIZE_MAX) {
        CUSR_INSPECT_ERROR_RET(0);
    }

    *value_ret = (size_t)value;
    return 1;
}

static int
cusr_inspect_parse_u32(const char* text, uint32_t* value_ret)
{
    char* end = NULL;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        CUSR_INSPECT_ERROR_RET(0);
    }

    *value_ret = (uint32_t)value;
    return 1;
}

static int
cusr_inspect_read_file(const char* path, CusrInspectFile* file)
{
    FILE* stream = fopen(path, "rb");
    long end;

    file->data = NULL;
    file->size = 0u;

    if (stream == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        CUSR_INSPECT_ERROR_RET(0);
    }

    if (fseek(stream, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek %s\n", path);
        fclose(stream);
        CUSR_INSPECT_ERROR_RET(0);
    }

    end = ftell(stream);
    if (end < 0) {
        fprintf(stderr, "failed to measure %s\n", path);
        fclose(stream);
        CUSR_INSPECT_ERROR_RET(0);
    }

    if (fseek(stream, 0, SEEK_SET) != 0) {
        fprintf(stderr, "failed to rewind %s\n", path);
        fclose(stream);
        CUSR_INSPECT_ERROR_RET(0);
    }

    file->size = (size_t)end;
    file->data = (unsigned char*)malloc(file->size == 0u ? 1u : file->size);
    if (file->data == NULL) {
        fprintf(stderr, "failed to allocate %zu bytes\n", file->size);
        fclose(stream);
        CUSR_INSPECT_ERROR_RET(0);
    }

    if (fread(file->data, 1u, file->size, stream) != file->size) {
        fprintf(stderr, "failed to read %s\n", path);
        fclose(stream);
        CUSR_INSPECT_ERROR_RET(0);
    }

    if (fclose(stream) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", path, strerror(errno));
        CUSR_INSPECT_ERROR_RET(0);
    }

    return 1;
}

static void
cusr_inspect_usage(const char* argv0)
{
    fprintf(
        stderr,
        "usage: %s [--json] <input.cubin> <function-pattern> <num-kernels> <ast-capacity> <first-marker-bits> [expected-occurrences-per-kernel]\n"
        "\n"
        "example:\n"
        "  %s build/site.cubin 'cusr_tile_static_mse_f32_%%03d' 2 32 0x7fc0ffee 1\n",
        argv0,
        argv0
    );
}

static int
cusr_inspect_make_function_names(const char* pattern, size_t count, char*** function_names_ret)
{
    size_t pointers_size;
    size_t storage_size;
    char** function_names;
    char* storage;
    size_t kernel_index;

    *function_names_ret = NULL;

    if (count > SIZE_MAX / sizeof(char*) ||
        count > SIZE_MAX / CUSR_SASS_INSPECT_NAME_BYTES) {
        CUSR_INSPECT_ERROR_RET(0);
    }

    pointers_size = count * sizeof(char*);
    storage_size = count * CUSR_SASS_INSPECT_NAME_BYTES;
    if (pointers_size > SIZE_MAX - storage_size) {
        CUSR_INSPECT_ERROR_RET(0);
    }

    function_names = (char**)malloc(pointers_size + storage_size);
    if (function_names == NULL) {
        CUSR_INSPECT_ERROR_RET(0);
    }

    storage = (char*)(function_names + count);
    for (kernel_index = 0u; kernel_index < count; ++kernel_index) {
        int written;

        function_names[kernel_index] = storage + kernel_index * CUSR_SASS_INSPECT_NAME_BYTES;
        written = snprintf(function_names[kernel_index], CUSR_SASS_INSPECT_NAME_BYTES, pattern, (int)kernel_index);
        if (written <= 0 || (size_t)written >= CUSR_SASS_INSPECT_NAME_BYTES) {
            free(function_names);
            CUSR_INSPECT_ERROR_RET(0);
        }
    }

    *function_names_ret = function_names;
    return 1;
}

static int
cusr_inspect_main_impl(int argc, char** argv, CusrInspectFile* file, void** workspace_ret, char*** function_names_ret)
{
    size_t arg_index = 1u;
    int print_json = 0;
    const char* cubin_path;
    const char* function_pattern;
    size_t num_kernels = 0u;
    size_t ast_capacity = 0u;
    uint32_t first_marker_bits = 0u;
    size_t expected_occurrences = 0u;
    size_t workspace_size = 0u;
    CusrSassInspectHandle handle;
    CusrSassInspectResult result;

    *workspace_ret = NULL;
    *function_names_ret = NULL;

    if (argc > 1 && strcmp(argv[1], "--json") == 0) {
        print_json = 1;
        arg_index += 1u;
    }

    if ((size_t)argc != arg_index + 5u && (size_t)argc != arg_index + 6u) {
        cusr_inspect_usage(argv[0]);
        CUSR_INSPECT_ERROR_RET(2);
    }

    cubin_path = argv[arg_index + 0u];
    function_pattern = argv[arg_index + 1u];

    if (!cusr_inspect_parse_size(argv[arg_index + 2u], &num_kernels) ||
        !cusr_inspect_parse_size(argv[arg_index + 3u], &ast_capacity) ||
        !cusr_inspect_parse_u32(argv[arg_index + 4u], &first_marker_bits)) {
        cusr_inspect_usage(argv[0]);
        CUSR_INSPECT_ERROR_RET(2);
    }

    if ((size_t)argc == arg_index + 6u && !cusr_inspect_parse_size(argv[arg_index + 5u], &expected_occurrences)) {
        cusr_inspect_usage(argv[0]);
        CUSR_INSPECT_ERROR_RET(2);
    }

    CUSR_INSPECT_CHECK_RET(cusr_inspect_read_file(cubin_path, file));
    CUSR_INSPECT_CHECK_RET(cusr_inspect_make_function_names(function_pattern, num_kernels, function_names_ret));

    result = cusr_sass_inspect_workspace_size(
        file->data,
        file->size,
        (const char* const*)*function_names_ret,
        num_kernels,
        ast_capacity,
        first_marker_bits,
        expected_occurrences,
        &workspace_size
    );
    if (result != CUSR_SASS_INSPECT_SUCCESS) {
        fprintf(stderr, "cusr_sass_inspect_workspace_size failed: %s\n", cusr_sass_inspect_result_to_string(result));
        CUSR_INSPECT_ERROR_RET(0);
    }

    *workspace_ret = malloc(workspace_size == 0u ? 1u : workspace_size);
    if (*workspace_ret == NULL) {
        fprintf(stderr, "failed to allocate %zu workspace bytes\n", workspace_size);
        CUSR_INSPECT_ERROR_RET(0);
    }

    result = cusr_sass_inspect(
        file->data,
        file->size,
        (const char* const*)*function_names_ret,
        num_kernels,
        ast_capacity,
        first_marker_bits,
        expected_occurrences,
        *workspace_ret,
        workspace_size,
        &handle
    );
    if (result != CUSR_SASS_INSPECT_SUCCESS) {
        fprintf(stderr, "cusr_sass_inspect failed: %s\n", cusr_sass_inspect_result_to_string(result));
        CUSR_INSPECT_ERROR_RET(0);
    }

    if (print_json) {
        cusr_sass_inspect_print_json(stdout, &handle);
    } else {
        cusr_sass_inspect_print(stdout, cubin_path, &handle);
    }
    return 1;
}

int
main(int argc, char** argv)
{
    CusrInspectFile file;
    void* workspace;
    char** function_names;
    int result;

    memset(&file, 0, sizeof(file));
    workspace = NULL;
    function_names = NULL;

    result = cusr_inspect_main_impl(argc, argv, &file, &workspace, &function_names);

    free(function_names);
    free(workspace);
    free(file.data);

    if (result == 1) {
        return 0;
    }

    if (result == 2) {
        return 2;
    }

    return 1;
}
