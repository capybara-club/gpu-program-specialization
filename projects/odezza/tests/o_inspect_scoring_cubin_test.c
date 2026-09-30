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

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define O_TRY(expression) \
    do { \
        OdezzaResult o_result_ = (expression); \
        if (o_result_ != ODEZZA_SUCCESS) return o_result_; \
    } while (0)

static OdezzaResult o_expect_result(OdezzaResult actual, OdezzaResult expected, const char *label) {
    if (label == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (actual != expected) {
        fprintf(stderr, "%s: got result %d, expected %d\n", label, (int)actual, (int)expected);
        return ODEZZA_ERROR_FORMAT;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_read_file(const char *path, unsigned char **data_ret, size_t *size_ret) {
    FILE *file;
    long end;
    unsigned char *data;
    size_t size;
    if (path == NULL || data_ret == NULL || size_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *data_ret = NULL;
    *size_ret = 0u;
    file = fopen(path, "rb");
    if (file == NULL) return ODEZZA_ERROR_IO;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return ODEZZA_ERROR_IO;
    }
    end = ftell(file);
    if (end < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return ODEZZA_ERROR_IO;
    }
    size = (size_t)end;
    data = (unsigned char *)malloc(size == 0u ? 1u : size);
    if (data == NULL) {
        fclose(file);
        return ODEZZA_ERROR_ALLOCATION;
    }
    if (fread(data, 1u, size, file) != size || fclose(file) != 0) {
        free(data);
        return ODEZZA_ERROR_IO;
    }
    *data_ret = data;
    *size_ret = size;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_test_rejections(void) {
    static const unsigned char invalid[] = {0u, 1u, 2u, 3u};
    size_t required = 99u;
    OdezzaResult result;
    result = odezza_inspect_scoring_cubin(invalid, sizeof(invalid), NULL, 0u, NULL, NULL);
    if (o_expect_result(result, ODEZZA_ERROR_INVALID_ARGUMENT, "NULL arena-size output") != ODEZZA_SUCCESS) return ODEZZA_ERROR_FORMAT;
    result = odezza_inspect_scoring_cubin(invalid, sizeof(invalid), NULL, 0u, &required, NULL);
    if (o_expect_result(result, ODEZZA_ERROR_FORMAT, "non-ELF input") != ODEZZA_SUCCESS || required != 0u) return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_pointer_in_arena(const void *pointer, const void *arena, size_t arena_size) {
    uintptr_t value = (uintptr_t)pointer;
    uintptr_t begin = (uintptr_t)arena;
    if (pointer == NULL || value < begin || value >= begin + arena_size) return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_print_size_array(const size_t *values, size_t count) {
    size_t index;
    if (values == NULL && count != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (putchar('[') == EOF) return ODEZZA_ERROR_IO;
    for (index = 0u; index < count; ++index) {
        if (printf("%s%zu", index == 0u ? "" : ",", values[index]) < 0) return ODEZZA_ERROR_IO;
    }
    return putchar(']') == EOF ? ODEZZA_ERROR_IO : ODEZZA_SUCCESS;
}

static OdezzaResult o_print_u8_array(const uint8_t *values, size_t count) {
    size_t index;
    if (values == NULL && count != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (putchar('[') == EOF) return ODEZZA_ERROR_IO;
    for (index = 0u; index < count; ++index) {
        if (printf("%s%u", index == 0u ? "" : ",", (unsigned int)values[index]) < 0) return ODEZZA_ERROR_IO;
    }
    return putchar(']') == EOF ? ODEZZA_ERROR_IO : ODEZZA_SUCCESS;
}

static OdezzaResult o_print_u32_array(const uint32_t *values, size_t count) {
    size_t index;
    if (values == NULL && count != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (putchar('[') == EOF) return ODEZZA_ERROR_IO;
    for (index = 0u; index < count; ++index) {
        if (printf("%s%" PRIu32, index == 0u ? "" : ",", values[index]) < 0) return ODEZZA_ERROR_IO;
    }
    return putchar(']') == EOF ? ODEZZA_ERROR_IO : ODEZZA_SUCCESS;
}

static OdezzaResult o_print_instruction_array(const OdezzaScoringInstruction *values, size_t count) {
    size_t index;
    if (values == NULL && count != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (putchar('[') == EOF) return ODEZZA_ERROR_IO;
    for (index = 0u; index < count; ++index) {
        if (printf("%s[\"%016" PRIx64 "\",\"%016" PRIx64 "\"]", index == 0u ? "" : ",", values[index].word0, values[index].word1) < 0) return ODEZZA_ERROR_IO;
    }
    return putchar(']') == EOF ? ODEZZA_ERROR_IO : ODEZZA_SUCCESS;
}

static OdezzaResult o_print_inspection(const OdezzaScoringCubinInspection *inspection, size_t arena_size) {
    size_t index;
    if (inspection == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (printf("{\"arena_bytes\":%zu,\"cubin_size\":%zu,\"architecture\":%u,\"register_count\":%u,"
               "\"state_capacity\":%zu,\"constant_capacity\":%u,"
               "\"input_count\":%zu,\"output_count\":%zu,\"system_capacity\":%zu,"
               "\"function_symbol_index\":%u,\"function_file_offset\":%zu,\"function_byte_size\":%zu,"
               "\"scaffold_start\":%zu,\"scaffold_end\":%zu,\"shared_start\":%zu,\"shared_instruction_count\":%zu,"
               "\"arena_start\":%zu,\"arena_instruction_count\":%zu,\"requested_arena_instruction_count\":%zu,"
               "\"system_patch_capacity\":%zu,\"final_fallthrough_branch_elided\":%s,\"arena_end\":%zu,"
               "\"continuation\":%zu,\"incoming_wait_mask\":%u,\"predicate_register\":%u,",
               arena_size, inspection->cubin_size, inspection->architecture, inspection->register_count, inspection->state_capacity,
               inspection->constant_capacity, inspection->input_count, inspection->output_count, inspection->system_capacity,
               inspection->function_symbol_index, inspection->function_file_offset, inspection->function_byte_size,
               inspection->scaffold_start_file_offset, inspection->scaffold_end_file_offset, inspection->shared_start_file_offset,
               inspection->shared_instruction_count, inspection->arena_start_file_offset, inspection->arena_instruction_count,
               inspection->requested_arena_instruction_count, inspection->system_patch_capacity, inspection->final_fallthrough_branch_elided ? "true" : "false",
               inspection->arena_end_file_offset, inspection->continuation_file_offset, inspection->incoming_wait_mask, inspection->predicate_register) < 0)
        return ODEZZA_ERROR_IO;
#define O_PRINT_ARRAY(name, function, values, count) \
    do { \
        if (printf("\"%s\":", name) < 0) return ODEZZA_ERROR_IO; \
        if (function(values, count) != ODEZZA_SUCCESS) return ODEZZA_ERROR_IO; \
        if (putchar(',') == EOF) return ODEZZA_ERROR_IO; \
    } while (0)
    O_PRINT_ARRAY(
        "register_count_offsets",
        o_print_size_array,
        inspection->register_count_file_offsets,
        inspection->register_count_file_offset_count
    );
    O_PRINT_ARRAY(
        "register_count_header_offsets",
        o_print_size_array,
        inspection->register_count_header_file_offsets,
        inspection->register_count_header_file_offset_count
    );
    O_PRINT_ARRAY(
        "dispatch_offsets",
        o_print_size_array,
        inspection->dispatch_file_offsets,
        inspection->dispatch_instruction_count
    );
    O_PRINT_ARRAY(
        "dispatch_instructions",
        o_print_instruction_array,
        inspection->dispatch_instructions,
        inspection->dispatch_instruction_count
    );
    O_PRINT_ARRAY("input_registers", o_print_u8_array, inspection->input_registers, inspection->input_count);
    O_PRINT_ARRAY("output_registers", o_print_u8_array, inspection->output_registers, inspection->output_count);
    O_PRINT_ARRAY(
        "final_output_registers",
        o_print_u8_array,
        inspection->final_output_registers,
        inspection->output_count
    );
    O_PRINT_ARRAY(
        "output_materialization_offsets",
        o_print_size_array,
        inspection->output_materialization_file_offsets,
        inspection->output_count
    );
    O_PRINT_ARRAY(
        "available_registers",
        o_print_u8_array,
        inspection->available_registers,
        inspection->available_register_count
    );
    O_PRINT_ARRAY(
        "cleanup_offsets",
        o_print_size_array,
        inspection->cleanup_file_offsets,
        inspection->cleanup_file_offset_count
    );
    O_PRINT_ARRAY(
        "target_table_offsets",
        o_print_size_array,
        inspection->target_table_file_offsets,
        inspection->system_capacity
    );
    O_PRINT_ARRAY(
        "original_target_values",
        o_print_u32_array,
        inspection->original_target_values,
        inspection->system_capacity
    );
#undef O_PRINT_ARRAY
    if (printf("\"template_id_file_offset\":%zu,\"template_id\":\"", inspection->template_id_file_offset) < 0) return ODEZZA_ERROR_IO;
    for (index = 0u; index < sizeof(inspection->template_id); ++index)
        if (printf("%02x", inspection->template_id[index]) < 0) return ODEZZA_ERROR_IO;
    return puts("\"}") < 0 ? ODEZZA_ERROR_IO : ODEZZA_SUCCESS;
}

static OdezzaResult o_test_valid_cubin(const char *path) {
    unsigned char *cubin;
    size_t cubin_size;
    size_t required;
    size_t reported;
    unsigned char *allocation;
    unsigned char *arena;
    uintptr_t aligned;
    const OdezzaScoringCubinInspection *inspection;
    OdezzaResult result;
    size_t index;

    O_TRY(o_read_file(path, &cubin, &cubin_size));
    result = odezza_inspect_scoring_cubin(cubin, cubin_size, NULL, 123u, &required, NULL);
    if (result != ODEZZA_SUCCESS || required < sizeof(*inspection)) {
        free(cubin);
        return ODEZZA_ERROR_FORMAT;
    }
    allocation = (unsigned char *)malloc(required + 8u);
    if (allocation == NULL) {
        free(cubin);
        return ODEZZA_ERROR_ALLOCATION;
    }
    aligned = ((uintptr_t)allocation + 7u) & ~(uintptr_t)7u;
    arena = (unsigned char *)aligned;
    memset(arena, 0xa5, required);
    reported = 0u;
    result = odezza_inspect_scoring_cubin(cubin, cubin_size, arena, required - 1u, &reported, &inspection);
    if (o_expect_result(result, ODEZZA_ERROR_INSUFFICIENT_BUFFER, "undersized arena") != ODEZZA_SUCCESS || reported != required || inspection != NULL) {
        free(allocation);
        free(cubin);
        return ODEZZA_ERROR_FORMAT;
    }
    for (index = 0u; index < required; ++index) {
        if (arena[index] != 0xa5u) {
            free(allocation);
            free(cubin);
            return ODEZZA_ERROR_FORMAT;
        }
    }
    result = odezza_inspect_scoring_cubin(cubin, cubin_size, arena + 1u, required - 1u, &reported, &inspection);
    if (o_expect_result(result, ODEZZA_ERROR_INVALID_ARGUMENT, "misaligned arena") != ODEZZA_SUCCESS || inspection != NULL) {
        free(allocation);
        free(cubin);
        return ODEZZA_ERROR_FORMAT;
    }
    result = odezza_inspect_scoring_cubin(cubin, cubin_size, arena, required, &reported, &inspection);
    if (result != ODEZZA_SUCCESS || reported != required || inspection == NULL) {
        free(allocation);
        free(cubin);
        return ODEZZA_ERROR_FORMAT;
    }
    if (o_pointer_in_arena(inspection, arena, required) != ODEZZA_SUCCESS ||
        (inspection->dispatch_instruction_count != 0u &&
         o_pointer_in_arena(inspection->dispatch_instructions, arena, required) != ODEZZA_SUCCESS) ||
        (inspection->cleanup_file_offset_count != 0u &&
         o_pointer_in_arena(inspection->cleanup_file_offsets, arena, required) != ODEZZA_SUCCESS) ||
        (inspection->system_capacity != 1u &&
         o_pointer_in_arena(inspection->original_target_values, arena, required) != ODEZZA_SUCCESS)) {
        free(allocation);
        free(cubin);
        return ODEZZA_ERROR_FORMAT;
    }
    result = o_print_inspection(inspection, required);
    free(allocation);
    free(cubin);
    return result;
}

int main(int argc, char **argv) {
    OdezzaResult result = o_test_rejections();
    if (result == ODEZZA_SUCCESS && argc == 2)
        result = o_test_valid_cubin(argv[1]);
    else if (result == ODEZZA_SUCCESS && argc != 1) {
        fprintf(stderr, "usage: %s [scoring.cubin]\n", argv[0]);
        return 2;
    }
    if (result != ODEZZA_SUCCESS) {
        fprintf(stderr, "C99 scoring CUBIN inspection tests failed with result %d\n", (int)result);
        return 1;
    }
    return 0;
}
