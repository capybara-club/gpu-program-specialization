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
#include "o_ast_tools.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct OCollect {
    uint64_t ranks[64];
    size_t count;
    const float *states;
    size_t state_count;
    const float *constants;
    size_t constant_count;
    uint32_t toggle_bit_count;
    uint32_t permutation;
} OCollect;

typedef struct OHash {
    uint64_t value;
    uint64_t count;
} OHash;

typedef struct OSystemCollect {
    OHash hash;
    size_t expected_rhs_count;
    uint32_t expected_depth[8];
} OSystemCollect;

static int o_close(float lhs, float rhs) {
    return fabsf(lhs - rhs) <= 1.0e-6f;
}

static OAstToolResult o_collect(const OAstSpaceItem *item, void *context) {
    OCollect *collect = (OCollect *)context;
    float value;
    if (collect->count == sizeof(collect->ranks) / sizeof(collect->ranks[0])) return O_AST_TOOL_ERROR_CAPACITY;
    if (o_ast_cpu_evaluate(
            item->program,
            collect->states,
            collect->state_count,
            collect->constants,
            collect->constant_count,
            collect->toggle_bit_count,
            collect->permutation,
            &value
        ) != O_AST_TOOL_SUCCESS
    ) {
        return O_AST_TOOL_ERROR_PROGRAM;
    }
    collect->ranks[collect->count++] = item->rank;
    return O_AST_TOOL_SUCCESS;
}

static void o_hash_bytes(OHash *hash, const uint8_t *bytes, size_t byte_count) {
    size_t index;
    for (index = 0u; index < byte_count; ++index) {
        hash->value ^= bytes[index];
        hash->value *= UINT64_C(1099511628211);
    }
}

static OAstToolResult o_hash_item(const OAstSpaceItem *item, void *context) {
    OHash *hash = (OHash *)context;
    uint8_t metadata[8];
    size_t index;
    for (index = 0u; index < 4u; ++index) metadata[index] = (uint8_t)(item->depth >> (8u * index));
    for (index = 0u; index < 4u; ++index) metadata[4u + index] = (uint8_t)(item->program.byte_count >> (8u * index));
    o_hash_bytes(hash, metadata, sizeof(metadata));
    o_hash_bytes(hash, item->program.bytes, item->program.byte_count);
    ++hash->count;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_collect_system(const OAstSystemSpaceItem *item, void *context) {
    OSystemCollect *collect = (OSystemCollect *)context;
    size_t index;
    if (item->rhs_count != collect->expected_rhs_count) return O_AST_TOOL_ERROR_PROGRAM;
    for (index = 0u; index < item->rhs_count; ++index) {
        uint8_t metadata[6];
        if (item->rhs[index].state_index != index || item->rhs[index].depth != collect->expected_depth[index]) {
            return O_AST_TOOL_ERROR_PROGRAM;
        }
        metadata[0] = item->rhs[index].state_index;
        metadata[1] = (uint8_t)item->rhs[index].depth;
        metadata[2] = (uint8_t)item->rhs[index].program.byte_count;
        metadata[3] = (uint8_t)(item->rhs[index].program.byte_count >> 8u);
        metadata[4] = (uint8_t)(item->rhs[index].program.byte_count >> 16u);
        metadata[5] = (uint8_t)(item->rhs[index].program.byte_count >> 24u);
        o_hash_bytes(&collect->hash, metadata, sizeof(metadata));
        o_hash_bytes(&collect->hash, item->rhs[index].program.bytes, item->rhs[index].program.byte_count);
    }
    ++collect->hash.count;
    return O_AST_TOOL_SUCCESS;
}

static int o_test_system_generation(void) {
    static const uint8_t add[] = {O_AST_ADD_F32};
    static const uint8_t neg[] = {O_AST_NEG_F32};
    OAstRhsSpaceConfig rhs[2];
    OAstSystemSpaceConfig config;
    OAstSelection selection;
    OAstSystemSpaceReport report;
    OSystemCollect collect;
    void *workspace;
    size_t workspace_size;
    size_t workspace_alignment;
    uint64_t count;
    int exact;
    int success;
    memset(rhs, 0, sizeof(rhs));
    rhs[0].state_index = 0u;
    rhs[0].expression.state_count = 2u;
    rhs[0].expression.operators = add;
    rhs[0].expression.operator_count = 1u;
    rhs[0].expression.maximum_depth = 1u;
    rhs[0].expression.depth_mode = O_AST_DEPTH_EXACT;
    rhs[1].state_index = 1u;
    rhs[1].expression.state_count = 2u;
    rhs[1].expression.operators = neg;
    rhs[1].expression.operator_count = 1u;
    rhs[1].expression.maximum_depth = 1u;
    rhs[1].expression.depth_mode = O_AST_DEPTH_EXACT;
    config.rhs = rhs;
    config.rhs_count = 2u;
    if (o_ast_system_space_count(&config, &count, &exact) != O_AST_TOOL_SUCCESS || count != 8u || !exact) return 0;
    if (o_ast_system_space_workspace_size(&config, &workspace_size, &workspace_alignment) != O_AST_TOOL_SUCCESS) return 0;
    workspace = malloc(workspace_size);
    if (workspace == NULL || (uintptr_t)workspace % workspace_alignment != 0u) {
        free(workspace);
        return 0;
    }
    memset(&selection, 0, sizeof(selection));
    selection.mode = O_AST_SELECT_EXHAUSTIVE;
    memset(&collect, 0, sizeof(collect));
    collect.hash.value = UINT64_C(1469598103934665603);
    collect.expected_rhs_count = 2u;
    collect.expected_depth[0] = 1u;
    collect.expected_depth[1] = 1u;
    success = o_ast_system_space_iterate(
        &config,
        &selection,
        workspace,
        workspace_size,
        o_collect_system,
        &collect,
        &report
    ) == O_AST_TOOL_SUCCESS && collect.hash.count == 8u && report.emitted_count == 8u &&
        report.selection_uses_space_ranks;
    free(workspace);
    return success;
}

static int o_test_structural_random_system_generation(void) {
    static const uint8_t operators[] = {
        O_AST_ADD_F32,
        O_AST_SUB_F32,
        O_AST_MUL_F32,
        O_AST_DIV_F32,
        O_AST_NEG_F32,
        O_AST_SIN_F32,
        O_AST_COS_F32
    };
    OAstRhsSpaceConfig rhs[4];
    OAstSystemSpaceConfig config;
    OAstSelection selection;
    OAstSystemSpaceReport report;
    OSystemCollect first;
    OSystemCollect second;
    void *workspace;
    size_t workspace_size;
    size_t workspace_alignment;
    uint64_t count;
    int exact;
    int success;
    size_t index;
    memset(rhs, 0, sizeof(rhs));
    for (index = 0u; index < 4u; ++index) {
        rhs[index].state_index = (uint8_t)index;
        rhs[index].expression.state_count = 4u;
        rhs[index].expression.constant_count = 4u;
        rhs[index].expression.operators = operators;
        rhs[index].expression.operator_count = sizeof(operators) / sizeof(operators[0]);
        rhs[index].expression.maximum_depth = 4u;
        rhs[index].expression.depth_mode = O_AST_DEPTH_EXACT;
    }
    config.rhs = rhs;
    config.rhs_count = 4u;
    if (o_ast_system_space_count(&config, &count, &exact) != O_AST_TOOL_SUCCESS || count != UINT64_MAX || exact) return 0;
    if (o_ast_system_space_workspace_size(&config, &workspace_size, &workspace_alignment) != O_AST_TOOL_SUCCESS) return 0;
    workspace = malloc(workspace_size);
    if (workspace == NULL || (uintptr_t)workspace % workspace_alignment != 0u) {
        free(workspace);
        return 0;
    }
    memset(&selection, 0, sizeof(selection));
    selection.mode = O_AST_SELECT_STRUCTURAL_RANDOM;
    selection.count = 32u;
    selection.seed = UINT64_C(0xdecafbad12345678);
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    first.hash.value = second.hash.value = UINT64_C(1469598103934665603);
    first.expected_rhs_count = second.expected_rhs_count = 4u;
    for (index = 0u; index < 4u; ++index) first.expected_depth[index] = second.expected_depth[index] = 4u;
    success = o_ast_system_space_iterate(
        &config,
        &selection,
        workspace,
        workspace_size,
        o_collect_system,
        &first,
        &report
    ) == O_AST_TOOL_SUCCESS && report.emitted_count == 32u && !report.selection_uses_space_ranks;
    if (success) {
        success = o_ast_system_space_iterate(
            &config,
            &selection,
            workspace,
            workspace_size,
            o_collect_system,
            &second,
            NULL
        ) == O_AST_TOOL_SUCCESS && first.hash.count == second.hash.count && first.hash.value == second.hash.value;
    }
    free(workspace);
    return success;
}

static int o_test_counts_and_selection(void) {
    static const uint8_t operators[] = {O_AST_ADD_F32};
    static const float states[] = {2.0f, 5.0f};
    OAstSpaceConfig config;
    OAstSelection selection;
    OAstSpaceReport report;
    OCollect first;
    OCollect second;
    uint64_t count;
    int exact;
    size_t index;
    size_t other;
    memset(&config, 0, sizeof(config));
    config.state_count = 2u;
    config.operators = operators;
    config.operator_count = 1u;
    config.maximum_depth = 2u;
    config.depth_mode = O_AST_DEPTH_UP_TO;
    if (o_ast_space_count(&config, &count, &exact) != O_AST_TOOL_SUCCESS || count != 38u || !exact) return 0;
    config.depth_mode = O_AST_DEPTH_EXACT;
    if (o_ast_space_count(&config, &count, &exact) != O_AST_TOOL_SUCCESS || count != 32u || !exact) return 0;
    config.depth_mode = O_AST_DEPTH_UP_TO;
    memset(&selection, 0, sizeof(selection));
    selection.mode = O_AST_SELECT_TRUNCATE;
    selection.count = 3u;
    memset(&first, 0, sizeof(first));
    first.states = states;
    first.state_count = 2u;
    if (o_ast_space_iterate(&config, &selection, o_collect, &first, &report) != O_AST_TOOL_SUCCESS) return 0;
    if (first.count != 3u || first.ranks[0] != 0u || first.ranks[1] != 1u || first.ranks[2] != 2u) return 0;
    selection.mode = O_AST_SELECT_RANDOM;
    selection.count = 16u;
    selection.seed = UINT64_C(0x123456789abcdef0);
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    first.states = second.states = states;
    first.state_count = second.state_count = 2u;
    if (o_ast_space_iterate(&config, &selection, o_collect, &first, NULL) != O_AST_TOOL_SUCCESS) return 0;
    if (o_ast_space_iterate(&config, &selection, o_collect, &second, NULL) != O_AST_TOOL_SUCCESS) return 0;
    if (first.count != 16u || second.count != 16u || memcmp(first.ranks, second.ranks, 16u * sizeof(uint64_t)) != 0) return 0;
    for (index = 0u; index < first.count; ++index) {
        if (first.ranks[index] >= 38u) return 0;
        for (other = 0u; other < index; ++other) {
            if (first.ranks[index] == first.ranks[other]) return 0;
        }
    }
    return 1;
}

static int o_test_toggle_evaluation(void) {
    static const float states[] = {2.0f, 5.0f};
    static const float constants[] = {7.0f};
    static const uint8_t toggle2_bytes[] = {
        O_AST_STATE_F32, 0u,
        O_AST_STATE_F32, 1u,
        O_AST_TOGGLE2_F32, 1u,
        O_AST_RETURN_F32
    };
    static const uint8_t toggle4_bytes[] = {
        O_AST_STATE_F32, 0u,
        O_AST_STATE_F32, 1u,
        O_AST_LITERAL_F32, 0u, 0u, 64u, 64u,
        O_AST_CONSTANT_F32, 0u,
        O_AST_TOGGLE4_F32, 0u, 1u,
        O_AST_RETURN_F32
    };
    OAstProgramView program;
    float value;
    program.bytes = toggle2_bytes;
    program.byte_count = sizeof(toggle2_bytes);
    if (o_ast_cpu_evaluate(
            program,
            states,
            2u,
            constants,
            1u,
            2u,
            0u,
            &value
        ) != O_AST_TOOL_SUCCESS || !o_close(value, 2.0f)
    ) {
        return 0;
    }
    if (o_ast_cpu_evaluate(
            program,
            states,
            2u,
            constants,
            1u,
            2u,
            2u,
            &value
        ) != O_AST_TOOL_SUCCESS || !o_close(value, 5.0f)
    ) {
        return 0;
    }
    program.bytes = toggle4_bytes;
    program.byte_count = sizeof(toggle4_bytes);
    if (o_ast_cpu_evaluate(
            program,
            states,
            2u,
            constants,
            1u,
            2u,
            0u,
            &value
        ) != O_AST_TOOL_SUCCESS || !o_close(value, 2.0f)
    ) {
        return 0;
    }
    if (o_ast_cpu_evaluate(
            program,
            states,
            2u,
            constants,
            1u,
            2u,
            1u,
            &value
        ) != O_AST_TOOL_SUCCESS || !o_close(value, 5.0f)
    ) {
        return 0;
    }
    if (o_ast_cpu_evaluate(
            program,
            states,
            2u,
            constants,
            1u,
            2u,
            2u,
            &value
        ) != O_AST_TOOL_SUCCESS || !o_close(value, 3.0f)
    ) {
        return 0;
    }
    if (o_ast_cpu_evaluate(
            program,
            states,
            2u,
            constants,
            1u,
            2u,
            3u,
            &value
        ) != O_AST_TOOL_SUCCESS || !o_close(value, 7.0f)
    ) {
        return 0;
    }
    return 1;
}

static int o_test_generated_toggles(void) {
    static const float states[] = {2.0f, 5.0f};
    OAstSpaceConfig config;
    OAstSelection selection;
    OAstSpaceReport report;
    OCollect collect;
    memset(&config, 0, sizeof(config));
    config.state_count = 2u;
    config.toggle_bit_count = 1u;
    config.include_toggle2 = 1;
    config.depth_mode = O_AST_DEPTH_EXACT;
    memset(&selection, 0, sizeof(selection));
    selection.mode = O_AST_SELECT_EXHAUSTIVE;
    memset(&collect, 0, sizeof(collect));
    collect.states = states;
    collect.state_count = 2u;
    collect.toggle_bit_count = 1u;
    collect.permutation = 1u;
    if (o_ast_space_iterate(&config, &selection, o_collect, &collect, &report) != O_AST_TOOL_SUCCESS) return 0;
    return report.space_count == 6u && report.emitted_count == 6u && collect.count == 6u;
}

static int o_test_saturated_prefix(void) {
    static const uint8_t operators[] = {O_AST_ADD_F32};
    static const float states[] = {2.0f, 5.0f};
    OAstSpaceConfig config;
    OAstSelection selection;
    OAstSpaceReport report;
    OCollect collect;
    memset(&config, 0, sizeof(config));
    config.state_count = 2u;
    config.operators = operators;
    config.operator_count = 1u;
    config.maximum_depth = 10u;
    config.depth_mode = O_AST_DEPTH_UP_TO;
    memset(&selection, 0, sizeof(selection));
    selection.mode = O_AST_SELECT_TRUNCATE;
    selection.count = 5u;
    memset(&collect, 0, sizeof(collect));
    collect.states = states;
    collect.state_count = 2u;
    if (o_ast_space_iterate(&config, &selection, o_collect, &collect, &report) != O_AST_TOOL_SUCCESS) return 0;
    if (report.space_count != UINT64_MAX || report.space_count_is_exact || report.emitted_count != 5u) return 0;
    config.depth_mode = O_AST_DEPTH_EXACT;
    return o_ast_space_iterate(&config, &selection, o_collect, &collect, NULL) == O_AST_TOOL_ERROR_OVERFLOW;
}

static int o_replace_file_byte(const char *path, long offset, int origin, uint8_t replacement, uint8_t *previous_ret) {
    FILE *file = fopen(path, "r+b");
    int value;
    int success = 0;
    if (file != NULL && fseek(file, offset, origin) == 0) {
        value = fgetc(file);
        if (value != EOF && fseek(file, -1L, SEEK_CUR) == 0 && fputc(replacement, file) != EOF && fflush(file) == 0) {
            *previous_ret = (uint8_t)value;
            success = 1;
        }
    }
    if (file != NULL && fclose(file) != 0) success = 0;
    return success;
}

static int o_append_file_byte(const char *path, uint8_t value) {
    FILE *file = fopen(path, "ab");
    int success = file != NULL && fputc(value, file) != EOF && fflush(file) == 0;
    if (file != NULL && fclose(file) != 0) success = 0;
    return success;
}

static int o_test_file_roundtrip(void) {
    static const uint8_t operators[] = {O_AST_ADD_F32, O_AST_MUL_F32, O_AST_NEG_F32};
    static const uint32_t literal_bits[] = {0u, UINT32_C(0x3f800000)};
    static const char path[] = "build/o_ast_tools_roundtrip.bin";
    OAstSpaceConfig config;
    OAstSelection selection;
    OAstSpaceReport space_report;
    OAstFileReport write_report;
    OAstFileReport read_report;
    OHash expected;
    OHash actual;
    uint8_t previous;
    int success = 1;
    memset(&config, 0, sizeof(config));
    config.state_count = 3u;
    config.constant_count = 2u;
    config.literal_bits = literal_bits;
    config.literal_count = sizeof(literal_bits) / sizeof(literal_bits[0]);
    config.toggle_bit_count = 2u;
    config.include_toggle2 = 1;
    config.include_toggle4 = 1;
    config.operators = operators;
    config.operator_count = sizeof(operators) / sizeof(operators[0]);
    config.maximum_depth = 2u;
    config.depth_mode = O_AST_DEPTH_EXACT;
    memset(&selection, 0, sizeof(selection));
    selection.mode = O_AST_SELECT_RANDOM;
    selection.count = 257u;
    selection.seed = UINT64_C(0x4f4445415845);
    memset(&expected, 0, sizeof(expected));
    expected.value = UINT64_C(1469598103934665603);
    if (o_ast_space_iterate(&config, &selection, o_hash_item, &expected, &space_report) != O_AST_TOOL_SUCCESS) success = 0;
    if (success && o_ast_file_write(path, &config, &selection, &write_report) != O_AST_TOOL_SUCCESS) success = 0;
    memset(&actual, 0, sizeof(actual));
    actual.value = UINT64_C(1469598103934665603);
    if (success && o_ast_file_read(path, o_hash_item, &actual, &read_report) != O_AST_TOOL_SUCCESS) success = 0;
    if (success && (expected.count != 257u || actual.count != expected.count || actual.value != expected.value)) success = 0;
    if (
        success && (
            write_report.record_count != expected.count ||
            memcmp(&write_report, &read_report, sizeof(write_report)) != 0
        )
    ) {
        success = 0;
    }
    if (success && !o_replace_file_byte(path, 0L, SEEK_SET, (uint8_t)'X', &previous)) success = 0;
    if (success && o_ast_file_read(path, NULL, NULL, NULL) != O_AST_TOOL_ERROR_FILE_FORMAT) success = 0;
    if (success && !o_replace_file_byte(path, 0L, SEEK_SET, previous, &previous)) success = 0;
    if (success && o_ast_file_read(path, NULL, NULL, NULL) != O_AST_TOOL_SUCCESS) success = 0;
    if (success && !o_replace_file_byte(path, -1L, SEEK_END, UINT8_C(0xa5), &previous)) success = 0;
    if (success && o_ast_file_read(path, NULL, NULL, NULL) != O_AST_TOOL_ERROR_FILE_FORMAT) success = 0;
    if (success && !o_replace_file_byte(path, -1L, SEEK_END, previous, &previous)) success = 0;
    if (success && !o_append_file_byte(path, 0u)) success = 0;
    if (success && o_ast_file_read(path, NULL, NULL, NULL) != O_AST_TOOL_ERROR_FILE_FORMAT) success = 0;
    if (remove(path) != 0) success = 0;
    return success;
}

int main(void) {
    if (!o_test_counts_and_selection()) {
        fprintf(stderr, "AST count or selection test failed\n");
        return 1;
    }
    if (!o_test_toggle_evaluation()) {
        fprintf(stderr, "AST toggle evaluation test failed\n");
        return 1;
    }
    if (!o_test_generated_toggles()) {
        fprintf(stderr, "generated toggle test failed\n");
        return 1;
    }
    if (!o_test_saturated_prefix()) {
        fprintf(stderr, "saturated prefix test failed\n");
        return 1;
    }
    if (!o_test_file_roundtrip()) {
        fprintf(stderr, "AST file round-trip or corruption test failed\n");
        return 1;
    }
    if (!o_test_system_generation()) {
        fprintf(stderr, "multi-RHS system generation test failed\n");
        return 1;
    }
    if (!o_test_structural_random_system_generation()) {
        fprintf(stderr, "structural-random system generation test failed\n");
        return 1;
    }
    printf("C99 AST enumeration and CPU evaluation: verified\n");
    return 0;
}
