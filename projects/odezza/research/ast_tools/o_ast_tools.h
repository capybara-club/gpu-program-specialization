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
#ifndef O_AST_TOOLS_H
#define O_AST_TOOLS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define O_AST_TOOL_MAX_DEPTH 32u
#define O_AST_TOOL_MAX_INSTRUCTIONS 2048u
#define O_AST_TOOL_MAX_STACK_DEPTH 128u
#define O_AST_TOOL_MAX_PROGRAM_BYTES (5u * O_AST_TOOL_MAX_INSTRUCTIONS)

/* These values intentionally match odezza-postorder-f32-v2. */
typedef enum OAstToolOpcode {
    O_AST_RETURN_F32 = 0x80,
    O_AST_STATE_F32 = 0x81,
    O_AST_CONSTANT_F32 = 0x82,
    O_AST_LITERAL_F32 = 0x83,
    O_AST_TOGGLE2_F32 = 0x84,
    O_AST_TOGGLE4_F32 = 0x85,
    O_AST_ADD_F32 = 0x90,
    O_AST_SUB_F32 = 0x91,
    O_AST_MUL_F32 = 0x92,
    O_AST_DIV_F32 = 0x93,
    O_AST_NEG_F32 = 0x94,
    O_AST_SQRT_F32 = 0x95,
    O_AST_RCP_F32 = 0x96,
    O_AST_ABS_F32 = 0x97,
    O_AST_MIN_F32 = 0x98,
    O_AST_MAX_F32 = 0x99,
    O_AST_FMA_F32 = 0x9a,
    O_AST_SIN_F32 = 0x9b,
    O_AST_COS_F32 = 0x9c,
    O_AST_EX2_F32 = 0x9d,
    O_AST_LG2_F32 = 0x9e,
    O_AST_RSQRT_F32 = 0x9f,
    O_AST_TANH_F32 = 0xa0,
    O_AST_EXP_F32 = 0xa1,
    O_AST_LOG_F32 = 0xa2
} OAstToolOpcode;

typedef enum OAstToolResult {
    O_AST_TOOL_SUCCESS = 0,
    O_AST_TOOL_ERROR_INVALID_ARGUMENT = 1,
    O_AST_TOOL_ERROR_OVERFLOW = 2,
    O_AST_TOOL_ERROR_CAPACITY = 3,
    O_AST_TOOL_ERROR_PROGRAM = 4,
    O_AST_TOOL_ERROR_CALLBACK = 5,
    O_AST_TOOL_ERROR_IO = 6,
    O_AST_TOOL_ERROR_FILE_FORMAT = 7
} OAstToolResult;

typedef enum OAstDepthMode {
    O_AST_DEPTH_UP_TO = 0,
    O_AST_DEPTH_EXACT = 1
} OAstDepthMode;

typedef enum OAstSelectionMode {
    O_AST_SELECT_EXHAUSTIVE = 0,
    O_AST_SELECT_TRUNCATE = 1,
    O_AST_SELECT_RANDOM = 2,
    O_AST_SELECT_STRUCTURAL_RANDOM = 3
} OAstSelectionMode;

typedef struct OAstSpaceConfig {
    uint32_t state_count;
    uint32_t constant_count;
    const uint32_t *literal_bits;
    size_t literal_count;

    uint32_t toggle_bit_count;
    int include_toggle2;
    int include_toggle4;

    const uint8_t *operators;
    size_t operator_count;
    uint32_t maximum_depth;
    OAstDepthMode depth_mode;
} OAstSpaceConfig;

typedef struct OAstSelection {
    OAstSelectionMode mode;
    uint64_t count;
    uint64_t seed;
} OAstSelection;

typedef struct OAstProgramView {
    const uint8_t *bytes;
    size_t byte_count;
} OAstProgramView;

typedef struct OAstSpaceItem {
    uint64_t rank;
    uint32_t depth;
    OAstProgramView program;
} OAstSpaceItem;

typedef struct OAstSpaceReport {
    uint64_t space_count;
    uint64_t selected_count;
    uint64_t emitted_count;
    int space_count_is_exact;
} OAstSpaceReport;

typedef struct OAstRhsSpaceConfig {
    uint8_t state_index;
    OAstSpaceConfig expression;
} OAstRhsSpaceConfig;

typedef struct OAstSystemSpaceConfig {
    const OAstRhsSpaceConfig *rhs;
    size_t rhs_count;
} OAstSystemSpaceConfig;

typedef struct OAstSystemRhsItem {
    uint8_t state_index;
    uint64_t expression_rank;
    uint32_t depth;
    OAstProgramView program;
} OAstSystemRhsItem;

typedef struct OAstSystemSpaceItem {
    uint64_t rank;
    int rank_is_space_rank;
    const OAstSystemRhsItem *rhs;
    size_t rhs_count;
} OAstSystemSpaceItem;

typedef struct OAstSystemSpaceReport {
    uint64_t space_count;
    uint64_t selected_count;
    uint64_t emitted_count;
    int space_count_is_exact;
    int selection_uses_space_ranks;
} OAstSystemSpaceReport;

typedef struct OAstFileReport {
    uint64_t record_count;
    uint64_t program_bytes;
    uint32_t maximum_program_bytes;
} OAstFileReport;

typedef OAstToolResult (*OAstSpaceVisit)(const OAstSpaceItem *item, void *context);
typedef OAstToolResult (*OAstSystemSpaceVisit)(const OAstSystemSpaceItem *item, void *context);

OAstToolResult o_ast_space_count(
    const OAstSpaceConfig *config,
    uint64_t *count_ret,
    int *count_is_exact_ret
);

/* The item byte storage remains valid only for the duration of the callback. */
OAstToolResult o_ast_space_iterate(
    const OAstSpaceConfig *config,
    const OAstSelection *selection,
    OAstSpaceVisit visit,
    void *context,
    OAstSpaceReport *report_ret
);

OAstToolResult o_ast_system_space_count(
    const OAstSystemSpaceConfig *config,
    uint64_t *count_ret,
    int *count_is_exact_ret
);

OAstToolResult o_ast_system_space_workspace_size(
    const OAstSystemSpaceConfig *config,
    size_t *workspace_size_ret,
    size_t *workspace_alignment_ret
);

/* The item and program storage remain valid only for the callback duration. */
OAstToolResult o_ast_system_space_iterate(
    const OAstSystemSpaceConfig *config,
    const OAstSelection *selection,
    void *workspace,
    size_t workspace_size,
    OAstSystemSpaceVisit visit,
    void *context,
    OAstSystemSpaceReport *report_ret
);

OAstToolResult o_ast_cpu_evaluate(
    OAstProgramView program,
    const float *states,
    size_t state_count,
    const float *constants,
    size_t constant_count,
    uint32_t toggle_bit_count,
    uint32_t permutation,
    float *value_ret
);

OAstToolResult o_ast_file_write(
    const char *path,
    const OAstSpaceConfig *config,
    const OAstSelection *selection,
    OAstFileReport *report_ret
);

/* Writes every RHS consecutively in the existing AST file format. */
OAstToolResult o_ast_system_file_write_flat(
    const char *path,
    const OAstSystemSpaceConfig *config,
    const OAstSelection *selection,
    void *workspace,
    size_t workspace_size,
    OAstFileReport *report_ret,
    OAstSystemSpaceReport *system_report_ret
);

/* The item byte storage remains valid only for the duration of the callback. */
OAstToolResult o_ast_file_read(
    const char *path,
    OAstSpaceVisit visit,
    void *context,
    OAstFileReport *report_ret
);

const char *o_ast_tool_result_string(OAstToolResult result);

#ifdef __cplusplus
}
#endif

#endif
