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

#include <stdio.h>
#include <string.h>

#define O_AST_FILE_HEADER_BYTES 24u
#define O_AST_FILE_VERSION 1u

static const uint8_t o_ast_file_magic[8] = {'O', 'D', 'E', 'Z', 'Z', 'A', 'A', 'S'};

typedef struct OAstFileWriter {
    FILE *file;
    OAstFileReport report;
    OAstToolResult result;
} OAstFileWriter;

typedef struct OAstSystemFileWriter {
    OAstFileWriter ast;
    uint64_t system_count;
} OAstSystemFileWriter;

typedef struct OAstFileStackValue {
    uint32_t tree_depth;
    int direct_leaf;
} OAstFileStackValue;

static void o_store_u32(uint8_t bytes[4], uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static void o_store_u64(uint8_t bytes[8], uint64_t value) {
    size_t index;
    for (index = 0u; index < 8u; ++index) bytes[index] = (uint8_t)(value >> (8u * index));
}

static uint32_t o_load_u32(const uint8_t bytes[4]) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static uint64_t o_load_u64(const uint8_t bytes[8]) {
    uint64_t value = 0u;
    size_t index;
    for (index = 0u; index < 8u; ++index) value |= (uint64_t)bytes[index] << (8u * index);
    return value;
}

static int o_write_all(FILE *file, const void *bytes, size_t byte_count) {
    return byte_count == 0u || fwrite(bytes, 1u, byte_count, file) == byte_count;
}

static int o_read_byte(FILE *file, uint8_t *value_ret) {
    int value = fgetc(file);
    if (value == EOF) return 0;
    *value_ret = (uint8_t)value;
    return 1;
}

static void o_encode_header(uint8_t header[O_AST_FILE_HEADER_BYTES], const OAstFileReport *report) {
    memset(header, 0, O_AST_FILE_HEADER_BYTES);
    memcpy(header, o_ast_file_magic, sizeof(o_ast_file_magic));
    o_store_u32(header + 8u, O_AST_FILE_VERSION);
    o_store_u32(header + 12u, report->maximum_program_bytes);
    o_store_u64(header + 16u, report->record_count);
}

static OAstToolResult o_decode_header(const uint8_t header[O_AST_FILE_HEADER_BYTES], OAstFileReport *report) {
    if (memcmp(header, o_ast_file_magic, sizeof(o_ast_file_magic)) != 0 || o_load_u32(header + 8u) != O_AST_FILE_VERSION) {
        return O_AST_TOOL_ERROR_FILE_FORMAT;
    }
    report->maximum_program_bytes = o_load_u32(header + 12u);
    report->record_count = o_load_u64(header + 16u);
    if (report->maximum_program_bytes > O_AST_TOOL_MAX_PROGRAM_BYTES ||
        (report->record_count != 0u && report->maximum_program_bytes == 0u)
    ) {
        return O_AST_TOOL_ERROR_FILE_FORMAT;
    }
    return O_AST_TOOL_SUCCESS;
}

static size_t o_ast_file_operator_arity(uint8_t opcode) {
    switch ((OAstToolOpcode)opcode) {
    case O_AST_NEG_F32:
    case O_AST_SQRT_F32:
    case O_AST_RCP_F32:
    case O_AST_ABS_F32:
    case O_AST_SIN_F32:
    case O_AST_COS_F32:
    case O_AST_EX2_F32:
    case O_AST_LG2_F32:
    case O_AST_RSQRT_F32:
    case O_AST_TANH_F32:
    case O_AST_EXP_F32:
    case O_AST_LOG_F32:
        return 1u;
    case O_AST_ADD_F32:
    case O_AST_SUB_F32:
    case O_AST_MUL_F32:
    case O_AST_DIV_F32:
    case O_AST_MIN_F32:
    case O_AST_MAX_F32:
        return 2u;
    case O_AST_FMA_F32:
        return 3u;
    default:
        return 0u;
    }
}

static OAstToolResult o_ast_file_read_immediates(
    FILE *file,
    uint8_t *program,
    size_t capacity,
    size_t *byte_count,
    size_t immediate_count
) {
    size_t index;
    if (immediate_count > capacity - *byte_count) return O_AST_TOOL_ERROR_FILE_FORMAT;
    for (index = 0u; index < immediate_count; ++index) {
        if (!o_read_byte(file, program + *byte_count)) return ferror(file) ? O_AST_TOOL_ERROR_IO : O_AST_TOOL_ERROR_FILE_FORMAT;
        ++*byte_count;
    }
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_ast_file_read_program(
    FILE *file,
    uint32_t maximum_program_bytes,
    uint8_t program[O_AST_TOOL_MAX_PROGRAM_BYTES],
    size_t *byte_count_ret,
    uint32_t *tree_depth_ret
) {
    OAstFileStackValue stack[O_AST_TOOL_MAX_STACK_DEPTH];
    size_t stack_count = 0u;
    size_t byte_count = 0u;
    size_t instruction_count = 0u;
    OAstToolResult result = O_AST_TOOL_SUCCESS;
    while (result == O_AST_TOOL_SUCCESS) {
        uint8_t opcode;
        size_t arity;
        if (byte_count == maximum_program_bytes || instruction_count == O_AST_TOOL_MAX_INSTRUCTIONS) {
            return O_AST_TOOL_ERROR_FILE_FORMAT;
        }
        if (!o_read_byte(file, &opcode)) return ferror(file) ? O_AST_TOOL_ERROR_IO : O_AST_TOOL_ERROR_FILE_FORMAT;
        program[byte_count++] = opcode;
        ++instruction_count;
        switch ((OAstToolOpcode)opcode) {
        case O_AST_STATE_F32:
        case O_AST_CONSTANT_F32:
            if (stack_count == O_AST_TOOL_MAX_STACK_DEPTH) return O_AST_TOOL_ERROR_FILE_FORMAT;
            result = o_ast_file_read_immediates(file, program, maximum_program_bytes, &byte_count, 1u);
            if (result == O_AST_TOOL_SUCCESS) {
                stack[stack_count].tree_depth = 0u;
                stack[stack_count++].direct_leaf = 1;
            }
            break;
        case O_AST_LITERAL_F32:
            if (stack_count == O_AST_TOOL_MAX_STACK_DEPTH) return O_AST_TOOL_ERROR_FILE_FORMAT;
            result = o_ast_file_read_immediates(file, program, maximum_program_bytes, &byte_count, 4u);
            if (result == O_AST_TOOL_SUCCESS) {
                uint32_t bits = o_load_u32(program + byte_count - 4u);
                if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return O_AST_TOOL_ERROR_FILE_FORMAT;
                stack[stack_count].tree_depth = 0u;
                stack[stack_count++].direct_leaf = 1;
            }
            break;
        case O_AST_TOGGLE2_F32:
            if (stack_count < 2u || !stack[stack_count - 2u].direct_leaf || !stack[stack_count - 1u].direct_leaf) {
                return O_AST_TOOL_ERROR_FILE_FORMAT;
            }
            result = o_ast_file_read_immediates(file, program, maximum_program_bytes, &byte_count, 1u);
            if (result == O_AST_TOOL_SUCCESS) {
                --stack_count;
                stack[stack_count - 1u].tree_depth = 0u;
                stack[stack_count - 1u].direct_leaf = 0;
            }
            break;
        case O_AST_TOGGLE4_F32:
            if (stack_count < 4u || !stack[stack_count - 4u].direct_leaf || !stack[stack_count - 3u].direct_leaf ||
                !stack[stack_count - 2u].direct_leaf || !stack[stack_count - 1u].direct_leaf
            ) {
                return O_AST_TOOL_ERROR_FILE_FORMAT;
            }
            result = o_ast_file_read_immediates(file, program, maximum_program_bytes, &byte_count, 2u);
            if (result == O_AST_TOOL_SUCCESS) {
                stack_count -= 3u;
                stack[stack_count - 1u].tree_depth = 0u;
                stack[stack_count - 1u].direct_leaf = 0;
            }
            break;
        case O_AST_RETURN_F32:
            if (stack_count != 1u) return O_AST_TOOL_ERROR_FILE_FORMAT;
            *byte_count_ret = byte_count;
            *tree_depth_ret = stack[0].tree_depth;
            return O_AST_TOOL_SUCCESS;
        default:
            arity = o_ast_file_operator_arity(opcode);
            if (arity == 0u || stack_count < arity) return O_AST_TOOL_ERROR_FILE_FORMAT;
            {
                size_t index;
                uint32_t maximum_depth = 0u;
                for (index = stack_count - arity; index < stack_count; ++index) {
                    if (maximum_depth < stack[index].tree_depth) maximum_depth = stack[index].tree_depth;
                }
                stack_count = stack_count - arity + 1u;
                stack[stack_count - 1u].tree_depth = maximum_depth + 1u;
                stack[stack_count - 1u].direct_leaf = 0;
            }
            break;
        }
    }
    return result;
}

static OAstToolResult o_ast_file_write_item(const OAstSpaceItem *item, void *context) {
    OAstFileWriter *writer = (OAstFileWriter *)context;
    if (item->program.byte_count == 0u || item->program.byte_count > O_AST_TOOL_MAX_PROGRAM_BYTES ||
        item->program.byte_count > UINT32_MAX || writer->report.record_count == UINT64_MAX ||
        UINT64_MAX - writer->report.program_bytes < item->program.byte_count
    ) {
        writer->result = O_AST_TOOL_ERROR_OVERFLOW;
        return writer->result;
    }
    if (!o_write_all(writer->file, item->program.bytes, item->program.byte_count)) {
        writer->result = O_AST_TOOL_ERROR_IO;
        return writer->result;
    }
    ++writer->report.record_count;
    writer->report.program_bytes += item->program.byte_count;
    if (writer->report.maximum_program_bytes < item->program.byte_count) {
        writer->report.maximum_program_bytes = (uint32_t)item->program.byte_count;
    }
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_ast_system_file_write_item(const OAstSystemSpaceItem *item, void *context) {
    OAstSystemFileWriter *writer = (OAstSystemFileWriter *)context;
    size_t index;
    for (index = 0u; index < item->rhs_count; ++index) {
        OAstSpaceItem expression;
        OAstToolResult result;
        expression.rank = item->rhs[index].expression_rank;
        expression.depth = item->rhs[index].depth;
        expression.program = item->rhs[index].program;
        result = o_ast_file_write_item(&expression, &writer->ast);
        if (result != O_AST_TOOL_SUCCESS) return result;
    }
    ++writer->system_count;
    return O_AST_TOOL_SUCCESS;
}

OAstToolResult o_ast_file_write(
    const char *path,
    const OAstSpaceConfig *config,
    const OAstSelection *selection,
    OAstFileReport *report_ret
) {
    OAstFileWriter writer;
    OAstSpaceReport space_report;
    uint8_t header[O_AST_FILE_HEADER_BYTES];
    OAstToolResult result;
    if (report_ret != NULL) memset(report_ret, 0, sizeof(*report_ret));
    if (path == NULL || path[0] == '\0' || config == NULL || selection == NULL) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    memset(&writer, 0, sizeof(writer));
    writer.file = fopen(path, "wb+");
    if (writer.file == NULL) return O_AST_TOOL_ERROR_IO;
    memset(header, 0, sizeof(header));
    writer.result = o_write_all(writer.file, header, sizeof(header)) ? O_AST_TOOL_SUCCESS : O_AST_TOOL_ERROR_IO;
    if (writer.result == O_AST_TOOL_SUCCESS) {
        result = o_ast_space_iterate(
            config,
            selection,
            o_ast_file_write_item,
            &writer,
            &space_report
        );
        if (result != O_AST_TOOL_SUCCESS) writer.result = writer.result == O_AST_TOOL_SUCCESS ? result : writer.result;
    }
    if (writer.result == O_AST_TOOL_SUCCESS && writer.report.record_count != space_report.emitted_count) {
        writer.result = O_AST_TOOL_ERROR_FILE_FORMAT;
    }
    if (writer.result == O_AST_TOOL_SUCCESS) {
        o_encode_header(header, &writer.report);
        if (fseek(writer.file, 0L, SEEK_SET) != 0 || !o_write_all(writer.file, header, sizeof(header)) || fflush(writer.file) != 0) {
            writer.result = O_AST_TOOL_ERROR_IO;
        }
    }
    if (fclose(writer.file) != 0 && writer.result == O_AST_TOOL_SUCCESS) writer.result = O_AST_TOOL_ERROR_IO;
    if (writer.result != O_AST_TOOL_SUCCESS) {
        (void)remove(path);
        return writer.result;
    }
    if (report_ret != NULL) *report_ret = writer.report;
    return O_AST_TOOL_SUCCESS;
}

OAstToolResult o_ast_system_file_write_flat(
    const char *path,
    const OAstSystemSpaceConfig *config,
    const OAstSelection *selection,
    void *workspace,
    size_t workspace_size,
    OAstFileReport *report_ret,
    OAstSystemSpaceReport *system_report_ret
) {
    OAstSystemFileWriter writer;
    OAstSystemSpaceReport system_report;
    uint8_t header[O_AST_FILE_HEADER_BYTES];
    OAstToolResult result;
    if (report_ret != NULL) memset(report_ret, 0, sizeof(*report_ret));
    if (system_report_ret != NULL) memset(system_report_ret, 0, sizeof(*system_report_ret));
    if (path == NULL || path[0] == '\0' || config == NULL || selection == NULL) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    memset(&writer, 0, sizeof(writer));
    writer.ast.file = fopen(path, "wb+");
    if (writer.ast.file == NULL) return O_AST_TOOL_ERROR_IO;
    memset(header, 0, sizeof(header));
    writer.ast.result = o_write_all(writer.ast.file, header, sizeof(header)) ? O_AST_TOOL_SUCCESS : O_AST_TOOL_ERROR_IO;
    if (writer.ast.result == O_AST_TOOL_SUCCESS) {
        result = o_ast_system_space_iterate(
            config,
            selection,
            workspace,
            workspace_size,
            o_ast_system_file_write_item,
            &writer,
            &system_report
        );
        if (result != O_AST_TOOL_SUCCESS) writer.ast.result = writer.ast.result == O_AST_TOOL_SUCCESS ? result : writer.ast.result;
    }
    if (writer.ast.result == O_AST_TOOL_SUCCESS && writer.system_count != system_report.emitted_count) {
        writer.ast.result = O_AST_TOOL_ERROR_FILE_FORMAT;
    }
    if (writer.ast.result == O_AST_TOOL_SUCCESS) {
        o_encode_header(header, &writer.ast.report);
        if (fseek(writer.ast.file, 0L, SEEK_SET) != 0 || !o_write_all(writer.ast.file, header, sizeof(header)) || fflush(writer.ast.file) != 0) {
            writer.ast.result = O_AST_TOOL_ERROR_IO;
        }
    }
    if (fclose(writer.ast.file) != 0 && writer.ast.result == O_AST_TOOL_SUCCESS) writer.ast.result = O_AST_TOOL_ERROR_IO;
    if (writer.ast.result != O_AST_TOOL_SUCCESS) {
        (void)remove(path);
        return writer.ast.result;
    }
    if (report_ret != NULL) *report_ret = writer.ast.report;
    if (system_report_ret != NULL) *system_report_ret = system_report;
    return O_AST_TOOL_SUCCESS;
}

OAstToolResult o_ast_file_read(
    const char *path,
    OAstSpaceVisit visit,
    void *context,
    OAstFileReport *report_ret
) {
    FILE *file;
    OAstFileReport expected;
    OAstFileReport actual;
    uint8_t header[O_AST_FILE_HEADER_BYTES];
    uint8_t program[O_AST_TOOL_MAX_PROGRAM_BYTES];
    uint64_t index;
    OAstToolResult result = O_AST_TOOL_SUCCESS;
    if (report_ret != NULL) memset(report_ret, 0, sizeof(*report_ret));
    if (path == NULL || path[0] == '\0') return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    file = fopen(path, "rb");
    if (file == NULL) return O_AST_TOOL_ERROR_IO;
    memset(&actual, 0, sizeof(actual));
    if (fread(header, 1u, sizeof(header), file) != sizeof(header)) {
        result = ferror(file) ? O_AST_TOOL_ERROR_IO : O_AST_TOOL_ERROR_FILE_FORMAT;
    }
    if (result == O_AST_TOOL_SUCCESS) result = o_decode_header(header, &expected);
    for (index = 0u; result == O_AST_TOOL_SUCCESS && index < expected.record_count; ++index) {
        OAstSpaceItem item;
        result = o_ast_file_read_program(
            file,
            expected.maximum_program_bytes,
            program,
            &item.program.byte_count,
            &item.depth
        );
        if (result != O_AST_TOOL_SUCCESS) break;
        item.rank = index;
        item.program.bytes = program;
        ++actual.record_count;
        actual.program_bytes += item.program.byte_count;
        if (actual.maximum_program_bytes < item.program.byte_count) actual.maximum_program_bytes = (uint32_t)item.program.byte_count;
        if (visit != NULL && visit(&item, context) != O_AST_TOOL_SUCCESS) result = O_AST_TOOL_ERROR_CALLBACK;
    }
    if (result == O_AST_TOOL_SUCCESS) {
        int trailing = fgetc(file);
        if (ferror(file)) {
            result = O_AST_TOOL_ERROR_IO;
        } else if (trailing != EOF || actual.record_count != expected.record_count ||
                   actual.maximum_program_bytes != expected.maximum_program_bytes
        ) {
            result = O_AST_TOOL_ERROR_FILE_FORMAT;
        }
    }
    if (fclose(file) != 0 && result == O_AST_TOOL_SUCCESS) result = O_AST_TOOL_ERROR_IO;
    if (result != O_AST_TOOL_SUCCESS) return result;
    if (report_ret != NULL) *report_ret = actual;
    return O_AST_TOOL_SUCCESS;
}
