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
#define _POSIX_C_SOURCE 200809L

#include "o_ast_tools.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define O_MAX_LITERALS 64u
#define O_MAX_OPERATORS 32u
#define O_MAX_VALUES 255u

typedef struct OPrintContext {
    const float *states;
    size_t state_count;
    const float *constants;
    size_t constant_count;
    uint32_t toggle_bit_count;
    uint32_t permutation;
    int evaluate;
} OPrintContext;

typedef struct OPrintSystemContext {
    OPrintContext expression;
} OPrintSystemContext;

static void o_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s [--states N] [--rhs-count N] [--constants N] [--literal F] [--operators LIST] "
        "[--toggle-bits N] [--include-toggle2] [--include-toggle4] [--depth N] "
        "[--depth-mode up-to|exact] [--mode exhaustive|truncate|random|structural-random] [--count N] "
        "[--seed N] [--state-values CSV] [--constant-values CSV] [--permutation N] [--no-evaluate] "
        "[--binary-output FILE | --binary-input FILE | --verify-binary FILE]\n",
        program
    );
}

static int o_parse_u32(const char *text, uint32_t *value_ret) {
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) return 0;
    *value_ret = (uint32_t)value;
    return 1;
}

static int o_parse_u64(const char *text, uint64_t *value_ret) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *value_ret = (uint64_t)value;
    return 1;
}

static int o_parse_float(const char *text, float *value_ret) {
    char *end = NULL;
    float value;
    errno = 0;
    value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) return 0;
    *value_ret = value;
    return 1;
}

static int o_parse_float_list(const char *text, float *values, size_t expected_count) {
    const char *cursor = text;
    size_t count = 0u;
    while (*cursor != '\0') {
        char *end = NULL;
        float value;
        errno = 0;
        value = strtof(cursor, &end);
        if (errno != 0 || end == cursor || !isfinite(value) || count == expected_count) return 0;
        values[count++] = value;
        if (*end == '\0') {
            cursor = end;
        } else if (*end == ',') {
            cursor = end + 1;
            if (*cursor == '\0') return 0;
        } else {
            return 0;
        }
    }
    return count == expected_count;
}

static int o_operator(const char *name, size_t length, uint8_t *opcode_ret) {
#define O_OPERATOR(text, opcode) \
    if (length == sizeof(text) - 1u && memcmp(name, text, sizeof(text) - 1u) == 0) { \
        *opcode_ret = opcode; \
        return 1; \
    }
    O_OPERATOR("add", O_AST_ADD_F32)
    O_OPERATOR("sub", O_AST_SUB_F32)
    O_OPERATOR("mul", O_AST_MUL_F32)
    O_OPERATOR("div", O_AST_DIV_F32)
    O_OPERATOR("neg", O_AST_NEG_F32)
    O_OPERATOR("sqrt", O_AST_SQRT_F32)
    O_OPERATOR("rcp", O_AST_RCP_F32)
    O_OPERATOR("abs", O_AST_ABS_F32)
    O_OPERATOR("min", O_AST_MIN_F32)
    O_OPERATOR("max", O_AST_MAX_F32)
    O_OPERATOR("fma", O_AST_FMA_F32)
    O_OPERATOR("sin", O_AST_SIN_F32)
    O_OPERATOR("cos", O_AST_COS_F32)
    O_OPERATOR("ex2", O_AST_EX2_F32)
    O_OPERATOR("lg2", O_AST_LG2_F32)
    O_OPERATOR("rsqrt", O_AST_RSQRT_F32)
    O_OPERATOR("tanh", O_AST_TANH_F32)
    O_OPERATOR("exp", O_AST_EXP_F32)
    O_OPERATOR("log", O_AST_LOG_F32)
#undef O_OPERATOR
    return 0;
}

static int o_parse_operators(const char *text, uint8_t operators[O_MAX_OPERATORS], size_t *count_ret) {
    const char *cursor = text;
    size_t count = 0u;
    if (strcmp(text, "none") == 0) {
        *count_ret = 0u;
        return 1;
    }
    while (*cursor != '\0') {
        const char *end = strchr(cursor, ',');
        size_t length = end == NULL ? strlen(cursor) : (size_t)(end - cursor);
        if (count == O_MAX_OPERATORS || !o_operator(cursor, length, &operators[count])) return 0;
        ++count;
        if (end == NULL) break;
        cursor = end + 1;
        if (*cursor == '\0') return 0;
    }
    *count_ret = count;
    return 1;
}

static OAstToolResult o_print_item(const OAstSpaceItem *item, void *context) {
    OPrintContext *print = (OPrintContext *)context;
    float value = 0.0f;
    uint32_t value_bits = 0u;
    size_t index;
    OAstToolResult result = O_AST_TOOL_SUCCESS;
    if (print->evaluate) {
        result = o_ast_cpu_evaluate(
            item->program,
            print->states,
            print->state_count,
            print->constants,
            print->constant_count,
            print->toggle_bit_count,
            print->permutation,
            &value
        );
        if (result != O_AST_TOOL_SUCCESS) return result;
        memcpy(&value_bits, &value, sizeof(value_bits));
    }
    printf("{\"rank\":%" PRIu64 ",\"depth\":%u,\"program\":\"", item->rank, item->depth);
    for (index = 0u; index < item->program.byte_count; ++index) printf("%02x", item->program.bytes[index]);
    if (print->evaluate) {
        if (isfinite(value)) {
            printf("\",\"value_bits\":\"%08" PRIx32 "\",\"value\":%.9g}\n", value_bits, value);
        } else {
            printf("\",\"value_bits\":\"%08" PRIx32 "\",\"value\":null}\n", value_bits);
        }
    } else {
        printf("\"}\n");
    }
    return ferror(stdout) ? O_AST_TOOL_ERROR_CALLBACK : O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_print_system_item(const OAstSystemSpaceItem *item, void *context) {
    OPrintSystemContext *print = (OPrintSystemContext *)context;
    size_t rhs_index;
    if (item->rank_is_space_rank) {
        printf("{\"rank\":%" PRIu64 ",\"rhs\":[", item->rank);
    } else {
        printf("{\"generation_index\":%" PRIu64 ",\"rank\":null,\"rhs\":[", item->rank);
    }
    for (rhs_index = 0u; rhs_index < item->rhs_count; ++rhs_index) {
        const OAstSystemRhsItem *rhs = item->rhs + rhs_index;
        float value = 0.0f;
        uint32_t value_bits = 0u;
        size_t byte_index;
        OAstToolResult result = O_AST_TOOL_SUCCESS;
        if (print->expression.evaluate) {
            result = o_ast_cpu_evaluate(
                rhs->program,
                print->expression.states,
                print->expression.state_count,
                print->expression.constants,
                print->expression.constant_count,
                print->expression.toggle_bit_count,
                print->expression.permutation,
                &value
            );
            if (result != O_AST_TOOL_SUCCESS) return result;
            memcpy(&value_bits, &value, sizeof(value_bits));
        }
        if (rhs_index != 0u) putchar(',');
        printf("{\"state_index\":%u,\"expression_rank\":", rhs->state_index);
        if (rhs->expression_rank == UINT64_MAX) {
            printf("null");
        } else {
            printf("%" PRIu64, rhs->expression_rank);
        }
        printf(",\"depth\":%u,\"program\":\"", rhs->depth);
        for (byte_index = 0u; byte_index < rhs->program.byte_count; ++byte_index) {
            printf("%02x", rhs->program.bytes[byte_index]);
        }
        if (print->expression.evaluate) {
            if (isfinite(value)) {
                printf("\",\"value_bits\":\"%08" PRIx32 "\",\"value\":%.9g}", value_bits, value);
            } else {
                printf("\",\"value_bits\":\"%08" PRIx32 "\",\"value\":null}", value_bits);
            }
        } else {
            printf("\"}");
        }
    }
    printf("]}\n");
    return ferror(stdout) ? O_AST_TOOL_ERROR_CALLBACK : O_AST_TOOL_SUCCESS;
}

static int o_ast_tools_run(int argc, char **argv) {
    static const uint8_t default_operators[] = {
        O_AST_ADD_F32,
        O_AST_SUB_F32,
        O_AST_MUL_F32,
        O_AST_DIV_F32,
        O_AST_NEG_F32
    };
    OAstSpaceConfig config;
    OAstSelection selection;
    OAstSpaceReport report;
    OPrintContext print;
    OPrintSystemContext system_print;
    OAstRhsSpaceConfig rhs_spaces[O_MAX_VALUES];
    OAstSystemSpaceConfig system_config;
    uint32_t literal_bits[O_MAX_LITERALS];
    uint8_t operators[O_MAX_OPERATORS];
    float states[O_MAX_VALUES];
    float constants[O_MAX_VALUES];
    const char *state_values_text = NULL;
    const char *constant_values_text = NULL;
    const char *binary_output_path = NULL;
    const char *binary_input_path = NULL;
    const char *verify_binary_path = NULL;
    size_t operator_count = sizeof(default_operators) / sizeof(default_operators[0]);
    size_t literal_count = 0u;
    uint32_t rhs_count = 1u;
    size_t index;
    int argument = 1;
    OAstToolResult result;
    memset(&config, 0, sizeof(config));
    memset(&selection, 0, sizeof(selection));
    memset(&print, 0, sizeof(print));
    memcpy(operators, default_operators, sizeof(default_operators));
    config.state_count = 2u;
    config.constant_count = 1u;
    config.maximum_depth = 2u;
    config.depth_mode = O_AST_DEPTH_UP_TO;
    selection.mode = O_AST_SELECT_TRUNCATE;
    selection.count = 16u;
    selection.seed = 1u;
    print.evaluate = 1;

    while (argument < argc) {
        const char *option = argv[argument++];
        if (strcmp(option, "--help") == 0) {
            o_usage(argv[0]);
            return 0;
        }
        if (strcmp(option, "--include-toggle2") == 0) {
            config.include_toggle2 = 1;
            continue;
        }
        if (strcmp(option, "--include-toggle4") == 0) {
            config.include_toggle4 = 1;
            continue;
        }
        if (strcmp(option, "--no-evaluate") == 0) {
            print.evaluate = 0;
            continue;
        }
        if (argument == argc) {
            o_usage(argv[0]);
            return 2;
        }
        if (strcmp(option, "--states") == 0) {
            if (!o_parse_u32(argv[argument], &config.state_count)) return 2;
        } else if (strcmp(option, "--rhs-count") == 0) {
            if (!o_parse_u32(argv[argument], &rhs_count) || rhs_count == 0u) return 2;
        } else if (strcmp(option, "--constants") == 0) {
            if (!o_parse_u32(argv[argument], &config.constant_count)) return 2;
        } else if (strcmp(option, "--toggle-bits") == 0) {
            if (!o_parse_u32(argv[argument], &config.toggle_bit_count)) return 2;
        } else if (strcmp(option, "--depth") == 0) {
            if (!o_parse_u32(argv[argument], &config.maximum_depth)) return 2;
        } else if (strcmp(option, "--count") == 0) {
            if (!o_parse_u64(argv[argument], &selection.count)) return 2;
        } else if (strcmp(option, "--seed") == 0) {
            if (!o_parse_u64(argv[argument], &selection.seed)) return 2;
        } else if (strcmp(option, "--permutation") == 0) {
            if (!o_parse_u32(argv[argument], &print.permutation)) return 2;
        } else if (strcmp(option, "--literal") == 0) {
            float value;
            if (literal_count == O_MAX_LITERALS || !o_parse_float(argv[argument], &value)) return 2;
            memcpy(&literal_bits[literal_count++], &value, sizeof(value));
        } else if (strcmp(option, "--operators") == 0) {
            if (!o_parse_operators(argv[argument], operators, &operator_count)) return 2;
        } else if (strcmp(option, "--state-values") == 0) {
            state_values_text = argv[argument];
        } else if (strcmp(option, "--constant-values") == 0) {
            constant_values_text = argv[argument];
        } else if (strcmp(option, "--binary-output") == 0) {
            binary_output_path = argv[argument];
        } else if (strcmp(option, "--binary-input") == 0) {
            binary_input_path = argv[argument];
        } else if (strcmp(option, "--verify-binary") == 0) {
            verify_binary_path = argv[argument];
        } else if (strcmp(option, "--depth-mode") == 0) {
            if (strcmp(argv[argument], "up-to") == 0) {
                config.depth_mode = O_AST_DEPTH_UP_TO;
            } else if (strcmp(argv[argument], "exact") == 0) {
                config.depth_mode = O_AST_DEPTH_EXACT;
            } else {
                return 2;
            }
        } else if (strcmp(option, "--mode") == 0) {
            if (strcmp(argv[argument], "exhaustive") == 0) {
                selection.mode = O_AST_SELECT_EXHAUSTIVE;
            } else if (strcmp(argv[argument], "truncate") == 0) {
                selection.mode = O_AST_SELECT_TRUNCATE;
            } else if (strcmp(argv[argument], "random") == 0) {
                selection.mode = O_AST_SELECT_RANDOM;
            } else if (strcmp(argv[argument], "structural-random") == 0) {
                selection.mode = O_AST_SELECT_STRUCTURAL_RANDOM;
            } else {
                return 2;
            }
        } else {
            o_usage(argv[0]);
            return 2;
        }
        ++argument;
    }
    if (config.state_count > O_MAX_VALUES || config.constant_count > O_MAX_VALUES || rhs_count > config.state_count) return 2;
    for (index = 0u; index < config.state_count; ++index) states[index] = (float)(index + 1u);
    for (index = 0u; index < config.constant_count; ++index) constants[index] = 10.0f * (float)(index + 1u);
    if (state_values_text != NULL && !o_parse_float_list(state_values_text, states, config.state_count)) return 2;
    if (
        constant_values_text != NULL &&
        !o_parse_float_list(
            constant_values_text,
            constants,
            config.constant_count
        )
    ) {
        return 2;
    }
    config.literal_bits = literal_bits;
    config.literal_count = literal_count;
    config.operators = operators;
    config.operator_count = operator_count;
    print.states = states;
    print.state_count = config.state_count;
    print.constants = constants;
    print.constant_count = config.constant_count;
    print.toggle_bit_count = config.toggle_bit_count;
    memset(&system_config, 0, sizeof(system_config));
    for (index = 0u; index < rhs_count; ++index) {
        rhs_spaces[index].state_index = (uint8_t)index;
        rhs_spaces[index].expression = config;
    }
    system_config.rhs = rhs_spaces;
    system_config.rhs_count = rhs_count;
    memset(&system_print, 0, sizeof(system_print));
    system_print.expression = print;
    if ((binary_output_path != NULL) + (binary_input_path != NULL) + (verify_binary_path != NULL) > 1) {
        fprintf(stderr, "choose only one binary file mode\n");
        return 2;
    }
    if (binary_output_path != NULL) {
        OAstFileReport file_report;
        OAstSystemSpaceReport system_report;
        void *system_workspace = NULL;
        size_t system_workspace_size = 0u;
        size_t system_workspace_alignment = 0u;
        if (rhs_count == 1u) {
            result = o_ast_file_write(binary_output_path, &config, &selection, &file_report);
        } else {
            result = o_ast_system_space_workspace_size(
                &system_config,
                &system_workspace_size,
                &system_workspace_alignment
            );
            if (result == O_AST_TOOL_SUCCESS) {
                system_workspace = malloc(system_workspace_size);
                if (system_workspace == NULL || (uintptr_t)system_workspace % system_workspace_alignment != 0u) {
                    result = O_AST_TOOL_ERROR_CAPACITY;
                }
            }
            if (result == O_AST_TOOL_SUCCESS) {
                result = o_ast_system_file_write_flat(
                    binary_output_path,
                    &system_config,
                    &selection,
                    system_workspace,
                    system_workspace_size,
                    &file_report,
                    &system_report
                );
            }
            free(system_workspace);
        }
        if (result != O_AST_TOOL_SUCCESS) {
            fprintf(stderr, "AST file write failed: %s\n", o_ast_tool_result_string(result));
            return 1;
        }
        if (rhs_count == 1u) {
            fprintf(stderr, "records=%" PRIu64 " program_bytes=%" PRIu64 " max_program=%u\n",
                    file_report.record_count, file_report.program_bytes, file_report.maximum_program_bytes);
        } else {
            fprintf(stderr, "systems=%" PRIu64 " rhs_per_system=%u records=%" PRIu64
                    " program_bytes=%" PRIu64 " max_program=%u ranked_selection=%s\n",
                    system_report.emitted_count, rhs_count, file_report.record_count, file_report.program_bytes,
                    file_report.maximum_program_bytes, system_report.selection_uses_space_ranks ? "true" : "false");
        }
        return 0;
    }
    if (binary_input_path != NULL || verify_binary_path != NULL) {
        OAstFileReport file_report;
        result = o_ast_file_read(
            binary_input_path != NULL ? binary_input_path : verify_binary_path,
            binary_input_path != NULL ? o_print_item : NULL,
            &print,
            &file_report
        );
        if (result != O_AST_TOOL_SUCCESS) {
            fprintf(stderr, "AST file read failed: %s\n", o_ast_tool_result_string(result));
            return 1;
        }
        fprintf(
            stderr,
            "verified_records=%" PRIu64 " program_bytes=%" PRIu64 " max_program=%u\n",
            file_report.record_count,
            file_report.program_bytes,
            file_report.maximum_program_bytes
        );
        return 0;
    }
    if (rhs_count == 1u) {
        result = o_ast_space_iterate(&config, &selection, o_print_item, &print, &report);
    } else {
        OAstSystemSpaceReport system_report;
        void *system_workspace;
        size_t system_workspace_size;
        size_t system_workspace_alignment;
        result = o_ast_system_space_workspace_size(
            &system_config,
            &system_workspace_size,
            &system_workspace_alignment
        );
        if (result != O_AST_TOOL_SUCCESS) {
            fprintf(stderr, "system workspace query failed: %s\n", o_ast_tool_result_string(result));
            return 1;
        }
        system_workspace = malloc(system_workspace_size);
        if (system_workspace == NULL || (uintptr_t)system_workspace % system_workspace_alignment != 0u) {
            free(system_workspace);
            fprintf(stderr, "system workspace allocation failed\n");
            return 1;
        }
        result = o_ast_system_space_iterate(
            &system_config,
            &selection,
            system_workspace,
            system_workspace_size,
            o_print_system_item,
            &system_print,
            &system_report
        );
        free(system_workspace);
        if (result != O_AST_TOOL_SUCCESS) {
            fprintf(stderr, "system AST helper failed: %s\n", o_ast_tool_result_string(result));
            return 1;
        }
        fprintf(stderr, "space=%" PRIu64 "%s selected=%" PRIu64 " emitted=%" PRIu64
                " rhs_per_system=%u ranked_selection=%s\n",
                system_report.space_count, system_report.space_count_is_exact ? "" : "+", system_report.selected_count,
                system_report.emitted_count, rhs_count, system_report.selection_uses_space_ranks ? "true" : "false");
        return 0;
    }
    if (result != O_AST_TOOL_SUCCESS) {
        fprintf(stderr, "AST helper failed: %s\n", o_ast_tool_result_string(result));
        return 1;
    }
    fprintf(
        stderr,
        "space=%" PRIu64 "%s selected=%" PRIu64 " emitted=%" PRIu64 "\n",
        report.space_count,
        report.space_count_is_exact ? "" : "+",
        report.selected_count,
        report.emitted_count
    );
    return 0;
}

int main(int argc, char **argv) {
    return o_ast_tools_run(argc, argv);
}
