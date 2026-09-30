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

#include <limits.h>
#include <string.h>

typedef struct OAstCounts {
    uint64_t direct_terminal_count;
    uint64_t terminal_count;
    uint64_t up_to[O_AST_TOOL_MAX_DEPTH + 1u];
    uint64_t exact[O_AST_TOOL_MAX_DEPTH + 1u];
    int count_is_exact;
} OAstCounts;

typedef struct OAstWriter {
    uint8_t *bytes;
    size_t capacity;
    size_t count;
    size_t instruction_count;
} OAstWriter;

static size_t o_operator_arity(uint8_t opcode) {
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

static uint64_t o_saturating_add(uint64_t lhs, uint64_t rhs, int *exact) {
    if (UINT64_MAX - lhs < rhs) {
        *exact = 0;
        return UINT64_MAX;
    }
    return lhs + rhs;
}

static uint64_t o_saturating_multiply(uint64_t lhs, uint64_t rhs, int *exact) {
    if (rhs != 0u && lhs > UINT64_MAX / rhs) {
        *exact = 0;
        return UINT64_MAX;
    }
    return lhs * rhs;
}

static uint64_t o_saturating_power(uint64_t base, size_t exponent, int *exact) {
    uint64_t value = 1u;
    size_t index;
    for (index = 0u; index < exponent; ++index) {
        value = o_saturating_multiply(value, base, exact);
    }
    return value;
}

static uint64_t o_power_difference(uint64_t base, uint64_t excluded, size_t exponent, int *exact) {
    int base_exact = 1;
    int excluded_exact = 1;
    uint64_t base_power = o_saturating_power(base, exponent, &base_exact);
    uint64_t excluded_power = o_saturating_power(excluded, exponent, &excluded_exact);
    if (!base_exact) {
        *exact = 0;
        return UINT64_MAX;
    }
    if (!excluded_exact || excluded_power > base_power) {
        *exact = 0;
        return UINT64_MAX;
    }
    return base_power - excluded_power;
}

static OAstToolResult o_validate_config(const OAstSpaceConfig *config) {
    size_t index;
    size_t other;
    size_t maximum_arity = 1u;
    size_t terminal_instructions = 1u;
    size_t maximum_instructions;
    if (config == NULL || config->state_count == 0u || config->state_count > 255u || config->constant_count > 255u ||
        config->toggle_bit_count > 30u || config->maximum_depth > O_AST_TOOL_MAX_DEPTH ||
        (config->depth_mode != O_AST_DEPTH_UP_TO && config->depth_mode != O_AST_DEPTH_EXACT) ||
        (config->literal_count != 0u && config->literal_bits == NULL) || (config->operator_count != 0u && config->operators == NULL) ||
        (config->include_toggle2 != 0 && config->include_toggle2 != 1) || (config->include_toggle4 != 0 && config->include_toggle4 != 1)
    ) {
        return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    if ((config->include_toggle2 && config->toggle_bit_count == 0u) || (config->include_toggle4 && config->toggle_bit_count < 2u)) {
        return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    for (index = 0u; index < config->literal_count; ++index) {
        if ((config->literal_bits[index] & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    for (index = 0u; index < config->operator_count; ++index) {
        size_t arity = o_operator_arity(config->operators[index]);
        if (arity == 0u) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
        if (maximum_arity < arity) maximum_arity = arity;
        for (other = 0u; other < index; ++other) {
            if (config->operators[other] == config->operators[index]) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
        }
    }
    if (config->include_toggle4) {
        terminal_instructions = 5u;
    } else if (config->include_toggle2) {
        terminal_instructions = 3u;
    }
    maximum_instructions = terminal_instructions;
    for (index = 0u; index < config->maximum_depth; ++index) {
        if (maximum_instructions > (O_AST_TOOL_MAX_INSTRUCTIONS - 1u) / maximum_arity) return O_AST_TOOL_ERROR_CAPACITY;
        maximum_instructions = 1u + maximum_arity * maximum_instructions;
    }
    if (maximum_instructions + 1u > O_AST_TOOL_MAX_INSTRUCTIONS) return O_AST_TOOL_ERROR_CAPACITY;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_build_counts(const OAstSpaceConfig *config, OAstCounts *counts) {
    uint32_t depth;
    size_t index;
    int exact = 1;
    OAstToolResult result = o_validate_config(config);
    if (result != O_AST_TOOL_SUCCESS || counts == NULL) return result == O_AST_TOOL_SUCCESS ? O_AST_TOOL_ERROR_INVALID_ARGUMENT : result;
    memset(counts, 0, sizeof(*counts));
    counts->direct_terminal_count = (uint64_t)config->state_count + config->constant_count;
    counts->direct_terminal_count = o_saturating_add(
        counts->direct_terminal_count,
        (uint64_t)config->literal_count,
        &exact
    );
    counts->terminal_count = counts->direct_terminal_count;
    if (config->include_toggle2) {
        uint64_t choices = o_saturating_power(counts->direct_terminal_count, 2u, &exact);
        counts->terminal_count = o_saturating_add(
            counts->terminal_count,
            o_saturating_multiply(config->toggle_bit_count, choices, &exact),
            &exact
        );
    }
    if (config->include_toggle4) {
        uint64_t choices = o_saturating_power(counts->direct_terminal_count, 4u, &exact);
        uint64_t bit_pairs = o_saturating_multiply(config->toggle_bit_count, config->toggle_bit_count - 1u, &exact);
        counts->terminal_count = o_saturating_add(
            counts->terminal_count,
            o_saturating_multiply(bit_pairs, choices, &exact),
            &exact
        );
    }
    counts->up_to[0] = counts->terminal_count;
    counts->exact[0] = counts->terminal_count;
    for (depth = 1u; depth <= config->maximum_depth; ++depth) {
        uint64_t total = counts->terminal_count;
        uint64_t exact_total = 0u;
        uint64_t excluded = depth == 1u ? 0u : counts->up_to[depth - 2u];
        for (index = 0u; index < config->operator_count; ++index) {
            size_t arity = o_operator_arity(config->operators[index]);
            total = o_saturating_add(total, o_saturating_power(counts->up_to[depth - 1u], arity, &exact), &exact);
            exact_total = o_saturating_add(
                exact_total,
                o_power_difference(counts->up_to[depth - 1u], excluded, arity, &exact),
                &exact
            );
        }
        counts->up_to[depth] = total;
        counts->exact[depth] = exact_total;
    }
    counts->count_is_exact = exact;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_write_byte(OAstWriter *writer, uint8_t value) {
    if (writer->count == writer->capacity) return O_AST_TOOL_ERROR_CAPACITY;
    writer->bytes[writer->count++] = value;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_write_instruction(OAstWriter *writer, uint8_t opcode) {
    if (writer->instruction_count == O_AST_TOOL_MAX_INSTRUCTIONS) return O_AST_TOOL_ERROR_CAPACITY;
    ++writer->instruction_count;
    return o_write_byte(writer, opcode);
}

static OAstToolResult o_emit_direct_terminal(
    const OAstSpaceConfig *config,
    uint64_t rank,
    OAstWriter *writer
) {
    OAstToolResult result;
    if (rank < config->state_count) {
        result = o_write_instruction(writer, O_AST_STATE_F32);
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)rank);
    }
    rank -= config->state_count;
    if (rank < config->constant_count) {
        result = o_write_instruction(writer, O_AST_CONSTANT_F32);
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)rank);
    }
    rank -= config->constant_count;
    if (rank < config->literal_count) {
        uint32_t bits = config->literal_bits[rank];
        result = o_write_instruction(writer, O_AST_LITERAL_F32);
        if (result != O_AST_TOOL_SUCCESS) return result;
        result = o_write_byte(writer, (uint8_t)bits);
        if (result != O_AST_TOOL_SUCCESS) return result;
        result = o_write_byte(writer, (uint8_t)(bits >> 8u));
        if (result != O_AST_TOOL_SUCCESS) return result;
        result = o_write_byte(writer, (uint8_t)(bits >> 16u));
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)(bits >> 24u));
    }
    return O_AST_TOOL_ERROR_PROGRAM;
}

static void o_decode_tuple(uint64_t rank, uint64_t base, size_t arity, uint64_t tuple[3]) {
    size_t index = arity;
    while (index != 0u) {
        --index;
        tuple[index] = rank % base;
        rank /= base;
    }
}

static OAstToolResult o_emit_terminal(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint64_t rank,
    OAstWriter *writer
) {
    uint64_t tuple[4];
    uint64_t toggle2_choices;
    uint64_t toggle4_choices;
    OAstToolResult result;
    size_t index;
    int exact = 1;
    if (rank < counts->direct_terminal_count) return o_emit_direct_terminal(config, rank, writer);
    rank -= counts->direct_terminal_count;
    toggle2_choices = config->include_toggle2 ? o_saturating_power(counts->direct_terminal_count, 2u, &exact) : 0u;
    if (config->include_toggle2 && rank < toggle2_choices * config->toggle_bit_count) {
        uint32_t bit = (uint32_t)(rank / toggle2_choices);
        o_decode_tuple(rank % toggle2_choices, counts->direct_terminal_count, 2u, tuple);
        for (index = 0u; index < 2u; ++index) {
            result = o_emit_direct_terminal(config, tuple[index], writer);
            if (result != O_AST_TOOL_SUCCESS) return result;
        }
        result = o_write_instruction(writer, O_AST_TOGGLE2_F32);
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)bit);
    }
    if (config->include_toggle2) rank -= toggle2_choices * config->toggle_bit_count;
    toggle4_choices = config->include_toggle4 ? o_saturating_power(counts->direct_terminal_count, 4u, &exact) : 0u;
    if (config->include_toggle4) {
        uint64_t pair = rank / toggle4_choices;
        uint32_t bit0 = (uint32_t)(pair / (config->toggle_bit_count - 1u));
        uint32_t reduced_bit1 = (uint32_t)(pair % (config->toggle_bit_count - 1u));
        uint32_t bit1 = reduced_bit1 >= bit0 ? reduced_bit1 + 1u : reduced_bit1;
        o_decode_tuple(rank % toggle4_choices, counts->direct_terminal_count, 4u, tuple);
        for (index = 0u; index < 4u; ++index) {
            result = o_emit_direct_terminal(config, tuple[index], writer);
            if (result != O_AST_TOOL_SUCCESS) return result;
        }
        result = o_write_instruction(writer, O_AST_TOGGLE4_F32);
        if (result != O_AST_TOOL_SUCCESS) return result;
        result = o_write_byte(writer, (uint8_t)bit0);
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)bit1);
    }
    return O_AST_TOOL_ERROR_PROGRAM;
}

static OAstToolResult o_emit_up_to(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint32_t depth,
    uint64_t rank,
    OAstWriter *writer,
    uint32_t *actual_depth_ret
);

static OAstToolResult o_emit_operator_children(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint32_t child_depth,
    uint64_t tuple_rank,
    size_t arity,
    OAstWriter *writer,
    uint32_t *maximum_depth_ret
) {
    uint64_t tuple[3];
    size_t index;
    uint32_t maximum_depth = 0u;
    o_decode_tuple(tuple_rank, counts->up_to[child_depth], arity, tuple);
    for (index = 0u; index < arity; ++index) {
        uint32_t actual_depth;
        OAstToolResult result = o_emit_up_to(config, counts, child_depth, tuple[index], writer, &actual_depth);
        if (result != O_AST_TOOL_SUCCESS) return result;
        if (maximum_depth < actual_depth) maximum_depth = actual_depth;
    }
    *maximum_depth_ret = maximum_depth;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_emit_up_to(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint32_t depth,
    uint64_t rank,
    OAstWriter *writer,
    uint32_t *actual_depth_ret
) {
    size_t index;
    int exact = 1;
    if (rank >= counts->up_to[depth]) return O_AST_TOOL_ERROR_PROGRAM;
    if (rank < counts->terminal_count) {
        *actual_depth_ret = 0u;
        return o_emit_terminal(config, counts, rank, writer);
    }
    if (depth == 0u) return O_AST_TOOL_ERROR_PROGRAM;
    rank -= counts->terminal_count;
    for (index = 0u; index < config->operator_count; ++index) {
        size_t arity = o_operator_arity(config->operators[index]);
        uint64_t group = o_saturating_power(counts->up_to[depth - 1u], arity, &exact);
        if (rank < group) {
            uint32_t maximum_child_depth;
            OAstToolResult result = o_emit_operator_children(
                config,
                counts,
                depth - 1u,
                rank,
                arity,
                writer,
                &maximum_child_depth
            );
            if (result != O_AST_TOOL_SUCCESS) return result;
            result = o_write_instruction(writer, config->operators[index]);
            if (result != O_AST_TOOL_SUCCESS) return result;
            *actual_depth_ret = maximum_child_depth + 1u;
            return O_AST_TOOL_SUCCESS;
        }
        rank -= group;
    }
    return O_AST_TOOL_ERROR_PROGRAM;
}

static OAstToolResult o_unrank_exact_tuple(
    uint64_t rank,
    uint64_t base,
    uint64_t excluded,
    size_t arity,
    uint64_t tuple[3]
) {
    size_t position;
    int already_outside = 0;
    for (position = 0u; position < arity; ++position) {
        size_t remaining = arity - position - 1u;
        int exact = 1;
        uint64_t all_completions = o_saturating_power(base, remaining, &exact);
        if (!exact || all_completions == 0u) return O_AST_TOOL_ERROR_OVERFLOW;
        if (already_outside) {
            tuple[position] = rank / all_completions;
            rank %= all_completions;
        } else {
            uint64_t inside_completions = o_power_difference(base, excluded, remaining, &exact);
            uint64_t inside_total = o_saturating_multiply(excluded, inside_completions, &exact);
            if (!exact) return O_AST_TOOL_ERROR_OVERFLOW;
            if (rank < inside_total) {
                if (inside_completions == 0u) return O_AST_TOOL_ERROR_PROGRAM;
                tuple[position] = rank / inside_completions;
                rank %= inside_completions;
            } else {
                rank -= inside_total;
                tuple[position] = excluded + rank / all_completions;
                rank %= all_completions;
                already_outside = 1;
            }
        }
        if (tuple[position] >= base) return O_AST_TOOL_ERROR_PROGRAM;
    }
    return already_outside ? O_AST_TOOL_SUCCESS : O_AST_TOOL_ERROR_PROGRAM;
}

static OAstToolResult o_emit_exact(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint32_t depth,
    uint64_t rank,
    OAstWriter *writer,
    uint32_t *actual_depth_ret
) {
    size_t index;
    uint64_t excluded;
    if (rank >= counts->exact[depth]) return O_AST_TOOL_ERROR_PROGRAM;
    if (depth == 0u) {
        *actual_depth_ret = 0u;
        return o_emit_terminal(config, counts, rank, writer);
    }
    excluded = depth == 1u ? 0u : counts->up_to[depth - 2u];
    for (index = 0u; index < config->operator_count; ++index) {
        uint64_t tuple[3];
        size_t arity = o_operator_arity(config->operators[index]);
        int exact = 1;
        uint64_t group = o_power_difference(counts->up_to[depth - 1u], excluded, arity, &exact);
        if (rank < group) {
            size_t child;
            OAstToolResult result = o_unrank_exact_tuple(rank, counts->up_to[depth - 1u], excluded, arity, tuple);
            if (result != O_AST_TOOL_SUCCESS) return result;
            for (child = 0u; child < arity; ++child) {
                uint32_t child_actual_depth;
                result = o_emit_up_to(config, counts, depth - 1u, tuple[child], writer, &child_actual_depth);
                if (result != O_AST_TOOL_SUCCESS) return result;
            }
            result = o_write_instruction(writer, config->operators[index]);
            if (result != O_AST_TOOL_SUCCESS) return result;
            *actual_depth_ret = depth;
            return O_AST_TOOL_SUCCESS;
        }
        rank -= group;
    }
    return O_AST_TOOL_ERROR_PROGRAM;
}

static uint64_t o_splitmix64(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static uint64_t o_random_bounded(uint64_t *state, uint64_t bound) {
    uint64_t threshold = (UINT64_C(0) - bound) % bound;
    uint64_t value;
    do {
        value = o_splitmix64(state);
    } while (value < threshold);
    return value % bound;
}

static OAstToolResult o_emit_ranked(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint64_t rank,
    uint8_t *program,
    size_t program_capacity,
    OAstSpaceItem *item_ret
) {
    OAstWriter writer;
    OAstToolResult result;
    memset(&writer, 0, sizeof(writer));
    writer.bytes = program;
    writer.capacity = program_capacity;
    if (config->depth_mode == O_AST_DEPTH_EXACT) {
        result = o_emit_exact(config, counts, config->maximum_depth, rank, &writer, &item_ret->depth);
    } else {
        result = o_emit_up_to(config, counts, config->maximum_depth, rank, &writer, &item_ret->depth);
    }
    if (result != O_AST_TOOL_SUCCESS) return result;
    result = o_write_instruction(&writer, O_AST_RETURN_F32);
    if (result != O_AST_TOOL_SUCCESS) return result;
    item_ret->rank = rank;
    item_ret->program.bytes = writer.bytes;
    item_ret->program.byte_count = writer.count;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_emit_random_terminal(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint64_t *random_state,
    OAstWriter *writer
) {
    uint64_t family_count = 1u + (uint64_t)config->include_toggle2 + (uint64_t)config->include_toggle4;
    uint64_t family = o_random_bounded(random_state, family_count);
    size_t index;
    OAstToolResult result;
    if (family == 0u) {
        return o_emit_direct_terminal(
            config,
            o_random_bounded(random_state, counts->direct_terminal_count),
            writer
        );
    }
    if (config->include_toggle2 && family == 1u) {
        for (index = 0u; index < 2u; ++index) {
            result = o_emit_direct_terminal(
                config,
                o_random_bounded(random_state, counts->direct_terminal_count),
                writer
            );
            if (result != O_AST_TOOL_SUCCESS) return result;
        }
        result = o_write_instruction(writer, O_AST_TOGGLE2_F32);
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)o_random_bounded(random_state, config->toggle_bit_count));
    }
    for (index = 0u; index < 4u; ++index) {
        result = o_emit_direct_terminal(
            config,
            o_random_bounded(random_state, counts->direct_terminal_count),
            writer
        );
        if (result != O_AST_TOOL_SUCCESS) return result;
    }
    result = o_write_instruction(writer, O_AST_TOGGLE4_F32);
    if (result != O_AST_TOOL_SUCCESS) return result;
    {
        uint32_t bit0 = (uint32_t)o_random_bounded(random_state, config->toggle_bit_count);
        uint32_t bit1 = (uint32_t)o_random_bounded(random_state, config->toggle_bit_count - 1u);
        if (bit1 >= bit0) ++bit1;
        result = o_write_byte(writer, (uint8_t)bit0);
        if (result != O_AST_TOOL_SUCCESS) return result;
        return o_write_byte(writer, (uint8_t)bit1);
    }
}

static OAstToolResult o_emit_random_exact(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint32_t depth,
    uint64_t *random_state,
    OAstWriter *writer
) {
    size_t arity;
    size_t deep_child;
    size_t child;
    uint8_t opcode;
    OAstToolResult result;
    if (depth == 0u) return o_emit_random_terminal(config, counts, random_state, writer);
    if (config->operator_count == 0u) return O_AST_TOOL_ERROR_PROGRAM;
    opcode = config->operators[o_random_bounded(random_state, config->operator_count)];
    arity = o_operator_arity(opcode);
    deep_child = (size_t)o_random_bounded(random_state, arity);
    for (child = 0u; child < arity; ++child) {
        uint32_t child_depth = child == deep_child
            ? depth - 1u
            : (uint32_t)o_random_bounded(random_state, depth);
        result = o_emit_random_exact(config, counts, child_depth, random_state, writer);
        if (result != O_AST_TOOL_SUCCESS) return result;
    }
    return o_write_instruction(writer, opcode);
}

static OAstToolResult o_emit_structural_random(
    const OAstSpaceConfig *config,
    const OAstCounts *counts,
    uint64_t *random_state,
    uint8_t *program,
    size_t program_capacity,
    OAstSpaceItem *item_ret
) {
    OAstWriter writer;
    OAstToolResult result;
    uint32_t depth = config->maximum_depth;
    if (config->depth_mode == O_AST_DEPTH_UP_TO) {
        depth = config->operator_count == 0u
            ? 0u
            : (uint32_t)o_random_bounded(random_state, (uint64_t)config->maximum_depth + 1u);
    }
    memset(&writer, 0, sizeof(writer));
    writer.bytes = program;
    writer.capacity = program_capacity;
    result = o_emit_random_exact(config, counts, depth, random_state, &writer);
    if (result != O_AST_TOOL_SUCCESS) return result;
    result = o_write_instruction(&writer, O_AST_RETURN_F32);
    if (result != O_AST_TOOL_SUCCESS) return result;
    item_ret->rank = UINT64_MAX;
    item_ret->depth = depth;
    item_ret->program.bytes = program;
    item_ret->program.byte_count = writer.count;
    return O_AST_TOOL_SUCCESS;
}

static uint64_t o_gcd(uint64_t lhs, uint64_t rhs) {
    while (rhs != 0u) {
        uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

static uint64_t o_add_mod(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    return lhs >= modulus - rhs ? lhs - (modulus - rhs) : lhs + rhs;
}

static uint64_t o_multiply_mod(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    uint64_t value = 0u;
    lhs %= modulus;
    while (rhs != 0u) {
        if (rhs & 1u) value = o_add_mod(value, lhs, modulus);
        rhs >>= 1u;
        if (rhs != 0u) lhs = o_add_mod(lhs, lhs, modulus);
    }
    return value;
}

OAstToolResult o_ast_space_count(
    const OAstSpaceConfig *config,
    uint64_t *count_ret,
    int *count_is_exact_ret
) {
    OAstCounts counts;
    OAstToolResult result;
    if (count_ret == NULL || count_is_exact_ret == NULL) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    *count_ret = 0u;
    *count_is_exact_ret = 0;
    result = o_build_counts(config, &counts);
    if (result != O_AST_TOOL_SUCCESS) return result;
    *count_ret = config->depth_mode == O_AST_DEPTH_EXACT ? counts.exact[config->maximum_depth] : counts.up_to[config->maximum_depth];
    *count_is_exact_ret = counts.count_is_exact;
    return O_AST_TOOL_SUCCESS;
}

OAstToolResult o_ast_space_iterate(
    const OAstSpaceConfig *config,
    const OAstSelection *selection,
    OAstSpaceVisit visit,
    void *context,
    OAstSpaceReport *report_ret
) {
    OAstCounts counts;
    OAstSpaceReport report;
    uint8_t program[O_AST_TOOL_MAX_PROGRAM_BYTES];
    uint64_t space_count;
    uint64_t selected_count;
    uint64_t multiplier = 1u;
    uint64_t offset = 0u;
    uint64_t index;
    OAstToolResult result;
    if (report_ret != NULL) memset(report_ret, 0, sizeof(*report_ret));
    if (selection == NULL || visit == NULL ||
        (selection->mode != O_AST_SELECT_EXHAUSTIVE && selection->mode != O_AST_SELECT_TRUNCATE &&
         selection->mode != O_AST_SELECT_RANDOM && selection->mode != O_AST_SELECT_STRUCTURAL_RANDOM)
    ) {
        return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    result = o_build_counts(config, &counts);
    if (result != O_AST_TOOL_SUCCESS) return result;
    space_count = config->depth_mode == O_AST_DEPTH_EXACT ? counts.exact[config->maximum_depth] : counts.up_to[config->maximum_depth];
    memset(&report, 0, sizeof(report));
    report.space_count = space_count;
    report.space_count_is_exact = counts.count_is_exact;
    if (!counts.count_is_exact && config->depth_mode == O_AST_DEPTH_EXACT &&
        selection->mode != O_AST_SELECT_STRUCTURAL_RANDOM
    ) {
        return O_AST_TOOL_ERROR_OVERFLOW;
    }
    if (selection->mode == O_AST_SELECT_STRUCTURAL_RANDOM) {
        selected_count = selection->count;
    } else if (selection->mode == O_AST_SELECT_EXHAUSTIVE) {
        if (!counts.count_is_exact) return O_AST_TOOL_ERROR_OVERFLOW;
        selected_count = space_count;
    } else {
        selected_count = selection->count < space_count ? selection->count : space_count;
    }
    if (selection->mode == O_AST_SELECT_RANDOM) {
        uint64_t random_state = selection->seed;
        if (!counts.count_is_exact) return O_AST_TOOL_ERROR_OVERFLOW;
        if (space_count > 1u) {
            multiplier = o_splitmix64(&random_state) % space_count;
            if (multiplier == 0u) multiplier = 1u;
            while (o_gcd(multiplier, space_count) != 1u) {
                ++multiplier;
                if (multiplier == space_count) multiplier = 1u;
            }
            offset = o_splitmix64(&random_state) % space_count;
        }
    }
    report.selected_count = selected_count;
    {
        uint64_t random_state = selection->seed;
    for (index = 0u; index < selected_count; ++index) {
        OAstSpaceItem item;
        uint64_t rank = selection->mode == O_AST_SELECT_RANDOM
            ? o_add_mod(o_multiply_mod(multiplier, index, space_count), offset, space_count)
            : index;
        if (selection->mode == O_AST_SELECT_STRUCTURAL_RANDOM) {
            result = o_emit_structural_random(config, &counts, &random_state, program, sizeof(program), &item);
        } else {
            result = o_emit_ranked(config, &counts, rank, program, sizeof(program), &item);
        }
        if (result != O_AST_TOOL_SUCCESS) return result;
        result = visit(&item, context);
        if (result != O_AST_TOOL_SUCCESS) return O_AST_TOOL_ERROR_CALLBACK;
        ++report.emitted_count;
    }
    }
    if (report_ret != NULL) *report_ret = report;
    return O_AST_TOOL_SUCCESS;
}

static OAstToolResult o_validate_system_config(const OAstSystemSpaceConfig *config) {
    size_t index;
    size_t other;
    if (config == NULL || config->rhs == NULL || config->rhs_count == 0u || config->rhs_count > 255u) {
        return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    for (index = 0u; index < config->rhs_count; ++index) {
        const OAstSpaceConfig *expression = &config->rhs[index].expression;
        OAstToolResult result = o_validate_config(expression);
        if (result != O_AST_TOOL_SUCCESS) return result;
        if (config->rhs[index].state_index >= expression->state_count) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
        if (index != 0u &&
            (expression->state_count != config->rhs[0].expression.state_count ||
             expression->constant_count != config->rhs[0].expression.constant_count ||
             expression->toggle_bit_count != config->rhs[0].expression.toggle_bit_count)
        ) {
            return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
        }
        for (other = 0u; other < index; ++other) {
            if (config->rhs[other].state_index == config->rhs[index].state_index) {
                return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
            }
        }
    }
    return O_AST_TOOL_SUCCESS;
}

OAstToolResult o_ast_system_space_count(
    const OAstSystemSpaceConfig *config,
    uint64_t *count_ret,
    int *count_is_exact_ret
) {
    uint64_t count = 1u;
    int exact = 1;
    size_t index;
    OAstToolResult result;
    if (count_ret == NULL || count_is_exact_ret == NULL) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    *count_ret = 0u;
    *count_is_exact_ret = 0;
    result = o_validate_system_config(config);
    if (result != O_AST_TOOL_SUCCESS) return result;
    for (index = 0u; index < config->rhs_count; ++index) {
        uint64_t rhs_count;
        int rhs_exact;
        result = o_ast_space_count(&config->rhs[index].expression, &rhs_count, &rhs_exact);
        if (result != O_AST_TOOL_SUCCESS) return result;
        if (!rhs_exact) exact = 0;
        count = o_saturating_multiply(count, rhs_count, &exact);
    }
    *count_ret = count;
    *count_is_exact_ret = exact;
    return O_AST_TOOL_SUCCESS;
}

static int o_align_size(size_t value, size_t alignment, size_t *aligned_ret) {
    size_t remainder = value % alignment;
    size_t padding = remainder == 0u ? 0u : alignment - remainder;
    if (value > SIZE_MAX - padding) return 0;
    *aligned_ret = value + padding;
    return 1;
}

OAstToolResult o_ast_system_space_workspace_size(
    const OAstSystemSpaceConfig *config,
    size_t *workspace_size_ret,
    size_t *workspace_alignment_ret
) {
    const size_t alignment = sizeof(uint64_t);
    size_t counts_bytes;
    size_t items_bytes;
    size_t programs_bytes;
    size_t size;
    OAstToolResult result;
    if (workspace_size_ret == NULL || workspace_alignment_ret == NULL) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    *workspace_size_ret = 0u;
    *workspace_alignment_ret = 0u;
    result = o_validate_system_config(config);
    if (result != O_AST_TOOL_SUCCESS) return result;
    if (config->rhs_count > SIZE_MAX / sizeof(OAstCounts) ||
        config->rhs_count > SIZE_MAX / sizeof(OAstSystemRhsItem) ||
        config->rhs_count > SIZE_MAX / O_AST_TOOL_MAX_PROGRAM_BYTES
    ) {
        return O_AST_TOOL_ERROR_OVERFLOW;
    }
    counts_bytes = config->rhs_count * sizeof(OAstCounts);
    items_bytes = config->rhs_count * sizeof(OAstSystemRhsItem);
    programs_bytes = config->rhs_count * O_AST_TOOL_MAX_PROGRAM_BYTES;
    if (!o_align_size(counts_bytes, alignment, &size) || size > SIZE_MAX - items_bytes) return O_AST_TOOL_ERROR_OVERFLOW;
    size += items_bytes;
    if (!o_align_size(size, alignment, &size) || size > SIZE_MAX - programs_bytes) return O_AST_TOOL_ERROR_OVERFLOW;
    *workspace_size_ret = size + programs_bytes;
    *workspace_alignment_ret = alignment;
    return O_AST_TOOL_SUCCESS;
}

OAstToolResult o_ast_system_space_iterate(
    const OAstSystemSpaceConfig *config,
    const OAstSelection *selection,
    void *workspace,
    size_t workspace_size,
    OAstSystemSpaceVisit visit,
    void *context,
    OAstSystemSpaceReport *report_ret
) {
    OAstSystemSpaceReport report;
    OAstCounts *counts;
    OAstSystemRhsItem *rhs_items;
    uint8_t *programs;
    size_t required_size;
    size_t alignment;
    size_t offset;
    uint64_t space_count = 1u;
    uint64_t selected_count;
    uint64_t multiplier = 1u;
    uint64_t random_offset = 0u;
    uint64_t random_state;
    int all_components_exact = 1;
    int total_exact = 1;
    uint64_t system_index;
    size_t rhs_index;
    OAstToolResult result;
    if (report_ret != NULL) memset(report_ret, 0, sizeof(*report_ret));
    if (selection == NULL || visit == NULL ||
        (selection->mode != O_AST_SELECT_EXHAUSTIVE && selection->mode != O_AST_SELECT_TRUNCATE &&
         selection->mode != O_AST_SELECT_RANDOM && selection->mode != O_AST_SELECT_STRUCTURAL_RANDOM)
    ) {
        return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    }
    result = o_ast_system_space_workspace_size(config, &required_size, &alignment);
    if (result != O_AST_TOOL_SUCCESS) return result;
    if (workspace == NULL || (uintptr_t)workspace % alignment != 0u) return O_AST_TOOL_ERROR_INVALID_ARGUMENT;
    if (workspace_size < required_size) return O_AST_TOOL_ERROR_CAPACITY;
    counts = (OAstCounts *)workspace;
    offset = config->rhs_count * sizeof(*counts);
    if (!o_align_size(offset, alignment, &offset)) return O_AST_TOOL_ERROR_OVERFLOW;
    rhs_items = (OAstSystemRhsItem *)((uint8_t *)workspace + offset);
    offset += config->rhs_count * sizeof(*rhs_items);
    if (!o_align_size(offset, alignment, &offset)) return O_AST_TOOL_ERROR_OVERFLOW;
    programs = (uint8_t *)workspace + offset;
    for (rhs_index = 0u; rhs_index < config->rhs_count; ++rhs_index) {
        uint64_t expression_count;
        result = o_build_counts(&config->rhs[rhs_index].expression, counts + rhs_index);
        if (result != O_AST_TOOL_SUCCESS) return result;
        expression_count = config->rhs[rhs_index].expression.depth_mode == O_AST_DEPTH_EXACT
            ? counts[rhs_index].exact[config->rhs[rhs_index].expression.maximum_depth]
            : counts[rhs_index].up_to[config->rhs[rhs_index].expression.maximum_depth];
        if (!counts[rhs_index].count_is_exact) all_components_exact = 0;
        space_count = o_saturating_multiply(space_count, expression_count, &total_exact);
    }
    total_exact = total_exact && all_components_exact;
    memset(&report, 0, sizeof(report));
    report.space_count = space_count;
    report.space_count_is_exact = total_exact;
    report.selection_uses_space_ranks = selection->mode != O_AST_SELECT_STRUCTURAL_RANDOM;
    if (selection->mode != O_AST_SELECT_STRUCTURAL_RANDOM && !all_components_exact) return O_AST_TOOL_ERROR_OVERFLOW;
    if (selection->mode == O_AST_SELECT_EXHAUSTIVE) {
        if (!total_exact) return O_AST_TOOL_ERROR_OVERFLOW;
        selected_count = space_count;
    } else if (selection->mode == O_AST_SELECT_STRUCTURAL_RANDOM) {
        selected_count = selection->count;
    } else {
        selected_count = total_exact && space_count < selection->count ? space_count : selection->count;
    }
    random_state = selection->seed;
    if (selection->mode == O_AST_SELECT_RANDOM) {
        if (!total_exact) return O_AST_TOOL_ERROR_OVERFLOW;
        if (space_count > 1u) {
            multiplier = o_splitmix64(&random_state) % space_count;
            if (multiplier == 0u) multiplier = 1u;
            while (o_gcd(multiplier, space_count) != 1u) {
                ++multiplier;
                if (multiplier == space_count) multiplier = 1u;
            }
            random_offset = o_splitmix64(&random_state) % space_count;
        }
    }
    report.selected_count = selected_count;
    for (system_index = 0u; system_index < selected_count; ++system_index) {
        OAstSystemSpaceItem item;
        uint64_t rank = selection->mode == O_AST_SELECT_RANDOM
            ? o_add_mod(o_multiply_mod(multiplier, system_index, space_count), random_offset, space_count)
            : system_index;
        uint64_t remaining_rank = rank;
        for (rhs_index = config->rhs_count; rhs_index != 0u; --rhs_index) {
            size_t current = rhs_index - 1u;
            const OAstSpaceConfig *expression = &config->rhs[current].expression;
            OAstSpaceItem expression_item;
            uint64_t expression_count = expression->depth_mode == O_AST_DEPTH_EXACT
                ? counts[current].exact[expression->maximum_depth]
                : counts[current].up_to[expression->maximum_depth];
            if (selection->mode == O_AST_SELECT_STRUCTURAL_RANDOM) {
                result = o_emit_structural_random(
                    expression,
                    counts + current,
                    &random_state,
                    programs + current * O_AST_TOOL_MAX_PROGRAM_BYTES,
                    O_AST_TOOL_MAX_PROGRAM_BYTES,
                    &expression_item
                );
            } else {
                uint64_t expression_rank = remaining_rank % expression_count;
                remaining_rank /= expression_count;
                result = o_emit_ranked(
                    expression,
                    counts + current,
                    expression_rank,
                    programs + current * O_AST_TOOL_MAX_PROGRAM_BYTES,
                    O_AST_TOOL_MAX_PROGRAM_BYTES,
                    &expression_item
                );
            }
            if (result != O_AST_TOOL_SUCCESS) return result;
            rhs_items[current].state_index = config->rhs[current].state_index;
            rhs_items[current].expression_rank = expression_item.rank;
            rhs_items[current].depth = expression_item.depth;
            rhs_items[current].program = expression_item.program;
        }
        item.rank = rank;
        item.rank_is_space_rank = selection->mode != O_AST_SELECT_STRUCTURAL_RANDOM;
        item.rhs = rhs_items;
        item.rhs_count = config->rhs_count;
        result = visit(&item, context);
        if (result != O_AST_TOOL_SUCCESS) return O_AST_TOOL_ERROR_CALLBACK;
        ++report.emitted_count;
    }
    if (report_ret != NULL) *report_ret = report;
    return O_AST_TOOL_SUCCESS;
}

const char *o_ast_tool_result_string(OAstToolResult result) {
    switch (result) {
    case O_AST_TOOL_SUCCESS:
        return "success";
    case O_AST_TOOL_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case O_AST_TOOL_ERROR_OVERFLOW:
        return "space exceeds an exact 64-bit count";
    case O_AST_TOOL_ERROR_CAPACITY:
        return "program capacity exceeded";
    case O_AST_TOOL_ERROR_PROGRAM:
        return "invalid generated program";
    case O_AST_TOOL_ERROR_CALLBACK:
        return "visitor rejected a generated program";
    case O_AST_TOOL_ERROR_IO:
        return "file input/output failure";
    case O_AST_TOOL_ERROR_FILE_FORMAT:
        return "invalid AST file format";
    default:
        return "unknown result";
    }
}
