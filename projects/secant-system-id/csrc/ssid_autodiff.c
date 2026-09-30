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
#include "ssid_internal.h"

#include <string.h>

#define SSID_AD_HEADER_BYTES 24u
#define SSID_AD_NODE_BYTES 16u
#define SSID_AD_MAX_NODES 1024u
#define SSID_AD_NO_ARGUMENT 0xffffu

#define SSID_CONSTANT_BITS_F32 0x81u
#define SSID_RETURN_F32 0x83u
#define SSID_ADD_F32 0x85u
#define SSID_SUB_F32 0x86u
#define SSID_MUL_F32 0x87u
#define SSID_DIV_F32 0x88u
#define SSID_NEG_F32 0x89u
#define SSID_SQRT_F32 0x8au
#define SSID_RCP_F32 0x8bu
#define SSID_ABS_F32 0x8cu
#define SSID_MIN_F32 0x8du
#define SSID_MAX_F32 0x8eu
#define SSID_FMA_F32 0x8fu
#define SSID_SIN_F32 0x90u
#define SSID_COS_F32 0x91u
#define SSID_EX2_F32 0x92u
#define SSID_LG2_F32 0x93u
#define SSID_RSQRT_F32 0x94u
#define SSID_TANH_F32 0x95u
#define SSID_STATIC_COLUMN_INPUT_F32 0xb6u
#define SSID_EXP_F32 0xbau
#define SSID_LOG_F32 0xbbu

typedef struct ssid_ad_node {
    uint32_t operand;
    uint16_t arguments[3];
    uint8_t opcode;
    uint8_t arity;
} ssid_ad_node;

static void ssid_ad_store_u16(uint8_t *output, uint16_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
}

static void ssid_ad_store_u32(uint8_t *output, uint32_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
    output[2] = (uint8_t)(value >> 16);
    output[3] = (uint8_t)(value >> 24);
}

static uint32_t ssid_ad_load_u32(const uint8_t *input) {
    return (uint32_t)input[0] | ((uint32_t)input[1] << 8) |
        ((uint32_t)input[2] << 16) | ((uint32_t)input[3] << 24);
}

static int ssid_ad_arity(uint8_t opcode, uint8_t *arity) {
    switch (opcode) {
        case SSID_ADD_F32:
        case SSID_SUB_F32:
        case SSID_MUL_F32:
        case SSID_DIV_F32:
            *arity = 2u;
            return SSID_OK;
        case SSID_FMA_F32:
            *arity = 3u;
            return SSID_OK;
        case SSID_NEG_F32:
        case SSID_SQRT_F32:
        case SSID_RCP_F32:
        case SSID_SIN_F32:
        case SSID_COS_F32:
        case SSID_EX2_F32:
        case SSID_LG2_F32:
        case SSID_RSQRT_F32:
        case SSID_TANH_F32:
        case SSID_EXP_F32:
        case SSID_LOG_F32:
            *arity = 1u;
            return SSID_OK;
        case SSID_ABS_F32:
        case SSID_MIN_F32:
        case SSID_MAX_F32:
            ssid_set_error("opcode 0x%02x is nonsmooth and cannot be promoted to trajectory LM", opcode);
            return SSID_UNSUPPORTED;
        default:
            ssid_set_error("no trajectory-LM derivative rule exists for opcode 0x%02x", opcode);
            return SSID_UNSUPPORTED;
    }
}

SSID_API int ssid_ad_tape_build(
    const uint8_t *program,
    size_t program_byte_count,
    uint32_t input_count,
    uint8_t *output_tape,
    size_t output_capacity,
    size_t *required_byte_count,
    ssid_ad_tape_stats *stats)
{
    ssid_ad_node nodes[SSID_AD_MAX_NODES];
    uint16_t stack[SSID_AD_MAX_NODES];
    uint32_t node_count = 0u;
    uint32_t stack_count = 0u;
    uint32_t maximum_stack_depth = 0u;
    uint32_t node_index;
    size_t cursor = 0u;
    int returned = 0;
    size_t required;

    if (program == NULL || program_byte_count == 0u || input_count == 0u ||
        required_byte_count == NULL || stats == NULL) {
        ssid_set_error("AD tape build requires a program, positive input count, size output, and stats");
        return SSID_INVALID_ARGUMENT;
    }
    memset(stats, 0, sizeof(*stats));

    while (cursor < program_byte_count) {
        uint8_t opcode = program[cursor++];
        ssid_ad_node node;
        uint8_t arity = 0u;
        uint32_t argument;
        if (returned) {
            ssid_set_error("postorder instructions follow RETURN_F32 in AD source");
            return SSID_INVALID_ARGUMENT;
        }
        memset(&node, 0, sizeof(node));
        node.arguments[0] = SSID_AD_NO_ARGUMENT;
        node.arguments[1] = SSID_AD_NO_ARGUMENT;
        node.arguments[2] = SSID_AD_NO_ARGUMENT;
        node.opcode = opcode;

        if (opcode == SSID_RETURN_F32) {
            if (cursor != program_byte_count || stack_count != 1u) {
                ssid_set_error("RETURN_F32 must terminate an AD source with one value");
                return SSID_INVALID_ARGUMENT;
            }
            returned = 1;
            continue;
        }
        if (node_count >= SSID_AD_MAX_NODES) {
            ssid_set_error("postorder AD source exceeds %u nodes", SSID_AD_MAX_NODES);
            return SSID_OUT_OF_RANGE;
        }

        if (opcode == SSID_STATIC_COLUMN_INPUT_F32) {
            if (cursor >= program_byte_count) {
                ssid_set_error("truncated static input in AD source");
                return SSID_INVALID_ARGUMENT;
            }
            node.operand = program[cursor++];
            if (node.operand >= input_count) {
                ssid_set_error("AD source input %u is outside the tape ABI", node.operand);
                return SSID_INVALID_ARGUMENT;
            }
        } else if (opcode == SSID_CONSTANT_BITS_F32) {
            if (program_byte_count - cursor < 4u) {
                ssid_set_error("truncated FP32 constant in AD source");
                return SSID_INVALID_ARGUMENT;
            }
            node.operand = ssid_ad_load_u32(program + cursor);
            cursor += 4u;
        } else {
            int status = ssid_ad_arity(opcode, &arity);
            if (status != SSID_OK) return status;
            if (stack_count < arity) {
                ssid_set_error("postorder stack underflow while building AD tape");
                return SSID_INVALID_ARGUMENT;
            }
            node.arity = arity;
            for (argument = 0u; argument < arity; ++argument) {
                node.arguments[argument] = stack[stack_count - arity + argument];
            }
            stack_count -= arity;
        }
        nodes[node_count] = node;
        stack[stack_count++] = (uint16_t)node_count;
        node_count += 1u;
        if (stack_count > maximum_stack_depth) maximum_stack_depth = stack_count;
    }

    if (!returned || stack_count != 1u || stack[0] != node_count - 1u) {
        ssid_set_error("postorder AD source did not produce one final root");
        return SSID_INVALID_ARGUMENT;
    }
    required = SSID_AD_HEADER_BYTES + (size_t)node_count * SSID_AD_NODE_BYTES;
    *required_byte_count = required;
    stats->node_count = node_count;
    stats->maximum_stack_depth = maximum_stack_depth;
    stats->scratch_floats_per_thread = 2u * node_count;
    stats->tape_byte_count = required;
    if (output_tape == NULL) return SSID_OK;
    if (output_capacity < required) {
        ssid_set_error("AD tape output needs %zu bytes but has %zu", required, output_capacity);
        return SSID_OUT_OF_RANGE;
    }

    memcpy(output_tape, "SSIDAT01", 8u);
    ssid_ad_store_u32(output_tape + 8u, input_count);
    ssid_ad_store_u32(output_tape + 12u, node_count);
    ssid_ad_store_u32(output_tape + 16u, node_count - 1u);
    ssid_ad_store_u32(output_tape + 20u, maximum_stack_depth);
    for (node_index = 0u; node_index < node_count; ++node_index) {
        const ssid_ad_node *node = &nodes[node_index];
        uint8_t *record = output_tape + SSID_AD_HEADER_BYTES + (size_t)node_index * SSID_AD_NODE_BYTES;
        record[0] = node->opcode;
        record[1] = node->arity;
        record[2] = 0u;
        record[3] = 0u;
        ssid_ad_store_u32(record + 4u, node->operand);
        ssid_ad_store_u16(record + 8u, node->arguments[0]);
        ssid_ad_store_u16(record + 10u, node->arguments[1]);
        ssid_ad_store_u16(record + 12u, node->arguments[2]);
        record[14] = 0u;
        record[15] = 0u;
    }
    return SSID_OK;
}
